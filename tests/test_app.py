"""CP-5c tests — sandbox JSONL parsing, input validation, end-to-end rank_sample.

The end-to-end test is guarded by the presence of the vendored model dir
(matches the pattern in ``tests/test_embeddings.py`` for the BGE integration
case). Cold-clone CI without the model still passes — the guard skips.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from app import (
    MAX_SAMPLE_SIZE,
    _render_csv,
    parse_jsonl_text,
    rank_from_inputs,
)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
CANDIDATES_PARQUET = ARTIFACTS_DIR / "candidates.parquet"


def _model_available() -> bool:
    return MODEL_DIR.exists() and any(MODEL_DIR.iterdir())


def _real_records(n: int) -> list[dict]:
    """Pull `n` real candidates from the vendored parquet for integration tests."""
    if not CANDIDATES_PARQUET.exists():
        pytest.skip("artifacts/candidates.parquet missing")

    def _convert(obj):
        import numpy as np

        if isinstance(obj, np.ndarray):
            return [_convert(x) for x in obj.tolist()]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        return obj

    table = pq.read_table(CANDIDATES_PARQUET).slice(0, n).to_pylist()
    return [_convert(r) for r in table]


def test_parse_jsonl_text_empty_returns_empty():
    assert parse_jsonl_text("") == []
    assert parse_jsonl_text("\n\n  \n") == []


def test_parse_jsonl_text_strips_blank_lines():
    text = '\n{"a": 1}\n\n{"b": 2}\n'
    records = parse_jsonl_text(text)
    assert records == [{"a": 1}, {"b": 2}]


def test_parse_jsonl_text_invalid_raises_line_number():
    text = '{"a": 1}\n{"oops":}\n'
    with pytest.raises(ValueError, match="line 2"):
        parse_jsonl_text(text)


def test_render_csv_keeps_only_validator_columns():
    df = pd.DataFrame(
        {
            "candidate_id": ["A", "B"],
            "rank": [1, 2],
            "score": [0.9, 0.8],
            "reasoning": ["r1", "r2"],
            "tier": ["A", "B"],
            "final_score": [0.91, 0.81],
        }
    )
    csv = _render_csv(df)
    header = csv.splitlines()[0]
    assert header == "candidate_id,rank,score,reasoning"
    assert "tier" not in header


def test_rank_from_inputs_returns_message_when_no_input():
    df, csv_path, status = rank_from_inputs(None, "", 10)
    assert df.empty
    assert csv_path == ""
    assert "JSONL upload" in status


def test_rank_from_inputs_rejects_oversize_sample():
    big_text = "\n".join(json.dumps({"candidate_id": f"X{i}"}) for i in range(MAX_SAMPLE_SIZE + 5))
    df, csv_path, status = rank_from_inputs(None, big_text, 10)
    assert df.empty
    assert csv_path == ""
    assert "exceeds free-tier cap" in status


def test_rank_from_inputs_reports_json_parse_failure():
    df, csv_path, status = rank_from_inputs(None, '{"a": 1}\n{"oops":}\n', 10)
    assert df.empty
    assert csv_path == ""
    assert "Input error" in status


@pytest.mark.skipif(not _model_available(), reason="vendored BGE model not present")
def test_rank_sample_end_to_end_returns_ranked_dataframe():
    from app import rank_sample

    records = _real_records(15)
    df = rank_sample(records, target_size=5)
    assert len(df) <= 5
    assert list(df.columns) == [
        "candidate_id",
        "rank",
        "score",
        "reasoning",
        "tier",
        "final_score",
    ]
    assert list(df["rank"]) == sorted(df["rank"]), "rank ascending"
    scores = df["score"].tolist()
    assert all(
        scores[i] >= scores[i + 1] for i in range(len(scores) - 1)
    ), "score must be monotonic non-increasing"
    assert all(isinstance(r, str) and r for r in df["reasoning"]), "reasoning populated"


@pytest.mark.skipif(not _model_available(), reason="vendored BGE model not present")
def test_rank_from_inputs_via_paste_path_writes_csv(tmp_path: Path):
    records = _real_records(10)
    text = "\n".join(json.dumps(r, default=str) for r in records)
    df, csv_path, status = rank_from_inputs(None, text, 5)
    assert not df.empty
    assert csv_path != ""
    csv_body = Path(csv_path).read_text(encoding="utf-8")
    assert csv_body.splitlines()[0] == "candidate_id,rank,score,reasoning"
    assert "Ranked" in status


@pytest.mark.skipif(not _model_available(), reason="vendored BGE model not present")
def test_rank_sample_drops_oversize_input():
    from app import rank_sample

    with pytest.raises(ValueError, match="exceeds MAX_SAMPLE_SIZE"):
        rank_sample([{"candidate_id": f"X{i}"} for i in range(MAX_SAMPLE_SIZE + 1)])


@pytest.mark.skipif(not _model_available(), reason="vendored BGE model not present")
def test_rank_sample_empty_records_raises():
    from app import rank_sample

    with pytest.raises(ValueError, match="no candidates"):
        rank_sample([])
