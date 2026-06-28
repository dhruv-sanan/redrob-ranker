"""Phase-1 tests for `src/parsing.py` (JSONL → Parquet single source of truth)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.io_utils import read_parquet
from src.parsing import candidates_to_parquet, parse_jsonl


def _write_jsonl(records: list[dict], path: Path) -> Path:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def test_fixture_size_matches_spec(synthetic_50: list[dict]) -> None:
    assert len(synthetic_50) == 50
    ids = {c["candidate_id"] for c in synthetic_50}
    assert len(ids) == 50, "candidate IDs must be unique"
    for cid in ids:
        assert cid.startswith("CAND_") and len(cid) == 12, cid


def test_parse_jsonl_roundtrip(tmp_path: Path, synthetic_50: list[dict]) -> None:
    src = _write_jsonl(synthetic_50, tmp_path / "src.jsonl")
    parsed = list(parse_jsonl(src))
    assert len(parsed) == 50
    assert parsed[0]["candidate_id"] == synthetic_50[0]["candidate_id"]
    assert parsed[-1]["candidate_id"] == synthetic_50[-1]["candidate_id"]


def test_parse_jsonl_skips_blank_lines(tmp_path: Path, synthetic_50: list[dict]) -> None:
    p = tmp_path / "with_blanks.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(synthetic_50[0]) + "\n")
        f.write("\n")
        f.write("   \n")
        f.write(json.dumps(synthetic_50[1]) + "\n")
    parsed = list(parse_jsonl(p))
    assert len(parsed) == 2
    assert parsed[0]["candidate_id"] == synthetic_50[0]["candidate_id"]
    assert parsed[1]["candidate_id"] == synthetic_50[1]["candidate_id"]


def test_parse_jsonl_does_not_mutate_source(tmp_path: Path, synthetic_50: list[dict]) -> None:
    pristine = deepcopy(synthetic_50)
    src = _write_jsonl(synthetic_50, tmp_path / "src.jsonl")
    _ = list(parse_jsonl(src))
    assert synthetic_50 == pristine


def test_candidates_to_parquet_preserves_nested_structure(
    tmp_path: Path, synthetic_50: list[dict]
) -> None:
    out = tmp_path / "candidates.parquet"
    rows = candidates_to_parquet(synthetic_50, out)
    assert rows == 50
    assert out.exists() and out.stat().st_size > 0

    schema = pq.read_schema(out)
    names = set(schema.names)
    # Every top-level key required by `candidate_schema.json` must survive.
    required = {
        "candidate_id",
        "profile",
        "career_history",
        "education",
        "skills",
        "redrob_signals",
        "certifications",
        "languages",
    }
    assert required <= names, f"missing top-level columns: {required - names}"


def test_candidates_parquet_round_trip_via_pandas(tmp_path: Path, synthetic_50: list[dict]) -> None:
    out = tmp_path / "candidates.parquet"
    candidates_to_parquet(synthetic_50, out)
    df = read_parquet(out)
    assert len(df) == 50
    assert "candidate_id" in df.columns
    assert df["candidate_id"].iloc[0] == synthetic_50[0]["candidate_id"]


def test_dates_remain_strings_in_parquet(tmp_path: Path, synthetic_50: list[dict]) -> None:
    """Phase-1 parsing keeps date strings raw. Date math lives in Phase-2 feature builders."""
    out = tmp_path / "candidates.parquet"
    candidates_to_parquet(synthetic_50, out)
    df = read_parquet(out)
    first_role = df["career_history"].iloc[0][0]
    # start_date must round-trip as the same string we wrote
    assert isinstance(first_role["start_date"], str)
    assert first_role["start_date"] == synthetic_50[0]["career_history"][0]["start_date"]


def test_parsing_module_does_not_use_today() -> None:
    """Static check: `src/parsing.py` source must not call `date.today()` or `datetime.now()`."""
    src = Path(__file__).resolve().parent.parent / "src" / "parsing.py"
    text = src.read_text(encoding="utf-8")
    assert "date.today()" not in text
    assert "datetime.now()" not in text


def test_parse_jsonl_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(parse_jsonl(tmp_path / "does_not_exist.jsonl"))


def test_parse_jsonl_raises_on_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "broken.jsonl"
    p.write_text("not valid json\n", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(parse_jsonl(p))


def test_reference_date_constant_is_frozen() -> None:
    """Sanity: the constant has exactly the locked value."""
    from datetime import date

    from src.reference_date import REFERENCE_DATE

    assert date(2026, 6, 1) == REFERENCE_DATE
