"""Phase-1 tests for `src/manifest.py` (write / load / verify + ArtifactError contract)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.io_utils import sha256_file, write_parquet
from src.manifest import (
    FEATURE_SCHEMA_VERSION,
    ArtifactError,
    Manifest,
    load_manifest,
    verify_manifest,
    write_manifest,
)


def _make_manifest(
    candidates_sha256: str = "a" * 64,
    parquet_rows: int = 5,
    parquet_required_columns: list[str] | None = None,
    embedding_shape: list[int] | None = None,
) -> Manifest:
    return Manifest(
        candidates_sha256=candidates_sha256,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        reference_date="2026-06-01",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_model_sha256="b" * 64,
        embedding_shape=embedding_shape or [5, 4],
        embedding_dtype="float16",
        embedding_normalized=True,
        parquet_rows=parquet_rows,
        parquet_required_columns=parquet_required_columns or ["candidate_id", "score"],
        artifact_sizes_bytes={"features.parquet": 100, "candidate_emb.npy": 200},
        build_command="python build_features.py --candidates ./candidates.jsonl --out ./artifacts/",
        rank_command="python rank.py --artifacts ./artifacts/ --out ./submission.csv",
        validate_command="python validate_submission.py submission.csv",
        python_version="3.11.15",
        requirements_lock_sha256="c" * 64,
    )


def _write_features_parquet(
    artifacts_dir: Path,
    rows: int = 5,
    columns: list[str] | None = None,
) -> Path:
    columns = columns or ["candidate_id", "score"]
    data = {col: list(range(rows)) for col in columns}
    if "candidate_id" in data:
        data["candidate_id"] = [f"CAND_{i:07d}" for i in range(rows)]
    df = pd.DataFrame(data)
    out = artifacts_dir / "features.parquet"
    write_parquet(df, out)
    return out


def _write_candidate_emb(
    artifacts_dir: Path,
    shape: tuple[int, ...] = (5, 4),
    dtype: str = "float16",
) -> Path:
    arr = np.zeros(shape, dtype=np.dtype(dtype))
    out = artifacts_dir / "candidate_emb.npy"
    np.save(out, arr)
    return out


# ---------------------------------------------------------------------------
# write_manifest / load_manifest roundtrip
# ---------------------------------------------------------------------------


def test_write_then_load_roundtrip(tmp_path: Path) -> None:
    m = _make_manifest()
    p = tmp_path / "manifest.json"
    write_manifest(m, p)
    assert p.exists()
    loaded = load_manifest(p)
    assert loaded == m


def test_loaded_manifest_json_is_sorted_and_pretty(tmp_path: Path) -> None:
    m = _make_manifest()
    p = tmp_path / "manifest.json"
    write_manifest(m, p)
    txt = p.read_text(encoding="utf-8")
    # pretty print check — multiple lines and 2-space indent for nested keys
    assert "\n" in txt and "  " in txt
    # key ordering is alphabetic
    parsed_keys = list(json.loads(txt).keys())
    assert parsed_keys == sorted(parsed_keys)


def test_load_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError):
        load_manifest(tmp_path / "absent.json")


def test_load_manifest_bad_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("not json", encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_manifest(p)


def test_manifest_from_dict_rejects_missing_required(tmp_path: Path) -> None:
    p = tmp_path / "partial.json"
    p.write_text(json.dumps({"reference_date": "2026-06-01"}), encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_manifest(p)


# ---------------------------------------------------------------------------
# verify_manifest happy path
# ---------------------------------------------------------------------------


def test_verify_manifest_ok_on_matching_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_features_parquet(artifacts, rows=5, columns=["candidate_id", "score"])
    _write_candidate_emb(artifacts, shape=(5, 4), dtype="float16")
    m = _make_manifest(parquet_rows=5, embedding_shape=[5, 4])
    # No exception → pass.
    verify_manifest(m, artifacts)


# ---------------------------------------------------------------------------
# verify_manifest negative paths — every drift raises ArtifactError
# ---------------------------------------------------------------------------


def test_verify_manifest_raises_on_missing_features_parquet(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_candidate_emb(artifacts, shape=(5, 4), dtype="float16")
    m = _make_manifest()
    with pytest.raises(ArtifactError, match="features.parquet"):
        verify_manifest(m, artifacts)


def test_verify_manifest_raises_on_missing_required_column(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_features_parquet(artifacts, rows=5, columns=["candidate_id"])  # missing "score"
    _write_candidate_emb(artifacts, shape=(5, 4), dtype="float16")
    m = _make_manifest(parquet_required_columns=["candidate_id", "score"])
    with pytest.raises(ArtifactError, match="missing required columns"):
        verify_manifest(m, artifacts)


def test_verify_manifest_raises_on_parquet_row_count_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_features_parquet(artifacts, rows=5)
    _write_candidate_emb(artifacts, shape=(5, 4), dtype="float16")
    m = _make_manifest(parquet_rows=10)  # mismatch
    with pytest.raises(ArtifactError, match="row count mismatch"):
        verify_manifest(m, artifacts)


def test_verify_manifest_raises_on_missing_emb_npy(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_features_parquet(artifacts, rows=5)
    m = _make_manifest(parquet_rows=5)
    with pytest.raises(ArtifactError, match="candidate_emb.npy"):
        verify_manifest(m, artifacts)


def test_verify_manifest_raises_on_emb_shape_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_features_parquet(artifacts, rows=5)
    _write_candidate_emb(artifacts, shape=(5, 8), dtype="float16")  # expected (5,4)
    m = _make_manifest(parquet_rows=5, embedding_shape=[5, 4])
    with pytest.raises(ArtifactError, match="shape mismatch"):
        verify_manifest(m, artifacts)


def test_verify_manifest_raises_on_emb_dtype_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_features_parquet(artifacts, rows=5)
    _write_candidate_emb(artifacts, shape=(5, 4), dtype="float32")  # expected float16
    m = _make_manifest(parquet_rows=5, embedding_shape=[5, 4])
    with pytest.raises(ArtifactError, match="dtype mismatch"):
        verify_manifest(m, artifacts)


def test_verify_manifest_raises_on_candidates_hash_mismatch(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_features_parquet(artifacts, rows=5)
    _write_candidate_emb(artifacts, shape=(5, 4), dtype="float16")
    fake_jsonl = tmp_path / "candidates.jsonl"
    fake_jsonl.write_text('{"candidate_id": "CAND_0000001"}\n', encoding="utf-8")

    m = _make_manifest(parquet_rows=5, embedding_shape=[5, 4])  # candidates_sha256 = "a"*64
    with pytest.raises(ArtifactError, match="candidates.jsonl hash"):
        verify_manifest(m, artifacts, candidates_path=fake_jsonl)


def test_verify_manifest_passes_when_candidates_hash_matches(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _write_features_parquet(artifacts, rows=5)
    _write_candidate_emb(artifacts, shape=(5, 4), dtype="float16")
    fake_jsonl = tmp_path / "candidates.jsonl"
    fake_jsonl.write_text('{"candidate_id": "CAND_0000001"}\n', encoding="utf-8")
    real_hash = sha256_file(fake_jsonl)
    m = replace(_make_manifest(parquet_rows=5, embedding_shape=[5, 4]), candidates_sha256=real_hash)
    verify_manifest(m, artifacts, candidates_path=fake_jsonl)


# ---------------------------------------------------------------------------
# sha256_file
# ---------------------------------------------------------------------------


def test_sha256_file_is_deterministic_across_two_reads(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(b"hackathon" * 4096)
    assert sha256_file(p) == sha256_file(p)


def test_sha256_file_changes_when_content_changes(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(b"a")
    h1 = sha256_file(p)
    p.write_bytes(b"b")
    h2 = sha256_file(p)
    assert h1 != h2


def test_sha256_file_matches_known_empty_hash(tmp_path: Path) -> None:
    p = tmp_path / "empty.bin"
    p.touch()
    assert sha256_file(p) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ---------------------------------------------------------------------------
# Manifest dataclass shape
# ---------------------------------------------------------------------------


def test_manifest_extra_keys_round_trip(tmp_path: Path) -> None:
    m = _make_manifest()
    m_dict = m.to_dict()
    m_dict.pop("extra", None)
    m_dict["future_key"] = "future_value"  # forward-compatible extra
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m_dict), encoding="utf-8")
    loaded = load_manifest(p)
    assert loaded.extra == {"future_key": "future_value"}


def test_parquet_features_uses_pyarrow_engine(tmp_path: Path) -> None:
    """Sanity: verify_manifest works with a Parquet written via our io_utils path."""
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    df = pd.DataFrame({"candidate_id": ["CAND_0000001", "CAND_0000002"], "score": [0.9, 0.8]})
    write_parquet(df, artifacts / "features.parquet")
    pq.read_schema(artifacts / "features.parquet")  # smoke — no error
    table = pa.Table.from_pandas(df)
    assert table.num_rows == 2
