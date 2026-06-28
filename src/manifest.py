"""Artifact manifest — Stage-3 reproducibility spine.

`build_features.py` writes a manifest describing every artifact it produced
(hashes, shapes, dtypes, column lists, build/rank commands). `rank.py` calls
`verify_manifest` before doing anything else; on any drift it raises
`ArtifactError` and aborts. There is NO silent rebuild, NO model auto-download,
NO fallback resolution — the build is either intact or the run aborts.

Phase 1 ships the data model + read/write/verify surface. Hash verification of
`candidates.jsonl` is optional in `verify_manifest` because `rank.py` does not
re-read the raw JSONL (only the Parquet derivative).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .io_utils import sha256_file

MANIFEST_FILENAME = "manifest.json"
FEATURE_SCHEMA_VERSION = "2.0"


class ArtifactError(Exception):
    """Raised by `verify_manifest` when on-disk artifacts disagree with the manifest."""


@dataclass(frozen=True)
class Manifest:
    """Build-time artifact metadata. See `problem.md §0` for the locked schema."""

    candidates_sha256: str
    feature_schema_version: str
    reference_date: str
    embedding_model: str
    embedding_model_sha256: str
    embedding_shape: list[int]
    embedding_dtype: str
    embedding_normalized: bool
    parquet_rows: int
    parquet_required_columns: list[str]
    artifact_sizes_bytes: dict[str, int]
    build_command: str
    rank_command: str
    validate_command: str
    python_version: str
    requirements_lock_sha256: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: d[k] for k in known if k in d}
        missing = known - set(d.keys())
        if missing:
            raise ArtifactError(f"manifest missing required fields: {sorted(missing)}")
        kwargs["extra"] = {k: v for k, v in d.items() if k not in known}
        return cls(**kwargs)


def write_manifest(manifest: Manifest, path: Path) -> None:
    """Serialize `manifest` to JSON at `path` (pretty-printed, key-sorted)."""
    payload = manifest.to_dict()
    extra = payload.pop("extra", {})
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> Manifest:
    """Load a manifest from JSON; raise `ArtifactError` if the file is malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ArtifactError(f"manifest not found: {path}") from e
    except json.JSONDecodeError as e:
        raise ArtifactError(f"manifest is not valid JSON: {path}") from e
    if not isinstance(data, dict):
        raise ArtifactError(f"manifest root must be an object: {path}")
    return Manifest.from_dict(data)


def verify_manifest(
    manifest: Manifest,
    artifacts_dir: Path,
    candidates_path: Path | None = None,
) -> None:
    """Verify on-disk artifacts agree with `manifest`. Raise `ArtifactError` on drift.

    Checks (Phase 1 surface):
      * `features.parquet` exists, row count and required columns match.
      * `candidate_emb.npy` exists, shape and dtype match.
      * If `candidates_path` is provided, its sha256 must match `candidates_sha256`.
    """
    _verify_features_parquet(manifest, artifacts_dir)
    _verify_candidate_embedding(manifest, artifacts_dir)
    if candidates_path is not None:
        actual = sha256_file(candidates_path)
        if actual != manifest.candidates_sha256:
            raise ArtifactError(
                "candidates.jsonl hash mismatch: "
                f"manifest={manifest.candidates_sha256!s:.16}… actual={actual!s:.16}…"
            )


def _verify_features_parquet(manifest: Manifest, artifacts_dir: Path) -> None:
    path = artifacts_dir / "features.parquet"
    if not path.exists():
        raise ArtifactError(f"missing artifact: {path}")
    pf = pq.ParquetFile(path)
    if pf.metadata.num_rows != manifest.parquet_rows:
        raise ArtifactError(
            f"features.parquet row count mismatch: "
            f"manifest={manifest.parquet_rows} actual={pf.metadata.num_rows}"
        )
    schema_names = set(pf.schema.names)
    missing = set(manifest.parquet_required_columns) - schema_names
    if missing:
        raise ArtifactError(f"features.parquet missing required columns: {sorted(missing)}")


def _verify_candidate_embedding(manifest: Manifest, artifacts_dir: Path) -> None:
    path = artifacts_dir / "candidate_emb.npy"
    if not path.exists():
        raise ArtifactError(f"missing artifact: {path}")
    arr = np.load(path, mmap_mode="r")
    expected_shape = tuple(manifest.embedding_shape)
    if arr.shape != expected_shape:
        raise ArtifactError(
            f"candidate_emb.npy shape mismatch: " f"manifest={expected_shape} actual={arr.shape}"
        )
    if str(arr.dtype) != manifest.embedding_dtype:
        raise ArtifactError(
            f"candidate_emb.npy dtype mismatch: "
            f"manifest={manifest.embedding_dtype} actual={arr.dtype}"
        )
