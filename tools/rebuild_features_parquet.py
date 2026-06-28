#!/usr/bin/env python3
"""Re-derive ``features.parquet`` + ``manifest.json`` from an existing
``candidates.jsonl`` and the already-encoded ``candidate_emb.npy``.

Why this exists: the full ``build_features.py`` pipeline re-runs BGE-small,
which costs ~60 min wall-clock on M-series CPU. When the feature schema
changes but the embedding contract (model + doc-builder) does NOT, we can
keep the existing ``candidate_emb.npy`` and only regenerate the feature
matrix + manifest. The hash chain is preserved by re-reading the existing
candidates hash and model hash from the on-disk manifest.

Usage:
    python tools/rebuild_features_parquet.py \\
        --candidates /path/to/candidates.jsonl \\
        --out ./artifacts/
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

# Allow `python tools/rebuild_features_parquet.py` from repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from src.feature_pipeline import FEATURE_COLUMNS, build_features_df  # noqa: E402
from src.io_utils import sha256_file, write_parquet  # noqa: E402
from src.manifest import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    Manifest,
    load_manifest,
    verify_manifest,
    write_manifest,
)
from src.parsing import parse_jsonl  # noqa: E402
from src.reference_date import REFERENCE_DATE  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    out: Path = args.out
    candidates_path: Path = args.candidates
    repo_root = Path(__file__).resolve().parent.parent
    requirements_path = repo_root / "requirements.txt"
    model_dir = out / "model"
    emb_path = out / "candidate_emb.npy"
    manifest_path = out / MANIFEST_FILENAME

    if not emb_path.exists():
        raise SystemExit(f"missing {emb_path}; run build_features.py first")
    existing = load_manifest(manifest_path)

    t0 = time.perf_counter()
    records = list(parse_jsonl(candidates_path))
    n = len(records)
    print(f"[rebuild] parsed {n} candidates in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    df = build_features_df(records, REFERENCE_DATE)
    write_parquet(df, out / "features.parquet")
    print(
        f"[rebuild] computed + wrote features.parquet ({len(df)} rows) in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    emb = np.load(emb_path, mmap_mode="r")
    sizes = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}

    manifest = Manifest(
        candidates_sha256=existing.candidates_sha256,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        reference_date=REFERENCE_DATE.isoformat(),
        embedding_model=existing.embedding_model,
        embedding_model_sha256=existing.embedding_model_sha256,
        embedding_shape=[int(emb.shape[0]), int(emb.shape[1])],
        embedding_dtype=str(emb.dtype),
        embedding_normalized=True,
        parquet_rows=int(len(df)),
        parquet_required_columns=list(FEATURE_COLUMNS),
        artifact_sizes_bytes=sizes,
        build_command=existing.build_command,
        rank_command=existing.rank_command,
        validate_command=existing.validate_command,
        python_version=platform.python_version(),
        requirements_lock_sha256=sha256_file(requirements_path),
    )
    write_manifest(manifest, manifest_path)
    print(f"[rebuild] wrote manifest at {manifest_path}", flush=True)

    # Sanity: hashes still match the on-disk artifacts.
    verify_manifest(manifest, out, candidates_path=candidates_path)
    print(
        f"[rebuild] OK — emb_shape={list(emb.shape)} parquet_rows={len(df)} "
        f"required_columns={len(FEATURE_COLUMNS)} model_dir_exists={model_dir.exists()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
