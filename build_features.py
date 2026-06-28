#!/usr/bin/env python3
"""Offline feature build — Phase-3 final form.

Steps:
  1. Parse candidates JSONL → ``<out>/candidates.parquet``
  2. Build candidate docs → encode via BGE-small → ``<out>/candidate_emb.npy``
  3. Encode JD intents → ``<out>/jd_intent_vecs.npy``
  4. Run feature pipeline → ``<out>/features.parquet``
  5. Write ``<out>/manifest.json`` (hashes, shapes, dtypes, commands)

Runtime budget: ≤ 10 min on M-series, ≤ 15 min on x86 for 100K candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from src.config_loader import load_config
from src.embeddings import (
    EMBEDDING_DIM,
    encode_candidates,
    encode_jd_intents,
    hash_model_dir,
    load_model,
)
from src.feature_pipeline import FEATURE_COLUMNS, build_features_df
from src.io_utils import sha256_file, write_parquet
from src.manifest import FEATURE_SCHEMA_VERSION, MANIFEST_FILENAME, Manifest, write_manifest
from src.parsing import candidates_to_parquet, parse_jsonl
from src.reference_date import REFERENCE_DATE

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def _ensure_model_dir(model_dir: Path, model_name: str) -> None:
    """Vendor the model if `model_dir` is empty. Mirrors tools/vendor_model.py."""
    from tools.vendor_model import looks_vendored, vendor

    if looks_vendored(model_dir):
        return
    rc = vendor(model_name, model_dir)
    if rc != 0:
        raise RuntimeError(f"model vendoring failed for {model_name}")


def _build_manifest(
    *,
    candidates_path: Path,
    out: Path,
    model_dir: Path,
    model_name: str,
    rows: int,
    requirements_path: Path,
) -> Manifest:
    sizes = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    return Manifest(
        candidates_sha256=sha256_file(candidates_path),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        reference_date=REFERENCE_DATE.isoformat(),
        embedding_model=model_name,
        embedding_model_sha256=hash_model_dir(model_dir),
        embedding_shape=[rows, EMBEDDING_DIM],
        embedding_dtype="float16",
        embedding_normalized=True,
        parquet_rows=rows,
        parquet_required_columns=list(FEATURE_COLUMNS),
        artifact_sizes_bytes=sizes,
        build_command=f"python build_features.py --candidates {candidates_path} --out {out}",
        rank_command=f"python rank.py --artifacts {out} --out ./top_100_submission.csv",
        validate_command="python validate_submission.py top_100_submission.csv",
        python_version=platform.python_version(),
        requirements_lock_sha256=sha256_file(requirements_path),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--candidates", type=Path, required=True, help="Path to candidates.jsonl")
    p.add_argument("--out", type=Path, required=True, help="Output artifacts directory")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Vendored model directory (default: <out>/model)",
    )
    p.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model id (only used if model-dir is empty and we must vendor)",
    )
    p.add_argument(
        "--batch-size", type=int, default=512, help="Encoder batch size (default: 512)"
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable encoder progress bar (useful in CI/redirected output)",
    )
    args = p.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    model_dir: Path = args.model_dir or (out / "model")
    candidates_path: Path = args.candidates
    requirements_path = Path(__file__).resolve().parent / "requirements.txt"

    print(f"[build_features] reference_date={REFERENCE_DATE.isoformat()}", flush=True)
    print(f"[build_features] candidates={candidates_path}", flush=True)
    print(f"[build_features] out={out}", flush=True)
    print(f"[build_features] model_dir={model_dir}", flush=True)

    # 1. Vendor model if missing.
    t0 = time.perf_counter()
    _ensure_model_dir(model_dir, args.model_name)
    print(f"[build_features] model ready in {time.perf_counter() - t0:.1f}s", flush=True)

    # 2. Parse JSONL.
    t0 = time.perf_counter()
    records = list(parse_jsonl(candidates_path))
    n = len(records)
    print(f"[build_features] parsed {n} candidates in {time.perf_counter() - t0:.1f}s", flush=True)

    # 3. Write candidates.parquet (single source of truth).
    t0 = time.perf_counter()
    candidates_to_parquet(records, out / "candidates.parquet")
    print(
        f"[build_features] wrote candidates.parquet in {time.perf_counter() - t0:.1f}s", flush=True
    )

    # 4. Load model + encode candidates.
    t0 = time.perf_counter()
    model = load_model(model_dir)
    print(f"[build_features] loaded model in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    cand_emb = encode_candidates(
        records, model, batch_size=args.batch_size, show_progress=not args.no_progress
    )
    assert cand_emb.shape == (n, EMBEDDING_DIM), f"shape {cand_emb.shape}"
    np.save(out / "candidate_emb.npy", cand_emb)
    print(
        f"[build_features] encoded {n} candidates in {time.perf_counter() - t0:.1f}s — "
        f"emb shape={cand_emb.shape} dtype={cand_emb.dtype}",
        flush=True,
    )

    # 5. Encode JD intents.
    t0 = time.perf_counter()
    jd_intents = load_config("jd_intents")["intents"]
    jd_emb = encode_jd_intents(list(jd_intents), model)
    assert jd_emb.shape == (len(jd_intents), EMBEDDING_DIM)
    np.save(out / "jd_intent_vecs.npy", jd_emb)
    print(
        f"[build_features] encoded {len(jd_intents)} JD intents in "
        f"{time.perf_counter() - t0:.1f}s — emb shape={jd_emb.shape}",
        flush=True,
    )

    # 6. Run feature pipeline.
    t0 = time.perf_counter()
    df = build_features_df(records, REFERENCE_DATE)
    write_parquet(df, out / "features.parquet")
    print(
        f"[build_features] computed + wrote features.parquet ({len(df)} rows) in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    # 7. Manifest.
    t0 = time.perf_counter()
    manifest = _build_manifest(
        candidates_path=candidates_path,
        out=out,
        model_dir=model_dir,
        model_name=args.model_name,
        rows=n,
        requirements_path=requirements_path,
    )
    write_manifest(manifest, out / MANIFEST_FILENAME)
    print(f"[build_features] wrote manifest in {time.perf_counter() - t0:.1f}s", flush=True)

    # 8. Summary (in addition to manifest — for quick human inspection).
    summary = {
        "reference_date": REFERENCE_DATE.isoformat(),
        "candidates_parsed": n,
        "features_rows": int(len(df)),
        "candidate_emb_shape": list(cand_emb.shape),
        "jd_intent_vecs_shape": list(jd_emb.shape),
        "tier_histogram": df["tier"].value_counts().to_dict(),
        "honeypot_drop_count": int(df["honeypot_drop"].sum()),
        "honeypot_audit_count": int(df["honeypot_audit"].sum()),
        "anti_pattern_archetype_counts": {
            k: int(v) for k, v in df["anti_pattern_ceiling"].value_counts().to_dict().items()
        },
        "artifacts_total_bytes": sum(
            p.stat().st_size for p in out.rglob("*") if p.is_file()
        ),
    }
    (out / "build_features_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"[build_features] summary: {json.dumps(summary, default=str)}", flush=True)
    return 0


if __name__ == "__main__":
    # Ensure CWD-based imports work when invoked from repo root.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    # Reduce noise from transformers logging.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.exit(main())
