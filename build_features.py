#!/usr/bin/env python3
"""Offline feature build entry — Phase-3 will add embeddings + manifest writes here.

Phase-2 surface (what this skeleton does today):
  * parse `--candidates <path.jsonl>` → `<out>/candidates.parquet`
  * run feature pipeline on every candidate
  * write `<out>/features.parquet`

Phase-3 will additionally:
  * embed candidate docs with BGE-small → `<out>/candidate_emb.npy`
  * embed JD intents → `<out>/jd_intent_vecs.npy`
  * write `<out>/manifest.json` with all hashes

Runtime budget: ≤ 10 min on M-series (≤ 15 min on x86) for 100K candidates.
This is the *offline* binary; `rank.py` is the bounded online step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.feature_pipeline import build_features_df
from src.io_utils import write_parquet
from src.parsing import candidates_to_parquet, parse_jsonl
from src.reference_date import REFERENCE_DATE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--candidates", type=Path, required=True, help="Path to candidates.jsonl")
    p.add_argument("--out", type=Path, required=True, help="Output artifacts directory")
    args = p.parse_args(argv)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    candidates_path: Path = args.candidates

    print(f"[build_features] reference_date={REFERENCE_DATE.isoformat()}", flush=True)
    print(f"[build_features] reading {candidates_path}", flush=True)
    t0 = time.perf_counter()
    records = list(parse_jsonl(candidates_path))
    print(
        f"[build_features] parsed {len(records)} candidates in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    t0 = time.perf_counter()
    candidates_parquet = out / "candidates.parquet"
    candidates_to_parquet(records, candidates_parquet)
    print(
        f"[build_features] wrote {candidates_parquet} in {time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    t0 = time.perf_counter()
    df = build_features_df(records, REFERENCE_DATE)
    print(
        f"[build_features] computed features for {len(df)} rows in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    features_parquet = out / "features.parquet"
    write_parquet(df, features_parquet)
    print(f"[build_features] wrote {features_parquet}", flush=True)

    # Phase-2 summary — no manifest, no embeddings yet (Phase 3 work).
    summary = {
        "reference_date": REFERENCE_DATE.isoformat(),
        "candidates_parsed": len(records),
        "features_rows": int(len(df)),
        "tier_histogram": df["tier"].value_counts().to_dict(),
        "honeypot_drop_count": int(df["honeypot_drop"].sum()),
        "honeypot_audit_count": int(df["honeypot_audit"].sum()),
        "anti_pattern_archetype_counts": {
            k: int(v)
            for k, v in df["anti_pattern_ceiling"].value_counts().to_dict().items()
        },
    }
    (out / "build_features_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[build_features] summary: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
