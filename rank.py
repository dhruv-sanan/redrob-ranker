#!/usr/bin/env python3
"""Online ranking entry point — restricted imports, ≤ 5 min budget.

Reads the precomputed artifacts produced by ``build_features.py``, scores
all 100K candidates with the capped-contribution linear blend, applies
anti-pattern score ceilings, sorts by tier + score, builds a top-10 promotion
pool, fills ranks 11–100, renders deterministic reasoning, and writes three
CSVs:

  - ``top_100_submission.csv`` (validator-format: candidate_id,rank,score,reasoning)
  - ``top_100_audit.csv``      (every feature + ledger column for human review)
  - ``top_300_debug.csv``      (extended view for ablation / weight tuning)

Allowed imports at this layer: ``json, sys, pathlib, yaml, numpy, pandas``
plus project-internal ``src.*`` modules. ``sentence-transformers``, ``torch``,
``requests``, ``httpx``, ``openai``, ``anthropic``, ``cohere`` MUST NOT be
loaded — a static AST check in ``tests/test_rank_imports.py`` enforces the
allow-list, and the runtime guard below aborts if any forbidden module is
already in ``sys.modules`` when rank.py starts.

Usage:
    python rank.py --artifacts ./artifacts/ --out ./top_100_submission.csv \\
        [--audit ./top_100_audit.csv] [--debug ./top_300_debug.csv]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.manifest import MANIFEST_FILENAME, load_manifest, verify_manifest
from src.ranking import assemble_top_100
from src.reasoning import load_skeletons_from_yaml, render_top_100_reasoning
from src.scoring import compute_scores

FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "sentence_transformers",
        "torch",
        "tensorflow",
        "transformers",
        "huggingface_hub",
        "requests",
        "httpx",
        "openai",
        "anthropic",
        "cohere",
        "google.generativeai",
    }
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = REPO_ROOT / "config" / "weights.yaml"
DEFAULT_SKELETONS = REPO_ROOT / "config" / "skeletons.yaml"
DEFAULT_TOP10_CONFIG = REPO_ROOT / "config" / "thresholds.yaml"


def assert_no_forbidden_modules() -> None:
    """Abort if any forbidden module is already imported by the time rank.py runs."""
    seen = FORBIDDEN_MODULES & set(sys.modules.keys())
    if seen:
        raise ImportError(
            f"rank.py forbidden modules present in sys.modules: {sorted(seen)} — "
            "this indicates a transitive import leak. Check rank.py's allow-list."
        )


def parse_argv(argv: list[str]) -> dict[str, Path]:
    """Tiny argv parser — argparse intentionally avoided to keep the allow-list tight."""
    opts: dict[str, Path] = {}
    expected = {"--artifacts", "--out", "--audit", "--debug", "--weights", "--skeletons"}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in expected:
            if i + 1 >= len(argv):
                raise SystemExit(f"missing value for {tok}")
            opts[tok.lstrip("-")] = Path(argv[i + 1])
            i += 2
        elif tok in {"-h", "--help"}:
            print(__doc__)
            raise SystemExit(0)
        else:
            raise SystemExit(f"unknown argument: {tok}")
    if "artifacts" not in opts:
        raise SystemExit("--artifacts required")
    if "out" not in opts:
        raise SystemExit("--out required")
    return opts


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _build_raw_lookup(candidates_path: Path, wanted_ids: list[str]) -> dict[str, dict]:
    """Load only the top-100 rows from `candidates.parquet` and return a
    `{candidate_id: raw_record_dict}` mapping. Reading the whole parquet is
    fast (~1-2s on 100K rows), but we filter immediately to keep memory
    bounded for reasoning generation."""
    df = pd.read_parquet(candidates_path)
    sub = df[df["candidate_id"].isin(wanted_ids)].copy()
    # Convert numpy arrays / pandas types to plain Python for the reasoning layer.
    records: dict[str, dict] = {}
    for _, row in sub.iterrows():
        rec = row.to_dict()
        # pyarrow-decoded list columns come back as numpy arrays; coerce.
        for key in ("career_history", "education", "skills", "certifications", "languages"):
            val = rec.get(key)
            if val is None:
                rec[key] = []
                continue
            # pyarrow list-of-struct → numpy array of dict. Pass through fine.
            rec[key] = list(val) if not isinstance(val, list) else val
        records[rec["candidate_id"]] = rec
    return records


def _features_top_n_dicts(df: pd.DataFrame, n: int) -> list[dict]:
    """Return the top-n rows as a list of plain dicts (no pandas types)."""
    return df.head(n).to_dict(orient="records")


def _audit_csv_columns() -> list[str]:
    """Columns kept in top_100_audit.csv (engineer-facing)."""
    return [
        "rank",
        "candidate_id",
        "score",
        "tier",
        "final_score",
        "final_score_uncapped",
        "base_score",
        "embedding_similarity",
        "capped_factor_emb",
        "capped_factor_skill",
        "embedding_contribution",
        "skill_contribution",
        "must_have_sum_div_6",
        "retrieval_evidence",
        "title_career_fit",
        "skill_depth_trust",
        "experience_band_fit",
        "education_signal",
        "availability_signal",
        "contactability_signal",
        "market_interest_signal",
        "external_validation_signal",
        "logistics_multiplier",
        "logistics_top_10_eligible",
        "honeypot_risk_score",
        "honeypot_audit",
        "stuffer_risk",
        "anti_pattern_ceiling",
        "anti_pattern_archetypes",
        "has_production_retrieval_evidence",
        "has_vector_or_hybrid_search_evidence",
        "has_python_backend_depth",
        "has_ranking_eval_evidence",
        "has_product_company_applied_ml_context",
        "has_shipper_signal",
        "template_id",
        "reasoning",
    ]


def main(argv: list[str] | None = None) -> int:
    assert_no_forbidden_modules()
    opts = parse_argv(list(argv if argv is not None else sys.argv[1:]))
    artifacts: Path = opts["artifacts"]
    out_path: Path = opts["out"]
    audit_path: Path = opts.get("audit", out_path.with_name("top_100_audit.csv"))
    debug_path: Path = opts.get("debug", out_path.with_name("top_300_debug.csv"))
    weights_path: Path = opts.get("weights", DEFAULT_WEIGHTS)
    skeletons_path: Path = opts.get("skeletons", DEFAULT_SKELETONS)

    # 1. Manifest + verify.
    manifest = load_manifest(artifacts / MANIFEST_FILENAME)
    verify_manifest(manifest, artifacts, candidates_path=None)

    # 2. Load precomputed artifacts.
    features = pd.read_parquet(artifacts / "features.parquet")
    candidate_emb = np.load(artifacts / "candidate_emb.npy")
    jd_intents = np.load(artifacts / "jd_intent_vecs.npy")
    weights = _load_yaml(weights_path)

    # 3. Score 100K rows.
    scored, score_telemetry = compute_scores(features, candidate_emb, jd_intents, weights)

    # 4. Assemble top 100 (honeypot drop → tier sort → top-10 gate → fill → monotonic).
    top_10_cfg = _load_yaml(DEFAULT_TOP10_CONFIG).get("top_10_promotion", None)
    top_100, rank_telemetry = assemble_top_100(scored, top_10_config=top_10_cfg)
    assert len(top_100) == 100, f"top_100 has {len(top_100)} rows (need 100)"

    # 5. Build raw lookup for the top 100 and render reasoning.
    raw_lookup = _build_raw_lookup(
        artifacts / "candidates.parquet", top_100["candidate_id"].tolist()
    )
    skeletons = load_skeletons_from_yaml(skeletons_path.read_text(encoding="utf-8"))
    rendered = render_top_100_reasoning(_features_top_n_dicts(top_100, 100), raw_lookup, skeletons)
    top_100["reasoning"] = [r["reasoning"] for r in rendered]
    top_100["template_id"] = [r["template_id"] for r in rendered]

    # 6. Submission CSV — exact 4-col schema validate_submission.py expects.
    submission = top_100[["candidate_id", "rank", "score", "reasoning"]].copy()
    submission["rank"] = submission["rank"].astype(int)
    submission["score"] = submission["score"].astype(float)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(out_path, index=False)

    # 7. Audit CSV — every feature column + ledger.
    audit_cols = [c for c in _audit_csv_columns() if c in top_100.columns]
    audit = top_100[audit_cols].copy()
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)

    # 8. Debug CSV — top 300 with the full feature view.
    debug_cols = [
        c for c in scored.columns if c not in {"honeypot_risks", "anti_pattern_archetypes"}
    ]
    top_300 = (
        scored.sort_values(
            ["tier_priority", "final_score", "candidate_id"], ascending=[True, False, True]
        )
        .head(300)
        .copy()
    )
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    top_300[debug_cols].to_csv(debug_path, index=False)

    summary = {
        "artifacts": str(artifacts),
        "out": str(out_path),
        "audit": str(audit_path),
        "debug": str(debug_path),
        "rows_scored": int(len(scored)),
        "rows_after_honeypot_drop": rank_telemetry["survivors_count"],
        "top_10_gate_pool_size": rank_telemetry["gate_pool_size"],
        "top_10_relaxation_used": rank_telemetry["relaxation_used"],
        "ceilings": score_telemetry,
        "submission": str(out_path),
    }
    print(json.dumps(summary, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
