#!/usr/bin/env python3
"""CP-5b ablation driver — A0 baseline + A1..A5 variants.

Loads the precomputed artifacts once and re-runs the scoring + ranking
pipeline under five distinct ablations to measure top-100 overlap deltas
vs the baseline. The point is to attribute the current ranking to its
components and surface the strongest tuning levers for CP-S2.

Variants:

  A0 baseline           current weights + ceilings + multipliers + gate
  A1 no_embedding       zero out the embedding_contribution blend weight
  A2 no_skill_blend     zero out the skill_contribution blend weight
                        (proxy for "no skills in embedding doc" — full
                        re-encoding without skills would cost ~50 min)
  A3 no_behavioral_mult drop availability × logistics × market_tiebreak
  A4 no_anti_pattern    skip the rank_50 / rank_100 score ceilings
  A5 no_top_10_gate     skip promotion gate; take top 10 by global order

Each variant returns its top-100 candidate_id list; the driver writes
ablation_report.md with overlap %, top-10 stability, and inter-variant
Jaccard.

Usage:
    python tools/ablations.py \\
        --artifacts ./artifacts/ \\
        --report ablation_report.md \\
        [--out-dir ablations/]
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.ranking import (  # noqa: E402
    assemble_top_100,
    drop_honeypots,
    enforce_monotonic_scores,
    tier_sort,
)
from src.scoring import compute_scores  # noqa: E402


@dataclass(frozen=True)
class Variant:
    code: str
    label: str
    description: str


VARIANTS: tuple[Variant, ...] = (
    Variant("A0", "baseline", "current weights + ceilings + multipliers + gate"),
    Variant("A1", "no_embedding", "blend.embedding_contribution = 0.0"),
    Variant("A2", "no_skill_blend", "blend.skill_contribution = 0.0"),
    Variant("A3", "no_behavioral_mult", "skip availability × logistics × market_tiebreak"),
    Variant("A4", "no_anti_pattern", "skip rank_50 / rank_100 score ceilings"),
    Variant("A5", "no_top_10_gate", "no promotion gate; first 10 by global order"),
)


def _zero_weight(weights: dict, key: str) -> dict:
    out = copy.deepcopy(weights)
    out["blend"][key] = 0.0
    return out


def _strip_behavioral(scored: pd.DataFrame) -> pd.DataFrame:
    """Replace final_score with base_score (no avail × logistics × market)."""
    df = scored.copy()
    df["final_score_uncapped"] = df["base_score"].astype(np.float32)
    df["final_score"] = df["base_score"].astype(np.float32)
    return df


def _strip_ceiling(scored: pd.DataFrame) -> pd.DataFrame:
    """Replace final_score with final_score_uncapped (no rank_50/100 clip)."""
    df = scored.copy()
    df["final_score"] = df["final_score_uncapped"].astype(np.float32)
    return df


def _no_gate_top_100(scored: pd.DataFrame, target: int = 100) -> pd.DataFrame:
    """Bypass build_top_10_pool — take top `target` from global ordered list."""
    survivors = drop_honeypots(scored)
    sorted_df = tier_sort(survivors)
    top = sorted_df.head(target).copy().reset_index(drop=True)
    top["rank"] = np.arange(1, len(top) + 1, dtype=np.int64)
    top["score"] = enforce_monotonic_scores(top["final_score"].to_numpy())
    return top


def run_variant(
    variant_code: str,
    features: pd.DataFrame,
    candidate_emb: np.ndarray,
    jd_intent_vecs: np.ndarray,
    weights: dict,
    top_10_cfg: dict | None,
) -> pd.DataFrame:
    """Return the top-100 DataFrame for the given variant code."""
    if variant_code == "A1":
        w = _zero_weight(weights, "embedding_contribution")
        scored, _ = compute_scores(features, candidate_emb, jd_intent_vecs, w)
        top_100, _ = assemble_top_100(scored, top_10_config=top_10_cfg)
        return top_100
    if variant_code == "A2":
        w = _zero_weight(weights, "skill_contribution")
        scored, _ = compute_scores(features, candidate_emb, jd_intent_vecs, w)
        top_100, _ = assemble_top_100(scored, top_10_config=top_10_cfg)
        return top_100
    if variant_code == "A3":
        scored, _ = compute_scores(features, candidate_emb, jd_intent_vecs, weights)
        scored = _strip_behavioral(scored)
        top_100, _ = assemble_top_100(scored, top_10_config=top_10_cfg)
        return top_100
    if variant_code == "A4":
        scored, _ = compute_scores(features, candidate_emb, jd_intent_vecs, weights)
        scored = _strip_ceiling(scored)
        top_100, _ = assemble_top_100(scored, top_10_config=top_10_cfg)
        return top_100
    if variant_code == "A5":
        scored, _ = compute_scores(features, candidate_emb, jd_intent_vecs, weights)
        return _no_gate_top_100(scored)
    if variant_code == "A0":
        scored, _ = compute_scores(features, candidate_emb, jd_intent_vecs, weights)
        top_100, _ = assemble_top_100(scored, top_10_config=top_10_cfg)
        return top_100
    raise KeyError(f"unknown variant: {variant_code}")


def overlap_metrics(baseline_ids: list[str], variant_ids: list[str]) -> dict[str, float]:
    """Compute |∩| and Jaccard between baseline and variant top-100 sets."""
    a = set(baseline_ids)
    b = set(variant_ids)
    inter = len(a & b)
    union = len(a | b)
    jaccard = float(inter) / float(union) if union else 0.0
    return {
        "overlap_count": float(inter),
        "overlap_pct": float(inter) / float(max(1, len(a))) * 100.0,
        "jaccard": jaccard,
    }


def top_k_stability(baseline_ids: list[str], variant_ids: list[str], k: int) -> int:
    """Return |baseline_top_k ∩ variant_top_k| (rank-aware top-K)."""
    return len(set(baseline_ids[:k]) & set(variant_ids[:k]))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _format_report(
    results: dict[str, dict],
    timings: dict[str, float],
    artifacts_dir: Path,
) -> str:
    baseline = "A0"
    base_ids = results[baseline]["ids"]
    rows = [
        "# Ablation report — CP-5b",
        "",
        f"Source artifacts: `{artifacts_dir}`",
        f"Baseline: **{baseline}** ({results[baseline]['label']})",
        "",
        "## Overlap vs baseline",
        "",
        "| variant | label | top-100 overlap | top-100 % | top-10 stability | top-50 stability | jaccard | wall (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for v in VARIANTS:
        r = results[v.code]
        ids = r["ids"]
        if v.code == baseline:
            rows.append(f"| {v.code} | {v.label} | — | — | — | — | — | {timings[v.code]:.2f} |")
            continue
        m = overlap_metrics(base_ids, ids)
        t10 = top_k_stability(base_ids, ids, 10)
        t50 = top_k_stability(base_ids, ids, 50)
        rows.append(
            f"| {v.code} | {v.label} | {int(m['overlap_count'])}/100 | "
            f"{m['overlap_pct']:.0f}% | {t10}/10 | {t50}/50 | "
            f"{m['jaccard']:.3f} | {timings[v.code]:.2f} |"
        )

    rows += [
        "",
        "## Variant definitions",
        "",
        "| variant | description |",
        "|---|---|",
    ]
    for v in VARIANTS:
        rows.append(f"| {v.code} ({v.label}) | {v.description} |")

    rows += [
        "",
        "## Interpretation guide",
        "",
        "- **Top-10 stability** is the rank-aware overlap of ranks 1–10. A drop here ",
        "  means the promotion gate / score contribution that the variant disables ",
        "  is load-bearing for the very top of the list.",
        "- **Top-100 overlap %** measures membership churn; rank order changes within ",
        "  the set are NOT visible to this metric.",
        "- **Jaccard** is symmetric and reflects how much the variant rearranges the ",
        "  shortlist boundary.",
    ]
    return "\n".join(rows) + "\n"


def run_all(
    artifacts_dir: Path,
    report_path: Path,
    out_dir: Path | None,
    weights_path: Path,
    thresholds_path: Path,
) -> dict[str, dict]:
    features = pd.read_parquet(artifacts_dir / "features.parquet")
    candidate_emb = np.load(artifacts_dir / "candidate_emb.npy")
    jd_intent_vecs = np.load(artifacts_dir / "jd_intent_vecs.npy")
    weights = _load_yaml(weights_path)
    top_10_cfg = _load_yaml(thresholds_path).get("top_10_promotion", None)

    results: dict[str, dict] = {}
    timings: dict[str, float] = {}
    for v in VARIANTS:
        t0 = time.perf_counter()
        top_100 = run_variant(v.code, features, candidate_emb, jd_intent_vecs, weights, top_10_cfg)
        timings[v.code] = time.perf_counter() - t0
        ids = top_100["candidate_id"].tolist()
        results[v.code] = {"label": v.label, "ids": ids, "top_100": top_100}
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            top_100[["candidate_id", "rank", "final_score"]].to_csv(
                out_dir / f"{v.code}_{v.label}.csv", index=False
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_format_report(results, timings, artifacts_dir))
    return results


def _parse_argv(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CP-5b ablation suite.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--report", type=Path, default=Path("ablation_report.md"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="optional dir for per-variant top-100 CSV exports",
    )
    parser.add_argument("--weights", type=Path, default=Path("config/weights.yaml"))
    parser.add_argument("--thresholds", type=Path, default=Path("config/thresholds.yaml"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_argv(argv or sys.argv[1:])
    results = run_all(args.artifacts, args.report, args.out_dir, args.weights, args.thresholds)
    base = results["A0"]["ids"]
    print(f"[ablations] wrote {args.report}")
    for v in VARIANTS:
        if v.code == "A0":
            print(f"  {v.code} ({v.label}): baseline, 100 rows")
            continue
        m = overlap_metrics(base, results[v.code]["ids"])
        t10 = top_k_stability(base, results[v.code]["ids"], 10)
        print(
            f"  {v.code} ({v.label}): overlap={int(m['overlap_count'])}/100 "
            f"top10_stability={t10}/10 jaccard={m['jaccard']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
