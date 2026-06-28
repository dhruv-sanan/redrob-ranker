#!/usr/bin/env python3
"""CP-S2b weight search — label-grounded re-tune.

Random-search 200 blend-weight vectors in a ±0.10 box around the
current ``config/weights.yaml`` (V8) baseline. For each variant,
re-runs ``compute_scores → assemble_top_100`` in-process against
the loaded artifacts (~0.25 s per variant on M4) and scores the
result by:

    multi = 10 * label_grounded_pass_count
          +  5 * predicate_bucket_pass_count
          +  5 * jaccard@100_vs_V8
          -  2 * |plain_language_median - 45|

Reports the top-5 Pareto candidates, the best PASS-all variant
(if any), and writes ``weight_search_results.json`` + a brief
markdown table to stdout. **Does NOT mutate config/weights.yaml**
— operator reviews and applies manually.

Usage:
    python tools/weight_search.py [--n 200] [--seed 42] \\
        [--artifacts ./artifacts/] [--holdout holdout_labels.csv]
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ranking import assemble_top_100  # noqa: E402
from src.scoring import compute_scores  # noqa: E402
from tools.holdout_report import (  # noqa: E402
    _join_with_ranks,
    label_grounded_assertions,
    predicate_assertions,
)

BLEND_KEYS = (
    "title_career_fit",
    "skill_contribution",
    "retrieval_evidence",
    "embedding_contribution",
    "must_have_sum_div_6",
    "experience_band_fit",
    "education_signal",
    "external_validation_signal",
)

BOX_HALF_WIDTH = 0.10  # default; CLI --box overrides


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.clip(vec, 0.0, None)
    s = vec.sum()
    return vec / s if s > 0 else vec


def _sample_variant(
    rng: np.random.Generator, anchor: np.ndarray, box: float = BOX_HALF_WIDTH
) -> np.ndarray:
    perturb = rng.uniform(-box, box, size=anchor.shape)
    return _normalize(anchor + perturb)


def _pl_median(joined: pd.DataFrame) -> float:
    pl = joined.loc[joined["bucket"] == "plain_language"]
    ranked = pl.loc[pl["current_rank"].notna(), "current_rank"]
    return float(ranked.median()) if not ranked.empty else 100.0


def _churn(base_ids: list[str], var_ids: list[str], k: int) -> int:
    return len(set(var_ids[:k]) - set(base_ids[:k]))


def _run_pipeline(
    weights: dict,
    features: pd.DataFrame,
    candidate_emb: np.ndarray,
    jd_intent_vecs: np.ndarray,
    top_10_cfg: dict | None,
) -> pd.DataFrame:
    scored, _ = compute_scores(features, candidate_emb, jd_intent_vecs, weights)
    top_100, _ = assemble_top_100(scored, top_10_config=top_10_cfg)
    return top_100


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="CP-S2b weight random-search.")
    parser.add_argument("--n", type=int, default=200, help="number of variants")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--box", type=float, default=BOX_HALF_WIDTH, help="±box around V8")
    parser.add_argument("--artifacts", type=Path, default=_REPO_ROOT / "artifacts")
    parser.add_argument("--holdout", type=Path, default=_REPO_ROOT / "holdout_labels.csv")
    parser.add_argument("--weights", type=Path, default=_REPO_ROOT / "config" / "weights.yaml")
    parser.add_argument(
        "--thresholds", type=Path, default=_REPO_ROOT / "config" / "thresholds.yaml"
    )
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "weight_search_results.json")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    print("[weight_search] loading artifacts...")
    features = pd.read_parquet(args.artifacts / "features.parquet")
    candidate_emb = np.load(args.artifacts / "candidate_emb.npy")
    jd_intent_vecs = np.load(args.artifacts / "jd_intent_vecs.npy")
    base_weights = yaml.safe_load(args.weights.read_text(encoding="utf-8"))
    thresholds = yaml.safe_load(args.thresholds.read_text(encoding="utf-8"))
    top_10_cfg = thresholds.get("top_10_promotion")
    holdout_seed = pd.read_csv(args.holdout)

    anchor = np.array([base_weights["blend"][k] for k in BLEND_KEYS], dtype=np.float64)
    print(f"[weight_search] V8 anchor: {dict(zip(BLEND_KEYS, anchor, strict=True))}")

    # --- baseline (V8) -------------------------------------------------------
    print("[weight_search] running V8 baseline...")
    t0 = time.perf_counter()
    base_top_100 = _run_pipeline(base_weights, features, candidate_emb, jd_intent_vecs, top_10_cfg)
    base_ids = base_top_100["candidate_id"].tolist()
    base_joined = _join_with_ranks(holdout_seed, base_top_100)
    base_label = label_grounded_assertions(base_joined)
    base_pred = predicate_assertions(base_joined)
    base_label_pass = sum(1 for r in base_label if r.passed)
    base_pred_pass = sum(1 for r in base_pred if r.passed)
    base_pl_med = _pl_median(base_joined)
    print(
        f"[weight_search] V8: label={base_label_pass}/4 pred={base_pred_pass}/9 "
        f"pl_median={base_pl_med:.1f} wall={time.perf_counter() - t0:.2f}s"
    )

    # --- sweep ---------------------------------------------------------------
    results: list[dict] = []
    t_sweep = time.perf_counter()
    for i in range(args.n):
        weight_vec = _sample_variant(rng, anchor, box=args.box)
        variant = copy.deepcopy(base_weights)
        for k, v in zip(BLEND_KEYS, weight_vec, strict=True):
            variant["blend"][k] = float(v)

        top_100 = _run_pipeline(variant, features, candidate_emb, jd_intent_vecs, top_10_cfg)
        ids = top_100["candidate_id"].tolist()
        joined = _join_with_ranks(holdout_seed, top_100)

        label = label_grounded_assertions(joined)
        pred = predicate_assertions(joined)
        label_pass = sum(1 for r in label if r.passed)
        pred_pass = sum(1 for r in pred if r.passed)
        jacc_100 = len(set(base_ids) & set(ids)) / len(set(base_ids) | set(ids))
        pl_med = _pl_median(joined)
        c10 = _churn(base_ids, ids, 10)
        c50 = _churn(base_ids, ids, 50)
        c100 = _churn(base_ids, ids, 100)

        multi = 10.0 * label_pass + 5.0 * pred_pass + 5.0 * jacc_100 - 2.0 * abs(pl_med - 45.0)

        results.append(
            {
                "variant_idx": i,
                "weights": dict(zip(BLEND_KEYS, [float(x) for x in weight_vec], strict=True)),
                "label_pass": label_pass,
                "pred_pass": pred_pass,
                "jaccard_100": float(jacc_100),
                "pl_median": float(pl_med),
                "churn_10": int(c10),
                "churn_50": int(c50),
                "churn_100": int(c100),
                "multi_score": float(multi),
                "label_fail_names": [r.name for r in label if not r.passed],
                "pred_fail_names": [r.name for r in pred if not r.passed],
            }
        )

    sweep_wall = time.perf_counter() - t_sweep
    print(f"[weight_search] swept {args.n} variants in {sweep_wall:.1f}s")

    # --- analysis ------------------------------------------------------------
    results.sort(key=lambda r: r["multi_score"], reverse=True)

    # Filter to "PASS all" candidates per final.md §2.1 acceptance.
    pass_all = [
        r
        for r in results
        if r["label_pass"] == 4
        and r["pred_pass"] == 9
        and r["churn_10"] <= 3
        and r["churn_50"] <= 5
    ]
    pass_all.sort(key=lambda r: (r["churn_10"] + r["churn_50"], -r["jaccard_100"]))

    print()
    print("=== Top-5 by multi_score ===")
    print(
        f"  {'idx':>4}  {'label':>5}  {'pred':>4}  {'jacc':>5}  "
        f"{'pl_med':>6}  {'c10':>4}  {'c50':>4}  {'c100':>4}  {'multi':>7}"
    )
    for r in results[:5]:
        print(
            f"  {r['variant_idx']:>4}  {r['label_pass']:>2}/4   "
            f"{r['pred_pass']:>2}/9  {r['jaccard_100']:.3f}  "
            f"{r['pl_median']:>6.1f}  {r['churn_10']:>4}  "
            f"{r['churn_50']:>4}  {r['churn_100']:>4}  {r['multi_score']:>7.2f}"
        )

    print()
    if pass_all:
        winner = pass_all[0]
        print(f"=== PASS-all winner: variant {winner['variant_idx']} ===")
        print(
            f"  label_pass={winner['label_pass']}/4  "
            f"pred_pass={winner['pred_pass']}/9  "
            f"churn_10={winner['churn_10']}  churn_50={winner['churn_50']}  "
            f"churn_100={winner['churn_100']}  pl_med={winner['pl_median']:.1f}  "
            f"jaccard={winner['jaccard_100']:.3f}"
        )
        print("  weight deltas vs V8:")
        for k in BLEND_KEYS:
            delta = winner["weights"][k] - base_weights["blend"][k]
            print(
                f"    {k:>28}  {base_weights['blend'][k]:.4f} → "
                f"{winner['weights'][k]:.4f}  ({delta:+.4f})"
            )
    else:
        print("=== No variant PASSed all acceptance criteria ===")
        print("Best multi-score variant is recommended for partial uplift:")
        best = results[0]
        print(
            f"  variant {best['variant_idx']}: label={best['label_pass']}/4  "
            f"pred={best['pred_pass']}/9  churn_10={best['churn_10']}  "
            f"churn_50={best['churn_50']}  pl_med={best['pl_median']:.1f}"
        )
        print(f"  label fails: {best['label_fail_names']}")
        print(f"  pred  fails: {best['pred_fail_names']}")

    args.out.write_text(
        json.dumps(
            {
                "baseline": {
                    "weights": {k: base_weights["blend"][k] for k in BLEND_KEYS},
                    "label_pass": base_label_pass,
                    "pred_pass": base_pred_pass,
                    "pl_median": base_pl_med,
                },
                "all_results": results,
                "pass_all": pass_all,
                "sweep_wall_s": sweep_wall,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(f"[weight_search] wrote {args.out.relative_to(_REPO_ROOT)}")

    return 0 if pass_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
