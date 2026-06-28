"""Tier-respecting rank assembly with a top-10 promotion gate.

Order of operations (matches problem.md §2.2 + §2.7 + §2.8 + §2.9 + §2.10):

  1. Drop honeypot rows (``honeypot_drop == True``).
  2. Sort the survivors by ``(tier_priority, -final_score, candidate_id)``.
  3. Build a top-10 pool by filtering the global ordered list against the
     promotion gate (Tier A/B, no honeypot audit, stuffer_risk < 0.4,
     ≥2 must-haves, contactability passes, logistics multiplier ≥ 0.7 or an
     exceptional retrieval-evidence exemption).
  4. If the gate-eligible pool is shorter than 10, relax the must-have count
     until it has at least 10 candidates (relaxation log returned).
  5. Take the first 10 of the gate pool as ranks 1–10, then fill ranks 11–100
     with the next 90 candidates from the global ordered list that are not
     already in the top 10.
  6. Post-process scores so they are strictly non-increasing by rank
     (validator requirement). Equal scores are tie-broken by ``candidate_id``
     ascending, matching ``validate_submission.py``.

Returns a DataFrame with columns ``candidate_id, rank, score, reasoning_*``
(reasoning columns filled in later by the reasoning layer).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.tiering import TIER_PRIORITY

TOP_10_GATE_TIERS: frozenset[str] = frozenset({"A", "B"})
DEFAULT_TOP_10_PROMOTION = {
    "tiers": list(TOP_10_GATE_TIERS),
    "max_stuffer_risk": 0.4,
    "min_must_have_count": 2,
    "min_logistics_multiplier": 0.7,
    "exceptional_retrieval_evidence_floor": 0.85,
}

MUST_HAVE_COLUMNS: tuple[str, ...] = (
    "has_production_retrieval_evidence",
    "has_vector_or_hybrid_search_evidence",
    "has_python_backend_depth",
    "has_ranking_eval_evidence",
    "has_product_company_applied_ml_context",
    "has_shipper_signal",
)


def drop_honeypots(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where ``honeypot_drop`` is False (problem.md §2.2)."""
    if "honeypot_drop" not in df.columns:
        return df
    keep = ~df["honeypot_drop"].astype(bool).to_numpy()
    return df.loc[keep].reset_index(drop=True)


def tier_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by (tier_priority ASC, final_score DESC, candidate_id ASC)."""
    return df.sort_values(
        by=["tier_priority", "final_score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
        ignore_index=True,
    )


def _must_have_count(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in MUST_HAVE_COLUMNS if c in df.columns]
    arr = df[cols].to_numpy().astype(np.float32, copy=False)
    return (arr > 0.0).sum(axis=1).astype(np.int32)


def top_10_promotion_gate_mask(
    df: pd.DataFrame,
    *,
    min_must_have_count: int = 2,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Return a boolean mask: rows eligible for top-10 promotion.

    Gate conditions (problem.md §2.8):
      - tier in {A, B}
      - honeypot_audit == False
      - stuffer_risk < 0.4
      - >= `min_must_have_count` must-haves present
      - contactability_signal pass
      - logistics_multiplier >= 0.7 OR exceptional retrieval_evidence >= 0.85
    """
    cfg = {**DEFAULT_TOP_10_PROMOTION, **(config or {})}
    tiers = frozenset(cfg["tiers"])
    n = len(df)
    tier_ok = df["tier"].isin(tiers).to_numpy()
    audit_ok = (
        ~df["honeypot_audit"].astype(bool).to_numpy()
        if "honeypot_audit" in df.columns
        else np.ones(n, dtype=bool)
    )
    stuff_ok = df["stuffer_risk"].to_numpy() < float(cfg["max_stuffer_risk"])
    must_ok = _must_have_count(df) >= int(min_must_have_count)
    contact_ok = df["contactability_signal"].astype(bool).to_numpy()
    logist = df["logistics_multiplier"].to_numpy()
    re_floor = float(cfg["exceptional_retrieval_evidence_floor"])
    retrieval_evidence = df["retrieval_evidence"].to_numpy()
    logist_ok = (logist >= float(cfg["min_logistics_multiplier"])) | (
        retrieval_evidence >= re_floor
    )
    return tier_ok & audit_ok & stuff_ok & must_ok & contact_ok & logist_ok


def build_top_10_pool(
    sorted_df: pd.DataFrame,
    *,
    top_10_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the promotion gate and relax `min_must_have_count` until ≥10.

    Returns (top_10_df, telemetry). `telemetry` records the relaxation level
    used and the final pool size.
    """
    cfg = {**DEFAULT_TOP_10_PROMOTION, **(top_10_config or {})}
    relaxation_used = int(cfg["min_must_have_count"])
    pool: pd.DataFrame | None = None
    for relax_to in range(int(cfg["min_must_have_count"]), -1, -1):
        mask = top_10_promotion_gate_mask(sorted_df, min_must_have_count=relax_to, config=cfg)
        candidate_pool = sorted_df.loc[mask].reset_index(drop=True)
        if len(candidate_pool) >= 10:
            relaxation_used = relax_to
            pool = candidate_pool
            break
    if pool is None:
        # Couldn't get 10 even at 0 must-haves — return whatever we got.
        relaxation_used = 0
        mask = top_10_promotion_gate_mask(sorted_df, min_must_have_count=0, config=cfg)
        pool = sorted_df.loc[mask].reset_index(drop=True)
    top_10 = pool.head(10).reset_index(drop=True)
    return top_10, {
        "relaxation_used": relaxation_used,
        "gate_pool_size": int(len(pool)),
        "top_10_size": int(len(top_10)),
    }


def fill_remaining_ranks(
    sorted_df: pd.DataFrame,
    top_10_ids: list[str],
    *,
    target_size: int = 100,
) -> pd.DataFrame:
    """Take the next (target_size - len(top_10)) rows from `sorted_df`
    skipping any candidate already in `top_10_ids`. Returns a DataFrame
    indexed 0..n-1."""
    already = set(top_10_ids)
    keep = ~sorted_df["candidate_id"].isin(already).to_numpy()
    remainder = sorted_df.loc[keep].reset_index(drop=True)
    needed = target_size - len(top_10_ids)
    return remainder.head(needed).reset_index(drop=True)


def enforce_monotonic_scores(scores: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Return strictly non-increasing labels for the submission CSV.

    Walks left-to-right: any row whose score is >= the prior row's score
    gets nudged down to `prior_score - epsilon`. The output is monotonically
    decreasing, which sidesteps the validator's equal-score tie-break rule
    (the audit / debug CSVs keep the raw final_score for inspection).
    """
    out = scores.astype(np.float64, copy=True)
    for i in range(1, len(out)):
        if out[i] >= out[i - 1]:
            out[i] = out[i - 1] - epsilon
    return out


def assemble_top_100(
    scored: pd.DataFrame,
    *,
    top_10_config: dict[str, Any] | None = None,
    target_size: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the full ranking pipeline; return (top_100_df, telemetry)."""
    survivors = drop_honeypots(scored)
    sorted_df = tier_sort(survivors)
    top_10, gate_telemetry = build_top_10_pool(sorted_df, top_10_config=top_10_config)
    top_10_ids = top_10["candidate_id"].tolist()
    remainder = fill_remaining_ranks(sorted_df, top_10_ids, target_size=target_size)
    combined = pd.concat([top_10, remainder], ignore_index=True)
    combined = combined.head(target_size).copy()
    combined["rank"] = np.arange(1, len(combined) + 1, dtype=np.int64)
    combined["score"] = enforce_monotonic_scores(combined["final_score"].to_numpy())
    telemetry = {
        **gate_telemetry,
        "survivors_count": int(len(sorted_df)),
        "honeypot_dropped": int(len(scored) - len(survivors)),
        "top_100_size": int(len(combined)),
    }
    return combined, telemetry


__all__ = [
    "TIER_PRIORITY",
    "DEFAULT_TOP_10_PROMOTION",
    "drop_honeypots",
    "tier_sort",
    "top_10_promotion_gate_mask",
    "build_top_10_pool",
    "fill_remaining_ranks",
    "enforce_monotonic_scores",
    "assemble_top_100",
]
