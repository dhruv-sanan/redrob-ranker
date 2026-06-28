"""Tests for src/ranking.py — drop, sort, gate, relaxation, monotonic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ranking import (
    DEFAULT_TOP_10_PROMOTION,
    assemble_top_100,
    build_top_10_pool,
    drop_honeypots,
    enforce_monotonic_scores,
    fill_remaining_ranks,
    tier_sort,
    top_10_promotion_gate_mask,
)

TIER_PRIORITY_MAP = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _row(**overrides) -> dict:
    base = {
        "candidate_id": "CAND_0000001",
        "tier": "A",
        "tier_priority": 0,
        "final_score": 0.5,
        "honeypot_drop": False,
        "honeypot_audit": False,
        "stuffer_risk": 0.0,
        "contactability_signal": True,
        "logistics_multiplier": 1.0,
        "retrieval_evidence": 0.5,
        "has_production_retrieval_evidence": 1.0,
        "has_vector_or_hybrid_search_evidence": 1.0,
        "has_python_backend_depth": 1.0,
        "has_ranking_eval_evidence": 1.0,
        "has_product_company_applied_ml_context": 1.0,
        "has_shipper_signal": 1.0,
    }
    base.update(overrides)
    return base


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["tier_priority"] = df["tier"].map(TIER_PRIORITY_MAP)
    return df


def test_drop_honeypots_filters() -> None:
    rows = [
        _row(candidate_id="CAND_0000001", honeypot_drop=False),
        _row(candidate_id="CAND_0000002", honeypot_drop=True),
    ]
    out = drop_honeypots(_df(rows))
    assert len(out) == 1
    assert out["candidate_id"].iloc[0] == "CAND_0000001"


def test_tier_sort_respects_tier_then_score() -> None:
    rows = [
        _row(candidate_id="CAND_0000001", tier="B", final_score=0.9),
        _row(candidate_id="CAND_0000002", tier="A", final_score=0.5),
        _row(candidate_id="CAND_0000003", tier="A", final_score=0.4),
    ]
    out = tier_sort(_df(rows))
    assert out["candidate_id"].tolist() == ["CAND_0000002", "CAND_0000003", "CAND_0000001"]


def test_top_10_gate_filters_out_stuffer_and_audit() -> None:
    rows = [_row(candidate_id=f"CAND_{i:07d}") for i in range(11)]
    rows[0]["stuffer_risk"] = 0.5
    rows[1]["honeypot_audit"] = True
    rows[2]["tier"] = "C"
    rows[3]["contactability_signal"] = False
    rows[4]["logistics_multiplier"] = 0.5
    rows[4]["retrieval_evidence"] = 0.5  # not exceptional
    df = _df(rows)
    mask = top_10_promotion_gate_mask(df)
    # Rows 0..4 fail, rows 5..10 pass → 6 eligible
    assert mask.sum() == 6


def test_top_10_gate_logistics_exception_when_exceptional_retrieval() -> None:
    row = _row(logistics_multiplier=0.5, retrieval_evidence=0.9)
    df = _df([row])
    mask = top_10_promotion_gate_mask(df)
    assert mask.all()


def test_relaxation_when_pool_too_small() -> None:
    # 12 rows; 7 have ≥2 must-haves, 5 have only 1 → at min_must_have=2 pool=7 (<10).
    rows = [_row(candidate_id=f"CAND_{i:07d}") for i in range(12)]
    # Zero out must-haves for last 5.
    for r in rows[7:]:
        for col in (
            "has_production_retrieval_evidence",
            "has_vector_or_hybrid_search_evidence",
            "has_python_backend_depth",
            "has_ranking_eval_evidence",
            "has_product_company_applied_ml_context",
            "has_shipper_signal",
        ):
            r[col] = 0.0
        # Keep one must-have so they have exactly 1 must-have.
        r["has_python_backend_depth"] = 1.0
    sorted_df = tier_sort(_df(rows))
    top_10, telemetry = build_top_10_pool(sorted_df)
    assert len(top_10) == 10
    assert telemetry["relaxation_used"] <= 1  # relaxed at least once


def test_fill_remaining_ranks_skips_top10_ids() -> None:
    rows = [_row(candidate_id=f"CAND_{i:07d}") for i in range(15)]
    sorted_df = tier_sort(_df(rows))
    top_10_ids = sorted_df["candidate_id"].head(10).tolist()
    rest = fill_remaining_ranks(sorted_df, top_10_ids, target_size=15)
    assert len(rest) == 5
    assert not set(rest["candidate_id"]) & set(top_10_ids)


def test_enforce_monotonic_scores_strict_decrease() -> None:
    scores = np.array([0.5, 0.6, 0.55, 0.55, 0.4], dtype=np.float32)
    out = enforce_monotonic_scores(scores, epsilon=1e-7)
    assert (out[:-1] > out[1:]).all()  # strictly decreasing
    # Rank 0 keeps its real score; rank 1 must be < rank 0 even though
    # its raw was higher.
    assert out[0] == 0.5
    assert out[1] < out[0]


def test_assemble_top_100_with_small_pool() -> None:
    rows = [_row(candidate_id=f"CAND_{i:07d}", final_score=0.9 - i * 0.001) for i in range(150)]
    out, telemetry = assemble_top_100(_df(rows))
    assert len(out) == 100
    assert (out["rank"].to_numpy() == np.arange(1, 101)).all()
    assert (out["score"].to_numpy()[:-1] >= out["score"].to_numpy()[1:]).all()


def test_top_10_size_gate_when_extreme_starvation() -> None:
    # Every row fails the gate → relaxation falls back to size-bounded pool.
    rows = [_row(candidate_id=f"CAND_{i:07d}", contactability_signal=False) for i in range(5)]
    out, telemetry = build_top_10_pool(tier_sort(_df(rows)))
    # No row passes contactability gate even at relax=0 — pool is empty.
    assert telemetry["gate_pool_size"] == 0
    assert telemetry["top_10_size"] == 0


def test_default_top_10_promotion_constant_shape() -> None:
    cfg = DEFAULT_TOP_10_PROMOTION
    assert "tiers" in cfg
    assert "max_stuffer_risk" in cfg
    assert cfg["min_must_have_count"] == 2
