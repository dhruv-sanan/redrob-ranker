"""Tests for src/scoring.py — capped factors, linear blend, ceilings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.scoring import (
    MARKET_TIEBREAK_WEIGHT,
    MUST_HAVE_COLUMNS,
    apply_anti_pattern_ceilings,
    apply_multipliers_and_tiebreak,
    capped_factor_emb,
    capped_factor_skill,
    compute_embedding_similarity,
    compute_scores,
    linear_blend,
    must_have_sum_div_6,
)

WEIGHTS = {
    "blend": {
        "title_career_fit": 0.20,
        "skill_contribution": 0.18,
        "retrieval_evidence": 0.22,
        "embedding_contribution": 0.12,
        "must_have_sum_div_6": 0.10,
        "experience_band_fit": 0.08,
        "education_signal": 0.05,
        "external_validation_signal": 0.05,
    },
    "capped_factor_emb": {
        "technical_floor": {"title_career_fit": 0.4, "retrieval_evidence": 0.2},
        "cap_when_below": 0.2,
    },
    "capped_factor_skill": {
        "technical_floor": {"title_career_fit": 0.4, "retrieval_evidence": 0.0},
        "cap_when_below": 0.3,
    },
}


def _row(**overrides) -> dict:
    base = {
        "candidate_id": "CAND_0000001",
        "retrieval_evidence": 0.5,
        "title_career_fit": 0.6,
        "skill_depth_trust": 0.7,
        "experience_band_fit": 1.0,
        "education_signal": 0.75,
        "has_production_retrieval_evidence": 1.0,
        "has_vector_or_hybrid_search_evidence": 1.0,
        "has_python_backend_depth": 1.0,
        "has_ranking_eval_evidence": 0.5,
        "has_product_company_applied_ml_context": 1.0,
        "has_shipper_signal": 0.8,
        "availability_signal": 1.0,
        "contactability_signal": True,
        "market_interest_signal": 0.5,
        "external_validation_signal": 0.4,
        "logistics_multiplier": 1.0,
        "logistics_top_10_eligible": True,
        "honeypot_risk_score": 0.0,
        "honeypot_drop": False,
        "honeypot_audit": False,
        "stuffer_risk": 0.0,
        "anti_pattern_ceiling": "none",
        "anti_pattern_archetypes": [],
        "tier": "A",
        "tier_priority": 0,
    }
    base.update(overrides)
    return base


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_compute_embedding_similarity_shape_and_clip() -> None:
    rng = np.random.default_rng(0)
    cand = rng.standard_normal((20, 8)).astype(np.float32)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    jd = rng.standard_normal((4, 8)).astype(np.float32)
    jd /= np.linalg.norm(jd, axis=1, keepdims=True)
    sim = compute_embedding_similarity(cand, jd)
    assert sim.shape == (20,)
    assert (sim >= 0.0).all()
    assert (sim <= 1.0).all()


def test_capped_factor_emb_low_when_no_evidence() -> None:
    df = _df([_row(title_career_fit=0.0, retrieval_evidence=0.0)])
    assert abs(float(capped_factor_emb(df, WEIGHTS)[0]) - 0.2) < 1e-5


def test_capped_factor_emb_full_when_tcf_only() -> None:
    df = _df([_row(title_career_fit=0.5, retrieval_evidence=0.0)])
    assert float(capped_factor_emb(df, WEIGHTS)[0]) == 1.0


def test_capped_factor_emb_full_when_retrieval_only() -> None:
    df = _df([_row(title_career_fit=0.0, retrieval_evidence=0.3)])
    assert float(capped_factor_emb(df, WEIGHTS)[0]) == 1.0


def test_capped_factor_skill_low_when_no_evidence() -> None:
    df = _df([_row(title_career_fit=0.0, retrieval_evidence=0.0)])
    assert abs(float(capped_factor_skill(df, WEIGHTS)[0]) - 0.3) < 1e-5


def test_capped_factor_skill_full_when_tcf() -> None:
    df = _df([_row(title_career_fit=0.5, retrieval_evidence=0.0)])
    assert float(capped_factor_skill(df, WEIGHTS)[0]) == 1.0


def test_must_have_sum_div_6_equal_one() -> None:
    df = _df([_row(**{c: 1.0 for c in MUST_HAVE_COLUMNS})])
    assert must_have_sum_div_6(df)[0] == 1.0


def test_must_have_sum_div_6_partial() -> None:
    df = _df([_row(**{c: 0.0 for c in MUST_HAVE_COLUMNS})])
    assert must_have_sum_div_6(df)[0] == 0.0


def test_linear_blend_with_unit_features() -> None:
    df = _df([_row()])
    df["capped_factor_emb"] = 1.0
    df["capped_factor_skill"] = 1.0
    df["embedding_similarity"] = 0.8
    base = linear_blend(df, WEIGHTS)
    # Loosely: sum of weighted terms within a sane range
    assert 0.4 < float(base[0]) < 1.1


def test_multipliers_and_tiebreak() -> None:
    df = _df([_row(availability_signal=0.9, logistics_multiplier=0.8, market_interest_signal=0.5)])
    base = np.array([0.5], dtype=np.float32)
    final = apply_multipliers_and_tiebreak(df, base)
    expected = 0.5 * 0.9 * 0.8 + MARKET_TIEBREAK_WEIGHT * 0.5
    assert abs(float(final[0]) - expected) < 1e-5


def test_anti_pattern_ceilings_clip_rank_50() -> None:
    # 60 rows ranked descending, top 30 tagged rank_50.
    rows = [_row(candidate_id=f"CAND_{i:07d}", anti_pattern_ceiling="rank_50") for i in range(30)]
    rows += [_row(candidate_id=f"CAND_{i:07d}", anti_pattern_ceiling="none") for i in range(30, 60)]
    df = _df(rows)
    scores = np.linspace(1.0, 0.0, 60, dtype=np.float32)
    capped, thr = apply_anti_pattern_ceilings(df, scores)
    # rank_50-tagged rows should now sit at-or-below score_at_rank_51
    rank_50_mask = df["anti_pattern_ceiling"].to_numpy() == "rank_50"
    assert (capped[rank_50_mask] <= thr["score_at_rank_51"] + 1e-6).all()
    assert thr["rank_50_clipped_count"] == 30


def test_compute_scores_end_to_end_smoke() -> None:
    rows = [_row(candidate_id=f"CAND_{i:07d}") for i in range(20)]
    df = _df(rows)
    rng = np.random.default_rng(0)
    cand = rng.standard_normal((20, 8)).astype(np.float32)
    cand /= np.linalg.norm(cand, axis=1, keepdims=True)
    jd = rng.standard_normal((4, 8)).astype(np.float32)
    jd /= np.linalg.norm(jd, axis=1, keepdims=True)
    out, telemetry = compute_scores(df, cand, jd, WEIGHTS)
    assert "final_score" in out.columns
    assert "base_score" in out.columns
    assert out["final_score"].notna().all()
    assert isinstance(telemetry["score_at_rank_51"], float)


def test_marketing_stuffer_capped_below_real_fit() -> None:
    """End-to-end: high-skill non-tech stuffer must not outscore Tier-A real fit."""
    stuffer = _row(
        candidate_id="CAND_STUFFER",
        title_career_fit=0.0,
        retrieval_evidence=0.0,
        skill_depth_trust=1.0,
        anti_pattern_ceiling="rank_100",
        tier="D",
        availability_signal=1.0,
        logistics_multiplier=1.0,
        **{c: 0.0 for c in MUST_HAVE_COLUMNS},
    )
    real_fit = _row(
        candidate_id="CAND_REALFIT",
        title_career_fit=0.9,
        retrieval_evidence=0.8,
        skill_depth_trust=0.9,
        tier="A",
    )
    df = _df([stuffer, real_fit])
    # Both rows get a fake similar embedding to remove that as the differentiator.
    cand = np.array([[1.0] + [0.0] * 7, [1.0] + [0.0] * 7], dtype=np.float32)
    jd = np.array([[1.0] + [0.0] * 7], dtype=np.float32)
    out, _ = compute_scores(df, cand, jd, WEIGHTS)
    stuffer_final = float(out[out["candidate_id"] == "CAND_STUFFER"]["final_score"].iloc[0])
    real_final = float(out[out["candidate_id"] == "CAND_REALFIT"]["final_score"].iloc[0])
    assert real_final > stuffer_final, (real_final, stuffer_final)
