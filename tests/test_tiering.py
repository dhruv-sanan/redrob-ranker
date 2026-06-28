"""Tests for src/features/tiering.py — Tier A–E assignment."""

from __future__ import annotations

from src.features.tiering import TIER_PRIORITY, assign_tier


def _full_must_haves(**overrides):
    base = {
        "has_production_retrieval_evidence": 0.0,
        "has_vector_or_hybrid_search_evidence": 0.0,
        "has_python_backend_depth": 0.0,
        "has_ranking_eval_evidence": 0.0,
        "has_product_company_applied_ml_context": 0.0,
        "has_shipper_signal": 0.0,
    }
    base.update(overrides)
    return base


def test_tier_e_when_honeypot_drop() -> None:
    tier = assign_tier({}, _full_must_haves(), stuffer_risk_score=0.0, honeypot_drop=True)
    assert tier == "E"


def test_tier_e_when_stuffer_risk_high() -> None:
    tier = assign_tier({}, _full_must_haves(), stuffer_risk_score=0.8, honeypot_drop=False)
    assert tier == "E"


def test_tier_a_when_4_of_5_must_haves_pass() -> None:
    tier = assign_tier(
        {"current_title": "Senior ML Engineer"},
        _full_must_haves(
            has_production_retrieval_evidence=0.8,
            has_vector_or_hybrid_search_evidence=0.5,
            has_python_backend_depth=0.7,
            has_ranking_eval_evidence=0.5,
            has_product_company_applied_ml_context=1.0,
            has_shipper_signal=0.8,
        ),
        stuffer_risk_score=0.0,
        honeypot_drop=False,
    )
    assert tier == "A"


def test_tier_b_when_two_must_haves_and_shipper() -> None:
    tier = assign_tier(
        {"current_title": "Backend Engineer"},
        _full_must_haves(
            has_product_company_applied_ml_context=1.0,
            has_python_backend_depth=0.7,
            has_shipper_signal=0.7,
        ),
        stuffer_risk_score=0.0,
        honeypot_drop=False,
    )
    assert tier == "B"


def test_tier_d_for_non_tech_title() -> None:
    tier = assign_tier(
        {"current_title": "Marketing Manager"},
        _full_must_haves(),
        stuffer_risk_score=0.5,
        honeypot_drop=False,
    )
    assert tier == "D"


def test_tier_c_for_adjacent_engineer() -> None:
    """Some backend depth or product company exposure but missing retrieval evidence."""
    tier = assign_tier(
        {"current_title": "Software Engineer"},
        _full_must_haves(has_python_backend_depth=0.7),
        stuffer_risk_score=0.0,
        honeypot_drop=False,
    )
    assert tier == "C"


def test_tier_priority_ordering() -> None:
    assert TIER_PRIORITY["A"] < TIER_PRIORITY["B"] < TIER_PRIORITY["C"]
    assert TIER_PRIORITY["C"] < TIER_PRIORITY["D"] < TIER_PRIORITY["E"]
