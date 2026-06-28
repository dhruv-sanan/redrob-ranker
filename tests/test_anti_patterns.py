"""Tests for src/features/anti_patterns.py — archetype boolean rules + ceiling tags."""

from __future__ import annotations

from src.features.anti_patterns import compute_anti_pattern_ceiling
from src.reference_date import REFERENCE_DATE


def _ceiling(by_id, cid, *, retrieval=0.0, product_company=0.0):
    return compute_anti_pattern_ceiling(
        by_id[cid],
        retrieval_evidence=retrieval,
        has_product_company_applied_ml_context=product_company,
        reference=REFERENCE_DATE,
    )


def test_tier_a_no_ceiling(by_id: dict) -> None:
    out = _ceiling(by_id, "CAND_0000001", retrieval=0.8, product_company=1.0)
    assert out["ceiling"] is None
    assert out["archetypes"] == []


def test_stuffer_non_tech_title_ceiling(by_id: dict) -> None:
    """Marketing Manager with retrieval_evidence=0 → rank_50 ceiling."""
    out = _ceiling(by_id, "CAND_0000009", retrieval=0.0, product_company=0.0)
    assert out["ceiling"] == "rank_50"
    assert "non_tech_title" in out["archetypes"]


def test_services_only_ceiling_fires(by_id: dict) -> None:
    out = _ceiling(by_id, "CAND_0000017", retrieval=0.0, product_company=0.0)  # TCS
    assert "services_only" in out["archetypes"]
    assert out["ceiling"] == "rank_50"


def test_services_to_product_ceiling_does_not_fire(by_id: dict) -> None:
    """Has a product-company applied-ML role → services_only must NOT fire."""
    out = _ceiling(by_id, "CAND_0000022", retrieval=0.6, product_company=1.0)
    assert "services_only" not in out["archetypes"]


def test_cv_only_ceiling_fires(by_id: dict) -> None:
    out = _ceiling(by_id, "CAND_0000025", retrieval=0.0, product_company=0.0)
    assert "cv_speech_robotics_only" in out["archetypes"]


def test_recent_langchain_ceiling_fires(by_id: dict) -> None:
    out = _ceiling(by_id, "CAND_0000028", retrieval=0.0, product_company=0.0)
    assert "recent_only_langchain" in out["archetypes"]


def test_inactive_architect_ceiling_fires(by_id: dict) -> None:
    out = _ceiling(by_id, "CAND_0000030", retrieval=0.0, product_company=0.0)
    assert "inactive_architect" in out["archetypes"]


def test_tier_a_strong_retrieval_clears_non_tech_industry(by_id: dict) -> None:
    """Even if industry matches, high retrieval_evidence avoids the rank_100 cap."""
    out = _ceiling(by_id, "CAND_0000001", retrieval=0.8, product_company=1.0)
    assert "non_tech_industry" not in out["archetypes"]


def test_tier5_plain_no_ceiling(by_id: dict) -> None:
    """Plain-language Tier-5 fit must NOT receive an anti-pattern ceiling."""
    out = _ceiling(by_id, "CAND_0000006", retrieval=0.4, product_company=1.0)
    assert out["ceiling"] is None
