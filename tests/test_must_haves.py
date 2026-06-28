"""Tests for src/features/must_haves.py — 6 graded must-haves."""

from __future__ import annotations

from src.features.must_haves import (
    compute_must_haves,
    has_product_company_applied_ml_context,
    has_production_retrieval_evidence,
    has_python_backend_depth,
    has_ranking_eval_evidence,
    has_shipper_signal,
    has_vector_or_hybrid_search_evidence,
)


def test_tier_a_real_fit_all_must_haves(by_id: dict) -> None:
    cand = by_id["CAND_0000001"]
    out = compute_must_haves(cand)
    assert out["has_production_retrieval_evidence"] >= 0.5
    assert out["has_vector_or_hybrid_search_evidence"] >= 0.4
    assert out["has_python_backend_depth"] >= 0.5
    assert out["has_ranking_eval_evidence"] >= 0.3
    assert out["has_product_company_applied_ml_context"] == 1.0
    assert out["has_shipper_signal"] >= 0.5


def test_stuffer_no_must_haves(by_id: dict) -> None:
    cand = by_id["CAND_0000009"]  # Marketing Manager
    out = compute_must_haves(cand)
    assert out["has_production_retrieval_evidence"] == 0.0
    assert out["has_vector_or_hybrid_search_evidence"] == 0.0
    assert out["has_python_backend_depth"] == 0.0
    assert out["has_ranking_eval_evidence"] == 0.0
    assert out["has_product_company_applied_ml_context"] == 0.0


def test_tier5_plain_has_product_company_context(by_id: dict) -> None:
    out = compute_must_haves(by_id["CAND_0000006"])
    assert out["has_product_company_applied_ml_context"] == 1.0
    assert out["has_python_backend_depth"] >= 0.4


def test_services_only_no_product_company_context(by_id: dict) -> None:
    out = compute_must_haves(by_id["CAND_0000017"])  # TCS
    assert out["has_product_company_applied_ml_context"] == 0.0


def test_services_to_product_does_have_product_context(by_id: dict) -> None:
    out = compute_must_haves(by_id["CAND_0000022"])  # Cognizant -> BookMyShow
    assert out["has_product_company_applied_ml_context"] == 1.0


def test_has_production_retrieval_grades_by_hits() -> None:
    one_hit_role = [
        {
            "description": "Built retrieval system",
            "title": "Engineer",
            "start_date": "2023-01-01",
            "end_date": None,
            "duration_months": 24,
            "is_current": True,
            "industry": "Internet",
            "company": "X",
            "company_size": "501-1000",
        }
    ]
    score = has_production_retrieval_evidence(one_hit_role)
    assert 0.2 < score < 0.5  # 1 X + ownership


def test_has_python_backend_depth_requires_advanced_expert() -> None:
    skills = [{"name": "Python", "proficiency": "intermediate", "endorsements": 10}]
    assert (
        has_python_backend_depth(skills, [{"title": "ML Engineer", "duration_months": 36}]) == 0.0
    )


def test_has_vector_or_hybrid_search_picks_up_named_dbs(by_id: dict) -> None:
    cand = by_id["CAND_0000002"]  # Vectorly + Qdrant
    out = has_vector_or_hybrid_search_evidence(cand["career_history"])
    assert out >= 0.4


def test_has_ranking_eval_fires_on_ndcg() -> None:
    career = [{"description": "Owned NDCG and MRR dashboards"}]
    assert has_ranking_eval_evidence(career) >= 0.4


def test_has_shipper_signal_zero_without_shipping_verbs() -> None:
    career = [{"description": "Researched fundamentals"}]
    assert has_shipper_signal(career) == 0.0


def test_has_product_company_skips_services() -> None:
    career = [{"industry": "IT Services", "description": "machine learning work"}]
    assert has_product_company_applied_ml_context(career) == 0.0
