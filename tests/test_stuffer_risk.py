"""Tests for src/features/stuffer_risk.py — 5-term clipped risk score."""

from __future__ import annotations

from src.features.stuffer_risk import stuffer_risk


def test_stuffer_marketing_manager_high_risk(by_id: dict) -> None:
    cand = by_id["CAND_0000009"]  # 9 AI skills, no assessments, Marketing Manager
    risk = stuffer_risk(cand, has_production_retrieval_evidence=0.0)
    assert risk >= 0.7


def test_tier_a_low_stuffer_risk(by_id: dict) -> None:
    cand = by_id["CAND_0000001"]
    risk = stuffer_risk(cand, has_production_retrieval_evidence=0.8)
    assert risk < 0.3


def test_tier5_plain_low_stuffer_risk(by_id: dict) -> None:
    cand = by_id["CAND_0000006"]
    risk = stuffer_risk(cand, has_production_retrieval_evidence=0.4)
    assert risk < 0.3


def test_services_only_moderate_risk(by_id: dict) -> None:
    """Services-only doesn't have stuffed AI skills, but flagged via no_x_or_y."""
    cand = by_id["CAND_0000017"]
    risk = stuffer_risk(cand, has_production_retrieval_evidence=0.0)
    assert risk < 0.7
    assert risk > 0.0


def test_production_retrieval_evidence_deducts_risk(by_id: dict) -> None:
    """Same stuffer profile should drop in risk when retrieval evidence is present."""
    cand = by_id["CAND_0000009"]
    high_risk = stuffer_risk(cand, has_production_retrieval_evidence=0.0)
    low_risk = stuffer_risk(cand, has_production_retrieval_evidence=0.5)
    assert low_risk < high_risk


def test_stuffer_risk_clipped_to_unit_interval(by_id: dict) -> None:
    risk = stuffer_risk(by_id["CAND_0000009"], has_production_retrieval_evidence=0.0)
    assert 0.0 <= risk <= 1.0
