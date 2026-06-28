"""Tests for src/features/evidence_channels.py — 3-channel detection + recency."""

from __future__ import annotations

from datetime import date

from src.features.evidence_channels import (
    _recency_weight,
    channel_hits_anywhere,
    retrieval_evidence,
    role_evidence,
)
from src.reference_date import REFERENCE_DATE


def test_role_evidence_zero_on_empty() -> None:
    r = role_evidence("")
    assert r == {"x": 0.0, "y": 0.0, "z": 0.0, "raw": 0.0}


def test_role_evidence_full_x_z() -> None:
    text = "Built BGE retrieval with NDCG eval"
    r = role_evidence(text)
    assert r["x"] > 0.0  # retrieval, NDCG, BGE
    assert r["z"] > 0.0  # Built
    assert r["raw"] > 0.3


def test_role_evidence_plain_language_y_only() -> None:
    """JD's 'right answer' clause: candidate-job matching without buzzwords."""
    text = "Built candidate-role matching system for personalized recommendations"
    r = role_evidence(text)
    assert r["x"] == 0.0  # zero exact technical
    assert r["y"] > 0.0  # matching, personalized, recommendations
    assert r["z"] > 0.0  # Built
    assert r["raw"] > 0.0


def test_tier_a_first_has_strong_retrieval_evidence(by_id: dict) -> None:
    cand = by_id["CAND_0000001"]
    score = retrieval_evidence(cand["career_history"], REFERENCE_DATE)
    assert score >= 0.5, f"Tier-A real fit must have strong retrieval evidence, got {score}"


def test_tier5_plain_has_moderate_retrieval_evidence(by_id: dict) -> None:
    cand = by_id["CAND_0000006"]
    score = retrieval_evidence(cand["career_history"], REFERENCE_DATE)
    assert score > 0.2, (
        "Plain-language fit must surface meaningful retrieval evidence "
        "via channel-Y/Z (JD's 'right answer' clause)"
    )


def test_stuffer_has_near_zero_retrieval_evidence(by_id: dict) -> None:
    cand = by_id["CAND_0000009"]  # Marketing Manager
    score = retrieval_evidence(cand["career_history"], REFERENCE_DATE)
    assert score == 0.0


def test_channel_hits_anywhere_stuffer_no_x_no_y(by_id: dict) -> None:
    cand = by_id["CAND_0000009"]
    hits = channel_hits_anywhere(cand["career_history"])
    assert hits["x"] == 0
    assert hits["y"] == 0


def test_channel_hits_anywhere_tier_a_has_x_hits(by_id: dict) -> None:
    cand = by_id["CAND_0000001"]
    hits = channel_hits_anywhere(cand["career_history"])
    assert hits["x"] >= 3


def test_recency_weight_current_role_is_full() -> None:
    assert _recency_weight(None, REFERENCE_DATE) == 1.0


def test_recency_weight_decays_with_age() -> None:
    five_years_ago = date(REFERENCE_DATE.year - 5, REFERENCE_DATE.month, REFERENCE_DATE.day)
    assert _recency_weight(five_years_ago.isoformat(), REFERENCE_DATE) == 0.3


def test_recency_weight_floored_for_ancient_roles() -> None:
    ancient = date(2000, 1, 1)
    assert _recency_weight(ancient.isoformat(), REFERENCE_DATE) == 0.3


def test_retrieval_evidence_no_career() -> None:
    assert retrieval_evidence([], REFERENCE_DATE) == 0.0
