"""Tests for src/features/skill_trust.py — conditional trust + alias normalization."""

from __future__ import annotations

from src.features.skill_trust import canonical_name, per_skill_score, skill_depth_trust


def test_canonical_name_normalizes_alias() -> None:
    aliases = {"Fine-tuning LLMs": "Fine-tuning"}
    assert canonical_name("Fine-tuning LLMs", aliases) == "Fine-tuning"
    assert canonical_name("Python", aliases) == "Python"


def test_per_skill_high_assessment_full_trust() -> None:
    skill = {"name": "NLP", "proficiency": "expert", "endorsements": 50, "duration_months": 60}
    out = per_skill_score(skill, {"NLP": 90.0}, "expert NLP work", {})
    assert out["evidence_factor"] == 1.0
    assert out["trust_factor"] == 0.9
    assert out["contribution"] > 1.0


def test_per_skill_low_assessment_halves_trust() -> None:
    skill = {"name": "NLP", "proficiency": "advanced", "endorsements": 20, "duration_months": 24}
    out = per_skill_score(skill, {"NLP": 38.0}, "", {})
    # 38/100 * 0.5 (halved because advanced+<50) = 0.19
    assert out["trust_factor"] == 0.19


def test_per_skill_no_assessment_with_description_evidence() -> None:
    skill = {"name": "NLP", "proficiency": "advanced", "endorsements": 10, "duration_months": 24}
    out = per_skill_score(skill, {}, "did some NLP work", {})
    assert out["evidence_factor"] == 1.0
    assert out["trust_factor"] == 0.55


def test_per_skill_no_assessment_no_description_evidence() -> None:
    skill = {"name": "NLP", "proficiency": "expert", "endorsements": 5, "duration_months": 12}
    out = per_skill_score(skill, {}, "marketing strategy", {})
    assert out["evidence_factor"] == 0.35
    assert out["trust_factor"] == 0.25


def test_per_skill_alias_lookup_resolves_assessment() -> None:
    """Skill listed as 'Natural Language Processing'; assessment recorded under 'NLP'."""
    aliases = {"Natural Language Processing": "NLP"}
    skill = {
        "name": "Natural Language Processing",
        "proficiency": "expert",
        "endorsements": 30,
        "duration_months": 48,
    }
    out = per_skill_score(skill, {"NLP": 88.0}, "production NLP", aliases)
    assert out["trust_factor"] == 0.88


def test_per_skill_alias_lookup_reverse_direction() -> None:
    """Reverse case: skill under canonical, assessment under verbose alias."""
    aliases = {"Fine-tuning LLMs": "Fine-tuning"}
    skill = {
        "name": "Fine-tuning",
        "proficiency": "expert",
        "endorsements": 20,
        "duration_months": 24,
    }
    out = per_skill_score(skill, {"Fine-tuning LLMs": 85.0}, "fine-tuning work", aliases)
    assert out["trust_factor"] == 0.85


def test_skill_depth_trust_tier_a_high(by_id: dict) -> None:
    cand = by_id["CAND_0000001"]
    score = skill_depth_trust(
        cand["skills"],
        cand["redrob_signals"]["skill_assessment_scores"],
        cand["career_history"],
    )
    assert score > 0.5


def test_skill_depth_trust_stuffer_low(by_id: dict) -> None:
    cand = by_id["CAND_0000009"]  # Marketing Manager, listed AI skills, no assessment
    score = skill_depth_trust(
        cand["skills"],
        cand["redrob_signals"]["skill_assessment_scores"],
        cand["career_history"],
    )
    assert score < 0.3, f"Stuffer skill_depth_trust should be low, got {score}"


def test_skill_depth_trust_no_skills() -> None:
    assert skill_depth_trust([], {}, []) == 0.0
