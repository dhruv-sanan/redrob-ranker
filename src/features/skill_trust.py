"""Conditional skill-list trust per problem.md §1.6.

evidence_factor: 1.0 if skill appears in any career description; 0.35 if only in skills[].
trust_factor:
    if assessment_score available:
        trust = score/100; halved if (score<50 AND proficiency in {advanced, expert}).
    else:
        trust = 0.55 if evidence_factor == 1.0 else 0.25.

per_skill = proficiency_ord * log(1+duration_months) * log(1+endorsements) * evidence_factor * trust_factor
skill_depth_trust = clip(sum / norm, 0, 1)
"""

from __future__ import annotations

import math
from typing import Any

from src.config_loader import load_config

PROFICIENCY_ORD = {
    "beginner": 0.25,
    "intermediate": 0.5,
    "advanced": 0.75,
    "expert": 1.0,
}

_DEFAULT_NORM = 20.0  # divisor before clipping; tuned so a Tier-A profile lands ~1.0


def _aliases(config: dict[str, str] | None) -> dict[str, str]:
    return config if config is not None else load_config("aliases")


def canonical_name(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name, name)


def per_skill_score(
    skill: dict[str, Any],
    assessment_scores: dict[str, float],
    career_text: str,
    aliases: dict[str, str],
) -> dict[str, float]:
    """Return per-skill scalars: evidence_factor, trust_factor, contribution."""
    raw_name = skill.get("name", "")
    canon = canonical_name(raw_name, aliases)
    proficiency = skill.get("proficiency", "intermediate")
    duration_months = float(skill.get("duration_months", 0) or 0)
    endorsements = float(skill.get("endorsements", 0) or 0)

    # evidence_factor — does the skill appear in any role description?
    career_lower = career_text.lower()
    appears_in_desc = raw_name.lower() in career_lower or canon.lower() in career_lower
    evidence_factor = 1.0 if appears_in_desc else 0.35

    # trust_factor — assessment-score-based, halved on under-assessment with high claim.
    canon_assess = {canonical_name(k, aliases): v for k, v in assessment_scores.items()}
    score = canon_assess.get(canon)
    if score is None:
        score = canon_assess.get(raw_name)
    if score is not None:
        trust_factor = float(score) / 100.0
        if score < 50 and proficiency in {"advanced", "expert"}:
            trust_factor *= 0.5
    else:
        trust_factor = 0.55 if appears_in_desc else 0.25

    prof_ord = PROFICIENCY_ORD.get(proficiency, 0.5)
    dur_log = math.log(1.0 + duration_months)
    end_log = math.log(1.0 + endorsements)
    contribution = prof_ord * dur_log * end_log * evidence_factor * trust_factor
    return {
        "evidence_factor": evidence_factor,
        "trust_factor": trust_factor,
        "contribution": contribution,
    }


def skill_depth_trust(
    skills: list[dict[str, Any]],
    assessment_scores: dict[str, float],
    career_history: list[dict[str, Any]],
    aliases_config: dict[str, str] | None = None,
    norm: float = _DEFAULT_NORM,
) -> float:
    """Career-level skill trust in [0, 1]."""
    if not skills:
        return 0.0
    aliases = _aliases(aliases_config)
    career_text = " ".join((r.get("description") or "") for r in (career_history or []))
    total = 0.0
    for skill in skills:
        total += per_skill_score(skill, assessment_scores or {}, career_text, aliases)[
            "contribution"
        ]
    return min(1.0, total / norm)
