"""Stuffer risk score per problem.md §1.7.

5-term clipped score in [0, 1]:
    0.6 * (>=6 AI advanced/expert skills)
  + 0.3 * (zero channel-X/Y hits across roles)
  + 0.2 * (>=4 AI skills AND no AI assessment for any)
  + 0.2 * (current_title is non-technical)
  - 0.4 * (has_production_retrieval_evidence > 0.3)
"""

from __future__ import annotations

from typing import Any

from src.config_loader import load_config
from src.features.evidence_channels import channel_hits_anywhere
from src.features.skill_trust import canonical_name

NON_TECHNICAL_TITLES = (
    "Marketing Manager",
    "HR Manager",
    "Sales Manager",
    "Customer Support",
    "Operations Manager",
    "Accountant",
    "Content Writer",
)


def _is_ai_skill(name: str, ai_set: set[str], aliases: dict[str, str]) -> bool:
    canon = canonical_name(name, aliases).lower()
    return name.lower() in ai_set or canon in ai_set


def stuffer_risk(
    candidate: dict[str, Any],
    has_production_retrieval_evidence: float,
    *,
    thresholds_config: dict[str, Any] | None = None,
    aliases_config: dict[str, str] | None = None,
    channels_config: dict[str, Any] | None = None,
) -> float:
    cfg = thresholds_config if thresholds_config is not None else load_config("thresholds")
    aliases = aliases_config if aliases_config is not None else load_config("aliases")
    ai_set = {s.lower() for s in cfg["stuffer"]["ai_core_skills"]}
    non_tech_titles = tuple(cfg["stuffer"].get("non_technical_titles", NON_TECHNICAL_TITLES))

    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    career = candidate.get("career_history", []) or []
    title = profile.get("current_title", "") or ""

    ai_adv = sum(
        1
        for s in skills
        if s.get("proficiency") in {"advanced", "expert"}
        and _is_ai_skill(s.get("name", ""), ai_set, aliases)
    )
    ai_listed_total = sum(1 for s in skills if _is_ai_skill(s.get("name", ""), ai_set, aliases))

    hits = channel_hits_anywhere(career, channels_config)
    no_x_or_y = (hits["x"] + hits["y"]) == 0

    scores = signals.get("skill_assessment_scores", {}) or {}
    no_ai_assess = not any(
        _is_ai_skill(k, ai_set, aliases) and float(v or 0) > 0 for k, v in scores.items()
    )
    non_tech_title_match = any(t.lower() in title.lower() for t in non_tech_titles)

    risk = 0.0
    if ai_adv >= 6:
        risk += 0.6
    if no_x_or_y:
        risk += 0.3
    if ai_listed_total >= 4 and no_ai_assess:
        risk += 0.2
    if non_tech_title_match:
        risk += 0.2
    if has_production_retrieval_evidence > 0.3:
        risk -= 0.4
    return max(0.0, min(1.0, risk))
