"""Tier assignment per problem.md §1.12.

Tier A: ≥4 must-haves at the configured thresholds.
Tier B: ≥2 must-haves AND has_shipper_signal ≥ 0.5.
Tier C: adjacent ML/data/backend, missing explicit retrieval/ranking proof.
Tier D: non-technical title or skills-list-only AI footprint.
Tier E: honeypot_drop OR stuffer_risk ≥ tier_e.stuffer_risk threshold.

`tier_priority`: A=0, B=1, C=2, D=3, E=4. Lower wins.
"""

from __future__ import annotations

from typing import Any

from src.config_loader import load_config

TIER_PRIORITY = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _adjacent(must_haves: dict[str, float]) -> bool:
    """Tier-C heuristic — some Python/backend or ranking eval but not enough for B."""
    return (
        must_haves.get("has_python_backend_depth", 0.0) >= 0.4
        or must_haves.get("has_product_company_applied_ml_context", 0.0) >= 1.0
        or must_haves.get("has_shipper_signal", 0.0) >= 0.5
    )


def _non_tech_or_skills_only(
    profile: dict[str, Any], must_haves: dict[str, float], stuffer_risk_score: float
) -> bool:
    title = (profile.get("current_title", "") or "").lower()
    non_tech_titles = ("marketing", "sales", "hr", "customer support", "operations")
    if any(t in title for t in non_tech_titles):
        return True
    if stuffer_risk_score >= 0.4:
        return True
    return sum(1 for v in must_haves.values() if v >= 0.5) == 0


def assign_tier(
    profile: dict[str, Any],
    must_haves: dict[str, float],
    stuffer_risk_score: float,
    honeypot_drop: bool,
    *,
    thresholds_config: dict[str, Any] | None = None,
) -> str:
    cfg = thresholds_config if thresholds_config is not None else load_config("thresholds")
    tier_a_cfg = cfg["tiering"]["tier_a"]
    tier_b_cfg = cfg["tiering"]["tier_b"]
    tier_e_cfg = cfg["tiering"]["tier_e"]

    if honeypot_drop or stuffer_risk_score >= float(tier_e_cfg["stuffer_risk"]):
        return "E"

    must_have_keys = (
        "has_production_retrieval_evidence",
        "has_vector_or_hybrid_search_evidence",
        "has_python_backend_depth",
        "has_ranking_eval_evidence",
        "has_product_company_applied_ml_context",
    )
    tier_a_passes = sum(1 for k in must_have_keys if must_haves.get(k, 0.0) >= float(tier_a_cfg[k]))
    if tier_a_passes >= int(tier_a_cfg["min_must_haves"]):
        return "A"

    passes_any = sum(1 for k in must_have_keys if must_haves.get(k, 0.0) >= 0.4)
    if passes_any >= int(tier_b_cfg["min_must_haves"]) and must_haves.get(
        "has_shipper_signal", 0.0
    ) >= float(tier_b_cfg["has_shipper_signal"]):
        return "B"

    if _non_tech_or_skills_only(profile, must_haves, stuffer_risk_score):
        return "D"

    if _adjacent(must_haves):
        return "C"

    return "D"
