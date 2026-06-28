"""Anti-pattern archetype detection per problem.md §1.9.

Each archetype yields a (matched: bool, ceiling: str | None) decision. Ceilings
are tags: ``"rank_50"`` or ``"rank_100"`` — applied post-blend in Phase 4.

The output dict has a single ``ceiling`` field equal to the strongest ceiling
across all archetypes (rank_50 > rank_100 > None).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.config_loader import load_config
from src.features.evidence_channels import _compile_patterns

CEILING_RANK = {"rank_50": 2, "rank_100": 1, "none": 0}


def _strongest(*ceilings: str | None) -> str | None:
    best = "none"
    for c in ceilings:
        if c and CEILING_RANK[c] > CEILING_RANK[best]:
            best = c
    return None if best == "none" else best


def _title_matches(title: str, candidates: list[str]) -> bool:
    title_low = (title or "").lower()
    return any(c.lower() in title_low for c in candidates)


def _industry_matches(industry: str, candidates: list[str]) -> bool:
    ind_low = (industry or "").lower()
    return any(c.lower() in ind_low for c in candidates)


def _company_in(company: str, services: list[str]) -> bool:
    comp_low = (company or "").lower()
    return any(s.lower() in comp_low for s in services)


def _months_since(date_str: str | None, reference: date) -> int:
    if not date_str:
        return 9999
    try:
        d = date.fromisoformat(date_str)
    except (TypeError, ValueError):
        return 9999
    return max(0, int((reference - d).days / 30.4375))


def non_tech_title_ceiling(
    profile: dict[str, Any],
    retrieval_evidence: float,
    config: dict[str, Any],
) -> str | None:
    cfg = config["non_tech_title"]
    if _title_matches(
        profile.get("current_title", ""), cfg["titles"]
    ) and retrieval_evidence < float(cfg["retrieval_evidence_threshold"]):
        return cfg["ceiling"]
    return None


def non_tech_industry_ceiling(
    profile: dict[str, Any],
    retrieval_evidence: float,
    config: dict[str, Any],
) -> str | None:
    cfg = config["non_tech_industry"]
    if _industry_matches(
        profile.get("current_industry", ""), cfg["industries"]
    ) and retrieval_evidence < float(cfg["retrieval_evidence_threshold"]):
        return cfg["ceiling"]
    return None


def services_only_ceiling(
    career_history: list[dict[str, Any]],
    has_product_company_applied_ml_context: float,
    config: dict[str, Any],
) -> str | None:
    cfg = config["services_only"]
    services = cfg["services_companies"]
    if not career_history:
        return None
    all_services = all(_company_in(r.get("company", ""), services) for r in career_history)
    if all_services and has_product_company_applied_ml_context < 0.5:
        return cfg["ceiling"]
    return None


def cv_speech_robotics_ceiling(
    career_history: list[dict[str, Any]],
    config: dict[str, Any],
) -> str | None:
    cfg = config["cv_speech_robotics_only"]
    niche_pats = _compile_patterns(tuple(cfg["niche_terms"]))
    absent_pats = _compile_patterns(tuple(cfg["required_absent_terms"]))
    found_niche = False
    found_absent = False
    for role in career_history or []:
        text = role.get("description", "") or ""
        if any(p.search(text) for p in niche_pats):
            found_niche = True
        if any(p.search(text) for p in absent_pats):
            found_absent = True
    if found_niche and not found_absent:
        return cfg["ceiling"]
    return None


def recent_only_langchain_ceiling(
    career_history: list[dict[str, Any]],
    reference: date,
    config: dict[str, Any],
) -> str | None:
    cfg = config["recent_only_langchain"]
    recent_window_months = int(cfg["recent_months_window"])
    lc_pats = _compile_patterns(tuple(cfg["langchain_terms"]))
    prior_pats = _compile_patterns(tuple(cfg["pre_llm_ml_terms"]))

    has_recent_lc = False
    has_prior_ml = False
    for role in career_history or []:
        end_str = role.get("end_date")
        months_ago = 0 if end_str is None else _months_since(end_str, reference)
        text = role.get("description", "") or ""
        if months_ago <= recent_window_months and any(p.search(text) for p in lc_pats):
            has_recent_lc = True
        if months_ago > recent_window_months and any(p.search(text) for p in prior_pats):
            has_prior_ml = True
    if has_recent_lc and not has_prior_ml:
        return cfg["ceiling"]
    return None


def inactive_architect_ceiling(
    profile: dict[str, Any],
    career_history: list[dict[str, Any]],
    reference: date,
    config: dict[str, Any],
) -> str | None:
    cfg = config["inactive_architect"]
    if not _title_matches(profile.get("current_title", ""), cfg["titles"]):
        return None
    window = int(cfg["inactive_months_window"])
    prod_pats = _compile_patterns(tuple(cfg["production_code_terms"]))
    has_recent_production = False
    for role in career_history or []:
        end = role.get("end_date")
        months_ago = _months_since(end, reference) if end else 0
        text = role.get("description", "") or ""
        if months_ago <= window and any(p.search(text) for p in prod_pats):
            has_recent_production = True
            break
    return None if has_recent_production else cfg["ceiling"]


def compute_anti_pattern_ceiling(
    candidate: dict[str, Any],
    *,
    retrieval_evidence: float,
    has_product_company_applied_ml_context: float,
    reference: date,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``{'ceiling': str | None, 'archetypes': list[str]}``."""
    cfg = config if config is not None else load_config("anti_patterns")
    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []

    archetypes: list[tuple[str, str | None]] = [
        ("non_tech_title", non_tech_title_ceiling(profile, retrieval_evidence, cfg)),
        ("non_tech_industry", non_tech_industry_ceiling(profile, retrieval_evidence, cfg)),
        (
            "services_only",
            services_only_ceiling(career, has_product_company_applied_ml_context, cfg),
        ),
        ("cv_speech_robotics_only", cv_speech_robotics_ceiling(career, cfg)),
        ("recent_only_langchain", recent_only_langchain_ceiling(career, reference, cfg)),
        ("inactive_architect", inactive_architect_ceiling(profile, career, reference, cfg)),
    ]
    fired = [name for name, c in archetypes if c is not None]
    ceiling = _strongest(*[c for _, c in archetypes])
    return {"ceiling": ceiling, "archetypes": fired}
