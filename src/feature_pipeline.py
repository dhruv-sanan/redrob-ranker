"""Compose all Phase-2 feature builders into a single row-by-row pipeline.

`build_feature_row(candidate, reference)` returns a flat dict — one row per
candidate. `build_features_df(records, reference)` materializes the DataFrame
that `build_features.py` writes to `artifacts/features.parquet`.

The dict is intentionally flat so it round-trips through Parquet without
list-of-struct gymnastics; nested outputs (`honeypot_risks`, archetype lists)
remain as Python lists, which pyarrow encodes as list columns.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from src.features.anti_patterns import compute_anti_pattern_ceiling
from src.features.behavioral import (
    availability_signal,
    contactability_signal,
    external_validation_signal,
    logistics_signal,
    market_interest_signal,
)
from src.features.education import education_signal
from src.features.evidence_channels import retrieval_evidence
from src.features.experience_band import experience_band_fit
from src.features.honeypot_ledger import honeypot_risk
from src.features.must_haves import compute_must_haves
from src.features.skill_trust import skill_depth_trust
from src.features.stuffer_risk import stuffer_risk
from src.features.tiering import TIER_PRIORITY, assign_tier
from src.features.title_career import title_career_fit


def build_feature_row(candidate: dict[str, Any], reference: date) -> dict[str, Any]:
    career = candidate.get("career_history", []) or []
    profile = candidate.get("profile", {}) or {}
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    scores = signals.get("skill_assessment_scores", {}) or {}
    education = candidate.get("education", []) or []

    retr_ev = retrieval_evidence(career, reference)
    must_h = compute_must_haves(candidate)
    tc_fit = title_career_fit(career, reference)
    skill_d = skill_depth_trust(skills, scores, career)
    avail = availability_signal(signals, reference)
    contact = contactability_signal(signals)
    market = market_interest_signal(signals)
    ext_val = external_validation_signal(signals)
    logist = logistics_signal(profile, signals)
    honeypot = honeypot_risk(candidate, reference)
    stuff = stuffer_risk(candidate, must_h["has_production_retrieval_evidence"])
    anti = compute_anti_pattern_ceiling(
        candidate,
        retrieval_evidence=retr_ev,
        has_product_company_applied_ml_context=must_h["has_product_company_applied_ml_context"],
        reference=reference,
    )
    exp_band = experience_band_fit(profile)
    edu = education_signal(education)
    tier = assign_tier(profile, must_h, stuff, honeypot["drop"])

    row: dict[str, Any] = {
        "candidate_id": candidate.get("candidate_id"),
        "retrieval_evidence": retr_ev,
        "title_career_fit": tc_fit,
        "skill_depth_trust": skill_d,
        "experience_band_fit": exp_band,
        "education_signal": edu,
        "availability_signal": avail,
        "contactability_signal": contact,
        "market_interest_signal": market,
        "external_validation_signal": ext_val,
        "logistics_multiplier": logist["multiplier"],
        "logistics_top_10_eligible": logist["top_10_eligible"],
        "honeypot_risk_score": honeypot["risk_score"],
        "honeypot_drop": honeypot["drop"],
        "honeypot_audit": honeypot["audit"],
        "honeypot_risks": honeypot["risks"],
        "stuffer_risk": stuff,
        "anti_pattern_ceiling": anti["ceiling"] or "none",
        "anti_pattern_archetypes": anti["archetypes"],
        "tier": tier,
        "tier_priority": TIER_PRIORITY[tier],
    }
    row.update(must_h)
    return row


def build_features_df(candidates: list[dict[str, Any]], reference: date) -> pd.DataFrame:
    return pd.DataFrame([build_feature_row(c, reference) for c in candidates])


FEATURE_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "retrieval_evidence",
    "title_career_fit",
    "skill_depth_trust",
    "experience_band_fit",
    "education_signal",
    "has_production_retrieval_evidence",
    "has_vector_or_hybrid_search_evidence",
    "has_python_backend_depth",
    "has_ranking_eval_evidence",
    "has_product_company_applied_ml_context",
    "has_shipper_signal",
    "availability_signal",
    "contactability_signal",
    "market_interest_signal",
    "external_validation_signal",
    "logistics_multiplier",
    "logistics_top_10_eligible",
    "honeypot_risk_score",
    "honeypot_drop",
    "honeypot_audit",
    "honeypot_risks",
    "stuffer_risk",
    "anti_pattern_ceiling",
    "anti_pattern_archetypes",
    "tier",
    "tier_priority",
)
