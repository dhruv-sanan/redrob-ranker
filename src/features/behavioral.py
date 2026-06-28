"""5 behavioral signals split out of the monolithic multiplier per problem.md §1.10.

availability  — multiplier ∈ [0.55, 1.05].
contactability — boolean (top-10 gate, not a multiplier).
market_interest — tiebreaker ∈ [0, 1].
external_validation — additive boost ∈ [0, 0.05].
logistics — multiplier ∈ [0.6, 1.0] + boolean top-10 gate.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

ACCEPTABLE_INDIA_CITIES = {
    "Bangalore",
    "Bengaluru",
    "Pune",
    "Noida",
    "Hyderabad",
    "Mumbai",
    "Delhi",
    "Gurgaon",
    "Gurugram",
    "Chennai",
}


def _days_since(date_str: str | None, reference: date) -> float:
    if not date_str:
        return 365.0
    try:
        d = date.fromisoformat(date_str)
    except (TypeError, ValueError):
        return 365.0
    return max(0.0, (reference - d).days)


def availability_signal(signals: dict[str, Any], reference: date) -> float:
    """Multiplier ∈ [0.55, 1.05]."""
    decay = math.exp(-_days_since(signals.get("last_active_date"), reference) / 90.0)
    open_flag = 1.0 if signals.get("open_to_work_flag") else 0.4
    response_rate = float(signals.get("recruiter_response_rate", 0.0) or 0.0)
    response_time_hours = float(signals.get("avg_response_time_hours", 48.0) or 48.0)
    response_time_score = max(0.0, 1.0 - response_time_hours / 48.0)
    notice = int(signals.get("notice_period_days", 90) or 90)
    notice_score = 1.0 if notice <= 30 else (0.7 if notice <= 60 else 0.4)
    interview_rate = float(signals.get("interview_completion_rate", 0.5) or 0.5)
    raw = 0.30 * decay + 0.15 * open_flag + 0.20 * response_rate + 0.10 * response_time_score
    raw += 0.15 * notice_score + 0.10 * interview_rate
    return 0.55 + 0.50 * raw


def contactability_signal(signals: dict[str, Any]) -> bool:
    """Pass iff at least 2 of {verified_email, verified_phone, linkedin_connected}."""
    flags = [
        bool(signals.get("verified_email")),
        bool(signals.get("verified_phone")),
        bool(signals.get("linkedin_connected")),
    ]
    return sum(flags) >= 2


def market_interest_signal(signals: dict[str, Any]) -> float:
    """Tiebreaker in [0, 1]."""
    saved = math.log1p(int(signals.get("saved_by_recruiters_30d", 0) or 0))
    views = math.log1p(int(signals.get("profile_views_received_30d", 0) or 0))
    appearances = math.log1p(int(signals.get("search_appearance_30d", 0) or 0))
    raw = (saved + views + appearances) / 3.0
    return min(1.0, raw / 5.0)


def external_validation_signal(signals: dict[str, Any]) -> float:
    """Additive boost ∈ [0, 0.05]. Rewards OSS/external validation per JD."""
    gh = float(signals.get("github_activity_score", -1.0) or -1.0)
    gh_norm = max(0.0, gh) / 100.0
    endorsements = math.log1p(int(signals.get("endorsements_received", 0) or 0)) / 10.0
    assess_n = len(signals.get("skill_assessment_scores", {}) or {})
    assess_norm = min(1.0, assess_n / 10.0)
    raw = (gh_norm + endorsements + assess_norm) / 3.0
    return min(0.05, 0.05 * raw)


def logistics_signal(
    profile: dict[str, Any],
    signals: dict[str, Any],
    *,
    senior_max_lpa: float = 80.0,
) -> dict[str, Any]:
    """Multiplier ∈ [0.6, 1.0] + boolean top-10 gate."""
    country = (profile.get("country") or "").strip()
    location = (profile.get("location") or "").strip()
    city = location.split(",")[0].strip() if location else ""
    willing_relocate = bool(signals.get("willing_to_relocate"))
    work_mode = signals.get("preferred_work_mode", "")
    salary_max = float(signals.get("expected_salary_range_inr_lpa", {}).get("max", 0.0) or 0.0)

    in_india_acceptable_city = country == "India" and city in ACCEPTABLE_INDIA_CITIES
    can_join_india = in_india_acceptable_city or willing_relocate
    remote_ok = work_mode in {"remote", "flexible"}

    if can_join_india:
        mult = 1.0
    elif remote_ok:
        mult = 0.8
    else:
        mult = 0.6

    if salary_max > senior_max_lpa:
        mult *= 0.9

    top_10_eligible = can_join_india
    return {"multiplier": mult, "top_10_eligible": top_10_eligible}
