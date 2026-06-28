"""Honeypot risk ledger per problem.md §1.8.

Each candidate accumulates a list of risk signals (named strings) and a numeric
risk_score ∈ [0, 1]. Weights are loaded from `config/thresholds.yaml`.

Signals (8 cheap, deterministic — title/history semantic-distance check from
problem.md is deferred to Phase 4 when embeddings are available):
  * impossible_future_start          role start_date > REFERENCE_DATE
  * impossible_past_start            role start_date < min(REFERENCE_DATE - 50y, 1975)
  * severe_yoe_span_mismatch         yoe*12 > 4*span OR span > 4*yoe*12
  * huge_yoe_span_mismatch           yoe*12 > 3*span OR span > 3*yoe*12 (subsumed by severe)
  * role_duration_mismatch (per)     per-role |claimed - calc| > 6 months
  * zero_duration_expert             >=3 advanced/expert skills with duration_months=0
  * skill_count_anomaly              >=4 adv/exp AI skills + zero channel-X/Y hits + zero AI assess
  * education_chronology_anomaly     earliest end_year < earliest career start_year - 2
  * suspicious_perfect               >=8 skills + completeness<50 + (no linkedin OR no email)
"""

from __future__ import annotations

from datetime import date
from typing import Any

from src.config_loader import load_config
from src.features.evidence_channels import channel_hits_anywhere
from src.features.skill_trust import canonical_name

AI_CORE_SKILL_HINTS = {
    "nlp",
    "rag",
    "llm",
    "lora",
    "vector search",
    "embeddings",
    "bge",
    "pinecone",
    "weaviate",
    "qdrant",
    "information retrieval",
    "fine-tuning",
    "fine-tuning llms",
    "llm fine-tuning",
    "pytorch",
    "tensorflow",
    "transformers",
    "sentence-transformers",
}


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _months_between(start: date, end: date) -> int:
    if end < start:
        return 0
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
    return max(0, months)


def flatten_intervals(intervals: list[tuple[date, date]]) -> int:
    """Merge overlapping date intervals, return total months in the union.

    Concurrent roles (founder + advisor + OSS) merge into a single span — the
    sum-of-durations heuristic would punish them; this does not.
    """
    cleaned = [(s, e) for s, e in intervals if s and e and e >= s]
    if not cleaned:
        return 0
    cleaned.sort(key=lambda iv: iv[0])
    merged: list[list[date]] = [list(cleaned[0])]
    for s, e in cleaned[1:]:
        last = merged[-1]
        if s <= last[1]:
            last[1] = max(last[1], e)
        else:
            merged.append([s, e])
    return sum(_months_between(s, e) for s, e in merged)


def _career_intervals(
    career_history: list[dict[str, Any]], reference: date
) -> list[tuple[date, date]]:
    out: list[tuple[date, date]] = []
    for role in career_history or []:
        s = _parse_date(role.get("start_date"))
        e = _parse_date(role.get("end_date"))
        if s is None:
            continue
        if e is None:
            e = reference
        out.append((s, e))
    return out


def _yoe_span_signal(
    yoe_months: float, span_months: int, weights: dict[str, float]
) -> tuple[str | None, float]:
    if span_months <= 0 and yoe_months > 0:
        return "severe_yoe_span_mismatch", weights["severe_yoe_span_mismatch"]
    if span_months == 0:
        return None, 0.0
    if yoe_months > 4 * span_months or span_months > 4 * yoe_months:
        return "severe_yoe_span_mismatch", weights["severe_yoe_span_mismatch"]
    if yoe_months > 3 * span_months or span_months > 3 * yoe_months:
        return "huge_yoe_span_mismatch", weights["huge_yoe_span_mismatch"]
    return None, 0.0


def _role_duration_mismatch_count(career_history: list[dict[str, Any]], reference: date) -> int:
    count = 0
    for role in career_history or []:
        s = _parse_date(role.get("start_date"))
        e = _parse_date(role.get("end_date")) or reference
        if s is None or e < s:
            continue
        claimed = float(role.get("duration_months", 0) or 0)
        calc = _months_between(s, e)
        if abs(claimed - calc) > 6:
            count += 1
    return count


def _zero_duration_expert(skills: list[dict[str, Any]]) -> bool:
    n = 0
    for s in skills or []:
        prof = s.get("proficiency")
        if prof in {"advanced", "expert"} and float(s.get("duration_months", 0) or 0) == 0:
            n += 1
    return n >= 3


def _is_ai_skill(name: str, aliases: dict[str, str]) -> bool:
    canon = canonical_name(name, aliases).lower()
    return name.lower() in AI_CORE_SKILL_HINTS or canon in AI_CORE_SKILL_HINTS


def _skill_count_anomaly(
    skills: list[dict[str, Any]],
    assessment_scores: dict[str, float],
    career_history: list[dict[str, Any]],
    aliases: dict[str, str],
    channels_config: dict[str, Any] | None,
) -> bool:
    ai_adv = sum(
        1
        for s in skills or []
        if s.get("proficiency") in {"advanced", "expert"}
        and _is_ai_skill(s.get("name", ""), aliases)
    )
    if ai_adv < 4:
        return False
    hits = channel_hits_anywhere(career_history, channels_config)
    if hits["x"] + hits["y"] > 0:
        return False
    ai_assess_present = any(
        _is_ai_skill(k, aliases) and float(v or 0) > 0 for k, v in (assessment_scores or {}).items()
    )
    return not ai_assess_present


def _education_chronology_anomaly(
    education: list[dict[str, Any]], career_history: list[dict[str, Any]]
) -> bool:
    if not education or not career_history:
        return False
    earliest_career_year = min(
        (
            _parse_date(r.get("start_date")).year
            for r in career_history
            if _parse_date(r.get("start_date")) is not None
        ),
        default=None,
    )
    if earliest_career_year is None:
        return False
    earliest_end_year = min(
        (int(e.get("end_year", 0) or 0) for e in education),
        default=0,
    )
    if earliest_end_year == 0:
        return False
    return earliest_end_year < earliest_career_year - 2


def _suspicious_perfect(skills: list[dict[str, Any]], signals: dict[str, Any]) -> bool:
    n_skills = len(skills or [])
    completeness = float(signals.get("profile_completeness_score", 100) or 100)
    no_linkedin = not bool(signals.get("linkedin_connected"))
    no_email = not bool(signals.get("verified_email"))
    return n_skills >= 8 and completeness < 50 and (no_linkedin or no_email)


def honeypot_risk(
    candidate: dict[str, Any],
    reference: date,
    *,
    thresholds_config: dict[str, Any] | None = None,
    aliases_config: dict[str, str] | None = None,
    channels_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``{'risks': list[str], 'risk_score': float, 'drop': bool, 'audit': bool}``."""
    cfg = thresholds_config if thresholds_config is not None else load_config("thresholds")
    aliases = aliases_config if aliases_config is not None else load_config("aliases")
    hp = cfg["honeypot"]
    w: dict[str, float] = {k: float(v) for k, v in hp["weights"].items()}

    profile = candidate.get("profile", {}) or {}
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}
    education = candidate.get("education", []) or []
    yoe = float(profile.get("years_of_experience", 0.0) or 0.0)

    risks: list[str] = []
    score = 0.0

    # impossible future / past
    future_floor = reference
    past_floor = max(date(1975, 1, 1), date(reference.year - 50, reference.month, reference.day))
    saw_future = False
    saw_past = False
    for role in career:
        s = _parse_date(role.get("start_date"))
        if s is None:
            continue
        if s > future_floor:
            saw_future = True
        if s < past_floor:
            saw_past = True
    if saw_future:
        risks.append("impossible_future_start")
        score += w["impossible_future_start"]
    if saw_past:
        risks.append("impossible_past_start")
        score += w["impossible_past_start"]

    # yoe vs flattened span
    span_months = flatten_intervals(_career_intervals(career, reference))
    yoe_months = yoe * 12.0
    label, weight = _yoe_span_signal(yoe_months, span_months, w)
    if label:
        risks.append(label)
        score += weight

    # role duration mismatch
    mismatch_n = _role_duration_mismatch_count(career, reference)
    if mismatch_n > 0:
        risks.append(f"role_duration_mismatch[{mismatch_n}]")
        score += w["role_duration_mismatch_each"] * mismatch_n

    # zero-duration expert
    if _zero_duration_expert(skills):
        risks.append("zero_duration_expert")
        score += w["zero_duration_expert"]

    # skill_count_anomaly
    if _skill_count_anomaly(
        skills, signals.get("skill_assessment_scores", {}) or {}, career, aliases, channels_config
    ):
        risks.append("skill_count_anomaly")
        score += w["skill_count_anomaly"]

    # education chronology
    if _education_chronology_anomaly(education, career):
        risks.append("education_chronology_anomaly")
        score += w["education_chronology_anomaly"]

    # suspicious_perfect
    if _suspicious_perfect(skills, signals):
        risks.append("suspicious_perfect")
        score += w["suspicious_perfect"]

    risk_score = min(1.0, score)
    return {
        "risks": risks,
        "risk_score": risk_score,
        "drop": risk_score >= float(hp["drop_threshold"]),
        "audit": risk_score >= float(hp["audit_threshold"]),
    }
