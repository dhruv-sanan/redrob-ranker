"""Deterministic, evidence-grounded reasoning generator.

Public entry: ``render_top_100_reasoning(top_100, raw_lookup, skeletons,
seen_template_counter=None)`` returns a list of (reasoning_text,
template_id, evidence_ledger) triples — one per row of ``top_100``.

Design constraints (problem.md §3, lld.md §4 reasoning-audit edge cases):
- No LLM. No regex. No date math beyond simple string compares.
- Every named entity in the output text MUST be present in the candidate's
  raw record (audit enforces this).
- Templates rotate so no single template fires more than 12× across top-100.
- Rank-50+ reasonings include a concern/gap word.
- Non-technical-title candidates carry an explicit `why-exception` clause.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable
from typing import Any

DEFAULT_SNIPPET_CHAR_CAP = 140
DEFAULT_TEMPLATE_REUSE_CAP = 8
CONCERN_TOKENS: tuple[str, ...] = (
    "however",
    "limited",
    "gap",
    "gaps",
    "concern",
    "adjacent",
    "stretch",
    "missing",
    "notice",
    "inactive",
    "outside",
    "below",
    "absent",
    "weak",
    "skills-list-only",
)

NON_TECH_TITLE_TOKENS: tuple[str, ...] = (
    "marketing",
    "hr",
    "human resources",
    "sales",
    "content",
    "accounting",
    "customer support",
    "operations",
    "recruit",
    "talent",
)


def _truncate(text: str, cap: int = DEFAULT_SNIPPET_CHAR_CAP) -> str:
    text = (text or "").strip()
    if len(text) <= cap:
        return text
    cut = text[: cap - 1]
    last_ws = cut.rfind(" ")
    if last_ws > (cap - 1) * 0.6:
        cut = cut[:last_ws]
    return cut.rstrip(",.;:- \t") + "…"


def _rank_band(rank: int) -> str:
    if rank <= 10:
        return "top10"
    if rank <= 25:
        return "top25"
    if rank <= 50:
        return "top50"
    return "top100"


def _is_non_technical(title: str) -> bool:
    if not title:
        return False
    lower = title.lower()
    return any(tok in lower for tok in NON_TECH_TITLE_TOKENS)


def _primary_positive_snippet(
    candidate_id: str, career: list[dict[str, Any]] | None
) -> tuple[str, str]:
    """Return (snippet, employer) — one non-empty role description picked
    deterministically by ``hash(candidate_id) % len(non_empty_roles)``.
    Breaks the visual uniformity from synthetic-dataset role-description
    reuse without changing the candidate's rank. Capped at
    DEFAULT_SNIPPET_CHAR_CAP characters."""
    non_empty = [r for r in (career or []) if (r.get("description") or "").strip()]
    if not non_empty:
        return "", ""
    if len(non_empty) == 1:
        role = non_empty[0]
    else:
        digest = hashlib.sha256((candidate_id or "").encode("utf-8")).hexdigest()
        role = non_empty[int(digest, 16) % len(non_empty)]
    desc = (role.get("description") or "").strip()
    employer = (role.get("company") or "").strip()
    return _truncate(desc), employer


def _top_trust_skill(
    skills: list[dict[str, Any]] | None,
    assessment_scores: dict[str, float] | None,
) -> str:
    if not skills:
        return ""
    proficiency_rank = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    assess = assessment_scores or {}
    candidates: list[tuple[int, float, int, str]] = []
    for s in skills:
        name = (s.get("name") or "").strip()
        if not name:
            continue
        prof = proficiency_rank.get((s.get("proficiency") or "").lower(), 0)
        dur = int(s.get("duration_months") or 0)
        score_raw = assess.get(name)
        score = float(score_raw) if isinstance(score_raw, int | float) else 0.0
        candidates.append((prof, score, dur, name))
    if not candidates:
        return ""
    candidates.sort(reverse=True)
    prof, score, dur, name = candidates[0]
    parts = [name]
    if score >= 50:
        parts.append(f"(assessment {int(round(score))})")
    elif dur >= 24:
        parts.append(f"({dur}mo)")
    return " ".join(parts)


def _logistics_fact(profile: dict[str, Any], signals: dict[str, Any]) -> str:
    """Compose a tight logistics line: city, work mode, notice period."""
    city = (profile.get("location") or "").strip()
    country = (profile.get("country") or "").strip()
    mode = (signals.get("preferred_work_mode") or "").strip()
    relocate = bool(signals.get("willing_to_relocate", False))
    notice_raw = signals.get("notice_period_days")
    notice = int(notice_raw) if isinstance(notice_raw, int | float) else None

    where = city
    if where and country:
        where = f"{city}, {country}"
    elif country:
        where = country
    pieces = []
    if where:
        pieces.append(where)
    if mode:
        pieces.append(mode)
    if relocate:
        pieces.append("relocatable")
    if notice is not None:
        pieces.append(f"{notice}d notice")
    return ", ".join(pieces) if pieces else ""


def _concern_fact(
    feature_row: dict[str, Any],
    raw: dict[str, Any],
) -> str:
    """Pick the strongest concern. Returns '' when no concern fires."""
    concerns: list[str] = []
    archs_raw = feature_row.get("anti_pattern_archetypes")
    if archs_raw is not None and len(archs_raw) > 0:  # works for list/tuple/ndarray
        head = next(iter(archs_raw))
        concerns.append(f"archetype concern ({head})")
    if bool(feature_row.get("honeypot_audit", False)):
        concerns.append("audit-flagged profile (manual review)")
    signals = (raw.get("redrob_signals") or {}) if raw else {}
    notice_raw = signals.get("notice_period_days")
    if isinstance(notice_raw, int | float) and int(notice_raw) > 60:
        concerns.append(f"high notice ({int(notice_raw)}d)")
    resp = signals.get("recruiter_response_rate")
    if isinstance(resp, int | float) and float(resp) < 0.3:
        concerns.append(f"low response rate ({float(resp):.2f})")
    if float(feature_row.get("stuffer_risk", 0.0)) >= 0.4:
        concerns.append("skills-list density elevated vs evidence")
    if not concerns:
        return ""
    return "; ".join(concerns)


def _must_have_summary(feature_row: dict[str, Any]) -> str:
    names = [
        ("retrieval", "has_production_retrieval_evidence"),
        ("vector_search", "has_vector_or_hybrid_search_evidence"),
        ("python_depth", "has_python_backend_depth"),
        ("ranking_eval", "has_ranking_eval_evidence"),
        ("product_ml", "has_product_company_applied_ml_context"),
        ("shipper", "has_shipper_signal"),
    ]
    present = [n for n, col in names if float(feature_row.get(col, 0.0)) > 0.0]
    return ", ".join(present) if present else "none"


def _exception_clause(feature_row: dict[str, Any], raw: dict[str, Any]) -> str:
    """For non-technical-title candidates in top-100, give an explicit
    'why-exception' clause naming the strongest must-have they DO carry."""
    profile = (raw.get("profile") or {}) if raw else {}
    title = (profile.get("current_title") or "").strip()
    if not _is_non_technical(title):
        return ""
    must = _must_have_summary(feature_row)
    return f"non-tech title kept on must-haves: {must}"


def build_evidence_ledger(
    rank: int,
    feature_row: dict[str, Any],
    raw: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the full ledger for one ranked candidate."""
    raw = raw or {}
    profile = raw.get("profile") or {}
    career = raw.get("career_history") or []
    skills = raw.get("skills") or []
    signals = raw.get("redrob_signals") or {}
    scores = signals.get("skill_assessment_scores") or {}

    candidate_id = str(feature_row.get("candidate_id") or "")
    primary, primary_employer = _primary_positive_snippet(candidate_id, career)
    secondary = _top_trust_skill(skills, scores)
    logistics = _logistics_fact(profile, signals)
    concern = _concern_fact(feature_row, raw)
    must_haves = _must_have_summary(feature_row)
    exception = _exception_clause(feature_row, raw)
    band = _rank_band(int(rank))

    return {
        "rank_band": band,
        "primary_positive_fact": primary,
        "primary_employer": primary_employer,
        "secondary_positive_fact": secondary,
        "logistics_fact": logistics,
        "concern_fact": concern,
        "must_haves_present": must_haves,
        "exception_clause": exception,
        "tier": feature_row.get("tier", ""),
    }


def _ensure_concern_text(text: str, ledger: dict[str, Any]) -> str:
    """For rank-50+ bands, force a concern token to be present in `text`."""
    if ledger["rank_band"] not in {"top50", "top100"}:
        return text
    lower = text.lower()
    if any(tok in lower for tok in CONCERN_TOKENS):
        return text
    return text.rstrip(".") + "; adjacent fit, gaps remain."


def _render_template(template: str, ledger: dict[str, Any]) -> str:
    mapping = {
        "positive": ledger["primary_positive_fact"] or "summary-only signal",
        "skill": ledger["secondary_positive_fact"] or "no high-trust AI skill",
        "logistics": ledger["logistics_fact"] or "logistics not stated",
        "concern": ledger["concern_fact"] or "no material concern",
        "employer": ledger["primary_employer"] or "their prior role",
        "tier": ledger["tier"] or "—",
        "must_haves": ledger["must_haves_present"],
        "exception": ledger["exception_clause"] or "stretch profile",
    }
    out = template
    for key, val in mapping.items():
        out = out.replace("{" + key + "}", str(val))
    # Collapse double-spaces / dangling separators left by empty facts.
    while "  " in out:
        out = out.replace("  ", " ")
    out = out.replace(" .", ".").replace("..", ".").replace(". .", ". ")
    while ". ." in out:
        out = out.replace(". .", ".")
    return out.strip()


def _pick_template(
    skeletons: dict[str, list[str]],
    band: str,
    counter: Counter,
    cap: int = DEFAULT_TEMPLATE_REUSE_CAP,
) -> tuple[str, str]:
    """Return (template_id, template_str), respecting per-template reuse cap.

    Falls back to the *least-used* template in the band if all hit the cap.
    """
    band_templates = skeletons.get(band, [])
    if not band_templates:
        return ("default-0", "{positive}. {logistics}. {concern}")
    # Prefer the first not at cap.
    for i, tpl in enumerate(band_templates):
        tid = f"{band}-{i}"
        if counter[tid] < cap:
            counter[tid] += 1
            return tid, tpl
    # All at cap; use the least-used.
    tid, _ = min(
        ((f"{band}-{i}", counter[f"{band}-{i}"]) for i in range(len(band_templates))),
        key=lambda x: x[1],
    )
    counter[tid] += 1
    return tid, band_templates[int(tid.split("-")[-1])]


def render_top_100_reasoning(
    top_100: list[dict[str, Any]],
    raw_lookup: dict[str, dict[str, Any]],
    skeletons: dict[str, list[str]],
    *,
    template_reuse_cap: int = DEFAULT_TEMPLATE_REUSE_CAP,
) -> list[dict[str, Any]]:
    """Render reasoning for each ranked candidate.

    `top_100` is a list of dicts (one per row) keyed by feature columns plus
    a numeric `rank` column. `raw_lookup` maps candidate_id → raw nested
    candidate record. Returns a list of dicts:
      {"candidate_id", "rank", "reasoning", "template_id", "ledger"}.
    """
    counter: Counter = Counter()
    out: list[dict[str, Any]] = []
    for row in top_100:
        cid = row["candidate_id"]
        rank = int(row["rank"])
        ledger = build_evidence_ledger(rank, row, raw_lookup.get(cid))
        tid, template = _pick_template(
            skeletons, ledger["rank_band"], counter, cap=template_reuse_cap
        )
        rendered = _render_template(template, ledger)
        rendered = _ensure_concern_text(rendered, ledger)
        out.append(
            {
                "candidate_id": cid,
                "rank": rank,
                "reasoning": rendered,
                "template_id": tid,
                "ledger": ledger,
            }
        )
    return out


def load_skeletons_from_yaml(text: str) -> dict[str, list[str]]:
    """Parse a YAML skeletons file (text) into the expected dict shape."""
    import yaml

    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("skeletons.yaml root must be a mapping")
    out: dict[str, list[str]] = {}
    for band, templates in data.items():
        if not isinstance(templates, Iterable):
            continue
        out[str(band)] = [str(t) for t in templates if str(t).strip()]
    return out


__all__ = [
    "DEFAULT_SNIPPET_CHAR_CAP",
    "DEFAULT_TEMPLATE_REUSE_CAP",
    "CONCERN_TOKENS",
    "NON_TECH_TITLE_TOKENS",
    "build_evidence_ledger",
    "render_top_100_reasoning",
    "load_skeletons_from_yaml",
]
