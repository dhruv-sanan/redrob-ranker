"""Title-career fit: recency-weighted regex over `title` field only.

Embeddings encode similarity, not negation (problem.md §4 Fix 4). Anti-title
detection uses fielded title strings + boolean rules, not vectors.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

POSITIVE_TITLE_PATTERNS = [
    re.compile(r"\b(ML|Machine Learning) Engineer\b", re.IGNORECASE),
    re.compile(r"\bData (Engineer|Scientist)\b", re.IGNORECASE),
    re.compile(r"\b(Senior |Staff |Principal |Lead )?Software Engineer\b", re.IGNORECASE),
    re.compile(r"\b(Senior |Staff |Principal |Lead )?Backend Engineer\b", re.IGNORECASE),
    re.compile(r"\bNLP Engineer\b", re.IGNORECASE),
    re.compile(r"\bSearch Engineer\b", re.IGNORECASE),
    re.compile(r"\bApplied (ML|Scientist|Engineer)\b", re.IGNORECASE),
    re.compile(r"\b(Founding )?AI Engineer\b", re.IGNORECASE),
    re.compile(r"\bLLM Engineer\b", re.IGNORECASE),
    re.compile(r"\bRanking Engineer\b", re.IGNORECASE),
    re.compile(r"\bResearch Engineer\b", re.IGNORECASE),
    re.compile(r"\bRecsys Engineer\b", re.IGNORECASE),
]

NEGATIVE_TITLE_PATTERNS = [
    re.compile(r"\bMarketing\b", re.IGNORECASE),
    re.compile(r"\bSales\b", re.IGNORECASE),
    re.compile(r"\bHR\b", re.IGNORECASE),
    re.compile(r"\bCustomer Support\b", re.IGNORECASE),
    re.compile(r"\bAccountant\b", re.IGNORECASE),
    re.compile(r"\bOperations Manager\b", re.IGNORECASE),
    re.compile(r"\bContent Writer\b", re.IGNORECASE),
    re.compile(r"\bBrand Manager\b", re.IGNORECASE),
    re.compile(r"\bCommunity Manager\b", re.IGNORECASE),
]


def _title_score(title: str) -> float:
    """Single title → fit in [0, 1]. Negative match dominates."""
    if not title:
        return 0.0
    for pat in NEGATIVE_TITLE_PATTERNS:
        if pat.search(title):
            return 0.0
    for pat in POSITIVE_TITLE_PATTERNS:
        if pat.search(title):
            return 1.0
    return 0.4  # neutral / adjacent


def _recency_weight(end_date_str: str | None, reference: date) -> float:
    if end_date_str is None:
        return 1.0
    try:
        end = date.fromisoformat(end_date_str)
    except (TypeError, ValueError):
        return 0.5
    years_ago = max(0.0, (reference - end).days / 365.25)
    return max(0.2, 1.0 - years_ago / 5.0)


def title_career_fit(career_history: list[dict[str, Any]], reference: date) -> float:
    """Recency-weighted mean of per-role title scores in [0, 1]."""
    if not career_history:
        return 0.0
    weighted = 0.0
    total_w = 0.0
    for role in career_history:
        w = _recency_weight(role.get("end_date"), reference)
        weighted += w * _title_score(role.get("title", ""))
        total_w += w
    return weighted / total_w if total_w > 0 else 0.0
