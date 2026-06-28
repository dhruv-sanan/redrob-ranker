"""3-channel evidence detection per problem.md §1.5.

X = exact technical, Y = plain-language product, Z = ownership.
Per-role raw evidence: alpha*X + beta*Y + min(1, X+Y) * z_bonus.
Career-level evidence: recency-weighted mean across roles, clipped to [0, 1].

Recency weight: current role = 1.0, ended N years ago = linearly decays to 0.3 floor.
"""

from __future__ import annotations

import re
from datetime import date
from functools import cache
from typing import Any

from src.config_loader import load_config


@cache
def _compile_patterns(terms: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile case-insensitive word-boundary patterns for a list of phrases."""
    return tuple(re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in terms)


def _hit_count(text: str, patterns: tuple[re.Pattern[str], ...]) -> int:
    return sum(1 for p in patterns if p.search(text))


def _channels_from_config(config: dict[str, Any] | None) -> dict[str, Any]:
    return config if config is not None else load_config("regex_channels")


def role_evidence(description: str, config: dict[str, Any] | None = None) -> dict[str, float]:
    """Score a single role description across the 3 channels.

    Returns: {"x": float, "y": float, "z": float, "raw": float} all in [0, 1].
    """
    cfg = _channels_from_config(config)
    text = description or ""
    x = _hit_count(text, _compile_patterns(tuple(cfg["exact_technical"])))
    y = _hit_count(text, _compile_patterns(tuple(cfg["plain_language_product"])))
    z = _hit_count(text, _compile_patterns(tuple(cfg["ownership"])))

    # Saturate each channel at 3 hits.
    x_norm = min(1.0, x / 3.0)
    y_norm = min(1.0, y / 3.0)
    z_norm = min(1.0, z / 3.0)

    alpha = float(cfg["alpha"])
    beta = float(cfg["beta"])
    z_cap = float(cfg["z_bonus_cap"])
    raw_base = alpha * x_norm + beta * y_norm
    z_bonus = z_cap * z_norm * min(1.0, x_norm + y_norm)
    raw = min(1.0, raw_base + z_bonus)
    return {"x": x_norm, "y": y_norm, "z": z_norm, "raw": raw}


def _recency_weight(end_date_str: str | None, reference: date) -> float:
    """Linear decay from 1.0 (current) to 0.3 floor across 5 years since role end."""
    if end_date_str is None:
        return 1.0
    try:
        end = date.fromisoformat(end_date_str)
    except (TypeError, ValueError):
        return 0.5
    years_ago = max(0.0, (reference - end).days / 365.25)
    return max(0.3, 1.0 - years_ago / 5.0)


def retrieval_evidence(
    career_history: list[dict[str, Any]],
    reference: date,
    config: dict[str, Any] | None = None,
) -> float:
    """Career-level retrieval evidence in [0, 1]. Recency-weighted mean over roles."""
    if not career_history:
        return 0.0
    weighted = 0.0
    total_weight = 0.0
    for role in career_history:
        w = _recency_weight(role.get("end_date"), reference)
        ev = role_evidence(role.get("description", ""), config)["raw"]
        weighted += w * ev
        total_weight += w
    return weighted / total_weight if total_weight > 0 else 0.0


def channel_hits_anywhere(
    career_history: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Total hit counts across all roles for each channel. Used by stuffer/honeypot logic."""
    cfg = _channels_from_config(config)
    x_pat = _compile_patterns(tuple(cfg["exact_technical"]))
    y_pat = _compile_patterns(tuple(cfg["plain_language_product"]))
    z_pat = _compile_patterns(tuple(cfg["ownership"]))
    x_total = 0
    y_total = 0
    z_total = 0
    for role in career_history or []:
        text = role.get("description", "") or ""
        x_total += _hit_count(text, x_pat)
        y_total += _hit_count(text, y_pat)
        z_total += _hit_count(text, z_pat)
    return {"x": x_total, "y": y_total, "z": z_total}
