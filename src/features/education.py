"""Education-tier signal per problem.md §1.11.

`education_signal ∈ [0, 1]` is the highest tier value across all education
entries, mapped via the YAML tier table (`weights.yaml.education_tiers`).
Missing or unknown tiers fall back to the `unknown` value.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

DEFAULT_TIERS: dict[str, float] = {
    "tier_1": 1.0,
    "tier_2": 0.75,
    "tier_3": 0.55,
    "tier_4": 0.4,
    "unknown": 0.5,
}


def education_signal(
    education: Iterable[dict[str, Any]] | None,
    *,
    tier_map: dict[str, float] | None = None,
) -> float:
    """Return the max education-tier value across `education` entries.

    No entries → returns the `unknown` value (0.5 by default), matching the
    "no education info" treatment in problem.md §1.11.
    """
    tiers = tier_map or DEFAULT_TIERS
    unknown_val = float(tiers.get("unknown", 0.5))
    if not education:
        return unknown_val
    best = -1.0
    for entry in education:
        tier_key = (entry.get("tier") or "unknown").strip()
        val = float(tiers.get(tier_key, unknown_val))
        if val > best:
            best = val
    if best < 0:
        return unknown_val
    return best
