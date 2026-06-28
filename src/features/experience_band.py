"""Gaussian experience-band fit per problem.md §1.11.

`experience_band_fit ∈ [0, 1]` peaks at `μ` years of experience with width `σ`.
Default `μ=7.0, σ=2.2` (slightly widened from the JD's 5–9 range so seniors
just outside that band are not collapsed to ~0 — see Codex #17).
"""

from __future__ import annotations

import math
from typing import Any

DEFAULT_MU = 7.0
DEFAULT_SIGMA = 2.2


def experience_band_fit(
    profile: dict[str, Any],
    *,
    mu: float = DEFAULT_MU,
    sigma: float = DEFAULT_SIGMA,
) -> float:
    """Return `exp(-((yoe - mu)^2) / (2 * sigma^2))`, clipped to `[0, 1]`."""
    yoe_raw = profile.get("years_of_experience")
    if yoe_raw is None:
        return 0.0
    try:
        yoe = float(yoe_raw)
    except (TypeError, ValueError):
        return 0.0
    if sigma <= 0:
        return 0.0
    val = math.exp(-((yoe - mu) ** 2) / (2.0 * sigma * sigma))
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val
