"""Tests for src/features/experience_band.py."""

from __future__ import annotations

import math

from src.features.experience_band import experience_band_fit


def test_peaks_at_mu() -> None:
    assert experience_band_fit({"years_of_experience": 7.0}) == 1.0


def test_drops_outside_band() -> None:
    near = experience_band_fit({"years_of_experience": 5.0})
    far = experience_band_fit({"years_of_experience": 0.0})
    assert near > far
    assert near > 0.3  # 2σ ≈ within band edge
    assert far < 0.1


def test_clipped_above_one_impossible_but_safe() -> None:
    val = experience_band_fit({"years_of_experience": 7.0}, mu=7.0, sigma=10.0)
    assert val <= 1.0


def test_missing_yoe_returns_zero() -> None:
    assert experience_band_fit({}) == 0.0


def test_non_numeric_yoe_returns_zero() -> None:
    assert experience_band_fit({"years_of_experience": "junior"}) == 0.0


def test_decay_symmetry() -> None:
    lo = experience_band_fit({"years_of_experience": 5.0})
    hi = experience_band_fit({"years_of_experience": 9.0})
    assert math.isclose(lo, hi, abs_tol=1e-9)
