"""Tests for src/features/education.py."""

from __future__ import annotations

from src.features.education import education_signal


def test_max_tier_across_entries() -> None:
    edu = [
        {"tier": "tier_3", "institution": "x", "degree": "BSc"},
        {"tier": "tier_1", "institution": "y", "degree": "MSc"},
        {"tier": "tier_2", "institution": "z", "degree": "PhD"},
    ]
    assert education_signal(edu) == 1.0  # tier_1 dominates


def test_empty_education_returns_unknown() -> None:
    assert education_signal([]) == 0.5
    assert education_signal(None) == 0.5


def test_unknown_tier_falls_back() -> None:
    assert education_signal([{"tier": "unknown"}]) == 0.5
    assert education_signal([{"tier": None}]) == 0.5
    assert education_signal([{}]) == 0.5


def test_each_tier_value() -> None:
    assert education_signal([{"tier": "tier_1"}]) == 1.0
    assert education_signal([{"tier": "tier_2"}]) == 0.75
    assert education_signal([{"tier": "tier_3"}]) == 0.55
    assert education_signal([{"tier": "tier_4"}]) == 0.4


def test_custom_tier_map_overrides_defaults() -> None:
    custom = {"tier_1": 0.9, "unknown": 0.1}
    assert education_signal([{"tier": "tier_1"}], tier_map=custom) == 0.9
    assert education_signal([{"tier": "tier_3"}], tier_map=custom) == 0.1
