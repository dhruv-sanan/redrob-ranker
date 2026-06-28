"""Tests for src/features/behavioral.py — 5 split components."""

from __future__ import annotations

from src.features.behavioral import (
    availability_signal,
    contactability_signal,
    external_validation_signal,
    logistics_signal,
    market_interest_signal,
)
from src.reference_date import REFERENCE_DATE


def test_availability_high_for_tier_a(by_id: dict) -> None:
    cand = by_id["CAND_0000001"]
    s = availability_signal(cand["redrob_signals"], REFERENCE_DATE)
    assert 0.55 <= s <= 1.05
    assert s > 0.9  # recent active, high response, short notice


def test_availability_lower_for_high_notice_low_response(by_id: dict) -> None:
    high = availability_signal(by_id["CAND_0000001"]["redrob_signals"], REFERENCE_DATE)
    low = availability_signal(by_id["CAND_0000038"]["redrob_signals"], REFERENCE_DATE)
    assert low < high


def test_availability_in_range_for_ghost(by_id: dict) -> None:
    cand = by_id["CAND_0000039"]
    s = availability_signal(cand["redrob_signals"], REFERENCE_DATE)
    assert 0.55 <= s <= 1.05


def test_contactability_pass_default(by_id: dict) -> None:
    assert contactability_signal(by_id["CAND_0000001"]["redrob_signals"]) is True


def test_contactability_fails_when_only_one_verified() -> None:
    signals = {"verified_email": True, "verified_phone": False, "linkedin_connected": False}
    assert contactability_signal(signals) is False


def test_contactability_passes_when_two_verified() -> None:
    signals = {"verified_email": True, "verified_phone": True, "linkedin_connected": False}
    assert contactability_signal(signals) is True


def test_market_interest_in_range(by_id: dict) -> None:
    s = market_interest_signal(by_id["CAND_0000001"]["redrob_signals"])
    assert 0.0 <= s <= 1.0


def test_external_validation_capped(by_id: dict) -> None:
    s = external_validation_signal(by_id["CAND_0000001"]["redrob_signals"])
    assert 0.0 <= s <= 0.05


def test_external_validation_zero_for_no_github_or_assess() -> None:
    signals = {
        "github_activity_score": -1,
        "endorsements_received": 0,
        "skill_assessment_scores": {},
    }
    s = external_validation_signal(signals)
    assert s == 0.0


def test_logistics_india_acceptable_city_top_10_eligible(by_id: dict) -> None:
    out = logistics_signal(
        by_id["CAND_0000001"]["profile"], by_id["CAND_0000001"]["redrob_signals"]
    )
    assert out["multiplier"] == 1.0
    assert out["top_10_eligible"] is True


def test_logistics_outside_india_no_relocate_not_top_10(by_id: dict) -> None:
    cand = by_id["CAND_0000032"]  # Germany, willing_to_relocate=False, prefer_work_mode=remote
    out = logistics_signal(cand["profile"], cand["redrob_signals"])
    assert out["top_10_eligible"] is False
    assert out["multiplier"] < 1.0
