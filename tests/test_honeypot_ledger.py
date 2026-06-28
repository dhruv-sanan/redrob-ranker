"""Tests for src/features/honeypot_ledger.py — interval flatten + 8 risk signals."""

from __future__ import annotations

from datetime import date

from src.features.honeypot_ledger import flatten_intervals, honeypot_risk
from src.reference_date import REFERENCE_DATE


def test_flatten_intervals_empty() -> None:
    assert flatten_intervals([]) == 0


def test_flatten_intervals_single() -> None:
    intervals = [(date(2020, 1, 1), date(2022, 1, 1))]
    assert flatten_intervals(intervals) == 24


def test_flatten_intervals_disjoint() -> None:
    intervals = [
        (date(2018, 1, 1), date(2020, 1, 1)),
        (date(2021, 1, 1), date(2023, 1, 1)),
    ]
    assert flatten_intervals(intervals) == 48


def test_flatten_intervals_fully_overlapping() -> None:
    """3 concurrent advisor/maintainer roles must merge to one span."""
    intervals = [
        (date(2023, 6, 1), date(2026, 6, 1)),
        (date(2023, 6, 1), date(2026, 6, 1)),
        (date(2024, 1, 1), date(2026, 6, 1)),
    ]
    months = flatten_intervals(intervals)
    assert 35 <= months <= 37


def test_flatten_intervals_partial_overlap() -> None:
    intervals = [
        (date(2020, 1, 1), date(2022, 6, 1)),
        (date(2022, 1, 1), date(2024, 1, 1)),
    ]
    months = flatten_intervals(intervals)
    # 2020-01 to 2024-01 = ~48 months
    assert 47 <= months <= 49


def test_future_start_honeypot_drops(by_id: dict) -> None:
    out = honeypot_risk(by_id["CAND_0000014"], REFERENCE_DATE)
    assert out["drop"] is True
    assert "impossible_future_start" in out["risks"]


def test_yoe_vs_span_honeypot_drops(by_id: dict) -> None:
    """CAND_15: claims 8 yrs but role duration_months=84 in a 12-mo role and span=12."""
    out = honeypot_risk(by_id["CAND_0000015"], REFERENCE_DATE)
    assert out["drop"] is True
    assert any("yoe_span" in r for r in out["risks"])
    assert any("role_duration_mismatch" in r for r in out["risks"])


def test_zero_duration_expert_honeypot_drops(by_id: dict) -> None:
    out = honeypot_risk(by_id["CAND_0000016"], REFERENCE_DATE)
    assert out["drop"] is True
    assert "zero_duration_expert" in out["risks"]


def test_tier_a_real_fit_does_not_drop(by_id: dict) -> None:
    for cid in ["CAND_0000001", "CAND_0000002", "CAND_0000003", "CAND_0000004", "CAND_0000005"]:
        out = honeypot_risk(by_id[cid], REFERENCE_DATE)
        assert out["drop"] is False, f"{cid} should not be flagged ({out['risks']})"
        assert out["risk_score"] < 0.65


def test_concurrent_advisor_does_not_drop(by_id: dict) -> None:
    """CAND_40: 3 overlapping roles must not be flagged by interval-flatten anomaly."""
    out = honeypot_risk(by_id["CAND_0000040"], REFERENCE_DATE)
    assert out["drop"] is False, f"concurrent advisor wrongly flagged: {out['risks']}"


def test_plain_language_tier5_does_not_drop(by_id: dict) -> None:
    for cid in ["CAND_0000006", "CAND_0000007", "CAND_0000008"]:
        out = honeypot_risk(by_id[cid], REFERENCE_DATE)
        assert out["drop"] is False, f"{cid} wrongly flagged: {out['risks']}"


def test_stuffers_do_not_drop_via_honeypot(by_id: dict) -> None:
    """Stuffer detection is stuffer_risk, not honeypot_drop. They should remain rankable."""
    for cid in ["CAND_0000009", "CAND_0000010"]:
        out = honeypot_risk(by_id[cid], REFERENCE_DATE)
        assert out["drop"] is False


def test_audit_threshold_lower_than_drop(by_id: dict) -> None:
    """Drop implies audit; audit alone does not imply drop."""
    out = honeypot_risk(by_id["CAND_0000016"], REFERENCE_DATE)
    assert out["audit"] is True
