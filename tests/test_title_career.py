"""Tests for src/features/title_career.py — positive/negative title regex + recency."""

from __future__ import annotations

from src.features.title_career import title_career_fit
from src.reference_date import REFERENCE_DATE


def _role(title: str, start: str, end: str | None) -> dict:
    return {
        "title": title,
        "start_date": start,
        "end_date": end,
        "duration_months": 12,
        "is_current": end is None,
        "industry": "Internet",
        "company": "Anyco",
        "company_size": "201-500",
        "description": "",
    }


def test_title_career_empty_history() -> None:
    assert title_career_fit([], REFERENCE_DATE) == 0.0


def test_positive_titles_score_one() -> None:
    for title in [
        "Senior ML Engineer",
        "Founding AI Engineer",
        "Staff Search Engineer",
        "Backend Engineer",
        "NLP Engineer",
        "Data Scientist",
    ]:
        assert title_career_fit([_role(title, "2024-01-01", None)], REFERENCE_DATE) == 1.0


def test_negative_titles_score_zero() -> None:
    for title in [
        "Marketing Manager",
        "Sales Manager",
        "HR Manager",
        "Customer Support",
        "Accountant",
        "Operations Manager",
    ]:
        assert title_career_fit([_role(title, "2024-01-01", None)], REFERENCE_DATE) == 0.0


def test_neutral_titles_get_partial_credit() -> None:
    score = title_career_fit([_role("Project Coordinator", "2024-01-01", None)], REFERENCE_DATE)
    assert 0.3 < score < 0.5


def test_recency_weighting_favors_current_role() -> None:
    history = [
        _role("Senior ML Engineer", "2024-01-01", None),
        _role("Marketing Manager", "2010-01-01", "2015-01-01"),
    ]
    score = title_career_fit(history, REFERENCE_DATE)
    assert score > 0.5  # current dominates


def test_stuffer_fixture_score_zero(by_id: dict) -> None:
    cand = by_id["CAND_0000009"]
    assert title_career_fit(cand["career_history"], REFERENCE_DATE) == 0.0


def test_tier_a_fixture_score_one(by_id: dict) -> None:
    cand = by_id["CAND_0000001"]
    assert title_career_fit(cand["career_history"], REFERENCE_DATE) == 1.0
