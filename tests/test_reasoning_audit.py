"""Tests for reasoning_audit.py — failure modes and pass path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import reasoning_audit as ra


def _raw() -> dict:
    return {
        "profile": {"current_title": "Senior ML Engineer", "location": "Pune", "country": "India"},
        "career_history": [
            {
                "company": "Stripe",
                "description": "Built and owned production ranking and recsys at scale with BGE embeddings.",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 50, "endorsements": 50},
        ],
        "redrob_signals": {
            "notice_period_days": 30,
            "recruiter_response_rate": 0.6,
            "skill_assessment_scores": {"Python": 92},
        },
    }


def _row(**overrides) -> dict:
    base = {
        "rank": 1,
        "candidate_id": "CAND_0000001",
        "reasoning": "Production retrieval ownership: Built and owned production ranking and recsys at scale. Python (assessment 92). Pune, India, hybrid. no material concern",
        "template_id": "top10-0",
    }
    base.update(overrides)
    return base


def test_passes_well_formed_reasoning() -> None:
    counts: dict[str, int] = {}
    failures = ra.audit_row(_row(), _raw(), template_counts=counts)
    assert failures == [], failures


def test_fails_when_rank50_missing_concern() -> None:
    counts: dict[str, int] = {}
    row = _row(rank=60, reasoning="Solid fit. Built ranking.")
    failures = ra.audit_row(row, _raw(), template_counts=counts)
    assert any("concern" in f for f in failures)


def test_fails_when_template_overused() -> None:
    counts: dict[str, int] = {"top10-0": ra.DEFAULT_TEMPLATE_REUSE_CAP}
    failures = ra.audit_row(_row(), _raw(), template_counts=counts)
    assert any("reused" in f for f in failures)


def test_fails_non_tech_without_exception() -> None:
    raw = _raw()
    raw["profile"]["current_title"] = "Marketing Manager"
    row = _row(
        rank=80,
        reasoning="Strong fit. Built things. Pune, India. all good.",
        template_id="top100-0",
    )
    counts: dict[str, int] = {}
    failures = ra.audit_row(row, raw, template_counts=counts)
    assert any("non-tech" in f for f in failures)


def test_high_notice_must_be_mentioned_top50() -> None:
    raw = _raw()
    raw["redrob_signals"]["notice_period_days"] = 90
    row = _row(rank=25, reasoning="Strong fit. Python (assessment 92). Pune, India, hybrid.")
    counts: dict[str, int] = {}
    failures = ra.audit_row(row, raw, template_counts=counts)
    assert any("notice" in f for f in failures)


def test_run_audit_pass(tmp_path: Path) -> None:
    audit_csv = tmp_path / "audit.csv"
    candidates = tmp_path / "candidates.parquet"
    pd.DataFrame([_row()]).to_csv(audit_csv, index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "CAND_0000001",
                "profile": _raw()["profile"],
                "career_history": _raw()["career_history"],
                "skills": _raw()["skills"],
                "education": [],
                "redrob_signals": _raw()["redrob_signals"],
                "certifications": [],
                "languages": [],
            }
        ]
    ).to_parquet(candidates)
    rc = ra.run_audit(audit_csv, candidates, tmp_path / "audit_out.csv")
    assert rc == 0
