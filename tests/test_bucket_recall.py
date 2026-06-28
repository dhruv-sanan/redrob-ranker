"""Bucket-recall regression gates on ``top_100_audit.csv``.

Codifies three of the eleven ``hld.md §3.5`` blocking checks as pytest so
any future weight tweak that breaks them fails CI loudly instead of being
caught only by the ad-hoc ``§3`` shell suite:

  #5 zero honeypot_drop in top-100
  #6 zero non_tech_industry archetype in top-50
  #11 every top-25 row has >=2 must-haves at relaxed bar OR an
      exception clause in its reasoning

The audit CSV is produced by ``rank.py`` at repo root and is gitignored,
so the tests skip cleanly when it is absent (cold clone before any
``rank.py`` run).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_CSV = REPO_ROOT / "top_100_audit.csv"
THRESHOLDS_YAML = REPO_ROOT / "config" / "thresholds.yaml"

MUST_HAVE_COLS = (
    "has_production_retrieval_evidence",
    "has_vector_or_hybrid_search_evidence",
    "has_python_backend_depth",
    "has_ranking_eval_evidence",
    "has_product_company_applied_ml_context",
    "has_shipper_signal",
)
MUST_HAVE_RELAXED_THRESHOLD = 0.3
EXCEPTION_TOKENS = ("exception", "despite", "even though", "atypical")


@cache
def _audit() -> pd.DataFrame:
    return pd.read_csv(AUDIT_CSV)


@cache
def _drop_threshold() -> float:
    return float(yaml.safe_load(THRESHOLDS_YAML.read_text())["honeypot"]["drop_threshold"])


pytestmark = pytest.mark.skipif(
    not AUDIT_CSV.exists(), reason="top_100_audit.csv not present (run rank.py first)"
)


def test_zero_honeypot_drop_in_top_100() -> None:
    audit = _audit()
    violators = audit[audit["honeypot_risk_score"] >= _drop_threshold()]
    assert len(violators) == 0, (
        f"{len(violators)} top-100 rows have honeypot_risk_score >= "
        f"{_drop_threshold()} (drop_threshold)"
    )


def test_zero_non_tech_industry_in_top_50() -> None:
    audit = _audit()
    top50 = audit[audit["rank"] <= 50]
    archs = top50["anti_pattern_archetypes"].fillna("").astype(str)
    hits = archs.str.contains("non_tech_industry", regex=False)
    assert hits.sum() == 0, (
        f"{hits.sum()} top-50 rows carry the non_tech_industry archetype " f"(hld.md §3.5 check #6)"
    )


def test_top_25_must_have_floor_or_exception_clause() -> None:
    audit = _audit()
    top25 = audit[audit["rank"] <= 25].copy()
    top25["must_have_count"] = (
        top25[list(MUST_HAVE_COLS)].ge(MUST_HAVE_RELAXED_THRESHOLD).sum(axis=1)
    )
    reasoning_lower = top25["reasoning"].fillna("").astype(str).str.lower()
    has_exception = reasoning_lower.apply(lambda s: any(tok in s for tok in EXCEPTION_TOKENS))
    violators = top25[(top25["must_have_count"] < 2) & ~has_exception]
    assert len(violators) == 0, (
        f"{len(violators)} top-25 rows have <2 must-haves at the relaxed "
        f"{MUST_HAVE_RELAXED_THRESHOLD} bar without an exception clause "
        f"(hld.md §3.5 check #11)"
    )
