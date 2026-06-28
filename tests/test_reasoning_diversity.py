"""Reasoning-diversity regression gates.

Pair to the ``DEFAULT_TEMPLATE_REUSE_CAP`` + hash-based role-idx pick in
``src/reasoning.py``. The audit-CSV test guards top-25 template rotation;
the unit test guards the role-pick determinism + cross-candidate spread.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.reasoning import _primary_positive_snippet

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_CSV = REPO_ROOT / "top_100_audit.csv"

MAX_SINGLE_TEMPLATE_SHARE = 0.40
MIN_DISTINCT_TEMPLATES_TOP_25 = 4


@pytest.mark.skipif(
    not AUDIT_CSV.exists(), reason="top_100_audit.csv not present (run rank.py first)"
)
def test_top_25_reasoning_diversity() -> None:
    audit = pd.read_csv(AUDIT_CSV)
    top25 = audit[audit["rank"] <= 25]
    template_counts = top25["template_id"].value_counts()
    distinct = int(template_counts.index.nunique())
    max_share = template_counts.max() / len(top25)
    assert distinct >= MIN_DISTINCT_TEMPLATES_TOP_25, (
        f"top-25 has only {distinct} distinct template_id values "
        f"(floor {MIN_DISTINCT_TEMPLATES_TOP_25}); distribution={dict(template_counts)}"
    )
    assert max_share <= MAX_SINGLE_TEMPLATE_SHARE, (
        f"single template covers {max_share:.2%} of top-25 "
        f"(cap {MAX_SINGLE_TEMPLATE_SHARE:.0%}); distribution={dict(template_counts)}"
    )


def test_primary_snippet_hash_pick_is_deterministic_and_spreads() -> None:
    career = [
        {"company": "Stripe", "description": "Built a ranking pipeline for 50M users."},
        {"company": "Lyft", "description": "Owned embeddings retrieval at scale."},
        {"company": "Airbnb", "description": "Shipped a recommendation system end to end."},
    ]
    a1, _ = _primary_positive_snippet("CAND_0000001", career)
    a2, _ = _primary_positive_snippet("CAND_0000001", career)
    assert a1 == a2, "same candidate_id must pick the same role across calls"

    picks = {_primary_positive_snippet(f"CAND_{i:07d}", career)[0] for i in range(500)}
    assert (
        len(picks) >= 3
    ), f"500 distinct candidate_ids should spread across all 3 roles; got {len(picks)} unique snippets"


def test_primary_snippet_single_role_returns_that_role() -> None:
    career = [{"company": "Stripe", "description": "Built a ranking pipeline."}]
    snippet, employer = _primary_positive_snippet("CAND_X", career)
    assert "Built a ranking pipeline" in snippet
    assert employer == "Stripe"


def test_primary_snippet_skips_empty_descriptions() -> None:
    career = [
        {"company": "Filler1", "description": ""},
        {"company": "Real", "description": "Owned the ranking system end-to-end."},
        {"company": "Filler2", "description": None},
    ]
    snippet, employer = _primary_positive_snippet("CAND_X", career)
    assert "ranking system" in snippet
    assert employer == "Real"


def test_primary_snippet_empty_career_returns_empty() -> None:
    assert _primary_positive_snippet("CAND_X", []) == ("", "")
    assert _primary_positive_snippet("CAND_X", None) == ("", "")
