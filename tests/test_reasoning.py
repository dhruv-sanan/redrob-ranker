"""Tests for src/reasoning.py — ledger + template selection + concern guard."""

from __future__ import annotations

from collections import Counter

from src.reasoning import (
    CONCERN_TOKENS,
    DEFAULT_TEMPLATE_REUSE_CAP,
    _pick_template,
    _rank_band,
    build_evidence_ledger,
    load_skeletons_from_yaml,
    render_top_100_reasoning,
)


def _feature_row(**overrides) -> dict:
    base = {
        "candidate_id": "CAND_0000001",
        "tier": "A",
        "stuffer_risk": 0.0,
        "honeypot_audit": False,
        "has_production_retrieval_evidence": 1.0,
        "has_vector_or_hybrid_search_evidence": 1.0,
        "has_python_backend_depth": 1.0,
        "has_ranking_eval_evidence": 0.5,
        "has_product_company_applied_ml_context": 1.0,
        "has_shipper_signal": 0.5,
        "anti_pattern_archetypes": [],
    }
    base.update(overrides)
    return base


def _raw(**overrides) -> dict:
    base = {
        "profile": {
            "current_title": "Senior ML Engineer",
            "location": "Bengaluru, KA",
            "country": "India",
        },
        "career_history": [
            {
                "company": "Stripe",
                "description": "Built and shipped a candidate-job matching system "
                "using BGE embeddings and Pinecone hybrid search across 30M docs.",
            }
        ],
        "skills": [
            {"name": "Python", "proficiency": "expert", "duration_months": 60, "endorsements": 50},
        ],
        "redrob_signals": {
            "skill_assessment_scores": {"Python": 92},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "notice_period_days": 30,
            "recruiter_response_rate": 0.6,
        },
    }
    base.update(overrides)
    return base


def test_rank_band_buckets() -> None:
    assert _rank_band(1) == "top10"
    assert _rank_band(10) == "top10"
    assert _rank_band(11) == "top25"
    assert _rank_band(25) == "top25"
    assert _rank_band(26) == "top50"
    assert _rank_band(50) == "top50"
    assert _rank_band(51) == "top100"
    assert _rank_band(100) == "top100"


def test_ledger_populates_primary_snippet_employer() -> None:
    ledger = build_evidence_ledger(1, _feature_row(), _raw())
    assert "candidate-job matching system" in ledger["primary_positive_fact"]
    assert ledger["primary_employer"] == "Stripe"


def test_ledger_picks_high_assessment_skill() -> None:
    ledger = build_evidence_ledger(1, _feature_row(), _raw())
    assert "Python" in ledger["secondary_positive_fact"]
    assert "(assessment 92)" in ledger["secondary_positive_fact"]


def test_ledger_concern_for_anti_pattern() -> None:
    raw = _raw()
    feature = _feature_row(anti_pattern_archetypes=["services_only_ceiling"])
    ledger = build_evidence_ledger(60, feature, raw)
    assert "archetype concern" in ledger["concern_fact"]
    assert "services_only_ceiling" in ledger["concern_fact"]


def test_ledger_concern_for_high_notice_and_low_response() -> None:
    raw = _raw()
    raw["redrob_signals"]["notice_period_days"] = 75
    raw["redrob_signals"]["recruiter_response_rate"] = 0.15
    ledger = build_evidence_ledger(20, _feature_row(), raw)
    assert "high notice" in ledger["concern_fact"]
    assert "low response rate" in ledger["concern_fact"]


def test_ledger_non_tech_exception_clause() -> None:
    raw = _raw()
    raw["profile"]["current_title"] = "Marketing Manager"
    ledger = build_evidence_ledger(80, _feature_row(), raw)
    assert "non-tech title kept on must-haves" in ledger["exception_clause"]


def test_pick_template_respects_reuse_cap() -> None:
    skeletons = {"top10": ["A {positive}", "B {positive}"]}
    counter: Counter = Counter()
    chosen_ids = set()
    for _ in range(20):
        tid, _ = _pick_template(skeletons, "top10", counter, cap=5)
        chosen_ids.add(tid)
    # With cap=5 we should see both templates rotate.
    assert len(chosen_ids) == 2


def test_render_top_100_skel_smoke() -> None:
    skeletons = {
        "top10": ["[10] {positive}. {skill}. {logistics}. {concern}"],
        "top25": ["[25] {positive}. {concern}"],
        "top50": ["[50] {positive}. {concern}"],
        "top100": ["[100] {positive}. {concern}"],
    }
    rows = [{**_feature_row(candidate_id=f"CAND_{i:07d}"), "rank": i + 1} for i in range(60)]
    raw_lookup = {f"CAND_{i:07d}": _raw() for i in range(60)}
    out = render_top_100_reasoning(
        rows, raw_lookup, skeletons, template_reuse_cap=DEFAULT_TEMPLATE_REUSE_CAP
    )
    assert len(out) == 60
    bands = {o["template_id"].split("-")[0] for o in out}
    assert "top10" in bands and "top25" in bands and "top50" in bands and "top100" in bands


def test_rank_50_plus_must_have_concern_token() -> None:
    skeletons = {"top50": ["[50] {positive}."]}
    rows = [{**_feature_row(candidate_id="CAND_0000060"), "rank": 60}]
    out = render_top_100_reasoning(rows, {"CAND_0000060": _raw()}, skeletons)
    lower = out[0]["reasoning"].lower()
    assert any(tok in lower for tok in CONCERN_TOKENS)


def test_load_skeletons_from_yaml_round_trip() -> None:
    text = "top10:\n  - 'hello {positive}'\ntop100:\n  - 'bye'\n"
    out = load_skeletons_from_yaml(text)
    assert out == {"top10": ["hello {positive}"], "top100": ["bye"]}
