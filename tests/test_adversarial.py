"""Adversarial red-team smoke test for the feature pipeline.

Five hand-crafted edge-case candidates exercise the honeypot ledger,
anti-pattern ceilings, and tier assignment end-to-end. These are
deliberately more pathological than the ``synthetic_50`` baseline fixture
— each row targets one specific rule the ranker must (or must not) fire.

Schema mirrors ``conftest._cand`` so the rows are valid inputs to
``build_feature_row`` without touching the embedding pipeline.
"""

from __future__ import annotations

from typing import Any

from src.feature_pipeline import build_feature_row
from src.reference_date import REFERENCE_DATE

from .conftest import _cand, _edu, _profile, _role, _signals, _skill

# ---------------------------------------------------------------------------
# (1) Perfect stuffer — 8 advanced AI skills with zero role descriptions
# ---------------------------------------------------------------------------
PERFECT_STUFFER = _cand(
    "ADV_0000001",
    _profile(
        headline="AI Engineer | NLP | RAG | LLM | LoRA | BGE | Pinecone",
        summary="Senior AI engineer specializing in advanced LLM systems.",
        current_title="Marketing Manager",
        current_company="Acme Brands",
        current_industry="Marketing",
        yoe=5.0,
    ),
    [
        _role(
            "Acme Brands",
            "Marketing Manager",
            "2023-01-01",
            None,
            41,
            True,
            "Marketing",
            "",
        ),
        _role(
            "Beta Marketing",
            "Marketing Coordinator",
            "2020-01-01",
            "2022-12-31",
            36,
            False,
            "Marketing",
            "",
        ),
    ],
    [_edu("State University")],
    [
        _skill("NLP", "advanced", endorsements=10),
        _skill("RAG", "advanced", endorsements=10),
        _skill("LLM", "expert", endorsements=10),
        _skill("LoRA", "advanced", endorsements=10),
        _skill("BGE", "expert", endorsements=10),
        _skill("Pinecone", "advanced", endorsements=10),
        _skill("Embeddings", "expert", endorsements=10),
        _skill("Fine-tuning", "expert", endorsements=10),
    ],
    _signals(skill_assessment_scores={}, profile_completeness_score=42.0, verified_email=False),
)

# ---------------------------------------------------------------------------
# (2) Plain-language fit — solid prose, only 4 skills, no fancy keywords
# ---------------------------------------------------------------------------
PLAIN_LANGUAGE_FIT = _cand(
    "ADV_0000002",
    _profile(
        headline="ML Engineer — product recommendations",
        summary="Builds search and recommendation systems at product companies.",
        current_title="Senior ML Engineer",
        current_company="Flipkart",
        current_industry="Internet",
        yoe=7.0,
    ),
    [
        _role(
            "Flipkart",
            "Senior ML Engineer",
            "2022-06-01",
            None,
            48,
            True,
            "Internet",
            "Owned the recommendation system that powered the homepage feed for 80M users. "
            "Designed the offline evaluation harness with NDCG and online A/B tests. "
            "Shipped a hybrid retrieval pipeline combining dense embeddings with BM25 fallback.",
        ),
        _role(
            "Myntra",
            "ML Engineer",
            "2019-03-01",
            "2022-05-31",
            38,
            False,
            "Internet",
            "Built the candidate ranking model for search. Wrote the production Python service "
            "and the offline NDCG evaluation pipeline. Owned the A/B test infrastructure.",
        ),
    ],
    [_edu("IIT Bombay", tier="tier_1")],
    [
        _skill("Python", "expert", endorsements=80, duration=84),
        _skill("Machine Learning", "expert", endorsements=60, duration=72),
        _skill("SQL", "advanced", endorsements=40, duration=60),
        _skill("Statistics", "advanced", endorsements=30, duration=48),
    ],
    _signals(
        skill_assessment_scores={"Python": 88, "Machine Learning": 82},
        recruiter_response_rate=0.7,
        notice_period_days=30,
    ),
)

# ---------------------------------------------------------------------------
# (3) CV/robotics domain mismatch — applies for NLP/IR role
# ---------------------------------------------------------------------------
CV_DOMAIN_MISMATCH = _cand(
    "ADV_0000003",
    _profile(
        headline="Computer Vision Engineer | Robotics",
        summary="Image classification, segmentation, SLAM, sensor fusion.",
        current_title="Computer Vision Engineer",
        current_company="DroneAI",
        current_industry="Robotics",
        yoe=6.0,
    ),
    [
        _role(
            "DroneAI",
            "Computer Vision Engineer",
            "2022-07-01",
            None,
            47,
            True,
            "Robotics",
            "Built image classification and object detection models for aerial drones. "
            "Owned the semantic segmentation pipeline. Implemented SLAM and sensor fusion "
            "for autonomous navigation.",
        ),
        _role(
            "VisionCo",
            "ML Engineer",
            "2019-01-01",
            "2022-06-30",
            42,
            False,
            "Robotics",
            "Trained speech recognition (ASR) and text-to-speech (TTS) models. "
            "Worked on motion planning and SLAM for robotics platforms.",
        ),
    ],
    [_edu("State University")],
    [
        _skill("Computer Vision", "expert", endorsements=50, duration=72),
        _skill("PyTorch", "advanced", endorsements=40, duration=60),
        _skill("Object Detection", "advanced", endorsements=20, duration=48),
        _skill("Robotics", "advanced", endorsements=15, duration=36),
    ],
    _signals(skill_assessment_scores={"Computer Vision": 85}),
)

# ---------------------------------------------------------------------------
# (4) Late switcher — 15yr marketing, then 2yr AI engineer with real evidence
# ---------------------------------------------------------------------------
LATE_SWITCHER = _cand(
    "ADV_0000004",
    _profile(
        headline="AI Engineer (career switch from marketing)",
        summary="Switched from marketing to AI engineering 2 years ago.",
        current_title="AI Engineer",
        current_company="Razorpay",
        current_industry="Internet",
        yoe=17.0,
    ),
    [
        _role(
            "Razorpay",
            "AI Engineer",
            "2024-06-01",
            None,
            24,
            True,
            "Internet",
            "Built a fraud-detection ranking system in production using BGE embeddings and "
            "a hybrid retrieval pipeline. Owned the offline NDCG evaluation harness and the "
            "online A/B tests. Wrote the Python service end to end.",
        ),
        _role(
            "GrowthLabs",
            "Marketing Manager",
            "2019-01-01",
            "2024-05-31",
            65,
            False,
            "Marketing",
            "Ran paid acquisition and brand campaigns.",
        ),
        _role(
            "BrandWorks",
            "Senior Marketing Associate",
            "2014-01-01",
            "2018-12-31",
            60,
            False,
            "Marketing",
            "Owned brand storytelling and content marketing for a consumer brand.",
        ),
        _role(
            "ContentCo",
            "Marketing Associate",
            "2009-01-01",
            "2013-12-31",
            60,
            False,
            "Marketing",
            "Wrote brand copy and ran small social campaigns.",
        ),
    ],
    [_edu("State University", start=2005, end=2009)],
    [
        _skill("Python", "advanced", endorsements=30, duration=24),
        _skill("Machine Learning", "advanced", endorsements=25, duration=24),
        _skill("Embeddings", "advanced", endorsements=15, duration=20),
        _skill("Ranking", "intermediate", endorsements=10, duration=20),
    ],
    _signals(skill_assessment_scores={"Python": 80, "Machine Learning": 76}),
)

# ---------------------------------------------------------------------------
# (5) Tier-A signals at a services company — services_only ceiling must fire
# ---------------------------------------------------------------------------
TIER_A_IN_SERVICES = _cand(
    "ADV_0000005",
    _profile(
        headline="Senior ML Engineer — search and ranking",
        summary="Production retrieval and ranking systems at scale.",
        current_title="Senior ML Engineer",
        current_company="Infosys",
        current_industry="IT Services",
        yoe=7.5,
    ),
    [
        _role(
            "Infosys",
            "Senior ML Engineer",
            "2022-06-01",
            None,
            48,
            True,
            "IT Services",
            "Owned a production retrieval system using BGE embeddings deployed to enterprise "
            "users. Shipped an end-to-end ranking pipeline with offline NDCG and online A/B. "
            "Wrote the Python backend that serves the model in production.",
        ),
        _role(
            "TCS",
            "ML Engineer",
            "2019-01-01",
            "2022-05-31",
            41,
            False,
            "IT Services",
            "Built a recommendation system for an internal HR-tech product. Designed the "
            "evaluation harness with NDCG and ran A/B tests with recruiter feedback.",
        ),
        _role(
            "Wipro",
            "Junior ML Engineer",
            "2017-01-01",
            "2018-12-31",
            24,
            False,
            "IT Services",
            "Worked on a production search-ranking pipeline using embeddings and hybrid retrieval.",
        ),
    ],
    [_edu("IIT Madras", tier="tier_1")],
    [
        _skill("Python", "expert", endorsements=80, duration=90),
        _skill("Machine Learning", "expert", endorsements=70, duration=84),
        _skill("Embeddings", "advanced", endorsements=40, duration=60),
        _skill("Ranking", "advanced", endorsements=30, duration=60),
        _skill("BGE", "advanced", endorsements=20, duration=36),
    ],
    _signals(skill_assessment_scores={"Python": 92, "Machine Learning": 88}),
)


def _features(candidate: dict[str, Any]) -> dict[str, Any]:
    return build_feature_row(candidate, REFERENCE_DATE)


def test_perfect_stuffer_drops_or_lands_in_tier_e() -> None:
    row = _features(PERFECT_STUFFER)
    dropped = bool(row["honeypot_drop"])
    tier_e = row["tier"] == "E"
    assert dropped or tier_e, (
        f"perfect-stuffer must be honeypot-dropped or Tier E; "
        f"got tier={row['tier']} honeypot_drop={row['honeypot_drop']} "
        f"stuffer_risk={row['stuffer_risk']:.2f}"
    )


def test_plain_language_fit_lands_in_tier_a_or_b() -> None:
    row = _features(PLAIN_LANGUAGE_FIT)
    assert row["tier"] in {"A", "B"}, (
        f"plain-language fit must reach Tier A/B; got tier={row['tier']}, "
        f"retrieval_evidence={row['retrieval_evidence']:.2f}"
    )
    assert not row["honeypot_drop"]


def test_cv_domain_mismatch_hits_rank_50_ceiling() -> None:
    row = _features(CV_DOMAIN_MISMATCH)
    archetypes = row["anti_pattern_archetypes"]
    assert "cv_speech_robotics_only" in archetypes, (
        f"CV/robotics-only candidate must fire cv_speech_robotics_only archetype; "
        f"got archetypes={archetypes}"
    )
    assert row["anti_pattern_ceiling"] == "rank_50"


def test_late_switcher_not_honeypot_dropped() -> None:
    row = _features(LATE_SWITCHER)
    assert not row["honeypot_drop"], (
        f"late switcher (long marketing + recent AI with real evidence) "
        f"must NOT be honeypot-dropped; honeypot_risks={row['honeypot_risks']!r}"
    )


def test_tier_a_signals_in_services_hit_rank_50_ceiling() -> None:
    row = _features(TIER_A_IN_SERVICES)
    archetypes = row["anti_pattern_archetypes"]
    assert "services_only" in archetypes, (
        f"all-services-company candidate must fire services_only archetype; "
        f"got archetypes={archetypes}"
    )
    assert (
        row["anti_pattern_ceiling"] == "rank_50"
    ), f"services_only must apply rank_50 ceiling; got {row['anti_pattern_ceiling']}"
