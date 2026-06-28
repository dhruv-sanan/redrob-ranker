"""Phase-1 test fixtures.

Hand-crafted 50-candidate synthetic dataset covering every edge case from
`lld.md §4`. Used by Phase-1 parsing/manifest tests and reused by Phase-2+
feature-builder tests. Fully deterministic — no randomness, no `faker`, no
real `candidate_id` from the 100K dataset.

Bucket counts (sum = 50):
    5 Tier-A real fits          (retrieval/ranking/eval/vector evidence dense)
    3 Plain-language Tier-5     (channel-Y/Z only, zero channel-X)
    5 Keyword stuffers          (non-tech title + bloated AI skill list)
    3 Honeypots                 (future date / duration mismatch / zero-duration expert)
    5 Services-only             (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini)
    3 Services-to-product       (prior services, current product-co applied ML)
    3 Pure CV/speech/robotics   (no NLP/IR evidence)
    2 Recent-only LangChain     (<12mo, no prior ML production)
    2 Inactive Architect/VP     (no production code 18mo, managerial roles)
    2 Outside India no-relocate (strong AI but logistics blocked)
    2 Skill alias drift         (assessment under aliased name)
    2 Missing assessments       (strong descriptions, empty scores dict)
    2 High notice / low response (logistics signal stress)
    1 Concurrent advisor        (overlapping roles — must NOT trigger interval anomaly)
   10 Filler mid-tier
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _signals(**overrides: Any) -> dict[str, Any]:
    """Return a fully-populated `redrob_signals` block. All 23 schema fields present."""
    base: dict[str, Any] = {
        "profile_completeness_score": 78.0,
        "signup_date": "2024-01-15",
        "last_active_date": "2026-05-22",
        "open_to_work_flag": True,
        "profile_views_received_30d": 14,
        "applications_submitted_30d": 3,
        "recruiter_response_rate": 0.55,
        "avg_response_time_hours": 9.0,
        "skill_assessment_scores": {},
        "connection_count": 320,
        "endorsements_received": 38,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 28.0, "max": 52.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 25.0,
        "search_appearance_30d": 6,
        "saved_by_recruiters_30d": 2,
        "interview_completion_rate": 0.85,
        "offer_acceptance_rate": 0.5,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    }
    base.update(overrides)
    return base


def _profile(
    headline: str,
    summary: str,
    current_title: str,
    current_company: str,
    current_industry: str,
    yoe: float = 7.0,
    location: str = "Pune, MH",
    country: str = "India",
    current_company_size: str = "501-1000",
) -> dict[str, Any]:
    return {
        "anonymized_name": f"Candidate ({current_title})",
        "headline": headline,
        "summary": summary,
        "location": location,
        "country": country,
        "years_of_experience": yoe,
        "current_title": current_title,
        "current_company": current_company,
        "current_company_size": current_company_size,
        "current_industry": current_industry,
    }


def _role(
    company: str,
    title: str,
    start: str,
    end: str | None,
    duration: int,
    is_current: bool,
    industry: str,
    description: str,
    company_size: str = "501-1000",
) -> dict[str, Any]:
    return {
        "company": company,
        "title": title,
        "start_date": start,
        "end_date": end,
        "duration_months": duration,
        "is_current": is_current,
        "industry": industry,
        "company_size": company_size,
        "description": description,
    }


def _edu(
    institution: str,
    degree: str = "B.Tech",
    field: str = "Computer Science",
    start: int = 2014,
    end: int = 2018,
    tier: str = "tier_2",
    grade: str | None = "First Class",
) -> dict[str, Any]:
    return {
        "institution": institution,
        "degree": degree,
        "field_of_study": field,
        "start_year": start,
        "end_year": end,
        "grade": grade,
        "tier": tier,
    }


def _skill(name: str, prof: str, endorsements: int = 10, duration: int = 24) -> dict[str, Any]:
    return {
        "name": name,
        "proficiency": prof,
        "endorsements": endorsements,
        "duration_months": duration,
    }


def _cand(
    cid: str,
    profile: dict[str, Any],
    career: list[dict[str, Any]],
    education: list[dict[str, Any]],
    skills: list[dict[str, Any]],
    signals: dict[str, Any],
    certifications: list[dict[str, Any]] | None = None,
    languages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": cid,
        "profile": profile,
        "career_history": career,
        "education": education,
        "skills": skills,
        "redrob_signals": signals,
        "certifications": certifications or [],
        "languages": languages or [{"language": "English", "proficiency": "professional"}],
    }


# ---------------------------------------------------------------------------
# Bucket A — Tier-A real fits (5)
# ---------------------------------------------------------------------------

TIER_A: list[dict[str, Any]] = [
    _cand(
        "CAND_0000001",
        _profile(
            "Senior ML Engineer — production search and recommendations",
            "8 yrs ML engineer. Owned ranking and retrieval at product scale.",
            "Senior ML Engineer",
            "Acme Recsys Inc",
            "Internet",
            yoe=7.5,
            location="Pune, MH",
        ),
        [
            _role(
                "Acme Recsys Inc",
                "Senior ML Engineer",
                "2023-04-01",
                None,
                38,
                True,
                "Internet",
                "Built and shipped BGE-based dense retrieval with hybrid BM25 fallback. "
                "Owned NDCG@10 dashboard. Productionized cross-encoder re-rank serving 12M qps.",
            ),
            _role(
                "DataCo Search",
                "ML Engineer",
                "2020-02-01",
                "2023-03-15",
                37,
                False,
                "Internet",
                "Designed offline retrieval evaluation harness (NDCG, MRR, MAP). "
                "Deployed Pinecone-backed candidate retrieval pipeline with A/B testing.",
            ),
        ],
        [_edu("IIT Bombay", tier="tier_1")],
        [
            _skill("Python", "expert", 80, 90),
            _skill("NLP", "expert", 50, 60),
            _skill("RAG", "advanced", 20, 18),
            _skill("Pinecone", "advanced", 12, 16),
            _skill("Information Retrieval", "expert", 30, 70),
        ],
        _signals(
            skill_assessment_scores={
                "Python": 92.0,
                "NLP": 88.0,
                "RAG": 85.0,
                "Pinecone": 80.0,
                "Information Retrieval": 90.0,
            },
            last_active_date="2026-05-28",
            github_activity_score=75.0,
            recruiter_response_rate=0.8,
            notice_period_days=30,
        ),
    ),
    _cand(
        "CAND_0000002",
        _profile(
            "Founding AI Engineer — RAG and LLM-powered product search",
            "Built retrieval stacks at two product startups.",
            "Founding AI Engineer",
            "Vectorly",
            "Internet",
            yoe=6.5,
            location="Bangalore, KA",
        ),
        [
            _role(
                "Vectorly",
                "Founding AI Engineer",
                "2024-06-01",
                None,
                24,
                True,
                "Internet",
                "Designed retrieval-augmented generation pipeline using BGE embeddings + Qdrant. "
                "Built ranking evaluation suite with NDCG, recall@k, A/B harness.",
            ),
            _role(
                "SearchHub",
                "Senior Software Engineer",
                "2019-08-01",
                "2024-05-15",
                57,
                False,
                "Internet",
                "Owned dense retrieval microservice. Migrated from Elasticsearch BM25 to "
                "hybrid sparse-dense ranking. Shipped LoRA fine-tuning for query rewriting.",
            ),
        ],
        [_edu("BITS Pilani", tier="tier_1")],
        [
            _skill("Python", "expert", 70, 78),
            _skill("LLM Fine-tuning", "expert", 22, 22),
            _skill("Vector Search", "expert", 18, 28),
            _skill("PyTorch", "advanced", 35, 60),
        ],
        _signals(
            skill_assessment_scores={
                "Python": 90.0,
                "LLM Fine-tuning": 82.0,
                "Vector Search": 86.0,
                "PyTorch": 84.0,
            },
            last_active_date="2026-05-30",
            github_activity_score=82.0,
            recruiter_response_rate=0.75,
        ),
    ),
    _cand(
        "CAND_0000003",
        _profile(
            "Staff Search Engineer — hybrid retrieval and LTR",
            "Search and ranking veteran with deployment-first mindset.",
            "Staff Search Engineer",
            "FindIt",
            "E-commerce",
            yoe=8.5,
            location="Noida, UP",
        ),
        [
            _role(
                "FindIt",
                "Staff Search Engineer",
                "2022-01-01",
                None,
                53,
                True,
                "E-commerce",
                "Shipped hybrid BM25 + dense retrieval re-ranker. Designed LambdaMART LTR offline "
                "evaluation pipeline with MAP, NDCG, recall@k. On-call rotation for search relevance.",
            ),
            _role(
                "Bigcom Marketplace",
                "Senior Software Engineer",
                "2017-07-01",
                "2021-12-15",
                54,
                False,
                "E-commerce",
                "Owned candidate retrieval service for product search. Built embedding-based "
                "personalization. Migrated rerank stack to cross-encoder with measured NDCG gains.",
            ),
        ],
        [_edu("IIIT Hyderabad", tier="tier_1")],
        [
            _skill("Python", "expert", 90, 100),
            _skill("Elasticsearch", "expert", 50, 80),
            _skill("LambdaMART", "advanced", 8, 30),
            _skill("Information Retrieval", "expert", 60, 90),
        ],
        _signals(
            skill_assessment_scores={
                "Python": 95.0,
                "Elasticsearch": 89.0,
                "LambdaMART": 78.0,
                "Information Retrieval": 92.0,
            },
            last_active_date="2026-05-25",
            github_activity_score=68.0,
        ),
    ),
    _cand(
        "CAND_0000004",
        _profile(
            "Applied ML Engineer — production NLP and ranking",
            "Built recommender and search systems at consumer product companies.",
            "Senior Applied ML Engineer",
            "ReelMedia",
            "Media",
            yoe=7.0,
            location="Hyderabad, TS",
        ),
        [
            _role(
                "ReelMedia",
                "Senior Applied ML Engineer",
                "2022-05-01",
                None,
                49,
                True,
                "Media",
                "Built personalized feed retrieval using sentence-transformer embeddings + FAISS. "
                "Owned ranking eval (NDCG, MRR). Productionized re-rank serving 8M DAU.",
            ),
            _role(
                "Streamly",
                "ML Engineer",
                "2019-06-01",
                "2022-04-15",
                34,
                False,
                "Media",
                "Designed candidate-content matching pipeline; deployed similarity-search retriever; "
                "A/B-tested ranking variants and measured offline-to-online correlation.",
            ),
        ],
        [_edu("NIT Trichy", tier="tier_2")],
        [
            _skill("Python", "expert", 65, 80),
            _skill("FAISS", "advanced", 12, 30),
            _skill("Sentence-Transformers", "advanced", 14, 24),
            _skill("Ranking Evaluation", "advanced", 6, 50),
        ],
        _signals(
            skill_assessment_scores={
                "Python": 88.0,
                "FAISS": 79.0,
                "Sentence-Transformers": 81.0,
                "Ranking Evaluation": 85.0,
            },
            last_active_date="2026-05-27",
            github_activity_score=60.0,
        ),
    ),
    _cand(
        "CAND_0000005",
        _profile(
            "Senior NLP Engineer — LoRA, RAG, retrieval at scale",
            "Production NLP across two product companies plus OSS.",
            "Senior NLP Engineer",
            "TalkGPT",
            "AI",
            yoe=6.0,
            location="Mumbai, MH",
        ),
        [
            _role(
                "TalkGPT",
                "Senior NLP Engineer",
                "2023-09-01",
                None,
                33,
                True,
                "AI",
                "Shipped retrieval-augmented chatbot using Weaviate vector index. "
                "Fine-tuned LoRA adapters for domain QA. Owned NDCG@10 and recall@100 dashboards.",
            ),
            _role(
                "OpenLM",
                "ML Engineer",
                "2020-09-01",
                "2023-08-15",
                36,
                False,
                "AI",
                "Built dense retrieval and re-rank stack. Designed offline evaluation harness "
                "(NDCG, MAP, MRR). Owned production deployment.",
            ),
        ],
        [_edu("IIT Madras", tier="tier_1")],
        [
            _skill("Python", "expert", 60, 72),
            _skill("LoRA", "advanced", 10, 18),
            _skill("Weaviate", "advanced", 5, 14),
            _skill("RAG", "expert", 25, 28),
        ],
        _signals(
            skill_assessment_scores={
                "Python": 91.0,
                "LoRA": 84.0,
                "Weaviate": 78.0,
                "RAG": 88.0,
            },
            last_active_date="2026-05-29",
            github_activity_score=85.0,
            willing_to_relocate=True,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Bucket B — Plain-language Tier-5 fits (3) — JD "right answer"
# ---------------------------------------------------------------------------

TIER_5_PLAIN: list[dict[str, Any]] = [
    _cand(
        "CAND_0000006",
        _profile(
            "Backend Engineer building matching and recommendation systems",
            "Built candidate-role matching used by recruiters. No fancy buzzwords.",
            "Backend Engineer",
            "JobsAtScale",
            "HR-Tech",
            yoe=6.5,
            location="Bangalore, KA",
        ),
        [
            _role(
                "JobsAtScale",
                "Backend Engineer",
                "2022-03-01",
                None,
                51,
                True,
                "HR-Tech",
                "Built and owned candidate-role matching system used by recruiters every day. "
                "Designed similarity-based candidate retrieval; productionized search relevance "
                "improvements with A/B tests; learned from feedback signals.",
            ),
            _role(
                "ShopCo",
                "Software Engineer",
                "2019-01-01",
                "2022-02-15",
                37,
                False,
                "E-commerce",
                "Owned personalized feed and recommendations for the storefront. "
                "Shipped similar-item retrieval; instrumented and monitored production ranking.",
            ),
        ],
        [_edu("VIT", tier="tier_3")],
        [
            _skill("Python", "expert", 40, 70),
            _skill("Django", "advanced", 30, 60),
            _skill("Search", "advanced", 8, 36),
        ],
        _signals(
            skill_assessment_scores={"Python": 86.0, "Search": 78.0},
            last_active_date="2026-05-24",
            github_activity_score=35.0,
        ),
    ),
    _cand(
        "CAND_0000007",
        _profile(
            "Software Engineer — semantic matching and search relevance",
            "Built matching pipelines at two product companies.",
            "Software Engineer",
            "PeopleMatch",
            "HR-Tech",
            yoe=7.0,
            location="Pune, MH",
        ),
        [
            _role(
                "PeopleMatch",
                "Software Engineer",
                "2021-07-01",
                None,
                59,
                True,
                "HR-Tech",
                "Designed and shipped semantic matching for candidate recommendations. "
                "Migrated keyword search to dense retrieval; A/B tested every ranking change; "
                "instrumented offline-to-online correlation.",
            ),
        ],
        [_edu("PEC Chandigarh", tier="tier_3")],
        [
            _skill("Python", "expert", 38, 72),
            _skill("PostgreSQL", "advanced", 22, 60),
        ],
        _signals(
            skill_assessment_scores={"Python": 84.0},
            last_active_date="2026-05-23",
            github_activity_score=20.0,
        ),
    ),
    _cand(
        "CAND_0000008",
        _profile(
            "Senior Software Engineer — recommendations and ranking",
            "Product-company recommender system veteran.",
            "Senior Software Engineer",
            "LearnUp",
            "EdTech",
            yoe=6.0,
            location="Delhi, DL",
        ),
        [
            _role(
                "LearnUp",
                "Senior Software Engineer",
                "2022-09-01",
                None,
                45,
                True,
                "EdTech",
                "Owned course recommendations service. Built similar-content retrieval; "
                "monitored ranking metrics; ran experiments comparing variants on real users.",
            ),
        ],
        [_edu("DAIICT", tier="tier_3")],
        [
            _skill("Python", "expert", 30, 60),
            _skill("Recommendation Systems", "advanced", 6, 36),
        ],
        _signals(
            skill_assessment_scores={"Python": 80.0, "Recommendation Systems": 78.0},
            last_active_date="2026-05-26",
        ),
    ),
]

# ---------------------------------------------------------------------------
# Bucket C — Keyword stuffers (5) — non-tech title + bloated AI skill list
# ---------------------------------------------------------------------------

_STUFFER_SKILLS = [
    _skill("NLP", "expert", 6, 12),
    _skill("RAG", "expert", 4, 10),
    _skill("Pinecone", "advanced", 3, 8),
    _skill("LLM", "expert", 5, 11),
    _skill("Vector Search", "advanced", 2, 9),
    _skill("LoRA", "advanced", 1, 7),
    _skill("BGE", "advanced", 1, 6),
    _skill("Embeddings", "expert", 4, 12),
    _skill("Information Retrieval", "advanced", 2, 8),
]


def _stuffer(
    cid: str, title: str, industry: str, company: str, location: str = "Mumbai, MH"
) -> dict[str, Any]:
    return _cand(
        cid,
        _profile(
            f"{title} — strategy, ops, and team leadership",
            f"{title} leading cross-functional team. Listed AI skills are exploratory.",
            title,
            company,
            industry,
            yoe=5.5,
            location=location,
        ),
        [
            _role(
                company,
                title,
                "2021-08-01",
                None,
                58,
                True,
                industry,
                "Led marketing/ops team; built brand strategy; managed budgets; "
                "ran weekly planning; coordinated with vendors and external agencies.",
            ),
        ],
        [_edu("Generic University", tier="tier_3")],
        deepcopy(_STUFFER_SKILLS),
        _signals(
            skill_assessment_scores={},
            last_active_date="2026-05-15",
            github_activity_score=-1.0,
            profile_completeness_score=68.0,
        ),
    )


STUFFERS: list[dict[str, Any]] = [
    _stuffer("CAND_0000009", "Marketing Manager", "Marketing", "BrandPlus"),
    _stuffer("CAND_0000010", "HR Manager", "HR", "PeopleOps"),
    _stuffer("CAND_0000011", "Sales Manager", "Sales", "RevDrive"),
    _stuffer("CAND_0000012", "Customer Support", "Customer Support", "HelpFast"),
    _stuffer("CAND_0000013", "Operations Manager", "Operations", "FlowOps"),
]

# ---------------------------------------------------------------------------
# Bucket D — Honeypots (3)
# ---------------------------------------------------------------------------

HONEYPOTS: list[dict[str, Any]] = [
    # H1: role start_date strictly after REFERENCE_DATE (2026-06-01)
    _cand(
        "CAND_0000014",
        _profile(
            "Senior ML Engineer",
            "Profile dates obviously broken.",
            "Senior ML Engineer",
            "FutureCorp",
            "Internet",
            yoe=7.0,
        ),
        [
            _role(
                "FutureCorp",
                "Senior ML Engineer",
                "2027-01-01",
                None,
                12,
                True,
                "Internet",
                "Working on retrieval and ranking systems.",
            ),
        ],
        [_edu("Random College", tier="tier_3")],
        [_skill("Python", "expert", 5, 24)],
        _signals(skill_assessment_scores={"Python": 70.0}),
    ),
    # H2: 8 years claimed but `duration_months=12` mismatch on a single 12-month role
    _cand(
        "CAND_0000015",
        _profile(
            "Senior ML Engineer",
            "8 yrs claimed but career rows say 1 yr total.",
            "Senior ML Engineer",
            "MismatchCorp",
            "Internet",
            yoe=8.0,
        ),
        [
            _role(
                "MismatchCorp",
                "Senior ML Engineer",
                "2025-06-01",
                None,
                12,
                True,
                "Internet",
                "Working on machine learning systems.",
            ),
        ],
        [_edu("Random College", tier="tier_3")],
        [_skill("Python", "expert", 5, 24)],
        _signals(skill_assessment_scores={}),
    ),
    # H3: 8 expert AI skills with zero duration_months each
    _cand(
        "CAND_0000016",
        _profile(
            "Senior AI Engineer",
            "Expert in everything, but no time spent on any of it.",
            "Senior AI Engineer",
            "InstantExpert",
            "AI",
            yoe=4.0,
        ),
        [
            _role(
                "InstantExpert",
                "Senior AI Engineer",
                "2023-06-01",
                None,
                36,
                True,
                "AI",
                "Working on AI projects.",
            ),
        ],
        [_edu("Some Institute", tier="tier_3")],
        [
            _skill("NLP", "expert", 0, 0),
            _skill("RAG", "expert", 0, 0),
            _skill("LoRA", "expert", 0, 0),
            _skill("LLM Fine-tuning", "expert", 0, 0),
            _skill("Vector Search", "expert", 0, 0),
            _skill("Embeddings", "expert", 0, 0),
            _skill("Information Retrieval", "expert", 0, 0),
            _skill("PyTorch", "expert", 0, 0),
        ],
        _signals(
            skill_assessment_scores={},
            profile_completeness_score=30.0,
            linkedin_connected=False,
            verified_email=False,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Bucket E — Services-only (5)
# ---------------------------------------------------------------------------


def _services_only(cid: str, services_co: str) -> dict[str, Any]:
    return _cand(
        cid,
        _profile(
            f"AI Engineer — {services_co}",
            f"Career entirely at {services_co} doing client engagements.",
            "Senior Software Engineer",
            services_co,
            "IT Services",
            yoe=8.0,
            location="Pune, MH",
        ),
        [
            _role(
                services_co,
                "Senior Software Engineer",
                "2018-07-01",
                None,
                95,
                True,
                "IT Services",
                "Worked on client engagement delivering enterprise software. "
                "Used Python, SQL, and various AI/ML tools as per project requirement.",
            ),
        ],
        [_edu("Generic Engineering College", tier="tier_3")],
        [
            _skill("Python", "expert", 30, 96),
            _skill("NLP", "advanced", 5, 24),
            _skill("Machine Learning", "advanced", 8, 36),
        ],
        _signals(skill_assessment_scores={"Python": 78.0}),
    )


SERVICES_ONLY: list[dict[str, Any]] = [
    _services_only("CAND_0000017", "Tata Consultancy Services"),
    _services_only("CAND_0000018", "Infosys"),
    _services_only("CAND_0000019", "Wipro"),
    _services_only("CAND_0000020", "Accenture"),
    _services_only("CAND_0000021", "Capgemini"),
]

# ---------------------------------------------------------------------------
# Bucket F — Services-to-product transition (3) — ceiling must NOT fire
# ---------------------------------------------------------------------------


def _services_to_product(cid: str, services_co: str, product_co: str) -> dict[str, Any]:
    return _cand(
        cid,
        _profile(
            f"ML Engineer — transitioned from {services_co} to {product_co}",
            f"Started at {services_co}, now ML at {product_co} (product company applied ML).",
            "Senior ML Engineer",
            product_co,
            "Internet",
            yoe=7.5,
        ),
        [
            _role(
                product_co,
                "Senior ML Engineer",
                "2023-02-01",
                None,
                40,
                True,
                "Internet",
                "Built applied ML pipelines for search and ranking. "
                "Shipped retrieval improvements measured by NDCG and MAP.",
            ),
            _role(
                services_co,
                "Software Engineer",
                "2018-06-01",
                "2023-01-15",
                56,
                False,
                "IT Services",
                "Client engagement work; some ML proof-of-concepts.",
            ),
        ],
        [_edu("NIT Surathkal", tier="tier_2")],
        [
            _skill("Python", "expert", 45, 84),
            _skill("Information Retrieval", "advanced", 8, 30),
        ],
        _signals(
            skill_assessment_scores={"Python": 84.0, "Information Retrieval": 78.0},
            last_active_date="2026-05-26",
        ),
    )


SERVICES_TO_PRODUCT: list[dict[str, Any]] = [
    _services_to_product("CAND_0000022", "Cognizant", "BookMyShow"),
    _services_to_product("CAND_0000023", "Infosys", "Razorpay"),
    _services_to_product("CAND_0000024", "TCS", "Flipkart"),
]

# ---------------------------------------------------------------------------
# Bucket G — Pure CV / speech / robotics, no NLP/IR (3)
# ---------------------------------------------------------------------------


def _cv_only(cid: str, niche: str, description: str) -> dict[str, Any]:
    return _cand(
        cid,
        _profile(
            f"Senior ML Engineer — {niche}",
            f"Career in {niche}, no NLP/IR exposure.",
            "Senior ML Engineer",
            f"{niche.title()}Corp",
            "AI",
            yoe=7.0,
        ),
        [
            _role(
                f"{niche.title()}Corp",
                "Senior ML Engineer",
                "2020-01-01",
                None,
                77,
                True,
                "AI",
                description,
            ),
        ],
        [_edu("IIT Kanpur", tier="tier_1")],
        [
            _skill("Python", "expert", 30, 80),
            _skill("PyTorch", "expert", 25, 60),
            _skill(niche.title(), "expert", 18, 60),
        ],
        _signals(skill_assessment_scores={"Python": 88.0, niche.title(): 86.0}),
    )


CV_ONLY: list[dict[str, Any]] = [
    _cv_only(
        "CAND_0000025",
        "computer vision",
        "Built image classification and object detection models for autonomous driving. "
        "Trained ResNet/ViT backbones; deployed inference services.",
    ),
    _cv_only(
        "CAND_0000026",
        "speech recognition",
        "Trained ASR models for low-resource languages. Owned wake-word detection in production.",
    ),
    _cv_only(
        "CAND_0000027",
        "robotics",
        "Built motion planning and sensor fusion for autonomous robots. "
        "Trained perception models for obstacle detection.",
    ),
]

# ---------------------------------------------------------------------------
# Bucket H — Recent-only LangChain (2)
# ---------------------------------------------------------------------------


def _recent_langchain(cid: str) -> dict[str, Any]:
    return _cand(
        cid,
        _profile(
            "Senior LLM Engineer",
            "Switched to LLMs last year. Mostly chaining OpenAI calls.",
            "Senior LLM Engineer",
            "ChainCo",
            "AI",
            yoe=5.5,
        ),
        [
            _role(
                "ChainCo",
                "Senior LLM Engineer",
                "2025-09-01",
                None,
                9,
                True,
                "AI",
                "Built LangChain agents calling OpenAI APIs. Prototype RAG with managed services.",
            ),
            _role(
                "WebCo",
                "Frontend Engineer",
                "2020-01-01",
                "2025-08-15",
                67,
                False,
                "Internet",
                "Frontend dashboards in React. No ML production.",
            ),
        ],
        [_edu("Anna University", tier="tier_3")],
        [
            _skill("LangChain", "expert", 5, 9),
            _skill("OpenAI", "advanced", 8, 9),
        ],
        _signals(skill_assessment_scores={}),
    )


RECENT_LANGCHAIN: list[dict[str, Any]] = [
    _recent_langchain("CAND_0000028"),
    _recent_langchain("CAND_0000029"),
]
# patch second one for variety
RECENT_LANGCHAIN[1]["candidate_id"] = "CAND_0000029"
RECENT_LANGCHAIN[1]["profile"]["current_company"] = "PromptStudio"
RECENT_LANGCHAIN[1]["career_history"][0]["company"] = "PromptStudio"

# ---------------------------------------------------------------------------
# Bucket I — Inactive Architect / VP (2)
# ---------------------------------------------------------------------------


def _inactive_architect(cid: str, title: str) -> dict[str, Any]:
    return _cand(
        cid,
        _profile(
            f"{title} — strategy and stakeholder management",
            f"{title} with 18+ months entirely in management.",
            title,
            "EnterpriseCorp",
            "Enterprise Software",
            yoe=14.0,
            location="Bangalore, KA",
        ),
        [
            _role(
                "EnterpriseCorp",
                title,
                "2022-01-01",
                None,
                53,
                True,
                "Enterprise Software",
                "Led architecture reviews, set up team OKRs, presented roadmap to executives. "
                "Oversaw vendor selection. Mentored engineering leads.",
            ),
        ],
        [_edu("BITS Pilani", tier="tier_1")],
        [
            _skill("Software Architecture", "expert", 80, 120),
            _skill("Leadership", "expert", 50, 100),
        ],
        _signals(
            skill_assessment_scores={},
            last_active_date="2025-11-10",
            recruiter_response_rate=0.15,
            github_activity_score=-1.0,
        ),
    )


INACTIVE_ARCHITECT: list[dict[str, Any]] = [
    _inactive_architect("CAND_0000030", "VP Engineering"),
    _inactive_architect("CAND_0000031", "Tech Architect"),
]

# ---------------------------------------------------------------------------
# Bucket J — Outside India, no relocation (2)
# ---------------------------------------------------------------------------


def _outside_india(cid: str, country: str, city: str) -> dict[str, Any]:
    return _cand(
        cid,
        _profile(
            "Senior ML Engineer — international",
            f"Strong AI engineering but based in {country} and not relocating.",
            "Senior ML Engineer",
            "GlobalAI",
            "AI",
            yoe=7.0,
            location=f"{city}, {country}",
            country=country,
        ),
        [
            _role(
                "GlobalAI",
                "Senior ML Engineer",
                "2023-01-01",
                None,
                41,
                True,
                "AI",
                "Built dense retrieval pipelines with BGE embeddings. "
                "Owned NDCG@10 dashboard. Productionized RAG.",
            ),
        ],
        [_edu("Technical University Munich", tier="tier_1")],
        [
            _skill("Python", "expert", 40, 80),
            _skill("RAG", "expert", 12, 24),
            _skill("BGE", "advanced", 6, 16),
        ],
        _signals(
            skill_assessment_scores={"Python": 90.0, "RAG": 85.0, "BGE": 80.0},
            willing_to_relocate=False,
            preferred_work_mode="remote",
        ),
    )


OUTSIDE_INDIA: list[dict[str, Any]] = [
    _outside_india("CAND_0000032", "Germany", "Munich"),
    _outside_india("CAND_0000033", "Singapore", "Singapore"),
]

# ---------------------------------------------------------------------------
# Bucket K — Skill alias drift (2)
# ---------------------------------------------------------------------------


def _alias_drift(cid: str, skill_in_list: str, assessment_key: str) -> dict[str, Any]:
    return _cand(
        cid,
        _profile(
            "Senior ML Engineer — skill naming drift",
            "Skill listed with full name; assessment recorded with alias.",
            "Senior ML Engineer",
            "AliasCorp",
            "Internet",
            yoe=7.0,
        ),
        [
            _role(
                "AliasCorp",
                "Senior ML Engineer",
                "2022-04-01",
                None,
                50,
                True,
                "Internet",
                "Built dense retrieval pipelines using sentence-transformers and Qdrant. "
                "Owned ranking eval (NDCG, MRR, MAP).",
            ),
        ],
        [_edu("IIT Kharagpur", tier="tier_1")],
        [
            _skill("Python", "expert", 50, 80),
            _skill(skill_in_list, "expert", 20, 40),
        ],
        _signals(
            skill_assessment_scores={"Python": 90.0, assessment_key: 86.0},
        ),
    )


ALIAS_DRIFT: list[dict[str, Any]] = [
    _alias_drift("CAND_0000034", "Fine-tuning LLMs", "Fine-tuning"),
    _alias_drift("CAND_0000035", "Natural Language Processing", "NLP"),
]

# ---------------------------------------------------------------------------
# Bucket L — Missing assessment scores (2)
# ---------------------------------------------------------------------------

MISSING_ASSESSMENT: list[dict[str, Any]] = [
    _cand(
        "CAND_0000036",
        _profile(
            "Senior ML Engineer — strong evidence but no platform assessments",
            "Strong production retrieval record but skipped Redrob skill assessments.",
            "Senior ML Engineer",
            "QuietCorp",
            "Internet",
            yoe=7.5,
        ),
        [
            _role(
                "QuietCorp",
                "Senior ML Engineer",
                "2022-06-01",
                None,
                48,
                True,
                "Internet",
                "Built and shipped BGE retrieval + cross-encoder rerank for search. "
                "Owned NDCG@10 dashboard and on-call rotation.",
            ),
        ],
        [_edu("IIT Roorkee", tier="tier_1")],
        [
            _skill("Python", "expert", 60, 90),
            _skill("RAG", "advanced", 14, 30),
            _skill("BGE", "advanced", 8, 20),
        ],
        _signals(skill_assessment_scores={}),
    ),
    _cand(
        "CAND_0000037",
        _profile(
            "Founding ML Engineer — startup mode, no assessments",
            "Career retrieval + LTR. Never logged in to take Redrob assessments.",
            "Founding ML Engineer",
            "StealthCo",
            "AI",
            yoe=6.0,
        ),
        [
            _role(
                "StealthCo",
                "Founding ML Engineer",
                "2023-04-01",
                None,
                38,
                True,
                "AI",
                "Shipped dense retrieval with hybrid BM25 fallback. Owned ranking eval pipeline.",
            ),
        ],
        [_edu("IIIT Bangalore", tier="tier_2")],
        [
            _skill("Python", "expert", 30, 80),
            _skill("LambdaMART", "advanced", 4, 20),
            _skill("Vector Search", "advanced", 6, 18),
        ],
        _signals(skill_assessment_scores={}),
    ),
]

# ---------------------------------------------------------------------------
# Bucket M — High notice / low response (2)
# ---------------------------------------------------------------------------

HIGH_NOTICE_LOW_RESPONSE: list[dict[str, Any]] = [
    _cand(
        "CAND_0000038",
        _profile(
            "Senior ML Engineer — locked in 90-day notice",
            "Strong skills but currently bonded to 90-day notice period.",
            "Senior ML Engineer",
            "LockedCorp",
            "Internet",
            yoe=7.0,
        ),
        [
            _role(
                "LockedCorp",
                "Senior ML Engineer",
                "2022-07-01",
                None,
                47,
                True,
                "Internet",
                "Built dense retrieval and ranking eval. NDCG@10 owner.",
            ),
        ],
        [_edu("IIT Guwahati", tier="tier_1")],
        [
            _skill("Python", "expert", 55, 84),
            _skill("RAG", "advanced", 10, 24),
        ],
        _signals(
            skill_assessment_scores={"Python": 88.0, "RAG": 82.0},
            notice_period_days=90,
            recruiter_response_rate=0.35,
        ),
    ),
    _cand(
        "CAND_0000039",
        _profile(
            "Senior ML Engineer — strong fit but unresponsive",
            "Excellent retrieval record but rarely responds to recruiters.",
            "Senior ML Engineer",
            "GhostCorp",
            "Internet",
            yoe=6.5,
        ),
        [
            _role(
                "GhostCorp",
                "Senior ML Engineer",
                "2023-03-01",
                None,
                39,
                True,
                "Internet",
                "Owned BGE retrieval + cross-encoder rerank. Built MAP/NDCG eval suite.",
            ),
        ],
        [_edu("IIT Delhi", tier="tier_1")],
        [
            _skill("Python", "expert", 45, 78),
            _skill("Cross-Encoder", "advanced", 6, 20),
        ],
        _signals(
            skill_assessment_scores={"Python": 86.0, "Cross-Encoder": 80.0},
            recruiter_response_rate=0.05,
            last_active_date="2025-12-10",
        ),
    ),
]

# ---------------------------------------------------------------------------
# Bucket N — Concurrent advisor + full-time (1)
# ---------------------------------------------------------------------------

CONCURRENT_ADVISOR: list[dict[str, Any]] = [
    _cand(
        "CAND_0000040",
        _profile(
            "Founding ML Engineer — concurrent advisor and OSS",
            "Holds three concurrent roles. Must NOT trigger interval honeypot.",
            "Founding ML Engineer",
            "MainCo",
            "AI",
            yoe=7.0,
        ),
        [
            _role(
                "MainCo",
                "Founding ML Engineer",
                "2023-06-01",
                None,
                36,
                True,
                "AI",
                "Shipped retrieval + ranking + LoRA fine-tuning pipelines.",
            ),
            _role(
                "StartupAdvise",
                "Technical Advisor",
                "2023-06-01",
                None,
                36,
                True,
                "AI",
                "Advised seed-stage AI startups on RAG architecture.",
            ),
            _role(
                "OSS Fellowship",
                "OSS Maintainer",
                "2024-01-01",
                None,
                29,
                True,
                "Internet",
                "Maintained an open-source embedding evaluation library.",
            ),
        ],
        [_edu("IIT Kanpur", tier="tier_1")],
        [
            _skill("Python", "expert", 60, 84),
            _skill("RAG", "expert", 14, 30),
        ],
        _signals(
            skill_assessment_scores={"Python": 92.0, "RAG": 88.0},
            github_activity_score=92.0,
        ),
    ),
]

# ---------------------------------------------------------------------------
# Bucket O — Filler mid-tier (10)
# ---------------------------------------------------------------------------


def _filler(cid: str, idx: int) -> dict[str, Any]:
    title_options = [
        "Data Engineer",
        "Backend Engineer",
        "Full Stack Engineer",
        "Data Scientist",
        "DevOps Engineer",
        "Site Reliability Engineer",
        "QA Engineer",
        "Mobile Engineer",
        "Cloud Engineer",
        "Platform Engineer",
    ]
    title = title_options[idx % len(title_options)]
    return _cand(
        cid,
        _profile(
            f"{title} — adjacent skills, mid-pack target rank",
            f"Generic {title.lower()} with adjacent skills to AI work.",
            title,
            f"FillerCo{idx}",
            "Software",
            yoe=4.5 + (idx % 5) * 0.5,
            location="Hyderabad, TS",
        ),
        [
            _role(
                f"FillerCo{idx}",
                title,
                "2021-06-01",
                None,
                60 - idx,
                True,
                "Software",
                f"Worked as {title.lower()} on internal tooling and platform services. "
                "Used Python and SQL for data work.",
            ),
        ],
        [_edu("Generic Engineering College", tier="tier_3")],
        [
            _skill("Python", "advanced", 20, 48),
            _skill("SQL", "advanced", 18, 48),
        ],
        _signals(skill_assessment_scores={"Python": 70.0}),
    )


FILLERS: list[dict[str, Any]] = [_filler(f"CAND_00000{41+i}"[:12], i) for i in range(10)]

# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

SYNTHETIC_50: list[dict[str, Any]] = (
    TIER_A
    + TIER_5_PLAIN
    + STUFFERS
    + HONEYPOTS
    + SERVICES_ONLY
    + SERVICES_TO_PRODUCT
    + CV_ONLY
    + RECENT_LANGCHAIN
    + INACTIVE_ARCHITECT
    + OUTSIDE_INDIA
    + ALIAS_DRIFT
    + MISSING_ASSESSMENT
    + HIGH_NOTICE_LOW_RESPONSE
    + CONCURRENT_ADVISOR
    + FILLERS
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_50() -> list[dict[str, Any]]:
    """50 hand-crafted candidates, deep-copied so tests can't mutate the master list."""
    return deepcopy(SYNTHETIC_50)


@pytest.fixture
def fixture_count() -> int:
    return len(SYNTHETIC_50)
