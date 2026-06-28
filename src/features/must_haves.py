"""6 graded must-have features per problem.md §1.4.

Each ∈ [0, 1]. These exist OUTSIDE the linear blend so tier-driving logic and
top-10 promotion gates can inspect them directly.
"""

from __future__ import annotations

from typing import Any

from src.features.evidence_channels import _compile_patterns, channel_hits_anywhere

VECTOR_DB_TERMS = (
    "Pinecone",
    "Weaviate",
    "Qdrant",
    "Milvus",
    "FAISS",
    "Elasticsearch",
    "OpenSearch",
    "hybrid search",
    "dense retrieval",
    "ANN index",
    "vector index",
)

RANKING_EVAL_TERMS = (
    "NDCG",
    "MRR",
    "MAP",
    "A/B",
    "A_B",
    "offline evaluation",
    "recall@k",
    "precision@k",
)

OWNERSHIP_TERMS = ("built", "shipped", "owned", "deployed", "productionized", "designed")

ML_DATA_TITLE_TERMS = (
    "ML",
    "Machine Learning",
    "Data Engineer",
    "Data Scientist",
    "Backend Engineer",
    "Software Engineer",
    "AI Engineer",
    "NLP Engineer",
    "Search Engineer",
)

SERVICES_INDUSTRIES = {"IT Services", "Consulting", "Outsourcing"}

ML_DESCRIPTION_TERMS = (
    "machine learning",
    "ml",
    "recommendation",
    "ranking",
    "search",
    "retrieval",
    "personalization",
    "NLP",
)


def has_production_retrieval_evidence(career_history: list[dict[str, Any]]) -> float:
    """Channel-X hits + ownership verb in same role. Graded by hit count."""
    if not career_history:
        return 0.0
    hits = channel_hits_anywhere(career_history)
    own_pat = _compile_patterns(OWNERSHIP_TERMS)
    has_ownership_anywhere = any(
        any(p.search(r.get("description", "") or "") for p in own_pat) for r in career_history
    )
    if hits["x"] >= 2 and has_ownership_anywhere:
        return min(1.0, 0.5 + 0.1 * (hits["x"] - 2))
    if hits["x"] == 1 and has_ownership_anywhere:
        return 0.4
    if hits["x"] >= 1:
        return 0.2
    return 0.0


def has_vector_or_hybrid_search_evidence(career_history: list[dict[str, Any]]) -> float:
    pats = _compile_patterns(VECTOR_DB_TERMS)
    score = 0.0
    for role in career_history or []:
        text = role.get("description", "") or ""
        hits = sum(1 for p in pats if p.search(text))
        if hits:
            score = max(score, min(1.0, 0.5 + 0.2 * hits))
    return score


def has_python_backend_depth(
    skills: list[dict[str, Any]], career_history: list[dict[str, Any]]
) -> float:
    py_prof = ""
    for s in skills or []:
        if s.get("name", "").strip().lower() == "python":
            py_prof = s.get("proficiency", "")
            break
    if py_prof not in {"advanced", "expert"}:
        return 0.0
    title_pat = _compile_patterns(ML_DATA_TITLE_TERMS)
    backend_months = 0
    for role in career_history or []:
        title = role.get("title", "") or ""
        if any(p.search(title) for p in title_pat):
            backend_months += int(role.get("duration_months", 0) or 0)
    years = backend_months / 12.0
    if years >= 5:
        return 1.0
    if years >= 3:
        return 0.7
    if years >= 1:
        return 0.4
    return 0.0


def has_ranking_eval_evidence(career_history: list[dict[str, Any]]) -> float:
    pats = _compile_patterns(RANKING_EVAL_TERMS)
    total = 0
    for role in career_history or []:
        text = role.get("description", "") or ""
        total += sum(1 for p in pats if p.search(text))
    if total == 0:
        return 0.0
    if total == 1:
        return 0.4
    if total == 2:
        return 0.7
    return 1.0


def has_product_company_applied_ml_context(career_history: list[dict[str, Any]]) -> float:
    ml_pat = _compile_patterns(ML_DESCRIPTION_TERMS)
    for role in career_history or []:
        industry = (role.get("industry") or "").strip()
        if industry in SERVICES_INDUSTRIES:
            continue
        desc = role.get("description", "") or ""
        if any(p.search(desc) for p in ml_pat):
            return 1.0
    return 0.0


def has_shipper_signal(career_history: list[dict[str, Any]]) -> float:
    pats = _compile_patterns(("deployed", "productionized", "scaled", "monitored", "on-call"))
    n = 0
    for role in career_history or []:
        text = role.get("description", "") or ""
        if any(p.search(text) for p in pats):
            n += 1
    if n == 0:
        return 0.0
    if n == 1:
        return 0.6
    return 1.0


def compute_must_haves(candidate: dict[str, Any]) -> dict[str, float]:
    career = candidate.get("career_history", []) or []
    skills = candidate.get("skills", []) or []
    return {
        "has_production_retrieval_evidence": has_production_retrieval_evidence(career),
        "has_vector_or_hybrid_search_evidence": has_vector_or_hybrid_search_evidence(career),
        "has_python_backend_depth": has_python_backend_depth(skills, career),
        "has_ranking_eval_evidence": has_ranking_eval_evidence(career),
        "has_product_company_applied_ml_context": has_product_company_applied_ml_context(career),
        "has_shipper_signal": has_shipper_signal(career),
    }
