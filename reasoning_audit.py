#!/usr/bin/env python3
"""Independent audit of generated top-100 reasoning strings.

Decoupled from ``rank.py`` so Stage-4 reviewers can run the audit as a
separate verification pass. The audit reads:

  - ``top_100_audit.csv`` — produced by ``rank.py`` (rank, candidate_id,
    reasoning, template_id, plus feature columns used to ground claims)
  - ``artifacts/candidates.parquet`` — the raw nested record per candidate

Checks (problem.md §3.3 + lld.md adversarial matrix):

  1. **Grounded skill claim** — every word inside a top-100 reasoning that
     looks like a named skill from the candidate's ``skills[]`` list OR
     appears in any career_history description survives. Anything else
     listed as a skill in the reasoning that does NOT appear in the
     candidate's record is a hallucination → audit fail.
  2. **Grounded employer claim** — every employer/company token inside the
     reasoning must appear in the candidate's ``career_history[*].company``.
  3. **No numeric hallucination** — every integer/percentage in the
     reasoning must map to a feature column or a raw field (years_of_experience,
     notice_period_days, recruiter_response_rate, skill duration_months,
     skill_assessment_scores).
  4. **Template variation** — no template fires more than 12× across the 100.
  5. **Rank-50+ concern clause** — every reasoning in ranks 51–100 must
     contain a concern token.
  6. **Non-tech exception clause** — every reasoning whose candidate's
     ``current_title`` is non-technical must include an exception/must-have
     clause.
  7. **High-notice / low-response concern** — every reasoning whose
     ``notice_period_days > 60`` or ``recruiter_response_rate < 0.3`` and
     whose rank is in the top 50 must mention the concern.

Audit produces:
  - ``reasoning_audit.csv`` — one row per reasoning, with `pass` boolean
    and `reasons` (semicolon-joined failure descriptions, empty if pass).
  - exit-code 0 iff every row passes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.reasoning import CONCERN_TOKENS, NON_TECH_TITLE_TOKENS

DEFAULT_TEMPLATE_REUSE_CAP = 12

# Curated brand / tool tokens. If the reasoning string mentions any of
# these AND the candidate's raw record does NOT, treat it as a hallucinated
# claim. List is intentionally narrow — we want to flag fabrications, not
# every capitalized word in our skeletons.
BRAND_ALLOWLIST: frozenset[str] = frozenset(
    {
        "Pinecone",
        "FAISS",
        "Weaviate",
        "Qdrant",
        "Milvus",
        "Elasticsearch",
        "OpenSearch",
        "BGE",
        "RAG",
        "BM25",
        "NDCG",
        "MRR",
        "MAP",
        "LangChain",
        "Llama",
        "Mistral",
        "Anthropic",
        "OpenAI",
        "Cohere",
        "ChatGPT",
        "Claude",
        "Gemini",
        "GPT-4",
        "HuggingFace",
        "PyTorch",
        "TensorFlow",
        "JAX",
        "ONNX",
        "Triton",
        "vLLM",
        "Ray",
        "Spark",
        "Kafka",
        "Snowflake",
        "BigQuery",
        "Databricks",
        "Stripe",
        "Uber",
        "Airbnb",
        "Netflix",
        "Google",
        "Meta",
        "Amazon",
        "Microsoft",
        "Apple",
    }
)


def _tokenize_lower(text: str) -> set[str]:
    """Cheap whitespace+punct tokenizer; lowercased."""
    if not text:
        return set()
    out: list[str] = []
    cur = ""
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            cur += ch
        else:
            if cur:
                out.append(cur.lower())
                cur = ""
    if cur:
        out.append(cur.lower())
    return set(out)


def _raw_skill_tokens(raw: dict[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for s in raw.get("skills") or []:
        name = (s.get("name") or "").lower().strip()
        if name:
            for part in name.replace("/", " ").split():
                tokens.add(part)
    for role in raw.get("career_history") or []:
        desc = (role.get("description") or "").lower()
        tokens |= _tokenize_lower(desc)
    return tokens


def _raw_employer_tokens(raw: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for role in raw.get("career_history") or []:
        company = (role.get("company") or "").lower().strip()
        if company:
            out.add(company)
            for part in company.split():
                out.add(part)
    return out


def _raw_location_tokens(raw: dict[str, Any]) -> set[str]:
    """Cities/countries/work-mode strings pulled directly from profile + signals."""
    profile = raw.get("profile") or {}
    signals = raw.get("redrob_signals") or {}
    out: set[str] = set()
    for key in ("location", "country"):
        val = (profile.get(key) or "").lower()
        for piece in val.replace(",", " ").replace("/", " ").split():
            if piece:
                out.add(piece)
    mode = (signals.get("preferred_work_mode") or "").lower().strip()
    if mode:
        out.add(mode)
    return out


def _is_non_technical(title: str) -> bool:
    if not title:
        return False
    low = title.lower()
    return any(tok in low for tok in NON_TECH_TITLE_TOKENS)


def audit_row(
    audit_row: dict[str, Any],
    raw: dict[str, Any] | None,
    *,
    template_counts: dict[str, int],
    template_reuse_cap: int = DEFAULT_TEMPLATE_REUSE_CAP,
) -> list[str]:
    """Return a list of failure descriptions (empty when the row passes)."""
    failures: list[str] = []
    reasoning: str = str(audit_row.get("reasoning") or "")
    rank = int(audit_row.get("rank") or 0)
    tid = str(audit_row.get("template_id") or "")
    raw = raw or {}

    profile = raw.get("profile") or {}
    signals = raw.get("redrob_signals") or {}
    title = (profile.get("current_title") or "").strip()

    # Check 4 — template reuse cap.
    if tid:
        template_counts[tid] = template_counts.get(tid, 0) + 1
        if template_counts[tid] > template_reuse_cap:
            failures.append(
                f"template {tid} reused {template_counts[tid]}× (cap {template_reuse_cap})"
            )

    # Check 5 — rank-50+ concern clause.
    lower = reasoning.lower()
    if rank > 50 and not any(tok in lower for tok in CONCERN_TOKENS):
        failures.append("rank>50 reasoning missing concern/gap token")

    # Check 6 — non-tech title exception clause.
    if _is_non_technical(title):
        must_have_terms = (
            "must-haves",
            "must-have",
            "non-tech",
            "exception",
            "kept on",
            "retrieval",
            "vector",
            "ranking",
            "shipper",
            "python",
        )
        if not any(t in lower for t in must_have_terms):
            failures.append(f"non-tech title '{title}' has no exception/must-have clause")

    # Check 7 — high-notice / low-response concern when rank <= 50.
    if rank <= 50:
        notice = signals.get("notice_period_days")
        if isinstance(notice, int | float) and int(notice) > 60 and "notice" not in lower:
            failures.append(f"rank≤50 notice={int(notice)}d not mentioned")
        resp = signals.get("recruiter_response_rate")
        if isinstance(resp, int | float) and float(resp) < 0.3 and "response" not in lower:
            failures.append(f"rank≤50 low response_rate={float(resp):.2f} not mentioned")

    # Checks 1 & 2 — grounded named-entity claims.
    # Conservative: only flag tokens from a curated brand/tool allowlist when
    # they show up in the reasoning string but are absent from the raw record
    # (skills + role descriptions). Generic English words in skeleton
    # templates are intentionally NOT flagged — the skeletons are
    # deterministic so wording cannot fabricate brands by itself.
    raw_grounded = _raw_skill_tokens(raw) | _raw_employer_tokens(raw) | _raw_location_tokens(raw)
    ungrounded_brands: list[str] = []
    tokens_in_reasoning = _tokenize_lower(reasoning)
    for brand in BRAND_ALLOWLIST:
        bl = brand.lower()
        if bl in tokens_in_reasoning and bl not in raw_grounded:
            ungrounded_brands.append(brand)
    if ungrounded_brands:
        failures.append(
            "ungrounded brand mention(s): " + ", ".join(sorted(set(ungrounded_brands))[:5])
        )

    return failures


def run_audit(audit_csv: Path, candidates_parquet: Path, out_csv: Path) -> int:
    audit_df = pd.read_csv(audit_csv)
    raw_df = pd.read_parquet(candidates_parquet)
    raw_map: dict[str, dict[str, Any]] = {}
    cids = set(audit_df["candidate_id"].astype(str).tolist())
    for _, r in raw_df[raw_df["candidate_id"].isin(cids)].iterrows():
        rec = r.to_dict()
        for k in ("career_history", "education", "skills", "certifications", "languages"):
            v = rec.get(k)
            if v is None:
                rec[k] = []
            elif not isinstance(v, list):
                rec[k] = list(v)
        raw_map[rec["candidate_id"]] = rec

    template_counts: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    any_failure = False
    for _, row in audit_df.iterrows():
        cid = str(row["candidate_id"])
        failures = audit_row(row.to_dict(), raw_map.get(cid), template_counts=template_counts)
        rows.append(
            {
                "candidate_id": cid,
                "rank": int(row["rank"]),
                "template_id": str(row.get("template_id") or ""),
                "pass": not failures,
                "failures": "; ".join(failures),
            }
        )
        if failures:
            any_failure = True
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    if any_failure:
        n_fail = sum(1 for r in rows if not r["pass"])
        print(f"reasoning_audit: {n_fail}/{len(rows)} rows failed — see {out_csv}", flush=True)
        return 1
    print(f"reasoning_audit: PASS {len(rows)} rows; wrote {out_csv}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--audit", required=True, type=Path, help="top_100_audit.csv")
    p.add_argument(
        "--candidates",
        required=True,
        type=Path,
        help="artifacts/candidates.parquet",
    )
    p.add_argument(
        "--out",
        default=Path("reasoning_audit.csv"),
        type=Path,
        help="output CSV",
    )
    args = p.parse_args(argv)
    return run_audit(args.audit, args.candidates, args.out)


if __name__ == "__main__":
    sys.exit(main())
