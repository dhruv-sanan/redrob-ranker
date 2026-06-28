#!/usr/bin/env python3
"""Source a 100-row stratified holdout for CP-5a hand-labeling.

Pulls ~11 candidates per bucket across 9 strata defined to exercise
problem.md §5 + the blocking-check-#10 holdout assertion. The intent is
NOT to be a labeled dataset — only to surface representative profiles
per bucket with their predicate evidence so a human labeler can spend
1.5 hr filling ``true_label`` + ``notes`` columns by inspecting each
candidate in ``candidates.parquet``.

Bucket choices reflect what actually fires in the real 100K data:

    1. plain_language         retrieval_evidence>=0.3 + has_production>=0.5
    2. stuffer                stuffer_risk>=0.6 + retrieval_evidence<0.2
    3. honeypot_drop          honeypot_drop=True
    4. honeypot_audit         honeypot_audit & not drop & risk>=0.45
    5. irrelevant_tail        tier=D + retrieval<0.05 + no honeypot flag
    6. services_only          archetype contains 'services_only'
    7. services_to_product    services-company in history but NOT services_only
    8. outside_india          profile.country != 'India'
    9. non_tech_title         archetype contains 'non_tech_title'

Original CP-5a brief named three additional buckets (CV-Speech-Expert,
Strong-Recsys-Weak-Skill-List, Strong-AI-but-Inactive) that DO NOT
fire on the real candidate distribution (anti_pattern_archetypes
counts at v0.9.3 of the feature pipeline: ``non_tech_title=34339,
services_only=8945`` and no others). Substituting these with three
buckets that exercise real archetype + ledger paths gives the holdout
real assertion power.

Usage:
    python tools/build_holdout.py \\
        --artifacts ./artifacts/ \\
        --out holdout_labels.csv \\
        [--per-bucket 11] [--seed 42] [--force]

By default the script refuses to overwrite an existing ``holdout_labels.csv``
so that hand-filled ``true_label`` / ``notes`` columns are not silently
clobbered. Pass ``--force`` to regenerate from scratch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERVICES_COMPANIES: frozenset[str] = frozenset(
    {
        "tcs",
        "tata consultancy",
        "infosys",
        "wipro",
        "cognizant",
        "accenture",
        "hcl",
        "tech mahindra",
        "capgemini",
        "mindtree",
        "mphasis",
        "ltimindtree",
        "l&t infotech",
        "hexaware",
        "ibm services",
    }
)

BUCKET_ORDER: tuple[str, ...] = (
    "plain_language",
    "stuffer",
    "honeypot_drop",
    "honeypot_audit",
    "irrelevant_tail",
    "services_only",
    "services_to_product",
    "outside_india",
    "non_tech_title",
)

EXPECTED_BAND: dict[str, str] = {
    "plain_language": "rank<=50",
    "stuffer": "rank>100 OR not_in_top_100",
    "honeypot_drop": "dropped (no_rank)",
    "honeypot_audit": "rank>50 OR not_in_top_100",
    "irrelevant_tail": "not_in_top_100",
    "services_only": "rank>50 (ceiling)",
    "services_to_product": "no_ceiling — may rank top-100 if strong",
    "outside_india": "informational (no country penalty)",
    "non_tech_title": "rank>50 (ceiling)",
}

OUT_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "bucket",
    "predicate_evidence",
    "current_tier",
    "current_ceiling",
    "current_rank",
    "expected_band",
    "true_label",
    "notes",
)


def _archetypes_contains(archetypes_field: object, name: str) -> bool:
    if archetypes_field is None:
        return False
    try:
        return name in archetypes_field
    except TypeError:
        return False


def _has_services_history(career_history: object) -> bool:
    if not isinstance(career_history, list | np.ndarray):
        return False
    for role in career_history:
        if not isinstance(role, dict):
            continue
        company = (role.get("company") or "").lower()
        if not company:
            continue
        if any(needle in company for needle in SERVICES_COMPANIES):
            return True
    return False


def _country_outside_india(profile: object) -> bool:
    if not isinstance(profile, dict):
        return False
    country = profile.get("country")
    if not country:
        return False
    return str(country).strip().lower() != "india"


def _evidence_string(row: pd.Series, bucket: str) -> str:
    base = (
        f"tier={row['tier']} "
        f"retrieval={row['retrieval_evidence']:.2f} "
        f"stuffer={row['stuffer_risk']:.2f} "
        f"honey={row['honeypot_risk_score']:.2f}"
    )
    if bucket == "plain_language":
        return base + f" prod_retrieval={row['has_production_retrieval_evidence']:.2f}"
    if bucket == "stuffer":
        return base + " (high stuffer + low retrieval)"
    if bucket == "honeypot_drop":
        return base + " honeypot_drop=True"
    if bucket == "honeypot_audit":
        return base + " honeypot_audit=True drop=False"
    if bucket == "irrelevant_tail":
        return base + " (deep-tail D-tier)"
    if bucket == "services_only":
        return base + " archetype=services_only"
    if bucket == "services_to_product":
        return base + " services_history + not services_only archetype"
    if bucket == "outside_india":
        return base + " country!=India"
    if bucket == "non_tech_title":
        return base + " archetype=non_tech_title"
    return base


BucketPredicate = Callable[[pd.DataFrame, pd.DataFrame], pd.Series]


def _mask_plain_language(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    del candidates
    return (features["retrieval_evidence"] >= 0.30) & (
        features["has_production_retrieval_evidence"] >= 0.50
    )


def _mask_stuffer(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    del candidates
    return (features["stuffer_risk"] >= 0.60) & (features["retrieval_evidence"] < 0.20)


def _mask_honeypot_drop(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    del candidates
    return features["honeypot_drop"].astype(bool)


def _mask_honeypot_audit(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    del candidates
    return (
        features["honeypot_audit"].astype(bool)
        & ~features["honeypot_drop"].astype(bool)
        & (features["honeypot_risk_score"] >= 0.45)
    )


def _mask_irrelevant_tail(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    del candidates
    return (
        (features["tier"] == "D")
        & (features["retrieval_evidence"] < 0.05)
        & ~features["honeypot_audit"].astype(bool)
        & ~features["honeypot_drop"].astype(bool)
    )


def _mask_services_only(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    del candidates
    return features["anti_pattern_archetypes"].apply(
        lambda a: _archetypes_contains(a, "services_only")
    )


def _mask_services_to_product(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    services_only = features["anti_pattern_archetypes"].apply(
        lambda a: _archetypes_contains(a, "services_only")
    )
    has_serv = candidates["career_history"].apply(_has_services_history)
    return has_serv.reset_index(drop=True) & (~services_only.reset_index(drop=True))


def _mask_outside_india(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    del features
    return candidates["profile"].apply(_country_outside_india)


def _mask_non_tech_title(features: pd.DataFrame, candidates: pd.DataFrame) -> pd.Series:
    del candidates
    return features["anti_pattern_archetypes"].apply(
        lambda a: _archetypes_contains(a, "non_tech_title")
    )


BUCKET_PREDICATES: dict[str, BucketPredicate] = {
    "plain_language": _mask_plain_language,
    "stuffer": _mask_stuffer,
    "honeypot_drop": _mask_honeypot_drop,
    "honeypot_audit": _mask_honeypot_audit,
    "irrelevant_tail": _mask_irrelevant_tail,
    "services_only": _mask_services_only,
    "services_to_product": _mask_services_to_product,
    "outside_india": _mask_outside_india,
    "non_tech_title": _mask_non_tech_title,
}


def _read_current_rank(submission_csv: Path) -> dict[str, int]:
    """Return {candidate_id: rank} from an existing top_100_submission.csv if present."""
    if not submission_csv.exists():
        return {}
    df = pd.read_csv(submission_csv)
    return {str(cid): int(r) for cid, r in zip(df["candidate_id"], df["rank"], strict=True)}


def sample_bucket(
    features: pd.DataFrame,
    candidates: pd.DataFrame,
    bucket: str,
    per_bucket: int,
    rng: np.random.Generator,
    already_picked: set[str],
) -> pd.DataFrame:
    """Draw up to ``per_bucket`` rows for a bucket; never re-pick already-claimed candidates."""
    if bucket not in BUCKET_PREDICATES:
        raise KeyError(f"unknown bucket: {bucket}")

    predicate = BUCKET_PREDICATES[bucket]
    mask = predicate(features, candidates).reset_index(drop=True)
    if hasattr(mask, "astype"):
        mask = mask.astype(bool)
    pool = features.loc[mask].copy()
    if already_picked:
        pool = pool.loc[~pool["candidate_id"].isin(already_picked)]
    if pool.empty:
        return pd.DataFrame(columns=list(OUT_COLUMNS))

    pool = pool.sort_values("candidate_id").reset_index(drop=True)
    n_take = min(per_bucket, len(pool))
    idx = rng.choice(len(pool), size=n_take, replace=False)
    idx.sort()
    picks = pool.iloc[idx].copy()

    rows = []
    for _, row in picks.iterrows():
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "bucket": bucket,
                "predicate_evidence": _evidence_string(row, bucket),
                "current_tier": row["tier"],
                "current_ceiling": row["anti_pattern_ceiling"],
                "current_rank": "",
                "expected_band": EXPECTED_BAND[bucket],
                "true_label": "",
                "notes": "",
            }
        )
    return pd.DataFrame(rows, columns=list(OUT_COLUMNS))


def build_holdout(
    artifacts_dir: Path,
    out_csv: Path,
    per_bucket: int = 11,
    seed: int = 42,
    force: bool = False,
) -> pd.DataFrame:
    """Source ``per_bucket`` candidates per bucket; write seed CSV to ``out_csv``.

    Refuses to overwrite an existing ``out_csv`` unless ``force=True`` to
    protect hand-filled label columns.
    """
    if out_csv.exists() and not force:
        raise FileExistsError(
            f"{out_csv} already exists — pass --force to regenerate (will clobber labels)."
        )
    features_path = artifacts_dir / "features.parquet"
    candidates_path = artifacts_dir / "candidates.parquet"
    submission_csv = _REPO_ROOT / "top_100_submission.csv"

    features = pd.read_parquet(features_path)
    candidates = pd.read_parquet(candidates_path)

    if not features["candidate_id"].equals(candidates["candidate_id"]):
        raise ValueError("features.parquet and candidates.parquet row order mismatch")

    rng = np.random.default_rng(seed)
    picked: set[str] = set()
    pieces: list[pd.DataFrame] = []
    for bucket in BUCKET_ORDER:
        piece = sample_bucket(features, candidates, bucket, per_bucket, rng, picked)
        if not piece.empty:
            picked.update(piece["candidate_id"].tolist())
            pieces.append(piece)

    seed_df = (
        pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=list(OUT_COLUMNS))
    )

    rank_map = _read_current_rank(submission_csv)
    if rank_map:
        seed_df["current_rank"] = seed_df["candidate_id"].map(rank_map).astype("Int64")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    seed_df.to_csv(out_csv, index=False)
    return seed_df


def _parse_argv(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CP-5a stratified holdout seed CSV.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("holdout_labels.csv"))
    parser.add_argument("--per-bucket", type=int, default=11)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing out CSV (will clobber any hand-filled labels)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_argv(argv or sys.argv[1:])
    seed_df = build_holdout(args.artifacts, args.out, args.per_bucket, args.seed, args.force)
    counts = seed_df.groupby("bucket", sort=False).size().to_dict()
    total = len(seed_df)
    print(f"[build_holdout] wrote {args.out} ({total} rows)")
    for bucket in BUCKET_ORDER:
        print(f"  {bucket}: {counts.get(bucket, 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
