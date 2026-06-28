"""Capped-contribution linear blend + anti-pattern score ceilings.

Pure numpy / pandas — no JSON, no regex, no date math, no model load. This
module is the heart of the online step and must run in well under a second
on 100K rows.

Pipeline overview (matches problem.md §2.4 – §2.10):

  1. ``embedding_similarity = max(candidate_emb @ jd_intent_vecs.T, axis=1)``
  2. ``embedding_contribution = embedding_similarity * capped_factor_emb``
     where ``capped_factor_emb`` collapses to 0.2 unless the row has
     ``title_career_fit >= 0.4`` OR ``retrieval_evidence >= 0.2``.
  3. ``skill_contribution = skill_depth_trust * capped_factor_skill``
     where ``capped_factor_skill`` collapses to 0.3 unless the row has
     ``retrieval_evidence > 0`` OR ``title_career_fit >= 0.4``.
  4. Linear blend ``base`` over 8 weighted scalar terms.
  5. ``final = base * availability_signal * logistics_multiplier``
     ``+ market_interest_signal * MARKET_TIEBREAK_WEIGHT``.
  6. Apply hard score ceilings derived from the current distribution:
     - ``rank_50`` ceiling rows are clipped just below the 50th-best score.
     - ``rank_100`` ceiling rows are clipped just below the 100th-best score.

The output DataFrame *augments* the input with derived columns, never drops
rows. Row removal (honeypot drop, etc.) is the ranking layer's job.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MARKET_TIEBREAK_WEIGHT = 0.01

MUST_HAVE_COLUMNS: tuple[str, ...] = (
    "has_production_retrieval_evidence",
    "has_vector_or_hybrid_search_evidence",
    "has_python_backend_depth",
    "has_ranking_eval_evidence",
    "has_product_company_applied_ml_context",
    "has_shipper_signal",
)


def compute_embedding_similarity(
    candidate_emb: np.ndarray, jd_intent_vecs: np.ndarray
) -> np.ndarray:
    """Return per-candidate max cosine to any JD intent.

    Both inputs are expected to be unit-normed; the dot product IS the cosine.
    `candidate_emb.shape == (N, D)`, `jd_intent_vecs.shape == (K, D)` → output
    shape `(N,)` of float32. Stored as float16 on disk; we widen for math.
    """
    cand = candidate_emb.astype(np.float32, copy=False)
    jd = jd_intent_vecs.astype(np.float32, copy=False)
    sims = cand @ jd.T  # (N, K)
    out = sims.max(axis=1)
    # Numerical noise can push slightly outside [-1, 1]; clip for safety.
    return np.clip(out, 0.0, 1.0)


def capped_factor_emb(features: pd.DataFrame, weights: dict[str, Any]) -> np.ndarray:
    """Return per-row 1.0 / 0.2 cap on the embedding channel.

    1.0 iff ``title_career_fit >= threshold_tcf`` OR
            ``retrieval_evidence >= threshold_re``; else `cap_when_below`.
    """
    cfg = weights["capped_factor_emb"]
    tcf_floor = float(cfg["technical_floor"]["title_career_fit"])
    re_floor = float(cfg["technical_floor"]["retrieval_evidence"])
    cap_low = float(cfg["cap_when_below"])
    eligible = (features["title_career_fit"].to_numpy() >= tcf_floor) | (
        features["retrieval_evidence"].to_numpy() >= re_floor
    )
    return np.where(eligible, 1.0, cap_low).astype(np.float32)


def capped_factor_skill(features: pd.DataFrame, weights: dict[str, Any]) -> np.ndarray:
    """Return per-row 1.0 / 0.3 cap on the skill channel.

    1.0 iff ``retrieval_evidence > threshold_re`` OR
            ``title_career_fit >= threshold_tcf``; else `cap_when_below`.
    """
    cfg = weights["capped_factor_skill"]
    tcf_floor = float(cfg["technical_floor"]["title_career_fit"])
    re_floor = float(cfg["technical_floor"]["retrieval_evidence"])
    cap_low = float(cfg["cap_when_below"])
    eligible = (features["retrieval_evidence"].to_numpy() > re_floor) | (
        features["title_career_fit"].to_numpy() >= tcf_floor
    )
    return np.where(eligible, 1.0, cap_low).astype(np.float32)


def must_have_sum_div_6(features: pd.DataFrame) -> np.ndarray:
    """Sum of the 6 must-have evidence scalars, divided by 6 (= mean)."""
    arr = features[list(MUST_HAVE_COLUMNS)].to_numpy().astype(np.float32, copy=False)
    return arr.sum(axis=1) / float(len(MUST_HAVE_COLUMNS))


def linear_blend(features: pd.DataFrame, weights: dict[str, Any]) -> np.ndarray:
    """Return the per-row base score (pre multipliers, pre ceilings).

    Output is float32; ranges roughly [0, 1.05] but is NOT clipped — callers
    layer multipliers + ceilings on top.
    """
    w = weights["blend"]
    f = features
    must_sum6 = must_have_sum_div_6(f)
    skill_contrib = f["skill_depth_trust"].to_numpy() * f["capped_factor_skill"].to_numpy()
    emb_contrib = f["embedding_similarity"].to_numpy() * f["capped_factor_emb"].to_numpy()
    base = (
        float(w["title_career_fit"]) * f["title_career_fit"].to_numpy()
        + float(w["skill_contribution"]) * skill_contrib
        + float(w["retrieval_evidence"]) * f["retrieval_evidence"].to_numpy()
        + float(w["embedding_contribution"]) * emb_contrib
        + float(w["must_have_sum_div_6"]) * must_sum6
        + float(w["experience_band_fit"]) * f["experience_band_fit"].to_numpy()
        + float(w["education_signal"]) * f["education_signal"].to_numpy()
        + float(w["external_validation_signal"]) * f["external_validation_signal"].to_numpy()
    ).astype(np.float32)
    return base


def apply_multipliers_and_tiebreak(features: pd.DataFrame, base: np.ndarray) -> np.ndarray:
    """`final = base * availability_signal * logistics_multiplier + 0.01 * market_interest`.

    The tiebreak boost is tiny by construction — it only orders score-tied
    candidates within their tier band.
    """
    avail = features["availability_signal"].to_numpy().astype(np.float32, copy=False)
    logist = features["logistics_multiplier"].to_numpy().astype(np.float32, copy=False)
    market = features["market_interest_signal"].to_numpy().astype(np.float32, copy=False)
    final = base * avail * logist + MARKET_TIEBREAK_WEIGHT * market
    return final.astype(np.float32)


def _ceiling_threshold(scores: np.ndarray, k: int) -> float:
    """Return the k-th highest score in `scores` (1-indexed); 0 if shorter."""
    if scores.size == 0:
        return 0.0
    k = max(1, min(k, scores.size))
    # argpartition top-k then sort within for stability.
    return float(np.sort(scores)[-k])


def apply_anti_pattern_ceilings(
    features: pd.DataFrame, final_uncapped: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Clip rank_50 / rank_100 ceiling rows to just below the rank-band score.

    Returns (capped_scores, thresholds) where thresholds is the float
    score-at-rank-51 / score-at-rank-101 (i.e. one below the band cap).
    """
    ceilings = features["anti_pattern_ceiling"].to_numpy()
    valid_mask = ~features["honeypot_drop"].to_numpy()
    valid_scores = final_uncapped[valid_mask]
    # 1-indexed rank target r maps to score-at-rank-r = sorted_desc[r-1].
    # To force "cannot enter top R", we clip down to score-at-rank-(R+1).
    thr_50 = _ceiling_threshold(valid_scores, k=51)
    thr_100 = _ceiling_threshold(valid_scores, k=101)
    capped = final_uncapped.copy()
    is_rank_50 = ceilings == "rank_50"
    is_rank_100 = ceilings == "rank_100"
    capped[is_rank_50] = np.minimum(capped[is_rank_50], thr_50)
    capped[is_rank_100] = np.minimum(capped[is_rank_100], thr_100)
    return capped, {
        "score_at_rank_51": thr_50,
        "score_at_rank_101": thr_100,
        "rank_50_clipped_count": int(is_rank_50.sum()),
        "rank_100_clipped_count": int(is_rank_100.sum()),
    }


def compute_scores(
    features: pd.DataFrame,
    candidate_emb: np.ndarray,
    jd_intent_vecs: np.ndarray,
    weights: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Augment `features` with derived columns; return (df, telemetry).

    Added columns: embedding_similarity, capped_factor_emb, capped_factor_skill,
    embedding_contribution, skill_contribution, must_have_sum_div_6,
    base_score, final_score_uncapped, final_score.

    Telemetry: ceiling thresholds, ceiling-clipped counts.
    """
    if len(features) != candidate_emb.shape[0]:
        raise ValueError(
            f"features row count {len(features)} != " f"candidate_emb rows {candidate_emb.shape[0]}"
        )

    df = features.copy()
    df["embedding_similarity"] = compute_embedding_similarity(candidate_emb, jd_intent_vecs)
    df["capped_factor_emb"] = capped_factor_emb(df, weights)
    df["capped_factor_skill"] = capped_factor_skill(df, weights)
    df["embedding_contribution"] = (
        df["embedding_similarity"].to_numpy() * df["capped_factor_emb"].to_numpy()
    )
    df["skill_contribution"] = (
        df["skill_depth_trust"].to_numpy() * df["capped_factor_skill"].to_numpy()
    )
    df["must_have_sum_div_6"] = must_have_sum_div_6(df)
    df["base_score"] = linear_blend(df, weights)
    df["final_score_uncapped"] = apply_multipliers_and_tiebreak(df, df["base_score"].to_numpy())
    capped, thresholds = apply_anti_pattern_ceilings(df, df["final_score_uncapped"].to_numpy())
    df["final_score"] = capped
    return df, thresholds
