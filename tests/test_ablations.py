"""CP-5b tests — variant dispatch, mutation isolation, runtime helpers."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.ablations import (
    VARIANTS,
    Variant,
    _no_gate_top_100,
    _strip_behavioral,
    _strip_ceiling,
    overlap_metrics,
    run_variant,
    top_k_stability,
)
from tools.runtime_report import (
    ARTIFACT_FILES,
    _bytes_to_mb,
    load_build_summary,
    measure_artifacts,
)


def test_variants_have_six_codes_a0_through_a5():
    codes = [v.code for v in VARIANTS]
    assert codes == ["A0", "A1", "A2", "A3", "A4", "A5"]
    assert all(isinstance(v, Variant) for v in VARIANTS)


def test_overlap_metrics_identical_sets():
    ids = ["A", "B", "C", "D"]
    m = overlap_metrics(ids, ids)
    assert m["overlap_count"] == 4.0
    assert m["overlap_pct"] == 100.0
    assert m["jaccard"] == 1.0


def test_overlap_metrics_disjoint_sets():
    m = overlap_metrics(["A", "B"], ["C", "D"])
    assert m["overlap_count"] == 0.0
    assert m["overlap_pct"] == 0.0
    assert m["jaccard"] == 0.0


def test_overlap_metrics_partial_intersection():
    m = overlap_metrics(["A", "B", "C", "D"], ["C", "D", "E", "F"])
    assert m["overlap_count"] == 2.0
    assert m["overlap_pct"] == 50.0
    assert m["jaccard"] == pytest.approx(2.0 / 6.0)


def test_top_k_stability_rank_aware():
    base = ["A", "B", "C", "D"]
    var = ["A", "X", "C", "D"]
    assert top_k_stability(base, var, 2) == 1
    assert top_k_stability(base, var, 4) == 3


def test_strip_behavioral_overrides_final_score():
    df = pd.DataFrame(
        {
            "base_score": [0.5, 0.3, 0.1],
            "final_score_uncapped": [0.9, 0.6, 0.2],
            "final_score": [0.8, 0.5, 0.2],
        }
    )
    stripped = _strip_behavioral(df)
    np.testing.assert_array_equal(
        stripped["final_score"].to_numpy(), df["base_score"].astype(np.float32).to_numpy()
    )
    # Original untouched
    assert df.loc[0, "final_score"] == 0.8


def test_strip_ceiling_replaces_with_uncapped():
    df = pd.DataFrame(
        {
            "base_score": [0.5, 0.3, 0.1],
            "final_score_uncapped": [0.95, 0.7, 0.2],
            "final_score": [0.6, 0.6, 0.2],
        }
    )
    stripped = _strip_ceiling(df)
    np.testing.assert_array_equal(
        stripped["final_score"].to_numpy(),
        df["final_score_uncapped"].astype(np.float32).to_numpy(),
    )


def _make_scored_frame(n: int = 20) -> pd.DataFrame:
    ids = [f"CAND_{i:04d}" for i in range(n)]
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    return pd.DataFrame(
        {
            "candidate_id": ids,
            "tier": ["A" if i < 5 else "C" for i in range(n)],
            "tier_priority": [0 if i < 5 else 2 for i in range(n)],
            "final_score": scores,
            "honeypot_drop": [False] * n,
            "honeypot_audit": [False] * n,
        }
    )


def test_no_gate_top_100_takes_top_by_score_within_tier():
    df = _make_scored_frame(15)
    top = _no_gate_top_100(df, target=10)
    assert len(top) == 10
    assert list(top["rank"]) == list(range(1, 11))
    a_tier_first = (top["tier"] == "A").iloc[:5].all()
    assert a_tier_first
    scores = top["score"].to_numpy()
    assert np.all(scores[:-1] >= scores[1:]), "scores must be monotonic non-increasing"


def test_no_gate_drops_honeypots():
    df = _make_scored_frame(15)
    df.loc[0, "honeypot_drop"] = True
    top = _no_gate_top_100(df, target=10)
    assert df.loc[0, "candidate_id"] not in top["candidate_id"].values


def test_run_variant_unknown_code_raises():
    with pytest.raises(KeyError):
        run_variant(
            "Z9",
            pd.DataFrame(),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            {"blend": {}},
            None,
        )


def test_run_variant_mutation_does_not_leak_to_caller_weights():
    weights = {
        "blend": {
            "embedding_contribution": 0.5,
            "skill_contribution": 0.5,
            "title_career_fit": 0.0,
            "retrieval_evidence": 0.0,
            "must_have_sum_div_6": 0.0,
            "experience_band_fit": 0.0,
            "education_signal": 0.0,
            "external_validation_signal": 0.0,
        },
        "capped_factor_emb": {
            "technical_floor": {"title_career_fit": 0.4, "retrieval_evidence": 0.2},
            "cap_when_below": 0.2,
        },
        "capped_factor_skill": {
            "technical_floor": {"title_career_fit": 0.4, "retrieval_evidence": 0.0},
            "cap_when_below": 0.3,
        },
    }
    weights_copy = copy.deepcopy(weights)
    features = _make_realistic_features(8)
    candidate_emb = np.random.default_rng(0).standard_normal((8, 4)).astype(np.float32)
    candidate_emb /= np.linalg.norm(candidate_emb, axis=1, keepdims=True)
    jd = candidate_emb[:2].copy()
    run_variant("A1", features, candidate_emb, jd, weights, None)
    assert weights == weights_copy, "A1 must deep-copy weights before mutating"


def _make_realistic_features(n: int) -> pd.DataFrame:
    ids = [f"CAND_{i:04d}" for i in range(n)]
    return pd.DataFrame(
        {
            "candidate_id": ids,
            "tier": ["A"] * 2 + ["B"] * 2 + ["C"] * 2 + ["D"] * 2,
            "tier_priority": [0, 0, 1, 1, 2, 2, 3, 3],
            "retrieval_evidence": np.linspace(0.0, 0.5, n).astype(np.float32),
            "title_career_fit": np.linspace(0.0, 0.6, n).astype(np.float32),
            "skill_depth_trust": np.linspace(0.2, 0.8, n).astype(np.float32),
            "experience_band_fit": np.full(n, 0.7, dtype=np.float32),
            "education_signal": np.full(n, 0.5, dtype=np.float32),
            "availability_signal": np.full(n, 0.95, dtype=np.float32),
            "contactability_signal": np.full(n, True),
            "market_interest_signal": np.full(n, 0.5, dtype=np.float32),
            "external_validation_signal": np.full(n, 0.02, dtype=np.float32),
            "logistics_multiplier": np.full(n, 1.0, dtype=np.float32),
            "logistics_top_10_eligible": np.full(n, True),
            "honeypot_risk_score": np.full(n, 0.1, dtype=np.float32),
            "honeypot_drop": np.full(n, False),
            "honeypot_audit": np.full(n, False),
            "stuffer_risk": np.full(n, 0.1, dtype=np.float32),
            "anti_pattern_ceiling": ["none"] * (n - 2) + ["rank_50"] * 2,
            "anti_pattern_archetypes": [[] for _ in range(n)],
            "has_production_retrieval_evidence": np.full(n, 0.7, dtype=np.float32),
            "has_vector_or_hybrid_search_evidence": np.full(n, 0.5, dtype=np.float32),
            "has_python_backend_depth": np.full(n, 0.6, dtype=np.float32),
            "has_ranking_eval_evidence": np.full(n, 0.4, dtype=np.float32),
            "has_product_company_applied_ml_context": np.full(n, 1.0, dtype=np.float32),
            "has_shipper_signal": np.full(n, 0.6, dtype=np.float32),
        }
    )


@pytest.mark.parametrize("variant_code", ["A0", "A1", "A2", "A3", "A4", "A5"])
def test_run_variant_returns_dataframe_with_candidate_id_and_rank(variant_code: str):
    n = 30
    features = _make_realistic_features(8)
    features = pd.concat([features] * 4, ignore_index=True).iloc[:n].reset_index(drop=True)
    features["candidate_id"] = [f"CAND_{i:04d}" for i in range(n)]
    candidate_emb = np.random.default_rng(1).standard_normal((n, 6)).astype(np.float32)
    candidate_emb /= np.linalg.norm(candidate_emb, axis=1, keepdims=True)
    jd = candidate_emb[:3].copy()
    weights = {
        "blend": {
            "title_career_fit": 0.2,
            "skill_contribution": 0.2,
            "retrieval_evidence": 0.2,
            "embedding_contribution": 0.2,
            "must_have_sum_div_6": 0.1,
            "experience_band_fit": 0.05,
            "education_signal": 0.025,
            "external_validation_signal": 0.025,
        },
        "capped_factor_emb": {
            "technical_floor": {"title_career_fit": 0.4, "retrieval_evidence": 0.2},
            "cap_when_below": 0.2,
        },
        "capped_factor_skill": {
            "technical_floor": {"title_career_fit": 0.4, "retrieval_evidence": 0.0},
            "cap_when_below": 0.3,
        },
    }
    top = run_variant(variant_code, features, candidate_emb, jd, weights, None)
    assert "candidate_id" in top.columns
    assert "rank" in top.columns
    assert len(top) > 0


def test_bytes_to_mb_conversion():
    assert _bytes_to_mb(1024 * 1024) == pytest.approx(1.0)
    assert _bytes_to_mb(0) == 0.0


def test_load_build_summary_missing_returns_empty(tmp_path: Path):
    assert load_build_summary(tmp_path) == {}


def test_load_build_summary_reads_json(tmp_path: Path):
    payload = {"tier_histogram": {"A": 1, "B": 2}, "honeypot_drop_count": 5}
    (tmp_path / "build_features_summary.json").write_text(json.dumps(payload))
    assert load_build_summary(tmp_path) == payload


def test_measure_artifacts_records_per_file_sizes(tmp_path: Path):
    for name in ARTIFACT_FILES:
        (tmp_path / name).write_bytes(b"x" * 1024)
    model = tmp_path / "model"
    model.mkdir()
    (model / "foo.bin").write_bytes(b"x" * 2048)
    sizes = measure_artifacts(tmp_path)
    assert sizes["per_file"]["manifest.json"] == 1024
    assert sizes["per_file"]["model/ (recursive)"] == 2048
    assert sizes["total_no_model"] == 1024 * len(ARTIFACT_FILES)
    assert sizes["total_with_model"] == sizes["total_no_model"] + 2048


def test_measure_artifacts_missing_files_skipped(tmp_path: Path):
    sizes = measure_artifacts(tmp_path)
    assert sizes["per_file"]["model/ (recursive)"] == 0
    assert sizes["total_no_model"] == 0
