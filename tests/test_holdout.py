"""CP-5a tests — bucket predicates, sampling determinism, and report assertions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tools.build_holdout import (
    BUCKET_ORDER,
    BUCKET_PREDICATES,
    OUT_COLUMNS,
    _country_outside_india,
    _evidence_string,
    _has_services_history,
    _mask_honeypot_audit,
    _mask_honeypot_drop,
    _mask_irrelevant_tail,
    _mask_non_tech_title,
    _mask_outside_india,
    _mask_plain_language,
    _mask_services_only,
    _mask_services_to_product,
    _mask_stuffer,
    build_holdout,
    sample_bucket,
)
from tools.holdout_report import (
    VALID_LABELS,
    AssertionResult,
    _bucket_summary_table,
    _frac_in_top_n,
    _join_with_ranks,
    _label_summary_table,
    _median_rank_in_top_100,
    label_grounded_assertions,
    predicate_assertions,
    run_report,
)


def _ids(n: int) -> list[str]:
    return [f"CAND_{i:07d}" for i in range(1, n + 1)]


def _features_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "candidate_id": "CAND_X",
        "retrieval_evidence": 0.0,
        "stuffer_risk": 0.0,
        "honeypot_drop": False,
        "honeypot_audit": False,
        "honeypot_risk_score": 0.0,
        "has_production_retrieval_evidence": 0.0,
        "tier": "D",
        "anti_pattern_ceiling": "none",
        "anti_pattern_archetypes": [],
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_bucket_order_matches_predicates():
    assert set(BUCKET_ORDER) == set(BUCKET_PREDICATES.keys())
    assert len(BUCKET_ORDER) == 9


def test_out_columns_contains_required_fields():
    for col in ("candidate_id", "bucket", "true_label", "notes", "expected_band"):
        assert col in OUT_COLUMNS


def test_mask_plain_language():
    f = _features_frame(
        [
            {
                "candidate_id": "A",
                "retrieval_evidence": 0.50,
                "has_production_retrieval_evidence": 0.80,
            },
            {
                "candidate_id": "B",
                "retrieval_evidence": 0.50,
                "has_production_retrieval_evidence": 0.00,
            },
            {
                "candidate_id": "C",
                "retrieval_evidence": 0.20,
                "has_production_retrieval_evidence": 0.80,
            },
        ]
    )
    assert _mask_plain_language(f, pd.DataFrame()).tolist() == [True, False, False]


def test_mask_stuffer():
    f = _features_frame(
        [
            {"candidate_id": "A", "stuffer_risk": 0.70, "retrieval_evidence": 0.10},
            {"candidate_id": "B", "stuffer_risk": 0.50, "retrieval_evidence": 0.10},
            {"candidate_id": "C", "stuffer_risk": 0.80, "retrieval_evidence": 0.30},
        ]
    )
    assert _mask_stuffer(f, pd.DataFrame()).tolist() == [True, False, False]


def test_mask_honeypot_drop():
    f = _features_frame(
        [
            {"candidate_id": "A", "honeypot_drop": True},
            {"candidate_id": "B", "honeypot_drop": False},
        ]
    )
    assert _mask_honeypot_drop(f, pd.DataFrame()).tolist() == [True, False]


def test_mask_honeypot_audit_excludes_drop_and_low_risk():
    f = _features_frame(
        [
            {
                "candidate_id": "A",
                "honeypot_audit": True,
                "honeypot_drop": False,
                "honeypot_risk_score": 0.50,
            },
            {
                "candidate_id": "B",
                "honeypot_audit": True,
                "honeypot_drop": True,
                "honeypot_risk_score": 0.70,
            },
            {
                "candidate_id": "C",
                "honeypot_audit": True,
                "honeypot_drop": False,
                "honeypot_risk_score": 0.30,
            },
            {
                "candidate_id": "D",
                "honeypot_audit": False,
                "honeypot_drop": False,
                "honeypot_risk_score": 0.50,
            },
        ]
    )
    assert _mask_honeypot_audit(f, pd.DataFrame()).tolist() == [True, False, False, False]


def test_mask_irrelevant_tail():
    f = _features_frame(
        [
            {"candidate_id": "A", "tier": "D", "retrieval_evidence": 0.01},
            {"candidate_id": "B", "tier": "D", "retrieval_evidence": 0.10},
            {"candidate_id": "C", "tier": "C", "retrieval_evidence": 0.01},
            {"candidate_id": "D", "tier": "D", "retrieval_evidence": 0.01, "honeypot_audit": True},
            {"candidate_id": "E", "tier": "D", "retrieval_evidence": 0.01, "honeypot_drop": True},
        ]
    )
    assert _mask_irrelevant_tail(f, pd.DataFrame()).tolist() == [True, False, False, False, False]


def test_mask_services_only():
    f = _features_frame(
        [
            {"candidate_id": "A", "anti_pattern_archetypes": ["services_only"]},
            {"candidate_id": "B", "anti_pattern_archetypes": ["non_tech_title"]},
            {"candidate_id": "C", "anti_pattern_archetypes": []},
        ]
    )
    assert _mask_services_only(f, pd.DataFrame()).tolist() == [True, False, False]


def test_mask_services_to_product_requires_history_minus_archetype():
    f = _features_frame(
        [
            {"candidate_id": "A", "anti_pattern_archetypes": []},
            {"candidate_id": "B", "anti_pattern_archetypes": ["services_only"]},
            {"candidate_id": "C", "anti_pattern_archetypes": []},
        ]
    )
    c = pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C"],
            "career_history": [
                [{"company": "Tata Consultancy Services"}, {"company": "Stripe"}],
                [{"company": "Infosys"}],
                [{"company": "Stripe"}],
            ],
        }
    )
    mask = _mask_services_to_product(f, c)
    assert mask.tolist() == [True, False, False]


def test_mask_outside_india():
    f = _features_frame([{"candidate_id": x} for x in ("A", "B", "C", "D")])
    c = pd.DataFrame(
        {
            "candidate_id": ["A", "B", "C", "D"],
            "profile": [
                {"country": "India"},
                {"country": "USA"},
                {"country": None},
                None,
            ],
        }
    )
    mask = _mask_outside_india(f, c)
    assert mask.tolist() == [False, True, False, False]


def test_mask_non_tech_title():
    f = _features_frame(
        [
            {"candidate_id": "A", "anti_pattern_archetypes": ["non_tech_title"]},
            {"candidate_id": "B", "anti_pattern_archetypes": ["non_tech_title", "services_only"]},
            {"candidate_id": "C", "anti_pattern_archetypes": []},
        ]
    )
    assert _mask_non_tech_title(f, pd.DataFrame()).tolist() == [True, True, False]


def test_has_services_history_case_insensitive():
    assert _has_services_history([{"company": "TATA Consultancy"}]) is True
    assert _has_services_history([{"company": "Wipro Limited"}]) is True
    assert _has_services_history([{"company": "Stripe Inc."}]) is False
    assert _has_services_history(None) is False
    assert _has_services_history([{"company": None}]) is False


def test_country_outside_india_handles_missing_and_blank():
    assert _country_outside_india({"country": "USA"}) is True
    assert _country_outside_india({"country": "India"}) is False
    assert _country_outside_india({"country": "  india  "}) is False
    assert _country_outside_india({"country": None}) is False
    assert _country_outside_india({}) is False
    assert _country_outside_india(None) is False


def test_evidence_string_contains_signals():
    row = pd.Series(
        {
            "tier": "C",
            "retrieval_evidence": 0.42,
            "stuffer_risk": 0.10,
            "honeypot_risk_score": 0.30,
            "has_production_retrieval_evidence": 0.80,
        }
    )
    s = _evidence_string(row, "plain_language")
    assert "tier=C" in s
    assert "retrieval=0.42" in s
    assert "prod_retrieval=0.80" in s


def test_sample_bucket_deterministic_same_seed():
    ids = _ids(20)
    f = _features_frame(
        [{"candidate_id": cid, "stuffer_risk": 0.7, "retrieval_evidence": 0.05} for cid in ids]
    )
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    picks_a = sample_bucket(f, pd.DataFrame(), "stuffer", 5, rng_a, set())
    picks_b = sample_bucket(f, pd.DataFrame(), "stuffer", 5, rng_b, set())
    assert picks_a["candidate_id"].tolist() == picks_b["candidate_id"].tolist()
    assert len(picks_a) == 5


def test_sample_bucket_excludes_already_picked():
    ids = _ids(5)
    f = _features_frame(
        [{"candidate_id": cid, "stuffer_risk": 0.7, "retrieval_evidence": 0.05} for cid in ids]
    )
    rng = np.random.default_rng(42)
    picks = sample_bucket(f, pd.DataFrame(), "stuffer", 3, rng, already_picked={ids[0], ids[1]})
    assert set(picks["candidate_id"]).isdisjoint({ids[0], ids[1]})
    assert len(picks) == 3


def test_sample_bucket_empty_pool_returns_empty_frame():
    f = _features_frame([{"candidate_id": "X", "stuffer_risk": 0.0}])
    rng = np.random.default_rng(0)
    picks = sample_bucket(f, pd.DataFrame(), "stuffer", 5, rng, set())
    assert picks.empty
    assert list(picks.columns) == list(OUT_COLUMNS)


def test_sample_bucket_unknown_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(KeyError):
        sample_bucket(
            _features_frame([{"candidate_id": "A"}]), pd.DataFrame(), "nope", 1, rng, set()
        )


def test_build_holdout_writes_csv_with_schema(tmp_path: Path):
    n = 60
    ids = _ids(n)
    features = _features_frame(
        [
            {
                "candidate_id": cid,
                "retrieval_evidence": 0.50 if i < 15 else 0.05,
                "has_production_retrieval_evidence": 0.80 if i < 15 else 0.0,
                "stuffer_risk": 0.70 if 15 <= i < 30 else 0.0,
                "honeypot_drop": 30 <= i < 45,
                "honeypot_audit": (30 <= i < 45) or (i >= 50),
                "honeypot_risk_score": 0.50,
                "tier": "D" if i >= 45 else "C",
                "anti_pattern_archetypes": (
                    ["non_tech_title"] if i % 3 == 0 else ["services_only"] if i % 3 == 1 else []
                ),
            }
            for i, cid in enumerate(ids)
        ]
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ids,
            "profile": [{"country": "India" if i % 2 == 0 else "USA"} for i in range(n)],
            "career_history": [
                [{"company": "Stripe"}] if i % 4 != 0 else [{"company": "Infosys"}]
                for i in range(n)
            ],
        }
    )
    arts = tmp_path / "arts"
    arts.mkdir()
    features.to_parquet(arts / "features.parquet")
    candidates.to_parquet(arts / "candidates.parquet")
    out_csv = tmp_path / "seed.csv"

    df = build_holdout(arts, out_csv, per_bucket=3, seed=42)
    assert out_csv.exists()
    on_disk = pd.read_csv(out_csv)
    assert list(on_disk.columns) == list(OUT_COLUMNS)
    assert (df["candidate_id"].value_counts() == 1).all(), "no duplicates across buckets"
    assert df["bucket"].isin(BUCKET_ORDER).all()


def test_build_holdout_refuses_overwrite_without_force(tmp_path: Path):
    features = _features_frame(
        [{"candidate_id": "A", "stuffer_risk": 0.8, "retrieval_evidence": 0.0}]
    )
    candidates = pd.DataFrame(
        {
            "candidate_id": ["A"],
            "profile": [{"country": "India"}],
            "career_history": [[{"company": "Stripe"}]],
        }
    )
    arts = tmp_path / "arts"
    arts.mkdir()
    features.to_parquet(arts / "features.parquet")
    candidates.to_parquet(arts / "candidates.parquet")
    out = tmp_path / "labels.csv"
    out.write_text("pre-existing,user,labels\n")
    with pytest.raises(FileExistsError, match="--force"):
        build_holdout(arts, out, per_bucket=1, seed=0)
    # force=True overwrites
    build_holdout(arts, out, per_bucket=1, seed=0, force=True)
    assert "candidate_id" in out.read_text().splitlines()[0]


def test_build_holdout_rejects_mismatched_row_order(tmp_path: Path):
    features = _features_frame([{"candidate_id": "A"}, {"candidate_id": "B"}])
    candidates = pd.DataFrame(
        {
            "candidate_id": ["B", "A"],
            "profile": [{"country": "India"}, {"country": "India"}],
            "career_history": [[{"company": "Stripe"}], [{"company": "Stripe"}]],
        }
    )
    arts = tmp_path / "arts"
    arts.mkdir()
    features.to_parquet(arts / "features.parquet")
    candidates.to_parquet(arts / "candidates.parquet")
    with pytest.raises(ValueError, match="row order mismatch"):
        build_holdout(arts, tmp_path / "out.csv", per_bucket=1, seed=0)


def test_join_with_ranks_attaches_rank_for_matches():
    seed = pd.DataFrame({"candidate_id": ["A", "B", "C"], "bucket": ["x", "y", "z"]})
    sub = pd.DataFrame({"candidate_id": ["A", "C"], "rank": [5, 80]})
    joined = _join_with_ranks(seed, sub)
    assert joined.loc[0, "current_rank"] == 5
    assert pd.isna(joined.loc[1, "current_rank"])
    assert joined.loc[2, "current_rank"] == 80


def test_frac_in_top_n_counts_only_ranked():
    df = pd.DataFrame({"current_rank": [5.0, 50.0, 150.0, np.nan]})
    assert _frac_in_top_n(df, 100) == pytest.approx(0.5)
    assert _frac_in_top_n(df, 10) == pytest.approx(0.25)
    assert _frac_in_top_n(pd.DataFrame({"current_rank": []}), 100) == 0.0


def test_median_rank_returns_none_when_no_ranked():
    df = pd.DataFrame({"current_rank": [np.nan, np.nan]})
    assert _median_rank_in_top_100(df) is None


def test_median_rank_excludes_nan():
    df = pd.DataFrame({"current_rank": [10.0, 30.0, np.nan]})
    assert _median_rank_in_top_100(df) == 20.0


def _joined_frame(rows: list[dict]) -> pd.DataFrame:
    defaults = {"bucket": "plain_language", "current_rank": np.nan, "true_label": ""}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_predicate_assertions_fail_when_honeypot_drop_in_top_100():
    joined = _joined_frame([{"bucket": "honeypot_drop", "current_rank": 50.0}])
    results = predicate_assertions(joined)
    honeypot = next(r for r in results if "honeypot_drop" in r.name)
    assert not honeypot.passed


def test_predicate_assertions_pass_when_honeypot_drop_absent_from_top_100():
    joined = _joined_frame([{"bucket": "honeypot_drop", "current_rank": np.nan} for _ in range(3)])
    results = predicate_assertions(joined)
    honeypot = next(r for r in results if "honeypot_drop" in r.name)
    assert honeypot.passed


def test_predicate_assertions_stuffer_fail_when_many_in_top_100():
    rows = [{"bucket": "stuffer", "current_rank": 50.0} for _ in range(3)]
    rows += [{"bucket": "stuffer", "current_rank": np.nan} for _ in range(7)]
    joined = _joined_frame(rows)
    results = predicate_assertions(joined)
    stuffer = next(r for r in results if r.name.startswith("stuffer"))
    assert not stuffer.passed


def test_label_grounded_skipped_when_empty():
    joined = _joined_frame([{"bucket": "plain_language", "true_label": ""} for _ in range(3)])
    assert label_grounded_assertions(joined) == []


def test_label_grounded_fit_assertion_runs():
    joined = _joined_frame(
        [
            {"bucket": "plain_language", "true_label": "fit", "current_rank": 10.0},
            {"bucket": "plain_language", "true_label": "fit", "current_rank": 12.0},
        ]
    )
    results = label_grounded_assertions(joined)
    fit = next(r for r in results if "label=fit" in r.name)
    assert fit.passed


def test_label_grounded_unknown_label_fails():
    joined = _joined_frame([{"bucket": "plain_language", "true_label": "WAT", "current_rank": 5.0}])
    results = label_grounded_assertions(joined)
    schema = next(r for r in results if "label schema" in r.name)
    assert not schema.passed


def test_label_grounded_honeypot_in_top_100_fails():
    joined = _joined_frame(
        [{"bucket": "honeypot_drop", "true_label": "honeypot", "current_rank": 50.0}]
    )
    results = label_grounded_assertions(joined)
    hp = next(r for r in results if "label=honeypot" in r.name)
    assert not hp.passed


def test_valid_labels_contain_expected_taxonomy():
    assert {"fit", "near_fit", "not_fit", "honeypot", "stuffer"} == set(VALID_LABELS)


def test_run_report_writes_markdown_and_returns_pass(tmp_path: Path):
    seed = pd.DataFrame(
        [
            {
                "candidate_id": "A",
                "bucket": "honeypot_drop",
                "predicate_evidence": "",
                "current_tier": "E",
                "current_ceiling": "none",
                "current_rank": "",
                "expected_band": "dropped",
                "true_label": "",
                "notes": "",
            }
        ]
    )
    seed_path = tmp_path / "seed.csv"
    seed.to_csv(seed_path, index=False)
    sub = pd.DataFrame({"candidate_id": ["Z"], "rank": [1], "score": [0.9], "reasoning": ["x"]})
    sub_path = tmp_path / "sub.csv"
    sub.to_csv(sub_path, index=False)
    out = tmp_path / "report.md"
    overall, results = run_report(seed_path, sub_path, out)
    assert overall is True
    assert out.exists()
    body = out.read_text()
    assert "# Holdout report" in body
    assert "## Bucket summary" in body
    assert all(isinstance(r, AssertionResult) for r in results)


def test_bucket_and_label_summary_tables_handle_empty(tmp_path: Path):
    joined = _joined_frame([{"bucket": "plain_language", "current_rank": np.nan}])
    bs = _bucket_summary_table(joined)
    assert "plain_language" in bs
    ls = _label_summary_table(joined)
    assert "No `true_label`" in ls
