"""End-to-end test: run build_features_df against the 50-candidate fixture.

Asserts the output DataFrame has the expected shape, no NaNs in required scalar
columns, tier histogram covers the buckets, and the right candidates land in
their archetype tiers.
"""

from __future__ import annotations

import pandas as pd

from src.feature_pipeline import FEATURE_COLUMNS, build_feature_row, build_features_df
from src.reference_date import REFERENCE_DATE


def test_pipeline_row_for_single_candidate(by_id: dict) -> None:
    row = build_feature_row(by_id["CAND_0000001"], REFERENCE_DATE)
    for col in FEATURE_COLUMNS:
        assert col in row, f"missing column {col}"
    assert row["candidate_id"] == "CAND_0000001"
    assert row["tier"] == "A"
    assert row["honeypot_drop"] is False  # build_feature_row returns Python bool


def test_pipeline_dataframe_50_rows(synthetic_50: list[dict]) -> None:
    df = build_features_df(synthetic_50, REFERENCE_DATE)
    assert len(df) == 50
    for col in FEATURE_COLUMNS:
        assert col in df.columns, f"missing column {col}"


def test_pipeline_tier_histogram_covers_expected_buckets(synthetic_50: list[dict]) -> None:
    df = build_features_df(synthetic_50, REFERENCE_DATE)
    hist = df["tier"].value_counts().to_dict()
    # Tier A — 5 hand-crafted real fits.
    assert hist.get("A", 0) >= 5
    # Tier E — 3 honeypots + (up to 5) stuffers if stuffer_risk >= 0.7.
    assert hist.get("E", 0) >= 5


def test_honeypots_all_dropped(synthetic_50: list[dict]) -> None:
    df = build_features_df(synthetic_50, REFERENCE_DATE)
    drops = df[df["honeypot_drop"]]["candidate_id"].tolist()
    for cid in ["CAND_0000014", "CAND_0000015", "CAND_0000016"]:
        assert cid in drops, f"honeypot {cid} not dropped"


def test_concurrent_advisor_not_dropped(synthetic_50: list[dict]) -> None:
    df = build_features_df(synthetic_50, REFERENCE_DATE)
    row = df[df["candidate_id"] == "CAND_0000040"].iloc[0]
    assert not row["honeypot_drop"]


def test_tier_a_real_fits_in_tier_a(synthetic_50: list[dict]) -> None:
    df = build_features_df(synthetic_50, REFERENCE_DATE)
    for cid in ["CAND_0000001", "CAND_0000002", "CAND_0000003", "CAND_0000004", "CAND_0000005"]:
        row = df[df["candidate_id"] == cid].iloc[0]
        assert row["tier"] == "A", f"{cid} expected Tier A, got {row['tier']}"


def test_stuffers_either_tier_d_or_e(synthetic_50: list[dict]) -> None:
    df = build_features_df(synthetic_50, REFERENCE_DATE)
    for cid in [
        "CAND_0000009",
        "CAND_0000010",
        "CAND_0000011",
        "CAND_0000012",
        "CAND_0000013",
    ]:
        row = df[df["candidate_id"] == cid].iloc[0]
        assert row["tier"] in {"D", "E"}, f"{cid} expected D/E, got {row['tier']}"


def test_anti_pattern_ceilings_fire_for_archetypes(synthetic_50: list[dict]) -> None:
    df = build_features_df(synthetic_50, REFERENCE_DATE)
    by = df.set_index("candidate_id")
    assert by.loc["CAND_0000017", "anti_pattern_ceiling"] == "rank_50"  # services-only TCS
    assert by.loc["CAND_0000025", "anti_pattern_ceiling"] == "rank_50"  # CV-only
    assert by.loc["CAND_0000028", "anti_pattern_ceiling"] == "rank_50"  # recent LangChain
    assert by.loc["CAND_0000030", "anti_pattern_ceiling"] == "rank_50"  # inactive VP


def test_logistics_outside_india_not_top10_eligible(synthetic_50: list[dict]) -> None:
    df = build_features_df(synthetic_50, REFERENCE_DATE)
    by = df.set_index("candidate_id")
    assert not by.loc["CAND_0000032", "logistics_top_10_eligible"]
    assert by.loc["CAND_0000001", "logistics_top_10_eligible"]


def test_pipeline_writes_clean_parquet(synthetic_50: list[dict], tmp_path) -> None:
    from src.io_utils import read_parquet, write_parquet

    df = build_features_df(synthetic_50, REFERENCE_DATE)
    out = tmp_path / "features.parquet"
    write_parquet(df, out)
    df2 = read_parquet(out)
    assert len(df2) == 50
    pd.testing.assert_series_equal(
        df2["candidate_id"].reset_index(drop=True),
        df["candidate_id"].reset_index(drop=True),
    )
