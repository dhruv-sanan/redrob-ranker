#!/usr/bin/env python3
"""Run CP-5a holdout assertions against the current rank.py output.

Joins ``artifacts/holdout_seed.csv`` (with optional human ``true_label``
fills) against ``top_100_submission.csv`` and reports bucket-level
invariants per blocking check #10 in ``holdout_report.md``.

Assertion classes:

  Predicate-bucket  — assertions on the 9 strata produced by
                      ``tools.build_holdout``. These run even with
                      blank ``true_label`` columns and constitute the
                      machine-checkable holdout pass/fail.
  Label-grounded    — assertions tied to user-supplied ``true_label``
                      values (``fit`` / ``near_fit`` / ``not_fit`` /
                      ``honeypot`` / ``stuffer``). Skipped if no labels
                      are filled.

Usage:
    python tools/holdout_report.py \\
        --seed holdout_labels.csv \\
        --submission top_100_submission.csv \\
        --out holdout_report.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from tools.build_holdout import BUCKET_ORDER  # noqa: E402

VALID_LABELS: frozenset[str] = frozenset({"fit", "near_fit", "not_fit", "honeypot", "stuffer"})


@dataclass(frozen=True)
class AssertionResult:
    name: str
    passed: bool
    observed: str
    threshold: str


def _join_with_ranks(seed: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    """Attach current_rank from submission CSV; non-ranked rows get rank=NaN."""
    rank_map = dict(zip(submission["candidate_id"], submission["rank"], strict=True))
    out = seed.copy()
    out["current_rank"] = out["candidate_id"].map(rank_map)
    return out


def _bucket_slice(joined: pd.DataFrame, bucket: str) -> pd.DataFrame:
    return joined.loc[joined["bucket"] == bucket]


def _frac_in_top_n(slice_df: pd.DataFrame, n: int) -> float:
    if len(slice_df) == 0:
        return 0.0
    in_top = slice_df["current_rank"].notna() & (slice_df["current_rank"] <= n)
    return float(in_top.sum()) / float(len(slice_df))


def _median_rank_in_top_100(slice_df: pd.DataFrame) -> float | None:
    ranked = slice_df.loc[slice_df["current_rank"].notna(), "current_rank"]
    if ranked.empty:
        return None
    return float(ranked.median())


def _vacuous_pass(name: str, threshold: str) -> AssertionResult:
    return AssertionResult(
        name=name, passed=True, observed="n=0 (bucket empty in seed)", threshold=threshold
    )


def predicate_assertions(joined: pd.DataFrame) -> list[AssertionResult]:
    results: list[AssertionResult] = []

    pl = _bucket_slice(joined, "plain_language")
    if pl.empty:
        results.append(
            _vacuous_pass(
                "plain_language: median rank ≤ 50 in top-100 AND ≥40% reach top-100",
                "median≤50 AND frac≥40%",
            )
        )
    else:
        med = _median_rank_in_top_100(pl)
        frac = _frac_in_top_n(pl, 100)
        pl_pass = (med is not None and med <= 50) and frac >= 0.40
        results.append(
            AssertionResult(
                name="plain_language: median rank ≤ 50 in top-100 AND ≥40% reach top-100",
                passed=pl_pass,
                observed=f"median={med}, frac_top_100={frac:.2%}",
                threshold="median≤50 AND frac≥40%",
            )
        )

    st = _bucket_slice(joined, "stuffer")
    frac = _frac_in_top_n(st, 100)
    results.append(
        AssertionResult(
            name="stuffer: ≤ 20% reach top-100",
            passed=frac <= 0.20,
            observed=f"frac_top_100={frac:.2%}",
            threshold="≤20%",
        )
    )

    hd = _bucket_slice(joined, "honeypot_drop")
    in100 = int((hd["current_rank"].notna() & (hd["current_rank"] <= 100)).sum())
    results.append(
        AssertionResult(
            name="honeypot_drop: 0 in top-100 (blocking check #5)",
            passed=in100 == 0,
            observed=f"in_top_100={in100}",
            threshold="==0",
        )
    )

    ha = _bucket_slice(joined, "honeypot_audit")
    frac = _frac_in_top_n(ha, 100)
    results.append(
        AssertionResult(
            name="honeypot_audit: ≤ 30% reach top-100",
            passed=frac <= 0.30,
            observed=f"frac_top_100={frac:.2%}",
            threshold="≤30%",
        )
    )

    it = _bucket_slice(joined, "irrelevant_tail")
    in100 = int((it["current_rank"].notna() & (it["current_rank"] <= 100)).sum())
    results.append(
        AssertionResult(
            name="irrelevant_tail: 0 in top-100",
            passed=in100 == 0,
            observed=f"in_top_100={in100}",
            threshold="==0",
        )
    )

    so = _bucket_slice(joined, "services_only")
    frac_top_50 = _frac_in_top_n(so, 50)
    results.append(
        AssertionResult(
            name="services_only: ≤ 50% reach top-50 (rank_50 ceiling pressure)",
            passed=frac_top_50 <= 0.50,
            observed=f"frac_top_50={frac_top_50:.2%}",
            threshold="≤50%",
        )
    )

    sp = _bucket_slice(joined, "services_to_product")
    sp_frac_100 = _frac_in_top_n(sp, 100)
    so_frac_100 = _frac_in_top_n(so, 100)
    results.append(
        AssertionResult(
            name="services_to_product: ≥ services_only top-100 fraction",
            passed=sp_frac_100 >= so_frac_100,
            observed=f"sp_top_100={sp_frac_100:.2%}, so_top_100={so_frac_100:.2%}",
            threshold="sp≥so",
        )
    )

    nt = _bucket_slice(joined, "non_tech_title")
    frac_top_50 = _frac_in_top_n(nt, 50)
    results.append(
        AssertionResult(
            name="non_tech_title: ≤ 50% reach top-50 (rank_50 ceiling + blocking check #6)",
            passed=frac_top_50 <= 0.50,
            observed=f"frac_top_50={frac_top_50:.2%}",
            threshold="≤50%",
        )
    )

    oi = _bucket_slice(joined, "outside_india")
    frac_100 = _frac_in_top_n(oi, 100)
    results.append(
        AssertionResult(
            name="outside_india: informational",
            passed=True,
            observed=f"frac_top_100={frac_100:.2%} (no auto-pass/fail)",
            threshold="informational",
        )
    )

    return results


def label_grounded_assertions(joined: pd.DataFrame) -> list[AssertionResult]:
    labels = joined["true_label"].fillna("").astype(str).str.strip().str.lower()
    if (labels == "").all():
        return []

    unknown = labels.loc[(labels != "") & (~labels.isin(VALID_LABELS))]
    results: list[AssertionResult] = []
    if not unknown.empty:
        results.append(
            AssertionResult(
                name=f"label schema: unknown label values {sorted(set(unknown))}",
                passed=False,
                observed=f"{len(unknown)} rows have non-canonical labels",
                threshold=f"labels in {sorted(VALID_LABELS)}",
            )
        )

    labeled = joined.assign(_label=labels)

    fit = labeled.loc[labeled["_label"] == "fit"]
    if not fit.empty:
        med = _median_rank_in_top_100(fit)
        results.append(
            AssertionResult(
                name="label=fit: median rank ≤ 25 in top-100",
                passed=med is not None and med <= 25,
                observed=f"median={med}, n={len(fit)}",
                threshold="median≤25",
            )
        )

    not_fit = labeled.loc[labeled["_label"] == "not_fit"]
    if not not_fit.empty:
        in_50 = int((not_fit["current_rank"].notna() & (not_fit["current_rank"] <= 50)).sum())
        results.append(
            AssertionResult(
                name="label=not_fit: 0 in top-50",
                passed=in_50 == 0,
                observed=f"in_top_50={in_50}, n={len(not_fit)}",
                threshold="==0",
            )
        )

    hp = labeled.loc[labeled["_label"] == "honeypot"]
    if not hp.empty:
        in_100 = int((hp["current_rank"].notna() & (hp["current_rank"] <= 100)).sum())
        results.append(
            AssertionResult(
                name="label=honeypot: 0 in top-100",
                passed=in_100 == 0,
                observed=f"in_top_100={in_100}, n={len(hp)}",
                threshold="==0",
            )
        )

    st = labeled.loc[labeled["_label"] == "stuffer"]
    if not st.empty:
        in_100 = int((st["current_rank"].notna() & (st["current_rank"] <= 100)).sum())
        results.append(
            AssertionResult(
                name="label=stuffer: 0 in top-100",
                passed=in_100 == 0,
                observed=f"in_top_100={in_100}, n={len(st)}",
                threshold="==0",
            )
        )

    return results


def _format_table(assertions: list[AssertionResult]) -> str:
    lines = ["| status | assertion | observed | threshold |", "|---|---|---|---|"]
    for r in assertions:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"| {status} | {r.name} | `{r.observed}` | `{r.threshold}` |")
    return "\n".join(lines)


def _bucket_summary_table(joined: pd.DataFrame) -> str:
    lines = [
        "| bucket | n_seed | in_top_100 | in_top_50 | median_rank_in_top_100 |",
        "|---|---:|---:|---:|---:|",
    ]
    for bucket in BUCKET_ORDER:
        sl = _bucket_slice(joined, bucket)
        n = len(sl)
        in_100 = int((sl["current_rank"].notna() & (sl["current_rank"] <= 100)).sum())
        in_50 = int((sl["current_rank"].notna() & (sl["current_rank"] <= 50)).sum())
        med = _median_rank_in_top_100(sl)
        med_str = f"{med:.1f}" if med is not None else "—"
        lines.append(f"| {bucket} | {n} | {in_100} | {in_50} | {med_str} |")
    return "\n".join(lines)


def _label_summary_table(joined: pd.DataFrame) -> str:
    labels = joined["true_label"].fillna("").astype(str).str.strip().str.lower()
    if (labels == "").all():
        return "_No `true_label` values present — run after hand-labeling fills the seed CSV._"
    lines = [
        "| label | n | in_top_100 | in_top_50 | median_rank_in_top_100 |",
        "|---|---:|---:|---:|---:|",
    ]
    grouped = joined.assign(_label=labels)
    for lbl, sub in grouped.groupby("_label", sort=True):
        if lbl == "":
            continue
        n = len(sub)
        in_100 = int((sub["current_rank"].notna() & (sub["current_rank"] <= 100)).sum())
        in_50 = int((sub["current_rank"].notna() & (sub["current_rank"] <= 50)).sum())
        med = _median_rank_in_top_100(sub)
        med_str = f"{med:.1f}" if med is not None else "—"
        lines.append(f"| {lbl} | {n} | {in_100} | {in_50} | {med_str} |")
    return "\n".join(lines)


def run_report(
    seed_csv: Path,
    submission_csv: Path,
    out_md: Path,
) -> tuple[bool, list[AssertionResult]]:
    seed = pd.read_csv(seed_csv)
    submission = pd.read_csv(submission_csv)
    joined = _join_with_ranks(seed, submission)

    pred = predicate_assertions(joined)
    label = label_grounded_assertions(joined)
    all_results = pred + label
    overall_pass = all(r.passed for r in all_results)

    blocks = [
        "# Holdout report — CP-5a",
        "",
        f"Source seed: `{seed_csv}` ({len(seed)} rows)",
        f"Submission: `{submission_csv}` ({len(submission)} rows)",
        f"Overall: **{'PASS' if overall_pass else 'FAIL'}**",
        "",
        "## Bucket summary",
        "",
        _bucket_summary_table(joined),
        "",
        "## Label summary",
        "",
        _label_summary_table(joined),
        "",
        "## Predicate-bucket assertions",
        "",
        _format_table(pred),
        "",
        "## Label-grounded assertions",
        "",
        _format_table(label) if label else "_No labels filled in seed CSV — skipped._",
        "",
    ]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(blocks))
    return overall_pass, all_results


def _parse_argv(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CP-5a holdout assertions.")
    parser.add_argument("--seed", type=Path, default=Path("holdout_labels.csv"))
    parser.add_argument("--submission", type=Path, default=Path("top_100_submission.csv"))
    parser.add_argument("--out", type=Path, default=Path("holdout_report.md"))
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero if any assertion fails (CI use)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_argv(argv or sys.argv[1:])
    overall_pass, results = run_report(args.seed, args.submission, args.out)
    fails = [r for r in results if not r.passed]
    status = "PASS" if overall_pass else "FAIL"
    print(f"[holdout_report] {status} — wrote {args.out} ({len(results)} assertions)")
    for r in fails:
        print(f"  FAIL: {r.name} — observed={r.observed} (threshold={r.threshold})")
    if args.strict and not overall_pass:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
