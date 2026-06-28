#!/usr/bin/env python3
"""CP-5b runtime report — wall-clock, RAM, disk, and tier histogram.

Composes a single ``runtime_report.md`` from:

  * ``artifacts/build_features_summary.json`` — written by ``build_features.py``
    at offline-build time (tier histogram, honeypot counts, artifact bytes).
  * A fresh ``rank.py`` subprocess invocation — wall + peak RSS measured
    here (rank.py itself is the online step the reviewer reruns).
  * On-disk artifact stat — current bytes per file.

Build-time wall is NOT re-measured (the encoder dominates at ~50 min on M4
CPU); the historical value from ``progress.md`` / ``completed.md`` is the
source of truth for that figure.

Usage:
    python tools/runtime_report.py \\
        --artifacts ./artifacts/ \\
        --out runtime_report.md \\
        [--build-wall-seconds 3845]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ARTIFACT_FILES: tuple[str, ...] = (
    "manifest.json",
    "candidates.parquet",
    "features.parquet",
    "candidate_emb.npy",
    "jd_intent_vecs.npy",
    "build_features_summary.json",
)


def _bytes_to_mb(b: int | float) -> float:
    return float(b) / (1024.0 * 1024.0)


def measure_rank(rank_script: Path, artifacts_dir: Path, tmp_dir: Path) -> dict:
    """Run rank.py as a subprocess; return wall + peak RSS + telemetry."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / "top_100_submission.csv"
    audit = tmp_dir / "top_100_audit.csv"
    debug = tmp_dir / "top_300_debug.csv"
    cmd = [
        sys.executable,
        str(rank_script),
        "--artifacts",
        str(artifacts_dir),
        "--out",
        str(out),
        "--audit",
        str(audit),
        "--debug",
        str(debug),
    ]
    rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT), check=False)
    wall = time.perf_counter() - t0
    rss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if proc.returncode != 0:
        raise RuntimeError(f"rank.py failed: {proc.stderr}")
    rank_summary: dict = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                rank_summary = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
    return {
        "wall_seconds": wall,
        "peak_rss_bytes_delta": max(0, rss_after - rss_before),
        "peak_rss_bytes_total": rss_after,
        "telemetry": rank_summary,
    }


def measure_artifacts(artifacts_dir: Path) -> dict:
    sizes = {}
    total = 0
    for name in ARTIFACT_FILES:
        p = artifacts_dir / name
        if p.exists():
            n = p.stat().st_size
            sizes[name] = n
            total += n
    model_dir = artifacts_dir / "model"
    model_bytes = 0
    if model_dir.exists() and model_dir.is_dir():
        for p in model_dir.rglob("*"):
            if p.is_file():
                model_bytes += p.stat().st_size
    sizes["model/ (recursive)"] = model_bytes
    return {"per_file": sizes, "total_with_model": total + model_bytes, "total_no_model": total}


def load_build_summary(artifacts_dir: Path) -> dict:
    path = artifacts_dir / "build_features_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format_sizes_table(sizes: dict) -> str:
    rows = ["| artifact | size (MB) |", "|---|---:|"]
    for name, n in sizes["per_file"].items():
        rows.append(f"| {name} | {_bytes_to_mb(n):.2f} |")
    rows.append(f"| **total (no model)** | **{_bytes_to_mb(sizes['total_no_model']):.2f}** |")
    rows.append(f"| **total (with model)** | **{_bytes_to_mb(sizes['total_with_model']):.2f}** |")
    return "\n".join(rows)


def _format_tier_histogram(summary: dict) -> str:
    tiers = summary.get("tier_histogram", {})
    if not tiers:
        return "_no tier histogram in summary — re-run build_features.py_"
    rows = ["| tier | n |", "|---|---:|"]
    for tier in ("A", "B", "C", "D", "E"):
        if tier in tiers:
            rows.append(f"| {tier} | {tiers[tier]} |")
    return "\n".join(rows)


def _format_report(
    rank_info: dict,
    sizes: dict,
    build_summary: dict,
    build_wall_seconds: float | None,
) -> str:
    rank_wall = rank_info["wall_seconds"]
    rank_rss_mb = _bytes_to_mb(rank_info["peak_rss_bytes_total"])
    bw_str = (
        "—"
        if build_wall_seconds is None
        else f"{build_wall_seconds:.0f} s ({build_wall_seconds / 60:.1f} min)"
    )

    blocks = [
        "# Runtime report — CP-5b",
        "",
        "## Wall-clock",
        "",
        "| stage | wall | notes |",
        "|---|---:|---|",
        f"| build_features.py (offline) | {bw_str} | encoder-dominated; CPU M4 |",
        f"| rank.py (online) | {rank_wall:.2f} s | this run; hard cap 300 s |",
        "",
        "## Peak memory",
        "",
        f"- rank.py subprocess peak RSS: **{rank_rss_mb:.1f} MB**",
        "- (build_features.py peak ≈ 2 GB, dominated by BGE model in RAM + batched encoding)",
        "",
        "## Artifact disk",
        "",
        _format_sizes_table(sizes),
        "",
        "## Tier histogram (100K candidates)",
        "",
        _format_tier_histogram(build_summary),
        "",
        "## Honeypot counts",
        "",
    ]
    blocks.append(f"- honeypot_drop: **{build_summary.get('honeypot_drop_count', '—')}**")
    blocks.append(f"- honeypot_audit: **{build_summary.get('honeypot_audit_count', '—')}**")
    blocks += ["", "## rank.py telemetry (current run)", "", "```json"]
    blocks.append(json.dumps(rank_info.get("telemetry", {}), indent=2))
    blocks.append("```")
    blocks.append("")
    return "\n".join(blocks)


def run_runtime_report(
    artifacts_dir: Path,
    out_path: Path,
    build_wall_seconds: float | None,
    rank_script: Path,
) -> dict:
    sizes = measure_artifacts(artifacts_dir)
    summary = load_build_summary(artifacts_dir)
    tmp_dir = artifacts_dir / "_runtime_tmp"
    rank_info = measure_rank(rank_script, artifacts_dir, tmp_dir)
    for p in tmp_dir.glob("*.csv"):
        p.unlink()
    if tmp_dir.exists():
        with contextlib.suppress(OSError):
            tmp_dir.rmdir()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_format_report(rank_info, sizes, summary, build_wall_seconds))
    return {"rank": rank_info, "sizes": sizes, "build_summary": summary}


def _parse_argv(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CP-5b runtime report.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--out", type=Path, default=Path("runtime_report.md"))
    parser.add_argument(
        "--build-wall-seconds",
        type=float,
        default=3050.0,
        help="historical build_features.py wall (default ~50 min M4 CPU)",
    )
    parser.add_argument(
        "--rank-script",
        type=Path,
        default=Path("rank.py"),
        help="path to rank.py entry point",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_argv(argv or sys.argv[1:])
    result = run_runtime_report(args.artifacts, args.out, args.build_wall_seconds, args.rank_script)
    rank_wall = result["rank"]["wall_seconds"]
    print(f"[runtime_report] wrote {args.out}")
    print(f"  rank.py wall: {rank_wall:.2f} s")
    print(f"  artifacts total (no model): {_bytes_to_mb(result['sizes']['total_no_model']):.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
