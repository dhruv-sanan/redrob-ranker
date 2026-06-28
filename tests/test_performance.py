"""Performance + determinism gates for ``rank.py``.

Runs against the real 100K artifacts in a subprocess (same rationale as
``test_end_to_end.py``: the forbidden-module guard inside ``rank.py`` needs
a clean ``sys.modules`` — pytest's shared interpreter otherwise carries
over ``torch`` / ``transformers`` from the embeddings tests). Skipped when
artifacts are absent (cold clone before ``build_features.py``).
"""

from __future__ import annotations

import resource
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"

# Current rank.py wall on M4 CPU is ~2 s. 10 s is a 5x safety margin —
# catches drift early without flaking on slower CI hardware. Hard cap from
# hld.md §3.5 check #2 is 300 s.
WALL_BUDGET_S = 10.0

# Peak RSS on the V8 floor is ~1.8 GB (CP-5b runtime_report). 4 GB cap is a
# 2x margin — catches a future regression that allocates an unintended
# 100K × 384-d float32 view (~150 MB) plus other slack without flaking on
# CI containers. Allocation budget is well below the Stage-3 5 GB cap.
RSS_BUDGET_GB = 4.0


def _maxrss_bytes() -> int:
    """Return current ru_maxrss for child processes in bytes.

    macOS reports ru_maxrss in bytes; Linux reports kibibytes
    (per getrusage(2)). Normalize so the test asserts in real GB.
    """
    raw = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def _artifacts_present() -> bool:
    needed = [
        ARTIFACTS / "manifest.json",
        ARTIFACTS / "features.parquet",
        ARTIFACTS / "candidate_emb.npy",
        ARTIFACTS / "jd_intent_vecs.npy",
        ARTIFACTS / "candidates.parquet",
    ]
    return all(p.exists() for p in needed)


def _run_rank(out_dir: Path, suffix: str = "") -> tuple[float, Path, Path, Path]:
    sub_csv = out_dir / f"submission{suffix}.csv"
    audit_csv = out_dir / f"audit{suffix}.csv"
    debug_csv = out_dir / f"debug{suffix}.csv"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "rank.py"),
            "--artifacts",
            str(ARTIFACTS),
            "--out",
            str(sub_csv),
            "--audit",
            str(audit_csv),
            "--debug",
            str(debug_csv),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = time.perf_counter() - t0
    assert (
        proc.returncode == 0
    ), f"rank.py exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return elapsed, sub_csv, audit_csv, debug_csv


@pytest.mark.skipif(not _artifacts_present(), reason="100K artifacts not present")
def test_rank_py_wall_under_10s(tmp_path: Path) -> None:
    elapsed, *_ = _run_rank(tmp_path)
    assert elapsed < WALL_BUDGET_S, (
        f"rank.py wall {elapsed:.2f}s exceeds {WALL_BUDGET_S:.0f}s budget " f"(hard cap 300s)"
    )


@pytest.mark.skipif(not _artifacts_present(), reason="100K artifacts not present")
def test_rank_py_deterministic(tmp_path: Path) -> None:
    _, a_sub, a_audit, a_debug = _run_rank(tmp_path, "_a")
    _, b_sub, b_audit, b_debug = _run_rank(tmp_path, "_b")
    assert (
        a_sub.read_bytes() == b_sub.read_bytes()
    ), "rank.py submission CSV not deterministic across runs"
    assert (
        a_audit.read_bytes() == b_audit.read_bytes()
    ), "rank.py audit CSV not deterministic across runs"
    assert (
        a_debug.read_bytes() == b_debug.read_bytes()
    ), "rank.py debug CSV not deterministic across runs"


@pytest.mark.skipif(not _artifacts_present(), reason="100K artifacts not present")
def test_rank_py_rss_under_4gb(tmp_path: Path) -> None:
    """Cap peak resident memory of rank.py subprocess at 4 GB.

    Reads ``ru_maxrss`` for ``RUSAGE_CHILDREN`` after the subprocess has
    been reaped. ``ru_maxrss`` is a high-water mark across all reaped
    children in this pytest process — earlier rank.py subprocesses in the
    same session inflate it to the same peak, so the assertion remains
    correct under any test order.
    """
    _run_rank(tmp_path, "_rss")
    rss_bytes = _maxrss_bytes()
    rss_gb = rss_bytes / (1024**3)
    assert (
        rss_gb < RSS_BUDGET_GB
    ), f"rank.py peak RSS {rss_gb:.2f} GB exceeds {RSS_BUDGET_GB:.1f} GB budget"
