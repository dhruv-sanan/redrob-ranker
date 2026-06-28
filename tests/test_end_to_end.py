"""End-to-end: run ``rank.py`` against the real 100K artifacts and verify
the submission CSV passes ``validate_submission.py``.

``rank.py`` is invoked in a subprocess so its forbidden-module guard runs
against a clean ``sys.modules``. Pytest's shared interpreter would
otherwise carry over ``torch`` / ``transformers`` imports from the
embeddings tests and the guard would (correctly) abort.

Skipped unless the full 100K artifacts (model + features + embedding) are
present on disk; that's the case after ``build_features.py`` finishes a real
run, but not on a cold clone before CP-3 artifacts exist.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
VALIDATE_PY = (
    REPO_ROOT.parent
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "validate_submission.py"
)


def _artifacts_present() -> bool:
    needed = [
        ARTIFACTS / "manifest.json",
        ARTIFACTS / "features.parquet",
        ARTIFACTS / "candidate_emb.npy",
        ARTIFACTS / "jd_intent_vecs.npy",
        ARTIFACTS / "candidates.parquet",
    ]
    return all(p.exists() for p in needed)


def _load_validate_module():
    spec = importlib.util.spec_from_file_location("validate_submission", VALIDATE_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load validate_submission at {VALIDATE_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _run_rank_subprocess(out_dir: Path) -> tuple[float, Path, Path, Path]:
    """Run ``rank.py`` in a clean subprocess; return (elapsed_s, paths...)."""
    sub_csv = out_dir / "submission.csv"
    audit_csv = out_dir / "audit.csv"
    debug_csv = out_dir / "debug.csv"
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
        timeout=600,
    )
    dt = time.perf_counter() - t0
    assert (
        proc.returncode == 0
    ), f"rank.py exited {proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return dt, sub_csv, audit_csv, debug_csv


@pytest.mark.skipif(not _artifacts_present(), reason="100K artifacts not present")
@pytest.mark.skipif(not VALIDATE_PY.exists(), reason="validate_submission.py not bundled")
def test_rank_end_to_end_on_100k(tmp_path: Path) -> None:
    dt, sub_csv, audit_csv, debug_csv = _run_rank_subprocess(tmp_path)
    assert dt < 300.0, f"rank.py took {dt:.1f}s (hard cap 300s)"
    assert sub_csv.exists()
    assert audit_csv.exists()
    assert debug_csv.exists()

    vm = _load_validate_module()
    errors = vm.validate_submission(str(sub_csv))
    assert errors == [], errors

    with open(sub_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 100
    assert all(r["reasoning"].strip() for r in rows)


@pytest.mark.skipif(not _artifacts_present(), reason="100K artifacts not present")
def test_reasoning_audit_clean(tmp_path: Path) -> None:
    _, _, audit_csv, _ = _run_rank_subprocess(tmp_path)
    import reasoning_audit

    audit_out = tmp_path / "audit_out.csv"
    rc = reasoning_audit.run_audit(audit_csv, ARTIFACTS / "candidates.parquet", audit_out)
    assert rc == 0, f"reasoning_audit failed — see {audit_out}"
