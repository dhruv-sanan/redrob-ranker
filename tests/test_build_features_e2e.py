"""End-to-end test for build_features.py.

Runs the full CLI pipeline on a 10-candidate JSONL using a fresh tmp `--out`
directory and the vendored BGE-small model. Asserts every artifact lands and
`verify_manifest` passes.

Skipped when ``artifacts/model/`` is absent — the vendor step is exercised by
`tools/vendor_model.py` and is not part of this fast-path E2E.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import build_features as bf
from src.manifest import MANIFEST_FILENAME, load_manifest, verify_manifest

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDORED_MODEL = REPO_ROOT / "artifacts" / "model"


def _model_available() -> bool:
    return VENDORED_MODEL.exists() and any(VENDORED_MODEL.glob("*.json"))


@pytest.mark.skipif(not _model_available(), reason="BGE model not vendored")
def test_build_features_end_to_end(synthetic_50: list[dict], tmp_path: Path) -> None:
    subset = synthetic_50[:10]
    cands = tmp_path / "candidates.jsonl"
    cands.write_text(
        "".join(json.dumps(rec) + "\n" for rec in subset),
        encoding="utf-8",
    )
    out = tmp_path / "out"

    rc = bf.main(
        [
            "--candidates",
            str(cands),
            "--out",
            str(out),
            "--model-dir",
            str(VENDORED_MODEL),
            "--batch-size",
            "8",
            "--no-progress",
        ]
    )
    assert rc == 0

    for name in (
        "candidates.parquet",
        "features.parquet",
        "candidate_emb.npy",
        "jd_intent_vecs.npy",
        MANIFEST_FILENAME,
        "build_features_summary.json",
    ):
        assert (out / name).exists(), f"missing artifact: {name}"

    emb = np.load(out / "candidate_emb.npy")
    assert emb.shape == (10, 384)
    assert emb.dtype == np.float16
    norms = np.linalg.norm(emb.astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2)

    jd = np.load(out / "jd_intent_vecs.npy")
    assert jd.shape[1] == 384
    assert jd.dtype == np.float16

    manifest = load_manifest(out / MANIFEST_FILENAME)
    assert manifest.parquet_rows == 10
    assert manifest.embedding_shape == [10, 384]
    assert manifest.reference_date == "2026-06-01"
    verify_manifest(manifest, out, candidates_path=cands)
