"""Tests for src/embeddings.py — candidate_doc + hash_model_dir.

Encoder integration tests require the vendored BGE-small model at
``artifacts/model/``; they're guarded with ``pytest.skip`` if missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.embeddings import (
    DOC_CHAR_CAP,
    EMBEDDING_DIM,
    candidate_doc,
    candidate_docs,
    encode_candidates,
    encode_jd_intents,
    hash_model_dir,
    load_model,
    resolve_device,
)

MODEL_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "model"


def _model_available() -> bool:
    return MODEL_DIR.exists() and any(MODEL_DIR.glob("*.json"))


def test_candidate_doc_includes_headline_summary_descriptions_skills(by_id: dict) -> None:
    doc = candidate_doc(by_id["CAND_0000001"])
    assert "Senior ML Engineer" in doc  # headline
    assert "BGE" in doc  # role description
    assert "Pinecone" in doc  # role description
    assert "skills:" in doc
    assert "Python" in doc


def test_candidate_doc_caps_length() -> None:
    record = {
        "profile": {"headline": "a" * 5000, "summary": "b" * 5000},
        "career_history": [{"description": "c" * 5000}],
        "skills": [{"name": "Python"}],
    }
    doc = candidate_doc(record)
    assert len(doc) <= DOC_CHAR_CAP


def test_candidate_doc_handles_missing_fields() -> None:
    doc = candidate_doc({})
    assert doc == ""


def test_candidate_doc_skips_empty_descriptions() -> None:
    record = {
        "profile": {"headline": "h", "summary": "s"},
        "career_history": [
            {"description": ""},
            {"description": None},
            {"description": "real description"},
        ],
        "skills": [{"name": "Python"}, {"name": ""}],
    }
    doc = candidate_doc(record)
    assert "real description" in doc
    assert "skills: Python" in doc


def test_candidate_docs_returns_list_of_strings(synthetic_50: list[dict]) -> None:
    docs = candidate_docs(synthetic_50)
    assert len(docs) == 50
    assert all(isinstance(d, str) for d in docs)


def test_hash_model_dir_deterministic(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("hello")
    (tmp_path / "b.bin").write_bytes(b"x" * 1024)
    h1 = hash_model_dir(tmp_path)
    h2 = hash_model_dir(tmp_path)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_hash_model_dir_changes_on_content_change(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("hello")
    h1 = hash_model_dir(tmp_path)
    (tmp_path / "a.json").write_text("hello world")
    h2 = hash_model_dir(tmp_path)
    assert h1 != h2


def test_hash_model_dir_changes_on_new_file(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("hello")
    h1 = hash_model_dir(tmp_path)
    (tmp_path / "b.json").write_text("world")
    h2 = hash_model_dir(tmp_path)
    assert h1 != h2


def test_load_model_raises_on_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "absent")


def test_resolve_device_passthrough() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("mps") == "mps"
    assert resolve_device("cuda") == "cuda"


def test_resolve_device_auto_returns_supported_backend() -> None:
    import torch

    expected = (
        "mps" if (torch.backends.mps.is_available() and torch.backends.mps.is_built()) else "cpu"
    )
    assert resolve_device("auto") == expected


@pytest.mark.skipif(not _model_available(), reason="BGE model not vendored")
def test_real_encoding_smoke(synthetic_50: list[dict]) -> None:
    """Integration: encode 5 candidates with the real BGE model."""
    model = load_model(MODEL_DIR)
    subset = synthetic_50[:5]
    emb = encode_candidates(subset, model, batch_size=8, show_progress=False)
    assert emb.shape == (5, EMBEDDING_DIM)
    assert emb.dtype == np.float16
    # Unit-normed: row norms close to 1 (within float16 tolerance).
    norms = np.linalg.norm(emb.astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2)


@pytest.mark.skipif(not _model_available(), reason="BGE model not vendored")
def test_real_jd_intent_encoding_smoke() -> None:
    from src.config_loader import load_config

    model = load_model(MODEL_DIR)
    intents = list(load_config("jd_intents")["intents"])
    emb = encode_jd_intents(intents, model)
    assert emb.shape == (len(intents), EMBEDDING_DIM)
    assert emb.dtype == np.float16
