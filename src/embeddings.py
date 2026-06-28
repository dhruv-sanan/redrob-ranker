"""BGE-small-en-v1.5 encoding. Offline-only — imported by `build_features.py`,
NEVER by `rank.py`.

Pipeline:
  candidate_doc(record)       — assemble headline + summary + descriptions + skills
  load_model(model_dir)       — load a vendored SentenceTransformer (no network)
  encode_candidates(...)      — batch encode docs → (N, 384) float16, unit-normed
  encode_jd_intents(...)      — encode JD intent strings → (K, 384) float16
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

DOC_CHAR_CAP = 4096
DEFAULT_BATCH_SIZE = 512
EMBEDDING_DIM = 384


def candidate_doc(record: dict[str, Any]) -> str:
    """Assemble the embedding input document for one candidate.

    Format: ``headline . summary . role1_desc . role2_desc ... . skills: a, b, c``
    Capped at ``DOC_CHAR_CAP`` chars to bound encoder cost.
    """
    profile = record.get("profile", {}) or {}
    parts: list[str] = []
    headline = (profile.get("headline") or "").strip()
    summary = (profile.get("summary") or "").strip()
    if headline:
        parts.append(headline)
    if summary:
        parts.append(summary)
    for role in record.get("career_history", []) or []:
        desc = (role.get("description") or "").strip()
        if desc:
            parts.append(desc)
    skill_names = [
        (s.get("name") or "").strip()
        for s in (record.get("skills", []) or [])
        if (s.get("name") or "").strip()
    ]
    if skill_names:
        parts.append("skills: " + ", ".join(skill_names))
    doc = " . ".join(parts)
    return doc[:DOC_CHAR_CAP]


def candidate_docs(records: Iterable[dict[str, Any]]) -> list[str]:
    return [candidate_doc(r) for r in records]


def load_model(model_dir: Path) -> Any:
    """Load a vendored SentenceTransformer from disk. Force CPU; no network."""
    from sentence_transformers import SentenceTransformer

    if not model_dir.exists():
        raise FileNotFoundError(
            f"model dir not found: {model_dir}. Run `python tools/vendor_model.py` first."
        )
    return SentenceTransformer(str(model_dir), device="cpu")


def encode_strings(
    strings: list[str],
    model: Any,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode strings to a unit-normed float16 ndarray of shape (len(strings), 384)."""
    arr = model.encode(
        strings,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return arr.astype(np.float16)


def encode_candidates(
    records: list[dict[str, Any]],
    model: Any,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    return encode_strings(
        candidate_docs(records), model, batch_size=batch_size, show_progress=show_progress
    )


def encode_jd_intents(
    intents: list[str],
    model: Any,
    *,
    show_progress: bool = False,
) -> np.ndarray:
    return encode_strings(intents, model, batch_size=len(intents) or 1, show_progress=show_progress)


def hash_model_dir(model_dir: Path) -> str:
    """Stable hash over the vendored model directory. Order-deterministic."""
    import hashlib

    h = hashlib.sha256()
    for path in sorted(model_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(model_dir).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            with path.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
    return h.hexdigest()
