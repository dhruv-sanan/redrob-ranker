#!/usr/bin/env python3
"""HuggingFace Space entry — accepts a small candidate sample and returns
a ranked CSV using the same scoring + ranking pipeline as ``rank.py``.

Constraints (per ``hld.md`` Sandbox checklist):
  * Accepts ≤100-candidate JSONL upload.
  * Completes < 5 min on HF Space free CPU tier.
  * Reuses ``src.scoring`` + ``src.ranking`` + ``src.reasoning`` —
    no parallel implementation.

Unlike ``rank.py``, this file is NOT under the restricted-imports allow-list:
it loads ``sentence-transformers`` to encode the sample on the fly (the
sample is fresh data, so the precomputed ``candidate_emb.npy`` does not
apply). The vendored BGE-small model is loaded once at startup.

Run locally:
    python app.py
Then visit http://127.0.0.1:7860/.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import starlette.templating as _st
import yaml

from src.embeddings import candidate_docs, encode_strings, load_model
from src.feature_pipeline import build_features_df
from src.ranking import (
    drop_honeypots,
    enforce_monotonic_scores,
    tier_sort,
)
from src.reasoning import load_skeletons_from_yaml, render_top_100_reasoning
from src.reference_date import REFERENCE_DATE
from src.scoring import compute_scores

# Starlette 1.x removed TemplateResponse(name, context) backwards-compat; gradio 4.x
# calls that form internally. Restore the shim here before any HTTP requests are served.
_orig_tr = _st.Jinja2Templates.TemplateResponse


def _compat_tr(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN201
    if args and isinstance(args[0], str):
        name, context = args[0], (args[1] if len(args) > 1 else {})
        return _orig_tr(self, context.get("request"), name, context)
    return _orig_tr(self, *args, **kwargs)


_st.Jinja2Templates.TemplateResponse = _compat_tr  # type: ignore[method-assign]

REPO_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "model"
JD_VECS_PATH = ARTIFACTS_DIR / "jd_intent_vecs.npy"
WEIGHTS_PATH = REPO_ROOT / "config" / "weights.yaml"
SKELETONS_PATH = REPO_ROOT / "config" / "skeletons.yaml"
THRESHOLDS_PATH = REPO_ROOT / "config" / "thresholds.yaml"

MAX_SAMPLE_SIZE = 100


class RankerState:
    """Lazily-initialized singleton holding the model + cached vectors."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.jd_intent_vecs: np.ndarray | None = None
        self.weights: dict[str, Any] | None = None
        self.skeletons: Any | None = None
        self.top_10_cfg: dict[str, Any] | None = None

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        if not MODEL_DIR.exists():
            self._vendor_model()
        self.model = load_model(MODEL_DIR, device="cpu")
        if JD_VECS_PATH.exists():
            self.jd_intent_vecs = np.load(JD_VECS_PATH)
        else:
            self.jd_intent_vecs = self._encode_jd_intents()
        self.weights = yaml.safe_load(WEIGHTS_PATH.read_text(encoding="utf-8"))
        self.skeletons = load_skeletons_from_yaml(SKELETONS_PATH.read_text(encoding="utf-8"))
        thresholds = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8")) or {}
        self.top_10_cfg = thresholds.get("top_10_promotion", None)

    def _vendor_model(self) -> None:
        from tools.vendor_model import DEFAULT_MODEL, vendor

        vendor(DEFAULT_MODEL, MODEL_DIR)

    def _encode_jd_intents(self) -> np.ndarray:
        cfg = yaml.safe_load((REPO_ROOT / "config" / "jd_intents.yaml").read_text(encoding="utf-8"))
        prompts = list(cfg.get("intents", []))
        return encode_strings(prompts, self.model, show_progress=False)


STATE = RankerState()


_LIST_FIELDS = ("career_history", "skills", "education")


def _coerce_record(record: dict) -> dict:
    """Coerce string-serialized list fields produced by the old pandas+json.dumps path.

    pandas leaves nested list<struct> columns as numpy arrays; json.dumps(default=str)
    then stringifies them. ast.literal_eval recovers the original Python structure.
    """
    for key in _LIST_FIELDS:
        val = record.get(key)
        if isinstance(val, str):
            with contextlib.suppress(ValueError, SyntaxError):
                record[key] = ast.literal_eval(val)
    return record


def parse_jsonl_text(text: str) -> list[dict]:
    """Parse a JSONL string into a list of candidate dicts."""
    records: list[dict] = []
    for i, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(_coerce_record(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {i}: invalid JSON ({exc.msg})") from exc
    return records


def rank_sample(records: list[dict], *, target_size: int | None = None) -> pd.DataFrame:
    """Rank a fresh sample end-to-end; return ranked DataFrame.

    Output columns: candidate_id, rank, score, reasoning, tier, final_score.
    """
    STATE.ensure_loaded()
    assert STATE.model is not None
    assert STATE.weights is not None
    assert STATE.jd_intent_vecs is not None
    assert STATE.skeletons is not None

    if not records:
        raise ValueError("no candidates in sample")
    if len(records) > MAX_SAMPLE_SIZE:
        raise ValueError(f"sample size {len(records)} exceeds MAX_SAMPLE_SIZE={MAX_SAMPLE_SIZE}")

    features = build_features_df(records, REFERENCE_DATE)
    docs = candidate_docs(records)
    candidate_emb = encode_strings(docs, STATE.model, show_progress=False)

    scored, _ = compute_scores(features, candidate_emb, STATE.jd_intent_vecs, STATE.weights)
    survivors = drop_honeypots(scored)
    sorted_df = tier_sort(survivors)

    n_keep = min(target_size or len(sorted_df), len(sorted_df))
    top = sorted_df.head(n_keep).copy().reset_index(drop=True)
    if top.empty:
        return pd.DataFrame(
            columns=["candidate_id", "rank", "score", "reasoning", "tier", "final_score"]
        )

    top["rank"] = np.arange(1, len(top) + 1, dtype=np.int64)
    top["score"] = enforce_monotonic_scores(top["final_score"].to_numpy())

    raw_lookup = {str(r.get("candidate_id")): r for r in records}
    feature_dicts = top.to_dict(orient="records")
    rendered = render_top_100_reasoning(feature_dicts, raw_lookup, STATE.skeletons)
    top["reasoning"] = [r["reasoning"] for r in rendered]

    return top[["candidate_id", "rank", "score", "reasoning", "tier", "final_score"]]


def _render_csv(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df[["candidate_id", "rank", "score", "reasoning"]].to_csv(buf, index=False)
    return buf.getvalue()


def rank_from_inputs(uploaded_file, pasted_text, target_size):  # noqa: ANN001, ANN201
    """Gradio adapter — read upload or paste, return (preview_df, csv_path, status).

    Type annotations intentionally omitted: Gradio 4.x introspects the
    function signature to generate API schema, and ``typing.Any`` /
    ``pd.DataFrame`` produce a schema with ``additionalProperties: True``
    (boolean) that crashes ``gradio_client.utils.get_type`` until
    gradio 5.0. See requirements-app.txt pin commentary.
    """
    t0 = time.perf_counter()
    try:
        if uploaded_file is not None:
            text = Path(uploaded_file.name).read_text(encoding="utf-8")
            source = f"upload `{Path(uploaded_file.name).name}`"
        elif pasted_text and pasted_text.strip():
            text = pasted_text
            source = "pasted text"
        else:
            return (
                pd.DataFrame(),
                "",
                "Provide a JSONL upload OR paste JSONL into the text area.",
            )
        records = parse_jsonl_text(text)
        if len(records) > MAX_SAMPLE_SIZE:
            return (
                pd.DataFrame(),
                "",
                f"Sample of {len(records)} exceeds free-tier cap ({MAX_SAMPLE_SIZE}). Trim and retry.",
            )
        ranked = rank_sample(records, target_size=int(target_size))
        csv_path = Path(tempfile.gettempdir()) / "ranked_sample.csv"
        csv_path.write_text(_render_csv(ranked), encoding="utf-8")
        wall = time.perf_counter() - t0
        status = (
            f"Ranked {len(ranked)} of {len(records)} from {source} in {wall:.2f} s "
            f"(survivors after honeypot drop)."
        )
        return ranked, str(csv_path), status
    except (ValueError, json.JSONDecodeError) as exc:
        return pd.DataFrame(), "", f"Input error: {exc}"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Redrob Ranker — sandbox") as demo:
        gr.Markdown(
            "# Redrob Ranker — candidate sample sandbox\n"
            "Upload a JSONL of up to 100 candidates (same schema as `candidates.jsonl`). "
            "The app encodes them with vendored BGE-small, runs the production scoring + "
            "ranking pipeline, and returns the ranked CSV."
        )
        with gr.Row():
            with gr.Column(scale=1):
                upload = gr.File(label="candidates.jsonl (≤100 rows)", file_types=[".jsonl"])
                paste = gr.Textbox(
                    label="…or paste JSONL here",
                    lines=12,
                    placeholder='{"candidate_id": "CAND_0000001", ...}',
                )
                target = gr.Slider(1, MAX_SAMPLE_SIZE, value=10, step=1, label="Target ranked rows")
                go = gr.Button("Rank sample", variant="primary")
            with gr.Column(scale=2):
                table = gr.Dataframe(
                    label="Ranked sample",
                    headers=["candidate_id", "rank", "score", "reasoning", "tier", "final_score"],
                    wrap=True,
                )
                download = gr.File(label="Download ranked CSV")
                status = gr.Markdown()
        go.click(
            fn=rank_from_inputs,
            inputs=[upload, paste, target],
            outputs=[table, download, status],
            api_name=False,  # skip API schema introspection — bypasses gradio_client.utils 4.x crash
        )
        gr.Markdown(
            "_Architecture: scoring + ranking match `rank.py` on the full 100K. "
            "The promotion-gate path is bypassed for samples < 10 — small pools fall "
            "through to a plain tier-sorted head._"
        )
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_api=False)


if __name__ == "__main__":
    main()
