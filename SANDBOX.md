# Redrob Ranker — sandbox deployment

The sandbox is a Gradio app (`app.py`) that accepts a ≤100-candidate JSONL
sample, encodes it with the vendored BGE-small model, runs the same
scoring + ranking pipeline used by `rank.py`, and returns the ranked CSV.

Stage-3 / Stage-4 reviewers can verify the live pipeline without
re-running the 50-minute offline build over the full 100K corpus.

## Local

```bash
.venv/bin/python -m pip install -r requirements.txt -r requirements-app.txt
.venv/bin/python tools/vendor_model.py     # if artifacts/model/ missing
.venv/bin/python app.py                    # → http://127.0.0.1:7860
```

The first request loads the model (~2 s on M4 CPU). Subsequent requests
encode + rank ~10–100 candidates in 1–5 s.

## HuggingFace Space (free CPU tier)

1. Create a new Space (Gradio SDK, free CPU). Choose Python 3.11.
2. Set the Space's `app_file` to `app.py`.
3. Push this repo to the Space's git remote. Do **not** push the
   `artifacts/` directory — the model is re-vendored at startup
   (`tools.vendor_model.vendor`) which downloads ~135 MB from HF Hub.
4. Add `requirements.txt` + `requirements-app.txt` to the Space root
   (HF auto-merges both).
5. First boot takes ~60 s (model download + load). Subsequent boots
   reuse the HF persistent disk and start in ~5 s.

The Space's `README.md` front-matter should be:

```yaml
---
title: Redrob Ranker
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
python_version: 3.11
---
```

## Docker fallback

If HF Spaces is unavailable (rate-limited, account flagged, etc.) the
included `Dockerfile` is the drop-in fallback. The model is vendored at
**build** time so cold-start latency stays low.

```bash
docker build -t redrob-ranker-sandbox .
docker run --rm -p 7860:7860 redrob-ranker-sandbox
```

Open http://localhost:7860/.

## Smoke-check

Paste 10 lines from `artifacts/candidates.parquet` (converted to JSONL)
into the upload panel. Expected: ranked CSV with `candidate_id, rank,
score, reasoning` in <5 s end-to-end on free CPU.

```bash
.venv/bin/python -c "import pandas as pd, json; \
  print('\\n'.join(json.dumps(r, default=str) for r in \
  pd.read_parquet('artifacts/candidates.parquet').head(10).to_dict('records')))" \
  > /tmp/sample10.jsonl
```

## What the sandbox does NOT do

- It does **not** rebuild `features.parquet` over the full 100K — the
  reviewer cold-clone path runs `build_features.py` separately
  (`~50 min` on M4 CPU, longer on x86).
- It does **not** load the precomputed `candidate_emb.npy` — the sample
  is fresh data, encoded on the fly with the vendored BGE model.
- It does **not** trip `rank.py`'s import allow-list. `app.py` is a
  separate entry point that may import `sentence-transformers`,
  `gradio`, etc. The restricted-imports guarantee applies to `rank.py`
  alone.
