---
title: Redrob Ranker
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
python_version: "3.11"
pinned: false
license: apache-2.0
---

# redrob-ranker

[![ci](https://github.com/dhruv-sanan/redrob-ranker/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/dhruv-sanan/redrob-ranker/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-292%20passing-brightgreen)](https://github.com/dhruv-sanan/redrob-ranker/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/downloads/release/python-3110/)
[![rank.py](https://img.shields.io/badge/rank.py-%E2%89%A42s%20on%20100K-blue)](#commands)

Deterministic CPU-only candidate ranker for the Redrob "Intelligent Candidate Discovery & Ranking" challenge.

> Architecture: see `../problem.md §3 Solution A v2` (LOCKED).
> Build playbook: see `../lld.md §2` (5-phase plan, CP-1 → CP-S3).
> Submission gates: see `../hld.md`.

## 30-second demo

After the one-time offline build (Phase 3, ~60 min on M4 CPU) is cached,
the online ranking loop is **~2 s wall** on the full 100K. Four-command
cold-clone walkthrough:

```bash
git clone https://github.com/dhruv-sanan/redrob-ranker.git && cd redrob-ranker
python -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt
python tools/vendor_model.py                          # ~30 s — vendors BGE-small
python build_features.py --candidates <candidates.jsonl> --out ./artifacts/   # ~60 min cold
python rank.py --artifacts ./artifacts/ --out top_100_submission.csv          # ~2 s
python validate_submission.py top_100_submission.csv  # ships from challenge bundle
```

Repeat the last two lines after any weight / threshold tweak — Phase 4 is
the iteration loop. Full detail in the sections below.

## Environment

- Python 3.11.x (vendored via `uv` recommended)
- See `requirements.txt` for pinned deps. No GPU, no network at rank time.
- `REFERENCE_DATE = 2026-06-01` (frozen, single source: `src/reference_date.py`).
- `rank.py` import allow-list: `numpy, pandas, pyarrow, json, sys, pathlib, yaml`.
  `sentence-transformers` is offline-only and never imported by `rank.py`.

## Cold-clone setup

The BGE-small model and the 100K artifacts are **not committed** (see
`.gitignore`). Reproduce them like this:

```bash
# 1. Create venv (Python 3.11; uv recommended).
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements-dev.txt

# 2. Vendor BAAI/bge-small-en-v1.5 into artifacts/model/ (one network call, ~135 MB).
python tools/vendor_model.py

# 3. Build features + embeddings + manifest for the 100K candidates JSONL.
#    Wall-clock on Apple M4 CPU is ~60 min (the encoder is the dominant cost).
python build_features.py \
    --candidates "/path/to/candidates.jsonl" \
    --out ./artifacts/

# 4. Run the green gates.
ruff check src tests tools
ruff format --check src tests tools
pytest -q
```

After step 3 the `artifacts/` directory holds `manifest.json`,
`candidates.parquet`, `features.parquet`, `candidate_emb.npy`,
`jd_intent_vecs.npy`, and `build_features_summary.json`. `rank.py` (Phase 4)
consumes these and hash-verifies the manifest before doing anything else.

## Commands

```bash
# Phase 3 (offline, ~60 min on M4 CPU): parse + embed + features + manifest
python build_features.py --candidates /path/to/candidates.jsonl --out ./artifacts/

# Phase 4 (online, ≤5 min, restricted imports): rank top 100
python rank.py --artifacts ./artifacts/ --out ./top_100_submission.csv

# Post-rank: verify reasoning is grounded
python reasoning_audit.py --audit ./top_100_audit.csv --out ./reasoning_audit.csv

# Submission gate (ships from challenge bundle)
python validate_submission.py top_100_submission.csv
```

## Status

- **Phase 1 (CP-1):** ✅ Skeleton + data + manifest layer.
- **Phase 2 (CP-2):** ✅ 9 feature builders + 6 config YAMLs + pipeline.
- **Phase 3 (CP-3, CP-3.5):** ✅ BGE-small vendored + 100K embeddings + manifest; encoder perf patch.
- **Phase 4 (CP-4):** ✅ `rank.py` (~2 s wall on 100K), scoring, ranking, reasoning, audit. `validate_submission.py` clean.
- **Phase 5 (CP-5a..d):** ✅ Stratified holdout + ablations + HF sandbox + submission deck.
- **Submission (CP-S1..S3, CP-2.3..S2b):** ✅ V8 weights, 11/11 HLD blocking checks PASS, 13/13 holdout assertions PASS, 292 pytest pass, ruff clean, CI on GitHub Actions. See `final.md` for the punchlist and `bias_audit.md` for the top-100 fairness sweep.

See `progress.md` for the checkpoint log and `completed.md` for the
session-resume offload.
