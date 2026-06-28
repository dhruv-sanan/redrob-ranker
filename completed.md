# Redrob Ranker — Completed Work Log

> Append-only. Onboard a fresh session by reading this top-to-bottom; no need
> to re-read `problem.md` / `lld.md` / `hld.md` unless changing architecture.
> Last updated: 2026-06-28.

---

## Quick context (read first)

- **Goal:** win the Redrob hackathon (deadline **2026-07-02 23:59 IST**). Rank
  100,000 candidates for a Senior AI Engineer JD; produce a 100-row CSV.
- **Architecture LOCKED** in `../problem.md §3 Solution A v2`. No re-debate.
- **Build playbook LOCKED** in `../lld.md`. 5 phases → CP-1 … CP-5d → 3
  submissions (CP-S1, CP-S2, CP-S3).
- **Repo:** `/Users/dhruvsanan/Desktop/India_runs/redrob-ranker/`. `main`
  branch. Local-only; no remote configured.
- **REFERENCE_DATE = 2026-06-01.** Only `src/reference_date.py` constructs it.
  All other files anchor to it. Stage-3 Docker reproduction is date-stable
  regardless of when the reviewer runs the sandbox.
- **`rank.py` import allow-list** (enforced via static check in CP-4):
  `{numpy, pandas, pyarrow, json, sys, pathlib, yaml}`. Anything else = bug.
  `sentence-transformers` is offline-only and never imported by `rank.py`.

---

## CP-1 — Skeleton + data + manifest layer ✅

**Commit:** `13542e4` (2026-06-28).
**Files:** 15 added, +2,136 LOC.

### Shipped
- `src/reference_date.py` — `REFERENCE_DATE = date(2026, 6, 1)`; sole source.
- `src/parsing.py` — streaming `parse_jsonl` + `candidates_to_parquet` (pyarrow,
  nested-structure-preserving, no date math).
- `src/io_utils.py` — `sha256_file` (1 MiB chunks), `read_parquet`,
  `write_parquet`.
- `src/manifest.py` — frozen `Manifest` dataclass + `ArtifactError` +
  `write_manifest` / `load_manifest` / `verify_manifest`. Forward-compatible
  via `Manifest.extra` field. **Fail-loud** on missing artifacts / row-count
  drift / column drop / shape mismatch / dtype mismatch / candidates hash
  mismatch.
- `tests/conftest.py` — `synthetic_50` fixture: 50 hand-crafted candidates
  across **15 edge-case buckets** (Tier-A real fits, plain-language Tier-5
  fits, keyword stuffers, honeypots, services-only, services→product
  transition, CV/speech/robotics-only, recent-only-LangChain, inactive
  architect/VP, outside-India no-relocate, skill alias drift, missing
  assessments, high-notice/low-response, concurrent advisor, fillers).
  `by_id` session fixture maps candidate_id → dict.
- `tests/test_parsing.py` (11) + `tests/test_manifest.py` (19) — **30 green**.
  Includes a static check that `src/parsing.py` calls neither `date.today()`
  nor `datetime.now()`.
- Scaffolding: `.gitignore` (excludes 465 MB `candidates.jsonl`, `.venv/`,
  `artifacts/*.npy`, `artifacts/*.parquet`, `artifacts/model/`),
  `README.md`, `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`
  (ruff line-length=100, py311, pytest testpaths=["tests"]).

### Toolchain
- Python 3.11.15 (vendored by uv).
- uv-managed venv at `.venv/`. `uv pip install -r requirements-dev.txt`.
- ruff (lint + format) and pytest are the only dev tools. **mypy deferred to
  CP-4** once surface area justifies the cost.

### Gates at commit
- `ruff check src tests` clean.
- `ruff format --check src tests` clean.
- `pytest -v` — 30 passed in 0.29 s.

---

## CP-2 — 9 feature builders + config + pipeline ✅

**Commit:** `7037ed9` (2026-06-28).
**Files:** 31 added/modified, +2,571 LOC.

### Feature builders (`src/features/*`)

| Module | Output | Surface |
|---|---|---|
| `evidence_channels.py` | `retrieval_evidence ∈ [0,1]`, `channel_hits_anywhere` (x/y/z counts) | 3-channel detection (X exact / Y plain-language / Z ownership), recency-weighted aggregation. `α=0.6, β=0.4, z_bonus_cap=0.3`. |
| `skill_trust.py` | `skill_depth_trust ∈ [0,1]` | Conditional assessment-trust: `evidence_factor × trust_factor`. Halves trust on adv/expert claims with assessment < 50. Aliases normalized via `config/aliases.yaml`. |
| `title_career.py` | `title_career_fit ∈ [0,1]` | Recency-weighted positive/negative title regex on fielded `title` only (no embedding-based negation). |
| `behavioral.py` | 5 split signals | `availability ∈ [0.55, 1.05]` multiplier; `contactability` boolean (top-10 gate); `market_interest ∈ [0,1]` tiebreaker; `external_validation` additive boost ≤ 0.05; `logistics` multiplier + top-10 gate. |
| `honeypot_ledger.py` | `risks: list[str]`, `risk_score ∈ [0,1]`, `drop`, `audit` | Interval flatten + **8 signals** (impossible_future_start, impossible_past_start, severe/huge yoe-vs-span, role_duration_mismatch per role, zero_duration_expert, skill_count_anomaly, education_chronology_anomaly, suspicious_perfect). `drop ≥ 0.65`; `audit ≥ 0.35`. Concurrent-advisor profiles correctly NOT flagged. Title/history semantic-distance signal (problem.md §1.8) deferred to CP-4 (needs embeddings). |
| `must_haves.py` | 6 graded scalars | `has_production_retrieval_evidence`, `has_vector_or_hybrid_search_evidence`, `has_python_backend_depth`, `has_ranking_eval_evidence`, `has_product_company_applied_ml_context`, `has_shipper_signal`. Tier-driving signals that live OUTSIDE the linear blend. |
| `anti_patterns.py` | `ceiling: str \| None`, `archetypes: list[str]` | 6 archetypes → `rank_50` / `rank_100` ceiling tags. Non-tech-title, non-tech-industry, services-only (only fires if no product context), CV/speech/robotics-only (with required-absent NLP/IR check), recent-only-LangChain (`<12mo` window), inactive-architect (no production code in 18mo). |
| `stuffer_risk.py` | `stuffer_risk ∈ [0,1]` | 5-term clipped score: `0.6*(>=6 adv/expert AI skills) + 0.3*(zero channel-X/Y) + 0.2*(>=4 AI skills + no AI assess) + 0.2*(non-tech title) - 0.4*(production_retrieval > 0.3)`. |
| `tiering.py` | `tier ∈ {A,B,C,D,E}`, `TIER_PRIORITY` | A: ≥4 must-haves at tier-A thresholds. B: ≥2 at relaxed threshold + shipper ≥0.5. C: adjacent (Python/product/shipper). D: non-tech-or-skills-only. E: honeypot_drop OR stuffer_risk ≥ 0.7. |

### Config (`config/*.yaml`)

Every regex list, archetype rule, weight, and threshold lives in YAML so
tuning between submissions is a config change + ablation run, not a code
change.

- `regex_channels.yaml` — channel-X/Y/Z term lists + alpha/beta/z_bonus_cap.
- `aliases.yaml` — skill name normalization (Fine-tuning LLMs → Fine-tuning, etc.).
- `anti_patterns.yaml` — 6 archetypes' rules + ceiling tags.
- `thresholds.yaml` — honeypot weights + drop/audit thresholds, stuffer
  AI-core skill list, tier thresholds.
- `weights.yaml` — linear-blend weights for CP-4 `rank.py`.
- `jd_intents.yaml` — 8 positive-intent strings for embedding (CP-3 consumed).

### Pipeline + CLI
- `src/feature_pipeline.py` — `build_feature_row(candidate, reference)` wires
  every feature builder into one flat dict (Parquet-friendly).
  `build_features_df` materializes the DataFrame.
  `FEATURE_COLUMNS: tuple[str, ...]` = the 25 expected output columns.
- `src/config_loader.py` — `@functools.cache`-d YAML loader.
- `build_features.py` — argparse CLI; CP-2 form does parse → parquet →
  pipeline → summary (no embeddings yet).

### Tests (10 new files, +96 cases → **126 total green**)

`test_evidence_channels`, `test_skill_trust`, `test_title_career`,
`test_behavioral`, `test_honeypot_ledger`, `test_must_haves`,
`test_anti_patterns`, `test_stuffer_risk`, `test_tiering`,
`test_feature_pipeline`.

### Smoke result on 50-candidate fixture
```
tier histogram   {A: 18, B: 1, C: 14, D: 9, E: 8}
honeypot drops    3   (CAND_14/15/16 — all 3 fixture honeypots)
audit only       13
rank_50 ceilings 17  (stuffers + services-only + CV + LC + architect)
```

All Tier-A real fits → Tier A. All stuffers → Tier E. Concurrent advisor
(CAND_40) NOT dropped. Anti-pattern ceilings fire on the expected archetypes.

### Fixture changes during CP-2
- `CAND_0000015`: `duration_months` 12 → 84. Triggers
  `role_duration_mismatch` (0.2) alongside `severe_yoe_span_mismatch` (0.5)
  for total 0.7 ≥ 0.65 drop threshold.

### Bug fixes during CP-2
- `recent_only_langchain_ceiling`: was treating current roles (end_date=None)
  as ancient (months_ago=9999); fixed to treat them as recent (months_ago=0).
- `_months_between` in `honeypot_ledger`: was using `days/30.4375` (truncation
  caused 2-yr intervals → 23 months instead of 24); fixed to compute months as
  `(end.year - start.year) * 12 + (end.month - start.month)` with day-of-month
  adjustment.
- `verify_manifest`: was reading physical Parquet schema names (which expose
  inner `element` fields of list types instead of the list column name);
  switched to `schema_arrow.names` which gives logical column names. (Fixed
  during CP-3 smoke testing.)

---

## CP-3 — Embeddings + full build_features.py ✅

**Commit:** `<TBD-pending-CP-3-commit>` (2026-06-28).
**Files:** 9 added/modified, +~600 LOC (`src/embeddings.py`,
`tools/vendor_model.py`, `tests/test_embeddings.py`,
`tests/test_build_features_e2e.py`, full-form `build_features.py`,
README + progress + completed updates, `requirements.txt`,
`src/manifest.py` schema-name fix).

### Shipped this session

- `src/embeddings.py`:
  - `candidate_doc(record)` — assembles `headline . summary . role_descs .
    skills: <names>`, capped at `DOC_CHAR_CAP = 4096` chars.
  - `load_model(model_dir)` — load a vendored SentenceTransformer; **forces
    CPU**; raises `FileNotFoundError` if model_dir absent.
  - `encode_strings` / `encode_candidates` / `encode_jd_intents` — batch
    encode → unit-normed `float16` ndarray (shape `(N, 384)`).
  - `hash_model_dir(model_dir)` — deterministic sha256 over the entire model
    directory (sorted file paths + file contents). Used for manifest
    integrity.
  - Constants: `EMBEDDING_DIM = 384`, `DEFAULT_BATCH_SIZE = 512`.

- `tools/vendor_model.py` — one-off CLI to download
  `BAAI/bge-small-en-v1.5` from HuggingFace and save it under
  `artifacts/model/`. Idempotent: skips download if the dir already looks
  vendored (config.json or pytorch_model.bin or model.safetensors or
  modules.json present).

- `build_features.py` (full Phase-3 form). New pipeline:
  1. Vendor model if `<out>/model/` empty (delegates to
     `tools.vendor_model.vendor`).
  2. Parse JSONL.
  3. Write `candidates.parquet`.
  4. Load model (CPU); encode candidate docs; save
     `candidate_emb.npy` (float16, unit-normed, shape `(N, 384)`).
  5. Load `jd_intents` from `config/jd_intents.yaml`; encode; save
     `jd_intent_vecs.npy` (shape `(8, 384)`).
  6. Run feature pipeline; save `features.parquet`.
  7. Build manifest from artifact metadata + run `sha256_file` on
     `candidates.jsonl` and `requirements.txt`; save `manifest.json`.
  8. Write `build_features_summary.json` (tier histogram, sizes, shapes —
     human-readable).
  - CLI args: `--candidates`, `--out`, `--model-dir`, `--model-name`,
    `--batch-size`, `--no-progress`.
  - Env: forces `TRANSFORMERS_VERBOSITY=error`,
    `TOKENIZERS_PARALLELISM=false` to keep logs clean.

- `tests/test_embeddings.py` — 9 cases.
  - Unit tests (always run): `candidate_doc` structure, length cap, missing
    fields, empty descriptions; `candidate_docs` list shape;
    `hash_model_dir` determinism + content sensitivity + new-file
    sensitivity; `load_model` raises on missing dir.
  - **Integration tests** (skipped if `artifacts/model/` absent — guarded by
    `_model_available()`): real BGE encoding produces shape `(N, 384)`
    `float16` unit-normed; JD intent encoding shape `(8, 384)`.

### Dependencies added
- `sentence-transformers>=2.7,<3.0` → pulls `torch==2.12.1`,
  `transformers==4.57.6`, `tokenizers`, `safetensors`, `scipy`,
  `scikit-learn`, `huggingface-hub`. CPU-only (macOS Apple Silicon, no CUDA).
- `requirements.txt` updated to declare `sentence-transformers` as a
  production runtime dep with a comment that **rank.py never imports it**.

### Vendored model artifacts
- `artifacts/model/BAAI_bge-small-en-v1.5/` ← actually saved directly under
  `artifacts/model/` per `model.save(str(out_dir))`.
- 11 files, 134.5 MB total. **gitignored** (`.gitignore` excludes
  `artifacts/model/`). Re-vendored via `python tools/vendor_model.py` on
  fresh clones / Stage-3 Docker.

### Smoke result on 50-candidate fixture (synthetic_50)
```
[build_features] model ready in 0.0s
[build_features] parsed 50 candidates in 0.0s
[build_features] wrote candidates.parquet in 0.0s
[build_features] loaded model in 1.8s
[build_features] encoded 50 candidates in 0.7s — emb shape=(50, 384) dtype=float16
[build_features] encoded 8 JD intents in 0.0s — emb shape=(8, 384)
[build_features] computed + wrote features.parquet (50 rows) in 0.0s
[build_features] wrote manifest in 0.1s
```
`verify_manifest(...)` succeeds. Tier histogram unchanged vs CP-2:
`{A:18, B:1, C:14, D:9, E:8}`.

### Full 100K run (completed)
- Source: `/Users/dhruvsanan/Desktop/India_runs/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl` (465 MB).
- Out: `./artifacts/`.
- Hardware: Apple M4, arm64, CPU device forced (`device="cpu"`).
- **Wall-clock: 64 min 5 s total**. Per-step:
  - parse 100K JSONL → 2.8 s
  - write `candidates.parquet` (14 MB) → 2.2 s
  - load model → 3.2 s
  - **encode 100K candidates → 3670.8 s (61.2 min)** — the dominant cost
  - encode 8 JD intents → 0.1 s
  - feature pipeline + `features.parquet` (3.7 MB) → 164.2 s
  - write `manifest.json` → 0.3 s
- The encoder under-utilizes M4: only ~2 of 10 cores active under
  `device="cpu"`. Originally predicted 5–8 min was optimistic; the real
  prediction for cold-clone reproduction is **~60 min on M4 CPU** (longer on
  x86). The encoder cost is amortized — `rank.py` never re-runs it.
- Tier histogram: `{A: 82, B: 384, C: 10188, D: 87275, E: 2071}` (sum = 100000).
- `honeypot_drop` = **348** (above the 80–150 calibration target — flag for
  CP-4 review: thresholds may need tightening once we see top-100 audit
  results).
- `honeypot_audit` = 15143.
- Anti-pattern rank_50 ceilings = 41093 (`none` = 58907).
- Artifact total bytes (excluding `model/`) = 95.7 MB; with `model/` = 230 MB.
  Well under 5 GB cap.
- `verify_manifest(...)` against the produced artifacts: PASS.

### Gates at CP-3 commit
- `ruff check src tests tools` clean.
- `ruff format --check src tests tools` clean.
- `pytest -q` — **138 passed** (137 unit + 1 new e2e). No skips when the
  vendored model is present; the BGE integration + e2e tests are
  `pytest.skip`-guarded for cold-clone CI without the model.

---

## Cumulative repo layout (after CP-3 commit lands)

```
redrob-ranker/
├── README.md
├── completed.md                ← this file (offload doc)
├── progress.md                 ← 2-line entry per CP
├── pyproject.toml              # ruff/pytest config, python = ">=3.11,<3.12"
├── requirements.txt            # production runtime
├── requirements-dev.txt        # + pytest + ruff
├── build_features.py           # ENTRY (offline, ≤10 min on 100K)
├── config/
│   ├── aliases.yaml
│   ├── anti_patterns.yaml
│   ├── jd_intents.yaml
│   ├── regex_channels.yaml
│   ├── thresholds.yaml
│   └── weights.yaml
├── src/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── embeddings.py            # BGE-small encoding
│   ├── feature_pipeline.py      # row-builder + FEATURE_COLUMNS
│   ├── io_utils.py
│   ├── manifest.py              # Manifest + verify_manifest + ArtifactError
│   ├── parsing.py
│   ├── reference_date.py        # REFERENCE_DATE = date(2026, 6, 1)
│   └── features/
│       ├── __init__.py
│       ├── anti_patterns.py
│       ├── behavioral.py
│       ├── evidence_channels.py
│       ├── honeypot_ledger.py
│       ├── must_haves.py
│       ├── skill_trust.py
│       ├── stuffer_risk.py
│       ├── tiering.py
│       └── title_career.py
├── tests/
│   ├── conftest.py              # synthetic_50 + by_id fixtures
│   ├── test_anti_patterns.py
│   ├── test_behavioral.py
│   ├── test_embeddings.py       # NEW in CP-3
│   ├── test_evidence_channels.py
│   ├── test_feature_pipeline.py
│   ├── test_honeypot_ledger.py
│   ├── test_manifest.py
│   ├── test_must_haves.py
│   ├── test_parsing.py
│   ├── test_skill_trust.py
│   ├── test_stuffer_risk.py
│   ├── test_tiering.py
│   └── test_title_career.py
├── tools/
│   └── vendor_model.py          # NEW in CP-3
└── artifacts/                   # gitignored except model/ (also gitignored)
    ├── manifest.json            # ← only file kept; .npy/.parquet ignored
    ├── candidates.parquet       # gitignored (large)
    ├── features.parquet         # gitignored
    ├── candidate_emb.npy        # gitignored
    ├── jd_intent_vecs.npy       # gitignored
    └── model/                   # gitignored, re-vendored per clone
```

---

## Commit history

```
<TBD>    feat(embeddings): vendored BGE-small + full build_features pipeline (CP-3)
7037ed9  feat(features): 9 feature builders + config + pipeline + build_features.py skeleton (CP-2)
13542e4  feat(skeleton): repo init + reference_date + parsing + manifest + 50-candidate fixture (CP-1)
```

---

## What's left (post-CP-3)

| CP | Scope | ETA |
|---|---|---|
| **CP-4** | `src/scoring.py` (capped contributions + linear blend), `src/ranking.py` (tier sort + top-10 promotion gate + relaxation rule), `src/reasoning.py` (evidence ledger + ~20 skeletons), `rank.py` (entry point with **import allow-list static check**), `reasoning_audit.py`. End-to-end test through `validate_submission.py`. | ~3 hr |
| **CP-5a** | Hand-label `holdout_labels.csv` across 9 stratified buckets; `holdout_report.md` bucket assertions. | ~1.5 hr |
| **CP-5b** | `ablations.py` (A0–A5 variants); `ablation_report.md`; `runtime_report.md`. | ~1 hr |
| **CP-5c** | HuggingFace Space (or Docker fallback) live, ≤ 5 min on free tier. | ~1.5 hr |
| **CP-5d** | 11-slide PPTX → PDF per `pptx.md`. | ~1.5 hr |
| **CP-S1** | Floor submission. All 11 `hld.md` blocking checks pass. | day-3 evening |
| **CP-S2** | Tuned weights from holdout + ablation deltas + top-25 manual review. | day-4 evening |
| **CP-S3** | Final, cold-clone-verified. | day-5 evening |

---

## Resuming from a fresh session

1. `cd /Users/dhruvsanan/Desktop/India_runs/redrob-ranker`
2. `source .venv/bin/activate` (Python 3.11.15 inside).
3. `git log --oneline | head -3` to confirm latest commit.
4. Read this file + the latest entry in `progress.md`.
5. Check `git status` — should be clean unless CP-3 commit not yet made.
6. If model dir absent (`ls artifacts/model/`): `python tools/vendor_model.py`.
7. `pytest -q` — should be green at the latest CP boundary.
8. Open the next-CP section in `lld.md §2` and follow its checklist.

Do **not** re-derive architecture decisions from `problem.md`. They are locked.
Read `problem.md` §3 v2 only if a *new* design question arises that isn't
already settled.
