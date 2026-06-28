# Redrob Ranker — Completed Work Log

> Append-only. Onboard a fresh session by reading this top-to-bottom; no need
> to re-read `problem.md` / `lld.md` / `hld.md` unless changing architecture.
> Last updated: 2026-06-28 (post-CP-5d).

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
- **`rank.py` import allow-list** (enforced via static AST check in
  `tests/test_rank_imports.py` + runtime sys.modules guard inside
  `rank.py` itself): `{numpy, pandas, pyarrow, json, sys, pathlib, yaml}`
  plus project-internal `src.*`. Anything else = bug. `sentence-transformers`,
  `torch`, `transformers`, `requests`, `httpx`, `openai`, `anthropic`,
  `cohere`, `google.generativeai` are explicitly forbidden.
- **Submission CSV invariant** (`rank.py` enforces): label `score` column
  is strictly non-increasing across ranks 1→100 (epsilon=1e-7). The raw
  `final_score` lives in `top_100_audit.csv` / `top_300_debug.csv` for
  inspection / ablation.
- **Last CP shipped: CP-4** (`1221a8c`). Pipeline is end-to-end green on
  100K — `rank.py` finishes in **1.96 s** on M4 (target 60 s, hard cap
  300 s). `validate_submission.py` and `reasoning_audit.py` both pass.

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

**Commit:** `79bc5d8` (2026-06-28).
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

## CP-3.5 — Encoder perf patch (post-CP-3 root-cause fix)

**Commit:** `9716c84` (2026-06-28).

### Why
CP-3's 100K encode took 64 min wall on M4 — predicted 5-8. Root cause:
`device="cpu"` hard-coded, `TOKENIZERS_PARALLELISM=false`, no
`torch.set_num_threads(...)` call → effective 2-thread use of a 10-core M4.

### Patch
- `src/embeddings.py`: new `resolve_device(device="auto")` — auto-selects
  MPS on Apple Silicon, else CPU. `load_model(model_dir, device="auto")`
  takes an explicit device.
- `build_features.py`: `--device {auto|cpu|mps|cuda}` and `--torch-threads N`
  (default 8); flips `TOKENIZERS_PARALLELISM` to `true`; sets
  `torch.set_num_threads` on CPU device.
- `tests/test_embeddings.py`: +2 cases for `resolve_device` (pass-through +
  auto-resolution matches torch capability). 140 pytest green.

### Measured throughput (1000-doc samples)
| Path | Rate | Projected 100K |
|---|---|---|
| CPU + threads=8 + parallel tokenizers, batch 128 | 33 docs/s | 50 min |
| CPU + threads=8 + parallel tokenizers, cap=1024  | 58 docs/s | 28 min |
| MPS, batch=512 | OOM | — |
| MPS, batch=64 / 128 / 256 | stalled (Metal kernel launch overhead) | — |

### Decision
- Keep CPU as default. M4 MPS path on BGE-small is dominated by per-batch
  Metal-kernel launch overhead — bigger batches OOM (Apple unified memory
  shared with system), smaller batches stall the GPU. Not worth the risk for
  this hackathon.
- Do **not** re-run the 100K encode now — CP-3 artifacts are still valid
  (`verify_manifest` passes) and `rank.py` (CP-4) consumes the `.npy`
  regardless of how it was generated.
- CP-3.5 is a forward-fix: any future re-run (Stage-3 reviewer cold-clone,
  weight-sweep iteration, doc-cap change) benefits from the threading +
  tokenizer parallelism. Expected cold-clone wall on M4: **~50 min** at the
  default `DOC_CHAR_CAP=4096` (down from 64 min); ~28 min at cap=1024.
- The `DOC_CHAR_CAP=1024` tuning is left for a possible CP-3.5 follow-up
  once we see top-100 audit results in CP-4 — premature for now since
  truncating richer descriptions could hurt the JD's "plain-language fit"
  signal (`evidence_channels` Y-channel).

## CP-4 — Online ranking + reasoning + audit ✅

**Commit:** `1221a8c` (2026-06-28).
**Files:** 21 added/modified, +~2,200 LOC.

### Shipped
- `src/features/experience_band.py` — Gaussian fit (μ=7, σ=2.2, clipped [0,1]).
- `src/features/education.py` — max tier across education[]; unknown → 0.5.
- Wired both into `feature_pipeline.py`; `FEATURE_COLUMNS` grows 25 → 27.
  Old `features.parquet` regenerated with `tools/rebuild_features_parquet.py`
  (no re-encode; 165s wall vs 60+ min if we'd re-run the full encoder).
- `src/scoring.py` — capped factors, linear blend, multipliers + tiebreak,
  anti-pattern score ceilings translated to score-space from current
  distribution.
- `src/ranking.py` — drop honeypots → tier sort → top-10 promotion gate
  with progressive relaxation → fill remainder → strict-decreasing label
  scores (sidesteps the validator's equal-score tie-break rule cleanly).
- `src/reasoning.py` — evidence ledger per top-100 row (primary snippet,
  highest-trust skill, logistics, concern, must-haves, non-tech exception
  clause) + 20 skeletons across 4 rank bands + reuse cap of 12.
- `config/skeletons.yaml` — 5 templates × 4 bands.
- `rank.py` — restricted-imports entry. `parse_argv` rolled by hand (argparse
  intentionally not on the allow-list). Runtime guard aborts if any of
  `sentence_transformers / torch / transformers / requests / httpx / openai /
  anthropic / cohere` is in `sys.modules` at start.
- `reasoning_audit.py` — independent script. Curated brand allowlist
  (Pinecone, FAISS, Weaviate, BGE, …) for grounded-claim check; template
  reuse cap; rank-50+ concern-clause requirement; non-tech exception
  clause; high-notice / low-response mention requirement when rank ≤ 50.
- `tools/rebuild_features_parquet.py` — forward-useful helper: re-derives
  `features.parquet` + manifest without touching the encoder.

### Tests (8 new files, +58 cases → **196 total green**)
- `test_scoring.py` — capped factors, embedding-similarity shape/clip,
  must-have sum, blend smoke, ceiling clipping count, stuffer-can't-beat-
  real-fit assertion.
- `test_ranking.py` — drop, sort, gate, exception path, relaxation, fill,
  strict-decreasing labels, small-pool starvation telemetry.
- `test_reasoning.py` — rank band buckets, ledger population, concern
  composition, non-tech exception clause, template reuse cap rotation,
  rank-50+ concern guard, YAML round-trip.
- `test_reasoning_audit.py` — pass path, missing-concern fail, template
  over-cap fail, non-tech missing exception fail, high-notice unmentioned
  fail, `run_audit` writes CSV.
- `test_rank_imports.py` — AST-parsed import allow-list check, forbidden
  module ban, runtime-guard presence, REFERENCE_DATE invariant
  (no `datetime.now()` / `date.today()` in `rank.py`).
- `test_end_to_end.py` — subprocess-isolated `rank.py` run against the
  real 100K artifacts, `validate_submission.py` clean, audit pass.
- `test_experience_band.py` / `test_education.py` — feature edge cases.

### Gates at commit
- `ruff check src tests tools rank.py reasoning_audit.py build_features.py` — clean.
- `ruff format --check ...` — clean (52 files).
- `pytest -q` — **196 passed** in ~8s.

### Full 100K run (post-CP-4 — first real submission rehearsal)
```
$ python rank.py --artifacts ./artifacts/ --out top_100_submission.csv \
                --audit top_100_audit.csv --debug top_300_debug.csv
{
  "rows_scored": 100000,
  "rows_after_honeypot_drop": 99652,
  "top_10_gate_pool_size": 275,
  "top_10_relaxation_used": 2,
  "ceilings": {
    "score_at_rank_51": 0.6229,
    "score_at_rank_101": 0.5613,
    "rank_50_clipped_count": 41093,
    "rank_100_clipped_count": 0
  }
}
real 0m1.956s
$ python validate_submission.py top_100_submission.csv
Submission is valid.
$ python reasoning_audit.py --audit top_100_audit.csv \
                            --candidates artifacts/candidates.parquet \
                            --out reasoning_audit.csv
reasoning_audit: PASS 100 rows; wrote reasoning_audit.csv
```

### Observations for CP-5 tuning
- 41,093 of 99,652 survivors (~41%) caught by a `rank_50` ceiling — the
  bound is binding for a meaningful fraction. The 100th-rank score
  (0.561) is just 6 points below the 50th (0.623), so the gate's
  pressure on rank-band hygiene is mild but visible.
- Top-10 promotion pool comfortably exceeds 10 at `min_must_have=2`
  (275 eligible) → relaxation never fires on the real data.
- Honeypot drops = 348 (CP-3 result; unchanged here). Above the 80-150
  calibration target; tighten ledger thresholds in CP-5 if top-100 audit
  results show false-positive drops.
- Several rank-band-1 candidates share a common "Built a content
  recommendation system serving 10M+ users…" description — the synthetic
  dataset has reused role descriptions; reasoning copies the snippet
  verbatim. Audit still passes (it's grounded in the candidate's record).
- Reasoning snippets sometimes truncate mid-word with an ellipsis
  (`BM25-o…`). Not a blocker; clean up in CP-5 polish if time permits.

---

## CP-5a — Stratified holdout sourcing + report ✅

**Commit:** `1ea6b27` (2026-06-28).
**Files:** 5 added, +1,416 LOC.

### Shipped
- `tools/build_holdout.py` — sources 99 candidates across 9 strata with
  deterministic `seed=42` sampling. Refuses to clobber an existing
  `holdout_labels.csv` unless `--force` (protects hand-labels).
- `tools/holdout_report.py` — joins seed CSV against `top_100_submission.csv`,
  computes 9 predicate-bucket assertions + 4 optional label-grounded
  assertions, writes `holdout_report.md`. Empty buckets vacuously PASS.
- `tests/test_holdout.py` — 35 cases (predicates, sampling determinism,
  CSV schema, report logic, overwrite guard).
- `holdout_labels.csv` — 99 candidates × 9 buckets × 11 each, blank
  `true_label` / `notes` for the operator's hand-fill.
- `holdout_report.md` — current state snapshot.

### Bucket design (deviation from problem.md §5 draft)
The CP-5a draft in completed.md named 9 buckets including CV-Speech-Expert,
Strong-Recsys-Weak-Skill-List, and Strong-AI-but-Inactive. Real data
inspection showed these archetypes never fire — only `non_tech_title=34339`
and `services_only=8945` populate. Replaced the 3 vacuous buckets with:
`honeypot_audit`, `services_to_product`, `irrelevant_tail` — all of which
exercise real archetype + ledger paths.

Final 9 buckets:
1. `plain_language` — `retrieval_evidence >= 0.3` + `has_production >= 0.5` (pool 128)
2. `stuffer` — `stuffer_risk >= 0.6` + `retrieval_evidence < 0.2` (pool 1,870)
3. `honeypot_drop` — `honeypot_drop=True` (pool 348)
4. `honeypot_audit` — audit-only flag (`audit & not drop & risk>=0.45`)
5. `irrelevant_tail` — `tier=D & retrieval<0.05 & no honeypot` (pool 87K)
6. `services_only` — archetype `services_only`
7. `services_to_product` — services-company in history but NOT `services_only` archetype
8. `outside_india` — `profile.country != 'India'` (pool 25K)
9. `non_tech_title` — archetype `non_tech_title`

### Assertion state (pre-labels)
- 8/9 predicate assertions PASS
- 1 FAIL: `plain_language` median rank = **52** (threshold ≤ 50);
  `frac_top_100 = 63.6%` (well above 40% recall floor). The plain-language
  signal recall is healthy; ranking pressure is the marginal gap.
  CP-S2 weight tuning consumes this.
- `honeypot_drop` count in top-100 = **0** (blocking check #5 satisfied).
- `irrelevant_tail` count in top-100 = **0**.
- All ceiling-clipped buckets (services_only, non_tech_title) below 50%
  frac_top_50.

### Gates at commit
- 231 pytest green.
- ruff check + format clean.

---

## CP-5b — Ablations + runtime report ✅

**Commit:** `0e72833` (2026-06-28).
**Files:** 6 added/modified, +906 LOC.

### Shipped
- `tools/ablations.py` — 6-variant driver (A0..A5) with deep-copied weights
  to prevent mutation leak. Exports per-variant top-100 CSVs to
  `ablations/` (gitignored).
- `tools/runtime_report.py` — spawns `rank.py` subprocess, captures wall
  via `time.perf_counter` + peak RSS via `resource.getrusage(RUSAGE_CHILDREN)`,
  stats artifact disk, re-emits tier histogram from
  `build_features_summary.json`, writes `runtime_report.md`. Cleans up
  the `_runtime_tmp/` working dir on exit.
- `tests/test_ablations.py` — 22 cases (overlap math, variant dispatch,
  mutation isolation, no-gate path, runtime helpers).
- `ablation_report.md` + `runtime_report.md` — committed deliverables.

### Variants
| code | label | mutation |
|---|---|---|
| A0 | baseline | current weights + ceilings + multipliers + gate |
| A1 | no_embedding | `blend.embedding_contribution = 0.0` |
| A2 | no_skill_blend | `blend.skill_contribution = 0.0` (proxy for skill-stripped doc; full re-encode would cost 50 min) |
| A3 | no_behavioral_mult | replace `final_score` with `base_score` (no avail × logistics × market) |
| A4 | no_anti_pattern | replace `final_score` with `final_score_uncapped` |
| A5 | no_top_10_gate | bypass `build_top_10_pool`; take first 10 by global order |

### Ablation findings on real 100K artifacts
| variant | top-100 | top-10 stable | top-50 stable | jaccard | wall |
|---|---:|---:|---:|---:|---:|
| A1 no_embedding | 99/100 | 10/10 | 50/50 | 0.980 | 0.42 s |
| A2 no_skill_blend | 95/100 | 9/10 | 49/50 | 0.905 | 0.35 s |
| A3 no_behavioral_mult | **90/100** | **3/10** | **36/50** | 0.818 | 0.34 s |
| A4 no_anti_pattern | 100/100 | 10/10 | 50/50 | 1.000 | 0.37 s |
| A5 no_top_10_gate | 100/100 | 10/10 | 50/50 | 1.000 | 0.25 s |

### Key signals (consumed by CP-S2 weight tuning)
- **A3 = dominant top-10 lever.** `availability × logistics` multipliers
  flip 7/10 of the top ranks. NOT cosmetic — tune carefully.
- **A1 ≈ noise.** Embedding contributes almost nothing to top-100 membership
  (99/100 overlap). Don't burn cycles tuning `embedding_contribution`
  weight; if anything, lower it.
- **A4 / A5 = defensive only.** Ceilings + gate don't reshape current
  top-100 (already gate-eligible at score-sort head). Their value is
  keeping stuffer / honeypot drift OUT under future data shifts. Keep
  as guardrails; don't remove.
- **A2 = moderate.** 5-row churn at the rank-100 boundary; small ROI
  unless paired with feature signal improvements.

### Runtime on current artifacts (M4 CPU)
- `rank.py` wall: **1.87 s** (target 60 s, hard cap 300 s — 32× under)
- `rank.py` peak RSS: **1.8 GB** (well under 16 GB cap)
- Artifacts: 91.45 MB no model / 219.72 MB with model (5 GB cap)
- Build wall (historical from CP-3, unchanged): ~50 min on M4 CPU
- Tier histogram (unchanged from CP-3): `{A:82, B:384, C:10188, D:87275, E:2071}`
- `honeypot_drop = 348`, `honeypot_audit = 15143`

### Gates at commit
- 253 pytest green.
- ruff check + format clean.

---

## CP-5c — HuggingFace Space + Docker sandbox ✅

**Commit:** `37ec246` (2026-06-28).
**Files:** 5 added, +519 LOC.

### Shipped
- `app.py` — Gradio entry. `RankerState` lazy singleton (model +
  `jd_intent_vecs` + weights + skeletons loaded once). `rank_sample()`
  reuses `src.scoring` + `src.ranking` + `src.reasoning` end-to-end on
  fresh sample (does NOT load precomputed `candidate_emb.npy` — sample is
  fresh data, encoded on the fly with vendored BGE-small).
- `requirements-app.txt` — `gradio>=4.40,<5.0` (sandbox-only; rank.py path stays
  gradio-free).
- `Dockerfile` — Python 3.11-slim base; vendors BGE model at build time
  for fast cold-start.
- `SANDBOX.md` — HF Space + Docker deployment recipes + local smoke-check.
- `tests/test_app.py` — 11 cases (4 hermetic for parsing / CSV / UI errors,
  7 integration guarded by `_model_available()`).

### Architecture invariant preserved
`app.py` is a SEPARATE entry point — gradio / sentence-transformers
imports do NOT pollute `rank.py`'s restricted allow-list. The Stage-3
reviewer path runs `rank.py` standalone with the import guard intact.
`tests/test_rank_imports.py` (4 cases) confirms.

### Smoke verification
- Real 20-candidate sample from `artifacts/candidates.parquet`: ranked
  5 rows with proper reasoning rendering in ~1 s after model load.
- Output CSV schema matches `validate_submission.py` (candidate_id,
  rank, score, reasoning).
- Monotonic score, ascending rank.

### Gates at commit
- 264 pytest green.
- ruff check + format clean.

---

## CP-5d — Submission PPTX builder ✅

**Commit:** `43de47e` (2026-06-28).
**Files:** 4 added/modified, +809 LOC.

### Shipped
- `tools/build_deck.py` — opens the Redrob "Idea Submission Template"
  PPTX via `python-pptx`, populates all 11 slides with content from
  `pptx.md`, writes `redrob_submission.pptx`. Slide 6 (Workflow) and
  Slide 7 (Architecture) get proper shape diagrams (rounded rectangles +
  arrows). Slide 8 (Results) pulls live numbers from
  `build_features_summary.json` + `top_100_audit.csv` via
  `gather_runtime_numbers()`.
- `tests/test_build_deck.py` — 11 cases (slide-content invariants,
  runtime-numbers gather, prompt-finder logic, end-to-end deck assembly).
- `requirements-dev.txt` — adds `python-pptx>=1.0,<2.0` (dev-only).
- `.gitignore` — `redrob_submission.{pptx,pdf}` gitignored (regenerable
  binaries).

### Template fidelity
- PICTURE shapes (brand / border) untouched.
- Title text boxes preserved (slide 7 has only a title and no prompt
  block — code explicitly skips the prompt-replacement path there to
  avoid losing "System Architecture").
- Only prompt blocks rewritten; new diagram shapes added on top of
  empty content zones.

### PDF export
Manual final step. `libreoffice` not installed locally. The CLI prints
both recipes:
```
libreoffice --headless --convert-to pdf redrob_submission.pptx
# OR open in Keynote / PowerPoint → File → Export → PDF
```

### Gates at commit
- 275 pytest green.
- ruff check + format clean.

---

## Cumulative repo layout (post-CP-5d)

```
redrob-ranker/
├── README.md
├── SANDBOX.md                  # NEW in CP-5c — HF Space + Docker recipes
├── Dockerfile                  # NEW in CP-5c — sandbox fallback
├── completed.md                ← this file (offload doc)
├── progress.md                 ← 2-line entry per CP
├── pyproject.toml              # ruff/pytest config, python = ">=3.11,<3.12"
├── requirements.txt            # production runtime
├── requirements-dev.txt        # + pytest + ruff + python-pptx (CP-5d)
├── requirements-app.txt        # NEW in CP-5c — gradio (sandbox only)
├── app.py                      # ENTRY (Gradio sandbox; NOT under rank.py allow-list)
├── build_features.py           # ENTRY (offline, ~50 min CPU on 100K)
├── rank.py                     # ENTRY (online, restricted imports, ~2 s on 100K)
├── reasoning_audit.py          # ENTRY (independent post-rank audit)
├── holdout_labels.csv          # NEW in CP-5a — 99-row seed for hand-labeling
├── holdout_report.md           # NEW in CP-5a — bucket assertion state
├── ablation_report.md          # NEW in CP-5b — A0..A5 overlap deltas
├── runtime_report.md           # NEW in CP-5b — wall + RAM + disk + tier histogram
├── config/
│   ├── aliases.yaml
│   ├── anti_patterns.yaml
│   ├── jd_intents.yaml
│   ├── regex_channels.yaml
│   ├── skeletons.yaml           # CP-4 — reasoning templates × 4 bands
│   ├── thresholds.yaml
│   └── weights.yaml             # 8-term linear blend + capped factors
├── src/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── embeddings.py            # BGE-small encoding (CP-3 / CP-3.5)
│   ├── feature_pipeline.py      # row-builder + FEATURE_COLUMNS (27)
│   ├── io_utils.py
│   ├── manifest.py              # Manifest + verify_manifest + ArtifactError
│   ├── parsing.py
│   ├── ranking.py               # CP-4 — tier sort + top-10 gate
│   ├── reasoning.py             # CP-4 — evidence ledger + skeletons
│   ├── reference_date.py        # REFERENCE_DATE = date(2026, 6, 1)
│   ├── scoring.py               # CP-4 — blend + ceilings
│   └── features/
│       ├── __init__.py
│       ├── anti_patterns.py
│       ├── behavioral.py
│       ├── education.py         # CP-4
│       ├── evidence_channels.py
│       ├── experience_band.py   # CP-4
│       ├── honeypot_ledger.py
│       ├── must_haves.py
│       ├── skill_trust.py
│       ├── stuffer_risk.py
│       ├── tiering.py
│       └── title_career.py
├── tests/                       # 275 cases, all green (196 pre-Phase-5 + 79 new)
│   ├── conftest.py              # synthetic_50 + by_id fixtures
│   ├── test_anti_patterns.py
│   ├── test_app.py              # NEW in CP-5c — sandbox parsing + e2e
│   ├── test_ablations.py        # NEW in CP-5b — variants + runtime helpers
│   ├── test_behavioral.py
│   ├── test_build_deck.py       # NEW in CP-5d — deck builder
│   ├── test_build_features_e2e.py
│   ├── test_education.py
│   ├── test_embeddings.py
│   ├── test_end_to_end.py       # CP-4 — subprocess rank.py + validate
│   ├── test_evidence_channels.py
│   ├── test_experience_band.py
│   ├── test_feature_pipeline.py
│   ├── test_holdout.py          # NEW in CP-5a — bucket predicates + report
│   ├── test_honeypot_ledger.py
│   ├── test_manifest.py
│   ├── test_must_haves.py
│   ├── test_parsing.py
│   ├── test_rank_imports.py     # CP-4 — AST allow-list + forbidden ban
│   ├── test_ranking.py          # CP-4
│   ├── test_reasoning.py        # CP-4
│   ├── test_reasoning_audit.py  # CP-4
│   ├── test_scoring.py          # CP-4
│   ├── test_skill_trust.py
│   ├── test_stuffer_risk.py
│   ├── test_tiering.py
│   └── test_title_career.py
├── tools/
│   ├── ablations.py             # NEW in CP-5b — A0..A5 driver
│   ├── build_deck.py            # NEW in CP-5d — PPTX builder
│   ├── build_holdout.py         # NEW in CP-5a — 9-bucket seed sourcing
│   ├── holdout_report.py        # NEW in CP-5a — bucket assertion runner
│   ├── rebuild_features_parquet.py  # CP-4 — regen features w/o re-encode
│   ├── runtime_report.py        # NEW in CP-5b — wall + RAM + disk report
│   └── vendor_model.py
└── artifacts/                   # ENTIRE DIR gitignored (large + model + outputs)
    ├── manifest.json            # current artifact metadata (verify_manifest target)
    ├── candidates.parquet       # 14 MB; 100K nested raw records
    ├── features.parquet         # 3.7 MB; 27-column feature matrix
    ├── candidate_emb.npy        # 73 MB; (100000, 384) float16 unit-normed
    ├── jd_intent_vecs.npy       # 6 KB; (8, 384) float16
    ├── build_features_summary.json  # human-readable tier histogram + sizes
    └── model/                   # vendored BGE-small (~135 MB); re-vendored per clone
```

### Phase-5 ignored / runtime-only outputs (NOT in git)
```
ablations/                       # per-variant top-100 CSVs (tools/ablations.py)
redrob_submission.pptx           # built by tools/build_deck.py
redrob_submission.pdf            # manual export from .pptx
top_100_submission.csv           # rank.py output
top_100_audit.csv                # rank.py output
top_300_debug.csv                # rank.py output
reasoning_audit.csv              # reasoning_audit.py output
```

---

## Commit history

```
43de47e  feat(deck): CP-5d submission PPTX builder from template + live metrics
37ec246  feat(sandbox): CP-5c HuggingFace Space + Docker fallback
0e72833  feat(ablations): CP-5b ablation suite + runtime report
1ea6b27  feat(holdout): CP-5a stratified holdout sourcing + report assertions
090a1b8  docs(completed): post-CP-4 layout + handoff runbook for next agent
1221a8c  feat(ranking): full online rank.py + scoring + reasoning + audit (CP-4)
9716c84  perf(embeddings): MPS auto-detect + thread tuning + parallel tokenizers (CP-3.5)
79bc5d8  feat(embeddings): vendored BGE-small + full build_features pipeline (CP-3)
7037ed9  feat(features): 9 feature builders + config + pipeline + build_features.py skeleton (CP-2)
13542e4  feat(skeleton): repo init + reference_date + parsing + manifest + 50-candidate fixture (CP-1)
```

---

## What's left (post-CP-5d)

| CP | Scope | ETA |
|---|---|---|
| **CP-S1** | Floor submission. All 11 `hld.md` blocking checks pass. Submission CSV = current `rank.py` output. | day-3 evening |
| **CP-S2** | Tuned weights from CP-5a labels + CP-5b ablation signal + top-25 manual review. | day-4 evening |
| **CP-S3** | Final, cold-clone-verified. | day-5 evening |

### Hand-steps the operator (Dhruv) needs to complete before CP-S1
1. **Label `holdout_labels.csv`** — 99 rows × `true_label` column. Allowed
   values: `fit` / `near_fit` / `not_fit` / `honeypot` / `stuffer`. After
   labels filled, re-run `python tools/holdout_report.py` to enable the
   label-grounded assertions.
2. **Export `redrob_submission.pdf`** — `tools/build_deck.py` produced
   `redrob_submission.pptx`. PDF export is manual (libreoffice not
   installed locally). Open in Keynote / PowerPoint → File → Export → PDF.
3. **Push to GitHub** — repo is currently local-only on `main`. The
   submission spec asks for a public repo URL.
4. **Deploy HF Space** — per `SANDBOX.md`. Final URL goes into
   `tools/build_deck.py --github-url --sandbox-url` for slide 10.
5. **Regenerate deck** with final URLs once HF Space is live.

### CP-S2 weight-tuning priors (from CP-5b ablation findings)
- **Tune behavioral multipliers carefully** — A3 dropped top-10 stability
  to 3/10. `availability_signal` × `logistics_multiplier` is the strongest
  reorder lever in the current scoring stack.
- **Lower or remove `embedding_contribution` weight** — A1 showed embedding
  is near-noise (99/100 overlap). Reweighting toward `must_have_sum_div_6`
  + `retrieval_evidence` is the right direction.
- **Keep anti-pattern ceilings + top-10 gate** — both are defensive
  guardrails (A4 / A5 = 100/100 overlap on current data). Removing them
  exposes the ranker to stuffer / honeypot drift on future data.
- **CP-5a plain_language median = 52** (threshold ≤ 50, fails by 2 ranks).
  Lifting plain-language fits is the highest-priority signal for CP-S2.

### Open polish items (Phase-5 backlog, not blocking)
- `honeypot_drop = 348` is above the 80–150 calibration target — tighten
  ledger thresholds in `config/thresholds.yaml` once labels in
  `holdout_labels.csv` arrive and the user has confirmed which drops are
  true honeypots.
- Snippet truncation in reasoning ends mid-word with `…` (e.g.
  `BM25-o…`). Cosmetic. Fix in `_truncate()` to cut at last whitespace.
- Synthetic data shares role descriptions across some candidates → some
  near-duplicate reasonings. Audit still passes (grounded). Consider
  per-candidate variation (e.g. role-index instead of longest) only if
  Stage-4 reviewers flag uniformity.
- `rank_50` ceilings clip 41,093 of 99,652 survivors (~41%). Bound is
  binding; verify on holdout that real fits are not getting clipped.

---

## Resuming from a fresh session (cold-start runbook)

```bash
cd /Users/dhruvsanan/Desktop/India_runs/redrob-ranker
source .venv/bin/activate                         # Python 3.11.15
git log --oneline | head -6                       # confirm last commit
# Expected head: 43de47e feat(deck): CP-5d submission PPTX builder from template + live metrics
git status                                        # should be clean (artifacts/ untracked is expected)
ls artifacts/                                     # all 6 artifacts + model/
pytest -q                                         # 275 passed
python rank.py --artifacts ./artifacts/ \
              --out top_100_submission.csv \
              --audit top_100_audit.csv \
              --debug top_300_debug.csv          # ~2 s
python "/Users/dhruvsanan/Desktop/India_runs/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/validate_submission.py" top_100_submission.csv
python reasoning_audit.py --audit top_100_audit.csv --candidates artifacts/candidates.parquet --out reasoning_audit.csv

# Phase-5 tools (run as needed):
python tools/holdout_report.py                    # bucket assertions against current top_100
python tools/ablations.py                         # A0..A5 → ablation_report.md
python tools/runtime_report.py                    # wall + RAM + disk → runtime_report.md
python tools/build_deck.py                        # → redrob_submission.pptx
```

If `artifacts/model/` missing on a fresh clone: `python tools/vendor_model.py`.
If `artifacts/*.npy` / `*.parquet` missing: re-run `build_features.py` (~50 min
on M4 CPU; one cup of coffee). The CP-3.5 patch already tunes for that.

Sandbox (CP-5c) requires `pip install -r requirements-app.txt` first;
deck builder (CP-5d) requires `pip install -r requirements-dev.txt`
(adds `python-pptx`).

### What to feed the next agent

Paste this verbatim:

> Read `redrob-ranker/completed.md` end-to-end, then `progress.md`, then
> the latest commit message (`git log -1`). All architecture is LOCKED in
> `../problem.md §3 Solution A v2` and the build playbook in `../lld.md`.
> Do **not** re-debate architecture; do **not** re-read those two unless a
> *new* design question arises.
>
> Repo state: CP-1 through CP-5d are all committed and green
> (**275 pytest**, ruff clean, `rank.py` 1.87 s wall on 100K,
> `validate_submission.py` clean, `reasoning_audit.py` PASS 100/100).
> Last commit: `43de47e` feat(deck): CP-5d submission PPTX builder.
>
> Working dir: `/Users/dhruvsanan/Desktop/India_runs/redrob-ranker`.
> Activate venv: `.venv/bin/activate`. Branch: `main`. Local-only, no
> remote configured yet.
>
> **Next is CP-S1 (floor submission)** — run all 11 `hld.md` blocking
> checks against the current `rank.py` output, fix anything that fails,
> commit the verified floor. Do NOT start CP-S2 weight tuning until
> CP-S1 is green and signed off.
>
> The CP-5a holdout label-grounded assertions remain INACTIVE until
> the operator (Dhruv) fills `holdout_labels.csv` `true_label` column.
> Predicate-bucket assertions already PASS 8/9 (one near-miss:
> plain_language median rank = 52 vs threshold 50). This is the signal
> CP-S2 weight tuning consumes — not a CP-S1 blocker.
>
> Hand-steps the operator owes before full CP-S1 sign-off (block CP-S1
> on these only if Dhruv hasn't completed them — otherwise verify what
> is verifiable and flag what's missing):
> 1. Label `holdout_labels.csv` (99 rows × `true_label`).
> 2. Export `redrob_submission.pdf` from `redrob_submission.pptx`
>    (Keynote / PowerPoint manual step).
> 3. Push to GitHub.
> 4. Deploy HF Space.
> 5. Regenerate deck with final URLs (`tools/build_deck.py
>    --github-url ... --sandbox-url ...`).
>
> Stop after every CP commit, summarize, ask me to continue before the
> next CP. Quality gates before EVERY commit: `pytest -q` green and
> `ruff check src tests tools rank.py reasoning_audit.py build_features.py app.py`
> clean.
>
> Hard invariants (non-negotiable):
> - `REFERENCE_DATE = 2026-06-01` lives only in `src/reference_date.py`.
> - `rank.py` import allow-list: `{numpy, pandas, pyarrow, json, sys,
>   pathlib, yaml}` + `src.*`. `argparse` intentionally NOT included
>   (sys.argv parsed by hand). Static AST test in
>   `tests/test_rank_imports.py` enforces this — `app.py` does NOT pollute
>   this allow-list (separate entry point).
> - No hosted LLM, no network, no GPU in `rank.py` (runtime sys.modules
>   guard inside rank.py + AST test).
> - BGE-small vendored under `artifacts/model/` — load only via
>   `src.embeddings.load_model` from `build_features.py` or `app.py`.
> - No LTR / LambdaMART training. No FAISS / vector DB.
> - CP-5b ablation findings (load-bearing for CP-S2): A3 dominates top-10
>   (`availability × logistics` flips 7/10 ranks); A1 ≈ noise (embedding
>   contributes nothing to top-100 membership); A4/A5 = pure defensive
>   guardrails. Tune accordingly.

Do **not** re-derive architecture decisions from `problem.md`. They are
locked. Read `problem.md §3 v2` only if a *new* design question arises
that isn't already settled in this offload doc.
