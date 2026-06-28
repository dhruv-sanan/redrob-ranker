# Progress Log

| CP | Date | Outcome |
|---|---|---|
| **CP-1** | 2026-06-28 | Skeleton + data + manifest layer green. `src/reference_date.py`, `src/parsing.py`, `src/io_utils.py`, `src/manifest.py` shipped. 50-candidate edge-case fixture in `tests/conftest.py`. 30/30 pytest green. ruff check + format clean. |
| **CP-2** | 2026-06-28 | All 9 feature builders shipped (`src/features/*`) + 6 config YAMLs (`config/*.yaml`) + `src/feature_pipeline.py` + `build_features.py` skeleton. 126/126 pytest green; ruff check + format clean. Smoke run on the 50-candidate fixture produces tier histogram `{A:18, B:1, C:14, D:9, E:8}` — all 3 honeypots drop, all 5 stuffers land in Tier E. |
| **CP-3** | 2026-06-28 | BGE-small-en-v1.5 vendored under `artifacts/model/` (~135 MB, gitignored). `src/embeddings.py` + `tools/vendor_model.py` shipped. Full `build_features.py` produced all 6 artifacts from 100K candidates in **64 min 5 s** wall-clock on M4 CPU (encoder dominates: 3670.8 s / 61.2 min for 100K × BGE-small at batch=512). `verify_manifest` passes against the produced manifest. Tier histogram `{A:82, B:384, C:10188, D:87275, E:2071}`; honeypot_drop=348 (audit=15143); rank_50 ceilings=41093. 138 pytest green (137 unit + new e2e). ruff check + format clean. |

> Next: Phase 4 (CP-4) — `src/scoring.py` (capped contributions + linear blend), `src/ranking.py` (tier-sort + top-10 promotion gate + relaxation), `src/reasoning.py` (evidence ledger + skeletons), `rank.py` (entry with import allow-list static check), `reasoning_audit.py`. End-to-end run through `validate_submission.py`.
