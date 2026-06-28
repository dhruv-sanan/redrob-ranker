# Progress Log

| CP | Date | Outcome |
|---|---|---|
| **CP-1** | 2026-06-28 | Skeleton + data + manifest layer green. `src/reference_date.py`, `src/parsing.py`, `src/io_utils.py`, `src/manifest.py` shipped. 50-candidate edge-case fixture in `tests/conftest.py`. 30/30 pytest green. ruff check + format clean. |
| **CP-2** | 2026-06-28 | All 9 feature builders shipped (`src/features/*`) + 6 config YAMLs (`config/*.yaml`) + `src/feature_pipeline.py` + `build_features.py` skeleton. 126/126 pytest green; ruff check + format clean. Smoke run on the 50-candidate fixture produces tier histogram `{A:18, B:1, C:14, D:9, E:8}` — all 3 honeypots drop, all 5 stuffers land in Tier E. |

> Next: Phase 3 (CP-3) — embed candidate docs with BGE-small-en-v1.5 (vendored under `artifacts/model/`), embed JD intents, wire full `build_features.py` against the 100K candidates.jsonl (target ≤ 10 min wall-clock), write `manifest.json`.
