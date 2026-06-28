# redrob-ranker

Deterministic CPU-only candidate ranker for the Redrob "Intelligent Candidate Discovery & Ranking" challenge.

> Architecture: see `../problem.md §3 Solution A v2` (LOCKED).
> Build playbook: see `../lld.md §2` (5-phase plan, CP-1 → CP-S3).
> Submission gates: see `../hld.md`.

## Environment

- Python 3.11.x
- See `requirements.txt` for pinned deps. No GPU, no network at rank time.
- `REFERENCE_DATE = 2026-06-01` (frozen, single source: `src/reference_date.py`).

## Commands (placeholders — wired up over Phases 3–4)

```bash
# Phase 3 (offline, ≤10 min): parse + embed + features + manifest
python build_features.py --candidates /path/to/candidates.jsonl --out ./artifacts/

# Phase 4 (online, ≤5 min, restricted imports): rank top 100
python rank.py --artifacts ./artifacts/ --out ./top_100_submission.csv

# Post-rank: verify reasoning is grounded
python reasoning_audit.py --audit ./top_100_audit.csv --out ./reasoning_audit.csv

# Submission gate (ships from challenge bundle)
python validate_submission.py top_100_submission.csv
```

## Status

- **Phase 1 (CP-1):** in progress — data + manifest skeleton.

See `progress.md` for checkpoint log.
