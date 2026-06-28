# Runtime report — CP-5b

## Wall-clock

| stage | wall | notes |
|---|---:|---|
| build_features.py (offline) | 3050 s (50.8 min) | encoder-dominated; CPU M4 |
| rank.py (online) | 1.87 s | this run; hard cap 300 s |

## Peak memory

- rank.py subprocess peak RSS: **1774.8 MB**
- (build_features.py peak ≈ 2 GB, dominated by BGE model in RAM + batched encoding)

## Artifact disk

| artifact | size (MB) |
|---|---:|
| manifest.json | 0.00 |
| candidates.parquet | 14.43 |
| features.parquet | 3.77 |
| candidate_emb.npy | 73.24 |
| jd_intent_vecs.npy | 0.01 |
| build_features_summary.json | 0.00 |
| model/ (recursive) | 128.27 |
| **total (no model)** | **91.45** |
| **total (with model)** | **219.72** |

## Tier histogram (100K candidates)

| tier | n |
|---|---:|
| A | 82 |
| B | 384 |
| C | 10188 |
| D | 87275 |
| E | 2071 |

## Honeypot counts

- honeypot_drop: **348**
- honeypot_audit: **15143**

## rank.py telemetry (current run)

```json
{
  "artifacts": "artifacts",
  "out": "artifacts/_runtime_tmp/top_100_submission.csv",
  "audit": "artifacts/_runtime_tmp/top_100_audit.csv",
  "debug": "artifacts/_runtime_tmp/top_300_debug.csv",
  "rows_scored": 100000,
  "rows_after_honeypot_drop": 99652,
  "top_10_gate_pool_size": 275,
  "top_10_relaxation_used": 2,
  "ceilings": {
    "score_at_rank_51": 0.6229228377342224,
    "score_at_rank_101": 0.5612537860870361,
    "rank_50_clipped_count": 41093,
    "rank_100_clipped_count": 0
  },
  "submission": "artifacts/_runtime_tmp/top_100_submission.csv"
}
```
