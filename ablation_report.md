# Ablation report — CP-5b

Source artifacts: `artifacts`
Baseline: **A0** (baseline)

## Overlap vs baseline

| variant | label | top-100 overlap | top-100 % | top-10 stability | top-50 stability | jaccard | wall (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | baseline | — | — | — | — | — | 0.21 |
| A1 | no_embedding | 99/100 | 99% | 10/10 | 50/50 | 0.980 | 0.22 |
| A2 | no_skill_blend | 95/100 | 95% | 9/10 | 49/50 | 0.905 | 0.23 |
| A3 | no_behavioral_mult | 90/100 | 90% | 3/10 | 36/50 | 0.818 | 0.20 |
| A4 | no_anti_pattern | 100/100 | 100% | 10/10 | 50/50 | 1.000 | 0.20 |
| A5 | no_top_10_gate | 100/100 | 100% | 10/10 | 50/50 | 1.000 | 0.17 |

## Variant definitions

| variant | description |
|---|---|
| A0 (baseline) | current weights + ceilings + multipliers + gate |
| A1 (no_embedding) | blend.embedding_contribution = 0.0 |
| A2 (no_skill_blend) | blend.skill_contribution = 0.0 |
| A3 (no_behavioral_mult) | skip availability × logistics × market_tiebreak |
| A4 (no_anti_pattern) | skip rank_50 / rank_100 score ceilings |
| A5 (no_top_10_gate) | no promotion gate; first 10 by global order |

## Interpretation guide

- **Top-10 stability** is the rank-aware overlap of ranks 1–10. A drop here 
  means the promotion gate / score contribution that the variant disables 
  is load-bearing for the very top of the list.
- **Top-100 overlap %** measures membership churn; rank order changes within 
  the set are NOT visible to this metric.
- **Jaccard** is symmetric and reflects how much the variant rearranges the 
  shortlist boundary.
