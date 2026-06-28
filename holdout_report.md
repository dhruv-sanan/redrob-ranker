# Holdout report — CP-5a

Source seed: `holdout_labels.csv` (99 rows)
Submission: `top_100_submission.csv` (100 rows)
Overall: **FAIL**

## Bucket summary

| bucket | n_seed | in_top_100 | in_top_50 | median_rank_in_top_100 |
|---|---:|---:|---:|---:|
| plain_language | 11 | 7 | 3 | 52.0 |
| stuffer | 11 | 0 | 0 | — |
| honeypot_drop | 11 | 0 | 0 | — |
| honeypot_audit | 11 | 0 | 0 | — |
| irrelevant_tail | 11 | 0 | 0 | — |
| services_only | 11 | 0 | 0 | — |
| services_to_product | 11 | 0 | 0 | — |
| outside_india | 11 | 0 | 0 | — |
| non_tech_title | 11 | 0 | 0 | — |

## Label summary

_No `true_label` values present — run after hand-labeling fills the seed CSV._

## Predicate-bucket assertions

| status | assertion | observed | threshold |
|---|---|---|---|
| FAIL | plain_language: median rank ≤ 50 in top-100 AND ≥40% reach top-100 | `median=52.0, frac_top_100=63.64%` | `median≤50 AND frac≥40%` |
| PASS | stuffer: ≤ 20% reach top-100 | `frac_top_100=0.00%` | `≤20%` |
| PASS | honeypot_drop: 0 in top-100 (blocking check #5) | `in_top_100=0` | `==0` |
| PASS | honeypot_audit: ≤ 30% reach top-100 | `frac_top_100=0.00%` | `≤30%` |
| PASS | irrelevant_tail: 0 in top-100 | `in_top_100=0` | `==0` |
| PASS | services_only: ≤ 50% reach top-50 (rank_50 ceiling pressure) | `frac_top_50=0.00%` | `≤50%` |
| PASS | services_to_product: ≥ services_only top-100 fraction | `sp_top_100=0.00%, so_top_100=0.00%` | `sp≥so` |
| PASS | non_tech_title: ≤ 50% reach top-50 (rank_50 ceiling + blocking check #6) | `frac_top_50=0.00%` | `≤50%` |
| PASS | outside_india: informational | `frac_top_100=0.00% (no auto-pass/fail)` | `informational` |

## Label-grounded assertions

_No labels filled in seed CSV — skipped._
