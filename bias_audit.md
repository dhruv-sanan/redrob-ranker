# Bias Audit — Top-100 V8 Floor

Scope: descriptive distribution comparison of the top-100 cohort vs the
full 100K candidate baseline across observable profile dimensions. Per
`final.md §4.2`.

Generated: 2026-06-29 against the V8 floor
(`top_100_audit.csv` × `artifacts/candidates.parquet`, HEAD = `91189bf`).

> **V8b refinement (2026-07-02).** The premier-institution finding below
> flagged 9.09× over-representation. In response, `education_signal` was
> halved (0.05 → 0.025) with the freed 0.025 redistributed to
> `must_have_sum_div_6` (0.14 → 0.165). Effect on the top-100:
> premier count **33 → 30**, all 13/13 holdout assertions preserved,
> top-10 + top-50 unchanged, tail (rank 51-100) swaps 3 rows. The 9.09×
> ratio compresses to **~8.26×** vs the 3.63% baseline. Full numbers
> below reflect the original V8 snapshot; the V8b delta is only on the
> premier-institution row of §C.

---

## Method

For each dimension D, compare `top_100_pct(D=v) / base_pct(D=v)` for
every category `v`. Ratios > 1.0 indicate over-representation in the
ranked output; < 1.0 indicate under-representation. Categories with
base_pct = 0 are dropped.

Each finding is then tagged as **JD-traceable** (defensible per the
hand-tuned feature blend and the locked architecture) or
**non-JD-traceable** (potential bias signal worth flagging).

`anonymized_name` is included for first-character distribution only.
Gender is **not inferred** — the dataset has no gender field, and any
name-based inference would itself import the bias being measured.
Documented as a known scope gap.

---

## A. Country  (JD-traceable)

| country    | top_100 % | base % | ratio |
|------------|----------:|-------:|------:|
| India      | 85.0      | 75.11  | 1.13  |
| USA        | 4.0       | 9.98   | 0.40  |
| Canada     | 3.0       | 2.51   | 1.20  |
| Australia  | 3.0       | 2.58   | 1.16  |
| Germany    | 2.0       | 2.47   | 0.81  |
| Singapore  | 1.0       | 2.45   | 0.41  |
| UAE        | 1.0       | 2.43   | 0.41  |
| UK         | 1.0       | 2.47   | 0.40  |

**Driver:** `logistics_multiplier × availability_signal` (CP-2
behavioral). Outside-India candidates without `willing_to_relocate`
take a logistics ceiling per `src/features/behavioral.py`. Redrob is an
India-domiciled startup; "willing to work India hours / relocate" is an
implicit JD constraint. **Defensible.**

## B. State / non-India country  (mixed)

| state            | top_100 % | base % | ratio |
|------------------|----------:|-------:|------:|
| Tamil Nadu       | 15.0      | 8.28   | 1.81  |
| Maharashtra      | 11.0      | 8.23   | 1.34  |
| Andhra Pradesh   | 10.0      | 4.09   | 2.44  |
| West Bengal      | 9.0       | 4.23   | 2.13  |
| Kerala           | 8.0       | 8.22   | 0.97  |
| Uttar Pradesh    | 8.0       | 4.28   | 1.87  |
| Karnataka        | 4.0       | 4.24   | 0.94  |
| Delhi            | 2.0       | 4.16   | 0.48  |
| Telangana        | 1.0       | 4.28   | 0.23  |
| Chandigarh       | 0.0       | 4.13   | 0.00  |

**Concern.** No JD signal differentiates by Indian state, yet ratios
swing 0.0×–2.44× across them. Karnataka and Telangana — home to most
AI/ML product companies in India — are *under-represented* (0.94× and
0.23×), which rules out a "tech-hub" hypothesis. Distribution likely
reflects synthetic-dataset correlations between `current_industry`
labels and `location` strings rather than a deliberate feature.

**Risk:** low to moderate. State is not used as a feature; the swing is
an emergent artifact of which industries the synthetic generator placed
where. **No fix planned** (would require synthetic-data review out of
scope). Flag for Stage-5 in case reviewer asks.

## C. Institution tier (first degree)  (mixed)

| tier              | top_100 % | base % | ratio |
|-------------------|----------:|-------:|------:|
| other-recognised  | 48.0      | 73.40  | 0.65  |
| premier           | 33.0      | 3.63   | 9.09  |
| other             | 19.0      | 22.96  | 0.83  |

Premier = `IIT | NIT | IIIT | BITS | IISc` (Indian-tech-canonical).

**Driver:** `education_signal` (blend weight 0.05) materially rewards
premier-tier alumni per `src/features/education.py`. JD does **not**
mention degree requirements explicitly, but the design contract per
`hld.md` includes an education term to disambiguate same-evidence
candidates.

**Risk:** moderate. 9.09× over-representation is the strongest single
deviation in the audit and is *not* directly grounded in the JD text.
Defensible position for Stage-5: education_signal is the smallest
blend weight (0.05) and ablation A1/A4/A5 confirmed it is a tiebreaker
under noise rather than a primary lever. **Considered intentional;
no fix.** Future re-tune could zero it out and re-measure if reviewer
pushes back.

## D. Top named institutions (count ≥ 2)

| institution                       | count |
|-----------------------------------|------:|
| Thapar University                 | 7     |
| RV College of Engineering         | 6     |
| SRM University                    | 5     |
| IIT Hyderabad                     | 5     |
| IISc Bangalore                    | 5     |
| Manipal Institute of Technology   | 4     |
| COEP Pune                         | 4     |
| IIT Delhi                         | 4     |
| SRM Chennai                       | 4     |
| Stanford University               | 3     |
| Carnegie Mellon University        | 3     |
| Georgia Tech                      | 3     |
| Anna University                   | 3     |
| PES University                    | 3     |
| IIIT Bangalore                    | 3     |
| Symbiosis International           | 3     |
| IIT Roorkee                       | 3     |
| BITS Pilani                       | 3     |
| IIT Kharagpur                     | 2     |
| Jadavpur University               | 2     |
| IIT Bombay                        | 2     |
| VIT Vellore                       | 2     |

No single institution dominates (max 7%). Distribution is broadly
spread across IIT-family + private engineering colleges. International
universities (Stanford / CMU / Georgia Tech) appear at 3 each — these
correspond to the diaspora candidates in `country ∈ {USA, Canada,
Australia}` who passed logistics gates.

## E. Current industry (top 15)  (JD-traceable)

| industry           | top_100 % | base %  | ratio    |
|--------------------|----------:|--------:|---------:|
| AI/ML              | 14.0      | 0.28    | 50.36    |
| E-commerce         | 12.0      | 1.53    | 7.85     |
| Fintech            | 11.0      | 2.81    | 3.92     |
| Software           | 10.0      | 22.42   | 0.45     |
| EdTech             | 9.0       | 0.61    | 14.75    |
| Internet           | 7.0       | 0.02    | 318.18   |
| Media              | 5.0       | 0.01    | 833.33   |
| IT Services        | 4.0       | 29.88   | 0.13     |
| Conversational AI  | 4.0       | 0.06    | 64.52    |
| Food Delivery      | 4.0       | 2.51    | 1.59     |
| Gaming             | 3.0       | 0.15    | 20.13    |
| SaaS               | 3.0       | 0.33    | 9.15     |
| AI Services        | 3.0       | 0.04    | 71.43    |
| Insurance Tech     | 2.0       | 0.16    | 12.90    |
| HealthTech AI      | 2.0       | 0.07    | 29.41    |

**Driver:** the explicit `has_product_company_applied_ml_context`
must-have (weight 0.14 in the blend via `must_have_sum_div_6`) plus the
`services_only` anti-pattern ceiling. IT Services collapses to 0.13×
and Software to 0.45× exactly as the JD's "product-company applied ML"
clause directs. AI-adjacent industries are intentionally amplified.
**Defensible.**

## F. Company size  (mixed)

| size        | top_100 % | base %  | ratio |
|-------------|----------:|--------:|------:|
| 10001+      | 39.0      | 40.46   | 0.96  |
| 1001-5000   | 23.0      | 18.20   | 1.26  |
| 5001-10000  | 15.0      | 3.42    | 4.39  |
| 51-200      | 12.0      | 7.73    | 1.55  |
| 201-500     | 8.0       | 15.10   | 0.53  |
| 11-50       | 2.0       | 7.57    | 0.26  |
| 501-1000    | 1.0       | 7.52    | 0.13  |

5001-10000 over-represented at 4.39×; small companies (≤500)
under-represented. Likely a correlated artifact of the product-company
industry bias — mid-large product-cos cluster in the 1001-10000 size
range. **Not directly JD-driven** but emergent from the industry
preference, hence not separately actionable.

## G. Years of experience band  (JD-traceable)

| band   | top_100 % | base %  | ratio |
|--------|----------:|--------:|------:|
| 5-8    | 71.0      | 25.90   | 2.74  |
| 2-5    | 17.0      | 26.40   | 0.64  |
| 8-12   | 9.0       | 25.36   | 0.35  |
| 12+    | 3.0       | 14.87   | 0.20  |
| 0-2    | 0.0       | 7.47    | 0.00  |

**Driver:** `experience_band_fit` (blend weight 0.06) peaks in the 4–8
yoe band per the JD's "senior AI/ML engineer" framing. JD explicitly
asks for *senior* engineers with production ML ownership — not juniors
(0–2 dropped to 0%) and not founder/architect tiers (12+ collapsed).
**Defensible.**

## H. Name first-character (top 10)

| char | top_100 % | base %  | ratio |
|------|----------:|--------:|------:|
| A    | 27.0      | 27.49   | 0.98  |
| S    | 13.0      | 13.04   | 1.00  |
| K    | 11.0      | 7.25    | 1.52  |
| D    | 9.0       | 7.24    | 1.24  |
| M    | 7.0       | 5.86    | 1.20  |
| R    | 7.0       | 8.77    | 0.80  |
| P    | 7.0       | 5.68    | 1.23  |
| N    | 6.0       | 5.87    | 1.02  |
| I    | 4.0       | 2.92    | 1.37  |
| V    | 4.0       | 5.66    | 0.71  |

Maximum 1.52× swing on a 100-row cohort is within sampling noise
(SE on each cell ≈ √(p·(1-p)/100), ~1–5 pp). No signal.

## I. Gender — NOT MEASURED  (scope gap)

The synthetic dataset records `anonymized_name` only; no `gender` field
exists in the `profile` dict (probed all 10 keys:
`anonymized_name, country, current_company, current_company_size,
current_industry, current_title, headline, location, summary,
years_of_experience`). Any name-based gender inference would import
its own bias (rare names, unisex names, transliteration artifacts) and
encode it as audit signal.

**Scope:** declined. The honest answer is "the dataset does not
support a gender audit." Stage-5 defense: state this explicitly if
asked, and offer to run a gender-inference audit on a labeled subset
if Redrob's HR provides one.

---

## Summary

| dimension          | over-rep direction   | JD-traceable | risk     |
|--------------------|----------------------|:------------:|:--------:|
| country            | India 1.13×          | yes          | low      |
| state (India)      | TN/AP/WB/UP up; KA/TS/DL/CH down | **no**       | moderate |
| institution tier   | premier 9.09×        | partial      | moderate |
| current industry   | AI-product up; IT Services down | yes  | low      |
| company size       | 5001-10000 4.39×     | indirect     | low      |
| YOE band           | 5-8 yr 2.74×         | yes          | low      |
| name first-char    | within noise         | n/a          | n/a      |
| gender             | (not measured)       | n/a          | scope-gap |

**Two findings worth flagging for Stage-5:**

1. **State-of-India skew.** Karnataka and Telangana — the densest
   real-world AI/ML hubs — are under-represented in the top-100,
   while AP/WB/UP/TN are over. Likely a synthetic-dataset correlation
   between industry labels and location strings, not a feature choice.
   No code or config to change.

2. **Premier-institution amplification (9.09×).** Driven by the
   `education_signal` blend term (weight 0.05). Defensible as the
   smallest blend coefficient and an explicit tiebreaker, but the
   ratio is the largest single deviation in the audit. Would zero out
   cleanly if a reviewer pushes back — `config/weights.yaml`
   `education_signal: 0.00` and renormalize blend.

All other dimensions either reflect direct JD encoding (country, YOE,
industry) or are within sampling noise (name first-char). No fix is
applied; this file is the record per `final.md §4.2`.
