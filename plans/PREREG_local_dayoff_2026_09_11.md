# Pre-registration — arm LD: `dayoff` on the airport's LOCAL date (Rome lane), fixed at its origin

**Written 2026-09-11 10:52:17 (from `date`), session prc-challenge-c4, BEFORE the harness exists and before any number of
this arm exists.** Tag `LD`. Owner, 2026-09-11: "target 10th, then 5th, then 1st". The defect is handoff item D1 (known
since the 2026-09-10 review; `plans/PIPELINE_DESIGN_2026_09_10.md` step 3 says it enters only as a registered change after
A1 — A1 is met).

## Why (measured before writing, descriptive)

`build_submission.derive` computes `dayoff = clip(date(MVT) − date(SCHED), 0, 1)` on UTC dates. The Rome date-slip is a
LOCAL-date mechanism: on 2025, all 12 date-slips have take-off and schedule on different Rome dates and date(BLOCK) =
date(SCHED) locally (12/12). UTC mislabels rows: 115 LIRF unmatched 2025 rows are "same UTC day, next Rome day" — 84 fills,
1 slip, 30 other — and R2 under-predicts them (mean 8,556 vs 9,291). In the 2026 scored file 31 LIRF unmatched rows are of
that kind and 4 the reverse; v10 predicts several far below sp.

## Hypothesis

**H-LD:** recomputing `dayoff` on the airport's local date (Europe/Rome for LIRF; each airport's own zone elsewhere, from
`configs/pipeline.yaml`) everywhere the Rome lane reads it — S1's cells and body, the fill classifier's features, the
date-slip segment — lowers Rome's unmatched error out of month.

## Instrument (fixed now)

- The chain exactly as shipped (v10's LIRF lane), re-run end to end on the 2025 frames (`data/cache_rome/`):
  `stratum_fold.score_lomo` + `score_fold_a` (S1 parts, seeds 0–2) → `rome_fill.run` (R) → `rome_dateslip.run` (R2) →
  E1's floor `max(rint(R2), rint(sp))` on `24,000 ≤ sp < 86,400`.
- **Guard (a precondition, not a clause):** the chain on the UNCHANGED (UTC) frames reproduces the stored
  `data/cache_stand/rome_dateslip_preds.parquet` R2 on every row of both folds (max |Δ| ≤ 1e-6 s), or nothing is scored.
- **Arm LD:** the same chain on frames whose `dayoff` is the local-date version (only that column changes).
- `g = SE(R2E1_UTC) − SE(R2E1_LD)` on LIRF unmatched rows.

## Clauses, locked

1. **C1** — month-block bootstrap (12 LOMO months, 2,000 draws, seed 0) of Σg on the LOMO fold: 95% lower bound > 0.
2. **C2** — 2026-priced gain ≥ **+300** board MSE: per (UTC dayoff, local dayoff) cell, the LOMO mean g per row × the 2026
   scored LIRF unmatched rows in that cell ÷ 344,841 (cells with no 2025 row contribute 0, counted).
3. **C3** — fold A agrees in sign: Σg(fold A) > 0.
**Verdict:** WORKING iff all three; NOT WORKING iff C1 fails; else INCONCLUSIVE. Never revised.
**Reported, not decisional:** by cell (the (UTC 0, local 1) cell is where the effect should live), by month; S1's change
on non-LIRF rows (the same derive feeds them; their shipped body is E3C's — a separate registration if it matters); the
largest single-row contributions (a two-row result is named as such).
**Ship rule:** WORKING → the pipeline's derive gains the local-date `dayoff` (airport tz from the config) for the LIRF
rules lane, A1-style guard (with the UTC switch the pipeline still equals v10 exactly), board Δ read against the C2 price.

## Predicted shapes (written now)

- **TRUE:** Σg > 0 carried by the (UTC 0, local 1) cell — R2 moves toward sp on those rows, most of them fills — with the
  other cells ≈ 0; C2 ≈ +500–1,500.
- **FALSE:** Σg ≈ 0 or negative: the fill classifier already reads the local-day signal through other inputs (hour, sched
  minute), or the 30 "other" rows in that cell pay for the fills.

## Limits (named now)

Sibling defect, NOT in this arm: `hr`, `tmin`, `dow`, `doy` are also UTC (summer time shifts local clock by an hour between
January and July) — to be priced separately. The 2026 price assumes 2025's per-cell gain; 2026's Rome transfer has been
poor before (R2 0.15×). Light: CPU minutes, < 2 GB.

## RESULT LD · 2026-09-11 11:01:23 (from `date`) — **NOT WORKING** (C1, C2, C3 all fail)

Source `reports/rome_localday.json` (`scripts/rome_localday.py`, exit 0, 391 s, 0.70 GB; tests `tests/test_rome_localday.py` 6,
5 mutants killed). **Guard passed:** the UTC chain (stratum S1 → rome_fill R → rome_dateslip R2) reproduced the stored R2 on
all 1,885 rows, max |Δ| 0.0.

| clause | value | bar | pass |
|---|---|---|---|
| C1 month-block Σg (12 months) | −1.048e9, 95% [−2.628e9, +1.784e8] | lower > 0 | ✗ |
| C2 2026 price | **−777** board MSE (utc0_local1 cell −833) | ≥ +300 | ✗ |
| C3 fold A sign | Σg −3.14e8 | > 0 | ✗ |

LOMO RMSE (R2 + E1): UTC 3,916.6 → local 4,005.5. By cell (LOMO Σg): utc0_local0 −4.8e7 (1,305 rows), **utc0_local1 −1.065e9**
(115), utc1_local0 −4.3e7 (15), utc1_local1 +1.08e8 (53). July alone −3.7e8, September −6.7e8 (one row).

**Against the pre-written shapes: the FALSE shape appeared as written** — "the 30 'other' rows in that cell pay for the fills".
The local flag raised the fill probability for ordinary late-evening departures that leave just after local midnight
(y ≈ 780–1,020 s, sp ≈ 10,900–13,000 s: predictions moved from 2,400–6,600 s to 13,400–14,500 s), and the largest single loss
is a true date-slip (sp 50,581, y 87,543) predicted 62,505 → 53,033. The date-slip MECHANISM is local-date (12/12); as a model
input the UTC flag separates the ordinary late flights better. **Handoff defect D1 is closed as NOT WORKING for this lane;
do not reopen with the same estimator.** The sibling (UTC hour / minute / weekday / day-of-year) was not measured.
