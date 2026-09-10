# Pre-registration — arm R: a stake-weighted fill classifier for Rome's unmatched rows

**Written 2026-09-10 09:45 local, BEFORE the harness exists and before any of its numbers exists.**
Standalone file (the concurrent session owns `plans/PREREG_taxiout_2026_09_08.md`). Tag: **`R`**.

**Why this arm.** `reports/GAP_LOCATED_2026_09_10.md`: on July 2025 the whole gap to a public
271-board system is LIRF's unmatched rows (S1 RMSE 3,925.6 vs their 3,437.9, 1,210 M SSE). Measured
09:40 on `stratum_fold_v7_preds.parquet`, Rome unmatched SSE:

| oracle | fold A | 12-month LOMO |
|---|---|---|
| perfect fill call, same nf | **−80.5%** | **−61.1%** |
| perfect nf, same p | −12.3% | −33.6% |

and `p_hat`'s AUC is 0.857/0.876 unweighted but **0.760/0.733 weighted by sp²** — it is weakest
exactly where a wrong call is expensive. The classifier is the lever; the normal expert is not.

**Owner's target:** board 288.14 → ~274 (top 10) needs −7,950 MSE. This arm is the largest
candidate; arm F (−746, established) ships alongside.

---

## 1. Arms (Rome unmatched rows only; every other row untouched)

Every arm is `max(p·sp + (1−p)·nf, 1)` with `nf` = S1's stored non-fill term (`nf_fit` where
`routed_nf_fit`, else `nf_cells`) — verified to rebuild S1 exactly (max abs error 0.0). **Only `p`
changes**, so any difference is the classifier's.

| arm | p |
|---|---|
| `S1` | the shipped `p_hat` (L-e logistic) — the control |
| `R` | LightGBM binary, **mean of seeds 0,1,2**, trained on ALL LIRF departures of the training months (matched + unmatched), label `|BLOCK − SCHED| ≤ 60 s` (the shipped fill definition) |
| `R_seed{0,1,2}` | single seeds (the seed sd) |
| `R_unm` | diagnostic: as `R` but trained on unmatched LIRF rows only |
| `R_perm` | negative control: as `R` (seed 0) with the training labels permuted within month |

**Weights, fixed now:** `w = clip(1 + (max(sp, 0)/3600)², 1, 25) × (8 if unmatched else 1)` — the
stake weight read from the public Rome recipe, and the unmatched up-weight from the same source. Not
tuned.

**Features, fixed now** (all populated on scored rows; none reads BLOCK): `sp`, `log1p(max(sp,0))`,
`dayoff`, `hr`, `tmin`, `dow`, `sched_min` (minute of SCHED), `unmatched`; native categoricals
`airline`, `STAND_mvt`, `stand_c1`, `ADES_mvt`, `ades_p2`, `RUNWAY_mvt`, `FLIGHT_RULE_mvt`.

**Params, fixed now:** `objective=binary, num_leaves=15, learning_rate=0.05, n_estimators=400,
min_data_in_leaf=100, lambda_l2=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
cat_smooth=50, min_data_per_group=50, max_cat_to_onehot=4, num_threads=4`. No early stopping — there
is no tuning set, by design.

## 2. Folds

- **LOMO-12**: for each month m of 2025, train on the other eleven, predict month m. The stored
  `nf` for fold `lomo` rows was fitted the same way.
- **Fold A** (confirmation): train on the ten months excluding 1 and 7, predict 1 and 7; stored `nf`
  for fold `A`.

Nothing is tuned on either; the design above is frozen before the first fit.

## 3. Thresholds, locked

On Rome unmatched rows (LOMO-12: ~1,488 rows; fold A: 397):

**`R` ESTABLISHED iff all five:**
1. the paired **month-block** bootstrap (12 months resampled with replacement, 2,000 draws, seed 0)
   interval on LOMO-12 Rome SSE(S1) − SSE(R) excludes zero in R's favour;
2. LOMO-12 Rome SSE reduction **≥ 15%**;
3. fold-A Rome SSE reduction **≥ 15%** (the confirmation);
4. the LOMO-12 gain in Rome RMSE > **2 × seed sd** (sd over `R_seed0..2`);
5. **≥ 8 of 12** months have lower Rome SSE under `R`.

**NOT WORKING** iff (1) fails. Anything between is INCONCLUSIVE, reported with every number.

**Controls that must pass before any R number is read:** S1 reproduces its stored column exactly
on every row; `R_perm` does NOT beat S1 (its LOMO-12 SSE must be ≥ S1's) — if the permuted
classifier "wins", the harness is measuring something other than information.

**Reported, not decisional:** fold-A July Rome RMSE against the public anchor **3,437.9**; per-sp-band
SSE; stake-weighted AUC; the projected board MSE = fold-A Rome SSE reduction × (383 / 397) / 344,841
(383 = Rome unmatched rows in the 2026 scored file).

## 4. Named limitations

- Rome's unmatched rows are ~1,900 a year; month-block intervals will be wide, which is why the bar
  also requires fold-A confirmation and a month count.
- The stake weight and unmatched up-weight come from a public recipe, not from our data; they are
  fixed rather than tuned precisely so they cannot be fitted to our folds.
- The matched-lane fill (the same phenomenon on LIRF's matched rows) is untouched here.

## 5. What a pass licenses

A twelve-month refit of `R` and a board version with arm F (owner confirms the upload). Nothing else.

---

## Quality gate, 2026-09-10 09:56 (the run started 09:57:32; "~10:05" in the first version of this line was a guess, corrected from the log) — YELLOW, waivers on record

(a) full suite 352 passed / 4 deliberate failures (`test_stationarity` x2, `test_serve_time_contract`
x2, standing instruction: never "fixed"); (d) 0 skipped; (e) 0 xfail; (g) fixture sized like the real
stratum (~150 LIRF unmatched rows/month); (h)+(i) real-data smoke ~09:55: 12 months loaded, one LOMO
fold + fold A, S1 rebuilds exactly (max abs error 0.0), peak 2.56 GB; (k) no clamps (the floor is
the shipped `build_submission.mixture`, imported); (l) no feedback-loop values.
**Waived, standing project gaps (no tool exists):** (b) mutation runner — 8 manual mutations
rehearsed RED instead, one survivor (floor deletion) answered structurally by importing the shipped
mixture; (c) coverage tool; (f) metrics tracking. Proceeding under the owner's direction of
2026-09-10 ("first build and get to 274"); any failure traced to a waived item is a post-mortem trigger.

---

# RESULT R · 2026-09-10 09:58 local · NOT WORKING as registered — and what it located

`scripts/rome_fill.py`, wall 48 s, peak 0.57 GB. Records `reports/rome_fill.{json,log,console.log}`,
`data/cache_stand/rome_fill_preds.parquet`. Controls: S1 rebuilds exactly (0.0); R_perm far worse
than S1 (LOMO SSE 87,921 M vs 29,476 M).

| | LOMO-12 (1,488 rows) | fold A (397 rows) |
|---|---|---|
| S1 Rome RMSE / SSE | 4,450.7 / 29,476 M | 3,935.3 / 6,148 M |
| **R** Rome RMSE / SSE | **4,200.4 / 26,254 M** | **3,360.6 / 4,484 M** |
| reduction | 10.9% | **27.1%** |
| sp²-weighted AUC S1 → R | 0.733 → 0.855 | 0.760 → 0.877 |
| R_unm (unmatched-only training) | 28,477 M | 6,308 M |

| clause | measured | |
|---|---|---|
| (1) month-block interval excludes zero | +3,222 M **[−1,511, +7,396]** | **fail** |
| (2) LOMO reduction ≥ 15% | 10.9% | **fail** |
| (3) fold-A reduction ≥ 15% | 27.1% | pass |
| (4) gain > 2 × seed sd | +250.3 vs 94.7 | pass |
| (5) ≥ 8 of 12 months improve | 10 / 12 | pass |

**VERDICT: NOT WORKING** (clause 1 fails; the registered rule makes that NOT WORKING, not
INCONCLUSIVE). Reported: fold-A July Rome RMSE 3,925.6 → 3,538.8 against the public 3,437.9;
projected board ≈ 4,657 MSE — **not licensed**, because the arm did not clear its bar.

**Why the month-block interval spans zero** — two months: September +1,432 M (two high-`sp` ITY
long-haul FILLS where R lowered p from 0.34/0.93 to 0.03/0.32) and October +221 M. Matched rows ARE
the information: training on unmatched rows alone (R_unm) loses to S1.

**What it located — a third record class (H-DS), not a rescue of R.** 12 LIRF unmatched rows in 2025
have `y ∈ [86,400, 90,000)` with `y − 86,400` = an ordinary taxi (p10/50/90 771 / 1,020 / 1,679 s):
the airport kept the ACTUAL off-block clock time under the SCHEDULE's date (checked: sched 21:40 D,
take-off 13:22 D+1, y 87,480 → recorded BLOCK 13:04 on D). 11 of the 12 have take-off on the day
after the schedule. Above `sp` 56,000 s with `dayoff = 1`: 10 date-slips vs 5 fills. These 12 rows
carry **33.6% of LOMO Rome SSE** — but one of them (sp 11,694, indistinguishable from an ordinary
late flight) is ~25% on its own. No mixture of `sp` and `nf` can represent 87,000 except through the
cells, which S1's `[ap, sp_fine, dayoff]` hierarchy already partly does (nf ≈ 73–84 k there).
A three-class design (fill → sp, date-slip → 86,400 + taxi, ordinary → nf) is a NEW hypothesis and
needs its own pre-registration before any number of it is measured.
