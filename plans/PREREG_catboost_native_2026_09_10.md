# Pre-registration — arms C_delta and C_sched: native-categorical CatBoost, two targets (SCREEN tier)

**Written 2026-09-10 13:29 local, BEFORE the runner exists and before any number of it exists.**
Tags `C_delta`, `C_sched`. Follows `plans/FRESH_PATH_TO_246_2026_09_10.md` §2 (owner's plan) and the
owner's choice of 2026-09-10: pairwise categorical combinations, higher learning rate.
A timing-only probe (142,611 rows, accuracy not recorded) measured 0.67 ms / iteration / 1k rows at
`max_ctr_complexity=2`.

## Hypotheses

- **H-C1:** native categorical handling of full identity and airport-scoped keys (with learned pairwise
  combinations) beats arm F's representation on fold A's matched holdout.
- **H-C2:** with identical inputs and settings, the schedule-relative target `G = BLOCK − SCHED = sp − y`
  beats the NM-relative target `delta` — or, as the plan allows, adds complementary information.
  **Prior, written now:** for matched rows `G = delta + (AOBT_3 − SCHED)`, so a tree must re-learn an
  identity in a wide-range feature — the reason `delta` beat `y` (285 → 252). Expected: `C_sched` worse
  than `C_delta` on ordinary rows, better on schedule-fill rows (where `G ≈ 0`).

## Design (fixed now)

Fold A exactly as arm F (`lgbm_fold.fold_masks`: holdout months 1, 7; early stopping on months 3, 9 carved
out of training). Matched rows of the stand cache, joined positionally with the queue and order blocks.
**Numeric:** arm F's 83-column design MINUS the 24 in-fold target/delta encodings (so BC-1 does not
apply: no target-reading statistic is fitted outside CatBoost). **Categorical (13, strings, null →
"NA"):** `ap`, `ap|STAND`, `ap|RUNWAY`, `ap|ADES`, `FLIGHT_mvt`, `CALLSIGN_flt` (both joined from raw
by MVT_ID through the order cache), `airline`, `AIRCRAFT_OPERATOR_flt`, `AIRCRAFT_TYPE_mvt`,
`MARKET_SEGMENT_flt`, `WK_TBL_CAT_flt`, `FLIGHT_TYPE_flt`, `stand_pref`.
**CatBoost:** RMSE, depth 8, learning_rate 0.15, max_ctr_complexity 2, iterations ≤ 12,000, od_type Iter,
od_wait 200 on the early-stopping rows, seed 0, thread_count 6. Refit on ALL training rows with
`n_ref = round(best_iter × n_train / n_fit)` (arm F's rule). One seed: **screening**.
**Recovery:** C_delta `max(proxy − pred, 1)`; C_sched `max(sp − pred, 1)`.

## Screening rule, locked (not a confirmation)

For each of `C_delta`, `C_sched`, `0.5·C_delta + 0.5·F`, `0.5·C_sched + 0.5·F` (taxi-time scale; the
blend weight fixed now, never fitted on the holdout), against F on the same 339,015 rows:
- **SHORTLIST** iff the paired row-bootstrap interval (2,000, seed 0) excludes zero in its favour AND
  its net gain ≥ **1,000 weighted fold MSE** (0.98466 × ΔSSE / 339,015).
- otherwise **NOT SHORTLISTED** (not a refutation — one seed, one fold).
Reported: fill / non-fill / |delta| band / per-airport cuts, C_delta vs C_sched head-to-head, best_iter,
walls, peak memory. A shortlisted arm goes to confirmation (three seeds, inner-OOF blending, date/airport
resampling) under its own registration.

## Limits

One seed, one fold; January/July are inspected development months. Memory at full volume is measured on
a four-month probe before the full run; if it exceeds ~12 GB the run is not launched.

## Quality gate · 2026-09-10T15:42:43 — YELLOW, before C_delta's full run
Suite 412 passed / 4 deliberate (15:42). Real-data probe 14:50: 1,015,306-row input (6 months), 300 iterations capped,
peak 3.67 GB, exit 0 — not a result. Full input built 14:53: 2,062,440 rows (339,015 holdout = arm F's fold). A
name-collision bug (`proxy`/`sp` scoring copies overwrote the float32 features) was caught by the worker's
contract before any fit and fixed (`assemble_output`, test + mutation RED). Memory extrapolated 8–10 GB < 12 GB
bar. AC power. Waivers (b) mutation runner, (c) coverage, (f) metrics: standing.

## RESULT C_delta (screen) · 2026-09-10 18:49 — **NOT SHORTLISTED** (both arms)

Source: `reports/catboost_native.json` (`catboost_native.py score`, exit 0); fit `reports/catboost_native_delta.console.log`
(launched 16:52 from prc-challenge-25 after its own gate; best_iter 5,670 of ≤ 12,000; refit 7,090 trees on 1,723,425
rows; wall 6,953 s; peak RSS 6.4 GB). 339,015 holdout rows, arm F 222.5632.

| arm | RMSE | gain | 95% CI | net weighted fold MSE | shortlist rule |
|---|---|---|---|---|---|
| `C_delta` | 225.784 | −3.22 s | [−4.02, −2.42] | **−1,422** | not shortlisted (interval against) |
| `0.5·C_delta + 0.5·F` | 221.345 | **+1.22 s** | [+0.83, +1.61] | **+533** | not shortlisted (net < 1,000) |

H-C1 as stated (native categoricals beat F's representation) is **NOT WORKING at screen level**: C_delta loses at 9 of
10 airports (LIRF +0.3 s) and in every |delta| band. Reported, not decisional: the fixed 0.5 blend's gain is carried by
the clock-disagreement tail (|delta| > 10 min +5.24 s; > 20 min +11.32 s) and LIRF (+4.88 s), with EHAM −0.72 and LSZH
−0.49 — the two learners' errors differ where F is weakest. A screen is one seed, one fold: not a refutation of
blending. The registered next step for a blend is weights fitted on inner OOF (plan Stage 3d), which needs C_delta's
inner-OOF predictions (another heavy run) — a compute-allocation decision, not a verdict. C_sched not run.
