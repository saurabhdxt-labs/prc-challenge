# Feature importance, matched lane (arm F) and unmatched lane (S1C body) — 2026-09-10

A diagnostic, not a registered arm. Every number below carries its source file. No verdict
language beyond "measured".

**Sources.** Every number below comes from `reports/feature_importance.json`, which
`scratchpad/importance/analyze.py` builds from the predictions under `data/importance/`, unless
another file is named. The scratchpad is
`/private/tmp/claude-501/-Users-saurabhdxt-Projects-prc-challenge/81b461ea-ab3b-47e3-926a-3198c439b0c1/scratchpad/importance/`.
ΔRMSE is always variant − proxy in seconds: positive means the variant is worse, i.e. the group
matters.

## Summary

- **Proxy:** 225.1908 s on the 339,015 holdout rows (arm F 222.5632; the gate is 3 s, so this is
  2.63 s away and passes). This needed Deviation D1: the first recipe scored 226.666 and failed the
  gate. Per-cut RMSE tracks arm F closely: fill 285.61 vs 285.63, tail 554.92 vs 551.74,
  ordinary 174.46 vs 171.47.
- **Retraining-noise sd:** 0.207 s, from **2 seeds only** (225.19 and 225.48). Seed 2 was not run
  (D4), so 2·sd = 0.41 s is the threshold for the classes.
- **Drop-group value** (12 of 14 groups retrained, per D3), with 95% day-block intervals:
  clock **+36.6** [31.5, 42.3] · stand_id **+3.9** [3.2, 4.5] · stand_hist **+3.3** [2.8, 3.8] ·
  airline_seg **+3.3** [2.5, 4.1] · queue_counts **+2.6** [2.2, 3.0] · ordering **+1.9** [0.6, 3.7] ·
  destination **+1.1** [0.6, 1.6] · actype_wake **+0.95** [0.57, 1.32] · recent_taxi **+0.64** [0.13, 1.16] ·
  airport **+0.51** [0.11, 0.92] · season_time +0.30 [−0.43, +0.97] · flags −0.13 [−0.52, +0.28].
  traffic_freq and runway were not retrained.
- **Candidate-to-drop:** `flags` (drop −0.13 s; permutation +0.15 s) and `season_time` (drop +0.30 s,
  interval spans 0). Two constant columns have 0 splits in the proxy: `sched_sec` and
  `actype_null`. They are dead weight whatever happens to their groups.
- **Reliance ≠ value.** Permutation reliance overstates what a group is worth once other groups
  can substitute for it. `airport` is 12.9 s of reliance but +0.5 s of value; `recent_taxi` is 11.7 vs
  +0.6; `stand_id` 85.5 vs +3.9. The one reversal is `ordering`. Permuting it improves the fill
  and tail rows and hurts the ordinary rows. Dropping it and refitting does the opposite: it hurts
  the tail (+14.6 s within the cut) and helps the ordinary rows (−1.4 s).
- **Unmatched lane:** `AIRCRAFT_OPERATOR_flt` is 100.000% null on all 22,219 LOMO rows, so its
  target encoding is a constant input to the S1C body. `AIRCRAFT_TYPE_mvt` is present on 93.2% of
  rows (99.9% of non-LIRF rows) with 192 types. An out-of-month residual screen of type on S1C
  gives −0.52 s [−1.70, +0.45] on non-LIRF rows. That is not separated from 0, and not from its
  permuted-label control (−0.18 s [−0.65, +0.50]). The registered body refit and the S1C
  permutation importance were **not run** (D4).

---

## 0. Method — written before any model was fitted; not changed afterwards

Written at the start of the session, before the proxy model or any permutation ran. Anything
that had to change once runs started is logged in §0.9 with the reason. The method is not revised
in place.

### 0.1 Fold and design (arm F exactly)

- Loaded exactly as arm F:
  `lf.load_fold(lf.CACHE, None, lf.FEATS_QUEUE_ORDER, qcache=lf.QCACHE, ocache=lf.OCACHE, check_counts=True, label="holdout (months (1, 7))")`.
  83 columns, fold A: holdout months 1 and 7 (339,015 matched rows), early stopping on months 3 and
  9 (344,840 rows), fit on the other eight months (1,378,585 rows).
- Guards before anything is fitted: (a) arm F's stored record
  `data/cache_stand/fold_preds_queue_order.parquet` recovers 222.5632 as `max(proxy − treatment, 1)`
  (BC-2, `reports/bug_classes.md`); (b) the loaded fold's `base` equals that record on `row`, `y`,
  `proxy`, `sp`, `delta`, `ap`, `month`, exactly.
- The shipped submission boosters (`data/cache_stand/lgbm_v6_*`) are NOT used: they saw the holdout.

### 0.2 Proxy fold model

- One LightGBM on arm F's 83 columns, `L.P` with `learning_rate = 0.05` and nothing else changed
  (num_leaves 255, min_data_in_leaf 40, feature_fraction 0.8, bagging 0.8/1, num_threads 4),
  seed 0, trained on the `fit` rows, early-stopped on the `es` rows (patience `L.PATIENCE` = 200,
  cap `L.NEST` = 50,000), holdout predicted at `best_iteration`. No refit on `tr`: the early-stopped
  booster is the model (this is the "lightgbm directly with the fold masks" option; it halves the
  cost of every retrain and keeps the proxy and all drop-group models on one identical recipe).
- Prediction `taxi = max(proxy − delta_hat, 1)`; RMSE on the 339,015 holdout rows.
- **Representativeness gate:** holdout RMSE must be within 3 s of 222.56 (i.e. ≤ 225.56 and
  ≥ 219.56). If not: stop and report; no importance number is produced from it.
- **Retraining-noise floor:** the same full model refitted with seeds 1 and 2 (same recipe). The
  ddof=1 sd of the three holdout RMSEs (`lf.seed_sd`) is the training-noise scale against which
  drop-group differences are read. (Added by me; the brief did not ask for it, but without it a
  drop-group difference cannot be told from refit noise.)
- Caveat carried from BC-1: the in-fold target encodings are fitted on `tr`, which contains the ES
  months, so the ES rows' `te_*`/`de_*` read their own labels. `best_iteration` of every model here
  is therefore chosen on a slightly optimistic stopping metric, by an amount that depends on how
  many encoding columns the model has. Holdout RMSEs are unaffected (the holdout is outside `tr`).
  Drop-group models that remove encoding groups stop under a different amount of this bias; that
  is a known confound of §0.4 for encoding groups and is named, not corrected.

### 0.3 Groups (every one of the 83 columns in exactly one group)

| group | columns |
|---|---|
| `clock` — clock gaps | proxy, sp, eobt_p, lobt_p, iobt_p, aobt_sched, eobt_sched, aobt_eobt, lobt_sched, eobt_lobt, iobt_lobt, airborne3, airborne1, arvt_diff, aobt_sec, mvt_sec, sched_sec (17) |
| `stand_id` — stand identity | te/de_STAND_mvt, te/de_stand_pref, te/de_ars (6) |
| `stand_hist` — stand history | prev_arr_gap, prev_arr_taxiin, prev_arr_cdiff, prev_dep_gap (4) |
| `recent_taxi` — recent taxi times | dep_dur, rwy_dur, arr_dur, arr_ob60_taxiin, arr_ob60_cdiff, arr_ob180_cdiff, q_arr_taxiin_sym30_push, q_arr_taxiin_sym30_tko, q_rwy_ambient_proxy, q_next_arr_onblock_gap (10) |
| `queue_counts` — queue counts | n_push, rwy_b30, apt_b30, arr_b30, q_apt_at_push, q_rwy_at_push, q_pushed_after_me, q_rwy_tko_in_taxi, q_rwy_push_pre20, q_dep_tko_sym15, q_rwy_tko_sym10, q_arr_taxiing_at_push (12) |
| `traffic_freq` — traffic frequency | rwy_rate, sched_prev60, sched_next60, sched_day (4) |
| `season_time` — season/time | hr, tmin, dow, doy (4) |
| `airport` — airport identity | te/de_ADEP_mvt (2) |
| `runway` — runway identity | te/de_RUNWAY_mvt (2) |
| `actype_wake` — aircraft type + wake | te/de_AIRCRAFT_TYPE_mvt, te/de_WK_TBL_CAT_flt, actype_null (5) |
| `airline_seg` — airline/operator/segment/flight type | te/de_airline, te/de_AIRCRAFT_OPERATOR_flt, te/de_MARKET_SEGMENT_flt, te/de_FLIGHT_TYPE_flt (8) |
| `destination` | te/de_ADES_mvt (2) |
| `flags` — record flags | f_callsign, f_ades, f_type, f_rule (4) |
| `ordering` | o_dev_mvt, o_dev_flt, o_n_line (3) |

`aobt_sec` / `mvt_sec` / `sched_sec` are the second-of-minute of the AOBT_3 / MVT / SCHED
timestamps (`stand_ab.py:194–196`), i.e. clock-record structure, so they sit in `clock`, not in
`season_time`. `arr_b30` is a count, so it sits in `queue_counts` (`*_b30`), not `recent_taxi`.

### 0.4 Measures

1. **Permutation (reliance) on the holdout.** For each group, one row permutation (numpy
   `default_rng(seed)`) applied jointly to all of the group's columns across the 339,015 holdout
   rows; the seed-0 proxy predicts; ΔRMSE = RMSE(permuted) − RMSE(proxy). Permutation seeds 0, 1, 2;
   mean ± sd (ddof=1). The model is never refitted.
   **Single features:** groups ranked by mean permutation ΔRMSE; whole groups are taken in that
   order until at least 20 features are covered (so the count may exceed 20); each such feature is
   permuted alone, 3 seeds.
2. **Drop-group retraining (value).** For each of the 14 groups, the proxy recipe (§0.2, seed 0)
   refitted without that group's columns; ΔRMSE = RMSE(drop) − RMSE(full, seed 0) on the holdout.
   Interval: paired calendar-date block bootstrap over the take-off dates of the holdout rows
   (`capacity_f.fold_dates`; the `capacity_f.date_block` algorithm, 2,000 draws, seed 0, percentile
   2.5/97.5), sign flipped so that positive = dropping the group costs RMSE.
   **Budget rule (from the brief):** if the 14 retrains would take longer than the ~2 h budget,
   retrain the 8 groups with the largest permutation ΔRMSE plus the 4 smallest, and say so.
3. **Cuts.** For every group and both measures: ΔRMSE within the fill rows (|y − sp| ≤ 60 s), the
   tail (|delta| > 600 s), and the ordinary rows (neither); the overlap of fill and tail is counted
   in both and reported. Also each cut's share of the pooled ΔMSE (sum of ΔSE in the cut / 339,015),
   and ΔRMSE per airport (10 airports).

### 0.5 Classification rule (fixed now)

With `sd` = the retraining-noise sd of §0.2 and `[lo, hi]` = the drop-group day-block interval:
- **keep:** lo > 0 AND point > 2·sd.
- **candidate-to-drop:** point ≤ 2·sd AND lo ≤ 0 (dropping is not separated from refit noise or
  helps). Sub-labelled **dead** if the group's permutation ΔRMSE is < 0.05 s (the model barely reads
  it), **redundant** otherwise (the model reads it, but other groups substitute when it is gone).
- **not separated:** every other case (one condition of "keep" holds, the other does not).
- **under-used:** (a) matched lane — a group whose pooled drop cost is not "keep" but whose drop
  costs ≥ 1 s RMSE at some airport or within the fill or tail cut (local value hidden in the
  pooled number); (b) unmatched lane — an input present on the rows but absent from, or dead in,
  the S1C body.

### 0.6 The unmatched lane (S1C body)

- Frames through `unm_congestion.load_real` (the registered recipe, 12 raw months). The S1C body
  (`CongestionRegressor`: sp, dayoff, hr, hprox_med, hprox_n + te_ of ADEP_mvt, RUNWAY_mvt,
  stand_pref, airline, AIRCRAFT_OPERATOR_flt) refitted per LOMO month, seeds 0, 1, 2, and the S1C
  mixture rebuilt from the stored `p_hat` / `nf_cells` of `data/cache_stand/unm_congestion_lomo.parquet`.
  **Guard:** the rebuilt S1C must equal the stored S1C to ≤ 1e-6 s on every row (joined on MVT_ID),
  or no unmatched number is reported.
- Permutation: each of the 10 body inputs permuted within its test month (3 seeds), ΔRMSE of the
  S1C mixture on non-LIRF rows (the lane's scored population) and on all rows, plus on the rows
  routed to the body.
- `AIRCRAFT_OPERATOR_flt`: null share on the unmatched rows counted directly.
- `AIRCRAFT_TYPE_mvt`: presence share counted; then S1C + `te_AIRCRAFT_TYPE_mvt` as a single-key
  **out-of-fold** encoding (5-fold within each LOMO training set for training rows, the full
  training-set map for the test month; same smoothing and winsorised target as the body's other
  keys), 3 seeds, ΔRMSE vs the S1C rebuild with a date-block interval (2,000 draws, seed 0).

### 0.7 Compute discipline

AC power checked, no other `scripts/(lgbm_|capacity|catboost|rome_|unm_)` job running (verified
at start). Every process `OMP_NUM_THREADS=1 nice -n 19 /opt/homebrew/bin/python3.11 -B`, LightGBM
`num_threads` 4. The design matrix is written once to scratch as float32 and memory-mapped by the
workers. Peak memory target < 7 GB total.

### 0.8 Files

Scripts: `/private/tmp/claude-501/-Users-saurabhdxt-Projects-prc-challenge/81b461ea-ab3b-47e3-926a-3198c439b0c1/scratchpad/importance/`.
Predictions: `data/importance/`. Results: `reports/feature_importance.json` and this file.

### 0.9 Deviations logged during the run

**D1 (21:49, before any importance number existed): the §0.2 recipe failed the gate, so I switched
to the brief's other recipe once, with the gate unchanged.**
The direct early-stopped booster (fit rows only, stopped on months 3 and 9) scored **226.6661** on
the holdout (best_iter 3,327, 31 floor-bind rows; `data/importance/fit_direct_full_s0.json`), which
is 4.10 s from 222.56. It fails the 3 s gate. As §0.2 requires, no importance number is produced
from it, and its files are kept renamed `*_direct_full_s0*` as the record.
The brief allows two recipes: "`lf.fit_arm`, or lightgbm directly". I had picked the cheaper one.
I am now trying the other, **once**. It is `fit_arm`'s recipe at lr 0.05: early stopping (seed 0)
exactly as above gives best_iter, then n_ref = `L.n_refit(best_iter, n_tr, n_fit)`, then a refit on
all 1,723,425 training rows with n_ref trees at the model's seed, then a prediction on the holdout.
This is how arm F itself was built. The early-stopping run is identical to the one just done, so
the full model reuses best_iter 3,327 and only the refit is new. Seeds 1 and 2 for the noise floor
use the same seed-0 early-stopping run and refit at their own seeds, which is `fit_arm`'s seed
semantics. Each drop-group model runs its own seed-0 early stopping, then a seed-0 refit.
The gate stays at 3 s. If this recipe also fails, I stop and report, and there is no third try.
This is a forking-path risk and I am naming it: two recipes tried against one gate.

**D1 outcome (21:54):** the refit recipe, seed 0 (best_iter 3,327 → n_ref 4,159), scored
**225.1908** on the holdout, 2.63 s from 222.56, so the gate passes. The margin is narrow
(`data/importance/fit_full_s0.json`). This booster (`data/importance/booster_full_s0.txt`) is the
proxy for every number below.

Cost consequence: each drop-group model is now an early stopping plus a refit (≈ 7–8 min). To stay
near the 2 h budget, two training worker processes run concurrently, plus one permutation
process. Each is `OMP_NUM_THREADS=1 nice -n 19`, LightGBM `num_threads` 4 (12 threads in total on
an 18-core M5 Pro), and total memory is checked against the 7 GB target.

**D2 (≈22:05): the coordinator intervened; one training process only.** The coordinator reported
that `pmset` read Battery Power at some point after my launches. My only power check was in the
first (failed) launch command at 21:54, where it read AC Power. The relaunch 20 s later had no
check of its own. The coordinator SIGSTOPped everything, killed worker B (pid 1772), and resumed
worker A (pid 1683) and the permutation process (pid 1417) on AC. Audit: 1772 had skipped
`full_s1` (claimed by 1683) and was refitting `full_s2` when killed. It wrote no output: no
`pred_full_s2` or `fit_full_s2` exists, and every existing output has exactly one writer. From
here on: at most one training process, and `pmset -g batt` in its own step before every launch.
Because 1683 had already passed over `full_s2`, that seed runs later as a separate single launch.
Correction from the coordinator (22:36): 1772 was worker B of the claim-based queue, not an
accidental duplicate, and it was stopped under the owner's one-heavy-job rule. **One worker was
stopped mid-run.** Re-queued job: `full_s2`. Its orphaned claim (owner pid 1772 dead, no output)
was removed at 22:37, and it runs as a single launch after worker A's list ends. Every other
claim was checked against a live owner and a finished output. `claim_full_s0` (owner exited, output
present) is complete. `drop_clock`, `drop_stand_id`, `drop_recent_taxi` and `full_s1` belong to
1683 and are complete. `drop_queue_counts` belongs to 1683 and is running. `claim_drop_traffic_freq`
and `claim_drop_runway` contain the text `skipped-D3`: they are deliberate skips under D3, not
orphans, and they stay.

**D3 (≈22:05): the brief's budget fallback for drop-group retraining.** With one training process
at ≈ 8.5 min per drop-group model (early stopping plus refit, alongside the permutation process),
14 groups is ≈ 2 h of retraining alone, over budget. Per the brief and §0.4.2: retrain the 8
groups with the largest 3-seed permutation ΔRMSE plus the 4 smallest. The 2 middle groups are
skipped by placing claim files so that the running worker passes over them; it is not relaunched.
The 3-seed ranking (22:25) put `clock, stand_id, stand_hist, airline_seg, queue_counts, airport,
recent_taxi, ordering` in the top 8 and `season_time, destination, actype_wake, flags` in the bottom 4.
The skipped middle two are **`traffic_freq` (8.07 s) and `runway` (6.39 s)**: for them there is
permutation evidence only, no drop-group value.

**D4 (≈23:52): the quality gate came back RED, so `full_s2` and the registered unmatched refit were
not run.** A PreToolUse hook blocked the `full_s2` launch until the `/quality-gate` checklist had
passed. Item (a), the full suite `pytest tests/`, gave 574 passed and **4 failed** in 379.6 s. The
failures are `test_serve_time_contract.py` ×2 (BLOCK_TIME_UTC_mvt 100% null on the 344,841 scored
rows) and `test_stationarity.py` ×2 (the 2025→2026 unmatched-stratum rate shifts +0.69 pp in month 1
and −0.58 pp in month 7, at 8.7× and 6.9× the 95% interval). These four were already in the
last-failed cache before this session and none touch this diagnostic's code. Items (b), (c) and (f)
cannot run: there is no mutation runner, no coverage floor and no `tests/metrics.jsonl`. The gate
needs the user's acknowledgment for any waiver, and a subagent cannot obtain it, so I did not
write the pass token. Consequences: (1) the retraining-noise sd rests on 2 seeds, not 3; (2) the
unmatched-lane body refit (§0.6: S1C rebuild guard, per-input permutation, `+te_AIRCRAFT_TYPE`
out-of-fold refit) was not run. `scratchpad/importance/unm.py` is written and ready, and
`load_real` measured 23 s wall and 3.16 GB peak in `reports/unm_congestion_lomo.log`. In its
place I ran `unm_light.py`, a minimal read of 5 raw columns plus arithmetic on the stored LOMO
record: no model fit, 0.24 GB peak. It gives the null and presence counts exactly, and an
out-of-month **residual screen** of aircraft type. The screen is a weaker substitute for the body
refit and is labelled as such below.

---

## 1. Proxy model

| model | recipe | best_iter / n_ref | holdout RMSE | source |
|---|---|---|---|---|
| arm F (record) | lr 0.01, 3-seed refit mean | 21,722 / 27,155 | 222.5632 | `data/cache_stand/fold_preds_queue_order.parquet` |
| direct, seed 0 | lr 0.05, fit rows only, stopped on months 3 and 9 | 3,327 / — | 226.6661 (**fails the gate**) | `data/importance/fit_direct_full_s0.json` |
| **proxy, seed 0** | lr 0.05, fit_arm (stop, then refit on tr) | 3,327 / 4,159 | **225.1908** | `data/importance/fit_full_s0.json` |
| proxy, seed 1 | same stopping run, refit seed 1 | 3,327 / 4,159 | 225.4832 | `data/importance/fit_full_s1.json` |

Retraining sd (ddof=1, 2 seeds) = 0.207 s.

| cut (rows) | proxy | arm F |
|---|---|---|
| fill, \|y − sp\| ≤ 60 (30,167) | 285.61 | 285.63 |
| tail, \|delta\| > 600 (25,870) | 554.92 | 551.74 |
| ordinary, neither (285,942) | 174.46 | 171.47 |
| fill ∩ tail (2,964; counted in both) | 808.73 | 811.25 |

Proxy RMSE per airport: EDDF 178.1, EDDM 167.8, EGLL 232.4, EHAM 166.5, LEBL 220.0, LEMD 176.5,
LFPG 246.6, LIRF 380.0, LSZH 176.5, LTFM 244.9.

## 2. Group ranking — permutation (reliance) and drop-group retraining (value)

Permutation: 3 seeds, joint permutation of the group's columns on the holdout, using the seed-0
proxy (never refitted). Drop-group: the proxy recipe refitted without the group (seed 0 stopping
run, seed 0 refit). The interval is a paired calendar-date block bootstrap over 62 take-off dates
(2,000 draws, seed 0). Sorted by drop-group ΔRMSE.

| group | cols | permutation ΔRMSE, mean ± sd | drop-group ΔRMSE | 95% day-block | drop best_iter | class (§0.5) |
|---|---|---|---|---|---|---|
| `clock` | 17 | +268.297 ± 0.173 | **+36.644** | [+31.503, +42.298] | 2,611 | keep |
| `stand_id` | 6 | +85.522 ± 0.026 | **+3.885** | [+3.231, +4.527] | 3,461 | keep |
| `stand_hist` | 4 | +23.837 ± 0.196 | **+3.288** | [+2.766, +3.788] | 3,034 | keep |
| `airline_seg` | 8 | +17.567 ± 0.198 | **+3.257** | [+2.472, +4.050] | 3,860 | keep |
| `queue_counts` | 12 | +14.036 ± 0.063 | **+2.585** | [+2.188, +2.998] | 3,086 | keep |
| `ordering` | 3 | +9.764 ± 0.047 | **+1.918** | [+0.578, +3.671] | 3,197 | keep |
| `destination` | 2 | +3.207 ± 0.026 | **+1.142** | [+0.635, +1.632] | 2,697 | keep |
| `actype_wake` | 5 | +3.125 ± 0.132 | **+0.954** | [+0.567, +1.321] | 3,381 | keep |
| `recent_taxi` | 10 | +11.707 ± 0.075 | **+0.641** | [+0.127, +1.164] | 3,232 | keep |
| `airport` | 2 | +12.859 ± 0.148 | **+0.507** | [+0.105, +0.918] | 3,915 | keep |
| `season_time` | 4 | +4.333 ± 0.061 | +0.304 | [−0.429, +0.967] | 2,394 | candidate-to-drop (redundant) · under-used (local) |
| `flags` | 4 | +0.149 ± 0.021 | −0.131 | [−0.521, +0.276] | 3,130 | candidate-to-drop (redundant) |
| `traffic_freq` | 4 | +8.068 ± 0.076 | — | — | — | not retrained (D3) |
| `runway` | 2 | +6.386 ± 0.081 | — | — | — | not retrained (D3) |

Sensitivity to the 2-seed sd: `airport` (+0.507) and `recent_taxi` (+0.641) clear 2·sd = 0.41 s by
small margins. If a third seed pushed the sd above 0.25 s or 0.32 s respectively, they would move
to "not separated". Their intervals exclude 0 in both cases. Every other class is stable to any
plausible sd.

Structurally dead columns (`data/importance/column_structure.json`): `sched_sec` and `actype_null`
are constant on the training rows, have 0 splits in the 4,159-tree proxy, and permute to exactly
+0.000 s (`sched_sec`). They can be removed whatever the class of their groups.

### Single features (3 seeds each; the 23 columns of the top two groups, per §0.4.1)

| feature | group | perm ΔRMSE | | feature | group | perm ΔRMSE |
|---|---|---|---|---|---|---|
| `aobt_eobt` | clock | +133.745 ± 0.066 | | `arvt_diff` | clock | +1.777 ± 0.047 |
| `proxy` | clock | +96.366 ± 0.269 | | `airborne3` | clock | +1.017 ± 0.022 |
| `te_ars` | stand_id | +59.867 ± 0.104 | | `lobt_p` | clock | +0.995 ± 0.005 |
| `de_ars` | stand_id | +21.200 ± 0.245 | | `airborne1` | clock | +0.948 ± 0.035 |
| `aobt_sched` | clock | +14.055 ± 0.035 | | `de_STAND_mvt` | stand_id | +0.914 ± 0.067 |
| `lobt_sched` | clock | +11.890 ± 0.015 | | `te_stand_pref` | stand_id | +0.774 ± 0.066 |
| `eobt_lobt` | clock | +9.112 ± 0.484 | | `te_STAND_mvt` | stand_id | +0.740 ± 0.039 |
| `eobt_p` | clock | +9.054 ± 0.090 | | `de_stand_pref` | stand_id | +0.602 ± 0.034 |
| `sp` | clock | +2.510 ± 0.032 | | `iobt_lobt` | clock | +0.486 ± 0.014 |
| `eobt_sched` | clock | +2.144 ± 0.064 | | `aobt_sec` | clock | +0.120 ± 0.014 |
| `iobt_p` | clock | +1.942 ± 0.022 | | `mvt_sec` | clock | +0.101 ± 0.013 |
| | | | | `sched_sec` | clock | +0.000 ± 0.000 |

Within `stand_id`, reliance sits almost entirely on the airport|stand|runway key (`te_ars`
+59.9, `de_ars` +21.2). The STAND and stand-prefix encodings each carry < 1 s of reliance.

## 3. Where each group earns (cuts)

### 3a. Drop-group ΔRMSE within each cut (s)

| group | fill (30,167) | tail (25,870) | ordinary (285,942) | fill ∩ tail (2,964) |
|---|---|---|---|---|
| `clock` | +75.72 | +150.67 | +8.092 | +214.84 |
| `stand_id` | +3.61 | +6.23 | +3.938 | +8.63 |
| `stand_hist` | +1.44 | +10.01 | +2.084 | +3.84 |
| `airline_seg` | +11.67 | +11.80 | +1.525 | +40.59 |
| `queue_counts` | +0.38 | +5.45 | +2.303 | −0.36 |
| `ordering` | +4.81 | +14.63 | −1.394 | +15.78 |
| `destination` | +7.92 | +6.66 | −0.157 | +28.73 |
| `actype_wake` | +3.24 | +3.44 | +0.409 | +10.43 |
| `recent_taxi` | −1.05 | +0.71 | +0.799 | −3.28 |
| `airport` | +1.28 | +2.69 | −0.013 | +4.33 |
| `season_time` | +0.71 | +1.05 | +0.181 | +2.93 |
| `flags` | +0.23 | +0.70 | −0.351 | +1.89 |

### 3b. Drop-group ΔMSE contribution (s² per holdout row; total = fill + tail + ordinary − overlap)

| group | fill | tail | ordinary | overlap | where most of it is |
|---|---|---|---|---|---|
| `clock` | +4,359 | +14,492 | +2,437 | +3,442 | tail |
| `stand_id` | +185 | +531 | +1,172 | +123 | ordinary |
| `stand_hist` | +74 | +855 | +617 | +54 | tail, then ordinary |
| `airline_seg` | +606 | +1,010 | +451 | +588 | fill + tail (Rome) |
| `queue_counts` | +20 | +464 | +682 | −5 | ordinary, then tail |
| `ordering` | +246 | +1,255 | −409 | +225 | tail; costs on ordinary |
| `destination` | +408 | +567 | −46 | +414 | fill + tail (Rome) |
| `actype_wake` | +166 | +293 | +121 | +148 | fill + tail |
| `recent_taxi` | −53 | +61 | +236 | −46 | ordinary |
| `airport` | +65 | +229 | −4 | +61 | tail |
| `season_time` | +36 | +89 | +53 | +42 | nothing separated |
| `flags` | +12 | +60 | −103 | +27 | nothing separated |

### 3c. Drop-group ΔRMSE per airport (s)

| group | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|
| `clock` | +24.83 | +36.21 | +44.18 | +14.46 | +16.79 | +18.67 | +41.90 | +89.29 | +24.17 | +31.00 |
| `stand_id` | +6.65 | +0.59 | +1.46 | +1.50 | +4.23 | +5.58 | +2.81 | +2.08 | +11.16 | +6.62 |
| `stand_hist` | +2.53 | +0.74 | +8.17 | −0.57 | +1.14 | +0.05 | +7.13 | +1.64 | +1.82 | +5.30 |
| `airline_seg` | +1.60 | +0.52 | +0.50 | +0.89 | +1.19 | +0.93 | +1.98 | **+16.75** | +0.15 | +2.15 |
| `queue_counts` | +3.85 | +2.19 | +2.32 | +1.26 | +3.79 | +1.51 | +2.46 | +0.67 | +3.32 | +5.02 |
| `ordering` | +1.40 | +2.26 | −0.62 | −0.29 | +1.51 | +0.07 | +1.54 | **+9.25** | −0.38 | +1.71 |
| `destination` | +0.76 | +0.10 | −0.94 | +0.25 | +0.95 | −0.97 | +1.36 | **+6.73** | −0.16 | +0.83 |
| `actype_wake` | +0.75 | +1.04 | +0.22 | −0.52 | +1.42 | −0.46 | +2.25 | +1.68 | +0.79 | +1.54 |
| `recent_taxi` | −0.05 | +0.11 | +1.15 | +0.12 | +1.47 | −0.49 | +1.82 | −1.72 | +0.81 | +2.45 |
| `airport` | −0.01 | +0.01 | +0.19 | +0.49 | +1.21 | −0.36 | +1.77 | +0.79 | +0.46 | +0.16 |
| `season_time` | +0.31 | −0.58 | +1.22 | +0.14 | +0.10 | −0.63 | +2.33 | +1.01 | −0.17 | −1.51 |
| `flags` | +0.55 | −0.44 | −0.21 | −0.83 | +0.40 | −0.20 | +0.33 | −0.88 | −0.13 | +0.06 |

Per-airport differences are single-seed refits on 21,811–45,882 rows. Individual cells of
about 1 s sit inside refit noise at the airport level; the airport-level noise was not measured.

### 3d. Permutation ΔRMSE within each cut (s, mean of 3 seeds)

| group | fill | tail | ordinary | fill ∩ tail |
|---|---|---|---|---|
| `clock` | +317.28 | +596.12 | +215.199 | +625.49 |
| `stand_id` | +90.25 | +128.28 | +86.281 | +205.37 |
| `stand_hist` | +15.18 | +61.53 | +18.030 | +39.22 |
| `airline_seg` | +50.70 | +51.75 | +11.257 | +166.24 |
| `queue_counts` | +5.03 | +16.74 | +15.793 | +2.64 |
| `airport` | +20.52 | +35.70 | +9.150 | +68.39 |
| `recent_taxi` | +2.54 | +7.16 | +15.203 | −0.02 |
| `ordering` | **−4.59** | **−3.63** | +15.493 | −19.22 |
| `traffic_freq` | +3.89 | +8.13 | +9.625 | +7.38 |
| `runway` | +17.14 | +19.74 | +3.828 | +56.09 |
| `season_time` | +3.80 | +4.37 | +5.199 | +10.44 |
| `destination` | +12.64 | +11.37 | +1.534 | +43.30 |
| `actype_wake` | +2.18 | +2.31 | +3.955 | +4.76 |
| `flags` | +0.47 | +1.13 | −0.106 | +1.50 |

### 3e. Permutation ΔRMSE per airport (s, mean of 3 seeds)

| group | EDDF | EDDM | EGLL | EHAM | LEBL | LEMD | LFPG | LIRF | LSZH | LTFM |
|---|---|---|---|---|---|---|---|---|---|---|
| `clock` | +185.20 | +284.07 | +327.79 | +212.13 | +209.32 | +229.36 | +221.26 | +327.03 | +172.07 | +396.24 |
| `stand_id` | +85.69 | +80.33 | +82.51 | +134.84 | +78.78 | +82.52 | +64.54 | +84.60 | +136.71 | +70.00 |
| `stand_hist` | +17.29 | +12.69 | +65.02 | +16.68 | +13.68 | +16.72 | +20.86 | +10.65 | +8.18 | +32.90 |
| `airline_seg` | +11.79 | +6.59 | +6.04 | +12.87 | +4.00 | +11.67 | +7.04 | +71.91 | +9.97 | +12.97 |
| `queue_counts` | +15.65 | +12.98 | +12.42 | +10.32 | +18.52 | +13.97 | +13.67 | +13.38 | +16.18 | +17.58 |
| `airport` | +7.53 | +10.15 | +20.48 | +17.44 | +7.25 | +8.19 | +13.47 | +22.47 | +14.37 | +6.63 |
| `recent_taxi` | +10.94 | +13.82 | +17.35 | +7.85 | +14.22 | +12.17 | +10.06 | +7.37 | +16.53 | +13.10 |
| `ordering` | +6.60 | +10.00 | +4.00 | +2.10 | +12.50 | +10.94 | +8.81 | +17.44 | +2.87 | +16.88 |
| `traffic_freq` | +7.29 | +10.42 | +8.94 | +15.33 | +5.80 | +6.38 | +6.58 | +4.20 | +9.45 | +10.41 |
| `runway` | +5.77 | +4.90 | +5.91 | +8.53 | +2.83 | +3.49 | +5.30 | +11.97 | +7.11 | +7.11 |
| `season_time` | +3.99 | +2.71 | +5.34 | +3.94 | +3.11 | +5.36 | +5.66 | +6.58 | +2.38 | +3.19 |
| `destination` | +0.58 | +0.65 | +1.02 | +0.61 | +0.83 | +0.76 | +1.29 | +16.01 | +0.59 | +3.49 |
| `actype_wake` | +2.84 | +4.46 | +1.87 | +3.07 | +1.64 | +3.85 | +2.98 | +3.97 | +2.79 | +4.40 |
| `flags` | +0.27 | +0.15 | +0.03 | +0.77 | +0.02 | −0.05 | +0.06 | +0.11 | +0.07 | +0.15 |

### 3f. Reading the cuts (measured, not interpreted beyond the tables)

- `clock` earns everywhere, and by far most in the tail (+150.7 s within the tail). LIRF is the
  largest airport (+89.3 s). Two columns carry its reliance: `aobt_eobt` and `proxy`.
- `stand_id`, `queue_counts` and `recent_taxi` earn mainly on the **ordinary** rows and at every
  airport, i.e. they are the body of the model. `stand_id` peaks at LSZH (+11.2 s).
- `stand_hist` earns in the **tail** (+10.0 s within the tail) and at EGLL (+8.2) and LFPG (+7.1).
- `airline_seg`, `destination` and `ordering` earn in the **fill and tail** rows, concentrated at
  **LIRF** (+16.8 / +6.7 / +9.3 s). Pooled, they look like mid-sized groups. Most of their value
  is the Rome record-structure lane.
- `ordering`: reliance and value point opposite ways by cut (3a vs 3d). The refit without it
  loses 14.6 s in the tail and gains 1.4 s on ordinary rows. Permuting it in the fitted model
  gains on fill/tail and loses 15.5 s on ordinary rows.
- `airport` and `recent_taxi`: the model leans on them (12.9 s and 11.7 s of reliance), but a refit
  without them gets most of it back (+0.5 s and +0.6 s). Other groups substitute: stand identity
  already encodes the airport.

## 4. Keep / candidate-to-drop / under-used (the §0.5 rule applied)

- **Keep (10):** clock, stand_id, stand_hist, airline_seg, queue_counts, ordering, destination,
  actype_wake, recent_taxi, airport. The last two are only marginally above 2·sd with a 2-seed sd
  (§2).
- **Candidate-to-drop (2):** `flags` (drop −0.131 [−0.521, +0.276]; reliance +0.149 s, just above
  the 0.05 s "dead" line, hence labelled "redundant") and `season_time` (drop +0.304
  [−0.429, +0.967]; reliance +4.3 s, redundant).
- **Dead columns (2):** `sched_sec` and `actype_null` are constant with 0 splits (§2), in addition to
  the group classes.
- **Under-used (a), matched:** `season_time`. Its pooled drop cost is not separated, but it costs
  ≥ 1 s in the tail (+1.05) and at LFPG (+2.33), EGLL (+1.22) and LIRF (+1.01). Those local cells
  are single-seed and near the refit-noise scale.
- **Not classified:** `traffic_freq` and `runway` (not retrained, D3). They have permutation reliance
  only: 8.07 s and 6.39 s.
- **Under-used (b), unmatched:** `AIRCRAFT_OPERATOR_flt` is dead in the S1C body (100% null), and
  `AIRCRAFT_TYPE_mvt` is present but absent from the body (§5).

## 5. Unmatched lane (S1C body)

Source: `data/importance/unm_light.json` (`unm_light.py`). The registered §0.6 refit was not run
(D4).

| item | measured |
|---|---|
| stored LOMO rows | 22,219 (non-LIRF 20,731), `data/cache_stand/unm_congestion_lomo.parquet` |
| `AIRCRAFT_OPERATOR_flt` null share | **1.000000** on all 22,219 rows (joined to the raw months on MVT_ID; every row has AOBT_3 null) |
| `AIRCRAFT_TYPE_mvt` present | 0.9323 of all rows; **0.9992 of non-LIRF rows**; 192 types |

The operator key encodes to the prior for every row (`NonFillRegressor.design` maps
`astype(str)` of a null to one level), so `te_AIRCRAFT_OPERATOR_flt` is a constant column: one of
the body's five encodings does nothing.

**AIRCRAFT_TYPE residual screen** (not the registered refit). Aircraft type is encoded out-of-month
on S1C's residual y − S1C: winsorised at 3,000 s, shrunk toward 0 with 50 pseudo-rows, fitted on
the other 11 months, and added to S1C on the held-out month, floored at 1 s. The negative control
does the same with type labels permuted across rows.

| cut | n | S1C RMSE | + type ΔRMSE [95% day-block] | control ΔRMSE [95%] |
|---|---|---|---|---|
| non-LIRF | 20,731 | 1,104.75 | −0.521 [−1.695, +0.452] | −0.177 [−0.645, +0.497] |
| all | 22,219 | 1,569.48 | +0.461 [−0.512, +1.465] | −0.322 [−0.801, +0.236] |
| non-LIRF, y ≤ 10,800 | 20,725 | 581.53 | −0.821 [−2.340, +0.910] | +0.375 [−0.441, +1.384] |

Measured: the screen does not separate an aircraft-type correction from 0, nor from the
permuted-label control. What it cannot rule out: an interaction the body's trees would find
between type and sp/hour/witness. Only the registered refit (`unm.py`) tests that.

## 6. Compute

Wall time, first launch to last result: 21:44 → 00:06, ≈ 2 h 22 min. That includes the gate's
6.3 min test suite and the coordinator's SIGSTOP pause, whose length I did not measure. Per model:
early stopping 143–304 s, refit 175–367 s, holdout predict 32–75 s. The permutation predicts took
40–48 s each: 42 group predicts and 69 single predicts. Peak RSS per process: prep 3.46 GB;
training 2.46–3.33 GB (`fit_*.json`); permutation 0.87 GB; `unm_light` 0.24 GB. The concurrent peak
(21:54–22:05, two training workers plus the permutation process) was ≈ 6.1 GB of summed RSS,
counting the shared memory-mapped design matrix once per process. After D2 it was ≈ 4.0 GB.

## 7. Files

- `data/importance/`: `holdout_base.parquet` (holdout rows + take-off dates), `pred_{full_s0,full_s1,drop_*}.parquet`,
  `fit_*.json`, `booster_full_s0.txt` (the proxy), `perm/{group,single}__*.npy` (permuted predictions,
  delta scale), `column_structure.json`, `unm_light.json`. The gate-failed first recipe is kept as
  `*_direct_full_s0*`.
- `reports/feature_importance.json` holds every table above plus the classification record.
- Scripts are in the scratchpad: `common.py`, `prep.py`, `train.py`, `perm.py`, `analyze.py`,
  `unm_light.py`, and `unm.py` (not run).
- No existing repo file was edited. No git.
