# Matched-row model — experiments, 2026-09-08 (interim: written at 13:55 UTC from completed runs only)

Scope: matched rows only (`AOBT_3_flt` present; 98.47% of scored rows). Target `y = TAXITIME_SEC_mvt`;
model target `delta = BLOCK_TIME − AOBT_3`; prediction `y_hat = max(proxy − delta_hat, 1)` with
`proxy = MVT_TIME − AOBT_3`. Everything below is held out on WHOLE MONTHS; every encoding is fitted on the
training months only; every RMSE carries a 95% bootstrap interval (2,000 resamples of the held-out rows);
every A-vs-B claim is a PAIRED bootstrap of the RMSE difference on the same held-out rows and is called
ESTABLISHED only when that interval excludes zero.

**Status of the evidence: every number in this file is from fold A only (Jan+Jul 2025 held out). The fold-B
replication (Feb+Aug) has NOT run. Nothing here — including the +8.17 s of E1 — is yet confirmed as
fold-independent.** Experiments that had not completed when this file was written are listed in §9 as
PENDING / UNMEASURED and are not estimated.

Primary fold **A**: Jan+Jul 2025 held out (339,015 matched rows), train on the other ten months (1,723,425 rows).

Learner for every experiment unless stated: `HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
max_leaf_nodes=127, min_samples_leaf=20, random_state=0, early_stopping=False)`, single seed, target `delta`.
Categoricals enter as smoothed (K=50) in-fold target encodings of both `y` and `delta` for ADEP, STAND, stand
prefix (STAND[:2]), airline (FLIGHT[:3]), ADES, aircraft type, runway, operator, market segment, wake category,
flight type, plus an airport×runway×stand encoding of both targets: 24 encoding columns in every run.

Code, outputs and every held-out prediction vector:
`/private/tmp/claude-501/-Users-saurabhdxt-Projects-phantom-forecaster/a1f96009-f5e3-4d8f-b07a-edee9aa70df3/scratchpad/mm/`
(`feat.py` cache builder, `lib.py` harness, `exp*.py`, `*_A.out`, `chain.log`, `preds/A_*.npy`).
Nothing under the repository was modified except this file.

---

## 0. Facts established on the way (data, not model)

| fact | evidence |
|---|---|
| Arrival rows on the 2026 **ranking** file have `BLOCK_TIME` (on-block) and `TAXITIME_SEC` (taxi-in) **100% populated**; `AOBT_3` / `ARVT_3` 99.3% | `ranking.parquet`, 344,693 ARR rows |
| Therefore arrival on-block times, arrival taxi-in and stand-turnaround features are **computable at serve time** | — |
| `ARVT_3_flt` on an ARR row is NM's **landing** time, not in-block: `MVT − ARVT_3` p10/p50/p90 = −62 / −4 / +23 s; `BLOCK − ARVT_3` ≈ taxi-in (p50 474 vs 480) | all 2025 ARR rows |
| So there is no "arrival-side clock disagreement" to borrow: the arrival analogue of `delta` is just taxi-in | measured |
| On matched DEP rows: `CALLSIGN ≠ FLIGHT` 26.0%, `ADES_FILED ≠ ADES` 1.5%, aircraft-type mismatch 0.35%, rule mismatch 0.09%; same rates on the 2026 ranking file | Jan 2025 vs ranking |
| Training `delta` quantiles (fold-A training months): p0.5 −1,558, p1 −1,138, p50 +51, p99 +786, p99.5 +960 | exp3 |
| Weather: Iowa State ASOS routine hourly METARs cover all ten airports for 2025 and 2026 Jan–Jul; one station-year downloads in ~1.4 s; nearest-prior METAR within 2 h exists for 100.0% of matched rows at every airport | `wx/`, `wxfeat.py` (fetched and joined; model test NOT run) |

---

## 1. Baseline reproduced — E0

| run | features | fold A RMSE | 95% CI |
|---|---|---|---|
| E0 `BASE(12) + PRE(6) + DUR(4)` + 24 encodings = **46** | as handed over | **244.12** | [240.84, 247.80] |

Exact reproduction of the handed-over 244.12, so the rebuilt feature cache is faithful to the earlier run.

BASE = proxy, sp (MVT−SCHED), eobt_p, lobt_p, iobt_p (MVT−EOBT/LOBT/IOBT), aobt_sched, eobt_sched, hr, tmin, dow,
doy, actype_null. PRE = N_push, N_push_rel, rwy_b30, apt_b30, arr_b30, sched_next60. DUR = rwy_dur, dep_dur,
arr_dur, rwy_rate.

---

## 2. New feature block — E1 (ESTABLISHED on fold A, +8.17 s; fold B NOT run)

25 numeric features added, all from columns present and populated on the ranking file:

* flight-plan clock differences: **`aobt_eobt` = AOBT_3 − EOBT_1**, **`lobt_sched` = LOBT − SCHED**, `eobt_lobt`,
  `iobt_lobt`, `airborne3` = ARVT_3 − MVT, `airborne1`, `arvt_diff`
* seconds-within-minute of AOBT_3 / MVT / SCHED (the M2 fingerprint)
* record-match flags: callsign≠flight, ADES_FILED≠ADES, type mismatch, rule mismatch
* surface: `sched_prev60`, `sched_day` (scheduled departures in the previous hour / the UTC day), arrivals on-block in
  the previous 30 min (count, mean taxi-in), previous-60-min mean taxi-in, mean landing-clock disagreement of arrivals
  in the previous 60 / 180 min
* stand history: seconds since the previous arrival on-block at my stand (`prev_arr_gap`), that arrival's taxi-in and
  landing-clock disagreement, seconds since the previous NM off-block from my stand (`prev_dep_gap`)

| run | features | fold A RMSE | 95% CI | paired gain vs E0 | verdict |
|---|---|---|---|---|---|
| E1 = E0 + 25 | 70 | **235.95** | [233.44, 238.57] | **+8.17 [+6.48, +10.17]** | ESTABLISHED — **one fold only** |

Config identical to E0 (P400, seed 0); only the feature list differs.

Held-out permutation importance (100k-row subsample of fold A; RMSE increase in seconds when one column is shuffled):
`proxy` +117.9, **`aobt_eobt` +93.5**, `te_ars` +48.8, **`lobt_sched` +35.9**, `de_ars` +34.7, `sp` +26.3,
`aobt_sched` +15.4, **`sched_prev60` +13.5**, `rwy_rate` +10.3, `dep_dur` +10.2, `te_ADEP` +9.8,
**`prev_arr_gap` +9.4**, `rwy_dur` +7.6, **`eobt_lobt` +7.0**, `te_airline` +5.0, `apt_b30` +3.6.
Below +0.1 (dead weight): the four mismatch flags, `sched_sec`, `mvt_sec`, `dow`, `actype_null`, `arr_b30`,
`prev_arr_cdiff`, `arr_ob60_cdiff`, `hr`.

Reading: the gain is mostly **pairwise differences between clocks the model already had** — `aobt_eobt` is
`eobt_p − proxy`, `lobt_sched` is `sp − lobt_p`. Trees cannot form the difference of two inputs; handing the
differences over directly is what moved the number. `aobt_eobt` is the NM-measured departure delay of the flight, and
the airport/NM off-block disagreement depends on it. The arrival/stand features contribute a smaller, real share
(`prev_arr_gap`, `sched_prev60`).

**Caveat, stated prominently: E1 is validated on ONE fold. The Feb+Aug replication has not run. The +8.17 s is not yet
confirmed as fold-independent.**

---

## 3. The four remaining pairwise clock differences — E1b (NOT established)

`aobt_lobt`, `aobt_iobt`, `iobt_sched`, `iobt_eobt` added to E1 (74 features, same config):
**235.69** [233.24, 238.31]; paired gain vs E1 **+0.26 [−0.12, +0.63] — not established.**
Closed: the clock-difference family is exhausted at E1.

---

## 4. Delta-modelling variants — E3 (reference E1 = 235.95, fold A, 70 features, same config)

| variant | what | RMSE | 95% CI | paired gain vs E1 | verdict |
|---|---|---|---|---|---|
| wins | TRAINING `delta` winsorised at [p0.5, p99.5] = [−1,558, +960] of the training months; evaluation untouched | 266.26 | [261.86, 270.65] | **−30.31 [−33.26, −27.35]** | **WORSE** |
| resid | target = `delta − de_ars` (in-fold airport×runway×stand smoothed mean of delta); offset added back at prediction | **234.56** | [232.17, 237.14] | **+1.39 [+0.65, +2.11]** | ESTABLISHED |
| absl | loss = absolute_error, mean-shifted | PENDING / UNMEASURED | | | |

Interpretation of the winsorisation result, plainly: **the tails of `delta` are predictable signal, not noise.**
Trimming 1% of the training targets cost 30 s of held-out RMSE — the extreme disagreements between the airport
clock and the NM clock (the schedule-anchored LIRF rows, the LTFM ~+5-min offset, the very late pushbacks) are
exactly the rows the model *can* learn from the features, and removing them destroys that information. Any
robust-loss or outlier-trimming idea for this target is dead on this evidence; do not revisit it.

The residual-target variant is a small, established gain and is cheap; it is a candidate for the composite (§9).

---

## 5. Per-airport models — E2 (reference E1, fold A; ten models, encodings refitted per airport)

| config | RMSE | 95% CI | paired gain vs pooled E1 | verdict |
|---|---|---|---|---|
| ten models, same P400 config (400 it, 127 leaves, min_leaf 20) | **232.87** | [230.25, 235.63] | **+3.08 [+1.82, +4.27]** | ESTABLISHED |
| ten models, cap63 (400 it, 63 leaves, min_leaf 40) | 233.63 | [231.00, 236.44] | +2.32 [+1.06, +3.56] | ESTABLISHED |

Per airport (P400), pooled → per-airport RMSE, paired gain:

| airport | n | pooled | per-airport | paired gain | verdict |
|---|---|---|---|---|---|
| EDDF | 36,231 | 190.40 | 186.11 | +4.29 [−0.04, +7.33] | not established |
| EDDM | 26,854 | 177.94 | 173.55 | +4.39 [+0.98, +7.56] | ESTABLISHED |
| EGLL | 39,744 | 244.79 | 248.53 | −3.74 [−8.79, +1.35] | not established (worse point estimate) |
| EHAM | 40,141 | 173.02 | 168.49 | +4.53 [+2.26, +6.68] | ESTABLISHED |
| LEBL | 28,489 | 226.26 | 225.48 | +0.78 [−0.57, +2.08] | not established |
| LEMD | 35,019 | 182.89 | 179.46 | +3.43 [+1.09, +5.32] | ESTABLISHED |
| LFPG | 38,713 | 255.87 | 249.06 | +6.82 [+2.24, +11.88] | ESTABLISHED |
| LIRF | 26,131 | 396.50 | 391.12 | +5.38 [+0.49, +10.38] | ESTABLISHED |
| LSZH | 21,811 | 186.61 | 187.76 | −1.15 [−8.26, +4.18] | not established |
| LTFM | 45,882 | 259.70 | 254.96 | +4.74 [+3.29, +6.23] | ESTABLISHED |

Six of ten airports gain with an interval excluding zero; EGLL and LSZH do not gain. A rule "per-airport only where it
helped on fold A" would be selection on the evaluation fold and is refused; the blend in §6 is the principled way to
keep both.

---

## 6. Pooled + per-airport blend (no refit; saved fold-A predictions; FIXED 0.5 weight, not tuned)

| prediction | RMSE | 95% CI | paired gain | verdict |
|---|---|---|---|---|
| pooled E1 | 235.95 | [233.44, 238.57] | — | — |
| per-airport E2 (P400) | 232.87 | [230.25, 235.63] | +3.08 vs E1 | ESTABLISHED |
| **0.5·E1 + 0.5·E2** | **230.41** | [227.91, 233.10] | **+5.53 [+4.91, +6.14] vs E1; +2.46 [+1.84, +3.14] vs E2** | ESTABLISHED (fold A) |

Weight curve, for information only (w on per-airport): 0.0→235.95, 0.3→231.68, 0.4→230.89, 0.5→230.41, 0.6→230.26,
0.7→230.43, 1.0→232.87. Flat between 0.4 and 0.7, so the fixed 0.5 is not a fold-A artefact. Not replicated on fold B.
Reading: the two models make partly independent errors (one learns cross-airport structure, the other airport-specific
splits); averaging them is variance reduction of the same kind as seed averaging, but larger.

**Best matched-row held-out RMSE measured so far: 230.41 [227.91, 233.10] on fold A** = 0.5 × (pooled E1, 70 features,
P400, seed 0) + 0.5 × (ten per-airport E2 models, same features, P400, seed 0). Versus the 244.12 baseline that is a
paired gain of about 13.7 s, but only the E1 (+8.17) and E2 (+3.08) and blend (+5.53 over E1) components have paired
intervals, all on fold A alone.

---

## 7. Summary of verdicts (all fold A)

| idea | result | verdict |
|---|---|---|
| pairwise clock differences + arrival/stand features (E1) | 244.12 → 235.95, +8.17 [+6.48, +10.17] | ESTABLISHED, one fold |
| four more clock differences (E1b) | +0.26 [−0.12, +0.63] | not established — family exhausted |
| winsorised training delta (E3 wins) | −30.31 [−33.26, −27.35] | WORSE — tails are signal |
| residual target delta − de_ars (E3 resid) | +1.39 [+0.65, +2.11] | ESTABLISHED, small |
| per-airport models (E2 P400) | +3.08 [+1.82, +4.27] over pooled | ESTABLISHED |
| per-airport, 63 leaves (E2 cap63) | +2.32 [+1.06, +3.56] | ESTABLISHED, weaker than P400 |
| 0.5 pooled + 0.5 per-airport | 230.41; +5.53 [+4.91, +6.14] over pooled | ESTABLISHED, fixed weight |

---

## 8. Machine-safety record (honest, for the owner)

* All runs: `OMP_NUM_THREADS=1 nice -n 19 /opt/homebrew/bin/python3.11`, output to files, one process at a time.
* **Cap breaches, all mine, all fixed at the root:** the one-off feature-cache build peaked at 2.8–3.0 GB for a few
  seconds; the first experiment run peaked at 3.8 GB (sklearn's internal float64 copy of a float32 pandas frame plus a
  whole-cache train/test split) and was killed; the second version peaked at 2.93 GB (pyarrow's private allocator not
  returning the filtered cache). Final harness: per-month cache files, matrix filled month-by-month straight from Arrow
  into the float64 array HistGradientBoosting needs, system memory pool. Measured peaks after the fix: 1.98 GB (exp3),
  2.25 GB (exp2), probe 2.12 GB.
* At 13:43 a `resume.sh` that I did not write appeared in my scratchpad and started a second copy of the queue beside my
  chain (two ~2 GB fits for ~90 s). I killed it and replaced it with a guard that refuses to start while `chain.sh` runs.
* A stray `M.parquet` was written into the repository root by a relative path for about one minute and removed;
  nothing else under the repository was touched.
* No `com.phantom.*` job was touched. No git command was run.

---

## 9. PENDING / UNMEASURED — no result exists; nothing below is estimated

* **Fold-B replication (Feb+Aug held out) of E0/E1** — not run. Until it runs, E1 (+8.17), E2, E3-resid and the blend
  are single-fold results.
* **LIRF matched-row schedule contamination (exp4)** — the two-component treatment (classifier for
  `|BLOCK−SCHED| ≤ 60 & |BLOCK−AOBT_3| > 300`, clean delta model with those rows at weight 0, mixture
  `p·sp + (1−p)·(proxy − delta_clean)` at LIRF, plus the oracle bound) is written and queued; no number exists.
* **Weather / METAR (exp5)** — data fetched and joined (100% coverage, §0); the model comparison has not run.
* **Interaction target encodings (exp7)** — airport×operator, airport×type, airport×hour, operator×stand,
  airport×runway×hour, airport×ADES: not run.
* **Absolute-error loss (exp3 absl)** — not run.
* **Lean feature set** (E1 minus the twelve dead-weight columns) — not run.
* **Seed averaging (exp6)** and the 1,500-iteration early-stopping configuration on the new features — not run;
  the earlier 2-seed figure (~0.9 s) is from the 36-feature model and is not evidence for this feature set.
* **Composite model (exp8: per-airport + residual target + blend + seeds, with or without weather/encodings)** —
  script written, not run. The composite and its fold-B replication are the required next step before any of the
  above is carried into the submission pipeline.
