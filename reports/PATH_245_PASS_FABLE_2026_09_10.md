# Path to 245 — second independent pass (whole system, every closure), 2026-09-10 evening

**Premise attacked:** the leader scores 245.02 on the same data, so a path exists and we are missing it.
**Prior verdict attacked:** `reports/PATH_TO_245_FABLE_2026_09_10.md` (≈ 3 % reachable).

**Verdict up front.** The pipeline is not broken: target, floor, proxy, calibration, feature parity between the
fold and the submission path, unseen categories, and v7's file all check out (§1). The fold is not lying about
the matched lane except by a regime term of +1.2 k to +2.5 k MSE that 2026 January adds (§2). **One structural
defect is real and unpriced by the project:** the shipped unmatched estimator S1 predicts its non-fill body from
`sp, dayoff, hr` and five identity encodings and carries **no congestion information at all**
(`scripts/build_submission.py:66-67`), while 2026 January contains disruption hours far outside 2025's support.
Measured out of sample on the 2025 fold, S1 is biased low by 340–1,060 s in congested hours and the matched
model's own hour-mean prediction is an unbiased serve-time estimate of the unmatched rows' level; the gated
substitution rule earns **+509 MSE [210, 896] date-block 95 %, 8/9 airports** on 2025's 857 exposed rows, and
2026 has **1,242 exposed rows with 4,743 MSE of bias² at stake** (§3). That, plus the E1 band floor already built
as `v8`, is a credible **≈ +8–10 k** and puts **274 in reach and 266 plausible**. **245 is still not reachable on
evidence**: the remaining 12–14 k sits in LIRF's 2–4 h fill band (≈ 2.7 k oracle), LIRF's non-fill class
(≈ 8.5 k, date-slips and holds — Family B), and a matched lane where no arm has ever moved more than 1.7 s.
Honest P(245 by 11 Oct) ≈ **5–8 %**, up from 3 % because E3 was under-priced by 2–3×, not because a 21 k object
was found. The measurement that changes the verdict is named in §6.

Compute: read-only parquet passes, ≤ 1.6 GB, ≤ 2 min each, `OMP_NUM_THREADS=1 nice -n 19`. No fitting, no
cache build, no submission, no board reading beyond the totals already in the ledger. Scripts in §8.
Conventions obeyed (BC-2): arm F's record is DELTA, recovered as `max(proxy − treatment, 1)`, reproduces
**222.56316**; `stratum_fold_v7_preds` / `rome_dateslip_preds` are taxi times; `row` indexes the twelve-month
cache concatenation (alignment asserted on `proxy`/`sp` for all 339,015 rows).

---

## 1. Pipeline audit — WORKING (nothing to fix), with the numbers

All on `data/cache_stand/fold_preds_queue_order.parquet` (arm F, 339,015 rows) unless stated; weighted MSE =
`w_m · SSE / n`, `w_m = 339551/344841` (`scratch fold_target.py`).

| check | measurement | verdict |
|---|---|---|
| Target / floor | floor `max(·,1)` binds on **22 rows**, 0.10 % of matched SSE; unfloored RMSE 223.07 vs 222.56 | floor costs nothing, helps 0.5 s |
| Proxy sanity | `proxy ≤ 0` on 248 rows (RMSE 830, **496 wMSE**), `proxy > 7,200` on 4 rows (103 wMSE); 336,784 rows in (300, 7,200] carry 47,566 of 48,774 | proxy is not wrong on any class worth a lane |
| Calibration | per airport×month mean-shift oracle **309 wMSE**; per-airport affine oracle **119**; `delta ~ delta_hat` slope 1.011; scaling `delta_hat` by 0.9/1.1/1.2 gives 225.7/225.1/233.0 | no bias, no scale error, L2 on delta is the right objective for RMSE |
| Seeds | 223.20 / 223.09 / 222.94 → mean 222.56 | consistent |
| Fold ↔ submission parity | ES months (3, 9) in both (`lgbm_fold.py:676` → `L.ES_MONTHS`; `lgbm_submit.py:146`); `n_ref = best_iter·n_all/n_fit` in both; **83 identical features** (v6 meta, 68 + 12 queue + 3 order); encodings fitted on all training rows in both; submission trains 12 months, fold 10 — the only intended difference | no silent divergence |
| Base chain | v3 base v2 → v4 → v5 base v4 → v6 base v5 → v7 base v6 (metas); v7 differs from v6 on 382 LIRF unmatched rows only; **`v8` exists on disk (14:47, base v7, E1 band floor, 13 rows differ) — not in the handoff** | chain intact; v8 is unrecorded in the handoff and needs the owner's decision |
| v7 internal sanity | 28 rows at 1 s (10 matched ≤ 60 s), max 95,974 (the sp 111,654 hedge); per airport×month prediction quantiles track 2025 label quantiles except **January 2026 at EDDF/EDDM/EHAM/LSZH/LTFM, where predictions sit 10–17 % above any 2025 January** — consistent with the serve-time shift in §2, not a bug | sane |
| Unseen categories 2026 vs 2025 | STAND 0.46 % (fold holdout 0.04 %), `ars` 0.71 % (0.13 %); EDDM stands **4.5 %** unseen, EDDF `ars` 2.3 %; runways 0 new at all ten airports (`scratch unseen.py`) | ≤ ~50 MSE even at EDDM; closed |
| 2026 labels outside the template | 0 DEP rows with TAXITIME; 344,841 DEP = template; 344,693 ARR all labelled (already used) (`scratch labels2026.py`) | no hidden training data |
| Queue-feature density | `n_push`, `rwy_b30`, `apt_b30`, `arr_b30` quantiles identical 2026 vs 2025 per airport-month | the 2026 file is complete traffic; block features are comparable |
| `doy` date memory | v6 seed-0 booster (30,398 trees, streamed): `doy` = Column_20, **0.90 % of gain, 227,530 splits**; top gains `aobt_eobt` 22.4 %, `proxy` 14.2 %, `de_ars` 8.6 %, `prev_dep_gap` 4.2 % (`scratch doy_gain.py`) | bounded, **OPEN**: the fold cannot see this channel (holdout `doy` is out of training support; the submission's is in support from the wrong year). Test: predict a 20 k-row 2026 sample with `doy` shifted ±7 d; if RMS shift > 20 s the channel is adding variance |

---

## 2. Validation vs board — the fold under-weights 2026's January regime, by 1–2.5 k, not 21 k

Serve-time features only (`data/cache_stand/ranking.parquet` vs `training_2025-0{1,7}`; `scratch drift.py`,
`regime_adjust.py`, `labels2026.py`).

- **January 2026 is a heavier month than January 2025 at six airports.** `sp` means: EDDF 2,262 vs 1,764,
  EDDM 1,835 vs 1,428, EHAM 2,514 vs 2,022 (p99 17,309 vs 10,933), LFPG 2,460 vs 2,176, LSZH 1,970 vs 1,555,
  LTFM 2,227 vs 1,683. `proxy` p90: EDDF 1,584 vs 1,250, EHAM 1,682 vs 1,400, LSZH 1,278 vs 934. July 2026 is
  slightly lighter than July 2025 almost everywhere.
- Airport-days with `proxy` p90 > 2,000 s: **38 in 2026 vs 28 in 2025**; > 1,600: 110 vs 93. The EHAM 2–7 Jan
  event (matched-mean v7 prediction 2,100–2,800 s, 26 % unmatched) has no 2025 analogue (2025's worst EHAM day:
  p90 2,905, 35 unmatched).
- **Regime re-weighting of arm F's own error.** Bin 2025 airport-days by `proxy` p90 relative to the airport's
  median day; arm F's RMSE by bin is 215 / 206 / 226 / 240 / 271 / 299 (bins 0–5). Apply 2026's bin membership:
  pooled **222.56 → 225.20 (+1,163 wMSE)**; airport-specific **228.25 (+2,526)**. By `sp` bins the same
  exercise gives 220.85 / 223.23. Honest band for the hidden matched lane: **49.0–51.3 k**, so the implied
  unmatched lane on the board is **30.4–32.7 k**, not 33 k.
- Unmatched share: 1.534 % (2026) vs 1.545 % (fold), labelled identically (`AOBT_3_flt` null on DEP rows).
  EHAM January 2026 is 4.96 % unmatched vs ≈ 1 % in 2025 — this is the regime the fold never sees, and it
  lands in the **unmatched** lane (§3).

**What the leader's 245 requires under this decomposition** (identity `MSE = 0.98466·m² + 0.01534·u²`):
with our matched at 50 k, their unmatched must be ≈ 10 k; with our unmatched at 31 k, their matched must be
≈ 171 (a 23 % RMSE gain — no measured arm exceeds 0.8 %). The parsimonious reading stands: **their edge is the
unmatched lane, and a few % on matched.**

---

## 3. The structural defect: the unmatched body is congestion-blind — and 2026 January is congested

**What the code does.** `build_submission.NonFillRegressor` (S1's body, shipped since v4):
`NF_NUMERIC = ["sp", "dayoff", "hr"]`, `NF_ENCODED = [ADEP, RUNWAY, stand_pref, airline, AIRCRAFT_OPERATOR]`
(`scripts/build_submission.py:66-67, 257-292`). HGB, 200 iterations, 15 leaves, target winsorised at 3,000 s.
The cells behind it are `airport × sp band × dayoff`. **Neither reads any queue, arrival-taxi-in, runway-rate or
hour-level clock feature.** The matched model's top gains are exactly those (`aobt_eobt` 22 %, `proxy` 14 %,
`prev_dep_gap` 4 %). The all-rows arm U *did* carry the queue block (80 features) and was closed on matched rows
(RESULT 15); nobody scored it on the non-Rome unmatched rows by regime.

**2025 evidence that unmatched taxi-out tracks the airport-hour (labels, ex-LIRF).** Raw Jan/Jul 2025
airport-days: unmatched non-fill mean vs matched mean by matched-mean bin = 757/784, 1,054/998, 1,251/1,190,
1,422/1,420, **2,002/1,630** (ratio 0.93–1.23) (`scratch unm_days.py`).

**Out-of-sample test on the fold's own unmatched rows** (`scratch e3_fold_test.py`; fold A, 4,924 ex-LIRF
unmatched rows from `stratum_fold_v7_preds`, hour from raw `MVT_TIME`, selector = mean arm-F prediction of the
MATCHED rows in the same airport-hour, `m_h`, n_m ≥ 5 — a serve-time quantity):

| airport-hour `m_h` bin | rows | y mean | S1 mean | `m_h` | U_w1 mean | MSE S1 | MSE U | MSE max(S1,`m_h`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ≤ 900 | 1,664 | 832 | 790 | 768 | 816 | 30,991† | 31,185 | 31,160 |
| 900–1,100 (**control**) | 1,660 | 1,016 | 1,033 | 999 | 1,011 | 1,010 | 912 | 1,156 |
| 1,100–1,300 | 674 | 1,166 | 1,121 | 1,176 | 1,130 | 436 | 421 | 491 |
| 1,300–1,500 | 334 | 1,329 | 1,210 | 1,387 | 1,304 | 330 | 295 | 345 |
| 1,500–2,000 | 393 | 1,679 | 1,388 | 1,701 | 1,653 | 1,083 | 918 | 972 |
| **> 2,000** | 130 | **2,395** | **1,332** | **2,305** | 2,121 | 1,432 | 1,014 | 1,064 |

† the two LFPG Family-B rows. Board-MSE units (SSE / 344,841).

- S1's bias is −120 s at 1,300–1,500, −290 s at 1,500–2,000, **−1,063 s above 2,000**; `m_h` is within 4 % of
  the label mean in every bin ≥ 1,100. Arm U, which has the queue block, sits between.
- **Gated rule** `pred = S1 + k·(m_h − S1)` where `m_h > 1,300`, else S1: on 857 rows, **k = 0.75: +509 MSE,
  date-block 95 % [210, 896] over 62 dates, 8/9 airports improve** (EDDF is the exception: `m_h` 1,701 vs y
  1,150 there); k = 1.0: +481 [166, 903]; U-substitution: +619. **Negative control:** the ungated rule *hurts*
  calm hours (1,010 → 1,156), so the gate is load-bearing and the effect is not a blanket upward shift.
- LOMO is not available for this rule (arm F predictions exist only for months 1 and 7), so the evidence is
  one fold, 857 rows, 62 dates. Two independent instruments agree (the rule and arm U).

**2026 exposure (serve-time only; `scratch e3_price.py`).** Non-Rome unmatched rows in airport-hours where the
matched v7 mean exceeds 1,300 s (n_m ≥ 5): **1,242 rows, v7 mean 1,116 vs matched level 1,942; bias² at stake
4,743 MSE at ratio 1.0, 8,958 at 2025's top-bin ratio 1.23**; > 1,500: 887 rows, 4,566; > 2,000: 382 rows
(EHAM 225, LFPG 61, EDDF 26, LSZH 22), 3,766. In 2025 the rule realised ≈ 75–85 % of the bias² stake, so the
expected board value is **≈ +3.5–4 k** if 2026's disrupted-hour unmatched rows behave like 2025's; the downside
if EHAM's storm-day unmatched rows are not ordinary departures (diversions, cancellations mis-recorded) is
bounded at ≈ −2.5 k. **P(mechanism holds in 2026) ≈ 0.75**; 2025's support ends at `m_h` ≈ 2,300 and EHAM
Jan 5 sits at 2,818.

The prior review's E3 (≈ 1–2 k, arrival taxi-in witness) is this lane priced at the day level and without the
fold test; the hour-level exposure is 2–3× larger and the mechanism is now measured out of sample.

---

## 4. The LIRF lane — where the oracle lives, and what is left after R2 and E1

`data/cache_stand/rome_dateslip_preds.parquet` (fold A 397 rows; LOMO 1,488 rows, twelve months); board-MSE
units summed over the record's rows; 2026 has 383 LIRF unmatched rows, so LOMO figures × ≈ 0.257 approximate a
2026 expectation under the 2025 class mix (`scratch r_quality.py`).

| `sp` band (LOMO) | n | fill | R AUC | MSE S1 | MSE R2 | band-rate mix | perfect-fill oracle | sp only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–2 h | 748 | 0.37 | 0.91 | 6,418 | 6,377 | 12,280 | 782 | 29,048 |
| **2–4 h** | 500 | 0.59 | 0.89 | 37,640 | **37,091** | 45,591 | **26,440** | 58,637 |
| 4–6.7 h | 71 | 0.79 | 0.90 | 10,084 | 6,986 | 9,112 | 16 | 11,264 |
| 6.7–11 h | 20 | 1.00 | — | 7,733 | 3,257 | 0 | 0 | 0 |
| 11–15.5 h | 21 | 0.95 | 0.85 | 10,416 | 5,537 | 3,900 | 576 | 3,962 |
| > 15.5 h | 15 | 0.33 (10 slips) | 0.82 | 6,814 | 8,078 | 8,515 | 1,062 | 12,757 |
| **total** | 1,488 | | | **85,477** | **72,488** | | **33,248** | |

- R2 recovers 25 % of the perfect-fill prize in LOMO (30 % on fold A). Scaled to 2026: **≈ 18.6 k expected LIRF
  cost under v7, of which ≈ 10 k is fill-classification error a perfect classifier removes and ≈ 8.5 k is the
  non-fill class itself** (date-slips with `dayoff = 1` at 2–4 h, e.g. LOMO row `sp 11,694, y 87,168, p_R 0.04`
  = 21,219 units alone; long holds). The 6.7–15.5 h bands (≈ 3.3 k of R2's LOMO residual, 0.85 k board) are
  what `v8`/E1 addresses; the 2–4 h band holds **≈ 2.7 k board of recoverable classification variance** and R
  already captures 44 % of it there (45,591 → 37,091 of 45,591 → 26,440).
- **Arrival-leg row witnesses — NOT WORKING** (`scratch arr_leg_witness.py`, 12 months, 1,480 LIRF unmatched
  departures, previous/next arrival on the same stand within 24 h): previous arrival is itself a schedule fill →
  fill 0.508 vs 0.484; next arrival fill → 0.500 vs 0.484; no same-stand arrival within 24 h → 0.745 vs 0.476
  (55 rows); previous arrival NM-unmatched → **17/17 fills** (too small to price, ≈ 4 rows in 2026). Matched
  control shows the expected weak stamping correlation (0.36 vs 0.18). Arrival BLOCK is the only label-adjacent
  field visible in the scored file; it does not name departure fills row by row.
- **Regime check on R's load-bearing feature (airline):** LIRF arrival fill rate by airline correlates 0.75
  with unmatched-departure fill rate (36 airlines, 2025) and **2025 → 2026 arrival propensities correlate 0.60**
  over 43 airlines (RYR 0.13 → 0.22, THY 0.07 → 0.33, VLG 0.13 → 0.08); overall 0.107 → 0.117
  (`scratch lirf_arr_2026.py`). The regime is broadly stable; the airline drift is a reason to shrink R's
  airline effect toward the band rate, not to distrust the lane.
- Serve-time witnesses R already consumes (`rome_fill.py:47-48`: `sp, log_sp, dayoff, hr, tmin, dow, sched_min,
  airline, STAND, stand_c1, ADES, ades_p2, RUNWAY, FLIGHT_RULE`) cover the separations I found (stand prefix 4
  → 5 % fill vs 8/9 → 62–64 %; night 0.74 vs 0.44; designator length ≥ 8 → 0.74 vs 0.45, the last not a
  feature — `FLIGHT_mvt` length/format is a cheap addition, ≤ a few hundred MSE).

---

## 5. Closures that should be reopened (evidence narrower than the conclusion)

From the delegated sweep of all eleven `PREREG_*.md` plus my measurements. Reopen means "the registered verdict
stands for what it tested; the generalisation drawn from it does not."

| closure | what was tested | why reopen | expected value / cost |
|---|---|---|---|
| **RESULT 13 / RESULT 15 as applied to the unmatched lane** ("unmatched lane priced and closed"; "the standard approach loses") | fill-decision oracle only; arm U scored pooled on matched rows | §3: on non-Rome unmatched rows arm U beats S1 by 527 MSE in fold A and by 618 in congested hours; S1's body has no congestion features | **+3.5–4 k board (2026 exposure)** via the gated `m_h` rule (0 heavy hours) or a congestion-aware unmatched body (≈ 0.5 h) |
| **Amendment 7 "native categoricals lose"** | one seed, one fold, one param set; native arm stopped at 2,175 trees under BC-1's biased stopping and **dropped the delta-encodings** | the conclusion is being reused as the prior against CatBoost native (prior review P1) | keep C_delta queued; prior 0–2 k |
| **Amendment 15.2 capacity sweep** | screened at lr 0.05 on 4 months; incumbent `255,40,0.8` ranked **10/18**; `127,20,0.6` top; **confirmation never run**; lr, lambda, max_bin never varied anywhere | shipped learner has never been tuned | 0.5–1.5 k, one 2.5 h confirmation |
| RESULT 8 ("fill lane closed unless AUC > 0.95") | mixture over an all-rows m(X) — double counting | already reopened by G/B/B_all; B_all_cont measured **+0.72 s ≈ 320 MSE**, LIRF-only | small; ship if it clears its own clause |
| RESULTs 6, 11, 11.4, 11.5 (AUC screens) | unweighted AUC on single indicators | 11.5's NOT WORKING became arm F's ESTABLISHED (+1.69 s) when fitted; RESULT 6 used unweighted AUC where the stake is sp²-weighted | reopen only where a fitted arm is cheap (designator format into R) |
| Amendment 5 stand block | on the 400-iteration HGB baseline; LightGBM re-run skipped by the ≥ 10 % rule | a gain measured against a weak baseline is not a closure either | ≤ 500 |
| Amendment 17 fold B | never run | every matched verdict rests on fold A; date-block intervals are 30 % wider than the row intervals quoted | confirmation, not gain |
| Amendment 6 ADS-B | two days | "4 of 10 airports zero on-ground" is a coverage fact; the 2,243 MSE figure is larger than several shipped lanes | do not reopen the ingest; note the inconsistency |
| "Unmatched fold projections do not transfer (0.15×)" | one shipment priced on a fold-A row draw | §4: priced on 2026 composition under 2025 rates, R2 was ≈ 0 before upload; the board's −630 is consistent | price unmatched lanes on 2026 composition, as the prior review says |

Not reopened: the delta target (§1: every alternative loses by 30–140 s and the calibration oracles are < 310
MSE), the matched fill/date-slip class (0 matched rows with |delta| > 20,000), Family B.

---

## 6. The path, priced, and what would change the verdict

Board 81,682. Nothing below is banked; every figure is an expectation with its evidence tier. **Never add
oracles; the lanes below are disjoint row sets except where noted.**

| # | lane | rows touched | expected board MSE | evidence | P(transfers) | heavy hours |
|---|---|---:|---:|---|---:|---:|
| 1 | **E1 band floor (`v8` on disk)** | 13 LIRF unmatched | **+5,600** (two rows carry 5,612) | 2025: 56/56 rows with sp ≥ 24 k have y ≥ sp−60 | 0.85 | 0 |
| 2 | **E3′ congested-hour rule for non-Rome unmatched rows** (`S1 + 0.75·(m_h − S1)` where matched-hour v7 mean > 1,300, n_m ≥ 5; LIRF excluded) | 1,242 | **+3,500 to +4,000** (bias² stake 4,743) | fold A +509 [210, 896], 8/9 airports, control passes; arm U +619 independently | 0.75 | 0 (prereg + splice) |
| 3 | 15.2 confirmation (`127,20,0.6`) + BC-1-clean stopping | matched | +500 to +1,500 | screen rank 10/18; never confirmed | 0.7 | 2.5 |
| 4 | C_delta native CatBoost, blended | matched | 0 to +2,000 | prior from a mis-specified closure | 0.4 | 3 |
| 5 | arm B_all_cont | LIRF matched fills | +300 | measured +0.72 s [0.13, 1.32] | 0.9 | 2.5 (refit) |
| 6 | R recalibration + designator format | LIRF 2–4 h | +300 to +800 | R captures 44 % of band variance | 0.5 | 0.5 |
| 7 | arm FW | matched | ≤ +300 | smoke only | 0.9 | 2 |

- **Expected sum ≈ 11–14 k → ≈ 68–71 k → 261–266.** That is top-5 territory and clears 274 with margin;
  it is **not 245**.
- **What 245 (60,025) would still need after all of it: ≈ 8–11 k more.** The only places it can come from:
  (a) LIRF 2–4 h fills, oracle ≈ 2.7 k, realistic ≤ 1 k more; (b) LIRF's non-fill class, ≈ 8.5 k expected, of
  which the 2–4 h `dayoff = 1` date-slips are ≈ 3 rows in 2026 each worth ≈ 20 k — a coin the leader may
  simply have won; (c) a matched-lane gain of ≈ 8 k = a 9 % RMSE improvement, against a project record of 1.7 s
  per arm and an in-sample calibration ceiling of 310 MSE.
- **The measurement that would change my verdict:** run the all-rows arm U (or a congestion-aware unmatched
  body) **and score it on the 2025 unmatched rows by regime bin, LOMO over twelve months**, not fold A. If the
  disrupted-hour gain replicates at ≥ 1,000 MSE per 100 exposed rows across ≥ 8 months, the 2026 exposure of
  1,242 rows prices at ≥ 8 k, not 4 k, and with E1 that is 14 k from the unmatched lane alone — then 245 needs
  only ≈ 5 k from the matched lane, which lanes 3–5 can plausibly supply. Conversely, if the U/`m_h` gain does
  not replicate outside January/July, lane 2 falls to its downside and 266 is the ceiling.

---

## 7. Hypotheses closed this pass

| # | hypothesis | verdict | reason |
|---|---|---|---|
| H1 | The delta target / floor / L2 objective leaves ≥ 1 k on the table | **NOT WORKING** | §1: floor 22 rows; calibration oracles 119–309; every alternative target loses |
| H2 | The proxy is wrong on an identifiable row class | **NOT WORKING** | `proxy ≤ 0` 496 wMSE, `> 7,200` 103 wMSE |
| H3 | The submission path diverges silently from the fold | **NOT WORKING** | §1 parity table; `doy` channel **OPEN** (0.9 % gain, test named) |
| H4 | The fold mis-measures 2026 by a regime term | **WORKING, small** | +1.2 k to +2.5 k weighted on the matched side (§2) |
| H5 | 2026 contains departure labels outside the template | **NOT WORKING** | 0 rows |
| H6 | 2026 re-numbered stands/runways enough to hurt | **NOT WORKING** | EDDM 4.5 % stands unseen, ≤ ~50 MSE |
| H7 | The unmatched body ignores congestion and 2026 January exposes it | **WORKING** | §3: code, 2025 out-of-sample +509 [210, 896] with control, 2026 stake 4,743 |
| H8 | Arrival-leg records name LIRF departure fills row by row | **NOT WORKING** | 0.508 vs 0.484 |
| H9 | LIRF's 2026 fill regime differs from 2025 | **NOT WORKING** (broadly stable) | arrival propensity corr 0.60, overall 0.107 → 0.117 |
| H10 | A perfect LIRF fill classifier reaches 245 | **NOT WORKING** | ≈ 10 k board; the non-fill class keeps ≈ 8.5 k |
| H11 | 245 is reachable on current evidence | **NOT WORKING** (P ≈ 5–8 %) | §6 arithmetic; measurement that would flip it named |

---

## 8. Reproduction

Scratch (`/private/tmp/claude-501/-Users-saurabhdxt-Projects-prc-challenge/81b461ea-ab3b-47e3-926a-3198c439b0c1/scratchpad/fable2/`):
`fold_target.py` (§1 target/floor/calibration), `drift.py` (§1 v7 sanity, §2 feature drift), `labels2026.py`
(§1 labels, §2 airport-days), `regime_adjust.py` (§2 re-weighting), `doy_gain.py <booster.txt>` (§1 gain
shares), `unseen.py` (§1), `lirf_witness.py` / `arr_leg_witness.py` / `lirf_arr_2026.py` (§4 witnesses and
regime), `r_quality.py` (§4 band table), `unm_days.py` / `e3_price.py` / `e3_fold_test.py` (§3). Run as
`OMP_NUM_THREADS=1 nice -n 19 /opt/homebrew/bin/python3.11 -B <script>` from the repo root. Inputs:
`data/cache_stand/{fold_preds_queue_order, fold_preds_queue_allrows, stratum_fold_v7_preds, rome_dateslip_preds,
ranking, training_2025-*}.parquet`, `data/cache_stand/lgbm_v6_queue_order_seed0.txt`,
`data/raw/{ranking, submitting, training_2025-*}.parquet`, `submissions/merry-quicksand_v{5,6,7,8}.parquet` and
their `.meta.json`. The closure sweep (§5) was a read-only pass over `plans/PREREG_*.md`,
`reports/PRIORITY{0,1,2}_*.md`, `reports/ARMS_QUEUE.log`, `reports/{fresh_path_246_audit,top_path_audit,
lgbm_fold_sweep,rome_body_all}.json`.
