# FM1 — method report (2026-09-28; §5 filled 2026-09-30; §6, §8 and the EOBT clip 2026-10-02; corrections 2026-10-04)

Every number below comes from a registered study (its registration is named where it is used); none is a result
from an unregistered look. Registrations (`plans/…`), reports (`reports/…`) and run logs are the project's development
record, cited by name; the model needs none of them to run or to be rebuilt.

## 1. Problem and row classes

Target: `TAXITIME_SEC_mvt` of each scheduled departure (taxi-out, seconds). Scored rows split by a label-free rule:

- **matched** — the departure has an NM off-block (`AOBT_3_flt`); proxy = MVT − AOBT_3 is available. Served by the base
  model; covered and gated rows by the ADS-B stack; the others by the CatBoost half-blend.
- **unmatched** — no `AOBT_3_flt`; served by the unmatched lane: the shipped v26 body, then (LIRF) the Rome module and ARC,
  then (optional) the storm anchor — §4b.

Every row's `decided_by` names its last value stage (`prc/pipeline/fm1/stages.py`, closed vocabulary).

## 2. Features (`prc/pipeline/fm1/features.py`)

Built per calendar-month chunk through the shipped builders' public functions (never copied):

| block | builder | columns | note |
|---|---|---|---|
| base | `stand_ab.build_features` | BASELINE_FEATS minus `aobt_sec`, `mvt_sec`, `sched_sec` | NSC drops the explicit seconds-of-minute columns; the schedule offset `sp` still carries the take-off's seconds (SCHED is minute-precise) |
| queue | `stand_ab.build_queue` | 12 `q_*` | push-anchored surface congestion; `q_next_arr_onblock_gap` uses the next arrival's on-block time at the same stand, a later event of another flight in the supplied file (other supplied rows may be used: DATA_SOURCES) |
| order | `stand_ab.build_order` | `o_dev_mvt`, `o_dev_flt`, `o_n_line` | per-airport id line |
| weather | `stand_ab.build_weather` | 10 `w_*` | METAR as of AOBT_3 |
| ORL | `orl_build.attach` | 6 `orl_*` | static OSM stand→runway route |
| ADS-B | the R12 tables, by MVT id | the table's features + `coverage`, `adsb_day_present` | the table defines the block (33 for stamp 801f9bcac9a5) |
| encodings | package `EncodingTables` | 24 `te_*` / `de_*` | fitted on training rows only — all of them, early-stopping months included, so each row's encoding holds its own label (LIMITATIONS 17); bit-equal to `stand_ab.infold_encodings` (tested) |

Every block is airport-local (verified: `sched_day`, the order line and the stand chains group by airport), which is what
lets the AT14 gate rebuild one airport-month at a time. Some features count over the whole chunk they are built on
(`sched_day` is the airport's departures in the calendar-month chunk, not in a day), so they assume the scored file
holds whole months, as the 2026 file does.

Forbidden inputs (refused at every boundary, AT2): `TAXITIME_SEC_mvt`, `BLOCK_TIME_UTC_mvt`, `y`, `delta`, `taxi`, `label`,
`target`, any `*_sec`. A scored departure carrying a label is refused (`LabelLeakError`).

## 3. Learners (`prc/pipeline/fm1/learners.py`)

One protocol: `fit(X, y, fit_mask, es_mask, seeds, feature_names)`, `predict`, `save`, `load_learner`, `describe`.
Early stopping on the ES rows at the first seed; one refit per seed on fit ∪ ES rows at
n_ref = max(50, ⌊best · n_all / n_fit⌋) (lgbm_submit's rule); prediction = the mean over seeds.

- LightGBM, in-process: `deterministic=True`, `force_col_wise=True`, fixed `num_threads` and seeds (AT3 verified bit-equal).
- CatBoost, only in `~/.venvs/prc-catboost` via `catboost_worker.py` (BC-5); the main process never imports it (tested).
- HGB (sklearn): the v26 unmatched body's estimators are the SHIPPED ones (build_submission's S1 cells + NonFillRegressor,
  unm_congestion's congestion regressor, mxj_build.fit_body), called through their public functions; the provisional `hgb`
  body (FM1's own HGBLearner) remains a non-default alternative.
- XGBoost is CUT (owner decision 5); a generic MLP learner is selection arm A6 (not built).

Difference from lgbm_submit stated: FM1's early stopping evaluates every ES row; lgbm_submit's ES evaluation had a guard-row
filter (`es_guard_rows`). Their best iterations can differ; the day-4 study fits both sides with FM1.

## 4. Lanes and modules

value formulas (tested equal to the shipped ones, `components.py`):
base `rint(max(proxy − F, 1))`; stack `rint(max(proxy − (F + resid), 1))`; blend `rint(w·max(proxy − C, 1) + (1 − w)·partner)`.

Routing difference from v36 (a design choice for the day-4 study, not a replay): FM1 blends every matched row the stack does
not serve; v34/v36 blended only rows outside the v13-era ADS-B table (N rows without coverage kept the base value).

ADS-B support gate: `manifest` (the stack's stored ADN.7 gate, as v36 served) or `coverage_ratio`; a rule can narrow the
stack's airports, never widen them. The label-free ESS rule planned for 2025 LOMO was not built; the coverage-ratio
rule was tested on the board instead (v52, −10.6 MSE vs v50, not kept; §8), so the final model serves the stored gate.

Calibration (MTX) runs LAST, structurally (plan check) and at run time (the ledger refuses any value write after it) — the
v43 EGLL-7 defect class cannot be expressed.

EOBT clip (`prc/pipeline/fm1/eobt_clip.py`, `modules.eobt_clip`, off by default): a serve-time repair of one raw input,
run before the feature path (stage `eobt_clip`). In an (airport, UTC month) cell of the served rows where the share of
eligible departures with EOBT_1 later than LOBT is at least 0.15 (on at least 1,000 rows; fewer is refused), every eligible
departure takes EOBT_1 := min(EOBT_1, LOBT); a batch of the scored file is decided on the scored file's whole cell. Fit never
applies it. On the 2026 scored file it fires at LFPG July only (35.1 %; LFPG January 6.1 % and every 2025 month 3.8 – 7.1 %
do not). Why and what it is worth: LIMITATIONS 9 / 9b. The provenance marks each clipped row (`eobt_clipped`) and the
predict report lists every cell's share and decision. **It is not part of the final configuration:** served on the board as
v54 (v51 + clip) it scored −214 MSE worse than v51 (§8), so it stays off.

## 4b. The unmatched lane (`prc/pipeline/fm1/unmatched.py`; PREREG Amendment A3)

The shipped v36 unmatched chain, ported through the shipped modules' public functions (`shipped.unmatched_deps()` =
`urp_replay.load_deps()`, plus urp_replay's public composition helpers). Nothing is re-implemented.

| stage | rows | what it does | shipped step |
|---|---|---|---|
| `unmatched_body` | every unmatched departure | S1's hierarchical cells give the fill rate p (LIRF: its logistic) and the tail cells; the non-fill term is the MJS body (congestion design with the serve file's airport-hour witness; 1-in-20 matched extra rows at 25 % mass; HGB, patience 50, mean of the run seeds); value = rint(max(p·sp + (1−p)·nf_hybrid(cells, body), 1)) | v9 → v24 → v25 → v26 |
| `rome` | LIRF | R2 fill classifier (LightGBM) → the mixture with S1's non-fill term → 24 h date-slip expectation → rint → E1 floor (24,000 ≤ sp < 86,400) → RLD local-day rule | v7, v10, v12 |
| `arc` | `modules.arc.airports` (LIRF) | the departure's own flight's ARR row (callsign-matched at its destination) carries LOBT / AOBT_3: `ord` → max(rint(MVT − AOBT_3), 1), `fill` → max(rint(sp), 1) | v17 |
| `rome_e2` | LIRF rows still at their Rome value, 3,600 ≤ sp < 24,000 | the frozen stand-inbound logit replaces p; a UTC day whose move energy per moved row exceeds the p99 threshold reverts | v31 |
| `rome_e2w` | the rows E2 moved | E2's form with beta_W, refitted at FIT time by stake-weighted Brier on the training months' LIRF band rows (p_R and the non-fill term nested leave-one-month-out); its own day check | v33 |
| `storm` | rows the body still decides | rint(v + max(A − 300, 0)), A = the airport-day anomaly of the scored rows' matched MVT − AOBT_3 | v16 (inside v24–v26) |

ARN (v21) is dropped; its 8 scored rows keep their Rome / E2 values (measured: no other row moves). The shipped S1 cells and
R2 classifier fit and predict in one call, so the artefact stores the lane's training frames (`unmatched_lane/`) and predict
re-runs every fit, seeded (bit-reproducible, tested). Set-level rules (witness, anomaly, ARC's ARR join, E2's day checks and
its stand-inbound context read from `data.raw_dir`) mean the whole scored file must be served together.

## 5. Selection study (2026-09-30; `plans/PREREG_swap_selection_2026_09_30.md`, `plans/PREREG_weather_S_check_2026_09_30.md`)

**Why the strict swap is the primary selector.** Our 2025 Jan / Jul holdout skill matches or beats teams that score 11–14 RMSE
better than us on the 2026 board, so the gap is in 2025 → 2026 transfer, not in-season fit (`reports/MSE_LEDGER.md`,
2026-09-30 08:20). Two changes priced on 2025 this campaign did far worse on the board than priced (HPX-S −703 vs +120;
MOVE −1,121 vs −398). The swap protocol (`prc/pipeline/fm1/protocols.py`: X1to7 = fit January only, test July; X7to1 = fit
July only, test January; early stopping on days ≥ 24 of the training month) trains and tests on different seasons, so a
component that only fits its training season loses there.

**Setting.** Screen tier (learning rate 0.05, seed 0). Fixed in every arm, as registered amendments, before any fit: the
ADS-B stack off (SWAP.1: it was trained on January + July 2025, so a swap fold would test on its training months) and E2W
off (SWAP.2: its nested leave-one-month-out needs two training months). Statistic: paired gain of the arm over A0 from
`python -m prc compare` (day-block bootstrap, 2,000 resamples, seed 20260928; units bmse = SSE / 344,841; gain > 0 = the arm
is better). An arm counts as a candidate only if the CI's lower bound is above 0.

| arm | change | gain vs A0 | 95 % CI | reading |
|---|---|---|---|---|
| A1 | ARC off | −1,028.1 | [−2,179.8, −150.2] | ARC transfers → kept |
| A2 | Rome module off | −3,089.3 | [−19,650.6, +10,586.9] | ambiguous → kept; ≤ 3 rows carry its net (ex-top-3 rows: +8,756.8), a lead only |
| A3 | E2 off | −2,970.5 | [−4,349.2, −1,817.0] | kept |
| A4 | queue features off | −824.6 | [−1,062.7, −595.6] | kept |
| A5 | order features off | −1,499.7 | [−1,916.9, −1,137.9] | kept |
| A6 | weather features off | **+493.6** | **[+131.8, +828.9]** | the only candidate; same-season check below |
| A7 | route-length feature off | −947.1 | [−1,203.9, −702.3] | kept |
| A8 | ADS-B features off | −9,960.1 | [−11,174.4, −8,774.5] | the largest single contribution → kept |
| A9 | CatBoost as the matched learner | −1,759.6 | [−2,161.8, −1,405.7] | LightGBM base kept |
| A10 | unmatched `hgb` body | not run | — | refused at plan time (the Rome module needs the v26 body); not re-specified, per the registration |

Registered prediction vs outcome: P(at least one candidate) 0.35 → one candidate. The predicted candidates (E2, ARC, route
length) were all wrong: they transfer. ADS-B off and Rome off were predicted worse, as measured.

**Weather.** METAR weather is the most season-specific input (freezing in January, convective weather in July), and the
swap makes it extrapolate across seasons, which the final setting does not (it trains on all twelve 2025 months and scores
the same seasons of 2026). The swap result is therefore checked under protocol S (leave-one-month-out on January and July,
training on the other ten months), registered before either run: drop weather if the S gain's CI is above 0 or straddles 0;
keep it only if weather clearly helps within season (CI below 0). LDI reads the weather block, so dropping weather also
turns LDI off (amendment registered before the second run). **Result: keep weather.** Weather off vs on under protocol S:
−269.0 bmse [−423.9, −120.5] (excluding LIRF −175.2 [−300.5, −53.4]; excluding the 3 largest rows −241.2 [−386.7, −102.4]);
the largest losses without weather are at LIRF, LFPG and LTFM. The two protocols disagree in sign, as the season argument
predicts: the swap's weather gain is an extrapolation artefact, and the final model keeps weather and LDI.

**Not testable in these folds:** the ADS-B stack, E2W, MTX (a 2026 calibration), the storm anchor.

## 6. Validation (2025 labels only; filled 2026-10-02)

All numbers are RMSE in seconds on held-out 2025 months, screen tier (learning rate 0.05, one seed), **ADS-B stack off**
(it was trained on January + July 2025 and cannot be validated on them) and, in the swap folds, E2W off. They are a floor
for a reduced configuration, not the final model's score (a reference point, not a bound).

| protocol | fold | held out | n | RMSE | worst airports (RMSE) |
|---|---|---|---|---|---|
| S (leave one month out; train on the other ten) | F1 | January | 153,658 | 327.07 | LFPG 799.2, LIRF 364.5, LTFM 220.6 |
| S | F2 | July | 190,678 | 266.21 | LIRF 614.6, EGLL 278.0, LTFM 274.5 |
| swap (train one month, test the other) | X7to1 | January | 153,658 | 351.83 | LFPG 823.6, LIRF 411.6 |
| swap | X1to7 | July | 190,678 | 376.94 | LIRF 972.1, LFPG 414.9, EGLL 329.6 |

Reading: the January figures are dominated by LFPG, where two departures with a stale recorded off-block time carry most
of that airport's squared error (the comparison in §5 is paired, so they cancel there); Rome (LIRF) is the largest
structural error in both months — its unmatched departures have no flight-plan record and no ADS-B. Best airports sit
near 100–160 s (EDDM 98.8, LEBL 102.7 in January; EHAM 118.0 in July).

**Determinism at scale.** The final configuration on a three-month slice, fitted twice and served on the full scored file:
344,841 rows × 30 columns, 0 different cells (2026-10-01).

**Consistency with the previous best file (label-free).** On the 233,132 matched rows of the fifteen airport-months where
the final model and v46 share a design, the row-level difference after removing each airport-month's constant is 30.5 s RMS
(registered limit 81.6 s = twice the measured refit-to-refit difference). The constants themselves are the calibration
(MTX) the forecaster omits by design.

**2025 → 2026 drift that validation on 2025 cannot see.** Share of matched departures with ground ADS-B coverage, stack
training months vs the 2026 scored file: EDDF 0.909 → 0.623, EDDM 0.837 → 0.970, EGLL 0.384 → 0.902, EHAM 0.972 → 0.978,
LEBL 0.957 → 0.946, LEMD 0.679 → 0.795, LFPG 0.077 → 0.141, LIRF 0.405 → 0.299, LSZH 0.939 → 0.959, LTFM ≈ 0. The stack
serves EDDF, EDDM, EHAM, LEBL, LIRF and LSZH (its stored gate); `adsb.support_gate.rule=coverage_ratio` is the label-free
rule that would switch it off where the share moved outside [0.7, 1.43] × training (EDDF).

**Not validated here:** the ADS-B stack, E2W, MTX and the storm anchor (no clean 2025 fold exists for them; their evidence
is the board history in §8); the full-tier learners (validated only through the determinism slice and the consistency
check above); the CatBoost half reached its round cap in the final fit (REPRODUCE.md). The unmatched lane's frozen
constants (E2, E1, date-slip, local-day rule, ARC, the day-check thresholds) and LDI's shifts were chosen on 2025 data
that includes the held-out months, so the fold figures overstate them (LIMITATIONS 17); the stored stack is served on
a differently made F than it was trained on (LIMITATIONS 16).

## 7. Ablations and NOT WORKING results of the final three weeks

The swap-study arms (§5) are the ablations of the final configuration. Below are the other levers tried in the last three
weeks, each registered before it was measured, with its verdict as recorded. Units as recorded: **s** = paired RMSE gain on
a 2025 holdout; **bmse** = SSE / 344,841 on 2025 holdout rows; **board MSE** = change on the 2026 public board. Positive =
the lever helped. Sources: `reports/MSE_LEDGER.md` and each registration's hash log (`reports/<arm>_prereg_hashes.log`).

| date | arm | what was tried | headline | verdict |
|---|---|---|---|---|
| 09-17 | TRC | correct the target residual (WIN / RES heads) | −2.661 s / −1,100 board (WIN); −0.698 s / −288 (RES) | NOT WORKING |
| 09-18 | TNS | tree ↔ neural models on the residual | +0.007 s / +3 board at best; −0.073 s / −30 | NOT WORKING (function class closed in three directions) |
| 09-18 | OPX | 96 ordered peer-clock columns added to the ADS-B stage | composed −0.400 s [−0.501, −0.300]; a shuffled null −0.748 s | NOT WORKING (capacity, not content) |
| 09-18 | GHJ | adsb.fi `globe_history` heatmap archive as a source | stopped at the distribution-shift gate, no score | NOT WORKING |
| 09-19 | MAH | extra LOBT- / EOBT-anchored residual heads | +86.3 board (40–55 % of the +200 bar) | NOT WORKING |
| 09-19 | v22 (BLS) | production blend splice on the CatBoost rows | −133.79 board MSE vs v21 (registered +51.53) | NOT WORKING (wrong sign) |
| 09-21 | AGF / v27 | ACX-G bag as an ADS-B-stage ensemble member | forward +69.69 [17.8, 120.4]; board −153.50 vs v26 | NOT WORKING on the board |
| 09-21 | MXK | unmatched body capacity (leaves 15 → 31) | forward +18.96 [−0.68, 42.49] | NOT WORKING (over-fitting shape) |
| 09-23 | CSF.2 | Rome callsign-frequency identifier | AUC 0.914 < 0.95 bar; −2,020.2 board [−5,645.4, +796.2] | NOT WORKING |
| 09-23 | RCR2 | direct target y instead of the delta target | January −1.73 s [−3.15, −0.35] | NOT WORKING |
| 09-23 | OMW | Open-Meteo weather instead of / beside METAR | July −0.036 s [−0.695, +0.691]; January +0.110 [−0.399, +0.620] | NOT WORKING (gate G1) |
| 09-23 | DFC.3 | destination departure-count columns | +11.84 bmse [−7.34, +31.24] (MDE 27.6) | NOT WORKING (underpowered) |
| 09-24 | CLN | a clean-room LightGBM on the raw columns vs the pipeline | matched −13,560.5 [−15,154.6, −12,119.9] | FALSE: the architecture is not the cap |
| 09-25 | UMR | rebuilt unmatched lane | −18,618.8 [−27,816, −10,797] | NOT WORKING |
| 09-25 | ODX | off-block time detected from ADS-B | −779,498 [−813,210, −745,285] | NOT WORKING |
| 09-26 | BLX | blind airports (LFPG / LTFM) residual model | forward +4.98 [−3.9, +13.5]; swap +13.83 [−0.6, +27.9] | NOT WORKING (first sight of the LFPG EOBT drift) |
| 09-29 | RC-1 | best-of-8-clocks oracle on stale off-block rows | −47.2 bmse net of a permutation null (raw 904 < 1,000 bar) | NOT WORKING |
| 09-30 | HPX-S (v48) | Rome delayed-departure rule | −703.0 board MSE vs v46 (registered +120) | NOT WORKING |
| 09-30 | MOVE (v49) | movement-only ADS-B base | −1,120.7 board MSE vs v43 (null −398.5) | NOT WORKING |
| 10-01 | WSC | weather off, same-season protocol S | −269.0 bmse [−423.9, −120.5] | weather kept (§5) |
| 10-02 | DRF | label-free drift audit of the 133 inputs | LFPG July EOBT field 6.6 % → 35.1 % revised | TRUE (LIMITATIONS 9) |
| 10-02 | EPX | price of that change on a held-out 2025 July | damage 265.6 [196.2, 339.0]; clip net ≈ +195 … +290 | NOT established (LIMITATIONS 9b) |
| 10-02 | CLP (v54) | the EOBT clip served at LFPG July | −214.0 board MSE vs v51 (registered +100 … +300) | FALSE: not in the final configuration |
| 10-02 | ACR | training with ADS-B randomly hidden (coverage-regime repair) | the 2026 coverage loss costs 862.7 bmse [671.7, 1124.8]; the repair recovers 25.7 [−120.5, 163.9] | FALSE: information, not calibration |
| 10-03 | AFI (v55) | adsb.fi July-2026 capture merged into the serve table (no refit) | +609.6 board MSE vs v51 (registered +150 … +600) | TRUE: in the scored-period build |
| 10-03 | AFB (v56) | adsb.fi only where adsb.lol had no track | −205.8 vs v55 | FALSE |

Reading: almost every lever that looked good on a 2025 holdout either failed its own registered gate or lost on the 2026
board (AGF, v22, HPX-S, MOVE). The final model is what survived this, not the best of many tries.

## 8. Board history as external evidence (public leaderboard, RMSE; lower is better)

The public board scores the same 344,841 departures of January and July 2026 for every upload. A step was kept only if
the board improved when it was added; the list is therefore selected on the public set and is NOT an unbiased estimate
for unseen data (the final-model audit of 2026-09-28 put about 28 % of the cumulative gain in board-specific patches).

| file | what changed | board | change in MSE |
|---|---|---|---|
| v12 | Rome local-day rule (pipeline with the Rome module) | 266.3738 | — |
| v16 | storm anchor | 260.366 | +2,432 better than v14 |
| v17 | ARC at LIRF (42 rows) | 259.4012 | +501 |
| v31 | Rome E2 | 254.8221 | +1,057 |
| v35 | 12-month ADS-B refit (R12, screen) | 252.0706 | +1,015 vs v34 |
| v36 | full R12 at five airports | 251.3146 | +381 vs v35 |
| v40 | R12 at EGLL / LIRF / LEMD | 251.1991 | +58 |
| v41 | LDI (LTFM January de-icing) | 251.0977 | +51 |
| v46 | LightGBM + CatBoost stack base at the R12 cells (with MTX) | **250.7314** | +38.5 vs v43 — best file until v55 |
| v48 | Rome delayed-departure rule (HPX-S) | 252.1294 | −703 (worse); not kept |
| v49 | movement-only ADS-B base (MOVE) | 253.0326 | −1,121 vs v43 (worse); not kept |
| **v50** | **FM1, the final model (forecaster, full fit)** | 251.4767 | −374 vs v46 (registered band met: +0.745 RMSE) |
| v51 | v50 + MTX (scored-period build) | **251.1335** | +172.5 vs v50 — best FM1 file |
| v52 | v50 + coverage-ratio gate (stack off at EDDF) | 251.4978 | −10.6 vs v50; not kept |
| v53 | v50 without the CatBoost half-blend | 251.8068 | −166.1 vs v50; not kept |
| v54 | v51 + the EOBT clip at LFPG July | 251.5593 | −214.0 vs v51; not kept |
| **v55** | **v51 + adsb.fi July-2026 serve table (adsb.lol ∪ adsb.fi, LTFM excluded)** | **249.9168** | **+609.6 vs v51; +407.8 vs v46 — best file** |
| v56 | as v55, adsb.fi only where adsb.lol had no track | 250.3282 | −205.8 vs v55; not kept |

Rows read from the project's board records (v50–v53 from the scoring service's result files, 2026-10-02); v42's MTX gain
(+124) and the individual Rome steps v7 / v10 are quoted in the ledger and not repeated here. v51–v53 are board-selected
whole-model variants of v50: each differs in one documented knob and changes at least 10,000 rows. The documented forecaster is v50; v51 adds MTX, which reads the target period's published
monthly aggregates and is disclosed as such (LIMITATIONS 8). Change in MSE = difference of the squared board RMSEs.
