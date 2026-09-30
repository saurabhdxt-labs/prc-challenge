# FM1 — method report (skeleton, 2026-09-28)

Sections marked **TODO (day N)** are filled by the registered studies; nothing here is a result until then.

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
| base | `stand_ab.build_features` | BASELINE_FEATS minus `aobt_sec`, `mvt_sec`, `sched_sec` | NSC: the label-seconds leak is dropped |
| queue | `stand_ab.build_queue` | 12 `q_*` | push-anchored surface congestion |
| order | `stand_ab.build_order` | `o_dev_mvt`, `o_dev_flt`, `o_n_line` | per-airport id line |
| weather | `stand_ab.build_weather` | 10 `w_*` | METAR as of AOBT_3 |
| ORL | `orl_build.attach` | 6 `orl_*` | static OSM stand→runway route |
| ADS-B | the R12 tables, by MVT id | the table's features + `coverage`, `adsb_day_present` | the table defines the block (33 for stamp 801f9bcac9a5) |
| encodings | package `EncodingTables` | 24 `te_*` / `de_*` | fitted on training rows only; bit-equal to `stand_ab.infold_encodings` (tested) |

Every block is airport-local (verified: `sched_day`, the order line and the stand chains group by airport), which is what
lets the AT14 gate rebuild one airport-month at a time.

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
stack's airports, never widen them. The label-free ESS rule set on 2025 LOMO is TODO (days 4–6).

Calibration (MTX) runs LAST, structurally (plan check) and at run time (the ledger refuses any value write after it) — the
v43 EGLL-7 defect class cannot be expressed.

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

## 5. Selection study — TODO (days 4–6)

Primary folds LOMO-Jan and LOMO-Jul; secondary strict Jan↔Jul swap; forward F3/F4 tertiary (Amendment A2). Arms A0–A7 + H1
(FMA §4.3); the ADS-B stack is off in these folds unless retrained per fold (leakage guard: it was trained on Jan + Jul).

## 6. Validation — TODO (day 7)

LOMO / forward / swap / 2026 drift (domain AUC, ESS, coverage, schedule-field rates) / standalone component numbers.

## 7. Ablations, including every NOT WORKING result — TODO (day 9)

To be filled from the ledger with numbers: CLN, UMR, TNS, GHJ, OPX, DSG, v22, AGF/v27, MXK, R13, UCE, APT, TRC, RCR2,
CSF.2, OMW, DFC, BLX, ODX, ... (memory index and `reports/MSE_LEDGER.md`).

## 8. Board history as external evidence — TODO (day 9)

k per component (FMA §1.3).
