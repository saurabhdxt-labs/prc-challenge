# FM1 config reference

GENERATED from `prc/pipeline/fm1/knobs.py` by `python -m prc config-reference --write`. Do not edit by hand:
`tests/fm1/test_fm1_refdoc.py` fails when this file differs from the schema.

One config, `configs/fm1.yaml`, states every knob at its default (the documented forecaster). A run that differs
says so with `--set path=value` (validated like the file and recorded in the resolved-config hash), for example the
scored-period build: `--set run.purpose=scored_period`.

Status: `implemented` = runs in FM1; `provisional` = runs, but its reproduction of the shipped component is not verified yet; `not_ported` = declared; `fit`/`predict` refuse while it is switched on (ComponentNotPortedError); `schema_only` = validated and recorded, not read by any stage yet.

## `run`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `run.purpose` | choice | `forecaster` | `forecaster` / `scored_period` | implemented | What the run is for. `forecaster` = the documented model (no target-period information); `scored_period` = the board build for a scored period, which may read that period's published aggregates (turns MTX on when calibration.mtx.enabled is auto). | `plans/PREREG_fm1_2026_09_28.md §2` |
| `run.team` | str | `merry-quicksand` | non-empty string | implemented | Team name in the submission file name `<team>_v<N>.parquet` (the organiser's convention). | `scripts/build_submission.py` |
| `run.seeds` | int_list | `[0, 1, 2]` | non-empty list of integers in [0, 2.14748e+09], no repeats | implemented | Seeds of every stochastic learner; the matched and unmatched predictions are the mean over seeds. The early-stopping run always uses the first seed. | `scripts/lgbm_submit.py:179` |
| `run.threads` | int | `4` | [1, 64] | implemented | Thread count of every learner (LightGBM num_threads, CatBoost thread_count). Fixed so a fit is bit-reproducible on this machine (AT3); changing it can change LightGBM's floating-point summation order. | `scripts/lgbm_fold.py:32` |
| `run.artifacts_dir` | path | `artifacts/fm1` | path (relative to the repo root, or absolute); never under `submissions/` | implemented | Where fitted artefacts live, one directory per resolved-config hash; an existing one is never overwritten. An output location: it is NOT part of the config hash (it cannot change a prediction). | `plans/PREREG_fm1_2026_09_28.md §3` |

## `data`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `data.raw_dir` | path | `data/raw` | path (relative to the repo root, or absolute); never under `submissions/` | implemented | Directory of the organiser's raw monthly parquet files (training months and the scored file). | `data/raw/FETCH_MANIFEST.json` |
| `data.training_glob` | str | `training_2025-*.parquet` | non-empty string | implemented | Glob (inside data.raw_dir) of the labelled training months. | `configs/pipeline.yaml` |
| `data.train_months` | int_list | `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]` | non-empty list of integers in [1, 12], no repeats | implemented | Calendar months (of the training files) that fit the models; a validation protocol replaces this per fold. | `reports/final_model/AUDIT_AND_PLAN_2026_09_28.md §4.3` |
| `data.scored_file` | path | `data/raw/ranking.parquet` | path (relative to the repo root, or absolute); never under `submissions/` | implemented | The raw file whose departures are scored (it holds DEP and ARR rows; ARR rows feed the stand/queue features). | `configs/pipeline.yaml` |
| `data.template_file` | path | `data/raw/submitting.parquet` | path (relative to the repo root, or absolute); never under `submissions/` | implemented | The id template the scorer expects; build-submission writes exactly its ids, in its order. | `configs/pipeline.yaml` |
| `data.weather_dir` | path | `data/weather` | path (relative to the repo root, or absolute); never under `submissions/` | implemented | Frozen METAR archive (IEM) read through prc.weather; the weather block's as-of join reads it. | `scripts/stand_ab.py:1076` |
| `data.dayoff_tz` | choice | `UTC` | `UTC` / `local` | implemented | Time zone of the `dayoff` feature (take-off date minus scheduled date, clipped to [0, 1]) of the unmatched body. UTC reproduces every shipped file; `local` uses airports.tz. | `reports/final_model/AUDIT_AND_PLAN_2026_09_28.md §2.2 G11` |

## `airports`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `airports.codes` | str_list | `[EDDF, EDDM, EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM]` | non-empty list of ICAO codes, no repeats | implemented | The airport registry. A scored row at an airport outside it is refused (input contract). | `reports/final_model/AUDIT_AND_PLAN_2026_09_28.md` |
| `airports.tz` | tz_map | `EDDF: Europe/Berlin, EDDM: Europe/Berlin, EGLL: Europe/London, EHAM: Europe/Amsterdam, LEBL: Europe/Madrid, LEMD: Europe/Madrid, LFPG: Europe/Paris, LIRF: Europe/Rome, LSZH: Europe/Zurich, LTFM: Europe/Istanbul` | airport -> IANA zone, exactly airports.codes | implemented | IANA time zone per airport; must cover exactly airports.codes. Read by dayoff_tz=local and the MTX local month. | `scripts/analysis/mtx_source.py:24` |

## `lanes`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `lanes.matched` | bool | `true` | true / false | implemented | Serve matched departures (AOBT_3 present): base model -> ADS-B stack -> CatBoost half-blend. | `reports/final_model/AUDIT_AND_PLAN_2026_09_28.md §4.2` |
| `lanes.unmatched` | bool | `true` | true / false | implemented | Serve unmatched departures (AOBT_3 absent) with the unmatched body. | `reports/final_model/AUDIT_AND_PLAN_2026_09_28.md §4.2` |

## `features`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `features.queue` | bool | `true` | true / false | implemented | Push-anchored surface-congestion block (12 q_* columns, stand_ab.build_queue). | `scripts/stand_ab.py:106` |
| `features.order` | bool | `true` | true / false | implemented | Movement-order block (o_dev_mvt, o_dev_flt, o_n_line; stand_ab.build_order). | `scripts/stand_ab.py:967` |
| `features.weather` | bool | `true` | true / false | implemented | METAR block (10 w_* columns, as-of the row's AOBT_3; stand_ab.build_weather). | `scripts/stand_ab.py:1096` |
| `features.orl_route_length` | bool | `true` | true / false | implemented | OSM stand->runway route-length block (6 orl_* columns; orl_build.attach, a static lookup). | `plans/PREREG_orl_osm_route_length_2026_09_23.md` |
| `features.orl_routes_file` | path | `data/osm/orl_routes.parquet` | path (relative to the repo root, or absolute); never under `submissions/` | implemented | The ORL route table built from OpenStreetMap (ODbL) by scripts/analysis/orl_build.py build. | `data/osm/manifest.json` |
| `features.adsb` | bool | `true` | true / false | implemented | 12-month ADS-B block (33 features + coverage + adsb_day_present) joined by MVT id from the R12 tables; NaN where a row has no table row. | `plans/PREREG_v35_r12_2026_09_26.md` |

## `adsb`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `adsb.sources.adsb_lol` | bool | `true` | true / false | implemented | adsb.lol globe_history (ODbL 1.0), the only source the R12 tables hold; required by features.adsb. | `plans/PREREG_adsb_nn_2026_09_10.md:5` |
| `adsb.sources.adsb_fi` | bool | `false` | true / false | implemented | Declares that the ADS-B tables hold adsb.fi points (the owner's July-2026 opendata.adsb.fi capture, merged with adsb.lol by rule AUN.1; non-commercial terms, never redistributed, cannot be re-fetched by others). Checked against the tables' report both ways at fit and predict. Off in the documented forecaster; a scored-period build may turn it on with tables built from the union (docs/fm1/REPRODUCE.md). | `plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md` |
| `adsb.table_dir` | path | `data/r12/tables_801f9bcac9a5` | path (relative to the repo root, or absolute); never under `submissions/` | implemented | Directory with the ADS-B tables train.parquet (2025) and serve.parquet (scored rows), keyed by MVT_ID_mvt. | `data/r12/tables_801f9bcac9a5/report.json` |
| `adsb.serve_missing_max_share` | float | `0.001` | [0, 1] | implemented | At predict time, the largest share of scored matched rows that may lack an ADS-B table row; such a row is served as 'no ADS-B data' (coverage 0) and counted. Above it the run refuses. At fit a missing row is always refused. | `reports/final_model/FM1_PACKAGE_NOTES_2026_09_28.md §N1` |
| `adsb.min_coverage` | int | `2` | [1, 3] | implemented | A row is ADS-B-covered iff its coverage code >= this (2 = seen on the ground). The stack serves covered rows only. | `prc/pipeline/adsb_stage.py:32` |
| `adsb.support_gate.rule` | choice | `manifest` | `manifest` / `coverage_ratio` | implemented | Which airports the ADS-B stack may serve. `manifest` = the gate stored with the stack models (ADN.7 rule, as v36 served); `coverage_ratio` = allowed iff the scored rows' covered share is within a factor `threshold` of the training share. The label-free ESS rule is a day 4-6 item (FMA §4.2 item 2). | `data/r12/stack_801f9bcac9a5_full/models_r12/MANIFEST.json` |
| `adsb.support_gate.threshold` | float | `0.7` | [0.01, 1] | implemented | coverage_ratio rule: allowed iff threshold <= covered_share_serve / covered_share_train <= 1 / threshold. | `data/r12/stack_801f9bcac9a5_full/models_r12/MANIFEST.json` |
| `adsb.train_dropout.flight` | float | `0.0` | [0, 0.9] | implemented | Training-time ADS-B dropout, flight level: a seeded random share of the JOINED matched training rows is fitted as an unheard flight (joined 0, coverage 1, flight-level ADS-B columns NaN; the airport-level counts kept), so the model learns to predict without the flight's own track at every airport and season. Fit only; predict never hides ADS-B. Why: which flights have ADS-B differs by airport-month between 2025 and 2026 (LIRF July 93 % -> 0.2 %). | `plans/PREREG_acr_adsb_coverage_regime_2026_10_02.md` |
| `adsb.train_dropout.day` | float | `0.0` | [0, 0.9] | implemented | Training-time ADS-B dropout, network level: every matched training row of a seeded random share of (airport, UTC day) groups is fitted as a network blackout (the flight-level state plus ground_seen_mvt_m10 = 0 and moving_mvt_m10 = NaN), the state of LIRF in July 2026. Fit only. | `plans/PREREG_acr_adsb_coverage_regime_2026_10_02.md` |
| `adsb.stack.enabled` | bool | `true` | true / false | implemented | Serve the ADS-B stack residual (LightGBM + MLP; MLP in its worker, BC-5) on covered, gated matched rows: value = rint(max(proxy - (F + resid), 1)). | `plans/PREREG_adsb_ship_v11_2026_09_11.md` |
| `adsb.stack.train_months` | int_list | `[1, 7]` | non-empty list of integers in [1, 12], no repeats | implemented | 2025 months whose labels trained the stored stack (r12_stack: the val run's Jan + Jul records). A validation fold that tests one of them with the stack on is refused (leakage). | `scripts/analysis/r12_stack.py oof` |
| `adsb.stack.models_dir` | path | `data/r12/stack_801f9bcac9a5_full/models_r12` | path (relative to the repo root, or absolute); never under `submissions/` | implemented | The stack's stored models (G_*.txt LightGBM, N_*.pt MLP, MANIFEST.json with feats and gate). | `data/r12/stack_801f9bcac9a5_full/report_r12.json` |
| `adsb.stack.f_model` | choice | `lightgbm` | `lightgbm` / `lgbcat` | implemented | The F the ADS-B stack corrects on its rows: `lightgbm` = the base model's F (v36's stack, models_r12); `lgbcat` = 0.5 F + 0.5 C, C = the matched CatBoost's delta (v46: R13's blend; its stack r13_stack_full/models_lgbcat was trained on the DWG tables 51348d3e1cb3). The stack must be the one trained on that base (r12_stack writes models_<arm>); fit fits the CatBoost whenever the blend needs it. | `plans/PREREG_blf_blend_serve_2026_09_28.md RESULT (v46 = 250.7314, +38.5 board MSE vs v43)` |

## `matched`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `matched.learner` | choice | `lightgbm` | `lightgbm` / `catboost` | implemented | Matched base learner: `lightgbm` (in-process) or `catboost` (~/.venvs/prc-catboost worker). xgboost is CUT. | `plans/PREREG_fm1_2026_09_28.md §2` |
| `matched.target` | choice | `delta` | `delta` / `y` | implemented | `delta` = proxy - taxi, recovered as max(proxy - F, 1) (every shipped matched model); `y` = taxi directly (selection arm A4; lost three times before). | `scripts/lgbm_submit.py:161` |
| `matched.es_months` | int_list | `[3, 9]` | non-empty list of integers in [1, 12], no repeats | implemented | Training months carved out for early stopping (the refit then uses every training month). | `scripts/lgbm_submit.py:146` |
| `matched.lightgbm.learning_rate` | float | `0.01` | [0.0001, 1] | implemented | LightGBM learning rate (0.01 = full tier; protocol S screens at validate.screen_learning_rate). | `scripts/lgbm_submit.py:174` |
| `matched.lightgbm.num_leaves` | int | `255` | [2, 4096] | implemented | LightGBM num_leaves: the tree size; larger fits finer interactions and needs more rows per leaf. | `scripts/lgbm_submit.py:174` |
| `matched.lightgbm.min_data_in_leaf` | int | `40` | [1, 100000] | implemented | LightGBM min_data_in_leaf: the smallest leaf; a floor against fitting single-flight noise. | `scripts/lgbm_submit.py:175` |
| `matched.lightgbm.feature_fraction` | float | `0.8` | [0.05, 1] | implemented | LightGBM feature_fraction (column subsample per tree). | `scripts/lgbm_submit.py:175` |
| `matched.lightgbm.bagging_fraction` | float | `0.8` | [0.05, 1] | implemented | LightGBM bagging_fraction (row subsample). | `scripts/lgbm_submit.py:175` |
| `matched.lightgbm.bagging_freq` | int | `1` | [0, 100] | implemented | LightGBM bagging_freq (0 = no bagging). | `scripts/lgbm_submit.py:175` |
| `matched.lightgbm.max_rounds` | int | `50000` | [1, 500000] | implemented | Cap on boosting rounds of the early-stopping run. | `scripts/lgbm_submit.py:178` |
| `matched.lightgbm.patience` | int | `200` | [1, 100000] | implemented | Early-stopping patience (rounds without improvement on es_months). | `scripts/lgbm_submit.py:178` |
| `matched.catboost.learning_rate` | float | `0.02` | [0.0001, 1] | implemented | CatBoost learning rate (0.02 = the R13 full tier; 0.05 screen). | `scripts/analysis/r13_cat.py CB_LR` |
| `matched.catboost.depth` | int | `8` | [1, 16] | implemented | CatBoost symmetric-tree depth (2^depth leaves per tree). | `scripts/analysis/r13_cat.py CB_BASE` |
| `matched.catboost.l2_leaf_reg` | float | `3.0` | [0, 1000] | implemented | CatBoost L2 regularisation of the leaf values (shrinks small leaves). | `scripts/analysis/r13_cat.py CB_BASE` |
| `matched.catboost.subsample` | float | `0.8` | [0.05, 1] | implemented | CatBoost Bernoulli row subsample per tree (stochastic regularisation). | `scripts/analysis/r13_cat.py CB_BASE` |
| `matched.catboost.max_rounds` | int | `30000` | [1, 500000] | implemented | Cap on CatBoost iterations of the early-stopping run. | `scripts/analysis/r13_cat.py CB_CAP` |
| `matched.catboost.patience` | int | `200` | [1, 100000] | implemented | CatBoost early-stopping patience (od_wait). | `scripts/analysis/r13_cat.py CB_PATIENCE` |
| `matched.blend.catboost_weight` | float | `0.5` | [0, 1] | implemented | Weight w of the CatBoost half-blend on matched rows the ADS-B stack does not serve: rint(w * max(proxy - C, 1) + (1 - w) * taxi_F). 0 turns the blend (and the CatBoost fit) off. | `plans/PREREG_ship_cbv_2026_09_13.md` |

## `unmatched`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `unmatched.learner` | choice | `v26` | `v26` / `hgb` | implemented | Unmatched body. `v26` = the SHIPPED v26 body (Amendment A3): build_submission.fit_unmatched's hierarchical S1 cells give the fill rate p and the tail cells; the non-fill term is the MJS body (unm_congestion's congestion design + MAU's 1-in-20 matched extra rows at 25 % mass, HistGradientBoosting with the unmatched.hgb.* knobs); value = rint(max(p * sp + (1 - p) * nf_hybrid(cells, body), 1)); replayed exactly on all 4,907 non-LIRF 2026 rows. `hgb` = the days 1-3 PROVISIONAL E3C-shaped body on FM1's own unmatched frame (never replayed against a shipped file; kept only as a selection-study alternative). | `plans/PREREG_mjs_mxj_ship_2026_09_21.md` |
| `unmatched.hgb.max_iter` | int | `1000` | [1, 100000] | implemented | HGB max_iter of the unmatched body (MXI raised it 200 -> 1000; v26). | `plans/PREREG_mjs_mxj_ship_2026_09_21.md` |
| `unmatched.hgb.n_iter_no_change` | int | `50` | [1, 10000] | implemented | HGB early-stopping patience of the unmatched body (MJS: 50; v26). | `plans/PREREG_mjs_mxj_ship_2026_09_21.md` |
| `unmatched.hgb.max_leaf_nodes` | int | `15` | [2, 4096] | implemented | HGB max_leaf_nodes: the tree size of the unmatched body (small: ~5k training rows per year). | `scripts/build_submission.py:65` |
| `unmatched.hgb.min_samples_leaf` | int | `100` | [1, 100000] | implemented | HGB min_samples_leaf: the smallest leaf of the unmatched body. | `scripts/build_submission.py:65` |
| `unmatched.hgb.learning_rate` | float | `0.1` | [0.0001, 1] | implemented | HGB learning rate (sklearn's default, as every shipped body). | `scripts/build_submission.py:65` |
| `unmatched.encoding_smooth` | float | `50.0` | [0, 100000] | implemented | Smoothing count k of the y target encodings of the `hgb` body. The `v26` body encodes with the shipped build_submission.SMOOTH = 50 inside its estimators, so v26 refuses any other value (it would be ignored). | `scripts/build_submission.py:35` |

## `modules`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `modules.arc.enabled` | bool | `true` | true / false | implemented | ARC (v17): an unmatched departure whose own flight's ARR row (callsign-matched, at its destination) carries LOBT and AOBT_3 takes a clock-recovered value: `ord` -> max(rint(MVT - AOBT_3), 1), `fill` -> max(rint(sp), 1) (arr_row_clocks.sub_rule, arc_ship_build.arc_values). Runs after the Rome rules and before E2 (the shipped order). | `plans/PREREG_arr_row_clocks_2026_09_16.md` |
| `modules.arc.airports` | str_list | `[LIRF]` | non-empty list of strings | implemented | Departure airports ARC rules at. The shipped scope is LIRF only (v17 changed 42 LIRF rows; its thresholds were set on Rome). Other airports are UNVALIDATED: the FMA's all-airport ARC (CORE item 3) needs a LOMO outside Rome first. Every code must be in airports.codes. | `scripts/analysis/arc_ship_build.py:43` |
| `modules.rome.enabled` | bool | `true` | true / false | implemented | LIRF recording-regime module (optional, default ON: owner decision 1), documented as modelling Rome's label recording, in the shipped order on LIRF's unmatched rows: R2 fill classifier (rome_fill, LightGBM, mean of the run seeds) -> mixture with S1's non-fill term -> 24 h date-slip expectation (rome_dateslip) -> rint -> E1 schedule floor (24,000 <= sp < 86,400) -> RLD local-day rule; then (after ARC) E2 and E2W. ARN (v21) is DROPPED. The rules' constants were selected on 2025 labels of all months (in-sample, PREREG_urp_u_chain_replay). | `plans/PREREG_rome_fill_2026_09_10.md` |
| `modules.rome.e2` | bool | `true` | true / false | implemented | E2 / RSE (v31), read only while modules.rome.enabled: on LIRF rows still at their Rome value with 3,600 <= sp < 24,000 (rse_build.row_set), the stand-inbound logit (agent_E_rules.rule_E2, frozen coefficients) replaces the fill probability; a UTC day whose move energy per moved row exceeds e2_day_p99 is reverted. Reads LIRF arrival in-blocks from data.raw_dir itself (the training file of the row's month; ranking.parquet for 2026). | `plans/PREREG_rse_rome_stand_inbound_e2_2026_09_23.md` |
| `modules.rome.e2w` | bool | `true` | true / false | implemented | E2W (v33), read only while modules.rome.enabled: on the rows E2 moved, E2's form with coefficients beta_W refitted at FIT time (stake-weighted Brier, e2w_build.fit_brier_w) on the training months' LIRF band rows, their p_R and non-fill term nested leave-one-month-out; then its own per-day check. Needs modules.rome.e2. | `plans/PREREG_e2w_ship_v33_2026_09_24.md` |
| `modules.rome.e2_day_p99` | float | `15.941366958491995` | [0, 1e+09] | implemented | E2's per-day revert threshold (RSE.1): the 99th percentile of 2025's per-day move energy per moved row (sum of squared moves / 344,841 / rows moved). A 2025-derived constant frozen at the v31 ship; FM1 does not refit it. | `submissions/merry-quicksand_v31_rse.meta.json` |
| `modules.rome.e2w_day_p99` | float | `9.042848041672684` | [0, 1e+09] | implemented | E2W's per-day revert threshold, the same statistic for the beta_W move. A 2025-derived constant frozen at the v33 ship; FM1 does not refit it. | `submissions/merry-quicksand_v33_e2w.meta.json` |
| `modules.storm.mode` | choice | `off` | `off` / `anchor` | implemented | Storm/congestion anchor (v16, re-applied inside the v24-v26 bodies): `off` (the default until the leave-one-event-out re-validation, owner decision 2) or `anchor`: every row the unmatched body decided takes rint(v + max(A - 300, 0)), A = the airport-day median MVT - AOBT_3 of the scored rows' matched departures minus the airport-month median of those day medians (congestion_anchor_ship.day_anomaly / anchored_values; knot 300 s fixed by the shipped function). The constants were chosen post hoc on 2026 label-free data (FMA §1.3 #4). | `plans/PREREG_ship_congestion_anchor_2026_09_13.md` |
| `modules.storm.feature` | bool | `false` | true / false | not_ported | The anomaly A as a body FEATURE with a leave-one-event-out re-validation (owner decision 2's replacement for the anchor rule). NOT PORTED (not built): fit refuses while this is true. | `plans/PREREG_congestion_anchor_loeo_2026_09_13.md` |
| `modules.ldi.enabled` | bool | `false` | true / false | implemented | LTFM de-icing correction (v41): on LTFM's matched departures, a row with w_freezing == 1 on a SEVERE local day (UTC + 3 h; share(MVT - SCHED > 1800 s) >= 0.5 AND min w_temp_c <= 1.0 over the day's LTFM matched departures) takes value + the delta of its MVT - SCHED band (the three modules.ldi.delta_* knobs), before MTX. The day test reads every LTFM departure of the day, so a batch holding part of a scored-file day is refused (PartialDayError). Optional, off; with LTFM outside airports.codes it serves no row (reported). | `plans/PREREG_ldi_ltfm_deicing_2026_09_27.md` |
| `modules.ldi.delta_le30_s` | float | `8.003118629519271` | [0, 3600] | implemented | LDI's shift (s) for target rows with MVT - SCHED <= 30 min. The registered Delta_apply = max(0, 0.5 x (February 2025 OOF cell gap - the 2026 transfer adjustment)); a constant frozen at the v41 ship, not refitted by FM1. | `logs/ldi_scratch/result_20260927T1410/result.json jan26.delta_apply` |
| `modules.ldi.delta_30_60_s` | float | `35.44180751815995` | [0, 3600] | implemented | LDI's shift (s) for target rows with 30 min < MVT - SCHED <= 60 min (registered Delta_apply; frozen at v41). | `logs/ldi_scratch/result_20260927T1410/result.json jan26.delta_apply` |
| `modules.ldi.delta_gt60_s` | float | `122.4855891190231` | [0, 3600] | implemented | LDI's shift (s) for target rows with MVT - SCHED > 60 min (registered Delta_apply; frozen at v41). | `logs/ldi_scratch/result_20260927T1410/result.json jan26.delta_apply` |
| `modules.eobt_clip.enabled` | bool | `false` | true / false | implemented | Serve-time EOBT clip (prc/pipeline/fm1/eobt_clip.py): in an (airport, UTC month) cell of the served rows whose EOBT_1 field carries delay revisions (share of eligible departures with EOBT_1 > LOBT >= min_share), every eligible departure takes EOBT_1 := min(EOBT_1, LOBT) before the feature path. Eligible = a departure at an airport in modules.eobt_clip.airports with AOBT_3, EOBT_1, LOBT and MVT_TIME present. Predict time only (the model was trained on the unclipped field). A batch of the scored file is decided on the scored file's whole cell. Why: from July 2026 LFPG's EOBT_1 carries revisions it did not carry in 2025 (35.1 % vs 3.9 - 7.1 %); priced on a held-out 2025 simulation at about +195 ... +290 bmse (LightGBM only). | `plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md` |
| `modules.eobt_clip.airports` | str_list | `[LFPG]` | non-empty list of strings | implemented | Airports the EOBT clip may act on. Default LFPG only: elsewhere EOBT_1 > LOBT is 8 - 35 % in both 2025 and 2026 (normal practice the model trained on), and the clip was priced at LFPG only. Each must be in airports.codes while the clip is on. | `plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md` |
| `modules.eobt_clip.min_share` | float | `0.15` | [0, 1] | implemented | Share of a cell's eligible departures with EOBT_1 > LOBT at or above which the clip acts in that cell. 0.15 is more than twice the highest unrevised LFPG month (7.1 % in 2025, 6.1 % in January 2026) and less than half of the revised one (35.1 %, July 2026). | `plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md` |
| `modules.eobt_clip.min_rows` | int | `1000` | [1, 1e+08] | implemented | Fewest eligible departures a cell needs before the clip's share test is read; a cell with 1 ... min_rows - 1 is refused (DetectorError), never passed through. At 1,000 rows the share's standard error near 0.15 is 1.1 points. | `plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md` |

## `calibration`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `calibration.mtx.enabled` | choice | `auto` | `auto` / `on` / `off` | implemented | MTX monthly calibration (shift each airport-month mean toward the published EUROCONTROL taxi-out mean). `auto` = on iff run.purpose is scored_period. ALWAYS the last value stage. | `plans/PREREG_mtx_monthly_calibration_2026_09_28.md` |
| `calibration.mtx.reference_csv` | path | `data/mtx/raw/taxi_out_additional_time_2025.csv` | path (relative to the repo root, or absolute); never under `submissions/` | implemented | Published monthly CSV of the training year (fits the excess mapping c, e per airport). | `data/mtx/raw/MANIFEST.json` |
| `calibration.mtx.published_csv` | path | `data/mtx/raw/taxi_out_additional_time_2026.csv` | path (relative to the repo root, or absolute); never under `submissions/` | implemented | Published monthly CSV of the scored period (the target means). Exists only after the month is published. | `data/mtx/raw/MANIFEST.json` |
| `calibration.mtx.lambda_rule` | choice | `james_stein` | `james_stein` / `full` | implemented | Shrink of each cell's shift: `james_stein` = max(0, 1 - sigma^2/gap^2) (V1, registered); `full` = 1 (V1F). | `scripts/analysis/mtx_calib.py:80` |
| `calibration.mtx.mode` | choice | `add` | `add` / `mul` | implemented | `add` = uniform shift per cell (V1, registered); `mul` = scale (V2, not separable on 2025). | `scripts/analysis/mtx_calib.py:87` |

## `validate`

| knob | type | default | range | status | effect | evidence |
|---|---|---|---|---|---|---|
| `validate.screen_learning_rate` | float | `0.05` | [0.0001, 1] | implemented | Learning rate of the screen tier (protocol S ranks arms at this rate, one seed). | `scripts/analysis/r12_validate.py SCREEN_LR` |
| `validate.tier` | choice | `screen` | `screen` / `full` | implemented | `screen` = screen_learning_rate + the first seed only; `full` = the configured learner rates and all seeds. | `reports/final_model/AUDIT_AND_PLAN_2026_09_28.md §4.3` |
| `validate.n_boot` | int | `2000` | [100, 100000] | implemented | Day-block bootstrap draws of every paired comparison. | `scripts/analysis/mtx_calib.py:111` |
| `validate.boot_seed` | int | `20260928` | [0, 2.14748e+09] | implemented | Seed of the day-block bootstrap. | `plans/PREREG_fm1_2026_09_28.md §5` |
