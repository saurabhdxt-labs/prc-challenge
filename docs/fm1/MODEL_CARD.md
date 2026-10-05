# FM1 — model card (updated 2026-10-05)

Status: **final.** The forecaster (only the challenge files and public open data: no scored-period aggregate, no private capture) is board file v50
(`configs/fm1_v50.yaml`, public board 251.4767); the scored-period build is v55 (`configs/fm1_v55.yaml`, 249.9168); the
best single model is v57, v55 plus the stand-exit stage for departures without a flight-plan record
(`configs/fm1_v57.yaml`, 247.3187). The team's best board file is v58 (246.2224): an average of board files with
weights fitted on their board scores, which is not an FM1 output (METHOD §8). Claims and their evidence: CLAIMS.md;
risks: RISK_REPORT_v55.md, RISK_REPORT_v57.md and RISK_REPORT_v58.md.

## Intended use

Predict the taxi-out time (seconds, `TAXITIME_SEC_mvt`) of each scheduled departure at ten European airports (EDDF, EDDM,
EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM) from the organiser's movement files, for the PRC Data Challenge 2026 —
a post-operations estimate (the organiser's framing), not a tactical forecast. A new airport needs a config entry
(`airports.codes`, `airports.tz`), its rows in the data, its weather, route and ADS-B inputs, and a stack-gate entry.

Not intended for: operational ATC decisions; airports without training history; periods whose data-recording rules differ
from 2025's without a drift check (LIMITATIONS.md 9, 14).

## Inputs and outputs

- Inputs: the organiser's raw movement rows (30 columns, one row per DEP or ARR; contract `prc/pipeline/fm1/rows.py`),
  the METAR archive, the OpenStreetMap route table and the ADS-B tables built from adsb.lol tracks. The scored-period
  build (v51, v55) adds EUROCONTROL's published monthly taxi-out means (MTX) and, in v55, the owner's adsb.fi capture of
  July 2026 in the serve table. The optional `uad` stage reads a stand-exit table built from the same adsb.lol tracks. A scored departure never carries a label; no design column is a label, a label
  derivative or a `*_sec` column (refused, AT2).
- Output: one row per scored departure — `MVT_ID_mvt`, `TAXITIME_SEC_mvt` (int32, >= 1), **`decided_by`** (the last value
  stage that wrote the row: `base_lightgbm`, `base_catboost`, `adsb_stack`, `cbv_blend`, `unmatched_body`, `rome`, `arc`, `rome_e2`,
  `rome_e2w`, `storm`, `uad`, `ldi`, with `+mtx` when calibration moved it), `lane`, and every component's own output (`F`,
  `base_value`, `stack_resid`, `stack_eligible`, `catboost_C`, `unmatched_pred`, `value_before_calibration`,
  `calibration_shift`, `ldi_delta`, `ldi_band`, `eobt_clipped`; for the unmatched lane also `unm_p_fill`, `unm_nf_cells`, `unm_body`, `rome_p_R`, `rome_v7`,
  `rome_v10`, `rome_value`, `rome_rule`, `arc_rule`, `arc_value`, `e2_p`, `e2w_p`, `storm_A`, `storm_shift`; with the
  `uad` stage on also `uad_cls`, `uad_est`, `uad_shift`).

## Components in the final configuration

| component | in v50 | in v55 | rows deciding v55 (`decided_by`, with and without `+mtx`) |
|---|---|---|---|
| matched base model (LightGBM, `delta` target; queue, order, weather, ORL, ADS-B blocks) | yes | yes | base of every matched row |
| ADS-B support gate + stacked residual (LightGBM + MLP; stored lgbcat stack) | yes | yes | 145,538 |
| CatBoost half-blend on matched rows the stack does not serve | yes | yes | 192,128 |
| unmatched body (S1 cells + congestion body) | yes | yes | 4,326 |
| Rome (LIRF) module: R2 + date-slip, E1, local-day rule, E2, E2W | yes | yes | 341 |
| ARC arrival-row clock recovery (LIRF) | yes | yes | 42 |
| storm anchor | yes | yes | 581 |
| LDI LTFM de-icing shift | yes | yes | 1,885 |
| ADS-B stand-exit estimate for departures without a flight-plan record (`uad`) | no | no | — (optional; on in v57, `configs/fm1_v57.yaml`, where it decides 501 rows: 377 the unmatched body decides in v55 and 124 the storm anchor) |
| tail-cell rule of the unmatched body (`modules.tail_cells.enabled`) | on | on | — (the shipped rule; off in the candidate `configs/fm1_v57_tc.yaml`, where 10 rows change) |
| MTX monthly calibration (last) | no (forecaster) | yes | moved 236,515 rows |
| adsb.fi July-2026 serve rows | no | yes (serve table only; the learners are v50's) | — |
| EOBT clip (LFPG July) | off | off | — (lost on the board, v54) |
| coverage-ratio support gate | off | off | — (no better on the board, v52) |
| XGBoost | cut | cut | refused by name |

Row counts from `submissions/merry-quicksand_v55.parquet.meta.json` (`decided_by`).

## Metrics

| measurement | value | what it is |
|---|---|---|
| public board, v58 | 246.2224 s RMSE | an average of board files, weights fitted on the board (not an FM1 output); one evaluation |
| public board, v57 | 247.3187 s RMSE | v55 + the stand-exit stage (501 rows); the best single model; one evaluation of the scored set |
| public board, v55 | 249.9168 s RMSE | the scored-period build; one evaluation of the scored set |
| public board, v50 | 251.4767 s RMSE | the forecaster |
| 2025 validation, protocol S (leave one month out), screen tier, ADS-B stack off | January 327.07 s, July 266.21 s (pooled 294.92 s) | a 2025 reference for a reduced configuration, not the final model's score (METHOD §6) |
| determinism | 344,841 × 30 cells, 0 different | two fits of a three-month slice, served on the full scored file |

No metric has a run-to-run variance estimate: each board number is one run of the three-seed ensemble and each
screen-tier number one seed (LIMITATIONS 1). Standalone contributions of
each component: the swap-study ablations (METHOD §5) and the board history of whole-model variants (METHOD §8).

## Replay evidence

- AT7: the package's composition functions reproduce the v36 R12 lane on all 339,551 matched rows exactly.
- The unmatched lane reproduces the shipped v26–v33 chain on all 5,290 unmatched scored rows except ARN's 8 (ARN dropped;
  `tests/fm1/test_fm1_unmatched.py`).
- The two final configs resolve to the stored artefacts' config hashes (`tests/fm1/test_fm1_final_configs.py`).

## Licences

See DATA_SOURCES.md. Code: GPLv3 (`LICENSE`).
