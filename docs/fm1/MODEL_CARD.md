# FM1 — model card

Status 2026-09-28 (FM1 day 1–3): **package built; not yet a validated model.** Every metric below is TODO until the day 4–7
studies run, and no FM1 number counts until the AT14 end-to-end replay gate passes (`plans/PREREG_fm1_2026_09_28.md`,
Amendment A1). The safe board file meanwhile is `merry-quicksand_v43.parquet` (250.8082), with its defects disclosed
(`reports/final_model/AUDIT_AND_PLAN_2026_09_28.md` §5).

## Intended use

Predict the taxi-out time (seconds, `TAXITIME_SEC_mvt`) of each scheduled departure at ten European airports (EDDF, EDDM,
EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM) from the organiser's movement files, for the PRC Data Challenge 2026. A new
airport is a config entry (`airports.codes`, `airports.tz`) plus its rows in the data.

Not intended for: operational ATC decisions; airports without training history; periods whose data-recording rules differ
from 2025's without a drift check (see LIMITATIONS.md).

## Inputs and outputs

- Inputs: the organiser's raw movement rows (30 columns, one row per DEP or ARR; contract `prc/pipeline/fm1/rows.py`),
  the METAR archive, the OSM route table, the adsb.lol-derived ADS-B tables, and — only for the scored-period build — the
  published EUROCONTROL monthly taxi-out CSV. A scored departure never carries a label; no design column is a label, a label
  derivative or a `*_sec` column (refused, AT2).
- Output: one row per scored departure — `MVT_ID_mvt`, `TAXITIME_SEC_mvt` (int32, >= 1), **`decided_by`** (the last value
  stage that wrote the row: `base_lightgbm`, `adsb_stack`, `cbv_blend`, `unmatched_body`, `rome`, `arc`, `rome_e2`,
  `rome_e2w`, `storm`, ... with `+mtx` when calibration
  moved it), `lane`, and every component's own output (`F`, `base_value`, `stack_resid`, `stack_eligible`, `catboost_C`,
  `unmatched_pred`, `value_before_calibration`, `calibration_shift`; for the unmatched lane also `unm_p_fill`, `unm_nf_cells`,
  `unm_body`, `rome_p_R`, `rome_v7`, `rome_v10`, `rome_value`, `rome_rule`, `arc_rule`, `arc_value`, `e2_p`, `e2w_p`,
  `storm_A`, `storm_shift`).

## Components (owner decisions of 2026-09-28)

| component | class | default | status |
|---|---|---|---|
| matched base model (LightGBM; FW-NSC + ORL + ADS-B blocks; delta target) | core | on | implemented |
| ADS-B support gate + stack residual (LightGBM + MLP) | core | on | implemented (stored stack models) |
| CatBoost half-blend on matched rows the stack does not serve | core | on (w = 0.5) | implemented |
| unmatched body (the shipped v26 body: S1 cells + MJS congestion body) | core | on | implemented (replayed exactly on all 4,907 non-LIRF 2026 rows) |
| ARC arrival-row clock recovery | core | on, LIRF (the shipped scope) | implemented (v17 exact); other airports unvalidated |
| LIRF / Rome recording-regime module (R2 + date-slip, E1, RLD, E2, E2W) | optional | on | implemented (v7/v10/v12 exact; v31/v33 exact except the 8 ARN rows) |
| storm anchor | optional | off (on in the v36 replay) | implemented; the storm-as-feature form not ported |
| ARN | dropped | — | its 8 rows keep their Rome / E2 values |
| LDI LTFM de-icing | optional | off (on for the scored-period build) | implemented: the registered v41 constants (+8.0 / +35.4 / +122.5 s by MVT - SCHED band), target rows computed in the package; replayed exactly vs v41 (1,885 rows) |
| MTX monthly calibration (always last) | optional | auto: on only for the scored-period build | implemented |
| adsb.fi source | optional | off | not ported |
| XGBoost | — | CUT | refused by name |

## Metrics

TODO (days 4–7): per protocol S fold (LOMO-Jan, LOMO-Jul; primary), the strict Jan↔Jul swap (secondary) and forward folds
(tertiary): RMSE and board MSE with day-block 95 % CIs, by airport-month; standalone numbers for every component and the
pipeline number (component-validation rule); ≥ 3 seeds on the final comparison.

## Replay evidence (VERIFIED 2026-09-28)

- AT7: the package's composition functions reproduce the v36 R12 lane on all 339,551 matched rows exactly.
- AT14 stored tier: 5,703 / 5,703 sampled matched rows (68 strata, all ten airports × Jan/Jul) reproduce v36 exactly
  through the package path; the 1,940 sampled unmatched rows are not producible yet → gate FAIL
  (`reports/final_model/fm1_replay_gate_stored_2026_09_28/`).

## Licences

See DATA_SOURCES.md. Code: GPLv3 (`LICENSE`).
