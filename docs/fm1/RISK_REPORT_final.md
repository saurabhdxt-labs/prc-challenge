# Risk report — the final submission `merry-quicksand_final.parquet` (2026-10-08)

Structure: claim, evidence, threats to validity, residual uncertainty, reproduction. The model is the final model of
RISK_REPORT_v59.md, unchanged; this report covers what running it on the organiser's final file adds: two new months,
the dithered January + July rows, and the first end-to-end run of the documented new-period path.

## 1. Claim

`merry-quicksand_final.parquet` (sha256 `a514c644545825…`, 670,790 rows) is the build of `configs/fm1_final.yaml`
(config hash `2950c7552ed7…`: `fm1_v57_open.yaml` with three path knobs, all reuse-free) by the code in this
repository, from the final model's own matched learners (`fit --reuse-from` the v59 artefact: no LightGBM / CatBoost training; the
unmatched lane and MTX recomputed, as on every derived fit; the training table byte-identical), on `final_ranking.parquet` (sha256 `4b0692a7…`), with open inputs only (adsb.lol day
files, IEM weather, EUROCONTROL's published monthly means for months 1–7 of 2026, the organiser's files).

## 2. Evidence

- The path ran end to end on 2026-10-08 (REPRODUCE "Final phase"): every step's acceptance check and output are in the
  table there; the build's manifest records no input changed since the fit.
- January + July against v59 (the same model on the same rows, take-off dithered ±12 s): all 344,841 rows present;
  |Δ| ≤ 12 s on 84.9 %, ≤ 60 s on 99.71 %; mean Δ +0.06 s; 59 rows changed deciding lane; monthly means identical
  (January 1079.3 = v59; July 994.9 vs 994.8). The pipeline reproduced the board-scored model on the dithered rows.
- The new months: the ADS-B stack decided 39 % (February) and 40 % (June) of rows against 42 % / 40 % in January /
  July — adsb.lol coverage is comparable; means 988 s and 985 s against 1079 / 995 s.
- The dither's footprint: 999 rows (0.29 %) moved more than 60 s, mostly in the learned lanes (CatBoost half-blend 544,
  up to 711 s; ADS-B stack 162; unmatched body 202), whose queue / order / ADS-B features are anchored on the dithered
  take-off time; the threshold rules contribute less (LDI 43, storm 14, the Rome E2W rule 7 — one of them the largest
  single move, 6,589 s). Nothing systematic: mean Δ within ±1 s in every lane but `rome+mtx` (+13.7 s on 61 rows) and
  `rome_e2w+mtx` (−12.1 s on 278 rows); per lane, the share within 12 s is 89–91 % for the stack, 81 % for the half-blend,
  60–72 % for LDI and the unmatched body.

## 3. Threats to validity

1. **No held-out check exists for February and June 2026.** The model was selected on 2025 months and the January /
   July 2026 board; February / June 2026 is in-season extrapolation (the 2025 LOMO folds cover those months of 2025,
   METHOD §4, not 2026's traffic). No mitigation is possible without labels; the design is frozen by the organiser's rule.
2. **The dither.** Take-off times moved by up to 12 s on 91.9 % of the January / July rows and the labels follow the
   moved times. Every lane's features that are anchored on take-off move with it; the learned lanes carry most of the
   larger moves (§2).
3. **adsb.lol coverage of the new months** is whatever the archive holds: 2026-06-11 is a partial upstream day (328,510
   samples; 6,028 of its 6,052 departures below `adsb.min_coverage`), served by the designed fallback (the CatBoost
   half-blend), not a tuned choice.
4. **MTX for 2026 months 2 and 6** uses EUROCONTROL's published file as fetched (`data/mtx/raw/MANIFEST.json`); a later
   revision would change the calibration by the published delta.
5. **Rows with predictions above a day** (max 95,937 s) follow the file's own clocks, exactly as v59 did on the same
   rows (95,933 s); unchanged by design.
6. **One blind upload.** Nothing can be selected on the final; every check above is label-free.

## 4. Residual uncertainty

Two repairs of the development tree during the run are not part of the model: the weather cache was rebuilt for the
archive grown by the two new months (its manifest digest), and the earlier pipeline's v6 boosters were refit (LIMITATIONS 26).
A subprocess-timeout test (`tests/test_run_with_heavy_lock.py::test_no_command_is_a_usage_error`) timed out once while
the memory guard paused the whole suite for a Phantom job and passed on the rerun; it is timing-sensitive under that guard.
The score on February / June and on all four months is unknown until the organiser publishes it. The plumbing findings
of the first end-to-end run (a manifest rebuilt from a run's day list — BC-49, fixed with a regression test; a fresh store
for repeated days; the launcher's lock) changed no prediction.

## 5. Reproduction

REPRODUCE "Final phase": the commands in order with the hashes; the final-phase artefact `artifacts/fm1/2950c7552ed7-12679217`;
the four inputs that are not byte-reproducible (DATA_SOURCES) remain the same four.
