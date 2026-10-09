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
- Held-out February and June 2025 with the final design (fit on the other eleven months of 2025, early stopping on months
  3 + 9, screen tier, ADS-B stack on, MTX off, the 2025 stand-exit table; METHOD §6): RMSE 249.36 s on 143,693 February
  departures and 275.26 s on 183,080 June departures (rows under one hour: 173.7 / 180.8 s); the January / July folds of
  the same protocol were 327 / 266 s with the stack off. The months the final phase added were never selected on, and
  the design holds there. These two folds were not pre-registered with a TRUE/FALSE shape, ran with the ADS-B stack on
  (the January/July folds in METHOD §6 ran with it off), and the stand-exit and unmatched-lane constants were fitted on
  the ten design months that include February and June (LIMITATIONS 3, 17, METHOD §7): the numbers are a development-record
  check, not a closed claim.
- Coverage of the shipped encodings' keys in the new months (label-free, against the 2025 training departures): airline
  (callsign prefix) unseen on 0.1 / 0.1 / 0.4 / 0.5 % of rows in January / February / June / July, stand 0.0 / 0.0 / 0.8 / 0.8 %,
  operator 1.6 / 0.9 / 1.4 / 1.6 % — the encodings reach the new months as they reach the scored ones (an unseen key
  falls back to the prior). Flight numbers themselves churn with the summer schedule (36 % of June / July rows carry
  one unseen in 2025; 47 % at LIRF), which is why no component keys on the callsign.
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
   July 2026 board; the held-out February / June 2025 folds above are the closest labelled evidence (2025 traffic, not
   2026's). No mitigation is possible without labels; the design is frozen by the organiser's rule.
2. **The dither.** Take-off times moved by up to 12 s on 91.9 % of the January / July rows and the labels follow the
   moved times. Every lane's features that are anchored on take-off move with it; the learned lanes carry most of the
   larger moves (§2).
3. **adsb.lol coverage of the new months** is whatever the archive holds: 2026-06-11 is a partial upstream day (328,510
   samples; 6,028 of its 6,052 departures below `adsb.min_coverage`), served by the designed fallback (the CatBoost
   half-blend), not a tuned choice.
4. **MTX on the two new months.** The calibration shifts every airport × UTC-month cell toward EUROCONTROL's published
   monthly taxi-out figure (James–Stein shrinkage; METHOD §3); on the final file it moved the February and June cells by
   −32 … +19 s per airport-month (EGLL −13 / −27 s, LEBL −12 / −32 s, LIRF +7 / +19 s, EDDF +1 / +15 s; the January / July
   cells by −41 … +20 s, as in v59), from the same published file (months 1–7 of 2026, as fetched; `data/mtx/raw/MANIFEST.json`).
   The basis, as the organiser stated it (DATA_SOURCES; Discord, recorded by the team): on 2026-09-27, asked whether the
   published monthly average taxi-out figures may be a model input, "yes"; on 2026-10-02, on open data, "anything you think
   that helps, use it" (data without a licence is not open); on 2026-10-08, for the final phase, "the design of the model
   shouldn't change". The published figures are open data, the calibration is part of the model's design, and the final
   phase asks for that design unchanged, so the same component runs on the four months from the same published file. The
   2026-09-27 question itself named January / July, the scored months of the time; on 2026-10-08 the organiser answered
   another team's question about this exact step on the final ("mean taxi-out per airport-month, including the scored
   months, shift predictions toward it"): "no change. still ok" (DATA_SOURCES, verbatim). A later revision of the published
   file would change the calibration by the published delta.
5. **The ADS-B support gate is the stored one, decided on January / July 2026.** `adsb.support_gate.rule: manifest` serves
   the stack at EDDF, EDDM, EHAM, LEBL, LIRF and LSZH exactly as v59 did; no gate was recomputed for February / June. The
   class-3 (joined) coverage of the final serve table per gated airport, January / February / June / July: EDDF 0.114 /
   0.066 / 0.123 / 0.107; EDDM 0.769 / 0.768 / 0.657 / 0.739; EHAM 0.610 / 0.581 / 0.597 / 0.625; LEBL 0.354 / 0.358 /
   0.491 / 0.463; LIRF 0.181 / 0.171 / 0.000 / 0.000; LSZH 0.480 / 0.472 / 0.285 / 0.295 — the new months sit next to the
   scored ones at every gated airport, and the stack decided 39–40 % of their rows against 40–42 %. The gate's LIRF entry is
   an override: the stored 0.7× coverage rule failed there (0.299 against 0.378) and the author kept LIRF on (the stack's
   `MANIFEST.json` records it; the decision's registration is in the development record, not shipped).
6. **Rows with predictions above a day** (max 95,937 s) follow the file's own clocks, exactly as v59 did on the same
   rows (95,933 s); unchanged by design.
7. **What was decided on the scored period's unlabelled data or on the public board** (all disclosed; collected here for
   the review of the final): whole model versions were selected by their board score (METHOD §8); the storm anchor's form
   and 300 s knot were chosen post hoc on 2026 label-free data and the anchor was kept on a board result (CONFIG_REFERENCE,
   `modules.storm.mode`); the LDI deltas carry a transfer adjustment computed on the January 2026 scored rows without labels
   (`modules.ldi.*`); the ADS-B stack gate was set on January / July 2026 coverage with the Rome override (item 5); the
   CatBoost half-blend and the stack's `lgbcat` form were kept on board comparisons. The queue block and the order block read
   other supplied rows, some later than the departure (LIMITATIONS 27–28), and the ADS-B features and the stand-exit stage
   measure the flight's own surface movement before take-off (METHOD §2, §3; DATA_SOURCES, the 2026-09-18 statement). Nothing
   reads a scored departure's label; the organiser statements before 2026-10-08 are recorded as paraphrases; the
   2026-10-08 and 2026-10-09 statements are quoted verbatim (DATA_SOURCES).
8. **The Rome exploration of 2026-10-09 changed nothing.** With the final file built, a registered search for a label-free,
   reuse-free stage that would reduce Rome's error ran on the eight non-gate 2025 months (METHOD §7, MFI; LIMITATIONS 30): the
   error is dominated by three label-artefact rows and by schedule copies the learners already hedge; the one candidate the
   ledger had not closed in a reuse-free form reached +2.4 bmse per month against a bar of +100, and its cheating ceiling is
   1–28. No component was added or changed; the organiser's final-phase sentence — "the design of the model shouldn't change" —
   was kept as written, and the owner's reading (an improved component inside the same pipeline would have been exploration,
   not a design change) was never exercised. Label reads during the search: the 2025 training labels (held out by month) and
   one label-free count of the final file's LIRF rows per band; the final file's labels do not exist.
9. **One blind upload.** Nothing can be selected on the final; every check above is label-free.

## 4. Residual uncertainty

Two repairs of the development tree during the run are not part of the model: the weather cache was rebuilt for the
archive grown by the two new months (its manifest digest), and the earlier pipeline's v6 boosters were refit (LIMITATIONS 26).
A subprocess-timeout test (`tests/test_run_with_heavy_lock.py::test_no_command_is_a_usage_error`) timed out once while
the memory guard paused the whole suite for another scheduled job on the build machine and passed on the rerun; it is timing-sensitive under that guard.
The score on February / June and on all four months is unknown until the organiser publishes it. The plumbing findings
of the first end-to-end run (a manifest rebuilt from a run's day list — BC-49, fixed with a regression test; a fresh store
for repeated days; the launcher's lock) changed no prediction.

## 5. Reproduction

REPRODUCE "Final phase": the commands in order with the hashes; the final-phase artefact `artifacts/fm1/2950c7552ed7-12679217`;
the five inputs that are not byte-reproducible (DATA_SOURCES) remain the same five.
