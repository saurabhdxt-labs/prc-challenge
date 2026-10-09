# Risk report — board file v59, the final model (2026-10-06)

Structure: claim, evidence, threats to validity, residual uncertainty, reproduction. v59 is the final model: v57 served
on the open ADS-B table instead of the adsb.fi-merged one, so this report covers what that change means; the stand-exit
stage's risks are in RISK_REPORT_v57.md and the forecaster's in RISK_REPORT_v55.md, and both still apply. Numbers come
from the files named; registrations and reports cited by name are the project's development record, which the rebuild
does not need.

## 1. Claim

`merry-quicksand_v59.parquet` (sha256 `7c64bf9fef4d8bae…`, 344,841 rows) is the build of `configs/fm1_v57_open.yaml`
(config hash `1dc07433357e…`) by the code in this repository: the forecaster (v50) + MTX monthly calibration + the
stand-exit stage `uad`, served on the open ADS-B table `data/r12/tables_51348d3e1cb3` with `adsb.sources.adsb_fi`
false. Every input is the challenge's own files or open, re-fetchable data. It scores **248.5486 s RMSE** on the public
board (all 344,841 scored departures of January and July 2026): 609.9 MSE behind v57 (247.3187, the same model on the
adsb.fi-merged table) and 1,291.6 MSE ahead of v51 (251.1335, the same model without the stand-exit stage).

Not claimed: a rank on a hidden period; that the two differences above would be the same on another period; any
accuracy figure with a confidence interval over seeds (none exists); that v59 is the best file this project can
produce (v57 and v58 score better and use the adsb.fi capture, which the final model excludes by choice:
DATA_SOURCES.md, LIMITATIONS 12).

## 2. Evidence

| claim part | evidence |
|---|---|
| board score | the scoring service's result file for v59: status Succeeded, `used_pairs` 344,841, score 248.5486 |
| the score was predicted | registered before the upload: 248.5481 (v57 plus the v51 → v55 adsb.fi gap of 609.6 MSE, assumed additive); the board gave 248.5486, +0.25 MSE off: the adsb.fi gap was additive here |
| the code produced the file | written by `python -m prc build-submission --config configs/fm1_v57_open.yaml --out <dir> --version 59` from an artefact derived from the v50 donor (`fit --reuse-from`); rebuilt on 2026-10-06 from a re-derived artefact: byte-identical (CLAIMS 26) |
| the config is the final one | `tests/fm1/test_fm1_final_configs.py` pins `configs/fm1_v57_open.yaml` to its config hash; it differs from `configs/fm1_v57.yaml` in two knobs (`adsb.sources.adsb_fi`, `adsb.table_dir`) |
| open data only | `features.check_table_sources` refuses an ADS-B table that declares adsb.fi points while the knob is false, and `uad.check_table_sources` holds the stand-exit table to the same rule (both tested); the tables v59 reads declare adsb.lol only |
| the components' part in the file | `decided_by` on the 344,841 rows: ADS-B stack 140,997 (37,885 + 103,112 with MTX); CatBoost half-blend 196,669 (80,465 + 116,204); unmatched body 3,949; Rome 341; ARC 42; storm anchor 457; LDI 1,885; `uad` 501 (268 + 233); MTX moved 222,081 rows |
| test suite | `tools/run_suite.py` (whole suite, two isolated passes); clone tier `tools/clone_check.py` (counts in REPRODUCE.md) |

## 3. Threats to validity

1. **The open table covers July 2026 less well.** The adsb.fi capture was added because adsb.lol's ground coverage
   fell between 2025 and 2026 (LIMITATIONS 14); without it, more July rows are served by the CatBoost half-blend
   instead of the stack. The 609.9 MSE is the price of that on the board; on another period the price depends on
   that period's adsb.lol coverage, which is not known in advance.
2. **One board evaluation.** Every number in §1 is a single evaluation of the same scored set; the prediction's
   accuracy (+0.25 MSE) shows the adsb.fi gap is additive here, not that it is stable.
3. **Selection on the scored set** (RISK_REPORT_v55 §3.1) applies to every component the final model keeps: each was
   kept when the board improved. The final model itself was not selected by the board (it is v57 with a source
   removed for licence reasons, and it scores worse).
4. **The stand-exit stage's thin evidence** (RISK_REPORT_v57 §3.1–3.3) carries over unchanged: its gain is the same
   1,291.6 MSE here against 1,291.9 between v55 and v57, concentrated at EHAM in January 2026.
5. **MTX uses 2026 published aggregates by design** (CLAIMS 10, LIMITATIONS): the calibration reads EUROCONTROL's
   monthly taxi-out figures for the scored months; on a new period it needs that period's publication or is left off.
6. **The open-data check reads each table's own build report.** The stand-exit table's report declares the sources
   of its day files; the sources of an `--alias-from` table are not merged into it (every alias table in use declares
   adsb.lol only), and the stored stack models carry no source declaration (their provenance is the ODbL notice in
   DATA_SOURCES.md). A reader who rebuilds those inputs from another source must check them by hand.
7. **Byte-for-byte reproduction holds on the author's machine only.** The stack models are shipped rather than
   rebuilt; the OSM route table, the MTX CSVs and the adsb.lol day files are fetched from live sources (REPRODUCE.md,
   "What a clone can and cannot rebuild").

## 4. Residual uncertainty

We cannot exclude that: the adsb.fi gap is larger or smaller on another period; the stand-exit gain is specific to
January 2026 at EHAM; the model's rank on a hidden period differs from its board rank in either direction (CLAIMS 23
is NOT CLAIMED). We can say that no label-derived value and no board-inferred value entered any component.

## 5. Reproduction

| item | value |
|---|---|
| commands | REPRODUCE.md, "Reproducing the final model (open-data v57, board file v59)": derive from the v50 artefact with `--reuse-from`, then `build-submission --artifact` |
| code | the commit that adds `configs/fm1_v57_open.yaml` built the artefact and the file with a clean tree (the artefact records it); the repository's history was rewritten on 2026-10-06, so the artefact's recorded hash is the pre-rewrite one (LIMITATIONS 18) |
| environment, hardware, seeds | as RISK_REPORT_v55 §5 |
| data | DATA_SOURCES.md; no input from the owner's adsb.fi capture |
