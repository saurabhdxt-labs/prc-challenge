# PRC Data Challenge 2026 — taxi-out prediction (team merry-quicksand)

FM1 predicts the taxi-out time (`TAXITIME_SEC_mvt`, seconds) of each departure at ten European airports (EDDF, EDDM,
EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM) from the organiser's movement files, scored by RMSE on the 344,841
departures of January and July 2026.

**The final model is open-data v57** (`configs/fm1_v57_open.yaml`): the forecaster (v50) + monthly calibration to
EUROCONTROL's published 2026 taxi-out means (MTX) + the ADS-B stand-exit stage for departures without a
flight-plan record (`uad`) — every input is the challenge's own files or open, re-fetchable data; nothing from the
owner's adsb.fi capture. Board: **248.5486** (board file v59, 2026-10-06).

| file | what it is | public board (RMSE, s) |
|---|---|---|
| `merry-quicksand_v50.parquet` | the **forecaster**: only the challenge files and public open data — no scored-period aggregate, no private capture (`configs/fm1_v50.yaml`) | 251.4767 |
| `merry-quicksand_v51.parquet` | v50 + MTX | 251.1335 |
| `merry-quicksand_v55.parquet` | v51 + the owner's adsb.fi capture of July 2026 in the ADS-B serve table (`configs/fm1_v55.yaml`) | 249.9168 |
| `merry-quicksand_v57.parquet` | v55 + the aircraft's own ADS-B stand-exit time for 501 departures without a flight-plan record — the best single board file (`configs/fm1_v57.yaml`) | 247.3187 |
| **open-data v57** (`configs/fm1_v57_open.yaml`) | **THE FINAL MODEL**: v57 served on the open ADS-B table instead of the adsb.fi-merged one (`adsb.sources.adsb_fi: false`) — no adsb.fi anywhere | **248.5486** (`merry-quicksand_v59.parquet`) |
| `merry-quicksand_v58.parquet` | an average of board files v33, v36, v41, v43 and v55, with weights fitted on their board scores, plus v57's stand-exit change; v57 unchanged on departures without a flight-plan record — **best board file**; not an FM1 output, and carries adsb.fi through its v55 and v57 members (`docs/fm1/METHOD.md` §8) | 246.2224 |

The board is a single evaluation of the same scored set; no seed-variance estimate exists for these differences, and
the steps kept along the way were kept when the board improved (`docs/fm1/METHOD.md` §8). Board files v55 (249.9168),
v57 (247.3187) and v58 (246.2224) used the owner's adsb.fi capture (v55, v57) or are an average of board files (v58,
not rebuildable from public code); the final model does not use adsb.fi.

## The model in brief

Take-off (`MVT_TIME_UTC_mvt`) is given exactly on every scored row and off-block (`BLOCK_TIME_UTC_mvt`) is withheld, so
predicting taxi-out is predicting off-block. For 98.47 % of scored departures (339,551 rows) the Network Manager supplies
a second off-block clock, `AOBT_3_flt`. FM1 models the **disagreement between the two clocks** rather than the duration:
`taxi = proxy − delta` with `proxy = MVT − AOBT_3` (known) and `delta = BLOCK − AOBT_3` (predicted). A direct model of
the duration was measured worse on held-out January 2025 (−1.73 s [−3.15, −0.35]; `docs/fm1/METHOD.md` §7, RCR2).

- **Matched lane** (rows with `AOBT_3`): a LightGBM model of `delta` on movement clocks, surface queues, movement order,
  METAR weather, an OpenStreetMap stand-to-runway route length and features from public ADS-B tracks (adsb.lol);
  where ADS-B covers the flight at the six airports the stack's gate serves, a stacked residual model (LightGBM + MLP);
  elsewhere a 50/50 blend with a CatBoost model.
- **Unmatched lane** (5,290 rows with no flight-plan record): the shipped estimator of schedule-copied off-blocks versus
  ordinary ones, plus rules for Rome (LIRF), whose labels follow a handler-recording regime. An optional stage (`uad`;
  on in the final model, `configs/fm1_v57_open.yaml`, and in board file v57, `configs/fm1_v57.yaml`) uses the
  aircraft's own ADS-B track where it shows the time the aircraft left its stand (503 of those rows; the stage writes
  the 501 that the body or the storm anchor decided).
- **Calibration** (scored-period build only): each airport-month's mean is pulled toward EUROCONTROL's published monthly
  taxi-out figure, always as the last step.

Every output row names the stage that decided it (`decided_by`) and carries every component's own value, so each
component's contribution is visible. Component ablations, the selection study, validation and the NOT WORKING results
of the final three weeks are in `docs/fm1/METHOD.md`; load-bearing claims and their verdicts in `docs/fm1/CLAIMS.md`.

## Reproducing

```bash
brew install libomp                                        # macOS only: LightGBM needs the OpenMP runtime
python3.11 -m venv .venv && . .venv/bin/activate          # Python version: .python-version
pip install -r requirements-dev.txt                        # runtime pins in requirements.txt
python3.11 -m venv ~/.venvs/prc-catboost && ~/.venvs/prc-catboost/bin/pip install -r requirements-catboost.txt
python tools/env_check.py                                  # both interpreters match the pins, or it says what differs
python tools/clone_check.py                                # the clone tier of the test suite, in a fresh tree
```

CatBoost runs in its own environment so its OpenMP runtime never shares a process with LightGBM's. The data are not
redistributed: the challenge files come from the organiser (`scripts/fetch_data.py` checks each object against the
bucket's ETag or size), and the other inputs are rebuilt by code in this repository from the sources in
`docs/fm1/DATA_SOURCES.md`. Five inputs cannot be regenerated byte for byte: the stored ADS-B stack models
(`data/r12/r13_stack_full/models_lgbcat/`) are shipped in the repository, so a clone has them already, and rebuilding
them from scratch needs the development record and is not a documented public path; the OpenStreetMap route table, the
EUROCONTROL taxi-out CSVs, the adsb.lol day files and the IEM METAR weather are fetched from live sources, whose later
state may differ (each fetch is recorded with a manifest). The adsb.fi capture
behind board files v55 and v57 is extracted by a local step that is not shipped and cannot be re-fetched (it cannot
be redistributed); the final model does not use it. The commands, the derived inputs and what a clone can and cannot
rebuild are in `docs/fm1/REPRODUCE.md`, including how to get the data (`docs/fm1/REPRODUCE.md`, "Getting the data")
and how to run the final model on a new period (`docs/fm1/REPRODUCE.md`, "Running the final model on a new period").

**Final phase (2026-10-08).** The organiser's single blind submission on `final_ranking.parquet` (January, February, June,
July 2026) is the final model run on that file by the documented new-period path, nothing else: `docs/fm1/REPRODUCE.md`,
"Final phase" (every command, hash and check), and `docs/fm1/RISK_REPORT_final.md`. Uploaded 2026-10-10 04:05Z as
`merry-quicksand_final.parquet` (sha256 `a514c644…`, 670,790 rows; the scorer answered `Succeeded`, no score).

**The final model is open-data v57** (`configs/fm1_v57_open.yaml`):

```bash
python -m prc fit --config configs/fm1_v50.yaml --allow-heavy                              # the forecaster; ~10 h, ~20 GB
python -m prc fit --config configs/fm1_v57_open.yaml --allow-heavy \
    --reuse-from <the directory the v50 fit printed>                                       # + MTX + the stand-exit stage
python -m prc build-submission --config configs/fm1_v57_open.yaml --out <dir> --version <N>
```

## Tests

`python tools/run_suite.py` runs the whole suite in two isolated processes (torch and LightGBM never share one; the
tracked tests import no torch, so a clone's second pass is empty). Tests
that need the project's data or development record are marked `devtree`; a fresh clone runs `pytest -m "not devtree"`
(`tools/clone_check.py` does this in a tree built from exactly the tracked files). `--random-order=SEED` shuffles the
collected tests reproducibly.

## Repository layout

| path | contents |
|---|---|
| `prc/` | the FM1 package (`prc/pipeline/fm1/`), its API (`prc/api.py`) and CLI (`python -m prc`); the other modules under `prc/pipeline/` are the earlier pipeline (`python -m prc.pipeline`, board files up to v46) that FM1 superseded — tested and runnable, not the final model |
| `configs/` | `fm1.yaml` (every knob at its documented default), `fm1_v50.yaml` (the forecaster), `fm1_v55.yaml` and `fm1_v57.yaml` (board files using the owner's adsb.fi capture), `fm1_v57_open.yaml` (**the final model**, open data only), `fm1_v57_tc.yaml` (a candidate: v57 with the unmatched body's tail-cell rule off), `fm1_final.yaml` (the final model with the final-period paths, three knobs), `uad_constants_2025.json` (the stand-exit stage's offsets and weights) |
| `scripts/` | the shipped feature builders and estimators FM1 loads, and the builders of its derived inputs |
| `research/` | research and analysis code from FM1's development (pricing harnesses, checks, experiments): not used by the final model, kept open under the same licence (`research/README.md`) |
| `tools/` | suite runner, environment and clone checks, run queue, memory guard, the owner's upload verifier (`verify_upload.py` reads the development record, so it does not run from a clone) |
| `tests/` | the test suite (`tests/fm1/` for the package) |
| `docs/fm1/` | method, limitations, model card, risk report, claims, data sources, reproduction, config reference |

The project's development record (registrations, ledgers, run logs) is **not part of the repository**: it is not
needed to run, test or rebuild FM1, and the documents above describe what it established without naming a path into
it — except the generated `docs/fm1/CONFIG_REFERENCE.md`, whose evidence column names the record's files (its header
says so).

## Licence

Copyright (C) 2026 Saurabh Dixit. Licensed under the GNU GPL, version 3 (`LICENSE`).

Code: GNU General Public License v3.0 (`LICENSE`). Data: see `docs/fm1/DATA_SOURCES.md` (adsb.lol and OpenStreetMap are
ODbL 1.0; the adsb.fi capture is under non-commercial terms, is never redistributed and is not used by the final
model; attribution "Data: adsb.fi (https://adsb.fi)" where it is used). The shipped ADS-B stack models
(`data/r12/r13_stack_full/models_lgbcat/`) contain information from ADSB.lol contributors, made available under the
ODbL 1.0 (`docs/fm1/DATA_SOURCES.md`).
