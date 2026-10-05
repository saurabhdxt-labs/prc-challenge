# PRC Data Challenge 2026 — taxi-out prediction (team merry-quicksand)

FM1 predicts the taxi-out time (`TAXITIME_SEC_mvt`, seconds) of each departure at ten European airports (EDDF, EDDM,
EGLL, EHAM, LEBL, LEMD, LFPG, LIRF, LSZH, LTFM) from the organiser's movement files, scored by RMSE on the 344,841
departures of January and July 2026.

| file | what it is | public board (RMSE, s) |
|---|---|---|
| `merry-quicksand_v50.parquet` | FM1 as a **forecaster**: only the challenge files and public open data — no scored-period aggregate, no private capture (`configs/fm1_v50.yaml`) | 251.4767 |
| `merry-quicksand_v51.parquet` | v50 + monthly calibration to EUROCONTROL's published 2026 monthly taxi-out means (MTX) | 251.1335 |
| `merry-quicksand_v55.parquet` | v51 + the owner's adsb.fi capture of July 2026 in the ADS-B serve table (`configs/fm1_v55.yaml`) | 249.9168 |
| `merry-quicksand_v57.parquet` | v55 + the aircraft's own ADS-B stand-exit time for 501 departures without a flight-plan record — **the team's best file** (`configs/fm1_v57.yaml`) | 247.3187 |

The board is a single evaluation of the same scored set; no seed-variance estimate exists for these differences, and
the steps kept along the way were kept when the board improved (`docs/fm1/METHOD.md` §8).

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
  on in board file v57, `configs/fm1_v57.yaml`) uses the aircraft's own ADS-B track where it shows the time the
  aircraft left its stand (503 of those rows; the stage writes the 501 that the body or the storm anchor decided).
- **Calibration** (scored-period build only): each airport-month's mean is pulled toward EUROCONTROL's published monthly
  taxi-out figure, always as the last step.

Every output row names the stage that decided it (`decided_by`) and carries every component's own value, so each
component's contribution is visible. Component ablations, the selection study, validation and the NOT WORKING results
of the final three weeks are in `docs/fm1/METHOD.md`; load-bearing claims and their verdicts in `docs/fm1/CLAIMS.md`.

## Reproducing

```bash
python3.11 -m venv .venv && . .venv/bin/activate          # Python version: .python-version
pip install -r requirements-dev.txt                        # runtime pins in requirements.txt
python3.11 -m venv ~/.venvs/prc-catboost && ~/.venvs/prc-catboost/bin/pip install -r requirements-catboost.txt
python tools/env_check.py                                  # both interpreters match the pins, or it says what differs
python tools/clone_check.py                                # the clone tier of the test suite, in a fresh tree
```

CatBoost runs in its own environment so its OpenMP runtime never shares a process with LightGBM's. The data are not
redistributed: the challenge files come from the organiser (`scripts/fetch_data.py` checks each object against the
bucket's ETag or size), and the other inputs are rebuilt by code in this repository from the sources in
`docs/fm1/DATA_SOURCES.md` — except the stored ADS-B stack models, which the current code cannot regenerate byte for byte,
and the adsb.fi capture behind v55 and v57, which is extracted by a local step that is not shipped (it cannot be redistributed). The commands, the
derived inputs and what a clone can and cannot rebuild are in `docs/fm1/REPRODUCE.md`; the final model is

```bash
python -m prc fit --config configs/fm1_v50.yaml --allow-heavy                    # ~10 h, ~20 GB
python -m prc build-submission --config configs/fm1_v50.yaml --out <dir> --version 50
```

## Tests

`python tools/run_suite.py` runs the whole suite in two isolated processes (torch and LightGBM never share one). Tests
that need the project's data or development record are marked `devtree`; a fresh clone runs `pytest -m "not devtree"`
(`tools/clone_check.py` does this in a tree built from exactly the tracked files). `--random-order=SEED` shuffles the
collected tests reproducibly.

## Repository layout

| path | contents |
|---|---|
| `prc/` | the FM1 package (`prc/pipeline/fm1/`), its API (`prc/api.py`) and CLI (`python -m prc`) |
| `configs/` | `fm1.yaml` (every knob at its documented default), `fm1_v50.yaml`, `fm1_v55.yaml` and `fm1_v57.yaml` (the final runs), `fm1_v57_tc.yaml` (a candidate: v57 with the unmatched body's tail-cell rule off), `uad_constants_2025.json` (the stand-exit stage's offsets and weights) |
| `scripts/` | the shipped feature builders and estimators FM1 loads, and the builders of its derived inputs |
| `tools/` | suite runner, environment and clone checks, run queue, memory guard, upload verifier |
| `tests/` | the test suite (`tests/fm1/` for the package) |
| `docs/fm1/` | method, limitations, model card, risk report, claims, data sources, reproduction, config reference |

The project's development record (registrations, measurement reports, run logs) is not needed to run, test or rebuild
FM1; the documents above cite it by name where a number comes from it.

## Licence

Code: GNU General Public License v3.0 (`LICENSE`). Data: see `docs/fm1/DATA_SOURCES.md` (adsb.lol and OpenStreetMap are
ODbL 1.0; the adsb.fi capture is under non-commercial terms and is never redistributed).
