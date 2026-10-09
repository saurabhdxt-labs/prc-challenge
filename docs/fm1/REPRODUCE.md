# FM1 — reproduction guide (updated 2026-10-06: the final model, open-data v57; getting the data; running on a new period)

Environment: Python 3.11.15 (`.python-version`) with `requirements.txt` (runtime) and `requirements-dev.txt` (tests);
CatBoost 1.2.10 lives only in `~/.venvs/prc-catboost` (`requirements-catboost.txt`) and is used through the worker
process, so its OpenMP runtime never shares a process with LightGBM's. `python tools/env_check.py` compares both
interpreters with the pins (exit 1 names every difference). On macOS, LightGBM needs the OpenMP runtime
(`brew install libomp`). On the development machine the project interpreter is
`/opt/homebrew/bin/python3.11`; there, a bare `python3` resolves to the CatBoost venv.

## Commands

```
python tools/env_check.py                                               # both interpreters match the pins
python tools/clone_check.py                                             # the clone tier, in a tree of the tracked files
python -m prc config-reference --check                                  # the reference doc is current
python -m prc validate --config configs/fm1.yaml --protocol dry-run     # config, plan, every input path; fits nothing
python -m prc fit --config configs/fm1.yaml [--set path=value ...] [--allow-heavy]
python -m prc fit --config configs/fm1.yaml --reuse-from <donor artefact dir> --set <allow-listed knob>=<value> [--allow-heavy]
python -m prc predict --config configs/fm1.yaml --rows <raw scored parquet> --out <file.parquet>
python -m prc build-submission --config configs/fm1.yaml --out <dir> --version <N> [--set run.purpose=scored_period]
python -m prc validate --config configs/fm1.yaml --protocol S|swap|forward|lomo|forward_monthly --out <dir> [--allow-heavy]
python -m prc compare --a <oof_A_F1> <oof_A_F2> --b <oof_B_F1> <oof_B_F2> --out <file.json>   # pooled, paired
python -m prc blend --a <oof_A ...> --b <oof_B ...> --w 0.5 --out <blend.parquet>                # arm A2 offline
python -B tools/fm1_replay_gate.py --out <new dir> [--per-stratum 100 --runs 2]                        # AT14
python -B tools/run_suite.py                                                                          # the suite
```

`configs/fm1.yaml` holds every knob at its documented default (a test pins this). Five config files write their
command-line overrides in (`tests/fm1/test_fm1_final_configs.py` pins each one's config hash and its exact overrides);
four resolve to stored artefacts (v50, v55, v57, and the final model `fm1_v57_open.yaml`, derived on 2026-10-06), and
`fm1_v57_tc.yaml` (a candidate) is pinned to its own hash because no artefact was ever built for it:

- `configs/fm1_v50.yaml` — the forecaster (lgbcat ADS-B stack on the `tables_51348d3e1cb3` tables, storm anchor on,
  LDI on, 12 threads).
- `configs/fm1_v55.yaml` — the scored-period build (v50 + MTX + the adsb.fi serve table; `adsb.sources.adsb_fi`).
- `configs/fm1_v57.yaml` — board file v57, the best single board file: v55 with the optional stand-exit stage on
  (`modules.uad.enabled`).
- **`configs/fm1_v57_open.yaml` — THE FINAL MODEL**: v57 on open data only (`adsb.sources.adsb_fi: false`,
  `adsb.table_dir: data/r12/tables_51348d3e1cb3`, the open table v50 and v51 serve); board file v59 (248.5486).
- `configs/fm1_v57_tc.yaml` — a candidate not uploaded on its own: v57 with the unmatched body's tail-cell rule off
  (`modules.tail_cells.enabled` false). No stored artefact; its pinned hash is the configuration's own.

The storm-as-feature form is refused by name when switched on. LDI needs whole local days per predict call. A fit
over the twelve months is a heavy job (`--allow-heavy`, the pre-run checks first, one heavy job at a time). Validation folds
with a single training month must switch E2W off (`--set modules.rome.e2w=false`: its beta_W is fitted leave-one-month-out).

## Getting the data

In this order (every command's flags are from its own `--help`):

1. **Challenge files.** `python scripts/fetch_data.py [--bucket prc-2026-datasets] [--dst data/raw] [--force]` —
   needs `PRC_ACCESS_KEY` and `PRC_SECRET_KEY` in the environment, or a `KEY=value` file at `~/.prc-challenge.env`
   (`prc/s3.py`, `load_credentials`). Pulls the organiser's movement parquet files into `data/raw/`, verified against the
   bucket's ETag (single-part uploads) or size (multipart); recorded in `data/raw/FETCH_MANIFEST.json`. On this
   machine `data/raw/` holds 315 MB (`du -sh`, 2026-10-06); fetch duration is network-bound and not measured here.

2. **adsb.lol.** `python scripts/adsb_fetch.py index --year <year>` builds `data/adsb/v2/RELEASES_<year>.json` from
   the GitHub API; then `python scripts/adsb_fetch.py run [--year <year >= 2025>] [--months M1,M2,...] [--limit N]`
   fetches daily archives and extracts ground-level samples at the ten scored airports into `data/adsb/v2/` (2025
   January + July), `data/adsb/v2/y2025_other` (2025, other months) or `data/adsb/v2/y2026` (2026); resumable, one
   day at a time, and stops if free disk falls below its floor. Where it writes (`scripts/adsb_fetch.py`
   `out_dir_for`): a run WITH `--months` goes to `data/adsb/v2/y<year>_other`; a run without `--months`
   fetches January and July of that year (`day_list`) into `data/adsb/v2` for 2025 and `data/adsb/v2/y<year>` otherwise. On this machine `data/adsb/`
   holds 31 GB; duration is network- and disk-bound and not measured here.

3. **Stand / order caches.** `python scripts/stand_ab.py cache` and `python scripts/stand_ab.py order-cache` write
   per-month feature caches to `data/cache_stand/` and `data/cache_order/` for the training months;
   `python scripts/stand_ab.py --ranking cache` and `python scripts/stand_ab.py --ranking order-cache` (the flag goes
   BEFORE the sub-command) write `<cache dir>/ranking.parquet` for the scored file instead. `scripts/analysis/r12_build.py` reads both
   (`STAND_DIR` / `ORDER_DIR` = `data/cache_stand` / `data/cache_order`, `r12_build.py:68`; the training-universe
   join at `r12_build.py:419-449`) plus `cache_stand/ranking.parquet`. On this machine `data/cache_stand/` holds
   9.5 GB and `data/cache_order/` holds 38 MB.

4. **Weather.** `python scripts/fetch_weather.py [--smoke] [--out-dir DIR] [--refetch] [--months MONTHS]` writes
   one CSV per (airport, month) to `data/weather/`, for every month the caches use plus both scored months by
   default; frozen on disk and never re-fetched by a run once saved (a month already on disk is skipped). On this
   machine `data/weather/` holds 24 MB.

5. **MTX.** No fetch script: download `taxi_out_additional_time_2025.csv` and `taxi_out_additional_time_2026.csv`
   by hand from `https://www.eurocontrol.int/performance/data/download/csv/...` and save them to
   `data/mtx/raw/taxi_out_additional_time_<year>.csv`; record their sha256 in `data/mtx/raw/MANIFEST.json`.
   `scripts/analysis/mtx_source.py` parses the files (`load_monthly_csv`); the 2026 file is a live publication (it
   grows through the year) and is never re-fetched by a run once saved. On this machine `data/mtx/` holds 18 MB.

6. **The uad stand-exit table** (only needed for the `uad` stage, on in the final model and in board file v57): four
   passes, as they were run for the final configuration (`data/adsb/v2` holds the January and July 2025 day files,
   `y2025_other` the other 2025 months, `y2026` the 2026 months; the directory names under `data/uad/` are the
   author's — the stage checks a table's build report, not its path):
   `python scripts/analysis/uad_build.py parked --rows data/raw/training_2025-*.parquet --adsb-dir data/adsb/v2
   data/adsb/v2/y2025_other data/adsb/v2/y2026 --out data/uad/2025_parked`, then
   `python scripts/analysis/uad_build.py table --rows data/raw/training_2025-*.parquet --adsb-dir data/adsb/v2
   data/adsb/v2/y2025_other data/adsb/v2/y2026 --parked data/uad/2025_parked --out data/uad/2025_table_a3`, then
   `python scripts/analysis/uad_build.py parked --rows data/raw/ranking.parquet --adsb-dir data/adsb/v2/y2026
   data/adsb/v2/y2025_other --out data/uad/2026_parked`, then
   `python scripts/analysis/uad_build.py table --rows data/raw/ranking.parquet --adsb-dir data/adsb/v2/y2026
   data/adsb/v2/y2025_other --parked data/uad/2025_parked data/uad/2026_parked --alias-from data/uad/2025_table_a3
   --out data/uad/2026_table_a3` (the 2026 table the final model reads; its build report records the sha256 of its
   row files and alias tables, and tables built from now on also record the parked positions). On this machine
   `data/uad/` holds 145 MB.

Sizes above are what this machine's `data/` directories hold today (`du -sh`, 2026-10-06), not a specification; a
fresh fetch may differ, and no step above prints a wall-clock duration this project has recorded, so none is given
beyond what is noted.

## Reproducing the board file v55

```
python -m prc fit --config configs/fm1_v50.yaml --allow-heavy                                     # the donor (~10 h)
python -m prc fit --config configs/fm1_v55.yaml --allow-heavy --reuse-from <the directory the v50 fit printed>
python -m prc build-submission --config configs/fm1_v55.yaml --out <new dir> --version 55
```

On the author's machine the v50 fit's own artefact directory was `artifacts/fm1/2d29ad344b45-1c61fcc2`.

The uploaded file is `merry-quicksand_v55.parquet`, sha256 `225ab0be205a0a254d0c27e726ec33d6007981b98f937f9b00739499e5ac537d`
(344,841 rows). The rebuild check — the current code serving the stored v55 artefact
(`artifacts/fm1/d811c69dc93e-141862a0`) — is recorded below.

**Rebuild check (VERIFIED 2026-10-04 05:38–05:56 EDT, with the committed code).** `python -m prc build-submission
--config configs/fm1_v55.yaml --out <new dir> --version 55` served the stored artefact and wrote a file with sha256
`225ab0be205a0a254d0c27e726ec33d6007981b98f937f9b00739499e5ac537d` — identical to the uploaded v55. Its manifest hashes
204 inputs and names none as changed since the fit. The full fit itself (~10 h) was not re-run for this check.

## Reproducing the board file v57

```
python -m prc fit --config configs/fm1_v57.yaml --allow-heavy --reuse-from <the directory the v50 fit printed>
python -m prc build-submission --config configs/fm1_v57.yaml --out <new dir> --version 57
```

On the author's machine the v50 fit's own artefact directory was `artifacts/fm1/2d29ad344b45-1c61fcc2`.

The uploaded file is `merry-quicksand_v57.parquet`, sha256 `4bf0f915b90ae7374aa97e435a254d9d2bf73292bda30f47cad55d97c154350b`
(344,841 rows). These two commands wrote it on 2026-10-05 with the committed code and a clean tree (artefact
`artifacts/fm1/e3c1d6b473f4-141862a0`; the v50 artefact is the donor, as for v55), so the uploaded file is the
committed code's own output. Its manifest records the stage's two inputs: the
stand-exit table (`data/uad/2026_table_a3/table.parquet`, sha256
`15ab2b0862a6d435f35891a157a823d9589af5d799a30ae89c8708d45eb285e8`) and the constants (`configs/uad_constants_2025.json`,
sha256 `7d1b6bbcec5ab0f9318fb41d40d641c63f30a78906830bcbe16c92dcc3e147b7`). The table is a derived input (below).

**Rebuild checks (VERIFIED 2026-10-05, with the code of the commit that adds this paragraph).** (1) `python -m prc
build-submission --config configs/fm1_v57.yaml --out <new dir> --version 57` served the stored artefact and wrote a
file with sha256 `4bf0f915b90ae7374aa97e435a254d9d2bf73292bda30f47cad55d97c154350b`, identical to the uploaded v57;
its manifest names no input as changed since the fit. The provenance file's values are equal in every column; its
bytes are not, because it embeds the lane's wall time. (2) `scripts/analysis/uad_build.py table`, run again on the
same ADS-B day store and the same two parked directories, wrote `table.parquet`, `cands.parquet` and `refs.parquet`
byte for byte as before (table sha256 `15ab2b08…` above). A rebuild from a fresh download of the day files has not
been tried. The stage reads `report.json` beside the table (the build report: parameter hash, the label-free checks,
and for tables built from now on the sha256 of the files written) and refuses a table whose report fails the
registered checks or names another table; keep the two files together.

## Reproducing the final model (open-data v57)

```
python -m prc fit --config configs/fm1_v50.yaml --allow-heavy                                     # the donor (~10 h)
python -m prc fit --config configs/fm1_v57_open.yaml --allow-heavy --reuse-from <the directory the v50 fit printed>
python -m prc build-submission --config configs/fm1_v57_open.yaml --out <new dir> --version <N>
```

Builds from the artefacts stored before 2026-10-06 (v50, v55, v57) log one changed input,
`stack/MANIFEST.json`: its one absolute path was made repository-relative after they were fitted. No stage reads that
field; the v57 rebuild after the change wrote the same file (sha256 `4bf0f915…`).

When more than one artefact exists for the same config hash (for example after re-deriving it on changed inputs),
`build-submission` refuses to guess ("2 artefacts for config … (different data); pass the artefact directory
explicitly"): add `--artifact <the directory the fit printed>`.

`configs/fm1_v57_open.yaml` is `configs/fm1_v57.yaml` with two changes: `adsb.sources.adsb_fi: false` and
`adsb.table_dir: data/r12/tables_51348d3e1cb3` (the open table v50 and the MTX build, v51, serve) in place of
`data/r12/tables_51348d3e1cb3_afi`. Every input is the challenge's own files or open, re-fetchable data
(DATA_SOURCES.md); nothing comes from the owner's adsb.fi capture, so unlike v55 and v57 this build does not need
the machine that holds that capture. Board file v59 (2026-10-06) is its build: public board **248.5486**, file
sha256 `7c64bf9fef4d8bae…`. Rebuild check (2026-10-06): after the shipped stack manifest's one absolute path was made
repository-relative, the artefact was re-derived from the v50 donor (`1dc07433357e-ef56e51e` on the author's machine;
a reader's directory name follows its own input digest) and `build-submission --artifact` from it wrote a file
byte-identical to the uploaded v59 (same sha256). The stand-exit table and constants it reads (`modules.uad`) are the same derived inputs v57 uses,
below.

## Reproducing the board file v58

```
python scripts/analysis/blend_exact.py plan --sub-dir <dir with the member files and their result files> --base 55 --top 57 \
    --pool 33,34,35,36,40,41,42,43,44,46,49,50,52,53,55 --lane-file <v57's provenance file> --out plan.json
python scripts/analysis/blend_exact.py build --plan plan.json --sub-dir <same dir> --lane-file <same file> --out merry-quicksand_v58.parquet
```

The uploaded file has sha256 `0764ca6ec531efad95b0943765abb0659a37fe383366669f5cbf760607fa1b59` (344,841 rows); `build`, run with the
committed tool on 2026-10-05, wrote exactly that file. v58 is an average of board files, not an FM1 output (METHOD §8).
It needs the member files (`merry-quicksand_v33.parquet` … `_v57.parquet`) and each one's result file from the scoring
service, none of which is in the repository. FM1 rebuilds v50, v55 and v57; the shipped code does not rebuild
v33–v46, which came from the chain that preceded FM1, so a clone cannot rebuild v58 from source. `plan` prints the
weights and the exact board MSE of the average; `build` refuses if a member, a result file or the tool itself changed
since the plan.

## Determinism (VERIFIED on the e2e world, this machine)

The same config and data fitted twice give bit-identical predictions (LightGBM deterministic + fixed threads and seeds,
CatBoost fixed seed and thread count, HGB fixed seed). A saved artefact reloads bit-exactly (every file's sha256 is in its
manifest and re-checked on load). Cross-machine determinism is NOT claimed.

## Artefacts

`run.artifacts_dir/<config hash[:12]>-<input data digest[:8]>/` — never overwritten; the manifest records the git sha,
library versions, every input file's sha256 and every artefact file's sha256. A submission's manifest records the
sha256 of the scored file, the template, the inputs the fit recorded as they are at build time (training files present,
weather, routes, ADS-B tables, stack models, MTX reference, E2 rules) and the published MTX file, and names every recorded
input whose hash differs from the artefact's fit-time record (`build_inputs`, run before the prediction: a missing training
file or MTX reference is named, not refused; a broken raw directory stops the build at once).

## Derived variants (no retraining)

`fit --reuse-from <donor artefact dir>` builds a variant artefact from a donor's fitted matched learners when only
predict-time or light-fit knobs differ (`knobs.REUSE_FREE_KNOBS`: `run.purpose`, `run.artifacts_dir`, `calibration.mtx.*`,
`modules.*`, `matched.blend.catboost_weight`, `adsb.stack.*`, `adsb.support_gate.*`, `adsb.min_coverage`,
`adsb.sources.adsb_fi`, `adsb.table_dir`, `data.scored_file`, `data.template_file`). Frames, encodings,
the unmatched lane and MTX are recomputed; the recomputed design columns, encodings and row counts must equal the donor's,
and the learners' input files (raw training months, the training months' weather, routes, the ADS-B training table
`train.parquet`) must be the same — otherwise `ReuseError`; a new serve table, a new scored file and weather for months
the learners do not read are allowed (that is how v55 was derived and how a new period is served).
A derived artefact predicts bit-identically to a full fit of the same config (tests/fm1/test_fm1_reuse.py, five variants).

## Runtimes and resources (measured 2026-10-01 / 02, this machine: 18 cores, 24 GiB)

Full fit of the final configuration (twelve raw months, 2,062,440 matched rows, 133 features, `run.threads=12`,
three seeds): wall **9 h 40 min** including 44 pauses for other jobs (LightGBM 5 h 55 min: best_iter 40,550, n_ref 48,691;
CatBoost 3 h 43 min: best_iter 29,993 of `max_rounds` 30,000 — the cap bound, early stopping did not fire; n_ref 36,014).
Artefact 3.5 GB. `build-submission` of the scored file: 18 min, 344,841 rows.
Memory: the data path alone peaks at 9.9 GB (1-round fit; it was 17.8 GB before the 2026-10-01 design / learner refactor).
During the full fit the main process reaches ~15 GB (three 48,691-tree boosters on top of the frame and the design) and
the CatBoost worker runs beside it at ~5 GB: plan for ~20 GB, or run with a memory guard that pauses the fit. A 1-round
dry run does NOT measure the model-size term or the worker.
Determinism at scale: two fits of a three-month slice of the final configuration predicted the full scored file
identically (344,841 × 30 cells, 0 different).
Measured earlier: AT14 stored-tier gate 158 s, peak 2.6 GB (parent + one booster child); AT9 e2e world: `fit` 4 s,
`build-submission` 21 s, parent peak 0.31 GB, two independent runs byte-identical.

## What a clone of the repository can run (2026-10-06)

A clone carries the package (`prc/`), the configs, the code FM1 loads by path (`scripts/`, the E2 rules file), the tools,
the tests and these docs. Without data, artefacts or the project's development record (registrations, reports, logs),
FM1 imports, resolves its configs to the documented hashes and runs its CLI
(`tests/fm1/test_fm1_clean_checkout.py`). The knobs' evidence pointers name files of the development record: they
document why a default was chosen and are checked only in the development tree (`knobs.check_evidence_files`).

The test suite has a **clone tier**: every test not marked `devtree` (tests/conftest.py) runs from a fresh clone —
`python -m pytest -m "not devtree" tests`, or `python tools/clone_check.py`, which builds a tree of exactly the tracked
files in a new git repository and runs that tier there (2026-10-09, on the final-phase tree: 3,585 passed,
76 skipped, 149 deselected, 9 min 7 s on the author's machine; 2026-10-07 before the final phase: 3,556 passed). Tests that read the data, the stored artefacts or the
development record are marked `devtree` (automatically for every test on the real-data-shaped worlds,
tests/fm1/conftest.py) and run in the full suite on the development machine.

To fit or predict, a clone also needs the inputs every path knob names (CONFIG_REFERENCE.md): the raw challenge files and
weather (DATA_SOURCES.md) and the derived inputs below, each rebuilt by code in the repository. Heavy steps run through
`scripts/analysis/r12_launch.py`, which takes the machine-wide lock; its memory guard (`tools/mem_guard.py`, macOS
`memory_pressure`), power check (`pmset`) and reserved-window check run only with `--machine-guards` (the author's own
machine, where the run queue `tools/prc_queue.py` always passes it). A reader runs the launcher without that flag, or
runs the step's runner directly (e.g. `python scripts/analysis/r12_build.py features ...` in place of
`r12_launch.py --step features --detach -- features ...`):

| input (knob) | built by | final configuration |
|---|---|---|
| ADS-B tables (`adsb.table_dir`) | `python scripts/analysis/r12_build.py features --store data/r12/features_51348d3e1cb3 --final` (the default months, all of 2025 plus January and July 2026, and the default three day-file directories; `--final` records missing days once the fetch is complete), then `python scripts/analysis/r12_build.py tables --store data/r12/features_51348d3e1cb3 --out data/r12/tables_51348d3e1cb3` (no `--days`: every day of the universe) | `data/r12/tables_51348d3e1cb3` — served by v50, v51 and the final model (open-data v57); the adsb.fi-merged `data/r12/tables_51348d3e1cb3_afi` (below) is served by v55 and board file v57 |
| holdout fits feeding the stack | the LightGBM base, on the older tables: `python scripts/analysis/r12_validate.py --split holdout --arms r12 --tables data/r12/tables_801f9bcac9a5 --out data/r12/val_801f9bcac9a5_full --seeds 0,1,2 --rss-limit-gib 11`; the CatBoost half, on the final tables (learning rate 0.05): `python scripts/analysis/r12_validate.py --split holdout --arms cat --screen --seeds 0,1,2 --tables data/r12/tables_51348d3e1cb3 --out data/r12/r13_cat_51348d3e1cb3_full --rss-limit-gib 11`; the two assembled into one holdout directory: `python scripts/analysis/r13_cat.py assemble --base-val data/r12/val_801f9bcac9a5_full --cat-val data/r12/r13_cat_51348d3e1cb3_full --out data/r12/r13_arms_full --split holdout` (no `--yhead-val`: the lgbcat stack only). The directory names and the 11 GiB memory cap are the author's (each output's report.json records its inputs) | `data/r12/r13_arms_full` |
| ADS-B stack models (`adsb.stack.models_dir`) | `python scripts/analysis/r12_stack.py oof --val data/r12/r13_arms_full --tables data/r12/tables_51348d3e1cb3 --arms r12,lgbcat --out data/r12/r13_stack_full --rss-limit-gib 11` | `data/r12/r13_stack_full/models_lgbcat` (MANIFEST.json: seeds 0–2, 5 folds, gate, git sha) |
| ORL route table (`features.orl_routes_file`) | `scripts/analysis/orl_build.py fetch`, then `build` | `data/osm/orl_routes.parquet` |
| stand-exit table (`modules.uad.table`; only with the `uad` stage) | `scripts/analysis/uad_build.py parked`, then `table`, on the ADS-B day files (2025 first; the 2026 build takes both years' parked positions and `--alias-from` the 2025 table) | `data/uad/2026_table_a3/table.parquet` |
| stand-exit constants (`modules.uad.constants`) | `scripts/analysis/uad_price.py constants` on the 2025 stand-exit table and leave-one-month-out predictions of the 2025 months (the command refuses a table that fails its registered checks) | `configs/uad_constants_2025.json` (in the repository) |
| MTX CSVs (`calibration.mtx.*_csv`) | downloaded from EUROCONTROL's performance data (`taxi_out_additional_time_<year>.csv`; the URLs are in the manifest) and parsed by `scripts/analysis/mtx_source.py` | `data/mtx/raw/` (sha256 manifest; the 2026 file is a live publication) |

**Optional second source (scored-period build only).** `scripts/analysis/afi_build.py days` merges the owner's adsb.fi
capture of July 2026 into the adsb.lol day files (rule AUN.1, `scripts/analysis/adsb_union.py`; the union directory's
MANIFEST.json declares both sources), `afi_build.py store` copies the R12 store without July 2026, `r12_build features
--sources <union dir> --days <July 2026>` rebuilds those days with the same feature code, and `r12_build tables` writes
tables whose `report.json` declares `["adsb.fi", "adsb.lol"]`. FM1 refuses such tables unless `adsb.sources.adsb_fi=true`
(and the reverse). Only the serve table changes (fit reads `train.parquet` only), so a derived fit reuses the full fit's
learners. The adsb.fi archive is non-commercial, never redistributed and cannot be re-fetched by others: this step is
reproducible only on the machine that holds the capture.

The stack's MLP half trains and predicts in `scripts/adsb_nn_worker.py` (a separate process; torch never shares a process
with LightGBM), and the CatBoost learner runs in `~/.venvs/prc-catboost` (`prc/pipeline/fm1/catboost_worker.py`). A fitted
artefact carries its own copy of the stack models, so predicting with a saved artefact needs only the tables and raw files.

**What HEAD cannot rebuild exactly.** The stored stack (`models_lgbcat`) was trained on a LightGBM holdout built on the
`tables_801f9bcac9a5` tables, whose feature version the current `scripts/adsb_features.py` no longer produces
(`r12_validate` refuses a stamp mismatch), so the stack is a stored input of FM1 rather than something a clone can
regenerate byte for byte. It is shipped in the repository (`data/r12/r13_stack_full/models_lgbcat/`, 46 files: 15
LightGBM text models, 15 torch nets and 15 normalisation-stat files, plus MANIFEST.json; fitted models, no raw
data), so a clone has it already — regenerating it from scratch needs the development record and is not a
documented public path. Its MANIFEST records seeds, folds and gate, and a git sha that predates its producer scripts
(LIMITATIONS 18). The OpenStreetMap route table is fetched live (Overpass has no snapshot date in the query; the manifest
records the OSM base time of the fetch used). The adsb.fi capture cannot be re-fetched (LIMITATIONS 12); the final
model does not need it.

## Running the final model on a new period

To score a period other than January + July 2026 (say October 2026): rebuild the period's derived inputs, point a
copy of `configs/fm1_v57_open.yaml` at them, make a derived artefact from the final model's (no learner is
retrained) and build. Every knob changed below is one that `fit --reuse-from` accepts without retraining
(`knobs.REUSE_FREE_KNOBS`, `prc/pipeline/fm1/knobs.py`; `data.scored_file` and `data.template_file` were added on
2026-10-06 after a test showed the fit never reads them, `tests/fm1/test_fm1_reuse.py`).

1. **The challenge files.** Put the new period's scored departures at `data/raw/ranking.parquet` (keep the old file
   elsewhere) and its submission template where `data.template_file` points. The Rome E2 rule reads the scored
   file under that exact name (its code is hash-frozen), which is also `data.scored_file`'s default.
2. **Caches, then ADS-B.** First the scored file's caches: `python scripts/stand_ab.py --ranking cache` and
   `python scripts/stand_ab.py --ranking order-cache` (`r12_build` reads them for the serve universe). They overwrite
   the January + July 2026 caches `data/cache_stand/ranking.parquet` and `data/cache_order/ranking.parquet`: move those
   aside first to keep them. Then
   `python scripts/adsb_fetch.py index --year <Y>` and `python scripts/adsb_fetch.py run --year <Y> --months <M,...>`
   (with `--months` it writes `data/adsb/v2/y<Y>_other`). Add the new days to the feature store the final
   model's training table came from (or, when the period REPEATS days of an earlier build, to a fresh store seeded with
   the training days — the paragraph after the guards): `python scripts/analysis/r12_build.py features --store <that store> --months
   <spec> --sources data/adsb/v2,data/adsb/v2/y2025_other,data/adsb/v2/y2026,data/adsb/v2/y<Y>_other --final`
   (always pass `--sources` listing the new directory: the default lists only the three directories named there).
   Then rebuild the tables WITHOUT `--days`: `python scripts/analysis/r12_build.py tables --store <that store>
   --out data/r12/tables_<new>`. Its `train.parquet` is assembled from the 2025 training days exactly as before
   (`fit --reuse-from` compares its hash with the donor's), and `serve.parquet` holds the new scored file's days.
   With `--days` limited to the new period no `train.parquet` is written and the derived fit stops. `--months`
   syntax (`prc/months.py`): `2026-10`, or an inclusive range `2026-08..2026-10`, comma-separated.
3. **Weather.** `python scripts/fetch_weather.py --months <spec>`.
4. **Stand-exit table** (the `uad` stage), the two passes as they were run for 2026:
   `python scripts/analysis/uad_build.py parked --rows data/raw/ranking.parquet --adsb-dir data/adsb/v2/y<Y>_other
   --out data/uad/<period>_parked`, then `python scripts/analysis/uad_build.py table --rows data/raw/ranking.parquet
   --adsb-dir data/adsb/v2/y<Y>_other --parked data/uad/2025_parked data/uad/<period>_parked --alias-from
   data/uad/2025_table_a3 --out data/uad/<period>_table` (the 2026 table was built with `--adsb-dir
   data/adsb/v2/y2026 data/adsb/v2/y2025_other --parked data/uad/2025_parked data/uad/2026_parked`); the stage's
   offsets and weights (`configs/uad_constants_2025.json`) are fixed and are not refitted.
5. **MTX.** If EUROCONTROL has published the period's year file, set `calibration.mtx.published_csv` to it; an
   airport-month with no published value in it is left unchanged, and MTX calibrates one scored year at a time (a
   period spanning two years must be built per year). If no file exists yet for that year, set
   `calibration.mtx.enabled: off` (the file is a required input while MTX is on: the dry run and the build refuse a
   missing one).
6. **Config and build.** In a copy of `configs/fm1_v57_open.yaml` set `adsb.table_dir: data/r12/tables_<new>`,
   `modules.uad.table` to the new stand-exit table and, if changed, `calibration.mtx.published_csv` and
   `data.template_file`. Then
   `python -m prc fit --config <copy>.yaml --allow-heavy --reuse-from <the final model's artefact directory>` (minutes: the matched
   learners are the donor's) and `python -m prc build-submission --config <copy>.yaml --out <dir> --version <N>`.

**Guards a reader will meet.** The build refuses when more than 0.1 % of the scored rows have no row in the ADS-B
table (`adsb.serve_missing_max_share`, LIMITATIONS 10): fetch the missing days and rebuild with `--final`. The E2
check refuses when `data/raw/ranking.parquet` is missing or is not `data.scored_file`.

**Verified end-to-end on 2026-10-08** (the final phase below): the path ran on the organiser's `final_ranking.parquet` and
produced the final submission. Two things the first run taught, now part of the path: (1) `r12_build.py` takes the
heavy-compute lock itself for a real store, so it is run directly or through `scripts/analysis/r12_launch.py`, never
inside another lock holder; (2) **a period that repeats days of an earlier build with a changed flight set** (the final
file's January/July rows carry the organiser's ±12 s take-off dither; 31 flights moved across UTC midnight) is refused by
the store's per-day flight-set guard — build it in a FRESH store seeded with the training days (copy the `day_2025-*`
files and `STAMP.json` into a new directory and point `--store` at it); the training table assembled from the copied
days is byte-identical, which `fit --reuse-from` checks.

## Final phase (2026-10-08): the final model on `final_ranking.parquet`

On 2026-10-08 the organiser opened a final phase: one blind submission per team on `final_ranking.parquet` (January,
February, June and July 2026; 670,790 departures; the January and July rows are the earlier scored rows with the take-off
time dithered by ±12 s on 91.9 % of rows), named `<team-name>_final.parquet`, due 2026-10-14T10:00Z, ranked on the initial
two months, the two new months and all four; "the design of the model shouldn't change". The final model
(`configs/fm1_v57_open.yaml`) was run on the new period by the section above, nothing else. What ran, in order, with the
evidence each step left (times EDT, this machine):

| step | command | evidence |
|---|---|---|
| files | `final_ranking.parquet` (sha256 `4b0692a7…`, 85,853,399 bytes, bucket 2026-10-08T11:04Z) and `final_submitting.parquet` (`b3445d80…`) fetched to `data/raw_final/` with a manifest | 08:45 |
| ADS-B days | `python scripts/adsb_fetch.py index --year 2026`; `python scripts/adsb_fetch.py run --year 2026 --months 2,6` → `data/adsb/v2/y2026_other/` | 58 days ok, 0 failed, `unparseable_members` 0 on every day; 160–245 s per day; a re-fetch of February was byte-identical to the first (28/28). 2026-06-11 is a partial upstream archive (328,510 samples against ~1.7 M on its neighbours) |
| scored file | `data/raw/ranking.parquet` and `submitting.parquet` = the final files while building (the Rome E2 rule reads the scored file by that name); the January + July files and their caches set aside and restored afterwards | sha256 of the copies checked |
| caches | `python scripts/stand_ab.py --ranking cache`; `python scripts/stand_ab.py --ranking order-cache` | 661,713 rows each = the matched departures (those with an AOBT_3) of the final file; 11.5 s, 1.9 GB |
| weather | `python scripts/fetch_weather.py --months 2026-02,2026-06` | 20 files (10 airports × 2 months) |
| weather cache | `python scripts/stand_ab.py weather-cache` and `python scripts/stand_ab.py --ranking weather-cache` after moving the old cache aside (the cache manifest pins the archive digest, which the two new months changed; the archive's row-count pins in `tests/test_weather_parser.py` were updated the same way) | manifest digest `17a86452…` |
| features | `python scripts/analysis/r12_build.py features --store data/r12/features_final --months 2025-01..2025-12,2026-01,2026-02,2026-06,2026-07 --sources data/adsb/v2,data/adsb/v2/y2025_other,data/adsb/v2/y2026,data/adsb/v2/y2026_other --final --rss-limit-gib 5` (fresh store seeded with the 365 `day_2025-*` files) | 485 universe days: built 120, done 365, missing 0; 1,098 s; peak RSS 2.96 GiB |
| tables | `python scripts/analysis/r12_build.py tables --store data/r12/features_final --out data/r12/tables_final --rss-limit-gib 5` | `train.parquet` sha256 identical to `tables_51348d3e1cb3/train.parquet` (2,062,440 rows); `serve.parquet` 661,713 rows, 0 on missing days; sources adsb.lol only; 8 s |
| stand-exit table | `python scripts/analysis/uad_build.py parked --rows data/raw/ranking.parquet --adsb-dir data/adsb/v2/y2026 data/adsb/v2/y2026_other data/adsb/v2/y2025_other --out data/uad/final_parked`; `python scripts/analysis/uad_build.py table --rows data/raw/ranking.parquet --adsb-dir data/adsb/v2/y2026 data/adsb/v2/y2026_other data/adsb/v2/y2025_other --parked data/uad/2025_parked data/uad/final_parked --alias-from data/uad/2025_table_a3 --out data/uad/final_table_a3` | report params `d196b652a9a7`, sources adsb.lol only, 670,790 rows (every departure); 421 s + 355 s, ≤ 2.7 GB |
| config | `configs/fm1_final.yaml` = `fm1_v57_open.yaml` with `adsb.table_dir: data/r12/tables_final`, `modules.uad.table: data/uad/final_table_a3/table.parquet`, `data.template_file: data/raw_final/final_submitting.parquet` (all in `knobs.REUSE_FREE_KNOBS`; `knobs.reuse_blockers` empty against the final model) | config hash `2950c7552ed7…` |
| dry run, fit, build | `python -m prc validate --config configs/fm1_final.yaml --protocol dry-run`; `python -m prc fit --config configs/fm1_final.yaml --allow-heavy --reuse-from artifacts/fm1/1dc07433357e-ef56e51e`; `python -m prc build-submission --config configs/fm1_final.yaml --out submissions/final --version 1` | artefact `2950c7552ed7-12679217` (matched learners reused; the unmatched lane and MTX recomputed; 137 s); `merry-quicksand_v1.parquet` sha256 `a514c644…`, 670,790 rows, no input changed since the fit; 166 s, peak footprint 8.6 GB |
| checks | label-free, by session scripts kept with the run record (not shipped): rows = template, ids in the template's order, no null, every value ≥ 1, int32, manifest read; `decided_by` by month; the 344,841 January + July rows joined to `merry-quicksand_v59.parquet` by id (provenance files) | ±12 s on 84.9 % of rows, 60 s on 99.71 %, mean +0.06 s, 59 rows changed lane; monthly means January 1079.3 = v59, July 994.9 vs 994.8 |
| upload | `python tools/verify_upload.py --final --file submissions/final/merry-quicksand_v1.parquet` (dry run; `--upload` on the owner's go): key `merry-quicksand_final.parquet`, the same bytes | RISK_REPORT_final.md |

The build is label-free throughout (the final file carries no taxi-out times; no board exists for it); the dither is the
organiser's and is left as it is. The ADS-B stack decided 42 / 39 / 40 / 40 % of the rows of January / February / June /
July: adsb.lol coverage of the new months matches the scored months'. Per-month means: January 1079, February 988, June
985, July 995 s.
