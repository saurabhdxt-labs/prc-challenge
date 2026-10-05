# FM1 — reproduction guide (updated 2026-10-04: final configs, environment pins, clone tier)

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

`configs/fm1.yaml` holds every knob at its documented default (a test pins this). The final runs are three config files
that write their command-line overrides in and resolve to the stored artefacts' config hashes
(`tests/fm1/test_fm1_final_configs.py`): `configs/fm1_v50.yaml`, the forecaster (lgbcat ADS-B stack on the
`tables_51348d3e1cb3` tables, storm anchor on, LDI on, 12 threads), and `configs/fm1_v55.yaml`, the scored-period build
(v50 + MTX + the adsb.fi serve table; `adsb.sources.adsb_fi`). `configs/fm1_v57.yaml` is board file v57, the team's
best: v55 with the optional stand-exit stage on (`modules.uad.enabled`). `configs/fm1_v57_tc.yaml` is a candidate that is
not uploaded on its own: v57 with the unmatched body's tail-cell rule off (`modules.tail_cells.enabled` false). The storm-as-feature form is refused by name when
switched on. LDI needs whole local days per predict call. A fit
over the twelve months is a heavy job (`--allow-heavy`, the pre-run checks first, one heavy job at a time). Validation folds
with a single training month must switch E2W off (`--set modules.rome.e2w=false`: its beta_W is fitted leave-one-month-out).

## Reproducing the board file v55

```
python -m prc fit --config configs/fm1_v50.yaml --allow-heavy                                     # the donor (~10 h)
python -m prc fit --config configs/fm1_v55.yaml --allow-heavy --reuse-from artifacts/fm1/2d29ad344b45-1c61fcc2
python -m prc build-submission --config configs/fm1_v55.yaml --out <new dir> --version 55
```

The uploaded file is `merry-quicksand_v55.parquet`, sha256 `225ab0be205a0a254d0c27e726ec33d6007981b98f937f9b00739499e5ac537d`
(344,841 rows). The rebuild check — the current code serving the stored v55 artefact
(`artifacts/fm1/d811c69dc93e-141862a0`) — is recorded below.

**Rebuild check (VERIFIED 2026-10-04 05:38–05:56 EDT, with the committed code).** `python -m prc build-submission
--config configs/fm1_v55.yaml --out <new dir> --version 55` served the stored artefact and wrote a file with sha256
`225ab0be205a0a254d0c27e726ec33d6007981b98f937f9b00739499e5ac537d` — identical to the uploaded v55. Its manifest hashes
204 inputs and names none as changed since the fit. The full fit itself (~10 h) was not re-run for this check.

## Reproducing the board file v57

```
python -m prc fit --config configs/fm1_v57.yaml --allow-heavy --reuse-from artifacts/fm1/2d29ad344b45-1c61fcc2
python -m prc build-submission --config configs/fm1_v57.yaml --out <new dir> --version 57
```

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
`adsb.sources.adsb_fi`, `adsb.table_dir`). Frames, encodings,
the unmatched lane and MTX are recomputed; the recomputed design columns, encodings and row counts must equal the donor's,
and the learners' input files (raw months, weather, routes, the ADS-B training table `train.parquet`) must be the
same — otherwise `ReuseError`; a new serve table is allowed (that is how v55 was derived).
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

## What a clone of the repository can run (2026-10-04)

A clone carries the package (`prc/`), the configs, the code FM1 loads by path (`scripts/`, the E2 rules file), the tools,
the tests and these docs. Without data, artefacts or the project's development record (registrations, reports, logs),
FM1 imports, resolves its configs to the documented hashes and runs its CLI
(`tests/fm1/test_fm1_clean_checkout.py`). The knobs' evidence pointers name files of the development record: they
document why a default was chosen and are checked only in the development tree (`knobs.check_evidence_files`).

The test suite has a **clone tier**: every test not marked `devtree` (tests/conftest.py) runs from a fresh clone —
`python -m pytest -m "not devtree" tests`, or `python tools/clone_check.py`, which builds a tree of exactly the tracked
files in a new git repository and runs that tier there (2026-10-04, the committed files: 2,887 passed, 76 skipped where
an optional local file is absent, 138 devtree tests deselected). Tests that read the data, the stored artefacts or the
development record are marked `devtree` (automatically for every test on the real-data-shaped worlds,
tests/fm1/conftest.py) and run in the full suite on the development machine.

To fit or predict, a clone also needs the inputs every path knob names (CONFIG_REFERENCE.md): the raw challenge files and
weather (DATA_SOURCES.md) and the derived inputs below, each rebuilt by code in the repository (heavy steps run through
`scripts/analysis/r12_launch.py`, which takes the machine-wide lock and a memory guard):

| input (knob) | built by | final configuration |
|---|---|---|
| ADS-B tables (`adsb.table_dir`) | `scripts/analysis/r12_build.py features`, then `tables` | `data/r12/tables_51348d3e1cb3` |
| holdout fits feeding the stack | `scripts/analysis/r12_validate.py` (LightGBM base on the older `tables_801f9bcac9a5`; CatBoost half via `scripts/analysis/r12_validate.py --arms cat --screen --seeds 0,1,2`, learning rate 0.05), assembled by `r13_cat.py assemble` | `data/r12/r13_arms_full` |
| ADS-B stack models (`adsb.stack.models_dir`) | `scripts/analysis/r12_stack.py oof --val data/r12/r13_arms_full --tables data/r12/tables_51348d3e1cb3 --arms r12,lgbcat --out data/r12/r13_stack_full` | `data/r12/r13_stack_full/models_lgbcat` (MANIFEST.json: seeds 0–2, 5 folds, gate, git sha) |
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
regenerate byte for byte; its MANIFEST records seeds, folds and gate, and a git sha that predates its producer scripts
(LIMITATIONS 18). The OpenStreetMap route table is fetched live (Overpass has no snapshot date in the query; the manifest
records the OSM base time of the fetch used). The adsb.fi capture cannot be re-fetched (LIMITATIONS 12).

