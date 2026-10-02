# FM1 — reproduction guide (updated 2026-10-02 after the first full fit)

Interpreter: `/opt/homebrew/bin/python3.11` (pandas 3.0.1, numpy 2.4.3, lightgbm 4.5.0, sklearn 1.8.0). CatBoost 1.2.10 lives
only in `~/.venvs/prc-catboost` (numpy 1.26.4) and is used through the worker. Plain `python3` in this shell is the CatBoost
venv — always call the interpreter above.

## Commands

```
python -m prc config-reference --check                                  # the reference doc is current
python -m prc validate --config configs/fm1.yaml --protocol dry-run     # config, plan, every input path; fits nothing
python -m prc fit --config configs/fm1.yaml [--set path=value ...] [--allow-heavy]
python -m prc fit --config configs/fm1.yaml --reuse-from <donor artefact dir> --set <allow-listed knob>=<value> [--allow-heavy]
python -m prc predict --config configs/fm1.yaml --rows <raw scored parquet> --out <file.parquet>
python -m prc build-submission --config configs/fm1.yaml --out <dir> --version <N> [--set run.purpose=scored_period]
python -m prc validate --config configs/fm1.yaml --protocol S|swap|forward|lomo|forward_monthly --out <dir> [--allow-heavy]
python -m prc compare --a <oof_A_F1> <oof_A_F2> --b <oof_B_F1> <oof_B_F2> --out <file.json>   # pooled, paired
python -m prc blend --a <oof_A ...> --b <oof_B ...> --w 0.5 --out <blend.parquet>                # arm A2 offline
/opt/homebrew/bin/python3.11 -B tools/fm1_replay_gate.py --out <new dir> [--per-stratum 100 --runs 2]   # AT14
/opt/homebrew/bin/python3.11 -B tools/run_suite.py                                                     # the suite
```

`configs/fm1.yaml` states the documented forecaster with every component ported (Amendment A3): the v26 unmatched body, the
Rome module (owner decision 1: on) and ARC at LIRF; the storm anchor is off (owner decision 2). The v36 replay switches it on
(`--set modules.storm.mode=anchor`); LDI is optional and off (`--set modules.ldi.enabled=true`; it needs the whole scored
file per predict call); adsb.fi and the storm-as-feature form are refused by name when switched on. A fit
over the twelve months is a heavy job (`--allow-heavy`, the quality gate first, one heavy job at a time). Validation folds
with a single training month must switch E2W off (`--set modules.rome.e2w=false`: its beta_W is fitted leave-one-month-out).

## Determinism (VERIFIED on the e2e world, this machine)

The same config and data fitted twice give bit-identical predictions (LightGBM deterministic + fixed threads and seeds,
CatBoost fixed seed and thread count, HGB fixed seed). A saved artefact reloads bit-exactly (every file's sha256 is in its
manifest and re-checked on load). Cross-machine determinism is NOT claimed.

## Artefacts

`run.artifacts_dir/<config hash[:12]>-<input data digest[:8]>/` — never overwritten; the manifest records the git sha,
library versions, every input file's sha256 and every artefact file's sha256.

## Derived variants (no retraining)

`fit --reuse-from <donor artefact dir>` builds a variant artefact from a donor's fitted matched learners when only
predict-time or light-fit knobs differ (`knobs.REUSE_FREE_KNOBS`: `run.purpose`, `run.artifacts_dir`, `calibration.mtx.*`,
`modules.*`, `matched.blend.catboost_weight`, `adsb.stack.*`, `adsb.support_gate.*`, `adsb.min_coverage`). Frames, encodings,
the unmatched lane and MTX are recomputed; the recomputed design columns, encodings and row counts must equal the donor's,
and the learners' input files (raw months, weather, routes, ADS-B tables) must be the same — otherwise `ReuseError`.
A derived artefact predicts bit-identically to a full fit of the same config (tests/fm1/test_fm1_reuse.py, five variants).

## Runtimes and resources (measured 2026-10-01 / 02, this machine: 18 cores, 24 GB)

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
`build-submission` 21 s, parent peak 0.31 GB, two independent runs byte-identical (notes N7).

## What a clone of the repository can run (2026-10-02)

A clone carries the package (`prc/`), the config (`configs/fm1.yaml`), the code FM1 loads by path (`scripts/`, the E2
rules file) and these docs; it never carries data, artefacts or the project's development record (registrations under
`plans/`, `logs/`, `reports/`). From a clone, FM1 imports, resolves its config to the documented hash and runs its CLI
(`tests/fm1/test_fm1_clean_checkout.py` checks exactly this in a temporary tree). The knobs' evidence pointers name files
of the development record: they document why a default was chosen and are checked only in the development tree
(`knobs.check_evidence_files`). The test suite likewise reads that record (registered results, golden files) and the data;
it is a development-tree tool.

To fit or predict, a clone also needs the inputs every path knob names (CONFIG_REFERENCE.md): the raw challenge files and
weather (DATA_SOURCES.md) and the derived inputs below, each rebuilt by code in the repository (heavy steps run through
`scripts/analysis/r12_launch.py`, which takes the machine-wide lock and a memory guard):

| input (knob) | built by | final configuration |
|---|---|---|
| ADS-B tables (`adsb.table_dir`) | `scripts/analysis/r12_build.py features`, then `tables` | `data/r12/tables_51348d3e1cb3` |
| holdout fits feeding the stack | `scripts/analysis/r12_validate.py` (LightGBM base; CatBoost half via `scripts/analysis/r13_cat.py`), assembled by `r13_cat.py assemble` | `data/r12/r13_arms_full` |
| ADS-B stack models (`adsb.stack.models_dir`) | `scripts/analysis/r12_stack.py oof --val data/r12/r13_arms_full --tables data/r12/tables_51348d3e1cb3 --arms r12,lgbcat --out data/r12/r13_stack_full` | `data/r12/r13_stack_full/models_lgbcat` (MANIFEST.json: seeds 0–2, 5 folds, gate, git sha) |
| ORL route table (`features.orl_routes_file`) | `scripts/analysis/orl_build.py fetch`, then `build` | `data/osm/orl_routes.parquet` |
| MTX CSVs (`calibration.mtx.*_csv`) | `scripts/analysis/mtx_source.py` | `data/mtx/raw/` (sha256 manifest) |

The stack's MLP half trains and predicts in `scripts/adsb_nn_worker.py` (a separate process; torch never shares a process
with LightGBM), and the CatBoost learner runs in `~/.venvs/prc-catboost` (`prc/pipeline/fm1/catboost_worker.py`). A fitted
artefact carries its own copy of the stack models, so predicting with a saved artefact needs only the tables and raw files.
