# FM1 — reproduction guide (days 1–3 state)

Interpreter: `/opt/homebrew/bin/python3.11` (pandas 3.0.1, numpy 2.4.3, lightgbm 4.5.0, sklearn 1.8.0). CatBoost 1.2.10 lives
only in `~/.venvs/prc-catboost` (numpy 1.26.4) and is used through the worker. Plain `python3` in this shell is the CatBoost
venv — always call the interpreter above.

## Commands

```
python -m prc config-reference --check                                  # the reference doc is current
python -m prc validate --config configs/fm1.yaml --protocol dry-run     # config, plan, every input path; fits nothing
python -m prc fit --config configs/fm1.yaml [--set path=value ...] [--allow-heavy]
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

## Runtimes and resources — TODO (day 8–9)

Full-tier fit, validation and clean-checkout rebuild times, disk and peak RAM, and the expected hashes.
Measured so far: AT14 stored-tier gate 158 s, peak 2.6 GB (parent + one booster child); AT9 e2e world: `fit` 4 s,
`build-submission` 21 s, parent peak 0.31 GB, two independent runs byte-identical (notes N7).
NOT measured: the full-scale fit (twelve raw months). The design is held as float64 (a 2 M x ~130 copy is ~2 GB on top of
the frames); run a 1-step dry run under the heavy lock and the quality gate before the first real fit.
