"""CatBoost fold worker for Amendment 15.1 (H-15a): the second learner family on the SAME design
matrix, target and fold masks that scripts/lgbm_fold.py measures LightGBM on.

    ~/.venvs/prc-catboost/bin/python scripts/catboost_fold_worker.py \\
        --input data/cache_stand/catboost_fold_input.npz \\
        --output data/cache_stand/catboost_fold_preds.npy \\
        [--depth 8 --learning-rate 0.03 --iterations 10000 --od-wait 200 --seed 0 --threads 4]

This file runs ONLY under the isolated venv: CatBoost needs numpy < 2, the global interpreter
is numpy 2.4 and its site-packages are shared with a live fleet, so catboost is never installed
globally and never imported by lgbm_fold.py (which launches this script as a subprocess and
checks its exit code and output shape). The import is inside the fitting function so the
argument parsing and the input-contract validation are testable in the main interpreter.

Input contract (written by lgbm_fold.write_worker_input): a .npz with
    X        float32 (n, p)   the design matrix, every row of the fold
    target   float64 (n,)     delta on the training rows; NaN on the holdout rows (the holdout
                              targets never leave the parent - nothing here can read them)
    train    bool (n,)        the ten training months
    fit      bool (n,)        training minus the early-stopping months
    es       bool (n,)        the early-stopping months (3, 9), carved out of training
    holdout  bool (n,)        the rows to predict (months 1, 7)

Procedure: CatBoostRegressor(depth, learning_rate, iterations, loss RMSE, random_seed,
thread_count) early-stopped on the ES rows (od_type Iter, od_wait) -> best iteration; then a
refit on ALL training rows at that iteration count (best + 1 trees, no data-ratio scaling -
the task's literal rule, recorded in the sidecar); predictions for the holdout rows -> .npy
(float64) plus a .json sidecar next to it (best iteration, refit count, row counts, walls,
peak RSS, versions). Peak RSS is printed on exit.
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import resource
import sys
import time

import numpy as np

DEFAULTS = dict(depth=8, learning_rate=0.03, iterations=10_000, od_wait=200, seed=0, threads=4)
INPUT_KEYS = ("X", "target", "train", "fit", "es", "holdout")
LOG_PERIOD = 500

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.0f}s] {msg}", flush=True)


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="the .npz written by lgbm_fold.write_worker_input")
    ap.add_argument("--output", required=True, help="where the holdout predictions (.npy) go; the sidecar "
                                                    "json is written next to it")
    ap.add_argument("--depth", type=int, default=DEFAULTS["depth"])
    ap.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    ap.add_argument("--iterations", type=int, default=DEFAULTS["iterations"], help="upper bound; early stopping decides")
    ap.add_argument("--od-wait", type=int, default=DEFAULTS["od_wait"], help="early-stopping patience (od_type Iter)")
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--threads", type=int, default=DEFAULTS["threads"])
    args = ap.parse_args(argv)
    for name in ("depth", "iterations", "od_wait", "threads"):
        if getattr(args, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be >= 1")
    if not 0.0 < args.learning_rate <= 1.0:
        ap.error("--learning-rate must be in (0, 1]")
    return args


def load_input(path) -> dict:
    """Read and validate the .npz contract; returns {key: array}."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input {path} missing")
    with np.load(path) as z:
        missing = [k for k in INPUT_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"{path.name}: missing arrays {missing}; expected {list(INPUT_KEYS)}")
        d = {k: z[k] for k in INPUT_KEYS}
    X = d["X"]
    if X.ndim != 2 or X.dtype != np.float32:
        raise ValueError(f"X must be a float32 matrix, got {X.dtype} with shape {X.shape}")
    n = X.shape[0]
    for k in ("target", "train", "fit", "es", "holdout"):
        if d[k].shape != (n,):
            raise ValueError(f"{k} has shape {d[k].shape}, expected ({n},)")
    for k in ("train", "fit", "es", "holdout"):
        if d[k].dtype != np.bool_:
            raise ValueError(f"{k} must be boolean, got {d[k].dtype}")
    if d["target"].dtype != np.float64:
        raise ValueError(f"target must be float64, got {d['target'].dtype}")
    train, fit, es, hold = d["train"], d["fit"], d["es"], d["holdout"]
    if (train & hold).any():
        raise ValueError("training and holdout rows overlap")
    if (fit & es).any():
        raise ValueError("fit and early-stopping rows overlap")
    if not np.array_equal(fit | es, train):
        raise ValueError("fit + early-stop rows do not make up the training rows")
    for k in ("fit", "es", "holdout"):
        if not d[k].any():
            raise ValueError(f"no {k} rows")
    if not np.isfinite(d["target"][train]).all():
        raise ValueError("non-finite target on training rows")
    return d


def _rmse(a, b) -> float:
    return float(np.sqrt(((np.asarray(a, dtype="float64") - np.asarray(b, dtype="float64")) ** 2).mean()))


def fit_and_predict(X, target, train, fit, es, holdout, params: dict, log=log):
    """Early stop on `es`, refit on `train` at the found count, predict `holdout`.

    Returns (predictions float64 over the holdout rows, info dict for the sidecar)."""
    from catboost import CatBoostRegressor, __version__ as catboost_version

    common = dict(depth=int(params["depth"]), learning_rate=float(params["learning_rate"]), loss_function="RMSE",
                  random_seed=int(params["seed"]), thread_count=int(params["threads"]),
                  allow_writing_files=False, verbose=LOG_PERIOD)
    n_fit, n_es, n_train, n_hold = (int(m.sum()) for m in (fit, es, train, holdout))
    log(f"catboost {catboost_version} (numpy {np.__version__}): early stopping on {n_es:,} rows, fit on "
        f"{n_fit:,} rows, depth {common['depth']}, lr {common['learning_rate']}, up to "
        f"{int(params['iterations']):,} iterations, od_wait {int(params['od_wait'])}, threads {common['thread_count']}")
    t = time.time()
    model = CatBoostRegressor(iterations=int(params["iterations"]), od_type="Iter", od_wait=int(params["od_wait"]),
                              use_best_model=True, **common)
    model.fit(X[fit], target[fit], eval_set=(X[es], target[es]))
    best = int(model.get_best_iteration())
    if best < 0:
        raise RuntimeError(f"best iteration {best}: early stopping found nothing")
    n_refit = best + 1
    es_rmse = _rmse(target[es], model.predict(X[es]))
    es_wall = time.time() - t
    log(f"best iteration {best:,} -> refit count {n_refit:,}; stopping-set RMSE (delta) {es_rmse:.3f} "
        f"[optimistic by construction; NOT a fold result]   {es_wall:.0f}s   peak RSS {peak_rss_gb():.2f} GB")
    del model
    gc.collect()

    t = time.time()
    refit = CatBoostRegressor(iterations=n_refit, **common)
    refit.fit(X[train], target[train])
    preds = np.asarray(refit.predict(X[holdout]), dtype="float64")
    refit_wall = time.time() - t
    if preds.shape != (n_hold,) or not np.isfinite(preds).all():
        raise RuntimeError(f"refit produced {preds.shape} predictions, {int((~np.isfinite(preds)).sum())} non-finite")
    log(f"refit on all {n_train:,} training rows: {int(refit.tree_count_):,} trees in {refit_wall:.0f}s; "
        f"predicted {n_hold:,} holdout rows   peak RSS {peak_rss_gb():.2f} GB")
    info = dict(best_iteration=best, n_refit=n_refit, tree_count=int(refit.tree_count_), n_fit=n_fit, n_es=n_es,
                n_train=n_train, n_holdout=n_hold, es_rmse_delta_NOT_A_RESULT=es_rmse,
                es_wall_s=round(es_wall, 1), refit_wall_s=round(refit_wall, 1),
                refit_rule="best iteration + 1 trees on all training rows, no data-ratio scaling",
                params={k: (float(v) if k == "learning_rate" else int(v)) for k, v in params.items()},
                catboost_version=catboost_version, numpy_version=np.__version__, python=sys.executable)
    return preds, info


def main(argv=None) -> int:
    args = parse_args(argv)
    params = dict(depth=args.depth, learning_rate=args.learning_rate, iterations=args.iterations,
                  od_wait=args.od_wait, seed=args.seed, threads=args.threads)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    d = load_input(args.input)
    log(f"loaded {args.input}: X {d['X'].shape[0]:,} x {d['X'].shape[1]} float32 ({d['X'].nbytes / 1e9:.2f} GB); "
        f"peak RSS {peak_rss_gb():.2f} GB")
    preds, info = fit_and_predict(d["X"], d["target"], d["train"], d["fit"], d["es"], d["holdout"], params)
    del d
    gc.collect()
    np.save(out, preds)
    info.update(input=str(args.input), output=str(out), wall_s=round(time.time() - started, 1),
                peak_rss_gb=round(peak_rss_gb(), 3))
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    log(f"wrote {len(preds):,} holdout predictions -> {out} (+ sidecar {out.with_suffix('.json').name}); "
        f"wall {info['wall_s']:.0f}s; peak RSS {info['peak_rss_gb']:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
