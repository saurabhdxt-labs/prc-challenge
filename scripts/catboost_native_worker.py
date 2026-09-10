"""Native-categorical CatBoost worker for plans/PREREG_catboost_native_2026_09_10.md (arms C_delta, C_sched).

    ~/.venvs/prc-catboost/bin/python scripts/catboost_native_worker.py \\
        --input data/cache_stand/catboost_native_input.parquet --target delta \\
        --output data/cache_stand/catboost_native_delta.npy [--learning-rate 0.15 --ctr 2 ...]

Runs ONLY under the isolated venv (CatBoost needs numpy < 2; never installed globally). The catboost
import is inside fit_and_predict, so the input contract and the refit rule are testable in the main
interpreter. Column lists come from the sidecar `<input>.cols.json` written by catboost_native.py.

Input contract: a parquet with the numeric columns (float32), the categorical columns (strings, no
nulls), the target column (float64), and boolean masks train / fit / es / holdout with
fit | es == train, fit & es empty, train & holdout empty. Procedure: early stop on `es` (od_type Iter)
-> best iteration; refit on ALL `train` rows with n_ref = round(best_iter x n_train / n_fit) - arm F's
rule; predict `holdout` -> .npy (float64) + .json sidecar.
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

DEFAULTS = dict(depth=8, learning_rate=0.15, iterations=12_000, od_wait=200, seed=0, threads=6, ctr=2)
MASKS = ("train", "fit", "es", "holdout")
LOG_PERIOD = 250
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.0f}s] {msg}", flush=True)


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def refit_count(best_iter: int, n_train: int, n_fit: int) -> int:
    """Arm F's refit rule: the early-stopped count scaled by the training/fit row ratio."""
    if best_iter < 1 or n_fit < 1 or n_train < n_fit:
        raise ValueError(f"refit_count: best_iter {best_iter}, n_train {n_train}, n_fit {n_fit}")
    return int(round(best_iter * n_train / n_fit))


def load_input(path, num, cat, target) -> dict:
    """Read and validate the parquet contract; returns {X (DataFrame num+cat), target, masks...}."""
    import pandas as pd
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"input {path} missing")
    d = pd.read_parquet(path, columns=list(num) + list(cat) + [target] + list(MASKS))
    for c in num:
        if d[c].dtype != np.float32:
            raise ValueError(f"numeric column {c} must be float32, got {d[c].dtype}")
    for c in cat:
        if not d[c].map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"categorical column {c} must be strings with no nulls")
    for m in MASKS:
        if d[m].dtype != bool:
            raise ValueError(f"mask {m} must be boolean, got {d[m].dtype}")
    tr, fit, es, ho = (d[m].to_numpy() for m in MASKS)
    if (tr & ho).any():
        raise ValueError("training and holdout rows overlap")
    if (fit & es).any():
        raise ValueError("fit and early-stopping rows overlap")
    if not np.array_equal(fit | es, tr):
        raise ValueError("fit + early-stop rows do not make up the training rows")
    for name, m in (("fit", fit), ("es", es), ("holdout", ho)):
        if not m.any():
            raise ValueError(f"no {name} rows")
    y = d[target].to_numpy(dtype="float64")
    if not np.isfinite(y[tr]).all():
        raise ValueError(f"non-finite {target} on training rows")
    return {"X": d[list(num) + list(cat)], "target": y, "train": tr, "fit": fit, "es": es, "holdout": ho}


def fit_and_predict(d: dict, cat, params: dict, log=log):
    from catboost import CatBoostRegressor, Pool, __version__ as cbv
    X, y, tr, fit, es, ho = d["X"], d["target"], d["train"], d["fit"], d["es"], d["holdout"]
    common = dict(depth=int(params["depth"]), learning_rate=float(params["learning_rate"]), loss_function="RMSE",
                  random_seed=int(params["seed"]), thread_count=int(params["threads"]),
                  max_ctr_complexity=int(params["ctr"]), allow_writing_files=False, verbose=LOG_PERIOD)
    n_fit, n_es, n_tr, n_ho = (int(m.sum()) for m in (fit, es, tr, ho))
    log(f"catboost {cbv}: early stop on {n_es:,} rows, fit {n_fit:,}; {X.shape[1]} columns ({len(cat)} categorical); "
        f"depth {common['depth']} lr {common['learning_rate']} ctr {common['max_ctr_complexity']} <= {params['iterations']:,} iters")
    t = time.time()
    m = CatBoostRegressor(iterations=int(params["iterations"]), od_type="Iter", od_wait=int(params["od_wait"]),
                          use_best_model=True, **common)
    m.fit(Pool(X[fit], y[fit], cat_features=list(cat)), eval_set=Pool(X[es], y[es], cat_features=list(cat)))
    best = int(m.get_best_iteration())
    es_wall = time.time() - t
    n_ref = refit_count(best + 1, n_tr, n_fit)
    log(f"best iteration {best:,}; refit count {n_ref:,} (arm F's rule)   {es_wall:.0f}s   peak RSS {peak_rss_gb():.2f} GB")
    del m
    gc.collect()
    t = time.time()
    r = CatBoostRegressor(iterations=n_ref, **common)
    r.fit(Pool(X[tr], y[tr], cat_features=list(cat)))
    pred = np.asarray(r.predict(Pool(X[ho], cat_features=list(cat))), dtype="float64")
    if pred.shape != (n_ho,) or not np.isfinite(pred).all():
        raise RuntimeError(f"refit produced {pred.shape} predictions, {int((~np.isfinite(pred)).sum())} non-finite")
    refit_wall = time.time() - t
    log(f"refit {int(r.tree_count_):,} trees on {n_tr:,} rows in {refit_wall:.0f}s; predicted {n_ho:,}   peak RSS {peak_rss_gb():.2f} GB")
    return pred, dict(best_iteration=best, n_ref=n_ref, tree_count=int(r.tree_count_), n_fit=n_fit, n_es=n_es,
                      n_train=n_tr, n_holdout=n_ho, es_wall_s=round(es_wall, 1), refit_wall_s=round(refit_wall, 1),
                      params=params, catboost_version=cbv, numpy_version=np.__version__)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--target", required=True, choices=("delta", "G"))
    ap.add_argument("--output", required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args(argv)
    params = {k: getattr(a, k) for k in DEFAULTS}
    cols = json.loads(pathlib.Path(a.input + ".cols.json").read_text())
    d = load_input(a.input, cols["num"], cols["cat"], a.target)
    log(f"loaded {a.input}: {len(d['target']):,} rows x {d['X'].shape[1]} columns; peak RSS {peak_rss_gb():.2f} GB")
    pred, info = fit_and_predict(d, cols["cat"], params)
    out = pathlib.Path(a.output)
    np.save(out, pred)
    info.update(target=a.target, input=a.input, peak_rss_gb=round(peak_rss_gb(), 2), wall_s=round(time.time() - _T0, 1))
    out.with_suffix(".json").write_text(json.dumps(info, indent=1, default=float))
    log(f"wrote {out} (+ .json); wall {info['wall_s']:.0f}s; peak RSS {info['peak_rss_gb']} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
