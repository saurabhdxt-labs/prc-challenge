"""CatBoost worker for FM1's Learner protocol. Runs ONLY under ~/.venvs/prc-catboost (numpy + catboost; BC-5: CatBoost's
OpenMP runtime never shares a process with LightGBM's). Standalone: it imports nothing from `prc`.

    <venv python> -B catboost_worker.py fit     <dir>   # reads X.npy y.npy fit.npy es.npy spec.json; writes cb_seed<s>.cbm + result.json
    <venv python> -B catboost_worker.py predict <dir>   # reads X.npy spec.json cb_seed<s>.cbm; writes pred.npy (mean over seeds)

fit: early stopping on `es` rows (od_type Iter, patience) at the first seed -> best iteration (1-based) -> per seed, a refit on
fit|es rows at n_ref = max(50, int(best * n_all / n_fit)) iterations, no early stopping (lgbm_submit's rule, as FM1's LightGBM).
Deterministic for a fixed seed and thread_count (asserted by tests/fm1/test_fm1_learners.py).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np


def _n_refit(best: int, n_all: int, n_fit: int) -> int:
    return max(50, int(best * n_all / n_fit))


def _model(params: dict, iterations: int, seed: int, threads: int, **extra):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=int(iterations), random_seed=int(seed), thread_count=int(threads), verbose=0,
                             allow_writing_files=False, loss_function="RMSE", bootstrap_type="Bernoulli",
                             boosting_type="Plain", border_count=254, **params, **extra)


def fit(d: pathlib.Path) -> None:
    from catboost import Pool, __version__
    spec = json.loads((d / "spec.json").read_text())
    X = np.load(d / "X.npy", allow_pickle=False)
    y = np.load(d / "y.npy", allow_pickle=False)
    fitm = np.load(d / "fit.npy", allow_pickle=False)
    es = np.load(d / "es.npy", allow_pickle=False)
    seeds, threads, params = [int(s) for s in spec["seeds"]], int(spec["threads"]), dict(spec["params"])
    m = _model(params, spec["max_rounds"], seeds[0], threads, od_type="Iter", od_wait=int(spec["patience"]),
               use_best_model=True)
    m.fit(Pool(np.ascontiguousarray(X[fitm]), y[fitm]), eval_set=Pool(np.ascontiguousarray(X[es]), y[es]))
    best = int(m.get_best_iteration()) + 1
    tr = fitm | es
    n_ref = _n_refit(best, int(tr.sum()), int(fitm.sum()))
    pool = Pool(np.ascontiguousarray(X[tr]), y[tr])
    for s in seeds:
        r = _model(params, n_ref, s, threads)
        r.fit(pool)
        r.save_model(str(d / f"cb_seed{s}.cbm"))
    (d / "result.json").write_text(json.dumps({"best_iter": best, "n_ref": n_ref, "catboost_version": __version__}))


def predict(d: pathlib.Path) -> None:
    from catboost import CatBoostRegressor
    spec = json.loads((d / "spec.json").read_text())
    X = np.ascontiguousarray(np.load(d / "X.npy", allow_pickle=False))
    preds = []
    for s in [int(v) for v in spec["seeds"]]:
        m = CatBoostRegressor()
        m.load_model(str(d / f"cb_seed{s}.cbm"))
        preds.append(np.asarray(m.predict(X), dtype="float64"))
    np.save(d / "pred.npy", np.mean(preds, axis=0), allow_pickle=False)


def main(argv) -> int:
    if len(argv) != 3 or argv[1] not in ("fit", "predict"):
        sys.stderr.write("usage: catboost_worker.py fit|predict <dir>\n")
        return 2
    d = pathlib.Path(argv[2])
    (fit if argv[1] == "fit" else predict)(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
