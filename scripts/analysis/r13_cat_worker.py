"""R13 CatBoost worker — runs ONLY in ~/.venvs/prc-catboost and never imports lightgbm (BC-5: one OpenMP runtime per
process). Spawned by scripts/analysis/r13_cat.py (CatFitter); talks to it through files in a work directory under
data/r12 (never /tmp).

    ~/.venvs/prc-catboost/bin/python3 -B scripts/analysis/r13_cat_worker.py es <workdir>
    ~/.venvs/prc-catboost/bin/python3 -B scripts/analysis/r13_cat_worker.py refit <workdir>

Inputs (written by r13_cat.write_work): X.npy (float32 design, all rows), in.npz (label, fit / es / tr / pred masks,
seeds, es_seed, params json, cap, patience; refit mode also n_ref).
  es     fit on `fit` rows, early stopping on `es` rows (od Iter, patience) up to `cap` iterations, seed es_seed ->
         es_out.npz: best_iter (1-based tree count), pred_es (the `es` rows at best_iter).
  refit  per seed: fit on `tr` rows at n_ref iterations, no early stopping -> refit_out.npz: preds (seeds x pred rows).
On any exception the out file carries `error` and the exit status is 3; the parent raises it.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

MODES = ("es", "refit")


def _model(params: dict, iterations: int, seed: int, **extra):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(iterations=int(iterations), random_seed=int(seed), verbose=0, allow_writing_files=False,
                             **params, **extra)


def run_es(X, z) -> dict:
    from catboost import Pool
    params = json.loads(str(z["params"]))
    fit, es, label = z["fit"], z["es"], z["label"]
    m = _model(params, int(z["cap"]), int(z["es_seed"]), od_type="Iter", od_wait=int(z["patience"]),
               use_best_model=True)
    t = time.time()
    m.fit(Pool(np.ascontiguousarray(X[fit]), label[fit]), eval_set=Pool(np.ascontiguousarray(X[es]), label[es]))
    best = int(m.get_best_iteration()) + 1                    # CatBoost reports the 0-based index of the best tree
    pred_es = np.asarray(m.predict(np.ascontiguousarray(X[es]), ntree_end=best), dtype="float64")
    print(f"[r13_cat_worker] es: best_iter {best:,} of {m.tree_count_:,} trees, cap {int(z['cap']):,}  "
          f"[{time.time() - t:.0f}s]", flush=True)
    return {"best_iter": np.array([best]), "tree_count": np.array([int(m.tree_count_)]), "pred_es": pred_es,
            "es_s": np.array([time.time() - t])}


def run_refit(X, z) -> dict:
    from catboost import Pool
    params = json.loads(str(z["params"]))
    tr, pred, label, n_ref = z["tr"], z["pred"], z["label"], int(z["n_ref"])
    Xp = np.ascontiguousarray(X[pred])
    pool = Pool(np.ascontiguousarray(X[tr]), label[tr])
    out = []
    for s in [int(v) for v in z["seeds"]]:
        t = time.time()
        m = _model(params, n_ref, s)
        m.fit(pool)
        out.append(np.asarray(m.predict(Xp), dtype="float64"))
        print(f"[r13_cat_worker] refit seed {s}: {n_ref:,} trees on {int(tr.sum()):,} rows  [{time.time() - t:.0f}s]",
              flush=True)
        del m
    return {"preds": np.vstack(out), "seeds": np.asarray(z["seeds"])}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in MODES:
        print(f"usage: r13_cat_worker.py {{{'|'.join(MODES)}}} <workdir>", file=sys.stderr)
        return 2
    assert "lightgbm" not in sys.modules, "BC-5: the CatBoost worker must never share a process with lightgbm"
    mode, d = argv[0], pathlib.Path(argv[1])
    out_p = d / f"{mode}_out.npz"
    try:
        X = np.load(d / "X.npy", mmap_mode="r")
        z = dict(np.load(d / "in.npz", allow_pickle=False))
        res = run_es(X, z) if mode == "es" else run_refit(X, z)
        bad = [k for k, v in res.items() if np.asarray(v).dtype.kind == "f" and not np.isfinite(v).all()]
        if bad:
            raise FloatingPointError(f"non-finite values in {bad}")
    except Exception as exc:                                  # noqa: BLE001 — reported to the parent, never swallowed
        np.savez(out_p, error=np.array(f"{type(exc).__name__}: {exc}"))
        print(f"[r13_cat_worker] {mode} FAILED: {type(exc).__name__}: {exc}", flush=True)
        return 3
    np.savez(out_p, **res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
