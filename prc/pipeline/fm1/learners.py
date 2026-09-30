"""FM1 learners behind ONE protocol (plans/PREREG_fm1_2026_09_28.md §1; AT3, AT4, AT11).

    fit(X, y, fit_mask, es_mask, seeds, feature_names)   early stopping on es rows at the first seed, then one refit per
                                                          seed on fit|es rows at n_ref = max(50, int(best * n_all / n_fit))
    predict(X) -> float64 (n,)                            the mean over seeds
    save(dir) -> [file names]                             learner.json + one model file per seed; bit-exact on reload
    load_learner(dir) -> Learner                          dispatches on learner.json's name
    describe() -> dict

LightGBM runs in-process (deterministic=True, force_col_wise=True, fixed num_threads and seeds). CatBoost runs only in
~/.venvs/prc-catboost through catboost_worker.py (BC-5: one native model family per process); this module never imports
catboost. The HGB learner (sklearn, the unmatched body) runs in-process. XGBoost is CUT (owner decision 5); a generic MLP
learner is selection arm A6 and is not built (the MLP exists only inside the ADS-B stack's own worker).
"""
from __future__ import annotations

import json
import pathlib
import pickle
import subprocess
import tempfile
from typing import Protocol, Sequence

import numpy as np

from . import errors as E
from . import rows

CATBOOST_PYTHON = pathlib.Path.home() / ".venvs" / "prc-catboost" / "bin" / "python3"
CATBOOST_WORKER = pathlib.Path(__file__).resolve().parent / "catboost_worker.py"
WORKER_TIMEOUT_S = 24 * 3600
N_REF_FLOOR = 50


class Learner(Protocol):
    name: str

    def fit(self, X, y, fit_mask, es_mask, seeds: Sequence[int], feature_names: Sequence[str]) -> None: ...

    def predict(self, X) -> np.ndarray: ...

    def save(self, directory) -> list: ...

    def describe(self) -> dict: ...


def n_refit(best_iter: int, n_all: int, n_fit: int) -> int:
    """lgbm_submit.n_refit: best_iter scaled by the data ratio, truncated, floored at 50."""
    if best_iter < 1 or n_fit < 1 or n_all < n_fit:
        raise ValueError(f"n_refit: best_iter {best_iter}, n_all {n_all}, n_fit {n_fit}")
    return max(N_REF_FLOOR, int(best_iter * n_all / n_fit))


def _check_fit_inputs(X, y, fit_mask, es_mask, seeds, feature_names, need_es: bool = True) -> tuple:
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y, dtype="float64")
    fit_mask, es_mask = np.asarray(fit_mask, dtype=bool), np.asarray(es_mask, dtype=bool)
    if X.ndim != 2:
        raise E.SchemaError(f"X must be 2-D, got shape {X.shape}")
    if not (len(y) == len(fit_mask) == len(es_mask) == X.shape[0]):
        raise E.SchemaError(f"length mismatch: X {X.shape[0]}, y {len(y)}, fit {len(fit_mask)}, es {len(es_mask)}")
    if len(feature_names) != X.shape[1]:
        raise E.SchemaError(f"{len(feature_names)} feature names for {X.shape[1]} columns")
    rows.assert_design_clean(feature_names, "learner design")
    if (fit_mask & es_mask).any():
        raise E.SchemaError(f"fit and early-stopping rows overlap on {int((fit_mask & es_mask).sum())} row(s)")
    if not fit_mask.any():
        raise E.SchemaError("no fit rows")
    if need_es and not es_mask.any():
        raise E.SchemaError("no early-stopping rows")
    used = fit_mask | es_mask
    if not np.isfinite(y[used]).all():
        raise E.SchemaError(f"non-finite target on {int((~np.isfinite(y[used])).sum())} fit/early-stopping row(s)")
    seeds = tuple(int(s) for s in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise E.SchemaError(f"seeds must be non-empty and unique, got {list(seeds)}")
    return X, y, fit_mask, es_mask, seeds, list(feature_names)


def _check_predict_input(X, n_features: int, fitted: bool) -> np.ndarray:
    if not fitted:
        raise E.PipelineError("learner is not fitted (or loaded)")
    X = np.asarray(X, dtype="float64")
    if X.ndim != 2 or X.shape[1] != n_features:
        raise E.SchemaError(f"predict: expected {n_features} columns, got shape {X.shape}")
    return X


def _write_json(path: pathlib.Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))


# =================================================================================================================
# LightGBM (in-process)
# =================================================================================================================

def lgb_params(params: dict, seed: int, threads: int) -> dict:
    return {"objective": "regression", "metric": "rmse", "verbosity": -1, **params, "deterministic": True,
            "force_col_wise": True, "num_threads": int(threads), "seed": int(seed), "bagging_seed": int(seed),
            "feature_fraction_seed": int(seed), "data_random_seed": int(seed)}


class LightGBMLearner:
    name = "lightgbm"

    def __init__(self, params: dict, max_rounds: int, patience: int, threads: int):
        self.params, self.max_rounds, self.patience, self.threads = dict(params), int(max_rounds), int(patience), int(threads)
        self.boosters: list = []
        self.seeds: tuple = ()
        self.feature_names: list = []
        self.best_iter = self.n_ref = None

    def fit(self, X, y, fit_mask, es_mask, seeds, feature_names) -> None:
        import lightgbm as lgb
        X, y, fit_mask, es_mask, seeds, names = _check_fit_inputs(X, y, fit_mask, es_mask, seeds, feature_names)
        dfit = lgb.Dataset(X[fit_mask], y[fit_mask], feature_name=names, free_raw_data=False)
        des = lgb.Dataset(X[es_mask], y[es_mask], reference=dfit, feature_name=names, free_raw_data=False)
        es_model = lgb.train(lgb_params(self.params, seeds[0], self.threads), dfit, self.max_rounds, valid_sets=[des],
                             callbacks=[lgb.early_stopping(self.patience, verbose=False)])
        best = int(es_model.best_iteration) or int(es_model.current_iteration())
        tr = fit_mask | es_mask
        n_ref = n_refit(best, int(tr.sum()), int(fit_mask.sum()))
        dtr = lgb.Dataset(X[tr], y[tr], feature_name=names, free_raw_data=False)
        self.boosters = [lgb.train(lgb_params(self.params, s, self.threads), dtr, n_ref) for s in seeds]
        self.seeds, self.feature_names, self.best_iter, self.n_ref = seeds, names, best, n_ref

    def predict(self, X) -> np.ndarray:
        X = _check_predict_input(X, len(self.feature_names), bool(self.boosters))
        return np.mean([b.predict(X, num_threads=self.threads) for b in self.boosters], axis=0).astype("float64")

    def describe(self) -> dict:
        return {"name": self.name, "params": dict(self.params), "max_rounds": self.max_rounds, "patience": self.patience,
                "threads": self.threads, "seeds": list(self.seeds), "best_iter": self.best_iter, "n_ref": self.n_ref,
                "feature_names": list(self.feature_names)}

    def save(self, directory) -> list:
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        files = []
        for s, b in zip(self.seeds, self.boosters):
            b.save_model(str(d / f"lgb_seed{s}.txt"))
            files.append(f"lgb_seed{s}.txt")
        _write_json(d / "learner.json", self.describe())
        return [*files, "learner.json"]

    @classmethod
    def _load(cls, d: pathlib.Path, meta: dict) -> "LightGBMLearner":
        import lightgbm as lgb
        m = cls(meta["params"], meta["max_rounds"], meta["patience"], meta["threads"])
        m.seeds, m.feature_names = tuple(meta["seeds"]), list(meta["feature_names"])
        m.best_iter, m.n_ref = meta["best_iter"], meta["n_ref"]
        m.boosters = [lgb.Booster(model_file=str(d / f"lgb_seed{s}.txt")) for s in m.seeds]
        return m


# =================================================================================================================
# CatBoost (worker process only)
# =================================================================================================================

class CatBoostWorkerLearner:
    name = "catboost"

    def __init__(self, params: dict, max_rounds: int, patience: int, threads: int):
        self.params, self.max_rounds, self.patience, self.threads = dict(params), int(max_rounds), int(patience), int(threads)
        self.models: dict = {}                    # seed -> .cbm bytes (the worker's files, kept verbatim)
        self.seeds: tuple = ()
        self.feature_names: list = []
        self.best_iter = self.n_ref = None
        self.catboost_version = None

    def _run(self, cmd: str, d: pathlib.Path) -> None:
        py = pathlib.Path(CATBOOST_PYTHON)
        if not py.exists():
            raise E.WorkerError(f"CatBoost worker interpreter {py} does not exist (CatBoost lives only in "
                                "~/.venvs/prc-catboost; BC-5)")
        r = subprocess.run([str(py), "-B", str(CATBOOST_WORKER), cmd, str(d)], capture_output=True, text=True,
                           timeout=WORKER_TIMEOUT_S)
        if r.returncode != 0:
            raise E.WorkerError(f"CatBoost worker `{cmd}` failed (exit {r.returncode}): {r.stderr[-2000:].strip()}")

    def _spec(self) -> dict:
        return {"params": self.params, "max_rounds": self.max_rounds, "patience": self.patience,
                "threads": self.threads, "seeds": list(self.seeds)}

    def fit(self, X, y, fit_mask, es_mask, seeds, feature_names) -> None:
        X, y, fit_mask, es_mask, seeds, names = _check_fit_inputs(X, y, fit_mask, es_mask, seeds, feature_names)
        self.seeds = seeds
        with tempfile.TemporaryDirectory(prefix="fm1_cb_") as t:
            d = pathlib.Path(t)
            for n, a in (("X", X), ("y", y), ("fit", fit_mask), ("es", es_mask)):
                np.save(d / f"{n}.npy", a, allow_pickle=False)
            _write_json(d / "spec.json", self._spec())
            self._run("fit", d)
            try:
                res = json.loads((d / "result.json").read_text())
                self.models = {s: (d / f"cb_seed{s}.cbm").read_bytes() for s in seeds}
            except (OSError, ValueError) as exc:
                raise E.WorkerError(f"CatBoost worker returned a malformed result: {exc}") from None
        self.feature_names, self.best_iter, self.n_ref = names, int(res["best_iter"]), int(res["n_ref"])
        self.catboost_version = res.get("catboost_version")

    def predict(self, X) -> np.ndarray:
        X = _check_predict_input(X, len(self.feature_names), bool(self.models))
        with tempfile.TemporaryDirectory(prefix="fm1_cbp_") as t:
            d = pathlib.Path(t)
            np.save(d / "X.npy", X, allow_pickle=False)
            _write_json(d / "spec.json", self._spec())
            for s, blob in self.models.items():
                (d / f"cb_seed{s}.cbm").write_bytes(blob)
            self._run("predict", d)
            pred = np.load(d / "pred.npy", allow_pickle=False)
        if pred.shape != (X.shape[0],) or not np.isfinite(pred).all():
            raise E.WorkerError(f"CatBoost worker returned {pred.shape} predictions for {X.shape[0]} rows or non-finite values")
        return pred.astype("float64")

    def describe(self) -> dict:
        return {"name": self.name, "params": dict(self.params), "max_rounds": self.max_rounds, "patience": self.patience,
                "threads": self.threads, "seeds": list(self.seeds), "best_iter": self.best_iter, "n_ref": self.n_ref,
                "feature_names": list(self.feature_names), "catboost_version": self.catboost_version}

    def save(self, directory) -> list:
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        files = []
        for s in self.seeds:
            (d / f"cb_seed{s}.cbm").write_bytes(self.models[s])
            files.append(f"cb_seed{s}.cbm")
        _write_json(d / "learner.json", self.describe())
        return [*files, "learner.json"]

    @classmethod
    def _load(cls, d: pathlib.Path, meta: dict) -> "CatBoostWorkerLearner":
        m = cls(meta["params"], meta["max_rounds"], meta["patience"], meta["threads"])
        m.seeds, m.feature_names = tuple(meta["seeds"]), list(meta["feature_names"])
        m.best_iter, m.n_ref, m.catboost_version = meta["best_iter"], meta["n_ref"], meta.get("catboost_version")
        m.models = {s: (d / f"cb_seed{s}.cbm").read_bytes() for s in m.seeds}
        return m


# =================================================================================================================
# HGB (sklearn, in-process): the unmatched body
# =================================================================================================================

class HGBLearner:
    name = "hgb"

    def __init__(self, params: dict):
        self.params = dict(params)
        self.models: list = []
        self.seeds: tuple = ()
        self.feature_names: list = []
        self.validation = None

    def fit(self, X, y, fit_mask, es_mask, seeds, feature_names) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGR
        X, y, fit_mask, es_mask, seeds, names = _check_fit_inputs(X, y, fit_mask, es_mask, seeds, feature_names,
                                                                  need_es=False)
        self.models = []
        for s in seeds:
            m = HGR(random_state=s, early_stopping=True, **self.params)
            if es_mask.any():
                m.fit(X[fit_mask], y[fit_mask], X_val=X[es_mask], y_val=y[es_mask])
            else:
                m.fit(X[fit_mask], y[fit_mask])
            self.models.append(m)
        self.validation = "es rows" if es_mask.any() else "internal 10% split"
        self.seeds, self.feature_names = seeds, names

    def predict(self, X) -> np.ndarray:
        X = _check_predict_input(X, len(self.feature_names), bool(self.models))
        return np.mean([m.predict(X) for m in self.models], axis=0).astype("float64")

    def describe(self) -> dict:
        return {"name": self.name, "params": dict(self.params), "seeds": list(self.seeds),
                "feature_names": list(self.feature_names), "validation": self.validation,
                "n_iter": [int(m.n_iter_) for m in self.models]}

    def save(self, directory) -> list:
        d = pathlib.Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        files = []
        for s, m in zip(self.seeds, self.models):
            (d / f"hgb_seed{s}.pkl").write_bytes(pickle.dumps(m, protocol=5))
            files.append(f"hgb_seed{s}.pkl")
        _write_json(d / "learner.json", self.describe())
        return [*files, "learner.json"]

    @classmethod
    def _load(cls, d: pathlib.Path, meta: dict) -> "HGBLearner":
        m = cls(meta["params"])
        m.seeds, m.feature_names, m.validation = tuple(meta["seeds"]), list(meta["feature_names"]), meta["validation"]
        # the artefact loader verifies every file's sha256 before this unpickles it (artifact.load_artifact)
        m.models = [pickle.loads((d / f"hgb_seed{s}.pkl").read_bytes()) for s in m.seeds]
        return m


# =================================================================================================================
# registry
# =================================================================================================================

LEARNERS = {"lightgbm": LightGBMLearner, "catboost": CatBoostWorkerLearner, "hgb": HGBLearner}
CUT = {"xgboost": "is CUT (owner decision 5, 2026-09-28)",
       "mlp": "is not a days 1-3 learner (selection arm A6; the MLP exists only inside the ADS-B stack's worker)"}


def make_learner(name: str, cfg, **overrides):
    """A learner configured from the FM1 config; `overrides` replace learner params (the validation tier's learning rate)."""
    if name in CUT:
        raise E.LearnerCutError(f"learner {name!r} {CUT[name]}")
    if name not in LEARNERS:
        raise E.LearnerCutError(f"unknown learner {name!r}; learners are {sorted(LEARNERS)}")
    threads = cfg["run.threads"]
    if name == "lightgbm":
        p = {k: cfg[f"matched.lightgbm.{k}"] for k in ("learning_rate", "num_leaves", "min_data_in_leaf",
                                                         "feature_fraction", "bagging_fraction", "bagging_freq")}
        return LightGBMLearner({**p, **overrides}, cfg["matched.lightgbm.max_rounds"], cfg["matched.lightgbm.patience"],
                               threads)
    if name == "catboost":
        p = {k: cfg[f"matched.catboost.{k}"] for k in ("learning_rate", "depth", "l2_leaf_reg", "subsample")}
        return CatBoostWorkerLearner({**p, **overrides}, cfg["matched.catboost.max_rounds"],
                                     cfg["matched.catboost.patience"], threads)
    p = {k: cfg[f"unmatched.hgb.{k}"] for k in ("max_iter", "n_iter_no_change", "max_leaf_nodes", "min_samples_leaf",
                                                 "learning_rate")}
    return HGBLearner({**p, **overrides})


def load_learner(directory):
    d = pathlib.Path(directory)
    meta_path = d / "learner.json"
    if not meta_path.exists():
        raise E.ArtifactError(f"{d}: no learner.json")
    meta = json.loads(meta_path.read_text())
    cls = LEARNERS.get(meta.get("name"))
    if cls is None:
        raise E.ArtifactError(f"{d}: unknown learner {meta.get('name')!r}")
    return cls._load(d, meta)
