"""The model plug-in interface: every lane holds its model behind ONE protocol, so a tree, CatBoost, a
tabular NN or a blend is a config choice judged by the same harness ("if tree works better then tree it
is, but we have to be able to handle it properly" -- owner, 2026-09-10).

    fit(X, y, masks, seeds) -> state        X: a design (ndarray for "matrix" models, DataFrame for "frame")
    predict(state, X) -> np.ndarray         the model's own output scale, float64 (describe()["output"])
    save(state, directory) -> [paths]       everything load() needs, with provenance beside it
    load(source) -> state
    describe() -> dict                      name, implemented, input kind, output scale, what it wraps

Adapters wrap the SHIPPED fit functions, never re-implement them:
  * LightGBMModel     -> scripts/lgbm_submit.py (early_stop, refit, predict_delta, mean_delta,
                         reuse_seeds / check_provenance for saved boosters);
  * NonFillBodyModel  -> build_submission.fit_nf_regressor (S1's body, Amendment 18) or, with
                         congestion=True, unm_congestion.fit_congestion_regressor (E3C's body);
  * CatBoostModel / NNModel / BlendModel -> stubs that raise NotImplementedError with the reason.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import pickle
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from . import errors as E
from . import legacy

#: the fit.json provenance fields lgbm_submit writes for a matched-only, weight-1 booster
LGBM_PROVENANCE = {"all_rows": False, "unmatched_weight": 1.0}
DESCRIBE_KEYS = ("name", "implemented", "input_kind", "output", "wraps")


# =================================================================================================
# the contract
# =================================================================================================

@dataclass(frozen=True)
class Masks:
    """Row masks over X. `train`: the rows a final fit sees. `fit` / `es`: an early-stopping split of
    `train` (both or neither; disjoint; fit | es == train)."""
    train: np.ndarray
    fit: np.ndarray | None = None
    es: np.ndarray | None = None

    def __post_init__(self):
        arrs = {"train": self.train, "fit": self.fit, "es": self.es}
        for k, a in arrs.items():
            if a is None:
                continue
            a = np.asarray(a)
            if a.dtype != bool or a.ndim != 1:
                raise ValueError(f"Masks.{k} must be a 1-D boolean array, got dtype {a.dtype} ndim {a.ndim}")
            object.__setattr__(self, k, a)
        if not self.train.any():
            raise ValueError("Masks.train selects no row")
        if (self.fit is None) != (self.es is None):
            raise ValueError("Masks: give both fit and es, or neither")
        if self.fit is not None:
            if not (len(self.fit) == len(self.es) == len(self.train)):
                raise ValueError(f"Masks: lengths differ (train {len(self.train)}, fit {len(self.fit)}, es {len(self.es)})")
            if (self.fit & self.es).any():
                raise ValueError("Masks: fit and es overlap")
            if not np.array_equal(self.fit | self.es, self.train):
                raise ValueError("Masks: fit | es must equal train")


@dataclass(frozen=True)
class Labels:
    """What a model may learn from, row-aligned with X. `taxi` = TAXITIME (the competition's y);
    `delta` = BLOCK - AOBT_3; `proxy` = MVT - AOBT_3 (the anchor a delta model is subtracted from)."""
    taxi: np.ndarray | None = None
    delta: np.ndarray | None = None
    proxy: np.ndarray | None = None
    regime: np.ndarray | None = None            # P3: regime index on training rows (regimes.label_matched), else −1


@runtime_checkable
class ModelPlugin(Protocol):
    name: str
    input_kind: str

    def fit(self, X, y: Labels, masks: Masks, seeds: Sequence[int]) -> Any: ...

    def predict(self, state, X) -> np.ndarray: ...

    def save(self, state, directory) -> list: ...

    def load(self, source) -> Any: ...

    def describe(self) -> dict: ...


def check_prediction(model, pred, n_rows: int) -> np.ndarray:
    """The predict() half of the contract, enforced where a plug-in's output enters a lane: a 1-D float64 ndarray with
    one finite value per row. `isinstance(m, ModelPlugin)` only proves the methods exist (runtime_checkable checks
    names), so a predict() returning None, a list, a column vector or NaN would pass it; this refuses each, naming the
    plug-in."""
    name = getattr(model, "name", type(model).__name__)
    if not isinstance(pred, np.ndarray):
        raise E.LaneError(f"model plug-in {name!r}: predict() returned {type(pred).__name__}, not a numpy array")
    if pred.ndim != 1 or len(pred) != n_rows:
        raise E.LaneError(f"model plug-in {name!r}: predict() returned shape {pred.shape}, expected ({n_rows},)")
    if pred.dtype != np.float64:
        raise E.LaneError(f"model plug-in {name!r}: predict() returned dtype {pred.dtype}, expected float64")
    bad = int((~np.isfinite(pred)).sum())
    if bad:
        raise E.LaneError(f"model plug-in {name!r}: predict() returned {bad:,} non-finite value(s)")
    return pred


def _seeds(seeds) -> tuple:
    out = tuple(int(s) for s in seeds)
    if not out or len(set(out)) != len(out) or any(s < 0 for s in out):
        raise ValueError(f"seeds must be distinct non-negative integers, got {tuple(seeds)!r}")
    return out


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# =================================================================================================
# LightGBM (the matched lane's incumbent)
# =================================================================================================

@dataclass
class LightGBMState:
    seeds: tuple
    params: dict
    info: dict                                  # the early-stopping record (best_iter, n_ref, n_all, ...)
    n_features: int
    target: str
    n_all: int                                  # training rows (a saved booster must have been fit on as many)
    boosters: dict | None = None                # {seed: lgb.Booster} after fit
    files: list | None = None                   # [(model file, fit.json)] after load: loaded ONE AT A TIME

    @property
    def lazy(self) -> bool:
        return self.boosters is None


class LightGBMModel:
    """lgbm_submit's matched regressor: an early-stopping run (seed 0) on masks.fit / masks.es gives
    best_iter -> n_ref; one refit per seed on masks.train at n_ref; the prediction is the arithmetic mean
    of the per-seed predictions on the label's scale (delta by default: the lane recovers the taxi time as
    max(proxy - delta_hat, 1), rounded)."""

    name = "lightgbm"
    input_kind = "matrix"

    def __init__(self, params: dict | None = None, target: str = "delta", nest: int | None = None,
                 patience: int | None = None):
        ls = legacy.matched().ls
        if target not in ls.TARGETS:
            raise ValueError(f"target must be one of {ls.TARGETS}, got {target!r}")
        self.params = dict(ls.P if params is None else params)
        self.target = target
        self.nest = int(ls.NEST if nest is None else nest)
        self.patience = int(ls.PATIENCE if patience is None else patience)

    def fit(self, X, y: Labels, masks: Masks, seeds) -> LightGBMState:
        ls = legacy.matched().ls
        seeds = _seeds(seeds)
        if masks.fit is None:
            raise ValueError("LightGBMModel.fit needs an early-stopping split (Masks.fit / Masks.es)")
        if y.taxi is None or y.delta is None or y.proxy is None:
            raise ValueError("LightGBMModel.fit needs Labels(taxi, delta, proxy): the stopping-set guard reads all three")
        X = np.asarray(X)
        if X.ndim != 2 or len(X) != len(masks.train):
            raise ValueError(f"X has shape {X.shape}; masks cover {len(masks.train)} rows")
        taxi, delta, proxy = (np.asarray(a, dtype="float64") for a in (y.taxi, y.delta, y.proxy))
        info = ls.early_stop(X, delta, taxi, proxy, masks.train, masks.fit, masks.es, dict(self.params, seed=ls.ES_SEED),
                             self.nest, self.patience, target=self.target)
        label = ls.regression_label(delta, taxi, self.target)
        boosters = {}
        for s in seeds:
            b = ls.refit(X, label, masks.train, dict(self.params, seed=s), info["n_ref"])
            if b.num_feature() != X.shape[1]:
                raise ValueError(f"booster has {b.num_feature()} features, X has {X.shape[1]}")
            boosters[s] = b
        return LightGBMState(seeds=seeds, params=dict(self.params), info=dict(info), n_features=int(X.shape[1]),
                             target=self.target, n_all=int(masks.train.sum()), boosters=boosters)

    def predict(self, state: LightGBMState, X) -> np.ndarray:
        ls = legacy.matched().ls
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != state.n_features:
            raise ValueError(f"X has shape {X.shape}; the model has {state.n_features} features")
        if state.lazy:
            # lgbm_submit.reuse_seeds: load one booster, check its provenance (params incl. seed, n_features,
            # target, n_all, tree count == n_ref, smoke flag), predict, free -- then the next one
            preds, _ = ls.reuse_seeds(state.files, state.seeds, X, state.params, False, state.n_all,
                                      n_features=state.n_features, target=state.target, provenance=LGBM_PROVENANCE)
        else:
            preds = {s: ls.predict_delta(state.boosters[s], X, state.params) for s in state.seeds}
        return ls.mean_delta([preds[s] for s in state.seeds])

    def save(self, state: LightGBMState, directory) -> list:
        """One `lgbm_seed{s}.txt` + `.fit.json` per seed, in the fit.json shape lgbm_submit writes, so
        lgbm_submit.reuse_seeds / check_provenance accept them."""
        ls = legacy.matched().ls
        if state.lazy:
            raise ValueError("a loaded (lazy) state is already on disk; save() writes fitted boosters")
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        out = []
        for s in state.seeds:
            path, fjson = directory / f"lgbm_seed{s}.txt", directory / f"lgbm_seed{s}.fit.json"
            state.boosters[s].save_model(str(path))
            fjson.write_text(json.dumps({**state.info, "smoke": False, "params": dict(state.params, seed=s),
                                         "es_seed": ls.ES_SEED, "n_features": state.n_features, "target": state.target,
                                         **LGBM_PROVENANCE}, indent=2))
            out += [path, fjson]
        return out

    def load(self, source, n_all: int | None = None, n_features: int | None = None, seeds=None) -> LightGBMState:
        """`source`: a directory written by save(), or [(model file, fit.json), ...] one per seed in seed
        order (lgbm_submit's boosters). Every fit.json is read and checked NOW (params incl. its seed,
        target, n_features, n_all, smoke, one early-stopping run); the boosters themselves are loaded one at a
        time at predict(), where lgbm_submit.check_provenance also checks the tree count."""
        if isinstance(source, (str, pathlib.Path)):
            d = pathlib.Path(source)
            files = sorted(d.glob("lgbm_seed*.txt"), key=lambda p: int(p.stem.split("seed")[-1]))
            pairs = [(p, p.with_name(p.stem + ".fit.json")) for p in files]
            seeds = tuple(int(p.stem.split("seed")[-1]) for p in files) if seeds is None else _seeds(seeds)
        else:
            pairs = [(pathlib.Path(a), pathlib.Path(b)) for a, b in source]
            if seeds is None:
                raise ValueError("load([(model, fit.json), ...]) needs the seeds, one per pair, in order")
            seeds = _seeds(seeds)
        if not pairs or len(pairs) != len(seeds):
            raise ValueError(f"{len(pairs)} booster file(s) for seeds {seeds}")
        infos = []
        for (p, j), s in zip(pairs, seeds):
            if not p.exists() or not j.exists():
                raise FileNotFoundError(f"booster {p} or its provenance {j} is missing")
            i = json.loads(j.read_text())
            problems = []
            if i.get("params") != dict(self.params, seed=s):
                problems.append(f"params differ from lgbm_submit's with seed {s}")
            absent = [k for k in ("target", *LGBM_PROVENANCE) if k not in i]
            if absent:      # never assumed: a booster that does not say what it is was not produced by this config
                problems.append(f"provenance field(s) {absent} absent")
            elif i["target"] != self.target:
                problems.append(f"target {i['target']!r} != {self.target!r}")
            if n_features is not None and i.get("n_features") != n_features:
                problems.append(f"n_features {i.get('n_features')} != {n_features}")
            if n_all is not None and i.get("n_all") != n_all:
                problems.append(f"n_all {i.get('n_all')} != {n_all} training rows")
            if i.get("smoke") is not False:
                problems.append(f"smoke={i.get('smoke')!r}")
            for k, v in LGBM_PROVENANCE.items():
                if k in i and i[k] != v:
                    problems.append(f"{k}={i[k]!r}")
            if problems:
                raise ValueError(f"{p.name}: saved booster was not produced by this configuration: " + "; ".join(problems))
            infos.append(i)
        if len({(i["best_iter"], i["n_ref"]) for i in infos}) != 1:
            raise ValueError("the saved boosters disagree on best_iter / n_ref: not one early-stopping run")
        i0 = infos[0]
        return LightGBMState(seeds=seeds, params=dict(self.params), info={k: v for k, v in i0.items() if k != "params"},
                             n_features=int(i0["n_features"]), target=self.target, n_all=int(i0["n_all"]),
                             files=pairs)

    def describe(self) -> dict:
        return {"name": self.name, "implemented": True, "input_kind": self.input_kind,
                "output": f"{self.target} (mean over seeds)", "params": dict(self.params), "target": self.target,
                "max_rounds": self.nest, "patience": self.patience,
                "wraps": ["lgbm_submit.early_stop", "lgbm_submit.refit", "lgbm_submit.predict_delta",
                          "lgbm_submit.mean_delta", "lgbm_submit.reuse_seeds"],
                "determinism": "predict is deterministic; a refit at num_threads > 1 is not bit-deterministic "
                               "(lgbm_fold.py docstring)"}


# =================================================================================================
# the unmatched body (S1 / E3C)
# =================================================================================================

@dataclass
class NonFillState:
    name: str
    seeds: tuple
    regressors: dict                            # {seed: NonFillRegressor | CongestionRegressor}
    n_train: int


class NonFillBodyModel:
    """S1's non-fill body (build_submission.NonFillRegressor, Screen B's B2) or, with congestion=True,
    E3C's (unm_congestion.CongestionRegressor: the same with the airport-hour witness appended). One
    regressor per seed on the masks.train rows -- which must be NON-FILL rows (the shipped fit functions
    refuse fill rows); the prediction is build_submission.mean_over_seeds, float64, unfloored (the lane's
    mixture applies the floor). X is a frame: the regressors read their own columns from it."""

    input_kind = "frame"

    def __init__(self, congestion: bool = False):
        self.congestion = bool(congestion)
        self.name = "nonfill_body_congestion" if self.congestion else "nonfill_body"

    def _fit_fn(self):
        st = legacy.stratum()
        return st.uc.fit_congestion_regressor if self.congestion else st.bs.fit_nf_regressor

    def fit(self, X, y: Labels, masks: Masks, seeds) -> NonFillState:
        seeds = _seeds(seeds)
        if len(X) != len(masks.train):
            raise ValueError(f"X has {len(X)} rows; masks cover {len(masks.train)}")
        frame = X[masks.train]
        if y is not None and y.taxi is not None:
            taxi = np.asarray(y.taxi, dtype="float64")
            if len(taxi) != len(masks.train):
                raise ValueError(f"Labels.taxi has {len(taxi)} rows; masks cover {len(masks.train)}")
            frame = frame.assign(y=taxi[masks.train])
        fit_fn = self._fit_fn()
        regs = {s: fit_fn(frame, s) for s in seeds}
        return NonFillState(name=self.name, seeds=seeds, regressors=regs, n_train=int(len(frame)))

    def predict(self, state: NonFillState, X) -> np.ndarray:
        if state.name != self.name:
            raise ValueError(f"state is a {state.name!r} model, this plug-in is {self.name!r}")
        bs = legacy.stratum().bs
        return bs.mean_over_seeds({s: state.regressors[s].predict(X) for s in state.seeds})

    def _cls(self):
        st = legacy.stratum()
        return st.uc.CongestionRegressor if self.congestion else st.bs.NonFillRegressor

    def save(self, state: NonFillState, directory) -> list:
        """The scripts load build_submission through importlib specs (stratum_fold._load, lgbm_submit._load), so
        several module objects carry a `NonFillRegressor` and pickle cannot resolve the class by name. What is
        stored is what pickle's default reduce stores -- each regressor's instance state -- with the class NAME;
        load() rebuilds the instances against the one module legacy.stratum() uses (P2b: make the scripts a
        package so their objects pickle by reference)."""
        import sklearn

        cls = self._cls()
        bad = [type(r).__name__ for r in state.regressors.values() if type(r).__name__ != cls.__name__]
        if bad:
            raise ValueError(f"{self.name} state holds {bad}, expected {cls.__name__}")
        directory = pathlib.Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.pkl"
        with open(path, "wb") as fh:
            pickle.dump({"name": state.name, "seeds": state.seeds, "cls": cls.__name__, "n_train": state.n_train,
                         "states": {s: dict(r.__dict__) for s, r in state.regressors.items()}}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        meta = directory / f"{self.name}.json"
        meta.write_text(json.dumps({"name": state.name, "seeds": list(state.seeds), "n_train": state.n_train,
                                    "sklearn": sklearn.__version__, "sha256": _sha256(path)}, indent=2))
        return [path, meta]

    def load(self, source) -> NonFillState:
        """Refuses a file whose sha256 differs from its sidecar, another model's file, or a pickle written by
        another scikit-learn version (the trees would load, but not necessarily predict the same)."""
        import sklearn

        d = pathlib.Path(source)
        path, meta = d / f"{self.name}.pkl", d / f"{self.name}.json"
        if not path.exists() or not meta.exists():
            raise FileNotFoundError(f"{path} or {meta} is missing")
        m = json.loads(meta.read_text())
        if m.get("sha256") != _sha256(path):
            raise ValueError(f"{path.name}: sha256 differs from {meta.name}; refusing to unpickle")
        if m.get("sklearn") != sklearn.__version__:
            raise ValueError(f"{path.name} was written by scikit-learn {m.get('sklearn')}, this is {sklearn.__version__}")
        with open(path, "rb") as fh:
            blob = pickle.load(fh)                # noqa: S301 -- our own file, hash-checked above
        cls = self._cls()
        if blob.get("name") != self.name or blob.get("cls") != cls.__name__:
            raise ValueError(f"{path.name} holds a {blob.get('name')!r} / {blob.get('cls')!r} model, not {self.name!r}")
        regs = {}
        for s, st_ in blob["states"].items():
            r = object.__new__(cls)                 # pickle's own reconstruction, against the resolved class
            r.__dict__.update(st_)
            regs[s] = r
        return NonFillState(name=blob["name"], seeds=tuple(blob["seeds"]), regressors=regs, n_train=int(blob["n_train"]))

    def describe(self) -> dict:
        bs = legacy.stratum().bs
        numeric = list(bs.NF_NUMERIC) + (["hprox_med", "hprox_n"] if self.congestion else [])
        return {"name": self.name, "implemented": True, "input_kind": self.input_kind,
                "output": "non-fill taxi seconds (mean over seeds, unfloored)", "params": dict(bs.NF_PARAMS),
                "numeric": numeric, "encoded": list(bs.NF_ENCODED), "winsor_s": bs.WINSOR_S,
                "wraps": ["unm_congestion.fit_congestion_regressor" if self.congestion
                          else "build_submission.fit_nf_regressor", "build_submission.mean_over_seeds"],
                "determinism": "deterministic given the seed (sklearn HistGradientBoosting)"}


# =================================================================================================
# stubs: known names, not implemented in P2a
# =================================================================================================

class _Stub:
    """Constructed exactly like a real plug-in (a lane passes its config, e.g. target=), so a config that names
    it reaches the clear refusal below instead of a TypeError; every operation is refused."""
    name = "stub"
    input_kind = "matrix"
    why = ""

    def __init__(self, **config):
        self.config = dict(config)

    def _refuse(self, what: str):
        raise NotImplementedError(f"model plug-in {self.name!r}: {what} is not implemented in P2a -- {self.why}")

    def fit(self, X, y: Labels, masks: Masks, seeds):
        self._refuse("fit")

    def predict(self, state, X):
        self._refuse("predict")

    def save(self, state, directory):
        self._refuse("save")

    def load(self, source):
        self._refuse("load")

    def describe(self) -> dict:
        return {"name": self.name, "implemented": False, "input_kind": self.input_kind, "output": None,
                "wraps": [], "why": self.why}


class CatBoostModel(_Stub):
    name = "catboost"
    why = ("CatBoost runs in its own venv (scripts/catboost_native_worker.py); it is wired through this interface "
           "as a candidate in P4, judged by the evaluation harness on the same folds and day-block intervals")


class NNModel(_Stub):
    name = "nn"
    why = "the tabular NN is a P4 candidate and needs /ml-preflight first (plans/PIPELINE_DESIGN_2026_09_10.md)"


class BlendModel(_Stub):
    name = "blend"
    why = "a blend's weights are fitted on inner out-of-fold predictions only; it arrives with the harness (P4)"


REGISTRY = {
    "lightgbm": LightGBMModel,
    "nonfill_body": lambda **kw: NonFillBodyModel(congestion=False, **kw),
    "nonfill_body_congestion": lambda **kw: NonFillBodyModel(congestion=True, **kw),
    "catboost": CatBoostModel,
    "nn": NNModel,
    "blend": BlendModel,
}


def make_model(name: str, **kwargs):
    if name not in REGISTRY:
        raise E.UnknownModelError(f"model {name!r} is not in the plug-in registry ({', '.join(REGISTRY)})")
    return REGISTRY[name](**kwargs)
