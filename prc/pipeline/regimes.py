"""P3 — regime-gated multi-model plug-ins (plans/PIPELINE_DESIGN_2026_09_10.md "P3" and its interface detail).

    label_matched(y, sp, delta)      the ONE regime rule for matched rows (training rows only)
    check_probabilities(...)          the gate's boundary contract
    Design                            what a mixture receives: X plus the anchors its experts need (proxy, sp)
    GateModel                         LightGBM multiclass gate: early stop on masks.es, refit on masks.train
    MixtureModel                      gate × one expert per regime, each expert fitted ONLY on its regime's rows;
                                      predict = max(Σ_k p_k μ_k, 1), every μ recovered to taxi time and checked

Prototype and first measurement: scripts/regime_experts.py (arm REG, plans/PREREG_regime_experts_2026_09_11.md). Not in
models.REGISTRY yet: config wiring and save/load arrive with the fold-mode harness (P3 step 3); until then the mixture
refuses save/load with that reason.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass, field

import numpy as np

from . import errors as E
from . import legacy
from .models import Labels, Masks, _seeds, check_prediction

MATCHED = ("agree", "fill", "early", "late")
TAIL_S = 600.0
SCALES = ("delta", "sp_copy")


def label_matched(y, sp, delta) -> np.ndarray:
    """0 agree, 1 fill (|y − sp| ≤ 60, takes precedence), 2 early (δ < −600), 3 late (δ > +600)."""
    y, sp, delta = (np.asarray(v, dtype="float64") for v in (y, sp, delta))
    if not (y.shape == sp.shape == delta.shape):
        raise ValueError(f"y {y.shape}, sp {sp.shape}, delta {delta.shape} differ")
    if not (np.isfinite(y).all() and np.isfinite(sp).all() and np.isfinite(delta).all()):
        raise E.LabelLeakError("label_matched needs finite labels: it is only ever applied to training rows")
    fill_tol = legacy.matched().ls.FILL_TOL_S
    return np.select([np.abs(y - sp) <= fill_tol, delta < -TAIL_S, delta > TAIL_S], [1, 2, 3], default=0).astype("int64")


def check_probabilities(model, P, n_rows: int, k: int) -> np.ndarray:
    name = getattr(model, "name", type(model).__name__)
    if not isinstance(P, np.ndarray) or P.shape != (n_rows, k):
        raise E.LaneError(f"gate {name!r}: probabilities of shape {getattr(P, 'shape', type(P).__name__)}, expected ({n_rows}, {k})")
    if P.dtype != np.float64 or not np.isfinite(P).all():
        raise E.LaneError(f"gate {name!r}: probabilities must be finite float64")
    if (P < 0).any() or not np.allclose(P.sum(axis=1), 1.0, atol=1e-6):
        raise E.LaneError(f"gate {name!r}: probabilities must be non-negative and sum to 1 per row")
    return P


@dataclass
class Design:
    X: np.ndarray
    proxy: np.ndarray
    sp: np.ndarray

    def __post_init__(self):
        self.X = np.asarray(self.X)
        self.proxy = np.asarray(self.proxy, dtype="float64")
        self.sp = np.asarray(self.sp, dtype="float64")
        if self.X.ndim != 2 or not (len(self.X) == len(self.proxy) == len(self.sp)):
            raise ValueError(f"Design: X {self.X.shape}, proxy {self.proxy.shape}, sp {self.sp.shape} must align")


@dataclass
class GateState:
    booster: object
    n_classes: int
    info: dict


class GateModel:
    """P(regime | x): LightGBM multiclass, early-stopped on masks.es (multi_logloss), refit on masks.train at n_refit.
    Uncalibrated here; calibration on inner out-of-fold predictions is the harness's job (never the holdout)."""

    name = "gate_lightgbm"
    input_kind = "matrix"

    def __init__(self, params: dict | None = None, n_classes: int = len(MATCHED), nest: int = 60_000, patience: int = 400):
        ls = legacy.matched().ls
        self.params = dict(ls.P if params is None else params)
        self.n_classes, self.nest, self.patience = int(n_classes), int(nest), int(patience)

    def fit(self, X, y: Labels, masks: Masks, seeds) -> GateState:
        import lightgbm as lgb
        ls = legacy.matched().ls
        seeds = _seeds(seeds)
        if masks.fit is None:
            raise ValueError("GateModel.fit needs an early-stopping split")
        lab = getattr(y, "regime", None)
        if lab is None:
            raise ValueError("GateModel.fit needs Labels.regime")
        lab = np.asarray(lab)
        if ((lab[masks.train] < 0) | (lab[masks.train] >= self.n_classes)).any():
            raise ValueError("a training row has no valid regime label")
        cp = dict(self.params, objective="multiclass", num_class=self.n_classes, metric="multi_logloss", seed=seeds[0])
        b = lgb.train(cp, lgb.Dataset(X[masks.fit], label=lab[masks.fit]), num_boost_round=self.nest,
                      valid_sets=[lgb.Dataset(X[masks.es], label=lab[masks.es])],
                      callbacks=[lgb.early_stopping(self.patience, verbose=False), lgb.log_evaluation(0)])
        best = int(b.best_iteration)
        n_ref = int(ls.n_refit(best, int(masks.train.sum()), int(masks.fit.sum())))
        del b
        gc.collect()
        rb = lgb.train(cp, lgb.Dataset(X[masks.train], label=lab[masks.train]), num_boost_round=n_ref)
        return GateState(booster=rb, n_classes=self.n_classes, info={"best_iter": best, "n_ref": n_ref})

    def predict(self, state: GateState, X) -> np.ndarray:
        P = np.asarray(state.booster.predict(np.asarray(X), num_iteration=state.info["n_ref"]), dtype="float64")
        return check_probabilities(self, P, len(X), state.n_classes)

    def save(self, state, directory):
        raise NotImplementedError("GateModel.save arrives with the fold-mode harness (P3 step 3)")

    def load(self, source):
        raise NotImplementedError("GateModel.load arrives with the fold-mode harness (P3 step 3)")

    def describe(self) -> dict:
        return {"name": self.name, "implemented": True, "input_kind": self.input_kind, "output": "(n, k) probabilities",
                "wraps": ["lightgbm multiclass", "lgbm_submit.n_refit"], "n_classes": self.n_classes}


@dataclass
class MixtureState:
    gate: GateState
    experts: dict                       # regime index -> (state, scale); scale sp_copy holds the fill correction
    regimes: tuple
    info: dict = field(default_factory=dict)


class MixtureModel:
    """gate × experts. `experts`: {regime name: (plugin, scale)} with scale `delta` (a delta plug-in, recovered as
    max(proxy − δ̂, 1)) or `sp_copy` (no model: max(sp + c, 1), c = mean(y − sp) over the regime's training rows)."""

    name = "mixture"
    input_kind = "design"

    def __init__(self, gate: GateModel, experts: dict, regimes: tuple = MATCHED, min_expert_rows: int = 200):
        missing = [r for r in regimes if r not in experts]
        extra = [r for r in experts if r not in regimes]
        if missing or extra:
            raise ValueError(f"experts must cover exactly the regimes {regimes}: missing {missing}, extra {extra}")
        bad = {r: s for r, (_, s) in experts.items() if s not in SCALES}
        if bad:
            raise ValueError(f"expert scales {bad} not in {SCALES}")
        if int(min_expert_rows) < 1:
            raise ValueError("min_expert_rows must be positive")
        self.gate, self.experts, self.regimes = gate, dict(experts), tuple(regimes)
        self.min_expert_rows = int(min_expert_rows)

    def fit(self, D: Design, y: Labels, masks: Masks, seeds) -> MixtureState:
        seeds = _seeds(seeds)
        if y.regime is None:
            raise ValueError("MixtureModel.fit needs Labels.regime")
        lab = np.asarray(y.regime)
        gate_state = self.gate.fit(D.X, y, masks, seeds)
        experts, info = {}, {"n_train_by_regime": {}}
        for k, name in enumerate(self.regimes):
            plugin, scale = self.experts[name]
            rows = lab == k
            tr = masks.train & rows
            info["n_train_by_regime"][name] = int(tr.sum())
            if not tr.any():
                raise ValueError(f"regime {name!r} has no training row")
            if scale == "sp_copy":
                taxi = np.asarray(y.taxi, dtype="float64")
                experts[k] = (float(np.mean(taxi[tr] - D.sp[tr])), scale)
            else:
                sub = Masks(train=tr, fit=masks.fit & rows, es=masks.es & rows)
                n_fit, n_es = int(sub.fit.sum()), int(sub.es.sum())
                if n_fit < self.min_expert_rows or n_es < max(1, self.min_expert_rows // 4):
                    raise E.LaneError(f"regime {name!r}: {n_fit} fit / {n_es} stopping rows, below the minimum "
                                      f"({self.min_expert_rows} / {max(1, self.min_expert_rows // 4)}) an expert is fitted on")
                experts[k] = (plugin.fit(D.X, y, sub, seeds), scale)
        return MixtureState(gate=gate_state, experts=experts, regimes=self.regimes, info=info)

    def expert_values(self, state: MixtureState, D: Design) -> tuple:
        """(MU (n, k) taxi seconds, floor binds per regime)."""
        cols, binds = [], {}
        for k, name in enumerate(state.regimes):
            st, scale = state.experts[k]
            if scale == "sp_copy":
                raw = D.sp + st
            else:
                plugin = self.experts[name][0]
                raw = D.proxy - check_prediction(plugin, plugin.predict(st, D.X), len(D.X))
            binds[name] = int((raw < 1.0).sum())
            cols.append(np.maximum(raw, 1.0))
        return np.column_stack(cols), binds

    def predict(self, state: MixtureState, D: Design) -> np.ndarray:
        P = self.gate.predict(state.gate, D.X)
        MU, binds = self.expert_values(state, D)
        state.info["floor_binds_last_predict"] = binds
        return check_prediction(self, np.maximum((P * MU).sum(axis=1), 1.0), len(D.X))

    def save(self, state, directory):
        raise NotImplementedError("MixtureModel.save arrives with the fold-mode harness (P3 step 3)")

    def load(self, source):
        raise NotImplementedError("MixtureModel.load arrives with the fold-mode harness (P3 step 3)")

    def describe(self) -> dict:
        return {"name": self.name, "implemented": True, "input_kind": self.input_kind,
                "output": "taxi seconds (mixture mean, floored)", "regimes": list(self.regimes),
                "gate": self.gate.describe(), "experts": {r: {"plugin": getattr(p, "name", type(p).__name__), "scale": s}
                                                          for r, (p, s) in self.experts.items()},
                "wraps": ["prc.pipeline.models.LightGBMModel (experts)", "GateModel"]}
