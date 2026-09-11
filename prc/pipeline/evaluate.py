"""P3 — the fold-mode harness (plans/PIPELINE_DESIGN_2026_09_10.md "P3", acceptance M2).

The SAME pipeline code in fold mode: a candidate (a model configuration for a lane) is fitted on fold A's training months
and predicts its holdout; the predictions land in a fold record stamped with their convention, the candidate, the fold,
and the hashes of the code that ran (BC-2, BC-7). Candidates are compared with `scoring.paired` on identical rows.

    load_matched_fold(stand, qcache, ocache, months, features)   lgbm_fold.load_fold (arm F's loader), incumbent encoder
    predict_single(fold, cand)       LightGBMModel (lgbm_submit's early_stop / refit / seed mean) -> per-seed + pooled δ̂
    predict_mixture(fold, cand)      regimes.MixtureModel (gate × per-regime delta experts) -> taxi seconds
    evaluate(...)                    fit, predict, write the record; returns its path

Acceptance M2 (tests/pipeline/test_pipeline_evaluate.py): `predict_single` equals `lgbm_fold.fit_arm` bit for bit on
the same fold at one thread. The real-data reproduction of arm F's record (222.5632; lr 0.01, 3 seeds, ≈ 2 h, ≈ 5 GB)
is a gated heavy test, not run by default. Open (named): the separated (BC-1-clean) encoder in fold mode; LOMO folds;
inner out-of-fold predictions for stacking.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import errors as E
from . import legacy, scoring, write
from .models import Labels, LightGBMModel, Masks, check_prediction
from . import regimes as R

FEATURE_SETS = {"queue_order": "FEATS_QUEUE_ORDER", "queue": "FEATS_QUEUE", "base": "FEATS"}
KINDS = ("single", "mixture")
PARAM_KEYS = ("learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction", "num_threads")
CANDIDATE_KEYS = {"tag", "prereg", "kind", "features", "seeds", "params", "nest", "patience", "min_expert_rows"}


@dataclass(frozen=True)
class Candidate:
    tag: str
    prereg: str
    kind: str = "single"
    features: str = "queue_order"
    seeds: tuple = (0,)
    params: dict = field(default_factory=dict)          # overrides of lgbm_submit.P, keys in PARAM_KEYS
    nest: int = 60_000
    patience: int = 400
    min_expert_rows: int = 200                        # mixture only: fewer fit rows -> refused by name

    def lgbm_params(self) -> dict:
        return dict(legacy.matched().ls.P, **self.params)


def parse_candidate(raw: dict) -> Candidate:
    """Strict: unknown keys, an unknown kind / feature set / parameter, or non-distinct seeds are refused by name."""
    if not isinstance(raw, dict):
        raise E.ConfigTypeError(f"a candidate must be a mapping, got {type(raw).__name__}")
    unknown = sorted(set(raw) - CANDIDATE_KEYS)
    if unknown:
        raise E.UnknownKeyError(f"candidate keys {unknown} are not in {sorted(CANDIDATE_KEYS)}")
    for k in ("tag", "prereg"):
        if not isinstance(raw.get(k), str) or not raw[k]:
            raise E.ConfigValueError(f"candidate.{k} must be a non-empty string (every candidate answers to a prereg)")
    kind = raw.get("kind", "single")
    if kind not in KINDS:
        raise E.ConfigValueError(f"candidate.kind {kind!r} not in {KINDS}")
    feats = raw.get("features", "queue_order")
    if feats not in FEATURE_SETS:
        raise E.ConfigValueError(f"candidate.features {feats!r} not in {sorted(FEATURE_SETS)}")
    params = dict(raw.get("params", {}))
    bad = sorted(set(params) - set(PARAM_KEYS))
    if bad:
        raise E.UnknownKeyError(f"candidate.params {bad} not in {PARAM_KEYS}")
    seeds = tuple(int(s) for s in raw.get("seeds", (0,)))
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise E.ConfigValueError(f"candidate.seeds must be distinct non-negative integers, got {seeds}")
    nest, patience = int(raw.get("nest", 60_000)), int(raw.get("patience", 400))
    min_rows = int(raw.get("min_expert_rows", 200))
    if nest < 1 or patience < 1 or min_rows < 1:
        raise E.ConfigValueError("candidate.nest, candidate.patience and candidate.min_expert_rows must be positive")
    return Candidate(tag=raw["tag"], prereg=raw["prereg"], kind=kind, features=feats, seeds=seeds, params=params,
                     nest=nest, patience=patience, min_expert_rows=min_rows)


def load_matched_fold(stand_cache, qcache, ocache, months, features: str = "queue_order") -> dict:
    """arm F's loader exactly: lgbm_fold.load_fold with the feature list named by `features` and fold_masks."""
    lf = legacy.fold().lf
    feats = list(getattr(lf, FEATURE_SETS[features]))
    fold = lf.load_fold(pathlib.Path(stand_cache), tuple(months), feats, qcache=qcache, ocache=ocache)
    fold["features"] = features
    return fold


def _labels(fold: dict, with_regime: bool = False) -> Labels:
    reg = None
    if with_regime:
        tr = fold["tr"]
        reg = np.full(len(fold["y"]), -1, dtype="int64")
        reg[tr] = R.label_matched(fold["y"][tr], fold["sp"][tr], fold["dlt"][tr])
    return Labels(taxi=fold["y"], delta=fold["dlt"], proxy=fold["proxy"], regime=reg)


def _masks(fold: dict) -> Masks:
    return Masks(train=fold["tr"], fit=fold["fit"], es=fold["es"])


def predict_single(fold: dict, cand: Candidate) -> dict:
    """{'per_seed': {seed: δ̂ on the holdout}, 'pooled': mean δ̂, 'info': the early-stopping record}."""
    ls = legacy.matched().ls
    params = cand.lgbm_params()
    model = LightGBMModel(params=params, target="delta", nest=cand.nest, patience=cand.patience)
    state = model.fit(fold["X"], _labels(fold), _masks(fold), cand.seeds)
    Xte = fold["X"][fold["te"]]
    per_seed = {s: check_prediction(model, np.asarray(ls.predict_delta(state.boosters[s], Xte, params), dtype="float64"),
                                    len(Xte)) for s in cand.seeds}
    pooled = check_prediction(model, model.predict(state, Xte), len(Xte))
    return {"per_seed": per_seed, "pooled": pooled, "info": dict(state.info)}


def predict_mixture(fold: dict, cand: Candidate) -> dict:
    """{'taxi': the mixture's holdout taxi seconds, 'P': gate probabilities, 'info'}."""
    params = cand.lgbm_params()
    gate = R.GateModel(params=params, nest=cand.nest, patience=cand.patience)
    expert = lambda: (LightGBMModel(params=params, target="delta", nest=cand.nest, patience=cand.patience), "delta")
    mix = R.MixtureModel(gate, {"agree": expert(), "fill": (None, "sp_copy"), "early": expert(), "late": expert()},
                         min_expert_rows=cand.min_expert_rows)
    D = R.Design(fold["X"], fold["proxy"], fold["sp"])
    state = mix.fit(D, _labels(fold, with_regime=True), _masks(fold), cand.seeds)
    te = fold["te"]
    Dte = R.Design(fold["X"][te], fold["proxy"][te], fold["sp"][te])
    return {"taxi": mix.predict(state, Dte), "P": mix.gate.predict(state.gate, Dte.X), "info": dict(state.info)}


def holdout_frame(fold: dict, arms: dict) -> pd.DataFrame:
    """fold['base'] (row, month, ap, y, proxy, delta, sp) + one column per arm, each aligned to the holdout rows."""
    base = fold["base"].copy()
    for name, v in arms.items():
        v = np.asarray(v, dtype="float64")
        if len(v) != len(base):
            raise E.LaneError(f"arm {name!r} has {len(v)} values for {len(base)} holdout rows")
        base[name] = v
    return base


def evaluate(fold: dict, cand: Candidate, out_dir, root: pathlib.Path | None = None) -> pathlib.Path:
    """Fit `cand` on the fold, write its stamped record to out_dir/<tag>.parquet (refuses to overwrite)."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cand.tag}.parquet"
    if path.exists():
        raise E.ValidationError(f"refusing to overwrite {path}")
    if cand.kind == "single":
        res = predict_single(fold, cand)
        arms = {f"delta_hat_seed{s}": v for s, v in res["per_seed"].items()}
        arms["delta_hat"] = res["pooled"]
        convention = "delta"
    else:
        res = predict_mixture(fold, cand)
        arms = {"taxi_hat": res["taxi"], **{f"p_{r}": res["P"][:, k] for k, r in enumerate(R.MATCHED)}}
        convention = "taxi_time"
    root = pathlib.Path(root) if root is not None else legacy.ROOT
    meta = {"candidate": cand.__dict__, "fold": {"features": fold.get("features"), "n_holdout": int(fold["te"].sum()),
                                                 "n_train": int(fold["tr"].sum()), "n_months": fold.get("n_months")},
            "info": res["info"], "code": write.code_state(root)}
    scoring.write_record(holdout_frame(fold, arms), path, convention, cand.tag, cand.prereg, meta)
    return path
