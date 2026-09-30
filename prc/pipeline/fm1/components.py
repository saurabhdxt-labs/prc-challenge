"""FM1 component functions: the value formulas, the ADS-B support gate, the ADS-B stack stage, the MTX calibration.

Each formula is the shipped one (tests pin the equality):
    base          rint(max(proxy - F, 1))  (delta target)  |  rint(max(F, 1))  (y target)
    stack         rint(max(proxy - (F + resid), 1))        prc/pipeline/adsb_stage.apply_stage
    CBV blend     rint(w * max(proxy - C, 1) + (1 - w) * partner_int)   v34_compose.matched_values (w = 0.5)
    MTX           mtx_calib.shift with the registered target / sigma / lambda rule (mtx_run.calibrate, mtx_run.build)
The stack serves through prc.pipeline.adsb_stage's contract (stage_frame + apply_stage) and adsb_stack.predict_residual
(its MLP half in adsb_nn_worker.py; BC-5). The effective gate is the stored stack gate AND the configured rule: a rule can
narrow the airports a stack serves, never widen them (predict_residual applies its own manifest gate regardless).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

from .. import adsb_stage as AST
from . import errors as E
from . import shipped

PREDICTOR_MIN_COVERAGE = 2          # adsb_stack.predict_residual zeroes coverage < 2 itself


# =================================================================================================================
# value formulas
# =================================================================================================================

def taxi_from_delta(proxy, delta) -> np.ndarray:
    return np.maximum(np.asarray(proxy, dtype="float64") - np.asarray(delta, dtype="float64"), 1.0)


def base_value(proxy, F, target: str) -> np.ndarray:
    if target == "delta":
        return np.rint(taxi_from_delta(proxy, F))
    if target == "y":
        return np.rint(np.maximum(np.asarray(F, dtype="float64"), 1.0))
    raise E.ConfigValueError(f"unknown target {target!r}")


def staged_value(proxy, F, resid) -> np.ndarray:
    return np.rint(np.maximum(np.asarray(proxy, dtype="float64")
                              - (np.asarray(F, dtype="float64") + np.asarray(resid, dtype="float64")), 1.0))


def blend_value(proxy, C, partner_int, w: float) -> np.ndarray:
    return np.rint(w * taxi_from_delta(proxy, C) + (1.0 - w) * np.asarray(partner_int, dtype="float64"))


STACK_BASES = ("lightgbm", "lgbcat")


def stack_base(F, C, base: str) -> np.ndarray:
    """The F the ADS-B stack corrects: `lightgbm` = the base model's F; `lgbcat` = 0.5 F + 0.5 C with C the matched
    CatBoost's delta (r13_cat.assemble's pred_lgbcat, the F v46's stack was trained and served on)."""
    F = np.asarray(F, dtype="float64")
    if base == "lightgbm":
        return F
    if base == "lgbcat":
        if C is None:
            raise E.PipelineError("adsb.stack.f_model lgbcat needs the matched CatBoost's F (the artefact has no CatBoost)")
        return 0.5 * F + 0.5 * np.asarray(C, dtype="float64")
    raise E.ConfigValueError(f"adsb.stack.f_model {base!r}: expected one of {list(STACK_BASES)}")


# =================================================================================================================
# the ADS-B support gate
# =================================================================================================================

def gate_manifest(models_dir) -> dict:
    gate, _ = AST.read_gate(models_dir)
    return dict(sorted(gate.items()))


def covered_share(ap, coverage, min_coverage: int) -> dict:
    s = pd.Series(np.asarray(coverage, dtype="float64") >= min_coverage).groupby(np.asarray(ap).astype(str)).mean()
    return {str(a): float(v) for a, v in s.items()}


def gate_coverage_ratio(train_share: dict, serve_share: dict, threshold: float) -> dict:
    """allowed iff threshold <= serve/train <= 1/threshold (and train > 0): the served coverage is within a factor of what
    the stack was trained on. Newly covered airports (EGLL 0.43 -> 0.90, the ADN.7 transfer failure) are refused."""
    out = {}
    for a, s in serve_share.items():
        t = train_share.get(a, 0.0)
        out[a] = bool(t > 0 and threshold <= s / t <= 1.0 / threshold)
    return out


def effective_gate(rule: str, manifest: dict, train_share: dict, serve_share: dict, threshold: float) -> dict:
    if rule == "manifest":
        return dict(manifest)
    if rule == "coverage_ratio":
        ratio = gate_coverage_ratio(train_share, serve_share, threshold)
        return {a: bool(ok and ratio.get(a, False)) for a, ok in manifest.items()}
    raise E.ConfigValueError(f"unknown support-gate rule {rule!r}")


# =================================================================================================================
# the ADS-B stack stage
# =================================================================================================================

def stack_values(frame: pd.DataFrame, F, feats, models_dir, gate: dict, min_coverage: int, predictor=None) -> tuple:
    """(values for every row of `frame`, eligible mask, residual, info). `frame` carries MVT_ID_mvt, ap, hr, proxy,
    coverage and the stack's feature columns; F is the base model's delta. Eligible = coverage >= min_coverage AND
    gate[ap]; the residual is zeroed elsewhere; the stage contract (adsb_stage.apply_stage) checks the rest."""
    if min_coverage < PREDICTOR_MIN_COVERAGE:
        raise E.ConfigValueError(f"adsb.min_coverage {min_coverage} < {PREDICTOR_MIN_COVERAGE}: the stored stack serves "
                                 f"coverage >= {PREDICTOR_MIN_COVERAGE} only")
    predictor = predictor or shipped.adsb_predictor()
    ids = frame.MVT_ID_mvt.to_numpy(dtype="float64")
    table = frame.set_index(pd.Index(ids))[[*feats, "coverage"]]
    fr = AST.stage_frame(ids, frame.ap.to_numpy(), frame.hr.to_numpy(dtype="float64"),
                         frame.proxy.to_numpy(dtype="float64"), np.asarray(F, dtype="float64"), table)
    eligible = (frame.coverage.to_numpy(dtype="float64") >= min_coverage) & \
        frame.ap.map(lambda a: bool(gate.get(a, False))).to_numpy(dtype=bool)
    seen = {}

    def fn(f):
        r = np.asarray(predictor(f, models_dir, oof=False), dtype="float64")
        if r.shape != (len(f),) or not np.isfinite(r).all():
            raise E.LaneError(f"stack predictor returned shape {r.shape} or non-finite values")
        seen["r"] = np.where(eligible, r, 0.0)
        return seen["r"]
    values, info = AST.apply_stage(fr, fn, gate)
    return values, eligible, seen["r"], info


def stack_feats(models_dir) -> list:
    return list(json.loads((pathlib.Path(models_dir) / "MANIFEST.json").read_text())["feats"])


# =================================================================================================================
# MTX monthly calibration (mtx_calib / mtx_run: the registered V1 rule)
# =================================================================================================================

def shift_cells(frame: pd.DataFrame, target: dict, lam: dict, mode: str) -> tuple:
    key = np.char.add(np.char.add(frame.ap.to_numpy().astype(str), "|"), frame.month.astype(int).astype(str).to_numpy())
    return shipped.mtx_calib().shift(frame.pred.to_numpy(dtype="float64"), key, target, lam, mode=mode)


class MtxCalibration:
    """fit: per airport, the excess mapping (c, e) on ALL training-year cells and its leave-month-out errors (mtx_run.build).
    apply: per scored (airport, UTC month) cell, T from the published cell of the scored year, sigma from the LMO errors,
    lambda by the rule, then mtx_calib.shift. Rows keyed by the UTC month of MVT_TIME, as v42 was built."""

    def __init__(self, params: dict | None = None):
        self.params = params            # {ap: {"c", "e", "lmo": records}}

    def fit(self, dep: pd.DataFrame, reference_csv) -> "MtxCalibration":
        run, src, cal = shipped.mtx_run(), shipped.mtx_source(), shipped.mtx_calib()
        pub = src.load_monthly_csv(reference_csv)
        cells = run.label_cells(dep, pub)
        # BC-31: mtx_calib.lmo_targets fits each month on the OTHER months (leave-month-out); one month -> an empty fit
        n = cells.groupby("ap").month.nunique()
        thin = sorted(str(a) for a in n.index[n < 2])
        if thin:
            raise E.SchemaError(f"MTX: leave-month-out needs >= 2 training months per airport; {thin} have 1")
        self.params = {}
        for ap in sorted(cells.ap.unique()):
            ca = cells[cells.ap == ap]
            c, e = cal.fit_excess(ca)
            self.params[ap] = {"c": float(c), "e": float(e), "lmo": cal.lmo_targets(ca).to_dict(orient="records")}
        return self

    def apply(self, frame: pd.DataFrame, published_csv, lambda_rule: str, mode: str) -> tuple:
        """frame: ap, month (UTC), year, pred. Returns (new pred, per-cell report)."""
        if self.params is None:
            raise E.PipelineError("MTX calibration is not fitted")
        run, src = shipped.mtx_run(), shipped.mtx_source()
        pub = src.load_monthly_csv(published_csv)
        years = sorted(set(frame.year.astype(int)))
        if len(years) != 1:
            raise E.SchemaError(f"MTX calibrates one scored year at a time (cells are keyed by airport x month); got {years}")
        pub = pub[pub.year == years[0]]
        params = {}
        for (ap, mo) in sorted(set(zip(frame.ap.astype(str), frame.month.astype(int)))):
            if ap not in self.params or pub[(pub.ap == ap) & (pub.month == mo)].empty:
                continue                                       # no published cell: the cell is left unchanged
            p = self.params[ap]
            params[(ap, mo)] = (p["c"], p["e"], pd.DataFrame(p["lmo"]))
        pub_cells = pub[pub.set_index(["ap", "month"]).index.isin(list(params))]
        new, rep = run.calibrate(frame[["ap", "month", "pred"]].reset_index(drop=True), pub_cells, params, runway=None,
                                 mode=mode, full_lambda=lambda_rule == "full")
        return np.asarray(new, dtype="float64"), rep

    def to_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "MtxCalibration":
        return cls(json.loads(text))
