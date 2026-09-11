"""SEL — a learned model selector over the stored fold-A experts (plans/PREREG_model_selector_2026_09_11.md, screen).

    python3.11 -B scripts/model_select.py        -> reports/model_select.json

Every expert is a stored out-of-sample prediction of the SAME 339,015 holdout rows. Selectors are CROSS-FITTED BY MONTH:
fitted on January's rows and applied to July's, and the reverse, so no row is scored by a selector that saw it.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import bundle_stage0 as B  # noqa: E402  (F / CAP / C_delta loaders and their guards)
import nm_param_diag as N  # noqa: E402  (holdout_days)
from prc.pipeline import scoring as SC  # noqa: E402

CACHE = ROOT / "data" / "cache_stand"
OUT = ROOT / "reports" / "model_select.json"
GUARDS = {"F": 222.5632, "Y": 260.6511, "W": 223.533, "D": 224.3119, "PA": 225.388}   # each record's own RMSE
GUARD_TOL = 5e-4
BUILD_BAR, PER_AIRPORT_BAR = 500.0, 200.0


# ------------------------------------------------------------------ pure pieces (unit-tested)

def simplex_ls(E: np.ndarray, y: np.ndarray) -> np.ndarray:
    """argmin_w ||E w − y||² subject to w ≥ 0, Σw = 1 (SLSQP on the Gram form). E: (n, k)."""
    from scipy.optimize import minimize
    E, y = np.asarray(E, dtype="float64"), np.asarray(y, dtype="float64")
    if E.ndim != 2 or len(E) != len(y) or len(y) == 0:
        raise ValueError(f"E {E.shape} and y {y.shape} must align and be non-empty")
    k = E.shape[1]
    G, b = E.T @ E, E.T @ y
    scale = float(np.abs(G).max()) or 1.0
    res = minimize(lambda w: (w @ G @ w - 2 * b @ w) / scale, np.full(k, 1.0 / k),
                   jac=lambda w: (2 * G @ w - 2 * b) / scale, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones(k)}],
                   options={"maxiter": 500, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"simplex least squares did not converge: {res.message}")
    w = np.clip(res.x, 0.0, None)
    return w / w.sum()


def crossfit(month: np.ndarray, fit_predict) -> np.ndarray:
    """Apply fit_predict(fit_rows, apply_rows) -> predictions for apply_rows, once per direction between the TWO months."""
    month = np.asarray(month)
    ms = sorted(set(month.tolist()))
    if len(ms) != 2:
        raise ValueError(f"cross-fitting needs exactly two months, got {ms}")
    out = np.full(len(month), np.nan)
    for a, b in ((ms[0], ms[1]), (ms[1], ms[0])):
        fit, app = month == a, month == b
        out[app] = fit_predict(fit, app)
    if np.isnan(out).any():
        raise AssertionError("a row got no cross-fitted prediction")
    return out


def stack_global(E, y, month) -> tuple:
    weights = {}

    def fp(fit, app):
        w = simplex_ls(E[fit], y[fit])
        weights[int(np.asarray(month)[app][0])] = w.tolist()
        return E[app] @ w
    return np.maximum(crossfit(month, fp), 1.0), weights


def stack_by_group(E, y, month, group) -> np.ndarray:
    group = np.asarray(group)

    def fp(fit, app):
        pred = np.empty(int(app.sum()))
        g_app = group[app]
        for g in np.unique(g_app):
            f = fit & (group == g)
            if f.sum() < E.shape[1] * 20:           # too few rows for a per-group fit: the global weights
                f = fit
            w = simplex_ls(E[f], y[f])
            pred[g_app == g] = E[app][g_app == g] @ w
        return pred
    return np.maximum(crossfit(month, fp), 1.0)


def oracle(E, y) -> np.ndarray:
    return E[np.arange(len(y)), np.abs(E - y[:, None]).argmin(axis=1)]


# ------------------------------------------------------------------ learned selectors (smoke-tested)

def _day_split(day_codes: np.ndarray) -> np.ndarray:
    return day_codes % 2 == 0


def lgbm_stacker(Xm: np.ndarray, y: np.ndarray, month, day_codes, seed: int = 0) -> np.ndarray:
    import lightgbm as lgb
    p = dict(objective="regression", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.9,
             verbosity=-1, num_threads=4, seed=seed)

    def fp(fit, app):
        es = fit & _day_split(day_codes)
        tr = fit & ~_day_split(day_codes)
        b = lgb.train(p, lgb.Dataset(Xm[tr], y[tr]), 5_000, valid_sets=[lgb.Dataset(Xm[es], y[es])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return b.predict(Xm[app], num_iteration=b.best_iteration)
    return np.maximum(crossfit(month, fp), 1.0)


def lgbm_classifier_selector(Xm, E, y, month, day_codes, seed: int = 0) -> np.ndarray:
    import lightgbm as lgb
    k = E.shape[1]
    lab = np.abs(E - y[:, None]).argmin(axis=1)
    p = dict(objective="multiclass", num_class=k, learning_rate=0.05, num_leaves=31, min_data_in_leaf=100,
             verbosity=-1, num_threads=4, seed=seed)

    def fp(fit, app):
        es = fit & _day_split(day_codes)
        tr = fit & ~_day_split(day_codes)
        b = lgb.train(p, lgb.Dataset(Xm[tr], lab[tr]), 2_000, valid_sets=[lgb.Dataset(Xm[es], lab[es])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        P = b.predict(Xm[app], num_iteration=b.best_iteration)
        return (P * E[app]).sum(axis=1)
    return np.maximum(crossfit(month, fp), 1.0)


# ------------------------------------------------------------------ real data

def _taxi(proxy, d):
    return np.maximum(np.asarray(proxy, dtype="float64") - np.asarray(d, dtype="float64"), 1.0)


def load_experts() -> tuple:
    m = B.load()["m"]
    y, proxy = m.y.to_numpy(dtype="float64"), m.proxy.to_numpy(dtype="float64")
    ex = {"F": B.taxi(proxy, m.treatment), "CAP": B.taxi(proxy, m.cap), "CB": B.taxi(proxy, m.cb)}
    rec = lambda f, cols: pd.read_parquet(CACHE / f, columns=["row", "y", *cols])
    for name, f, col in (("Y", "fold_preds_queue_ytarget.parquet", "Y"), ("W", "fold_preds_queue_weather.parquet", "treatment"),
                         ("D", "fold_preds_queue_day.parquet", "treatment"), ("PA", "fold_v4_preds.parquet", "A3")):
        r = rec(f, [col])
        if not (np.array_equal(r.row.to_numpy(), m.row.to_numpy()) and np.array_equal(r.y.to_numpy(), y)):
            raise AssertionError(f"{f} is not row-aligned with arm F")
        ex[name] = _taxi(proxy, r[col])
    reg, _ = SC.read_record(CACHE / "regime_experts_preds.parquet", "taxi_time", tag="REG")
    if not (np.array_equal(reg.row.to_numpy(), m.row.to_numpy()) and np.array_equal(reg.y.to_numpy(), y)):
        raise AssertionError("REG record is not row-aligned with arm F")
    for c in ("REG", "mu_agree", "mu_early", "mu_late", "mu_fill"):
        ex[c] = reg[c].to_numpy(dtype="float64")
    guards = {}
    for name, want in GUARDS.items():
        got = B.rmse(y, ex[name])
        if abs(got - want) > GUARD_TOL:
            raise AssertionError(f"guard {name}: {got:.4f} vs {want} (BC-2: convention?)")
        guards[name] = got
    meta = {"ap": m.ap.astype(str).to_numpy(), "month": m.month.to_numpy(), "row": m.row.to_numpy(), "proxy": proxy,
            "sp": m.sp.to_numpy(dtype="float64"), "delta": m.delta.to_numpy(dtype="float64"),
            "P": reg[["p_agree", "p_fill", "p_early", "p_late"]].to_numpy(dtype="float64")}
    return y, ex, meta, guards


def hours(rows: np.ndarray, y: np.ndarray) -> np.ndarray:
    files = sorted(glob.glob(str(CACHE / "training_2025-*.parquet")))
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    start = np.concatenate([[0], np.cumsum(counts)])
    hr = np.empty(len(rows))
    for m in (1, 7):
        s = (rows >= start[m - 1]) & (rows < start[m])
        t = pd.read_parquet(files[m - 1], columns=["y", "hr"])
        loc = rows[s] - start[m - 1]
        if not np.array_equal(t.y.to_numpy()[loc], y[s]):
            raise AssertionError("hours do not align")
        hr[s] = t.hr.to_numpy()[loc]
    return hr


def main() -> int:
    y, ex, meta, guards = load_experts()
    names = list(ex)
    E = np.column_stack([ex[k] for k in names])
    month, ap = meta["month"], meta["ap"]
    day = N.holdout_days(meta["row"], month, y)
    day_codes = pd.factorize(pd.Series(day))[0]
    ap_code = pd.factorize(pd.Series(ap))[0].astype("float64")
    Xm = np.column_stack([E, ap_code, hours(meta["row"], y), meta["proxy"], meta["sp"], meta["P"]])
    arms = {"S0_CAP": ex["CAP"]}
    arms["S1_global_stack"], weights = stack_global(E, y, month)
    arms["S2_per_airport_stack"] = stack_by_group(E, y, month, ap)
    arms["S3_lgbm_stacker"] = lgbm_stacker(Xm, y, month, day_codes)
    arms["S4_classifier_selector"] = lgbm_classifier_selector(Xm, E, y, month, day_codes)
    orc = oracle(E, y)
    tail = np.abs(meta["delta"]) > 600
    cuts = {"LIRF": ap == "LIRF", "tail_|delta|>600": tail, "clock_agree": ~tail}
    res = {"tag": "SEL", "prereg": "plans/PREREG_model_selector_2026_09_11.md",
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "experts": names, "guards": guards, "rmse": {k: B.rmse(y, v) for k, v in {**ex, **arms}.items()},
           "global_weights_by_fit_month": {str(k): dict(zip(names, np.round(v, 4))) for k, v in weights.items()},
           "vs_CAP": {}, "oracle_NOT_A_RESULT": SC.paired(y, ex["CAP"], orc, day, month, n_boot=200)["net"]}
    for k, v in arms.items():
        if k == "S0_CAP":
            continue
        res["vs_CAP"][k] = SC.paired(y, ex["CAP"], v, day, month, cuts=cuts)
        res["vs_CAP"][k]["by_airport"] = {a: SC.weighted_fold_mse(float((((y - ex["CAP"]) ** 2 - (y - v) ** 2)[ap == a]).sum()), len(y))
                                          for a in sorted(set(ap))}
    res["S2_vs_S1"] = SC.paired(y, arms["S1_global_stack"], arms["S2_per_airport_stack"], day, month)
    best = max(res["vs_CAP"], key=lambda k: res["vs_CAP"][k]["net"])
    b = res["vs_CAP"][best]
    res["decision"] = {"best": best, "net": b["net"], "lower": b["dayblock95"][0],
                       "BUILD": bool(b["net"] >= BUILD_BAR and b["dayblock95"][0] > 0),
                       "per_airport_matters": bool(res["S2_vs_S1"]["net"] >= PER_AIRPORT_BAR and res["S2_vs_S1"]["dayblock95"][0] > 0)}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({"rmse": {k: round(v, 3) for k, v in res["rmse"].items()}, "oracle": round(res["oracle_NOT_A_RESULT"]),
                      "decision": res["decision"]}, default=float))
    for k, v in res["vs_CAP"].items():
        print(f"{k:24s} vs CAP net {v['net']:+8.1f} [{v['dayblock95'][0]:+.1f}, {v['dayblock95'][1]:+.1f}]  cuts "
              f"{ {c: round(x['net']) for c, x in v['cuts'].items()} }")
    s = res["S2_vs_S1"]
    print(f"S2 per-airport vs S1 global: {s['net']:+.1f} [{s['dayblock95'][0]:+.1f}, {s['dayblock95'][1]:+.1f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
