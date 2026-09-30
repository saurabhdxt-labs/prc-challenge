"""E2W — E2 refit with the board's own loss, 12-month LOMO on 2025 (plans/PREREG_e2w_stake_weighted_refit_2026_09_23.md).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/e2w_build.py      -> reports/e2w_report.json

Arms on the 1,488 LIRF unmatched 2025 rows, both refit per held-out month on the other 11: L = E2's form by unweighted
log-loss (the recipe control); W = the same 4 coefficients minimising the stake-weighted Brier with w = (sp - nf)^2, the
per-row squared-error stake of the shipped mixture. Scored through `rrh_eval.score_rule` (W vs L) on the RRH.1 verdict
rows of all 12 months. No board data is read.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = pathlib.Path(__file__).resolve().parents[2]
RULE = ROOT / "data/rome_hunt_2026_09_23/agent_E_rules.py"
RULE_SHA = "bd01f4ed927d473585b6a182192073c59ed75fa220101db34035343c7629b606"
OUT = ROOT / "reports/e2w_report.json"
E2_BETA = np.array([0.685, 0.601, 1.254, -0.451])
SP_LO, SP_HI, SP_CAP = 3600.0, 24000.0, 24000.0
EPS = 1e-4


def design(p_R, k_in, turn) -> np.ndarray:
    """[logit(p_R), 1[k_in >= 1], turn, 1], p_R clipped to [1e-4, 1 - 1e-4] exactly as E2 does."""
    p = np.clip(np.asarray(p_R, float), EPS, 1 - EPS)
    return np.column_stack([np.log(p / (1 - p)), (np.asarray(k_in, float) >= 1).astype(float),
                            np.nan_to_num(np.asarray(turn, float), nan=0.0), np.ones(len(p))])


def predict(X, beta) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.asarray(X, float) @ np.asarray(beta, float))))


def fit_logloss(X, fill, w=None, beta0=None) -> np.ndarray:
    y = np.asarray(fill, float)

    def f(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z)), X.T @ (1 / (1 + np.exp(-z)) - y)
    return minimize(f, E2_BETA if beta0 is None else beta0, jac=True, method="L-BFGS-B").x


def fit_brier_w(X, fill, w, beta0=None) -> np.ndarray:
    """argmin sum w (fill - sigmoid(X b))^2; w is normalised to mean 1 for conditioning only."""
    y = np.asarray(fill, float)
    w = np.asarray(w, float) / np.mean(w)

    def f(b):
        p = 1 / (1 + np.exp(-(X @ b)))
        r = p - y
        return float(np.sum(w * r * r)), X.T @ (2 * w * r * p * (1 - p))
    return minimize(f, E2_BETA if beta0 is None else beta0, jac=True, method="L-BFGS-B").x


def lomo(X, fill, w, month, fit_fn, fit_mask=None) -> np.ndarray:
    """Out-of-month predictions: month m's rows scored by a fit on the fit_mask rows of the other months."""
    month = np.asarray(month)
    fit_mask = np.ones(len(month), bool) if fit_mask is None else np.asarray(fit_mask, bool)
    out = np.full(len(month), np.nan)
    for m in np.unique(month):
        tr = fit_mask & (month != m)
        b = fit_fn(X[tr], np.asarray(fill)[tr], np.asarray(w)[tr])
        out[month == m] = predict(X[month == m], b)
    return out


def verdict(ci_lo: float, minus_top3: float, g_janjul: float, g_by_month: dict) -> str:
    ok = (ci_lo > 0 and minus_top3 > 0 and g_janjul >= 0
          and sum(1 for v in g_by_month.values() if v > 0) >= 7)
    return "WORKING" if ok else "NOT WORKING"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def arc_rule_all_months(rows: pd.DataFrame, A) -> np.ndarray:
    files = sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))
    arr = A.arr_frame(A.load_raw(A.ARR_COLS, files))
    dep = pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt.to_numpy(), "ADEP_mvt": rows.ADEP_mvt.to_numpy(),
                        "ADES_mvt": rows.ADES_mvt.to_numpy(), "FLIGHT_mvt": rows.FLIGHT_mvt.to_numpy(),
                        "mvt": A.secs(rows.MVT_TIME_UTC_mvt)})
    j = A.join_arr(dep, arr)
    return A.sub_rule(rows.sp.to_numpy(float), A.secs(rows.SCHED_TIME_UTC_mvt), j.arr_lobt, j.arr_aobt3)


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "scripts/analysis"))
    import build_submission as bs
    import arr_row_clocks as A
    import hashlib
    if hashlib.sha256(RULE.read_bytes()).hexdigest() != RULE_SHA:
        raise AssertionError("frozen E2 rule bytes differ")
    if OUT.exists():
        raise FileExistsError(f"{OUT} exists")
    E = _load("agent_E_rules", RULE)
    R = _load("rrh_eval", ROOT / "scripts/analysis/rrh_eval.py")
    c = pd.read_parquet(ROOT / "reports/csf2/csf2_preds.parquet")[["MVT_ID_mvt", "month", "sp", "y", "day", "cls", "nf", "p_R"]]
    raw = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))],
                    ignore_index=True)
    rows = c.merge(raw.drop(columns=["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]), on="MVT_ID_mvt", how="left",
                   validate="1:1").reset_index(drop=True)
    f = E.stand_inbound_features(rows)
    sp, nf = rows.sp.to_numpy(float), rows.nf.to_numpy(float)
    band = (sp >= SP_LO) & (sp <= SP_HI) & f.k_in.notna().to_numpy()
    fill = (rows.cls == "fill").to_numpy()
    fit_mask = band & (rows.cls != "slip").to_numpy()
    X = design(rows.p_R, f.k_in, f.turn)
    w = (sp - nf) ** 2
    month = rows.month.to_numpy()
    pL = lomo(X, fill, w, month, lambda X_, y_, w_: fit_logloss(X_, y_), fit_mask)
    pW = lomo(X, fill, w, month, fit_brier_w, fit_mask)
    pE = predict(X, E2_BETA)
    V = R.verdict_rows(sp, arc_rule_all_months(rows, A)) & band
    args = lambda p_ref: (p_ref[V], sp[V], nf[V], rows.y.to_numpy(float)[V], month[V], rows.day.to_numpy()[V], bs.mixture)
    wl = R.score_rule(pW[V], *args(pL))
    by_month = {}
    for m in range(1, 13):
        mm = month[V] == m
        d = ((bs.mixture(pL[V], sp[V], nf[V]) - rows.y.to_numpy(float)[V]) ** 2
             - (bs.mixture(pW[V], sp[V], nf[V]) - rows.y.to_numpy(float)[V]) ** 2)
        by_month[m] = float(d[mm].sum() / 344_841.0)
    g_jj = by_month[1] + by_month[7]
    base = np.where(V, rows.p_R.to_numpy(float), np.nan)
    rep = {"prereg": "plans/PREREG_e2w_stake_weighted_refit_2026_09_23.md", "n_rows": int(len(rows)),
           "n_band": int(band.sum()), "n_fit": int(fit_mask.sum()), "n_verdict": int(V.sum()),
           "W_vs_L": wl, "W_vs_L_by_month": by_month, "W_vs_L_janjul": g_jj,
           "verdict": verdict(wl["ci95"][0], wl["minus_top3"], g_jj, by_month),
           "W_vs_R": R.score_rule(pW[V], *args(rows.p_R.to_numpy(float))),
           "L_vs_R": R.score_rule(pL[V], *args(rows.p_R.to_numpy(float))),
           "E2frozen_vs_R_in_sample_for_10_months": R.score_rule(pE[V], *args(rows.p_R.to_numpy(float))),
           "beta_E2": E2_BETA.tolist(),
           "beta_W_all12": fit_brier_w(X[fit_mask], fill[fit_mask], w[fit_mask]).tolist(),
           "beta_L_all12": fit_logloss(X[fit_mask], fill[fit_mask]).tolist()}
    D = bs.mixture(pW[V], sp[V], nf[V]) - bs.mixture(pL[V], sp[V], nf[V])
    e = rows.y.to_numpy(float)[V] - bs.mixture(pL[V], sp[V], nf[V])
    rep["k_W_vs_L"] = float((D * e).sum() / (D ** 2).sum())
    del base
    with OUT.open("x") as fh:
        json.dump(rep, fh, indent=1, default=float)
    print(json.dumps({k: rep[k] for k in ("verdict", "W_vs_L_janjul", "k_W_vs_L", "beta_E2", "beta_W_all12", "beta_L_all12",
                                          "n_verdict")}, indent=1, default=float))
    for k in ("W_vs_L", "W_vs_R", "L_vs_R", "E2frozen_vs_R_in_sample_for_10_months"):
        r = rep[k]
        print(k, {x: r[x] for x in ("g", "ci95", "minus_top3", "n_fired")})
    print("by month", {m: round(v, 1) for m, v in by_month.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
