"""BND Stage 0 — does C_delta still add to CAP? (plans/PREREG_bundle_cap_catboost_2026_09_11.md, zero compute)

    python3.11 -B scripts/bundle_stage0.py        -> reports/bundle_stage0.json

Reads three stored fold-A records (arm F, CAP, C_delta), proves each reproduces its own recorded RMSE and that all three
are the same rows (BC-2 guards), then scores B0 = 0.5·C_delta + 0.5·CAP against CAP with the date-block interval. Never
fits anything; never a ship.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nm_param_diag as N  # noqa: E402  (holdout_days, dayblock_sum, to_board: one tested implementation)

CACHE = ROOT / "data" / "cache_stand"
REPORTS = ROOT / "reports"
OUT = REPORTS / "bundle_stage0.json"
BLEND = 0.5                         # fixed by the C_delta screen's registration before it ran; never fitted here
G0_NET = 300.0                      # weighted fold MSE vs CAP
GUARD_TOL = 5e-4                    # seconds of RMSE
TAIL_S = 600.0


def taxi(proxy: np.ndarray, delta_hat: np.ndarray) -> np.ndarray:
    """Every DELTA-scale record in this lane recovers as max(proxy − delta_hat, 1) (BC-2)."""
    return np.maximum(np.asarray(proxy, dtype="float64") - np.asarray(delta_hat, dtype="float64"), 1.0)


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def blend(a: np.ndarray, b: np.ndarray, w: float = BLEND) -> np.ndarray:
    """w·a + (1 − w)·b on the taxi-time scale; both inputs must already be floored taxi times."""
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"blend weight {w} outside [0, 1]")
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    if a.shape != b.shape:
        raise ValueError(f"blend of shapes {a.shape} and {b.shape}")
    if (a < 1.0).any() or (b < 1.0).any():
        raise ValueError("blend inputs must be floored taxi times (>= 1 s)")
    return w * a + (1.0 - w) * b


def guard(name: str, got: float, want: float, tol: float = GUARD_TOL) -> dict:
    if not np.isfinite(want) or abs(got - want) > tol:
        raise AssertionError(f"guard {name}: {got:.6f} vs its record's {want:.6f} (BC-2: wrong convention or rows?)")
    return {"got": got, "want": want}


def best_weight_diagnostic(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """The w in [0, 1] minimising Σ(y − (w·a + (1−w)·b))² in closed form — a DIAGNOSTIC printed, never used."""
    d = a - b
    den = float((d * d).sum())
    if den == 0.0:
        return float("nan")
    return float(np.clip(((y - b) * d).sum() / den, 0.0, 1.0))


def cuts(y, base, arm, ap, month, fill, delta, n_fold) -> dict:
    g = (y - base) ** 2 - (y - arm) ** 2
    out = {"by_airport_board": {a: N.to_board(float(g[ap == a].sum()), n_fold) for a in sorted(set(ap))},
           "by_month_board": {int(m): N.to_board(float(g[month == m].sum()), n_fold) for m in sorted(set(month))},
           "fill_board": N.to_board(float(g[fill].sum()), n_fold),
           "tail_board": N.to_board(float(g[np.abs(delta) > TAIL_S].sum()), n_fold)}
    for lab, m in (("fill", fill), ("tail", np.abs(delta) > TAIL_S)):
        out[f"gain_{lab}_s"] = rmse(y[m], base[m]) - rmse(y[m], arm[m])
    return out


def load() -> dict:
    f = pd.read_parquet(CACHE / "fold_preds_queue_order.parquet",
                        columns=["row", "month", "ap", "y", "proxy", "delta", "sp", "treatment"])
    c = pd.read_parquet(CACHE / "capacity_f_preds.parquet", columns=["row", "y", "proxy", "cap"])
    h = pd.read_parquet(CACHE / "catboost_native_input.parquet", columns=["row", "score_y", "score_proxy", "holdout"])
    h = h[h.holdout].reset_index(drop=True)
    cb = np.load(CACHE / "catboost_native_delta.npy")
    if len(cb) != len(h):
        raise AssertionError(f"C_delta: {len(cb)} predictions for {len(h)} holdout rows")
    h = h.assign(cb=cb)
    m = f.merge(c, on="row", suffixes=("", "_c"), validate="one_to_one").merge(h, on="row", validate="one_to_one")
    if len(m) != len(f) or len(m) != len(c) or len(m) != len(h):
        raise AssertionError(f"the three records are not the same rows: F {len(f)}, CAP {len(c)}, C {len(h)}, joined {len(m)}")
    for other in ("y_c", "score_y"):
        if not np.array_equal(m.y.to_numpy(), m[other].to_numpy()):
            raise AssertionError(f"labels differ between records ({other}) on the joined rows")
    for other in ("proxy_c", "score_proxy"):
        if not np.array_equal(m.proxy.to_numpy(), m[other].to_numpy()):
            raise AssertionError(f"proxies differ between records ({other})")
    return {"m": m}


def main() -> int:
    m = load()["m"]
    y, proxy = m.y.to_numpy(dtype="float64"), m.proxy.to_numpy(dtype="float64")
    F, CAP, C = taxi(proxy, m.treatment), taxi(proxy, m.cap), taxi(proxy, m.cb)
    cap_rec = json.loads((REPORTS / "capacity_f.json").read_text())["rmse"]
    cb_rec = json.loads((REPORTS / "catboost_native.json").read_text())
    guards = {"F": guard("F", rmse(y, F), cap_rec["F"]), "CAP": guard("CAP", rmse(y, CAP), cap_rec["CAP"]),
              "C_delta": guard("C_delta", rmse(y, C), cb_rec["C_delta"]["rmse"]),
              "C_delta_blend_F": guard("0.5C+0.5F", rmse(y, blend(C, F)), cb_rec["C_delta_blend"]["rmse"])}
    B0 = blend(C, CAP)
    n = len(y)
    month, ap = m.month.to_numpy(), m.ap.astype(str).to_numpy()
    day = N.holdout_days(m.row.to_numpy(), month, y)
    fill = (np.abs(y - m.sp.to_numpy()) <= 60)
    delta = m.delta.to_numpy()
    res = {"tag": "BND-stage0", "prereg": "plans/PREREG_bundle_cap_catboost_2026_09_11.md#stage-0", "n_rows": n,
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "guards": guards, "rmse": {"F": rmse(y, F), "CAP": rmse(y, CAP), "C_delta": rmse(y, C), "B0": rmse(y, B0)}}
    for base_name, base in (("CAP", CAP), ("F", F)):
        g = (y - base) ** 2 - (y - B0) ** 2
        tot, lo, hi, nd = N.dayblock_sum(g, day, month)
        res[f"B0_vs_{base_name}"] = {"net_weighted_fold_mse": N.to_board(tot, n), "dayblock95_board": [N.to_board(lo, n), N.to_board(hi, n)],
                                     "n_dates": nd, "gain_s": rmse(y, base) - rmse(y, B0),
                                     **cuts(y, base, B0, ap, month, fill, delta, n)}
    v = res["B0_vs_CAP"]
    res["G0"] = {"net_bar": G0_NET, "net": v["net_weighted_fold_mse"], "lower": v["dayblock95_board"][0],
                 "pass": bool(v["net_weighted_fold_mse"] >= G0_NET and v["dayblock95_board"][0] > 0)}
    res["diagnostic_holdout_optimal_w_NOT_USED"] = best_weight_diagnostic(y, C, CAP)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("rmse", "G0", "diagnostic_holdout_optimal_w_NOT_USED")}, default=float))
    for b in ("B0_vs_CAP", "B0_vs_F"):
        x = res[b]
        print(f"{b}: net {x['net_weighted_fold_mse']:+.1f} [{x['dayblock95_board'][0]:+.1f}, {x['dayblock95_board'][1]:+.1f}] "
              f"months {x['by_month_board']} fill {x['gain_fill_s']:+.2f}s tail {x['gain_tail_s']:+.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
