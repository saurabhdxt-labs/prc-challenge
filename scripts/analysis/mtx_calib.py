"""Published-monthly-mean calibration — arm MTX (plans/PREREG_mtx_monthly_calibration_2026_09_28.md).

Mapping (registered, fitted on 2025 only): the published mean covers the P = VALID - NO_REF flights of an
airport-month; our population is all TF departures. The excluded flights are NO_REF (NR, valid but no stand-runway
reference) and INV = TF - VALID (filtered: de-icing after off-block, taxi-out > 120 min, missing data). Model their
excess over the published mean with two per-airport constants:

    TF * (mean_all - pub) = NR * c + INV * e          (least squares, no intercept)
    target T = pub + (NR * c + INV * e) / TF

Shift: delta = lambda * (T - mean(pred)); lambda = max(0, 1 - sigma^2 / gap^2) (James-Stein on the gap), sigma = the
2025 leave-month-out RMS mapping error of the same airport in the same invalid-share regime (INV/TF >= 2 % or not).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_THR = 0.02


def fit_excess(cells: pd.DataFrame) -> tuple[float, float]:
    """(c, e) from cells with columns n, mean_label, pub_mean_s, NR, INV. Min-norm least squares, so a column that
    is all zero gets coefficient 0."""
    if len(cells) == 0:
        raise ValueError("fit_excess: empty cells")
    A = cells[["NR", "INV"]].to_numpy(float)
    x = (cells.n * (cells.mean_label - cells.pub_mean_s)).to_numpy(float)
    if not (np.isfinite(A).all() and np.isfinite(x).all()):
        raise ValueError("fit_excess: non-finite input")
    coef = np.linalg.lstsq(A, x, rcond=None)[0]
    c, e = (float(v) for v in coef)
    return c, (0.0 if not A[:, 1].any() else e)


def target_mean(pub, NR, INV, TF, c, e):
    """T = pub + (NR*c + INV*e)/TF (seconds)."""
    TF = np.asarray(TF, float)
    if (TF <= 0).any():
        raise ValueError("target_mean: TF must be > 0")
    return np.asarray(pub, float) + (np.asarray(NR, float) * c + np.asarray(INV, float) * e) / TF


def lmo_targets(cells: pd.DataFrame) -> pd.DataFrame:
    """Leave-month-out targets for ONE airport's 12 cells: month, T, err = T - mean_label, inv_share, c, e."""
    rows = []
    for m in cells.month.to_numpy():
        tr, te = cells[cells.month != m], cells[cells.month == m].iloc[0]
        c, e = fit_excess(tr)
        T = float(target_mean(te.pub_mean_s, te.NR, te.INV, te.TF, c, e))
        rows.append({"month": int(m), "T": T, "err": T - float(te.mean_label), "inv_share": float(te.INV / te.TF),
                     "c": c, "e": e})
    return pd.DataFrame(rows)


def regime_sigma(lmo: pd.DataFrame, inv_share: float, thr: float = REGIME_THR) -> float:
    """RMS of the LMO errors over the months in the same regime as `inv_share` (>= thr or < thr). One month in the
    regime -> |its error|; none -> inf (no evidence -> no shift)."""
    hi = lmo.inv_share.to_numpy() >= thr
    sel = hi if inv_share >= thr else ~hi
    err = lmo.err.to_numpy()[sel]
    if len(err) == 0:
        return np.inf
    return float(np.sqrt(np.mean(err ** 2)))


def cell_sigma(lmo: pd.DataFrame, inv_share: float, thr: float = REGIME_THR) -> float:
    """regime_sigma, inflated by inv_share / (largest 2025 share in that regime) when the cell extrapolates."""
    s = regime_sigma(lmo, inv_share, thr)
    if not np.isfinite(s):
        return s
    hi = lmo.inv_share.to_numpy() >= thr
    shares = lmo.inv_share.to_numpy()[hi if inv_share >= thr else ~hi]
    mx = float(shares.max())
    if mx > 0 and inv_share > mx:
        s *= inv_share / mx
    return float(s)


def shrink(gap: float, sigma: float) -> float:
    """lambda = max(0, 1 - sigma^2/gap^2); 0 for a zero gap or infinite sigma."""
    if not np.isfinite(sigma) or gap == 0:
        return 0.0
    return float(max(0.0, 1.0 - sigma ** 2 / gap ** 2))


def shift(pred, key, target: dict, lam: dict, mode: str = "add"):
    """Move each group's mean toward its target by lambda*(T - mean). mode 'add' (uniform shift) or 'mul' (scale).
    Groups absent from `target` are unchanged. Returns (new array, {group: {n, mean_before, gap, lam, delta}})."""
    if mode not in ("add", "mul"):
        raise ValueError(f"mode must be 'add' or 'mul', got {mode!r}")
    pred = np.asarray(pred, float)
    key = np.asarray(key)
    out = pred.copy()
    rep = {}
    for k, T in target.items():
        sel = key == k
        if not sel.any():
            continue
        pm = float(pred[sel].mean())
        gap = float(T) - pm
        delta = float(lam[k]) * gap
        if mode == "add":
            out[sel] = pred[sel] + delta
        else:
            out[sel] = pred[sel] * (pm + delta) / pm
        rep[k] = {"n": int(sel.sum()), "mean_before": pm, "gap": gap, "lam": float(lam[k]), "delta": delta}
    return out, rep


def gain_ci(y, before, after, day, month, denom: float, n_boot: int = 2000, seed: int = 0):
    """G = (SSE_before - SSE_after)/denom with a 95 % day-block bootstrap CI (days resampled within month)."""
    y, before, after = (np.asarray(v, float) for v in (y, before, after))
    g_row = (y - before) ** 2 - (y - after) ** 2
    df = pd.DataFrame({"g": g_row, "day": np.asarray(day), "month": np.asarray(month)})
    per_day = df.groupby(["month", "day"]).g.sum()
    G = float(g_row.sum() / denom)
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    for mo, s in per_day.groupby(level=0):
        v = s.to_numpy()
        boots += v[rng.integers(0, len(v), (n_boot, len(v)))].sum(axis=1)
    boots /= denom
    return G, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
