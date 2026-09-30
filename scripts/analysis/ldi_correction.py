"""LDI — the LTFM January-2026 de-icing correction (registered: plans/PREREG_ldi_ltfm_deicing_2026_09_27.md).

Pure functions (tested in tests/test_ldi_correction.py) plus a real-data driver (`evaluate`, `build_values`).

Definitions (LTFM matched departures; sp = MVT - SCHED in seconds; local day = UTC + 3 h):
  band          (-inf, 30], (30, 60], (60, inf) minutes of sp
  severe day    share(sp > 1800 s) >= 0.5 AND min w_temp_c <= 1.0 over the day's LTFM matched departures
  target row    w_freezing == 1 on a severe day                        (the correction touches only these)
  cell gap      Delta_b = mean(y - p | target, band b) - mean(y - p | w_freezing == 0, band b)   (difference form)
  LOSDO         per severe day d: Delta fitted without day d, shrink * Delta applied to day d's target rows
  transfer adj  adj_b = mean(p_seen - p_unseen | 2026 target, b) - mean(p_seen - p_unseen | 2026 non-freezing, b)
  applied       Delta_apply_b = max(0, shrink * (Delta_b - adj_b))
Units: bmse = SSE / 344,841 (the board's row count); gain > 0 = the correction is better.
No 2026 label is read anywhere in this module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BOARD_N = 344_841.0
BANDS = ("<30", "30-60", ">60")
EDGES_S = (1800, 3600)
SEVERE_SHARE, SEVERE_TMIN_C, DELAY_S = 0.5, 1.0, 1800
LOCAL_OFFSET_S = 3 * 3600
BOOT_SEED, N_BOOT = 20260927, 10_000


def band(sp) -> np.ndarray:
    sp = np.asarray(sp, dtype="float64")
    if not np.isfinite(sp).all():
        raise ValueError("non-finite sp")
    return np.where(sp <= EDGES_S[0], BANDS[0], np.where(sp <= EDGES_S[1], BANDS[1], BANDS[2])).astype(object)


def local_day(t_utc_s) -> np.ndarray:
    return pd.to_datetime(np.asarray(t_utc_s, dtype="int64") + LOCAL_OFFSET_S, unit="s").date


def severe_days(day, sp, temp) -> pd.Series:
    temp = pd.Series(np.asarray(temp, dtype="float64"))
    if temp.isna().all():
        raise ValueError("no temperature at all: severity undefined")
    f = pd.DataFrame({"day": np.asarray(day), "late": np.asarray(sp, dtype="float64") > DELAY_S, "t": temp.to_numpy()})
    g = f.groupby("day").agg(share=("late", "mean"), tmin=("t", "min"))
    return (g.share >= SEVERE_SHARE) & (g.tmin <= SEVERE_TMIN_C)


def _band_means(v, band_, mask) -> dict:
    s = pd.Series(np.asarray(v, float))[np.asarray(mask, bool)].groupby(np.asarray(band_)[np.asarray(mask, bool)]).mean()
    return {b: float(s.get(b, np.nan)) for b in BANDS}


def cell_gaps(y, p, band_, target, nonfrz) -> dict:
    r = np.asarray(y, float) - np.asarray(p, float)
    t, n = _band_means(r, band_, target), _band_means(r, band_, nonfrz)
    return {b: (t[b] - n[b]) if np.isfinite(t[b]) and np.isfinite(n[b]) else 0.0 for b in BANDS}


def _gain(y, p, corr) -> float:
    r = np.asarray(y, float) - np.asarray(p, float)
    return float(((r ** 2).sum() - ((r - corr) ** 2).sum()) / BOARD_N)


def losdo_gains(f: pd.DataFrame, shrink: float) -> dict:
    """f: day, band, target, nonfrz, y, p. Per day holding target rows: out-of-day gain in bmse."""
    out = {}
    for d in sorted(f.loc[f.target.astype(bool), "day"].unique(), key=str):
        rest, cur = f[f.day != d], f[f.day == d]
        g = cell_gaps(rest.y, rest.p, rest.band, rest.target.astype(bool), rest.nonfrz.astype(bool))
        corr = np.where(cur.target.astype(bool), cur.band.map(g).astype(float) * shrink, 0.0)
        out[d] = _gain(cur.y, cur.p, corr)
    return out


def day_bootstrap_ci(vals, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple:
    v = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    tot = v[rng.integers(0, len(v), size=(n_boot, len(v)))].sum(axis=1)
    return float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5))


def transfer_adjustment(p_seen, p_unseen, band_, target, nonfrz) -> dict:
    dlt = np.asarray(p_seen, float) - np.asarray(p_unseen, float)
    t, n = _band_means(dlt, band_, target), _band_means(dlt, band_, nonfrz)
    return {b: (t[b] - n[b]) if np.isfinite(t[b]) and np.isfinite(n[b]) else 0.0 for b in BANDS}


def applied_deltas(gaps: dict, adj: dict, shrink: float) -> dict:
    return {b: max(0.0, shrink * (gaps[b] - adj[b])) for b in BANDS}


def apply_to_values(values: pd.DataFrame, target: pd.DataFrame, deltas: dict) -> pd.DataFrame:
    """values: MVT_ID_mvt, TAXITIME_SEC_mvt (the base submission); target: MVT_ID_mvt, band. Returns a copy with
    deltas[band] added on the target ids only."""
    if target.MVT_ID_mvt.duplicated().any():
        raise ValueError("duplicate target ids")
    pos = pd.Index(values.MVT_ID_mvt).get_indexer(target.MVT_ID_mvt)
    if (pos < 0).any():
        raise ValueError(f"{int((pos < 0).sum())} target ids not in the values")
    out = values.copy()
    col = out.columns.get_loc("TAXITIME_SEC_mvt")
    add = target.band.map(deltas).astype(float).to_numpy()
    vals = out.TAXITIME_SEC_mvt.to_numpy(copy=True)
    vals[pos] = vals[pos] + add
    out.iloc[:, col] = vals.astype(values.TAXITIME_SEC_mvt.dtype)
    return out


def gain_range(n: dict, delta: dict, g_max: dict) -> tuple:
    lo = sum(-n[k] * delta[k] ** 2 for k in n) / BOARD_N
    hi = sum(n[k] * (2 * g_max[k] * delta[k] - delta[k] ** 2) for k in n) / BOARD_N
    return float(lo), float(hi)


def rmse_after(rmse: float, gain_bmse: float) -> float:
    return float(np.sqrt(rmse ** 2 - gain_bmse))


# ---------------------------------------------------------------- frame assembly (real files -> the tested frames)

EPOCH_MIN_S, EPOCH_MAX_S = 1_500_000_000, 2_000_000_000       # 2017 .. 2033: guards a unit mix-up


def to_epoch_s(x) -> np.ndarray:
    s = pd.Series(x)
    if pd.api.types.is_datetime64_any_dtype(s):
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize("UTC")
        out = ((s - pd.Timestamp("1970-01-01T00:00:00Z")) // pd.Timedelta(seconds=1)).to_numpy("int64")
    else:
        out = s.to_numpy("int64")
    if len(out) and (out.min() < EPOCH_MIN_S or out.max() > EPOCH_MAX_S):
        raise ValueError("value outside the 2017-2033 epoch-seconds range: wrong unit")
    return out


def _flags(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    t, s = to_epoch_s(f.MVT_TIME_UTC_mvt), to_epoch_s(f.SCHED_TIME_UTC_mvt)
    f["sp"] = t - s
    f["day"] = local_day(t)
    f["band"] = band(f.sp)
    sev = severe_days(f.day, f.sp, f.w_temp_c)
    f["severe"] = f.day.map(sev).astype(bool)
    f["target"] = (f.w_freezing == 1) & f.severe
    f["nonfrz"] = f.w_freezing == 0
    return f


def frame_2025(pred: pd.DataFrame, dep: pd.DataFrame, wx: pd.DataFrame) -> pd.DataFrame:
    """pred: an r12_validate predictions.parquet (MVT_ID_mvt, ap, month, y, taxi_r12); dep: 2025 LTFM rows with
    PHASE_mvt, ADEP_mvt, MVT/SCHED times, y; wx: the weather cache (MVT_ID_mvt, w_freezing, w_temp_c)."""
    P = pred.loc[pred.ap == "LTFM", ["MVT_ID_mvt", "month", "y", "taxi_r12"]]
    D = dep.loc[(dep.PHASE_mvt == "DEP") & (dep.ADEP_mvt == "LTFM"),
                ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "y"]]
    if D.MVT_ID_mvt.duplicated().any() or P.MVT_ID_mvt.duplicated().any():
        raise AssertionError("duplicate MVT ids")
    f = P.merge(D, on="MVT_ID_mvt", how="inner", suffixes=("", "_dep"))
    if (np.abs(f.y.to_numpy(float) - f.y_dep.to_numpy(float)) > 1e-6).any():
        raise AssertionError("label mismatch between the predictions and the departures file: misaligned join")
    f = f.drop(columns="y_dep").merge(wx[["MVT_ID_mvt", "w_freezing", "w_temp_c"]], on="MVT_ID_mvt", how="left")
    f = _flags(f).rename(columns={"taxi_r12": "p"})
    return f.sort_values("MVT_ID_mvt", kind="stable").reset_index(drop=True)


def frame_2026(rank: pd.DataFrame, wx: pd.DataFrame, seen: pd.DataFrame, unseen: pd.DataFrame,
               base: pd.DataFrame) -> pd.DataFrame:
    """January-2026 LTFM matched departures with p_seen (a storm-seen fit's taxi_r12), p_unseen (the February-out
    fit's) and base (the incumbent submission's value). No 2026 label is read."""
    R = rank.loc[(rank.PHASE_mvt == "DEP") & (rank.ADEP_mvt == "LTFM") & rank.AOBT_3_flt.notna(),
                 ["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]]
    R = R[pd.to_datetime(to_epoch_s(R.MVT_TIME_UTC_mvt), unit="s").month == 1]
    f = R.merge(wx[["MVT_ID_mvt", "w_freezing", "w_temp_c"]], on="MVT_ID_mvt", how="left")
    for name, src, col in (("p_seen", seen, "taxi_r12"), ("p_unseen", unseen, "taxi_r12"),
                           ("base", base, "TAXITIME_SEC_mvt")):
        m = src.set_index("MVT_ID_mvt")[col]
        f[name] = m.reindex(f.MVT_ID_mvt.to_numpy()).to_numpy(float)
        if f[name].isna().any():
            raise AssertionError(f"{int(f[name].isna().sum())} January-2026 LTFM matched rows missing {name}")
    return _flags(f).reset_index(drop=True)
