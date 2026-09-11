"""NMD — the four-airport lead: does arm F mishandle rows whose AOBT_3 is the NM planned value?

Registered in reports/NM_PARAMETER_DIAGNOSTIC_2026_09_11.md (Part 1, written before any residual was read).
Light: numpy / pandas on arm F's stored fold record, no fitting of a model.

    python3 -B scripts/nm_param_diag.py          -> reports/nm_param_diag.json

The NM block is serve-time: it is built from proxies only (never labels) on the ten non-holdout months.
The residual corrections are cross-fitted across the two holdout months, so no row is ever corrected
by a value its own month contributed to.
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
CACHE = ROOT / "data" / "cache_stand"
RECORD = CACHE / "fold_preds_queue_order.parquet"
OUT = ROOT / "reports" / "nm_param_diag.json"
FOUR = ("LEBL", "LEMD", "EDDM", "LTFM")
HOLDOUT = (1, 7)
W_MATCHED = 0.9846596
ARM_F_RMSE = 222.5632
K2_SHRINK = 50.0
UNSEEN = "unseen"
# K1 dev classes (integer minutes from the pair's modal minute), registered in Part 1
K1_EDGES = ((-10**9, -6, "<=-6"), (-5, -2, "-5..-2"), (-1, -1, "-1"), (0, 0, "0"), (1, 1, "+1"),
            (2, 5, "+2..+5"), (6, 15, "+6..+15"), (16, 10**9, ">+15"))


def pair_modes(pair: np.ndarray, pm: np.ndarray) -> pd.DataFrame:
    """Per pair: modal minute m_star (ties -> smallest), support n_pair, mode share s_pair."""
    if len(pair) != len(pm):
        raise ValueError(f"pair ({len(pair)}) and pm ({len(pm)}) differ in length")
    if len(pair) == 0:
        raise ValueError("no rows to build the NM table from")
    if np.isnan(pm.astype("float64")).any():
        raise ValueError("pm holds NaN: the proxy must be present on every matched row")
    c = pd.DataFrame({"pair": pair, "pm": pm}).groupby(["pair", "pm"]).size().rename("n").reset_index()
    c = c.sort_values(["pair", "n", "pm"], ascending=[True, False, True])
    top = c.drop_duplicates("pair", keep="first").set_index("pair")
    tot = c.groupby("pair")["n"].sum()
    return pd.DataFrame({"m_star": top["pm"].astype("int64"), "n_pair": tot, "s_pair": top["n"] / tot})


def attach_block(pair: np.ndarray, pm: np.ndarray, table: pd.DataFrame) -> pd.DataFrame:
    """dev (minutes, NaN when the pair is unseen), at_mode, seen, s_pair for scored rows."""
    t = table.reindex(pair)
    seen = t["m_star"].notna().to_numpy()
    dev = np.where(seen, pm - t["m_star"].to_numpy(), np.nan)
    return pd.DataFrame({"dev": dev, "at_mode": seen & (dev == 0), "seen": seen, "s_pair": t["s_pair"].to_numpy()})


def dev_class(dev: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """K1 classes; every row lands in exactly one."""
    out = np.full(len(dev), UNSEEN, dtype=object)
    for lo, hi, name in K1_EDGES:
        out[seen & (dev >= lo) & (dev <= hi)] = name
    if (seen & (out == UNSEEN)).any():
        raise AssertionError("a seen row fell outside every K1 class")
    return out


def dev3(at_mode: np.ndarray, seen: np.ndarray) -> np.ndarray:
    return np.where(~seen, UNSEEN, np.where(at_mode, "at", "off")).astype(object)


def _other(month: np.ndarray) -> list:
    ms = sorted(set(month.tolist()))
    if len(ms) != 2:
        raise ValueError(f"cross-fitting needs exactly two months, got {ms}")
    return ms


def crossfit_k1(r: np.ndarray, ap: np.ndarray, cls: np.ndarray, month: np.ndarray) -> np.ndarray:
    """Mean residual of (airport, class) fitted on the OTHER month; 0 where the other month lacks the cell."""
    a, b = _other(month)
    c = np.zeros(len(r))
    key = pd.Series(ap.astype(str)) + "#" + pd.Series(cls.astype(str))
    for fit_m, app_m in ((a, b), (b, a)):
        f, s = month == fit_m, month == app_m
        means = pd.Series(r[f]).groupby(key[f].to_numpy()).mean()
        c[s] = key[s].map(means).fillna(0.0).to_numpy()
    return c


def crossfit_k2(r: np.ndarray, pair: np.ndarray, ap: np.ndarray, cls3: np.ndarray, month: np.ndarray,
                k: float = K2_SHRINK) -> np.ndarray:
    """Mean residual of (pair, class) shrunk toward (airport, class) with weight n/(n+k), fitted on the OTHER month."""
    if k <= 0:
        raise ValueError("k must be positive")
    a, b = _other(month)
    c = np.zeros(len(r))
    kp = pd.Series(pair.astype(str)) + "#" + pd.Series(cls3.astype(str))
    ka = pd.Series(ap.astype(str)) + "#" + pd.Series(cls3.astype(str))
    for fit_m, app_m in ((a, b), (b, a)):
        f, s = month == fit_m, month == app_m
        prior = pd.Series(r[f]).groupby(ka[f].to_numpy()).mean()
        g = pd.Series(r[f]).groupby(kp[f].to_numpy()).agg(["sum", "count"])
        pr_s = ka[s].map(prior).fillna(0.0).to_numpy()
        sm = kp[s].map(g["sum"]).fillna(0.0).to_numpy()
        n = kp[s].map(g["count"]).fillna(0.0).to_numpy()
        c[s] = (sm + k * pr_s) / (n + k)
    return c


def gain_rows(y: np.ndarray, yhat: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Per-row SE(F) - SE(F + c), with the shipped floor applied to the corrected arm."""
    return (y - yhat) ** 2 - (y - np.maximum(yhat + c, 1.0)) ** 2


def to_board(sigma_g: float, n_fold: int) -> float:
    return float(sigma_g) / n_fold * W_MATCHED


def dayblock_sum(g: np.ndarray, day: np.ndarray, month: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple:
    """95% date-block interval of Σg: dates resampled with replacement within each month separately."""
    codes, uniq = pd.factorize(pd.Series(day))
    if (codes < 0).any():
        raise ValueError("a row has no date")
    S = np.bincount(codes, weights=g)
    mon_of = pd.Series(month).groupby(codes).agg(["min", "max"])
    if (mon_of["min"] != mon_of["max"]).any():
        raise ValueError("a date spans two months")
    mon_of = mon_of["min"].to_numpy()
    groups = [np.flatnonzero(mon_of == m) for m in sorted(set(mon_of.tolist()))]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        draws[i] = sum(S[rng.choice(gr, size=len(gr), replace=True)].sum() for gr in groups)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(g.sum()), float(lo), float(hi), len(uniq)


def within_pair_slope(v: np.ndarray, dev: np.ndarray, pair: np.ndarray) -> float:
    """OLS slope of v on dev after demeaning both within pair (seconds of v per minute of dev)."""
    df = pd.DataFrame({"v": v, "d": dev, "p": pair})
    dm = df.groupby("p")[["v", "d"]].transform("mean")
    vv, dd = (df["v"] - dm["v"]).to_numpy(), (df["d"] - dm["d"]).to_numpy()
    den = float((dd * dd).sum())
    return float("nan") if den == 0 else float((vv * dd).sum() / den)


def between_ss(r: np.ndarray, cell: np.ndarray, base: np.ndarray) -> float:
    """In-sample Σ_cells n_c (mean_c − mean_base)² with every cell nested in one base group: the SE an oracle
    additive cell correction removes beyond the base group's own mean (post-hoc ceiling, never decisional)."""
    df = pd.DataFrame({"r": r, "c": cell, "b": base})
    if (df.groupby("c")["b"].nunique() > 1).any():
        raise ValueError("a cell spans two base groups")
    mc = df.groupby("c")["r"].transform("mean").to_numpy()
    mb = df.groupby("b")["r"].transform("mean").to_numpy()
    return float(((mc - mb) ** 2).sum())


def permute_within(x: np.ndarray, groups: pd.Series, seed: int) -> np.ndarray:
    """x shuffled within each group (the negative control's instrument)."""
    rng = np.random.default_rng(seed)
    out = np.array(x, copy=True)
    for idx in groups.groupby(groups.to_numpy()).indices.values():
        out[idx] = out[rng.permutation(idx)]
    return out


# ------------------------------------------------------------------ real data (not unit-tested; guarded)

def _holdout_dates() -> dict:
    out = {}
    for m, f in ((1, "training_2025-01-01_2025-02-01.parquet"), (7, "training_2025-07-01_2025-08-01.parquet")):
        y_stand = pq.read_table(CACHE / f, columns=["y"]).column("y").to_numpy()
        ids = pq.read_table(ROOT / "data" / "cache_order" / f, columns=["MVT_ID_mvt"]).column("MVT_ID_mvt").to_numpy()
        if len(ids) != len(y_stand):
            raise RuntimeError(f"month {m}: order cache {len(ids)} rows vs stand cache {len(y_stand)}")
        raw = pq.read_table(ROOT / "data" / "raw" / f, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]).to_pandas()
        day = raw.set_index("MVT_ID_mvt").MVT_TIME_UTC_mvt.reindex(ids).dt.strftime("%Y-%m-%d").to_numpy()
        if pd.isna(day).any():
            raise RuntimeError(f"month {m}: some ids have no take-off time")
        out[m] = (y_stand.astype("float64"), day)
    return out


def holdout_days(row: np.ndarray, month: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Take-off date (YYYY-MM-DD) of fold-A holdout rows given their `row` (position in the twelve concatenated stand-cache
    months) — verified: the stand cache's y at each row must equal the record's y, or it raises."""
    months = sorted(set(np.asarray(month).tolist()))
    if not set(months) <= set(HOLDOUT):
        raise ValueError(f"holdout_days serves months {HOLDOUT}, got {months}")
    counts = [pq.ParquetFile(p).metadata.num_rows for p in sorted(glob.glob(str(CACHE / "training_2025-*.parquet")))]
    if len(counts) != 12:
        raise RuntimeError(f"expected 12 stand-cache months, found {len(counts)}")
    start = np.concatenate([[0], np.cumsum(counts)])
    hd = _holdout_dates()
    out = np.empty(len(row), dtype=object)
    for m in months:
        s = np.asarray(month) == m
        local = np.asarray(row)[s] - start[m - 1]
        y_stand, d = hd[m]
        if (local < 0).any() or (local >= len(y_stand)).any() or not np.array_equal(y_stand[local], np.asarray(y)[s]):
            raise RuntimeError(f"month {m}: rows do not align with the stand cache (dates would be wrong)")
        out[s] = d[local]
    return out


def load() -> pd.DataFrame:
    rec = pd.read_parquet(RECORD)
    yhat = np.maximum(rec.proxy.to_numpy() - rec.treatment.to_numpy(), 1.0)
    rmse = float(np.sqrt(((rec.y.to_numpy() - yhat) ** 2).mean()))
    if abs(rmse - ARM_F_RMSE) > 5e-4:
        raise RuntimeError(f"arm F does not reproduce: {rmse:.4f} vs {ARM_F_RMSE} (BC-2: wrong convention?)")
    files = sorted(glob.glob(str(CACHE / "training_2025-*.parquet")))
    if len(files) != 12:
        raise RuntimeError(f"expected 12 stand-cache months, found {len(files)}")
    cache = pd.concat([pd.read_parquet(f, columns=["y", "proxy", "delta", "ap", "month", "ars"]) for f in files],
                      ignore_index=True)
    j = cache.iloc[rec.row.to_numpy()].reset_index(drop=True)
    for col in ("y", "proxy", "delta"):
        if not np.array_equal(j[col].to_numpy(), rec[col].to_numpy()):
            raise RuntimeError(f"positional join broken on {col}")
    if not (np.array_equal(j.ap.to_numpy(), rec.ap.to_numpy()) and np.array_equal(j.month.to_numpy(), rec.month.to_numpy())):
        raise RuntimeError("positional join broken on ap / month")
    pref = j.ars.astype(str).str.split("|").str[0].to_numpy()
    if not np.array_equal(pref, rec.ap.to_numpy()):
        raise RuntimeError("ars is not airport-scoped on every holdout row")
    train = cache[~cache.month.isin(HOLDOUT)]
    table = pair_modes(train.ars.astype(str).to_numpy(), np.floor(train.proxy.to_numpy() / 60).astype("int64"))
    df = rec[["row", "month", "ap", "y", "proxy"]].copy()
    df["pair"] = j.ars.astype(str).to_numpy()
    df["yhat"] = yhat
    df["r"] = df.y - df.yhat
    df["pm"] = np.floor(df.proxy / 60).astype("int64")
    blk = attach_block(df.pair.to_numpy(), df.pm.to_numpy(), table)
    for c in blk.columns:
        df[c] = blk[c].to_numpy()
    df["day"] = holdout_days(df.row.to_numpy(), df.month.to_numpy(), df.y.to_numpy())
    return df, rmse, len(table)


def d1(df: pd.DataFrame, n_fold: int) -> dict:
    out = {}
    for a, g in df.groupby("ap"):
        at, off = g[g.at_mode], g[g.seen & ~g.at_mode]
        near = g[g.seen & (g.dev.abs() <= 15)]
        out[a] = {
            "n": int(len(g)), "at_mode_share": float(g.at_mode.mean()), "unseen_share": float((~g.seen).mean()),
            "board_mse_at": to_board(float((at.r ** 2).sum()), n_fold),
            "board_mse_off": to_board(float((off.r ** 2).sum()), n_fold),
            "board_mse_unseen": to_board(float((g[~g.seen].r ** 2).sum()), n_fold),
            "rmse_at": float(np.sqrt((at.r ** 2).mean())) if len(at) else None,
            "rmse_off": float(np.sqrt((off.r ** 2).mean())) if len(off) else None,
            "mean_r_at": {int(m): float(x.r.mean()) for m, x in at.groupby("month")},
            "mean_r_off": {int(m): float(x.r.mean()) for m, x in off.groupby("month")},
            "slope_y_on_dev": within_pair_slope(near.y.to_numpy(), near.dev.to_numpy(), near.pair.to_numpy()),
            "slope_yhat_on_dev": within_pair_slope(near.yhat.to_numpy(), near.dev.to_numpy(), near.pair.to_numpy()),
            "n_slope_rows": int(len(near)), "n_slope_offmode": int((near.dev != 0).sum()),
        }
    return out


def d2(df: pd.DataFrame, n_fold: int, rows: np.ndarray, label: str) -> dict:
    s = df[rows].reset_index(drop=True)
    y, yh, r, mo = s.y.to_numpy(), s.yhat.to_numpy(), s.r.to_numpy(), s.month.to_numpy()
    ap, pair = s.ap.to_numpy(), s.pair.to_numpy()
    cls, c3 = dev_class(s.dev.to_numpy(), s.seen.to_numpy()), dev3(s.at_mode.to_numpy(), s.seen.to_numpy())
    grp = pd.Series(ap.astype(str)) + "#" + pd.Series(mo.astype(str))
    arms = {
        "K1": crossfit_k1(r, ap, cls, mo),
        "K2": crossfit_k2(r, pair, ap, c3, mo),
        "NC_K1": crossfit_k1(r, ap, permute_within(cls, grp, 11), mo),
        "NC_K2": crossfit_k2(r, permute_within(pair, grp, 12), ap, c3, mo),
    }
    out = {"rows": int(len(s)), "label": label}
    for name, c in arms.items():
        g = gain_rows(y, yh, c)
        tot, lo, hi, nd = dayblock_sum(g, s.day.to_numpy(), mo)
        out[name] = {"sigma_g": tot, "ci95": [lo, hi], "n_dates": nd, "board_mse": to_board(tot, n_fold),
                     "board_ci95": [to_board(lo, n_fold), to_board(hi, n_fold)],
                     "per_row_gain_s2": tot / len(s),
                     "by_airport_board": {a: to_board(float(g[ap == a].sum()), n_fold) for a in sorted(set(ap))},
                     "by_class_board": {k: to_board(float(g[c3 == k].sum()), n_fold) for k in ("at", "off", UNSEEN)},
                     "by_month_board": {int(m): to_board(float(g[mo == m].sum()), n_fold) for m in HOLDOUT}}
    return out


def main() -> int:
    df, rmse, n_pairs = load()
    n_fold = len(df)
    four = df.ap.isin(FOUR).to_numpy()
    rec = {
        "tag": "NMD", "prereg": "reports/NM_PARAMETER_DIAGNOSTIC_2026_09_11.md#part-1",
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "arm_f_rmse": rmse, "n_fold": n_fold, "n_pairs_table": n_pairs,
        "d1": d1(df, n_fold),
        "d2_four": d2(df, n_fold, four, "LEBL/LEMD/EDDM/LTFM"),
        "d2_control": d2(df, n_fold, ~four, "six control airports"),
    }
    OUT.write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: rec[k] for k in ("arm_f_rmse", "n_fold", "n_pairs_table")}))
    for k in ("d2_four", "d2_control"):
        for arm in ("K1", "K2", "NC_K1", "NC_K2"):
            x = rec[k][arm]
            print(f"{k:10s} {arm:6s} board {x['board_mse']:+9.1f}  95% [{x['board_ci95'][0]:+9.1f}, "
                  f"{x['board_ci95'][1]:+9.1f}]  per-row {x['per_row_gain_s2']:+8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
