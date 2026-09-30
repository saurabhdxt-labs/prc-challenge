"""RDC — does Rome's delay context separate schedule copies from ordinary delayed departures?
(plans/PREREG_rome_delay_context_2026_09_13.md)

    /opt/homebrew/bin/python3.11 -B scripts/analysis/rome_delay_context.py --out reports/rome_delay_context_2026_09_13.json

Composes scripts/rome_fill.py (fit_predict_fill, mixture, stored_nf, load_frames, rome_rows) — edits nothing. Context
features are built from OTHER Rome departures' take-off and schedule times only.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_fill as RF  # noqa: E402
import rome_local_day as RLD  # noqa: E402

RAW = ROOT / "data" / "raw"
STORED_R = ROOT / "data" / "cache_stand" / "rome_dateslip_preds.parquet"
CTX = ["ctx_n", "ctx_med_ms", "ctx_late_share"]
WINDOW_S = 2 * 3600.0
LATE_S = 3600.0
N_SCORED = 344_841
N26_G1, N26_LIRF_UNM = 288, 383
C1_BAR, C5_BAR, C6_BAR, MIN_MONTHS = 300.0, 50.0, -50.0, 8
GUARD_P_TOL = 1e-6
SEEDS = (0, 1, 2)


def context_features(ids, t_s, ms, window: float = WINDOW_S, late: float = LATE_S) -> pd.DataFrame:
    """For each departure: count, median (MVT − SCHED) and share ≥ `late` of the OTHER departures whose take-off lies
    within ±window (inclusive). ids/t_s/ms are aligned arrays; t_s in seconds."""
    ids, t_s, ms = np.asarray(ids), np.asarray(t_s, dtype="float64"), np.asarray(ms, dtype="float64")
    order = np.argsort(t_s, kind="mergesort")
    ts, mss = t_s[order], ms[order]
    lo = np.searchsorted(ts, ts - window, side="left")
    hi = np.searchsorted(ts, ts + window, side="right")
    n = np.empty(len(ts)); med = np.full(len(ts), np.nan); share = np.full(len(ts), np.nan)
    for i in range(len(ts)):
        w = np.concatenate([mss[lo[i]:i], mss[i + 1:hi[i]]])
        n[i] = len(w)
        if len(w):
            med[i] = np.median(w)
            share[i] = float((w >= late).mean())
    out = pd.DataFrame({"MVT_ID_mvt": ids[order], "ctx_n": n, "ctx_med_ms": med, "ctx_late_share": share})
    return out


def departures_frame(frames) -> pd.DataFrame:
    """All LIRF DEP rows (any match state) with take-off seconds and MVT − SCHED; duplicates of MVT_ID refused."""
    d = pd.concat(frames, ignore_index=True)
    d = d[(d.PHASE_mvt == "DEP") & (d.ADEP_mvt == "LIRF")]
    d = d.dropna(subset=["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]).drop_duplicates("MVT_ID_mvt")
    t = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True)
    # AMENDMENT 1: resolution-independent seconds (the raw files are datetime64[us]; astype(int64)//1e9 gave kiloseconds)
    t_s = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype="float64")
    return pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt.to_numpy(), "t_s": t_s,
                         "ms": (t - pd.to_datetime(d.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds().to_numpy()})


CTX_N_MEDIAN_RANGE = (0.0, 500.0)


def check_context_scale(ctx: pd.DataFrame, lo_hi=CTX_N_MEDIAN_RANGE) -> float:
    """AMENDMENT 1 guard: a ±2 h window at one airport holds tens of departures, not thousands."""
    med = float(np.median(ctx.ctx_n.to_numpy(dtype="float64")))
    if not (lo_hi[0] < med < lo_hi[1]):
        raise AssertionError(f"median ctx_n {med:.0f} outside {lo_hi}: the time unit is wrong (BC-2)")
    return med


def permute_within_month(frame: pd.DataFrame, cols, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    f = frame.copy()
    mo = f.month.to_numpy()
    for m in np.unique(mo):
        idx = np.flatnonzero(mo == m)
        perm = rng.permutation(idx)
        for c in cols:
            f.iloc[idx, f.columns.get_loc(c)] = f[c].to_numpy()[perm]
    return f


def priced(net_sse: float, n_rows: int, n_2026: int) -> float:
    if n_rows <= 0:
        raise ValueError("empty population")
    return float(net_sse) / n_rows * n_2026 / N_SCORED


def month_block_lower(g, month, n_rows: int, n_2026: int, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    g, month = np.asarray(g, float), np.asarray(month)
    ms = np.unique(month)
    s = np.array([g[month == m].sum() for m in ms])
    rng = np.random.default_rng(seed)
    draws = [priced(s[rng.integers(0, len(ms), len(ms))].sum(), n_rows, n_2026) for _ in range(n_boot)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def minus_top_positive(g, k: int) -> float:
    g = np.sort(np.asarray(g, float))[::-1]
    return float(g.sum() - g[:k][g[:k] > 0].sum())


def verdict(c: dict) -> str:
    if not c["C5"]:
        return "HARNESS INVALID"
    if all(c.values()):
        return "SHORTLIST"
    if not c["C1"] and not c["C3"]:
        return "NOT WORKING"
    return "PARTIAL"


def fit_folds(unm, lirf, stored_r, ctx25, fit=RF.fit_predict_fill, seeds=SEEDS, log=print) -> tuple:
    """(stored LOMO rows, {"C": p, "D": p, "P": p}) — the RDC fold loop, returned per row (RDX refactor; unchanged maths)."""
    st = stored_r[stored_r.fold == "lomo"].reset_index(drop=True)
    u = unm.merge(ctx25, on="MVT_ID_mvt", how="left")
    l = lirf.merge(ctx25, on="MVT_ID_mvt", how="left")
    if u[u.ADEP_mvt == "LIRF"][CTX[0]].isna().any() or l[CTX[0]].isna().any():
        raise AssertionError("a LIRF departure has no context row")
    u_p, l_p = permute_within_month(u[u.ADEP_mvt == "LIRF"], CTX), permute_within_month(l, CTX)
    u_p = pd.concat([u[u.ADEP_mvt != "LIRF"], u_p], ignore_index=True)
    ul = u[u.ADEP_mvt == "LIRF"].set_index("MVT_ID_mvt", drop=False)
    ulp = u_p[u_p.ADEP_mvt == "LIRF"].set_index("MVT_ID_mvt", drop=False)
    P = {k: np.full(len(st), np.nan) for k in ("C", "D", "P")}
    months = sorted(st.month.unique())
    all_months = sorted(set(unm.month.unique()) | set(lirf.month.unique()))
    for mo in months:
        rows = np.flatnonzero(st.month.to_numpy() == mo)
        ids = st.MVT_ID_mvt.to_numpy()[rows]
        train = [m for m in all_months if m != mo]
        tr, trp = RF.rome_rows(u, l, train), RF.rome_rows(u_p, l_p, train)
        te, tep = ul.loc[ids], ulp.loc[ids]
        P["C"][rows] = np.mean([fit(tr, te, seed=s) for s in seeds], axis=0)
        P["D"][rows] = np.mean([fit(tr, te, seed=s, extra_num=CTX) for s in seeds], axis=0)
        P["P"][rows] = fit(trp, tep, seed=seeds[0], extra_num=CTX)
        log(f"  month {mo}: {len(rows)} rows")
    return st, P


def run(unm, lirf, stored_r, ctx25, fit=RF.fit_predict_fill, seeds=SEEDS, log=print) -> dict:
    st, P = fit_folds(unm, lirf, stored_r, ctx25, fit=fit, seeds=seeds, log=log)
    months = sorted(st.month.unique())
    gap = float(np.nanmax(np.abs(P["C"] - st.p_R.to_numpy(float))))
    y, sp, nf = st.y.to_numpy(float), st.sp.to_numpy(float), st.nf.to_numpy(float)
    pred = {k: RF.mixture(v, sp, nf) for k, v in P.items()}
    control = pred["C"]
    ld = RLD.local_dayoff(st.SCHED_TIME_UTC_mvt, st.MVT_TIME_UTC_mvt) if "SCHED_TIME_UTC_mvt" in st else st.ld.to_numpy()
    g1 = (sp >= 3600) & (sp < 24000) & (np.asarray(ld) == 0)
    other = ~g1
    g = (y - control) ** 2 - (y - pred["D"]) ** 2
    gp = (y - control) ** 2 - (y - pred["P"]) ** 2
    month = st.month.to_numpy()
    n1 = int(g1.sum())
    net = priced(g[g1].sum(), n1, N26_G1)
    lo, hi = month_block_lower(g[g1], month[g1], n1, N26_G1)
    by_month = {int(m): float(g[g1 & (month == m)].sum()) for m in months}
    fill = np.abs(y - sp) <= RF.FILL_TOL_S
    from sklearn.metrics import roc_auc_score
    auc = {k: float(roc_auc_score(fill[g1], P[k][g1])) for k in ("C", "D")}
    c = {"C1": net >= C1_BAR, "C2": lo > 0.0, "C3": sum(v > 0 for v in by_month.values()) >= MIN_MONTHS,
         "C4": minus_top_positive(g[g1], 1) > 0 and minus_top_positive(g[g1], 5) > 0,
         "C5": priced(gp[g1].sum(), n1, N26_G1) <= C5_BAR,
         "C6": priced(g[other].sum(), int(other.sum()), N26_LIRF_UNM - N26_G1) >= C6_BAR}
    return {"tag": "RDC", "prereg": "plans/PREREG_rome_delay_context_2026_09_13.md",
            "guard_control_vs_stored_pR_maxabs": gap, "guard_within_tol": gap <= GUARD_P_TOL,
            "n_lomo": int(len(st)), "n_g1": n1, "priced_net_g1": net, "month_block95": [lo, hi],
            "by_month_sse": by_month, "months_positive": int(sum(v > 0 for v in by_month.values())),
            "minus_top1_sse": minus_top_positive(g[g1], 1), "minus_top5_sse": minus_top_positive(g[g1], 5),
            "permuted_priced_net_g1": priced(gp[g1].sum(), n1, N26_G1),
            "other_priced_net": priced(g[other].sum(), int(other.sum()), N26_LIRF_UNM - N26_G1),
            "auc_g1": auc, "gain_on_fills_sse": float(g[g1 & fill].sum()), "gain_on_ordinary_sse": float(g[g1 & ~fill].sum()),
            "clauses": c, "verdict": verdict(c)}


def load_all(log=print):
    unm, lirf = RF.load_frames(log)
    stored = pd.read_parquet(STORED_R)
    stored = stored.merge(unm[["MVT_ID_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"]], on="MVT_ID_mvt", how="left",
                          validate="many_to_one")
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    files = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
    d25 = departures_frame([pd.read_parquet(f, columns=cols) for f in files])
    ctx25 = context_features(d25.MVT_ID_mvt, d25.t_s, d25.ms)
    check_context_scale(ctx25)
    d26 = departures_frame([pd.read_parquet(RAW / "ranking.parquet", columns=cols + ["AOBT_3_flt"])])
    ctx26 = context_features(d26.MVT_ID_mvt, d26.t_s, d26.ms)
    return unm, lirf, stored, ctx25, ctx26


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f"refusing to overwrite {a.out}")
    t0 = time.time()
    unm, lirf, stored, ctx25, ctx26 = load_all()
    res = run(unm, lirf, stored, ctx25)
    r26 = pd.read_parquet(RAW / "ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "AOBT_3_flt",
                                                              "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    r26 = r26[(r26.PHASE_mvt == "DEP") & (r26.ADEP_mvt == "LIRF") & r26.AOBT_3_flt.isna()]
    sp26 = (pd.to_datetime(r26.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(r26.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    g26 = r26[(sp26 >= 3600) & (sp26 < 24000) & (RLD.local_dayoff(r26.SCHED_TIME_UTC_mvt, r26.MVT_TIME_UTC_mvt) == 0)]
    q = lambda s: [float(x) for x in np.nanpercentile(s, [10, 50, 90])]
    c26 = ctx26[ctx26.MVT_ID_mvt.isin(g26.MVT_ID_mvt)]
    res["context_2026_g1_p10_p50_p90"] = {c: q(c26[c]) for c in CTX}
    res["n_2026_g1_with_context"] = int(len(c26))
    res["wall_s"] = time.time() - t0
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
