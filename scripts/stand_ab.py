"""Paired A/B: does the stand-occupancy block remove global MSE over an E1-equivalent model?

Pre-registered in plans/PREREG_taxiout_2026_09_08.md Amendment 5.

    python3.11 scripts/stand_ab.py cache [--smoke]   # build per-month feature caches
    python3.11 scripts/stand_ab.py fit   [--smoke]   # fold A, baseline vs +stand, paired bootstrap

Design. Fold A holds out Jan+Jul 2025; encodings are fitted on the ten training months only.
Target is `delta = BLOCK_TIME - AOBT_3`; the prediction is `y_hat = max(proxy - delta_hat, 1)`.
Baseline and variant differ ONLY by STAND_BLOCK.

`prev_arr_gap` (= MVT - T_c) sits in the BASELINE deliberately. RED_TEAM.md 4.3 argues the
incumbent already carries the stand signal through it; putting it in the baseline makes the
stand block prove itself against the strongest plausible incumbent rather than a strawman.
What the block adds is what a tree cannot construct from inputs it already holds - notably
`gapa = AOBT_3 - T_c`, which is exactly `prev_arr_gap - proxy`, a difference of two features.

Memory: months are cached separately and the design matrix is filled month-by-month straight
into one preallocated float32 array, because a float32 pandas frame handed to sklearn is copied
to float64 (this killed a 3.8 GB run in an earlier session).
"""
from __future__ import annotations

import argparse
import gc
import glob
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor as HGR

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache_stand"
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
HOLDOUT = (1, 7)
SMOOTH = 50.0
EPOCH = pd.Timestamp("2024-12-01", tz="UTC")

COLS = ["PHASE_mvt", "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt",
        "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "MVT_TIME_UTC_mvt",
        "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt",
        "EOBT_1_flt", "LOBT_flt", "IOBT_flt", "ARVT_1_flt", "ARVT_3_flt", "CALLSIGN_flt",
        "ADES_FILED_flt", "AIRCRAFT_TYPE_flt", "FLIGHT_RULE_flt", "AIRCRAFT_OPERATOR_flt",
        "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"]

BASE = ["proxy", "sp", "eobt_p", "lobt_p", "iobt_p", "aobt_sched", "eobt_sched",
        "aobt_eobt", "lobt_sched", "eobt_lobt", "iobt_lobt", "airborne3", "airborne1",
        "arvt_diff", "aobt_sec", "mvt_sec", "sched_sec", "hr", "tmin", "dow", "doy",
        "actype_null", "f_callsign", "f_ades", "f_type", "f_rule"]
SURF = ["n_push", "rwy_b30", "apt_b30", "arr_b30", "sched_prev60", "sched_next60",
        "sched_day", "rwy_rate", "dep_dur", "rwy_dur", "arr_dur", "arr_ob60_taxiin",
        "arr_ob60_cdiff", "arr_ob180_cdiff"]
STANDHIST = ["prev_arr_gap", "prev_arr_taxiin", "prev_arr_cdiff", "prev_dep_gap"]
STAND_BLOCK = ["gapa", "gaps", "n_sch_aobt", "n_aobt_mvt", "n_2h_mvt", "witness",
               "ty_match", "gapa_pos", "stand_slack_med", "stand_slack_iqr"]
#: filled in-fold at fit time from training months only (they read the target)
STAND_INFOLD = ["stand_slack_med", "stand_slack_iqr"]
ENC_KEYS = ["ADEP_mvt", "STAND_mvt", "stand_pref", "airline", "ADES_mvt",
            "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt",
            "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "ars"]
ENC = [f"{p}_{k}" for k in ENC_KEYS for p in ("te", "de")]
BASELINE_FEATS = BASE + SURF + STANDHIST + ENC
VARIANT_FEATS = BASELINE_FEATS + STAND_BLOCK

es = lambda s: (s - EPOCH).dt.total_seconds().to_numpy()
# pandas 3: .astype(str) on an Arrow string column leaves NA as float nan, which then
# fails string concatenation. Fill before casting. Never use "\x00" as a key separator -
# the Arrow string path treats NUL as a terminator and collapses composite keys.
sarr = lambda s: s.fillna("NA").astype(str).to_numpy()


def _win_count(anchor, times, lo, hi):
    """#events with time in (anchor+lo, anchor+hi]; times must be sorted."""
    return (np.searchsorted(times, anchor + hi, "right")
            - np.searchsorted(times, anchor + lo, "right"))


def _grouped(keys_d, keys_a, times_a, vals_a=None):
    idx = {}
    for k, g in pd.Series(np.arange(len(keys_a))).groupby(keys_a, sort=False):
        idx[k] = g.to_numpy()
    assert len(idx) > 1, f"grouping collapsed to {len(idx)} keys - NUL separator?"
    return idx


def build_month(path: pathlib.Path, serve: bool = False) -> pd.DataFrame:
    """Feature frame for one calendar-month file. `serve=True` builds the scored rows.

    Serve mode changes exactly two things and is guarded by tests/test_serve_mode.py: the
    departure filter no longer requires a label (TAXITIME / BLOCK are blank on every scored
    row), and `delta` / `y` are NaN. Every reference stream is built the same way, so the
    only rows that differ from training mode are the handful the relaxed filter admits.
    """
    return build_features(pq.read_table(path, columns=COLS).to_pandas(), serve=serve)


def build_ranking(path: pathlib.Path) -> pd.DataFrame:
    """Serve-mode features for the evaluation file, built one calendar month at a time.

    Every training cache is one calendar month and `sched_day` counts departures per file;
    the evaluation file holds January AND July, so building it as one unit would double a
    training feature at serve time. Split on the movement month, as the training files are.
    """
    t = pq.read_table(path, columns=COLS).to_pandas()
    if t.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError(f"{t.MVT_TIME_UTC_mvt.isna().sum()} rows without MVT_TIME cannot "
                         "be assigned to a calendar month")
    ym = (t.MVT_TIME_UTC_mvt.dt.year * 100 + t.MVT_TIME_UTC_mvt.dt.month).to_numpy()
    parts = [build_features(t[ym == m], serve=True) for m in np.unique(ym)]
    o = pd.concat(parts, ignore_index=True)
    assert o.MVT_ID_mvt.is_unique, "duplicate MVT_ID across the per-month builds"
    return o


def build_features(t: pd.DataFrame, serve: bool = False) -> pd.DataFrame:
    arr = t[(t.PHASE_mvt == "ARR") & t.BLOCK_TIME_UTC_mvt.notna()]
    arr = arr[arr.ADES_mvt.isin(APTS)]
    a_apt = sarr(arr.ADES_mvt)
    a_t = es(arr.BLOCK_TIME_UTC_mvt)
    a_taxi = arr.TAXITIME_SEC_mvt.astype(float).to_numpy()
    a_cdiff = np.where(arr.ARVT_3_flt.notna().to_numpy(),
                       es(arr.BLOCK_TIME_UTC_mvt) - es(arr.ARVT_3_flt), np.nan)
    a_stand = sarr(arr.STAND_mvt)
    a_type = sarr(arr.AIRCRAFT_TYPE_mvt)
    a_ok = arr.STAND_mvt.notna().to_numpy()

    if serve:
        # the scored rows carry no TAXITIME and no BLOCK; admit every departure that has
        # the clocks the features read. Neither hidden column is consulted.
        d = t[(t.PHASE_mvt == "DEP") & t.AOBT_3_flt.notna()
              & t.MVT_TIME_UTC_mvt.notna() & t.SCHED_TIME_UTC_mvt.notna()]
    else:
        d = t[(t.PHASE_mvt == "DEP") & t.TAXITIME_SEC_mvt.notna()
              & t.BLOCK_TIME_UTC_mvt.notna() & (t.TAXITIME_SEC_mvt > 0)
              & t.AOBT_3_flt.notna()]
    d = d[d.ADEP_mvt.isin(APTS)].copy()
    d = d.sort_values("MVT_TIME_UTC_mvt", kind="mergesort").reset_index(drop=True)

    mvt, sch = es(d.MVT_TIME_UTC_mvt), es(d.SCHED_TIME_UTC_mvt)
    aobt, eobt = es(d.AOBT_3_flt), es(d.EOBT_1_flt)
    lobt, iobt = es(d.LOBT_flt), es(d.IOBT_flt)
    arvt1, arvt3 = es(d.ARVT_1_flt), es(d.ARVT_3_flt)

    o = pd.DataFrame(index=d.index)
    if serve:
        o["delta"] = np.nan
        o["y"] = np.nan
    else:
        o["delta"] = es(d.BLOCK_TIME_UTC_mvt) - aobt
        o["y"] = d.TAXITIME_SEC_mvt.astype(float).to_numpy()
    o["proxy"] = mvt - aobt
    o["sp"] = mvt - sch
    o["eobt_p"] = mvt - eobt
    o["lobt_p"] = mvt - lobt
    o["iobt_p"] = mvt - iobt
    o["aobt_sched"] = aobt - sch
    o["eobt_sched"] = eobt - sch
    o["aobt_eobt"] = aobt - eobt
    o["lobt_sched"] = lobt - sch
    o["eobt_lobt"] = eobt - lobt
    o["iobt_lobt"] = iobt - lobt
    o["airborne3"] = arvt3 - mvt
    o["airborne1"] = arvt1 - mvt
    o["arvt_diff"] = arvt3 - arvt1
    o["aobt_sec"] = d.AOBT_3_flt.dt.second.to_numpy()
    o["mvt_sec"] = d.MVT_TIME_UTC_mvt.dt.second.to_numpy()
    o["sched_sec"] = d.SCHED_TIME_UTC_mvt.dt.second.to_numpy()
    o["hr"] = d.MVT_TIME_UTC_mvt.dt.hour.to_numpy()
    o["tmin"] = (d.MVT_TIME_UTC_mvt.dt.hour * 60 + d.MVT_TIME_UTC_mvt.dt.minute).to_numpy()
    o["dow"] = d.MVT_TIME_UTC_mvt.dt.dayofweek.to_numpy()
    o["doy"] = d.MVT_TIME_UTC_mvt.dt.dayofyear.to_numpy()
    o["actype_null"] = d.AIRCRAFT_TYPE_mvt.isna().to_numpy().astype(float)
    o["f_callsign"] = (sarr(d.CALLSIGN_flt)
                       != sarr(d.FLIGHT_mvt)).astype(float)
    o["f_ades"] = (sarr(d.ADES_FILED_flt)
                   != sarr(d.ADES_mvt)).astype(float)
    o["f_type"] = (sarr(d.AIRCRAFT_TYPE_flt)
                   != sarr(d.AIRCRAFT_TYPE_mvt)).astype(float)
    o["f_rule"] = (sarr(d.FLIGHT_RULE_flt)
                   != sarr(d.FLIGHT_RULE_mvt)).astype(float)

    d_apt = sarr(d.ADEP_mvt)
    d_rwy = (d_apt + "|" + sarr(d.RUNWAY_mvt))
    d_stand = (d_apt + "|" + sarr(d.STAND_mvt))
    a_key = (a_apt + "|" + a_stand)

    for name in SURF + STANDHIST + [c for c in STAND_BLOCK if c not in STAND_INFOLD]:
        o[name] = np.nan

    # ---- airport / runway ambient, backward windows anchored at take-off ----
    for keyd, keya, cols in ((d_apt, a_apt, ("apt_b30", "arr_b30", "sched_prev60",
                                             "sched_next60", "sched_day", "dep_dur",
                                             "arr_dur", "arr_ob60_taxiin",
                                             "arr_ob60_cdiff", "arr_ob180_cdiff",
                                             "n_push")),
                             (d_rwy, None, ("rwy_b30", "rwy_rate", "rwy_dur"))):
        for k, rows in pd.Series(np.arange(len(d))).groupby(keyd, sort=False):
            j = rows.to_numpy()
            m_sorted = mvt[j]                       # already sorted by MVT
            if "apt_b30" in cols:
                o.loc[d.index[j], "apt_b30"] = _win_count(m_sorted, m_sorted, -1800, 0)
                o.loc[d.index[j], "n_push"] = _win_count(m_sorted, np.sort(aobt[j]), -1800, 0)
                s_sorted = np.sort(sch[j])
                o.loc[d.index[j], "sched_prev60"] = _win_count(m_sorted, s_sorted, -3600, 0)
                o.loc[d.index[j], "sched_next60"] = _win_count(m_sorted, s_sorted, 0, 3600)
                o.loc[d.index[j], "sched_day"] = len(j)
                px = pd.Series(proxy_j := (mvt[j] - aobt[j]))
                o.loc[d.index[j], "dep_dur"] = px.rolling(30, min_periods=3).median().to_numpy()
                if keya is not None:
                    am = a_apt == k
                    at, ax, ac = a_t[am], a_taxi[am], a_cdiff[am]
                    so = np.argsort(at, kind="mergesort")
                    at, ax, ac = at[so], ax[so], ac[so]
                    o.loc[d.index[j], "arr_b30"] = _win_count(m_sorted, at, -1800, 0)
                    hi = np.searchsorted(at, m_sorted, "right")
                    lo60 = np.searchsorted(at, m_sorted - 3600, "right")
                    lo180 = np.searchsorted(at, m_sorted - 10800, "right")
                    cx, cc = np.concatenate([[0.0], np.nancumsum(ax)]), np.concatenate([[0.0], np.nancumsum(np.nan_to_num(ac))])
                    n60 = np.maximum(hi - lo60, 1)
                    o.loc[d.index[j], "arr_dur"] = (cx[hi] - cx[lo60]) / n60
                    o.loc[d.index[j], "arr_ob60_taxiin"] = (cx[hi] - cx[lo60]) / n60
                    o.loc[d.index[j], "arr_ob60_cdiff"] = (cc[hi] - cc[lo60]) / n60
                    o.loc[d.index[j], "arr_ob180_cdiff"] = ((cc[hi] - cc[lo180])
                                                            / np.maximum(hi - lo180, 1))
            else:
                o.loc[d.index[j], "rwy_b30"] = _win_count(m_sorted, m_sorted, -1800, 0)
                o.loc[d.index[j], "rwy_rate"] = _win_count(m_sorted, m_sorted, -3600, 0) / 60.0
                o.loc[d.index[j], "rwy_dur"] = pd.Series(
                    mvt[j] - aobt[j]).rolling(20, min_periods=3).median().to_numpy()

    # ---- stand chain: T_c anchored at take-off (baseline) and at AOBT_3 (block) ----
    aidx = {}
    ok = a_ok & pd.notna(a_stand)
    ka, ta, tx, cd, ty = a_key[ok], a_t[ok], a_taxi[ok], a_cdiff[ok], a_type[ok]
    so = np.argsort(ta, kind="mergesort")
    ka, ta, tx, cd, ty = ka[so], ta[so], tx[so], cd[so], ty[so]
    for k, g in pd.Series(np.arange(len(ta))).groupby(ka, sort=False):
        aidx[k] = g.to_numpy()
    assert len(aidx) > 100, f"stand index collapsed to {len(aidx)} keys - NUL separator?"

    d_type = sarr(d.AIRCRAFT_TYPE_mvt)
    for k, rows in pd.Series(np.arange(len(d))).groupby(d_stand, sort=False):
        j = rows.to_numpy()
        pos = aidx.get(k)
        # previous NM off-block from this stand
        dj = np.sort(aobt[j])
        o.loc[d.index[j], "prev_dep_gap"] = mvt[j] - dj[np.maximum(
            np.searchsorted(dj, mvt[j], "right") - 1, 0)]
        if pos is None:
            continue
        st, sx, sc, sy = ta[pos], tx[pos], cd[pos], ty[pos]
        hm = np.searchsorted(st, mvt[j], "right") - 1        # last ARR <= take-off
        ha = np.searchsorted(st, aobt[j], "right") - 1       # last ARR <= AOBT_3
        okm, oka = hm >= 0, ha >= 0
        Tm = np.where(okm, st[np.maximum(hm, 0)], np.nan)
        Ta = np.where(oka, st[np.maximum(ha, 0)], np.nan)
        o.loc[d.index[j], "prev_arr_gap"] = mvt[j] - Tm
        o.loc[d.index[j], "prev_arr_taxiin"] = np.where(okm, sx[np.maximum(hm, 0)], np.nan)
        o.loc[d.index[j], "prev_arr_cdiff"] = np.where(okm, sc[np.maximum(hm, 0)], np.nan)
        o.loc[d.index[j], "gapa"] = aobt[j] - Ta
        o.loc[d.index[j], "gaps"] = sch[j] - Ta
        o.loc[d.index[j], "gapa_pos"] = (aobt[j] - Ta > 0).astype(float)
        o.loc[d.index[j], "witness"] = oka.astype(float)
        o.loc[d.index[j], "ty_match"] = np.where(
            oka, (d_type[j] == sy[np.maximum(ha, 0)]).astype(float), np.nan)
        o.loc[d.index[j], "n_sch_aobt"] = (np.searchsorted(st, aobt[j], "right")
                                           - np.searchsorted(st, sch[j], "right"))
        o.loc[d.index[j], "n_aobt_mvt"] = (np.searchsorted(st, mvt[j], "right")
                                           - np.searchsorted(st, aobt[j], "right"))
        o.loc[d.index[j], "n_2h_mvt"] = (np.searchsorted(st, mvt[j], "right")
                                         - np.searchsorted(st, mvt[j] - 7200, "right"))

    o["month"] = d.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    o["MVT_ID_mvt"] = d.MVT_ID_mvt.to_numpy()      # the splice key; not a feature
    o["ap"] = d_apt
    for k in ("STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt",
              "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt",
              "FLIGHT_TYPE_flt"):
        o[k] = sarr(d[k])
    o["ADEP_mvt"] = d_apt
    o["stand_pref"] = pd.Series(sarr(d.STAND_mvt)).str[:2].to_numpy()
    o["airline"] = pd.Series(sarr(d.FLIGHT_mvt)).str[:3].to_numpy()
    o["ars"] = d_stand + "|" + sarr(d.RUNWAY_mvt)
    return o.reset_index(drop=True)


def cmd_cache(smoke: bool, ranking: bool = False):
    CACHE.mkdir(parents=True, exist_ok=True)
    if ranking:
        if smoke:
            raise ValueError("--smoke does not apply to --ranking: a partial ranking cache "
                             "at the real path would be consumed by the full fit")
        out = CACHE / "ranking.parquet"
        o = build_ranking(RAW / "ranking.parquet")
        o.to_parquet(out, index=False)
        print(f"ranking.parquet -> {out.name}  rows={len(o):,}  cols={o.shape[1]}  "
              f"months={sorted(o.month.unique().tolist())}", flush=True)
        return
    paths = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
    if smoke:
        paths = paths[:1]
    for p in paths:
        p = pathlib.Path(p)
        out = CACHE / (p.stem + ".parquet")
        o = build_month(p)
        o.to_parquet(out, index=False)
        print(f"{p.name} -> {out.name}  rows={len(o):,}  cols={o.shape[1]}")
        del o
        gc.collect()


def _smooth_enc(tr_key, tr_val, keys, prior, k=SMOOTH):
    g = pd.DataFrame({"k": tr_key, "v": tr_val}).groupby("k", observed=True).v.agg(["sum", "count"])
    enc = (g["sum"] + prior * k) / (g["count"] + k)
    return pd.Series(keys).map(enc).fillna(prior).to_numpy(np.float32)


def infold_encodings(keys, y, delta, tr_mask, enc_keys=ENC_KEYS):
    """The 24 smoothed target encodings, fitted on TRAINING rows only.

    The leak lives at this call site, not inside `_smooth_enc`: passing the full key array
    instead of `keys[tr_mask]` lets held-out targets into the features and every downstream
    interval becomes meaningless. Guarded by tests/test_stand_fit.py.
    """
    y_tr, d_tr = np.asarray(y)[tr_mask], np.asarray(delta)[tr_mask]
    py, pdl = float(y_tr.mean()), float(d_tr.mean())
    out = {}
    for k in enc_keys:
        kk = keys[k].to_numpy()
        out[f"te_{k}"] = _smooth_enc(kk[tr_mask], y_tr, kk, py)
        out[f"de_{k}"] = _smooth_enc(kk[tr_mask], d_tr, kk, pdl)
    return out


def infold_slack_stats(gapa, delta, ars, tr_mask):
    """Per-stand median/IQR of stand-vacancy slack, fitted on TRAINING rows only.

    `slack = T_c - BLOCK = -(gapa + delta)` reads `delta`, which is the target. Every
    statistic here must therefore be computed under `tr_mask` and merely *applied* to the
    held-out rows; if the mask is dropped the held-out targets enter the features and the
    A/B silently measures a leak. Guarded by tests/test_stand_fit.py.
    """
    slack = -(np.asarray(gapa, dtype=float) + np.asarray(delta, dtype=float))
    ok = tr_mask & np.isfinite(slack) & (slack > 0) & (slack < 21600)
    if not ok.any():
        raise ValueError("no admissible training rows for the slack statistics")
    g = pd.DataFrame({"k": np.asarray(ars)[ok], "s": slack[ok]}).groupby("k", observed=True).s
    med, iqr = g.median(), g.quantile(.75) - g.quantile(.25)
    gm = float(np.median(slack[ok]))
    k = pd.Series(np.asarray(ars))
    return (k.map(med).fillna(gm).to_numpy(np.float32),
            k.map(iqr).fillna(float(np.nanmedian(iqr))).to_numpy(np.float32))


def cmd_fit(smoke: bool):
    paths = sorted(glob.glob(str(CACHE / "training_2025-*.parquet")))
    assert paths, "run `cache` first"
    d = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    gc.collect()
    te_m = d.month.isin(HOLDOUT).to_numpy()
    if smoke:
        keep = (d.month.isin([1, 2, 3])).to_numpy()
        d, te_m = d[keep].reset_index(drop=True), te_m[keep]
    tr_m = ~te_m
    print(f"rows {len(d):,}   train {tr_m.sum():,}   test(Jan+Jul) {te_m.sum():,}")

    # ---- in-fold encodings: fitted on training months only ----
    for col, vals in infold_encodings(d, d.y.to_numpy(), d.delta.to_numpy(), tr_m).items():
        d[col] = vals

    # ---- in-fold stand slack statistics: slack = -(gapa + delta), training rows only ----
    d["stand_slack_med"], d["stand_slack_iqr"] = infold_slack_stats(
        d.gapa.to_numpy(), d.delta.to_numpy(), d.ars.to_numpy(), tr_m)

    # the string columns have done their work (encodings + per-stand slack); drop them
    # before the design matrix is allocated, or 2M rows of Arrow strings sit beside it
    d = d.drop(columns=[c for c in set(ENC_KEYS) | {"ars"} if c in d.columns and c != "ap"])
    gc.collect()

    #: >=3 seeds. A single-seed paired bootstrap certified seed noise as ESTABLISHED once
    #: (Protocol B, 2026-09-08: increments +0.245 / -0.212 / -0.253, sd 0.28 s, sign flips).
    SEEDS = (0,) if smoke else (0, 1, 2)
    cfg = dict(max_iter=60 if smoke else 400, learning_rate=0.05,
               max_leaf_nodes=127, min_samples_leaf=20, early_stopping=False)
    yv, proxy, dlt = d.y.to_numpy(), d.proxy.to_numpy(), d.delta.to_numpy()
    out = {}
    for name, feats in (("baseline", BASELINE_FEATS), ("+stand", VARIANT_FEATS)):
        X = np.empty((len(d), len(feats)), dtype=np.float32)
        for i, c in enumerate(feats):
            X[:, i] = pd.to_numeric(d[c], errors="coerce").to_numpy(np.float32)
        preds = []
        for sd_ in SEEDS:
            mo = HGR(random_state=sd_, **cfg).fit(X[tr_m], dlt[tr_m])
            preds.append(mo.predict(X[te_m]))
            del mo
            gc.collect()
        per_seed = [np.sqrt(((yv[te_m] - np.maximum(proxy[te_m] - p_, 1.0)) ** 2).mean())
                    for p_ in preds]
        print(f"    per-seed RMSE: " + " ".join(f"{v:.2f}" for v in per_seed)
              + f"   sd {np.std(per_seed):.3f}")
        raw = proxy[te_m] - np.mean(preds, axis=0)
        # runtime guards. The floor at 1 s is required (taxi-out is strictly positive) but
        # it must not be doing real work: if it binds often the delta model is broken, not
        # merely imprecise, and the RMSE would be flattered by the clamp.
        assert np.isfinite(raw).all(), f"{name}: non-finite predictions"
        bound = float((raw < 1.0).mean())
        assert bound < 0.005, f"{name}: positivity floor binds on {100*bound:.2f}% of rows"
        out.setdefault("_seeds", {})[name] = per_seed
        print(f"    guard: floor binds on {100*bound:.3f}% of rows; "
              f"pred range [{raw.min():.0f}, {raw.max():.0f}]")
        pred = np.maximum(raw, 1.0)
        out[name] = pred
        print(f"  {name:9s} {len(feats):3d} feats  RMSE = "
              f"{np.sqrt(((yv[te_m] - pred) ** 2).mean()):8.2f}")
        del X, m
        gc.collect()

    yt = yv[te_m]
    sA, sB = out["_seeds"]["baseline"], out["_seeds"]["+stand"]
    seed_sd = float(np.std([a - b for a, b in zip(sA, sB)]))
    e0, e1 = yt - out["baseline"], yt - out["+stand"]
    r0, r1 = np.sqrt((e0 ** 2).mean()), np.sqrt((e1 ** 2).mean())
    W_M = 339551 / 344841
    dmse = W_M * (r0 ** 2 - r1 ** 2)
    rng = np.random.default_rng(0)
    bs = np.empty(2000)
    for i in range(2000):
        j = rng.integers(0, len(yt), len(yt))
        bs[i] = np.sqrt((e0[j] ** 2).mean()) - np.sqrt((e1[j] ** 2).mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print("\n" + "=" * 92)
    print(f"PAIRED A/B - fold A (Jan+Jul 2025 held out), matched rows")
    print("=" * 92)
    print(f"  baseline           {r0:8.2f}      (E1 reference 235.95)")
    print(f"  + stand block      {r1:8.2f}")
    est = lo > 0 and (r0 - r1) > 2 * seed_sd
    print(f"  paired gain        {r0 - r1:+8.2f} s   95% CI [{lo:+.2f}, {hi:+.2f}]"
          f"   seed sd {seed_sd:.3f}"
          f"   {'ESTABLISHED' if est else 'NOT ESTABLISHED'}")
    if lo > 0 and not est:
        print("    (bootstrap excludes zero but the gain is inside 2x seed noise - "
              "the interval is measuring fitting variance, not signal)")
    print(f"  GLOBAL MSE REMOVED {dmse:+,.0f}   of 25,979 remaining "
          f"({100 * dmse / 25979:+.1f}%)")
    band = "MAJOR" if dmse > 4000 else "USEFUL" if dmse > 1000 else "CLOSED (<1000)"
    print(f"  Amendment 5 band:  {band}")

    sub = d[te_m]
    print(f"\n{'cut':28s}{'n':>9s}{'baseline':>11s}{'+stand':>10s}{'gain':>9s}")
    cuts = [("witness gapa<=600", (sub.gapa <= 600).to_numpy()),
            ("witness gapa<=1800", (sub.gapa <= 1800).to_numpy()),
            ("non-witness", ~(sub.gapa <= 1800).fillna(False).to_numpy()),
            ("delta < -600 (the tail)", (dlt[te_m] < -600)),
            ("delta >= -600", (dlt[te_m] >= -600))]
    for nm, m in cuts:
        m = np.nan_to_num(m).astype(bool)
        if m.sum() < 30: continue
        a, b = np.sqrt((e0[m] ** 2).mean()), np.sqrt((e1[m] ** 2).mean())
        print(f"{nm:28s}{m.sum():9,d}{a:11.1f}{b:10.1f}{a - b:+9.2f}")
    print()
    ap = sub.ap.to_numpy()
    print(f"{'airport':28s}{'n':>9s}{'baseline':>11s}{'+stand':>10s}{'gain':>9s}")
    for a_ in APTS:
        m = ap == a_
        if m.sum() < 30: continue
        x, z = np.sqrt((e0[m] ** 2).mean()), np.sqrt((e1[m] ** 2).mean())
        print(f"{a_:28s}{m.sum():9,d}{x:11.1f}{z:10.1f}{x - z:+9.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["cache", "fit"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ranking", action="store_true",
                    help="cache: build data/cache_stand/ranking.parquet (serve mode) instead "
                         "of the training months")
    a = ap.parse_args()
    if a.cmd == "cache":
        cmd_cache(a.smoke, ranking=a.ranking)
    else:
        cmd_fit(a.smoke)
