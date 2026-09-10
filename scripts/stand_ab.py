"""Paired A/B: does the stand-occupancy block remove global MSE over an E1-equivalent model?

Pre-registered in plans/PREREG_taxiout_2026_09_08.md Amendment 5.

    python3.11 scripts/stand_ab.py cache [--smoke]   # build per-month feature caches
    python3.11 scripts/stand_ab.py fit   [--smoke]   # fold A, baseline vs +stand, paired bootstrap
    python3.11 scripts/stand_ab.py queue-cache [--ranking] [--smoke]
                                  # v6 push-anchored queue block -> data/cache_queue/ (Amendment 14)
    python3.11 scripts/stand_ab.py day-cache   [--ranking] [--smoke]
    python3.11 scripts/stand_ab.py order-cache [--ranking] [--smoke]   # Amendment 22, Arm F
                                  # airport-day regime block -> data/cache_day/ (Amendment 19, arm D)
    python3.11 scripts/stand_ab.py unmatched-cache [--ranking] [--smoke]
                                  # the UNMATCHED departures' features -> data/cache_unmatched/ (the
                                  # unified all-rows arm; every block, NaN where the row's own AOBT_3 is read)

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
import resource
import sys
import time

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

# ---- v6 queue block (PREREG Amendment 14): push-anchored surface congestion ----------------
QCACHE = ROOT / "data" / "cache_queue"
#: the raw columns the queue block reads - a subset of COLS. A departure's own BLOCK_TIME and
#: TAXITIME appear only in the training-mode row filter (label present and positive), never in
#: a feature; both are blank on every scored row.
QCOLS = ["PHASE_mvt", "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt",
         "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
         "AOBT_3_flt"]
#: per matched departure i (a = AOBT_3 push, t = MVT_TIME take-off), j = other departures of the
#: same reference stream, arr = arrivals (land = MVT_TIME, onblock = BLOCK_TIME):
#:   q_apt_at_push          #{j: a_j < a_i < t_j}            airport   pushed, not yet airborne
#:   q_rwy_at_push          same                              runway
#:   q_pushed_after_me      #{j: a_i < a_j < t_i}            airport
#:   q_rwy_tko_in_taxi      #{j: a_i < t_j < t_i}            runway    take-offs during my taxi
#:   q_rwy_push_pre20       #{j: a_i - 1200 <= a_j < a_i}    runway
#:   q_dep_tko_sym15        #{j: |t_j - t_i| <= 900} - 1     airport
#:   q_rwy_tko_sym10        #{j: |t_j - t_i| <= 600} - 1     runway
#:   q_arr_taxiing_at_push  #{arr: land < a_i < onblock}     airport
#:   q_arr_taxiin_sym30_push  mean arr taxi-in, onblock in [a_i - 1800, a_i + 1800]; NaN if none
#:   q_arr_taxiin_sym30_tko   same around t_i
#:   q_rwy_ambient_proxy    median proxy_j, runway, t_j in [t_i - 3600, t_i + 3600], self excluded
#:   q_next_arr_onblock_gap (first arr onblock at my stand after a_i, within 24 h) - t_i
QUEUE_FEATS = ["q_apt_at_push", "q_rwy_at_push", "q_pushed_after_me", "q_rwy_tko_in_taxi",
               "q_rwy_push_pre20", "q_dep_tko_sym15", "q_rwy_tko_sym10", "q_arr_taxiing_at_push",
               "q_arr_taxiin_sym30_push", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy",
               "q_next_arr_onblock_gap"]
#: NaN when the window is empty; every other queue feature is a count and is 0 when empty
QUEUE_VALUE_FEATS = ["q_arr_taxiin_sym30_push", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy",
                     "q_next_arr_onblock_gap"]
QUEUE_WINDOWS = dict(push_pre=1200, tko_sym_apt=900, tko_sym_rwy=600, arr_sym=1800,
                     ambient=3600, next_arr=86400)

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


def _own_clock_features(d: pd.DataFrame, serve: bool) -> tuple:
    """The per-row columns a departure computes from its OWN clocks - delta, y and the BASE
    features - for a frame already filtered and in take-off order; returns (frame, mvt, sch,
    aobt) in seconds since EPOCH. Shared by build_features (the matched cache) and
    build_unmatched (the unmatched cache) so the two cannot drift: on an unmatched row AOBT_3
    is NaT and every AOBT_3-derived value here is NaN by arithmetic, not by a special case."""
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
    return o, mvt, sch, aobt


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

    o, mvt, sch, aobt = _own_clock_features(d, serve)

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


def infold_encodings(keys, y, delta, tr_mask, enc_keys=ENC_KEYS, delta_mask=None):
    """The 24 smoothed target encodings, fitted on TRAINING rows only.

    The leak lives at this call site, not inside `_smooth_enc`: passing the full key array
    instead of `keys[tr_mask]` lets held-out targets into the features and every downstream
    interval becomes meaningless. Guarded by tests/test_stand_fit.py.

    `delta_mask` (the unified all-rows design) names the rows whose delta is defined - the
    matched ones; the 12 delta encodings and their prior are then fitted on `tr_mask &
    delta_mask` while the 12 y encodings stay on every training row. Without it a NaN delta
    on a training row is refused: it would poison the prior and every delta encoding.
    """
    tr_mask = np.asarray(tr_mask, dtype=bool)
    d_mask = tr_mask if delta_mask is None else (tr_mask & np.asarray(delta_mask, dtype=bool))
    y_tr, d_tr = np.asarray(y)[tr_mask], np.asarray(delta)[d_mask]
    if np.isnan(np.asarray(d_tr, dtype="float64")).any():
        raise ValueError(f"NaN delta on {int(np.isnan(np.asarray(d_tr, dtype='float64')).sum()):,} rows of the "
                         "delta-encoding set: pass delta_mask to fit the delta encodings where delta is defined")
    py, pdl = float(y_tr.mean()), float(d_tr.mean())
    out = {}
    for k in enc_keys:
        kk = keys[k].to_numpy()
        out[f"te_{k}"] = _smooth_enc(kk[tr_mask], y_tr, kk, py)
        out[f"de_{k}"] = _smooth_enc(kk[d_mask], d_tr, kk, pdl)
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


# =============================================================================================
# v6 queue block: push-anchored surface congestion for the matched rows (Amendment 14)
# =============================================================================================

def _count_lt(sorted_vals, x):
    """#{v in sorted_vals: v < x}, vectorised over x."""
    return np.searchsorted(sorted_vals, x, "left")


def _count_le(sorted_vals, x):
    """#{v in sorted_vals: v <= x}, vectorised over x."""
    return np.searchsorted(sorted_vals, x, "right")


def _count_open(sorted_vals, lo, hi):
    """#{v in sorted_vals: lo < v < hi}, vectorised over (lo, hi); 0 when hi <= lo.

    The difference #{v < hi} - #{v <= lo} is the count only for a non-empty interval; for an
    empty one it goes negative. The intervals here are (push, take-off), and 177 evaluation
    rows (64 in January 2025) have take-off at or before push, so the clamp is the definition
    ("nothing lies in an empty interval"), not a patch - surfaced 2026-09-09 by the count-sign
    guard in tests/test_queue_features.py.
    """
    return np.maximum(_count_lt(sorted_vals, hi) - _count_le(sorted_vals, lo), 0)


def _groups(keys):
    """Row indices per key, in order of first appearance, rows ascending within each group.

    Keys are numpy object arrays joined with "|" (see `sarr`); the caller checks the group count
    against the coarser key, which is what catches a collapsed separator - the historical NUL
    bug truncated "EGLL|27L" to "EGLL" in pandas Arrow strings and every join went empty.
    """
    return {k: g.to_numpy() for k, g in pd.Series(np.arange(len(keys))).groupby(keys, sort=False)}


def _dep_stream(t: pd.DataFrame, serve: bool) -> pd.DataFrame:
    """build_features's departure filter and take-off order, verbatim.

    A copy rather than a refactor of build_features, because the v4 fold consumes those caches
    while this ships; tests/test_queue_features.py pins the MVT_ID sequence of both modes to
    build_month's, so any drift between the two copies is caught there.
    """
    if serve:
        d = t[(t.PHASE_mvt == "DEP") & t.AOBT_3_flt.notna()
              & t.MVT_TIME_UTC_mvt.notna() & t.SCHED_TIME_UTC_mvt.notna()]
    else:
        d = t[(t.PHASE_mvt == "DEP") & t.TAXITIME_SEC_mvt.notna()
              & t.BLOCK_TIME_UTC_mvt.notna() & (t.TAXITIME_SEC_mvt > 0)
              & t.AOBT_3_flt.notna()]
    d = d[d.ADEP_mvt.isin(APTS)]
    return d.sort_values("MVT_TIME_UTC_mvt", kind="mergesort").reset_index(drop=True)


def _arr_stream(t: pd.DataFrame) -> pd.DataFrame:
    """build_features's arrival filter, verbatim: arrivals at the ten airports with an on-block."""
    arr = t[(t.PHASE_mvt == "ARR") & t.BLOCK_TIME_UTC_mvt.notna()]
    return arr[arr.ADES_mvt.isin(APTS)]


def _on_ground(start, end, x):
    """#{j: start_j < x < end_j} for every x - movements between two events when x happens.

    Exact through two one-dimensional counts: an interval can contain x only if start_j < end_j,
    and on that subset {end_j <= x} is nested inside {start_j < x}, so the count is
    #{start_j < x} - #{end_j <= x}. Intervals with end <= start (proxy <= 0, taxi-in <= 0) can
    never contain x and are dropped - that is the definition, not an approximation.
    """
    m = end > start
    return _count_lt(np.sort(start[m]), x) - _count_le(np.sort(end[m]), x)


def _window_mean(sorted_t, vals, x, w):
    """Mean of vals over rows with sorted_t in [x - w, x + w]; NaN where the window is empty.

    `vals` is aligned with `sorted_t`; NaN values are skipped. The sums are exact: every input
    is integer-valued seconds.
    """
    lo, hi = _count_lt(sorted_t, x - w), _count_le(sorted_t, x + w)
    fin = np.isfinite(vals)
    cs = np.concatenate([[0.0], np.cumsum(np.where(fin, vals, 0.0))])
    cn = np.concatenate([[0], np.cumsum(fin)])
    n = cn[hi] - cn[lo]
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def _excluded_median(sorted_t, vals, w):
    """Median of vals over the OTHER rows with sorted_t in [t_i - w, t_i + w]; NaN if none.

    A windowed median with the row itself removed has no cumulative form; the window is a
    contiguous slice of the sorted group, so this is one np.median per row on a short slice.
    """
    lo, hi = _count_lt(sorted_t, sorted_t - w), _count_le(sorted_t, sorted_t + w)
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        if hi[i] - lo[i] > 1:
            out[i] = np.median(np.delete(vals[lo[i]:hi[i]], i - lo[i]))
    return out


def build_queue(t: pd.DataFrame, serve: bool = False) -> pd.DataFrame:
    """`MVT_ID_mvt` + QUEUE_FEATS (float32) for exactly the rows build_features(t, serve) returns,
    in the same take-off order.

    Per departure i: push a_i = AOBT_3, take-off t_i = MVT_TIME. Other departures j come from
    the same reference stream as build_features - never a different filter, because mixing
    streams once made a cumulative count drift between modes and killed the signal. Arrivals
    land at MVT_TIME and go on-block at BLOCK_TIME; both, and their taxi-in, are populated on
    the evaluation file. Later events of OTHER movements are legitimate inputs (Amendment 5).
    Counts are 0 when nothing qualifies; the taxi-in levels, the ambient median and the stand
    gap are NaN when their window is empty. Nothing reads a departure's own BLOCK or TAXITIME
    (guarded by the hidden-clock test).
    """
    d, arr = _dep_stream(t, serve), _arr_stream(t)
    n = len(d)
    if n == 0:
        raise ValueError(f"no admissible departures (serve={serve}) - nothing to build")
    a, tk = es(d.AOBT_3_flt), es(d.MVT_TIME_UTC_mvt)
    px = tk - a
    d_apt = sarr(d.ADEP_mvt)
    d_rwy = d_apt + "|" + sarr(d.RUNWAY_mvt)
    d_stand = d_apt + "|" + sarr(d.STAND_mvt)
    g_apt, g_rwy, g_stand = _groups(d_apt), _groups(d_rwy), _groups(d_stand)
    assert len(g_rwy) > len(g_apt) and len(g_stand) > len(g_apt), (
        f"composite keys collapsed: {len(g_apt)} airports, {len(g_rwy)} runways, "
        f"{len(g_stand)} stands - separator?")

    W = QUEUE_WINDOWS
    out = {c: np.zeros(n) for c in QUEUE_FEATS if c not in QUEUE_VALUE_FEATS}
    out.update({c: np.full(n, np.nan) for c in QUEUE_VALUE_FEATS})

    # ---- other departures at my airport ----
    for j in g_apt.values():
        aj, tj = a[j], tk[j]                        # tj ascending: d is in take-off order
        assert np.all(np.diff(tj) >= 0), "reference stream is not in take-off order"
        a_s = np.sort(aj)
        out["q_apt_at_push"][j] = _on_ground(aj, tj, aj)
        out["q_pushed_after_me"][j] = _count_open(a_s, aj, tj)
        out["q_dep_tko_sym15"][j] = (_count_le(tj, tj + W["tko_sym_apt"])
                                     - _count_lt(tj, tj - W["tko_sym_apt"]) - 1)
    # ---- other departures on my runway ----
    for j in g_rwy.values():
        aj, tj = a[j], tk[j]
        a_s = np.sort(aj)
        out["q_rwy_at_push"][j] = _on_ground(aj, tj, aj)
        out["q_rwy_tko_in_taxi"][j] = _count_open(tj, aj, tj)
        out["q_rwy_push_pre20"][j] = _count_lt(a_s, aj) - _count_lt(a_s, aj - W["push_pre"])
        out["q_rwy_tko_sym10"][j] = (_count_le(tj, tj + W["tko_sym_rwy"])
                                     - _count_lt(tj, tj - W["tko_sym_rwy"]) - 1)
        out["q_rwy_ambient_proxy"][j] = _excluded_median(tj, px[j], W["ambient"])

    # ---- arrivals at my airport ----
    land, onb = es(arr.MVT_TIME_UTC_mvt), es(arr.BLOCK_TIME_UTC_mvt)
    taxi_in = arr.TAXITIME_SEC_mvt.astype(float).to_numpy()
    a_apt = sarr(arr.ADES_mvt)
    g_arr = _groups(a_apt)
    for k, j in g_apt.items():
        pos = g_arr.get(k)
        if pos is None:
            continue                                # counts stay 0, levels stay NaN
        so = np.argsort(onb[pos], kind="mergesort")
        onb_s, taxi_s = onb[pos][so], taxi_in[pos][so]
        out["q_arr_taxiing_at_push"][j] = _on_ground(land[pos], onb[pos], a[j])
        out["q_arr_taxiin_sym30_push"][j] = _window_mean(onb_s, taxi_s, a[j], W["arr_sym"])
        out["q_arr_taxiin_sym30_tko"][j] = _window_mean(onb_s, taxi_s, tk[j], W["arr_sym"])
    # ---- arrivals at my stand: the first on-block after my push, within 24 h ----
    ok = arr.STAND_mvt.notna().to_numpy()
    g_arr_stand = _groups((a_apt + "|" + sarr(arr.STAND_mvt))[ok])
    onb_ok = onb[ok]
    for k, j in g_stand.items():
        pos = g_arr_stand.get(k)
        if pos is None:
            continue
        b_s = np.sort(onb_ok[pos])
        p = _count_le(b_s, a[j])                    # index of the first on-block > a_i
        hit = p < len(b_s)
        nxt = np.where(hit, b_s[np.minimum(p, len(b_s) - 1)], np.nan)
        hit &= (nxt - a[j]) <= W["next_arr"]
        out["q_next_arr_onblock_gap"][j] = np.where(hit, nxt - tk[j], np.nan)

    o = pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt.to_numpy()})
    for c in QUEUE_FEATS:
        o[c] = out[c].astype(np.float32)
    return o


def build_queue_month(path: pathlib.Path, serve: bool = False) -> pd.DataFrame:
    """Queue features for one calendar-month file (the training caches are one month each)."""
    return build_queue(pq.read_table(path, columns=QCOLS).to_pandas(), serve=serve)


def build_queue_ranking(path: pathlib.Path) -> pd.DataFrame:
    """Serve-mode queue features for the evaluation file, one calendar month at a time.

    Every training cache is one calendar month, so no training window and no 24 h stand
    look-ahead ever crosses a month boundary; the evaluation file holds January AND July and is
    split on the movement month exactly as build_ranking splits it.
    """
    t = pq.read_table(path, columns=QCOLS).to_pandas()
    if t.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError(f"{t.MVT_TIME_UTC_mvt.isna().sum()} rows without MVT_TIME cannot "
                         "be assigned to a calendar month")
    ym = (t.MVT_TIME_UTC_mvt.dt.year * 100 + t.MVT_TIME_UTC_mvt.dt.month).to_numpy()
    parts = [build_queue(t[ym == m], serve=True) for m in np.unique(ym)]
    o = pd.concat(parts, ignore_index=True)
    assert o.MVT_ID_mvt.is_unique, "duplicate MVT_ID across the per-month builds"
    return o


def _peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def cmd_queue_cache(smoke: bool, ranking: bool = False):
    """Write data/cache_queue/<raw stem>.parquet: MVT_ID_mvt + QUEUE_FEATS, one file per calendar
    month, joined onto the stand caches by MVT_ID (or positionally: same rows, same order)."""
    QCACHE.mkdir(parents=True, exist_ok=True)
    if ranking:
        if smoke:
            raise ValueError("--smoke does not apply to --ranking: a partial ranking cache "
                             "at the real path would be consumed by the full fold")
        t0 = time.time()
        o = build_queue_ranking(RAW / "ranking.parquet")
        out = QCACHE / "ranking.parquet"
        o.to_parquet(out, index=False)
        print(f"ranking.parquet -> {out.name}  rows={len(o):,}  cols={o.shape[1]}  "
              f"wall {time.time() - t0:.1f}s  peak RSS {_peak_rss_gb():.2f} GB", flush=True)
        return
    paths = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
    if smoke:
        paths = paths[:1]
    for p in paths:
        p = pathlib.Path(p)
        t0 = time.time()
        o = build_queue_month(p)
        out = QCACHE / (p.stem + ".parquet")
        o.to_parquet(out, index=False)
        print(f"{p.name} -> {out.name}  rows={len(o):,}  cols={o.shape[1]}  "
              f"wall {time.time() - t0:.1f}s  peak RSS {_peak_rss_gb():.2f} GB", flush=True)
        del o
        gc.collect()


# =============================================================================================
# Amendment 19 arm D: the airport-day regime block (DAY_FEATS)
# =============================================================================================

DCACHE = ROOT / "data" / "cache_day"
#: the raw columns the day block reads - a subset of QCOLS (it never keys on STAND or RUNWAY).
#: As in the queue block, a departure's own BLOCK_TIME and TAXITIME appear only in the
#: training-mode row filter, never in a feature; both are blank on every scored row.
DCOLS = ["PHASE_mvt", "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
         "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"]
#: per (airport, UTC calendar day of the row's OWN MVT_TIME), computed over the SAME file the row
#: lives in - transductive, which Amendment 19.1 permits - and WITHOUT self-exclusion: the row
#: is one of hundreds in its airport-day, a day percentile is not moved by it beyond the
#: percentile's own resolution, and the scored file is built the same way, so excluding the row
#: would buy nothing and cost a per-row recomputation. Departures are the reference stream
#: build_features uses (same filter, same take-off order); arrivals are the same arrival stream
#: (ARR with an on-block at the ten airports), dated by their landing MVT_TIME.
#:   d_prx_p50, d_prx_p90   percentiles (numpy linear interpolation) of proxy = MVT_TIME - AOBT_3
#:                          over the day's departures at the airport
#:   d_prx_le0              share of those departures with proxy <= 0
#:   d_arr_p50, d_arr_p90   percentiles of arrival taxi-in (TAXITIME) over the day's arrivals at
#:                          the airport that carry a finite taxi-in; NaN when none
#:   d_arr_long             share of those arrivals with taxi-in > 1,200 s (strict); NaN when none
#:   d_n_dep, d_n_arr       the day's departure count and arrival count (a NaN taxi-in counts)
#:   d_prx_minus_p50        the row's own proxy - d_prx_p50
DAY_FEATS = ["d_prx_p50", "d_prx_p90", "d_prx_le0", "d_arr_p50", "d_arr_p90", "d_arr_long",
             "d_n_dep", "d_n_arr", "d_prx_minus_p50"]
#: NaN together on an airport-day without an arrival taxi-in; every other day feature is finite
DAY_ARR_FEATS = ["d_arr_p50", "d_arr_p90", "d_arr_long"]
DAY_ARR_LONG_S = 1_200.0
DAY_PERCENTILES = (50, 90)
DAY_S = 86_400.0


def _day_index(seconds) -> np.ndarray:
    """The UTC calendar day of a clock as whole days since EPOCH (a midnight UTC): floor
    division of the seconds, so 23:59:59 and the following 00:00:00 are one day apart."""
    return np.floor_divide(np.asarray(seconds, dtype="float64"), DAY_S).astype(np.int64)


def _day_groups(apt, day) -> dict:
    """{(airport, day): row positions}, grouped on TWO key arrays rather than a joined string,
    so no separator can collapse the composite key (the historical NUL bug)."""
    return pd.DataFrame({"a": np.asarray(apt), "d": np.asarray(day)}).groupby(["a", "d"], sort=False).indices


def _day_stats(d_apt, d_day, px, a_apt, a_day, taxi_in) -> tuple:
    """Per (airport, UTC day): the departure statistics {key: (proxy p50, p90, share <= 0,
    count)} over the matched reference stream, and the arrival statistics {key: (taxi-in p50,
    p90, share > 1,200 s, count)} - the three levels NaN when no arrival of the day carries a
    finite taxi-in. Shared by build_day (rows in the stream) and build_unmatched (rows outside
    it, looked up by their own airport-day)."""
    dep, dep_groups = {}, _day_groups(d_apt, d_day)
    for key, idx in dep_groups.items():
        v = px[idx]
        p50, p90 = np.percentile(v, DAY_PERCENTILES)
        dep[key] = (p50, p90, float((v <= 0).mean()), len(idx))
    arr = {}
    if len(a_apt):
        for key, pos in _day_groups(a_apt, a_day).items():
            tx = taxi_in[pos]
            tx = tx[np.isfinite(tx)]
            if len(tx):
                a50, a90 = np.percentile(tx, DAY_PERCENTILES)
                arr[key] = (a50, a90, float((tx > DAY_ARR_LONG_S).mean()), len(pos))
            else:
                arr[key] = (np.nan, np.nan, np.nan, len(pos))
    return dep, arr, dep_groups


def build_day(t: pd.DataFrame, serve: bool = False) -> pd.DataFrame:
    """`MVT_ID_mvt` + DAY_FEATS (float32) for exactly the rows build_features(t, serve) returns,
    in the same take-off order.

    Both streams are build_features's own (via _dep_stream / _arr_stream), never a different
    filter. The day of a departure is the UTC date of its MVT_TIME; the day of an arrival is the
    UTC date of its landing MVT_TIME. Counts are 0 when nothing qualifies; the three arrival
    levels are NaN when the airport-day has no arrival with a finite taxi-in. Nothing reads a
    departure's own BLOCK or TAXITIME (guarded by the hidden-clock tests).
    """
    d, arr = _dep_stream(t, serve), _arr_stream(t)
    n = len(d)
    if n == 0:
        raise ValueError(f"no admissible departures (serve={serve}) - nothing to build")
    a, tk = es(d.AOBT_3_flt), es(d.MVT_TIME_UTC_mvt)
    px = tk - a
    d_apt, d_day = sarr(d.ADEP_mvt), _day_index(tk)
    a_apt, a_day = sarr(arr.ADES_mvt), _day_index(es(arr.MVT_TIME_UTC_mvt))
    taxi_in = arr.TAXITIME_SEC_mvt.astype(float).to_numpy()
    dep_s, arr_s, groups = _day_stats(d_apt, d_day, px, a_apt, a_day, taxi_in)
    out = {c: np.full(n, np.nan) for c in DAY_FEATS}
    for key, idx in groups.items():
        p50, p90, le0, n_dep = dep_s[key]
        out["d_prx_p50"][idx] = p50
        out["d_prx_p90"][idx] = p90
        out["d_prx_le0"][idx] = le0
        out["d_n_dep"][idx] = n_dep
        out["d_prx_minus_p50"][idx] = px[idx] - p50
        a50, a90, long, n_arr = arr_s.get(key, (np.nan, np.nan, np.nan, 0))
        out["d_arr_p50"][idx] = a50
        out["d_arr_p90"][idx] = a90
        out["d_arr_long"][idx] = long
        out["d_n_arr"][idx] = n_arr
    o = pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt.to_numpy()})
    for c in DAY_FEATS:
        o[c] = out[c].astype(np.float32)
    return o


def build_day_month(path: pathlib.Path, serve: bool = False) -> pd.DataFrame:
    """Day features for one calendar-month file (the training caches are one month each)."""
    return build_day(pq.read_table(path, columns=DCOLS).to_pandas(), serve=serve)


def build_day_ranking(path: pathlib.Path) -> pd.DataFrame:
    """Serve-mode day features for the evaluation file, one calendar month at a time.

    A UTC day never straddles two calendar months, so the per-month build equals the single-unit
    one; it is split anyway so every cache is built the same way as the training files and the
    MVT_ID uniqueness across the months is asserted, exactly as build_queue_ranking does.
    """
    t = pq.read_table(path, columns=DCOLS).to_pandas()
    if t.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError(f"{t.MVT_TIME_UTC_mvt.isna().sum()} rows without MVT_TIME cannot "
                         "be assigned to a calendar month")
    ym = (t.MVT_TIME_UTC_mvt.dt.year * 100 + t.MVT_TIME_UTC_mvt.dt.month).to_numpy()
    parts = [build_day(t[ym == m], serve=True) for m in np.unique(ym)]
    o = pd.concat(parts, ignore_index=True)
    assert o.MVT_ID_mvt.is_unique, "duplicate MVT_ID across the per-month builds"
    return o


def cmd_day_cache(smoke: bool, ranking: bool = False):
    """Write data/cache_day/<raw stem>.parquet: MVT_ID_mvt + DAY_FEATS, one file per calendar
    month, joined onto the stand caches by MVT_ID (or positionally: same rows, same order)."""
    DCACHE.mkdir(parents=True, exist_ok=True)
    if ranking:
        if smoke:
            raise ValueError("--smoke does not apply to --ranking: a partial ranking cache "
                             "at the real path would be consumed by the full fold")
        t0 = time.time()
        o = build_day_ranking(RAW / "ranking.parquet")
        out = DCACHE / "ranking.parquet"
        o.to_parquet(out, index=False)
        print(f"ranking.parquet -> {out.name}  rows={len(o):,}  cols={o.shape[1]}  "
              f"wall {time.time() - t0:.1f}s  peak RSS {_peak_rss_gb():.2f} GB", flush=True)
        return
    paths = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
    if smoke:
        paths = paths[:1]
    for p in paths:
        p = pathlib.Path(p)
        t0 = time.time()
        o = build_day_month(p)
        out = DCACHE / (p.stem + ".parquet")
        o.to_parquet(out, index=False)
        print(f"{p.name} -> {out.name}  rows={len(o):,}  cols={o.shape[1]}  "
              f"wall {time.time() - t0:.1f}s  peak RSS {_peak_rss_gb():.2f} GB", flush=True)
        del o
        gc.collect()


# =============================================================================================
# The unified all-rows arm: the UNMATCHED departures' features (data/cache_unmatched/)
# =============================================================================================

# =============================================================================================
# Amendment 22: the record-ordering block (Arm F) - data/cache_order/
# =============================================================================================
OCACHE = ROOT / "data" / "cache_order"
#: the raw columns the order block reads: the day block's plus FLIGHT_ID. A departure's own
#: BLOCK_TIME and TAXITIME appear only in the training-mode row filter, never in a feature.
OCOLS = ["PHASE_mvt", "MVT_ID_mvt", "FLIGHT_ID_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt",
         "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"]
#: per airport and per file, over EVERY departure at the airport (matched or not, labelled or
#: not - the same reference stream in both modes, so a row's values never depend on the mode;
#: RESULT 11.5 found the signal on this line, Amendment 22 registers it), rows sorted by
#: MVT_TIME then MVT_ID:
#:   o_dev_mvt   (MVT_ID - the rolling median of the +-250 neighbouring rows) / max(rolling p90 - p10, 1)
#:   o_dev_flt   the same for FLIGHT_ID; NaN where FLIGHT_ID is null (a null id leaves the
#:               window's statistics and gets no deviation)
#:   o_n_line    the window's row count, ORDER_MIN_ROWS..ORDER_WINDOW (NaN deviations below it)
ORDER_FEATS = ["o_dev_mvt", "o_dev_flt", "o_n_line"]
ORDER_WINDOW = 501
ORDER_MIN_ROWS = 50
ORDER_QUANTILES = (0.10, 0.90)


def _order_line(ids: np.ndarray) -> tuple:
    """(deviation, window row count) of every id from the rolling line of its neighbours, in
    the given (time) order: pandas' centred window of ORDER_WINDOW rows with at least
    ORDER_MIN_ROWS non-null ids, else NaN. NaN ids are skipped by the statistics and get NaN."""
    s = pd.Series(np.asarray(ids, dtype="float64"))
    r = s.rolling(ORDER_WINDOW, center=True, min_periods=ORDER_MIN_ROWS)
    med = r.median().to_numpy()
    spread = (r.quantile(ORDER_QUANTILES[1]) - r.quantile(ORDER_QUANTILES[0])).to_numpy()
    dev = (s.to_numpy() - med) / np.maximum(spread, 1.0)
    n = pd.Series(np.ones(len(s))).rolling(ORDER_WINDOW, center=True, min_periods=1).sum().to_numpy()
    return dev, n


def _order_stream(t: pd.DataFrame) -> pd.DataFrame:
    """The reference line's rows in both modes: every departure at the ten airports, sorted by
    take-off time then MVT_ID. A departure without a take-off time or an id cannot sit on the
    line and is refused rather than placed at an end (none exists in the challenge files)."""
    d = t[(t.PHASE_mvt == "DEP") & t.ADEP_mvt.isin(APTS)]
    if d.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError(f"{int(d.MVT_TIME_UTC_mvt.isna().sum())} departures without MVT_TIME cannot sit on the id line")
    if d.MVT_ID_mvt.isna().any():
        raise ValueError(f"{int(d.MVT_ID_mvt.isna().sum())} departures without MVT_ID cannot sit on the id line")
    return d.sort_values(["MVT_TIME_UTC_mvt", "MVT_ID_mvt"], kind="mergesort").reset_index(drop=True)


def build_order(t: pd.DataFrame, serve: bool = False) -> pd.DataFrame:
    """`MVT_ID_mvt` + ORDER_FEATS (float32) for exactly the rows build_features(t, serve) returns,
    in the same take-off order; the line itself is per airport over _order_stream (every
    departure), so a row's values are identical in training and serve mode."""
    d = _dep_stream(t, serve)
    if len(d) == 0:
        raise ValueError(f"no admissible departures (serve={serve}) - nothing to build")
    ref = _order_stream(t)
    apt = ref.ADEP_mvt.to_numpy()
    mvt_id, flt_id = ref.MVT_ID_mvt.to_numpy(dtype="float64"), ref.FLIGHT_ID_mvt.to_numpy(dtype="float64")
    dev_m, dev_f, n_line = (np.full(len(ref), np.nan) for _ in range(3))
    for a in np.unique(apt):
        pos = np.flatnonzero(apt == a)
        dev_m[pos], n_line[pos] = _order_line(mvt_id[pos])
        dev_f[pos], _ = _order_line(flt_id[pos])
    line = pd.DataFrame({"o_dev_mvt": dev_m, "o_dev_flt": dev_f, "o_n_line": n_line}, index=mvt_id)
    assert line.index.is_unique, "duplicate MVT_ID on the id line"
    vals = line.loc[d.MVT_ID_mvt.to_numpy(dtype="float64")]
    o = pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt.to_numpy()})
    for c in ORDER_FEATS:
        o[c] = vals[c].to_numpy().astype(np.float32)
    return o


def build_order_month(path: pathlib.Path, serve: bool = False) -> pd.DataFrame:
    """Order features for one calendar-month file (the training caches are one month each)."""
    return build_order(pq.read_table(path, columns=OCOLS).to_pandas(), serve=serve)


def build_order_ranking(path: pathlib.Path) -> pd.DataFrame:
    """Serve-mode order features for the evaluation file, one calendar month at a time - the
    line is per file in training, so the scored months are lined separately too; MVT_ID
    uniqueness across the months asserted, as build_day_ranking does."""
    t = pq.read_table(path, columns=OCOLS).to_pandas()
    if t.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError(f"{t.MVT_TIME_UTC_mvt.isna().sum()} rows without MVT_TIME cannot "
                         "be assigned to a calendar month")
    ym = (t.MVT_TIME_UTC_mvt.dt.year * 100 + t.MVT_TIME_UTC_mvt.dt.month).to_numpy()
    parts = [build_order(t[ym == m], serve=True) for m in np.unique(ym)]
    o = pd.concat(parts, ignore_index=True)
    assert o.MVT_ID_mvt.is_unique, "duplicate MVT_ID across the per-month builds"
    return o


def cmd_order_cache(smoke: bool, ranking: bool = False):
    """Write data/cache_order/<raw stem>.parquet: MVT_ID_mvt + ORDER_FEATS, one file per calendar
    month, joined onto the stand caches by MVT_ID (or positionally: same rows, same order)."""
    OCACHE.mkdir(parents=True, exist_ok=True)
    if ranking:
        if smoke:
            raise ValueError("--smoke does not apply to --ranking: a partial ranking cache "
                             "at the real path would be consumed by the full fold")
        t0 = time.time()
        o = build_order_ranking(RAW / "ranking.parquet")
        out = OCACHE / "ranking.parquet"
        o.to_parquet(out, index=False)
        print(f"ranking.parquet -> {out.name}  rows={len(o):,}  cols={o.shape[1]}  "
              f"wall {time.time() - t0:.1f}s  peak RSS {_peak_rss_gb():.2f} GB", flush=True)
        return
    paths = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
    if smoke:
        paths = paths[:1]
    for p in paths:
        p = pathlib.Path(p)
        t0 = time.time()
        o = build_order_month(p)
        out = OCACHE / (p.stem + ".parquet")
        o.to_parquet(out, index=False)
        print(f"{p.name} -> {out.name}  rows={len(o):,}  cols={o.shape[1]}  "
              f"wall {time.time() - t0:.1f}s  peak RSS {_peak_rss_gb():.2f} GB", flush=True)
        del o
        gc.collect()


UCACHE = ROOT / "data" / "cache_unmatched"
#: the reference columns the unmatched builder needs of the OTHER movements (the matched
#: departures and the arrivals): QCOLS plus the arrivals' ARVT_3 (cdiff). The unmatched rows
#: themselves are read with the full COLS.
RCOLS = QCOLS + ["ARVT_3_flt"]
#: NaN by construction on every unmatched row: each reads the row's own AOBT_3, directly or as
#: the anchor of a window. Asserted after the build, never filled.
UNMATCHED_NAN_COLS = ["proxy", "aobt_sched", "aobt_eobt", "aobt_sec",
                      "q_apt_at_push", "q_rwy_at_push", "q_pushed_after_me", "q_rwy_tko_in_taxi",
                      "q_rwy_push_pre20", "q_arr_taxiing_at_push", "q_arr_taxiin_sym30_push",
                      "q_next_arr_onblock_gap", "d_prx_minus_p50"]
#: NaN whenever the row's *_flt clocks are null - on the data, every unmatched row (Amendment 3
#: section 2; measured 2026-09-09: EOBT null on 1,408 / 1,408 January and 5,290 / 5,290 scored
#: unmatched rows). Not forced: the arithmetic yields it, and the tests assert it.
UNMATCHED_FLT_COLS = ["eobt_p", "lobt_p", "iobt_p", "eobt_sched", "lobt_sched", "eobt_lobt", "iobt_lobt",
                      "airborne3", "airborne1", "arvt_diff"]
#: the queue features an unmatched row CAN carry: anchored on its own take-off, counting the
#: matched reference stream (the row itself is not in it, so no self-exclusion applies)
UNMATCHED_QUEUE_FEATS = ["q_dep_tko_sym15", "q_rwy_tko_sym10", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy"]
UNMATCHED_KEY_COLS = ["STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt",
                      "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "ADEP_mvt", "stand_pref", "airline", "ars"]
#: the unmatched cache's contract: id, labels, every BASELINE feature except the encodings
#: (fitted in-fold), month, ap, the encoding keys, the queue block, the day block - one file per
#: month, no twins (the rows are few: ~1.8k of ~170k per month)
UNMATCHED_COLS = (["MVT_ID_mvt", "delta", "y"] + BASE + SURF + STANDHIST + ["month", "ap"] + UNMATCHED_KEY_COLS
                  + QUEUE_FEATS + DAY_FEATS)
UNMATCHED_FLOAT32 = BASE + SURF + STANDHIST + QUEUE_FEATS + DAY_FEATS


def _unmatched_stream(t: pd.DataFrame, serve: bool) -> pd.DataFrame:
    """The stratum's row set: departures without an NM off-block at the ten airports with the
    clocks the features read (MVT_TIME, SCHED); in training mode labelled exactly as
    build_submission.admissible requires, one row per MVT_ID by its keep-first rule (sort by
    BLOCK, MVT_TIME, TAXITIME, MVT_ID, mergesort); in serve mode every such row. Take-off order."""
    u = t[(t.PHASE_mvt == "DEP") & t.AOBT_3_flt.isna() & t.MVT_TIME_UTC_mvt.notna() & t.SCHED_TIME_UTC_mvt.notna()]
    u = u[u.ADEP_mvt.isin(APTS)]
    if not serve:
        u = u[u.TAXITIME_SEC_mvt.notna() & u.BLOCK_TIME_UTC_mvt.notna() & (u.TAXITIME_SEC_mvt > 0)]
        u = u.sort_values(["BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "MVT_ID_mvt"], kind="mergesort")
        u = u[~u.MVT_ID_mvt.duplicated(keep="first")]
    return u.sort_values("MVT_TIME_UTC_mvt", kind="mergesort").reset_index(drop=True)


def _trailing_median(sorted_t, vals, x, window: int, min_periods: int) -> np.ndarray:
    """The median of the last `window` values (aligned with sorted_t) at times <= x, NaN below
    `min_periods`: pandas' rolling(window, min_periods).median() evaluated at the position x
    would take in the stream - the matched builder's dep_dur / rwy_dur for a row outside it."""
    hi = _count_le(sorted_t, x)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo = max(0, hi[i] - window)
        if hi[i] - lo >= min_periods:
            out[i] = np.median(vals[lo:hi[i]])
    return out


def _window_median(sorted_t, vals, x, w) -> np.ndarray:
    """Median of vals over rows with sorted_t in [x - w, x + w]; NaN where the window is empty."""
    lo, hi = _count_lt(sorted_t, x - w), _count_le(sorted_t, x + w)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if hi[i] > lo[i]:
            out[i] = np.median(vals[lo[i]:hi[i]])
    return out


def build_unmatched(t: pd.DataFrame, serve: bool = False, u: pd.DataFrame | None = None) -> pd.DataFrame:
    """UNMATCHED_COLS for the admissible unmatched departures of `t` (or of `u`, when the
    unmatched rows were read separately with COLS and `t` holds the RCOLS of every movement),
    in take-off order.

    The reference streams are build_features's own - the matched departures (_dep_stream, the
    same filter and take-off order as the caches) and the arrivals (_arr_stream) - and every
    window is the matched builder's formula anchored on the unmatched row's OWN take-off (and
    schedule) against those streams; the row is never in a stream, so nothing is self-excluded
    and a window counts every matched neighbour (tests/test_unmatched_features.py pins the
    parity on cloned rows). Everything that reads the row's own AOBT_3 - proxy, the three
    off-block offsets, aobt_sec, the eight push-anchored queue features, d_prx_minus_p50 - is
    NaN by construction (UNMATCHED_NAN_COLS, asserted); everything that reads the row's *_flt
    clocks is NaN whenever they are null (every unmatched row of the data). The encoding keys
    of null *_flt columns read "NA", as build_features's do.
    """
    d, arr = _dep_stream(t, serve), _arr_stream(t)
    if u is None:
        u = _unmatched_stream(t, serve)
    else:
        u = _unmatched_stream(u, serve)
    n = len(u)
    if n == 0:
        raise ValueError(f"no admissible unmatched departures (serve={serve}) - nothing to build")
    if serve and not u.MVT_ID_mvt.is_unique:
        raise ValueError(f"duplicate MVT_ID among the unmatched departures: {int(u.MVT_ID_mvt.duplicated().sum())}")
    o, tu, su, _ = _own_clock_features(u, serve)
    for name in SURF + STANDHIST + QUEUE_FEATS + DAY_FEATS:
        o[name] = np.nan
    W = QUEUE_WINDOWS

    # ---- the matched reference stream ----
    tk, a, sch = es(d.MVT_TIME_UTC_mvt), es(d.AOBT_3_flt), es(d.SCHED_TIME_UTC_mvt)
    px = tk - a
    d_apt = sarr(d.ADEP_mvt)
    d_rwy = d_apt + "|" + sarr(d.RUNWAY_mvt)
    d_stand = d_apt + "|" + sarr(d.STAND_mvt)
    g_apt, g_rwy, g_stand = (_groups(k) if len(d) else {} for k in (d_apt, d_rwy, d_stand))
    if len(d):
        assert len(g_rwy) >= len(g_apt) and len(g_stand) >= len(g_apt), "composite keys collapsed - separator?"
    u_apt = sarr(u.ADEP_mvt)
    u_rwy = u_apt + "|" + sarr(u.RUNWAY_mvt)
    u_stand = u_apt + "|" + sarr(u.STAND_mvt)

    # ---- the arrival stream: landing, on-block, taxi-in, cdiff, stand ----
    a_apt = sarr(arr.ADES_mvt)
    a_land, a_t = es(arr.MVT_TIME_UTC_mvt), es(arr.BLOCK_TIME_UTC_mvt)
    a_taxi = arr.TAXITIME_SEC_mvt.astype(float).to_numpy()
    a_cdiff = np.where(arr.ARVT_3_flt.notna().to_numpy(), a_t - es(arr.ARVT_3_flt), np.nan)
    a_stand = sarr(arr.STAND_mvt)
    a_ok = arr.STAND_mvt.notna().to_numpy()
    a_key = a_apt + "|" + a_stand

    # ---- airport windows anchored on the row's take-off ----
    for k, rows in pd.Series(np.arange(n)).groupby(u_apt, sort=False):
        q = rows.to_numpy()
        idx, tq = o.index[q], tu[q]
        j = g_apt.get(k)
        if j is not None:
            m_sorted = tk[j]
            o.loc[idx, "apt_b30"] = _win_count(tq, m_sorted, -1800, 0)
            o.loc[idx, "n_push"] = _win_count(tq, np.sort(a[j]), -1800, 0)
            s_sorted = np.sort(sch[j])
            o.loc[idx, "sched_prev60"] = _win_count(tq, s_sorted, -3600, 0)
            o.loc[idx, "sched_next60"] = _win_count(tq, s_sorted, 0, 3600)
            o.loc[idx, "sched_day"] = len(j)
            o.loc[idx, "dep_dur"] = _trailing_median(m_sorted, px[j], tq, 30, 3)
            o.loc[idx, "q_dep_tko_sym15"] = (_count_le(m_sorted, tq + W["tko_sym_apt"])
                                             - _count_lt(m_sorted, tq - W["tko_sym_apt"]))
        am = a_apt == k
        if am.any():
            at, ax, ac = a_t[am], a_taxi[am], a_cdiff[am]
            so = np.argsort(at, kind="mergesort")
            at, ax, ac = at[so], ax[so], ac[so]
            o.loc[idx, "arr_b30"] = _win_count(tq, at, -1800, 0)
            hi = np.searchsorted(at, tq, "right")
            lo60 = np.searchsorted(at, tq - 3600, "right")
            lo180 = np.searchsorted(at, tq - 10800, "right")
            cx, cc = np.concatenate([[0.0], np.nancumsum(ax)]), np.concatenate([[0.0], np.nancumsum(np.nan_to_num(ac))])
            n60 = np.maximum(hi - lo60, 1)
            o.loc[idx, "arr_dur"] = (cx[hi] - cx[lo60]) / n60
            o.loc[idx, "arr_ob60_taxiin"] = (cx[hi] - cx[lo60]) / n60
            o.loc[idx, "arr_ob60_cdiff"] = (cc[hi] - cc[lo60]) / n60
            o.loc[idx, "arr_ob180_cdiff"] = (cc[hi] - cc[lo180]) / np.maximum(hi - lo180, 1)
            o.loc[idx, "q_arr_taxiin_sym30_tko"] = _window_mean(at, ax, tq, W["arr_sym"])
    # ---- runway windows ----
    for k, rows in pd.Series(np.arange(n)).groupby(u_rwy, sort=False):
        q = rows.to_numpy()
        idx, tq = o.index[q], tu[q]
        j = g_rwy.get(k)
        if j is None:
            continue
        m_sorted = tk[j]
        o.loc[idx, "rwy_b30"] = _win_count(tq, m_sorted, -1800, 0)
        o.loc[idx, "rwy_rate"] = _win_count(tq, m_sorted, -3600, 0) / 60.0
        o.loc[idx, "rwy_dur"] = _trailing_median(m_sorted, px[j], tq, 20, 3)
        o.loc[idx, "q_rwy_tko_sym10"] = (_count_le(m_sorted, tq + W["tko_sym_rwy"])
                                         - _count_lt(m_sorted, tq - W["tko_sym_rwy"]))
        o.loc[idx, "q_rwy_ambient_proxy"] = _window_median(m_sorted, px[j], tq, W["ambient"])
    # ---- the stand chain, anchored on the row's take-off ----
    ok = a_ok & pd.notna(a_stand)
    ka, ta, tx, cd = a_key[ok], a_t[ok], a_taxi[ok], a_cdiff[ok]
    so = np.argsort(ta, kind="mergesort")
    ka, ta, tx, cd = ka[so], ta[so], tx[so], cd[so]
    aidx = {k: g.to_numpy() for k, g in pd.Series(np.arange(len(ta))).groupby(ka, sort=False)}
    assert len(aidx) == len(set(zip(a_apt[ok], a_stand[ok]))), "stand index collapsed - separator?"
    for k, rows in pd.Series(np.arange(n)).groupby(u_stand, sort=False):
        q = rows.to_numpy()
        idx, tq = o.index[q], tu[q]
        j = g_stand.get(k)
        if j is not None:
            dj = np.sort(a[j])
            o.loc[idx, "prev_dep_gap"] = tq - dj[np.maximum(np.searchsorted(dj, tq, "right") - 1, 0)]
        pos = aidx.get(k)
        if pos is None:
            continue
        st, sx, sc = ta[pos], tx[pos], cd[pos]
        hm = np.searchsorted(st, tq, "right") - 1
        okm = hm >= 0
        o.loc[idx, "prev_arr_gap"] = tq - np.where(okm, st[np.maximum(hm, 0)], np.nan)
        o.loc[idx, "prev_arr_taxiin"] = np.where(okm, sx[np.maximum(hm, 0)], np.nan)
        o.loc[idx, "prev_arr_cdiff"] = np.where(okm, sc[np.maximum(hm, 0)], np.nan)
    # ---- the day block: the row's own airport-day, looked up in the reference statistics ----
    dep_s, arr_s, _ = _day_stats(d_apt, _day_index(tk), px, a_apt, _day_index(a_land), a_taxi)
    u_day = _day_index(tu)
    for i in range(n):
        key = (u_apt[i], u_day[i])
        p50, p90, le0, n_dep = dep_s.get(key, (np.nan, np.nan, np.nan, 0))
        a50, a90, long, n_arr = arr_s.get(key, (np.nan, np.nan, np.nan, 0))
        o.iloc[i, o.columns.get_loc("d_prx_p50")] = p50
        o.iloc[i, o.columns.get_loc("d_prx_p90")] = p90
        o.iloc[i, o.columns.get_loc("d_prx_le0")] = le0
        o.iloc[i, o.columns.get_loc("d_n_dep")] = n_dep
        o.iloc[i, o.columns.get_loc("d_arr_p50")] = a50
        o.iloc[i, o.columns.get_loc("d_arr_p90")] = a90
        o.iloc[i, o.columns.get_loc("d_arr_long")] = long
        o.iloc[i, o.columns.get_loc("d_n_arr")] = n_arr

    # ---- the NaN pattern: by construction, asserted, never filled ----
    carrying = [c for c in UNMATCHED_NAN_COLS if o[c].notna().any()]
    assert not carrying, f"an AOBT_3-anchored column carries a value on an unmatched row: {carrying}"
    o["month"] = u.MVT_TIME_UTC_mvt.dt.month.to_numpy()
    o["MVT_ID_mvt"] = u.MVT_ID_mvt.to_numpy()
    o["ap"] = u_apt
    for k in ("STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt",
              "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"):
        o[k] = sarr(u[k])
    o["ADEP_mvt"] = u_apt
    o["stand_pref"] = pd.Series(sarr(u.STAND_mvt)).str[:2].to_numpy()
    o["airline"] = pd.Series(sarr(u.FLIGHT_mvt)).str[:3].to_numpy()
    o["ars"] = u_stand + "|" + sarr(u.RUNWAY_mvt)
    out = o[UNMATCHED_COLS].reset_index(drop=True)
    for c in UNMATCHED_FLOAT32:
        out[c] = out[c].astype(np.float32)
    return out


def build_unmatched_month(path: pathlib.Path, serve: bool = False) -> pd.DataFrame:
    """Unmatched features for one calendar-month file, read in two passes to stay small: the
    unmatched departures with the full COLS (a few thousand rows, filtered at the Arrow level),
    then every movement's RCOLS for the reference streams."""
    import pyarrow.compute as pc
    tab = pq.read_table(path, columns=COLS, filters=[("PHASE_mvt", "==", "DEP")])
    u = tab.filter(pc.is_null(tab["AOBT_3_flt"])).to_pandas()
    del tab
    gc.collect()
    return build_unmatched(pq.read_table(path, columns=RCOLS).to_pandas(), serve=serve, u=u)


def build_unmatched_ranking(path: pathlib.Path) -> pd.DataFrame:
    """Serve-mode unmatched features for the evaluation file, one calendar month at a time, as
    every other cache is built; MVT_ID uniqueness across the months asserted."""
    import pyarrow.compute as pc
    tab = pq.read_table(path, columns=COLS, filters=[("PHASE_mvt", "==", "DEP")])
    u = tab.filter(pc.is_null(tab["AOBT_3_flt"])).to_pandas()
    del tab
    gc.collect()
    t = pq.read_table(path, columns=RCOLS).to_pandas()
    if t.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError(f"{t.MVT_TIME_UTC_mvt.isna().sum()} rows without MVT_TIME cannot "
                         "be assigned to a calendar month")
    ym = (t.MVT_TIME_UTC_mvt.dt.year * 100 + t.MVT_TIME_UTC_mvt.dt.month).to_numpy()
    uym = (u.MVT_TIME_UTC_mvt.dt.year * 100 + u.MVT_TIME_UTC_mvt.dt.month).to_numpy()
    parts = [build_unmatched(t[ym == m], serve=True, u=u[uym == m]) for m in np.unique(ym) if (uym == m).any()]
    o = pd.concat(parts, ignore_index=True)
    assert o.MVT_ID_mvt.is_unique, "duplicate MVT_ID across the per-month builds"
    return o


def cmd_unmatched_cache(smoke: bool, ranking: bool = False):
    """Write data/cache_unmatched/<raw stem>.parquet: UNMATCHED_COLS for the unmatched
    departures, one file per calendar month; --ranking writes ranking.parquet (serve mode)."""
    UCACHE.mkdir(parents=True, exist_ok=True)
    if ranking:
        if smoke:
            raise ValueError("--smoke does not apply to --ranking: a partial ranking cache "
                             "at the real path would be consumed by the full fold")
        t0 = time.time()
        o = build_unmatched_ranking(RAW / "ranking.parquet")
        out = UCACHE / "ranking.parquet"
        o.to_parquet(out, index=False)
        print(f"ranking.parquet -> {out.name}  rows={len(o):,}  cols={o.shape[1]}  "
              f"wall {time.time() - t0:.1f}s  peak RSS {_peak_rss_gb():.2f} GB", flush=True)
        return
    paths = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
    if smoke:
        paths = paths[:1]
    for p in paths:
        p = pathlib.Path(p)
        t0 = time.time()
        o = build_unmatched_month(p)
        out = UCACHE / (p.stem + ".parquet")
        o.to_parquet(out, index=False)
        print(f"{p.name} -> {out.name}  rows={len(o):,}  cols={o.shape[1]}  "
              f"wall {time.time() - t0:.1f}s  peak RSS {_peak_rss_gb():.2f} GB", flush=True)
        del o
        gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["cache", "fit", "queue-cache", "day-cache", "unmatched-cache", "order-cache"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ranking", action="store_true",
                    help="cache / queue-cache / day-cache / unmatched-cache / order-cache: build <cache dir>/ranking.parquet "
                         "(serve mode) instead of the training months")
    a = ap.parse_args()
    if a.cmd == "cache":
        cmd_cache(a.smoke, ranking=a.ranking)
    elif a.cmd == "queue-cache":
        cmd_queue_cache(a.smoke, ranking=a.ranking)
    elif a.cmd == "day-cache":
        cmd_day_cache(a.smoke, ranking=a.ranking)
    elif a.cmd == "unmatched-cache":
        cmd_unmatched_cache(a.smoke, ranking=a.ranking)
    elif a.cmd == "order-cache":
        cmd_order_cache(a.smoke, ranking=a.ranking)
    else:
        cmd_fit(a.smoke)
