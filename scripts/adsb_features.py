"""Label-free per-flight ADS-B features for departures (arm ADN, plans/PREREG_adsb_nn_2026_09_10.md).

    build(flights, adsb) -> one row per flight (MVT_ID_mvt + feature columns)

Pipeline, all without BLOCK_TIME / TAXITIME (the build drops them on entry; a test permutes them):
  1. segment_tracks  — per aircraft (icao), a new track after a 30-minute gap, so an inbound and an outbound leg
                       under one callsign are separate tracks (the E4b join defect's root).
  2. takeoff_events  — per track, every last-ground -> first-airborne transition (airborne at >= 60 kt).
  3. join_flights    — take-off-anchored: the flight's airport, callsign (CALLSIGN_flt or FLIGHT_mvt) and a take-off
                       within ±180 s of MVT; else a UNIQUE take-off within ±30 s; one event never serves two flights.
  4. stand_centroids — parked position around AOBT_3 (a serve-time column) of OTHER flights at the same stand; a flight
                       never seen parked contributes its first slow ground samples instead (the FIRST-APPEARANCE
                       fallback). Also a PARKED-only reference and the parked share of the others (provenance).
  5. flight_features — gate dwell, stops, path, speeds, coverage; sensor off-block at the stand centroid;
                       surface congestion (other aircraft moving on the ground).
                       The gate dwell (DWG, BC-26) is the first stationary run >= GATE_DWELL_MIN_S that STARTS at or
                       before AOBT_3 + DWELL_START_MAX_S and, when a PARKED-provenance stand reference exists, whose
                       median lies within GATE_RADIUS_M of it; otherwise every dwell-anchored column is NaN and
                       stand_visible is 0. A first-appearance reference never decides a dwell (it is not the stand).
Times are seconds; features relative to MVT are "seconds before take-off" (positive).
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("adsb_extract", ROOT / "scripts" / "adsb_extract.py")
ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae)

TRACK_GAP_S = 1800.0
TAKEOFF_MIN_GS = 60.0
LANDING_SLOW_GS = 40.0           # a touchdown is followed by taxi speed; a take-off roll never slows back to it
ROLL_MERGE_S = 120.0             # ground->air flips of one track this close together are one roll (flicker)
JOIN_CALLSIGN_S = 180.0
JOIN_TIME_S = 30.0
GROUND_LOOKBACK_S = 3 * 3600.0
STATIONARY_GS = 1.0
GATE_DWELL_MIN_S = 300.0
STOP_MIN_S = 60.0
CENTROID_WINDOW = (-900.0, 60.0)      # parked samples in [AOBT_3 - 900, AOBT_3 + 60]
CENTROID_MIN_OTHERS = 2
GATE_RADIUS_M = 100.0
DWELL_START_MAX_S = CENTROID_WINDOW[1]  # DWG: a gate dwell starts by AOBT_3 + 60 s. Anchor (2025 R12 train table): dwells
#                                       <= 50 m from the stand start later in 2.6 %, dwells > 100 m off in 47 % (BC-26)
MOVING_GS = 3.0
CONGESTION_HALF_S = 300.0
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON_EQ = 111_320.0
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
FLIGHT_COLS = ["MVT_ID_mvt", "ADEP_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "STAND_mvt"]
SLOW_FIRST_GS = 5.0                   # a first ground sample below this is at / leaving the stand, not mid-taxi
FIRST_N = 3                           # first-ground speed = median of the first FIRST_N segment samples
KNOT_MS = 0.514444                    # ADX: gs is in knots; the reconstruction works in metres and seconds
SPAN_MIN_S = 0.001                    # AFJ (plans/PREREG_adsbfi_july_fallback_2026_09_16.md §2): a ground period whose
#                                       samples span < 1 ms is ONE fix recorded twice (both sources timestamp at a 10 ms
#                                       tick): ground_rate_pm is unknown, as for an exact duplicate. The shipped tables'
#                                       smallest span is 0.0099999 s, so no shipped row changes. Two adsb.fi records of
#                                       one fix 2.4e-7 s apart gave 503,316,480 samples/min and an MLP residual of -1.9e6 s
MIN_TAXI_GS = 2.0                     # ADX: below this a flight's own taxi speed is unusable; fall back to the
#                                       airport's median usable speed in the same period (a fallback, not a clamp;
#                                       when the airport has none either, the block stays NaN — BC-3)
FEATURES = ["joined", "join_callsign", "dt_takeoff", "first_seen_rel_mvt", "n_ground", "ground_rate_pm", "mlat_share",
            "first_ground_rel_mvt", "first_ground_gs", "first_ground_slow", "first_move_rel_mvt",
            "first_ground_rel_aobt3", "turn_in_track",
            "dwell_end_rel_mvt", "dwell_s", "dwell_end_rel_aobt3", "n_stops", "stop_s", "path_m", "taxi_med_gs",
            "sensor_rel_mvt", "sensor_rel_aobt3", "dwell_to_centroid_m", "moving_mvt_m10", "moving_dwell_end",
            "ground_seen_mvt_m10",
            # arm ADX (plans/PREREG_adsb_taxistart_2026_09_11.md): the unobserved start of taxi, reconstructed
            "first_ground_to_stand_m", "stand_others_n", "travel_s_first_to_stand", "implied_offblock_rel_aobt3",
            "implied_taxi_s",
            # arm ADJ (plans/PREREG_adsb_join_recall_2026_09_12.md): the second, MVT-anchored join path
            "join_mvt_anchored", "mvt_anchor_gap_s",
            # DWG (BC-26, 2026-09-27): 1 iff a gate dwell was found (joined rows; NaN unjoined), and the share of the
            # stand reference's other flights that were seen PARKED (vs the first-appearance fallback; NaN: no reference)
            "stand_visible", "stand_ref_parked_share"]
#: the ADX block alone (the twin in arm ADX is FEATURES minus this)
ADX_BLOCK = ["first_ground_to_stand_m", "stand_others_n", "travel_s_first_to_stand", "implied_offblock_rel_aobt3",
             "implied_taxi_s"]


def _ground_segment(s: pd.DataFrame, t_takeoff: float) -> tuple[pd.DataFrame, int]:
    """(on-ground samples of the departure's final ground period before take-off, turn flag). The period starts after
    the last LANDING of the same track within GROUND_LOOKBACK_S (a turnaround, flagged 1), else at the first sample in
    the lookback window. A landing is an airborne sample followed later by a ground sample at taxi speed
    (gs <= LANDING_SLOW_GS): an 'airborne' flag on the take-off roll (EHAM reports on_ground=0 from ~53 kt, before MVT)
    or a roll flicker is never followed by a slow ground sample, so it is not a landing (post-mortem 2026-09-25)."""
    b = s[(s.t < t_takeoff) & (s.t >= t_takeoff - GROUND_LOOKBACK_S)]
    og, gs = b.on_ground.to_numpy(), b.gs.to_numpy(dtype=float)
    slow_ground = np.flatnonzero((og == 1) & (gs <= LANDING_SLOW_GS))
    air = np.flatnonzero(og == 0)
    turn = 0
    if len(air) and len(slow_ground):
        landed = air[air < slow_ground[-1]]                  # airborne samples that a slow ground sample follows
        if len(landed):
            b = b.iloc[landed[-1] + 1:]
            turn = 1
    return b[b.on_ground == 1], turn


def _secs(ts: pd.Series) -> np.ndarray:
    """UTC timestamps -> float seconds (never .astype(int64): that yields MICROseconds on these frames)."""
    s = pd.to_datetime(ts, utc=True)
    return (s - EPOCH).dt.total_seconds().to_numpy(dtype="float64")


def _xy(lat, lon, airport):
    a, o = ae.APTS[airport]
    return ((np.asarray(lon, float) - o) * np.cos(np.radians(a)) * M_PER_DEG_LON_EQ,
            (np.asarray(lat, float) - a) * M_PER_DEG_LAT)


def _norm_cs(x) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
        return None
    s = str(x).replace(" ", "").upper()
    return s or None


# ---------------------------------------------------------------- 1. tracks

def segment_tracks(adsb: pd.DataFrame) -> pd.DataFrame:
    d = adsb.sort_values(["icao", "t"], kind="mergesort").reset_index(drop=True)
    new = (d.icao != d.icao.shift()) | (d.t.diff() > TRACK_GAP_S)
    d["track_id"] = np.cumsum(new.to_numpy()) - 1
    return d


# ---------------------------------------------------------------- 2. take-off events

def takeoff_events(tr: pd.DataFrame, per_event_callsign: bool = False) -> pd.DataFrame:
    """Every last-ground -> first-airborne (gs >= TAKEOFF_MIN_GS) transition per track, named with a callsign.

    Default (the shipped behaviour): ONE callsign per track — the track's last non-null callsign — on every take-off
    event of that track.
    per_event_callsign=True (arm ECF, plans/PREREG_event_callsign_fix_2026_09_18.md; opt-in): event i carries the last
    non-null callsign among its track's samples with t <= t_takeoff_i (equal-t samples included, frame order breaks the
    tie); a track that reports no callsign at or before the take-off falls back to its FIRST non-null callsign; a track
    that never reports one keeps None. Rows, order, dtypes and every other column are the same under both flags. The
    True path relies on segment_tracks' (icao, t) order, as the transition test below already does."""
    g = tr.on_ground.to_numpy()
    tid = tr.track_id.to_numpy()
    nxt_same = np.r_[tid[1:] == tid[:-1], False]
    nxt_air = np.r_[g[1:] == 0, False]
    nxt_fast = np.r_[tr.gs.to_numpy()[1:] >= TAKEOFF_MIN_GS, False]
    i = np.flatnonzero((g == 1) & nxt_same & nxt_air & nxt_fast)
    if len(i) > 1:                  # one roll, one event: a flickering roll keeps its LAST ground->air transition
        t_air = tr.t.to_numpy()[i + 1]
        later_same_roll = (tid[i[1:]] == tid[i[:-1]]) & (t_air[1:] - t_air[:-1] <= ROLL_MERGE_S)
        i = i[np.r_[~later_same_roll, True]]
    if len(i) == 0:
        return pd.DataFrame(columns=["track_id", "icao", "callsign", "airport", "t_takeoff"])
    ev = pd.DataFrame({"track_id": tid[i], "icao": tr.icao.to_numpy()[i + 1], "airport": tr.airport.to_numpy()[i],
                       "t_takeoff": tr.t.to_numpy()[i + 1]})
    if per_event_callsign:
        t = tr.t.to_numpy()
        same_fix = np.r_[(tid[1:] == tid[:-1]) & (t[1:] == t[:-1]), False]   # samples k, k+1: one track, one t
        run_end = np.flatnonzero(~same_fix)                                    # last position of every equal-t run
        last = run_end[np.searchsorted(run_end, i + 1)]                        # last sample with t <= t_takeoff_i
        by_track = tr.groupby("track_id", sort=False).callsign
        at_takeoff = by_track.ffill().to_numpy(dtype=object)[last]             # last non-null at or before it
        first = ev.track_id.map(by_track.first()).to_numpy(dtype=object)       # the fallback
        picked = np.where(pd.isna(at_takeoff), first, at_takeoff)
        ev["callsign"] = pd.Series([None if pd.isna(c) else c for c in picked], index=ev.index)
    else:
        cs = tr.groupby("track_id").callsign.agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else None)
        ev["callsign"] = ev.track_id.map(cs)
    return ev[["track_id", "icao", "callsign", "airport", "t_takeoff"]]


# ---------------------------------------------------------------- 3. join

def join_flights(flights: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    fl = flights.reset_index(drop=True).copy()
    mvt = _secs(fl.MVT_TIME_UTC_mvt)
    track = np.full(len(fl), np.nan)
    ttk = np.full(len(fl), np.nan)
    jtype = np.array([None] * len(fl), dtype=object)
    score = np.full(len(fl), np.inf)
    ev_idx = np.full(len(fl), -1)
    if len(events):
        ev = events.reset_index(drop=True)
        ev_cs = np.array([_norm_cs(c) for c in ev.callsign], dtype=object)
        for ap, grp in fl.groupby("ADEP_mvt").groups.items():
            e_ap = np.flatnonzero(ev.airport.to_numpy() == ap)
            if len(e_ap) == 0:
                continue
            order = e_ap[np.argsort(ev.t_takeoff.to_numpy()[e_ap], kind="mergesort")]
            et = ev.t_takeoff.to_numpy()[order]
            for r in grp:
                lo, hi = np.searchsorted(et, mvt[r] - JOIN_CALLSIGN_S), np.searchsorted(et, mvt[r] + JOIN_CALLSIGN_S,
                                                                                         side="right")
                if lo == hi:
                    continue
                cand = order[lo:hi]
                dts = ev.t_takeoff.to_numpy()[cand] - mvt[r]
                want = {_norm_cs(fl.at[r, "CALLSIGN_flt"]), _norm_cs(fl.at[r, "FLIGHT_mvt"])} - {None}
                csm = np.array([c in want for c in ev_cs[cand]], dtype=bool)
                if csm.any():
                    k = np.flatnonzero(csm)[np.argmin(np.abs(dts[csm]))]
                    jt, sc = "callsign", abs(dts[k])
                else:
                    near = np.flatnonzero(np.abs(dts) <= JOIN_TIME_S)
                    if len(near) != 1:
                        continue
                    k, jt, sc = near[0], "time", 1e6 + abs(dts[near[0]])     # callsign joins win any conflict
                ev_idx[r], jtype[r], score[r] = cand[k], jt, sc
        # one event serves at most one flight: the best-scored claim keeps it
        claimed = pd.DataFrame({"r": np.arange(len(fl)), "e": ev_idx, "s": score})
        claimed = claimed[claimed.e >= 0].sort_values(["e", "s"], kind="mergesort")
        keep = claimed.drop_duplicates("e", keep="first").r.to_numpy()
        lose = np.setdiff1d(claimed.r.to_numpy(), keep)
        ev_idx[lose], jtype[lose] = -1, None
        ok = ev_idx >= 0
        track[ok] = ev.track_id.to_numpy()[ev_idx[ok]]
        ttk[ok] = ev.t_takeoff.to_numpy()[ev_idx[ok]]
    fl["track_id"], fl["t_takeoff"], fl["join_type"] = track, ttk, jtype
    fl["dt_takeoff"] = ttk - mvt
    return fl


JOIN_MVT_WINDOW_S = 10_800.0     # arm ADJ: how far before MVT a same-callsign ground observation may lie


def join_mvt_anchored(flights: pd.DataFrame, tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arm ADJ (plans/PREREG_adsb_join_recall_2026_09_12.md): a second join path for departures the take-off-anchored
    join missed.

    A row qualifies iff it has no join today, our extract holds ON-GROUND samples at its ADEP in
    [MVT - JOIN_MVT_WINDOW_S, MVT) whose normalised callsign matches the flight's, and EXACTLY ONE aircraft does so.
    That aircraft's samples in the window become one segment (so a >30-minute gap no longer splits it) and the anchor
    is MVT, a serve-time column. Returns (flights with track_id / t_takeoff / join_type / dt_takeoff filled,
    tr plus the synthetic segments).

    The confound is registered: a same-callsign ground track can be the inbound arrival of a turnaround."""
    fl = flights.reset_index(drop=True).copy()
    todo = np.flatnonzero(fl.track_id.isna().to_numpy())
    if not len(todo) or tr.empty:
        return fl, tr
    mvt = _secs(fl.MVT_TIME_UTC_mvt)
    g = tr[tr.on_ground == 1]
    if g.empty:
        return fl, tr
    cs = g.callsign.map(_norm_cs)
    groups = {k: v for k, v in g.assign(_cs=cs).groupby(["airport", "_cs"])}
    next_id = float(np.nanmax(tr.track_id.to_numpy(dtype=float))) + 1.0 if len(tr) else 1.0
    extra, tid, ttk, jt = [], fl.track_id.to_numpy(dtype=float).copy(), fl.t_takeoff.to_numpy(dtype=float).copy(), \
        fl.join_type.to_numpy(dtype=object).copy()
    by_icao = {k: v for k, v in tr.groupby("icao")}
    for r in todo:
        want = {_norm_cs(fl.at[r, "CALLSIGN_flt"]), _norm_cs(fl.at[r, "FLIGHT_mvt"])} - {None}
        cand = set()
        for c in want:
            sub = groups.get((fl.at[r, "ADEP_mvt"], c))
            if sub is not None:
                m = (sub.t >= mvt[r] - JOIN_MVT_WINDOW_S) & (sub.t < mvt[r])
                cand.update(sub.loc[m, "icao"].unique().tolist())
        if len(cand) != 1:
            continue
        icao = cand.pop()
        s = by_icao[icao]
        s = s[(s.airport == fl.at[r, "ADEP_mvt"]) & (s.t >= mvt[r] - JOIN_MVT_WINDOW_S) & (s.t < mvt[r])]
        if s.empty:
            continue
        tid[r], ttk[r], jt[r] = next_id, mvt[r], "mvt"
        extra.append(s.assign(track_id=next_id))
        next_id += 1.0
    fl["track_id"], fl["t_takeoff"], fl["join_type"] = tid, ttk, jt
    fl["dt_takeoff"] = ttk - mvt
    return fl, (pd.concat([tr] + extra, ignore_index=True) if extra else tr)


def stitch_takeoff_joined(flights: pd.DataFrame, tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Arm STS (plans/PREREG_track_stitch_arm_2026_09_14.md), opt-in: a take-off-joined flight (join_type `callsign` /
    `time`) reads only its own track, and `segment_tracks` splits a track after TRACK_GAP_S of silence, so a parked
    aircraft that went quiet before pushback loses its stand samples. Each such flight gets a new segment = its track's
    samples plus every sample of the same icao at its ADEP in [t_takeoff - GROUND_LOOKBACK_S, t_takeoff), de-duplicated
    on (icao, t). The join itself (track's aircraft, take-off anchor, join type) is unchanged; other rows untouched."""
    fl = flights.reset_index(drop=True).copy()
    jt = fl.join_type.to_numpy(dtype=object)
    rows = np.flatnonzero(fl.track_id.notna().to_numpy() & np.isin(jt, ["callsign", "time"]))
    if not len(rows) or tr.empty:
        return fl, tr
    by_track = {k: v for k, v in tr.groupby("track_id")}
    by_icao = {k: v for k, v in tr.groupby("icao")}
    next_id = float(np.nanmax(tr.track_id.to_numpy(dtype=float))) + 1.0
    tid = fl.track_id.to_numpy(dtype=float).copy()
    extra = []
    for r in rows:
        own = by_track.get(fl.track_id.iat[r])
        if own is None:
            continue
        tko, ap = float(fl.t_takeoff.iat[r]), fl.ADEP_mvt.iat[r]
        s = by_icao[own.icao.iat[0]]
        s = s[(s.airport == ap) & (s.t >= tko - GROUND_LOOKBACK_S) & (s.t < tko)]
        seg = pd.concat([own, s], ignore_index=True).drop_duplicates(["icao", "t"]).sort_values("t", kind="mergesort")
        tid[r] = next_id
        extra.append(seg.assign(track_id=next_id))
        next_id += 1.0
    fl["track_id"] = tid
    return fl, (pd.concat([tr] + extra, ignore_index=True) if extra else tr)


# ---------------------------------------------------------------- 4. stand centroids

def _parked_positions(j: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """Median parked position (m, airport frame) of each joined flight around its AOBT_3; `parked` = 1.0 when it comes
    from stationary samples in CENTROID_WINDOW, 0.0 when from the first-appearance fallback (first slow samples)."""
    cols = ["MVT_ID_mvt", "ADEP_mvt", "STAND_mvt", "px", "py", "parked"]
    ok = j.track_id.notna() & j.AOBT_3_flt.notna() & j.STAND_mvt.notna()
    if not ok.any():
        return pd.DataFrame(columns=cols)
    by_track = {k: v for k, v in tr.groupby("track_id")}
    aobt = _secs(j.AOBT_3_flt)
    out = []
    for r in np.flatnonzero(ok.to_numpy()):
        s = by_track.get(j.track_id.iat[r])
        if s is None:
            continue
        m = ((s.on_ground == 1) & (s.gs < STATIONARY_GS) & (s.t >= aobt[r] + CENTROID_WINDOW[0])
             & (s.t <= aobt[r] + CENTROID_WINDOW[1]))
        pos = s[m]
        parked = 1.0
        if pos.empty:                                    # rarely seen parked: use its first slow ground samples
            parked = 0.0
            seg, _ = _ground_segment(s, j.t_takeoff.iat[r])
            head = seg.iloc[:FIRST_N]
            pos = head[head.gs < SLOW_FIRST_GS]
        if pos.empty:
            continue
        x, y = _xy(pos.lat, pos.lon, j.ADEP_mvt.iat[r])
        out.append((j.MVT_ID_mvt.iat[r], j.ADEP_mvt.iat[r], j.STAND_mvt.iat[r], float(np.median(x)), float(np.median(y)),
                    parked))
    return pd.DataFrame(out, columns=cols)


CENTROID_COLS = ["cx_m", "cy_m", "n_others", "parked_share", "pcx_m", "pcy_m", "n_parked_others"]


def stand_centroids(j: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-flight-out references per flight, from the OTHER flights at its (airport, stand):
      cx_m, cy_m       median position of all others (parked or first-appearance); NaN with < CENTROID_MIN_OTHERS
      parked_share     share of those others seen parked (provenance); NaN when cx_m is NaN
      pcx_m, pcy_m     median of the PARKED others only; NaN with < CENTROID_MIN_OTHERS parked others (DWG: the only
                       reference allowed to place a gate dwell)"""
    pp = _parked_positions(j, tr)
    res = pd.DataFrame({"MVT_ID_mvt": j.MVT_ID_mvt.to_numpy(), "cx_m": np.nan, "cy_m": np.nan, "n_others": 0,
                        "parked_share": np.nan, "pcx_m": np.nan, "pcy_m": np.nan, "n_parked_others": 0})
    if pp.empty:
        return res
    pos = {}
    for (ap, st), g in pp.groupby(["ADEP_mvt", "STAND_mvt"]):
        pos[(ap, st)] = g
    rows = j.set_index("MVT_ID_mvt")
    cx, cy, n, share, pcx, pcy, pn = [], [], [], [], [], [], []
    for mid in res.MVT_ID_mvt:
        g = pos.get((rows.at[mid, "ADEP_mvt"], rows.at[mid, "STAND_mvt"]))
        o = g[g.MVT_ID_mvt != mid] if g is not None else pp.iloc[:0]
        p = o[o.parked == 1.0]
        n.append(len(o)), pn.append(len(p))
        if len(o) < CENTROID_MIN_OTHERS:
            cx.append(np.nan), cy.append(np.nan), share.append(np.nan)
        else:
            cx.append(float(o.px.median())), cy.append(float(o.py.median())), share.append(float(o.parked.mean()))
        if len(p) < CENTROID_MIN_OTHERS:
            pcx.append(np.nan), pcy.append(np.nan)
        else:
            pcx.append(float(p.px.median())), pcy.append(float(p.py.median()))
    res["cx_m"], res["cy_m"], res["n_others"], res["parked_share"] = cx, cy, n, share
    res["pcx_m"], res["pcy_m"], res["n_parked_others"] = pcx, pcy, pn
    return res


# ---------------------------------------------------------------- 5. features

def _runs(stationary: np.ndarray):
    """(start, end) index pairs of consecutive True runs."""
    if not stationary.any():
        return []
    d = np.diff(np.r_[0, stationary.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _gate_dwell(runs, t, x, y, aobt: float, pref) -> tuple | None:
    """DWG (BC-26): the first stationary run >= GATE_DWELL_MIN_S that starts at or before aobt + DWELL_START_MAX_S and,
    when pref = (px, py) is a PARKED-provenance stand reference, whose median lies within GATE_RADIUS_M of it. A NaN
    aobt places no run before off-block (no gate dwell). None when no run qualifies."""
    for a, b in runs:
        if t[b] - t[a] < GATE_DWELL_MIN_S or not (t[a] <= aobt + DWELL_START_MAX_S):
            continue
        if pref is not None and np.hypot(np.median(x[a:b + 1]) - pref[0], np.median(y[a:b + 1]) - pref[1]) > GATE_RADIUS_M:
            continue
        return a, b
    return None


def flight_features(j: pd.DataFrame, tr: pd.DataFrame, centroids: pd.DataFrame | None = None) -> pd.DataFrame:
    n = len(j)
    if centroids is not None:
        miss = [c for c in CENTROID_COLS if c not in centroids.columns]
        if miss:
            raise ValueError(f"centroids frame lacks {miss} (build it with stand_centroids)")
    f = {c: np.full(n, np.nan) for c in FEATURES}
    f["joined"] = j.track_id.notna().to_numpy().astype(float)
    joined = j.track_id.notna().to_numpy()
    f["stand_visible"] = np.where(joined, 0.0, np.nan)                  # DWG; BC-3: unknown when unjoined
    f["join_callsign"] = np.where(joined, (j.join_type == "callsign").to_numpy().astype(float), np.nan)   # BC-3
    f["dt_takeoff"] = j.dt_takeoff.to_numpy(dtype=float)
    if "join_type" in j:                                     # ADJ: which join found this row, and how stale its anchor
        f["join_mvt_anchored"] = np.where(joined, (j.join_type == "mvt").to_numpy().astype(float), np.nan)
    mvt, aobt = _secs(j.MVT_TIME_UTC_mvt), _secs(j.AOBT_3_flt)
    cen = None if centroids is None else centroids.set_index("MVT_ID_mvt")
    by_track = {k: v for k, v in tr.groupby("track_id")} if n and j.track_id.notna().any() else {}
    for r in np.flatnonzero(j.track_id.notna().to_numpy()):
        s = by_track.get(j.track_id.iat[r])
        if s is None:
            continue
        tko = j.t_takeoff.iat[r]
        f["first_seen_rel_mvt"][r] = mvt[r] - s.t.iat[0]
        g, turn = _ground_segment(s, tko)
        f["turn_in_track"][r] = turn
        if g.empty:
            continue
        t, gs = g.t.to_numpy(), g.gs.to_numpy()
        if j.join_type.iat[r] == "mvt":                      # ADJ: how long before MVT the last ground sample lies
            f["mvt_anchor_gap_s"][r] = mvt[r] - t[-1]
        f["first_ground_rel_mvt"][r] = mvt[r] - t[0]
        f["first_ground_rel_aobt3"][r] = t[0] - aobt[r]
        g0 = float(np.nanmedian(gs[:FIRST_N])) if np.isfinite(gs[:FIRST_N]).any() else np.nan
        f["first_ground_gs"][r] = g0
        f["first_ground_slow"][r] = float(g0 < SLOW_FIRST_GS) if np.isfinite(g0) else np.nan
        mv_i = np.flatnonzero(gs >= MOVING_GS)
        f["first_move_rel_mvt"][r] = mvt[r] - t[mv_i[0]] if len(mv_i) else np.nan
        f["n_ground"][r] = len(g)
        span = t[-1] - t[0]
        f["ground_rate_pm"][r] = 60.0 * len(g) / span if span >= SPAN_MIN_S else np.nan     # AFJ: < 1 ms is one fix
        f["mlat_share"][r] = float((g.src == "mlat").mean())
        x, y = _xy(g.lat, g.lon, j.ADEP_mvt.iat[r])
        runs = _runs(np.nan_to_num(gs, nan=np.inf) < STATIONARY_GS)
        mid = j.MVT_ID_mvt.iat[r]
        has_cen = cen is not None and mid in cen.index
        pref = None
        if has_cen and np.isfinite(cen.at[mid, "pcx_m"]):
            pref = (float(cen.at[mid, "pcx_m"]), float(cen.at[mid, "pcy_m"]))
        dwell = _gate_dwell(runs, t, x, y, aobt[r], pref)
        start = 0
        if dwell is not None:
            a, b = dwell
            f["stand_visible"][r] = 1.0
            f["dwell_end_rel_mvt"][r] = mvt[r] - t[b]
            f["dwell_s"][r] = t[b] - t[a]
            f["dwell_end_rel_aobt3"][r] = t[b] - aobt[r]
            start = b
            later = [(a2, b2) for a2, b2 in runs if a2 > b and t[b2] - t[a2] >= STOP_MIN_S]
            f["n_stops"][r] = len(later)
            f["stop_s"][r] = float(sum(t[b2] - t[a2] for a2, b2 in later))
            f["path_m"][r] = float(np.sum(np.hypot(np.diff(x[b:]), np.diff(y[b:])))) if len(x) - b > 1 else np.nan
            mv = gs[b + 1:]
            mv = mv[mv >= MOVING_GS]
            f["taxi_med_gs"][r] = float(np.median(mv)) if len(mv) else np.nan
        if cen is not None and j.MVT_ID_mvt.iat[r] in cen.index:
            cx, cy = cen.at[j.MVT_ID_mvt.iat[r], "cx_m"], cen.at[j.MVT_ID_mvt.iat[r], "cy_m"]
            f["stand_others_n"][r] = float(cen.at[j.MVT_ID_mvt.iat[r], "n_others"])
            f["stand_ref_parked_share"][r] = float(cen.at[j.MVT_ID_mvt.iat[r], "parked_share"])
            if np.isfinite(cx):
                # ADX: how far from its stand the aircraft already was when ADS-B first saw it on the ground
                f["first_ground_to_stand_m"][r] = float(np.hypot(x[0] - cx, y[0] - cy))
                inside = np.flatnonzero(np.hypot(x - cx, y - cy) <= GATE_RADIUS_M)
                if len(inside):
                    f["sensor_rel_mvt"][r] = mvt[r] - t[inside[-1]]
                    f["sensor_rel_aobt3"][r] = t[inside[-1]] - aobt[r]
                if dwell is not None:
                    a, b = dwell
                    f["dwell_to_centroid_m"][r] = float(np.hypot(np.median(x[a:b + 1]) - cx, np.median(y[a:b + 1]) - cy))
        del start
    out = pd.DataFrame(f)
    out.insert(0, "MVT_ID_mvt", j.MVT_ID_mvt.to_numpy())
    return out


def taxi_start_block(f: pd.DataFrame) -> pd.DataFrame:
    """Arm ADX: the reconstructed start of taxi, from `ap`, `first_ground_to_stand_m`, `taxi_med_gs`,
    `first_ground_rel_aobt3` and `first_ground_rel_mvt`.

        travel_s_first_to_stand   = distance / speed, speed = the flight's own taxi_med_gs when >= MIN_TAXI_GS,
                                    else the airport's median usable taxi speed in this period; NaN when neither exists
        implied_offblock_rel_aobt3 = first_ground_rel_aobt3 - travel_s      (negative = off-block before AOBT_3)
        implied_taxi_s             = first_ground_rel_mvt + travel_s        (MVT minus the reconstructed off-block)

    It is a LOWER bound on the missing time: a straight line, and no allowance for holding (registered up front)."""
    d = f.first_ground_to_stand_m.to_numpy(dtype=float)
    own = f.taxi_med_gs.to_numpy(dtype=float)
    usable = np.where(own >= MIN_TAXI_GS, own, np.nan)
    med = pd.Series(usable).groupby(np.asarray(f.ap)).transform("median").to_numpy(dtype=float)
    speed = np.where(np.isfinite(usable), usable, med)
    travel = np.where(np.isfinite(d) & np.isfinite(speed) & (speed >= MIN_TAXI_GS), d / (speed * KNOT_MS), np.nan)
    return pd.DataFrame({"travel_s_first_to_stand": travel,
                         "implied_offblock_rel_aobt3": f.first_ground_rel_aobt3.to_numpy(dtype=float) - travel,
                         "implied_taxi_s": f.first_ground_rel_mvt.to_numpy(dtype=float) + travel}, index=f.index)


def _moving_index(tr: pd.DataFrame, moving_only: bool = True) -> dict:
    """{airport: (t sorted, icao aligned)} of on-ground samples (moving faster than MOVING_GS if moving_only)."""
    mv = tr[(tr.on_ground == 1) & (tr.gs >= MOVING_GS)] if moving_only else tr[tr.on_ground == 1]   # BC-4: >= everywhere
    out = {}
    for ap, g in mv.groupby("airport"):
        g = g.sort_values("t", kind="mergesort")
        out[ap] = (g.t.to_numpy(), g.icao.to_numpy())
    return out


def _distinct_near(idx, ap, center, own) -> float:
    """Distinct OTHER aircraft in idx[ap] within ±CONGESTION_HALF_S of center; 0.0 when ap has no samples."""
    if ap not in idx:
        return 0.0
    t, ic = idx[ap]
    lo, hi = np.searchsorted(t, center - CONGESTION_HALF_S), np.searchsorted(t, center + CONGESTION_HALF_S, side="right")
    s = set(ic[lo:hi])
    s.discard(own)
    return float(len(s))


def _count_moving(idx, ap, center, own, seen_idx) -> float:
    """Distinct other aircraft MOVING on the ground near center. BC-3: NaN — never 0 — when no other aircraft is
    seen on the ground at all in the window (no reception is not 'no traffic')."""
    if not np.isfinite(center) or _distinct_near(seen_idx, ap, center, own) == 0.0:
        return np.nan
    return _distinct_near(idx, ap, center, own)


FEATURES_VERSION = ("adsb_features v4 (take-off-anchored join + ADJ MVT-anchored second join; first-seen-on-ground; "
                    "LOO stand refs from parked or first-slow samples; ADX taxi-start reconstruction; "
                    "TRC landing = airborne then taxi-speed ground, one event per take-off roll [PM 2026-09-25]; "
                    "DWG gate dwell = stationary >= 300 s starting <= AOBT_3 + 60 s, within 100 m of a PARKED stand "
                    "ref when one exists, + stand_visible + stand-ref parked share [PM 2026-09-27])")
EXTRACTOR = "adsb_fetch.parse_member_v2"
NEXT_DAY_HEAD_S = 900.0               # a take-off just after midnight lives in the next day's file


def _day_slice(adsb_dir: pathlib.Path, day: pd.Timestamp, lo: float, hi: float) -> pd.DataFrame | None:
    p = pathlib.Path(adsb_dir) / f"adsb_{day:%Y-%m-%d}.parquet"
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    return d[(d.t >= lo) & (d.t < hi)]


def coverage_code(has_day: bool, feats: pd.DataFrame) -> np.ndarray:
    """0 = no ADS-B file for the flight's day; 1 = day present, flight not joined; 2 = joined, first seen mid-taxi;
    3 = joined and seen from the stand (a gate dwell, or a slow first ground sample)."""
    from_stand = np.isfinite(feats.dwell_s.to_numpy()) | (feats.first_ground_slow.to_numpy() == 1)
    code = np.where(feats.joined.to_numpy() == 1, np.where(from_stand, 3, 2), 1)
    return code if has_day else np.zeros(len(feats), dtype=int)


def build_period(flights: pd.DataFrame, adsb_dir, git_sha: str, out_path=None, stitch: bool = False,
                 per_event_callsign: bool = False) -> pd.DataFrame:
    """Features for every flight, one MVT day at a time: that day's ADS-B plus the previous day's last
    GROUND_LOOKBACK_S and the next day's first NEXT_DAY_HEAD_S. Adds `coverage`; stamps the parquet metadata.
    per_event_callsign=True: arm ECF (`takeoff_events`); default off reproduces the shipped tables."""
    fl = flights[FLIGHT_COLS].copy()                       # label columns never enter
    day = pd.to_datetime(fl.MVT_TIME_UTC_mvt, utc=True).dt.normalize()
    parts = []
    for d, idx in fl.groupby(day).groups.items():
        start = (d - EPOCH).total_seconds()
        cur = _day_slice(adsb_dir, d, start, start + 86_400)
        sub = fl.loc[idx]
        if cur is None:
            feats = build(sub, pd.DataFrame(columns=["icao", "reg", "type", "callsign", "airport", "t", "lat", "lon",
                                                     "on_ground", "alt_missing", "alt", "gs", "track", "src"])
                          .astype({"t": float, "lat": float, "lon": float, "on_ground": int, "gs": float}), stitch=stitch,
                          per_event_callsign=per_event_callsign)
            feats["coverage"] = coverage_code(False, feats)
        else:
            prev = _day_slice(adsb_dir, d - pd.Timedelta(days=1), start - GROUND_LOOKBACK_S, start)
            nxt = _day_slice(adsb_dir, d + pd.Timedelta(days=1), start + 86_400, start + 86_400 + NEXT_DAY_HEAD_S)
            adsb = pd.concat([x for x in (prev, cur, nxt) if x is not None and len(x)], ignore_index=True)
            feats = build(sub, adsb, stitch=stitch, per_event_callsign=per_event_callsign)
            feats["coverage"] = coverage_code(True, feats)
        parts.append(feats)
    out = pd.concat(parts, ignore_index=True)
    out = out.set_index("MVT_ID_mvt").loc[fl.MVT_ID_mvt.to_numpy()].reset_index()
    if out_path is not None:
        import pyarrow as pa
        import pyarrow.parquet as pq
        t = pa.Table.from_pandas(out, preserve_index=False)
        meta = dict(t.schema.metadata or {})
        meta.update({b"git_sha": git_sha.encode(), b"extractor": EXTRACTOR.encode(),
                     b"features_version": features_version(stitch).encode(),
                     b"source": b"adsb.lol globe_history, ODbL 1.0 (attribution: ADSB.lol contributors)"})
        pq.write_table(t.replace_schema_metadata(meta), out_path)
    return out


def fold_flights(record_path, stand_dir, order_dir, raw_dir) -> pd.DataFrame:
    """Serve-time columns (FLIGHT_COLS) + `row` for every row of a fold prediction record, in the record's order.
    row -> MVT_ID through the order cache (same rows, same order as the stand cache); then the raw training files.
    Guard (label-free): MVT - AOBT_3 from the raw file equals the record's `proxy` within 1 s on every row."""
    import glob
    import pyarrow.parquet as pq
    rec = pd.read_parquet(record_path, columns=["row", "proxy"])
    cs = sorted(glob.glob(str(pathlib.Path(stand_dir) / "training_2025-*.parquet")))
    co = sorted(glob.glob(str(pathlib.Path(order_dir) / "training_2025-*.parquet")))
    if [pathlib.Path(p).name for p in cs] != [pathlib.Path(p).name for p in co]:
        raise ValueError("stand/order cache file lists differ")
    ids = []
    for a, b in zip(cs, co):
        ob = pq.read_table(b, columns=["MVT_ID_mvt"]).column(0).to_numpy()
        if len(ob) != pq.read_metadata(a).num_rows:
            raise ValueError(f"row count differs for {pathlib.Path(a).name}")
        ids.append(ob)
    mvt_ids = np.concatenate(ids)[rec.row.to_numpy()]
    months = {pathlib.Path(p).name for p in cs}
    raw = pd.concat([pd.read_parquet(pathlib.Path(raw_dir) / m, columns=FLIGHT_COLS + ["PHASE_mvt"])
                     for m in sorted(months) if (pathlib.Path(raw_dir) / m).exists()], ignore_index=True)
    raw = raw[raw.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt").drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    missing = np.setdiff1d(mvt_ids, raw.index.to_numpy())
    if len(missing):
        raise ValueError(f"{len(missing)} record MVT_IDs absent from the raw files (e.g. {missing[:3]})")
    fl = raw.loc[mvt_ids].reset_index()
    gap = _secs(fl.MVT_TIME_UTC_mvt) - _secs(fl.AOBT_3_flt)
    bad = np.abs(gap - rec.proxy.to_numpy()) > 1.0
    if bad.any():
        raise ValueError(f"alignment guard: {int(bad.sum())} rows where MVT - AOBT_3 != record proxy")
    fl.insert(0, "row", rec.row.to_numpy())
    return fl[["row"] + FLIGHT_COLS]


def ranking_flights(ranking_path, submitting_path, matched: bool = True) -> pd.DataFrame:
    """Serve-time columns (FLIGHT_COLS) of the scored MATCHED departures (AOBT_3 present; matched=False: the scored
    UNMATCHED departures, AOBT_3 null), in the submission template's order."""
    sub = pd.read_parquet(submitting_path, columns=["MVT_ID_mvt"])
    rk = pd.read_parquet(ranking_path, columns=FLIGHT_COLS + ["PHASE_mvt"])
    lane = rk.AOBT_3_flt.notna() if matched else rk.AOBT_3_flt.isna()
    rk = rk[(rk.PHASE_mvt == "DEP") & lane].drop(columns="PHASE_mvt")
    rk = rk.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    ids = sub.MVT_ID_mvt.to_numpy()
    ids = ids[np.isin(ids, rk.index.to_numpy())]
    return rk.loc[ids].reset_index()[FLIGHT_COLS]


def record_flights(record_path, raw_paths, months=(1, 7)) -> pd.DataFrame:
    """Serve-time columns (FLIGHT_COLS) for a record keyed by MVT_ID_mvt (e.g. S1C's unmatched LOMO record), restricted
    to `months`, in the record's order. Arm ADU."""
    rec = pd.read_parquet(record_path, columns=["MVT_ID_mvt", "month"])
    rec = rec[rec.month.isin(months)]
    raw = pd.concat([pd.read_parquet(p, columns=FLIGHT_COLS + ["PHASE_mvt"]) for p in raw_paths], ignore_index=True)
    raw = raw[raw.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt").drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    ids = rec.MVT_ID_mvt.to_numpy()
    missing = np.setdiff1d(ids, raw.index.to_numpy())
    if len(missing):
        raise ValueError(f"{len(missing)} record ids absent from the raw files (e.g. {missing[:3]})")
    return raw.loc[ids].reset_index()[FLIGHT_COLS]


def _git_sha() -> str:
    import subprocess
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
    sha = r.stdout.strip() if r.returncode == 0 else "unknown"
    d = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "scripts/adsb_features.py",
                        "scripts/adsb_fetch.py"], capture_output=True, text=True)
    return sha + ("-dirty" if d.stdout.strip() else "")


def main(argv=None) -> int:
    """build-fold: the label-free ADS-B feature table for a fold record's rows (one row per record row, same order,
    plus `row`)."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-fold")
    b.add_argument("--record", default=str(ROOT / "data" / "cache_stand" / "fold_preds_queue_order.parquet"))
    b.add_argument("--stand-dir", default=str(ROOT / "data" / "cache_stand"))
    b.add_argument("--order-dir", default=str(ROOT / "data" / "cache_order"))
    b.add_argument("--raw-dir", default=str(ROOT / "data" / "raw"))
    b.add_argument("--adsb-dir", default=str(ROOT / "data" / "adsb" / "v2"))
    b.add_argument("--out", default=str(ROOT / "data" / "adsb" / "v2" / "features_2025janjul.parquet"))
    b.add_argument("--stitch", action="store_true", help="arm STS: stitched ground segments for take-off-joined flights")
    u = sub.add_parser("build-unmatched", help="arm ADU: S1C's LOMO record, January + July 2025")
    u.add_argument("--record", default=str(ROOT / "data" / "cache_stand" / "unm_congestion_lomo.parquet"))
    u.add_argument("--adsb-dir", default=str(ROOT / "data" / "adsb" / "v2"))
    u.add_argument("--out", default=str(ROOT / "data" / "adsb" / "v2" / "features_unm_2025janjul.parquet"))
    ru = sub.add_parser("build-ranking-unmatched", help="the 2026 scored UNMATCHED departures (for c4's fill work)")
    ru.add_argument("--ranking", default=str(ROOT / "data" / "raw" / "ranking.parquet"))
    ru.add_argument("--submitting", default=str(ROOT / "data" / "raw" / "submitting.parquet"))
    ru.add_argument("--adsb-dir", default=str(ROOT / "data" / "adsb" / "v2" / "y2026"))
    ru.add_argument("--out", default=str(ROOT / "data" / "adsb" / "v2" / "features_unm_2026janjul.parquet"))
    r = sub.add_parser("build-ranking", help="the 2026 scored matched departures (the ship side)")
    r.add_argument("--ranking", default=str(ROOT / "data" / "raw" / "ranking.parquet"))
    r.add_argument("--submitting", default=str(ROOT / "data" / "raw" / "submitting.parquet"))
    r.add_argument("--adsb-dir", default=str(ROOT / "data" / "adsb" / "v2" / "y2026"))
    r.add_argument("--out", default=str(ROOT / "data" / "adsb" / "v2" / "features_2026janjul.parquet"))
    r.add_argument("--stitch", action="store_true", help="arm STS: stitched ground segments for take-off-joined flights")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    if a.cmd in ("build-ranking", "build-ranking-unmatched"):
        fl = ranking_flights(a.ranking, a.submitting, matched=(a.cmd == "build-ranking"))
        feats = build_period(fl, a.adsb_dir, git_sha=_git_sha(), stitch=getattr(a, "stitch", False))
        src = str(a.ranking)
    elif a.cmd == "build-unmatched":
        raw = [ROOT / "data" / "raw" / "training_2025-01-01_2025-02-01.parquet",
               ROOT / "data" / "raw" / "training_2025-07-01_2025-08-01.parquet"]
        fl = record_flights(a.record, raw, months=(1, 7))
        feats = build_period(fl, a.adsb_dir, git_sha=_git_sha())
        src = str(a.record)
    else:
        fl = fold_flights(a.record, a.stand_dir, a.order_dir, a.raw_dir)
        feats = build_period(fl.drop(columns="row"), a.adsb_dir, git_sha=_git_sha(), stitch=a.stitch)
        feats.insert(0, "row", fl.row.to_numpy())
        src = str(a.record)
    if not (feats.MVT_ID_mvt.to_numpy() == fl.MVT_ID_mvt.to_numpy()).all():
        raise SystemExit("feature rows are not in the flights' order")
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pa.Table.from_pandas(feats, preserve_index=False)
    meta = dict(t.schema.metadata or {})
    meta.update({b"git_sha": _git_sha().encode(), b"extractor": EXTRACTOR.encode(),
                 b"features_version": features_version(getattr(a, "stitch", False)).encode(), b"record": src.encode(),
                 b"source": b"adsb.lol globe_history, ODbL 1.0 (attribution: ADSB.lol contributors)"})
    pq.write_table(t.replace_schema_metadata(meta), out)
    print(f"wrote {out}: {len(feats):,} rows, {len(FEATURES)} features")
    return 0


def features_version(stitch: bool) -> str:
    """The table stamp: FEATURES_VERSION, plus the STS marker only when the opt-in stitch was applied."""
    return FEATURES_VERSION + (" + STS stitch (take-off-joined ground segments across track gaps)" if stitch else "")


def build(flights: pd.DataFrame, adsb: pd.DataFrame, stitch: bool = False, per_event_callsign: bool = False) -> pd.DataFrame:
    """Per-flight feature frame (MVT_ID_mvt + FEATURES). Reads ONLY the serve-time columns in FLIGHT_COLS.
    stitch=True: arm STS (`stitch_takeoff_joined`); per_event_callsign=True: arm ECF (`takeoff_events`); both default
    off reproduce v4."""
    fl = flights[FLIGHT_COLS].copy()                       # label columns never enter
    tr = segment_tracks(adsb)
    j = join_flights(fl, takeoff_events(tr, per_event_callsign=per_event_callsign))
    j, tr = join_mvt_anchored(j, tr)                       # arm ADJ: the second, MVT-anchored join path
    if stitch:
        j, tr = stitch_takeoff_joined(j, tr)               # arm STS
    feats = flight_features(j, tr, stand_centroids(j, tr))
    idx = _moving_index(tr)
    seen = _moving_index(tr, moving_only=False)
    own = {}
    if j.track_id.notna().any():
        own = tr.drop_duplicates("track_id").set_index("track_id").icao.to_dict()
    mvt = _secs(j.MVT_TIME_UTC_mvt)
    own_icao = [own.get(t) if np.isfinite(t) else None for t in j.track_id.to_numpy(dtype=float)]
    ap = j.ADEP_mvt.to_numpy()
    feats["ground_seen_mvt_m10"] = [_distinct_near(seen, ap[r], mvt[r] - 600.0, own_icao[r]) for r in range(len(j))]
    feats["moving_mvt_m10"] = [_count_moving(idx, ap[r], mvt[r] - 600.0, own_icao[r], seen) for r in range(len(j))]
    dwell_t = mvt - feats.dwell_end_rel_mvt.to_numpy()
    feats["moving_dwell_end"] = [_count_moving(idx, ap[r], dwell_t[r], own_icao[r], seen) for r in range(len(j))]
    for c, v in taxi_start_block(feats.assign(ap=ap)).items():       # arm ADX
        feats[c] = v
    return feats[["MVT_ID_mvt"] + FEATURES]


if __name__ == "__main__":
    raise SystemExit(main())
