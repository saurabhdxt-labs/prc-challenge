"""scripts/adsb_storm.py -- arm E4 (plans/PREREG_adsb_eham_storm_2026_09_10.md) on synthetic tracks and synthetic
ranking rows.

Pins: leave-one-flight-out centroids (a flight never contributes to its own stand); the 100 m gate and "last
sample before take-off"; the leave-one-day-out bias; the runway + take-off-second join with its ambiguity
counting; each clause's arithmetic at its threshold boundary and the verdict mapping; the ship step refusing
without --eligibility-confirmed, changing only the intended rows, preserving the dtype, passing check_submission
and not mutating the base. Every RNG is seeded. Mutation rehearsals (break -> RED -> restore byte-identical via
backup + `diff -q` -> GREEN) are recorded in the docstrings.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import adsb_storm as st  # noqa: E402

bs = st.bs
LAT0, LON0 = st.ax.APTS["EHAM"]
#: stands sit >= 250 m apart and >= 1 km from either runway's roll line (a stand ON a roll line would let the
#: take-off roll fire the gate -- physically a mis-recorded stand, not a sensor property)
STANDS = {"A1": (0.0, 0.0), "A2": (300.0, 0.0), "A3": (600.0, 0.0), "B1": (0.0, 400.0), "B2": (300.0, 400.0), "C9": (-900.0, 900.0)}
RUNWAYS = {"36L": ((-1500.0, -2000.0), (0.0, 1.0)), "24": ((2500.0, 2500.0), (-np.sqrt(0.5), -np.sqrt(0.5)))}
PUSH_DELAY_S = 180.0        # the aircraft starts moving this long after its true off-block
SPEED_MS = 6.0              # taxi speed
ROLL_TO_MVT_S = 35.0        # take-off recorded 35 s into the roll; gs = 4 kt/s -> 80 kt at +20 s, offset -15 s
CADENCE_S = 5.0


def latlon(x, y):
    return LAT0 + np.asarray(y) / st.M_PER_DEG_LAT, LON0 + np.asarray(x) / (st.M_PER_DEG_LON_EQ * np.cos(np.radians(LAT0)))


def _track(k: int, stand_xy, rw, t0: float, first_dt: float, callsign: str):
    """Samples for one departure: parked at the stand from t0 + first_dt, moving at PUSH_DELAY_S, taxi to the runway
    start at SPEED_MS, roll with gs = 4 kt/s, take-off at roll + ROLL_TO_MVT_S, four airborne samples after."""
    (rx, ry), (ux, uy) = RUNWAYS[rw]
    sx, sy = stand_xy
    dist = float(np.hypot(rx - sx, ry - sy))
    t_move, t_roll = t0 + PUSH_DELAY_S, t0 + PUSH_DELAY_S + dist / SPEED_MS
    mvt = t_roll + ROLL_TO_MVT_S
    rows = []
    t = t0 + first_dt
    while t < mvt + 4 * CADENCE_S:
        if t < t_move:
            x, y, gs, og, alt = sx, sy, 0.0, 1, "ground"
        elif t < t_roll:
            f = (t - t_move) * SPEED_MS / dist
            x, y, gs, og, alt = sx + f * (rx - sx), sy + f * (ry - sy), SPEED_MS * 1.944, 1, "ground"
        else:
            gs = 4.0 * (t - t_roll)
            s = 0.5 * (gs / 1.944) * (t - t_roll)
            x, y = rx + ux * s, ry + uy * s
            og, alt = (1, "ground") if t <= mvt else (0, 500.0)
        rows.append((f"{k:06x}", "T-EST", "A320", callsign, "EHAM", float(t), x, y, og, np.nan if alt == "ground" else alt, gs))
        t += CADENCE_S
    return rows, mvt


def synthetic_world(seed: int = 7, n_matched: int = 24, n_unmatched: int = 6, days=st.DAYS, dead_share: float = 0.1):
    """(ranking-like frame, truth frame, samples frame). AOBT_3 = true off-block - 60 s (NM's stamp runs early, as
    in the real file); BLOCK_TIME = true off-block; TAXITIME = MVT - BLOCK_TIME. `dead_share` of the matched
    flights have no track. Unmatched rows have a track (callsign UNM...) but no CALLSIGN_flt."""
    rng = np.random.default_rng(seed)
    stands, rws = list(STANDS), list(RUNWAYS)
    rank, truth, samples, k = [], [], [], 0
    for day in days:
        day_s = (pd.Timestamp(day, tz="UTC") - st.EPOCH).total_seconds()
        for j in range(n_matched + n_unmatched):
            matched = j < n_matched
            stand, rw = stands[int(rng.integers(len(stands)))], rws[int(rng.integers(len(rws)))]
            t0 = day_s + 6 * 3600 + float(rng.uniform(0, 14 * 3600))
            cs = f"TST{k:04d}" if matched else f"UNM{k:04d}"
            rows, mvt = _track(k, STANDS[stand], rw, t0, float(rng.uniform(-60, 150)), cs)
            dead = matched and rng.random() < dead_share
            if not dead:
                samples.extend(rows)
            mid = float(100_000 + k)
            rank.append({"MVT_ID_mvt": mid, "ADEP_mvt": "EHAM", "PHASE_mvt": "DEP",
                         "MVT_TIME_UTC_mvt": pd.Timestamp(mvt, unit="s", tz="UTC"),
                         "AOBT_3_flt": pd.Timestamp(t0 - 60.0, unit="s", tz="UTC") if matched else pd.NaT,
                         "BLOCK_TIME_UTC_mvt": pd.Timestamp(t0, unit="s", tz="UTC"),
                         "TAXITIME_SEC_mvt": int(round(mvt - t0)),
                         "STAND_mvt": stand, "RUNWAY_mvt": rw, "CALLSIGN_flt": cs if matched else pd.NA})
            truth.append({"MVT_ID_mvt": mid, "day": day, "t0": t0, "mvt": mvt, "matched": matched, "dead": dead, "stand": stand, "runway": rw})
            k += 1
    rank = pd.DataFrame(rank)
    # rows the pipeline must ignore: an arrival, another airport, a day outside the registered eight
    extra = rank.iloc[:3].copy()
    extra["MVT_ID_mvt"] = [999_001.0, 999_002.0, 999_003.0]
    extra.loc[extra.index[0], "PHASE_mvt"] = "ARR"
    extra.loc[extra.index[1], "ADEP_mvt"] = "LSZH"
    extra.loc[extra.index[2], "MVT_TIME_UTC_mvt"] = pd.Timestamp("2026-01-15 10:00", tz="UTC")
    rank = pd.concat([rank, extra], ignore_index=True)
    rank["MVT_TIME_UTC_mvt"] = pd.to_datetime(rank.MVT_TIME_UTC_mvt, utc=True)
    rank["AOBT_3_flt"] = pd.to_datetime(rank.AOBT_3_flt, utc=True)
    rank["BLOCK_TIME_UTC_mvt"] = pd.to_datetime(rank.BLOCK_TIME_UTC_mvt, utc=True)
    rank["CALLSIGN_flt"] = rank.CALLSIGN_flt.astype("string")
    samples = pd.DataFrame(samples, columns=["icao", "reg", "type", "callsign", "airport", "t", "lat", "lon", "on_ground", "alt", "gs"])
    samples["lat"], samples["lon"] = latlon(samples.lat.to_numpy(), samples.lon.to_numpy())   # x, y were stored there
    return rank, pd.DataFrame(truth), samples


@pytest.fixture(scope="module")
def world():
    rank, truth, samples = synthetic_world()
    flights = st.flights_frame(rank, st.DAYS)
    tracks = st.tracks_frame(samples, LAT0, LON0)
    est, info = st.measure(flights, tracks, st.DAYS, bias_mode="loo_day")
    return dict(rank=rank, truth=truth.set_index("MVT_ID_mvt"), samples=samples, flights=flights, tracks=tracks, est=est, info=info)


# =============================================================================================
# primitives and frames
# =============================================================================================

def test_to_epoch_s_is_seconds_and_refuses_microseconds():
    """ADSB_GATE landmine.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    epoch-band check in to_epoch_s disabled -> RED.
    """
    s = pd.Series(pd.to_datetime(["2026-01-05 12:00:00", None], utc=True))
    out = st.to_epoch_s(s)
    assert out[0] == pytest.approx(1_767_614_400.0) and np.isnan(out[1])       # exact integer seconds, fp64
    far = pd.Series(pd.to_datetime(["2200-01-01"], utc=True))                   # 7.3e9: outside the 2020-2030 band
    with pytest.raises(ValueError, match="not seconds"):
        st.to_epoch_s(far)
    with pytest.raises(ValueError, match="not seconds"):
        st.to_epoch_s(pd.Series(pd.to_datetime(["2001-01-01"], utc=True)))      # 9.8e8: the microsecond/1e3 shape


def test_xy_m_round_trips_the_synthetic_geometry():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): cos(lat0)
    dropped from the east component of xy_m -> RED.
    """
    lat, lon = latlon([0.0, 1000.0, -250.0], [0.0, -300.0, 400.0])
    x, y = st.xy_m(lat, lon, LAT0, LON0)
    np.testing.assert_allclose(x, [0.0, 1000.0, -250.0], atol=1e-6)   # linear map, fp64 round-off only
    np.testing.assert_allclose(y, [0.0, -300.0, 400.0], atol=1e-6)


def test_flights_frame_keeps_only_eham_departures_of_the_registered_days_and_flags_matched(world):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    PHASE_mvt == DEP filter removed from flights_frame -> RED.
    """
    fl, rank = world["flights"], world["rank"]
    assert len(fl) == 8 * 30 and not fl.MVT_ID_mvt.isin([999_001.0, 999_002.0, 999_003.0]).any()
    assert list(fl.columns) == ["MVT_ID_mvt", "day", "stand", "runway", "callsign", "mvt_s", "ref_s", "matched"]
    assert fl.matched.sum() == 8 * 24 and set(fl.day) == set(st.DAYS)
    r = rank.set_index("MVT_ID_mvt")
    np.testing.assert_allclose(fl.mvt_s.to_numpy(), st.to_epoch_s(r.MVT_TIME_UTC_mvt.loc[fl.MVT_ID_mvt]), atol=1e-6)
    assert fl.callsign[~fl.matched].isna().all() and (fl.callsign[fl.matched].str.startswith("TST")).all()
    with pytest.raises(ValueError, match="need column 'STAND_mvt'"):
        st.flights_frame(rank.drop(columns=["STAND_mvt"]), st.DAYS)


def test_tracks_frame_splits_on_aircraft_callsign_and_gap_and_fills_missing_callsigns():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the gap
    term dropped from the track split -> RED.
    """
    lat, lon = latlon(0.0, 0.0)
    t0 = 1_767_614_400.0
    s = pd.DataFrame({"icao": ["a", "a", "a", "a", "a", "b"], "callsign": [None, "KLM1", "KLM1", "KLM2", "KLM2", "KLM1"],
                      "t": [t0, t0 + 10, t0 + 20, t0 + 30, t0 + 30 + 7200, t0], "lat": lat, "lon": lon,
                      "on_ground": 1, "gs": 0.0})
    tr = st.tracks_frame(s, LAT0, LON0)
    assert tr.track_id.tolist() == [0, 0, 0, 1, 2, 3]
    assert tr.callsign.tolist() == ["KLM1", "KLM1", "KLM1", "KLM2", "KLM2", "KLM1"]     # the None inherits the segment's
    assert tr.x.abs().max() < 1e-6 and tr.y.abs().max() < 1e-6
    with pytest.raises(ValueError, match="samples need"):
        st.tracks_frame(s.drop(columns=["gs"]), LAT0, LON0)
    with pytest.raises(ValueError, match="epoch band"):
        st.tracks_frame(s.assign(t=s.t * 1e6), LAT0, LON0)


LABELS = {"BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"}


def test_the_storm_flights_are_read_without_any_label_column(world, tmp_path, monkeypatch):
    """Nothing may read BLOCK_TIME_UTC_mvt or a label on 2026 rows: load_storm_flights (used by gate AND ship) asks
    parquet for FLIGHT_COLUMNS only, which hold no label. The file written here DOES carry both label columns.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again):
    BLOCK_TIME_UTC_mvt added to FLIGHT_COLUMNS -> RED.
    """
    p = tmp_path / "ranking.parquet"
    world["rank"].to_parquet(p, index=False)
    assert LABELS <= set(pd.read_parquet(p).columns)
    seen, real = [], st.pq.read_table

    def spy(path, *a, columns=None, **kw):
        seen.append(None if columns is None else list(columns))
        return real(path, *a, columns=columns, **kw)

    monkeypatch.setattr(st.pq, "read_table", spy)
    fl = st.load_storm_flights(p)
    ids = st.eligible_unmatched_ids(p)
    assert seen == [st.FLIGHT_COLUMNS, st.FLIGHT_COLUMNS] and not LABELS & set(st.FLIGHT_COLUMNS)
    pd.testing.assert_frame_equal(fl, st.flights_frame(world["rank"], st.DAYS))
    assert ids == set(fl.MVT_ID_mvt[~fl.matched])


def test_sample_span_reports_first_last_hours_and_aircraft():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again):
    sample_span divides by 60 instead of 3600 -> RED.
    """
    t0 = 1_767_571_200.0                                                   # 2026-01-05 00:00:00 UTC
    s = pd.DataFrame({"icao": ["a", "a", "b"], "t": [t0 + 60.0, t0 + 86_000.0, np.nan]})
    assert st.sample_span(s) == "2026-01-05 00:01:00 .. 2026-01-05 23:53:20 UTC (23.9 h, 2 aircraft)"
    assert st.sample_span(s.iloc[:0]) == "(no sample)"
    with pytest.raises(ValueError, match="epoch band"):
        st.sample_span(s.assign(t=s.t * 1e6))


# =============================================================================================
# the matched join, the stand fixes, leave-one-flight-out centroids
# =============================================================================================

def test_join_matched_picks_the_same_callsign_track_in_the_window(world):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    callsign lookup replaced by every track -> RED.
    """
    fl, tr, truth = world["flights"], world["tracks"], world["truth"]
    tof = st.join_matched(fl, tr)
    cs_of_track = tr.groupby("track_id").callsign.first()
    m = fl.matched.to_numpy()
    dead = truth.dead.loc[fl.MVT_ID_mvt].to_numpy()
    assert (tof[~m] == -1).all() and (tof[m & dead] == -1).all() and (tof[m & ~dead] >= 0).all()
    assert (cs_of_track.loc[tof[m & ~dead]].to_numpy() == fl.callsign[m & ~dead].to_numpy()).all()
    # two aircraft carrying the same callsign on the day: the one whose samples sit in the flight's window wins
    lat, lon = latlon(0.0, 0.0)
    t0 = fl.mvt_s.iloc[0]
    two = st.tracks_frame(pd.DataFrame({"icao": ["x"] * 3 + ["y"] * 3, "callsign": "SAME", "t": [t0 - 9 * 3600, t0 - 9 * 3600 + 5, t0 - 9 * 3600 + 10,
                                                                                                 t0 - 600, t0 - 300, t0 - 10],
                                        "lat": lat, "lon": lon, "on_ground": 1, "gs": 0.0}), LAT0, LON0)
    one = fl.iloc[[0]].assign(callsign="SAME")
    assert st.join_matched(one, two).tolist() == [int(two.track_id.iloc[-1])]


def test_stand_fixes_take_the_nearest_on_ground_sample_within_tolerance(world):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the fix
    tolerance check inverted -> RED; the on-ground filter removed from stand_fixes -> RED.
    """
    fl, tr, truth = world["flights"], world["tracks"], world["truth"]
    tof = st.join_matched(fl, tr)
    fx = st.stand_fixes(fl, tr, tof)
    live = fl[fl.matched.to_numpy() & (tof >= 0)]
    assert len(fx) == len(live) and set(fx.MVT_ID_mvt) == set(live.MVT_ID_mvt)
    for r in fx.itertuples(index=False):
        sx, sy = STANDS[r.stand]
        assert abs(r.x - sx) < 0.01 and abs(r.y - sy) < 0.01, "the fix must be the parked position"      # parked, not moving
        assert r.dt_s <= st.FIX_TOL_S
    # a track whose only near-AOBT_3 samples are airborne yields no fix; one beyond the tolerance yields none
    lat, lon = latlon(0.0, 0.0)
    ref = fl.ref_s.iloc[0]
    air = st.tracks_frame(pd.DataFrame({"icao": "z", "callsign": fl.callsign.iloc[0], "t": [ref, ref + 5], "lat": lat, "lon": lon,
                                        "on_ground": 0, "gs": 150.0}), LAT0, LON0)
    assert len(st.stand_fixes(fl.iloc[[0]], air, np.array([0]))) == 0
    far = air.assign(on_ground=1, t=air.t + st.FIX_TOL_S + 1)
    assert len(st.stand_fixes(fl.iloc[[0]], far, np.array([0]))) == 0
    near = air.assign(on_ground=1, t=air.t + st.FIX_TOL_S)
    assert len(st.stand_fixes(fl.iloc[[0]], near, np.array([0]))) == 1


def _fixes(ids, stands, tids, xs, ys):
    return pd.DataFrame({"MVT_ID_mvt": ids, "stand": stands, "track_id": tids, "x": xs, "y": ys, "dt_s": 0.0})


def test_row_centroids_never_use_the_rows_own_fix_or_track_and_need_two_others():
    """Per ROW: the median of the stand's fixes that are neither the row's own MVT_ID nor taken on its own track;
    NaN below two. Rows 11 (matched, no fix of its own) and 12 (unmatched) see every fix at their stand.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the own-
    track exclusion removed from row_centroids -> RED; the own-MVT_ID exclusion removed from row_centroids -> RED;
    >= min_other -> > min_other -> RED.
    """
    fx = _fixes([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], ["A", "A", "A", "A", "B", "B", "C", "C", "C"],
                [101, 102, 103, 104, 105, 106, 107, 108, 109],
                [0.0, 10.0, 0.0, 1000.0, 5.0, 6.0, 1.0, 2.0, 3.0], [0.0, 0.0, 10.0, 1000.0, 5.0, 6.0, 1.0, 2.0, 3.0])
    rows = pd.DataFrame({"MVT_ID_mvt": [4.0, 1.0, 5.0, 7.0, 11.0, 12.0, 13.0, 14.0, 15.0, 9.0],
                         "stand": ["A", "A", "B", "C", "A", "B", "C", "Z", None, "C"]})
    # row 13 is joined to track 107, which is row 7's: it must not see that fix; row 9 is passed with no track, so only
    # its MVT_ID keeps its own fix (x = 3) out
    tof = np.array([104, 101, 105, 107, -1, 777, 107, -1, -1, -1])
    cx, cy, n = st.row_centroids(rows, tof, fx)
    assert (cx[9], cy[9], n[9]) == (1.5, 1.5, 2)             # its own-ID fix excluded without any track to match
    assert (cx[0], cy[0], n[0]) == (0.0, 0.0, 3)             # the outlier does not see itself: median of {0,10,0} x {0,0,10}
    assert (cx[1], cy[1], n[1]) == (10.0, 10.0, 3)           # median of {10, 0, 1000} x {0, 10, 1000}
    assert np.isnan(cx[2]) and np.isnan(cy[2]) and n[2] == 1  # one other flight is not enough
    assert (cx[3], cy[3], n[3]) == (2.5, 2.5, 2)             # exactly two others: allowed
    assert (cx[4], cy[4], n[4]) == (5.0, 5.0, 4)             # a matched row with no fix: every fix at its stand
    assert (cx[5], cy[5], n[5]) == (5.5, 5.5, 2)             # an unmatched row: both B fixes
    assert (cx[6], cy[6], n[6]) == (2.5, 2.5, 2)             # its own track's fix (row 7's, x = 1) is excluded
    assert np.isnan(cx[7]) and n[7] == 0 and np.isnan(cx[8]) and n[8] == 0   # unknown and null stand
    e = st.row_centroids(rows, tof, fx.iloc[:0])
    assert np.isnan(e[0]).all() and (e[2] == 0).all()


def _relabel_stand(rank, flights, stand, k):
    """The first k live matched flights parked at `stand` get the new STAND_mvt 'Z1' (they still park where they did)."""
    cand = flights.MVT_ID_mvt[flights.matched & (flights.stand == stand)].to_numpy()[:k]
    r = rank.copy()
    r["STAND_mvt"] = r.STAND_mvt.astype(object)
    r.loc[r.MVT_ID_mvt.isin(cand), "STAND_mvt"] = "Z1"
    return r, cand


def test_measure_never_lets_a_flight_gate_on_its_own_fix(world):
    """A stand holding exactly two fixes: each flight has ONE other, so neither gets a centroid and neither fires;
    with three, each has two others and all three fire. If a flight could see its own fix, the two-flight stand
    would fire.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again):
    row_centroids keeps every fix at the stand (a flight sees its own fix) -> RED.
    """
    rank, truth, tracks = world["rank"], world["truth"], world["tracks"]
    fl0 = world["flights"]
    live = fl0[fl0.matched & ~truth.dead.loc[fl0.MVT_ID_mvt].to_numpy()]
    for k, want in ((2, False), (3, True)):
        r, ids = _relabel_stand(rank, live, "A1", k)
        est, _ = st.measure(st.flights_frame(r, st.DAYS), tracks, st.DAYS)
        e = est.set_index("MVT_ID_mvt")
        assert len(ids) == k and (e.gate_fired.loc[ids] == want).all(), f"{k} flights at the stand"


def test_measure_gives_a_matched_flight_without_its_own_fix_the_other_flights_centroid(world):
    """A matched flight whose AOBT_3 sits 1,000 s before its first sample has no fix of its own (> FIX_TOL_S); it must
    still be gated on the other flights' centroid, and its raw estimate must be the one it had with a fix.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    draft's defect reinstated: a matched row without its own fix gets no centroid -> RED.
    """
    rank, truth, tracks, est0 = world["rank"].copy(), world["truth"], world["tracks"], world["est"]
    fl0 = world["flights"]
    mids = fl0.MVT_ID_mvt[fl0.matched & ~truth.dead.loc[fl0.MVT_ID_mvt].to_numpy()].to_numpy()[:5]
    sel = rank.MVT_ID_mvt.isin(mids)
    rank.loc[sel, "AOBT_3_flt"] = rank.loc[sel, "AOBT_3_flt"] - pd.Timedelta(seconds=1000)
    fl = st.flights_frame(rank, st.DAYS)
    tof = st.join_matched(fl, tracks)
    assert not st.stand_fixes(fl, tracks, tof).MVT_ID_mvt.isin(mids).any()        # the premise: no fix of their own
    est, _ = st.measure(fl, tracks, st.DAYS)
    e, e0 = est.set_index("MVT_ID_mvt"), est0.set_index("MVT_ID_mvt")
    assert e.gate_fired.loc[mids].all()
    np.testing.assert_allclose(e.offblock_raw_s.loc[mids].to_numpy(), e0.offblock_raw_s.loc[mids].to_numpy(), atol=0.0)


# =============================================================================================
# the gate: 100 m, last sample before take-off
# =============================================================================================

def test_last_sample_in_gate_is_the_last_inside_the_radius_before_take_off():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): d <=
    radius_m -> d < radius_m -> RED; the t <= MVT (before take-off) filter removed -> RED.
    """
    t = np.arange(0.0, 110.0, 10.0)
    tr = pd.DataFrame({"t": t, "x": 2.0 * t, "y": 0.0})                 # 0, 20, ..., 200 m from the origin
    assert st.last_sample_in_gate(tr, 0.0, 0.0, mvt_s=1e9) == 50.0    # 100 m exactly is inside
    assert st.last_sample_in_gate(tr, 0.0, 0.0, mvt_s=30.0) == 30.0   # only samples before take-off count
    assert np.isnan(st.last_sample_in_gate(tr, 0.0, 0.0, mvt_s=-1.0))
    assert np.isnan(st.last_sample_in_gate(tr, 5000.0, 0.0, mvt_s=1e9))
    assert st.last_sample_in_gate(tr, 0.0, 0.0, mvt_s=1e9, radius_m=99.0) == 40.0


def test_raw_estimate_is_the_moment_the_aircraft_leaves_the_stand_circle(world):
    """On the synthetic world the aircraft leaves 100 m of its stand at t0 + 180 + 100/6 s; the raw estimate is the
    last sample before that, so raw - t0 lies in [PUSH_DELAY, PUSH_DELAY + 100/6] within one cadence.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): RADIUS_M
    100 -> 300 -> RED; the t <= MVT filter removed (the draft docstring admits this one is NOT caught here) ->
    RED. The draft said the t <= MVT mutation is not caught here; it is (airborne samples after take-off pass
    within 100 m of a stand on the 24 departure line).
    """
    est, truth = world["est"], world["truth"]
    m = est.matched.to_numpy() & est.gate_fired.to_numpy()
    raw_minus_t0 = est.offblock_raw_s.to_numpy()[m] - truth.t0.loc[est.MVT_ID_mvt[m]].to_numpy()
    assert m.sum() > 150
    assert raw_minus_t0.min() >= PUSH_DELAY_S - CADENCE_S and raw_minus_t0.max() <= PUSH_DELAY_S + 100.0 / SPEED_MS + 1e-6
    assert (est.matched & ~est.gate_fired).sum() == int(truth.dead.loc[est.MVT_ID_mvt[est.matched.to_numpy()]].sum())


# =============================================================================================
# the leave-one-day-out bias
# =============================================================================================

def test_loo_day_bias_is_the_median_of_the_other_days_only():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): day != d
    -> day == d in loo_day_bias -> RED.
    """
    day = np.array(["d1", "d1", "d1", "d2", "d2", "d3", "d3", "d3"])
    err = np.array([100.0, 110.0, 120.0, 500.0, 520.0, 10.0, 20.0, np.nan])
    use = np.array([True, True, True, True, True, True, False, True])       # one d3 row is not matched
    b = st.loo_day_bias(day, err, use)
    assert b == {"d1": pytest.approx(np.median([500.0, 520.0, 10.0])), "d2": pytest.approx(np.median([100.0, 110.0, 120.0, 10.0])),
                 "d3": pytest.approx(np.median([100.0, 110.0, 120.0, 500.0, 520.0]))}    # exact medians of small sets
    assert np.isnan(st.loo_day_bias(np.array(["d1", "d1"]), np.array([1.0, 2.0]), np.array([True, True]))["d1"])
    assert st.pooled_bias(err, use) == pytest.approx(np.median([100.0, 110.0, 120.0, 500.0, 520.0, 10.0]))


def test_measure_applies_the_held_out_days_bias_to_every_row_of_that_day(world):
    """Matched AND unmatched rows of day d carry the median raw error of matched rows on the OTHER days (the prereg's
    estimator; the unmatched rows ship, so they must carry the bias C1 validated). Then one day's matched raw errors are
    shifted +1000 s: that day's own bias must not move and every other day's must rise.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    draft's rule reinstated: unmatched rows get the pooled bias -> RED; matched rows given the pooled bias -> RED;
    the draft's centroid defect reinstated (the original failure 1) -> RED.
    """
    fl, tr = world["flights"], world["tracks"]
    est0, info0 = world["est"], world["info"]
    m = est0.matched.to_numpy()
    for d in st.DAYS:
        on_d = (est0.day == d).to_numpy() & est0.gate_fired.to_numpy()
        others = m & (est0.day != d).to_numpy() & np.isfinite(info0["err_raw"])
        want = np.median(info0["err_raw"][others])
        assert (m & on_d).sum() > 5 and (~m & on_d).sum() > 0
        np.testing.assert_allclose(est0.bias_s.to_numpy()[on_d], want, atol=0.0)   # the same float, not a recomputation
    # the premise that makes the equality above bite: on most days the LOO median differs from the pooled one, so giving
    # any class of row the pooled bias (the draft's unmatched rule) is caught
    assert sum(info0["bias_by_day_s"][d] != info0["pooled_bias_s"] for d in st.DAYS) >= 6
    # shift one day's raw errors by +1000 s (its flights push back 1000 s later than everyone): that day's own
    # bias must not move (it is the other days' median) while every other day's LOO bias moves by the median shift
    d0 = st.DAYS[3]
    tr2 = tr.copy()
    ids_d0 = set(fl.MVT_ID_mvt[(fl.day == d0)])
    cs_d0 = set(fl.callsign[fl.MVT_ID_mvt.isin(ids_d0) & fl.matched])
    tr2.loc[tr2.callsign.isin(cs_d0), "t"] += 1000.0
    fl2 = fl.copy()
    fl2.loc[fl2.day == d0, "mvt_s"] += 1000.0
    est2, info2 = st.measure(fl2, tr2, st.DAYS)
    # premise: d0's flights lost their own fixes (AOBT_3 now > FIX_TOL_S from every sample) yet stay gated on the other
    # flights' centroids, their raw errors exactly +1000 s (float64 sums of ~1.7e9 s values: 1e-6 s is round-off)
    md0 = fl.matched.to_numpy() & (fl.day == d0).to_numpy()
    assert np.isfinite(info2["err_raw"][md0]).sum() == np.isfinite(info0["err_raw"][md0]).sum() > 15
    ok = np.isfinite(info0["err_raw"]) & md0
    np.testing.assert_allclose(info2["err_raw"][ok], info0["err_raw"][ok] + 1000.0, atol=1e-6)
    assert info2["bias_by_day_s"][d0] == pytest.approx(info0["bias_by_day_s"][d0], abs=1e-9)
    for d in st.DAYS:
        if d != d0:
            assert info2["bias_by_day_s"][d] > info0["bias_by_day_s"][d]


# =============================================================================================
# the unmatched join: runway + take-off second
# =============================================================================================

def test_takeoff_events_first_fast_sample_after_ground_with_heading(world):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    preceding-on-ground requirement removed -> RED.
    """
    tr = world["tracks"]
    ev = st.takeoff_events(tr)
    assert len(ev) == tr.track_id.nunique()                       # every synthetic track departs
    for r in ev.itertuples(index=False):
        g = tr[tr.track_id == r.track_id]
        fast = g[g.gs > st.TAKEOFF_GS_KT]
        assert r.t == fast.t.iloc[0] and g[g.t < r.t].on_ground.eq(1).any()
        assert np.hypot(r.ux, r.uy) == pytest.approx(1.0, abs=1e-9)
    # an arriving track: fast and airborne from its first sample, then on the ground -> no event
    lat, lon = latlon(np.array([0.0, 100.0, 200.0]), np.array([0.0, 0.0, 0.0]))
    arr = st.tracks_frame(pd.DataFrame({"icao": "arr", "callsign": "ARR1", "t": [1.7e9, 1.7e9 + 5, 1.7e9 + 10], "lat": lat, "lon": lon,
                                        "on_ground": [0, 0, 1], "gs": [140.0, 120.0, 60.0]}), LAT0, LON0)
    assert len(st.takeoff_events(arr)) == 0
    dep = arr.assign(on_ground=[1, 1, 0], gs=[10.0, 90.0, 140.0])
    e = st.takeoff_events(dep)
    assert len(e) == 1 and e.t.iloc[0] == 1.7e9 + 5 and (e.ux.iloc[0], e.uy.iloc[0]) == (1.0, 0.0)


def test_learn_and_label_runways_from_matched_events(world):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    heading term removed from label_runway -> RED; the along-track bound removed (the draft's infinite line) ->
    RED.
    """
    fl, tr, truth = world["flights"], world["tracks"], world["truth"]
    tof = st.join_matched(fl, tr)
    ev = st.takeoff_events(tr).set_index("track_id")
    m = fl.matched.to_numpy() & (tof >= 0)
    e = ev.loc[tof[m]].reset_index()
    rw = st.learn_runways(e, fl.runway.to_numpy()[m])
    assert set(rw.runway) == {"36L", "24"} and (rw.n >= st.MIN_RUNWAY_EVENTS).all()
    for r in rw.itertuples(index=False):
        (px, py), (ux, uy) = RUNWAYS[r.runway]
        assert abs(r.ux * ux + r.uy * uy - 1.0) < 1e-6 and abs(r.ux * (py - r.my) - r.uy * (px - r.mx)) < 1.0   # on the line
    lab = st.label_runway(e, rw)
    assert (lab == fl.runway.to_numpy()[m]).all()
    (px, py), (ux, uy) = RUNWAYS["36L"]
    r24 = rw.set_index("runway").loc["24"]
    # probe points; 36L's line is x = -1500 heading north, 24's is y = x heading south-west
    #   1  on 36L's line, 36L heading                                   -> 36L
    #   2  151 m east of 36L's line                                     -> None (lateral)
    #   3  on 36L's line, OPPOSITE heading, 283 m off 24's line         -> None (heading: it would be 18R)
    #   4  5 m off 36L's line                                           -> 36L
    #   5  on 24's line, 24 heading, 5.2 km past 24's learned point     -> None (along: the draft's infinite line said 24)
    #   6  on 24's line, 24 heading, 3.7 km from 24's learned point     -> 24
    far = 5_245.0
    near = st.RUNWAY_ALONG_M - 100.0
    probe = pd.DataFrame({"track_id": [1, 2, 3, 4, 5, 6], "t": 0.0,
                          "x": [px, px + st.RUNWAY_LATERAL_M + 1.0, px, px + 5.0, r24.mx + far * r24.ux, r24.mx + near * r24.ux],
                          "y": [py + 500.0, py + 500.0, py + 900.0, py + 700.0, r24.my + far * r24.uy, r24.my + near * r24.uy],
                          "ux": [ux, ux, -ux, ux, r24.ux, r24.ux], "uy": [uy, uy, -uy, uy, r24.uy, r24.uy]})
    assert st.label_runway(probe, rw).tolist() == ["36L", None, None, "36L", None, "24"]
    assert st.label_runway(probe, rw.iloc[:0]).tolist() == [None] * 6
    assert len(st.learn_runways(e.iloc[:st.MIN_RUNWAY_EVENTS - 1], fl.runway.to_numpy()[m][:st.MIN_RUNWAY_EVENTS - 1])) == 0


def test_join_unmatched_counts_ambiguity_and_conflicts():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    conflict term removed -> RED; the window edge <= -> < -> RED.
    """
    ev = pd.DataFrame({"track_id": [10, 11, 12, 13, 14], "t": [1000.0, 1030.0, 5000.0, 9000.0, 20000.0],
                       "runway": ["36L", "36L", "24", "36L", None]})
    fl = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "runway": ["36L", "24", "36L", "36L", "36L", "09"],
                       "mvt_s": [1015.0, 5000.0 + 15.0, 9010.0, 9020.0, 20000.0 + 15.0, 1015.0]})
    j = st.join_unmatched(fl, ev, offset_s=-15.0, tol_s=60.0).set_index("MVT_ID_mvt")
    assert j.n_candidates.tolist() == [2, 1, 1, 1, 0, 0]
    assert j.ambiguous.tolist() == [True, False, True, True, False, False]
    assert j.track_id.tolist() == [-1, 12, -1, -1, -1, -1]
    c0 = st.clause_c0(j.reset_index())
    assert c0["n_with_candidate"] == 4 and c0["n_ambiguous_or_unmatched"] == 3 and c0["share"] == 0.75 and not c0["pass"]
    assert c0["n_no_candidate"] == 2 and c0["no_candidate_share_of_rows"] == pytest.approx(2 / 6)
    # the window edge: exactly tol_s away is a candidate
    edge = st.join_unmatched(fl.iloc[[1]].assign(mvt_s=5000.0 + 15.0 + 60.0), ev, -15.0, 60.0)
    assert edge.n_candidates.iloc[0] == 1
    assert st.join_unmatched(fl.iloc[[1]].assign(mvt_s=5000.0 + 15.0 + 60.5), ev, -15.0, 60.0).n_candidates.iloc[0] == 0


def test_measure_joins_unmatched_rows_by_runway_and_take_off_second_and_recovers_their_taxi(world):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): tracks
    joined to matched flights left in the candidate pool -> RED. The draft claimed this RED; it SURVIVED (the
    synthetic world never puts two take-offs within 60 s on one runway) until the constructed same-runway
    collision was added.
    """
    est, info, truth = world["est"], world["info"], world["truth"]
    um = ~est.matched.to_numpy()
    assert info["takeoff_offset_s"] == pytest.approx(-15.0, abs=CADENCE_S)       # 80 kt at roll + 20 s, take-off at +35
    j = info["joined"]
    assert len(j) == um.sum() and (j.n_candidates >= 1).mean() > 0.9 and j.ambiguous.mean() < 0.15
    fired = um & est.gate_fired.to_numpy()
    assert fired.sum() > 0.7 * um.sum()
    true_taxi = (truth.mvt - truth.t0).loc[est.MVT_ID_mvt[fired]].to_numpy()
    err = est.taxi_hat.to_numpy()[fired] - true_taxi
    # the sensor is calibrated to AOBT_3, which the world puts 60 s before the true off-block: the estimate runs
    # 60 s long, within the cadence; a wrong join would be off by minutes
    assert abs(np.median(err) - 60.0) <= CADENCE_S and np.abs(err - 60.0).max() <= 3 * CADENCE_S
    assert (est.track_id.to_numpy()[fired] >= 0).all() and not est.join_ambiguous.to_numpy()[fired].any()
    assert (est.n_candidates.to_numpy()[est.join_ambiguous.to_numpy() & um] >= 1).all()
    # the joined track is the row's own aircraft
    cs = world["tracks"].groupby("track_id").callsign.first()
    own = np.array([f"UNM{int(i) - 100_000:04d}" for i in est.MVT_ID_mvt[fired]])
    assert (cs.loc[est.track_id[fired]].to_numpy() == own).all()
    # a matched flight taking off on the same runway 10 s after an unmatched one: its track is already joined to the
    # matched row, so it is NOT a candidate and the unmatched row still has exactly its own track (the synthetic world
    # alone never puts two take-offs that close, which is why the draft's claimed rehearsal of this rule survived)
    fl, tr = world["flights"], world["tracks"]
    ev = st.takeoff_events(tr).set_index("track_id")
    e = est.set_index("MVT_ID_mvt")
    u = float(est.MVT_ID_mvt[fired].iloc[0])
    tid_u, rw_u = int(e.track_id.loc[u]), e.runway.loc[u]
    m_ids = est.MVT_ID_mvt[est.matched & est.gate_fired & (est.runway == rw_u)].to_numpy()
    mid = float(m_ids[0])
    tid_m = int(e.track_id.loc[mid])
    shift = float(ev.t.loc[tid_u]) + 10.0 - float(ev.t.loc[tid_m])
    tr2 = tr.copy()
    tr2.loc[tr2.track_id == tid_m, "t"] += shift
    fl2 = fl.copy()
    sel = fl2.MVT_ID_mvt == mid
    fl2.loc[sel, "mvt_s"] += shift
    fl2.loc[sel, "ref_s"] += shift
    est2, _ = st.measure(fl2, tr2, st.DAYS)
    e2 = est2.set_index("MVT_ID_mvt")
    assert int(e2.track_id.loc[mid]) == tid_m and e2.gate_fired.loc[mid]                 # still the matched row's track
    assert e2.n_candidates.loc[u] == 1 and not e2.join_ambiguous.loc[u] and int(e2.track_id.loc[u]) == tid_u
    assert e2.gate_fired.loc[u]


# =============================================================================================
# clauses: arithmetic, thresholds, the verdict
# =============================================================================================

def test_clause_c1_arithmetic_and_boundaries():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): r <=
    rmse_max -> r < rmse_max -> RED; cov >= cov_min -> cov > cov_min -> RED.
    """
    matched = np.array([True] * 4 + [False] * 2)
    err = np.array([250.0, -250.0, np.nan, np.nan, 5.0, np.nan])          # unmatched rows never count
    c = st.clause_c1(err, matched)
    assert c["n_matched"] == 4 and c["n_gated"] == 2 and c["coverage"] == 0.5 and c["rmse_s"] == 250.0 and c["pass"]
    assert not st.clause_c1(np.array([250.01, -250.0, np.nan, np.nan, 5.0, np.nan]), matched)["pass"]
    assert not st.clause_c1(np.array([1.0, np.nan, np.nan, np.nan, 5.0, np.nan]), matched)["pass"]     # coverage 0.25
    assert st.clause_c1(np.array([1.0, 1.0, np.nan, np.nan, 5.0, np.nan]), matched, cov_min=0.5)["pass"]
    assert not st.clause_c1(np.array([1.0, 1.0, np.nan, np.nan, 5.0, np.nan]), matched, cov_min=0.51)["pass"]
    assert not st.clause_c1(np.full(6, np.nan), matched)["pass"]
    assert c["rmse_threshold_s"] == 250.0 and c["coverage_threshold"] == 0.5


def test_clause_c2_counts_days_within_300_s_and_needs_six_of_eight():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): ok >=
    min_days -> ok > min_days -> RED; the matched mask dropped from clause_c2's per-day RMSE -> RED.
    """
    days = st.DAYS
    day = np.repeat(days, 2)
    matched = np.ones(16, dtype=bool)
    err = np.zeros(16)
    err[0:2] = 300.0                       # day 1 exactly at the bar: within
    err[2:4] = 300.5                       # day 2 out
    err[4:6] = np.nan                      # day 3 no gated row: out
    c = st.clause_c2(day, err, matched, days)
    assert c["n_days_within"] == 6 and c["pass"] and c["per_day"][days[0]]["rmse_s"] == 300.0
    assert np.isnan(c["per_day"][days[2]]["rmse_s"]) and c["per_day"][days[2]]["n_gated"] == 0 and c["per_day"][days[2]]["n_matched"] == 2
    err[6:8] = 1000.0
    assert not st.clause_c2(day, err, matched, days)["pass"]
    # unmatched rows never count: day 4 gets two unmatched rows 1,000 s out beside matched rows at 0 (the draft's
    # version emptied day 4 of matched rows instead, which correctly counts as out -- the test contradicted itself)
    err[6:8] = 0.0
    day2 = np.concatenate([day, [days[3], days[3]]])
    err2 = np.concatenate([err, [1000.0, 1000.0]])
    matched2 = np.concatenate([matched, [False, False]])
    c2 = st.clause_c2(day2, err2, matched2, days)
    assert c2["pass"] and c2["n_days_within"] == 6 and c2["per_day"][days[3]]["rmse_s"] == 0.0
    assert c2["per_day"][days[3]]["n_matched"] == 2
    # a day with no matched row at all has no RMSE and is out
    matched2[6:8] = False
    c2 = st.clause_c2(day2, err2, matched2, days)
    assert not c2["pass"] and c2["n_days_within"] == 5 and np.isnan(c2["per_day"][days[3]]["rmse_s"])


def test_clause_c3_is_a_strict_beat_of_the_model_rmse():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): < -> <=
    -> RED.
    """
    assert st.clause_c3(173.1, 0.6, 300)["pass"] and not st.clause_c3(173.2, 0.6, 300)["pass"]
    assert not st.clause_c3(1.0, 0.0, 0)["pass"]
    assert st.clause_c3(100.0, 0.5, 10)["model_rmse_s"] == 173.2 and st.MODEL_RMSE_EHAM_S == 173.2


def test_clause_c0_boundary_and_empty():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): share <=
    max_share -> < -> RED.
    """
    j = pd.DataFrame({"n_candidates": [1] * 20, "ambiguous": [True] + [False] * 19})
    assert st.clause_c0(j)["share"] == 0.05 and st.clause_c0(j)["pass"]
    j2 = pd.DataFrame({"n_candidates": [1] * 19, "ambiguous": [True] + [False] * 18})
    assert not st.clause_c0(j2)["pass"]
    assert not st.clause_c0(pd.DataFrame({"n_candidates": [0, 0], "ambiguous": [False, False]}))["pass"]
    assert not st.clause_c0(pd.DataFrame({"n_candidates": [], "ambiguous": []}))["pass"]


def test_verdict_mapping():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the C1 ->
    NOT WORKING branch removed -> RED.
    """
    ok, no = {"pass": True}, {"pass": False}
    assert st.verdict(ok, ok, ok, ok) == "WORKING"
    assert st.verdict(no, no, no, no) == "NOT WORKING" and st.verdict(ok, no, ok, ok) == "NOT WORKING"
    assert st.verdict(no, ok, ok, ok) == "INCONCLUSIVE" and st.verdict(ok, ok, no, ok) == "INCONCLUSIVE"
    assert st.verdict(ok, ok, ok, no) == "INCONCLUSIVE"
    assert set(st.DAYS) == {f"2026-01-0{d}" for d in range(2, 10)} and (st.RADIUS_M, st.MIN_OTHER_FLIGHTS) == (100.0, 2)
    assert (st.C1_RMSE_S, st.C1_MIN_COVERAGE, st.C2_RMSE_S, st.C2_MIN_DAYS, st.C0_MAX_SHARE, st.TAKEOFF_GS_KT) == (250.0, 0.5, 300.0, 6, 0.05, 80.0)


def test_gate_report_on_the_synthetic_world_is_working_and_consistent(world):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): per-day
    coverage computed over all rows of the day -> RED.
    """
    est, rep = st.gate_report(world["flights"], world["tracks"], st.clause_c3(150.0, 0.6, 300))
    c = rep["clauses"]
    assert rep["verdict"] == "WORKING" and c["C1"]["pass"] and c["C2"]["pass"] and c["C0"]["pass"]
    assert c["C1"]["rmse_s"] < 3 * CADENCE_S and 0.8 < c["C1"]["coverage"] < 1.0
    assert list(est.columns) == st.ESTIMATE_COLUMNS and len(est) == 240
    for d in st.DAYS:
        r, m = rep["per_day"][d], (est.day == d).to_numpy()
        assert r["n_matched"] == int((m & est.matched).sum()) and r["n_gated"] == int((m & est.matched & est.gate_fired).sum())
        assert r["coverage"] == pytest.approx(r["n_gated"] / r["n_matched"])
        assert r["n_unmatched"] == int((m & ~est.matched).sum())
    assert rep["n_unmatched_gate_fired"] == int((est.gate_fired & ~est.matched).sum()) and rep["judgement_calls"] == st.JUDGEMENT_CALLS
    assert rep["eligibility"] == st.ELIGIBILITY_RULE and rep["days"] == list(st.DAYS)
    fired = est.gate_fired.to_numpy()
    mvt = world["flights"].mvt_s.to_numpy()
    np.testing.assert_allclose(est.taxi_hat.to_numpy()[fired], mvt[fired] - st.to_epoch_s(est.offblock_hat)[fired], atol=1e-3)  # ns stamps
    assert est.offblock_hat.isna().to_numpy().tolist() == (~fired).tolist()
    json.dumps(st._json_ready(rep))                                            # serialisable, NaN -> null


def test_census_c3_scores_taxi_hat_against_the_label_and_reports_the_block_diagnostic(world):
    """On a synthetic 2025 day.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the label
    read as MVT - AOBT_3 instead of TAXITIME_SEC_mvt -> RED; the 2025-only guard removed -> RED.
    """
    d = st.CENSUS_DAY
    rank, _, samples = synthetic_world(seed=3, days=[d])
    with pytest.raises(ValueError, match="2025 days only"):
        st.census_c3(world["rank"], world["samples"], day=st.DAYS[0])          # a 2026 day: labels must not be read
    c3 = st.census_c3(rank, samples, day=d)
    assert c3["day"] == d and c3["n_matched"] == 24 and c3["n_gated"] > 15
    # calibrated to AOBT_3 (60 s before the true off-block) the taxi estimate runs 60 s long against the label
    assert 60.0 - CADENCE_S <= c3["rmse_gated_vs_label_s"] <= 60.0 + 2 * CADENCE_S
    assert c3["rmse_vs_aobt3_s"] < 2 * CADENCE_S
    assert c3["diagnostic_block_calibrated"]["rmse_vs_label_s"] < 2 * CADENCE_S
    assert c3["diagnostic_median_aobt3_minus_block_s"] == -60.0
    assert c3["diagnostic_block_calibrated"]["bias_s"] == pytest.approx(c3["bias_at_aobt3_s"] - 60.0, abs=1e-6)
    assert c3["pass"] is (c3["rmse_gated_vs_label_s"] < st.MODEL_RMSE_EHAM_S)
    with pytest.raises(ValueError, match="census truth needs 'TAXITIME_SEC_mvt'"):
        st.census_c3(rank.drop(columns=["TAXITIME_SEC_mvt"]), samples, day=d)


# =============================================================================================
# ship
# =============================================================================================

@pytest.fixture(scope="module")
def shipworld(world, tmp_path_factory):
    """A base submission of int32 over the 240 storm rows plus 60 matched-elsewhere rows, the estimates table and a
    WORKING report on disk, a synthetic ranking parquet and a template."""
    rng = np.random.default_rng(11)
    est = world["est"]
    other = np.arange(5_000_000, 5_000_060, dtype="float64")
    ids = np.concatenate([est.MVT_ID_mvt.to_numpy(), other])
    base = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": rng.integers(300, 2000, len(ids)).astype("int32")})
    template = base[["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=0)
    d = tmp_path_factory.mktemp("e4")
    base_path, rank_path, tmpl_path = d / "merry-quicksand_v10.parquet", d / "ranking.parquet", d / "submitting.parquet"
    base.to_parquet(base_path, index=False)
    world["rank"].to_parquet(rank_path, index=False)
    template.to_parquet(tmpl_path, index=False)
    est_path, rep_path = d / "adsb_storm_estimates.parquet", d / "adsb_storm.json"
    est.to_parquet(est_path, index=False)
    _, rep = st.gate_report(world["flights"], world["tracks"], st.clause_c3(150.0, 0.6, 300))
    rep_path.write_text(json.dumps(st._json_ready(rep)))
    eligible = st.eligible_unmatched_ids(rank_path)
    # the detail file of the version that last wrote the unmatched rows: every EHAM unmatched storm row plus others
    b = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    det_ids = np.concatenate([np.asarray(sorted(eligible)), other[:10]])
    detail = pd.DataFrame({"MVT_ID_mvt": det_ids, "ADEP_mvt": "EHAM", "before": 1.0, "after": b.loc[det_ids].to_numpy(dtype="float64")})
    detail_path = d / "merry-quicksand_v9.unm_rows.parquet"
    detail.to_parquet(detail_path, index=False)
    (d / "LICENCE_adsb.md").write_text("Derived from adsb.lol globe_history, ODbL-1.0, share-alike.\n")
    return dict(base=base, template=template, dir=d, base_path=base_path, rank_path=rank_path, tmpl_path=tmpl_path,
                est_path=est_path, rep_path=rep_path, eligible=eligible, est=est, rep=rep, detail=detail,
                detail_path=detail_path, licence="LICENCE_adsb.md")


def _ship(w, version, confirmed, sub_dir, **kw):
    args = dict(detail=str(w["detail_path"]), licence_doc=w["licence"], report_path=w["rep_path"], estimates_path=w["est_path"],
                sub_dir=sub_dir, ranking_path=w["rank_path"], template_path=w["tmpl_path"], root=w["dir"])
    args.update(kw)
    return st.step_ship(str(w["base_path"]), version, confirmed, **args)


def test_ship_rows_are_gated_unambiguous_unmatched_rows_of_the_registered_days_floored_at_one(shipworld):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    ~matched term removed from ship_rows -> RED; the floor at 1 removed -> RED.
    """
    est = shipworld["est"]
    r = st.ship_rows(est)
    want = (~est.matched) & est.gate_fired & ~est.join_ambiguous
    assert len(r) == int(want.sum()) and set(r.MVT_ID_mvt) == set(est.MVT_ID_mvt[want])
    np.testing.assert_array_equal(r.after.to_numpy(), np.maximum(np.rint(r.taxi_hat.to_numpy()), 1.0))
    tiny = est.copy()
    i = int(np.flatnonzero(want.to_numpy())[0])
    tiny.loc[i, "taxi_hat"] = -40.0
    assert st.ship_rows(tiny).set_index("MVT_ID_mvt").after.loc[est.MVT_ID_mvt.iloc[i]] == 1.0
    assert len(st.ship_rows(est, days=["2026-01-15"])) == 0
    bad = est.copy()
    bad.loc[i, "taxi_hat"] = np.nan
    with pytest.raises(AssertionError, match="non-finite taxi_hat"):
        st.ship_rows(bad)


def test_build_ship_changes_only_the_intended_rows_preserves_dtype_and_does_not_mutate_the_base(shipworld):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): subset
    guard disabled AND the last (non-storm) row edited -> RED; the output cast to int64 after the splice -> RED;
    the splice written into the base in place -> RED.
    """
    base, est, eligible = shipworld["base"], shipworld["est"], shipworld["eligible"]
    snap = base.copy()
    out, info = st.build_ship(base, est, eligible)
    pd.testing.assert_frame_equal(base, snap)
    rows = st.ship_rows(est).set_index("MVT_ID_mvt")
    o, b = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt, base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    changed = set(o.index[(o != b).to_numpy()])
    assert changed <= set(rows.index) and len(changed) > 0.9 * len(rows)
    np.testing.assert_array_equal(o.loc[rows.index].to_numpy(), rows.after.to_numpy().astype("int32"))
    untouched = [i for i in b.index if i not in rows.index]
    np.testing.assert_array_equal(o.loc[untouched].to_numpy(), b.loc[untouched].to_numpy())
    assert out.TAXITIME_SEC_mvt.dtype == np.dtype("int32") and out.MVT_ID_mvt.dtype == base.MVT_ID_mvt.dtype
    assert list(out.columns) == list(base.columns) and out.MVT_ID_mvt.tolist() == base.MVT_ID_mvt.tolist()
    bs.check_submission(out, shipworld["template"])
    assert info["n_rows_replaced"] == len(rows) and info["n_values_differ"] == len(changed)
    d = info["detail"]
    assert list(d.columns) == st.DETAIL_COLUMNS and len(d) == len(rows)
    shift = d.after.to_numpy() - d.before.to_numpy()
    assert info["sse_shift_vs_base_board_mse"] == pytest.approx((shift ** 2).sum() / len(base), rel=1e-12)  # integers, fp64 sum
    assert info["rms_shift_vs_base_s"] == pytest.approx(np.sqrt((shift ** 2).mean()), rel=1e-12)
    assert sum(info["per_day_n"].values()) == len(rows) and set(info["per_day_n"]) <= set(st.DAYS)


def test_build_ship_refuses_rows_outside_the_eligible_set_and_an_empty_ship(shipworld):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    eligibility subset check disabled -> RED.
    """
    base, est, eligible = shipworld["base"], shipworld["est"], shipworld["eligible"]
    forged = est.copy()
    i = int(np.flatnonzero(est.matched.to_numpy() & est.gate_fired.to_numpy())[0])
    forged.loc[i, "matched"] = False
    with pytest.raises(AssertionError, match="not EHAM unmatched departures of the registered days"):
        st.build_ship(base, forged, eligible)
    with pytest.raises(AssertionError, match="no row to ship"):
        st.build_ship(base, est.assign(gate_fired=False), eligible)
    with pytest.raises(ValueError, match="not in the base"):
        st.build_ship(base.iloc[:-100].reset_index(drop=True), est, eligible)


def test_write_ship_outputs_round_trips_and_refuses_to_overwrite(shipworld, tmp_path):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    dest.exists() refusal disabled -> RED; check_submission on the frame (before writing) removed -> RED.
    """
    base, est, eligible, rep = shipworld["base"], shipworld["est"], shipworld["eligible"], shipworld["rep"]
    out, info = st.build_ship(base, est, eligible)
    dest, meta = st.write_ship_outputs(out, info, shipworld["template"], 99, "merry-quicksand_v10.parquet", tmp_path, rep, 0.0)
    assert dest == tmp_path / "merry-quicksand_v99.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(dest), out)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "merry-quicksand_v99.adsb_rows.parquet"), info["detail"])
    disk = json.loads((tmp_path / "merry-quicksand_v99.meta.json").read_text())
    assert disk == st._json_ready(meta) and meta["version"] == 99 and meta["base"] == "merry-quicksand_v10.parquet"
    assert meta["eligibility_confirmed_by_owner"] is True and meta["eligibility_rule"] == st.ELIGIBILITY_RULE
    assert meta["gate_verdict"] == "WORKING" and meta["n_rows_replaced"] == info["n_rows_replaced"] and "ODbL" in meta["licence"]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        st.write_ship_outputs(out, info, shipworld["template"], 99, "x", tmp_path, rep, 0.0)
    with pytest.raises(ValueError, match="row count"):
        st.write_ship_outputs(out.iloc[:-1], info, shipworld["template"], 98, "x", tmp_path, rep, 0.0)
    assert not (tmp_path / "merry-quicksand_v98.parquet").exists()


def test_step_ship_refuses_without_the_eligibility_flag_and_prints_the_rule(shipworld, tmp_path, capsys):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    eligibility-flag check removed -> RED.
    """
    w = shipworld
    with pytest.raises(SystemExit, match="eligibility-confirmed"):
        _ship(w, 77, False, tmp_path)
    assert "NO SUBMISSION MAY DEPEND ON THIS ARM" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(SystemExit, match="eligibility-confirmed"):
        st.main(["ship", "--base", str(w["base_path"]), "--version", "77", "--detail", str(w["detail_path"]),
                 "--licence-doc", str(w["dir"] / w["licence"])])
    assert not (ROOT / "submissions" / "merry-quicksand_v77.parquet").exists()


def test_step_ship_refuses_without_the_licence_documented_in_the_repo(shipworld, tmp_path):
    """The prereg: the derived table's ODbL licence is documented in the repo BEFORE any ship. A missing file, a file
    outside the repo and a file that does not name ODbL + adsb.lol each refuse, and nothing is written.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again):
    check_licence_doc bypassed in step_ship -> RED; the inside-the-repo check disabled -> RED; the ODbL / adsb.lol
    text check disabled -> RED.
    """
    w = shipworld
    (w["dir"] / "NOTES.md").write_text("adsb.lol traces, licence to be decided\n")
    outside = tmp_path / "LICENCE.md"
    outside.write_text("adsb.lol ODbL-1.0\n")
    for doc, msg in (("MISSING.md", "does not exist"), ("NOTES.md", "does not document"), (str(outside), "not inside the repository")):
        with pytest.raises(SystemExit, match=msg):
            _ship(w, 76, True, tmp_path / "sub", licence_doc=doc)
    assert not (tmp_path / "sub").exists()
    assert st.check_licence_doc(w["licence"], w["dir"]) == "LICENCE_adsb.md"


def test_verify_base_is_the_rebuild_guard(shipworld):
    """The base must BE the version the detail file describes on every EHAM unmatched storm row.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the value
    comparison disabled -> RED; the missing-from-detail check disabled -> RED.
    """
    base, detail, eligible = shipworld["base"], shipworld["detail"], shipworld["eligible"]
    assert st.verify_base(base, detail, eligible) == {"n_checked": len(eligible)}
    moved = base.copy()
    i = int(np.flatnonzero(moved.MVT_ID_mvt.isin(eligible).to_numpy())[0])
    moved.loc[i, "TAXITIME_SEC_mvt"] += 1
    with pytest.raises(AssertionError, match="differs from the detail file's `after` on 1 of"):
        st.verify_base(moved, detail, eligible)
    short = detail[detail.MVT_ID_mvt != moved.MVT_ID_mvt.iloc[i]]
    with pytest.raises(AssertionError, match="1 of .* are not in the detail file"):
        st.verify_base(base, short, eligible)
    with pytest.raises(AssertionError, match="repeats"):
        st.verify_base(base, pd.concat([detail, detail.iloc[:1]]), eligible)
    with pytest.raises(ValueError, match="needs 'after'"):
        st.verify_base(base, detail.drop(columns=["after"]), eligible)
    with pytest.raises(AssertionError, match="no target row"):
        st.verify_base(base, detail, set())
    # a base row OUTSIDE the target set may differ from the detail freely: the detail file carries other airports' rows
    other = base.copy()
    j = int(np.flatnonzero(other.MVT_ID_mvt.isin(detail.MVT_ID_mvt).to_numpy() & ~other.MVT_ID_mvt.isin(eligible).to_numpy())[0])
    other.loc[j, "TAXITIME_SEC_mvt"] += 1
    assert st.verify_base(other, detail, eligible) == {"n_checked": len(eligible)}


def test_step_ship_refuses_a_base_that_fails_the_rebuild_guard(shipworld, tmp_path):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again):
    verify_base bypassed in step_ship -> RED.
    """
    w = shipworld
    det = w["detail"].copy()
    det.loc[0, "after"] = det.loc[0, "after"] + 7.0
    p = tmp_path / "stale.unm_rows.parquet"
    det.to_parquet(p, index=False)
    with pytest.raises(AssertionError, match="not the version it claims to be"):
        _ship(w, 75, True, tmp_path / "sub", detail=str(p))
    assert not (tmp_path / "sub").exists()


def test_step_ship_refuses_a_non_working_verdict_and_ships_with_the_flag(shipworld, tmp_path, capsys):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    verdict check disabled -> RED.
    """
    w = shipworld
    bad = tmp_path / "inconclusive.json"
    bad.write_text(json.dumps({**json.loads(w["rep_path"].read_text()), "verdict": "INCONCLUSIVE"}))
    with pytest.raises(SystemExit, match="verdict is 'INCONCLUSIVE'"):
        _ship(w, 78, True, tmp_path, report_path=bad)
    assert not (tmp_path / "merry-quicksand_v78.parquet").exists()
    assert _ship(w, 78, True, tmp_path) == 0
    out = pd.read_parquet(tmp_path / "merry-quicksand_v78.parquet")
    want, _ = st.build_ship(w["base"], w["est"], w["eligible"])
    pd.testing.assert_frame_equal(out, want)
    assert out.TAXITIME_SEC_mvt.dtype == np.dtype("int32")
    pd.testing.assert_frame_equal(pd.read_parquet(w["base_path"]), w["base"])           # the base file is untouched
    meta = json.loads((tmp_path / "merry-quicksand_v78.meta.json").read_text())
    assert meta["rebuild_guard"] == {"n_checked": len(w["eligible"]), "detail_file": "merry-quicksand_v9.unm_rows.parquet"}
    assert meta["licence_doc"] == "LICENCE_adsb.md" and meta["transfer_rule"] == st.TRANSFER_RULE
    assert meta["expected_price"]["conservative_event_excluded_board_mse"] == 800.0 and meta["expected_price"]["full_board_mse"] == 1800.0
    assert "ELIGIBILITY" in capsys.readouterr().out
