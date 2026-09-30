"""Tests for scripts/adsb_features.py — take-off-anchored join and label-free per-flight ADS-B features.

Written before the module (red first). Each test names the mutation or defect it must catch.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("adsb_features", ROOT / "scripts" / "adsb_features.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adsb_features"] = mod
    spec.loader.exec_module(mod)
    return mod


fx = _load()

LAT0, LON0 = 52.3086, 4.7639          # EHAM reference
M_PER_DEG_LAT = 110_540.0


def _ll(dx_m, dy_m):
    """(lat, lon) of a point dx_m east, dy_m north of the EHAM reference."""
    return LAT0 + dy_m / M_PER_DEG_LAT, LON0 + dx_m / (111_320.0 * np.cos(np.radians(LAT0)))


def _samples(icao, callsign, pts, airport="EHAM"):
    """pts: list of (t, dx_m, dy_m, on_ground, gs)."""
    rows = []
    for t, dx, dy, g, gs in pts:
        lat, lon = _ll(dx, dy)
        rows.append(dict(icao=icao, reg=None, type=None, callsign=callsign, airport=airport, t=float(t), lat=lat, lon=lon,
                         on_ground=int(g), alt_missing=0, alt=np.nan if g else 1500.0, gs=float(gs), track=0.0,
                         src="adsb_icao"))
    return rows


def _departure(icao, callsign, t_park0, park_s, taxi_pts, t_to, stand_xy=(0.0, 0.0), airport="EHAM"):
    """Parked at stand for park_s seconds from t_park0 (sampled every 30 s), then taxi_pts (t, dx, dy, gs),
    take-off at t_to (ground sample at t_to - 5, airborne at t_to)."""
    pts = [(t_park0 + k * 30, stand_xy[0], stand_xy[1], 1, 0.0) for k in range(int(park_s // 30) + 1)]
    pts += [(t, dx, dy, 1, gs) for t, dx, dy, gs in taxi_pts]
    pts += [(t_to - 5, 3000.0, 0.0, 1, 140.0), (t_to, 3100.0, 0.0, 0, 160.0), (t_to + 20, 5000.0, 0.0, 0, 180.0)]
    return _samples(icao, callsign, pts, airport)


def _arrival(icao, callsign, t_land, park_s, stand_xy=(0.0, 0.0)):
    """Airborne approach, landing at t_land, taxi-in, parked for park_s (same callsign as the next departure)."""
    pts = [(t_land - 60, -4000.0, 0.0, 0, 150.0), (t_land, -3000.0, 0.0, 1, 130.0), (t_land + 120, -1000.0, 0.0, 1, 20.0)]
    pts += [(t_land + 300 + k * 30, stand_xy[0], stand_xy[1], 1, 0.0) for k in range(int(park_s // 30) + 1)]
    return _samples(icao, callsign, pts)


def _flight(mvt_id, callsign, mvt_t, aobt3_t, stand="S1", airport="EHAM", flight_mvt=None):
    return dict(MVT_ID_mvt=mvt_id, ADEP_mvt=airport, CALLSIGN_flt=callsign, FLIGHT_mvt=flight_mvt or callsign,
                MVT_TIME_UTC_mvt=pd.Timestamp(mvt_t, unit="s", tz="UTC"),
                AOBT_3_flt=pd.Timestamp(aobt3_t, unit="s", tz="UTC"), STAND_mvt=stand, RUNWAY_mvt="18L")


T0 = 1_736_000_000.0


# ---------------------------------------------------------------- tracks and take-off events

def test_tracks_split_on_a_30_minute_gap_per_aircraft():
    """Kills: one track per icao (inbound and outbound legs merged — the root of the E4b join defect)."""
    df = pd.DataFrame(_samples("a1", "X", [(T0, 0, 0, 1, 0), (T0 + 60, 0, 0, 1, 0), (T0 + 60 + 1801, 0, 0, 1, 0)])
                      + _samples("b2", "Y", [(T0, 0, 0, 1, 0)]))
    tr = fx.segment_tracks(df)
    assert tr.groupby("icao").track_id.nunique().to_dict() == {"a1": 2, "b2": 1}


def test_takeoff_event_is_the_last_ground_to_air_transition_and_absent_without_ground_samples():
    """Kills: using the first sample as take-off; kills inventing a take-off for an airborne-only track."""
    dep = pd.DataFrame(_departure("a1", "KLM1", T0, 600, [(T0 + 700, 100, 0, 10)], T0 + 1500))
    air = pd.DataFrame(_samples("c3", "Z", [(T0, 0, 0, 0, 200), (T0 + 30, 900, 0, 0, 210)]))
    ev = fx.takeoff_events(fx.segment_tracks(pd.concat([dep, air], ignore_index=True)))
    assert len(ev) == 1
    assert ev.iloc[0].icao == "a1" and ev.iloc[0].t_takeoff == T0 + 1500 and ev.iloc[0].callsign == "KLM1"


# ---------------------------------------------------------------- join (the E4b defect's regression test)

def test_join_picks_the_outbound_leg_not_the_same_callsign_inbound_leg():
    """Regression for E4b VOID (adsb_storm.join_matched took the callsign's track with the MOST samples, i.e. the
    inbound leg). The inbound leg here has far more samples; the join must still take the track whose take-off is
    at MVT. Kills: any sample-count or first-track rule."""
    mvt = T0 + 3 * 3600
    inbound = _arrival("a1", "VJT409", T0, park_s=4000)                         # 140+ samples, no take-off
    outbound = _departure("a1", "VJT409", mvt - 900, 300, [(mvt - 500, 50, 0, 8)], mvt)
    df = pd.DataFrame(inbound + outbound)
    tr = fx.segment_tracks(df)
    j = fx.join_flights(pd.DataFrame([_flight(1, "VJT409", mvt, mvt - 800)]), fx.takeoff_events(tr))
    chosen = tr[tr.track_id == j.iloc[0].track_id]
    assert j.iloc[0].join_type == "callsign"
    assert chosen.t.min() >= mvt - 900 - 1                                    # the outbound track, not the inbound
    assert abs(j.iloc[0].dt_takeoff) <= 1


def test_join_by_time_needs_a_unique_takeoff_within_30_s():
    """Kills: a time join when two aircraft take off together; kills a time join outside 30 s."""
    mvt = T0 + 5000
    a = _departure("a1", "AAA1", mvt - 900, 300, [], mvt + 10)
    b = _departure("b2", "BBB2", mvt - 900, 300, [], mvt + 20)
    ev = fx.takeoff_events(fx.segment_tracks(pd.DataFrame(a + b)))
    fl = pd.DataFrame([_flight(1, "NOPE9", mvt, mvt - 800)])
    assert fx.join_flights(fl, ev).track_id.isna().all()                          # two candidates: ambiguous
    ev1 = fx.takeoff_events(fx.segment_tracks(pd.DataFrame(a)))
    j = fx.join_flights(fl, ev1)
    assert j.iloc[0].join_type == "time" and not pd.isna(j.iloc[0].track_id)
    far = fx.takeoff_events(fx.segment_tracks(pd.DataFrame(_departure("a1", "AAA1", mvt - 900, 300, [], mvt + 45))))
    assert fx.join_flights(fl, far).track_id.isna().all()


def test_join_callsign_window_is_180_s_and_matches_flight_mvt_too():
    """Kills: a callsign join outside ±180 s; kills ignoring FLIGHT_mvt when CALLSIGN_flt differs."""
    mvt = T0 + 5000
    ev = fx.takeoff_events(fx.segment_tracks(pd.DataFrame(_departure("a1", "KLM7", mvt - 900, 300, [], mvt + 170))))
    j = fx.join_flights(pd.DataFrame([_flight(1, "OTHER", mvt, mvt - 800, flight_mvt="KLM7")]), ev)
    assert j.iloc[0].join_type == "callsign"
    ev2 = fx.takeoff_events(fx.segment_tracks(pd.DataFrame(_departure("a1", "KLM7", mvt - 900, 300, [], mvt + 200))))
    assert fx.join_flights(pd.DataFrame([_flight(1, "KLM7", mvt, mvt - 800)]), ev2).track_id.isna().all()


def test_one_takeoff_event_is_never_given_to_two_flights():
    """Kills: two flights sharing one track (the orphan/steal class the review found in adsb_storm)."""
    mvt = T0 + 5000
    ev = fx.takeoff_events(fx.segment_tracks(pd.DataFrame(_departure("a1", "KLM7", mvt - 900, 300, [], mvt + 5))))
    fl = pd.DataFrame([_flight(1, "KLM7", mvt, mvt - 800), _flight(2, "KLM7", mvt + 60, mvt - 700)])
    j = fx.join_flights(fl, ev).set_index("MVT_ID_mvt")
    assert j.loc[1, "join_type"] == "callsign" and pd.isna(j.loc[2, "track_id"])


def test_join_never_crosses_airports():
    """Kills: joining a take-off at another airport's box."""
    mvt = T0 + 5000
    ev = fx.takeoff_events(fx.segment_tracks(pd.DataFrame(
        _departure("a1", "KLM7", mvt - 900, 300, [], mvt, airport="EDDM"))))
    assert fx.join_flights(pd.DataFrame([_flight(1, "KLM7", mvt, mvt - 800)]), ev).track_id.isna().all()


# ---------------------------------------------------------------- per-flight features

def test_dwell_features_measure_the_gate_dwell_end_stops_and_path():
    """Parked 20 min, pushes at t_park_end, taxis with one 180 s stop, takes off. Kills: taking the dwell's START;
    kills counting the take-off roll as a stop; kills a path length in degrees."""
    t_park0, mvt = T0, T0 + 1200 + 900
    park_end = t_park0 + 1200
    taxi = [(park_end + 30, 30, 0, 3), (park_end + 90, 200, 0, 12), (park_end + 150, 600, 0, 15)]
    taxi += [(park_end + 180 + k * 30, 800, 0, 0.0) for k in range(7)]           # stop 180 s at the holding point
    taxi += [(park_end + 420, 1200, 0, 14), (park_end + 600, 2500, 0, 16)]
    df = pd.DataFrame(_departure("a1", "KLM1", t_park0, 1200, taxi, mvt))
    tr = fx.segment_tracks(df)
    j = fx.join_flights(pd.DataFrame([_flight(1, "KLM1", mvt, park_end)]), fx.takeoff_events(tr))
    f = fx.flight_features(j, tr).iloc[0]
    assert f.dwell_end_rel_mvt == pytest.approx(mvt - park_end, abs=1)         # seconds before take-off
    assert f.dwell_s == pytest.approx(1200, abs=31)
    assert f.n_stops == 1 and f.stop_s == pytest.approx(180, abs=31)
    assert 2900 < f.path_m < 3300
    assert f.first_seen_rel_mvt == pytest.approx(mvt - t_park0, abs=1)


def test_unjoined_flights_get_nan_features_not_zeros():
    """Kills: filling missing ADS-B with 0 (a model would read 0 s taxi)."""
    j = pd.DataFrame([dict(MVT_ID_mvt=9, ADEP_mvt="EHAM", MVT_TIME_UTC_mvt=pd.Timestamp(T0, unit="s", tz="UTC"),
                           AOBT_3_flt=pd.Timestamp(T0 - 600, unit="s", tz="UTC"), STAND_mvt="S1",
                           track_id=np.nan, t_takeoff=np.nan, dt_takeoff=np.nan, join_type=None)])
    f = fx.flight_features(j, fx.segment_tracks(pd.DataFrame(_samples("a1", "X", [(T0, 0, 0, 1, 0)]))))
    assert np.isnan(f.iloc[0].dwell_end_rel_mvt) and np.isnan(f.iloc[0].path_m) and f.iloc[0].joined == 0


def test_stand_centroid_is_leave_one_flight_out_and_needs_two_others():
    """Kills: a flight contributing to its own centroid (a self-fitted gate); kills centroids from one flight."""
    mvt = T0 + 10_000
    rows, fl = [], []
    for k, dx in enumerate([0.0, 20.0, 40.0]):
        m = mvt + k * 4000
        rows += _departure(f"a{k}", f"KL{k}", m - 1800, 900, [(m - 850, dx + 60, 0, 8)], m, stand_xy=(dx, 0.0))
        fl.append(_flight(k, f"KL{k}", m, m - 900))
    tr = fx.segment_tracks(pd.DataFrame(rows))
    j = fx.join_flights(pd.DataFrame(fl), fx.takeoff_events(tr))
    c = fx.stand_centroids(j, tr).set_index("MVT_ID_mvt")
    assert c.loc[0, "cx_m"] == pytest.approx(30.0, abs=2)                        # mean of the OTHER two (20, 40)
    two = j.iloc[:2]
    c2 = fx.stand_centroids(two, tr)
    assert c2.cx_m.isna().all()                                                  # each has only one other flight


def test_features_never_read_the_label_columns():
    """Label-free contract (prereg): permuting BLOCK_TIME / TAXITIME leaves every feature bit-identical."""
    mvt = T0 + 5000
    df = pd.DataFrame(_departure("a1", "KLM1", mvt - 1800, 900, [(mvt - 850, 60, 0, 8)], mvt))
    fl = pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 900)])
    a = fl.assign(BLOCK_TIME_UTC_mvt=pd.Timestamp(mvt - 900, unit="s", tz="UTC"), TAXITIME_SEC_mvt=900)
    b = fl.assign(BLOCK_TIME_UTC_mvt=pd.Timestamp(mvt - 5, unit="s", tz="UTC"), TAXITIME_SEC_mvt=5)
    fa, fb = fx.build(a, df), fx.build(b, df)
    pd.testing.assert_frame_equal(fa, fb)
    assert not any(c.startswith(("BLOCK", "TAXITIME")) for c in fa.columns)


# ---------------------------------------------------------------- first-seen-on-ground features (no gate dwell)

def test_first_ground_features_when_the_transponder_comes_on_at_pushback():
    """Real data (day-1 diagnosis): aircraft are rarely seen parked. Seen first at the stand at 2 kt, then taxiing.
    Kills: requiring a gate dwell to produce any timing; kills flagging a slow first sample as mid-taxi."""
    mvt = T0 + 5000
    pts = [(mvt - 800 + k * 10, 0, 0, 1, 2.0) for k in range(3)]                # pushback, slow, at the stand
    pts += [(mvt - 700 + k * 30, 100 + 100 * k, 0, 1, 14.0) for k in range(20)]
    pts += [(mvt - 5, 3000, 0, 1, 140.0), (mvt, 3100, 0, 0, 160.0)]
    df = pd.DataFrame(_samples("a1", "KLM1", pts))
    f = fx.build(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 780)]), df).iloc[0]
    assert np.isnan(f.dwell_s)
    assert f.first_ground_rel_mvt == pytest.approx(800, abs=1)
    assert f.first_ground_slow == 1 and f.first_ground_gs == pytest.approx(2.0)
    assert f.first_move_rel_mvt == pytest.approx(700, abs=1)                    # first sample at >= 3 kt
    assert f.first_ground_rel_aobt3 == pytest.approx(-20, abs=1)
    assert f.turn_in_track == 0


def test_first_ground_flags_a_track_first_seen_mid_taxi():
    """Kills: treating a mid-taxi first sample (14 kt) as the start of taxi."""
    mvt = T0 + 5000
    pts = [(mvt - 400 + k * 30, 1000 + 100 * k, 0, 1, 14.0) for k in range(10)]
    pts += [(mvt - 5, 3000, 0, 1, 140.0), (mvt, 3100, 0, 0, 160.0)]
    f = fx.build(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 780)]), pd.DataFrame(_samples("a1", "KLM1", pts))).iloc[0]
    assert f.first_ground_slow == 0 and f.first_ground_rel_mvt == pytest.approx(400, abs=1)


def test_a_turnaround_inside_one_track_starts_the_ground_segment_after_landing_and_is_flagged():
    """Transponder left on through a quick turn: approach, landing, taxi-in, park 20 min, depart — one track.
    Kills: a ground segment that begins before the landing (first_ground would be the approach)."""
    mvt = T0 + 5000
    pts = [(mvt - 3300, -9000, 0, 1, 200.0)]                                     # spurious 'ground' flag in the air
    pts += [(mvt - 3000, -4000, 0, 0, 150.0), (mvt - 2940, -3000, 0, 1, 120.0), (mvt - 2800, -500, 0, 1, 15.0)]
    pts += [(mvt - 2600 + k * 30, 0, 0, 1, 0.0) for k in range(41)]              # parked 1,200 s
    pts += [(mvt - 1390 + k * 60, 100 * (k + 1), 0, 1, 12.0) for k in range(20)]
    pts += [(mvt - 5, 3000, 0, 1, 140.0), (mvt, 3100, 0, 0, 160.0)]
    f = fx.build(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 1400)]), pd.DataFrame(_samples("a1", "KLM1", pts))).iloc[0]
    assert f.turn_in_track == 1
    assert f.first_ground_rel_mvt == pytest.approx(2940, abs=1)                 # the landing, flagged
    assert f.dwell_end_rel_mvt == pytest.approx(1400, abs=31)                   # the dwell still anchors taxi-out


def test_stand_reference_comes_from_other_flights_first_slow_samples_without_any_dwell():
    """No aircraft is ever seen parked; the stand reference must still form from first slow samples (LOO).
    Kills: centroids that require parked-at-AOBT_3 samples only."""
    rows, fl = [], []
    for k, dx in enumerate([0.0, 20.0, 40.0]):
        m = T0 + 10_000 + k * 4000
        pts = [(m - 800 + s * 10, dx, 0, 1, 2.0) for s in range(3)]
        pts += [(m - 700 + s * 30, dx + 100 * (s + 1), 0, 1, 14.0) for s in range(20)]
        pts += [(m - 5, 3000, 0, 1, 140.0), (m, 3100, 0, 0, 160.0)]
        rows += _samples(f"a{k}", f"KL{k}", pts)
        fl.append(_flight(k, f"KL{k}", m, m - 780))
    tr = fx.segment_tracks(pd.DataFrame(rows))
    j = fx.join_flights(pd.DataFrame(fl), fx.takeoff_events(tr))
    c = fx.stand_centroids(j, tr).set_index("MVT_ID_mvt")
    assert c.loc[0, "cx_m"] == pytest.approx(30.0, abs=2)


# ---------------------------------------------------------------- build_period (day files, midnight, stamps)

def _write_day(d, day, rows):
    df = pd.DataFrame(rows)
    df.to_parquet(d / f"adsb_{day}.parquet", index=False)


def _utc(day, secs):
    return (pd.Timestamp(day, tz="UTC") - fx.EPOCH).total_seconds() + secs


def test_build_period_uses_the_previous_days_tail_for_a_just_after_midnight_departure(tmp_path):
    """Dwell starts 23:30 on day D-1, take-off 00:10 on D. Kills: building each day from its own file only
    (the flight would lose its gate dwell)."""
    mvt = _utc("2025-01-02", 600)
    rows = _departure("a1", "KLM1", mvt - 2400, 1500, [(mvt - 850, 60, 0, 8)], mvt)
    prev = [r for r in rows if r["t"] < _utc("2025-01-02", 0)]
    cur = [r for r in rows if r["t"] >= _utc("2025-01-02", 0)]
    _write_day(tmp_path, "2025-01-01", prev)
    _write_day(tmp_path, "2025-01-02", cur)
    fl = pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 900)])
    out = fx.build_period(fl, tmp_path, git_sha="abc123")
    r = out.iloc[0]
    assert r.joined == 1 and r.dwell_s == pytest.approx(1500, abs=31) and r.coverage == 3


def test_build_period_finds_a_takeoff_sample_in_the_next_days_file(tmp_path):
    """MVT 23:59:50 on D, first airborne sample at 00:00:05 on D+1. Kills: not reading the next day's head."""
    mvt = _utc("2025-01-02", 86_390)
    rows = _departure("a1", "KLM1", mvt - 1800, 900, [(mvt - 850, 60, 0, 8)], mvt + 15)
    _write_day(tmp_path, "2025-01-02", [r for r in rows if r["t"] < _utc("2025-01-03", 0)])
    _write_day(tmp_path, "2025-01-03", [r for r in rows if r["t"] >= _utc("2025-01-03", 0)])
    out = fx.build_period(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 900)]), tmp_path, git_sha="abc123")
    assert out.iloc[0].joined == 1


def test_build_period_coverage_codes_and_stamps(tmp_path):
    """coverage: 0 no ADS-B day file, 1 day present but not joined, 3 joined with a gate dwell. Stamps: git sha,
    extractor and feature versions in the parquet metadata. Kills: silent NaN rows with no reason code."""
    mvt = _utc("2025-01-02", 40_000)
    _write_day(tmp_path, "2025-01-02", _departure("a1", "KLM1", mvt - 1800, 900, [(mvt - 850, 60, 0, 8)], mvt))
    fl = pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 900), _flight(2, "NOPE1", mvt + 5000, mvt + 4000),
                       _flight(3, "KLM9", _utc("2025-01-05", 40_000), _utc("2025-01-05", 39_000))])
    out_path = tmp_path / "features.parquet"
    out = fx.build_period(fl, tmp_path, git_sha="abc123", out_path=out_path).set_index("MVT_ID_mvt")
    assert out.coverage.to_dict() == {1: 3, 2: 1, 3: 0}
    assert np.isnan(out.loc[3, "dwell_s"])
    import pyarrow.parquet as pq
    meta = pq.read_schema(out_path).metadata
    assert meta[b"git_sha"] == b"abc123"
    assert meta[b"extractor"] == b"adsb_fetch.parse_member_v2"
    assert meta[b"features_version"].startswith(b"adsb_features")


def test_build_period_returns_rows_in_the_input_order(tmp_path):
    """Kills: output grouped or sorted by day/ID — a consumer aligning by position would pair features with the wrong
    flight (the row-alignment class this project has hit before)."""
    m1, m2, m3 = _utc("2025-01-03", 40_000), _utc("2025-01-02", 40_000), _utc("2025-01-02", 50_000)
    _write_day(tmp_path, "2025-01-02", _departure("a1", "KLM1", m2 - 1800, 900, [(m2 - 850, 60, 0, 8)], m2))
    fl = pd.DataFrame([_flight(30, "X1", m1, m1 - 900), _flight(10, "KLM1", m2, m2 - 900), _flight(20, "X2", m3, m3 - 900)])
    out = fx.build_period(fl, tmp_path, git_sha="x")
    assert out.MVT_ID_mvt.tolist() == [30, 10, 20]
    assert out.joined.tolist() == [0, 1, 0]


def test_coverage_codes_for_flights_seen_from_the_stand_or_mid_taxi_without_a_dwell(tmp_path):
    """3 = seen from the stand via a slow first sample (no dwell); 2 = first seen mid-taxi.
    Kills: a coverage code that only credits a gate dwell (almost never observed on real data)."""
    m1, m2 = _utc("2025-01-02", 40_000), _utc("2025-01-02", 50_000)
    slow = [(m1 - 800 + k * 10, 0, 0, 1, 2.0) for k in range(3)] + [(m1 - 700 + k * 30, 100 + 100 * k, 0, 1, 14.0)
                                                                   for k in range(20)]
    slow += [(m1 - 5, 3000, 0, 1, 140.0), (m1, 3100, 0, 0, 160.0)]
    fast = [(m2 - 400 + k * 30, 1000 + 100 * k, 0, 1, 14.0) for k in range(10)]
    fast += [(m2 - 5, 3000, 0, 1, 140.0), (m2, 3100, 0, 0, 160.0)]
    _write_day(tmp_path, "2025-01-02", _samples("a1", "KLM1", slow) + _samples("b2", "KLM2", fast))
    fl = pd.DataFrame([_flight(1, "KLM1", m1, m1 - 780), _flight(2, "KLM2", m2, m2 - 780)])
    out = fx.build_period(fl, tmp_path, git_sha="x").set_index("MVT_ID_mvt")
    assert out.coverage.to_dict() == {1: 3, 2: 2}


def test_build_period_is_label_free_end_to_end(tmp_path):
    """The period builder must also ignore label columns. Kills: a build_period that forwards the full frame."""
    mvt = _utc("2025-01-02", 40_000)
    _write_day(tmp_path, "2025-01-02", _departure("a1", "KLM1", mvt - 1800, 900, [(mvt - 850, 60, 0, 8)], mvt))
    fl = pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 900)])
    a = fx.build_period(fl.assign(TAXITIME_SEC_mvt=900, BLOCK_TIME_UTC_mvt=pd.Timestamp(mvt - 900, unit="s", tz="UTC")),
                        tmp_path, git_sha="x")
    b = fx.build_period(fl.assign(TAXITIME_SEC_mvt=7, BLOCK_TIME_UTC_mvt=pd.Timestamp(mvt - 7, unit="s", tz="UTC")),
                        tmp_path, git_sha="x")
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- fold-A flight frame (row -> MVT_ID -> serve-time columns)

def _fold_world(tmp_path, shuffle_raw=False, bad_proxy=False):
    """Two cache months (3 + 2 rows), a fold record of 3 rows, raw DEP rows with serve-time columns."""
    (tmp_path / "stand").mkdir(); (tmp_path / "order").mkdir(); (tmp_path / "raw").mkdir()
    ids = {"training_2025-01-01_2025-02-01.parquet": [11, 12, 13], "training_2025-07-01_2025-08-01.parquet": [21, 22]}
    raw_rows = []
    for name, mids in ids.items():
        pd.DataFrame({"delta": np.zeros(len(mids))}).to_parquet(tmp_path / "stand" / name)
        pd.DataFrame({"MVT_ID_mvt": mids}).to_parquet(tmp_path / "order" / name)
        for k, mid in enumerate(mids):
            mvt = T0 + mid * 1000
            raw_rows.append(dict(_flight(mid, f"C{mid}", mvt, mvt - (600 + 10 * k)), PHASE_mvt="DEP",
                                 TAXITIME_SEC_mvt=999))
    raw = pd.DataFrame(raw_rows)
    if shuffle_raw:
        raw = raw.sample(frac=1.0, random_state=0)
    raw[raw.MVT_ID_mvt < 20].to_parquet(tmp_path / "raw" / "training_2025-01-01_2025-02-01.parquet")
    raw[raw.MVT_ID_mvt > 20].to_parquet(tmp_path / "raw" / "training_2025-07-01_2025-08-01.parquet")
    rows = [4, 0, 2]                                          # record rows index the concatenated stand caches
    proxy = [600.0 + 10 * 1, 600.0, 600.0 + 10 * 2]          # MVT - AOBT_3 of ids 22, 11, 13
    if bad_proxy:
        proxy[0] += 5
    pd.DataFrame({"row": rows, "proxy": proxy}).to_parquet(tmp_path / "rec.parquet")
    return tmp_path


def test_fold_flights_follow_the_record_order_and_carry_only_serve_time_columns(tmp_path):
    """Kills: returning raw-file order (features would pair with the wrong F prediction); kills carrying a label."""
    w = _fold_world(tmp_path, shuffle_raw=True)
    fl = fx.fold_flights(w / "rec.parquet", w / "stand", w / "order", w / "raw")
    assert fl.MVT_ID_mvt.tolist() == [22, 11, 13]
    assert fl.row.tolist() == [4, 0, 2]
    assert not any(c.startswith(("TAXITIME", "BLOCK")) for c in fl.columns)


def test_fold_flights_refuse_a_misaligned_record(tmp_path):
    """The label-free alignment guard: MVT - AOBT_3 from the raw file must equal the record's proxy within 1 s.
    Kills: a silent row -> MVT_ID shift (the BC-2 / row-alignment class)."""
    w = _fold_world(tmp_path, bad_proxy=True)
    with pytest.raises(ValueError, match="alignment"):
        fx.fold_flights(w / "rec.parquet", w / "stand", w / "order", w / "raw")


REAL_REC = ROOT / "data" / "cache_stand" / "fold_preds_queue_order.parquet"


@pytest.mark.skipif(not REAL_REC.exists(), reason="fold-A record absent (data/ is not in git)")
def test_fold_flights_on_the_real_fold_a_record():
    """Real data: all 339,015 fold-A rows map to unique MVT_IDs and pass the proxy alignment guard."""
    fl = fx.fold_flights(REAL_REC, ROOT / "data" / "cache_stand", ROOT / "data" / "cache_order", ROOT / "data" / "raw")
    assert len(fl) == 339_015 and fl.MVT_ID_mvt.is_unique
    assert set(pd.to_datetime(fl.MVT_TIME_UTC_mvt, utc=True).dt.month.unique()) == {1, 7}


# ---------------------------------------------------------------- BC-3: no sentinel stands in for "not observed"

def test_bc3_join_callsign_is_unknown_not_zero_for_unjoined_flights():
    """BC-3 (sentinel ingested as measurement): 209,575 real unjoined rows carried join_callsign = 0, a definite value
    for an unknown. Kills: `(join_type == 'callsign').astype(float)` on a nullable column."""
    j = pd.DataFrame([dict(MVT_ID_mvt=9, ADEP_mvt="EHAM", MVT_TIME_UTC_mvt=pd.Timestamp(T0, unit="s", tz="UTC"),
                           AOBT_3_flt=pd.Timestamp(T0 - 600, unit="s", tz="UTC"), STAND_mvt="S1",
                           track_id=np.nan, t_takeoff=np.nan, dt_takeoff=np.nan, join_type=None)])
    f = fx.flight_features(j, fx.segment_tracks(pd.DataFrame(_samples("a1", "X", [(T0, 0, 0, 1, 0)]))))
    assert np.isnan(f.iloc[0].join_callsign)


def test_bc3_moving_count_is_unknown_when_no_aircraft_is_seen_on_the_ground():
    """BC-3: a day or airport without ground reception (real: EDDM 2025-07-15 had 30 slow samples in 67k; LFPG
    none) gave moving counts of 0 = 'no congestion'. The count must be NaN when no OTHER aircraft is seen on the
    ground in the window, and ground_seen_mvt_m10 must say how many were seen. Kills: 0 for 'not observed'."""
    mvt = T0 + 5000
    dep = _departure("a1", "KLM1", mvt - 1800, 900, [(mvt - 850, 60, 0, 8)], mvt)
    airborne_only = _samples("z9", "Z9", [(mvt - 600 + s, 0, 900, 0, 150.0) for s in range(0, 120, 10)])
    f = fx.build(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 900)]), pd.DataFrame(dep + airborne_only)).iloc[0]
    assert np.isnan(f.moving_mvt_m10) and f.ground_seen_mvt_m10 == 0
    parked = _samples("p0", "P0", [(mvt - 600 + s, 500, 500, 1, 0.0) for s in range(0, 120, 10)])
    f2 = fx.build(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 900)]), pd.DataFrame(dep + parked)).iloc[0]
    assert f2.moving_mvt_m10 == 0 and f2.ground_seen_mvt_m10 == 1          # seen, parked: a real zero


def test_bc3_moving_count_is_unknown_at_an_airport_with_no_ground_samples_at_all():
    """BC-3: LTFM / LFPG-like — the flight's airport has no on-ground sample of any aircraft that day.
    Kills: falling through to 0 when the airport is absent from the ground index."""
    mvt = T0 + 5000
    elsewhere = _departure("a1", "KLM1", mvt - 1800, 900, [(mvt - 850, 60, 0, 8)], mvt, airport="EHAM")
    f = fx.build(pd.DataFrame([_flight(1, "TK1", mvt, mvt - 900, airport="LTFM")]), pd.DataFrame(elsewhere)).iloc[0]
    assert np.isnan(f.moving_mvt_m10) and f.ground_seen_mvt_m10 == 0 and f.joined == 0


def _parsed(icao, callsign, pts):
    """Samples pushed through the REAL v2 parser (adsb_fetch.parse_member_v2) from a readsb-style trace."""
    import json as _json
    spec = importlib.util.spec_from_file_location("adsb_fetch_t", ROOT / "scripts" / "adsb_fetch.py")
    af = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(af)
    trace = []
    for t, dx, dy, g, gs in pts:
        lat, lon = _ll(dx, dy)
        trace.append([t - T0, lat, lon, "ground" if g else 1500, gs, 0.0, 0, 0, {"flight": callsign}, "adsb_icao"])
    raw = _json.dumps({"icao": icao, "r": None, "t": None, "timestamp": T0, "trace": trace}).encode()
    return pd.DataFrame(af.parse_member_v2(raw), columns=af.COLUMNS)


def test_bc4_a_speed_exactly_on_the_moving_threshold_is_moving_everywhere():
    """BC-4 (threshold on a quantised value): ground speed is on a 0.1-kt grid, so 3.0 kt occurs. The congestion index
    used `gs > 3` while first-move used `gs >= 3`. Kills: either comparison flipping; checked through the real parser."""
    mvt = T0 + 5000
    own = _parsed("a1", "KLM1", [(mvt - 800 + k * 10, 0, 0, 1, 2.0) for k in range(3)]
                  + [(mvt - 700, 100, 0, 1, 3.0)] + [(mvt - 600 + k * 30, 200 + 100 * k, 0, 1, 14.0) for k in range(10)]
                  + [(mvt - 5, 3000, 0, 1, 140.0), (mvt, 3100, 0, 0, 160.0)])
    other = _parsed("m1", "M1", [(mvt - 600 + s, 500, 50, 1, 3.0) for s in range(0, 60, 10)])
    f = fx.build(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 780)]), pd.concat([own, other], ignore_index=True)).iloc[0]
    assert f.first_move_rel_mvt == pytest.approx(700, abs=1)                     # 3.0 kt starts the move
    assert f.moving_mvt_m10 == 1                                                 # and 3.0 kt counts as moving


def test_bc3_path_is_unknown_not_zero_when_it_cannot_be_measured():
    """BC-3: one ground sample after the reference point gave path_m = 0.0. Kills: 0 for 'not measurable'."""
    mvt = T0 + 5000
    pts = [(mvt - 800 + k * 30, 0, 0, 1, 0.0) for k in range(12)] + [(mvt, 3100, 0, 0, 160.0)]   # dwell, then air
    f = fx.build(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 500)]), pd.DataFrame(_samples("a1", "KLM1", pts))).iloc[0]
    assert np.isfinite(f.dwell_s) and np.isnan(f.path_m)


def test_surface_congestion_counts_distinct_moving_aircraft_in_the_window():
    """Kills: counting samples instead of aircraft; kills counting parked aircraft as moving."""
    mvt = T0 + 5000
    dep = _departure("a1", "KLM1", mvt - 1800, 900, [(mvt - 850, 60, 0, 8)], mvt)
    movers = []
    for k in range(3):                                                           # 3 aircraft taxiing at MVT - 600
        movers += _samples(f"m{k}", f"M{k}", [(mvt - 600 + s, 100 * k, 50, 1, 12.0) for s in range(0, 120, 10)])
    parked = _samples("p0", "P0", [(mvt - 600 + s, 500, 500, 1, 0.0) for s in range(0, 120, 10)])
    f = fx.build(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 900)]), pd.DataFrame(dep + movers + parked)).iloc[0]
    assert f.moving_mvt_m10 == 3


def test_build_fold_cli_writes_record_ordered_stamped_table_and_refuses_overwrite(tmp_path):
    """Kills: a table not in record order; a missing git/extractor stamp; silently overwriting a built table."""
    w = _fold_world(tmp_path)
    (w / "adsb").mkdir()
    mvt = T0 + 22 * 1000
    day = pd.Timestamp(mvt, unit="s", tz="UTC").strftime("%Y-%m-%d")
    _write_day(w / "adsb", day, _departure("a1", "C22", mvt - 1800, 900, [(mvt - 850, 60, 0, 8)], mvt))
    out = w / "feat.parquet"
    args = ["build-fold", "--record", str(w / "rec.parquet"), "--stand-dir", str(w / "stand"), "--order-dir",
            str(w / "order"), "--raw-dir", str(w / "raw"), "--adsb-dir", str(w / "adsb"), "--out", str(out)]
    assert fx.main(args) == 0
    t = pd.read_parquet(out)
    assert t.row.tolist() == [4, 0, 2] and t.MVT_ID_mvt.tolist() == [22, 11, 13]
    assert t.set_index("MVT_ID_mvt").loc[22, "joined"] == 1
    import pyarrow.parquet as pq
    meta = pq.read_schema(out).metadata
    assert meta[b"extractor"] == b"adsb_fetch.parse_member_v2" and len(meta[b"git_sha"]) >= 7
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        fx.main(args)


# ---------------------------------------------------------------- 2026 scored flights (ranking file)

def _ranking_world(tmp_path):
    rk = pd.DataFrame([
        dict(_flight(11, "A1", T0 + 1000, T0 + 400), PHASE_mvt="DEP", TAXITIME_SEC_mvt=np.nan),
        dict(_flight(12, "A2", T0 + 2000, T0 + 1400), PHASE_mvt="DEP", TAXITIME_SEC_mvt=np.nan),
        dict(_flight(13, "A3", T0 + 3000, T0 + 2400), PHASE_mvt="ARR", TAXITIME_SEC_mvt=500.0),   # arrival: not scored
        dict(_flight(14, "A4", T0 + 4000, T0 + 3400), PHASE_mvt="DEP", TAXITIME_SEC_mvt=np.nan),
    ])
    rk.loc[rk.MVT_ID_mvt == 14, "AOBT_3_flt"] = pd.NaT                                      # unmatched: no NM clock
    rk.to_parquet(tmp_path / "ranking.parquet")
    pd.DataFrame({"MVT_ID_mvt": [14.0, 12.0, 11.0], "TAXITIME_SEC_mvt": np.nan}).to_parquet(tmp_path / "submitting.parquet")
    return tmp_path


def test_ranking_flights_are_the_scored_matched_departures_in_submission_order(tmp_path):
    """Kills: including arrivals or unmatched rows (no AOBT_3 -> no stand reference, not the matched lane); returning
    ranking-file order instead of the submission template's; carrying a label column."""
    w = _ranking_world(tmp_path)
    fl = fx.ranking_flights(w / "ranking.parquet", w / "submitting.parquet")
    assert fl.MVT_ID_mvt.tolist() == [12, 11]
    assert list(fl.columns) == fx.FLIGHT_COLS


def test_build_ranking_cli_writes_the_2026_table_in_submission_order(tmp_path):
    """The ship-side table: scored matched departures, submission order, stamped, label-free. Kills: a table keyed
    in another order; a missing coverage column; overwriting a built table."""
    w = _ranking_world(tmp_path)
    (w / "adsb").mkdir()
    mvt = T0 + 2000
    day = pd.Timestamp(mvt, unit="s", tz="UTC").strftime("%Y-%m-%d")
    _write_day(w / "adsb", day, _departure("a1", "A2", mvt - 1800, 900, [(mvt - 850, 60, 0, 8)], mvt))
    out = w / "f2026.parquet"
    args = ["build-ranking", "--ranking", str(w / "ranking.parquet"), "--submitting", str(w / "submitting.parquet"),
            "--adsb-dir", str(w / "adsb"), "--out", str(out)]
    assert fx.main(args) == 0
    t = pd.read_parquet(out)
    assert t.MVT_ID_mvt.tolist() == [12, 11] and "coverage" in t.columns
    assert t.set_index("MVT_ID_mvt").loc[12, "joined"] == 1
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        fx.main(args)


# ---------------------------------------------------------------- arm ADU: unmatched flights from a record keyed by MVT_ID

def test_record_flights_follow_the_record_order_for_the_requested_months(tmp_path):
    """Kills: raw-file order; rows from other months; a label column carried along; a missing id passing silently."""
    raw = pd.DataFrame([dict(_flight(1, "A1", T0, T0 - 600), PHASE_mvt="DEP", TAXITIME_SEC_mvt=900),
                        dict(_flight(2, "A2", T0 + 10, T0 - 590), PHASE_mvt="DEP", TAXITIME_SEC_mvt=800),
                        dict(_flight(3, "A3", T0 + 20, T0 - 580), PHASE_mvt="DEP", TAXITIME_SEC_mvt=700)])
    raw.loc[:, "AOBT_3_flt"] = pd.NaT
    raw.to_parquet(tmp_path / "raw.parquet")
    rec = pd.DataFrame({"MVT_ID_mvt": [3.0, 1.0, 2.0], "month": [1, 1, 2]})
    rec.to_parquet(tmp_path / "rec.parquet")
    fl = fx.record_flights(tmp_path / "rec.parquet", [tmp_path / "raw.parquet"], months=(1,))
    assert fl.MVT_ID_mvt.tolist() == [3, 1] and list(fl.columns) == fx.FLIGHT_COLS
    rec2 = pd.DataFrame({"MVT_ID_mvt": [9.0], "month": [1]}); rec2.to_parquet(tmp_path / "rec2.parquet")
    with pytest.raises(ValueError, match="absent"):
        fx.record_flights(tmp_path / "rec2.parquet", [tmp_path / "raw.parquet"], months=(1,))


def test_ranking_flights_can_select_the_scored_unmatched_departures(tmp_path):
    """c4's request (a Rome fill classifier later): the scored UNMATCHED departures (AOBT_3 null) in submission order.
    Kills: returning matched rows; dropping unmatched rows; arrivals leaking in."""
    w = _ranking_world(tmp_path)
    fl = fx.ranking_flights(w / "ranking.parquet", w / "submitting.parquet", matched=False)
    assert fl.MVT_ID_mvt.tolist() == [14]


# ---------------------------------------------------------------- arm ADX (plans/PREREG_adsb_taxistart_2026_09_11.md)

def test_first_ground_to_stand_is_the_distance_from_the_first_ground_sample_to_the_loo_centroid():
    """ADX feature 1. A code-2 flight first seen 400 m from its stand must report ~400 m; a code-3 flight first seen ON
    the stand ~0 m; a flight whose stand has too few others reports NaN (BC-3, no sentinel).
    Kills: the distance measured from the dwell (code-2 rows have none) or from the last sample; a 0.0 fallback."""
    fx = _load()
    j = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "ADEP_mvt": "EHAM", "track_id": [10.0, 20.0, 30.0],
                      "join_type": "callsign", "dt_takeoff": 0.0, "t_takeoff": 3000.0,
                      "MVT_TIME_UTC_mvt": pd.to_datetime("2025-01-02 01:00", utc=True),
                      "AOBT_3_flt": pd.to_datetime("2025-01-02 00:30", utc=True), "STAND_mvt": "A1"})
    cx, cy = fx._xy(pd.Series([52.31]), pd.Series([4.76]), "EHAM")
    far_lat = 52.31 + 400.0 / 111_320.0                       # ~400 m north
    def track(tid, lat, n=12, t0=1000.0, gs=10.0):
        """The aircraft taxis AWAY from the stand, so the first sample and the last are at different distances."""
        return pd.DataFrame({"track_id": tid, "t": t0 + np.arange(n) * 10.0,
                             "lat": lat + np.arange(n) * 100.0 / 111_320.0, "lon": 4.76,
                             "gs": gs, "on_ground": 1, "src": "adsb", "alt_missing": 0, "alt": np.nan})
    tr = pd.concat([track(10.0, far_lat), track(20.0, 52.31, gs=0.5), track(30.0, far_lat)], ignore_index=True)
    cen = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "cx_m": [cx[0], cx[0], np.nan], "cy_m": [cy[0], cy[0], np.nan],
                        "n_others": [5, 5, 1], "parked_share": [0.0, 0.0, np.nan],       # DWG columns (BC-26):
                        "pcx_m": np.nan, "pcy_m": np.nan, "n_parked_others": 0})          # first-appearance refs only
    f = fx.flight_features(j, tr, cen).set_index("MVT_ID_mvt")
    assert f.first_ground_to_stand_m[1.0] == pytest.approx(400.0, abs=15.0)
    assert f.first_ground_to_stand_m[2.0] == pytest.approx(0.0, abs=15.0)
    assert np.isnan(f.first_ground_to_stand_m[3.0]) and f.stand_others_n[3.0] == 1


def test_taxi_start_block_reconstructs_the_off_block_time_from_distance_and_speed():
    """ADX features 2-4: travel = distance / speed (knots -> m/s), implied off-block = first_ground_rel_aobt3 - travel,
    implied taxi = first_ground_rel_mvt + travel. A flight with no speed of its own uses its AIRPORT's median taxi speed;
    a flight below MIN_TAXI_GS also falls back (a fallback, not a clamp).
    Kills: knots used as m/s; the sign of the correction flipped; the airport fallback taken from the whole period;
    NaN distance producing a number."""
    fx = _load()
    df = pd.DataFrame({"ap": ["EHAM", "EHAM", "EHAM", "LSZH", "EHAM", "EHAM"],
                       "first_ground_to_stand_m": [514.444, 514.444, np.nan, 1000.0, 514.444, 514.444],
                       "taxi_med_gs": [10.0, np.nan, 10.0, 0.5, 0.5, 0.5],      # two unusable speeds at EHAM
                       "first_ground_rel_aobt3": [200.0] * 6,
                       "first_ground_rel_mvt": [600.0] * 6})
    out = fx.taxi_start_block(df)
    assert out.travel_s_first_to_stand[0] == pytest.approx(100.0, abs=0.5)      # 514.444 m at 10 kt = 5.14444 m/s
    assert out.implied_offblock_rel_aobt3[0] == pytest.approx(100.0, abs=0.5)   # 200 - 100
    assert out.implied_taxi_s[0] == pytest.approx(700.0, abs=0.5)               # 600 + 100
    assert out.travel_s_first_to_stand[1] == pytest.approx(100.0, abs=0.5)      # EHAM median speed is 10 kt
    assert np.isnan(out.travel_s_first_to_stand[2]) and np.isnan(out.implied_offblock_rel_aobt3[2])
    assert np.isnan(out.travel_s_first_to_stand[3])            # LSZH has no usable speed at all -> NaN, not a guess
    # rows 4 and 5 have an unusable own speed but EHAM has a median, so they take the fallback, like row 1; and that
    # median is over EHAM's USABLE speeds (10 kt), not over all of them (which would be 0.5 kt and give 2,000 s)
    assert out.travel_s_first_to_stand[4] == pytest.approx(100.0, abs=0.5)
    assert out.travel_s_first_to_stand[5] == pytest.approx(100.0, abs=0.5)


def test_adx_features_are_in_the_feature_list_and_the_version_is_bumped():
    """Kills: the block computed but never handed to the stacker; a table written with a stale version stamp."""
    fx = _load()
    for c in ("first_ground_to_stand_m", "stand_others_n", "travel_s_first_to_stand", "implied_offblock_rel_aobt3",
              "implied_taxi_s"):
        assert c in fx.FEATURES
    assert "taxi-start" in fx.FEATURES_VERSION                 # ADX's marker; the version number itself moves with
    assert "adsb_features v2 " not in fx.FEATURES_VERSION      # later arms, but the stamp must never fall back to v2


# ---------------------------------------------------------------- arm ADJ (plans/PREREG_adsb_join_recall_2026_09_12.md)

def _adj_world(fx, cs="KLM123", n_air=0):
    """One departure at EHAM, MVT at t=10,000, with an aircraft seen on the ground under its callsign beforehand."""
    mvt_ts = pd.Timestamp("2025-01-02 02:46:40", tz="UTC")
    mvt_s = mvt_ts.timestamp()                                  # the track's t is epoch seconds, as in the real files
    fl = pd.DataFrame({"MVT_ID_mvt": [1.0], "ADEP_mvt": ["EHAM"], "CALLSIGN_flt": [cs], "FLIGHT_mvt": ["KL123"],
                       "MVT_TIME_UTC_mvt": [mvt_ts],
                       "AOBT_3_flt": [pd.Timestamp("2025-01-02 02:30:00", tz="UTC")], "STAND_mvt": ["A1"],
                       "track_id": [np.nan], "t_takeoff": [np.nan], "join_type": [None], "dt_takeoff": [np.nan]})
    t = mvt_s - np.array([2000.0, 1700, 1400, 1100, 800])
    tr = pd.DataFrame({"track_id": 77.0, "icao": "aaa111", "airport": "EHAM", "callsign": cs, "t": t,
                       "on_ground": 1, "alt_missing": 0, "alt": np.nan, "gs": [0.5, 0.5, 4.0, 9.0, 12.0],
                       "lat": 52.31, "lon": 4.76, "src": "adsb", "track": 0.0})
    return fl, tr


def test_the_second_join_takes_a_single_candidate_aircraft_seen_under_the_callsign_before_mvt():
    """ADJ condition (1)-(3): an unjoined departure joins when EXACTLY ONE aircraft was on the ground under its
    callsign at its airport inside [MVT - 3 h, MVT). Kills: joining a row that already has a take-off join; joining on
    two candidates (guessing); ignoring the airport; ignoring the 3 h window; an anchor other than MVT."""
    fx = _load()
    fl, tr = _adj_world(fx)
    j, tr2 = fx.join_mvt_anchored(fl, tr)
    assert j.join_type[0] == "mvt" and j.t_takeoff[0] == pd.Timestamp("2025-01-02 02:46:40", tz="UTC").timestamp()
    assert np.isfinite(j.track_id[0]) and j.dt_takeoff[0] == 0.0
    assert (tr2.track_id == j.track_id[0]).sum() == 5                      # the aircraft's samples form one segment
    two = pd.concat([tr, tr.assign(icao="bbb222", track_id=78.0)], ignore_index=True)
    assert fx.join_mvt_anchored(fl, two)[0].join_type[0] is None            # two candidates: no guess
    assert fx.join_mvt_anchored(fl, tr.assign(airport="LSZH"))[0].join_type[0] is None
    assert fx.join_mvt_anchored(fl, tr.assign(t=tr.t - 20_000.0))[0].join_type[0] is None      # outside the 3 h window
    assert fx.join_mvt_anchored(fl, tr.assign(t=tr.t + 5_000.0))[0].join_type[0] is None       # at or after MVT
    joined = fl.assign(track_id=[5.0], join_type=["callsign"])
    assert fx.join_mvt_anchored(joined, tr)[0].join_type[0] == "callsign"   # today's join is never overwritten


def test_the_second_join_marks_its_rows_and_reports_how_stale_the_anchor_is():
    """The two ADJ features. Kills: the flag missing (the learner cannot discount these rows); the gap measured from
    the FIRST ground sample; a 0 written where there is no such join (BC-3)."""
    fx = _load()
    fl, tr = _adj_world(fx)
    j, tr2 = fx.join_mvt_anchored(fl, tr)
    f = fx.flight_features(j, tr2, fx.stand_centroids(j, tr2)).set_index("MVT_ID_mvt")
    assert f.join_mvt_anchored[1.0] == 1.0
    assert f.mvt_anchor_gap_s[1.0] == pytest.approx(800.0)                  # 10,000 - 9,200, the LAST ground sample
    fl2, tr3 = _adj_world(fx, cs="ZZZ999")
    fl2["CALLSIGN_flt"], fl2["FLIGHT_mvt"] = "KLM123", "KL123"              # no candidate under this flight's callsign
    j2, tr4 = fx.join_mvt_anchored(fl2, tr3)
    f2 = fx.flight_features(j2, tr4, fx.stand_centroids(j2, tr4)).set_index("MVT_ID_mvt")
    assert np.isnan(f2.join_mvt_anchored[1.0]) and np.isnan(f2.mvt_anchor_gap_s[1.0])
    for c in ("join_mvt_anchored", "mvt_anchor_gap_s"):
        assert c in fx.FEATURES


def test_a_row_joined_the_new_way_gets_the_ordinary_ground_features():
    """The point of the arm: the existing feature block must work on the recovered rows. Kills: the segment built from
    the wrong anchor (no ground samples found); coverage left at 1 for a row that is now joined."""
    fx = _load()
    fl, tr = _adj_world(fx)
    j, tr2 = fx.join_mvt_anchored(fl, tr)
    f = fx.flight_features(j, tr2, fx.stand_centroids(j, tr2)).set_index("MVT_ID_mvt")
    assert f.joined[1.0] == 1.0
    assert f.first_ground_rel_mvt[1.0] == pytest.approx(2000.0)            # the first ground sample, 2,000 s before MVT
    assert f.first_move_rel_mvt[1.0] == pytest.approx(1400.0)              # the first sample at gs >= 3 (4.0 kt)
    assert f.n_ground[1.0] == 5


def test_the_candidate_count_is_decided_by_the_airport_and_the_window_not_by_a_later_filter():
    """The 'exactly one aircraft' test must be applied to aircraft INSIDE the window at THIS airport. A second aircraft
    under the same callsign outside the window, after MVT, or at another airport must not make the row ambiguous.
    Kills: the window bounds or the airport key dropped from the candidate search (a later filter then hides it, but
    the candidate COUNT is already wrong and a good row is refused)."""
    fx = _load()
    fl, tr = _adj_world(fx)
    stale = tr.assign(icao="ccc333", track_id=79.0, t=tr.t - 20_000.0)          # same callsign, long before the window
    j, _ = fx.join_mvt_anchored(fl, pd.concat([tr, stale], ignore_index=True))
    assert j.join_type[0] == "mvt"                                              # still exactly one candidate
    later = tr.assign(icao="ddd444", track_id=80.0, t=tr.t + 5_000.0)           # same callsign, at/after MVT
    j2, _ = fx.join_mvt_anchored(fl, pd.concat([tr, later], ignore_index=True))
    assert j2.join_type[0] == "mvt"
    elsewhere = tr.assign(icao="eee555", track_id=81.0, airport="LSZH")         # same callsign, another airport
    j3, _ = fx.join_mvt_anchored(fl, pd.concat([tr, elsewhere], ignore_index=True))
    assert j3.join_type[0] == "mvt"


def test_the_recovered_segment_holds_only_samples_at_the_departure_airport():
    """A turnaround aircraft can have in-window samples at ANOTHER airport (it flew in from there). Those must not enter
    the recovered ground segment. Kills: the airport filter dropped when the aircraft's samples are collected — the
    segment then starts at the wrong airport and every ground feature is wrong."""
    fx = _load()
    fl, tr = _adj_world(fx)
    inbound = tr.assign(track_id=76.0, airport="LSZH", t=tr.t - 4_000.0)        # same aircraft, earlier, elsewhere
    j, tr2 = fx.join_mvt_anchored(fl, pd.concat([inbound, tr], ignore_index=True))
    assert j.join_type[0] == "mvt"
    seg = tr2[tr2.track_id == j.track_id[0]]
    assert len(seg) == 5 and set(seg.airport) == {"EHAM"}
    f = fx.flight_features(j, tr2, fx.stand_centroids(j, tr2)).set_index("MVT_ID_mvt")
    assert f.n_ground[1.0] == 5 and f.first_ground_rel_mvt[1.0] == pytest.approx(2000.0)


def test_build_uses_the_second_join_so_recovered_rows_reach_the_table():
    """End to end: a departure with no take-off event but with same-callsign ground samples must come out of build()
    joined, with a coverage code of 2 or 3. Kills: join_mvt_anchored implemented but never called by build(); the
    recovered rows left at coverage 1 (then the stacker can never move them)."""
    fx = _load()
    fl, tr = _adj_world(fx)
    adsb = tr.rename(columns={"track_id": "_t"}).drop(columns=["_t"]).assign(
        reg=None, type=None)                                       # raw observation columns, no track ids
    out = fx.build(fl[fx.FLIGHT_COLS], adsb)
    assert out.joined[0] == 1.0 and out.join_mvt_anchored[0] == 1.0
    assert fx.coverage_code(True, out)[0] in (2, 3)
    assert "v4" in fx.FEATURES_VERSION and "ADJ" in fx.FEATURES_VERSION


# ---------------------------------------------------------------- arm AFJ: the ground_rate_pm span floor (defence in depth)

def _two_ground_samples_then_takeoff(dt_s, t0=T0):
    """A joined departure whose final ground period holds exactly two samples dt_s apart, then the airborne sample."""
    t_to = t0 + dt_s + 30.0
    pts = [(t0, 0.0, 0.0, 1, 0.0), (t0 + dt_s, 0.0, 0.0, 1, 0.0), (t_to, 3100.0, 0.0, 0, 160.0)]
    tr = fx.segment_tracks(pd.DataFrame(_samples("g1", "KLM9", pts)))
    j = fx.join_flights(pd.DataFrame([_flight(91, "KLM9", t_to, t0 - 600.0)]), fx.takeoff_events(tr))
    assert j.track_id.notna().all()
    return fx.flight_features(j, tr).iloc[0]


def test_regression_ground_rate_pm_is_unknown_below_the_1ms_span_floor_and_bit_identical_above_it():
    """Regression (reports/RESEARCH_ADSBFI_JULY_2026_09_16.md §3; plans/PREREG_adsbfi_july_fallback_2026_09_16.md §2): two
    adsb.fi records of ONE fix 2.4e-7 s apart gave ground_rate_pm = 60 * 2 / 2.4e-7 = 503,316,480 and the MLP -1.9e6 s.
    Below SPAN_MIN_S = 1 ms (both sources timestamp at a 10 ms tick, so < 1 ms cannot separate two fixes) the rate is
    unknown, exactly as the existing `span > 0` rule treats exact duplicates. Above it NOTHING changes: the shipped tables'
    smallest span is 0.0099999 s (rate 12,000.01, 653 / 436 / 632 / 426 rows in the four shipped tables), so the floor must sit
    below 10 ms or shipped rows would move. Fails when the floor is removed (5e8 comes back), when it is set at the report's
    1 s (the 10 ms case would become NaN: a shipped-row change), when it is strict (the t0 = 0 equality case), or when the
    exact-duplicate NaN is lost. Rehearsed 2026-09-16 17:1x EDT (backup, break, run, restore byte-identical): floor removed
    -> RED; floor 1 s -> RED; strict > -> RED (after the t0 = 0 case was added; at epoch magnitude it is unreachable)."""
    assert fx.SPAN_MIN_S == 0.001
    blow = _two_ground_samples_then_takeoff(2.4e-7)
    assert blow.n_ground == 2 and np.isnan(blow.ground_rate_pm)
    assert np.isnan(_two_ground_samples_then_takeoff(0.0).ground_rate_pm)                # exact duplicate: as before
    assert np.isnan(_two_ground_samples_then_takeoff(0.0009).ground_rate_pm)             # just under the floor
    ship = _two_ground_samples_then_takeoff(0.01)
    assert ship.ground_rate_pm == 60.0 * 2 / (ship_span := (T0 + 0.01) - T0) and ship_span > 0.001    # the shipped formula, bit for bit
    assert ship.ground_rate_pm == pytest.approx(12_000.0, rel=1e-6)
    near = _two_ground_samples_then_takeoff(0.0012)                                       # just above the floor
    assert near.ground_rate_pm == 60.0 * 2 / ((T0 + 0.0012) - T0) and ((T0 + 0.0012) - T0) >= fx.SPAN_MIN_S
    assert _two_ground_samples_then_takeoff(60.0).ground_rate_pm == pytest.approx(2.0)
    # the floor is INCLUSIVE: at epoch magnitude a span of exactly 0.001 is not representable (float step 2.4e-7), so the
    # equality case is exercised at t0 = 0, where (0 + 0.001) - 0 is the same double as SPAN_MIN_S
    edge = _two_ground_samples_then_takeoff(0.001, t0=0.0)
    assert (0.0 + 0.001) - 0.0 == fx.SPAN_MIN_S and edge.ground_rate_pm == 60.0 * 2 / 0.001


# ---------------------------------------------------------------- arm ECF: per-event callsign (opt-in, default off)
# plans/PREREG_event_callsign_fix_2026_09_18.md. `takeoff_events(tr)` labels EVERY take-off of a track with the track's
# LAST non-null callsign; on union (two-source) days 68 % of events sit on tracks carrying >= 2 callsigns, so most events
# are named after some other flight. `per_event_callsign=True` names event i after the callsign in force at its own
# take-off sample. Default False must stay bit-identical to the shipped behaviour.

# Mutation rehearsal 2026-09-18 17:5x-18:1x EDT (each: mutate the REAL module, run `-k ecf`, restore byte-identical; POST-change
# sha256 d1a4964c...; outcomes in scratchpad ecf_mutations.json and in each docstring's "Rehearsed" line). Mutations:
#   m1 `<=` -> `<` (searchsorted on i, not i+1) | m2 last -> next non-null (ffill -> bfill) | m3 fallback removed |
#   m4 fallback to the LAST callsign | m5 flag ignored | m6 flag inverted | m7 whole-track groupby retained under True |
#   m8 default flipped to True | m9 equal-t tie-break ignored (position i+1) | m10 ffill not grouped by track |
#   m11 build drops the flag | m12a/b build_period drops the flag (day-present / day-absent branch) | m13 True path returns
#   object dtype | m14 fallback branches swapped | m15 equal-t run ignores the track boundary | m16 missing callsign ->
#   "" sentinel | m17 True path strips the callsign | m18 run_end drops the final run | m19 empty early return skipped
#   under True. All 20 RED (m2 only after test_ecf_takeoff_sample_without_a_callsign... was added: it had SURVIVED).

ECF_FIXTURE = ROOT / "tests" / "fixtures" / "ecf_takeoff_events_fi_2026-07-08.parquet"
ECF_DAY = ROOT / "data" / "adsb" / "fi" / "y2026" / "adsb_2026-07-08.parquet"


def _cs_samples(icao, pts, airport="EHAM"):
    """pts: list of (t, callsign, on_ground, gs) — one aircraft, callsign per SAMPLE (None = not reported)."""
    rows = []
    for t, cs, g, gs in pts:
        lat, lon = _ll(0.0, 0.0)
        rows.append(dict(icao=icao, reg=None, type=None, callsign=cs, airport=airport, t=float(t), lat=lat, lon=lon,
                         on_ground=int(g), alt_missing=0, alt=np.nan if g else 1500.0, gs=float(gs), track=0.0,
                         src="adsb_icao"))
    return rows


def _events(rows, **kw):
    return fx.takeoff_events(fx.segment_tracks(pd.DataFrame(rows)), **kw)


def _naive_event_callsigns(tr: pd.DataFrame, ev: pd.DataFrame) -> list:
    """The prereg's rule, written as a plain per-event loop (an independent oracle for the vectorised path)."""
    out = []
    for tid, tto in zip(ev.track_id, ev.t_takeoff):
        s = tr[tr.track_id == tid]
        before = s[s.t <= tto].callsign.dropna()
        any_cs = s.callsign.dropna()
        out.append(before.iloc[-1] if len(before) else (any_cs.iloc[0] if len(any_cs) else None))
    return out


def test_ecf_flag_defaults_to_false_and_is_keyword_optional():
    """The shipped call `takeoff_events(tr)` must keep meaning the whole-track callsign. Kills: default flipped to True;
    the parameter made required.
    Rehearsed 2026-09-18 EDT: RED on m8 (see the legend above)."""
    import inspect
    p = inspect.signature(fx.takeoff_events).parameters
    assert list(p) == ["tr", "per_event_callsign"] and p["per_event_callsign"].default is False


@pytest.mark.skipif(not (ECF_FIXTURE.exists() and ECF_DAY.exists()),
                    reason="real adsb.fi day file / captured fixture absent (data/ is not in git)")
def test_ecf_default_path_is_bit_identical_to_the_pre_ecf_output_on_a_real_day():
    """The hard requirement (prereg 'third shape'): the default reproduces the PRE-change function on a real day file —
    same rows, same order, same dtypes, same callsigns. The fixture was captured 2026-09-18 17:51 EDT from
    adsb_features.py sha256 11aa6399... (before any ECF edit; `tests/fixtures/ecf_takeoff_events_fi_2026-07-08.json`
    records the day file's sha256 and the dtypes), so the comparison is against the old code, not against itself.
    Invariant permitting a fixture update: NONE — a difference here is an ECF implementation failure, never a new
    baseline. Kills: default flipped to True (104 events sit on multi-callsign tracks that day); a dtype change of the
    callsign column; a reordering.
    Rehearsed 2026-09-18 EDT: RED on m6, m8 (see the legend above)."""
    import json
    meta = json.loads(ECF_FIXTURE.with_suffix(".json").read_text())
    want = pd.read_parquet(ECF_FIXTURE)
    assert len(want) == meta["n_events"] == 3617
    got = fx.takeoff_events(fx.segment_tracks(pd.read_parquet(ECF_DAY)))
    assert {c: str(t) for c, t in got.dtypes.items()} == meta["dtypes"]
    pd.testing.assert_frame_equal(got, want, check_exact=True, check_dtype=True, check_index_type=True)


@pytest.mark.skipif(not ECF_DAY.exists(), reason="real adsb.fi day file absent (data/ is not in git)")
def test_ecf_true_path_on_a_real_day_changes_only_callsigns_and_matches_the_naive_oracle():
    """Real data: the opt-in path keeps every event (track, icao, airport, t_takeoff, order) and only re-names some; the
    changed events all sit on tracks with >= 2 distinct callsigns (2.9 % = 104 events on this adsb.fi day); and every
    callsign equals the prereg rule computed by an independent per-event loop. Kills: a vectorised path that drops or
    re-orders events, or that reads the wrong track / the wrong side of t_takeoff.
    Rehearsed 2026-09-18 EDT: RED on m1, m3, m5, m6, m7, m8, m10, m13, m14 (see the legend above)."""
    tr = fx.segment_tracks(pd.read_parquet(ECF_DAY))
    old, new = fx.takeoff_events(tr), fx.takeoff_events(tr, per_event_callsign=True)
    pd.testing.assert_frame_equal(old.drop(columns="callsign"), new.drop(columns="callsign"), check_exact=True)
    assert str(old.callsign.dtype) == str(new.callsign.dtype)
    changed = old.callsign.fillna("<NA>").to_numpy() != new.callsign.fillna("<NA>").to_numpy()
    multi = tr.groupby("track_id").callsign.nunique()
    assert 0 < changed.sum() <= (new.track_id.map(multi) >= 2).sum() == 104
    assert new.callsign.fillna("<NA>").tolist() == [c if c is not None else "<NA>" for c in _naive_event_callsigns(tr, new)]


def test_ecf_two_takeoffs_on_one_track_each_get_their_own_callsign():
    """The defect itself: one track (no 30-minute gap), two departures under two callsigns. Default: both events carry
    the LAST callsign (today's behaviour, pinned). True: each event carries the callsign in force at its take-off.
    Kills: the flag ignored; the whole-track groupby retained under True; last -> first (both would read AAA1).
    Rehearsed 2026-09-18 EDT: RED on m5, m6, m7, m8, m14, m18 (see the legend above)."""
    rows = _cs_samples("a1", [(T0, "AAA1", 1, 0), (T0 + 60, "AAA1", 1, 30), (T0 + 90, "AAA1", 0, 150),
                              (T0 + 900, "BBB2", 1, 0), (T0 + 960, "BBB2", 1, 30), (T0 + 990, "BBB2", 0, 150)])
    old, new = _events(rows), _events(rows, per_event_callsign=True)
    assert old.track_id.nunique() == 1 and old.t_takeoff.tolist() == [T0 + 90, T0 + 990]
    assert old.callsign.tolist() == ["BBB2", "BBB2"]
    assert new.callsign.tolist() == ["AAA1", "BBB2"]


def test_ecf_callsign_reported_only_after_the_takeoff_falls_back_to_the_tracks_first_callsign():
    """A take-off before the track ever reported a callsign takes the track's FIRST non-null callsign (the prereg's
    fallback), not the last one. Kills: fallback removed (None); fallback to the last callsign (END); bfill instead
    of the first.
    Rehearsed 2026-09-18 EDT: RED on m3, m4, m5, m6, m7, m8, m14 (see the legend above)."""
    rows = _cs_samples("a1", [(T0, None, 1, 0), (T0 + 60, None, 0, 150),                     # take-off 1, unnamed
                              (T0 + 600, "MID", 1, 0), (T0 + 660, "MID", 0, 150),            # take-off 2, named MID
                              (T0 + 1200, "END", 1, 0)])                                       # later, another name
    new = _events(rows, per_event_callsign=True)
    assert new.callsign.tolist() == ["MID", "MID"]
    assert _events(rows).callsign.tolist() == ["END", "END"]


def test_ecf_track_without_any_callsign_gives_none_under_both_flags_with_the_same_dtype():
    """No callsign ever: None (missing) as today, and the column dtype is the same as the default path's. Kills: a
    fallback that invents a value; a True path whose missing marker or dtype differs from the default's.
    Rehearsed 2026-09-18 EDT: RED on m16, m18 (see the legend above)."""
    rows = _cs_samples("a1", [(T0, None, 1, 0), (T0 + 60, None, 0, 150)])
    old, new = _events(rows), _events(rows, per_event_callsign=True)
    assert len(old) == len(new) == 1 and old.callsign.isna().all() and new.callsign.isna().all()
    assert str(old.callsign.dtype) == str(new.callsign.dtype)
    pd.testing.assert_frame_equal(old, new, check_exact=True)


def test_ecf_single_callsign_track_is_unchanged_under_the_flag():
    """One callsign, one take-off: both flags agree exactly (frame, dtypes, index). Kills: a True path that alters
    dtype, index or row order even when nothing needs renaming.
    Rehearsed 2026-09-18 EDT: RED on m13 (see the legend above)."""
    rows = _departure("a1", "KLM1", T0, 600, [(T0 + 700, 100, 0, 10)], T0 + 1500)
    old, new = _events(rows), _events(rows, per_event_callsign=True)
    assert old.callsign.tolist() == ["KLM1"]
    pd.testing.assert_frame_equal(old, new, check_exact=True, check_dtype=True, check_index_type=True)


def test_ecf_padded_callsign_is_returned_raw_and_the_join_normalises_it():
    """The event carries the sample's callsign VERBATIM (padding / lower case kept); `_norm_cs` in join_flights strips
    it as today, so the flight still joins by callsign. Kills: normalising inside takeoff_events (the default path
    does not, and the two paths must agree on the string); a raw value that `_norm_cs` cannot match.
    Rehearsed 2026-09-18 EDT: RED on m17, m18 (see the legend above)."""
    mvt = T0 + 5000
    rows = _cs_samples("a1", [(mvt - 600, "klm1  ", 1, 0), (mvt - 300, "klm1  ", 1, 20), (mvt, "klm1  ", 0, 150)])
    new = _events(rows, per_event_callsign=True)
    assert new.callsign.tolist() == ["klm1  "] and fx._norm_cs(new.callsign.iloc[0]) == "KLM1"
    j = fx.join_flights(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 800)]), new)
    assert j.iloc[0].join_type == "callsign"


def test_ecf_events_at_the_first_and_the_last_sample_of_the_frame():
    """A take-off whose ground sample is the very first row of its track (and of the frame) and one whose airborne
    sample is the very last row of the frame: both are named without an index error. Kills: an off-by-one at either
    end of the run / break arrays.
    Rehearsed 2026-09-18 EDT: RED on m18 (see the legend above)."""
    rows = (_cs_samples("a1", [(T0, "FIRST", 1, 0), (T0 + 5, "FIRST", 0, 150)])
            + _cs_samples("b2", [(T0 + 100, "LAST", 1, 0), (T0 + 105, None, 0, 150)]))
    new = _events(rows, per_event_callsign=True)
    assert new.icao.tolist() == ["a1", "b2"] and new.callsign.tolist() == ["FIRST", "LAST"]


def test_ecf_boundary_the_only_callsign_sample_is_exactly_at_t_takeoff():
    """`t <= t_takeoff` INCLUDES the first airborne sample. Pinned two ways: (1) the only callsign of the track is on
    the sample at t_takeoff -> that callsign; (2) an EARLY callsign, then none, then ATTO exactly at t_takeoff, then
    LATER -> ATTO. Case 2 is the one that bites: under `<` the answer would be EARLY (case 1 alone would be rescued by
    the fallback). Kills: `<=` -> `<`; reading the airborne sample's successor (LATER).
    Rehearsed 2026-09-18 EDT: RED on m1, m5, m6, m7, m14 (see the legend above)."""
    only = _cs_samples("a1", [(T0, None, 1, 0), (T0 + 30, None, 1, 20), (T0 + 60, "ATTO", 0, 150), (T0 + 90, None, 0, 200)])
    assert _events(only, per_event_callsign=True).callsign.tolist() == ["ATTO"]
    rows = _cs_samples("a1", [(T0, "EARLY", 1, 0), (T0 + 30, None, 1, 20), (T0 + 60, "ATTO", 0, 150),
                              (T0 + 90, "LATER", 0, 200)])
    assert _events(rows, per_event_callsign=True).callsign.tolist() == ["ATTO"]


def test_ecf_equal_timestamp_samples_at_t_takeoff_count_as_at_or_before_in_frame_order():
    """Two sources can record one fix at the SAME t (union days). A second sample at exactly t_takeoff that carries a
    callsign is `<= t_takeoff`, so it is the last one in frame order and wins. Kills: reading position i+1 instead of
    the last position with t <= t_takeoff.
    Rehearsed 2026-09-18 EDT: RED on m1, m5, m6, m7, m9, m14 (see the legend above)."""
    rows = _cs_samples("a1", [(T0, "EARLY", 1, 0), (T0 + 60, None, 0, 150), (T0 + 60, "DUP", 0, 150),
                              (T0 + 90, "LATER", 0, 200)])
    new = _events(rows, per_event_callsign=True)
    assert len(new) == 1 and new.callsign.tolist() == ["DUP"]


def test_ecf_callsigns_never_leak_across_tracks():
    """The forward-fill is per track: aircraft b2 (no callsign before its take-off, none at all) must not inherit a1's
    callsign, and a1's second track (after a 30-minute gap) must not inherit its first track's callsign. Kills: a global
    (ungrouped) ffill; a fallback that reads another track.
    Rehearsed 2026-09-18 EDT: RED on m10, m16, m18 (see the legend above)."""
    rows = (_cs_samples("a1", [(T0, "AAA1", 1, 0), (T0 + 60, "AAA1", 0, 150),
                               (T0 + 60 + 1801, None, 1, 0), (T0 + 60 + 1861, None, 0, 150)])
            + _cs_samples("b2", [(T0 + 100, None, 1, 0), (T0 + 160, None, 0, 150)]))
    new = _events(rows, per_event_callsign=True)
    assert new.icao.tolist() == ["a1", "a1", "b2"]
    assert new.callsign.fillna("<NA>").tolist() == ["AAA1", "<NA>", "<NA>"]


def test_ecf_no_events_returns_the_same_empty_frame_under_the_flag():
    """Airborne-only input: the empty frame with the five columns, whichever flag. Kills: the True path touching the
    empty early return.
    Rehearsed 2026-09-18 EDT: RED on m19 (see the legend above)."""
    rows = _samples("c3", "Z", [(T0, 0, 0, 0, 200), (T0 + 30, 900, 0, 0, 210)])
    old, new = _events(rows), _events(rows, per_event_callsign=True)
    assert list(new.columns) == ["track_id", "icao", "callsign", "airport", "t_takeoff"] and len(new) == 0
    pd.testing.assert_frame_equal(old, new)


def test_ecf_flag_is_threaded_through_build_and_build_period(monkeypatch, tmp_path):
    """`build(..., per_event_callsign=True)` and `build_period(..., per_event_callsign=True)` reach takeoff_events with
    True (spied), on the day-file-present branch AND the day-file-absent branch; the defaults reach it with False.
    Kills: build or build_period not passing the flag (either branch); the flag inverted on the way.
    Rehearsed 2026-09-18 EDT: RED on m11, m12a, m12b (see the legend above)."""
    import inspect
    seen = []
    orig = fx.takeoff_events

    def spy(*a, **k):
        b = inspect.signature(orig).bind(*a, **k)
        b.apply_defaults()
        seen.append(b.arguments["per_event_callsign"])
        return orig(*a, **k)

    monkeypatch.setattr(fx, "takeoff_events", spy)
    mvt = T0 + 5000
    adsb = pd.DataFrame(_departure("a1", "KLM1", mvt - 1800, 900, [(mvt - 850, 60, 0, 8)], mvt))
    fl = pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 900)])
    fx.build(fl, adsb)
    fx.build(fl, adsb, per_event_callsign=True)
    assert seen == [False, True]
    day = pd.Timestamp(mvt, unit="s", tz="UTC").normalize()
    adsb.to_parquet(tmp_path / f"adsb_{day:%Y-%m-%d}.parquet", index=False)
    absent = _flight(2, "KLM2", mvt + 30 * 86_400, mvt + 30 * 86_400 - 900)         # a day with no file
    two = pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 900), absent])
    seen.clear()
    fx.build_period(two, tmp_path, git_sha="test")
    assert seen == [False, False]
    seen.clear()
    fx.build_period(two, tmp_path, git_sha="test", per_event_callsign=True)
    assert seen == [True, True]


def test_ecf_equal_timestamp_on_the_next_track_is_not_read():
    """Adjacent tracks in frame order (a1 then b2) with b2's first sample at EXACTLY a1's t_takeoff: the equal-t run
    must stop at the track boundary, so a1's event is not named BBB2. Kills: the track test dropped from the equal-t
    run (`same_fix` on t alone).
    Rehearsed 2026-09-18 EDT: RED on m15, m18 (see the legend above)."""
    rows = (_cs_samples("a1", [(T0, "AAA1", 1, 0), (T0 + 60, None, 0, 150)])
            + _cs_samples("b2", [(T0 + 60, "BBB2", 1, 0), (T0 + 160, None, 0, 150)]))
    new = _events(rows, per_event_callsign=True)
    assert new.icao.tolist() == ["a1", "b2"] and new.callsign.tolist() == ["AAA1", "BBB2"]


def test_ecf_takeoff_sample_without_a_callsign_reads_the_last_earlier_one_not_the_next():
    """The airborne sample at t_takeoff carries no callsign; EARLY was reported on the ground before it and LATER
    after it: the event is EARLY (last non-null at or before), never LATER. This is the shape that separates a
    forward-fill from a backward-fill. Kills: ffill -> bfill (m2, which survived every other fixture until this one).
    Rehearsed 2026-09-18 EDT: RED on m2 (SURVIVED before this test existed) (see the legend above)."""
    rows = _cs_samples("a1", [(T0, "EARLY", 1, 0), (T0 + 30, None, 1, 20), (T0 + 60, None, 0, 150),
                              (T0 + 90, "LATER", 0, 200), (T0 + 120, "LATER", 0, 200)])
    assert _events(rows, per_event_callsign=True).callsign.tolist() == ["EARLY"]
    assert _events(rows).callsign.tolist() == ["LATER"]


# ---------------------------------------------------------------- take-off roll reported airborne before MVT (ODX, 2026-09-25)

def _rolling_departure(mvt, roll_pts):  # used by the flicker test (callsign join path)
    """Parked 900 s at the stand, taxi 600 s at 12 kt, then `roll_pts` (t_rel_mvt, on_ground, gs) on the runway."""
    pts = [(mvt - 1500 + k * 30, 0, 0, 1, 0.0) for k in range(31)]
    pts += [(mvt - 590 + k * 60, 100 * (k + 1), 0, 1, 12.0) for k in range(9)]
    pts += [(mvt + dt, 2000 + 50 * i, 0, g, gs) for i, (dt, g, gs) in enumerate(roll_pts)]
    return _samples("a1", "KLM1", pts)


def test_take_off_roll_reported_airborne_before_mvt_keeps_the_whole_ground_segment():
    """EHAM (below sea level) reports on_ground=0 from ~53 kt on the take-off roll, before MVT (MVT_ID 191611864:
    alt -50 ft at -24.6 s). On an MVT-anchored (ADJ) join the ground period ends at MVT, so those roll samples lie
    inside it. Kills: cutting the ground segment after them, which empties it (28,948 2025 LANE M rows went to
    coverage 2 with every ground feature NaN)."""
    fx = _load()
    fl, tr = _adj_world(fx)
    mvt_s = pd.Timestamp("2025-01-02 02:46:40", tz="UTC").timestamp()
    roll = pd.DataFrame({"track_id": 77.0, "icao": "aaa111", "airport": "EHAM", "callsign": "KLM123",
                         "t": mvt_s - np.array([60.0, 25.0, 10.0]), "on_ground": [1, 0, 0], "alt_missing": 0,
                         "alt": [np.nan, -50.0, 100.0], "gs": [45.0, 90.0, 130.0], "lat": 52.31, "lon": 4.76,
                         "src": "adsb", "track": 0.0})
    j, tr2 = fx.join_mvt_anchored(fl, pd.concat([tr, roll], ignore_index=True))
    assert j.join_type[0] == "mvt"
    f = fx.flight_features(j, tr2, fx.stand_centroids(j, tr2)).set_index("MVT_ID_mvt")
    assert f.turn_in_track[1.0] == 0
    assert f.first_ground_rel_mvt[1.0] == pytest.approx(2000.0)
    assert f.mvt_anchor_gap_s[1.0] == pytest.approx(60.0)                   # the last GROUND sample, on the roll


def test_on_ground_flicker_during_the_roll_is_not_a_landing():
    """air / ground / air flicker at roll speed: the ground sample after an 'airborne' one is fast (> taxi speed),
    so it is not a touchdown. Kills: a landing rule that only asks for ANY later ground sample."""
    mvt = T0 + 5000
    s = _rolling_departure(mvt, [(-40, 0, 60.0), (-30, 1, 80.0), (-20, 0, 110.0), (-12, 1, 125.0), (0, 0, 150.0)])
    f = fx.build(pd.DataFrame([_flight(1, "KLM1", mvt, mvt - 600)]), pd.DataFrame(s)).iloc[0]
    assert f.turn_in_track == 0
    assert f.first_ground_rel_mvt == pytest.approx(1500, abs=1)


def test_a_flickering_take_off_roll_is_one_take_off_event_at_the_last_transition():
    """Sibling of the take-off-roll cut (same class: a noisy on_ground flag used as a hard boundary). A roll whose
    on_ground flag flickers air/ground/air at >= TAKEOFF_MIN_GS made 2-3 'take-offs' on one track: the time join
    (exactly one candidate within JOIN_TIME_S) then refused the row. Kills: one event per flicker; keeping the FIRST
    flicker (the lift-off is the last ground->air transition of the roll)."""
    mvt = T0 + 5000
    s = pd.DataFrame(_rolling_departure(mvt, [(-40, 0, 60.0), (-30, 1, 80.0), (-20, 0, 110.0), (-12, 1, 125.0),
                                              (0, 0, 150.0), (30, 0, 170.0)]))
    ev = fx.takeoff_events(fx.segment_tracks(s))
    assert len(ev) == 1 and ev.t_takeoff.iat[0] == pytest.approx(mvt, abs=1)      # rel default = ~1,700 s at epoch scale


def test_two_real_take_offs_of_one_track_far_apart_stay_two_events():
    """The roll merge must not swallow a real second departure of the same airframe (touch-and-go / quick turn):
    transitions more than ROLL_MERGE_S apart are separate events. Kills: merging every transition of a track."""
    mvt = T0 + 5000
    pts = [(mvt - 3000 + k * 30, 0, 0, 1, 0.0) for k in range(10)]
    pts += [(mvt - 2700, 2000, 0, 1, 50.0), (mvt - 2690, 2100, 0, 0, 150.0)]               # first take-off
    pts += [(mvt - 1500, -3000, 0, 1, 120.0), (mvt - 1400, 0, 0, 1, 10.0)]                  # lands, taxis
    pts += [(mvt - 1300 + k * 30, 0, 0, 1, 0.0) for k in range(40)]
    pts += [(mvt - 10, 2000, 0, 1, 50.0), (mvt, 2100, 0, 0, 150.0)]                         # second take-off
    ev = fx.takeoff_events(fx.segment_tracks(pd.DataFrame(_samples("a1", "KLM1", pts))))
    assert sorted(ev.t_takeoff.round().tolist()) == [round(mvt - 2690), round(mvt)]


def test_filter_sites_only_drop_single_airborne_samples_never_a_period():
    """Protected siblings (category b of the 2026-09-25 post-mortem): join_mvt_anchored's candidate set, the
    centroid parked-position mask and the congestion counts FILTER on on_ground == 1, sample by sample. A flipped
    sample can only remove itself; it never re-anchors the rest of the period. Invariant pinned here: one flipped
    roll sample leaves the ADJ join and its candidate aircraft unchanged."""
    fx2 = _load()
    fl, tr = _adj_world(fx2)
    flipped = tr.copy()
    flipped.loc[flipped.index[2], "on_ground"] = 0
    j0, _ = fx2.join_mvt_anchored(fl, tr)
    j1, _ = fx2.join_mvt_anchored(fl, flipped)
    assert j0.join_type[0] == j1.join_type[0] == "mvt" and j0.t_takeoff[0] == j1.t_takeoff[0]


def test_the_features_stamp_marks_the_take_off_roll_fix():
    """Tables built before and after the 2026-09-25 take-off-roll fix differ on ~29 k 2025 rows; the stamp must tell
    them apart so a pre-fix cache is never mixed with post-fix serving. Kills: a fix shipped under the old stamp."""
    assert "TRC" in fx.FEATURES_VERSION and "v4" in fx.FEATURES_VERSION


# ---------------------------------------------------------------- DWG post-mortem (BC-26, 2026-09-27)
# The "gate dwell" was the FIRST stationary run >= 300 s ANYWHERE in the ground segment (flight_features, 2026-09-26
# l.399-406). When the stand is not visible it is the post-pushback hold (EGLL Jul 2026: median 499 s, ending +611 s
# after AOBT_3, 103 m off the stand; 2025 training: 47 % of > 100 m dwells start after AOBT_3 + 60 s). Fix at the
# origin: a run is the gate dwell only if it STARTS at or before AOBT_3 + DWELL_START_MAX_S and, when a PARKED-provenance
# stand reference exists, its median lies within GATE_RADIUS_M of it. Scratch origin:
# logs/adsbshift_scratch/test_red_dwell_is_not_at_the_gate.py (moved here).

def _run(t0, dur, x, y, step=20, gs=0.0):
    """Stationary samples at (x, y) from t0 for exactly dur seconds (dur a multiple of step)."""
    return [(t0 + s * step, x, y, 1, gs) for s in range(int(dur // step) + 1)]


def _parked_others(n=3, xy=(0.0, 0.0), t0=T0 + 20_000, first=10, prefix="o"):
    """n OTHER departures seen PARKED (gs 0) at xy around their AOBT_3: parked-provenance stand references."""
    rows, fl = [], []
    for k in range(n):
        m = t0 + k * 5000
        rows += _departure(f"{prefix}{k}", f"{prefix.upper()}T{k}", m - 2400, 1500,
                           [(m - 870, xy[0] + 60, xy[1], 8), (m - 600, 900, 0, 15)], m, stand_xy=xy)
        fl.append(_flight(first + k, f"{prefix.upper()}T{k}", m, m - 900))
    return rows, fl


def _fallback_others(n=3, xy=(0.0, 0.0), t0=T0 + 40_000, first=20, prefix="f"):
    """n OTHER departures never seen parked: first seen at the stand at 2 kt (the _parked_positions FIRST-APPEARANCE
    fallback), then taxiing."""
    rows, fl = [], []
    for k in range(n):
        m = t0 + k * 4000
        pts = [(m - 800 + s * 10, xy[0], xy[1], 1, 2.0) for s in range(3)]
        pts += [(m - 700 + s * 30, xy[0] + 100 * (s + 1), xy[1], 1, 14.0) for s in range(20)]
        pts += [(m - 5, 3000, 0, 1, 140.0), (m, 3100, 0, 0, 160.0)]
        rows += _samples(f"{prefix}{k}", f"{prefix.upper()}T{k}", pts)
        fl.append(_flight(first + k, f"{prefix.upper()}T{k}", m, m - 780))
    return rows, fl


def _target(pts, mvt, aobt3, cs="TGT1"):
    """The flight under test (MVT_ID 1): pts, then a take-off at mvt."""
    rows = _samples("tgt", cs, pts + [(mvt - 5, 3000, 0, 1, 140.0), (mvt, 3100, 0, 0, 160.0)])
    return rows, _flight(1, cs, mvt, aobt3)


def _world():
    """Scratch red test's world: 3 others parked at S1 (0, 0); the target is first seen at pushback (2 kt, at the stand,
    30 s), then HOLDS 420 s 150 m off the stand starting after AOBT_3, then taxis and takes off."""
    rows, fl = _parked_others()
    mvt = T0 + 5000
    aobt3 = mvt - 1200
    pts = [(aobt3 + 10 + s * 10, 0.0, 0.0, 1, 2.0) for s in range(3)]                 # pushback at the stand, 2 kt
    pts += [(aobt3 + 150 + s * 20, 0.0, 150.0, 1, 0.0) for s in range(22)]            # 420 s hold, 150 m off stand
    pts += [(aobt3 + 600 + s * 30, 200 + 100 * s, 150.0, 1, 14.0) for s in range(18)]  # taxi
    r, f = _target(pts, mvt, aobt3)
    return pd.DataFrame(fl + [f]), pd.DataFrame(rows + r), aobt3


DWELL_COLS = ["dwell_end_rel_mvt", "dwell_s", "dwell_end_rel_aobt3", "n_stops", "stop_s", "path_m", "taxi_med_gs",
              "moving_dwell_end", "dwell_to_centroid_m"]


def test_regression_bug_26_post_push_hold_is_not_a_gate_dwell():
    """BC-26 (reports/bug_classes.md), surfaced 2026-09-26 (EGLL Jul 2026 store features), origin
    scripts/adsb_features.py flight_features l.399-406 (pre-fix). RED on the pre-fix code: dwell_s = 420,
    dwell_end_rel_aobt3 = +570, dwell_to_centroid_m = 150.
    The hold both starts after AOBT_3 and lies 150 m off, so either gate alone makes this green: it goes RED only with
    BOTH gates removed (= the pre-fix code, seen RED 2026-09-27). Each gate's own mutants (removed, '<', +1 s, +10 m,
    mixed/fallback ref) are killed by the per-gate tests below: 16/16 (logs/r12_scratch/mutation_dwg_20260927.log)."""
    fl, adsb, _ = _world()
    f = fx.build(fl, adsb).set_index("MVT_ID_mvt")
    assert f.loc[1, "stand_others_n"] == 3                                             # fixture sanity: centroid exists
    assert f.loc[1, "first_ground_to_stand_m"] == pytest.approx(0.0, abs=1.0)          # ... at the stand (0, 0)
    tgt = f.loc[1]
    assert tgt.joined == 1 and tgt.first_ground_slow == 1                              # fixture sanity: seen at pushback
    assert np.isnan(tgt.dwell_s), f"post-push hold counted as a gate dwell: dwell_s={tgt.dwell_s}"
    assert np.isnan(tgt.dwell_end_rel_aobt3)
    assert np.isnan(tgt.dwell_to_centroid_m)
    assert tgt.stand_visible == 0


def test_a_real_gate_dwell_is_still_measured():
    """Guard for the fix: the other flights' true stand dwells keep their values and are flagged stand_visible.
    Kills a fix that drops every dwell, or one that gates on the flight's OWN position (it is not in its LOO ref)."""
    fl, adsb, _ = _world()
    f = fx.build(fl, adsb).set_index("MVT_ID_mvt")
    for k in (10, 11, 12):
        assert f.loc[k, "dwell_s"] == pytest.approx(1500, abs=31)
        assert f.loc[k, "dwell_to_centroid_m"] == pytest.approx(0.0, abs=1.0)
        assert f.loc[k, "stand_visible"] == 1


def _hold_world(with_mover=True):
    """Target first seen MID-TAXI (14 kt) after AOBT_3, holds 420 s 150 m off the parked stand ref, then taxis with a
    90 s stop and takes off; a second aircraft taxis near the hold's end (so moving_dwell_end is measurable)."""
    rows, fl = _parked_others()
    mvt = T0 + 5000
    aobt3 = mvt - 1300
    pts = [(aobt3 + 100 + s * 10, 400 - 100 * s, 150.0, 1, 14.0) for s in range(3)]   # first seen taxiing
    pts += _run(aobt3 + 150, 420, 0.0, 150.0)                                         # the post-push hold
    pts += [(aobt3 + 600, 300, 150, 1, 12.0), (aobt3 + 630, 600, 150, 1, 14.0)]
    pts += _run(aobt3 + 660, 90, 800.0, 150.0, step=30)                                # a 90 s stop
    pts += [(aobt3 + 780, 1200, 150, 1, 15.0), (aobt3 + 900, 2000, 150, 1, 16.0)]
    r, f = _target(pts, mvt, aobt3)
    rows += r
    if with_mover:
        rows += _samples("mv1", "MOV1", [(aobt3 + 470 + s, 2000 + 10 * s, -500, 1, 12.0) for s in range(0, 200, 10)])
    return pd.DataFrame(fl + [f]), pd.DataFrame(rows)


def test_regression_bug_26_every_dwell_anchored_column_is_unknown_without_a_gate_dwell():
    """Sibling sweep (category a): every column anchored on the dwell's end inherits the gate. Pre-fix values on this
    fixture: dwell_s 420, n_stops 1, stop_s 90, path_m ~2,000, taxi_med_gs ~14, moving_dwell_end 1,
    dwell_to_centroid_m 150, coverage 3 (a mid-taxi first sample credited as 'seen from the stand').
    Kills: gating dwell_s alone while n_stops / path_m / taxi_med_gs / moving_dwell_end keep reading the hold."""
    fl, adsb = _hold_world()
    buggy_guard = fx.build(fl, adsb).set_index("MVT_ID_mvt").loc[1]
    assert buggy_guard.first_ground_slow == 0 and buggy_guard.joined == 1               # fixture sanity
    for c in DWELL_COLS:
        assert np.isnan(buggy_guard[c]), f"{c} = {buggy_guard[c]} read from a post-push hold"
    assert buggy_guard.stand_visible == 0


def test_regression_bug_26_coverage_3_needs_a_gate_dwell_or_a_slow_first_sample(tmp_path):
    """coverage 3 means 'seen from the stand'. A mid-taxi first sample plus a post-push hold is coverage 2.
    Kills: coverage_code crediting any >= 300 s stationary run."""
    fl, adsb = _hold_world(with_mover=False)
    day = pd.Timestamp(T0 + 5000, unit="s", tz="UTC").strftime("%Y-%m-%d")
    _write_day(tmp_path, day, adsb.to_dict("records"))
    out = fx.build_period(fl, tmp_path, git_sha="x").set_index("MVT_ID_mvt")
    assert out.loc[1, "coverage"] == 2
    assert out.loc[10, "coverage"] == 3                                                # a real stand dwell stays 3


def _single(start_rel_aobt3, xy=(0.0, 0.0), dur=420):
    """One flight, no stand reference (no other flights): a stationary run of dur s at xy starting at AOBT_3 + start."""
    mvt = T0 + 5000
    aobt3 = mvt - 1500
    pts = _run(aobt3 + start_rel_aobt3, dur, *xy)
    t1 = aobt3 + start_rel_aobt3 + dur
    pts += [(t1 + 30, 300, 0, 1, 12.0), (t1 + 90, 1200, 0, 1, 15.0)]
    r, f = _target(pts, mvt, aobt3)
    return fx.build(pd.DataFrame([f]), pd.DataFrame(r)).iloc[0]


def test_dwell_start_gate_is_inclusive_at_aobt3_plus_60_s_and_needs_no_stand_reference():
    """The time gate alone (no stand ref exists): a run starting exactly at AOBT_3 + 60 s is a gate dwell; one starting
    at + 61 s is not. Anchor: 2025 R12 train table, dwells <= 50 m from the stand start after AOBT_3 + 60 s in 2.6 %;
    dwells > 100 m off in 47 % (reports/bug_classes.md BC-26, 'Anchor').
    Kills: '<' for '<=' (the +60 row goes NaN); a looser constant (the +61 row survives); the gate dropped."""
    on = _single(60.0)
    off = _single(61.0)
    assert np.isnan(on.stand_others_n) or on.stand_others_n == 0                       # fixture sanity: no stand ref
    assert on.dwell_s == pytest.approx(420, abs=1) and on.stand_visible == 1
    assert on.dwell_end_rel_aobt3 == pytest.approx(480, abs=1)
    assert np.isnan(off.dwell_s) and off.stand_visible == 0
    early = _single(-600.0)                           # (earlier starts would leave a > 30 min gap: a new track)
    assert early.dwell_s == pytest.approx(420, abs=1) and early.dwell_end_rel_aobt3 == pytest.approx(-180, abs=1)


def _gated_by_location(dy_m, others="parked"):
    """Target stationary 420 s at (0, dy_m) BEFORE AOBT_3 (the time gate passes), stand refs at (0, 0)."""
    if others == "parked":
        rows, fl = _parked_others()
    elif others == "fallback":
        rows, fl = _fallback_others()
    else:                                                     # "mixed": 2 parked at (0, 0), 2 first-appearance at (0, 300)
        rows, fl = _parked_others(n=2)
        r2, f2 = _fallback_others(n=2, xy=(0.0, 300.0))
        rows, fl = rows + r2, fl + f2
    mvt = T0 + 5000
    aobt3 = mvt - 900
    pts = _run(aobt3 - 600, 420, 0.0, dy_m)
    pts += [(aobt3 + 30, 300, dy_m, 1, 12.0), (aobt3 + 120, 1200, 0, 1, 15.0)]
    r, f = _target(pts, mvt, aobt3)
    return fx.build(pd.DataFrame(fl + [f]), pd.DataFrame(rows + r)).set_index("MVT_ID_mvt")


def test_dwell_location_gate_uses_the_parked_stand_reference_and_gate_radius():
    """With a PARKED stand ref at (0, 0): a pre-AOBT_3 run 95 m away is the gate dwell, one 105 m away is not (a remote
    hold / another stand). Kills: the location gate removed; the radius loosened or tightened by > 5 m."""
    near = _gated_by_location(95.0).loc[1]
    far = _gated_by_location(105.0).loc[1]
    assert near.stand_ref_parked_share == 1.0                                           # fixture sanity: parked ref
    assert near.dwell_s == pytest.approx(420, abs=1) and near.stand_visible == 1
    assert np.isnan(far.dwell_s) and far.stand_visible == 0
    for c in DWELL_COLS:
        assert np.isnan(far[c]), c


def test_dwell_location_gate_uses_only_the_parked_others_not_the_mixed_reference():
    """2 others parked at (0, 0), 2 first-appearance (fallback) others at (0, 300): the mixed LOO median is ~150 m off,
    the parked-only reference is (0, 0). A dwell at (0, 0) is a gate dwell. Kills: gating on the mixed centroid."""
    f = _gated_by_location(0.0, others="mixed").loc[1]
    assert f.stand_others_n == 4 and f.stand_ref_parked_share == pytest.approx(0.5)
    assert f.dwell_s == pytest.approx(420, abs=1) and f.stand_visible == 1


def test_a_first_appearance_stand_reference_never_validates_or_rejects_a_dwell():
    """Main-session ruling 2026-09-27: a stand ref built by the _parked_positions FIRST-APPEARANCE fallback is where
    aircraft first appear, not the stand, so it must not decide a dwell. Category (b) invariant: the fallback itself is
    unchanged (the centroid users still read it) and its provenance is reported (share 0).
    Kills: the fallback centroid used for the location gate (the 150 m dwell would go NaN)."""
    f = _gated_by_location(150.0, others="fallback").loc[1]
    assert f.stand_others_n == 3 and f.stand_ref_parked_share == 0.0
    assert f.dwell_s == pytest.approx(420, abs=1) and f.stand_visible == 1
    assert f.first_ground_to_stand_m == pytest.approx(150.0, abs=2.0)                   # centroid users: unchanged ref
    assert f.dwell_to_centroid_m == pytest.approx(150.0, abs=2.0)


def test_stand_reference_provenance_is_the_parked_share_of_the_loo_others():
    """stand_ref_parked_share = share of the OTHER flights behind the LOO stand ref seen parked (not first-appearance);
    NaN when there is no ref (fewer than CENTROID_MIN_OTHERS others; BC-3: unknown, not 0).
    Kills: counting the flight itself; a 0 for 'no reference'."""
    rows, fl = _parked_others(n=2)
    r2, f2 = _fallback_others(n=1)
    tr = fx.segment_tracks(pd.DataFrame(rows + r2))
    j = fx.join_flights(pd.DataFrame(fl + f2), fx.takeoff_events(tr))
    c = fx.stand_centroids(j, tr).set_index("MVT_ID_mvt")
    assert c.loc[20, "parked_share"] == pytest.approx(1.0)          # the fallback flight's others: 2 parked
    assert c.loc[10, "parked_share"] == pytest.approx(0.5)          # a parked flight's others: 1 parked + 1 fallback
    assert c.loc[10, "n_parked_others"] == 1 and np.isnan(c.loc[10, "pcx_m"])   # < 2 parked others: no parked ref
    assert c.loc[20, "n_parked_others"] == 2 and c.loc[20, "pcx_m"] == pytest.approx(0.0, abs=2)
    one = fx.stand_centroids(j.iloc[:2], tr).set_index("MVT_ID_mvt")
    assert one.parked_share.isna().all()                            # one other each: no ref, share unknown


def test_stand_visible_is_unknown_for_unjoined_flights():
    """BC-3: an unjoined flight has no observation, so stand_visible is NaN, never 0."""
    j = pd.DataFrame([dict(MVT_ID_mvt=9, ADEP_mvt="EHAM", MVT_TIME_UTC_mvt=pd.Timestamp(T0, unit="s", tz="UTC"),
                           AOBT_3_flt=pd.Timestamp(T0 - 600, unit="s", tz="UTC"), STAND_mvt="S1",
                           track_id=np.nan, t_takeoff=np.nan, dt_takeoff=np.nan, join_type=None)])
    f = fx.flight_features(j, fx.segment_tracks(pd.DataFrame(_samples("a1", "X", [(T0, 0, 0, 1, 0)]))))
    assert np.isnan(f.iloc[0].stand_visible)


def test_without_aobt3_no_run_can_be_placed_before_off_block_so_there_is_no_gate_dwell():
    """AOBT_3 missing (the unmatched lane): the time gate cannot be evaluated, so no run is a gate dwell (unknown is not
    'passes'); stand_visible is 0 for the joined flight. Kills: a NaN comparison treated as passing."""
    mvt = T0 + 5000
    pts = _run(mvt - 2000, 420, 0.0, 0.0) + [(mvt - 1500, 300, 0, 1, 12.0), (mvt - 1400, 1200, 0, 1, 15.0)]
    r, f = _target(pts, mvt, mvt - 1500)
    f["AOBT_3_flt"] = pd.NaT
    out = fx.build(pd.DataFrame([f]), pd.DataFrame(r)).iloc[0]
    assert out.joined == 1 and np.isnan(out.dwell_s) and out.stand_visible == 0


def test_flight_features_refuses_a_centroid_frame_without_the_provenance_columns():
    """Fail loud at the boundary: a caller-built centroid frame without the parked reference columns would silently
    disable the location gate. Kills: a .get() default for the missing columns."""
    j = pd.DataFrame([dict(MVT_ID_mvt=1, ADEP_mvt="EHAM", MVT_TIME_UTC_mvt=pd.Timestamp(T0, unit="s", tz="UTC"),
                           AOBT_3_flt=pd.Timestamp(T0 - 600, unit="s", tz="UTC"), STAND_mvt="S1",
                           track_id=np.nan, t_takeoff=np.nan, dt_takeoff=np.nan, join_type=None)])
    cen = pd.DataFrame({"MVT_ID_mvt": [1], "cx_m": [0.0], "cy_m": [0.0], "n_others": [3]})
    with pytest.raises(ValueError, match="centroids frame lacks.*pcx_m"):
        fx.flight_features(j, fx.segment_tracks(pd.DataFrame(_samples("a1", "X", [(T0, 0, 0, 1, 0)]))), cen)


def test_the_features_stamp_marks_the_gate_dwell_fix_and_the_new_columns_are_features():
    """Tables built before and after the DWG fix differ on every dwell-anchored column; the stamp must tell them apart
    (r12_build derives its store stamp from it). Kills: the fix shipped under the old stamp; the visibility or
    provenance column computed but never handed to the model."""
    assert "DWG" in fx.FEATURES_VERSION and "TRC" in fx.FEATURES_VERSION and "taxi-start" in fx.FEATURES_VERSION
    assert "stand_visible" in fx.FEATURES and "stand_ref_parked_share" in fx.FEATURES
    assert fx.DWELL_START_MAX_S == fx.CENTROID_WINDOW[1] == 60.0
