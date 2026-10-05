"""Tests for scripts/analysis/uad_build.py — the label-blind stand-exit table for departures without a flight-plan
record (plans/PREREG_uad_unmatched_adsb_offblock_2026_10_04.md).

Written before the module (red first). Every test docstring ends with "Fails when ..." naming the mutant or defect
it must catch. Synthetic ADS-B frames and row frames are built locally in this file (not imported from another
test module's private helpers).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("uad_build", "scripts/analysis/uad_build.py")
AF = B.AF
AE = B.AE
ISO = _load("prc_test_isolation", "tests/isolation.py")

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _secs(ts) -> float:
    return (pd.Timestamp(ts) - EPOCH).total_seconds()


def _day_start(day: str) -> float:
    return _secs(pd.Timestamp(day, tz="UTC"))


# ---------------------------------------------------------------- synthetic ADS-B helpers (own copies: tests must
# not import another test module's private fixtures)

def _ll(dx_m, dy_m, airport="EHAM"):
    """(lat, lon) of a point dx_m east, dy_m north of `airport`'s reference point."""
    a0, o0 = AE.APTS[airport]
    return a0 + dy_m / AF.M_PER_DEG_LAT, o0 + dx_m / (AF.M_PER_DEG_LON_EQ * np.cos(np.radians(a0)))


def _samples(icao, callsign, pts, airport="EHAM", actype="A320"):
    """pts: list of (t, dx_m, dy_m, on_ground, gs[, type_override])."""
    rows = []
    for p in pts:
        t, dx, dy, og, gs = p[:5]
        ty = p[5] if len(p) > 5 else actype
        lat, lon = _ll(dx, dy, airport)
        rows.append(dict(icao=icao, reg="REGX", type=ty, callsign=callsign, airport=airport, t=float(t), lat=lat,
                         lon=lon, on_ground=int(og), alt_missing=0, alt=(np.nan if og else 1200.0), gs=float(gs),
                         track=0.0, src="adsb_icao"))
    return rows


def _departure_track(icao, callsign, t_park0, n_park, stand_xy, t_leave, t_takeoff, airport="EHAM", actype="A320"):
    """Parked at stand_xy every 30 s from t_park0 for n_park samples, one sample 30 s later at 50 m east (leaving),
    a mid-taxi sample, a pre-roll sample, then airborne at t_takeoff."""
    pts = [(t_park0 + k * 30, stand_xy[0], stand_xy[1], 1, 0.0) for k in range(n_park)]
    pts += [(t_leave, stand_xy[0] + 50.0, stand_xy[1], 1, 10.0),
           (t_leave + 270, stand_xy[0] + 500.0, stand_xy[1], 1, 15.0),
           (t_takeoff - 5, stand_xy[0] + 3000.0, stand_xy[1], 1, 140.0),
           (t_takeoff, stand_xy[0] + 3100.0, stand_xy[1], 0, 160.0),
           (t_takeoff + 20, stand_xy[0] + 3500.0, stand_xy[1], 0, 180.0)]
    return _samples(icao, callsign, pts, airport, actype)


def _row(mvt_id, callsign, mvt_t, aobt_t, stand="S1", airport="EHAM", phase="DEP", actype="A320", flight_mvt=None):
    return dict(MVT_ID_mvt=float(mvt_id), ADEP_mvt=airport, CALLSIGN_flt=callsign, FLIGHT_mvt=flight_mvt or callsign,
                MVT_TIME_UTC_mvt=pd.Timestamp(mvt_t, unit="s", tz="UTC"),
                AOBT_3_flt=(pd.Timestamp(aobt_t, unit="s", tz="UTC") if aobt_t is not None else pd.NaT),
                STAND_mvt=stand, PHASE_mvt=phase, AIRCRAFT_TYPE_mvt=actype)


def _rows_df(rows, extra=None):
    df = pd.DataFrame(rows)
    if extra:
        for k, v in extra.items():
            df[k] = v
    return df


def _write_parquet(path: pathlib.Path, df: pd.DataFrame) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _adsb_df(rows) -> pd.DataFrame:
    cols = ["icao", "reg", "type", "callsign", "airport", "t", "lat", "lon", "on_ground", "alt_missing", "alt", "gs",
           "track", "src"]
    return pd.DataFrame(rows, columns=cols)


def _write_adsb_day(d: pathlib.Path, day: str, rows) -> pathlib.Path:
    return _write_parquet(d / f"adsb_{day}.parquet", _adsb_df(rows))


def _manifest(d: pathlib.Path, contains=None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    body = {"source": "test"}
    if contains is not None:
        body["contains"] = contains
    (d / "MANIFEST.json").write_text(json.dumps(body))


T0 = 1_741_000_000.0  # 2025-03-03T15:06:40Z-ish; a fixed, arbitrary epoch base for pure numeric tests


@pytest.fixture
def few_joins_make_a_window(monkeypatch):
    """AMEND-A3 gives an airport an offset window only from 200 callsign joins of matched departures; the synthetic
    worlds hold a handful. The tests that ask for this fixture lower the bound to 3 (forked --jobs children inherit
    it)."""
    monkeypatch.setattr(B, "WINDOW_MIN_ROWS", 3)


# ================================================================== params / params_hash

def test_params_contains_every_registered_constant():
    """Fails when a registered constant is left out of params() (silent drift between the code and what is
    recorded in every output report.json)."""
    p = B.params()
    for name in ("STAND_RADIUS_M", "EXIT_NEXT_MAX_S", "EXIT_LOOKBACK_N", "STATIONARY_KT", "REPARK_MIN_S",
                "RUN_GAP_MAX_S", "REF_WINDOW_S", "REF_MIN_FLIGHTS", "REF_MAX_MAD_M", "A1_MAX_S", "A2_MAX_S",
                "ALIAS_MIN_ROWS", "ALIAS_MIN_AIRFRAMES", "ALIAS_MIN_SHARE", "CAND_BRACKET_S", "WINDOW_Q",
                "WINDOW_MARGIN_S", "WINDOW_FLOOR_S", "WINDOW_MIN_ROWS", "CHECK_SAMPLE_MOD"):
        want = getattr(B, name)
        got = tuple(p[name]) if isinstance(want, tuple) else p[name]
        assert got == want, name
    assert len(p) == 20


def test_params_hash_is_twelve_hex_and_stable():
    """Fails when params_hash() is not deterministic, or not exactly 12 hex characters."""
    h1, h2 = B.params_hash(), B.params_hash()
    assert h1 == h2
    assert len(h1) == 12 and all(c in "0123456789abcdef" for c in h1)


def test_params_hash_changes_with_a_param(monkeypatch):
    """Fails when params_hash() ignores a changed constant (hard-codes the hash instead of hashing params())."""
    before = B.params_hash()
    monkeypatch.setattr(B, "STAND_RADIUS_M", 41.0)
    assert B.params_hash() != before


# ================================================================== read_rows

def test_read_rows_never_returns_a_label_column(tmp_path):
    """A file with both label columns present (ordinary dtypes) must not leak them into the result. Fails when
    read_rows reads the full file and only filters columns afterwards (still correct here, but see the projection
    test below for the read-path itself)."""
    rows = [_row(1, "AAA1", T0, T0 - 600)]
    df = _rows_df(rows, extra={"BLOCK_TIME_UTC_mvt": pd.Timestamp(T0 + 600, unit="s", tz="UTC"),
                               "TAXITIME_SEC_mvt": 600})
    p = _write_parquet(tmp_path / "f1.parquet", df)
    out = B.read_rows([p])
    assert "BLOCK_TIME_UTC_mvt" not in out.columns and "TAXITIME_SEC_mvt" not in out.columns


def test_read_rows_projects_columns_at_the_parquet_read_itself(tmp_path, monkeypatch):
    """The read must request exactly ROW_COLS from pyarrow (pq.read_table(path, columns=...)), not read everything
    and drop columns after. Fails when read_rows calls pq.read_table without a `columns=` kwarg (or with the wrong
    one)."""
    rows = [_row(1, "AAA1", T0, T0 - 600)]
    p = _write_parquet(tmp_path / "f1.parquet", _rows_df(rows))
    seen = {}
    real_read_table = B.pq.read_table

    def fake_read_table(path, columns=None):
        seen["columns"] = columns
        return real_read_table(path, columns=columns)

    monkeypatch.setattr(B.pq, "read_table", fake_read_table)
    B.read_rows([p])
    assert seen["columns"] == B.ROW_COLS
    assert "BLOCK_TIME_UTC_mvt" not in seen["columns"] and "TAXITIME_SEC_mvt" not in seen["columns"]


def test_read_rows_refuses_duplicate_ids_across_two_files(tmp_path):
    """Fails when the duplicate-id check only looks within one file (never compares across files)."""
    p1 = _write_parquet(tmp_path / "a.parquet", _rows_df([_row(1, "AAA1", T0, T0 - 600)]))
    p2 = _write_parquet(tmp_path / "b.parquet", _rows_df([_row(1, "BBB2", T0 + 100, T0 - 500)]))
    with pytest.raises(ValueError, match="2"):
        B.read_rows([p1, p2])


def test_read_rows_refuses_a_null_id(tmp_path):
    """Fails when a null MVT_ID_mvt is silently kept (e.g. dropna() instead of raising)."""
    rows = [_row(1, "AAA1", T0, T0 - 600), _row(2, "BBB2", T0 + 50, T0 - 550)]
    rows[1]["MVT_ID_mvt"] = np.nan
    p = _write_parquet(tmp_path / "f.parquet", _rows_df(rows))
    with pytest.raises(ValueError, match="1"):
        B.read_rows([p])


def test_read_rows_refuses_a_file_missing_a_row_col(tmp_path):
    """Fails when a missing column is not checked explicitly and pyarrow's own error message is swallowed instead of
    surfaced as a clear ValueError."""
    df = _rows_df([_row(1, "AAA1", T0, T0 - 600)]).drop(columns=["STAND_mvt"])
    p = _write_parquet(tmp_path / "f.parquet", df)
    with pytest.raises(ValueError, match="STAND_mvt"):
        B.read_rows([p])


def test_read_rows_keeps_only_dep_rows(tmp_path):
    """Fails when an ARR row is kept (the DEP filter is missing or inverted)."""
    rows = [_row(1, "AAA1", T0, T0 - 600, phase="DEP"), _row(2, "BBB2", T0 + 50, T0 - 550, phase="ARR")]
    p = _write_parquet(tmp_path / "f.parquet", _rows_df(rows))
    out = B.read_rows([p])
    assert out.MVT_ID_mvt.tolist() == [1.0]


def test_read_rows_unmatched_and_stand_are_right(tmp_path):
    """Fails when `unmatched` is not exactly AOBT_3_flt.isna(), or `stand` is not the normalised STAND_mvt."""
    rows = [_row(1, "AAA1", T0, T0 - 600, stand=" a1 "), _row(2, "BBB2", T0 + 50, None, stand=None)]
    p = _write_parquet(tmp_path / "f.parquet", _rows_df(rows))
    out = B.read_rows([p]).set_index("MVT_ID_mvt")
    assert out.loc[1.0, "unmatched"] == False and out.loc[1.0, "stand"] == "A1"
    assert out.loc[2.0, "unmatched"] == True and out.loc[2.0, "stand"] is None


def test_read_rows_mvt_s_equals_epoch_seconds_of_mvt_time(tmp_path):
    """Fails when mvt_s is computed with the wrong unit (e.g. astype(int64) nanoseconds, BC-class bug seen elsewhere
    in this codebase's ADS-B code)."""
    rows = [_row(1, "AAA1", T0, T0 - 600)]
    p = _write_parquet(tmp_path / "f.parquet", _rows_df(rows))
    out = B.read_rows([p])
    assert out.iloc[0].mvt_s == pytest.approx(T0)


def test_read_rows_mvt_id_is_float64(tmp_path):
    """Fails when MVT_ID_mvt is cast to int or object, breaking downstream float64 joins."""
    p = _write_parquet(tmp_path / "f.parquet", _rows_df([_row(1, "AAA1", T0, T0 - 600)]))
    out = B.read_rows([p])
    assert out.MVT_ID_mvt.dtype == np.float64


# ================================================================== norm_stand

def test_norm_stand_strips_and_upper_cases():
    """Fails when the normalisation is not strip().upper()."""
    assert B.norm_stand(" a12 ") == "A12"


def test_norm_stand_none_for_null_and_empty():
    """Fails when an empty or null stand returns "" instead of None."""
    assert B.norm_stand(None) is None
    assert B.norm_stand(np.nan) is None
    assert B.norm_stand("") is None
    assert B.norm_stand("   ") is None


# ================================================================== xy

def test_xy_matches_the_construction_within_half_a_metre():
    """Fails when the east/west or north/south sign is flipped, or the latitude-cosine scaling is dropped."""
    lat, lon = _ll(100.0, 50.0, "EHAM")
    x, y = B.xy(np.array([lat]), np.array([lon]), "EHAM")
    assert x[0] == pytest.approx(100.0, abs=0.5) and y[0] == pytest.approx(50.0, abs=0.5)


def test_xy_refuses_an_unknown_airport():
    """Fails when an unknown airport silently returns (0, 0) or KeyErrors instead of a clear ValueError."""
    with pytest.raises(ValueError):
        B.xy(np.array([52.0]), np.array([4.0]), "ZZZZ")


# ================================================================== read_day

def test_read_day_takes_the_first_directory_that_holds_the_day(tmp_path):
    """Fails when a later directory in the list wins, or the directories are not tried in order."""
    day = "2025-03-10"
    a, b = tmp_path / "a", tmp_path / "b"
    start = _day_start(day)
    _write_adsb_day(a, day, _samples("icaoA", "AAA", [(start + 100, 0, 0, 1, 0.0)]))
    _write_adsb_day(b, day, _samples("icaoB", "BBB", [(start + 100, 0, 0, 1, 0.0)]))
    out = B.read_day([a, b], day, "EHAM")
    assert set(out.icao) == {"icaoA"}


def test_read_day_includes_previous_day_tail_and_excludes_earlier_part(tmp_path):
    """Fails when the previous day's whole file is pulled in (not just the last GROUND_LOOKBACK_S), or when the tail
    window is dropped entirely."""
    day = "2025-03-10"
    start = _day_start(day)
    d = tmp_path / "a"
    inside = start - AF.GROUND_LOOKBACK_S + 10
    outside = start - AF.GROUND_LOOKBACK_S - 10
    _write_adsb_day(d, "2025-03-09", _samples("icaoIN", "IN1", [(inside, 0, 0, 1, 0.0)])
                    + _samples("icaoOUT", "OUT1", [(outside, 0, 0, 1, 0.0)]))
    _write_adsb_day(d, day, _samples("icaoCUR", "CUR1", [(start + 5, 0, 0, 1, 0.0)]))
    out = B.read_day([d], day, "EHAM")
    assert set(out.icao) == {"icaoIN", "icaoCUR"}


def test_read_day_includes_next_day_head_and_excludes_later_part(tmp_path):
    """Fails when the next day's head window (NEXT_DAY_HEAD_S) is not pulled in, or pulls in too much."""
    day = "2025-03-10"
    start = _day_start(day)
    d = tmp_path / "a"
    _write_adsb_day(d, day, _samples("icaoCUR", "CUR1", [(start + 5, 0, 0, 1, 0.0)]))
    inside = start + 86_400 + AF.NEXT_DAY_HEAD_S - 10
    outside = start + 86_400 + AF.NEXT_DAY_HEAD_S + 10
    _write_adsb_day(d, "2025-03-11", _samples("icaoIN", "IN1", [(inside, 0, 0, 1, 0.0)])
                    + _samples("icaoOUT", "OUT1", [(outside, 0, 0, 1, 0.0)]))
    out = B.read_day([d], day, "EHAM")
    assert set(out.icao) == {"icaoCUR", "icaoIN"}


def test_read_day_returns_none_when_the_day_file_is_absent_everywhere(tmp_path):
    """Fails when a missing file is treated as an empty frame instead of None."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    assert B.read_day([a, b], "2025-03-10", "EHAM") is None


def test_read_day_relative_directories_work_from_another_cwd(tmp_path, monkeypatch):
    """BC-40 sibling: a relative source directory must reach the reader as an absolute, readable path regardless of
    the caller's cwd. Fails when the directory is used as given (relative) and resolved against the wrong cwd."""
    day = "2025-03-10"
    start = _day_start(day)
    _write_adsb_day(tmp_path / "rel", day, _samples("icaoA", "AAA", [(start + 5, 0, 0, 1, 0.0)]))
    monkeypatch.chdir(tmp_path)
    out = B.read_day(["rel"], day, "EHAM")
    assert out is not None and len(out) == 1


def test_read_day_an_unreadable_file_raises(tmp_path):
    """Fails when a corrupt file that exists is treated as 'no data' (returns None) instead of raising."""
    d = tmp_path / "a"
    d.mkdir()
    (d / "adsb_2025-03-10.parquet").write_bytes(b"not a parquet file")
    with pytest.raises(Exception):
        B.read_day([d], "2025-03-10", "EHAM")


# ================================================================== join_day

def test_join_day_callsign_joined_flight_is_usable():
    """Fails when a plain callsign join is marked not usable."""
    mvt = T0 + 900
    adsb = pd.DataFrame(_departure_track("ic1", "KLM1", T0, 3, (0.0, 0.0), T0 + 90, mvt))
    fl = _rows_df([_row(1, "KLM1", mvt, T0 - 100)])
    j, tr = B.join_day(fl, adsb)
    assert j.iloc[0].join_type == "callsign" and j.iloc[0].usable == True


def test_join_day_time_joined_with_equal_types_is_usable():
    """Fails when a time join with matching ADS-B/row aircraft types is marked not usable."""
    mvt = T0 + 900
    adsb = pd.DataFrame(_departure_track("ic1", "NOTCS", T0, 3, (0.0, 0.0), T0 + 90, mvt, actype="A320"))
    fl = _rows_df([_row(1, "OTHER9", mvt, T0 - 100, actype="A320")])
    j, tr = B.join_day(fl, adsb)
    assert j.iloc[0].join_type == "time" and j.iloc[0].type_agree == 1.0 and j.iloc[0].usable == True


def test_join_day_time_joined_with_different_types_is_not_usable():
    """Fails when the ADS-B/row aircraft-type disagreement check is missing (usable follows join_type alone)."""
    mvt = T0 + 900
    adsb = pd.DataFrame(_departure_track("ic1", "NOTCS", T0, 3, (0.0, 0.0), T0 + 90, mvt, actype="B738"))
    fl = _rows_df([_row(1, "OTHER9", mvt, T0 - 100, actype="A320")])
    j, tr = B.join_day(fl, adsb)
    assert j.iloc[0].join_type == "time" and j.iloc[0].type_agree == 0.0 and j.iloc[0].usable == False


def test_join_day_missing_type_gives_nan_agree_and_time_join_not_usable():
    """Fails when a missing type on either side is treated as agreement (or disagreement) instead of NaN."""
    mvt = T0 + 900
    adsb = pd.DataFrame(_departure_track("ic1", "NOTCS", T0, 3, (0.0, 0.0), T0 + 90, mvt, actype=None))
    fl = _rows_df([_row(1, "OTHER9", mvt, T0 - 100, actype="A320")])
    j, tr = B.join_day(fl, adsb)
    assert j.iloc[0].join_type == "time" and pd.isna(j.iloc[0].type_agree) and j.iloc[0].usable == False


def test_join_day_unjoined_flight_is_not_usable():
    """Fails when an unjoined flight (no nearby take-off at all) is marked usable."""
    adsb = pd.DataFrame(_samples("other", "ZZZ9", [(T0, 9000, 9000, 1, 0.0)]))
    fl = _rows_df([_row(1, "NOPE1", T0 + 900, T0 - 100)])
    j, tr = B.join_day(fl, adsb)
    assert j.iloc[0].join_type is None and j.iloc[0].usable == False


def test_join_day_names_the_joined_aircraft_and_leaves_an_unjoined_row_without_one():
    """`adsb_icao` is the aircraft address of the joined take-off (with t_takeoff it identifies the take-off across
    two days' views). Fails when the address is taken from another track, or an unjoined row is given one."""
    mvt = T0 + 900
    adsb = pd.DataFrame(_departure_track("ic1", "KLM1", T0, 3, (0.0, 0.0), T0 + 90, mvt)
                        + _departure_track("ic2", "KLM2", T0 + 5_000, 3, (0.0, 0.0), T0 + 5_090, mvt + 5_000))
    fl = _rows_df([_row(1, "KLM2", mvt + 5_000, T0 - 100), _row(2, "KLM1", mvt, T0 - 100),
                   _row(3, "NOPE1", mvt + 20_000, T0 - 100)])
    j, tr = B.join_day(fl, adsb)
    by_id = j.set_index("MVT_ID_mvt")
    assert by_id.loc[1.0, "adsb_icao"] == "ic2" and by_id.loc[2.0, "adsb_icao"] == "ic1"
    assert by_id.loc[1.0, "t_takeoff"] == mvt + 5_000 and by_id.loc[2.0, "t_takeoff"] == mvt
    assert pd.isna(by_id.loc[3.0, "adsb_icao"]) and pd.isna(by_id.loc[3.0, "t_takeoff"])


# ================================================================== parked_position

def test_parked_position_picks_the_last_stationary_sample_in_window():
    """Fails when the FIRST qualifying sample is picked instead of the LAST (and when the window's own upper bound
    does not exclude the trailing out-of-window sample)."""
    aobt = T0
    t = np.array([aobt - 890, aobt - 400, aobt - 100, aobt - 5])   # the last point is OUTSIDE the window (> aobt-20)
    x = np.array([1.0, 2.0, 3.0, 999.0])
    y = np.array([1.0, 2.0, 3.0, 999.0])
    gs = np.array([0.0, 0.0, 0.0, 0.0])
    pos = B.parked_position(t, x, y, gs, aobt)
    assert pos == (3.0, 3.0)


def test_parked_position_ignores_a_sample_10s_before_aobt():
    """Fails when the window's upper bound (aobt - 20) is not enforced (off-by-constant in REF_WINDOW_S[1])."""
    aobt = T0
    t = np.array([aobt - 400, aobt - 10])
    x = np.array([5.0, 999.0])
    y = np.array([5.0, 999.0])
    gs = np.array([0.0, 0.0])
    assert B.parked_position(t, x, y, gs, aobt) == (5.0, 5.0)


def test_parked_position_ignores_a_sample_1000s_before_aobt():
    """Fails when the window's lower bound (aobt - 900) is not enforced (off-by-constant in REF_WINDOW_S[0])."""
    aobt = T0
    t = np.array([aobt - 1000, aobt - 400])
    x = np.array([999.0, 6.0])
    y = np.array([999.0, 6.0])
    gs = np.array([0.0, 0.0])
    assert B.parked_position(t, x, y, gs, aobt) == (6.0, 6.0)


def test_parked_position_none_when_the_only_sample_is_just_before_the_window():
    """The window's lower bound must actually EXCLUDE a too-early sample, not just lose a tie-break against a later
    one: with only one (too-early) candidate, a loosened lower bound would wrongly return it instead of None.
    Fails when REF_WINDOW_S[0] is widened (e.g. treated as 1400 s instead of 900 s)."""
    aobt = T0
    t = np.array([aobt - 901])
    assert B.parked_position(t, np.array([1.0]), np.array([1.0]), np.array([0.0]), aobt) is None


def test_parked_position_none_when_aobt_is_nan():
    """Fails when a NaN aobt_s is not checked and arithmetic on it silently produces a bogus window."""
    t = np.array([T0 - 400])
    assert B.parked_position(t, np.array([1.0]), np.array([1.0]), np.array([0.0]), np.nan) is None


def test_parked_position_none_when_nothing_is_stationary_in_window():
    """Fails when the gs < STATIONARY_KT guard is dropped (any sample in the time window is accepted)."""
    aobt = T0
    t = np.array([aobt - 400])
    assert B.parked_position(t, np.array([1.0]), np.array([1.0]), np.array([5.0]), aobt) is None


# ================================================================== stand_references

def test_stand_references_needs_at_least_three_flights():
    """Fails when REF_MIN_FLIGHTS is not enforced (2 contributing flights still form a reference)."""
    two = pd.DataFrame({"ADEP_mvt": ["EHAM"] * 2, "stand": ["S1"] * 2, "x": [0.0, 1.0], "y": [0.0, 1.0]})
    three = pd.DataFrame({"ADEP_mvt": ["EHAM"] * 3, "stand": ["S2"] * 3, "x": [0.0, 1.0, -1.0], "y": [0.0, 1.0, -1.0]})
    out = B.stand_references(pd.concat([two, three], ignore_index=True))
    assert list(out.stand) == ["S2"]


def test_stand_references_drops_scattered_positions():
    """Fails when REF_MAX_MAD_M is not enforced (a 30 m MAD stand still forms a reference)."""
    p = pd.DataFrame({"ADEP_mvt": ["EHAM"] * 3, "stand": ["S1"] * 3, "x": [0.0, 30.0, -30.0], "y": [0.0, 0.0, 0.0]})
    out = B.stand_references(p)
    assert len(out) == 0


def test_stand_references_null_stand_never_forms_a_reference():
    """Fails when a null stand key reaches groupby/merge and silently forms a reference (the pandas NaN-key bug
    class named in the brief)."""
    p = pd.DataFrame({"ADEP_mvt": ["EHAM"] * 3, "stand": [None, None, None], "x": [0.0, 1.0, -1.0],
                      "y": [0.0, 1.0, -1.0]})
    out = B.stand_references(p)
    assert len(out) == 0


def test_stand_references_medians_are_right_and_order_is_deterministic():
    """Fails when the reference position is a mean (sensitive to an outlier) instead of a median, or the output
    order is not sorted by (airport, stand)."""
    p = pd.DataFrame({"ADEP_mvt": ["EHAM", "EHAM", "EHAM", "EGLL", "EGLL", "EGLL"],
                      "stand": ["S2", "S2", "S2", "S1", "S1", "S1"],
                      "x": [0.0, 2.0, 100.0, 10.0, 10.0, 10.0], "y": [0.0, 2.0, 100.0, 5.0, 5.0, 5.0]})
    out = B.stand_references(p)
    assert list(zip(out.ADEP_mvt, out.stand)) == [("EGLL", "S1"), ("EHAM", "S2")]
    row = out[out.ADEP_mvt == "EHAM"].iloc[0]
    assert row.x == 2.0 and row.y == 2.0 and row.n == 3


# ================================================================== stationary_runs

def test_stationary_runs_130s_run_found_100s_run_not():
    """Fails when REPARK_MIN_S is not enforced as a strict '>=' floor on run duration."""
    t_ok = np.arange(0, 131, 10, dtype=float)       # 0..130, span 130 s
    gs_ok = np.zeros_like(t_ok)
    assert B.stationary_runs(t_ok, gs_ok) == [(0, len(t_ok) - 1)]
    t_short = np.arange(0, 101, 10, dtype=float)    # span 100 s < REPARK_MIN_S
    gs_short = np.zeros_like(t_short)
    assert B.stationary_runs(t_short, gs_short) == []


def test_stationary_runs_split_by_a_200s_gap():
    """Fails when RUN_GAP_MAX_S is not enforced (a 200 s gap between stationary samples is kept as one run)."""
    t = np.array([0.0, 60.0, 130.0, 330.0, 400.0, 470.0])    # gap 130->330 is 200 s > RUN_GAP_MAX_S
    gs = np.zeros_like(t)
    assert B.stationary_runs(t, gs) == [(0, 2), (3, 5)]


def test_stationary_runs_gap_boundary_is_inclusive_at_180s():
    """Fails when the gap comparison uses '>=' instead of '>' (an exactly-180 s gap must NOT split the run)."""
    t = np.array([0.0, 180.0, 300.0])      # gap exactly RUN_GAP_MAX_S, then a short gap; span 300 >= REPARK_MIN_S
    gs = np.zeros_like(t)
    assert B.stationary_runs(t, gs) == [(0, 2)]


def test_stationary_runs_ignores_non_stationary_samples():
    """Fails when a fast sample is counted as part of a run instead of ending it."""
    t = np.array([0.0, 50.0, 100.0])
    gs = np.array([0.0, 50.0, 0.0])          # the middle sample is moving, not stationary
    assert B.stationary_runs(t, gs) == []    # neither stationary sample alone reaches REPARK_MIN_S


def test_a_sample_without_a_ground_speed_is_not_stationary():
    """The day store holds samples with a null ground speed (0.6 % of all samples, 2026-10-05). Such a sample is not
    known to stand still: it ends a stationary run, it is no parked position and it does not make parked_seen true.
    Fails when a missing ground speed is read as 0 kt (stationary)."""
    t = np.arange(0.0, 300.0, 30.0)                       # ten samples, 270 s
    gs = np.zeros(10)
    gs[4] = np.nan
    assert B.stationary_runs(t, gs) == [(5, 9)]            # 0..3 spans 90 s (dropped), 5..9 spans 120 s (kept)
    assert B.parked_position(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([np.nan]), 100.0) is None
    d = B.stand_exit(np.array([0.0, 30.0]), np.array([0.0, 500.0]), np.zeros(2), np.array([np.nan, 20.0]), 200.0,
                     REF, np.array([REF]))
    assert d["exit_t"] == 0.0 and not d["parked_seen"]


def test_stationary_runs_a_moving_sample_between_two_stops_splits_the_run():
    """A run is made of CONSECUTIVE samples that are all stationary: an aircraft that stops, creeps forward and stops
    again (a queue) has two short stops, not one long one. Fails when runs are built over the stationary samples
    only, so that a moving sample in between does not end the run (the two 60 s stops then read as one 190 s run)."""
    t = np.array([0.0, 60.0, 70.0, 130.0, 190.0])
    gs = np.array([0.0, 0.0, 5.0, 0.0, 0.0])
    assert B.stationary_runs(t, gs) == []


def test_stationary_runs_two_real_stops_around_a_moving_sample_are_two_runs():
    """Fails when a moving sample between two stops of at least REPARK_MIN_S each is bridged into one run, or when
    either stop is dropped."""
    t = np.array([0.0, 60.0, 130.0, 140.0, 200.0, 330.0])
    gs = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    assert B.stationary_runs(t, gs) == [(0, 2), (4, 5)]


# ================================================================== stand_exit

REF = (0.0, 0.0)


def test_stand_exit_normal_pushback():
    """Fails when the LAST near-stand sample is not picked (e.g. the first is picked instead)."""
    t = np.array([0.0, 30.0, 60.0, 90.0])
    x = np.array([0.0, 5.0, 100.0, 500.0])
    y = np.array([0.0, 0.0, 0.0, 0.0])
    gs = np.array([0.0, 0.0, 10.0, 50.0])
    d = B.stand_exit(t, x, y, gs, 200.0, REF, np.array([REF]))
    assert d["exit_t"] == 30.0 and d["continuous"] and d["parked_seen"] and not d["repark"]
    assert B.exit_ok(d)


def test_stand_exit_a_sample_at_45m_is_outside_the_radius():
    """Fails when STAND_RADIUS_M is applied as '< 45' or similar (the 45 m sample must never be picked)."""
    t = np.array([0.0, 30.0, 60.0])
    x = np.array([0.0, 45.0, 100.0])
    y = np.array([0.0, 0.0, 0.0])
    gs = np.array([0.0, 0.0, 10.0])
    d = B.stand_exit(t, x, y, gs, 200.0, REF, np.array([REF]))
    assert d["exit_t"] == 0.0


def test_stand_exit_radius_boundary_is_inclusive_at_40m():
    """Fails when STAND_RADIUS_M uses a strict '<' (a sample at exactly 40 m would then be excluded)."""
    t = np.array([0.0, 30.0])
    x = np.array([40.0, 200.0])
    y = np.array([0.0, 0.0])
    gs = np.array([0.0, 20.0])
    d = B.stand_exit(t, x, y, gs, 200.0, REF, np.array([REF]))
    assert d["exit_t"] == 0.0


def test_stand_exit_never_within_radius_gives_no_exit():
    """Fails when a track that never nears the stand is still assigned an exit (k defaults to 0 instead of 'none')."""
    t = np.array([0.0, 30.0, 60.0])
    x = np.array([100.0, 110.0, 120.0])
    y = np.array([0.0, 0.0, 0.0])
    gs = np.array([0.0, 0.0, 0.0])
    d = B.stand_exit(t, x, y, gs, 200.0, REF, np.array([REF]))
    assert np.isnan(d["exit_t"]) and not d["continuous"] and not d["parked_seen"] and not d["repark"]
    assert d["first_ground_dist_m"] == pytest.approx(100.0)
    assert not B.exit_ok(d)


def test_stand_exit_continuity_boundary_and_a_3h_silence():
    """Fails when EXIT_NEXT_MAX_S is not enforced as an upper bound (a 3 h gap after the stand sample is still
    called continuous)."""
    t = np.array([0.0, 30.0, 3630.0])        # gap after the stand sample is 3600 s
    x = np.array([0.0, 5.0, 2000.0])
    y = np.array([0.0, 0.0, 0.0])
    gs = np.array([0.0, 0.0, 100.0])
    d = B.stand_exit(t, x, y, gs, 4000.0, REF, np.array([REF]))
    assert d["exit_t"] == 30.0 and not d["continuous"]
    assert not B.exit_ok(d)
    t2 = np.array([0.0, 30.0, 30.0 + B.EXIT_NEXT_MAX_S])
    d2 = B.stand_exit(t2, x, y, gs, 4000.0, REF, np.array([REF]))
    assert d2["continuous"]                  # boundary is inclusive


def test_stand_exit_an_exit_at_the_last_ground_sample_is_not_continuous():
    """Guard (i) asks for a NEXT ground sample within EXIT_NEXT_MAX_S. A track whose last ground sample before
    take-off is still at the stand (ground coverage ends there) has none, so when it left is unknown. Fails when a
    missing next sample is read as continuous, or when looking past the last sample raises."""
    t = np.array([0.0, 30.0, 60.0])
    x = np.array([0.0, 2.0, 5.0])
    y = np.zeros(3)
    gs = np.zeros(3)
    d = B.stand_exit(t, x, y, gs, 500.0, REF, np.array([REF]))
    assert d["exit_t"] == 60.0 and d["parked_seen"] and not d["repark"]
    assert not d["continuous"]
    assert not B.exit_ok(d)


def test_stand_exit_taxi_past_without_stopping_fails_parked_seen():
    """Fails when parked_seen does not require an actually-stationary sample in the lookback window (any ground
    sample counts)."""
    t = np.arange(0, 21, dtype=float)
    x = np.linspace(0, 20, 21)
    y = np.zeros(21)
    gs = np.full(21, 5.0)          # moving the whole time, never below STATIONARY_KT
    d = B.stand_exit(t, x, y, gs, 100.0, REF, np.array([REF]))
    assert not d["parked_seen"]
    assert not B.exit_ok(d)


def test_stand_exit_lookback_window_boundary():
    """Fails when EXIT_LOOKBACK_N's window is off by one (a stationary sample at k-19 must count; one at k-20 must
    not)."""
    n = 25
    t = np.arange(n, dtype=float) * 10
    x = np.zeros(n)
    y = np.zeros(n)
    gs = np.full(n, 5.0)
    k = 24
    gs[k - 19] = 0.0                 # exactly inside the 20-sample lookback window ending at k
    x[k] = 0.0
    d = B.stand_exit(t, x, y, gs, t[-1] + 100, REF, np.array([REF]))
    assert d["parked_seen"]
    gs2 = np.full(n, 5.0)
    gs2[k - 20] = 0.0                # one sample too early
    d2 = B.stand_exit(t, x, y, gs2, t[-1] + 100, REF, np.array([REF]))
    assert not d2["parked_seen"]


def test_stand_exit_repark_near_another_reference():
    """Fails when repark only checks the row's own reference instead of ANY reference of the airport. The gap from
    leaving (t=60) to the later stop (t=500) is 440 s > RUN_GAP_MAX_S, so stationary_runs sees two separate runs and
    the second (150 s, >= REPARK_MIN_S) counts as a run that starts after k."""
    other_ref = (1000.0, 0.0)
    t = np.array([0.0, 30.0, 60.0, 500.0, 530.0, 560.0, 590.0, 620.0, 650.0, 680.0])
    x = np.array([0.0, 5.0, 200.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    y = np.zeros(10)
    gs = np.array([0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0])
    d = B.stand_exit(t, x, y, gs, 1000.0, REF, np.array([REF, other_ref]))
    assert d["exit_t"] == 30.0 and d["repark"]
    assert not B.exit_ok(d)


def test_stand_exit_long_stop_far_from_every_reference_keeps_the_exit():
    """Fails when hold_s is not the plain duration of later stationary runs, or a distant stop is mistaken for a
    repark. The gap from leaving (t=60) to the far stop (t=300) is 240 s > RUN_GAP_MAX_S, so the far stop is its own
    run (900 s span) rather than being merged with the near-stand samples."""
    far = (50_000.0, 50_000.0)
    t = np.array([0.0, 30.0, 60.0] + [300.0 + k * 30 for k in range(31)])   # 31 pts, 30 s apart: span 900 s
    x = np.array([0.0, 5.0, 200.0] + [far[0]] * 31)
    y = np.array([0.0, 0.0, 0.0] + [far[1]] * 31)
    gs = np.array([0.0, 0.0, 20.0] + [0.0] * 31)
    d = B.stand_exit(t, x, y, gs, t[-1] + 100, REF, np.array([REF]))
    assert d["exit_t"] == 30.0 and d["hold_s"] == pytest.approx(900.0) and not d["repark"]
    assert B.exit_ok(d)


def test_stand_exit_no_reference_gives_no_exit_and_nan_distance():
    """Fails when ref_xy=None is not special-cased (crashes, or silently computes a distance to (0, 0))."""
    t = np.array([0.0, 30.0])
    x = np.array([0.0, 5.0])
    y = np.array([0.0, 0.0])
    gs = np.array([0.0, 0.0])
    d = B.stand_exit(t, x, y, gs, 100.0, None, np.empty((0, 2)))
    assert np.isnan(d["exit_t"]) and np.isnan(d["first_ground_dist_m"])
    assert not B.exit_ok(d)


# ================================================================== classify

@pytest.mark.parametrize("est, usable, ok, want", [
    (3600.0, True, True, "A1"),
    (3601.0, True, True, "A2"),
    (5400.0, True, True, "A2"),
    (5401.0, True, True, ""),
    (1800.0, False, True, ""),
    (1800.0, True, False, ""),
    (np.nan, True, True, ""),
    (0.0, True, True, ""),
])
def test_classify_boundaries(est, usable, ok, want):
    """Fails when any class boundary (A1_MAX_S, A2_MAX_S), the usable gate, the ok gate, the NaN guard, or the
    0 < est lower bound is off by one value or missing. Each parametrised case targets one of those."""
    assert B.classify(est, usable, ok) == want


# ================================================================== build_parked / build_table end to end

DAY1, DAY2, DAY3 = "2025-03-10", "2025-03-11", "2025-03-12"
MID = _day_start(DAY2)
D1START = _day_start(DAY1)


def _world(tmp_path):
    """Two days with ADS-B (DAY1, DAY2) + one day without any file (DAY3); two airports (EHAM, EGLL).

    EHAM/S1: three matched, callsign-joined flights parked at (0, 0) create the stand reference; a fourth,
    UNMATCHED, time-joined (type-matched) departure from S1 rolls over midnight (parked on DAY1, take-off on DAY2)
    and must get cls A1 with a hand-constructed exit_rel_s of 600 s. A fifth EHAM row (DAY3) has no ADS-B at all.
    EGLL/E1: one matched flight alone (not enough to form a reference) plus one unmatched, time-joined departure
    from the same stand, so EGLL's rows are reachable but never get an estimate (ref_n NaN).
    """
    adsb_dir = tmp_path / "adsb"
    rows = []
    day1_rows, day2_rows = [], []

    # ---- EHAM / S1: three reference-forming matched flights -------------------------------------------------
    for i in range(3):
        tpark = D1START + 40_000 + i * 2_000
        ttake = tpark + 900
        mvt = ttake
        aobt = tpark + 200
        mid = 100 + i
        day1_rows += _departure_track(f"ehS1-{i}", f"CS{i}", tpark, 11, (0.0, 0.0), tpark + 300, ttake)
        rows.append(_row(mid, f"CS{i}", mvt, aobt, stand="S1", airport="EHAM"))

    # ---- EHAM / S1: one unmatched, time-joined, midnight-rollover departure ---------------------------------
    tpark_u = MID - 600
    ttake_u = tpark_u + 900              # = MID + 300, on DAY2
    mvt_u = ttake_u
    track_u = _departure_track("ehS1-u", "ZZZZZ", tpark_u, 11, (0.0, 0.0), tpark_u + 300, ttake_u)
    for r in track_u:
        (day1_rows if r["t"] < MID else day2_rows).append(r)
    rows.append(_row(104, "NOPE-U", mvt_u, None, stand="S1", airport="EHAM"))   # callsign never matches -> time join

    # ---- EHAM: a departure on DAY3, which has no ADS-B file at all -------------------------------------------
    rows.append(_row(105, "NOPE-3", _secs(pd.Timestamp(DAY3 + "T10:00:00", tz="UTC")), None, stand="S9",
                     airport="EHAM"))

    # ---- EGLL / E1: one matched (alone: not enough for a reference) + one unmatched, same stand --------------
    tpark_g = D1START + 50_000
    ttake_g = tpark_g + 900
    day1_rows += _departure_track("eglE1-m", "BA100", tpark_g, 11, (0.0, 0.0), tpark_g + 300, ttake_g,
                                  airport="EGLL", actype="B738")
    rows.append(_row(201, "BA100", ttake_g, tpark_g + 200, stand="E1", airport="EGLL", actype="B738"))

    tpark_gu = D1START + 54_000
    ttake_gu = tpark_gu + 900
    day1_rows += _departure_track("eglE1-u", "ZZGU", tpark_gu, 11, (0.0, 0.0), tpark_gu + 300, ttake_gu,
                                  airport="EGLL", actype="B738")
    rows.append(_row(202, "NOPE-GU", ttake_gu, None, stand="E1", airport="EGLL", actype="B738"))

    _write_adsb_day(adsb_dir, DAY1, day1_rows)
    _write_adsb_day(adsb_dir, DAY2, day2_rows)
    _manifest(adsb_dir)

    rows_df = _rows_df(rows)
    rows_path = _write_parquet(tmp_path / "rows.parquet", rows_df)
    return {"adsb_dir": adsb_dir, "rows_path": rows_path, "rows_df": rows_df}


def test_build_parked_and_build_table_end_to_end(tmp_path, few_joins_make_a_window):
    """One fixture, several load-bearing assertions (see sub-asserts below); each names the mutant it catches."""
    w = _world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    parked, preport = B.build_parked(rows, [w["adsb_dir"]])

    # unmatched rows never contribute a parked position. Fails when build_parked drops the AOBT_3_flt.notna() filter.
    assert 104.0 not in set(parked.MVT_ID_mvt) and 202.0 not in set(parked.MVT_ID_mvt)
    assert set(parked[parked.ADEP_mvt == "EHAM"].MVT_ID_mvt) == {100.0, 101.0, 102.0}

    table, treport = B.build_table(rows, [w["adsb_dir"]], parked)

    # output sorted by id. Fails when the final sort_values("MVT_ID_mvt") step is dropped.
    assert table.MVT_ID_mvt.tolist() == sorted(table.MVT_ID_mvt.tolist())
    assert table.MVT_ID_mvt.tolist() == [100.0, 101.0, 102.0, 104.0, 105.0, 201.0, 202.0]

    t = table.set_index("MVT_ID_mvt")

    # the reference forms from 3 flights and the unmatched departure from S1 gets A1 with the hand-built value.
    # Fails when exit_rel_s is computed with the wrong sign (exit_t - mvt_s instead of mvt_s - exit_t) or the
    # reference is not built at all.
    assert t.loc[104.0, "cls"] == "A1"
    assert t.loc[104.0, "exit_rel_s"] == pytest.approx(600.0, abs=1e-6)
    assert t.loc[104.0, "unmatched"] == True and t.loc[104.0, "join_type"] == "stand"

    # a departure just after midnight UTC used the previous day's ground samples (if it hadn't, there would be no
    # exit at all: every (0,0) sample lives on DAY1). Fails when read_day's previous-day stitching is broken for
    # the day the unmatched row's own MVT falls on.
    assert not np.isnan(t.loc[104.0, "exit_rel_s"])

    # rows of a day without any ADS-B file are present with NaNs, not dropped and not erroring.
    assert t.loc[105.0, "join_type"] == "" and np.isnan(t.loc[105.0, "exit_rel_s"]) and t.loc[105.0, "cls"] == ""
    assert t.loc[105.0, "row_type"] == "A320" and t.loc[105.0, "adsb_icao"] == "" and not t.loc[105.0, "exit_ok"]

    # EGLL never reaches REF_MIN_FLIGHTS (only one contributing flight): no estimate for either EGLL row.
    assert np.isnan(t.loc[201.0, "ref_n"]) and np.isnan(t.loc[202.0, "ref_n"])
    assert t.loc[201.0, "cls"] == "" and t.loc[202.0, "cls"] == ""

    # the matched reference-forming flights also get their own (non-leave-one-out) exit and class.
    for mid in (100.0, 101.0, 102.0):
        assert t.loc[mid, "exit_rel_s"] == pytest.approx(600.0, abs=1e-6)
        assert t.loc[mid, "cls"] == "A1"

    # hand-computed stop-rule statistics (report structure is this module's own; values are hand-derived from the
    # fixture above). Fails when a stop-rule statistic uses the wrong row population or the wrong agreement test.
    eham = treport["stop_rules"]["EHAM"]
    assert eham["callsign_matched_type_agree_share"] == pytest.approx(1.0)
    assert eham["stand_unmatched_type_agree_share"] == pytest.approx(1.0) and eham["n_stand_unmatched"] == 1
    assert np.isnan(eham["time_unmatched_type_agree_share"])        # row 104's time join became a stand join
    assert eham["matched_exit_mad_s"] == pytest.approx(0.0, abs=1e-6)
    egll = treport["stop_rules"]["EGLL"]
    assert egll["callsign_matched_type_agree_share"] == pytest.approx(1.0)
    assert egll["time_unmatched_type_agree_share"] == pytest.approx(1.0)
    assert np.isnan(egll["matched_exit_mad_s"])

    assert treport["missing_days"] == [DAY3]


def test_an_unjoined_row_on_a_day_with_adsb_has_an_empty_join_type(tmp_path):
    """The table's join_type is "" for every row without a join, whether its day has no ADS-B file or the file holds
    no take-off for it. Fails when an unjoined row on a covered day carries a null join_type (a missing value where
    the schema promises a string, which a later filter such as join_type == "" would silently miss)."""
    adsb_dir = tmp_path / "adsb"
    t0 = D1START + 30_000
    _write_adsb_day(adsb_dir, DAY1, _departure_track("other", "OTH1", t0, 11, (0.0, 0.0), t0 + 300, t0 + 900))
    _manifest(adsb_dir)
    rows = _rows_df([_row(1, "OTH1", t0 + 900, t0 + 200, stand="S1"),        # joined by callsign
                     _row(2, "LONELY", t0 + 20_000, None, stand="S1")])        # no take-off within reach
    r = B.read_rows([_write_parquet(tmp_path / "rows.parquet", rows)])
    parked, _ = B.build_parked(r, [adsb_dir])
    table, _ = B.build_table(r, [adsb_dir], parked)
    t = table.set_index("MVT_ID_mvt")
    assert t.loc[1.0, "join_type"] == "callsign"
    assert t.loc[2.0, "join_type"] == "" and not bool(t.loc[2.0, "usable"])
    assert table.join_type.notna().all()


def test_build_functions_report_progress_once_per_day_airport_pair(tmp_path):
    """A long build must show a heartbeat. Fails when the progress callback is not called for every (day, airport)
    pair in order, when its counters are wrong, or when a day without an ADS-B file is skipped silently."""
    w = _world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    n_pairs = len(rows[["day", "ADEP_mvt"]].drop_duplicates())
    seen = []
    parked, _ = B.build_parked(rows, [w["adsb_dir"]], progress=lambda i, n, day, ap: seen.append((i, n, day, ap)))
    assert [s[0] for s in seen] == list(range(1, n_pairs + 1)) and {s[1] for s in seen} == {n_pairs}
    seen2 = []
    B.build_table(rows, [w["adsb_dir"]], parked, progress=lambda i, n, day, ap: seen2.append((i, n, day, ap)))
    assert [s[0] for s in seen2] == list(range(1, n_pairs + 1)) and {s[1] for s in seen2} == {n_pairs}
    assert [(s[2], s[3]) for s in seen2] == [(s[2], s[3]) for s in seen]
    assert len({(s[2], s[3]) for s in seen}) == n_pairs


def test_build_parked_and_build_table_jobs_1_vs_2_are_byte_identical(tmp_path):
    """Fails when the --jobs>1 path reorders rows, or recomputes anything non-deterministically across chunks."""
    w = _world(tmp_path)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    rc1 = B.main(["parked", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--out", str(out1),
                 "--jobs", "1"])
    rc2 = B.main(["parked", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--out", str(out2),
                 "--jobs", "2"])
    assert rc1 == 0 and rc2 == 0
    assert (out1 / "parked.parquet").read_bytes() == (out2 / "parked.parquet").read_bytes()

    out3, out4 = tmp_path / "out3", tmp_path / "out4"
    rc3 = B.main(["table", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--parked", str(out1),
                 "--out", str(out3), "--jobs", "1"])
    rc4 = B.main(["table", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--parked", str(out1),
                 "--out", str(out4), "--jobs", "2"])
    assert rc3 == 0 and rc4 == 0
    assert (out3 / "table.parquet").read_bytes() == (out4 / "table.parquet").read_bytes()


# ================================================================== CLI

def test_cli_refuses_an_existing_non_empty_out(tmp_path):
    """Fails when --out is overwritten instead of refused (the brief: 'never overwrite')."""
    w = _world(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "junk.txt").write_text("x")
    rc = B.main(["parked", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--out", str(out)])
    assert rc != 0
    assert list(out.iterdir()) == [out / "junk.txt"]


def test_cli_parked_writes_three_files_with_params_hash_and_input_sha(tmp_path):
    """Fails when report.json omits params_hash or the input file's sha256 (no way to audit what a table was
    built from)."""
    w = _world(tmp_path)
    out = tmp_path / "out"
    rc = B.main(["parked", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--out", str(out)])
    assert rc == 0
    assert (out / "parked.parquet").exists() and (out / "report.json").exists()
    report = json.loads((out / "report.json").read_text())
    assert report["params_hash"] == B.params_hash()
    assert report["params"] == B.params()
    want_sha = hashlib.sha256(w["rows_path"].read_bytes()).hexdigest()
    assert report["inputs"][str(w["rows_path"].resolve())] == want_sha


def test_cli_table_pooling_two_parked_dirs_sharing_an_id_is_refused(tmp_path):
    """Fails when two --parked directories that both contributed the same MVT_ID_mvt are silently concatenated
    (double-counted evidence for one flight's parked position)."""
    w = _world(tmp_path)
    out = tmp_path / "pk"
    rc = B.main(["parked", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--out", str(out)])
    assert rc == 0
    out2 = tmp_path / "tbl"
    rc2 = B.main(["table", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--parked", str(out),
                 str(out), "--out", str(out2)])
    assert rc2 != 0


# ================================================================== AMEND-A1 (2026-10-05): registered hash, report

def test_params_hash_is_the_registered_one():
    """The hash of the registered set after AMEND-A3 (the eleven constants of REG 17a570a6, the three alias constants
    of AMEND-A2, the six stand-join constants of AMEND-A3), required by uad_price's stop-rule check. Fails when a
    registered constant changes, or when a report-only constant (the agreement bands, the join rank) is added to
    params()."""
    assert B.params_hash() == "d196b652a9a7"


def test_agreement_by_band_shares_and_boundaries():
    """Bands are lo < est <= hi, agreement is |resid| <= 600 s, a row without a finite residual is not counted.
    Fails when a band edge or the 600 s bound is strict on the wrong side, or a NaN residual counts as a miss."""
    est = [1800.0, 1800.5, 3600.0, 3600.5, 5400.0, 100.0, 100.0, 100.0]
    resid = [600.0, -600.5, 0.0, 700.0, -10.0, np.nan, 599.0, -601.0]
    got = B.agreement_by_band(est, resid)
    assert got["0-1800"] == {"n": 3, "share": pytest.approx(2 / 3)}
    assert got["1800-3600"] == {"n": 2, "share": pytest.approx(0.5)}
    assert got["3600-5400"] == {"n": 2, "share": pytest.approx(0.5)}
    assert list(got) == ["0-1800", "1800-3600", "3600-5400"]


def test_agreement_by_band_an_empty_band_has_a_nan_share_and_out_of_band_rows_count_nowhere():
    """Fails when an empty band reports 0.0 or 1.0 instead of NaN, or an estimate outside every band (zero,
    negative, above 5,400 s, NaN) is put into one."""
    got = B.agreement_by_band([100.0, 0.0, -5.0, 5400.5, np.nan], [0.0, 0.0, 0.0, 0.0, 0.0])
    assert got["0-1800"] == {"n": 1, "share": 1.0}
    assert got["1800-3600"]["n"] == 0 and np.isnan(got["1800-3600"]["share"])
    assert got["3600-5400"]["n"] == 0 and np.isnan(got["3600-5400"]["share"])


def test_table_report_type_shares_count_only_rows_with_both_types(tmp_path):
    """Stop rules (a) and (b) are shares over rows where the row and the joined track both report an aircraft type;
    the share of time-joined unmatched rows with a missing type is reported beside it (AMEND-A1 item 1). The
    agreement readout is over matched, classed rows. Fails when a missing type is counted as a disagreement (or as
    agreement) in either share, when the missing share is taken over another population, when the agreement
    readout takes the wrong residual, or when an unmatched row's time join still gives a class."""
    adsb_dir = tmp_path / "adsb"
    spec = [  # id, row callsign, track callsign, row type, track type, MVT - AOBT_3 (None = unmatched), parked at x,
              # stand, seconds of silence between the last parked sample and the next ground sample
        (1, "U1", "X1", "A320", "A320", None, 0.0, "S1", 0),     # time join, types agree
        (2, "U2", "X2", "A320", "B738", None, 0.0, "S1", 200),   # time join, types differ; its track is at the
                                                                 # row's stand but leaves it after a 200 s silence
        (3, "U3", "X3", None, "A320", None, 0.0, "S1", 0),       # time join, the row has no type
        (4, "M4", "M4", "A320", "A320", 700, 0.0, "S1", 0),      # callsign join, types agree
        (5, "M5", "M5", "A320", None, 650, 0.0, "S1", 0),        # callsign join, the track has no type
        (6, "M6", "M6", "A320", "B738", 1500, 0.0, "S1", 0),     # callsign join, types differ; AOBT_3 900 s early
        (7, "M7", "M7", "A320", "A320", 600, 0.0, "S1", 0),      # callsign join, types agree
        (8, "U8", "X8", "A320", "B738", None, 500.0, "S1", 0),   # time join, types differ, parked 500 m from S1
        (10, "U10", "X10", "A320", "A320", None, 0.0, "S9", 0),  # time join, types agree, a stand with no reference
        (11, "M11", "X11", "A320", "B738", 2000, 0.0, "S1", 0),  # MATCHED but only time-joined, types differ: not
                                                                 # usable, so no class although its track has an exit
    ]
    day, rows = [], []
    for i, (mid, row_cs, trk_cs, row_ty, trk_ty, mvt_minus_aobt, park_x, stand, silence) in enumerate(spec):
        tp = D1START + 20_000 + i * 5_000
        day += _departure_track(f"ic{mid}", trk_cs, tp, 11, (park_x, 0.0), tp + 300 + silence, tp + 900,
                                actype=trk_ty)
        aobt = None if mvt_minus_aobt is None else tp + 900 - mvt_minus_aobt
        rows.append(_row(mid, row_cs, tp + 900, aobt, stand=stand, actype=row_ty))
    rows.append(_row(9, "LONELY", D1START + 80_000, None, stand="S1"))      # unmatched, no take-off within reach
    _write_adsb_day(adsb_dir, DAY1, day)
    _manifest(adsb_dir)
    r = B.read_rows([_write_parquet(tmp_path / "rows.parquet", _rows_df(rows))])
    parked, _ = B.build_parked(r, [adsb_dir])
    table, rep = B.build_table(r, [adsb_dir], parked)

    assert table.set_index("MVT_ID_mvt").join_type.to_dict() == {
        1.0: "time", 2.0: "time", 3.0: "time", 4.0: "callsign", 5.0: "callsign", 6.0: "callsign", 7.0: "callsign",
        8.0: "time", 9.0: "", 10.0: "time", 11.0: "time"}
    s = rep["stop_rules"]["EHAM"]
    assert s["time_unmatched_type_agree_share"] == pytest.approx(0.5)       # rows 1 and 10 of rows 1, 2, 8, 10
    assert s["time_unmatched_type_missing_share"] == pytest.approx(0.2)     # row 3 of the five time joins
    assert s["n_time_unmatched"] == 5 and s["n_time_unmatched_typed"] == 4
    assert s["time_unmatched_type_equal_share"] == pytest.approx(0.5)
    assert s["callsign_matched_type_equal_share"] == pytest.approx(2 / 3)
    assert s["callsign_matched_type_agree_share"] == pytest.approx(2 / 3)
    assert s["n_callsign_matched_typed"] == 3
    assert rep["aliases"] == []                 # three joins of known aircraft are far below ALIAS_MIN_ROWS
    # matched classed rows 4, 5, 6, 7 each leave the stand 600 s before take-off; est - (MVT - AOBT_3) is
    # -100, -50, -900 and 0 s: median -75 s, absolute deviations 25, 25, 825, 75 -> rule (c)'s statistic is 50 s.
    # Row 11 (matched, exit found, residual -1,400 s) has no class and must not enter it (it would make it 100 s).
    assert s["n_matched_classed"] == 4
    assert s["matched_exit_mad_s"] == pytest.approx(50.0)
    assert rep["readouts"]["EHAM"]["matched_agree_600s_by_band"]["0-1800"] == {"n": 4, "share": pytest.approx(0.75)}
    # four matched callsign joins are far below WINDOW_MIN_ROWS: no window, no stand join, every time join stays
    assert rep["offset_windows"] == {} and (table.join_type != "stand").all()
    # an unmatched row's time join gives no class any more (AMEND-A3 item 4); a matched row's does when types agree
    t = table.set_index("MVT_ID_mvt")
    assert t.loc[1.0, "type_agree"] == 1.0 and not bool(t.loc[1.0, "usable"]) and t.loc[1.0, "cls"] == ""


def test_build_parked_takes_no_position_from_a_time_join_of_another_aircraft_type(tmp_path):
    """Stand references come from USABLE matched departures only (AMEND-A1 item 2). Fails when build_parked records
    a parked position for a matched flight whose time join failed the aircraft-type check."""
    adsb_dir = tmp_path / "adsb"
    t1, t2 = D1START + 30_000, D1START + 40_000
    day = _departure_track("ic-a", "XXA", t1, 11, (0.0, 0.0), t1 + 300, t1 + 900, actype="B738")
    day += _departure_track("ic-b", "XXB", t2, 11, (500.0, 0.0), t2 + 300, t2 + 900, actype="A320")
    _write_adsb_day(adsb_dir, DAY1, day)
    _manifest(adsb_dir)
    rows = _rows_df([_row(1, "ROW1", t1 + 900, t1 + 200, stand="S1", actype="A320"),
                     _row(2, "ROW2", t2 + 900, t2 + 200, stand="S2", actype="A320")])
    r = B.read_rows([_write_parquet(tmp_path / "rows.parquet", rows)])
    parked, rep = B.build_parked(r, [adsb_dir])
    assert parked.MVT_ID_mvt.tolist() == [2.0]
    assert rep["by_airport"]["EHAM"]["joined_by_type"] == {"time": 2}
    assert rep["by_airport"]["EHAM"]["usable"] == 1


def test_build_functions_keep_only_the_airports_asked_for(tmp_path):
    """Fails when the `airports` argument of either pass is ignored (rows of every airport come back)."""
    w = _world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    parked_all, _ = B.build_parked(rows, [w["adsb_dir"]])
    parked, rep = B.build_parked(rows, [w["adsb_dir"]], airports=["EGLL"])
    assert set(parked.ADEP_mvt) == {"EGLL"} and set(rep["by_airport"]) == {"EGLL"}
    table, _ = B.build_table(rows, [w["adsb_dir"]], parked_all, airports=["EGLL"])
    assert table.MVT_ID_mvt.tolist() == [201.0, 202.0] and set(table.ADEP_mvt) == {"EGLL"}


def test_build_table_over_days_without_any_departure_returns_an_empty_table_with_the_schema(tmp_path):
    """Fails when a day list that holds no departure row crashes the build (a table without columns) instead of
    returning zero rows with every column."""
    w = _world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    parked, _ = B.build_parked(rows, [w["adsb_dir"]])
    table, rep = B.build_table(rows, [w["adsb_dir"]], parked, days=["2025-06-01"])
    assert len(table) == 0
    assert list(table.columns) == B.TABLE_COLS
    assert rep["takeoff_claims_dropped"] == 0 and rep["stop_rules"] == {} and rep["aliases"] == []


# ================================================================== AMEND-A1 item 3: one take-off, one flight

def _claim(mvt_id, join_type="callsign", dt=10.0, icao="abc123", t_takeoff=T0, airport="EHAM", unmatched=False,
           row_type="A320", adsb_type="A320"):
    """One table row as finalise_table emits it for a joined, classed departure."""
    eq = 1.0 if row_type == adsb_type else 0.0
    return {"MVT_ID_mvt": float(mvt_id), "ADEP_mvt": airport, "day": "2025-03-10", "unmatched": unmatched,
            "join_type": join_type, "adsb_icao": icao, "t_takeoff": float(t_takeoff), "dt_takeoff": float(dt),
            "row_type": row_type, "adsb_type": adsb_type, "type_equal": eq, "type_agree": eq, "usable": True,
            "ref_n": 5.0, "exit_rel_s": 600.0, "continuous": True, "parked_seen": True, "repark": False,
            "exit_ok": True, "hold_s": 0.0, "n_ground": 14.0, "first_ground_rel_s": 900.0,
            "first_ground_dist_m": 0.0, "cls": "A1"}


def _unjoined(mvt_id, airport="EHAM"):
    """One table row as finalise_table emits it for a departure without a take-off."""
    return {"MVT_ID_mvt": float(mvt_id), "ADEP_mvt": airport, "day": "2025-03-10", "unmatched": True,
            "join_type": "", "adsb_icao": "", "t_takeoff": np.nan, "dt_takeoff": np.nan, "row_type": "A320",
            "adsb_type": "", "type_equal": np.nan, "type_agree": np.nan, "usable": False, "ref_n": np.nan,
            "exit_rel_s": np.nan, "continuous": False, "parked_seen": False, "repark": False, "exit_ok": False,
            "hold_s": np.nan, "n_ground": np.nan, "first_ground_rel_s": np.nan, "first_ground_dist_m": np.nan,
            "cls": ""}


def _assert_unjoined(r):
    assert r["join_type"] == "" and r["adsb_icao"] == "" and r["adsb_type"] == "" and r["cls"] == ""
    assert not r["usable"] and not r["continuous"] and not r["parked_seen"] and not r["repark"] and not r["exit_ok"]
    for k in ("t_takeoff", "dt_takeoff", "type_equal", "type_agree", "exit_rel_s", "hold_s", "n_ground",
              "first_ground_rel_s", "first_ground_dist_m"):
        assert pd.isna(r[k]), k
    assert r["row_type"] == "A320"             # the row's own aircraft type is not a property of the join


def test_claim_helpers_carry_exactly_the_table_columns():
    """The hand-built rows below stand in for build_table's output. Fails when the table gains or loses a column
    and these helpers are not updated with it (the de-duplication tests would then test a frame the builder never
    emits)."""
    assert list(_claim(1)) == B.TABLE_COLS and list(_unjoined(2)) == B.TABLE_COLS


def test_dedupe_takeoffs_a_callsign_claim_beats_a_time_claim_with_a_smaller_time_difference():
    """Fails when the claims are ranked by time difference before join type, or the loser keeps any join-derived
    value."""
    tb = pd.DataFrame([_claim(1, "time", dt=-5.0), _claim(2, "callsign", dt=150.0)])
    out, n = B.dedupe_takeoffs(tb)
    o = out.set_index("MVT_ID_mvt")
    assert n == 1
    assert o.loc[2.0, "join_type"] == "callsign" and o.loc[2.0, "cls"] == "A1" and o.loc[2.0, "exit_rel_s"] == 600.0
    assert o.loc[2.0, "adsb_icao"] == "abc123" and o.loc[2.0, "t_takeoff"] == T0 and bool(o.loc[2.0, "usable"])
    _assert_unjoined(o.loc[1.0])
    assert o.loc[1.0, "ref_n"] == 5.0          # the stand's reference count is not a property of the join


@pytest.mark.parametrize("dt_a, dt_b, keeps", [(-25.0, 10.0, 2.0), (25.0, -10.0, 2.0), (-10.0, 25.0, 1.0)])
def test_dedupe_takeoffs_the_smaller_absolute_time_difference_wins_within_a_join_type(dt_a, dt_b, keeps):
    """Fails when the signed difference is compared (in either direction) or the larger difference is kept."""
    tb = pd.DataFrame([_claim(1, "time", dt=dt_a), _claim(2, "time", dt=dt_b)])
    out, n = B.dedupe_takeoffs(tb)
    o = out.set_index("MVT_ID_mvt")
    loses = 3.0 - keeps
    assert n == 1 and o.loc[keeps, "join_type"] == "time"
    _assert_unjoined(o.loc[loses])


def test_dedupe_takeoffs_equal_claims_keep_the_smaller_id():
    """Fails when a tie is broken by frame order (the first row here has the larger id) or by the larger id."""
    tb = pd.DataFrame([_claim(7, "time", dt=10.0), _claim(3, "time", dt=-10.0)])
    out, n = B.dedupe_takeoffs(tb)
    o = out.set_index("MVT_ID_mvt")
    assert n == 1 and o.loc[3.0, "join_type"] == "time"
    _assert_unjoined(o.loc[7.0])


def test_dedupe_takeoffs_leaves_claims_on_different_takeoffs_alone():
    """The same aircraft at another take-off time, another aircraft at the same time, the same aircraft and time at
    another airport, and unjoined rows share no take-off. Fails when the event key drops the aircraft address, the
    take-off time or the airport, or when unjoined rows are grouped as one event."""
    tb = pd.DataFrame([_claim(1), _claim(2, t_takeoff=T0 + 3600), _claim(3, icao="def456"),
                       _claim(4, airport="EGLL"), _unjoined(5), _unjoined(6)])
    out, n = B.dedupe_takeoffs(tb)
    assert n == 0
    pd.testing.assert_frame_equal(out, tb)


def test_dedupe_takeoffs_keeps_row_order_dtypes_and_the_input_and_is_idempotent():
    """Fails when the result is re-ordered, a column changes dtype (the parquet schema would change), the input
    frame is modified in place, or a second pass drops anything more."""
    tb = pd.DataFrame([_claim(5, "time", dt=20.0), _claim(1, "callsign", dt=90.0), _claim(9, "time", dt=1.0),
                       _claim(4, icao="zzz999")])
    tb.index = [10, 10, 11, 12]                 # a duplicated index label must not confuse the row selection
    before = tb.copy()
    out, n = B.dedupe_takeoffs(tb)
    assert n == 2
    assert out.MVT_ID_mvt.tolist() == [5.0, 1.0, 9.0, 4.0]
    assert out.join_type.tolist() == ["", "callsign", "", "callsign"]
    assert (out.dtypes == tb.dtypes).all()
    pd.testing.assert_frame_equal(tb, before)
    out2, n2 = B.dedupe_takeoffs(out)
    assert n2 == 0
    pd.testing.assert_frame_equal(out2, out)


def test_dedupe_takeoffs_refuses_a_joined_row_without_an_aircraft_or_a_takeoff_time():
    """Fails when a joined row whose take-off cannot be identified is silently grouped with other such rows (two
    unrelated flights would then be treated as one take-off) instead of refused."""
    with pytest.raises(ValueError, match="1 joined row"):
        B.dedupe_takeoffs(pd.DataFrame([_claim(1), _claim(2, icao="")]))
    with pytest.raises(ValueError, match="2 joined row"):
        B.dedupe_takeoffs(pd.DataFrame([_claim(1, t_takeoff=np.nan), _claim(2, t_takeoff=np.nan)]))


@pytest.mark.parametrize("a, b, keeps", [(("time", 1.0), ("stand", 40.0), 2.0), (("stand", 1.0), ("callsign", 150.0), 2.0),
                                         (("stand", 9.0), ("time", 3.0), 1.0)])
def test_dedupe_takeoffs_ranks_a_stand_join_between_a_callsign_join_and_a_time_join(a, b, keeps):
    """AMEND-A3 item 5. Fails when a stand join loses to a time join, beats a callsign join, or the time difference
    decides across join types."""
    tb = pd.DataFrame([_claim(1, a[0], dt=a[1]), _claim(2, b[0], dt=b[1])])
    out, n = B.dedupe_takeoffs(tb)
    o = out.set_index("MVT_ID_mvt")
    assert n == 1 and o.loc[keeps, "join_type"] != "" and o.loc[3.0 - keeps, "join_type"] == ""


def test_dedupe_takeoffs_refuses_an_unknown_join_type():
    """Fails when a join type the ranking does not know is ranked silently (first or last) instead of refused."""
    with pytest.raises(ValueError, match="unknown join type"):
        B.dedupe_takeoffs(pd.DataFrame([_claim(1, "callsign"), _claim(2, "stitched")]))


def test_near_duplicate_takeoffs_counts_one_aircraft_taking_off_twice_within_a_roll():
    """Fails when two claims of one aircraft at one airport at most ROLL_MERGE_S apart are not counted, when the
    bound is strict, or when another aircraft, another airport, an identical time or an unjoined row is counted."""
    m = AF.ROLL_MERGE_S
    assert B.near_duplicate_takeoffs(pd.DataFrame([_claim(1), _claim(2, t_takeoff=T0 + 60)])) == 1
    assert B.near_duplicate_takeoffs(pd.DataFrame([_claim(1), _claim(2, t_takeoff=T0 + m)])) == 1
    assert B.near_duplicate_takeoffs(pd.DataFrame([_claim(1), _claim(2, t_takeoff=T0 + m + 1)])) == 0
    assert B.near_duplicate_takeoffs(pd.DataFrame([_claim(1), _claim(2, t_takeoff=T0 + 60, icao="def456")])) == 0
    assert B.near_duplicate_takeoffs(pd.DataFrame([_claim(1, airport="EGLL"), _claim(2, t_takeoff=T0 + 60)])) == 0
    assert B.near_duplicate_takeoffs(pd.DataFrame([_claim(1), _claim(2)])) == 0
    assert B.near_duplicate_takeoffs(pd.DataFrame([_unjoined(1), _unjoined(2)])) == 0
    assert B.near_duplicate_takeoffs(pd.DataFrame([_claim(2, t_takeoff=T0 + 60), _claim(1), _unjoined(3)])) == 1


def _midnight_world(tmp_path):
    """One aircraft ("mid-1", callsign MIDN1, stand S1) takes off 10 s after UTC midnight between DAY1 and DAY2.
    Row 301 (DAY1, movement time 20 s before midnight, matched) carries its callsign. Row 302 (DAY2, movement time
    25 s after midnight, unmatched) has another callsign, the same aircraft type and no take-off of its own. Each
    day's join, run alone, hands the take-off to its own row: 301 by callsign, 302 as the only take-off within
    30 s. Three more matched flights on DAY1 give stand S1 its reference; row 303 (DAY2, unmatched, own take-off
    5,000 s after midnight) needs that DAY1-built reference for its class."""
    adsb_dir = tmp_path / "adsb"
    day1, day2, rows = [], [], []
    for i in range(3):
        tp = D1START + 40_000 + i * 2_000
        day1 += _departure_track(f"ref-{i}", f"RF{i}", tp, 11, (0.0, 0.0), tp + 300, tp + 900)
        rows.append(_row(100 + i, f"RF{i}", tp + 900, tp + 200, stand="S1"))

    ttake = MID + 10
    for r in _departure_track("mid-1", "MIDN1", ttake - 900, 11, (0.0, 0.0), ttake - 600, ttake):
        (day1 if r["t"] < MID else day2).append(r)
    rows.append(_row(301, "MIDN1", MID - 20, MID - 720, stand="S1"))
    rows.append(_row(302, "OTHER2", MID + 25, None, stand="S1"))

    tp3 = MID + 4_100
    day2 += _departure_track("day2-a", "DAY2A", tp3, 11, (0.0, 0.0), tp3 + 300, tp3 + 900)
    rows.append(_row(303, "DAY2A", tp3 + 900, None, stand="S1"))

    _write_adsb_day(adsb_dir, DAY1, day1)
    _write_adsb_day(adsb_dir, DAY2, day2)
    _manifest(adsb_dir)
    rows_path = _write_parquet(tmp_path / "rows.parquet", _rows_df(rows))
    return {"adsb_dir": adsb_dir, "rows_path": rows_path}


def test_each_day_alone_hands_the_midnight_takeoff_to_its_own_row(tmp_path):
    """The premise of the de-duplication: joined one day at a time, both rows claim the take-off. Fails when the
    fixture stops producing the double claim (the tests below would then pass without any de-duplication)."""
    w = _midnight_world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    j1, _ = B.join_day(rows[rows.day == DAY1], B.read_day([w["adsb_dir"]], DAY1, "EHAM"))
    j2, _ = B.join_day(rows[rows.day == DAY2], B.read_day([w["adsb_dir"]], DAY2, "EHAM"))
    a = j1.set_index("MVT_ID_mvt").loc[301.0]
    b = j2.set_index("MVT_ID_mvt").loc[302.0]
    assert a.join_type == "callsign" and b.join_type == "time" and bool(b.usable)
    assert a.adsb_icao == b.adsb_icao == "mid-1" and a.t_takeoff == b.t_takeoff == MID + 10


def test_build_table_a_takeoff_claimed_from_both_sides_of_midnight_serves_one_flight(tmp_path):
    """Fails when the row of the other day keeps its claim on a take-off that a callsign join already holds (an
    estimate read off another flight's track), or when the winner loses anything."""
    w = _midnight_world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    parked, _ = B.build_parked(rows, [w["adsb_dir"]])
    table, rep = B.build_table(rows, [w["adsb_dir"]], parked)
    t = table.set_index("MVT_ID_mvt")

    assert t.loc[301.0, "join_type"] == "callsign" and t.loc[301.0, "cls"] == "A1"
    assert t.loc[301.0, "adsb_icao"] == "mid-1" and t.loc[301.0, "t_takeoff"] == MID + 10
    assert t.loc[301.0, "exit_rel_s"] == pytest.approx(570.0, abs=1e-6)     # (MID - 20) - (MID - 590)
    _assert_unjoined(t.loc[302.0])
    assert bool(t.loc[302.0, "unmatched"]) and t.loc[302.0, "ref_n"] == 4.0
    assert t.loc[303.0, "join_type"] == "callsign" and t.loc[303.0, "cls"] == "A1"
    assert rep["takeoff_claims_dropped"] == 1 and rep["near_duplicate_takeoffs"] == 0
    assert list(table.columns) == B.TABLE_COLS


def test_build_table_the_stand_rule_does_not_offer_a_takeoff_that_a_callsign_join_holds(tmp_path,
                                                                                          few_joins_make_a_window):
    """With a window at the airport, row 302's only candidate is the take-off that row 301 holds by callsign (its
    track did leave stand S1, row 302's stand). Fails when the stand rule hands it to row 302."""
    w = _midnight_world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    parked, _ = B.build_parked(rows, [w["adsb_dir"]])
    raw, cands, _ = B.build_raw(rows, [w["adsb_dir"]], parked)
    mine = cands[cands.MVT_ID_mvt == 302.0]
    assert mine.adsb_icao.tolist() == ["mid-1"] and bool(mine.exit_ok.iloc[0]) and mine.dt_takeoff.tolist() == [-15.0]
    table, rep = B.build_table(rows, [w["adsb_dir"]], parked)
    t = table.set_index("MVT_ID_mvt")
    assert "EHAM" in rep["offset_windows"]
    assert t.loc[301.0, "join_type"] == "callsign" and t.loc[302.0, "join_type"] == "" and t.loc[302.0, "cls"] == ""


def test_cli_table_jobs_2_resolves_a_claim_across_chunks_and_pools_references_over_every_day(tmp_path):
    """With --jobs 2 the two days are built in different processes. Fails when a take-off claimed from both chunks
    still serves two rows, when a chunk builds its stand references from its own days only (row 303's class needs
    DAY1's parked flights), or when the --jobs output differs from the single-process one in any byte."""
    w = _midnight_world(tmp_path)
    pk = tmp_path / "pk"
    assert B.main(["parked", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--out", str(pk)]) == 0
    outs = {}
    for jobs in ("1", "2"):
        out = tmp_path / f"tb{jobs}"
        assert B.main(["table", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--parked", str(pk),
                       "--out", str(out), "--jobs", jobs]) == 0
        outs[jobs] = out
    assert (outs["1"] / "table.parquet").read_bytes() == (outs["2"] / "table.parquet").read_bytes()
    t = pd.read_parquet(outs["2"] / "table.parquet").set_index("MVT_ID_mvt")
    assert t.loc[301.0, "join_type"] == "callsign" and t.loc[302.0, "join_type"] == "" and t.loc[302.0, "cls"] == ""
    assert t.loc[303.0, "cls"] == "A1"
    for jobs in ("1", "2"):
        rep = json.loads((outs[jobs] / "report.json").read_text())
        assert rep["takeoff_claims_dropped"] == 1 and rep["near_duplicate_takeoffs"] == 0


def test_cli_reports_carry_the_sha256_of_the_code_that_built_them(tmp_path):
    """Fails when a report does not name the builder and the shipped join it ran with (a table could then not be
    told apart from one built by an earlier version of this module)."""
    w = _world(tmp_path)
    pk, tb = tmp_path / "pk", tmp_path / "tb"
    assert B.main(["parked", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--out", str(pk)]) == 0
    assert B.main(["table", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--parked", str(pk),
                   "--out", str(tb)]) == 0
    for out in (pk, tb):
        code = json.loads((out / "report.json").read_text())["code_sha256"]
        assert set(code) == {"scripts/analysis/uad_build.py", "scripts/adsb_features.py", "scripts/adsb_extract.py"}
        for rel, sha in code.items():
            assert sha == hashlib.sha256((ROOT / rel).read_bytes()).hexdigest(), rel


# ================================================================== AMEND-A2: coding aliases of the aircraft type

def _ident(n, row_type, adsb_type, airframes, airport="EHAM", id0=1000):
    """n joins whose aircraft is known without the type, spread evenly over `airframes` aircraft addresses."""
    return pd.DataFrame({"MVT_ID_mvt": [float(id0 + i) for i in range(n)], "ADEP_mvt": airport,
                         "row_type": row_type, "adsb_type": adsb_type,
                         "adsb_icao": [f"{airport}-{row_type}-{adsb_type}-{i % airframes}" for i in range(n)]})


def test_type_aliases_keeps_a_pair_that_meets_all_three_bounds():
    """Schiphol writes the 777 freighter (B77L in ADS-B) as B772: 20 joins of known aircraft, 5 airframes, every
    B77L row. Fails when a qualifying pair is not listed, its counts are wrong, or an equal pair is listed."""
    ident = pd.concat([_ident(20, "B772", "B77L", 5), _ident(40, "B738", "B738", 10, id0=2000)], ignore_index=True)
    al = B.type_aliases(ident)
    assert al.to_dict("records") == [{"ADEP_mvt": "EHAM", "row_type": "B772", "adsb_type": "B77L", "n": 20,
                                      "airframes": 5, "share": 1.0}]


@pytest.mark.parametrize("n, airframes, n_other, listed", [
    (20, 5, 20, True),       # exactly half of the airport's B77L joins: kept (every bound is inclusive)
    (19, 5, 0, False),       # one row short
    (20, 4, 0, False),       # one airframe short
    (20, 5, 21, False),      # 20 of 41 B77L joins: the airport writes B77L another way more often
])
def test_type_aliases_bounds_are_inclusive_and_each_one_binds(n, airframes, n_other, listed):
    """Fails when a bound is strict, or when one of the three (rows, airframes, share of that ADS-B type's joins)
    is not applied."""
    parts = [_ident(n, "B772", "B77L", airframes)]
    if n_other:
        parts.append(_ident(n_other, "B77L", "B77L", 7, id0=5000))
    al = B.type_aliases(pd.concat(parts, ignore_index=True))
    assert (len(al) == 1) == listed
    if listed:
        assert al.iloc[0].share == pytest.approx(n / (n + n_other))


def test_type_aliases_are_per_airport():
    """Fails when the counts are pooled across airports (two airports with 10 joins each would make an alias that
    neither has), or when the share is taken over every airport's joins of that ADS-B type."""
    ident = pd.concat([_ident(10, "B772", "B77L", 5, airport="EHAM"),
                       _ident(10, "B772", "B77L", 5, airport="EGLL", id0=3000),
                       _ident(20, "AT72", "AT76", 5, airport="LFPG", id0=4000),
                       _ident(60, "AT76", "AT76", 9, airport="EDDM", id0=6000)], ignore_index=True)
    al = B.type_aliases(ident)
    assert al[["ADEP_mvt", "row_type", "adsb_type", "n", "share"]].values.tolist() == [["LFPG", "AT72", "AT76", 20, 1.0]]


def test_type_aliases_of_no_join_is_an_empty_frame_with_the_columns():
    """Fails when a build without any join of a known aircraft crashes instead of having no alias."""
    al = B.type_aliases(pd.DataFrame(columns=B.IDENTITY_COLS))
    assert len(al) == 0 and list(al.columns) == ["ADEP_mvt", "row_type", "adsb_type", "n", "airframes", "share"]


def test_type_aliases_refuses_a_join_without_a_type_or_an_address():
    """Fails when a row with a blank type or aircraft address is counted (a blank key pools unrelated rows)."""
    bad = _ident(3, "B772", "B77L", 3)
    bad.loc[1, "adsb_icao"] = ""
    with pytest.raises(ValueError, match="1 identity row"):
        B.type_aliases(bad)
    bad2 = _ident(3, "B772", "B77L", 3)
    bad2.loc[[0, 2], "row_type"] = None
    with pytest.raises(ValueError, match="2 identity row"):
        B.type_aliases(bad2)


def test_identity_rows_are_callsign_joins_of_matched_rows_with_both_types():
    """Fails when a time join, an unmatched row or a row with a missing type is taken as a join whose aircraft is
    known without the type."""
    tb = pd.DataFrame([_claim(1), _claim(2, "time"), _claim(3, unmatched=True), _claim(4, row_type=""),
                       _claim(5, adsb_type=""), _unjoined(6), _claim(7, row_type="B772", adsb_type="B77L")])
    got = B.identity_rows(tb)
    assert got.MVT_ID_mvt.tolist() == [1.0, 7.0] and list(got.columns) == B.IDENTITY_COLS


ALIAS_EHAM = pd.DataFrame([{"ADEP_mvt": "EHAM", "row_type": "B772", "adsb_type": "B77L", "n": 20, "airframes": 5,
                            "share": 1.0}])


def test_type_agreement_is_equality_or_an_alias_at_the_rows_airport():
    """Fails when an alias is applied at another airport or in the reverse direction, when equal types are not
    agreement, or when a missing type is anything but NaN."""
    tb = pd.DataFrame([
        _claim(1, row_type="B772", adsb_type="B77L"),                    # the alias
        _claim(2, row_type="B772", adsb_type="B77L", airport="EGLL"),    # not an alias at EGLL
        _claim(3, row_type="B77L", adsb_type="B772"),                    # the reverse pair is not one
        _claim(4, row_type="A320", adsb_type="A320"),                    # equal
        _claim(5, row_type="A320", adsb_type="B738"),                    # different
        _claim(6, row_type="", adsb_type="A320"),                        # the row has no type
        _claim(7, row_type="A320", adsb_type=""),                        # the track has no type
        _unjoined(8)])
    got = B.type_agreement(tb, ALIAS_EHAM)
    assert got[:5].tolist() == [1.0, 0.0, 0.0, 1.0, 0.0]
    assert len(got) == 8 and np.isnan(got[5:]).all()
    without = B.type_agreement(tb, B.type_aliases(pd.DataFrame(columns=B.IDENTITY_COLS)))
    assert without[:5].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def _raw(mvt_id, join_type="callsign", **kw):
    """One row as build_raw leaves it (no type_equal / type_agree / usable / cls yet)."""
    exit_ok = kw.pop("exit_ok", True)
    exit_rel_s = kw.pop("exit_rel_s", 600.0)
    full = _claim(mvt_id, join_type, **kw)
    full.update(exit_ok=exit_ok, exit_rel_s=exit_rel_s)
    return {c: full[c] for c in B.RAW_COLS}


def _raw_unjoined(mvt_id, airport="EHAM"):
    return {c: _unjoined(mvt_id, airport)[c] for c in B.RAW_COLS}


def _cand(mvt_id, dt=0.0, icao="c1", t_takeoff=T0, airport="EHAM", exit_rel_s=600.0, exit_ok=True, own=False,
          adsb_type="A320"):
    """One candidate take-off of one row, as build_raw gathers it."""
    return {"MVT_ID_mvt": float(mvt_id), "ADEP_mvt": airport, "own": own, "adsb_icao": icao,
            "t_takeoff": float(t_takeoff), "dt_takeoff": float(dt), "adsb_type": adsb_type, "exit_rel_s": exit_rel_s,
            "continuous": exit_ok, "parked_seen": exit_ok, "repark": False, "exit_ok": exit_ok, "hold_s": 0.0,
            "n_ground": 14.0, "first_ground_rel_s": 900.0, "first_ground_dist_m": 0.0}


def test_cand_and_raw_helpers_carry_exactly_the_builders_columns():
    """Fails when the candidate or raw schema changes and these stand-in rows are not updated with it."""
    assert list(_cand(1)) == B.CAND_COLS
    assert list(_raw(1)) == B.RAW_COLS == [c for c in B.TABLE_COLS if c not in ("type_equal", "type_agree", "usable", "cls")]


def _freighter_identity_raw(n, id0=1000, dt=10.0):
    """n callsign joins of matched 777 freighters (row B772, ADS-B B77L) from 5 airframes, each its own take-off."""
    return [_raw(id0 + i, row_type="B772", adsb_type="B77L", icao=f"frt{i % 5}", t_takeoff=T0 + 4_000 * i, dt=dt)
            for i in range(n)]


# ---- the airport's offset window (AMEND-A3 item 2)

def _joins_with_offsets(dts, airport="EHAM", id0=1, **kw):
    return [_claim(id0 + i, dt=float(d), icao=f"w{id0 + i}", t_takeoff=T0 + 1_000 * (id0 + i), airport=airport, **kw)
            for i, d in enumerate(dts)]


def test_offset_windows_are_the_quantiles_plus_a_margin_and_never_narrower_than_30s():
    """An airport whose take-offs run early (EHAM-like) and one whose take-offs run late (EGLL-like). Fails when
    the quantiles, the 5 s margin or the +-30 s floor is wrong, or when one airport's joins move another's
    window."""
    early = np.linspace(-60.0, 10.0, 201)                 # quantiles -59.65 and 9.65
    late = np.linspace(0.0, 70.0, 201)                    # quantiles 0.35 and 69.65
    tb = pd.DataFrame(_joins_with_offsets(early) + _joins_with_offsets(late, airport="EGLL", id0=1000))
    w = B.offset_windows(tb)
    assert set(w) == {"EHAM", "EGLL"}
    assert w["EHAM"] == (pytest.approx(-64.65), 30.0)
    assert w["EGLL"] == (-30.0, pytest.approx(74.65))


def test_offset_windows_need_enough_callsign_joins_of_matched_departures():
    """199 joins give no window; time joins, stand joins and unmatched rows do not count toward the 200. Fails when
    the bound is off by one or another kind of join is counted."""
    rows = (_joins_with_offsets(np.zeros(199)) + _joins_with_offsets(np.zeros(50), id0=1000, join_type="time")
            + _joins_with_offsets(np.zeros(50), id0=2000, join_type="stand")
            + _joins_with_offsets(np.zeros(50), id0=3000, unmatched=True)
            + _joins_with_offsets(np.zeros(200), id0=4000, airport="EGLL"))
    assert set(B.offset_windows(pd.DataFrame(rows))) == {"EGLL"}
    assert B.offset_windows(pd.DataFrame(rows + _joins_with_offsets([0.0], id0=9000))).keys() == {"EHAM", "EGLL"}


def test_offset_windows_ignore_the_offsets_of_other_joins():
    """Fails when the quantiles are taken over time joins or unmatched rows as well (their offsets would drag the
    window)."""
    rows = (_joins_with_offsets(np.zeros(200)) + _joins_with_offsets(np.full(300, -100.0), id0=1000, join_type="time")
            + _joins_with_offsets(np.full(300, 100.0), id0=2000, unmatched=True))
    assert B.offset_windows(pd.DataFrame(rows))["EHAM"] == (-30.0, 30.0)


# ---- which candidates are stand-consistent, and which row gets one (AMEND-A3 item 3)

W = {"EHAM": (-54.0, 30.0)}


@pytest.mark.parametrize("kw, want", [
    ({}, True),
    ({"dt": -54.0}, True), ({"dt": -54.5}, False), ({"dt": 30.0}, True), ({"dt": 30.5}, False),
    ({"exit_ok": False}, False),
    ({"exit_rel_s": 5400.0}, True), ({"exit_rel_s": 5400.5}, False), ({"exit_rel_s": 0.0}, False),
    ({"exit_rel_s": np.nan, "exit_ok": False}, False),
    ({"airport": "EGLL"}, False),
])
def test_consistent_candidates_window_guards_and_band(kw, want):
    """The window is inclusive at both ends, the exit must pass its guards, 0 < est <= 5,400 s, and an airport
    without a window has no consistent candidate. Fails when one of these is dropped or an edge is strict on the
    wrong side."""
    assert B.consistent_candidates(pd.DataFrame([_cand(1, **kw)]), W, {}).tolist() == [want]


def test_consistent_candidates_a_takeoff_held_by_another_rows_callsign_join_is_out():
    """Fails when a take-off that a callsign join already holds is offered to another row, or when the holder itself
    is refused its own take-off (the tested replication needs it)."""
    c = pd.DataFrame([_cand(1, icao="x"), _cand(2, icao="x"), _cand(3, icao="y"), _cand(4, icao="x", airport="EGLL")])
    held = {("EHAM", "x", T0): 2.0}
    assert B.consistent_candidates(c, dict(W, EGLL=(-30.0, 30.0)), held).tolist() == [False, True, True, True]


def test_stand_joins_picks_the_only_consistent_candidate_and_counts_the_rest():
    """Fails when a row with several consistent candidates is given one of them, when a row's only consistent
    candidate is not picked because it has inconsistent ones beside it, or when the counts are wrong."""
    c = pd.DataFrame([
        _cand(1, icao="a"),                                               # one candidate, consistent
        _cand(2, icao="b"), _cand(2, icao="c", dt=5.0),                   # two consistent: ambiguous
        _cand(3, icao="d", exit_ok=False),                                # none consistent
        _cand(4, icao="e", exit_ok=False), _cand(4, icao="f", dt=-40.0, t_takeoff=T0 - 40.0),   # the second only
    ])
    picks, counts = B.stand_joins(c, W, {})
    assert picks.MVT_ID_mvt.tolist() == [1.0, 4.0] and picks.adsb_icao.tolist() == ["a", "f"]
    assert list(picks.columns) == B.CAND_COLS
    assert counts == {"rows": 4, "one": 2, "several": 1, "none": 1}
    none, zero = B.stand_joins(c.iloc[0:0], W, {})
    assert len(none) == 0 and zero == {"rows": 0, "one": 0, "several": 0, "none": 0}


def test_finalise_table_gives_an_unmatched_row_the_takeoff_that_left_its_stand(few_joins_make_a_window):
    """Twenty callsign joins of matched freighters put this airport's take-offs 40 s before its clock (window
    [-45, 30]) and teach the B772 / B77L alias. Fails when the stand join does not replace the shipped time join,
    when a row with several or with no consistent candidate is joined, when a row with a callsign join is re-joined,
    when the usable rule is not the one of AMEND-A3 item 4, or when the result is not TABLE_COLS sorted by id."""
    t9 = T0 + 900_000
    raw = _freighter_identity_raw(20, dt=-40.0) + [
        _raw(90, "time", unmatched=True, row_type="B772", adsb_type="A320", icao="other", t_takeoff=t9 + 20, dt=20.0,
             exit_ok=False, exit_rel_s=np.nan),                       # the shipped join took another aircraft
        _raw(91, "time", unmatched=True, icao="u91", t_takeoff=t9 + 5_000, dt=1.0, exit_ok=False),
        _raw_unjoined(92),
        _raw(93, "callsign", unmatched=True, row_type="A320", adsb_type="B738", icao="u93", t_takeoff=t9 + 9_000),
        _raw_unjoined(94), _raw_unjoined(95), _raw_unjoined(96),
        _raw(97, "time", icao="m97", t_takeoff=t9 + 11_000),          # MATCHED, time join, equal types
        _raw(98, "callsign", icao="m98", t_takeoff=t9 + 12_000, dt=-40.0, exit_ok=False),   # usable, guards failed
        dict(_raw_unjoined(99), unmatched=False),                     # MATCHED and unjoined, with a candidate
    ]
    cands = [
        _cand(90, dt=-44.0, icao="own90", t_takeoff=t9 - 44, adsb_type="B77L", exit_rel_s=700.0),
        _cand(90, dt=20.0, icao="other", t_takeoff=t9 + 20, exit_ok=False, exit_rel_s=np.nan, own=True),
        _cand(90, dt=-46.0, icao="early", t_takeoff=t9 - 46),         # consistent exit, but outside the window
        _cand(91, dt=1.0, icao="u91", t_takeoff=t9 + 5_000, exit_ok=False, own=True),
        _cand(93, dt=5.0, icao="zz", t_takeoff=t9 + 9_005),           # row 93 has a callsign join: not re-joined
        _cand(94, icao="p", t_takeoff=t9 + 7_000), _cand(94, dt=5.0, icao="q", t_takeoff=t9 + 7_005),
        _cand(95, icao="r", t_takeoff=t9 + 8_000, adsb_type="B738"),
        _cand(96, icao="u93", t_takeoff=t9 + 9_000),                  # held by row 93's callsign join
        _cand(99, icao="m99", t_takeoff=t9 + 13_000),                 # consistent, but row 99 is not unmatched
    ]
    table, info = B.finalise_table(pd.DataFrame(raw[::-1]), pd.DataFrame(cands))
    assert list(table.columns) == B.TABLE_COLS
    assert table.MVT_ID_mvt.tolist() == sorted(table.MVT_ID_mvt.tolist())
    t = table.set_index("MVT_ID_mvt")
    got = {i: (t.loc[i, "join_type"], t.loc[i, "adsb_icao"], bool(t.loc[i, "usable"]), t.loc[i, "cls"])
           for i in (90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0)}
    assert got == {90.0: ("stand", "own90", True, "A1"),      # found by its stand; B772 / B77L is the airport's alias
                   91.0: ("time", "u91", False, ""),          # an unmatched row's time join gives no class
                   92.0: ("", "", False, ""),
                   93.0: ("callsign", "u93", True, "A1"),     # a callsign join is left alone and needs no type
                   94.0: ("", "", False, ""),                 # two stand-consistent take-offs: none is taken
                   95.0: ("stand", "r", False, ""),           # found by its stand, but another aircraft type
                   96.0: ("", "", False, ""),                 # its only candidate is held by row 93's callsign join
                   97.0: ("time", "m97", True, "A1"),         # a matched row's time join with equal types, as before
                   98.0: ("callsign", "m98", True, ""),       # usable, but its exit failed the guards: no class
                   99.0: ("", "", False, "")}                 # the stand rule is for unmatched rows only
    assert (t.loc[90.0, "t_takeoff"], t.loc[90.0, "dt_takeoff"], t.loc[90.0, "exit_rel_s"]) == (t9 - 44, -44.0, 700.0)
    assert (t.loc[90.0, "adsb_type"], t.loc[90.0, "type_equal"], t.loc[90.0, "type_agree"]) == ("B77L", 0.0, 1.0)
    assert bool(t.loc[90.0, "exit_ok"]) and t.loc[90.0, "ref_n"] == 5.0
    assert (t.loc[95.0, "type_equal"], t.loc[95.0, "type_agree"]) == (0.0, 0.0)
    assert info["offset_windows"] == {"EHAM": [-45.0, 30.0]}
    assert info["stand_joins"] == {"rows": 5, "one": 2, "several": 1, "none": 2}
    assert info["takeoff_claims_dropped"] == 0
    assert [(a["row_type"], a["adsb_type"], a["n"]) for a in info["aliases"]] == [("B772", "B77L", 20)]
    assert str(table.usable.dtype) == "bool" and str(table.type_agree.dtype) == "float64"


def test_finalise_table_without_a_window_changes_no_join():
    """Twenty callsign joins are below the registered 200: no window, so no stand join, whatever the candidates say.
    Fails when a stand join is made at an airport without an offset window."""
    raw = _freighter_identity_raw(20) + [_raw_unjoined(90)]
    table, info = B.finalise_table(pd.DataFrame(raw), pd.DataFrame([_cand(90, icao="a", t_takeoff=T0 + 900_000)]))
    assert info["offset_windows"] == {} and info["stand_joins"] == {"rows": 1, "one": 0, "several": 0, "none": 1}
    assert table.set_index("MVT_ID_mvt").loc[90.0, "join_type"] == ""


def test_finalise_table_a_stand_join_takes_its_takeoff_from_a_time_join_of_another_row(few_joins_make_a_window):
    """Row 90 is found by its stand at a take-off that row 91 holds by the clock alone: one take-off, one flight, and
    the stand join ranks first (AMEND-A3 item 5). Fails when both rows keep it or the time join wins."""
    t9 = T0 + 900_000
    raw = _freighter_identity_raw(20) + [
        _raw_unjoined(90), _raw(91, "time", icao="shared", t_takeoff=t9, dt=1.0, exit_ok=False)]
    table, info = B.finalise_table(pd.DataFrame(raw), pd.DataFrame([_cand(90, dt=12.0, icao="shared", t_takeoff=t9)]))
    t = table.set_index("MVT_ID_mvt")
    assert t.loc[90.0, "join_type"] == "stand" and t.loc[91.0, "join_type"] == ""
    assert info["takeoff_claims_dropped"] == 1


def test_finalise_table_pools_the_joins_of_known_aircraft_of_another_table(few_joins_make_a_window):
    """For the 2026 build the aliases come from 2025 and 2026 pooled. Fails when `extra_identity` is ignored (ten own
    joins are too few for an alias) or replaces the table's own joins."""
    t9 = T0 + 900_000
    own = _freighter_identity_raw(10) + [_raw_unjoined(90)]
    own[-1]["row_type"] = "B772"
    cands = pd.DataFrame([_cand(90, icao="u90", t_takeoff=t9, adsb_type="B77L")])
    extra = B.identity_rows(B.finalise_table(pd.DataFrame(_freighter_identity_raw(10, id0=7000)))[0])
    alone, info_alone = B.finalise_table(pd.DataFrame(own), cands)
    pooled, info_pooled = B.finalise_table(pd.DataFrame(own), cands, extra)
    a, q = alone.set_index("MVT_ID_mvt"), pooled.set_index("MVT_ID_mvt")
    assert info_alone["aliases"] == [] and (a.loc[90.0, "join_type"], a.loc[90.0, "cls"]) == ("stand", "")
    assert [x["n"] for x in info_pooled["aliases"]] == [20]
    assert q.loc[90.0, "type_agree"] == 1.0 and bool(q.loc[90.0, "usable"]) and q.loc[90.0, "cls"] == "A1"
    assert len(pooled) == 11                                         # the other table's rows are not added


def test_finalise_table_drops_a_shared_takeoff_before_it_counts_joins_of_known_aircraft():
    """Fails when the aliases are counted before the de-duplication (a claim that loses its take-off is no longer a
    join of a known aircraft: 19 are left here, one short of an alias)."""
    rows = _freighter_identity_raw(20)
    rows[1] = dict(rows[1], adsb_icao=rows[0]["adsb_icao"], t_takeoff=rows[0]["t_takeoff"], dt_takeoff=50.0)
    table, info = B.finalise_table(pd.DataFrame(rows))
    assert info["takeoff_claims_dropped"] == 1 and info["aliases"] == []


# ---- the tested replication on departures whose take-off is known by callsign (AMEND-A3 stop rule (d))

def test_check_shares_are_right_and_wrong_over_the_rows_counted():
    """Fails when a share is taken over another count (ambiguous or none read as wrong), or an airport without a
    counted row reports 0 or 1 instead of NaN."""
    got = B.check_shares({"n": 20, "right": 14, "wrong": 3, "ambiguous": 1, "none": 2})
    assert got == {"stand_check_right_share": 0.7, "stand_check_wrong_share": 0.15, "n_stand_check": 20}
    empty = B.check_shares({"n": 0, "right": 0, "wrong": 0, "ambiguous": 0, "none": 0})
    assert np.isnan(empty["stand_check_right_share"]) and np.isnan(empty["stand_check_wrong_share"])
    assert empty["n_stand_check"] == 0


def test_stand_rule_check_scores_the_rule_against_the_callsign_join():
    """Five matched, callsign-joined check rows at an airport whose window is [-45, 30] (400 other callsign joins at
    -40 s keep the two rows at -50 s below the 0.5 % quantile): the rule finds the own take-off (right), is
    ambiguous, finds nothing because the own take-off is outside the window (none), finds another aircraft's
    (wrong); a row whose own take-off is not stand-consistent is not counted. Fails when the row's own callsign join
    blocks its own take-off, when the outcomes are mixed up, or when a row that is not a matched callsign join is
    scored."""
    t9 = T0 + 900_000
    def own_row(i, dt):
        return _raw(i, icao=f"k{i}", t_takeoff=t9 + 1_000 * i, dt=dt)
    raw = _freighter_identity_raw(400, dt=-40.0) + [own_row(110, -40.0), own_row(120, -40.0), own_row(130, -50.0),
                                                   own_row(140, -50.0), own_row(150, -40.0),
                                                   _raw(160, "time", unmatched=True, icao="k160", t_takeoff=t9 + 160_000)]
    def own_cand(i, dt, **kw):
        return _cand(i, dt=dt, icao=f"k{i}", t_takeoff=t9 + 1_000 * i, own=True, **kw)
    cands = pd.DataFrame([
        own_cand(110, -40.0),                                                    # right
        own_cand(120, -40.0), _cand(120, dt=3.0, icao="n120", t_takeoff=t9 + 120_043),      # ambiguous
        own_cand(130, -50.0),                                                    # own outside the window: none
        own_cand(140, -50.0), _cand(140, dt=3.0, icao="n140", t_takeoff=t9 + 140_053),      # wrong
        own_cand(150, -40.0, exit_ok=False), _cand(150, dt=3.0, icao="n150", t_takeoff=t9 + 150_043),   # not counted
        own_cand(160, 0.0),                                                      # an unmatched time join: not a check row
    ])
    table, info = B.finalise_table(pd.DataFrame(raw), cands)
    assert info["offset_windows"] == {"EHAM": [-45.0, 30.0]}
    assert info["stand_rule_check"] == {"EHAM": {"n": 4, "right": 1, "wrong": 1, "ambiguous": 1, "none": 1}}
    # the check never changes a join: every check row keeps its callsign join
    assert (table[table.MVT_ID_mvt.isin([110.0, 120.0, 130.0, 140.0, 150.0])].join_type == "callsign").all()


def _freighter_world(tmp_path, days=(DAY1, DAY2), n=20, id0=500, extra_rows=True):
    """n matched, callsign-joined 777 freighters (row type B772, ADS-B type B77L, 5 airframes, stand S1) split evenly
    over `days`; on the last day two unmatched, time-joined departures from S1 on B77L tracks: row 401 written
    B772 (the airport's alias; in lower case with blanks, as is its track's type, so both must be normalised) and
    row 402 written A320 (another aircraft)."""
    adsb_dir = tmp_path / "adsb"
    by_day = {d: [] for d in days}
    rows = []
    per_day = n // len(days)
    for i in range(n):
        d = days[i // per_day]
        tp = _day_start(d) + 10_000 + (i % per_day) * 4_000
        by_day[d] += _departure_track(f"frt{i % 5}", f"FR{i}", tp, 11, (0.0, 0.0), tp + 300, tp + 900,
                                      actype=(" b77l" if i == 0 else "B77L"))      # normalised like the row's type
        rows.append(_row(id0 + i, f"FR{i}", tp + 900, tp + 200, stand="S1", actype="B772"))
    if extra_rows:
        last = days[-1]
        for k, (mid, row_ty, trk_ty) in enumerate([(401, " b772 ", "b77l "), (402, "A320", "B77L")]):
            tp = _day_start(last) + 60_000 + k * 4_000
            by_day[last] += _departure_track(f"u{mid}", f"ZZ{mid}", tp, 11, (0.0, 0.0), tp + 300, tp + 900,
                                             actype=trk_ty)
            rows.append(_row(mid, f"NOPE{mid}", tp + 900, None, stand="S1", actype=row_ty))
    for d, samples in by_day.items():
        _write_adsb_day(adsb_dir, d, samples)
    _manifest(adsb_dir)
    rows_path = _write_parquet(tmp_path / "rows.parquet", _rows_df(rows))
    return {"adsb_dir": adsb_dir, "rows_path": rows_path}


def test_build_table_a_stand_join_written_in_the_airports_alias_is_usable(tmp_path, few_joins_make_a_window):
    """End to end over two days: the alias is learned from the callsign joins of matched freighters and makes the
    unmatched freighter's stand join usable; the same track under another row type is not. Fails when build_table
    judges a stand join by exact type equality, or when the per-day pass decides usable before the whole table's
    joins are known."""
    w = _freighter_world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    parked, _ = B.build_parked(rows, [w["adsb_dir"]])
    table, rep = B.build_table(rows, [w["adsb_dir"]], parked)
    t = table.set_index("MVT_ID_mvt")
    assert rep["aliases"] == [{"ADEP_mvt": "EHAM", "row_type": "B772", "adsb_type": "B77L", "n": 20, "airframes": 5,
                               "share": 1.0}]
    assert (t.loc[401.0, "join_type"], t.loc[401.0, "type_equal"], t.loc[401.0, "type_agree"]) == ("stand", 0.0, 1.0)
    assert bool(t.loc[401.0, "usable"]) and t.loc[401.0, "cls"] == "A1" and t.loc[401.0, "exit_rel_s"] == 600.0
    assert (t.loc[402.0, "join_type"], t.loc[402.0, "type_agree"]) == ("stand", 0.0)
    assert not bool(t.loc[402.0, "usable"]) and t.loc[402.0, "cls"] == ""
    assert t.loc[401.0, "row_type"] == "B772" and t.loc[401.0, "adsb_type"] == "B77L"
    # type_equal is the exact comparison everywhere: it is what rule (a) and the reported shares are built on
    typed = table[(table.row_type != "") & (table.adsb_type != "")]
    assert (typed.type_equal == (typed.row_type == typed.adsb_type).astype(float)).all()
    s = rep["stop_rules"]["EHAM"]
    assert s["callsign_matched_type_equal_share"] == 0.0 and s["callsign_matched_type_agree_share"] == 1.0
    assert s["stand_unmatched_type_equal_share"] == 0.0 and s["stand_unmatched_type_agree_share"] == 0.5
    assert s["n_stand_unmatched"] == 2 and s["n_stand_unmatched_typed"] == 2
    assert s["stand_unmatched_type_missing_share"] == 0.0 and np.isnan(s["time_unmatched_type_agree_share"])


def test_build_table_without_a_window_reports_its_time_joins_both_ways(tmp_path):
    """Under the registered bound there is no window in this world: rows 401 and 402 keep their time joins. Fails
    when the alias-aware share of the time joins is read from exact equality (or the other way round), or when an
    unmatched row's time join still gives a class."""
    w = _freighter_world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    parked, _ = B.build_parked(rows, [w["adsb_dir"]])
    table, rep = B.build_table(rows, [w["adsb_dir"]], parked)
    t = table.set_index("MVT_ID_mvt")
    assert (t.loc[401.0, "join_type"], t.loc[401.0, "type_agree"], t.loc[401.0, "cls"]) == ("time", 1.0, "")
    s = rep["stop_rules"]["EHAM"]
    assert s["time_unmatched_type_agree_share"] == 0.5 and s["time_unmatched_type_equal_share"] == 0.0
    assert s["n_time_unmatched"] == 2 and s["n_stand_unmatched"] == 0


def test_cli_table_jobs_2_learns_aliases_over_the_whole_table(tmp_path, few_joins_make_a_window):
    """Each of the two day-chunks holds ten joins of known freighters - too few for an alias alone. Fails when a
    chunk finalises on its own (row 401 would not be classed), or when the two outputs differ in any byte."""
    w = _freighter_world(tmp_path)
    pk = tmp_path / "pk"
    assert B.main(["parked", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--out", str(pk)]) == 0
    outs = {}
    for jobs in ("1", "2"):
        outs[jobs] = tmp_path / f"tb{jobs}"
        assert B.main(["table", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--parked", str(pk),
                       "--out", str(outs[jobs]), "--jobs", jobs]) == 0
    assert (outs["1"] / "table.parquet").read_bytes() == (outs["2"] / "table.parquet").read_bytes()
    t = pd.read_parquet(outs["2"] / "table.parquet").set_index("MVT_ID_mvt")
    assert t.loc[401.0, "cls"] == "A1" and t.loc[402.0, "cls"] == ""
    rep = json.loads((outs["2"] / "report.json").read_text())
    assert [(a["row_type"], a["adsb_type"], a["n"]) for a in rep["aliases"]] == [("B772", "B77L", 20)]
    assert rep["alias_from"] == {}
    assert (outs["1"] / "cands.parquet").read_bytes() == (outs["2"] / "cands.parquet").read_bytes()
    c = pd.read_parquet(outs["2"] / "cands.parquet")
    assert list(c.columns) == B.CAND_COLS and set(c.MVT_ID_mvt) >= {401.0, 402.0}


def _two_tables(tmp_path):
    """Table A: twenty joins of known freighters on DAY1 / DAY2 (the alias forms). World B, a later day: three matched
    A320 flights that give stand S1 its reference and one unmatched freighter (row 401, written B772) plus row 402."""
    wa = _freighter_world(tmp_path / "a", extra_rows=False)
    pka, tba = tmp_path / "a" / "pk", tmp_path / "a" / "tb"
    assert B.main(["parked", "--rows", str(wa["rows_path"]), "--adsb-dir", str(wa["adsb_dir"]), "--out", str(pka)]) == 0
    assert B.main(["table", "--rows", str(wa["rows_path"]), "--adsb-dir", str(wa["adsb_dir"]), "--parked", str(pka),
                   "--out", str(tba)]) == 0
    day_b = "2025-04-02"
    adsb_b = tmp_path / "b" / "adsb"
    samples, rows = [], []
    for i in range(3):
        tp = _day_start(day_b) + 10_000 + i * 4_000
        samples += _departure_track(f"ref{i}", f"RB{i}", tp, 11, (0.0, 0.0), tp + 300, tp + 900)
        rows.append(_row(800 + i, f"RB{i}", tp + 900, tp + 200, stand="S1"))
    for k, (mid, row_ty) in enumerate([(401, "B772"), (402, "A320")]):
        tp = _day_start(day_b) + 60_000 + k * 4_000
        samples += _departure_track(f"u{mid}", f"ZZ{mid}", tp, 11, (0.0, 0.0), tp + 300, tp + 900, actype="B77L")
        rows.append(_row(mid, f"NOPE{mid}", tp + 900, None, stand="S1", actype=row_ty))
    _write_adsb_day(adsb_b, day_b, samples)
    _manifest(adsb_b)
    rows_b = _write_parquet(tmp_path / "b" / "rows.parquet", _rows_df(rows))
    pkb = tmp_path / "b" / "pk"
    assert B.main(["parked", "--rows", str(rows_b), "--adsb-dir", str(adsb_b), "--out", str(pkb)]) == 0
    return {"tba": tba, "rows_a": wa["rows_path"], "adsb_a": wa["adsb_dir"], "pka": pka,
            "rows_b": rows_b, "adsb_b": adsb_b, "pkb": pkb}


def test_cli_table_alias_from_pools_another_tables_joins_of_known_aircraft(tmp_path, few_joins_make_a_window):
    """Fails when --alias-from is ignored (the later build has no freighter of a known aircraft of its own, so row
    401 stays unusable), or when the report does not name the table it pooled, with its sha256."""
    w = _two_tables(tmp_path)
    alone, pooled = tmp_path / "alone", tmp_path / "pooled"
    base = ["table", "--rows", str(w["rows_b"]), "--adsb-dir", str(w["adsb_b"]), "--parked", str(w["pkb"])]
    assert B.main(base + ["--out", str(alone)]) == 0
    assert B.main(base + ["--out", str(pooled), "--alias-from", str(w["tba"])]) == 0
    ta = pd.read_parquet(alone / "table.parquet").set_index("MVT_ID_mvt")
    tp = pd.read_parquet(pooled / "table.parquet").set_index("MVT_ID_mvt")
    assert ta.loc[401.0, "join_type"] == "stand" and ta.loc[401.0, "type_agree"] == 0.0 and ta.loc[401.0, "cls"] == ""
    assert tp.loc[401.0, "join_type"] == "stand" and tp.loc[401.0, "type_agree"] == 1.0 and tp.loc[401.0, "cls"] == "A1"
    assert tp.loc[402.0, "type_agree"] == 0.0 and tp.loc[402.0, "cls"] == ""
    assert len(tp) == 5                                              # the other table's rows are not part of this one
    rep = json.loads((pooled / "report.json").read_text())
    src = str((w["tba"] / "table.parquet").resolve())
    assert rep["alias_from"] == {src: hashlib.sha256((w["tba"] / "table.parquet").read_bytes()).hexdigest()}
    assert [(a["row_type"], a["adsb_type"], a["n"], a["airframes"]) for a in rep["aliases"]] == [("B772", "B77L", 20, 5)]


def test_cli_table_alias_from_refuses_a_table_of_other_parameters_or_of_the_same_rows(tmp_path, capsys,
                                                                                     few_joins_make_a_window):
    """Fails when a directory without a report, a table built with another parameter set, or a table that holds this
    build's own rows (its joins would be counted twice) is pooled instead of refused; nothing may be written."""
    w = _two_tables(tmp_path)
    base = ["table", "--rows", str(w["rows_b"]), "--adsb-dir", str(w["adsb_b"]), "--parked", str(w["pkb"])]

    no_report = tmp_path / "no_report"
    no_report.mkdir()
    (no_report / "table.parquet").write_bytes((w["tba"] / "table.parquet").read_bytes())
    out1 = tmp_path / "o1"
    assert B.main(base + ["--out", str(out1), "--alias-from", str(no_report)]) == 1
    assert "no report.json" in capsys.readouterr().err and not out1.exists()

    other = tmp_path / "other_params"
    other.mkdir()
    (other / "table.parquet").write_bytes((w["tba"] / "table.parquet").read_bytes())
    rep = json.loads((w["tba"] / "report.json").read_text())
    rep["params_hash"] = "9b2caaacbb52"                              # the parameter set before AMEND-A3
    (other / "report.json").write_text(json.dumps(rep))
    out2 = tmp_path / "o2"
    assert B.main(base + ["--out", str(out2), "--alias-from", str(other)]) == 1
    assert "9b2caaacbb52" in capsys.readouterr().err and not out2.exists()

    out3 = tmp_path / "o3"
    same = ["table", "--rows", str(w["rows_a"]), "--adsb-dir", str(w["adsb_a"]), "--parked", str(w["pka"])]
    assert B.main(same + ["--out", str(out3), "--alias-from", str(w["tba"])]) == 1
    assert "20 MVT_ID_mvt" in capsys.readouterr().err and not out3.exists()

    out4 = tmp_path / "o4"
    assert B.main(base + ["--out", str(out4), "--alias-from", str(w["tba"]), str(w["tba"])]) == 1
    assert "20 MVT_ID_mvt" in capsys.readouterr().err and not out4.exists()


# ================================================================== AMEND-A3: the stand finds the take-off

def _late_clock_world(tmp_path):
    """An airport whose movement clock runs 45 s after the ADS-B take-off (as EHAM's does by about 20 s). Four
    matched, callsign-joined flights from stand S1 (ids 110, 120, 130 and 111) give the reference and the offset
    window [-50, 30]. Unmatched row 500 (stand S1, movement time T) took off 45 s before T; another aircraft of the
    same type, parked 500 m away, took off 10 s after T, others exactly 120 s before T, exactly 120 s after and 121 s
    after. Matched row 140 (stand S1) is joined by the clock only: it is no callsign join, so not in the sample."""
    adsb_dir = tmp_path / "adsb"
    day, rows = [], []
    for i, mid in enumerate((110, 120, 130, 111)):
        tp = D1START + 10_000 + i * 4_000
        day += _departure_track(f"ref{mid}", f"RF{mid}", tp, 11, (0.0, 0.0), tp + 300, tp + 900)
        rows.append(_row(mid, f"RF{mid}", tp + 945, tp + 200, stand="S1"))
    T = D1START + 60_000.0
    own_takeoff = T - 45
    day += _departure_track("own500", "OWNCS", own_takeoff - 900, 11, (0.0, 0.0), own_takeoff - 600, own_takeoff)
    day += _departure_track("near500", "NEARCS", T + 10 - 900, 11, (500.0, 0.0), T + 10 - 600, T + 10)
    day += _departure_track("edge500", "EDGECS", T - 120 - 900, 11, (900.0, 0.0), T - 120 - 600, T - 120)
    day += _departure_track("out500", "OUTCS", T + 121 - 900, 11, (1300.0, 0.0), T + 121 - 600, T + 121)
    day += _departure_track("plus500", "PLUSCS", T + 120 - 900, 11, (1700.0, 0.0), T + 120 - 600, T + 120)
    rows.append(_row(500, "NOPE500", T, None, stand="S1"))
    tp = D1START + 30_000                 # a matched departure joined by the clock only (its callsign is another)
    day += _departure_track("t140", "X140", tp, 11, (0.0, 0.0), tp + 300, tp + 900)
    rows.append(_row(140, "M140", tp + 920, tp + 200, stand="S1"))
    _write_adsb_day(adsb_dir, DAY1, day)
    _manifest(adsb_dir)
    rows_path = _write_parquet(tmp_path / "rows.parquet", _rows_df(rows))
    return {"adsb_dir": adsb_dir, "rows_path": rows_path, "T": T}


def test_build_raw_gathers_every_takeoff_within_the_bracket_for_the_rows_that_need_them(tmp_path):
    """Candidates are gathered for an unmatched row with a stand reference and for the 1-in-10 sample of matched
    callsign joins (ids divisible by 10), never for other rows; the bracket is +-120 s, inclusive; `own` marks the
    take-off the shipped join gave the row; the exit is measured at the ROW's stand. Fails when a boundary candidate
    is lost, a row outside the two groups gets candidates, or a candidate's exit is taken at another stand."""
    w = _late_clock_world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    parked, _ = B.build_parked(rows, [w["adsb_dir"]])
    raw, cands, bits = B.build_raw(rows, [w["adsb_dir"]], parked)
    assert list(raw.columns) == B.RAW_COLS and list(cands.columns) == B.CAND_COLS and bits == {"missing_days": []}
    assert sorted(set(cands.MVT_ID_mvt)) == [110.0, 120.0, 130.0, 500.0]          # 111 is not in the sample
    c = cands[cands.MVT_ID_mvt == 500.0]
    assert c.adsb_icao.tolist() == ["edge500", "own500", "near500", "plus500"]      # ordered by take-off time
    assert c.dt_takeoff.tolist() == [-120.0, -45.0, 10.0, 120.0]                    # both bracket ends are included
    assert c.own.tolist() == [False, False, True, False]                           # the +-30 s rule took "near500"
    assert c.exit_ok.tolist() == [False, True, False, False]
    assert c.exit_rel_s.iloc[1] == 645.0 and c.exit_rel_s.isna().tolist() == [True, False, True, True]
    assert (c.adsb_type == "A320").all() and (c.ADEP_mvt == "EHAM").all()
    for mid in (110.0, 120.0, 130.0):
        k = cands[cands.MVT_ID_mvt == mid]
        assert len(k) == 1 and bool(k.own.iloc[0]) and bool(k.exit_ok.iloc[0]) and k.dt_takeoff.iloc[0] == -45.0
    r = raw.set_index("MVT_ID_mvt")
    assert r.loc[500.0, "join_type"] == "time" and r.loc[500.0, "adsb_icao"] == "near500"
    assert r.loc[140.0, "join_type"] == "time" and not bool(r.loc[140.0, "unmatched"])


def test_build_raw_marks_as_own_only_the_takeoff_the_row_was_joined_to(tmp_path):
    """One aircraft with two take-offs 129 s apart (the shipped detector merges only those within 120 s): the row is
    joined to the later one, and only that candidate is its own. The track's two airborne samples between them are
    not ground samples. Fails when `own` is decided by the aircraft alone, or when airborne samples are counted
    among a track's ground samples."""
    adsb_dir = tmp_path / "adsb"
    day, rows = [], []
    for i in range(3):
        tp = D1START + 10_000 + i * 4_000
        day += _departure_track(f"ref{i}", f"RF{i}", tp, 11, (0.0, 0.0), tp + 300, tp + 900)
        rows.append(_row(101 + i, f"RF{i}", tp + 900, tp + 200, stand="S1"))
    T = D1START + 60_000.0
    pts = [(T - 900 + 30 * k, 0.0, 0.0, 1, 0.0) for k in range(11)]                 # parked until T - 600
    pts += [(T - 600, 50.0, 0.0, 1, 10.0), (T - 330, 500.0, 0.0, 1, 15.0), (T - 124, 3000.0, 0.0, 1, 140.0),
            (T - 119, 3100.0, 0.0, 0, 160.0), (T - 100, 3500.0, 0.0, 0, 170.0),      # first take-off, airborne twice
            (T - 60, 3000.0, 0.0, 1, 30.0), (T + 5, 3000.0, 0.0, 1, 140.0),          # on the ground again
            (T + 10, 3100.0, 0.0, 0, 160.0), (T + 30, 3500.0, 0.0, 0, 180.0)]        # second take-off
    day += _samples("flk", "FLK1", pts)
    rows.append(_row(150, "FLK1", T, T - 700, stand="S1"))
    _write_adsb_day(adsb_dir, DAY1, day)
    _manifest(adsb_dir)
    r = B.read_rows([_write_parquet(tmp_path / "rows.parquet", _rows_df(rows))])
    parked, _ = B.build_parked(r, [adsb_dir])
    raw, cands, _ = B.build_raw(r, [adsb_dir], parked)
    row = raw.set_index("MVT_ID_mvt").loc[150.0]
    assert row.join_type == "callsign" and row.t_takeoff == T + 10 and row.n_ground == 16.0
    c = cands[cands.MVT_ID_mvt == 150.0]
    assert c.adsb_icao.tolist() == ["flk", "flk"] and c.dt_takeoff.tolist() == [-119.0, 10.0]
    assert c.own.tolist() == [False, True]
    assert c.n_ground.tolist() == [14.0, 16.0]          # ground samples before each take-off; never the airborne ones


def test_build_table_finds_the_own_takeoff_where_the_clock_rule_took_another_aircraft(tmp_path,
                                                                                      few_joins_make_a_window):
    """The registered rule joined row 500 to the only take-off within 30 s of its movement time - another aircraft of
    the same type that never stood at its stand. Fails when the row is not re-joined to the take-off that left its
    own stand 45 s before the movement time (inside this airport's window), or when its estimate is not that track's
    stand exit."""
    w = _late_clock_world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    parked, _ = B.build_parked(rows, [w["adsb_dir"]])
    table, rep = B.build_table(rows, [w["adsb_dir"]], parked)
    t = table.set_index("MVT_ID_mvt")
    assert rep["offset_windows"] == {"EHAM": [-50.0, 30.0]}
    assert (t.loc[500.0, "join_type"], t.loc[500.0, "adsb_icao"], t.loc[500.0, "dt_takeoff"]) == ("stand", "own500", -45.0)
    assert t.loc[500.0, "t_takeoff"] == w["T"] - 45 and t.loc[500.0, "exit_rel_s"] == 645.0
    assert bool(t.loc[500.0, "usable"]) and t.loc[500.0, "cls"] == "A1" and bool(t.loc[500.0, "exit_ok"])
    assert rep["stand_joins"] == {"rows": 1, "one": 1, "several": 0, "none": 0}
    s = rep["stop_rules"]["EHAM"]
    assert s["stand_unmatched_type_agree_share"] == 1.0 and s["n_stand_unmatched"] == 1
    # stop rule (d): the three sampled matched callsign joins, scored with their callsign join ignored
    assert rep["readouts"]["EHAM"]["stand_rule_check"] == {"n": 3, "right": 3, "wrong": 0, "ambiguous": 0, "none": 0}
    assert s["stand_check_right_share"] == 1.0 and s["stand_check_wrong_share"] == 0.0 and s["n_stand_check"] == 3


def test_a_ground_sample_stamped_with_the_takeoff_time_is_not_before_takeoff(tmp_path):
    """The exit is searched over the ground samples BEFORE the take-off. This track has an on-ground sample at the
    stand position stamped with the very second of its first airborne sample. Fails when that sample is taken as
    the stand exit (the estimate would collapse to the join offset, 0 s here, and the row would lose its class)."""
    adsb_dir = tmp_path / "adsb"
    day, rows = [], []
    for i in range(3):
        tp = D1START + 10_000 + i * 4_000
        day += _departure_track(f"ref{i}", f"RF{i}", tp, 11, (0.0, 0.0), tp + 300, tp + 900)
        rows.append(_row(100 + i, f"RF{i}", tp + 900, tp + 200, stand="S1"))
    tp = D1START + 40_000
    trk = _departure_track("odd", "ODD1", tp, 11, (0.0, 0.0), tp + 300, tp + 900)
    k = next(i for i, r in enumerate(trk) if r["on_ground"] == 0)            # the first airborne sample (t = tp + 900)
    trk.insert(k, _samples("odd", "ODD1", [(tp + 900, 0.0, 0.0, 1, 0.0)])[0])
    day += trk
    rows.append(_row(200, "ODD1", tp + 900, tp + 200, stand="S1"))
    _write_adsb_day(adsb_dir, DAY1, day)
    _manifest(adsb_dir)
    r = B.read_rows([_write_parquet(tmp_path / "rows.parquet", _rows_df(rows))])
    parked, _ = B.build_parked(r, [adsb_dir])
    table, _ = B.build_table(r, [adsb_dir], parked)
    t = table.set_index("MVT_ID_mvt")
    assert t.loc[200.0, "join_type"] == "callsign" and t.loc[200.0, "t_takeoff"] == tp + 900
    assert t.loc[200.0, "exit_rel_s"] == 600.0 and t.loc[200.0, "cls"] == "A1"


def test_build_table_without_enough_callsign_joins_keeps_the_shipped_join(tmp_path):
    """The same world under the registered bound of 200 callsign joins: no window, so row 500 keeps the clock rule's
    take-off and - an unmatched row's time join giving no class - has none. Fails when a stand join is made without
    an offset window."""
    w = _late_clock_world(tmp_path)
    rows = B.read_rows([w["rows_path"]])
    parked, _ = B.build_parked(rows, [w["adsb_dir"]])
    table, rep = B.build_table(rows, [w["adsb_dir"]], parked)
    t = table.set_index("MVT_ID_mvt")
    assert rep["offset_windows"] == {} and rep["stand_joins"] == {"rows": 1, "one": 0, "several": 0, "none": 1}
    assert (t.loc[500.0, "join_type"], t.loc[500.0, "adsb_icao"], t.loc[500.0, "cls"]) == ("time", "near500", "")
    s = rep["stop_rules"]["EHAM"]
    assert np.isnan(s["stand_unmatched_type_agree_share"]) and s["n_stand_unmatched"] == 0
    assert np.isnan(s["stand_check_right_share"]) and s["n_stand_check"] == 0


def test_cli_table_writes_the_candidates_beside_the_table(tmp_path, few_joins_make_a_window):
    """Fails when cands.parquet is missing, is not in CAND_COLS order sorted by row and take-off time, or differs
    between --jobs 1 and --jobs 2."""
    w = _late_clock_world(tmp_path)
    pk = tmp_path / "pk"
    assert B.main(["parked", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--out", str(pk)]) == 0
    out = tmp_path / "tb"
    assert B.main(["table", "--rows", str(w["rows_path"]), "--adsb-dir", str(w["adsb_dir"]), "--parked", str(pk),
                   "--out", str(out)]) == 0
    c = pd.read_parquet(out / "cands.parquet")
    assert list(c.columns) == B.CAND_COLS
    assert c[["MVT_ID_mvt", "t_takeoff"]].values.tolist() == sorted(c[["MVT_ID_mvt", "t_takeoff"]].values.tolist())
    t = pd.read_parquet(out / "table.parquet").set_index("MVT_ID_mvt")
    assert t.loc[500.0, "join_type"] == "stand"
    rep = json.loads((out / "report.json").read_text())
    assert rep["offset_windows"] == {"EHAM": [-50.0, 30.0]} and rep["stand_joins"]["one"] == 1


# ================================================================== real-data smoke (devtree + slow)

EHAM_FEB = ROOT / "data" / "adsb" / "v2" / "y2025_other"
FEB_RAW = ROOT / "data" / "raw" / "training_2025-02-01_2025-03-01.parquet"
REAL_FILES_PRESENT = (EHAM_FEB / "adsb_2025-02-09.parquet").exists() and \
                     (EHAM_FEB / "adsb_2025-02-10.parquet").exists() and \
                     (EHAM_FEB / "adsb_2025-02-11.parquet").exists() and FEB_RAW.exists()


@pytest.mark.devtree
@pytest.mark.slow
@ISO.isolated
def test_real_data_smoke_eham_2025_02_10():
    """Real data, read-only, at most 3 ADS-B day files (2025-02-09/10/11) and one raw training file. Skips (not
    fails) when the files are absent. The floors are smoke floors for ONE day (measured 2026-10-04: callsign share
    0.600, 70 parked positions, 3 classed matched departures): most aircraft are first seen after off-block, and a
    stand needs three parked flights for a reference, so a single day forms few references; the registered build
    pools a whole year. Fails when the join, the parked-position search or the stand-exit rule stops producing
    anything on real data."""
    if not REAL_FILES_PRESENT:
        pytest.skip("real ADS-B / raw files absent (data/ is not in git)")
    rows = B.read_rows([FEB_RAW])
    rows = rows[(rows.ADEP_mvt == "EHAM") & (rows.day == "2025-02-10")].reset_index(drop=True)
    adsb = B.read_day([EHAM_FEB], "2025-02-10", "EHAM")
    assert adsb is not None
    j, tr = B.join_day(rows, adsb)
    matched = j[j.AOBT_3_flt.notna()]
    callsign_share = (matched.join_type == "callsign").mean()
    assert callsign_share >= 0.60

    parked, _ = B.build_parked(rows, [EHAM_FEB], days=["2025-02-10"], airports=["EHAM"])
    table, _ = B.build_table(rows, [EHAM_FEB], parked, days=["2025-02-10"], airports=["EHAM"])
    assert len(parked) >= 30
    matched_cls = table[(~table.unmatched) & (table.cls.isin(["A1", "A2"]))]
    assert len(matched_cls) >= 1
    print(f"uad_build smoke EHAM 2025-02-10: callsign_share={callsign_share:.3f} parked={len(parked)} "
         f"matched_A1_A2={len(matched_cls)} n_dep={len(table)}")
