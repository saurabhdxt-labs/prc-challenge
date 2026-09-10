"""Amendment 24: the weather block (stand_ab.load_weather / build_weather / weather-cache).

The invariant that matters most is the one a leak would break: the observation joined to a row is
the last one valid AT OR BEFORE the row's own pushback anchor, never a later one, and the anchor
is a schedule-side quantity - so nothing downstream of the hidden BLOCK_TIME can reach a feature.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MARCH = RAW / "training_2025-03-01_2025-04-01.parquet"
RANKING = RAW / "ranking.parquet"
WEATHER_RAW = ROOT / "data" / "weather"

_spec = importlib.util.spec_from_file_location("stand_ab", ROOT / "scripts" / "stand_ab.py")
stand_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stand_ab)

needs_data = pytest.mark.skipif(not (MARCH.exists() and RANKING.exists()),
                                reason="challenge data is not redistributable and is absent from a clean checkout")
needs_weather = pytest.mark.skipif(not (WEATHER_RAW.exists() and len(list(WEATHER_RAW.glob("*.csv"))) >= 140),
                                   reason="the weather archive is built by scripts/fetch_weather.py and is not in the repo")

#: the contract, spelled out rather than read back from the module
WEATHER_FEATS = ["w_temp_c", "w_dewspread_c", "w_wind_kt", "w_gust_kt", "w_vis_km", "w_precip_mm",
                 "w_freezing", "w_lowvis", "w_thunder", "w_age_s"]
WCOLS = ["PHASE_mvt", "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
         "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt"]
T0 = pd.Timestamp("2025-03-10 06:00:00", tz="UTC")


def archive_csv(rows) -> str:
    head = "station,valid,tmpf,dwpf,sknt,gust,vsby,p01i,wxcodes,metar"
    return "\n".join([head] + [",".join(str(v) for v in r) for r in rows]) + "\n"


def micro_archive(tmp_path) -> pathlib.Path:
    """EGLL: four observations an hour apart, each a different regime; LEMD: two."""
    d = tmp_path / "weather"; d.mkdir()
    (d / "EGLL_2025-03.csv").write_text(archive_csv([
        # 05:00 mild and clear                       32F = 0C
        ("EGLL", "2025-03-10 05:00", 50.00, 41.00, 10.00, "M", 6.21, 0.00, "M", "x"),
        # 06:00 freezing WITH precipitation -> w_freezing 1
        ("EGLL", "2025-03-10 06:00", 32.00, 32.00, 20.00, 35.00, 0.62, 0.04, "-SN", "x"),
        # 07:00 cold and dry, no code -> NOT freezing (no precipitation, no code)
        ("EGLL", "2025-03-10 07:00", 33.80, 30.20, 5.00, "M", 5.00, 0.00, "M", "x"),
        # 08:00 thunderstorm, trace precipitation
        ("EGLL", "2025-03-10 08:00", 68.00, 59.00, 12.00, "M", 3.00, "T", "TSRA", "x"),
        # 09:00 and 10:00 STRADDLE the 1.5 km edge by one float step through the real parse path
        # (string -> to_numeric -> x 1.609344): 0.9320567883560009 mi is the LARGEST visibility
        # whose km value is below 1.5 and 0.932056788356001 the SMALLEST at or above it. No decimal
        # literal lands exactly on 1.5 km through that path, so strictness is pinned by the pair.
        ("EGLL", "2025-03-10 09:00", 50.00, 41.00, 10.00, "M", "0.9320567883560009", 0.00, "M", "x"),
        ("EGLL", "2025-03-10 10:00", 50.00, 41.00, 10.00, "M", "0.932056788356001", 0.00, "M", "x"),
    ]))
    (d / "LEMD_2025-03.csv").write_text(archive_csv([
        ("LEMD", "2025-03-10 05:30", 59.00, 41.00, 8.00, "M", 6.21, 0.00, "M", "x"),
        ("LEMD", "2025-03-10 07:30", 60.80, 42.80, 9.00, "M", 6.21, 0.00, "M", "x"),
    ]))
    return d


def movements(anchor_offsets, blank_departure_clocks: bool = False, drop_aobt: bool = False) -> pd.DataFrame:
    """One EGLL departure per offset (seconds after T0 for its AOBT_3), plus two LEMD departures."""
    rows = []
    for i, off in enumerate(anchor_offsets):
        a = T0 + pd.Timedelta(seconds=int(off))
        rows.append(dict(PHASE_mvt="DEP", MVT_ID_mvt=float(100 + i), ADEP_mvt="EGLL", ADES_mvt="LEMD",
                         MVT_TIME_UTC_mvt=a + pd.Timedelta(minutes=15),
                         SCHED_TIME_UTC_mvt=a - pd.Timedelta(minutes=10),
                         BLOCK_TIME_UTC_mvt=a - pd.Timedelta(minutes=1),
                         TAXITIME_SEC_mvt=960.0, AOBT_3_flt=pd.NaT if drop_aobt else a))
    for k, off in enumerate((-1800, 5400)):
        a = T0 + pd.Timedelta(seconds=int(off))
        rows.append(dict(PHASE_mvt="DEP", MVT_ID_mvt=float(900 + k), ADEP_mvt="LEMD", ADES_mvt="EGLL",
                         MVT_TIME_UTC_mvt=a + pd.Timedelta(minutes=12),
                         SCHED_TIME_UTC_mvt=a - pd.Timedelta(minutes=5),
                         BLOCK_TIME_UTC_mvt=a - pd.Timedelta(minutes=2),
                         TAXITIME_SEC_mvt=720.0, AOBT_3_flt=a))
    t = pd.DataFrame(rows)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"):
        t[c] = pd.to_datetime(t[c], utc=True)
    if blank_departure_clocks:
        t["BLOCK_TIME_UTC_mvt"] = pd.NaT
        t["TAXITIME_SEC_mvt"] = np.nan
    return t[WCOLS]


def test_weather_contract_and_constants():
    """The module's list and raw-column set are the contract, in this order, and the physical
    thresholds are the registered ones (24.3). Fails on any rename or silent threshold change."""
    assert list(stand_ab.WEATHER_FEATS) == WEATHER_FEATS
    assert list(stand_ab.WCOLS) == WCOLS
    assert stand_ab.WX_FREEZE_C == 3.0 and stand_ab.WX_LOWVIS_KM == 1.5 and stand_ab.WX_TRACE_MM == 0.05
    assert "FZ" in stand_ab.WX_FREEZE_CODES and "SN" in stand_ab.WX_FREEZE_CODES
    assert stand_ab.WCACHE == ROOT / "data" / "cache_weather"


def test_load_weather_converts_units_and_derives_the_flags(tmp_path):
    """Fahrenheit -> Celsius, statute miles -> km, inches -> mm, 'M' -> NaN, 'T' -> 0.05 mm, and a
    MISSING GUST is 0 kt (a METAR omits the group when there is no gust) while a missing wind is
    NaN. w_freezing needs cold AND (precipitation OR a frozen code) - cold and dry is not freezing;
    w_lowvis is strictly below 1.5 km; w_thunder is a TS code.

    Fails when a conversion is dropped or a flag's AND becomes an OR. Rehearsed 2026-09-09, each
    RED: `(tmpf - 32)` without the 5/9; `w_gust_kt` left NaN; `|` for the freezing AND;
    `<=` for the low-visibility edge.
    """
    w = stand_ab.load_weather(micro_archive(tmp_path))
    assert list(w.station.unique()) == ["EGLL", "LEMD"] or set(w.station) == {"EGLL", "LEMD"}
    e = w[w.station == "EGLL"].sort_values("valid").reset_index(drop=True)
    assert len(e) == 6 and str(e.valid.dt.tz) == "UTC"
    assert e.w_temp_c.tolist() == pytest.approx([10.0, 0.0, 1.0, 20.0, 10.0, 10.0], abs=1e-9)
    assert e.w_dewspread_c.tolist() == pytest.approx([5.0, 0.0, 2.0, 5.0, 5.0, 5.0], abs=1e-9)
    assert e.w_vis_km.tolist()[:4] == pytest.approx([6.21 * 1.609344, 0.62 * 1.609344, 5.0 * 1.609344,
                                                     3.0 * 1.609344], abs=1e-9)
    assert e.w_precip_mm.tolist() == pytest.approx([0.0, 0.04 * 25.4, 0.0, 0.05 * 25.4, 0.0, 0.0], abs=1e-9)
    assert e.w_gust_kt.tolist() == [0.0, 35.0, 0.0, 0.0, 0.0, 0.0], "a missing gust is no gust, not unknown"
    assert e.w_wind_kt.tolist() == [10.0, 20.0, 5.0, 12.0, 10.0, 10.0]
    assert e.w_freezing.tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], "cold AND wet/frozen; cold and dry is not"
    below, above = e.w_vis_km.iloc[4], e.w_vis_km.iloc[5]
    assert below < 1.5 < above and float(np.nextafter(below, 2.0)) >= 1.5, \
        "the pair must straddle the threshold by one float step, or it pins nothing"
    assert e.w_lowvis.tolist() == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0], \
        "one float step below the bound is low visibility, one step above is not"
    assert e.w_thunder.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    with pytest.raises(FileNotFoundError, match="weather archive"):
        stand_ab.load_weather(tmp_path / "nowhere")


def test_the_join_never_reaches_forward_in_time(tmp_path):
    """THE leak test. A row anchored at 06:59 must take the 06:00 observation, not the 07:00 one
    that is 60 s away; a row anchored exactly ON an observation takes that observation (age 0); a
    row anchored before the first observation of its airport gets NaN, not the first one. Each
    row's airport is respected: an EGLL row never takes a LEMD observation.

    Fails when the merge direction flips to 'nearest' or 'forward' - the shape that would let a
    reading taken AFTER the hidden off-block reach a feature. Rehearsed 2026-09-09, each RED:
    `direction="nearest"` (the 06:59 row jumps to 07:00); `direction="forward"`; the `left_by /
    right_by` station keys removed (LEMD's 05:30 observation reaches an EGLL row).
    """
    w = stand_ab.load_weather(micro_archive(tmp_path))
    # offsets from 06:00: -3660 = 04:59 (before the first), -60 = 05:59, 0 = 06:00, 3540 = 06:59, 7200 = 08:00
    t = movements([-3660, -60, 0, 3540, 7200])
    o = stand_ab.build_weather(t, serve=False, w=w)
    assert list(o.columns) == ["MVT_ID_mvt"] + WEATHER_FEATS
    assert all(o[c].dtype == np.float32 for c in WEATHER_FEATS)
    e = o[o.MVT_ID_mvt < 900].set_index("MVT_ID_mvt")
    assert np.isnan(e.loc[100.0, "w_temp_c"]), "a row before the first observation must be NaN, not filled forward"
    assert e.loc[101.0, "w_temp_c"] == pytest.approx(10.0) and e.loc[101.0, "w_age_s"] == pytest.approx(3540.0)
    assert e.loc[102.0, "w_temp_c"] == pytest.approx(0.0) and e.loc[102.0, "w_age_s"] == pytest.approx(0.0)
    assert e.loc[103.0, "w_temp_c"] == pytest.approx(0.0) and e.loc[103.0, "w_age_s"] == pytest.approx(3540.0), \
        "06:59 must take 06:00, never the 07:00 reading 60 s later"
    assert e.loc[104.0, "w_thunder"] == 1.0 and e.loc[104.0, "w_age_s"] == pytest.approx(0.0)
    lemd = o[o.MVT_ID_mvt >= 900].set_index("MVT_ID_mvt")
    assert lemd.loc[900.0, "w_temp_c"] == pytest.approx(15.0), "05:30 LEMD, not any EGLL observation"
    assert lemd.loc[901.0, "w_temp_c"] == pytest.approx(16.0)


def test_the_anchor_is_aobt_or_the_airport_median_proxy_never_the_hidden_block(tmp_path):
    """With AOBT_3 the anchor is AOBT_3; without it the anchor is MVT_TIME minus the AIRPORT's
    median proxy over the rows that do have one (24.3). Blanking every departure's BLOCK and
    TAXITIME and rebuilding in serve mode changes nothing for the rows present in both builds -
    no feature reads the hidden clocks.

    Fails when the anchor is taken from BLOCK_TIME or from MVT_TIME directly. Rehearsed
    2026-09-09, each RED: `anchor = es(d.BLOCK_TIME_UTC_mvt)`; the median-proxy fallback replaced
    by `tk` (the 15-minute shift moves the 06:59 row onto the 07:00 observation).
    """
    w = stand_ab.load_weather(micro_archive(tmp_path))
    a = stand_ab.build_weather(movements([-60, 0, 3540]), serve=False, w=w)
    b = stand_ab.build_weather(movements([-60, 0, 3540], blank_departure_clocks=True), serve=True, w=w)
    assert a.MVT_ID_mvt.tolist() == b.MVT_ID_mvt.tolist()
    for c in WEATHER_FEATS:
        assert np.array_equal(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True), c
    # 24.3 also registered a median-proxy fallback for rows without an AOBT_3. It is unreachable:
    # _dep_stream requires AOBT_3 in BOTH modes, so a row without one never enters the block at
    # all - it is dropped upstream, not anchored by a fallback. Pinned here so a future change to
    # the stream's filter (which WOULD make the fallback reachable) fails this test rather than
    # silently anchoring rows on a guess.
    t = movements([3540], drop_aobt=True)
    for serve in (False, True):
        o = stand_ab.build_weather(t, serve=serve, w=w)
        assert o.MVT_ID_mvt.tolist() == [900.0, 901.0], "an AOBT_3-less row must not reach the block"
        assert o.w_temp_c.notna().all()
    assert stand_ab._dep_stream(t, False).AOBT_3_flt.notna().all()
    assert stand_ab._dep_stream(t, True).AOBT_3_flt.notna().all()


def test_build_weather_refuses_an_empty_departure_stream(tmp_path):
    """No admissible departure -> ValueError naming the mode. Fails when the guard is dropped."""
    w = stand_ab.load_weather(micro_archive(tmp_path))
    t = movements([0])
    with pytest.raises(ValueError, match="no admissible departures"):
        stand_ab.build_weather(t[t.PHASE_mvt == "ARR"], serve=False, w=w)


def _write(frame: pd.DataFrame, path: pathlib.Path) -> pathlib.Path:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return path


def test_cmd_weather_cache_writes_the_contract_and_refuses_a_smoke_ranking(tmp_path, monkeypatch):
    """`weather-cache --smoke` writes exactly one file with MVT_ID_mvt + WEATHER_FEATS (float32)
    equal to build_weather_month; `--ranking --smoke` is refused before anything is written;
    `--ranking` writes only ranking.parquet in serve mode. Nothing lands outside the cache dir."""
    raw, cache = tmp_path / "raw", tmp_path / "cache_weather"
    raw.mkdir()
    month = _write(movements([-60, 0, 3540]), raw / "training_2025-03-01_2025-04-01.parquet")
    _write(movements([0]), raw / "training_2025-04-01_2025-05-01.parquet")
    _write(movements([-60, 0, 3540], blank_departure_clocks=True), raw / "ranking.parquet")
    monkeypatch.setattr(stand_ab, "RAW", raw)
    monkeypatch.setattr(stand_ab, "WCACHE", cache)
    monkeypatch.setattr(stand_ab, "WEATHER_RAW", micro_archive(tmp_path))
    monkeypatch.setattr(stand_ab, "CACHE", tmp_path / "must_not_be_touched")
    stand_ab.cmd_weather_cache(smoke=True, ranking=False)
    assert sorted(p.name for p in cache.iterdir()) == ["training_2025-03-01_2025-04-01.parquet"]
    o = pd.read_parquet(cache / "training_2025-03-01_2025-04-01.parquet")
    want = stand_ab.build_weather_month(month, w=stand_ab.load_weather(micro_archive(tmp_path) if False else stand_ab.WEATHER_RAW))
    assert list(o.columns) == ["MVT_ID_mvt"] + WEATHER_FEATS
    assert o.MVT_ID_mvt.tolist() == want.MVT_ID_mvt.tolist()
    assert all(np.array_equal(o[c].to_numpy(), want[c].to_numpy(), equal_nan=True) for c in WEATHER_FEATS)
    with pytest.raises(ValueError, match="smoke"):
        stand_ab.cmd_weather_cache(smoke=True, ranking=True)
    assert sorted(p.name for p in cache.iterdir()) == ["training_2025-03-01_2025-04-01.parquet"]
    stand_ab.cmd_weather_cache(smoke=False, ranking=True)
    assert (cache / "ranking.parquet").exists()
    assert not (tmp_path / "must_not_be_touched").exists()


@needs_data
@needs_weather
def test_real_march_rows_are_the_reference_stream_and_the_observations_are_fresh():
    """Real March against the frozen archive: the rows are build_month's rows in its order; every
    row gets an observation (the archive spans the month); the median observation age is under an
    hour and the 99th percentile under three (48 reports a day); temperatures are physical; the
    serve build on the same file with the departures' clocks blanked reproduces every shared row.
    """
    t = pq.read_table(MARCH, columns=WCOLS).to_pandas()
    o = stand_ab.build_weather(t, serve=False)
    ref = stand_ab._dep_stream(t, False)
    assert o.MVT_ID_mvt.tolist() == ref.MVT_ID_mvt.tolist()
    assert o.w_age_s.notna().mean() > 0.99, "the frozen archive must cover the training months"
    assert o.w_age_s.median() < 3_600 and o.w_age_s.quantile(0.99) < 3 * 3_600
    assert -40 < o.w_temp_c.min() and o.w_temp_c.max() < 55
    assert o.w_vis_km.max() < 100 and (o.w_precip_mm >= 0).all()
    s = t.copy()
    dep = (s.PHASE_mvt == "DEP").to_numpy()
    s.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    s.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    so = stand_ab.build_weather(s, serve=True).set_index("MVT_ID_mvt")
    for c in WEATHER_FEATS:
        assert np.array_equal(o[c].to_numpy(), so.loc[o.MVT_ID_mvt.to_numpy(), c].to_numpy(), equal_nan=True), c


@needs_data
@needs_weather
def test_the_scored_file_is_covered_and_january_is_the_freezing_month():
    """The serve build on the real scored file gives one row per matched scored departure with an
    observation, and the mechanism the amendment named is present in the data: January carries
    materially more freezing hours than July at the northern airports."""
    r = stand_ab.build_weather_ranking(RANKING)
    stand_rank = pd.read_parquet(stand_ab.CACHE / "ranking.parquet", columns=["MVT_ID_mvt"])
    assert r.MVT_ID_mvt.is_unique and set(stand_rank.MVT_ID_mvt) <= set(r.MVT_ID_mvt)
    assert r.w_age_s.notna().mean() > 0.99
    w = stand_ab.load_weather()
    jan = w[w.valid.dt.month == 1]; jul = w[w.valid.dt.month == 7]
    assert jan.w_freezing.mean() > jul.w_freezing.mean(), "January must be the freezing season"
