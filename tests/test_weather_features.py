"""Amendment 24: the weather block (stand_ab.load_weather / build_weather / weather-cache).

The invariant that matters most is the one a leak would break: the observation joined to a row is
the last one valid AT OR BEFORE the row's own pushback anchor, never a later one, and the anchor
is a schedule-side quantity - so nothing downstream of the hidden BLOCK_TIME can reach a feature.

Post-mortem 2026-09-10/11 (bug class BC-3): the block now reads through prc.weather. Mutation
rehearsal 2026-09-11 between 01:21:39 and 01:48:18 EDT (two `date` reads) on load_weather, each applied alone, each RED by the named test:
w_precip_int zeroed as in v1 and the intensity offset shifted [converts_units]; precipitation
dropped from w_freezing and its bound moved to 0 C [freezing_flag_follows]; w_lowvis and
w_freezing turning an unknown into 0 [an_unknown_input]; vicinity thunder dropped
[tests/test_sentinel_values.py::test_thunder]; the as-of join turned forward [never_reaches_forward].
"""
from __future__ import annotations

import importlib.util
import json
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
WEATHER_FEATS = ["w_temp_c", "w_dewspread_c", "w_wind_kt", "w_gust_kt", "w_vis_km", "w_precip_int",
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
    thresholds are the registered ones (24.3). Fails on any rename or silent threshold change.
    2026-09-11: w_precip_mm became w_precip_int (post-mortem BC-3) and the parse moved to
    prc.weather, whose frozen-code list is the registered one (FZ as the descriptor)."""
    assert list(stand_ab.WEATHER_FEATS) == WEATHER_FEATS
    assert list(stand_ab.WCOLS) == WCOLS
    assert stand_ab.WX_FREEZE_C == 3.0 and stand_ab.WX_LOWVIS_KM == 1.5
    assert stand_ab.WX.__name__ == "prc.weather" and stand_ab.WX.TRACE_MM == 0.05
    assert set(stand_ab.WX.FROZEN_PHENOMENA) == {"SN", "PL", "GS", "GR", "IC"}
    assert stand_ab.WCACHE == ROOT / "data" / "cache_weather"


def test_load_weather_converts_units_and_derives_the_flags(tmp_path):
    """Fahrenheit -> Celsius, statute miles -> km, 'M' -> NaN, precipitation from the weather
    group (-SN light = 1, TSRA moderate = 2), and a MISSING GUST with a reported wind is 0 kt (a
    METAR omits the group when there is no gust) while a missing wind is NaN. (p01i and the trace
    are pinned in tests/test_weather_parser.py since 2026-09-11.) w_freezing needs cold AND (precipitation OR a frozen code) - cold and dry is not freezing;
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
    assert e.w_precip_int.tolist() == [0.0, 1.0, 0.0, 2.0, 0.0, 0.0]
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
    assert sorted(p.name for p in cache.iterdir()) == ["manifest.json", "training_2025-03-01_2025-04-01.parquet"]
    o = pd.read_parquet(cache / "training_2025-03-01_2025-04-01.parquet")
    want = stand_ab.build_weather_month(month, w=stand_ab.load_weather(micro_archive(tmp_path) if False else stand_ab.WEATHER_RAW))
    assert list(o.columns) == ["MVT_ID_mvt"] + WEATHER_FEATS
    assert o.MVT_ID_mvt.tolist() == want.MVT_ID_mvt.tolist()
    assert all(np.array_equal(o[c].to_numpy(), want[c].to_numpy(), equal_nan=True) for c in WEATHER_FEATS)
    with pytest.raises(ValueError, match="smoke"):
        stand_ab.cmd_weather_cache(smoke=True, ranking=True)
    assert sorted(p.name for p in cache.iterdir()) == ["manifest.json", "training_2025-03-01_2025-04-01.parquet"]
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
    assert o.w_vis_km.max() < 100 and o.w_precip_int.dropna().isin([0.0, 1.0, 2.0, 3.0]).all()
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


# ---- post-mortem 2026-09-10: the archive's p01i is a fabricated zero at every European station ----
#: present-weather codes that report precipitation falling at the station (VC = in the vicinity, not
#: at the station, and BR/FG/HZ are obscurations, not precipitation)
PRECIP_CODES = ("RA", "DZ", "SN", "SG", "PL", "GS", "GR", "IC", "UP")


def _precip_at_station(codes: pd.Series) -> np.ndarray:
    """An independent reading: some space-separated token that is not a VC (vicinity) token and not
    blowing/drifting snow contains a precipitation code. Token by token, so '-RA VCTS' is rain at
    the station (the first draft excluded any row containing 'VC', which let that case through)."""
    def wet(s: str) -> bool:
        return any(not t.startswith("VC") and t not in ("BLSN", "DRSN") and any(k in t for k in PRECIP_CODES)
                   for t in s.split())
    return codes.fillna("").astype(str).str.upper().map(wet).to_numpy(dtype=bool)


def european_archive(tmp_path) -> pathlib.Path:
    """The archive AS THE SERVICE ACTUALLY EMITS IT for a European station: p01i is the literal
    "0.00" on every row, including the rows whose METAR reports rain, drizzle or snow. The original
    unit fixture used US-format p01i values, which is why it never saw this."""
    d = tmp_path / "weather_eu"; d.mkdir()
    (d / "EDDM_2025-01.csv").write_text(archive_csv([
        ("EDDM", "2025-01-10 05:00", 30.20, 28.40, 8.00, "M", 6.21, "0.00", "M", "x"),      # -1 C, dry
        ("EDDM", "2025-01-10 06:00", 35.60, 33.80, 10.00, "M", 2.00, "0.00", "-RA", "x"),  # +2 C, light rain
        ("EDDM", "2025-01-10 07:00", 33.80, 32.00, 12.00, "M", 1.20, "0.00", "DZ", "x"),   # +1 C, drizzle
        ("EDDM", "2025-01-10 08:00", 50.00, 44.60, 9.00, "M", 5.00, "0.00", "RA", "x"),    # +10 C, rain
        ("EDDM", "2025-01-10 09:00", 33.80, 30.20, 7.00, "M", 6.21, "0.00", "VCSH", "x"),  # showers NEARBY
    ]))
    return d


def test_load_weather_never_reports_zero_precipitation_when_the_metar_says_it_is_raining(tmp_path):
    """Post-mortem 2026-09-10. The Iowa Mesonet archive returns p01i = "0.00" (a literal zero, not "M")
    on every European observation, because European METARs carry no US hourly-precipitation group.
    Trusting it made w_precip_mm 0.0 on every row of every cache while the weather group reported
    rain. On a row whose present-weather code reports precipitation AT the station, the block must not
    assert that none fell.

    Written against the v1 column and seen RED on 2026-09-10 (w_precip_mm = p01i x 25.4 = 0.0 on the
    -RA, DZ and RA rows). Since the fix the block carries w_precip_int, read from the weather group:
    light rain 1, drizzle and rain 2 (moderate), and 0 on the dry row and the VICINITY showers.
    """
    w = stand_ab.load_weather(european_archive(tmp_path)).sort_values("valid").reset_index(drop=True)
    assert w.w_precip_int.tolist() == [0.0, 1.0, 2.0, 2.0, 0.0], w.w_precip_int.tolist()


def test_freezing_flag_follows_the_registered_rule_on_cold_rain(tmp_path):
    """Post-mortem 2026-09-10. Amendment 24.3 registered w_freezing = temp <= 3 C AND (precipitation OR
    a FZ/SN/PL/GS code). The rule was right; the implementation fed it a fabricated zero, so it
    collapsed to 'cold AND a frozen code' and missed cold rain and cold drizzle — the conditions under
    which aircraft are de-iced. Light rain at +2 C and drizzle at +1 C must flag; rain at +10 C and a
    dry -1 C hour must not; showers in the VICINITY at +1 C are not precipitation at the station.

    Fails on the pre-fix loader: the -RA (+2 C) and DZ (+1 C) rows carry w_freezing 0.
    """
    w = stand_ab.load_weather(european_archive(tmp_path)).sort_values("valid").reset_index(drop=True)
    assert w.w_freezing.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0], w.w_freezing.tolist()


@needs_weather
def test_real_archive_precipitation_and_freezing_are_consistent_with_the_weather_codes():
    """Post-mortem 2026-09-10, on the real archive — the surface the synthetic fixture never saw. Every
    observation whose code reports precipitation at the station must carry a non-zero precipitation
    value, every one of those at <= 3 C must flag freezing, and a row the codes call dry must not
    report precipitation. Before the fix the first two failed on thousands of rows (18,947 raining
    observations reported exactly 0): p01i is "0.00" on all 205,417 observations.
    """
    w = stand_ab.load_weather()
    raw = pd.concat([pd.read_csv(f, dtype=str, usecols=["station", "valid", "wxcodes"])
                     for f in sorted(stand_ab.WEATHER_RAW.glob("*.csv"))], ignore_index=True)
    assert len(raw) == len(w)
    raw["valid"] = pd.to_datetime(raw.valid, utc=True)
    raw = raw.sort_values("valid", kind="mergesort").reset_index(drop=True)
    assert (raw.valid.to_numpy() == w.valid.to_numpy()).all() and (raw.station.to_numpy() == w.station.to_numpy()).all()
    wet = _precip_at_station(raw.wxcodes)
    assert wet.sum() > 10_000, "the archive must contain the precipitation the codes report"
    pint = w.w_precip_int.to_numpy()
    dry_on_wet = int((pint[wet] < 1.0).sum()) + int(np.isnan(pint[wet]).sum())
    assert dry_on_wet == 0, f"{dry_on_wet:,} raining observations report no (or unknown) precipitation"
    known_dry = ~wet & ~np.isnan(pint)
    assert (pint[known_dry] == 0.0).all(), "a row the codes call dry must not report precipitation"
    cold_wet = wet & (w.w_temp_c.to_numpy() <= stand_ab.WX_FREEZE_C)
    missed = int((w.w_freezing.to_numpy()[cold_wet] != 1.0).sum())
    assert missed == 0, f"{missed:,} of {int(cold_wet.sum()):,} cold, wet observations do not flag freezing"


# ---- post-mortem step 5 (2026-09-11): the error paths the coverage audit found never ran ----------
def test_build_weather_refuses_a_frame_that_did_not_come_from_load_weather(tmp_path):
    """A weather frame without `valid_s` (not built by load_weather, so its time scale is unknown)
    is refused before any join. Uncovered until 2026-09-11."""
    w = stand_ab.load_weather(micro_archive(tmp_path)).drop(columns=["valid_s"])
    with pytest.raises(ValueError, match="must come from load_weather"):
        stand_ab.build_weather(movements([0]), serve=False, w=w)


def test_build_weather_asserts_the_stream_guarantees_an_anchor(tmp_path, monkeypatch):
    """`_dep_stream` guarantees every row an AOBT_3; build_weather re-asserts it rather than
    anchoring a row on nothing. Forced by a stream that breaks the guarantee (the real stream
    cannot; test_the_anchor_is_aobt... pins that). Uncovered until 2026-09-11."""
    w = stand_ab.load_weather(micro_archive(tmp_path))
    real = stand_ab._dep_stream

    def broken(t, serve):
        d = real(t, serve).copy()
        d.loc[d.index[0], "AOBT_3_flt"] = pd.NaT
        return d
    monkeypatch.setattr(stand_ab, "_dep_stream", broken)
    with pytest.raises(ValueError, match="1 rows of the departure stream have no AOBT_3"):
        stand_ab.build_weather(movements([0, 60]), serve=False, w=w)


def test_build_weather_ranking_refuses_rows_without_a_movement_time(tmp_path):
    """A scored row without MVT_TIME cannot be assigned to a calendar month, so the ranking build
    refuses the file rather than dropping the row. Uncovered until 2026-09-11."""
    t = movements([0, 60])
    t.loc[0, "MVT_TIME_UTC_mvt"] = pd.NaT
    path = _write(t, tmp_path / "ranking.parquet")
    with pytest.raises(ValueError, match="1 rows without MVT_TIME"):
        stand_ab.build_weather_ranking(path, w=stand_ab.load_weather(micro_archive(tmp_path)))


def test_an_unknown_input_is_unknown_in_every_derived_flag(tmp_path):
    """Sibling of the post-mortem (BC-3): v1 computed each flag as `(x <= c).astype(float)`, so an
    UNKNOWN temperature read as 'not freezing', an unknown visibility as 'not low', an unobservable
    weather group as 'no thunder, no precipitation' — a fabricated 0 each time. Now each is NaN,
    unless the known part decides it: a WARM row with an unobservable weather group is still
    definitely not freezing (0)."""
    d = tmp_path / "wx_unknown"; d.mkdir()
    (d / "EDDF_2025-01.csv").write_text(archive_csv([
        ("EDDF", "2025-01-10 05:00", "M", 30.0, 8.0, "M", 6.21, "0.00", "-RA", "x"),   # temperature unknown
        ("EDDF", "2025-01-10 06:00", 35.6, 30.0, 8.0, "M", "M", "0.00", "M", "x"),     # visibility unknown
        ("EDDF", "2025-01-10 07:00", 35.6, 30.0, 8.0, "M", 6.21, "0.00", "", "x"),     # weather group unobservable, +2 C
        ("EDDF", "2025-01-10 08:00", 50.0, 30.0, 8.0, "M", 6.21, "0.00", "", "x"),     # unobservable, +10 C
    ]))
    w = stand_ab.load_weather(d).reset_index(drop=True)
    assert np.isnan(w.w_freezing.iloc[0]) and w.w_precip_int.iloc[0] == 1.0
    assert np.isnan(w.w_lowvis.iloc[1]) and w.w_lowvis.iloc[0] == 0.0
    assert np.isnan(w.w_thunder.iloc[2]) and np.isnan(w.w_precip_int.iloc[2]) and np.isnan(w.w_freezing.iloc[2])
    assert w.w_freezing.iloc[3] == 0.0, "warm is not freezing whatever the unobserved weather was"


def test_cmd_weather_cache_never_mixes_months_from_two_parses(tmp_path, monkeypatch):
    """The cache directory is stamped with the parser version, the archive digest and the feature
    list. A build refuses (i) a directory with parquet files and no manifest — a v1 cache — and
    (ii) a directory whose manifest names another parse; nothing is written in either case. A
    rebuild under the same parse is allowed. Rehearsed 2026-09-11: RED with the manifest check
    removed (the build writes v2 months beside the v1 file)."""
    raw, cache = tmp_path / "raw", tmp_path / "cache_weather"
    raw.mkdir(); cache.mkdir()
    _write(movements([-60, 0, 3540]), raw / "training_2025-03-01_2025-04-01.parquet")
    arch = micro_archive(tmp_path)
    monkeypatch.setattr(stand_ab, "RAW", raw)
    monkeypatch.setattr(stand_ab, "WCACHE", cache)
    monkeypatch.setattr(stand_ab, "WEATHER_RAW", arch)
    (cache / "training_2025-01-01_2025-02-01.parquet").write_bytes(b"v1")
    with pytest.raises(ValueError, match="parquet files but no manifest"):
        stand_ab.cmd_weather_cache(smoke=True)
    assert sorted(p.name for p in cache.iterdir()) == ["training_2025-01-01_2025-02-01.parquet"]
    (cache / "training_2025-01-01_2025-02-01.parquet").unlink()
    stand_ab.cmd_weather_cache(smoke=True)
    m = json.loads((cache / "manifest.json").read_text())
    assert m == {"parser_version": stand_ab.WX.VERSION, "archive_digest": stand_ab.WX.archive_digest(arch),
                 "weather_feats": WEATHER_FEATS}
    stand_ab.cmd_weather_cache(smoke=True)                      # same parse: allowed
    (cache / "manifest.json").write_text(json.dumps({**m, "parser_version": "1"}))
    before = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}
    with pytest.raises(ValueError, match="built by a different parse"):
        stand_ab.cmd_weather_cache(smoke=True)
    assert {p.name: p.stat().st_mtime_ns for p in cache.iterdir()} == before
