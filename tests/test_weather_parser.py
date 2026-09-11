"""prc.weather — the one METAR parser (post-mortem 2026-09-10, bug class BC-3, reports/bug_classes.md).

What must never happen again: a provider's placeholder read as a measurement. So the tests pin, on
fixtures shaped like what the archive actually emits for a European station, that (1) precipitation
comes from the present-weather group and a constant p01i is never trusted, (2) an unknown input
yields NaN and never a definite 0, (3) every flag follows the FM 15 grammar token by token, and
(4) the hourly join never reads an observation after the hour's end or from another station.

Mutation rehearsal, 2026-09-11 between 01:21:39 and 01:48:18 EDT (two `date` reads), 23 mutants of prc/weather.py, each applied alone, all RED
(the killing test in brackets): VC branch removed, BL/DR lift ignored, first-token intensity, FZ
dropped from wx_frozen [parse_codes_sets_exactly]; v1 gust nan_to_num [units_and_the_gust_rule];
p01i gate removed, trace converted from inches [a_constant_p01i]; distinctness on raw strings
[distinct_values]; le3 without unknown, or3 as nanmax, and3 needing both zeros, frost spread
bound lifted, mist dropped from moisture [deicing_condition]; `<` in le3 [three_valued_helpers];
exact hour-end excluded, direction forward, direction nearest, station key removed, hour end =
hour start [hourly_uses_only]; bare qualifier accepted [refuses_what_the_grammar]; duplicate key
accepted [read_archive_refuses]; unknown codes read as "M" [unobservable_weather_group]; garbage
coerced to NaN [numeric_fields_refuse_garbage].
Second round, finished 2026-09-11 02:02 EDT (from `date`), all RED: phenomena not split into pairs
[fm15_grammar]; missing-columns and timestamp checks removed [read_archive_refuses]; unversioned
output name [main_writes_versioned]; naive and off-hour keys accepted [hourly_refuses_malformed].
Third round (code review findings), finished 2026-09-11 02:20 EDT (from `date`), all RED: a bare descriptor
accepted and an intensity on a non-precipitation token [refuses_what_the_grammar]; a row without a
station accepted [read_archive_refuses]; a local (non-UTC) hour accepted [hourly_refuses_malformed].
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from prc import weather as W

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEATHER_RAW = ROOT / "data" / "weather"
needs_weather = pytest.mark.skipif(not WEATHER_RAW.exists() or len(list(WEATHER_RAW.glob("*.csv"))) < 140,
                                   reason="the weather archive is built by scripts/fetch_weather.py and is not in the repo")
HEAD = "station,valid,tmpf,dwpf,sknt,gust,vsby,p01i,wxcodes,metar"


def archive(tmp_path, files: dict[str, list[tuple]]) -> pathlib.Path:
    d = tmp_path / "wx"
    d.mkdir(parents=True, exist_ok=True)
    for name, rows in files.items():
        (d / name).write_text("\n".join([HEAD] + [",".join(str(v) for v in r) for r in rows]) + "\n")
    return d


# ------------------------------------------------------------------ the grammar -------------------
@pytest.mark.parametrize("token, want", [
    ("-RA", ("-", "", ("RA",))), ("RA", ("", "", ("RA",))), ("+RA", ("+", "", ("RA",))),
    ("-SHRAGSSN", ("-", "SH", ("RA", "GS", "SN"))), ("+TSGRRA", ("+", "TS", ("GR", "RA"))),
    ("VCSH", ("VC", "SH", ())), ("VCTSRA", ("VC", "TS", ("RA",))), ("TS", ("", "TS", ())),
    ("BCFG", ("", "BC", ("FG",))), ("FZFG", ("", "FZ", ("FG",))), ("-FZRADZ", ("-", "FZ", ("RA", "DZ"))),
    ("BLSN", ("", "BL", ("SN",))), ("SHUP", ("", "SH", ("UP",))), ("BR", ("", "", ("BR",))),
])
def test_parse_token_reads_the_fm15_grammar(token, want):
    """Each token splits into proximity/intensity, one descriptor, and two-letter phenomena, in
    order. Every token here occurs in the frozen archive. Fails if the descriptor were read as a
    phenomenon (e.g. SH of -SHRA taken as nothing, TS lost from +TSGRRA)."""
    assert W.parse_token(token) == want


@pytest.mark.parametrize("token", ["RERA", "VC", "-", "", "XX", "RAX", "TSSHRA", "+", "FZ", "SH", "BL", "+BR", "-FG", "+TS"])
def test_parse_token_refuses_what_the_grammar_does_not_produce(token):
    """A recent-weather group, a bare qualifier, an unknown code, two descriptors, a descriptor with
    no phenomenon (only TS, and VCSH/VCTS, may stand alone), an intensity on something that is not
    precipitation (or FC/SS/DS): all raise, naming the token, rather than being read (a bare "FZ"
    used to set wx_frozen; review 2026-09-11)."""
    with pytest.raises(ValueError, match="unparseable present-weather token"):
        W.parse_token(token)


#: expected NON-ZERO flags per code string (every other flag must be exactly 0), and the intensity
CODES = {
    "-RA": ({"wx_ra", "wx_precip"}, -1.0),
    "RA +SHRA": ({"wx_ra", "wx_sh", "wx_precip"}, 1.0),
    "+TSGRRA": ({"wx_ts", "wx_gr", "wx_ra", "wx_frozen", "wx_precip"}, 1.0),
    "-FZDZ": ({"wx_fzdz", "wx_dz", "wx_frozen", "wx_precip"}, -1.0),
    "-FZRA": ({"wx_fzra", "wx_ra", "wx_frozen", "wx_precip"}, -1.0),
    "FZFG": ({"wx_fzfg", "wx_fg", "wx_frozen"}, np.nan),
    "BLSN": ({"wx_blsn", "wx_frozen"}, np.nan),
    "-SN BLSN": ({"wx_sn", "wx_blsn", "wx_frozen", "wx_precip"}, -1.0),
    "VCSH": ({"wx_vcsh"}, np.nan),
    "VCTS": ({"wx_vcts"}, np.nan),
    "VCFG": ({"wx_vcfg"}, np.nan),
    "-RA VCTS": ({"wx_ra", "wx_vcts", "wx_precip"}, -1.0),
    "BR": ({"wx_br"}, np.nan),
    "MIFG": ({"wx_fg"}, np.nan),
    "-SHRAGS": ({"wx_sh", "wx_ra", "wx_gs", "wx_frozen", "wx_precip"}, -1.0),
    "PL": ({"wx_pl", "wx_frozen", "wx_precip"}, 0.0),
    "-SG": ({"wx_sg", "wx_precip"}, -1.0),
    "IC": ({"wx_ic", "wx_frozen", "wx_precip"}, 0.0),
    "SHUP": ({"wx_sh", "wx_up", "wx_precip"}, 0.0),
    "HZ": (set(), np.nan),
    "M": (set(), np.nan),
}


def test_parse_codes_sets_exactly_the_flags_each_code_carries():
    """For every code string: exactly the expected flags are 1 and every other flag is 0 — so a
    vicinity code never counts at the station, blowing snow is not snowfall, freezing fog is not
    precipitation, and the heaviest precipitation intensity wins."""
    got = W.parse_codes(pd.Series(list(CODES)))
    for i, (code, (on, intensity)) in enumerate(CODES.items()):
        row = got.iloc[i]
        assert {f for f in W.FLAGS if row[f] == 1.0} == on, code
        assert all(row[f] == 0.0 for f in W.FLAGS if f not in on), code
        assert (np.isnan(row.wx_intensity) and np.isnan(intensity)) or row.wx_intensity == intensity, code


def test_an_unobservable_weather_group_is_unknown_not_dry():
    """An empty `wxcodes` (an AUTO report whose sensor could not observe: "/////////") is unknown —
    every flag NaN — while "M" (no weather group) is no significant weather — every flag 0. v1 read
    both as dry. Fails if the empty string were routed through the "M" branch."""
    got = W.parse_codes(pd.Series(["M", np.nan, "", "  "], dtype=object))
    assert (got.iloc[0][W.FLAGS] == 0.0).all()
    assert got.iloc[1:][W.FLAGS + ["wx_intensity"]].isna().all().all()


# ------------------------------------------------------------------ numbers, units, p01i ----------
def test_units_and_the_gust_rule(tmp_path):
    """F -> C for temperature and dew point, statute miles -> km, knots unchanged. A missing gust
    with a reported wind is 0 kt; with the wind missing the gust is NaN; a reported gust is kept."""
    d = archive(tmp_path, {"EDDF_2025-01.csv": [
        ("EDDF", "2025-01-10 05:00", 50.0, 41.0, 12, "M", 6.21, "0.00", "M", "x"),
        ("EDDF", "2025-01-10 05:30", 50.0, 41.0, 22, 35, 3.00, "0.00", "M", "x"),
        ("EDDF", "2025-01-10 06:00", "M", 41.0, "M", "M", "M", "0.00", "M", "x"),
    ]})
    o, _ = W.load_observations(d)
    assert list(o.columns) == W.OBS_COLUMNS and str(o.valid.dt.tz) == "UTC"
    assert o.temp_c.iloc[:2].tolist() == pytest.approx([10.0, 10.0], abs=1e-9)
    assert o.dewpoint_c.iloc[0] == pytest.approx(5.0) and o.dewspread_c.iloc[0] == pytest.approx(5.0)
    assert o.vis_km.iloc[:2].tolist() == pytest.approx([6.21 * 1.609344, 3.0 * 1.609344])
    assert o.wind_kt.iloc[:2].tolist() == [12.0, 22.0]
    assert o.gust_kt.iloc[0] == 0.0 and o.gust_kt.iloc[1] == 35.0
    assert np.isnan(o.gust_kt.iloc[2]), "with the wind unknown the gust is unknown, not calm"
    assert np.isnan(o.temp_c.iloc[2]) and np.isnan(o.dewspread_c.iloc[2]) and np.isnan(o.vis_km.iloc[2])


def test_a_constant_p01i_is_never_read_as_zero_precipitation(tmp_path):
    """THE regression. EDDM January, as the archive emits it: p01i "0.00" on every row while two
    rows report rain. That station-month is uninformative, so precip_mm is NaN on EVERY row of it
    (unknown), never 0.0 — and the rain still shows, in wx_precip. A station-month whose p01i does
    vary (a US-format one) is used, with a trace as 0.05 mm (v1 wrote 1.27 mm: it converted the
    placeholder from inches)."""
    d = archive(tmp_path, {
        "EDDM_2025-01.csv": [
            ("EDDM", "2025-01-10 05:00", 30.2, 28.4, 8, "M", 6.21, "0.00", "M", "x"),
            ("EDDM", "2025-01-10 06:00", 35.6, 33.8, 10, "M", 2.00, "0.00", "-RA", "x"),
            ("EDDM", "2025-01-10 07:00", 33.8, 32.0, 12, "M", 1.20, "0.00", "DZ", "x"),
        ],
        "KBOS_2025-01.csv": [
            ("KBOS", "2025-01-10 05:00", 35.6, 33.8, 10, "M", 2.00, "0.00", "M", "x"),
            ("KBOS", "2025-01-10 06:00", 35.6, 33.8, 10, "M", 2.00, "0.04", "-RA", "x"),
            ("KBOS", "2025-01-10 07:00", 35.6, 33.8, 10, "M", 2.00, "T", "-RA", "x"),
        ]})
    o, info = W.load_observations(d)
    eddm, kbos = o[o.station == "EDDM"], o[o.station == "KBOS"]
    assert eddm.precip_mm.isna().all(), eddm.precip_mm.tolist()
    assert eddm.wx_precip.tolist() == [0.0, 1.0, 1.0]
    assert kbos.precip_mm.tolist() == pytest.approx([0.0, 0.04 * 25.4, 0.05])
    p = info[info.field == "p01i"].set_index("station")
    assert not p.loc["EDDM", "informative"] and p.loc["EDDM", "n_distinct"] == 1
    assert p.loc["EDDM", "n_precip_codes"] == 2, "the contradiction the check reports"
    assert p.loc["KBOS", "informative"] and p.loc["KBOS", "n_distinct"] == 3


def test_distinct_values_are_counted_as_numbers(tmp_path):
    """'0.00', '0.0' and '0' are one value: a provider changing its formatting must not make a
    constant field look informative."""
    d = archive(tmp_path, {"EDDM_2025-02.csv": [
        ("EDDM", f"2025-02-10 0{i}:00", 35.6 + i, 33.8, 10, "M", 2.00, p, "-RA", "x")
        for i, p in enumerate(["0.00", "0.0", "0"])]})
    _, info = W.load_observations(d)
    p = info[info.field == "p01i"].iloc[0]
    assert p.n_distinct == 1 and not p.informative and p.n_reported == 3 and p.n_obs == 3


def test_numeric_fields_refuse_garbage(tmp_path):
    """A value that is neither a number nor 'M' raises (v1 coerced it to NaN silently), and a trace
    'T' outside p01i raises."""
    bad = archive(tmp_path, {"EDDF_2025-01.csv": [("EDDF", "2025-01-10 05:00", "abc", 41, 12, "M", 6.21, "0.00", "M", "x")]})
    with pytest.raises(ValueError, match="tmpf: 1 values are neither numeric nor 'M'"):
        W.load_observations(bad)
    trace = archive(tmp_path / "t", {"EDDF_2025-01.csv": [("EDDF", "2025-01-10 05:00", 50, 41, 12, "M", "T", "0.00", "M", "x")]})
    with pytest.raises(ValueError, match="vsby: a trace 'T' is only meaningful for p01i"):
        W.load_observations(trace)


# ------------------------------------------------------------------ the de-icing condition --------
def test_deicing_condition_follows_its_documented_rule_and_three_valued_logic(tmp_path):
    """dc_moist_cold = temp <= +3 AND (precip | fog | mist | vis <= 1.5 km); dc_frost = temp <= 0
    AND spread <= 3; deicing = either. Unknown inputs give NaN unless the known part decides it
    (cold and frost-prone with unknown weather is still 1; warm with unknown weather is still 0)."""
    rows = [  # temp F, dew F, vsby mi, wxcodes     -> moist, frost, deice
        (35.6, 33.8, 6.21, "-RA"),    # +2 C rain                         1 0 1
        (35.6, 33.8, 6.21, "M"),      # +2 C dry, clear                   0 0 0
        (35.6, 33.8, "0.932056788356001", "M"),   # +2 C, vis one float step above 1.5 km   0 0 0
        (35.6, 33.8, "0.9320567883560009", "M"),  # +2 C, vis one step below 1.5 km          1 0 1
        (35.6, 33.8, 1.86, "BR"),     # +2 C mist                         1 0 1
        (41.0, 39.2, 6.21, "-RA"),    # +5 C rain                         0 0 0
        (30.2, 26.6, 6.21, "M"),      # -1 C, spread 2                    0 1 1
        (30.2, 21.2, 6.21, "M"),      # -1 C, spread 5                    0 0 0
        (33.8, 32.0, 6.21, "VCSH"),   # +1 C showers NEARBY               0 0 0
        ("M", 33.8, 6.21, "-RA"),     # temperature unknown               N N N
        (30.2, 26.6, 6.21, ""),       # -1 C spread 2, weather unknown    N 1 1
        (41.0, 39.2, 6.21, ""),       # +5 C, weather unknown             0 0 0
    ]
    d = archive(tmp_path, {"EDDM_2025-01.csv": [
        ("EDDM", f"2025-01-10 {i:02d}:00", t, dp, 8, "M", v, "0.00", c, "x") for i, (t, dp, v, c) in enumerate(rows)]})
    o, _ = W.load_observations(d)
    N = np.nan
    want = np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0], [1, 0, 1], [1, 0, 1], [0, 0, 0], [0, 1, 1],
                     [0, 0, 0], [0, 0, 0], [N, N, N], [N, 1, 1], [0, 0, 0]], dtype=float)
    got = o[["dc_moist_cold", "dc_frost", "deicing_condition"]].to_numpy()
    np.testing.assert_array_equal(got, want)


def test_three_valued_helpers():
    """The Kleene tables the de-icing rule is built on: 0 dominates AND, 1 dominates OR, and an
    unknown survives otherwise."""
    N = np.nan
    a, b = np.array([0, 0, 0, 1, 1, 1, N, N, N]), np.array([0, 1, N, 0, 1, N, 0, 1, N])
    np.testing.assert_array_equal(W.and3(a, b), [0, 0, 0, 0, 1, N, 0, N, N])
    np.testing.assert_array_equal(W.or3(a, b), [0, 1, N, 1, 1, 1, N, 1, N])
    np.testing.assert_array_equal(W.le3(np.array([2.0, 3.0, 3.5, N]), 3.0), [1, 1, 0, N])


# ------------------------------------------------------------------ the hourly join ---------------
def hourly_obs(tmp_path):
    d = archive(tmp_path, {
        "EDDF_2025-01.csv": [("EDDF", f"2025-01-10 {t}", temp, 30.0, 8, "M", 6.21, "0.00", "M", "x")
                             for t, temp in [("09:50", 41.0), ("10:20", 42.8), ("10:50", 44.6), ("11:00", 46.4)]],
        "LEMD_2025-01.csv": [("LEMD", "2025-01-10 10:30", 59.0, 41.0, 8, "M", 6.21, "0.00", "M", "x")]})
    return W.load_observations(d)[0]


def test_hourly_uses_only_observations_at_or_before_the_hours_end(tmp_path):
    """THE join rule. Hour 10:00 at EDDF takes the 11:00 report (exactly at the hour's end:
    inclusive), age 0; hour 09:00 takes 09:50, age 600 s; hour 08:00 has nothing at or before 09:00
    -> NaN. LEMD hour 10:00 takes LEMD's 10:30 (age 1800 s) and LEMD hour 09:00 is NaN — never an
    EDDF report. Keys come back in their own order with their own columns."""
    obs = hourly_obs(tmp_path)
    keys = pd.DataFrame({"flight": ["a", "b", "c", "d", "e"],
                         "station": ["EDDF", "LEMD", "EDDF", "EDDF", "LEMD"],
                         "hour": pd.to_datetime(["2025-01-10 10:00", "2025-01-10 10:00", "2025-01-10 09:00",
                                                 "2025-01-10 08:00", "2025-01-10 09:00"], utc=True)})
    h = W.hourly(obs, keys)
    assert h.flight.tolist() == ["a", "b", "c", "d", "e"] and len(h) == 5
    assert h.temp_c.iloc[:3].tolist() == pytest.approx([8.0, 15.0, 5.0])
    assert h.obs_age_s.iloc[:3].tolist() == [0.0, 1800.0, 600.0]
    assert h.obs_valid.iloc[0] == pd.Timestamp("2025-01-10 11:00", tz="UTC")
    assert h.iloc[3:][["temp_c", "obs_age_s"]].isna().all().all() and h.obs_valid.iloc[3:].isna().all()
    assert (h.obs_valid.dropna() <= h.hour[h.obs_valid.notna()] + pd.Timedelta(hours=1)).all()


def test_hourly_refuses_malformed_keys(tmp_path):
    obs = hourly_obs(tmp_path)
    h = pd.to_datetime(["2025-01-10 10:00"], utc=True)
    with pytest.raises(ValueError, match="need a 'hour' column"):
        W.hourly(obs, pd.DataFrame({"station": ["EDDF"]}))
    with pytest.raises(ValueError, match="need a 'station' column"):
        W.hourly(obs, pd.DataFrame({"hour": h}))
    with pytest.raises(ValueError, match="timezone-aware"):
        W.hourly(obs, pd.DataFrame({"station": ["EDDF"], "hour": pd.to_datetime(["2025-01-10 10:00"])}))
    with pytest.raises(ValueError, match="on the hour"):
        W.hourly(obs, pd.DataFrame({"station": ["EDDF"], "hour": pd.to_datetime(["2025-01-10 10:05"], utc=True)}))
    with pytest.raises(ValueError, match="must be UTC"):
        W.hourly(obs, pd.DataFrame({"station": ["EDDF"], "hour": pd.to_datetime(["2025-01-10 10:00"]).tz_localize("Asia/Kolkata")}))
    with pytest.raises(ValueError, match="already carry output columns"):
        W.hourly(obs, pd.DataFrame({"station": ["EDDF"], "hour": h, "temp_c": [1.0]}))


# ------------------------------------------------------------------ the archive boundary ----------
def test_read_archive_refuses_a_broken_archive(tmp_path):
    """No file, a missing column, an unparseable timestamp, a repeated (station, valid) key: each
    raises naming the defect, before any value is derived."""
    with pytest.raises(FileNotFoundError, match="no weather archive"):
        W.read_archive(tmp_path / "nowhere")
    d = tmp_path / "a"; d.mkdir()
    (d / "EDDF_2025-01.csv").write_text("station,valid,tmpf\nEDDF,2025-01-10 05:00,50\n")
    with pytest.raises(ValueError, match="missing columns"):
        W.read_archive(d)
    row = ("EDDF", "2025-01-10 05:00", 50, 41, 12, "M", 6.21, "0.00", "M", "x")
    with pytest.raises(ValueError, match="1 weather rows have an unparseable timestamp"):
        W.read_archive(archive(tmp_path / "b", {"EDDF_2025-01.csv": [row, ("EDDF", "not a time") + row[2:]]}))
    with pytest.raises(ValueError, match="1 observations repeat a"):
        W.read_archive(archive(tmp_path / "c", {"EDDF_2025-01.csv": [row, row]}))
    with pytest.raises(ValueError, match="1 weather rows have no station"):
        W.read_archive(archive(tmp_path / "d", {"EDDF_2025-01.csv": [row, ("", "2025-01-10 06:00") + row[2:]]}))


def test_main_writes_versioned_files_and_a_manifest(tmp_path):
    """`python -m prc.weather` writes three files, each carrying VERSION in its name, and a manifest
    whose digest is the archive's and whose p01i verdict matches the info table. Nothing else is
    written."""
    d = archive(tmp_path, {"EDDM_2025-01.csv": [
        ("EDDM", "2025-01-10 05:00", 30.2, 28.4, 8, "M", 6.21, "0.00", "-RA", "x"),
        ("EDDM", "2025-01-10 06:00", 31.2, 28.4, 8, "M", 6.21, "0.00", "M", "x")]})
    out = tmp_path / "out"
    assert W.main(["--raw-dir", str(d), "--out-dir", str(out)]) == 0
    v = W.VERSION
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [f"weather_obs_v{v}.parquet", f"info_content_v{v}.parquet", f"manifest_v{v}.json"])
    m = json.loads((out / f"manifest_v{v}.json").read_text())
    assert m["version"] == v and m["archive_digest"] == W.archive_digest(d)
    assert m["n_observations"] == 2 and m["p01i_informative_station_months"] == 0 and m["station_months"] == 1
    assert "EDDM 2025-01 p01i" in m["uninformative"]
    back = pd.read_parquet(out / f"weather_obs_v{v}.parquet")
    assert list(back.columns) == W.OBS_COLUMNS and back.precip_mm.isna().all()


# ------------------------------------------------------------------ the real archive --------------
@needs_weather
def test_the_real_archive_parses_and_matches_an_independent_reading_of_its_codes():
    """On all 205,417 observations: every token parses; p01i is uninformative in every one of the
    140 station-months, so precip_mm is NaN everywhere; the flags agree with a plain substring
    reading of the codes done here, independently of the grammar; the 15 unobservable rows are
    NaN; and freezing precipitation and de-icing conditions concentrate in winter."""
    obs, info = W.load_observations()
    assert len(obs) == 205_417 and obs.station.nunique() == 10
    p = info[info.field == "p01i"]
    assert len(p) == 140 and not p.informative.any()
    assert obs.precip_mm.isna().all()
    assert info[info.field.isin(["tmpf", "dwpf", "sknt"])].informative.all()
    codes = obs.wxcodes
    unknown = codes.isna()
    assert int(unknown.sum()) == 15 and obs.loc[unknown, W.FLAGS].isna().all().all()
    toks = codes.fillna("").str.split()

    def any_token(pred):
        return toks.apply(lambda ts: any(pred(t) for t in ts)).to_numpy()

    at_station = lambda t: not t.startswith("VC")                                    # noqa: E731
    wet = any_token(lambda t: at_station(t) and t not in ("BLSN", "DRSN")
                    and any(c in t for c in ("RA", "DZ", "SN", "SG", "PL", "GS", "GR", "IC", "UP")))
    known = ~unknown.to_numpy()
    assert np.array_equal(obs.wx_precip.to_numpy()[known] == 1.0, wet[known])
    # the FZ descriptor qualifies EVERY phenomenon of its group: -FZRADZ is freezing rain AND drizzle
    for flag, ph in [("wx_fzra", "RA"), ("wx_fzdz", "DZ"), ("wx_fzfg", "FG")]:
        assert np.array_equal(obs[flag].to_numpy()[known] == 1.0,
                              any_token(lambda t, p=ph: "FZ" in t and p in t)[known]), flag
    assert np.array_equal(obs.wx_ts.to_numpy()[known] == 1.0, any_token(lambda t: at_station(t) and "TS" in t)[known])
    assert int((obs.wx_fzra == 1).sum()) == 34 and int((obs.wx_fzdz == 1).sum()) == 18
    m = obs.valid.dt.month
    assert obs.deicing_condition[m == 1].mean() > 0.1 > obs.deicing_condition[m == 7].mean()
