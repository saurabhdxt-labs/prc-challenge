"""scripts/adsb_storm.py -- arm E4b (plans/PREREG_adsb_eham_v2_2026_09_10.md) on synthetic tracks and rows.

Pins: the ONE bias constant is median(sensor - BLOCK_TIME) on the 2025-01-09 census day only and never moves with the
validation days; V1 / V2 / V3 arithmetic at their boundaries (200 s, 50%, the 15-row minimum, 5%) and the verdict
mapping; S1C is compared only under the stamped taxi-time convention (BC-2); no 2026 label is read; `gate --e4b` and
`ship --e4b` use the constant, and the 2026 path refuses without a WORKING validation; the extract allowlist; the
write-once calibration and verdict. The synthetic world is test_adsb_storm's: BLOCK_TIME = the true off-block t0,
AOBT_3 = t0 - 60 s, the aircraft moves at t0 + 180 s, so the raw sensor sits 180 ... 196.7 s after BLOCK_TIME.
Every RNG is seeded. Mutation rehearsals (break -> RED -> restore byte-identical via backup + `diff -q` -> GREEN),
2026-09-10, are recorded in each docstring.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import adsb_storm as st  # noqa: E402
import test_adsb_storm as world_mod  # noqa: E402  (the synthetic world; only the module object is imported)

synthetic_world = world_mod.synthetic_world
LAT0, LON0 = world_mod.LAT0, world_mod.LON0
CADENCE_S, PUSH_DELAY_S, SPEED_MS = world_mod.CADENCE_S, world_mod.PUSH_DELAY_S, world_mod.SPEED_MS
S1C_LOW_S = 400.0          # the synthetic incumbent runs 400 s low on every unmatched row


def _s1c_frame(rank: pd.DataFrame) -> pd.DataFrame:
    """An out-of-month S1C record for the 2025 unmatched rows of `rank`, in the taxi-time convention."""
    un = rank[rank.CALLSIGN_flt.isna() & (rank.MVT_TIME_UTC_mvt.dt.year == 2025) & (rank.ADEP_mvt == "EHAM")]
    y = un.TAXITIME_SEC_mvt.to_numpy(dtype="float64")
    return pd.DataFrame({"MVT_ID_mvt": un.MVT_ID_mvt.to_numpy(), "fold": "lomo",
                         "date": un.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy(), "ADEP_mvt": "EHAM", "y": y,
                         "hprox_med": np.where(np.arange(len(un)) % 2, 1600.0, 900.0), "S1C": y - S1C_LOW_S})


def _write_s1c(df: pd.DataFrame, path: pathlib.Path, convention: bytes | None = b"taxi_time") -> pathlib.Path:
    t = pa.Table.from_pandas(df, preserve_index=False)
    md = dict(t.schema.metadata or {})
    if convention is not None:
        md[b"convention"] = convention
    pq.write_table(t.replace_schema_metadata(md), path)
    return path


def _write_day_files(samples: pd.DataFrame, days, d: pathlib.Path) -> None:
    """One adsb_<day>_EHAM.parquet per day, split by the UTC date of each sample (as the extract writes them)."""
    day = pd.to_datetime(samples.t, unit="s", utc=True).dt.strftime("%Y-%m-%d")
    for dd in days:
        samples[day == dd].to_parquet(d / f"adsb_{dd}_EHAM.parquet", index=False)


def _rename(rank: pd.DataFrame, samples: pd.DataFrame, id_off: float, tag: str) -> tuple:
    """Give a second synthetic world distinct MVT ids, callsigns and aircraft so two worlds can share one frame."""
    r = rank.copy()
    r["MVT_ID_mvt"] = r.MVT_ID_mvt + id_off
    r["CALLSIGN_flt"] = (tag + r.CALLSIGN_flt).astype("string")
    s = samples.copy()
    s["callsign"] = tag + s.callsign
    s["icao"] = tag + s.icao
    return r, s


@pytest.fixture(scope="module")
def census():
    rank, _, samples = synthetic_world(seed=3, days=[st.CENSUS_DAY])
    return dict(rank=rank, samples=samples, cal=st.calibrate_bias(rank, samples))


@pytest.fixture(scope="module")
def val(census):
    rank, _, samples = synthetic_world(seed=6, n_matched=24, n_unmatched=15, days=st.VALIDATION_DAYS)
    tracks = st.tracks_frame(samples, LAT0, LON0)
    s1c = _s1c_frame(rank)
    rows, rep = st.validate_e4b(rank, tracks, census["cal"]["bias_s"], s1c)
    return dict(rank=rank, samples=samples, tracks=tracks, s1c=s1c, rows=rows, rep=rep, bias=census["cal"]["bias_s"])


@pytest.fixture(scope="module")
def storm():
    rank, truth, samples = synthetic_world()
    flights = st.flights_frame(rank, st.DAYS)
    return dict(rank=rank, samples=samples, flights=flights, tracks=st.tracks_frame(samples, LAT0, LON0))


def _validation_dict(verdict: str, bias: float, clauses=None) -> dict:
    ok = {"pass": True}
    return {"prereg": st.PREREG_E4B, "verdict": verdict, "bias": {"bias_s": bias},
            "clauses": clauses or {"V1": {"clause": "V1", **ok}, "V2": {"clause": "V2", **ok}, "V3": {"clause": "V3", **ok}}}


# =============================================================================================
# measure: the constant mode
# =============================================================================================

def test_measure_constant_mode_applies_one_constant_and_leaves_the_raw_sensor_alone(storm):
    """Every row, matched or unmatched, every day, carries the constant; the raw sensor and the gate are E4's.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the constant
    branch of measure() given the pooled AOBT_3 median instead -> RED.
    """
    fl, tr = storm["flights"], storm["tracks"]
    e_loo, _ = st.measure(fl, tr, st.DAYS)
    e_c, info = st.measure(fl, tr, st.DAYS, bias_mode="constant", bias_s=123.5)
    assert (e_c.bias_s.to_numpy() == 123.5).all() and info["bias_by_day_s"] == {d: 123.5 for d in st.DAYS}
    np.testing.assert_array_equal(e_c.offblock_raw_s.to_numpy(), e_loo.offblock_raw_s.to_numpy())    # NaNs in place
    fired = e_c.gate_fired.to_numpy()
    assert fired.tolist() == np.isfinite(e_c.offblock_raw_s.to_numpy()).tolist() and fired.sum() > 180
    raw, mvt = e_c.offblock_raw_s.to_numpy(), fl.mvt_s.to_numpy()
    # fp64 at 1.7e9 s carries ~2.4e-7 s of spacing: 1e-6 s is round-off, not tolerance for a wrong bias
    np.testing.assert_allclose(e_c.taxi_hat.to_numpy()[fired], (mvt - (raw - 123.5))[fired], atol=1e-6)
    e_c2, _ = st.measure(fl, tr, st.DAYS, bias_mode="constant", bias_s=223.5)
    np.testing.assert_allclose(e_c2.taxi_hat.to_numpy()[fired] - e_c.taxi_hat.to_numpy()[fired], 100.0, atol=1e-6)
    with pytest.raises(ValueError, match="needs a finite bias_s"):
        st.measure(fl, tr, st.DAYS, bias_mode="constant")
    with pytest.raises(ValueError, match="needs a finite bias_s"):
        st.measure(fl, tr, st.DAYS, bias_mode="constant", bias_s=float("nan"))
    with pytest.raises(ValueError, match="only for bias_mode 'constant'"):
        st.measure(fl, tr, st.DAYS, bias_mode="loo_day", bias_s=1.0)


# =============================================================================================
# calibrate: the ONE constant, from the census day only
# =============================================================================================

def test_calibrate_bias_is_the_census_median_of_sensor_minus_block_with_n_and_iqr(census):
    """Recomputed independently from the measured raw sensor and the BLOCK_TIME label; anchored physically (the synthetic
    aircraft leaves the 100 m circle 180 ... 196.7 s after BLOCK_TIME) and against census_c3's BLOCK diagnostic.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the constant
    referenced to AOBT_3 instead of BLOCK_TIME (E4's defect) -> RED; IQR computed as q75 - q50 -> RED.
    """
    rank, samples, cal = census["rank"], census["samples"], census["cal"]
    fl = st.flights_frame(rank, [st.CENSUS_DAY])
    est, _ = st.measure(fl, st.tracks_frame(samples, LAT0, LON0), [st.CENSUS_DAY], bias_mode="pooled")
    raw = est.offblock_raw_s.to_numpy()
    m = est.matched.to_numpy() & np.isfinite(raw)
    block = st.to_epoch_s(rank.set_index("MVT_ID_mvt").BLOCK_TIME_UTC_mvt.loc[est.MVT_ID_mvt.to_numpy()[m]])
    d = raw[m] - block
    assert cal["bias_s"] == float(np.median(d)) and cal["n"] == int(m.sum()) == 21 and cal["n_matched"] == 24
    assert cal["iqr_s"] == float(np.percentile(d, 75) - np.percentile(d, 25)) and cal["iqr_s"] > 0
    assert cal["q25_s"] < cal["bias_s"] < cal["q75_s"]
    assert PUSH_DELAY_S - CADENCE_S <= cal["bias_s"] <= PUSH_DELAY_S + 100.0 / SPEED_MS + 1e-6
    assert cal["median_aobt3_minus_block_s"] == pytest.approx(-60.0, abs=1e-6)       # ns stamps of float seconds
    assert cal["coverage"] == pytest.approx(21 / 24, abs=1e-12)
    assert cal["bias_s"] == st.census_c3(rank, samples)["diagnostic_block_calibrated"]["bias_s"]
    assert cal["calibration_day"] == st.CENSUS_DAY and cal["prereg"] == st.PREREG_E4B
    with pytest.raises(ValueError, match="census day 2025-01-09 only"):
        st.calibrate_bias(rank, samples, day="2025-12-27")


def test_the_bias_is_computed_from_the_calibration_day_only_and_validate_never_re_estimates_it(census, val):
    """(1) calibrate on a frame holding the census day AND both validation days: identical to census-only, and unmoved
    when the validation days' labels shift +777 s and their tracks +333 s. (2) validate_e4b carries the constant it is
    handed on every row; +50 s in -> every estimate -50 s out. (3) the calibration day is refused as a validation day;
    a bias file for another day or prereg is refused.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): calibrate_bias
    measures every day in `truth` instead of the census day -> RED; validate_e4b re-estimates the bias in-sample on the
    validation days (bias_mode 'pooled') -> RED.
    """
    rv, sv = _rename(val["rank"], val["samples"], 500_000.0, "V")
    both_r = pd.concat([census["rank"], rv], ignore_index=True)
    both_s = pd.concat([census["samples"], sv], ignore_index=True)
    cal_both = st.calibrate_bias(both_r, both_s)
    keys = ("bias_s", "n", "iqr_s", "q25_s", "q75_s", "n_matched")
    assert {k: cal_both[k] for k in keys} == {k: census["cal"][k] for k in keys}
    on_val = both_r.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").isin(st.VALIDATION_DAYS)
    moved_r = both_r.copy()
    moved_r.loc[on_val, "BLOCK_TIME_UTC_mvt"] = moved_r.loc[on_val, "BLOCK_TIME_UTC_mvt"] + pd.Timedelta(seconds=777)
    moved_r.loc[on_val, "TAXITIME_SEC_mvt"] = moved_r.loc[on_val, "TAXITIME_SEC_mvt"] - 777
    moved_s = both_s.copy()
    moved_s.loc[moved_s.icao.str.startswith("V"), "t"] += 333.0
    cal_moved = st.calibrate_bias(moved_r, moved_s)
    assert {k: cal_moved[k] for k in keys} == {k: census["cal"][k] for k in keys}
    # (2) the constant handed to validate is the constant used, on every row
    b = val["bias"]
    rows = val["rows"]
    assert (rows.bias_s.to_numpy() == b).all()
    rows2, rep2 = st.validate_e4b(val["rank"], val["tracks"], b + 50.0, val["s1c"])
    f = rows.gate_fired.to_numpy()
    np.testing.assert_allclose(rows2.offblock_hat_s.to_numpy()[f] - rows.offblock_hat_s.to_numpy()[f], -50.0, atol=1e-6)
    v1 = rows.in_v1.to_numpy()
    np.testing.assert_allclose(rows2.err_vs_block_s.to_numpy()[v1] - rows.err_vs_block_s.to_numpy()[v1], -50.0, atol=1e-6)
    assert rep2["bias"]["bias_s"] == b + 50.0
    assert rep2["clauses"]["V1"]["rmse_s"] == pytest.approx(50.0, abs=CADENCE_S)       # a constant 50 s offset, cadence-limited
    # (3)
    with pytest.raises(ValueError, match="never a validation day"):
        st.validate_e4b(census["rank"], val["tracks"], b, val["s1c"], days=[st.CENSUS_DAY, st.VALIDATION_DAYS[0]])


def test_load_bias_refuses_a_file_that_is_not_the_census_calibration(tmp_path):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    calibration-day check removed from load_bias -> RED.
    """
    p = tmp_path / "bias.json"
    good = {"prereg": st.PREREG_E4B, "calibration_day": st.CENSUS_DAY, "bias_s": 150.25, "n": 300}
    p.write_text(json.dumps(good))
    assert st.load_bias(p) == 150.25
    for bad, msg in (({**good, "calibration_day": "2025-12-27"}, "not E4b's 2025-01-09 calibration"),
                     ({**good, "prereg": st.PREREG}, "not E4b's 2025-01-09 calibration"),
                     ({**good, "bias_s": None}, "no finite bias_s")):
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match=msg):
            st.load_bias(p)
    with pytest.raises(SystemExit, match="run `adsb_storm.py calibrate` first"):
        st.load_bias(tmp_path / "missing.json")


# =============================================================================================
# clauses and the verdict
# =============================================================================================

def test_clause_v1_arithmetic_and_boundaries():
    """RMSE over matched gated rows <= 200 s, coverage over ALL matched rows >= 50%; unmatched rows never count.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): V1_RMSE_S
    200 -> 250 -> RED; clause_v1 forwarding C1's thresholds instead of its own -> RED. In the first version both went RED
    only at the reported-threshold assertion; that assertion was moved last and both re-run: RED at the 200.01 s case.
    """
    matched = np.array([True] * 4 + [False] * 2)
    c = st.clause_v1(np.array([200.0, -200.0, np.nan, np.nan, 5000.0, np.nan]), matched)
    assert c["clause"] == "V1" and c["n_matched"] == 4 and c["n_gated"] == 2 and c["coverage"] == 0.5
    assert c["rmse_s"] == 200.0 and c["pass"]
    assert not st.clause_v1(np.array([200.01, -200.0, np.nan, np.nan, 5.0, np.nan]), matched)["pass"]
    assert not st.clause_v1(np.array([220.0, 220.0, np.nan, np.nan, 5.0, np.nan]), matched)["pass"]     # 220: within E4's 250, not V1's 200
    assert st.clause_v1(np.array([240.0, 0.0, 0.0, 0.0, np.nan, np.nan]), matched)["pass"]            # RMSE 120, coverage 1
    five = np.array([True] * 10)
    err = np.array([1.0] * 5 + [np.nan] * 5)                                                             # exactly 50%
    assert st.clause_v1(err, five)["pass"]
    err[4] = np.nan                                                                                      # 40%
    assert not st.clause_v1(err, five)["pass"]
    assert not st.clause_v1(np.full(6, np.nan), matched)["pass"]
    # the reported thresholds, checked last so the boundary cases above are what bite a threshold change
    assert c["rmse_threshold_s"] == 200.0 and c["coverage_threshold"] == 0.5
    assert (st.V1_RMSE_S, st.V1_MIN_COVERAGE) == (200.0, 0.5)


def test_clause_v2_strict_beat_and_the_15_row_minimum():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): n < min_rows
    -> n <= min_rows -> RED; r_hat < r_s1c -> r_hat <= r_s1c -> RED; V2_MIN_ROWS 15 -> 14 -> RED (first version: only at
    the reported min_rows field; the 14-row case now runs first and the re-run is RED there).
    """
    y = np.full(15, 1000.0)
    hat, s1c = y + 100.0, y - 400.0
    c14 = st.clause_v2(hat[:14], s1c[:14], y[:14])                                          # one row short: undecided
    assert c14["status"] == "INCONCLUSIVE" and not c14["pass"] and c14["rmse_taxi_hat_s"] == 100.0
    c = st.clause_v2(hat, s1c, y)                                                            # exactly 15 rows: decides
    assert c["status"] == "PASS" and c["pass"]
    assert c == {"clause": "V2", "n_rows": 15, "min_rows": 15, "rmse_taxi_hat_s": 100.0, "rmse_s1c_s": 400.0,
                 "status": "PASS", "pass": True}
    tie = st.clause_v2(y + 400.0, s1c, y)                                                    # equal RMSE: not a beat
    assert tie["status"] == "FAIL" and not tie["pass"]
    worse = st.clause_v2(y + 401.0, s1c, y)
    assert worse["status"] == "FAIL" and not worse["pass"]
    empty = st.clause_v2([], [], [])
    assert empty["status"] == "INCONCLUSIVE" and empty["n_rows"] == 0 and not empty["pass"]
    with pytest.raises(ValueError, match="finite taxi_hat, S1C and y"):
        st.clause_v2(np.r_[hat[:14], np.nan], s1c, y)
    with pytest.raises(ValueError, match="differ in length"):
        st.clause_v2(hat[:3], s1c, y)
    assert st.V2_MIN_ROWS == 15


def test_clause_v3_boundary_and_empty():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): V3_MAX_SHARE
    0.05 -> 0.06 -> RED (first version: only at the reported threshold; now RED at the 1-in-19 case).
    """
    j = pd.DataFrame({"n_candidates": [1] * 20 + [0] * 7, "ambiguous": [True] + [False] * 26})
    c = st.clause_v3(j)
    assert c["clause"] == "V3" and c["share"] == 0.05 and c["pass"] and c["n_no_candidate"] == 7
    j2 = pd.DataFrame({"n_candidates": [1] * 19, "ambiguous": [True] + [False] * 18})                 # 5.26%
    assert not st.clause_v3(j2)["pass"]
    assert not st.clause_v3(pd.DataFrame({"n_candidates": [0, 0], "ambiguous": [False, False]}))["pass"]
    assert c["threshold"] == 0.05 and st.V3_MAX_SHARE == 0.05          # reported threshold, last (see test_clause_v1)


def test_verdict_e4b_mapping():
    """WORKING iff V1-V3 pass; NOT WORKING iff V1 fails; else INCONCLUSIVE (a V2 short of 15 rows included).
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the V1 -> NOT
    WORKING branch removed -> RED; V2 dropped from the WORKING condition -> RED.
    """
    ok, no = {"pass": True}, {"pass": False}
    v2_short = {"pass": False, "status": "INCONCLUSIVE"}
    assert st.verdict_e4b(ok, ok, ok) == "WORKING"
    assert st.verdict_e4b(no, ok, ok) == "NOT WORKING" and st.verdict_e4b(no, no, no) == "NOT WORKING"
    assert st.verdict_e4b(ok, v2_short, ok) == "INCONCLUSIVE" and st.verdict_e4b(ok, no, ok) == "INCONCLUSIVE"
    assert st.verdict_e4b(ok, ok, no) == "INCONCLUSIVE"
    assert st.VALIDATION_DAYS == ("2025-12-27", "2025-12-28") and st.CENSUS_DAY not in st.VALIDATION_DAYS


# =============================================================================================
# validate, end to end
# =============================================================================================

def test_validate_e4b_on_the_synthetic_validation_days_is_working_and_every_number_reconciles(val):
    """Calibrated on the census, the sensor lands within a cadence of BLOCK_TIME on matched rows and of the label on
    unmatched ones, where the synthetic S1C runs 400 s low: V1-V3 pass. The per-row table reconciles with the report.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): V1's error taken
    against AOBT_3 instead of BLOCK_TIME -> RED; the V2 row mask drops the unmatched condition -> RED (as an ERROR in the
    fixture: the S1C presence guard refuses the matched rows that entered V2, before the mask assertion runs); per-day
    coverage over all rows of the day -> RED.
    """
    rows, rep = val["rows"], val["rep"]
    c = rep["clauses"]
    assert rep["verdict"] == "WORKING" and c["V1"]["pass"] and c["V2"]["pass"] and c["V3"]["pass"]
    assert c["V1"]["rmse_s"] < CADENCE_S and 0.85 < c["V1"]["coverage"] < 1.0
    assert c["V2"]["status"] == "PASS" and c["V2"]["n_rows"] == 30 and c["V2"]["rmse_taxi_hat_s"] < CADENCE_S
    assert c["V2"]["rmse_s1c_s"] == pytest.approx(S1C_LOW_S, abs=1e-9)
    m, f, a = rows.matched.to_numpy(), rows.gate_fired.to_numpy(), rows.join_ambiguous.to_numpy()
    assert rows.in_v1.tolist() == (m & f).tolist() and rows.in_v2.tolist() == (~m & f & ~a).tolist()
    hat = rows.offblock_raw_s.to_numpy() - rows.bias_s.to_numpy()
    np.testing.assert_allclose(rows.offblock_hat_s.to_numpy()[f], hat[f], atol=0.0)
    v1 = rows.in_v1.to_numpy()
    np.testing.assert_allclose(rows.err_vs_block_s.to_numpy()[v1], (hat - rows.block_s.to_numpy())[v1], atol=0.0)
    assert np.isnan(rows.err_vs_block_s.to_numpy()[~v1]).all()
    np.testing.assert_allclose(rows.taxi_hat.to_numpy()[f], np.maximum(rows.mvt_s.to_numpy() - hat, 1.0)[f], atol=1e-6)
    np.testing.assert_allclose(rows.y.to_numpy(), rows.mvt_s.to_numpy() - rows.block_s.to_numpy(), atol=0.5)  # int label
    assert c["V1"]["rmse_s"] == st.rmse(rows.err_vs_block_s.to_numpy()[v1])
    for d in st.VALIDATION_DAYS:
        r, on = rep["per_day"][d], (rows.day == d).to_numpy()
        assert r["n_matched"] == int((on & m).sum()) and r["n_gated"] == int((on & v1).sum())
        assert r["coverage"] == pytest.approx(r["n_gated"] / r["n_matched"], abs=1e-12)
        assert r["n_v2_rows"] == int((on & rows.in_v2.to_numpy()).sum())
    assert sum(rep["per_day"][d]["n_gated"] for d in st.VALIDATION_DAYS) == c["V1"]["n_gated"]
    assert rep["diagnostics"]["rmse_v1_rows_vs_aobt3_s"] > 50.0           # AOBT_3 sits 60 s before BLOCK_TIME here
    assert rep["diagnostics"]["v2_rows_hprox_med_ge_1500"]["n"] == int((rows.in_v2 & (rows.hprox_med >= 1500)).sum()) > 0
    assert rep["judgement_calls"] == st.JUDGEMENT_CALLS_E4B and rep["days"] == list(st.VALIDATION_DAYS)
    json.dumps(st._json_ready(rep))


def test_validate_e4b_floors_taxi_hat_at_one_second_and_maps_a_v3_failure_to_inconclusive(val):
    """A bias of -1e5 s puts every estimated off-block after take-off: every gated taxi_hat must be exactly 1 s. A second
    seeded world (seed 7) has 2 of 30 unmatched rows ambiguous: V3 fails, V1 passes -> INCONCLUSIVE.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the 1 s floor
    removed from validate_e4b -> RED.
    """
    rows, _ = st.validate_e4b(val["rank"], val["tracks"], -1e5, val["s1c"])
    f = rows.gate_fired.to_numpy()
    assert f.sum() > 40 and (rows.taxi_hat.to_numpy()[f] == 1.0).all() and np.isnan(rows.taxi_hat.to_numpy()[~f]).all()
    rank, _, samples = synthetic_world(seed=7, n_matched=24, n_unmatched=15, days=st.VALIDATION_DAYS)
    _, rep = st.validate_e4b(rank, st.tracks_frame(samples, LAT0, LON0), val["bias"], _s1c_frame(rank))
    c = rep["clauses"]
    assert c["V1"]["pass"] and not c["V3"]["pass"] and c["V3"]["n_ambiguous_or_unmatched"] == 2
    assert rep["verdict"] == "INCONCLUSIVE"


def test_validate_e4b_compares_s1c_only_under_the_stamped_label_convention(val, tmp_path):
    """BC-2: the S1C record's `y` must BE TAXITIME_SEC_mvt on every V2 row, every V2 row must have an out-of-month
    prediction of the same EHAM day, and the file must be stamped convention=taxi_time.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the y ==
    TAXITIME check removed -> RED; the convention-stamp check removed from load_s1c -> RED.
    """
    s1c = val["s1c"]
    v2_id = float(val["rows"].MVT_ID_mvt[val["rows"].in_v2].iloc[0])
    shifted = s1c.assign(y=np.where(s1c.MVT_ID_mvt == v2_id, s1c.y + 1.0, s1c.y))
    with pytest.raises(ValueError, match="differs from TAXITIME_SEC_mvt"):
        st.validate_e4b(val["rank"], val["tracks"], val["bias"], shifted)
    with pytest.raises(ValueError, match="no S1C prediction"):
        st.validate_e4b(val["rank"], val["tracks"], val["bias"], s1c[s1c.MVT_ID_mvt != v2_id])
    with pytest.raises(ValueError, match="not the out-of-month"):
        st.validate_e4b(val["rank"], val["tracks"], val["bias"], s1c.assign(fold="in_month"))
    ok = st.load_s1c(_write_s1c(s1c, tmp_path / "ok.parquet"))
    pd.testing.assert_frame_equal(ok, s1c[["MVT_ID_mvt", "fold", "date", "ADEP_mvt", "y", "hprox_med", "S1C"]])
    with pytest.raises(ValueError, match="not stamped convention=taxi_time"):
        st.load_s1c(_write_s1c(s1c, tmp_path / "none.parquet", convention=None))
    with pytest.raises(ValueError, match="not stamped convention=taxi_time"):
        st.load_s1c(_write_s1c(s1c, tmp_path / "delta.parquet", convention=b"delta"))


# =============================================================================================
# no 2026 label is ever read
# =============================================================================================

def test_labels_are_read_for_2025_take_offs_only(val, tmp_path):
    """labels_2025 is the one label reader on the E4b path; the synthetic frame carries a 2026-01-15 row WITH labels.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the 2025-year
    check removed from labels_2025 -> RED; the 2025-day check removed from load_validation_truth -> RED. First version
    asked for 2026-01-02, a day the file does not hold, so the removed guard surfaced as an empty-file SystemExit rather
    than a missed refusal; an unreachable year re-check in load_validation_truth was deleted and the test re-pointed at
    the labelled 2026-01-15 row: now DID NOT RAISE.
    """
    rank = val["rank"]
    ids25 = rank.MVT_ID_mvt[(rank.MVT_TIME_UTC_mvt.dt.year == 2025)].to_numpy()[:5]
    block, taxi = st.labels_2025(rank, ids25)
    r = rank.set_index("MVT_ID_mvt")
    np.testing.assert_array_equal(block, st.to_epoch_s(r.BLOCK_TIME_UTC_mvt.loc[ids25]))
    np.testing.assert_array_equal(taxi, r.TAXITIME_SEC_mvt.loc[ids25].to_numpy(dtype="float64"))
    id26 = float(rank.MVT_ID_mvt[rank.MVT_TIME_UTC_mvt.dt.year == 2026].iloc[0])
    assert np.isfinite(r.TAXITIME_SEC_mvt.loc[id26])                              # the premise: the 2026 row HAS a label
    with pytest.raises(ValueError, match="label read refused: 1 requested rows"):
        st.labels_2025(rank, np.r_[ids25, id26])
    nat = rank.copy()
    nat.loc[nat.MVT_ID_mvt == ids25[0], "MVT_TIME_UTC_mvt"] = pd.NaT
    with pytest.raises(ValueError, match="label read refused"):
        st.labels_2025(nat, ids25)
    with pytest.raises(ValueError, match="not in the label frame"):
        st.labels_2025(rank, [123.0])
    p = tmp_path / "training.parquet"
    rank.to_parquet(p, index=False)
    with pytest.raises(ValueError, match="2025 days only"):
        st.load_validation_truth(p, days=["2026-01-15"])          # the file DOES hold a labelled 2026-01-15 EHAM departure
    t = st.load_validation_truth(p)
    # 2 days x 39 departures; the arrival, the LSZH row and the 2026-01-15 row are not read
    assert set(t.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d")) == set(st.VALIDATION_DAYS) and len(t) == 2 * 39
    with pytest.raises(ValueError, match="2025 days only"):
        st.validate_e4b(rank, val["tracks"], val["bias"], val["s1c"], days=["2026-01-02"])


# =============================================================================================
# the 2026 path: gate --e4b
# =============================================================================================

def test_gate_report_e4b_uses_the_constant_and_carries_the_validation_verdict(storm):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again):
    gate_report_e4b measuring with E4's leave-one-day-out AOBT_3 bias -> RED.
    """
    fl, tr = storm["flights"], storm["tracks"]
    val_d = _validation_dict("WORKING", 191.25)
    est, rep = st.gate_report_e4b(fl, tr, 191.25, val_d)
    assert (est.bias_s.to_numpy() == 191.25).all() and list(est.columns) == st.ESTIMATE_COLUMNS
    est2, _ = st.gate_report_e4b(fl, tr, 291.25, val_d)
    f = est.gate_fired.to_numpy()
    np.testing.assert_allclose(est2.taxi_hat.to_numpy()[f] - est.taxi_hat.to_numpy()[f], 100.0, atol=1e-6)
    assert rep["mode"] == "e4b" and rep["verdict"] == "WORKING" and rep["clauses"] == val_d["clauses"]
    assert rep["bias_s"] == 191.25 and rep["prereg"] == st.PREREG_E4B and rep["arm"] == st.ARM_E4B
    dg = rep["diagnostic_2026"]
    assert "pass" not in dg and "pass" not in rep["join_2026"] and dg["n_matched"] == int(est.matched.sum())
    g = est.matched.to_numpy() & f
    a3 = fl.ref_s.to_numpy()
    assert dg["rmse_vs_aobt3_s"] == pytest.approx(st.rmse((est.offblock_raw_s.to_numpy() - 191.25 - a3)[g]), abs=1e-9)
    assert sum(r["n_gated"] for r in rep["per_day"].values()) == dg["n_gated"] == int(g.sum())
    _, rep_inc = st.gate_report_e4b(fl, tr, 191.25, _validation_dict("INCONCLUSIVE", 191.25))
    assert rep_inc["verdict"] == "INCONCLUSIVE"
    json.dumps(st._json_ready(rep))


@pytest.fixture()
def gate_dir(storm, tmp_path):
    d = tmp_path / "g"
    d.mkdir()
    storm["rank"].to_parquet(d / "ranking.parquet", index=False)
    _write_day_files(storm["samples"], st.DAYS, d)
    (d / "bias.json").write_text(json.dumps({"prereg": st.PREREG_E4B, "calibration_day": st.CENSUS_DAY, "bias_s": 191.25}))
    return d


def _gate(d, **kw):
    args = dict(ranking_path=d / "ranking.parquet", adsb_dir=d, bias_path=d / "bias.json", validate_path=d / "val.json",
                report_path=d / "rep.json", estimates_path=d / "est.parquet")
    args.update(kw)
    return st.step_gate_e4b(**args)


def test_step_gate_e4b_refuses_without_a_working_validation_and_reads_no_label(gate_dir, monkeypatch):
    """The 2026 path runs only if the validation verdict is WORKING and was reached with the same constant; the ranking
    file (which here DOES carry both label columns) is read for FLIGHT_COLUMNS only.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again):
    require_e4b_working bypassed in step_gate_e4b -> RED; the validation-bias equality check removed -> RED.
    """
    d = gate_dir
    with pytest.raises(SystemExit, match="run `adsb_storm.py validate` first"):
        _gate(d)
    for verdict in ("INCONCLUSIVE", "NOT WORKING"):
        (d / "val.json").write_text(json.dumps(_validation_dict(verdict, 191.25)))
        with pytest.raises(SystemExit, match="runs only if WORKING"):
            _gate(d)
    (d / "val.json").write_text(json.dumps(_validation_dict("WORKING", 191.0)))
    with pytest.raises(SystemExit, match="the validation ran with bias 191.0"):
        _gate(d)
    assert not (d / "rep.json").exists() and not (d / "est.parquet").exists()
    (d / "val.json").write_text(json.dumps(_validation_dict("WORKING", 191.25)))
    seen, real = [], st.pq.read_table

    def spy(path, *a, columns=None, **kw):
        if isinstance(path, (str, pathlib.Path)) and pathlib.Path(path).name == "ranking.parquet":
            seen.append(None if columns is None else list(columns))
        return real(path, *a, columns=columns, **kw)

    monkeypatch.setattr(st.pq, "read_table", spy)
    assert _gate(d) == 0
    assert seen == [st.FLIGHT_COLUMNS] and not set(st.LABEL_COLUMNS) & set(st.FLIGHT_COLUMNS)
    est = pd.read_parquet(d / "est.parquet")
    rep = json.loads((d / "rep.json").read_text())
    assert (est.bias_s == 191.25).all() and rep["mode"] == "e4b" and rep["verdict"] == "WORKING" and len(est) == 240


# =============================================================================================
# ship --e4b
# =============================================================================================

@pytest.fixture(scope="module")
def e4bship(storm, tmp_path_factory):
    rng = np.random.default_rng(13)
    bias = 191.25
    val_d = _validation_dict("WORKING", bias)
    est, rep = st.gate_report_e4b(storm["flights"], storm["tracks"], bias, val_d)
    est_e4, rep_e4 = st.gate_report(storm["flights"], storm["tracks"], st.clause_c3(150.0, 0.6, 300))
    d = tmp_path_factory.mktemp("e4b_ship")
    other = np.arange(5_000_000, 5_000_060, dtype="float64")
    ids = np.concatenate([est.MVT_ID_mvt.to_numpy(), other])
    base = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": rng.integers(300, 2000, len(ids)).astype("int32")})
    paths = {k: d / v for k, v in dict(base="merry-quicksand_v10.parquet", rank="ranking.parquet", tmpl="submitting.parquet",
                                        est="e4b_est.parquet", rep="e4b_rep.json", est_e4="e4_est.parquet",
                                        rep_e4="e4_rep.json", val="val.json", bias="bias.json",
                                        detail="merry-quicksand_v9.unm_rows.parquet").items()}
    base.to_parquet(paths["base"], index=False)
    storm["rank"].to_parquet(paths["rank"], index=False)
    base[["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=0).to_parquet(paths["tmpl"], index=False)
    est.to_parquet(paths["est"], index=False)
    paths["rep"].write_text(json.dumps(st._json_ready(rep)))
    est_e4.to_parquet(paths["est_e4"], index=False)
    paths["rep_e4"].write_text(json.dumps(st._json_ready(rep_e4)))
    paths["val"].write_text(json.dumps(val_d))
    paths["bias"].write_text(json.dumps({"prereg": st.PREREG_E4B, "calibration_day": st.CENSUS_DAY, "bias_s": bias}))
    eligible = st.eligible_unmatched_ids(paths["rank"])
    b = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    det_ids = np.asarray(sorted(eligible))
    pd.DataFrame({"MVT_ID_mvt": det_ids, "after": b.loc[det_ids].to_numpy(dtype="float64")}).to_parquet(paths["detail"], index=False)
    (d / "LICENCE_adsb.md").write_text("Derived from adsb.lol globe_history, ODbL-1.0, share-alike.\n")
    return dict(d=d, p=paths, base=base, est=est, est_e4=est_e4, eligible=eligible, bias=bias)


def _ship4b(w, version, confirmed, sub_dir, **kw):
    p = w["p"]
    args = dict(detail=str(p["detail"]), licence_doc="LICENCE_adsb.md", report_path=p["rep"], estimates_path=p["est"],
                sub_dir=sub_dir, ranking_path=p["rank"], template_path=p["tmpl"], root=w["d"], e4b=True,
                validate_path=p["val"], bias_path=p["bias"])
    args.update(kw)
    return st.step_ship(str(p["base"]), version, confirmed, **args)


def test_step_ship_e4b_keeps_the_eligibility_and_licence_refusals(e4bship, tmp_path, capsys):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    eligibility-flag check removed from step_ship -> RED (also caught by test_adsb_storm's own E4 test).
    """
    w = e4bship
    with pytest.raises(SystemExit, match="eligibility-confirmed"):
        _ship4b(w, 71, False, tmp_path / "sub")
    out = capsys.readouterr().out
    assert "NO SUBMISSION MAY DEPEND ON THIS ARM" in out and "E4b" in out and "inherits this eligibility rule" in out
    with pytest.raises(SystemExit, match="does not exist"):
        _ship4b(w, 71, True, tmp_path / "sub", licence_doc="MISSING.md")
    with pytest.raises(SystemExit, match="eligibility-confirmed"):
        st.main(["ship", "--e4b", "--base", str(w["p"]["base"]), "--version", "71", "--detail", str(w["p"]["detail"]),
                 "--licence-doc", str(w["d"] / "LICENCE_adsb.md")])
    assert not (tmp_path / "sub").exists() and not (ROOT / "submissions" / "merry-quicksand_v71.parquet").exists()


def test_step_ship_e4b_refuses_e4_artifacts_a_non_working_validation_and_a_foreign_bias(e4bship, tmp_path):
    """ship --e4b must ship the `gate --e4b` output: an E4 report or E4 estimates refuse, a validation file that is not
    WORKING refuses, a bias file that disagrees with the report refuses; E4's ship refuses an E4b report. Nothing written.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the estimates'
    constant check removed -> RED; the validation-file check (require_e4b_working) removed from step_ship -> RED; the
    report-mode check removed -> RED (the e4b ship of the E4 report is then refused later, by the bias check, so the RED
    is the message mismatch: the bias check is a second line behind the mode check).
    """
    w, p = e4bship, e4bship["p"]
    sub = tmp_path / "sub"
    with pytest.raises(SystemExit, match="is a 'e4' gate report and this is the e4b ship"):
        _ship4b(w, 72, True, sub, report_path=p["rep_e4"])
    with pytest.raises(SystemExit, match="is a 'e4b' gate report and this is the e4 ship"):
        _ship4b(w, 72, True, sub, e4b=False)
    with pytest.raises(SystemExit, match="does not carry the calibrated constant"):
        _ship4b(w, 72, True, sub, estimates_path=p["est_e4"])
    bad_val = tmp_path / "val_inc.json"
    bad_val.write_text(json.dumps(_validation_dict("INCONCLUSIVE", w["bias"])))
    with pytest.raises(SystemExit, match="runs only if WORKING"):
        _ship4b(w, 72, True, sub, validate_path=bad_val)
    other_bias = tmp_path / "bias2.json"
    other_bias.write_text(json.dumps({"prereg": st.PREREG_E4B, "calibration_day": st.CENSUS_DAY, "bias_s": w["bias"] + 1}))
    with pytest.raises(SystemExit, match="is not the calibrated constant"):
        _ship4b(w, 72, True, sub, bias_path=other_bias)
    assert not sub.exists()


def test_step_ship_e4b_ships_the_constant_estimates_under_the_e4b_arm(e4bship, tmp_path):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): step_ship
    passing E4's arm to write_ship_outputs in e4b mode -> RED.
    """
    w = e4bship
    assert _ship4b(w, 73, True, tmp_path) == 0
    out = pd.read_parquet(tmp_path / "merry-quicksand_v73.parquet")
    want, _ = st.build_ship(w["base"], w["est"], w["eligible"])
    pd.testing.assert_frame_equal(out, want)
    meta = json.loads((tmp_path / "merry-quicksand_v73.meta.json").read_text())
    assert meta["arm"] == st.ARM_E4B and meta["prereg"] == st.PREREG_E4B and meta["bias_s"] == w["bias"]
    assert meta["validation_verdict"] == "WORKING" and meta["gate_verdict"] == "WORKING"
    assert meta["eligibility_confirmed_by_owner"] is True and meta["licence_doc"] == "LICENCE_adsb.md"
    assert meta["transfer_rule"] == st.TRANSFER_RULE and meta["expected_price"] == st.EXPECTED_PRICE
    # the shipped values are the constant's: rint(MVT - (raw - bias)) floored at 1, not E4's bias
    rows = st.ship_rows(w["est"]).set_index("MVT_ID_mvt")
    o = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    np.testing.assert_array_equal(o.loc[rows.index].to_numpy(), rows.after.to_numpy().astype("int32"))
    e4_rows = st.ship_rows(w["est_e4"]).set_index("MVT_ID_mvt")
    common = rows.index.intersection(e4_rows.index)
    assert len(common) > 20 and (rows.after.loc[common] != e4_rows.after.loc[common]).any()


# =============================================================================================
# extract: the allowlist and the 2026 ingest gate
# =============================================================================================

def test_step_extract_allows_the_validation_days_and_gates_the_2026_ingest_on_a_working_validation(tmp_path, monkeypatch):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    validation days dropped from the allowlist -> RED; the 2026 full-extract gate removed -> RED.
    """
    calls = []

    def fetch(day, probe):
        p = tmp_path / f"{day}.tar.aa{'.probe' if probe else ''}"
        p.write_bytes(b"x")
        calls.append((day, probe))
        return [p]

    lat, lon = st.ax.APTS["EHAM"]
    t0 = (pd.Timestamp("2025-12-27 08:00", tz="UTC") - st.EPOCH).total_seconds()
    df = pd.DataFrame({"icao": ["a", "a", "b"], "reg": None, "type": None, "callsign": ["KLM1", "KLM1", "SWR2"],
                       "airport": ["EHAM", "EHAM", "LSZH"], "t": [t0, t0 + 5, t0], "lat": lat, "lon": lon,
                       "on_ground": 1, "alt": np.nan, "gs": 0.0})
    monkeypatch.setattr(st.ax, "fetch", fetch)
    monkeypatch.setattr(st.ax, "extract", lambda paths, probe: df)
    monkeypatch.setattr(st, "ADSB_DIR", tmp_path / "adsb")
    vp = tmp_path / "val.json"
    assert st.step_extract("2025-12-27", False, validate_path=vp) == 0
    out = pd.read_parquet(tmp_path / "adsb" / "adsb_2025-12-27_EHAM.parquet")
    assert len(out) == 2 and (out.airport == "EHAM").all() and not (tmp_path / "2025-12-27.tar.aa").exists()
    for day in ("2025-12-26", "2025-01-09", "2026-01-10"):
        with pytest.raises(SystemExit, match="not a registered day"):
            st.step_extract(day, False, validate_path=vp)
    assert calls == [("2025-12-27", False)]
    with pytest.raises(SystemExit, match="run `adsb_storm.py validate` first"):
        st.step_extract("2026-01-02", False, validate_path=vp)
    vp.write_text(json.dumps(_validation_dict("NOT WORKING", 1.0)))
    with pytest.raises(SystemExit, match="runs only if WORKING"):
        st.step_extract("2026-01-02", False, validate_path=vp)
    assert calls == [("2025-12-27", False)] and not (tmp_path / "adsb" / "adsb_2026-01-02_EHAM.parquet").exists()
    assert st.step_extract("2026-01-05", True, validate_path=vp) == 0                  # a probe writes nothing
    assert not (tmp_path / "adsb" / "adsb_2026-01-05_EHAM.parquet").exists()
    vp.write_text(json.dumps(_validation_dict("WORKING", 1.0)))
    assert st.step_extract("2026-01-02", False, validate_path=vp) == 0
    assert (tmp_path / "adsb" / "adsb_2026-01-02_EHAM.parquet").exists()
    assert st.step_extract("2025-12-28", True, validate_path=vp) == 0
    assert not (tmp_path / "adsb" / "adsb_2025-12-28_EHAM.parquet").exists() and not list(tmp_path.glob("*.tar.aa*"))


# =============================================================================================
# write-once: the calibration and the verdict
# =============================================================================================

def test_step_calibrate_writes_once_and_a_rerun_must_reproduce(census, tmp_path):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the refusal
    to replace a different constant removed (the file overwritten) -> RED.
    """
    p = tmp_path / "bias.json"
    assert st.step_calibrate(p, census["rank"], census["samples"]) == 0
    j = json.loads(p.read_text())
    assert j["bias_s"] == census["cal"]["bias_s"] and j["n"] == 21 and j["iqr_s"] == census["cal"]["iqr_s"]
    assert j["cross_check_census_c3_block_calibrated_bias_s"] == j["bias_s"] and j["calibration_day"] == st.CENSUS_DAY
    assert st.load_bias(p) == census["cal"]["bias_s"]
    raw = p.read_bytes()
    assert st.step_calibrate(p, census["rank"], census["samples"]) == 0 and p.read_bytes() == raw    # reproduced, untouched
    p.write_text(json.dumps({**j, "bias_s": j["bias_s"] + 0.5}))
    tampered = p.read_bytes()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        st.step_calibrate(p, census["rank"], census["samples"])
    assert p.read_bytes() == tampered


def test_step_validate_reads_the_files_writes_once_and_never_revises(census, val, tmp_path):
    """The on-disk step: the bias file's constant (with its n and IQR) drives the run, the verdict and the per-row table
    are written, and a second run refuses to overwrite.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the overwrite
    refusal removed from step_validate -> RED.
    """
    d = tmp_path
    st.step_calibrate(d / "bias.json", census["rank"], census["samples"])
    val["rank"].to_parquet(d / "training.parquet", index=False)
    _write_s1c(val["s1c"], d / "s1c.parquet")
    _write_day_files(val["samples"], st.VALIDATION_DAYS, d)
    args = dict(report_path=d / "val.json", rows_path=d / "rows.parquet", bias_path=d / "bias.json",
                truth_path=d / "training.parquet", s1c_path=d / "s1c.parquet", adsb_dir=d)
    assert st.step_validate(**args) == 0
    rep = json.loads((d / "val.json").read_text())
    rows = pd.read_parquet(d / "rows.parquet")
    assert rep["verdict"] == val["rep"]["verdict"] == "WORKING"
    assert rep["bias"] == {"bias_s": census["cal"]["bias_s"], "source_file": "bias.json", "calibration_day": st.CENSUS_DAY,
                           "n": 21, "iqr_s": census["cal"]["iqr_s"]}
    assert rep["clauses"]["V1"]["rmse_s"] == pytest.approx(val["rep"]["clauses"]["V1"]["rmse_s"], abs=1e-9)
    assert rep["clauses"]["V2"]["n_rows"] == val["rep"]["clauses"]["V2"]["n_rows"] and len(rows) == len(val["rows"])
    assert st.require_e4b_working(d / "val.json")["verdict"] == "WORKING"
    before = (d / "val.json").read_bytes()
    with pytest.raises(SystemExit, match="never revised"):
        st.step_validate(**args)
    assert (d / "val.json").read_bytes() == before


def test_main_routes_the_e4b_steps_and_leaves_e4s_gate_and_ship_on_their_own_path(monkeypatch):
    """`gate` stays E4's, `gate --e4b` is E4b's; `ship --e4b` passes e4b=True and plain `ship` does not.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): main() routing
    `gate --e4b` to E4's step_gate -> RED; main() dropping --e4b from the ship call -> RED.
    """
    calls = []
    for name in ("step_gate", "step_gate_e4b", "step_calibrate", "step_validate", "step_census"):
        monkeypatch.setattr(st, name, lambda *a, _n=name, **k: calls.append((_n, a, k)) or 0)
    monkeypatch.setattr(st, "step_ship", lambda *a, **k: calls.append(("step_ship", a, k)) or 0)
    for argv in (["gate"], ["gate", "--e4b"], ["calibrate"], ["validate"], ["census"]):
        assert st.main(argv) == 0
    assert [c[0] for c in calls] == ["step_gate", "step_gate_e4b", "step_calibrate", "step_validate", "step_census"]
    ship = ["ship", "--base", "b.parquet", "--version", "9", "--detail", "d.parquet", "--licence-doc", "L.md"]
    calls.clear()
    st.main(ship + ["--e4b", "--eligibility-confirmed"])
    st.main(ship)
    assert calls[0][1] == ("b.parquet", 9, True, "d.parquet", "L.md") and calls[0][2] == {"e4b": True}
    assert calls[1][1] == ("b.parquet", 9, False, "d.parquet", "L.md") and calls[1][2] == {"e4b": False}
