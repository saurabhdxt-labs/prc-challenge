"""Tests for scripts/analysis/ldi_correction.py — the LTFM de-icing (LDI) severity-cell correction.

Synthetic fixtures only; no real data is read here. Mutation rehearsals are recorded in the docstrings ("Fails when").
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("ldi_correction", "scripts/analysis/ldi_correction.py")


def test_bands_are_minutes_with_right_closed_edges():
    """sp in SECONDS; bands (-inf, 30], (30, 60], (60, inf) minutes. Fails when sp is read as minutes or an edge
    moves (1800 s is '<30', 1801 s is '30-60', 3600 is '30-60', 3601 is '>60')."""
    b = C.band(np.array([-600, 0, 1800, 1801, 3600, 3601, 90000]))
    assert b.tolist() == ["<30", "<30", "<30", "30-60", "30-60", ">60", ">60"]
    with pytest.raises(ValueError, match="non-finite"):
        C.band(np.array([np.nan]))


def test_local_day_is_utc_plus_3():
    """LTFM local day = UTC + 3 h (Turkey, no DST since 2016). 2025-02-19 21:30Z is 2025-02-20 local."""
    t = (pd.DatetimeIndex(["2025-02-19T20:59:00Z", "2025-02-19T21:00:00Z"]) - pd.Timestamp("1970-01-01T00:00:00Z")
         ) // pd.Timedelta(seconds=1)
    assert [str(d) for d in C.local_day(t)] == ["2025-02-19", "2025-02-20"]


def _day_frame():
    # day A: 4 rows, 3 with sp > 30 min (share 0.75), min temp 0 -> severe
    # day B: 4 rows, 2 with sp > 30 min (share 0.5), min temp 1 -> severe (edges inclusive)
    # day C: 4 rows, 3 delayed but min temp 2 -> not severe; day D: cold but share 0.25 -> not severe
    return pd.DataFrame({
        "day": ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4,
        "sp": [3000, 3000, 3000, 0, 1801, 1801, 1800, 0, 4000, 4000, 4000, 0, 4000, 0, 0, 0],
        "w_temp_c": [0, 3, 3, 3, 1, 4, 4, 4, 2, 2, 2, 2, -5, -5, -5, -5]})


def test_severe_days_edges_inclusive_and_both_conditions_needed():
    """Severe = share(sp > 1800 s) >= 0.5 AND min temp <= 1 C, per local day. Fails when either condition is dropped,
    an inequality made strict, or sp > 1800 becomes >= 1800 (day B would have share 0.75, still severe; day D with
    share 0.25 must stay mild)."""
    d = _day_frame()
    s = C.severe_days(d.day, d.sp, d.w_temp_c)
    assert s.to_dict() == {"A": True, "B": True, "C": False, "D": False}
    e = d.copy()
    e.loc[e.day == "B", "sp"] = [1801, 1800, 1800, 0]        # share 0.25 (1800 s is not > 1800) -> not severe
    assert not C.severe_days(e.day, e.sp, e.w_temp_c)["B"]


def test_severe_days_ignore_missing_temperature_but_refuse_all_missing():
    d = pd.DataFrame({"day": ["A", "A"], "sp": [4000, 4000], "w_temp_c": [np.nan, 0.0]})
    assert C.severe_days(d.day, d.sp, d.w_temp_c)["A"]
    with pytest.raises(ValueError, match="temperature"):
        C.severe_days(d.day, d.sp, pd.Series([np.nan, np.nan]))


def _frame(seed=0, n_days=6, gap=(0.0, 400.0, 1000.0), n=40, base_bias=50.0):
    """Synthetic Feb-like OOF frame: per severe day, n target rows per band with residual gap_b + base_bias + noise;
    non-freezing rows with residual base_bias + noise (so the difference-form cell gap is gap_b)."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for b, g in zip(C.BANDS, gap):
            for tgt in (True, False):
                r = (g if tgt else 0.0) + base_bias + rng.normal(0, 30, n)
                p = rng.uniform(600, 900, n)
                rows.append(pd.DataFrame({"day": f"d{d}", "band": b, "target": tgt, "nonfrz": not tgt,
                                          "p": p, "y": p + r}))
    return pd.concat(rows, ignore_index=True)


def test_cell_gaps_difference_form_removes_the_month_bias():
    """Delta_b = mean residual on target rows - mean residual on non-freezing rows, per band. Fails when the
    non-freezing mean is not subtracted (would add base_bias 50) or bands are pooled."""
    f = _frame(base_bias=50.0)
    g = C.cell_gaps(f.y, f.p, f.band, f.target, f.nonfrz)
    assert g["<30"] == pytest.approx(0.0, abs=6) and g["30-60"] == pytest.approx(400, abs=6)
    assert g[">60"] == pytest.approx(1000, abs=6)


def test_cell_gaps_empty_band_is_zero_not_nan():
    f = _frame()
    f = f[~((f.band == ">60") & f.target)]
    g = C.cell_gaps(f.y, f.p, f.band, f.target, f.nonfrz)
    assert g[">60"] == 0.0


def test_losdo_gains_are_out_of_day_and_in_bmse_units():
    """Each day's correction is fitted on the OTHER days (difference form), shrunk, and applied to that day's target
    rows only; the gain is (SSE before - SSE after) / 344,841. Checked against a hand computation.
    Fails when day d's own rows fit its correction (in-sample: the gain would rise), when non-target rows are
    corrected, or when the shrink is ignored."""
    f = _frame(seed=1)
    out = C.losdo_gains(f, shrink=0.5)
    assert sorted(out) == [f"d{i}" for i in range(6)]
    d0 = f[f.day == "d0"]
    rest = f[f.day != "d0"]
    g = C.cell_gaps(rest.y, rest.p, rest.band, rest.target, rest.nonfrz)
    c = np.where(d0.target, d0.band.map(g).astype(float) * 0.5, 0.0)
    want = (((d0.y - d0.p) ** 2).sum() - ((d0.y - d0.p - c) ** 2).sum()) / C.BOARD_N
    assert out["d0"] == pytest.approx(want, rel=1e-12)
    assert out["d0"] > 0


def test_losdo_gain_negative_when_the_gap_is_absent():
    """No gap on target rows -> the out-of-day corrections are noise-sized; with an injected per-day sign flip the
    corrections hurt. A harness that could not go negative would fail here."""
    f = _frame(seed=2, gap=(0.0, 400.0, 1000.0))
    flip = f.target & (f.day == "d3")
    f.loc[flip, "y"] = f.loc[flip, "p"] + 50.0 - (f.loc[flip, "y"] - f.loc[flip, "p"] - 50.0)
    out = C.losdo_gains(f, shrink=0.5)
    assert out["d3"] < 0 and out["d0"] > 0


def test_day_bootstrap_ci_is_seeded_and_brackets_the_total():
    """Mutation rehearsal 2026-09-27 (logs/ldi_scratch/mutation_ldi_correction.log): 19 mutants over bands, local day,
    severity edges, difference form, out-of-day fit, shrink, target-only correction, floor, adj sign, id guards, gain
    range, all-NaN temperature: 18/19 KILLED at first; the 25th-percentile CI mutant escaped -> this exact-percentile
    assertion was added and kills it."""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    lo, hi = C.day_bootstrap_ci(vals, n_boot=2000, seed=7)
    assert lo < vals.sum() < hi and lo > 0
    assert C.day_bootstrap_ci(vals, n_boot=2000, seed=7) == (lo, hi)
    lo2, _ = C.day_bootstrap_ci(np.array([-5.0, 1.0, 1.0, 1.0]), n_boot=2000, seed=7)
    assert lo2 < 0
    rng = np.random.default_rng(7)                          # the 2.5 / 97.5 percentiles of resampled day sums
    tot = vals[rng.integers(0, 6, size=(2000, 6))].sum(axis=1)
    assert (lo, hi) == (np.percentile(tot, 2.5), np.percentile(tot, 97.5))


def test_transfer_adjustment_is_the_storm_seen_minus_storm_unseen_excess():
    """adj_b = mean(p_seen - p_unseen) on target rows of band b - the same on non-freezing rows of band b."""
    band = pd.Series(["30-60"] * 4 + [">60"] * 4)
    tgt = pd.Series([True, True, False, False] * 2)
    nonfrz = ~tgt
    p_unseen = pd.Series([100.0] * 8)
    p_seen = pd.Series([400, 400, 150, 150, 700, 900, 100, 100], dtype=float)
    a = C.transfer_adjustment(p_seen, p_unseen, band, tgt, nonfrz)
    assert a["30-60"] == pytest.approx(250.0) and a[">60"] == pytest.approx(700.0) and a["<30"] == 0.0


def test_applied_deltas_shrink_adjust_and_floor_at_zero():
    """Delta_apply_b = max(0, shrink * (Delta_b - adj_b)). Fails when the floor or the shrink is dropped or adj added."""
    d = C.applied_deltas({"<30": -20.0, "30-60": 600.0, ">60": 1500.0}, {"<30": 0.0, "30-60": 100.0, ">60": 1700.0}, 0.5)
    assert d == {"<30": 0.0, "30-60": 250.0, ">60": 0.0}


def test_apply_to_values_changes_only_target_rows():
    """v39 values: v36 + Delta_apply_band on target rows; every other row bit-identical; ids / order / dtype kept.
    Fails when a non-target row moves or the id order changes."""
    v = pd.DataFrame({"MVT_ID_mvt": [5.0, 3.0, 9.0, 1.0], "TAXITIME_SEC_mvt": [600.0, 700.0, 800.0, 900.0]})
    tgt = pd.DataFrame({"MVT_ID_mvt": [3.0, 1.0], "band": ["30-60", ">60"]})
    out = C.apply_to_values(v, tgt, {"<30": 0.0, "30-60": 250.0, ">60": 500.0})
    assert out.MVT_ID_mvt.tolist() == [5.0, 3.0, 9.0, 1.0]
    assert out.TAXITIME_SEC_mvt.tolist() == [600.0, 950.0, 800.0, 1400.0]
    assert out.dtypes.equals(v.dtypes)
    with pytest.raises(ValueError, match="not in the values"):
        C.apply_to_values(v, pd.DataFrame({"MVT_ID_mvt": [42.0], "band": [">60"]}), {">60": 1.0})
    with pytest.raises(ValueError, match="duplicate"):
        C.apply_to_values(v, pd.DataFrame({"MVT_ID_mvt": [3.0, 3.0], "band": [">60", ">60"]}), {">60": 1.0})


def test_gain_range_bmse():
    """Board-gain range for Delta applied on n rows with true gap g in [0, g_max]: low = -n Delta^2 / N (g = 0),
    high = n (2 g_max Delta - Delta^2) / N. With Delta = g_max / 2: high = 3 * |low|."""
    lo, hi = C.gain_range({"x": 1000}, {"x": 500.0}, {"x": 1000.0})
    assert lo == pytest.approx(-1000 * 500.0 ** 2 / C.BOARD_N)
    assert hi == pytest.approx(1000 * (2 * 1000 * 500 - 500 ** 2) / C.BOARD_N) and hi == pytest.approx(-3 * lo)
    assert C.rmse_after(250.0, 0.0) == 250.0 and C.rmse_after(250.0, 100.0) == pytest.approx(np.sqrt(250.0 ** 2 - 100))


# ---------------------------------------------------------------- frame assembly

def _t(s):
    return (pd.DatetimeIndex(s) - pd.Timestamp("1970-01-01T00:00:00Z")) // pd.Timedelta(seconds=1)


def test_to_epoch_s_accepts_numeric_seconds_and_datetimes_of_any_unit():
    s = pd.Series(pd.to_datetime(["2025-02-19T21:00:00"]).as_unit("us"))
    assert C.to_epoch_s(s).tolist() == [int(_t(["2025-02-19T21:00:00Z"])[0])]
    assert C.to_epoch_s(pd.Series([1740000000])).tolist() == [1740000000]
    with pytest.raises(ValueError, match="epoch"):
        C.to_epoch_s(pd.Series([1740000000000]))            # milliseconds read as seconds -> year 57,000


def _pred_dep_wx():
    t = _t(["2025-02-19T08:00:00Z"] * 4 + ["2025-02-20T08:00:00Z"] * 2)
    pred = pd.DataFrame({"MVT_ID_mvt": [1.0, 2, 3, 4, 5, 6, 7], "ap": ["LTFM"] * 6 + ["EGLL"],
                         "month": 2, "y": [900.0, 800, 700, 600, 500, 400, 300],
                         "taxi_r12": [600.0, 600, 600, 600, 600, 600, 600]})
    dep = pd.DataFrame({"MVT_ID_mvt": [1.0, 2, 3, 4, 5, 6, 7], "PHASE_mvt": ["DEP"] * 5 + ["ARR", "DEP"],
                        "ADEP_mvt": ["LTFM"] * 6 + ["EGLL"], "y": [900.0, 800, 700, 600, 500, 400, 300],
                        "MVT_TIME_UTC_mvt": list(t) + [int(t[0])],
                        "SCHED_TIME_UTC_mvt": [int(x) - 3000 for x in t] + [int(t[0])]})
    wx = pd.DataFrame({"MVT_ID_mvt": [1.0, 2, 3, 4, 5, 6, 7], "w_freezing": [1, 1, 0, np.nan, 1, 0, 0],
                       "w_temp_c": [0.0, 0.0, 1.0, np.nan, 5.0, 5.0, 5.0]})
    return pred, dep, wx


def test_frame_2025_keeps_ltfm_departures_labels_guarded_and_flags_targets():
    """LTFM DEP rows only (ARR row 6 and EGLL row 7 dropped); sp = MVT - SCHED; severity per local day; missing
    weather is neither target nor non-freezing. Fails when arrivals or other airports enter, or NaN weather counts as
    non-freezing."""
    pred, dep, wx = _pred_dep_wx()
    f = C.frame_2025(pred, dep, wx)
    assert f.MVT_ID_mvt.tolist() == [1.0, 2, 3, 4, 5]
    assert f.sp.tolist() == [3000] * 5 and set(f.band) == {"30-60"}
    assert f.target.tolist() == [True, True, False, False, False]      # 02-20 has min temp 5: not severe
    assert f.nonfrz.tolist() == [False, False, True, False, False]
    assert (f.p == 600.0).all() and f.day.astype(str).tolist()[:1] == ["2025-02-19"]


def test_frame_2025_refuses_label_mismatch_and_duplicate_ids():
    pred, dep, wx = _pred_dep_wx()
    bad = dep.copy()
    bad.loc[0, "y"] = 1.0
    with pytest.raises(AssertionError, match="label"):
        C.frame_2025(pred, bad, wx)
    with pytest.raises(AssertionError, match="duplicate"):
        C.frame_2025(pred, pd.concat([dep, dep.iloc[:1]]), wx)


def test_frame_2026_joins_both_2026_predictions_and_the_base_values_for_january_matched_rows():
    """January LTFM matched DEP rows; p_seen / p_unseen / base joined by MVT id; a row missing any of them is refused
    (the correction must never touch a row it cannot price). July and unmatched rows are dropped."""
    t = _t(["2026-01-12T08:00:00Z"] * 3 + ["2026-07-12T08:00:00Z"])
    rank = pd.DataFrame({"MVT_ID_mvt": [11.0, 12, 13, 14], "PHASE_mvt": "DEP", "ADEP_mvt": "LTFM",
                         "MVT_TIME_UTC_mvt": pd.to_datetime(t, unit="s").as_unit("us"),
                         "SCHED_TIME_UTC_mvt": pd.to_datetime(t - 4000, unit="s").as_unit("us"),
                         "AOBT_3_flt": [1.0, np.nan, 1.0, 1.0]})
    wx = pd.DataFrame({"MVT_ID_mvt": [11.0, 12, 13, 14], "w_freezing": [1, 1, 0, 1], "w_temp_c": [0.0, 0, 0, 0]})
    seen = pd.DataFrame({"MVT_ID_mvt": [11.0, 13, 14], "taxi_r12": [900.0, 700, 1.0]})
    unseen = pd.DataFrame({"MVT_ID_mvt": [11.0, 13, 14], "taxi_r12": [800.0, 690, 1.0]})
    base = pd.DataFrame({"MVT_ID_mvt": [11.0, 12, 13, 14], "TAXITIME_SEC_mvt": [1000.0, 1, 900, 1]})
    f = C.frame_2026(rank, wx, seen, unseen, base)
    assert f.MVT_ID_mvt.tolist() == [11.0, 13] and f.band.tolist() == [">60", ">60"]
    assert f.target.tolist() == [True, False] and f.nonfrz.tolist() == [False, True]
    assert f.p_seen.tolist() == [900.0, 700] and f.p_unseen.tolist() == [800.0, 690] and f.base.tolist() == [1000.0, 900]
    with pytest.raises(AssertionError, match="missing"):
        C.frame_2026(rank, wx, seen.iloc[1:], unseen, base)
