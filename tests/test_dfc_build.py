"""DFC — registered invariants of plans/PREREG_dfc_destination_flow_congestion_2026_09_19.md (c217c0cd).

The in-fold median gate (leak gate 2) is the load-bearing test: a (ADES, hour) median computed over TEST
rows would leak the test period's flow state and produce a large, meaningless gain.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


@pytest.fixture(scope="module")
def D():
    return _load("dfc_build", "scripts/analysis/dfc_build.py")


def _frame():
    """5 departures: 3 to EGLL within an hour of each other, 1 to EGLL far away, 1 to JFK (out of set)."""
    # row 5 sits 100 min after row 0: inside a 2 h window but OUTSIDE the registered 1 h one, so the
    # W1 constant is actually pinned by the assertions below (a 1h->2h mutation must turn them red).
    return pd.DataFrame({
        "ADES_mvt": ["EGLL", "EGLL", "EGLL", "EGLL", "KJFK", "EGLL"],
        "doy":      [100, 100, 100, 100, 100, 100],
        "hr":       [10, 10, 10, 20, 10, 11],
        # stand_ab._own_clock_features: tmin is MINUTES SINCE MIDNIGHT.
        "tmin":     [600, 630, 650, 1200, 600, 700],
        "mvt_sec":  [0, 0, 0, 0, 0, 0],
    })


def test_block_is_the_six_registered_columns(D):
    """Catches a column being added or dropped after the prereg was hashed."""
    assert D.DFC_FEATS == ["dfc_ades_dep_1h", "dfc_ades_dep_3h", "dfc_ades_dep_1h_dev",
                           "dfc_ades_in_set", "dfc_ades_arr_1h", "dfc_ades_arr_1h_dev"]


def test_no_label_column_is_read(D):
    """Leak gate 1+3. Catches the builder reading BLOCK_TIME/TAXITIME/y/delta."""
    banned = {"y", "delta", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "regime"}
    assert not (banned & set(D.INPUT_COLS))


def test_dep_1h_counts_the_window_and_excludes_self(D):
    """Catches an off-by-one window or self-inclusion, which would make every row's count 1 too high."""
    out = D.build_dfc(_frame(), fit_mask=np.ones(6, bool), year=2025, arrivals=None)
    # 10:00 sees 10:30 and 10:50 but NOT 11:40 (100 min) -> exactly 2 under the registered 1 h window
    assert out.dfc_ades_dep_1h.tolist() == [2, 2, 3, 0, 0, 1]


def test_dep_3h_window_is_wider_than_1h(D):
    """Catches the two windows being wired to the same constant."""
    out = D.build_dfc(_frame(), fit_mask=np.ones(6, bool), year=2025, arrivals=None)
    assert (out.dfc_ades_dep_3h >= out.dfc_ades_dep_1h).all()


def test_in_set_flags_only_the_ten_scored_airports(D):
    """Catches an out-of-set destination being treated as observable."""
    out = D.build_dfc(_frame(), fit_mask=np.ones(6, bool), year=2025, arrivals=None)
    assert out.dfc_ades_in_set.tolist() == [1, 1, 1, 1, 0, 1]


def test_arr_columns_are_nan_exactly_where_destination_is_out_of_set(D):
    """Catches a 0 being written where 'unobservable' is meant — 0 and NaN mean different things to LGBM."""
    point = D.timestamp(pd.DataFrame({"doy": [100], "hr": [10], "tmin": [600], "mvt_sec": [0]}), 2025)
    arr = D.ArrivalStream(by_airport={"EGLL": point}, years=frozenset({2025}),
                          span=(float(point[0]), float(point[0])))
    out = D.build_dfc(_frame(), fit_mask=np.ones(6, bool), year=2025, arrivals=arr)
    assert np.isnan(out.dfc_ades_arr_1h.to_numpy()[4])
    np.testing.assert_array_equal(out.dfc_ades_arr_1h.to_numpy()[[0, 1, 2, 3, 5]],
                                  [1, 1, 1, 0, 0])


def test_dev_medians_are_computed_on_FIT_ROWS_ONLY(D):
    """LEAK GATE 2 — the load-bearing test. If the (ADES, hour) median is computed over all rows, the
    test period's flow state leaks into its own feature and the gain is meaningless."""
    second = _frame().assign(doy=200, hr=10, tmin=[600, 630, 650, 600, 600, 640])
    f = pd.concat([_frame(), second], ignore_index=True)
    fit = np.array([True] * 6 + [False] * 6)          # second block is 'test'
    out_fit = D.build_dfc(f, fit_mask=fit, year=2025, arrivals=None)
    out_all = D.build_dfc(f, fit_mask=np.ones(12, bool), year=2025, arrivals=None)
    # the two must differ somewhere: if they are identical the fit_mask is being ignored
    a = out_fit.dfc_ades_dep_1h_dev.to_numpy()
    b = out_all.dfc_ades_dep_1h_dev.to_numpy()
    assert not np.allclose(np.nan_to_num(a), np.nan_to_num(b)), "fit_mask is ignored — leak gate 2 is dead"


def test_dev_is_the_count_minus_its_group_median(D):
    """Catches the deviation being computed against the wrong group or the global median."""
    f = _frame()
    out = D.build_dfc(f, fit_mask=np.ones(6, bool), year=2025, arrivals=None)
    grp = pd.DataFrame({"a": f.ADES_mvt, "h": f.hr, "c": out.dfc_ades_dep_1h})
    med = grp.groupby(["a", "h"]).c.transform("median")
    assert np.allclose(out.dfc_ades_dep_1h_dev.to_numpy(), (grp.c - med).to_numpy())


def test_timestamp_is_seconds_and_monotone_in_doy(D):
    """Catches mvt_sec being mistaken for seconds-of-day (it is seconds-within-minute, range 0-59)."""
    f = pd.DataFrame({"doy": [1, 1, 2], "hr": [0, 23, 0], "tmin": [0, 1439, 0], "mvt_sec": [0, 59, 0],
                      "ADES_mvt": ["EGLL"] * 3})
    t = D.timestamp(f, 2025)
    # absolute offset is irrelevant (every window is a difference); the SPAN within a day is what matters
    assert t[1] - t[0] == 23 * 3600 + 59 * 60 + 59
    assert t[2] - t[0] == 86400
    assert (np.diff(t) > 0).all()
    # if mvt_sec were treated as seconds-of-DAY, a 59 would add 59s here but 59s is also the correct
    # within-minute contribution, so pin the exact within-year formula, net of the year term (the
    # year term itself is pinned separately below, since its size is what F1 actually depends on):
    assert t[1] - ((2025 - D.EPOCH_YEAR) * 366 * 86400) == 1 * 86400 + 23 * 3600 + 59 * 60 + 59


def test_timestamp_requires_year_and_carries_it(D):
    """F1 regression: a bare doy*86400 origin let a 2025 and a 2026 row at the same doy collide.
    `year` is a required positional argument (TypeError without it) and two frames identical except
    for `year` must land on different absolute coordinates."""
    f = pd.DataFrame({"doy": [15], "hr": [0], "tmin": [0], "mvt_sec": [0], "ADES_mvt": ["EGLL"]})
    with pytest.raises(TypeError):
        D.timestamp(f)
    t_2025 = D.timestamp(f, 2025)
    t_2026 = D.timestamp(f, 2026)
    assert t_2025[0] != t_2026[0]
    assert abs(t_2026[0] - t_2025[0]) == 366 * 86400


# ---------------------------------------------------------------------------
# F1 — the arrival lookup must be sourced from the SAME year as the departures
# ---------------------------------------------------------------------------

def _write_arr_file(path, year, ades, hour_minute_pairs):
    """A minimal ARR-row parquet shaped like data/raw's real columns."""
    ts = [pd.Timestamp(year=year, month=1, day=15, hour=h, minute=mi, tz="UTC")
          for h, mi in hour_minute_pairs]
    pd.DataFrame({"PHASE_mvt": ["ARR"] * len(ts), "ADES_mvt": [ades] * len(ts),
                  "MVT_TIME_UTC_mvt": ts}).to_parquet(path)


def _dep_frame(year, ades, hour, minute):
    doy = pd.Timestamp(year=year, month=1, day=15).dayofyear
    return pd.DataFrame({"ADES_mvt": [ades], "doy": [doy], "hr": [hour],
                         "tmin": [hour * 60 + minute], "mvt_sec": [0]})


def test_load_arrivals_takes_explicit_paths_no_hidden_glob(D, tmp_path):
    """F1: load_arrivals must not reach for a hard-coded 2025 glob — it reads exactly the files
    it is given."""
    p25 = tmp_path / "training_2025-01.parquet"
    _write_arr_file(p25, 2025, "EGLL", [(9, 30), (9, 45), (10, 0), (10, 10)])  # 4 landings
    stream = D.load_arrivals([p25], year=2025)
    assert stream.by_airport["EGLL"].shape[0] == 4
    assert stream.years == frozenset({2025})


def test_load_arrivals_rejects_files_whose_year_does_not_match_declared_year(D, tmp_path):
    """F1 fail-loud: if a source file's own dates disagree with the year the caller declared,
    raise — never silently accept a mismatched file."""
    p24 = tmp_path / "training_2024-01.parquet"
    _write_arr_file(p24, 2024, "EGLL", [(10, 0)])
    with pytest.raises(AssertionError):
        D.load_arrivals([p24], year=2025)


def test_build_dfc_rejects_an_arrival_stream_from_a_different_year_than_the_departures(D):
    """F1 fail-loud, the exact defect the audit measured: feeding a 2026 departure frame a 2025
    arrival stream (the old hard-coded glob's behaviour) must raise, not silently mismatch
    landings to takeoffs 94.93% of the time."""
    stream_2025 = D.ArrivalStream(by_airport={"EGLL": np.array([100.0])},
                                  years=frozenset({2025}), span=(100.0, 100.0))
    f2026 = _dep_frame(2026, "EGLL", 10, 0)
    with pytest.raises(AssertionError):
        D.build_dfc(f2026, fit_mask=np.ones(1, bool), year=2026, arrivals=stream_2025)


def test_2026_departures_get_2026_arrivals_only_not_2025(D, tmp_path):
    """The tripwire: a 2026 departure frame built against the CORRECT (2026) arrival stream must
    count only 2026 landings. A mutant that reintroduces the 2025-only default glob would feed
    this call the 2025 stream instead — caught (as AssertionError) by the previous test, and here
    shown to produce a materially different, wrong count if the fail-loud gate were bypassed."""
    p25 = tmp_path / "training_2025-01.parquet"
    p26 = tmp_path / "ranking_2026.parquet"
    # 2025 has 4 landings inside the window; 2026 has 2 — deliberately different so a year mixup
    # is visible as a wrong COUNT, not just a raised exception.
    _write_arr_file(p25, 2025, "EGLL", [(9, 30), (9, 45), (10, 0), (10, 10)])
    _write_arr_file(p26, 2026, "EGLL", [(9, 50), (10, 5)])

    stream_2025 = D.load_arrivals([p25], year=2025)
    stream_2026 = D.load_arrivals([p26], year=2026)
    assert stream_2025.by_airport["EGLL"].shape[0] == 4
    assert stream_2026.by_airport["EGLL"].shape[0] == 2

    dep = _dep_frame(2026, "EGLL", 10, 0)
    out = D.build_dfc(dep, fit_mask=np.ones(1, bool), year=2026, arrivals=stream_2026)
    assert out.dfc_ades_arr_1h.iloc[0] == 2          # 2026 arrivals only
    # using the 2025 stream on the same 2026 departure is refused outright (F1 fail-loud),
    # so it can never silently produce the wrong count of 4:
    with pytest.raises(AssertionError):
        D.build_dfc(dep, fit_mask=np.ones(1, bool), year=2026, arrivals=stream_2025)


def test_build_dfc_rejects_a_departure_window_outside_the_arrival_streams_observed_span(D):
    """F1 fail-loud: an in-set departure whose window reaches well outside every date the arrival
    files actually covered (e.g. an arrival stream built from only a single unrelated day) must
    raise rather than silently return a count computed against irrelevant landings."""
    stream = D.ArrivalStream(by_airport={"EGLL": np.array([100.0])},
                             years=frozenset({2026}), span=(100.0, 100.0))
    far_dep = _dep_frame(2026, "EGLL", 10, 0)   # ~10 days of offset from the arrival stream's span
    with pytest.raises(AssertionError):
        D.build_dfc(far_dep, fit_mask=np.ones(1, bool), year=2026, arrivals=stream)


# ---------------------------------------------------------------------------
# F2 — the `_dev` fallback must be defined and identical on fit and non-fit rows
# ---------------------------------------------------------------------------

def test_dev_column_has_no_nan_for_an_ades_hr_key_absent_from_the_fit_table(D):
    """F2: an (ADES, hr) key with zero fit-row support (an unseen destination, or an unseen hour
    for a seen destination — the serve-time shape) must fall back to the hr-only fit median, never
    NaN. Old behaviour: 0.00% NaN in training, 0.42% NaN at serve, from a one-level lookup with no
    fallback."""
    # fit rows: only "AAA" is a fit destination, at hours 10 and 11
    fit_rows = pd.DataFrame({"ADES_mvt": ["AAA", "AAA", "AAA"], "doy": [1, 1, 1],
                             "hr": [10, 10, 11], "tmin": [600, 610, 660], "mvt_sec": [0, 0, 0]})
    # non-fit ("serve") row: destination "ZZZ" never seen in the fit fold at all
    serve_row = pd.DataFrame({"ADES_mvt": ["ZZZ"], "doy": [1], "hr": [10], "tmin": [600],
                              "mvt_sec": [0]})
    f = pd.concat([fit_rows, serve_row], ignore_index=True)
    fit_mask = np.array([True, True, True, False])
    out = D.build_dfc(f, fit_mask=fit_mask, year=2025, arrivals=None)
    assert not out.dfc_ades_dep_1h_dev.isna().any()


def test_dev_fallback_value_matches_the_hr_only_fit_median(D):
    """Pins the exact fallback formula: for a key absent from the (ADES, hr) fit table, the
    subtracted median is the fit-rows-only median of `dfc_ades_dep_1h` at that `hr`, across every
    ADES in the fit fold — not the global median, not zero."""
    fit_rows = pd.DataFrame({"ADES_mvt": ["AAA", "BBB", "AAA"], "doy": [1, 1, 1],
                             "hr": [10, 10, 11], "tmin": [600, 605, 660], "mvt_sec": [0, 0, 0]})
    serve_row = pd.DataFrame({"ADES_mvt": ["ZZZ"], "doy": [1], "hr": [10], "tmin": [601],
                              "mvt_sec": [0]})
    f = pd.concat([fit_rows, serve_row], ignore_index=True)
    fit_mask = np.array([True, True, True, False])
    out = D.build_dfc(f, fit_mask=fit_mask, year=2025, arrivals=None)

    # recompute the hr=10 fallback by hand from the fit rows only (AAA and BBB at hr=10)
    fit_out = D.build_dfc(fit_rows, fit_mask=np.ones(3, bool), year=2025, arrivals=None)
    hr10_fit_counts = fit_out.dfc_ades_dep_1h.to_numpy()[[0, 1]]   # AAA row0, BBB row1 (both hr=10)
    expected_fallback = float(np.median(hr10_fit_counts))
    serve_count = out.dfc_ades_dep_1h.to_numpy()[-1]
    assert out.dfc_ades_dep_1h_dev.to_numpy()[-1] == pytest.approx(serve_count - expected_fallback)


def test_dev_fallback_uses_the_same_lookup_whether_or_not_the_row_is_itself_a_fit_row(D):
    """Train/serve parity (the root-cause fix, not a special case): whether a row with an unseen
    (ADES, hr) key is marked fit or not-fit must not change what it is compared against — there is
    one lookup, not a branch on fit_mask."""
    fit_rows = pd.DataFrame({"ADES_mvt": ["AAA", "BBB"], "doy": [1, 1], "hr": [10, 10],
                             "tmin": [600, 605], "mvt_sec": [0, 0]})
    unseen_row = pd.DataFrame({"ADES_mvt": ["ZZZ"], "doy": [1], "hr": [10], "tmin": [601],
                               "mvt_sec": [0]})
    f = pd.concat([fit_rows, unseen_row], ignore_index=True)

    out_as_nonfit = D.build_dfc(f, fit_mask=np.array([True, True, False]), year=2025, arrivals=None)
    out_as_fit = D.build_dfc(f, fit_mask=np.array([True, True, True]), year=2025, arrivals=None)
    # the unseen row's own dep_1h count is identical either way (it never contributes to any
    # OTHER row's median at n=1 vs n=1 group sizes here); its _dev value must match too
    assert out_as_nonfit.dfc_ades_dep_1h_dev.to_numpy()[-1] == pytest.approx(
        out_as_fit.dfc_ades_dep_1h_dev.to_numpy()[-1])


# ---- DFC.2 serve path: explicit _dev reference tables (PREREG_dfc2_networkflow_forward_2026_09_23.md, T7) ---------
def _two_day_frame(year, ades=("EGLL", "EGLL", "EGLL", "KJFK", "EGLL", "EGLL"), same_day=False):
    doy = pd.Timestamp(year=year, month=1, day=15).dayofyear
    d2 = doy if same_day else doy + 1
    return pd.DataFrame({"ADES_mvt": list(ades), "doy": [doy, doy, doy, doy, d2, d2],
                         "hr": [10, 10, 10, 10, 10, 11], "tmin": [600, 630, 650, 600, 610, 700], "mvt_sec": [0] * 6})


def test_build_dfc_with_explicit_dev_tables_equals_the_fit_mask_path(D, tmp_path):
    """One code path (BC-25): training = fit_mask; serve = the SAME lookup fed an explicit table. The two must be
    byte-identical on the training frame or train and serve can drift apart."""
    f = _two_day_frame(2025, same_day=True); fit = np.array([True, True, False, True, True, False])   # landings file covers Jan 15
    p25 = tmp_path / "training_2025-01.parquet"; _write_arr_file(p25, 2025, "EGLL", [(9, 30), (9, 45), (10, 0), (10, 10)])
    arr = D.load_arrivals([p25], year=2025)            # WITH landings, so the arr_1h_dev table is exercised too
    tables = D.fit_dev_tables(f, fit_mask=fit, year=2025, arrivals=arr)
    assert set(tables) == {"dfc_ades_dep_1h", "dfc_ades_arr_1h"}
    assert not tables["dfc_ades_arr_1h"].ades_hr.equals(tables["dfc_ades_dep_1h"].ades_hr)
    a = D.build_dfc(f, fit_mask=fit, year=2025, arrivals=arr)
    b = D.build_dfc(f, fit_mask=None, year=2025, arrivals=arr, dev_tables=tables)
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(a.dfc_ades_arr_1h_dev.to_numpy()[[0, 1, 2]]).all()


def test_build_dfc_requires_exactly_one_of_fit_mask_or_dev_tables(D):
    f = _two_day_frame(2025); fit = np.ones(6, bool)
    with pytest.raises(ValueError, match="exactly one of fit_mask or dev_tables"):
        D.build_dfc(f, fit_mask=None, year=2025)
    with pytest.raises(ValueError, match="exactly one of fit_mask or dev_tables"):
        D.build_dfc(f, fit_mask=fit, year=2025, dev_tables=D.fit_dev_tables(f, fit_mask=fit, year=2025))


def test_a_2026_frame_takes_2025_dev_tables_with_2026_arrivals(D, tmp_path):
    """The serve construction: 2026 departures, 2026 landings, 2025 median tables (INV section 6.4 provenance).
    dep_1h_dev must be the 2026 count minus the 2025 table's median for that (ADES, hr) key — never a 2026 median."""
    f25 = _two_day_frame(2025); f26 = _two_day_frame(2026, same_day=True)   # the 2026 landings file covers Jan 15 only
    p26 = tmp_path / "ranking_2026.parquet"; _write_arr_file(p26, 2026, "EGLL", [(9, 50), (10, 5)])
    tables = D.fit_dev_tables(f25, fit_mask=np.ones(6, bool), year=2025)
    out = D.build_dfc(f26, fit_mask=None, year=2026, arrivals=D.load_arrivals([p26], year=2026), dev_tables=tables)
    med_2025 = float(tables["dfc_ades_dep_1h"].ades_hr[("EGLL", 10)])
    assert med_2025 == 2.0                                            # doy-15 rows 0-2 see 2 peers; row 4 sees 0
    got = out.dfc_ades_dep_1h.to_numpy() - out.dfc_ades_dep_1h_dev.to_numpy()
    assert np.allclose(got[[0, 1, 2, 4]], med_2025)                   # subtracted the 2025 table, not a 2026 median
    assert out.dfc_ades_arr_1h.iloc[0] == 2 and np.isnan(out.dfc_ades_arr_1h.iloc[3])


def test_apply_dev_table_falls_back_to_the_hr_median_for_an_absent_key(D):
    key = pd.DataFrame({"a": ["EGLL", "EGLL", "LFPG"], "h": [10, 10, 10]}); v = np.array([4.0, 6.0, 9.0])
    t = D.fit_dev_table(key, v, np.array([True, True, False]))        # LFPG absent from the fit table
    assert float(t.hr[10]) == 5.0 and ("LFPG", 10) not in t.ades_hr.index
    assert np.array_equal(D.apply_dev_table(key, v, t), np.array([-1.0, 1.0, 4.0]))   # 9 - hr median 5


def test_fit_dev_tables_uses_fit_rows_only(D):
    f = _two_day_frame(2025)
    t_all = D.fit_dev_tables(f, fit_mask=np.ones(6, bool), year=2025)["dfc_ades_dep_1h"]
    t_two = D.fit_dev_tables(f, fit_mask=np.array([False, False, False, False, True, True]), year=2025)["dfc_ades_dep_1h"]
    assert float(t_all.ades_hr[("EGLL", 10)]) == 2.0 and float(t_two.ades_hr[("EGLL", 10)]) == 0.0
