"""CSF.2 — registered invariants of plans/PREREG_csf2_rome_identifier_2026_09_22.md.

The arm adds identity inputs (callsign frequency, imputed aircraft type / market segment from the same
callsign's other legs, the callsign's matched-elsewhere rate) to the SHIPPED LIRF fill classifier
(`rome_fill.fit_predict_fill`, LightGBM arm R, mean of seeds 0-2). Its characteristic failures, each with
its own test below:
  * an imputation that reads across calendar months or groups null callsigns together (BC-15);
  * a `cs_match_rate` that counts the row itself;
  * a baseline that is not the shipped artefact (`replicate_baseline` must ABORT on a mismatch);
  * a value clause the perfect oracle cannot pass (`value_clauses` is run through the oracle here);
  * post-compute lines (JSON, hash append, summary) that crash after the fits (BC-24).

Every test was seen RED against a targeted mutant of the mirror copy under logs/csf2_scratch_tools/
(rehearsal recorded in the registration's RESULT).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
PY = sys.executable


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def M():
    return _load("csf2_build", "scripts/analysis/csf2_build.py")


T = pd.Timestamp


def _mv(mid, cs, when, ap="LIRF", phase="DEP", typ=None, typ_flt=None, seg=None, aobt=None, ades="EGLL",
        sched=None, block=None):
    return {"MVT_ID_mvt": float(mid), "FLIGHT_mvt": cs, "MVT_TIME_UTC_mvt": T(when, tz="UTC"),
            "ADEP_mvt": ap, "ADES_mvt": ades, "PHASE_mvt": phase,
            "AIRCRAFT_TYPE_mvt": typ, "AIRCRAFT_TYPE_flt": typ_flt, "MARKET_SEGMENT_flt": seg,
            "AOBT_3_flt": (T(aobt, tz="UTC") if aobt else pd.NaT),
            "SCHED_TIME_UTC_mvt": (T(sched, tz="UTC") if sched else pd.NaT),
            "BLOCK_TIME_UTC_mvt": (T(block, tz="UTC") if block else pd.NaT)}


def _pool(recs):
    d = pd.DataFrame(recs)
    for c in ("AOBT_3_flt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"):
        d[c] = pd.to_datetime(d[c], utc=True)
    return d


# ------------------------------------------------------------------ registration constants

def test_constants_match_the_registration(M):
    assert M.EXPECT_ROWS == 1488 and M.EXPECT_JANJUL == 397 and M.N_MONTHS == 12
    assert M.SEEDS == (0, 1, 2)
    assert M.AUC_TRUE == 0.95 and M.AUC_FALSE == 0.92
    assert M.SLIP_Y_S == 86_400.0 and M.SLIP_TAXI_S == 1020.0
    assert M.SCORED_LIRF_UNM == 383 and M.BOARD_N == 344_841
    assert M.N_BOOT == 2000 and M.BOOT_SEED == 0
    assert M.FORWARD_MONTHS == (7, 8, 9, 10, 11, 12) and M.DECISION_MONTHS == (1, 7)
    assert M.K_RANGE == (1.38, 5.87)
    assert M.RSS_ABORT_GIB == 6.0 and M.REPLICATION_ATOL == 1e-6
    assert M.ARMS["R"] == ((), ())
    assert M.ARMS["CS"] == (("log_n_cs", "one_off", "has_twin36", "twin_h"), ())
    assert M.ARMS["FULL"] == (("log_n_cs", "one_off", "has_twin36", "twin_h", "cs_match_rate", "actype_src",
                               "segment_src"), ("actype_any", "segment_any"))
    assert M.DECISIONAL_ARM == "FULL"


# ------------------------------------------------------------------ labels

def _lab_frame(y, sp):
    t = pd.to_datetime(np.zeros(len(y)), unit="s", utc=True)
    sched = t - pd.to_timedelta(sp, unit="s")
    block = t - pd.to_timedelta(y, unit="s")
    return pd.DataFrame({"y": np.asarray(y, float), "sp": np.asarray(sp, float),
                         "MVT_TIME_UTC_mvt": t, "SCHED_TIME_UTC_mvt": sched, "BLOCK_TIME_UTC_mvt": block})


def test_labels_fill_is_the_shipped_60s_definition_and_slip_is_nonfill_ge_86400(M):
    f = _lab_frame(y=[1000.0, 5000.0, 87_170.0, 131_167.0, 5060.0, 5061.0], sp=[1000.0, 3000.0, 93_535.0, 131_163.0,
                                                                                  5000.0, 5000.0])
    L = M.labels(f)
    assert L.fill.tolist() == [True, False, False, True, True, False]
    assert L.slip.tolist() == [False, False, True, False, False, False]
    assert L.cls.tolist() == ["fill", "ordinary", "slip", "fill", "fill", "ordinary"]


def test_labels_refuse_a_non_finite_label(M):
    f = _lab_frame(y=[1000.0, np.nan], sp=[1000.0, 1000.0])
    with pytest.raises(ValueError, match="finite"):
        M.labels(f)


def test_label_integrity_counts(M):
    f = _lab_frame(y=[1000.0, 20_000.0, 87_170.0, 131_167.0], sp=[1000.0, 3000.0, 93_535.0, 131_163.0])
    s = M.label_integrity(f)
    assert s == {"n_rows": 4, "max_y_s": 131_167.0, "n_y_ge_86400": 2, "n_y_gt_10800": 3, "n_fill": 2, "n_slip": 1,
                 "min_y_s": 1000.0}


# ------------------------------------------------------------------ identity imputation

def test_own_aircraft_type_is_kept_when_present(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00", typ="A320"), _mv(2, "AZA1", "2025-03-05 10:00", typ="B738")])
    f = M.identity_features(pool.iloc[[0]], pool)
    assert f.actype_any.tolist() == ["A320"] and f.actype_src.tolist() == [0]


def test_null_type_is_imputed_from_the_same_callsign_in_the_same_month(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-05 10:00", typ="B738"),
                  _mv(3, "AZA1", "2025-03-06 10:00", typ="B738"), _mv(4, "AZA1", "2025-03-07 10:00", typ="A320")])
    f = M.identity_features(pool.iloc[[0]], pool)
    assert f.actype_any.tolist() == ["B738"] and f.actype_src.tolist() == [1]


def test_imputation_uses_the_NM_type_when_the_airport_type_is_null_on_the_other_leg(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-05 10:00", typ_flt="E190", phase="ARR")])
    f = M.identity_features(pool.iloc[[0]], pool)
    assert f.actype_any.tolist() == ["E190"] and f.actype_src.tolist() == [1]


def test_imputation_does_NOT_read_across_calendar_months(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-31 23:00"), _mv(2, "AZA1", "2025-04-01 01:00", typ="B738")])
    f = M.identity_features(pool.iloc[[0]], pool)
    assert f.actype_any.tolist() == ["NA"] and f.actype_src.tolist() == [2]


def test_null_callsign_is_never_imputed_from_other_null_callsigns(M):
    """BC-15 shape: two no-callsign flights must not lend each other an identity."""
    pool = _pool([_mv(1, None, "2025-03-04 10:00"), _mv(2, None, "2025-03-05 10:00", typ="B738", seg="Mainline")])
    f = M.identity_features(pool.iloc[[0]], pool)
    assert f.actype_any.tolist() == ["NA"] and f.actype_src.tolist() == [2]
    assert f.segment_any.tolist() == ["NA"] and f.segment_src.tolist() == [2]


def test_imputation_tie_breaks_alphabetically_so_the_run_is_deterministic(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-05 10:00", typ="B738"),
                  _mv(3, "AZA1", "2025-03-06 10:00", typ="A320")])
    f = M.identity_features(pool.iloc[[0]], pool)
    assert f.actype_any.tolist() == ["A320"]


def test_segment_is_imputed_like_the_type(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-05 10:00", seg="Lowcost"),
                  _mv(3, "AZA1", "2025-03-06 10:00", seg="Lowcost"), _mv(4, "BAW2", "2025-03-06 10:00", seg="Mainline")])
    f = M.identity_features(pool.iloc[[0, 3]], pool)
    assert f.segment_any.tolist() == ["Lowcost", "Mainline"] and f.segment_src.tolist() == [1, 0]


def test_cs_match_rate_excludes_the_row_itself_and_counts_only_DEP_at_the_ten(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00"),                                    # the row: unmatched
                  _mv(2, "AZA1", "2025-03-05 10:00", aobt="2025-03-05 09:40"),           # matched DEP
                  _mv(3, "AZA1", "2025-03-06 10:00"),                                    # unmatched DEP
                  _mv(4, "AZA1", "2025-03-07 10:00", aobt="2025-03-07 09:40", phase="ARR"),   # ARR: not counted
                  _mv(5, "AZA1", "2025-03-08 10:00", aobt="2025-03-08 09:40", ap="ZZZZ")])    # outside the ten
    f = M.identity_features(pool.iloc[[0]], pool)
    assert f.cs_match_rate.tolist() == [0.5]


def test_cs_match_rate_does_not_count_a_MATCHED_row_as_its_own_peer(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00", aobt="2025-03-04 09:40"),
                  _mv(2, "AZA1", "2025-03-05 10:00")])
    f = M.identity_features(pool.iloc[[0]], pool)
    assert f.cs_match_rate.tolist() == [0.0]


def test_cs_match_rate_sentinel_when_the_callsign_has_no_other_departure(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, None, "2025-03-05 10:00")])
    f = M.identity_features(pool, pool)
    assert f.cs_match_rate.tolist() == [-1.0, -1.0]


def test_build_features_columns_one_off_and_log_n_cs(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "BAW2", "2025-03-05 10:00"),
                  _mv(3, "BAW2", "2025-03-06 10:00"), _mv(4, None, "2025-03-06 11:00")])
    f = M.build_features(pool, pool)
    assert list(f.columns) == list(M.FEATURE_COLS)
    assert f.n_cs.tolist() == [1, 2, 2, 0]
    assert f.one_off.tolist() == [1, 0, 0, 0]
    assert f.has_twin36.tolist() == [0, 1, 1, 0]          # BAW2's two legs are 24 h apart: twins
    np.testing.assert_allclose(f.log_n_cs.to_numpy(), np.log1p([1, 2, 2, 0]), rtol=0, atol=1e-12)
    assert list(f.index) == list(pool.index)


def test_attach_appends_the_feature_columns_without_reordering_rows(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "BAW2", "2025-03-05 10:00")])
    rows = pool.iloc[[1, 0]]
    out = M.attach(rows, pool)
    assert out.MVT_ID_mvt.tolist() == [2.0, 1.0]
    assert set(M.FEATURE_COLS) <= set(out.columns)


def test_features_never_read_a_label(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00"), _mv(2, "AZA1", "2025-03-05 10:00", typ="B738")])
    a = M.build_features(pool, pool)
    pool2 = pool.assign(y=[99999.0, 1.0], TAXITIME_SEC_mvt=[5.0, 6.0])
    pool2["BLOCK_TIME_UTC_mvt"] = pd.NaT
    b = M.build_features(pool2, pool2)
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------ folds

def test_lomo_folds_are_leave_one_month_out(M):
    f = M.lomo_folds([1, 2, 3])
    assert f == [(1, [2, 3]), (2, [1, 3]), (3, [1, 2])]


def test_forward_folds_train_only_on_strictly_earlier_months(M):
    f = M.forward_folds(list(range(1, 13)))
    assert [m for m, _ in f] == [7, 8, 9, 10, 11, 12]
    assert f[0] == (7, [1, 2, 3, 4, 5, 6]) and f[-1] == (12, list(range(1, 12)))


def test_forward_folds_refuse_a_missing_training_month(M):
    with pytest.raises(ValueError, match="month"):
        M.forward_folds([1, 2, 3, 7, 8, 9, 10, 11, 12])


# ------------------------------------------------------------------ value arithmetic

def test_scale_board_is_delta_sse_times_383_over_n_over_344841(M):
    assert M.scale_board(344_841.0 * 397.0, 397) == pytest.approx(383.0, rel=0, abs=1e-9)


def test_remove_top_drops_the_largest_abs_rows_and_reports_their_share(M):
    d = np.array([1.0, -100.0, 5.0, 40.0])
    s1, sh1 = M.remove_top(d, 1)
    assert s1 == pytest.approx(46.0) and sh1 == pytest.approx(100.0 / 146.0)
    s3, sh3 = M.remove_top(d, 3)
    assert s3 == pytest.approx(1.0) and sh3 == pytest.approx(145.0 / 146.0)


def test_remove_top_with_k_ge_n_returns_zero_and_share_one(M):
    s, sh = M.remove_top(np.array([2.0, -3.0]), 5)
    assert s == 0.0 and sh == 1.0


def test_rome_day_is_the_local_date_not_utc(M):
    t = pd.Series(pd.to_datetime(["2025-07-01 22:30", "2025-01-31 23:30"], utc=True))
    assert M.rome_day(t).tolist() == ["2025-07-02", "2025-02-01"]


def test_day_block_ci_resamples_DAYS_not_rows(M):
    """Two days: one day carries a huge paired gain, the other nothing. A day bootstrap must reach a
    draw with zero gain (both draws pick the quiet day), so the lower bound is 0, not a positive
    row-bootstrap value."""
    se_ref = np.array([1e6] * 10 + [1.0] * 10)
    se_new = np.array([0.0] * 10 + [1.0] * 10)
    days = np.array(["d1"] * 10 + ["d2"] * 10)
    r = M.day_block_ci(se_ref, se_new, days, n_boot=400, seed=0)
    assert r["gain"] == pytest.approx(1e7) and r["n_days"] == 2
    assert r["ci95"][0] == pytest.approx(0.0) and r["ci95"][1] == pytest.approx(2e7)
    # a ROW bootstrap of the same 20 rows can almost never draw twenty quiet rows: its lower bound is far above 0
    rng = np.random.default_rng(0)
    d = se_ref - se_new
    rows = np.array([d[rng.integers(0, 20, 20)].sum() for _ in range(400)])
    assert np.percentile(rows, 2.5) > 1e6


def test_day_block_ci_is_deterministic_in_the_seed(M):
    se_ref = np.arange(20, dtype=float) ** 2
    se_new = np.arange(20, dtype=float)
    days = np.array([f"d{i % 5}" for i in range(20)])
    a = M.day_block_ci(se_ref, se_new, days, n_boot=100, seed=0)
    b = M.day_block_ci(se_ref, se_new, days, n_boot=100, seed=0)
    assert a == b


def _oracle_case():
    y = np.array([5000.0, 5000.0, 1000.0, 1200.0, 87_170.0, 900.0])
    sp = np.array([5000.0, 5000.0, 4000.0, 6000.0, 60_000.0, 3000.0])
    nf = np.array([1000.0, 1000.0, 1100.0, 1100.0, 1100.0, 1000.0])
    cls = np.array(["fill", "fill", "ordinary", "ordinary", "slip", "ordinary"])
    return y, sp, nf, cls


def test_oracle_preds_two_and_three_class(M):
    y, sp, nf, cls = _oracle_case()
    two, three = M.oracle_preds(cls, sp, nf)
    np.testing.assert_allclose(two, [5000, 5000, 1100, 1100, 1100, 1000], rtol=0, atol=1e-9)
    np.testing.assert_allclose(three, [5000, 5000, 1100, 1100, 86_400 + 1020, 1000], rtol=0, atol=1e-9)


def test_se_rows_uses_the_shipped_mixture_with_its_floor(M):
    se = M.se_rows(np.array([0.0, 1.0]), np.array([-500.0, 4000.0]), np.array([0.5, 1000.0]), np.array([1.0, 4000.0]))
    np.testing.assert_allclose(se, [0.0, 0.0], rtol=0, atol=1e-12)     # floor max(.,1) makes pred 1 on row 0


def test_value_clauses_pass_for_the_ORACLE_and_fail_for_a_null_delta(M):
    """The clause the oracle fails is deleted: here the perfect two-class identifier must pass every
    value clause and an identical prediction (delta 0 everywhere) must fail C2 and C4."""
    rng = np.random.default_rng(0)
    n = 120
    months = np.where(np.arange(n) < 40, 1, 7)
    days = np.array([f"{m:02d}-{i % 15:02d}" for m, i in zip(months, range(n))])
    sp = rng.uniform(3600, 20_000, n)
    fill = rng.random(n) < 0.5
    nf = rng.uniform(900, 1500, n)
    y = np.where(fill, sp, nf + rng.normal(0, 200, n))
    p_inc = np.clip(0.5 + rng.normal(0, 0.2, n), 0.01, 0.99)
    se_inc = M.se_rows(p_inc, sp, nf, y)
    two, _ = M.oracle_preds(np.where(fill, "fill", "ordinary"), sp, nf)
    se_or = (two - y) ** 2
    v = M.value_clauses(se_inc, se_or, days, months, n_boot=300, seed=0)
    assert v["c2_ci_excludes_zero"] and v["c3_both_months_positive"] and v["c4_minus_top3_positive"]
    assert v["gain_sse"] > 0 and v["by_month"]["1"] > 0 and v["by_month"]["7"] > 0
    assert v["n_rows"] == n and set(v) >= {"minus_top1_sse", "minus_top3_sse", "top1_share", "top3_share",
                                           "mde_sse", "ci95"}
    z = M.value_clauses(se_inc, se_inc, days, months, n_boot=300, seed=0)
    assert not z["c2_ci_excludes_zero"] and not z["c4_minus_top3_positive"] and z["gain_sse"] == 0.0


def test_value_clauses_c3_requires_BOTH_months_positive(M):
    se_ref = np.array([100.0, 100.0, 1.0, 1.0])
    se_new = np.array([0.0, 0.0, 5.0, 5.0])
    v = M.value_clauses(se_ref, se_new, np.array(["a", "b", "c", "d"]), np.array([1, 1, 7, 7]), n_boot=50, seed=0)
    assert v["by_month"] == {"1": 200.0, "7": -8.0} and not v["c3_both_months_positive"]


def test_value_clauses_mde_is_2_8_half_width_over_1_96(M):
    se_ref = np.arange(1, 41, dtype=float) ** 2
    se_new = np.arange(1, 41, dtype=float)
    days = np.array([f"d{i % 8}" for i in range(40)])
    v = M.value_clauses(se_ref, se_new, days, np.where(np.arange(40) < 20, 1, 7), n_boot=100, seed=0)
    assert v["mde_sse"] == pytest.approx(2.8 * (v["ci95"][1] - v["ci95"][0]) / 2 / 1.96)


def test_band_split_and_move_split(M):
    d = np.array([10.0, 20.0, 30.0, 40.0])
    sp = np.array([100.0, 3600.0, 23_999.0, 24_000.0])
    assert M.band_split(d, sp) == {"lt3600": 10.0, "3600_24000": 50.0, "ge24000": 40.0}
    ref = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    new = np.array([1100.0, 1300.0, 5000.0, 1000.0])
    assert M.move_split(ref, new, d) == {"le300s": 30.0, "300_3600s": 0.0, "gt3600s": 30.0, "unchanged": 40.0}


# ------------------------------------------------------------------ the decision

def test_decide_all_branches(M):
    ok = {"c2_ci_excludes_zero": True, "c3_both_months_positive": True, "c4_minus_top3_positive": True,
          "c5_forward_positive": True}
    r = M.decide(0.95, ok)
    assert r == {"auc_band": "TRUE", "value": "WORKING", "verdict": "WORKING", "build_authorised": True}
    assert M.decide(0.9499, ok)["verdict"] == "NOT WORKING" and M.decide(0.9499, ok)["auc_band"] == "BETWEEN"
    assert M.decide(0.92, ok)["auc_band"] == "BETWEEN" and M.decide(0.9199, ok)["auc_band"] == "FALSE"
    for k in ok:
        bad = dict(ok, **{k: False})
        r = M.decide(0.99, bad)
        assert r["value"] == "NOT WORKING" and r["verdict"] == "NOT WORKING" and not r["build_authorised"], k


def test_decide_refuses_a_non_finite_auc(M):
    with pytest.raises(ValueError, match="AUC"):
        M.decide(float("nan"), {"c2_ci_excludes_zero": True, "c3_both_months_positive": True,
                                "c4_minus_top3_positive": True, "c5_forward_positive": True})


# ------------------------------------------------------------------ the ARR-row witness

def test_witness_table_buckets_by_callsign_frequency_and_reports_the_ratio(M):
    recs = []
    # one-off callsigns (no DEP with that callsign in the month): 3 arrivals, 2 copies
    for i, cp in enumerate([True, True, False]):
        recs.append(_mv(100 + i, f"ONE{i}", f"2025-03-0{i + 1} 10:00", ap="EGLL", ades="LIRF", phase="ARR",
                        sched="2025-03-01 09:00", block=("2025-03-01 09:00:30" if cp else "2025-03-01 09:30")))
    # a frequent callsign: 10 departures in the month + 4 arrivals, 1 copy
    for i in range(10):
        recs.append(_mv(200 + i, "FRQ1", f"2025-03-{i + 1:02d} 08:00"))
    for i, cp in enumerate([True, False, False, False]):
        recs.append(_mv(300 + i, "FRQ1", f"2025-03-{i + 1:02d} 12:00", ap="EGLL", ades="LIRF", phase="ARR",
                        sched="2025-03-01 09:00", block=("2025-03-01 09:00:30" if cp else "2025-03-01 09:30")))
    pool = _pool(recs)
    arr = pool[(pool.PHASE_mvt == "ARR") & (pool.ADES_mvt == "LIRF")]
    w = M.witness_table(arr, pool)
    assert w["buckets"]["0-1"] == {"n": 3, "copy_rate": pytest.approx(2 / 3)}
    assert w["buckets"]["10+"] == {"n": 4, "copy_rate": pytest.approx(0.25)}
    assert w["ratio_oneoff_vs_10plus"] == pytest.approx((2 / 3) / 0.25)
    assert w["n_rows"] == 7


def test_witness_table_ratio_is_none_when_a_bucket_is_empty(M):
    pool = _pool([_mv(1, "ONE0", "2025-03-01 10:00", ap="EGLL", ades="LIRF", phase="ARR",
                      sched="2025-03-01 09:00", block="2025-03-01 09:00:10")])
    w = M.witness_table(pool, pool)
    assert w["ratio_oneoff_vs_10plus"] is None and w["buckets"]["10+"] == {"n": 0, "copy_rate": None}


def test_witness_clause_needs_both_years_above_one(M):
    assert M.witness_clause({"ratio_oneoff_vs_10plus": 2.0}, {"ratio_oneoff_vs_10plus": 1.5})
    assert not M.witness_clause({"ratio_oneoff_vs_10plus": 2.0}, {"ratio_oneoff_vs_10plus": 0.9})
    assert not M.witness_clause({"ratio_oneoff_vs_10plus": None}, {"ratio_oneoff_vs_10plus": 1.5})


# ------------------------------------------------------------------ post-compute lines (BC-24)

def test_write_report_refuses_an_existing_file_and_writes_json(M, tmp_path):
    p = tmp_path / "r.json"
    M.write_report({"a": np.float64(1.5), "b": np.array([1, 2]), "c": np.int64(3)}, p)
    assert json.loads(p.read_text()) == {"a": 1.5, "b": [1, 2], "c": 3}
    with pytest.raises(FileExistsError, match="second look"):
        M.write_report({"a": 1}, p)


def test_append_hash_line_format(M, tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}")
    log = tmp_path / "log"
    line = M.append_hash(log, "RESULT", f, note="hello")
    sha = hashlib.sha256(b"{}").hexdigest()
    assert line.endswith(f"RESULT  {f}  sha256={sha[:8]}  (full {sha})  -- hello")
    assert line[:4] == "2026" or line[4] == "-"
    assert line[19] == "Z" or "T" in line[:20]
    assert log.read_text() == line + "\n"
    M.append_hash(log, "X", f)
    assert log.read_text().count("\n") == 2


def test_summary_lines_cover_the_verdict_and_every_arm(M):
    R = {"verdict": {"verdict": "NOT WORKING", "auc_band": "FALSE", "value": "NOT WORKING", "build_authorised": False},
         "lomo": {"R": {"auc_pooled": 0.9}, "FULL": {"auc_pooled": 0.91}},
         "clauses": {"c1_auc_ge_0_95": False}, "wall_s": 1.0}
    lines = M.summary_lines(R)
    joined = "\n".join(lines)
    assert "NOT WORKING" in joined and "FULL" in joined and "0.91" in joined and "c1_auc_ge_0_95" in joined


def test_rss_guard_aborts_the_process_with_exit_3(M):
    code = ("import sys, time; sys.path.insert(0, %r); sys.path.insert(0, %r); "
            "import importlib.util as u; s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); "
            "s.loader.exec_module(m); m.rss_guard(limit_gib=1e-6, interval_s=0.01); time.sleep(5); print('alive')"
            % (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts/analysis/csf2_build.py")))
    p = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 3 and "alive" not in p.stdout and "RSS" in (p.stdout + p.stderr)


def test_heavy_lock_refuses_when_held(M, tmp_path):
    lock = tmp_path / "heavy.lock"
    fh = M.acquire_heavy_lock(lock)
    try:
        code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import importlib.util as u; "
                "s = u.spec_from_file_location('m', %r); m = u.module_from_spec(s); s.loader.exec_module(m); "
                "m.acquire_heavy_lock(%r)" % (str(ROOT), str(ROOT / "scripts"),
                                              str(ROOT / "scripts/analysis/csf2_build.py"), str(lock)))
        p = subprocess.run([PY, "-c", code], capture_output=True, text=True, timeout=60)
        assert p.returncode == 75 and "HELD" in (p.stdout + p.stderr)
    finally:
        M.release_heavy_lock(fh)
    fh2 = M.acquire_heavy_lock(lock)
    M.release_heavy_lock(fh2)


# ------------------------------------------------------------------ the fits and the baseline guard

@pytest.fixture(scope="module")
def synth(M):
    return M.synthetic_inputs(seed=1, n_per_month=160, n_matched_per_month=120, seeds=(0,))


def test_synthetic_inputs_have_the_real_shapes(M, synth):
    unm, lirf, pool, stored, arr25, arr26 = synth
    assert (unm.ADEP_mvt == "LIRF").sum() > 100 and len(lirf) == 12 * 120
    assert set(M.POOL_COLS) <= set(pool.columns) and (pool.PHASE_mvt == "ARR").any()
    assert {"MVT_ID_mvt", "fold", "month", "y", "sp", "nf", "p_R", "p_R_seed0", "R"} <= set(stored.columns)
    assert (stored.fold == "lomo").sum() == (unm.ADEP_mvt == "LIRF").sum()
    assert len(arr25) > 50 and len(arr26) > 20 and (arr25.ADES_mvt == "LIRF").all() and (arr26.ADES_mvt == "LIRF").all()


@pytest.mark.slow
def test_fit_arm_means_the_seeds_and_a_planted_identity_signal_is_recovered(M, synth):
    unm, lirf, pool, stored, _, _ = synth
    import rome_fill as rf
    u = M.attach(unm[unm.ADEP_mvt == "LIRF"], pool)
    l_ = M.attach(lirf, pool)
    tr = rf.rome_rows(u[u.month != 3], l_[l_.month != 3], [m for m in range(1, 13) if m != 3])
    te = u[u.month == 3].copy()
    p, by = M.fit_arm(tr, te, seeds=(0, 1), extra_num=(), extra_cat=())
    np.testing.assert_allclose(p, 0.5 * (by[0] + by[1]), rtol=0, atol=1e-12)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all() and len(p) == len(te)
    # plant: a categorical that IS the label on train and test; the arm must recover it
    lab_tr = M.labels(tr).fill.to_numpy()
    lab_te = M.labels(te).fill.to_numpy()
    tr2 = tr.assign(actype_any=np.where(lab_tr, "FILL", "ORD"))
    te2 = te.assign(actype_any=np.where(lab_te, "FILL", "ORD"))
    p2, _ = M.fit_arm(tr2, te2, seeds=(0,), extra_num=(), extra_cat=("actype_any",))
    assert M.auc(lab_te, p2) > 0.99 > M.auc(lab_te, p) + 0.05


def test_fit_arm_refuses_a_missing_extra_column(M, synth):
    unm, lirf, pool, _, _, _ = synth
    import rome_fill as rf
    u = unm[unm.ADEP_mvt == "LIRF"]
    tr = rf.rome_rows(u[u.month != 3], lirf[lirf.month != 3], [m for m in range(1, 13) if m != 3])
    with pytest.raises(KeyError, match="log_n_cs"):
        M.fit_arm(tr, u[u.month == 3], seeds=(0,), extra_num=("log_n_cs",), extra_cat=())


def test_stored_lookup_aligns_by_id_and_refuses_a_missing_row(M, synth):
    _, _, _, stored, _, _ = synth
    ids = stored[stored.fold == "lomo"].MVT_ID_mvt.to_numpy()[[5, 2, 9]]
    got = M.stored_lookup(stored, ids, "lomo")
    assert got.MVT_ID_mvt.tolist() == ids.tolist()
    with pytest.raises(ValueError, match="absent"):
        M.stored_lookup(stored, np.array([ids[0], -1.0]), "lomo")


def test_replicate_baseline_passes_on_agreement_and_ABORTS_on_a_mismatch(M):
    a = {0: np.array([0.1, 0.2]), 1: np.array([0.3, 0.4])}
    r = M.replicate_baseline(a, {0: a[0].copy(), 1: a[1] + 1e-9}, atol=1e-6)
    assert r["passed"] and r["max_abs_delta_by_seed"] == {"0": 0.0, "1": pytest.approx(1e-9)}
    with pytest.raises(ValueError, match="does not replicate"):
        M.replicate_baseline(a, {0: a[0] + 1e-3, 1: a[1]}, atol=1e-6)


@pytest.mark.slow
def test_run_end_to_end_on_synthetic_inputs(M, synth, tmp_path):
    unm, lirf, pool, stored, arr25, arr26 = synth
    R = M.run(unm, lirf, pool, stored, arr25, arr26, seeds=(0,), n_boot=30, log=lambda m: None, smoke=True)
    assert R["smoke"] is True and R["replication"]["passed"] is True
    assert set(R["lomo"]) == {"R", "CS", "FULL"} and set(R["forward"]) == {"R", "CS", "FULL"}
    for arm in ("R", "CS", "FULL"):
        blk = R["lomo"][arm]
        assert 0.0 <= blk["auc_pooled"] <= 1.0 and len(blk["auc_by_month"]) == 12
        assert set(blk["value_janjul"]) >= {"gain_sse", "ci95", "c2_ci_excludes_zero", "board_scaled"}
    assert R["verdict"]["verdict"] in ("WORKING", "NOT WORKING")
    assert set(R["clauses"]) == {"c1_auc_ge_0_95", "c2_ci_excludes_zero", "c3_both_months_positive",
                                 "c4_minus_top3_positive", "c5_forward_positive"}
    # the oracle blocks carry every clause key; their SIGN on synthetic frames is not a property worth pinning:
    # the synthetic nf is a plain non-fill median, so a hedge toward sp can beat the pure nf on the synthetic tail rows
    # (seen 2026-09-22: one ordinary row y 26,022 / sp 78,861 / nf 1,673 cost the two-class oracle 40M). The oracle's
    # clause passage is demonstrated on a realistic nf in test_value_clauses_pass_for_the_ORACLE_... and measured on the
    # real artefact by the `baseline` subcommand.
    for name in ("two_class", "three_class"):
        blk = R["oracle"][name]["janjul"]
        assert blk["n_rows"] == R["n_janjul"] and np.isfinite(blk["gain_sse"])
        assert {"c2_ci_excludes_zero", "c3_both_months_positive", "c4_minus_top3_positive", "board_scaled"} <= set(blk)
    p = tmp_path / "csf2_report.json"
    M.write_report(R, p)
    json.loads(p.read_text())


def test_main_smoke_subcommand_writes_a_report_and_a_hash_line(M, tmp_path):
    rc = M.main(["smoke", "--out-dir", str(tmp_path), "--seeds", "0", "--n-boot", "20", "--n-per-month", "120",
                 "--n-matched-per-month", "90"])
    assert rc == 0
    R = json.loads((tmp_path / "csf2_smoke_report.json").read_text())
    assert R["smoke"] is True and "verdict" in R
    log = (tmp_path / "csf2_smoke_hashes.log").read_text()
    assert "SMOKE" in log and "sha256=" in log


def test_main_refuses_an_unknown_subcommand(M):
    with pytest.raises(SystemExit):
        M.main(["frobnicate"])


# ------------------------------------------------------------------ the baseline read (step 1)

def test_weighted_auc_equals_the_rank_auc_under_unit_weights_and_moves_under_sp2(M):
    y = np.array([1, 0, 1, 0, 1, 0], bool)
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    assert M.weighted_auc(y, p, np.ones(6)) == pytest.approx(M.auc(y, p))
    w = np.array([1.0, 100.0, 1.0, 1.0, 1.0, 1.0])       # the mis-ranked negative dominates
    assert M.weighted_auc(y, p, w) < M.auc(y, p) - 0.2
    assert M.weighted_auc(np.ones(3, bool), p[:3], np.ones(3)) is None


@pytest.mark.slow
def test_baseline_on_synthetic_inputs_replicates_and_prices_the_oracle(M, synth):
    unm, lirf, pool, stored, _, _ = synth
    B = M.baseline(unm, lirf, stored, seeds=(0,), n_boot=30, log=lambda m: None, smoke=True)
    assert B["replication"]["passed"] is True
    assert 0.0 <= B["auc"]["stored_p_R"] <= 1.0 and B["auc"]["refit_p_R"] == pytest.approx(B["auc"]["stored_p_R"])
    assert set(B["oracle"]["two_class"]) == {"janjul", "all12", "foldA"}
    assert np.isfinite(B["oracle"]["two_class"]["janjul"]["gain_sse"]) and B["oracle"]["two_class"]["foldA"]["n_rows"] > 0
    assert B["oracle"]["two_class"]["janjul"]["n_rows"] == B["n_janjul"]
    assert B["oracle"]["two_class"]["janjul"]["scaling"].startswith("sse x 383/n")
    json.dumps(B, default=str)


# ------------------------------------------------------------------ hardening after the mutation plan

def test_value_clauses_c2_needs_the_CI_not_just_the_sign(M):
    """gain > 0 but one of two days is negative: the day-block interval spans zero and C2 must fail."""
    v = M.value_clauses(np.array([10.0, 0.0]), np.array([0.0, 8.0]), np.array(["a", "b"]), np.array([1, 7]),
                        n_boot=200, seed=0)
    assert v["gain_sse"] == pytest.approx(2.0) and v["ci95"][0] < 0 and not v["c2_ci_excludes_zero"]


def test_value_clauses_c4_removes_three_rows_not_one(M):
    d_ref = np.array([100.0, 0.0, 0.0, 5.0])
    d_new = np.array([0.0, 60.0, 30.0, 0.0])          # deltas +100, -60, -30, +5
    v = M.value_clauses(d_ref, d_new, np.array(["a", "b", "c", "d"]), np.array([1, 1, 7, 7]), n_boot=50, seed=0)
    assert v["minus_top1_sse"] == pytest.approx(-85.0) and v["minus_top3_sse"] == pytest.approx(5.0)
    assert v["c4_minus_top3_positive"] is True


def test_fit_arm_refuses_a_non_finite_probability(M, monkeypatch):
    monkeypatch.setattr(M.rf, "fit_predict_fill", lambda tr, te, seed, permute=False, extra_num=(), extra_cat=():
                        np.array([0.5, np.nan]))
    with pytest.raises(ValueError, match="non-finite"):
        M.fit_arm(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1, 2]}), seeds=(0,), extra_num=(), extra_cat=())


def test_lomo_and_forward_fits_never_see_the_test_month(M, monkeypatch):
    """Recorder in place of the LightGBM fit: every training frame handed to the fit must exclude the held-out
    month (LOMO) or contain only STRICTLY earlier months (forward), for the unmatched AND the matched rows."""
    calls = []

    def rec(tr, te, seed, permute=False, extra_num=(), extra_cat=()):
        calls.append((sorted(tr.month.unique().tolist()), int(te.month.iloc[0]), bool(tr.unmatched.all()),
                      bool((~tr.unmatched).any())))
        return np.zeros(len(te))
    monkeypatch.setattr(M.rf, "fit_predict_fill", rec)
    n = 12 * 3
    u = pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float), "ADEP_mvt": "LIRF", "month": np.repeat(range(1, 13), 3),
                      "unmatched": True})
    l_ = pd.DataFrame({"MVT_ID_mvt": 1e6 + np.arange(n, dtype=float), "ADEP_mvt": "LIRF",
                       "month": np.repeat(range(1, 13), 3), "unmatched": False})
    M.lomo_fit(u, l_, list(range(1, 13)), {"R": ((), ())}, (0,), lambda m: None)
    assert len(calls) == 12
    for months, te_m, only_unm, has_matched in calls:
        assert te_m not in months and len(months) == 11 and has_matched and not only_unm
    calls.clear()
    M.forward_fit(u, l_, list(range(1, 13)), {"R": ((), ())}, (0,), lambda m: None)
    assert [c[1] for c in calls] == [7, 8, 9, 10, 11, 12]
    for months, te_m, _, has_matched in calls:
        assert max(months) < te_m and months == list(range(1, te_m)) and has_matched


def test_align_stored_refuses_a_stored_row_whose_sp_differs(M, synth):
    unm, _, _, stored, _, _ = synth
    u = unm[unm.ADEP_mvt == "LIRF"].reset_index(drop=True)
    M._align_stored(u, stored, "lomo")
    bad = stored.copy()
    bad.loc[bad.index[0], "sp"] += 1.0
    with pytest.raises(ValueError, match="differ"):
        M._align_stored(u, bad, "lomo")


@pytest.mark.slow
def test_run_ABORTS_when_the_stored_artefact_does_not_replicate(M, synth):
    unm, lirf, pool, stored, arr25, arr26 = synth
    bad = stored.copy()
    i = bad.index[bad.fold == "lomo"][0]
    bad.loc[i, "p_R_seed0"] = float(np.clip(bad.loc[i, "p_R_seed0"] + 0.01, 0, 1))
    with pytest.raises(ValueError, match="does not replicate"):
        M.run(unm, lirf, pool, bad, arr25, arr26, seeds=(0,), n_boot=20, log=lambda m: None, smoke=True)


@pytest.mark.slow
def test_run_clauses_are_the_decisional_arm_s_own_flags(M, synth):
    unm, lirf, pool, stored, arr25, arr26 = synth
    R = M.run(unm, lirf, pool, stored, arr25, arr26, seeds=(0,), n_boot=30, log=lambda m: None, smoke=True)
    D, F = R["lomo"]["FULL"], R["forward"]["FULL"]
    assert R["clauses"]["c1_auc_ge_0_95"] == (D["auc_pooled"] >= 0.95)
    for k in ("c2_ci_excludes_zero", "c3_both_months_positive", "c4_minus_top3_positive"):
        assert R["clauses"][k] == D["value_janjul"][k]
    assert R["clauses"]["c5_forward_positive"] == (F["value_vs_forward_R"]["gain_sse"] > 0
                                                   and F["auc_pooled_7_12"] >= R["forward"]["R"]["auc_pooled_7_12"])
    assert R["verdict"] == M.decide(D["auc_pooled"], R["clauses"])
    assert R["decisional_arm"] == "FULL" and R["decisional_auc"] == D["auc_pooled"] != R["lomo"]["R"]["auc_pooled"]
    for blk in (D["value_janjul"], D["value_12m"], R["oracle"]["two_class"]["janjul"]):
        assert blk["board_scaled"] == pytest.approx(M.scale_board(blk["gain_sse"], blk["n_rows"]))
        assert blk["scaling"] == f"sse x 383/n / 344841 with n = {blk['n_rows']}"
    assert R["witness"]["clause_both_ratios_above_one"] == M.witness_clause(R["witness"]["y2025"], R["witness"]["y2026"])
    assert R["n_janjul"] == D["value_janjul"]["n_rows"] and D["value_12m"]["n_rows"] == R["n_rows"]


def test_baseline_without_refit_touches_no_learner_and_reads_the_artefact(M, synth, monkeypatch):
    """The coordinator's order: the oracle and the artefact AUC are priced BEFORE any fit. With refit=False the
    baseline must never call the LightGBM fit and must mark the replication guard as deferred."""
    unm, lirf, pool, stored, _, _ = synth

    def boom(*a, **k):
        raise AssertionError("a fit was attempted in the no-refit baseline")
    monkeypatch.setattr(M.rf, "fit_predict_fill", boom)
    B = M.baseline(unm, lirf, stored, seeds=(0,), n_boot=30, log=lambda m: None, smoke=True, refit=False)
    assert B["refit"] is False and B["replication"]["passed"] is None and "deferred" in B["replication"]
    assert B["auc"]["refit_p_R"] is None and 0.0 <= B["auc"]["stored_p_R"] <= 1.0
    assert set(B["oracle"]["two_class"]) == {"janjul", "all12", "foldA"}
    json.dumps(B, default=str)


# ------------------------------------------------------------------ guard branches the coverage trace found untested

def test_prepare_refuses_the_wrong_row_count_and_a_missing_month(M):
    u = pd.DataFrame({"ADEP_mvt": ["LIRF"] * 12, "month": list(range(1, 13))})
    with pytest.raises(ValueError, match="registration says 1,488"):
        M._prepare(u, smoke=False)
    with pytest.raises(ValueError, match="registration fixed 12"):
        M._prepare(u.iloc[:11], smoke=True)


def test_lomo_fit_refuses_a_fold_without_test_rows(M):
    u = pd.DataFrame({"MVT_ID_mvt": [1.0], "ADEP_mvt": "LIRF", "month": [2], "unmatched": True})   # month 1 has no rows
    l_ = pd.DataFrame({"MVT_ID_mvt": [2.0], "ADEP_mvt": "LIRF", "month": [2], "unmatched": False})
    with pytest.raises(ValueError, match="no test rows"):
        M.lomo_fit(u, l_, [1, 2], {"R": ((), ())}, (0,), lambda m: None)


def test_witness_table_on_an_empty_frame(M):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00")])
    w = M.witness_table(pool.iloc[:0], pool)
    assert w == {"n_rows": 0, "copy_rate_all": None, "ratio_oneoff_vs_10plus": None,
                 "buckets": {k: {"n": 0, "copy_rate": None} for k in ("0-1", "2-4", "5-9", "10+")}}


def test_jsonable_covers_every_numpy_scalar_and_falls_back_to_str(M):
    assert M._jsonable(np.int64(3)) == 3 and M._jsonable(np.float32(1.5)) == 1.5 and M._jsonable(np.bool_(True)) is True
    assert M._jsonable(np.array([1])) == [1] and M._jsonable(pathlib.Path("x")) == "x"


def test_load_pool_reads_only_pool_columns_and_keeps_utc(M, tmp_path):
    pool = _pool([_mv(1, "AZA1", "2025-03-04 10:00", typ="A320"), _mv(2, "BAW2", "2025-03-05 10:00")])
    pool["EXTRA"] = 1
    pool.to_parquet(tmp_path / "a.parquet")
    pool.iloc[:1].to_parquet(tmp_path / "b.parquet")
    got = M.load_pool([tmp_path / "a.parquet", tmp_path / "b.parquet"])
    assert list(got.columns) == M.POOL_COLS and len(got) == 3
    assert str(got.MVT_TIME_UTC_mvt.dt.tz) == "UTC" and got.MVT_ID_mvt.tolist() == [1.0, 2.0, 1.0]


def test_main_refuses_an_existing_report(M, tmp_path):
    (tmp_path / "csf2_smoke_report.json").write_text("{}")
    with pytest.raises(SystemExit, match="second look"):
        M.main(["smoke", "--out-dir", str(tmp_path), "--seeds", "0", "--n-boot", "5"])
