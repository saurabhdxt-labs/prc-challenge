"""scripts/analysis/slc_slices.py — SLC: fixed slices of the January + July 2025 holdout (a public reference's definitions).

Registration: plans/PREREG_slc_slice_comparison_2026_10_04.md (REG 3ab3460f). Constructed rows only: each mask (no
flight-plan record, tail, normal, near a planned time, lottery), every slice RMSE against hand-computed values, the
false-alarm count and share, the per-airport table, the label-checked join to the raw rows, and the verdict's bars.

Written first and seen red 2026-10-04 (no module). Mutation rehearsal before the measurement (in place, one at a
time, file restored byte-identical): 15 / 15 killed — SL1 tail >= 3,600, SL2 normal includes 10,800, SL3 near strict,
SL4 near needs every planned time, SL5 lottery ignores near, SL6 nm inverted, SL7 normais_nm keeps rows without a plan
record, SL8 false alarms count tail rows, SL9 label check off, SL10 missing raw rows accepted, SL11 join by position,
SL12 / SL13 the verdict bars moved, SL14 cauda_copia = tail, SL15 sem_loteria drops nothing.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("slc_slices", ROOT / "scripts" / "analysis" / "slc_slices.py")
S = importlib.util.module_from_spec(_spec)
sys.modules["slc_slices"] = S
_spec.loader.exec_module(S)

T0 = pd.Timestamp("2025-01-10 12:00:00", tz="UTC")


def _frame(rows):
    """rows: dicts with y, pred and optional ap / fid / gaps {plan column: MVT − t in seconds}."""
    out = []
    for i, r in enumerate(rows):
        d = {"MVT_ID_mvt": float(i), "y": float(r["y"]), "pred": float(r["pred"]), "ap": r.get("ap", "EDDF"),
             "FLIGHT_ID_mvt": r.get("fid", 1.0), "MVT_TIME_UTC_mvt": T0}
        for c in S.PLAN_REFS:
            gap = r.get("gaps", {}).get(c)
            d[c] = T0 - pd.Timedelta(seconds=float(gap)) if gap is not None else pd.NaT
        out.append(d)
    f = pd.DataFrame(out)
    for c in S.PLAN_REFS + ["MVT_TIME_UTC_mvt"]:
        f[c] = pd.to_datetime(f[c], utc=True)
    return f


# ---------------------------------------------------------------------------------------------- masks

def test_plan_refs_are_the_five_planned_times():
    assert S.PLAN_REFS == ["SCHED_TIME_UTC_mvt", "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]


def test_near_is_within_300_s_of_any_planned_gap_and_null_is_never_near():
    f = _frame([{"y": 5000, "pred": 0, "gaps": {"SCHED_TIME_UTC_mvt": 5300}},      # exactly 300 away -> near
                {"y": 5000, "pred": 0, "gaps": {"SCHED_TIME_UTC_mvt": 5301}},      # 301 away -> not near
                {"y": 5000, "pred": 0, "gaps": {"SCHED_TIME_UTC_mvt": 9000, "AOBT_3_flt": 4800}},   # the last ref explains it
                {"y": 5000, "pred": 0}])                                           # no planned time at all
    np.testing.assert_array_equal(S.masks(f)["near"], [True, False, True, False])


def test_nm_tail_and_normal_bounds():
    f = _frame([{"y": 3600, "pred": 0}, {"y": 3601, "pred": 0}, {"y": 10800, "pred": 0}, {"y": 10799, "pred": 0},
                {"y": 600, "pred": 0, "fid": None}])
    m = S.masks(f)
    np.testing.assert_array_equal(m["tail"], [False, True, True, True, False])       # tail is y > 3,600
    np.testing.assert_array_equal(m["normal"], [True, True, False, True, True])      # normal is 0 < y < 10,800
    np.testing.assert_array_equal(m["nm"], [False, False, False, False, True])       # no flight-plan record


def test_lottery_is_above_three_hours_and_unexplained():
    f = _frame([{"y": 20000, "pred": 0},                                              # unexplained -> lottery
                {"y": 20000, "pred": 0, "gaps": {"LOBT_flt": 20100}},                 # explained by LOBT -> kept
                {"y": 10800, "pred": 0},                                              # not above 3 h -> kept
                {"y": 900, "pred": 0}])
    np.testing.assert_array_equal(S.masks(f)["lottery"], [True, False, False, False])


# ---------------------------------------------------------------------------------------------- metrics

def _mixed():
    return _frame([
        {"y": 600, "pred": 700, "ap": "EDDF"},                                        # normal, plan record: err 100
        {"y": 1200, "pred": 900, "ap": "LIRF"},                                       # normal, plan record: err 300
        {"y": 900, "pred": 4000, "ap": "LIRF"},                                       # false alarm: err 3,100
        {"y": 5000, "pred": 4000, "ap": "EDDF", "gaps": {"SCHED_TIME_UTC_mvt": 5000}},   # tail copy: err 1,000
        {"y": 20000, "pred": 2000, "ap": "LIRF", "fid": None},                        # lottery, no plan record: err 18,000
        {"y": 800, "pred": 600, "ap": "EDDF", "fid": None},                           # normal, no plan record: err 200
    ])


def _rmse(errs):
    return float(np.sqrt(np.mean(np.square(errs))))


def test_metrics_match_hand_computed_slices():
    m = S.metrics(_mixed())
    assert m["n"] == 6
    assert m["completo"] == pytest.approx(_rmse([100, 300, 3100, 1000, 18000, 200]))
    assert m["sem_outliers"] == pytest.approx(_rmse([100, 300, 3100, 1000, 200]))     # 0 < y < 10,800
    assert m["nm_presente"] == pytest.approx(_rmse([100, 300, 3100, 1000]))
    assert m["nm_ausente"] == pytest.approx(_rmse([18000, 200]))
    assert m["y_gt_1h"] == pytest.approx(_rmse([1000, 18000]))
    assert m["normais_nm"] == pytest.approx(_rmse([100, 300, 3100]))                  # y <= 3,600 with a plan record
    assert m["cauda_copia"] == pytest.approx(1000.0)
    assert m["sem_loteria"] == pytest.approx(_rmse([100, 300, 3100, 1000, 200]))
    assert m["n_by_slice"]["normais_nm"] == 3 and m["n_by_slice"]["lottery"] == 1


def test_false_alarms_count_and_share_of_squared_error():
    m = S.metrics(_mixed())
    total = sum(e * e for e in [100, 300, 3100, 1000, 18000, 200])
    assert m["alarmes_falsos"]["n"] == 1
    assert m["alarmes_falsos"]["parte_erro2"] == pytest.approx(3100 ** 2 / total)


def test_per_airport_rmse_uses_every_row_of_the_airport():
    m = S.metrics(_mixed())
    assert m["por_aeroporto"]["EDDF"] == pytest.approx(_rmse([100, 1000, 200]))
    assert m["por_aeroporto"]["LIRF"] == pytest.approx(_rmse([300, 3100, 18000]))


def test_empty_slice_is_nan_not_a_crash():
    m = S.metrics(_frame([{"y": 600, "pred": 700}]))
    assert np.isnan(m["nm_ausente"]) and np.isnan(m["cauda_copia"])


# ---------------------------------------------------------------------------------------------- the join

def _raw(ids, y):
    f = pd.DataFrame({"MVT_ID_mvt": np.asarray(ids, float), "TAXITIME_SEC_mvt": np.asarray(y, float),
                      "FLIGHT_ID_mvt": 1.0, "MVT_TIME_UTC_mvt": T0})
    for c in S.PLAN_REFS:
        f[c] = T0
    return f


def test_attach_refuses_an_oof_row_without_a_raw_row():
    oof = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "y": [600.0, 700.0], "pred": [1.0, 1.0], "ap": ["EDDF", "EDDF"]})
    with pytest.raises(ValueError, match="1 OOF row"):
        S.attach(oof, _raw([1], [600]))


def test_attach_refuses_a_label_mismatch():
    oof = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "y": [600.0, 700.0], "pred": [1.0, 1.0], "ap": ["EDDF", "EDDF"]})
    with pytest.raises(ValueError, match="labels disagree on 1 row"):
        S.attach(oof, _raw([1, 2], [600, 999]))


def test_attach_aligns_by_id():
    oof = pd.DataFrame({"MVT_ID_mvt": [2.0, 1.0], "y": [700.0, 600.0], "pred": [1.0, 1.0], "ap": ["EDDF", "EDDF"]})
    raw = _raw([1, 2], [600, 700])
    raw.loc[raw.MVT_ID_mvt == 2.0, "FLIGHT_ID_mvt"] = np.nan
    out = S.attach(oof, raw)
    assert list(out.MVT_ID_mvt) == [2.0, 1.0]
    np.testing.assert_array_equal(out.FLIGHT_ID_mvt.isna().to_numpy(), [True, False])


# ---------------------------------------------------------------------------------------------- the verdict

@pytest.mark.parametrize("ours,want", [(191.04, "TRUE"), (191.03, "NOT ESTABLISHED"), (189.55, "NOT ESTABLISHED"),
                                       (189.54, "FALSE"), (187.04, "FALSE (AHEAD)"), (187.05, "FALSE")])
def test_verdict_bars(ours, want):
    assert S.verdict_of(ours) == want
    assert S.REF_NORMAIS_NM == 189.04
