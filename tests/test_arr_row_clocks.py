"""Tests for ARC (scripts/analysis/arr_row_clocks.py; plans/PREREG_arr_row_clocks_2026_09_16.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import arr_row_clocks as A  # noqa: E402

H = 3600.0


def _dep():
    return pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0],
                         "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "EHAM", "LIRF"],
                         "ADES_mvt": ["EDDF", "EDDF", "LEMD", "LIRF", "EDDF"],
                         "FLIGHT_mvt": ["DLH1", "DLH2", "IBE3", "KLM4", None],
                         "mvt": [10_000.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0]})


def _arr():
    return pd.DataFrame({
        "ADEP_flt": ["LIRF", "LIRF", "LIRF", "LIRF", "EHAM", "LIRF"],
        "ADES_mvt": ["EDDF", "EDDF", "EDDF", "LEMD", "LIRF", "EDDF"],
        "CALLSIGN_flt": ["DLH1", "DLH1", "DLH2", "IBE3", "KLM4", None],
        # DLH1: two candidates in the window (earliest wins); DLH2: exactly at +600 (excluded); IBE3: at +8 h (included);
        # KLM4: in window but lobt missing (excluded); a callsign-less row never joins
        "mvt": [10_000 + 7_000.0, 10_000 + 5_000.0, 10_000 + 600.0, 10_000 + 8 * H, 10_000 + 3_000.0, 10_000 + 3_000.0],
        "lobt": [111.0, 222.0, 333.0, 444.0, np.nan, 666.0],
        "aobt3": [1111.0, 2222.0, 3333.0, 4444.0, 5555.0, 6666.0]})


def test_join_arr_window_callsign_earliest_and_missing_clocks():
    """Fails if the lower bound becomes inclusive, the upper exclusive, the earliest-candidate rule flips, a row without lobt
    joins, or a callsign-less row joins by time."""
    j = A.join_arr(_dep(), _arr())
    assert j.MVT_ID_mvt.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert j.arr_lobt.tolist()[0] == 222.0 and j.arr_aobt3.tolist()[0] == 2222.0
    assert np.isnan(j.arr_lobt[1]) and np.isnan(j.arr_lobt[3]) and np.isnan(j.arr_lobt[4])
    assert j.arr_lobt[2] == 444.0


def test_join_arr_requires_same_origin_and_destination():
    """Fails if ADEP or ADES is dropped from the join key."""
    a = _arr().assign(ADEP_flt="LFPG")
    assert A.join_arr(_dep(), a).arr_lobt.isna().all()
    a = _arr().assign(ADES_mvt="EGLL")
    assert A.join_arr(_dep(), a).arr_lobt.isna().all()


def test_join_arr_refuses_duplicate_departures():
    d = pd.concat([_dep(), _dep().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate MVT_ID_mvt"):
        A.join_arr(d, _arr())


def test_join_reads_no_label():
    """G2: corrupting label-like columns on either side changes nothing."""
    d, a = _dep(), _arr()
    base = A.join_arr(d, a)
    d2 = d.assign(y=np.arange(5) * 1e6, TAXITIME_SEC_mvt=1, BLOCK_TIME_UTC_mvt=pd.Timestamp("2000-01-01", tz="UTC"))
    a2 = a.assign(TAXITIME_SEC_mvt=-5, BLOCK_TIME_UTC_mvt=pd.Timestamp("1999-01-01", tz="UTC"))
    assert A.join_arr(d2, a2).equals(base)


@pytest.mark.parametrize("sp,sched,lobt,aobt3,want", [
    (5_000.0, 0.0, 3_600.0, 99_999.0, "ord"),        # |lobt − sched| == 3,600 → ord (inclusive)
    (5_000.0, 0.0, 3_601.0, 3_601.0 + 3_606, "fill"),  # re-filed; NM push within 3,606 of lobt (inclusive) → fill
    (5_000.0, 0.0, 3_601.0, 3_601.0 + 3_607, "none"),  # re-filed but NM push disagrees → none
    (3_599.0, 0.0, 0.0, 0.0, ""),                      # below scope
    (3_600.0, 0.0, 0.0, 0.0, "ord"),                   # scope lower bound inclusive
    (24_000.0, 0.0, 0.0, 0.0, ""),                     # scope upper bound exclusive
    (-5_000.0, 0.0, 9_000.0, 9_000.0, ""),             # negative sp never in scope
    (5_000.0, 0.0, np.nan, 0.0, ""),                   # not joined
    (np.nan, 0.0, 0.0, 0.0, ""),                       # sp missing
])
def test_sub_rule_boundaries(sp, sched, lobt, aobt3, want):
    """Fails if any threshold's inclusivity changes, the scope is dropped, or R_ord no longer takes precedence."""
    assert A.sub_rule([sp], [sched], [lobt], [aobt3]).tolist() == [want]


def test_arm_pred_applies_only_its_rule_and_floors_at_one():
    rule = np.array(["ord", "fill", "none", "", "ord"])
    inc = np.array([900.0, 1_000.0, 1_100.0, 1_200.0, 1_300.0])
    mvt = np.array([10_000.0] * 5)
    aobt3 = np.array([9_000.4, 0.0, 0.0, 0.0, 10_500.0])       # row 4: AOBT_3 after take-off → floor 1
    sp = np.array([4_000.0, 5_000.6, 6_000.0, 7_000.0, 8_000.0])
    assert A.arm_pred(inc, rule, "C", mvt, aobt3, sp).tolist() == [1_000.0, 1_000.0, 1_100.0, 1_200.0, 1.0]
    assert A.arm_pred(inc, rule, "D", mvt, aobt3, sp).tolist() == [900.0, 5_001.0, 1_100.0, 1_200.0, 1_300.0]
    assert A.arm_pred(inc, rule, "CD", mvt, aobt3, sp).tolist() == [1_000.0, 5_001.0, 1_100.0, 1_200.0, 1.0]
    assert inc.tolist() == [900.0, 1_000.0, 1_100.0, 1_200.0, 1_300.0]      # input not mutated
    with pytest.raises(ValueError, match="unknown arm"):
        A.arm_pred(inc, rule, "X", mvt, aobt3, sp)


def test_arm_mask():
    rule = np.array(["ord", "fill", "none", ""])
    assert A.arm_mask(rule, "C").tolist() == [True, False, False, False]
    assert A.arm_mask(rule, "D").tolist() == [False, True, False, False]
    assert A.arm_mask(rule, "CD").tolist() == [True, True, False, False]


def test_label_class_order_and_bounds():
    """ARC.1 order: fill before slip before ordinary. Fails if the order or a bound changes."""
    y = np.array([5_060.0, 5_061.0, 90_000.0, 82_800.0, 10_799.0, 10_800.0])
    sp = np.array([5_000.0, 5_000.0, 90_000.0, 1_000.0, 1_000.0, 1_000.0])
    assert A.label_class(y, sp).tolist() == ["fill", "ordinary", "fill", "slip", "ordinary", "other"]


def test_evaluate_arithmetic_moves_months_and_worst_row():
    """Fails if price omits the 2026/2025 scale or the board divisor, month counting sums the wrong sign, move-size splits use
    the label instead of the move, top-5 removal is ascending, or the worst row is scaled."""
    y = np.array([1_000.0, 1_000.0, 5_000.0, 1_000.0])
    inc = np.array([1_100.0, 1_000.0, 1_000.0, 1_000.0])
    pred = np.array([1_000.0, 1_050.0, 5_000.0, 1_000.0])
    # gains: 10,000 ; −2,500 ; 16,000,000 (move 4,000) ; 0
    ev = A.evaluate(y, inc, pred, month=[1, 2, 3, 3], block=["a", "b", "c", "c"], n26=8, n_boot=50)
    total = 10_000 - 2_500 + 16_000_000
    assert ev["gain_sse"] == pytest.approx(total)
    assert ev["price"] == pytest.approx(total / A.BOARD_N * 2.0, rel=1e-12)
    assert ev["months_with_rows"] == 3 and ev["months_positive"] == 2
    assert ev["gain_moves_gt300_sse"] == pytest.approx(16_000_000) and ev["gain_moves_gt1800_sse"] == pytest.approx(16_000_000)
    assert ev["share_gain_moves_gt1800"] == pytest.approx(16_000_000 / total)
    assert ev["gain_minus_top5_sse"] == pytest.approx(0.0)
    assert ev["rows_losing"] == 1 and ev["worst_row_board"] == pytest.approx(-2_500 / A.BOARD_N)


def test_evaluate_bootstrap_is_seeded_and_block_based():
    """Blocks, not rows, are resampled. Case 1: block 'a' = rows with gains +1, −1, +1 (sum +1), block 'b' = one row +1 (sum +1)
    → every block draw totals exactly 2, so the CI collapses to the price; per-row resampling would spread it. Case 2: 'a' = +3,
    'b' = −1 → totals ∈ {6, 2, −2}; the 95 % interval spans exactly −2…6. Fails if resampling becomes per-row or the seed is
    ignored (same seed must give identical intervals)."""
    y, inc, pred = np.array([1.0, 0.0, 1.0, 1.0]), np.zeros(4), np.ones(4)       # row gains +1, −1, +1, +1
    ev = A.evaluate(y, inc, pred, [1, 1, 1, 1], ["a", "a", "a", "b"], n26=4, n_boot=400)
    assert ev["price"] == pytest.approx(2.0 / A.BOARD_N)
    assert ev["ci95"] == pytest.approx([ev["price"], ev["price"]])
    y2 = np.array([1.0, 1.0, 1.0, 0.0])                                           # row gains +1, +1, +1, −1
    a = A.evaluate(y2, inc, pred, [1, 1, 1, 1], ["a", "a", "a", "b"], n26=4, n_boot=400, rng_seed=0)
    b = A.evaluate(y2, inc, pred, [1, 1, 1, 1], ["a", "a", "a", "b"], n26=4, n_boot=400, rng_seed=0)
    assert a["ci95"] == b["ci95"]
    assert a["ci95"] == pytest.approx([-2.0 / A.BOARD_N, 6.0 / A.BOARD_N])


def test_evaluate_empty():
    ev = A.evaluate([], [], [], [], [], n26=3)
    assert ev["n_2025"] == 0 and ev["price"] == 0.0 and ev["months_with_rows"] == 0


def _ev(**kw):
    base = dict(n_2025=100, price=150.0, months_positive=12, months_with_rows=12, ci95=[10.0, 300.0],
                worst_row_board=-10.0, gain_minus_top5_sse=1.0)
    return {**base, **kw}


@pytest.mark.parametrize("kw,want", [
    ({}, "WORKING"),
    (dict(price=99.9), "NOT WORKING"),
    (dict(months_positive=10), "NOT WORKING"),
    (dict(ci95=[0.0, 300.0]), "NOT WORKING"),
    (dict(worst_row_board=-60.1), "NOT WORKING"),
    (dict(worst_row_board=-60.0), "WORKING"),
    (dict(n_2025=14), "UNDERPOWERED"),
])
def test_verdict_C(kw, want):
    assert A.verdict("C", _ev(**kw)) == want


@pytest.mark.parametrize("kw,prec,calls,want", [
    (dict(price=300.0), 0.85, 15, "WORKING"),
    (dict(price=299.9), 0.9, 20, "NOT WORKING"),
    (dict(price=400.0), 0.84, 20, "NOT WORKING"),
    (dict(price=400.0), 0.9, 14, "NOT WORKING"),
    (dict(price=400.0), None, 20, "NOT WORKING"),
    (dict(price=400.0, months_positive=8, months_with_rows=12), 0.9, 20, "NOT WORKING"),
    (dict(price=400.0, months_positive=9, months_with_rows=12), 0.9, 20, "WORKING"),
    (dict(price=400.0, gain_minus_top5_sse=0.0), 0.9, 20, "NOT WORKING"),
])
def test_verdict_D(kw, prec, calls, want):
    assert A.verdict("D", _ev(**kw), prec, calls) == want


def test_verdict_CD_is_reported_only():
    assert A.verdict("CD", _ev()) == "REPORTED"


def test_sub_rule_ord_takes_precedence_when_both_conditions_hold():
    """LOBT within 1 h of SCHED AND NM's push within 3,606 s of LOBT → ord. (Dropping `~ord_` from the fill mask is an
    equivalent mutant: the output np.where already selects 'ord' first; this pins the observable behaviour.)"""
    assert A.sub_rule([5_000.0], [0.0], [100.0], [200.0]).tolist() == ["ord"]


def test_transfer_expectation_arithmetic():
    """Fails if the C term forgets to subtract the residual variance, the D term ignores the wrong-call branch or p, or rows of
    the other rule leak in."""
    rule = np.array(["ord", "fill", ""])
    v16 = np.array([1_000.0, 2_000.0, 5.0])
    new = np.array([800.0, 900.0, 5.0])
    sp = np.array([4_000.0, 6_000.0, 9.0])
    e = np.array([-10.0, 10.0])
    t = A.transfer_expectation(v16, new, sp, e, rule, p_fill=0.75)
    c = np.mean((800 + e - 1_000) ** 2 - e ** 2)                         # 40,000
    fill_g = (6_000 - 2_000) ** 2
    wrong = np.mean((900 + e - 2_000) ** 2 - (900 + e - 6_000) ** 2)
    assert c == pytest.approx(40_000.0)
    assert t["C"] == pytest.approx(c / A.BOARD_N) and t["C_rows"] == 1 and t["C_rows_moved_gt300"] == 0
    assert t["D"] == pytest.approx((0.75 * fill_g + 0.25 * wrong) / A.BOARD_N) and t["D_rows"] == 1
    assert t["D_rows_moved_gt1800"] == 1 and t["D_worst_wrong_call"] == pytest.approx(wrong / A.BOARD_N)
    with pytest.raises(ValueError, match="no 2025 R_ord residuals"):
        A.transfer_expectation(v16, new, sp, [], rule, 0.5)


def test_evaluate_bootstrap_same_seed_same_interval_on_many_blocks():
    """50 blocks with distinct gains and n_boot 30: the percentile interval depends on which draws occur, so an unseeded
    generator gives a different interval on the second call (mutation rehearsal 2026-09-16: default_rng() → RED)."""
    k = np.arange(50, dtype="float64")
    y, inc, pred = k, np.zeros(50), k * 0.5                     # distinct per-row gains
    kw = dict(month=np.ones(50), block=[f"b{i}" for i in range(50)], n26=50, n_boot=30)
    assert A.evaluate(y, inc, pred, **kw)["ci95"] == A.evaluate(y, inc, pred, **kw)["ci95"]
    assert A.evaluate(y, inc, pred, **kw, rng_seed=1)["ci95"] != A.evaluate(y, inc, pred, **kw)["ci95"]
