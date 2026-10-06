"""scripts/analysis/rul_ruler.py — RUL: does our offline 2025 gain predict the board, and which part of it?

Registration: plans/PREREG_rul_ruler_validation_2026_10_04.md (REG 7d789c25). These tests pin the statistics, the
verdict thresholds, the generic pair loader and the order rule (every version's reproduction check is decided before
any body / tail split is computed) on constructed rows; the real pairs are checked at run time by reproducing each
version's recorded price.

Written first and seen red 2026-10-04 (no module). Mutation rehearsal before the measurement (scratch copy, one at a
time, source verified unchanged): 12 / 12 killed — RU1 body cut exclusive, RU2 top rows by gain not |gain| (killed once
the |g| test existed), RU3 / RU4 threshold edges, RU5 board delta sign, RU6 tolerance exclusive, RU7 an unreproduced
version split anyway, RU8 the price checked on the split rows (killed once the values cleared the tolerance), RU9 no
floor at 1 s in the delta conversion, RU10 a keyed join without the reindex, RU11 no label check, RU12 cells by airport only.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rul_ruler", ROOT / "research" / "analysis" / "rul_ruler.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["rul_ruler"] = R                     # dataclasses resolve their module through sys.modules
_spec.loader.exec_module(R)


# ---------------------------------------------------------------------------------------------- statistics

def test_row_gain_is_positive_when_the_new_prediction_is_closer():
    g = R.row_gain(np.array([100.0, 200.0]), np.array([110.0, 200.0]), np.array([100.0, 190.0]))
    assert g.tolist() == [100.0, -100.0]


def test_row_gain_refuses_non_finite_or_misaligned_input():
    with pytest.raises(ValueError, match="non-finite"):
        R.row_gain(np.array([1.0, np.nan]), np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="lengths"):
        R.row_gain(np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0]))


def test_gain_split_separates_body_and_tail_and_drops_the_top_k_by_absolute_gain():
    y = np.array([600.0, 3600.0, 3601.0, 900.0, 5000.0])          # 3,600 s is body (inclusive), 3,601 s is tail
    a = np.array([610.0, 3700.0, 3601.0, 900.0, 4000.0])
    b = np.array([600.0, 3600.0, 3701.0, 950.0, 5000.0])
    g = (y - a) ** 2 - (y - b) ** 2                                # [100, 10000, -10000, -2500, 1000000]
    s = R.gain_split(y, a, b, top_k=2)
    n = R.BOARD_N
    assert s["G"] == pytest.approx(g.sum() / n)
    assert s["B"] == pytest.approx((100 + 10000 - 2500) / n)
    assert s["T"] == pytest.approx((-10000 + 1000000) / n)
    # the two largest |g| go: 1e6, then the first of the |1e4| tie (stable order: +1e4, row 1)
    assert s["G_minus_top"] == pytest.approx((100 - 10000 - 2500) / n)
    assert s["n"] == 5 and s["n_body"] == 3


def test_the_top_k_rows_are_the_largest_by_absolute_gain_losses_included():
    """A huge loss is as much a 'few rows carry it' signal as a huge gain: |g| ranks, not g (RU2)."""
    y = np.array([100.0, 100.0, 100.0])
    a = np.array([105.0, 100.0, 103.0])                          # g = [25, -10000, 9]
    b = np.array([100.0, 200.0, 100.0])
    assert R.gain_split(y, a, b, top_k=1)["G_minus_top"] == pytest.approx((25 + 9) / R.BOARD_N)


def test_board_delta_is_positive_when_the_new_file_scores_better():
    assert R.board_delta(254.0762, 252.0706) == pytest.approx(254.0762 ** 2 - 252.0706 ** 2)
    assert R.board_delta(250.7314, 252.1294) < 0


def test_reproduction_names_the_convention_the_price_used_or_refuses():
    n_rows = 10_000
    g_sum = 99.31 * R.BOARD_N
    assert R.reproduces(g_sum, n_rows, 99.31) == "bmse"
    assert R.reproduces(99.31 * n_rows, n_rows, 99.31) == "per_row"
    assert R.reproduces(g_sum, n_rows, 99.31 + 0.5) == "bmse"                 # tolerance is inclusive
    assert R.reproduces(g_sum, n_rows, 99.31 + 0.51) is None


def test_spearman_ratio_and_sign_agreement():
    b = np.array([1.0, 2.0, 3.0, -1.0])
    d = np.array([2.0, 5.0, 4.0, 1.0])                       # ranks 2,4,3,1 vs 2,3,4,1: rho = 1 - 6*2/60 = 0.8
    assert R.spearman(b, d) == pytest.approx(0.8)
    assert R.sign_agreement(b, d) == 0.75
    assert R.ratio_through_origin(b, d) == pytest.approx(float((b * d).sum() / (b * b).sum()))


@pytest.mark.parametrize("rho, sign, ratio, want", [
    (0.60, 0.75, 0.30, "TRUE"), (0.60, 0.75, 3.00, "TRUE"),
    (0.60, 0.75, 3.01, "NOT ESTABLISHED"), (0.60, 0.74, 1.0, "NOT ESTABLISHED"),
    (0.59, 0.90, 1.0, "NOT ESTABLISHED"), (0.399, 0.90, 1.0, "FALSE"), (0.90, 0.599, 1.0, "FALSE"),
    (0.40, 0.60, 1.0, "NOT ESTABLISHED")])
def test_verdict_applies_the_registered_thresholds_at_their_edges(rho, sign, ratio, want):
    assert R.verdict(rho, sign, ratio) == want


# ---------------------------------------------------------------------------------------------- the generic pair loader

def _write(tmp_path, name, frame):
    p = tmp_path / name
    frame.to_parquet(p, index=False)
    return p


def test_a_pair_in_one_file_with_a_filter_and_the_delta_conversion(tmp_path):
    f = _write(tmp_path, "p.parquet", pd.DataFrame({"y": [100.0, 200.0, 300.0], "proxy": [150.0, 260.0, 10.0],
                                                    "inc": [40.0, 50.0, 20.0], "new": [50.0, 70.0, 30.0],
                                                    "ap": ["EDDF", "LIRF", "EDDF"]}))
    spec = R.Spec(version=1, base_rmse=1.0, new_rmse=1.0, ref=R.Side(str(f), "inc", delta=True),
                  new=R.Side(str(f), "new", delta=True), rows={"ap": ("EDDF",)}, recorded=0.0)
    y, a, b = R.load_pair(spec, rows=spec.rows)
    assert y.tolist() == [100.0, 300.0]
    assert a.tolist() == [110.0, 1.0] and b.tolist() == [100.0, 1.0]           # max(proxy - delta, 1)


def test_two_files_joined_on_row_in_any_order_and_refused_when_labels_differ(tmp_path):
    ref = _write(tmp_path, "r.parquet", pd.DataFrame({"row": [0, 1, 2], "y": [10.0, 20.0, 30.0], "proxy": [0.0] * 3,
                                                      "p": [11.0, 21.0, 31.0]}))
    new = _write(tmp_path, "n.parquet", pd.DataFrame({"row": [2, 0, 1], "y": [30.0, 10.0, 20.0], "proxy": [0.0] * 3,
                                                      "q": [30.0, 10.0, 22.0]}))
    spec = R.Spec(version=2, base_rmse=1.0, new_rmse=1.0, ref=R.Side(str(ref), "p"), new=R.Side(str(new), "q"),
                  key="row", recorded=0.0)
    y, a, b = R.load_pair(spec, rows=None)
    assert y.tolist() == [10.0, 20.0, 30.0] and b.tolist() == [10.0, 22.0, 30.0]
    bad = _write(tmp_path, "bad.parquet", pd.DataFrame({"row": [2, 0, 1], "y": [30.0, 99.0, 20.0],
                                                        "proxy": [0.0] * 3, "q": [1.0] * 3}))
    with pytest.raises(ValueError, match="labels differ"):
        R.load_pair(R.Spec(version=2, base_rmse=1.0, new_rmse=1.0, ref=R.Side(str(ref), "p"),
                           new=R.Side(str(bad), "q"), key="row", recorded=0.0), rows=None)


def test_two_files_by_position_are_refused_unless_rows_and_labels_match(tmp_path):
    ref = _write(tmp_path, "r.parquet", pd.DataFrame({"row": [0, 1], "y": [10.0, 20.0], "p": [11.0, 21.0]}))
    new = _write(tmp_path, "n.parquet", pd.DataFrame({"row": [1, 0], "y": [20.0, 10.0], "p": [20.0, 10.0]}))
    with pytest.raises(ValueError, match="row order differs"):
        R.load_pair(R.Spec(version=3, base_rmse=1.0, new_rmse=1.0, ref=R.Side(str(ref), "p"),
                           new=R.Side(str(new), "p"), recorded=0.0), rows=None)


def test_rows_by_cell_select_airport_month_pairs(tmp_path):
    f = _write(tmp_path, "c.parquet", pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "a": [1.0] * 4, "b": [1.0] * 4,
                                                    "ap": ["EGLL", "EGLL", "EDDF", "LFPG"], "month": [1, 7, 7, 1]}))
    spec = R.Spec(version=4, base_rmse=1.0, new_rmse=1.0, ref=R.Side(str(f), "a"), new=R.Side(str(f), "b"),
                  rows={"cells": (("EGLL", 1), ("EDDF", 7))}, recorded=0.0)
    y, _, _ = R.load_pair(spec, rows=spec.rows)
    assert y.tolist() == [1.0, 3.0]


# ---------------------------------------------------------------------------------------------- the order rule

def test_a_version_that_does_not_reproduce_its_price_is_excluded_before_any_split(tmp_path, monkeypatch):
    f = _write(tmp_path, "v.parquet", pd.DataFrame({"y": [100.0, 5000.0], "a": [110.0, 4000.0],
                                                    "b": [100.0, 5000.0]}))
    g_sum = (100.0 + 1_000_000.0)
    ok = R.Spec(version=5, base_rmse=250.0, new_rmse=249.0, ref=R.Side(str(f), "a"), new=R.Side(str(f), "b"),
                recorded=g_sum / R.BOARD_N)
    bad = R.Spec(version=6, base_rmse=250.0, new_rmse=249.0, ref=R.Side(str(f), "a"), new=R.Side(str(f), "b"),
                 recorded=123.0)
    calls = []
    real_split = R.gain_split

    def spy(*a, **k):
        calls.append(len(calls))
        return real_split(*a, **k)
    monkeypatch.setattr(R, "gain_split", spy)
    checks, rows = R.evaluate([ok, bad])
    assert [c["version"] for c in checks] == [5, 6] and checks[0]["convention"] == "bmse"
    assert checks[1]["convention"] is None and "does not reproduce" in checks[1]["excluded"]
    assert [r["version"] for r in rows] == [5] and len(calls) == 1          # no split for the excluded version
    assert rows[0]["D"] == pytest.approx(250.0 ** 2 - 249.0 ** 2)


def test_the_price_check_uses_its_own_rows_when_they_differ_from_the_rows_used(tmp_path):
    """v46 / v49: the recorded raw price is on all rows, the split on the served cells."""
    f = _write(tmp_path, "w.parquet", pd.DataFrame({"y": [100.0, 100.0], "a": [1100.0, 1600.0], "b": [100.0, 100.0],
                                                    "ap": ["EDDF", "LFPG"], "month": [1, 1]}))
    recorded = (1000.0 ** 2 + 1500.0 ** 2) / R.BOARD_N          # 9.42 on all rows; EDDF alone would give 2.90 (RU8)
    spec = R.Spec(version=7, base_rmse=1.0, new_rmse=1.0, ref=R.Side(str(f), "a"), new=R.Side(str(f), "b"),
                  rows={"ap": ("EDDF",)}, price_rows=None, price_on_all=True, recorded=recorded)
    checks, rows = R.evaluate([spec])
    assert checks[0]["convention"] == "bmse" and checks[0]["n_price"] == 2
    assert rows[0]["G"] == pytest.approx(1000.0 ** 2 / R.BOARD_N) and rows[0]["n"] == 1
