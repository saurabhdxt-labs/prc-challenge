"""Tests for scripts/analysis/arscat_build.py (ARS-CAT, plans/PREREG_arscat_joint_key_catboost_2026_09_20.md).

Written FIRST and seen RED (ImportError) before the module existed.
`ars` is reconstructed from the existing airport-scoped keys, never rebuilt from source, and it enters
CatBoost ONLY through CAT_COLS (ordered target statistics) -- raw `ars` measured -919.5 and is refused.

Mutation rehearsal (ARS-CAT), __pycache__ purged between mutants (BC-16):
  EXECUTED 2026-09-21, all KILLED: M1, M3, M8, M9, M10, M11, M12, M13
  NOT EXECUTED (intended set, design notes only -- NOT evidence): M2, M4, M5, M6, M7
An earlier revision of this file asserted "KILLED" for every mutant in the list without
having run them. That was a false record; it was corrected at end-work after an adversarial
reviewer caught it, and three of the unexecuted mutants were then found to ESCAPE and the
corresponding tests strengthened.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis import arscat_build as A


def test_joint_key_matches_the_lightgbm_definition():
    """stand_ab.py:344 builds ars = ap|STAND|RUNWAY; ap_stand = ap|STAND, ap_rwy = ap|RUNWAY."""
    ap_stand = pd.Series(["EDDF|A12", "LIRF|305"])
    ap_rwy = pd.Series(["EDDF|25L", "LIRF|16R"])
    assert list(A.joint_key(ap_stand, ap_rwy)) == ["EDDF|A12|25L", "LIRF|305|16R"]


def test_joint_key_preserves_the_NA_literal():
    got = A.joint_key(pd.Series(["EDDF|NA"]), pd.Series(["EDDF|NA"]))
    assert list(got) == ["EDDF|NA|NA"]


def test_joint_key_refuses_an_airport_prefix_mismatch():
    with pytest.raises(ValueError):
        A.joint_key(pd.Series(["EDDF|A12"]), pd.Series(["LIRF|25L"]))


def test_joint_key_refuses_a_key_without_a_separator():
    with pytest.raises(ValueError):
        A.joint_key(pd.Series(["EDDF|A12"]), pd.Series(["EDDF25L"]))


def test_joint_key_keeps_a_runway_containing_a_separator():
    got = A.joint_key(pd.Series(["EDDF|A12"]), pd.Series(["EDDF|25|L"]))
    assert list(got) == ["EDDF|A12|25|L"]


def test_cat_cols_with_ars_appends_once_and_keeps_the_shipped_order():
    base = ["ap", "ap_stand", "ap_rwy", "ap_ades"]
    got = A.cat_cols_with_ars(base)
    assert got[: len(base)] == base
    assert got.count("ars") == 1 and got[-1] == "ars"
    assert A.cat_cols_with_ars(got) == got          # idempotent


def test_control_cat_cols_never_contain_ars():
    assert "ars" not in A.cat_cols_control(["ap", "ap_stand", "ars"])


def test_gain_board_is_control_minus_treatment_over_N():
    y = np.array([0.0, 0.0])
    ctrl = np.array([10.0, 0.0])
    trt = np.array([0.0, 0.0])
    assert A.gain_board(y, ctrl, trt, n=100.0) == pytest.approx(1.0)
    assert A.gain_board(y, trt, ctrl, n=100.0) == pytest.approx(-1.0)


def test_stage1_gate_requires_positive_mean_and_ci_excluding_zero():
    assert A.stage1_passes(gain=10.0, ci=(1.0, 20.0)) is True
    assert A.stage1_passes(gain=10.0, ci=(-1.0, 20.0)) is False     # CI includes 0
    assert A.stage1_passes(gain=-10.0, ci=(-20.0, -1.0)) is False   # wrong sign
    assert A.stage1_passes(gain=0.0, ci=(0.0, 5.0)) is False        # not strictly positive
    # CI cleanly excludes zero, so ONLY the strict `gain > 0` clause can reject this one.
    assert A.stage1_passes(gain=0.0, ci=(1.0, 5.0)) is False


def test_stage1_gate_is_the_registered_clause_not_a_size_bar():
    """The registration forbids an RMSE-second size bar; a tiny positive with a clean CI must pass."""
    assert A.stage1_passes(gain=0.01, ci=(0.001, 0.05)) is True


def test_iteration_counts_are_the_registered_ones():
    assert A.STAGE1_ITERS == 1000
    assert A.SHIPPED_ITERS == 6664
    assert A.STAGE1_ITERS < A.SHIPPED_ITERS


def test_both_arms_must_use_the_same_iteration_count():
    with pytest.raises(ValueError):
        A.assert_fair_screen(control_iters=1000, candidate_iters=1500)
    A.assert_fair_screen(control_iters=1000, candidate_iters=1000)


def test_refuses_raw_ars_as_a_numeric_feature():
    """Raw high-cardinality ars measured -919.5; it may only enter through CAT_COLS."""
    with pytest.raises(ValueError):
        A.assert_ars_is_categorical(numeric_cols=["proxy", "ars"], cat_cols=["ap", "ars"])
    A.assert_ars_is_categorical(numeric_cols=["proxy"], cat_cols=["ap", "ars"])


def test_regression_numeric_design_never_contains_the_label():
    """Caught by the quality gate dry run 2026-09-21: deriving the design by exclusion leaked score_y."""
    with pytest.raises(ValueError, match="score_y"):
        A.assert_no_label_columns(["proxy", "sp", "score_y"])


@pytest.mark.parametrize("col", ["score_y", "score_proxy", "score_sp", "delta", "G",
                                 "train", "fit", "es", "holdout", "row", "MVT_ID_mvt", "month"])
def test_regression_every_label_or_mask_column_is_refused(col):
    with pytest.raises(ValueError):
        A.assert_no_label_columns(["proxy", col])


def test_regression_day_blocked_bootstrap_refuses_degenerate_blocks():
    """2026-09-21: `day` fell back to `month`, giving ONE block per month and a zero-width CI."""
    month = np.array([1] * 50 + [7] * 50)
    with pytest.raises(ValueError, match="day blocks"):
        A.assert_day_blocks_are_real(month.copy(), month)          # day == month: 1 block per month


def test_real_day_blocks_are_accepted():
    month = np.array([1] * 50 + [7] * 50)
    day = np.r_[np.arange(50) % 25 + 1, np.arange(50) % 25 + 182]
    A.assert_day_blocks_are_real(day, month)


def test_a_clean_numeric_design_is_accepted():
    A.assert_no_label_columns(["proxy", "sp", "hr", "dow", "q_rwy_at_push"])


