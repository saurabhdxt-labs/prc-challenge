"""RCR — the registered invariants of plans/PREREG_rcr_recipe_chronological_rebuild_2026_09_19.md.

Every test below names the failure mode it catches. The arm table and the chronological property are
the two things that, if wrong, would make the whole run silently meaningless.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def R():
    return _load("rcr_build", "scripts/analysis/rcr_build.py")


@pytest.fixture(scope="module")
def S():
    return _load("stand_ab", "scripts/stand_ab.py")


def test_arms_are_exactly_the_five_registered(R):
    """Catches an arm silently added, dropped or renamed after the prereg was hashed (the
    registration forbids it; a 6th arm would be an unregistered degree of freedom)."""
    assert list(R.ARMS) == ["A0", "A1", "A2", "A3", "A4"]


def test_a0_is_exactly_the_shipped_93_feature_recipe(R, S):
    """Catches the baseline drifting off the shipped recipe — the FCS defect that made its result
    non-binding (it compared against a proxy baseline, not the shipped model)."""
    feats = R.arm_feats("A0")
    assert len(feats) == 93, f"A0 must be the shipped 93, got {len(feats)}"
    expected = list(S.BASELINE_FEATS) + list(S.QUEUE_FEATS) + list(S.ORDER_FEATS) + list(S.WEATHER_FEATS)
    assert feats == expected


def test_a1_adds_exactly_the_nine_day_features_and_nothing_else(R, S):
    """Catches the DAY block being partially attached or duplicating a column already in A0."""
    extra = [c for c in R.arm_feats("A1") if c not in R.arm_feats("A0")]
    assert extra == list(S.DAY_FEATS)
    assert len(R.arm_feats("A1")) == 102


def test_only_a2_uses_target_y(R):
    """Catches the target silently flipping: scoring a delta model as if it predicted y (or the
    reverse) yields a plausible-looking but meaningless RMSE."""
    assert R.ARMS["A2"]["target"] == "y"
    assert all(R.ARMS[a]["target"] == "delta" for a in ["A0", "A1", "A3", "A4"])


def test_capacity_overrides_match_the_registration(R):
    """Catches a grid point drifting from the hashed registration."""
    assert R.ARMS["A3"]["params"] == {"num_leaves": 511, "min_data_in_leaf": 20}
    assert R.ARMS["A4"]["params"] == {"num_leaves": 127, "min_data_in_leaf": 20, "feature_fraction": 0.6}
    assert R.ARMS["A0"]["params"] == {}


def test_no_label_column_can_reach_any_arms_feature_list(R):
    """Leak gate 1. Catches a label or label-derived column entering the design matrix — the exact
    defect HCE's encoder had (its prior leaked a row's own label)."""
    banned = {"y", "delta", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "regime"}
    for arm in R.ARMS:
        assert not (banned & set(R.arm_feats(arm))), f"{arm} leaks a label column"


def test_chronological_split_puts_every_training_day_strictly_before_every_test_day(R):
    """THE registered property (leak gate 4). Catches the split degrading to interleaved, which is
    precisely the harness defect CHR measured at 6.38 RMSE s and FWD died of."""
    day = np.array([1, 2, 3, 10, 11, 12])
    is_test = np.array([False, False, False, True, True, True])
    tr = R.chrono_train_mask(day, is_test)
    assert tr.tolist() == [True, True, True, False, False, False]
    assert day[tr].max() < day[is_test].min()


def test_chrono_train_mask_never_includes_a_test_row(R):
    """Catches a TEST row being trained on — silent, and it would make every arm look better."""
    day = np.array([5, 5, 9, 9, 2])
    is_test = np.array([True, True, False, False, False])
    tr = R.chrono_train_mask(day, is_test)
    assert not (tr & is_test).any()
    assert day[tr].max() < day[is_test].min()


def test_holdout_months_are_never_scorable(R):
    """Catches months 1 or 7 (the board's scored season) entering the run."""
    assert set(R.HOLDOUT_MONTHS) == {1, 7}


def test_test_months_and_seed_match_the_registration(R):
    """Catches the split drifting from the hashed values, which would make the arms incomparable
    to the registration that authorised them."""
    assert tuple(R.TEST_MONTHS) == (11, 12)
    assert R.TEST_DAY_FRAC == 0.40
    assert R.SEED == 1709


def test_bar_is_the_registered_two_seconds(R):
    """Catches the verdict bar being softened after the fact."""
    assert R.GAIN_BAR_S == 2.0


def test_verdict_is_not_working_when_gain_below_bar(R):
    """Catches the ambiguous band being reported as WORKING — the registration says
    0 < gain < 2.0 is NOT WORKING."""
    ev = R.evaluate_arm(gain=1.2, ci=[0.4, 2.0])
    assert ev["verdict"].startswith("NOT WORKING")


def test_verdict_is_not_working_when_ci_includes_zero(R):
    """Catches a large point gain with a zero-spanning CI being called WORKING — the LSC failure
    shape (+388 with CI [-178, +1004])."""
    ev = R.evaluate_arm(gain=5.0, ci=[-1.0, 9.0])
    assert ev["verdict"].startswith("NOT WORKING")


def test_verdict_is_working_only_when_both_conditions_hold(R):
    ev = R.evaluate_arm(gain=2.5, ci=[0.8, 4.1])
    assert ev["verdict"] == "WORKING"
