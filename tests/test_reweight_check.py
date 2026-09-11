"""scripts/reweight_check.py: RWC (plans/PREREG_reweight_check_2026_09_11.md), on synthetic data with a known answer."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("reweight_check", ROOT / "scripts" / "reweight_check.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _X(rng, n, shift=0.0):
    return pd.DataFrame({"first_ground_rel_aobt3": rng.normal(shift, 1, n), "first_move_rel_mvt": rng.normal(0, 1, n),
                         "first_ground_gs": rng.normal(0, 1, n), "coverage": rng.choice([2, 3], n)})


def test_crossfit_ratio_is_flat_without_shift_and_tilts_toward_the_target_with_one():
    """No shift: AUC near 0.5 and weights near 1. A +1 shift in one feature: AUC well above 0.5, and the base rows that look
    like the target get the larger weights. Rehearsed 2026-09-11 (c70): the odds p/(1-p) inverted -> RED."""
    rng = np.random.default_rng(0)
    base = _X(rng, 4000)
    w0, auc0 = R.crossfit_ratio(_X(rng, 4000), base)
    assert auc0 < 0.55 and abs(w0.mean() - 1) < 1e-9 and R.kish_ess(w0) > 0.9 * len(w0)
    w1, auc1 = R.crossfit_ratio(_X(rng, 4000, shift=1.0), base)
    assert auc1 > 0.65
    hi = base.first_ground_rel_aobt3.to_numpy() > 1
    assert w1[hi].mean() > 2 * w1[~hi].mean()


def test_crossfit_ratio_scores_every_base_row_with_a_model_that_did_not_see_it(monkeypatch):
    """Cross-fitting is a property of the fold loop, not of the classifier's settings. So the test gives the classifier
    the capacity to memorise (min_data_in_leaf 5, 400 rounds) on two independent samples of ONE distribution: scored out
    of fold the AUC stays near 0.5; scored in-sample it memorises. Rehearsed 2026-09-11 (c70): predicting every row with a
    model fitted on all rows -> RED (the registered, less overfit settings alone could not show this: the first version
    of this test survived that mutation)."""
    monkeypatch.setattr(R, "PARAMS", dict(R.PARAMS, min_data_in_leaf=5, num_leaves=63))
    monkeypatch.setattr(R, "ROUNDS", 400)
    rng = np.random.default_rng(1)
    _, auc = R.crossfit_ratio(_X(rng, 1500), _X(rng, 1500))
    assert auc < 0.56


def test_kish_ess_and_weighted_gain():
    assert R.kish_ess(np.ones(10)) == pytest.approx(10.0)
    assert R.kish_ess(np.array([1.0, 0.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert R.weighted_gain(np.array([1.0, 3.0]), np.array([1.0, 3.0])) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="non-finite"):
        R.weighted_gain(np.array([1.0, np.nan]), np.ones(2))


@pytest.mark.parametrize("G,Gw,lo,ess,n,want", [
    (100.0, 80.0, 5.0, 900.0, 1000, "SHIP"),
    (100.0, 49.0, 5.0, 900.0, 1000, "KEEP_F"),        # C2: below half the unweighted gain
    (100.0, 80.0, -1.0, 900.0, 1000, "KEEP_F"),       # C1: the interval reaches zero
    (100.0, 80.0, 5.0, 199.0, 1000, "INCONCLUSIVE"),  # V1: ESS below 0.2 n
    (100.0, 50.0, 0.001, 200.0, 1000, "SHIP"),        # every bar met exactly
])
def test_verdict_applies_the_three_clauses_as_registered(G, Gw, lo, ess, n, want):
    """Rehearsed (c70): C2's 0.5 -> 0.4 -> RED (row 2); V1's >= -> > -> RED (last row)."""
    assert R.verdict(G, Gw, lo, ess, n)["decision"] == want


@pytest.mark.parametrize("auc,ess_frac,inside,want", [
    (0.51, 0.97, True, "PASS"),
    (0.56, 0.97, True, "FAIL"),
    (0.51, 0.85, True, "FAIL_ESS_ONLY"),
    (0.51, 0.97, False, "FAIL"),
])
def test_k0_rule_and_its_ess_only_shape(auc, ess_frac, inside, want):
    """RWC.1: a failure on the ESS clause alone is reported as its own shape; any K0 failure still keeps F everywhere."""
    assert R.k0_rule(auc, ess_frac, inside) == want


def test_kplus_rule():
    assert R.kplus_rule(G=100.0, Gw=90.0, Gtrue=80.0) == "PASS"           # recovered half of a 20-unit plant
    assert R.kplus_rule(G=100.0, Gw=99.0, Gtrue=80.0) == "FAIL"
    assert R.kplus_rule(G=100.0, Gw=100.0, Gtrue=99.0) == "INCONCLUSIVE"   # plant < 2% of G


def test_normal_score_is_within_airport_and_zero_for_missing():
    ap = np.array(["A", "A", "A", "B", "B"])
    x = np.array([1.0, 2.0, 3.0, 10.0, np.nan])
    z = R.normal_score(x, ap)
    assert z[1] == pytest.approx(0.0) and z[0] < 0 < z[2] and z[4] == 0.0 and z[3] == pytest.approx(0.0)
