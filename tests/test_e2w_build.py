"""Tests for scripts/analysis/e2w_build.py (plans/PREREG_e2w_stake_weighted_refit_2026_09_23.md)."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("e2w_build", ROOT / "scripts/analysis/e2w_build.py")
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)


def _world(n=3000, seed=0, beta=(0.7, 0.6, 1.2, -0.4)):
    rng = np.random.default_rng(seed)
    p_R = rng.uniform(0.05, 0.95, n)
    k_in = (rng.random(n) < 0.4).astype(float)
    turn = (rng.random(n) < 0.3).astype(float)
    X = W.design(p_R, k_in, turn)
    fill = rng.random(n) < W.predict(X, np.array(beta))
    month = rng.integers(1, 13, n)
    return X, fill, month


def test_design_columns_are_logit_pr_kin_turn_one():
    X = W.design(np.array([0.5, 0.8]), np.array([1., 0.]), np.array([0., 1.]))
    assert X.tolist() == [[0.0, 1.0, 0.0, 1.0], [pytest.approx(np.log(4)), 0.0, 1.0, 1.0]]


def test_design_clips_extreme_probabilities():
    X = W.design(np.array([0.0, 1.0]), np.zeros(2), np.zeros(2))
    assert np.isfinite(X).all()


def test_logloss_fit_recovers_generating_coefficients():
    X, fill, _ = _world(n=20000)
    b = W.fit_logloss(X, fill)
    assert b == pytest.approx([0.7, 0.6, 1.2, -0.4], abs=0.12)


def test_weighted_brier_with_unit_weights_recovers_coefficients_too():
    X, fill, _ = _world(n=20000)
    b = W.fit_brier_w(X, fill, np.ones(len(fill)), beta0=np.zeros(4))
    assert b == pytest.approx([0.7, 0.6, 1.2, -0.4], abs=0.2)


def test_weights_pull_the_fit_toward_heavy_rows():
    """Heavy rows are all fills with turn = 1: the weighted fit must raise the turn coefficient."""
    X, fill, _ = _world(n=5000, seed=1)
    w = np.ones(len(fill))
    heavy = (X[:, 2] == 1) & fill
    w[heavy] = 50.0
    b1 = W.fit_brier_w(X, fill, np.ones(len(fill)), beta0=np.zeros(4))
    b50 = W.fit_brier_w(X, fill, w, beta0=np.zeros(4))
    assert b50[2] > b1[2] + 0.5


def test_lomo_never_sees_the_held_out_month():
    """Scrambling month 5's labels must not change month 5's predictions."""
    X, fill, month = _world(n=4000, seed=2)
    w = np.ones(len(fill))
    a = W.lomo(X, fill, w, month, W.fit_brier_w)
    f2 = fill.copy()
    f2[month == 5] = ~f2[month == 5]
    b = W.lomo(X, f2, w, month, W.fit_brier_w)
    assert np.array_equal(a[month == 5], b[month == 5])
    assert not np.array_equal(a[month != 5], b[month != 5])


def test_lomo_fit_mask_limits_training_rows():
    X, fill, month = _world(n=4000, seed=3)
    w = np.ones(len(fill))
    fit_mask = np.ones(len(fill), bool)
    fit_mask[::2] = False
    a = W.lomo(X, fill, w, month, W.fit_logloss, fit_mask=fit_mask)
    f2 = fill.copy()
    f2[~fit_mask] = ~f2[~fit_mask]                     # rows outside the fit mask never reach any fit
    b = W.lomo(X, f2, w, month, W.fit_logloss, fit_mask=fit_mask)
    assert np.array_equal(a, b)


def test_verdict_each_clause_in_isolation():
    ok_m = {m: 1.0 for m in range(1, 13)}
    assert W.verdict(1.0, 1.0, 1.0, ok_m) == "WORKING"
    assert W.verdict(0.0, 1.0, 1.0, ok_m) == "NOT WORKING"                   # CI lower > 0
    assert W.verdict(1.0, 0.0, 1.0, ok_m) == "NOT WORKING"                   # minus-top-3 > 0
    assert W.verdict(1.0, 1.0, -1e-9, ok_m) == "NOT WORKING"                 # Jan+Jul >= 0
    assert W.verdict(1.0, 1.0, 0.0, ok_m) == "WORKING"
    six = {m: (1.0 if m <= 6 else -1.0) for m in range(1, 13)}
    seven = {m: (1.0 if m <= 7 else 0.0) for m in range(1, 13)}
    assert W.verdict(1.0, 1.0, 1.0, six) == "NOT WORKING"                    # >= 7 of 12 months > 0
    assert W.verdict(1.0, 1.0, 1.0, seven) == "WORKING"
