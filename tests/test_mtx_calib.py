"""Tests for scripts/analysis/mtx_calib.py — published-monthly-mean calibration (arm MTX,
plans/PREREG_mtx_monthly_calibration_2026_09_28.md). Written before the module (red first)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("mtx_calib", ROOT / "scripts" / "analysis" / "mtx_calib.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mtx_calib"] = mod
    spec.loader.exec_module(mod)
    return mod


mc = _load()


def _cells(c=40.0, e=900.0, seed=0):
    """12 synthetic airport-months that follow the excluded-mean model exactly: n*(mean - pub) = NR*c + INV*e."""
    rng = np.random.default_rng(seed)
    n = rng.integers(10_000, 20_000, 12).astype(float)
    nr = rng.integers(100, 900, 12).astype(float)
    inv = np.r_[rng.integers(500, 3000, 3), rng.integers(0, 20, 9)].astype(float)
    pub = rng.uniform(700, 900, 12)
    mean = pub + (nr * c + inv * e) / n
    return pd.DataFrame({"ap": "EDDM", "month": np.arange(1, 13), "n": n, "TF": n, "NR": nr, "INV": inv,
                         "pub_mean_s": pub, "mean_label": mean})


def test_fit_excess_recovers_the_generating_parameters():
    """Kills: an intercept in the fit; normalising by P instead of n; swapping the NR and INV coefficients."""
    c, e = mc.fit_excess(_cells(c=40.0, e=900.0))
    assert c == pytest.approx(40.0, abs=1e-6) and e == pytest.approx(900.0, abs=1e-6)


def test_fit_excess_with_no_invalid_flights_gives_e_zero_not_garbage():
    d = _cells()
    d["INV"] = 0.0
    d["mean_label"] = d.pub_mean_s + d.NR * 40.0 / d.n
    c, e = mc.fit_excess(d)
    assert c == pytest.approx(40.0, abs=1e-6) and e == 0.0


def test_fit_excess_refuses_nan_and_empty():
    d = _cells()
    d.loc[3, "mean_label"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mc.fit_excess(d)
    with pytest.raises(ValueError, match="empty"):
        mc.fit_excess(d.iloc[:0])


def test_target_mean_is_pub_plus_excluded_excess_per_flight():
    t = mc.target_mean(pub=800.0, NR=100.0, INV=50.0, TF=1000.0, c=40.0, e=900.0)
    assert t == pytest.approx(800.0 + (100 * 40.0 + 50 * 900.0) / 1000.0)
    with pytest.raises(ValueError, match="TF"):
        mc.target_mean(pub=800.0, NR=1.0, INV=1.0, TF=0.0, c=1.0, e=1.0)


def test_lmo_targets_never_use_the_scored_month():
    """Kills leakage: perturb month 1's label mean; its own LMO target must not move, other months' may."""
    d = _cells()
    base = mc.lmo_targets(d).set_index("month")
    d2 = d.copy()
    d2.loc[d2.month == 1, "mean_label"] += 500.0
    pert = mc.lmo_targets(d2).set_index("month")
    assert pert.loc[1, "T"] == pytest.approx(base.loc[1, "T"], abs=1e-9)
    assert abs(pert.loc[2, "T"] - base.loc[2, "T"]) > 1e-6
    assert pert.loc[1, "err"] == pytest.approx(base.loc[1, "err"] - 500.0, abs=1e-9)


def test_regime_sigma_uses_months_of_the_same_invalid_share_regime():
    """High-INV target -> RMS of the high-INV months' LMO errors; a regime with one month -> |that error|;
    a regime with no month -> inf (no shift)."""
    lmo = pd.DataFrame({"inv_share": [0.10, 0.05, 0.001, 0.0, 0.002], "err": [3.0, -4.0, 1.0, -1.0, 1.0]})
    assert mc.regime_sigma(lmo, 0.08) == pytest.approx(np.sqrt((9 + 16) / 2))
    assert mc.regime_sigma(lmo, 0.0005) == pytest.approx(1.0)
    assert mc.regime_sigma(lmo.iloc[[0, 2, 3]], 0.03) == pytest.approx(3.0)
    assert mc.regime_sigma(lmo.iloc[[2, 3]], 0.03) == np.inf


def test_shrink_is_james_stein_on_the_gap_and_zero_when_the_gap_is_inside_the_noise():
    assert mc.shrink(10.0, 5.0) == pytest.approx(0.75)
    assert mc.shrink(-10.0, 5.0) == pytest.approx(0.75)
    assert mc.shrink(3.0, 5.0) == 0.0
    assert mc.shrink(0.0, 5.0) == 0.0
    assert mc.shrink(10.0, np.inf) == 0.0
    assert mc.shrink(10.0, 0.0) == 1.0


def test_shift_additive_moves_each_group_mean_by_lambda_times_the_gap():
    """Kills: using the global mean; ignoring lambda; moving groups absent from the target table."""
    pred = np.array([100.0, 200.0, 300.0, 50.0])
    key = np.array(["a", "a", "b", "c"])
    tgt = {"a": 170.0, "b": 290.0}
    lam = {"a": 0.5, "b": 1.0}
    out, rep = mc.shift(pred, key, tgt, lam, mode="add")
    assert out.tolist() == [110.0, 210.0, 290.0, 50.0]
    assert rep["a"]["delta"] == pytest.approx(10.0) and rep["b"]["gap"] == pytest.approx(-10.0)
    assert pred.tolist() == [100.0, 200.0, 300.0, 50.0]


def test_shift_multiplicative_hits_the_same_mean_as_additive_at_lambda_one():
    pred = np.array([100.0, 300.0])
    key = np.array(["a", "a"])
    out, _ = mc.shift(pred, key, {"a": 220.0}, {"a": 1.0}, mode="mul")
    assert out.tolist() == pytest.approx([110.0, 330.0]) and out.mean() == pytest.approx(220.0)
    with pytest.raises(ValueError, match="mode"):
        mc.shift(pred, key, {"a": 1.0}, {"a": 1.0}, mode="x")


def test_gain_and_day_block_ci():
    """G = (SSE_before - SSE_after)/denominator; the CI resamples whole days within month."""
    y = np.array([10.0, 10.0, 10.0, 10.0])
    before = np.array([0.0, 0.0, 0.0, 0.0])
    after = np.array([10.0, 10.0, 10.0, 10.0])
    day = np.array(["d1", "d1", "d2", "d2"])
    month = np.array([1, 1, 1, 1])
    g, lo, hi = mc.gain_ci(y, before, after, day, month, denom=4.0, n_boot=200, seed=1)
    assert g == pytest.approx(100.0) and lo == pytest.approx(100.0) and hi == pytest.approx(100.0)
    after2 = np.array([10.0, 10.0, 0.0, 0.0])
    g2, lo2, hi2 = mc.gain_ci(y, before, after2, day, month, denom=4.0, n_boot=2000, seed=1)
    assert g2 == pytest.approx(50.0) and lo2 == pytest.approx(0.0) and hi2 == pytest.approx(100.0)


def test_cell_sigma_inflates_when_the_cell_extrapolates_beyond_the_2025_regime_share():
    """Registered step 3: sigma *= share / max_share_2025 of the regime when the target share is larger."""
    lmo = pd.DataFrame({"inv_share": [0.10, 0.05, 0.001], "err": [3.0, -4.0, 1.0]})
    base = np.sqrt((9 + 16) / 2)
    assert mc.cell_sigma(lmo, 0.08) == pytest.approx(base)
    assert mc.cell_sigma(lmo, 0.30) == pytest.approx(base * 3.0)
    assert mc.cell_sigma(lmo, 0.0015) == pytest.approx(1.0 * 1.5)
    assert mc.cell_sigma(lmo, 0.0005) == pytest.approx(1.0)
    assert mc.cell_sigma(lmo.iloc[[2]], 0.05) == np.inf
