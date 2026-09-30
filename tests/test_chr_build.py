"""Tests for scripts/analysis/chr_build.py (arm CHR).

CHR exists to repair three verified defects in the retracted FWD arm. Each repair is asserted here, because
a silent regression in any one of them reproduces exactly the flaw that made FWD's headline void.
"""
from __future__ import annotations
import importlib.util, pathlib, sys
import numpy as np, pandas as pd, pytest
ROOT = pathlib.Path(__file__).resolve().parents[1]
def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m
CHR = _load("chr_build", "scripts/analysis/chr_build.py")


def test_regression_holdout_months_excluded():
    assert CHR.HOLDOUT_MONTHS == (1, 7)
    src = (ROOT / "scripts/analysis/chr_build.py").read_text()
    assert 'assert not d.month.isin(HOLDOUT_MONTHS).any()' in src


def test_regression_equal_training_volume_is_asserted():
    """FWD defect 2: DAY had 1.63x FWD's training rows. CHR must enforce equality."""
    src = (ROOT / "scripts/analysis/chr_build.py").read_text()
    assert 'assert tr_chrono.sum() == tr_inter.sum() == N' in src


def test_regression_test_rows_never_enter_training():
    src = (ROOT / "scripts/analysis/chr_build.py").read_text()
    assert 'assert not (tr_chrono & is_test).any() and not (tr_inter & is_test).any()' in src
    assert 'assert not (es & is_test).any() and not (fit & is_test).any()' in src


def test_regression_days_are_numeric_not_lexical():
    """A doy kept as a string compares '99' > '302' and silently breaks the chronological split."""
    src = (ROOT / "scripts/analysis/chr_build.py").read_text()
    assert 'd["real_day"] = d.doy.astype(int)' in src
    assert 'int(x) < first_test_day' in src, "chronological cut must be a numeric comparison"


def test_encodings_are_fitted_on_fit_rows_only():
    """stand_ab.infold_encodings' own docstring names this call site as where the leak lives."""
    src = (ROOT / "scripts/analysis/chr_build.py").read_text()
    assert "S.infold_encodings(keys, y_all, delta_all, fit)" in src


def test_verdict_working_when_interleaved_overstates_beyond_the_bar():
    out = CHR.evaluate(chrono_rmse=212.0, inter_rmse=209.0, ci=[1.2, 4.9])
    assert out["verdict"] == "WORKING" and out["gap_s_chrono_minus_interleaved"] == pytest.approx(3.0)


def test_verdict_not_working_when_ci_includes_zero():
    assert CHR.evaluate(212.0, 209.0, [-0.5, 6.0])["verdict"] == "NOT WORKING"


def test_verdict_not_working_when_chronological_is_better():
    """A negative gap means interleaved does NOT overstate — the registered FALSE shape."""
    assert CHR.evaluate(209.0, 212.0, [-6.0, -1.0])["verdict"] == "NOT WORKING"


def test_ambiguous_band_under_the_bar():
    out = CHR.evaluate(210.5, 210.0, [0.1, 0.9])
    assert 0 < out["gap_s_chrono_minus_interleaved"] < CHR.GAP_BAR_S
    assert out["verdict"] == "NOT WORKING (ambiguous, per registration)"


def test_day_block_ci_detects_a_uniformly_worse_arm():
    """Errors identical on every day => zero resampling variance => a degenerate but correct interval."""
    days = np.repeat(np.arange(30), 50)
    y = np.full(len(days), 100.0)
    lo, hi = CHR.day_block_ci(y, days, y + 5.0, y + 1.0)
    assert lo == pytest.approx(4.0) and hi == pytest.approx(4.0)


def test_day_block_ci_widens_when_days_differ_and_covers_zero_for_equal_arms():
    """With per-DAY variation the interval must have real width, and two equally good arms must span zero."""
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(40), 60)
    y = rng.normal(100, 20, size=len(days))
    per_day = rng.normal(0, 8, size=40)[days]          # a different error pattern each day
    lo, hi = CHR.day_block_ci(y, days, y + per_day, y - per_day)
    assert hi > lo, "resampling whole days must produce a non-degenerate interval"
    assert lo < 0 < hi, "two symmetric arms must not be separated"
