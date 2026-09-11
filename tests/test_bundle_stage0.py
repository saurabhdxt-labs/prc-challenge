"""Tests for scripts/bundle_stage0.py (BND Stage 0) and nm_param_diag.holdout_days. Synthetic and seeded, except the
real-data date test (skipped when the challenge caches are absent)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bundle_stage0 as B  # noqa: E402
import nm_param_diag as N  # noqa: E402


def test_taxi_recovers_the_delta_scale_with_the_floor():
    """Fails if taxi() uses proxy + delta_hat or drops the floor (BC-2)."""
    assert B.taxi(np.array([900.0, 100.0]), np.array([100.0, 500.0])).tolist() == [800.0, 1.0]


def test_blend_is_the_convex_combination_and_refuses_bad_inputs():
    a, b = np.array([100.0, 300.0]), np.array([200.0, 100.0])
    np.testing.assert_allclose(B.blend(a, b), [150.0, 200.0], rtol=0, atol=1e-12)   # exact halves in float64
    np.testing.assert_allclose(B.blend(a, b, 0.25), [175.0, 150.0], rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="outside"):
        B.blend(a, b, 1.5)
    with pytest.raises(ValueError, match="shapes"):
        B.blend(a, b[:1])
    with pytest.raises(ValueError, match="floored"):
        B.blend(np.array([0.5, 2.0]), b)


def test_guard_passes_within_tolerance_and_names_the_arm_otherwise():
    assert B.guard("F", 222.5632, 222.56316)["want"] == 222.56316
    with pytest.raises(AssertionError, match="guard CAP"):
        B.guard("CAP", 222.0, 221.985)
    with pytest.raises(AssertionError, match="guard X"):
        B.guard("X", 1.0, float("nan"))


def test_best_weight_diagnostic_is_the_least_squares_weight_clipped():
    rng = np.random.default_rng(0)
    a, b = rng.normal(900, 100, 5000), rng.normal(900, 100, 5000)
    y = 0.3 * a + 0.7 * b
    assert B.best_weight_diagnostic(y, a, b) == pytest.approx(0.3, abs=1e-9)
    assert B.best_weight_diagnostic(a + 10 * (a - b), a, b) == 1.0        # would be > 1 -> clipped
    assert np.isnan(B.best_weight_diagnostic(y, a, a))


def test_cuts_price_the_fill_and_tail_rows_separately():
    y = np.array([100.0, 100.0, 100.0, 100.0])
    base = np.array([110.0, 110.0, 110.0, 110.0])
    arm = np.array([100.0, 100.0, 100.0, 110.0])          # fill rows gain 200, the one tail row 100: they differ
    fill = np.array([True, True, False, False])
    delta = np.array([0.0, 0.0, 700.0, 0.0])
    c = B.cuts(y, base, arm, np.array(["A", "A", "B", "B"]), np.array([1, 7, 1, 7]), fill, delta, n_fold=4)
    assert c["fill_board"] == pytest.approx(200.0 / 4 * N.W_MATCHED)
    assert c["tail_board"] == pytest.approx(100.0 / 4 * N.W_MATCHED)
    assert c["by_month_board"] == {1: pytest.approx(200.0 / 4 * N.W_MATCHED), 7: pytest.approx(100.0 / 4 * N.W_MATCHED)}
    assert c["gain_tail_s"] == pytest.approx(10.0)


REAL = (ROOT / "data" / "cache_stand" / "fold_preds_queue_order.parquet").exists()


@pytest.mark.skipif(not REAL, reason="challenge caches not present")
def test_holdout_days_dates_real_rows_and_refuses_misaligned_labels():
    """Fails if holdout_days uses the wrong month offset (every date shifts) or skips the label check."""
    rec = pd.read_parquet(ROOT / "data" / "cache_stand" / "fold_preds_queue_order.parquet", columns=["row", "month", "y"])
    sub = rec.iloc[np.r_[0:50, -50:0]]
    days = N.holdout_days(sub.row.to_numpy(), sub.month.to_numpy(), sub.y.to_numpy())
    assert all(str(d).startswith("2025-01-") for d in days[:50]) and all(str(d).startswith("2025-07-") for d in days[50:])
    bad_y = sub.y.to_numpy().copy()
    bad_y[3] += 1.0
    with pytest.raises(RuntimeError, match="do not align"):
        N.holdout_days(sub.row.to_numpy(), sub.month.to_numpy(), bad_y)
    with pytest.raises(ValueError, match="serves months"):
        N.holdout_days(sub.row.to_numpy(), np.full(len(sub), 3), sub.y.to_numpy())
