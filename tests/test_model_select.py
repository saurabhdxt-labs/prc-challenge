"""Tests for scripts/model_select.py (SEL). Synthetic and seeded."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import model_select as S  # noqa: E402


def test_simplex_ls_recovers_true_weights_and_respects_the_simplex():
    rng = np.random.default_rng(0)
    E = rng.normal(1000, 200, (4000, 3))
    y = E @ np.array([0.2, 0.5, 0.3])
    w = S.simplex_ls(E, y)
    np.testing.assert_allclose(w, [0.2, 0.5, 0.3], atol=1e-3)
    y2 = 2.0 * E[:, 0]                                  # unconstrained answer is w = (2, 0, 0): the simplex caps it
    w2 = S.simplex_ls(E, y2)
    assert (w2 >= 0).all() and w2.sum() == pytest.approx(1.0) and w2[0] == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="align"):
        S.simplex_ls(E, y[:-1])


def test_crossfit_uses_only_the_other_month():
    """Fails if a direction fits on its own month: July's predictions must not change when July's labels change."""
    month = np.array([1] * 5 + [7] * 5)
    y = np.arange(10, dtype=float)
    fp = lambda fit, app: np.full(int(app.sum()), y[fit].mean())
    out = S.crossfit(month, fp)
    assert out[:5].tolist() == [7.0] * 5 and out[5:].tolist() == [2.0] * 5
    with pytest.raises(ValueError, match="exactly two months"):
        S.crossfit(np.ones(4), fp)


def test_stack_by_group_learns_a_different_weight_per_airport():
    """Airport A is model 0's, airport B model 1's: a per-airport stack is ~exact, the global one is not."""
    rng = np.random.default_rng(1)
    n = 8000
    ap = np.where(np.arange(n) % 2 == 0, "A", "B")
    month = np.where(np.arange(n) < n // 2, 1, 7)
    E = rng.normal(1000, 200, (n, 2))
    y = np.where(ap == "A", E[:, 0], E[:, 1])
    per = S.stack_by_group(E, y, month, ap)
    glob_, _ = S.stack_global(E, y, month)
    assert np.sqrt(((per - y) ** 2).mean()) < 1.0 < np.sqrt(((glob_ - y) ** 2).mean())


def test_oracle_picks_the_closest_expert_per_row():
    E = np.array([[10.0, 20.0], [10.0, 20.0]])
    assert S.oracle(E, np.array([12.0, 19.0])).tolist() == [10.0, 20.0]


def test_the_learned_selectors_find_a_planted_rule():
    """Expert 0 is right where feature f > 0, expert 1 elsewhere: both LightGBM selectors beat the best single expert."""
    rng = np.random.default_rng(2)
    n = 6000
    f = rng.normal(size=n)
    truth = rng.normal(1000, 200, n)
    E = np.column_stack([np.where(f > 0, truth, truth + 300), np.where(f > 0, truth - 300, truth)])
    month = np.where(np.arange(n) < n // 2, 1, 7)
    day_codes = np.arange(n) % 20
    Xm = np.column_stack([E, f])
    best_single = min(np.sqrt(((E[:, k] - truth) ** 2).mean()) for k in range(2))
    for fn in (lambda: S.lgbm_stacker(Xm, truth, month, day_codes),
               lambda: S.lgbm_classifier_selector(Xm, E, truth, month, day_codes)):
        pred = fn()
        assert np.sqrt(((pred - truth) ** 2).mean()) < 0.5 * best_single
