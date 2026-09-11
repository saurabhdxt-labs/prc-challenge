"""Tests for scripts/regime_experts.py (arm REG). Synthetic and seeded; the gate test fits a tiny real LightGBM."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import regime_experts as R  # noqa: E402


def test_regime_label_precedence_and_boundaries():
    """fill first; then |delta| > 600 strictly. Fails on a moved boundary or fill losing precedence."""
    y = np.array([900.0, 900.0, 900.0, 900.0, 900.0, 900.0])
    sp = np.array([960.0, 5_000.0, 5_000.0, 5_000.0, 5_000.0, 900.0])
    dlt = np.array([-5_000.0, -600.0, -601.0, 600.0, 601.0, 5_000.0])
    assert R.regime_label(y, sp, dlt).tolist() == [1, 0, 2, 0, 3, 1]
    with pytest.raises(ValueError, match="finite"):
        R.regime_label(np.array([1.0]), np.array([np.nan]), np.array([0.0]))


def test_expert_taxi_floors_and_counts_binds():
    mu, b = R.expert_taxi(np.array([900.0, 500.0]), np.array([100.0, 700.0]))
    assert mu.tolist() == [800.0, 1.0] and b == 1
    with pytest.raises(ValueError, match="non-finite"):
        R.expert_taxi(np.array([1.0]), np.array([np.inf]))


def test_assemble_mixture_is_the_probability_weighted_mean_and_refuses_bad_gates():
    P = np.array([[0.5, 0.5, 0.0, 0.0], [0.1, 0.0, 0.9, 0.0]])
    MU = np.array([[100.0, 300.0, 0.0, 0.0], [1000.0, 1.0, 2000.0, 5.0]])
    np.testing.assert_allclose(R.assemble_mixture(P, MU), [200.0, 1900.0], rtol=0, atol=1e-9)
    with pytest.raises(ValueError, match="sum to 1"):
        R.assemble_mixture(np.array([[0.5, 0.2, 0.0, 0.0]]), np.ones((1, 4)))
    with pytest.raises(ValueError, match="non-negative"):
        R.assemble_mixture(np.array([[1.2, -0.2, 0.0, 0.0]]), np.ones((1, 4)))
    with pytest.raises(ValueError, match=r"\(n, 4\)"):
        R.assemble_mixture(np.ones((2, 3)) / 3, np.ones((2, 3)))
    with pytest.raises(ValueError, match="non-finite"):
        R.assemble_mixture(np.array([[1.0, 0, 0, 0]]), np.array([[np.nan, 0, 0, 0]]))
    assert R.assemble_mixture(np.array([[1.0, 0, 0, 0]]), np.array([[-50.0, 0, 0, 0]])).tolist() == [1.0]


def test_one_vs_rest_auc_and_reliability():
    lab = np.array([0, 0, 2, 2])
    P = np.array([[0.9, 0.0, 0.1, 0.0], [0.8, 0.0, 0.2, 0.0], [0.2, 0.0, 0.8, 0.0], [0.4, 0.0, 0.6, 0.0]])
    auc = R.one_vs_rest_auc(P, lab)
    assert auc["agree"] == 1.0 and auc["early"] == 1.0 and np.isnan(auc["fill"])
    rel = R.reliability(np.linspace(0, 1, 100), np.linspace(0, 1, 100) > 0.5, bins=4)
    assert sum(r["n"] for r in rel) == 100 and rel[0]["rate"] == 0.0 and rel[-1]["rate"] == 1.0


def _score_world(reg_shift_agree=0.0):
    rng = np.random.default_rng(0)
    n = 4000
    day = np.array([f"2025-01-{1 + i % 20:02d}" if i < n // 2 else f"2025-07-{1 + i % 20:02d}" for i in range(n)])
    month = np.where(np.arange(n) < n // 2, 1, 7)
    ap = np.array([f"A{i % 10}" for i in range(n)])
    regime = np.where(rng.random(n) < 0.1, 2, 0)
    y = 900 + np.where(regime == 2, 400.0, 0.0) + rng.normal(0, 50, n)
    base = np.full(n, 900.0)
    reg = base + np.where(regime == 2, 400.0, reg_shift_agree)
    return y, base, reg, regime, ap, month, day


def test_score_shortlists_a_real_tail_gain_and_refuses_one_the_body_pays_for():
    """Fails if C3 is not wired (the body-pays arm would SHORTLIST) or the verdict ignores a clause."""
    y, base, reg, regime, ap, month, day = _score_world()
    s = R.score(y, base, reg, regime, ap, month, day, n_fold=len(y) // 20)
    assert s["verdict"] == "SHORTLIST" and s["by_regime"]["early"] > 0 and s["airports_improving"] == 10
    y, base, reg, regime, ap, month, day = _score_world(reg_shift_agree=120.0)   # the agree rows pay heavily
    s2 = R.score(y, base, reg, regime, ap, month, day, n_fold=len(y) // 20)
    assert s2["clauses"]["C3_agree_net_ge_minus_300"] is False and s2["verdict"] == "NOT SHORTLISTED"


def test_fit_gate_learns_a_planted_regime_and_not_a_null():
    """A tiny real LightGBM: the early regime is a function of feature 0 -> AUC high; permuted labels -> AUC ~ 0.5."""
    rng = np.random.default_rng(1)
    n = 6000
    X = rng.normal(size=(n, 5)).astype("float32")
    lab = np.where(X[:, 0] > 1.0, 2, np.where(X[:, 1] > 1.2, 3, np.where(X[:, 2] > 1.5, 1, 0))).astype("int64")
    idx = np.arange(n)
    te, fit, es = idx >= 5000, idx < 4000, (idx >= 4000) & (idx < 5000)
    tr = fit | es
    params = dict(objective="regression", learning_rate=0.1, num_leaves=15, min_data_in_leaf=20, verbosity=-1,
                  num_threads=1, seed=0)
    P, info = R.fit_gate(X, lab, fit, es, tr, te, params, nest=200, patience=20)
    assert P.shape == (1000, 4) and np.allclose(P.sum(axis=1), 1.0) and info["n_ref"] >= info["best_iter"]
    assert R.one_vs_rest_auc(P, lab[te])["early"] > 0.95
    lab_null = rng.permutation(lab)
    Pn, _ = R.fit_gate(X, lab_null, fit, es, tr, te, params, nest=200, patience=20)
    assert abs(R.one_vs_rest_auc(Pn, lab_null[te])["early"] - 0.5) < 0.1


def test_score_c4_needs_seven_airports():
    """Six improving airports fail C4; seven pass. Fails if the threshold moves."""
    y, base, reg, regime, ap, month, day = _score_world()
    for n_bad, want in ((4, False), (3, True)):
        bad = np.isin(ap, [f"A{i}" for i in range(n_bad)])
        reg2 = np.where(bad, base - 300.0, reg)              # those airports get worse
        s = R.score(y, base, reg2, regime, ap, month, day, n_fold=len(y) // 20)
        assert s["airports_improving"] == 10 - n_bad and s["clauses"]["C4_airports_ge_7"] is want


def test_fit_gate_refits_on_all_training_rows():
    """Class 3 exists ONLY in the early-stopping rows: a refit on `tr` learns it, a refit on `fit` alone cannot.
    Fails if fit_gate's refit uses the fit rows."""
    rng = np.random.default_rng(2)
    n = 6000
    X = rng.normal(size=(n, 4)).astype("float32")
    idx = np.arange(n)
    te, fit, es = idx >= 5000, idx < 3000, (idx >= 3000) & (idx < 5000)
    tr = fit | es
    lab = np.where(X[:, 0] > 1.0, 2, 0).astype("int64")
    late = X[:, 1] > 0.8
    lab[late & (es | te)] = 3                                  # never in the fit rows
    params = dict(objective="regression", learning_rate=0.1, num_leaves=15, min_data_in_leaf=20, verbosity=-1,
                  num_threads=1, seed=0)
    P, _ = R.fit_gate(X, lab, fit, es, tr, te, params, nest=200, patience=20)
    assert R.one_vs_rest_auc(P, lab[te])["late"] > 0.9
