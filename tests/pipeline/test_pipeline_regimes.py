"""prc.pipeline.regimes — the regime rule, the gate boundary, and the mixture's routing of rows to experts.
Synthetic and seeded; the gate and the composition test fit tiny real LightGBM models (< 2 s)."""
from __future__ import annotations

import numpy as np
import pytest

from prc.pipeline import errors as E
from prc.pipeline import models as M
from prc.pipeline import regimes as R


def test_label_matched_precedence_boundaries_and_refusal():
    y = np.array([900.0, 900.0, 900.0, 900.0, 900.0, 900.0])
    sp = np.array([960.0, 5_000.0, 5_000.0, 5_000.0, 5_000.0, 900.0])
    dlt = np.array([-5_000.0, -600.0, -601.0, 600.0, 601.0, 5_000.0])
    assert R.label_matched(y, sp, dlt).tolist() == [1, 0, 2, 0, 3, 1]
    with pytest.raises(E.LabelLeakError, match="training rows"):
        R.label_matched(np.array([np.nan]), np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="differ"):
        R.label_matched(y, sp[:-1], dlt)


class _Named:
    name = "g"


@pytest.mark.parametrize("P,msg", [(np.ones((2, 3)) / 3, "shape"), (np.array([[0.5, 0.5, 0, np.nan]] * 2), "finite"),
                                   (np.array([[0.6, 0.6, -0.2, 0.0]] * 2), "non-negative"),
                                   (np.array([[0.5, 0.2, 0.1, 0.1]] * 2), "sum to 1")])
def test_check_probabilities_refusals(P, msg):
    with pytest.raises(E.LaneError, match=msg):
        R.check_probabilities(_Named(), P, 2, 4)
    ok = np.array([[0.25] * 4] * 2)
    assert R.check_probabilities(_Named(), ok, 2, 4) is ok


def test_design_refuses_misaligned_parts():
    with pytest.raises(ValueError, match="must align"):
        R.Design(np.zeros((3, 2)), np.zeros(3), np.zeros(2))


class _FixedGate:
    """A gate that returns fixed probabilities (the mixture logic in isolation)."""
    name = "fixed_gate"
    input_kind = "matrix"

    def __init__(self, P):
        self.P = P

    def fit(self, X, y, masks, seeds):
        return M.check_prediction  # any sentinel state

    def predict(self, state, X):
        return self.P[: len(X)]

    def describe(self):
        return {"name": self.name}


class _RecordingExpert:
    """Records the training mask it was given; predicts the mean delta of its own training rows."""
    input_kind = "matrix"

    def __init__(self, name):
        self.name, self.seen = name, None

    def fit(self, X, y, masks, seeds):
        self.seen = masks
        return float(np.mean(np.asarray(y.delta)[masks.train]))

    def predict(self, state, X):
        return np.full(len(X), state, dtype="float64")


def _world(n=400, seed=0):
    rng = np.random.default_rng(seed)
    regime = rng.integers(0, 4, n)
    x1 = rng.normal(size=n)                                  # within-regime structure an expert can learn
    delta = np.select([regime == 2, regime == 3], [-1_000.0, 800.0], default=10.0) + 150.0 * x1 + rng.normal(0, 5, n)
    proxy = rng.uniform(1_200, 2_000, n)
    sp = proxy + 300.0
    y = np.where(regime == 1, sp + 20.0, proxy - delta)
    X = np.column_stack([regime + rng.normal(0, 0.05, n), x1]).astype("float32")
    idx = np.arange(n)
    masks = M.Masks(train=idx < 300, fit=idx < 200, es=(idx >= 200) & (idx < 300))
    labels = M.Labels(taxi=y, delta=delta, proxy=proxy, regime=np.where(idx < 300, regime, -1))
    return X, proxy, sp, y, delta, regime, masks, labels


def test_each_expert_sees_only_its_regimes_training_rows_and_the_mixture_is_the_weighted_mean():
    """Fails if an expert is fitted on all rows (RESULT 8's double counting) or the mixture weights the wrong expert."""
    X, proxy, sp, y, delta, regime, masks, labels = _world()
    P = np.tile([0.1, 0.2, 0.3, 0.4], (len(X), 1))
    experts = {"agree": (_RecordingExpert("a"), "delta"), "fill": (None, "sp_copy"),
               "early": (_RecordingExpert("e"), "delta"), "late": (_RecordingExpert("l"), "delta")}
    mix = R.MixtureModel(_FixedGate(P), experts, min_expert_rows=10)      # ~50 fit rows per regime here
    st = mix.fit(R.Design(X, proxy, sp), labels, masks, (0,))
    for name, k in (("agree", 0), ("early", 2), ("late", 3)):
        seen = experts[name][0].seen
        want = masks.train & (regime == k)
        assert np.array_equal(seen.train, want) and np.array_equal(seen.fit, masks.fit & (regime == k))
        assert st.info["n_train_by_regime"][name] == int(want.sum())
    D = R.Design(X[300:], proxy[300:], sp[300:])
    MU, _ = mix.expert_values(st, D)
    c_fill = float(np.mean((y - sp)[masks.train & (regime == 1)]))
    np.testing.assert_allclose(MU[:, 1], np.maximum(sp[300:] + c_fill, 1.0), rtol=0, atol=1e-9)
    np.testing.assert_allclose(mix.predict(st, D), np.maximum((P[:100] * MU).sum(axis=1), 1.0), rtol=0, atol=1e-9)
    assert st.experts[2][0] == pytest.approx(-1_000.0, abs=25.0)   # the early expert's mean delta, not the pooled one


def test_mixture_refuses_an_incomplete_expert_map_or_an_unknown_scale():
    with pytest.raises(ValueError, match="missing \\['late'\\]"):
        R.MixtureModel(_FixedGate(None), {"agree": (None, "delta"), "fill": (None, "sp_copy"), "early": (None, "delta")})
    with pytest.raises(ValueError, match="scales"):
        R.MixtureModel(_FixedGate(None), {r: (None, "seconds") for r in R.MATCHED})


def test_gate_and_mixture_compose_with_real_lightgbm_experts():
    """End to end on a world whose regime is visible in feature 0: the gate separates it (AUC-free check: argmax
    accuracy), the delta experts recover their regime's level, and every prediction passes check_prediction."""
    X, proxy, sp, y, delta, regime, masks, labels = _world(n=3_000, seed=1)
    idx = np.arange(3_000)
    masks = M.Masks(train=idx < 2_400, fit=idx < 1_800, es=(idx >= 1_800) & (idx < 2_400))
    labels = M.Labels(taxi=y, delta=delta, proxy=proxy, regime=np.where(idx < 2_400, regime, -1))
    params = dict(objective="regression", metric="rmse", learning_rate=0.1, num_leaves=15, min_data_in_leaf=20,
                  feature_fraction=1.0, bagging_fraction=1.0, bagging_freq=0, verbosity=-1, num_threads=1)
    gate = R.GateModel(params=params, nest=200, patience=20)
    expert = lambda: (M.LightGBMModel(params=params, nest=200, patience=20), "delta")
    mix = R.MixtureModel(gate, {"agree": expert(), "fill": (None, "sp_copy"), "early": expert(), "late": expert()})
    st = mix.fit(R.Design(X, proxy, sp), labels, masks, (0,))
    D = R.Design(X[2_400:], proxy[2_400:], sp[2_400:])
    P = gate.predict(st.gate, D.X)
    assert (P.argmax(axis=1) == regime[2_400:]).mean() > 0.95
    pred = mix.predict(st, D)
    rmse = float(np.sqrt(((pred - y[2_400:]) ** 2).mean()))
    assert pred.dtype == np.float64 and rmse < 60.0, rmse


def test_the_gate_refits_on_every_training_row():
    """The late regime appears in the FIT rows and the holdout, never in the stopping rows: a refit on masks.train
    learns it; a refit on the stopping rows alone could not. Fails if GateModel's refit uses masks.es (or masks.fit
    only, which the early-stopping-only class in test_fit_gate of the REG prototype covers)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    n = 5_000
    X = rng.normal(size=(n, 3)).astype("float32")
    idx = np.arange(n)
    masks = M.Masks(train=idx < 4_000, fit=idx < 3_000, es=(idx >= 3_000) & (idx < 4_000))
    regime = np.where(X[:, 0] > 1.0, 2, 0).astype("int64")
    late = (X[:, 1] > 0.8) & ~masks.es                          # never among the stopping rows
    regime[late] = 3
    params = dict(objective="regression", learning_rate=0.1, num_leaves=15, min_data_in_leaf=20, verbosity=-1,
                  num_threads=1)
    g = R.GateModel(params=params, nest=200, patience=20)
    st = g.fit(X, M.Labels(regime=np.where(masks.train, regime, -1)), masks, (0,))
    P = g.predict(st, X[4_000:])
    assert roc_auc_score(regime[4_000:] == 3, P[:, 3]) > 0.9


def test_mixture_refuses_an_expert_regime_below_the_minimum_rows():
    """Fails if MixtureModel.fit fits an expert on fewer rows than min_expert_rows (the refusal must name the regime)."""
    X, proxy, sp, y, delta, regime, masks, labels = _world()
    experts = {"agree": (_RecordingExpert("a"), "delta"), "fill": (None, "sp_copy"),
               "early": (_RecordingExpert("e"), "delta"), "late": (_RecordingExpert("l"), "delta")}
    mix = R.MixtureModel(_FixedGate(np.tile([0.25] * 4, (len(X), 1))), experts, min_expert_rows=500)
    with pytest.raises(E.LaneError, match="regime 'agree'.*below the minimum"):
        mix.fit(R.Design(X, proxy, sp), labels, masks, (0,))
    with pytest.raises(ValueError, match="min_expert_rows"):
        R.MixtureModel(_FixedGate(None), experts, min_expert_rows=0)
