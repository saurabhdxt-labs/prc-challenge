"""prc.pipeline.evaluate — the fold-mode harness (acceptance M2). Synthetic caches from tests/synthetic_caches.py (the
generator lgbm_fold's own tests use); tiny LightGBM fits at one thread (bit-deterministic)."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("synthetic_caches", ROOT / "tests" / "synthetic_caches.py")
SYN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SYN)

from prc.pipeline import errors as E  # noqa: E402
from prc.pipeline import evaluate as EV  # noqa: E402
from prc.pipeline import legacy  # noqa: E402
from prc.pipeline import scoring as SC  # noqa: E402

PARAMS = {"learning_rate": 0.1, "num_threads": 1}
CAND = dict(tag="TEST", prereg="plans/PREREG_test.md", kind="single", features="queue_order", seeds=[0, 1],
            params=PARAMS, nest=40, patience=10)


@pytest.fixture(scope="module")
def fold(tmp_path_factory):
    root = tmp_path_factory.mktemp("evalfold") / "data"
    stand, queue = SYN.synthetic_caches(root, (1, 2, 3), n=600, seed=4)
    return EV.load_matched_fold(stand, queue, root / "cache_order", (1, 2, 3), "queue_order")


def test_predict_single_equals_lgbm_folds_fit_arm_bit_for_bit(fold):
    """M2's unit form: the pipeline's single-model fold path == lgbm_fold.fit_arm (the harness every recorded arm used),
    same fold, params, seeds, one thread. Fails if the pipeline's early stop / refit / seed mean drifts from the shipped
    fold code (e.g. a different n_ref rule or seed handling)."""
    lf = legacy.fold().lf
    cand = EV.parse_candidate(CAND)
    got = EV.predict_single(fold, cand)
    ref = lf.fit_arm(fold, fold["X"], cand.lgbm_params(), cand.nest, cand.patience, 20, cand.seeds, False, "es", "ref")
    assert got["info"]["n_ref"] == ref["n_ref"] and got["info"]["best_iter"] == ref["best_iter"]
    for s in cand.seeds:
        assert np.array_equal(got["per_seed"][s], np.asarray(ref["single"][s], dtype="float64")), s
    assert np.array_equal(got["pooled"], np.asarray(ref["pooled"], dtype="float64"))
    assert not np.array_equal(got["per_seed"][0], got["per_seed"][1])            # the seeds matter in this world


def test_evaluate_writes_a_stamped_record_with_the_candidate_and_the_code_and_never_overwrites(fold, tmp_path):
    cand = EV.parse_candidate(CAND)
    p = EV.evaluate(fold, cand, tmp_path)
    frame, meta = SC.read_record(p, "delta", tag="TEST")
    assert list(frame.columns[:7]) == ["row", "month", "ap", "y", "proxy", "delta", "sp"]
    assert {"delta_hat_seed0", "delta_hat_seed1", "delta_hat"} <= set(frame.columns) and len(frame) == int(fold["te"].sum())
    assert set(frame.month) == {1} and meta["prereg"] == "plans/PREREG_test.md"
    m = meta["meta"]
    assert m["candidate"]["seeds"] == [0, 1] and "prc/pipeline/evaluate.py" in m["code"]["files"]
    assert m["fold"]["n_holdout"] == len(frame) and m["info"]["n_ref"] >= m["info"]["best_iter"]
    with pytest.raises(E.ValidationError, match="overwrite"):
        EV.evaluate(fold, cand, tmp_path)


def test_a_mixture_candidate_is_refused_by_name_when_a_regime_is_too_small(fold, tmp_path):
    """The small synthetic world has a handful of early-off-block rows: the harness refuses by name, not through a
    downstream guard. Fails if MixtureModel's minimum-rows check is removed."""
    cand = EV.parse_candidate(dict(CAND, tag="TESTMIX0", kind="mixture", seeds=[0]))
    with pytest.raises(E.LaneError, match="regime 'early'.*below the minimum"):
        EV.evaluate(fold, cand, tmp_path)


def _with_tails(fold, seed=7):
    """A copy of the fold with 15% early and 10% late rows planted consistently (y = proxy − delta) and made visible in
    feature 0, so every regime expert has rows to learn from."""
    rng = np.random.default_rng(seed)
    f = dict(fold)
    n = len(f["y"])
    kind = rng.random(n)
    dlt = f["dlt"].copy()
    z = (f["X"][:, 1] - np.nanmean(f["X"][:, 1])) / (np.nanstd(f["X"][:, 1]) + 1e-9)    # within-regime structure
    z = np.nan_to_num(z)
    early, late = kind < 0.15, (kind >= 0.15) & (kind < 0.25)
    dlt[early] = -1_000.0 + 250.0 * z[early] + rng.normal(0, 20, int(early.sum()))
    dlt[late] = 900.0 + 250.0 * z[late] + rng.normal(0, 20, int(late.sum()))
    f["dlt"], f["y"] = dlt, np.maximum(f["proxy"] - dlt, 1.0)
    X = f["X"].copy()
    X[:, 0] = np.select([early, late], [5.0, -5.0], default=0.0) + rng.normal(0, 0.3, n)
    f["X"] = X
    base = f["base"].copy()
    base["y"], base["delta"] = f["y"][f["te"]], dlt[f["te"]]
    f["base"] = base
    return f


def test_a_mixture_candidate_runs_through_the_same_harness(fold, tmp_path):
    """The multi-model lane in fold mode: gate probabilities sum to 1, taxi predictions finite and floored, stamped
    taxi_time, per-regime training counts recorded. (Its value is measured by registered arms, never here.)"""
    cand = EV.parse_candidate(dict(CAND, tag="TESTMIX", kind="mixture", seeds=[0], min_expert_rows=40,   # small world:
                                   params=dict(PARAMS, min_data_in_leaf=5)))                        # ~70-row experts
    p = EV.evaluate(_with_tails(fold), cand, tmp_path)
    frame, meta = SC.read_record(p, "taxi_time", tag="TESTMIX")
    P = frame[[f"p_{r}" for r in ("agree", "fill", "early", "late")]].to_numpy()
    assert np.allclose(P.sum(axis=1), 1.0) and (frame.taxi_hat >= 1.0).all() and np.isfinite(frame.taxi_hat).all()
    assert set(meta["meta"]["info"]["n_train_by_regime"]) == {"agree", "fill", "early", "late"}
    early = frame.delta.to_numpy() < -600
    assert frame.p_early.to_numpy()[early].mean() > 0.5                     # the planted regime is found


@pytest.mark.parametrize("patch,err,msg", [
    ({"colour": "red"}, E.UnknownKeyError, "colour"),
    ({"prereg": ""}, E.ConfigValueError, "prereg"),
    ({"kind": "forest"}, E.ConfigValueError, "kind"),
    ({"features": "everything"}, E.ConfigValueError, "features"),
    ({"params": {"max_depth": 3}}, E.UnknownKeyError, "max_depth"),
    ({"seeds": [0, 0]}, E.ConfigValueError, "distinct"),
    ({"nest": 0}, E.ConfigValueError, "positive"),
    ({"min_expert_rows": 0}, E.ConfigValueError, "positive"),
])
def test_parse_candidate_refuses_by_name(patch, err, msg):
    with pytest.raises(err, match=msg):
        EV.parse_candidate(dict(CAND, **patch))
    with pytest.raises(E.ConfigTypeError, match="mapping"):
        EV.parse_candidate(["not", "a", "mapping"])


def test_holdout_frame_refuses_a_misaligned_arm(fold):
    with pytest.raises(E.LaneError, match="holdout rows"):
        EV.holdout_frame(fold, {"bad": np.zeros(3)})
