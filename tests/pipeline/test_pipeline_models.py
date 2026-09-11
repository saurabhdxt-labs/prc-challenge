"""prc.pipeline.models -- the plug-in interface: every registered model conforms to the protocol; the adapters
reproduce the shipped fit functions bit for bit; save/load round-trips are exact and refuse foreign or tampered
artefacts; the stubs refuse with their reason.

Mutation rehearsal (2026-09-11, scratchpad p2a/mutate.py): the break named in each docstring turned the test
RED against prc/pipeline/models.py, GREEN on restore.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pipeline_world", pathlib.Path(__file__).with_name("pipeline_world.py"))
pw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pw)

from prc.pipeline import config as C  # noqa: E402
from prc.pipeline import errors as E  # noqa: E402
from prc.pipeline import legacy  # noqa: E402
from prc.pipeline import models as M  # noqa: E402


# ---- conformance ------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(M.REGISTRY))
def test_every_registered_model_conforms_to_the_plugin_protocol(name):
    """Rehearsed: LightGBMModel.describe's 'wraps' key renamed -> RED for lightgbm; NonFillBodyModel's -> RED for
    both bodies; the stubs' describe claiming implemented=True -> RED for catboost / nn / blend."""
    m = M.make_model(name)
    assert isinstance(m, M.ModelPlugin)
    d = m.describe()
    assert set(M.DESCRIBE_KEYS) <= set(d) and d["name"] == name == m.name
    assert d["input_kind"] in ("matrix", "frame")
    assert d["implemented"] is (name in ("lightgbm", "nonfill_body", "nonfill_body_congestion"))


class _Named:
    name = "fake"


@pytest.mark.parametrize("pred,msg", [
    (None, "returned NoneType"),
    ([1.0, 2.0, 3.0], "returned list"),
    (np.ones((3, 1)), r"shape \(3, 1\)"),
    (np.ones(1), r"shape \(1,\)"),                          # would BROADCAST through raw_taxi_time silently
    (np.ones(3, dtype="float32"), "dtype float32"),
    (np.array([1.0, np.nan, np.inf]), "2 non-finite"),
])
def test_check_prediction_refuses_what_the_protocol_isinstance_lets_through(pred, msg):
    """isinstance(m, ModelPlugin) only checks method names (runtime_checkable), so each of these passes it.
    Rehearsed 2026-09-11 (c4) per case: that case's own check in check_prediction deleted -> RED (four mutations: type,
    shape, dtype, finiteness)."""
    with pytest.raises(E.LaneError, match=f"'fake'.*{msg}"):
        M.check_prediction(_Named(), pred, 3)
    ok = np.array([1.0, 2.0, 3.0])
    assert M.check_prediction(_Named(), ok, 3) is ok


def test_every_implemented_plugin_meets_the_predict_contract_on_a_real_fit(mw, sw):
    """Behavioural conformance, not attribute conformance: each implemented plug-in, fitted, returns a 1-D float64 vector
    with one finite value per row (check_prediction passes). Fails if an adapter's predict returns anything else."""
    Xr = mw.X[mw.is_rank]
    assert len(M.check_prediction(mw.model, mw.model.predict(mw.state, Xr), len(Xr))) == len(Xr)
    st = sw.st
    for congestion in (False, True):
        m = M.NonFillBodyModel(congestion=congestion)
        scored = st.uc.attach_witness(sw.scored, st.uc.airport_hour_witness(sw.serve_dep))
        state = m.fit(sw.train_unm, M.Labels(), M.Masks(train=~st.bs.schedule_fill(sw.train_unm)), (0, 1))
        assert len(M.check_prediction(m, m.predict(state, scored), len(scored))) == len(scored)
    implemented = {n for n in M.REGISTRY if M.make_model(n).describe()["implemented"]}
    assert implemented == {"lightgbm", "nonfill_body", "nonfill_body_congestion"}   # a new plug-in must join this test


def test_the_registry_and_the_config_know_the_same_model_names():
    """Rehearsed: 'blend' removed from REGISTRY -> RED."""
    assert set(M.REGISTRY) == set(C.MODEL_NAMES)
    with pytest.raises(E.UnknownModelError, match="xgboost"):
        M.make_model("xgboost")


@pytest.mark.parametrize("name", ["catboost", "nn", "blend"])
@pytest.mark.parametrize("method", ["fit", "predict", "save", "load"])
def test_a_stub_refuses_every_operation_naming_itself_and_why(name, method):
    """Built with a lane's config kwargs (as check_lane_inputs builds a matched lane's model) and still refusing.
    Rehearsed: _Stub._refuse returning None -> RED; _Stub without **config -> RED (TypeError, found 2026-09-11)."""
    m = M.make_model(name, target="delta")
    args = {"fit": (None, None, None, (0,)), "predict": (None, None), "save": (None, "x"), "load": ("x",)}[method]
    with pytest.raises(NotImplementedError, match=f"{name}.*{method}.*P2a"):
        getattr(m, method)(*args)


@pytest.mark.parametrize("kw,msg", [
    (dict(train=np.array([0, 1])), "boolean"),
    (dict(train=np.zeros(3, bool)), "selects no row"),
    (dict(train=np.ones(3, bool), fit=np.ones(3, bool)), "both fit and es"),
    (dict(train=np.ones(3, bool), fit=np.array([1, 1, 0], bool), es=np.array([0, 1, 1], bool)), "overlap"),
    (dict(train=np.ones(3, bool), fit=np.array([1, 0, 0], bool), es=np.array([0, 1, 0], bool)), "must equal train"),
])
def test_masks_refuse_an_inconsistent_split(kw, msg):
    """Rehearsed per case: that case's own check in Masks.__post_init__ deleted -> RED (five mutations)."""
    with pytest.raises(ValueError, match=msg):
        M.Masks(**kw)


# ---- the unmatched body -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sw():
    return pw.stratum_world(seed=7, n_per_month=300, n_matched_per_month=200)


@pytest.fixture(scope="module")
def sw_big():
    """> 10,000 non-fill rows: sklearn's early_stopping='auto' switches ON above 10,000 rows on a seed-dependent
    split, so only here do the seeds give different bodies (the real stratum has 20,823 non-fill rows). Below it
    every seed predicts the same and a seed-averaging bug is invisible -- the first rehearsal of the test below
    SURVIVED on a 3,300-row world for exactly that reason (2026-09-11)."""
    return pw.stratum_world(seed=7, n_per_month=1100, n_matched_per_month=100)


@pytest.mark.parametrize("congestion", [False, True])
def test_the_nonfill_body_equals_the_shipped_regressors_mean_bit_for_bit(sw_big, congestion):
    """The adapter must be the shipped body: fit_nf_regressor (S1) / fit_congestion_regressor (E3C) per seed on
    the non-fill rows, mean_over_seeds. Rehearsed: NonFillBodyModel.predict averaging only the first seed -> RED
    (SURVIVED on the first, too-small world; this world makes the seeds differ, asserted below)."""
    sw = sw_big
    st = sw.st
    train = sw.train_unm
    nonfill = train[~st.bs.schedule_fill(train)]
    assert len(nonfill) > 10_000
    scored = st.uc.attach_witness(sw.scored, st.uc.airport_hour_witness(sw.serve_dep)) if congestion else sw.scored
    fn = st.uc.fit_congestion_regressor if congestion else st.bs.fit_nf_regressor
    per_seed = {s: fn(nonfill, s).predict(scored) for s in (0, 1)}
    assert not np.array_equal(per_seed[0], per_seed[1])            # the precondition: the seeds matter here
    want = st.bs.mean_over_seeds(per_seed)
    m = M.NonFillBodyModel(congestion=congestion)
    state = m.fit(train, M.Labels(taxi=train.y.to_numpy()), M.Masks(train=~st.bs.schedule_fill(train)), (0, 1))
    assert np.array_equal(m.predict(state, scored), want)
    assert state.n_train == len(nonfill)


def test_the_nonfill_body_refuses_fill_rows(sw):
    """The shipped fit refuses fill rows (their label is sp); the adapter must not bypass that.
    Rehearsed: NonFillBodyModel.fit dropping BLOCK_TIME before fitting -> RED."""
    m = M.NonFillBodyModel()
    with pytest.raises(ValueError, match="NON-FILL"):
        m.fit(sw.train_unm, M.Labels(), M.Masks(train=np.ones(len(sw.train_unm), bool)), (0,))


@pytest.mark.parametrize("congestion", [False, True])
def test_nonfill_save_load_round_trip_is_exact_and_refuses_tampering(sw, tmp_path, congestion):
    """Rehearsed: load() skipping the sha256 comparison -> RED (the tampered file is accepted)."""
    st = sw.st
    m = M.NonFillBodyModel(congestion=congestion)
    other = M.NonFillBodyModel(congestion=not congestion)
    scored = st.uc.attach_witness(sw.scored, st.uc.airport_hour_witness(sw.serve_dep))
    state = m.fit(sw.train_unm, M.Labels(), M.Masks(train=~st.bs.schedule_fill(sw.train_unm)), (0, 1))
    paths = m.save(state, tmp_path / "body")
    back = m.load(tmp_path / "body")
    assert np.array_equal(m.predict(back, scored), m.predict(state, scored))
    assert type(back.regressors[0]) is type(state.regressors[0]) and back.seeds == (0, 1)
    assert {p.suffix for p in paths} == {".pkl", ".json"}
    with pytest.raises(ValueError, match=other.name):
        other.predict(state, scored)
    pkl = tmp_path / "body" / f"{m.name}.pkl"
    pkl.write_bytes(pkl.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        m.load(tmp_path / "body")


# ---- LightGBM -----------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mw(tmp_path_factory):
    w = pw.matched_world(tmp_path_factory.mktemp("models_matched"))
    m = M.LightGBMModel(params=w.params, nest=w.nest, patience=w.patience)
    w.model, w.state = m, m.fit(w.X, w.labels, w.masks, (0, 1))
    return w


def test_the_lightgbm_adapter_is_lgbm_submits_early_stop_refit_and_seed_mean(mw):
    """Rehearsed: LightGBMModel.fit refitting at best_iter instead of info['n_ref'] -> RED."""
    ls = legacy.matched().ls
    X, lb, mk = mw.X, mw.labels, mw.masks
    info = ls.early_stop(X, lb.delta, lb.taxi, lb.proxy, mk.train, mk.fit, mk.es, dict(mw.params, seed=0), mw.nest, mw.patience)
    preds = [ls.predict_delta(ls.refit(X, lb.delta, mk.train, dict(mw.params, seed=s), info["n_ref"]), X[mw.is_rank], mw.params)
             for s in (0, 1)]
    assert mw.state.info["n_ref"] == info["n_ref"] and mw.state.n_all == int(mk.train.sum())
    assert np.array_equal(mw.model.predict(mw.state, X[mw.is_rank]), ls.mean_delta(preds))


def test_lightgbm_save_then_lazy_load_predicts_bit_for_bit(mw, tmp_path):
    """The saved-booster path (what the matched lane ships from) equals the in-memory prediction.
    Rehearsed: save() writing params with seed 0 for every seed -> RED (load's provenance check refuses seed 1)."""
    paths = mw.model.save(mw.state, tmp_path / "b")
    assert [p.name for p in paths] == ["lgbm_seed0.txt", "lgbm_seed0.fit.json", "lgbm_seed1.txt", "lgbm_seed1.fit.json"]
    lazy = mw.model.load(tmp_path / "b", n_all=mw.state.n_all, n_features=mw.X.shape[1])
    assert lazy.lazy and lazy.seeds == (0, 1)
    Xr = mw.X[mw.is_rank]
    assert np.array_equal(mw.model.predict(lazy, Xr), mw.model.predict(mw.state, Xr))


@pytest.mark.parametrize("field,value,msg", [
    ("params", {"learning_rate": 0.5}, "params differ"),
    ("n_features", 7, "n_features"),
    ("n_all", 3, "n_all"),
    ("smoke", True, "smoke"),
    ("target", "y", "target"),
    ("all_rows", True, "all_rows"),
])
def test_lightgbm_load_refuses_a_booster_from_another_configuration(mw, tmp_path, field, value, msg):
    """Rehearsed: load()'s params comparison deleted -> RED for the params case (likewise per field)."""
    d = tmp_path / "b"
    mw.model.save(mw.state, d)
    j = d / "lgbm_seed1.fit.json"
    rec = json.loads(j.read_text())
    rec[field] = {**rec["params"], **value} if field == "params" else value
    j.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match=msg):
        mw.model.load(d, n_all=mw.state.n_all, n_features=mw.X.shape[1])


@pytest.mark.parametrize("field", ["target", "all_rows", "unmatched_weight"])
def test_lightgbm_load_refuses_a_booster_whose_provenance_omits_a_field(mw, tmp_path, field):
    """A missing field used to be ASSUMED (target 'delta', all_rows False, weight 1.0) -- review finding 2026-09-11.
    Rehearsed 2026-09-11 (c4): load()'s absent-field check deleted -> RED for all three."""
    d = tmp_path / "b"
    mw.model.save(mw.state, d)
    j = d / "lgbm_seed0.fit.json"
    rec = json.loads(j.read_text())
    del rec[field]
    j.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match=rf"provenance field\(s\) \['{field}'\] absent"):
        mw.model.load(d, n_all=mw.state.n_all, n_features=mw.X.shape[1])


def test_the_shipped_v6_boosters_carry_every_required_provenance_field():
    """The strict load must not refuse the boosters v10's matched lane ships from (skipped without the data)."""
    fits = sorted((ROOT / "data" / "cache_stand").glob("lgbm_v6_queue_order_seed*.fit.json"))
    if not fits:
        pytest.skip("v6 boosters not present")
    for f in fits:
        rec = json.loads(f.read_text())
        assert (rec["target"], rec["all_rows"], rec["unmatched_weight"]) == ("delta", False, 1.0), f.name


def test_lightgbm_predict_refuses_a_design_of_the_wrong_width(mw):
    """Rehearsed: predict's width check deleted -> RED (LightGBM would raise its own, different error)."""
    with pytest.raises(ValueError, match="features"):
        mw.model.predict(mw.state, mw.X[:5, :-1])
