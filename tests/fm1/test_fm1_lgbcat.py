"""The ADS-B stack's base F as a LightGBM + CatBoost blend (`adsb.stack.f_model: lgbcat`): the v46 composition
(plans/PREREG_blf_blend_serve_2026_09_28.md; owner decision 2026-09-29: port v46's blend into FM1).

v46 served, on the ADS-B-stage rows, rint(max(proxy - (F + resid), 1)) with F = 0.5 F_lightgbm + 0.5 F_catboost
(r13_cat.assemble's `pred_lgbcat`) and resid from the stack trained on that blend (r13_stack_full/models_lgbcat, on the
DWG tables 51348d3e1cb3). FM1 uses its own matched CatBoost (the model its half-blend already fits) as F_catboost.

Tiers: the formula and the knob contract (fast); v46's stored components (slow: three 340k-row parquet reads); the e2e
world through api.fit / predict (slow); the REAL stack stage on the DWG tables against v46's stored residual (slow, child).
Mutation rehearsal per test is recorded in its docstring and in tests/fm1/MUTATION_FM1.md (M-LGBCAT rows).
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import components as C
from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARMS = ROOT / "data/r12/r13_arms_refit/predictions_2026.parquet"
SERVE = ROOT / "data/r12/serve_blf_lgbcat"
LGBCAT_MODELS = "data/r12/r13_stack_full/models_lgbcat"
DWG_TABLES = "data/r12/tables_51348d3e1cb3"
QUIET = lambda *_a, **_k: None          # noqa: E731


# ---------------------------------------------------------------------------------------------- formula + knobs

def test_stack_base_is_the_r13_blend_formula():
    """lightgbm -> F; lgbcat -> 0.5 F + 0.5 C (r13_cat.assemble's pred_lgbcat). Mutation: weight 0.6 -> red."""
    r = np.random.default_rng(0)
    F, Cd = r.normal(0, 300, 200), r.normal(0, 300, 200)
    assert np.array_equal(C.stack_base(F, Cd, "lightgbm"), F)
    assert np.array_equal(C.stack_base(F, Cd, "lgbcat"), 0.5 * F + 0.5 * Cd)
    assert np.array_equal(C.stack_base(F, None, "lightgbm"), F)
    with pytest.raises(E.ConfigValueError, match="adsb.stack.f_model"):
        C.stack_base(F, Cd, "blendy")
    with pytest.raises(E.PipelineError, match="CatBoost"):
        C.stack_base(F, None, "lgbcat")


def test_stack_base_knob_default_and_contract():
    """Default lightgbm (the v36 stack). lgbcat needs the LightGBM base learner and the stack trained on the blend
    (r12_stack writes models_<arm>: models_lgbcat); the r12 stack refuses the blend and vice versa.
    Mutation: drop the models_<arm> check -> the mismatched configs parse -> red."""
    assert K.parse_config({})["adsb.stack.f_model"] == "lightgbm"
    ok = K.parse_config({"adsb": {"stack": {"f_model": "lgbcat", "models_dir": LGBCAT_MODELS}}})
    assert ok["adsb.stack.f_model"] == "lgbcat"
    with pytest.raises(E.ConfigValueError, match="adsb.stack.f_model lgbcat needs a stack trained on the blend"):
        K.parse_config({"adsb": {"stack": {"f_model": "lgbcat"}}})               # default models_dir is models_r12
    with pytest.raises(E.ConfigValueError, match="adsb.stack.f_model lightgbm cannot serve"):
        K.parse_config({"adsb": {"stack": {"models_dir": LGBCAT_MODELS}}})
    with pytest.raises(E.ConfigValueError, match="adsb.stack.f_model lgbcat needs matched.learner lightgbm"):
        K.parse_config({"matched": {"learner": "catboost", "blend": {"catboost_weight": 0.0}},
                        "adsb": {"stack": {"f_model": "lgbcat", "models_dir": LGBCAT_MODELS}}})
    with pytest.raises(E.ConfigError, match="adsb.stack.f_model"):
        K.parse_config({"adsb": {"stack": {"f_model": "cat"}}})


# ---------------------------------------------------------------------------------------------- v46's stored parts

@pytest.mark.slow
def test_stack_base_reproduces_v46s_blend_and_served_values_from_its_stored_components():
    """On v46's own stored parts: stack_base(pred_r12, pred_cat, lgbcat) == pred_lgbcat on all 339,551 rows, and
    staged_value(proxy, that F, resid) == v46's served lgbcat taxi on every ADS-B-stage row (224,371 N_v4 rows).
    Mutation: stack_base uses (F + C) / 2 with C first -> identical (commutative, not a mutant); weight 0.5 -> 0.51
    -> red on both."""
    p = pd.read_parquet(ARMS).set_index("MVT_ID_mvt")
    F = C.stack_base(p.pred_r12.to_numpy(float), p.pred_cat.to_numpy(float), "lgbcat")
    assert len(p) == 339_551 and np.array_equal(F, p.pred_lgbcat.to_numpy(float))
    d = pd.read_parquet(SERVE / "matched_2026_lgbcat_detail.parquet").set_index("MVT_ID_mvt")
    taxi = pd.read_parquet(SERVE / "matched_2026_lgbcat.parquet").set_index("MVT_ID_mvt").taxi
    n = d.index[d.lane == "N_v4"]
    Fn = pd.Series(F, index=p.index).reindex(n).to_numpy(float)
    got = C.staged_value(p.proxy.reindex(n).to_numpy(float), Fn, d.resid.reindex(n).to_numpy(float))
    assert len(n) == 224_371
    assert np.array_equal(got, taxi.reindex(n).to_numpy(float))


# ---------------------------------------------------------------------------------------------- the e2e world

@pytest.fixture(scope="module")
def lgbcat_world(e2e_world, fm1_world_mod):
    from prc import api
    world = e2e_world[0]
    scored = pd.read_parquet(world / "raw" / "scored.parquet")
    out = {}
    for base, models in (("lightgbm", None), ("lgbcat", str(ROOT / LGBCAT_MODELS))):
        extra = {"adsb.stack.f_model": base}
        if models:
            extra["adsb.stack.models_dir"] = models
        cfg = K.parse_config(fm1_world_mod.config_raw(world, **extra))
        art = api.fit(cfg, save=False, log=QUIET)
        out[base] = (art, api.predict(art, scored, log=QUIET).set_index("MVT_ID_mvt"))
    return out


@pytest.mark.slow
def test_world_lgbcat_serves_the_blend_on_stack_rows_only(lgbcat_world):
    """Stack rows: rint(max(proxy - (0.5 F + 0.5 C + resid), 1)) from the provenance columns; every other row equals the
    lightgbm-base run's value (the half-blend and the unmatched lane are untouched). Mutation: predict passes F (not
    the blend) to the stack -> the formula check fails -> red."""
    _, lgb = lgbcat_world["lightgbm"]
    _, cat = lgbcat_world["lgbcat"]
    st = cat.index[cat.stack_eligible.astype(bool)]
    assert len(st) > 0
    F, Cd = cat.F[st].to_numpy(float), cat.catboost_C[st].to_numpy(float)
    assert np.array_equal(cat.stack_F[st].to_numpy(float), 0.5 * F + 0.5 * Cd)
    want = C.staged_value(cat.proxy[st].to_numpy(float), cat.stack_F[st].to_numpy(float), cat.stack_resid[st].to_numpy(float))
    assert np.array_equal(cat.TAXITIME_SEC_mvt[st].to_numpy(float), want)
    assert np.array_equal(lgb.stack_F[st].to_numpy(float), lgb.F[st].to_numpy(float))    # the default base is F itself
    other = cat.index.difference(st)
    assert (cat.TAXITIME_SEC_mvt[other] == lgb.TAXITIME_SEC_mvt[other]).all()
    assert not np.array_equal(cat.TAXITIME_SEC_mvt[st].to_numpy(), lgb.TAXITIME_SEC_mvt[st].to_numpy())


@pytest.mark.slow
def test_world_lgbcat_fits_the_catboost_even_without_the_half_blend(e2e_world, fm1_world_mod):
    """The blend needs the matched CatBoost; with the half-blend off (weight 0) fit still fits it. Mutation: fit the
    CatBoost only for cbv_blend -> art.catboost is None -> red."""
    from prc import api
    world = e2e_world[0]
    cfg = K.parse_config(fm1_world_mod.config_raw(world, **{"adsb.stack.f_model": "lgbcat",
                                                            "adsb.stack.models_dir": str(ROOT / LGBCAT_MODELS),
                                                            "matched.blend.catboost_weight": 0.0}))
    art = api.fit(cfg, save=False, log=QUIET)
    assert art.catboost is not None
    pred = api.predict(art, pd.read_parquet(world / "raw" / "scored.parquet"), log=QUIET)
    assert "cbv_blend" not in pred.decided_by.unique()


# ---------------------------------------------------------------------------------------------- the REAL stack stage

@pytest.mark.slow
def test_lgbcat_stack_stage_on_the_dwg_tables_replays_v46s_residual():
    """FM1's stack stage (C.stack_values: the package frame from the RAW scored rows with adsb.table_dir = the DWG tables,
    models_lgbcat, the manifest gate) fed v46's stored blend F reproduces v46's stored residual on every covered, gated
    ADS-B-stage row of two cells (EDDF January, LSZH July). TRUE: max |resid - stored| <= 1e-6 s and the eligible set ==
    the stored nonzero-residual set. Runs in a child (feature build + the stack's MLP worker)."""
    code = f"""
import sys, json, resource, numpy as np, pandas as pd
ROOT = {str(ROOT)!r}
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/tools')
import fm1_replay_gate as GT
from prc.pipeline.fm1 import components as C, features as F, knobs as K
cfg = K.parse_config(K.nest({{"adsb.table_dir": {DWG_TABLES!r}, "adsb.stack.models_dir": {LGBCAT_MODELS!r},
                              "adsb.stack.f_model": "lgbcat"}}))
weather = F.stand_ab().load_weather(cfg.path("data.weather_dir"))
orl = F.read_orl_routes(cfg.path("features.orl_routes_file"))
p = pd.read_parquet(ROOT + '/data/r12/r13_arms_refit/predictions_2026.parquet').set_index('MVT_ID_mvt')
d = pd.read_parquet(ROOT + '/data/r12/serve_blf_lgbcat/matched_2026_lgbcat_detail.parquet').set_index('MVT_ID_mvt')
models = cfg.path("adsb.stack.models_dir")
gate, feats = C.gate_manifest(models), C.stack_feats(models)
rep = {{}}
for ap, mo in (("EDDF", 1), ("LSZH", 7)):
    m = F.build_frames(GT.airport_month_rows(GT.PATHS, ap, mo), serve=True, cfg=cfg, weather=weather, orl_routes=orl).matched
    m = m[m.MVT_ID_mvt.isin(d.index[d.lane == 'N_v4'])].reset_index(drop=True)
    ids = m.MVT_ID_mvt.to_numpy(float)
    vals, elig, resid, info = C.stack_values(m, p.pred_lgbcat.reindex(ids).to_numpy(float), feats, models, gate, 2)
    stored = d.resid.reindex(ids).to_numpy(float)
    rep[f"{{ap}}-{{mo}}"] = {{"n": int(len(ids)), "n_elig": int(elig.sum()), "n_stored_nonzero": int((stored != 0).sum()),
        "elig_equals_stored_nonzero": bool(np.array_equal(elig, stored != 0)),
        "max_abs_resid_diff": float(np.max(np.abs(resid - stored))) if len(ids) else 0.0}}
rep["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3
print(json.dumps(rep))
"""
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=1200, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    for cell in ("EDDF-1", "LSZH-7"):
        c = rep[cell]
        assert c["n"] > 1_000 and c["n_elig"] > 500, (cell, c)
        assert c["elig_equals_stored_nonzero"], (cell, c)
        assert c["max_abs_resid_diff"] <= 1e-6, (cell, c)
    assert rep["peak_rss_gib"] <= 4.5
