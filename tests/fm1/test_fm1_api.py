"""prc.api end to end on the real-data-shaped e2e world: AT3 (determinism), AT4 (artefacts), AT5 (provenance), AT6 (MTX
last, on the real path), AT9 (raw -> fit -> predict -> build-submission). The world is two airports x two training days in
Jan + Feb 2025 and a scored Jan-16 file whose departures' labels are hidden (no 2026 label anywhere)."""
from __future__ import annotations

import json
import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest

from prc import api
from prc.pipeline.fm1 import artifact as A
from prc.pipeline.fm1 import engine as G
from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import stages as S

pytestmark = pytest.mark.slow          # ~90 s: three real fits (two with the CatBoost worker and the stack's MLP worker)
QUIET = lambda *_a, **_k: None          # noqa: E731


@pytest.fixture(scope="module")
def world(e2e_world):
    return e2e_world[0]


@pytest.fixture(scope="module")
def cfg(world, fm1_world_mod, tmp_path_factory):
    raw = fm1_world_mod.config_raw(world)
    raw["run"]["artifacts_dir"] = str(tmp_path_factory.mktemp("fm1_artifacts"))
    return K.parse_config(raw)


@pytest.fixture(scope="module")
def art(cfg):
    return api.fit(cfg, log=QUIET)


@pytest.fixture(scope="module")
def scored(world):
    return pd.read_parquet(world / "raw" / "scored.parquet")


@pytest.fixture(scope="module")
def pred(art, scored):
    return api.predict(art, scored, log=QUIET)


# ---------------------------------------------------------------------------------------------- fit + artefact

def test_fit_saves_a_versioned_artefact(art, cfg):
    d = art.directory
    assert d.parent == cfg.path("run.artifacts_dir") and d.name == f"{cfg.short_hash}-{art.data_digest[:8]}"
    man = json.loads((d / "manifest.json").read_text())
    assert man["config_hash"] == cfg.hash and man["git"]["sha"] and man["libraries"]["lightgbm"] == "4.5.0"
    assert {"learners/base/learner.json", "learners/catboost/learner.json", "unmatched_lane/lane.json",
            "unmatched_lane/unm.parquet", "stack/MANIFEST.json", "encodings_matched.json", "config.json"} <= set(man["files"])
    assert any(k.startswith("raw/training_2025-01") for k in man["inputs"])
    assert art.fit_report["stages"] == ["input_contract", "features", "base_model", "adsb_gate", "adsb_stack", "cbv_blend",
                                        "unmatched_body", "finalise"]


def test_artefact_is_never_overwritten(art):
    with pytest.raises(E.ArtifactError, match="already exists"):
        A.save(art)


# ---------------------------------------------------------------------------------------------- predict + provenance

def test_predict_serves_every_scored_departure_once(pred, scored, world):
    dep = scored[scored.PHASE_mvt == "DEP"]
    assert pred.MVT_ID_mvt.tolist() == dep.MVT_ID_mvt.tolist()
    assert pred.TAXITIME_SEC_mvt.dtype == np.int32 and (pred.TAXITIME_SEC_mvt >= 1).all()
    assert set(pred.decided_by) <= S.DECIDED_BY and pred.decided_by.notna().all()
    info = json.loads((world / "world.json").read_text())
    assert len(pred) == info["n_scored_dep"]


def test_decided_by_follows_the_lanes(pred):
    """AT5 + the routing rule: matched rows -> adsb_stack where eligible (covered AND gated), else cbv_blend; unmatched
    rows -> unmatched_body. Mutation: write the blend on ALL matched rows -> eligible rows read cbv_blend -> red."""
    m = pred.lane == "matched"
    assert (pred.decided_by[m & pred.stack_eligible] == "adsb_stack").all()
    assert (pred.decided_by[m & ~pred.stack_eligible] == "cbv_blend").all()
    assert (pred.decided_by[~m] == "unmatched_body").all()
    assert pred.stack_eligible.any() and (~pred.stack_eligible & m).any() and (~m).any()
    rep = pred.attrs["fm1_report"]
    assert rep["decided_by"] == pred.decided_by.value_counts().sort_index().to_dict()
    assert rep["gate"]["EDDF"] is True and rep["gate"]["LSZH"] is True and rep["gate"]["EGLL"] is False


def test_component_outputs_reproduce_the_served_value(pred):
    """Every row's value is recomputable from its own component columns (the formulas in components.py)."""
    from prc.pipeline.fm1 import components as C
    m = pred[pred.lane == "matched"]
    st = m[m.stack_eligible]
    assert np.array_equal(st.TAXITIME_SEC_mvt, np.maximum(C.staged_value(_proxy(st), st.F, st.stack_resid), 1))
    bl = m[~m.stack_eligible]
    assert np.array_equal(bl.TAXITIME_SEC_mvt, np.maximum(C.blend_value(_proxy(bl), bl.catboost_C, bl.base_value, 0.5), 1))
    u = pred[pred.lane == "unmatched"]
    assert np.array_equal(u.TAXITIME_SEC_mvt, np.maximum(np.rint(u.unmatched_pred), 1))


def _proxy(frame):
    # base_value = rint(max(proxy - F, 1)); proxy is not a prediction column, recover it from the scored rows
    return _PROXY.reindex(frame.MVT_ID_mvt.to_numpy()).to_numpy()


_PROXY = pd.Series(dtype="float64")


@pytest.fixture(scope="module", autouse=True)
def _proxy_lookup(scored):
    global _PROXY
    dep = scored[(scored.PHASE_mvt == "DEP") & scored.AOBT_3_flt.notna()]
    _PROXY = pd.Series((dep.MVT_TIME_UTC_mvt - dep.AOBT_3_flt).dt.total_seconds().to_numpy(),
                       index=dep.MVT_ID_mvt.to_numpy(dtype="float64"))


# ---------------------------------------------------------------------------------------------- AT3 + AT4

def test_fit_is_bit_deterministic(cfg, scored, pred):
    """AT3: a second fit of the same config and data predicts bit-identically (every column)."""
    again = api.predict(api.fit(cfg, save=False, log=QUIET), scored, log=QUIET)
    pd.testing.assert_frame_equal(again, pred, check_exact=True)


def test_loaded_artefact_predicts_bit_exactly(art, scored, pred):
    """AT4: predictions from the reloaded artefact equal the in-memory ones exactly. Mutation: load the LightGBM boosters
    with num_iteration=best_iter -> differs -> red."""
    back = api.load_artifact(art.directory)
    pd.testing.assert_frame_equal(api.predict(back, scored, log=QUIET), pred, check_exact=True)
    assert back.key == art.key and back.config.hash == art.config.hash


def test_find_artifact_by_config(cfg, art, scored, pred):
    pd.testing.assert_frame_equal(api.predict(cfg, scored, log=QUIET), pred, check_exact=True)


def test_tampered_extra_or_missing_files_are_refused(art, tmp_path):
    d = tmp_path / "copy"
    shutil.copytree(art.directory, d)
    A.load(d)
    p = d / "learners" / "base" / "lgb_seed0.txt"
    p.write_text(p.read_text() + "\n")
    with pytest.raises(E.ArtifactError, match=r"changed since the artefact was written: \['learners/base/lgb_seed0.txt'\]"):
        A.load(d)
    shutil.rmtree(d)
    shutil.copytree(art.directory, d)
    (d / "notes.txt").write_text("x")
    with pytest.raises(E.ArtifactError, match=r"unexpected \['notes.txt'\]"):
        A.load(d)
    (d / "notes.txt").unlink()
    (d / "gate.json").unlink()
    with pytest.raises(E.ArtifactError, match=r"missing \['gate.json'\]"):
        A.load(d)
    with pytest.raises(E.ArtifactError, match="no manifest.json"):
        A.load(tmp_path)


# ---------------------------------------------------------------------------------------------- refusals

def test_predict_refuses_labelled_scored_rows(art, world):
    labelled = pd.read_parquet(world / "raw" / "training_2025-01-01_2025-02-01.parquet")
    with pytest.raises(E.LabelLeakError, match="scored departure\\(s\\) carry a label"):
        api.predict(art, labelled, log=QUIET)


def test_predict_refuses_an_unknown_airport(art, scored):
    bad = scored.copy()
    bad.loc[bad.index[bad.PHASE_mvt == "DEP"][0], "ADEP_mvt"] = "EKCH"
    with pytest.raises(E.UnknownAirportError, match="EKCH"):
        api.predict(art, bad, log=QUIET)


def test_fit_refuses_unported_components_and_heavy_runs(cfg, monkeypatch):
    with pytest.raises(E.ComponentNotPortedError, match="modules.storm.feature"):
        api.fit(cfg, overrides=["modules.storm.feature=true"], save=False, log=QUIET)
    # the Rome module is ported but LIRF-only: a registry without LIRF refuses it by name (Amendment A3)
    with pytest.raises(E.ConfigValueError, match="modules.rome.enabled needs LIRF in airports.codes"):
        api.fit(cfg, overrides=["modules.rome.enabled=true"], save=False, log=QUIET)
    monkeypatch.setattr(G, "HEAVY_ROWS", 100)
    with pytest.raises(E.PipelineError, match="heavy job"):
        api.fit(cfg, save=False, log=QUIET)


def test_catboost_never_loads_in_this_process(pred):
    """AT11 negative invariant after a real fit + predict with the CBV blend on."""
    import sys
    assert "catboost" not in sys.modules
    assert "torch" not in sys.modules          # the stack's MLP ran in its worker (BC-5)


# ---------------------------------------------------------------------------------------------- build-submission

def test_build_submission_writes_the_template_order(art, world, tmp_path, pred):
    res = api.build_submission(art, tmp_path / "out", version=99, log=QUIET)
    sub = pd.read_parquet(res["submission"])
    tpl = pd.read_parquet(world / "raw" / "template.parquet")
    assert res["submission"].name == "merry-quicksand_v99.parquet"
    assert sub.MVT_ID_mvt.tolist() == tpl.MVT_ID_mvt.tolist() and list(sub.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    assert sub.TAXITIME_SEC_mvt.dtype == np.int32
    got = sub.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    assert (got == pred.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(got.index)).all()
    man = json.loads(res["manifest"].read_text())
    assert man["sha256"] == A.sha256_file(res["submission"]) and man["artifact_dir"] == str(art.directory)
    assert pd.read_parquet(res["provenance"]).decided_by.notna().all()
    with pytest.raises(E.ArtifactError, match="never overwrites"):
        api.build_submission(art, tmp_path / "out", version=99, log=QUIET)


def test_build_submission_refuses_a_previous_submission_as_input(art, tmp_path):
    fake = tmp_path / "submissions" / "x.parquet"
    fake.parent.mkdir()
    fake.write_bytes(b"")
    with pytest.raises(E.SpliceRefusedError, match="never reads a previous submission"):
        G.build_submission(art, tmp_path / "o", template_file=fake, log=QUIET)


# ---------------------------------------------------------------------------------------------- MTX on the real path

def test_mtx_runs_last_and_hits_its_designed_means(cfg, scored, world):
    """AT6 on the real path: scored-period purpose turns MTX on; it fits the 2025 excess mapping on the training rows and
    shifts each (airport, month) mean toward the PUBLISHED 2025 cell (the world is a 2025 day). After finalise, each
    calibrated cell's pre-rint mean equals its designed mean; decided_by carries +mtx on moved rows."""
    ref = str(K.ROOT / "data/mtx/raw/taxi_out_additional_time_2025.csv")
    c = K.parse_config(K.nest({**cfg.values, "run.purpose": "scored_period", "calibration.mtx.published_csv": ref,
                               "calibration.mtx.lambda_rule": "full"}))
    a = api.fit(c, save=False, log=QUIET)
    p = api.predict(a, scored, log=QUIET)
    rep = p.attrs["fm1_report"]["mtx"]
    assert rep, "no calibrated cell"
    for key, r in rep.items():
        ap, mo = key.split("|")
        sel = (p.MVT_ID_mvt.isin(scored[scored.ADEP_mvt == ap].MVT_ID_mvt))
        before = p.value_before_calibration[sel].to_numpy()
        after = before + p.calibration_shift[sel].to_numpy()
        assert abs(before.mean() - r["mean_before"]) < 1e-9
        assert abs(after.mean() - (r["mean_before"] + r["delta"])) < 1e-9       # fp64 over < 1,000 values
    moved = p.calibration_shift != 0
    assert moved.any() and p.decided_by[moved].str.endswith("+mtx").all()
    assert not p.decided_by[~moved].str.contains("mtx").any()


def test_regression_subset_airports_fit_on_multi_airport_files(cfg, scored):
    """Found in self-review 2026-09-28 (FM1 notes N5): fit validated the raw training rows against the airport registry
    BEFORE filtering them to it, so a config with a subset of the airports refused the organiser's monthly files (which hold
    all ten). A scored file with a departure outside the registry is still refused at predict (the contract)."""
    sub = K.parse_config(K.nest({**cfg.values, "airports.codes": ["EDDF"], "airports.tz": {"EDDF": "Europe/Berlin"}}))
    art = api.fit(sub, save=False, log=QUIET)
    assert art.fit_report["matched"]["n_rows"] > 0
    only = scored[np.where(scored.PHASE_mvt == "DEP", scored.ADEP_mvt == "EDDF", scored.ADES_mvt == "EDDF")]
    p = api.predict(art, only, log=QUIET)
    assert len(p) == int(((only.PHASE_mvt == "DEP")).sum()) and (p.TAXITIME_SEC_mvt >= 1).all()
    with pytest.raises(E.UnknownAirportError, match="LSZH"):
        api.predict(art, scored, log=QUIET)
