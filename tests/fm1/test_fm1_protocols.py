"""FM1 validation protocols (prc/pipeline/fm1/protocols.py): the fold definitions of Amendment A2 (primary LOMO-Jan /
LOMO-Jul, secondary strict Jan<->Jul swap, tertiary forward), a real run on the protocol world, the stack-leakage guard,
and the paired day-block comparison (== mtx_calib.gain_ci, the project's registered bootstrap)."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from prc import api
from prc.pipeline.fm1 import artifact as A
from prc.pipeline.fm1 import engine as G
from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import protocols as P
from prc.pipeline.fm1 import rows as R
from prc.pipeline.fm1 import shipped

QUIET = lambda *_a, **_k: None          # noqa: E731
ALL = set(range(1, 13))


# ---------------------------------------------------------------------------------------------- definitions

def test_primary_protocol_is_lomo_jan_and_jul():
    """Amendment A2. Mutation: hold out month 1 in F2 -> red."""
    f1, f2 = P.PROTOCOLS["S"]
    assert (f1.name, set(f1.train), f1.es, f1.test) == ("F1", ALL - {1}, (3, 9), (1,))
    assert (f2.name, set(f2.train), f2.es, f2.test) == ("F2", ALL - {7}, (3, 9), (7,))


def test_swap_is_strict_one_month_each_way():
    a, b = P.PROTOCOLS["swap"]
    assert (a.train, a.test, a.es, a.es_days_from) == ((1,), (7,), (), 24)
    assert (b.train, b.test, b.es, b.es_days_from) == ((7,), (1,), (), 24)
    assert P.split_of(a) == {"train_months": [1], "es_days_from": 24}


def test_forward_and_lomo_folds():
    f3, f4 = P.PROTOCOLS["forward"]
    assert (f3.train, f3.es, f3.test) == ((1, 2, 3, 4, 5, 6), (3,), (7,))
    assert (f4.train, f4.es, f4.test) == (tuple(range(1, 10)), (3, 9), (10, 11, 12))
    lomo = P.PROTOCOLS["lomo"]
    assert [f.test for f in lomo] == [(m,) for m in range(1, 13)]
    for f in lomo:
        assert f.test[0] not in f.train and f.test[0] not in f.es and set(f.es) <= set(f.train)
    assert [f.test for f in P.PROTOCOLS["forward_monthly"]] == [(m,) for m in range(7, 13)]
    for f in P.PROTOCOLS["forward_monthly"]:
        assert max(f.train) < f.test[0]


def test_folds_restrict_to_available_months():
    got = P.folds("forward", [1, 2, 3, 6, 7, 9, 10])
    assert got[0].train == (1, 2, 3, 6) and got[0].test == (7,)
    assert got[1].test == (10,) and got[1].es == (3, 9)
    got = P.folds("S", [2, 3, 7])
    assert got[0].test == () and not P._usable(got[0])
    with pytest.raises(E.ConfigValueError, match="unknown protocol 'kfold'"):
        P.folds("kfold", [1])


def test_tiers():
    cfg = K.parse_config({})
    assert P.tier_overrides(cfg) == {"learning_rate": 0.05, "seeds": [0]}
    assert P.tier_overrides(K.parse_config({"validate": {"tier": "full"}})) == {}


# ---------------------------------------------------------------------------------------------- the paired comparison

def _records(n=400, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    y = rng.normal(900, 300, n).clip(60)
    day = np.repeat([f"2025-01-{d:02d}" for d in range(1, 21)], n // 20)
    ap = np.where(np.arange(n) % 3 == 0, "EDDF", "LSZH")
    a = pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float), "y": y, "pred": y + rng.normal(0, 120, n), "day": day,
                      "month": 1, "ap": ap})
    b = a.assign(pred=a.pred - shift * (a.pred - a.y))
    return a, b


def test_compare_equals_the_registered_bootstrap():
    """gain > 0 = b better; the CI is mtx_calib.gain_ci on the same arrays. Mutation: swap a and b inside compare -> the
    sign flips -> red."""
    a, b = _records(shift=0.3)
    rep = P.compare(a, b, n_boot=300, seed=7)
    g, lo, hi = shipped.mtx_calib().gain_ci(a.y, a.pred, b.pred, a.day, a.month, P.BOARD_N, n_boot=300, seed=7)
    assert rep["gain_bmse"] == g > 0 and rep["ci95"] == [lo, hi] and lo > 0
    assert rep["n"] == 400
    assert sum(rep["by_airport_bmse"].values()) == pytest.approx(g, rel=1e-12)     # airport parts add to the whole
    assert rep["excluding_dominant_airport"]["airport"] in ("EDDF", "LSZH")
    assert rep["excluding_top3_rows"]["gain_bmse"] < g


def test_compare_refuses_disjoint_or_disagreeing_records():
    a, b = _records()
    with pytest.raises(E.ValidationError, match="share no row"):
        P.compare(a, b.assign(MVT_ID_mvt=b.MVT_ID_mvt + 10_000), n_boot=100, seed=0)
    with pytest.raises(E.ValidationError, match="disagree on a label"):
        P.compare(a, b.assign(y=b.y + 1), n_boot=100, seed=0)


# ---------------------------------------------------------------------------------------------- real runs

@pytest.fixture(scope="module")
def pcfg(protocol_world, fm1_world_mod):
    d = protocol_world[0]
    raw = fm1_world_mod.config_raw(d)
    raw["data"]["train_months"] = [1, 2, 3, 6, 7, 9, 10]
    raw["matched"]["es_months"] = [3, 9]
    raw["adsb"]["stack"]["enabled"] = False                    # the stored stack trained on Jan + Jul (leakage guard)
    return K.parse_config(raw)


@pytest.mark.slow
def test_protocol_s_runs_and_writes_oof(pcfg, tmp_path):
    rep = api.validate(pcfg, "S", out_dir=tmp_path / "S", log=QUIET)
    assert set(rep["folds"]) == {"F1", "F2"} and rep["skipped"] == {}
    for name, f in rep["folds"].items():
        oof = pd.read_parquet(tmp_path / "S" / f"oof_{name}.parquet")
        assert len(oof) == f["n"] > 0 and (oof.y > 0).all() and set(oof.month) == set(f["test"])
        assert np.isclose(f["rmse"], np.sqrt(((oof.y - oof.pred) ** 2).mean()))
        assert f["fit"]["train_months"] == f["train"] and f["fit"]["seeds"] == [0]          # screen tier
        assert f["fit"]["tier"] == {"learning_rate": 0.05}
        assert not set(f["test"]) & set(f["train"])
        # save_models defaults to False: no models dir, no artefact bookkeeping in the report or the oof file
        assert "artifact_dir" not in f and "artifact_manifest_sha256" not in f
        assert "artifact_dir" not in oof.columns and "artifact_manifest_sha256" not in oof.columns
    assert not (tmp_path / "S" / "models").exists()
    with pytest.raises(E.ArtifactError, match="not empty"):
        api.validate(pcfg, "S", out_dir=tmp_path / "S", log=QUIET)


# ---------------------------------------------------------------------------------------------- save_models (plan v3)

@pytest.fixture(scope="module")
def s_saved(pcfg, tmp_path_factory):
    d = tmp_path_factory.mktemp("s_saved")
    rep = api.validate(pcfg, "S", out_dir=d, log=QUIET, save_models=True)
    return d, rep


@pytest.mark.slow
def test_save_models_writes_one_artefact_per_fold_reproducing_oof(s_saved):
    """Plan v3 (owner-adopted 2026-09-28): every fit saves its boosters, feature manifest, predictions, row order,
    config, seed and hashes. Each fold's saved artefact, re-loaded, reproduces that fold's OOF predictions
    bit-exactly on the fold's own test rows. Mutation: save under a shared parent across folds (collision) or drop
    the artefact_dir/sha bookkeeping -> red."""
    d, rep = s_saved
    seen_dirs = set()
    for name, f in rep["folds"].items():
        art_dir = pathlib.Path(f["artifact_dir"])
        assert art_dir.is_dir() and art_dir not in seen_dirs
        seen_dirs.add(art_dir)
        assert art_dir.parent == d / "models" / name          # the keyed artefact dir sits under <out>/models/<fold>/
        man = A.verify(art_dir)          # re-hashes every file against the manifest (AT4); raises if anything is off
        assert man["config_hash"] == rep["config_hash"]
        assert f["artifact_manifest_sha256"] == A.sha256_file(art_dir / A.MANIFEST)

        oof = pd.read_parquet(d / f"oof_{name}.parquet")
        assert set(oof.artifact_dir) == {str(art_dir)}
        assert set(oof.artifact_manifest_sha256) == {f["artifact_manifest_sha256"]}

        loaded = api.load_artifact(art_dir)
        [m] = f["test"]
        raw = pd.read_parquet(G.month_file(loaded.config, m))
        raw = raw[G.in_registry(raw, loaded.config)]
        served, _ = R.hide_labels(raw)
        pred = G.predict(loaded, served, log=QUIET)
        got = pred.set_index(pred.MVT_ID_mvt.astype(float)).TAXITIME_SEC_mvt.astype("float64")
        want_ids = oof.MVT_ID_mvt.to_numpy(float)
        np.testing.assert_array_equal(got.reindex(want_ids).to_numpy(), oof.pred.to_numpy())


def test_save_models_needs_out_dir(pcfg):
    with pytest.raises(E.ConfigValueError, match="save_models=True needs out_dir"):
        api.validate(pcfg, "S", out_dir=None, log=QUIET, save_models=True)


@pytest.mark.slow
def test_save_models_never_overwrites_an_existing_artefact(s_saved):
    """The package's 'never overwrite an artefact' rule (artifact.py) stays in force for fold artefacts too.
    Mutation: pass a fresh, non-colliding parent to A.save -> no raise -> red."""
    d, rep = s_saved
    name = next(iter(rep["folds"]))
    art_dir = pathlib.Path(rep["folds"][name]["artifact_dir"])
    loaded = api.load_artifact(art_dir)
    with pytest.raises(E.ArtifactError, match="already exists; FM1 never overwrites an artefact"):
        A.save(loaded, parent=art_dir.parent)


@pytest.mark.slow
def test_swap_protocol_trains_on_one_month(pcfg, tmp_path):
    rep = api.validate(pcfg, "swap", out_dir=tmp_path / "X", log=QUIET)
    assert set(rep["folds"]) == {"X1to7", "X7to1"}
    x = rep["folds"]["X1to7"]
    assert x["fit"]["train_months"] == [1] and x["fit"]["early_stopping"] == {"days_from": 24} and x["test"] == [7]


def test_leakage_guard_refuses_the_stack_on_its_training_months(pcfg):
    """Mutation: drop the guard -> the fold fits and scores with a stack that saw January's labels -> red."""
    on = pcfg.replace(**{"adsb.stack.enabled": True})
    with pytest.raises(E.ConfigValueError, match=r"fold F1 tests month\(s\) \[1\] that trained the stored ADS-B stack"):
        api.validate(on, "S", log=QUIET)


def test_validate_refuses_mtx(pcfg):
    with pytest.raises(E.ConfigValueError, match="validate runs the forecaster"):
        api.validate(pcfg.replace(**{"calibration.mtx.enabled": "on"}), "S", log=QUIET)


def test_dry_run_reports_without_fitting():
    rep = api.validate(K.parse_config({}), "dry-run")
    assert rep["stages"][-2:] == ["rome_e2w", "finalise"] and "plan_refused" not in rep
    rep = api.validate(K.parse_config({"modules": {"storm": {"feature": True}}}), "dry-run")
    assert rep["stages"] is None and "modules.storm.feature" in rep["plan_refused"]
    assert rep["paths"]["data.raw_dir"]["exists"] is True
    assert rep["training_months"][1].startswith("training_2025-01")


# ---------------------------------------------------------------------------------------------- day-4 enablers

def test_compare_pools_several_records_per_side():
    """The primary decision statistic pools F1 + F2 (Amendment A2): compare([a1, a2], [b1, b2]) == compare on the
    concatenations. Mutation: use only the first record of each side -> n halves -> red."""
    a1, b1 = _records(seed=1, shift=0.2)
    a2, b2 = _records(seed=2, shift=0.2)
    a2 = a2.assign(MVT_ID_mvt=a2.MVT_ID_mvt + 10_000, month=7)
    b2 = b2.assign(MVT_ID_mvt=b2.MVT_ID_mvt + 10_000, month=7)
    pooled = api.compare([a1, a2], [b1, b2], n_boot=200, seed=3)
    ref = P.compare(pd.concat([a1, a2], ignore_index=True), pd.concat([b1, b2], ignore_index=True), n_boot=200, seed=3)
    assert pooled == ref and pooled["n"] == 800
    with pytest.raises(E.ValidationError, match="duplicate"):
        api.compare([a1, a1], [b1, b1], n_boot=100, seed=0)


def test_blend_oof_is_the_cbv_formula_on_common_rows():
    """Arm A2 offline (FMA §4.3): rint(w * pred_b + (1 - w) * pred_a) on the rows both records hold; the labels must agree.
    Mutation: weight the wrong side (w on a) -> red."""
    a, b = _records(seed=4, shift=0.5)
    bl = api.blend_oof(a, b, 0.25)
    assert np.array_equal(bl.pred.to_numpy(), np.rint(0.25 * b.pred.to_numpy() + 0.75 * a.pred.to_numpy()))
    assert bl.y.tolist() == a.y.tolist() and set(bl.decided_by) == {"blend_oof"}
    with pytest.raises(E.ConfigValueError, match=r"w must be in \[0, 1\]"):
        api.blend_oof(a, b, 1.5)
    with pytest.raises(E.ValidationError, match="disagree on a label"):
        api.blend_oof(a, b.assign(y=b.y + 1), 0.5)
