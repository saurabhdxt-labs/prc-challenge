"""ADS-B sources in FM1 (plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md): fit reads only the training table, so the serve
table is not a learner input; the tables declare their sources and `adsb.sources.adsb_fi` must agree with them.

Mutation rehearsal per test in its docstring and in tests/fm1/MUTATION_FM1.md (M-AFI rows).
"""
from __future__ import annotations

import dataclasses
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import features as F
from prc.pipeline.fm1 import knobs as K


def _tables(d, train_ids, serve_ids, report=None):
    d.mkdir(parents=True, exist_ok=True)
    keys = {"month": 7, "pos": 0, "ap": "LIRF", "mvt_day": "2025-07-01", "adsb_day_present": 1, "coverage": 3}

    def frame(ids, v):
        f = pd.DataFrame({"MVT_ID_mvt": np.asarray(ids, float), **{k: [x] * len(ids) for k, x in keys.items()}})
        f["joined"] = 1.0
        f["dt_takeoff"] = v
        return f[["month", "pos", "MVT_ID_mvt", "ap", "mvt_day", "adsb_day_present", "coverage", "joined", "dt_takeoff"]]
    frame(train_ids, 1.0).to_parquet(d / "train.parquet", index=False)
    frame(serve_ids, 2.0).to_parquet(d / "serve.parquet", index=False)
    if report is not None:
        (d / "report.json").write_text(json.dumps(report))
    return d


# ---------------------------------------------------------------- fit reads the training table only

def test_fit_reads_only_the_training_table(tmp_path):
    """A training id is looked up in train.parquet only: a serve.parquet that (wrongly) repeats it changes nothing at fit
    (before AFI: DuplicateIdError — the serve table was read at fit), and a training id found only in serve.parquet is
    refused, never served from it. Serving still reads both and still refuses an id in both. Mutation: read both files
    at fit -> red."""
    d = _tables(tmp_path / "t", [1.0, 2.0], [2.0, 9.0])
    fr = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0]})
    out, missing = F.attach_adsb(fr, d, serve=False)
    assert missing == 0 and out.dt_takeoff.tolist() == [1.0, 1.0]
    with pytest.raises(E.SchemaError, match="no ADS-B table row"):
        F.attach_adsb(pd.DataFrame({"MVT_ID_mvt": [9.0]}), d, serve=False)
    with pytest.raises(E.DuplicateIdError):
        F.attach_adsb(fr, d, serve=True, max_missing_share=1.0)
    out, _ = F.attach_adsb(pd.DataFrame({"MVT_ID_mvt": [9.0]}), d, serve=True, max_missing_share=1.0)
    assert out.dt_takeoff.tolist() == [2.0]


# ---------------------------------------------------------------- declared sources

def test_the_knob_must_agree_with_the_tables_declared_sources(tmp_path):
    """adsb.fi in the tables needs the knob on (it is non-commercial and not redistributable: never used unawares); the
    knob on needs adsb.fi in the tables; a report without `sources` (tables built before AFI) is adsb.lol. Mutation:
    check one direction only -> red."""
    lol = _tables(tmp_path / "lol", [1.0], [], {"sources": ["adsb.lol"]})
    fi = _tables(tmp_path / "fi", [1.0], [], {"sources": ["adsb.fi", "adsb.lol"]})
    legacy = _tables(tmp_path / "legacy", [1.0], [], {"stamp": "51348d3e1cb3"})
    F.check_table_sources(lol, adsb_fi=False)
    F.check_table_sources(fi, adsb_fi=True)
    assert F.check_table_sources(legacy, adsb_fi=False) == ["adsb.lol"]
    with pytest.raises(E.ConfigValueError, match="hold adsb.fi points.*adsb.sources.adsb_fi=true"):
        F.check_table_sources(fi, adsb_fi=False)
    with pytest.raises(E.ConfigValueError, match="adsb.sources.adsb_fi is true but"):
        F.check_table_sources(lol, adsb_fi=True)
    with pytest.raises(E.ConfigValueError, match="adsb.sources.adsb_fi is true but"):
        F.check_table_sources(legacy, adsb_fi=True)


@pytest.mark.parametrize("adsb_fi", [False, True])
def test_tables_without_a_report_are_refused(tmp_path, adsb_fi):
    """End-of-session review 2026-10-03 (finding 3): a table directory without report.json was read as adsb.lol, so an
    adsb.fi table written without its report passed with the knob off. Every producer (r12_build tables, afi_build
    fallback) writes one; a directory without it is refused by name. Seen red 2026-10-03 08:25-08:42 EDT."""
    bare = _tables(tmp_path / "bare", [1.0], [])
    with pytest.raises(E.SchemaError, match="report.json"):
        F.check_table_sources(bare, adsb_fi=adsb_fi)


def test_the_adsb_fi_knob_is_ported_and_a_derived_fit_may_switch_it():
    """It only declares what the serve table holds; the learner-input hash (adsb/train.parquet) guards the training data.
    Mutation: leave it not_ported -> a config with it on is refused at plan -> red; keep it out of the allow-list -> red."""
    k = K.KNOBS["adsb.sources.adsb_fi"]
    assert k.status == "implemented" and "adsb.sources.adsb_fi" in K.REUSE_FREE_KNOBS
    assert K.parse_config({"adsb": {"sources": {"adsb_fi": True}}}).not_ported() == []


# ---------------------------------------------------------------- the derived fit (slow, world)

@pytest.mark.slow
def test_a_new_serve_table_does_not_block_a_derived_fit_but_a_new_training_table_does(e2e_world, fm1_world_mod,
                                                                                       tmp_path):
    """v55's path: v50's learners reused with a different serve table (adsb.fi in July 2026) — allowed, and predicts what
    a fresh fit on the new tables predicts; a different training table is still refused by name. Mutation: count
    adsb/serve.parquet as a learner input again -> the reuse is refused -> red."""
    from prc import api
    world = e2e_world[0]
    quiet = lambda *a, **k: None  # noqa: E731
    donor = api.fit(K.parse_config(fm1_world_mod.config_raw(world, **{"run.artifacts_dir": str(tmp_path / "a0")})),
                    save=True, log=quiet)
    new = tmp_path / "tables_fi"
    shutil.copytree(world / "adsb", new)
    srv = pd.read_parquet(new / "train.parquet").head(3).assign(MVT_ID_mvt=[9e9, 9e9 + 1, 9e9 + 2])
    srv.to_parquet(new / "serve.parquet", index=False)
    (new / "report.json").write_text(json.dumps({"sources": ["adsb.fi", "adsb.lol"]}))
    over = {"adsb.table_dir": str(new), "adsb.sources.adsb_fi": True, "run.artifacts_dir": str(tmp_path / "a1")}
    cfg = K.parse_config(fm1_world_mod.config_raw(world, **over))
    derived = api.fit(cfg, save=False, log=quiet, reuse_from=donor.directory)
    scratch = api.fit(cfg, save=False, log=quiet)
    scored = pd.read_parquet(world / "raw" / "scored.parquet")
    a, b = api.predict(derived, scored, log=quiet), api.predict(scratch, scored, log=quiet)
    assert (a.TAXITIME_SEC_mvt.to_numpy() == b.TAXITIME_SEC_mvt.to_numpy()).all()
    # fit and predict both check the declaration: adsb.fi tables with the knob off are refused at each
    off = K.parse_config(fm1_world_mod.config_raw(world, **dict(over, **{"adsb.sources.adsb_fi": False})))
    with pytest.raises(E.ConfigValueError, match="hold adsb.fi points"):
        api.fit(off, save=False, log=quiet, reuse_from=donor.directory)
    with pytest.raises(E.ConfigValueError, match="hold adsb.fi points"):
        api.predict(dataclasses.replace(derived, config=off), scored, log=quiet)
    bad = tmp_path / "tables_bad"
    shutil.copytree(new, bad)
    tr = pd.read_parquet(bad / "train.parquet")
    tr.loc[0, "dt_takeoff"] = 123.0
    tr.to_parquet(bad / "train.parquet", index=False)
    cfg_bad = K.parse_config(fm1_world_mod.config_raw(world, **dict(over, **{"adsb.table_dir": str(bad)})))
    with pytest.raises(E.ReuseError, match="adsb/train.parquet"):
        api.fit(cfg_bad, save=False, log=quiet, reuse_from=donor.directory)


def test_the_adsb_fi_knob_needs_the_adsb_block():
    """End-of-session review 2026-10-03 (finding 7): `adsb.sources.adsb_fi=true` with the ADS-B block off declared a source
    nothing reads, and no check refused it. Seen red 2026-10-03 08:25-08:42 EDT (accepted)."""
    with pytest.raises(E.ConfigValueError, match="adsb.sources.adsb_fi needs features.adsb"):
        K.parse_config({"features": {"adsb": False}, "adsb": {"sources": {"adsb_fi": True}, "stack": {"enabled": False}}})
    K.parse_config({"features": {"adsb": False}, "adsb": {"stack": {"enabled": False}}})        # off with the block off
