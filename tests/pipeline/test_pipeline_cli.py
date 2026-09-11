"""prc.pipeline.cli -- `python -m prc.pipeline run --config ... --out ... [--dry-run]`.

The acceptance-A3 shape in P2a: a synthetic ELEVENTH airport (EKCH, no history) flows through the dry run
on files written to disk -- config, schemas, routing, lane inputs -- lands in the pooled lanes and is flagged
as a fallback; the same file under a ten-airport config is refused by name. The dry run fits and loads
nothing (proven by making every fit / booster load raise). There is no --base flag.

Mutation rehearsal (2026-09-11, scratchpad p2a/mutate.py): the break named in each docstring turned the test
RED against prc/pipeline/cli.py (or the module named), GREEN on restore.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pipeline_world", pathlib.Path(__file__).with_name("pipeline_world.py"))
pw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pw)

from prc.pipeline import cli, ingest  # noqa: E402
from prc.pipeline import config as C  # noqa: E402
from prc.pipeline import errors as E  # noqa: E402
from prc.pipeline import models as M  # noqa: E402


def _no_fitting(monkeypatch):
    import lightgbm as lgb

    def boom(*a, **k):
        raise AssertionError("the dry run fitted or loaded a model")
    monkeypatch.setattr(M.LightGBMModel, "fit", boom)
    monkeypatch.setattr(M.LightGBMModel, "predict", boom)
    monkeypatch.setattr(M.NonFillBodyModel, "fit", boom)
    monkeypatch.setattr(lgb, "Booster", boom)
    monkeypatch.setattr(lgb, "train", boom)


def test_an_eleventh_airport_flows_through_the_dry_run_flagged_as_a_fallback(tmp_path, monkeypatch):
    """Rehearsed: dry_run() computing fallback_rows from every row (not routing.fallback) -> RED."""
    w = pw.dry_run_world(tmp_path / "w")
    _no_fitting(monkeypatch)
    out = tmp_path / "out"
    assert cli.main(["run", "--config", str(w.config), "--out", str(out), "--dry-run"]) == 0
    rep = json.loads((out / "dry_run.json").read_text())
    assert sorted(p.name for p in out.iterdir()) == ["dry_run.json"]                 # nothing predicted, nothing written
    assert rep["fallback_airports"] == ["EKCH"] and rep["fallback_rows"] == {"EKCH": w.n_by_airport["EKCH"]}
    assert rep["routing"]["by_airport"]["EKCH"] == {"matched": 12, "unmatched": 3}
    assert rep["routing"]["by_airport"]["LIRF"] == {"matched": 12, "lirf_rules": 3}
    assert sum(rep["routing"]["counts"].values()) == rep["scored"]["n_scored"] == len(w.dep) == 11 * 15
    assert len(rep["routing"]["by_airport"]) == 11
    assert rep["no_splice"]["asserted"] and rep["inputs"]["scored_file"]["sha256"]
    assert rep["lane_inputs"]["matched"]["n_ref"] == 48


def test_the_dry_run_refuses_the_eleventh_airport_under_a_ten_airport_config(tmp_path, capsys):
    """Rehearsed: main() catching PipelineError and returning 0 -> RED."""
    w = pw.dry_run_world(tmp_path / "w", config_airports=pw.TEN)
    rc = cli.main(["run", "--config", str(w.config), "--out", str(tmp_path / "out"), "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 1 and "UnknownAirportError" in err and "EKCH" in err
    assert not (tmp_path / "out" / "dry_run.json").exists()


def _with_adsb_stage(w, tmp_path, gate):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    t = pa.Table.from_pandas(pd.DataFrame({"MVT_ID_mvt": w.matched_ids.astype("float64"), "coverage": 3,
                                           "first_ground_rel_aobt3": 0.0}), preserve_index=False)
    table = tmp_path / "features_2026.parquet"
    pq.write_table(t.replace_schema_metadata({**(t.schema.metadata or {}), b"git_sha": b"x", b"extractor": b"e",
                                              b"features_version": b"v2"}), table)
    models = tmp_path / "models"
    models.mkdir()
    (models / "MANIFEST.json").write_text(json.dumps({"feats": ["first_ground_rel_aobt3"], "seeds": [0], "folds": 1,
                                                      "airports": pw.TEN, "gate": gate}))
    for n in ("G_s0_k0.txt", "N_s0_k0.pt", "N_s0_k0_stats.npz"):
        (models / n).write_bytes(b"x")
    raw = dict(w.raw)
    raw["lanes"]["matched"]["adsb_stage"] = {"table": str(table), "models_dir": str(models)}
    w.config.write_text(yaml.safe_dump(raw, sort_keys=False))


def test_the_dry_run_checks_the_adsb_stage_inputs_before_the_matched_lane_runs(tmp_path, monkeypatch, capsys):
    """With lanes.matched.adsb_stage configured, the dry run reports the stage (gate, model files, table) and refuses an
    empty gate by name -- a missing table or model would otherwise surface only after the matched lane's ~20 minutes.
    Rehearsed 2026-09-11 (c4): _check_matched_inputs without the stage call -> RED (no report, the empty gate passes)."""
    w = pw.dry_run_world(tmp_path / "w")
    _no_fitting(monkeypatch)
    _with_adsb_stage(w, tmp_path, {"EHAM": {"allowed": True, "rule": "r", "c25": 0.7, "c26": 0.8}})
    out = tmp_path / "out"
    assert cli.main(["run", "--config", str(w.config), "--out", str(out), "--dry-run"]) == 0
    st = json.loads((out / "dry_run.json").read_text())["lane_inputs"]["matched"]["adsb_stage"]
    assert st["gate_allowed"] == ["EHAM"] and st["n_rows"] == len(w.matched_ids) and st["n_model_files"] == 3
    w2 = pw.dry_run_world(tmp_path / "w2")
    (tmp_path / "e").mkdir()
    _with_adsb_stage(w2, tmp_path / "e", {})
    rc = cli.main(["run", "--config", str(w2.config), "--out", str(tmp_path / "out2"), "--dry-run"])
    assert rc == 1 and "no `gate`" in capsys.readouterr().err


def test_there_is_no_base_flag_and_no_abbreviation_can_stand_in_for_one():
    """Rehearsed: allow_abbrev=True on the run parser -> RED (--dry is accepted as --dry-run)."""
    base = ["run", "--config", "c.yaml", "--out", "o"]
    for extra in (["--base", "submissions/merry-quicksand_v2.parquet"], ["--dry"], ["--allow"]):
        with pytest.raises(SystemExit) as exc:
            cli.parse_args(base + extra)
        assert exc.value.code == 2
    assert cli.parse_args(base + ["--dry-run"]).dry_run is True


def test_a_refit_is_refused_without_allow_heavy_before_any_data_is_read(tmp_path, monkeypatch):
    """Rehearsed: run()'s heavy-lane check deleted -> RED (ingest is reached; the spy refuses)."""
    w = pw.dry_run_world(tmp_path / "w")
    raw = w.raw
    raw["lanes"]["matched"]["source"] = "refit"
    del raw["lanes"]["matched"]["boosters"], raw["lanes"]["matched"]["booster_dir"]
    cfg = C.parse_config(raw, root=ROOT)

    def boom(*a, **k):
        raise AssertionError("data read before the heavy-run gate")
    monkeypatch.setattr(ingest, "ingest_scored", boom)
    with pytest.raises(E.PipelineError, match="allow-heavy"):
        cli.run(cfg, tmp_path / "out", log=lambda m: None)


def _manifest(files, outputs=None, out_dir="o1"):
    from prc.pipeline import write
    m = {"code": {"files": dict(files), "differs_from_head": sorted(files)}, "outputs": outputs or {"sub": "abc"},
         "config": {"hash": "c"}, "created_utc": "t", "out_dir": out_dir}
    m["digest"] = write.manifest_digest(m)
    return m


def test_same_run_holds_when_nothing_changed_and_refuses_a_digest_that_differs_anyway(tmp_path):
    """assert_same_run's strict branch: no code file changed between the runs -> the full digests must be equal.
    Rehearsed 2026-09-11 (c4): the strict digest comparison deleted -> RED (the out_dir-leaking pair passes)."""
    import time
    since = time.time()
    pw.assert_same_run(_manifest({"scripts/a.py": "h1"}), _manifest({"scripts/a.py": "h1"}, out_dir="o2"), since, tmp_path)
    m2 = _manifest({"scripts/a.py": "h1"})
    m2["digest"] = "leaked"                                         # e.g. manifest_digest hashing out_dir
    with pytest.raises(AssertionError, match="digest"):
        pw.assert_same_run(_manifest({"scripts/a.py": "h1"}), m2, since, tmp_path)


def test_regression_bug_8_a_code_file_saved_between_the_two_runs_is_an_edit_not_nondeterminism(tmp_path):
    """Regression (2026-09-11 12:05:26): the full suite failed the end-to-end digest test because another session saved
    scripts/adsb_features.py -- loaded in the pytest process, hashed into every manifest's code block -- between the
    test's two runs. The digest was RIGHT to differ; the test's assumption (no edit between runs) was wrong in a shared
    working tree. A hash change is accepted only for a file whose mtime is at or after the first run's start; every other
    part of the manifest must still agree. Bug class BC-8 (reports/bug_classes.md). Rehearsed 2026-09-11 (c4): the mtime
    check deleted -> RED (the no-edit pair passes); the code-less digest comparison deleted -> RED (outputs pair passes)."""
    import os
    import time
    (tmp_path / "scripts").mkdir()
    f = tmp_path / "scripts" / "a.py"
    f.write_text("x = 1\n")
    os.utime(f, (time.time() - 3600, time.time() - 3600))
    since = time.time()
    m1 = _manifest({"scripts/a.py": "h1"})
    edited = _manifest({"scripts/a.py": "h2"}, out_dir="o2")
    assert m1["digest"] != edited["digest"]                       # the old strict comparison fails here: the flake
    with pytest.raises(AssertionError, match="without an on-disk edit"):
        pw.assert_same_run(m1, edited, since, tmp_path)           # hash changed but the file is older than the runs
    f.write_text("x = 2\n")                                       # the other session's save, between the runs
    pw.assert_same_run(m1, edited, since, tmp_path)
    with pytest.raises(AssertionError, match="apart from the code"):
        pw.assert_same_run(m1, _manifest({"scripts/a.py": "h2"}, outputs={"sub": "zzz"}), since, tmp_path)
    with pytest.raises(AssertionError, match="loaded"):
        pw.assert_same_run(m1, _manifest({"scripts/a.py": "h1", "scripts/b.py": "h"}), since, tmp_path)


def test_the_whole_pipeline_runs_end_to_end_on_eleven_airports_and_two_runs_share_one_digest(tmp_path):
    """cli.run on a world where every lane really fits and predicts: one submission row per template id, int32,
    the contract passed, every row in exactly one lane, EKCH's rows flagged as fallback in the provenance and the
    manifest, and a second run into another directory produces the SAME manifest digest and the same bytes.
    Rehearsed: run() dropping the airport_rules lane's result -> RED (assemble refuses a missing lane);
    manifest_digest including out_dir -> RED (the two digests differ)."""
    import pandas as pd
    w = pw.full_world(tmp_path / "w")
    cfg = C.load_config(w.config)
    logs = []
    import time
    since = time.time()
    runs = [cli.run(cfg, tmp_path / f"o{i}", log=logs.append, matched_model=M.LightGBMModel(params=w.params))
            for i in (1, 2)]
    m1, m2 = (r["manifest"] for r in runs)
    pw.assert_same_run(m1, m2, since, cfg.paths.root)
    sub = pd.read_parquet(tmp_path / "o1" / "submission.parquet")
    tmpl = pd.read_parquet(cfg.paths.template_file)
    from prc.pipeline import legacy
    legacy.stratum().bs.check_submission(sub, tmpl)
    assert sub.TAXITIME_SEC_mvt.dtype == "int32" and list(sub.MVT_ID_mvt) == list(tmpl.MVT_ID_mvt)
    prov = pd.read_parquet(tmp_path / "o1" / "submission.provenance.parquet")
    assert m1["routing"]["counts"] == {"matched": len(w.rank_ids), "unmatched": w.n_unmatched - w.n_lirf_unmatched,
                                       "lirf_rules": w.n_lirf_unmatched}
    assert prov.MVT_ID_mvt.is_unique and len(prov) == len(tmpl)
    assert set(prov.MVT_ID_mvt[prov.lane == "matched"]) == set(w.rank_ids.tolist())
    ek = prov[prov.airport == "EKCH"]
    assert len(ek) == 6 and ek.fallback.all() and not prov[prov.airport != "EKCH"].fallback.any()
    assert set(ek.lane) == {"matched", "unmatched"}
    assert m1["fallback_airports"] == ["EKCH"] and m1["airports"]["EKCH"]["fallback"] is True
    assert m1["validation"]["fallback_rows"] == {"EKCH": 6} and m1["validation"]["check_submission"] == "passed"
    assert set(m1["lanes"]) == {"matched", "unmatched", "lirf_rules"} and m1["lanes"]["matched"]["n_ref"] == 90
    assert not any("submissions" in rec["path"] for rec in m1["inputs"].values())
    assert any(k.startswith("boosters/") for k in m1["inputs"]) and any(k.startswith("training/") for k in m1["inputs"])


def test_main_reports_a_config_error_by_name_and_exits_1(tmp_path, capsys):
    """Rehearsed: main() without its PipelineError handler -> RED (the exception escapes)."""
    bad = tmp_path / "bad.yaml"
    raw = pw.fresh(pw.real_raw())
    raw["lanes"]["matched"]["learning_rate"] = 0.1
    import yaml
    bad.write_text(yaml.safe_dump(raw))
    assert cli.main(["run", "--config", str(bad), "--out", str(tmp_path / "o"), "--dry-run"]) == 1
    assert "UnknownKeyError" in capsys.readouterr().err
