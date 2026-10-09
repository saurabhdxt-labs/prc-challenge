"""`python -m prc` (AT10): thin wrappers over prc.api; a refused run exits 1 with the named error; usage errors exit 2;
there is no --base."""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import pytest
import yaml
from isolation import run_guarded

ROOT = pathlib.Path(__file__).resolve().parents[2]
FM1 = ROOT / "configs" / "fm1.yaml"


def _prc(*args, timeout=600):
    return run_guarded([sys.executable, "-B", "-m", "prc", *map(str, args)], capture_output=True, text=True,
                          timeout=timeout, cwd=str(ROOT))


def test_config_reference_check_passes():
    r = _prc("config-reference", "--check")
    assert r.returncode == 0 and "current" in r.stdout


def test_dry_run_of_the_default_config():
    """Amendment A3: with Rome, ARC and the storm anchor ported, the documented forecaster plans every stage.
    The plan does not depend on the data being present; the exit code does (1 = a required input is missing,
    api.dry_run_refused), so this test asserts the plan, and that the exit code agrees with the report — on the
    development machine (data present: 0) and in a fresh clone (no data: 1). Regression 2026-10-06: it asserted exit 0
    and so passed only where the data exists (BC-45: a test that depends on the machine's state); the clone-tier check
    caught it. Mutation: assert returncode == 0 again -> red in a clone; drop the CLI's refusal exit -> red here
    whenever the report is refused."""
    from prc import api
    r = _prc("validate", "--config", FM1, "--protocol", "dry-run")
    rep = json.loads(r.stdout)
    assert rep["not_ported"] == [] and rep["stages"][-6:] == ["unmatched_body", "rome", "arc", "rome_e2", "rome_e2w",
                                                               "finalise"]
    assert r.returncode == (1 if api.dry_run_refused(rep) else 0), r.stderr


@pytest.mark.devtree
def test_unported_component_and_heavy_default_fit_are_refused_with_the_named_errors():
    """A switched-on unported component exits 1 with ComponentNotPortedError before any read; the default config's
    12-month fit exits 1 as a heavy job without --allow-heavy (no compute starts)."""
    r = _prc("fit", "--config", FM1, "--set", "modules.storm.feature=true")
    assert r.returncode == 1
    assert "fm1 refused: ComponentNotPortedError" in r.stderr and "modules.storm.feature" in r.stderr
    r = _prc("fit", "--config", FM1)
    assert r.returncode == 1 and "heavy job" in r.stderr


def test_there_is_no_base_flag_and_no_abbreviation():
    r = _prc("build-submission", "--config", FM1, "--out", "x", "--base", "submissions/v2.parquet")
    assert r.returncode == 2 and "unrecognized arguments: --base" in r.stderr
    r = _prc("validate", "--config", FM1, "--proto", "S")
    assert r.returncode == 2


def test_bad_override_is_refused():
    r = _prc("validate", "--config", FM1, "--protocol", "dry-run", "--set", "run.threads=0")
    assert r.returncode == 1 and "ConfigValueError: run.threads: 0 is outside [1, 64]" in r.stderr


def test_save_models_flag_defaults_off_and_reaches_the_api(monkeypatch):
    """--save-models is a plain flag (default False) that main() forwards to api.validate as save_models=; no real
    fit needed to prove the wiring. Mutation: drop save_models from the argparser or from the api.validate call ->
    captured["save_models"] stays the wrong value -> red."""
    from prc import __main__ as cli
    from prc import api as api_mod

    args = cli.parse_args(["validate", "--config", str(FM1), "--protocol", "S"])
    assert args.save_models is False
    args = cli.parse_args(["validate", "--config", str(FM1), "--protocol", "S", "--save-models"])
    assert args.save_models is True

    captured = {}

    def fake_validate(config, protocol, out_dir=None, allow_heavy=False, overrides=(), log=None, save_models=False):
        captured["save_models"] = save_models
        return {"protocol": protocol, "folds": {}}

    monkeypatch.setattr(api_mod, "validate", fake_validate)
    rc = cli.main(["validate", "--config", str(FM1), "--protocol", "S", "--save-models"])
    assert rc == 0 and captured["save_models"] is True
    rc = cli.main(["validate", "--config", str(FM1), "--protocol", "S"])
    assert rc == 0 and captured["save_models"] is False


def test_dry_run_exit_code_follows_the_report_not_a_hard_coded_zero(monkeypatch):
    """The help text says 1 = refused, but a dry-run report that names a refusal (training_months the REFUSED
    string, or plan_refused present) used to still exit 0 because the validate branch always `return 0`-ed. No
    real config or fit needed: api.validate is faked to return a canned report. Mutation: drop the
    api.dry_run_refused(rep) check from main() -> rc stays 0 for the refused report -> red."""
    from prc import __main__ as cli
    from prc import api as api_mod

    monkeypatch.setattr(api_mod, "validate",
                        lambda *a, **k: {"training_months": "REFUSED: no training file for 2025-01",
                                          "stages": ["a"], "paths": {}})
    rc = cli.main(["validate", "--config", str(FM1), "--protocol", "dry-run"])
    assert rc == 1

    monkeypatch.setattr(api_mod, "validate",
                        lambda *a, **k: {"training_months": {1: "training_2025-01-01_x.parquet"},
                                          "stages": ["a", "b"], "paths": {}})
    rc = cli.main(["validate", "--config", str(FM1), "--protocol", "dry-run"])
    assert rc == 0


def test_reuse_from_flag_defaults_to_none_and_reaches_the_api(monkeypatch):
    """`fit --reuse-from DIR` forwards DIR to api.fit as reuse_from= (None when absent); no real fit needed for the
    wiring. Mutation: drop reuse_from from the api.fit call in main() -> captured stays None -> red."""
    from prc import __main__ as cli
    from prc import api as api_mod

    assert cli.parse_args(["fit", "--config", str(FM1)]).reuse_from is None
    captured = {}

    class _Art:
        directory = "somewhere"

    def fake_fit(config, save=True, allow_heavy=False, overrides=(), log=None, reuse_from=None):
        captured["reuse_from"] = reuse_from
        return _Art()

    monkeypatch.setattr(api_mod, "fit", fake_fit)
    assert cli.main(["fit", "--config", str(FM1), "--reuse-from", "artifacts/fm1/donor"]) == 0
    assert captured["reuse_from"] == "artifacts/fm1/donor"
    assert cli.main(["fit", "--config", str(FM1)]) == 0 and captured["reuse_from"] is None


@pytest.mark.slow
def test_fit_predict_build_on_the_world(e2e_world, fm1_world_mod, tmp_path):
    world = e2e_world[0]
    raw = fm1_world_mod.config_raw(world)
    raw["run"]["artifacts_dir"] = str(tmp_path / "artifacts")
    cfg = tmp_path / "world.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    r = _prc("fit", "--config", cfg)
    assert r.returncode == 0, r.stderr[-3000:]
    art = pathlib.Path(r.stdout.strip().splitlines()[-1])
    assert (art / "manifest.json").exists()
    r = _prc("predict", "--config", cfg, "--rows", world / "raw" / "scored.parquet", "--out", tmp_path / "p.parquet")
    assert r.returncode == 0, r.stderr[-3000:]
    p = pd.read_parquet(tmp_path / "p.parquet")
    assert p.decided_by.notna().all() and (p.TAXITIME_SEC_mvt >= 1).all()
    r = _prc("predict", "--artifact", art, "--config", cfg, "--rows", world / "raw" / "scored.parquet",
             "--out", tmp_path / "p.parquet")
    assert r.returncode == 1 and "never overwritten" in r.stderr
    r = _prc("build-submission", "--config", cfg, "--out", tmp_path / "sub", "--version", 7)
    assert r.returncode == 0, r.stderr[-3000:]
    sub = pd.read_parquet(tmp_path / "sub" / "merry-quicksand_v7.parquet")
    tpl = pd.read_parquet(world / "raw" / "template.parquet")
    assert sub.MVT_ID_mvt.tolist() == tpl.MVT_ID_mvt.tolist()
    s = p.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    assert (sub.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt == s.reindex(sub.MVT_ID_mvt)).all()


def test_compare_command(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    n = 200
    y = rng.normal(900, 200, n)
    base = pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float), "y": y, "pred": y + rng.normal(0, 100, n),
                         "day": np.repeat([f"2025-01-{d:02d}" for d in range(1, 11)], 20), "month": 1,
                         "ap": np.where(np.arange(n) % 2, "EDDF", "LSZH")})
    base.to_parquet(tmp_path / "a.parquet")
    base.assign(pred=base.y).to_parquet(tmp_path / "b.parquet")
    r = _prc("compare", "--a", tmp_path / "a.parquet", "--b", tmp_path / "b.parquet", "--n-boot", 200,
             "--out", tmp_path / "c.json")
    assert r.returncode == 0, r.stderr
    rep = json.loads((tmp_path / "c.json").read_text())
    want = float(((base.y - base.pred) ** 2).sum()) / 344_841.0               # b is exact: its gain is a's whole SSE
    assert rep["n"] == n and rep["gain_bmse"] == pytest.approx(want, rel=1e-9)
    assert sum(rep["by_airport_bmse"].values()) == pytest.approx(want, rel=1e-9) and set(rep["by_airport_bmse"]) == \
        {"EDDF", "LSZH"}
    assert rep["ci95"][0] > 0 and rep["ci95"][0] < want < rep["ci95"][1]


def test_compare_and_blend_commands_take_several_files(tmp_path):
    import numpy as np
    rng = np.random.default_rng(1)
    ys = {k: rng.normal(900, 200, 100) for k in (1, 2)}              # one label set per month, shared by both sides
    for side, shift in (("a", 0.0), ("b", 0.4)):
        for k in (1, 2):
            n = 100
            y = ys[k]
            f = pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float) + 1000 * k, "y": y, "pred": y + 80,
                              "day": np.repeat([f"2025-0{k}-{d:02d}" for d in range(1, 11)], 10), "month": k,
                              "ap": np.where(np.arange(n) % 2, "EDDF", "LSZH"), "decided_by": "base_lightgbm"})
            f = f.assign(pred=f.pred - shift * (f.pred - f.y))
            f.to_parquet(tmp_path / f"{side}{k}.parquet")
    r = _prc("compare", "--a", tmp_path / "a1.parquet", tmp_path / "a2.parquet", "--b", tmp_path / "b1.parquet",
             tmp_path / "b2.parquet", "--n-boot", 200)
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    # a misses by 80 s on every row, b by 48 s (moved 40 % toward the label): each row gains 80^2 - 48^2 = 4,096
    assert rep["n"] == 200 and rep["gain_bmse"] == pytest.approx(200 * 4096 / 344_841.0, rel=1e-9)
    r = _prc("blend", "--a", tmp_path / "a1.parquet", "--b", tmp_path / "b1.parquet", "--w", 0.3, "--out",
             tmp_path / "ab.parquet")
    assert r.returncode == 0, r.stderr
    ab = pd.read_parquet(tmp_path / "ab.parquet").set_index("MVT_ID_mvt").sort_index()
    a1 = pd.read_parquet(tmp_path / "a1.parquet").set_index("MVT_ID_mvt").sort_index()
    b1 = pd.read_parquet(tmp_path / "b1.parquet").set_index("MVT_ID_mvt").sort_index()
    assert len(ab) == 100 and (ab.decided_by == "blend_oof").all()
    # w weights b (asymmetric w, so swapped weights fail; rehearsed RED, BL1): rint(0.3 (y + 48) + 0.7 (y + 80))
    assert (ab.pred == np.rint(0.3 * b1.pred + 0.7 * a1.pred)).all()
    assert (ab.pred - (ys[1] + 70.4)).abs().max() <= 0.5 + 1e-9


def test_overrides_with_an_explicit_artefact_are_refused_not_dropped(tmp_path):
    """Line-by-line audit 2026-10-04 (C4): `build-submission --artifact DIR --set k=v` silently dropped the --set (an
    artefact carries its own configuration), so the user believed a knob had changed. Now refused by name, at the API
    boundary, before anything is loaded. Seen red 2026-10-04 (an ArtifactError about the missing manifest, i.e. the
    override was ignored). Mutation rehearsal (scratch copy with the whole package): 2 / 2 killed by this test — the
    check removed; the check limited to ModelArtifact objects (a directory path slipped through)."""
    from prc import api
    with pytest.raises(api.errors.ConfigValueError, match=r"overrides do not apply to an artefact"):
        api.build_submission(tmp_path, tmp_path / "out", overrides=["modules.ldi.enabled=false"])
    r = _prc("build-submission", "--config", FM1, "--artifact", tmp_path, "--set", "modules.ldi.enabled=false",
             "--out", tmp_path / "o")
    assert r.returncode == 1
    assert "fm1 refused: ConfigValueError: overrides do not apply to an artefact" in r.stderr
