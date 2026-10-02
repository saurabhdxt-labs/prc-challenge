"""FM1 loads from what the repository ships, not from the development record (bug class BC-38, reports/bug_classes.md).

A clean clone holds the package, its config and the code it loads (prc/, configs/, scripts/ and the one rules file under
data/), but none of plans/, logs/, reports/, artifacts/, submissions/ or the large data files. On 2026-10-02 importing
`prc.pipeline.fm1.knobs` from such a tree failed: the import-time schema check required every knob's evidence FILE to
exist, and 29 of the 39 named files are development records that are never committed. These tests rebuild that tree in a
temporary directory and load FM1 from it in a child process (the parent's imports would mask the failure).

Mutation rehearsal per test in its docstring; results in tests/fm1/MUTATION_FM1.md (M-CCO rows).
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from prc.pipeline.fm1 import knobs as K

ROOT = pathlib.Path(__file__).resolve().parents[2]
#: what a clone of the repository carries that FM1 reads: the package, the config, the code it loads by path
SHIPPED_CODE = ("prc", "scripts")
SHIPPED_FILES = ("configs/fm1.yaml", "data/rome_hunt_2026_09_23/agent_E_rules.py")


def _code_files(d: str) -> list:
    """prc/: every module (a new one ships in the same commit as the code that imports it); scripts/: only what git tracks,
    so a loader that needs an uncommitted script fails here instead of on a clone."""
    if d == "prc":
        return [p for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts]
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--", f"{d}/*.py"], capture_output=True, text=True,
                         check=True).stdout.split()
    assert out, f"git lists no tracked {d}/*.py"
    return [ROOT / p for p in out]


def _shipped_tree(dest: pathlib.Path) -> pathlib.Path:
    for d in SHIPPED_CODE:
        for src in _code_files(d):
            out = dest / src.relative_to(ROOT)
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out)
    for f in SHIPPED_FILES:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dest / f)
    for absent in ("plans", "logs", "reports", "artifacts", "submissions"):
        assert not (dest / absent).exists()
    return dest


def _env(tree: pathlib.Path) -> dict:
    """Only the repository is swapped for the shipped tree; the user's installed environment stays as it is (HOME holds
    the CatBoost venv FM1's worker runs in, REPRODUCE.md; PATH holds the system tools libraries call, e.g. sysctl)."""
    return {"PATH": os.environ["PATH"], "PYTHONPATH": str(tree), "HOME": os.environ["HOME"]}


def _run(tree: pathlib.Path, code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", "-c", code], cwd=tree, capture_output=True, text=True, timeout=300,
                          env=_env(tree))


@pytest.mark.slow
def test_fm1_imports_and_parses_its_config_from_the_shipped_tree_alone(tmp_path):
    """Import the public API and every FM1 module, parse configs/fm1.yaml: the resolved hash must equal the development
    tree's (the evidence pointers are documentation; they never change a value). Mutation: put the evidence-file
    existence check back into the import-time schema check -> ConfigError in the child -> red."""
    tree = _shipped_tree(tmp_path / "clone")
    code = ("import json, prc.api, prc.pipeline.fm1.engine, prc.pipeline.fm1.eobt_clip;"
            "from prc.pipeline.fm1 import knobs as K;"
            "print(json.dumps({'hash': K.load_config('configs/fm1.yaml').hash, 'root': str(K.ROOT)}))")
    r = _run(tree, code)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["hash"] == K.load_config(ROOT / "configs" / "fm1.yaml").hash
    assert pathlib.Path(got["root"]).resolve() == tree.resolve()          # it really ran from the clone


@pytest.mark.slow
def test_the_cli_starts_from_the_shipped_tree_alone(tmp_path):
    """`python -m prc config-reference --write` runs in the clone and generates the same docs/fm1/CONFIG_REFERENCE.md as
    the development tree. Mutation: as above -> the command fails in the child -> red."""
    tree = _shipped_tree(tmp_path / "clone")
    r = subprocess.run([sys.executable, "-B", "-m", "prc", "config-reference", "--write"], cwd=tree, capture_output=True,
                       text=True, timeout=300,
                       env=_env(tree))
    assert r.returncode == 0, r.stderr[-2000:]
    made = tree / "docs" / "fm1" / "CONFIG_REFERENCE.md"
    assert made.read_text() == (ROOT / "docs" / "fm1" / "CONFIG_REFERENCE.md").read_text()


@pytest.mark.slow
def test_every_code_loader_fm1_uses_runs_from_the_shipped_tree_alone(tmp_path):
    """Sibling of BC-38 (category b): the scripts FM1 loads by path (shipped.* and legacy.*: stand_ab, unm_congestion,
    orl_build, mtx_*, ldi_correction, urp_replay and its dependencies, rome_local_day, ...) read no development-record
    file when loaded. Protecting invariant: every one of them is tracked code and none reads a file at import (measured
    2026-10-02: 39 repo files loaded, all tracked but the new eobt_clip.py). Mutation: make a loader read
    `plans/PREREG_fm1_2026_09_28.md` at import -> FileNotFoundError in the child -> red."""
    tree = _shipped_tree(tmp_path / "clone")
    code = ("import inspect;"
            "from prc.pipeline.fm1 import shipped as S; from prc.pipeline import legacy as L;"
            "n = 0\n"
            "for mod in (S, L):\n"
            "    for name, fn in inspect.getmembers(mod, inspect.isfunction):\n"
            "        if fn.__module__ == mod.__name__ and not name.startswith('_') and not inspect.signature(fn).parameters:\n"
            "            fn(); n += 1\n"
            "print(n)")
    r = _run(tree, code)
    assert r.returncode == 0, r.stderr[-2000:]
    assert int(r.stdout.strip().splitlines()[-1]) >= 10           # the loaders really ran (13 on 2026-10-02)


@pytest.mark.slow
def test_fm1_fits_and_predicts_from_the_shipped_tree_like_the_development_tree(e2e_world, fm1_world_mod, tmp_path):
    """The whole FM1 path — fit, then predict through every stage including the ADS-B stack, whose MLP half runs as a
    subprocess by path (scripts/adsb_nn_worker.py) — runs from the shipped tree and predicts exactly what the development
    tree predicts on the real-data-shaped world. Found 2026-10-02: the worker was never committed, so a clone failed at
    the stack stage while every import-level test passed. Mutation: leave a by-path worker script out of the shipped tree
    -> the child's predict fails -> red."""
    from prc import api
    world = e2e_world[0]
    raw = fm1_world_mod.config_raw(world, **{"run.artifacts_dir": str(tmp_path / "art")})
    quiet = lambda *a, **k: None  # noqa: E731
    ref = api.predict(api.fit(K.parse_config(raw), save=False, log=quiet),
                      pd.read_parquet(world / "raw" / "scored.parquet"), log=quiet)
    tree = _shipped_tree(tmp_path / "clone")
    out = tmp_path / "clone_pred.parquet"
    code = ("import json, sys, pandas as pd; from prc import api; from prc.pipeline.fm1 import knobs as K;"
            "raw, scored, out = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3];"
            "q = lambda *a, **k: None;"
            "art = api.fit(K.parse_config(raw), save=False, log=q);"
            "api.predict(art, pd.read_parquet(scored), log=q)[['MVT_ID_mvt', 'TAXITIME_SEC_mvt', 'decided_by']]"
            ".to_parquet(out, index=False)")
    r = subprocess.run([sys.executable, "-B", "-c", code, json.dumps(raw), str(world / "raw" / "scored.parquet"), str(out)],
                       cwd=tree, capture_output=True, text=True, timeout=900,
                       env=_env(tree))
    assert r.returncode == 0, r.stderr[-3000:]
    got = pd.read_parquet(out)
    assert "adsb_stack" in set(got.decided_by)                     # the stack (and so the worker) really served rows
    pd.testing.assert_frame_equal(got.reset_index(drop=True),
                                  ref[["MVT_ID_mvt", "TAXITIME_SEC_mvt", "decided_by"]].reset_index(drop=True),
                                  check_dtype=False)


#: the scripts that rebuild FM1's derived inputs (docs/fm1/REPRODUCE.md) and the launcher the queue tool calls
PRODUCERS = ("r12_build", "r12_validate", "r12_stack", "r13_cat", "orl_build", "mtx_source", "r12_launch")


@pytest.mark.slow
def test_the_scripts_that_rebuild_fms_inputs_load_from_the_shipped_tree_alone(tmp_path):
    """Every script REPRODUCE.md names for a derived input (ADS-B tables, holdout fits, stack models, ORL table, MTX CSVs)
    and the launcher tools/prc_queue.py calls is git-tracked and loads, with everything it loads by path, in the shipped
    tree. Found 2026-10-02: the stack's producers (r12_stack, r12_validate, r13_cat + its CatBoost worker) were never
    committed. Mutation: leave one by-path dependency out of the shipped tree -> red."""
    tree = _shipped_tree(tmp_path / "clone")
    missing = [p for p in PRODUCERS if not (tree / "scripts" / "analysis" / f"{p}.py").exists()]
    assert not missing, f"not tracked by git: {missing}"
    workers = ("scripts/adsb_nn_worker.py", "scripts/analysis/r13_cat_worker.py")
    assert all((tree / w).exists() for w in workers), [w for w in workers if not (tree / w).exists()]
    code = ("import importlib.util, sys\n"
            f"for name in {PRODUCERS!r}:\n"
            "    spec = importlib.util.spec_from_file_location(name, f'scripts/analysis/{name}.py')\n"
            "    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)\n"
            "print('loaded', len(sys.modules))")
    r = _run(tree, code)
    assert r.returncode == 0, r.stderr[-2000:]
