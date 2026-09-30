"""Tests for scripts/analysis/r12_launch.py — the detached launcher for the R12 heavy steps (pure parts; process
orchestration is exercised by the smoke launches). Mutation rehearsals are recorded in the docstrings."""
from __future__ import annotations

import datetime as dt
import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("r12_launch", ROOT / "scripts" / "analysis" / "r12_launch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["r12_launch"] = mod
    spec.loader.exec_module(mod)
    return mod


L = _load()
AC = "Now drawing from 'AC Power'\n -InternalBattery-0 100%; charged"
BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0 80%; discharging"
NIGHT = dt.datetime(2026, 9, 26, 7, 0)


def test_steps_and_limits():
    """Every step names its runner and a watchdog above its self-abort."""
    assert set(L.STEPS) == {"features", "tables", "val", "stack"}
    for s in L.STEPS.values():
        assert (ROOT / s.runner).exists() and s.watchdog_gib > s.rss_abort_gib > 0


def test_child_cmd_injects_the_rss_limit_and_smoke_only_for_val():
    """Fails when the launcher's limit is not passed to the runner (the in-process abort would be the default)."""
    c = L.child_cmd("val", ["--split", "holdout"], smoke=True)
    assert c[:3] == [L.PY, "-B", str(ROOT / L.STEPS["val"].runner)]
    assert c[-3:] == ["--rss-limit-gib", f"{L.STEPS['val'].rss_abort_gib:g}", "--smoke"]
    f = L.child_cmd("features", ["features", "--store", "x"], smoke=True)
    assert "--smoke" not in f and f[-2:] == ["--rss-limit-gib", f"{L.STEPS['features'].rss_abort_gib:g}"]
    with pytest.raises(ValueError, match="rss-limit-gib"):
        L.child_cmd("val", ["--rss-limit-gib", "99"], smoke=False)
    with pytest.raises(ValueError, match="no R12 step"):
        L.child_cmd("train", [], smoke=False)


def test_fetch_alive_parses_ps():
    ps = "  101 /usr/bin/python -B scripts/adsb_fetch.py run --months 2,3\n  102 /bin/zsh -c sleep 1\n"
    assert L.fetch_alive(ps) is True
    assert L.fetch_alive("  102 /bin/zsh -c sleep 1\n") is False


def test_fetch_alive_ignores_watchers_that_grep_for_it():
    """A monitoring shell whose command line quotes the pattern is not the fetch (pgrep -f's classic false positive)."""
    ps = '  7 /bin/zsh -c while pgrep -f "adsb_fetch.py run --months" >/dev/null; do sleep 60; done\n'
    assert L.fetch_alive(ps) is False
    ps = "  8 /bin/zsh -c ps -A | grep scripts/adsb_fetch.py run\n"            # quotes the full path (mutant #36)
    assert L.fetch_alive(ps) is False


@pytest.mark.parametrize("kw,reason", [
    (dict(pmset=BATT), "not on AC power"),
    (dict(now=dt.datetime(2026, 9, 26, 11, 50)), "track_record window"),
    (dict(lock_free=False), "heavy-compute lock"),
    (dict(others=True), "another R12 or v34 launcher"),
    (dict(fetch=True, final=True), "A12 fetch is still running"),
    (dict(fetch=True, step="val"), "A12 fetch is still running"),
])
def test_preflight_refusals(kw, reason):
    """Each refusal fires on its own. Fails when any single check is dropped."""
    args = dict(pmset=AC, now=NIGHT, lock_free=True, others=False, fetch=False, step="features", final=False)
    args.update(kw)
    assert any(reason in r for r in L.preflight(**args))


def test_preflight_smoke_may_run_beside_the_fetch():
    """A < 5 min smoke (scratch outputs, no lock) is not refused for the fetch; every other check still applies."""
    assert L.preflight(AC, NIGHT, True, False, True, "val", False, smoke=True) == []
    assert any("AC" in r for r in L.preflight(BATT, NIGHT, True, False, True, "val", False, smoke=True))


def test_preflight_passes_and_allows_a_resumable_build_during_the_fetch():
    """A non-final feature build defers the days the fetch has not finished, so it may run beside the fetch."""
    assert L.preflight(AC, NIGHT, True, False, False, "val", False) == []
    assert L.preflight(AC, NIGHT, True, False, True, "features", False) == []


def test_output_paths_must_match_the_mode():
    """Smoke writes only under logs/r12_scratch; a real run only under data/r12. Fails when a smoke could write into
    data/r12 (a smoke store mistaken for the real one) or a real run into the scratch."""
    L.check_paths(["features", "--store", "logs/r12_scratch/s/store", "--days", "2025-03-10"], smoke=True)
    L.check_paths(["--split", "holdout", "--tables", "data/r12/t", "--out", "data/r12/v"], smoke=False)
    with pytest.raises(PermissionError, match="smoke"):
        L.check_paths(["features", "--store", "data/r12/features_x"], smoke=True)
    with pytest.raises(PermissionError, match="real"):
        L.check_paths(["--out", "logs/r12_scratch/v"], smoke=False)
    with pytest.raises(ValueError, match="no output path"):
        L.check_paths(["--split", "holdout"], smoke=False)


def test_stack_step_inputs_follow_the_mode():
    """The stack reads a val / stack / refit dir: a real stack run may not read smoke outputs (and vice versa)."""
    L.check_paths(["oof", "--val", "data/r12/v", "--tables", "data/r12/t", "--out", "data/r12/s"], smoke=False)
    with pytest.raises(PermissionError, match="real"):
        L.check_paths(["oof", "--val", "logs/r12_scratch/v", "--tables", "data/r12/t", "--out", "data/r12/s"], smoke=False)
    with pytest.raises(PermissionError, match="real"):
        L.check_paths(["serve", "--f26", "logs/r12_scratch/x.parquet", "--out", "data/r12/s"], smoke=False)
    assert L.child_cmd("stack", ["oof"], smoke=True)[-1] == "--smoke"


def test_real_features_run_must_be_final_or_explicitly_partial():
    """The owner-acked full build is --final (every missing day counted); a partial real build is refused unless
    --partial is given to the launcher, so a half-built store is never mistaken for the full one."""
    with pytest.raises(ValueError, match="--final"):
        L.check_features_mode(["features", "--store", "data/r12/f"], smoke=False, partial=False)
    L.check_features_mode(["features", "--store", "data/r12/f", "--final"], smoke=False, partial=False)
    L.check_features_mode(["features", "--store", "data/r12/f"], smoke=False, partial=True)
    L.check_features_mode(["features", "--store", "logs/r12_scratch/f", "--days", "2025-01-01"], smoke=True,
                          partial=False)


def test_pidfiles_cover_r12_and_v34():
    names = {p.name for p in L.all_pidfiles()}
    assert {"launch_r12_features.pid", "launch_r12_tables.pid", "launch_r12_val.pid", "launch_r12_stack.pid"} <= names
    assert any(n.startswith("launch_v34_step") for n in names)


def test_log_path_names_step_and_mode():
    p = L.log_path("val", smoke=True, now=dt.datetime(2026, 9, 26, 7, 0, 5))
    assert p == ROOT / "logs" / "r12_scratch" / "r12_val_smoke_20260926T070005.log"
