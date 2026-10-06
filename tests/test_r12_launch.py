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
    (dict(pmset=BATT, machine_guards=True), "not on AC power"),
    (dict(now=dt.datetime(2026, 9, 26, 11, 50), machine_guards=True), "track_record window"),
    (dict(lock_free=False), "heavy-compute lock"),
    (dict(others=True), "another R12 or v34 launcher"),
    (dict(fetch=True, final=True), "A12 fetch is still running"),
    (dict(fetch=True, step="val"), "A12 fetch is still running"),
])
def test_preflight_refusals(kw, reason):
    """Each refusal fires on its own. Fails when any single check is dropped. The AC-power and track_record-window
    cases now pass machine_guards=True explicitly (the test that asserted the old hard-coded refusal is
    replaced by a test of the new argument) -- --machine-guards defaults to off, so these two checks no
    longer fire without it."""
    args = dict(pmset=AC, now=NIGHT, lock_free=True, others=False, fetch=False, step="features", final=False)
    args.update(kw)
    assert any(reason in r for r in L.preflight(**args))


def test_preflight_machine_guards_off_by_default_skips_ac_power_and_window_checks():
    """Fails when `machine_guards` does not exist (TypeError) or defaults to True (BATT / the window would still
    refuse without passing --machine-guards)."""
    assert L.preflight(BATT, NIGHT, True, False, False, "features", False) == []
    assert L.preflight(AC, dt.datetime(2026, 9, 26, 11, 50), True, False, False, "features", False) == []


def test_preflight_machine_guards_on_reproduces_todays_refusals_and_messages():
    """machine_guards=True must reproduce exactly the refusals and messages the launcher gave before --machine-guards
    existed. Fails when the gated branch's messages differ from the pre-existing ones."""
    assert L.preflight(BATT, NIGHT, True, False, False, "features", False, machine_guards=True) == \
        ["not on AC power"]
    assert L.preflight(AC, dt.datetime(2026, 9, 26, 11, 50), True, False, False, "features", False,
                       machine_guards=True) == ["inside Phantom's track_record window 11:40-12:15"]


def test_preflight_smoke_may_run_beside_the_fetch():
    """A < 5 min smoke (scratch outputs, no lock) is not refused for the fetch; every other check still applies
    (machine_guards=True here to exercise the AC-power check, consistent with the refusal tests above)."""
    assert L.preflight(AC, NIGHT, True, False, True, "val", False, smoke=True, machine_guards=True) == []
    assert any("AC" in r for r in L.preflight(BATT, NIGHT, True, False, True, "val", False, smoke=True,
                                              machine_guards=True))


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


def test_detach_argv_forwards_machine_guards_when_given_and_omits_it_when_not():
    """Item 4b (FP-W3, finding 5): the detached re-launch's argv was built inline inside `main()` (process
    orchestration, pragma: no cover) and untested for --machine-guards forwarding specifically. Extracted to a
    pure function: --machine-guards must reach the child's argv exactly when the parent had it, and not
    otherwise; --smoke / --partial and the runner's own args (after --) still forward too. Mutation: drop the
    --machine-guards branch -> the first assertion goes red; always append it -> the second assertion goes red."""
    lp = pathlib.Path("/x/r12_features_run_20260101T000000.log")
    with_guards = L.detach_argv("features", True, True, True, ["--store", "x"], lp)
    assert with_guards == [L.PY, "-B", str(pathlib.Path(L.__file__).resolve()), "--step", "features", "--log",
                           str(lp), "--smoke", "--partial", "--machine-guards", "--", "--store", "x"]
    without_guards = L.detach_argv("features", False, False, False, ["--store", "x"], lp)
    assert without_guards == [L.PY, "-B", str(pathlib.Path(L.__file__).resolve()), "--step", "features", "--log",
                              str(lp), "--", "--store", "x"]
    assert "--machine-guards" not in without_guards


def test_machine_guards_defaults_off_and_is_settable():
    """Fails when --machine-guards is missing from the parser (AttributeError) or defaults to True (the launcher
    would then refuse off-AC-power / in-window runs even without the flag, breaking Linux / non-owner use)."""
    assert L.parse_args(["--step", "features", "--", "features", "--store", "x"]).machine_guards is False
    assert L.parse_args(["--step", "features", "--machine-guards", "--", "features", "--store",
                         "x"]).machine_guards is True


def test_gather_preflight_inputs_skips_pmset_when_machine_guards_is_off():
    """Fails when pmset is still shelled out to regardless of `machine_guards` (pmset is macOS-only; the launcher
    must not call it when machine_guards is off, e.g. on Linux)."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        class R:
            stdout = ""
        return R()
    pm, swap, ps = L._gather_preflight_inputs(False, run=fake_run)
    assert pm == "" and calls == ["sysctl", "ps"]


def test_gather_preflight_inputs_calls_pmset_when_machine_guards_is_on():
    """Fails when pmset is skipped even with machine_guards=True (the AC-power check would then always pass)."""
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[0])
        class R:
            stdout = "Now drawing from 'AC Power'\n"
        return R()
    pm, swap, ps = L._gather_preflight_inputs(True, run=fake_run)
    assert pm == "Now drawing from 'AC Power'\n" and calls == ["pmset", "sysctl", "ps"]


def test_attach_memguard_starts_only_under_machine_guards():
    """Item 6 (FP-W3, finding 7): `_run` started mem_guard (V.start_memguard) unconditionally, although
    tools/mem_guard.py shells out to the macOS-only `memory_pressure` (~line 282) -- the same machine assumption
    --machine-guards already gates for the AC-power / track_record checks. machine_guards=False must skip
    start_memguard entirely (not just ignore its result), so the launcher also runs on a machine without it (e.g.
    Linux). machine_guards=True must start it exactly as before. Mutation: drop the `if not machine_guards: return
    None` guard -> start is called even when machine_guards is False -> the first assertion goes red."""
    logs = []
    calls = []

    def fake_start(pid, lp):
        calls.append((pid, lp))
        return ["mem_guard", "--pid", str(pid)], object()
    lp = pathlib.Path("/x/r12_features_run_20260101T000000.log")
    assert L._attach_memguard(123, lp, False, logs.append, start=fake_start) is None
    assert calls == []
    assert any("skipped" in m and "machine-guards" in m for m in logs)
    logs.clear()
    g = L._attach_memguard(123, lp, True, logs.append, start=fake_start)
    assert calls == [(123, lp)]
    assert g == ["mem_guard", "--pid", "123"]
    assert any("mem_guard:" in m for m in logs)
