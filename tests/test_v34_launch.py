"""scripts/analysis/v34_launch.py — the pure parts of the v34 detached launcher: the per-step RSS limits (6/7 GiB default,
8/9 GiB for steps 3 and 4 only), the child and mem_guard commands, the process-tree RSS, preflight and the watchdog.
No process is launched and no lock is touched. Mutation rehearsal 2026-09-24: see docstrings."""
from __future__ import annotations

import datetime as dt
import importlib.util
import inspect
import os
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v34_launch", ROOT / "scripts/analysis/v34_launch.py")
VL = importlib.util.module_from_spec(_spec)
sys.modules["v34_launch"] = VL          # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(VL)

AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged;"
BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%; discharging;"
PY = sys.executable                 # the launcher runs its children with its own interpreter


@pytest.mark.parametrize("step,abort,kill", [(1, 6, 7), (2, 6, 7), (3, 8, 9), (4, 8, 9), (5, 6, 7), (6, 11, 12), (7, 6, 7)])
def test_rss_limits_per_step_are_the_registered_ones(step, abort, kill):
    """6 GiB self-abort / 7 GiB watchdog by default; 8 / 9 ONLY for steps 3, 4 and 6 (owner-approved exceptions: 3/4 on
    2026-09-24, step 6 CatBoost raised to 11 / 12 on 2026-09-25 ~12:30 EDT: the CBV record's true peak footprint is 11.3 GB = 10.5 GiB; its 7.03 was ru_maxrss). Rehearsed RED: giving step 5 the 8/9
    exception, leaving step 6 at 6/7 (seen RED 2026-09-25 before the launcher change), and passing the watchdog value as the self-abort."""
    s = VL.spec(step)
    assert (s.rss_abort_gib, s.watchdog_gib) == (abort, kill)
    cmd = VL.child_cmd(step, smoke=False)
    assert cmd[cmd.index("--rss-limit-gib") + 1] == f"{abort}"


def test_only_steps_3_4_and_6_carry_the_exception():
    assert sorted(n for n, s in VL.STEPS.items() if s.rss_abort_gib > 6 or s.watchdog_gib > 7) == [3, 4, 6]


def test_child_cmd_runs_each_steps_runner_with_its_args_and_optional_smoke():
    assert VL.child_cmd(1, False) == [PY, "-B", str(ROOT / "scripts/analysis/v34_f_fold.py"), "--rss-limit-gib", "6"]
    assert VL.child_cmd(3, False)[3:5] == ["--design", "F"]
    assert VL.child_cmd(4, False)[3:5] == ["--design", "FW"]
    assert VL.child_cmd(5, False)[3:5] == ["--design", "ORL"]
    assert VL.child_cmd(6, True)[-1] == "--smoke" and "--smoke" not in VL.child_cmd(6, False)
    for n, s in VL.STEPS.items():
        assert (ROOT / s.runner).exists(), f"step {n} runner {s.runner} missing"
    with pytest.raises(ValueError, match="no v34 step"):
        VL.child_cmd(8, False)


def test_memguard_cmd_carries_the_child_pid_and_the_run_log_stem():
    """Owner requirement: tools/mem_guard.py --pid <the fit child's pid> --log <run log stem>_memguard.log, and
    --pause-on-phantom on every launched heavy run (owner ruling 2026-09-28: "when phantom starts just pause").
    Rehearsed RED: passing the launcher's own pid (os.getpid()) instead of the
    child's, a fixed log name, and (2026-09-29, seen red before the fix) the guard built without --pause-on-phantom."""
    log = ROOT / "logs/nsc_v34_scratch/v34_step3_run_20260925T010203.log"
    cmd = VL.memguard_cmd(4242, log)
    assert cmd == [PY, "-B", str(ROOT / "tools/mem_guard.py"), "--pid", "4242", "--log",
                   str(ROOT / "logs/nsc_v34_scratch/v34_step3_run_20260925T010203_memguard.log"), "--pause-on-phantom"]
    assert (ROOT / "tools/mem_guard.py").exists()
    for bad in (0, 1, -5, "4242"):
        with pytest.raises(ValueError):
            VL.memguard_cmd(bad, log)


def test_run_spawns_mem_guard_with_the_child_pid_right_after_the_child_starts():
    """Static check of the orchestration (not unit-runnable): the guard is built from `p.pid` of the runner Popen."""
    src = inspect.getsource(VL._run)
    i_child = src.index("p = subprocess.Popen(cmd")
    i_guard = src.index("start_memguard(p.pid, log_path)")
    i_watch = src.index("watch(p, rss_of")
    assert i_child < i_guard < i_watch


def test_the_guard_is_started_detached_with_its_own_output_kept_in_its_log(tmp_path, monkeypatch):
    """BC-43 sibling (2026-10-04): the launchers started mem_guard with stdout / stderr sent to /dev/null, so a guard that
    refused its target or crashed left no trace and the run went on unguarded with nothing on record. Its output now goes
    to its own log. Rehearsed RED: output back to DEVNULL; start_new_session dropped."""
    run_log = tmp_path / "r12_x.log"
    monkeypatch.setattr(VL, "memguard_cmd", lambda pid, log: [sys.executable, "-c",
                        "import os, sys; sys.stderr.write(f'refused: no target; session leader {os.getsid(0) == os.getpid()}')"
                        "; sys.exit(2)"])
    g, proc = VL.start_memguard(4242, run_log)
    assert proc.wait(timeout=30) == 2
    assert VL.memguard_log(run_log).read_text() == "refused: no target; session leader True"


def test_a_replacement_guard_is_started_with_restart(tmp_path):
    """Rehearsed RED (RS1): the restart flag dropped from start_memguard."""
    calls = []
    g, _ = VL.start_memguard(4242, tmp_path / "r.log", restart=True, popen=lambda cmd, **kw: calls.append(cmd))
    assert g[-1] == "--restart" and calls == [g] and g[:-1] == VL.memguard_cmd(4242, tmp_path / "r.log")
    g, _ = VL.start_memguard(4242, tmp_path / "r.log", popen=lambda cmd, **kw: calls.append(cmd))
    assert "--restart" not in g


def test_there_is_no_wall_time_kill_so_a_paused_run_is_never_killed_for_time():
    """mem_guard may SIGSTOP the tree for a long time; the only kill is RSS. Rehearsed RED: adding an elapsed-time kill."""
    src = inspect.getsource(VL.watch) + inspect.getsource(VL._run)
    assert src.count(".kill()") == 1 and "WATCHDOG KILL: tree RSS" in src
    assert "timeout" not in src


def test_ps_tree_rss_sums_the_runner_and_every_descendant_only():
    """The CatBoost / NN workers are grandchildren: their RSS must count. Rehearsed RED: root-only RSS; all-process sum."""
    ps = "  100     1  1048576\n  200   100  2097152\n  300   200  1048576\n  400     1  9999999\n garbage line\n"
    t = VL.parse_ps(ps)
    assert t == {100: (1, 1048576), 200: (100, 2097152), 300: (200, 1048576), 400: (1, 9999999)}
    assert VL.tree_rss_gib(100, t) == 4.0
    assert VL.tree_rss_gib(300, t) == 1.0
    assert VL.tree_rss_gib(999, t) == 0.0


def test_phantom_window_is_inclusive_11_40_to_12_15():
    d = dt.datetime(2026, 9, 25)
    assert VL.in_phantom_window(d.replace(hour=11, minute=40)) and VL.in_phantom_window(d.replace(hour=12, minute=15))
    assert not VL.in_phantom_window(d.replace(hour=11, minute=39)) and not VL.in_phantom_window(d.replace(hour=12, minute=16))


def test_preflight_lists_every_reason_and_is_empty_when_clear():
    r = VL.preflight(BATT, dt.datetime(2026, 9, 25, 12, 0), lock_free=False, other_alive=True)
    assert r == ["not on AC power", "inside Phantom's track_record window 11:40-12:15",
                 "heavy-compute lock is held by another job", "another v34 launcher is alive"]
    assert VL.preflight(AC, dt.datetime(2026, 9, 25, 20, 0), lock_free=True, other_alive=False) == []
    assert not VL.on_ac_power("Now drawing from 'Battery Power' (AC adapter disconnected)")


def test_one_heavy_job_at_a_time_across_all_seven_steps(tmp_path):
    pfs = [VL.pidfile(n) for n in VL.STEPS]
    assert len(set(pfs)) == 7
    assert VL.others_alive(pfs, lambda p: p == pfs[4]) and not VL.others_alive(pfs, lambda p: False)


class _Proc:
    def __init__(self, polls_before_exit, rc=0):
        self.pid, self.n, self.rc, self.killed = 4242, polls_before_exit, rc, False

    def poll(self):
        if self.killed:
            return -9
        self.n -= 1
        return None if self.n >= 0 else self.rc

    def kill(self):
        self.killed = True

    def wait(self):
        return -9 if self.killed else self.rc


def test_watch_kills_only_above_the_limit_and_reports_the_peak():
    """Rehearsed RED: dropping the kill or the break."""
    lines = []
    rss = iter([1.0, 8.9, 9.5, 1.0])
    rc, peak, killed = VL.watch(_Proc(10), lambda pid: next(rss), lines.append, kill_gib=9.0, sleep=lambda s: None)
    assert killed and rc == -9 and peak == 9.5 and lines == ["WATCHDOG KILL: tree RSS 9.50 GiB > 9.0"]
    rc, peak, killed = VL.watch(_Proc(3), lambda pid: 2.0, lines.append, kill_gib=7.0, sleep=lambda s: None)
    assert (rc, peak, killed) == (0, 2.0, False)


def test_parse_args_requires_a_known_step():
    assert VL.parse_args(["--step", "4", "--detach"]).step == 4
    with pytest.raises(SystemExit):
        VL.parse_args(["--step", "9"])


def test_memguard_log_sits_beside_the_run_log():
    assert VL.memguard_log("/a/b/run_1.log") == pathlib.Path("/a/b/run_1_memguard.log")


def test_pid_alive_reads_a_pid_file_and_counts_a_process_it_may_not_signal_as_alive(tmp_path):
    """A missing or unreadable PID file is not alive; our own process is; PID 1 exists but belongs to root, so signalling
    it raises PermissionError — the process is alive (the old helper said False, so a launcher could miss a live run
    of another user). Mutation: PermissionError -> False -> red."""
    pf = tmp_path / "x.pid"
    assert VL.pid_alive(pf) is False
    pf.write_text("not a pid")
    assert VL.pid_alive(pf) is False
    pf.write_text(str(os.getpid()))
    assert VL.pid_alive(pf) is True
    pf.write_text("1")
    assert VL.pid_alive(pf) is True
