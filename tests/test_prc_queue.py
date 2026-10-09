"""tools/prc_queue.py — the campaign queue: registered steps run one after another; between heavy steps the heavy lock is
left free until Phantom's batch runner reports "nothing due" (capped); a PAUSE file holds the queue at the next boundary;
state survives a restart; the first failed step or acceptance check stops the queue. No process is launched and no lock
is touched here: the loop runs on injected fakes.

Seen red first (2026-09-29 22:3x EDT: collection error, module absent). Mutation rehearsal 2026-09-29 (scratch runner,
each mutant applied to tools/prc_queue.py, this file run with -x, source restored after): 22 of 22 RED — stale >= fresh,
idle->busy, unknown->idle, cap > instead of >=, light-step early return off, lock / power / window checks off,
interrupted not stopping, check rc 1 accepted, pause time counted toward the cap, rc 75 treated as failure, check run
after a failed step, recover off, `$(` not refused, duplicate ids allowed, first END instead of last (killed only after
the two-END case was added), WAIT logged every poll, running state not saved before execute, queue read once, refusal
count reset, PAUSE ignored for heavy steps.
Groups (2026-09-29 ~23:05): seen red (4 failed), then 6 of 6 mutants RED (group never blocked, blocked ignored, failure
reported as all done, last stop reason instead of first — killed after the file-order test was added, validation off,
group dropped); the 22 above and wait_rc's 2 re-run RED.
Guard (2026-09-29 ~23:45): seen red (2 failed), then 4 of 4 mutants RED (type check off, kind check off, guard
dropped, a hand-built mem_guard command without --pause-on-phantom).
Guard supervision (2026-10-04, BC-43 / tooling audit T4): 5 tests seen red (no supervise_guarded), then 5 of 5 mutants
RED on a scratch copy — PQ1 a dead guard at the start check ignored, PQ2 the step's own exit code lost to the guard
check (a lock refusal would stop retrying), PQ3 no release when the guard dies, PQ4 no restart, PQ5 unlimited restarts.
BC-35 lock race (2026-09-30 00:29): seen red (7 failed), then 4 of 4 RED (lock message always 75, ignored, machine_state
inverted, lock ignored). Reviewer MAJOR 1/2 (01:1x): seen red (5 failed), then 12 of 12 RED (stale-line > not >=, check
off, stale/busy swapped, deferral >=, deferred group not held, deferral ignored, heavy clock not reset — killed after
the timed heavy-step test —, refusal cap off, no retry_at, now not passed, blocked ignored, group dropped)."""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

import pytest

from isolation import Deadline

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prc_queue", ROOT / "tools/prc_queue.py")
Q = importlib.util.module_from_spec(_spec)
sys.modules["prc_queue"] = Q            # dataclasses resolve annotations through sys.modules
_spec.loader.exec_module(Q)

AC = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged;"
BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t80%; discharging;"
PY = "/opt/homebrew/bin/python3.11"
NIGHT = dt.datetime(2026, 9, 30, 1, 0)


def _q(*steps):
    return {"steps": list(steps)}


def _light(i, check=None):
    d = {"id": i, "kind": "light", "argv": [PY, "-B", "x.py", i]}
    if check is not None:
        d["check"] = check
    return d


def _heavy(i):
    return {"id": i, "kind": "lock", "argv": [PY, "-B", "x.py", i]}


# ------------------------------------------------------------------ queue file

def test_load_queue_keeps_order_kinds_argv_and_checks():
    steps = Q.load_queue(_q(_light("a", check=[PY, "c.py"]), _heavy("b"),
                            {"id": "c", "kind": "r12", "step": "stack", "argv": ["serve", "--arm", "lgbcat"]}))
    assert [s.id for s in steps] == ["a", "b", "c"]
    assert [s.kind for s in steps] == ["light", "lock", "r12"]
    assert steps[0].check == (PY, "c.py") and steps[1].check is None
    assert steps[2].r12_step == "stack" and steps[2].argv == ("serve", "--arm", "lgbcat")


@pytest.mark.parametrize("bad,msg", [
    (_q(_light("a"), _light("a")), "duplicate step id 'a'"),
    (_q({"id": "a", "kind": "gpu", "argv": ["x"]}), "unknown kind 'gpu'"),
    (_q({"id": "a", "kind": "r12", "argv": ["serve"]}), "r12 step 'a' needs 'step'"),
    (_q({"id": "a", "kind": "light", "argv": []}), "step 'a' has an empty argv"),
    (_q({"id": "a", "kind": "light", "argv": [PY, "$OUT"]}), "step 'a' argv has a shell expansion: '$OUT'"),
    (_q({"id": "a", "kind": "light", "argv": [PY, "x"], "check": [PY, "--out=$(date)"]}),
     "step 'a' check has a shell expansion: '--out=$(date)'"),
    (_q({"id": "", "kind": "light", "argv": [PY]}), "every step needs a non-empty string id"),
    (_q({"id": "a", "kind": "light", "argv": [PY, 3]}), "step 'a' argv must be a list of strings"),
    ({"steps": "a"}, "queue needs a 'steps' list"),
    (_q({"id": "a", "kind": "light", "argv": [PY], "extra": 1}), "step 'a' has unknown keys ['extra']"),
])
def test_load_queue_refuses_bad_files_loudly(bad, msg):
    """zsh does not word-split and argv is never shell-expanded: a `$VAR` in a
    queue file would reach the runner literally, so it is refused at load, not discovered in a log hours later."""
    with pytest.raises(ValueError) as e:
        Q.load_queue(bad)
    assert str(e.value) == msg


# ------------------------------------------------------------------ Phantom's runner

@pytest.mark.parametrize("line,state", [
    ("batch runner: no job started — nothing due (load1=2.1 mem_free=20GB lock=False)", "idle"),
    ("batch runner: starting com.phantom.r17_shadow (P1, peak 7.0GB, load1=4.37, mem_free=19.2GB)", "busy"),
    ("batch runner: com.phantom.r17_shadow kickstart rc=0 in 0.0s", "busy"),
    ("batch runner: no job started — managed job com.phantom.system_memory is still running (serial: one at a time) ",
     "busy"),
    ("batch runner: no job started — reserved for com.phantom.notam_oracle (overdue, waiting for capacity): memory",
     "busy"),
    ("batch runner: no job started — 1 job(s) due, none admissible right now (load1=9)", "busy"),
    ("batch runner: no job started — machine-wide heavy lock is held by the realtime lane (load1=4)", "lock"),
    ("something new the runner never printed before", "busy"),
])
def test_phantom_state_reads_the_runners_last_line(line, state):
    """Unknown text counts as busy: waiting costs at most the cap, stacking onto a Phantom job costs a reboot (09-27)."""
    assert Q.phantom_state(line, age_s=120) == state


def test_phantom_state_is_stale_when_the_runner_has_not_ticked():
    """The runner ticks every 300 s; a log untouched for > 600 s means nothing to yield to."""
    assert Q.phantom_state("batch runner: starting com.phantom.x", age_s=601) == "stale"
    assert Q.phantom_state("batch runner: starting com.phantom.x", age_s=600) == "busy"
    assert Q.phantom_state(None, age_s=None) == "stale"


# ------------------------------------------------------------------ gate

def _gate(step, **kw):
    base = dict(now=NIGHT, pmset=AC, paused=False, phantom="idle", phantom_waited_s=0.0)
    base.update(kw)
    return Q.gate(step, **base)


H = Q.Step(id="h", kind="lock", argv=(PY,), check=None, r12_step=None)
L = Q.Step(id="l", kind="light", argv=(PY,), check=None, r12_step=None)


def test_gate_goes_when_everything_is_clear():
    assert _gate(H) == (True, "go", "go")


@pytest.mark.parametrize("kw,kind", [
    (dict(paused=True), "pause"),
    (dict(pmset=BATT), "power"),
    (dict(now=dt.datetime(2026, 9, 30, 11, 40)), "window"),
    (dict(now=dt.datetime(2026, 9, 30, 12, 15)), "window"),
    (dict(phantom="busy"), "phantom"),
    (dict(phantom="lock"), "lock"),
])
def test_gate_holds_a_heavy_step(kw, kind):
    go, _, k = _gate(H, **kw)
    assert (go, k) == (False, kind)


def test_gate_window_edges_are_the_launchers():
    assert _gate(H, now=dt.datetime(2026, 9, 30, 11, 39, 59))[0] is True
    assert _gate(H, now=dt.datetime(2026, 9, 30, 12, 15, 1))[0] is True


def test_gate_yield_to_phantom_is_capped_but_lock_is_never_stacked():
    """After YIELD_CAP_S of waiting on a busy runner (e.g. notam_oracle reserved for memory it never gets) the step goes;
    a held lock is someone's live heavy job and is never overridden, however long."""
    assert _gate(H, phantom="busy", phantom_waited_s=Q.YIELD_CAP_S - 1)[0] is False
    go, reason, kind = _gate(H, phantom="busy", phantom_waited_s=Q.YIELD_CAP_S)
    assert (go, kind) == (True, "go") and "yield cap" in reason
    assert _gate(H, phantom="lock", phantom_waited_s=10 * Q.YIELD_CAP_S)[0] is False


def test_gate_stale_runner_does_not_hold_the_queue():
    assert _gate(H, phantom="stale")[:1] == (True,)


def test_gate_light_steps_only_honour_pause():
    assert _gate(L, pmset=BATT, phantom="busy", now=dt.datetime(2026, 9, 30, 11, 50))[0] is True
    assert _gate(L, paused=True) == (False, "paused (PAUSE file present)", "pause")


# ------------------------------------------------------------------ state machine

def test_next_step_skips_done_and_stops_on_failed_or_interrupted():
    steps = Q.load_queue(_q(_light("a"), _light("b"), _light("c")))
    assert Q.next_step(steps, {})[0].id == "a"
    assert Q.next_step(steps, {"a": {"status": "done"}})[0].id == "b"
    s, why = Q.next_step(steps, {"a": {"status": "done"}, "b": {"status": "failed", "reason": "rc 3"}})
    assert s is None and why == "stopped: b failed (rc 3)"
    s, why = Q.next_step(steps, {"a": {"status": "interrupted", "reason": "was running at start"}})
    assert s is None and why == "stopped: a interrupted (was running at start)"
    assert Q.next_step(steps, {i: {"status": "done"} for i in "abc"}) == (None, "all done")


def test_recover_marks_a_running_step_interrupted_and_retry_clears_it():
    """A step left 'running' means the driver died mid-step (reboot, kill): its outputs may be partial, so it is never
    re-run silently."""
    st = Q.recover({"a": {"status": "done"}, "b": {"status": "running", "start": 1.0}})
    assert st["a"] == {"status": "done"}
    assert st["b"]["status"] == "interrupted" and st["b"]["reason"] == "was running when the driver started"
    st2 = Q.retry(st, "b")
    assert "b" not in st2 and st["b"]["status"] == "interrupted"          # pure: input untouched
    with pytest.raises(ValueError, match="step 'a' is done; retry only clears failed or interrupted steps"):
        Q.retry(st, "a")


@pytest.mark.parametrize("rc,check_rc,status,reason", [
    (0, 0, "done", None),
    (0, None, "done", None),
    (0, 1, "failed", "acceptance check rc 1"),
    (3, None, "failed", "rc 3"),
    (-1, None, "failed", "rc -1"),
])
def test_record_sets_the_outcome(rc, check_rc, status, reason):
    st = Q.record({"a": {"status": "running", "start": 5.0}}, "a", rc=rc, check_rc=check_rc, now=9.0)
    assert st["a"]["status"] == status and st["a"].get("reason") == reason
    assert (st["a"]["start"], st["a"]["end"], st["a"]["rc"]) == (5.0, 9.0, rc)


def test_state_round_trips_through_disk(tmp_path):
    p = tmp_path / "state.json"
    st = {"a": {"status": "done", "rc": 0, "start": 1.5, "end": 2.5}, "b": {"status": "failed", "reason": "rc 3"}}
    Q.save_state(p, st)
    assert Q.load_state(p) == st
    assert not list(tmp_path.glob("*.tmp"))                              # atomic: no temp file left behind
    assert Q.load_state(tmp_path / "missing.json") == {}


# ------------------------------------------------------------------ r12_launch

def test_parse_launch_reads_the_launchers_one_line():
    out = ("launched pid 4242; log /x/logs/r12_scratch/r12_stack_run_20260930T010000.log; "
           "mem_guard log /x/logs/r12_scratch/r12_stack_run_20260930T010000_memguard.log\n")
    assert Q.parse_launch(out) == (4242, pathlib.Path("/x/logs/r12_scratch/r12_stack_run_20260930T010000.log"))
    with pytest.raises(ValueError, match="r12_launch did not report a launch"):
        Q.parse_launch("usage: r12_launch.py [-h] --step {features,stack,tables,val}\n")


@pytest.mark.parametrize("text,rc", [
    ("[..] preflight: vm.swapusage ...; OK\n[..] END rc=0 wall=20s\n[..] R12 STEP FINISHED\n", 0),
    ("[..] END rc=2 wall=5s\n[..] STOPPING: step failed\n", 2),
    ("[..] END rc=0 wall=20s\n[..] START again\n[..] END rc=4 wall=3s\n", 4),              # the last END wins
    ("[..] preflight: vm.swapusage ...; REFUSED: heavy-compute lock is held by another job\n", 75),
    ("[..] START ...\n[12:00] iter 1,000\n", None),
    ("", None),
])
def test_parse_r12_log(text, rc):
    """No END line = the launcher is still running or died without one; the caller decides which by the pid."""
    assert Q.parse_r12_log(text) == rc


# ------------------------------------------------------------------ the loop (injected fakes)

class World:
    """Fake clock / sleep / executor. `phantom` is a list of states served one per gate poll (last one repeats)."""

    def __init__(self, rcs=None, checks=None, phantom=("idle",), paused=(), exec_s=None):
        self.exec_s = dict(exec_s or {})
        self.t = 0.0
        self.rcs = dict(rcs or {})
        self.checks = dict(checks or {})
        self.phantom = list(phantom)
        self.paused = list(paused)
        self.ran, self.checked, self.saved, self.logs, self.since = [], [], [], [], []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s

    def execute(self, step):
        self.ran.append(step.id)
        self.t += self.exec_s.get(step.id, 0.0)
        r = self.rcs.get(step.id, 0)
        return r.pop(0) if isinstance(r, list) else r

    def check(self, step):
        self.checked.append(step.id)
        return self.checks.get(step.id, 0)

    def inputs(self, since_heavy_s=None):
        self.since.append(since_heavy_s)
        ph = self.phantom.pop(0) if len(self.phantom) > 1 else self.phantom[0]
        pa = self.paused.pop(0) if self.paused else False
        return dict(now=NIGHT, pmset=AC, paused=pa, phantom=ph)

    def run(self, steps, state=None):
        return Q.run_queue(lambda: steps, dict(state or {}), execute=self.execute, check=self.check,
                           gate_inputs=self.inputs, sleep=self.sleep, clock=self.clock,
                           save=lambda st: self.saved.append(json.loads(json.dumps(st))), log=self.logs.append)


def test_loop_runs_every_step_in_order_and_checks_those_with_a_check():
    steps = Q.load_queue(_q(_light("a", check=[PY, "c"]), _heavy("b"), _light("c")))
    w = World()
    assert w.run(steps) == 0
    assert w.ran == ["a", "b", "c"] and w.checked == ["a"]
    assert {k: v["status"] for k, v in w.saved[-1].items()} == {"a": "done", "b": "done", "c": "done"}


def test_loop_never_runs_a_step_after_a_failed_step_or_check():
    """Negative invariant: the chain stops at the first failure (an unaccepted v47 must never feed a later step)."""
    steps = Q.load_queue(_q(_light("a"), _light("b"), _light("c")))
    w = World(rcs={"b": 3})
    assert w.run(steps) == 1
    assert w.ran == ["a", "b"] and w.saved[-1]["b"]["status"] == "failed"
    w2 = World(checks={"a": 1})
    steps2 = Q.load_queue(_q(_light("a", check=[PY, "c"]), _light("b")))
    assert w2.run(steps2) == 1 and w2.ran == ["a"]
    assert w2.saved[-1]["a"]["reason"] == "acceptance check rc 1"


def test_loop_does_not_check_a_step_that_failed():
    steps = Q.load_queue(_q(_light("a", check=[PY, "c"])))
    w = World(rcs={"a": 2})
    assert w.run(steps) == 1 and w.checked == []


def test_loop_marks_running_before_executing_so_a_crash_is_detectable():
    steps = Q.load_queue(_q(_light("a")))
    w = World()
    w.run(steps)
    assert w.saved[0] == {} and w.saved[1]["a"]["status"] == "running" and w.saved[-1]["a"]["status"] == "done"


def test_loop_resumes_after_done_steps_and_refuses_to_pass_an_interrupted_one():
    steps = Q.load_queue(_q(_light("a"), _light("b")))
    w = World()
    assert w.run(steps, {"a": {"status": "done"}}) == 0 and w.ran == ["b"]
    w2 = World()
    assert w2.run(steps, {"a": {"status": "running"}}) == 1 and w2.ran == []
    assert w2.saved[-1]["a"]["status"] == "interrupted"


def test_loop_waits_for_phantom_then_runs_and_counts_only_phantom_wait_toward_the_cap():
    steps = Q.load_queue(_q(_heavy("b")))
    w = World(phantom=["busy", "busy", "idle"])
    assert w.run(steps) == 0 and w.ran == ["b"]
    assert w.t == 2 * Q.POLL_S
    # paused for longer than the cap, then Phantom busy: the pause must not have used up the cap
    polls = int(Q.YIELD_CAP_S / Q.POLL_S) + 5
    w2 = World(phantom=["busy"] * (polls + 2) + ["idle"], paused=[True] * polls)
    w2.run(steps)
    assert w2.t == (polls + 2) * Q.POLL_S


def test_loop_goes_after_the_yield_cap_on_a_runner_that_stays_busy():
    steps = Q.load_queue(_q(_heavy("b")))
    w = World(phantom=["busy"])
    assert w.run(steps) == 0 and w.ran == ["b"] and w.t == Q.YIELD_CAP_S
    assert any("yield cap" in m for m in w.logs)


def test_loop_retries_a_refused_launch_and_does_not_count_it_as_a_failure():
    """rc 75 = the launcher's preflight refused (lock held / another launcher alive / window): wait and try again."""
    steps = Q.load_queue(_q(_heavy("b"), _light("c")))
    w = World(rcs={"b": [75, 75, 0]})
    assert w.run(steps) == 0 and w.ran == ["b", "b", "b", "c"]
    assert w.saved[-1]["b"]["refusals"] == 2 and w.saved[-1]["b"]["status"] == "done"


def test_loop_rereads_the_queue_each_step_so_steps_can_be_appended_while_it_runs():
    first = Q.load_queue(_q(_light("a")))
    both = Q.load_queue(_q(_light("a"), _light("b")))
    calls = iter([first, both, both, both])
    w = World()
    rc = Q.run_queue(lambda: next(calls), {}, execute=w.execute, check=w.check, gate_inputs=w.inputs, sleep=w.sleep,
                     clock=w.clock, save=lambda st: None, log=w.logs.append)
    assert rc == 0 and w.ran == ["a", "b"]


def test_loop_logs_a_wait_reason_once_not_every_poll():
    steps = Q.load_queue(_q(_heavy("b")))
    w = World(phantom=["busy"] * 10 + ["idle"])
    w.run(steps)
    assert sum("WAIT" in m for m in w.logs) == 1


# ------------------------------------------------------------------ wait (a queue step on a job launched outside it)

@pytest.mark.parametrize("text,rc", [
    ("[..] END rc=0 wall=14000s\n[..] R12 STEP FINISHED\n", 0),
    ("[..] END rc=70 wall=900s\n", 70),
    ("[..] iter 39,000\n", 3),                                   # died without an END line
    ("[..] preflight: ...; REFUSED: heavy-compute lock is held\n", 3),   # never ran: not a retryable 75 here
])
def test_wait_rc_maps_the_log_to_a_step_rc(text, rc):
    """A refused launch must not come back as 75, or the queue would retry a wait step forever."""
    assert Q.wait_rc(text) == rc


# ------------------------------------------------------------------ groups (independent chains, 2026-09-29)

def _g(i, group):
    return {"id": i, "kind": "light", "argv": [PY, "-B", "x.py", i], "group": group}


def test_load_queue_reads_an_optional_group_defaulting_to_main():
    steps = Q.load_queue(_q(_light("a"), _g("b", "hpx")))
    assert [s.group for s in steps] == ["main", "hpx"]
    with pytest.raises(ValueError, match="step 'b' group must be a non-empty string"):
        Q.load_queue(_q(_g("b", "")))


def test_next_step_skips_only_the_failed_steps_group():
    """A failed HPX step must not stop the independent MOVE chain; later steps of the failed group never run."""
    steps = Q.load_queue(_q(_g("d1", "dwl"), _g("h1", "hpx"), _g("h2", "hpx"), _g("m1", "move")))
    st = {"d1": {"status": "done"}, "h1": {"status": "failed", "reason": "rc 1"}}
    s, why = Q.next_step(steps, st)
    assert s.id == "m1"
    st["m1"] = {"status": "done"}
    assert Q.next_step(steps, st) == (None, "stopped: h1 failed (rc 1)")


def test_next_step_interrupted_blocks_its_group_too():
    steps = Q.load_queue(_q(_g("m1", "move"), _g("m2", "move"), _g("x", "other")))
    s, _ = Q.next_step(steps, {"m1": {"status": "interrupted", "reason": "was running when the driver started"}})
    assert s.id == "x"


def test_loop_runs_other_groups_after_a_failure_and_still_exits_1():
    steps = Q.load_queue(_q(_g("d1", "dwl"), _g("h1", "hpx"), _g("h2", "hpx"), _g("m1", "move")))
    w = World(rcs={"h1": 2})
    assert w.run(steps) == 1
    assert w.ran == ["d1", "h1", "m1"]                        # h2 skipped, m1 still ran
    assert "h2" not in w.saved[-1] and w.saved[-1]["m1"]["status"] == "done"


def test_next_step_reports_the_first_failure_in_file_order():
    steps = Q.load_queue(_q(_g("a", "g1"), _g("b", "g2")))
    st = {"a": {"status": "failed", "reason": "rc 1"}, "b": {"status": "interrupted", "reason": "x"}}
    assert Q.next_step(steps, st) == (None, "stopped: a failed (rc 1)")


# ------------------------------------------------------------------ guard (mem_guard on lock steps, 2026-09-29)

def test_load_queue_reads_guard_on_lock_steps_only():
    """A lock step can ask for mem_guard (--pause-on-phantom) on its process tree; r12 steps already get it from
    r12_launch and light steps are < 1 GB, so `guard` there is a mistake and is refused."""
    st = Q.load_queue(_q({**_heavy("b"), "guard": True}, _heavy("c")))
    assert (st[0].guard, st[1].guard) == (True, False)
    with pytest.raises(ValueError, match="step 'a' guard applies to lock steps only"):
        Q.load_queue(_q({**_light("a"), "guard": True}))
    with pytest.raises(ValueError, match="step 'b' guard must be true or false"):
        Q.load_queue(_q({**_heavy("b"), "guard": "yes"}))


def test_guard_command_is_the_launchers_pause_on_phantom_mem_guard_beside_the_step_log(tmp_path):
    cmd = Q.guard_cmd(4242, tmp_path / "move_dryrun.log")
    assert cmd[-1] == "--pause-on-phantom" and cmd[cmd.index("--pid") + 1] == "4242"
    assert cmd[cmd.index("--log") + 1] == str(tmp_path / "move_dryrun_memguard.log")


def test_r12_launch_command_always_passes_machine_guards():
    """The owner's machine has AC power and must respect Phantom's track_record window, so every r12 step the
    queue launches must carry --machine-guards (r12_launch.py defaults it off, for environments that cannot check
    either). Fails when --machine-guards is dropped from the built command (the owner's runs would silently stop
    being refused on battery / inside the window)."""
    step = Q.Step(id="f", kind="r12", argv=("features", "--store", "data/r12/features_x", "--final"),
                 check=None, r12_step="features")
    cmd = Q.r12_launch_cmd(step)
    assert cmd == [Q.PY, "-B", str(ROOT / "scripts/analysis/r12_launch.py"), "--step", "features", "--detach",
                   "--machine-guards", "--", "features", "--store", "data/r12/features_x", "--final"]


# ------------------------------------------------------------------ lock race (BC-35, 2026-09-30 00:26)

LOCK_RACE_LOG = """[2026-09-30 00:25:22] preflight: vm.swapusage: total = 11264.00M  used = 9640.06M; OK
[2026-09-30 00:25:22] START /opt/homebrew/bin/python3.11 -B scripts/analysis/r12_stack.py serve ...
Traceback (most recent call last):
  File "scripts/run_guards.py", line 40, in acquire_lock
    raise RuntimeError(f"heavy-compute lock {path} is held by another job; not stacking") from None
RuntimeError: heavy-compute lock /home/u/.phantom/phantom_heavy_compute.lock is held by another job; not stacking
[2026-09-30 00:26:42] END rc=1 wall=80s (includes any mem_guard pause) watchdog_peak_tree_rss=0.03GiB killed=False
"""


def test_regression_bug_35_parse_r12_log_reads_a_runner_that_lost_the_lock_race_as_a_refusal():
    """The launcher's preflight saw the lock free; 80 s later the runner's own acquire found a Phantom job
    (build_clim_state_daily, a launchd job that takes the lock itself) holding it and exited rc 1 before doing any work.
    That is a refusal to wait out (75), not a failure of the step."""
    assert Q.parse_r12_log(LOCK_RACE_LOG) == 75


def test_regression_bug_35_parse_r12_log_keeps_a_failure_whose_log_has_no_lock_message():
    """The lock message only maps to 75 when the run ended non-zero; a later clean END stays that END."""
    assert Q.parse_r12_log(LOCK_RACE_LOG.replace("END rc=1", "END rc=0")) == 0
    assert Q.parse_r12_log("[..] START\nValueError: boom\n[..] END rc=1 wall=3s\n") == 1


@pytest.mark.parametrize("lock_free,runner,state", [
    (True, "idle", "idle"), (True, "busy", "busy"), (True, "stale", "stale"),
    (False, "idle", "lock"), (False, "stale", "lock"), (False, "busy", "lock"),
])
def test_regression_bug_35_machine_state_treats_a_held_lock_as_lock_whatever_the_runner_says(lock_free, runner, state):
    """The batch runner's line cannot see Phantom launchd jobs that take the lock directly (clim_state did at 00:25);
    the lock itself is the authoritative signal."""
    assert Q.machine_state(lock_free, runner) == state


# ------------------------------------------------------------------ reviewer MAJOR 1 / 2 (2026-09-30 01:1x)

def test_phantom_state_waits_for_a_runner_tick_newer_than_our_last_heavy_step():
    """MAJOR 1: a "nothing due" line written BEFORE our previous heavy step ended says nothing about what came due while
    we held the lock (00:34:21 line reused for four heavy starts). Older line -> busy (wait for the next tick, <= 5 min);
    no tick for longer than FRESH_S since our step ended -> stale (runner silent)."""
    idle = "batch runner: no job started — nothing due (load1=1)"
    assert Q.phantom_state(idle, age_s=200, since_heavy_s=100) == "busy"          # written 100 s before our step ended
    assert Q.phantom_state(idle, age_s=100, since_heavy_s=100) == "busy"          # same instant: not newer
    assert Q.phantom_state(idle, age_s=50, since_heavy_s=100) == "idle"           # a tick after our step ended
    assert Q.phantom_state(idle, age_s=700, since_heavy_s=650) == "stale"         # runner silent > FRESH_S since
    assert Q.phantom_state(idle, age_s=200, since_heavy_s=None) == "idle"         # no heavy step yet: age rule only


def test_loop_hands_the_gate_the_time_since_the_last_heavy_step_ended():
    steps = Q.load_queue(_q(_heavy("a"), _light("l"), _heavy("b")))
    w = World()
    assert w.run(steps) == 0
    heavy_polls = [s for s in w.since if s is not None]
    assert heavy_polls and heavy_polls[0] == 0.0                                # driver start counts as a heavy end
    assert w.since[-1] == 0.0                                                   # b's gate: a ended at the same fake time


def test_next_step_lets_other_groups_run_while_a_refused_step_waits_and_holds_its_own_group():
    """MAJOR 2: a refused step (retry_at in the future) must not block independent groups, and nothing after it in ITS
    group may overtake it."""
    steps = Q.load_queue(_q(_g("h1", "hpx"), _g("h2", "hpx"), _g("m1", "move")))
    st = {"h1": {"status": "pending", "refusals": 1, "retry_at": 500.0}}
    assert Q.next_step(steps, st, now=100.0)[0].id == "m1"
    st["m1"] = {"status": "done"}
    assert Q.next_step(steps, st, now=100.0) == (None, "deferred")
    assert Q.next_step(steps, st, now=500.0)[0].id == "h1"


def test_loop_runs_another_group_right_after_a_refusal_then_retries():
    steps = Q.load_queue(_q(_g("h1", "hpx"), _g("m1", "move")))
    w = World(rcs={"h1": [75, 0]})
    assert w.run(steps) == 0
    assert w.ran == ["h1", "m1", "h1"]                                          # m1 did not wait RETRY_S behind h1


def test_loop_gives_up_on_a_step_refused_max_times_and_keeps_other_groups():
    steps = Q.load_queue(_q(_g("h1", "hpx"), _g("m1", "move")))
    w = World(rcs={"h1": [75] * Q.MAX_REFUSALS})
    assert w.run(steps) == 1
    assert w.saved[-1]["h1"]["status"] == "failed"
    assert w.saved[-1]["h1"]["reason"] == f"refused {Q.MAX_REFUSALS} times (lock or launcher never free)"
    assert w.saved[-1]["m1"]["status"] == "done" and w.ran.count("h1") == Q.MAX_REFUSALS


def test_loop_resets_the_heavy_clock_when_a_heavy_step_ends():
    """After a 1,000 s heavy step, the next heavy gate must be told 0 s (a tick newer than THAT step), not 1,000 s."""
    steps = Q.load_queue(_q(_heavy("a"), _light("l"), _heavy("b")))
    w = World(exec_s={"a": 1_000.0, "l": 50.0})
    assert w.run(steps) == 0
    assert w.since == [0.0, 0.0, 50.0]            # a at start; l right after a; b after l's 50 s (l is light)


def test_process_alive_counts_a_process_it_may_not_signal_as_alive():
    """The driver waits on a launched step's PID. PID 1 exists but belongs to root (PermissionError on signal 0): alive;
    our own PID: alive; a reaped child's PID: gone. Mutation: PermissionError -> False (the old helper) -> red."""
    import os
    import subprocess
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert Q.process_alive(os.getpid()) is True
    assert Q.process_alive(1) is True
    assert Q.process_alive(child.pid) is False


# ------------------------------------------------------------------ BC-43 sibling (2026-10-04, tooling audit T4): the
# driver started mem_guard with its output discarded and never looked at it again, so a guard that refused or died left
# the heavy step running unguarded (or SIGSTOPped, before mem_guard released on exit) while the step log said "guarded".
# Mutation rehearsal 2026-10-04 (scratch copy, one at a time): 12 / 12 killed — PQ1 start check off, PQ2 an early step
# exit not returned, PQ3 one respawn too many, PQ4 restart without releasing, PQ5 replacement not marked RESTART, PQ6 /
# PQ7 a stop that does not release first or does not terminate, PQ8 a step left running after a failed release, PQ9 the
# restart flag dropped in the wiring, PQ10 a failed release reported as done, PQ11 the wrong pid released, PQ12 notes not
# written. PQ9 and PQ10 survived until the wiring moved out of _execute into guard_hooks with its own tests.
# PQ13 (Protocol B review 02:50): an unbounded wait after SIGTERM -> the stubborn-step test. WR1 the "mem_guard:" line
# dropped from _execute -> the real-wrapper wiring test.

class _FakeProc:
    """The lock wrapper as the driver sees it. `waits` scripts each p.wait(timeout) (None = still running). terminate()
    models the wrapper: the whole command ends and it returns 143 = 128 + SIGTERM (tests/test_run_with_heavy_lock.py
    pins the real one)."""

    def __init__(self, waits, pid=4242):
        self.waits, self.pid, self.terminated = list(waits), pid, False

    def poll(self):
        return self.waits[0] if self.waits and self.waits[0] is not None else None

    def wait(self, timeout=None):
        if timeout is None:
            return next((w for w in self.waits if w is not None), 143)
        w = self.waits.pop(0) if self.waits else None
        if w is None:
            raise subprocess.TimeoutExpired("step", timeout)
        return w

    def terminate(self):
        self.terminated = True
        self.waits = [143]


class _FakeGuard:
    """A guard whose poll() follows `polls` and then stays at the last value (None = alive)."""

    def __init__(self, polls):
        self.polls = list(polls)
        self.returncode = None

    def poll(self):
        v = self.polls.pop(0) if len(self.polls) > 1 else self.polls[0]
        self.returncode = v
        return v


def _supervise(proc, guards, release_ok=True, **kw):
    notes, released, started = [], [], []

    def start(restart):
        g = guards.pop(0) if guards else _FakeGuard([None])          # never runs out: a kill must come from an assertion
        started.append(restart)
        return g

    def release(pid):
        released.append(pid)
        return release_ok
    rc = Q.supervise_guarded(proc, start, release, notes.append, poll_s=0, start_s=0, sleep=lambda s: None, **kw)
    return rc, notes, released, started


def test_a_guarded_step_that_ends_normally_returns_its_code():
    rc, notes, released, started = _supervise(_FakeProc([None, None, 0]), [_FakeGuard([None])])
    assert (rc, notes, released, started) == (0, [], [], [False])


def test_regression_bug_43_a_guard_that_does_not_start_stops_the_step():
    proc = _FakeProc([None, None, 0])
    rc, notes, released, _ = _supervise(proc, [_FakeGuard([2])])
    assert rc == Q.GUARD_FAILED_RC and proc.terminated and released == [4242]
    assert notes == ["mem_guard did not start (exit 2); the step was stopped rather than run unguarded"]


def test_a_step_that_ends_before_the_guard_check_keeps_its_own_code():
    """A lock refusal (exit 75) ends the step at once; its code must reach the retry logic unchanged."""
    proc = _FakeProc([75])
    rc, notes, released, _ = _supervise(proc, [_FakeGuard([0])])
    assert rc == 75 and not proc.terminated and notes == [] and released == []


def test_regression_bug_43_a_guard_that_dies_mid_step_is_released_and_restarted():
    proc = _FakeProc([None, None, None, 0])
    rc, notes, released, started = _supervise(proc, [_FakeGuard([None, 1]), _FakeGuard([None])])
    assert rc == 0 and released == [4242] and started == [False, True] and not proc.terminated
    assert notes == ["mem_guard exited (exit 1) while the step runs: its tree is released and the guard restarted (1 of 3)"]


def test_a_guard_that_keeps_dying_stops_the_step():
    proc = _FakeProc([None] * 20 + [0])
    guards = [_FakeGuard([None, 1]) for _ in range(5)]
    rc, notes, released, started = _supervise(proc, guards, max_respawns=2)
    assert rc == Q.GUARD_FAILED_RC and proc.terminated and started == [False, True, True] and released == [4242] * 3
    assert notes[-1] == "mem_guard died 3 times (last exit 1); the step was stopped rather than run unguarded"


def test_a_tree_that_cannot_be_released_is_never_waited_on():
    """D6: a failed release used to be logged as a success, and terminating a still-stopped wrapper then hung the driver
    (SIGTERM stays pending on a stopped process)."""
    proc = _FakeProc([None, None, 0])
    rc, notes, released, _ = _supervise(proc, [_FakeGuard([2])], release_ok=False)
    assert rc == Q.GUARD_FAILED_RC and not proc.terminated and released == [4242]
    assert notes == ["mem_guard did not start (exit 2); its tree could not be released, so the step is left as it is for inspection"]
    proc2 = _FakeProc([None, None, None, 0])
    rc2, notes2, _, started2 = _supervise(proc2, [_FakeGuard([None, 1])], release_ok=False)
    assert rc2 == Q.GUARD_FAILED_RC and not proc2.terminated and started2 == [False]
    assert notes2 == ["mem_guard exited (exit 1) and the step's tree could not be released; left as it is for inspection"]


@pytest.mark.slow
def test_a_guarded_lock_step_runs_through_the_real_wrapper_and_guard(tmp_path, monkeypatch):
    """The wiring the fakes above stand in for: _execute starts the lock wrapper, a real mem_guard beside it (its own
    output in its log), and returns the step's code once the step, the lock and the guard are done with. A temporary
    lock file, never the machine's."""
    monkeypatch.setenv("PHANTOM_HEAVY_LOCK_PATH", str(tmp_path / "heavy.lock"))
    step = Q.Step(id="tiny", kind="lock", argv=(sys.executable, "-c", "print('step ran'); raise SystemExit(3)"),
                  check=None, r12_step=None, guard=True)
    assert Q._execute(tmp_path, step) == 3
    log = (tmp_path / "tiny.log").read_text()
    assert "step ran" in log and "heavy lock RELEASED" in log
    assert f"mem_guard: {sys.executable} -B {ROOT / 'tools/mem_guard.py'} --pid " in log
    guard_log = tmp_path / "tiny_memguard.log"
    d = Deadline(30)                             # the guard notices the step has gone at its next poll (POLL_S = 15 s)
    while "MEMGUARD END" not in (guard_log.read_text() if guard_log.exists() else ""):
        assert not d.expired(), "the guard never ended after its step"
        time.sleep(0.1)
    assert "MEMGUARD START pid=" in guard_log.read_text()


def test_guard_hooks_pass_the_restart_flag_and_name_the_step_wrapper(tmp_path):
    started = []
    with open(tmp_path / "step.log", "a") as fh:
        start_guard, _, note = Q.guard_hooks(4242, tmp_path / "step.log", fh,
                                             start=lambda pid, log, restart: started.append((pid, log, restart)) or
                                             ("cmd", f"proc{len(started)}"))
        assert start_guard(False) == "proc1" and start_guard(True) == "proc2"
        note("mem_guard exited (exit 1)")
    assert started == [(4242, tmp_path / "step.log", False), (4242, tmp_path / "step.log", True)]
    line = (tmp_path / "step.log").read_text()
    assert line.endswith("] mem_guard exited (exit 1)\n") and dt.datetime.strptime(line[1:20], "%Y-%m-%d %H:%M:%S")


@pytest.mark.slow
def test_guard_hooks_release_continues_a_stopped_tree_and_reports_a_refusal(tmp_path):
    """release() runs the real `mem_guard.py --release`: a stopped process of ours is continued (True); a target the
    guard refuses (a Phantom process) is reported as not released (False), never as done."""
    import signal
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    phantom = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)", "Phantom/track_record.py"])
    try:
        child.send_signal(signal.SIGSTOP)
        with open(tmp_path / "step.log", "a") as fh:
            _, release, _ = Q.guard_hooks(child.pid, tmp_path / "step.log", fh)
            assert release(child.pid) is True
            assert release(phantom.pid) is False
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(child.pid)], capture_output=True, text=True).stdout
        assert not state.strip().startswith("T")
        assert "refusing to signal a Phantom process" in (tmp_path / "step.log").read_text()
    finally:
        for proc in (child, phantom):
            proc.kill()
            proc.wait()


def test_a_step_that_does_not_end_after_sigterm_is_left_running_not_waited_on_forever():
    """Protocol B review 2026-10-04 (MEDIUM-4): the stop waited on the wrapper without a bound, so a command that would
    not end (or a stopped worker the release missed) hung the whole queue. Now the driver gives it STOP_WAIT_S, says so
    and moves on; the step keeps the heavy lock, so nothing stacks on it."""
    class Stubborn(_FakeProc):
        def terminate(self):
            self.terminated = True                   # ignores the signal: every timed wait expires

    proc = Stubborn([None, None, None])
    notes = []
    rc = Q._stop_step(proc, lambda pid: True, notes.append, "mem_guard did not start (exit 2)", stop_s=0.01)
    assert rc == Q.GUARD_FAILED_RC and proc.terminated
    assert notes == ["mem_guard did not start (exit 2); the step did not end within 0 s of SIGTERM and is left running "
                     "for inspection, UNGUARDED (its memory guard is gone) and holding the heavy lock"]
