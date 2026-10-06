"""PRC campaign queue — registered steps, one after another, with room for Phantom between heavy steps.

    /opt/homebrew/bin/python3.11 -B tools/prc_queue.py run --queue logs/campaign/queue.json --detach
    /opt/homebrew/bin/python3.11 -B tools/prc_queue.py status
    /opt/homebrew/bin/python3.11 -B tools/prc_queue.py retry --id <step>      # clear a failed / interrupted step
    /opt/homebrew/bin/python3.11 -B tools/prc_queue.py wait --pid <launcher pid> --log <its log>   # as a light step
    touch logs/campaign/PAUSE        # hold the queue at the next step boundary; rm it to resume

Why: our heavy steps hold Phantom's machine-wide lock, and Phantom's batch runner (Phantom scripts/batch_runner.py)
starts none of its managed heavy jobs while it is held. Chaining our steps back to back would starve Phantom for a day.
Between heavy steps the lock is left free until the runner's own last log line says "nothing due" (capped at
YIELD_CAP_S, because a job reserved for memory it never gets would otherwise hold the queue forever).

Rules the driver enforces (each has a test in tests/test_prc_queue.py):
- a queue file is refused at load if any argv/check element carries a shell expansion ($VAR, $(...)): argv is never
  shell-expanded (zsh does not word-split an unquoted $VAR).
- heavy steps (kind r12 / lock) start only on AC power, never inside Phantom's 11:40-12:15 window (v34_launch's own
  tested rule), never while the lock is held by anyone, and only after the Phantom yield. Light steps honour PAUSE only.
- a step is marked running BEFORE it executes; a step found running at start-up is marked interrupted and the queue
  stops there (partial outputs are never re-run silently: `retry` is a decision).
- the first failed step or failed acceptance check stops ITS GROUP (a step's optional "group", default "main"): no
  later step of that group runs; independent groups continue; the driver exits 1 at the end if anything failed.
- rc 75 (the launcher's preflight refused: lock held, another launcher alive, window) is a wait, not a failure: the step
  is retried after RETRY_S while OTHER groups run; after MAX_REFUSALS refusals it fails (6 h).
- the Phantom yield only trusts a batch-runner line NEWER than our last heavy step's end (reviewer MAJOR 1, 2026-09-30).
- the queue file is re-read before every step, so steps can be appended while the driver runs.
- a running heavy step is NOT paused by PAUSE or by the Phantom yield: an flock dies only with its holder. Inside a step,
  mem_guard --pause-on-phantom still stops our process tree while a Phantom heavy job runs: r12_launch attaches it to r12
  steps; a lock step with "guard": true gets the same command (v34_launch.memguard_cmd) on its wrapper's tree.

Mutation rehearsal 2026-09-29 (each mutant run against tests/test_prc_queue.py, all RED, then reverted): see
tests/test_prc_queue.py's module docstring.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib.util
import json
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable                 # the interpreter running this tool runs its children
CAMPAIGN = ROOT / "logs" / "campaign"
PHANTOM_RUNNER_LOG = pathlib.Path.home() / "Projects" / "Phantom" / "reports" / "batch_runner" / "cron.stdout.log"
KINDS = ("r12", "lock", "light")
STEP_KEYS = {"id", "kind", "argv", "check", "step", "note", "group", "guard"}
FRESH_S = 600            # the runner ticks every 300 s; two missed ticks = it is not running
YIELD_CAP_S = 3 * 3600
POLL_S = 60
RETRY_S = 120
MAX_REFUSALS = 180       # x RETRY_S = 6 h of refusals -> the step fails (a launcher / lock that never frees)
LAUNCH_RE = re.compile(r"launched pid (\d+); log (\S+?);")
END_RE = re.compile(r"END rc=(-?\d+)")
SHELL_RE = re.compile(r"\$[A-Za-z_{(]")
#: run_guards.acquire_lock's message when a RUNNER finds the lock taken after its launcher's preflight probe (BC-35)
LOCK_REFUSED = "is held by another job; not stacking"
HEAVY_LOCK = pathlib.Path(os.environ.get("PHANTOM_HEAVY_LOCK_PATH",
                                         pathlib.Path.home() / ".phantom" / "phantom_heavy_compute.lock"))


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load("v34_launch", "scripts/analysis/v34_launch.py")      # on_ac_power, in_phantom_window: the launchers' rules


@dataclasses.dataclass(frozen=True)
class Step:
    id: str
    kind: str
    argv: tuple
    check: tuple | None
    r12_step: str | None
    group: str = "main"
    guard: bool = False


def _strings(sid: str, what: str, v) -> tuple:
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ValueError(f"step {sid!r} {what} must be a list of strings")
    if not v:
        raise ValueError(f"step {sid!r} has an empty {what}")
    for x in v:
        if SHELL_RE.search(x):
            raise ValueError(f"step {sid!r} {what} has a shell expansion: {x!r}")
    return tuple(v)


def load_queue(obj) -> list:
    """The queue file's steps, validated; any defect raises ValueError with the reason."""
    if not isinstance(obj, dict) or not isinstance(obj.get("steps"), list):
        raise ValueError("queue needs a 'steps' list")
    out, seen = [], set()
    for d in obj["steps"]:
        sid = d.get("id") if isinstance(d, dict) else None
        if not isinstance(sid, str) or not sid:
            raise ValueError("every step needs a non-empty string id")
        extra = sorted(set(d) - STEP_KEYS)
        if extra:
            raise ValueError(f"step {sid!r} has unknown keys {extra}")
        if sid in seen:
            raise ValueError(f"duplicate step id {sid!r}")
        seen.add(sid)
        kind = d.get("kind")
        if kind not in KINDS:
            raise ValueError(f"unknown kind {kind!r}")
        if kind == "r12" and not isinstance(d.get("step"), str):
            raise ValueError(f"r12 step {sid!r} needs 'step'")
        argv = _strings(sid, "argv", d.get("argv"))
        check = _strings(sid, "check", d["check"]) if d.get("check") is not None else None
        group = d.get("group", "main")
        if not isinstance(group, str) or not group:
            raise ValueError(f"step {sid!r} group must be a non-empty string")
        guard = d.get("guard", False)
        if not isinstance(guard, bool):
            raise ValueError(f"step {sid!r} guard must be true or false")
        if guard and kind != "lock":
            raise ValueError(f"step {sid!r} guard applies to lock steps only")
        out.append(Step(id=sid, kind=kind, argv=argv, check=check, r12_step=d.get("step") if kind == "r12" else None,
                        group=group, guard=guard))
    return out


GUARD_FAILED_RC = 70          # a guarded step whose mem_guard could not run: stopped, or (if it could not be released or
                              # would not end) left as it is, still holding the heavy lock so nothing stacks on it
GUARD_START_S = 3.0           # seconds a fresh guard gets before the driver checks that it is alive
MAX_GUARD_RESPAWNS = 3
STOP_WAIT_S = 600.0           # how long a stopped step gets to end after SIGTERM before the driver moves on — a chosen
                              # bound, not a measurement: a Python runner without a SIGTERM handler ends at once; ten
                              # minutes leaves room for one that cleans up first


def _stop_step(p, release, note, why: str, stop_s: float = STOP_WAIT_S) -> int:
    """Stop a guarded step: release its tree first (a stopped process cannot act on SIGTERM), then terminate the lock
    wrapper, which forwards the signal to the whole command and holds the lock until it has ended
    (scripts/run_with_heavy_lock.py). A tree that cannot be released, or a step that has not ended `stop_s` seconds after
    SIGTERM, is left as it is and said so — never waited on forever. Either way the step still holds the heavy lock, so
    the next heavy step refuses (and retries) rather than stacking on it."""
    if not release(p.pid):
        note(f"{why}; its tree could not be released, so the step is left as it is for inspection")
        return GUARD_FAILED_RC
    p.terminate()
    try:
        p.wait(timeout=stop_s)
    except subprocess.TimeoutExpired:
        note(f"{why}; the step did not end within {stop_s:.0f} s of SIGTERM and is left running for inspection, "
             "UNGUARDED (its memory guard is gone) and holding the heavy lock")
        return GUARD_FAILED_RC
    note(f"{why}; the step was stopped rather than run unguarded")
    return GUARD_FAILED_RC


def supervise_guarded(p, start_guard, release, note, poll_s=POLL_S, start_s=GUARD_START_S, sleep=time.sleep,
                      max_respawns=MAX_GUARD_RESPAWNS) -> int:
    """Wait for a guarded lock step while watching its mem_guard (BC-43, tooling audit T4).

    `start_guard(restart)` launches a guard (restart=True for a replacement, so its log continues the run instead of
    starting a new one); `release(pid)` SIGCONTs the step's tree and returns whether it worked. A guard that is gone at
    the start check stops the step (a step that already ended keeps its own code: a lock refusal must still reach the
    retry logic). A guard that dies while the step runs has the tree released (it may have died holding it stopped) and
    is replaced, at most `max_respawns` times; past that the step is stopped. Returns the step's code, or
    GUARD_FAILED_RC."""
    guard = start_guard(False)
    sleep(start_s)
    rc = p.poll()
    if rc is not None:
        return rc
    if guard.poll() is not None:
        return _stop_step(p, release, note, f"mem_guard did not start (exit {guard.returncode})")
    respawns = 0
    while True:
        try:
            return p.wait(timeout=poll_s)
        except subprocess.TimeoutExpired:
            pass
        if guard.poll() is None:
            continue
        code = guard.returncode
        if respawns >= max_respawns:
            return _stop_step(p, release, note, f"mem_guard died {respawns + 1} times (last exit {code})")
        if not release(p.pid):
            note(f"mem_guard exited (exit {code}) and the step's tree could not be released; left as it is for inspection")
            return GUARD_FAILED_RC
        respawns += 1
        guard = start_guard(True)
        note(f"mem_guard exited (exit {code}) while the step runs: its tree is released and the guard restarted "
             f"({respawns} of {max_respawns})")


def guard_hooks(pid: int, step_log, fh, start=None, run=subprocess.run) -> tuple:
    """(start_guard, release, note) for supervise_guarded, for the lock step whose wrapper has `pid`: start_guard(restart)
    starts mem_guard detached with its own output in its log (v34_launch.start_memguard); release(target) runs
    `mem_guard.py --release target` and says whether it worked; note(msg) writes a timestamped line to the step log."""
    start = start or V.start_memguard

    def start_guard(restart: bool):
        return start(pid, step_log, restart=restart)[1]

    def release(target: int) -> bool:
        r = run([PY, "-B", str(ROOT / "tools/mem_guard.py"), "--release", str(target)], cwd=ROOT, stdout=fh,
                stderr=subprocess.STDOUT)
        return r.returncode == 0

    def note(msg: str) -> None:
        fh.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
        fh.flush()

    return start_guard, release, note


def guard_cmd(pid: int, step_log) -> list:
    """mem_guard for a guarded lock step: the launchers' own command (always --pause-on-phantom), log beside the step's."""
    return V.memguard_cmd(pid, step_log)


def r12_launch_cmd(step: Step) -> list:
    """The command that launches one r12 step through scripts/analysis/r12_launch.py: always --machine-guards (the
    owner's own machine has AC power to check and a reserved Phantom window to respect; --machine-guards defaults
    off in r12_launch.py only for environments, like Linux or a test sandbox, that cannot check `pmset` at all)."""
    return [PY, "-B", str(ROOT / "scripts/analysis/r12_launch.py"), "--step", step.r12_step, "--detach",
            "--machine-guards", "--", *step.argv]


def phantom_state(last_line, age_s, fresh_s: float = FRESH_S, since_heavy_s=None) -> str:
    """idle / busy / lock / stale, from the batch runner's last stdout line and the log's age. With `since_heavy_s`
    (seconds since our last heavy step ended), a line no newer than that step is not evidence about what came due while
    we held the lock: busy (wait for the next tick) — or stale if the runner has not ticked for > fresh_s since."""
    if last_line is None or age_s is None:
        return "stale"
    if since_heavy_s is not None and age_s >= since_heavy_s:
        return "stale" if since_heavy_s > fresh_s else "busy"
    if age_s > fresh_s:
        return "stale"
    if "nothing due" in last_line:
        return "idle"
    if "heavy lock is held" in last_line:
        return "lock"
    return "busy"


def machine_state(lock_free: bool, runner_state: str) -> str:
    """The Phantom / lock state a heavy step's gate sees: a held heavy lock is 'lock' whatever the batch runner's line
    says (Phantom launchd jobs such as build_clim_state_daily take the lock directly, invisible to the runner)."""
    return runner_state if lock_free else "lock"


def gate(step: Step, *, now: dt.datetime, pmset: str, paused: bool, phantom: str, phantom_waited_s: float,
         cap_s: float = YIELD_CAP_S) -> tuple:
    """(go, reason, kind). kind is the wait class: pause / power / window / lock / phantom, or go."""
    if paused:
        return False, "paused (PAUSE file present)", "pause"
    if step.kind == "light":
        return True, "go", "go"
    if not V.on_ac_power(pmset):
        return False, "not on AC power", "power"
    if V.in_phantom_window(now):
        return False, "inside Phantom's 11:40-12:15 window", "window"
    if phantom == "lock":
        return False, "the heavy lock is held by another job", "lock"
    if phantom == "busy":
        if phantom_waited_s >= cap_s:
            return True, f"yield cap reached after {phantom_waited_s / 3600:.1f} h; Phantom still busy", "go"
        return False, "yielding to Phantom's batch runner (jobs due or running)", "phantom"
    return True, "go", "go"


def recover(state: dict) -> dict:
    st = json.loads(json.dumps(state))
    for v in st.values():
        if v.get("status") == "running":
            v["status"], v["reason"] = "interrupted", "was running when the driver started"
    return st


def retry(state: dict, step_id: str) -> dict:
    st = json.loads(json.dumps(state))
    cur = st.get(step_id, {}).get("status")
    if cur not in ("failed", "interrupted"):
        raise ValueError(f"step {step_id!r} is {cur or 'pending'}; retry only clears failed or interrupted steps")
    del st[step_id]
    return st


def next_step(steps, state: dict, now=None) -> tuple:
    """The first step still to run whose group has no failed / interrupted step before it and is not waiting out a
    refusal (retry_at > now). Groups are independent chains: a failure blocks the rest of ITS group, a refused step
    holds its group only. (None, "deferred") while any group waits; else (None, "all done") / (None, "stopped: ...")."""
    blocked, deferred, first_stop = set(), set(), None
    for s in steps:
        v = state.get(s.id, {})
        stt = v.get("status")
        if stt in ("failed", "interrupted"):
            blocked.add(s.group)
            first_stop = first_stop or f"stopped: {s.id} {stt} ({v.get('reason')})"
            continue
        if stt == "done" or s.group in blocked or s.group in deferred:
            continue
        if now is not None and v.get("retry_at", 0.0) > now:
            deferred.add(s.group)
            continue
        return s, "next"
    if deferred:
        return None, "deferred"
    return (None, first_stop) if first_stop else (None, "all done")


def record(state: dict, step_id: str, *, rc: int, check_rc, now: float) -> dict:
    st = json.loads(json.dumps(state))
    v = st.setdefault(step_id, {})
    v.update(end=now, rc=rc)
    if rc != 0:
        v["status"], v["reason"] = "failed", f"rc {rc}"
    elif check_rc not in (None, 0):
        v["status"], v["reason"] = "failed", f"acceptance check rc {check_rc}"
    else:
        v["status"] = "done"
        v.pop("reason", None)
    return st


def save_state(path: pathlib.Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True))
    os.replace(tmp, path)


def load_state(path: pathlib.Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def parse_launch(stdout: str) -> tuple:
    m = LAUNCH_RE.search(stdout)
    if not m:
        raise ValueError(f"r12_launch did not report a launch: {stdout.strip()[:200]!r}")
    return int(m.group(1)), pathlib.Path(m.group(2))


def parse_r12_log(text: str):
    """The launcher log's last END rc; 75 (retry) when the preflight refused, or when the run ended non-zero AND the log
    carries run_guards' lock-held message (acquire_lock prints it; runners call it before their work, so a real
    mid-run failure logs no such line); None when there is no END yet."""
    ends = END_RE.findall(text)
    if ends:
        rc = int(ends[-1])
        return 75 if (rc != 0 and LOCK_REFUSED in text) else rc
    if "REFUSED:" in text:
        return 75
    return None


def wait_rc(text: str) -> int:
    """The rc a `wait` step returns for a launcher log: its END rc; 3 when there is none (died, or refused and never
    ran). Never 75: a wait step is not a launch, so a refusal must not be retried."""
    rc = parse_r12_log(text)
    return 3 if rc in (None, 75) else rc


def run_queue(read_steps, state: dict, *, execute, check, gate_inputs, sleep, clock, save, log) -> int:
    """0 when every step is done, 1 when the queue stopped on a failed / interrupted step. All I/O is injected."""
    state = recover(state)
    save(state)
    last_heavy_end = clock()                       # the driver's start counts: wait for a runner tick newer than it
    while True:
        steps = read_steps()
        step, why = next_step(steps, state, now=clock())
        if step is None:
            if why == "deferred":
                sleep(POLL_S)
                continue
            log(f"QUEUE {why}")
            return 0 if why == "all done" else 1
        waited, last_reason = 0.0, None
        while True:
            go, reason, kind = gate(step, phantom_waited_s=waited,
                                    **gate_inputs(since_heavy_s=clock() - last_heavy_end))
            if go:
                if reason != "go":
                    log(f"{step.id}: {reason}")
                break
            if reason != last_reason:
                log(f"WAIT {step.id}: {reason}")
                last_reason = reason
            sleep(POLL_S)
            if kind == "phantom":
                waited += POLL_S
        prev = state.get(step.id, {})
        state = json.loads(json.dumps(state))
        state[step.id] = {"status": "running", "start": clock(), "refusals": prev.get("refusals", 0)}
        save(state)
        log(f"START {step.id} ({step.kind})")
        rc = execute(step)
        if step.kind != "light":
            last_heavy_end = clock()
        if rc == 75:
            n = state[step.id]["refusals"] + 1
            if n >= MAX_REFUSALS:
                state[step.id] = {"status": "failed", "refusals": n, "end": clock(),
                                  "reason": f"refused {n} times (lock or launcher never free)"}
                save(state)
                log(f"END {step.id} -> failed ({state[step.id]['reason']})")
                continue
            state[step.id] = {"status": "pending", "refusals": n, "retry_at": clock() + RETRY_S}
            save(state)
            log(f"REFUSED {step.id} (lock or launcher busy); other groups may run; retry after {RETRY_S} s")
            continue
        check_rc = check(step) if (rc == 0 and step.check) else None
        state = record(state, step.id, rc=rc, check_rc=check_rc, now=clock())
        save(state)
        log(f"END {step.id} rc={rc}" + ("" if check_rc is None else f" check_rc={check_rc}")
            + f" -> {state[step.id]['status']}")


# ------------------------------------------------------------------ real I/O (thin; the decisions above are tested)

def _phantom_now(since_heavy_s=None) -> str:  # pragma: no cover - reads the live Phantom runner log
    try:
        age = time.time() - PHANTOM_RUNNER_LOG.stat().st_mtime
        with open(PHANTOM_RUNNER_LOG, "rb") as fh:
            fh.seek(max(0, PHANTOM_RUNNER_LOG.stat().st_size - 4096))
            lines = [x for x in fh.read().decode("utf-8", "replace").splitlines() if x.strip()]
        return phantom_state(lines[-1] if lines else None, age, since_heavy_s=since_heavy_s)
    except FileNotFoundError:
        return "stale"


def _lock_free() -> bool:  # pragma: no cover - touches the real lock: a non-blocking probe, released at once
    import fcntl
    HEAVY_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(HEAVY_LOCK, "a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return True


def _inputs(d: pathlib.Path, since_heavy_s=None) -> dict:  # pragma: no cover - live machine state
    pm = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    return dict(now=dt.datetime.now(), pmset=pm, paused=(d / "PAUSE").exists(),
                phantom=machine_state(_lock_free(), _phantom_now(since_heavy_s)))


def process_alive(pid: int) -> bool:
    """True while `pid` names a live process; a process we may not signal (PermissionError: it exists, owned by another
    user) is alive, so the driver never treats a running step as finished."""
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    return True


def _execute(d: pathlib.Path, step: Step) -> int:  # pragma: no cover - launches the real step
    out = d / f"{step.id}.log"
    with open(out, "a") as fh:
        fh.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {step.kind} {' '.join(step.argv)}\n")
        fh.flush()
        if step.kind == "light":
            return subprocess.call(list(step.argv), cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if step.kind == "lock":
            p = subprocess.Popen([PY, "-B", str(ROOT / "scripts/run_with_heavy_lock.py"), *step.argv], cwd=ROOT,
                                 stdout=fh, stderr=subprocess.STDOUT)
            if step.guard:                           # mem_guard SIGSTOPs the wrapper's whole tree (tools/mem_guard.py)
                fh.write(f"mem_guard: {' '.join(guard_cmd(p.pid, out))}\n")
                fh.flush()
                return supervise_guarded(p, *guard_hooks(p.pid, out, fh))
            return p.wait()
        r = subprocess.run(r12_launch_cmd(step), cwd=ROOT, capture_output=True, text=True)
        fh.write(r.stdout + r.stderr)
        fh.flush()
        if r.returncode != 0:
            return r.returncode
        pid, log_path = parse_launch(r.stdout)
        fh.write(f"waiting on r12_launch pid {pid}, log {log_path}\n")
        fh.flush()
        while process_alive(pid):
            time.sleep(POLL_S)
        rc = parse_r12_log(log_path.read_text() if log_path.exists() else "")
        return -1 if rc is None else rc


def _check(d: pathlib.Path, step: Step) -> int:  # pragma: no cover - runs the real acceptance command
    with open(d / f"{step.id}.check.log", "a") as fh:
        return subprocess.call(list(step.check), cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)


def _log(d: pathlib.Path, msg: str) -> None:  # pragma: no cover
    with open(d / "status.txt", "a") as fh:
        fh.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")


def main(argv=None) -> int:  # pragma: no cover - CLI shell around the tested functions
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(CAMPAIGN), help="state / status / step logs / PAUSE (default logs/campaign)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--queue", required=True)
    r.add_argument("--detach", action="store_true")
    sub.add_parser("status")
    t = sub.add_parser("retry")
    t.add_argument("--id", required=True)
    w = sub.add_parser("wait", help="block until a launcher pid exits, then exit with its log's END rc")
    w.add_argument("--pid", type=int, required=True)
    w.add_argument("--log", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "wait":
        while process_alive(a.pid):
            time.sleep(POLL_S)
        lp = pathlib.Path(a.log)
        rc = wait_rc(lp.read_text() if lp.exists() else "")
        print(f"pid {a.pid} gone; {lp.name}: rc {rc}")
        return rc
    d = pathlib.Path(a.dir).resolve()
    d.mkdir(parents=True, exist_ok=True)
    state_p, pid_p = d / "state.json", d / "queue.pid"
    driver_alive = pid_p.exists() and process_alive(int(pid_p.read_text()))
    if a.cmd == "status":
        print(json.dumps(load_state(state_p), indent=1, sort_keys=True))
        print("driver:", "alive" if driver_alive else "not running", "| paused:", (d / "PAUSE").exists())
        return 0
    if a.cmd == "retry":
        if driver_alive:
            raise SystemExit("the driver is running; stop it before editing its state")
        save_state(state_p, retry(load_state(state_p), a.id))
        _log(d, f"RETRY cleared {a.id}")
        return 0
    queue_p = pathlib.Path(a.queue).resolve()
    load_queue(json.loads(queue_p.read_text()))                   # fail loud before detaching
    if driver_alive:
        raise SystemExit(f"a queue driver is already running (pid {pid_p.read_text().strip()})")
    if a.detach:
        p = subprocess.Popen([PY, "-B", str(pathlib.Path(__file__).resolve()), "--dir", str(d), "run", "--queue",
                              str(queue_p)], cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=open(d / "driver.out", "a"), stderr=subprocess.STDOUT)
        print(f"queue driver pid {p.pid}; status {d / 'status.txt'}")
        return 0
    pid_p.write_text(str(os.getpid()))
    try:
        _log(d, f"DRIVER start pid {os.getpid()} queue {queue_p}")
        return run_queue(lambda: load_queue(json.loads(queue_p.read_text())), load_state(state_p),
                         execute=lambda s: _execute(d, s), check=lambda s: _check(d, s),
                         gate_inputs=lambda since_heavy_s=None: _inputs(d, since_heavy_s), sleep=time.sleep,
                         clock=time.time,
                         save=lambda st: save_state(state_p, st), log=lambda m: _log(d, m))
    finally:
        pid_p.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
