"""Shared test helper (not a test module): `isolated` runs a real-data test in a CHILD pytest process.

Why: pytest runs every test in one interpreter, and in-process RSS guards (e.g. run_guards.start_rss_guard) read the
process's LIFETIME peak (ru_maxrss). A real-data test that loads a big frame in the shared process raises that peak for
every later test: on 2026-09-23 the suite died with exit 70; on 2026-09-29 tests/test_dfc2_build.py raised it
1.77 -> 4.12 GB and a later guard-calling unit test failed in-suite while passing alone. The guards' contract is right
for CLI jobs; the fix is that heavy tests do not share the process (BC-34). Tested in tests/test_isolation.py.

    ISO = _load("prc_test_isolation", "tests/isolation.py")
    @ISO.isolated
    def test_on_real_data(...): ...

Also here (BC-50): `run_guarded` / `wait_guarded`, the only way a test waits on a child with a timeout. tools/mem_guard.py
SIGSTOPs the whole pytest tree for 105-120 s while a Phantom job runs, and a wall-clock timeout shorter than that expired
on resume (2026-10-09: the 90 s BC-5 test killed at -9, passing alone in 15 s). Their timeout counts only the seconds this
process was running: a ticking thread that returns more than PAUSE_GAP_S late has been stopped (or the machine slept),
and that gap is not charged. A hung child (0 % CPU, the parent ticking) still times out. `Deadline` is the same clock for
polling loops. tests/test_guarded_timeout_tripwire.py refuses raw timeouts and wall-clock deadlines in tests/.
"""
from __future__ import annotations

import functools
import inspect
import os
import pathlib
import subprocess
import sys
import threading
import time

CHILD_ENV = "PRC_ISOLATED_CHILD"
TIMEOUT_S = 1_800
TICK_S = 0.25            # the active clock's tick
PAUSE_GAP_S = 3.0        # a tick returning this late = the process was stopped; the gap is not charged to the timeout
POLL_S = 1.0             # how often a guarded wait re-reads the active clock


class ActiveClock:
    """Seconds this process has been running since construction: wall time minus every gap in which the tick thread
    could not run (SIGSTOP by mem_guard). close() stops the thread. time.monotonic does not advance while the
    machine sleeps, so a sleep is neither seen nor charged."""

    def __init__(self):
        self._t0 = time.monotonic()
        self._last = self._t0            # the last tick; a reader sees a gap still in progress through it
        self._paused = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._tick, name="active-clock", daemon=True)
        self._thread.start()

    @staticmethod
    def _stopped_for(gap: float) -> float:
        return gap - TICK_S if gap > PAUSE_GAP_S else 0.0

    def _tick(self):
        while not self._stop.wait(TICK_S):
            now = time.monotonic()
            self._paused += self._stopped_for(now - self._last)
            self._last = now

    def active(self) -> float:
        # On resume the waiting main thread runs before the tick thread has booked the gap: count the gap since the
        # last tick here as well (the tick thread books the same gap and moves _last, so it is never counted twice).
        now = time.monotonic()
        last = self._last                # read _last BEFORE _paused: if the tick books the gap in between, it is
        paused = self._paused            # counted twice (a longer timeout), never zero times (a spurious timeout)
        return now - self._t0 - paused - self._stopped_for(now - last)

    def expired(self, seconds: float) -> bool:
        return self.active() >= seconds

    def close(self):
        self._stop.set()
        self._thread.join()


class Deadline(ActiveClock):
    """A polling loop's deadline in running seconds: `while not d.expired(): ...`. Replaces `time.time() + N`
    deadlines, which the same stop makes expire (BC-50, the pid-file and guard-log waits)."""

    def __init__(self, seconds: float):
        super().__init__()
        self.seconds = seconds

    def expired(self, seconds: float | None = None) -> bool:      # type: ignore[override]
        return self.active() >= (self.seconds if seconds is None else seconds)



def run_guarded(argv, *, timeout: float, input=None, capture_output: bool = False, check: bool = False, **popen_kw):
    """subprocess.run(argv, timeout=...) whose timeout counts only running seconds (ActiveClock). Same keywords as
    subprocess.run for what the tests use (capture_output, text, cwd, env, input, check, stdout/stderr); on timeout the
    child is killed and subprocess.TimeoutExpired is raised with the partial output, as subprocess.run does; on any other
    exception the child is killed too."""
    if capture_output:
        if "stdout" in popen_kw or "stderr" in popen_kw:
            raise ValueError("stdout and stderr arguments may not be used with capture_output")
        popen_kw["stdout"] = subprocess.PIPE
        popen_kw["stderr"] = subprocess.PIPE
    if input is not None:
        if "stdin" in popen_kw:
            raise ValueError("stdin and input arguments may not both be used")
        popen_kw["stdin"] = subprocess.PIPE
    with subprocess.Popen(argv, **popen_kw) as p:        # the clock's thread starts after the fork, not before it
        clock = ActiveClock()
        try:
            pending = input
            while True:
                remaining = timeout - clock.active()
                if remaining <= 0:
                    p.kill()
                    out, err = p.communicate()           # what subprocess.run does: the output so far, decoded
                    raise subprocess.TimeoutExpired(p.args, timeout, output=out, stderr=err)
                try:
                    out, err = p.communicate(pending, timeout=min(remaining, POLL_S))
                    break
                except subprocess.TimeoutExpired:
                    pending = None                       # input is sent once; later calls only drain the pipes
        except subprocess.TimeoutExpired:
            raise
        except BaseException:
            p.kill()
            raise
        finally:
            clock.close()
    done = subprocess.CompletedProcess(p.args, p.returncode, out, err)
    if check:
        done.check_returncode()
    return done


def wait_guarded(p: subprocess.Popen, *, timeout: float) -> int:
    """Popen.wait(timeout=...) whose timeout counts only running seconds; raises subprocess.TimeoutExpired and, like
    Popen.wait, leaves the child to the caller."""
    clock = ActiveClock()
    try:
        while True:
            remaining = timeout - clock.active()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(p.args, timeout)
            try:
                return p.wait(timeout=min(remaining, POLL_S))
            except subprocess.TimeoutExpired:
                pass
    finally:
        clock.close()


def isolated(fn):
    """In the parent: run `<defining file>::<test name>` in a child pytest and assert it passed (child output in the
    failure message). In the child (CHILD_ENV=1): run the body. Parametrised tests are not supported (node ids)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get(CHILD_ENV) == "1":
            return fn(*args, **kwargs)
        node = f"{pathlib.Path(inspect.getsourcefile(fn)).resolve()}::{fn.__name__}"
        r = run_guarded([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
                        env={**os.environ, CHILD_ENV: "1"}, capture_output=True, text=True, timeout=TIMEOUT_S)
        assert r.returncode == 0, f"child pytest rc {r.returncode}: {r.stdout[-4000:]}{r.stderr[-2000:]}"
        tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        if "skipped" in tail and "passed" not in tail:
            import pytest
            pytest.skip(f"child skipped: {tail}")
    return wrapper
