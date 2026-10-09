"""tests/isolation.py — the shared `isolated` decorator: a real-data test runs in a CHILD pytest process so its memory never
raises the shared pytest process's lifetime peak RSS (BC-34, 2026-09-29: a later unit test read an inherited 3.89 GB
peak and tripped its own 2 GB guard in-suite). No real child is started here: run_guarded is faked.
Seen red first (collection error; the skip test: 1 failed). Mutation rehearsal 2026-09-30: 7 of 7 RED (child check
inverted, node without the test name, rc unchecked, CHILD_ENV not set, skip detection off, timeout dropped, no wraps)."""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prc_test_isolation", ROOT / "tests/isolation.py")
I = importlib.util.module_from_spec(_spec)
sys.modules["prc_test_isolation"] = I
_spec.loader.exec_module(I)


def _body_recorder():
    calls = []

    def body(tmp=None):
        calls.append(tmp)
    return body, calls


def test_isolated_runs_the_body_inline_inside_the_child(monkeypatch):
    monkeypatch.setenv(I.CHILD_ENV, "1")
    body, calls = _body_recorder()
    I.isolated(body)("x")
    assert calls == ["x"]


def test_isolated_starts_a_child_pytest_on_the_defining_file_and_test_name(monkeypatch):
    monkeypatch.delenv(I.CHILD_ENV, raising=False)
    seen = {}

    def fake_run(argv, env, capture_output, text, timeout):
        seen.update(argv=argv, env=env, timeout=timeout)
        return subprocess.CompletedProcess(argv, 0, "1 passed", "")
    monkeypatch.setattr(I, "run_guarded", fake_run)
    body, calls = _body_recorder()
    I.isolated(body)()
    assert calls == []                                              # the parent never runs the body
    assert seen["argv"][:3] == [sys.executable, "-B", "-m"] and seen["argv"][3] == "pytest"
    assert seen["argv"][-1] == f"{pathlib.Path(__file__).resolve()}::body"
    assert seen["env"][I.CHILD_ENV] == "1" and seen["timeout"] == I.TIMEOUT_S


def test_isolated_fails_loudly_with_the_childs_output(monkeypatch):
    monkeypatch.delenv(I.CHILD_ENV, raising=False)
    monkeypatch.setattr(I, "run_guarded",
                        lambda argv, env, capture_output, text, timeout: subprocess.CompletedProcess(argv, 1, "E boom", "err"))
    body, _ = _body_recorder()
    with pytest.raises(AssertionError, match="child pytest rc 1.*E boom"):
        I.isolated(body)()


def test_isolated_keeps_the_test_name_and_signature_for_pytest():
    """pytest injects fixtures by the wrapped signature (functools.wraps sets __wrapped__)."""
    def test_something(monkeypatch, tmp_path):
        pass
    w = I.isolated(test_something)
    assert w.__name__ == "test_something" and w.__wrapped__ is test_something


def test_isolated_reports_a_skipped_child_as_skipped_not_passed(monkeypatch):
    """A child that skips (e.g. its real file is absent) exits 0; the parent must say SKIPPED, not PASSED, or a missing
    real-data check hides behind a green run (reviewer MINOR, 2026-09-30)."""
    monkeypatch.delenv(I.CHILD_ENV, raising=False)
    monkeypatch.setattr(I, "run_guarded", lambda argv, env, capture_output, text, timeout:
                        subprocess.CompletedProcess(argv, 0, "s\n1 skipped in 0.01s\n", ""))
    body, _ = _body_recorder()
    with pytest.raises(pytest.skip.Exception, match="child skipped"):
        I.isolated(body)()


# ---- run_guarded / wait_guarded: subprocess timeouts that count only the seconds this process was running (BC-50) ----
# Seen red first 2026-10-09 (AttributeError: no run_guarded; the pause test fails against a raw subprocess.run).
# Mutation rehearsal 2026-10-09 (4 of 4 RED): M1 the tick never books the gap -> pause test; M2 a gap still in progress not
# counted by active() -> pause test (the resume race); M3 hung child not killed -> kill test (40 s); M4 stops never
# recognised -> pause test. Two earlier cuts of the pause test let M1 live: a sleeping grandchild returns at the resume
# (a sleep deadline keeps running while stopped), so it finished within the first poll; it now burns CPU time. Review
# 2026-10-09: READY handshake instead of a fixed 0.5 s, 1 s CPU under 4 s with a 6 s stop, SIGCONT + kill in finally.

import os
import signal
import time


def test_run_guarded_returns_a_completed_process_for_a_short_child():
    r = I.run_guarded([sys.executable, "-c", "import sys; sys.stdout.write('ok'); sys.stderr.write('e'); sys.exit(3)"],
                      capture_output=True, text=True, timeout=30)
    assert (r.returncode, r.stdout, r.stderr) == (3, "ok", "e")


@pytest.mark.slow
def test_run_guarded_kills_a_hung_child_at_the_active_timeout():
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        I.run_guarded([sys.executable, "-c", "import time; time.sleep(30)"], capture_output=True, text=True, timeout=1.5)
    assert time.monotonic() - t0 < 10                      # it did not wait for the child's own 30 s


def test_run_guarded_sends_input_and_refuses_the_conflicting_keywords():
    """The docstring's `input` claim: review 2026-10-09 found stdin was never a pipe, so input was silently dropped."""
    r = I.run_guarded([sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read().upper())"],
                      input="abc", capture_output=True, text=True, timeout=30)
    assert r.stdout == "ABC"
    with pytest.raises(ValueError, match="stdin and input"):
        I.run_guarded(["true"], input="x", stdin=subprocess.DEVNULL, timeout=5)
    with pytest.raises(ValueError, match="capture_output"):
        I.run_guarded(["true"], capture_output=True, stdout=subprocess.DEVNULL, timeout=5)


@pytest.mark.slow
def test_run_guarded_timeout_carries_the_partial_output():
    with pytest.raises(subprocess.TimeoutExpired) as e:
        I.run_guarded([sys.executable, "-c", "import sys, time; sys.stdout.write('partial'); sys.stdout.flush(); time.sleep(30)"],
                      capture_output=True, text=True, timeout=1.5)
    assert e.value.output == "partial"


@pytest.mark.slow
def test_wait_guarded_returns_the_exit_code_and_raises_on_a_hang():
    p = subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"])
    assert I.wait_guarded(p, timeout=30) == 7
    q = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            I.wait_guarded(q, timeout=1.0)
    finally:
        q.kill(); q.wait()


_PAUSED_CHILD = """
import importlib.util, subprocess, sys, time
spec = importlib.util.spec_from_file_location('iso', sys.argv[1]); I = importlib.util.module_from_spec(spec)
spec.loader.exec_module(I)
t0 = time.monotonic()
fn = I.run_guarded if sys.argv[2] == 'guarded' else subprocess.run
print('READY', flush=True)                      # the parent stops the group only once this line is out
r = fn([sys.executable, '-c', 'import time; [None for _ in iter(lambda: time.process_time() < 1.0, False)]'], capture_output=True, text=True, timeout=4.0)
print('DONE', round(time.monotonic() - t0, 1), r.returncode)
"""


def _run_child_stopped_for(seconds: float, mode: str) -> subprocess.CompletedProcess:
    """A child that runs a grandchild needing 1 s of CPU under a 4 s timeout; its whole process group is SIGSTOPped
    for `seconds` once the child has said READY — what tools/mem_guard.py does to the pytest tree while another job
    runs. The grandchild burns CPU rather than sleeping: a sleep's deadline keeps running while the process is stopped
    (it returned at the resume and hid mutant M1), CPU time does not, so most of its work remains after the resume.
    The group is continued, then killed, in `finally` so a failure leaves no stopped processes behind."""
    p = subprocess.Popen([sys.executable, "-B", "-c", _PAUSED_CHILD, str(ROOT / "tests/isolation.py"), mode],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        assert p.stdout.readline().strip() == "READY"
        time.sleep(0.3)                             # the grandchild is spawned right after READY
        os.killpg(p.pid, signal.SIGSTOP)
        try:
            time.sleep(seconds)
        finally:
            os.killpg(p.pid, signal.SIGCONT)
        I.wait_guarded(p, timeout=60)
        out, err = p.communicate()
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    return subprocess.CompletedProcess(p.args, p.returncode, out, err)


@pytest.mark.slow
def test_regression_bug_50_a_guard_pause_longer_than_the_timeout_does_not_fail_the_guarded_child():
    """BC-50: tools/mem_guard.py SIGSTOPs the whole pytest tree for 105-120 s while a Phantom job runs; a subprocess
    timeout shorter than that expired on resume (2026-10-08 a 30 s usage test; 2026-10-09 the 90 s BC-5 test, child
    returncode -9, passes alone in 15 s). The guarded run counts only running seconds."""
    r = _run_child_stopped_for(6.0, "guarded")
    assert r.returncode == 0 and "DONE" in r.stdout, r.stderr[-2000:]
    assert float(r.stdout.split()[1]) >= 6.0                # the pause really happened (wall time includes it)


@pytest.mark.slow
def test_the_pause_test_discriminates_a_raw_subprocess_run():
    """Negative control for the test above: the same pause fails a raw subprocess.run with the same timeout."""
    r = _run_child_stopped_for(6.0, "raw")
    assert r.returncode != 0 and "TimeoutExpired" in r.stderr
