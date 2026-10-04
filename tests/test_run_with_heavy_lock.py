"""scripts/run_with_heavy_lock.py holds the machine-wide heavy lock for exactly as long as its command runs.

BC-43 sibling (2026-10-04, Protocol B integration audit D1): the wrapper ran its command with `subprocess.call` and had no
signal handling. A SIGTERM to the wrapper (the queue driver stopping a step whose memory guard could not run) killed the
wrapper at once, the OS dropped the lock with it, and the command kept running — orphaned, unguarded and unlocked, so the
next heavy job (ours or the other system's) could start on top of it. The wrapper now forwards SIGTERM / SIGINT / SIGHUP
to its command, waits for it to end and only then releases the lock.

The command now runs in its own process group: signals reach every process of it (each after a SIGCONT, so a paused
command can act on them) and the lock is released only when the whole group has ended.

Every test uses a temporary lock file (PHANTOM_HEAVY_LOCK_PATH), never the real one. Written first and seen red
2026-10-04 (the child outlived the wrapper; the lock was free while it ran). Mutation rehearsal 2026-10-04 (scratch copy,
one at a time, source verified unchanged after): 13 / 13 killed — W1 no new session -> the signal tests; W2 the leader
signalled instead of the group -> grandchild test; W3 no SIGCONT -> paused-command test (it failed to kill until the
command was given its own SIGTERM handler: a stopped process with none is killed anyway); W4 the group not waited for ->
leftover-process test; W5 SIGHUP not forwarded -> SIGHUP case; W6 the raw negative code returned -> 128 + N test; W7 no
forwarding line -> SIGTERM test; W8 a command started after an early signal -> early-signal test; W9 / W12 wrong code or
no message when the command cannot start -> 127 test; W10 a member we may not signal counted as gone -> PermissionError
test; W11 signalling continues after "no such group" -> ended-group test (no behaviour differs; the test pins the
sequence); W13 a held lock not refused -> exit-75 test.
Part 2 (Protocol B review, 2026-10-04 02:50, HIGH-1: a failed log line was read as "cannot run", the lock released
and the signal never sent): the command now holds a copy of the lock's descriptor, forwarding comes before logging,
logging is best effort, only the start may give 127, nohup's ignored SIGHUP stays ignored. Seen red against the old
wrapper: closed-stdout, killed-wrapper and nohup tests. Rehearsal 5 / 5 killed — W14 no lock descriptor for the
command -> killed-wrapper test; W15 a log failure not absorbed -> closed-stdout test; W16 an ignored signal handled ->
nohup test; W17 a signal forwarded repeatedly -> exactly-once test; W18 the broad catch around the wait -> wait-error
test (it survived until that test existed).
Part 3 (second review, 2026-10-04 05:15: a process that left the group kept the inherited lock after "RELEASED",
reproduced): the wrapper unlocks once the group has ended, and continues leftovers once the command has exited.
Rehearsal 3 / 3 killed — W19 no explicit unlock -> escaped-process test; W20 no SIGCONT to leftovers -> stopped-leftover
test; U1 usage error exits 0 -> usage test.
"""
from __future__ import annotations

import fcntl
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "run_with_heavy_lock.py"


def _lock_is_free(path: Path) -> bool:
    with open(path, "a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return True


_STARTED = []                                                    # (wrapper, child pid) of the current test


@pytest.fixture(autouse=True)
def _no_process_outlives_its_test():
    """A failing test (or a mutant under rehearsal) must not leave a wrapper or a stopped command behind."""
    yield
    while _STARTED:
        p, child = _STARTED.pop()
        if p.poll() is None:
            p.kill()
            p.wait(timeout=20)
        for kill in (lambda: os.killpg(child, signal.SIGKILL), lambda: os.kill(child, signal.SIGKILL)):
            try:
                kill()
            except (ProcessLookupError, PermissionError):
                continue
            break


def _start(tmp_path: Path, child_code: str, **popen):
    lock = tmp_path / "heavy.lock"
    pidfile = tmp_path / "child.pid"
    code = f"import os, time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); {child_code}"
    p = subprocess.Popen([sys.executable, "-B", str(WRAPPER), sys.executable, "-c", code],
                         env={**os.environ, "PHANTOM_HEAVY_LOCK_PATH": str(lock)},
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **popen)
    deadline = time.time() + 20
    while not pidfile.exists() or not pidfile.read_text():
        assert time.time() < deadline, "the child never started"
        time.sleep(0.05)
    _STARTED.append((p, int(pidfile.read_text())))
    return p, lock, int(pidfile.read_text())


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie child of an exited parent is reaped by launchd; treat a non-running zombie as gone
    st = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(st) and not st.startswith("Z")


@pytest.mark.slow
def test_regression_bug_43_a_terminated_wrapper_ends_its_command_before_releasing_the_lock(tmp_path):
    p, lock, child = _start(tmp_path, "time.sleep(60)")
    assert not _lock_is_free(lock)                               # held while the command runs
    p.send_signal(signal.SIGTERM)
    rc = p.wait(timeout=20)
    deadline = time.time() + 10
    while _alive(child) and time.time() < deadline:
        time.sleep(0.05)
    assert not _alive(child), "the command outlived the wrapper (orphaned, unlocked)"
    assert _lock_is_free(lock)
    assert rc != 0
    out = p.stdout.read()
    assert "forwarded SIGTERM to the command" in out and out.rstrip().endswith("heavy lock RELEASED")


@pytest.mark.slow
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGHUP])
def test_other_termination_signals_are_forwarded_too(tmp_path, sig):
    p, lock, child = _start(tmp_path, "time.sleep(60)")
    p.send_signal(sig)
    p.wait(timeout=20)
    deadline = time.time() + 10
    while _alive(child) and time.time() < deadline:
        time.sleep(0.05)
    assert not _alive(child) and _lock_is_free(lock)


@pytest.mark.slow
def test_the_commands_exit_code_is_returned_and_the_lock_released(tmp_path):
    p, lock, _ = _start(tmp_path, "raise SystemExit(7)")
    assert p.wait(timeout=20) == 7
    assert _lock_is_free(lock)


def test_a_held_lock_is_refused_with_exit_75_and_the_command_never_runs(tmp_path):
    lock = tmp_path / "heavy.lock"
    marker = tmp_path / "ran"
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        r = subprocess.run([sys.executable, "-B", str(WRAPPER), sys.executable, "-c",
                            f"open({str(marker)!r}, 'w').write('x')"],
                           env={**os.environ, "PHANTOM_HEAVY_LOCK_PATH": str(lock)}, capture_output=True, text=True,
                           timeout=30)
    assert r.returncode == 75
    assert "HEAVY LOCK HELD" in r.stdout and not marker.exists()


def _wait_gone(pid: int, seconds: float = 10) -> bool:
    deadline = time.time() + seconds
    while _alive(pid) and time.time() < deadline:
        time.sleep(0.05)
    return not _alive(pid)


GRANDCHILD = ("import subprocess, sys; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep({secs})']); "
              "open({path!r}, 'w').write(str(g.pid)); ")


def _wait_file(path: Path) -> int:
    deadline = time.time() + 20
    while not path.exists() or not path.read_text():
        assert time.time() < deadline, f"{path.name} never written"
        time.sleep(0.05)
    return int(path.read_text())


@pytest.mark.slow
def test_a_terminated_wrapper_ends_the_commands_grandchildren_too(tmp_path):
    """The signal goes to the command's whole process group, not only to the process the wrapper started."""
    gfile = tmp_path / "grandchild.pid"
    p, lock, child = _start(tmp_path, GRANDCHILD.format(secs=60, path=str(gfile)) + "g.wait()")
    grandchild = _wait_file(gfile)
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=20)
    assert _wait_gone(child) and _wait_gone(grandchild), "a process of the command outlived the wrapper"
    assert _lock_is_free(lock)


@pytest.mark.slow
def test_the_lock_stays_held_while_a_process_left_in_the_group_runs(tmp_path):
    """The command exits at once but leaves a 3 s process behind: the lock is held until that process ends too."""
    gfile = tmp_path / "grandchild.pid"
    p, lock, child = _start(tmp_path, GRANDCHILD.format(secs=3, path=str(gfile)))
    grandchild = _wait_file(gfile)
    assert _wait_gone(child, 5)                                  # the command itself has exited
    assert _alive(grandchild) and not _lock_is_free(lock)        # ... but its leftover process still holds the lock
    assert p.wait(timeout=20) == 0
    assert not _alive(grandchild) and _lock_is_free(lock)
    assert "waiting for the processes left in its group" in p.stdout.read()


@pytest.mark.slow
def test_a_paused_command_is_resumed_so_it_can_act_on_the_forwarded_signal(tmp_path):
    """A memory guard may have SIGSTOPped the command. A stopped process runs its own SIGTERM handler (a graceful stop)
    only once it is continued; without the SIGCONT it would stay stopped and the wrapper would wait forever. (A command
    with no handler is killed even while stopped, so the command here installs one.)"""
    done = tmp_path / "handled"
    p, lock, child = _start(tmp_path, "import signal; signal.signal(signal.SIGTERM, lambda *a: (open(%r, 'w').write('t'), "
                                      "os._exit(0))); time.sleep(60)" % str(done))
    os.kill(child, signal.SIGSTOP)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=20) == 0                               # the command stopped gracefully
    assert done.exists() and _wait_gone(child) and _lock_is_free(lock)


@pytest.mark.slow
def test_a_command_killed_by_a_signal_returns_128_plus_the_signal(tmp_path):
    p, lock, child = _start(tmp_path, "time.sleep(60)")
    os.kill(child, signal.SIGKILL)
    assert p.wait(timeout=20) == 128 + signal.SIGKILL
    assert _lock_is_free(lock)


def test_a_command_that_cannot_start_returns_127_and_releases_the_lock(tmp_path):
    lock = tmp_path / "heavy.lock"
    r = subprocess.run([sys.executable, "-B", str(WRAPPER), str(tmp_path / "no-such-command")],
                       env={**os.environ, "PHANTOM_HEAVY_LOCK_PATH": str(lock)}, capture_output=True, text=True,
                       timeout=30)
    assert r.returncode == 127
    assert "cannot run" in r.stderr and r.stdout.rstrip().endswith("heavy lock RELEASED")
    assert _lock_is_free(lock)


def test_no_command_is_a_usage_error(tmp_path):
    r = subprocess.run([sys.executable, "-B", str(WRAPPER)], env={**os.environ, "PHANTOM_HEAVY_LOCK_PATH":
                       str(tmp_path / "heavy.lock")}, capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "usage:" in r.stderr


def _module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_with_heavy_lock_mod", WRAPPER)
    W = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(W)
    return W


def test_a_signal_before_the_command_started_means_it_never_starts(tmp_path, capfd):
    """The handlers are installed before the lock is taken; a stop request that arrives before the command exists ends
    the wrapper without starting it (starting it only to signal it at once would waste the lock)."""
    W = _module()
    marker = tmp_path / "ran"
    with open(tmp_path / "heavy.lock", "a+") as fh:
        rc = W.run_command([sys.executable, "-c", f"open({str(marker)!r}, 'w').write('x')"], [signal.SIGTERM],
                           fh.fileno())
    assert rc == 128 + signal.SIGTERM and not marker.exists()
    assert "SIGTERM arrived before the command started" in capfd.readouterr().out


def test_a_group_member_that_may_not_be_signalled_still_counts_as_running(monkeypatch):
    """killpg(pgid, 0) raises PermissionError when a member exists but belongs to another user: the group still runs."""
    W = _module()

    def killpg(pgid, sig, exc):
        raise exc
    monkeypatch.setattr(W.os, "killpg", lambda pgid, sig: killpg(pgid, sig, PermissionError()))
    assert W.group_alive(12345) is True
    monkeypatch.setattr(W.os, "killpg", lambda pgid, sig: killpg(pgid, sig, ProcessLookupError()))
    assert W.group_alive(12345) is False


def test_signalling_a_group_that_has_ended_is_not_an_error(monkeypatch):
    W = _module()
    sent = []

    def killpg(pgid, sig):
        sent.append(sig)
        raise ProcessLookupError
    monkeypatch.setattr(W.os, "killpg", killpg)
    W.signal_group(12345, signal.SIGTERM)
    assert sent == [signal.SIGCONT]                              # stops at the first "no such group"


@pytest.mark.slow
def test_regression_a_closed_stdout_never_drops_the_lock_under_a_running_command(tmp_path):
    """Protocol B review 2026-10-04 (HIGH-1): the wrapper printed "forwarding ..." BEFORE signalling, and a broad
    `except OSError` around the whole wait read the print's failure (a closed terminal: EIO; a closed pipe: EPIPE) as
    "cannot run": exit 127, lock released, signal never sent, command running on unlocked. Here the wrapper's stdout pipe
    is closed before the SIGTERM: the command must still be stopped, the lock held until then, and the code 128 + 15."""
    p, lock, child = _start(tmp_path, "time.sleep(60)")
    p.stdout.close()
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=20) == 128 + signal.SIGTERM
    assert _wait_gone(child) and _lock_is_free(lock)


@pytest.mark.slow
def test_a_killed_wrapper_leaves_the_lock_with_the_running_command(tmp_path):
    """The command holds a copy of the lock's descriptor: SIGKILL (which no handler sees) of the wrapper must not free
    the lock while the command runs; it frees once the command ends."""
    p, lock, child = _start(tmp_path, "time.sleep(60)")
    p.kill()
    p.wait(timeout=20)
    assert _alive(child) and not _lock_is_free(lock)
    os.kill(child, signal.SIGKILL)
    assert _wait_gone(child) and _lock_is_free(lock)


@pytest.mark.slow
def test_a_signal_ignored_at_start_stays_ignored(tmp_path):
    """Under nohup SIGHUP is ignored: the wrapper keeps ignoring it and never forwards it to the command."""
    p, lock, child = _start(tmp_path, "time.sleep(60)",
                            preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN))
    p.send_signal(signal.SIGHUP)
    time.sleep(0.5)
    assert p.poll() is None and _alive(child) and not _lock_is_free(lock)
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=20)
    assert "forwarded SIGHUP" not in p.stdout.read() and _wait_gone(child)


@pytest.mark.slow
def test_each_signal_is_forwarded_exactly_once(tmp_path):
    """The command counts the SIGTERMs it receives and reports them after a quiet half second: one sent, one received."""
    count = tmp_path / "count"
    p, lock, child = _start(tmp_path, "import signal; n = [0]; signal.signal(signal.SIGTERM, lambda *a: n.__setitem__(0, "
                                      "n[0] + 1)); t = time.time()\nwhile n[0] == 0 and time.time() - t < 20: "
                                      "time.sleep(0.01)\ntime.sleep(0.5); open(%r, 'w').write(str(n[0]))" % str(count))
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) == 0
    assert count.read_text() == "1"


def test_an_error_while_waiting_is_not_reported_as_cannot_run(tmp_path, monkeypatch, capfd):
    """Only starting the command may end in "cannot run" (127). An OS error while waiting propagates: the wrapper exits
    with the error (never a 127 that reads as "never started"), and the command keeps its copy of the lock's descriptor,
    so nothing stacks on it while it runs on."""
    W = _module()
    started = []

    def fail_wait(child, received):
        started.append(child.pid)
        raise OSError(5, "Input/output error")
    monkeypatch.setattr(W, "wait_group", fail_wait)
    with open(tmp_path / "heavy.lock", "a+") as fh:
        with pytest.raises(OSError, match="Input/output error"):
            W.run_command([sys.executable, "-c", "pass"], [], fh.fileno())
    assert started and "cannot run" not in capfd.readouterr().err


@pytest.mark.slow
def test_a_process_that_left_the_group_cannot_keep_the_lock(tmp_path):
    """Second review 2026-10-04 (MAJOR, reproduced): the command holds a copy of the lock's descriptor, so a descendant
    that leaves the group (here a job-control shell's background job) inherited it and kept the lock after the wrapper
    printed "heavy lock RELEASED" — Phantom's heavy jobs and the queue would then skip with no visible holder. The
    wrapper now unlocks once the group has ended: the lock is free while the escaped process still runs."""
    lock, gfile = tmp_path / "heavy.lock", tmp_path / "escaped.pid"
    r = subprocess.run([sys.executable, "-B", str(WRAPPER), "/bin/sh", "-c", f"set -m; sleep 30 >/dev/null 2>&1 & echo $! > {gfile}"],
                       env={**os.environ, "PHANTOM_HEAVY_LOCK_PATH": str(lock)}, capture_output=True, text=True,
                       timeout=30)                       # the background job writes nowhere, so the pipes close
    escaped = int(gfile.read_text())
    try:
        assert r.returncode == 0 and r.stdout.rstrip().endswith("heavy lock RELEASED")
        assert _alive(escaped) and os.getpgid(escaped) != os.getpgid(os.getpid())
        assert _lock_is_free(lock), "a process outside the command's group still holds the lock"
    finally:
        os.kill(escaped, signal.SIGKILL)


@pytest.mark.slow
def test_a_worker_stopped_when_its_parent_exits_is_continued_so_the_wait_ends(tmp_path):
    """Second review (finding 4): a group member stopped (e.g. by a memory guard) whose parent then exits would never
    run again, and the wrapper would wait on its group for ever. Once the command has exited the wrapper sends the
    leftovers a SIGCONT."""
    gfile = tmp_path / "grandchild.pid"
    p, lock, child = _start(tmp_path, GRANDCHILD.format(secs=2, path=str(gfile)) + "time.sleep(1.0)")
    grandchild = _wait_file(gfile)
    os.kill(grandchild, signal.SIGSTOP)                          # stopped while its parent is still alive
    assert p.wait(timeout=20) == 0
    assert not _alive(grandchild) and _lock_is_free(lock)
