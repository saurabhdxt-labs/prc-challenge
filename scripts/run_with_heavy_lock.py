"""Run a command while holding the machine-wide heavy-compute lock.

    python scripts/run_with_heavy_lock.py <cmd> [args...]

Any job over ~1 GB takes this lock, so two heavy jobs never run on the machine at once. The lock is a non-blocking fcntl
flock: it dies with its last holder and is freed explicitly when the command ends. If another job holds it the wrapper
refuses at once (exit 75) — skip, never stack. The lock file is `PHANTOM_HEAVY_LOCK_PATH` if set, else
~/.phantom/phantom_heavy_compute.lock (the lock scripts/run_guards.py takes).

The command starts in its own process group and inherits the lock's file descriptor, so the lock is held while the
wrapper or the command is alive: a wrapper that crashes or is killed (even with SIGKILL) leaves the lock with the
running command, never free under it. SIGTERM, SIGINT and SIGHUP sent to the wrapper are forwarded to the command's
whole group, each after a SIGCONT so a paused group can act on it; a signal the wrapper was started with ignored (nohup)
stays ignored. The signal handlers only record; the polling loop forwards each recorded signal exactly once, before it
logs anything, and log lines are best effort (a closed terminal or pipe never stops the forwarding or the wait). Once
the command has exited, the processes left in its group get a SIGCONT (a worker stopped while its parent died could
otherwise never finish) and the wrapper waits for them. When the whole group has ended the wrapper unlocks explicitly:
an unlock applies to every copy of the descriptor, so a process that left the group (setsid, a daemon, a job-control
shell) and inherited one cannot keep the lock — "heavy lock RELEASED" means the lock is free. Returns the command's exit
code (128 + N when it was killed by signal N), 2 for a usage error and 127 when the command cannot be started. Standard
library only.
"""
import fcntl
import os
import pathlib
import signal
import subprocess
import sys
import time

FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
EX_TEMPFAIL = 75
POLL_S = 0.1


def lock_path() -> pathlib.Path:
    return pathlib.Path(os.environ.get("PHANTOM_HEAVY_LOCK_PATH",
                                       pathlib.Path.home() / ".phantom" / "phantom_heavy_compute.lock"))


def say(msg: str, fd: int = 1) -> None:
    """One log line, unbuffered. Best effort: a closed terminal or pipe must never stop the forwarding or the wait, and
    nothing is left in a buffer to fail again when the interpreter exits."""
    try:
        os.write(fd, (msg + "\n").encode())
    except OSError:
        return


def signal_group(pgid: int, sig: int) -> None:
    """Send `sig` to every process in the group, after a SIGCONT (a stopped process acts on a signal only once resumed)."""
    for s in (signal.SIGCONT, sig):
        try:
            os.killpg(pgid, s)
        except ProcessLookupError:                   # the group has already ended
            return
        except PermissionError:                      # no member we may signal; nothing more can be done from here
            return


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                          # a member exists that we may not signal: it still runs
        return True
    return True


def start_command(argv, lock_fd: int) -> subprocess.Popen:
    """The command in a new session (its pgid is its pid), holding a copy of the lock's file descriptor."""
    return subprocess.Popen(argv, start_new_session=True, pass_fds=(lock_fd,))


def wait_group(child: subprocess.Popen, received: list) -> int:
    """Forward each signal number appended to `received` (by the handlers) to the command's group, once, until the
    command and every process left in its group have ended. Returns the command's exit code."""
    forwarded, said_waiting = 0, False
    while True:
        while forwarded < len(received):
            sig = received[forwarded]
            forwarded += 1
            signal_group(child.pid, sig)
            say(f"forwarded {signal.Signals(sig).name} to the command")
        if child.poll() is not None:
            if not group_alive(child.pid):
                break
            if not said_waiting:                     # a leftover stopped while its parent died must still be able to end
                signal_group(child.pid, signal.SIGCONT)
                say(f"the command exited ({child.returncode}); waiting for the processes left in its group")
                said_waiting = True
        time.sleep(POLL_S)
    rc = child.returncode
    return 128 - rc if rc < 0 else rc


def run_command(argv, received: list, lock_fd: int) -> int:
    """Start the command (unless a stop request already arrived: then it is never started) and wait for its group.
    Only starting it may fail with "cannot run"; an error while waiting propagates and the command keeps the lock."""
    if received:
        say(f"{signal.Signals(received[0]).name} arrived before the command started; not starting it")
        return 128 + received[0]
    try:
        child = start_command(argv, lock_fd)
    except OSError as e:
        say(f"cannot run {argv[0]!r}: {e}", fd=2)
        return 127
    return wait_group(child, received)


def main(argv) -> int:
    if not argv:
        say("usage: run_with_heavy_lock.py <cmd> [args...]", fd=2)
        return 2
    lock = lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    received = []
    previous = {}
    for s in FORWARDED:
        if signal.getsignal(s) is signal.SIG_IGN:   # started under nohup (or similar): keep ignoring it
            continue
        previous[s] = signal.signal(s, lambda signum, _frame: received.append(signum))
    try:
        with open(lock, "a+") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                say(f"HEAVY LOCK HELD by another job — skipping, not stacking. ({lock})")
                return EX_TEMPFAIL
            say(f"heavy lock ACQUIRED pid {os.getpid()} -> running: {' '.join(argv)}")
            rc = run_command(argv, received, fh.fileno())
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)   # the group has ended: free the lock for every descriptor copy
        say("heavy lock RELEASED")
        return rc
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
