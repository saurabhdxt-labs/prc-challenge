"""Pause OUR long run on machine memory pressure, resume on recovery (owner ruling 2026-09-24).

    /opt/homebrew/bin/python3.11 -B tools/mem_guard.py --pid <our run's pid> --log logs/<run>_memguard.log

Every POLL_S seconds: read free swap (sysctl vm.swapusage) and the system-wide free percentage (memory_pressure).
Below either pause threshold -> SIGSTOP the run's whole process tree; once free RAM is past the resume threshold
(hysteresis), free swap is not below its pause threshold, and the pause has lasted MIN_PAUSE_S -> SIGCONT. Never signals a Phantom process. Exits when the run ends.
One line per state change goes to the log (for a Monitor), plus a STILL PAUSED line every 10 minutes. A resume also
continues every process stopped at pause time that has left the tree since. However the guard ends (the run exits, an
error, SIGTERM / SIGINT / SIGHUP: logged EXIT), every process it stopped is continued first (RELEASED); after a SIGKILL,
`--release PID` does it. `--restart` marks a guard that replaces one that died during the same run (RESTART, not START).

Opt-in (--pause-on-phantom, owner ruling 2026-09-28: "keep watching the memory when phantom starts just pause and
resume"): ALSO pause our run while any Phantom HEAVY job (PHANTOM_HEAVY_JOBS) is alive, and resume only once none has
been alive for PHANTOM_QUIET_S seconds. The memory RESUME thresholds apply to that resume only if memory caused the pause
or fell below a PAUSE threshold during it; after a pause that only Phantom caused, memory must merely not be below the
PAUSE thresholds (BC-37, owner decision 2026-10-01). Off by default; with the flag unset, behaviour is unchanged
byte-for-byte.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import signal
import subprocess
import time

PAUSE_SWAP_MB = 350.0
PAUSE_FREE_PCT = 8
# Resume (owner decision 2026-10-01, BC-37): the hysteresis is on free RAM (pause < 8 %, resume >= 30 %). The swap resume
# line is the pause line: free swap is not a recovery signal on macOS — a paused run's own pages stay in swap until it
# runs again and the swap files are resized under it (a 1,000 MB line held the full fit paused after its cause had gone).
RESUME_SWAP_MB = PAUSE_SWAP_MB
RESUME_FREE_PCT = 30
MIN_PAUSE_S = 60
POLL_S = 15
PHANTOM_MARKERS = ("Phantom", "phantom", "r17_shadow", "aggregate_cell_hour", "refresh_recall", "track_record")

# Heavy Phantom jobs worth pausing OUR run for, from logs/move_scratch/watch.sh's own watch pattern.
# retrain_gulf_shadow_day_eval added 2026-09-28: reached 35 GB at 13:13, in the Jetsam report before the 13:20 reboot.
PHANTOM_HEAVY_JOBS = ("aggregate_cell_hour", "refresh_recall", "track_record", "verify_cron",
                      "retrain_gulf_shadow_day_eval")
# --pause-on-phantom only: seconds a heavy job must be continuously absent before resume is allowed.
PHANTOM_QUIET_S = 60

# BC-29: executables that must NEVER be treated as a Phantom heavy job, even if their own argv happens to
# mention a job name as plain text (a launcher shell's `pgrep -f "aggregate_cell_hour|..."` wait-loop, or a
# `grep`/`tail` watching a job's log, is not the job itself).
_NEVER_MATCH_EXE = frozenset({"zsh", "bash", "sh", "pgrep", "grep", "ps", "tail"})


def parse_swap_free_mb(s: str) -> float:
    m = re.search(r"free\s*=\s*([\d.,]+)M", s)
    if not m:
        raise ValueError(f"cannot parse vm.swapusage: {s!r}")
    return float(m.group(1).replace(",", ""))


def parse_pressure_free_pct(out: str) -> int:
    m = re.search(r"free percentage:\s*(\d+)%", out)
    if not m:
        raise ValueError(f"cannot parse memory_pressure output: {out[-200:]!r}")
    return int(m.group(1))


def decide(paused: bool, swap_free_mb: float, free_pct: int, paused_for_s: float) -> str:
    """RUN / PAUSE when running; HOLD / RESUME when paused."""
    if not paused:
        return "PAUSE" if (swap_free_mb < PAUSE_SWAP_MB or free_pct < PAUSE_FREE_PCT) else "RUN"
    recovered = swap_free_mb >= RESUME_SWAP_MB and free_pct >= RESUME_FREE_PCT
    return "RESUME" if (recovered and paused_for_s >= MIN_PAUSE_S) else "HOLD"


def phantom_heavy_alive(ps_text: str, ignore_pids: frozenset = frozenset()) -> list:
    """Pure parser: which PHANTOM_HEAVY_JOBS are alive right now, given `ps -A -o pid=,command=` text.

    BC-29 (2026-09-28): identity is decided by what a process EXECUTES, not by text anywhere in its argv. A
    line matches only when its argv contains a path element ending in `<name>.py` AND its own executable is
    a Python interpreter (or its command line includes `/Projects/Phantom`). Shells (`zsh`/`bash`/`sh`, with
    `-c`/`-lc`), `pgrep`, `grep`, `ps`, and `tail` are NEVER matched, regardless of what their argv contains
    -- so a launcher shell whose own argv happens to mention a job name (e.g. a wait-loop's
    `pgrep -f "aggregate_cell_hour|..."` pattern), or a `grep`/`tail` watching a job's log, is never mistaken
    for the job. `ignore_pids` excludes our own run, this guard, and (BC-29) the guard's own ancestor pid
    chain (see `_ancestor_pids`), so an ancestor process is never matched even if its line would otherwise
    look like a real job. One entry per matching alive process, in the order seen; a process command may
    match at most one job name (the first configured job found in it).
    """
    found = []
    for line in ps_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid_s, command = parts
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if pid in ignore_pids:
            continue
        tokens = command.split()
        if not tokens:
            continue
        exe = pathlib.Path(tokens[0]).name.lower().lstrip("-")
        if exe in _NEVER_MATCH_EXE:
            continue
        is_python = "python" in exe
        if not (is_python or "/Projects/Phantom" in command):
            continue
        for name in PHANTOM_HEAVY_JOBS:
            script = name + ".py"
            if any(tok == script or tok.endswith("/" + script) for tok in tokens[1:]):
                found.append(name)
                break
    return found


def below_pause(swap_free_mb: float, free_pct: int) -> bool:
    """Memory is past a PAUSE threshold (the same strict < that decide() pauses on)."""
    return swap_free_mb < PAUSE_SWAP_MB or free_pct < PAUSE_FREE_PCT


def decide_with_phantom(paused: bool, swap_free_mb: float, free_pct: int, paused_for_s: float,
                         phantom_names: list, phantom_quiet_for_s: float, memory_caused: bool = True) -> tuple:
    """decide() plus the --pause-on-phantom overlay. Returns (action, reason).

    action is RUN/PAUSE/HOLD/RESUME as in decide(). reason is "" for RUN and a plain RESUME, "memory" for a
    memory-pressure PAUSE or HOLD, or "phantom:<name>" while a heavy Phantom job is the (co-)cause of the pause/hold.
    Resuming always needs no heavy Phantom job seen for at least PHANTOM_QUIET_S seconds. On top of that:
      memory_caused=True  (memory caused the pause, or joined it while paused; the default): decide()'s own
                          memory-recovery + min-pause condition (the RESUME hysteresis).
      memory_caused=False (only a Phantom job caused it; BC-37, owner decision 2026-10-01): memory must merely not be
                          below the PAUSE thresholds. macOS sizes its swap files so that free swap hovers around
                          0.7-1.7 GB; the 1,000 MB RESUME line held a run paused 25 min with 57 % of RAM free.
    """
    mem_action = decide(paused, swap_free_mb, free_pct, paused_for_s)
    if not paused:
        if mem_action == "PAUSE":
            return "PAUSE", "memory"
        if phantom_names:
            return "PAUSE", f"phantom:{phantom_names[0]}"
        return "RUN", ""
    # currently paused
    if phantom_names:
        return "HOLD", f"phantom:{phantom_names[0]}"
    if phantom_quiet_for_s < PHANTOM_QUIET_S:
        return "HOLD", "phantom-cooldown"
    if not memory_caused:
        return ("HOLD", "memory") if below_pause(swap_free_mb, free_pct) else ("RESUME", "")
    if mem_action == "RESUME":
        return "RESUME", ""
    return "HOLD", ""


def next_cause(paused: bool, memory_caused: bool, action: str, reason: str, swap_free_mb: float, free_pct: int) -> bool:
    """Whether the pause counts as memory-caused after this tick: a new pause takes its cause from its reason; while
    paused, memory falling below a PAUSE threshold joins the cause and stays joined; RUN / RESUME clear it."""
    if action == "PAUSE":
        return reason == "memory"
    if not paused or action == "RESUME":
        return False
    return memory_caused or below_pause(swap_free_mb, free_pct)


def resume_label(memory_caused: bool) -> str:
    return "memory" if memory_caused else "phantom-quiet"


def check_target(command: str) -> None:
    if any(k in command for k in PHANTOM_MARKERS):
        raise ValueError(f"refusing to signal a Phantom process: {command[:120]!r}")


def tree(root: int, parents: dict) -> list:
    """root plus all descendants, from a {pid: ppid} map."""
    out, frontier = [root], [root]
    while frontier:
        nxt = [p for p, pp in parents.items() if pp in frontier and p not in out]
        out += nxt
        frontier = nxt
    return out


def release_set(root: int, parents: dict, groups: dict) -> list:
    """root's tree plus every process in a process group whose leader is in that tree. The lock wrapper starts its
    command as a group leader; a stopped worker whose parent died is reparented out of the tree but stays in the group,
    and must be continued too. `groups` is {pid: pgid}."""
    t = tree(root, parents)
    leaders = set(t)
    return t + sorted(p for p, g in groups.items() if g in leaders and p not in leaders)


def release(root: int, parents: dict, groups: dict, command, kill=None) -> list:
    """SIGCONT the release set of `root` (refused whole if any member is a Phantom process); returns the pids signalled.
    A pid that has ended, or that we may not signal, is skipped."""
    kill = kill or os.kill
    pids = release_set(root, parents, groups)
    for p in pids:
        check_target(command(p))
    done = []
    for p in pids:
        try:
            kill(p, signal.SIGCONT)
        except (ProcessLookupError, PermissionError):
            continue
        done.append(p)
    return done


def _groups() -> dict:  # pragma: no cover - thin OS wrapper
    rows = subprocess.run(["ps", "-A", "-o", "pid=,pgid="], capture_output=True, text=True, check=True).stdout.split("\n")
    return {int(a): int(b) for a, b in (r.split() for r in rows if r.strip())}


def _parents() -> dict:  # pragma: no cover - thin OS wrapper
    rows = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, check=True).stdout.split("\n")
    return {int(a): int(b) for a, b in (r.split() for r in rows if r.strip())}


def _command(pid: int) -> str:  # pragma: no cover
    return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def _signal_tree(root: int, sig: int) -> list:  # pragma: no cover
    pids = tree(root, _parents())
    for p in pids:
        check_target(_command(p))
    for p in pids:
        try:
            os.kill(p, sig)
        except ProcessLookupError:
            pass
    return pids


def _ps_all() -> str:  # pragma: no cover - thin OS wrapper
    return subprocess.run(["ps", "-A", "-o", "pid=,command="], capture_output=True, text=True).stdout


def _ancestor_pids(pid: int) -> frozenset:  # pragma: no cover - thin OS wrapper
    """BC-29: walk the ppid chain upward from `pid` via `ps -o ppid=`, so mem_guard's own ancestor processes
    (whatever launched it -- a shell or another launcher) are never mistaken for a Phantom heavy job just
    because their own argv happens to mention a job name. Capped at 64 hops as a defensive bound."""
    out = set()
    cur = pid
    for _ in range(64):
        s = subprocess.run(["ps", "-o", "ppid=", "-p", str(cur)], capture_output=True, text=True).stdout.strip()
        if not s:
            break
        try:
            ppid = int(s)
        except ValueError:
            break
        if ppid <= 1 or ppid in out:
            break
        out.add(ppid)
        cur = ppid
    return frozenset(out)


class OsOps:
    """The real operating-system side of the guard: one method per question or action, so `run` can be tested with a
    scripted stand-in."""

    def alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:                      # BC-39: it exists; we may not signal it
            return True
        return True

    def swap_free_mb(self) -> float:  # pragma: no cover - thin OS wrapper
        return parse_swap_free_mb(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)

    def free_pct(self) -> int:  # pragma: no cover - thin OS wrapper
        return parse_pressure_free_pct(subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout)

    def tree_pids(self, root: int) -> list:  # pragma: no cover - thin OS wrapper
        return tree(root, _parents())

    def signal_tree(self, root: int, sig: int) -> list:  # pragma: no cover - thin OS wrapper
        return _signal_tree(root, sig)

    def signal_pids(self, pids, sig: int) -> list:
        """Signal each listed pid that still exists and returns those signalled. A pid recorded earlier may since belong to
        another process: a Phantom process is never signalled, and one we may not signal is skipped."""
        done = []
        for p in pids:
            try:
                check_target(_command(p))
            except ValueError:                       # a Phantom process: never signalled
                continue
            except OSError:                          # `ps` could not run (memory pressure): identity unknown, so only
                if sig != signal.SIGCONT:            # the harmless SIGCONT is still sent
                    continue
            try:
                os.kill(p, sig)
            except (ProcessLookupError, PermissionError):
                continue
            done.append(p)
        return done

    def hold_signals(self) -> None:  # pragma: no cover - thin OS wrapper (tested in a child process)
        """The guard is exiting: a further SIGTERM / SIGINT / SIGHUP must not interrupt the release."""
        for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(s, signal.SIG_IGN)

    def phantom_names(self, ignore_pids: frozenset) -> list:  # pragma: no cover - thin OS wrapper
        return phantom_heavy_alive(_ps_all(), ignore_pids)


def run(pid: int, log, pause_on_phantom: bool, ops, ignore_pids: frozenset, now=time.time, sleep=time.sleep) -> int:
    """The guard's loop: poll, decide, signal, until the target ends.

    BC-43: whatever ends the loop — the target exiting, an exception, a termination signal (main turns SIGTERM / SIGINT
    / SIGHUP into SystemExit) — every process this guard stopped and has not resumed gets SIGCONT before `run` returns or
    re-raises, and the log says so. Only SIGKILL cannot be handled; `--release PID` resumes a tree after that."""
    stopped: list = []       # pids this guard SIGSTOPped and has not resumed (captured at pause time: orphans included)
    try:
        return _loop(pid, log, pause_on_phantom, ops, ignore_pids, now, sleep, stopped)
    except SystemExit as e:  # a termination signal (main's handlers)
        log(f"MEMGUARD EXIT code={e.code} (stopped by a signal)")
        raise
    except BaseException as e:
        log(f"MEMGUARD ERROR {type(e).__name__}: {e}")
        raise
    finally:
        if stopped:
            ops.hold_signals()                       # a second signal must not cut the release short
            got = ops.signal_pids(stopped, signal.SIGCONT)
            missed = sorted(set(stopped) - set(got))
            log(f"MEMGUARD RELEASED pids={got} (the guard is exiting; no process it stopped stays stopped)"
                + (f"; not signalled (ended, or the pid now belongs to another process): {missed}" if missed else ""))


def _pause(pid, ops, stopped: list) -> list:
    """SIGSTOP the tree. The tree is recorded before signalling, so a partial pause is released too."""
    stopped[:] = ops.tree_pids(pid)
    pids = ops.signal_tree(pid, signal.SIGSTOP)
    stopped[:] = sorted(set(stopped) | set(pids))
    return pids


def _resume(pid, ops, stopped: list) -> list:
    """SIGCONT the tree and every process stopped at pause time that has left it since (a child whose parent ended
    during the pause is no longer in the tree, and would otherwise stay stopped)."""
    pids = ops.signal_tree(pid, signal.SIGCONT)
    left = sorted(set(stopped) - set(pids))
    if left:
        pids = list(pids) + ops.signal_pids(left, signal.SIGCONT)
    stopped.clear()
    return pids


def _loop(pid, log, pause_on_phantom, ops, ignore_pids, now, sleep, stopped: list) -> int:
    paused, since, last_note = False, 0.0, 0.0
    mem_cause = False        # BC-37: did memory cause (or join) the current pause? decides the resume rule
    phantom_last_seen = 0.0  # 0.0 sentinel: never seen -> "quiet for" is huge (now - 0.0), never blocks resume
    while True:
        if not ops.alive(pid):
            if paused:
                log("MEMGUARD target gone while paused")
            log("MEMGUARD END target exited")
            return 0
        sw = ops.swap_free_mb()
        pct = ops.free_pct()
        t = now()
        if not pause_on_phantom:
            act = decide(paused, sw, pct, t - since if paused else 0.0)
            if act == "PAUSE":
                pids = _pause(pid, ops, stopped)
                paused, since, last_note = True, t, t
                log(f"MEMGUARD PAUSED pids={pids} swap_free={sw:.0f}M free={pct}%")
            elif act == "RESUME":
                pids = _resume(pid, ops, stopped)
                log(f"MEMGUARD RESUMED after {t - since:.0f}s pids={pids} swap_free={sw:.0f}M free={pct}%")
                paused = False
            elif paused and t - last_note >= 600:
                log(f"MEMGUARD STILL PAUSED {t - since:.0f}s swap_free={sw:.0f}M free={pct}%")
                last_note = t
        else:
            phantom_names = ops.phantom_names(ignore_pids)
            if phantom_names:
                phantom_last_seen = t
            act, reason = decide_with_phantom(paused, sw, pct, t - since if paused else 0.0,
                                               phantom_names, t - phantom_last_seen, memory_caused=mem_cause)
            was_mem = mem_cause
            mem_cause = next_cause(paused, mem_cause, act, reason, sw, pct)
            if act == "PAUSE":
                pids = _pause(pid, ops, stopped)
                paused, since, last_note = True, t, t
                log(f"MEMGUARD PAUSED reason={reason} pids={pids} swap_free={sw:.0f}M free={pct}%")
            elif act == "RESUME":
                pids = _resume(pid, ops, stopped)
                log(f"MEMGUARD RESUMED after {t - since:.0f}s reason={resume_label(was_mem)} "
                    f"pids={pids} swap_free={sw:.0f}M free={pct}%")
                paused = False
            elif paused and t - last_note >= 600:
                log(f"MEMGUARD STILL PAUSED reason={reason} {t - since:.0f}s swap_free={sw:.0f}M free={pct}%")
                last_note = t
        sleep(POLL_S)


def _exit_on_signal(signum, frame):
    """SIGTERM / SIGINT / SIGHUP end the guard through SystemExit, so run()'s release still happens (BC-43)."""
    raise SystemExit(128 + signum)


def main(argv=None) -> int:  # pragma: no cover - wiring; the loop is run(), tested with a scripted OS layer
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int)
    ap.add_argument("--log")
    ap.add_argument("--release", type=int, metavar="PID",
                    help="SIGCONT the process tree of PID, and every member of a process group one of them leads, and exit "
                         "(after a guard was killed with SIGKILL)")
    ap.add_argument("--restart", action="store_true",
                    help="this guard replaces one that died during the same run: the log says RESTART, not START, so "
                         "the run's earlier lines stay part of it")
    ap.add_argument("--pause-on-phantom", action="store_true",
                     help="also pause while a heavy Phantom job (PHANTOM_HEAVY_JOBS) is alive; "
                          "resume needs it absent for PHANTOM_QUIET_S seconds; the memory resume thresholds apply only "
                          "when memory caused or joined the pause (else memory must not be below the pause thresholds)")
    a = ap.parse_args(argv)
    if a.release is not None:
        pids = release(a.release, _parents(), _groups(), _command)
        print(f"mem_guard: released pids={pids}")
        return 0
    if a.pid is None or a.log is None:
        ap.error("--pid and --log are required (or --release PID)")
    check_target(_command(a.pid))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)

    def log(msg):
        with open(a.log, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    # BC-29: also exclude our own ancestor pid chain (whatever launched this guard), never just os.getpid()/a.pid.
    ignore_pids = frozenset({os.getpid(), a.pid}) | _ancestor_pids(os.getpid())
    log(f"MEMGUARD {'RESTART' if a.restart else 'START'} pid={a.pid} pause<{PAUSE_SWAP_MB:.0f}M swap or <{PAUSE_FREE_PCT}% free; "
        f"resume>={RESUME_SWAP_MB:.0f}M and >={RESUME_FREE_PCT}% after {MIN_PAUSE_S}s"
        + (f"; pause-on-phantom quiet>={PHANTOM_QUIET_S}s" if a.pause_on_phantom else ""))
    return run(a.pid, log, a.pause_on_phantom, OsOps(), ignore_pids)        # START is logged just before guarding


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
