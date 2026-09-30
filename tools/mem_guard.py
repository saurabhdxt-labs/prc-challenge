"""Pause OUR long run on machine memory pressure, resume on recovery (owner ruling 2026-09-24).

    /opt/homebrew/bin/python3.11 -B tools/mem_guard.py --pid <our run's pid> --log logs/<run>_memguard.log

Every POLL_S seconds: read free swap (sysctl vm.swapusage) and the system-wide free percentage (memory_pressure).
Below either pause threshold -> SIGSTOP the run's whole process tree; once BOTH recover past the resume thresholds
(hysteresis) and the pause has lasted MIN_PAUSE_S -> SIGCONT. Never signals a Phantom process. Exits when the run ends.
One line per state change goes to the log (for a Monitor), plus a STILL PAUSED line every 10 minutes.

Opt-in (--pause-on-phantom, owner ruling 2026-09-28: "keep watching the memory when phantom starts just pause and
resume"): ALSO pause our run while any Phantom HEAVY job (PHANTOM_HEAVY_JOBS) is alive, and resume only once none has
been alive for PHANTOM_QUIET_S seconds AND memory is past the existing resume thresholds. Off by default; with the
flag unset, behaviour is unchanged byte-for-byte.
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
RESUME_SWAP_MB = 1000.0
RESUME_FREE_PCT = 15
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


def decide_with_phantom(paused: bool, swap_free_mb: float, free_pct: int, paused_for_s: float,
                         phantom_names: list, phantom_quiet_for_s: float) -> tuple:
    """decide() plus the --pause-on-phantom overlay. Returns (action, reason).

    action is RUN/PAUSE/HOLD/RESUME as in decide(). reason is "" for RUN and a plain RESUME, "memory" for a
    memory-pressure PAUSE, or "phantom:<name>" while a heavy Phantom job is the (co-)cause of the pause/hold.
    Resuming from a pause needs BOTH: decide()'s own memory-recovery+min-pause condition, AND no heavy Phantom
    job seen for at least PHANTOM_QUIET_S seconds -- whichever reason originally caused the pause.
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
    if mem_action == "RESUME":
        return "RESUME", ""
    return "HOLD", ""


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
    (whatever launched it -- a shell, an agent harness) are never mistaken for a Phantom heavy job just
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


def main(argv=None) -> int:  # pragma: no cover - process loop; decisions and parsers above are tested
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--pause-on-phantom", action="store_true",
                     help="also pause while a heavy Phantom job (PHANTOM_HEAVY_JOBS) is alive; "
                          "resume needs it absent for PHANTOM_QUIET_S seconds AND memory past the resume thresholds")
    a = ap.parse_args(argv)
    check_target(_command(a.pid))

    def log(msg):
        with open(a.log, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

    log(f"MEMGUARD START pid={a.pid} pause<{PAUSE_SWAP_MB:.0f}M swap or <{PAUSE_FREE_PCT}% free; "
        f"resume>={RESUME_SWAP_MB:.0f}M and >={RESUME_FREE_PCT}% after {MIN_PAUSE_S}s"
        + (f"; pause-on-phantom quiet>={PHANTOM_QUIET_S}s" if a.pause_on_phantom else ""))
    paused, since, last_note = False, 0.0, 0.0
    phantom_last_seen = 0.0  # 0.0 sentinel: never seen -> "quiet for" is huge (now - 0.0), never blocks resume
    # BC-29: also exclude our own ancestor pid chain (whatever launched this guard), never just os.getpid()/a.pid.
    ignore_pids = frozenset({os.getpid(), a.pid}) | _ancestor_pids(os.getpid())
    while True:
        try:
            os.kill(a.pid, 0)
        except ProcessLookupError:
            if paused:
                log("MEMGUARD target gone while paused")
            log("MEMGUARD END target exited")
            return 0
        sw = parse_swap_free_mb(subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout)
        pct = parse_pressure_free_pct(subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout)
        now = time.time()
        if not a.pause_on_phantom:
            act = decide(paused, sw, pct, now - since if paused else 0.0)
            if act == "PAUSE":
                pids = _signal_tree(a.pid, signal.SIGSTOP)
                paused, since, last_note = True, now, now
                log(f"MEMGUARD PAUSED pids={pids} swap_free={sw:.0f}M free={pct}%")
            elif act == "RESUME":
                pids = _signal_tree(a.pid, signal.SIGCONT)
                log(f"MEMGUARD RESUMED after {now - since:.0f}s pids={pids} swap_free={sw:.0f}M free={pct}%")
                paused = False
            elif paused and now - last_note >= 600:
                log(f"MEMGUARD STILL PAUSED {now - since:.0f}s swap_free={sw:.0f}M free={pct}%")
                last_note = now
        else:
            phantom_names = phantom_heavy_alive(_ps_all(), ignore_pids)
            if phantom_names:
                phantom_last_seen = now
            act, reason = decide_with_phantom(paused, sw, pct, now - since if paused else 0.0,
                                               phantom_names, now - phantom_last_seen)
            if act == "PAUSE":
                pids = _signal_tree(a.pid, signal.SIGSTOP)
                paused, since, last_note = True, now, now
                log(f"MEMGUARD PAUSED reason={reason} pids={pids} swap_free={sw:.0f}M free={pct}%")
            elif act == "RESUME":
                pids = _signal_tree(a.pid, signal.SIGCONT)
                log(f"MEMGUARD RESUMED after {now - since:.0f}s reason={reason or 'memory'} "
                    f"pids={pids} swap_free={sw:.0f}M free={pct}%")
                paused = False
            elif paused and now - last_note >= 600:
                log(f"MEMGUARD STILL PAUSED reason={reason} {now - since:.0f}s swap_free={sw:.0f}M free={pct}%")
                last_note = now
        time.sleep(POLL_S)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
