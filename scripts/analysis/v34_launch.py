"""Detached launcher for the seven v34 no-seconds steps (plans/PREREG_nsc_no_seconds_2026_09_24.md §7, AMENDMENTS V34.1-V34.7),
on the NSC launcher's pattern (scripts/analysis/nsc_launch.py): its own session, a monitor-friendly log, preflight refusals,
a PID file, the runner's in-process RSS self-abort, an RSS watchdog that kills ONLY its own child tree, and the owner's
memory-pressure guard (tools/mem_guard.py) started beside every run.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/v34_launch.py --step N --detach            # the real step N
    /opt/homebrew/bin/python3.11 -B scripts/analysis/v34_launch.py --step N --detach --smoke    # its < 5 min smoke, no lock

Preflight (refuses, never waits): AC power; not inside Phantom's daily track_record window (11:40-12:15 local) at START;
the heavy lock free (probed nonblocking and released at once; the runner takes it for the run; smoke skips it); no other
v34 launcher alive (one heavy job at a time). RSS limits per step (STEPS, passed explicitly to the runner as
--rss-limit-gib): 6 GiB self-abort / 7 GiB watchdog by default; 8 / 9 GiB for steps 3 and 4 (owner-approved exception:
FW-class 12-month refits measured 7.53 GB peak, data/fwp_build/fw/merry-quicksand_v6.meta.json); 11 / 12 GiB for step 6
(owner-approved, V34.6b: CBV's peak memory footprint 10.5 GiB; its ru_maxrss 7.03 understated it). The watchdog sums the
RSS of the runner AND its descendants (the ADS-B NN worker, the CatBoost venv worker), polled with `ps`.

mem_guard: `tools/mem_guard.py --pid <runner pid> --log <run log stem>_memguard.log`, spawned detached right after the runner
starts; it SIGSTOPs the runner's tree under memory pressure and SIGCONTs on recovery, and exits when the runner ends.
There is NO wall-time kill anywhere in this launcher (a SIGSTOP-paused run is never killed for taking long); the only kill is
the RSS watchdog. Monitor with: tail -f <log>; kill -0 <pid> (never pgrep -f).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = sys.executable                 # the interpreter running this tool runs its children
LOG_DIR = ROOT / "logs" / "nsc_v34_scratch"
MEM_GUARD = ROOT / "tools" / "mem_guard.py"
POLL_S = 20.0
PHANTOM_WINDOW = (dt.time(11, 40), dt.time(12, 15))


@dataclasses.dataclass(frozen=True)
class StepSpec:
    runner: str
    args: tuple
    rss_abort_gib: float          # the runner's in-process self-abort (exit 70), passed as --rss-limit-gib
    watchdog_gib: float           # this launcher kills the runner's tree above this CURRENT tree RSS
    expected: str                 # wall / RSS from the recorded runs (the amendments carry the sources)


STEPS = {
    1: StepSpec("scripts/analysis/v34_f_fold.py", (), 6.0, 7.0, "~1 h 55 min-2 h 30 min, ~4.9 GB (arm F fold 6,756 s / 4.84 GB)"),
    2: StepSpec("scripts/analysis/v34_adsb_stack.py", (), 6.0, 7.0, "~5-10 min, ~1.5 GB (models_v4 train 101 s / 0.93 GB)"),
    3: StepSpec("scripts/analysis/v34_refit.py", ("--design", "F"), 8.0, 9.0, "~2 h 10 min-2 h 45 min, ~5.2 GB (v6 F refit 7,893 s / 5.13 GB)"),
    4: StepSpec("scripts/analysis/v34_refit.py", ("--design", "FW"), 8.0, 9.0, "~3 h-3 h 30 min, ~7.5 GB (FWP refit 10,833 s / 7.53 GB)"),
    5: StepSpec("scripts/analysis/v34_refit.py", ("--design", "ORL"), 6.0, 7.0, "~2 h 5 min-2 h 40 min, ~4.7 GiB (ORL refit 7,593 s / 4.70 GiB)"),
    6: StepSpec("scripts/analysis/v34_catboost.py", (), 11.0, 12.0, "~2 h 35 min-3 h, ~10.5 GiB footprint (CBV worker 9,329 s; peak memory footprint 11.3 GB = 10.5 GiB, ru_maxrss 7.03); owner-approved 11/12 GiB, amendment V34.6b)"),
    7: StepSpec("scripts/analysis/v34_compose.py", (), 6.0, 7.0, "~5-15 min, ~3 GB (measured by the replay smoke)"),
}


def spec(step: int) -> StepSpec:
    if step not in STEPS:
        raise ValueError(f"no v34 step {step!r}; steps are {sorted(STEPS)}")
    return STEPS[step]


def on_ac_power(pmset_text: str) -> bool:
    """`pmset -g batt` must say it draws from AC Power (a long run on battery can be cut short)."""
    return "'AC Power'" in pmset_text


def in_phantom_window(now: dt.datetime) -> bool:
    lo, hi = PHANTOM_WINDOW
    return lo <= now.time() <= hi


def child_cmd(step: int, smoke: bool) -> list:
    s = spec(step)
    return ([PY, "-B", str(ROOT / s.runner), *s.args, "--rss-limit-gib", f"{s.rss_abort_gib:g}"]
            + (["--smoke"] if smoke else []))


def memguard_log(run_log) -> pathlib.Path:
    """<run log stem>_memguard.log beside the run log."""
    p = pathlib.Path(run_log)
    return p.with_name(p.stem + "_memguard.log")


def memguard_cmd(pid: int, run_log) -> list:
    if not isinstance(pid, int) or pid <= 1:
        raise ValueError(f"mem_guard needs the runner's pid, got {pid!r}")
    # --pause-on-phantom: owner ruling 2026-09-28 ("when phantom starts just pause"), on every launched heavy run
    return [PY, "-B", str(MEM_GUARD), "--pid", str(pid), "--log", str(memguard_log(run_log)), "--pause-on-phantom"]


def start_memguard(pid: int, run_log, restart: bool = False, popen=subprocess.Popen):
    """Start mem_guard beside the run, detached, with its own output (a refusal, a traceback) appended to its log so a
    guard that could not run leaves a trace. `restart` marks a guard that replaces one that died during the same run.
    Returns (command, process)."""
    g = memguard_cmd(pid, run_log) + (["--restart"] if restart else [])
    with open(memguard_log(run_log), "a") as out:
        proc = popen(g, cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
    return g, proc


def parse_ps(ps_text: str) -> dict:
    """`ps -A -o pid=,ppid=,rss=` -> {pid: (ppid, rss_kib)}; malformed lines are skipped."""
    out = {}
    for line in ps_text.splitlines():
        parts = line.split()
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            continue
        pid, ppid, rss = (int(p) for p in parts)
        out[pid] = (ppid, rss)
    return out


def tree_rss_gib(root: int, table: dict) -> float:
    """RSS of `root` and every descendant, GiB (ps reports KiB). A gone root is 0."""
    if root not in table:
        return 0.0
    members, frontier = {root}, [root]
    while frontier:
        nxt = [p for p, (pp, _) in table.items() if pp in frontier and p not in members]
        members.update(nxt)
        frontier = nxt
    return sum(table[p][1] for p in members) / 1024 ** 2


def preflight(pmset_text: str, now: dt.datetime, lock_free: bool, other_alive: bool) -> list:
    """Every reason to refuse; empty = go."""
    reasons = []
    if not on_ac_power(pmset_text):
        reasons.append("not on AC power")
    if in_phantom_window(now):
        reasons.append("inside Phantom's track_record window 11:40-12:15")
    if not lock_free:
        reasons.append("heavy-compute lock is held by another job")
    if other_alive:
        reasons.append("another v34 launcher is alive")
    return reasons


def watch(proc, rss_of, log, kill_gib: float, poll_s: float = POLL_S, sleep=time.sleep) -> tuple:
    """Poll the runner tree's RSS until it exits; kill it (only it) above kill_gib. There is deliberately no wall-time
    limit: mem_guard may SIGSTOP the tree for as long as memory stays short. Returns (returncode, peak GiB, killed)."""
    peak, killed = 0.0, False
    while proc.poll() is None:
        r = rss_of(proc.pid)
        peak = max(peak, r)
        if r > kill_gib:
            proc.kill()
            killed = True
            log(f"WATCHDOG KILL: tree RSS {r:.2f} GiB > {kill_gib}")
            break
        sleep(poll_s)
    return proc.wait(), peak, killed


def pidfile(step: int) -> pathlib.Path:
    spec(step)
    return LOG_DIR / f"launch_v34_step{step}.pid"


def others_alive(pidfiles, alive) -> bool:
    return any(alive(p) for p in pidfiles)


def _lock_free() -> bool:  # pragma: no cover - touches the real lock
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_guards as G
    try:
        fh = G.acquire_lock()
    except RuntimeError:
        return False
    G.release_lock(fh)
    return True


def pid_alive(pf: pathlib.Path) -> bool:
    """True when the PID in `pf` names a live process. A missing or unreadable file, or no such process, is False; a
    process we may not signal (PermissionError: it exists, owned by another user) is alive."""
    try:
        os.kill(int(pf.read_text().strip()), 0)
    except PermissionError:
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError):
        return False
    return True


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, required=True, choices=sorted(STEPS))
    ap.add_argument("--detach", action="store_true", help="re-launch in its own session and return at once")
    ap.add_argument("--smoke", action="store_true", help="the step's smoke (no lock, < 5 min)")
    ap.add_argument("--log", default=None, help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None) -> int:  # pragma: no cover - process orchestration; the pure parts above are tested
    a = parse_args(argv)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if a.detach:
        tag = "smoke" if a.smoke else "run"
        log_path = LOG_DIR / f"v34_step{a.step}_{tag}_{dt.datetime.now():%Y%m%dT%H%M%S}.log"
        args = [PY, "-B", str(pathlib.Path(__file__).resolve()), "--step", str(a.step), "--log", str(log_path)] + (
            ["--smoke"] if a.smoke else [])
        p = subprocess.Popen(args, cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"launched pid {p.pid}; log {log_path}; mem_guard log {memguard_log(log_path)}")
        return 0
    if not a.log:
        raise SystemExit("use --detach (the foreground form is the detached child's)")
    return _run(a.step, a.smoke, pathlib.Path(a.log))


def _run(step: int, smoke: bool, log_path: pathlib.Path) -> int:  # pragma: no cover
    s = spec(step)
    with open(log_path, "a") as fh:
        def log(msg):
            fh.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
            fh.flush()
        pm = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        swap = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
        reasons = preflight(pm, dt.datetime.now(), True if smoke else _lock_free(),
                            others_alive([pidfile(n) for n in STEPS], pid_alive))
        log(f"step {step} ({'smoke' if smoke else 'run'}); limits self-abort {s.rss_abort_gib:g} GiB / watchdog "
            f"{s.watchdog_gib:g} GiB; expected {s.expected}")
        log(f"preflight: {swap}; {'OK' if not reasons else 'REFUSED: ' + '; '.join(reasons)}")
        if reasons:
            return 75
        pf = pidfile(step)
        pf.write_text(str(os.getpid()))
        try:
            cmd = child_cmd(step, smoke)
            log(f"START {' '.join(cmd)}")
            t0 = time.time()
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            g, _ = start_memguard(p.pid, log_path)
            log(f"mem_guard: {' '.join(g)}")

            def rss_of(pid):
                ps = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
                return tree_rss_gib(pid, parse_ps(ps))
            rc, peak, killed = watch(p, rss_of, log, s.watchdog_gib)
            log(f"END rc={rc} wall={time.time() - t0:.0f}s (includes any mem_guard pause) "
                f"watchdog_peak_tree_rss={peak:.2f}GiB killed={killed}")
            log("ALL STEPS FINISHED" if rc == 0 else "STOPPING: step failed")
            return rc
        finally:
            pf.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
