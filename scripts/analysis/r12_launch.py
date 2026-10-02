"""Detached launcher for the R12 heavy steps, on v34_launch's pattern (its tested pure helpers are imported, not copied):
its own session, a monitor-friendly log, preflight refusals, a PID file, the runner's in-process RSS self-abort, an RSS
watchdog that kills ONLY its own child tree, and a memory-pressure guard (tools/mem_guard.py) beside every run.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/r12_launch.py --step features --detach -- \
        features --store data/r12/features_<stamp> --final
    /opt/homebrew/bin/python3.11 -B scripts/analysis/r12_launch.py --step tables --detach -- \
        tables --store data/r12/features_<stamp> --out data/r12/tables_<stamp>
    /opt/homebrew/bin/python3.11 -B scripts/analysis/r12_launch.py --step val --detach -- \
        --split holdout --arms r12,control,yhead,r12_noeobt --tables data/r12/tables_<stamp> --out data/r12/val_<...>
    add --smoke (paths under logs/r12_scratch, --days for the build steps, the validator's --smoke) for a < 5 min run.

Preflight (refuses, never waits): AC power; not inside the machine's reserved window 11:40-12:15 at START; the heavy
lock free (real runs; probed nonblocking, released at once, the runner takes it); no other R12 or v34 launcher alive;
the A12 fetch (scripts/adsb_fetch.py run) not alive — except a non-final feature build, which defers unfinished days.
There is NO wall-time kill (a mem_guard-paused run is never killed for taking long); the only kill is the RSS watchdog.
Monitor with: tail -f <log>; kill -0 <pid> (never pgrep -f).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import importlib.util
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY = "/opt/homebrew/bin/python3.11"
SCRATCH, REAL = ROOT / "logs" / "r12_scratch", ROOT / "data" / "r12"     # smoke / real outputs
LOG_DIR = SCRATCH                         # launch logs and PID files
PATH_FLAGS = ("--store", "--out", "--tables", "--val", "--stack", "--f26")


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load("v34_launch", "scripts/analysis/v34_launch.py")


@dataclasses.dataclass(frozen=True)
class StepSpec:
    runner: str
    rss_abort_gib: float          # the runner's in-process self-abort (exit 70), passed as --rss-limit-gib
    watchdog_gib: float           # this launcher kills the runner's tree above this CURRENT tree RSS
    expected: str                 # wall / memory, from the smoke measurements


STEPS = {
    # smoke 2026-09-25: 3 days in 19 s (5-7 s/day), runner peak 1.77 GiB
    "features": StepSpec("scripts/analysis/r12_build.py", 5.0, 6.0,
                         "~45-75 min for 427 days (smoke 5-7 s/day), ~2.5-3.5 GiB (smoke 1.77 GiB + the full universe)"),
    # smoke: 13,834 rows in 2 s, 0.48 GiB; the full 2.40 M rows read 427 day files
    "tables": StepSpec("scripts/analysis/r12_build.py", 5.0, 6.0, "~3-8 min, ~2-3 GiB"),
    # full-frame dry run (2.40 M x 125, 4 arms, 60 trees): peak memory footprint 6.27 GiB, ru_maxrss 7.33 GiB; the real
    # run adds the full tables (~1 GiB) and 30k-tree boosters (~1 GiB). v34 FW refit: ES 36 min + 3 x 35 min seeds.
    "val": StepSpec("scripts/analysis/r12_validate.py", 11.0, 12.0,
                    "~2.5-3 h per arm at 3 seeds (control ~2.4 h); ~9-9.5 GiB (dry run 7.33 GiB + tables + boosters)"),
    # the ADS-B residual stack re-trained on an R12 F (models_v4 train 101 s / 0.93 GB) + incumbent + composition
    "stack": StepSpec("scripts/analysis/r12_stack.py", 6.0, 7.0, "~5-10 min per arm (see the gate log), ~2-3 GiB"),
}


def spec(step: str) -> StepSpec:
    if step not in STEPS:
        raise ValueError(f"no R12 step {step!r}; steps are {sorted(STEPS)}")
    return STEPS[step]


def child_cmd(step: str, runner_args, smoke: bool) -> list:
    s = spec(step)
    runner_args = list(runner_args)
    if "--rss-limit-gib" in runner_args:
        raise ValueError("--rss-limit-gib is the launcher's to set")
    return ([PY, "-B", str(ROOT / s.runner), *runner_args, "--rss-limit-gib", f"{s.rss_abort_gib:g}"]
            + (["--smoke"] if smoke and step in ("val", "stack") else []))


def fetch_alive(ps_text: str) -> bool:
    """`ps -A -o pid=,command=`: is a python process running `scripts/adsb_fetch.py run`? (A shell whose command line
    merely quotes the name, e.g. a watcher loop, does not count.)"""
    for line in ps_text.splitlines():
        tok = line.split()
        if len(tok) < 3 or "python" not in pathlib.Path(tok[1]).name.lower():
            continue
        for i, t in enumerate(tok[2:-1], start=2):
            if t.endswith("scripts/adsb_fetch.py") and tok[i + 1] == "run":
                return True
    return False


def preflight(pmset: str, now: dt.datetime, lock_free: bool, others: bool, fetch: bool, step: str,
              final: bool, smoke: bool = False) -> list:
    """Every reason to refuse; empty = go. A smoke (< 5 min, scratch outputs, no lock) may run beside the fetch."""
    reasons = []
    if not V.on_ac_power(pmset):
        reasons.append("not on AC power")
    if V.in_phantom_window(now):
        reasons.append("inside Phantom's track_record window 11:40-12:15")
    if not lock_free:
        reasons.append("heavy-compute lock is held by another job")
    if others:
        reasons.append("another R12 or v34 launcher is alive")
    if fetch and not smoke and (step != "features" or final):
        reasons.append("the A12 fetch is still running (only a non-final, resumable feature build may run beside it)")
    return reasons


def _paths(args) -> list:
    args = list(args)
    return [args[i + 1] for i, a in enumerate(args[:-1]) if a in PATH_FLAGS]


def check_paths(args, smoke: bool) -> None:
    """Smoke outputs (and the smoke tables a smoke validation reads) under logs/r12_scratch; real ones under data/r12."""
    paths = _paths(args)
    if not any(a in ("--store", "--out") for a in args):
        raise ValueError("no output path (--store / --out) in the runner arguments")
    root = SCRATCH if smoke else REAL
    for p in paths:
        q = (ROOT / p).resolve() if not pathlib.Path(p).is_absolute() else pathlib.Path(p).resolve()
        if not (q == root.resolve() or root.resolve() in q.parents):
            mode = "smoke" if smoke else "real"
            raise PermissionError(f"{mode} run path {p} must be under {root.relative_to(ROOT)}")


def check_features_mode(args, smoke: bool, partial: bool) -> None:
    """A real feature build is --final unless the launcher is told --partial; a smoke build names its --days."""
    args = list(args)
    if not args or args[0] not in ("features", "tables"):
        return
    if smoke and "--days" not in args:
        raise ValueError("a smoke build must name its --days")
    if args[0] == "features" and not smoke and "--final" not in args and not partial:
        raise ValueError("a real feature build must be --final (every missing day counted), or pass --partial")


def all_pidfiles() -> list:
    return [LOG_DIR / f"launch_r12_{s}.pid" for s in STEPS] + [V.pidfile(n) for n in V.STEPS]


def log_path(step: str, smoke: bool, now: dt.datetime) -> pathlib.Path:
    return LOG_DIR / f"r12_{step}_{'smoke' if smoke else 'run'}_{now:%Y%m%dT%H%M%S}.log"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", required=True, choices=sorted(STEPS))
    ap.add_argument("--detach", action="store_true", help="re-launch in its own session and return at once")
    ap.add_argument("--smoke", action="store_true", help="a < 5 min run under logs/r12_scratch, no lock")
    ap.add_argument("--partial", action="store_true", help="allow a real, non-final feature build")
    ap.add_argument("--log", default=None, help=argparse.SUPPRESS)
    ap.add_argument("runner_args", nargs=argparse.REMAINDER, help="after --: the runner's own arguments")
    a = ap.parse_args(argv)
    if a.runner_args and a.runner_args[0] == "--":
        a.runner_args = a.runner_args[1:]
    return a


def main(argv=None) -> int:  # pragma: no cover - process orchestration; the pure parts above are tested
    a = parse_args(argv)
    check_paths(a.runner_args, a.smoke)
    check_features_mode(a.runner_args, a.smoke, a.partial)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if a.detach:
        lp = log_path(a.step, a.smoke, dt.datetime.now())
        args = ([PY, "-B", str(pathlib.Path(__file__).resolve()), "--step", a.step, "--log", str(lp)]
                + (["--smoke"] if a.smoke else []) + (["--partial"] if a.partial else []) + ["--", *a.runner_args])
        p = subprocess.Popen(args, cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"launched pid {p.pid}; log {lp}; mem_guard log {V.memguard_log(lp)}")
        return 0
    if not a.log:
        raise SystemExit("use --detach (the foreground form is the detached child's)")
    return _run(a.step, a.smoke, a.runner_args, pathlib.Path(a.log))


def _lock_free() -> bool:  # pragma: no cover - touches the real lock
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    import run_guards as G
    try:
        fh = G.acquire_lock()
    except RuntimeError:
        return False
    G.release_lock(fh)
    return True


def _run(step: str, smoke: bool, runner_args, log_path_: pathlib.Path) -> int:  # pragma: no cover
    s = spec(step)
    with open(log_path_, "a") as fh:
        def log(msg):
            fh.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
            fh.flush()
        pm = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
        swap = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
        ps = subprocess.run(["ps", "-A", "-o", "pid=,command="], capture_output=True, text=True).stdout
        reasons = preflight(pm, dt.datetime.now(), True if smoke else _lock_free(),
                            V.others_alive(all_pidfiles(), V.pid_alive), fetch_alive(ps), step,
                            "--final" in runner_args, smoke=smoke)
        log(f"R12 {step} ({'smoke' if smoke else 'run'}); limits self-abort {s.rss_abort_gib:g} GiB / watchdog "
            f"{s.watchdog_gib:g} GiB; expected {s.expected}")
        log(f"preflight: {swap}; {'OK' if not reasons else 'REFUSED: ' + '; '.join(reasons)}")
        if reasons:
            return 75
        pf = LOG_DIR / f"launch_r12_{step}.pid"
        pf.write_text(str(os.getpid()))
        try:
            cmd = child_cmd(step, runner_args, smoke)
            log(f"START {' '.join(cmd)}")
            t0 = time.time()
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
            g = V.memguard_cmd(p.pid, log_path_)
            subprocess.Popen(g, cwd=ROOT, start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"mem_guard: {' '.join(g)}")

            def rss_of(pid):
                out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
                return V.tree_rss_gib(pid, V.parse_ps(out))
            rc, peak, killed = V.watch(p, rss_of, log, s.watchdog_gib)
            log(f"END rc={rc} wall={time.time() - t0:.0f}s (includes any mem_guard pause) "
                f"watchdog_peak_tree_rss={peak:.2f}GiB killed={killed}")
            log("R12 STEP FINISHED" if rc == 0 else "STOPPING: step failed")
            return rc
        finally:
            pf.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
