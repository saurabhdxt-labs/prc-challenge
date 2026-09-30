"""Run-time guards for session-launched compute: the machine-wide heavy-compute lock and a peak-RSS abort.

Standard library only (fcntl / resource / threading), so any runner can import it regardless of which native model
family it loads (bug class BC-5: one OpenMP runtime per process). The idiom was copied verbatim into
scripts/analysis/ags_f2026.py:135-174, ags_build.py and mst_gate.py before this module existed; those copies are frozen by
their registrations' runner hashes and are left as they are. New runners import this.

Lock: a non-blocking fcntl flock on PHANTOM's lock file (scripts/run_with_heavy_lock.py uses the same path and the same
`PHANTOM_HEAVY_LOCK_PATH` override). A held lock REFUSES — never waits, never kills the holder (2026-08-14 lesson).
RSS guard: a daemon thread polling this process's peak resident set; above the limit it exits the process with status 70.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import resource
import sys
import threading

LOCK = Path.home() / ".phantom" / "phantom_heavy_compute.lock"
RSS_LIMIT_6GIB = 6 * 1024 ** 3


def lock_path() -> Path:
    """The lock file: `PHANTOM_HEAVY_LOCK_PATH` if set, else PHANTOM's default."""
    return Path(os.environ.get("PHANTOM_HEAVY_LOCK_PATH", LOCK))


def acquire_lock(path=None):
    """Take the heavy-compute lock, nonblocking. Returns the open file handle (keep it: closing releases the lock)."""
    path = Path(lock_path() if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise RuntimeError(f"heavy-compute lock {path} is held by another job; not stacking") from None
    return fh


def release_lock(fh) -> None:
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    fh.close()


def peak_rss(getrusage=resource.getrusage, platform: str = sys.platform) -> int:
    """This process's peak resident set in bytes (ru_maxrss is bytes on macOS, KiB on Linux)."""
    r = getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if platform == "darwin" else r * 1024)


def _abort_on_rss(rss: int, limit: int) -> None:
    print(f"ABORT: peak RSS {rss / 1024 ** 3:.2f} GiB is above the {limit / 1024 ** 3:.2f} GiB limit", file=sys.stderr,
          flush=True)
    os._exit(70)


def start_rss_guard(limit: int = RSS_LIMIT_6GIB, interval: float = 2.0, getrss=peak_rss, on_breach=None):
    """A daemon thread polling `getrss` every `interval` s; above `limit` it calls `on_breach(rss)` once (default:
    exit the process with status 70). Returns (thread, stop) where `stop()` ends the thread quietly."""
    breach = on_breach if on_breach is not None else (lambda r: _abort_on_rss(r, limit))
    stop = threading.Event()

    def run():
        while not stop.is_set():
            r = getrss()
            if r > limit:
                breach(r)
                return
            stop.wait(interval)

    t = threading.Thread(target=run, name="rss-guard", daemon=True)
    t.start()
    return t, stop.set


def sha256_file(path) -> str:
    """sha256 of a file's bytes, streamed (never loads the whole file into a string)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assert_prereg_registered(prereg, hash_log) -> str:
    """A registration under this project's protocol is append-only (a RESULT, an ERRATUM, a Status each get
    appended over the file's life), so its CURRENT sha256 legitimately moves and can never be a value pinned
    once, at build time, into an equality check -- that pin goes stale the moment anything is appended (BC-25;
    mxl_build.py pinned `REGISTERED_SHA` and mxi_build.py's shared `preflight` compared a freshly-computed
    hash against it with `!=`, so MXL's own RESULT append broke `tests/test_mxl_build.py` deterministically).

    The invariant that does NOT move: the registration file's CURRENT sha256 must appear on some line of
    `hash_log` that also names the file, i.e. every past and future append is itself hashed and logged there.
    Mirrors the already-correct `scripts/analysis/mst_gate.py::assert_prereg_registered`
    (tests/test_mst_gate.py::test_default_prereg_path_matches_the_registered_hash); this is the shared,
    stdlib-only home for that idiom so a runner does not need to pin a build-time constant at all.

    Returns the current sha so callers don't hash the file twice."""
    prereg = Path(prereg)
    name = prereg.name
    current = sha256_file(prereg)
    lines = Path(hash_log).read_text().splitlines()
    if not any(line.startswith(current) and name in line for line in lines):
        raise RuntimeError(f"{prereg} sha256 {current} is not in {hash_log} on a line naming {name}; "
                            "refusing to run against an unregistered registration")
    return current
