"""Run a command while HOLDING PHANTOM's machine-wide heavy-compute lock.

Lesson of 2026-08-14 (the 01:00Z forecast hole): any session-launched compute over ~1 GB
takes this lock, research-tier or not. "It's just a screen" is the same anti-pattern as
"it's just a script". The lock is a non-blocking fcntl flock, so it dies with its holder
and can never wedge the queue (phantom-forecaster/scripts/batch_runner.py:67 lock_is_held).

Refuses to start if the lock is already held — skip, never stack onto a production fire.
"""
import fcntl, os, pathlib, subprocess, sys

LOCK = pathlib.Path(os.environ.get(
    "PHANTOM_HEAVY_LOCK_PATH", pathlib.Path.home() / ".phantom" / "phantom_heavy_compute.lock"))

def main(argv):
    if not argv:
        print("usage: run_with_heavy_lock.py <cmd> [args...]", file=sys.stderr); return 2
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOCK, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"HEAVY LOCK HELD by another job — skipping, not stacking. ({LOCK})", flush=True)
        return 75                                    # EX_TEMPFAIL
    print(f"heavy lock ACQUIRED pid {os.getpid()} -> running: {' '.join(argv)}", flush=True)
    try:
        return subprocess.call(argv)
    finally:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN); fh.close()
        print("heavy lock RELEASED", flush=True)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
