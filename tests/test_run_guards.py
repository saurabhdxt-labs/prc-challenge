"""Tests for scripts/run_guards.py — the stdlib-only heavy-compute lock and peak-RSS guard (MXL, 2026-09-22).

Written FIRST and seen RED (ImportError: run_guards did not exist). The idiom was copied three times before
(ags_f2026 / ags_build / mst_gate, each frozen by its registration's runner sha); this module is the import seam so the
fourth runner imports it. Mutation rehearsal recorded in the MXL registration RESULT.
"""
from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_guards as G  # noqa: E402

PY = sys.executable


def test_default_lock_path_is_phantoms_and_env_overrides_it(monkeypatch):
    assert G.LOCK == Path.home() / ".phantom" / "phantom_heavy_compute.lock"
    monkeypatch.setenv("PHANTOM_HEAVY_LOCK_PATH", "/nonexistent/x.lock")
    assert G.lock_path() == Path("/nonexistent/x.lock")
    monkeypatch.delenv("PHANTOM_HEAVY_LOCK_PATH")
    assert G.lock_path() == G.LOCK


def test_lock_is_nonblocking_and_a_held_lock_refuses(tmp_path):
    lock = tmp_path / "sub" / "heavy.lock"                          # the parent directory is created
    fh = G.acquire_lock(lock)
    try:
        with pytest.raises(RuntimeError, match="held by another job; not stacking"):
            G.acquire_lock(lock)                                      # a second open file description conflicts
    finally:
        G.release_lock(fh)
    assert fh.closed
    G.release_lock(G.acquire_lock(lock))                              # free again once released


def test_rss_guard_fires_above_the_limit_once_and_not_at_the_limit():
    hits = []
    t, stop = G.start_rss_guard(100, interval=0.005, getrss=lambda: 101, on_breach=hits.append)
    t.join(2)
    assert hits == [101] and not t.is_alive()
    quiet = []
    t, stop = G.start_rss_guard(100, interval=0.005, getrss=lambda: 100, on_breach=quiet.append)
    t.join(0.05)
    stop()
    t.join(2)
    assert quiet == [] and not t.is_alive() and t.daemon


def test_peak_rss_is_in_bytes():
    ru = lambda who: SimpleNamespace(ru_maxrss=1000)
    assert G.peak_rss(getrusage=ru, platform="darwin") == 1000                   # bytes on macOS
    assert G.peak_rss(getrusage=ru, platform="linux") == 1_024_000               # KiB on Linux
    assert 1024 ** 2 < G.peak_rss() < 32 * 1024 ** 3                             # the real call, in bytes, on this Mac


def test_gib_limit_is_six_gib():
    assert G.RSS_LIMIT_6GIB == 6 * 1024 ** 3


@pytest.mark.slow          # a fresh interpreter imports the module
def test_default_breach_exits_the_process_with_70_and_names_the_limit():
    code = ("import sys, time; sys.path.insert(0, %r); import run_guards as G; "
            "G.start_rss_guard(10, interval=0.001, getrss=lambda: 11); time.sleep(5); print('not aborted')"
            % str(ROOT / "scripts"))
    r = subprocess.run([PY, "-B", "-c", code], capture_output=True, text=True, timeout=30)
    assert r.returncode == 70 and "not aborted" not in r.stdout
    assert "ABORT: peak RSS 0.00 GiB is above the 0.00 GiB limit" in r.stderr


# ============================================================================= registration-hash gate
# BC-25: a registration is append-only (a RESULT/ERRATUM legitimately grows the file after it is hashed),
# so its CURRENT sha256 moves over time and can never be a value pinned once, at build time, in an equality
# check (`current != REGISTERED_SHA` -- exactly what mxl_build.py pinned as `REGISTERED_SHA` and mxi_build's
# shared `preflight` compared against; MXL's own RESULT append made the pin stale, tests/test_mxl_build.py
# went RED). The invariant that does NOT move: the CURRENT sha256 must appear on some line of the hash log
# that also names the registration file (mirrors scripts/analysis/mst_gate.py's already-correct
# `assert_prereg_registered`, tests/test_mst_gate.py::test_default_prereg_path_matches_the_registered_hash).

def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("some registration text\n")
    assert G.sha256_file(p) == _sha(p)


def test_assert_prereg_registered_passes_when_current_sha_is_logged(tmp_path):
    prereg = tmp_path / "PREREG_x.md"
    prereg.write_text("original registered text\n")
    log = tmp_path / "x_hashes.log"
    log.write_text(f"{_sha(prereg)}  PREREG_x.md  registered-before-measuring\n")
    assert G.assert_prereg_registered(prereg, log) == _sha(prereg)


def test_assert_prereg_registered_tolerates_a_legitimate_append(tmp_path):
    """The defect this guards against: an append (RESULT/ERRATUM) moves the file's current sha256. As long
    as the NEW current sha gets its own hash-log line naming the file, the check must still pass -- a pinned
    equality check would wrongly refuse here."""
    prereg = tmp_path / "PREREG_x.md"
    prereg.write_text("original registered text\n")
    log = tmp_path / "x_hashes.log"
    original_sha = _sha(prereg)
    log.write_text(f"{original_sha}  PREREG_x.md  registered-before-measuring\n")
    with prereg.open("a") as fh:                                    # simulate a RESULT append
        fh.write("\n## RESULT some outcome\n")
    appended_sha = _sha(prereg)
    assert appended_sha != original_sha                              # the append really did move the sha
    with pytest.raises(RuntimeError):                                # not yet logged -> refuses
        G.assert_prereg_registered(prereg, log)
    with log.open("a") as fh:
        fh.write(f"{appended_sha}  PREREG_x.md  RESULT some outcome\n")
    assert G.assert_prereg_registered(prereg, log) == appended_sha    # now logged -> passes


def test_assert_prereg_registered_refuses_when_current_sha_is_absent_from_the_log(tmp_path):
    prereg = tmp_path / "PREREG_x.md"
    prereg.write_text("original registered text\n")
    log = tmp_path / "x_hashes.log"
    log.write_text("0" * 64 + "  PREREG_x.md  registered-before-measuring\n")   # some OTHER sha logged
    with pytest.raises(RuntimeError, match="not in"):
        G.assert_prereg_registered(prereg, log)


def test_assert_prereg_registered_refuses_a_sha_logged_for_a_different_file(tmp_path):
    """The line must NAME the registration file, not just contain a matching sha256 anywhere in the log
    (a coincidental prefix collision on another file's line must not pass)."""
    prereg = tmp_path / "PREREG_x.md"
    prereg.write_text("original registered text\n")
    log = tmp_path / "x_hashes.log"
    log.write_text(f"{_sha(prereg)}  PREREG_other.md  registered-before-measuring\n")
    with pytest.raises(RuntimeError):
        G.assert_prereg_registered(prereg, log)
