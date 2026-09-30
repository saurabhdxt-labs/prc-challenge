"""Shared test helper (not a test module): `isolated` runs a real-data test in a CHILD pytest process.

Why: pytest runs every test in one interpreter, and in-process RSS guards (e.g. run_guards.start_rss_guard) read the
process's LIFETIME peak (ru_maxrss). A real-data test that loads a big frame in the shared process raises that peak for
every later test: on 2026-09-23 the suite died with exit 70; on 2026-09-29 tests/test_dfc2_build.py raised it
1.77 -> 4.12 GB and a later guard-calling unit test failed in-suite while passing alone. The guards' contract is right
for CLI jobs; the fix is that heavy tests do not share the process (BC-34). Tested in tests/test_isolation.py.

    ISO = _load("prc_test_isolation", "tests/isolation.py")
    @ISO.isolated
    def test_on_real_data(...): ...
"""
from __future__ import annotations

import functools
import inspect
import os
import pathlib
import subprocess
import sys

CHILD_ENV = "PRC_ISOLATED_CHILD"
TIMEOUT_S = 1_800


def isolated(fn):
    """In the parent: run `<defining file>::<test name>` in a child pytest and assert it passed (child output in the
    failure message). In the child (CHILD_ENV=1): run the body. Parametrised tests are not supported (node ids)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get(CHILD_ENV) == "1":
            return fn(*args, **kwargs)
        node = f"{pathlib.Path(inspect.getsourcefile(fn)).resolve()}::{fn.__name__}"
        r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
                           env={**os.environ, CHILD_ENV: "1"}, capture_output=True, text=True, timeout=TIMEOUT_S)
        assert r.returncode == 0, f"child pytest rc {r.returncode}: {r.stdout[-4000:]}{r.stderr[-2000:]}"
    return wrapper
