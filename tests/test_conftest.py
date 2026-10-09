"""tests/conftest.py registers the project's pytest markers, refuses unregistered ones and offers an opt-in random order.

A test over the 1 s unit budget moves to an explicit `slow` tier with a pytest marker, so
`-m "not slow"` gives the fast tier. The `devtree` marker names the tests that need files a fresh clone does not carry,
so a clone runs `-m "not devtree"`. A misspelled marker would silently move a test out of its tier, so an unregistered
marker fails the run. Every test must pass in any order: `--random-order=SEED` shuffles the collected tests reproducibly.

Seen RED 2026-09-21 before tests/conftest.py existed: 'slow' was absent from the registered markers. The devtree,
strict-marker and random-order tests were written first and seen red 2026-10-04 (no devtree marker, misspelled marker
collected silently, unknown option --random-order).

The behavioural tests run an inner pytest session in a temporary directory holding a copy of tests/conftest.py, the
way a fresh checkout would load it.

Mutation rehearsal 2026-10-04 (scratch copy, one mutant at a time, source sha-verified unchanged after): 8 / 8 killed.
  CF1 devtree not registered                       -> devtree-registered test, tier test
  CF2 the unknown-marker check removed             -> misspelled-marker test
  CF3 no shuffle                                   -> seed test
  CF4 shuffle ignores the given seed               -> seed test (seed 7 and 8 give one order)
  CF5 shuffle without a seed                       -> file-order test, tier test
  CF6 seed not printed in the header               -> seed test
  CF7 marker names parsed with their signature     -> built-in-markers test (skipif / parametrize refused)
  CF8 a non-standard import added                  -> import test
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from isolation import run_guarded

CONFTEST = Path(__file__).resolve().parent / "conftest.py"


def _registered(pytestconfig):
    return {line.split(":", 1)[0].split("(", 1)[0].strip() for line in pytestconfig.getini("markers")}


def test_slow_marker_is_registered(pytestconfig):
    assert "slow" in _registered(pytestconfig)


def test_devtree_marker_is_registered(pytestconfig):
    assert "devtree" in _registered(pytestconfig)


def _inner(tmp_path: Path, body: str, *args: str) -> subprocess.CompletedProcess:
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_inner.py").write_text(body)
    return run_guarded([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-p", "no:randomly",
                           "-v", *args, str(tmp_path)], cwd=tmp_path, capture_output=True, text=True, timeout=120)


ORDERED = "\n".join(f"def test_{i:02d}():\n    pass\n" for i in range(20))


def _order(r: subprocess.CompletedProcess) -> list:
    return re.findall(r"test_inner\.py::(test_\d\d) PASSED", r.stdout)


@pytest.mark.slow
def test_a_misspelled_marker_fails_the_run_and_is_named(tmp_path):
    r = _inner(tmp_path, "import pytest\n\n@pytest.mark.devtre\ndef test_x():\n    pass\n")
    assert r.returncode != 0
    assert "unregistered marker(s): devtre (test_inner.py::test_x)" in r.stdout + r.stderr


@pytest.mark.slow
def test_builtin_markers_registered_with_a_signature_are_accepted(tmp_path):
    """pytest registers e.g. `skipif(condition, ..., *, reason=...): ...`; the name is the part before the bracket."""
    body = ("import pytest\n\n@pytest.mark.skipif(False, reason='never')\n@pytest.mark.parametrize('x', [1, 2])\n"
            "@pytest.mark.filterwarnings('ignore')\ndef test_x(x):\n    pass\n")
    r = _inner(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 passed" in r.stdout


@pytest.mark.slow
def test_registered_markers_select_tiers(tmp_path):
    body = ("import pytest\n\n@pytest.mark.devtree\ndef test_dev():\n    pass\n\n"
            "@pytest.mark.slow\ndef test_slow():\n    pass\n\ndef test_plain():\n    pass\n")
    r = _inner(tmp_path, body, "-m", "not devtree")
    assert r.returncode == 0, r.stdout + r.stderr
    assert re.findall(r"test_inner\.py::(test_\w+) PASSED", r.stdout) == ["test_slow", "test_plain"]
    assert "1 deselected" in r.stdout


@pytest.mark.slow
def test_without_the_option_tests_run_in_file_order(tmp_path):
    r = _inner(tmp_path, ORDERED)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _order(r) == [f"test_{i:02d}" for i in range(20)]
    assert "random order" not in r.stdout


@pytest.mark.slow
def test_a_seed_gives_one_reproducible_shuffled_order_and_is_printed(tmp_path):
    a = _inner(tmp_path, ORDERED, "--random-order=7")
    b = _inner(tmp_path, ORDERED, "--random-order=7")
    c = _inner(tmp_path, ORDERED, "--random-order=8")
    assert a.returncode == b.returncode == c.returncode == 0, a.stdout + a.stderr
    assert sorted(_order(a)) == [f"test_{i:02d}" for i in range(20)]       # every test still runs, once
    assert _order(a) == _order(b)                                          # same seed, same order
    assert _order(a) != [f"test_{i:02d}" for i in range(20)]               # actually shuffled (P(identity) = 1/20!)
    assert _order(a) != _order(c)                                          # another seed, another order
    assert "random order: seed 7 (replay with --random-order=7)" in a.stdout


def test_the_conftest_imports_only_the_standard_library_and_pytest():
    """BC-5: this file loads in both the torch pass and the tree pass, so it must never pull in either runtime."""
    imported = set(re.findall(r"^\s*(?:import|from)\s+([\w.]+)", CONFTEST.read_text(), flags=re.M))
    assert imported <= {"__future__", "random", "pytest"}


def test_every_test_on_a_real_data_world_is_marked_devtree():
    """tests/fm1/conftest.py marks a test devtree when its fixture closure holds a world (e2e_world, protocol_world): the
    worlds are cut from data a clone does not carry. Without the mark such a test would join the clone tier and fail
    there. Rehearsed RED: the hook returning early; a world fixture left out of WORLD_FIXTURES."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fm1_conftest", CONFTEST.parent / "fm1" / "conftest.py")
    fm1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fm1)

    class Item:
        def __init__(self, *fixtures):
            self.fixturenames, self.marks = list(fixtures), []

        def add_marker(self, mark):
            self.marks.append(mark.name)
    items = [Item("tmp_path"), Item("e2e_world"), Item("art", "cfg", "protocol_world"), Item()]
    fm1.pytest_collection_modifyitems(None, items)
    assert [i.marks for i in items] == [[], ["devtree"], ["devtree"], []]
