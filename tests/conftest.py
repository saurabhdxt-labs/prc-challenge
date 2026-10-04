"""Project-wide pytest configuration: markers, a strict marker check and an opt-in random test order.

Imports only the standard library and pytest: tools/run_suite.py runs the tree and torch passes in separate
interpreters (BC-5) and this file loads in both. Pinned by tests/test_conftest.py.

Markers
  slow     over the 1 s unit-test budget; the fast tier is `-m "not slow"`.
  devtree  needs files a fresh clone may not carry (data/, logs/, artifacts/, submissions/, the development record,
           the CatBoost environment, or code that is not tracked in git); a clone runs `-m "not devtree"`.
An unregistered marker fails the run: a misspelling would otherwise move a test out of its tier without a sound.

`--random-order=SEED` shuffles the collected tests with that seed (every test must pass in any order) and prints the seed in
the header so a failing order can be replayed. Off by default.
"""
from __future__ import annotations

import random

import pytest

MARKERS = {
    "slow": "over the 1 s unit-test budget; deselect with -m 'not slow'",
    "devtree": "needs files a fresh clone does not carry (data, development record, untracked code); "
               "a clone runs -m 'not devtree'",
}


def pytest_addoption(parser):
    parser.addoption("--random-order", type=int, default=None, metavar="SEED",
                     help="shuffle the collected tests with this seed (every test must pass in any order)")


def pytest_configure(config):
    for name, text in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {text}")


def pytest_report_header(config):
    seed = config.getoption("random_order", default=None)
    return None if seed is None else f"random order: seed {seed} (replay with --random-order={seed})"


def pytest_collection_modifyitems(session, config, items):
    registered = {line.split(":", 1)[0].split("(", 1)[0].strip() for line in config.getini("markers")}
    unknown = sorted({f"{mark.name} ({item.nodeid})" for item in items for mark in item.iter_markers()
                      if mark.name not in registered})
    if unknown:
        raise pytest.UsageError(f"unregistered marker(s): {', '.join(unknown)} — register them in tests/conftest.py")
    seed = config.getoption("random_order", default=None)
    if seed is not None:
        random.Random(seed).shuffle(items)
