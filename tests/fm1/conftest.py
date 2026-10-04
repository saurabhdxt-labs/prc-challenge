"""Session fixtures for tests/fm1: the real-data-shaped worlds (built once per session, in a child process).

The worlds are cut from the project's raw data, weather, ADS-B tables, stack models and route table (fm1_world.py), none
of which a fresh clone carries. Every test that requests a world — directly or through a fixture built on one — is
marked `devtree` here, so `-m "not devtree"` is the clone tier without a hand-kept list (tests/conftest.py).
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location("fm1_world", pathlib.Path(__file__).with_name("fm1_world.py"))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)

#: fixtures that build a world from the project's data (item.fixturenames is the transitive closure)
WORLD_FIXTURES = frozenset({"e2e_world", "protocol_world"})


def pytest_collection_modifyitems(config, items):
    for item in items:
        if WORLD_FIXTURES & set(getattr(item, "fixturenames", ())):
            item.add_marker(pytest.mark.devtree)


@pytest.fixture(scope="session")
def fm1_world_mod():
    return W


@pytest.fixture(scope="session")
def e2e_world(tmp_path_factory):
    d = tmp_path_factory.mktemp("fm1_e2e_world")
    info = W.build_in_child(d, W.E2E)
    return d, info


@pytest.fixture(scope="session")
def protocol_world(tmp_path_factory):
    d = tmp_path_factory.mktemp("fm1_protocol_world")
    info = W.build_in_child(d, W.PROTOCOL)
    return d, info
