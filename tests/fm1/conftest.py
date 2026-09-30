"""Session fixtures for tests/fm1: the real-data-shaped worlds (built once per session, in a child process)."""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location("fm1_world", pathlib.Path(__file__).with_name("fm1_world.py"))
W = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(W)


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
