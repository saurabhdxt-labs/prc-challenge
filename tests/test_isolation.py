"""tests/isolation.py — the shared `isolated` decorator: a real-data test runs in a CHILD pytest process so its memory never
raises the shared pytest process's lifetime peak RSS (BC-34, 2026-09-29: a later unit test read an inherited 3.89 GB
peak and tripped its own 2 GB guard in-suite). No real child is started here: subprocess.run is faked."""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("prc_test_isolation", ROOT / "tests/isolation.py")
I = importlib.util.module_from_spec(_spec)
sys.modules["prc_test_isolation"] = I
_spec.loader.exec_module(I)


def _body_recorder():
    calls = []

    def body(tmp=None):
        calls.append(tmp)
    return body, calls


def test_isolated_runs_the_body_inline_inside_the_child(monkeypatch):
    monkeypatch.setenv(I.CHILD_ENV, "1")
    body, calls = _body_recorder()
    I.isolated(body)("x")
    assert calls == ["x"]


def test_isolated_starts_a_child_pytest_on_the_defining_file_and_test_name(monkeypatch):
    monkeypatch.delenv(I.CHILD_ENV, raising=False)
    seen = {}

    def fake_run(argv, env, capture_output, text, timeout):
        seen.update(argv=argv, env=env, timeout=timeout)
        return subprocess.CompletedProcess(argv, 0, "1 passed", "")
    monkeypatch.setattr(I.subprocess, "run", fake_run)
    body, calls = _body_recorder()
    I.isolated(body)()
    assert calls == []                                              # the parent never runs the body
    assert seen["argv"][:3] == [sys.executable, "-B", "-m"] and seen["argv"][3] == "pytest"
    assert seen["argv"][-1] == f"{pathlib.Path(__file__).resolve()}::body"
    assert seen["env"][I.CHILD_ENV] == "1" and seen["timeout"] == I.TIMEOUT_S


def test_isolated_fails_loudly_with_the_childs_output(monkeypatch):
    monkeypatch.delenv(I.CHILD_ENV, raising=False)
    monkeypatch.setattr(I.subprocess, "run",
                        lambda argv, env, capture_output, text, timeout: subprocess.CompletedProcess(argv, 1, "E boom", "err"))
    body, _ = _body_recorder()
    with pytest.raises(AssertionError, match="child pytest rc 1.*E boom"):
        I.isolated(body)()


def test_isolated_keeps_the_test_name_and_signature_for_pytest():
    """pytest injects fixtures by the wrapped signature (functools.wraps sets __wrapped__)."""
    def test_something(monkeypatch, tmp_path):
        pass
    w = I.isolated(test_something)
    assert w.__name__ == "test_something" and w.__wrapped__ is test_something
