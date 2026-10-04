"""The environment check must report every way an interpreter differs from the pinned requirements, and nothing else.

`tools/env_check.py` compares the project interpreter with `requirements.txt` + `requirements-dev.txt` and the CatBoost
worker's interpreter (`~/.venvs/prc-catboost`, BC-5) with `requirements-catboost.txt`, and both Python versions with
`.python-version` (the tests and the runs must use the same environment). It imports only the standard library, so it still runs
when the environment it checks is broken. These tests are the spec.

Written first and seen red 2026-10-04 00:11 EDT (no module). Mutation rehearsal 2026-10-04 (scratch copy, one mutant at
a time, source sha-verified unchanged after): 15 / 15 killed.
  EC1  names not lower-cased                         -> normalised-names test, two-versions test
  EC2  `>=` accepted as a pin                        -> exact-pin refusal [numpy>=2], configuration-error exit
  EC3  include-cycle check removed                   -> include-cycle test (RecursionError instead of the message)
  EC4  conflicting pins inside one file set accepted -> two-versions test
  EC5  a missing package not reported                -> missing-version test, every-mismatch test
  EC6  equality test inverted                        -> matching / mismatch / unpinned tests, CLI pass test
  EC7  a failing interpreter's exit code ignored     -> stderr test
  EC8  `main` always exits 0                         -> every-mismatch test, dev-requirements test
  EC9  Python version never compared                 -> every-mismatch test
  EC10 runtime-vs-dev conflicting pins merged        -> runtime-and-dev conflict test
  EC11 default CatBoost interpreter path drifts      -> default-interpreter test
  EC12 CatBoost interpreter's Python never compared  -> every-mismatch test
  EC13 inline comments not stripped                  -> normalised-names test
  EC14 missing interpreter not refused by name       -> missing-interpreter tests
  EC15 includes resolved from the repo root          -> include-relative, include-cycle, two-versions tests
Not separately tested: the probe's `-I` flag (isolates the probe from PYTHONPATH / user site); recorded here.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import env_check as E  # noqa: E402

PYTEST_VERSION = pytest.__version__
PY_VERSION = ".".join(str(x) for x in sys.version_info[:3])


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def _repo(tmp_path: Path, runtime: str, dev: str = "", catboost: str | None = None, python: str = PY_VERSION) -> Path:
    _write(tmp_path, "requirements.txt", runtime)
    _write(tmp_path, "requirements-dev.txt", dev)
    _write(tmp_path, "requirements-catboost.txt", catboost if catboost is not None else f"pytest=={PYTEST_VERSION}\n")
    _write(tmp_path, ".python-version", python + "\n")
    return tmp_path


# --------------------------------------------------------------------------- parsing

def test_pins_comments_and_blank_lines_parse_to_normalised_names(tmp_path):
    p = _write(tmp_path, "r.txt", "# header\n\nNumPy==2.4.3  # inline comment\nscikit_learn==1.8.0\nPyYAML==6.0.3\n")
    assert E.parse_requirements(p) == {"numpy": "2.4.3", "scikit-learn": "1.8.0", "pyyaml": "6.0.3"}


def test_an_include_is_read_relative_to_the_including_file(tmp_path):
    (tmp_path / "sub").mkdir()
    _write(tmp_path, "base.txt", "numpy==2.4.3\n")
    p = _write(tmp_path / "sub", "dev.txt", "-r ../base.txt\npytest==9.0.2\n")
    assert E.parse_requirements(p) == {"numpy": "2.4.3", "pytest": "9.0.2"}


def test_an_include_cycle_is_refused(tmp_path):
    _write(tmp_path, "a.txt", "-r b.txt\n")
    _write(tmp_path, "b.txt", "-r a.txt\n")
    with pytest.raises(E.EnvCheckError, match=r"include cycle: .*a\.txt"):
        E.parse_requirements(tmp_path / "a.txt")


@pytest.mark.parametrize("line", ["numpy>=2", "numpy", "numpy==", "numpy==2.4.3; python_version<'3.12'", "-e ."])
def test_anything_but_an_exact_pin_is_refused_with_its_location(tmp_path, line):
    p = _write(tmp_path, "r.txt", f"pandas==3.0.1\n{line}\n")
    with pytest.raises(E.EnvCheckError, match=r"r\.txt:2: not an exact pin"):
        E.parse_requirements(p)


def test_one_package_pinned_to_two_versions_is_refused(tmp_path):
    _write(tmp_path, "base.txt", "numpy==2.4.3\n")
    p = _write(tmp_path, "dev.txt", "-r base.txt\nNumPy==2.4.2\n")
    with pytest.raises(E.EnvCheckError, match=r"numpy pinned to both 2\.4\.3 and 2\.4\.2"):
        E.parse_requirements(p)


def test_a_missing_requirements_file_is_refused_by_name(tmp_path):
    with pytest.raises(E.EnvCheckError, match=r"missing requirements file .*nope\.txt"):
        E.parse_requirements(tmp_path / "nope.txt")


# --------------------------------------------------------------------------- comparison

def test_matching_versions_report_nothing():
    assert E.compare({"numpy": "2.4.3"}, {"numpy": "2.4.3"}) == []


def test_a_different_and_a_missing_version_are_each_reported():
    got = E.compare({"numpy": "2.4.3", "torch": "2.11.0"}, {"numpy": "2.4.2", "torch": None})
    assert got == ["numpy: pinned 2.4.3, installed 2.4.2", "torch: pinned 2.11.0, not installed"]


def test_packages_installed_but_not_pinned_are_not_reported():
    assert E.compare({"numpy": "2.4.3"}, {"numpy": "2.4.3", "rich": "13.0"}) == []


# --------------------------------------------------------------------------- reading interpreters

def test_installed_versions_reads_this_interpreter():
    assert E.installed_versions(["pytest", "no-such-package-xyz"]) == {"pytest": PYTEST_VERSION,
                                                                       "no-such-package-xyz": None}


def test_interpreter_versions_reads_another_interpreter_through_a_subprocess():
    py, versions = E.interpreter_versions(sys.executable, ["pytest", "no-such-package-xyz"])
    assert py == PY_VERSION
    assert versions == {"pytest": PYTEST_VERSION, "no-such-package-xyz": None}


def test_a_missing_interpreter_is_named(tmp_path):
    with pytest.raises(E.EnvCheckError, match=r"interpreter .*no-python does not exist"):
        E.interpreter_versions(tmp_path / "no-python", ["pytest"])


def test_an_interpreter_that_fails_is_reported_with_its_stderr(tmp_path):
    bad = _write(tmp_path, "bad-python", "#!/bin/sh\necho 'boom on stderr' >&2\nexit 3\n")
    bad.chmod(0o755)
    with pytest.raises(E.EnvCheckError, match=r"exited 3: boom on stderr"):
        E.interpreter_versions(bad, ["pytest"])


def test_an_interpreter_that_prints_garbage_is_refused(tmp_path):
    bad = _write(tmp_path, "odd-python", "#!/bin/sh\necho 'not json'\n")
    bad.chmod(0o755)
    with pytest.raises(E.EnvCheckError, match=r"unreadable output"):
        E.interpreter_versions(bad, ["pytest"])


# --------------------------------------------------------------------------- the CLI

def test_main_passes_when_both_interpreters_match(tmp_path, capsys):
    root = _repo(tmp_path, runtime=f"pytest=={PYTEST_VERSION}\n")
    assert E.main(["--root", str(root), "--catboost-python", sys.executable]) == 0
    out = capsys.readouterr().out
    assert "project interpreter: ok (1 pinned packages)" in out
    assert "catboost interpreter: ok (1 pinned packages)" in out


def test_main_fails_and_names_every_mismatch(tmp_path, capsys):
    root = _repo(tmp_path, runtime="pytest==0.0.1\n", catboost="no-such-package-xyz==1.0\n", python="3.0.0")
    assert E.main(["--root", str(root), "--catboost-python", sys.executable]) == 1
    out = capsys.readouterr().out
    assert f"project interpreter: pytest: pinned 0.0.1, installed {PYTEST_VERSION}" in out
    assert f"project interpreter: python: pinned 3.0.0, running {PY_VERSION}" in out
    assert "catboost interpreter: no-such-package-xyz: pinned 1.0, not installed" in out
    assert f"catboost interpreter: python: pinned 3.0.0, running {PY_VERSION}" in out


def test_main_reads_the_dev_requirements_too(tmp_path, capsys):
    root = _repo(tmp_path, runtime="", dev="pytest==0.0.1\n")
    assert E.main(["--root", str(root), "--catboost-python", sys.executable]) == 1
    assert "pytest: pinned 0.0.1" in capsys.readouterr().out


def test_main_refuses_runtime_and_dev_files_that_pin_one_package_differently(tmp_path, capsys):
    root = _repo(tmp_path, runtime=f"pytest=={PYTEST_VERSION}\n", dev="pytest==0.0.1\n")
    assert E.main(["--root", str(root), "--catboost-python", sys.executable]) == 2
    assert f"pytest pinned to both {PYTEST_VERSION} and 0.0.1" in capsys.readouterr().err


def test_main_exits_2_on_a_configuration_error(tmp_path, capsys):
    root = _repo(tmp_path, runtime="numpy>=2\n")
    assert E.main(["--root", str(root), "--catboost-python", sys.executable]) == 2
    assert "requirements.txt:1: not an exact pin" in capsys.readouterr().err


def test_main_exits_2_when_the_catboost_interpreter_is_missing(tmp_path, capsys):
    root = _repo(tmp_path, runtime="")
    assert E.main(["--root", str(root), "--catboost-python", str(tmp_path / "gone")]) == 2
    assert "does not exist" in capsys.readouterr().err


def test_the_cli_runs_as_a_script(tmp_path):
    root = _repo(tmp_path, runtime=f"pytest=={PYTEST_VERSION}\n")
    r = subprocess.run([sys.executable, "-B", str(ROOT / "tools" / "env_check.py"), "--root", str(root),
                        "--catboost-python", sys.executable], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "catboost interpreter: ok" in r.stdout


# --------------------------------------------------------------------------- the repository's own files

def test_the_repository_requirement_files_parse_and_pin_what_the_code_imports():
    runtime = E.parse_requirements(ROOT / "requirements.txt")
    dev = E.parse_requirements(ROOT / "requirements-dev.txt")
    catboost = E.parse_requirements(ROOT / "requirements-catboost.txt")
    assert set(runtime) == {"lightgbm", "numpy", "pandas", "pyarrow", "pyyaml", "scikit-learn", "scipy", "torch"}
    assert set(dev) == set(runtime) | {"pytest", "coverage", "hypothesis"}
    assert {"catboost", "numpy"} <= set(catboost)
    assert (ROOT / ".python-version").read_text().strip().count(".") == 2


def test_the_default_catboost_interpreter_is_the_one_fm1_launches():
    from prc.pipeline.fm1 import learners
    assert Path(E.DEFAULT_CATBOOST_PYTHON) == Path(learners.CATBOOST_PYTHON)


def test_the_probe_script_imports_only_the_standard_library():
    """The probe runs inside the CatBoost venv, which has no project packages; json + importlib.metadata only."""
    imported = set(__import__("re").findall(r"^\s*(?:import|from)\s+([\w.]+)", E.PROBE, flags=__import__("re").M))
    assert imported <= {"json", "sys", "importlib.metadata"}
    out = subprocess.run([sys.executable, "-I", "-c", E.PROBE, json.dumps(["pytest"])], capture_output=True,
                         text=True, timeout=60)
    assert json.loads(out.stdout)["versions"] == {"pytest": PYTEST_VERSION}
