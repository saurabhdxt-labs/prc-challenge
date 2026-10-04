"""Check that both interpreters FM1 uses match the pinned requirements.

FM1 runs in two interpreters: the project interpreter (LightGBM, pandas, torch in a separate worker process) and the
CatBoost worker's virtual environment (`~/.venvs/prc-catboost`; BC-5 keeps CatBoost's OpenMP runtime out of the
LightGBM process). Their pins live in

    requirements.txt           runtime packages of the project interpreter
    requirements-dev.txt       the test tooling (includes requirements.txt)
    requirements-catboost.txt  the CatBoost worker's environment
    .python-version            the Python version both run

Usage: python tools/env_check.py [--root DIR] [--catboost-python PATH]
Exit 0 when everything matches, 1 when anything differs (each difference printed), 2 on a configuration error.
Imports only the standard library, so it still runs when the environment it checks is broken.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: the interpreter prc/pipeline/fm1/learners.py launches the CatBoost worker with (pinned by a test)
DEFAULT_CATBOOST_PYTHON = Path.home() / ".venvs" / "prc-catboost" / "bin" / "python3"

_PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9][A-Za-z0-9.+!_-]*)$")
_INCLUDE = re.compile(r"^-r\s+(\S+)$")

#: run inside the other interpreter with `-I`; prints {"python": "3.11.15", "versions": {name: version or null}}
PROBE = """\
import json
import sys
import importlib.metadata as md
names = json.loads(sys.argv[1])
versions = {}
for n in names:
    try:
        versions[n] = md.version(n)
    except md.PackageNotFoundError:
        versions[n] = None
print(json.dumps({"python": ".".join(str(x) for x in sys.version_info[:3]), "versions": versions}))
"""


class EnvCheckError(Exception):
    """A requirements file or an interpreter could not be read; the check cannot give an answer."""


def normalise(name: str) -> str:
    """PEP 503 name normalisation: case-insensitive, runs of `-`, `_` and `.` are equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path, _seen=None) -> dict:
    """{normalised name: version} from a requirements file of exact pins, `-r` includes and comments only."""
    path = Path(path).resolve()
    seen = (_seen or ()) + (path,)
    if path in (_seen or ()):
        raise EnvCheckError(f"include cycle: {' -> '.join(str(p) for p in seen)}")
    if not path.is_file():
        raise EnvCheckError(f"missing requirements file {path}")
    pins: dict = {}

    def add(name, version):
        if name in pins and pins[name] != version:
            raise EnvCheckError(f"{path.name}: {name} pinned to both {pins[name]} and {version}")
        pins[name] = version

    for n, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        inc = _INCLUDE.match(line)
        if inc:
            for name, version in parse_requirements(path.parent / inc.group(1), seen).items():
                add(name, version)
            continue
        pin = _PIN.match(line)
        if not pin:
            raise EnvCheckError(f"{path.name}:{n}: not an exact pin (name==version): {raw.strip()!r}")
        add(normalise(pin.group(1)), pin.group(2))
    return pins


def merge(a: dict, b: dict) -> dict:
    """The union of two pin sets; a package pinned to different versions in the two is refused."""
    for name in a.keys() & b.keys():
        if a[name] != b[name]:
            raise EnvCheckError(f"{name} pinned to both {a[name]} and {b[name]}")
    return {**a, **b}


def compare(pins: dict, installed: dict) -> list:
    """One message per pinned package whose installed version differs or is absent, in pin order."""
    out = []
    for name, want in pins.items():
        have = installed.get(name)
        if have is None:
            out.append(f"{name}: pinned {want}, not installed")
        elif have != want:
            out.append(f"{name}: pinned {want}, installed {have}")
    return out


def installed_versions(names) -> dict:
    """{name: version or None} in THIS interpreter."""
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def interpreter_versions(python, names) -> tuple:
    """(python version, {name: version or None}) read inside another interpreter."""
    python = Path(python)
    if not python.exists():
        raise EnvCheckError(f"interpreter {python} does not exist")
    try:
        r = subprocess.run([str(python), "-I", "-c", PROBE, json.dumps(list(names))], capture_output=True, text=True,
                           timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise EnvCheckError(f"interpreter {python} could not be run: {e}") from e
    if r.returncode != 0:
        raise EnvCheckError(f"interpreter {python} exited {r.returncode}: {r.stderr.strip()}")
    try:
        data = json.loads(r.stdout)
        return data["python"], {k: data["versions"][k] for k in names}
    except (ValueError, KeyError, TypeError) as e:
        raise EnvCheckError(f"interpreter {python} gave unreadable output: {r.stdout.strip()[:200]!r}") from e


def _python_line(want: str, have: str) -> list:
    return [] if want == have else [f"python: pinned {want}, running {have}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT), help="directory holding the requirement files (default: the repo)")
    ap.add_argument("--catboost-python", default=str(DEFAULT_CATBOOST_PYTHON),
                    help=f"the CatBoost worker's interpreter (default {DEFAULT_CATBOOST_PYTHON})")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    try:
        want_py = (root / ".python-version").read_text().strip() if (root / ".python-version").is_file() else None
        if not want_py:
            raise EnvCheckError(f"missing or empty {root / '.python-version'}")
        project = merge(parse_requirements(root / "requirements.txt"), parse_requirements(root / "requirements-dev.txt"))
        catboost = parse_requirements(root / "requirements-catboost.txt")
        here_py = ".".join(str(x) for x in sys.version_info[:3])
        project_diff = _python_line(want_py, here_py) + compare(project, installed_versions(project))
        cb_py, cb_versions = interpreter_versions(a.catboost_python, list(catboost))
        catboost_diff = _python_line(want_py, cb_py) + compare(catboost, cb_versions)
    except EnvCheckError as e:
        print(f"env_check: {e}", file=sys.stderr)
        return 2
    for label, pins, diff in (("project interpreter", project, project_diff),
                              ("catboost interpreter", catboost, catboost_diff)):
        if diff:
            for d in diff:
                print(f"{label}: {d}")
        else:
            print(f"{label}: ok ({len(pins)} pinned packages)")
    return 1 if project_diff or catboost_diff else 0


if __name__ == "__main__":
    sys.exit(main())
