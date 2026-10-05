"""Tripwire for bug class BC-44 (reports/bug_classes.md): a working file in a temp directory that carries its source's
old access time is deleted by a cleaner that works by access time. On 2026-10-05 macOS's daily temp cleaner (03:35)
removed an artefact's stack models from the pytest temp tree while the full suite ran; `shutil.copytree` had copied them
with the timestamps of files written nine days earlier.

The rule this file pins: in prc/, tools/ and tests/ a copy never carries timestamps. `shutil.copy2`, `shutil.copystat`
and a `shutil.copytree` without an explicit `copy_function` are refused; use `shutil.copy` / `shutil.copyfile`, or
`copytree(..., copy_function=shutil.copy)`."""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCANNED = ("prc", "tools", "tests", "scripts")
FILE_COPIES = ("copy" + "2", "copy" + "stat")          # spelled apart so this file does not name what it refuses
TREE_COPY = "copy" + "tree"


def _call_text(source: str, start: int) -> str:
    """The text of the call whose opening parenthesis is at `start`, up to its matching parenthesis."""
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return source[start:]


def stale_copy_sites(source: str) -> list:
    """(line number, what) for every copy in `source` that would carry timestamps."""
    out = []
    for name in FILE_COPIES:
        for m in re.finditer(r"\b" + name + r"\s*\(", source):
            out.append((source.count("\n", 0, m.start()) + 1, name))
    for m in re.finditer(r"\b" + TREE_COPY + r"\s*\(", source):
        if "copy_function=" not in _call_text(source, m.end() - 1):
            out.append((source.count("\n", 0, m.start()) + 1, TREE_COPY + " without copy_function"))
    return sorted(out)


def test_the_scan_finds_each_kind_of_timestamp_carrying_copy():
    """Fails when the scan misses a copy2, a copystat, or a copytree that takes the default copy function, or flags
    a copytree that names one (also across lines) or a plain copy / copyfile."""
    src = "\n".join([
        "shutil." + FILE_COPIES[0] + "(a, b)",
        "shutil." + FILE_COPIES[1] + "(a, b)",
        "shutil." + TREE_COPY + "(a, b)",
        "shutil." + TREE_COPY + "(a, b,",
        "                copy_function=shutil.copy)",
        "shutil." + TREE_COPY + "(a, f(b), dirs_exist_ok=True)",
        "shutil.copy(a, b)",
        "shutil.copyfile(a, b)",
    ])
    assert stale_copy_sites(src) == [(1, FILE_COPIES[0]), (2, FILE_COPIES[1]), (3, TREE_COPY + " without copy_function"),
                                     (6, TREE_COPY + " without copy_function")]


def test_no_copy_in_the_package_the_tools_or_the_tests_carries_timestamps():
    """Fails when a file under prc/, tools/, tests/ or scripts/ copies with copy2 / copystat or with copytree's
    default copy function (BC-44: such a copy made in a temp directory is days old on arrival)."""
    found = []
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*.py")):
            for line, what in stale_copy_sites(path.read_text(encoding="utf-8", errors="replace")):
                found.append(f"{path.relative_to(ROOT)}:{line}: {what}")
    assert found == []
