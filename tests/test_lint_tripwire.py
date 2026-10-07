"""Lint tripwire: every tracked Python file is clean under pyflakes (undefined names, unused imports and variables,
duplicate imports, f-strings without placeholders).

Closing audit 2026-10-07: pyflakes over the tracked tree reported 82 messages, among them a real crash — `del X, m` in
scripts/stand_ab.py's `fit` command with no `m` in scope (a NameError after the first arm, on a command no test ran) —
and 16 closures reading a design matrix that the enclosing function later `del`eted (the calls happened before the
`del`, so they never failed, but the lint cannot know that and the next edit could move the call). Seen red against
that tree (82 messages); green once the tree was cleaned. Mutation rehearsal 2026-10-07: an unused `import os` added
to prc/months.py -> red (1 message naming the file and line), removed -> green.

Only TRACKED files are scanned (`git ls-files`), so the clone tier scans exactly what a clone holds and a research
scratch file in the working tree cannot fail the suite. pyflakes is a dev dependency (requirements-dev.txt); a missing
pyflakes fails this test rather than skipping it, because a skipped lint is a lint nobody runs."""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def tracked_python_files(root: pathlib.Path) -> list[str]:
    """Paths (relative to `root`) of every tracked *.py file, from git's index: never a directory walk."""
    out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--", "*.py"], capture_output=True, check=True).stdout
    return sorted(p.decode() for p in out.split(b"\0") if p)


def pyflakes_messages(root: pathlib.Path, files: list[str]) -> list[str]:
    """pyflakes' message lines for `files` (run from `root` so the lines carry repository-relative paths)."""
    r = subprocess.run([sys.executable, "-m", "pyflakes", *files], cwd=root, capture_output=True, text=True)
    if r.returncode not in (0, 1) or r.stderr.strip():
        raise RuntimeError(f"pyflakes could not run (rc {r.returncode}): {r.stderr.strip()[:500]}")
    return [line for line in r.stdout.splitlines() if line.strip()]


@pytest.mark.slow
def test_every_tracked_python_file_is_clean_under_pyflakes():
    files = tracked_python_files(ROOT)
    assert len(files) > 100, f"the index lists only {len(files)} Python files; is this the repository?"
    messages = pyflakes_messages(ROOT, files)
    assert messages == [], "\n".join(["pyflakes:"] + messages)
