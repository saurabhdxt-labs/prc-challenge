"""Build what a fresh clone of the repository holds and run the clone tier of the test suite there.

    python tools/clone_check.py [--repo DIR] [--out DIR] [-- <extra pytest arguments>]

The tree is exactly what the next commit would hold — the index (committed files plus staged changes, with their staged
content; unstaged edits are not part of it) — written into a new directory that becomes a git repository with one
commit, so tests that ask git about the checkout see a real one. Nothing untracked or ignored (data/, the development
record, local scratch) reaches it. The clone tier is every test not marked `devtree` (tests/conftest.py), run through
the tree's own `tools/run_suite.py` when it has one (isolated torch / LightGBM passes), else `pytest -m "not devtree"`.

Exit code: the tier's (0 = it passed); 2 for a usage or repository error. The output directory must not exist (it is
never overwritten); by default it is a new directory in the system temp area, outside the repository. Standard library
only.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIT_ID = ["-c", "user.name=clone-check", "-c", "user.email=clone-check@example.invalid", "-c", "commit.gpgsign=false",
          "-c", "core.hooksPath=/dev/null"]                           # no user or global hook runs in the snapshot


class CloneCheckError(Exception):
    """The source is not a usable git repository."""


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *GIT_ID, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise CloneCheckError(f"git {' '.join(args)} in {repo} failed: {r.stderr.strip()}")
    return r.stdout


def tracked_files(repo: Path) -> list:
    """Paths (relative) of every file in the index of `repo`."""
    repo = Path(repo).resolve()
    if subprocess.run(["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"], capture_output=True,
                      text=True).stdout.strip() != "true":
        raise CloneCheckError(f"{repo} is not a git repository")
    return [p for p in _git(repo, "ls-files", "-z").split("\0") if p]


def build_tree(repo, out) -> Path:
    """Write the index of `repo` (the next commit's content) into the new directory `out` and commit it there."""
    repo, out = Path(repo).resolve(), Path(out).resolve()
    tracked_files(repo)                                # refuses a directory that is not a repository
    if out.exists():
        raise FileExistsError(f"refusing to write into the existing {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "checkout-index", "--all", "--force", f"--prefix={out}/")
    if not out.is_dir():                               # an empty index writes nothing
        out.mkdir()
    _git(out, "init", "-q")
    _git(out, "add", "-A", "--force")                  # every written file was in the index, ignored or not
    _git(out, "commit", "-q", "--no-verify", "-m", "clone-check snapshot of the tracked files")
    return out


def run_clone_tier(tree, extra=()) -> tuple:
    """(return code, combined output) of the clone tier run inside the tree: through the tree's tools/run_suite.py when
    it ships one (it forwards `-m "not devtree"` and the extra arguments to both isolated passes), else pytest."""
    tree = Path(tree)
    if (tree / "tools" / "run_suite.py").is_file():
        cmd = [sys.executable, "-B", "tools/run_suite.py", "-m", "not devtree", *extra]
    else:
        cmd = [sys.executable, "-B", "-m", "pytest", "-m", "not devtree", *extra, "tests"]
    r = subprocess.run(cmd, cwd=tree, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def default_out(repo) -> Path:
    """A new directory in the system temp area (outside `repo`, so a bare pytest run never collects the copy)."""
    return Path(tempfile.gettempdir()) / f"prc_clone_check_{time.strftime('%Y%m%dT%H%M%S')}"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(ROOT), help="the repository to check (default: this one)")
    ap.add_argument("--out", default=None, help="new directory for the tree (default: a new temp directory)")
    a = ap.parse_args(own)
    out = Path(a.out) if a.out else default_out(a.repo)
    try:
        tree = build_tree(a.repo, out)
    except (CloneCheckError, FileExistsError) as e:
        print(f"clone_check: {e}", file=sys.stderr)
        return 2
    rc, text = run_clone_tier(tree, extra)
    print(text)
    print(f"clone tier: {'PASSED' if rc == 0 else f'FAILED (pytest exit {rc})'} — tree {tree}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
