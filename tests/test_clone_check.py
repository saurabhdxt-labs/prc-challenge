"""tools/clone_check.py builds what a fresh clone holds and runs the clone tier there.

The tree is exactly what the next commit would hold — the index (committed files plus staged changes, with their staged
content; unstaged edits are not part of it) — in a new git repository with one commit, so tests that ask git about the checkout see a real one. Nothing
untracked or ignored (data/, the development record, local scratch) reaches it. The clone tier is `-m "not devtree"`
(tests/conftest.py). These tests run the tool on a tiny constructed repository; the run on this repository is a
pre-commit gate step, not a unit test (it takes minutes).

Written first and seen red 2026-10-04 (no module). Mutation rehearsal 2026-10-04 (scratch copy, one at a time, source
sha-verified unchanged after): 6 / 6 killed — CC1 untracked files copied too -> exact-tree test; CC2 no repository made
in the tree -> exact-tree test; CC3 the devtree tests not deselected -> clone-tier, failing-tier and main tests; CC4 an
existing output directory overwritten -> refusal test; CC5 a non-repository accepted -> main exit-2 test (git's own
message also says "not a git repository", so the direct refusal test alone would not catch it); CC6 main always exits 0
-> failing-code test.
Protocol B review 2026-10-04 02:50: CC7 tracked-but-ignored files not force-added -> ignored-path test; CC9 both hook
guards removed (core.hooksPath and --no-verify; either alone is redundant with the other, CC8) -> global-hook test.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import clone_check as C  # noqa: E402

GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false"]


def _repo(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    (src / "tests").mkdir(parents=True)
    (src / "pkg").mkdir()
    (src / "pkg" / "mod.py").write_text("VALUE = 3\n")
    (src / "tests" / "conftest.py").write_text(
        "def pytest_configure(config):\n    config.addinivalue_line('markers', 'devtree: needs the dev tree')\n")
    (src / "tests" / "test_ok.py").write_text(
        "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n"
        "import pytest\nfrom pkg.mod import VALUE\n\n"
        "def test_value():\n    assert VALUE == 3\n\n"
        "@pytest.mark.devtree\ndef test_needs_data():\n    assert pathlib.Path(__file__).resolve().parents[1].joinpath('data', 'x').exists()\n")
    (src / ".gitignore").write_text("data/\n")
    subprocess.run([*GIT, "init", "-q", str(src)], check=True)
    subprocess.run([*GIT, "-C", str(src), "add", "-A"], check=True)
    subprocess.run([*GIT, "-C", str(src), "commit", "-q", "-m", "init"], check=True)
    (src / "data").mkdir()
    (src / "data" / "x").write_text("ignored data\n")
    (src / "untracked.txt").write_text("never reaches the clone\n")
    (src / "pkg" / "staged.py").write_text("STAGED = 1\n")
    subprocess.run([*GIT, "-C", str(src), "add", "pkg/staged.py"], check=True)
    return src


def test_the_tree_holds_exactly_the_tracked_files_in_a_fresh_repository(tmp_path):
    src = _repo(tmp_path)
    out = C.build_tree(src, tmp_path / "clone")
    got = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and ".git" not in p.parts)
    assert got == [".gitignore", "pkg/mod.py", "pkg/staged.py", "tests/conftest.py", "tests/test_ok.py"]
    tracked = subprocess.run(["git", "-C", str(out), "ls-files"], capture_output=True, text=True, check=True).stdout
    assert tracked.split() == got
    assert subprocess.run(["git", "-C", str(out), "status", "--porcelain"], capture_output=True, text=True,
                          check=True).stdout == ""
    assert subprocess.run(["git", "-C", str(out), "rev-list", "--count", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip() == "1"                  # a fresh repository, not a copy of src/.git


def test_the_tree_holds_the_staged_content_not_unstaged_edits(tmp_path):
    """Protocol B (test skeptic M8): copying the working tree certified edits the commit would not contain."""
    src = _repo(tmp_path)
    (src / "pkg" / "mod.py").write_text("VALUE = 4\n")
    subprocess.run([*GIT, "-C", str(src), "add", "pkg/mod.py"], check=True)        # staged: part of the next commit
    (src / "pkg" / "staged.py").write_text("STAGED = 2\n")                        # edited after staging: not part of it
    out = C.build_tree(src, tmp_path / "clone")
    assert (out / "pkg" / "mod.py").read_text() == "VALUE = 4\n"
    assert (out / "pkg" / "staged.py").read_text() == "STAGED = 1\n"


def test_an_existing_output_directory_is_refused(tmp_path):
    src = _repo(tmp_path)
    (tmp_path / "clone").mkdir()
    with pytest.raises(FileExistsError, match="refusing to write into the existing"):
        C.build_tree(src, tmp_path / "clone")


def test_a_directory_that_is_not_a_repository_is_refused(tmp_path):
    with pytest.raises(C.CloneCheckError, match="not a git repository"):
        C.build_tree(tmp_path, tmp_path / "clone")


def test_the_clone_tier_runs_without_the_devtree_tests_and_passes(tmp_path):
    src = _repo(tmp_path)
    rc, out = C.run_clone_tier(C.build_tree(src, tmp_path / "clone"), ["-p", "no:cacheprovider", "-q"])
    assert rc == 0, out
    assert "1 passed, 1 deselected" in out


def test_a_failing_clone_tier_returns_the_pytest_code(tmp_path):
    src = _repo(tmp_path)
    (src / "pkg" / "mod.py").write_text("VALUE = 5\n")
    subprocess.run([*GIT, "-C", str(src), "add", "pkg/mod.py"], check=True)
    rc, out = C.run_clone_tier(C.build_tree(src, tmp_path / "clone"), ["-p", "no:cacheprovider", "-q"])
    assert rc == 1
    assert "1 failed" in out


def test_main_prints_the_verdict_and_returns_the_code(tmp_path, capsys):
    src = _repo(tmp_path)
    rc = C.main(["--repo", str(src), "--out", str(tmp_path / "clone"), "--", "-p", "no:cacheprovider", "-q"])
    assert rc == 0
    assert "clone tier: PASSED" in capsys.readouterr().out


def test_main_returns_the_failing_pytest_code(tmp_path, capsys):
    src = _repo(tmp_path)
    (src / "pkg" / "mod.py").write_text("VALUE = 5\n")
    subprocess.run([*GIT, "-C", str(src), "add", "pkg/mod.py"], check=True)
    assert C.main(["--repo", str(src), "--out", str(tmp_path / "clone"), "--", "-p", "no:cacheprovider", "-q"]) == 1
    assert "clone tier: FAILED (pytest exit 1)" in capsys.readouterr().out


def test_main_returns_2_for_a_directory_that_is_not_a_repository(tmp_path, capsys):
    assert C.main(["--repo", str(tmp_path), "--out", str(tmp_path / "clone")]) == 2
    assert "is not a git repository" in capsys.readouterr().err


def test_a_tree_with_the_suite_runner_runs_the_tier_through_it(tmp_path):
    """D10: the clone tier keeps tools/run_suite.py's isolated torch / LightGBM passes when the tree ships the runner."""
    src = _repo(tmp_path)
    (src / "tools").mkdir()
    (src / "tools" / "run_suite.py").write_text(
        "import sys, pathlib\npathlib.Path('argv.txt').write_text(' '.join(sys.argv[1:]))\n")
    subprocess.run([*GIT, "-C", str(src), "add", "tools/run_suite.py"], check=True)
    tree = C.build_tree(src, tmp_path / "clone")
    rc, _ = C.run_clone_tier(tree, ["-q"])
    assert rc == 0
    assert (tree / "argv.txt").read_text() == "-m not devtree -q"


def test_the_default_output_directory_is_outside_the_repository(tmp_path):
    """A tree inside the repository would be collected by a bare `pytest` run from the repository root."""
    out = C.default_out(tmp_path / "repo")
    assert not out.is_relative_to(tmp_path / "repo") and not out.exists()


def test_a_tracked_file_under_an_ignored_path_stays_tracked_in_the_snapshot(tmp_path):
    """Protocol B review 2026-10-04 (MEDIUM-3): `git add -A` skips files that match .gitignore even when the source
    index tracks them (here data/rules.py under the ignored data/, like the shipped E2 rules file), so the snapshot did
    not match a clone. The snapshot force-adds what it wrote."""
    src = _repo(tmp_path)
    (src / "data" / "rules.py").write_text("RULE = 1\n")
    subprocess.run([*GIT, "-C", str(src), "add", "--force", "data/rules.py"], check=True)
    out = C.build_tree(src, tmp_path / "clone")
    tracked = subprocess.run(["git", "-C", str(out), "ls-files"], capture_output=True, text=True, check=True).stdout
    assert "data/rules.py" in tracked.split() and "data/x" not in tracked.split()


def test_a_global_commit_hook_never_runs_in_the_snapshot(tmp_path, monkeypatch):
    """A user's global pre-commit hook (core.hooksPath) must neither block nor change the snapshot commit."""
    hooks = tmp_path / "hooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text("#!/bin/sh\necho refused by a global hook >&2\nexit 1\n")
    (hooks / "pre-commit").chmod(0o755)
    cfg = tmp_path / "gitconfig"
    cfg.write_text(f"[core]\n\thooksPath = {hooks}\n")
    src = _repo(tmp_path)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(cfg))
    out = C.build_tree(src, tmp_path / "clone")
    assert subprocess.run(["git", "-C", str(out), "rev-list", "--count", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip() == "1"
