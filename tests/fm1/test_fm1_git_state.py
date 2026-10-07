"""prc/pipeline/fm1/artifact.py::git_state — the provenance every artefact manifest records (`git`: sha, dirty_fm1).

End-of-session review 2026-10-03: `dirty_fm1` was `git status --porcelain -- prc configs`, so the development tree's
untracked old-pipeline configs (configs/pipeline_*.yaml) made it True for every artefact whatever FM1's code state (v55
and v56 both read True; v56's prc/ was clean at 55849ac). The config's values are stored whole in config.json, so only
code counts: prc/ (tracked or untracked) and tracked scripts/ (FM1 loads only tracked scripts — BC-38).
Seen red 2026-10-03 08:25-08:42 EDT (the untracked-config case).
"""
from __future__ import annotations

import subprocess

import pytest

from prc.pipeline.fm1 import artifact as A


def _git(root, *a):
    subprocess.run(["git", *a], cwd=root, check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"})


@pytest.fixture
def repo(tmp_path):
    for rel in ("prc/pipeline/fm1/engine.py", "scripts/adsb_nn_worker.py", "configs/fm1.yaml"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "c")
    return tmp_path


def test_a_clean_tree_is_not_dirty_and_records_the_sha(repo):
    st = A.git_state(repo)
    assert st["dirty_fm1"] is False and len(st["sha"]) == 40


@pytest.mark.parametrize("rel", ["configs/pipeline_old.yaml", "scripts/analysis/new_study.py", "plans/PREREG_x.md"])
def test_untracked_files_outside_prc_are_not_fm1_code(repo, rel):
    """Mutation: the old `-- prc configs` porcelain -> the untracked config reads dirty -> red."""
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text("y = 2\n")
    assert A.git_state(repo)["dirty_fm1"] is False


@pytest.mark.parametrize("rel", ["prc/pipeline/fm1/engine.py", "prc/pipeline/fm1/new_module.py",
                                 "scripts/adsb_nn_worker.py"])
def test_fm1_code_that_differs_from_the_commit_is_dirty(repo, rel):
    """A modified or untracked file under prc/, or a modified tracked script (FM1 loads tracked scripts), is dirty.
    Mutation: drop the scripts/ check -> red; untracked files under prc/ ignored -> red."""
    (repo / rel).write_text("x = 2\n")
    assert A.git_state(repo)["dirty_fm1"] is True
