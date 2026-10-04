"""The suite runner must keep torch and the tree libraries in separate processes (bug class BC-5).

`tools/run_suite.py` is the project's "full suite" entry point: pytest loads every collected test module into ONE
interpreter, so a test file that imports torch and a test file that imports LightGBM (directly or through
`lgbm_fold` / `lgbm_submit`) put two incompatible OpenMP runtimes in the same process. On 2026-09-17 that deadlocked
the suite at 80 % for 43 minutes (main thread inside `torch … mul_kernel → __kmpc_fork_call → __kmp_join_barrier →
__kmp_suspend_64`) after the same clash had already segfaulted the TNS smokes. The runner therefore splits collection
into a torch pass and a tree pass and runs them as separate processes.

Discovery is by SOURCE SCAN, never a hand-maintained list: a new torch-importing test file is isolated automatically,
which is the property that makes this fix survive the next module. These tests are the spec.

Mutation rehearsal (2026-09-17, each break applied to `tools/run_suite.py`, suite file run, module restored):
  R1  `TORCH_MARKERS` loses "torch"                                            -> RED (discovery test)
  R2  `TORCH_MARKERS` loses "tabm_screen"                                      -> RED (real-repo test)
  R3  `TORCH_MARKERS` loses "tree_neural_stack"                                -> RED (real-repo test)
  R4  import regex anchored to line start without `^\\s*` (indented import)    -> RED (indented-import test)
  R5  regex matches bare substring (a comment mentioning torch counts)         -> RED (comment/string test)
  R6  `split_files` returns (all, []) - everything in one pass                 -> RED (partition test)
  R7  `split_files` returns (torch, all) - overlap                             -> RED (partition test)
  R8  `main` runs only the first pass                                          -> RED (two-invocation test)
  R9  `main` returns 0 when a pass fails                                       -> RED (exit-code tests)
  R10 `main` returns 0 when the SECOND pass fails                              -> RED (second-pass exit test)
  R11 both passes launched in one subprocess call (` && `)                     -> RED (two-invocation test)
  R12 `--ignore` dropped from the tree pass command                            -> RED (command-shape test)
  R13 tree pass passes the torch files explicitly                              -> RED (command-shape test)
  R14 `-p no:randomly` dropped                                                 -> RED (command-shape test)
  R15 empty torch set still spawns an empty second pass                        -> RED (empty-set test)
  R16 extra pytest args forwarded to only one pass                             -> RED (extra-args test)
  R17 `run_pass` ignores the returncode it is given                            -> RED (exit-code tests)
  EQ1 dict/set iteration order of the discovered files                         -> equivalent (sorted before use)

2026-10-04 (clean-clone audit): five tests read the real tests directory, whose torch files are development-tree files
(their modules are not in git), so they failed on a fresh clone. The pass tests now run on a constructed directory
(`tests_dir`) and assert the exact commands; the real-repo cross-check no longer requires a torch file to exist.
Re-rehearsed on a scratch copy (one mutant at a time, source sha-verified unchanged after): 8 / 8 killed —
  RS1 the torch pass never runs              -> two-subprocess, command, random-order, extra-args, both-run tests
  RS2 the tree pass loses its --ignore list  -> two-subprocess, tree-ignores, extra-args tests
  RS3 the tree pass names the torch files    -> same
  RS4 `-p no:randomly` dropped               -> two-subprocess, random-order, extra-args tests
  RS5 extra arguments reach the tree pass only -> extra-args test
  RS6 a failing torch pass passes the run    -> failing-torch-pass test
  RS7 run_pass ignores the returncode        -> failing-tree / failing-torch / both-run tests
  RS8 the torch pass collects the whole tests directory too -> two-subprocess, extra-args tests
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_suite as R  # noqa: E402


# --------------------------------------------------------------------------- discovery

def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def test_a_file_importing_torch_is_discovered(tmp_path):
    _write(tmp_path, "test_a.py", "import torch\n")
    assert R.torch_test_files(tmp_path) == [tmp_path / "test_a.py"]


def test_a_file_importing_a_torch_carrying_module_is_discovered(tmp_path):
    _write(tmp_path, "test_b.py", "import tabm_screen\n")
    _write(tmp_path, "test_c.py", "from tree_neural_stack import prep\n")
    assert R.torch_test_files(tmp_path) == [tmp_path / "test_b.py", tmp_path / "test_c.py"]


def test_an_indented_import_is_discovered(tmp_path):
    """R4: torch imported inside a function or a fixture still loads it into the process."""
    _write(tmp_path, "test_d.py", "def f():\n    import torch\n    return torch\n")
    assert R.torch_test_files(tmp_path) == [tmp_path / "test_d.py"]


def test_a_mention_in_a_comment_or_string_is_not_an_import(tmp_path):
    """R5: the scan must not isolate a file that only talks about torch."""
    _write(tmp_path, "test_e.py", "# we do not import torch here\nS = 'import torch'\n")
    assert R.torch_test_files(tmp_path) == []


def test_files_without_torch_are_not_discovered(tmp_path):
    _write(tmp_path, "test_f.py", "import numpy as np\nimport lgbm_fold\n")
    assert R.torch_test_files(tmp_path) == []


def test_discovery_is_sorted_and_deterministic(tmp_path):
    for n in ("test_z.py", "test_a.py", "test_m.py"):
        _write(tmp_path, n, "import torch\n")
    expected = [tmp_path / "test_a.py", tmp_path / "test_m.py", tmp_path / "test_z.py"]
    assert R.torch_test_files(tmp_path) == expected
    assert R.torch_test_files(tmp_path) == expected             # deterministic across calls


def test_the_real_repo_partition_is_exactly_the_torch_files():
    """The live contract, checked against Python's own parser rather than a hand-listed set.

    A hardcoded list of torch files is wrong twice over: it goes red the moment a legitimate new torch test
    file appears (it did, on 2026-09-18, for `test_tree_neural_matched.py`), and it never checks the property
    that matters. So: cross-validate the runner's regex scan against `ast`, which is an independent
    implementation. Any file the scanner isolates must really import a torch-carrying module, and any file it
    leaves in the tree pass must really not. The torch test files are development-tree files (their modules are not
    in git), so a fresh clone has none; the cross-check holds either way, and discovery itself is pinned on a
    constructed directory (test_a_file_importing_torch_is_discovered).
    """
    import ast

    # The semantic contract, written out here and NOT taken from the runner: these imports pull
    # `torch/lib/libomp.dylib` into the process. Deliberately a list of MODULES (stable, and a human should
    # think when it changes), never a list of FILES (grows with every new test file). If the runner's marker
    # set drifts from this one, that is the bug this test exists to catch.
    TORCH_IMPORTS = {"torch", "tabm_screen", "tree_neural_matched", "tree_neural_stack"}
    assert set(R.TORCH_MARKERS) == TORCH_IMPORTS, (
        f"runner markers {sorted(R.TORCH_MARKERS)} drifted from the contract {sorted(TORCH_IMPORTS)}; "
        "a module that imports torch was added or removed - update both, deliberately")

    def imports_via_ast(path: Path) -> bool:
        roots = set()
        for node in ast.walk(ast.parse(path.read_text(errors="replace"))):
            if isinstance(node, ast.Import):
                roots |= {a.name.split(".", 1)[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                roots.add(node.module.split(".", 1)[0])
        return bool(roots & TORCH_IMPORTS)

    all_tests = sorted((ROOT / "tests").rglob("test_*.py"))
    torch_files = R.torch_test_files(ROOT / "tests")
    for p in all_tests:
        assert imports_via_ast(p) == (p in torch_files), p.name
    tree, torch = R.split_files(all_tests, torch_files)
    assert set(tree) | set(torch) == set(all_tests)          # R6/R7: a partition
    assert not (set(tree) & set(torch))
    assert set(torch) == set(torch_files)


def test_split_files_partitions_without_overlap(tmp_path):
    files = [tmp_path / f"test_{i}.py" for i in range(5)]
    torch = [files[1], files[3]]
    tree, tf = R.split_files(files, torch)
    assert set(tree) == {files[0], files[2], files[4]}
    assert set(tf) == set(torch)
    assert not (set(tree) & set(tf))


# --------------------------------------------------------------------------- the two passes

@pytest.fixture
def tests_dir(tmp_path, monkeypatch):
    """A constructed tests directory: one tree file and two torch files (a direct and a `from` import of torch-carrying
    modules), so the pass commands are checked independently of which test files the checkout happens to hold."""
    plain = _write(tmp_path, "test_plain.py", "import numpy\n")
    tabm = _write(tmp_path, "test_tabm_screen.py", "import tabm_screen\n")
    tns = _write(tmp_path, "test_tree_neural_stack.py", "from tree_neural_stack import prep\n")
    monkeypatch.setattr(R, "TESTS_DIR", tmp_path)
    return tmp_path, plain, tabm, tns


def _expected(tests_dir, extra=()):
    d, _, tabm, tns = tests_dir
    tree = [sys.executable, "-m", "pytest", str(d), f"--ignore={tabm}", f"--ignore={tns}", "-p", "no:randomly", *extra]
    torch = [sys.executable, "-m", "pytest", str(tabm), str(tns), "-p", "no:randomly", *extra]
    return [tree, torch]


class _Recorder:
    def __init__(self, codes):
        self.codes, self.calls = list(codes), []

    def __call__(self, cmd, **kw):
        self.calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, self.codes[len(self.calls) - 1] if self.codes else 0)


def test_main_runs_exactly_two_separate_subprocesses(monkeypatch, tests_dir):
    """R8/R11: two passes, two processes - never one command, never one pass."""
    rec = _Recorder([0, 0])
    monkeypatch.setattr(R.subprocess, "run", rec)
    assert R.main([]) == 0
    assert rec.calls == _expected(tests_dir)
    for cmd in rec.calls:
        assert " && " not in " ".join(cmd)


def test_the_tree_pass_ignores_every_torch_file_and_the_torch_pass_names_them(monkeypatch, tests_dir):
    """R12/R13: the isolation lives in the commands themselves — the tree pass ignores exactly the torch files and the
    torch pass names exactly them, never the tree file."""
    rec = _Recorder([0, 0])
    monkeypatch.setattr(R.subprocess, "run", rec)
    R.main([])
    tree_cmd, torch_cmd = rec.calls
    _, plain, tabm, tns = tests_dir
    assert [c for c in tree_cmd if c.startswith("--ignore=")] == [f"--ignore={tabm}", f"--ignore={tns}"]
    assert str(plain) not in torch_cmd and str(tabm) in torch_cmd and str(tns) in torch_cmd
    assert not any(c.startswith("--ignore") for c in torch_cmd)


def test_both_passes_disable_random_order(monkeypatch, tests_dir):
    """R14: order plugins reshuffle collection across the passes and break the isolation contract."""
    rec = _Recorder([0, 0])
    monkeypatch.setattr(R.subprocess, "run", rec)
    R.main([])
    assert len(rec.calls) == 2
    for cmd in rec.calls:
        i = cmd.index("-p")
        assert cmd[i + 1] == "no:randomly"


def test_extra_arguments_reach_both_passes(monkeypatch, tests_dir):
    """R16: `-x`, `-k`, markers must not silently apply to half the suite."""
    rec = _Recorder([0, 0])
    monkeypatch.setattr(R.subprocess, "run", rec)
    R.main(["-x", "-k", "smoke"])
    assert rec.calls == _expected(tests_dir, ["-x", "-k", "smoke"])


def test_a_failing_tree_pass_fails_the_run(monkeypatch, tests_dir):
    rec = _Recorder([1, 0])
    monkeypatch.setattr(R.subprocess, "run", rec)
    assert R.main([]) == 1
    assert len(rec.calls) == 2


def test_a_failing_torch_pass_fails_the_run(monkeypatch, tests_dir):
    """R10: the second pass is not a courtesy run - its failure is the suite's failure."""
    rec = _Recorder([0, 1])
    monkeypatch.setattr(R.subprocess, "run", rec)
    assert R.main([]) == 1
    assert len(rec.calls) == 2


def test_both_passes_run_even_when_the_first_fails(monkeypatch, tests_dir):
    """A red tree pass must not hide the state of the torch tests."""
    rec = _Recorder([1, 1])
    monkeypatch.setattr(R.subprocess, "run", rec)
    assert R.main([]) != 0
    assert len(rec.calls) == 2


def test_no_second_pass_when_no_torch_file_exists(monkeypatch, tmp_path):
    """R15: an empty pytest invocation would collect the whole suite again, re-creating the clash."""
    rec = _Recorder([0, 0])
    monkeypatch.setattr(R.subprocess, "run", rec)
    monkeypatch.setattr(R, "TESTS_DIR", tmp_path)
    _write(tmp_path, "test_plain.py", "import numpy\n")
    assert R.main([]) == 0
    assert len(rec.calls) == 1


# --------------------------------------------------------------------------- the runner itself is torch-free

def test_the_runner_module_imports_neither_torch_nor_a_tree_library():
    """The runner must be safe to import anywhere, including inside either pass."""
    src = (ROOT / "tools" / "run_suite.py").read_text()
    # dogfood the scanner: what the runner ACTUALLY imports, not what its docstring mentions
    assert not (R.imported_modules(src) & (R.TORCH_MARKERS | {"lightgbm", "lgbm_fold", "lgbm_submit", "catboost"}))


def test_runner_help_runs_in_a_clean_process():
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "run_suite.py"), "--help"],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "two isolated passes" in (out.stdout + out.stderr).lower()


@pytest.mark.parametrize("marker", sorted(R.TORCH_MARKERS))
def test_every_marker_is_a_bare_module_name(marker):
    assert marker and not marker.startswith("#") and " " not in marker
