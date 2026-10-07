#!/usr/bin/env python3.11
"""Run the project's full test suite as two isolated passes — the project's "full suite green" entry point.

    /opt/homebrew/bin/python3.11 tools/run_suite.py            # both passes, quiet
    /opt/homebrew/bin/python3.11 tools/run_suite.py -x -k smoke   # extra pytest args reach BOTH passes

WHY (bug class BC-5, `reports/bug_classes.md`). pytest imports every collected test module into ONE interpreter.
Test files that import torch bring `torch/lib/libomp.dylib`; test files that import LightGBM — directly or through
`lgbm_fold` / `lgbm_submit` — bring Homebrew's `libomp.dylib`; sklearn brings a third. Two incompatible OpenMP
runtimes in one process is unsupported, and on this machine it is fatal: on 2026-09-17 it segfaulted the TNS smokes
(`EXC_BAD_ACCESS` inside `__kmp_fork_barrier`) and then deadlocked the full suite at 80 % for 43 minutes (main thread
in `torch … mul_kernel → __kmpc_fork_call → __kmp_join_barrier → __kmp_suspend_64`).

The fix is the same one the production stages use: one native model family per process. Pass 1 runs everything except
the torch-importing test files; pass 2 runs those files alone. Both must be green — the run's exit code is non-zero if
either pass fails, and both always run so a red first pass never hides the second's state.

Discovery is a SOURCE SCAN, not a hand-maintained list (`torch_test_files`): a new torch-importing test file is
isolated automatically. That property is what makes this survive the next module, and `tests/test_run_suite.py` pins it.
Refused alternatives: `KMP_DUPLICATE_LIB_OK=TRUE` (suppresses the duplicate-runtime check rather than removing the
duplication, documented to risk wrong results), and `OMP_NUM_THREADS=1` (hides the clash by disabling the parallel
region the fits need).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"

#: Importing any of these pulls `torch/lib/libomp.dylib` into the process.
TORCH_MARKERS = frozenset({"torch", "tabm_screen", "tree_neural_matched", "tree_neural_stack"})

#: `import X`, `import X as y`, `from X import …`, at any indentation (a torch import inside a fixture or a
#: function body loads the runtime just as surely as a module-level one).
_IMPORT = re.compile(
    r"^\s*(?:import\s+(?P<i>[A-Za-z_][\w.]*)|from\s+(?P<f>[A-Za-z_][\w.]*)\s+import\b)", re.MULTILINE)


def imported_modules(source: str) -> set:
    """Top-level names of every module imported by `source` (comments and string literals do not count)."""
    out = set()
    for m in _IMPORT.finditer(source):
        name = m.group("i") or m.group("f")
        out.add(name.split(".", 1)[0])
    return out


def torch_test_files(tests_dir: Path) -> list:
    """Every `test_*.py` under `tests_dir` that imports torch (directly or through a torch-carrying module), sorted."""
    hits = [p for p in sorted(Path(tests_dir).rglob("test_*.py"))
            if imported_modules(p.read_text(errors="replace")) & TORCH_MARKERS]
    return hits


def split_files(all_files, torch_files) -> tuple:
    """Partition `all_files` into (tree_pass, torch_pass); the two are disjoint and cover the input."""
    torch_set = set(torch_files)
    return [p for p in all_files if p not in torch_set], [p for p in all_files if p in torch_set]


def run_pass(label: str, args: list, extra: list) -> int:
    cmd = [sys.executable, "-m", "pytest", *args, "-p", "no:randomly", *extra]
    print(f"\n=== suite pass: {label} ===\n{' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def main(argv=None) -> int:
    # Every argument is forwarded verbatim to BOTH passes; argparse is not used for the pass-through because
    # pytest flags (-x, -k, -q, --ignore=…) must survive unchanged, including leading dashes.
    extra = list(sys.argv[1:] if argv is None else argv)
    if {"-h", "--help"} & set(extra):
        print(__doc__)
        print("Runs the suite as two isolated passes (bug class BC-5); every argument is forwarded to both.")
        return 0

    torch_files = torch_test_files(TESTS_DIR)
    tree_rc = run_pass("tree libraries (everything except the torch files)",
                       [TESTS_DIR, *(f"--ignore={p}" for p in torch_files)], extra)
    torch_rc = 0
    if torch_files:                        # an empty invocation would collect the whole suite and re-create the clash
        torch_rc = run_pass("torch only", list(torch_files), extra)
    else:
        print("\n=== suite pass: torch only — SKIPPED, no test file imports torch ===", flush=True)

    print(f"\n=== suite result: tree pass {'ok' if tree_rc == 0 else f'FAILED ({tree_rc})'}; "
          f"torch pass {'ok' if torch_rc == 0 else f'FAILED ({torch_rc})'} ===", flush=True)
    return tree_rc or torch_rc


if __name__ == "__main__":
    raise SystemExit(main())
