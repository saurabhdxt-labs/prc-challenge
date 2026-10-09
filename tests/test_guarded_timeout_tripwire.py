"""Tripwire for bug class BC-50 (reports/bug_classes.md): a subprocess timeout measured in wall-clock seconds expires
while tools/mem_guard.py has the whole pytest tree SIGSTOPped for a Phantom job (105-120 s on 2026-10-09; the 90 s
BC-5 test in tests/test_adsb_stack.py was killed at -9 and passes alone in 15 s; a 30 s usage test the day before).

The rule this file pins: in tests/ a child process is waited on through tests/isolation.py's `run_guarded` /
`wait_guarded`, whose timeout counts only the seconds this process was running, and a polling loop's deadline is an
`isolation.Deadline`. A `subprocess.run(...)`, `check_output`, `check_call`, `.wait(...)` or `.communicate(...)` with a
timeout (keyword or positional), a `time.time() + N` / `time.monotonic() + N` deadline, and a subprocess alias import
(which would hide the call) are refused. tests/isolation.py is
the implementation and this file names the shapes; neither is scanned; its own `isolated` goes through `run_guarded` (asserted below)."""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER = ROOT / "tests" / "isolation.py"
SELF = pathlib.Path(__file__).resolve()      # names the refused shapes in its docstring and its finder test
RAW = re.compile(r"\b(subprocess\.(run|check_output|check_call|call)|\.wait|\.communicate)\s*\(")
TIMEOUT_ARG = re.compile(r"\btimeout\s*=|^\s*[0-9]")                  # a keyword, or a bare number as the first argument
WALL_DEADLINE = re.compile(r"\btime\.(time|monotonic)\(\)\s*\+\s*[0-9]")  # `deadline = time.time() + N`
ALIAS = re.compile(r"^\s*(from subprocess import|import subprocess as)", re.M)   # hides the call from RAW


def _call_text(source: str, start: int) -> str:
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                return source[start:i + 1]
    return source[start:]


def raw_timeout_sites(source: str) -> list:
    """(line number, call head) for every raw subprocess wait in `source` that carries a `timeout=`."""
    out = []
    for m in RAW.finditer(source):
        args = _call_text(source, m.end() - 1)[1:-1]
        first = args.split(",")[0] if m.group(1) in (".wait", ".communicate") else ""
        if re.search(r"\btimeout\s*=", args) or (first and TIMEOUT_ARG.match(first) and m.group(1) == ".wait") \
                or (m.group(1) == ".communicate" and len(args.split(",")) > 1 and TIMEOUT_ARG.match(args.split(",")[1])):
            out.append((source.count("\n", 0, m.start()) + 1, m.group(1)))
    for m in WALL_DEADLINE.finditer(source):
        out.append((source.count("\n", 0, m.start()) + 1, "time." + m.group(1) + "() +"))
    for m in ALIAS.finditer(source):
        out.append((source.count("\n", 0, m.start()) + 1, m.group(1)))
    return sorted(out)


def test_finder_sees_each_raw_shape_and_ignores_the_guarded_ones():
    src = ("subprocess.run(a, timeout=3)\np.wait(timeout=2)\np.communicate(input=None,\n  timeout=4)\n"
           "subprocess.check_output(a, timeout=1)\nrun_guarded(a, timeout=3)\nwait_guarded(p, timeout=2)\n"
           "subprocess.run(a)\np.wait()\nopener(req, timeout=120)\np.wait(20)\np.communicate(None, 5)\n"
           "deadline = time.time() + 30\nd = time.monotonic() + 5.0\nfrom subprocess import run\nimport subprocess as sp\n"
           "p.communicate()\nx = time.time() - t0\n")
    assert raw_timeout_sites(src) == [(1, "subprocess.run"), (2, ".wait"), (3, ".communicate"), (5, "subprocess.check_output"),
                                      (11, ".wait"), (12, ".communicate"), (13, "time.time() +"), (14, "time.monotonic() +"),
                                      (15, "from subprocess import"), (16, "import subprocess as")]


def test_no_test_waits_on_a_child_with_a_raw_wall_clock_timeout():
    hits = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        if path.resolve() in (HELPER, SELF):
            continue
        for line, head in raw_timeout_sites(path.read_text()):
            hits.append(f"{path.relative_to(ROOT)}:{line} {head}(..., timeout=)")
    assert not hits, ("raw wall-clock subprocess timeouts, deadlines or subprocess aliases (use isolation.run_guarded / "
                      "wait_guarded / Deadline):\n" + "\n".join(hits))


def test_isolated_itself_waits_through_run_guarded():
    src = HELPER.read_text()
    start = src.index("def isolated(")
    end = src.find("\ndef ", start + 1)
    body = src[start:end if end > 0 else len(src)]
    assert "run_guarded(" in body and "subprocess.run(" not in body
