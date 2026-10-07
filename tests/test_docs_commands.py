"""Every command the public documents show (README.md, docs/fm1/REPRODUCE.md) names a script that exists, uses only
flags its parser knows, and names every flag its parser requires.

Closing audit 2026-10-07 (reviewer M4): REPRODUCE.md showed `r12_stack.py oof --val … --tables … --arms … --out …`
without the required `--rss-limit-gib`, and `r12_validate.py --arms cat --screen --seeds 0,1,2` without the required
`--split`, `--tables`, `--out` and `--rss-limit-gib`; both stop in argparse with a usage error, and no test read the
documents. Seen red on that text (those two and four more incomplete commands), green once the commands were completed.

What counts as a command: in a fenced code block, a line that runs `python … -m prc`, `python … scripts/…py` or
`python … tools/…py`; in prose, a backticked span (it may wrap over lines) that runs one of those, or a bare
`scripts/…py …` span that carries a flag. A span with no flag at all is a mention, not a runnable command: its script
and subcommand are checked, its required flags are not. Placeholders (`<dir>`, `…`, `...`), comments and the optional
parts a document brackets are ignored; only flag names are compared. Required flags are read from the parser's own
usage line (argparse brackets every optional group and parenthesises a required choice). Mutation rehearsal
2026-10-07: `--rss-limit-gib` removed again from the r12_stack command -> red (names the command and the flag);
restored -> green."""
from __future__ import annotations

import pathlib
import re
import shlex
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOCS = ("README.md", "docs/fm1/REPRODUCE.md")
_PY = r"python(?:3(?:\.11)?)?(?: -B)?"
_CMD_START = re.compile(rf"^(?:{_PY} -m prc\b|(?:{_PY} )?(?:scripts|tools)/[\w/.-]+\.py)")
_FLAG = re.compile(r"(?<![\w-])--[a-z][\w-]*")
_SUBCOMMAND = re.compile(r"^[a-z][a-z0-9-]*$")


def _normalise(cmd: str) -> str:
    cmd = re.sub(r"<[^>]*>", "X", cmd)                          # <dir>, <N>, <year >= 2025>
    cmd = re.sub(r"\[[^\[\]]*\]", " ", cmd)                       # optional parts the document brackets
    cmd = cmd.replace("…", " X ").replace("...", " X ")
    return " ".join(shlex.split(cmd, comments=True))


def documented_commands(text: str) -> list[str]:
    """The runnable command strings in `text`: fenced-block lines and backticked spans, normalised."""
    out = []
    for block in re.findall(r"```(?:bash|sh)?\n(.*?)```", text, re.S):
        for line in block.replace("\\\n", " ").splitlines():
            line = line.strip()
            if _CMD_START.match(line):
                out.append(_normalise(line))
    prose = re.sub(r"```.*?```", " ", text, flags=re.S)
    for span in re.findall(r"`([^`]+)`", prose):
        span = " ".join(span.split())
        if _CMD_START.match(span) and (span.startswith("python") or "--" in span):
            out.append(_normalise(span))
    return out


def _target(cmd: str) -> tuple[list[str], str | None, list[str]]:
    """(argv prefix that runs the parser, subcommand or None, flags the command uses)."""
    toks = cmd.split()
    i = 0
    while toks[i].startswith("python") or toks[i] == "-B":
        i += 1
    if toks[i] == "-m":
        prefix, rest = [sys.executable, "-m", toks[i + 1]], toks[i + 2:]
    else:
        prefix, rest = [sys.executable, "-B", str(ROOT / toks[i])], toks[i + 1:]
    if "--" in rest:
        rest = rest[:rest.index("--")]
    sub = rest[0] if rest and _SUBCOMMAND.match(rest[0]) and rest[0] != "X" else None
    return prefix, sub, sorted(set(_FLAG.findall(" ".join(rest))))


_HELP: dict = {}


def help_text(prefix: list[str], sub: str | None):
    """The parser's --help output for the target, or None when --help fails; cached per target."""
    key = (tuple(prefix), sub)
    if key not in _HELP:
        argv = prefix + ([sub] if sub else []) + ["--help"]
        r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=120)
        _HELP[key] = r.stdout + r.stderr if r.returncode == 0 else None
    return _HELP[key]


def required_flags(help_out: str) -> tuple[set, list]:
    """(flags outside any bracket in argparse's usage block, groups of alternatives one of which is required)."""
    usage = help_out.split("\n\n")[0].replace("\n", " ")
    while re.search(r"\[[^\[\]]*\]", usage):
        usage = re.sub(r"\[[^\[\]]*\]", " ", usage)
    choices = [set(_FLAG.findall(g)) for g in re.findall(r"\(([^()]*\|[^()]*)\)", usage)]
    usage = re.sub(r"\([^()]*\)", " ", usage)
    return set(_FLAG.findall(usage)) - {"--help"}, [c for c in choices if c]


def known_flags(help_out: str) -> set:
    return set(_FLAG.findall(help_out))


@pytest.mark.slow
def test_every_documented_command_exists_and_names_its_required_flags():
    problems, seen = [], 0
    for doc in DOCS:
        for cmd in documented_commands((ROOT / doc).read_text()):
            seen += 1
            prefix, sub, used = _target(cmd)
            if prefix[1] == "-B" and not pathlib.Path(prefix[2]).is_file():
                problems.append(f"{doc}: `{cmd}` names a script that does not exist")
                continue
            out = help_text(prefix, sub)
            if out is None and sub is not None:
                sub, out = None, help_text(prefix, None)                 # the token was a value, not a subcommand
            if out is None:
                problems.append(f"{doc}: `{cmd}`: --help fails")
                continue
            unknown = sorted(f for f in used if f not in known_flags(out))
            if unknown:
                problems.append(f"{doc}: `{cmd}` uses flags the parser does not know: {', '.join(unknown)}")
            if used:                                                       # a runnable command, not a mention
                required, choices = required_flags(out)
                missing = sorted(required - set(used))
                missing += [" | ".join(sorted(c)) for c in choices if not c & set(used)]
                if missing:
                    problems.append(f"{doc}: `{cmd}` leaves out required flags: {', '.join(missing)}")
    assert seen >= 15, f"only {seen} commands found in {DOCS}; the extraction broke"
    assert problems == [], "\n".join(problems)
