"""Generate docs/fm1/CONFIG_REFERENCE.md (and the body of configs/fm1.yaml) from the knob schema.

    /opt/homebrew/bin/python3.11 -B -m prc config-reference --write     # regenerate
    /opt/homebrew/bin/python3.11 -B -m prc config-reference --check     # exit 1 if the committed file is stale

The reference is never edited by hand: tests/fm1/test_fm1_refdoc.py fails when it differs from render_reference().
"""
from __future__ import annotations

import json
import pathlib

from . import knobs as K

REFERENCE = K.ROOT / "docs" / "fm1" / "CONFIG_REFERENCE.md"
SECTIONS = ("run", "data", "airports", "lanes", "features", "adsb", "matched", "unmatched", "modules", "calibration",
            "validate")
STATUS_NOTE = {
    "implemented": "runs in FM1",
    "provisional": "runs, but its reproduction of the shipped component is not verified yet",
    "not_ported": "declared; `fit`/`predict` refuse while it is switched on (ComponentNotPortedError)",
    "schema_only": "validated and recorded, not read by any stage yet",
}


def _cell(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def fmt_default(k: K.Knob) -> str:
    v = k.default
    if isinstance(v, bool):
        return "`true`" if v else "`false`"
    if isinstance(v, dict):
        return "`" + ", ".join(f"{a}: {b}" for a, b in sorted(v.items())) + "`"
    if isinstance(v, (list, tuple)):
        return "`[" + ", ".join(str(x) for x in v) + "]`"
    return f"`{v}`"


def fmt_range(k: K.Knob) -> str:
    if k.kind == "bool":
        return "true / false"
    if k.kind == "choice":
        return " / ".join(f"`{c}`" for c in k.choices)
    if k.kind in ("int", "float"):
        return f"[{k.lo:g}, {k.hi:g}]"
    if k.kind == "int_list":
        return f"non-empty list of integers in [{k.lo:g}, {k.hi:g}]" + (", no repeats" if k.unique else "")
    if k.kind == "str_list":
        return "non-empty list of ICAO codes, no repeats" if k.path == "airports.codes" else "non-empty list of strings"
    if k.kind == "path":
        return "path (relative to the repo root, or absolute); never under `submissions/`"
    if k.kind == "tz_map":
        return "airport -> IANA zone, exactly airports.codes"
    return "non-empty string"


def render_reference(schema=K.SCHEMA) -> str:
    K.check_schema(schema)
    lines = [
        "# FM1 config reference",
        "",
        "GENERATED from `prc/pipeline/fm1/knobs.py` by `python -m prc config-reference --write`. Do not edit by hand:",
        "`tests/fm1/test_fm1_refdoc.py` fails when this file differs from the schema.",
        "",
        "One config, `configs/fm1.yaml`, states every knob at its default (the documented forecaster). A run that differs",
        "says so with `--set path=value` (validated like the file and recorded in the resolved-config hash), for example the",
        "scored-period build: `--set run.purpose=scored_period`.",
        "",
        "Status: " + "; ".join(f"`{s}` = {n}" for s, n in STATUS_NOTE.items()) + ".",
        "",
        "The evidence column names where each value was established: a line of code in this repository, or a file of",
        "the development record (registrations, reports and run logs under `plans/`, `reports/`, `logs/`), which is not",
        "part of this repository.",
        "",
    ]
    by_section: dict = {}
    for k in schema:
        by_section.setdefault(k.path.split(".")[0], []).append(k)
    order = [s for s in SECTIONS if s in by_section] + sorted(s for s in by_section if s not in SECTIONS)
    for sec in order:
        lines += [f"## `{sec}`", "", "| knob | type | default | range | status | effect | evidence |",
                  "|---|---|---|---|---|---|---|"]
        for k in by_section[sec]:
            lines.append("| " + " | ".join([f"`{k.path}`", k.kind, _cell(fmt_default(k)), _cell(fmt_range(k)), k.status,
                                            _cell(k.effect), _cell(f"`{k.evidence}`")]) + " |")
        lines.append("")
    return "\n".join(lines)


def check_reference(path=REFERENCE) -> bool:
    path = pathlib.Path(path)
    return path.exists() and path.read_text() == render_reference(K.SCHEMA)


def write_reference(path=REFERENCE) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_reference(K.SCHEMA))
    return path


def _yaml_str(v: str) -> str:
    """v as YAML text that parses back to the same string (YAML 1.1 reads a bare on/off/yes/no as a bool)."""
    import yaml

    return v if yaml.safe_load(v) == v and not any(c in v for c in "*:#{}[]") else json.dumps(v)


def render_yaml(schema=K.SCHEMA) -> str:
    """The all-defaults config as YAML, one knob per line, in schema order, with each knob's status when not
    `implemented` as a trailing comment (the body of configs/fm1.yaml)."""
    K.check_schema(schema)
    out, open_sections = [], []
    for k in schema:
        parts = k.path.split(".")
        common = 0
        while common < min(len(open_sections), len(parts) - 1) and open_sections[common] == parts[common]:
            common += 1
        open_sections = open_sections[:common]
        for depth in range(common, len(parts) - 1):
            if depth == 0:
                out.append("")
            out.append("  " * depth + f"{parts[depth]}:")
            open_sections.append(parts[depth])
        v = k.default
        if isinstance(v, dict):
            text = "{" + ", ".join(f"{a}: {b}" for a, b in sorted(v.items())) + "}"
        else:
            text = json.dumps(v) if not isinstance(v, str) else _yaml_str(v)
        note = "" if k.status == "implemented" else f"   # {k.status.upper().replace('_', ' ')}"
        out.append("  " * (len(parts) - 1) + f"{parts[-1]}: {text}{note}")
    return "\n".join(out).lstrip("\n") + "\n"
