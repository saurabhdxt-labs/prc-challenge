"""docs/fm1/CONFIG_REFERENCE.md is GENERATED from the knob schema (AT1). A test fails if any knob lacks documentation, or if
a documented knob does not exist, or if the committed reference is stale."""
from __future__ import annotations

import pathlib
import re

import pytest
import yaml

from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import refdoc as R

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = ROOT / "docs" / "fm1" / "CONFIG_REFERENCE.md"
ROW = re.compile(r"^\| `([a-z0-9_.]+)` \|", re.MULTILINE)


def test_generated_reference_has_one_row_per_knob():
    """Mutation: skip knobs whose status is not_ported in render_reference -> 5 knobs missing -> red."""
    text = R.render_reference(K.SCHEMA)
    rows = ROW.findall(text)
    assert sorted(rows) == sorted(K.KNOBS)
    assert len(rows) == len(set(rows))


def test_each_row_carries_default_range_effect_evidence_status():
    text = R.render_reference(K.SCHEMA)
    for k in K.SCHEMA:
        line = next(l for l in text.splitlines() if l.startswith(f"| `{k.path}` |"))
        cells = [c.strip() for c in line.strip("|").split(" | ")]
        assert len(cells) == 7, line
        path, kind, default, rng, status, effect, evidence = cells
        assert kind == k.kind and status == k.status
        assert default == R.fmt_default(k)
        assert rng == R.fmt_range(k) and rng
        assert effect.startswith(k.effect[:20].replace("|", "\\|"))
        assert k.evidence.split(" ")[0].split(":")[0] in evidence


def test_committed_reference_is_current():
    """The committed file equals the generator's output. Regenerate: python -m prc config-reference --write.
    Mutation: edit one default in knobs.SCHEMA without regenerating -> red."""
    assert REF.exists(), "run: /opt/homebrew/bin/python3.11 -B -m prc config-reference --write"
    assert REF.read_text() == R.render_reference(K.SCHEMA)


def test_every_documented_knob_exists():
    """A documented knob that does not exist (a stale or hand-added row) fails."""
    for path in ROW.findall(REF.read_text()):
        assert path in K.KNOBS, path


def test_check_detects_a_stale_file(tmp_path):
    p = tmp_path / "ref.md"
    assert R.check_reference(p) is False                 # missing
    p.write_text(R.render_reference(K.SCHEMA))
    assert R.check_reference(p) is True
    p.write_text(R.render_reference(K.SCHEMA) + "\n| `run.ghost` | bool | x | x | x | x | x |\n")
    assert R.check_reference(p) is False


def test_write_reference_creates_parent_and_is_idempotent(tmp_path):
    p = tmp_path / "docs" / "ref.md"
    R.write_reference(p)
    first = p.read_text()
    R.write_reference(p)
    assert p.read_text() == first == R.render_reference(K.SCHEMA)


def test_rendered_yaml_parses_to_the_defaults():
    """render_yaml produced configs/fm1.yaml's body; parsing it gives the all-defaults config."""
    raw = yaml.safe_load(R.render_yaml(K.SCHEMA))
    assert K.parse_config(raw).hash == K.parse_config({}).hash
    assert set(K.flatten(raw)) == set(K.KNOBS)


def test_pipes_in_text_are_escaped():
    k = K.Knob("a.b", "bool", True, effect="x | y is a long enough effect", evidence="plans/PREREG_fm1_2026_09_28.md")
    line = [l for l in R.render_reference((k,)).splitlines() if l.startswith("| `a.b`")][0]
    assert "x \\| y" in line
    assert len([c for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]) == 7


def test_render_refuses_a_broken_schema():
    bad = K.Knob("a.b", "bool", True, effect="short", evidence="plans/PREREG_fm1_2026_09_28.md")
    with pytest.raises(E.ConfigError, match="effect"):
        R.render_reference((bad,))


def test_header_says_the_evidence_column_names_the_unshipped_development_record():
    """Final-product audit 2026-10-06: the evidence column cites files under plans/ and reports/, which are not part of
    the repository; without a sentence saying so a reader takes them for missing files. Mutation: drop the sentence
    from render_reference -> red."""
    text = R.render_reference()
    header = " ".join(text.split("## ", 1)[0].split())          # the header is line-wrapped markdown
    assert "evidence" in header and "development record" in header and "not part of this repository" in header
