"""scripts/analysis/clp_checks.py — CLP's label-free acceptance (plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md, 1-3).

Mutation rehearsal per test in its docstring; results in tests/fm1/MUTATION_FM1.md (M-CLC rows).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("clp_checks", ROOT / "scripts" / "analysis" / "clp_checks.py")
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)


def _report(**changed):
    cells = {"EDDF|7": {"n": 100, "n_changed": 0, "mean_shift_s": 0.0},
             "LFPG|1": {"n": 90, "n_changed": 0, "mean_shift_s": 0.0},
             "LFPG|7": {"n": 120, "n_changed": 0, "mean_shift_s": 0.0}}
    for k, n in changed.items():
        cells[k.replace("_", "|")]["n_changed"] = n
    return {"rows": 310, "n_changed": sum(changed.values()), "cells": cells}


def _clip(fired=(("LFPG", "2026-07"),), n_clipped=7484, population="scored_file"):
    cells = [{"airport": "LFPG", "month": "2026-01", "n": 17804, "n_late": 1079, "share": 0.0606, "fired": False},
             {"airport": "LFPG", "month": "2026-07", "n": 21334, "n_late": 7484, "share": 0.3508, "fired": False}]
    for c in cells:
        c["fired"] = (c["airport"], c["month"]) in set(fired)
    return {"eobt_clip": {"airports": ["LFPG"], "population": population, "cells": cells, "n_clipped": n_clipped}}


def test_scope_passes_when_every_change_is_inside_the_allowed_cells():
    """The positive control. Mutation (2026-10-03): every changed cell counted as outside -> red."""
    r = X.scope(_report(LFPG_7=50), ["LFPG|7"])
    assert r["ok"] and r["changed_cells"] == {"LFPG|7": 50}


def test_scope_fails_on_a_change_outside_the_allowed_cells():
    """Mutation: compare against all cells instead of the changed ones -> red; ignore cells outside the list -> red."""
    r = X.scope(_report(LFPG_7=50, LFPG_1=1), ["LFPG|7"])
    assert not r["ok"] and r["outside"] == {"LFPG|1": 1}


def test_scope_fails_when_nothing_changed():
    """A clip that moved no row means the module never ran. Mutation: drop the n_changed > 0 term -> red."""
    assert not X.scope(_report(), ["LFPG|7"])["ok"]


def test_detector_passes_on_the_registered_shape():
    """The positive control. Mutation (2026-10-03): the fired-cell comparison always fails -> red."""
    r = X.detector(_clip())
    assert r["ok"] and r["fired"] == [["LFPG", "2026-07"]] and r["n_clipped"] == 7484


@pytest.mark.parametrize("kw, why", [
    ({"fired": (("LFPG", "2026-07"), ("LFPG", "2026-01"))}, "fired"),
    ({"fired": ()}, "fired"),
    ({"n_clipped": 7399}, "n_clipped"),
    ({"n_clipped": 7601}, "n_clipped"),
    ({"population": "batch"}, "population"),
])
def test_detector_fails_off_the_registered_shape(kw, why):
    """Mutation: drop the range check -> 7399 / 7601 pass -> red; drop the population check -> red."""
    r = X.detector(_clip(**kw))
    assert not r["ok"] and why in r["failed"]


def test_detector_fails_when_the_manifest_has_no_clip_report():
    """A build without the module is not v54. Mutation: default to an empty report that passes -> red."""
    r = X.detector({"decided_by": {}})
    assert not r["ok"] and "eobt_clip" in r["failed"]


def test_cli_exit_codes(tmp_path):
    """Exit 0 = pass, 1 = fail (the queue reads the code). Mutation (2026-10-03): always 0 -> red; inverted -> red."""
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_text(json.dumps(_report(LFPG_7=5)))
    bad.write_text(json.dumps(_report(LFPG_7=5, EDDF_7=2)))
    assert X.main(["scope", "--report", str(good), "--allow", "LFPG|7"]) == 0
    assert X.main(["scope", "--report", str(bad), "--allow", "LFPG|7"]) == 1
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(_clip()))
    assert X.main(["detector", "--manifest", str(m)]) == 0
