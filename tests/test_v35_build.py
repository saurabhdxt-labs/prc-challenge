"""Tests for scripts/analysis/v35_build.py — v35 = v34 with the R12 matched values (plans/PREREG_v35_r12_2026_09_26.md).
Written before the module (red first)."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("v35_build", ROOT / "scripts" / "analysis" / "v35_build.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v35_build"] = mod
    spec.loader.exec_module(mod)
    return mod


vb = _load()


def _base():
    return pd.DataFrame({"MVT_ID_mvt": np.array([10.0, 11.0, 12.0, 13.0]),
                         "TAXITIME_SEC_mvt": np.array([600, 700, 800, 900], dtype="int32")})


def test_compose_replaces_only_the_given_ids_rounded_with_a_one_second_floor():
    """Kills: touching rows outside vals; truncating instead of rint; no floor; a dtype change (int32 is v34's)."""
    vals = pd.DataFrame({"MVT_ID_mvt": [11.0, 13.0], "taxi": [650.6, -5.0]})
    out, rep = vb.compose(_base(), vals)
    assert out.TAXITIME_SEC_mvt.tolist() == [600, 651, 800, 1]
    assert out.TAXITIME_SEC_mvt.dtype == np.int32 and out.MVT_ID_mvt.tolist() == [10.0, 11.0, 12.0, 13.0]
    assert rep["n_given"] == 2 and rep["n_changed"] == 2


def test_compose_reports_move_energy_over_the_board_denominator():
    """S = sum((new - old)^2) / 344,841 (the board's row count). Kills: dividing by the changed count."""
    vals = pd.DataFrame({"MVT_ID_mvt": [10.0], "taxi": [610.0]})
    _, rep = vb.compose(_base(), vals)
    assert rep["S"] == pytest.approx(100.0 / 344_841) and rep["mean_shift_s"] == pytest.approx(10.0)


@pytest.mark.parametrize("vals, msg", [
    (pd.DataFrame({"MVT_ID_mvt": [99.0], "taxi": [5.0]}), "not in the base"),
    (pd.DataFrame({"MVT_ID_mvt": [10.0, 10.0], "taxi": [5.0, 6.0]}), "duplicate"),
    (pd.DataFrame({"MVT_ID_mvt": [10.0], "taxi": [np.nan]}), "finite"),
    (pd.DataFrame({"MVT_ID_mvt": [10.0], "taxi": [np.inf]}), "finite"),
])
def test_compose_refuses_unknown_duplicate_or_non_finite_values(vals, msg):
    """Kills: silently dropping an id the base does not have; last-write-wins on duplicates; NaN -> int garbage."""
    with pytest.raises(ValueError, match=msg):
        vb.compose(_base(), vals)


def test_compose_never_mutates_the_base_frame():
    """Kills: editing v34's frame in place (the verifier compares against it)."""
    b = _base()
    vb.compose(b, pd.DataFrame({"MVT_ID_mvt": [12.0], "taxi": [1.0]}))
    assert b.TAXITIME_SEC_mvt.tolist() == [600, 700, 800, 900]


def test_filter_airports_keeps_only_the_named_airports_rows():
    """Amendment A1: v35 applies R12 only at five airports. Kills: keeping rows of other airports; losing an id order."""
    vals = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "taxi": [5.0, 6.0, 7.0]})
    ap = pd.Series(["EDDF", "LEMD", "EHAM"], index=[1.0, 2.0, 3.0])
    out = vb.filter_airports(vals, ap, ("EDDF", "EHAM"))
    assert out.MVT_ID_mvt.tolist() == [1.0, 3.0] and out.taxi.tolist() == [5.0, 7.0]


def test_filter_airports_refuses_an_id_without_an_airport_or_an_unknown_airport_name():
    """Kills: silently dropping rows the ranking file cannot place; a typo'd airport keeping nothing."""
    vals = pd.DataFrame({"MVT_ID_mvt": [1.0, 9.0], "taxi": [5.0, 6.0]})
    ap = pd.Series(["EDDF"], index=[1.0])
    with pytest.raises(ValueError, match="airport"):
        vb.filter_airports(vals, ap, ("EDDF",))
    with pytest.raises(ValueError, match="unknown"):
        vb.filter_airports(vals.iloc[:1], ap, ("EDDX",))


def test_exclude_airport_months_drops_only_the_named_airport_month_rows():
    """Amendment A3: v37 keeps LSZH July-2026 at the base (v35) values. Kills: dropping LSZH January too; dropping
    another airport's July; a month read from the wrong clock."""
    vals = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], "taxi": [5.0, 6.0, 7.0, 8.0]})
    ap = pd.Series(["LSZH", "LSZH", "EDDF", "LSZH"], index=[1.0, 2.0, 3.0, 4.0])
    month = pd.Series([7, 1, 7, 7], index=[1.0, 2.0, 3.0, 4.0])
    out = vb.exclude_airport_months(vals, ap, month, [("LSZH", 7)])
    assert out.MVT_ID_mvt.tolist() == [2.0, 3.0]


def test_exclude_airport_months_refuses_a_bad_spec():
    """Kills: silently ignoring a typo'd airport or a month outside 1..12."""
    vals = pd.DataFrame({"MVT_ID_mvt": [1.0], "taxi": [5.0]})
    ap, month = pd.Series(["LSZH"], index=[1.0]), pd.Series([7], index=[1.0])
    with pytest.raises(ValueError, match="unknown"):
        vb.exclude_airport_months(vals, ap, month, [("LSZX", 7)])
    with pytest.raises(ValueError, match="month"):
        vb.exclude_airport_months(vals, ap, month, [("LSZH", 13)])


def test_prereg_path_is_recorded_as_given_and_must_be_an_existing_plan(tmp_path):
    """LDI (2026-09-27): the meta's prereg was hard-coded to the v35 plan, so a v39 built with this tool would cite the
    wrong registration. --prereg (default: the v35 plan, unchanged behaviour) must name an existing file under plans/.
    Fails when a missing or non-plans path is accepted, or the default changes."""
    V = vb
    assert V.PREREG_DEFAULT == "plans/PREREG_v35_r12_2026_09_26.md"
    (tmp_path / "plans").mkdir()
    (tmp_path / "plans" / "PREREG_x.md").write_text("x")
    assert V.check_prereg("plans/PREREG_x.md", root=tmp_path) == "plans/PREREG_x.md"
    with pytest.raises(FileNotFoundError, match="prereg"):
        V.check_prereg("plans/PREREG_missing.md", root=tmp_path)
    (tmp_path / "other.md").write_text("x")
    with pytest.raises(ValueError, match="plans/"):
        V.check_prereg("other.md", root=tmp_path)
