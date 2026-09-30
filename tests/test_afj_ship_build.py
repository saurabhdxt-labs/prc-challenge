"""Tests for scripts/analysis/afj_ship_build.py — SHIP-AFJ's row selection, base reconstruction, splice and move-size rules
(plans/PREREG_adsbfi_july_fallback_2026_09_16.md §4, §6, Amendment AFJ.1). No model is loaded here. Each test names the
mutation it must catch.

Mutation rehearsal 2026-09-16 17:08 EDT (bytecode cache cleared before every run; backup, break, run, restore, GREEN again):
gate ignored, floor row moved, staged-in-v13 candidate not refused, concentration counted by rows, stake sign, base sign,
dtype widened -> each RED. The G3 outside-row assertion in splice() SURVIVES its removal: splice writes only at the named
positions, so the branch is unreachable by construction (an equivalent mutant, kept as a belt-and-braces check).
Lesson recorded the same day: a same-size restore within the same second was served from a stale .pyc (proxy - v13 read as
v13 - proxy); rehearsals now clear scripts/**/__pycache__ before every run.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("afj_ship_build", ROOT / "scripts" / "analysis" / "afj_ship_build.py")
sb = importlib.util.module_from_spec(_spec)
sys.modules["afj_ship_build"] = sb
_spec.loader.exec_module(sb)

GATE = {"LEMD": {"allowed": True, "rule": "r"}, "LIRF": True, "LSZH": {"allowed": False}}


def _world():
    """Eight scored matched rows in the row map: 1 AFJ (LEMD, fi, cov 3); 2 AFJ (LIRF, fi, cov 2); 3 fi but cov 1; 4 adsb.lol row;
    5 fi cov 2 at LSZH (gate closed); 6 fi cov 2 LEMD but v13 at the floor; 7 fi cov 3 LEMD but staged in v13 (a contradiction);
    8 fi cov 2 LEMD whose v4 coverage is 2 (never a fallback row)."""
    src = pd.DataFrame({"MVT_ID_mvt": np.arange(1.0, 9.0), "source": ["adsb.fi", "adsb.fi", "adsb.fi", "adsb.lol", "adsb.fi", "adsb.fi", "adsb.fi", "adsb.fi"],
                        "ap": ["LEMD", "LIRF", "LEMD", "LEMD", "LSZH", "LEMD", "LEMD", "LEMD"],
                        "coverage_v4": [1, 1, 1, 3, 1, 0, 1, 2], "coverage_afj": [3, 2, 1, 3, 2, 2, 3, 2]})
    stage = pd.Series([sb.NO_STAGE] * 6 + [sb.NO_STAGE + "+adsb", sb.NO_STAGE], index=np.arange(1.0, 9.0))
    v13 = pd.Series([900.0, 800, 700, 650, 600, 1, 500, 400], index=np.arange(1.0, 9.0))
    return src, stage, v13


def test_afj_rows_are_fi_rows_newly_joined_at_gated_airports_above_the_v13_floor():
    """Rows 1 and 2 qualify; 3 (coverage 1), 4 (adsb.lol), 5 (gate closed), 8 (v4 coverage 2) do not; 6 is a candidate at the
    1 s floor and is counted, not taken. Fails when the gate is ignored, when coverage 1 rows are taken, when the floor row
    is moved (F cannot be recovered), or when the counts drift."""
    src, stage, v13 = _world()
    src7 = src[src.MVT_ID_mvt != 7.0]
    ids, counts = sb.afj_rows(src7, GATE, stage, v13)
    assert ids.tolist() == [1.0, 2.0]
    assert counts == {"n_candidates": 4, "n_candidates_gated": 3, "n_at_v13_floor_left_at_v17": 1, "n_afj_rows": 2}


def test_afj_rows_refuse_a_candidate_already_staged_in_v13_and_a_candidate_missing_from_v13():
    """A candidate whose v13 stage is `+adsb` had v4 coverage >= 2 and cannot be a fallback row: the row map or the provenance
    is wrong; a candidate absent from v13 is refused too. Fails when either is silently skipped."""
    src, stage, v13 = _world()
    with pytest.raises(ValueError, match="already staged in v13"):
        sb.afj_rows(src, GATE, stage, v13)
    src7 = src[src.MVT_ID_mvt != 7.0]
    with pytest.raises(ValueError, match="missing from v13"):
        sb.afj_rows(src7, GATE, stage.drop(2.0), v13)
    with pytest.raises(TypeError, match="gate entry"):
        sb.afj_rows(src7, {"LEMD": "yes", "LIRF": True, "LSZH": False}, stage, v13)


def test_base_from_v13_is_proxy_minus_v13_and_refuses_the_floor():
    """Fails when the sign flips or a floor row is reconstructed."""
    assert sb.base_from_v13([1500.0, 1200.0], [900.0, 1100.0]).tolist() == [600.0, 100.0]
    with pytest.raises(ValueError, match="floor"):
        sb.base_from_v13([1500.0, 1200.0], [900.0, 1.0])


def test_splice_changes_only_the_named_rows_keeps_dtype_and_order_and_refuses_bad_values():
    """Fails when a row outside the set changes, when the dtype widens (int32 -> float), or when a non-integer / sub-1 value
    gets through."""
    v17 = pd.DataFrame({"MVT_ID_mvt": [10.0, 11, 12, 13], "TAXITIME_SEC_mvt": np.array([900, 800, 700, 600], dtype=np.int32)})
    cand, touched = sb.splice(v17, [11.0, 13.0], [780.0, 600.0])
    assert cand.TAXITIME_SEC_mvt.tolist() == [900, 780, 700, 600] and str(cand.TAXITIME_SEC_mvt.dtype) == "int32"
    assert touched.tolist() == [False, True, False, False] and cand.MVT_ID_mvt.tolist() == v17.MVT_ID_mvt.tolist()
    assert v17.TAXITIME_SEC_mvt.tolist() == [900, 800, 700, 600]                       # input untouched
    for ids, vals, msg in (([11.0, 11.0], [1.0, 2.0], "duplicate"), ([99.0], [5.0], "not in v17"), ([11.0], [780.5], "not an integer"),
                           ([11.0], [0.0], "below 1"), ([11.0], [np.nan], "non-finite")):
        with pytest.raises(ValueError, match=msg):
            sb.splice(v17, ids, vals)


def test_move_split_reports_squared_shares_the_two_registered_rules_and_the_noise_stake():
    """Ten moves: nine of 50 s and one of 700 s. Squared share > 300 s = 700² / (9·50² + 700²) = 95.6 % (> 47.8 % bar: fails);
    rows > 600 s = 10 % (> 0.19 %: fails); stake = −Σ move² / 344,841. A benign set passes both. Fails when shares are counted by
    rows instead of squared movement, when a bar moves, or when the stake sign flips."""
    move = np.array([50.0] * 9 + [700.0])
    s = sb.move_split(move, -move)
    t = s["total"]
    assert t["n"] == 10 and t["n_gt300"] == 1 and t["n_gt600"] == 1 and t["pct_rows_gt600"] == pytest.approx(10.0)
    assert t["sq_share_gt300_pct"] == pytest.approx(100 * 700 ** 2 / (9 * 50 ** 2 + 700 ** 2))
    assert t["stake_if_pure_noise_board_mse"] == pytest.approx(-(9 * 50 ** 2 + 700 ** 2) / 344_841)
    assert s["rules"] == {"concentration_gt300_le_47_8pct": False, "cap_rows_gt600_le_0_19pct": False, "recommendation": "do not upload"}
    assert s["stage_only"]["mean_abs_s"] == t["mean_abs_s"]
    ok = sb.move_split(np.array([40.0, -60.0, 120.0, -30.0]), np.array([40.0, -60.0, 120.0, -30.0]))
    assert ok["rules"]["recommendation"] == "owner decision" and ok["total"]["sq_share_gt300_pct"] == 0.0
    assert sb.CONC_BAR_PCT == 47.8 and sb.CAP_GT600_PCT == 0.19
    empty = sb.move_split(np.array([]), np.array([]))
    assert empty["total"]["n"] == 0 and empty["total"]["stake_if_pure_noise_board_mse"] == 0.0
