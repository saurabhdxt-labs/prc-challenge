"""Tests for scripts/analysis/orl_stage1.py (arm ORL Stage 1; plans/PREREG_orl_osm_route_length_2026_09_23.md).

The runner is thin by design: it reuses rcr2_build's leg masks, fit, paired statistics and gate shape. What is
new — the arm feature list, the join of the static route table onto the cache frame, the gate — is tested here on
synthetic frames BEFORE any real fit exists (BC-24). No test here fits a model.
"""
from __future__ import annotations

import importlib.util, pathlib, sys
import numpy as np, pandas as pd, pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(n, r):
    if n in sys.modules:
        return sys.modules[n]
    s = importlib.util.spec_from_file_location(n, ROOT / r); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


S1 = _load("orl_stage1", "scripts/analysis/orl_stage1.py")
O = _load("orl_build", "scripts/analysis/orl_build.py")


def test_imported_module_is_the_one_under_test():
    assert pathlib.Path(S1.__file__).resolve() == (ROOT / "scripts/analysis/orl_stage1.py").resolve()


def test_registration_constants():
    """Catches a leg, bar or seed moved after the hash."""
    assert list(S1.LEGS) == ["L-Jul", "L-Jan"]
    assert S1.LEGS["L-Jul"] == {"fit_months": (1, 2, 3, 4, 5, 6), "test_month": 7}
    assert S1.LEGS["L-Jan"] == {"fit_months": (8, 9, 10, 11, 12), "test_month": 1}
    assert (S1.LR, S1.MAX_ROUNDS, S1.PATIENCE) == (0.05, 8000, 200)
    assert S1.ARM_COLS == O.ORL_COLS + [O.ORL_FLAG]
    assert S1.PREREG.name == "PREREG_orl_osm_route_length_2026_09_23.md"


def test_arm_feats_is_the_production_93_plus_the_six_orl_columns():
    """Catches an ORL column dropped, a production column lost, or a label column smuggled in."""
    class FakeS:
        BASELINE_FEATS = [f"b{i}" for i in range(68)]; QUEUE_FEATS = [f"q{i}" for i in range(12)]
        ORDER_FEATS = ["o0", "o1", "o2"]; WEATHER_FEATS = [f"w{i}" for i in range(10)]
    ctrl, arm = S1.arm_feats(FakeS)
    assert len(ctrl) == 93 and arm == ctrl + S1.ARM_COLS and len(arm) == 99
    FakeS.WEATHER_FEATS = FakeS.WEATHER_FEATS[:-1] + ["y"]
    with pytest.raises(AssertionError, match="leak gate"):
        S1.arm_feats(FakeS)


def test_attach_to_cache_frame_joins_on_ap_stand_runway_and_keeps_row_order():
    """Catches the cache's `ap` column not being presented as ADEP_mvt (attach requires the organiser names)."""
    routes = pd.DataFrame({"icao": ["EDDF", "EDDF"], "stand": ["A1", "A1"], "rwy": ["18", "25C"],
                           "route_m": [1000.0, 2000.0], "n_edges": [2, 3], "straight_m": [900.0, 1900.0],
                           "stand_snap_m": [1.0, 1.0], "rwy_snap_m": [0.0, 0.0]})
    d = pd.DataFrame({"ap": ["EDDF", "EDDF", "LIRF"], "STAND_mvt": ["A1", "zz", "A1"], "RUNWAY_mvt": ["25C", "18", "25"],
                      "y": [1.0, 2.0, 3.0]})
    out = S1.attach_to_cache_frame(d, routes)
    assert list(out.columns) == list(d.columns) + S1.ARM_COLS and len(out) == 3
    assert out.orl_route_m.tolist()[0] == 2000.0 and np.isnan(out.orl_route_m.tolist()[1]) and np.isnan(out.orl_route_m.tolist()[2])
    assert out.orl_matched.tolist() == [1, 0, 0]
    assert "ADEP_mvt" not in out.columns, "the helper column must not leak into the design"


def test_attach_to_cache_frame_refuses_a_frame_without_the_keys():
    with pytest.raises(KeyError, match="STAND_mvt"):
        S1.attach_to_cache_frame(pd.DataFrame({"ap": ["EDDF"], "RUNWAY_mvt": ["25C"]}), pd.DataFrame())


def test_gate_g1_requires_both_legs_raw_and_minus_top3_nonnegative():
    """Catches a gate reading only one leg, or ignoring the remove-and-recompute clause."""
    ok = {"arm_vs_ctrl": {"gain_s": 0.1, "gain_minus_top3_s": 0.05}}
    bad_raw = {"arm_vs_ctrl": {"gain_s": -0.1, "gain_minus_top3_s": 0.05}}
    bad_top = {"arm_vs_ctrl": {"gain_s": 0.1, "gain_minus_top3_s": -0.01}}
    assert S1.gate_g1({"L-Jul": ok, "L-Jan": ok}) is True
    assert S1.gate_g1({"L-Jul": ok, "L-Jan": bad_raw}) is False
    assert S1.gate_g1({"L-Jul": bad_top, "L-Jan": ok}) is False
    assert S1.gate_g1({"L-Jul": ok, "L-Jan": {"arm_vs_ctrl": {"gain_s": 0.0, "gain_minus_top3_s": 0.0}}}) is True


def test_matched_share_tripwire_reports_per_leg_and_refuses_zero_matched_rows():
    """Catches a join that silently matched nothing (the arm would then be the control with six NaN columns)."""
    m = np.array([1, 1, 0, 1], dtype=np.int8)
    assert S1.matched_share(m) == 0.75
    with pytest.raises(AssertionError, match="no matched rows"):
        S1.matched_share(np.zeros(4, dtype=np.int8))


def test_main_refuses_unknown_args():
    with pytest.raises(SystemExit):
        S1.main(["--bogus"])
