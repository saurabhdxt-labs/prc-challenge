"""scripts/analysis/acr_price.py (plans/PREREG_acr_adsb_coverage_regime_2026_10_02.md): the scenario masking, the
statistics, the registered verdict rule and the M0 / M1 pairing check. Written after the code (stated); each test's
mutation rehearsal is in its docstring.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import features as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("acr_price", ROOT / "scripts" / "analysis" / "acr_price.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
COLS = ["joined", "dt_takeoff", "first_move_rel_mvt", "ground_seen_mvt_m10", "moving_mvt_m10", "coverage",
        "adsb_day_present"]


def _table(n_ap: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(5)
    aps = np.repeat(["LIRF", "EDDF", "EHAM"], n_ap)
    n = len(aps)
    return pd.DataFrame({"month": np.tile([7, 7, 7, 1], n // 4 + 1)[:n], "pos": np.arange(n), "MVT_ID_mvt": np.arange(n, dtype=float),
                         "ap": aps, "joined": (rng.random(n) < 0.9).astype(float), "dt_takeoff": 3.0,
                         "first_move_rel_mvt": -500.0, "ground_seen_mvt_m10": 11.0, "moving_mvt_m10": 2.0, "coverage": 3.0,
                         "adsb_day_present": 1.0})


def test_the_scenario_blacks_out_lirf_july_thins_eddf_july_and_leaves_the_rest():
    """LIRF July: every row a network blackout; EDDF July: ~33.7 % of joined rows unheard (counts kept); EHAM and every
    January row untouched; the input is not modified; seeded. Mutation: blackout without network=True -> LIRF counts kept
    -> red; EDDF share over all rows (not joined) -> red; mask every month -> red."""
    t = _table()
    before = t.copy()
    s = A.scenario_table(t, COLS, 0, F.set_unheard)
    pd.testing.assert_frame_equal(t, before)
    lj = (s.ap == "LIRF") & (s.month == 7)
    assert (s.joined[lj] == 0).all() and (s.ground_seen_mvt_m10[lj] == 0).all() and s.moving_mvt_m10[lj].isna().all()
    ej = (t.ap == "EDDF") & (t.month == 7) & (t.joined == 1)
    hit = ej & (s.joined == 0)
    assert abs(hit.sum() / ej.sum() - A.PARTIAL["EDDF"]) < 0.06
    assert (s.ground_seen_mvt_m10[hit] == 11).all()
    keep = (t.ap == "EHAM") | (t.month == 1)
    pd.testing.assert_frame_equal(s[keep], t[keep])
    s2 = A.scenario_table(t, COLS, 0, F.set_unheard)
    pd.testing.assert_frame_equal(s, s2)
    assert not s.joined.equals(A.scenario_table(t, COLS, 1, F.set_unheard).joined)


def test_stats_returns_d0_g_k_with_point_and_interval():
    """D0 = scenario minus as-is for M0, G = M0 minus M1 under the scenario (mean over seeds), K = M1 minus M0 as-is;
    each {point, lo, hi}. Mutation: G with the sign reversed -> red; seeds not averaged (last seed only) -> red."""
    day = np.array(["a", "a", "b", "b"])
    e0, e1 = np.array([1.0, 1.0, 1.0, 1.0]), np.array([2.0, 1.0, 1.0, 1.0])
    s0 = [np.array([5.0, 1.0, 1.0, 1.0]), np.array([3.0, 1.0, 1.0, 1.0])]
    s1 = [np.array([2.0, 1.0, 1.0, 1.0]), np.array([2.0, 1.0, 1.0, 1.0])]
    ci = lambda d, day: (float(np.sum(d)), float(np.sum(d)) - 1, float(np.sum(d)) + 1)  # noqa: E731
    r = A.stats(e0, s0, e1, s1, day, ci)
    assert r["D0"] == {"point": 3.0, "lo": 2.0, "hi": 4.0}
    assert r["G"]["point"] == 2.0 and r["K"]["point"] == 1.0


@pytest.mark.parametrize("g, glo, k, want", [(150.0, 10.0, 75.0, "TRUE"), (150.0, 10.0, 75.1, "FALSE"),
                                             (99.9, 10.0, 0.0, "FALSE"), (150.0, -1.0, 0.0, "FALSE")])
def test_verdict_is_the_registered_rule(g, glo, k, want):
    """G >= 100 with its CI above 0, and K <= G / 2. Mutation: K < G (not G / 2) -> the second case passes -> red;
    drop the CI condition -> the fourth passes -> red."""
    res = {"G": {"point": g, "lo": glo, "hi": g + 50}, "K": {"point": k, "lo": k - 5, "hi": k + 5}}
    assert A.verdict(res)["H_ACR"] == want


def test_check_pair_allows_only_the_dropout_knobs_to_differ():
    """Mutation: compare nothing -> a learning-rate difference passes -> red."""
    knobs = ("adsb.train_dropout.flight", "adsb.train_dropout.day", "run.artifacts_dir", "matched.lightgbm.learning_rate")
    c0 = {"adsb.train_dropout.flight": 0.0, "adsb.train_dropout.day": 0.0, "run.artifacts_dir": "a",
          "matched.lightgbm.learning_rate": 0.05}
    c1 = dict(c0, **{"adsb.train_dropout.flight": 0.3, "adsb.train_dropout.day": 0.15, "run.artifacts_dir": "b"})
    A.check_pair(c0, c1, knobs)
    with pytest.raises(ValueError, match="learning_rate"):
        A.check_pair(c0, dict(c1, **{"matched.lightgbm.learning_rate": 0.02}), knobs)
    with pytest.raises(ValueError, match="M1 must train with dropout"):
        A.check_pair(c0, c0, knobs)
