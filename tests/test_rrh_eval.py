"""Tests for scripts/analysis/rrh_eval.py (RRH holdout scorer, plans/PREREG_rrh_rome_row_hunt_2026_09_23.md)."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("rrh_eval", ROOT / "scripts/analysis/rrh_eval.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


def _mix(p, sp, nf):
    return np.maximum(p * sp + (1 - p) * nf, 1.0)


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    sp = rng.uniform(3600, 24000, n)
    nf = rng.uniform(600, 1500, n)
    fill = rng.random(n) < 0.45
    y = np.where(fill, sp, nf + rng.normal(0, 200, n))
    month = np.where(np.arange(n) < n // 2, 1, 7)
    day = np.array([f"2025-{m:02d}-{1 + i % 20:02d}" for i, m in enumerate(month)])
    p_R = np.clip(fill * 0.6 + rng.uniform(0, 0.4, n), 0, 1)
    return dict(p_R=p_R, sp=sp, nf=nf, y=y, month=month, day=day, fill=fill)


def test_all_nan_rule_scores_exactly_zero():
    f = _frame()
    r = E.score_rule(np.full(len(f["sp"]), np.nan), f["p_R"], f["sp"], f["nf"], f["y"], f["month"], f["day"], _mix)
    assert r["g"] == 0.0 and r["n_fired"] == 0
    assert r["verdict"] == "NOT WORKING"


def test_oracle_rule_is_working_and_positive():
    f = _frame()
    r = E.score_rule(f["fill"].astype(float), f["p_R"], f["sp"], f["nf"], f["y"], f["month"], f["day"], _mix)
    assert r["g"] > 0 and r["ci95"][0] > 0 and r["minus_top3"] > 0
    assert all(v >= 0 for v in r["by_month"].values())
    assert r["verdict"] == "WORKING"
    assert r["n_fired"] == len(f["sp"])


def test_inverted_oracle_is_negative_and_not_working():
    f = _frame()
    r = E.score_rule((~f["fill"]).astype(float), f["p_R"], f["sp"], f["nf"], f["y"], f["month"], f["day"], _mix)
    assert r["g"] < 0 and r["ci95"][1] < 0
    assert r["verdict"] == "NOT WORKING"


def test_g_is_in_board_units_and_matches_hand_computation():
    f = _frame()
    rule = np.where(np.arange(len(f["sp"])) % 2 == 0, f["fill"].astype(float), np.nan)
    p_new = np.where(np.isnan(rule), f["p_R"], rule)
    want = (((_mix(f["p_R"], f["sp"], f["nf"]) - f["y"]) ** 2).sum()
            - ((_mix(p_new, f["sp"], f["nf"]) - f["y"]) ** 2).sum()) / 344_841.0
    r = E.score_rule(rule, f["p_R"], f["sp"], f["nf"], f["y"], f["month"], f["day"], _mix)
    assert r["g"] == pytest.approx(want, rel=1e-12)
    assert r["n_fired"] == int((~np.isnan(rule)).sum())


def test_one_month_negative_blocks_working():
    """Oracle in Jan, inverted oracle in Jul: g may be positive overall but the both-months clause must fail."""
    f = _frame(n=400)
    jan = f["month"] == 1
    rule = np.where(jan, f["fill"].astype(float), np.where(f["fill"], 0.0, np.nan))
    r = E.score_rule(rule, f["p_R"], f["sp"], f["nf"], f["y"], f["month"], f["day"], _mix)
    assert r["by_month"]["7"] < 0
    assert r["verdict"] == "NOT WORKING"


def test_top3_driven_gain_is_not_working():
    """A rule that fixes only the 3 largest-stake rows: g > 0 but minus-top-3 == 0 -> NOT WORKING."""
    f = _frame()
    se = (_mix(f["p_R"], f["sp"], f["nf"]) - f["y"]) ** 2
    top = np.argsort(-se)[:3]
    rule = np.full(len(se), np.nan)
    rule[top] = f["fill"][top].astype(float)
    r = E.score_rule(rule, f["p_R"], f["sp"], f["nf"], f["y"], f["month"], f["day"], _mix)
    assert r["g"] > 0 and r["minus_top3"] <= 0
    assert r["verdict"] == "NOT WORKING"


def test_bonferroni_interval_is_wider_than_ci95():
    f = _frame()
    r = E.score_rule(f["fill"].astype(float), f["p_R"], f["sp"], f["nf"], f["y"], f["month"], f["day"], _mix)
    assert r["ci_bonf"][0] <= r["ci95"][0] and r["ci_bonf"][1] >= r["ci95"][1]


def test_bootstrap_is_deterministic_and_day_blocked():
    f = _frame()
    a = E.score_rule(f["fill"].astype(float), f["p_R"], f["sp"], f["nf"], f["y"], f["month"], f["day"], _mix)
    b = E.score_rule(f["fill"].astype(float), f["p_R"], f["sp"], f["nf"], f["y"], f["month"], f["day"], _mix)
    assert a["ci95"] == b["ci95"] and a["n_days"] == len(set(f["day"]))


@pytest.mark.parametrize("bad", [np.array([0.5, 1.2]), np.array([-0.1, 0.5])])
def test_rule_outside_unit_interval_is_refused(bad):
    with pytest.raises(ValueError, match="outside"):
        E.score_rule(bad, np.array([.5, .5]), np.array([5e3, 5e3]), np.array([900., 900.]), np.array([5e3, 900.]),
                     np.array([1, 7]), np.array(["2025-01-02", "2025-07-02"]), _mix)


def test_rule_of_wrong_length_is_refused():
    with pytest.raises(ValueError, match="length"):
        E.score_rule(np.array([0.5]), np.array([.5, .5]), np.array([5e3, 5e3]), np.array([900., 900.]),
                     np.array([5e3, 900.]), np.array([1, 7]), np.array(["2025-01-02", "2025-07-02"]), _mix)


def test_holdout_frame_refuses_discovery_months():
    rows = pd.DataFrame({"MVT_ID_mvt": [1, 2], "month": [1, 3]})
    with pytest.raises(ValueError, match="holdout"):
        E.check_holdout_months(rows)
    E.check_holdout_months(pd.DataFrame({"MVT_ID_mvt": [1, 2], "month": [1, 7]}))


def test_is_working_each_clause_in_isolation():
    """Mutation rehearsal 2026-09-23 (copy mirror): deleting the minus-top-3 clause or the both-months clause
    survived the score_rule tests (the CI clause failed first there); these isolate each clause."""
    ok = {"1": 1.0, "7": 0.5}
    assert E.is_working(0.1, 0.1, ok)
    assert not E.is_working(0.0, 0.1, ok)                       # CI lower must be > 0
    assert not E.is_working(0.1, 0.0, ok)                       # minus-top-3 must be > 0
    assert not E.is_working(0.1, 0.1, {"1": 1.0, "7": -1e-9})   # every month >= 0
    assert E.is_working(0.1, 0.1, {"1": 1.0, "7": 0.0})         # >= 0, not > 0


def test_band_split_sums_to_g_and_separates_the_owned_tail():
    """sp >= 24,000 is already committed to fill by the shipped chain (v30 p = 1.0 there), so the report must show
    how much of g lives in that band."""
    f = _frame()
    f["sp"][:10] = 30_000.0
    f["y"][:10] = 30_000.0
    f["fill"][:10] = True
    f["p_R"][:10] = 0.1
    r = E.score_rule(f["fill"].astype(float), f["p_R"], f["sp"], f["nf"], f["y"], f["month"], f["day"], _mix)
    assert r["g_sp_lt24000"] + r["g_sp_ge24000"] == pytest.approx(r["g"], rel=1e-12)
    assert r["g_sp_ge24000"] > 0


def test_verdict_rows_exclude_arc_reached_and_owned_tail():
    """RRH.1: rows ARC reaches (sub_rule 'ord'/'fill') and rows at sp >= 24,000 are production-owned."""
    sp = np.array([5e3, 5e3, 5e3, 5e3, 24_000.0, 23_999.0])
    arc = np.array(["", "none", "ord", "fill", "", ""])
    assert E.verdict_rows(sp, arc).tolist() == [True, True, False, False, False, True]


def test_verdict_rows_refuses_unknown_arc_code():
    with pytest.raises(ValueError, match="sub_rule"):
        E.verdict_rows(np.array([5e3]), np.array(["maybe"]))
