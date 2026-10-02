"""scripts/analysis/drf_audit.py — the label-free 2025 -> 2026 input drift audit (plans/PREREG_drf_feature_drift_2026_10_02.md).
Synthetic frames only; the real feature frames are read by the CLI.

Seen red first 2026-10-02 (collection error, module absent). Mutation rehearsal 2026-10-02 (sources restored): 13 of 13
RED — NaN ranked highest, min ranks for ties, the A-side rank sum, empty-side check off, signed AUC for the distance
(survived until the downward-shift assertion), each of the three flag clauses dropped, smallest neighbour as the null,
zero-gain check off, the second side's row minimum dropped (survived until the Z case), floor 0.20, unseeded folds."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("drf_audit", ROOT / "scripts/analysis/drf_audit.py")
D = importlib.util.module_from_spec(_spec)
sys.modules["drf_audit"] = D
_spec.loader.exec_module(D)


# ---------------------------------------------------------------- uni_auc

def test_uni_auc_is_the_probability_that_b_ranks_above_a_with_ties_at_half():
    assert D.uni_auc(np.array([1.0, 2.0]), np.array([3.0, 4.0])) == 1.0
    assert D.uni_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 0.0
    assert D.uni_auc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5
    # pairs: (1,2) b>a, (1,1) tie, (3,2) b<a, (3,1) b<a -> (1 + 0.5 + 0 + 0) / 4
    assert D.uni_auc(np.array([1.0, 3.0]), np.array([2.0, 1.0])) == pytest.approx(0.375)


def test_uni_auc_ranks_nan_lowest_so_a_changed_missing_rate_is_drift():
    """Same finite values, but B is half missing: NaN ranks below every value, so B ranks lower. Mutation: drop the NaN
    rows instead -> 0.5 -> red."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([np.nan, np.nan, 3.0, 4.0])
    assert D.uni_auc(a, b) < 0.5
    assert D.uni_auc(np.array([np.nan, np.nan]), np.array([np.nan])) == 0.5      # all missing on both sides: no drift


def test_uni_auc_agrees_with_a_brute_force_count():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 6, 300).astype(float), rng.integers(1, 8, 200).astype(float)
    brute = np.mean([(1.0 if y > x else 0.5 if y == x else 0.0) for x in a for y in b])
    assert D.uni_auc(a, b) == pytest.approx(brute, abs=1e-12)


def test_uni_auc_refuses_an_empty_side():
    with pytest.raises(ValueError, match="empty"):
        D.uni_auc(np.array([]), np.array([1.0]))


# ---------------------------------------------------------------- drift / tables

def _frame(n, shift=0.0, seed=0, nan_share=0.0):
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"same": rng.normal(size=n), "moved": rng.normal(size=n) + shift, "const": np.ones(n)})
    if nan_share:
        f.loc[rng.random(n) < nan_share, "same"] = np.nan
    return f


def test_drift_is_the_distance_of_the_auc_from_one_half_per_column():
    a, b = _frame(4000, seed=1), _frame(4000, shift=1.0, seed=2)
    d = D.drift(a, b, ["same", "moved", "const"])
    assert list(d.index) == ["same", "moved", "const"]
    assert d["same"] < 0.03 and d["const"] == 0.0
    assert d["moved"] == pytest.approx(0.26, abs=0.03)          # AUC of two unit normals one sd apart = 0.76
    # a downward move is drift of the same size (the distance from one half, not the signed AUC)
    down = D.drift(b, a, ["moved"])["moved"]
    assert down == pytest.approx(d["moved"], abs=1e-12) and down > 0.2


def test_cell_table_takes_the_largest_neighbour_as_the_null_and_applies_the_flag_rule():
    a = _frame(4000, seed=1)
    b = _frame(4000, shift=1.0, seed=2)
    n1 = _frame(4000, shift=0.1, seed=3)
    n2 = _frame(4000, shift=0.3, seed=4)
    gain = pd.Series({"same": 0.5, "moved": 0.4, "const": 0.1})
    t = D.cell_table(a, b, [n1, n2], ["same", "moved", "const"], gain)
    assert list(t.columns) == ["drift", "null", "excess", "gain_share", "flag"]
    assert t.loc["moved", "null"] == pytest.approx(D.drift(a, n2, ["moved"])["moved"])        # the larger neighbour
    assert t.loc["moved", "excess"] == pytest.approx(t.loc["moved", "drift"] - t.loc["moved", "null"])
    assert bool(t.loc["moved", "flag"]) is True and not t.loc[["same", "const"], "flag"].any()


@pytest.mark.parametrize("drift,null,gain,expect", [
    (0.10, 0.05, 0.01, True),            # exactly at every threshold
    (0.099, 0.0, 0.5, False),            # below the effect-size floor
    (0.30, 0.251, 0.5, False),           # within 0.05 of what adjacent months already show
    (0.30, 0.25, 0.5, True),
    (0.40, 0.0, 0.0099, False),          # a feature the model barely uses
])
def test_flag_rule_is_the_registered_one(drift, null, gain, expect):
    out = D.flag(np.array([drift]), np.array([null]), np.array([gain]))
    assert bool(out[0]) is expect


def test_gain_share_normalises_and_refuses_a_zero_total():
    s = D.gain_share(["a", "b", "c"], np.array([2.0, 6.0, 0.0]))
    assert s.to_dict() == {"a": 0.25, "b": 0.75, "c": 0.0}
    with pytest.raises(ValueError, match="no gain"):
        D.gain_share(["a"], np.array([0.0]))


def test_cells_needs_the_minimum_rows_on_both_sides():
    a = pd.DataFrame({"ap": ["X"] * 5 + ["Y"] * 2, "v": range(7)})
    b = pd.DataFrame({"ap": ["X"] * 4 + ["Y"] * 9, "v": range(13)})
    assert D.cells(a, b, min_rows=4) == ["X"]
    assert D.cells(a, b, min_rows=2) == ["X", "Y"]
    # the second side alone can exclude an airport
    a2 = pd.DataFrame({"ap": ["X"] * 5 + ["Z"] * 6, "v": range(11)})
    b2 = pd.DataFrame({"ap": ["X"] * 4 + ["Z"] * 1, "v": range(5)})
    assert D.cells(a2, b2, min_rows=4) == ["X"]


# ---------------------------------------------------------------- multivariate

def test_domain_auc_is_half_for_one_distribution_and_high_for_a_shift():
    a, b = _frame(3000, seed=5), _frame(3000, seed=6)
    same = D.domain_auc(a[["same", "moved"]].to_numpy(), b[["same", "moved"]].to_numpy(), seed=0)
    assert 0.45 <= same <= 0.55
    c = _frame(3000, shift=2.0, seed=7)
    assert D.domain_auc(a[["same", "moved"]].to_numpy(), c[["same", "moved"]].to_numpy(), seed=0) > 0.9
    assert D.domain_auc(a[["same", "moved"]].to_numpy(), b[["same", "moved"]].to_numpy(), seed=0) == same   # seeded
