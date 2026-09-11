"""Tests for scripts/nm_param_diag.py (the NMD diagnostic). Synthetic, seeded, fast."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "nm_param_diag", Path(__file__).resolve().parents[1] / "scripts" / "nm_param_diag.py")
D = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(D)


# ---------------------------------------------------------------- pair_modes / attach_block

def test_pair_modes_picks_the_mode_breaks_ties_low_and_reports_share():
    pair = np.array(["A"] * 5 + ["B"] * 4)
    pm = np.array([10, 10, 10, 11, 12, 7, 7, 5, 5])
    t = D.pair_modes(pair, pm)
    assert t.loc["A", "m_star"] == 10 and t.loc["A", "n_pair"] == 5 and t.loc["A", "s_pair"] == pytest.approx(0.6)
    assert t.loc["B", "m_star"] == 5  # tie 7/5 -> smallest
    assert t.loc["B", "s_pair"] == pytest.approx(0.5)


@pytest.mark.parametrize("pair, pm, msg", [
    (np.array(["A"]), np.array([1, 2]), "differ in length"),
    (np.array([], dtype=object), np.array([]), "no rows"),
    (np.array(["A"]), np.array([np.nan]), "NaN"),
])
def test_pair_modes_refuses_bad_input(pair, pm, msg):
    with pytest.raises(ValueError, match=msg):
        D.pair_modes(pair, pm)


def test_attach_block_marks_unseen_pairs_and_mode_rows():
    t = D.pair_modes(np.array(["A", "A", "A"]), np.array([10, 10, 12]))
    b = D.attach_block(np.array(["A", "A", "Z"]), np.array([10, 13, 10]), t)
    assert b.at_mode.tolist() == [True, False, False]
    assert b.seen.tolist() == [True, True, False]
    assert b.dev.iloc[1] == 3 and np.isnan(b.dev.iloc[2])


# ---------------------------------------------------------------- classes

def test_dev_class_covers_every_boundary_value_exactly_once():
    dev = np.array([-7, -6, -5, -2, -1, 0, 1, 2, 5, 6, 15, 16, np.nan], dtype=float)
    seen = ~np.isnan(dev)
    got = D.dev_class(dev, seen).tolist()
    assert got == ["<=-6", "<=-6", "-5..-2", "-5..-2", "-1", "0", "+1", "+2..+5", "+2..+5", "+6..+15",
                   "+6..+15", ">+15", "unseen"]


def test_dev3_three_valued():
    assert D.dev3(np.array([True, False, False]), np.array([True, True, False])).tolist() == ["at", "off", "unseen"]


# ---------------------------------------------------------------- cross-fitting

def _toy(n=4000, seed=0, bias=0.0):
    rng = np.random.default_rng(seed)
    month = np.where(np.arange(n) < n // 2, 1, 7)
    ap = rng.choice(["X", "Y"], n)
    at = rng.random(n) < 0.6
    r = rng.normal(0, 100, n) + np.where(at & (ap == "X"), bias, 0.0)
    return r, ap, at, month


def test_crossfit_k1_never_uses_the_rows_own_month():
    """Fails if crossfit_k1 fits on the applied month (e.g. swap fit_m/app_m)."""
    r, ap, at, month = _toy()
    cls = np.where(at, "0", "+1").astype(object)
    c = D.crossfit_k1(r, ap, cls, month)
    r2 = r.copy()
    r2[month == 7] += 1e6  # July's residuals change -> July's own corrections must not
    c2 = D.crossfit_k1(r2, ap, cls, month)
    assert np.array_equal(c[month == 7], c2[month == 7])
    assert not np.array_equal(c[month == 1], c2[month == 1])


def test_crossfit_k2_never_uses_the_rows_own_month_and_shrinks():
    r, ap, at, month = _toy()
    pair = np.char.add(ap.astype(str), np.where(np.arange(len(r)) % 3 == 0, "|1", "|2"))
    c3 = D.dev3(at, np.ones(len(r), bool))
    c = D.crossfit_k2(r, pair, ap, c3, month)
    r2 = r.copy()
    r2[month == 1] -= 1e6
    c2 = D.crossfit_k2(r2, pair, ap, c3, month)
    assert np.array_equal(c[month == 1], c2[month == 1])
    # shrinkage: a single-row cell sits between its prior and its own mean with weight 1/(1+k)
    r1 = np.array([0.0, 0.0, 100.0, 7.0])
    m1 = np.array([1, 1, 1, 7])
    ap1 = np.array(["X"] * 4)
    p1 = np.array(["a", "a", "b", "b"])
    cl = np.array(["at"] * 4, dtype=object)
    got = D.crossfit_k2(r1, p1, ap1, cl, m1, k=50)
    prior = 100.0 / 3
    assert got[3] == pytest.approx((100.0 + 50 * prior) / (1 + 50))  # pair b, fitted on month 1's single row


def test_crossfit_refuses_other_than_two_months():
    with pytest.raises(ValueError, match="exactly two months"):
        D.crossfit_k1(np.zeros(3), np.array(["X"] * 3), np.array(["0"] * 3, dtype=object), np.array([1, 1, 1]))
    with pytest.raises(ValueError, match="k must be positive"):
        D.crossfit_k2(np.zeros(2), np.array(["a", "b"]), np.array(["X"] * 2), np.array(["at"] * 2, dtype=object),
                      np.array([1, 7]), k=0)


def test_planted_bias_is_recovered_and_null_is_not():
    """Planted: at-mode rows at airport X carry +150 s and off-mode rows -100 s (zero airport-level mean, 0.6/0.4 split)
    in both months -> gain > 0 and the permuted control well below it. Null: no bias -> no material gain.
    Fails if crossfit_k1 ignores the class key (then K1 == its own negative control)."""
    r, ap, at, month = _toy(n=20000, seed=3, bias=150.0)
    r = r - np.where(~at & (ap == "X"), 100.0, 0.0)
    yhat = np.full(len(r), 900.0)
    y = yhat + r
    cls = np.where(at, "0", "+1").astype(object)
    g = D.gain_rows(y, yhat, D.crossfit_k1(r, ap, cls, month))
    grp = pd.Series(ap.astype(str)) + "#" + pd.Series(month.astype(str))
    g_nc = D.gain_rows(y, yhat, D.crossfit_k1(r, ap, D.permute_within(cls, grp, 1), month))
    assert g.sum() > 0.5 * (150.0 ** 2) * (at & (ap == "X")).sum()
    assert g_nc.sum() < 0.25 * g.sum()
    r0, ap0, at0, month0 = _toy(n=20000, seed=4, bias=0.0)
    g0 = D.gain_rows(900.0 + r0, np.full(len(r0), 900.0),
                     D.crossfit_k1(r0, ap0, np.where(at0, "0", "+1").astype(object), month0))
    assert g0.sum() < 0.01 * (r0 ** 2).sum()


def test_negative_control_recovers_an_airport_level_mean_by_design():
    """Amendment NMD.1: permuting the class within airport-month keeps each airport's mean residual, so the control
    prices the airport-mean part of K1 — it is NOT zero by construction when an airport mean bias exists."""
    r, ap, at, month = _toy(n=20000, seed=5, bias=0.0)
    r = r + np.where(ap == "X", 80.0, 0.0)  # airport-level bias only, same in every class
    yhat = np.full(len(r), 900.0)
    cls = np.where(at, "0", "+1").astype(object)
    grp = pd.Series(ap.astype(str)) + "#" + pd.Series(month.astype(str))
    g = D.gain_rows(900.0 + r, yhat, D.crossfit_k1(r, ap, cls, month)).sum()
    g_nc = D.gain_rows(900.0 + r, yhat, D.crossfit_k1(r, ap, D.permute_within(cls, grp, 2), month)).sum()
    assert g_nc > 0.5 * (80.0 ** 2) * (ap == "X").sum()
    assert g_nc == pytest.approx(g, rel=0.15)  # both are the airport mean when the class carries nothing


def test_gain_rows_applies_the_floor():
    g = D.gain_rows(np.array([5.0]), np.array([10.0]), np.array([-100.0]))
    assert g[0] == pytest.approx(25.0 - 16.0)  # corrected arm floored at 1, not -90


def test_to_board_scale():
    assert D.to_board(339015.0, 339015) == pytest.approx(0.9846596)


# ---------------------------------------------------------------- interval / slope / permutation

def test_dayblock_sum_resamples_dates_within_month():
    rng = np.random.default_rng(0)
    day = np.repeat([f"2025-01-{d:02d}" for d in range(1, 11)] + [f"2025-07-{d:02d}" for d in range(1, 11)], 50)
    month = np.where(np.char.startswith(day.astype(str), "2025-01"), 1, 7)
    g = rng.normal(1.0, 1.0, len(day))
    tot, lo, hi, nd = D.dayblock_sum(g, day, month, n_boot=500)
    assert nd == 20 and lo < tot < hi
    assert tot == pytest.approx(g.sum())
    g_const = np.ones(len(day))  # every date sums to 50 -> every draw equals the total exactly
    t2, lo2, hi2, _ = D.dayblock_sum(g_const, day, month, n_boot=50)
    assert lo2 == pytest.approx(t2) and hi2 == pytest.approx(t2)


def test_dayblock_keeps_each_months_date_count_fixed():
    """January dates each sum to 50, July dates to 0: resampling WITHIN month gives every draw = 500 exactly;
    pooled resampling would vary the January count. Fails if the months are pooled."""
    day = np.repeat([f"2025-01-{d:02d}" for d in range(1, 11)] + [f"2025-07-{d:02d}" for d in range(1, 11)], 50)
    month = np.where(np.char.startswith(day.astype(str), "2025-01"), 1, 7)
    g = np.where(month == 1, 1.0, 0.0)
    tot, lo, hi, _ = D.dayblock_sum(g, day, month, n_boot=200)
    assert tot == 500.0 and lo == pytest.approx(500.0) and hi == pytest.approx(500.0)


def test_dayblock_refuses_a_date_spanning_two_months():
    with pytest.raises(ValueError, match="spans two months"):
        D.dayblock_sum(np.ones(2), np.array(["d", "d"]), np.array([1, 7]))


def test_within_pair_slope_removes_pair_level_differences():
    """Pair offsets correlate with the pairs' dev levels, so a slope without within-pair demeaning is far from 30.
    Fails if within_pair_slope skips the demeaning."""
    base = np.arange(-2, 3).astype(float)
    dev = np.concatenate([base, base + 5, base - 4, base + 9])
    pair = np.repeat(["a", "b", "c", "d"], 5)
    v = 30.0 * np.tile(base, 4) + np.repeat([0.0, 500.0, -300.0, 1000.0], 5)
    assert D.within_pair_slope(v, dev, pair) == pytest.approx(30.0)
    assert np.isnan(D.within_pair_slope(v, np.repeat([1.0, 2.0, 3.0, 4.0], 5), pair))


def test_between_ss_is_the_se_removed_by_cell_means_beyond_the_base_mean():
    r = np.array([1.0, 3.0, 10.0, 14.0, 0.0, 0.0])
    cell = np.array(["a", "a", "b", "b", "c", "c"])
    base = np.array(["X", "X", "X", "X", "Y", "Y"])
    # X mean 7: cell a mean 2 -> 2*(5^2)=50, cell b mean 12 -> 2*(5^2)=50; Y contributes 0
    assert D.between_ss(r, cell, base) == pytest.approx(100.0)
    se_base = ((r - np.array([7, 7, 7, 7, 0, 0])) ** 2).sum()
    se_cell = ((r - np.array([2, 2, 12, 12, 0, 0])) ** 2).sum()
    assert D.between_ss(r, cell, base) == pytest.approx(se_base - se_cell)
    with pytest.raises(ValueError, match="spans two base groups"):
        D.between_ss(r, np.array(["a"] * 6), base)


def test_permute_within_keeps_group_multisets():
    x = np.arange(12)
    grp = pd.Series(["a"] * 6 + ["b"] * 6)
    p = D.permute_within(x, grp, 0)
    assert sorted(p[:6]) == list(range(6)) and sorted(p[6:]) == list(range(6, 12))
    assert not np.array_equal(p, x)
