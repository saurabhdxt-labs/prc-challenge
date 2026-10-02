"""scripts/analysis/epx_price.py — pricing the LFPG EOBT practice change on held-out 2025 rows
(plans/PREREG_epx_lfpg_eobt_pricing_2026_10_02.md): the label-free revision pattern, imposing it on 2025 rows, the
serve-time clip, and the day-block interval. Synthetic arrays only; the real files and the model are read by the CLI.

Seen red first 2026-10-02 (collection error, module absent). Mutation rehearsal 2026-10-02 (sources restored): 14 of 14
RED — left-closed bins, finite check off, share counting zeros, pool holding non-revisions, q = p26, |q| for max(0, q),
already-revised rows redrawn, the row's bin ignored, empty-pool check off, max for min in the clip, NaN handling dropped,
rows for days in the bootstrap, mean for sum, board scaling dropped."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("epx_price", ROOT / "scripts/analysis/epx_price.py")
X = importlib.util.module_from_spec(_spec)
sys.modules["epx_price"] = X
_spec.loader.exec_module(X)


# ---------------------------------------------------------------- bins

def test_bins_are_the_registered_seven_with_right_closed_edges():
    al = np.array([-50.0, 0.0, 0.1, 300.0, 300.1, 600.0, 900.0, 900.1, 1800.0, 3600.0, 3600.1, 1e6])
    assert X.bin_of(al).tolist() == [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 6, 6]
    assert X.N_BINS == 7
    with pytest.raises(ValueError, match="non-finite"):
        X.bin_of(np.array([1.0, np.nan]))


# ---------------------------------------------------------------- pattern

def test_pattern_is_the_revised_share_and_the_pool_of_revisions_per_bin():
    al = np.array([-10.0, 0.0, 100.0, 100.0, 100.0, 5000.0])
    el = np.array([0.0, 60.0, 0.0, 120.0, 240.0, -30.0])
    p = X.pattern(al, el)
    assert p[0]["n"] == 2 and p[0]["p"] == 0.5 and p[0]["pool"].tolist() == [60.0]
    assert p[1]["n"] == 3 and p[1]["p"] == pytest.approx(2 / 3) and sorted(p[1]["pool"].tolist()) == [120.0, 240.0]
    assert p[6]["n"] == 1 and p[6]["p"] == 0.0 and len(p[6]["pool"]) == 0
    assert p[3]["n"] == 0 and p[3]["p"] == 0.0             # an empty bin is a zero share, not a division error


# ---------------------------------------------------------------- impose

def _pats(p25, p26, pool=(600.0, 900.0)):
    a = {b: {"n": 100, "p": p25, "pool": np.array([])} for b in range(X.N_BINS)}
    c = {b: {"n": 100, "p": p26, "pool": np.array(pool)} for b in range(X.N_BINS)}
    return a, c


def test_impose_brings_the_revised_share_to_the_target_and_draws_from_the_bins_pool():
    """q = (p26 - p25) / (1 - p25) on the not-yet-revised rows lifts a 10 % share to 40 %. Mutation: q = p26 -> the share
    overshoots to 46 % -> red."""
    rng = np.random.default_rng(0)
    n = 40_000
    al = np.full(n, 450.0)
    el = np.where(rng.random(n) < 0.10, 300.0, 0.0)
    p25, p26 = _pats(0.10, 0.40)
    new, revised = X.impose(al, el, p25, p26, np.random.default_rng(1))
    assert (new > 0).mean() == pytest.approx(0.40, abs=0.01)
    assert set(np.unique(new[revised])) <= {600.0, 900.0} and revised.sum() > 0
    assert np.array_equal(new[~revised], el[~revised])                     # every other row untouched
    assert not revised[el > 0].any()                                       # an already-revised row is never redrawn


def test_impose_never_lowers_a_share_and_is_seeded():
    al = np.full(1000, 450.0)
    el = np.zeros(1000)
    p25, p26 = _pats(0.30, 0.20)                                           # 2026 below 2025 in this bin: nothing to impose
    new, revised = X.impose(al, el, p25, p26, np.random.default_rng(0))
    assert not revised.any() and np.array_equal(new, el)
    p25, p26 = _pats(0.0, 0.5)
    a1 = X.impose(al, el, p25, p26, np.random.default_rng(7))[0]
    a2 = X.impose(al, el, p25, p26, np.random.default_rng(7))[0]
    a3 = X.impose(al, el, p25, p26, np.random.default_rng(8))[0]
    assert np.array_equal(a1, a2) and not np.array_equal(a1, a3)


def test_impose_uses_each_rows_own_bin_and_refuses_an_empty_pool():
    al = np.r_[np.full(5000, 100.0), np.full(5000, 5000.0)]                # bins 1 and 6
    el = np.zeros(10_000)
    p25 = {b: {"n": 1, "p": 0.0, "pool": np.array([])} for b in range(X.N_BINS)}
    p26 = {b: {"n": 1, "p": 0.0, "pool": np.array([])} for b in range(X.N_BINS)}
    p26[1] = {"n": 1, "p": 1.0, "pool": np.array([111.0])}
    new, revised = X.impose(al, el, p25, p26, np.random.default_rng(0))
    assert revised[:5000].all() and not revised[5000:].any() and set(new[:5000]) == {111.0}
    p26[6] = {"n": 1, "p": 0.5, "pool": np.array([])}
    with pytest.raises(ValueError, match="bin 6: revisions to impose but an empty 2026 pool"):
        X.impose(al, el, p25, p26, np.random.default_rng(0))


# ---------------------------------------------------------------- clip

def test_clip_takes_the_earlier_of_eobt_and_lobt_and_keeps_missing_values():
    eobt = np.array([100.0, 500.0, np.nan, 300.0])
    lobt = np.array([200.0, 400.0, 400.0, np.nan])
    out = X.clip_eobt(eobt, lobt)
    assert out[:2].tolist() == [100.0, 400.0] and np.isnan(out[2]) and out[3] == 300.0


# ---------------------------------------------------------------- day-block interval

def test_day_block_ci_scales_by_the_board_size_and_brackets_the_point():
    rng = np.random.default_rng(0)
    day = np.repeat(np.arange(31), 200)
    delta = rng.normal(50.0, 400.0, len(day))                 # per-row SSE difference
    point, lo, hi = X.day_block_ci(delta, day, n_boot=500, seed=1)
    assert point == pytest.approx(delta.sum() / X.BOARD_N)
    assert lo < point < hi
    assert X.day_block_ci(delta, day, n_boot=500, seed=1) == (point, lo, hi)          # seeded
    assert X.day_block_ci(delta, day, n_boot=500, seed=2)[1] != lo


def test_day_block_ci_resamples_whole_days():
    """All of the difference sits on one day: resampling rows would give a tight interval, resampling days a wide one
    (that day is drawn 0, 1, 2 ... times). Mutation: bootstrap rows instead of days -> red."""
    day = np.repeat(np.arange(10), 100)
    delta = np.where(day == 3, 1000.0, 0.0)
    point, lo, hi = X.day_block_ci(delta, day, n_boot=2000, seed=0)
    assert lo == 0.0 and hi >= 2.5 * point


# ---------------------------------------------------------------- EPX.2: matching the marginal revised share

def test_calibrate_scale_finds_the_factor_that_hits_the_target_share_in_expectation():
    """EPX.2 (2026-10-02): the per-bin pattern imposed on a month with more delayed flights over-shoots the 2026 share
    (44 % for 35 %). One common factor s on every q_b brings the expected share to the target. Mutation: return 1.0
    always -> red."""
    rng = np.random.default_rng(0)
    al = np.r_[np.full(6000, 100.0), np.full(4000, 1000.0)]                # bins 1 and 4
    assert X.bin_of(np.array([100.0, 1000.0])).tolist() == [1, 4]
    el = np.zeros(10_000)
    p25 = {b: {"n": 1, "p": 0.0, "pool": np.array([])} for b in range(X.N_BINS)}
    p26 = {b: {"n": 1, "p": 0.0, "pool": np.array([600.0])} for b in range(X.N_BINS)}
    p26[1]["p"], p26[4]["p"] = 0.2, 0.8                                    # unscaled expectation: 0.12 + 0.32 = 0.44
    assert X.expected_share(al, el, p25, p26, 1.0) == pytest.approx(0.44)
    s = X.calibrate_scale(al, el, p25, p26, 0.35)
    assert s == pytest.approx(0.35 / 0.44, abs=1e-6)
    assert X.expected_share(al, el, p25, p26, s) == pytest.approx(0.35, abs=1e-9)
    new, revised = X.impose(al, el, p25, p26, rng, scale=s)
    assert (new > 0).mean() == pytest.approx(0.35, abs=0.012)
    # the bins keep their ordering: the high-delay bin is still revised four times as often
    assert revised[6000:].mean() / revised[:6000].mean() == pytest.approx(4.0, rel=0.15)


def test_calibrate_scale_handles_a_share_already_present_and_refuses_an_unreachable_target():
    al = np.full(1000, 100.0)
    el = np.r_[np.full(100, 300.0), np.zeros(900)]                         # 10 % already revised
    p25 = {b: {"n": 1, "p": 0.1, "pool": np.array([])} for b in range(X.N_BINS)}
    p26 = {b: {"n": 1, "p": 0.5, "pool": np.array([600.0])} for b in range(X.N_BINS)}
    assert X.expected_share(al, el, p25, p26, 1.0) == pytest.approx(0.5)   # 0.1 + 0.9 * (0.4 / 0.9)
    assert X.calibrate_scale(al, el, p25, p26, 0.3) == pytest.approx(0.5, abs=1e-6)
    assert X.calibrate_scale(al, el, p25, p26, 0.5) == pytest.approx(1.0, abs=1e-6)
    with pytest.raises(ValueError, match="cannot reach"):
        X.calibrate_scale(al, el, p25, p26, 1.5)
    with pytest.raises(ValueError, match="below the share already revised"):
        X.calibrate_scale(al, el, p25, p26, 0.05)


def test_impose_scale_one_is_the_registered_behaviour():
    al = np.full(2000, 450.0)
    el = np.zeros(2000)
    p25, p26 = _pats(0.0, 0.5)
    a = X.impose(al, el, p25, p26, np.random.default_rng(3))[0]
    b = X.impose(al, el, p25, p26, np.random.default_rng(3), scale=1.0)[0]
    assert np.array_equal(a, b)
