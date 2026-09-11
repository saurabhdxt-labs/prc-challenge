"""Tests for scripts/unm_diag.py (the UMD diagnostic). Synthetic, exact arithmetic."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import unm_diag as U  # noqa: E402


def test_row_class_boundaries_and_order():
    """fill wins over everything; then date-slip, Family B, other monsters, ordinary. Fails on any reordered rule or
    a moved boundary (60 s, 80,000 s, 10,800 s)."""
    y = np.array([5_000, 5_060, 5_061, 87_000, 12_000, 12_000, 10_800, 800, 80_000, 10_801])
    sp = np.array([5_000, 5_000, 5_000, 30_000, 10_799, 10_800, 1_000, 1_000, 30_000, 10_799])
    got = U.row_class(y, sp).tolist()
    assert got == ["fill", "fill", "ordinary", "dateslip", "familyB", "monster_other", "ordinary", "ordinary",
                   "monster_other", "fill"]   # y = 80,000 is not > 80,000; y 10,801 / sp 10,799 is a fill (fill first)
    assert U.row_class(np.array([87_000.0]), np.array([87_000.0])).tolist() == ["fill"]
    with pytest.raises(ValueError, match="every row"):
        U.row_class(np.array([np.nan]), np.array([1.0]))


def test_sp_band_is_left_closed_on_every_edge():
    sp = np.array([-1, 0, 1_799, 1_800, 3_600, 10_799, 10_800, 24_000, 86_399, 86_400])
    assert U.sp_band(sp).tolist() == ["<0", "0-1800", "0-1800", "1800-3600", "3600-10800", "3600-10800", "10800-24000",
                                      "24000-86400", "24000-86400", ">=86400"]


def test_e1_floor_touches_only_lirf_rows_inside_the_band():
    pred = np.array([1_000.0, 1_000.0, 1_000.0, 1_000.0, 90_000.0])
    sp = np.array([30_000.4, 30_000.0, 23_999.0, 86_400.0, 30_000.0])
    ap = np.array(["LIRF", "EGLL", "LIRF", "LIRF", "LIRF"])
    assert U.e1_floor(pred, sp, ap).tolist() == [30_000.0, 1_000.0, 1_000.0, 1_000.0, 90_000.0]


def test_learnable_is_ordinary_and_small_sp_fills_only():
    cls = np.array(["ordinary", "fill", "fill", "familyB", "dateslip", "monster_other"], dtype=object)
    sp = np.array([900, 10_799, 10_800, 500, 30_000, 20_000])
    assert U.learnable(cls, sp).tolist() == [True, True, False, False, False, False]


def test_kish_and_top_share():
    assert U.kish(np.ones(10)) == pytest.approx(10.0)
    assert U.kish(np.array([100.0, 0, 0, 0])) == pytest.approx(1.0)
    assert U.top_share(np.array([1.0, 2.0, 7.0]), 1) == pytest.approx(0.7)
    assert np.isnan(U.top_share(np.zeros(3), 1))


def test_expectation_2026_scales_per_stratum_means_by_2026_counts_and_borrows_the_airport_mean():
    """2025: stratum (A, b1) mean SE 100 (fill 60 + ordinary 40 per row); airport A overall mean (100·2 + 400·2)/4 = 250.
    2026: 3 rows in (A, b1) -> 300 / N; 1 row in (A, b2) with no 2025 row -> borrows 250 / N."""
    train = pd.DataFrame({"ap": ["A", "A", "A", "A"], "band": ["b1", "b1", "b3", "b3"],
                          "cls": ["fill", "ordinary", "ordinary", "ordinary"], "se": [120.0, 80.0, 400.0, 400.0]})
    serve = pd.DataFrame({"ap": ["A", "A", "A", "A"], "band": ["b1", "b1", "b1", "b2"]})
    e = U.expectation_2026(train, serve, n_scored=10)
    assert e["total"] == pytest.approx((3 * 100.0 + 250.0) / 10)
    assert e["by_class"]["fill"] == pytest.approx((3 * 60.0 + 30.0) / 10)          # borrowed row: 120/4 of fill
    assert e["by_class"]["ordinary"] == pytest.approx((3 * 40.0 + 220.0) / 10)
    assert e["borrowed_strata"] == [{"ap": "A", "band": "b2", "n_2026": 1}] and e["n_2026_rows"] == 4
    with pytest.raises(ValueError, match="without any 2025 row"):
        U.expectation_2026(train, pd.DataFrame({"ap": ["Z"], "band": ["b1"]}))
