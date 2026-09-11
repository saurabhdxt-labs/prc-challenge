"""prc.pipeline.scoring — paired pricing, day-block intervals, convention-stamped fold records. Synthetic and seeded."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from prc.pipeline import errors as E  # noqa: E402
from prc.pipeline import scoring as SC  # noqa: E402


def _days(n_per=50, n_dates=10):
    day = np.repeat([f"2025-01-{d:02d}" for d in range(1, n_dates + 1)] + [f"2025-07-{d:02d}" for d in range(1, n_dates + 1)],
                    n_per)
    month = np.where(np.char.startswith(day.astype(str), "2025-01"), 1, 7)
    return day, month


def test_weighted_fold_mse_scales_and_refuses_nonsense():
    assert SC.weighted_fold_mse(339_015.0, 339_015) == pytest.approx(SC.W_MATCHED)
    assert SC.weighted_fold_mse(100.0, 10, w=0.5) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="n_fold"):
        SC.weighted_fold_mse(1.0, 0)
    with pytest.raises(ValueError, match="lane weight"):
        SC.weighted_fold_mse(1.0, 10, w=1.5)


def test_dayblock_interval_resamples_within_month_and_equals_the_nmd_harness():
    """Same algorithm as scripts/nm_param_diag.dayblock_sum (the recorded diagnostics) — equal on the same draw — and
    within-month: January dates summing to 50, July to 0 give every draw exactly 500. Fails if months are pooled."""
    import nm_param_diag as N
    day, month = _days()
    rng = np.random.default_rng(0)
    g = rng.normal(1.0, 3.0, len(day))
    assert SC.dayblock_interval(g, day, month, n_boot=300) == N.dayblock_sum(g, day, month, n_boot=300)
    tot, lo, hi, nd = SC.dayblock_interval(np.where(month == 1, 1.0, 0.0), day, month, n_boot=200)
    assert (tot, nd) == (500.0, 20) and lo == pytest.approx(500.0) and hi == pytest.approx(500.0)


@pytest.mark.parametrize("bad,msg", [("len", "differ in length"), ("nan", "non-finite"), ("span", "spans two months")])
def test_dayblock_interval_refusals(bad, msg):
    day, month = _days(n_per=2, n_dates=2)
    g = np.ones(len(day))
    if bad == "len":
        g = g[:-1]
    elif bad == "nan":
        g[0] = np.nan
    else:
        month = month.copy()
        month[0] = 7
    with pytest.raises(ValueError, match=msg):
        SC.dayblock_interval(g, day, month, n_boot=10)


def test_paired_prices_the_treatment_against_the_control_on_identical_rows_with_cuts():
    day, month = _days(n_per=10, n_dates=5)
    y = np.full(len(day), 100.0)
    control = np.full(len(day), 110.0)
    treatment = np.where(np.arange(len(day)) % 2 == 0, 100.0, 110.0)   # half the rows fixed: Σg = 50 rows x 100
    cut = np.arange(len(day)) % 2 == 0
    out = SC.paired(y, control, treatment, day, month, n_fold=100, w=1.0, cuts={"fixed": cut, "rest": ~cut}, n_boot=100)
    assert out["net"] == pytest.approx(50 * 100.0 / 100)
    assert out["cuts"]["fixed"] == {"n": 50, "net": pytest.approx(50.0)} and out["cuts"]["rest"]["net"] == 0.0
    assert out["rmse_control"] == pytest.approx(10.0) and out["rmse_treatment"] == pytest.approx(np.sqrt(50.0))
    with pytest.raises(ValueError, match="not the same rows"):
        SC.paired(y, control[:-1], treatment, day, month)
    with pytest.raises(ValueError, match="cut 'bad'"):
        SC.paired(y, control, treatment, day, month, cuts={"bad": cut[:-1]}, n_boot=10)


def test_fold_records_round_trip_with_their_stamp_and_refuse_unstamped_or_foreign(tmp_path):
    f = pd.DataFrame({"row": [1, 2], "y": [900.0, 800.0], "arm": [880.0, 810.0]})
    p = tmp_path / "rec.parquet"
    SC.write_record(f, p, "taxi_time", "REG", "plans/PREREG_x.md", {"seed": 0})
    back, meta = SC.read_record(p, "taxi_time", tag="REG")
    pd.testing.assert_frame_equal(back, f)
    assert meta["convention"] == "taxi_time" and meta["prereg"] == "plans/PREREG_x.md" and meta["meta"] == {"seed": 0}
    with pytest.raises(E.SchemaError, match="'taxi_time' scale, the caller expects 'delta'"):
        SC.read_record(p, "delta")
    with pytest.raises(E.SchemaError, match="tagged 'REG'"):
        SC.read_record(p, "taxi_time", tag="CAP")
    plain = tmp_path / "plain.parquet"
    f.to_parquet(plain, index=False)
    with pytest.raises(E.SchemaError, match="no convention stamp"):
        SC.read_record(plain, "taxi_time")
    with pytest.raises(ValueError, match="convention must be"):
        SC.write_record(f, tmp_path / "x.parquet", "seconds", "REG", "p")
    with pytest.raises(ValueError, match="tag and the prereg"):
        SC.write_record(f, tmp_path / "x.parquet", "delta", "", "p")
