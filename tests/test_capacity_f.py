"""scripts/capacity_f.py - arm CAP (plans/PREREG_capacity_f_2026_09_10.md): arm F's design at setting 127,20,0.6,
scored against arm F's stored record with five locked clauses.

Pins: the setting touches exactly three params; the join refuses a different row set / labels / missing columns;
the DELTA recovery floors at 1 s; each clause and the verdict mapping on hand-built frames; the date-block
bootstrap is deterministic.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capacity_f as cf  # noqa: E402


def dates_for(f):
    """60 synthetic calendar dates (30 per holdout month) so the day-block C1 has real blocks."""
    return np.array([f"2025-{m:02d}-{1 + i % 30:02d}" for i, m in enumerate(f.month.to_numpy())])

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def frames(n=4_000, improve=0.9, seed=0, cap_noise=0.0):
    """A synthetic F record and CAP record on the same rows. CAP's error is F's scaled by `improve`."""
    rng = np.random.default_rng(seed)
    y = rng.gamma(4.0, 250.0, n)
    proxy = y + rng.normal(300.0, 400.0, n)          # |delta| > 600 s on ~23% of rows: the tail cut exists
    true_delta = proxy - y
    err = rng.normal(0.0, 200.0, n)
    f_delta = true_delta + err
    f = pd.DataFrame({"row": np.arange(n) * 3, "month": np.where(np.arange(n) % 2 == 0, 1, 7),
                      "ap": np.array(AIRPORTS)[np.arange(n) % 10], "y": y, "proxy": proxy,
                      "sp": y + rng.normal(0, 400, n), "delta": true_delta, "treatment": f_delta})
    c = pd.DataFrame({"row": f.row, "y": y})
    c["cap"] = true_delta + improve * err + cap_noise * rng.normal(0, 1, n)
    for s in cf.SEEDS:
        f[f"delta_hat_seed{s}"] = f_delta + rng.normal(0, 5, n)
        c[f"cap_seed{s}"] = c["cap"] + rng.normal(0, 5, n)
    return f, c


def test_setting_replaces_exactly_three_params():
    """Rehearsed 2026-09-10 RED: SETTING (127,20,0.6) -> (127,20,0.8) (the feature_fraction assertion fires)."""
    import lgbm_fold as lf
    import lgbm_submit as L
    p = lf.setting_params(dict(L.P), cf.SETTING)
    changed = {k for k in p if p[k] != L.P.get(k)}
    assert changed == {"num_leaves", "min_data_in_leaf", "feature_fraction"}
    assert (p["num_leaves"], p["min_data_in_leaf"], p["feature_fraction"]) == (127, 20, 0.6)
    assert p["learning_rate"] == L.P["learning_rate"] == 0.01


def test_taxi_floors_at_one_second():
    """Rehearsed 2026-09-10 RED: the floor 1.0 -> 0.0 in cf.taxi."""
    np.testing.assert_array_equal(cf.taxi([100.0, 50.0], [200.0, 20.0]), [1.0, 30.0])


def test_align_refuses_different_rows_labels_or_missing_columns():
    """Rehearsed 2026-09-10 RED: the label comparison deleted from align (the relabelled frame passes)."""
    f, c = frames(n=200)
    m = cf.align(f, c)
    assert len(m) == 200
    with pytest.raises(AssertionError, match="row sets differ"):
        cf.align(f, c.iloc[1:])
    bad = c.copy()
    bad.loc[5, "y"] += 1.0
    with pytest.raises(AssertionError, match="labels differ"):
        cf.align(f, bad)
    with pytest.raises(ValueError, match="lacks columns"):
        cf.align(f, c.drop(columns=["cap_seed2"]))


def test_a_clear_improvement_is_established():
    """Rehearsed 2026-09-10 RED: the gain sign flipped in clauses (gain = CAP - F) -> C2 fails -> INCONCLUSIVE."""
    f, c = frames(improve=0.85)
    res = cf.clauses(cf.align(f, c), dates_for(f), n_boot=300)
    assert res["verdict"] == "ESTABLISHED", res["clauses"]
    assert res["gain_s"] > 0 and res["ci95_day_block"][0] > 0 and res["net_weighted_fold_mse"] > 0
    want = cf.W_MATCHED * (((cf.taxi(f.proxy, f.treatment) - f.y) ** 2).sum()
                           - ((cf.taxi(f.proxy, c.cap) - f.y) ** 2).sum()) / cf.N_FOLD
    assert res["net_weighted_fold_mse"] == pytest.approx(want, rel=1e-12)  # same sums, fp64


def test_a_worse_arm_is_not_working():
    """Rehearsed 2026-09-10 RED: the verdict mapping's NOT WORKING branch relabelled INCONCLUSIVE. (A C1 check on
    ci95[1] instead of ci95[0] would SURVIVE this test -- both bounds are negative for a worse arm -- so the
    interval bound itself is pinned by test_an_interval_straddling_zero_fails_c1.)"""
    f, c = frames(improve=1.15)
    res = cf.clauses(cf.align(f, c), dates_for(f), n_boot=300)
    assert res["verdict"] == "NOT WORKING" and not res["clauses"]["C1_interval"]


def test_an_interval_straddling_zero_fails_c1():
    """A null arm: CAP's error is an INDEPENDENT draw of the same size as F's (improve=0, cap_noise=200 = F's
    error sd), so the interval straddles zero; C1 fails and the verdict is NOT WORKING. Rehearsed 2026-09-10 RED:
    C1 read from the day-block interval's upper bound instead of its lower (the straddling interval passes)."""
    f, c = frames(improve=0.0, cap_noise=200.0, n=6_000, seed=3)
    res = cf.clauses(cf.align(f, c), dates_for(f), n_boot=500)
    lo, hi = res["ci95_day_block"]
    assert lo < 0.0 < hi, res["ci95_day_block"]
    assert not res["clauses"]["C1_interval"] and res["verdict"] == "NOT WORKING"


def test_a_fill_cut_loss_over_one_second_blocks_establishment():
    """Rehearsed 2026-09-10 RED: C5 tolerance 1.0 -> 100.0 (the damaged fill cut passes)."""
    f, c = frames(improve=0.85, n=6_000)
    fill = np.abs(f.y - f.sp) <= cf.FILL_TOL_S
    c.loc[fill.to_numpy(), "cap"] = c.loc[fill.to_numpy(), "cap"] - 150.0      # CAP worse on fill rows, better pooled
    res = cf.clauses(cf.align(f, c), dates_for(f), n_boot=300)
    assert res["cuts"]["fill"]["loss_s"] > cf.CUT_TOL_S
    assert not res["clauses"]["C5_cuts_bounded"] and res["verdict"] == "INCONCLUSIVE"


def test_airport_clause_needs_seven_of_ten():
    """Rehearsed 2026-09-10 RED: MIN_AIRPORTS 7 -> 4 (four damaged airports no longer block C4)."""
    f, c = frames(improve=0.85, n=6_000)
    bad = f.ap.isin(["EDDF", "EDDM", "EGLL", "EHAM"]).to_numpy()
    c.loc[bad, "cap"] = f.loc[bad, "delta"] + 1.4 * (f.loc[bad, "treatment"] - f.loc[bad, "delta"])
    res = cf.clauses(cf.align(f, c), dates_for(f), n_boot=300)
    assert res["airports_improving"] == 6 and not res["clauses"]["C4_airports"]


def test_an_empty_decisional_cut_is_refused():
    """Rehearsed 2026-09-10 RED: the empty-cut refusal deleted (C5 silently reads NaN as a fail)."""
    f, c = frames(n=500)
    f["delta"] = 100.0                                                   # no |delta| > 600 row
    with pytest.raises(ValueError, match="tail cut is empty"):
        cf.clauses(cf.align(f, c), dates_for(f), n_boot=50)


def test_date_block_is_deterministic_and_brackets_the_gain():
    """Rehearsed 2026-09-10 RED: the rng seeded from None (two calls differ)."""
    f, c = frames(improve=0.85)
    r1 = cf.clauses(cf.align(f, c), dates_for(f), n_boot=300)
    r2 = cf.clauses(cf.align(f, c), dates_for(f), n_boot=300)
    assert r1["ci95_day_block"] == r2["ci95_day_block"]
    lo, hi = r1["ci95_day_block"]
    assert lo < r1["gain_s"] < hi


def test_c1_is_the_day_block_interval_not_the_row_bootstrap():
    """Amendment CAP.1. Day-clustered gains: CAP helps a lot on some days and hurts on others (a shared per-day
    factor), so the ROW bootstrap is narrow and excludes zero while the DAY-BLOCK interval straddles it; C1 must
    follow the day block. Rehearsed 2026-09-10 RED: C1 read from the row bootstrap's lower bound (C1 passes)."""
    rng = np.random.default_rng(11)
    f, c = frames(improve=1.0, n=12_000, seed=11)
    d = dates_for(f)
    days = np.unique(d)
    factor = dict(zip(days, rng.choice([0.4, 1.3], size=len(days), p=[0.5, 0.5])))
    err = f.treatment.to_numpy() - f.delta.to_numpy()
    c["cap"] = f.delta.to_numpy() + np.array([factor[k] for k in d]) * err
    res = cf.clauses(cf.align(f, c), d, n_boot=600)
    assert res["ci95_row_bootstrap_reported"][0] > 0.0, res["ci95_row_bootstrap_reported"]
    lo, hi = res["ci95_day_block"]
    assert lo < 0.0 < hi, res["ci95_day_block"]
    assert not res["clauses"]["C1_interval"] and res["verdict"] == "NOT WORKING"


def test_c1_refuses_to_run_without_dates():
    """Rehearsed 2026-09-10 RED: the missing-dates refusal deleted (C1 would fall over or silently change)."""
    f, c = frames(n=300)
    with pytest.raises(ValueError, match="one take-off date per row"):
        cf.clauses(cf.align(f, c), None, n_boot=50)
