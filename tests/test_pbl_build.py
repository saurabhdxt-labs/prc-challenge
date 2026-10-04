"""PBL — registered invariants of plans/PREREG_pbl_production_anchored_blend_2026_09_20.md (0ae2edf6).

PBL's characteristic failure is the one it exists to expose: scoring a candidate against the WRONG
baseline. Two guards carry that — `board_gain` must be antisymmetric in (base, cand) so a sign slip
cannot pass unnoticed, and `decide` must be able to return NOT WORKING on a large positive mean when
a month disagrees or the tail carries the gain.

The second failure mode is the delta/taxi confusion that produced 1.4 M nonsense rows once already
(handoff 5.3): `taxi_from_delta` is `max(proxy - delta, 1)`, never `proxy + delta` and never unclamped.

The third is a silent join shortfall: `join_predictions` must REPORT missing rows, never drop them
quietly, because a shrinking population flatters a gain that divides by a fixed 344,841.

Mutation rehearsal runs under the BC-16-safe harness (bytecode purged between mutants).
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def M():
    return _load("pbl_build", "scripts/analysis/pbl_build.py")


# --- constants are the registration, not the implementer's taste -------------------------------

def test_constants_match_the_registration(M):
    assert M.BOARD_ROWS == 344841              # the fixed normaliser; never the population size
    assert M.WEIGHT == 0.5                     # fixed in the registration, never searched
    assert M.SEED_SPREAD_BAR == 49.0           # measured seed spread, handoff section 4
    assert M.TRIM_FRAC == 0.01                 # top-1 % tail rule
    assert M.INC_RMSE == 208.31040447352203    # opc_screen.py:129
    assert M.N_ROWS == 334132 and M.N_POP == 161956   # opc_screen.py:127


# --- taxi_from_delta: the semantics that produced 1.4 M nonsense rows once ----------------------

def test_taxi_from_delta_subtracts_and_clamps(M):
    proxy = np.array([100.0, 100.0, 5.0])
    delta = np.array([30.0, -30.0, 99.0])
    got = M.taxi_from_delta(proxy, delta)
    assert np.allclose(got, [70.0, 130.0, 1.0])


def test_taxi_from_delta_never_returns_below_one(M):
    proxy = np.array([10.0, 0.0, -5.0])
    delta = np.array([1e6, 1e6, 1e6])
    assert (M.taxi_from_delta(proxy, delta) == 1.0).all()


def test_taxi_from_delta_is_not_addition(M):
    """A sign slip here is invisible in aggregate RMSE but inverts every row."""
    proxy = np.array([100.0])
    delta = np.array([30.0])
    got = M.taxi_from_delta(proxy, delta)[0]
    assert got == 70.0 and got != 130.0   # pinning the value, not merely "not the wrong one"


# --- blend --------------------------------------------------------------------------------------

def test_blend_is_the_registered_convex_combination(M):
    inc = np.array([100.0, 200.0])
    cat = np.array([200.0, 100.0])
    assert np.allclose(M.blend(inc, cat, 0.5), [150.0, 150.0])


def test_blend_clamps_at_one(M):
    assert M.blend(np.array([0.0]), np.array([0.0]), 0.5)[0] == 1.0


def test_blend_weight_one_returns_the_candidate_member(M):
    inc = np.array([100.0, 200.0])
    cat = np.array([7.0, 9.0])
    assert np.allclose(M.blend(inc, cat, 1.0), cat)


def test_blend_weight_zero_returns_the_base(M):
    inc = np.array([100.0, 200.0])
    cat = np.array([7.0, 9.0])
    assert np.allclose(M.blend(inc, cat, 0.0), inc)


# --- board_gain: the sign discipline this whole arm turns on -----------------------------------

def test_board_gain_is_positive_when_the_candidate_is_closer(M):
    y = np.array([100.0, 100.0])
    base = np.array([110.0, 110.0])
    cand = np.array([105.0, 105.0])
    assert M.board_gain(y, base, cand) > 0


def test_board_gain_is_negative_when_the_candidate_is_worse(M):
    y = np.array([100.0, 100.0])
    base = np.array([105.0, 105.0])
    cand = np.array([110.0, 110.0])
    assert M.board_gain(y, base, cand) < 0


def test_board_gain_is_antisymmetric(M):
    y = np.array([100.0, 130.0, 90.0])
    a = np.array([110.0, 120.0, 95.0])
    b = np.array([105.0, 140.0, 80.0])
    assert M.board_gain(y, a, b) == pytest.approx(-M.board_gain(y, b, a))


def test_board_gain_divides_by_the_fixed_board_row_count_not_the_population(M):
    """Dividing by len(y) would make a shrinking population look like a bigger gain."""
    y = np.array([0.0, 0.0])
    base = np.array([10.0, 10.0])
    cand = np.array([0.0, 0.0])
    assert M.board_gain(y, base, cand) == pytest.approx(200.0 / M.BOARD_ROWS)


def test_board_gain_of_a_candidate_equal_to_base_is_zero(M):
    y = np.array([1.0, 2.0, 3.0])
    base = np.array([4.0, 5.0, 6.0])
    assert M.board_gain(y, base, base) == 0.0


# --- the tail rule ------------------------------------------------------------------------------

def test_trim_mask_drops_the_largest_moves(M):
    base = np.zeros(100)
    cand = np.arange(100, dtype=float)
    keep = M.trim_mask(base, cand, 0.01)
    assert keep.sum() == 99 and not keep[99] and keep[98]


def test_trim_mask_uses_absolute_move_not_signed(M):
    base = np.zeros(100)
    cand = np.arange(100, dtype=float)
    cand[0] = -1e6
    keep = M.trim_mask(base, cand, 0.01)
    assert not keep[0], "a large negative move must be trimmed too"


def test_trim_mask_keeps_everything_at_zero_fraction(M):
    assert M.trim_mask(np.zeros(10), np.arange(10, dtype=float), 0.0).all()


# --- decide: the verdict must be able to say NOT WORKING on a positive mean ---------------------

def _stats(gain, lo, hi, m1=(1.0, 2.0), m7=(1.0, 2.0)):
    return {"gain": gain, "ci95": [lo, hi],
            "monthly": {1: {"gain": m1[0], "ci95": [m1[0], m1[1]]},
                        7: {"gain": m7[0], "ci95": [m7[0], m7[1]]}}}


def test_decide_working_when_every_clause_holds(M):
    v, why = M.decide(_stats(500.0, 400.0, 600.0), trimmed_gain=480.0)
    assert v == "WORKING" and why == []


def test_decide_not_working_when_pooled_ci_includes_zero(M):
    v, why = M.decide(_stats(500.0, -10.0, 900.0), trimmed_gain=480.0)
    assert v == "NOT WORKING" and any("pooled" in w for w in why)


def test_decide_not_working_when_a_month_ci_includes_zero(M):
    v, why = M.decide(_stats(500.0, 400.0, 600.0, m7=(-5.0, 900.0)), trimmed_gain=480.0)
    assert v == "NOT WORKING" and any("month" in w for w in why)


def test_decide_not_working_when_months_disagree_in_sign(M):
    v, why = M.decide(_stats(500.0, 400.0, 600.0, m1=(-900.0, -100.0), m7=(900.0, 1500.0)),
                      trimmed_gain=480.0)
    assert v == "NOT WORKING"
    # without this the month-CI clause carries the test and the sign clause is never exercised
    assert any("disagree" in w for w in why), why


def test_decide_not_working_inside_the_seed_spread(M):
    v, why = M.decide(_stats(40.0, 10.0, 70.0, m1=(5.0, 60.0), m7=(5.0, 60.0)), trimmed_gain=38.0)
    assert v == "NOT WORKING" and any("seed spread" in w for w in why)


def test_decide_not_working_when_the_gain_is_tail_driven(M):
    v, why = M.decide(_stats(500.0, 400.0, 600.0), trimmed_gain=-20.0)
    assert v == "NOT WORKING" and any("trim" in w for w in why)


def test_decide_not_working_on_a_negative_mean(M):
    v, why = M.decide(_stats(-800.0, -900.0, -700.0, m1=(-900.0, -700.0), m7=(-900.0, -700.0)),
                      trimmed_gain=-800.0)
    assert v == "NOT WORKING" and any("mean" in w for w in why)


def test_decide_collects_every_failed_clause_not_just_the_first(M):
    v, why = M.decide(_stats(-10.0, -900.0, 900.0, m1=(-900.0, 900.0), m7=(-900.0, 900.0)),
                      trimmed_gain=-50.0)
    assert v == "NOT WORKING"
    assert len(why) == 6, why        # mean, pooled CI, both month CIs, seed spread, trim
    assert any("mean" in w for w in why) and any("pooled" in w for w in why)
    assert any("seed spread" in w for w in why) and any("trim" in w for w in why)


# --- the join must report a shortfall, never swallow it -----------------------------------------

def test_join_predictions_reports_rows_with_no_prediction(M):
    s = pd.DataFrame({"row": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})
    preds = pd.DataFrame({"row": [1, 2, 3], "cat_delta": [0.1, 0.2, 0.3]})
    out, missing = M.join_predictions(s, preds)
    assert missing == 1 and len(out) == 3
    assert set(out.row) == {1, 2, 3}


def test_join_predictions_is_exact_on_row_not_positional(M):
    s = pd.DataFrame({"row": [10, 20, 30], "y": [1.0, 2.0, 3.0]})
    preds = pd.DataFrame({"row": [30, 10, 20], "cat_delta": [3.0, 1.0, 2.0]})
    out, missing = M.join_predictions(s, preds)
    assert missing == 0
    assert np.allclose(out.sort_values("row").cat_delta, [1.0, 2.0, 3.0])


def test_join_predictions_rejects_duplicate_prediction_rows(M):
    """BC-15's neighbourhood: a duplicated key silently multiplies the population."""
    s = pd.DataFrame({"row": [1, 2], "y": [1.0, 2.0]})
    preds = pd.DataFrame({"row": [1, 1, 2], "cat_delta": [0.1, 0.9, 0.2]})
    with pytest.raises(AssertionError):
        M.join_predictions(s, preds)


def test_join_predictions_rejects_a_nan_key(M):
    """BC-15: pandas.merge joins NaN to NaN, so a missing key must never reach the merge."""
    s = pd.DataFrame({"row": [1.0, np.nan], "y": [1.0, 2.0]})
    preds = pd.DataFrame({"row": [1.0, np.nan], "cat_delta": [0.1, 0.2]})
    with pytest.raises(AssertionError):
        M.join_predictions(s, preds)


# --- boundary cases: the `<=` vs `<` mutants the clause tests above would miss -----------------

def test_decide_not_working_exactly_at_the_seed_spread(M):
    """The bar is `gain > 49`, so exactly 49 must FAIL; `>=` would be a silent widening."""
    v, why = M.decide(_stats(49.0, 10.0, 90.0, m1=(5.0, 60.0), m7=(5.0, 60.0)), trimmed_gain=48.0)
    assert v == "NOT WORKING" and any("seed spread" in w for w in why)


def test_decide_not_working_when_pooled_ci_lower_bound_is_exactly_zero(M):
    v, why = M.decide(_stats(500.0, 0.0, 900.0), trimmed_gain=480.0)
    assert v == "NOT WORKING" and any("pooled" in w for w in why)


def test_decide_not_working_when_a_month_ci_lower_bound_is_exactly_zero(M):
    v, why = M.decide(_stats(500.0, 400.0, 600.0, m7=(0.0, 900.0)), trimmed_gain=480.0)
    assert v == "NOT WORKING" and any("month" in w for w in why)


def test_decide_not_working_when_the_trimmed_gain_is_exactly_zero(M):
    v, why = M.decide(_stats(500.0, 400.0, 600.0), trimmed_gain=0.0)
    assert v == "NOT WORKING" and any("trim" in w for w in why)


def test_decide_not_working_when_the_mean_gain_is_exactly_zero(M):
    v, why = M.decide(_stats(0.0, 400.0, 600.0), trimmed_gain=480.0)
    assert v == "NOT WORKING" and any("mean" in w for w in why)


def test_decide_working_just_above_the_seed_spread(M):
    """The complement of the boundary test: 49.0001 with every other clause clean must PASS."""
    v, why = M.decide(_stats(49.0001, 10.0, 90.0, m1=(5.0, 60.0), m7=(5.0, 60.0)), trimmed_gain=48.0)
    assert v == "WORKING" and why == []


# --- escapes caught by the mutation rehearsal (2026-09-20): tests that passed for the WRONG reason

def _stats2(gain, lo, hi, months):
    """`_stats` ties a month's gain to its CI lower bound, so a month-CI test could be satisfied by
    the sign-disagreement clause instead. This helper decouples them."""
    return {"gain": gain, "ci95": [lo, hi],
            "monthly": {m: {"gain": g, "ci95": [clo, chi]} for m, (g, clo, chi) in months.items()}}


def test_decide_month_ci_bound_fires_on_its_own_clause_not_the_sign_clause(M):
    """Both months positive (so the sign clause cannot fire); month 7's CI lower bound is exactly 0."""
    v, why = M.decide(_stats2(500.0, 400.0, 600.0,
                              {1: (10.0, 5.0, 20.0), 7: (5.0, 0.0, 900.0)}), trimmed_gain=480.0)
    assert v == "NOT WORKING"
    assert any(w.startswith("month 7 CI includes 0") for w in why), why
    assert not any("disagree" in w for w in why), why


def test_decide_sign_clause_fires_independently_of_the_ci_clause(M):
    """A month measuring exactly zero gain must not pass a 'months agree' check silently.

    A significantly-negative month always trips the CI clause too, so this zero-gain month is the
    only input that exercises the sign clause alone.
    """
    v, why = M.decide(_stats2(500.0, 400.0, 600.0,
                              {1: (10.0, 5.0, 20.0), 7: (0.0, 1.0, 2.0)}), trimmed_gain=480.0)
    assert v == "NOT WORKING"
    assert any("disagree" in w for w in why), why
    assert not any("CI includes 0" in w for w in why), why


def test_join_predictions_rejects_a_nan_key_in_s_specifically(M):
    s = pd.DataFrame({"row": [1.0, np.nan], "y": [1.0, 2.0]})
    preds = pd.DataFrame({"row": [1.0, 2.0], "cat_delta": [0.1, 0.2]})
    with pytest.raises(AssertionError, match="NaN key in s.row"):
        M.join_predictions(s, preds)


def test_join_predictions_rejects_a_nan_key_in_preds_specifically(M):
    s = pd.DataFrame({"row": [1.0, 2.0], "y": [1.0, 2.0]})
    preds = pd.DataFrame({"row": [1.0, np.nan], "cat_delta": [0.1, 0.2]})
    with pytest.raises(AssertionError, match="NaN key in preds.row"):
        M.join_predictions(s, preds)


def test_join_predictions_rejects_duplicate_rows_in_s(M):
    s = pd.DataFrame({"row": [1, 1, 2], "y": [1.0, 2.0, 3.0]})
    preds = pd.DataFrame({"row": [1, 2], "cat_delta": [0.1, 0.2]})
    with pytest.raises(AssertionError, match="duplicate row in s"):
        M.join_predictions(s, preds)


# --- the reason string must describe what happened, not just which clause failed ----------------
# Caught on the first real PBL run 2026-09-20: a CI of [-2333, -1675] was reported as "includes 0".
# Correct as a bar failure, wrong as a description, and the shape that becomes a phantom in a report.

def test_decide_describes_an_entirely_negative_pooled_ci_as_negative_not_as_including_zero(M):
    v, why = M.decide(_stats2(-1964.9, -2333.3, -1675.0,
                              {1: (-781.4, -945.0, -634.6), 7: (-1183.4, -1517.8, -954.3)}),
                      trimmed_gain=-1348.0)
    assert v == "NOT WORKING"
    assert any("pooled CI is entirely negative" in w for w in why), why
    assert not any("pooled CI includes 0" in w for w in why), why


def test_decide_describes_an_entirely_negative_month_ci_as_negative(M):
    v, why = M.decide(_stats2(-100.0, -200.0, -50.0,
                              {1: (-40.0, -60.0, -20.0), 7: (-60.0, -90.0, -30.0)}),
                      trimmed_gain=-90.0)
    assert any("month 1 CI is entirely negative" in w for w in why), why
    assert not any("month 1 CI includes 0" in w for w in why), why


def test_decide_still_says_includes_zero_when_the_ci_straddles_zero(M):
    v, why = M.decide(_stats2(500.0, -10.0, 900.0,
                              {1: (10.0, 5.0, 20.0), 7: (20.0, 6.0, 30.0)}), trimmed_gain=480.0)
    assert any("pooled CI includes 0" in w for w in why), why


# --- day_block_bootstrap: the function that produces EVERY CI this arm reports -------------------
# It had NO test. The "X1 cross-check" that was claimed to guard it (pbl_build.py score_arm) is an
# algebraic identity -- opc's gain_weighted_mse * n / 339551 reduces to gain.sum()/344841 exactly, so
# it can never fail. Caught by adversarial review at the end-of-session review 2026-09-20.

def _flat(n_days, rows_per_day, gain_per_row, month=1, gain_days=None):
    """y=0, cand=0, base=sqrt(gain) on the chosen days -> per-row gain is exactly gain_per_row."""
    days = np.repeat(np.arange(n_days), rows_per_day)
    n = len(days)
    y = np.zeros(n)
    cand = np.zeros(n)
    base = np.zeros(n)
    sel = np.isin(days, np.arange(n_days) if gain_days is None else np.asarray(gain_days))
    base[sel] = np.sqrt(gain_per_row)
    return y, base, cand, days, np.full(n, month)


def test_day_block_bootstrap_point_estimate_equals_board_gain(M):
    y, base, cand, day, month = _flat(10, 50, 400.0)
    s = M.day_block_bootstrap(y, base, cand, day, month)
    assert s["gain"] == pytest.approx(M.board_gain(y, base, cand))


def test_day_block_bootstrap_is_deterministic_at_a_fixed_seed(M):
    y, base, cand, day, month = _flat(8, 30, 900.0)
    a = M.day_block_bootstrap(y, base, cand, day, month)
    b = M.day_block_bootstrap(y, base, cand, day, month)
    assert a["ci95"] == b["ci95"] and a["gain"] == b["gain"]


def test_day_block_bootstrap_resamples_DAYS_not_rows(M):
    """THE property that makes this a day-blocked bootstrap.

    All the gain sits in ONE of ten days. Resampling days, that day is absent from
    (9/10)^10 = 34.9 % of draws, so the 2.5 % quantile must be exactly 0. Resampling ROWS, its 100
    rows would be absent from ~(0.9)^1000 of draws, i.e. never, and the lower bound would sit near
    the point estimate. Correlated rows inside a day are the whole reason the block exists.
    """
    y, base, cand, day, month = _flat(10, 100, 10_000.0, gain_days=[0])
    s = M.day_block_bootstrap(y, base, cand, day, month)
    assert s["gain"] > 0
    assert s["ci95"][0] == 0.0, f"lower bound {s['ci95'][0]} -- this is resampling rows, not days"


def test_day_block_bootstrap_ci_collapses_when_every_day_is_identical(M):
    """The complement: with the signal spread evenly, every resample is the same, so the interval
    has zero width and excludes 0. A bootstrap that shuffled rows across days would show spread."""
    y, base, cand, day, month = _flat(10, 100, 10_000.0)
    s = M.day_block_bootstrap(y, base, cand, day, month)
    assert s["ci95"][0] > 0
    assert s["ci95"][0] == pytest.approx(s["gain"]) and s["ci95"][1] == pytest.approx(s["gain"])


def test_day_block_bootstrap_monthly_gains_sum_to_the_pooled_gain(M):
    y1, b1, c1, d1, m1 = _flat(6, 40, 500.0, month=1)
    y7, b7, c7, d7, m7 = _flat(6, 40, 900.0, month=7)
    cat = lambda a, b: np.concatenate([a, b])
    s = M.day_block_bootstrap(cat(y1, y7), cat(b1, b7), cat(c1, c7), cat(d1, d7), cat(m1, m7))
    assert sum(v["gain"] for v in s["monthly"].values()) == pytest.approx(s["gain"])
    assert set(s["monthly"]) == {1, 7}


def test_day_block_bootstrap_monthly_row_counts_sum_to_the_population(M):
    y1, b1, c1, d1, m1 = _flat(6, 40, 500.0, month=1)
    y7, b7, c7, d7, m7 = _flat(5, 30, 900.0, month=7)
    cat = lambda a, b: np.concatenate([a, b])
    s = M.day_block_bootstrap(cat(y1, y7), cat(b1, b7), cat(c1, c7), cat(d1, d7), cat(m1, m7))
    assert s["n"] == 6 * 40 + 5 * 30
    assert sum(v["n"] for v in s["monthly"].values()) == s["n"]


def test_day_block_bootstrap_normalises_by_the_board_rows_not_the_population(M):
    """A 500-row population and a 5,000-row one with the same TOTAL gain must report the same
    number. Rescaling by the resampled count instead would make the small population look larger."""
    ya, ba, ca, da, ma = _flat(10, 50, 1000.0)
    yb, bb, cb, db, mb = _flat(10, 500, 100.0)
    sa = M.day_block_bootstrap(ya, ba, ca, da, ma)
    sb = M.day_block_bootstrap(yb, bb, cb, db, mb)
    assert sa["gain"] == pytest.approx(sb["gain"])
    assert sa["gain"] == pytest.approx(10 * 50 * 1000.0 / M.BOARD_ROWS)


def test_day_block_bootstrap_reports_a_negative_gain_with_a_negative_interval(M):
    """The shape this arm actually measured: the candidate is worse on every day."""
    y, base, cand, day, month = _flat(10, 100, 10_000.0)
    s = M.day_block_bootstrap(y, cand, base, day, month)   # base and cand swapped
    assert s["gain"] < 0 and s["ci95"][1] < 0


def test_ci_quantiles_are_the_registered_95_percent_interval(M):
    assert M.CI_Q == (0.025, 0.975)


def test_day_block_bootstrap_pooled_ci_brackets_the_pooled_point_estimate(M):
    """Catches a pooled accumulator that keeps only the LAST month (`vals = sums` for `vals += sums`):
    the interval would then describe month 7 alone and would not contain the pooled gain."""
    y1, b1, c1, d1, m1 = _flat(6, 40, 100.0, month=1)
    y7, b7, c7, d7, m7 = _flat(6, 40, 10_000.0, month=7)
    cat = lambda a, b: np.concatenate([a, b])
    s = M.day_block_bootstrap(cat(y1, y7), cat(b1, b7), cat(c1, c7), cat(d1, d7), cat(m1, m7))
    assert s["ci95"][0] <= s["gain"] <= s["ci95"][1], (s["ci95"], s["gain"])


def test_day_block_bootstrap_each_month_ci_is_scaled_by_ITS_OWN_row_count(M):
    """Catches rescaling a month's interval by the POOLED population. Months here have deliberately
    different sizes; with every day identical, each month's interval collapses onto its own gain, so
    using n_pop instead of n_m inflates both months' intervals."""
    y1, b1, c1, d1, m1 = _flat(5, 20, 10_000.0, month=1)     # 100 rows
    y7, b7, c7, d7, m7 = _flat(5, 200, 10_000.0, month=7)    # 1,000 rows
    cat = lambda a, b: np.concatenate([a, b])
    s = M.day_block_bootstrap(cat(y1, y7), cat(b1, b7), cat(c1, c7), cat(d1, d7), cat(m1, m7))
    for m in (1, 7):
        lo, hi = s["monthly"][m]["ci95"]
        assert lo == pytest.approx(s["monthly"][m]["gain"]), (m, lo, s["monthly"][m]["gain"])
        assert hi == pytest.approx(s["monthly"][m]["gain"])
    assert s["monthly"][7]["gain"] == pytest.approx(10 * s["monthly"][1]["gain"])


def test_decide_seed_spread_phrase_describes_the_observation_not_the_clause(M):
    """BC-19 inside the function that defined it: -1,964.9 is not 'inside' a spread of 49."""
    v, why = M.decide(_stats2(-1964.9, -2333.3, -1675.0,
                              {1: (-781.4, -945.0, -634.6), 7: (-1183.4, -1517.8, -954.3)}),
                      trimmed_gain=-1348.0)
    assert any("cannot clear the seed spread" in w for w in why), why
    assert not any("is inside the seed spread" in w for w in why), why
    v2, why2 = M.decide(_stats2(40.0, 10.0, 70.0, {1: (5.0, 1.0, 60.0), 7: (5.0, 1.0, 60.0)}),
                        trimmed_gain=38.0)
    assert any("is inside the seed spread" in w for w in why2), why2
