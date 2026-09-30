"""Tests for scripts/analysis/mxi_build.py — MXI (plans/PREREG_mxi_unmatched_capacity_2026_09_21.md, e98627ad...).

Written FIRST and seen RED (ImportError: mxi_build did not exist) before the runner was written.

What must not go wrong: (1) the MXI body must be v24's body with ONLY max_iter changed (at 200 it must be
bit-identical to MAU's candidate fit, via msw_build.fit_candidate which is test-pinned to MAU's inline code);
(2) the composition of each leg must be MAU's / MAV's own; (3) the clauses are the registration's, exactly;
(4) every bootstrap refuses a degenerate day vector (BC-23); (5) the incumbent gate is exact.

Mutation rehearsal 2026-09-21, WITHOUT -x, python -B, in a scratch mirror (real runner sha verified unchanged):
X1 extra weight 0.30, X2 max_iter ignored, X3 extra target unclipped, X4 no storm anchor, X5 BC-23 guard dropped,
X6 monster rows kept, X7 forward clause dropped, X8 LFPG `> 0` -> `>= 0`, X9 WORSE branch dropped, X10 gate uses
allclose, X11 hash-log check dropped, X12 MXI_ITER 500, X13 NF_PARAMS mutated in place, X14 incumbent/arm swapped
in delta_stats -> 14/14 KILLED; every test function seen red against a targeted mutant.
Post-mortem round (BC-24, after the 13:23 UTC crash): test_stats_... was RED against the real bug (AttributeError at
mxi_build.py:222); test_assemble_... red before assemble existed; then Y1 LFPG exclusion ignored, Y2 cap count `>`,
Y3 gate skips seeds, Y4 gate skips ids, Y5 rows_keep ignored, Y6 seed columns from the mean, Y7 rows_changed `==`,
Y8 prediction legs swapped -> 8/8 KILLED.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "scripts/analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.analysis import mxi_build as X  # noqa: E402
from scripts.analysis import msw_build as W  # noqa: E402


def _xy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 5, n).astype(float)})
    y = 400 * np.sin(2 * x.a) + 300 * np.tanh(x.b * x.c) + 200 * x.a * x.b + rng.normal(0, 20, n)
    return x, np.clip(y.to_numpy(), -3000, 3000)


def test_constants_are_the_registered_ones():
    assert X.INCUMBENT_ITER == 200 == X.uc.bs.NF_PARAMS["max_iter"]
    assert X.MXI_ITER == 1000
    assert X.REGISTERED_SHA.startswith("e98627ad")
    assert X.MONSTER_IDS == (182378789.0, 182377662.0)


def test_body_params_change_only_max_iter():
    p = X.body_params(1000)
    assert p == {**X.uc.bs.NF_PARAMS, "max_iter": 1000}
    assert X.uc.bs.NF_PARAMS["max_iter"] == 200                     # the shipped dict is not mutated


@pytest.mark.slow          # 0.9-2.3 s depending on machine load
def test_fit_body_at_200_is_maus_candidate_fit_bit_for_bit():
    x, y = _xy()
    xe, ye = _xy(1200, seed=1)
    fi, va = np.arange(2700), np.arange(2700, 3000)
    ours = X.fit_body(x.iloc[fi], y[fi], xe, ye, 0, x.iloc[va], y[va], 200)
    theirs, _ = W.fit_candidate(x.iloc[fi], y[fi], xe, ye, 0.25, 0, x.iloc[va], y[va])   # pinned to MAU's inline fit
    np.testing.assert_array_equal(ours.predict(x), theirs.predict(x))


@pytest.mark.slow          # 2.0 s: needs enough rows to keep improving past 200 rounds
def test_fit_body_max_iter_reaches_the_model_and_early_stopping_ends_it():
    x, y = _xy(6000)
    xe, ye = _xy(1500, seed=2)
    fi, va = np.arange(5400), np.arange(5400, 6000)
    capped = X.fit_body(x.iloc[fi], y[fi], xe, ye, 0, x.iloc[va], y[va], 200)
    free = X.fit_body(x.iloc[fi], y[fi], xe, ye, 0, x.iloc[va], y[va], 1000)
    assert capped.n_iter_ == 200                                      # this fixture keeps improving past 200
    assert 200 < free.n_iter_ < 1000                                  # ...and early stopping, not the cap, ends it


def test_fit_body_clips_the_extra_target_like_mau():
    x, y = _xy(800)
    xe, ye = _xy(300, seed=3)
    big = ye.copy()
    big[:50] = 9e5                                                    # beyond the 3,000 s cap
    fi, va = np.arange(700), np.arange(700, 800)
    a = X.fit_body(x.iloc[fi], y[fi], xe, np.clip(big, -3000, 3000), 0, x.iloc[va], y[va], 50)
    b = X.fit_body(x.iloc[fi], y[fi], xe, big, 0, x.iloc[va], y[va], 50)
    np.testing.assert_array_equal(a.predict(x), b.predict(x))


# -------------------------------------------------------------------------------------------- composition
def _parts(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return (pd.Series(rng.uniform(0, 0.4, n)), pd.Series(rng.uniform(-600, 9000, n)),
            pd.Series(np.r_[rng.uniform(300, 900, n - 2), 5000.0, 4000.0]), pd.Series(np.r_[0.0, 350.0, np.full(n - 2, np.nan)]))


def test_compose_is_mixture_then_nf_hybrid_then_storm_anchor():
    p, sp, cells, a = _parts()
    body = np.linspace(400, 900, len(p))
    want = X.ca.anchored_values(np.rint(X.uc.bs.mixture(p, sp, X.uc.bs.nf_hybrid(cells, body))), a)
    np.testing.assert_array_equal(X.compose(p, sp, cells, body, a), want)
    assert (X.compose(p, sp, cells, body, a)[1] - np.rint(X.uc.bs.mixture(p, sp, X.uc.bs.nf_hybrid(cells, body)))[1]) == 50.0


# -------------------------------------------------------------------------------------------- stats + clauses
def _days(n_days=12, per=6):
    day = np.r_[np.repeat(np.arange(1, n_days + 1), per), np.repeat(np.arange(182, 182 + n_days), per)]
    month = np.r_[np.full(per * n_days, 1), np.full(per * n_days, 7)]
    return day, month


def test_delta_stats_refuses_a_degenerate_day_vector_bc23():
    day, month = _days()
    y = np.full(day.size, 600.0)
    with pytest.raises(ValueError, match="distinct day blocks"):
        X.delta_stats(y, y + 10, y + 5, month.copy(), month)


def test_delta_stats_is_the_board_difference_and_identity_is_exactly_zero():
    day, month = _days()
    rng = np.random.default_rng(4)
    y = rng.uniform(300, 900, day.size)
    inc, arm = y + rng.normal(0, 60, y.size), y + rng.normal(0, 40, y.size)
    st = X.delta_stats(y, inc, arm, day, month)
    assert st["gain"] == pytest.approx((((y - inc) ** 2) - ((y - arm) ** 2)).sum() / 344841, rel=1e-12)  # fp64 floor
    null = X.delta_stats(y, inc, inc.copy(), day, month)
    assert null["gain"] == 0.0 and null["ci95"] == [0.0, 0.0]


def test_monster_split_removes_exactly_the_two_rows():
    ids = np.array([1.0, 182378789.0, 5.0, 182377662.0, 9.0])
    y = np.array([500.0, 84240.0, 600.0, 58206.0, 700.0])
    inc, arm = y + 100, y + np.array([50.0, -900.0, 50.0, 400.0, 50.0])
    got = X.without_monsters(ids, y, inc, arm)
    keep = np.array([True, False, True, False, True])
    assert got == pytest.approx((((y - inc) ** 2 - (y - arm) ** 2)[keep]).sum() / 344841, rel=1e-12)


def _st(gain, lo, hi, months=((1, 1.0), (7, 1.0))):
    return {"gain": gain, "ci95": [lo, hi], "monthly": {m: {"gain": g, "ci95": [g - 1, g + 1]} for m, g in months}}


def test_verdict_needs_all_three_clauses():
    good = _st(10, 1, 20)
    assert X.verdict(good, _st(5, 1, 9, ((10, 1), (11, 1), (12, 1))), {"gain": 3.0}) == "TRUE"
    assert X.verdict(_st(10, -1, 20), good, {"gain": 3.0}) == "NOT WORKING"                    # screen CI
    assert X.verdict(_st(10, 1, 20, ((1, -0.1), (7, 5))), good, {"gain": 3.0}) == "NOT WORKING"  # a screen month < 0
    assert X.verdict(good, _st(5, -1, 9), {"gain": 3.0}) == "NOT WORKING"                     # forward CI
    assert X.verdict(good, good, {"gain": 0.0}) == "NOT WORKING"                              # LFPG-excluded point
    assert X.verdict(_st(-30, -60, -1), good, {"gain": 3.0}) == "WORSE"                       # screen upper < 0


def test_assert_reproduces_is_exact():
    X.assert_reproduces(np.array([1.0, 2.0]), np.array([1.0, 2.0]), "x")
    with pytest.raises(AssertionError, match="VOID"):
        X.assert_reproduces(np.array([1.0, 2.0]), np.array([1.0, np.nextafter(2.0, 3.0)]), "x")


def test_preflight_refuses_existing_outputs_and_unhashed_registration(tmp_path):
    prereg, log = tmp_path / "p.md", tmp_path / "h.log"
    prereg.write_text("text\n")
    sha = X.U.sha(prereg)
    log.write_text(f"{sha}  p.md  registered\n")
    X.preflight(tmp_path / "r.json", tmp_path / "p.parquet", prereg, log, sha)
    with pytest.raises(AssertionError, match="registration"):
        X.preflight(tmp_path / "r.json", tmp_path / "p.parquet", prereg, log, "0" * 64)
    log.write_text("nothing\n")
    with pytest.raises(AssertionError, match="hash log"):
        X.preflight(tmp_path / "r.json", tmp_path / "p.parquet", prereg, log, sha)
    (tmp_path / "r.json").write_text("{}")
    with pytest.raises(FileExistsError, match="no silent rerun"):
        X.preflight(tmp_path / "r.json", tmp_path / "p.parquet", prereg, log, sha)


# -------------------------------------------------------------------------------------------- post-compute path
# Regression for the 2026-09-21 13:23 UTC crash: `_stats` grouped a numpy array (AttributeError) AFTER both legs had
# run and both incumbent gates had passed, voiding the run. Everything after the fits is exercised here on
# real-shaped leg frames, including JSON serialisation, so a post-compute defect can never again surface first in
# the real run. Bug class BC-24 (reports/bug_classes.md).
import json  # noqa: E402


def _leg_frame(months, seed=0, n_days=12, per=5):
    rng = np.random.default_rng(seed)
    rows = []
    for m in months:
        base = {1: 1, 7: 182, 10: 274, 11: 305, 12: 335}[m]
        for d in range(n_days):
            for k in range(per):
                rows.append({"month": m, "day": base + d, "ADEP_mvt": ("LFPG", "EHAM", "EDDF")[k % 3]})
    f = pd.DataFrame(rows)
    n = len(f)
    f["MVT_ID_mvt"] = np.arange(1000.0, 1000.0 + n)
    f.loc[0, "MVT_ID_mvt"] = X.MONSTER_IDS[0]
    f["date"], f["sp"], f["A"] = "2025-01-01", 0.0, np.nan
    f["y"] = rng.uniform(300, 1500, n)
    for it, noise in ((200, 90.0), (1000, 70.0)):
        seeds = [f.y + rng.normal(0, noise, n) for _ in X.U.SEEDS]
        for s, v in zip(X.U.SEEDS, seeds):
            f[f"it{it}_seed{s}"] = np.rint(v)
        f[f"it{it}"] = np.rint(np.mean(seeds, axis=0))
    f.loc[3, "it1000"] = f.loc[3, "it200"]                            # at least one unchanged row
    return f


def test_stats_breaks_the_delta_down_by_airport_and_seed():
    f = _leg_frame((1, 7))
    st = X._stats(f)
    y, inc, arm = f.y.to_numpy(), f.it200.to_numpy(), f.it1000.to_numpy()
    total = (((y - inc) ** 2) - ((y - arm) ** 2)).sum() / 344841
    assert st["delta"]["gain"] == pytest.approx(total, rel=1e-12)                       # fp64 floor
    assert set(st["airport_deltas"]) == {"LFPG", "EHAM", "EDDF"}
    assert sum(st["airport_deltas"].values()) == pytest.approx(total, rel=1e-9)        # breakdown sums to the total
    assert st["identity"]["gain"] == 0.0 and st["identity"]["ci95"] == [0.0, 0.0]
    assert st["oracle"]["gain"] == pytest.approx((((y - inc) ** 2).sum()) / 344841, rel=1e-12)
    assert st["delta_without_monsters"] == pytest.approx(X.without_monsters(f.MVT_ID_mvt, y, inc, arm), rel=1e-12)
    assert set(st["seed_deltas"]) == {"0", "1", "2"}
    assert st["rows_changed"] == int((inc != arm).sum()) and st["rows_changed"] < len(f)
    assert st["n"] == len(f)


def test_assemble_builds_a_serialisable_report_and_the_prediction_frame():
    screen, forward = _leg_frame((1, 7)), _leg_frame((10, 11, 12), seed=1)
    fits = [{"leg": "screen", "month": 1, "seed": 0, "max_iter": 1000, "n_iter": 1000},
            {"leg": "screen", "month": 1, "seed": 0, "max_iter": 200, "n_iter": 200},
            {"leg": "forward", "month": 10, "seed": 0, "max_iter": 1000, "n_iter": 412}]
    report, preds = X.assemble(screen, forward, fits)
    json.dumps(report, allow_nan=False)                               # the step the real run never reached
    assert report["verdict"] == X.verdict(report["screen"]["delta"], report["forward"]["delta"],
                                          report["screen_lfpg_excluded"])
    lf = screen.ADEP_mvt.ne("LFPG").to_numpy()
    want = X.delta_stats(screen.y[lf], screen.it200[lf], screen.it1000[lf], screen.day[lf], screen.month[lf])
    assert report["screen_lfpg_excluded"] == want
    assert report["mxi_fits_at_cap"] == 1 and report["incumbent_fits_at_cap"] == 1
    assert report["mxi_n_iter"] == [412, 1000]
    assert list(preds.leg.unique()) == ["screen", "forward"] and len(preds) == len(screen) + len(forward)


def test_gate_checks_ids_candidate_and_every_seed_exactly():
    f = _leg_frame((1, 7))
    ref = pd.DataFrame({"MVT_ID_mvt": f.MVT_ID_mvt, "candidate": f.it200,
                        **{f"candidate_seed{s}": f[f"it200_seed{s}"] for s in X.U.SEEDS}})
    X._gate(f, ref, "fixture")
    for col, what in (("candidate", "candidate"), ("candidate_seed2", "seed 2"), ("MVT_ID_mvt", "row ids")):
        bad = ref.copy()
        bad.loc[bad.index[5], col] += 1.0
        with pytest.raises(AssertionError, match=f"VOID: incumbent does not reproduce fixture {what}"):
            X._gate(f, bad, "fixture")


def test_stack_composes_each_iteration_and_keeps_only_the_requested_rows():
    rng = np.random.default_rng(5)
    n = 8
    te = pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float), "ADEP_mvt": ["LIRF", "EHAM"] * 4, "date": "2025-07-01",
                       "month": 7, "y": rng.uniform(300, 900, n), "sp": rng.uniform(-600, 9000, n),
                       "A": np.r_[np.nan, 400.0, np.full(n - 2, np.nan)],
                       "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-07-02T10:00Z"] * n)})
    p, cells = pd.Series(rng.uniform(0, 0.4, n)), pd.Series(np.r_[rng.uniform(300, 900, n - 1), 5000.0])
    preds = {it: [rng.uniform(300, 900, n) for _ in X.U.SEEDS] for it in X.ITERS}
    keep = te.ADEP_mvt.ne("LIRF").to_numpy()
    out = X._stack(te, keep, p, te.sp, cells, preds)
    assert len(out) == int(keep.sum()) and (out.ADEP_mvt != "LIRF").all()
    assert (out.day == 183).all()
    for it in X.ITERS:
        body = np.mean(preds[it], axis=0)
        np.testing.assert_array_equal(out[f"body_{it}"], body[keep])
        np.testing.assert_array_equal(out[f"it{it}"], X.compose(p, te.sp, cells, body, te.A)[keep])
        for j, s in enumerate(X.U.SEEDS):
            np.testing.assert_array_equal(out[f"it{it}_seed{s}"], X.compose(p, te.sp, cells, preds[it][j], te.A)[keep])
