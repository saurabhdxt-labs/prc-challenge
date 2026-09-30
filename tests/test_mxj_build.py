"""Tests for scripts/analysis/mxj_build.py — MXJ (plans/PREREG_mxj_unmatched_patience_2026_09_21.md, b6da70bd...).

Written FIRST and seen RED (ImportError: mxj_build did not exist) before the runner was written.

What must not go wrong: (1) the incumbent is v25's body exactly (MXI's max_iter-1000 fit, bit-identical) and the arm
differs ONLY in n_iter_no_change = 50; (2) the incumbent gate compares against MXI's persisted it1000 columns exactly;
(3) every line after the fits runs here on a synthetic output before the real run, including JSON (bug class BC-24).

Mutation rehearsal 2026-09-21, WITHOUT -x, python -B, in a scratch mirror (real runner sha verified unchanged):
J1 arm patience 10, J2 sample weight dropped, J3 gate skips seeds, J4 gate skips ids, J5 the BC-24 numpy group-by
re-injected, J6 LFPG exclusion ignored, J7 cap count `>`, J8 rows_keep ignored, J9 seed columns from the mean,
J10 inc_n_iter includes arm fits -> 10/10 KILLED; every test function seen red.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "scripts/analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scripts.analysis import mxj_build as J  # noqa: E402

XI = J.XI


def test_constants_are_the_registered_ones():
    assert J.REGISTERED_SHA.startswith("b6da70bd")
    assert J.ARMS["inc"] == XI.body_params(1000)
    diff = {k: v for k, v in J.ARMS["arm"].items() if J.ARMS["inc"].get(k) != v}
    assert diff == {"n_iter_no_change": 50}
    assert XI.uc.bs.NF_PARAMS["max_iter"] == 200                      # the shipped dict is not mutated


def _xy(n=2500, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    return x, np.clip((400 * np.sin(2 * x.a) + 150 * x.a * x.b + rng.normal(0, 20, n)).to_numpy(), -3000, 3000)


@pytest.mark.slow          # three HGB fits
def test_fit_body_incumbent_is_mxis_fit_and_the_arm_only_changes_patience():
    x, y = _xy()
    xe, ye = _xy(800, seed=1)
    fi, va = np.arange(2200), np.arange(2200, 2500)
    ours = J.fit_body(x.iloc[fi], y[fi], xe, ye, 0, x.iloc[va], y[va], J.ARMS["inc"])
    theirs = XI.fit_body(x.iloc[fi], y[fi], xe, ye, 0, x.iloc[va], y[va], 1000)
    np.testing.assert_array_equal(ours.predict(x), theirs.predict(x))
    arm = J.fit_body(x.iloc[fi], y[fi], xe, ye, 0, x.iloc[va], y[va], J.ARMS["arm"])
    assert arm.n_iter_no_change == 50 and ours.n_iter_no_change == 10
    assert arm.n_iter_ >= ours.n_iter_                                # more patience can only fit longer


# -------------------------------------------------------------------------------------------- post-compute path
def _leg_frame(months, seed=0, n_days=12, per=5):
    rng = np.random.default_rng(seed)
    rows = [{"month": m, "day": {1: 1, 7: 182, 10: 274, 11: 305, 12: 335}[m] + d, "ADEP_mvt": ("LFPG", "EHAM", "EDDF")[k % 3]}
            for m in months for d in range(n_days) for k in range(per)]
    f = pd.DataFrame(rows)
    n = len(f)
    f["MVT_ID_mvt"] = np.arange(1000.0, 1000.0 + n)
    f.loc[0, "MVT_ID_mvt"] = XI.MONSTER_IDS[0]
    f["date"], f["sp"], f["A"] = "2025-01-01", 0.0, np.nan
    f["y"] = rng.uniform(300, 1500, n)
    for name, noise in (("inc", 90.0), ("arm", 70.0)):
        seeds = [f.y + rng.normal(0, noise, n) for _ in XI.U.SEEDS]
        for s, v in zip(XI.U.SEEDS, seeds):
            f[f"{name}_seed{s}"] = np.rint(v)
        f[name] = np.rint(np.mean(seeds, axis=0))
    return f


def test_gate_compares_the_incumbent_with_mxis_it1000_exactly():
    f = _leg_frame((1, 7))
    ref = pd.DataFrame({"MVT_ID_mvt": f.MVT_ID_mvt, "it1000": f.inc, **{f"it1000_seed{s}": f[f"inc_seed{s}"] for s in XI.U.SEEDS}})
    J._gate(f, ref, "fixture")
    for col, what in (("it1000", "it1000"), ("it1000_seed1", "seed 1"), ("MVT_ID_mvt", "row ids")):
        bad = ref.copy()
        bad.loc[bad.index[4], col] += 1.0
        with pytest.raises(AssertionError, match=f"VOID: incumbent does not reproduce fixture {what}"):
            J._gate(f, bad, "fixture")


def test_stats_breaks_the_delta_down_by_airport_and_seed():
    f = _leg_frame((1, 7))
    st = J._stats(f)
    y, inc, arm = f.y.to_numpy(), f.inc.to_numpy(), f.arm.to_numpy()
    total = (((y - inc) ** 2) - ((y - arm) ** 2)).sum() / 344841
    assert st["delta"]["gain"] == pytest.approx(total, rel=1e-12)                   # fp64 floor
    assert sum(st["airport_deltas"].values()) == pytest.approx(total, rel=1e-9)
    assert st["identity"]["gain"] == 0.0 and st["identity"]["ci95"] == [0.0, 0.0]
    assert st["delta_without_monsters"] == pytest.approx(XI.without_monsters(f.MVT_ID_mvt, y, inc, arm), rel=1e-12)
    assert set(st["seed_deltas"]) == {"0", "1", "2"} and st["rows_changed"] == int((inc != arm).sum())


def test_assemble_builds_a_serialisable_report():
    screen, forward = _leg_frame((1, 7)), _leg_frame((10, 11, 12), seed=1)
    fits = [{"leg": "screen", "arm": "arm", "n_iter": 1000}, {"leg": "screen", "arm": "inc", "n_iter": 300},
            {"leg": "forward", "arm": "arm", "n_iter": 412}]
    report, preds = J.assemble(screen, forward, fits)
    json.dumps(report, allow_nan=False)
    assert report["verdict"] == XI.verdict(report["screen"]["delta"], report["forward"]["delta"], report["screen_lfpg_excluded"])
    lf = screen.ADEP_mvt.ne("LFPG").to_numpy()
    assert report["screen_lfpg_excluded"] == XI.delta_stats(screen.y[lf], screen.inc[lf], screen.arm[lf], screen.day[lf], screen.month[lf])
    assert report["arm_fits_at_cap"] == 1 and report["arm_n_iter"] == [412, 1000] and report["inc_n_iter"] == [300]
    assert list(preds.leg.unique()) == ["screen", "forward"] and len(preds) == len(screen) + len(forward)


def test_stack_composes_both_arms_and_keeps_only_the_requested_rows():
    rng = np.random.default_rng(5)
    n = 8
    te = pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float), "ADEP_mvt": ["LIRF", "EHAM"] * 4, "date": "2025-07-01",
                       "month": 7, "y": rng.uniform(300, 900, n), "sp": rng.uniform(-600, 9000, n),
                       "A": np.r_[np.nan, 400.0, np.full(n - 2, np.nan)],
                       "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-07-02T10:00Z"] * n)})
    p, cells = pd.Series(rng.uniform(0, 0.4, n)), pd.Series(np.r_[rng.uniform(300, 900, n - 1), 5000.0])
    preds = {name: [rng.uniform(300, 900, n) for _ in XI.U.SEEDS] for name in J.ARMS}
    keep = te.ADEP_mvt.ne("LIRF").to_numpy()
    out = J._stack(te, keep, p, te.sp, cells, preds)
    assert len(out) == int(keep.sum())
    for name in J.ARMS:
        np.testing.assert_array_equal(out[name], XI.compose(p, te.sp, cells, np.mean(preds[name], axis=0), te.A)[keep])
        for j, s in enumerate(XI.U.SEEDS):
            np.testing.assert_array_equal(out[f"{name}_seed{s}"], XI.compose(p, te.sp, cells, preds[name][j], te.A)[keep])
