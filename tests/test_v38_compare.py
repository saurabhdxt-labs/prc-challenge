"""Tests for scripts/analysis/v38_compare.py (arm v38 "DWG", plans/PREREG_v38_dwg_2026_09_27.md).

Written before the module (seen RED: ModuleNotFoundError / missing functions). Each test names what it kills.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("v38_compare", ROOT / "scripts" / "analysis" / "v38_compare.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["v38_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load()
N = C.BOARD_N


def _oof(seed=0, n_per=200, shift=None):
    """Synthetic OOF frame in the r12_stack layout: 10 airports, 20 day blocks."""
    rng = np.random.default_rng(seed)
    aps = ["EDDF", "EDDM", "EHAM", "LEBL", "LSZH", "EGLL", "LEMD", "LIRF", "LFPG", "LTFM"]
    ap = np.repeat(aps, n_per)
    n = len(ap)
    y = rng.normal(600, 200, n)
    pipe = y + rng.normal(0, 100, n)
    st = y + rng.normal(0, 90, n)
    d = pd.DataFrame({"row": np.arange(n), "ap": ap, "y": y, "block": np.tile(np.arange(20), n // 20),
                      "w": 1.0, "PIPE_M": pipe, "stacked_v34lanes": st, "single": pipe + 1.0})
    if shift is not None:
        for a, v in shift.items():
            d.loc[d.ap == a, "stacked_v34lanes"] = d.loc[d.ap == a, "y"] + v
    return d


def test_move_energy_is_the_mean_board_scaled_squared_move_on_finite_comparator_rows():
    """S = sum((cand - comp)^2) / 344,841 over rows where the comparator is finite. Kills: a mean instead of the
    board scaling; NaN comparator rows counted."""
    cand = np.array([10.0, 20.0, 30.0])
    comp = np.array([7.0, 16.0, np.nan])
    assert C.move_energy(cand, comp) == pytest.approx((9 + 16) / N, rel=1e-12)   # exact arithmetic on small ints


def test_airport_gains_add_up_to_the_pooled_gain_and_match_paired_gain():
    """G is SSE-additive: the 5-airport G equals the sum of its airports' G. Kills: a pooled G recomputed as a mean or
    RMSE difference; the wrong comparator column."""
    d = _oof()
    t = C.airport_table(d, "stacked_v34lanes", n_boot=50)
    five = C.pooled(d, "stacked_v34lanes", C.AIRPORTS_5, n_boot=50)
    assert five["G"] == pytest.approx(sum(t[a]["G"] for a in C.AIRPORTS_5), rel=1e-9)
    y, st, pm = d.y.to_numpy(), d.stacked_v34lanes.to_numpy(), d.PIPE_M.to_numpy()
    m = d.ap.isin(C.AIRPORTS_5).to_numpy()
    assert five["G"] == pytest.approx((((pm - y) ** 2 - (st - y) ** 2)[m]).sum() / N, rel=1e-9)
    assert five["S"] == pytest.approx((((st - pm) ** 2)[m]).sum() / N, rel=1e-9)
    assert five["ci95"][0] <= five["G"] <= five["ci95"][1]


def test_align_refuses_frames_with_different_rows_labels_blocks_or_comparator():
    """The paired difference is only valid on identical rows. Kills: comparing runs over different row sets."""
    a = _oof()
    for col, bad in (("row", lambda s: s[::-1].to_numpy()), ("y", lambda s: s + 1.0),
                     ("block", lambda s: (s + 1) % 20), ("PIPE_M", lambda s: s + 1.0)):
        b = a.copy()
        b[col] = bad(b[col])
        with pytest.raises(ValueError, match=col):
            C.align(a, b)
    with pytest.raises(ValueError, match="row count"):
        C.align(a, a.iloc[:-1])


def test_the_verdict_needs_plus_100_and_a_positive_ci_lower_bound():
    """Registered TRUE shape (PREREG v38 H1): paired 5-airport dG >= +100 bmse AND its CI lower bound > 0.
    Kills: '>' for '>=' at the boundary; a verdict ignoring the CI; one ignoring the size."""
    assert C.h1_verdict(100.0, [0.1, 200.0]) == "WORKING"
    assert C.h1_verdict(99.99, [50.0, 150.0]) == "NOT WORKING"
    assert C.h1_verdict(300.0, [0.0, 600.0]) == "NOT WORKING"
    assert C.h1_verdict(float("nan"), [1.0, 2.0]) == "NOT WORKING"


def test_compare_reports_a_known_improvement_at_the_five_airports_and_none_elsewhere():
    """New run = old run with a perfect stack at EDDF only: dG_5 = old EDDF squared error / N, every other airport's
    paired difference is exactly 0. Kills: new/old swapped (sign), per-airport rows mixed up."""
    old = _oof()
    new = _oof(shift={"EDDF": 0.0})
    r = C.compare(old, new, n_boot=50)
    m = (old.ap == "EDDF").to_numpy()
    want = (((old.stacked_v34lanes - old.y) ** 2)[m]).sum() / N
    assert r["stacked_v34lanes"]["paired_new_vs_old"]["five"]["gain_bmse"] == pytest.approx(want, rel=1e-9)
    assert r["stacked_v34lanes"]["paired_new_vs_old"]["airports"]["EGLL"]["gain_bmse"] == pytest.approx(0.0, abs=1e-12)
    assert r["stacked_v34lanes"]["new"]["airports"]["EDDF"]["G"] > r["stacked_v34lanes"]["old"]["airports"]["EDDF"]["G"]
    assert r["h1"]["dG5"] == pytest.approx(want, rel=1e-9)
    assert set(r) >= {"stacked_v34lanes", "single", "h1"}


def test_airport_mask_selects_the_named_airports_and_none_means_all():
    """Mutation: ignore `airports` -> red; None -> no rows -> red."""
    oof = pd.DataFrame({"ap": ["EDDF", "LFPG", "EDDF", "LSZH"]})
    assert C.airport_mask(oof, ("EDDF", "LSZH")).tolist() == [True, False, True, True]
    assert C.airport_mask(oof, None).tolist() == [True] * 4


OLD_SCREEN = ROOT / "data" / "r12" / "stack_801f9bcac9a5_screen" / "oof_r12.parquet"


@pytest.mark.skipif(not OLD_SCREEN.exists(), reason="real screen OOF absent")
def test_real_old_screen_reproduces_the_registered_per_airport_s_and_g():
    """External anchor: PREREG_v35 AMENDMENT A1 recorded the old screen's per-airport [S, G] (EDDF [455, +303], EDDM
    [205, +112], EHAM [568, +338], LEBL [371, +172], LSZH [289, +230]; 5-airport G +1,155, S 1,888; pooled G +1,109.6).
    The comparison arithmetic must reproduce them (to the recorded rounding, +-1 bmse). Run in a child process so the
    334k-row frame never inflates pytest's peak memory."""
    code = ("import importlib.util,sys,json;import pandas as pd;"
            "s=importlib.util.spec_from_file_location('v38_compare',r'%s');m=importlib.util.module_from_spec(s);"
            "sys.modules['v38_compare']=m;s.loader.exec_module(m);d=pd.read_parquet(r'%s');"
            "t=m.airport_table(d,'stacked_v34lanes',n_boot=10);f=m.pooled(d,'stacked_v34lanes',m.AIRPORTS_5,n_boot=10);"
            "a=m.pooled(d,'stacked_v34lanes',None,n_boot=10);"
            "print(json.dumps({'t':{k:[v['S'],v['G']] for k,v in t.items()},'f':[f['S'],f['G']],'a':a['G']}))"
            % (ROOT / "scripts" / "analysis" / "v38_compare.py", OLD_SCREEN))
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True).stdout
    import json
    r = json.loads(out.strip().splitlines()[-1])
    reg = {"EDDF": (455, 303), "EDDM": (205, 112), "EHAM": (568, 338), "LEBL": (371, 172), "LSZH": (289, 230)}
    for ap, (s, g) in reg.items():
        assert r["t"][ap][0] == pytest.approx(s, abs=1.0) and r["t"][ap][1] == pytest.approx(g, abs=1.0), ap
    assert r["f"][0] == pytest.approx(1888, abs=1.5) and r["f"][1] == pytest.approx(1155, abs=1.5)
    assert r["a"] == pytest.approx(1109.6239967403071, abs=1e-6)
