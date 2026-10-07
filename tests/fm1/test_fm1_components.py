"""FM1 component functions (prc/pipeline/fm1/components.py): the value formulas equal the shipped ones, the support
gate rules, the ADS-B stack stage contract (with an injected predictor), and the MTX shift (mtx_calib's registered rule)."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import components as C
from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import shipped

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ROOT / "data/r12/stack_801f9bcac9a5_full/models_r12"


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------------------------- value formulas

def test_base_and_staged_values_are_the_stage_contracts_formula():
    """adsb_stage.apply_stage: value = rint(max(proxy - (F + resid), 1)); base = the same with resid 0."""
    r = _rng()
    proxy, F, resid = r.normal(900, 400, 500), r.normal(100, 300, 500), r.normal(0, 30, 500)
    assert np.array_equal(C.staged_value(proxy, F, resid), np.rint(np.maximum(proxy - (F + resid), 1.0)))
    assert np.array_equal(C.base_value(proxy, F, "delta"), C.staged_value(proxy, F, np.zeros(500)))
    assert np.array_equal(C.base_value(proxy, F, "y"), np.rint(np.maximum(F, 1.0)))
    with pytest.raises(E.ConfigValueError, match="target"):
        C.base_value(proxy, F, "log")


def test_blend_is_cbv_ship_builds_formula():
    """v34_compose.matched_values: rint(0.5 * max(proxy - C, 1) + 0.5 * partner_int). Mutation: drop the max(., 1) -> red
    (rows with proxy < C exist below)."""
    r = _rng(1)
    proxy, c = r.normal(900, 400, 500), r.normal(300, 500, 500)
    partner = np.rint(np.maximum(proxy - r.normal(200, 300, 500), 1))
    ref = np.rint(0.5 * np.maximum(proxy - c, 1.0) + 0.5 * partner)
    assert (proxy < c).any()
    assert np.array_equal(C.blend_value(proxy, c, partner, 0.5), ref)
    assert np.array_equal(C.blend_value(proxy, c, partner, 0.0), np.rint(partner))


# ---------------------------------------------------------------------------------------------- the support gate

def test_covered_share():
    ap = np.array(["A", "A", "A", "B", "B"])
    cov = np.array([0, 2, 3, 1, 1])
    assert C.covered_share(ap, cov, 2) == {"A": 2 / 3, "B": 0.0}


@pytest.mark.parametrize("train,serve,expected", [
    (0.60, 0.62, True),       # EDDF-like: similar shares
    (0.88, 0.62, True),       # ratio 0.705 >= 0.7
    (0.90, 0.60, False),      # ratio 0.667 < 0.7: coverage collapsed
    (0.43, 0.90, False),      # EGLL-like: newly covered (ratio 2.1 > 1 / 0.7)
    (0.00, 0.50, False),      # no training support
])
def test_coverage_ratio_gate(train, serve, expected):
    """Mutation: test only the lower bound -> the EGLL-like case turns True -> red."""
    assert C.gate_coverage_ratio({"X": train}, {"X": serve}, 0.7) == {"X": expected}


def test_effective_gate_never_widens_the_manifest():
    man = {"A": True, "B": False}
    got = C.effective_gate("coverage_ratio", man, {"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}, 0.7)
    assert got == {"A": True, "B": False}
    assert C.effective_gate("manifest", man, {}, {}, 0.7) == man
    with pytest.raises(E.ConfigValueError, match="rule"):
        C.effective_gate("ess", man, {}, {}, 0.7)


@pytest.mark.devtree
def test_manifest_gate_of_the_v36_stack():
    """The stored gate v36 served: EDDF/EDDM/EHAM/LEBL/LIRF/LSZH allowed; EGLL/LEMD/LFPG/LTFM not
    (data/r12/serve_801f9bcac9a5_full/serve_r12.json)."""
    import json
    served = json.loads((ROOT / "data/r12/serve_801f9bcac9a5_full/serve_r12.json").read_text())["gate"]
    assert C.gate_manifest(MODELS) == served


# ---------------------------------------------------------------------------------------------- the stack stage

def _frame(n=40):
    r = _rng(2)
    f = pd.DataFrame({"MVT_ID_mvt": np.arange(1.0, n + 1), "ap": np.where(np.arange(n) % 2, "EDDF", "EGLL"),
                      "hr": r.integers(0, 24, n).astype(float), "proxy": r.normal(900, 100, n),
                      "coverage": np.tile([0, 1, 2, 3], n // 4)})
    for c in ("dwell_s", "n_ground"):
        f[c] = r.normal(size=n)
    return f


def test_stack_values_serve_only_covered_gated_rows():
    """Injected predictor (code-design rule 4). Eligible = coverage >= 2 AND the airport gated in; the predictor's
    residual is zeroed elsewhere before the stage contract checks it. Mutation: skip the zeroing -> LaneError -> red."""
    f = _frame()
    F = np.full(len(f), 100.0)
    calls = {}

    def fake(frame, models_dir, oof=False):
        calls["cols"] = list(frame.columns)
        return np.full(len(frame), 10.0)

    vals, elig, resid, info = C.stack_values(f, F, ["dwell_s", "n_ground"], MODELS, {"EDDF": True, "EGLL": False}, 2,
                                             predictor=fake)
    want = (f.ap.to_numpy() == "EDDF") & (f.coverage.to_numpy() >= 2)
    assert np.array_equal(elig, want)
    assert np.array_equal(resid, np.where(want, 10.0, 0.0))
    assert np.array_equal(vals[want], np.rint(np.maximum(f.proxy.to_numpy()[want] - 110.0, 1)))
    assert np.array_equal(vals[~want], np.rint(np.maximum(f.proxy.to_numpy()[~want] - 100.0, 1)))
    assert {"MVT_ID_mvt", "ap", "hour", "proxy", "F", "coverage", "dwell_s", "n_ground"} <= set(calls["cols"])
    assert info["n_eligible"] == int(want.sum())


def test_stack_values_refuse_min_coverage_below_the_predictors():
    f = _frame()
    with pytest.raises(E.ConfigValueError, match="min_coverage 1 < 2"):
        C.stack_values(f, np.zeros(len(f)), ["dwell_s"], MODELS, {"EDDF": True}, 1, predictor=lambda *a, **k: None)


def test_stack_values_surface_a_broken_predictor():
    f = _frame()
    with pytest.raises(E.LaneError, match="non-finite"):
        C.stack_values(f, np.zeros(len(f)), ["dwell_s"], MODELS, {"EDDF": True}, 2,
                       predictor=lambda fr, m, oof=False: np.full(len(fr), np.nan))


# ---------------------------------------------------------------------------------------------- MTX

def test_shift_cells_is_mtx_calib_shift():
    r = _rng(3)
    frame = pd.DataFrame({"ap": r.choice(["EGLL", "LSZH"], 300), "month": r.choice([1, 7], 300),
                          "pred": r.normal(1000, 200, 300)})
    target = {"EGLL|1": 1100.0, "LSZH|7": 900.0, "EDDF|1": 5.0}
    lam = {k: 0.5 for k in target}
    got, rep = C.shift_cells(frame, target, lam, mode="add")
    key = np.char.add(np.char.add(frame.ap.to_numpy().astype(str), "|"), frame.month.astype(str).to_numpy())
    ref, rrep = shipped.mtx_calib().shift(frame.pred.to_numpy(), key, target, lam, mode="add")
    assert np.array_equal(got, ref) and rep == rrep
    assert "EDDF|1" not in rep


def test_mtx_apply_needs_a_fitted_calibration():
    with pytest.raises(E.PipelineError, match="not fitted"):
        C.MtxCalibration().apply(pd.DataFrame({"ap": ["EGLL"], "month": [1], "year": [2026], "pred": [1.0]}),
                                 ROOT / "data/mtx/raw/taxi_out_additional_time_2026.csv", "james_stein", "add")
