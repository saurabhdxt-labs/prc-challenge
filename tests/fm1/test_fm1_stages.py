"""The stage plan and the value ledger (prc/pipeline/fm1/stages.py): AT5 (provenance) and AT6 (calibration last, the v43
EGLL-7 defect class: BLK re-valued EGLL-July AFTER MTX had computed its shift on the earlier values, FMA §0.2 / G8)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import stages as S


def _cfg(**flat):
    base = {"modules.arc.enabled": False, "modules.rome.enabled": False, **flat}
    return K.parse_config(K.nest(base))


# ---------------------------------------------------------------------------------------------- the plan

def test_plan_order_is_fixed_and_calibration_is_last():
    """Mutation: move MTX before unmatched_body in STAGE_ORDER -> check_order raises at import -> red everywhere."""
    names = [s.name for s in S.plan(_cfg(**{"run.purpose": "scored_period"}))]
    assert names == ["input_contract", "features", "base_model", "adsb_gate", "adsb_stack", "cbv_blend", "unmatched_body",
                     "mtx", "finalise"]
    kinds = [s.kind for s in S.STAGE_ORDER]
    last_value = max(i for i, k in enumerate(kinds) if k == "value")
    assert all(k in ("format",) for k in kinds[last_value + 1:] if k != "calibration")
    assert kinds.index("calibration") > last_value


def test_plan_follows_the_switches():
    names = [s.name for s in S.plan(_cfg(**{"adsb.stack.enabled": False, "matched.blend.catboost_weight": 0.0,
                                            "lanes.unmatched": False}))]
    assert names == ["input_contract", "features", "base_model", "finalise"]
    names = [s.name for s in S.plan(_cfg(**{"lanes.matched": False}))]
    assert names == ["input_contract", "features", "unmatched_body", "finalise"]


def test_plan_refuses_unported_components():
    """plans/PREREG_fm1_2026_09_28.md §2 + Amendment A3: Rome, ARC and the storm anchor are ported (the default fm1.yaml
    plans), LDI too (owner decision 2026-09-29), adsb.fi too (AFI, 2026-10-03); the storm-as-feature form is not, and is
    refused by name while switched on."""
    names = [s.name for s in S.plan(K.parse_config({}))]
    assert names[names.index("unmatched_body"):] == ["unmatched_body", "rome", "arc", "rome_e2", "rome_e2w", "finalise"]
    with pytest.raises(E.ComponentNotPortedError, match=r"not ported to FM1: modules\.storm\.feature \("):
        S.plan(_cfg(**{"modules.storm.feature": True, "adsb.sources.adsb_fi": True}))
    assert "storm" in [s.name for s in S.plan(_cfg(**{"modules.storm.mode": "anchor"}))]
    assert "ldi" in [s.name for s in S.plan(_cfg(**{"modules.ldi.enabled": True}))]


def test_check_order_refuses_a_value_stage_after_calibration():
    """The structural half of AT6. Mutation: make check_order return early -> red."""
    order = list(S.STAGE_ORDER)
    mtx = next(s for s in order if s.name == "mtx")
    body = next(s for s in order if s.name == "unmatched_body")
    bad = [s for s in order if s is not mtx]
    bad.insert(bad.index(body), mtx)
    with pytest.raises(E.StageOrderError, match="value stage 'unmatched_body' after calibration stage 'mtx'"):
        S.check_order(bad)
    with pytest.raises(E.StageOrderError, match="not in the fixed order"):
        S.check_order(list(reversed(order)))
    S.check_order(order)


# ---------------------------------------------------------------------------------------------- the ledger

def _stage(name):
    return next(s for s in S.STAGE_ORDER if s.name == name)


def test_ledger_records_the_last_value_writer():
    """AT5. Mutation: keep the first writer's decided_by instead of the last -> red."""
    L = S.Ledger(np.array([1.0, 2.0, 3.0]))
    L.write(_stage("base_model"), np.array([1.0, 2.0]), np.array([100.0, 200.0]), label="base_lightgbm")
    L.write(_stage("adsb_stack"), np.array([2.0]), np.array([190.0]))
    L.write(_stage("unmatched_body"), np.array([3.0]), np.array([300.0]))
    out = L.finalise()
    assert out.MVT_ID_mvt.tolist() == [1.0, 2.0, 3.0]
    assert out.TAXITIME_SEC_mvt.tolist() == [100, 190, 300] and out.TAXITIME_SEC_mvt.dtype == np.int32
    assert out.decided_by.tolist() == ["base_lightgbm", "adsb_stack", "unmatched_body"]
    assert set(out.decided_by) <= S.DECIDED_BY
    assert L.counts() == {"adsb_stack": 1, "base_lightgbm": 1, "unmatched_body": 1}


def test_ledger_refuses_a_value_write_after_calibration():
    """The runtime half of AT6 (the v43 EGLL-7 class). Mutation: drop the `calibrated` check in write() -> red."""
    L = S.Ledger(np.array([1.0, 2.0]))
    L.write(_stage("base_model"), np.array([1.0, 2.0]), np.array([100.0, 200.0]), label="base_lightgbm")
    L.calibrate(_stage("mtx"), np.array([1.0, 2.0]), np.array([90.0, 210.0]))
    with pytest.raises(E.StageOrderError, match="'adsb_stack' wrote a value after calibration 'mtx'"):
        L.write(_stage("adsb_stack"), np.array([1.0]), np.array([95.0]))
    with pytest.raises(E.StageOrderError, match="calibrated twice"):
        L.calibrate(_stage("mtx"), np.array([1.0]), np.array([91.0]))


def test_calibration_marks_rows_and_keeps_the_pre_calibration_value():
    L = S.Ledger(np.array([1.0, 2.0, 3.0]))
    L.write(_stage("base_model"), np.array([1.0, 2.0, 3.0]), np.array([100.0, 200.0, 300.0]), label="base_lightgbm")
    L.calibrate(_stage("mtx"), np.array([1.0, 2.0]), np.array([90.0, 200.0]))
    out = L.finalise()
    assert out.decided_by.tolist() == ["base_lightgbm+mtx", "base_lightgbm", "base_lightgbm"]
    assert out.value_before_calibration.tolist() == [100.0, 200.0, 300.0]
    assert out.calibration_shift.tolist() == [-10.0, 0.0, 0.0]


def test_ledger_rejects_bad_writes():
    L = S.Ledger(np.array([1.0, 2.0]))
    with pytest.raises(E.StageOrderError, match="'features' is not a value stage"):
        L.write(_stage("features"), np.array([1.0]), np.array([1.0]))
    with pytest.raises(E.RoutingError, match="1 id\\(s\\) not in the ledger"):
        L.write(_stage("base_model"), np.array([9.0]), np.array([1.0]), label="base_lightgbm")
    with pytest.raises(E.LaneError, match="non-finite"):
        L.write(_stage("base_model"), np.array([1.0]), np.array([np.nan]), label="base_lightgbm")
    with pytest.raises(E.LaneError, match="2 values for 1 ids"):
        L.write(_stage("base_model"), np.array([1.0]), np.array([1.0, 2.0]), label="base_lightgbm")
    with pytest.raises(E.StageOrderError, match="label 'base_xgboost' is not in the decided_by vocabulary"):
        L.write(_stage("base_model"), np.array([1.0]), np.array([1.0]), label="base_xgboost")
    with pytest.raises(E.RoutingError, match="duplicate ids"):
        S.Ledger(np.array([1.0, 1.0]))


def test_finalise_refuses_unserved_rows_and_floors_at_one():
    L = S.Ledger(np.array([1.0, 2.0]))
    L.write(_stage("base_model"), np.array([1.0]), np.array([0.2]), label="base_lightgbm")
    with pytest.raises(E.RoutingError, match="1 row\\(s\\) no stage served"):
        L.finalise()
    L.write(_stage("unmatched_body"), np.array([2.0]), np.array([-5.0]))
    out = L.finalise()
    assert out.TAXITIME_SEC_mvt.tolist() == [1, 1]


def test_designed_mean_after_mtx_matches_within_1e_9():
    """AT6 numerical half: after calibration each calibrated cell's mean equals mean_before + lambda * gap (the value MTX
    designed); no later stage can move it (the ledger refuses). With BLK's order (re-value after MTX) the EGLL-7 mean
    missed by -18.85 s; here the attempt raises instead."""
    from prc.pipeline.fm1 import components as C
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({"ap": np.where(np.arange(n) < 200, "EGLL", "LSZH"), "month": 7,
                          "pred": rng.normal(1300, 200, n).round()})
    target = {"EGLL|7": 1345.2, "LSZH|7": 1100.0}
    lam = {"EGLL|7": 0.8, "LSZH|7": 1.0}
    new, rep = C.shift_cells(frame, target, lam, mode="add")
    for k, r in rep.items():
        ap = k.split("|")[0]
        sel = frame.ap.to_numpy() == ap
        designed = r["mean_before"] + r["lam"] * r["gap"]
        assert abs(new[sel].mean() - designed) < 1e-9          # fp64 mean of <= 200 values of magnitude ~1e3
    L = S.Ledger(np.arange(n, dtype=float))
    L.write(_stage("base_model"), np.arange(n, dtype=float), frame.pred.to_numpy(), label="base_lightgbm")
    L.calibrate(_stage("mtx"), np.arange(n, dtype=float), new)
    with pytest.raises(E.StageOrderError):
        L.write(_stage("adsb_stack"), np.arange(200, dtype=float), frame.pred.to_numpy()[:200] - 31.0)
