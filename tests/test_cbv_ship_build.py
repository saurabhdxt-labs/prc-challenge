"""Tests for the SHIP-CBV build (scripts/analysis/cbv_ship_build.py; plans/PREREG_ship_cbv_2026_09_13.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import cbv_ship_build as S  # noqa: E402


def test_taxi_orientation_and_floor():
    """BC-2: taxi = max(proxy − delta, 1). Fails if the sign flips or the floor is dropped."""
    assert S.taxi([900.0, 100.0], [300.0, 500.0]).tolist() == [600.0, 1.0]


def test_blend_values_half_weight_rounded():
    """Fails if the weight is not 0.5, if v13 and C are swapped in a way that changes the value, or rounding is skipped."""
    assert S.blend_values([601.0, 100.0], [400.0, 1.0]).tolist() == [500.0, 50.0]   # rint(500.5)=500 banker's; rint(50.5)=50
    assert S.blend_values([700.0], [300.0]).tolist() == [500.0]
    with pytest.raises(AssertionError, match="S5"):
        S.blend_values([np.nan], [300.0])


def _ship(ids, proxy, holdout):
    return pd.DataFrame({"MVT_ID_mvt": ids, "score_proxy": proxy, "holdout": holdout})


def test_holdout_predictions_align_to_holdout_rows_in_input_order():
    """The worker writes one prediction per holdout row, in input order. Fails if training rows are included, or order is lost."""
    ship = _ship([1.0, 2.0, 3.0, 4.0], [1000.0, 900.0, 800.0, 700.0], [False, True, False, True])
    h = S.holdout_predictions(ship, [100.0, 50.0])
    assert h.MVT_ID_mvt.tolist() == [2.0, 4.0] and h.taxi_C.tolist() == [800.0, 650.0]
    with pytest.raises(AssertionError, match="S4: 3 predictions for 2"):
        S.holdout_predictions(ship, [1.0, 2.0, 3.0])
    with pytest.raises(AssertionError, match="non-finite"):
        S.holdout_predictions(ship, [1.0, np.inf])


def _prov():
    stages = ["lightgbm:delta:mean3"] * 3 + ["lightgbm:delta:mean3+adsb"] * 2 + ["S1C:body"]
    return pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "airport": "EDDF", "stage": stages})


def test_scope_frame_takes_only_the_no_adsb_stage_and_checks_the_matched_id_set(monkeypatch):
    """Fails if staged (+adsb) rows enter the scope, or if the holdout id set is not exactly the matched lane."""
    monkeypatch.setattr(S, "STAGE_COUNTS", {"lightgbm:delta:mean3": 3, "lightgbm:delta:mean3+adsb": 2})
    v13 = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "TAXITIME_SEC_mvt": np.array([10, 20, 30, 40, 50, 60], "int32")})
    hold = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0], "proxy": 0.0, "C": 0.0, "taxi_C": [11.0, 21.0, 31.0, 41.0, 51.0]})
    s = S.scope_frame(_prov(), v13, hold)
    assert s.MVT_ID_mvt.tolist() == [1.0, 2.0, 3.0] and s.v13.tolist() == [10, 20, 30]
    with pytest.raises(AssertionError, match="S3"):
        S.scope_frame(_prov(), v13, hold.iloc[:4])
    monkeypatch.setattr(S, "STAGE_COUNTS", {"lightgbm:delta:mean3": 4, "lightgbm:delta:mean3+adsb": 2})
    with pytest.raises(AssertionError, match="S1"):
        S.scope_frame(_prov(), v13, hold)


def test_transfer_checks_bands_and_minimum_rows():
    """T1 sd ratio in [0.7, 1.3]; T2 |Δmean| ≤ 30 s; T3 share of |rint(0.5·d)| > 60 s within ±10 pp; airports under 500 rows
    are reported only. Fails if a band is loosened or the halving of d for the value change is dropped."""
    rng = np.random.default_rng(0)
    fold = pd.DataFrame({"ap": ["A"] * 1000 + ["B"] * 100, "d": np.concatenate([rng.normal(0, 100, 1000), rng.normal(0, 100, 100)])})
    same = pd.DataFrame({"ap": ["A"] * 1000, "d": rng.normal(0, 100, 1000)})
    wide = pd.DataFrame({"ap": ["A"] * 1000, "d": rng.normal(50, 200, 1000)})
    ok, bad = S.transfer_checks(fold, same), S.transfer_checks(fold, wide)
    assert ok["A"]["T1"] and ok["A"]["T2"] and ok["A"]["T3"]
    assert not bad["A"]["T1"] and not bad["A"]["T2"] and not bad["A"]["T3"]
    assert ok["B"].get("reported_only") is True
    assert ok["A"]["share_gt60_fold_pct"] == pytest.approx(float((np.abs(np.rint(0.5 * fold.d[:1000])) > 60).mean() * 100), abs=1e-9)
