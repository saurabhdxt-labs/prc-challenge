"""Tests for the SHIP-CBV input builder (scripts/analysis/cbv_ship_input.py; plans/PREREG_ship_cbv_2026_09_13.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import cbv_ship_input as I  # noqa: E402


def test_ship_masks_train_everything_and_early_stop_on_months_3_and_9():
    """Fails if a 2025 month is left out of training (the screen held out 1 and 7), if the early-stopping months change, or
    if fit and es overlap."""
    m = np.array([1, 2, 3, 7, 9, 12])
    k = I.ship_masks(m)
    assert k["train"].all() and not k["holdout"].any()
    assert k["es"].tolist() == [False, False, True, False, True, False]
    assert np.array_equal(k["fit"], ~k["es"])


def _rank(n=4):
    num = ["proxy", "sp", "hr"]
    r = pd.DataFrame({"proxy": np.arange(n) + 600.0, "sp": np.arange(n) + 900.0, "hr": np.arange(n, dtype=float),
                      "month": [1, 1, 7, 7], "MVT_ID_mvt": np.arange(n) + 100.0, "ap": "EDDF", "STAND_mvt": ["A1", None, "B2", "C3"],
                      "RUNWAY_mvt": "25C", "ADES_mvt": "EGLL", "FLIGHT_mvt": "DLH1", "CALLSIGN_flt": "DLH1", "airline": "DLH",
                      "AIRCRAFT_OPERATOR_flt": "DLH", "AIRCRAFT_TYPE_mvt": "A320", "MARKET_SEGMENT_flt": "S",
                      "WK_TBL_CAT_flt": "M", "FLIGHT_TYPE_flt": "S", "stand_pref": "A"})
    return r, num


def test_assemble_2026_contract_holdout_only_no_labels_and_offset_rows():
    """Fails if a 2026 row is marked train, carries a target, loses float32, or its `row` collides with the 2025 rows."""
    r, num = _rank()
    o = I.assemble_2026(r, num, row_offset=2_062_440)
    assert o.holdout.all() and not (o.train | o.fit | o.es).any()
    assert np.isnan(o.delta).all() and np.isnan(o.score_y).all()
    assert all(o[c].dtype == np.float32 for c in num)
    assert o.row.tolist() == [2_062_440, 2_062_441, 2_062_442, 2_062_443]
    assert o.ap_stand.tolist()[1] == "EDDF|NA"                         # nulls become the scoped "NA" string, as screened
    assert o.score_proxy.tolist() == [600.0, 601.0, 602.0, 603.0]


def test_assemble_2026_refuses_a_missing_numeric_column():
    r, num = _rank()
    with pytest.raises(ValueError, match="lacks numeric columns"):
        I.assemble_2026(r.drop(columns=["hr"]), num, row_offset=0)


def _combined():
    r, num = _rank()
    cat = I.CN.CAT_COLS
    o = I.assemble_2026(r, num, row_offset=10)
    t = o.copy()
    t["delta"] = 5.0
    t["score_y"] = 1000.0
    m = I.ship_masks(np.array([3, 1, 9, 7]))
    for k, v in m.items():
        t[k] = v
    return pd.concat([t, o], ignore_index=True), num, cat


def test_check_contract_accepts_a_valid_frame_and_refuses_each_violation():
    """Fails if the builder could write a file the worker would train on a 2026 label, or with a broken mask contract."""
    d, num, cat = _combined()
    I.check_contract(d, num, cat)
    bad = d.copy(); bad.loc[len(bad) - 1, "delta"] = 3.0
    with pytest.raises(ValueError, match="carries a label"):
        I.check_contract(bad, num, cat)
    bad = d.copy(); bad.loc[0, "holdout"] = True
    with pytest.raises(ValueError, match="mask contract"):
        I.check_contract(bad, num, cat)
    bad = d.copy(); bad["hr"] = bad["hr"].astype("float64")
    with pytest.raises(ValueError, match="numeric hr"):
        I.check_contract(bad, num, cat)
    bad = d.copy(); bad.loc[0, "delta"] = np.nan
    with pytest.raises(ValueError, match="non-finite target"):
        I.check_contract(bad, num, cat)


def test_main_refuses_to_overwrite(monkeypatch, tmp_path):
    p = tmp_path / "x.parquet"
    p.write_bytes(b"x")
    monkeypatch.setattr(I, "OUT", p)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        I.main()
