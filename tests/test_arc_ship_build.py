"""Tests for SHIP-ARC (scripts/analysis/arc_ship_build.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import arc_ship_build as S  # noqa: E402


def test_arc_values_per_rule_rounding_and_floor():
    """Fails if ord and fill swap formulas, rounding is dropped, or the 1 s floor is removed."""
    v = S.arc_values(["ord", "fill", "ord"], [10_000.0, 0.0, 10_000.0], [9_100.6, 0.0, 10_500.0], [0.0, 6_000.4, 0.0])
    assert v.tolist() == [899.0, 6_000.0, 1.0]


@pytest.mark.parametrize("rule,aobt3", [(["none"], [0.0]), ([""], [0.0]), (["ord"], [np.nan])])
def test_arc_values_refuses_unruled_or_nonfinite(rule, aobt3):
    with pytest.raises(ValueError, match="arc_values"):
        S.arc_values(rule, [1.0], aobt3, [1.0])


def _base():
    return pd.DataFrame({"MVT_ID_mvt": np.array([5.0, 6.0, 7.0, 8.0]),
                         "TAXITIME_SEC_mvt": np.array([100, 200, 300, 400], dtype="int32")})


def test_apply_rows_changes_only_named_ids_and_keeps_order_and_dtype():
    """Fails if values land on the wrong rows (positional instead of id lookup), dtype changes, or the base is mutated."""
    b = _base()
    cand, touched = S.apply_rows(b, [8.0, 6.0], [4_000.0, 200.0])
    assert cand.TAXITIME_SEC_mvt.tolist() == [100, 200, 300, 4_000]
    assert touched.tolist() == [False, False, False, True]            # 6.0 kept its value → not touched
    assert cand.dtypes.equals(b.dtypes) and cand.MVT_ID_mvt.tolist() == [5.0, 6.0, 7.0, 8.0]
    assert b.TAXITIME_SEC_mvt.tolist() == [100, 200, 300, 400]


@pytest.mark.parametrize("ids,msg", [([9.0], "not in the base"), ([5.0, 5.0], "duplicate")])
def test_apply_rows_refusals(ids, msg):
    with pytest.raises(ValueError, match=msg):
        S.apply_rows(_base(), ids, [1.0] * len(ids))
