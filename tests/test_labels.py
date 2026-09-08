"""The censoring policy, fixed before any cross-validation number exists.

RMSE is dominated by its tail. In 2025-01 the taxi-out maximum is 87,177
seconds against a median of 906, and one retained value like that contributes
roughly 220 seconds to the RMSE of a 153,000-row month on its own. So which
rows survive is a larger decision than any feature, and it is made here, in
advance, against rules written in the pre-registration rather than against a
score.

Mutation rehearsals at write time, each restored immediately:
  * C1 `<= 0` -> `< 0`            -> test_c1 RED (a zero-second taxi survives)
  * C3 drop instead of flag        -> test_c3_flags_but_keeps RED
  * duplicate keep-last not -first -> test_c5 RED
  * label check tolerance widened  -> test_label_disagreement_is_fatal RED
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import labels  # noqa: E402

T0 = pd.Timestamp("2025-01-01 06:00:00", tz="UTC")


def movements(rows: list[dict]) -> pd.DataFrame:
    """Build a movement frame with the real column names and sane defaults."""
    base = dict(PHASE_mvt="DEP", ADEP_mvt="EDDF", ADES_mvt="EGLL",
                BLOCK_TIME_UTC_mvt=T0, RUNWAY_mvt="25C", STAND_mvt="A12")
    out = []
    for i, r in enumerate(rows):
        d = dict(base, MVT_ID_mvt=float(2025_00000 + i))
        d.update(r)
        taxi = d.get("TAXITIME_SEC_mvt")
        if "MVT_TIME_UTC_mvt" not in d:
            d["MVT_TIME_UTC_mvt"] = (pd.NaT if taxi is None or pd.isna(taxi)
                                     else d["BLOCK_TIME_UTC_mvt"] + pd.Timedelta(seconds=float(taxi)))
        out.append(d)
    return pd.DataFrame(out)


def test_only_departures_are_kept():
    df = movements([{"TAXITIME_SEC_mvt": 900}, {"TAXITIME_SEC_mvt": 800, "PHASE_mvt": "ARR"}])
    kept, rep = labels.clean(df)
    assert len(kept) == 1 and (kept.PHASE_mvt == "DEP").all()
    assert rep["input_rows"] == 2 and rep["dropped"]["not_departure"] == 1


def test_c1_non_positive_taxi_is_dropped_including_exactly_zero():
    df = movements([{"TAXITIME_SEC_mvt": 900}, {"TAXITIME_SEC_mvt": 0},
                    {"TAXITIME_SEC_mvt": -12}])
    kept, rep = labels.clean(df)
    assert rep["dropped"]["C1_non_positive"] == 2, "zero seconds is not a taxi-out"
    assert kept.TAXITIME_SEC_mvt.tolist() == [900]


def test_c2_missing_label_or_timestamp_is_dropped():
    df = movements([{"TAXITIME_SEC_mvt": 900}, {"TAXITIME_SEC_mvt": np.nan},
                    {"TAXITIME_SEC_mvt": 700}])
    df.loc[2, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    kept, rep = labels.clean(df)
    assert rep["dropped"]["C2_missing"] == 2 and len(kept) == 1


def test_c3_flags_the_long_tail_but_keeps_it():
    """C3 must not silently drop. Long taxi-outs are real at congested airports;
    the ceiling is a judgement and has to stay visible in both directions."""
    df = movements([{"TAXITIME_SEC_mvt": 900}, {"TAXITIME_SEC_mvt": 87_177}])
    kept, rep = labels.clean(df, ceiling_sec=10_800)
    assert len(kept) == 2, "C3 flags, it does not drop"
    assert kept.above_ceiling.tolist() == [False, True]
    assert rep["flagged"]["C3_above_ceiling"] == 1
    below = labels.apply_ceiling(kept)
    assert len(below) == 1, "callers can exclude, but only deliberately"


def test_c5_duplicate_movement_ids_resolve_deterministically():
    df = movements([{"TAXITIME_SEC_mvt": 900}, {"TAXITIME_SEC_mvt": 950}])
    df.loc[1, "MVT_ID_mvt"] = df.loc[0, "MVT_ID_mvt"]
    kept, rep = labels.clean(df)
    assert rep["dropped"]["C5_duplicate_id"] == 1 and len(kept) == 1
    assert kept.TAXITIME_SEC_mvt.iloc[0] == 900, "first occurrence in time order wins"
    # and the rule is stable under input order
    kept2, _ = labels.clean(df.iloc[::-1].reset_index(drop=True))
    assert kept2.TAXITIME_SEC_mvt.iloc[0] == 900


def test_label_disagreement_with_the_timestamps_is_fatal():
    """The supplied label equalled take-off minus off-block on 100.0000% of
    2025-01 departures. If that ever stops being true the assumption behind
    every feature has changed, and it must stop the run, not be tolerated."""
    df = movements([{"TAXITIME_SEC_mvt": 900}])
    df.loc[0, "MVT_TIME_UTC_mvt"] += pd.Timedelta(seconds=5)
    with pytest.raises(ValueError, match="disagree"):
        labels.clean(df)


def test_report_accounts_for_every_input_row():
    df = movements([{"TAXITIME_SEC_mvt": 900}, {"TAXITIME_SEC_mvt": 0},
                    {"TAXITIME_SEC_mvt": 800, "PHASE_mvt": "ARR"},
                    {"TAXITIME_SEC_mvt": 87_177}])
    kept, rep = labels.clean(df, ceiling_sec=10_800)
    assert rep["input_rows"] == 4
    assert rep["kept_rows"] == len(kept) == 2
    assert rep["input_rows"] - sum(rep["dropped"].values()) == rep["kept_rows"], (
        "every dropped row is attributed to exactly one rule")
    q = rep["distribution"]
    assert q["median"] == pytest.approx(44038.5) and q["max"] == 87177
    assert set(q) >= {"min", "p1", "median", "p99", "p99_9", "max", "mean"}


def test_clean_never_mutates_its_input():
    df = movements([{"TAXITIME_SEC_mvt": 900}, {"TAXITIME_SEC_mvt": 0}])
    before = df.copy()
    labels.clean(df)
    pd.testing.assert_frame_equal(df, before)
