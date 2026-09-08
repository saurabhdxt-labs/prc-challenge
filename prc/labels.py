"""The label: which departures are admissible, and why each rejected one was.

This module exists because RMSE is decided in the tail. Measured on 2025-01,
taxi-out has a median of 906 seconds and a maximum of 87,177 — a little over 24
hours. A single retained value that large contributes roughly 220 seconds to
the RMSE of a 153,000-row month by itself, which is more than any feature will
ever move. So the rules here matter more than the model, and they are fixed in
`plans/PREREG_taxiout_2026_09_08.md` before any score exists, so that they
cannot be tuned toward one.

Two policy choices are deliberate and worth stating plainly:

* **C3 flags, it does not drop.** Long taxi-outs are genuine at congested
  airports. Silently removing them would be a claim about the world dressed up
  as data cleaning. Callers exclude them via `apply_ceiling`, deliberately, and
  results are reported both ways.
* **A label that disagrees with its timestamps is fatal, not tolerated.** The
  supplied `TAXITIME_SEC_mvt` equalled take-off minus off-block on 100.0000% of
  the 153,706 departures in 2025-01. Every feature is built on that identity
  holding. If it stops holding, the run stops.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Above this many seconds a taxi-out is flagged, never dropped. 180 minutes.
DEFAULT_CEILING_SEC = 10_800

LABEL = "TAXITIME_SEC_mvt"
OFF_BLOCK = "BLOCK_TIME_UTC_mvt"
TAKE_OFF = "MVT_TIME_UTC_mvt"
MOVEMENT_ID = "MVT_ID_mvt"


def _quantiles(values: pd.Series) -> dict:
    v = values.astype("float64")
    return {"min": float(v.min()), "p1": float(v.quantile(0.01)),
            "median": float(v.median()), "mean": float(v.mean()),
            "p99": float(v.quantile(0.99)), "p99_9": float(v.quantile(0.999)),
            "max": float(v.max())}


def check_label_identity(frame: pd.DataFrame, tolerance_sec: float = 0.0) -> None:
    """Raise if the supplied label is not take-off minus off-block.

    Tolerance is zero by default and that is not fussiness: the measured
    agreement was exact on every one of 153,706 rows, so any drift is a change
    in the data's meaning rather than rounding.
    """
    both = frame[TAKE_OFF].notna() & frame[OFF_BLOCK].notna() & frame[LABEL].notna()
    if not both.any():
        return
    derived = (frame.loc[both, TAKE_OFF] - frame.loc[both, OFF_BLOCK]).dt.total_seconds()
    disagreement = (derived - frame.loc[both, LABEL].astype("float64")).abs()
    bad = int((disagreement > tolerance_sec).sum())
    if bad:
        worst = float(disagreement.max())
        raise ValueError(
            f"{bad} of {int(both.sum())} rows disagree with their timestamps "
            f"(worst {worst:.0f}s). The supplied label was exact on 100.0000% of "
            f"2025-01; this changes what the target means, so the run stops here.")


def clean(frame: pd.DataFrame, ceiling_sec: int = DEFAULT_CEILING_SEC) -> tuple[pd.DataFrame, dict]:
    """Apply the pre-registered censoring policy to a movement frame.

    Returns the admissible departures and a report in which every input row is
    accounted for by exactly one outcome.
    """
    report: dict = {"input_rows": int(len(frame)), "ceiling_sec": ceiling_sec,
                    "dropped": {}, "flagged": {}}
    df = frame.copy()

    # Departures only. For an ARR row the airport would be ADES, and its taxi
    # time is taxi-IN, a different quantity entirely.
    is_dep = df["PHASE_mvt"] == "DEP"
    report["dropped"]["not_departure"] = int((~is_dep).sum())
    df = df[is_dep]

    # C2 before C1: a null cannot be compared, and attributing it to C1 would
    # misreport why it went.
    missing = df[LABEL].isna() | df[OFF_BLOCK].isna() | df[TAKE_OFF].isna()
    report["dropped"]["C2_missing"] = int(missing.sum())
    df = df[~missing]

    # The identity check runs on rows that still have both timestamps, before
    # any value-based filtering, so a systematic shift cannot hide behind C1.
    check_label_identity(df)

    # C1: a taxi-out of zero or less is not physical.
    non_positive = df[LABEL] <= 0
    report["dropped"]["C1_non_positive"] = int(non_positive.sum())
    df = df[~non_positive]

    # C5: duplicate movement identifiers. The sort key must fully order the
    # duplicates from the DATA, or a stable sort silently falls back to input
    # order and the same file yields different rows depending on how it was
    # read. Off-block and identifier are equal by construction among duplicates,
    # so take-off and the label are what actually break the tie.
    df = df.sort_values([OFF_BLOCK, TAKE_OFF, LABEL, MOVEMENT_ID], kind="mergesort")
    duplicated = df[MOVEMENT_ID].duplicated(keep="first")
    report["dropped"]["C5_duplicate_id"] = int(duplicated.sum())
    df = df[~duplicated]

    # C3: flag the tail, keep it. `apply_ceiling` is how a caller excludes it.
    df = df.assign(above_ceiling=df[LABEL] > ceiling_sec)
    report["flagged"]["C3_above_ceiling"] = int(df.above_ceiling.sum())

    report["kept_rows"] = int(len(df))
    report["distribution"] = _quantiles(df[LABEL]) if len(df) else {}
    accounted = report["kept_rows"] + sum(report["dropped"].values())
    if accounted != report["input_rows"]:                  # a guard on the report itself
        raise AssertionError(f"row accounting is wrong: {accounted} != {report['input_rows']}")
    return df.reset_index(drop=True), report


def apply_ceiling(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows C3 flagged. Deliberate, never implicit."""
    return frame[~frame["above_ceiling"]].reset_index(drop=True)
