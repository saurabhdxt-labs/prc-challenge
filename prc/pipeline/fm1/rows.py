"""FM1's input contract (AT2): the raw rows `fit` and `predict` accept, and the columns no FM1 design may contain.

Raw rows are the organiser's movement files (one row per DEP or ARR movement, 30 columns). Arrival rows legitimately carry
BLOCK/TAXITIME (their taxi-in feeds the stand and queue features); a SCORED departure never does, and no design column
may be a label, a label derivative or a label-seconds column (`*_sec`, the NSC leak, FMA §1.3 #16).

Every failure raises a named error with the offending column / count; nothing is coerced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import errors as E

#: column -> family ("float" id/number, "int" label, "str", "ts" = tz-aware UTC timestamp)
RAW_COLUMNS = {
    "MVT_ID_mvt": "float", "FLIGHT_ID_mvt": "float", "FLIGHT_mvt": "str", "FLIGHT_RULE_mvt": "str", "ADEP_mvt": "str",
    "ADES_mvt": "str", "PHASE_mvt": "str", "MVT_TIME_UTC_mvt": "ts", "BLOCK_TIME_UTC_mvt": "ts",
    "SCHED_TIME_UTC_mvt": "ts", "AIRCRAFT_TYPE_mvt": "str", "RUNWAY_mvt": "str", "STAND_mvt": "str",
    "TAXITIME_SEC_mvt": "int", "LOBT_flt": "ts", "CALLSIGN_flt": "str", "ADEP_flt": "str", "ADES_flt": "str",
    "ADES_FILED_flt": "str", "MARKET_SEGMENT_flt": "str", "IOBT_flt": "ts", "FLIGHT_RULE_flt": "str",
    "FLIGHT_TYPE_flt": "str", "AIRCRAFT_TYPE_flt": "str", "WK_TBL_CAT_flt": "str", "AIRCRAFT_OPERATOR_flt": "str",
    "EOBT_1_flt": "ts", "ARVT_1_flt": "ts", "AOBT_3_flt": "ts", "ARVT_3_flt": "ts",
}
LABEL_COLUMNS = ("TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt")
PHASES = ("DEP", "ARR")
MODES = ("train", "serve")
#: names no design may contain: the labels and their direct derivatives
FORBIDDEN_EXACT = frozenset({"TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "y", "delta", "taxi", "label", "target"})
#: suffixes no design may contain: label-seconds columns (aobt_sec, mvt_sec, sched_sec; nsc_fold.SECONDS_COLS)
FORBIDDEN_SUFFIXES = ("_sec",)
#: informational only (BC-28): production (stand_ab.py:236/589/1268) keeps every training row with
#: TAXITIME_SEC_mvt > 0 and BLOCK not null, with NO upper bound -- real 2025 date-slip labels (BLOCK on the
#: previous UTC day's clock, mostly LIRF/Rome) legitimately exceed it (max observed 131,167 s). The contract must
#: not refuse a real label the production filter would keep; `validate_raw` reports the count, it does not raise.
DAY_S = 86_400


def forbidden(columns) -> list:
    """The columns of `columns` that no FM1 design may contain, in input order."""
    return [c for c in columns if c in FORBIDDEN_EXACT or any(str(c).endswith(s) for s in FORBIDDEN_SUFFIXES)]


def assert_design_clean(columns, where: str = "design") -> None:
    bad = forbidden(columns)
    if bad:
        raise E.ForbiddenInputError(f"{where}: forbidden input column(s) {bad} (labels, label derivatives and `*_sec` "
                                    "label-seconds columns never enter an FM1 design)")


def _family_ok(s: pd.Series, family: str) -> bool:
    if family == "ts":
        return isinstance(s.dtype, pd.DatetimeTZDtype) and str(s.dtype.tz) == "UTC"
    if family == "str":
        return pd.api.types.is_string_dtype(s.dtype) or s.dtype == object
    if family in ("float", "int"):
        return pd.api.types.is_numeric_dtype(s.dtype) and not pd.api.types.is_bool_dtype(s.dtype)
    raise ValueError(family)                                                       # pragma: no cover


def validate_raw(frame: pd.DataFrame, mode: str, airports, where: str = "rows") -> dict:
    """Check raw movement rows against the contract. mode 'train' (labelled months) or 'serve' (a scored file: every
    DEP row's labels must be null). Returns a small report; raises a named SchemaError / LabelLeakError otherwise."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not isinstance(frame, pd.DataFrame):
        raise E.SchemaError(f"{where}: expected a pandas DataFrame, got {type(frame).__name__}")
    missing = [c for c in RAW_COLUMNS if c not in frame.columns]
    if missing:
        raise E.MissingColumnError(f"{where}: missing column(s) {missing}")
    extra = [c for c in frame.columns if c not in RAW_COLUMNS]
    if extra:
        raise E.SchemaError(f"{where}: unexpected column(s) {extra}; FM1 reads the organiser's 30 raw columns only")
    for c, fam in RAW_COLUMNS.items():
        if not _family_ok(frame[c], fam):
            want = "a tz-aware UTC timestamp" if fam == "ts" else fam
            raise E.DtypeError(f"{where}.{c}: dtype {frame[c].dtype} is not {want}")
    if len(frame) == 0:
        raise E.SchemaError(f"{where}: no rows")
    ids = frame.MVT_ID_mvt
    if ids.isna().any():
        raise E.SchemaError(f"{where}.MVT_ID_mvt: {int(ids.isna().sum())} null id(s)")
    if not ids.is_unique:
        raise E.DuplicateIdError(f"{where}.MVT_ID_mvt: {int(ids.duplicated().sum())} duplicate id(s)")
    phase = frame.PHASE_mvt.astype(str)
    bad_phase = sorted(set(phase.unique()) - set(PHASES))
    if bad_phase:
        raise E.SchemaError(f"{where}.PHASE_mvt: unknown phase(s) {bad_phase}")
    dep = (phase == "DEP").to_numpy()
    ap = frame.ADEP_mvt.astype(str).to_numpy()
    reg = set(airports)
    unknown = sorted(set(ap[dep]) - reg)
    if unknown:
        raise E.UnknownAirportError(f"{where}: departures from airport(s) {unknown} outside the registry "
                                    f"(airports.codes); add them to the config")
    if frame.MVT_TIME_UTC_mvt[dep].isna().any():
        raise E.RangeError(f"{where}: {int(frame.MVT_TIME_UTC_mvt[dep].isna().sum())} departure(s) without MVT_TIME "
                           "(cannot be routed or dated)")
    taxi = frame.TAXITIME_SEC_mvt.to_numpy(dtype="float64", na_value=np.nan)
    n_beyond_a_day = 0
    if mode == "serve":
        leaked = dep & (np.isfinite(taxi) | frame.BLOCK_TIME_UTC_mvt.notna().to_numpy())
        if leaked.any():
            raise E.LabelLeakError(f"{where}: {int(leaked.sum())} scored departure(s) carry a label "
                                   "(TAXITIME_SEC_mvt / BLOCK_TIME_UTC_mvt); a scored file never does")
    else:
        # information only (BC-28): production keeps every TAXITIME_SEC_mvt > 0 & BLOCK-not-null training row with
        # no upper bound, so a real date-slip label above a day is not refused here -- see DAY_S above.
        n_beyond_a_day = int((np.isfinite(taxi) & (np.abs(taxi) > DAY_S)).sum())
    return {"n_rows": int(len(frame)), "n_dep": int(dep.sum()), "n_arr": int((~dep).sum()),
            "n_dep_matched": int((dep & frame.AOBT_3_flt.notna().to_numpy()).sum()),
            "n_dep_unmatched": int((dep & frame.AOBT_3_flt.isna().to_numpy()).sum()), "mode": mode,
            "n_labels_beyond_a_day": n_beyond_a_day}


def hide_labels(frame: pd.DataFrame) -> tuple:
    """(serve-shaped copy with every DEP row's labels nulled, the hidden labels by MVT id). For validation protocols:
    the model predicts from the copy; the labels are read back only by the scorer."""
    dep = frame.PHASE_mvt.astype(str).to_numpy() == "DEP"
    labels = pd.Series(frame.TAXITIME_SEC_mvt.to_numpy(dtype="float64", na_value=np.nan)[dep],
                       index=frame.MVT_ID_mvt.to_numpy(dtype="float64")[dep], name="y")
    out = frame.copy()
    out.loc[dep, "TAXITIME_SEC_mvt"] = pd.NA
    out.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    return out, labels
