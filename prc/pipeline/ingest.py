"""Raw parquet -> canonical frames, behind declarative schemas that fail loud.

Three rules, each tested:
  1. every file is schema-checked from its parquet METADATA before a row is read (columns present, arrow
     type of the right kind); every frame is checked again after it is built (dtype kind, nullability);
  2. the scored file is read through SERVE_COLUMNS, which EXCLUDES the label columns -- a departure's
     BLOCK_TIME_UTC_mvt / TAXITIME_SEC_mvt are never requested for the scored rows, so no feature can read
     them (they are 100% null there; a feature that read them would be silently dead in production --
     tests/test_serve_time_contract.py's defect class). A scored frame that carries a label column is
     refused (LabelLeakError);
  3. a departure at an airport the config does not carry is refused (UnknownAirportError), never routed
     to a default.

The canonical scored frame is `build_submission.derive()` of the scored departures -- the exact frame every
ship script built -- restricted to the template's ids in the scored file's order.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import errors as E
from . import legacy

LABEL_COLUMNS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")


@dataclass(frozen=True)
class Col:
    """One column's contract: its name, the KIND of its type, and whether nulls are allowed."""
    name: str
    kind: str                      # string | float | int | number | timestamp | bool
    nullable: bool = True


KINDS = ("string", "float", "int", "number", "timestamp", "bool")

#: build_submission.COLUMNS with their kinds -- the raw movement columns every loader reads. Spelled out
#: (not imported) so this contract is independent of the code it guards; a test asserts the name list
#: equals build_submission.COLUMNS, so the two cannot drift.
RAW_MOVEMENT = (
    Col("PHASE_mvt", "string", nullable=False), Col("MVT_ID_mvt", "float", nullable=False),
    Col("ADEP_mvt", "string"), Col("ADES_mvt", "string"), Col("STAND_mvt", "string"), Col("RUNWAY_mvt", "string"),
    Col("AIRCRAFT_TYPE_mvt", "string"), Col("FLIGHT_mvt", "string"), Col("FLIGHT_RULE_mvt", "string"),
    Col("MVT_TIME_UTC_mvt", "timestamp"), Col("SCHED_TIME_UTC_mvt", "timestamp"),
    Col("BLOCK_TIME_UTC_mvt", "timestamp"), Col("TAXITIME_SEC_mvt", "number"), Col("AOBT_3_flt", "timestamp"),
    Col("EOBT_1_flt", "timestamp"), Col("LOBT_flt", "timestamp"), Col("IOBT_flt", "timestamp"),
    Col("AIRCRAFT_OPERATOR_flt", "string"), Col("MARKET_SEGMENT_flt", "string"), Col("WK_TBL_CAT_flt", "string"),
    Col("FLIGHT_TYPE_flt", "string"),
)
RAW_BY_NAME = {c.name: c for c in RAW_MOVEMENT}
#: what the scored file is read through: the raw columns minus the labels
SERVE_COLUMNS = tuple(c.name for c in RAW_MOVEMENT if c.name not in LABEL_COLUMNS)
#: on a scored departure these must be populated: derive() computes sp from SCHED and routes on AOBT_3
SCORED_NOT_NULL = ("MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
TEMPLATE = (Col("MVT_ID_mvt", "float", nullable=False), Col("TAXITIME_SEC_mvt", "number"))
#: the 2025 witness reads only these (training rows: the labels ARE read here, for admissibility)
TRAINING_CLOCK_COLUMNS = ("PHASE_mvt", "MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
                          "TAXITIME_SEC_mvt", "AOBT_3_flt")
#: the unmatched-stratum frames (data/cache_rome, stratum_fold.load_real's recipe): what fit_unmatched,
#: rome_fill, rome_dateslip, the non-fill bodies and the witness attach read
ROME_FRAME = (
    Col("MVT_ID_mvt", "float", nullable=False), Col("ADEP_mvt", "string", nullable=False),
    Col("month", "int", nullable=False), Col("y", "float", nullable=False), Col("sp", "float", nullable=False),
    Col("dayoff", "int", nullable=False), Col("hr", "int", nullable=False), Col("tmin", "int", nullable=False),
    Col("dow", "int", nullable=False), Col("MVT_TIME_UTC_mvt", "timestamp", nullable=False),
    Col("SCHED_TIME_UTC_mvt", "timestamp", nullable=False), Col("BLOCK_TIME_UTC_mvt", "timestamp", nullable=False),
    Col("AOBT_3_flt", "timestamp"), Col("airline", "string"), Col("stand_c1", "string"), Col("ades_p2", "string"),
    Col("stand_pref", "string"), Col("STAND_mvt", "string"), Col("ADES_mvt", "string"), Col("RUNWAY_mvt", "string"),
    Col("FLIGHT_RULE_mvt", "string"), Col("AIRCRAFT_OPERATOR_flt", "string"), Col("unmatched", "bool", nullable=False),
)


# =================================================================================================
# schema checks
# =================================================================================================

def _arrow_kind_ok(t: pa.DataType, kind: str) -> bool:
    if kind == "string":
        return pa.types.is_string(t) or pa.types.is_large_string(t)
    if kind == "float":
        return pa.types.is_floating(t)
    if kind == "int":
        return pa.types.is_integer(t)
    if kind == "number":
        return pa.types.is_integer(t) or pa.types.is_floating(t)
    if kind == "timestamp":
        return pa.types.is_timestamp(t) and t.tz in ("UTC", "+00:00", "Etc/UTC")
    if kind == "bool":
        return pa.types.is_boolean(t)
    raise ValueError(f"unknown kind {kind!r}; one of {KINDS}")


def _frame_kind_ok(s: pd.Series, kind: str) -> bool:
    t = s.dtype
    if kind == "string":
        return pd.api.types.is_string_dtype(t) or t == object
    if kind == "float":
        return pd.api.types.is_float_dtype(t)
    if kind == "int":
        return pd.api.types.is_integer_dtype(t) and not pd.api.types.is_bool_dtype(t)
    if kind == "number":
        return pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_bool_dtype(t)
    if kind == "timestamp":
        return isinstance(t, pd.DatetimeTZDtype) and str(t.tz) in ("UTC", "+00:00", "Etc/UTC")
    if kind == "bool":
        return pd.api.types.is_bool_dtype(t)
    raise ValueError(f"unknown kind {kind!r}; one of {KINDS}")


def check_file_schema(path, cols, where: str | None = None) -> pa.Schema:
    """The parquet METADATA of `path` against `cols` (names present, type kinds right). No row is read."""
    path = pathlib.Path(path)
    where = where or path.name
    if not path.exists():
        raise E.SchemaError(f"{where}: file {path} does not exist")
    schema = pq.read_schema(path)
    names = set(schema.names)
    missing = [c.name for c in cols if c.name not in names]
    if missing:
        raise E.MissingColumnError(f"{where}: missing column(s) {missing}")
    bad = [f"{c.name} is {schema.field(c.name).type} (want {c.kind})" for c in cols
           if not _arrow_kind_ok(schema.field(c.name).type, c.kind)]
    if bad:
        raise E.DtypeError(f"{where}: " + "; ".join(bad))
    return schema


def check_frame(frame: pd.DataFrame, cols, where: str, not_null=()) -> None:
    """A built frame against `cols`: every column present, dtype of the right kind, and no null in a column
    that is non-nullable (or named in `not_null`)."""
    missing = [c.name for c in cols if c.name not in frame.columns]
    if missing:
        raise E.MissingColumnError(f"{where}: missing column(s) {missing}")
    bad = [f"{c.name} is {frame[c.name].dtype} (want {c.kind})" for c in cols if not _frame_kind_ok(frame[c.name], c.kind)]
    if bad:
        raise E.DtypeError(f"{where}: " + "; ".join(bad))
    must = [c.name for c in cols if not c.nullable] + [n for n in not_null]
    nulls = {n: int(frame[n].isna().sum()) for n in dict.fromkeys(must) if frame[n].isna().any()}
    if nulls:
        raise E.NullabilityError(f"{where}: nulls in non-nullable column(s) {nulls}")


def guard_no_labels(frame: pd.DataFrame, where: str) -> None:
    """A scored-row frame must not carry a label column at all (not even an all-null one): its absence is
    what makes a feature that reads it fail loud instead of silently reading nothing."""
    present = [c for c in LABEL_COLUMNS if c in frame.columns]
    if present:
        raise E.LabelLeakError(f"{where}: scored rows carry label column(s) {present}; they are never read at serve time")


def check_airports(airports, known, where: str) -> None:
    """Every airport code in `airports` must be in `known` (the config registry)."""
    s = pd.Series(np.asarray(airports).astype(object))
    s = s.where(s.notna(), "<null>").astype(str)
    unknown = s[~s.isin(set(known))]
    if len(unknown):
        counts = unknown.value_counts().to_dict()
        raise E.UnknownAirportError(f"{where}: {len(unknown):,} departure(s) at airport(s) not in the config registry "
                                    f"{counts}; add an entry under `airports` (history: false routes them to the pooled "
                                    "lanes with a fallback flag)")


# =================================================================================================
# the scored file
# =================================================================================================

@dataclass(frozen=True)
class ScoredData:
    """`dep`: every departure of the scored file, derived, label-free (the serve witness reads it);
    `scored`: the template's rows of `dep`, in the scored file's order, index reset;
    `template`: the template ids."""
    dep: pd.DataFrame
    scored: pd.DataFrame
    template: pd.DataFrame
    scored_path: pathlib.Path
    template_path: pathlib.Path


def read_serve_departures(path, airports) -> pd.DataFrame:
    """The scored file's departures through SERVE_COLUMNS (never a label column), schema-checked on the
    metadata and on the frame, airports checked against the registry, then build_submission.derive()."""
    path = pathlib.Path(path)
    check_file_schema(path, [RAW_BY_NAME[c] for c in SERVE_COLUMNS], where=f"scored file {path.name}")
    assert not set(SERVE_COLUMNS) & set(LABEL_COLUMNS)
    raw = pq.read_table(path, columns=list(SERVE_COLUMNS)).to_pandas()
    guard_no_labels(raw, f"scored file {path.name}")
    dep = raw[raw.PHASE_mvt == "DEP"]                       # build_submission.load_movements's filter
    del raw
    check_frame(dep, [RAW_BY_NAME[c] for c in SERVE_COLUMNS], f"scored departures of {path.name}",
                not_null=SCORED_NOT_NULL)
    if not dep.MVT_ID_mvt.is_unique:
        raise E.SchemaError(f"scored file {path.name}: duplicate MVT_ID_mvt among the departures")
    check_airports(dep.ADEP_mvt, airports, f"scored file {path.name}")
    out = legacy.stratum().bs.derive(dep)
    guard_no_labels(out, f"derived scored departures of {path.name}")
    return out


def load_template(path) -> pd.DataFrame:
    path = pathlib.Path(path)
    check_file_schema(path, TEMPLATE, where=f"template {path.name}")
    t = pq.read_table(path, columns=["MVT_ID_mvt"]).to_pandas()
    check_frame(t, TEMPLATE[:1], f"template {path.name}")
    if not t.MVT_ID_mvt.is_unique:
        raise E.SchemaError(f"template {path.name}: duplicate MVT_ID_mvt")
    return t


def scored_rows(dep: pd.DataFrame, template: pd.DataFrame, where: str = "scored file") -> pd.DataFrame:
    """The template's rows of `dep`, in `dep`'s order, index reset (build_submission.main's join). Every
    template id must be a departure of the scored file, exactly once."""
    ids = set(template.MVT_ID_mvt)
    scored = dep[dep.MVT_ID_mvt.isin(ids)].reset_index(drop=True)
    if len(scored) != len(template) or set(scored.MVT_ID_mvt) != ids:
        missing = len(ids - set(scored.MVT_ID_mvt))
        raise E.SchemaError(f"{where}: joined {len(scored):,} scored departures for {len(template):,} template ids "
                            f"({missing:,} template ids have no departure)")
    return scored


def ingest_scored(cfg) -> ScoredData:
    """config -> ScoredData for the configured scored file and template."""
    dep = read_serve_departures(cfg.paths.scored_file, cfg.airport_codes)
    template = load_template(cfg.paths.template_file)
    return ScoredData(dep=dep, scored=scored_rows(dep, template, f"scored file {cfg.paths.scored_file.name}"),
                      template=template, scored_path=cfg.paths.scored_file, template_path=cfg.paths.template_file)


# =================================================================================================
# training inputs
# =================================================================================================

def read_training_clock_rows(files) -> pd.DataFrame:
    """The departures of the raw training months through TRAINING_CLOCK_COLUMNS only (what admissibility
    and the airport-hour witness read), in file order -- build_submission.load_movements's rows, projected."""
    files = [pathlib.Path(f) for f in files]
    if not files:
        raise E.SchemaError("no training files")
    parts = []
    for f in files:
        check_file_schema(f, [RAW_BY_NAME[c] for c in TRAINING_CLOCK_COLUMNS], where=f"training file {f.name}")
        t = pq.read_table(f, columns=list(TRAINING_CLOCK_COLUMNS)).to_pandas()
        parts.append(t[t.PHASE_mvt == "DEP"])
    out = pd.concat(parts, ignore_index=True)
    check_frame(out, [RAW_BY_NAME[c] for c in TRAINING_CLOCK_COLUMNS], "training departures (clock columns)")
    return out


def load_rome_frames(rome_cache) -> tuple:
    """(unmatched stratum, LIRF matched) from data/cache_rome (stratum_fold.load_real's recipe, cached by
    rome_fill.load_frames), schema-checked, with derive()'s sp bins re-attached exactly as rome_fill does."""
    rome_cache = pathlib.Path(rome_cache)
    rf = legacy.stratum().rf
    out = []
    for name, want_unmatched in (("unm.parquet", True), ("lirf.parquet", False)):
        p = rome_cache / name
        check_file_schema(p, ROME_FRAME, where=f"rome cache {name}")
        f = rf.with_sp_bins(pd.read_parquet(p))
        check_frame(f, ROME_FRAME, f"rome cache {name}")
        if not f.MVT_ID_mvt.is_unique:
            raise E.SchemaError(f"rome cache {name}: duplicate MVT_ID_mvt")
        if not (f.unmatched.to_numpy() == want_unmatched).all():
            raise E.SchemaError(f"rome cache {name}: `unmatched` must be {want_unmatched} on every row")
        out.append(f)
    unm, lirf = out
    if not (lirf.ADEP_mvt == "LIRF").all():
        raise E.SchemaError("rome cache lirf.parquet: rows outside LIRF")
    return unm, lirf
