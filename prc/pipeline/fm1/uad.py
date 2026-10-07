"""FM1 UAD — the aircraft's own ADS-B ground track as the off-block estimate for a departure without a flight-plan
record (plans/PREREG_uad_unmatched_adsb_offblock_2026_10_04.md, REG 17a570a6; H-UAD-1 TRUE, 2026-10-05). The
per-airport offsets and the calm / storm weights are the registered ones, fitted on the 2025 design months by
scripts/analysis/uad_price.py; FM1 reads them from a file and never refits them. Runs after the unmatched lane's storm
anchor and before LDI.

The value rule is the one scripts/analysis/uad_price.py priced (its u_prime and apply):

    scope     a row still decided unmatched_body or storm (SCOPE), whose modules.uad.table row's class (cls) is A1
              or A2 (CLASSES); every other row (class "" or a decided_by outside SCOPE) is untouched
    c         the registered per-airport offset: constants.by_airport[ADEP_mvt], or the pooled value when the
              airport is not listed
    u         exit_rel_s + c
    u'        p*sp + (1-p)*u (p = the lane's fill probability unm_p_fill, sp = MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt
              in s); u when p or sp is NaN
    w         w1 when the row's airport-day anomaly storm_A is above STORM_A_S (300 s), else w0 (a NaN anomaly counts
              as calm); an A2 row takes A2_WEIGHT_FACTOR (half) of w
    value     rint(max(v + w*(u' - v), 1)); decided_by 'uad' (written even when the value does not change: the label
              alone means "this row has an ADS-B estimate")

A differential test (tests/fm1/test_fm1_uad.py) holds `blend` to uad_price.apply() bit for bit.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np
import pandas as pd

from . import errors as E

SCOPE = ("unmatched_body", "storm")
CLASSES = ("A1", "A2")
STORM_A_S = 300.0
A2_WEIGHT_FACTOR = 0.5
TABLE_COLS = ["MVT_ID_mvt", "unmatched", "cls", "exit_rel_s"]
_COMP_COLS = ("uad_cls", "uad_est", "uad_shift")

#: the table's own build report must pass these before FM1 will read it (scripts/analysis/uad_build.py's
#: registration; a differential test holds these equal to scripts/analysis/uad_price.py's STOP_AIRPORT / STOP_BARS /
#: TABLE_PARAMS_HASH, restated here with an explicit comparison operator instead of a "below"/"above" word).
TABLE_PARAMS_HASH = "d196b652a9a7"     # uad_build.params_hash() of the registered builder (after AMEND-A3)
STOP_AIRPORT = "EHAM"
TABLE_BARS = (("callsign_matched_type_equal_share", ">=", 0.97), ("stand_unmatched_type_agree_share", ">=", 0.95),
              ("matched_exit_mad_s", "<=", 200.0), ("stand_check_right_share", ">=", 0.95),
              ("stand_check_wrong_share", "<=", 0.005))


def load_constants(path) -> dict:
    """{"by_airport": {ap: offset}, "pooled": float, "w0": float, "w1": float, "registration": str} from the
    registered constants file (scripts/analysis/uad_price.py constants; configs/uad_constants_2025.json). Refuses a
    file missing offsets / offsets.by_airport / offsets.pooled / weights / weights.w0 / weights.w1 / registration,
    carrying anything but a finite int or float in any of them (a bool or a numeric string is refused, as the
    table report's bars are), or a weight outside [0, 1]."""
    path = pathlib.Path(path)
    raw = json.loads(path.read_text())
    try:
        offsets = raw["offsets"]
        by_airport_raw = offsets["by_airport"]
        pooled_raw = offsets["pooled"]
        weights = raw["weights"]
        w0_raw = weights["w0"]
        w1_raw = weights["w1"]
        registration = raw["registration"]
    except KeyError as exc:
        raise E.SchemaError(f"{path}: missing key {exc}") from None
    if not isinstance(by_airport_raw, dict):
        raise E.SchemaError(f"{path}: offsets.by_airport must be a mapping of airport -> offset")
    by_airport = {}
    for ap, v in by_airport_raw.items():
        if not _strict_number(v):
            raise E.SchemaError(f"{path}: offsets.by_airport.{ap} is not a finite number ({v!r})")
        by_airport[str(ap)] = float(v)
    if not _strict_number(pooled_raw):
        raise E.SchemaError(f"{path}: offsets.pooled is not a finite number ({pooled_raw!r})")
    out = {"by_airport": by_airport, "pooled": float(pooled_raw)}
    for name, v in (("w0", w0_raw), ("w1", w1_raw)):
        if not _strict_number(v):
            raise E.SchemaError(f"{path}: weights.{name} is not a finite number ({v!r})")
        v = float(v)
        if not (0.0 <= v <= 1.0):
            raise E.SchemaError(f"{path}: weights.{name} = {v} is outside [0, 1]")
        out[name] = v
    if not isinstance(registration, str) or not registration:
        raise E.SchemaError(f"{path}: registration is missing or not a string")
    out["registration"] = registration
    return out


def _strict_number(x) -> bool:
    """True only for an actual, finite int or float -- never a bool (a bool is an int subclass in Python, so a
    bare isinstance(x, (int, float)) check accepts it) and never a string that merely parses as a number. A
    table report's bar values must be real numbers, not text that happens to look like one."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_table_report(table_path) -> str:
    """Reads report.json beside `table_path` (table_path.resolve().parent, never table_path itself) and returns its
    sha256 hex digest. Refuses with E.SchemaError (naming the file and what failed): unreadable or malformed JSON;
    a top level that is not an object; no report.json; params_hash missing or different from TABLE_PARAMS_HASH; no
    stop_rules[STOP_AIRPORT]; a TABLE_BARS key missing, a bool, a string, or any other non-finite-number value; a
    bar missed (names the bar, its value and the limit); an "outputs" entry for table.parquet whose sha256 does
    not equal the real table file's sha256 (a report with no "outputs" at all -- every table built before this
    check existed -- is still accepted, and so is one whose "outputs" says nothing about table.parquet). These
    five bars are the label-free checks the registration fixed for the builder; the 2025-only class-count bracket
    in scripts/analysis/uad_price.py is a pricing-time sanity check, never applied here."""
    report_path = pathlib.Path(table_path).resolve().parent / "report.json"
    if not report_path.exists():
        raise E.SchemaError(f"{report_path}: no report.json beside {table_path}")
    raw = report_path.read_bytes()
    try:
        report = json.loads(raw.decode())
    except ValueError as exc:
        raise E.SchemaError(f"{report_path}: unreadable or malformed JSON ({exc})") from None
    if not isinstance(report, dict):
        raise E.SchemaError(f"{report_path}: top level is not an object ({type(report).__name__})")
    params_hash = report.get("params_hash")
    if params_hash != TABLE_PARAMS_HASH:
        raise E.SchemaError(f"{report_path}: params_hash {params_hash!r} != {TABLE_PARAMS_HASH!r}")
    stop_rules = report.get("stop_rules")
    stats = stop_rules.get(STOP_AIRPORT) if isinstance(stop_rules, dict) else None
    if not isinstance(stats, dict):
        raise E.SchemaError(f"{report_path}: no stop_rules[{STOP_AIRPORT!r}]")
    for key, op, limit in TABLE_BARS:
        v = stats.get(key)
        if not _strict_number(v):
            raise E.SchemaError(f"{report_path}: {key} missing or not a finite number ({v!r})")
        v = float(v)
        ok = (v >= limit) if op == ">=" else (v <= limit)
        if not ok:
            raise E.SchemaError(f"{report_path}: {key} = {v} fails {op} {limit}")
    outputs = report.get("outputs")
    if isinstance(outputs, dict) and "table.parquet" in outputs:
        table_sha256 = hashlib.sha256(pathlib.Path(table_path).read_bytes()).hexdigest()
        if outputs["table.parquet"] != table_sha256:
            raise E.SchemaError(f"{report_path}: the table is not the one this report describes")
    return hashlib.sha256(raw).hexdigest()


def check_table_sources(table_path, adsb_fi: bool) -> list:
    """The ADS-B sources the stand-exit table's build report declares (report.json `sources`: {input_dir: [names]} as
    scripts/analysis/uad_build.py writes it, or a flat list of names), checked against adsb.sources.adsb_fi one way:
    adsb.fi points (non-commercial, not redistributable) are refused unless the knob says the build may use them. A
    report without the key was written before the builder recorded sources, when every input was adsb.lol. One way
    only, unlike features.check_table_sources for the R12 tables: the knob describes those tables, and the stand-exit
    table may hold adsb.lol points alone under either setting (board file v57 did). Returns the sorted names."""
    report_path = pathlib.Path(table_path).resolve().parent / "report.json"
    declared = json.loads(report_path.read_text()).get("sources", ["adsb.lol"])
    names = sorted({str(n) for group in (declared.values() if isinstance(declared, dict) else [declared]) for n in group})
    if "adsb.fi" in names and not adsb_fi:
        raise E.ConfigValueError(f"the stand-exit table {table_path} was built from adsb.fi points (non-commercial, "
                                 f"not redistributable; {report_path}); to use them set adsb.sources.adsb_fi=true")
    return names


def load_table(path) -> pd.DataFrame:
    """TABLE_COLS only. Refuses a null or duplicated MVT_ID_mvt (rows.py's identical check on the same column)."""
    path = pathlib.Path(path)
    t = pd.read_parquet(path, columns=TABLE_COLS)
    t = t.assign(MVT_ID_mvt=t.MVT_ID_mvt.astype("float64"))
    if t.MVT_ID_mvt.isna().any():
        raise E.SchemaError(f"{path}: {int(t.MVT_ID_mvt.isna().sum())} null MVT_ID_mvt")
    if t.MVT_ID_mvt.duplicated().any():
        raise E.DuplicateIdError(f"{path}: {int(t.MVT_ID_mvt.duplicated().sum())} duplicate MVT_ID_mvt")
    return t


def blend(v, est, p, sp, storm_a, cls, c, w0: float, w1: float) -> np.ndarray:
    """new = rint(max(v + w*(u' - v), 1)) over arrays of rows already selected (every `cls` in CLASSES; refuses
    anything else). w = w1 when storm_a > STORM_A_S else w0 (NaN storm_a: w0); an A2 row takes A2_WEIGHT_FACTOR of w.
    u' = p*sp + (1-p)*(est+c), falling back to est+c when p or sp is NaN. Mirrors scripts/analysis/uad_price.py's
    u_prime + apply operation for operation, so the result is bit-identical to the reference
    (test_blend_matches_the_reference_uad_price_apply_exactly)."""
    v = np.asarray(v, dtype="float64")
    est = np.asarray(est, dtype="float64")
    p = np.asarray(p, dtype="float64")
    sp = np.asarray(sp, dtype="float64")
    storm_a = np.asarray(storm_a, dtype="float64")
    c = np.asarray(c, dtype="float64")
    cls = np.asarray(cls)
    bad = ~np.isin(cls, np.asarray(CLASSES))
    if bad.any():
        raise E.SchemaError(f"uad.blend: cls must be one of {CLASSES}, got {sorted(set(cls[bad].tolist()))}")
    fallback = est + c
    nan_fallback = np.isnan(p) | np.isnan(sp)
    with np.errstate(invalid="ignore"):
        blended = p * sp + (1.0 - p) * fallback
    up = np.where(nan_fallback, fallback, blended)
    storm = storm_a > STORM_A_S
    w = np.where(storm, w1, w0)
    factor = np.where(cls == "A2", A2_WEIGHT_FACTOR, 1.0)
    new = v + factor * w * (up - v)
    floored = np.maximum(new, 1.0)
    return np.rint(floored)


def _empty_comp() -> pd.DataFrame:
    return pd.DataFrame({"uad_cls": pd.Series(dtype=object), "uad_est": pd.Series(dtype="float64"),
                        "uad_shift": pd.Series(dtype="float64")},
                       index=pd.Index([], name="MVT_ID_mvt", dtype="float64"))


def serve(rows: pd.DataFrame, comp: pd.DataFrame, ledger, stage, table: pd.DataFrame, const: dict) -> tuple:
    """Select the rows ledger.decided_of in SCOPE, apply `blend` to the ones whose table row classifies (A1/A2), and
    write the result (also when it does not change `v`, so decided_by == 'uad' means "this row has an ADS-B
    estimate"). Returns (comp_out indexed by MVT_ID_mvt: uad_cls, uad_est [= exit_rel_s], uad_shift [= new - v];
    report: n_scope, n_a1, n_a2, n_storm [written rows with storm_A > STORM_A_S], by_airport [written rows per
    airport], w0, w1, registration, shift_mean_s, shift_rms_s). Refuses a scope row absent from `table`, or one whose
    table row says unmatched False (the table was built for other rows); among classed rows, refuses a non-finite or
    non-positive exit_rel_s, or one with no ledger value (no value stage served it: unreachable through the ledger's
    own write() path, a defensive check). With no row in scope, or none classed, returns an empty frame untouched."""
    rep = {"n_scope": 0, "n_a1": 0, "n_a2": 0, "n_storm": 0, "by_airport": {}, "w0": float(const["w0"]),
           "w1": float(const["w1"]), "registration": const["registration"], "shift_mean_s": 0.0, "shift_rms_s": 0.0}
    ids_all = comp.index.to_numpy(dtype="float64")
    decided = ledger.decided_of(ids_all)
    in_scope = np.isin(decided, np.array(SCOPE, dtype=object))
    rep["n_scope"] = int(in_scope.sum())
    if not in_scope.any():
        return _empty_comp(), rep
    scope_ids = ids_all[in_scope]
    t = table.set_index("MVT_ID_mvt")
    pos = t.index.get_indexer(scope_ids)
    missing = pos < 0
    if missing.any():
        raise E.RoutingError(f"{int(missing.sum())} scope row(s) have no table row (e.g. "
                             f"{scope_ids[missing][:3].tolist()})")
    trow = t.iloc[pos]
    unmatched_ok = trow["unmatched"].to_numpy(dtype=bool)
    if not unmatched_ok.all():
        bad = scope_ids[~unmatched_ok]
        raise E.SchemaError(f"{int((~unmatched_ok).sum())} scope row(s) have a table row built for other rows "
                            f"(unmatched False; e.g. {bad[:3].tolist()})")
    cls_all = trow["cls"].to_numpy()
    classed = np.isin(cls_all, np.asarray(CLASSES))
    if not classed.any():
        return _empty_comp(), rep
    c_ids = scope_ids[classed]
    est = trow["exit_rel_s"].to_numpy(dtype="float64")[classed]
    cls = cls_all[classed]
    bad_est = ~(np.isfinite(est) & (est > 0))
    if bad_est.any():
        raise E.RangeError(f"{int(bad_est.sum())} classed row(s) have a non-finite or non-positive exit_rel_s "
                           f"(e.g. {c_ids[bad_est][:3].tolist()})")
    v = ledger.values_of(c_ids)
    if np.isnan(v).any():
        raise E.RoutingError(f"{int(np.isnan(v).sum())} classed row(s) have no value to blend (no value stage "
                             "served them)")
    r = rows.set_index(rows.MVT_ID_mvt.to_numpy(dtype="float64")).loc[c_ids]
    ap = r["ADEP_mvt"].astype(str).to_numpy()
    mvt = pd.to_datetime(r["MVT_TIME_UTC_mvt"], utc=True)
    sched = pd.to_datetime(r["SCHED_TIME_UTC_mvt"], utc=True)
    sp = (mvt - sched).dt.total_seconds().to_numpy(dtype="float64")
    p = comp.loc[c_ids, "unm_p_fill"].to_numpy(dtype="float64")
    storm_a = comp.loc[c_ids, "storm_A"].to_numpy(dtype="float64")
    c = pd.Series(ap).map(const["by_airport"]).fillna(const["pooled"]).to_numpy(dtype="float64")
    new = blend(v, est, p, sp, storm_a, cls, c, const["w0"], const["w1"])
    shift = new - v
    ledger.write(stage, c_ids, new)
    comp_out = pd.DataFrame({"uad_cls": cls, "uad_est": est, "uad_shift": shift},
                            index=pd.Index(c_ids, name="MVT_ID_mvt"))
    storm_mask = storm_a > STORM_A_S
    by_ap = {str(a): int((ap == a).sum()) for a in np.unique(ap)}
    rep.update(n_a1=int((cls == "A1").sum()), n_a2=int((cls == "A2").sum()), n_storm=int(storm_mask.sum()),
               by_airport=by_ap, shift_mean_s=float(np.mean(shift)), shift_rms_s=float(np.sqrt(np.mean(shift ** 2))))
    return comp_out, rep


def apply(cfg, rows: pd.DataFrame, comp: pd.DataFrame, ledger, stage) -> tuple:
    """The engine's entry point: the table and constants named by modules.uad.table / modules.uad.constants, read
    fresh on every call (never at import). check_table_report runs before load_table, so a table whose build report
    does not pass is refused before its parquet is even opened; the report's own sha256 is carried into the returned
    report as "table_report_sha256". check_table_sources then refuses adsb.fi points under adsb.sources.adsb_fi=false,
    the same rule the R12 tables are held to."""
    table_path = cfg.path("modules.uad.table")
    table_report_sha256 = check_table_report(table_path)
    check_table_sources(table_path, cfg["adsb.sources.adsb_fi"])
    table = load_table(table_path)
    const = load_constants(cfg.path("modules.uad.constants"))
    comp_out, rep = serve(rows, comp, ledger, stage, table, const)
    rep["table_report_sha256"] = table_report_sha256
    return comp_out, rep
