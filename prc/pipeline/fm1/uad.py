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


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def load_constants(path) -> dict:
    """{"by_airport": {ap: offset}, "pooled": float, "w0": float, "w1": float, "registration": str} from the
    registered constants file (scripts/analysis/uad_price.py constants; configs/uad_constants_2025.json). Refuses a
    file missing offsets / offsets.by_airport / offsets.pooled / weights / weights.w0 / weights.w1 / registration,
    carrying a non-finite number in any of them, or a weight outside [0, 1]."""
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
        if not _finite(v):
            raise E.SchemaError(f"{path}: offsets.by_airport.{ap} is not a finite number ({v!r})")
        by_airport[str(ap)] = float(v)
    if not _finite(pooled_raw):
        raise E.SchemaError(f"{path}: offsets.pooled is not a finite number ({pooled_raw!r})")
    out = {"by_airport": by_airport, "pooled": float(pooled_raw)}
    for name, v in (("w0", w0_raw), ("w1", w1_raw)):
        if not _finite(v):
            raise E.SchemaError(f"{path}: weights.{name} is not a finite number ({v!r})")
        v = float(v)
        if not (0.0 <= v <= 1.0):
            raise E.SchemaError(f"{path}: weights.{name} = {v} is outside [0, 1]")
        out[name] = v
    if not isinstance(registration, str) or not registration:
        raise E.SchemaError(f"{path}: registration is missing or not a string")
    out["registration"] = registration
    return out


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
    fresh on every call (never at import)."""
    table = load_table(cfg.path("modules.uad.table"))
    const = load_constants(cfg.path("modules.uad.constants"))
    return serve(rows, comp, ledger, stage, table, const)
