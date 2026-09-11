"""P3 — the ADS-B stack stage on the matched lane (plans/PIPELINE_DESIGN_2026_09_10.md "P3 — the ADS-B stack stage").

Contract with prc-challenge-6e (agreed 2026-09-11): 6e owns the ADS-B feature table and the ADN stacker; the pipeline runs it.

    feature table   data/adsb/v2/features_<period>.parquet — every scored MATCHED MVT_ID exactly once, 26 label-free features
                    + `coverage` (0 no day file / 1 not joined / 2 joined mid-taxi / 3 seen from the stand); metadata stamps
                    git_sha, extractor, features_version
    models          <models_dir>/MANIFEST.json (git sha, feats, folds / seeds, `gate`: airport -> {allowed, rule, c25, c26}; the
                    gate is EMPTY until `adsb_ship.py gate` runs on the 2026 table, and an empty gate refuses the stage)
    entry point     predict_residual(frame, model_dir, oof=False) -> seconds per row; 0.0 where coverage < 2 or the airport is
                    not allowed. oof=False is the full bag (2026 rows); oof=True reads a `fold` column (the 2025 guard only)
    ship value      rint(max(proxy − (F + resid), 1)), F = the matched lane's DELTA prediction

The stage never trusts the predictor: it re-asserts the zero rows, finiteness and length, and records per-airport counts of
changed rows. With no stage configured the matched lane is exactly today's (A1).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import errors as E

REQUIRED_META = ("git_sha", "extractor", "features_version")
LABEL_COLUMNS = ("y", "delta", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt")
COVERAGE_CODES = (0, 1, 2, 3)
MIN_COVERAGE = 2


@dataclass(frozen=True)
class StageSpec:
    table: pathlib.Path
    models_dir: pathlib.Path


def read_table(path, expected_ids) -> tuple:
    """(frame indexed by MVT_ID_mvt, metadata). Refuses missing provenance, a label column, duplicate or missing ids, rows
    for flights the matched lane does not score, and an unknown coverage code."""
    path = pathlib.Path(path)
    if not path.exists():
        raise E.SchemaError(f"ADS-B feature table {path} does not exist")
    t = pq.read_table(path)
    meta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items() if k != b"pandas"}
    missing = [k for k in REQUIRED_META if not meta.get(k)]
    if missing:
        raise E.SchemaError(f"{path.name}: provenance {missing} missing from the metadata")
    f = t.to_pandas()
    leak = [c for c in LABEL_COLUMNS if c in f.columns]
    if leak:
        raise E.LabelLeakError(f"{path.name}: label column(s) {leak} in a serve-time feature table")
    for c in ("MVT_ID_mvt", "coverage"):
        if c not in f.columns:
            raise E.MissingColumnError(f"{path.name}: column {c!r} missing")
    ids = f.MVT_ID_mvt.to_numpy(dtype="float64")
    if not pd.Index(ids).is_unique:
        raise E.SchemaError(f"{path.name}: {int(pd.Index(ids).duplicated().sum())} duplicate MVT_IDs")
    want = set(np.asarray(expected_ids, dtype="float64").tolist())
    have = set(ids.tolist())
    if have != want:
        raise E.SchemaError(f"{path.name}: {len(want - have):,} scored matched flights missing, {len(have - want):,} unexpected")
    bad = sorted(set(f.coverage.unique().tolist()) - set(COVERAGE_CODES))
    if bad:
        raise E.SchemaError(f"{path.name}: unknown coverage code(s) {bad}")
    return f.set_index(pd.Index(ids, name="MVT_ID")), meta


def read_gate(models_dir) -> tuple:
    """(gate: airport -> bool, manifest). Entries are either a bool or 6e's {allowed: bool, rule, c25, c26}. Refuses a manifest
    without a gate table (an EMPTY gate means `adsb_ship.py gate` has not run on the 2026 table: nothing may ship) or an entry
    whose `allowed` is not a boolean."""
    p = pathlib.Path(models_dir) / "MANIFEST.json"
    if not p.exists():
        raise E.SchemaError(f"ADN models manifest {p} does not exist")
    man = json.loads(p.read_text())
    gate = man.get("gate")
    if not isinstance(gate, dict) or not gate:
        raise E.SchemaError(f"{p.name}: no `gate` table (airport -> allowed); run the gate on the 2026 table first")
    out, bad = {}, {}
    for a, v in gate.items():
        allowed = v.get("allowed") if isinstance(v, dict) else v
        if not isinstance(allowed, bool):
            bad[a] = v
        else:
            out[a] = allowed
    if bad:
        raise E.SchemaError(f"{p.name}: gate entries must be booleans or carry a boolean `allowed`, got {bad}")
    return out, man


def stage_frame(ids, ap, hour, proxy, F, table: pd.DataFrame) -> pd.DataFrame:
    """The predictor's input: MVT_ID_mvt, ap, hour, proxy, F, coverage + every table feature, in `ids` order."""
    ids = np.asarray(ids, dtype="float64")
    parts = {"ap": np.asarray(ap).astype(str), "hour": np.asarray(hour, dtype="float64"),
             "proxy": np.asarray(proxy, dtype="float64"), "F": np.asarray(F, dtype="float64")}
    for k, v in parts.items():
        if len(v) != len(ids):
            raise E.LaneError(f"stage input {k!r} has {len(v)} values for {len(ids)} rows")
    clash = sorted(set(parts) & set(table.columns))
    if clash:
        raise E.SchemaError(f"ADS-B table column(s) {clash} collide with the stage's own columns; the frame would carry "
                            "duplicate labels and the lane would read a 2-D block")
    got = table.reindex(ids)
    if got.coverage.isna().any():
        raise E.SchemaError(f"{int(got.coverage.isna().sum()):,} rows have no ADS-B table row")
    out = pd.DataFrame({"MVT_ID_mvt": ids, **parts})
    feats = got.drop(columns=["MVT_ID_mvt"], errors="ignore").reset_index(drop=True)
    return pd.concat([out, feats], axis=1)


def apply_stage(frame: pd.DataFrame, predict_residual, gate: dict) -> tuple:
    """(final taxi seconds, info). resid from the predictor; must be finite, one per row, and exactly 0.0 where coverage < 2
    or the airport is not gated. value = rint(max(proxy − (F + resid), 1))."""
    resid = predict_residual(frame)
    if not isinstance(resid, np.ndarray) or resid.shape != (len(frame),):
        raise E.LaneError(f"predict_residual returned {getattr(resid, 'shape', type(resid).__name__)}, expected ({len(frame)},)")
    resid = resid.astype("float64")
    if not np.isfinite(resid).all():
        raise E.LaneError(f"predict_residual returned {int((~np.isfinite(resid)).sum())} non-finite values")
    gated = frame.ap.map(lambda a: bool(gate.get(a, False))).to_numpy()
    covered = frame.coverage.to_numpy() >= MIN_COVERAGE
    must_zero = ~(gated & covered)
    if (resid[must_zero] != 0.0).any():
        raise E.LaneError(f"predict_residual moved {int((resid[must_zero] != 0.0).sum()):,} rows that are uncovered or at an "
                          "ungated airport (the contract says 0.0 there)")
    proxy, F = frame.proxy.to_numpy(dtype="float64"), frame.F.to_numpy(dtype="float64")
    base = np.rint(np.maximum(proxy - F, 1.0))
    value = np.rint(np.maximum(proxy - (F + resid), 1.0))
    changed = value != base
    info = {"n_rows": int(len(frame)), "n_eligible": int((~must_zero).sum()), "n_changed": int(changed.sum()),
            "changed_by_airport": {str(a): int(n) for a, n in pd.Series(changed).groupby(frame.ap.to_numpy()).sum().items() if n},
            "coverage_share_by_airport": {str(a): float(v) for a, v in pd.Series(covered).groupby(frame.ap.to_numpy()).mean().items()},
            "mean_abs_resid_eligible_s": float(np.abs(resid[~must_zero]).mean()) if (~must_zero).any() else 0.0}
    return value, info


def coverage_transfer(ap25, cov25, ap26, cov26) -> dict:
    """Guard (3): per airport, n rows, the movable share (coverage >= MIN_COVERAGE) and the from-the-stand share (coverage 3)
    in each period, and ratio = movable26 / movable25 (None where an airport has no 2025 rows or no movable 2025 row). The
    stack's 2025 gain transfers to 2026 only in proportion to where it can act."""
    parts = {}
    for name, ap, cov in (("25", ap25, cov25), ("26", ap26, cov26)):
        ap, cov = np.asarray(ap).astype(str), np.asarray(cov)
        if len(ap) != len(cov):
            raise E.LaneError(f"coverage_transfer: period {name} has {len(ap)} airports for {len(cov)} coverage codes (length)")
        bad = sorted(set(pd.unique(cov).tolist()) - set(COVERAGE_CODES))
        if bad:
            raise E.SchemaError(f"coverage_transfer: unknown coverage code(s) {bad} in period {name}")
        parts[name] = pd.DataFrame({"ap": ap, "movable": cov >= MIN_COVERAGE, "stand": cov == 3})
    out = {}
    for a in sorted(set(parts["25"].ap) | set(parts["26"].ap)):
        row = {}
        for name, f in parts.items():
            g = f[f.ap == a]
            row[f"n{name}"] = int(len(g))
            row[f"movable{name}"] = float(g.movable.mean()) if len(g) else None
            row[f"stand{name}"] = float(g.stand.mean()) if len(g) else None
        row["ratio"] = (row["movable26"] / row["movable25"]) if row["movable25"] and row["movable26"] is not None else None
        out[a] = {k: row[k] for k in ("n25", "n26", "movable25", "movable26", "stand25", "stand26", "ratio")}
    return out


def check_stage_inputs(table_path, models_dir, expected_ids) -> dict:
    """The dry run's check of everything the stage will read, before the matched lane spends its minutes: the table
    (read_table's refusals), the gate (read_gate's; every gated airport among the manifest's airports), every seed x fold
    model file the predictor loads (G booster, N state dict, N standardisation stats), and the manifest's features present
    in the table. Returns a report for the dry-run output."""
    table, meta = read_table(table_path, expected_ids)
    gate, man = read_gate(models_dir)
    unknown = sorted(set(gate) - set(man.get("airports", [])))
    if unknown:
        raise E.SchemaError(f"gate airport(s) {unknown} not in the manifest's airports {man.get('airports')}")
    d = pathlib.Path(models_dir)
    want = [f"{p}_s{s}_k{k}{ext}" for s in man["seeds"] for k in range(man["folds"])
            for p, ext in (("G", ".txt"), ("N", ".pt"), ("N", "_stats.npz"))]
    missing = [n for n in want if not (d / n).exists()]
    if missing:
        raise E.SchemaError(f"{d}: model file(s) {missing} missing")
    absent = [c for c in man["feats"] if c not in table.columns]
    if absent:
        raise E.MissingColumnError(f"{pathlib.Path(table_path).name}: the models' feature(s) {absent} are not in the table")
    if not any(gate.values()):
        raise E.SchemaError(f"{pathlib.Path(models_dir).name}/MANIFEST.json: the gate allows no airport; a build under it "
                            "would ship the lane's own values inside a manifest that claims an ADS-B stage")
    return {"table": str(table_path), "table_meta": meta, "n_rows": int(len(table)),
            "movable_share": float((table.coverage.to_numpy() >= MIN_COVERAGE).mean()),
            "models_dir": str(models_dir), "n_model_files": len(want),
            "gate_allowed": sorted(a for a, ok in gate.items() if ok)}
