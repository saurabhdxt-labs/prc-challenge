"""Validate the assembled submission before anything is written, and guard the no-splice rule.

  * build_submission.check_submission (columns, row count, id set, no null / non-finite / non-positive);
  * dtype int32, template row order, every value inside VALUE_RANGE_S;
  * per-lane row counts equal the routing's;
  * every airport that has scored rows has a prediction on each of them; a configured airport with no
    scored rows is a WARNING (a new scored file may lack one), recorded in the manifest;
  * fallback rows (airports without history) counted per airport.

`compare_to_reference` is a read-only diagnostic (acceptance test A1): it reports equality against another
file per lane and never feeds a value back. `assert_no_splice` refuses any pipeline input under
`submissions/`.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from . import errors as E
from . import legacy

#: taxi seconds: strictly positive, and at most two days -- the longest legitimate predictions are LIRF's
#: 24 h date-slip expectations (86,400 s + the date-slip median, ~87-90k s); anything past two days is a bug
VALUE_RANGE_S = (1, 172_800)
SUBMISSIONS_DIR = "submissions"


def assert_no_splice(paths) -> list:
    """Every pipeline input must come from data, never from a previous submission. Returns the checked list."""
    out = []
    for p in paths:
        p = pathlib.Path(p)
        if SUBMISSIONS_DIR in p.resolve().parts:
            raise E.SpliceRefusedError(f"input {p} is under {SUBMISSIONS_DIR}/: the pipeline never splices a previous "
                                       "submission (the lgbm_submit --base landmine)")
        out.append(str(p))
    return out


def validate(assembled, template: pd.DataFrame, routing, cfg) -> dict:
    bs = legacy.stratum().bs
    sub, prov = assembled.submission, assembled.provenance
    try:
        bs.check_submission(sub, template)
    except ValueError as exc:
        raise E.ValidationError(f"check_submission: {exc}") from None
    if sub.TAXITIME_SEC_mvt.dtype != np.int32:
        raise E.ValidationError(f"TAXITIME_SEC_mvt is {sub.TAXITIME_SEC_mvt.dtype}, expected int32")
    if not np.array_equal(sub.MVT_ID_mvt.to_numpy(dtype="float64"), template.MVT_ID_mvt.to_numpy(dtype="float64")):
        raise E.ValidationError("the submission is not in the template's row order")
    v = sub.TAXITIME_SEC_mvt.to_numpy()
    lo, hi = VALUE_RANGE_S
    out = (v < lo) | (v > hi)
    if out.any():
        raise E.ValidationError(f"{int(out.sum())} predictions outside [{lo}, {hi}] s (max {int(v.max())}, min {int(v.min())})")
    want = routing.counts()
    have = assembled.lane_counts()
    if have != want:
        raise E.ValidationError(f"per-lane row counts {have} != the routing's {want}")
    ap_scored = pd.Series(routing.airport).value_counts().to_dict()
    ap_pred = prov.airport.value_counts().to_dict()
    short = {a: (int(n), int(ap_pred.get(a, 0))) for a, n in ap_scored.items() if ap_pred.get(a, 0) != n}
    if short:
        raise E.ValidationError(f"airports without a prediction on every scored row (scored, predicted): {short}")
    warnings = [f"configured airport {a} has no scored rows" for a in cfg.airport_codes if a not in ap_scored]
    per_lane = {}
    for lane, g in prov.groupby("lane"):
        x = g.TAXITIME_SEC_mvt.to_numpy()
        per_lane[str(lane)] = {"n": int(len(x)), "min": int(x.min()), "median": float(np.median(x)), "max": int(x.max()),
                               "by_stage": {str(k): int(n) for k, n in g.stage.value_counts().sort_index().items()}}
    fb = prov[prov.fallback.to_numpy(dtype=bool)]
    return {"n_rows": int(len(sub)), "per_lane": per_lane,
            "per_airport": {str(a): int(n) for a, n in sorted(ap_scored.items())},
            "fallback_rows": {str(a): int(n) for a, n in fb.airport.value_counts().sort_index().items()},
            "value_range_s": list(VALUE_RANGE_S), "warnings": warnings, "check_submission": "passed"}


def compare_to_reference(assembled, reference_path) -> dict:
    """Per lane: rows, rows equal to the reference file, rows different, max |diff|, the first ids that
    differ. Read-only: the reference is never a pipeline input."""
    ref = pd.read_parquet(reference_path)
    r = ref.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    prov = assembled.provenance
    if set(r.index) != set(prov.MVT_ID_mvt):
        raise E.ValidationError(f"{reference_path}: a different id set from the assembled submission")
    have = prov.TAXITIME_SEC_mvt.to_numpy(dtype="int64")
    want = r.reindex(prov.MVT_ID_mvt.to_numpy()).to_numpy(dtype="int64")
    out = {}
    for lane in sorted(prov.lane.unique()):
        m = (prov.lane == lane).to_numpy()
        d = have[m] - want[m]
        ne = d != 0
        out[str(lane)] = {"n": int(m.sum()), "n_equal": int((~ne).sum()), "n_differ": int(ne.sum()),
                          "max_abs_diff_s": int(np.abs(d).max()) if len(d) else 0,
                          "first_differing_ids": [float(i) for i in prov.MVT_ID_mvt.to_numpy()[m][ne][:10]]}
    tot = have != want
    out["ALL"] = {"n": int(len(have)), "n_equal": int((~tot).sum()), "n_differ": int(tot.sum())}
    return out
