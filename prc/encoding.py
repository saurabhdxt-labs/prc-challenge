"""Target encoding with a correct fit / early-stopping / holdout separation.

WHY THIS MODULE EXISTS (verified in the shipped code, 2026-09-10)
-----------------------------------------------------------------
`scripts/lgbm_fold.py::load_fold` computes the 24 target/delta encodings with

    S.infold_encodings(d, y, dlt, tr)

and `fold_masks` puts the early-stopping months INSIDE `tr` -- its own docstring says
"ES months come out of the TRAINING fold" (lgbm_fold.py:657). Every early-stopping
row therefore carries encodings fitted on data that includes that row's own label.

What this does and does not invalidate:

* The outer Jan+Jul holdout is excluded from `tr`, so it never sees its own labels.
  Every reported holdout RMSE and every paired bootstrap interval in this repository
  remains a valid measurement. Nothing needs re-scoring.
* `best_iter` was selected against an optimistically biased stopping metric. Tree
  counts, and any comparison of tree counts or stopping RMSE ACROSS arms, are
  distorted by an unmeasured amount. That is a model-selection defect.

This module is the repair. It is a change of SCOPE, not of formula: the smoothing is
byte-identical to `stand_ab._smooth_enc`, pinned by
`tests/test_encoding_separation.py::test_scope_change_only_the_formula_is_the_incumbent_smoothing`.
It is a NEW module and imports nothing from `scripts/`, so the arms queue running
beneath it is untouched.

THE TWO STAGES
--------------
A single design matrix cannot serve both stages correctly, because a stopping row is
an evaluation row in stage 1 and a training row in stage 2, and it needs different
encodings in each. Callers should hold the 24 encoding columns separately and write
the stage-appropriate set into the matrix before each fit; at 12 months that is about
200 MB per stage, against several GB for a second full matrix.

    stage1 (early stopping)  fit rows -> cross-fitted, leave-one-MONTH-out within the
                                         fit months
                             ES rows  -> encoder fitted on the fit months only
    stage2 (refit + score)   training rows -> cross-fitted, leave-one-MONTH-out within
                                         all training months
                             holdout/serve -> encoder fitted on all training months

Whole-month exclusion is deliberate. Random row folds would leak through same-day,
same-stand structure, which is exactly the dependence the date-block bootstrap in
`tools/top_path_audit.py` was built to respect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: identical to stand_ab.SMOOTH; the equivalence test fails if either moves
SMOOTH = 50.0


def _agg(keys: np.ndarray, vals: np.ndarray) -> pd.DataFrame:
    """level -> (sum, count) over the rows handed in."""
    return (pd.DataFrame({"k": keys, "v": vals})
            .groupby("k", observed=True).v.agg(["sum", "count"]))


def _apply(g: pd.DataFrame, out_keys: np.ndarray, prior: float,
           k: float = SMOOTH) -> np.ndarray:
    """Smoothed level means, prior for unseen levels. The incumbent formula, exactly."""
    enc = (g["sum"] + prior * k) / (g["count"] + k)
    return pd.Series(np.asarray(out_keys)).map(enc).fillna(prior).to_numpy(np.float32)


def _counts(g: pd.DataFrame, out_keys: np.ndarray) -> np.ndarray:
    """Support of each row's level in the fitting set. Cross-fitting changes this."""
    return (pd.Series(np.asarray(out_keys)).map(g["count"])
            .fillna(0.0).to_numpy(np.float32))


def _check_delta(delta: np.ndarray, mask: np.ndarray) -> None:
    bad = int(np.isnan(np.asarray(delta, dtype="float64")[mask]).sum())
    if bad:
        raise ValueError(
            f"NaN delta on {bad:,} rows of the delta-encoding set: pass delta_mask to "
            "fit the delta encodings where delta is defined")


def _crossfit_one(key_col: np.ndarray, vals: np.ndarray, month: np.ndarray,
                  rows: np.ndarray, out: np.ndarray) -> None:
    """Write leave-one-month-out encodings for `rows` into `out`, in place.

    Implemented by subtracting each month's aggregate from the pooled one, so the cost
    is O(months x levels) rather than O(months x rows).
    """
    tot = _agg(key_col[rows], vals[rows])
    tot_sum, tot_n = float(vals[rows].sum()), int(rows.sum())
    fallback = tot_sum / tot_n if tot_n else 0.0
    for m in np.unique(month[rows]):
        sel = rows & (month == m)
        part = _agg(key_col[sel], vals[sel])
        g = tot.subtract(part, fill_value=0.0)
        g = g[g["count"] > 0]
        n_out = tot_n - int(sel.sum())
        # one fit month: nothing is left to fit on, so every row takes the prior
        prior = (tot_sum - float(vals[sel].sum())) / n_out if n_out > 0 else fallback
        out[sel] = _apply(g, key_col[sel], prior)


def fit_apply(keys: pd.DataFrame, y, delta, fit_rows, out_rows=None, enc_keys=None,
              with_counts: bool = False, delta_mask=None) -> dict:
    """Fit one encoder on `fit_rows` and transform every row.

    `out_rows` is accepted for call-site readability and does not restrict the output;
    the arrays are full length so callers can index them with any mask.
    """
    if enc_keys is None:
        raise ValueError("enc_keys is required: the caller owns the key list")
    y, delta = np.asarray(y, dtype="float64"), np.asarray(delta, dtype="float64")
    fit_rows = np.asarray(fit_rows, dtype=bool)
    d_rows = fit_rows if delta_mask is None else (fit_rows & np.asarray(delta_mask, bool))
    _check_delta(delta, d_rows)
    py, pd_ = float(y[fit_rows].mean()), float(delta[d_rows].mean())
    out = {}
    for k in enc_keys:
        col = keys[k].to_numpy()
        gy, gd = _agg(col[fit_rows], y[fit_rows]), _agg(col[d_rows], delta[d_rows])
        out[f"te_{k}"] = _apply(gy, col, py)
        out[f"de_{k}"] = _apply(gd, col, pd_)
        if with_counts:
            out[f"n_{k}"] = _counts(gy, col)
    return out


def separated_encodings(keys: pd.DataFrame, y, delta, month, fit_mask, es_mask,
                        tr_mask, enc_keys=None, delta_mask=None,
                        with_counts: bool = False) -> dict:
    """The two stage-appropriate encoding sets. See the module docstring.

    Returns ``{"stage1": {...}, "stage2": {...}}``, each a dict of full-length float32
    arrays keyed ``te_<k>`` / ``de_<k>`` (and ``n_<k>`` when `with_counts`).
    """
    if enc_keys is None:
        raise ValueError("enc_keys is required: the caller owns the key list")
    y, delta = np.asarray(y, dtype="float64"), np.asarray(delta, dtype="float64")
    month = np.asarray(month)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    es_mask = np.asarray(es_mask, dtype=bool)
    tr_mask = np.asarray(tr_mask, dtype=bool)
    if (fit_mask & es_mask).any():
        raise ValueError("fit and early-stopping masks overlap")
    if not (fit_mask & ~tr_mask == False).all():          # noqa: E712 - elementwise on arrays
        raise ValueError("the fit mask must lie inside the training mask")
    # `es` is NOT required to lie inside `tr`. Two tier shapes are supported and both are real:
    #   fold tier   fit | es == tr  (ES months carved out of training; lgbm_fold.fold_masks)
    #   screen tier fit == tr, es == the stop month, OUTSIDE tr (lgbm_fold.sweep_masks, which
    #               returns (stop, train, train, stop) and never refits)
    # Requiring es <= tr silently made the screening tier unrunnable for the repaired variants.
    dm = np.ones(len(y), bool) if delta_mask is None else np.asarray(delta_mask, bool)
    _check_delta(delta, tr_mask & dm)

    stages = {}
    for name, cross_rows, enc_rows in (("stage1", fit_mask, fit_mask),
                                       ("stage2", tr_mask, tr_mask)):
        # the encoder every row outside `cross_rows` is transformed by
        base = fit_apply(keys, y, delta, fit_rows=enc_rows, enc_keys=enc_keys,
                         with_counts=with_counts, delta_mask=dm)
        d_cross = cross_rows & dm
        for k in enc_keys:
            col = keys[k].to_numpy()
            _crossfit_one(col, y, month, cross_rows, base[f"te_{k}"])
            _crossfit_one(col, delta, month, d_cross, base[f"de_{k}"])
        stages[name] = base
    return stages


# --------------------------------------------------------------------------- scoped keys
#: Measured 2026-09-10 on data/cache_stand, June held out from March-May:
#: 55.3% of rows carry a STAND_mvt name that also exists at another airport (stand "210"
#: appears at six), and 44.6% carry a shared RUNWAY_mvt name. Within one shared name the
#: per-airport mean `delta` differs by a median of 79 s and a p90 of 317 s, so the shipped
#: encoding averages physically different stands. `ars` does not fix it either: it is
#: `STAND|RUNWAY` (stand_ab.py:340), with no airport in the key.
#:
#: Single-key out-of-fold RMSE on June, scoped against unscoped -- a paired comparison of
#: two versions of the SAME feature, everything else held identical:
#:     AIRCRAFT_TYPE_mvt  387.862 -> 363.760   (-24.102)
#:     ADES_mvt           383.027 -> 363.116   (-19.910)
#:     STAND_mvt          368.140 -> 361.256   ( -6.884)
#:     RUNWAY_mvt         369.643 -> 367.311   ( -2.333)
#: These are single-key magnitudes and WILL NOT transfer to a 68-feature model that already
#: carries ADEP_mvt and can recover part of the pooling. They establish that the statistic is
#: corrupted, not what repairing it is worth. The marginal value needs a paired fold A/B.
#: `ars` is DELIBERATELY ABSENT. It is built as `<airport>|<stand>|<runway>`
#: (stand_ab.py:340 -- `d_stand` is already airport-prefixed, verified on 100% of cached rows),
#: so re-scoping it would produce "LFPG|LFPG|I18|27L" and a redundant encoding. An earlier
#: version of this list included it and reports/PRIORITY1_IDENTITY_SCREEN.md claimed `ars` had
#: no airport in the key; both were wrong, from reading line 340 without resolving `d_stand`.
#: Corrected 2026-09-10 06:40. The four keys below ARE unscoped in stand_ab.ENC_KEYS.
SCOPED_DEFAULT = ("STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "ADES_mvt")


def add_airport_scoped(keys: pd.DataFrame, airport: str = "ADEP_mvt",
                       scope: "tuple[str, ...]" = SCOPED_DEFAULT,
                       sep: str = "|") -> "tuple[pd.DataFrame, list]":
    """Append `<airport>|<key>` composites and return (frame, new column names).

    The separator must never be NUL: the Arrow string path treats it as a terminator and
    silently collapses composite keys (a landmine this repository has already paid for).
    Missing values become the literal "NA" before joining, so a null key cannot swallow the
    airport prefix and collapse distinct rows into one level.
    """
    if sep == "\x00" or not sep:
        raise ValueError("the key separator must be a non-empty, non-NUL string")
    if airport not in keys.columns:
        raise KeyError(f"airport column {airport!r} absent from the key frame")
    out = keys.copy()
    ap = out[airport].fillna("NA").astype(str)
    made = []
    for k in scope:
        if k not in keys.columns:
            raise KeyError(f"key {k!r} asked for scoping but absent from the key frame")
        name = f"ap_{k}"
        out[name] = ap + sep + out[k].fillna("NA").astype(str)
        made.append(name)
    return out, made
