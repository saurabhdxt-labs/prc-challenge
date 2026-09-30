"""ARS-CAT -- add the joint (airport, stand, runway) key to the CatBoost half of the matched blend.

Registration: plans/PREREG_arscat_joint_key_catboost_2026_09_20.md (registered 2026-09-20, before any
ARS-CAT number existed).

The gap: production's matched prediction on `pop` rows is `inc = 0.5*(taxi_C + taxi_FW)`. The LightGBM
half carries `ars = ap|STAND|RUNWAY` (`stand_ab.py:344`, target-encoded at :401-402, the strongest single
feature in the file, holdout r2 0.3015). The CatBoost half does not: `catboost_native.py:31` lists
CAT_COLS without it and :53-54 builds `ap_stand` and `ap_rwy` SEPARATELY.

Two constraints from the registration, both enforced here in code:
  * `ars` enters ONLY through CAT_COLS, so CatBoost applies its own ordered target statistics to it.
    Raw high-cardinality `ars` was measured at -919.5 [-1118, -729] and is refused (assert_ars_is_categorical).
  * Stage 1 fits BOTH arms at the same reduced iteration count in the same run, so the comparison is
    fair even though neither is converged (assert_fair_screen).

`ars` is RECONSTRUCTED from the existing airport-scoped keys rather than rebuilt from source:
`ap_stand` is `ap|STAND` and `ap_rwy` is `ap|RUNWAY`, so `ars = ap_stand + "|" + runway`. This is
identical to the LightGBM definition and touches no source data.

CatBoost and LightGBM never share a process (BC-5): the fit runs in
~/.venvs/prc-catboost/bin/python3 (catboost 1.2.10).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

SEP = "|"
STAGE1_ITERS = 1_000          # the registered reduced count
SHIPPED_ITERS = 6_664         # the shipped count, for stage 2 only
BOARD_N = 344_841.0
VENV_PY = str(pathlib.Path.home() / ".venvs" / "prc-catboost" / "bin" / "python3")


def joint_key(ap_stand: pd.Series, ap_rwy: pd.Series) -> pd.Series:
    """ars = ap|STAND|RUNWAY, reconstructed from the two airport-scoped keys.

    Refuses a key without a separator and an airport-prefix mismatch between the two inputs, because
    either would silently produce a wrong joint key.
    """
    s = ap_stand.astype(str)
    r = ap_rwy.astype(str)
    if not r.str.contains(SEP, regex=False).all():
        raise ValueError("an ap_rwy value has no separator; it is not an airport-scoped key")
    if not s.str.contains(SEP, regex=False).all():
        raise ValueError("an ap_stand value has no separator; it is not an airport-scoped key")
    ap_s = s.str.split(SEP, n=1).str[0]
    ap_r = r.str.split(SEP, n=1).str[0]
    if not (ap_s.to_numpy() == ap_r.to_numpy()).all():
        raise ValueError("ap_stand and ap_rwy disagree on the airport prefix")
    runway = r.str.split(SEP, n=1).str[1]          # split ONCE: a runway containing a separator survives
    return s + SEP + runway


def cat_cols_with_ars(cat_cols) -> list:
    """The shipped categoricals in their shipped order, with `ars` appended exactly once."""
    out = list(cat_cols)
    if "ars" not in out:
        out.append("ars")
    return out


def cat_cols_control(cat_cols) -> list:
    """The control never sees `ars`."""
    return [c for c in cat_cols if c != "ars"]


def assert_ars_is_categorical(numeric_cols, cat_cols) -> None:
    """Raw `ars` as a numeric/native high-cardinality feature measured -919.5. Refuse it."""
    if "ars" in set(numeric_cols):
        raise ValueError("`ars` is in the numeric design; it may only enter through CAT_COLS "
                         "(raw ars measured -919.5 [-1118, -729])")
    if "ars" not in set(cat_cols):
        raise ValueError("`ars` is not in CAT_COLS; the candidate would be identical to the control")


LABEL_AND_MASK_COLS = frozenset({"score_y", "score_proxy", "score_sp", "delta", "G",
                                 "train", "fit", "es", "holdout", "row", "MVT_ID_mvt", "month"})


def assert_no_label_columns(numeric_cols) -> None:
    """The numeric design must never contain a label, a score column or a fold mask.

    Regression guard for the defect the quality gate's dry run caught on 2026-09-21: deriving the
    numeric design by EXCLUSION from the parquet's columns pulled in `score_y` -- the label itself --
    along with score_proxy/score_sp and the four boolean fold masks. Always take the design from
    catboost_native.numeric_columns(), handed over from the LightGBM side (BC-5).
    """
    bad = sorted(LABEL_AND_MASK_COLS & set(numeric_cols))
    if bad:
        raise ValueError(f"the numeric design contains label/mask columns: {bad}")


def assert_fair_screen(control_iters: int, candidate_iters: int) -> None:
    """Stage 1 compares two identically-reduced fits; unequal budgets make the screen meaningless."""
    if int(control_iters) != int(candidate_iters):
        raise ValueError(f"stage 1 must fit both arms at the same iteration count, "
                         f"got control={control_iters}, candidate={candidate_iters}")


MIN_DAY_BLOCKS = 10


def assert_day_blocks_are_real(day, month) -> None:
    """A day-blocked bootstrap resamples DAYS within month; it needs many days per month.

    Regression guard for the defect found 2026-09-21: `arscat_prep` fell back to `day = month` because
    opc_screen's frame has no `date` column, so each month held exactly ONE block, every resample was
    identical, and the bootstrap returned a ZERO-WIDTH interval that looked like a precise result.
    """
    import numpy as _np
    day = _np.asarray(day)
    month = _np.asarray(month)
    for mth in _np.unique(month):
        n = len(_np.unique(day[month == mth]))
        if n < MIN_DAY_BLOCKS:
            raise ValueError(f"month {mth} has only {n} distinct day blocks (need >= {MIN_DAY_BLOCKS}); "
                             f"a day-blocked bootstrap cannot vary and will return a zero-width CI")


def gain_board(y, control, treatment, n: float = BOARD_N) -> float:
    """(SSE_control - SSE_treatment) / n on the scored population. Positive = the candidate is better."""
    y = np.asarray(y, dtype="float64")
    c = np.asarray(control, dtype="float64")
    t = np.asarray(treatment, dtype="float64")
    return float((((y - c) ** 2).sum() - ((y - t) ** 2).sum()) / n)


def stage1_passes(gain: float, ci) -> bool:
    """The registered gate: mean gain > 0 AND the pooled CI excludes zero. No size bar (registered)."""
    lo, hi = float(ci[0]), float(ci[1])
    return bool(gain > 0.0 and lo > 0.0)
