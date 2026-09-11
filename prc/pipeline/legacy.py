"""The shipped, tested scripts the pipeline wraps -- imported, never copied and never edited (P2a).

The scripts are not a package: they import each other through `scripts/` on sys.path (rome_ship,
unm_congestion, ...). This module puts `scripts/` on the path once and imports them by name, so
every lane sees ONE instance of each (`stratum_fold.bs` is the build_submission every stratum script
uses). Imports are lazy: loading the config or checking a schema never pulls in lightgbm.
"""
from __future__ import annotations

import functools
import pathlib
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _ensure_path() -> None:
    s = str(SCRIPTS)
    if s not in sys.path:
        sys.path.insert(0, s)


@functools.lru_cache(maxsize=1)
def stratum() -> SimpleNamespace:
    """The unmatched / Rome stack: build_submission (bs), stratum_fold (sf), unm_congestion (uc),
    unm_congestion_ship (ucs), rome_fill (rf), rome_dateslip (ds), rome_bandfloor (rb)."""
    _ensure_path()
    import rome_bandfloor as rb  # noqa: E402
    import rome_dateslip as ds  # noqa: E402
    import rome_fill as rf  # noqa: E402
    import stratum_fold as sf  # noqa: E402
    import unm_congestion as uc  # noqa: E402
    import unm_congestion_ship as ucs  # noqa: E402
    return SimpleNamespace(bs=sf.bs, sf=sf, uc=uc, ucs=ucs, rf=rf, ds=ds, rb=rb)


@functools.lru_cache(maxsize=1)
def fold() -> SimpleNamespace:
    """The fold harness: lgbm_fold (lf) — load_fold, fold_masks, fit_arm, the FEATS lists. It loads its own copy of
    lgbm_submit through an importlib spec (lf.L); the functions are the same code as matched().ls."""
    _ensure_path()
    import lgbm_fold as lf  # noqa: E402
    return SimpleNamespace(lf=lf)


@functools.lru_cache(maxsize=1)
def matched() -> SimpleNamespace:
    """The matched stack: lgbm_submit (ls) and the stand_ab it loaded (S)."""
    _ensure_path()
    import lgbm_submit as ls  # noqa: E402
    return SimpleNamespace(ls=ls, S=ls.S)


@functools.lru_cache(maxsize=1)
def rome_local_day():
    """prc-challenge-6e/53's scripts/rome_local_day.py (rule RLD; read-only): apply_rld, SP_LO, SP_HI, AIRPORT, TZ."""
    _ensure_path()
    import rome_local_day as m  # noqa: E402
    return m


@functools.lru_cache(maxsize=1)
def adsb_stack():
    """prc-challenge-6e's scripts/adsb_stack.py (read-only): day_folds and predict_residual."""
    _ensure_path()
    import adsb_stack as m  # noqa: E402
    return m


def adsb_predictor():
    """prc-challenge-6e's ADN entry point `adsb_stack.predict_residual(frame, model_dir, oof)` (read-only use; the MLP runs in 6e's own worker
    subprocess, so this process never imports torch — BC-5). Refuses clearly if the module does not provide it yet."""
    fn = getattr(adsb_stack(), "predict_residual", None)
    if fn is None:
        raise NotImplementedError("scripts/adsb_stack.py has no predict_residual yet (prc-challenge-6e ships it with the artefacts)")
    return fn
