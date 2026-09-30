#!/usr/bin/env python3.11
"""CSF.2 — the Rome (LIRF) unmatched identifier: does an IDENTITY-augmented fill classifier beat the SHIPPED one?
Registration: plans/PREREG_csf2_rome_identifier_2026_09_22.md (hash log reports/csf2_prereg_hashes.log).

    /opt/homebrew/bin/python3.11 scripts/analysis/csf2_build.py baseline   # step 1: replicate R + the ORACLE (before registering)
    /opt/homebrew/bin/python3.11 scripts/analysis/csf2_build.py run        # the registered run (after registering)
    /opt/homebrew/bin/python3.11 scripts/analysis/csf2_build.py smoke      # synthetic inputs, code path only, NOT a result

The SHIPPED LIRF fill classifier at v26 is `rome_fill.fit_predict_fill` (LightGBM arm R, stake-weighted, mean of seeds
0-2, trained on every LIRF 2025 departure) — `configs/pipeline_afj.yaml:79` -> `prc/pipeline/lanes.py run_rules_lane`.
Its stored leave-one-month-out probabilities are `data/cache_stand/rome_fill_preds.parquet` (`p_R`, `p_R_seed{s}`) with
S1's non-fill term `nf`; the value of any classifier here is measured through the shipped mixture
`max(p*sp + (1-p)*nf, 1)` with `nf` held fixed, so every delta is the classifier's.

Arms (every input label-free; the fit is the shipped recipe with extra columns, `extra_num` / `extra_cat`):
  R     the shipped recipe refit — MUST replicate the stored artefact per seed (atol 1e-6) or the run ABORTS;
  CS    + callsign frequency (CSF's `log1p(n_cs)`, `has_twin36`, `twin_h`, unchanged definitions) + `one_off`;
  FULL  CS + imputed aircraft type / market segment from the same callsign's other legs in the same calendar month
        (+ their source flags) + the callsign's matched-elsewhere rate. FULL is the decisional arm.

Characteristic failures, each with its own test (tests/test_csf2_build.py): an imputation that reads across calendar
months or groups null callsigns together (BC-15); a `cs_match_rate` that counts the row itself; a baseline that is not
the shipped artefact; a value clause the perfect oracle cannot pass; post-compute lines that crash after the fits (BC-24).

Never writes under /tmp. Holds PHANTOM's heavy lock (non-blocking; exit 75 if held). Aborts at 6 GiB RSS.
No upload, no commit, no board read, no production change.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import rome_fill as rf  # noqa: E402  (stratum_fold -> build_submission -> sklearn at import; lightgbm inside the fit: the shipped order)
import stratum_fold as sf  # noqa: E402
from scripts.analysis import csf_build as CSF  # noqa: E402  (callsign_features and auc: CSF's registered definitions, unchanged)

# ---- registered constants ----
EXPECT_ROWS, EXPECT_JANJUL, N_MONTHS = 1488, 397, 12
SEEDS = (0, 1, 2)
AUC_TRUE, AUC_FALSE = 0.95, 0.92
SLIP_Y_S, SLIP_TAXI_S = 86_400.0, 1020.0             # slip := non-fill with y >= 86,400 s; median slip taxi (PREREG_rome_fill RESULT)
SCORED_LIRF_UNM, BOARD_N = 383, 344_841
N_BOOT, BOOT_SEED = 2000, 0
FORWARD_MONTHS = (7, 8, 9, 10, 11, 12)
DECISION_MONTHS = (1, 7)
K_RANGE = (1.38, 5.87)
RSS_ABORT_GIB = 6.0
REPLICATION_ATOL = 1e-6
TOP_K = (1, 3)
APTS = CSF.APTS
NUM_EXTRA_CS = ("log_n_cs", "one_off", "has_twin36", "twin_h")
NUM_EXTRA_ID = ("cs_match_rate", "actype_src", "segment_src")
CAT_EXTRA_ID = ("actype_any", "segment_any")
ARMS = {"R": ((), ()), "CS": (NUM_EXTRA_CS, ()), "FULL": (NUM_EXTRA_CS + NUM_EXTRA_ID, CAT_EXTRA_ID)}
DECISIONAL_ARM = "FULL"
FEATURE_COLS = ("n_cs", "has_twin36", "twin_h", "log_n_cs", "one_off", "actype_any", "actype_src", "segment_any",
                "segment_src", "cs_match_rate")
POOL_COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
             "BLOCK_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt", "MARKET_SEGMENT_flt", "AOBT_3_flt"]
VALUE_KEYS = ("c2_ci_excludes_zero", "c3_both_months_positive", "c4_minus_top3_positive", "c5_forward_positive")

PREREG = ROOT / "plans" / "PREREG_csf2_rome_identifier_2026_09_22.md"
HASHLOG = ROOT / "reports" / "csf2_prereg_hashes.log"
STORED = ROOT / "data" / "cache_stand" / "rome_fill_preds.parquet"
OUT_DIR = ROOT / "reports" / "csf2"
LOCK_PATH = pathlib.Path(os.environ.get("PHANTOM_HEAVY_LOCK_PATH",
                                        pathlib.Path.home() / ".phantom" / "phantom_heavy_compute.lock"))

auc = CSF.auc


# ================================================================== labels

def labels(frame: pd.DataFrame) -> pd.DataFrame:
    """fill := |BLOCK - SCHED| <= 60 s (the shipped definition, rome_fill.fill_label); slip := non-fill with
    y >= 86,400 s (the 24 h date-slip class); ordinary otherwise. Training rows only (reads BLOCK)."""
    y = frame.y.to_numpy(dtype="float64")
    if not np.isfinite(y).all():
        raise ValueError("labels: y must be finite on every row")
    fill = rf.fill_label(frame)
    slip = (~fill) & (y >= SLIP_Y_S)
    cls = np.where(fill, "fill", np.where(slip, "slip", "ordinary"))
    return pd.DataFrame({"fill": fill, "slip": slip, "cls": cls}, index=frame.index)


def label_integrity(frame: pd.DataFrame) -> dict:
    """Reported before any verdict: the maximum label and the counts above the physical bounds."""
    L = labels(frame)
    y = frame.y.to_numpy(dtype="float64")
    return {"n_rows": int(len(frame)), "max_y_s": float(y.max()), "min_y_s": float(y.min()),
            "n_y_ge_86400": int((y >= 86_400).sum()), "n_y_gt_10800": int((y > 10_800).sum()),
            "n_fill": int(L.fill.sum()), "n_slip": int(L.slip.sum())}


# ================================================================== features

def _utc(t: pd.Series) -> pd.Series:
    t = pd.to_datetime(t)
    return t.dt.tz_localize("UTC") if t.dt.tz is None else t.dt.tz_convert("UTC")


def ym_key(cs: pd.Series, t: pd.Series) -> pd.Series:
    """'CALLSIGN|YYYY-MM' per row; None where the callsign is null (an identity that matches NOTHING, BC-15)."""
    k = cs.astype("string") + "|" + _utc(t).dt.strftime("%Y-%m").astype("string")
    return k.astype("object").where(k.notna(), None)


def _modal(keys: pd.Series, values: pd.Series) -> pd.Series:
    """key -> the most frequent value among rows with a key and a value; ties broken alphabetically."""
    v = values.astype("object")
    ok = keys.notna().to_numpy() & v.notna().to_numpy()
    if not ok.any():
        return pd.Series(dtype=object)
    g = pd.DataFrame({"k": keys.to_numpy()[ok], "v": v.to_numpy()[ok].astype(str)})
    cnt = g.groupby(["k", "v"]).size().reset_index(name="n")
    cnt = cnt.sort_values(["k", "n", "v"], ascending=[True, False, True], kind="mergesort")
    first = cnt.drop_duplicates("k", keep="first")
    return pd.Series(first.v.to_numpy(), index=first.k.to_numpy())


def identity_features(rows: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """`actype_any`, `actype_src`, `segment_any`, `segment_src`, `cs_match_rate` for `rows`, aligned to `rows.index`.

    Label-free by construction. A null field is imputed from the SAME callsign's OTHER rows in the SAME calendar month
    (any phase, any airport) — the modal value; `*_src` is 0 own / 1 imputed / 2 none. A null callsign is never
    imputed (BC-15). `cs_match_rate` is the share of the callsign's OTHER departures at the ten airports in the month
    that carry an NM off-block (`AOBT_3_flt`), -1.0 where there is no other departure.
    """
    idx = rows.index
    kr = pd.Series(ym_key(rows.FLIGHT_mvt, rows.MVT_TIME_UTC_mvt).to_numpy(), index=idx)
    known = kr.notna().to_numpy()
    kp = ym_key(pool.FLIGHT_mvt, pool.MVT_TIME_UTC_mvt)
    typ_pool = pool.AIRCRAFT_TYPE_mvt.astype("object").where(pool.AIRCRAFT_TYPE_mvt.notna(),
                                                             pool.AIRCRAFT_TYPE_flt.astype("object"))
    m_typ, m_seg = _modal(kp, typ_pool), _modal(kp, pool.MARKET_SEGMENT_flt)

    def fill_in(own: pd.Series, mp: pd.Series):
        has_own = own.notna().to_numpy()
        imp = kr.map(mp) if len(mp) else pd.Series([None] * len(idx), index=idx, dtype=object)
        has_imp = (~has_own) & known & imp.notna().to_numpy()
        val = np.where(has_own, own.astype("object").to_numpy(dtype=object),
                       np.where(has_imp, imp.astype("object").to_numpy(dtype=object), "NA"))
        src = np.where(has_own, 0, np.where(has_imp, 1, 2)).astype(np.int64)
        return val.astype(str), src

    actype, asrc = fill_in(rows.AIRCRAFT_TYPE_mvt, m_typ)
    segment, ssrc = fill_in(rows.MARKET_SEGMENT_flt, m_seg)

    dp = pool[(pool.PHASE_mvt == "DEP") & pool.ADEP_mvt.isin(APTS) & pool.FLIGHT_mvt.notna()]
    kd = ym_key(dp.FLIGHT_mvt, dp.MVT_TIME_UTC_mvt)
    tab = pd.DataFrame({"k": kd.to_numpy(), "m": dp.AOBT_3_flt.notna().to_numpy()}).groupby("k").m.agg(["size", "sum"])
    self_in = rows.MVT_ID_mvt.isin(dp.MVT_ID_mvt).to_numpy()
    self_m = rows.AOBT_3_flt.notna().to_numpy()
    n = kr.map(tab["size"]).fillna(0).to_numpy(dtype="float64")
    m = kr.map(tab["sum"]).fillna(0).to_numpy(dtype="float64")
    n_other = n - self_in
    m_other = m - (self_in & self_m)
    rate = np.where(known & (n_other > 0), m_other / np.maximum(n_other, 1.0), -1.0)
    return pd.DataFrame({"actype_any": actype, "actype_src": asrc, "segment_any": segment, "segment_src": ssrc,
                         "cs_match_rate": rate}, index=idx)


def build_features(rows: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """CSF's callsign features (unchanged definitions) + the identity features, in FEATURE_COLS order."""
    cf = CSF.callsign_features(rows, pool, pool)
    out = pd.concat([cf, identity_features(rows, pool)], axis=1)
    out["log_n_cs"] = np.log1p(out.n_cs.to_numpy(dtype="float64"))
    out["one_off"] = (out.n_cs.to_numpy() == 1).astype(np.int64)
    return out[list(FEATURE_COLS)]


def attach(frame: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame, build_features(frame, pool)], axis=1)


# ================================================================== folds and fits

def lomo_folds(months) -> list:
    months = [int(m) for m in months]
    return [(m, [o for o in months if o != m]) for m in months]


def forward_folds(months, forward=FORWARD_MONTHS) -> list:
    """Fit on strictly EARLIER months, evaluate month m (the §M16 forward-chronological leg)."""
    months = sorted(int(m) for m in months)
    missing = [m for m in range(1, max(forward) + 1) if m not in months]
    if missing:
        raise ValueError(f"forward folds need every month 1..{max(forward)}: missing month(s) {missing}")
    return [(m, [o for o in months if o < m]) for m in forward]


def fit_arm(tr: pd.DataFrame, te: pd.DataFrame, seeds, extra_num, extra_cat) -> tuple:
    """(mean P(fill) over seeds, {seed: P(fill)}) from the shipped recipe with the arm's extra columns."""
    by = {}
    for s in seeds:
        p = rf.fit_predict_fill(tr, te, seed=int(s), extra_num=tuple(extra_num), extra_cat=tuple(extra_cat))
        if not (np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()):
            raise ValueError(f"seed {s}: non-finite or out-of-range P(fill)")
        by[int(s)] = p
    return np.mean([by[int(s)] for s in seeds], axis=0), by


def lomo_fit(u: pd.DataFrame, l_: pd.DataFrame, months, arms: dict, seeds, log) -> tuple:
    """Leave-one-month-out P(fill) for the LIRF unmatched rows `u` (all LIRF) with matched rows `l_`.
    Returns ({arm: mean p}, {arm: {seed: p}}) aligned to `u`'s row order."""
    mon = u.month.to_numpy(dtype=int)
    P = {a: np.full(len(u), np.nan) for a in arms}
    PS = {a: {int(s): np.full(len(u), np.nan) for s in seeds} for a in arms}
    for m, others in lomo_folds(months):
        te_mask = mon == m
        tr = rf.rome_rows(u[~te_mask], l_[l_.month != m], others)
        te = u[te_mask]
        if not te_mask.any():
            raise ValueError(f"LOMO fold {m}: no test rows")
        for a, (xn, xc) in arms.items():
            p, by = fit_arm(tr, te, seeds, xn, xc)
            P[a][te_mask] = p
            for s in seeds:
                PS[a][int(s)][te_mask] = by[int(s)]
        log(f"  lomo {m:2d}: train {len(tr):,} ({int(tr.unmatched.sum()):,} unm) test {int(te_mask.sum())}")
    return P, PS


def forward_fit(u: pd.DataFrame, l_: pd.DataFrame, months, arms: dict, seeds, log) -> dict:
    """Forward-chronological P(fill) (mean over seeds) for the months in FORWARD_MONTHS; NaN elsewhere."""
    mon = u.month.to_numpy(dtype=int)
    P = {a: np.full(len(u), np.nan) for a in arms}
    for m, earlier in forward_folds(months):
        te_mask = mon == m
        tr = rf.rome_rows(u[mon < m], l_[l_.month < m], earlier)
        te = u[te_mask]
        for a, (xn, xc) in arms.items():
            P[a][te_mask], _ = fit_arm(tr, te, seeds, xn, xc)
        log(f"  forward {m:2d}: train months {earlier[0]}..{earlier[-1]} ({len(tr):,} rows) test {int(te_mask.sum())}")
    return P


# ================================================================== stored artefact + replication guard

def load_pool(files) -> pd.DataFrame:
    import pyarrow.parquet as pq
    parts = [pq.read_table(f, columns=POOL_COLS).to_pandas() for f in files]
    pool = pd.concat(parts, ignore_index=True)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"):
        pool[c] = _utc(pool[c])
    return pool


def load_stored(path=STORED) -> pd.DataFrame:
    return pd.read_parquet(path)


def stored_lookup(stored: pd.DataFrame, ids, fold: str) -> pd.DataFrame:
    s = stored[stored.fold == fold].set_index("MVT_ID_mvt")
    ids = np.asarray(ids, dtype="float64")
    missing = np.setdiff1d(ids, s.index.to_numpy(dtype="float64"))
    if len(missing):
        raise ValueError(f"{len(missing)} row(s) absent from the stored fold {fold!r}")
    return s.loc[ids].reset_index()


def replicate_baseline(refit_by_seed: dict, stored_by_seed: dict, atol: float = REPLICATION_ATOL) -> dict:
    """Abort unless the refit shipped recipe reproduces the stored shipped artefact, seed by seed."""
    d = {str(s): float(np.max(np.abs(np.asarray(refit_by_seed[s], float) - np.asarray(stored_by_seed[s], float))))
         for s in refit_by_seed}
    worst = max(d.values()) if d else 0.0
    if worst > atol:
        raise ValueError(f"the refit baseline does not replicate the stored shipped artefact: max|delta| {worst:.3e} "
                         f"> atol {atol:.0e} (by seed {d})")
    return {"atol": atol, "max_abs_delta_by_seed": d, "passed": True}


# ================================================================== value arithmetic

def rome_day(t: pd.Series) -> pd.Series:
    return _utc(t).dt.tz_convert("Europe/Rome").dt.strftime("%Y-%m-%d")


def se_rows(p, sp, nf, y) -> np.ndarray:
    """Squared error of the SHIPPED mixture max(p*sp + (1-p)*nf, 1) — imported, never re-implemented."""
    return (rf.mixture(np.asarray(p, float), np.asarray(sp, float), np.asarray(nf, float)) - np.asarray(y, float)) ** 2


def oracle_preds(cls, sp, nf) -> tuple:
    """(two-class, three-class) perfect-identifier predictions with the incumbent's value terms:
    fill -> sp; ordinary -> nf; slip -> 86,400 + the median slip taxi (three-class only)."""
    cls = np.asarray(cls)
    two = rf.mixture((cls == "fill").astype("float64"), np.asarray(sp, float), np.asarray(nf, float))
    three = np.where(cls == "slip", SLIP_Y_S + SLIP_TAXI_S, two)
    return two, three


def scale_board(sse_delta: float, n_rows: int) -> float:
    """SSE on n 2025 rows -> board MSE at 2026's 383 LIRF unmatched rows: sse x (383/n) / 344,841 (PREREG_rome_fill §3)."""
    return float(sse_delta) * (SCORED_LIRF_UNM / float(n_rows)) / BOARD_N


def remove_top(delta, k: int) -> tuple:
    """(sum without the k largest-|delta| rows, share of sum|delta| those k rows carry)."""
    d = np.asarray(delta, dtype="float64")
    order = np.argsort(-np.abs(d), kind="stable")
    top = order[:k]
    tot = float(np.abs(d).sum())
    share = 1.0 if (k >= len(d) or tot == 0.0) else float(np.abs(d[top]).sum() / tot)
    return float(np.delete(d, top).sum()), share


def day_block_ci(se_ref, se_new, days, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Paired DAY-block bootstrap of SSE(ref) - SSE(new): calendar days resampled with replacement."""
    days = np.asarray(days)
    a = pd.Series(np.asarray(se_ref, float)).groupby(days).sum()
    b = pd.Series(np.asarray(se_new, float)).groupby(days).sum().reindex(a.index)
    r = rf.month_block_bootstrap(a.to_numpy(), b.to_numpy(), n_draws=n_boot, seed=seed)
    return {"gain": r["gain"], "ci95": r["ci95"], "n_days": int(len(a)), "n_draws": int(n_boot), "seed": int(seed)}


def value_clauses(se_ref, se_new, days, months, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """The registered value clauses on a paired squared-error comparison. C2: gain > 0 with the day-block CI excluding
    0; C3: every month's gain > 0; C4: the gain without the top-3 |delta| rows > 0. MDE = 2.8 x half-width / 1.96."""
    se_ref, se_new = np.asarray(se_ref, float), np.asarray(se_new, float)
    d = se_ref - se_new
    months = np.asarray(months)
    ci = day_block_ci(se_ref, se_new, days, n_boot, seed)
    by_month = {str(int(m)): float(d[months == m].sum()) for m in np.unique(months)}
    gain = float(d.sum())
    m1, s1 = remove_top(d, 1)
    m3, s3 = remove_top(d, 3)
    half = (ci["ci95"][1] - ci["ci95"][0]) / 2.0
    return {"n_rows": int(len(d)), "gain_sse": gain, "ci95": ci["ci95"], "n_days": ci["n_days"], "by_month": by_month,
            "minus_top1_sse": m1, "minus_top3_sse": m3, "top1_share": s1, "top3_share": s3,
            "mde_sse": float(2.8 * half / 1.96),
            "c2_ci_excludes_zero": bool(gain > 0 and ci["ci95"][0] > 0),
            "c3_both_months_positive": bool(all(v > 0 for v in by_month.values())),
            "c4_minus_top3_positive": bool(m3 > 0)}


def band_split(delta, sp) -> dict:
    d, sp = np.asarray(delta, float), np.asarray(sp, float)
    return {"lt3600": float(d[sp < 3600].sum()), "3600_24000": float(d[(sp >= 3600) & (sp < 24_000)].sum()),
            "ge24000": float(d[sp >= 24_000].sum())}


def move_split(pred_ref, pred_new, delta) -> dict:
    """The gain split by how far the prediction moved (feedback_prc_tail_driven_levers: gains living in a few
    large moves are a bet on those rows)."""
    mv = np.abs(np.asarray(pred_new, float) - np.asarray(pred_ref, float))
    d = np.asarray(delta, float)
    return {"le300s": float(d[(mv > 0) & (mv <= 300)].sum()), "300_3600s": float(d[(mv > 300) & (mv <= 3600)].sum()),
            "gt3600s": float(d[mv > 3600].sum()), "unchanged": float(d[mv == 0].sum())}


def value_block(se_ref, se_new, pred_ref, pred_new, days, months, sp, n_boot, seed) -> dict:
    v = value_clauses(se_ref, se_new, days, months, n_boot, seed)
    n = v["n_rows"]
    d = np.asarray(se_ref, float) - np.asarray(se_new, float)
    v.update({"scaling": f"sse x 383/n / 344841 with n = {n}",
              "board_scaled": scale_board(v["gain_sse"], n),
              "board_scaled_minus_top1": scale_board(v["minus_top1_sse"], n),
              "board_scaled_minus_top3": scale_board(v["minus_top3_sse"], n),
              "ci95_board": [scale_board(c, n) for c in v["ci95"]],
              "mde_board": scale_board(v["mde_sse"], n),
              "by_month_board": {k: scale_board(x, n) for k, x in v["by_month"].items()},
              "rmse_ref": float(np.sqrt(np.mean(se_ref))), "rmse_new": float(np.sqrt(np.mean(se_new))),
              "bands_sse": band_split(d, sp), "moves_sse": move_split(pred_ref, pred_new, d)})
    return v


def weighted_auc(y, p, w):
    """Sample-weighted AUC (the ledger's sp^2-weighted figure); None when one class is absent."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=bool)
    if y.all() or (~y).all():
        return None
    return float(roc_auc_score(y, np.asarray(p, float), sample_weight=np.asarray(w, float)))


# ================================================================== decision

def decide(auc_full: float, clauses: dict) -> dict:
    """H-CSF2 is WORKING iff the LOMO AUC >= 0.95 AND every value clause holds; FALSE band below 0.92."""
    if not np.isfinite(auc_full):
        raise ValueError("decide: the AUC must be finite")
    band = "TRUE" if auc_full >= AUC_TRUE else ("FALSE" if auc_full < AUC_FALSE else "BETWEEN")
    value_ok = all(bool(clauses[k]) for k in VALUE_KEYS)
    verdict = "WORKING" if (band == "TRUE" and value_ok) else "NOT WORKING"
    return {"auc_band": band, "value": "WORKING" if value_ok else "NOT WORKING", "verdict": verdict,
            "build_authorised": verdict == "WORKING"}


# ================================================================== the ARR-row witness

def witness_table(arr_rows: pd.DataFrame, pool: pd.DataFrame) -> dict:
    """Arrival in-block copy rate (|BLOCK - SCHED| <= 60 s on ARR rows, unmasked in 2026) by the callsign's
    departure frequency in the month (CSF's n_cs). Label-free for the scored DEP rows; a 2026 test of the mechanism."""
    n_rows = int(len(arr_rows))
    buckets = {"0-1": None, "2-4": None, "5-9": None, "10+": None}
    if n_rows == 0:
        return {"n_rows": 0, "copy_rate_all": None, "buckets": {k: {"n": 0, "copy_rate": None} for k in buckets},
                "ratio_oneoff_vs_10plus": None}
    cp = ((arr_rows.BLOCK_TIME_UTC_mvt - arr_rows.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= rf.FILL_TOL_S).to_numpy()
    n_cs = CSF.callsign_features(arr_rows, pool, pool).n_cs.to_numpy()
    masks = {"0-1": n_cs <= 1, "2-4": (n_cs >= 2) & (n_cs <= 4), "5-9": (n_cs >= 5) & (n_cs <= 9), "10+": n_cs >= 10}
    for k, mk in masks.items():
        buckets[k] = {"n": int(mk.sum()), "copy_rate": (float(cp[mk].mean()) if mk.any() else None)}
    a, b = buckets["0-1"]["copy_rate"], buckets["10+"]["copy_rate"]
    ratio = (a / b) if (a is not None and b is not None and b > 0) else None
    return {"n_rows": n_rows, "copy_rate_all": float(cp.mean()), "buckets": buckets, "ratio_oneoff_vs_10plus": ratio}


def witness_clause(w25: dict, w26: dict) -> bool:
    a, b = w25.get("ratio_oneoff_vs_10plus"), w26.get("ratio_oneoff_vs_10plus")
    return bool(a is not None and b is not None and a > 1.0 and b > 1.0)


# ================================================================== post-compute lines (BC-24)

def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_report(R: dict, path) -> pathlib.Path:
    path = pathlib.Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists - a rerun must not silently become a second look at the same bar")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(R, indent=1, default=_jsonable))
    return path


def append_hash(log, label: str, path, note: str = "") -> str:
    sha = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{ts}  {label}  {path}  sha256={sha[:8]}  (full {sha})"
    if note:
        line += f"  -- {note}"
    with open(log, "a") as fh:
        fh.write(line + "\n")
    return line


def summary_lines(R: dict) -> list:
    v = R["verdict"]
    out = [f"VERDICT {v['verdict']}  (AUC band {v['auc_band']}, value {v['value']}, build_authorised {v['build_authorised']})"]
    for arm, blk in R["lomo"].items():
        out.append(f"  {arm:5s} LOMO AUC {blk['auc_pooled']:.4f}" + (
            f"  janjul gain {blk['value_janjul']['board_scaled']:+.1f} board [{blk['value_janjul']['ci95_board'][0]:+.1f}, "
            f"{blk['value_janjul']['ci95_board'][1]:+.1f}]" if "value_janjul" in blk else ""))
    for k, c in R["clauses"].items():
        out.append(f"  {k:28s} {'PASS' if c else 'fail'}")
    out.append(f"wall {R.get('wall_s', float('nan')):.0f}s")
    return out


def rss_guard(limit_gib: float = RSS_ABORT_GIB, interval_s: float = 5.0) -> threading.Thread:
    """Abort the process (exit 3) the moment peak RSS exceeds `limit_gib`."""
    div = 1024 ** 3 if sys.platform == "darwin" else 1024 ** 2

    def _loop():
        while True:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / div
            if rss > limit_gib:
                sys.stderr.write(f"RSS ABORT: peak RSS {rss:.3f} GiB > {limit_gib} GiB\n")
                sys.stderr.flush()
                os._exit(3)
            time.sleep(interval_s)

    th = threading.Thread(target=_loop, daemon=True, name="rss-guard")
    th.start()
    return th


def acquire_heavy_lock(path=LOCK_PATH):
    """PHANTOM's machine-wide heavy-compute lock, non-blocking: exit 75 if another job holds it (never stack)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        print(f"HEAVY LOCK HELD by another job - skipping, not stacking ({path})", flush=True)
        raise SystemExit(75)
    return fh


def release_heavy_lock(fh) -> None:
    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    fh.close()


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def _git_sha() -> str | None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _versions() -> dict:
    import lightgbm
    import sklearn
    return {"python": sys.version.split()[0], "lightgbm": lightgbm.__version__, "sklearn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__}


# ================================================================== the compositions

def _prepare(unm: pd.DataFrame, smoke: bool) -> tuple:
    u = unm[unm.ADEP_mvt == "LIRF"].reset_index(drop=True)
    if not smoke and len(u) != EXPECT_ROWS:
        raise ValueError(f"LIRF unmatched rows = {len(u):,}, registration says {EXPECT_ROWS:,}")
    months = sorted(int(m) for m in u.month.unique())
    if len(months) != N_MONTHS:
        raise ValueError(f"{len(months)} months present, registration fixed {N_MONTHS}")
    return u, months


def _align_stored(u: pd.DataFrame, stored: pd.DataFrame, fold: str) -> pd.DataFrame:
    st = stored_lookup(stored, u.MVT_ID_mvt.to_numpy(), fold)
    if not (np.allclose(st.sp.to_numpy(float), u.sp.to_numpy(float), rtol=0, atol=1e-6)
            and np.allclose(st.y.to_numpy(float), u.y.to_numpy(float), rtol=0, atol=1e-6)):
        raise ValueError(f"stored fold {fold!r}: sp/y differ from the frame's")
    return st


def _oracle_blocks(cls, sp, nf, y, se_inc, pred_inc, day, mon, n_boot, seed) -> dict:
    two, three = oracle_preds(cls, sp, nf)
    jj = np.isin(mon, DECISION_MONTHS)
    out = {}
    for name, pred in (("two_class", two), ("three_class", three)):
        se_o = (pred - y) ** 2
        out[name] = {"janjul": value_block(se_inc[jj], se_o[jj], pred_inc[jj], pred[jj], day[jj], mon[jj], sp[jj], n_boot, seed),
                     "all12": value_block(se_inc, se_o, pred_inc, pred, day, mon, sp, n_boot, seed)}
    return out


def baseline(unm, lirf, stored, seeds=SEEDS, n_boot=N_BOOT, log=print, smoke=False, refit=True) -> dict:
    """Step 1: read the shipped artefact's AUCs and price the perfect identifier (the ORACLE) against the shipped
    mixture on LOMO Jan+Jul, all 12 months, and fold A. With `refit` the shipped recipe is also refit under LOMO and
    must replicate the stored artefact; `refit=False` touches no learner (the guard then runs inside `run`)."""
    t0 = time.time()
    u, months = _prepare(unm, smoke)
    L = labels(u)
    integrity = label_integrity(u)
    fill, cls = L.fill.to_numpy(), L.cls.to_numpy()
    sp, y = u.sp.to_numpy(float), u.y.to_numpy(float)
    day, mon = rome_day(u.MVT_TIME_UTC_mvt).to_numpy(), u.month.to_numpy(dtype=int)
    log(f"baseline: {len(u):,} LIRF unmatched rows, fill rate {fill.mean():.4f}, integrity {integrity}")
    st = _align_stored(u, stored, "lomo")
    if refit:
        P, PS = lomo_fit(u, lirf, months, {"R": ARMS["R"]}, seeds, log)
        rep = replicate_baseline({int(s): PS["R"][int(s)] for s in seeds},
                                 {int(s): st[f"p_R_seed{int(s)}"].to_numpy(float) for s in seeds}, REPLICATION_ATOL)
    else:
        P, PS = {"R": None}, {"R": {}}
        rep = {"atol": REPLICATION_ATOL, "max_abs_delta_by_seed": None, "passed": None,
               "deferred": "no refit before registration; the replication guard runs inside the registered run"}
    nf, p_st = st.nf.to_numpy(float), st.p_R.to_numpy(float)
    pos = sp > 0
    A = {"stored_p_R": auc(fill, p_st), "refit_p_R": (auc(fill, P["R"]) if refit else None),
         "stored_p_R_by_seed": {str(int(s)): auc(fill, st[f"p_R_seed{int(s)}"].to_numpy(float)) for s in seeds},
         "refit_p_R_by_seed": ({str(int(s)): auc(fill, PS["R"][int(s)]) for s in seeds} if refit else None),
         "stored_p_R_by_month": {str(m): auc(fill[mon == m], p_st[mon == m]) for m in months},
         "stored_p_R_janjul": auc(fill[np.isin(mon, DECISION_MONTHS)], p_st[np.isin(mon, DECISION_MONTHS)]),
         "stored_p_R_sp2_weighted": weighted_auc(fill[pos], p_st[pos], sp[pos] ** 2),
         "stored_p_S1_logistic": (auc(fill, st.p_S1.to_numpy(float)) if "p_S1" in st else None)}
    se_inc, pred_inc = se_rows(p_st, sp, nf, y), rf.mixture(p_st, sp, nf)
    oracle = _oracle_blocks(cls, sp, nf, y, se_inc, pred_inc, day, mon, n_boot, BOOT_SEED)
    # fold A (months 1, 7 predicted from the other ten), from the stored artefact only
    fa = stored[stored.fold == "A"]
    ua = u[u.MVT_ID_mvt.isin(fa.MVT_ID_mvt)].reset_index(drop=True)
    if len(ua):
        sa = _align_stored(ua, stored, "A")
        La = labels(ua)
        spa, ya, nfa, pa = ua.sp.to_numpy(float), ua.y.to_numpy(float), sa.nf.to_numpy(float), sa.p_R.to_numpy(float)
        se_a, pred_a = se_rows(pa, spa, nfa, ya), rf.mixture(pa, spa, nfa)
        two_a, three_a = oracle_preds(La.cls.to_numpy(), spa, nfa)
        daya, mona = rome_day(ua.MVT_TIME_UTC_mvt).to_numpy(), ua.month.to_numpy(dtype=int)
        for name, pred in (("two_class", two_a), ("three_class", three_a)):
            oracle[name]["foldA"] = value_block(se_a, (pred - ya) ** 2, pred_a, pred, daya, mona, spa, n_boot, BOOT_SEED)
        A["stored_p_R_foldA"] = auc(La.fill.to_numpy(), pa)
    else:
        for name in ("two_class", "three_class"):
            oracle[name]["foldA"] = {"n_rows": 0}
    return {"kind": "baseline", "smoke": bool(smoke), "refit": bool(refit), "seeds": [int(s) for s in seeds], "n_boot": int(n_boot),
            "n_rows": int(len(u)), "n_janjul": int(np.isin(mon, DECISION_MONTHS).sum()), "fill_rate": float(fill.mean()),
            "label_integrity": integrity, "replication": rep, "auc": A, "oracle": oracle,
            "git_sha": _git_sha(), "versions": _versions(), "wall_s": round(time.time() - t0, 1),
            "peak_rss_gb": round(_peak_rss_gb(), 2)}


def run(unm, lirf, pool, stored, arr25, arr26, seeds=SEEDS, n_boot=N_BOOT, log=print, smoke=False, pool26=None,
        preds_out=None) -> dict:
    """The registered run: features -> LOMO (R, CS, FULL) -> replication guard -> oracle -> value clauses -> forward
    leg -> ARR-row witness -> decision. Every number the RESULT quotes comes from the returned dict."""
    t0 = time.time()
    u, months = _prepare(unm, smoke)
    L = labels(u)
    integrity = label_integrity(u)
    fill, cls = L.fill.to_numpy(), L.cls.to_numpy()
    log(f"run: {len(u):,} LIRF unmatched rows, fill rate {fill.mean():.4f}, integrity {integrity}")
    uf, lf = attach(u, pool), attach(lirf, pool)
    fs = {"n_cs_median_unm": float(uf.n_cs.median()), "one_off_rate_unm": float(uf.one_off.mean()),
          "has_twin36_rate_unm": float(uf.has_twin36.mean()),
          "actype_src_unm": {str(k): int(v) for k, v in uf.actype_src.value_counts().sort_index().items()},
          "segment_src_unm": {str(k): int(v) for k, v in uf.segment_src.value_counts().sort_index().items()},
          "actype_imputed_share_unm": float((uf.actype_src == 1).mean()),
          "cs_match_rate_median_unm": float(uf.cs_match_rate.median()),
          "cs_match_rate_sentinel_share_unm": float((uf.cs_match_rate < 0).mean()),
          "null_callsign_unm": int(uf.FLIGHT_mvt.isna().sum()),
          "fill_rate_by_one_off": {str(k): float(v) for k, v in pd.Series(fill).groupby(uf.one_off.to_numpy()).mean().items()},
          "fill_rate_by_actype_src": {str(k): float(v) for k, v in pd.Series(fill).groupby(uf.actype_src.to_numpy()).mean().items()}}
    log(f"features: {fs}")
    sp, y = uf.sp.to_numpy(float), uf.y.to_numpy(float)
    day, mon = rome_day(uf.MVT_TIME_UTC_mvt).to_numpy(), uf.month.to_numpy(dtype=int)
    jj = np.isin(mon, DECISION_MONTHS)
    P, PS = lomo_fit(uf, lf, months, ARMS, seeds, log)
    st = _align_stored(uf, stored, "lomo")
    rep = replicate_baseline({int(s): PS["R"][int(s)] for s in seeds},
                             {int(s): st[f"p_R_seed{int(s)}"].to_numpy(float) for s in seeds}, REPLICATION_ATOL)
    log(f"replication guard PASSED: {rep['max_abs_delta_by_seed']}")
    nf, p_st = st.nf.to_numpy(float), st.p_R.to_numpy(float)
    se_inc, pred_inc = se_rows(p_st, sp, nf, y), rf.mixture(p_st, sp, nf)
    oracle = _oracle_blocks(cls, sp, nf, y, se_inc, pred_inc, day, mon, n_boot, BOOT_SEED)
    lomo = {}
    for a in ARMS:
        pa = P[a]
        se_a, pred_a = se_rows(pa, sp, nf, y), rf.mixture(pa, sp, nf)
        lomo[a] = {"auc_pooled": auc(fill, pa), "auc_by_seed": {str(int(s)): auc(fill, PS[a][int(s)]) for s in seeds},
                   "auc_by_month": {str(m): auc(fill[mon == m], pa[mon == m]) for m in months},
                   "auc_janjul": auc(fill[jj], pa[jj]),
                   "auc_sp2_weighted": weighted_auc(fill[sp > 0], pa[sp > 0], sp[sp > 0] ** 2),
                   "months_auc_above_R": int(sum(auc(fill[mon == m], pa[mon == m]) > auc(fill[mon == m], P["R"][mon == m])
                                                 for m in months)),
                   "value_janjul": value_block(se_inc[jj], se_a[jj], pred_inc[jj], pred_a[jj], day[jj], mon[jj], sp[jj], n_boot, BOOT_SEED),
                   "value_12m": value_block(se_inc, se_a, pred_inc, pred_a, day, mon, sp, n_boot, BOOT_SEED)}
        log(f"  {a:5s} LOMO AUC {lomo[a]['auc_pooled']:.4f} janjul gain {lomo[a]['value_janjul']['board_scaled']:+.1f} board")
    PF = forward_fit(uf, lf, months, ARMS, seeds, log)
    fw = np.isin(mon, FORWARD_MONTHS)
    se_fr, pred_fr = se_rows(PF["R"], sp, nf, y), rf.mixture(PF["R"], sp, nf)
    forward = {}
    for a in ARMS:
        se_fa, pred_fa = se_rows(PF[a], sp, nf, y), rf.mixture(PF[a], sp, nf)
        forward[a] = {"auc_pooled_7_12": auc(fill[fw], PF[a][fw]), "auc_july": auc(fill[mon == 7], PF[a][mon == 7]),
                      "auc_by_month": {str(m): auc(fill[mon == m], PF[a][mon == m]) for m in FORWARD_MONTHS},
                      "value_vs_forward_R": value_block(se_fr[fw], se_fa[fw], pred_fr[fw], pred_fa[fw], day[fw], mon[fw], sp[fw], n_boot, BOOT_SEED),
                      "value_july_vs_forward_R": value_block(se_fr[mon == 7], se_fa[mon == 7], pred_fr[mon == 7], pred_fa[mon == 7],
                                                             day[mon == 7], mon[mon == 7], sp[mon == 7], n_boot, BOOT_SEED)}
    w25 = witness_table(arr25, pool)
    w26 = witness_table(arr26, pool if pool26 is None else pool26)
    witness = {"y2025": w25, "y2026": w26, "clause_both_ratios_above_one": witness_clause(w25, w26)}
    D = lomo[DECISIONAL_ARM]
    clauses = {"c1_auc_ge_0_95": bool(D["auc_pooled"] >= AUC_TRUE),
               "c2_ci_excludes_zero": bool(D["value_janjul"]["c2_ci_excludes_zero"]),
               "c3_both_months_positive": bool(D["value_janjul"]["c3_both_months_positive"]),
               "c4_minus_top3_positive": bool(D["value_janjul"]["c4_minus_top3_positive"]),
               "c5_forward_positive": bool(forward[DECISIONAL_ARM]["value_vs_forward_R"]["gain_sse"] > 0
                                           and forward[DECISIONAL_ARM]["auc_pooled_7_12"] >= forward["R"]["auc_pooled_7_12"])}
    verdict = decide(D["auc_pooled"], clauses)
    if preds_out is not None:
        pr = uf[["MVT_ID_mvt", "month", "sp", "y"]].copy()
        pr["day"], pr["fill"], pr["cls"], pr["nf"], pr["p_stored_R"] = day, fill, cls, nf, p_st
        for a in ARMS:
            pr[f"p_{a}"] = P[a]
            pr[f"p_fwd_{a}"] = PF[a]
            for s in seeds:
                pr[f"p_{a}_seed{int(s)}"] = PS[a][int(s)]
        pathlib.Path(preds_out).parent.mkdir(parents=True, exist_ok=True)
        pr.to_parquet(preds_out)
    return {"kind": "run", "smoke": bool(smoke), "seeds": [int(s) for s in seeds], "n_boot": int(n_boot),
            "prereg_sha256": (hashlib.sha256(PREREG.read_bytes()).hexdigest() if PREREG.exists() else None),
            "bars": {"auc_true": AUC_TRUE, "auc_false": AUC_FALSE, "atol": REPLICATION_ATOL, "k_range": list(K_RANGE)},
            "n_rows": int(len(uf)), "n_janjul": int(jj.sum()), "fill_rate": float(fill.mean()),
            "label_integrity": integrity, "feature_summary": fs, "replication": rep,
            "oracle": oracle, "lomo": lomo, "forward": forward, "witness": witness, "clauses": clauses, "verdict": verdict,
            "decisional_arm": DECISIONAL_ARM, "decisional_auc": float(D["auc_pooled"]),
            "git_sha": _git_sha(), "versions": _versions(), "wall_s": round(time.time() - t0, 1),
            "peak_rss_gb": round(_peak_rss_gb(), 2), "preds_parquet": (str(preds_out) if preds_out else None)}


# ================================================================== synthetic inputs (smoke + tests)

def _stored_from_refit(unm, lirf, seeds) -> pd.DataFrame:
    """A stored-artefact stand-in in rome_fill_preds.parquet's schema: the R recipe refit under LOMO (fold 'lomo')
    and on months 1, 7 from the other ten (fold 'A'); nf = the training fold's non-fill median."""
    u = unm[unm.ADEP_mvt == "LIRF"].reset_index(drop=True)
    months = sorted(int(m) for m in u.month.unique())
    L = labels(u)
    parts = []
    groups = [(m, [o for o in months if o != m], "lomo") for m in months] + [
        (list(DECISION_MONTHS), [o for o in months if o not in DECISION_MONTHS], "A")]
    for held, others, fold in groups:
        held = [held] if isinstance(held, int) else held
        te_mask = u.month.isin(held).to_numpy()
        tr = rf.rome_rows(u[~te_mask], lirf[~lirf.month.isin(held)], others)
        te = u[te_mask]
        tr_u = u[~te_mask]
        nf = float(tr_u.y[~L.fill.to_numpy()[~te_mask]].median())
        rec = pd.DataFrame({"MVT_ID_mvt": te.MVT_ID_mvt.to_numpy(), "fold": fold, "month": te.month.to_numpy(),
                            "y": te.y.to_numpy(float), "sp": te.sp.to_numpy(float), "nf": nf})
        p_mean, by = fit_arm(tr, te, seeds, (), ())
        for s in seeds:
            rec[f"p_R_seed{int(s)}"] = by[int(s)]
        rec["p_R"] = p_mean
        rec["R"] = rf.mixture(p_mean, rec.sp.to_numpy(float), rec.nf.to_numpy(float))
        parts.append(rec)
    return pd.concat(parts, ignore_index=True)


def synthetic_inputs(seed: int = 1, n_per_month: int = 160, n_matched_per_month: int = 120, seeds=(0,)) -> tuple:
    """(unm, lirf, pool, stored, arr25, arr26) shaped like the real inputs: LIRF unmatched rows with a NULL type and
    segment (as in reality), ARR legs at LIRF for most of their callsigns (so imputation and the witness fire),
    a second ARR set standing in for 2026, and a stored artefact built from the R recipe itself."""
    unm, lirf = sf.synthetic_frame(n_per_month=n_per_month, seed=seed, n_matched_per_month=n_matched_per_month)
    rng = np.random.default_rng(seed)
    unm = unm.copy()
    lu = (unm.ADEP_mvt == "LIRF").to_numpy()
    unm.loc[lu, "AIRCRAFT_TYPE_mvt"] = None
    for d in (unm, lirf):
        d["AIRCRAFT_TYPE_flt"] = pd.Series([None] * len(d), index=d.index, dtype="str")
    lirf["AIRCRAFT_TYPE_flt"] = lirf.AIRCRAFT_TYPE_mvt.astype("str")

    def arr_legs(src: pd.DataFrame, id_offset: float, share: float, rng) -> pd.DataFrame:
        pick = src[rng.random(len(src)) < share]
        t = pick.SCHED_TIME_UTC_mvt - pd.Timedelta(hours=2)
        sched = t - pd.Timedelta(minutes=10)
        copy = rng.random(len(pick)) < 0.3
        one_off = ~pick.FLIGHT_mvt.isin(lirf.FLIGHT_mvt)          # never a LIRF departure callsign -> low n_cs
        copy = copy | (one_off.to_numpy() & (rng.random(len(pick)) < 0.5))
        block = np.where(copy, sched, sched + pd.Timedelta(minutes=15))
        return pd.DataFrame({"MVT_ID_mvt": id_offset + np.arange(len(pick), dtype="float64"), "PHASE_mvt": "ARR",
                             "ADEP_mvt": rng.choice(["EGLL", "LFPG", "EDDF"], len(pick)), "ADES_mvt": "LIRF",
                             "FLIGHT_mvt": pick.FLIGHT_mvt.to_numpy(), "MVT_TIME_UTC_mvt": t.to_numpy(),
                             "SCHED_TIME_UTC_mvt": sched.to_numpy(), "BLOCK_TIME_UTC_mvt": block,
                             "AIRCRAFT_TYPE_mvt": rng.choice(["A320", "B738", "E190"], len(pick)),
                             "AIRCRAFT_TYPE_flt": None,
                             "MARKET_SEGMENT_flt": rng.choice(["Mainline", "Lowcost", "Business"], len(pick)),
                             "AOBT_3_flt": pd.NaT})

    src = unm[lu]
    arr25 = arr_legs(src, 2e7, 0.7, rng)
    arr26 = arr_legs(src, 3e7, 0.5, np.random.default_rng(seed + 1))
    dep = pd.concat([unm, lirf], ignore_index=True)[POOL_COLS]
    pool = pd.concat([dep, arr25[POOL_COLS], arr26[POOL_COLS]], ignore_index=True)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"):
        pool[c] = _utc(pool[c])
        arr25[c] = _utc(arr25[c])
        arr26[c] = _utc(arr26[c])
    stored = _stored_from_refit(unm, lirf, seeds)
    return unm, lirf, pool, stored, arr25, arr26


# ================================================================== entry point

def _logger(path: pathlib.Path, t0: float):
    fh = open(path, "a")

    def log(msg):
        line = f"[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}Z +{time.time() - t0:6.0f}s] {msg}"
        print(line, flush=True)
        fh.write(line + "\n")
        fh.flush()
    return log, fh


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["baseline", "run", "smoke"])
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--n-per-month", type=int, default=160)
    ap.add_argument("--n-matched-per-month", type=int, default=120)
    ap.add_argument("--synthetic-seed", type=int, default=1)
    ap.add_argument("--no-refit", action="store_true", help="baseline: read the artefact and price the oracle without any fit")
    a = ap.parse_args(argv)
    seeds = tuple(int(s) for s in a.seeds.split(","))
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tag = {"baseline": "csf2_baseline_oracle", "run": "csf2_report", "smoke": "csf2_smoke_report"}[a.cmd]
    report_path = out / f"{tag}.json"
    if report_path.exists():
        raise SystemExit(f"{report_path} exists - a rerun must not silently become a second look at the same bar")
    log, fh = _logger(out / f"{tag}.log", t0)
    lock = None
    if a.cmd != "smoke":
        lock = acquire_heavy_lock(LOCK_PATH)
        rss_guard(RSS_ABORT_GIB, 5.0)
        log(f"heavy lock acquired ({LOCK_PATH}); RSS abort at {RSS_ABORT_GIB} GiB")
    try:
        if a.cmd == "smoke":
            log("SMOKE - synthetic inputs, code path only, NOT a result")
            unm, lirf, pool, stored, arr25, arr26 = synthetic_inputs(a.synthetic_seed, a.n_per_month,
                                                                     a.n_matched_per_month, seeds)
            R = run(unm, lirf, pool, stored, arr25, arr26, seeds=seeds, n_boot=a.n_boot, log=log, smoke=True,
                    preds_out=out / "csf2_smoke_preds.parquet")
            hashlog = out / "csf2_smoke_hashes.log"
            label = "SMOKE"
        else:
            unm, lirf = rf.load_frames(log)
            stored = load_stored(STORED)
            if a.cmd == "baseline":
                R = baseline(unm, lirf, stored, seeds=seeds, n_boot=a.n_boot, log=log, refit=not a.no_refit)
                label = "BASELINE-JSON" + ("-NOREFIT" if a.no_refit else "")
            else:
                files = sorted((ROOT / "data" / "raw").glob("training_2025-*.parquet"))
                if len(files) != 12:
                    raise FileNotFoundError(f"expected 12 training months, found {len(files)}")
                log("loading the 2025 movement pool (12 files, POOL_COLS)")
                pool = load_pool(files)
                log(f"pool: {len(pool):,} movements; peak RSS {_peak_rss_gb():.2f} GB")
                pool26 = load_pool([ROOT / "data" / "raw" / "ranking.parquet"])
                arr25 = pool[(pool.PHASE_mvt == "ARR") & (pool.ADES_mvt == "LIRF")]
                arr26 = pool26[(pool26.PHASE_mvt == "ARR") & (pool26.ADES_mvt == "LIRF")]
                log(f"witness pools: 2025 ARR@LIRF {len(arr25):,}, 2026 ARR@LIRF {len(arr26):,}")
                R = run(unm, lirf, pool, stored, arr25, arr26, seeds=seeds, n_boot=a.n_boot, log=log, smoke=False,
                        pool26=pool26, preds_out=out / "csf2_preds.parquet")
                label = "RESULT-JSON"
            hashlog = HASHLOG
        write_report(R, report_path)
        line = append_hash(hashlog, label, report_path, note=f"{a.cmd}; seeds {list(seeds)}; n_boot {a.n_boot}")
        for ln in (summary_lines(R) if a.cmd != "baseline" else [f"baseline AUC {R['auc']}", f"replication {R['replication']}"]):
            log(ln)
        log(line)
        log(f"json -> {report_path}   wall {R['wall_s']:.0f}s   peak RSS {R['peak_rss_gb']} GB")
        if a.cmd == "smoke":
            log("SMOKE - synthetic inputs, code path only, NOT a result")
    finally:
        fh.close()
        if lock is not None:
            release_heavy_lock(lock)
    return 0


if __name__ == "__main__":
    sys.exit(main())
