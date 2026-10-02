"""R12 validation harness — the 12-month ADS-B model against the 2026 coverage mix.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/r12_validate.py --split holdout --arms r12,control,yhead \
        --tables data/r12/tables_<stamp> --out data/r12/val_<stamp>_holdout [--seeds 0,1,2] [--rss-limit-gib 8] [--smoke]
    /opt/homebrew/bin/python3.11 -B scripts/analysis/r12_validate.py --split forward --arms r12,control ...
    (heavy runs go through scripts/analysis/r12_launch.py)

The frame is the production refit's (lgbm_submit.load_frames: the twelve 2025 training months + the 339,551 matched
ranking rows, queue / order / weather blocks), so ONE fit scores the 2025 test rows AND the 2026 rows. In-fold target
encodings are fitted on the split's training rows only (stand_ab.infold_encodings, as lgbm_fold.load_fold does); the
ADS-B columns come from the R12 tables: training rows by (cache month, position in the month's cache file), ranking rows
by MVT id (their month 1 / 7 would collide with January / July 2025).
Splits (ranking rows are never fit, stopping or test rows):
  holdout  fold A: holdout Jan + Jul 2025 (the board's months), early stopping on 3 + 9, fit on the other eight.
  forward  fit Feb-Jun + Aug-Nov (early stopping 3 + 9 carved out of them), test December (the 2025 month closest to the
           2026 reception); January and July are in neither set.
  feb      test February 2025 (the only 2025 LTFM snowstorm, 19-22 Feb); every other 2025 month trains
           (Jan / Jul included, as in the served refit), early stopping on 3 + 9.
  decmar   (controls for feb) test December + March 2025 (cold, not stormy); February trains; early stopping on 9 only.
Arms (design = nsc_fold.FEATS_NSC_FW, the 90-column v34 FW design without the label-seconds columns, + ADSB_COLS):
  r12         delta target, FW-NSC + ADS-B block                         candidate A
  r12_noeobt  r12 with EOBT_COLS NaN on every LFPG / LTFM row            candidate A' (the 2026 EOBT_1 drift there)
  janjul_adsb r12 with the ADS-B block NaN (+ adsb_day_present 0) on every training row outside Jan / Jul: today's
              situation. NOTE: in the holdout and forward splits Jan / Jul are never fit rows, so this arm trains with NO
              ADS-B and is in effect the no-ADS-B control; the ten extra months' value shows at the STACKED level
              (r12_stack: r12 F + stack vs janjul_adsb F + stack), where the stack is trained on Jan / Jul as v34's is.
  control     delta target, FW-NSC alone (same code, same fold)          the old no-ADS-B control
  yhead       y target, FW-NSC + ADS-B block                             with r12 -> candidate B = blend_taxi_times(yhead, r12)
  r12_move    r12 without HEARD_COLS (R13 AMEND-A2: movement / timing ADS-B columns only)
  cat         delta target, FW-NSC + ADS-B block, CatBoost (R13; fitted by r13_cat.CatFitter in its own process, BC-5)
Every LightGBM arm: lgbm_fold.fit_arm (lgbm_submit.P, ES seed 0, refit seeds, pooled mean, no per-airport blend); the
cat arm: r13_cat.CatFitter, the same call contract.

Scoring (units: board MSE = SSE / 344,841; gain = comparator SSE - candidate SSE, POSITIVE = the candidate is better):
  * RMSE unweighted and REWEIGHTED: per airport x month cell the coverage==3 share of the test rows is reweighted to
    the 2026 serve table's share (forward: the airport's pooled Jan+Jul 2026 share), cell mass kept. A cell whose
    2025 share is 0 (or 1) cannot carry any other share: weight 1, status "unrepresentable", listed in the report.
  * paired gains with a day-block bootstrap CI: every arm vs control, every arm vs each comparator on common rows.
  * comparators (holdout only): PIPE_M = cln_build.load_comparators()["M"] (the pipeline's matched lane, 334,132 rows,
    41,969.3 bmse) by MVT id, and the v34 step-1 F-NSC out-of-fold record (the ADS-B stage's base F; the composed v34
    matched lane has no stored out-of-fold replay) by (month, position).
  * label-free 2026 sensitivity (no 2026 label is read; none exists in the frame): per arm the predicted-taxi shift
    2026 vs the 2025 test rows per airport x month (forward: per airport); the share of 2026 rows whose EOBT-derived
    inputs lie outside the 2025 training rows' range at the same airport; r12 minus every other arm on the 2026 rows.
LightGBM only (no torch: BC-5).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import importlib.util
import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load("r12_build", "scripts/analysis/r12_build.py")
NSC = _load("nsc_fold", "scripts/analysis/nsc_fold.py")
LF = NSC.LF

BOARD_N = 344_841.0
BOOT_SEED, N_BOOT = 20260925, 1000
BASE = list(NSC.FEATS_NSC_FW)
ADSB_COLS = list(B.FEATURES) + ["coverage", "adsb_day_present"]
ARMS = ("r12", "control", "yhead", "r12_noeobt", "janjul_adsb", "cat", "r12_move")
TARGETS = {"r12": "delta", "control": "delta", "yhead": "y", "r12_noeobt": "delta", "janjul_adsb": "delta",
           "cat": "delta", "r12_move": "delta"}                               # cat: R13 CatBoost on the r12 design (scripts/analysis/r13_cat.py)
JANJUL = (1, 7)
#: R13 AMEND-A2 (WHERE an aircraft was heard does not transfer across receiver-network changes; HOW it moved does): reception / sample-count / first-heard / join-quality / stand-visibility columns; arm r12_move drops them
HEARD_COLS = ("coverage", "adsb_day_present", "joined", "join_callsign", "join_mvt_anchored", "mvt_anchor_gap_s",
              "first_seen_rel_mvt", "n_ground", "ground_rate_pm", "mlat_share", "first_ground_rel_mvt",
              "first_ground_gs", "first_ground_slow", "first_ground_rel_aobt3", "ground_seen_mvt_m10",
              "moving_mvt_m10", "stand_visible", "stand_ref_parked_share", "stand_others_n",
              "first_ground_to_stand_m", "travel_s_first_to_stand")
CONTROL_ARMS = ("janjul_adsb", "control")               # janjul_adsb (ADS-B only in Jan / Jul) is the key control
EOBT_COLS = [c for c in BASE if "eobt" in c]           # the EOBT_1-derived inputs (their 2026 drift at LFPG / LTFM)
MASK_AIRPORTS = ("LFPG", "LTFM")
SPLITS = ("holdout", "forward", "refit", "feb", "decmar")
FORWARD_FIT = (2, 3, 4, 5, 6, 8, 9, 10, 11)
STEP1_RECORD = ROOT / "data" / "nsc_v34" / "step1_f_fold" / "fold_preds_queue_order_nsc.parquet"
SMOKE_MONTHS = {"holdout": (1, 2, 3), "forward": (3, 11, 12), "refit": (1, 2, 3), "feb": (1, 2, 3),
                "decmar": (2, 9, 12)}
SCREEN_LR = 0.05
KEY_MULT = 10_000_000                      # month * KEY_MULT + pos: a month file never holds 10 M rows


# ---------------------------------------------------------------- cells, target shares, weights

def cell_keys(ap, month, pooled: bool = False) -> np.ndarray:
    ap = np.asarray(ap).astype(str)
    if pooled:
        return ap
    return np.char.add(np.char.add(ap, "|"), np.asarray(month).astype(int).astype(str))


def target_shares(serve: pd.DataFrame, pooled: bool) -> dict:
    """coverage == 3 share of the 2026 serve rows per airport x month (pooled: per airport over Jan + Jul)."""
    c3 = (serve.coverage.to_numpy() == 3).astype(float)
    keys = cell_keys(serve.ap.to_numpy(), serve.month.to_numpy(), pooled)
    s = pd.Series(c3).groupby(keys).mean()
    return {str(k): float(v) for k, v in s.items()}


def coverage_weights(keys, cov3, target: dict, allow_missing: bool = False) -> tuple:
    """Per-row weights: in each cell, coverage-3 rows get p_t / p_s and the others (1 - p_t) / (1 - p_s), so the
    weighted coverage-3 share equals the 2026 target and the cell's weight sum equals its row count. A cell with no
    target is refused, unless allow_missing (smoke tables only): weight 1, status no_target."""
    keys, cov3 = np.asarray(keys).astype(str), np.asarray(cov3)
    if len(keys) != len(cov3):
        raise ValueError(f"length mismatch: {len(keys)} keys, {len(cov3)} cov3 flags")
    if not np.isin(cov3, (0, 1)).all():
        raise ValueError("cov3 must be 0/1")
    cov3 = cov3.astype(bool)
    w = np.ones(len(keys))
    cells = {}
    for k in np.unique(keys):
        m = keys == k
        if k not in target:
            if not allow_missing:
                raise KeyError(f"no 2026 target share for cell {k}")
            cells[k] = {"n": int(m.sum()), "p_source": float(cov3[m].mean()), "p_target": None, "status": "no_target",
                        "w_cov3": 1.0, "w_other": 1.0, "n_eff": float(m.sum())}
            continue
        n, ps, pt = int(m.sum()), float(cov3[m].mean()), float(target[k])
        if ps == pt:
            status, wc, wo = "matched", 1.0, 1.0
        elif ps in (0.0, 1.0):
            status, wc, wo = "unrepresentable", 1.0, 1.0
        else:
            status, wc, wo = "reweighted", pt / ps, (1.0 - pt) / (1.0 - ps)
        w[m & cov3], w[m & ~cov3] = wc, wo
        wm = w[m]
        cells[k] = {"n": n, "p_source": ps, "p_target": pt, "status": status, "w_cov3": wc, "w_other": wo,
                    "n_eff": float(wm.sum() ** 2 / (wm ** 2).sum())}
    return w, cells


def weighted_rmse(y, pred, w=None) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    if not (np.isfinite(y).all() and np.isfinite(pred).all()):
        raise ValueError("non-finite value in an RMSE operand")
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    return float(np.sqrt((w * (y - pred) ** 2).sum() / w.sum()))


def paired_gain(y, cand, comp, blocks, w=None, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """(SSE_comp - SSE_cand) / 344,841 on the rows where the comparator is finite, optionally weighted, with a
    day-block bootstrap 95 % interval. Positive = the candidate is better."""
    y, cand, comp = (np.asarray(a, float) for a in (y, cand, comp))
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    m = np.isfinite(comp)
    if not np.isfinite(cand[m]).all():
        raise ValueError("non-finite candidate prediction")
    g = w[m] * ((comp[m] - y[m]) ** 2 - (cand[m] - y[m]) ** 2)
    per = pd.Series(g).groupby(np.asarray(blocks)[m]).sum().to_numpy()
    rng = np.random.default_rng(seed)
    draws = per[rng.integers(0, len(per), size=(n_boot, len(per)))].sum(1) / BOARD_N
    return {"gain_bmse": float(g.sum() / BOARD_N), "ci95": [float(np.percentile(draws, 2.5)),
                                                             float(np.percentile(draws, 97.5))],
            "n": int(m.sum()), "n_blocks": int(len(per)),
            "rmse_candidate": weighted_rmse(y[m], cand[m], w[m]), "rmse_comparator": weighted_rmse(y[m], comp[m], w[m])}


# ---------------------------------------------------------------- the ADS-B block on a load_fold frame

def month_positions(month) -> np.ndarray:
    """Position of each row inside its month's cache file; the frame must be in cache-file order (months ascending,
    each month contiguous), which lgbm_submit.training_frames guarantees (sorted file names, concatenated once)."""
    month = np.asarray(month).astype(int)
    if len(month) and (np.diff(month) < 0).any():
        raise ValueError("frame rows are not in cache-file order (month decreases)")
    starts = pd.Series(np.arange(len(month))).groupby(month).transform("min").to_numpy()
    return np.arange(len(month)) - starts


def attach_adsb(d: pd.DataFrame, table: pd.DataFrame, strict: bool) -> dict:
    """Add ADSB_COLS to `d` (in place) from the R12 training table by (month, position). strict (every real run): each
    month's frame row count must equal the table's and every row must join. Non-strict (smoke only): unjoined rows get
    NaN features, NaN coverage and adsb_day_present 0, and are counted."""
    month = d.month.to_numpy().astype(int)
    pos = month_positions(month)
    if strict:
        tc = table.groupby("month").size()
        for m, nf in pd.Series(month).value_counts().sort_index().items():
            nt = int(tc.get(m, 0))
            if nt != nf:
                raise ValueError(f"month {m}: frame has {nf} rows, table {nt}")
    t = table.set_index(table.month.to_numpy().astype("int64") * KEY_MULT + table.pos.to_numpy().astype("int64"))
    j = t.reindex(month.astype("int64") * KEY_MULT + pos)
    absent = j.MVT_ID_mvt.isna().to_numpy()
    if strict and absent.any():
        raise ValueError(f"{int(absent.sum())} frame rows have no table row")
    for c in B.FEATURES:
        d[c] = j[c].to_numpy(dtype="float64")
    d["coverage"] = j.coverage.to_numpy(dtype="float64")
    d["adsb_day_present"] = np.where(absent, 0.0, j.adsb_day_present.to_numpy(dtype="float64"))
    keys = pd.DataFrame({"month": month, "pos": pos, "MVT_ID_mvt": j.MVT_ID_mvt.to_numpy(dtype="float64"),
                         "mvt_day": j.mvt_day.to_numpy(dtype=object), "coverage": d.coverage.to_numpy()})
    return {"n_without_table": int(absent.sum()), "pos": pos, "keys": keys}


def rows_to_month_pos(rows, month_counts: dict) -> tuple:
    """Global row indices over the concatenated months (ascending) -> (month, position in the month)."""
    months = np.array(sorted(month_counts))
    sizes = np.array([month_counts[m] for m in months])
    ends = np.cumsum(sizes)
    rows = np.asarray(rows).astype("int64")
    if len(rows) and (rows.min() < 0 or rows.max() >= ends[-1]):
        raise ValueError(f"row index out of range [0, {ends[-1]})")
    i = np.searchsorted(ends, rows, side="right")
    return months[i], rows - (ends[i] - sizes[i])


def adsb_by_id(ids, table: pd.DataFrame, strict: bool) -> pd.DataFrame:
    """The serve table's ADSB_COLS + keys for ranking ids, in the ids' order (ranking rows may be a sample)."""
    ids = np.asarray(ids, dtype="float64")
    j = table.set_index("MVT_ID_mvt").reindex(ids)
    absent = j.month.isna().to_numpy()
    if strict and absent.any():
        raise ValueError(f"{int(absent.sum())} serve ids have no table row")
    out = pd.DataFrame({"month": j.month.to_numpy(dtype="float64"), "pos": j.pos.to_numpy(dtype="float64"),
                        "MVT_ID_mvt": ids, "mvt_day": j.mvt_day.to_numpy(dtype=object),
                        "coverage": j.coverage.to_numpy(dtype="float64")})
    for c in B.FEATURES:
        out[c] = j[c].to_numpy(dtype="float64")
    out["adsb_day_present"] = np.where(absent, 0.0, j.adsb_day_present.to_numpy(dtype="float64"))
    return out


def attach_all(d: pd.DataFrame, is_rank, rank_ids, train_table: pd.DataFrame, serve_table: pd.DataFrame,
               strict: bool) -> dict:
    """ADSB_COLS on a load_frames frame (training rows first, then ranking rows): training rows by (month, position)
    from the training table, ranking rows by MVT id from the serve table. Returns keys for every row."""
    is_rank = np.asarray(is_rank, bool)
    n_tr = int((~is_rank).sum())
    if is_rank[:n_tr].any():
        raise ValueError("ranking rows must come last in the frame")
    tr = pd.DataFrame({"month": d.month.to_numpy()[:n_tr]})
    info = attach_adsb(tr, train_table, strict)
    rk = adsb_by_id(rank_ids, serve_table, strict)
    if len(rk) != len(d) - n_tr:
        raise ValueError(f"{len(rk)} ranking ids for {len(d) - n_tr} ranking rows")
    for c in ADSB_COLS:
        d[c] = np.concatenate([tr[c].to_numpy(dtype="float64"), rk[c].to_numpy(dtype="float64")])
    keys = pd.concat([info["keys"], rk[["month", "pos", "MVT_ID_mvt", "mvt_day", "coverage"]]], ignore_index=True)
    return {"keys": keys, "n_without_table": info["n_without_table"] + int((rk.adsb_day_present == 0).sum())}


def cache_month_counts(stand_dir=None) -> dict:
    """{cache month: row count} from the stand cache files' parquet metadata (the row layout of every fold record)."""
    import pyarrow.parquet as pq
    d = pathlib.Path(stand_dir) if stand_dir is not None else B.STAND_DIR
    paths = sorted(d.glob("training_2025-*.parquet")) if d.exists() else []
    if not paths:
        raise FileNotFoundError(f"no training caches under {d}")
    out = {}
    for p in paths:
        m = pd.read_parquet(p, columns=["month"]).month.unique()
        if len(m) != 1:
            raise ValueError(f"{p.name}: more than one month")
        out[int(m[0])] = int(pq.read_metadata(p).num_rows)
    return out


# ---------------------------------------------------------------- splits, designs, candidates

def split_holdout(month):
    """Fold A (lgbm_fold.fold_masks): holdout 1 + 7, early stopping 3 + 9, fit on the rest."""
    return LF.fold_masks(month)


def split_refit(month):
    """The serve-side refit: every 2025 month trains (early stopping on 3 + 9, as lgbm_submit.split_masks), no test."""
    month = np.asarray(month)
    tr = np.ones(len(month), bool)
    es = np.isin(month, LF.L.ES_MONTHS)
    return np.zeros(len(month), bool), tr, tr & ~es, es


def split_forward(month):
    month = np.asarray(month)
    te = month == 12
    if not te.any():
        raise ValueError("no December rows to test")
    tr = np.isin(month, FORWARD_FIT)
    es = tr & np.isin(month, LF.L.ES_MONTHS)
    fit = tr & ~es
    return te, tr, fit, es


def split_months_out(month, test_months):
    """Test `test_months`; every other 2025 month trains; early stopping on the lgbm_submit ES months that are not
    test months. Refuses a split with no test, stopping or fit rows."""
    month = np.asarray(month)
    te = np.isin(month, test_months)
    if not te.any():
        raise ValueError(f"no test rows for months {tuple(test_months)}")
    tr = ~te
    es = tr & np.isin(month, LF.L.ES_MONTHS)          # a test month is never in tr, so never a stopping row
    fit = tr & ~es
    if not es.any():
        raise ValueError("no early-stopping rows left")
    if not fit.any():
        raise ValueError("no fit rows left")
    return te, tr, fit, es


def split_feb(month):
    return split_months_out(month, (2,))


def split_decmar(month):
    return split_months_out(month, (3, 12))


SPLIT_FNS = {"holdout": split_holdout, "forward": split_forward, "refit": split_refit, "feb": split_feb,
             "decmar": split_decmar}
POOLED_SPLITS = ("forward", "feb", "decmar")      # no 2026 month to key a coverage target on: pool per airport


def design_for(arm: str) -> list:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; arms are {ARMS}")
    cols = BASE if arm == "control" else BASE + ADSB_COLS
    if arm == "r12_move":
        cols = BASE + [c for c in ADSB_COLS if c not in HEARD_COLS]
    bad = [c for c in cols if c.endswith("_sec")]
    if bad or len(set(cols)) != len(cols):
        raise AssertionError(f"design for {arm}: seconds columns {bad} or duplicates")
    return list(cols)


def mask_eobt(X, feats, ap_code, airports, mask_airports=MASK_AIRPORTS) -> tuple:
    """A copy of X with EOBT_COLS set NaN on every row at `mask_airports` (arm r12_noeobt); X itself is untouched."""
    codes = [i for i, a in enumerate(airports) if a in mask_airports]
    rows = np.isin(np.asarray(ap_code), codes)
    X2 = np.array(X, copy=True)
    for c in EOBT_COLS:
        X2[rows, feats.index(c)] = np.nan
    return X2, int(rows.sum())


def mask_janjul(X, feats, month, rank, keep_months=JANJUL) -> tuple:
    """A copy of X for arm janjul_adsb (today's situation): on TRAINING rows outside `keep_months` every ADS-B column is
    NaN and adsb_day_present is 0; January / July training rows and the 2026 ranking rows keep theirs."""
    rows = ~np.asarray(rank, bool) & ~np.isin(np.asarray(month), keep_months)
    X2 = np.array(X, copy=True)
    for c in ADSB_COLS:
        X2[rows, feats.index(c)] = 0.0 if c == "adsb_day_present" else np.nan
    return X2, int(rows.sum())


# ---------------------------------------------------------------- label-free 2026 sensitivity

def _q(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def prediction_shift(ref: pd.DataFrame, new: pd.DataFrame, pooled: bool) -> dict:
    """Per cell of the 2026 rows (`new`: ap, month, taxi, proxy): predicted-taxi and proxy statistics vs the 2025 test
    rows of the same cell (`ref`). A shift mixes real 2026 change with input drift; read it with arm_differences."""
    rk, nk = cell_keys(ref.ap, ref.month, pooled), cell_keys(new.ap, new.month, pooled)
    out = {}
    for k in np.unique(nk):
        r, n = ref[rk == k], new[nk == k]
        c = {"n_2025": int(len(r)), "n_2026": int(len(n))}
        for tag, f in (("2025", r), ("2026", n)):
            t = f.taxi.to_numpy(float)
            c[f"mean_taxi_{tag}"] = float(t.mean()) if len(t) else None
            c[f"median_taxi_{tag}"] = _q(t, 50)
            c[f"p90_taxi_{tag}"] = _q(t, 90)
            c[f"mean_proxy_{tag}"] = float(f.proxy.to_numpy(float).mean()) if len(f) else None
        both = c["n_2025"] > 0
        c["shift_mean_taxi"] = c["mean_taxi_2026"] - c["mean_taxi_2025"] if both else None
        c["shift_mean_proxy"] = c["mean_proxy_2026"] - c["mean_proxy_2025"] if both else None
        out[str(k)] = c
    return out


def input_range(X, feats, cols, ref_mask, new_mask, ap, month, pooled: bool) -> dict:
    """Per cell of the 2026 rows and input column: the share of 2026 rows outside the 2025 TRAINING rows' [min, max] and
    [p0.5, p99.5] at the same airport (finite values; NaN rows are counted separately)."""
    ap = np.asarray(ap).astype(str)
    keys = cell_keys(ap, month, pooled)
    out = {}
    for k in np.unique(keys[new_mask]):
        cell = new_mask & (keys == k)
        a = ap[cell][0]
        out[str(k)] = {}
        for c in cols:
            j = feats.index(c)
            r = X[ref_mask & (ap == a), j].astype("float64")
            r = r[np.isfinite(r)]
            v = X[cell, j].astype("float64")
            fin = v[np.isfinite(v)]
            rec = {"n_2026": int(cell.sum()), "share_nan_2026": float(1 - len(fin) / max(len(v), 1)),
                   "median_2026": _q(fin, 50), "n_2025": int(len(r)), "median_2025": _q(r, 50),
                   "min_2025": float(r.min()) if len(r) else None, "max_2025": float(r.max()) if len(r) else None,
                   "p005_2025": _q(r, 0.5), "p995_2025": _q(r, 99.5)}
            if len(r) and len(fin):
                rec["share_outside_minmax"] = float(((fin < r.min()) | (fin > r.max())).mean())
                rec["share_outside_p005_p995"] = float(((fin < rec["p005_2025"]) | (fin > rec["p995_2025"])).mean())
            else:
                rec["share_outside_minmax"] = rec["share_outside_p005_p995"] = None
            out[str(k)][c] = rec
    return out


def arm_differences(taxis: dict, ref: pd.DataFrame, pooled: bool, pairs) -> dict:
    """Per cell: mean and RMS of taxi_a - taxi_b (rows of `ref`, aligned with every array in `taxis`)."""
    keys = cell_keys(ref.ap, ref.month, pooled)
    out = {}
    for a, b in pairs:
        d = np.asarray(taxis[a], float) - np.asarray(taxis[b], float)
        out[f"{a}_minus_{b}"] = {str(k): {"n": int((keys == k).sum()), "mean": float(d[keys == k].mean()),
                                          "rms": float(np.sqrt((d[keys == k] ** 2).mean()))} for k in np.unique(keys)}
    return out


# ---------------------------------------------------------------- the run

def json_clean(o):
    """JSON-ready copy; a non-finite float is a defect in a report, never a value."""
    if isinstance(o, dict):
        return {str(k): json_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        if not math.isfinite(float(o)):
            raise ValueError("non-finite number in the R12 report")
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def write_json_new(obj, path: pathlib.Path) -> None:
    """Write a report once (never overwrite), JSON-clean."""
    with open(path, "x") as fh:
        fh.write(json.dumps(json_clean(obj), indent=1, allow_nan=False))


def _arm_record(y, taxi, w, ap, cov) -> dict:
    rec = {"n": int(len(y)), "rmse_unweighted": weighted_rmse(y, taxi), "rmse_weighted": weighted_rmse(y, taxi, w),
           "by_airport": {}, "by_coverage": {}}
    for a in np.unique(ap):
        m = ap == a
        rec["by_airport"][str(a)] = {"n": int(m.sum()), "rmse_unweighted": weighted_rmse(y[m], taxi[m]),
                                     "rmse_weighted": weighted_rmse(y[m], taxi[m], w[m])}
    for c in (0, 1, 2, 3):
        m = cov == c
        if m.any():
            rec["by_coverage"][str(c)] = {"n": int(m.sum()), "rmse": weighted_rmse(y[m], taxi[m])}
    return rec


def run(split: str, arms, smoke: bool, serve_table: pd.DataFrame, out_dir, load, fit, comparators, seeds,
        root=ROOT, log=print, n_boot: int = N_BOOT, blend_weight: float = 0.5, params=None, meta=None) -> dict:
    """Fit the arms on one split, score them (unweighted and coverage-reweighted) and report the label-free 2026
    sensitivity; write predictions.parquet (test rows), predictions_2026.parquet (ranking rows) and report.json into a
    NEW out_dir. `load(feats, split_fn)` -> the fold dict of real_loader (X, y, dlt, proxy, te/tr/fit/es, rank, base,
    rank_base, ap_code, airports, feats, _keys); `fit` = lgbm_fold.fit_arm; `comparators(hold)` -> {name: taxi}."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; splits are {SPLITS}")
    out = B.check_out_dir(out_dir, root)
    if split == "holdout" and not smoke and comparators is None:
        raise ValueError("a real holdout run needs the PIPE_M comparator")
    arms = list(arms)
    for a in arms:
        design_for(a)
    if out.exists():
        raise FileExistsError(f"{out} exists: an R12 validation never overwrites")
    t0 = time.time()
    split_fn = SPLIT_FNS[split]
    pooled = split in POOLED_SPLITS
    scoring = split != "refit"
    feats_full = design_for("r12")
    fold = load(feats_full, split_fn)
    if list(fold["feats"]) != feats_full:
        raise AssertionError("the loaded fold's design is not FW-NSC + the ADS-B block")
    te, rank = np.asarray(fold["te"], bool), np.asarray(fold["rank"], bool)
    te_idx, rk_idx = np.flatnonzero(te), np.flatnonzero(rank)
    if (te & rank).any() or (len(te_idx) and len(rk_idx) and te_idx.max() > rk_idx.min()):
        raise AssertionError("every test row must precede every ranking row (and no row may be both)")
    n_te = len(te_idx)
    keys = fold["_keys"]
    k_te, k_rk = keys.loc[te].reset_index(drop=True), keys.loc[rank].reset_index(drop=True)
    hold = fold["base"].reset_index(drop=True)[["month", "ap", "y", "proxy"]].copy()
    hold["pos"], hold["MVT_ID_mvt"], hold["mvt_day"] = k_te.pos.to_numpy(), k_te.MVT_ID_mvt.to_numpy(), k_te.mvt_day
    rk = fold["rank_base"].reset_index(drop=True)[["month", "ap", "proxy", "MVT_ID_mvt"]].copy()
    rk["coverage"] = k_rk.coverage.to_numpy(dtype="float64")
    cov = k_te.coverage.to_numpy(dtype="float64")
    y, proxy, ap = hold.y.to_numpy(float), hold.proxy.to_numpy(float), hold.ap.astype(str).to_numpy()
    doy = np.asarray(fold["X"][te, feats_full.index("doy")], dtype="float64")
    if scoring and not np.isfinite(doy).all():
        raise AssertionError("non-finite doy on test rows: the day blocks are undefined")
    blocks = hold.month.to_numpy().astype("int64") * 1000 + doy.astype("int64")     # NSC / v34 pairing's day blocks
    w, cells = np.ones(n_te), {}
    if scoring:
        target = target_shares(serve_table, pooled)
        w, cells = coverage_weights(cell_keys(ap, hold.month.to_numpy(), pooled), (cov == 3).astype(int), target,
                                    allow_missing=smoke)
    p = dict(params or {})
    fold_fit = dict(fold, te=te | rank)                    # one fit predicts the test rows, then the ranking rows
    taxis, taxis_rk, fits, preds, preds_rk = {}, {}, {}, {}, {}
    month_all = np.asarray(keys.month.to_numpy(), dtype="float64")
    for a in arms:
        cols = design_for(a)
        X = fold["X"] if cols == feats_full else fold["X"][:, [feats_full.index(c) for c in cols]]
        n_masked = n_adsb_masked = 0
        if a == "r12_noeobt":
            X, n_masked = mask_eobt(X, cols, fold["ap_code"], fold["airports"])
        if a == "janjul_adsb":
            X, n_adsb_masked = mask_janjul(X, cols, month_all, rank)
        res = fit(fold_fit, X, p.get("params"), p.get("nest"), p.get("patience"), p.get("pa_floor"), tuple(seeds),
                  False, "es", f"r12:{a}", target=TARGETS[a])
        pred = np.asarray(res["delta"], float)
        if len(pred) != n_te + len(rk_idx):
            raise AssertionError(f"arm {a}: {len(pred)} predictions for {n_te} test + {len(rk_idx)} ranking rows")
        preds[a], preds_rk[a] = pred[:n_te], pred[n_te:]
        taxis[a] = np.asarray(LF.taxi_time(proxy, pred[:n_te], TARGETS[a]), float) if n_te else np.zeros(0)
        taxis_rk[a] = np.asarray(LF.taxi_time(rk.proxy.to_numpy(float), pred[n_te:], TARGETS[a]), float)
        single = {str(s): weighted_rmse(y, LF.taxi_time(proxy, np.asarray(v, float)[:n_te], TARGETS[a]))
                  for s, v in res["single"].items()} if n_te else {}
        fits[a] = {"best_iter": int(res["best_iter"]), "n_ref": int(res["n_ref"]), "n_features": len(cols),
                   "target": TARGETS[a], "single_seed_rmse": single, "n_rows_eobt_masked": n_masked,
                   "n_rows_adsb_masked": n_adsb_masked}
        del X
        gc.collect()
        if n_te:
            log(f"arm {a}: RMSE {weighted_rmse(y, taxis[a]):.4f} unweighted, {weighted_rmse(y, taxis[a], w):.4f} "
                f"reweighted on {n_te:,} rows")
    if "r12" in taxis and "yhead" in taxis:
        taxis["B_blend_y"] = LF.blend_taxi_times(taxis["yhead"], taxis["r12"], blend_weight)
        taxis_rk["B_blend_y"] = LF.blend_taxi_times(taxis_rk["yhead"], taxis_rk["r12"], blend_weight)
    comps = comparators(hold) if comparators is not None else {}
    rep = {"split": split, "smoke": bool(smoke), "arms": {}, "pairs": {}, "pairs_skipped": [], "fits": fits, "n_test": int(n_te),
           "n_rank": int(len(rk_idx)), "n_test_without_table": int(np.isnan(cov).sum()), "seeds": list(seeds),
           "design_base": "FEATS_NSC_FW", "adsb_cols": ADSB_COLS, "eobt_cols": EOBT_COLS,
           "weights": {"pooled_target": pooled, "cells": cells,
                       "n_eff_total": float(w.sum() ** 2 / (w ** 2).sum()) if n_te else None},
           "comparators": {}, "blend_weight": blend_weight,
           "units": "board MSE = SSE/344,841; gain > 0 = candidate better", "meta": meta or {}}
    for name, tx in (taxis.items() if scoring else ()):
        rep["arms"][name] = _arm_record(y, tx, w, ap, cov)
    for cname, cv in comps.items():
        cv = np.asarray(cv, float)
        m = np.isfinite(cv)
        rep["comparators"][cname] = {"n": int(m.sum()), "rmse_unweighted": weighted_rmse(y[m], cv[m]) if m.any() else None}
    for name, tx in (taxis.items() if scoring else ()):
        others = [(c, taxis[c]) for c in CONTROL_ARMS if c in taxis and c != name] + \
                 [(c, np.asarray(v, float)) for c, v in comps.items()]
        for oname, ov in others:
            if not np.isfinite(ov).any():                    # no comparable row in this split: listed, not scored
                rep["pairs_skipped"].append(f"{name}_vs_{oname}")
                continue
            rep["pairs"][f"{name}_vs_{oname}"] = {
                "unweighted": paired_gain(y, tx, ov, blocks, None, n_boot),
                "weighted": paired_gain(y, tx, ov, blocks, w, n_boot)}
    ref = hold[["ap", "month", "proxy"]]
    pairs = [("r12", b) for b in taxis if b != "r12"] if "r12" in taxis else []
    for name in ("control", "janjul_adsb"):             # the key control is also compared with every other arm
        if name in taxis and "r12" not in taxis:
            pairs = [(name, b) for b in taxis if b != name]
    ap_all = np.asarray(fold["airports"], dtype=object)[np.asarray(fold["ap_code"])]
    rep["sensitivity_2026"] = {} if not scoring else {
        "prediction_shift": {n: prediction_shift(ref.assign(taxi=taxis[n]), rk.assign(taxi=taxis_rk[n]), pooled)
                             for n in taxis},
        "input_range": input_range(fold["X"], feats_full, EOBT_COLS, np.asarray(fold["tr"], bool), rank, ap_all,
                                   keys.month.to_numpy(), pooled) if len(rk_idx) else {},
        "arm_differences_2026": arm_differences(taxis_rk, rk, pooled, pairs),
        "arm_differences_2025_test": arm_differences(taxis, hold, pooled, pairs),
        "note": "label-free: a 2026 shift mixes real change with input drift; arm_differences isolate the arms"}
    out.mkdir(parents=True)
    if scoring:
        pred = hold[["month", "pos", "MVT_ID_mvt", "ap", "mvt_day", "y", "proxy"]].copy()
        pred["w"], pred["coverage"], pred["doy"] = w, cov, doy
        for name, tx in taxis.items():
            pred[f"taxi_{name}"] = tx
        for name, pv in preds.items():
            pred[f"pred_{name}"] = pv                   # the raw prediction (delta, or y for yhead): the stack's F
        for cname, cv in comps.items():
            pred[f"comp_{cname}"] = np.asarray(cv, float)
        pred.to_parquet(out / "predictions.parquet", index=False)
    p26 = rk.copy()
    for name, tx in taxis_rk.items():
        p26[f"taxi_{name}"] = tx
    for name, pv in preds_rk.items():
        p26[f"pred_{name}"] = pv
    p26.to_parquet(out / "predictions_2026.parquet", index=False)
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json_new(rep, out / "report.json")
    for k, v in rep["pairs"].items():
        log(f"{k}: unweighted {v['unweighted']['gain_bmse']:+.2f} CI {[round(x, 2) for x in v['unweighted']['ci95']]}"
            f"  reweighted {v['weighted']['gain_bmse']:+.2f} CI {[round(x, 2) for x in v['weighted']['ci95']]}")
    return rep


# ---------------------------------------------------------------- real inputs

def seeds_for(arg, screen: bool) -> tuple:
    """--seeds as given; else (0,) for a screen (the ranking tier) and (0, 1, 2) for a real run."""
    if arg is None:
        return (0,) if screen else (0, 1, 2)
    try:
        seeds = tuple(int(x) for x in str(arg).split(",") if x.strip())
    except ValueError as e:
        raise ValueError(f"--seeds must be comma-separated integers, got {arg!r}") from e
    if not seeds:
        raise ValueError(f"--seeds names no seed: {arg!r}")
    return seeds


def frame_scope(split: str, smoke: bool, full_frame: bool) -> tuple:
    """(months, n_rank) for load_frames: real = everything; smoke = SMOKE_MONTHS + lgbm_submit's 5,000 ranking rows;
    full-frame smoke = the real frame with smoke trees (the peak-memory measurement and 1-step dry run)."""
    if full_frame and not smoke:
        raise ValueError("--full-frame is a smoke option (the real run already uses the full frame)")
    if not smoke or full_frame:
        return None, None
    return SMOKE_MONTHS[split], LF.L.SMOKE["n_rank"]


def fit_config(smoke: bool, screen: bool = False) -> dict:
    """lgbm_submit's production values; --screen: learning rate 0.05 with the production tree budget and patience (the
    ranking tier; only the winner gets the lr-0.01 run); --smoke: lgbm_fold's SMOKE values (lr 0.05, 60 trees)."""
    params, nest, patience, pa_floor = dict(LF.L.P), LF.L.NEST, LF.L.PATIENCE, LF.L.PA_TREE_FLOOR
    if screen:
        params["learning_rate"] = SCREEN_LR
    if smoke:
        params["learning_rate"] = LF.SMOKE["learning_rate"]
        nest, patience, pa_floor = LF.SMOKE["nest"], LF.SMOKE["patience"], LF.SMOKE["pa_floor"]
    return {"params": params, "nest": nest, "patience": patience, "pa_floor": pa_floor}


def real_loader(train_table: pd.DataFrame, serve_table: pd.DataFrame, strict: bool, months, n_rank):  # pragma: no cover
    """lgbm_submit.load_frames (training months + ranking rows, queue/order/weather), the ADS-B block, the split's
    masks (ranking rows in none), in-fold encodings on the split's training rows, the float32 design matrix."""
    L, S = LF.L, LF.S

    def load(feats, split_fn):
        d, is_rank, rank_ids = L.load_frames(L.CACHE, months=months, n_rank=n_rank, queue=True, order=True,
                                             weather=True)
        info = attach_all(d, is_rank, rank_ids, train_table, serve_table, strict)
        n_tr = int((~is_rank).sum())
        pad = np.zeros(int(is_rank.sum()), bool)
        te, tr, fit, es = (np.r_[np.asarray(m, bool), pad] for m in split_fn(d.month.to_numpy()[:n_tr]))
        y, dlt, proxy, sp = d.y.to_numpy(), d.delta.to_numpy(), d.proxy.to_numpy(), d.sp.to_numpy()
        ap_code, airports = L.airport_codes(d.ap)
        month = d.month.to_numpy()
        for col, vals in S.infold_encodings(d, y, dlt, tr).items():
            d[col] = vals
        d = d.drop(columns=[c for c in set(S.ENC_KEYS) | {"ars", "ap"} if c in d.columns])
        gc.collect()
        X = L.design_matrix(d, feats)
        del d
        gc.collect()
        ap_s = np.asarray(airports, dtype=object)[ap_code].astype(str)
        base = pd.DataFrame({"row": np.flatnonzero(te).astype("int64"), "month": month[te].astype("int32"),
                             "ap": ap_s[te], "y": y[te], "proxy": proxy[te], "delta": dlt[te], "sp": sp[te]})
        rank_base = pd.DataFrame({"month": month[is_rank].astype("int32"), "ap": ap_s[is_rank],
                                  "proxy": proxy[is_rank], "MVT_ID_mvt": np.asarray(rank_ids, dtype="float64")})
        LF.log(f"R12 frame: {n_tr:,} training rows (test {int(te.sum()):,}, fit {int(fit.sum()):,}, stop "
               f"{int(es.sum()):,}) + {int(is_rank.sum()):,} ranking rows; design {X.shape}; rows without an ADS-B "
               f"table row {info['n_without_table']:,}; peak RSS {L.peak_rss_gb():.2f} GB")
        return dict(X=X, y=y, dlt=dlt, proxy=proxy, sp=sp, month=month, te=te, tr=tr, fit=fit, es=es, rank=is_rank,
                    ap_code=ap_code, airports=airports, feats=list(feats), base=base, rank_base=rank_base,
                    _keys=info["keys"], _n_without_table=info["n_without_table"])
    return load


def join_comparators(hold: pd.DataFrame, lane_m: pd.DataFrame, step1: pd.DataFrame, month_counts: dict,
                     expect_pipe_n=None) -> dict:
    """The comparators on the test rows (NaN where a comparator has no row): PIPE_M (lane_m: MVT_ID_mvt, y, pipe) by
    MVT id, and the step-1 F-NSC record (row, y, proxy, treatment; taxi = max(proxy - treatment, 1)) by (month, pos)
    through the cache month counts. Both joins are label-guarded: y must agree on every joined row. Columns are read
    by name, never by attribute (`pipe` is also a DataFrame method: regression bug r12-1)."""
    ids = hold["MVT_ID_mvt"].to_numpy(float)
    y = hold["y"].to_numpy(float)
    m = lane_m.set_index("MVT_ID_mvt")
    pipe, yv = m["pipe"].reindex(ids).to_numpy(float), m["y"].reindex(ids).to_numpy(float)
    ok = np.isfinite(yv)
    if ok.any() and np.abs(yv[ok] - y[ok]).max() > 1e-6:
        raise AssertionError("PIPE_M y differs from the fold's y on joined rows: misaligned")
    if expect_pipe_n is not None and int(ok.sum()) != int(expect_pipe_n):
        raise AssertionError(f"PIPE_M joined {int(ok.sum())} test rows, expected {expect_pipe_n}")
    mo, po = rows_to_month_pos(step1["row"].to_numpy(), month_counts)
    r = pd.DataFrame({"k": mo.astype("int64") * KEY_MULT + po.astype("int64"), "y": step1["y"].to_numpy(float),
                      "taxi": np.maximum(step1["proxy"].to_numpy(float) - step1["treatment"].to_numpy(float), 1.0)})
    r = r.set_index("k").reindex(hold["month"].to_numpy().astype("int64") * KEY_MULT
                                 + hold["pos"].to_numpy().astype("int64"))
    ry = r["y"].to_numpy(float)
    ok = np.isfinite(ry)
    if ok.any() and np.abs(ry[ok] - y[ok]).max() > 1e-6:
        raise AssertionError("step-1 record y differs from the fold's y on joined rows: misaligned")
    return {"PIPE_M": pipe, "F_NSC_step1": r["taxi"].to_numpy(float)}


def real_comparators(month_counts: dict, strict: bool):  # pragma: no cover - shipped artefacts; smoke-exercised
    def comps(hold: pd.DataFrame) -> dict:
        CB = _load("cln_build", "scripts/analysis/cln_build.py")
        lane_m = CB.load_comparators()["M"]
        step1 = pd.read_parquet(STEP1_RECORD, columns=["row", "y", "proxy", "treatment"])
        return join_comparators(hold, lane_m, step1, month_counts,
                                expect_pipe_n=len(lane_m) if strict else None)
    return comps


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--arms", required=True, help=f"comma-separated subset of {','.join(ARMS)}")
    ap.add_argument("--tables", required=True, help="an r12_build.py tables output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=None, help="default 0,1,2; 0 under --screen")
    ap.add_argument("--screen", action="store_true", help="ranking tier: learning rate 0.05, one seed")
    ap.add_argument("--rss-limit-gib", type=float, required=True, help="in-process peak-RSS abort (the launcher passes it)")
    ap.add_argument("--smoke", action="store_true", help="3 months, 5,000 ranking rows, 60 trees, non-strict, no lock")
    ap.add_argument("--full-frame", action="store_true", help="with --smoke: the full real frame (memory / dry run)")
    return ap.parse_args(argv)


def main(argv=None) -> int:  # pragma: no cover - real-data entry point; exercised by the smoke runs
    import run_guards as G
    a = parse_args(argv)
    log = lambda m: print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)
    tables = pathlib.Path(a.tables)
    trep = json.loads((tables / "report.json").read_text())
    if trep.get("features_version") != B.require_trc(B.AF.FEATURES_VERSION):
        raise RuntimeError("the tables were built with another features_version: rebuild them")
    fh = None
    if not a.smoke:
        fh = G.acquire_lock()
        log("heavy-compute lock acquired (nonblocking)")
    G.start_rss_guard(int(a.rss_limit_gib * 1024 ** 3))
    log(f"in-process RSS abort at {a.rss_limit_gib:g} GiB")
    try:
        train = pd.read_parquet(tables / "train.parquet")
        serve = pd.read_parquet(tables / "serve.parquet")
        months, n_rank = frame_scope(a.split, a.smoke, a.full_frame)
        fit = LF.fit_arm
        if "cat" in a.arms.split(","):                   # R13: CatBoost in its own process (BC-5)
            C = _load("r13_cat", "scripts/analysis/r13_cat.py")
            work = pathlib.Path(str(pathlib.Path(a.out)) + "_catwork")
            if work.exists():
                raise FileExistsError(f"{work} exists: a CatBoost work directory is never reused")
            tier = "smoke" if a.smoke else ("screen" if a.screen else "full")
            fit = C.dispatch(LF.fit_arm, C.CatFitter(work, C.cat_config(tier), log=log))
            log(f"arm cat -> CatBoost worker ({tier}: {C.cat_config(tier)}), work dir {work}")
        comps = real_comparators(cache_month_counts(), strict=not a.smoke) if a.split == "holdout" else None
        rep = run(a.split, [s for s in a.arms.split(",") if s], a.smoke, serve, a.out,
                  real_loader(train, serve, strict=not a.smoke, months=months, n_rank=n_rank), fit, comps,
                  seeds_for(a.seeds, a.screen), log=log, params=fit_config(a.smoke, a.screen),
                  meta={"tables": str(tables), "features_version": trep["features_version"], "stamp": trep.get("stamp"),
                        "screen": bool(a.screen), "learning_rate": fit_config(a.smoke, a.screen)["params"]["learning_rate"]})
        log(f"report -> {pathlib.Path(a.out) / 'report.json'}  wall {rep['wall_s']:.0f}s  peak RSS "
            f"{G.peak_rss() / 1024 ** 3:.2f} GiB")
    finally:
        if fh is not None:
            G.release_lock(fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
