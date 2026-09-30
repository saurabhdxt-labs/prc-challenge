"""CLN -- clean-room model vs the full pipeline on fold A (plans/PREREG_cln_clean_room_model_2026_09_23.md).

Two arms, CLN-D (target delta, matched rows only) and CLN-Y (target y, all rows), built from the 30
raw columns only: no pipeline stage, splice, target encoding, cache or composition rule. 17 native
LightGBM categoricals (vocabulary from training rows only), 4 UTC clock features, 12 raw time
differences in seconds, 4 label-free traffic counts. Reused by import: scripts/lgbm_submit.py (P,
NEST, PATIENCE, early_stop, fit_seeds), scripts/analysis/rcr2_build.py (load_incumbent, delta_stats),
scripts/run_guards.py (heavy lock, RSS abort).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/cln_build.py smoke
    /opt/homebrew/bin/python3.11 -B scripts/analysis/cln_build.py run --arm CLN-D
    /opt/homebrew/bin/python3.11 -B scripts/analysis/cln_build.py run --arm CLN-Y
    /opt/homebrew/bin/python3.11 -B scripts/analysis/cln_build.py score

Nothing here uploads, builds a submission candidate, or runs git add/commit/push.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RAW_DIR = ROOT / "data" / "raw"
RUN_DIR = ROOT / "data" / "cln_run"
SMOKE_DIR = RUN_DIR / "smoke"
REPORT_PATH = ROOT / "reports" / "cln_report.json"

# ---- registered constants --------------------------------------------------------------------
FIT_MONTHS = (2, 4, 5, 6, 8, 10, 11, 12)
STOP_MONTHS = (3, 9)
HOLDOUT_MONTHS = (1, 7)
SPENT_DAY = "2025-01-09"

N_FIT_MATCHED, N_STOP_MATCHED, N_HOLD_MATCHED = 1_378_585, 344_840, 339_015
N_FIT_ALL, N_STOP_ALL = 1_391_968, 348_355
N_HOLD_UNMATCHED = 5_321

INC_RMSE = 208.31040447352203
PIPE_M_SSE_344841 = 41_969.3
PIPE_M_RMSE = 208.1212
PIPE_U_SSE_344841 = 53_097.0
N_LANE_M = 334_132
N_LANE_U = 5_276
BOARD_N = 344_841.0

MARGIN_LANE = {"M": 0.03 * PIPE_M_SSE_344841, "U": 0.03 * PIPE_U_SSE_344841}

N_BOOT, BOOT_SEED = 2000, 0

BANNED_COLUMNS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")

CATEGORICALS = [
    "ADEP_mvt", "ADES_mvt", "FLIGHT_RULE_mvt", "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt",
    "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_flt", "ADES_flt", "ADES_FILED_flt", "MARKET_SEGMENT_flt",
    "FLIGHT_RULE_flt", "FLIGHT_TYPE_flt", "AIRCRAFT_TYPE_flt", "WK_TBL_CAT_flt", "AIRCRAFT_OPERATOR_flt",
]
CLOCK_FEATURES = ["mvt_hr", "mvt_min", "mvt_dow", "sched_hr"]
# name -> (minuend column, subtrahend column); NaN when either operand is null.
TIME_DIFFS = {
    "d_mvt_sched": ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
    "d_mvt_aobt3": ("MVT_TIME_UTC_mvt", "AOBT_3_flt"),
    "d_mvt_lobt": ("MVT_TIME_UTC_mvt", "LOBT_flt"),
    "d_mvt_iobt": ("MVT_TIME_UTC_mvt", "IOBT_flt"),
    "d_mvt_eobt1": ("MVT_TIME_UTC_mvt", "EOBT_1_flt"),
    "d_aobt3_sched": ("AOBT_3_flt", "SCHED_TIME_UTC_mvt"),
    "d_aobt3_eobt1": ("AOBT_3_flt", "EOBT_1_flt"),
    "d_eobt1_lobt": ("EOBT_1_flt", "LOBT_flt"),
    "d_iobt_lobt": ("IOBT_flt", "LOBT_flt"),
    "d_arvt3_mvt": ("ARVT_3_flt", "MVT_TIME_UTC_mvt"),
    "d_arvt1_mvt": ("ARVT_1_flt", "MVT_TIME_UTC_mvt"),
    "d_arvt3_arvt1": ("ARVT_3_flt", "ARVT_1_flt"),
}
COUNT_FEATURES = ["n_dep_ap_30", "n_dep_ap_60", "n_dep_rwy_30", "n_arr_ap_30"]
FEATURE_ORDER = CATEGORICALS + CLOCK_FEATURES + list(TIME_DIFFS) + COUNT_FEATURES
assert len(FEATURE_ORDER) == 37, len(FEATURE_ORDER)
assert not (set(FEATURE_ORDER) & set(BANNED_COLUMNS))

RAW_REQUIRED_FOR_COUNTS = {"MVT_TIME_UTC_mvt", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "PHASE_mvt"}

TARGETS = ("delta", "y")
SEEDS = (0, 1, 2)
RSS_LIMIT = 8 * 1024 ** 3          # the delegation's guard (registration cost/guards section names 12 GiB; the
                                     # tighter 8 GiB is this build's own instruction and is the binding one here)
SMOKE_STRIDE = 200


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- raw loading ------------------------------------------------------------------------------
def _raw_files(root: pathlib.Path = ROOT):
    return sorted((root / "data" / "raw").glob("training_2025-*.parquet"))


def _month_of_path(path) -> int:
    """The month a raw training file covers, from its filename (training_2025-MM-01_...)."""
    stem = pathlib.Path(path).name
    marker = "training_2025-"
    if marker not in stem:
        raise ValueError(f"cannot read a month out of {stem!r}")
    return int(stem[len(marker):len(marker) + 2])


def load_raw(months, root: pathlib.Path = ROOT) -> pd.DataFrame:
    """All 30 raw columns + `month` (from the filename) for the given months' raw files, both phases.
    `months=None` loads all 12."""
    files = _raw_files(root)
    if not files:
        raise ValueError(f"no raw training files under {root / 'data' / 'raw'}")
    wanted = None if months is None else set(int(m) for m in months)
    frames = []
    for f in files:
        mo = _month_of_path(f)
        if wanted is not None and mo not in wanted:
            continue
        df = pd.read_parquet(f)
        df["month"] = mo
        frames.append(df)
    if not frames:
        raise ValueError(f"no raw files matched months {months}")
    return pd.concat(frames, ignore_index=True)


def _raw_path_for_month(month: int, root: pathlib.Path = ROOT) -> pathlib.Path:
    for f in _raw_files(root):
        if _month_of_path(f) == int(month):
            return f
    raise ValueError(f"no raw file for month {month}")


# ---- count features -----------------------------------------------------------------------------
def _epoch_seconds(t: pd.Series) -> np.ndarray:
    """Whole-second epoch ticks, forced through an explicit seconds cast (never astype(int64)//1e9 on a
    datetime64[us] column: that divides microsecond ticks by a nanosecond scale and is silently wrong, BC-2)."""
    t = pd.to_datetime(t, utc=True)
    return t.to_numpy().astype("datetime64[s]").astype("int64")


def _searchsorted_counts(sorted_times: np.ndarray, query_times: np.ndarray, window: int) -> np.ndarray:
    lo = np.searchsorted(sorted_times, query_times - window, side="left")
    hi = np.searchsorted(sorted_times, query_times + window, side="right")
    return hi - lo


def _grouped_counts(keys_all, t_all: np.ndarray, keys_src, t_src: np.ndarray, window: int,
                     exclude_self_mask: np.ndarray | None) -> np.ndarray:
    """|Δt| <= window count of `keys_src`/`t_src` rows sharing each row's own key, vectorised per group
    (searchsorted on each group's sorted times, never an O(n^2) pairwise compare). `exclude_self_mask`
    subtracts 1 for rows that are themselves members of the source group (self-exclusion)."""
    n = len(t_all)
    out = np.zeros(n, dtype=np.int64)
    src = pd.DataFrame({"k": keys_src, "t": t_src})
    groups = {}
    for k, g in src.groupby("k", sort=False):
        groups[k] = np.sort(g["t"].to_numpy())
    query = pd.DataFrame({"k": keys_all})
    for k, idx in query.groupby("k", sort=False).indices.items():
        ts = groups.get(k)
        if ts is None or len(ts) == 0:
            continue
        idx = np.asarray(idx)
        out[idx] = _searchsorted_counts(ts, t_all[idx], window)
    if exclude_self_mask is not None:
        out[exclude_self_mask] -= 1
    return out


def count_features(frame: pd.DataFrame) -> pd.DataFrame:
    """The 4 label-free traffic counts, added as new columns to a copy of `frame`. Event time is
    MVT_TIME_UTC_mvt for DEP and ARR rows alike; self excluded; windows inclusive at the edge."""
    missing = RAW_REQUIRED_FOR_COUNTS - set(frame.columns)
    if missing:
        raise ValueError(f"count_features requires {sorted(missing)}")
    t = _epoch_seconds(frame["MVT_TIME_UTC_mvt"])
    phase = frame["PHASE_mvt"].to_numpy()
    dep_mask = phase == "DEP"
    arr_mask = phase == "ARR"
    adep = frame["ADEP_mvt"].to_numpy()
    ades = frame["ADES_mvt"].to_numpy()
    rwy = frame["RUNWAY_mvt"].astype(object).to_numpy()
    rwy_key = np.array([f"{a}\x1f{r}" for a, r in zip(adep, rwy)], dtype=object)

    n_dep_ap_30 = _grouped_counts(adep, t, adep[dep_mask], t[dep_mask], 1800, dep_mask)
    n_dep_ap_60 = _grouped_counts(adep, t, adep[dep_mask], t[dep_mask], 3600, dep_mask)
    n_dep_rwy_30 = _grouped_counts(rwy_key, t, rwy_key[dep_mask], t[dep_mask], 1800, dep_mask)
    n_arr_ap_30 = _grouped_counts(adep, t, ades[arr_mask], t[arr_mask], 1800, None)

    out = frame.copy()
    out["n_dep_ap_30"] = n_dep_ap_30
    out["n_dep_ap_60"] = n_dep_ap_60
    out["n_dep_rwy_30"] = n_dep_rwy_30
    out["n_arr_ap_30"] = n_arr_ap_30
    return out


# ---- features -------------------------------------------------------------------------------------
def fit_vocab(frame_train: pd.DataFrame) -> dict:
    """Sorted vocabulary of each categorical's non-null training values."""
    return {c: sorted(frame_train[c].dropna().unique().tolist()) for c in CATEGORICALS}


def build_features(frame: pd.DataFrame, vocab: dict | None = None, feats=None):
    """37 registered columns in a fixed order, from the raw columns of the row itself plus the 4
    label-free counts (already attached by count_features). `vocab=None` fits it from `frame` (call
    only on training rows); otherwise it is applied (unseen level -> NaN). Refuses to build with
    BLOCK_TIME_UTC_mvt / TAXITIME_SEC_mvt in the feature list and never reads either column."""
    feats = list(FEATURE_ORDER) if feats is None else list(feats)
    banned = [c for c in feats if c in BANNED_COLUMNS]
    if banned:
        raise ValueError(f"feature list names banned columns: {banned}")
    frame = frame.drop(columns=[c for c in BANNED_COLUMNS if c in frame.columns])

    if vocab is None:
        vocab = fit_vocab(frame)

    X = pd.DataFrame(index=frame.index)
    for c in CATEGORICALS:
        if c not in feats:
            continue
        cats = vocab[c]
        seen = frame[c].where(frame[c].isin(cats))     # unseen level -> NaN before the Categorical is built
        X[c] = pd.Categorical(seen, categories=cats)

    if "mvt_hr" in feats or "mvt_min" in feats or "mvt_dow" in feats:
        mvt = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True)
        if "mvt_hr" in feats:
            X["mvt_hr"] = mvt.dt.hour.astype("float64")
        if "mvt_min" in feats:
            X["mvt_min"] = (mvt.dt.hour * 60 + mvt.dt.minute).astype("float64")
        if "mvt_dow" in feats:
            X["mvt_dow"] = mvt.dt.dayofweek.astype("float64")
    if "sched_hr" in feats:
        sched = pd.to_datetime(frame["SCHED_TIME_UTC_mvt"], utc=True)
        X["sched_hr"] = sched.dt.hour.astype("float64")

    for name, (a, b) in TIME_DIFFS.items():
        if name not in feats:
            continue
        ta = pd.to_datetime(frame[a], utc=True)
        tb = pd.to_datetime(frame[b], utc=True)
        X[name] = (ta - tb).dt.total_seconds()

    for name in COUNT_FEATURES:
        if name not in feats:
            continue
        if name not in frame.columns:
            raise ValueError(f"{name} is not attached; run count_features first")
        X[name] = frame[name].astype("float64")

    X = X[feats]
    return X, vocab


# ---- fold masks -------------------------------------------------------------------------------------
def fold_masks(frame: pd.DataFrame) -> dict:
    """fit / stop / holdout boolean masks for DEP rows with label > 0 (over `frame`'s full length),
    plus `matched` (AOBT_3_flt present) and `base` (DEP & label > 0, the universe the other three
    masks are carved from)."""
    dep = (frame["PHASE_mvt"] == "DEP").to_numpy()
    label = pd.to_numeric(frame["TAXITIME_SEC_mvt"], errors="coerce").to_numpy()
    labelled = np.nan_to_num(label, nan=-1.0) > 0
    base = dep & labelled
    month = frame["month"].to_numpy()
    fit = base & np.isin(month, FIT_MONTHS)
    stop = base & np.isin(month, STOP_MONTHS)
    holdout = base & np.isin(month, HOLDOUT_MONTHS)
    matched = frame["AOBT_3_flt"].notna().to_numpy()
    return {"base": base, "fit": fit, "stop": stop, "holdout": holdout, "matched": matched}


def check_counts(frame: pd.DataFrame | None = None) -> dict:
    """Asserts the registered row counts (fit / stop / holdout, matched / unmatched) on real data."""
    if frame is None:
        frame = load_raw(None)
    m = fold_masks(frame)
    n_fit_m = int((m["fit"] & m["matched"]).sum())
    n_stop_m = int((m["stop"] & m["matched"]).sum())
    n_hold_m = int((m["holdout"] & m["matched"]).sum())
    n_fit_all = int(m["fit"].sum())
    n_stop_all = int(m["stop"].sum())
    n_hold_u = int((m["holdout"] & ~m["matched"]).sum())
    got = {"fit_matched": n_fit_m, "stop_matched": n_stop_m, "holdout_matched": n_hold_m,
           "fit_all": n_fit_all, "stop_all": n_stop_all, "holdout_unmatched": n_hold_u}
    want = {"fit_matched": N_FIT_MATCHED, "stop_matched": N_STOP_MATCHED, "holdout_matched": N_HOLD_MATCHED,
            "fit_all": N_FIT_ALL, "stop_all": N_STOP_ALL, "holdout_unmatched": N_HOLD_UNMATCHED}
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise AssertionError(f"row counts do not match the registration: {bad}")
    return got


# ---- comparators ------------------------------------------------------------------------------------
def load_comparators() -> dict:
    """PIPE_M on LANE M (334,132 incumbent rows) and PIPE_U on LANE U (5,276 unmatched holdout rows,
    UTC day 2025-01-09 excluded), re-derived exactly as the registration's step-0 table. Refuses to
    return anything if the reproduced numbers do not match to the registered tolerance."""
    R = _load("rcr2_build", "scripts/analysis/rcr2_build.py")
    inc = R.load_incumbent()                                   # asserts inc RMSE == INC_RMSE internally
    r = float(np.sqrt(((inc.y - inc.inc) ** 2).mean()))
    if abs(r - INC_RMSE) > 1e-8:
        raise AssertionError(f"incumbent RMSE {r} != {INC_RMSE}")

    orl = pd.read_parquet(ROOT / "reports" / "orl_fold_predictions.parquet", columns=["row", "F_orl", "orl_matched"])
    j = inc.merge(orl, on="row", how="left")
    if j["F_orl"].isna().any():
        raise ValueError("ORL predictions are missing for some incumbent rows")
    orl_val = np.where(j["pop"].to_numpy(), .5 * (j.taxi_C.to_numpy() + np.maximum(j.proxy.to_numpy() - j.F_orl.to_numpy(), 1.0)),
                        j.taxi_v13_delta.to_numpy())
    pipe_m = np.where(j["pop"].to_numpy() & (j.orl_matched.to_numpy() == 1), orl_val, j["inc"].to_numpy())
    sse_m = float(((j.y.to_numpy() - pipe_m) ** 2).sum())
    if len(j) != N_LANE_M:
        raise AssertionError(f"LANE M has {len(j)} rows, expected {N_LANE_M}")
    if abs(sse_m / BOARD_N - PIPE_M_SSE_344841) > 0.05:
        raise AssertionError(f"PIPE_M SSE/344841 {sse_m / BOARD_N} != {PIPE_M_SSE_344841}")
    rmse_m = float(np.sqrt(sse_m / len(j)))
    if abs(rmse_m - PIPE_M_RMSE) > 1e-4:
        raise AssertionError(f"PIPE_M RMSE {rmse_m} != {PIPE_M_RMSE}")
    lane_m = pd.DataFrame({"MVT_ID_mvt": j.MVT_ID_mvt.to_numpy(), "y": j.y.to_numpy(), "pipe": pipe_m,
                           "month": j.month.to_numpy(), "doy": j.doy.to_numpy(), "ap": j.ap.to_numpy()})

    stratum = pd.read_parquet(ROOT / "data" / "cache_stand" / "stratum_fold_v7_preds.parquet")
    a = stratum[stratum["fold"] == "A"].copy().reset_index(drop=True)
    months = sorted(a["month"].unique().tolist())
    times = pd.concat([pd.read_parquet(_raw_path_for_month(mo), columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"])
                       for mo in months], ignore_index=True)
    a = a.merge(times, on="MVT_ID_mvt", how="left")
    if a["MVT_TIME_UTC_mvt"].isna().any():
        raise ValueError("some fold-A unmatched rows did not join to a raw MVT time")
    mvt = pd.to_datetime(a["MVT_TIME_UTC_mvt"], utc=True)
    doy = mvt.dt.dayofyear.to_numpy()
    spent = mvt.dt.strftime("%Y-%m-%d").to_numpy() == SPENT_DAY
    keep = ~spent
    a_u = a[keep].reset_index(drop=True)
    if len(a_u) != N_LANE_U:
        raise AssertionError(f"LANE U has {len(a_u)} rows, expected {N_LANE_U}")
    sse_u = float(((a_u.y.to_numpy() - a_u.S1.to_numpy()) ** 2).sum())
    if abs(sse_u / BOARD_N - PIPE_U_SSE_344841) > 0.05:
        raise AssertionError(f"PIPE_U SSE/344841 {sse_u / BOARD_N} != {PIPE_U_SSE_344841}")
    lane_u = pd.DataFrame({"MVT_ID_mvt": a_u.MVT_ID_mvt.to_numpy(), "y": a_u.y.to_numpy(), "pipe": a_u.S1.to_numpy(),
                           "month": a_u.month.to_numpy(), "doy": doy[keep], "ap": a_u.ADEP_mvt.to_numpy()})
    return {"M": lane_m, "U": lane_u}


# ---- scoring --------------------------------------------------------------------------------------
def _block_bootstrap(g: np.ndarray, blocks: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """Block bootstrap draws of sum(g)/BOARD_N over (month, doy) blocks; the same method as
    rcr2_build.block_bootstrap_delta, exposed here so arbitrary quantiles (Bonferroni) can be taken."""
    fr = pd.DataFrame({"b": blocks, "g": np.asarray(g, dtype="float64")})
    per = fr.groupby("b").g.sum().to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    return per[idx].sum(1) / BOARD_N


def verdict_from_ci(ci95, margin: float) -> str:
    """TRUE iff CI95 lower > 0; FALSE iff CI95 upper < -margin; NEAR otherwise (ambiguity is NOT
    a verdict of TRUE or FALSE)."""
    lo, hi = ci95
    if lo > 0:
        return "TRUE"
    if hi < -margin:
        return "FALSE"
    return "NEAR"


def score(y, pipe, clean, month, doy, ap=None, arms: int = 2) -> dict:
    """G = [SSE(pipe) - SSE(clean)] / 344,841 (positive = clean is better) with its 95% CI and a
    Bonferroni CI (`arms`-way correction) from a (month, doy) block bootstrap, 2,000 draws, seed 0.
    Also minus-top-1/3, per-month, per-airport (when `ap` is given). `ap` is not in the registration's
    literal signature; it defaults to an unknown placeholder so per-airport reporting degrades
    gracefully rather than being a hard requirement of the call."""
    y = np.asarray(y, dtype="float64"); pipe = np.asarray(pipe, dtype="float64"); clean = np.asarray(clean, dtype="float64")
    month = np.asarray(month); doy = np.asarray(doy)
    if ap is None:
        ap = np.full(len(y), "?", dtype=object)
    ap = np.asarray(ap)
    g = (y - pipe) ** 2 - (y - clean) ** 2
    blocks = month.astype("int64") * 1000 + doy.astype("int64")
    sums = _block_bootstrap(g, blocks)
    ci95 = [float(np.quantile(sums, .025)), float(np.quantile(sums, .975))]
    alpha = 0.05 / (2 * arms)
    ci_bonf = [float(np.quantile(sums, alpha)), float(np.quantile(sums, 1 - alpha))]
    G = float(g.sum() / BOARD_N)
    order = np.argsort(-np.abs(g))
    out = {"n": int(len(y)), "G": G, "ci95": ci95, "ci_bonferroni": ci_bonf,
           "rmse_pipe": float(np.sqrt(((y - pipe) ** 2).mean())), "rmse_clean": float(np.sqrt(((y - clean) ** 2).mean()))}
    for k in (1, 3):
        keep = np.ones(len(g), dtype=bool)
        if len(g) > k:
            keep[order[:k]] = False
        out[f"minus_top{k}"] = float(g[keep].sum() / BOARD_N)
    out["per_month"] = {int(m): float(g[month == m].sum() / BOARD_N) for m in sorted(set(month.tolist()))}
    out["per_airport"] = {str(a): float(g[ap == a].sum() / BOARD_N) for a in sorted(set(ap.tolist()))}
    pipe_mse = float(((y - pipe) ** 2).mean())
    margin = 0.03 * pipe_mse
    out["margin"] = margin
    out["verdict"] = verdict_from_ci(ci95, margin)
    return out


# ---- run --------------------------------------------------------------------------------------------
def _build_lgbm_frame(months, root=ROOT):
    """load_raw + count_features over all 12 months, and the fold masks (over that same frame)."""
    raw = load_raw(None, root=root)
    raw = count_features(raw)
    masks = fold_masks(raw)
    return raw, masks


def _design(frame: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    X, _ = build_features(frame, vocab=vocab)
    return X


def run(arm: str, seeds=SEEDS, smoke: bool = False) -> dict:
    if arm not in ("CLN-D", "CLN-Y"):
        raise ValueError(f"arm must be CLN-D or CLN-Y, got {arm!r}")
    started = time.monotonic()
    G = _load("run_guards", "scripts/run_guards.py")
    LS = _load("lgbm_submit", "scripts/lgbm_submit.py")
    out_dir = SMOKE_DIR if smoke else RUN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"cln_{arm}_preds.parquet"
    if preds_path.exists() and not smoke:
        raise FileExistsError(f"{preds_path} exists; no silent rerun")

    fh = G.acquire_lock()
    try:
        raw, masks = _build_lgbm_frame(None)
        if smoke:
            stride = np.arange(len(raw)) % SMOKE_STRIDE == 0
            raw = raw[stride].reset_index(drop=True)
            masks = {k: v[stride] for k, v in masks.items()}
        rss0 = G.peak_rss()
        if rss0 > RSS_LIMIT:
            raise RuntimeError(f"peak RSS {rss0 / 1024**3:.2f} GiB over the {RSS_LIMIT / 1024**3:.0f} GiB limit before any fit")

        base = masks["base"]
        dep = raw[base].reset_index(drop=True)
        matched_dep = masks["matched"][base]
        fit_dep = masks["fit"][base]
        stop_dep = masks["stop"][base]
        hold_dep = masks["holdout"][base]

        if arm == "CLN-D":
            keep = matched_dep
        else:
            keep = np.ones(len(dep), dtype=bool)
        d = dep[keep].reset_index(drop=True)
        del raw, dep                         # the 4.2 M-row raw frame and the DEP copy are not needed past here
        gc.collect()
        m_fit, m_stop, m_hold = fit_dep[keep], stop_dep[keep], hold_dep[keep]
        m_matched = matched_dep[keep]
        d = count_features(d) if not set(COUNT_FEATURES).issubset(d.columns) else d

        train_vocab_mask = m_fit | m_stop
        vocab = fit_vocab(d[train_vocab_mask])
        X, _ = build_features(d, vocab=vocab)

        proxy = (pd.to_datetime(d["MVT_TIME_UTC_mvt"], utc=True) - pd.to_datetime(d["AOBT_3_flt"], utc=True)).dt.total_seconds().to_numpy()
        y = pd.to_numeric(d["TAXITIME_SEC_mvt"], errors="coerce").to_numpy(dtype="float64")
        dlt = proxy - y

        train_mask = m_fit | m_stop
        fit_mask = m_fit
        es_mask = m_stop
        hold_mask = m_hold

        target = "delta" if arm == "CLN-D" else "y"
        es_eval = (m_stop & m_matched) if arm == "CLN-Y" else None

        p = dict(LS.P)
        nest, patience = (60, 50) if smoke else (LS.NEST, LS.PATIENCE)
        if smoke:
            p["learning_rate"] = LS.SMOKE["learning_rate"]

        info = LS.early_stop(X, dlt, y, proxy, train_mask, fit_mask, es_mask, p, nest, patience,
                             target=target, es_eval=es_eval)
        label = LS.regression_label(dlt, y, target)
        preds = LS.fit_seeds(X, label, train_mask, X[hold_mask], p, info["n_ref"], seeds,
                             n_features=len(FEATURE_ORDER), target=target)
        rss1 = G.peak_rss()
        # A post-fit check cannot prevent anything, it can only discard finished work (2026-09-23: CLN-D's three seeds
        # finished at 8.38 GiB and were thrown away). Record the overrun; the in-flight guard is the external watchdog.
        rss_over_after_fit = bool(rss1 > RSS_LIMIT)
        if rss_over_after_fit:
            print(f"WARNING: peak RSS {rss1 / 1024**3:.2f} GiB is over the {RSS_LIMIT / 1024**3:.0f} GiB limit after "
                  "fit_seeds (recorded; predictions kept)", flush=True)

        mean_pred = np.mean([preds[s] for s in seeds], axis=0)
        if arm == "CLN-D":
            pred = np.maximum(proxy[hold_mask] - mean_pred, 1.0)
        else:
            pred = np.maximum(mean_pred, 1.0)

        mvt_hold = pd.to_datetime(d.loc[hold_mask, "MVT_TIME_UTC_mvt"], utc=True)
        out = pd.DataFrame({
            "MVT_ID_mvt": d.loc[hold_mask, "MVT_ID_mvt"].to_numpy(),
            "month": d.loc[hold_mask, "month"].to_numpy(),
            "doy": mvt_hold.dt.dayofyear.to_numpy(),
            "ap": d.loc[hold_mask, "ADEP_mvt"].to_numpy(),
            "y": y[hold_mask],
            "matched": m_matched[hold_mask] if arm == "CLN-Y" else np.ones(int(hold_mask.sum()), dtype=bool),
            "pred": pred,
        })
        for s in seeds:
            out[f"pred_seed{s}"] = preds[s]

        mode = "wb" if smoke else "xb"
        with open(preds_path, mode) as fobj:
            out.to_parquet(fobj, index=False)

        report = {"tag": "CLN", "arm": arm, "target": target, "smoke": smoke, "n_train": int(train_mask.sum()),
                  "n_holdout": int(hold_mask.sum()), "best_iter": info["best_iter"], "n_ref": info["n_ref"],
                  "seconds": time.monotonic() - started, "peak_rss_GiB": G.peak_rss() / 1024 ** 3,
                  "rss_over_limit_after_fit": rss_over_after_fit}
        report_path = out_dir / f"cln_{arm}_report.json"
        rmode = "w" if smoke else "x"
        with open(report_path, rmode) as f:
            json.dump(report, f, indent=2, default=float)
        return report
    finally:
        G.release_lock(fh)


def score_arm_preds(arm: str, comparators: dict, smoke: bool = False) -> dict:
    out_dir = SMOKE_DIR if smoke else RUN_DIR
    preds = pd.read_parquet(out_dir / f"cln_{arm}_preds.parquet")
    result = {}
    lanes = ("M",) if arm == "CLN-D" else ("M", "U")
    for lane in lanes:
        comp = comparators[lane]
        if arm == "CLN-D":
            sub = preds
        else:
            want_matched = lane == "M"
            sub = preds[preds["matched"] == want_matched]
        j = comp.merge(sub[["MVT_ID_mvt", "pred"]], on="MVT_ID_mvt", how="inner")
        st = score(j["y"].to_numpy(), j["pipe"].to_numpy(), j["pred"].to_numpy(), j["month"].to_numpy(),
                  j["doy"].to_numpy(), ap=j["ap"].to_numpy())
        st["margin"] = MARGIN_LANE[lane]
        st["verdict"] = verdict_from_ci(st["ci95"], st["margin"])
        result[lane] = st
    return result


def score_cmd(smoke: bool = False) -> dict:
    comparators = load_comparators()
    report = {}
    for arm in ("CLN-D", "CLN-Y"):
        out_dir = SMOKE_DIR if smoke else RUN_DIR
        p = out_dir / f"cln_{arm}_preds.parquet"
        if p.exists():
            report[arm] = score_arm_preds(arm, comparators, smoke=smoke)
    # smoke writes its own report path so it never blocks (or is mistaken for) the real `score`
    # report at REPORT_PATH, which is opened with 'x' (no silent overwrite of a real result).
    report_path = SMOKE_DIR / "cln_report_smoke.json" if smoke else REPORT_PATH
    if smoke:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if smoke else "x"
    with open(report_path, mode) as f:
        json.dump(report, f, indent=2, default=float)
    return report


# ---- CLI --------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    p_run = sub.add_parser("run")
    p_run.add_argument("--arm", required=True, choices=("CLN-D", "CLN-Y"))
    sub.add_parser("score")
    args = parser.parse_args(argv)

    if args.cmd == "smoke":
        for arm in ("CLN-D", "CLN-Y"):
            rep = run(arm, smoke=True)
            print(json.dumps({"smoke": arm, **rep}, default=float))
        print(json.dumps(score_cmd(smoke=True), default=float))
        return 0
    if args.cmd == "run":
        rep = run(args.arm, smoke=False)
        print(json.dumps(rep, default=float))
        return 0
    if args.cmd == "score":
        rep = score_cmd(smoke=False)
        print(json.dumps(rep, default=float))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
