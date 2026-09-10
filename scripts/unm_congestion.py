"""`scripts/unm_congestion.py` -- arm E3C of plans/PREREG_unm_congestion_2026_09_10.md (owner decision 2026-09-10).

    python scripts/unm_congestion.py lomo            # the heavy step: 12 months from data/raw, ~3-4 GB, run ALONE
    python scripts/unm_congestion.py score           # the light step: clauses C1-C7 from the stored parquet + 2026 witness
    python scripts/unm_congestion.py lomo --smoke    # synthetic world, code path only, NOT a result
    python scripts/unm_congestion.py score --smoke

H-E3C: S1's unmatched body (`NonFillRegressor`: sp, dayoff, hr + five target encodings) is congestion-blind;
an airport-hour witness taken from the SAME file's MATCHED departures reduces non-Rome unmatched error out of
month, concentrated in congested hours, with no material loss in calm ones.

The witness (serve-time; never reads BLOCK or a label): per airport x UTC clock hour of MVT_TIME_UTC_mvt
(floored), over that hour's matched departures (AOBT_3_flt present): `hprox_med` = median of MVT - AOBT_3 in
seconds, `hprox_n` = their count; `hprox_med` is NaN where `hprox_n` < 5. Unmatched rows never contribute.

Arms (12-fold LOMO over 2025, `stratum_fold`'s folds and frames):
  S1    exactly `stratum_fold.stratum_arms` -- GUARD: must equal data/cache_stand/stratum_fold_v7_preds.parquet
        (fold == "lomo", column S1) on every row to <= 1e-6 s, joined on MVT_ID, or nothing is written;
  S1C   the same `p` and `nf_cells` (S1's own `fit_unmatched` parts); the body is S1's regressor with the numeric
        features NF_NUMERIC + ["hprox_med", "hprox_n"] and nothing else changed (params, winsorisation, encodings,
        seeds, non-fill training rows); nf = nf_hybrid(nf_cells, mean over seeds); pred = mixture(p, sp, nf).
Per-seed versions of both feed clause C7. The shipped modules are composed, never edited.

Clauses (locked in the prereg; g = SE(S1) - SE(S1C) per non-LIRF unmatched 2025 row, raw labels):
  C1 date-block bootstrap (calendar dates, 2,000 draws, seed 0) of sum g: lower 95% bound > 0
  C2 2026-priced gain >= +500 board MSE (per-bin mean g x 2026 scored non-LIRF unmatched rows in bin / 344,841;
     a bin with < 30 2025 rows takes the next LOWER bin's per-row gain, never a higher one)
  C3 sum g > 0 in >= 8 of 12 months          C4 sum g > 0 at >= 6 of the 9 non-LIRF airports
  C5 sum g over hprox_med < 1,100 >= -10% of the pooled sum g        C6 sum g over y <= 10,800 > 0
  C7 pooled gain > 2 x sd of the three single-seed gains
Verdict: WORKING iff all seven pass; NOT WORKING iff C1 fails; else INCONCLUSIVE. Never revised.

Outputs. `lomo`: data/cache_stand/unm_congestion_lomo.parquet (TAXI-TIME convention, stamped in the parquet
metadata -- reports/bug_classes.md BC-2) and reports/unm_congestion_lomo.log. `score`: reports/unm_congestion.json,
reports/unm_congestion_score.log and data/cache_stand/unm_congestion_2026_witness.parquet (the scored 2026 non-LIRF
unmatched rows with their witness and bin). `--smoke` writes everything under data/smoke_unm_congestion/ instead,
uses 200 draws, brackets the log with the banner and never touches the real paths.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import glob
import json
import pathlib
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import stratum_fold as sf  # noqa: E402

bs = sf.bs
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
CACHE = ROOT / "data" / "cache_stand"
SMOKE_DIR = ROOT / "data" / "smoke_unm_congestion"
REFERENCE = CACHE / "stratum_fold_v7_preds.parquet"
PREREG = "plans/PREREG_unm_congestion_2026_09_10.md"
TAG = "E3C"

# ---- the registered design, frozen ----
WITNESS = ["hprox_med", "hprox_n"]
MIN_N = 5                                   # hprox_med is NaN below this count
SEEDS = sf.SEEDS                            # (0, 1, 2)
N_BOOT, BOOT_SEED = 2_000, 0                # C1: date blocks
BIN_EDGES = [-np.inf, 900.0, 1_100.0, 1_300.0, 1_500.0, 2_000.0, 2_400.0, np.inf]
BIN_LABELS = ["<900", "900-1100", "1100-1300", "1300-1500", "1500-2000", "2000-2400", ">=2400"]
NAN_BIN = "NaN"
BINS = [NAN_BIN, *BIN_LABELS]
N_SCORED_2026 = 344_841                     # C2 denominator
THIN_BIN = 30                               # C2: fewer 2025 rows than this borrows the next LOWER bin
C2_MIN_MSE = 500.0
MIN_MONTHS, N_MONTHS = 8, 12                # C3
MIN_AIRPORTS, N_AIRPORTS = 6, 9             # C4
CALM_S = 1_100.0                            # C5
CALM_TOLERANCE = 0.10                       # C5: calm sum >= -10% of pooled
MONSTER_S = sf.MONSTER_S                    # C6: y <= 10,800 is "not a monster"
ROME = "LIRF"                               # report-only; keeps R2 / E1
FOLD_A = sf.FOLD_A                          # (1, 7): reported separately
REPRO_TOL_S = 1e-6
CONVENTION = "taxi_time"
SMOKE_BANNER = "SMOKE — code path only, NOT a result"
SMOKE = dict(n_boot=200)
ARMS_C = ("S1C", *[f"S1C_seed{s}" for s in SEEDS])
PRED_COLUMNS = ["MVT_ID_mvt", "fold", "month", "date", "ADEP_mvt", "y", "sp", *WITNESS, "p_hat", "nf_cells",
                "nf_fit", "nf_fit_c", "routed_nf_fit", "S1", *[f"S1_seed{s}" for s in SEEDS], *ARMS_C]
LOMO_NAME, WITNESS_2026_NAME = "unm_congestion_lomo.parquet", "unm_congestion_2026_witness.parquet"
JSON_NAME, LOMO_LOG, SCORE_LOG = "unm_congestion.json", "unm_congestion_lomo.log", "unm_congestion_score.log"
COMMAND_LOMO = ("cd ~/Projects/prc-challenge && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 "
                "/opt/homebrew/bin/python3.11 -B -u scripts/unm_congestion.py lomo > reports/unm_congestion_lomo.console.log 2>&1")
COMMAND_SCORE = ("cd ~/Projects/prc-challenge && PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 "
                 "/opt/homebrew/bin/python3.11 -B -u scripts/unm_congestion.py score > reports/unm_congestion_score.console.log 2>&1")

_T0 = time.time()


class _Log:
    """print + append to the readable log, flushing both."""
    def __init__(self, path: pathlib.Path | None = None):
        self.fh = open(path, "a") if path else None

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:7.0f}s] {msg}"
        print(line, flush=True)
        if self.fh:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


# =============================================================================================
# the witness
# =============================================================================================

def _hour_key(frame: pd.DataFrame) -> pd.Series:
    """UTC clock hour of MVT_TIME_UTC_mvt, floored (10:59:59 -> 10:00:00)."""
    return frame.MVT_TIME_UTC_mvt.dt.floor("h")


def airport_hour_witness(frame: pd.DataFrame) -> pd.DataFrame:
    """(ADEP_mvt, hour) -> hprox_med, hprox_n from the MATCHED departures of `frame` only.

    A row counts iff AOBT_3_flt AND MVT_TIME_UTC_mvt are present; an unmatched row (AOBT_3 null) can never
    contribute, whatever else is passed in. hprox_med is NaN where hprox_n < MIN_N. Nothing here reads
    BLOCK_TIME_UTC_mvt or the label.
    """
    for c in ("ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"):
        if c not in frame.columns:
            raise ValueError(f"the witness needs column {c!r}")
    m = frame.AOBT_3_flt.notna() & frame.MVT_TIME_UTC_mvt.notna()
    f = frame.loc[m, ["ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]]
    prox = (f.MVT_TIME_UTC_mvt - f.AOBT_3_flt).dt.total_seconds().to_numpy(dtype="float64")
    g = pd.DataFrame({"ADEP_mvt": f.ADEP_mvt.astype(str).to_numpy(), "hour": _hour_key(f).to_numpy(), "prox": prox})
    agg = g.groupby(["ADEP_mvt", "hour"], observed=True).prox.agg(["median", "size"])
    out = pd.DataFrame({"hprox_med": agg["median"].astype("float64"), "hprox_n": agg["size"].astype("int64")})
    out.loc[out.hprox_n < MIN_N, "hprox_med"] = np.nan
    return out


def attach_witness(rows: pd.DataFrame, witness: pd.DataFrame) -> pd.DataFrame:
    """`rows` with hprox_med / hprox_n looked up on (ADEP_mvt, floored hour). An airport-hour without any
    matched departure gets hprox_n = 0 and hprox_med = NaN. Row order and index are preserved."""
    idx = pd.MultiIndex.from_arrays([rows.ADEP_mvt.astype(str).to_numpy(), _hour_key(rows).to_numpy()],
                                    names=["ADEP_mvt", "hour"])
    got = witness.reindex(idx)
    return rows.assign(hprox_med=got.hprox_med.to_numpy(dtype="float64"),
                       hprox_n=np.nan_to_num(got.hprox_n.to_numpy(dtype="float64"), nan=0.0).astype("int64"))


def bin_of(hprox_med) -> np.ndarray:
    """The registered C2 bins: NaN, <900, 900-1100, 1100-1300, 1300-1500, 1500-2000, 2000-2400, >=2400
    (left-closed: 900 falls in 900-1100)."""
    h = np.asarray(hprox_med, dtype="float64")
    cut = pd.cut(h, BIN_EDGES, labels=BIN_LABELS, right=False).astype(object)
    return np.where(np.isnan(h), NAN_BIN, cut).astype(object)


# =============================================================================================
# the S1C body: S1's regressor with the witness appended to its numeric features, nothing else changed
# =============================================================================================

class CongestionRegressor(bs.NonFillRegressor):
    """`NonFillRegressor` whose numeric features are NF_NUMERIC + WITNESS. Same NF_PARAMS, same
    winsorisation of the fit target, same encodings from the same non-fill rows, same seed semantics
    (early_stopping='auto' as sklearn decides on the row count). NaN in hprox_med is the regressor's
    native missing value."""

    NUMERIC = [*bs.NF_NUMERIC, *WITNESS]

    def __init__(self, train_nonfill: pd.DataFrame, seed: int):
        missing = [c for c in WITNESS if c not in train_nonfill.columns]
        if missing:
            raise ValueError(f"CongestionRegressor needs the witness columns {missing}; run attach_witness first")
        super().__init__(train_nonfill, seed)
        self.columns = self.NUMERIC + ["te_" + c for c in bs.NF_ENCODED]

    def design(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self.NUMERIC].astype("float64").copy()
        for c in bs.NF_ENCODED:
            x["te_" + c] = frame[c].astype(str).map(self.maps[c]).astype("float64").fillna(self.prior)
        return x


def fit_congestion_regressor(train_unmatched_nonfill: pd.DataFrame, seed: int) -> CongestionRegressor:
    """The S1C body. The same refusals as `build_submission.fit_nf_regressor` (no fill rows, a label present)."""
    frame = train_unmatched_nonfill
    if len(frame) == 0 or "y" not in frame.columns:
        raise ValueError("fit_congestion_regressor needs non-fill training rows with a label `y`")
    if "BLOCK_TIME_UTC_mvt" in frame.columns and "SCHED_TIME_UTC_mvt" in frame.columns:
        n_fill = int(bs.schedule_fill(frame).sum())
        if n_fill:
            raise ValueError(f"fit_congestion_regressor expects NON-FILL rows only; {n_fill} fill rows found")
    return CongestionRegressor(frame, seed)


def congestion_arms(train_unm: pd.DataFrame, test_unm: pd.DataFrame, train_matched: pd.DataFrame, seeds=SEEDS) -> tuple:
    """S1 and its per-seed arms exactly as `stratum_fold.stratum_arms`, plus S1C / S1C_seed* rebuilt from
    S1's OWN parts: p and nf_cells untouched, only the body regressor replaced. `parts` gains
    nf_fit_c and nf_fit_c_by_seed."""
    arms, parts = sf.stratum_arms(train_unm, test_unm, train_matched, seeds=seeds)
    for c in WITNESS:
        if c not in train_unm.columns or c not in test_unm.columns:
            raise ValueError(f"witness column {c!r} missing from the unmatched frames")
    nonfill = train_unm[~bs.schedule_fill(train_unm)]          # fit_unmatched's `tr[~tr.fill]`, same rows, same order
    by_seed = {int(s): fit_congestion_regressor(nonfill, s).predict(test_unm) for s in seeds}
    nf_fit_c = bs.mean_over_seeds(by_seed)
    arms["S1C"] = bs.mixture(parts["p"], parts["sp"], bs.nf_hybrid(parts["nf_cells"], nf_fit_c))
    for s in seeds:
        arms[f"S1C_seed{s}"] = bs.mixture(parts["p"], parts["sp"], bs.nf_hybrid(parts["nf_cells"], by_seed[int(s)]))
    parts.update(nf_fit_c=nf_fit_c, nf_fit_c_by_seed=by_seed)
    return arms, parts


def _score_split(unm, lirf_matched, tr, te, test_months, seeds, fold_label: str) -> pd.DataFrame:
    if tuple(int(s) for s in seeds) != tuple(SEEDS):
        raise ValueError(f"the harness scores the registered seeds {SEEDS} (every per-seed column is stored), got {tuple(seeds)!r}")
    tr_unm, te_unm = unm[tr], unm[te]
    tr_mat = lirf_matched[~lirf_matched.month.isin(list(test_months))]
    arms, parts = congestion_arms(tr_unm, te_unm, tr_mat, seeds=seeds)
    out = pd.DataFrame({"MVT_ID_mvt": te_unm.MVT_ID_mvt.to_numpy(), "fold": fold_label,
                        "month": te_unm.month.to_numpy().astype("int64"),
                        "date": te_unm.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy(),
                        "ADEP_mvt": te_unm.ADEP_mvt.to_numpy(),
                        "y": te_unm.y.to_numpy(dtype="float64"), "sp": te_unm.sp.to_numpy(dtype="float64"),
                        "hprox_med": te_unm.hprox_med.to_numpy(dtype="float64"),
                        "hprox_n": te_unm.hprox_n.to_numpy(dtype="int64")})
    out["p_hat"] = np.asarray(parts["p"], dtype="float64")
    out["nf_cells"] = np.asarray(parts["nf_cells"], dtype="float64")
    out["nf_fit"] = np.asarray(parts["nf_fit"], dtype="float64")
    out["nf_fit_c"] = np.asarray(parts["nf_fit_c"], dtype="float64")
    out["routed_nf_fit"] = out.nf_cells.to_numpy() < bs.T_TAIL_S
    for a in PRED_COLUMNS:
        if a.startswith("S1"):
            out[a] = np.asarray(arms[a], dtype="float64")
    return out[PRED_COLUMNS]


def score_lomo(unm: pd.DataFrame, lirf_matched: pd.DataFrame, seeds, log) -> pd.DataFrame:
    """Out-of-fold predictions for every stratum row, `stratum_fold.score_lomo`'s loop with the S1C arms."""
    month = unm.month.to_numpy()
    parts, scored = [], np.zeros(len(unm), dtype="int64")
    for m, tr, te in sf.lomo_folds(month):
        t = time.time()
        part = _score_split(unm, lirf_matched, tr, te, [m], seeds, "lomo")
        scored += te
        parts.append(part)
        d = part.S1C.to_numpy() != part.S1.to_numpy()
        log(f"  fold {m:2d}: test {int(te.sum()):5,}  train {int(tr.sum()):6,}  routed {int(part.routed_nf_fit.sum()):5,} -> body; "
            f"S1C differs from S1 on {int(d.sum()):5,} rows; witness NaN on {int(part.hprox_med.isna().sum()):5,}  ({time.time() - t:.1f}s)")
    if not (scored == 1).all():
        raise AssertionError("LOMO did not score every row exactly once")
    return pd.concat(parts, ignore_index=True)[PRED_COLUMNS]


# =============================================================================================
# the reproduction guard (prereg: S1 must equal stratum_fold_v7_preds.parquet's lomo S1, or nothing is written)
# =============================================================================================

def check_reproduction(preds: pd.DataFrame, reference: pd.DataFrame, tol: float = REPRO_TOL_S) -> dict:
    """Every lomo row's S1 within `tol` of the stored S1, joined on MVT_ID; the id sets must match exactly.
    The reference carries two folds (lomo and A) -- reports/bug_classes.md BC-2 -- so it is filtered to lomo
    first, or the rows double. Raises AssertionError on any miss."""
    ref = reference[reference.fold == "lomo"]
    have = preds[preds.fold == "lomo"]
    if ref.MVT_ID_mvt.duplicated().any() or have.MVT_ID_mvt.duplicated().any():
        raise AssertionError("duplicate MVT_ID in the lomo rows; the reproduction guard cannot join")
    if set(ref.MVT_ID_mvt) != set(have.MVT_ID_mvt):
        raise AssertionError(f"the harness scored {len(have):,} lomo rows, the reference has {len(ref):,}; different row sets, "
                             "nothing is written")
    want = ref.set_index("MVT_ID_mvt").S1.reindex(have.MVT_ID_mvt.to_numpy()).to_numpy(dtype="float64")
    got = have.S1.to_numpy(dtype="float64")
    diff = np.abs(got - want)
    worst = float(diff.max()) if len(diff) else 0.0
    bad = int((diff > tol).sum())
    if bad:
        raise AssertionError(f"S1 does not reproduce the stored S1 on {bad} of {len(have):,} lomo rows (max |diff| {worst:.3g} s "
                             f"> {tol:g}); nothing is written")
    return {"status": "reproduced", "n_rows": int(len(have)), "max_abs_diff_s": worst, "tolerance_s": float(tol),
            "reference": str(REFERENCE.relative_to(ROOT))}


# =============================================================================================
# frames: the real recipe (with the witness computed before the matched rows are released) and the smoke world
# =============================================================================================

def load_real(files, log) -> tuple:
    """`stratum_fold.load_real`'s recipe -- derive(admissible(load_movements(12 months))) -- with the witness
    taken from the frame's MATCHED admissible departures and attached to every unmatched row before the full
    frame is released. The same fit_unmatched(all matched) == fit_unmatched(LIRF matched) check on the first
    fold, so the LIRF-only reduction stays bit-exact."""
    t = time.time()
    f = bs.derive(bs.admissible(bs.load_movements(files)))
    f["y"] = f.TAXITIME_SEC_mvt.astype("float64")
    f["delta"] = (f.BLOCK_TIME_UTC_mvt - f.AOBT_3_flt).dt.total_seconds()
    log(f"frame: {len(f):,} admissible DEP rows from {len(files)} files in {time.time() - t:.0f}s; peak RSS {peak_rss_gb():.2f} GB")
    wit = airport_hour_witness(f[~f.unmatched])
    unm = attach_witness(f[f.unmatched], wit).reset_index(drop=True)
    lirf = f[(~f.unmatched) & (f.ADEP_mvt == ROME)].reset_index(drop=True)
    log(f"witness: {len(wit):,} airport-hours from {int((~f.unmatched).sum()):,} matched rows; on the {len(unm):,} unmatched rows "
        f"hprox_med is NaN on {int(unm.hprox_med.isna().sum()):,} ({100 * unm.hprox_med.isna().mean():.1f}%), "
        f"median hprox_n {float(unm.hprox_n.median()):.0f}")
    m0 = int(unm.month.min())
    a = bs.fit_unmatched(unm[unm.month != m0], unm[unm.month == m0], train_matched=f[(~f.unmatched) & (f.month != m0)])
    b = bs.fit_unmatched(unm[unm.month != m0], unm[unm.month == m0], train_matched=lirf[lirf.month != m0])
    if not np.array_equal(a, b):
        raise AssertionError("fit_unmatched(all matched) != fit_unmatched(LIRF matched) on the first fold")
    log(f"fold {m0}: fit_unmatched(all matched) == fit_unmatched(LIRF matched) bit-exact; releasing the full frame")
    summary = {"n_airport_hours": int(len(wit)), "n_matched_rows": int((~f.unmatched).sum()),
               "unmatched_nan_share": float(unm.hprox_med.isna().mean()),
               "unmatched_bin_counts": {b_: int(v) for b_, v in pd.Series(bin_of(unm.hprox_med)).value_counts().items()}}
    del f, a, b
    gc.collect()
    return unm, lirf, summary


def smoke_matched_companion(unm: pd.DataFrame, seed: int, k: int = 40, share: float = 0.75) -> pd.DataFrame:
    """A dense synthetic MATCHED frame at every airport, cloned from the unmatched rows: a `share` of the rows
    spawn k clones each with the MVT time jittered inside +-30 min (the rest spawn none, so their hour usually
    stays below five and the NaN path is exercised); AOBT_3 = MVT - prox, prox planted with the same peak hours
    the synthetic body uses. Only the columns the witness reads. Code path only, never a result."""
    rng = np.random.default_rng(seed)
    reps = np.where(rng.random(len(unm)) < share, k, 0)
    n = int(reps.sum())
    base = pd.Series(np.repeat(unm.MVT_TIME_UTC_mvt.to_numpy(), reps))
    mvt = pd.to_datetime(base, utc=True) + pd.to_timedelta(rng.integers(-1800, 1800, n), unit="s")
    hr = mvt.dt.hour.to_numpy()
    peak = ((hr >= 6) & (hr <= 9)) | ((hr >= 16) & (hr <= 19))
    prox = 700.0 + np.where(peak, 900.0, 0.0) + rng.normal(0.0, 120.0, n)
    ap = np.repeat(unm.ADEP_mvt.astype(str).to_numpy(), reps)
    return pd.DataFrame({"ADEP_mvt": ap, "MVT_TIME_UTC_mvt": mvt, "AOBT_3_flt": mvt - pd.to_timedelta(prox, unit="s"),
                         "MVT_ID_mvt": 5e7 + np.arange(n, dtype="float64")})


def synthetic_world(seed: int = 0, **kw) -> tuple:
    """(unmatched stratum with the witness attached, LIRF matched, witness summary) -- `stratum_fold.synthetic_frame`
    plus a dense matched companion at every airport so hprox_med is populated in most hours and NaN in some."""
    unm, lirf = sf.synthetic_frame(seed=seed, **kw)
    companion = smoke_matched_companion(unm, seed=seed + 100)
    wit = airport_hour_witness(pd.concat([companion, lirf[["ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "MVT_ID_mvt"]]],
                                         ignore_index=True))
    unm = attach_witness(unm, wit)
    summary = {"n_airport_hours": int(len(wit)), "n_matched_rows": int(len(companion) + len(lirf)),
               "unmatched_nan_share": float(unm.hprox_med.isna().mean()),
               "unmatched_bin_counts": {b_: int(v) for b_, v in pd.Series(bin_of(unm.hprox_med)).value_counts().items()}}
    return unm, lirf, summary


# =============================================================================================
# clauses
# =============================================================================================

def gains(t: pd.DataFrame) -> np.ndarray:
    """g = SE(S1) - SE(S1C) per row, against the RAW label. Never winsorised."""
    y = t.y.to_numpy(dtype="float64")
    return (y - t.S1.to_numpy(dtype="float64")) ** 2 - (y - t.S1C.to_numpy(dtype="float64")) ** 2


def date_block_bootstrap(g, dates, n_draws=N_BOOT, seed=BOOT_SEED) -> dict:
    """C1: the calendar dates are the blocks; each draw resamples them with replacement (stratum_fold.block_draws,
    default_rng(seed)) and sums the per-date sums of g. Percentile interval 2.5 / 97.5. Deterministic in seed."""
    g, dates = np.asarray(g, dtype="float64"), np.asarray(dates).astype(str)
    if len(g) != len(dates):
        raise ValueError(f"g has {len(g)} rows, dates {len(dates)}")
    blocks, code = np.unique(dates, return_inverse=True)
    k = len(blocks)
    if k < 2:
        raise ValueError(f"the date-block bootstrap needs at least two dates, got {k}")
    sums = np.bincount(code, weights=g, minlength=k)
    draws = sf.block_draws(k, n_draws, seed) @ sums
    lo, hi = (float(v) for v in np.percentile(draws, [2.5, 97.5]))
    return {"sum_g": float(g.sum()), "ci95": [lo, hi], "n_blocks": int(k), "n_draws": int(n_draws), "seed": int(seed),
            "passes": bool(lo > 0.0)}


def bin_table(g, hprox_med) -> dict:
    """{bin: {n_2025, sum_g, per_row_gain}} over every registered bin (empty bins present with n 0)."""
    b = bin_of(hprox_med)
    g = np.asarray(g, dtype="float64")
    out = {}
    for name in BINS:
        m = b == name
        n = int(m.sum())
        out[name] = {"n_2025": n, "sum_g": float(g[m].sum()), "per_row_gain": float(g[m].mean()) if n else None}
    return out


def price_2026(table: dict, counts_2026: dict, n_scored: int = N_SCORED_2026, thin: int = THIN_BIN) -> dict:
    """C2: sum over bins of (per-row gain used) x (2026 rows in bin) / n_scored. A bin with < `thin` 2025 rows
    takes the per-row gain of the next LOWER numeric bin, chained downward; the NaN bin and the lowest numeric
    bin have no lower bin, so a thin one there is priced at 0 and flagged (no extrapolation in any direction)."""
    used = {}
    for i, name in enumerate(BIN_LABELS):
        rec = table[name]
        if rec["n_2025"] >= thin:
            used[name] = (rec["per_row_gain"], name)
        elif i > 0:
            used[name] = (used[BIN_LABELS[i - 1]][0], used[BIN_LABELS[i - 1]][1])
        else:
            used[name] = (0.0, None)
    rec = table[NAN_BIN]
    used[NAN_BIN] = (rec["per_row_gain"], NAN_BIN) if rec["n_2025"] >= thin else (0.0, None)
    rows, total = {}, 0.0
    for name in BINS:
        gain, source = used[name]
        n26 = int(counts_2026.get(name, 0))
        priced = float(gain) * n26 / n_scored
        total += priced
        rows[name] = {**table[name], "n_2026": n26, "per_row_gain_used": float(gain), "borrowed_from": source,
                      "thin": bool(table[name]["n_2025"] < thin), "priced_mse": priced}
    return {"priced_mse": float(total), "n_scored_2026": int(n_scored), "thin_threshold": int(thin), "bins": rows,
            "passes": bool(total >= C2_MIN_MSE)}


def _sum_by(g, key) -> dict:
    key = np.asarray(key)
    return {str(k): {"sum_g": float(g[key == k].sum()), "n": int((key == k).sum())} for k in sorted(np.unique(key), key=str)}


def clauses(t: pd.DataFrame, counts_2026: dict, n_boot=N_BOOT, seed=BOOT_SEED, n_scored=N_SCORED_2026) -> dict:
    """C1-C7 on the non-LIRF rows `t` (columns y, month, date, ADEP_mvt, hprox_med, S1, S1C and the per-seed arms).
    Every threshold is the prereg's; nothing here is tuned."""
    if (t.ADEP_mvt == ROME).any():
        raise ValueError("clauses are evaluated on non-LIRF rows only; LIRF rows were passed in")
    g = gains(t)
    pooled = float(g.sum())
    y, h = t.y.to_numpy(dtype="float64"), t.hprox_med.to_numpy(dtype="float64")
    c1 = date_block_bootstrap(g, t.date.to_numpy(), n_boot, seed)
    c2 = price_2026(bin_table(g, h), counts_2026, n_scored)
    months = _sum_by(g, t.month.to_numpy().astype("int64"))
    k3 = sum(1 for r in months.values() if r["sum_g"] > 0.0)
    c3 = {"months_positive": int(k3), "n_months": int(len(months)), "per_month": months,
          "passes": bool(k3 >= MIN_MONTHS and len(months) == N_MONTHS)}
    airports = _sum_by(g, t.ADEP_mvt.astype(str).to_numpy())
    k4 = sum(1 for r in airports.values() if r["sum_g"] > 0.0)
    c4 = {"airports_positive": int(k4), "n_airports": int(len(airports)), "per_airport": airports,
          "passes": bool(k4 >= MIN_AIRPORTS and len(airports) == N_AIRPORTS)}
    calm = (h < CALM_S)                                   # NaN compares False: the NaN bin is not "calm"
    calm_sum = float(g[calm].sum())
    c5 = {"calm_sum_g": calm_sum, "n_calm": int(calm.sum()), "pooled_sum_g": pooled, "bound": -CALM_TOLERANCE * pooled,
          "passes": bool(calm_sum >= -CALM_TOLERANCE * pooled)}
    exm = y <= MONSTER_S
    c6 = {"exmonster_sum_g": float(g[exm].sum()), "n_exmonster": int(exm.sum()), "passes": bool(g[exm].sum() > 0.0)}
    per_seed = {}
    for s in SEEDS:
        se1 = (y - t[f"S1_seed{s}"].to_numpy(dtype="float64")) ** 2
        sec = (y - t[f"S1C_seed{s}"].to_numpy(dtype="float64")) ** 2
        per_seed[str(s)] = float((se1 - sec).sum())
    sd = sf.seed_sd(per_seed.values())
    c7 = {"pooled_sum_g": pooled, "per_seed_sum_g": per_seed, "seed_sd": sd, "passes": sf.exceeds_2x_seed_sd(pooled, sd)}
    out = {"C1_date_block_bootstrap": c1, "C2_priced_2026": c2, "C3_months": c3, "C4_airports": c4, "C5_calm_control": c5,
           "C6_not_the_monsters": c6, "C7_seeds": c7, "pooled_sum_g": pooled, "n_rows": int(len(t))}
    out["verdict"] = verdict(out)
    return out


def verdict(c: dict) -> str:
    """WORKING iff all seven pass; NOT WORKING iff C1 fails; else INCONCLUSIVE."""
    keys = ["C1_date_block_bootstrap", "C2_priced_2026", "C3_months", "C4_airports", "C5_calm_control",
            "C6_not_the_monsters", "C7_seeds"]
    passes = [bool(c[k]["passes"]) for k in keys]
    if all(passes):
        return "WORKING"
    if not passes[0]:
        return "NOT WORKING"
    return "INCONCLUSIVE"


def rmse_pair(t: pd.DataFrame, mask=None) -> dict:
    y = t.y.to_numpy(dtype="float64")
    m = np.ones(len(t), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    out = {"n": int(m.sum())}
    for a in ("S1", "S1C"):
        e = (y[m] - t[a].to_numpy(dtype="float64")[m]) ** 2
        out[a] = float(np.sqrt(e.mean())) if m.any() else None
    out["gain_s"] = (out["S1"] - out["S1C"]) if m.any() else None
    return out


# =============================================================================================
# the 2026 witness (light): the scored non-LIRF unmatched rows of the ranking file, binned
# =============================================================================================

def witness_2026_from_frames(ranking: pd.DataFrame, template_ids) -> pd.DataFrame:
    """`ranking` = derive(load_movements([ranking file])) -- every DEP row of the serve file. The witness is
    taken from ITS matched departures (AOBT_3 present); the scored non-LIRF unmatched rows get their bin."""
    scored = ranking[ranking.MVT_ID_mvt.isin(set(template_ids))]
    if len(scored) != len(template_ids):
        raise ValueError(f"joined {len(scored)} scored rows, template has {len(template_ids)}")
    wit = airport_hour_witness(ranking)
    rows = scored[scored.unmatched.to_numpy() & (scored.ADEP_mvt != ROME).to_numpy()]
    rows = attach_witness(rows, wit)
    return pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt.to_numpy(), "ADEP_mvt": rows.ADEP_mvt.to_numpy(),
                         "month": rows.month.to_numpy().astype("int64"), "sp": rows.sp.to_numpy(dtype="float64"),
                         "hprox_med": rows.hprox_med.to_numpy(dtype="float64"), "hprox_n": rows.hprox_n.to_numpy(dtype="int64"),
                         "bin": bin_of(rows.hprox_med)})


def bin_counts(bins) -> dict:
    vc = pd.Series(np.asarray(bins, dtype=object)).value_counts()
    return {name: int(vc.get(name, 0)) for name in BINS}


def load_witness_2026(log) -> tuple:
    template = pq.read_table(RAW / "submitting.parquet").to_pandas()
    if len(template) != N_SCORED_2026:
        raise AssertionError(f"the template has {len(template):,} rows; the prereg's C2 denominator is {N_SCORED_2026:,}")
    ranking = bs.derive(bs.load_movements([RAW / "ranking.parquet"]))
    log(f"ranking: {len(ranking):,} DEP rows, {int((~ranking.unmatched).sum()):,} matched; template {len(template):,}")
    w = witness_2026_from_frames(ranking, template.MVT_ID_mvt.to_numpy())
    del ranking
    gc.collect()
    return w, int(len(template))


def smoke_witness_2026(seed: int = 1) -> pd.DataFrame:
    """A synthetic 'serve file': synthetic_world(seed)'s unmatched rows in months 1 and 7, non-LIRF."""
    unm, _, _ = synthetic_world(seed=seed)
    rows = unm[unm.month.isin(list(FOLD_A)) & (unm.ADEP_mvt != ROME)]
    return pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt.to_numpy(), "ADEP_mvt": rows.ADEP_mvt.to_numpy(),
                         "month": rows.month.to_numpy().astype("int64"), "sp": rows.sp.to_numpy(dtype="float64"),
                         "hprox_med": rows.hprox_med.to_numpy(dtype="float64"), "hprox_n": rows.hprox_n.to_numpy(dtype="int64"),
                         "bin": bin_of(rows.hprox_med)})


# =============================================================================================
# records, paths, steps
# =============================================================================================

def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2)


def _git() -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"git_sha": run("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def output_paths(smoke: bool) -> dict:
    """Real: the parquets under data/cache_stand/, the json and logs under reports/. Smoke: everything under
    data/smoke_unm_congestion/, never the real paths."""
    if smoke:
        return {k: SMOKE_DIR / v for k, v in dict(lomo=LOMO_NAME, witness_2026=WITNESS_2026_NAME, json=JSON_NAME,
                                                  lomo_log=LOMO_LOG, score_log=SCORE_LOG).items()}
    return {"lomo": CACHE / LOMO_NAME, "witness_2026": CACHE / WITNESS_2026_NAME, "json": REPORTS / JSON_NAME,
            "lomo_log": REPORTS / LOMO_LOG, "score_log": REPORTS / SCORE_LOG}


def write_lomo(preds: pd.DataFrame, path: pathlib.Path, extra: dict) -> None:
    """The per-row parquet with its convention stamped in the file metadata (BC-2)."""
    table = pa.Table.from_pandas(preds[PRED_COLUMNS], preserve_index=False)
    meta = {b"convention": CONVENTION.encode(), b"prereg": PREREG.encode(), b"tag": TAG.encode(),
            b"unm_congestion": json.dumps(extra, default=_jsonable).encode()}
    pq.write_table(table.replace_schema_metadata({**(table.schema.metadata or {}), **meta}), path)


def read_lomo(path: pathlib.Path) -> tuple:
    """The stored predictions and their metadata; refuses a file without the taxi-time stamp."""
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    conv = meta.get(b"convention", b"").decode()
    if conv != CONVENTION:
        raise ValueError(f"{path.name} carries convention {conv!r}, expected {CONVENTION!r} (reports/bug_classes.md BC-2)")
    return table.to_pandas(), json.loads(meta.get(b"unm_congestion", b"{}").decode())


def run_lomo(args, paths, log) -> int:
    started = dt.datetime.now(dt.timezone.utc)
    if args.smoke:
        unm, lirf, wsum = synthetic_world(seed=0)
        log(f"synthetic world: {len(unm):,} unmatched rows, {len(lirf):,} LIRF matched rows; witness {wsum}")
    else:
        files = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
        if len(files) != 12:
            raise SystemExit(f"expected the 12 training months under {RAW}, found {len(files)}")
        if not REFERENCE.exists():
            raise SystemExit(f"the reproduction reference {REFERENCE} is missing; the prereg's guard cannot run")
        unm, lirf, wsum = load_real(files, log)
    fill = bs.schedule_fill(unm)
    log(f"stratum: {len(unm):,} rows, fill share {100 * fill.mean():.1f}%, non-fill {int((~fill).sum()):,}; LIRF matched {len(lirf):,}; "
        f"witness bins on the unmatched rows {wsum['unmatched_bin_counts']}")
    log("--- 12-fold LOMO: S1 (stratum_fold.stratum_arms) and S1C ---")
    t = time.time()
    preds = score_lomo(unm, lirf, SEEDS, log)
    log(f"LOMO scored {len(preds):,} rows in {time.time() - t:.0f}s; peak RSS {peak_rss_gb():.2f} GB")
    if args.smoke:
        repro = {"status": "skipped (synthetic)", "tolerance_s": REPRO_TOL_S}
    else:
        repro = check_reproduction(preds, pd.read_parquet(REFERENCE))
    log(f"reproduction guard: {repro}")
    extra = {"prereg": PREREG, "tag": TAG, "smoke": bool(args.smoke), **_git(), "started_utc": started.isoformat(),
             "seeds": list(SEEDS), "witness": {"columns": WITNESS, "min_n": MIN_N, **wsum}, "reproduction": repro,
             "regressor": {"params": dict(bs.NF_PARAMS), "numeric": CongestionRegressor.NUMERIC, "encoded": list(bs.NF_ENCODED),
                           "winsor_s": bs.WINSOR_S, "T_tail_s": bs.T_TAIL_S},
             "n_rows": int(len(preds)), "n_non_lirf": int((preds.ADEP_mvt != ROME).sum()),
             "wall_s": round(time.time() - _T0, 1), "peak_rss_gb": round(peak_rss_gb(), 2)}
    write_lomo(preds, paths["lomo"], extra)
    non = preds[preds.ADEP_mvt != ROME]
    r = rmse_pair(non)
    log(f"predictions -> {paths['lomo']} ({len(preds):,} rows, convention {CONVENTION}); non-LIRF RMSE S1 {r['S1']:.2f} S1C {r['S1C']:.2f} "
        f"(descriptive; the clauses are the `score` step)   wall {extra['wall_s']:.0f}s   peak RSS {extra['peak_rss_gb']:.2f} GB")
    return 0


def run_score(args, paths, log) -> int:
    started = dt.datetime.now(dt.timezone.utc)
    n_boot = SMOKE["n_boot"] if args.smoke else N_BOOT
    preds, lomo_meta = read_lomo(paths["lomo"])
    if not args.smoke and lomo_meta.get("reproduction", {}).get("status") != "reproduced":
        raise AssertionError(f"the stored lomo parquet's reproduction guard reads {lomo_meta.get('reproduction')}; refusing to score")
    preds = preds[preds.fold == "lomo"].reset_index(drop=True)
    if args.smoke:
        w26, n_scored = smoke_witness_2026(seed=1), int(len(preds))
    else:
        w26, n_scored = load_witness_2026(log)
    w26.to_parquet(paths["witness_2026"], index=False)
    counts = bin_counts(w26["bin"])
    log(f"2026 witness: {len(w26):,} scored non-LIRF unmatched rows, bins {counts} -> {paths['witness_2026']}")
    non = preds[preds.ADEP_mvt != ROME].reset_index(drop=True)
    lirf = preds[preds.ADEP_mvt == ROME].reset_index(drop=True)
    c = clauses(non, counts, n_boot=n_boot, seed=BOOT_SEED, n_scored=N_SCORED_2026 if not args.smoke else n_scored)
    g = gains(non)
    mon = sf.monster_mask(non.y.to_numpy())
    fold_a = non.month.isin(list(FOLD_A)).to_numpy()
    lirf_g = gains(lirf) if len(lirf) else np.zeros(0)
    record = {"mode": "unm_congestion_score", "prereg": PREREG, "tag": TAG, "smoke": bool(args.smoke), **_git(),
              "started_utc": started.isoformat(), "commands": {"lomo": COMMAND_LOMO, "score": COMMAND_SCORE},
              "lomo_metadata": lomo_meta,
              "config": {"seeds": list(SEEDS), "n_boot": int(n_boot), "boot_seed": BOOT_SEED, "blocks": "calendar dates (UTC) of MVT_TIME",
                         "bins": BINS, "bin_edges": [float(e) for e in BIN_EDGES], "thin_bin": THIN_BIN, "n_scored_2026": N_SCORED_2026,
                         "c2_min_mse": C2_MIN_MSE, "min_months": MIN_MONTHS, "min_airports": MIN_AIRPORTS, "calm_s": CALM_S,
                         "calm_tolerance": CALM_TOLERANCE, "monster_s": MONSTER_S, "witness_min_n": MIN_N},
              "data": {"n_lomo_rows": int(len(preds)), "n_non_lirf": int(len(non)), "n_lirf": int(len(lirf)),
                       "n_2026_scored_non_lirf_unmatched": int(len(w26)), "bins_2026": counts,
                       "bins_2025_non_lirf": bin_counts(bin_of(non.hprox_med))},
              "clauses": c, "verdict": c["verdict"],
              "descriptive": {"non_lirf": {"pooled": rmse_pair(non), "exmonster": rmse_pair(non, ~mon)},
                              "fold_a_months": {"months": list(FOLD_A), "sum_g": float(g[fold_a].sum()), "n": int(fold_a.sum()),
                                                "rmse": rmse_pair(non, fold_a),
                                                "per_month": {str(m): float(g[non.month.to_numpy() == m].sum()) for m in FOLD_A}},
                              "other_months": {"sum_g": float(g[~fold_a].sum()), "n": int((~fold_a).sum()), "rmse": rmse_pair(non, ~fold_a)},
                              "lirf_report_only": {"n": int(len(lirf)), "sum_g": float(lirf_g.sum()),
                                                   "rmse": rmse_pair(lirf) if len(lirf) else None}},
              "judgement_calls": [
                  "the witness hour is the UTC clock hour of MVT_TIME_UTC_mvt floored; the C1 blocks are the UTC calendar dates of the same clock",
                  "hprox_n is 0 (and hprox_med NaN) for an airport-hour with no matched departure in the file",
                  "the 2026 witness is taken from every matched DEP row of data/raw/ranking.parquet (the serve file), not only the scored rows",
                  "C2 borrowing chains downward through the numeric bins; the NaN bin and the <900 bin have no lower bin, so a thin "
                  "one there is priced at 0 and flagged `borrowed_from: null` (unreachable on the real frame; named for honesty)",
                  "C5's calm mask is hprox_med < 1,100 -- NaN compares False, so the NaN bin is not calm",
                  "C7's seed sd is the ddof=1 sd of the three single-seed sums of g (stratum_fold.seed_sd); the rule is strict >"]}
    # ---- readable tables ----
    log(f"=== E3C on the non-LIRF LOMO rows: n={len(non):,}, sum g {c['pooled_sum_g']:+,.0f} s^2 ===")
    r = record["descriptive"]["non_lirf"]
    log(f"  RMSE pooled S1 {r['pooled']['S1']:.2f} S1C {r['pooled']['S1C']:.2f} gain {r['pooled']['gain_s']:+.2f} s; "
        f"ex-monster S1 {r['exmonster']['S1']:.2f} S1C {r['exmonster']['S1C']:.2f} gain {r['exmonster']['gain_s']:+.2f} s")
    c1 = c["C1_date_block_bootstrap"]
    log(f"  C1 date-block bootstrap ({c1['n_blocks']} dates, {c1['n_draws']} draws, seed {c1['seed']}): sum g {c1['sum_g']:+,.0f} "
        f"95% [{c1['ci95'][0]:+,.0f}, {c1['ci95'][1]:+,.0f}]  passes {c1['passes']}")
    c2 = c["C2_priced_2026"]
    log(f"  C2 2026-priced gain {c2['priced_mse']:+.1f} board MSE (>= {C2_MIN_MSE:.0f}: {c2['passes']}); denominator {c2['n_scored_2026']:,}")
    log(f"  {'bin':10s} {'n2025':>7s} {'per-row g':>12s} {'used':>12s} {'from':>10s} {'n2026':>7s} {'priced':>9s}")
    for name in BINS:
        b = c2["bins"][name]
        prg = "n/a" if b["per_row_gain"] is None else f"{b['per_row_gain']:+.1f}"
        log(f"  {name:10s} {b['n_2025']:7,} {prg:>12s} {b['per_row_gain_used']:+12.1f} {str(b['borrowed_from']):>10s} {b['n_2026']:7,} {b['priced_mse']:+9.2f}")
    c3, c4 = c["C3_months"], c["C4_airports"]
    log(f"  C3 months with sum g > 0: {c3['months_positive']} of {c3['n_months']} (>= {MIN_MONTHS}: {c3['passes']})  "
        + "  ".join(f"{m}:{v['sum_g']:+,.0f}" for m, v in c3["per_month"].items()))
    log(f"  C4 airports with sum g > 0: {c4['airports_positive']} of {c4['n_airports']} (>= {MIN_AIRPORTS}: {c4['passes']})  "
        + "  ".join(f"{a}:{v['sum_g']:+,.0f}" for a, v in c4["per_airport"].items()))
    c5, c6, c7 = c["C5_calm_control"], c["C6_not_the_monsters"], c["C7_seeds"]
    log(f"  C5 calm (hprox_med < {CALM_S:.0f}, n {c5['n_calm']:,}) sum g {c5['calm_sum_g']:+,.0f} >= {c5['bound']:+,.0f}: {c5['passes']}")
    log(f"  C6 ex-monster (y <= {MONSTER_S:.0f}, n {c6['n_exmonster']:,}) sum g {c6['exmonster_sum_g']:+,.0f} > 0: {c6['passes']}")
    log(f"  C7 pooled {c7['pooled_sum_g']:+,.0f} vs 2 x seed sd {2 * c7['seed_sd']:,.0f} (per seed {c7['per_seed_sum_g']}): {c7['passes']}")
    fa = record["descriptive"]["fold_a_months"]
    log(f"  fold-A months {FOLD_A} (seen by the exploratory look): sum g {fa['sum_g']:+,.0f} on {fa['n']:,} rows {fa['per_month']}; "
        f"other ten months sum g {record['descriptive']['other_months']['sum_g']:+,.0f}")
    lr = record["descriptive"]["lirf_report_only"]
    log(f"  LIRF (report-only, keeps R2/E1): n {lr['n']:,} sum g {lr['sum_g']:+,.0f} rmse {lr['rmse']}")
    log(f"  VERDICT ({PREREG}): {c['verdict']}")
    record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    record["wall_s"] = round(time.time() - _T0, 1)
    record["peak_rss_gb"] = round(peak_rss_gb(), 2)
    paths["json"].write_text(json.dumps(record, indent=2, default=_jsonable))
    log(f"json -> {paths['json']}   wall {record['wall_s']:.0f}s   peak RSS {record['peak_rss_gb']:.2f} GB")
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("lomo", "score"))
    ap.add_argument("--smoke", action="store_true", help=f"synthetic world, {SMOKE['n_boot']} draws, {SMOKE_DIR.relative_to(ROOT)}/, the banner")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    paths = output_paths(args.smoke)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    log = _Log(paths["lomo_log"] if args.step == "lomo" else paths["score_log"])
    try:
        if args.smoke:
            log(SMOKE_BANNER)
        log(f"unm_congestion {args.step}  smoke={args.smoke}  prereg={PREREG}  paths={ {k: str(v) for k, v in paths.items()} }")
        rc = run_lomo(args, paths, log) if args.step == "lomo" else run_score(args, paths, log)
    finally:
        if args.smoke:
            log(SMOKE_BANNER)
        log.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
