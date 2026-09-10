"""`scripts/unm_congestion_resid.py` -- arm E3R of plans/PREREG_unm_congestion_resid_2026_09_10.md (owner decision 2026-09-10).

    python scripts/unm_congestion_resid.py lomo            # the heavy step: 12 months from data/raw, ~3 GB, run ALONE
    python scripts/unm_congestion_resid.py score           # the light step: clauses C1-C7 from the stored parquet + 2026 witness
    python scripts/unm_congestion_resid.py lomo --smoke    # synthetic world, code path only, NOT a result
    python scripts/unm_congestion_resid.py score --smoke

H-E3R: for non-fill unmatched rows with a witness, predicting the OFFSET y - hprox_med (winsorised at +-3,000 s) and
adding hprox_med back beats S1C out of month on non-LIRF rows, with the gain in the hot bins and no material loss in
calm ones. S1C's fit target is y capped at 3,000 s, so it cannot follow a witness above the cap; the offset model
takes its level from the witness and caps only the offset.

Arms (12-fold LOMO over 2025, E3C's folds, frames and witness; `unm_congestion` is composed, never edited):
  S1C   exactly `unm_congestion.congestion_arms`' S1C (what v9 / v10 ship) -- GUARD: must equal
        data/cache_stand/unm_congestion_lomo.parquet's S1C on every row to <= 1e-6 s, joined on MVT_ID, or nothing
        is written (S1 and the per-seed S1C arms are held to the same bar, so the chain back to v7 holds);
  S1R   identical p, nf_cells, sp, seeds, params, encodings and non-fill training rows. The body: on the non-fill
        training rows WITH a witness, target r = clip(y - hprox_med, -3,000, +3,000); features = S1C's exactly
        (hprox_med and hprox_n stay inputs); target encodings and the prior computed on r, smoothed as S1C's;
        prediction nf_fit_R = hprox_med + r_hat. Rows WITHOUT a witness (train and test) use S1C's nf_fit unchanged
        (per seed). nf = nf_hybrid(nf_cells, mean over seeds of nf_fit_R); pred = mixture(p, sp, nf).
Per-seed versions of both feed clause C7.

Clauses (locked in the prereg; g = SE(S1C) - SE(S1R) per non-LIRF unmatched 2025 row, raw labels):
  C1 date-block bootstrap (calendar dates, 2,000 draws, seed 0) of sum g: lower 95% bound > 0
  C2 EVENT-EXCLUDED price: drop the ONE airport-month with the largest |sum g|, recompute the per-bin per-row gains
     (E3C's bins and < 30-rows-borrows-next-lower rule), price on the 2026 scored non-LIRF unmatched bin counts
     / 344,841: >= +300 board MSE. The full (non-excluded) price is reported, not decisional.
  C3 sum g > 0 in >= 8 of 12 months          C4 sum g > 0 at >= 6 of the 9 non-LIRF airports
  C5 sum g over hprox_med < 1,100 >= -10% of the pooled sum g        C6 sum g over y <= 10,800 > 0
  C7 pooled gain > 2 x sd of the three single-seed gains (S1R_seed s vs S1C_seed s)
Verdict: WORKING iff all seven pass; NOT WORKING iff C1 fails; else INCONCLUSIVE. Never revised.
C1 and C3-C7 are `unm_congestion.clauses`' arithmetic run on a renamed view (control S1C in the S1 slot, treatment
S1R in the S1C slot); C2 is rebuilt here from `bin_table` / `price_2026` on the event-excluded rows with the +300 bar.

Outputs. `lomo`: data/cache_stand/unm_congestion_resid_lomo.parquet (TAXI-TIME convention stamped in the metadata,
reports/bug_classes.md BC-2) and reports/unm_congestion_resid_lomo.log. `score`: reports/unm_congestion_resid.json
and reports/unm_congestion_resid_score.log. `--smoke` writes everything under data/smoke_unm_congestion_resid/,
uses 200 draws, brackets the log with the banner and never touches the real paths.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import unm_congestion as uc  # noqa: E402

sf = uc.sf
bs = uc.bs
RAW, REPORTS, CACHE = uc.RAW, uc.REPORTS, uc.CACHE
SMOKE_DIR = ROOT / "data" / "smoke_unm_congestion_resid"
REFERENCE = CACHE / uc.LOMO_NAME                      # E3C's stored LOMO predictions: the reproduction reference
E3C_WITNESS_2026 = CACHE / uc.WITNESS_2026_NAME       # E3C's stored 2026 witness rows: the bin counts must agree
PREREG = "plans/PREREG_unm_congestion_resid_2026_09_10.md"
PARENT_PREREG = uc.PREREG
TAG = "E3R"

# ---- the registered design, frozen ----
CLIP_S = bs.WINSOR_S                                  # the offset is winsorised at +-3,000 s (the prereg's number)
if CLIP_S != 3_000.0:
    raise ImportError(f"the prereg caps the offset at +-3,000 s; build_submission.WINSOR_S reads {CLIP_S}")
SEEDS = uc.SEEDS                                      # (0, 1, 2)
N_BOOT, BOOT_SEED = uc.N_BOOT, uc.BOOT_SEED           # C1: 2,000 date-block draws, seed 0
C2_MIN_MSE = 300.0                                    # C2: the event-excluded price bar (E3C's was 500 on the full price)
N_SCORED_2026 = uc.N_SCORED_2026                      # 344,841
CONTROL, TREATMENT = "S1C", "S1R"
ROME, FOLD_A, MONSTER_S, CALM_S = uc.ROME, uc.FOLD_A, uc.MONSTER_S, uc.CALM_S
REPRO_TOL_S = uc.REPRO_TOL_S
CONVENTION = uc.CONVENTION                            # "taxi_time"
SMOKE_BANNER = uc.SMOKE_BANNER
SMOKE = dict(n_boot=200)
ARMS_R = ("S1R", *[f"S1R_seed{s}" for s in SEEDS])
GUARD_COLUMNS = ("S1C", "S1", *[f"S1C_seed{s}" for s in SEEDS])   # S1C is the registered guard; the rest ride along
PRED_COLUMNS = [*uc.PRED_COLUMNS, "nf_fit_r", "r_hat", *ARMS_R]
META_KEY = b"unm_congestion_resid"
LOMO_NAME = "unm_congestion_resid_lomo.parquet"
JSON_NAME, LOMO_LOG, SCORE_LOG = "unm_congestion_resid.json", "unm_congestion_resid_lomo.log", "unm_congestion_resid_score.log"
_PY = "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 /opt/homebrew/bin/python3.11 -B -u"
COMMAND_LOMO = (f"cd ~/Projects/prc-challenge && {_PY} scripts/unm_congestion_resid.py lomo "
                "> reports/unm_congestion_resid_lomo.console.log 2>&1; echo \"exit $?\"")
COMMAND_SCORE = (f"cd ~/Projects/prc-challenge && {_PY} scripts/unm_congestion_resid.py score "
                 "> reports/unm_congestion_resid_score.console.log 2>&1; echo \"exit $?\"")

_T0 = time.time()
_Log = uc._Log
peak_rss_gb = uc.peak_rss_gb
_git = uc._git
_jsonable = uc._jsonable


# =============================================================================================
# the S1R body: S1C's regressor fitted on the clipped offset to the witness
# =============================================================================================

class ResidualRegressor(uc.CongestionRegressor):
    """S1C's regressor (NF_NUMERIC + the witness, the same NF_PARAMS, the same SMOOTH-smoothed encodings of
    NF_ENCODED, the same seed semantics) fitted on the non-fill training rows WITH a witness, on the target
    r = clip(y - hprox_med, -CLIP_S, +CLIP_S). The prior and the encodings are computed on r, exactly as the parent
    computes them on its (winsorised) target: the parent receives a frame whose `y` IS r, so its own winsorisation
    at WINSOR_S (== CLIP_S, asserted at import) is the identity. `predict_offset` returns r_hat; `predict` returns
    hprox_med + r_hat, NaN where the row has no witness -- the caller falls back to S1C's nf_fit there."""

    def __init__(self, train_nonfill_witness: pd.DataFrame, seed: int):
        frame = train_nonfill_witness
        missing = [c for c in ("y", *uc.WITNESS) if c not in frame.columns]
        if missing:
            raise ValueError(f"ResidualRegressor needs columns {missing}")
        h = frame.hprox_med.to_numpy(dtype="float64")
        n_nan = int(np.isnan(h).sum())
        if n_nan:
            raise ValueError(f"ResidualRegressor fits rows WITH a witness only; {n_nan} of {len(frame)} rows have hprox_med NaN")
        raw = frame.y.to_numpy(dtype="float64") - h
        r = np.clip(raw, -CLIP_S, CLIP_S)
        self.n_clipped_low = int((raw < -CLIP_S).sum())
        self.n_clipped_high = int((raw > CLIP_S).sum())
        self.n_train_witness = int(len(frame))
        super().__init__(frame.assign(y=r), seed)

    def predict_offset(self, frame: pd.DataFrame) -> np.ndarray:
        """r_hat for every row of `frame` (a row without a witness gets the model's native-missing answer; the
        caller never uses it)."""
        return bs.NonFillRegressor.predict(self, frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """hprox_med + r_hat; NaN where the row has no witness."""
        return frame.hprox_med.to_numpy(dtype="float64") + self.predict_offset(frame)


def fit_residual_regressor(train_unmatched_nonfill_witness: pd.DataFrame, seed: int) -> ResidualRegressor:
    """The S1R body. The same refusals as `unm_congestion.fit_congestion_regressor` (no fill rows, a label present)
    plus: every row must carry a witness."""
    frame = train_unmatched_nonfill_witness
    if len(frame) == 0 or "y" not in frame.columns:
        raise ValueError("fit_residual_regressor needs non-fill training rows with a label `y`")
    if "BLOCK_TIME_UTC_mvt" in frame.columns and "SCHED_TIME_UTC_mvt" in frame.columns:
        n_fill = int(bs.schedule_fill(frame).sum())
        if n_fill:
            raise ValueError(f"fit_residual_regressor expects NON-FILL rows only; {n_fill} fill rows found")
    return ResidualRegressor(frame, seed)


def residual_nf_fit(hprox_med, r_hat, nf_fit_c) -> np.ndarray:
    """nf_fit_R per row: hprox_med + r_hat where the row has a witness, S1C's nf_fit (`nf_fit_c`) where it has none.
    Finite everywhere by construction; raises otherwise."""
    h, r, c = (np.asarray(v, dtype="float64") for v in (hprox_med, r_hat, nf_fit_c))
    if not (h.shape == r.shape == c.shape):
        raise ValueError(f"shape mismatch: hprox_med {h.shape}, r_hat {r.shape}, nf_fit_c {c.shape}")
    has = ~np.isnan(h)
    out = np.where(has, h + r, c)
    if not np.isfinite(out).all():
        raise ValueError("nf_fit_R is not finite on every row")
    return out


def residual_body(train_unm: pd.DataFrame, test_unm: pd.DataFrame, nf_fit_c_by_seed: dict, seeds=SEEDS) -> dict:
    """The S1R body on `test_unm`, per seed and averaged, from S1C's per-seed body as the no-witness fallback.
    The regressor is fitted on `train_unm[~schedule_fill][hprox_med present]` -- fit_unmatched's own non-fill rows,
    reduced to those with a witness. Returns nf_fit_r, nf_fit_r_by_seed, r_hat (NaN where no witness), r_hat_by_seed,
    n_train_nonfill, n_train_witness, n_clipped_low / high (seed-independent)."""
    for c in uc.WITNESS:
        if c not in train_unm.columns or c not in test_unm.columns:
            raise ValueError(f"witness column {c!r} missing from the unmatched frames")
    seeds = tuple(int(s) for s in seeds)
    if set(seeds) != set(int(s) for s in nf_fit_c_by_seed):
        raise ValueError(f"nf_fit_c_by_seed carries seeds {sorted(nf_fit_c_by_seed)}, the body needs {seeds}")
    nonfill = train_unm[~bs.schedule_fill(train_unm)]              # fit_unmatched's `tr[~tr.fill]`, same rows, same order
    fitted = nonfill[nonfill.hprox_med.notna()]
    h = test_unm.hprox_med.to_numpy(dtype="float64")
    has = ~np.isnan(h)
    off_by_seed, nf_by_seed, clipped = {}, {}, None
    for s in seeds:
        reg = fit_residual_regressor(fitted, s)
        off_by_seed[s] = reg.predict_offset(test_unm)
        nf_by_seed[s] = residual_nf_fit(h, off_by_seed[s], nf_fit_c_by_seed[s])
        clipped = (reg.n_clipped_low, reg.n_clipped_high)
    return {"nf_fit_r": bs.mean_over_seeds(nf_by_seed), "nf_fit_r_by_seed": nf_by_seed,
            "r_hat": np.where(has, bs.mean_over_seeds(off_by_seed), np.nan),
            "r_hat_by_seed": {s: np.where(has, v, np.nan) for s, v in off_by_seed.items()},
            "n_train_nonfill": int(len(nonfill)), "n_train_witness": int(len(fitted)),
            "n_clipped_low": int(clipped[0]), "n_clipped_high": int(clipped[1])}


def residual_arms(train_unm: pd.DataFrame, test_unm: pd.DataFrame, train_matched: pd.DataFrame, seeds=SEEDS) -> tuple:
    """S1, S1C and their per-seed arms exactly as `unm_congestion.congestion_arms`, plus S1R / S1R_seed* rebuilt from
    S1's OWN parts: p, sp and nf_cells untouched, only the body replaced, S1C's per-seed body as the no-witness
    fallback. `parts` gains nf_fit_r, nf_fit_r_by_seed, r_hat, r_hat_by_seed and the training counts."""
    arms, parts = uc.congestion_arms(train_unm, test_unm, train_matched, seeds=seeds)
    body = residual_body(train_unm, test_unm, parts["nf_fit_c_by_seed"], seeds=seeds)
    arms["S1R"] = bs.mixture(parts["p"], parts["sp"], bs.nf_hybrid(parts["nf_cells"], body["nf_fit_r"]))
    for s in seeds:
        arms[f"S1R_seed{s}"] = bs.mixture(parts["p"], parts["sp"], bs.nf_hybrid(parts["nf_cells"], body["nf_fit_r_by_seed"][int(s)]))
    parts.update(body)
    return arms, parts


def _score_split(unm, lirf_matched, tr, te, test_months, seeds, fold_label: str) -> pd.DataFrame:
    if tuple(int(s) for s in seeds) != tuple(SEEDS):
        raise ValueError(f"the harness scores the registered seeds {SEEDS} (every per-seed column is stored), got {tuple(seeds)!r}")
    tr_unm, te_unm = unm[tr], unm[te]
    tr_mat = lirf_matched[~lirf_matched.month.isin(list(test_months))]
    arms, parts = residual_arms(tr_unm, te_unm, tr_mat, seeds=seeds)
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
    out["nf_fit_r"] = np.asarray(parts["nf_fit_r"], dtype="float64")
    out["r_hat"] = np.asarray(parts["r_hat"], dtype="float64")
    for a in PRED_COLUMNS:
        if a.startswith("S1"):
            out[a] = np.asarray(arms[a], dtype="float64")
    return out[PRED_COLUMNS]


def score_lomo(unm: pd.DataFrame, lirf_matched: pd.DataFrame, seeds, log) -> pd.DataFrame:
    """Out-of-fold predictions for every stratum row: E3C's LOMO loop with the S1R arms added."""
    month = unm.month.to_numpy()
    parts, scored = [], np.zeros(len(unm), dtype="int64")
    for m, tr, te in sf.lomo_folds(month):
        t = time.time()
        part = _score_split(unm, lirf_matched, tr, te, [m], seeds, "lomo")
        scored += te
        parts.append(part)
        d = part.S1R.to_numpy() != part.S1C.to_numpy()
        nan = part.hprox_med.isna().to_numpy()
        log(f"  fold {m:2d}: test {int(te.sum()):5,}  train {int(tr.sum()):6,}  routed {int(part.routed_nf_fit.sum()):5,} -> body; "
            f"S1R differs from S1C on {int(d.sum()):5,} rows; witness NaN on {int(nan.sum()):5,} (S1R == S1C there: {bool((~d[nan]).all())})  "
            f"({time.time() - t:.1f}s)")
    if not (scored == 1).all():
        raise AssertionError("LOMO did not score every row exactly once")
    return pd.concat(parts, ignore_index=True)[PRED_COLUMNS]


# =============================================================================================
# the reproduction guard (prereg: S1C must equal E3C's stored S1C on every row, or nothing is written)
# =============================================================================================

def check_reproduction(preds: pd.DataFrame, reference: pd.DataFrame, columns=GUARD_COLUMNS, tol: float = REPRO_TOL_S) -> dict:
    """Every lomo row's S1C (the registered guard) -- and S1 and the per-seed S1C arms with it -- within `tol` of
    E3C's stored value, joined on MVT_ID; the id sets must match exactly. Both frames are filtered to fold 'lomo'
    first (BC-2). Raises AssertionError on any miss."""
    ref = reference[reference.fold == "lomo"]
    have = preds[preds.fold == "lomo"]
    if ref.MVT_ID_mvt.duplicated().any() or have.MVT_ID_mvt.duplicated().any():
        raise AssertionError("duplicate MVT_ID in the lomo rows; the reproduction guard cannot join")
    if set(ref.MVT_ID_mvt) != set(have.MVT_ID_mvt):
        raise AssertionError(f"the harness scored {len(have):,} lomo rows, the reference has {len(ref):,}; different row sets, "
                             "nothing is written")
    ref_i = ref.set_index("MVT_ID_mvt")
    ids = have.MVT_ID_mvt.to_numpy()
    per_column = {}
    for col in columns:
        if col not in ref_i.columns:
            raise AssertionError(f"the reference has no column {col!r}; nothing is written")
        diff = np.abs(have[col].to_numpy(dtype="float64") - ref_i[col].reindex(ids).to_numpy(dtype="float64"))
        worst = float(diff.max()) if len(diff) else 0.0
        bad = int((diff > tol).sum())
        if bad:
            raise AssertionError(f"{col} does not reproduce E3C's stored {col} on {bad} of {len(have):,} lomo rows "
                                 f"(max |diff| {worst:.3g} s > {tol:g}); nothing is written")
        per_column[col] = worst
    return {"status": "reproduced", "n_rows": int(len(have)), "max_abs_diff_s": per_column, "tolerance_s": float(tol),
            "columns": list(columns), "reference": str(REFERENCE.relative_to(ROOT))}


def load_reference() -> pd.DataFrame:
    """E3C's stored LOMO parquet through `unm_congestion.read_lomo` (the taxi-time stamp is required, BC-2); its own
    reproduction guard against v7 must read 'reproduced'."""
    ref, meta = uc.read_lomo(REFERENCE)
    status = meta.get("reproduction", {}).get("status")
    if status != "reproduced":
        raise AssertionError(f"E3C's stored parquet carries reproduction status {status!r}; it cannot serve as the reference")
    return ref


# =============================================================================================
# clauses
# =============================================================================================

def gains(t: pd.DataFrame) -> np.ndarray:
    """g = SE(S1C) - SE(S1R) per row, against the RAW label. Never winsorised."""
    y = t.y.to_numpy(dtype="float64")
    return (y - t[CONTROL].to_numpy(dtype="float64")) ** 2 - (y - t[TREATMENT].to_numpy(dtype="float64")) ** 2


def e3c_view(t: pd.DataFrame) -> pd.DataFrame:
    """The frame `unm_congestion.clauses` scores, with E3R's control in the S1 slot and its treatment in the S1C slot:
    S1 := S1C, S1C := S1R, S1_seed s := S1C_seed s, S1C_seed s := S1R_seed s. Only the columns the clauses read."""
    cols = {"y": t.y, "month": t.month, "date": t.date, "ADEP_mvt": t.ADEP_mvt, "hprox_med": t.hprox_med,
            "S1": t[CONTROL], "S1C": t[TREATMENT]}
    for s in SEEDS:
        cols[f"S1_seed{s}"] = t[f"{CONTROL}_seed{s}"]
        cols[f"S1C_seed{s}"] = t[f"{TREATMENT}_seed{s}"]
    return pd.DataFrame({k: v.to_numpy() for k, v in cols.items()})


def airport_month_table(g, adep, month) -> pd.DataFrame:
    """sum g and n per (airport, month), sorted by |sum g| descending, ties by airport then month."""
    d = pd.DataFrame({"ADEP_mvt": np.asarray(adep).astype(str), "month": np.asarray(month).astype("int64"),
                      "g": np.asarray(g, dtype="float64")})
    t = d.groupby(["ADEP_mvt", "month"], observed=True).g.agg(sum_g="sum", n="size").reset_index()
    t["abs_sum_g"] = t.sum_g.abs()
    return t.sort_values(["abs_sum_g", "ADEP_mvt", "month"], ascending=[False, True, True]).reset_index(drop=True)


def largest_event(g, adep, month) -> dict:
    """The ONE airport-month with the largest |sum g| (a negative event counts as much as a positive one)."""
    t = airport_month_table(g, adep, month)
    if len(t) == 0:
        raise ValueError("no rows: no airport-month to exclude")
    r = t.iloc[0]
    return {"ADEP_mvt": str(r.ADEP_mvt), "month": int(r.month), "sum_g": float(r.sum_g), "n": int(r.n)}


def event_excluded_price(g, hprox_med, adep, month, counts_2026: dict, n_scored: int = N_SCORED_2026) -> dict:
    """C2: the largest |sum g| airport-month dropped, then E3C's bin_table / price_2026 (bins, thin rule, borrowing,
    denominator) on the rows that remain; passes iff >= C2_MIN_MSE (+300)."""
    g, h = np.asarray(g, dtype="float64"), np.asarray(hprox_med, dtype="float64")
    adep, month = np.asarray(adep).astype(str), np.asarray(month).astype("int64")
    ev = largest_event(g, adep, month)
    keep = ~((adep == ev["ADEP_mvt"]) & (month == ev["month"]))
    p = uc.price_2026(uc.bin_table(g[keep], h[keep]), counts_2026, n_scored)
    p["passes"] = bool(p["priced_mse"] >= C2_MIN_MSE)
    p["c2_min_mse"] = C2_MIN_MSE
    p["rule"] = "event-excluded (decisional)"
    p["excluded_airport_month"] = {**ev, "n_rows_kept": int(keep.sum()), "n_rows_dropped": int((~keep).sum()),
                                   "sum_g_kept": float(g[keep].sum())}
    return p


def clauses(t: pd.DataFrame, counts_2026: dict, n_boot=N_BOOT, seed=BOOT_SEED, n_scored=N_SCORED_2026) -> dict:
    """C1-C7 on the non-LIRF rows `t` (columns y, month, date, ADEP_mvt, hprox_med, S1C, S1R and the per-seed arms).
    C1, C3-C7 are E3C's arithmetic on the renamed view; C2 is the event-excluded price at +300; the full price is
    kept under its own key, not decisional. Every threshold is the prereg's; nothing here is tuned."""
    if (t.ADEP_mvt == ROME).any():
        raise ValueError("clauses are evaluated on non-LIRF rows only; LIRF rows were passed in")
    view = e3c_view(t)
    c = uc.clauses(view, counts_2026, n_boot=n_boot, seed=seed, n_scored=n_scored)
    g = gains(t)
    if not np.array_equal(g, uc.gains(view)):
        raise AssertionError("the renamed view's gains differ from E3R's gains; the view is wrong")
    full = dict(c["C2_priced_2026"])
    full.pop("passes")
    full["rule"] = "full price (reported, not decisional)"
    c["C2_full_price_not_decisional"] = full
    c["C2_priced_2026"] = event_excluded_price(g, t.hprox_med.to_numpy(dtype="float64"), t.ADEP_mvt.to_numpy(),
                                               t.month.to_numpy(), counts_2026, n_scored)
    c["airport_month_top"] = airport_month_table(g, t.ADEP_mvt.to_numpy(), t.month.to_numpy()).head(12).to_dict(orient="records")
    c["verdict"] = uc.verdict(c)
    return c


verdict = uc.verdict


def rmse_arms(t: pd.DataFrame, mask=None, arms=(CONTROL, TREATMENT)) -> dict:
    y = t.y.to_numpy(dtype="float64")
    m = np.ones(len(t), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    out = {"n": int(m.sum())}
    for a in arms:
        e = (y[m] - t[a].to_numpy(dtype="float64")[m]) ** 2
        out[a] = float(np.sqrt(e.mean())) if m.any() else None
    out["gain_s"] = (out[arms[0]] - out[arms[1]]) if m.any() else None
    return out


def bin_means(t: pd.DataFrame) -> dict:
    """Per registered witness bin: n, mean y, mean hprox_med, mean S1C, mean S1R, sum g -- the 'does S1R move toward
    the witness in the hot bins' table."""
    b = uc.bin_of(t.hprox_med)
    g = gains(t)
    y, h = t.y.to_numpy(dtype="float64"), t.hprox_med.to_numpy(dtype="float64")
    c, r = t[CONTROL].to_numpy(dtype="float64"), t[TREATMENT].to_numpy(dtype="float64")
    out = {}
    for name in uc.BINS:
        m = b == name
        n = int(m.sum())
        out[name] = {"n": n, "mean_y": float(y[m].mean()) if n else None,
                     "mean_hprox_med": float(np.nanmean(h[m])) if n and name != uc.NAN_BIN else None,
                     f"mean_{CONTROL}": float(c[m].mean()) if n else None, f"mean_{TREATMENT}": float(r[m].mean()) if n else None,
                     "sum_g": float(g[m].sum())}
    return out


def check_counts_match_e3c(counts: dict, stored_bins) -> dict:
    """The 2026 bin counts this run prices on must equal E3C's stored 2026 witness rows' (the same file, the same
    function); a difference means the serve file or the template changed under the price. Raises AssertionError."""
    stored = uc.bin_counts(stored_bins)
    if stored != counts:
        raise AssertionError(f"2026 bin counts {counts} differ from E3C's stored {stored}; the price would not be E3C's 2026")
    return stored


# =============================================================================================
# records, paths, steps
# =============================================================================================

def output_paths(smoke: bool) -> dict:
    """Real: the parquet under data/cache_stand/, the json and logs under reports/. Smoke: everything under
    data/smoke_unm_congestion_resid/, never the real paths."""
    if smoke:
        return {k: SMOKE_DIR / v for k, v in dict(lomo=LOMO_NAME, json=JSON_NAME, lomo_log=LOMO_LOG, score_log=SCORE_LOG).items()}
    return {"lomo": CACHE / LOMO_NAME, "json": REPORTS / JSON_NAME, "lomo_log": REPORTS / LOMO_LOG, "score_log": REPORTS / SCORE_LOG}


def write_lomo(preds: pd.DataFrame, path: pathlib.Path, extra: dict) -> None:
    """The per-row parquet with its convention stamped in the file metadata (BC-2)."""
    table = pa.Table.from_pandas(preds[PRED_COLUMNS], preserve_index=False)
    meta = {b"convention": CONVENTION.encode(), b"prereg": PREREG.encode(), b"tag": TAG.encode(),
            META_KEY: json.dumps(extra, default=_jsonable).encode()}
    pq.write_table(table.replace_schema_metadata({**(table.schema.metadata or {}), **meta}), path)


def read_lomo(path: pathlib.Path) -> tuple:
    """The stored predictions and their metadata; refuses a file without the taxi-time stamp."""
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    conv = meta.get(b"convention", b"").decode()
    if conv != CONVENTION:
        raise ValueError(f"{path.name} carries convention {conv!r}, expected {CONVENTION!r} (reports/bug_classes.md BC-2)")
    return table.to_pandas(), json.loads(meta.get(META_KEY, b"{}").decode())


def run_lomo(args, paths, log) -> int:
    started = dt.datetime.now(dt.timezone.utc)
    if args.smoke:
        unm, lirf, wsum = uc.synthetic_world(seed=0)
        log(f"synthetic world: {len(unm):,} unmatched rows, {len(lirf):,} LIRF matched rows; witness {wsum}")
    else:
        files = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
        if len(files) != 12:
            raise SystemExit(f"expected the 12 training months under {RAW}, found {len(files)}")
        if not REFERENCE.exists():
            raise SystemExit(f"the reproduction reference {REFERENCE} is missing; the prereg's guard cannot run")
        unm, lirf, wsum = uc.load_real(files, log)
    fill = bs.schedule_fill(unm)
    nonfill_w = (~fill) & unm.hprox_med.notna().to_numpy()
    raw = (unm.y - unm.hprox_med).to_numpy(dtype="float64")[nonfill_w]
    log(f"stratum: {len(unm):,} rows, fill share {100 * fill.mean():.1f}%, non-fill {int((~fill).sum()):,}, non-fill WITH a witness "
        f"{int(nonfill_w.sum()):,}; offset y - hprox_med on those: median {np.median(raw):+.0f} s, beyond +{CLIP_S:.0f} on "
        f"{int((raw > CLIP_S).sum()):,}, below -{CLIP_S:.0f} on {int((raw < -CLIP_S).sum()):,}; LIRF matched {len(lirf):,}")
    log("--- 12-fold LOMO: S1C (unm_congestion.congestion_arms) and S1R ---")
    t = time.time()
    preds = score_lomo(unm, lirf, SEEDS, log)
    log(f"LOMO scored {len(preds):,} rows in {time.time() - t:.0f}s; peak RSS {peak_rss_gb():.2f} GB")
    if args.smoke:
        repro = {"status": "skipped (synthetic)", "tolerance_s": REPRO_TOL_S, "columns": list(GUARD_COLUMNS)}
    else:
        repro = check_reproduction(preds, load_reference())
    log(f"reproduction guard: {repro}")
    extra = {"prereg": PREREG, "parent_prereg": PARENT_PREREG, "tag": TAG, "smoke": bool(args.smoke), **_git(),
             "started_utc": started.isoformat(), "seeds": list(SEEDS), "witness": {"columns": uc.WITNESS, "min_n": uc.MIN_N, **wsum},
             "reproduction": repro,
             "regressor": {"params": dict(bs.NF_PARAMS), "numeric": ResidualRegressor.NUMERIC, "encoded": list(bs.NF_ENCODED),
                           "target": "clip(y - hprox_med, -CLIP_S, CLIP_S) on non-fill rows with a witness; encodings on that target",
                           "clip_s": CLIP_S, "fallback": "S1C's nf_fit (per seed) where hprox_med is NaN", "T_tail_s": bs.T_TAIL_S},
             "n_rows": int(len(preds)), "n_non_lirf": int((preds.ADEP_mvt != ROME).sum()),
             "n_nan_witness_rows": int(preds.hprox_med.isna().sum()),
             "wall_s": round(time.time() - _T0, 1), "peak_rss_gb": round(peak_rss_gb(), 2)}
    write_lomo(preds, paths["lomo"], extra)
    non = preds[preds.ADEP_mvt != ROME]
    r = rmse_arms(non)
    log(f"predictions -> {paths['lomo']} ({len(preds):,} rows, convention {CONVENTION}); non-LIRF RMSE S1C {r['S1C']:.2f} S1R {r['S1R']:.2f} "
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
        w26, n_scored = uc.smoke_witness_2026(seed=1), int(len(preds))
        counts, counts_check = uc.bin_counts(w26["bin"]), "skipped (synthetic)"
    else:
        w26, _ = uc.load_witness_2026(log)
        n_scored = N_SCORED_2026
        counts = uc.bin_counts(w26["bin"])
        if not E3C_WITNESS_2026.exists():
            raise AssertionError(f"E3C's stored 2026 witness {E3C_WITNESS_2026} is missing; the 2026 counts cannot be cross-checked")
        check_counts_match_e3c(counts, pd.read_parquet(E3C_WITNESS_2026)["bin"].to_numpy())
        counts_check = f"equal to {E3C_WITNESS_2026.relative_to(ROOT)}"
    log(f"2026 witness: {len(w26):,} scored non-LIRF unmatched rows, bins {counts} (cross-check: {counts_check})")
    non = preds[preds.ADEP_mvt != ROME].reset_index(drop=True)
    lirf = preds[preds.ADEP_mvt == ROME].reset_index(drop=True)
    c = clauses(non, counts, n_boot=n_boot, seed=BOOT_SEED, n_scored=n_scored)
    g = gains(non)
    mon = sf.monster_mask(non.y.to_numpy())
    fold_a = non.month.isin(list(FOLD_A)).to_numpy()
    nan_w = non.hprox_med.isna().to_numpy()
    lirf_g = gains(lirf) if len(lirf) else np.zeros(0)
    record = {"mode": "unm_congestion_resid_score", "prereg": PREREG, "parent_prereg": PARENT_PREREG, "tag": TAG,
              "smoke": bool(args.smoke), **_git(), "started_utc": started.isoformat(),
              "commands": {"lomo": COMMAND_LOMO, "score": COMMAND_SCORE}, "lomo_metadata": lomo_meta,
              "config": {"control": CONTROL, "treatment": TREATMENT, "seeds": list(SEEDS), "n_boot": int(n_boot), "boot_seed": BOOT_SEED,
                         "blocks": "calendar dates (UTC) of MVT_TIME", "bins": uc.BINS, "bin_edges": [float(e) for e in uc.BIN_EDGES],
                         "thin_bin": uc.THIN_BIN, "n_scored_2026": N_SCORED_2026, "c2_min_mse": C2_MIN_MSE,
                         "c2_rule": "drop the ONE airport-month with the largest |sum g|, re-bin, re-price; the full price is reported only",
                         "min_months": uc.MIN_MONTHS, "min_airports": uc.MIN_AIRPORTS, "calm_s": CALM_S, "calm_tolerance": uc.CALM_TOLERANCE,
                         "monster_s": MONSTER_S, "witness_min_n": uc.MIN_N, "clip_s": CLIP_S},
              "data": {"n_lomo_rows": int(len(preds)), "n_non_lirf": int(len(non)), "n_lirf": int(len(lirf)),
                       "n_non_lirf_nan_witness": int(nan_w.sum()), "s1r_equals_s1c_on_nan_witness_rows": bool((g[nan_w] == 0.0).all()),
                       "n_2026_scored_non_lirf_unmatched": int(len(w26)), "bins_2026": counts, "bins_2026_check": counts_check,
                       "bins_2025_non_lirf": uc.bin_counts(uc.bin_of(non.hprox_med))},
              "clauses": c, "verdict": c["verdict"],
              "descriptive": {"non_lirf": {"pooled": rmse_arms(non), "exmonster": rmse_arms(non, ~mon)},
                              "by_witness_bin_2025": bin_means(non),
                              "fold_a_months": {"months": list(FOLD_A), "sum_g": float(g[fold_a].sum()), "n": int(fold_a.sum()),
                                                "rmse": rmse_arms(non, fold_a),
                                                "per_month": {str(m): float(g[non.month.to_numpy() == m].sum()) for m in FOLD_A}},
                              "other_months": {"sum_g": float(g[~fold_a].sum()), "n": int((~fold_a).sum()), "rmse": rmse_arms(non, ~fold_a)},
                              "lirf_report_only": {"n": int(len(lirf)), "sum_g": float(lirf_g.sum()),
                                                   "rmse": rmse_arms(lirf) if len(lirf) else None}},
              "judgement_calls": [
                  "the S1R body is unm_congestion.CongestionRegressor fitted on a frame whose y is clip(y - hprox_med, -3000, 3000); "
                  "the parent's own winsorisation at WINSOR_S is then the identity (CLIP_S == WINSOR_S asserted at import)",
                  "training rows without a witness are not in the S1R fit; test rows without a witness take S1C's per-seed nf_fit, "
                  "so S1R == S1C on them bit-exact",
                  "the reproduction guard holds S1C (registered), S1 and the per-seed S1C arms to 1e-6 s against E3C's stored parquet",
                  "the excluded airport-month is the largest |sum g| over (ADEP_mvt, month); a tie is broken by airport then month",
                  "the event-excluded table is re-binned before the thin rule and the borrowing apply (E3C's rule on the rows that remain)",
                  "the full price is reported without a pass flag; only the event-excluded price is decisional",
                  "the 2026 bin counts are recomputed from data/raw/ranking.parquet and must equal E3C's stored 2026 witness rows",
                  "C1 and C3-C7 are unm_congestion.clauses' arithmetic on a renamed view (S1 := S1C, S1C := S1R); the view's gains are "
                  "checked equal to E3R's gains on every call"]}
    # ---- readable tables ----
    log(f"=== E3R on the non-LIRF LOMO rows: n={len(non):,}, sum g {c['pooled_sum_g']:+,.0f} s^2 (g = SE(S1C) - SE(S1R)) ===")
    r = record["descriptive"]["non_lirf"]
    log(f"  RMSE pooled S1C {r['pooled']['S1C']:.2f} S1R {r['pooled']['S1R']:.2f} gain {r['pooled']['gain_s']:+.2f} s; "
        f"ex-monster S1C {r['exmonster']['S1C']:.2f} S1R {r['exmonster']['S1R']:.2f} gain {r['exmonster']['gain_s']:+.2f} s")
    log(f"  NaN-witness non-LIRF rows {int(nan_w.sum()):,}: S1R == S1C on all of them: {record['data']['s1r_equals_s1c_on_nan_witness_rows']}")
    c1 = c["C1_date_block_bootstrap"]
    log(f"  C1 date-block bootstrap ({c1['n_blocks']} dates, {c1['n_draws']} draws, seed {c1['seed']}): sum g {c1['sum_g']:+,.0f} "
        f"95% [{c1['ci95'][0]:+,.0f}, {c1['ci95'][1]:+,.0f}]  passes {c1['passes']}")
    c2, full = c["C2_priced_2026"], c["C2_full_price_not_decisional"]
    ev = c2["excluded_airport_month"]
    log(f"  C2 EVENT-EXCLUDED 2026 price {c2['priced_mse']:+.1f} board MSE (>= {C2_MIN_MSE:.0f}: {c2['passes']}); dropped {ev['ADEP_mvt']} "
        f"month {ev['month']} (sum g {ev['sum_g']:+,.0f} on {ev['n']:,} rows); kept {ev['n_rows_kept']:,} rows, sum g {ev['sum_g_kept']:+,.0f}; "
        f"denominator {c2['n_scored_2026']:,}")
    log(f"  C2 full price (reported, not decisional) {full['priced_mse']:+.1f} board MSE")
    for title, tab in (("event-excluded", c2), ("full", full)):
        log(f"  [{title}] {'bin':10s} {'n2025':>7s} {'per-row g':>12s} {'used':>12s} {'from':>10s} {'n2026':>7s} {'priced':>9s}")
        for name in uc.BINS:
            b = tab["bins"][name]
            prg = "n/a" if b["per_row_gain"] is None else f"{b['per_row_gain']:+.1f}"
            log(f"  [{title}] {name:10s} {b['n_2025']:7,} {prg:>12s} {b['per_row_gain_used']:+12.1f} {str(b['borrowed_from']):>10s} "
                f"{b['n_2026']:7,} {b['priced_mse']:+9.2f}")
    log("  airport-months by |sum g|: " + "  ".join(f"{r_['ADEP_mvt']}-{r_['month']:02d}:{r_['sum_g']:+,.0f}({r_['n']})" for r_ in c["airport_month_top"]))
    c3, c4 = c["C3_months"], c["C4_airports"]
    log(f"  C3 months with sum g > 0: {c3['months_positive']} of {c3['n_months']} (>= {uc.MIN_MONTHS}: {c3['passes']})  "
        + "  ".join(f"{m}:{v['sum_g']:+,.0f}" for m, v in c3["per_month"].items()))
    log(f"  C4 airports with sum g > 0: {c4['airports_positive']} of {c4['n_airports']} (>= {uc.MIN_AIRPORTS}: {c4['passes']})  "
        + "  ".join(f"{a}:{v['sum_g']:+,.0f}" for a, v in c4["per_airport"].items()))
    c5, c6, c7 = c["C5_calm_control"], c["C6_not_the_monsters"], c["C7_seeds"]
    log(f"  C5 calm (hprox_med < {CALM_S:.0f}, n {c5['n_calm']:,}) sum g {c5['calm_sum_g']:+,.0f} >= {c5['bound']:+,.0f}: {c5['passes']}")
    log(f"  C6 ex-monster (y <= {MONSTER_S:.0f}, n {c6['n_exmonster']:,}) sum g {c6['exmonster_sum_g']:+,.0f} > 0: {c6['passes']}")
    log(f"  C7 pooled {c7['pooled_sum_g']:+,.0f} vs 2 x seed sd {2 * c7['seed_sd']:,.0f} (per seed {c7['per_seed_sum_g']}): {c7['passes']}")
    log(f"  {'bin':10s} {'n':>7s} {'mean y':>9s} {'witness':>9s} {'S1C':>9s} {'S1R':>9s} {'sum g':>14s}")
    for name, b in record["descriptive"]["by_witness_bin_2025"].items():
        if b["n"]:
            w = "n/a" if b["mean_hprox_med"] is None else f"{b['mean_hprox_med']:9.0f}"
            log(f"  {name:10s} {b['n']:7,} {b['mean_y']:9.0f} {w:>9s} {b['mean_S1C']:9.0f} {b['mean_S1R']:9.0f} {b['sum_g']:+14,.0f}")
    fa = record["descriptive"]["fold_a_months"]
    log(f"  fold-A months {FOLD_A} (seen by E3C's exploratory look): sum g {fa['sum_g']:+,.0f} on {fa['n']:,} rows {fa['per_month']}; "
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
        log(f"unm_congestion_resid {args.step}  smoke={args.smoke}  prereg={PREREG}  paths={ {k: str(v) for k, v in paths.items()} }")
        rc = run_lomo(args, paths, log) if args.step == "lomo" else run_score(args, paths, log)
    finally:
        if args.smoke:
            log(SMOKE_BANNER)
        log.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
