#!/usr/bin/env python3.11
"""CSF — does CALLSIGN FREQUENCY sharpen the LIRF unmatched schedule-fill classifier? (c672efca)

This arm's entire signal is a CALLSIGN-IDENTITY COUNT, so its characteristic failure is a counting bug
that invents the result. Three guards, each with its own test:
  * a **null `FLIGHT_mvt` matches nothing** (BC-15: pandas groups NaN with NaN);
  * a row is **never its own twin**;
  * `n_cs` **never leaks across calendar months**.

The other load-bearing guard is `assert_replicates`: adding features means rebuilding the shipped
`_lirf_fill_model`'s design matrix, so the baseline is first checked against production's own
predicted probabilities and the run ABORTS on any disagreement. DAC's lesson — an unvalidated
replication of the incumbent makes every downstream delta meaningless.

2025 screen only. No fit beyond 24 logistic regressions, no production change, no upload, no commit,
no board read, and NO board-MSE figure is derived.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "csf_report.json"
PREREG = ROOT / "plans" / "PREREG_csf_callsign_frequency_fill_2026_09_19.md"

# ---- registered constants ----
AUC_BAR, FOLDS_BAR, N_FOLDS = 0.020, 10, 12
TWIN_H, NO_TWIN_SENTINEL = 36, 72.0
EXPECT_ROWS, REPLICATION_ATOL = 1488, 1e-10
MIN_ROWS_CARRYING = 20
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


# ------------------------------------------------------------------ features

def _epoch_s(t: pd.Series) -> np.ndarray:
    """Seconds since epoch as float64. A tz-aware column's `.to_numpy()` is an object array."""
    return (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype="float64")

def callsign_features(rows: pd.DataFrame, dep_pool: pd.DataFrame, mvt_pool: pd.DataFrame) -> pd.DataFrame:
    """`n_cs`, `has_twin36`, `twin_h` for `rows`, aligned to `rows.index`.

    Label-free by construction: reads only `FLIGHT_mvt`, `MVT_TIME_UTC_mvt`, `MVT_ID_mvt` and the
    pool's `PHASE_mvt`/`ADEP_mvt`. A null callsign is an IDENTITY THAT MATCHES NOTHING - it gets
    n_cs 0, no twin, and the sentinel - never grouped with other nulls.
    """
    idx = rows.index
    cs = rows.FLIGHT_mvt
    known = cs.notna().to_numpy()

    # ---- n_cs: DEP rows at the ten airports, same callsign, same calendar month ----
    dp = dep_pool[(dep_pool.PHASE_mvt == "DEP") & dep_pool.ADEP_mvt.isin(APTS) & dep_pool.FLIGHT_mvt.notna()]
    dkey = dp.FLIGHT_mvt.astype(str) + "|" + dp.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    counts = dkey.value_counts()
    rkey = cs.astype(str) + "|" + rows.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m")
    n_cs = np.where(known, rkey.map(counts).fillna(0).to_numpy(), 0).astype(np.int64)

    # ---- twins: any movement of the same callsign, excluding the row itself ----
    mp = mvt_pool[mvt_pool.FLIGHT_mvt.notna()]
    # epoch SECONDS, not numpy datetimes: a tz-aware pandas column comes back as an OBJECT array of
    # Timestamps, which silently breaks elementwise arithmetic (caught by the twin tests at write time)
    by = {k: (_epoch_s(g.MVT_TIME_UTC_mvt), g.MVT_ID_mvt.to_numpy())
          for k, g in mp.groupby(mp.FLIGHT_mvt.astype(str), sort=False)}
    gap = np.full(len(rows), NO_TWIN_SENTINEL, dtype="float64")
    t_row = _epoch_s(rows.MVT_TIME_UTC_mvt)
    id_row = rows.MVT_ID_mvt.to_numpy()
    for i, (k, ok) in enumerate(zip(cs.astype(str).to_numpy(), known)):
        if not ok or k not in by:
            continue
        ts, ids = by[k]
        other = ids != id_row[i]
        if not other.any():
            continue
        h = np.abs(ts[other] - t_row[i]) / 3600.0
        gap[i] = min(float(h.min()), NO_TWIN_SENTINEL)
    return pd.DataFrame({"n_cs": n_cs,
                         "has_twin36": (gap <= TWIN_H).astype(np.int64),
                         "twin_h": gap}, index=idx)


# ------------------------------------------------------------------ guards + decision

def assert_replicates(shipped: np.ndarray, mine: np.ndarray) -> None:
    """Abort unless my rebuilt baseline reproduces production's probabilities."""
    if shipped.shape != mine.shape:
        raise ValueError(f"baseline does not replicate: shapes {shipped.shape} vs {mine.shape}")
    d = float(np.max(np.abs(shipped - mine))) if len(shipped) else 0.0
    if d > REPLICATION_ATOL:
        raise ValueError(f"baseline does not replicate the shipped model: max|delta| = {d:.3e} "
                         f"> atol {REPLICATION_ATOL:.0e}")


def decide(auc_gain: float, folds_improved: int, trim_same_sign: bool, rows_carrying: int) -> str:
    """All three registered clauses, plus the pre-committed tail rule. Ambiguous = NOT WORKING."""
    if rows_carrying < MIN_ROWS_CARRYING:
        return "NOT WORKING"
    if auc_gain >= AUC_BAR and folds_improved >= FOLDS_BAR and trim_same_sign:
        return "WORKING"
    return "NOT WORKING"


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC; ties averaged. 0.5 when one class is absent."""
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return 0.5
    r = pd.Series(p).rank().to_numpy()
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _rows_carrying(gain_per_row: np.ndarray, frac: float = 0.5) -> int:
    """How many rows (largest first) account for `frac` of the total positive gain."""
    g = np.sort(gain_per_row[gain_per_row > 0])[::-1]
    if g.size == 0:
        return 0
    return int(np.searchsorted(np.cumsum(g), frac * g.sum()) + 1)


# ------------------------------------------------------------------ the run

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPORT))
    a_ = ap.parse_args(argv)
    out = pathlib.Path(a_.out)
    if out.exists():
        raise SystemExit(f"{out} exists - a rerun must not silently become a second look at the same bar")
    t0 = time.time()

    spec = importlib.util.spec_from_file_location("bs", ROOT / "scripts" / "build_submission.py")
    BS = importlib.util.module_from_spec(spec); sys.modules["bs"] = BS; spec.loader.exec_module(BS)
    from sklearn.linear_model import LogisticRegression

    files = sorted((ROOT / "data" / "raw").glob("training_*.parquet"))
    raw = BS.load_movements(files)
    df = BS.derive(BS.admissible(raw))
    df["fill"] = (df.BLOCK_TIME_UTC_mvt - df.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60
    df["y"] = df.TAXITIME_SEC_mvt.astype(float)
    unm = df[df.AOBT_3_flt.isna()]
    lirf = unm[unm.ADEP_mvt == "LIRF"].copy()
    if len(lirf) != EXPECT_ROWS:
        raise ValueError(f"LIRF unmatched rows = {len(lirf):,}, registration says {EXPECT_ROWS:,}")
    matched = df[df.AOBT_3_flt.notna()]

    feats = callsign_features(lirf, raw, raw)
    lirf = pd.concat([lirf, feats], axis=1)
    R = {"prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
         "bars": {"auc": AUC_BAR, "folds": FOLDS_BAR, "n_folds": N_FOLDS,
                  "min_rows_carrying": MIN_ROWS_CARRYING},
         "n_rows": int(len(lirf)), "fill_rate": round(float(lirf.fill.mean()), 6),
         "feature_summary": {"n_cs_median": float(lirf.n_cs.median()),
                             "has_twin36_rate": round(float(lirf.has_twin36.mean()), 6),
                             "twin_h_median": float(lirf.twin_h.median()),
                             "null_callsign_rows": int(lirf.FLIGHT_mvt.isna().sum())}}

    def design(fit_u, fit_m, frame, extra: bool):
        def lomap(f, col, tgt, k):
            g = f.groupby(col, observed=True)[tgt].agg(["mean", "size"]); pr = float(f[tgt].mean())
            return (g["mean"] * g["size"] + pr * k) / (g["size"] + k), pr
        prior = float(fit_u.fill.mean())
        m_air, _ = lomap(fit_u, "airline", "fill", 5)
        m_std, _ = lomap(fit_u, "stand_c1", "fill", 5)
        m_des, _ = lomap(fit_u, "ades_p2", "fill", 5)
        ml = fit_m[fit_m.ADEP_mvt == "LIRF"]
        copy = ((ml.BLOCK_TIME_UTC_mvt - ml.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60) & \
               ((ml.BLOCK_TIME_UTC_mvt - ml.AOBT_3_flt).dt.total_seconds().abs() > 300)
        m_prop, prop_prior = lomap(ml.assign(copy=copy), "airline", "copy", 20)
        lg = lambda v: np.log(np.clip(v, 1e-3, 1 - 1e-3) / (1 - np.clip(v, 1e-3, 1 - 1e-3)))
        cols = [lg(frame.airline.map(m_air).fillna(prior).to_numpy(dtype="float64")),
                lg(frame.stand_c1.map(m_std).fillna(prior).to_numpy(dtype="float64")),
                lg(frame.ades_p2.map(m_des).fillna(prior).to_numpy(dtype="float64")),
                lg(frame.airline.map(m_prop).fillna(prop_prior).to_numpy(dtype="float64")),
                frame.dayoff.to_numpy(dtype="float64"),
                np.log1p(np.clip(frame.sp.to_numpy(dtype="float64"), 0, None))]
        if extra:
            cols += [np.log1p(frame.n_cs.to_numpy(dtype="float64")),
                     frame.has_twin36.to_numpy(dtype="float64"),
                     frame.twin_h.to_numpy(dtype="float64")]
        return np.nan_to_num(np.column_stack(cols))

    # ---- replication guard: my rebuilt baseline vs the SHIPPED function, same rows ----
    got = BS._lirf_fill_model(unm, matched, lirf)
    if got is None:
        raise ValueError("shipped _lirf_fill_model returned None on the full-sample fit")
    mask, p_ship = got
    clf = LogisticRegression(C=BS.LIRF_C, max_iter=2000).fit(design(lirf, matched, lirf, False),
                                                             lirf.fill.to_numpy())
    assert_replicates(p_ship, clf.predict_proba(design(lirf, matched, lirf, False))[:, 1])
    R["replication"] = {"n": int(mask.sum()), "atol": REPLICATION_ATOL, "passed": True}

    # ---- leave-one-month-out ----
    months = sorted(lirf.MVT_TIME_UTC_mvt.dt.month.unique())
    if len(months) != N_FOLDS:
        raise ValueError(f"{len(months)} months present, registration fixed {N_FOLDS}")
    per_fold, yy, pb, pa, sq_b, sq_a = [], [], [], [], [], []
    for m in months:
        te = lirf.MVT_TIME_UTC_mvt.dt.month == m
        tr_u, tr_m = lirf[~te], matched[matched.MVT_TIME_UTC_mvt.dt.month != m]
        if tr_u.fill.nunique() < 2 or te.sum() == 0:
            raise ValueError(f"fold {m} degenerate")
        nf_med = float(tr_u.y[~tr_u.fill].median())
        row = {"month": int(m), "n_test": int(te.sum())}
        for tag, extra in (("base", False), ("arm", True)):
            c = LogisticRegression(C=BS.LIRF_C, max_iter=2000).fit(
                design(tr_u, tr_m, tr_u, extra), tr_u.fill.to_numpy())
            p = c.predict_proba(design(tr_u, tr_m, lirf[te], extra))[:, 1]
            pred = p * lirf.sp[te].to_numpy() + (1 - p) * nf_med
            se = (lirf.y[te].to_numpy() - pred) ** 2
            row[f"auc_{tag}"] = round(auc(lirf.fill[te].to_numpy(), p), 6)
            row[f"mse_{tag}"] = float(se.mean())
            (pb if tag == "base" else pa).append(p)
            (sq_b if tag == "base" else sq_a).append(se)
        row["mse_improved"] = bool(row["mse_arm"] < row["mse_base"])
        per_fold.append(row); yy.append(lirf.fill[te].to_numpy())
    y_all = np.concatenate(yy); pb_all, pa_all = np.concatenate(pb), np.concatenate(pa)
    sb, sa = np.concatenate(sq_b), np.concatenate(sq_a)
    gain = sb - sa
    k = max(1, len(gain) // 100)
    keep = np.argsort(np.abs(gain))[::-1][k:]
    R["folds"] = per_fold
    R["auc_base"] = round(auc(y_all, pb_all), 6)
    R["auc_arm"] = round(auc(y_all, pa_all), 6)
    R["auc_gain"] = round(R["auc_arm"] - R["auc_base"], 6)
    R["folds_improved"] = int(sum(f["mse_improved"] for f in per_fold))
    R["mse_base"] = float(sb.mean()); R["mse_arm"] = float(sa.mean())
    R["mse_gain_pct"] = round(100 * (1 - sa.mean() / sb.mean()), 4)
    R["trim"] = {"pooled_gain": float(gain.mean()),
                 "trimmed_gain": float(gain[keep].mean()),
                 "same_sign": bool(np.sign(gain.mean()) == np.sign(gain[keep].mean()) and gain.mean() > 0)}
    R["rows_carrying_half_the_gain"] = _rows_carrying(gain)
    R["verdict"] = decide(R["auc_gain"], R["folds_improved"], R["trim"]["same_sign"],
                          R["rows_carrying_half_the_gain"])
    R["wall_s"] = round(time.time() - t0, 2)
    out.write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({k2: R[k2] for k2 in ("verdict", "auc_base", "auc_arm", "auc_gain",
                                           "folds_improved", "mse_base", "mse_arm", "mse_gain_pct",
                                           "trim", "rows_carrying_half_the_gain", "wall_s")}, indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
