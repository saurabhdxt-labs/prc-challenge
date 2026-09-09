"""Build the submission, or reproduce the held-out validation number.

    python scripts/build_submission.py --validate    # held-out Jan+Jul 2025, prints RMSE
    python scripts/build_submission.py               # fits on all 12 months, writes the parquet

The method is documented in README.md. In one line: for rows with a Network Manager
off-block (98.47%) we model `delta = BLOCK_TIME - AOBT_3` and recover the target as
`proxy - delta`; for the rest we use an airport x schedule-offset mixture, because those
rows have no second clock to lean on.

Memory: the movement frames are large. Only required columns are read, intermediates are
released explicitly, and feature matrices are float32. Peak RSS stays near 3 GB.
"""
from __future__ import annotations

import argparse
import gc
import glob
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor as HGR

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
HOLDOUT_MONTHS = (1, 7)          # the evaluation is January and July; hold out their 2025 twins
SEEDS = (0, 1, 2)
SMOOTH = 50.0                    # target-encoding smoothing, fitted inside the training fold only
BLEND = 0.5                      # pooled vs per-airport, flat across 0.4-0.7, not tuned

COLUMNS = ["PHASE_mvt", "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "STAND_mvt", "RUNWAY_mvt",
           "AIRCRAFT_TYPE_mvt", "FLIGHT_mvt", "FLIGHT_RULE_mvt", "MVT_TIME_UTC_mvt",
           "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt",
           "EOBT_1_flt", "LOBT_flt", "IOBT_flt", "AIRCRAFT_OPERATOR_flt",
           "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt"]

NUMERIC = ["proxy", "sp", "eobt_p", "lobt_p", "iobt_p", "nmdelay", "aobt_eobt",
           "lobt_sched", "eobt_sched", "lobt_eobt", "iobt_eobt", "hr", "tmin", "dow", "doy"]
ENCODED = ["ADEP_mvt", "STAND_mvt", "stand_pref", "airline", "ADES_mvt", "AIRCRAFT_TYPE_mvt",
           "RUNWAY_mvt", "AIRCRAFT_OPERATOR_flt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt",
           "FLIGHT_TYPE_flt"]
#: sched-offset bands for the unmatched stratum. Coarse on purpose: finer cells were tested
#: and are inside the noise (see reports/STRATUM_MONSTERS.md).
SP_EDGES = [-np.inf, 900, 1800, 3600, 7200, 10800, 21600, 43200, np.inf]
#: finer bands used only as the deepest level of the hierarchical stratum estimator; they are
#: shrunk toward the coarse parent, so thin cells cost nothing where they have no support.
SP_FINE = [-np.inf, 900, 1800, 3600, 7200, 10800, 14400, 18000, 21600, 28800, 43200, 64800, np.inf]
K_SHRINK = 5.0                   # pseudo-counts toward the parent cell (STRATUM_MONSTERS.md 3a)
LIRF_C = 1.0                     # logistic regularisation; C=0.1/0.3 measured worse, C=3/10 flat
SP_LABELS = ["<15m", "15-30m", "30-60m", "1-2h", "2-3h", "3-6h", "6-12h", "12h+"]


def _seconds(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a - b).dt.total_seconds()


def load_movements(paths) -> pd.DataFrame:
    frames = [pq.read_table(p, columns=COLUMNS).to_pandas() for p in paths]
    df = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()
    return df[df.PHASE_mvt == "DEP"]


def admissible(df: pd.DataFrame) -> pd.DataFrame:
    """The pre-registered censoring policy. Labelled rows only; see prc/labels.py."""
    keep = df.TAXITIME_SEC_mvt.notna() & df.BLOCK_TIME_UTC_mvt.notna() & df.MVT_TIME_UTC_mvt.notna()
    df = df[keep]
    df = df[df.TAXITIME_SEC_mvt > 0]
    df = df.sort_values(["BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "MVT_ID_mvt"],
                        kind="mergesort")
    return df[~df.MVT_ID_mvt.duplicated(keep="first")].reset_index(drop=True)


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Every feature here is computable from columns populated on the scored rows.

    Off-block is never touched except to build the training target: it is 100% null at
    serve time, so any feature reading it would be silently dead in production. That
    failure mode is guarded by tests/test_serve_time_contract.py.
    """
    t = df.MVT_TIME_UTC_mvt
    df = df.assign(
        proxy=_seconds(t, df.AOBT_3_flt),
        sp=_seconds(t, df.SCHED_TIME_UTC_mvt),
        eobt_p=_seconds(t, df.EOBT_1_flt),
        lobt_p=_seconds(t, df.LOBT_flt),
        iobt_p=_seconds(t, df.IOBT_flt),
        nmdelay=_seconds(df.AOBT_3_flt, df.SCHED_TIME_UTC_mvt),
        aobt_eobt=_seconds(df.AOBT_3_flt, df.EOBT_1_flt),
        lobt_sched=_seconds(df.LOBT_flt, df.SCHED_TIME_UTC_mvt),
        eobt_sched=_seconds(df.EOBT_1_flt, df.SCHED_TIME_UTC_mvt),
        lobt_eobt=_seconds(df.LOBT_flt, df.EOBT_1_flt),
        iobt_eobt=_seconds(df.IOBT_flt, df.EOBT_1_flt),
        hr=t.dt.hour, tmin=t.dt.hour * 60 + t.dt.minute,
        dow=t.dt.dayofweek, doy=t.dt.dayofyear, month=t.dt.month,
        airline=df.FLIGHT_mvt.astype(str).str[:3],
        stand_pref=df.STAND_mvt.astype(str).str[:2],
        unmatched=df.AOBT_3_flt.isna(),
    )
    df["sp_band"] = pd.cut(df.sp, SP_EDGES, labels=SP_LABELS)
    df["sp_fine"] = pd.cut(df.sp, SP_FINE, labels=[str(i) for i in range(len(SP_FINE) - 1)])
    # date(take-off) - date(scheduled push), clipped: the 24h date-slip rows are 11/12 dayoff=1
    df["dayoff"] = np.clip(
        (t.dt.normalize() - df.SCHED_TIME_UTC_mvt.dt.normalize()).dt.days, 0, 1)
    df["stand_c1"] = df.STAND_mvt.astype(str).str[:1]
    df["ades_p2"] = df.ADES_mvt.astype(str).str[:2]
    return df


class Encoder:
    """Smoothed target encodings. Fitted on training rows ONLY — never across a fold."""

    def __init__(self, train: pd.DataFrame, target: str):
        self.prior = float(train[target].mean())
        self.maps = {}
        for col in ENCODED:
            g = train.groupby(col, observed=True)[target].agg(["mean", "size"])
            self.maps[col] = (g["mean"] * g["size"] + self.prior * SMOOTH) / (g["size"] + SMOOTH)
        g = train.groupby(["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"], observed=True)[target].agg(["mean", "size"])
        self.ars = (g["mean"] * g["size"] + self.prior * SMOOTH) / (g["size"] + SMOOTH)

    def transform(self, df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col in ENCODED:
            out[prefix + col] = df[col].map(self.maps[col]).astype("float32").fillna(self.prior)
        idx = pd.MultiIndex.from_frame(df[["ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]])
        vals = self.ars.reindex(idx).to_numpy(dtype="float64")
        out[prefix + "ars"] = np.nan_to_num(vals, nan=self.prior).astype("float32")
        return out


def design(df: pd.DataFrame, encoders) -> pd.DataFrame:
    x = df[NUMERIC].astype("float32")
    return pd.concat([x] + [e.transform(df, p) for p, e in encoders], axis=1)


def fit_matched(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Model `delta`, then recover the target as `proxy - delta`.

    A tree cannot represent `y = proxy - f(x)` -- it must bin the identity in a wide-range
    variable -- but it can represent `f(x)`. Measured on an identical fold, features and
    seed: 285.24 (predicting y) -> 252.66 (predicting delta).
    """
    encoders = [("y_", Encoder(train, "y")), ("d_", Encoder(train, "delta"))]
    x_train, x_test = design(train, encoders), design(test, encoders)
    target, proxy = train.delta.to_numpy(), test.proxy.to_numpy()

    params = dict(learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=20,
                  early_stopping=False)
    pooled = []
    for seed in SEEDS:
        model = HGR(max_iter=900, random_state=seed, **params)
        model.fit(x_train, target)
        pooled.append(model.predict(x_test))
    pooled = np.mean(pooled, axis=0)

    # Per-airport models: ten airports with different surface geometry were sharing one
    # model's capacity. Worth +3.08 [+1.82, +4.27] over pooled; the blend adds a further
    # +2.46. Airports without enough rows fall back to the pooled prediction.
    per_airport = pooled.copy()
    for airport, part in train.groupby("ADEP_mvt", observed=True):
        mask = (test.ADEP_mvt == airport).to_numpy()
        if len(part) < 20_000 or not mask.any():
            continue
        model = HGR(max_iter=400, random_state=0, **params)
        model.fit(x_train.loc[part.index], part.delta.to_numpy())
        per_airport[mask] = model.predict(x_test[mask])

    delta_hat = (1 - BLEND) * pooled + BLEND * per_airport
    return np.maximum(proxy - delta_hat, 1.0)


def _key(df: pd.DataFrame, cols) -> np.ndarray:
    """Composite cell key. NEVER use "\x00" as the separator: pandas' Arrow string path
    treats NUL as a terminator and silently collapses distinct keys (2,277 -> 10 once)."""
    out = df[cols[0]].astype(str)
    for c in cols[1:]:
        out = out + "|" + df[c].astype(str)
    return out.to_numpy()


def _lirf_fill_model(tr_unm: pd.DataFrame, tr_matched: pd.DataFrame, te: pd.DataFrame):
    """LIRF-only logistic for P(the airport stamped the SCHEDULE into off-block).

    STRATUM_MONSTERS.md 3b (variant L-e): established at -81 s on the stratum with the
    interval excluding zero under both a row and a month-block bootstrap, and a permutation
    control that returns to baseline. The load-bearing feature is airline identity -- some
    handling agents copy the schedule and some do not -- measured twice: on the unmatched
    rows themselves and, far more precisely, on LIRF's 159k MATCHED rows where the copy is
    visible as BLOCK==SCHED while NM disagrees.
    """
    from sklearn.linear_model import LogisticRegression

    u = tr_unm[tr_unm.ADEP_mvt == "LIRF"]
    if len(u) < 200 or u.fill.nunique() < 2:
        return None
    prior = float(u.fill.mean())

    def logodds_map(frame, col, target, k):
        g = frame.groupby(col, observed=True)[target].agg(["mean", "size"])
        pr = float(frame[target].mean())
        return (g["mean"] * g["size"] + pr * k) / (g["size"] + k), pr

    m_air, _ = logodds_map(u, "airline", "fill", 5)
    m_std, _ = logodds_map(u, "stand_c1", "fill", 5)
    m_des, _ = logodds_map(u, "ades_p2", "fill", 5)
    ml = tr_matched[tr_matched.ADEP_mvt == "LIRF"]
    copy = ((ml.BLOCK_TIME_UTC_mvt - ml.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60) & \
           ((ml.BLOCK_TIME_UTC_mvt - ml.AOBT_3_flt).dt.total_seconds().abs() > 300)
    ml = ml.assign(copy=copy)
    m_prop, prop_prior = logodds_map(ml, "airline", "copy", 20)

    lg = lambda v: np.log(np.clip(v, 1e-3, 1 - 1e-3) / (1 - np.clip(v, 1e-3, 1 - 1e-3)))

    def X(f):
        return np.nan_to_num(np.column_stack([
            lg(f.airline.map(m_air).fillna(prior).to_numpy(dtype="float64")),
            lg(f.stand_c1.map(m_std).fillna(prior).to_numpy(dtype="float64")),
            lg(f.ades_p2.map(m_des).fillna(prior).to_numpy(dtype="float64")),
            lg(f.airline.map(m_prop).fillna(prop_prior).to_numpy(dtype="float64")),
            f.dayoff.to_numpy(dtype="float64"),
            np.log1p(np.clip(f.sp.to_numpy(dtype="float64"), 0, None)),
        ]))

    clf = LogisticRegression(C=LIRF_C, max_iter=2000).fit(X(u), u.fill.to_numpy())
    mask = (te.ADEP_mvt == "LIRF").to_numpy()
    if not mask.any():
        return None
    return mask, clf.predict_proba(X(te[mask]))[:, 1]


def fit_unmatched(train: pd.DataFrame, test: pd.DataFrame,
                  train_matched: pd.DataFrame | None = None) -> np.ndarray:
    """Schedule-fill mixture: p * (MVT - SCHED) + (1 - p) * conditional mean.

    `p` is the rate at which the airport stamped the SCHEDULED push into the off-block
    field. It is strongly airport-specific -- LIRF 48.5% against 0.2-9.4% elsewhere -- and
    pooling across airports hides it entirely. A gradient booster in place of the cell
    estimator scored worse (1793.6 vs 1648.4) because of the heavy tail, so the cell
    estimator is deliberate; what changed is that the cells are now hierarchical and LIRF
    gets a proper classifier instead of a cell rate.

    Two changes over the flat estimator, both established in reports/STRATUM_MONSTERS.md:
      * hierarchical shrinkage [ap] -> [ap, coarse] -> [ap, fine, dayoff] with K pseudo-counts,
        so a thin cell falls back to its parent instead of to the global prior;
      * at LIRF only, `p` comes from the L-e logistic rather than the cell rate. The cell
        rate is the miscalibrated one there (reliability 0.49/0.71, 0.67/0.39 by quintile).
    """
    fill = (train.BLOCK_TIME_UTC_mvt - train.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60
    tr = train.assign(fill=fill)
    p = np.full(len(test), float(tr.fill.mean()))
    nf = np.full(len(test), float(tr.y[~tr.fill].mean()))

    for cols in (["ADEP_mvt"], ["ADEP_mvt", "sp_band"], ["ADEP_mvt", "sp_fine", "dayoff"]):
        k_tr, k_te = _key(tr, cols), _key(test, cols)
        grp = pd.DataFrame({"k": k_tr, "fill": tr.fill.to_numpy(), "y": tr.y.to_numpy()})
        agg = grp.groupby("k").fill.agg(["mean", "size"])
        clean = grp[~grp.fill].groupby("k").y.mean()
        ser = pd.Series(k_te)
        n = ser.map(agg["size"]).fillna(0).to_numpy(dtype="float64")
        pc = ser.map(agg["mean"]).to_numpy(dtype="float64")
        mc = ser.map(clean).to_numpy(dtype="float64")
        p = (np.nan_to_num(pc) * n + p * K_SHRINK) / (n + K_SHRINK)
        nf = (np.where(np.isnan(mc), nf, mc) * n + nf * K_SHRINK) / (n + K_SHRINK)

    if train_matched is not None and len(train_matched):
        got = _lirf_fill_model(tr, train_matched, test)
        if got is not None:
            mask, p_lirf = got
            p = p.copy()
            p[mask] = p_lirf

    return np.maximum(p * test.sp.to_numpy() + (1 - p) * nf, 1.0)


def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(test), dtype="float64")
    matched = ~test.unmatched.to_numpy()
    out[matched] = fit_matched(train[~train.unmatched], test[matched])
    out[~matched] = fit_unmatched(train[train.unmatched], test[~matched],
                                  train_matched=train[~train.unmatched])
    return out


def check_submission(frame: pd.DataFrame, template: pd.DataFrame) -> None:
    """Run before every upload. A malformed file wastes a submission slot."""
    if len(frame) != len(template):
        raise ValueError(f"row count {len(frame)} != template {len(template)}")
    if set(frame.MVT_ID_mvt) != set(template.MVT_ID_mvt):
        raise ValueError("identifier set does not match the template")
    v = frame.TAXITIME_SEC_mvt
    if v.isna().any():
        raise ValueError(f"{int(v.isna().sum())} null predictions")
    if not np.isfinite(v).all():
        raise ValueError("non-finite predictions")
    if (v <= 0).any():
        raise ValueError(f"{int((v <= 0).sum())} non-positive predictions")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true",
                    help="hold out Jan+Jul 2025 and print RMSE instead of writing a submission")
    ap.add_argument("--out", default="submissions/merry-quicksand.parquet")
    args = ap.parse_args()

    training_files = sorted(glob.glob(str(RAW / "training_*.parquet")))
    if not training_files:
        print(f"no training data under {RAW} -- run scripts/fetch_data.py first", file=sys.stderr)
        return 1
    frame = derive(admissible(load_movements(training_files)))
    frame["y"] = frame.TAXITIME_SEC_mvt.astype("float64")
    frame["delta"] = _seconds(frame.BLOCK_TIME_UTC_mvt, frame.AOBT_3_flt)

    if args.validate:
        held = frame.month.isin(HOLDOUT_MONTHS)
        train, test = frame[~held], frame[held].reset_index(drop=True)
        pred = predict(train, test)
        truth = test.y.to_numpy()
        rng = np.random.default_rng(0)
        err = (truth - pred) ** 2
        boot = [np.sqrt(err[rng.integers(0, len(err), len(err))].mean()) for _ in range(1000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        um = test.unmatched.to_numpy()
        print(f"held-out {HOLDOUT_MONTHS}: n={len(test)}")
        print(f"  TOTAL      {np.sqrt(err.mean()):8.2f}   95% [{lo:.1f}, {hi:.1f}]")
        print(f"  matched    {np.sqrt(err[~um].mean()):8.2f}   n={int((~um).sum())}")
        print(f"  unmatched  {np.sqrt(err[um].mean()):8.2f}   n={int(um.sum())}")
        return 0

    template = pq.read_table(RAW / "submitting.parquet").to_pandas()
    ranking = derive(load_movements([RAW / "ranking.parquet"]))
    scored = ranking[ranking.MVT_ID_mvt.isin(set(template.MVT_ID_mvt))].reset_index(drop=True)
    if len(scored) != len(template):
        raise ValueError(f"joined {len(scored)} scored rows, template has {len(template)}")

    pred = predict(frame, scored)
    out = pd.DataFrame({"MVT_ID_mvt": scored.MVT_ID_mvt.to_numpy(),
                        "TAXITIME_SEC_mvt": np.rint(pred).astype("int32")})
    out = template[["MVT_ID_mvt"]].merge(out, on="MVT_ID_mvt", how="left")
    check_submission(out, template)

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    reread = pq.read_table(dest).to_pandas()      # verify what actually landed on disk
    check_submission(reread, template)
    print(f"wrote {dest}  rows={len(reread)}  "
          f"median={reread.TAXITIME_SEC_mvt.median():.0f}s  "
          f"mean={reread.TAXITIME_SEC_mvt.mean():.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
