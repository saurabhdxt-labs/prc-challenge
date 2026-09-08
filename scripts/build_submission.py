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


def fit_unmatched(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Airport x schedule-offset mixture: p * (MVT - SCHED) + (1 - p) * conditional mean.

    `p` is the rate at which the airport stamped the SCHEDULED push into the off-block
    field. It is strongly airport-specific -- LIRF 48.5% against 0.2-9.4% elsewhere -- and
    pooling across airports hides it entirely. A gradient booster on these rows scored
    worse (1793.6 vs 1648.4), so the simple cell estimator is deliberate, not a shortcut.
    """
    fill = (train.BLOCK_TIME_UTC_mvt - train.SCHED_TIME_UTC_mvt).dt.total_seconds().abs() <= 60
    train = train.assign(fill=fill)
    clean = train[~train.fill]
    prior_mean, prior_rate = float(clean.y.mean()), float(train.fill.mean())

    means = clean.groupby(["ADEP_mvt", "sp_band"], observed=True).y.mean()
    rates = train.groupby(["ADEP_mvt", "sp_band"], observed=True).fill.mean()
    idx = pd.MultiIndex.from_frame(test[["ADEP_mvt", "sp_band"]])
    baseline = np.nan_to_num(means.reindex(idx).to_numpy(dtype="float64"), nan=prior_mean)
    p = np.nan_to_num(rates.reindex(idx).to_numpy(dtype="float64"), nan=prior_rate)
    return np.maximum(p * test.sp.to_numpy() + (1 - p) * baseline, 1.0)


def predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(test), dtype="float64")
    matched = ~test.unmatched.to_numpy()
    out[matched] = fit_matched(train[~train.unmatched], test[matched])
    out[~matched] = fit_unmatched(train[train.unmatched], test[~matched])
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
