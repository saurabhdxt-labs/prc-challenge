"""FM1 features: raw movement rows -> the matched and unmatched feature frames, the fitted encodings, the design columns.

Matched block (one row per admissible matched departure, built per calendar-month chunk exactly as the stand caches are:
`sched_day` counts departures per file, so a scored file is split on the movement month, stand_ab.build_ranking's rule):
    base      stand_ab.build_features  (BASELINE_FEATS; the three `*_sec` columns are DROPPED: NSC, FMA §1.3 #16)
    queue     stand_ab.build_queue      order  stand_ab.build_order      weather  stand_ab.build_weather
    ORL       orl_build.attach          (static OSM stand->runway lookup)
    ADS-B     the R12 tables (train.parquet + serve.parquet), joined by MVT_ID_mvt; every admitted row must have one
    encodings te_/de_ over stand_ab.ENC_KEYS, smoothed means fitted on TRAINING rows only (stand_ab.infold_encodings'
              formula; the tables are stored in the artefact and applied unchanged at predict)
Unmatched block (the provisional `hgb` body only; the v26 lane, prc/pipeline/fm1/unmatched.py, derives its own rows
through the shipped build_submission.derive): sp, hr, dayoff (data.dayoff_tz), the airport-hour matched witness
(unm_congestion.airport_hour_witness, label-free) and smoothed y-encodings of build_submission.NF_ENCODED + ars.

Every block is built through the shipped builders' public functions (imported, never copied, never edited). Labels are
read only on training rows, only into `y` / `delta`, and never into a design column (rows.assert_design_clean).
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import errors as E
from . import rows, shipped

ROOT = pathlib.Path(__file__).resolve().parents[3]
#: the ADS-B tables' key columns; every other column of train.parquet is an ADS-B feature (the table defines the block:
#: stamp 801f9bcac9a5 = the 33 features v36/v41 served, 51348d3e1cb3 = 35 after DWG; pinned by tests)
ADSB_TABLE_KEYS = ["month", "pos", "MVT_ID_mvt", "ap", "mvt_day", "adsb_day_present", "coverage"]
ADSB_EXTRA = ["coverage", "adsb_day_present"]
#: ADS-B columns describing the airport's surface traffic, not the flight: kept when only the flight is unheard
ADSB_AIRPORT_COLS = ("ground_seen_mvt_m10", "moving_mvt_m10")
#: the coverage state of a flight the receiver network did not hear on a day it heard others (a 2025 unjoined row)
ADSB_UNHEARD_STATE = {"joined": 0.0, "coverage": 1.0, "adsb_day_present": 1.0}
ORL_COLS = ["orl_route_m", "orl_route_med_m", "orl_straight_m", "orl_n_edges", "orl_rank", "orl_matched"]
UNM_NUMERIC = ["sp", "hr", "dayoff", "hprox_med", "hprox_n"]
UNM_KEYS = ["ADEP_mvt", "RUNWAY_mvt", "stand_pref", "airline", "AIRCRAFT_OPERATOR_flt", "ars"]
#: the matched frame's bookkeeping columns (never design columns)
MATCHED_META = ["MVT_ID_mvt", "ap", "month", "day", "proxy", "hr", "y", "delta"]


def stand_ab():
    return shipped.stand_ab()


def unm_congestion():
    return shipped.unm_congestion()


def orl_build():
    return shipped.orl_build()


# =================================================================================================================
# design columns
# =================================================================================================================

def base_columns(queue: bool, order: bool, weather: bool) -> list:
    """FEATS [+ QUEUE] [+ ORDER] [+ WEATHER] without the `*_sec` columns: nsc_fold.FEATS_NSC_FW when all three are on."""
    S = stand_ab()
    cols = list(S.BASELINE_FEATS) + (list(S.QUEUE_FEATS) if queue else []) + (list(S.ORDER_FEATS) if order else []) \
        + (list(S.WEATHER_FEATS) if weather else [])
    return [c for c in cols if not c.endswith("_sec")]


def table_features(table_dir) -> list:
    """The ADS-B feature columns of the tables in `table_dir` (train.parquet's columns minus the keys; serve.parquet must
    carry the same columns)."""
    table_dir = pathlib.Path(table_dir)
    names = {}
    for n in ("train.parquet", "serve.parquet"):
        p = table_dir / n
        if not p.exists():
            raise E.SchemaError(f"ADS-B table {p} does not exist")
        names[n] = pq.read_schema(p).names
    if names["train.parquet"] != names["serve.parquet"]:
        raise E.SchemaError(f"ADS-B tables in {table_dir}: train and serve columns differ")
    missing = [c for c in ADSB_TABLE_KEYS if c not in names["train.parquet"]]
    if missing:
        raise E.MissingColumnError(f"ADS-B tables in {table_dir}: missing key column(s) {missing}")
    return [c for c in names["train.parquet"] if c not in ADSB_TABLE_KEYS]


def adsb_columns(table_dir) -> list:
    return table_features(table_dir) + ADSB_EXTRA


def matched_design_columns(cfg) -> list:
    cols = base_columns(cfg["features.queue"], cfg["features.order"], cfg["features.weather"])
    if cfg["features.orl_route_length"]:
        cols += ORL_COLS
    if cfg["features.adsb"]:
        cols += adsb_columns(cfg.path("adsb.table_dir"))
    rows.assert_design_clean(cols, "matched design")
    if len(set(cols)) != len(cols):
        raise E.SchemaError("matched design: duplicate column names")
    return cols


def unmatched_design_columns() -> list:
    cols = UNM_NUMERIC + [f"te_{k}" for k in UNM_KEYS]
    rows.assert_design_clean(cols, "unmatched design")
    return cols


# =================================================================================================================
# encodings (fitted on training rows only; persisted verbatim)
# =================================================================================================================

@dataclass
class EncodingTables:
    """Smoothed target encodings: enc(key) = (sum + prior * k) / (count + k), fitted on training rows; unseen -> prior.
    stand_ab._smooth_enc's formula; float32 on output as stand_ab.infold_encodings returns."""
    smooth: float
    priors: dict            # target name -> prior
    tables: dict            # "<prefix>_<key>" -> {key value: encoding}
    targets: dict           # "<prefix>" -> target name

    @classmethod
    def fit(cls, frame: pd.DataFrame, targets: dict, keys, smooth: float) -> "EncodingTables":
        """targets: {prefix: 1-D array over frame's rows} (e.g. {"te": y, "de": delta}); every value must be finite."""
        priors, tables, names = {}, {}, {}
        for prefix, t in targets.items():
            t = np.asarray(t, dtype="float64")
            if len(t) != len(frame) or not np.isfinite(t).all():
                raise E.SchemaError(f"encoding target {prefix!r}: length {len(t)} for {len(frame)} rows or non-finite values")
            prior = float(t.mean())
            priors[prefix], names[prefix] = prior, prefix
            for k in keys:
                g = pd.DataFrame({"k": _key(frame, k), "v": t}).groupby("k", observed=True).v.agg(["sum", "count"])
                enc = (g["sum"] + prior * smooth) / (g["count"] + smooth)
                tables[f"{prefix}_{k}"] = {str(a): float(b) for a, b in enc.items()}
        return cls(smooth=float(smooth), priors=priors, tables=tables, targets=names)

    def transform(self, frame: pd.DataFrame) -> dict:
        out = {}
        for name, table in self.tables.items():
            prefix, key = name.split("_", 1)
            s = pd.Series(_key(frame, key)).map(pd.Series(table, dtype="float64"))
            out[name] = s.fillna(self.priors[prefix]).to_numpy(np.float32)
        return out

    def to_json(self) -> str:
        return json.dumps({"smooth": self.smooth, "priors": self.priors, "tables": self.tables, "targets": self.targets},
                          sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "EncodingTables":
        d = json.loads(text)
        return cls(smooth=d["smooth"], priors=d["priors"], tables=d["tables"], targets=d["targets"])


def _key(frame: pd.DataFrame, k: str) -> np.ndarray:
    if k not in frame.columns:
        raise E.MissingColumnError(f"encoding key {k!r} is not in the frame")
    return frame[k].astype(object).where(frame[k].notna(), "NA").astype(str).to_numpy()


# =================================================================================================================
# the ADS-B table and the ORL routes
# =================================================================================================================

def read_adsb_tables(table_dir, ids, allow_missing: bool = False) -> tuple:
    """(ADS-B rows for `ids` from train.parquet + serve.parquet indexed by MVT_ID_mvt, ids without a row). An id found in
    both files is refused; a missing id is refused unless allow_missing (serve time; the caller bounds the share)."""
    table_dir = pathlib.Path(table_dir)
    ids = np.unique(np.asarray(ids, dtype="float64"))
    cols = adsb_columns(table_dir)
    parts = []
    for name in ("train.parquet", "serve.parquet"):
        p = table_dir / name
        have = pq.read_table(p, columns=["MVT_ID_mvt"]).column(0).to_numpy()
        if not np.isin(ids, have).any():
            continue                       # never scan a file that holds none of the ids (the 2 M-row training table)
        t = pq.read_table(p, columns=["MVT_ID_mvt", *cols], filters=[("MVT_ID_mvt", "in", ids.tolist())])
        parts.append(t.to_pandas())
    if not parts:
        parts.append(pd.DataFrame({c: pd.Series(dtype="float64") for c in ["MVT_ID_mvt", *cols]}))
    t = pd.concat(parts, ignore_index=True)
    if not t.MVT_ID_mvt.is_unique:
        raise E.DuplicateIdError(f"ADS-B tables in {table_dir}: {int(t.MVT_ID_mvt.duplicated().sum())} id(s) in both files")
    t = t.set_index(t.MVT_ID_mvt.to_numpy(dtype="float64")).drop(columns=["MVT_ID_mvt"])
    missing = np.setdiff1d(ids, t.index.to_numpy())
    if len(missing) and not allow_missing:
        raise E.SchemaError(f"{len(missing):,} admitted row(s) have no ADS-B table row in {table_dir} (e.g. "
                            f"{missing[:3].tolist()}); the tables must cover every admitted matched departure")
    return t, missing


def read_orl_routes(path) -> pd.DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        raise E.SchemaError(f"ORL route table {p} does not exist (scripts/analysis/orl_build.py build)")
    return pd.read_parquet(p)


# =================================================================================================================
# raw -> frames
# =================================================================================================================

def month_chunks(raw: pd.DataFrame) -> list:
    """The raw rows split on the movement's calendar month (UTC), in ascending order (stand_ab.build_ranking's rule)."""
    t = raw.MVT_TIME_UTC_mvt
    if t.isna().any():
        raise E.RangeError(f"{int(t.isna().sum())} row(s) without MVT_TIME cannot be assigned to a month")
    ym = (t.dt.year * 100 + t.dt.month).to_numpy()
    return [raw[ym == m] for m in np.unique(ym)]


def matched_frame(raw_chunk: pd.DataFrame, serve: bool, cfg, weather: pd.DataFrame | None, orl_routes) -> pd.DataFrame:
    """One month chunk -> MATCHED_META + every feature column the config turns on (encodings excluded)."""
    S = stand_ab()
    base = S.build_features(raw_chunk, serve=serve)
    ids = base.MVT_ID_mvt.to_numpy(dtype="float64")
    out = base.drop(columns=[c for c in base.columns if c.endswith("_sec")])
    blocks = []
    if cfg["features.queue"]:
        blocks.append(S.build_queue(raw_chunk, serve=serve))
    if cfg["features.order"]:
        blocks.append(S.build_order(raw_chunk, serve=serve))
    if cfg["features.weather"]:
        blocks.append(S.build_weather(raw_chunk, serve=serve, w=weather))
    for b in blocks:
        if not np.array_equal(b.MVT_ID_mvt.to_numpy(dtype="float64"), ids):
            raise E.SchemaError("a feature block's rows are not build_features' rows in the same order")
        for c in b.columns:
            if c != "MVT_ID_mvt":
                out[c] = b[c].to_numpy()
    if cfg["features.orl_route_length"]:
        o = orl_build().attach(base[["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]].reset_index(drop=True), orl_routes)
        for c in ORL_COLS:
            out[c] = o[c].to_numpy(dtype="float64")
    out["day"] = raw_day(raw_chunk, ids)
    return out.reset_index(drop=True)


def raw_day(raw: pd.DataFrame, ids) -> np.ndarray:
    """UTC take-off date (YYYY-MM-DD) of each id, from the raw rows (the day-block bootstrap's block)."""
    s = pd.Series(raw.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m-%d").to_numpy(), index=raw.MVT_ID_mvt.to_numpy(dtype="float64"))
    return s.reindex(np.asarray(ids, dtype="float64")).to_numpy(dtype=object)


def attach_adsb(frame: pd.DataFrame, table_dir, serve: bool = False, max_missing_share: float = 0.0) -> tuple:
    """(frame + the ADS-B columns, number of rows served as 'no ADS-B data'). Training rows must all be in the table;
    a scored row without one gets coverage 0, adsb_day_present 0 and NaN features (FM1 notes N1), up to the share."""
    ids = frame.MVT_ID_mvt.to_numpy(dtype="float64")
    t, missing = read_adsb_tables(table_dir, ids, allow_missing=serve)
    share = len(missing) / max(len(ids), 1)
    if serve and share > max_missing_share:
        raise E.SchemaError(f"{len(missing):,} of {len(ids):,} scored matched rows ({share:.4%}) have no ADS-B table row in "
                            f"{table_dir}; above adsb.serve_missing_max_share = {max_missing_share:g}")
    got = t.reindex(ids)
    out = frame.copy()
    for c in t.columns:
        out[c] = got[c].to_numpy(dtype="float64")
    if len(missing):
        miss = np.isin(ids, missing)
        out.loc[miss, "coverage"] = 0.0
        out.loc[miss, "adsb_day_present"] = 0.0
    return out, int(len(missing))


def set_unheard(frame: pd.DataFrame, adsb_cols, rows, network: bool) -> None:
    """IN PLACE: make the rows `rows` (boolean mask) look like flights the receiver network did not hear
    (plans/PREREG_acr_adsb_coverage_regime_2026_10_02.md): ADSB_UNHEARD_STATE and every flight-level ADS-B column NaN; with
    `network` also the airport-level counts (ground_seen_mvt_m10 = 0, moving_mvt_m10 = NaN), a network blackout."""
    rows = np.asarray(rows, dtype=bool)
    if rows.shape != (len(frame),):
        raise ValueError(f"set_unheard: the row mask has shape {rows.shape}, the frame {len(frame):,} rows")
    missing = [c for c in [*ADSB_UNHEARD_STATE, *adsb_cols] if c not in frame.columns]
    if missing:
        raise E.MissingColumnError(f"set_unheard: missing ADS-B column(s) {sorted(set(missing))}")
    if not rows.any():
        return
    for c in adsb_cols:
        if c not in ADSB_UNHEARD_STATE and c not in ADSB_AIRPORT_COLS:
            frame.loc[rows, c] = np.nan
    for c, v in ADSB_UNHEARD_STATE.items():
        frame.loc[rows, c] = v
    if network:
        if "ground_seen_mvt_m10" in frame.columns:
            frame.loc[rows, "ground_seen_mvt_m10"] = 0.0
        if "moving_mvt_m10" in frame.columns:
            frame.loc[rows, "moving_mvt_m10"] = np.nan


def dropout_masks(frame: pd.DataFrame, flight_share: float, day_share: float, seed: int) -> tuple:
    """(flight mask, network mask) for training-time ADS-B dropout, seeded: the network mask takes every row of a random
    `day_share` of (ap, day) groups; the flight mask a random `flight_share` of the remaining JOINED rows."""
    rng = np.random.default_rng([int(seed), 20261002])
    n = len(frame)
    keys = pd.MultiIndex.from_arrays([frame.ap.astype(str).to_numpy(), frame.day.astype(str).to_numpy()])
    codes, uniq = pd.factorize(keys, sort=True)
    pick = rng.random(len(uniq)) < day_share
    network = pick[codes] if len(uniq) else np.zeros(n, dtype=bool)
    joined = frame.joined.to_numpy(dtype="float64", na_value=np.nan) == 1
    flight = joined & ~network & (rng.random(n) < flight_share)
    return flight, network


def _dayoff(mvt: pd.Series, sched: pd.Series, ap: np.ndarray, tz_mode: str, tz_map) -> np.ndarray:
    if tz_mode == "UTC":
        d = (mvt.dt.normalize() - sched.dt.normalize()).dt.days
        return np.clip(d.to_numpy(dtype="float64", na_value=np.nan), 0, 1)
    out = np.full(len(mvt), np.nan)
    for a in np.unique(ap):
        sel = ap == a
        z = tz_map[a]
        m, s = mvt[sel].dt.tz_convert(z), sched[sel].dt.tz_convert(z)
        d = (m.dt.normalize() - s.dt.normalize()).dt.days
        out[sel] = np.clip(d.to_numpy(dtype="float64", na_value=np.nan), 0, 1)
    return out


def unmatched_frame(raw_chunk: pd.DataFrame, serve: bool, cfg) -> pd.DataFrame:
    """Unmatched departures (AOBT_3 null) of one chunk: meta + UNM_NUMERIC + the encoding keys. Training rows need a
    positive label; scored rows need only MVT_TIME (the contract guarantees it)."""
    t = raw_chunk
    dep = (t.PHASE_mvt.astype(str) == "DEP") & t.AOBT_3_flt.isna() & t.MVT_TIME_UTC_mvt.notna()
    if not serve:
        dep &= t.TAXITIME_SEC_mvt.notna() & (t.TAXITIME_SEC_mvt.fillna(0) > 0)
    u = t[dep.to_numpy()].reset_index(drop=True)
    wit = unm_congestion().airport_hour_witness(t)
    u = unm_congestion().attach_witness(u, wit)
    mvt, sched = u.MVT_TIME_UTC_mvt, u.SCHED_TIME_UTC_mvt
    ap = u.ADEP_mvt.astype(str).to_numpy()
    out = pd.DataFrame({
        "MVT_ID_mvt": u.MVT_ID_mvt.to_numpy(dtype="float64"), "ap": ap, "month": mvt.dt.month.to_numpy(),
        "day": mvt.dt.strftime("%Y-%m-%d").to_numpy(dtype=object),
        "sp": (mvt - sched).dt.total_seconds().to_numpy(dtype="float64", na_value=np.nan),
        "hr": mvt.dt.hour.to_numpy(dtype="float64"),
        "dayoff": _dayoff(mvt, sched, ap, cfg["data.dayoff_tz"], cfg["airports.tz"]),
        "hprox_med": u.hprox_med.to_numpy(dtype="float64"), "hprox_n": u.hprox_n.to_numpy(dtype="float64"),
        "y": u.TAXITIME_SEC_mvt.to_numpy(dtype="float64", na_value=np.nan) if not serve else np.nan,
    })
    fill = lambda s: s.astype(object).where(s.notna(), "NA").astype(str)          # noqa: E731
    out["ADEP_mvt"] = ap
    out["RUNWAY_mvt"] = fill(u.RUNWAY_mvt).to_numpy()
    out["stand_pref"] = fill(u.STAND_mvt).str[:2].to_numpy()
    out["airline"] = fill(u.FLIGHT_mvt).str[:3].to_numpy()
    out["AIRCRAFT_OPERATOR_flt"] = fill(u.AIRCRAFT_OPERATOR_flt).to_numpy()
    # Plain numpy object-array concatenation, never pandas Series arithmetic: on a zero-row real chunk (a
    # month-boundary spillover chunk with no unmatched-eligible rows), an empty pandas 'str'/pyarrow-backed
    # operand infers Arrow type 'null', and pandas' '+' has no (null, large_string) string-concat kernel
    # (BC-30). Plain numpy object arrays concatenate with Python '+' regardless of length or dtype origin.
    ap_obj = np.asarray(ap, dtype=object)
    runway_obj = np.asarray(out["RUNWAY_mvt"].to_numpy(), dtype=object)
    stand_obj = np.asarray(fill(u.STAND_mvt).to_numpy(), dtype=object)
    out["ars"] = ap_obj + "|" + runway_obj + "|" + stand_obj
    return out


@dataclass
class Frames:
    matched: pd.DataFrame | None
    unmatched: pd.DataFrame | None
    n_dep: int
    dep_ids: np.ndarray
    n_adsb_missing: int = 0


def build_frames(raw: pd.DataFrame, serve: bool, cfg, weather=None, orl_routes=None) -> Frames:
    """Raw rows (one file's worth or a scored file) -> Frames, chunked on the movement month."""
    m_parts, u_parts = [], []
    for chunk in month_chunks(raw):
        if cfg["lanes.matched"]:
            has_matched = ((chunk.PHASE_mvt.astype(str) == "DEP") & chunk.AOBT_3_flt.notna()).any()
            if has_matched:
                m_parts.append(matched_frame(chunk, serve, cfg, weather, orl_routes))
        if cfg["lanes.unmatched"] and cfg["unmatched.learner"] == "hgb":        # the v26 lane builds its own rows
            u_parts.append(unmatched_frame(chunk, serve, cfg))
    m = pd.concat(m_parts, ignore_index=True) if m_parts else None
    n_missing = 0
    if m is not None and cfg["features.adsb"]:
        m, n_missing = attach_adsb(m, cfg.path("adsb.table_dir"), serve=serve,
                                   max_missing_share=cfg["adsb.serve_missing_max_share"])
    u = pd.concat(u_parts, ignore_index=True) if u_parts else None
    dep = raw.PHASE_mvt.astype(str).to_numpy() == "DEP"
    return Frames(matched=m, unmatched=u, n_dep=int(dep.sum()),
                  dep_ids=raw.MVT_ID_mvt.to_numpy(dtype="float64")[dep], n_adsb_missing=n_missing)


def matched_design(frame: pd.DataFrame, enc: EncodingTables, columns) -> np.ndarray:
    """The float32-valued design (lgbm_submit.design_matrix's precision) as float64, columns in `columns` order.

    Built column by column into one C-contiguous float64 array: each column is cast to float32 (the shipped precision)
    and widened, exactly what `frame[columns].to_numpy(dtype=float32).astype(float64)` computes, without that route's
    copy of the whole frame, second copy of the selected columns and column-major result (which LightGBM and the
    CatBoost worker then re-copied to C order). 2026-10-01: the full-scale fit peaked at 17.8 GB on this path."""
    rows.assert_design_clean(columns, "matched design")
    encoded = enc.transform(frame)                      # encodings take precedence over a frame column of the same name
    missing = [c for c in columns if c not in encoded and c not in frame.columns]
    if missing:
        raise E.MissingColumnError(f"matched design: missing column(s) {missing}")
    out = np.empty((len(frame), len(columns)), dtype=np.float64)
    for j, c in enumerate(columns):
        v = encoded[c] if c in encoded else frame[c].to_numpy()
        out[:, j] = np.asarray(v).astype(np.float32, copy=False)
    return out


def unmatched_design(frame: pd.DataFrame, enc: EncodingTables) -> np.ndarray:
    cols = unmatched_design_columns()
    f = frame.copy()
    for k, v in enc.transform(frame).items():
        f[k] = v
    return f[cols].to_numpy(dtype=np.float64)
