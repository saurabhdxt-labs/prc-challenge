"""UAD — the aircraft's own ADS-B ground track as the off-block estimate for departures without a flight-plan record
(plans/PREREG_uad_unmatched_adsb_offblock_2026_10_04.md, "Definitions (fixed)").

    /opt/homebrew/bin/python3.11 -B scripts/analysis/uad_build.py parked --rows F [F ...] --adsb-dir D [D ...]
        --out NEWDIR [--airports AP ...] [--days DAY ...] [--jobs N]
    /opt/homebrew/bin/python3.11 -B scripts/analysis/uad_build.py table --rows F [F ...] --adsb-dir D [D ...]
        --parked DIR [DIR ...] [--alias-from TABLEDIR ...] --out NEWDIR [--airports AP ...] [--days DAY ...]
        [--jobs N]

Label-blind: `read_rows` loads only ROW_COLS from each parquet through pyarrow column projection, so BLOCK_TIME_UTC_mvt
and TAXITIME_SEC_mvt (LABEL_COLS) never enter this module even when a file carries them.

What it does, in two passes over the per-day ADS-B store (data/adsb/v2 and its y2025_other / y2026 siblings):
  1. `parked` joins every day's departures to their take-off (adsb_features.segment_tracks / takeoff_events /
     join_flights, unchanged) and, for every MATCHED flight (AOBT_3_flt present) whose join is trustworthy enough to
     use (a callsign join, or a time join whose ADS-B aircraft type agrees with AIRCRAFT_TYPE_mvt), records where its
     track was parked just before AOBT_3_flt. Unmatched rows have no AOBT_3_flt, so they never contribute here —
     that absence is exactly the population pass 2 exists to estimate.
  2. `table` pools parked positions into one reference position per (airport, stand) and, for every departure row
     (matched or not), finds where its own track last left that reference before take-off (`stand_exit`) and turns
     that into `est = MVT_TIME_UTC_mvt - exit_time`. Days are joined one at a time (`build_raw`), which also
     gathers, for every unmatched row with a stand reference, each take-off near its movement time with that
     track's exit from the row's stand (the candidates). The rest is settled over the whole table
     (`finalise_table`): a take-off claimed by rows of two days serves one of them (`dedupe_takeoffs`, AMEND-A1
     item 3); the airport's own way of writing an aircraft type is learned from the joins whose aircraft is known
     without the type (`type_aliases`, AMEND-A2); an unmatched row without a callsign join is given the one
     take-off, inside the airport's own offset window, whose track left the row's stand (`offset_windows`,
     `stand_joins`, AMEND-A3); and each row gets `type_equal`, `type_agree`, `usable` and its class A1 / A2 by the
     registered time bands. The same stand rule is scored against the callsign join on a 1-in-10 sample of matched
     departures (`stand_rule_check`, AMEND-A3 stop rule (d)).

Both commands write a report.json carrying the registered parameters (`params`, `params_hash`), the sha256 of this
module and of the shipped join it runs, the absolute input paths with their sha256, the ADS-B sources each directory
declares, the day list and any day with no ADS-B file. The table's report adds the label-free stop-rule statistics,
the alias list, the offset windows and the non-decisional readouts of the registration; the candidates are written
beside the table (cands.parquet).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


AF = _load("adsb_features", "scripts/adsb_features.py")
AE = _load("adsb_extract", "scripts/adsb_extract.py")
R12 = _load("r12_build", "scripts/analysis/r12_build.py")

# ---------------------------------------------------------------- registered constants

STAND_RADIUS_M = 40.0
EXIT_NEXT_MAX_S = 120.0
EXIT_LOOKBACK_N = 20
STATIONARY_KT = 1.0
REPARK_MIN_S = 120.0
RUN_GAP_MAX_S = 180.0
REF_WINDOW_S = (900.0, 20.0)
REF_MIN_FLIGHTS = 3
REF_MAX_MAD_M = 15.0
A1_MAX_S = 3600.0
A2_MAX_S = 5400.0
ALIAS_MIN_ROWS = 20           # AMEND-A2: a coding alias is seen on this many joins whose aircraft is known,
ALIAS_MIN_AIRFRAMES = 5       # from this many different aircraft addresses,
ALIAS_MIN_SHARE = 0.5         # and on this share of the airport's such joins of that ADS-B type
CAND_BRACKET_S = 120.0        # AMEND-A3: a row's candidates are the take-offs this close to its movement time;
WINDOW_Q = (0.005, 0.995)     # the airport's offset window is these quantiles of take-off - MVT on callsign joins
WINDOW_MARGIN_S = 5.0         # of matched departures, widened by this on each side
WINDOW_FLOOR_S = 30.0         # and to at least +- this,
WINDOW_MIN_ROWS = 200         # from at least this many such joins;
CHECK_SAMPLE_MOD = 10         # stop rule (d) scores the matched callsign joins whose MVT_ID_mvt is divisible by this
ROW_COLS = list(AF.FLIGHT_COLS) + ["PHASE_mvt", "AIRCRAFT_TYPE_mvt"]
LABEL_COLS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")

# Not in params(): the order in which two claims on one take-off are ranked (AMEND-A1 item 3, AMEND-A3 item 5), the
# readout bands of the report, and the files a report names.
JOIN_RANK = {"callsign": 0, "stand": 1, "time": 2}
REPORT_AGREE_MAX_S = 600.0
REPORT_BANDS_S = ((0.0, 1800.0), (1800.0, 3600.0), (3600.0, 5400.0))
CODE_FILES = ("scripts/analysis/uad_build.py", "scripts/adsb_features.py", "scripts/adsb_extract.py")


def params() -> dict:
    """Every registered constant above, as recorded in every output report.json."""
    return {
        "STAND_RADIUS_M": STAND_RADIUS_M, "EXIT_NEXT_MAX_S": EXIT_NEXT_MAX_S, "EXIT_LOOKBACK_N": EXIT_LOOKBACK_N,
        "STATIONARY_KT": STATIONARY_KT, "REPARK_MIN_S": REPARK_MIN_S, "RUN_GAP_MAX_S": RUN_GAP_MAX_S,
        "REF_WINDOW_S": list(REF_WINDOW_S), "REF_MIN_FLIGHTS": REF_MIN_FLIGHTS, "REF_MAX_MAD_M": REF_MAX_MAD_M,
        "A1_MAX_S": A1_MAX_S, "A2_MAX_S": A2_MAX_S, "ALIAS_MIN_ROWS": ALIAS_MIN_ROWS,
        "ALIAS_MIN_AIRFRAMES": ALIAS_MIN_AIRFRAMES, "ALIAS_MIN_SHARE": ALIAS_MIN_SHARE,
        "CAND_BRACKET_S": CAND_BRACKET_S, "WINDOW_Q": list(WINDOW_Q), "WINDOW_MARGIN_S": WINDOW_MARGIN_S,
        "WINDOW_FLOOR_S": WINDOW_FLOOR_S, "WINDOW_MIN_ROWS": WINDOW_MIN_ROWS, "CHECK_SAMPLE_MOD": CHECK_SAMPLE_MOD,
    }


def params_hash() -> str:
    """sha256 of params()'s canonical JSON (sorted keys, compact separators), first 12 hex characters."""
    blob = json.dumps(params(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


# ---------------------------------------------------------------- small private helpers (internal to this module)

def _norm_str(s) -> str | None:
    if s is None or pd.isna(s):
        return None
    out = str(s).strip().upper()
    return out or None


def _norm_str_series(s: pd.Series) -> pd.Series:
    """Element-wise _norm_str over a Series, always yielding a real Python None for a null input. Series.map()
    cannot be used for this: pandas' Arrow-backed string dtype (the default since pandas 3) can turn a mapped
    function's `None` return back into float NaN when it reassembles the output Series, which would silently break
    the `str | None` contract this module promises for a normalised stand or aircraft type."""
    return pd.Series([_norm_str(v) for v in s.tolist()], index=s.index, dtype=object)


def _epoch_seconds(ts) -> np.ndarray:
    s = pd.to_datetime(ts, utc=True)
    return (s - AF.EPOCH).dt.total_seconds().to_numpy(dtype="float64")


PARKED_SCHEMA = {"MVT_ID_mvt": "float64", "ADEP_mvt": "object", "stand": "object", "day": "object",
                 "x": "float64", "y": "float64"}
REFS_SCHEMA = {"ADEP_mvt": "object", "stand": "object", "x": "float64", "y": "float64", "n": "int64",
              "mad_m": "float64"}
# What a stand exit gives one row from one track: the fields a table row and a candidate both carry.
EXIT_SCHEMA = {"exit_rel_s": "float64", "continuous": "bool", "parked_seen": "bool", "repark": "bool",
               "exit_ok": "bool", "hold_s": "float64", "n_ground": "float64", "first_ground_rel_s": "float64",
               "first_ground_dist_m": "float64"}
EXIT_COLS = list(EXIT_SCHEMA)
NO_EXIT = {"exit_rel_s": np.nan, "continuous": False, "parked_seen": False, "repark": False, "exit_ok": False,
           "hold_s": np.nan, "n_ground": np.nan, "first_ground_rel_s": np.nan, "first_ground_dist_m": np.nan}
# One row per departure as the per-day pass leaves it (build_raw); finalise_table adds type_equal, type_agree, usable
# and cls.
RAW_SCHEMA = {"MVT_ID_mvt": "float64", "ADEP_mvt": "object", "day": "object", "unmatched": "bool",
              "join_type": "object", "adsb_icao": "object", "t_takeoff": "float64", "dt_takeoff": "float64",
              "row_type": "object", "adsb_type": "object", "ref_n": "float64", **EXIT_SCHEMA}
RAW_COLS = list(RAW_SCHEMA)
TABLE_COLS = ["MVT_ID_mvt", "ADEP_mvt", "day", "unmatched", "join_type", "adsb_icao", "t_takeoff", "dt_takeoff",
              "row_type", "adsb_type", "type_equal", "type_agree", "usable", "ref_n", *EXIT_COLS, "cls"]
TABLE_SCHEMA = {c: {"type_equal": "float64", "type_agree": "float64", "usable": "bool", "cls": "object"}.get(c)
                or RAW_SCHEMA[c] for c in TABLE_COLS}
# One row per (departure, take-off near its movement time): `own` marks the take-off the shipped join gave the row.
CAND_SCHEMA = {"MVT_ID_mvt": "float64", "ADEP_mvt": "object", "own": "bool", "adsb_icao": "object",
               "t_takeoff": "float64", "dt_takeoff": "float64", "adsb_type": "object", **EXIT_SCHEMA}
CAND_COLS = list(CAND_SCHEMA)
IDENTITY_COLS = ["MVT_ID_mvt", "ADEP_mvt", "row_type", "adsb_type", "adsb_icao"]
IDENTITY_SOURCE_COLS = ["MVT_ID_mvt", "ADEP_mvt", "unmatched", "join_type", "row_type", "adsb_type", "adsb_icao"]
ALIAS_SCHEMA = {"ADEP_mvt": "object", "row_type": "object", "adsb_type": "object", "n": "int64",
               "airframes": "int64", "share": "float64"}
# What a row loses when its claim on a take-off is dropped: everything that came from the join. Its own aircraft
# type (row_type) and its stand's reference count (ref_n) are properties of the row and stay.
JOIN_BLANK = {"join_type": "", "adsb_icao": "", "t_takeoff": np.nan, "dt_takeoff": np.nan, "adsb_type": "",
              "type_equal": np.nan, "type_agree": np.nan, "usable": False, "exit_rel_s": np.nan,
              "continuous": False, "parked_seen": False, "repark": False, "exit_ok": False, "hold_s": np.nan,
              "n_ground": np.nan, "first_ground_rel_s": np.nan, "first_ground_dist_m": np.nan, "cls": ""}


def _empty_frame(spec: dict) -> pd.DataFrame:
    """An empty DataFrame with EXPLICIT per-column dtypes. `pd.DataFrame(columns=[...])` (no data) instead defaults
    every column to `object`, which is inconsistent with the same column's dtype once populated (float64 / int64)
    and silently changes the result of both `pd.concat` (an empty chunk can upcast a numeric column to object) and
    arrow-schema inference (the parquet file's bytes then differ for logically-identical data) depending on whether
    any one contributing chunk happened to be empty."""
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in spec.items()})[list(spec)]


# ---------------------------------------------------------------- 1. rows

def read_rows(paths) -> pd.DataFrame:
    """Read ROW_COLS only (pyarrow column projection: LABEL_COLS never load even when a file has them) from each
    parquet in `paths`, keep PHASE_mvt == "DEP" rows, and add `mvt_s` / `aobt_s` (float epoch seconds; aobt_s is NaN
    when AOBT_3_flt is null), `day` (UTC date string of MVT_TIME_UTC_mvt), `unmatched` (AOBT_3_flt is null) and
    `stand` (norm_stand(STAND_mvt)). Refuses a file missing a ROW_COLS column, a null MVT_ID_mvt, or an MVT_ID_mvt
    duplicated across the inputs — all ValueError, each naming the affected count."""
    frames = []
    for raw_path in paths:
        path = pathlib.Path(raw_path)
        names = set(pq.read_schema(str(path)).names)
        missing = [c for c in ROW_COLS if c not in names]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}")
        frames.append(pq.read_table(str(path), columns=ROW_COLS).to_pandas())
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ROW_COLS)
    rows = rows[rows.PHASE_mvt == "DEP"].reset_index(drop=True)

    n_null = int(rows.MVT_ID_mvt.isna().sum())
    if n_null:
        raise ValueError(f"{n_null} row(s) with a null MVT_ID_mvt")
    dup = rows.MVT_ID_mvt.duplicated(keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} row(s) share a duplicated MVT_ID_mvt across the input files")

    rows["MVT_ID_mvt"] = rows["MVT_ID_mvt"].astype("float64")
    rows["mvt_s"] = _epoch_seconds(rows.MVT_TIME_UTC_mvt)
    rows["aobt_s"] = _epoch_seconds(rows.AOBT_3_flt)
    rows["day"] = pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    rows["unmatched"] = rows.AOBT_3_flt.isna()
    rows["stand"] = _norm_str_series(rows.STAND_mvt)
    return rows


def norm_stand(s) -> str | None:
    """Stand identifier normalisation: strip, upper-case; null or empty -> None. The same strip/upper rule as
    scripts/analysis/orl_build.norm_ref, except that an unknown stand here is reported as None (orl_build.norm_ref
    returns "" for a null reference; this module never uses a sentinel string for "unknown")."""
    return _norm_str(s)


# ---------------------------------------------------------------- 2. geometry

def xy(lat, lon, airport) -> tuple[np.ndarray, np.ndarray]:
    """Metres east / north of APTS[airport]'s reference point (equirectangular at that latitude; the same formula as
    adsb_features._xy, which is private)."""
    if airport not in AE.APTS:
        raise ValueError(f"unknown airport: {airport!r}")
    lat0, lon0 = AE.APTS[airport]
    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)
    x = (lon_arr - lon0) * AF.M_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
    y = (lat_arr - lat0) * AF.M_PER_DEG_LAT
    return x, y


# ---------------------------------------------------------------- 3. the day store

def read_day(adsb_dirs, day, airport) -> pd.DataFrame | None:
    """One day's ADS-B rows for `airport`: that day's own file plus the previous day's last GROUND_LOOKBACK_S seconds
    and the next day's first NEXT_DAY_HEAD_S seconds (adsb_features.build_period's day-slicing convention, copied
    here since its `_day_slice` helper is private). `adsb_dirs` are resolved to absolute paths first (BC-40: a
    relative directory must reach the reader from the CALLER's cwd, not whatever cwd a later step runs under). The
    day's own file is taken from the FIRST directory that holds it; a neighbour missing from every directory simply
    contributes nothing. Returns None only when the day's OWN file is in no directory. A file that exists but cannot
    be read raises — it is never treated as 'no data'."""
    dirs = [pathlib.Path(d).resolve() for d in adsb_dirs]
    d = pd.Timestamp(day)
    d = d.tz_localize("UTC") if d.tz is None else d.tz_convert("UTC")
    d = d.normalize()
    start = (d - AF.EPOCH).total_seconds()

    def _find(day_ts):
        name = f"adsb_{day_ts:%Y-%m-%d}.parquet"
        for base in dirs:
            p = base / name
            if p.exists():
                return p
        return None

    def _slice(path, lo, hi):
        if path is None:
            return None
        frame = pd.read_parquet(path)
        return frame[(frame.t >= lo) & (frame.t < hi) & (frame.airport == airport)]

    cur_path = _find(d)
    if cur_path is None:
        return None
    prev_path = _find(d - pd.Timedelta(days=1))
    next_path = _find(d + pd.Timedelta(days=1))

    cur = _slice(cur_path, start, start + 86_400.0)
    prev = _slice(prev_path, start - AF.GROUND_LOOKBACK_S, start)
    nxt = _slice(next_path, start + 86_400.0, start + 86_400.0 + AF.NEXT_DAY_HEAD_S)

    parts = [p for p in (prev, cur, nxt) if p is not None]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- 4. the join

def _join_parts(flights: pd.DataFrame, adsb: pd.DataFrame):
    """join_day's work, with the take-off events and each track's ADS-B type also handed back (build_raw needs them
    for the candidates): (j, tr, ev, last_type)."""
    tr = AF.segment_tracks(adsb)
    ev = AF.takeoff_events(tr)
    j = AF.join_flights(flights[AF.FLIGHT_COLS], ev)

    type_by_id = flights.set_index("MVT_ID_mvt").AIRCRAFT_TYPE_mvt
    j["AIRCRAFT_TYPE_mvt"] = j.MVT_ID_mvt.map(type_by_id)
    j["adsb_icao"] = j.track_id.map(tr.groupby("track_id").icao.first())
    last_type = tr.groupby("track_id").type.agg(lambda s: s.dropna().iloc[-1] if s.notna().any() else None)
    j["adsb_type"] = j.track_id.map(last_type)

    a = _norm_str_series(j.adsb_type)
    b = _norm_str_series(j.AIRCRAFT_TYPE_mvt)
    both = a.notna() & b.notna()
    type_agree = pd.Series(np.nan, index=j.index)
    type_agree[both] = (a[both] == b[both]).astype(float)
    j["type_agree"] = type_agree

    j["usable"] = (j.join_type == "callsign") | ((j.join_type == "time") & (j.type_agree == 1.0))
    return j, tr, ev, last_type


def join_day(flights: pd.DataFrame, adsb: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The shipped take-off join (adsb_features.segment_tracks / takeoff_events / join_flights, unchanged) plus an
    aircraft-type agreement check. Returns (j, tr): j = join_flights(...) with `adsb_icao` (the joined track's
    aircraft address, null when unjoined), `adsb_type` (the joined track's last non-null `type`, None when
    unjoined), `type_agree` (1.0 / 0.0 by exact equality when both sides report a type, NaN when either is missing)
    and `usable` (a callsign join, or a time join whose type_agree is 1.0) added; tr = the segmented tracks, so
    callers reuse them (parked_position / stand_exit) instead of segmenting twice."""
    j, tr, _, _ = _join_parts(flights, adsb)
    return j, tr


# ---------------------------------------------------------------- 5. stand references

def parked_position(t, x, y, gs, aobt_s) -> tuple[float, float] | None:
    """The row's own LAST stationary (gs < STATIONARY_KT) ground sample in
    [aobt_s - REF_WINDOW_S[0], aobt_s - REF_WINDOW_S[1]], over the ON-GROUND samples of one track (arrays must
    already be filtered to on-ground and sorted by t). None without a finite aobt_s or without a qualifying
    sample."""
    if aobt_s is None or not np.isfinite(aobt_s):
        return None
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gs = np.asarray(gs, dtype=float)
    lo, hi = aobt_s - REF_WINDOW_S[0], aobt_s - REF_WINDOW_S[1]
    idx = np.flatnonzero((gs < STATIONARY_KT) & (t >= lo) & (t <= hi))
    if len(idx) == 0:
        return None
    k = idx[-1]
    return float(x[k]), float(y[k])


def stand_references(parked: pd.DataFrame) -> pd.DataFrame:
    """One row per (ADEP_mvt, stand) with the median (x, y) of every contributing flight's parked position, kept
    only when at least REF_MIN_FLIGHTS flights contribute and the median absolute deviation of their positions from
    that median point is at most REF_MAX_MAD_M. A null stand is filtered out before any grouping (pandas groupby /
    merge on a NaN key is a known bug class in this codebase). Deterministic order: sorted by (ADEP_mvt, stand)."""
    p = parked[parked.stand.notna()]
    if p.empty:
        return _empty_frame(REFS_SCHEMA)
    out_rows = []
    for (ap, st), g in p.groupby(["ADEP_mvt", "stand"], sort=False):
        mx, my = float(g.x.median()), float(g.y.median())
        d = np.hypot(g.x.to_numpy(dtype=float) - mx, g.y.to_numpy(dtype=float) - my)
        out_rows.append({"ADEP_mvt": ap, "stand": st, "x": mx, "y": my, "n": int(len(g)), "mad_m": float(np.median(d))})
    out = pd.DataFrame(out_rows, columns=list(REFS_SCHEMA))
    out = out.astype(REFS_SCHEMA)
    out = out[(out.n >= REF_MIN_FLIGHTS) & (out.mad_m <= REF_MAX_MAD_M)]
    return out.sort_values(["ADEP_mvt", "stand"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------- 6. stand exit

def stationary_runs(t, gs) -> list[tuple[int, int]]:
    """Maximal runs of CONSECUTIVE samples that are all stationary (gs < STATIONARY_KT) and at most RUN_GAP_MAX_S
    apart, kept when the run's span (t[end] - t[start]) is at least REPARK_MIN_S. Returns (start_index, end_index)
    pairs into the input arrays; every sample from start to end belongs to the run. A moving sample ends a run: an
    aircraft that stops, creeps forward and stops again (a queue) has two short stops, not one long one."""
    t = np.asarray(t, dtype=float)
    gs = np.asarray(gs, dtype=float)
    still = gs < STATIONARY_KT
    runs: list[tuple[int, int]] = []
    n = len(t)
    i = 0
    while i < n:
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and still[j + 1] and t[j + 1] - t[j] <= RUN_GAP_MAX_S:
            j += 1
        if t[j] - t[i] >= REPARK_MIN_S:
            runs.append((i, j))
        i = j + 1
    return runs


def stand_exit(t, x, y, gs, t_takeoff, ref_xy, all_refs_xy) -> dict:
    """Inputs are the ON-GROUND samples of one joined track with t < t_takeoff, sorted by t. `ref_xy` is the row's
    own stand reference (x, y) or None; `all_refs_xy` is an (n, 2) array of every stand reference of that airport.

    k = index of the LAST sample within STAND_RADIUS_M of ref_xy (none, no ref, or no samples -> exit_t NaN and
    every flag False). continuous = a sample k+1 exists and t[k+1] - t[k] <= EXIT_NEXT_MAX_S. parked_seen = any of
    the EXIT_LOOKBACK_N samples ending at k (max(0, k - (EXIT_LOOKBACK_N - 1)) .. k) has gs < STATIONARY_KT.
    repark = some stationary_runs() run that starts after k has its median position within STAND_RADIUS_M of ANY
    row of all_refs_xy. hold_s = total duration of every stationary_runs() run
    that starts after k, wherever it is (a long stop far from every reference does not itself invalidate the exit).
    first_ground_dist_m = distance of the very first input sample from ref_xy (NaN without a ref)."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    gs = np.asarray(gs, dtype=float)
    n = len(t)

    first_ground_t = float(t[0]) if n else np.nan
    if ref_xy is None or n == 0:
        first_ground_dist_m = np.nan
    else:
        first_ground_dist_m = float(np.hypot(x[0] - ref_xy[0], y[0] - ref_xy[1]))

    out = {"exit_t": np.nan, "continuous": False, "parked_seen": False, "repark": False, "hold_s": 0.0,
          "n_ground": int(n), "first_ground_t": first_ground_t, "first_ground_dist_m": first_ground_dist_m}
    if ref_xy is None or n == 0:
        return out

    dist = np.hypot(x - ref_xy[0], y - ref_xy[1])
    near = np.flatnonzero(dist <= STAND_RADIUS_M)
    if len(near) == 0:
        return out
    k = int(near[-1])
    out["exit_t"] = float(t[k])
    out["continuous"] = bool(k + 1 < n and (t[k + 1] - t[k]) <= EXIT_NEXT_MAX_S)
    lo = max(0, k - (EXIT_LOOKBACK_N - 1))
    out["parked_seen"] = bool(np.any(gs[lo:k + 1] < STATIONARY_KT))

    refs = np.asarray(all_refs_xy, dtype=float)
    refs = refs.reshape(-1, 2) if refs.size else np.empty((0, 2))
    runs = stationary_runs(t, gs)
    hold_s = 0.0
    repark = False
    for s, e in runs:
        if s <= k:
            continue
        hold_s += t[e] - t[s]
        if len(refs):
            mx, my = np.median(x[s:e + 1]), np.median(y[s:e + 1])
            if np.any(np.hypot(refs[:, 0] - mx, refs[:, 1] - my) <= STAND_RADIUS_M):
                repark = True
    out["hold_s"] = float(hold_s)
    out["repark"] = bool(repark)
    return out


def exit_ok(d: dict) -> bool:
    """Whether a stand_exit() result is trustworthy enough to price: a finite exit, a continuous departure from it,
    an actually-stationary approach, and no later return to any known stand."""
    return bool(np.isfinite(d["exit_t"]) and d["continuous"] and d["parked_seen"] and not d["repark"])


def classify(est_s, usable, ok) -> str:
    """"A1" / "A2" by the registered time bands, only when `usable` and `ok`; "" otherwise (including a NaN or
    non-positive estimate)."""
    if est_s is None or not np.isfinite(est_s):
        return ""
    if not (usable and ok):
        return ""
    if 0 < est_s <= A1_MAX_S:
        return "A1"
    if A1_MAX_S < est_s <= A2_MAX_S:
        return "A2"
    return ""


# ---------------------------------------------------------------- 7. what is settled over the whole table

def dedupe_takeoffs(table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One take-off serves one flight over the whole table (AMEND-A1 item 3). Days are joined one at a time with
    their neighbours' edges, so a row of each of two consecutive days can claim the same take-off: the same aircraft
    address leaving the same airport at the same take-off time. Of the rows that share one, the claim kept is a
    callsign join before a time join, then the smaller |dt_takeoff|, then the smaller MVT_ID_mvt; every other
    claimant becomes an unjoined row (JOIN_BLANK, for the columns the frame has). Returns (a new frame in the input's
    row order and dtypes, the number of claims dropped); `table` itself is not modified. Refuses a joined row with an
    unknown join type, or without an aircraft address or a take-off time (ValueError)."""
    out = table.copy()
    jt = table.join_type.to_numpy(dtype=object)
    pos = np.flatnonzero(jt != "")
    if len(pos) == 0:
        return out, 0
    unknown = sorted({str(v) for v in jt[pos]} - set(JOIN_RANK))
    if unknown:
        raise ValueError(f"unknown join type(s) {unknown}")
    icao = table.adsb_icao.to_numpy(dtype=object)[pos]
    t = table.t_takeoff.to_numpy(dtype=float)[pos]
    bad = int(np.sum(pd.isna(icao) | (icao == "") | ~np.isfinite(t)))
    if bad:
        raise ValueError(f"{bad} joined row(s) without an aircraft address or a take-off time")
    claims = pd.DataFrame({"pos": pos, "ap": table.ADEP_mvt.to_numpy(dtype=object)[pos], "icao": icao, "t": t,
                           "rank": [JOIN_RANK[v] for v in jt[pos]],
                           "adt": np.abs(table.dt_takeoff.to_numpy(dtype=float)[pos]),
                           "id": table.MVT_ID_mvt.to_numpy(dtype=float)[pos]})
    claims = claims.sort_values(["rank", "adt", "id"], kind="mergesort")
    lose = claims.pos[claims.duplicated(["ap", "icao", "t"], keep="first")].to_numpy()
    for col, val in JOIN_BLANK.items():
        if col in out.columns:
            out.iloc[lose, out.columns.get_loc(col)] = val
    return out, int(len(lose))


def near_duplicate_takeoffs(table: pd.DataFrame) -> int:
    """How many rows hold a take-off of the same aircraft at the same airport that is later than another row's by
    more than 0 and at most adsb_features.ROLL_MERGE_S seconds (an unjoined row has no take-off time and is never
    counted). The shipped event detector merges such flips into one roll, so one day's view holds none; a count
    above zero means two days' views resolved one roll differently, and matching take-offs by their exact time
    (dedupe_takeoffs) missed a shared one."""
    c = table.sort_values(["ADEP_mvt", "adsb_icao", "t_takeoff"], kind="mergesort")
    if len(c) < 2:
        return 0
    ap = c.ADEP_mvt.to_numpy(dtype=object)
    icao = c.adsb_icao.to_numpy(dtype=object)
    gap = np.diff(c.t_takeoff.to_numpy(dtype=float))
    same = (ap[1:] == ap[:-1]) & (icao[1:] == icao[:-1])
    return int(np.sum(same & (gap > 0) & (gap <= AF.ROLL_MERGE_S)))


def identity_rows(table: pd.DataFrame) -> pd.DataFrame:
    """The joins whose aircraft is known without looking at its type: callsign joins of matched departures with both
    aircraft types present (IDENTITY_COLS). They are what a coding alias is learned from (type_aliases)."""
    keep = ((~table.unmatched.to_numpy(dtype=bool)) & (table.join_type.to_numpy(dtype=object) == "callsign")
            & (table.row_type.to_numpy(dtype=object) != "") & (table.adsb_type.to_numpy(dtype=object) != ""))
    return table.loc[keep, IDENTITY_COLS].reset_index(drop=True)


def type_aliases(identity: pd.DataFrame) -> pd.DataFrame:
    """Coding aliases of the aircraft type, per airport (AMEND-A2): a pair (row_type R, adsb_type T), R != T, is
    listed when it occurs on at least ALIAS_MIN_ROWS rows of `identity`, from at least ALIAS_MIN_AIRFRAMES different
    aircraft addresses, and on at least ALIAS_MIN_SHARE of that airport's rows whose ADS-B type is T (the airport
    writes T as R more often than not). Columns ADEP_mvt, row_type, adsb_type, n, airframes, share; sorted by the
    first three. Refuses a row with a missing or blank airport, type or aircraft address (ValueError)."""
    if len(identity) == 0:
        return _empty_frame(ALIAS_SCHEMA)
    keys = identity[["ADEP_mvt", "row_type", "adsb_type", "adsb_icao"]].astype(object)
    blank = (keys.isna() | (keys == "")).any(axis=1)
    if blank.any():
        raise ValueError(f"{int(blank.sum())} identity row(s) with a missing airport, aircraft type or address")
    pairs = keys.groupby(["ADEP_mvt", "row_type", "adsb_type"], sort=True).adsb_icao.agg(
        n="size", airframes="nunique").reset_index()
    of_type = keys.groupby(["ADEP_mvt", "adsb_type"], sort=True).size().rename("n_type").reset_index()
    pairs = pairs.merge(of_type, on=["ADEP_mvt", "adsb_type"], how="left", validate="many_to_one")
    pairs["share"] = pairs.n / pairs.n_type
    keep = ((pairs.row_type != pairs.adsb_type) & (pairs.n >= ALIAS_MIN_ROWS)
            & (pairs.airframes >= ALIAS_MIN_AIRFRAMES) & (pairs.share >= ALIAS_MIN_SHARE))
    out = pairs.loc[keep, list(ALIAS_SCHEMA)].astype(ALIAS_SCHEMA)
    return out.sort_values(["ADEP_mvt", "row_type", "adsb_type"], kind="mergesort").reset_index(drop=True)


def type_agreement(table: pd.DataFrame, aliases: pd.DataFrame) -> np.ndarray:
    """Per row of `table`: 1.0 when row_type equals adsb_type or (ADEP_mvt, row_type, adsb_type) is a row of
    `aliases` (the row's own airport, that direction only); 0.0 when both types are present and neither holds; NaN
    when either type is missing (an unjoined row has no ADS-B type)."""
    ap = table.ADEP_mvt.to_numpy(dtype=object)
    row = table.row_type.to_numpy(dtype=object)
    adsb = table.adsb_type.to_numpy(dtype=object)
    both = (row != "") & (adsb != "")
    known = set(zip(aliases.ADEP_mvt.tolist(), aliases.row_type.tolist(), aliases.adsb_type.tolist()))
    alias = np.fromiter((k in known for k in zip(ap, row, adsb)), dtype=bool, count=len(table))
    out = np.full(len(table), np.nan)
    out[both] = ((row[both] == adsb[both]) | alias[both]).astype(float)
    return out


def offset_windows(table: pd.DataFrame) -> dict:
    """The airport's own offset between the ADS-B take-off and the movement clock (AMEND-A3 item 2): per airport,
    (lo, hi) = the WINDOW_Q quantiles of dt_takeoff over the table's callsign joins of matched departures, widened by
    WINDOW_MARGIN_S on each side and then to include +-WINDOW_FLOOR_S. An airport with fewer than WINDOW_MIN_ROWS
    such joins has no entry."""
    keep = (~table.unmatched.to_numpy(dtype=bool)) & (table.join_type.to_numpy(dtype=object) == "callsign")
    ap = table.ADEP_mvt.to_numpy(dtype=object)[keep]
    dt = table.dt_takeoff.to_numpy(dtype=float)[keep]
    out = {}
    for a in sorted(set(ap)):
        d = dt[ap == a]
        if len(d) < WINDOW_MIN_ROWS:
            continue
        lo, hi = np.quantile(d, WINDOW_Q[0]), np.quantile(d, WINDOW_Q[1])
        out[a] = (float(min(lo - WINDOW_MARGIN_S, -WINDOW_FLOOR_S)), float(max(hi + WINDOW_MARGIN_S, WINDOW_FLOOR_S)))
    return out


def consistent_candidates(cands: pd.DataFrame, windows: dict, held_by: dict) -> np.ndarray:
    """Per row of `cands`, whether the candidate is stand-consistent (AMEND-A3 item 3): its dt_takeoff inside its
    airport's window (both ends inclusive; no window, never), its exit passing the guards (exit_ok),
    0 < exit_rel_s <= A2_MAX_S, and its take-off not held by ANOTHER row's callsign join (`held_by`:
    {(ADEP_mvt, adsb_icao, t_takeoff): the MVT_ID_mvt of the callsign join that holds it})."""
    n = len(cands)
    if n == 0:
        return np.zeros(0, dtype=bool)
    ap = cands.ADEP_mvt.to_numpy(dtype=object)
    dt = cands.dt_takeoff.to_numpy(dtype=float)
    est = cands.exit_rel_s.to_numpy(dtype=float)
    lo = np.array([windows[a][0] if a in windows else np.nan for a in ap], dtype=float)
    hi = np.array([windows[a][1] if a in windows else np.nan for a in ap], dtype=float)
    inside = (dt >= lo) & (dt <= hi)
    fits = cands.exit_ok.to_numpy(dtype=bool) & (est > 0) & (est <= A2_MAX_S)
    keys = zip(ap, cands.adsb_icao.to_numpy(dtype=object), cands.t_takeoff.to_numpy(dtype=float))
    ids = cands.MVT_ID_mvt.to_numpy(dtype=float)
    held = np.fromiter((held_by.get(k, i) != i for k, i in zip(keys, ids)), dtype=bool, count=n)
    return inside & fits & ~held


def stand_joins(cands: pd.DataFrame, windows: dict, held_by: dict) -> tuple[pd.DataFrame, dict]:
    """The stand rule over the candidates of the rows it is asked about: a row with exactly one stand-consistent
    candidate is given it. Returns (those candidate rows, CAND_COLS, sorted by MVT_ID_mvt; {"rows": rows with any
    candidate, "one": rows joined, "several": rows with more than one consistent candidate, "none": rows with
    none})."""
    ok = consistent_candidates(cands, windows, held_by)
    n_rows = int(cands.MVT_ID_mvt.nunique())
    good = cands[ok]
    per_row = good.groupby("MVT_ID_mvt", sort=True).size()
    one = per_row.index[per_row == 1]
    picks = good[good.MVT_ID_mvt.isin(one)].sort_values("MVT_ID_mvt", kind="mergesort").reset_index(drop=True)
    counts = {"rows": n_rows, "one": int(len(one)), "several": int((per_row > 1).sum()),
              "none": int(n_rows - len(per_row))}
    return picks[CAND_COLS], counts


def stand_rule_check(cands: pd.DataFrame, table: pd.DataFrame, windows: dict, held_by: dict) -> dict:
    """AMEND-A3 stop rule (d): the stand rule scored where the answer is known. Over the matched, callsign-joined
    rows of `table` that have candidates, at the airports with a window, counting a row only when its own take-off
    (`own`) could give it a class (exit_ok, 0 < exit_rel_s <= A2_MAX_S): the rule, run as if the row had no callsign
    join, finds that take-off ("right"), another one ("wrong"), several ("ambiguous") or none ("none"). Per airport
    with a window: {"n", "right", "wrong", "ambiguous", "none"}."""
    out = {ap: {"n": 0, "right": 0, "wrong": 0, "ambiguous": 0, "none": 0} for ap in sorted(windows)}
    if len(cands) == 0 or not windows:
        return out
    known = (~table.unmatched.to_numpy(dtype=bool)) & (table.join_type.to_numpy(dtype=object) == "callsign")
    c = cands[cands.MVT_ID_mvt.isin(table.MVT_ID_mvt.to_numpy(dtype=float)[known]) & cands.ADEP_mvt.isin(list(windows))]
    if len(c) == 0:
        return out
    ok = consistent_candidates(c, windows, held_by)
    own = c.own.to_numpy(dtype=bool)
    est = c.exit_rel_s.to_numpy(dtype=float)
    per = pd.DataFrame({"id": c.MVT_ID_mvt.to_numpy(dtype=float), "ap": c.ADEP_mvt.to_numpy(dtype=object), "ok": ok,
                        "own_fits": own & c.exit_ok.to_numpy(dtype=bool) & (est > 0) & (est <= A2_MAX_S),
                        "own_ok": own & ok})
    per = per.groupby("id", sort=True).agg(ap=("ap", "first"), n_ok=("ok", "sum"), own_fits=("own_fits", "any"),
                                           own_ok=("own_ok", "any"))
    per = per[per.own_fits]
    for ap, g in per.groupby("ap", sort=True):
        single = g[g.n_ok == 1]
        out[ap] = {"n": int(len(g)), "right": int(single.own_ok.sum()), "wrong": int((~single.own_ok).sum()),
                   "ambiguous": int((g.n_ok > 1).sum()), "none": int((g.n_ok == 0).sum())}
    return out


def _apply_stand_joins(table: pd.DataFrame, picks: pd.DataFrame) -> pd.DataFrame:
    """The rows named by `picks` take that candidate as their join: join_type "stand", and the candidate's aircraft
    address, take-off time and offset, ADS-B type and stand exit."""
    out = table.copy()
    if len(picks) == 0:
        return out
    pos_of = pd.Series(np.arange(len(out)), index=out.MVT_ID_mvt.to_numpy(dtype=float))
    pos = pos_of.loc[picks.MVT_ID_mvt.to_numpy(dtype=float)].to_numpy()
    out.iloc[pos, out.columns.get_loc("join_type")] = "stand"
    for col in ["adsb_icao", "t_takeoff", "dt_takeoff", "adsb_type", *EXIT_COLS]:
        out.iloc[pos, out.columns.get_loc(col)] = picks[col].to_numpy()
    return out


def finalise_table(raw: pd.DataFrame, cands: pd.DataFrame | None = None,
                   extra_identity: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """The whole-table half of pass 2, over the rows (RAW_COLS) and candidates (CAND_COLS) build_raw left, in this
    order: dedupe_takeoffs; type_aliases over the table's own identity_rows pooled with `extra_identity` (another
    table's identity rows - for the 2026 build, 2025's); offset_windows; stand_joins for the unmatched rows without
    a callsign join, which replaces whatever the shipped join gave them; dedupe_takeoffs again (a stand join ranks
    after a callsign join and before a time join); `type_equal` (the two types equal, NaN when one is missing);
    `type_agree` = type_agreement with the aliases; `usable` = a callsign join, a stand join whose type_agree is 1.0,
    or a MATCHED row's time join whose type_agree is 1.0 (AMEND-A3 item 4: an unmatched row's time join gives no
    class); `cls` = classify(exit_rel_s, usable, exit_ok); stand_rule_check. Returns (the table: TABLE_COLS, sorted
    by MVT_ID_mvt; {"takeoff_claims_dropped", "aliases" (records), "offset_windows" ({airport: [lo, hi]}),
    "stand_joins" (stand_joins' counts), "stand_rule_check"})."""
    cands = _empty_frame(CAND_SCHEMA) if cands is None else cands
    t, n_dropped = dedupe_takeoffs(raw)
    identity = identity_rows(t)
    if extra_identity is not None and len(extra_identity):
        identity = pd.concat([identity, extra_identity[IDENTITY_COLS]], ignore_index=True)
    aliases = type_aliases(identity)
    windows = offset_windows(t)

    by_callsign = t.join_type.to_numpy(dtype=object) == "callsign"
    ids = t.MVT_ID_mvt.to_numpy(dtype=float)
    held_by = dict(zip(zip(t.ADEP_mvt.to_numpy(dtype=object)[by_callsign], t.adsb_icao.to_numpy(dtype=object)[by_callsign],
                           t.t_takeoff.to_numpy(dtype=float)[by_callsign]), ids[by_callsign]))
    asking = ids[t.unmatched.to_numpy(dtype=bool) & ~by_callsign]
    picks, counts = stand_joins(cands[cands.MVT_ID_mvt.isin(asking)], windows, held_by)
    t, n_more = dedupe_takeoffs(_apply_stand_joins(t, picks))

    row = t.row_type.to_numpy(dtype=object)
    adsb = t.adsb_type.to_numpy(dtype=object)
    both = (row != "") & (adsb != "")
    type_equal = np.full(len(t), np.nan)
    type_equal[both] = (row[both] == adsb[both]).astype(float)
    t["type_equal"] = type_equal
    t["type_agree"] = type_agreement(t, aliases)
    join_type = t.join_type.to_numpy(dtype=object)
    agree = t.type_agree.to_numpy(dtype=float) == 1.0
    matched = ~t.unmatched.to_numpy(dtype=bool)
    t["usable"] = (join_type == "callsign") | ((join_type == "stand") & agree) | ((join_type == "time") & matched & agree)
    t["cls"] = [classify(e, u, ok) for e, u, ok in zip(t.exit_rel_s.to_numpy(dtype=float),
                                                         t.usable.to_numpy(dtype=bool),
                                                         t.exit_ok.to_numpy(dtype=bool))]
    t = t[TABLE_COLS].astype(TABLE_SCHEMA).sort_values("MVT_ID_mvt", kind="mergesort").reset_index(drop=True)
    info = {"takeoff_claims_dropped": n_dropped + n_more, "aliases": aliases.to_dict("records"),
            "offset_windows": {ap: [lo, hi] for ap, (lo, hi) in windows.items()}, "stand_joins": counts,
            "stand_rule_check": stand_rule_check(cands, t, windows, held_by)}
    return t, info


def agreement_by_band(est_s, resid_s) -> dict:
    """Readout of the registration (AMEND-A1 item 5), not a parameter: for each band (lo, hi] of REPORT_BANDS_S,
    keyed "lo-hi", {"n": rows with lo < est_s <= hi and a finite resid_s, "share": the share of those with
    |resid_s| <= REPORT_AGREE_MAX_S, NaN when n is 0}."""
    est = np.asarray(est_s, dtype=float)
    resid = np.asarray(resid_s, dtype=float)
    out = {}
    for lo, hi in REPORT_BANDS_S:
        m = (est > lo) & (est <= hi) & np.isfinite(resid)
        n = int(m.sum())
        share = float(np.mean(np.abs(resid[m]) <= REPORT_AGREE_MAX_S)) if n else float("nan")
        out[f"{lo:.0f}-{hi:.0f}"] = {"n": n, "share": share}
    return out


# ---------------------------------------------------------------- 8. build_parked / build_raw / build_table

def build_parked(rows: pd.DataFrame, adsb_dirs, days=None, airports=None, log=print,
                 progress=None) -> tuple[pd.DataFrame, dict]:
    """Pass 1: for every (day, airport) pair that occurs in `rows` (optionally restricted to `days` / `airports`),
    join that day's ADS-B and, for every USABLE, MATCHED (AOBT_3_flt not null) departure with a known stand, record
    its parked position (parked_position() over its joined track's on-ground samples). Unmatched rows are never
    candidates (no AOBT_3_flt, so parked_position has no window to search). Returns (parked, report); report counts
    departures / joined-by-type / usable / parked-seen per airport and lists days with no ADS-B file anywhere.
    `progress(i, n, day, airport)` is called as pair i of n starts (a heartbeat for long builds)."""
    pairs = rows[["day", "ADEP_mvt"]].drop_duplicates()
    if days is not None:
        pairs = pairs[pairs.day.isin(list(days))]
    if airports is not None:
        pairs = pairs[pairs.ADEP_mvt.isin(list(airports))]
    pairs = pairs.sort_values(["day", "ADEP_mvt"], kind="mergesort")

    parked_rows = []
    by_airport: dict = {}
    missing_days: list = []
    seen_missing: set = set()
    n_pairs = len(pairs)
    for i, (day, ap) in enumerate(pairs.itertuples(index=False), 1):
        if progress is not None:
            progress(i, n_pairs, day, ap)
        sub = rows[(rows.day == day) & (rows.ADEP_mvt == ap)]
        stat = by_airport.setdefault(ap, {"departures": 0, "joined_by_type": {}, "usable": 0, "parked_seen": 0})
        stat["departures"] += int(len(sub))
        adsb = read_day(adsb_dirs, day, ap)
        if adsb is None:
            if day not in seen_missing:
                missing_days.append(day)
                seen_missing.add(day)
            log(f"uad_build parked: no ADS-B file for {day} ({ap}); {len(sub)} departure(s) uncovered")
            continue
        j, tr = join_day(sub, adsb)
        for jt, cnt in j.join_type.fillna("none").value_counts().items():
            stat["joined_by_type"][jt] = stat["joined_by_type"].get(jt, 0) + int(cnt)
        stat["usable"] += int(j.usable.sum())

        aobt_s = _epoch_seconds(j.AOBT_3_flt)
        stand = _norm_str_series(j.STAND_mvt)
        cand_mask = j.usable.to_numpy() & np.isfinite(aobt_s) & stand.notna().to_numpy()
        for mvt_id, track_id, a_s, st in zip(j.MVT_ID_mvt[cand_mask], j.track_id[cand_mask], aobt_s[cand_mask],
                                             stand[cand_mask]):
            trk = tr[(tr.track_id == track_id) & (tr.on_ground == 1)].sort_values("t", kind="mergesort")
            if len(trk) == 0:
                continue
            tx, ty = xy(trk.lat.to_numpy(dtype=float), trk.lon.to_numpy(dtype=float), ap)
            pos = parked_position(trk.t.to_numpy(dtype=float), tx, ty, trk.gs.to_numpy(dtype=float), a_s)
            if pos is not None:
                parked_rows.append({"MVT_ID_mvt": float(mvt_id), "ADEP_mvt": ap, "stand": st, "day": day,
                                   "x": pos[0], "y": pos[1]})
                stat["parked_seen"] += 1

    if parked_rows:
        parked = pd.DataFrame(parked_rows, columns=list(PARKED_SCHEMA)).astype(PARKED_SCHEMA)
    else:
        parked = _empty_frame(PARKED_SCHEMA)
    report = {"by_airport": by_airport, "missing_days": sorted(missing_days)}
    return parked, report


def _ground_by_track(tr: pd.DataFrame, ap: str) -> dict:
    """{track_id: (t, x, y, gs)}: every track's on-ground samples, as metres from `ap`'s reference point, in the
    order adsb_features.segment_tracks left them (it sorts by aircraft and time, so one track's samples are in time
    order and equal times keep the day file's order)."""
    g = tr[tr.on_ground == 1]
    if len(g) == 0:
        return {}
    tid = g.track_id.to_numpy()
    order = np.argsort(tid, kind="stable")
    tid = tid[order]
    t = g.t.to_numpy(dtype=float)[order]
    gs = g.gs.to_numpy(dtype=float)[order]
    x, y = xy(g.lat.to_numpy(dtype=float)[order], g.lon.to_numpy(dtype=float)[order], ap)
    cut = np.flatnonzero(np.r_[True, tid[1:] != tid[:-1], True])
    return {tid[a]: (t[a:b], x[a:b], y[a:b], gs[a:b]) for a, b in zip(cut[:-1], cut[1:])}


def _exit_fields(ground: dict, track_id, t_takeoff: float, mvt_s: float, ref_xy, all_refs: np.ndarray) -> dict:
    """The stand exit of one track before one take-off, as one row sees it (EXIT_COLS): stand_exit() over the
    track's on-ground samples with t < t_takeoff, with the exit and the first ground sample expressed as seconds
    before the row's movement time."""
    g = ground.get(track_id)
    if g is None:
        t = x = y = gs = np.empty(0)
    else:
        before = g[0] < t_takeoff
        t, x, y, gs = g[0][before], g[1][before], g[2][before], g[3][before]
    d = stand_exit(t, x, y, gs, float(t_takeoff), ref_xy, all_refs)
    return {"exit_rel_s": (mvt_s - d["exit_t"]) if np.isfinite(d["exit_t"]) else np.nan,
            "continuous": bool(d["continuous"]), "parked_seen": bool(d["parked_seen"]), "repark": bool(d["repark"]),
            "exit_ok": exit_ok(d), "hold_s": d["hold_s"], "n_ground": d["n_ground"],
            "first_ground_rel_s": (mvt_s - d["first_ground_t"]) if np.isfinite(d["first_ground_t"]) else np.nan,
            "first_ground_dist_m": d["first_ground_dist_m"]}


def _empty_row(mvt_id, ap, day, unmatched, row_type) -> dict:
    return {"MVT_ID_mvt": mvt_id, "ADEP_mvt": ap, "day": day, "unmatched": bool(unmatched), "join_type": "",
            "adsb_icao": "", "t_takeoff": np.nan, "dt_takeoff": np.nan, "row_type": _norm_str(row_type) or "",
            "adsb_type": "", "ref_n": np.nan, **NO_EXIT}


def _raw_row(jr, mvt_s: float, ground: dict, ap: str, day: str, ref, all_refs: np.ndarray) -> dict:
    row = {"MVT_ID_mvt": jr.MVT_ID_mvt, "ADEP_mvt": ap, "day": day, "unmatched": bool(pd.isna(jr.AOBT_3_flt)),
           "join_type": jr.join_type if isinstance(jr.join_type, str) else "",   # an unjoined row carries NaN
           "adsb_icao": jr.adsb_icao if isinstance(jr.adsb_icao, str) else "",
           "t_takeoff": jr.t_takeoff, "dt_takeoff": jr.dt_takeoff,
           "row_type": _norm_str(jr.AIRCRAFT_TYPE_mvt) or "", "adsb_type": _norm_str(jr.adsb_type) or "",
           "ref_n": float(ref[2]) if ref is not None else np.nan}
    if pd.isna(jr.track_id):
        row.update(NO_EXIT)
    else:
        ref_xy = (ref[0], ref[1]) if ref is not None else None
        row.update(_exit_fields(ground, jr.track_id, float(jr.t_takeoff), mvt_s, ref_xy, all_refs))
    return row


def _wants_candidates(row: dict) -> bool:
    """Candidates are gathered for an unmatched row (the stand rule decides later whether it asks) and for the
    1-in-CHECK_SAMPLE_MOD sample of matched callsign joins that stop rule (d) scores."""
    return row["unmatched"] or (row["join_type"] == "callsign" and int(row["MVT_ID_mvt"]) % CHECK_SAMPLE_MOD == 0)


def _agree_share(s: pd.Series) -> float:
    s = s.dropna()
    return float((s == 1.0).mean()) if len(s) else float("nan")


def _mad(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _share(k: int, n: int) -> float:
    return float(k / n) if n else float("nan")


def check_shares(chk: dict) -> dict:
    """Stop rule (d)'s two statistics from one airport's stand_rule_check counts: the share of the counted rows on
    which the stand rule found the row's own take-off, and the share on which it found another one (NaN when no row
    was counted)."""
    return {"stand_check_right_share": _share(chk["right"], chk["n"]),
            "stand_check_wrong_share": _share(chk["wrong"], chk["n"]), "n_stand_check": int(chk["n"])}


def _table_report(table: pd.DataFrame, rows: pd.DataFrame, check: dict) -> dict:
    """Per-airport-month counts, the label-free stop-rule statistics per airport (prereg 'Stop rules for the
    builder', as read by AMEND-A1 item 1, AMEND-A2 item 3 and AMEND-A3) and the non-decisional readouts. Takes
    `rows` (not just `table`) because the residual est - (MVT - AOBT_3) needs mvt_s / aobt_s, which are inputs to
    the table, not part of its schema; `check` is finalise_table's stand_rule_check. Rule (a) is
    `callsign_matched_type_equal_share` (exact equality on callsign joins of matched rows), rule (b)
    `stand_unmatched_type_agree_share` (equal or an alias, on stand joins of unmatched rows), rule (c)
    `matched_exit_mad_s`, rule (d) `stand_check_right_share` / `stand_check_wrong_share`; every type share is over
    the rows where both aircraft types are present."""
    t = table.copy()
    t["month"] = pd.to_datetime(t.day).dt.month
    by_am: dict = {}
    for (ap, mo), g in t.groupby(["ADEP_mvt", "month"], sort=True):
        unmatched, matched = g[g.unmatched], g[~g.unmatched]
        by_am[f"{ap}|{mo}"] = {
            "n_dep": int(len(g)), "n_unmatched": int(len(unmatched)), "usable": int(g.usable.sum()),
            "unmatched_A1": int((unmatched.cls == "A1").sum()), "unmatched_A2": int((unmatched.cls == "A2").sum()),
            "matched_A1": int((matched.cls == "A1").sum()), "matched_A2": int((matched.cls == "A2").sum()),
        }

    mvt_by_id = rows.set_index("MVT_ID_mvt").mvt_s
    aobt_by_id = rows.set_index("MVT_ID_mvt").aobt_s
    stop_rules: dict = {}
    readouts: dict = {}
    for ap, g in t.groupby("ADEP_mvt", sort=True):
        matched, unmatched = g[~g.unmatched], g[g.unmatched]
        cs_matched = matched[matched.join_type == "callsign"]
        ti_unmatched = unmatched[unmatched.join_type == "time"]
        st_unmatched = unmatched[unmatched.join_type == "stand"]
        mm = matched[matched.cls.isin(["A1", "A2"])]
        mvt = mm.MVT_ID_mvt.map(mvt_by_id).to_numpy(dtype=float)
        aobt = mm.MVT_ID_mvt.map(aobt_by_id).to_numpy(dtype=float)
        resid = mm.exit_rel_s.to_numpy(dtype=float) - (mvt - aobt)
        chk = check.get(ap, {"n": 0, "right": 0, "wrong": 0, "ambiguous": 0, "none": 0})
        stop_rules[ap] = {
            "callsign_matched_type_equal_share": _agree_share(cs_matched.type_equal),
            "callsign_matched_type_agree_share": _agree_share(cs_matched.type_agree),
            "n_callsign_matched_typed": int(cs_matched.type_equal.notna().sum()),
            "stand_unmatched_type_agree_share": _agree_share(st_unmatched.type_agree),
            "stand_unmatched_type_equal_share": _agree_share(st_unmatched.type_equal),
            "stand_unmatched_type_missing_share": _share(int(st_unmatched.type_agree.isna().sum()), len(st_unmatched)),
            "n_stand_unmatched": int(len(st_unmatched)),
            "n_stand_unmatched_typed": int(st_unmatched.type_agree.notna().sum()),
            "time_unmatched_type_agree_share": _agree_share(ti_unmatched.type_agree),
            "time_unmatched_type_equal_share": _agree_share(ti_unmatched.type_equal),
            "time_unmatched_type_missing_share": _share(int(ti_unmatched.type_agree.isna().sum()), len(ti_unmatched)),
            "n_time_unmatched": int(len(ti_unmatched)),
            "n_time_unmatched_typed": int(ti_unmatched.type_agree.notna().sum()),
            "matched_exit_mad_s": _mad(resid), "n_matched_classed": int(len(mm)), **check_shares(chk),
        }
        readouts[ap] = {"matched_agree_600s_by_band": agreement_by_band(mm.exit_rel_s.to_numpy(dtype=float), resid),
                        "stand_rule_check": chk}
    return {"by_airport_month": by_am, "stop_rules": stop_rules, "readouts": readouts,
            "near_duplicate_takeoffs": near_duplicate_takeoffs(table)}


def build_raw(rows: pd.DataFrame, adsb_dirs, parked: pd.DataFrame, days=None, airports=None,
              log=print, progress=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The per-day half of pass 2: refs = stand_references(parked); for the (day, airport) pairs asked for, returns
    (raw, cands, {"missing_days": [...]}).

    raw: one row (RAW_COLS) per departure row of `rows`: its shipped join (join_type "" when unjoined, the joined
    aircraft's address, take-off time and type), its own type, its stand's reference count (NaN without a
    reference) and its stand exit (EXIT_COLS: exit_rel_s = mvt_s - exit_t, the three guards, exit_ok, hold_s,
    n_ground, first_ground_rel_s, first_ground_dist_m). A flight's own parked position is NOT removed from its
    stand's reference before scoring it: unmatched rows never contribute to `parked` in the first place
    (build_parked only sees matched, usable, stand-known flights), and that absence is exactly the population this
    table exists to estimate -- there is no leave-one-out step here, by design. Rows of a day with no ADS-B file
    anywhere are present with join_type "" and NaN / False fields, never dropped. Sorted by MVT_ID_mvt; a pair list
    without any departure gives zero rows with every column.

    cands: for every row with a stand reference that is unmatched, or a callsign join of a matched departure whose
    MVT_ID_mvt is divisible by CHECK_SAMPLE_MOD, one row (CAND_COLS) per take-off of that day's view within
    CAND_BRACKET_S of its movement time (both ends included): the take-off's aircraft, time, offset and type, `own`
    (it is the take-off the shipped join gave the row) and that track's exit from the ROW's stand. Sorted by
    (MVT_ID_mvt, t_takeoff, adsb_icao).

    `progress(i, n, day, airport)` is called as pair i of n starts."""
    refs = stand_references(parked)
    ref_lookup = {(r.ADEP_mvt, r.stand): (r.x, r.y, r.n) for r in refs.itertuples()}
    refs_by_airport: dict = {}
    for r in refs.itertuples():
        refs_by_airport.setdefault(r.ADEP_mvt, []).append((r.x, r.y))

    pairs = rows[["day", "ADEP_mvt"]].drop_duplicates()
    if days is not None:
        pairs = pairs[pairs.day.isin(list(days))]
    if airports is not None:
        pairs = pairs[pairs.ADEP_mvt.isin(list(airports))]
    pairs = pairs.sort_values(["day", "ADEP_mvt"], kind="mergesort")

    out_rows = []
    cand_rows = []
    missing_days: list = []
    seen_missing: set = set()
    n_pairs = len(pairs)
    for i, (day, ap) in enumerate(pairs.itertuples(index=False), 1):
        if progress is not None:
            progress(i, n_pairs, day, ap)
        sub = rows[(rows.day == day) & (rows.ADEP_mvt == ap)]
        all_refs = np.array(refs_by_airport.get(ap, []), dtype=float)
        all_refs = all_refs.reshape(-1, 2) if all_refs.size else np.empty((0, 2))
        adsb = read_day(adsb_dirs, day, ap)
        if adsb is None:
            if day not in seen_missing:
                missing_days.append(day)
                seen_missing.add(day)
            log(f"uad_build table: no ADS-B file for {day} ({ap}); {len(sub)} departure(s) uncovered")
            for r in sub.itertuples():
                out_rows.append(_empty_row(r.MVT_ID_mvt, ap, day, r.unmatched, r.AIRCRAFT_TYPE_mvt))
            continue
        j, tr, ev, last_type = _join_parts(sub, adsb)
        ground = _ground_by_track(tr, ap)
        ev = ev.sort_values(["t_takeoff", "icao"], kind="mergesort")
        ev_t = ev.t_takeoff.to_numpy(dtype=float)
        ev_track = ev.track_id.to_numpy()
        ev_icao = ev.icao.to_numpy(dtype=object)
        mvt = _epoch_seconds(j.MVT_TIME_UTC_mvt)
        stand = _norm_str_series(j.STAND_mvt)
        for jr, st, m in zip(j.itertuples(), stand, mvt):
            ref = ref_lookup.get((ap, st))
            row = _raw_row(jr, float(m), ground, ap, day, ref, all_refs)
            out_rows.append(row)
            if ref is None or not _wants_candidates(row):
                continue
            first = np.searchsorted(ev_t, m - CAND_BRACKET_S)
            last = np.searchsorted(ev_t, m + CAND_BRACKET_S, side="right")
            for k in range(first, last):
                own = row["join_type"] != "" and ev_icao[k] == row["adsb_icao"] and ev_t[k] == row["t_takeoff"]
                cand_rows.append({"MVT_ID_mvt": row["MVT_ID_mvt"], "ADEP_mvt": ap, "own": bool(own),
                                  "adsb_icao": ev_icao[k], "t_takeoff": ev_t[k], "dt_takeoff": ev_t[k] - float(m),
                                  "adsb_type": _norm_str(last_type.get(ev_track[k])) or "",
                                  **_exit_fields(ground, ev_track[k], float(ev_t[k]), float(m), (ref[0], ref[1]),
                                                 all_refs)})

    raw = pd.DataFrame(out_rows, columns=RAW_COLS).astype(RAW_SCHEMA) if out_rows else _empty_frame(RAW_SCHEMA)
    raw = raw.sort_values("MVT_ID_mvt", kind="mergesort").reset_index(drop=True)
    cands = (pd.DataFrame(cand_rows, columns=CAND_COLS).astype(CAND_SCHEMA) if cand_rows
             else _empty_frame(CAND_SCHEMA))
    cands = cands.sort_values(["MVT_ID_mvt", "t_takeoff", "adsb_icao"], kind="mergesort").reset_index(drop=True)
    return raw, cands, {"missing_days": sorted(missing_days)}


def build_table(rows: pd.DataFrame, adsb_dirs, parked: pd.DataFrame, days=None, airports=None,
                log=print, progress=None, extra_identity=None) -> tuple[pd.DataFrame, dict]:
    """Pass 2 in one call: finalise_table(*build_raw(...), extra_identity). One output row per departure row of
    `rows` (TABLE_COLS, sorted by MVT_ID_mvt). The report holds by_airport_month, stop_rules, readouts,
    near_duplicate_takeoffs, missing_days, takeoff_claims_dropped, aliases, offset_windows and stand_joins."""
    raw, cands, bits = build_raw(rows, adsb_dirs, parked, days=days, airports=airports, log=log, progress=progress)
    table, info = finalise_table(raw, cands, extra_identity)
    report = _table_report(table, rows, info.pop("stand_rule_check"))
    report.update(bits)
    report.update(info)
    return table, report


# ---------------------------------------------------------------- CLI

def _write_output_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Writes with an explicit arrow schema (float64 / bool / int64 / string by pandas dtype) so the file is
    byte-identical for logically-identical data regardless of how the DataFrame was assembled. Without this,
    pandas' Arrow-backed string dtype can pick 'string' vs 'large_string' depending on whether the frame passed
    through pd.concat (e.g. a direct build_table call vs the --jobs>1 CLI path reassembling per-day chunks), which
    changes the file's bytes without changing any value in it."""
    fields = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            t = pa.bool_()
        elif pd.api.types.is_integer_dtype(s):
            t = pa.int64()
        elif pd.api.types.is_float_dtype(s):
            t = pa.float64()
        else:
            t = pa.string()
        fields.append(pa.field(col, t))
    table = pa.Table.from_pandas(df, schema=pa.schema(fields), preserve_index=False)
    pq.write_table(table, path)


def _check_out_dir(path) -> pathlib.Path:
    """`path` must not exist, or must exist empty: this CLI never overwrites an existing output."""
    p = pathlib.Path(path).resolve()
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f"{p} exists and is not empty")
    return p


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _declared_sources(adsb_dirs) -> dict:
    out = {}
    for d in adsb_dirs:
        p = pathlib.Path(d)
        try:
            out[str(p)] = R12.source_contains(p / "MANIFEST.json")
        except (FileNotFoundError, json.JSONDecodeError):
            out[str(p)] = None
    return out


def _read_rows_for_cli(paths) -> tuple[pd.DataFrame, dict]:
    resolved = [pathlib.Path(p).resolve() for p in paths]
    shas = {str(p): _sha256_file(p) for p in resolved}
    rows = read_rows(resolved)
    return rows, shas


def _report_base(adsb_dirs) -> dict:
    return {"params": params(), "params_hash": params_hash(), "sources": _declared_sources(adsb_dirs),
            "code_sha256": {rel: _sha256_file(ROOT / rel) for rel in CODE_FILES}}


def _day_chunks(days: list, jobs: int) -> list:
    n = min(jobs, len(days))
    return [[str(x) for x in c] for c in np.array_split(np.array(days, dtype=object), n) if len(c)]


PROGRESS_EVERY = 50


def _heartbeat(label: str):
    """A progress callback that prints the first, every PROGRESS_EVERY-th and the last pair (flushed: a detached
    run's log must show that the build is alive)."""
    def _p(i, n, day, ap):
        if i == 1 or i == n or i % PROGRESS_EVERY == 0:
            print(f"uad_build {label}: pair {i}/{n} ({day} {ap})", flush=True)
    return _p


def _parked_chunk(args):
    rows, adsb_dirs, days, airports = args
    return build_parked(rows, adsb_dirs, days=days, airports=airports, log=lambda *a, **k: None,
                        progress=_heartbeat(f"parked {days[0]}..{days[-1]}"))


def _table_chunk(args):
    rows, adsb_dirs, parked, days, airports = args
    return build_raw(rows, adsb_dirs, parked, days=days, airports=airports, log=lambda *a, **k: None,
                     progress=_heartbeat(f"table {days[0]}..{days[-1]}"))


def _merge_parked_reports(reports: list) -> dict:
    by_airport: dict = {}
    missing: set = set()
    for r in reports:
        for ap, stat in r["by_airport"].items():
            acc = by_airport.setdefault(ap, {"departures": 0, "joined_by_type": {}, "usable": 0, "parked_seen": 0})
            acc["departures"] += stat["departures"]
            acc["usable"] += stat["usable"]
            acc["parked_seen"] += stat["parked_seen"]
            for jt, cnt in stat["joined_by_type"].items():
                acc["joined_by_type"][jt] = acc["joined_by_type"].get(jt, 0) + cnt
        missing.update(r["missing_days"])
    return {"by_airport": by_airport, "missing_days": sorted(missing)}


def _pool_parked(parked_dirs) -> pd.DataFrame:
    frames = []
    seen: set = set()
    for d in parked_dirs:
        p = pathlib.Path(d).resolve() / "parked.parquet"
        pf = pd.read_parquet(p)
        ids = set(pf.MVT_ID_mvt.tolist())
        overlap = ids & seen
        if overlap:
            raise ValueError(f"{p}: {len(overlap)} MVT_ID_mvt already supplied by another --parked directory")
        seen |= ids
        frames.append(pf)
    if not frames:
        return _empty_frame(PARKED_SCHEMA)
    return pd.concat(frames, ignore_index=True).astype(PARKED_SCHEMA)


def _run_parked(a) -> int:
    out = _check_out_dir(a.out)
    rows, shas = _read_rows_for_cli(a.rows)
    dirs = [str(pathlib.Path(d).resolve()) for d in a.adsb_dir]
    days = sorted(set(a.days)) if a.days else sorted(rows.day.unique())
    jobs = max(1, a.jobs)

    if jobs == 1 or len(days) <= 1:
        parked, report = build_parked(rows, dirs, days=days, airports=a.airports, progress=_heartbeat("parked"))
    else:
        chunks = _day_chunks(days, jobs)
        ctx = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as ex:
            parts = list(ex.map(_parked_chunk, [(rows[rows.day.isin(c)], dirs, c, a.airports) for c in chunks]))
        parked = (pd.concat([p for p, _ in parts], ignore_index=True).astype(PARKED_SCHEMA) if parts
                 else _empty_frame(PARKED_SCHEMA))
        report = _merge_parked_reports([r for _, r in parts])

    if len(parked):
        parked = parked.sort_values(["day", "ADEP_mvt", "stand", "MVT_ID_mvt"],
                                    kind="mergesort").reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    _write_output_parquet(parked, out / "parked.parquet")
    full_report = _report_base(dirs)
    full_report.update({"inputs": shas, "days": days, **report})
    (out / "report.json").write_text(json.dumps(full_report, indent=2, sort_keys=True, default=str))
    return 0


def _alias_sources(table_dirs, own_ids: set) -> tuple[pd.DataFrame, dict]:
    """The identity rows of other tables (--alias-from) and {table path: sha256}. A directory must hold a
    report.json written with the current params_hash() and a table.parquet that shares no MVT_ID_mvt with this
    build's rows or with another --alias-from table (its joins would be counted twice): ValueError otherwise."""
    frames = []
    shas: dict = {}
    seen = set(own_ids)
    for d in table_dirs:
        base = pathlib.Path(d).resolve()
        report_path, table_path = base / "report.json", base / "table.parquet"
        if not report_path.exists():
            raise ValueError(f"{base}: no report.json (not a table directory)")
        built_with = json.loads(report_path.read_text()).get("params_hash")
        if built_with != params_hash():
            raise ValueError(f"{base}: built with parameters {built_with!r}, not {params_hash()!r}")
        other = pd.read_parquet(table_path, columns=IDENTITY_SOURCE_COLS)
        ids = set(other.MVT_ID_mvt.tolist())
        overlap = ids & seen
        if overlap:
            raise ValueError(f"{table_path}: {len(overlap)} MVT_ID_mvt already in this build's rows or in another "
                             f"--alias-from table")
        seen |= ids
        frames.append(identity_rows(other))
        shas[str(table_path)] = _sha256_file(table_path)
    if not frames:
        return pd.DataFrame(columns=IDENTITY_COLS), shas
    return pd.concat(frames, ignore_index=True), shas


def _run_table(a) -> int:
    out = _check_out_dir(a.out)
    rows, shas = _read_rows_for_cli(a.rows)
    dirs = [str(pathlib.Path(d).resolve()) for d in a.adsb_dir]
    parked = _pool_parked(a.parked)
    extra_identity, alias_shas = _alias_sources(a.alias_from or [], set(rows.MVT_ID_mvt.tolist()))
    days = sorted(set(a.days)) if a.days else sorted(rows.day.unique())
    jobs = max(1, a.jobs)

    if jobs == 1 or len(days) <= 1:
        raw, cands, bits = build_raw(rows, dirs, parked, days=days, airports=a.airports, progress=_heartbeat("table"))
        missing_days = bits["missing_days"]
    else:
        chunks = _day_chunks(days, jobs)
        ctx = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as ex:
            parts = list(ex.map(_table_chunk, [(rows[rows.day.isin(c)], dirs, parked, c, a.airports) for c in chunks]))
        raw = pd.concat([r for r, _, _ in parts], ignore_index=True)
        cands = pd.concat([c for _, c, _ in parts], ignore_index=True)
        cands = cands.sort_values(["MVT_ID_mvt", "t_takeoff", "adsb_icao"], kind="mergesort").reset_index(drop=True)
        missing_days = sorted({d for _, _, b in parts for d in b["missing_days"]})

    # what depends on the whole table (a take-off claimed from two days, the aliases, the offset windows, the stand
    # joins, usable, the class, every share and median of the report) is settled once, here, after the day-chunks
    # are back: none of it is additive over chunks, and doing it per chunk would not reproduce the single-process
    # answer.
    table, info = finalise_table(raw, cands, extra_identity)
    report = _table_report(table, rows, info.pop("stand_rule_check"))
    report["missing_days"] = missing_days
    report.update(info)

    out.mkdir(parents=True, exist_ok=True)
    _write_output_parquet(table, out / "table.parquet")
    _write_output_parquet(cands, out / "cands.parquet")
    _write_output_parquet(stand_references(parked), out / "refs.parquet")
    full_report = _report_base(dirs)
    full_report.update({"inputs": shas, "alias_from": alias_shas, "days": days, **report})
    (out / "report.json").write_text(json.dumps(full_report, indent=2, sort_keys=True, default=str))
    return 0


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="uad_build", description="Label-blind stand-exit table for unmatched departures.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pk = sub.add_parser("parked", help="pass 1: parked positions of matched, usable departures")
    pk.add_argument("--rows", nargs="+", required=True)
    pk.add_argument("--adsb-dir", dest="adsb_dir", nargs="+", required=True)
    pk.add_argument("--out", required=True)
    pk.add_argument("--airports", nargs="+", default=None)
    pk.add_argument("--days", nargs="+", default=None)
    pk.add_argument("--jobs", type=int, default=1)

    tb = sub.add_parser("table", help="pass 2: the stand-exit table for every departure")
    tb.add_argument("--rows", nargs="+", required=True)
    tb.add_argument("--adsb-dir", dest="adsb_dir", nargs="+", required=True)
    tb.add_argument("--parked", nargs="+", required=True)
    tb.add_argument("--alias-from", dest="alias_from", nargs="+", default=None,
                    help="table directories whose joins of known aircraft are pooled for the type aliases")
    tb.add_argument("--out", required=True)
    tb.add_argument("--airports", nargs="+", default=None)
    tb.add_argument("--days", nargs="+", default=None)
    tb.add_argument("--jobs", type=int, default=1)

    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        if a.cmd == "parked":
            return _run_parked(a)
        if a.cmd == "table":
            return _run_table(a)
        raise ValueError(f"unknown command: {a.cmd}")
    except (FileExistsError, ValueError, FileNotFoundError, PermissionError) as e:
        print(f"uad_build: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
