"""scripts/adsb_storm.py -- arm E4 (plans/PREREG_adsb_eham_storm_2026_09_10.md): an ADS-B off-block
measurement for EHAM's January 2026 storm days, 2026-01-02 ... 2026-01-09.

    python scripts/adsb_storm.py extract --day 2026-01-05 --probe      # 60 MB, confirms the 2026 format
    python scripts/adsb_storm.py extract --day 2026-01-02              # one day, ~3.1 GB streamed, deleted after
    python scripts/adsb_storm.py census                                # C3 alone, on the 2025-01-09 census data
    python scripts/adsb_storm.py gate                                  # C0-C3, the verdict, the per-row table
    python scripts/adsb_storm.py ship --base submissions/<team>_vN.parquet --version M \
        --detail submissions/<team>_vK.unm_rows.parquet --licence-doc <repo file naming ODbL + adsb.lol> --eligibility-confirmed

Design, fixed by the prereg and not tunable here: stand centroids are the position of matched EHAM departures at
their AOBT_3, per STAND_mvt, leave-one-flight-out, a stand needing >= 2 OTHER flights; the estimate is the last
sample within 100 m of the row's own stand centroid before take-off; the bias correction is the median
(sensor - AOBT_3) of matched rows on the OTHER days (leave-one-day-out); matched rows join by CALLSIGN_flt
(reports/ADSB_GATE.md: FLIGHT_mvt is the IATA number at EHAM and joins 2%), unmatched rows by runway +
take-off second (MVT_TIME_UTC_mvt <-> the first ADS-B sample with ground speed > 80 kt on that runway).

No 2026 label is ever read: the storm flights come from FLIGHT_COLUMNS only (no BLOCK_TIME_UTC_mvt, no TAXITIME_SEC_mvt),
and the label-reading census refuses any day that is not a 2025 day.

Clauses (locked): C0 join ambiguity <= 5% of rows with a candidate; C1 bias-corrected RMSE(offblock_hat - AOBT_3)
<= 250 s at coverage >= 50% on the matched storm rows; C2 per-held-out-day RMSE <= 300 s on >= 6 of 8 days;
C3 on the 2025-01-09 census (data/adsb/) the gated taxi estimate against the TRUE label beats the model RMSE
ADSB_GATE.md reports for EHAM (173 s). Verdict: WORKING iff C0-C3; NOT WORKING iff C1 fails; else INCONCLUSIVE.

Mechanical choices the prereg leaves open are collected in JUDGEMENT_CALLS below and written into the report.

Eligibility (binding, from ADSB_GATE.md): no submission may depend on this arm until the organisers' licence
answer is in writing; `ship` refuses to run without --eligibility-confirmed and prints the rule.

Arm E4b (plans/PREREG_adsb_eham_v2_2026_09_10.md), composed on the same harness; E4's own mode is untouched so its
registration record stays reproducible:

    python scripts/adsb_storm.py extract --day 2025-12-28 --probe      # format + EHAM samples on a validation day
    python scripts/adsb_storm.py extract --day 2025-12-27              # the two registered validation days
    python scripts/adsb_storm.py calibrate                             # ONE bias constant, 2025-01-09 census only
    python scripts/adsb_storm.py validate                              # V1-V3 on 2025-12-27/28, the verdict (write-once)
    python scripts/adsb_storm.py gate --e4b                            # 2026 path with the constant (needs WORKING)
    python scripts/adsb_storm.py ship --e4b ... --eligibility-confirmed

E4b differs from E4 in ONE place: the bias is ONE constant, median(sensor - BLOCK) on the census day's matched gated
rows, referenced to the label; centroids still sit at AOBT_3 (location only). Labels are read through `labels_2025`
only, which refuses any row whose take-off is not in 2025. A full 2026 extract refuses unless the E4b validation
verdict on disk is WORKING (the prereg: "Only if WORKING" does the 2026 ingest run).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_submission as bs  # noqa: E402

_spec = importlib.util.spec_from_file_location("adsb_extract", ROOT / "scripts" / "adsb_extract.py")
ax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ax)

PREREG = "plans/PREREG_adsb_eham_storm_2026_09_10.md"
ARM = f"E4 ADS-B off-block on EHAM unmatched storm rows ({PREREG})"
AIRPORT = "EHAM"
DAYS = tuple(f"2026-01-0{d}" for d in range(2, 10))                  # 2026-01-02 ... 2026-01-09, registered
CENSUS_DAY = "2025-01-09"
ADSB_DIR = ROOT / "data" / "adsb"
REPORT = ROOT / "reports" / "adsb_storm.json"
ESTIMATES = ADSB_DIR / "adsb_storm_estimates.parquet"

# ---- locked by the prereg -------------------------------------------------------------------------------------
RADIUS_M = 100.0                 # the stand gate
MIN_OTHER_FLIGHTS = 2            # a stand needs >= 2 OTHER flights for a leave-one-flight-out centroid
TAKEOFF_GS_KT = 80.0             # the take-off second: first sample with ground speed > 80 kt on the runway
C0_MAX_SHARE = 0.05
C1_RMSE_S = 250.0
C1_MIN_COVERAGE = 0.50
C2_RMSE_S = 300.0
C2_MIN_DAYS = 6
C2_N_DAYS = 8
MODEL_RMSE_EHAM_S = 173.2        # ADSB_GATE.md, "What it is worth": the model's EHAM RMSE the census beat

# ---- mechanical choices the prereg leaves open (reported, not tuned) -------------------------------------------
FIX_TOL_S = 600.0                # "position at AOBT_3": nearest on-ground sample within this many seconds (see JUDGEMENT_CALLS)
TRACK_WINDOW_S = 3 * 3600.0      # a matched flight's track: same callsign, samples in [MVT - 3 h, MVT + 60 s]
TRACK_SLACK_S = 60.0
TRACK_GAP_S = 1800.0             # a gap longer than this splits an aircraft's samples into two tracks
TAKEOFF_TOL_S = 60.0             # C0 candidate window: |t_80kt - (MVT + median offset on matched rows)| <= this
RUNWAY_LATERAL_M = 150.0         # C0: a take-off event belongs to a runway if within this of the learned line
RUNWAY_HEADING_COS = 0.7         # ... and heading within ~45 deg of the learned direction
RUNWAY_ALONG_M = 3_800.0         # ... and within this along the line of the learned point: EHAM's longest runway (18R/36L)
MIN_RUNWAY_EVENTS = 5            # a runway's geometry is learned from >= this many matched take-off events
DIRECTION_TOL_S = 30.0           # the heading at a take-off event comes from the neighbouring sample within this
JUDGEMENT_CALLS = [
    "a storm day is the UTC calendar date of MVT_TIME_UTC_mvt (the take-off), for flights and for the leave-one-day-out split alike",
    f"position at AOBT_3 = the nearest ON-GROUND sample within {FIX_TOL_S:.0f} s of AOBT_3 (an airborne fix cannot be a stand). "
    "The tolerance was set by the first implementer on the 2025-01-09 census, never on 2026 data. Its original justification "
    "('at 120 s the coverage was 16% against the census's 66%') was measured under a defect that gave a centroid only to flights "
    "with a fix of their own, so it no longer supports the value; the value is KEPT because re-choosing it after seeing census "
    "numbers would tune on C3's own data. It now only decides which flights contribute a fix to a stand's centroid; off-stand "
    "fixes are damped by the median centroid and by the 100 m gate itself",
    "centroid = coordinate-wise median of the fixes at the row's STAND_mvt that are not the row's own: neither its MVT_ID_mvt's fix "
    "nor any fix taken on its own track (two rows joined to one track must not feed each other). The rule is the same for a matched "
    "row with a fix, a matched row without one, and an unmatched row (which never has one); >= 2 such fixes are required",
    f"a matched flight's track = samples with callsign == CALLSIGN_flt in [MVT - {TRACK_WINDOW_S/3600:.0f} h, MVT + {TRACK_SLACK_S:.0f} s]; "
    "when several aircraft carry the callsign, the one with most samples in the window",
    f"tracks are split per aircraft on a callsign change or a gap > {TRACK_GAP_S:.0f} s; samples with no callsign are first forward- "
    "then backward-filled within the aircraft (icao)",
    "the gate uses samples with t <= MVT (before take-off); the last one inside the radius is the raw off-block",
    "every row, matched or unmatched, is bias-corrected with the median (sensor - AOBT_3) of matched gated rows on the OTHER days: "
    "the estimator that ships is the one C1 validates",
    "C0 runway geometry is learned from the matched flights' own take-off events (median point + mean heading per RUNWAY_mvt, "
    f">= {MIN_RUNWAY_EVENTS} events); an event is 'on the runway' within {RUNWAY_LATERAL_M:.0f} m laterally, cos(heading) >= "
    f"{RUNWAY_HEADING_COS} and within {RUNWAY_ALONG_M:.0f} m along the line of the learned point (two 80-kt points on one runway "
    "cannot lie further apart than EHAM's longest runway is long; without it the infinite centre-line captures points km away)",
    f"C0 take-off second: the matched rows calibrate the median (t_80kt - MVT); a candidate lies within +-{TAKEOFF_TOL_S:.0f} s of MVT + that offset; "
    "tracks already joined to a matched flight are not candidates",
    "C0 'ambiguous or unmatched' = rows with >= 2 candidates, plus rows whose single candidate is also another row's single candidate; "
    "the denominator is rows with >= 1 candidate; the no-candidate share is reported separately",
    "C2: a held-out day with no gated matched row has no RMSE and counts as NOT within 300 s",
    "C3 runs the SAME pipeline on 2025-01-09 (centroids and bias at AOBT_3; one day, so the bias is that day's own in-sample median, "
    "as in ADSB_GATE -- optimistic by construction) and scores taxi_hat = MVT - offblock_hat against TAXITIME_SEC_mvt; a "
    "BLOCK_TIME-calibrated variant is reported as a diagnostic only. Labels are read on 2025 rows only",
    f"C3's model RMSE is the constant ADSB_GATE.md reports for EHAM ({MODEL_RMSE_EHAM_S} s); no per-row model prediction exists for that day",
    "ship: the prereg's 'rebuild guard on the base' is the rome_bandfloor pattern -- the base's value on every EHAM unmatched storm "
    "row must equal the `after` of the detail file written with the version that last set those rows (v9's unm_rows for v10); "
    "the licence clause is checked as a repo file naming ODbL and adsb.lol",
]
# ---- arm E4b, locked by plans/PREREG_adsb_eham_v2_2026_09_10.md ------------------------------------------------
PREREG_E4B = "plans/PREREG_adsb_eham_v2_2026_09_10.md"
ARM_E4B = f"E4b EHAM ADS-B off-block, label-referenced bias, validated on unseen congested 2025 days ({PREREG_E4B})"
VALIDATION_DAYS = ("2025-12-27", "2025-12-28")       # registered; never read by any earlier ADS-B work
VALIDATION_TRUTH = ROOT / "data" / "raw" / "training_2025-12-01_2026-01-01.parquet"
S1C_PATH = ROOT / "data" / "cache_stand" / "unm_congestion_lomo.parquet"
BIAS_E4B = ROOT / "reports" / "adsb_e4b_bias.json"
VALIDATE_REPORT = ROOT / "reports" / "adsb_e4b_validate.json"
VALIDATION_ROWS = ADSB_DIR / "adsb_e4b_validation_rows.parquet"
REPORT_E4B = ROOT / "reports" / "adsb_e4b_storm.json"
ESTIMATES_E4B = ADSB_DIR / "adsb_e4b_storm_estimates.parquet"
LABEL_COLUMNS = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")
V1_RMSE_S = 200.0
V1_MIN_COVERAGE = 0.50
V2_MIN_ROWS = 15
V3_MAX_SHARE = 0.05
TAXI_FLOOR_S = 1.0
JUDGEMENT_CALLS_E4B = [
    "the two validation days run through ONE measure() call, as the eight 2026 days do: centroids are the AOBT_3 fixes of both "
    "days' matched rows (the prereg's 'same day(s)'; 'identical on the validation days and on 2026'), leave-one-flight-out; the "
    "runway geometry and the take-off offset of the unmatched join are learned from the same days' matched rows",
    "a matched row = AOBT_3_flt present (E4's rule; on 2025-12-27/28 it coincides with CALLSIGN_flt present, 1,193 rows)",
    "the bias constant = median over the census day's matched rows where the gate fires of (last in-gate sample - BLOCK_TIME); "
    "IQR = q75 - q25 by numpy's linear interpolation; the census truth is data/adsb/pilot_truth.parquet (E4's census loader; "
    "equal to the raw training file on all 602 EHAM rows of the day, every column checked 2026-09-10)",
    "V1: coverage = gated matched rows / ALL matched rows of the two days (rows with no callsign track count as not gated); "
    "pooled over both days; per day reported, not decisional",
    "V2: rows = unmatched rows of the two days where the gate fires and the join is unambiguous (a gated unmatched row is "
    "unambiguous by construction); taxi_hat = MVT - (sensor - bias) floored at 1 s, NOT rounded (ship rounds); S1C = the "
    "3-seed mean column `S1C` of unm_congestion_lomo.parquet, whose metadata must be stamped convention=taxi_time and whose `y` "
    "must equal TAXITIME_SEC_mvt on every scored row (BC-2 tripwire); the 15-row minimum counts exactly V2's scored rows",
    "V2 status: fewer than 15 rows -> INCONCLUSIVE (pass False); >= 15 rows and not strictly lower RMSE -> FAIL (pass False); "
    "either way the arm's verdict is INCONCLUSIVE when V1 passes, by the prereg's mapping",
    "V3 = E4's C0 rule: 'ambiguous or failed' = rows with >= 2 candidates plus rows whose single candidate is another row's too; "
    "denominator = unmatched rows with >= 1 candidate; no-candidate rows reported separately; zero rows with a candidate fails V3",
    "a day is the UTC date of MVT_TIME_UTC_mvt; tracks come from the two validation-day archives only, so a 2025-12-27 flight "
    "whose off-block fell on 2025-12-26 has no stand sample and counts as not gated (a few after-midnight rows at most)",
    "calibrate refuses to replace a different constant; validate writes once and refuses to overwrite ('never revised')",
    "a full 2026 extract refuses unless the E4b validation verdict on disk is WORKING; probes are allowed (they write nothing)",
    "gate --e4b: the 2026 in-file diagnostic (sensor vs AOBT_3) and the 2026 join ambiguity are reported only; the report's "
    "verdict is the validation verdict, which ship --e4b re-reads from the validation file",
]

#: the prereg's ship rule, recorded in the ship meta: the expected board improvement and the transfer verdict rule
EXPECTED_PRICE = {"conservative_event_excluded_board_mse": 800.0, "full_board_mse": 1800.0,
                  "source": "reports/PIPELINE_TO_245_2026_09_10.md Stage 2 (research lead)"}
TRANSFER_RULE = "board MSE delta vs the base >= 0 = NOT TRANSFERRED"

ELIGIBILITY_RULE = (
    "ELIGIBILITY (PREREG_adsb_eham_storm_2026_09_10.md, binding, from reports/ADSB_GATE.md): adsb.lol publishes under "
    "ODbL-1.0 and the challenge permits documented open datasets, but the licence question with the organisers is NOT "
    "closed. Measurement may proceed; NO SUBMISSION MAY DEPEND ON THIS ARM until the owner has the organisers' answer in "
    "writing. The derived table's licence (ODbL share-alike) is documented in the repo before any ship."
)

ESTIMATE_COLUMNS = ["MVT_ID_mvt", "matched", "day", "stand", "runway", "offblock_hat", "taxi_hat", "gate_fired",
                    "join_ambiguous", "n_candidates", "offblock_raw_s", "bias_s", "track_id"]
DETAIL_COLUMNS = ["MVT_ID_mvt", "day", "stand", "runway", "offblock_hat", "taxi_hat", "before", "after"]

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================================
# primitives
# =============================================================================================

def to_epoch_s(ts: pd.Series) -> np.ndarray:
    """tz-aware stamps -> float seconds since the epoch (NaN for NaT). ADSB_GATE landmine: `.astype('int64')` on
    timestamp[us] yields MICROseconds and made every join miss; every finite value is asserted into the 1.7e9 band."""
    out = (pd.to_datetime(ts, utc=True) - EPOCH).dt.total_seconds().to_numpy(dtype="float64")
    fin = out[np.isfinite(out)]
    if len(fin) and not ((fin > 1.6e9) & (fin < 1.9e9)).all():
        raise ValueError("epoch seconds outside the 2020-2030 band: the stamps are not seconds")
    return out


def xy_m(lat, lon, lat0: float, lon0: float) -> tuple:
    """Equirectangular metres east / north of (lat0, lon0); exact enough at the 100 m scale of an apron."""
    lat, lon = np.asarray(lat, dtype="float64"), np.asarray(lon, dtype="float64")
    return (lon - lon0) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat0)), (lat - lat0) * M_PER_DEG_LAT


def rmse(err) -> float:
    err = np.asarray(err, dtype="float64")
    return float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan")


# =============================================================================================
# flights (the ranking file) and tracks (the extracted ADS-B samples)
# =============================================================================================

FLIGHT_COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt", "STAND_mvt", "RUNWAY_mvt",
                  "CALLSIGN_flt"]


def flights_frame(df: pd.DataFrame, days, airport: str = AIRPORT, truth_col: str = "AOBT_3_flt") -> pd.DataFrame:
    """The departures of `airport` on `days`: MVT_ID_mvt, day, stand, runway, callsign, mvt_s, ref_s (the off-block
    reference, AOBT_3 by the prereg), matched (= reference present). Rows keep the input order."""
    for c in ("MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt", "CALLSIGN_flt", truth_col):
        if c not in df.columns:
            raise ValueError(f"flights need column {c!r}")
    d = df
    if "PHASE_mvt" in d.columns:
        d = d[d.PHASE_mvt == "DEP"]
    d = d[d.ADEP_mvt.astype(str) == airport]
    day = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    d = d[day.isin(list(days))]
    day = day.loc[d.index]
    mvt_s = to_epoch_s(d.MVT_TIME_UTC_mvt)
    ref_s = to_epoch_s(d[truth_col])
    cs = d.CALLSIGN_flt.astype("string").str.strip().str.upper()
    out = pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt.to_numpy(dtype="float64"), "day": day.to_numpy(),
                        "stand": d.STAND_mvt.astype("string").to_numpy(dtype=object),
                        "runway": d.RUNWAY_mvt.astype("string").to_numpy(dtype=object),
                        "callsign": cs.to_numpy(dtype=object), "mvt_s": mvt_s, "ref_s": ref_s,
                        "matched": np.isfinite(ref_s)})
    if out.MVT_ID_mvt.duplicated().any():
        raise ValueError("duplicate MVT_ID_mvt among the flights")
    return out.reset_index(drop=True)


def load_storm_flights(ranking_path: pathlib.Path, days=DAYS) -> pd.DataFrame:
    df = pq.read_table(ranking_path, columns=FLIGHT_COLUMNS, filters=[("ADEP_mvt", "=", AIRPORT), ("PHASE_mvt", "=", "DEP")]).to_pandas()
    return flights_frame(df, days)


def tracks_frame(samples: pd.DataFrame, lat0: float, lon0: float, gap_s: float = TRACK_GAP_S) -> pd.DataFrame:
    """ADS-B samples (icao, callsign, t, lat, lon, on_ground, gs) -> sorted by (icao, t) with x, y in metres and a
    `track_id`: a new track starts on an aircraft change, a callsign change or a gap > gap_s. Samples with no
    callsign yet inherit the segment's (forward- then backward-filled within the aircraft)."""
    need = ["icao", "callsign", "t", "lat", "lon", "on_ground", "gs"]
    missing = [c for c in need if c not in samples.columns]
    if missing:
        raise ValueError(f"samples need {missing}")
    s = samples[need].copy()
    s = s[np.isfinite(s.t.to_numpy(dtype="float64"))]
    if len(s) and not ((s.t > 1.6e9) & (s.t < 1.9e9)).all():
        raise ValueError("sample times outside the 2020-2030 epoch band")
    s["callsign"] = s.callsign.astype("string").str.strip().str.upper()
    s = s.sort_values(["icao", "t"], kind="mergesort").reset_index(drop=True)
    s["callsign"] = s.groupby("icao", sort=False).callsign.transform(lambda c: c.ffill().bfill())
    same_ac = s.icao.to_numpy() == np.roll(s.icao.to_numpy(), 1)
    cs = s.callsign.fillna("").to_numpy(dtype=object)
    same_cs = cs == np.roll(cs, 1)
    t = s.t.to_numpy(dtype="float64")
    small_gap = (t - np.roll(t, 1)) <= gap_s
    new = ~(same_ac & same_cs & small_gap)
    new[0] = True
    s["track_id"] = np.cumsum(new) - 1
    s["x"], s["y"] = xy_m(s.lat, s.lon, lat0, lon0)
    s["on_ground"] = s.on_ground.to_numpy(dtype="int64")
    s["gs"] = s.gs.to_numpy(dtype="float64")
    return s


def load_storm_tracks(days=DAYS, adsb_dir: pathlib.Path = ADSB_DIR) -> pd.DataFrame:
    frames = []
    for d in days:
        p = adsb_dir / f"adsb_{d}_{AIRPORT}.parquet"
        if not p.exists():
            raise SystemExit(f"missing {p}: run `adsb_storm.py extract --day {d}` first")
        f = pq.read_table(p).to_pandas()
        if "airport" in f.columns:
            f = f[f.airport == AIRPORT]
        if len(f) == 0:
            raise SystemExit(f"{p} holds no {AIRPORT} sample")
        frames.append(f)
    lat0, lon0 = ax.APTS[AIRPORT]
    return tracks_frame(pd.concat(frames, ignore_index=True), lat0, lon0)


# =============================================================================================
# the matched join (callsign) and the stand fixes
# =============================================================================================

def join_matched(flights: pd.DataFrame, tracks: pd.DataFrame, window_s: float = TRACK_WINDOW_S,
                 slack_s: float = TRACK_SLACK_S) -> np.ndarray:
    """Per flight the track_id (or -1): the track with the flight's callsign holding the most samples in
    [mvt_s - window_s, mvt_s + slack_s]. Only flights with a callsign can join."""
    by_cs = {k: g for k, g in tracks.groupby("callsign", sort=False)} if len(tracks) else {}
    out = np.full(len(flights), -1, dtype="int64")
    for i, (cs, mvt) in enumerate(zip(flights.callsign.to_numpy(), flights.mvt_s.to_numpy())):
        if cs is None or cs is pd.NA or not isinstance(cs, str) or cs == "" or cs not in by_cs:
            continue
        g = by_cs[cs]
        inwin = g[(g.t >= mvt - window_s) & (g.t <= mvt + slack_s)]
        if len(inwin) == 0:
            continue
        out[i] = int(inwin.track_id.value_counts().idxmax())
    return out


def stand_fixes(flights: pd.DataFrame, tracks: pd.DataFrame, track_of: np.ndarray, tol_s: float = FIX_TOL_S) -> pd.DataFrame:
    """For matched flights with a track: the position at the off-block reference -- the nearest on-ground sample
    within tol_s of ref_s. Columns MVT_ID_mvt, stand, track_id, x, y, dt_s."""
    by_track = {k: g for k, g in tracks.groupby("track_id", sort=False)} if len(tracks) else {}
    rows = []
    for mid, stand, ref, matched, tid in zip(flights.MVT_ID_mvt, flights.stand, flights.ref_s, flights.matched, track_of):
        if not matched or tid < 0 or not isinstance(stand, str):
            continue
        g = by_track[tid]
        g = g[g.on_ground == 1]
        if len(g) == 0:
            continue
        dt = np.abs(g.t.to_numpy() - ref)
        j = int(np.argmin(dt))
        if dt[j] <= tol_s:
            rows.append((mid, stand, int(tid), float(g.x.iloc[j]), float(g.y.iloc[j]), float(dt[j])))
    return pd.DataFrame(rows, columns=["MVT_ID_mvt", "stand", "track_id", "x", "y", "dt_s"])


def row_centroids(flights: pd.DataFrame, track_of: np.ndarray, fixes: pd.DataFrame,
                  min_other: int = MIN_OTHER_FLIGHTS) -> tuple:
    """Leave-one-flight-out, per ROW: (cx, cy, n_other) arrays in `flights` order. The centroid is the coordinate-wise
    median of the fixes at the row's stand that are not its own -- neither the fix keyed by its MVT_ID_mvt nor any fix
    taken on its own track (track_of[i]) -- and is NaN unless >= min_other such fixes remain. A matched row without a
    fix of its own and an unmatched row get the median of every fix at their stand, as the prereg's rule implies."""
    n = len(flights)
    cx, cy, n_other = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, dtype="int64")
    if len(fixes) == 0:
        return cx, cy, n_other
    by_stand = {k: (g.MVT_ID_mvt.to_numpy(), g.track_id.to_numpy(), g.x.to_numpy(), g.y.to_numpy())
                for k, g in fixes.groupby("stand", sort=False)}
    for i, (mid, stand, tid) in enumerate(zip(flights.MVT_ID_mvt.to_numpy(), flights.stand.to_numpy(), np.asarray(track_of))):
        if not isinstance(stand, str) or stand not in by_stand:
            continue
        ids, tids, xs, ys = by_stand[stand]
        keep = (ids != mid) & ((tids != tid) | (tid < 0))
        n_other[i] = int(keep.sum())
        if n_other[i] >= min_other:
            cx[i], cy[i] = float(np.median(xs[keep])), float(np.median(ys[keep]))
    return cx, cy, n_other


def last_sample_in_gate(track: pd.DataFrame, cx: float, cy: float, mvt_s: float, radius_m: float = RADIUS_M) -> float:
    """The time of the last sample with t <= mvt_s inside radius_m of (cx, cy); NaN when none."""
    g = track[track.t <= mvt_s]
    if len(g) == 0:
        return float("nan")
    d = np.hypot(g.x.to_numpy() - cx, g.y.to_numpy() - cy)
    inside = d <= radius_m
    return float(g.t.to_numpy()[inside].max()) if inside.any() else float("nan")


# =============================================================================================
# the unmatched join: runway + take-off second
# =============================================================================================

def takeoff_events(tracks: pd.DataFrame, gs_kt: float = TAKEOFF_GS_KT, dir_tol_s: float = DIRECTION_TOL_S) -> pd.DataFrame:
    """Per track the first sample with ground speed > gs_kt that is preceded, in the same track, by an on-ground
    sample (an arriving track's first fast sample is airborne and has none). Columns track_id, t, x, y, ux, uy
    (unit heading from the neighbouring sample within dir_tol_s; NaN when there is none)."""
    rows = []
    for tid, g in tracks.groupby("track_id", sort=False):
        t, x, y, gs, og = (g.t.to_numpy(), g.x.to_numpy(), g.y.to_numpy(), g.gs.to_numpy(), g.on_ground.to_numpy())
        fast = np.flatnonzero(gs > gs_kt)
        if len(fast) == 0:
            continue
        k = int(fast[0])
        if not (og[:k] == 1).any():
            continue
        ux = uy = np.nan
        if k + 1 < len(t) and t[k + 1] - t[k] <= dir_tol_s:
            dx, dy = x[k + 1] - x[k], y[k + 1] - y[k]
        elif k >= 1 and t[k] - t[k - 1] <= dir_tol_s:
            dx, dy = x[k] - x[k - 1], y[k] - y[k - 1]
        else:
            dx = dy = 0.0
        n = float(np.hypot(dx, dy))
        if n > 0:
            ux, uy = dx / n, dy / n
        rows.append((tid, float(t[k]), float(x[k]), float(y[k]), ux, uy))
    return pd.DataFrame(rows, columns=["track_id", "t", "x", "y", "ux", "uy"])


def learn_runways(events: pd.DataFrame, labels: np.ndarray, min_events: int = MIN_RUNWAY_EVENTS) -> pd.DataFrame:
    """Runway geometry from matched flights' take-off events: per RUNWAY_mvt label the median point and the unit
    direction (mean of the events' headings). Columns runway, mx, my, ux, uy, n."""
    e = events.assign(runway=np.asarray(labels, dtype=object))
    e = e[np.isfinite(e.ux) & e.runway.notna()]
    out = []
    for rw, g in e.groupby("runway", sort=False):
        if len(g) < min_events:
            continue
        ux, uy = float(g.ux.mean()), float(g.uy.mean())
        n = float(np.hypot(ux, uy))
        if n == 0:
            continue
        out.append((rw, float(g.x.median()), float(g.y.median()), ux / n, uy / n, int(len(g))))
    return pd.DataFrame(out, columns=["runway", "mx", "my", "ux", "uy", "n"])


def label_runway(events: pd.DataFrame, runways: pd.DataFrame, lateral_m: float = RUNWAY_LATERAL_M,
                 heading_cos: float = RUNWAY_HEADING_COS, along_m: float = RUNWAY_ALONG_M) -> np.ndarray:
    """Per event the runway label it lies on: lateral distance to the learned line <= lateral_m, heading agreement
    >= heading_cos, and |distance along the line from the learned point| <= along_m (a runway is a segment, not an
    infinite line); the laterally nearest when several; None otherwise."""
    out = np.full(len(events), None, dtype=object)
    if len(runways) == 0 or len(events) == 0:
        return out
    ex, ey, eux, euy = (events.x.to_numpy(), events.y.to_numpy(), events.ux.to_numpy(), events.uy.to_numpy())
    best = np.full(len(events), np.inf)
    for r in runways.itertuples(index=False):
        lat = np.abs(r.ux * (ey - r.my) - r.uy * (ex - r.mx))          # |u x (p - m)|
        along = np.abs(r.ux * (ex - r.mx) + r.uy * (ey - r.my))        # |u . (p - m)|
        cos = eux * r.ux + euy * r.uy
        ok = (lat <= lateral_m) & (along <= along_m) & (cos >= heading_cos) & (lat < best)
        out[ok] = r.runway
        best[ok] = lat[ok]
    return out


def join_unmatched(flights_unm: pd.DataFrame, events: pd.DataFrame, offset_s: float, tol_s: float = TAKEOFF_TOL_S) -> pd.DataFrame:
    """Runway + take-off second. `events` carries track_id, t, runway (labelled). Per unmatched row the candidates
    are events with the row's runway and |t - (mvt_s + offset_s)| <= tol_s. Columns MVT_ID_mvt, n_candidates,
    track_id (the single candidate's, else -1), ambiguous (>= 2 candidates, or the single candidate is another
    row's single candidate too)."""
    ev = events[events.runway.notna()] if len(events) else events
    by_rw = {k: g for k, g in ev.groupby("runway", sort=False)} if len(ev) else {}
    n_c, tid = np.zeros(len(flights_unm), dtype="int64"), np.full(len(flights_unm), -1, dtype="int64")
    for i, (rw, mvt) in enumerate(zip(flights_unm.runway.to_numpy(), flights_unm.mvt_s.to_numpy())):
        if rw not in by_rw:
            continue
        g = by_rw[rw]
        c = g[np.abs(g.t.to_numpy() - (mvt + offset_s)) <= tol_s]
        n_c[i] = len(c)
        if len(c) == 1:
            tid[i] = int(c.track_id.iloc[0])
    single = tid >= 0
    claimed = pd.Series(tid[single]).value_counts()
    n_claims = pd.Series(tid).map(claimed).fillna(0).to_numpy(dtype="int64")
    conflict = single & (n_claims > 1)
    ambiguous = (n_c >= 2) | conflict
    tid = np.where(conflict, -1, tid)
    return pd.DataFrame({"MVT_ID_mvt": flights_unm.MVT_ID_mvt.to_numpy(), "n_candidates": n_c, "track_id": tid,
                         "ambiguous": ambiguous})


def clause_c0(joined: pd.DataFrame, max_share: float = C0_MAX_SHARE) -> dict:
    with_c = int((joined.n_candidates >= 1).sum()) if len(joined) else 0
    amb = int(joined.ambiguous.sum()) if len(joined) else 0
    share = amb / with_c if with_c else float("nan")
    return {"clause": "C0", "n_unmatched_rows": int(len(joined)), "n_with_candidate": with_c,
            "n_no_candidate": int(len(joined)) - with_c, "n_ambiguous_or_unmatched": amb,
            "no_candidate_share_of_rows": (int(len(joined)) - with_c) / len(joined) if len(joined) else float("nan"),
            "share": share, "threshold": max_share, "pass": bool(with_c > 0 and share <= max_share)}


# =============================================================================================
# the estimate, the leave-one-day-out bias, the clauses
# =============================================================================================

def raw_estimates(flights: pd.DataFrame, tracks: pd.DataFrame, track_of: np.ndarray, cx: np.ndarray, cy: np.ndarray,
                  radius_m: float = RADIUS_M) -> np.ndarray:
    """Per flight the raw off-block (epoch s): the last sample of its track before take-off inside radius_m of
    (cx, cy); NaN without a track or a centroid or an inside sample."""
    by_track = {k: g for k, g in tracks.groupby("track_id", sort=False)} if len(tracks) else {}
    out = np.full(len(flights), np.nan)
    for i, (tid, mvt) in enumerate(zip(track_of, flights.mvt_s.to_numpy())):
        if tid < 0 or not (np.isfinite(cx[i]) and np.isfinite(cy[i])):
            continue
        out[i] = last_sample_in_gate(by_track[tid], cx[i], cy[i], mvt, radius_m)
    return out


def loo_day_bias(day: np.ndarray, err_raw: np.ndarray, use: np.ndarray) -> dict:
    """{day: median of err_raw over `use` rows on the OTHER days}; NaN when no other day has a usable row."""
    days = sorted(set(day))
    out = {}
    for d in days:
        m = use & (day != d) & np.isfinite(err_raw)
        out[d] = float(np.median(err_raw[m])) if m.any() else float("nan")
    return out


def pooled_bias(err_raw: np.ndarray, use: np.ndarray) -> float:
    m = use & np.isfinite(err_raw)
    return float(np.median(err_raw[m])) if m.any() else float("nan")


def clause_c1(err_corr: np.ndarray, matched: np.ndarray, rmse_max: float = C1_RMSE_S, cov_min: float = C1_MIN_COVERAGE) -> dict:
    """err_corr: bias-corrected (offblock_hat - AOBT_3) per row, NaN where the gate did not fire."""
    fired = matched & np.isfinite(err_corr)
    n_m = int(matched.sum())
    cov = fired.sum() / n_m if n_m else float("nan")
    r = rmse(err_corr[fired])
    return {"clause": "C1", "n_matched": n_m, "n_gated": int(fired.sum()), "coverage": float(cov), "rmse_s": r,
            "rmse_threshold_s": rmse_max, "coverage_threshold": cov_min,
            "pass": bool(n_m > 0 and fired.sum() > 0 and r <= rmse_max and cov >= cov_min)}


def clause_c2(day: np.ndarray, err_corr: np.ndarray, matched: np.ndarray, days=DAYS, rmse_max: float = C2_RMSE_S,
              min_days: int = C2_MIN_DAYS) -> dict:
    per = {}
    for d in days:
        m = matched & (day == d) & np.isfinite(err_corr)
        per[d] = {"n_gated": int(m.sum()), "n_matched": int((matched & (day == d)).sum()), "rmse_s": rmse(err_corr[m])}
    ok = sum(1 for v in per.values() if np.isfinite(v["rmse_s"]) and v["rmse_s"] <= rmse_max)
    return {"clause": "C2", "per_day": per, "n_days_within": ok, "n_days": len(days), "rmse_threshold_s": rmse_max,
            "min_days": min_days, "pass": bool(ok >= min_days)}


def clause_c3(rmse_gated_s: float, coverage: float, n_gated: int, model_rmse_s: float = MODEL_RMSE_EHAM_S) -> dict:
    return {"clause": "C3", "rmse_gated_vs_label_s": float(rmse_gated_s), "coverage": float(coverage), "n_gated": int(n_gated),
            "model_rmse_s": float(model_rmse_s), "pass": bool(n_gated > 0 and rmse_gated_s < model_rmse_s)}


def verdict(c0: dict, c1: dict, c2: dict, c3: dict) -> str:
    if c0["pass"] and c1["pass"] and c2["pass"] and c3["pass"]:
        return "WORKING"
    if not c1["pass"]:
        return "NOT WORKING"
    return "INCONCLUSIVE"


# =============================================================================================
# the pipeline on one set of flights + tracks
# =============================================================================================

def measure(flights: pd.DataFrame, tracks: pd.DataFrame, days, bias_mode: str = "loo_day", bias_s: float | None = None) -> tuple:
    """(estimates frame in `flights` order, info). Matched rows join by callsign, unmatched rows by runway +
    take-off second; then every row gets its leave-one-flight-out centroid (row_centroids), the gate, and the bias of
    its day: the median over matched gated rows of the OTHER days (bias_mode 'loo_day'), or, on the one-day census,
    that day's own median (bias_mode 'pooled'), or E4b's ONE label-referenced constant `bias_s` (bias_mode 'constant')."""
    if bias_mode not in ("loo_day", "pooled", "constant"):
        raise ValueError(f"bias_mode {bias_mode!r}")
    if bias_mode == "constant":
        if bias_s is None or not np.isfinite(bias_s):
            raise ValueError("bias_mode 'constant' needs a finite bias_s")
    elif bias_s is not None:
        raise ValueError("bias_s is only for bias_mode 'constant'")
    track_of = join_matched(flights, tracks)
    fixes = stand_fixes(flights, tracks, track_of)
    n = len(flights)
    matched = flights.matched.to_numpy()
    ids = flights.MVT_ID_mvt.to_numpy()

    # unmatched join
    events = takeoff_events(tracks)
    ev_of_track = events.set_index("track_id") if len(events) else events
    m_lab = matched & (track_of >= 0)
    lab_ev = []
    for tid, rw in zip(track_of[m_lab], flights.runway.to_numpy()[m_lab]):
        if tid in ev_of_track.index:
            r = ev_of_track.loc[tid]
            lab_ev.append((tid, float(r.t), float(r.x), float(r.y), float(r.ux), float(r.uy), rw))
    lab_ev = pd.DataFrame(lab_ev, columns=["track_id", "t", "x", "y", "ux", "uy", "runway"])
    runways = learn_runways(lab_ev[["track_id", "t", "x", "y", "ux", "uy"]], lab_ev.runway.to_numpy()) if len(lab_ev) else \
        pd.DataFrame(columns=["runway", "mx", "my", "ux", "uy", "n"])
    mvt_m = flights.mvt_s.to_numpy()[m_lab]
    off = []
    for tid, mvt in zip(track_of[m_lab], mvt_m):
        if tid in ev_of_track.index:
            off.append(float(ev_of_track.t.loc[tid]) - mvt)
    offset_s = float(np.median(off)) if off else float("nan")
    used = set(track_of[track_of >= 0].tolist())
    cand = events[~events.track_id.isin(used)].copy() if len(events) else events.copy()
    cand["runway"] = label_runway(cand, runways)
    unm = flights[~matched]
    joined = join_unmatched(unm, cand, offset_s) if np.isfinite(offset_s) else \
        pd.DataFrame({"MVT_ID_mvt": unm.MVT_ID_mvt.to_numpy(), "n_candidates": 0, "track_id": -1, "ambiguous": False})
    j = joined.set_index("MVT_ID_mvt")
    n_cand, amb = np.zeros(n, dtype="int64"), np.zeros(n, dtype=bool)
    for i in np.flatnonzero(~matched):
        track_of[i] = int(j.track_id.loc[ids[i]])
        n_cand[i] = int(j.n_candidates.loc[ids[i]])
        amb[i] = bool(j.ambiguous.loc[ids[i]])

    cx, cy, _ = row_centroids(flights, track_of, fixes)
    raw = raw_estimates(flights, tracks, track_of, cx, cy)
    err_raw = raw - flights.ref_s.to_numpy()
    day = flights.day.to_numpy()
    pooled = pooled_bias(err_raw, matched)
    if bias_mode == "loo_day":
        bias_by_day = loo_day_bias(day, err_raw, matched)
    elif bias_mode == "pooled":
        bias_by_day = {d: pooled for d in sorted(set(day))}
    else:
        bias_by_day = {d: float(bias_s) for d in sorted(set(day))}
    bias = np.array([bias_by_day[d] for d in day], dtype="float64")
    hat = raw - bias
    fired = np.isfinite(hat)
    est = pd.DataFrame({"MVT_ID_mvt": ids, "matched": matched, "day": day, "stand": flights.stand.to_numpy(),
                        "runway": flights.runway.to_numpy(),
                        "offblock_hat": pd.to_datetime(np.where(fired, hat, np.nan), unit="s", utc=True),
                        "taxi_hat": np.where(fired, flights.mvt_s.to_numpy() - hat, np.nan), "gate_fired": fired,
                        "join_ambiguous": amb, "n_candidates": n_cand, "offblock_raw_s": raw, "bias_s": bias,
                        "track_id": track_of})[ESTIMATE_COLUMNS]
    err_corr = np.where(matched, hat - flights.ref_s.to_numpy(), np.nan)
    fixes_per_stand = fixes.groupby("stand").size() if len(fixes) else pd.Series(dtype="int64")
    info = {"n_flights": n, "n_matched": int(matched.sum()), "n_matched_joined": int((matched & (track_of >= 0)).sum()),
            "n_fixes": int(len(fixes)), "n_stands_with_2plus_fixes": int((fixes_per_stand >= MIN_OTHER_FLIGHTS).sum()),
            "n_rows_with_centroid": int(np.isfinite(cx).sum()),
            "n_matched_with_own_fix": int(np.isin(ids[matched], fixes.MVT_ID_mvt.to_numpy()).sum()) if len(fixes) else 0,
            "runways_learned": {str(r.runway): int(r.n) for r in runways.itertuples(index=False)},
            "takeoff_offset_s": offset_s, "n_takeoff_events": int(len(events)),
            "bias_by_day_s": bias_by_day, "pooled_bias_s": pooled, "err_corr": err_corr, "err_raw": err_raw, "joined": joined}
    return est, info


# =============================================================================================
# C3: the 2025-01-09 census
# =============================================================================================

def census_c3(truth: pd.DataFrame, samples: pd.DataFrame, day: str = CENSUS_DAY) -> dict:
    """The same pipeline on the labelled census day: centroids and bias at AOBT_3 (the prereg's reference; one day,
    so the bias is that day's own pooled median), scored as taxi_hat against TAXITIME_SEC_mvt on matched rows where
    the gate fires. Also reports the RMSE against AOBT_3 (the C1 analogue ADSB_GATE's 127 s corresponds to) and a
    BLOCK_TIME-calibrated variant as diagnostics. It reads labels, so it refuses any day that is not a 2025 day."""
    if not str(day).startswith("2025-"):
        raise ValueError(f"census_c3 reads labels and runs on 2025 days only, not {day!r}: no 2026 label may be read")
    for c in ("TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"):
        if c not in truth.columns:
            raise ValueError(f"the census truth needs {c!r}")
    lat0, lon0 = ax.APTS[AIRPORT]
    tracks = tracks_frame(samples[samples.airport == AIRPORT] if "airport" in samples.columns else samples, lat0, lon0)
    fl = flights_frame(truth, [day])
    est, info = measure(fl, tracks, [day], bias_mode="pooled")
    fl_i = fl.set_index("MVT_ID_mvt")
    t_i = truth.set_index("MVT_ID_mvt")
    m = est.matched.to_numpy() & est.gate_fired.to_numpy()
    ids = est.MVT_ID_mvt.to_numpy()[m]
    label = t_i.TAXITIME_SEC_mvt.loc[ids].to_numpy(dtype="float64")
    taxi_hat = est.taxi_hat.to_numpy()[m]
    n_matched = int(est.matched.sum())
    out = clause_c3(rmse(taxi_hat - label), m.sum() / n_matched if n_matched else float("nan"), int(m.sum()))
    out["rmse_vs_aobt3_s"] = rmse(info["err_corr"][m])
    out["bias_at_aobt3_s"] = info["pooled_bias_s"]
    # diagnostic: the same raw sensor calibrated to BLOCK_TIME (the label's own off-block)
    block_s = to_epoch_s(t_i.BLOCK_TIME_UTC_mvt.loc[ids])
    raw = est.offblock_raw_s.to_numpy()[m]
    b_block = float(np.median(raw - block_s))
    out["diagnostic_block_calibrated"] = {"bias_s": b_block, "rmse_vs_label_s": rmse((raw - b_block) - block_s)}
    a3 = fl_i.ref_s.loc[ids].to_numpy()
    out["diagnostic_median_aobt3_minus_block_s"] = float(np.median(a3 - block_s))
    out["day"] = day
    out["n_matched"] = n_matched
    out["n_matched_joined"] = info["n_matched_joined"]
    return out


def load_census() -> tuple:
    truth = pq.read_table(ADSB_DIR / "pilot_truth.parquet").to_pandas()
    truth = truth[(truth.ADEP_mvt == AIRPORT) & (truth.day == CENSUS_DAY)]
    samples = pq.read_table(ADSB_DIR / f"adsb_{CENSUS_DAY}.parquet", filters=[("airport", "=", AIRPORT)]).to_pandas()
    return truth, samples


# =============================================================================================
# steps
# =============================================================================================

def sample_span(samples: pd.DataFrame) -> str:
    """'<first UTC> .. <last UTC> (<hours> h, <n> aircraft)' of a samples frame's `t`; '(no sample)' when empty. The
    epoch band is asserted, so a microsecond or offset-only time column fails here, not three steps later."""
    t = samples.t.to_numpy(dtype="float64") if len(samples) else np.array([])
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return "(no sample)"
    if not ((t > 1.6e9) & (t < 1.9e9)).all():
        raise ValueError("sample times outside the 2020-2030 epoch band")
    lo, hi = pd.Timestamp(t.min(), unit="s", tz="UTC"), pd.Timestamp(t.max(), unit="s", tz="UTC")
    return f"{lo:%Y-%m-%d %H:%M:%S} .. {hi:%Y-%m-%d %H:%M:%S} UTC ({(t.max() - t.min()) / 3600:.1f} h, {samples.icao.nunique()} aircraft)"


def step_extract(day: str, probe: bool, validate_path: pathlib.Path | None = None) -> int:
    """One registered day: E4/E4b's eight 2026 storm days or E4b's two 2025 validation days. The EHAM box only is kept,
    the archive is deleted after (probe or not). A FULL 2026 extract refuses unless the E4b validation verdict on disk is
    WORKING (PREREG_adsb_eham_v2: 'Only if WORKING' does the 2026 ingest run; 'If V1 fails ... nothing further is
    ingested'); E4's own registration runs no 2026 ingest (its PRE-INGEST FINDING)."""
    if day not in DAYS and day not in VALIDATION_DAYS:
        raise SystemExit(f"{day} is not a registered day: storm {DAYS[0]} ... {DAYS[-1]} or E4b validation "
                         f"{' and '.join(VALIDATION_DAYS)}")
    if day in DAYS and not probe:
        require_e4b_working(validate_path or VALIDATE_REPORT)
    out = ADSB_DIR / f"adsb_{day}_{AIRPORT}.parquet"
    if out.exists() and not probe:
        raise SystemExit(f"refusing to overwrite {out}")
    t0 = time.time()
    log(f"[{day}] fetching{' (probe, first 60 MB)' if probe else ''} ...")
    paths = ax.fetch(day, probe)
    t1 = time.time()
    size = sum(p.stat().st_size for p in paths)
    log(f"[{day}] {size/2**20:.0f} MiB on disk in {t1 - t0:.0f} s; extracting")
    try:
        df = ax.extract(paths, probe)
    finally:
        for p in paths:                                   # never keep an archive, probe or not
            if p.exists():
                p.unlink()
    t2 = time.time()
    eham = df[df.airport == AIRPORT].reset_index(drop=True)
    log(f"[{day}] {len(df):,} in-box samples, {len(eham):,} {AIRPORT}; parse {t2 - t1:.0f} s; "
        f"{eham.callsign.notna().mean() if len(eham) else float('nan'):.3f} with callsign, "
        f"{eham.on_ground.mean() if len(eham) else float('nan'):.3f} on ground")
    log(f"[{day}] {AIRPORT} span: {sample_span(eham)}")
    if probe:
        log(f"[{day}] probe: nothing written (a partial day must never look like a day file)")
        return 0
    if len(eham) == 0:
        raise SystemExit(f"[{day}] zero {AIRPORT} samples: bbox or parser wrong; nothing written")
    ADSB_DIR.mkdir(parents=True, exist_ok=True)
    eham.to_parquet(out, index=False)
    log(f"[{day}] wrote {out.name}: {len(eham):,} rows, {time.time() - t0:.0f} s total")
    return 0


def _json_ready(o):
    if isinstance(o, dict):
        return {str(k): _json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_ready(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def step_census() -> int:
    truth, samples = load_census()
    c3 = census_c3(truth, samples)
    print(json.dumps(_json_ready(c3), indent=1))
    return 0


def gate_report(flights: pd.DataFrame, tracks: pd.DataFrame, c3: dict, days=DAYS) -> tuple:
    """(estimates, report dict) -- every clause, the verdict, coverage and RMSE per day."""
    est, info = measure(flights, tracks, days, bias_mode="loo_day")
    matched, day, err = est.matched.to_numpy(), est.day.to_numpy(), info["err_corr"]
    c1, c2, c0 = clause_c1(err, matched), clause_c2(day, err, matched, days), clause_c0(info["joined"])
    per_day = {}
    for d in days:
        m = day == d
        um = m & ~matched
        per_day[d] = {**c2["per_day"][d], "coverage": (c2["per_day"][d]["n_gated"] / c2["per_day"][d]["n_matched"])
                      if c2["per_day"][d]["n_matched"] else None, "bias_loo_s": info["bias_by_day_s"].get(d),
                      "n_unmatched": int(um.sum()), "n_unmatched_gated": int((um & est.gate_fired.to_numpy()).sum()),
                      "n_unmatched_ambiguous": int((um & est.join_ambiguous.to_numpy()).sum()),
                      "unmatched_taxi_hat_mean_s": float(np.nanmean(est.taxi_hat.to_numpy()[um])) if (um & est.gate_fired.to_numpy()).any() else None}
    v = verdict(c0, c1, c2, c3)
    rep = {"arm": ARM, "prereg": PREREG, "airport": AIRPORT, "days": list(days), "verdict": v,
           "clauses": {"C0": c0, "C1": c1, "C2": c2, "C3": c3}, "per_day": per_day,
           "design": {"radius_m": RADIUS_M, "min_other_flights": MIN_OTHER_FLIGHTS, "takeoff_gs_kt": TAKEOFF_GS_KT,
                      "fix_tol_s": FIX_TOL_S, "track_window_s": TRACK_WINDOW_S, "takeoff_tol_s": TAKEOFF_TOL_S,
                      "runway_lateral_m": RUNWAY_LATERAL_M, "runway_heading_cos": RUNWAY_HEADING_COS},
           "join": {"n_matched_joined_by_callsign": info["n_matched_joined"], "n_fixes": info["n_fixes"],
                    "n_stands_with_2plus_fixes": info["n_stands_with_2plus_fixes"],
                    "n_rows_with_centroid": info["n_rows_with_centroid"],
                    "n_matched_with_own_fix": info["n_matched_with_own_fix"], "runways_learned": info["runways_learned"],
                    "takeoff_offset_s": info["takeoff_offset_s"], "n_takeoff_events": info["n_takeoff_events"]},
           "pooled_bias_s": info["pooled_bias_s"], "n_rows": int(len(est)), "n_gate_fired": int(est.gate_fired.sum()),
           "n_unmatched_gate_fired": int((est.gate_fired & ~est.matched).sum()),
           "judgement_calls": JUDGEMENT_CALLS, "eligibility": ELIGIBILITY_RULE}
    return est, rep


def step_gate() -> int:
    t0 = time.time()
    flights = load_storm_flights(ROOT / "data" / "raw" / "ranking.parquet")
    log(f"ranking: {len(flights):,} {AIRPORT} departures on the eight days, {int(flights.matched.sum()):,} matched")
    tracks = load_storm_tracks()
    log(f"tracks: {len(tracks):,} samples, {tracks.track_id.nunique():,} tracks")
    truth, samples = load_census()
    c3 = census_c3(truth, samples)
    log(f"C3 (2025-01-09): gated RMSE vs label {c3['rmse_gated_vs_label_s']:.1f} s at coverage {c3['coverage']:.3f} "
        f"vs model {c3['model_rmse_s']} -> {'pass' if c3['pass'] else 'FAIL'}")
    est, rep = gate_report(flights, tracks, c3)
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["written"] = {"report": str(REPORT.relative_to(ROOT)), "estimates": str(ESTIMATES.relative_to(ROOT))}
    est.to_parquet(ESTIMATES, index=False)
    REPORT.write_text(json.dumps(_json_ready(rep), indent=1))
    c = rep["clauses"]
    log(f"C0 share {c['C0']['share']} (n with candidate {c['C0']['n_with_candidate']}) -> {c['C0']['pass']}")
    log(f"C1 RMSE {c['C1']['rmse_s']:.1f} s, coverage {c['C1']['coverage']:.3f} -> {c['C1']['pass']}")
    log(f"C2 {c['C2']['n_days_within']} of {c['C2']['n_days']} days within {C2_RMSE_S:.0f} s -> {c['C2']['pass']}")
    for d, r in rep["per_day"].items():
        log(f"  {d}: matched {r['n_matched']:4d} gated {r['n_gated']:4d} cov {r['coverage']!s:>6.6} rmse {r['rmse_s']!s:>8.8} "
            f"bias_loo {r['bias_loo_s']!s:>8.8} | unmatched {r['n_unmatched']:3d} gated {r['n_unmatched_gated']:3d} "
            f"ambiguous {r['n_unmatched_ambiguous']:3d} taxi_hat mean {r['unmatched_taxi_hat_mean_s']!s:>8.8}")
    log(f"VERDICT: {rep['verdict']}")
    log(f"wrote {REPORT} and {ESTIMATES}")
    return 0


# =============================================================================================
# ship
# =============================================================================================

def ship_rows(est: pd.DataFrame, days=DAYS) -> pd.DataFrame:
    """The rows this arm may touch: unmatched rows of the registered days where the gate fired and the join was
    not ambiguous. Value = rint(taxi_hat) floored at 1."""
    m = (~est.matched.to_numpy()) & est.gate_fired.to_numpy() & (~est.join_ambiguous.to_numpy()) & est.day.isin(list(days)).to_numpy()
    r = est[m].reset_index(drop=True)
    if not np.isfinite(r.taxi_hat.to_numpy(dtype="float64")).all():
        raise AssertionError("a gated row carries a non-finite taxi_hat")
    return r.assign(after=np.maximum(np.rint(r.taxi_hat.to_numpy(dtype="float64")), 1.0))


def verify_base(base: pd.DataFrame, detail: pd.DataFrame, target_ids) -> dict:
    """The prereg's rebuild guard (the rome_bandfloor pattern): `base` must BE the version `detail` describes on every
    row this arm may touch -- every target id (the ranking file's EHAM unmatched storm departures) is in the detail
    file, and the base's value there equals the detail's `after`. Raises AssertionError otherwise."""
    for c in ("MVT_ID_mvt", "after"):
        if c not in detail.columns:
            raise ValueError(f"the detail file needs {c!r}")
    ids = np.asarray(sorted(target_ids), dtype="float64")
    if len(ids) == 0:
        raise AssertionError("no target row to verify the base on")
    d = detail.set_index("MVT_ID_mvt")
    if d.index.duplicated().any():
        raise AssertionError("the detail file repeats an MVT_ID_mvt")
    missing = int((~np.isin(ids, d.index.to_numpy(dtype="float64"))).sum())
    if missing:
        raise AssertionError(f"{missing} of {len(ids)} {AIRPORT} unmatched storm rows are not in the detail file: "
                             "it is not the file that last wrote these rows")
    have = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy(dtype="float64")
    want = d.after.reindex(ids).to_numpy(dtype="float64")
    bad = int((np.isnan(have) | (have != want)).sum())
    if bad:
        raise AssertionError(f"the base differs from the detail file's `after` on {bad} of {len(ids)} {AIRPORT} unmatched storm "
                             "rows: it is not the version it claims to be; nothing is written")
    return {"n_checked": int(len(ids))}


def check_licence_doc(path, root: pathlib.Path = ROOT) -> str:
    """The prereg: 'the derived table's licence (ODbL share-alike) is documented in the repo before any ship'. The
    file must exist inside the repo and name both ODbL and adsb.lol. Returns its repo-relative path."""
    p = pathlib.Path(path)
    p = (root / p) if not p.is_absolute() else p
    p = p.resolve()
    try:
        rel = p.relative_to(root.resolve())
    except ValueError:
        raise SystemExit(f"refusing to ship: the licence document {p} is not inside the repository {root}")
    if not p.is_file():
        raise SystemExit(f"refusing to ship: the licence document {p} does not exist")
    text = p.read_text(errors="replace")
    if "ODbL" not in text or "adsb.lol" not in text.lower():
        raise SystemExit(f"refusing to ship: {rel} does not document the ODbL licence of the adsb.lol-derived table")
    return str(rel)


def build_ship(base: pd.DataFrame, est: pd.DataFrame, eligible_ids, days=DAYS) -> tuple:
    """(new submission frame, info). Only `ship_rows` change, each must be in `eligible_ids` (the ranking file's
    EHAM unmatched departures of the registered days); everything else is byte-identical; the base is not mutated."""
    rows = ship_rows(est, days)
    ids, vals = rows.MVT_ID_mvt.to_numpy(dtype="float64"), rows.after.to_numpy(dtype="float64")
    if len(ids) == 0:
        raise AssertionError("no row to ship: the gate fired on no unmatched row of the registered days")
    outside = ~np.isin(ids, np.asarray(list(eligible_ids), dtype="float64"))
    if outside.any():
        raise AssertionError(f"{int(outside.sum())} rows to ship are not {AIRPORT} unmatched departures of the registered days in the ranking file")
    out = bs.splice_unmatched(base, ids, vals)
    pos = pd.Index(base.MVT_ID_mvt).get_indexer(ids)
    if not np.array_equal(out.TAXITIME_SEC_mvt.to_numpy()[pos].astype("int64"), vals.astype("int64")):
        raise AssertionError("rint(taxi_hat) does not fit the submission dtype")
    changed = np.flatnonzero(out.TAXITIME_SEC_mvt.to_numpy() != base.TAXITIME_SEC_mvt.to_numpy())
    if not set(changed.tolist()) <= set(pos.tolist()):
        raise AssertionError("a row outside the gated unmatched storm set changed; nothing is written")
    before = base.TAXITIME_SEC_mvt.to_numpy()[pos].astype("float64")
    shift = vals - before
    detail = pd.DataFrame({"MVT_ID_mvt": ids, "day": rows.day.to_numpy(), "stand": rows.stand.to_numpy(), "runway": rows.runway.to_numpy(),
                           "offblock_hat": rows.offblock_hat.to_numpy(), "taxi_hat": rows.taxi_hat.to_numpy(dtype="float64"),
                           "before": before, "after": vals})[DETAIL_COLUMNS]
    info = {"n_rows_replaced": int(len(ids)), "n_values_differ": int((shift != 0).sum()), "mean_shift_s": float(shift.mean()),
            "rms_shift_vs_base_s": float(np.sqrt(np.mean(shift ** 2))), "before_mean_s": float(before.mean()), "after_mean_s": float(vals.mean()),
            "per_day_n": {str(k): int(v) for k, v in rows.day.value_counts().sort_index().items()},
            "sse_shift_vs_base_board_mse": float((shift ** 2).sum() / len(base)), "detail": detail}
    return out, info


def _git() -> dict:
    def run(*a):
        r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain")
    return {"git_sha": run("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status)}


def write_ship_outputs(out: pd.DataFrame, info: dict, template: pd.DataFrame, version: int, base_name: str,
                       sub_dir: pathlib.Path, report: dict, t0: float, arm: str = ARM, prereg: str = PREREG,
                       extra_meta: dict | None = None) -> tuple:
    """check_submission on the frame, refuse an existing file, write, check the re-read file, detail + meta. `arm`,
    `prereg` and `extra_meta` name the arm in the meta (E4 by default; E4b adds its bias constant)."""
    bs.check_submission(out, template)
    dest = sub_dir / bs.submission_name(version)
    if dest.exists():
        raise SystemExit(f"refusing to overwrite {dest}")
    sub_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    reread = pq.read_table(dest).to_pandas()
    bs.check_submission(reread, template)
    if not np.array_equal(reread.TAXITIME_SEC_mvt.to_numpy(), out.TAXITIME_SEC_mvt.to_numpy()) or \
            not np.array_equal(reread.MVT_ID_mvt.to_numpy(), out.MVT_ID_mvt.to_numpy()):
        raise AssertionError(f"{dest} re-read differs from the frame that was written")
    info["detail"].to_parquet(sub_dir / f"{dest.stem}.adsb_rows.parquet", index=False)
    meta = {"version": int(version), "file": dest.name, "base": base_name, "arm": arm, "prereg": prereg, **_git(),
            "eligibility_confirmed_by_owner": True, "eligibility_rule": ELIGIBILITY_RULE, "gate_verdict": report["verdict"],
            "gate_clauses": {k: {kk: vv for kk, vv in v.items() if kk != "per_day"} for k, v in report["clauses"].items()},
            "licence": "derived table from adsb.lol (ODbL-1.0, share-alike)", "licence_doc": info.get("licence_doc"),
            "rebuild_guard": info.get("rebuild_guard"), "expected_price": EXPECTED_PRICE, "transfer_rule": TRANSFER_RULE,
            **{k: info[k] for k in ("n_rows_replaced", "n_values_differ", "mean_shift_s", "rms_shift_vs_base_s", "before_mean_s",
                                    "after_mean_s", "per_day_n", "sse_shift_vs_base_board_mse")},
            "detail_file": f"{dest.stem}.adsb_rows.parquet", "wall_s": round(time.time() - t0, 1), **(extra_meta or {})}
    (sub_dir / f"{dest.stem}.meta.json").write_text(json.dumps(_json_ready(meta), indent=1))
    return dest, meta


def eligible_unmatched_ids(ranking_path: pathlib.Path, days=DAYS) -> set:
    fl = load_storm_flights(ranking_path, days)
    return set(fl.MVT_ID_mvt[~fl.matched].tolist())


def step_ship(base: str, version: int, eligibility_confirmed: bool, detail: str, licence_doc: str,
              report_path: pathlib.Path | None = None, estimates_path: pathlib.Path | None = None, sub_dir: pathlib.Path | None = None,
              ranking_path: pathlib.Path | None = None, template_path: pathlib.Path | None = None,
              root: pathlib.Path = ROOT, e4b: bool = False, validate_path: pathlib.Path | None = None,
              bias_path: pathlib.Path | None = None) -> int:
    """Guards, in order, each refusing the write: the owner's eligibility flag; the licence document in the repo; a
    WORKING gate verdict; the base passes check_submission; the rebuild guard (verify_base against `detail`); build_ship's
    row, dtype and eligibility guards; check_submission on the frame and on the re-read file; no overwrite.
    e4b: the E4b gate outputs (reports/adsb_e4b_storm.json + its estimates), and before anything is read further: the
    report must be an E4b report, the validation file on disk must say WORKING, and the report and EVERY estimate row must
    carry the calibrated constant of reports/adsb_e4b_bias.json. E4's mode refuses an E4b report."""
    log(ELIGIBILITY_RULE)
    if e4b:
        log(f"E4b ({PREREG_E4B}) inherits this eligibility rule unchanged.")
    if not eligibility_confirmed:
        raise SystemExit("refusing to ship: pass --eligibility-confirmed only once the organisers' licence answer is in writing")
    licence_rel = check_licence_doc(licence_doc, root)
    t0 = time.time()
    report_path = report_path or (REPORT_E4B if e4b else REPORT)
    estimates_path = estimates_path or (ESTIMATES_E4B if e4b else ESTIMATES)
    if not report_path.exists() or not estimates_path.exists():
        raise SystemExit(f"run `adsb_storm.py gate{' --e4b' if e4b else ''}` first: need {report_path} and {estimates_path}")
    report = json.loads(report_path.read_text())
    mode = report.get("mode", "e4")
    if mode != ("e4b" if e4b else "e4"):
        raise SystemExit(f"refusing to ship: {report_path.name} is a {mode!r} gate report and this is the "
                         f"{'e4b' if e4b else 'e4'} ship")
    if report.get("verdict") != "WORKING":
        raise SystemExit(f"refusing to ship: the gate verdict is {report.get('verdict')!r}, the ship rule needs WORKING")
    extra_meta = None
    if e4b:
        validation = require_e4b_working(validate_path or VALIDATE_REPORT)
        bias = load_bias(bias_path or BIAS_E4B)
        if report.get("bias_s") != bias or validation["bias"]["bias_s"] != bias:
            raise SystemExit(f"refusing to ship: the gate report ({report.get('bias_s')}) / validation "
                             f"({validation['bias']['bias_s']}) bias is not the calibrated constant {bias}")
        extra_meta = {"bias_s": bias, "bias_source": str((bias_path or BIAS_E4B).name),
                      "validation_verdict": validation["verdict"], "validation_clauses": validation["clauses"]}
    sub_dir = sub_dir or ROOT / "submissions"
    raw = ROOT / "data" / "raw"
    base_path = ROOT / base if not pathlib.Path(base).is_absolute() else pathlib.Path(base)
    if not base_path.exists():
        raise SystemExit(f"--base {base_path} does not exist")
    base_df = pq.read_table(base_path).to_pandas()
    template = pq.read_table(template_path or raw / "submitting.parquet").to_pandas()
    bs.check_submission(base_df, template)
    est = pq.read_table(estimates_path).to_pandas()
    if e4b and not (len(est) and (est.bias_s.to_numpy(dtype="float64") == extra_meta["bias_s"]).all()):
        raise SystemExit(f"refusing to ship: {estimates_path.name} does not carry the calibrated constant "
                         f"{extra_meta['bias_s']} on every row: it is not the output of `gate --e4b`")
    eligible = eligible_unmatched_ids(ranking_path or raw / "ranking.parquet")
    detail_path = ROOT / detail if not pathlib.Path(detail).is_absolute() else pathlib.Path(detail)
    if not detail_path.exists():
        raise SystemExit(f"--detail {detail_path} does not exist")
    guard = verify_base(base_df, pq.read_table(detail_path).to_pandas(), eligible)
    log(f"rebuild guard: {base_path.name} equals {detail_path.name}'s `after` on all {guard['n_checked']:,} {AIRPORT} unmatched storm rows")
    out, info = build_ship(base_df, est, eligible)
    info["rebuild_guard"] = {**guard, "detail_file": detail_path.name}
    info["licence_doc"] = licence_rel
    log(f"{'E4b' if e4b else 'E4'}: replacing {info['n_rows_replaced']:,} {AIRPORT} unmatched storm rows ({info['n_values_differ']:,} values differ); "
        f"mean {info['before_mean_s']:.0f} -> {info['after_mean_s']:.0f} s, RMS shift {info['rms_shift_vs_base_s']:.0f} s, "
        f"sse shift {info['sse_shift_vs_base_board_mse']:.1f} board MSE; per day {info['per_day_n']}")
    dest, meta = write_ship_outputs(out, info, template, version, base_path.name, sub_dir, report, t0,
                                    arm=ARM_E4B if e4b else ARM, prereg=PREREG_E4B if e4b else PREREG, extra_meta=extra_meta)
    print(json.dumps(_json_ready(meta), indent=1))
    print(f"wrote {dest}")
    return 0


# =============================================================================================
# arm E4b (plans/PREREG_adsb_eham_v2_2026_09_10.md): ONE label-referenced bias, validated on 2025-12-27/28
# =============================================================================================

def labels_2025(truth: pd.DataFrame, ids) -> tuple:
    """The ONLY reader of label columns on the E4b path: (BLOCK_TIME as epoch s, TAXITIME_SEC_mvt as float) for `ids`,
    in `ids` order. Refuses (ValueError) if any requested row's take-off (MVT_TIME_UTC_mvt) is missing or not in 2025:
    no 2026 label may be read. Every requested id must be in `truth` exactly once."""
    for c in ("MVT_ID_mvt", "MVT_TIME_UTC_mvt", *LABEL_COLUMNS):
        if c not in truth.columns:
            raise ValueError(f"the label frame needs {c!r}")
    ids = np.asarray(ids, dtype="float64")
    t = truth.set_index("MVT_ID_mvt")
    if t.index.duplicated().any():
        raise ValueError("the label frame repeats an MVT_ID_mvt")
    missing = ~np.isin(ids, t.index.to_numpy(dtype="float64"))
    if missing.any():
        raise ValueError(f"{int(missing.sum())} requested ids are not in the label frame")
    rows = t.loc[ids]
    year = pd.to_datetime(rows.MVT_TIME_UTC_mvt, utc=True).dt.year.to_numpy(dtype="float64")
    bad = ~(year == 2025)
    if bad.any():
        raise ValueError(f"label read refused: {int(bad.sum())} requested rows are not 2025 take-offs "
                         "(no 2026 label may be read; a missing take-off time cannot prove 2025)")
    return to_epoch_s(rows.BLOCK_TIME_UTC_mvt), rows.TAXITIME_SEC_mvt.to_numpy(dtype="float64")


def load_validation_truth(path: pathlib.Path = VALIDATION_TRUTH, days=VALIDATION_DAYS) -> pd.DataFrame:
    """EHAM departures of the validation days WITH their labels (FLIGHT_COLUMNS + LABEL_COLUMNS). Refuses a day that is
    not a 2025 day; a row's day is the UTC date of its take-off, so every returned row is a 2025 take-off."""
    bad = [d for d in days if not str(d).startswith("2025-")]
    if bad:
        raise ValueError(f"labels are read on 2025 days only, not {bad}")
    df = pq.read_table(path, columns=FLIGHT_COLUMNS + list(LABEL_COLUMNS),
                       filters=[("ADEP_mvt", "=", AIRPORT), ("PHASE_mvt", "=", "DEP")]).to_pandas()
    day = pd.to_datetime(df.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    df = df[day.isin(list(days))].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"{path} holds no {AIRPORT} departure on {list(days)}")
    return df


def calibrate_bias(truth: pd.DataFrame, samples: pd.DataFrame, day: str = CENSUS_DAY) -> dict:
    """E4b's ONE bias constant: median(sensor - BLOCK_TIME) over the census day's matched EHAM rows where the gate fires,
    the sensor being E4's procedure exactly (centroids at AOBT_3, leave-one-flight-out, last sample within 100 m before
    take-off). Only the census day's rows are used, whatever else `truth` / `samples` hold; it refuses any other day."""
    if day != CENSUS_DAY:
        raise ValueError(f"E4b's bias is calibrated on the census day {CENSUS_DAY} only, not {day!r}")
    lat0, lon0 = ax.APTS[AIRPORT]
    tracks = tracks_frame(samples[samples.airport == AIRPORT] if "airport" in samples.columns else samples, lat0, lon0)
    fl = flights_frame(truth, [day])
    est, _ = measure(fl, tracks, [day], bias_mode="constant", bias_s=0.0)       # the raw sensor does not depend on the bias
    raw = est.offblock_raw_s.to_numpy()
    m = est.matched.to_numpy() & np.isfinite(raw)
    if not m.any():
        raise ValueError(f"no matched {AIRPORT} row of {day} has a gated sensor reading: no constant can be calibrated")
    block_s, _ = labels_2025(truth, est.MVT_ID_mvt.to_numpy()[m])
    if not np.isfinite(block_s).all():
        raise ValueError("a calibration row has no BLOCK_TIME label")
    d = raw[m] - block_s
    q25, q50, q75 = (float(v) for v in np.percentile(d, [25, 50, 75]))
    a3 = fl.set_index("MVT_ID_mvt").ref_s.loc[est.MVT_ID_mvt.to_numpy()[m]].to_numpy()
    n_matched = int(est.matched.sum())
    return {"arm": ARM_E4B, "prereg": PREREG_E4B, "airport": AIRPORT, "calibration_day": day,
            "bias_s": float(np.median(d)), "n": int(m.sum()), "iqr_s": q75 - q25, "q25_s": q25, "q75_s": q75,
            "n_matched": n_matched, "coverage": m.sum() / n_matched,
            "rmse_after_bias_in_sample_s": rmse(d - np.median(d)),
            "median_aobt3_minus_block_s": float(np.median(a3 - block_s)),
            "procedure": "median over matched gated census rows of (last ADS-B sample within 100 m of the row's leave-one-"
                         "flight-out AOBT_3 stand centroid before take-off - BLOCK_TIME_UTC_mvt)"}


def load_bias(path: pathlib.Path = BIAS_E4B) -> float:
    """The calibrated constant, refusing a file that is not E4b's census-day calibration."""
    if not path.exists():
        raise SystemExit(f"run `adsb_storm.py calibrate` first: {path} does not exist")
    j = json.loads(path.read_text())
    if j.get("prereg") != PREREG_E4B or j.get("calibration_day") != CENSUS_DAY:
        raise ValueError(f"{path.name} is not E4b's {CENSUS_DAY} calibration (prereg {j.get('prereg')!r}, "
                         f"day {j.get('calibration_day')!r})")
    b = j.get("bias_s")
    if not isinstance(b, (int, float)) or not np.isfinite(b):
        raise ValueError(f"{path.name} carries no finite bias_s")
    return float(b)


def clause_v1(err_block: np.ndarray, matched: np.ndarray, rmse_max: float = V1_RMSE_S, cov_min: float = V1_MIN_COVERAGE) -> dict:
    """err_block: (sensor - bias - BLOCK_TIME) per row, NaN where the gate did not fire. RMSE over matched gated rows
    <= rmse_max with coverage (gated / all matched) >= cov_min."""
    c = clause_c1(err_block, matched, rmse_max, cov_min)
    c["clause"] = "V1"
    return c


def clause_v2(taxi_hat, s1c, y, min_rows: int = V2_MIN_ROWS) -> dict:
    """On the V2 rows (unmatched, gate fired, join unambiguous): RMSE(taxi_hat - y) strictly below RMSE(S1C - y) on the
    same rows; fewer than min_rows rows -> INCONCLUSIVE. Every row must carry all three values."""
    taxi_hat, s1c, y = (np.asarray(v, dtype="float64") for v in (taxi_hat, s1c, y))
    if not (len(taxi_hat) == len(s1c) == len(y)):
        raise ValueError("V2 arrays differ in length")
    if not (np.isfinite(taxi_hat).all() and np.isfinite(s1c).all() and np.isfinite(y).all()):
        raise ValueError("every V2 row must carry a finite taxi_hat, S1C and y")
    n = int(len(y))
    r_hat, r_s1c = rmse(taxi_hat - y), rmse(s1c - y)
    if n < min_rows:
        status, ok = "INCONCLUSIVE", False
    else:
        ok = bool(r_hat < r_s1c)
        status = "PASS" if ok else "FAIL"
    return {"clause": "V2", "n_rows": n, "min_rows": min_rows, "rmse_taxi_hat_s": r_hat, "rmse_s1c_s": r_s1c,
            "status": status, "pass": ok}


def clause_v3(joined: pd.DataFrame, max_share: float = V3_MAX_SHARE) -> dict:
    """E4's C0 rule on the validation days' unmatched joins, at the prereg's 5%."""
    c = clause_c0(joined, max_share)
    c["clause"] = "V3"
    return c


def verdict_e4b(v1: dict, v2: dict, v3: dict) -> str:
    """WORKING iff V1-V3 pass; NOT WORKING iff V1 fails; else INCONCLUSIVE."""
    if v1["pass"] and v2["pass"] and v3["pass"]:
        return "WORKING"
    if not v1["pass"]:
        return "NOT WORKING"
    return "INCONCLUSIVE"


def load_s1c(path: pathlib.Path = S1C_PATH) -> pd.DataFrame:
    """S1C's out-of-month predictions (E3C). BC-2 tripwire: the record must be stamped convention=taxi_time in its
    parquet metadata, so `S1C` compares with the label directly."""
    meta = pq.read_schema(path).metadata or {}
    if meta.get(b"convention") != b"taxi_time":
        raise ValueError(f"{path.name} is not stamped convention=taxi_time (found {meta.get(b'convention')!r}): "
                         "BC-2, the S1C column cannot be compared with the label as is")
    df = pq.read_table(path, columns=["MVT_ID_mvt", "fold", "date", "ADEP_mvt", "y", "hprox_med", "S1C"]).to_pandas()
    if df.MVT_ID_mvt.duplicated().any():
        raise ValueError(f"{path.name} repeats an MVT_ID_mvt")
    return df


def validate_e4b(truth: pd.DataFrame, tracks: pd.DataFrame, bias_s: float, s1c: pd.DataFrame, days=VALIDATION_DAYS) -> tuple:
    """(per-row table, report). E4b's procedure on the validation days with the census constant `bias_s` (never
    re-estimated here), then V1 (matched, vs BLOCK_TIME), V2 (unmatched gated rows vs S1C) and V3 (join), the verdict."""
    if CENSUS_DAY in days:
        raise ValueError(f"the calibration day {CENSUS_DAY} is never a validation day")
    bad = [d for d in days if not str(d).startswith("2025-")]
    if bad:
        raise ValueError(f"validation reads labels and runs on 2025 days only, not {bad}")
    fl = flights_frame(truth, days)
    est, info = measure(fl, tracks, days, bias_mode="constant", bias_s=bias_s)
    ids = est.MVT_ID_mvt.to_numpy()
    block_s, y = labels_2025(truth, ids)
    matched, fired, amb = est.matched.to_numpy(), est.gate_fired.to_numpy(), est.join_ambiguous.to_numpy()
    raw, day, mvt_s = est.offblock_raw_s.to_numpy(), est.day.to_numpy(), fl.mvt_s.to_numpy()
    hat = raw - est.bias_s.to_numpy()
    in_v1 = matched & fired
    if not np.isfinite(block_s[in_v1]).all():
        raise ValueError("a V1 row has no BLOCK_TIME label")
    err_block = np.where(in_v1, hat - block_s, np.nan)
    taxi_hat = np.where(fired, np.maximum(mvt_s - hat, TAXI_FLOOR_S), np.nan)
    in_v2 = ~matched & fired & ~amb
    # S1C on the unmatched rows: present, out-of-month, the same airport-day, and its y IS the label (BC-2)
    s = s1c.set_index("MVT_ID_mvt")
    s1c_v = s.S1C.reindex(ids).to_numpy(dtype="float64")
    hprox = s.hprox_med.reindex(ids).to_numpy(dtype="float64")
    if in_v2.any():
        v2_ids = ids[in_v2]
        if not np.isin(v2_ids, s.index.to_numpy(dtype="float64")).all():
            raise ValueError("a V2 row has no S1C prediction in the out-of-month record")
        sv = s.loc[v2_ids]
        if not ((sv.fold == "lomo").all() and (sv.ADEP_mvt == AIRPORT).all() and (sv.date.to_numpy() == day[in_v2]).all()):
            raise ValueError("a V2 row's S1C record is not the out-of-month (lomo) prediction of the same EHAM day")
        if not np.array_equal(sv.y.to_numpy(dtype="float64"), y[in_v2]):
            raise ValueError("the S1C record's y differs from TAXITIME_SEC_mvt on a V2 row: not the label's convention (BC-2)")
    v1 = clause_v1(err_block, matched)
    v2 = clause_v2(taxi_hat[in_v2], s1c_v[in_v2], y[in_v2])
    v3 = clause_v3(info["joined"])
    v = verdict_e4b(v1, v2, v3)
    d_block = np.where(in_v1, raw - block_s, np.nan)
    per_day = {}
    for dd in days:
        on = day == dd
        g, m2 = on & in_v1, on & in_v2
        n_m = int((on & matched).sum())
        per_day[dd] = {"n_matched": n_m, "n_gated": int(g.sum()), "coverage": g.sum() / n_m if n_m else None,
                       "rmse_vs_block_s": rmse(err_block[g]),
                       "median_sensor_minus_block_s": float(np.median(d_block[g])) if g.any() else None,
                       "n_unmatched": int((on & ~matched).sum()), "n_unmatched_gated": int((on & ~matched & fired).sum()),
                       "n_unmatched_ambiguous": int((on & ~matched & amb).sum()), "n_v2_rows": int(m2.sum()),
                       "v2_rmse_taxi_hat_s": rmse(taxi_hat[m2] - y[m2]), "v2_rmse_s1c_s": rmse(s1c_v[m2] - y[m2])}
    busy = in_v2 & (hprox >= 1500.0)
    rows = pd.DataFrame({"MVT_ID_mvt": ids, "day": day, "matched": matched, "stand": est.stand.to_numpy(),
                         "runway": est.runway.to_numpy(), "track_id": est.track_id.to_numpy(),
                         "n_candidates": est.n_candidates.to_numpy(), "join_ambiguous": amb, "gate_fired": fired,
                         "mvt_s": mvt_s, "aobt3_s": fl.ref_s.to_numpy(), "block_s": block_s, "offblock_raw_s": raw,
                         "bias_s": est.bias_s.to_numpy(), "offblock_hat_s": hat, "err_vs_block_s": err_block,
                         "taxi_hat": taxi_hat, "y": y, "s1c": s1c_v, "hprox_med": hprox, "in_v1": in_v1, "in_v2": in_v2})
    rep = {"arm": ARM_E4B, "prereg": PREREG_E4B, "airport": AIRPORT, "days": list(days), "calibration_day": CENSUS_DAY,
           "verdict": v, "clauses": {"V1": v1, "V2": v2, "V3": v3}, "per_day": per_day,
           "bias": {"bias_s": float(bias_s), "source": str(BIAS_E4B.relative_to(ROOT))},
           "diagnostics": {
               "note": "reported only, never decisional",
               "rmse_v1_rows_vs_aobt3_s": rmse(hat[in_v1] - fl.ref_s.to_numpy()[in_v1]),
               "median_sensor_minus_block_s_validation_minus_census_constant_s":
                   float(np.nanmedian(d_block) - bias_s) if in_v1.any() else None,
               "v2_rows_hprox_med_ge_1500": {"n": int(busy.sum()), "rmse_taxi_hat_s": rmse(taxi_hat[busy] - y[busy]),
                                             "rmse_s1c_s": rmse(s1c_v[busy] - y[busy])}},
           "join": {"n_matched_joined_by_callsign": info["n_matched_joined"], "n_fixes": info["n_fixes"],
                    "n_rows_with_centroid": info["n_rows_with_centroid"], "runways_learned": info["runways_learned"],
                    "takeoff_offset_s": info["takeoff_offset_s"], "n_takeoff_events": info["n_takeoff_events"]},
           "n_rows": int(len(rows)), "judgement_calls": JUDGEMENT_CALLS_E4B, "eligibility": ELIGIBILITY_RULE}
    return rows, rep


def require_e4b_working(path: pathlib.Path = VALIDATE_REPORT) -> dict:
    """The E4b validation report on disk, refusing unless its verdict is WORKING."""
    if not path.exists():
        raise SystemExit(f"E4b: {path} does not exist; run `adsb_storm.py validate` first. The 2026 path runs only if WORKING")
    j = json.loads(path.read_text())
    if j.get("prereg") != PREREG_E4B:
        raise SystemExit(f"E4b: {path.name} is not an E4b validation report")
    if j.get("verdict") != "WORKING":
        raise SystemExit(f"E4b: the validation verdict is {j.get('verdict')!r}; the 2026 path runs only if WORKING")
    return j


def gate_report_e4b(flights: pd.DataFrame, tracks: pd.DataFrame, bias_s: float, validation: dict, days=DAYS) -> tuple:
    """(estimates, report) on the 2026 storm days with E4b's constant. The in-file diagnostic (sensor vs AOBT_3) and the
    join ambiguity are reported only; the verdict is the validation verdict."""
    est, info = measure(flights, tracks, days, bias_mode="constant", bias_s=bias_s)
    matched, day, err, fired = est.matched.to_numpy(), est.day.to_numpy(), info["err_corr"], est.gate_fired.to_numpy()
    g = matched & np.isfinite(err)
    n_m = int(matched.sum())
    join = {k: v for k, v in clause_c0(info["joined"]).items() if k not in ("clause", "pass", "threshold")}
    per_day = {}
    for d in days:
        on = day == d
        um = on & ~matched
        n_md = int((on & matched).sum())
        per_day[d] = {"n_matched": n_md, "n_gated": int((on & g).sum()), "coverage": (on & g).sum() / n_md if n_md else None,
                      "rmse_vs_aobt3_s": rmse(err[on & g]), "n_unmatched": int(um.sum()),
                      "n_unmatched_gated": int((um & fired).sum()), "n_unmatched_ambiguous": int((um & est.join_ambiguous.to_numpy()).sum()),
                      "unmatched_taxi_hat_mean_s": float(np.nanmean(est.taxi_hat.to_numpy()[um & fired])) if (um & fired).any() else None}
    rep = {"arm": ARM_E4B, "prereg": PREREG_E4B, "mode": "e4b", "airport": AIRPORT, "days": list(days), "bias_s": float(bias_s),
           "bias_source": str(BIAS_E4B.relative_to(ROOT)), "verdict": validation["verdict"],
           "validation": {"file": str(VALIDATE_REPORT.relative_to(ROOT)), "verdict": validation["verdict"],
                          "bias_s": validation["bias"]["bias_s"]},
           "clauses": validation["clauses"],
           "diagnostic_2026": {"note": "reported only: AOBT_3 is ~269 s from the true off-block, so it cannot be a pass/fail bar",
                               "n_matched": n_m, "n_gated": int(g.sum()), "coverage": g.sum() / n_m if n_m else None,
                               "rmse_vs_aobt3_s": rmse(err[g])},
           "join_2026": join, "per_day": per_day, "n_rows": int(len(est)), "n_gate_fired": int(fired.sum()),
           "n_unmatched_gate_fired": int((fired & ~matched).sum()), "judgement_calls": JUDGEMENT_CALLS_E4B,
           "eligibility": ELIGIBILITY_RULE}
    return est, rep


def step_calibrate(path: pathlib.Path = BIAS_E4B, truth: pd.DataFrame | None = None, samples: pd.DataFrame | None = None) -> int:
    """Compute the constant on the census data (data/adsb/) and write it once. A re-run must reproduce bias_s, n and
    iqr_s exactly (it then leaves the file untouched); a different result refuses."""
    if truth is None or samples is None:
        truth, samples = load_census()
    cal = calibrate_bias(truth, samples)
    c3 = census_c3(truth, samples)
    cal["cross_check_census_c3_block_calibrated_bias_s"] = c3["diagnostic_block_calibrated"]["bias_s"]
    if cal["cross_check_census_c3_block_calibrated_bias_s"] != cal["bias_s"]:
        raise AssertionError("calibrate and census_c3's BLOCK-calibrated diagnostic disagree on the same rows")
    if path.exists():
        old = json.loads(path.read_text())
        same = all(old.get(k) == _json_ready(cal[k]) for k in ("bias_s", "n", "iqr_s"))
        if not same:
            raise SystemExit(f"refusing to overwrite {path}: it holds bias_s={old.get('bias_s')} n={old.get('n')}, "
                             f"this run gives {cal['bias_s']} n={cal['n']}; the constant is locked once written")
        log(f"{path.name} reproduced exactly (bias_s {cal['bias_s']:.3f} s, n {cal['n']}); left untouched")
        return 0
    cal["written_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
    path.write_text(json.dumps(_json_ready(cal), indent=1))
    log(f"E4b bias constant = {cal['bias_s']:.3f} s (n {cal['n']}, IQR {cal['iqr_s']:.1f} s [{cal['q25_s']:.1f}, "
        f"{cal['q75_s']:.1f}], coverage {cal['coverage']:.3f}); wrote {path}")
    return 0


def step_validate(report_path: pathlib.Path = VALIDATE_REPORT, rows_path: pathlib.Path = VALIDATION_ROWS,
                  bias_path: pathlib.Path = BIAS_E4B, truth_path: pathlib.Path = VALIDATION_TRUTH,
                  s1c_path: pathlib.Path = S1C_PATH, adsb_dir: pathlib.Path = ADSB_DIR) -> int:
    """V1-V3 on the two validation days, written once: the verdict is never revised."""
    for p in (report_path, rows_path):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}: the E4b verdict is written once and never revised")
    bias = load_bias(bias_path)
    truth = load_validation_truth(truth_path)
    tracks = load_storm_tracks(VALIDATION_DAYS, adsb_dir)
    rows, rep = validate_e4b(truth, tracks, bias, load_s1c(s1c_path))
    cal = json.loads(bias_path.read_text())
    rep["bias"] = {"bias_s": bias, "source_file": bias_path.name, "calibration_day": cal["calibration_day"],
                   "n": cal.get("n"), "iqr_s": cal.get("iqr_s")}
    rows.to_parquet(rows_path, index=False)
    rep["written"] = {"report": report_path.name, "rows": rows_path.name}
    report_path.write_text(json.dumps(_json_ready(rep), indent=1))
    c = rep["clauses"]
    log(f"bias constant {bias:.3f} s (from {bias_path.name}, calibrated on {CENSUS_DAY})")
    log(f"V1 RMSE {c['V1']['rmse_s']} s, coverage {c['V1']['coverage']} -> {c['V1']['pass']}")
    log(f"V2 n {c['V2']['n_rows']}: taxi_hat {c['V2']['rmse_taxi_hat_s']} vs S1C {c['V2']['rmse_s1c_s']} -> {c['V2']['status']}")
    log(f"V3 share {c['V3']['share']} (n with candidate {c['V3']['n_with_candidate']}) -> {c['V3']['pass']}")
    for d, r in rep["per_day"].items():
        log(f"  {d}: {r}")
    log(f"VERDICT: {rep['verdict']}")
    log(f"wrote {report_path} and {rows_path}")
    return 0


def step_gate_e4b(ranking_path: pathlib.Path | None = None, adsb_dir: pathlib.Path = ADSB_DIR,
                  bias_path: pathlib.Path = BIAS_E4B, validate_path: pathlib.Path = VALIDATE_REPORT,
                  report_path: pathlib.Path = REPORT_E4B, estimates_path: pathlib.Path = ESTIMATES_E4B) -> int:
    """The 2026 path with E4b's constant; refuses unless the validation verdict is WORKING and was reached with the same
    constant. Reads the storm flights through load_storm_flights (FLIGHT_COLUMNS only: no label)."""
    validation = require_e4b_working(validate_path)
    bias = load_bias(bias_path)
    if validation["bias"]["bias_s"] != bias:
        raise SystemExit(f"E4b: the validation ran with bias {validation['bias']['bias_s']}, {bias_path.name} holds {bias}")
    t0 = time.time()
    flights = load_storm_flights(ranking_path or ROOT / "data" / "raw" / "ranking.parquet")
    tracks = load_storm_tracks(DAYS, adsb_dir)
    est, rep = gate_report_e4b(flights, tracks, bias, validation)
    rep["wall_s"] = round(time.time() - t0, 1)
    rep["written"] = {"report": report_path.name, "estimates": estimates_path.name}
    est.to_parquet(estimates_path, index=False)
    report_path.write_text(json.dumps(_json_ready(rep), indent=1))
    dg = rep["diagnostic_2026"]
    log(f"E4b 2026: bias {bias:.3f} s; matched gated {dg['n_gated']} of {dg['n_matched']}, RMSE vs AOBT_3 {dg['rmse_vs_aobt3_s']} "
        f"(diagnostic only); unmatched gated {rep['n_unmatched_gate_fired']}; validation verdict {rep['verdict']}")
    log(f"wrote {report_path} and {estimates_path}")
    return 0


# =============================================================================================
# main
# =============================================================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    e = sub.add_parser("extract", help="fetch one registered day's adsb.lol archive, keep the EHAM box, delete the archive")
    e.add_argument("--day", required=True)
    e.add_argument("--probe", action="store_true", help="first 60 MB only; prints counts, writes nothing")
    sub.add_parser("census", help="C3 alone, on data/adsb/ 2025-01-09")
    sub.add_parser("calibrate", help="E4b: the ONE bias constant on the 2025-01-09 census -> reports/adsb_e4b_bias.json")
    sub.add_parser("validate", help="E4b: V1-V3 on 2025-12-27/28, the verdict (write-once)")
    g = sub.add_parser("gate", help="C0-C3, the verdict, reports/adsb_storm.json, data/adsb/adsb_storm_estimates.parquet")
    g.add_argument("--e4b", action="store_true", help="E4b: the label-referenced constant; reports/adsb_e4b_storm.json")
    s = sub.add_parser("ship", help="splice the gated unmatched estimates into a base submission (eligibility-gated)")
    s.add_argument("--base", required=True)
    s.add_argument("--version", type=int, required=True)
    s.add_argument("--detail", required=True,
                   help="the detail file written with the version that last set the EHAM unmatched rows (v10: v9's unm_rows)")
    s.add_argument("--licence-doc", required=True, help="the repo file documenting the derived table's ODbL licence")
    s.add_argument("--eligibility-confirmed", action="store_true")
    s.add_argument("--e4b", action="store_true", help="E4b: ship the `gate --e4b` estimates (needs a WORKING validation)")
    a = ap.parse_args(argv)
    if a.step == "extract":
        return step_extract(a.day, a.probe)
    if a.step == "census":
        return step_census()
    if a.step == "calibrate":
        return step_calibrate()
    if a.step == "validate":
        return step_validate()
    if a.step == "gate":
        return step_gate_e4b() if a.e4b else step_gate()
    return step_ship(a.base, a.version, a.eligibility_confirmed, a.detail, a.licence_doc, e4b=a.e4b)


if __name__ == "__main__":
    sys.exit(main())
