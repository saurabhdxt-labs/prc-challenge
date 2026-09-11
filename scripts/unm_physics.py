"""`scripts/unm_physics.py` -- arm E5 of plans/PREREG_unm_physics_2026_09_11.md: a physics-informed unmatched body.

    python scripts/unm_physics.py lomo            # the heavy step: 12 months from data/raw, ~3-4 GB, run ALONE
    python scripts/unm_physics.py score           # the light step: clauses C1-C7 from the stored parquet + 2026 witness
    python scripts/unm_physics.py ship --base submissions/merry-quicksand_v10.parquet --version N   # ONLY after WORKING
    python scripts/unm_physics.py lomo --smoke    # synthetic world, code path only, NOT a result
    python scripts/unm_physics.py score --smoke

H-E5: adding take-off-anchored physical inputs to the unmatched body lowers non-Rome unmatched error out of month.

Arms (12-fold LOMO over 2025, E3C's folds and frames; `unm_congestion` / `unm_congestion_resid` are composed, never edited):
  S1C       exactly E3C's (what v9 / v10 ship) -- GUARD: equals data/cache_stand/unm_congestion_lomo.parquet's S1C on every
            row to <= 1e-6 s (S1, the per-seed S1C arms, p_hat, nf_cells and nf_fit_c are held to the same bar);
  E5w       S1C's body + the WEATHER block (reported: the attribution arm);
  E5        S1C's body + WEATHER + PHYSICS (decisional);
  E5_fog15  E5 with fog counted as visible moisture only at visibility <= 1.5 km (registered sensitivity, reported);
  E5_lt3    E5 with `< +3 C` in place of `<= +3 C` in the de-icing rule (registered sensitivity, reported).
  Every arm keeps S1C's p, nf_cells, seeds, params and winsorisation: nf = nf_hybrid(nf_cells, mean over seeds of the
  arm's body); pred = mixture(p, sp, nf). The body is `PhysicsRegressor`: S1C's regressor (NF_NUMERIC + the witness, the
  five SMOOTH-smoothed encodings, NF_PARAMS, the 3,000 s winsorisation) with the extra numeric inputs appended and, in
  E5, two extra encodings (AIRCRAFT_TYPE_mvt, airline x ADEP) fitted on the fold's non-fill rows OUTSIDE the regressor's
  early-stopping rows (BC-1-clean, reports/bug_classes.md).

WEATHER (take-off-anchored, serve-time): an as-of join of data/weather_obs/weather_obs_v2.0.1.parquet (prc/weather.py
v2.0.1, sha256 pinned) on station == ADEP_mvt: the latest observation with valid <= MVT_TIME_UTC_mvt, NaN when that
observation is more than 3 h old. Inputs: temp_c, dewspread_c, vis_km, wind_kt, gust_kt, wx_precip, wx_frozen, wx_sn,
wx_fzra, wx_fzdz, wx_fzfg, wx_fg, wx_br, wx_intensity, deicing_condition, and deicing_condition x de-icing airport
(EHAM, EDDM, EDDF, LSZH).
PHYSICS: stand_ab.UNMATCHED_QUEUE_FEATS (q_dep_tko_sym15, q_rwy_tko_sym10, q_arr_taxiin_sym30_tko, q_rwy_ambient_proxy)
and the stand history prev_arr_gap, prev_arr_taxiin, prev_dep_gap, all anchored on the row's own take-off and built by
stand_ab's unmatched builder (fresh, from data/raw, at run time); AIRCRAFT_TYPE_mvt and airline x ADEP as encodings.
Any registered input that cannot be built for unmatched rows is DROPPED and listed -- never imputed from matched rows.

Clauses (locked; non-LIRF LOMO rows; g = SE(S1C) - SE(E5), raw labels):
  C1 date-block bootstrap (calendar dates, 2,000 draws, seed 0) of sum g: lower 95% bound > 0
  C2 EVENT-EXCLUDED 2026 price >= +300 board MSE (drop the ONE airport-month with the largest |sum g|; E3C's bins, the
     < 30-rows-borrows-next-lower rule and the 2026 witness bin counts / 344,841). The full price is reported only.
  C3 sum g > 0 in >= 8 of 12 months          C4 sum g > 0 at >= 6 of the 9 non-LIRF airports
  C5 sum g over hprox_med < 1,100 >= -10% of the pooled sum g        C6 sum g over y <= 10,800 > 0
  C7 pooled gain > 2 x sd of the three single-seed gains
Verdict: WORKING iff all seven pass; NOT WORKING iff C1 fails; else INCONCLUSIVE. Never revised.
C1 and C3-C7 are `unm_congestion.clauses`' arithmetic on a renamed view (S1 := S1C, S1C := E5); C2 is
`unm_congestion_resid.event_excluded_price` (E3R's pattern, the same +300 bar).

Outputs. `lomo`: data/cache_stand/unm_physics_lomo.parquet (TAXI-TIME convention, git sha, weather version + sha256,
feature lists and the dropped-input audit stamped in the parquet metadata -- reports/bug_classes.md BC-2) and
reports/unm_physics_lomo.log. `score`: reports/unm_physics.json and reports/unm_physics_score.log. `ship`:
submissions/<team>_vN.parquet + .meta.json + .unm_rows.parquet. `--smoke` writes under data/smoke_unm_physics/ only,
uses 200 draws and brackets the log with the banner.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import glob
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor as HGR
from sklearn.model_selection import train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import unm_congestion as uc  # noqa: E402
import unm_congestion_resid as ucr  # noqa: E402
import unm_congestion_ship as us  # noqa: E402
import stand_ab as S  # noqa: E402
from prc import weather as WX  # noqa: E402

sf, bs = uc.sf, uc.bs
RAW, REPORTS, CACHE = uc.RAW, uc.REPORTS, uc.CACHE
SMOKE_DIR = ROOT / "data" / "smoke_unm_physics"
UCACHE = S.UCACHE                                       # data/cache_unmatched: cross-checked, never the source
REFERENCE = ucr.REFERENCE                               # E3C's stored LOMO predictions (the S1C guard)
E3C_WITNESS_2026 = ucr.E3C_WITNESS_2026
PREREG = "plans/PREREG_unm_physics_2026_09_11.md"
PARENT_PREREG = uc.PREREG
TAG = "E5"

# ---- the registered design, frozen ----
WEATHER_OBS = ROOT / "data" / "weather_obs" / "weather_obs_v2.0.1.parquet"
WEATHER_VERSION = "2.0.1"
WEATHER_SHA256 = "cb05439b6b9a57c159fd78049821821c6ac828087d2380c12d1c6f7f718022a8"
MAX_OBS_AGE = pd.Timedelta(hours=3)                    # NaN beyond 3 h
REG_WEATHER = ["temp_c", "dewspread_c", "vis_km", "wind_kt", "gust_kt", "wx_precip", "wx_frozen", "wx_sn", "wx_fzra",
               "wx_fzdz", "wx_fzfg", "wx_fg", "wx_br", "wx_intensity", "deicing_condition"]
DEICE_AIRPORTS = ("EDDF", "EDDM", "EHAM", "LSZH")      # fixed by the prereg from the atlas / literature
INTERACTION = "deice_x_airport"
REG_QUEUE = ["q_dep_tko_sym15", "q_rwy_tko_sym10", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy"]
REG_STANDHIST = ["prev_arr_gap", "prev_arr_taxiin", "prev_dep_gap"]
REG_ENCODED = ["AIRCRAFT_TYPE_mvt", "airline_adep"]
if REG_QUEUE != list(S.UNMATCHED_QUEUE_FEATS):
    raise ImportError(f"the prereg's queue block {REG_QUEUE} is no longer stand_ab.UNMATCHED_QUEUE_FEATS {S.UNMATCHED_QUEUE_FEATS}")
if not set(REG_STANDHIST) <= set(S.STANDHIST):
    raise ImportError(f"the prereg's stand history {REG_STANDHIST} is not in stand_ab.STANDHIST {S.STANDHIST}")
PHYSICS_COLUMNS = [*REG_QUEUE, *REG_STANDHIST]
# the de-icing rule's thresholds as prc/weather.py v2.0.1 applies them, and the registered sensitivities
MOIST_C = 3.0                                          # dc_moist_cold: temp_c <= +3 C (sensitivity: < +3 C)
FOG_VIS_KM = 1.5                                       # visibility clause (sensitivity: fog only at <= 1.5 km)
TEMP_DECIMALS = 6                                      # temperatures compared at 1e-6 C (see judgement call 6)
if (WX.DEICE_MOIST_C, WX.DEICE_VIS_KM) != (MOIST_C, FOG_VIS_KM):
    raise ImportError("prc.weather's de-icing thresholds are no longer the ones the table v2.0.1 was built with")
VARIANTS = ("fog15", "lt3")
OBS_NEEDED = ["station", "valid", *REG_WEATHER, "dewpoint_c", "dc_moist_cold", "dc_frost"]
JOIN_COLUMNS = [*REG_WEATHER, *[f"deicing_condition__{v}" for v in VARIANTS], "dc_frost_float_defect"]
INTERACTIONS = [INTERACTION, *[f"{INTERACTION}__{v}" for v in VARIANTS]]
ROW_INPUTS = [*JOIN_COLUMNS, *INTERACTIONS, "obs_age_s", *PHYSICS_COLUMNS]
ADDED_COLUMNS = [*JOIN_COLUMNS, "obs_valid", "obs_age_s", *INTERACTIONS, *PHYSICS_COLUMNS, "airline_adep"]

SEEDS = uc.SEEDS                                       # (0, 1, 2)
N_BOOT, BOOT_SEED = uc.N_BOOT, uc.BOOT_SEED            # C1: 2,000 date-block draws, seed 0
C2_MIN_MSE = 300.0                                     # C2: the event-excluded price bar
if ucr.C2_MIN_MSE != C2_MIN_MSE:
    raise ImportError(f"unm_congestion_resid.event_excluded_price prices against {ucr.C2_MIN_MSE}; the prereg's bar is {C2_MIN_MSE}")
N_SCORED_2026 = uc.N_SCORED_2026                       # 344,841
CONTROL, DECISIONAL = "S1C", "E5"
ARMS = ("E5w", "E5", "E5_fog15", "E5_lt3")
REPORTED_ARMS = ("E5w", "E5_fog15", "E5_lt3")
ROME, FOLD_A, MONSTER_S, CALM_S = uc.ROME, uc.FOLD_A, uc.MONSTER_S, uc.CALM_S
REPRO_TOL_S = uc.REPRO_TOL_S
GUARD_COLUMNS = ("S1C", "S1", *[f"S1C_seed{s}" for s in SEEDS], "p_hat", "nf_cells", "nf_fit_c")
CONVENTION = uc.CONVENTION                             # "taxi_time"
SKLEARN_VERIFIED = "1.8.0"                             # the early-stopping split below is verified against this release
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
REGISTERED_BASE = "merry-quicksand_v10.parquet"        # the ship rule: v_next = v10 + E5 on non-LIRF unmatched rows
SMOKE_BANNER = uc.SMOKE_BANNER
SMOKE = dict(n_boot=200)
PLANT_DEICE_S = 400.0                                  # smoke only: y += 400 s on non-fill rows with deice_x_airport == 1
PLANT_QUEUE_S = 15.0                                   # smoke only: y += 15 s per q_dep_tko_sym15 above 15
PRED_COLUMNS = [*uc.PRED_COLUMNS, *ROW_INPUTS, *[f"nf_fit_{a}" for a in ARMS],
                *[c for a in ARMS for c in (a, *[f"{a}_seed{s}" for s in SEEDS])]]
META_KEY = b"unm_physics"
LOMO_NAME = "unm_physics_lomo.parquet"
JSON_NAME, LOMO_LOG, SCORE_LOG = "unm_physics.json", "unm_physics_lomo.log", "unm_physics_score.log"
DETAIL_COLUMNS = ["MVT_ID_mvt", "ADEP_mvt", "month", "sp", "hprox_med", "hprox_n", "bin", "deicing_condition",
                  INTERACTION, "obs_age_s", "p", "before", "after"]
_PY = "PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 19 /opt/homebrew/bin/python3.11 -B -u"
COMMAND_LOMO = (f"cd ~/Projects/prc-challenge && {_PY} scripts/unm_physics.py lomo "
                "> reports/unm_physics_lomo.console.log 2>&1; echo \"exit $?\"")
COMMAND_SCORE = (f"cd ~/Projects/prc-challenge && {_PY} scripts/unm_physics.py score "
                 "> reports/unm_physics_score.console.log 2>&1; echo \"exit $?\"")
COMMAND_SHIP = (f"cd ~/Projects/prc-challenge && {_PY} scripts/unm_physics.py ship --base submissions/{REGISTERED_BASE} "
                "--version N > reports/unm_physics_ship.console.log 2>&1; echo \"exit $?\"")
JUDGEMENT_CALLS = [
    "prev_dep_gap is INCLUDED: on an unmatched row it is take-off minus the latest NM off-block of ANOTHER (matched) departure "
    "from the same stand; the row's own AOBT_3 (null) is never read -- read as 'computed without [the row's] AOBT_3', the way "
    "the registered queue block and the witness read their neighbours' AOBT_3. (On a MATCHED row the same name is ~ its own "
    "proxy; only unmatched rows are used here, in training and at serve time.)",
    "prev_dep_gap < 0 is the builder's 'no earlier matched off-block at this stand in the month' case (it indexes the first "
    "LATER off-block); the input does not exist there and is set to NaN (counts in the metadata); nothing else is altered",
    "the queue and stand-history inputs are built FRESH by stand_ab.build_unmatched_month / build_unmatched_ranking (the current "
    "code, the same builder for 2025 and 2026) and cross-checked against data/cache_unmatched (reported, not used)",
    "weather as-of: pd.merge_asof backward per station, exact matches allowed (valid == MVT_TIME counts), tolerance 3 h "
    "inclusive (an observation exactly 3 h old is used, older is NaN); a row with no MVT_TIME or an unknown station reads NaN",
    "deice_x_airport is the Kleene AND of deicing_condition and the de-icing-airport flag (prc/weather.py's three-valued logic): "
    "0 at the other airports even where the condition is unknown, NaN only at a de-icing airport with an unknown condition",
    "the two de-icing sensitivities are recomputed from the table's own fields with temperatures compared at 1e-6 C: the "
    "F->C conversion stores a reported +3 C as 2.999999999999999, so a literal `< 3` would equal `<= 3` on all such "
    "observations; the default rule recomputed this way equals the table's dc_moist_cold and deicing_condition on every "
    "observation (a guard); dc_frost is the table's in every variant",
    "fog15: fog counts as visible moisture only with vis_km <= 1.5 (so it adds nothing beyond the visibility clause); mist kept",
    "the two NEW encodings (AIRCRAFT_TYPE_mvt, airline|ADEP) and their prior are fitted on the fold's non-fill rows minus the "
    "regressor's own early-stopping rows (the split replicated from sklearn 1.8.0 and verified equal in the tests; the release "
    "is pinned); an unseen or missing level reads that prior. S1C's five encodings are left exactly as S1C computes them",
    "E5w is S1C + the whole registered WEATHER block, the de-icing-airport interaction included",
    "the sensitivity arms are E5 with the variant de-icing condition and the variant interaction; nothing else differs",
    "the reproduction guard holds S1C (registered) and S1, the per-seed S1C arms, p_hat, nf_cells, nf_fit_c to 1e-6 s",
    "an input is DROPPED iff it is absent or has no finite (non-null) value on the 2025 unmatched rows, decided once on the "
    "full label-free 2025 frame before any fold; ship refuses a 2026 frame where a kept input is NaN on every scored row",
    "the 2026 witness bin counts are recomputed from data/raw/ranking.parquet and must equal E3C's stored 2026 witness rows",
    "ship refuses any base but the registered v10 (an amendment would be needed to ship E5 on another base)",
]

_T0 = time.time()
_Log = uc._Log
peak_rss_gb = uc.peak_rss_gb
_git = uc._git
_jsonable = uc._jsonable


# =============================================================================================
# the weather table: loading, the de-icing variants, the as-of join
# =============================================================================================

def file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def lt3v(x, c: float) -> np.ndarray:
    """x < c in three-valued logic: NaN where x is unknown (prc.weather.le3's strict twin)."""
    x = np.asarray(x, dtype="float64")
    return np.where(np.isnan(x), np.nan, (x < c).astype("float64"))


def _nan_equal(a, b) -> np.ndarray:
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    return (a == b) | (np.isnan(a) & np.isnan(b))


def deicing_variants(obs: pd.DataFrame) -> pd.DataFrame:
    """`obs` plus deicing_condition__fog15, deicing_condition__lt3 and dc_frost_float_defect (1.0 where the table's
    deicing_condition is 0 only because a reported 3 C dew-point spread was stored as 3.0000000000000036 -- reported, never
    used as an input). GUARD: the default rule recomputed from the table's fields (temperature at 1e-6 C) must equal the
    table's dc_moist_cold and deicing_condition on every observation, or this raises."""
    missing = [c for c in ("temp_c", "dewspread_c", "vis_km", "wx_precip", "wx_fg", "wx_br", "dc_moist_cold", "dc_frost",
                           "deicing_condition") if c not in obs.columns]
    if missing:
        raise ValueError(f"the weather table lacks {missing}")
    t = np.round(obs.temp_c.to_numpy(dtype="float64"), TEMP_DECIMALS)
    precip, fg, br = (obs[c].to_numpy(dtype="float64") for c in ("wx_precip", "wx_fg", "wx_br"))
    vis_le = WX.le3(obs.vis_km.to_numpy(dtype="float64"), FOG_VIS_KM)
    moist = WX.or3(precip, fg, br, vis_le)
    cold = WX.le3(t, MOIST_C)
    frost = obs.dc_frost.to_numpy(dtype="float64")
    dmc = WX.and3(cold, moist)
    deice = WX.or3(dmc, frost)
    bad_m = int((~_nan_equal(dmc, obs.dc_moist_cold)).sum())
    bad_d = int((~_nan_equal(deice, obs.deicing_condition)).sum())
    if bad_m or bad_d:
        raise ValueError(f"the recomputed default de-icing rule differs from the table's on {bad_m} dc_moist_cold and {bad_d} "
                         "deicing_condition observations; the sensitivities would not be variants of the registered rule")
    moist_fog15 = WX.or3(precip, br, WX.and3(fg, vis_le), vis_le)
    out = obs.copy()
    out["deicing_condition__fog15"] = WX.or3(WX.and3(cold, moist_fog15), frost)
    out["deicing_condition__lt3"] = WX.or3(WX.and3(lt3v(t, MOIST_C), moist), frost)
    sp = obs.dewspread_c.to_numpy(dtype="float64")
    frost_documented = (np.abs(sp - 3.0) < 1e-9) & (sp > 3.0) & (WX.le3(t, 0.0) == 1.0)
    out["dc_frost_float_defect"] = (frost_documented & (frost == 0.0) & (dmc != 1.0)).astype("float64")
    return out


def weather_findings(obs: pd.DataFrame) -> dict:
    """Float-boundary facts about the table (reported): +3 C / 0 C temperatures not stored exactly, and the dew-point spread
    stored at 3.0000000000000036 that switches dc_frost off at temp <= 0 (a prc/weather.py v2.0.1 defect, not fixed here)."""
    t = obs.temp_c.to_numpy(dtype="float64")
    sp = obs.dewspread_c.to_numpy(dtype="float64")
    near3 = np.abs(t - 3.0) < 1e-9
    spread3 = np.abs(sp - 3.0) < 1e-9
    return {"n_obs": int(len(obs)), "temp_plus3_not_exact": int((near3 & (t != 3.0)).sum()),
            "temp_plus3_stored_below_3": int((near3 & (t < 3.0)).sum()),
            "spread3_stored_above_3": int((spread3 & (sp > 3.0)).sum()),
            "dc_frost_lost_to_float_spread": int((spread3 & (sp > 3.0) & (np.round(t, TEMP_DECIMALS) <= 0.0)).sum()),
            "deicing_condition_lost_to_float_spread": int(obs.dc_frost_float_defect.sum()) if "dc_frost_float_defect" in obs else None,
            "deicing_condition_counts": {str(k): int(v) for k, v in obs.deicing_condition.value_counts(dropna=False).items()},
            "variant_differs_from_default": {v: int((~_nan_equal(obs[f"deicing_condition__{v}"], obs.deicing_condition)).sum())
                                             for v in VARIANTS if f"deicing_condition__{v}" in obs}}


def load_weather_obs(path: pathlib.Path = WEATHER_OBS, sha256: str | None = WEATHER_SHA256) -> pd.DataFrame:
    """The frozen observation table (sha256 pinned; `sha256=None` only for synthetic tables), the registered columns
    asserted, (station, valid) unique, valid UTC; returned with the de-icing variants attached."""
    path = pathlib.Path(path)
    if sha256 is not None:
        got = file_sha256(path)
        if got != sha256:
            raise ValueError(f"{path.name} has sha256 {got[:12]}..., the prereg reads {sha256[:12]}... (weather v{WEATHER_VERSION})")
    names = pq.read_schema(path).names
    missing = [c for c in OBS_NEEDED if c not in names]
    if missing:
        raise ValueError(f"{path.name} lacks the registered columns {missing}")
    obs = pq.read_table(path, columns=OBS_NEEDED).to_pandas()
    return check_obs(deicing_variants(obs))


def check_obs(obs: pd.DataFrame) -> pd.DataFrame:
    tz = getattr(obs.valid.dt, "tz", None)
    if tz is None or str(tz) != "UTC":
        raise ValueError(f"the weather table's `valid` must be timezone-aware UTC, got {tz}")
    dup = obs.duplicated(["station", "valid"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} observations repeat a (station, valid) key")
    if obs.station.isna().any():
        raise ValueError("an observation has no station")
    return obs


def weather_asof(rows: pd.DataFrame, obs: pd.DataFrame, columns=None, max_age: pd.Timedelta = MAX_OBS_AGE) -> pd.DataFrame:
    """Per row of `rows` (ADEP_mvt, MVT_TIME_UTC_mvt): the latest observation of station == ADEP_mvt with
    valid <= MVT_TIME_UTC_mvt, and nothing when that observation is more than `max_age` old. Returns `columns` + obs_valid
    + obs_age_s in the rows' order (RangeIndex). Never a later observation, never another station's; a NaN field of the
    chosen observation stays NaN; a row without MVT_TIME or at a station the table does not carry reads NaN."""
    cols = list(JOIN_COLUMNS if columns is None else columns)
    for c in ("ADEP_mvt", "MVT_TIME_UTC_mvt"):
        if c not in rows.columns:
            raise ValueError(f"the as-of join needs rows.{c}")
    missing = [c for c in ("station", "valid", *cols) if c not in obs.columns]
    if missing:
        raise ValueError(f"the weather table lacks {missing}")
    check_obs(obs)
    t = pd.to_datetime(rows.MVT_TIME_UTC_mvt)
    tz = getattr(t.dt, "tz", None)
    if tz is None or str(tz) != "UTC":
        raise ValueError(f"MVT_TIME_UTC_mvt must be timezone-aware UTC, got {tz}")
    n = len(rows)
    codes = {s: i for i, s in enumerate(sorted(set(obs.station.astype(str))))}
    left = pd.DataFrame({"_row": np.arange(n, dtype="int64"),
                         "_code": pd.Series(rows.ADEP_mvt.astype(str).to_numpy()).map(codes).fillna(-1).astype("int64").to_numpy(),
                         "_t": t.astype("datetime64[ns, UTC]").reset_index(drop=True)})
    left = left[left._t.notna()].sort_values("_t", kind="mergesort")
    right = pd.DataFrame({"_code": obs.station.astype(str).map(codes).astype("int64").to_numpy(),
                          "obs_valid": obs.valid.astype("datetime64[ns, UTC]").reset_index(drop=True)})
    for c in cols:
        right[c] = obs[c].to_numpy(dtype="float64")
    right = right.sort_values("obs_valid", kind="mergesort")
    j = pd.merge_asof(left, right, left_on="_t", right_on="obs_valid", by="_code", direction="backward",
                      allow_exact_matches=True, tolerance=max_age)
    age = (j._t - j.obs_valid).dt.total_seconds().to_numpy(dtype="float64")
    hit = ~np.isnan(age)
    if (age[hit] < 0).any() or (age[hit] > max_age.total_seconds()).any():
        raise AssertionError("the as-of join used an observation after take-off or older than the cap")
    jj = j.set_index("_row").reindex(pd.RangeIndex(n))              # rows without MVT_TIME were never joined: NaN
    out = pd.DataFrame({c: jj[c].to_numpy(dtype="float64") for c in cols}, index=pd.RangeIndex(n))
    out["obs_valid"] = jj.obs_valid.reset_index(drop=True)
    out["obs_age_s"] = (jj._t - jj.obs_valid).dt.total_seconds().to_numpy(dtype="float64")
    miss = out.obs_valid.isna().to_numpy()
    if not np.isnan(out.loc[miss, cols].to_numpy(dtype="float64")).all():
        raise AssertionError("a row without an observation carries a weather value")
    return out


def deice_interaction(deicing_condition, adep) -> np.ndarray:
    """deicing_condition x de-icing airport, as the Kleene AND of the two (prc.weather.and3)."""
    at = np.isin(np.asarray(adep).astype(str), DEICE_AIRPORTS).astype("float64")
    return WX.and3(np.asarray(deicing_condition, dtype="float64"), at)


# =============================================================================================
# the physics block: stand_ab's unmatched builder, looked up by MVT_ID
# =============================================================================================

def physics_frame(built: pd.DataFrame) -> tuple:
    """(MVT_ID_mvt + PHYSICS_COLUMNS in float64, n of prev_dep_gap set to NaN) from stand_ab.build_unmatched's output.
    prev_dep_gap < 0 is the builder's 'no earlier matched off-block at this stand' case and becomes NaN."""
    missing = [c for c in ("MVT_ID_mvt", *PHYSICS_COLUMNS) if c not in built.columns]
    if missing:
        raise ValueError(f"the unmatched builder's output lacks {missing}")
    out = pd.DataFrame({"MVT_ID_mvt": built.MVT_ID_mvt.to_numpy(dtype="float64")})
    for c in PHYSICS_COLUMNS:
        out[c] = built[c].to_numpy(dtype="float64")
    neg = out.prev_dep_gap.to_numpy() < 0
    out.loc[neg, "prev_dep_gap"] = np.nan
    if not out.MVT_ID_mvt.is_unique:
        raise ValueError(f"{int(out.MVT_ID_mvt.duplicated().sum())} duplicate MVT_IDs in the physics frame")
    return out, int(neg.sum())


def compare_to_cache(fresh: pd.DataFrame, cache_path: pathlib.Path) -> dict:
    """The fresh build against the stored cache (reported, not used): same ids in the same order, every physics column
    NaN-equal. Returns {status, n_rows, n_differ per column}."""
    if not pathlib.Path(cache_path).exists():
        return {"status": "no cache", "path": str(cache_path)}
    cache = pq.read_table(cache_path, columns=["MVT_ID_mvt", *PHYSICS_COLUMNS]).to_pandas()
    if len(cache) != len(fresh) or not np.array_equal(cache.MVT_ID_mvt.to_numpy(), fresh.MVT_ID_mvt.to_numpy()):
        return {"status": "different rows", "n_fresh": int(len(fresh)), "n_cache": int(len(cache))}
    differ = {c: int((~_nan_equal(fresh[c], cache[c])).sum()) for c in PHYSICS_COLUMNS}
    return {"status": "equal" if not any(differ.values()) else "DIFFERENT", "n_rows": int(len(fresh)), "n_differ": differ}


def build_physics(paths, builder, cache_dir: pathlib.Path | None, log) -> tuple:
    """(physics frame over every file, info). `builder(path)` is stand_ab.build_unmatched_month for the training files and
    stand_ab.build_unmatched_ranking for the serve file; each build is compared to the stored cache of the same name."""
    parts, info = [], {"files": {}, "n_prev_dep_gap_negative_to_nan": 0}
    for p in paths:
        p = pathlib.Path(p)
        t = time.time()
        built = builder(p)
        cmp = compare_to_cache(built, cache_dir / p.name) if cache_dir is not None else {"status": "not compared"}
        frame, n_neg = physics_frame(built)
        del built
        gc.collect()
        parts.append(frame)
        info["files"][p.name] = {"n_rows": int(len(frame)), "n_prev_dep_gap_negative": n_neg, "cache": cmp}
        info["n_prev_dep_gap_negative_to_nan"] += n_neg
        log(f"  physics {p.name}: {len(frame):,} unmatched rows, prev_dep_gap < 0 -> NaN on {n_neg:,}; cache {cmp['status']} "
            f"({time.time() - t:.1f}s, peak RSS {peak_rss_gb():.2f} GB)")
    out = pd.concat(parts, ignore_index=True)
    if not out.MVT_ID_mvt.is_unique:
        raise ValueError("duplicate MVT_IDs across the per-file physics builds")
    return out, info


def attach_physics(rows: pd.DataFrame, obs: pd.DataFrame, physics: pd.DataFrame) -> pd.DataFrame:
    """`rows` with the WEATHER block (as-of join), the three de-icing interactions, the PHYSICS block (looked up by MVT_ID;
    every row must be found) and the airline|ADEP key. Row order and index are preserved. Refuses rows that already carry
    any of these columns, so no other year's inputs can ride in."""
    clash = [c for c in ADDED_COLUMNS if c in rows.columns]
    if clash:
        raise ValueError(f"the rows already carry {clash}; attach_physics builds them itself")
    need = [c for c in ("ADEP_mvt", "MVT_TIME_UTC_mvt", "MVT_ID_mvt", "airline", "AIRCRAFT_TYPE_mvt") if c not in rows.columns]
    if need:
        raise ValueError(f"attach_physics needs rows.{need}")
    w = weather_asof(rows, obs)
    out = rows.copy()
    for c in (*JOIN_COLUMNS, "obs_age_s"):
        out[c] = w[c].to_numpy(dtype="float64")
    out["obs_valid"] = w.obs_valid.array                           # positional; keeps datetime64[ns, UTC]
    adep = out.ADEP_mvt.astype(str).to_numpy()
    out[INTERACTION] = deice_interaction(out.deicing_condition.to_numpy(), adep)
    for v in VARIANTS:
        out[f"{INTERACTION}__{v}"] = deice_interaction(out[f"deicing_condition__{v}"].to_numpy(), adep)
    ph = physics.set_index("MVT_ID_mvt")
    if not ph.index.is_unique:
        raise ValueError("duplicate MVT_IDs in the physics frame")
    ids = out.MVT_ID_mvt.to_numpy(dtype="float64")
    found = np.isin(ids, ph.index.to_numpy(dtype="float64"))
    if not found.all():
        raise ValueError(f"{int((~found).sum())} of {len(ids):,} rows have no physics row (the unmatched builder did not "
                         "produce them); nothing is imputed")
    got = ph.reindex(ids)
    for c in PHYSICS_COLUMNS:
        out[c] = got[c].to_numpy(dtype="float64")
    out["airline_adep"] = out.ADEP_mvt.astype(str) + "|" + out.airline.astype(str)
    return out


# =============================================================================================
# which registered inputs can be built, and the arms' feature lists
# =============================================================================================

def audit_inputs(frame: pd.DataFrame) -> dict:
    """Per registered input (group, status built / dropped, reason, n_rows, n_finite or n_nonnull, n_nonzero). Dropped iff
    the column is absent or carries no finite (non-null) value on these rows. Label-free."""
    groups = {"weather": [*REG_WEATHER, INTERACTION], "queue": REG_QUEUE, "stand_history": REG_STANDHIST,
              "encoded": REG_ENCODED}
    out = {}
    for group, cols in groups.items():
        for c in cols:
            rec = {"group": group, "n_rows": int(len(frame))}
            if c not in frame.columns:
                out[c] = {**rec, "status": "dropped", "reason": "absent: the harness cannot build it for unmatched rows"}
                continue
            if group == "encoded":
                n = int(frame[c].notna().sum())
                rec["n_nonnull"] = n
                rec["n_levels"] = int(frame[c].nunique())
            else:
                v = frame[c].to_numpy(dtype="float64")
                n = int(np.isfinite(v).sum())
                rec["n_finite"] = n
                rec["n_nonzero"] = int((np.isfinite(v) & (v != 0)).sum())
            out[c] = {**rec, "status": "built" if n else "dropped",
                      "reason": None if n else "no finite value on any unmatched row"}
    if out["deicing_condition"]["status"] == "dropped" and out[INTERACTION]["status"] == "built":
        raise AssertionError("the interaction cannot be built without deicing_condition")
    return out


def dropped(audit: dict) -> list:
    return [c for c, r in audit.items() if r["status"] == "dropped"]


def arm_specs(audit: dict) -> dict:
    """{arm: {"numeric": [...], "extra": [...]}} from the audit: S1C's numeric inputs first, then the built inputs of the
    arm's blocks; the variants swap the de-icing condition and its interaction for their own columns."""
    ok = lambda c: audit[c]["status"] == "built"
    base = list(uc.CongestionRegressor.NUMERIC)

    def weather(variant=None):
        cols = []
        for c in [*REG_WEATHER, INTERACTION]:
            if not ok(c):
                continue
            cols.append(f"{c}__{variant}" if variant and c in ("deicing_condition", INTERACTION) else c)
        return cols

    physics = [c for c in PHYSICS_COLUMNS if ok(c)]
    extra = [c for c in REG_ENCODED if ok(c)]
    return {"E5w": {"numeric": base + weather(), "extra": []},
            "E5": {"numeric": base + weather() + physics, "extra": extra},
            "E5_fog15": {"numeric": base + weather("fog15") + physics, "extra": extra},
            "E5_lt3": {"numeric": base + weather("lt3") + physics, "extra": extra}}


# =============================================================================================
# the body: S1C's regressor with the extra inputs; the new encodings off the stopping rows
# =============================================================================================

def stopping_split(n: int, seed: int) -> tuple:
    """(fit positions, stopping positions) of the regressor's internal early-stopping split, replicated from sklearn's
    HistGradientBoostingRegressor.fit (early_stopping='auto' is on above 10,000 rows; the split is
    train_test_split(test_size=validation_fraction, random_state=RandomState(seed).randint(uint32 max))). Below the switch
    there are no stopping rows. Verified equal to the regressor's own split in tests/test_unm_physics.py."""
    if sklearn.__version__ != SKLEARN_VERIFIED:
        raise RuntimeError(f"the stopping split is verified against sklearn {SKLEARN_VERIFIED}, found {sklearn.__version__}")
    params = HGR(**bs.NF_PARAMS).get_params()
    es, vf = params["early_stopping"], params["validation_fraction"]
    on = (n > 10_000) if es == "auto" else bool(es)
    if not on or vf is None:
        return np.arange(n, dtype="int64"), np.zeros(0, dtype="int64")
    rs = np.random.RandomState(int(seed)).randint(np.iinfo(np.uint32).max, dtype="u8")
    fit_idx, stop_idx = train_test_split(np.arange(n, dtype="int64"), test_size=vf, stratify=None, random_state=rs)
    return np.asarray(fit_idx, dtype="int64"), np.asarray(stop_idx, dtype="int64")


def _smoothed(frame: pd.DataFrame, col: str, target: str, prior: float) -> pd.Series:
    """The parent's encoding formula exactly: (mean x n + prior x SMOOTH) / (n + SMOOTH) per level."""
    g = frame.groupby(col, observed=True)[target].agg(["mean", "size"])
    return (g["mean"] * g["size"] + prior * bs.SMOOTH) / (g["size"] + bs.SMOOTH)


class PhysicsRegressor(bs.NonFillRegressor):
    """S1C's body (`build_submission.NonFillRegressor`: NF_PARAMS, the fit target winsorised at WINSOR_S, SMOOTH-smoothed
    encodings of NF_ENCODED with their prior from all non-fill rows -- exactly as `unm_congestion.CongestionRegressor`)
    with `numeric` in place of CongestionRegressor.NUMERIC (which it must start with) and `extra` encodings appended. The
    extra encodings and their prior are computed on the non-fill rows OUTSIDE the regressor's early-stopping split
    (`stopping_split`), so no stopping row's label enters a new feature (BC-1). With numeric == S1C's and no extras it IS
    S1C's regressor, bit for bit."""

    def __init__(self, train_nonfill: pd.DataFrame, seed: int, numeric, extra=()):
        numeric, extra = list(numeric), list(extra)
        base = list(uc.CongestionRegressor.NUMERIC)
        if numeric[:len(base)] != base:
            raise ValueError(f"PhysicsRegressor's numeric inputs must start with S1C's {base}")
        if len(set(numeric)) != len(numeric) or len(set(extra)) != len(extra):
            raise ValueError("duplicate inputs")
        unknown = [c for c in extra if c not in REG_ENCODED]
        if unknown:
            raise ValueError(f"{unknown} are not registered encodings ({REG_ENCODED})")
        missing = [c for c in ("y", *numeric, *extra, *bs.NF_ENCODED) if c not in train_nonfill.columns]
        if missing:
            raise ValueError(f"PhysicsRegressor needs columns {missing}")
        frame = train_nonfill
        y_w = np.clip(frame.y.to_numpy(dtype="float64"), -bs.WINSOR_S, bs.WINSOR_S)
        fit_idx, stop_idx = stopping_split(len(frame), seed)
        self.numeric, self.extra = numeric, extra
        self.n_stop, self.stop_positions = int(len(stop_idx)), np.sort(stop_idx)
        clean = frame.iloc[np.sort(fit_idx)].assign(_y_w=y_w[np.sort(fit_idx)])
        self.extra_prior = float(clean._y_w.mean()) if extra else None
        self.extra_maps = {c: _smoothed(clean, c, "_y_w", self.extra_prior) for c in extra}
        super().__init__(frame, seed)                   # S1C's prior and five maps; fits through self.design below
        self.columns = numeric + ["te_" + c for c in bs.NF_ENCODED] + ["te_" + c for c in extra]
        if bool(getattr(self.model, "do_early_stopping_", False)) != (self.n_stop > 0):
            raise AssertionError("the replicated stopping split disagrees with the regressor's own early-stopping switch")

    def design(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self.numeric].astype("float64").copy()
        for c in bs.NF_ENCODED:
            x["te_" + c] = frame[c].astype(str).map(self.maps[c]).astype("float64").fillna(self.prior)
        for c in self.extra:
            x["te_" + c] = frame[c].astype(str).map(self.extra_maps[c]).astype("float64").fillna(self.extra_prior)
        return x


def fit_physics_regressor(train_unmatched_nonfill: pd.DataFrame, seed: int, numeric, extra=()) -> PhysicsRegressor:
    """An E5-family body. The same refusals as `unm_congestion.fit_congestion_regressor` (no fill rows, a label present)."""
    frame = train_unmatched_nonfill
    if len(frame) == 0 or "y" not in frame.columns:
        raise ValueError("fit_physics_regressor needs non-fill training rows with a label `y`")
    if "BLOCK_TIME_UTC_mvt" in frame.columns and "SCHED_TIME_UTC_mvt" in frame.columns:
        n_fill = int(bs.schedule_fill(frame).sum())
        if n_fill:
            raise ValueError(f"fit_physics_regressor expects NON-FILL rows only; {n_fill} fill rows found")
    return PhysicsRegressor(frame, seed, numeric, extra)


def physics_arms(train_unm: pd.DataFrame, test_unm: pd.DataFrame, train_matched: pd.DataFrame, specs: dict,
                 seeds=SEEDS) -> tuple:
    """S1, S1C and their per-seed arms exactly as `unm_congestion.congestion_arms`, plus every arm in `specs` rebuilt from
    S1's OWN parts: p, sp and nf_cells untouched, only the body replaced, fitted on fit_unmatched's own non-fill rows."""
    arms, parts = uc.congestion_arms(train_unm, test_unm, train_matched, seeds=seeds)
    nonfill = train_unm[~bs.schedule_fill(train_unm)]              # fit_unmatched's `tr[~tr.fill]`, same rows, same order
    for arm, spec in specs.items():
        by_seed = {int(s): fit_physics_regressor(nonfill, s, spec["numeric"], spec["extra"]).predict(test_unm) for s in seeds}
        nf_fit = bs.mean_over_seeds(by_seed)
        arms[arm] = bs.mixture(parts["p"], parts["sp"], bs.nf_hybrid(parts["nf_cells"], nf_fit))
        for s in seeds:
            arms[f"{arm}_seed{s}"] = bs.mixture(parts["p"], parts["sp"], bs.nf_hybrid(parts["nf_cells"], by_seed[int(s)]))
        parts[f"nf_fit_{arm}"], parts[f"nf_fit_{arm}_by_seed"] = nf_fit, by_seed
    parts["n_train_nonfill"] = int(len(nonfill))
    return arms, parts


# =============================================================================================
# LOMO and the reproduction guard
# =============================================================================================

def _score_split(unm, lirf_matched, tr, te, test_months, specs, seeds, fold_label: str) -> pd.DataFrame:
    if tuple(int(s) for s in seeds) != tuple(SEEDS):
        raise ValueError(f"the harness scores the registered seeds {SEEDS} (every per-seed column is stored), got {tuple(seeds)!r}")
    if set(specs) != set(ARMS):
        raise ValueError(f"the harness scores the registered arms {ARMS}, got {sorted(specs)}")
    tr_unm, te_unm = unm[tr], unm[te]
    tr_mat = lirf_matched[~lirf_matched.month.isin(list(test_months))]
    arms, parts = physics_arms(tr_unm, te_unm, tr_mat, specs, seeds=seeds)
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
    for c in ROW_INPUTS:
        out[c] = te_unm[c].to_numpy(dtype="float64")
    for a in ARMS:
        out[f"nf_fit_{a}"] = np.asarray(parts[f"nf_fit_{a}"], dtype="float64")
    for a in PRED_COLUMNS:
        if a.startswith("S1") or a.startswith("E5"):
            out[a] = np.asarray(arms[a], dtype="float64")
    return out[PRED_COLUMNS]


def score_lomo(unm: pd.DataFrame, lirf_matched: pd.DataFrame, specs: dict, seeds, log) -> pd.DataFrame:
    """Out-of-fold predictions for every stratum row: E3C's LOMO loop with the E5-family arms added."""
    month = unm.month.to_numpy()
    parts, scored = [], np.zeros(len(unm), dtype="int64")
    for m, tr, te in sf.lomo_folds(month):
        t = time.time()
        part = _score_split(unm, lirf_matched, tr, te, [m], specs, seeds, "lomo")
        scored += te
        parts.append(part)
        d = (part.E5.to_numpy() != part.S1C.to_numpy())
        log(f"  fold {m:2d}: test {int(te.sum()):5,}  train {int(tr.sum()):6,}  routed {int(part.routed_nf_fit.sum()):5,} -> body; "
            f"E5 differs from S1C on {int(d.sum()):5,}; weather NaN on {int(np.isnan(part.obs_age_s.to_numpy()).sum()):4,}; "
            f"deicing==1 on {int((part.deicing_condition.to_numpy() == 1).sum()):4,}  ({time.time() - t:.1f}s)")
    if not (scored == 1).all():
        raise AssertionError("LOMO did not score every row exactly once")
    return pd.concat(parts, ignore_index=True)[PRED_COLUMNS]


def check_reproduction(preds: pd.DataFrame, reference: pd.DataFrame, columns=GUARD_COLUMNS, tol: float = REPRO_TOL_S) -> dict:
    """E3R's guard (every lomo row joined on MVT_ID, the id sets equal, each column within `tol`) over GUARD_COLUMNS."""
    return ucr.check_reproduction(preds, reference, columns=columns, tol=tol)


# =============================================================================================
# clauses, the reported arms and the cuts
# =============================================================================================

def gains(t: pd.DataFrame, treatment: str = DECISIONAL, control: str = CONTROL) -> np.ndarray:
    """g = SE(control) - SE(treatment) per row, against the RAW label. Never winsorised."""
    y = t.y.to_numpy(dtype="float64")
    return (y - t[control].to_numpy(dtype="float64")) ** 2 - (y - t[treatment].to_numpy(dtype="float64")) ** 2


def e3c_view(t: pd.DataFrame, treatment: str = DECISIONAL) -> pd.DataFrame:
    """The frame `unm_congestion.clauses` scores: S1 := S1C (control), S1C := treatment, and the same per seed."""
    cols = {"y": t.y, "month": t.month, "date": t.date, "ADEP_mvt": t.ADEP_mvt, "hprox_med": t.hprox_med,
            "S1": t[CONTROL], "S1C": t[treatment]}
    for s in SEEDS:
        cols[f"S1_seed{s}"] = t[f"{CONTROL}_seed{s}"]
        cols[f"S1C_seed{s}"] = t[f"{treatment}_seed{s}"]
    return pd.DataFrame({k: v.to_numpy() for k, v in cols.items()})


def clauses(t: pd.DataFrame, counts_2026: dict, treatment: str = DECISIONAL, n_boot=N_BOOT, seed=BOOT_SEED,
            n_scored=N_SCORED_2026) -> dict:
    """C1-C7 on the non-LIRF rows `t` for `treatment` against S1C. C1, C3-C7 are E3C's arithmetic on the renamed view; C2
    is the event-excluded price at +300 (E3R's function); the full price is kept under its own key, not decisional.
    Every threshold is the prereg's; nothing here is tuned."""
    if (t.ADEP_mvt == ROME).any():
        raise ValueError("clauses are evaluated on non-LIRF rows only; LIRF rows were passed in")
    view = e3c_view(t, treatment)
    c = uc.clauses(view, counts_2026, n_boot=n_boot, seed=seed, n_scored=n_scored)
    g = gains(t, treatment)
    if not np.array_equal(g, uc.gains(view)):
        raise AssertionError("the renamed view's gains differ from E5's gains; the view is wrong")
    full = dict(c["C2_priced_2026"])
    full.pop("passes")
    full["rule"] = "full price (reported, not decisional)"
    c["C2_full_price_not_decisional"] = full
    c["C2_priced_2026"] = ucr.event_excluded_price(g, t.hprox_med.to_numpy(dtype="float64"), t.ADEP_mvt.to_numpy(),
                                                   t.month.to_numpy(), counts_2026, n_scored)
    c["airport_month_top"] = ucr.airport_month_table(g, t.ADEP_mvt.to_numpy(), t.month.to_numpy()).head(12).to_dict(orient="records")
    c["treatment"], c["control"] = treatment, CONTROL
    c["verdict"] = uc.verdict(c)
    return c


verdict = uc.verdict


def rmse_arms(t: pd.DataFrame, mask=None, arms=(CONTROL, "E5w", DECISIONAL)) -> dict:
    y = t.y.to_numpy(dtype="float64")
    m = np.ones(len(t), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    out = {"n": int(m.sum())}
    for a in arms:
        e = (y[m] - t[a].to_numpy(dtype="float64")[m]) ** 2
        out[a] = float(np.sqrt(e.mean())) if m.any() else None
    out["gain_s"] = (out[arms[0]] - out[DECISIONAL]) if m.any() else None
    return out


def deice_cut_key(adep, cond) -> np.ndarray:
    """'deice_apt|1' / 'deice_apt|0' / 'deice_apt|NaN' and the same for 'other_apt'."""
    at = np.where(np.isin(np.asarray(adep).astype(str), DEICE_AIRPORTS), "deice_apt", "other_apt")
    c = np.asarray(cond, dtype="float64")
    lab = np.where(np.isnan(c), "NaN", np.where(c == 1.0, "1", "0"))
    return np.char.add(np.char.add(at.astype(str), "|"), lab.astype(str)).astype(object)


def freezing_cut_key(adep, temp_c) -> np.ndarray:
    """de-icing airport x temp_c <= 0 (at 1e-6 C): '<=0' / '>0' / 'NaN'."""
    at = np.where(np.isin(np.asarray(adep).astype(str), DEICE_AIRPORTS), "deice_apt", "other_apt")
    t = np.round(np.asarray(temp_c, dtype="float64"), TEMP_DECIMALS)
    lab = np.where(np.isnan(t), "NaN", np.where(t <= 0.0, "<=0", ">0"))
    return np.char.add(np.char.add(at.astype(str), "|"), lab.astype(str)).astype(object)


def cut_table(t: pd.DataFrame, key) -> dict:
    """Per level of `key`: n, mean y, sum g of E5 and of E5w against S1C, the physics increment sum[SE(E5w) - SE(E5)]
    (= the E5 sum minus the E5w sum), RMSE of S1C / E5w / E5."""
    key = np.asarray(key, dtype=object)
    g5, gw = gains(t, DECISIONAL), gains(t, "E5w")
    out = {}
    for k in sorted({str(v) for v in key}):
        m = key.astype(str) == k
        r = rmse_arms(t, m)
        out[k] = {"n": int(m.sum()), "mean_y": float(t.y.to_numpy()[m].mean()), "sum_g_E5": float(g5[m].sum()),
                  "sum_g_E5w": float(gw[m].sum()), "sum_physics_increment": float((g5[m] - gw[m]).sum()),
                  "rmse_S1C": r[CONTROL], "rmse_E5w": r["E5w"], "rmse_E5": r[DECISIONAL]}
    return out


def attribution(t: pd.DataFrame) -> dict:
    """How much of E5's gain is weather: sum g(E5w) + sum[SE(E5w) - SE(E5)] == sum g(E5), per month and airport too."""
    g5, gw = gains(t, DECISIONAL), gains(t, "E5w")
    total, weather = float(g5.sum()), float(gw.sum())
    return {"sum_g_E5": total, "sum_g_E5w": weather, "sum_physics_increment": float((g5 - gw).sum()),
            "weather_share": (weather / total) if total != 0.0 else None,
            "per_month": cut_table(t, t.month.to_numpy().astype("int64").astype(str)),
            "per_airport": cut_table(t, t.ADEP_mvt.astype(str).to_numpy())}


def sensitivities(t: pd.DataFrame) -> dict:
    """The registered de-icing sensitivities (reported): each variant against S1C and against E5."""
    y = t.y.to_numpy(dtype="float64")
    se5 = (y - t[DECISIONAL].to_numpy(dtype="float64")) ** 2
    out = {}
    for a in ("E5_fog15", "E5_lt3"):
        sev = (y - t[a].to_numpy(dtype="float64")) ** 2
        col = f"deicing_condition__{a.split('_', 1)[1]}"
        differs = ~_nan_equal(t[col], t.deicing_condition)
        out[a] = {"sum_g_vs_S1C": float(gains(t, a).sum()), "sum_se_E5_minus_variant": float((se5 - sev).sum()),
                  "n_rows_condition_differs": int(differs.sum()),
                  "sum_se_E5_minus_variant_on_those_rows": float((se5 - sev)[differs].sum()),
                  "rmse": rmse_arms(t, None, (CONTROL, "E5w", DECISIONAL, a))[a]}
    return out


def score_tables(non: pd.DataFrame) -> dict:
    """Every reported cut on the non-LIRF rows."""
    return {"per_month": cut_table(non, non.month.to_numpy().astype("int64").astype(str)),
            "per_airport": cut_table(non, non.ADEP_mvt.astype(str).to_numpy()),
            "per_witness_bin": cut_table(non, uc.bin_of(non.hprox_med)),
            "per_season": cut_table(non, np.array([SEASONS[int(m)] for m in non.month.to_numpy()], dtype=object)),
            "deice_airport_x_deicing_condition": cut_table(non, deice_cut_key(non.ADEP_mvt, non.deicing_condition)),
            "deice_airport_x_freezing": cut_table(non, freezing_cut_key(non.ADEP_mvt, non.temp_c))}


# =============================================================================================
# frames: the real recipe and the smoke world
# =============================================================================================

def load_real_physics(files, log) -> tuple:
    """(unmatched stratum with the witness AND the physics attached, LIRF matched, witness summary, physics info). The
    physics block is built fresh from the raw files first (one month at a time, ~0.5 GB each), then E3C's load_real."""
    log("--- the physics block: stand_ab.build_unmatched_month per training file (fresh; compared to data/cache_unmatched) ---")
    physics, pinfo = build_physics(files, S.build_unmatched_month, UCACHE, log)
    unm, lirf, wsum = uc.load_real(files, log)
    obs = load_weather_obs()
    unm = attach_physics(unm, obs, physics).reset_index(drop=True)
    pinfo["weather_table"] = {"path": str(WEATHER_OBS.relative_to(ROOT)), "version": WEATHER_VERSION, "sha256": WEATHER_SHA256,
                              "findings": weather_findings(obs)}
    return unm, lirf, wsum, pinfo


def smoke_weather_obs(seed: int = 0, start: str = "2025-01-01", end: str = "2026-01-02") -> pd.DataFrame:
    """A synthetic observation table in the frozen table's shape: every station of the synthetic world every 30 min, cold
    winters, fog / mist / precipitation / snow codes, a 2% random loss and one 5-hour outage per station (the 3 h cap is
    exercised), a few unknown weather groups (NaN flags). Temperatures go through the archive's F->C path, so a +3 C
    reading is stored as 2.999999999999999 exactly as in the real table. The de-icing condition is prc.weather's rule.
    Code path only, never a result."""
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq="30min", tz="UTC", inclusive="left")
    parts = []
    for i, st in enumerate(sf.AIRPORTS):
        keep = rng.random(len(times)) > 0.02
        gap0 = rng.integers(0, len(times) - 10)
        keep[gap0:gap0 + 10] = False
        tt = times[keep]
        n = len(tt)
        doy = tt.dayofyear.to_numpy()
        temp = np.rint(9.0 - 10.0 * np.cos(2 * np.pi * (doy - 15) / 365.0) + rng.normal(0.0, 3.0, n) + (i % 3))
        spread = np.abs(np.rint(rng.normal(3.0, 2.5, n)))
        tmpf, dwpf = np.round(temp * 9 / 5 + 32, 1), np.round((temp - spread) * 9 / 5 + 32, 1)
        tc, dc = (tmpf - 32.0) * 5.0 / 9.0, (dwpf - 32.0) * 5.0 / 9.0
        fog = rng.random(n) < 0.04
        mist = ~fog & (rng.random(n) < 0.05)
        vis = np.where(fog, rng.uniform(0.2, 3.0, n), np.where(mist, rng.uniform(1.0, 5.0, n), 10.0))
        precip = rng.random(n) < np.where(doy < 60, 0.20, 0.10)
        snow = precip & (tc <= 1.0)
        fzra = precip & ~snow & (tc <= 0.0) & (rng.random(n) < 0.3)
        wind = np.rint(rng.gamma(2.0, 5.0, n))
        gust = np.where(rng.random(n) < 0.1, wind + 10.0, 0.0)
        unknown = rng.random(n) < 0.002
        f = lambda b: np.where(unknown, np.nan, b.astype("float64"))
        o = pd.DataFrame({"station": st, "valid": tt, "temp_c": tc, "dewpoint_c": dc, "dewspread_c": tc - dc,
                          "vis_km": vis, "wind_kt": wind, "gust_kt": gust,
                          "wx_precip": f(precip), "wx_frozen": f(snow | fzra), "wx_sn": f(snow), "wx_fzra": f(fzra),
                          "wx_fzdz": f(np.zeros(n, dtype=bool)), "wx_fzfg": f(fog & (tc <= 0.0)), "wx_fg": f(fog),
                          "wx_br": f(mist)})
        o["wx_intensity"] = np.where(o.wx_precip == 1.0, rng.integers(-1, 2, n).astype("float64"), np.nan)
        parts.append(o)
    obs = pd.concat(parts, ignore_index=True)
    moist = WX.or3(obs.wx_precip.to_numpy(), obs.wx_fg.to_numpy(), obs.wx_br.to_numpy(), WX.le3(obs.vis_km.to_numpy(), WX.DEICE_VIS_KM))
    obs["dc_moist_cold"] = WX.and3(WX.le3(obs.temp_c.to_numpy(), WX.DEICE_MOIST_C), moist)
    obs["dc_frost"] = WX.and3(WX.le3(obs.temp_c.to_numpy(), WX.DEICE_FROST_C), WX.le3(obs.dewspread_c.to_numpy(), WX.DEICE_SPREAD_C))
    obs["deicing_condition"] = WX.or3(obs.dc_moist_cold.to_numpy(), obs.dc_frost.to_numpy())
    return check_obs(deicing_variants(obs[OBS_NEEDED]))


def smoke_physics(unm: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A synthetic physics frame (MVT_ID + PHYSICS_COLUMNS) for the rows of `unm`: queue counts that peak with the witness
    hours, ~2% NaN, and ~3% negative prev_dep_gap (the builder's no-earlier-departure case) -- through physics_frame, so
    the negative ones arrive as NaN exactly as the real ones do. Code path only."""
    rng = np.random.default_rng(seed)
    n = len(unm)
    hr = unm.MVT_TIME_UTC_mvt.dt.hour.to_numpy()
    peak = ((hr >= 6) & (hr <= 9)) | ((hr >= 16) & (hr <= 19))
    raw = pd.DataFrame({"MVT_ID_mvt": unm.MVT_ID_mvt.to_numpy(dtype="float64"),
                        "q_dep_tko_sym15": rng.poisson(np.where(peak, 22.0, 12.0)).astype("float64"),
                        "q_rwy_tko_sym10": rng.poisson(np.where(peak, 10.0, 5.0)).astype("float64"),
                        "q_arr_taxiin_sym30_tko": rng.normal(570.0, 80.0, n),
                        "q_rwy_ambient_proxy": rng.normal(np.where(peak, 1_200.0, 850.0), 150.0),
                        "prev_arr_gap": rng.exponential(9_000.0, n), "prev_arr_taxiin": rng.normal(520.0, 100.0, n),
                        "prev_dep_gap": np.where(rng.random(n) < 0.03, -rng.exponential(50_000.0, n), rng.exponential(17_000.0, n))})
    for c in PHYSICS_COLUMNS:
        raw.loc[rng.random(n) < 0.02, c] = np.nan
    return physics_frame(raw)[0]


def plant_signal(unm: pd.DataFrame) -> pd.DataFrame:
    """Smoke only: y += PLANT_DEICE_S on NON-FILL rows with deice_x_airport == 1 and += PLANT_QUEUE_S per departure above
    15 in q_dep_tko_sym15 -- so the smoke can show the harness recovers a planted weather and queue effect."""
    nonfill = ~bs.schedule_fill(unm)
    dx = (unm[INTERACTION].to_numpy() == 1.0) & nonfill
    q = np.nan_to_num(unm.q_dep_tko_sym15.to_numpy(dtype="float64") - 15.0, nan=0.0) * nonfill
    return unm.assign(y=np.rint(unm.y.to_numpy(dtype="float64") + PLANT_DEICE_S * dx + PLANT_QUEUE_S * q))


def synthetic_world(seed: int = 0, plant: bool = True, **kw) -> tuple:
    """(unmatched stratum with the witness and the synthetic physics attached, LIRF matched, witness summary, physics info)."""
    unm, lirf, wsum = uc.synthetic_world(seed=seed, **kw)
    obs = smoke_weather_obs(seed=seed + 200)
    unm = attach_physics(unm, obs, smoke_physics(unm, seed=seed + 300)).reset_index(drop=True)
    if plant:
        unm = plant_signal(unm)
    info = {"synthetic": True, "planted": bool(plant), "weather_table": {"findings": weather_findings(obs)}}
    return unm, lirf, wsum, info


# =============================================================================================
# records, paths
# =============================================================================================

def output_paths(smoke: bool) -> dict:
    """Real: the parquet under data/cache_stand/, the json and logs under reports/. Smoke: everything under
    data/smoke_unm_physics/, never the real paths."""
    if smoke:
        return {k: SMOKE_DIR / v for k, v in dict(lomo=LOMO_NAME, json=JSON_NAME, lomo_log=LOMO_LOG, score_log=SCORE_LOG).items()}
    return {"lomo": CACHE / LOMO_NAME, "json": REPORTS / JSON_NAME, "lomo_log": REPORTS / LOMO_LOG, "score_log": REPORTS / SCORE_LOG}


def write_lomo(preds: pd.DataFrame, path: pathlib.Path, extra: dict) -> None:
    """The per-row parquet with its convention, prereg, tag, git sha, weather version and feature lists in the metadata."""
    table = pa.Table.from_pandas(preds[PRED_COLUMNS], preserve_index=False)
    meta = {b"convention": CONVENTION.encode(), b"prereg": PREREG.encode(), b"tag": TAG.encode(),
            b"git_sha": str(extra.get("git_sha")).encode(), b"weather_version": WEATHER_VERSION.encode(),
            b"feature_lists": json.dumps(extra.get("arms", {})).encode(), META_KEY: json.dumps(extra, default=_jsonable).encode()}
    pq.write_table(table.replace_schema_metadata({**(table.schema.metadata or {}), **meta}), path)


def read_lomo(path: pathlib.Path) -> tuple:
    """The stored predictions and their metadata; refuses a file without the taxi-time stamp or of another arm."""
    table = pq.read_table(path)
    meta = table.schema.metadata or {}
    conv = meta.get(b"convention", b"").decode()
    if conv != CONVENTION:
        raise ValueError(f"{pathlib.Path(path).name} carries convention {conv!r}, expected {CONVENTION!r} (reports/bug_classes.md BC-2)")
    if meta.get(b"tag", b"").decode() != TAG:
        raise ValueError(f"{pathlib.Path(path).name} is not an {TAG} parquet (tag {meta.get(b'tag', b'').decode()!r})")
    return table.to_pandas(), json.loads(meta.get(META_KEY, b"{}").decode())


# =============================================================================================
# steps: lomo and score
# =============================================================================================

def run_lomo(args, paths, log) -> int:
    started = dt.datetime.now(dt.timezone.utc)
    if args.smoke:
        unm, lirf, wsum, pinfo = synthetic_world(seed=0)
        log(f"synthetic world: {len(unm):,} unmatched rows, {len(lirf):,} LIRF matched rows; signal planted "
            f"({PLANT_DEICE_S:.0f} s de-icing, {PLANT_QUEUE_S:.0f} s per queued departure)")
    else:
        files = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
        if len(files) != 12:
            raise SystemExit(f"expected the 12 training months under {RAW}, found {len(files)}")
        if not REFERENCE.exists():
            raise SystemExit(f"the reproduction reference {REFERENCE} is missing; the prereg's guard cannot run")
        unm, lirf, wsum, pinfo = load_real_physics(files, log)
    audit = audit_inputs(unm)
    specs = arm_specs(audit)
    log(f"inputs: built {sum(r['status'] == 'built' for r in audit.values())} of {len(audit)}; DROPPED {dropped(audit) or 'none'}")
    for c, r in audit.items():
        log(f"  {c:26s} {r['group']:13s} {r['status']:8s} " + (f"finite {r['n_finite']:,} nonzero {r['n_nonzero']:,}" if "n_finite" in r
                                                                 else f"non-null {r.get('n_nonnull', 0):,} levels {r.get('n_levels', 0):,}"))
    for a, sp_ in specs.items():
        log(f"  arm {a}: numeric {len(sp_['numeric'])} ({', '.join(sp_['numeric'][len(uc.CongestionRegressor.NUMERIC):])}); extra {sp_['extra']}")
    log(f"weather: joined on {int(unm.obs_valid.notna().sum()):,} of {len(unm):,} rows (median age {np.nanmedian(unm.obs_age_s):.0f} s); "
        f"deicing_condition 1 on {int((unm.deicing_condition == 1).sum()):,}, interaction 1 on {int((unm[INTERACTION] == 1).sum()):,}; "
        f"table findings {pinfo['weather_table']['findings']}")
    log("--- 12-fold LOMO: S1C (unm_congestion.congestion_arms) and E5w / E5 / E5_fog15 / E5_lt3 ---")
    t = time.time()
    preds = score_lomo(unm, lirf, specs, SEEDS, log)
    log(f"LOMO scored {len(preds):,} rows in {time.time() - t:.0f}s; peak RSS {peak_rss_gb():.2f} GB")
    if args.smoke:
        repro = {"status": "skipped (synthetic)", "tolerance_s": REPRO_TOL_S, "columns": list(GUARD_COLUMNS)}
    else:
        repro = check_reproduction(preds, ucr.load_reference())
    log(f"reproduction guard: {repro}")
    extra = {"prereg": PREREG, "parent_prereg": PARENT_PREREG, "tag": TAG, "smoke": bool(args.smoke), **_git(),
             "started_utc": started.isoformat(), "seeds": list(SEEDS), "reproduction": repro,
             "weather": {"version": WEATHER_VERSION, "sha256": WEATHER_SHA256, "max_age_s": MAX_OBS_AGE.total_seconds(),
                         "deice_airports": list(DEICE_AIRPORTS), "variants": list(VARIANTS)},
             "physics": pinfo, "witness": {"columns": uc.WITNESS, "min_n": uc.MIN_N, **wsum},
             "audit": audit, "dropped_inputs": dropped(audit), "arms": specs,
             "regressor": {"params": dict(bs.NF_PARAMS), "encoded": list(bs.NF_ENCODED), "winsor_s": bs.WINSOR_S,
                           "T_tail_s": bs.T_TAIL_S, "smooth": bs.SMOOTH, "sklearn": sklearn.__version__},
             "judgement_calls": JUDGEMENT_CALLS, "n_rows": int(len(preds)), "n_non_lirf": int((preds.ADEP_mvt != ROME).sum()),
             "wall_s": round(time.time() - _T0, 1), "peak_rss_gb": round(peak_rss_gb(), 2)}
    write_lomo(preds, paths["lomo"], extra)
    non = preds[preds.ADEP_mvt != ROME]
    r = rmse_arms(non)
    log(f"predictions -> {paths['lomo']} ({len(preds):,} rows, convention {CONVENTION}); non-LIRF RMSE S1C {r['S1C']:.2f} "
        f"E5w {r['E5w']:.2f} E5 {r['E5']:.2f} (descriptive; the clauses are the `score` step)   wall {extra['wall_s']:.0f}s   "
        f"peak RSS {extra['peak_rss_gb']:.2f} GB")
    return 0


def _log_cut(log, title: str, tab: dict) -> None:
    log(f"  --- {title} ---  {'key':22s} {'n':>6s} {'mean y':>8s} {'sum g E5':>15s} {'sum g E5w':>15s} {'physics +':>15s} "
        f"{'S1C':>8s} {'E5':>8s}")
    for k, v in tab.items():
        log(f"  {'':{len(title) + 8}s} {k:22s} {v['n']:6,} {v['mean_y']:8.0f} {v['sum_g_E5']:+15,.0f} {v['sum_g_E5w']:+15,.0f} "
            f"{v['sum_physics_increment']:+15,.0f} {v['rmse_S1C']:8.1f} {v['rmse_E5']:8.1f}")


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
        ucr.check_counts_match_e3c(counts, pd.read_parquet(E3C_WITNESS_2026)["bin"].to_numpy())
        counts_check = f"equal to {E3C_WITNESS_2026.relative_to(ROOT)}"
    log(f"2026 witness: {len(w26):,} scored non-LIRF unmatched rows, bins {counts} (cross-check: {counts_check})")
    non = preds[preds.ADEP_mvt != ROME].reset_index(drop=True)
    lirf = preds[preds.ADEP_mvt == ROME].reset_index(drop=True)
    c = clauses(non, counts, DECISIONAL, n_boot=n_boot, seed=BOOT_SEED, n_scored=n_scored)
    reported = {a: clauses(non, counts, a, n_boot=n_boot, seed=BOOT_SEED, n_scored=n_scored) for a in REPORTED_ARMS}
    for rec in reported.values():
        rec["decisional"] = False
    mon = sf.monster_mask(non.y.to_numpy())
    g = gains(non)
    lirf_g = gains(lirf) if len(lirf) else np.zeros(0)
    record = {"mode": "unm_physics_score", "prereg": PREREG, "parent_prereg": PARENT_PREREG, "tag": TAG, "smoke": bool(args.smoke),
              **_git(), "started_utc": started.isoformat(),
              "commands": {"lomo": COMMAND_LOMO, "score": COMMAND_SCORE, "ship": COMMAND_SHIP}, "lomo_metadata": lomo_meta,
              "config": {"control": CONTROL, "treatment": DECISIONAL, "reported_arms": list(REPORTED_ARMS), "seeds": list(SEEDS),
                         "n_boot": int(n_boot), "boot_seed": BOOT_SEED, "blocks": "calendar dates (UTC) of MVT_TIME",
                         "bins": uc.BINS, "bin_edges": [float(e) for e in uc.BIN_EDGES], "thin_bin": uc.THIN_BIN,
                         "n_scored_2026": N_SCORED_2026, "c2_min_mse": C2_MIN_MSE,
                         "c2_rule": "drop the ONE airport-month with the largest |sum g|, re-bin, re-price; the full price is reported only",
                         "min_months": uc.MIN_MONTHS, "min_airports": uc.MIN_AIRPORTS, "calm_s": CALM_S, "calm_tolerance": uc.CALM_TOLERANCE,
                         "monster_s": MONSTER_S, "deice_airports": list(DEICE_AIRPORTS), "max_obs_age_s": MAX_OBS_AGE.total_seconds()},
              "data": {"n_lomo_rows": int(len(preds)), "n_non_lirf": int(len(non)), "n_lirf": int(len(lirf)),
                       "n_2026_scored_non_lirf_unmatched": int(len(w26)), "bins_2026": counts, "bins_2026_check": counts_check,
                       "non_lirf_weather_nan": int(non.obs_age_s.isna().sum()),
                       "non_lirf_deicing_condition": {str(k): int(v) for k, v in non.deicing_condition.value_counts(dropna=False).items()},
                       "non_lirf_rows_on_a_float_defect_observation": int((non.dc_frost_float_defect == 1.0).sum())},
              "dropped_inputs": lomo_meta.get("dropped_inputs"), "arms": lomo_meta.get("arms"),
              "clauses": c, "verdict": c["verdict"],
              "reported_not_decisional": {"clauses": reported, "attribution_E5w_vs_E5": attribution(non),
                                          "sensitivities": sensitivities(non), "cuts": score_tables(non)},
              "descriptive": {"non_lirf": {"pooled": rmse_arms(non), "exmonster": rmse_arms(non, ~mon)},
                              "lirf_report_only": {"n": int(len(lirf)), "sum_g": float(lirf_g.sum()),
                                                   "rmse": rmse_arms(lirf) if len(lirf) else None}},
              "judgement_calls": lomo_meta.get("judgement_calls", JUDGEMENT_CALLS) + [
                  "C1 and C3-C7 are unm_congestion.clauses' arithmetic on a renamed view (S1 := S1C, S1C := E5), the view's gains "
                  "checked equal to E5's on every call; the reported arms are scored by the same function, never decisionally"]}
    # ---- readable tables ----
    log(f"=== E5 on the non-LIRF LOMO rows: n={len(non):,}, sum g {c['pooled_sum_g']:+,.0f} s^2 (g = SE(S1C) - SE(E5)) ===")
    r = record["descriptive"]["non_lirf"]
    log(f"  RMSE pooled S1C {r['pooled']['S1C']:.2f} E5w {r['pooled']['E5w']:.2f} E5 {r['pooled']['E5']:.2f} gain {r['pooled']['gain_s']:+.2f} s; "
        f"ex-monster S1C {r['exmonster']['S1C']:.2f} E5 {r['exmonster']['E5']:.2f} gain {r['exmonster']['gain_s']:+.2f} s")
    log(f"  dropped inputs: {record['dropped_inputs'] or 'none'}")
    c1 = c["C1_date_block_bootstrap"]
    log(f"  C1 date-block bootstrap ({c1['n_blocks']} dates, {c1['n_draws']} draws, seed {c1['seed']}): sum g {c1['sum_g']:+,.0f} "
        f"95% [{c1['ci95'][0]:+,.0f}, {c1['ci95'][1]:+,.0f}]  passes {c1['passes']}")
    c2, full = c["C2_priced_2026"], c["C2_full_price_not_decisional"]
    ev = c2["excluded_airport_month"]
    log(f"  C2 EVENT-EXCLUDED 2026 price {c2['priced_mse']:+.1f} board MSE (>= {C2_MIN_MSE:.0f}: {c2['passes']}); dropped {ev['ADEP_mvt']} "
        f"month {ev['month']} (sum g {ev['sum_g']:+,.0f} on {ev['n']:,} rows); full price (reported) {full['priced_mse']:+.1f}")
    for title, tab in (("event-excluded", c2), ("full", full)):
        for name in uc.BINS:
            b = tab["bins"][name]
            prg = "n/a" if b["per_row_gain"] is None else f"{b['per_row_gain']:+.1f}"
            log(f"  [{title}] {name:10s} {b['n_2025']:7,} {prg:>12s} {b['per_row_gain_used']:+12.1f} {str(b['borrowed_from']):>10s} "
                f"{b['n_2026']:7,} {b['priced_mse']:+9.2f}")
    c3, c4 = c["C3_months"], c["C4_airports"]
    log(f"  C3 months with sum g > 0: {c3['months_positive']} of {c3['n_months']} (>= {uc.MIN_MONTHS}: {c3['passes']})")
    log(f"  C4 airports with sum g > 0: {c4['airports_positive']} of {c4['n_airports']} (>= {uc.MIN_AIRPORTS}: {c4['passes']})")
    c5, c6, c7 = c["C5_calm_control"], c["C6_not_the_monsters"], c["C7_seeds"]
    log(f"  C5 calm (hprox_med < {CALM_S:.0f}, n {c5['n_calm']:,}) sum g {c5['calm_sum_g']:+,.0f} >= {c5['bound']:+,.0f}: {c5['passes']}")
    log(f"  C6 ex-monster (y <= {MONSTER_S:.0f}, n {c6['n_exmonster']:,}) sum g {c6['exmonster_sum_g']:+,.0f} > 0: {c6['passes']}")
    log(f"  C7 pooled {c7['pooled_sum_g']:+,.0f} vs 2 x seed sd {2 * c7['seed_sd']:,.0f} (per seed {c7['per_seed_sum_g']}): {c7['passes']}")
    at = record["reported_not_decisional"]["attribution_E5w_vs_E5"]
    log(f"  attribution (reported): sum g E5w {at['sum_g_E5w']:+,.0f} + physics increment {at['sum_physics_increment']:+,.0f} "
        f"= E5 {at['sum_g_E5']:+,.0f}; weather share {at['weather_share']}")
    for a, rec in reported.items():
        log(f"  [reported] {a}: sum g {rec['pooled_sum_g']:+,.0f}, C1 95% [{rec['C1_date_block_bootstrap']['ci95'][0]:+,.0f}, "
            f"{rec['C1_date_block_bootstrap']['ci95'][1]:+,.0f}], event-excluded price {rec['C2_priced_2026']['priced_mse']:+.1f}, "
            f"months {rec['C3_months']['months_positive']}, airports {rec['C4_airports']['airports_positive']}; would read {rec['verdict']}")
    log(f"  [reported] sensitivities: {record['reported_not_decisional']['sensitivities']}")
    for name, tab in record["reported_not_decisional"]["cuts"].items():
        _log_cut(log, name, tab)
    lr = record["descriptive"]["lirf_report_only"]
    log(f"  LIRF (report-only): n {lr['n']:,} sum g {lr['sum_g']:+,.0f}")
    log(f"  VERDICT ({PREREG}): {c['verdict']}")
    record["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    record["wall_s"] = round(time.time() - _T0, 1)
    record["peak_rss_gb"] = round(peak_rss_gb(), 2)
    paths["json"].write_text(json.dumps(record, indent=2, default=_jsonable))
    log(f"json -> {paths['json']}   wall {record['wall_s']:.0f}s   peak RSS {record['peak_rss_gb']:.2f} GB")
    return 0


# =============================================================================================
# ship (only after a WORKING verdict): v10 with the non-LIRF unmatched rows replaced by E5
# =============================================================================================

def check_ship_allowed(record: dict, base_name: str) -> None:
    """The ship rule's preconditions: the stored E5 record reads WORKING, is not a smoke, carries a reproduced S1C guard,
    and the base is the registered v10. Raises SystemExit otherwise."""
    if record.get("tag") != TAG or record.get("prereg") != PREREG:
        raise SystemExit(f"the result record is not {TAG}'s ({record.get('tag')!r}, {record.get('prereg')!r})")
    if record.get("smoke"):
        raise SystemExit("the result record is a SMOKE run; nothing ships from a smoke")
    if record.get("verdict") != "WORKING":
        raise SystemExit(f"E5 reads {record.get('verdict')!r}; the ship rule is 'only if WORKING'")
    if record.get("lomo_metadata", {}).get("reproduction", {}).get("status") != "reproduced":
        raise SystemExit("the lomo parquet behind the record did not reproduce E3C's S1C; nothing ships")
    if base_name != REGISTERED_BASE:
        raise SystemExit(f"the prereg ships E5 on {REGISTERED_BASE}, not {base_name}; another base needs an amendment")


def verify_s1c_rebuilds_base(s1c, scored_unm: pd.DataFrame, base: pd.DataFrame) -> dict:
    """rint(S1C) from S1's refit parts, the 2026 witness and the congestion body must equal the base on EVERY scored
    non-LIRF unmatched row (v9 / v10 shipped exactly that); LIRF rows (R2 / E1) are not compared. Raises AssertionError."""
    m = us.non_lirf_mask(scored_unm)
    rebuilt = np.rint(np.asarray(s1c, dtype="float64")).astype("int64")[m]
    ids = scored_unm.MVT_ID_mvt.to_numpy()[m]
    have = base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy(dtype="float64")
    if np.isnan(have).any():
        raise AssertionError(f"{int(np.isnan(have).sum())} scored non-LIRF unmatched ids are missing from the base submission")
    bad = int((rebuilt != have.astype("int64")).sum())
    if bad:
        raise AssertionError(f"the S1C path does not rebuild the base on {bad} of {len(have)} scored non-LIRF unmatched rows; "
                             "nothing is written")
    return {"n_checked": int(m.sum()), "n_lirf_not_compared": int((~m).sum())}


def check_serve_coverage(scored_p: pd.DataFrame, spec: dict) -> dict:
    """Every input the E5 body reads beyond S1C's must carry at least one finite / non-null value on the scored non-LIRF
    unmatched rows; an input NaN everywhere means the 2026 build failed (it would silently route every row through the
    missing branch). Returns the per-input 2026 NaN share. Raises ValueError."""
    m = us.non_lirf_mask(scored_p)
    rows = scored_p[m]
    share, empty = {}, []
    for c in [*spec["numeric"][len(uc.CongestionRegressor.NUMERIC):], *spec["extra"]]:
        if c in spec["extra"]:
            ok = rows[c].notna().to_numpy()
        else:
            ok = np.isfinite(rows[c].to_numpy(dtype="float64"))
        share[c] = float(1.0 - ok.mean()) if len(ok) else 1.0
        if not ok.any():
            empty.append(c)
    if empty:
        raise ValueError(f"the 2026 inputs {empty} are NaN on every scored non-LIRF unmatched row; the 2026 build failed")
    return share


def build_ship(train_unm: pd.DataFrame, scored_unm: pd.DataFrame, parts: dict, base: pd.DataFrame, serve_frame: pd.DataFrame,
               obs: pd.DataFrame, scored_physics: pd.DataFrame, spec: dict) -> tuple:
    """(new submission frame, info). `scored_unm` and `parts` are row-aligned (fit_unmatched's order) and arrive WITHOUT
    witness or physics columns: the serve witness is attached here from `serve_frame` (the serve file's DEP rows), the
    physics from `obs` (as-of MVT_TIME) and `scored_physics` (the serve file's unmatched build). Guard 1 (the S1C path
    rebuilds the base on every non-LIRF row) sits here, before anything is spliced."""
    if not parts.get("hybrid"):
        raise ValueError("the parts are not the hybrid path's (fit_unmatched(..., hybrid=True))")
    scored_w, wsum = us.attach_serve_witness(scored_unm, serve_frame)
    s1c, nf_fit_c, _, n_nonfill = us.s1c_from_parts(train_unm, scored_w, parts)
    rebuild = verify_s1c_rebuilds_base(s1c, scored_w, base)                              # guard 1
    scored_p = attach_physics(scored_w, obs, scored_physics)
    coverage = check_serve_coverage(scored_p, spec)
    nonfill = train_unm[~bs.schedule_fill(train_unm)]
    seeds = tuple(int(s) for s in parts["seeds"])
    by_seed = {s: fit_physics_regressor(nonfill, s, spec["numeric"], spec["extra"]).predict(scored_p) for s in seeds}
    nf_fit = bs.mean_over_seeds(by_seed)
    e5 = bs.mixture(parts["p"], parts["sp"], bs.nf_hybrid(parts["nf_cells"], nf_fit))
    m = us.non_lirf_mask(scored_p)
    te = scored_p[m]
    ids = te.MVT_ID_mvt.to_numpy()
    vals = np.rint(e5[m])
    if not (np.isfinite(vals).all() and (vals >= 1.0).all()):
        raise AssertionError("E5 produced a non-finite or sub-floor value; nothing is written")
    out = bs.splice_unmatched(base, ids, vals)                                            # guard 2, part 1
    pos = pd.Index(base.MVT_ID_mvt).get_indexer(ids)
    if not np.array_equal(out.TAXITIME_SEC_mvt.to_numpy()[pos].astype("int64"), vals.astype("int64")):
        raise AssertionError("rint(E5) does not fit the submission dtype")
    changed = np.flatnonzero(out.TAXITIME_SEC_mvt.to_numpy() != base.TAXITIME_SEC_mvt.to_numpy())
    if not set(changed.tolist()) <= set(pos.tolist()):                                    # guard 2, part 2
        raise AssertionError("a row outside the scored non-LIRF unmatched set changed; nothing is written")
    before = base.TAXITIME_SEC_mvt.to_numpy()[pos].astype("float64")
    after = vals.astype("float64")
    moved = after != before
    detail = pd.DataFrame({"MVT_ID_mvt": ids, "ADEP_mvt": te.ADEP_mvt.astype(str).to_numpy(),
                           "month": te.month.to_numpy().astype("int64"), "sp": te.sp.to_numpy(dtype="float64"),
                           "hprox_med": te.hprox_med.to_numpy(dtype="float64"), "hprox_n": te.hprox_n.to_numpy(dtype="int64"),
                           "bin": uc.bin_of(te.hprox_med), "deicing_condition": te.deicing_condition.to_numpy(dtype="float64"),
                           INTERACTION: te[INTERACTION].to_numpy(dtype="float64"), "obs_age_s": te.obs_age_s.to_numpy(dtype="float64"),
                           "p": np.asarray(parts["p"], dtype="float64")[m], "before": before, "after": after})[DETAIL_COLUMNS]
    shift = after - before
    info = {"rebuild": rebuild, "witness_2026": wsum, "seeds": list(seeds), "n_train_nonfill": n_nonfill,
            "coverage_2026_nan_share": coverage, "spec": spec,
            "n_rows_replaced": int(m.sum()), "n_values_differ": int(moved.sum()),
            "n_routed_to_body": int((np.asarray(parts["nf_cells"], dtype="float64")[m] < bs.T_TAIL_S).sum()),
            "rms_shift_vs_base_s": float(np.sqrt(np.mean(shift ** 2))) if len(shift) else 0.0,
            "mean_shift_s": float(shift.mean()) if len(shift) else 0.0,
            "per_airport_mean_shift_s": {str(k): float(v) for k, v in pd.Series(shift).groupby(detail.ADEP_mvt.to_numpy()).mean().items()},
            "sse_shift_vs_base_board_mse": float((shift ** 2).sum() / len(base)),
            "nf_fit_e5": nf_fit[m], "nf_fit_c": nf_fit_c[m], "before": before, "after": after, "detail": detail}
    return out, info


def write_ship_outputs(out: pd.DataFrame, info: dict, template: pd.DataFrame, version: int, base_name: str,
                       sub_dir: pathlib.Path, record: dict, t0: float) -> tuple:
    """(dest, meta). check_submission on the frame, refuse an existing file, write, check the re-read file, then the detail
    parquet and the meta json beside it (the band to read the board delta against, from the record)."""
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
    info["detail"].to_parquet(sub_dir / f"{dest.stem}.unm_rows.parquet", index=False)
    cl = record.get("clauses", {})
    meta = {"version": int(version), "file": dest.name, "base": base_name, "arm": f"E5 on non-LIRF unmatched rows ({PREREG})",
            "prereg": PREREG, **_git(), "seeds": info["seeds"], "n_train_nonfill": info["n_train_nonfill"],
            "rebuild_guard": info["rebuild"], "witness_2026": info["witness_2026"], "spec": info["spec"],
            "coverage_2026_nan_share": info["coverage_2026_nan_share"], "weather_version": WEATHER_VERSION,
            "n_rows_replaced": info["n_rows_replaced"], "n_values_differ": info["n_values_differ"],
            "n_routed_to_body": info["n_routed_to_body"], "mean_shift_s": info["mean_shift_s"],
            "rms_shift_vs_base_s": info["rms_shift_vs_base_s"], "per_airport_mean_shift_s": info["per_airport_mean_shift_s"],
            "sse_shift_vs_base_board_mse": info["sse_shift_vs_base_board_mse"],
            "board_band": {"conservative_event_excluded_price": cl.get("C2_priced_2026", {}).get("priced_mse"),
                           "optimistic_full_price": cl.get("C2_full_price_not_decisional", {}).get("priced_mse"),
                           "read": "board delta vs v10, against the event-excluded price (conservative) and the full price (optimistic)"},
            "detail_file": f"{dest.stem}.unm_rows.parquet", "wall_s": round(time.time() - t0, 1)}
    (sub_dir / f"{dest.stem}.meta.json").write_text(json.dumps(meta, indent=1, default=_jsonable))
    return dest, meta


def run_ship(args, log) -> int:
    t0 = time.time()
    rec_path = REPORTS / JSON_NAME
    if not rec_path.exists():
        raise SystemExit(f"{rec_path} is missing: run `lomo` and `score` first")
    record = json.loads(rec_path.read_text())
    base_path = ROOT / args.base
    check_ship_allowed(record, base_path.name)
    if not base_path.exists():
        raise SystemExit(f"--base {base_path} does not exist")
    base = pq.read_table(base_path).to_pandas()
    template = pq.read_table(RAW / "submitting.parquet").to_pandas()
    bs.check_submission(base, template)
    log(f"base {base_path.name}: {len(base):,} rows, dtype {base.TAXITIME_SEC_mvt.dtype}; template {len(template):,}")
    files = sorted(glob.glob(str(RAW / "training_2025-*.parquet")))
    if len(files) != 12:
        raise SystemExit(f"expected the 12 training months under {RAW}, found {len(files)}")
    unm, lirf, _, pinfo = load_real_physics(files, log)
    audit = audit_inputs(unm)
    spec = arm_specs(audit)[DECISIONAL]
    if spec != record.get("arms", {}).get(DECISIONAL):
        raise SystemExit(f"the 2025 frame builds E5 as {spec}, the scored record has {record.get('arms', {}).get(DECISIONAL)}")
    ranking = bs.derive(bs.load_movements([RAW / "ranking.parquet"]))
    serve_frame = ranking[us.WITNESS_COLUMNS].copy()
    scored = ranking[ranking.MVT_ID_mvt.isin(set(template.MVT_ID_mvt))].reset_index(drop=True)
    if len(scored) != len(template):
        raise ValueError(f"joined {len(scored)} scored rows, template has {len(template)}")
    scored_unm = scored[scored.unmatched.to_numpy()]
    del ranking
    gc.collect()
    log("--- the 2026 physics block: stand_ab.build_unmatched_ranking (fresh; compared to data/cache_unmatched/ranking.parquet) ---")
    rank_phys, rinfo = build_physics([RAW / "ranking.parquet"], S.build_unmatched_ranking, UCACHE, log)
    obs = load_weather_obs()
    parts: dict = {}
    bs.fit_unmatched(unm, scored_unm, train_matched=lirf, hybrid=True, seeds=bs.SEEDS, parts=parts)
    out, info = build_ship(unm, scored_unm, parts, base, serve_frame, obs, rank_phys, spec)
    log(f"guard 1: the S1C path rebuilds {base_path.name} on all {info['rebuild']['n_checked']:,} scored non-LIRF unmatched rows "
        f"({info['rebuild']['n_lirf_not_compared']:,} LIRF rows not compared)")
    log(f"2026 inputs, NaN share on the scored non-LIRF rows: {info['coverage_2026_nan_share']}")
    log(f"E5: replaced {info['n_rows_replaced']:,} rows, {info['n_values_differ']:,} values differ; mean shift {info['mean_shift_s']:+.1f} s, "
        f"RMS shift {info['rms_shift_vs_base_s']:.1f} s, sse shift {info['sse_shift_vs_base_board_mse']:.1f} board MSE")
    us.render_tables(us.shift_tables(info["detail"]), log)
    dest, meta = write_ship_outputs(out, info, template, args.version, base_path.name, ROOT / "submissions", record, t0)
    meta["physics_2026"] = rinfo
    (ROOT / "submissions" / f"{dest.stem}.meta.json").write_text(json.dumps(meta, indent=1, default=_jsonable))
    log(json.dumps(meta, indent=1, default=_jsonable))
    log(f"wrote {dest}; read the board delta vs v10 against {meta['board_band']}")
    return 0


# =============================================================================================
# main
# =============================================================================================

def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("lomo", "score", "ship"))
    ap.add_argument("--smoke", action="store_true", help=f"lomo / score only: synthetic world, {SMOKE['n_boot']} draws, "
                                                         f"{SMOKE_DIR.relative_to(ROOT)}/, the banner")
    ap.add_argument("--base", help=f"ship only: the base submission ({REGISTERED_BASE})")
    ap.add_argument("--version", type=int, help="ship only: the new version N")
    a = ap.parse_args(argv)
    if a.step == "ship":
        if a.smoke:
            ap.error("ship has no --smoke")
        if a.base is None or a.version is None:
            ap.error("ship needs --base and --version")
    elif a.base is not None or a.version is not None:
        ap.error("--base / --version apply to ship only")
    return a


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.step == "ship":
        return run_ship(args, _Log())
    paths = output_paths(args.smoke)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    log = _Log(paths["lomo_log"] if args.step == "lomo" else paths["score_log"])
    try:
        if args.smoke:
            log(SMOKE_BANNER)
        log(f"unm_physics {args.step}  smoke={args.smoke}  prereg={PREREG}  paths={ {k: str(v) for k, v in paths.items()} }")
        rc = run_lomo(args, paths, log) if args.step == "lomo" else run_score(args, paths, log)
    finally:
        if args.smoke:
            log(SMOKE_BANNER)
        log.close()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
