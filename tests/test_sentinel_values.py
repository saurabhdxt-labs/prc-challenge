"""Bug class `sentinel-ingested-as-measurement` (post-mortem 2026-09-10, reports/bug_classes.md BC-3).

An external field is trusted as a measurement when its provider fills it with a sentinel — a literal
zero, a cap, a placeholder — for sources that never report it. The value is syntactically valid, so
every range and type check passes while it carries no information. It surfaced as the weather block's
precipitation: the Iowa Mesonet archive returns p01i = "0.00" on all 205,417 European observations,
so `w_precip_mm` was 0 on every row and the freezing flag missed cold rain (arm W, RESULT 22, was
measured on that block).

The class-level guard below walks every numeric column of every cache the feature builders write
and refuses a column that is constant across a whole month file, unless it is on an explicit
allow-list with the reason it is legitimately constant. The other tests pin each sibling found in
the codebase-wide sweep: fixed (a) or safe by a named invariant (b).

Rehearsed 2026-09-11 02:20 EDT (from `date`): one column dropped from stand_ab.UNMATCHED_NAN_COLS turns the
class guard RED (the all-NaN exemption is per column, not per cache); the stale-cache test was seen
RED on the 2.0.0 cache after the 2.0.1 bump, before the rebuild.
"""
from __future__ import annotations

import glob
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
JAN = "training_2025-01-01_2025-02-01.parquet"

_spec = importlib.util.spec_from_file_location("stand_ab", ROOT / "scripts" / "stand_ab.py")
stand_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stand_ab)

FEATURE_CACHES = ("cache_stand", "cache_weather", "cache_order", "cache_day", "cache_queue", "cache_unmatched")
needs_caches = pytest.mark.skipif(not all((DATA / c / JAN).exists() for c in FEATURE_CACHES),
                                  reason="the feature caches are built from challenge data that is not in the repo")
needs_weather = pytest.mark.skipif(not (DATA / "weather").exists() or len(list((DATA / "weather").glob("*.csv"))) < 140,
                                   reason="the weather archive is built by scripts/fetch_weather.py and is not in the repo")
needs_raw = pytest.mark.skipif(not (RAW / JAN).exists() or not (RAW / "ranking.parquet").exists(),
                               reason="challenge data is not redistributable")

#: columns that are LEGITIMATELY constant across every feature file of a cache, each with its reason.
#: A column not listed here that turns out constant is a sentinel until proven otherwise.
CONSTANT_OK = {
    ("*", "month"): "a month file holds one calendar month by construction",
    ("*", "sched_sec"): "SCHED_TIME is minute-precise (seconds are 0 on every row; pinned below)",
    ("cache_stand", "actype_null"): "AIRCRAFT_TYPE is never null on a matched departure (pinned below)",
}
#: cache_unmatched columns that are NaN on every row BY CONSTRUCTION, named one by one from the
#: module's own lists (tests/test_unmatched_features.py pins the pattern row by row): the columns
#: anchored on AOBT_3, the columns read from the *_flt clocks, and the target `delta`, which needs
#: an AOBT_3. NaN is the honest encoding of unknown there, not a sentinel. Review 2026-09-11: this
#: used to exempt the whole cache.
ALL_NAN_OK = {"cache_unmatched": set(stand_ab.UNMATCHED_NAN_COLS) | set(stand_ab.UNMATCHED_FLT_COLS) | {"delta"}}


def _feature_files(cache: str) -> list[pathlib.Path]:
    """The feature files of a cache: the training months and the scored file. cache_stand also holds
    fold-prediction records (fold_preds_*), which are not features and are not scanned."""
    d = DATA / cache
    return sorted(d.glob("training_*.parquet")) + [d / "ranking.parquet"]


@needs_caches
def test_no_cache_column_is_a_constant_sentinel():
    """The class guard. Every numeric column of every feature cache, over ALL its feature files
    (the twelve training months and the scored file), must take more than one value — or be on
    CONSTANT_OK with its reason, or be one of the named all-NaN-by-construction columns. A
    placeholder is constant everywhere; a rare event (thunder in December) is not, which is why the
    check is over the union of files rather than month by month.

    Failed on the pre-fix weather cache: `w_precip_mm` was the constant 0.0 on every row. Review
    2026-09-11 widened it: cache_stand was not scanned (it holds `actype_null`, constant 0 — true,
    pinned below), only January was read, and all of cache_unmatched was exempt from the NaN check.
    """
    offenders = []
    for cache in FEATURE_CACHES:
        lo, hi, seen = {}, {}, {}
        files = _feature_files(cache)
        assert len(files) == 13 and all(f.exists() for f in files), f"{cache}: expected 12 months + ranking"
        for f in files:
            d = pd.read_parquet(f)
            for c in d.columns:
                if c == "MVT_ID_mvt" or not pd.api.types.is_numeric_dtype(d[c]):
                    continue
                v = d[c].to_numpy(dtype="float64")
                v = v[np.isfinite(v)]
                seen.setdefault(c, False)
                if len(v):
                    seen[c] = True
                    lo[c], hi[c] = min(lo.get(c, np.inf), v.min()), max(hi.get(c, -np.inf), v.max())
        for c, ok in seen.items():
            if not ok and c not in ALL_NAN_OK.get(cache, set()):
                offenders.append(f"{cache}.{c}: NaN on every row of every file")
            if ok and c in ALL_NAN_OK.get(cache, set()):
                offenders.append(f"{cache}.{c}: named all-NaN by construction but carries values")
            if ok and lo[c] == hi[c] and (cache, c) not in CONSTANT_OK and ("*", c) not in CONSTANT_OK:
                offenders.append(f"{cache}.{c}: constant {lo[c]:g} across all {len(files)} files")
    assert not offenders, "constant columns ingested as data: " + "; ".join(offenders)


@needs_raw
def test_actype_null_is_zero_because_matched_departures_always_carry_a_type():
    """Sibling (b), on CONSTANT_OK. cache_stand.actype_null is 0 on every row of every file: true data,
    not a placeholder — AIRCRAFT_TYPE is populated on every matched departure (0 nulls among
    2,402,128, measured 2026-09-11 over the twelve months and the scored file). Checked here on
    January and the scored file; if a future file carries a null type the flag starts to vary and
    this reason must be re-examined."""
    for name in (JAN, "ranking.parquet"):
        t = pq.read_table(RAW / name, columns=["PHASE_mvt", "AIRCRAFT_TYPE_mvt", "AOBT_3_flt"]).to_pandas()
        dep = t[(t.PHASE_mvt == "DEP") & t.AOBT_3_flt.notna()]
        assert len(dep) > 100_000 and dep.AIRCRAFT_TYPE_mvt.notna().all(), name


def _archive(tmp_path, rows) -> pathlib.Path:
    d = tmp_path / "wx"; d.mkdir()
    head = "station,valid,tmpf,dwpf,sknt,gust,vsby,p01i,wxcodes,metar"
    (d / "EDDF_2025-01.csv").write_text("\n".join([head] + [",".join(map(str, r)) for r in rows]) + "\n")
    return d


def test_gust_is_unknown_when_the_wind_itself_is_unknown(tmp_path):
    """Sibling (a). A METAR omits the gust group when there is no gust, so a missing gust with a
    REPORTED wind is a true 0 kt. But when the wind group is missing too, nothing is known, and
    writing 0 kt fabricates calm — the same shape as the precipitation defect (7 real rows).

    Fails on the pre-fix loader, which turned every missing gust into 0.
    """
    w = stand_ab.load_weather(_archive(tmp_path, [
        ("EDDF", "2025-01-10 05:00", 41.0, 35.6, 12.0, "M", 6.21, "0.00", "M", "x"),    # wind reported, no gust
        ("EDDF", "2025-01-10 06:00", 41.0, 35.6, "M", "M", 6.21, "0.00", "M", "x"),     # wind MISSING
        ("EDDF", "2025-01-10 07:00", 41.0, 35.6, 22.0, 35.0, 6.21, "0.00", "M", "x"),   # gusting
    ])).sort_values("valid").reset_index(drop=True)
    assert w.w_gust_kt.iloc[0] == 0.0, "a reported wind with no gust group is a true 0 kt"
    assert np.isnan(w.w_gust_kt.iloc[1]), "with the wind missing, the gust is unknown, not 0"
    assert w.w_gust_kt.iloc[2] == 35.0


def test_visibility_cap_is_far_from_the_low_visibility_flag(tmp_path):
    """Sibling (b), safe by a named invariant. European METARs report "9999" for 10 km or more, which
    the archive gives as 6.21 statute miles on 87% of observations — a right-censored cap, not an
    exact value; since 2026-09-11 it is read back as exactly 10.0 km. The low-visibility flag reads
    only values below 1.5 km, so the cap can never reach it; the continuous w_vis_km column means
    ">= 10 km" at its maximum, which a tree treats as the top category. Pinned so a change to the
    flag's threshold cannot silently cross the cap."""
    w = stand_ab.load_weather(_archive(tmp_path, [
        ("EDDF", "2025-01-10 05:00", 41.0, 35.6, 8.0, "M", 6.21, "0.00", "M", "x"),
        ("EDDF", "2025-01-10 06:00", 41.0, 35.6, 8.0, "M", 0.62, "0.00", "FG", "x"),
    ])).sort_values("valid").reset_index(drop=True)
    cap_km = 10.0
    assert stand_ab.WX_LOWVIS_KM < cap_km / 5, "the flag threshold must stay far below the 10 km cap"
    assert w.w_vis_km.iloc[0] == cap_km and w.w_lowvis.iloc[0] == 0.0
    assert w.w_lowvis.iloc[1] == 1.0


@needs_weather
def test_thunder_is_rare_but_real_and_follows_the_codes():
    """Sibling (b). w_thunder is 0 on 99.94% of January rows — a rare event, not a sentinel: across the
    year the archive carries 884 TS codes and the flag is 1 on exactly those observations (the
    load-bearing assertion). Thunder is also more common in July than January, the physical
    direction; the ratio MEASURED on 2026-09-10 is 6.2x (0.81% vs 0.13%). An earlier draft of this
    test asserted ">10x" without measuring it and went red on a correct flag — the direction is the
    claim, not an invented multiple."""
    w = stand_ab.load_weather()
    raw = pd.concat([pd.read_csv(f, dtype=str, usecols=["valid", "station", "wxcodes"])
                     for f in sorted(stand_ab.WEATHER_RAW.glob("*.csv"))], ignore_index=True)
    raw["valid"] = pd.to_datetime(raw.valid, utc=True)
    raw = raw.sort_values("valid", kind="mergesort").reset_index(drop=True)
    ts = raw.wxcodes.fillna("").str.upper().str.contains("TS", regex=False).to_numpy()
    assert ts.sum() > 500
    assert np.array_equal(w.w_thunder.to_numpy() == 1.0, ts)
    # since 2026-09-11 an unobservable weather group (15 AUTO reports) is unknown, not "no thunder"
    assert np.array_equal(np.isnan(w.w_thunder.to_numpy()), raw.wxcodes.isna().to_numpy())
    month = w.valid.dt.month.to_numpy()
    thunder = w.w_thunder.to_numpy()
    assert np.nanmean(thunder[month == 7]) > np.nanmean(thunder[month == 1]) > 0


@needs_raw
def test_schedule_seconds_are_zero_because_schedules_are_minute_precise():
    """Sibling (b). cache_unmatched.sched_sec is the constant 0 — true data, not a sentinel:
    SCHED_TIME carries whole minutes on every departure of the raw files. A constant true column is
    never split on by a tree, so it is harmless; this pins the reason it is on CONSTANT_OK."""
    t = pq.read_table(RAW / JAN, columns=["PHASE_mvt", "SCHED_TIME_UTC_mvt"]).to_pandas()
    s = t.SCHED_TIME_UTC_mvt[t.PHASE_mvt == "DEP"].dropna()
    assert len(s) > 100_000 and (s.dt.second == 0).all()


@needs_raw
def test_scored_rows_have_complete_schedule_and_movement_times():
    """Sibling (b). The unmatched lane's design matrix (`build_submission.py`, the L-e logistic) ends in
    np.nan_to_num, which would fabricate sp = 0 or dayoff = 0 if a schedule or movement time were
    ever missing. It never fires, because both are populated on every scored departure. Pinned: if a
    future scored file breaks this, the nan_to_num becomes the same defect as the precipitation zero."""
    r = pq.read_table(RAW / "ranking.parquet", columns=["PHASE_mvt", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"]).to_pandas()
    dep = r[r.PHASE_mvt == "DEP"]
    assert len(dep) == 344_841
    assert dep.SCHED_TIME_UTC_mvt.notna().all() and dep.MVT_TIME_UTC_mvt.notna().all()


@needs_caches
def test_no_feature_list_sends_a_text_column_through_the_coercing_design_matrix():
    """Sibling (b), safe by an invariant this test makes explicit. Four design-matrix builders
    (`lgbm_submit.design_matrix`, `stand_ab` and `lgbm_ab` inline) convert every feature with
    `pd.to_numeric(errors="coerce")`, which would turn a text column into an all-NaN column — a dead
    feature a tree silently ignores, the same shape as the zero-precipitation column. It never
    happens because no feature list names a text cache column (the categoricals reach the model as
    numeric encodings). Pinned against the real cache schemas, so adding e.g. `ADEP_mvt` to a
    feature list fails here instead of training on NaN. The class guard above skips non-numeric
    columns, which is why this check is separate."""
    lf = _load("lgbm_fold")
    ls = lf.L
    lists = {n: getattr(m, n) for m in (ls, lf, stand_ab) for n in dir(m)
             if (n.startswith("FEATS") or n.endswith("_FEATS")) and isinstance(getattr(m, n), (list, tuple))}
    assert {"FEATS", "WEATHER_FEATS", "BASELINE_FEATS"} <= set(lists), sorted(lists)
    text = set()
    for cache in ("cache_stand", "cache_weather", "cache_order", "cache_day", "cache_queue", "cache_unmatched"):
        for f in pq.read_schema(DATA / cache / JAN):
            if not str(f.type).startswith(("double", "float", "int", "uint", "bool")):
                text.add(f.name)
    assert "ADEP_mvt" in text, "the schema read must see the known text columns, or it checks nothing"
    offenders = {n: sorted(set(v) & text) for n, v in lists.items() if set(v) & text}
    assert not offenders, f"feature lists naming text columns (would be coerced to all-NaN): {offenders}"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_caches
@needs_weather
def test_the_weather_cache_on_disk_was_built_by_this_parser():
    """A stale cache is the same defect one step removed: a fold reads v1 months while the code says
    v2. The readers (lgbm_fold, lgbm_submit) do not check the manifest, so this does: the cache's
    manifest must name this parser's VERSION, this archive's digest and this WEATHER_FEATS. Red the
    moment the parser is bumped and the cache not rebuilt (seen RED 2026-09-11 on the 2.0.0 cache
    after the 2.0.1 bump, before the rebuild)."""
    import json
    m = json.loads((DATA / "cache_weather" / "manifest.json").read_text())
    assert m["parser_version"] == stand_ab.WX.VERSION, (m["parser_version"], stand_ab.WX.VERSION)
    assert m["archive_digest"] == stand_ab.WX.archive_digest(stand_ab.WEATHER_RAW)
    assert m["weather_feats"] == list(stand_ab.WEATHER_FEATS)
