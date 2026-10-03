"""R12 — the ADS-B feature store over any set of days, and the train / serve tables built from it.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/r12_build.py features --store data/r12/features_<stamp> [--final]
        [--days 2025-03-10,2026-01-15] [--rss-limit-gib 6]
    /opt/homebrew/bin/python3.11 -B scripts/analysis/r12_build.py tables --store data/r12/features_<stamp>
        --out data/r12/tables_<stamp> [--days ...] [--rss-limit-gib 6]
    (heavy runs go through scripts/analysis/r12_launch.py)

What it is for: retrain the matched model on all 12 months of 2025 with ADS-B features (the "R12" arm). ADS-B day files
live in three directories — data/adsb/v2 (Jan + Jul 2025), data/adsb/v2/y2025_other (the other ten months; fetched by
the A12 run) and data/adsb/v2/y2026 (Jan + Jul 2026) — each with its own MANIFEST.json.

The store (data/r12/features_<stamp>/, stamp = sha256(adsb_features.FEATURES_VERSION)[:12], full text in STAMP.json):
  * one `day_<YYYY-MM-DD>.parquet` (MVT_ID_mvt, adsb_features.FEATURES [35 since DWG], coverage) + `day_<...>.json` per MVT
    day. The json is written last and is the "done" marker; the parquet is written to a temp name and renamed.
  * a day's flights = EVERY universe flight whose MVT falls on that UTC day (training: all matched rows of the twelve
    cached 2025 months; serve: the 339,551 matched ranking rows). The batch matters: adsb_features' leave-one-out stand
    centroids are computed from the batch.
  * a day is built by the shipped `adsb_features.build_period` (BC-25 "TRC" and BC-26 "DWG" fixes in its version) on a
    private directory of symlinks to exactly that day's previous / current / next files, so the three source
    directories can be read as one without touching them.
  * a day whose file or an in-universe neighbour is not yet available is DEFERRED (not written) unless --final, when a
    missing current day is recorded as status "missing" (counted; its rows become NaN + adsb_day_present 0) and a
    missing neighbour is recorded in the json.
  * resume: a done day is skipped only if its stamp, flight set and source files equal what a build now would use;
    otherwise the run REFUSES ("stale" / "flight set" / "features_version") — never mixes, never overwrites.

Tables: `assemble(universe, store)` is the ONE function for both the training table and the 2026 serve table (serve
parity): columns TABLE_COLS, keyed by (month, pos) = (cache month, position in that cache file / in the ranking cache).
Forbidden inputs never enter: build_period reads only adsb_features.FLIGHT_COLS (no label, no *_sec, no BLOCK/TAXITIME).
Standard library + numpy/pandas/pyarrow only (no model family: BC-5).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd

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

ADSB = ROOT / "data" / "adsb" / "v2"
SOURCE_DIRS = (ADSB, ADSB / "y2025_other", ADSB / "y2026")
STAND_DIR, ORDER_DIR, RAW_DIR = ROOT / "data" / "cache_stand", ROOT / "data" / "cache_order", ROOT / "data" / "raw"
RANK_CACHE, RANKING = STAND_DIR / "ranking.parquet", RAW_DIR / "ranking.parquet"
OUT_ROOTS = ("data/r12", "logs/r12_scratch")                 # relative to the root the caller passes
PROTECTED = tuple(ROOT / p for p in ("data/adsb", "data/cache_stand", "data/cache_order", "data/raw", "data/nsc_v34",
                                     "data/blx", "logs/a12_scratch", "submissions"))
FEATURES = list(AF.FEATURES)
UNIVERSE_COLS = ["month", "pos", "MVT_ID_mvt"] + [c for c in AF.FLIGHT_COLS if c != "MVT_ID_mvt"] + ["ap", "mvt_day"]
TABLE_COLS = ["month", "pos", "MVT_ID_mvt", "ap", "mvt_day", "adsb_day_present", "coverage"] + FEATURES
DAY_RE = re.compile(r"^adsb_(\d{4}-\d{2}-\d{2})\.parquet$")
TRC_MARK = "TRC"
#: day files that predate the fetcher's manifest regime (their MANIFEST entry is "skipped" with previous null) but are
#: complete and were read by the shipped v4 tables; accepted only after legacy_full_day() checks the content
LEGACY_UNLISTED = {
    "2025-01-01": "fetched 2026-09-10 before the manifest kept 'ok' records; 24.0 h span, 10 airports (checked 09-25)",
    "2026-01-01": "fetched 2026-09-11 before the manifest kept 'ok' records; 24.0 h span, 10 airports (checked 09-25)",
}
LEGACY_MIN_AIRPORTS, LEGACY_EDGE_S = 9, 600.0


# ---------------------------------------------------------------- stamp / version

def stamp_of(version: str) -> str:
    """The store directory's stamp: sha256 of the full FEATURES_VERSION text, first 12 hex digits."""
    return hashlib.sha256(version.encode()).hexdigest()[:12]


def require_trc(version: str) -> str:
    """Refuse a features version without the BC-25 take-off-roll fix (reports/bug_classes.md BC-25)."""
    if TRC_MARK not in version:
        raise ValueError(f"features version lacks the {TRC_MARK} fix (BC-25): {version!r}")
    return version


# ---------------------------------------------------------------- sources

def manifest_days(src_dir) -> set:
    """Days a directory's MANIFEST.json records as fetched: status ok, or status skipped with an ok previous record."""
    p = pathlib.Path(src_dir) / "MANIFEST.json"
    if not p.exists():
        raise FileNotFoundError(f"{p}: no MANIFEST.json")
    try:
        man = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"unreadable manifest {p}: {e}") from e
    out = set()
    for d in man.get("days", []):
        prev = d.get("previous") or {}
        if d.get("status") == "ok" or (d.get("status") == "skipped" and prev.get("status") == "ok"):
            out.add(d["day"])
    return out


def legacy_full_day(path, day: str) -> bool:
    """A legacy file is acceptable only if its samples span the whole UTC `day` (first within LEGACY_EDGE_S of 00:00,
    last within LEGACY_EDGE_S of 24:00) at >= LEGACY_MIN_AIRPORTS airports."""
    d = pd.read_parquet(path, columns=["t", "airport"])
    lo = (pd.Timestamp(day, tz="UTC") - AF.EPOCH).total_seconds()
    return bool(len(d) and d.t.min() >= lo and d.t.min() <= lo + LEGACY_EDGE_S and d.t.max() < lo + 86_400
                and d.t.max() >= lo + 86_400 - LEGACY_EDGE_S and d.airport.nunique() >= LEGACY_MIN_AIRPORTS)


def resolve_sources(dirs=SOURCE_DIRS, legacy=None) -> tuple:
    """(day -> file path, diagnostics). A day counts only if its file exists AND its directory's manifest records it
    (the fetcher renames a finished file into place, but on a resume it rebuilds the manifest; an unlisted file is
    'unverified' and excluded), or it is in `legacy` (default LEGACY_UNLISTED) and passes legacy_full_day. A day
    present in two directories is refused."""
    legacy = LEGACY_UNLISTED if legacy is None else legacy
    sources, unverified, orphans, accepted = {}, [], [], []
    for d in dirs:
        d = pathlib.Path(d).resolve()                  # every returned path is absolute (BC-40)
        listed = manifest_days(d)
        on_disk = {m.group(1) for m in (DAY_RE.match(p.name) for p in d.iterdir()) if m}
        for day in sorted((on_disk - listed) & set(legacy)):
            if legacy_full_day(d / f"adsb_{day}.parquet", day):
                listed = listed | {day}
                accepted.append(day)
        for day in sorted(on_disk & listed):
            if day in sources:
                raise ValueError(f"{day} is in two source directories: {sources[day].parent} and {d}")
            sources[day] = d / f"adsb_{day}.parquet"
        unverified += sorted(on_disk - listed)
        orphans += sorted(listed - on_disk)
    return sources, {"unverified_files": sorted(unverified), "manifest_without_file": sorted(orphans),
                     "legacy_accepted": sorted(accepted), "n_sources": len(sources)}


def source_contains(path) -> list:
    """The sources a day file holds, as its directory's MANIFEST.json declares them (`contains`); a manifest without the key
    is adsb.lol (every directory before plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md)."""
    man = json.loads((pathlib.Path(path).parent / "MANIFEST.json").read_text())
    return sorted(man.get("contains", ["adsb.lol"]))


def _plan_contains(plan: dict, sources: dict) -> list:
    """The sources the files of a day plan declare (their directories' MANIFEST.json)."""
    return sorted({c for d in plan["files"].values() for c in source_contains(sources[d])})


def _record_contains(rec: dict) -> list:
    """What a store record was built from. A record written before `source_contains` existed was built from adsb.lol when
    it read files (every store before AFI) and from nothing when it read none."""
    return rec.get("source_contains", ["adsb.lol"] if rec.get("files") else [])


def store_sources(store_dir, days) -> list:
    """The union of the sources the store's records of `days` were built from (a record without the key: adsb.lol)."""
    out = set()
    for day in days:
        p = pathlib.Path(store_dir) / f"day_{day}.json"
        if not p.exists():
            raise FileNotFoundError(f"{p}: no store record for {day}")
        out.update(json.loads(p.read_text()).get("source_contains", ["adsb.lol"]))
    return sorted(out)


def source_universe() -> set:
    """Every day ADS-B could exist for in this project: all of 2025, January and July 2026."""
    rng = [pd.date_range("2025-01-01", "2025-12-31"), pd.date_range("2026-01-01", "2026-01-31"),
           pd.date_range("2026-07-01", "2026-07-31")]
    return {d.strftime("%Y-%m-%d") for r in rng for d in r}


def plan_day(day: str, sources: dict, universe: set, final: bool) -> dict:
    """What a build of `day` would read now. status: ready | deferred (wait: the day or an in-universe neighbour is not
    available yet and not --final) | missing (--final and no file for the day)."""
    if day not in universe:
        raise ValueError(f"{day} is outside the source universe")
    t = pd.Timestamp(day)
    prev, nxt = (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), (t + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    need = [n for n in (prev, nxt) if n in universe]
    missing_nb = sorted(n for n in need if n not in sources)
    files = {k: v for k, v in (("prev", prev), ("cur", day), ("next", nxt)) if v in sources}
    if day not in sources:
        return {"day": day, "status": "missing" if final else "deferred", "files": {}, "missing_neighbours": missing_nb}
    status = "deferred" if (missing_nb and not final) else "ready"
    return {"day": day, "status": status, "files": files, "missing_neighbours": missing_nb}


# ---------------------------------------------------------------- output discipline

def check_out_dir(path, root=ROOT) -> pathlib.Path:
    """A store / table directory: never inside a protected directory (the fetch's data/adsb, the caches, the raw files,
    v34's outputs, BLX's data, submissions), and under `root`/data/r12 or `root`/logs/r12_scratch."""
    p = pathlib.Path(path).resolve()
    for d in PROTECTED:
        if p == d.resolve() or d.resolve() in p.parents:
            raise PermissionError(f"{p} is inside the protected directory {d}")
    roots = [(pathlib.Path(root) / r).resolve() for r in OUT_ROOTS]
    if not any(p == r or r in p.parents for r in roots):
        raise PermissionError(f"{p} is outside the R12 output roots {[str(r) for r in roots]}")
    return p


def is_scratch(path) -> bool:
    """A smoke path (logs/r12_scratch): no heavy lock. Everything else the CLI writes takes the lock."""
    p, r = pathlib.Path(path).resolve(), (ROOT / "logs" / "r12_scratch").resolve()
    return p == r or r in p.parents


def flights_sha(ids) -> str:
    a = np.sort(np.asarray(ids, dtype="float64"))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def _stamp_check(store: pathlib.Path, version: str, create: bool) -> None:
    sp = store / "STAMP.json"
    if sp.exists():
        have = json.loads(sp.read_text())
        if have.get("features_version") != version:
            raise RuntimeError(f"{store}: features_version {have.get('features_version')!r} != {version!r}; "
                               "refusing to mix stamps")
    elif create:
        store.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps({"features_version": version, "stamp": stamp_of(version),
                                  "created_utc": dt.datetime.now(dt.timezone.utc).isoformat()}, indent=1))
    else:
        raise FileNotFoundError(f"{store}: no STAMP.json (not an R12 feature store)")


def default_builder(flights: pd.DataFrame, adsb_dir, git_sha: str) -> pd.DataFrame:
    """The shipped extractor, unchanged."""
    return AF.build_period(flights, adsb_dir, git_sha=git_sha)


def _build_one(day: str, fl: pd.DataFrame, plan: dict, sources: dict, store: pathlib.Path, builder, git_sha: str):
    links = store / f"_links_{day}"
    links.mkdir()
    made = []
    try:
        for d in plan["files"].values():
            q = links / f"adsb_{d}.parquet"
            os.symlink(pathlib.Path(sources[d]).resolve(), q)     # absolute: a relative target resolves from `links` (BC-40)
            made.append(q)
            if not q.is_file():                                     # a link the builder could not read would score as "no ADS-B"
                raise RuntimeError(f"{day}: cannot read {sources[d]} through {q}; the day is not built (BC-40)")
        out = builder(fl[AF.FLIGHT_COLS].reset_index(drop=True), links, git_sha)
    finally:
        for q in made:
            os.unlink(q)
        links.rmdir()
    want = np.sort(fl.MVT_ID_mvt.to_numpy(float))
    got = out.MVT_ID_mvt.to_numpy(float) if "MVT_ID_mvt" in out else np.array([])
    if len(got) != len(want) or not np.array_equal(np.sort(got), want):
        raise AssertionError(f"{day}: builder returned {len(got)} rows for {len(want)} flights (ids differ)")
    miss = [c for c in FEATURES + ["coverage"] if c not in out.columns]
    if miss:
        raise AssertionError(f"{day}: builder returned no column(s) {miss}")
    return out[["MVT_ID_mvt"] + FEATURES + ["coverage"]].reset_index(drop=True)


def build_store(universe: pd.DataFrame, sources: dict, store_dir, version: str, final: bool, git_sha: str,
                builder=default_builder, days=None, root=ROOT, universe_days=None, log=print) -> dict:
    """Build every (or the listed) day of `universe` into the store; resumable; returns the counts."""
    store = check_out_dir(store_dir, root)
    _stamp_check(store, version, create=True)
    universe_days = source_universe() if universe_days is None else universe_days
    if not universe.MVT_ID_mvt.is_unique:
        raise ValueError("universe MVT ids are not unique")
    groups = universe.groupby("mvt_day").groups
    todo = sorted(groups) if days is None else sorted(set(days))
    unknown = [d for d in todo if d not in groups]
    if unknown:
        raise ValueError(f"days with no universe flights: {unknown}")
    counts = {"built": 0, "done": 0, "deferred": 0, "missing": 0, "n_days": len(todo)}
    for day in todo:
        fl = universe.loc[groups[day]]
        fsha = flights_sha(fl.MVT_ID_mvt)
        jp, pp = store / f"day_{day}.json", store / f"day_{day}.parquet"
        now = plan_day(day, sources, universe_days, final=True)            # what a build now would read
        if jp.exists():
            have = json.loads(jp.read_text())
            if have.get("features_version") != version:
                raise RuntimeError(f"{jp.name}: features_version differs from {version!r}")
            if have.get("flights_sha") != fsha:
                raise RuntimeError(f"{jp.name}: flight set changed ({have.get('flights_sha')} -> {fsha})")
            if have.get("files") != now["files"] or have.get("status") != now["status"]:
                raise RuntimeError(f"{jp.name} is stale: built from {have.get('files')} ({have.get('status')}), now "
                                   f"{now['files']} ({now['status']}); move it aside (mv) to rebuild")
            if _record_contains(have) != _plan_contains(now, sources):        # same day files, other sources (BC-41)
                raise RuntimeError(f"{jp.name} is stale: built from sources {_record_contains(have)}, the source "
                                   f"directories now declare {_plan_contains(now, sources)}; move it aside (mv) to rebuild")
            counts["done"] += 1
            continue
        if pp.exists():
            raise RuntimeError(f"{pp.name} exists without its json (an interrupted write); move it aside (mv)")
        plan = plan_day(day, sources, universe_days, final=final)
        if plan["status"] == "deferred":
            counts["deferred"] += 1
            continue
        t0 = time.time()
        rec = {"day": day, "status": plan["status"], "features_version": version, "stamp": stamp_of(version),
               "n_flights": int(len(fl)), "flights_sha": fsha, "files": plan["files"],
               "source_paths": {k: str(sources[d]) for k, d in plan["files"].items()},
               "source_bytes": {k: int(sources[d].stat().st_size) for k, d in plan["files"].items()},
               "source_contains": _plan_contains(plan, sources),
               "missing_neighbours": plan["missing_neighbours"], "git_sha": git_sha}
        if plan["status"] == "ready":
            out = _build_one(day, fl, plan, sources, store, builder, git_sha)
            tmp = store / f".day_{day}.parquet.tmp"
            out.to_parquet(tmp, index=False)
            tmp.rename(pp)
            rec["coverage_counts"] = {str(k): int(v) for k, v in out.coverage.value_counts().sort_index().items()}
            counts["built"] += 1
        else:
            counts["missing"] += 1
        rec.update(wall_s=round(time.time() - t0, 1), built_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        jp.write_text(json.dumps(rec, indent=1))
        log(f"{day}: {rec['status']} n={len(fl):,} files={list(plan['files'].values())} "
            f"missing_nb={plan['missing_neighbours']} wall={rec['wall_s']}s")
    return counts


# ---------------------------------------------------------------- tables (one function for train AND serve)

def table_report(t: pd.DataFrame) -> dict:
    """The per-table part of a tables directory's report.json: rows, rows on days without ADS-B data (and those days),
    coverage counts, and the coverage-3 share per (airport, month)."""
    cov3 = t.assign(c3=(t.coverage == 3).astype(float)).groupby(["ap", "month"]).c3.mean()
    return {"n_rows": int(len(t)), "n_missing_day_rows": int((t.adsb_day_present == 0).sum()),
            "missing_days": sorted(t.loc[t.adsb_day_present == 0, "mvt_day"].unique().tolist()),
            "coverage_counts": {str(k): int(v) for k, v in t.coverage.value_counts().sort_index().items()},
            "cov3_share": {f"{ap}|{m}": float(v) for (ap, m), v in cov3.items()}}


def assemble(universe: pd.DataFrame, store_dir, version: str) -> pd.DataFrame:
    """The table for `universe` (training or serve), in the universe's order, columns TABLE_COLS. A 'missing' day ->
    every feature NaN, coverage 0, adsb_day_present 0. A day with no store entry is refused (it may be unbuilt)."""
    store = pathlib.Path(store_dir)
    _stamp_check(store, version, create=False)
    groups = universe.groupby("mvt_day").groups
    absent = [d for d in sorted(groups) if not (store / f"day_{d}.json").exists()]
    if absent:
        raise KeyError(f"{len(absent)} day(s) have no store entry (unbuilt or deferred): {absent[:10]}")
    parts = []
    for day in sorted(groups):
        fl = universe.loc[groups[day], ["MVT_ID_mvt"]]
        rec = json.loads((store / f"day_{day}.json").read_text())
        if rec.get("features_version") != version:
            raise RuntimeError(f"day_{day}.json: features_version differs from {version!r}")
        if rec.get("flights_sha") != flights_sha(fl.MVT_ID_mvt):
            raise RuntimeError(f"day_{day}: flight set differs from the one the store was built for")
        if rec["status"] == "ready":
            f = pd.read_parquet(store / f"day_{day}.parquet")
            f["adsb_day_present"] = 1
        elif rec["status"] == "missing":
            f = pd.DataFrame({"MVT_ID_mvt": fl.MVT_ID_mvt.to_numpy(float), "coverage": 0, "adsb_day_present": 0})
            for c in FEATURES:
                f[c] = np.nan
        else:
            raise RuntimeError(f"day_{day}.json has status {rec['status']!r}")
        parts.append(f)
    feats = pd.concat(parts, ignore_index=True).set_index("MVT_ID_mvt")
    ids = universe.MVT_ID_mvt.to_numpy(float)
    feats = feats.reindex(ids)
    t = universe[["month", "pos", "MVT_ID_mvt", "ap", "mvt_day"]].reset_index(drop=True).copy()
    t["adsb_day_present"] = feats.adsb_day_present.to_numpy().astype("int64")
    t["coverage"] = feats.coverage.to_numpy().astype("int64")
    for c in FEATURES:
        t[c] = feats[c].to_numpy().astype("float64")
    t["month"] = t.month.astype("int64")
    t["pos"] = t.pos.astype("int64")
    t["MVT_ID_mvt"] = t.MVT_ID_mvt.astype("float64")
    return t[TABLE_COLS]


# ---------------------------------------------------------------- universes

def _finish_universe(fl: pd.DataFrame) -> pd.DataFrame:
    fl["ap"] = fl.ADEP_mvt.astype(str)
    fl["mvt_day"] = pd.to_datetime(fl.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    return fl[UNIVERSE_COLS]


def _raw_dep(path) -> pd.DataFrame:
    raw = pd.read_parquet(path, columns=AF.FLIGHT_COLS + ["PHASE_mvt"])
    return raw[raw.PHASE_mvt == "DEP"].drop(columns="PHASE_mvt").drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")


def training_universe(stand_dir=STAND_DIR, order_dir=ORDER_DIR, raw_dir=RAW_DIR, months=None) -> pd.DataFrame:
    """Every row of the cached 2025 training months: (cache month, position in the file) -> MVT id (order cache, by
    position) -> the serve-time flight columns (raw file of the same name, DEP rows). Guard: MVT - AOBT_3 from raw
    equals the cache's proxy within 1 s on every row (the same guard as adsb_features.fold_flights)."""
    parts = []
    for sp in sorted(pathlib.Path(stand_dir).glob("training_2025-*.parquet")):
        st = pd.read_parquet(sp, columns=["month", "proxy"])
        m = st.month.unique()
        if len(m) != 1:
            raise ValueError(f"{sp.name}: more than one cache month {m}")
        if months is not None and int(m[0]) not in months:
            continue
        ids = pd.read_parquet(pathlib.Path(order_dir) / sp.name, columns=["MVT_ID_mvt"]).MVT_ID_mvt.to_numpy(float)
        if len(ids) != len(st):
            raise ValueError(f"{sp.name}: row count differs between the stand ({len(st)}) and order ({len(ids)}) caches")
        raw = _raw_dep(pathlib.Path(raw_dir) / sp.name)
        missing = np.setdiff1d(ids, raw.index.to_numpy(float))
        if len(missing):
            raise ValueError(f"{sp.name}: {len(missing)} ids absent from the raw file (e.g. {missing[:3]})")
        fl = raw.loc[ids].reset_index()
        gap = AF._secs(fl.MVT_TIME_UTC_mvt) - AF._secs(fl.AOBT_3_flt)
        bad = ~(np.abs(gap - st.proxy.to_numpy(float)) <= 1.0)
        if bad.any():
            raise ValueError(f"{sp.name}: alignment guard: {int(bad.sum())} rows where MVT - AOBT_3 != cache proxy")
        fl.insert(0, "pos", np.arange(len(fl), dtype="int64"))
        fl.insert(0, "month", int(m[0]))
        parts.append(fl)
    if not parts:
        raise FileNotFoundError(f"no training caches for months {months} under {stand_dir}")
    return _finish_universe(pd.concat(parts, ignore_index=True))


def serve_universe(rank_cache=RANK_CACHE, ranking=RANKING) -> pd.DataFrame:
    """The matched ranking rows in the ranking cache's order (the order lgbm_submit.load_frames appends them in), with
    the raw ranking file's serve-time flight columns."""
    rc = pd.read_parquet(rank_cache, columns=["MVT_ID_mvt", "month"])
    raw = _raw_dep(ranking)
    ids = rc.MVT_ID_mvt.to_numpy(float)
    missing = np.setdiff1d(ids, raw.index.to_numpy(float))
    if len(missing):
        raise ValueError(f"{len(missing)} ranking-cache ids absent from the raw ranking file (e.g. {missing[:3]})")
    fl = raw.loc[ids].reset_index()
    fl.insert(0, "pos", np.arange(len(fl), dtype="int64"))
    fl.insert(0, "month", rc.month.to_numpy().astype("int64"))
    return _finish_universe(fl)


def check_universes(train_u: pd.DataFrame, serve_u: pd.DataFrame) -> None:
    """Training and serve universes must share no MVT id and no MVT day (a shared day's store batch would be their
    union, and the tables are assembled per universe)."""
    ids = np.intersect1d(train_u.MVT_ID_mvt.to_numpy(float), serve_u.MVT_ID_mvt.to_numpy(float))
    if len(ids):
        raise ValueError(f"the training and serve universes share {len(ids)} MVT id(s), e.g. {ids[:3]}")
    days = sorted(set(train_u.mvt_day) & set(serve_u.mvt_day))
    if days:
        raise ValueError(f"the training and serve universes share {len(days)} MVT day(s): {days[:5]}")


# ---------------------------------------------------------------- CLI

def _resolve_with_retry(dirs=SOURCE_DIRS, tries: int = 3, wait_s: float = 2.0, sleep=time.sleep):
    """The fetcher rewrites MANIFEST.json in place (not atomically): a torn read is retried, then raised."""
    for i in range(tries):
        try:
            return resolve_sources(dirs)
        except ValueError as e:
            if "unreadable manifest" not in str(e) or i == tries - 1:
                raise
            sleep(wait_s)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("features", "tables"):
        p = sub.add_parser(name)
        p.add_argument("--store", required=True)
        p.add_argument("--days", default=None, help="comma-separated YYYY-MM-DD (default: every universe day)")
        p.add_argument("--rss-limit-gib", type=float, default=6.0)
        if name == "features":
            p.add_argument("--final", action="store_true",
                           help="record missing days / neighbours instead of deferring (after the fetch finished)")
            p.add_argument("--sources", default=None, type=lambda v: [x.strip() for x in v.split(",") if x.strip()],
                           help="comma-separated day-file directories instead of the default three (e.g. a union directory)")
        else:
            p.add_argument("--out", required=True)
    return ap.parse_args(argv)


def _universes(days) -> tuple:  # pragma: no cover - real caches; exercised by the smoke run and the real tests
    """(training universe, serve universe), each restricted to `days` when given, checked disjoint."""
    months = None
    if days is not None:                    # a flight's cache file may be the adjacent month's: load month +- 1
        months = sorted({m + k for d in days if d.startswith("2025") for m in [int(d[5:7])] for k in (-1, 0, 1)
                         if 1 <= m + k <= 12})
    want_tr = days is None or any(d.startswith("2025") for d in days)
    want_sv = days is None or any(d.startswith("2026") for d in days)
    tr = training_universe(months=months) if want_tr else pd.DataFrame(columns=UNIVERSE_COLS)
    sv = serve_universe() if want_sv else pd.DataFrame(columns=UNIVERSE_COLS)
    if days is not None:
        tr = tr[tr.mvt_day.isin(days)].reset_index(drop=True)
        sv = sv[sv.mvt_day.isin(days)].reset_index(drop=True)
    check_universes(tr, sv)
    return tr, sv


def main(argv=None) -> int:  # pragma: no cover - real-data entry point; exercised by the smoke runs
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_guards as G
    a = parse_args(argv)
    G.start_rss_guard(int(a.rss_limit_gib * 1024 ** 3))
    log = lambda m: print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)
    version = require_trc(AF.FEATURES_VERSION)
    days = None if a.days is None else [d.strip() for d in a.days.split(",") if d.strip()]
    t0 = time.time()
    fh = None
    target = a.store if a.cmd == "features" else a.out
    if not is_scratch(target):
        fh = G.acquire_lock()
        log("heavy-compute lock acquired (nonblocking)")
    try:
        return _main_body(a, days, version, t0, log, G)
    finally:
        if fh is not None:
            G.release_lock(fh)


def _main_body(a, days, version, t0, log, G) -> int:  # pragma: no cover - real-data body of main
    tr_u, sv_u = _universes(days)
    u = pd.concat([x for x in (tr_u, sv_u) if len(x)], ignore_index=True)
    log(f"universe {len(u):,} flights on {u.mvt_day.nunique()} days ({len(tr_u):,} training, {len(sv_u):,} serve); "
        f"stamp {stamp_of(version)}")
    if a.cmd == "features":
        sources, diag = _resolve_with_retry(tuple(pathlib.Path(x) for x in a.sources) if a.sources else SOURCE_DIRS)
        log(f"sources {diag['n_sources']} days; legacy accepted {diag['legacy_accepted']}; unverified "
            f"{diag['unverified_files']}; "
            f"manifest-without-file {diag['manifest_without_file']}")
        counts = build_store(u, sources, a.store, version=version, final=a.final, git_sha=AF._git_sha(), days=days,
                             log=log)
        log(f"COUNTS {json.dumps(counts)}  wall {time.time() - t0:.0f}s  peak RSS {G.peak_rss() / 1024 ** 3:.2f} GiB")
        return 0
    out = check_out_dir(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    rep = {"features_version": version, "stamp": stamp_of(version), "store": str(a.store), "days": days,
           "sources": store_sources(a.store, sorted(set(u.mvt_day)))}
    for name, part in (("train", tr_u), ("serve", sv_u)):
        if not len(part):
            continue
        t = assemble(part.reset_index(drop=True), a.store, version)
        t.to_parquet(out / f"{name}.parquet", index=False)
        rep[name] = table_report(t)
        log(f"{name}: {len(t):,} rows, {rep[name]['n_missing_day_rows']:,} on missing days -> {out / (name + '.parquet')}")
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    log(f"tables done wall {time.time() - t0:.0f}s peak RSS {G.peak_rss() / 1024 ** 3:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
