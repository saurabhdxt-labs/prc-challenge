"""Tests for scripts/analysis/r12_build.py — the R12 ADS-B feature store over any set of days and the train / serve tables.

Synthetic fixtures for every branch; the real-data parity tests run in a CHILD pytest process (a heavy load in pytest's
own process would raise the whole session's peak memory) and skip only when the real files are absent.
Mutation rehearsals are recorded in each test's docstring ("Fails when ...").
"""
from __future__ import annotations

import functools
import importlib.util
import json
import os
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest
from isolation import run_guarded

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHILD_ENV = "R12_BUILD_CHILD"


def _load():
    name = "r12_build"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "analysis" / "r12_build.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load()
AF = B.AF
V = "adsb_features TEST VERSION"


def isolated(fn):
    """Run a real-data test in a child pytest process so its memory never reaches this one."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get(CHILD_ENV) == "1":
            return fn(*args, **kwargs)
        node = f"{pathlib.Path(__file__).resolve()}::{fn.__name__}"
        r = run_guarded([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
                            node], env={**os.environ, CHILD_ENV: "1"}, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    return wrapper


# ---------------------------------------------------------------- fixtures

def _day_file(d: pathlib.Path, day: str, rows: int = 0) -> pathlib.Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"adsb_{day}.parquet"
    pd.DataFrame({"t": np.arange(rows, dtype=float)}).to_parquet(p)
    return p


def _manifest(d: pathlib.Path, entries) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "MANIFEST.json").write_text(json.dumps({"source": "test", "days": entries}))


def _ok(day):
    return {"day": day, "status": "ok", "rows": 1}


def _skipped_prev_ok(day):
    return {"day": day, "status": "skipped", "previous": {"day": day, "status": "ok"}}


def _flights(days_counts, month=3, start_id=1000):
    """A universe frame: `days_counts` = [(day, n)], MVT at noon + i minutes."""
    rows, mid, pos = [], start_id, 0
    for day, n in days_counts:
        for i in range(n):
            t = pd.Timestamp(f"{day}T12:00:00", tz="UTC") + pd.Timedelta(minutes=i)
            rows.append({"month": month, "pos": pos, "MVT_ID_mvt": float(mid), "ADEP_mvt": "EGLL" if i % 2 else "EHAM",
                         "CALLSIGN_flt": f"CS{mid}", "FLIGHT_mvt": f"F{mid}", "MVT_TIME_UTC_mvt": t,
                         "AOBT_3_flt": t - pd.Timedelta(minutes=15), "STAND_mvt": "A1"})
            mid += 1
            pos += 1
    u = pd.DataFrame(rows)
    u["ap"] = u.ADEP_mvt
    u["mvt_day"] = pd.to_datetime(u.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m-%d")
    return u[B.UNIVERSE_COLS]


class FakeBuilder:
    """Stands in for adsb_features.build_period: records the files it could see, returns joined=1 / coverage 3 on the
    first flight of each day and a feature value that encodes the MVT id (so a mis-join is visible)."""
    def __init__(self):
        self.calls = []

    def __call__(self, flights, adsb_dir, git_sha):
        self.calls.append({"n": len(flights), "files": sorted(os.listdir(adsb_dir)), "git_sha": git_sha,
                           "cols": list(flights.columns)})
        out = pd.DataFrame({"MVT_ID_mvt": flights.MVT_ID_mvt.to_numpy()})
        for c in AF.FEATURES:
            out[c] = flights.MVT_ID_mvt.to_numpy() * 0.5
        out["coverage"] = 2
        out.loc[0, "coverage"] = 3
        return out


@pytest.fixture
def src(tmp_path):
    """Three source dirs shaped like data/adsb/v2 (root), y2025_other and y2026."""
    a, b, c = tmp_path / "v2", tmp_path / "v2" / "y2025_other", tmp_path / "v2" / "y2026"
    for day in ("2025-01-30", "2025-01-31"):
        _day_file(a, day)
    _manifest(a, [_skipped_prev_ok("2025-01-30"), _ok("2025-01-31")])
    for day in ("2025-02-01", "2025-02-02"):
        _day_file(b, day)
    _manifest(b, [_ok("2025-02-01"), _ok("2025-02-02")])
    _day_file(c, "2026-01-01")
    _manifest(c, [_ok("2026-01-01")])
    return [a, b, c]


# ---------------------------------------------------------------- stamp

def test_stamp_is_twelve_hex_and_tracks_the_version():
    """Fails when stamp_of ignores its input (returns a constant) or truncates to another width."""
    s1, s2 = B.stamp_of("x"), B.stamp_of("y")
    assert len(s1) == 12 and all(ch in "0123456789abcdef" for ch in s1)
    assert s1 == B.stamp_of("x") and s1 != s2


def test_stamp_refuses_a_version_without_the_trc_fix():
    """The store must be built with the BC-25-fixed extractor. Fails when require_trc is dropped."""
    with pytest.raises(ValueError, match="TRC"):
        B.require_trc("adsb_features v4 (no fix)")
    assert B.require_trc(AF.FEATURES_VERSION) == AF.FEATURES_VERSION


# ---------------------------------------------------------------- sources

def test_manifest_days_reads_ok_and_skipped_with_previous_ok(tmp_path):
    """Fails when the skipped-with-previous-ok branch is dropped (every Jan/Jul 2025 day in data/adsb/v2 is 'skipped')."""
    _manifest(tmp_path, [_ok("2025-02-01"), _skipped_prev_ok("2025-02-02"), {"day": "2025-02-03", "status": "failed"},
                         {"day": "2025-02-04", "status": "skipped", "previous": {"status": "failed"}}])
    assert B.manifest_days(tmp_path) == {"2025-02-01", "2025-02-02"}


def test_manifest_days_missing_and_corrupt_are_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="MANIFEST.json"):
        B.manifest_days(tmp_path)
    (tmp_path / "MANIFEST.json").write_text("{not json")
    with pytest.raises(ValueError, match="unreadable manifest"):
        B.manifest_days(tmp_path)


def test_resolve_sources_unions_dirs_and_excludes_unverified(src):
    """A file on disk the manifest does not list (the fetch's in-flight day) is excluded and counted; a manifest day
    without its file is excluded and counted. Fails when resolve_sources trusts the directory listing alone."""
    _day_file(src[1], "2025-02-03")                       # on disk, not in the manifest
    m = json.loads((src[2] / "MANIFEST.json").read_text())
    m["days"].append(_ok("2026-01-02"))                   # in the manifest, no file
    (src[2] / "MANIFEST.json").write_text(json.dumps(m))
    sources, diag = B.resolve_sources(src)
    assert sorted(sources) == ["2025-01-30", "2025-01-31", "2025-02-01", "2025-02-02", "2026-01-01"]
    assert sources["2025-02-01"] == src[1] / "adsb_2025-02-01.parquet"
    assert diag["unverified_files"] == ["2025-02-03"]
    assert diag["manifest_without_file"] == ["2026-01-02"]


def test_resolve_sources_refuses_a_day_in_two_dirs(src):
    """Fails when a duplicated day silently takes the first (or last) directory's file."""
    _day_file(src[1], "2025-01-31")
    m = json.loads((src[1] / "MANIFEST.json").read_text())
    m["days"].append(_ok("2025-01-31"))
    (src[1] / "MANIFEST.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="two source directories"):
        B.resolve_sources(src)


def _full_day(d: pathlib.Path, day: str, hours: float = 24.0, n_ap: int = 10):
    d.mkdir(parents=True, exist_ok=True)
    t0 = (pd.Timestamp(day, tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()
    n = 50
    pd.DataFrame({"t": t0 + np.linspace(1.0, hours * 3600 - 1.0, n),
                  "airport": [f"A{i % n_ap}" for i in range(n)]}).to_parquet(d / f"adsb_{day}.parquet")


def test_legacy_unlisted_day_accepted_only_with_full_day_content(tmp_path):
    """2025-01-01 / 2026-01-01 predate the manifest regime (status skipped, previous null) but are complete files the
    shipped v4 tables read. They are accepted only if listed in `legacy` AND the file covers the whole UTC day at >= 9
    airports; a partial file stays unverified. Fails when the content check is dropped or the allowlist is ignored."""
    d = tmp_path / "v2"
    _full_day(d, "2025-01-01")
    _full_day(d, "2025-01-02", hours=12.0)
    _full_day(d, "2025-01-03")
    _manifest(d, [{"day": "2025-01-01", "status": "skipped", "previous": None}])
    legacy = {"2025-01-01": "why", "2025-01-02": "why"}
    sources, diag = B.resolve_sources([d], legacy=legacy)
    assert sorted(sources) == ["2025-01-01"]
    assert diag["legacy_accepted"] == ["2025-01-01"]
    assert diag["unverified_files"] == ["2025-01-02", "2025-01-03"]
    assert B.LEGACY_UNLISTED.keys() == {"2025-01-01", "2026-01-01"}


def test_legacy_full_day_check(tmp_path):
    _full_day(tmp_path, "2025-01-01")
    assert B.legacy_full_day(tmp_path / "adsb_2025-01-01.parquet", "2025-01-01") is True
    _full_day(tmp_path, "2025-01-02", n_ap=5)
    assert B.legacy_full_day(tmp_path / "adsb_2025-01-02.parquet", "2025-01-02") is False
    _full_day(tmp_path, "2025-01-03")
    assert B.legacy_full_day(tmp_path / "adsb_2025-01-03.parquet", "2025-01-04") is False     # another day's file


def test_source_universe_is_2025_all_year_plus_2026_janjul():
    u = B.source_universe()
    assert len(u) == 365 + 31 + 31
    assert "2025-06-15" in u and "2026-07-31" in u and "2026-02-01" not in u and "2024-12-31" not in u


def test_source_universe_default_equals_an_independently_built_calendar():
    """The default (months=None) must equal the old hard-coded set exactly, built here with the stdlib `datetime`
    module (not pandas, not prc.months) so this check does not share a bug with the implementation it is checking.
    Fails when source_universe's month-to-days expansion drops or adds a day (e.g. an off-by-one at a month
    boundary) relative to this independent calendar walk."""
    import datetime as _dt
    want = set()
    d = _dt.date(2025, 1, 1)
    while d <= _dt.date(2025, 12, 31):
        want.add(d.isoformat())
        d += _dt.timedelta(days=1)
    d = _dt.date(2026, 1, 1)
    while d <= _dt.date(2026, 1, 31):
        want.add(d.isoformat())
        d += _dt.timedelta(days=1)
    d = _dt.date(2026, 7, 1)
    while d <= _dt.date(2026, 7, 31):
        want.add(d.isoformat())
        d += _dt.timedelta(days=1)
    assert len(want) == 365 + 31 + 31
    assert B.source_universe() == want


def test_source_universe_months_argument_overrides_the_default():
    """Fails when source_universe ignores `months` and always returns the hard-coded default (2026-10-15 would be
    absent, and the day count would stay 427 instead of 31)."""
    u = B.source_universe(months=["2026-10"])
    assert u == {f"2026-10-{d:02d}" for d in range(1, 32)}
    assert "2026-10-15" not in B.source_universe()  # the untouched default is unaffected


def test_months_argument_lets_a_new_period_be_planned():
    """The brief's CLI scenario at the function level: a day outside today's default universe raises from plan_day,
    but once --months widens the universe (here simulated directly) the same day plans (status != refused). Fails
    when source_universe's `months` plumbing is removed (plan_day would then raise "outside the source universe"
    for 2026-10-15 even with a widened `months` argument)."""
    with pytest.raises(ValueError, match="2026-10-15.*outside the source universe"):
        B.plan_day("2026-10-15", {}, B.source_universe(), final=True)
    widened = B.source_universe(months=["2026-10"])
    p = B.plan_day("2026-10-15", {}, widened, final=True)
    assert p["status"] == "missing"  # no source files for it, but not refused as outside the universe


# ---------------------------------------------------------------- per-day plan

def test_plan_day_ready_when_day_and_neighbours_present(src):
    sources, _ = B.resolve_sources(src)
    p = B.plan_day("2025-01-31", sources, B.source_universe(), final=False)
    assert p["status"] == "ready" and p["missing_neighbours"] == []
    assert p["files"] == {"prev": "2025-01-30", "cur": "2025-01-31", "next": "2025-02-01"}


def test_plan_day_defers_on_a_missing_neighbour_inside_the_universe(src):
    """2025-02-02's next day (02-03) is still downloading: not final -> deferred (a later build would differ). Fails
    when the neighbour check is removed."""
    sources, _ = B.resolve_sources(src)
    p = B.plan_day("2025-02-02", sources, B.source_universe(), final=False)
    assert p["status"] == "deferred" and p["missing_neighbours"] == ["2025-02-03"]
    q = B.plan_day("2025-02-02", sources, B.source_universe(), final=True)
    assert q["status"] == "ready" and q["missing_neighbours"] == ["2025-02-03"]
    assert q["files"] == {"prev": "2025-02-01", "cur": "2025-02-02"}


def test_plan_day_neighbour_outside_the_universe_is_not_awaited(src):
    """2026-01-01's next day is absent but inside the universe (deferred); its previous day 2025-12-31 is inside too.
    A day whose neighbours lie outside (2026-01-31 -> 02-01) does not wait for them."""
    sources, _ = B.resolve_sources(src)
    sources = dict(sources, **{"2026-01-31": sources["2026-01-01"], "2026-01-30": sources["2026-01-01"]})
    p = B.plan_day("2026-01-31", sources, B.source_universe(), final=False)
    assert p["status"] == "ready" and p["missing_neighbours"] == []
    assert "next" not in p["files"]


def test_plan_day_missing_current_day(src):
    """Not final: deferred (the fetch may still bring it). Final: 'missing' — counted, never silently built."""
    sources, _ = B.resolve_sources(src)
    assert B.plan_day("2025-03-10", sources, B.source_universe(), final=False)["status"] == "deferred"
    assert B.plan_day("2025-03-10", sources, B.source_universe(), final=True)["status"] == "missing"


def test_plan_day_refuses_a_day_outside_the_universe(src):
    sources, _ = B.resolve_sources(src)
    with pytest.raises(ValueError, match="outside the source universe"):
        B.plan_day("2024-12-31", sources, B.source_universe(), final=True)


# ---------------------------------------------------------------- the store

def _store(tmp_path):
    return tmp_path / "data" / "r12" / "features_test"


def test_build_store_builds_resumes_and_counts(src, tmp_path):
    """First pass builds the ready days and defers the rest; the second pass skips what is done. Observable state:
    one parquet + one json per built day, STAMP.json, no link directory left behind."""
    sources, _ = B.resolve_sources(src)
    u = _flights([("2025-01-31", 3), ("2025-02-02", 2), ("2025-03-10", 2)])
    fb = FakeBuilder()
    store = _store(tmp_path)
    c1 = B.build_store(u, sources, store, version=V, final=False, git_sha="abc", builder=fb, root=tmp_path)
    assert c1 == {"built": 1, "done": 0, "deferred": 2, "missing": 0, "n_days": 3}
    assert fb.calls[0]["files"] == ["adsb_2025-01-30.parquet", "adsb_2025-01-31.parquet", "adsb_2025-02-01.parquet"]
    assert fb.calls[0]["n"] == 3 and fb.calls[0]["cols"] == AF.FLIGHT_COLS          # label columns never enter
    assert sorted(p.name for p in store.iterdir()) == ["STAMP.json", "day_2025-01-31.json", "day_2025-01-31.parquet"]
    c2 = B.build_store(u, sources, store, version=V, final=True, git_sha="abc", builder=fb, root=tmp_path)
    assert c2 == {"built": 1, "done": 1, "deferred": 0, "missing": 1, "n_days": 3}
    meta = json.loads((store / "day_2025-03-10.json").read_text())
    assert meta["status"] == "missing" and meta["n_flights"] == 2
    meta = json.loads((store / "day_2025-02-02.json").read_text())
    assert meta["missing_neighbours"] == ["2025-02-03"]
    assert len(fb.calls) == 2


def test_build_store_days_filter(src, tmp_path):
    """Fails when `days` is ignored (the smoke would build the whole universe)."""
    sources, _ = B.resolve_sources(src)
    u = _flights([("2025-01-31", 1), ("2025-02-01", 1)])
    fb = FakeBuilder()
    c = B.build_store(u, sources, _store(tmp_path), version=V, final=False, git_sha="x", builder=fb,
                      days=["2025-02-01"], root=tmp_path)
    assert c["n_days"] == 1 and c["built"] == 1 and fb.calls[0]["n"] == 1


def test_build_store_refuses_to_mix_stamps(src, tmp_path):
    """Fails when STAMP.json is not compared (a pre-TRC store could be resumed with post-TRC days)."""
    sources, _ = B.resolve_sources(src)
    u = _flights([("2025-01-31", 1)])
    B.build_store(u, sources, _store(tmp_path), version=V, final=False, git_sha="x", builder=FakeBuilder(),
                  root=tmp_path)
    with pytest.raises(RuntimeError, match="features_version"):
        B.build_store(u, sources, _store(tmp_path), version=V + " changed", final=False, git_sha="x",
                      builder=FakeBuilder(), root=tmp_path)


def test_build_store_refuses_to_mix_stamps_on_a_store_with_no_day_yet(src, tmp_path):
    """A store whose every day was deferred holds only STAMP.json: the store-level stamp alone must refuse another
    version (mutation rehearsal 2026-09-25 #10: the per-day check masked this). Fails when _stamp_check is skipped."""
    u = _flights([("2025-03-10", 1)])                        # not yet fetched -> deferred, nothing but STAMP.json
    sources, _ = B.resolve_sources(src)
    c = B.build_store(u, sources, _store(tmp_path), version=V, final=False, git_sha="x", builder=FakeBuilder(),
                      root=tmp_path)
    assert c["deferred"] == 1 and [p.name for p in _store(tmp_path).iterdir()] == ["STAMP.json"]
    with pytest.raises(RuntimeError, match="refusing to mix stamps"):
        B.build_store(u, sources, _store(tmp_path), version=V + " changed", final=False, git_sha="x",
                      builder=FakeBuilder(), root=tmp_path)


def test_build_store_refuses_a_changed_flight_set(src, tmp_path):
    """A done day whose flights changed is refused, never silently kept or overwritten."""
    sources, _ = B.resolve_sources(src)
    B.build_store(_flights([("2025-01-31", 2)]), sources, _store(tmp_path), version=V, final=False, git_sha="x",
                  builder=FakeBuilder(), root=tmp_path)
    with pytest.raises(RuntimeError, match="flight set"):
        B.build_store(_flights([("2025-01-31", 3)]), sources, _store(tmp_path), version=V, final=False, git_sha="x",
                      builder=FakeBuilder(), root=tmp_path)


def test_build_store_refuses_a_stale_day_when_a_neighbour_arrived(src, tmp_path):
    """Built final with 02-03 missing; 02-03 then lands: the day is stale (its lookahead differs). Fails when resume
    skips on file existence alone."""
    sources, _ = B.resolve_sources(src)
    u = _flights([("2025-02-02", 1)])
    B.build_store(u, sources, _store(tmp_path), version=V, final=True, git_sha="x", builder=FakeBuilder(),
                  root=tmp_path)
    _day_file(src[1], "2025-02-03")
    m = json.loads((src[1] / "MANIFEST.json").read_text())
    m["days"].append(_ok("2025-02-03"))
    (src[1] / "MANIFEST.json").write_text(json.dumps(m))
    sources, _ = B.resolve_sources(src)
    with pytest.raises(RuntimeError, match="stale"):
        B.build_store(u, sources, _store(tmp_path), version=V, final=True, git_sha="x", builder=FakeBuilder(),
                      root=tmp_path)


def test_build_store_refuses_builder_output_that_does_not_cover_the_day(src, tmp_path):
    """Fails when the builder's rows are not checked against the day's flights."""
    sources, _ = B.resolve_sources(src)

    def short(flights, adsb_dir, git_sha):
        return FakeBuilder()(flights.iloc[:-1], adsb_dir, git_sha)
    with pytest.raises(AssertionError, match="builder returned"):
        B.build_store(_flights([("2025-01-31", 2)]), sources, _store(tmp_path), version=V, final=False, git_sha="x",
                      builder=short, root=tmp_path)
    assert not (_store(tmp_path) / "day_2025-01-31.parquet").exists()


def test_build_store_refuses_output_outside_the_allowed_roots(src, tmp_path):
    sources, _ = B.resolve_sources(src)
    with pytest.raises(PermissionError, match="outside"):
        B.build_store(_flights([("2025-01-31", 1)]), sources, tmp_path / "elsewhere", version=V, final=False,
                      git_sha="x", builder=FakeBuilder(), root=tmp_path)


def test_output_dir_refuses_protected_dirs():
    """data/adsb/v2 (where the fetch writes) is never an output. Fails when the protected check is dropped."""
    with pytest.raises(PermissionError, match="protected"):
        B.check_out_dir(ROOT / "data" / "r12" / ".." / "adsb" / "v2" / "x")
    assert B.check_out_dir(ROOT / "data" / "r12" / "features_x") == (ROOT / "data" / "r12" / "features_x").resolve()


def test_is_scratch_decides_the_heavy_lock():
    """Real stores/tables (data/r12) take the heavy lock; smoke ones (logs/r12_scratch) do not. Fails when a real
    build could skip the lock."""
    assert B.is_scratch(ROOT / "logs" / "r12_scratch" / "s" / "store") is True
    assert B.is_scratch(ROOT / "data" / "r12" / "features_x") is False
    assert B.is_scratch(ROOT / "logs" / "r12_scratch" / ".." / ".." / "data" / "r12" / "x") is False


# ---------------------------------------------------------------- assemble (train and serve tables)

def _built(src, tmp_path, u, final=True):
    sources, _ = B.resolve_sources(src)
    store = _store(tmp_path)
    B.build_store(u, sources, store, version=V, final=final, git_sha="x", builder=FakeBuilder(), root=tmp_path)
    return store


def test_assemble_joins_by_id_in_universe_order(src, tmp_path):
    """Fails when features are joined positionally instead of by MVT id (the universe is shuffled here)."""
    u = _flights([("2025-01-31", 3), ("2025-02-01", 2)])
    store = _built(src, tmp_path, u)
    shuffled = u.sample(frac=1.0, random_state=0).reset_index(drop=True)
    t = B.assemble(shuffled, store, version=V)
    assert list(t.columns) == B.TABLE_COLS
    assert (t.MVT_ID_mvt.to_numpy() == shuffled.MVT_ID_mvt.to_numpy()).all()
    np.testing.assert_array_equal(t.dwell_s.to_numpy(), shuffled.MVT_ID_mvt.to_numpy() * 0.5)
    assert (t.adsb_day_present == 1).all()
    assert set(t.coverage) == {2, 3}


def test_assemble_missing_day_is_nan_plus_flag(src, tmp_path):
    """A day with no ADS-B file: every feature NaN, coverage 0, adsb_day_present 0. Fails when the missing branch
    fills zeros or drops the rows."""
    u = _flights([("2025-01-31", 1), ("2025-03-10", 2)])
    t = B.assemble(u, _built(src, tmp_path, u), version=V)
    miss = t.mvt_day == "2025-03-10"
    assert miss.sum() == 2
    assert t.loc[miss, AF.FEATURES].isna().all().all()
    assert (t.loc[miss, "coverage"] == 0).all() and (t.loc[miss, "adsb_day_present"] == 0).all()
    assert (t.loc[~miss, "adsb_day_present"] == 1).all()


def test_assemble_refuses_an_unbuilt_or_deferred_day(src, tmp_path):
    """Fails when a day with no store entry is treated as missing (it may just not be built yet)."""
    u = _flights([("2025-01-31", 1), ("2025-03-10", 1)])
    store = _built(src, tmp_path, u, final=False)          # 03-10 deferred: no entry
    with pytest.raises(KeyError, match="2025-03-10"):
        B.assemble(u, store, version=V)


def test_assemble_refuses_a_version_mismatch(src, tmp_path):
    u = _flights([("2025-01-31", 1)])
    store = _built(src, tmp_path, u)
    with pytest.raises(RuntimeError, match="features_version"):
        B.assemble(u, store, version=V + "x")


def test_assemble_refuses_a_flight_set_mismatch(src, tmp_path):
    """The table's day must hold exactly the flights the store built it for."""
    store = _built(src, tmp_path, _flights([("2025-01-31", 2)]))
    with pytest.raises(RuntimeError, match="flight set"):
        B.assemble(_flights([("2025-01-31", 3)]), store, version=V)


def test_serve_parity_same_function_same_columns(src, tmp_path):
    """The training table and the serve table come from ONE function; columns and dtypes identical."""
    tr = _flights([("2025-01-31", 3), ("2025-03-10", 1)], month=1)
    sv = _flights([("2026-01-01", 2)], month=1, start_id=9000)
    both = pd.concat([tr, sv], ignore_index=True)
    store = _built(src, tmp_path, both)
    a, b = B.assemble(tr, store, version=V), B.assemble(sv, store, version=V)
    assert list(a.columns) == list(b.columns) == B.TABLE_COLS
    assert a.dtypes.equals(b.dtypes)


# ---------------------------------------------------------------- universes

def _caches(tmp_path, months=(1, 2)):
    stand, order, raw = tmp_path / "stand", tmp_path / "order", tmp_path / "raw"
    for d in (stand, order, raw):
        d.mkdir()
    mid = 1
    for m in months:
        name = f"training_2025-{m:02d}-01_2025-{m + 1:02d}-01.parquet"
        n = 3
        ids = np.arange(mid, mid + n, dtype=float)
        mvt = [pd.Timestamp(f"2025-{m:02d}-05T10:00", tz="UTC") + pd.Timedelta(minutes=i) for i in range(n)]
        aobt = [t - pd.Timedelta(seconds=600) for t in mvt]
        pd.DataFrame({"month": m, "proxy": 600.0, "y": 500.0, "delta": 100.0}, index=range(n)).to_parquet(stand / name)
        pd.DataFrame({"MVT_ID_mvt": ids}).to_parquet(order / name)
        rawf = pd.DataFrame({"MVT_ID_mvt": np.r_[ids, ids[:1], [999.0 + m]], "ADEP_mvt": "EGLL",
                             "CALLSIGN_flt": "C", "FLIGHT_mvt": "F", "MVT_TIME_UTC_mvt": mvt + mvt[:1] + mvt[:1],
                             "AOBT_3_flt": aobt + aobt[:1] + aobt[:1], "STAND_mvt": "S",
                             "PHASE_mvt": ["DEP"] * n + ["DEP", "ARR"]})
        rawf.to_parquet(raw / name)
        mid += n
    return stand, order, raw


def test_training_universe_positions_ids_and_dedup(tmp_path):
    """Per cache file: pos 0..n-1, MVT id from the order cache, flight columns from the raw DEP rows (duplicates
    dropped, ARR ignored). Fails when pos restarts wrongly or ids are taken from the wrong file."""
    u = B.training_universe(*_caches(tmp_path))
    assert list(u.columns) == B.UNIVERSE_COLS
    assert u.month.tolist() == [1, 1, 1, 2, 2, 2] and u.pos.tolist() == [0, 1, 2, 0, 1, 2]
    assert u.MVT_ID_mvt.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert u.mvt_day.tolist() == ["2025-01-05"] * 3 + ["2025-02-05"] * 3


def test_training_universe_months_filter(tmp_path):
    u = B.training_universe(*_caches(tmp_path), months=(2,))
    assert u.month.unique().tolist() == [2] and u.pos.tolist() == [0, 1, 2]


def test_training_universe_proxy_guard(tmp_path):
    """MVT - AOBT_3 from raw must equal the cache proxy (row alignment). Fails when the guard is removed."""
    stand, order, raw = _caches(tmp_path, months=(1,))
    p = stand / "training_2025-01-01_2025-02-01.parquet"
    f = pd.read_parquet(p)
    f.loc[1, "proxy"] = 999.0
    f.to_parquet(p)
    with pytest.raises(ValueError, match="alignment guard: 1 rows"):
        B.training_universe(stand, order, raw)


def test_training_universe_row_count_mismatch(tmp_path):
    stand, order, raw = _caches(tmp_path, months=(1,))
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0]}).to_parquet(order / "training_2025-01-01_2025-02-01.parquet")
    with pytest.raises(ValueError, match="row count"):
        B.training_universe(stand, order, raw)


def test_training_universe_id_absent_from_raw(tmp_path):
    stand, order, raw = _caches(tmp_path, months=(1,))
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 77.0]}).to_parquet(order / "training_2025-01-01_2025-02-01.parquet")
    with pytest.raises(ValueError, match="absent from the raw"):
        B.training_universe(stand, order, raw)


def test_serve_universe_follows_the_ranking_cache_order(tmp_path):
    """pos = position in the ranking cache (the order load_frames appends ranking rows in); flight columns from the
    raw ranking file. Fails when the order is taken from the raw file instead."""
    ids = [30.0, 10.0, 20.0]
    t = [pd.Timestamp("2026-01-03T08:00", tz="UTC") + pd.Timedelta(minutes=i) for i in range(3)]
    pd.DataFrame({"MVT_ID_mvt": ids, "month": 1}).to_parquet(tmp_path / "rank_cache.parquet")
    pd.DataFrame({"MVT_ID_mvt": [10.0, 20.0, 30.0, 40.0], "ADEP_mvt": "LEMD", "CALLSIGN_flt": "C", "FLIGHT_mvt": "F",
                  "MVT_TIME_UTC_mvt": t + t[:1], "AOBT_3_flt": t + [pd.NaT], "STAND_mvt": "S",
                  "PHASE_mvt": ["DEP", "DEP", "DEP", "DEP"]}).to_parquet(tmp_path / "ranking.parquet")
    u = B.serve_universe(tmp_path / "rank_cache.parquet", tmp_path / "ranking.parquet")
    assert u.MVT_ID_mvt.tolist() == ids and u.pos.tolist() == [0, 1, 2] and (u.month == 1).all()
    assert list(u.columns) == B.UNIVERSE_COLS


def test_serve_universe_refuses_ids_missing_from_raw(tmp_path):
    pd.DataFrame({"MVT_ID_mvt": [5.0], "month": 1}).to_parquet(tmp_path / "rank_cache.parquet")
    pd.DataFrame({"MVT_ID_mvt": [6.0], "ADEP_mvt": "LEMD", "CALLSIGN_flt": "C", "FLIGHT_mvt": "F",
                  "MVT_TIME_UTC_mvt": [pd.Timestamp("2026-01-03", tz="UTC")], "AOBT_3_flt": [pd.Timestamp("2026-01-03", tz="UTC")],
                  "STAND_mvt": "S", "PHASE_mvt": "DEP"}).to_parquet(tmp_path / "ranking.parquet")
    with pytest.raises(ValueError, match="absent from the raw"):
        B.serve_universe(tmp_path / "rank_cache.parquet", tmp_path / "ranking.parquet")


def test_check_universes_refuses_shared_ids_or_days():
    """Train and serve tables come from their own universes, never from a day-prefix split; a shared MVT day would
    make the store's batch their union (assemble would then refuse the train subset). Fails when either check goes."""
    tr = _flights([("2025-12-31", 2)], month=12)
    sv = _flights([("2026-01-01", 2)], month=1, start_id=9000)
    B.check_universes(tr, sv)
    with pytest.raises(ValueError, match="share 1 MVT day"):
        B.check_universes(tr, _flights([("2025-12-31", 1)], month=1, start_id=9000))
    with pytest.raises(ValueError, match="share 2 MVT id"):
        B.check_universes(tr, _flights([("2026-01-01", 2)], month=1, start_id=1000))


# ---------------------------------------------------------------- real data (child process)

REAL = (ROOT / "data" / "adsb" / "v2" / "y2026" / "adsb_2026-01-15.parquet").exists()


@pytest.mark.slow
@pytest.mark.skipif(not REAL, reason="real ADS-B day files absent")
@isolated
def test_real_store_day_equals_shipped_build_period(tmp_path):
    """The store's day build is adsb_features.build_period on that day's flights with exactly its prev/cur/next files:
    compare against build_period called directly on the y2026 directory for 2026-01-15 (all neighbours in y2026).
    Fails when the link directory omits a neighbour or the day's flight batch is cut (the LOO stand centroids depend on
    the batch)."""
    u = B.serve_universe(B.RANK_CACHE, B.RANKING)
    day = u[u.mvt_day == "2026-01-15"].reset_index(drop=True)
    sources, _ = B.resolve_sources(B.SOURCE_DIRS)
    store = tmp_path / "logs" / "r12_scratch" / "store"
    B.build_store(day, sources, store, version=AF.FEATURES_VERSION, final=False, git_sha="test",
                  builder=B.default_builder, days=["2026-01-15"], root=tmp_path)
    got = B.assemble(day, store, version=AF.FEATURES_VERSION)
    ref = AF.build_period(day[AF.FLIGHT_COLS], B.SOURCE_DIRS[2], git_sha="test")
    pd.testing.assert_frame_equal(got[["MVT_ID_mvt"] + AF.FEATURES + ["coverage"]].reset_index(drop=True),
                                  ref[["MVT_ID_mvt"] + AF.FEATURES + ["coverage"]].reset_index(drop=True),
                                  check_dtype=False)
    assert (got.adsb_day_present == 1).all()


@pytest.mark.slow
@pytest.mark.skipif(not (ROOT / "data" / "cache_order").exists(), reason="real caches absent")
@isolated
def test_real_training_universe_matches_the_fold_row_count():
    """All 12 cached months: 2,062,440 rows, unique ids, proxy guard passes on every row, every day in 2025 (or the
    first minutes of 2026 for late-Dec MVTs)."""
    u = B.training_universe(B.STAND_DIR, B.ORDER_DIR, B.RAW_DIR)
    assert len(u) == 2_062_440 and u.MVT_ID_mvt.is_unique
    assert u.groupby("month").size().to_dict()[1] == 152_250
    assert set(d[:4] for d in u.mvt_day.unique()) <= {"2025", "2026"}


def test_table_and_store_contract_carry_the_dwg_visibility_and_provenance_columns():
    """DWG (BC-26, 2026-09-27): the R12 model conditions on stand visibility and on the stand reference's provenance, so
    both columns must reach the store day files and TABLE_COLS (derived from adsb_features.FEATURES), and the stamp must
    be the DWG one. Kills: a hand-written column list that omits them; a store stamped with the pre-DWG version."""
    for c in ("stand_visible", "stand_ref_parked_share"):
        assert c in B.FEATURES and c in B.TABLE_COLS
    assert "DWG" in AF.FEATURES_VERSION and B.stamp_of(AF.FEATURES_VERSION) != "801f9bcac9a5"


# ---------------------------------------------------------------- declared sources (plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md)

def test_source_contains_reads_the_manifest_declaration_and_defaults_to_adsb_lol(tmp_path):
    """A source directory's MANIFEST.json may declare what it contains; a manifest without the key is adsb.lol (every
    directory before AFI). Mutation: default to [] -> red; ignore the key -> red."""
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        _day_file(d, "2026-07-01")
        _manifest(d, [_ok("2026-07-01")])
    man = json.loads((b / "MANIFEST.json").read_text())
    man["contains"] = ["adsb.lol", "adsb.fi"]
    (b / "MANIFEST.json").write_text(json.dumps(man))
    assert B.source_contains(a / "adsb_2026-07-01.parquet") == ["adsb.lol"]
    assert B.source_contains(b / "adsb_2026-07-01.parquet") == ["adsb.fi", "adsb.lol"]


def test_build_store_records_what_each_day_was_built_from(src, tmp_path):
    """Day 2025-01-31 reads its own file and 01-30 from the root dir (declared adsb.lol + adsb.fi) and 02-01 from
    y2025_other (undeclared = adsb.lol): the record holds the union. Mutation: drop source_contains -> red."""
    m = src[0] / "MANIFEST.json"
    man = json.loads(m.read_text())
    man["contains"] = ["adsb.lol", "adsb.fi"]
    m.write_text(json.dumps(man))
    sources, _ = B.resolve_sources(src)
    store = _store(tmp_path)
    B.build_store(_flights([("2025-01-31", 3)]), sources, store, version=V, final=True, git_sha="abc",
                  builder=FakeBuilder(), root=tmp_path)
    assert json.loads((store / "day_2025-01-31.json").read_text())["source_contains"] == ["adsb.fi", "adsb.lol"]


def test_store_sources_is_the_union_over_the_tables_days(tmp_path):
    """A day record without the key is adsb.lol (stores built before AFI); a day without a record is refused.
    Mutation: read only the first day -> red; skip missing records -> red."""
    d = tmp_path / "store"
    d.mkdir()
    (d / "day_2026-01-05.json").write_text(json.dumps({"day": "2026-01-05", "status": "ready"}))
    (d / "day_2026-07-05.json").write_text(json.dumps({"day": "2026-07-05", "status": "ready",
                                                       "source_contains": ["adsb.fi", "adsb.lol"]}))
    assert B.store_sources(d, ["2026-01-05"]) == ["adsb.lol"]
    assert B.store_sources(d, ["2026-01-05", "2026-07-05"]) == ["adsb.fi", "adsb.lol"]
    with pytest.raises(FileNotFoundError, match="2026-07-06"):
        B.store_sources(d, ["2026-07-06"])


def test_features_takes_the_source_directories_on_the_command_line():
    """Mutation: drop --sources -> red."""
    assert B.parse_args(["features", "--store", "x", "--sources", "d1,d2"]).sources == ["d1", "d2"]
    assert B.parse_args(["features", "--store", "x"]).sources is None


def test_features_months_defaults_to_the_prc_months_default_and_accepts_an_override():
    """Fails when --months is dropped from the features subparser (AttributeError / SystemExit) or its default is
    not prc.months.DEFAULT_MONTHS (today's universe would silently change)."""
    import prc.months as MO
    assert B.parse_args(["features", "--store", "x"]).months == MO.DEFAULT_MONTHS
    assert B.parse_args(["features", "--store", "x", "--months", "2026-10"]).months == "2026-10"


class ReadingBuilder(FakeBuilder):
    """As adsb_features.build_period does, opens every day file it is shown; a link it cannot follow is an error."""
    def __call__(self, flights, adsb_dir, git_sha):
        for name in sorted(os.listdir(adsb_dir)):
            pd.read_parquet(pathlib.Path(adsb_dir) / name)
        return super().__call__(flights, adsb_dir, git_sha)


def test_regression_bug_40_relative_source_directories_reach_the_builder_as_readable_files(tmp_path, monkeypatch):
    """BC-40 (2026-10-03): `features --sources data/adsb/union/y2026_r12` gave relative source paths; the store's per-day
    link directory symlinked them as relative targets, which resolve from the link's own directory — every link broken,
    and the real builder silently scored all 6,037 flights of each day as 'no ADS-B'. A source given relative to the
    working directory must reach the builder as a readable file. Mutation: symlink the path as given -> red."""
    monkeypatch.chdir(tmp_path)
    for day in ("2025-03-09", "2025-03-10", "2025-03-11"):
        _day_file(tmp_path / "rel_src", day, rows=3)
    _manifest(tmp_path / "rel_src", [_ok("2025-03-09"), _ok("2025-03-10"), _ok("2025-03-11")])
    sources = {d: pathlib.Path("rel_src") / f"adsb_{d}.parquet" for d in ("2025-03-09", "2025-03-10", "2025-03-11")}
    fb = ReadingBuilder()
    c = B.build_store(_flights([("2025-03-10", 2)]), sources, _store(tmp_path), version=V, final=True, git_sha="abc",
                      builder=fb, root=tmp_path)
    assert c["built"] == 1 and fb.calls[0]["files"] == ["adsb_2025-03-09.parquet", "adsb_2025-03-10.parquet",
                                                        "adsb_2025-03-11.parquet"]


def test_resolve_sources_returns_absolute_paths_for_a_relative_directory(tmp_path, monkeypatch):
    """BC-40 origin: the paths leave resolve_sources absolute, whatever the caller passed. Mutation: return them as
    given -> red."""
    monkeypatch.chdir(tmp_path)
    _day_file(tmp_path / "rel", "2025-03-10")
    _manifest(tmp_path / "rel", [_ok("2025-03-10")])
    sources, _ = B.resolve_sources([pathlib.Path("rel")])
    assert sources["2025-03-10"].is_absolute() and sources["2025-03-10"] == (tmp_path / "rel" / "adsb_2025-03-10.parquet").resolve()


def test_regression_bug_40_a_source_the_builder_cannot_read_is_refused_not_scored_as_no_adsb(tmp_path):
    """The link step refuses a day file it cannot read instead of handing the builder an empty day, and leaves no link
    directory behind (build_store already refuses a file that vanished, at its stat; this pins the link step itself).
    Mutation: drop the readability check -> the builder runs on a broken link -> red."""
    src = tmp_path / "src"
    _day_file(src, "2025-03-10", rows=3)
    store = tmp_path / "store"
    store.mkdir()
    plan = {"files": {"cur": "2025-03-10", "next": "2025-03-11"}}
    sources = {"2025-03-10": src / "adsb_2025-03-10.parquet", "2025-03-11": src / "adsb_2025-03-11.parquet"}
    fb = FakeBuilder()
    with pytest.raises(RuntimeError, match="cannot read .*adsb_2025-03-11.parquet"):
        B._build_one("2025-03-10", _flights([("2025-03-10", 2)]), plan, sources, store, fb, "abc")
    assert fb.calls == [] and list(store.iterdir()) == []


def test_table_report_describes_the_table_it_is_given():
    """End-of-session review 2026-10-03 (finding 4): the per-table statistics of report.json come from one tested function
    (they sat inline under the main body's pragma, so the AFB fallback copied another table's numbers instead). Seen red
    2026-10-03 08:25-08:42 EDT (no function). Rows: missing-day rows counted and their days listed once each, coverage counts
    keyed by the code as a string, the coverage-3 share per (airport, month)."""
    t = pd.DataFrame({"ap": ["EDDF", "EDDF", "EDDF", "LIRF", "LIRF"], "month": [7, 7, 7, 7, 1],
                      "mvt_day": ["2026-07-01", "2026-07-01", "2026-07-02", "2026-07-02", "2026-01-05"],
                      "adsb_day_present": [0, 0, 1, 1, 1], "coverage": [0, 0, 3, 1, 3]})
    r = B.table_report(t)
    assert r == {"n_rows": 5, "n_missing_day_rows": 2, "missing_days": ["2026-07-01"],
                 "coverage_counts": {"0": 2, "1": 1, "3": 2},
                 "cov3_share": {"EDDF|7": pytest.approx(1 / 3), "LIRF|1": 1.0, "LIRF|7": 0.0}}
    assert list(r["cov3_share"]) == ["EDDF|7", "LIRF|1", "LIRF|7"]


def test_regression_bug_41_a_done_day_built_from_other_sources_is_stale(src, tmp_path):
    """BC-41 sibling (r12_build's resume check): a day built from adsb.lol, then requested from a directory that declares
    adsb.fi with the same day files, was counted "done" — the record's sources were never compared, so a rebuild from a
    union directory kept the adsb.lol features silently. Seen red 2026-10-03 08:25-08:42 EDT."""
    sources, _ = B.resolve_sources(src)
    u = _flights([("2025-01-31", 2)])
    B.build_store(u, sources, _store(tmp_path), version=V, final=True, git_sha="x", builder=FakeBuilder(), root=tmp_path)
    for d in src:
        m = json.loads((d / "MANIFEST.json").read_text())
        m["contains"] = ["adsb.fi", "adsb.lol"]
        (d / "MANIFEST.json").write_text(json.dumps(m))
    sources, _ = B.resolve_sources(src)
    with pytest.raises(RuntimeError, match="stale.*adsb.fi"):
        B.build_store(u, sources, _store(tmp_path), version=V, final=True, git_sha="x", builder=FakeBuilder(),
                      root=tmp_path)


def test_a_done_day_recorded_before_source_contains_existed_is_still_done(src, tmp_path):
    """BC-41 (b): records written before the key existed hold no `source_contains`; a record with files was built from
    adsb.lol (every store before AFI) and one without files has no sources, so resuming over such a store must count
    both as done, not stale (the protecting invariant of the default)."""
    sources, _ = B.resolve_sources(src)
    u = _flights([("2025-01-31", 2), ("2025-02-03", 1)])              # 02-03 has no source file: a "missing" record
    store = _store(tmp_path)
    first = B.build_store(u, sources, store, version=V, final=True, git_sha="x", builder=FakeBuilder(), root=tmp_path)
    assert first["built"] == 1 and first["missing"] == 1
    for jp in store.glob("day_*.json"):
        rec = json.loads(jp.read_text())
        del rec["source_contains"]
        jp.write_text(json.dumps(rec))
    counts = B.build_store(u, sources, store, version=V, final=True, git_sha="x", builder=FakeBuilder(), root=tmp_path)
    assert counts["done"] == 2 and counts["built"] == 0


def _fake_universe(days, mvt_id_start):
    rows = [{c: None for c in B.UNIVERSE_COLS} for _ in days]
    for i, (r, d) in enumerate(zip(rows, days)):
        r.update({"MVT_ID_mvt": float(mvt_id_start + i), "mvt_day": d, "month": int(d[5:7])})
    return pd.DataFrame(rows, columns=B.UNIVERSE_COLS)


def test_regression_a_period_after_2026_is_served_not_dropped(monkeypatch):
    """Regression (final-product audit 2026-10-06): _universes took training days as "2025*" and serve days as
    "2026*", so the days of a 2027 period matched neither universe and the tables came out EMPTY without an error.
    A day outside the training year is a serve day. Mutation: restoring `d.startswith("2026")` for the serve
    universe turns this red."""
    monkeypatch.setattr(B, "training_universe", lambda months=None: _fake_universe(["2025-03-10"], 1))
    monkeypatch.setattr(B, "serve_universe", lambda: _fake_universe(["2027-02-03", "2026-01-15"], 100))
    tr, sv = B._universes(["2027-02-03"])
    assert len(tr) == 0
    assert list(sv.mvt_day) == ["2027-02-03"]


def test_universes_split_today_unchanged(monkeypatch):
    """Invariant: a 2025 day goes to training and a 2026 day to serve, exactly as before the change."""
    monkeypatch.setattr(B, "training_universe", lambda months=None: _fake_universe(["2025-03-10"], 1))
    monkeypatch.setattr(B, "serve_universe", lambda: _fake_universe(["2026-01-15"], 100))
    tr, sv = B._universes(["2025-03-10", "2026-01-15"])
    assert list(tr.mvt_day) == ["2025-03-10"] and list(sv.mvt_day) == ["2026-01-15"]
    tr, sv = B._universes(["2025-03-10"])
    assert list(tr.mvt_day) == ["2025-03-10"] and len(sv) == 0


def test_regression_months_restrict_the_days_built_when_no_days_are_given(src, tmp_path):
    """Regression (final-product audit 2026-10-06): `features --months 2026-10` without `--days` walked EVERY day of
    the flight universe (training 2025 + the scored file) and plan_day refused the first 2025 day as "outside the
    source universe". Without `days`, only the universe days inside `universe_days` are built. Mutation: walking
    sorted(groups) again turns this red."""
    sources, _ = B.resolve_sources(src)
    u = _flights([("2025-01-31", 1), ("2025-02-01", 1)])
    fb = FakeBuilder()
    c = B.build_store(u, sources, _store(tmp_path), version=V, final=True, git_sha="x", builder=fb, root=tmp_path,
                      universe_days={"2025-02-01"})
    assert c["n_days"] == 1 and len(fb.calls) == 1 and fb.calls[0]["n"] == 1


def test_build_store_without_months_still_walks_every_universe_day(src, tmp_path):
    """Invariant: with the default universe_days (2025 + Jan/Jul 2026) every day of a 2025 flight universe is walked,
    as before the change."""
    sources, _ = B.resolve_sources(src)
    u = _flights([("2025-01-31", 1), ("2025-02-01", 1)])
    c = B.build_store(u, sources, _store(tmp_path), version=V, final=True, git_sha="x", builder=FakeBuilder(),
                      root=tmp_path)
    assert c["n_days"] == 2


# ---------------------------------------------------------------- item 4a: --months wiring fails fast (FP-W3)

def test_build_store_logs_how_many_universe_days_it_skipped(src, tmp_path):
    """Review H1 note (FP-W3 item 5): when `days` is None and `universe_days` excludes some of the flight
    universe's days, build_store silently walked fewer days with no line saying so. Now logs one line naming the
    count and the requested months. Mutation: drop the `if skipped:` log call -> the first assertion below finds
    no matching line -> red."""
    sources, _ = B.resolve_sources(src)
    u = _flights([("2025-01-31", 1), ("2025-02-01", 1)])
    logged = []
    c = B.build_store(u, sources, _store(tmp_path), version=V, final=True, git_sha="x", builder=FakeBuilder(),
                      root=tmp_path, universe_days={"2025-02-01"}, log=logged.append)
    assert c["n_days"] == 1
    assert any("skipped 1" in m and "2025-01" in m for m in logged), logged


def test_build_store_does_not_log_a_skip_line_when_nothing_was_skipped(src, tmp_path):
    """Negative case for item 5: the default universe_days (every day of the flight universe) skips nothing, so no
    skip line is logged -- the new line must not fire on every build. Mutation: log the skip line unconditionally
    -> this assertion goes red."""
    sources, _ = B.resolve_sources(src)
    u = _flights([("2025-01-31", 1), ("2025-02-01", 1)])
    logged = []
    B.build_store(u, sources, _store(tmp_path), version=V, final=True, git_sha="x", builder=FakeBuilder(),
                  root=tmp_path, log=logged.append)
    assert not any("skipped" in m for m in logged), logged


def test_months_universe_days_parses_the_spec_then_builds_the_day_set():
    """The extracted pure function (item 4a): a valid spec builds the day set; a malformed one raises naming the
    bad piece (prc.months.parse_months) without any day-set work. Mutation: drop the parse_months call (pass the
    spec straight to source_universe) -> TypeError / wrong result -> red."""
    assert B.months_universe_days("2026-10") == {f"2026-10-{d:02d}" for d in range(1, 32)}
    with pytest.raises(ValueError, match="malformed month"):
        B.months_universe_days("not-a-month")


def test_main_body_wires_months_into_build_store_as_universe_days(monkeypatch):
    """Item 4a (finding 5): --months must reach build_store as exactly that month's universe_days. Mutation: wire
    universe_days from the default months instead of a.months -> the captured set would be today's default
    (427 days), not October's 31 -> red."""
    calls = []

    def fake_universes(days):
        calls.append("universes")
        return _flights([("2026-10-15", 1)]), pd.DataFrame(columns=B.UNIVERSE_COLS)

    def fake_resolve(dirs):
        return {}, {"n_sources": 0, "legacy_accepted": 0, "unverified_files": 0, "manifest_without_file": 0}
    captured = {}

    def fake_build_store(u, sources, store, **kw):
        captured.update(kw)
        return {"built": 0, "done": 0, "deferred": 0, "missing": 0, "n_days": 0}
    monkeypatch.setattr(B, "_universes", fake_universes)
    monkeypatch.setattr(B, "_resolve_with_retry", fake_resolve)
    monkeypatch.setattr(B, "build_store", fake_build_store)

    class FakeG:
        @staticmethod
        def peak_rss():
            return 0
    a = types.SimpleNamespace(cmd="features", months="2026-10", sources=None, store="store_x", final=False)
    rc = B._main_body(a, None, "verX", 0.0, lambda m: None, FakeG)
    assert rc == 0
    assert captured["universe_days"] == {f"2026-10-{d:02d}" for d in range(1, 32)}
    assert calls == ["universes"]


def test_main_body_refuses_a_malformed_months_spec_before_universes_load(monkeypatch):
    """Item 4a (finding 5): a malformed --months must fail in milliseconds, before the (real-cache) training/serve
    universes load -- not deep inside the features branch, after _universes(days) already ran. Mutation: move the
    months_universe_days call back after `tr_u, sv_u = _universes(days)` -> `calls` would be ["universes"] instead
    of [] when the ValueError fires -> red."""
    calls = []

    def fake_universes(days):
        calls.append("universes")
        return _flights([("2026-10-15", 1)]), pd.DataFrame(columns=B.UNIVERSE_COLS)
    monkeypatch.setattr(B, "_universes", fake_universes)
    a = types.SimpleNamespace(cmd="features", months="not-a-month", sources=None, store="store_x", final=False)
    with pytest.raises(ValueError, match="malformed month"):
        B._main_body(a, None, "verX", 0.0, lambda m: None, None)
    assert calls == []
