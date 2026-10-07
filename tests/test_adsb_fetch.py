"""Tests for scripts/adsb_fetch.py — the v2 ADS-B day extractor and the resumable multi-day fetch loop.

Written before the module (red first). Each test names the mutation it must kill.
"""
from __future__ import annotations

import gzip
import importlib.util
import io
import json
import pathlib
import sys
import tarfile

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("adsb_fetch", ROOT / "scripts" / "adsb_fetch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adsb_fetch"] = mod
    spec.loader.exec_module(mod)
    return mod


af = _load()

EHAM = (52.3086, 4.7639)
FAR = (55.0, 10.0)          # outside every airport box


def _trace(points, icao="484abc", base=1_736_400_000.0, gz=False):
    d = {"icao": icao, "r": "PH-ABC", "t": "B738", "timestamp": base, "trace": points}
    raw = json.dumps(d).encode()
    return gzip.compress(raw) if gz else raw


def _pt(dt, latlon, alt, gs=5.0, track=90.0, flags=0, ac=None, src="adsb_icao"):
    # readsb trace point: [dt, lat, lon, alt, gs, track, flags, vrate, aircraft, type_source, ...]
    return [dt, latlon[0], latlon[1], alt, gs, track, flags, 0, ac, src]


# ---------------------------------------------------------------- parse_member_v2

def test_callsign_set_outside_the_box_carries_into_box_points():
    """Kills: reading the callsign only on in-box points (the v1 parser's defect, review R3-8)."""
    raw = _trace([
        _pt(0, FAR, 3000, ac={"flight": "KLM123  "}),    # callsign arrives out of the box
        _pt(10, EHAM, "ground"),
        _pt(20, EHAM, "ground"),
    ])
    rows = af.parse_member_v2(raw)
    df = pd.DataFrame(rows, columns=af.COLUMNS)
    assert len(df) == 2
    assert df.callsign.tolist() == ["KLM123", "KLM123"]


def test_null_altitude_is_not_on_ground_and_is_flagged():
    """Kills: `alt is None` counted as on-ground (review R3-8); kills dropping the alt_missing flag."""
    raw = _trace([_pt(0, EHAM, None), _pt(1, EHAM, "ground"), _pt(2, EHAM, 1200)])
    df = pd.DataFrame(af.parse_member_v2(raw), columns=af.COLUMNS)
    assert df.on_ground.tolist() == [0, 1, 0]
    assert df.alt_missing.tolist() == [1, 0, 0]
    assert df.alt.iloc[2] == 1200.0 and pd.isna(df.alt.iloc[0]) and pd.isna(df.alt.iloc[1])


def test_time_track_source_and_airport_are_recorded():
    """Kills: t not offset by the member timestamp; track or source not stored; wrong airport box."""
    raw = _trace([_pt(7.5, EHAM, "ground", gs=12.0, track=271.0, src="mlat")], base=1000.0)
    df = pd.DataFrame(af.parse_member_v2(raw), columns=af.COLUMNS)
    r = df.iloc[0]
    assert r.t == 1007.5 and r.airport == "EHAM" and r.gs == 12.0 and r.track == 271.0 and r.src == "mlat"
    assert r.icao == "484abc"


def test_gzipped_member_parses_like_plain():
    """Kills: dropping the gzip branch."""
    pts = [_pt(0, EHAM, "ground")]
    assert af.parse_member_v2(_trace(pts, gz=True)) == af.parse_member_v2(_trace(pts))


def test_malformed_members_yield_no_rows():
    """Kills: raising on bad JSON / missing trace instead of skipping (one bad member must not kill a day)."""
    assert af.parse_member_v2(b"not json") == []
    assert af.parse_member_v2(json.dumps({"icao": "x", "timestamp": 1.0}).encode()) == []
    assert af.parse_member_v2(_trace([[0, None, None, 0, 0]])) == []


def test_out_of_box_points_are_dropped():
    """Kills: keeping every point (the output would be the whole globe)."""
    raw = _trace([_pt(0, FAR, 3000), _pt(1, EHAM, "ground")])
    assert len(af.parse_member_v2(raw)) == 1


# ---------------------------------------------------------------- day list

def test_day_list_interleaves_january_and_july_and_puts_spent_day_last():
    """Kills: month-blocked order (a stop mid-run would leave one month empty); kills omitting 62 days;
    kills putting the already-seen 2025-01-09 early."""
    days = af.day_list()
    assert len(days) == 62 and len(set(days)) == 62
    assert days[0].startswith("2025-01") and days[1].startswith("2025-07")
    assert days[-1] == "2025-01-09"
    first20 = days[:20]
    assert sum(d.startswith("2025-01") for d in first20) == 10


# ---------------------------------------------------------------- download_day (asset list from the release index)

def _index(tag, assets):
    return {tag: [{"name": n, "size": s, "url": f"https://x/{n}"} for n, s in assets]}


def _runner(sizes):
    """Fake curl: writes `sizes[name]` bytes to the -o path and returns rc 0 (or rc 22 when size is None)."""
    calls = []

    class R:
        def __init__(self, rc):
            self.returncode = rc

    def run(cmd, **kw):
        dst = pathlib.Path(cmd[cmd.index("-o") + 1])
        name = cmd[-1].rsplit("/", 1)[1]
        calls.append(name)
        if sizes.get(name) is None:
            return R(22)
        dst.write_bytes(b"x" * sizes[name])
        return R(0)
    run.calls = calls
    return run


def test_download_takes_a_single_tar_day_from_the_index(tmp_path):
    """Kills: guessing '.tar.aa' names (5 of the 62 days are published as one '.tar' — the smoke's 404)."""
    tag = "v2025.01.01-planes-readsb-prod-0"
    idx = _index(tag, [(f"{tag}.tar", 7)])
    run = _runner({f"{tag}.tar": 7})
    paths = af.download_day("2025-01-01", index=idx, raw_dir=tmp_path, runner=run)
    assert [p.name for p in paths] == [f"{tag}.tar"] and run.calls == [f"{tag}.tar"]


def test_download_takes_both_parts_in_name_order(tmp_path):
    """Kills: dropping part 'ab' or concatenating parts out of order (a corrupt tar stream)."""
    tag = "v2025.07.01-planes-readsb-prod-0"
    idx = _index(tag, [(f"{tag}.tar.ab", 3), (f"{tag}.tar.aa", 5)])
    run = _runner({f"{tag}.tar.aa": 5, f"{tag}.tar.ab": 3})
    paths = af.download_day("2025-07-01", index=idx, raw_dir=tmp_path, runner=run)
    assert [p.name for p in paths] == [f"{tag}.tar.aa", f"{tag}.tar.ab"]


def test_download_size_mismatch_raises_and_removes_every_part(tmp_path):
    """Kills: accepting a truncated download (curl can exit 0 on a short read); kills leaking parts on failure."""
    tag = "v2025.07.01-planes-readsb-prod-0"
    idx = _index(tag, [(f"{tag}.tar.aa", 5), (f"{tag}.tar.ab", 3)])
    run = _runner({f"{tag}.tar.aa": 5, f"{tag}.tar.ab": 2})
    with pytest.raises(RuntimeError, match="size mismatch"):
        af.download_day("2025-07-01", index=idx, raw_dir=tmp_path, runner=run)
    assert list(tmp_path.iterdir()) == []


def test_download_failed_request_raises_and_cleans(tmp_path):
    """Kills: ignoring a non-zero curl exit."""
    tag = "v2025.07.01-planes-readsb-prod-0"
    idx = _index(tag, [(f"{tag}.tar.aa", 5)])
    with pytest.raises(RuntimeError, match="download failed"):
        af.download_day("2025-07-01", index=idx, raw_dir=tmp_path, runner=_runner({}))
    assert list(tmp_path.iterdir()) == []


def test_download_day_absent_from_index_raises(tmp_path):
    """Kills: silently fetching a guessed URL when the release does not exist."""
    with pytest.raises(RuntimeError, match="no release"):
        af.download_day("2025-07-01", index={}, raw_dir=tmp_path, runner=_runner({}))


def _rel(n, start=0):
    return [{"tag_name": f"t{start + i}", "assets": [{"name": f"t{start + i}.tar", "size": 1,
                                                      "browser_download_url": "u"}]} for i in range(n)]


def test_build_index_stops_at_the_first_short_page(tmp_path, monkeypatch):
    """Kills: requesting the page after a short one (GitHub answers HTTP 422 past 1,000 results — the real failure)."""
    monkeypatch.setattr(af, "OUTDIR", tmp_path)
    pages = {1: _rel(100), 2: _rel(46, 100)}
    asked = []

    def get(url):
        p = int(url.rsplit("page=", 1)[1])
        asked.append(p)
        return pages[p]
    idx = af.build_index("2025", get=get)
    assert asked == [1, 2] and len(idx) == 146
    assert (tmp_path / "RELEASES_2025.json").exists()


def test_build_index_refuses_an_index_truncated_by_the_1000_cap(tmp_path, monkeypatch):
    """Kills: silently losing the oldest releases when a year has more than 1,000 of them; kills requesting page 11
    (GitHub answers 422 there) — the guard must fire after exactly 10 full pages."""
    monkeypatch.setattr(af, "OUTDIR", tmp_path)
    asked = []

    def get(url):
        p = int(url.rsplit("page=", 1)[1])
        asked.append(p)
        return _rel(100, 100 * p)
    with pytest.raises(RuntimeError, match="1,000"):
        af.build_index("2025", get=get)
    assert asked == list(range(1, 11))
    assert not (tmp_path / "RELEASES_2025.json").exists()


def test_exit_code_is_nonzero_when_any_day_failed_or_the_run_stopped():
    """Kills: the smoke's exit 0 after 0 of 1 days succeeded."""
    assert af.exit_code({"stopped": None, "days": [{"status": "ok"}, {"status": "skipped"}]}) == 0
    assert af.exit_code({"stopped": None, "days": [{"status": "ok"}, {"status": "failed"}]}) == 4
    assert af.exit_code({"stopped": "3 consecutive failures", "days": []}) == 3


# ---------------------------------------------------------------- run_days loop

class FakeIO:
    """Stands in for download + extract; records calls and can fail on chosen days."""

    def __init__(self, tmp: pathlib.Path, fail=(), rows=5, corrupt=0):
        self.tmp, self.fail, self.rows, self.calls, self.corrupt = tmp, set(fail), rows, [], corrupt

    def fetch(self, day):
        self.calls.append(day)
        p = self.tmp / f"raw_{day}.tar"
        p.write_bytes(b"x" * 10)
        return [p]

    def extract(self, paths, day, unparseable=None):
        if day in self.fail:
            raise RuntimeError(f"boom {day}")
        if unparseable is not None:
            unparseable.extend([1] * self.corrupt)
        return pd.DataFrame([["a", None, None, "C", "EHAM", 1.0, 0.0, 0.0, 1, 0, float("nan"), 0.0, 0.0, "adsb_icao"]]
                            * self.rows, columns=af.COLUMNS)


def _run(tmp, io, days, **kw):
    out = tmp / "out"
    kw.setdefault("free_gib", lambda: 100.0)
    return af.run_days(days, out_dir=out, fetch=io.fetch, extract=io.extract, log=lambda *a: None, **kw), out


def test_run_days_writes_one_parquet_per_day_and_deletes_raw(tmp_path):
    """Kills: raw archive left on disk (the volume is 94% full); kills not writing the day file."""
    io = FakeIO(tmp_path)
    man, out = _run(tmp_path, io, ["2025-01-01", "2025-07-01"])
    assert sorted(p.name for p in out.glob("adsb_*.parquet")) == ["adsb_2025-01-01.parquet", "adsb_2025-07-01.parquet"]
    assert not list(tmp_path.glob("raw_*.tar"))
    assert [d["status"] for d in man["days"]] == ["ok", "ok"]
    assert man["days"][0]["rows"] == 5


def test_run_days_resumes_by_skipping_days_already_written(tmp_path):
    """Kills: refetching days that exist (a restart would redo hours)."""
    io = FakeIO(tmp_path)
    _run(tmp_path, io, ["2025-01-01"])
    io2 = FakeIO(tmp_path)
    man, _ = _run(tmp_path, io2, ["2025-01-01", "2025-07-01"])
    assert io2.calls == ["2025-07-01"]
    assert [d["status"] for d in man["days"]] == ["skipped", "ok"]


def test_a_failed_day_is_logged_raw_deleted_and_the_loop_continues(tmp_path):
    """Kills: one bad day aborting the batch (per-day-fatal class); kills leaking the raw file on failure."""
    io = FakeIO(tmp_path, fail={"2025-01-02"})
    man, out = _run(tmp_path, io, ["2025-01-01", "2025-01-02", "2025-01-03"])
    assert [d["status"] for d in man["days"]] == ["ok", "failed", "ok"]
    assert "boom 2025-01-02" in man["days"][1]["error"]
    assert not list(tmp_path.glob("raw_*.tar"))
    assert not (out / "adsb_2025-01-02.parquet").exists()


def test_three_consecutive_failures_stop_the_run(tmp_path):
    """Kills: burning hours on a systematic failure (network down, layout change)."""
    days = [f"2025-01-0{i}" for i in range(1, 7)]
    io = FakeIO(tmp_path, fail=set(days))
    man, _ = _run(tmp_path, io, days)
    assert len(io.calls) == 3
    assert man["stopped"] == "3 consecutive failures"


def test_low_disk_stops_before_fetching(tmp_path):
    """Kills: fetching a 2.9 GiB archive with < 20 GiB free."""
    io = FakeIO(tmp_path)
    man, _ = _run(tmp_path, io, ["2025-01-01"], free_gib=lambda: 19.0)
    assert io.calls == []
    assert man["stopped"].startswith("free disk 19.0 GiB")


def test_zero_row_day_counts_as_failure(tmp_path):
    """Kills: silently accepting an empty day (parser or bbox broke)."""
    io = FakeIO(tmp_path, rows=0)
    man, out = _run(tmp_path, io, ["2025-01-01"])
    assert man["days"][0]["status"] == "failed"
    assert not (out / "adsb_2025-01-01.parquet").exists()


def test_manifest_is_written_to_disk_after_every_day(tmp_path):
    """Kills: a manifest only written at the end (a crash would lose the record)."""
    io = FakeIO(tmp_path, fail={"2025-01-02"})
    _, out = _run(tmp_path, io, ["2025-01-01", "2025-01-02"])
    m = json.loads((out / "MANIFEST.json").read_text())
    assert [d["day"] for d in m["days"]] == ["2025-01-01", "2025-01-02"]
    assert m["source"].startswith("adsb.lol globe_history")
    assert "ODbL" in m["licence"]


def test_day_list_for_2026_is_january_and_july_interleaved_with_no_spent_days():
    """The ship fetch (RESULT ADN → 2026). Kills: reusing the 2025 list; dropping days; carrying 2025's spent day."""
    days = af.day_list(2026)
    assert len(days) == 62 and len(set(days)) == 62
    assert days[:2] == ["2026-01-01", "2026-07-01"] and all(d.startswith(("2026-01", "2026-07")) for d in days)
    assert af.day_list() == af.day_list(2025) and af.day_list(2025)[-1] == "2025-01-09"


def test_each_year_writes_its_own_directory_and_manifest():
    """Kills: a 2026 run overwriting the 2025 MANIFEST.json (the record of what was fetched and how)."""
    assert af.out_dir_for(2025) == af.OUTDIR
    assert af.out_dir_for(2026) == af.OUTDIR / "y2026"


# ---------------------------------------------------------------- the other ten months of 2025 (arm A12, 2026-09-25)

def test_day_list_months_is_every_day_of_the_named_months_in_date_order():
    """Kills: dropping month-end days (Feb has 28 in 2025, Sep 30); kills leaking Jan/Jul into an extra-months run."""
    days = af.day_list_months(2025, (2, 9))
    assert len(days) == 28 + 30 and days == sorted(days)
    assert days[0] == "2025-02-01" and days[27] == "2025-02-28" and days[-1] == "2025-09-30"
    assert not any(d.startswith(("2025-01", "2025-07")) for d in days)


def test_day_list_months_refuses_a_month_outside_1_to_12():
    """Kills: silently returning nothing for a typo'd month."""
    with pytest.raises(ValueError, match="month"):
        af.day_list_months(2025, (13,))


def test_extra_months_write_to_their_own_directory_never_the_janjul_one():
    """Kills: an extra-months run writing into data/adsb/v2, where the served Jan/Jul builders glob adsb_*.parquet
    and would silently pick up ten more months, and whose MANIFEST.json would be overwritten."""
    d = af.out_dir_for(2025, months=(2, 3))
    assert d == af.OUTDIR / "y2025_other" and d != af.OUTDIR and d.parent == af.OUTDIR
    assert af.out_dir_for(2025) == af.OUTDIR


def test_download_falls_back_to_another_replica_when_prod_is_absent(tmp_path):
    """The '404 days': some days are published only as -staging-0tmp / -prod-0tmp / -staging-0 replicas.
    Kills: failing a day that has a replica."""
    tag = "v2025.06.05-planes-readsb-staging-0tmp"
    idx = _index(tag, [(f"{tag}.tar", 4)])
    paths = af.download_day("2025-06-05", index=idx, raw_dir=tmp_path, runner=_runner({f"{tag}.tar": 4}))
    assert [p.name for p in paths] == [f"{tag}.tar"]


def test_download_prefers_prod_over_every_other_replica(tmp_path):
    """Kills: a replica order that picks staging when prod exists (prod is what Jan/Jul were built from)."""
    idx = {}
    for t in ("staging-0", "prod-0", "prod-0tmp", "staging-0tmp", "mlatonly-0"):
        tag = f"v2025.06.06-planes-readsb-{t}"
        idx.update(_index(tag, [(f"{tag}.tar", 2)]))
    run = _runner({n: 2 for n in (f"{t}.tar" for t in idx)})
    paths = af.download_day("2025-06-06", index=idx, raw_dir=tmp_path, runner=run)
    assert [p.name for p in paths] == ["v2025.06.06-planes-readsb-prod-0.tar"]


def test_download_never_uses_the_mlat_only_release(tmp_path):
    """Kills: treating mlatonly-0 (a partial feed) as a replica of the day."""
    tag = "v2025.06.07-planes-readsb-mlatonly-0"
    with pytest.raises(RuntimeError, match="no release"):
        af.download_day("2025-06-07", index=_index(tag, [(f"{tag}.tar", 2)]), raw_dir=tmp_path, runner=_runner({}))


def test_manifest_records_which_release_tag_each_day_came_from(tmp_path):
    """Kills: a day fetched from a fallback replica being indistinguishable from a prod day in the record."""
    io = FakeIO(tmp_path)
    tag = "v2025.06.05-planes-readsb-staging-0tmp"

    def fetch(day):
        p = tmp_path / f"{tag}.tar"
        p.write_bytes(b"x")
        return [p]
    out = tmp_path / "out"
    man = af.run_days(["2025-06-05"], out_dir=out, fetch=fetch, extract=io.extract, log=lambda *a: None,
                      free_gib=lambda: 100.0)
    assert man["days"][0]["status"] == "ok" and man["days"][0]["tag"] == tag


def test_cli_months_routes_the_extra_months_days_to_their_own_directory(monkeypatch):
    """Kills: the CLI ignoring --months (fetching Jan/Jul again) or writing into the Jan/Jul directory."""
    seen = {}

    def fake_run_days(days, out_dir, log):
        seen.update(days=list(days), out=out_dir)
        return {"days": [], "stopped": None}
    monkeypatch.setattr(af, "run_days", fake_run_days)
    with pytest.raises(SystemExit) as e:
        af.main(["run", "--months", "2,12", "--limit", "3"])
    assert e.value.code == 0
    assert seen["days"] == ["2025-02-01", "2025-02-02", "2025-02-03"] and seen["out"] == af.OUTDIR / "y2025_other"


def test_a_year_end_day_is_found_in_the_next_years_release_index(tmp_path, monkeypatch):
    """2025-12-31 is published in adsblol/globe_history_2026, not _2025 (index rebuilt 2026-09-25).
    Kills: failing the last days of a year that the archive files under the next year's repository."""
    tag = "v2025.12.31-planes-readsb-prod-0"
    indexes = {"2025": {}, "2026": _index(tag, [(f"{tag}.tar", 3)])}
    monkeypatch.setattr(af, "_load_index", lambda year: indexes[year])
    for y in indexes:
        (tmp_path / f"RELEASES_{y}.json").write_text("{}")
    monkeypatch.setattr(af, "index_path", lambda year: tmp_path / f"RELEASES_{year}.json")
    paths = af.download_day("2025-12-31", raw_dir=tmp_path, runner=_runner({f"{tag}.tar": 3}))
    assert [p.name for p in paths] == [f"{tag}.tar"]


def test_regression_bug_manifest_provenance_survives_two_resumes(tmp_path):
    """Post-mortem 2026-09-26: a resume recorded a skipped day's `previous` only from days whose status was 'ok' in
    the manifest it replaced, so a day skipped in run 2 lost its record in run 3 (2025-02-01 of y2025_other became
    'unverified' and R12's feature build refused it). Kills: carrying provenance one hop only."""
    io = FakeIO(tmp_path)
    out = tmp_path / "out"
    kw = dict(out_dir=out, fetch=io.fetch, extract=io.extract, log=lambda *a: None, free_gib=lambda: 100.0)
    af.run_days(["2025-02-01"], **kw)                                     # run 1: fetched
    af.run_days(["2025-02-01", "2025-02-02"], **kw)                       # run 2: skipped (previous = ok)
    man = af.run_days(["2025-02-01", "2025-02-02"], **kw)                 # run 3: both skipped
    rec = {d["day"]: d for d in man["days"]}
    for day in ("2025-02-01", "2025-02-02"):
        assert rec[day]["status"] == "skipped"
        assert rec[day]["previous"] is not None and rec[day]["previous"]["status"] == "ok"
        assert rec[day]["previous"]["rows"] == 5


def test_a_skipped_day_whose_previous_record_is_not_ok_carries_no_provenance(tmp_path):
    """Only an 'ok' record is provenance. Kills: forwarding a failed or malformed `previous` (e.g. a hand-edited
    manifest) as if the day had been verified."""
    io = FakeIO(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "adsb_2025-02-01.parquet").write_bytes(b"")
    (out / "MANIFEST.json").write_text(json.dumps({"days": [
        {"day": "2025-02-01", "status": "skipped", "previous": {"day": "2025-02-01", "status": "failed"}}]}))
    man = af.run_days(["2025-02-01"], out_dir=out, fetch=io.fetch, extract=io.extract, log=lambda *a: None,
                      free_gib=lambda: 100.0)
    assert man["days"][0]["status"] == "skipped" and man["days"][0]["previous"] is None


def test_year_type_accepts_any_integer_from_2025_on():
    """Fails when year_type is hard-coded to {2025, 2026} (a `choices`-style check) instead of a >= 2025 bound
    (2030 would then be refused)."""
    assert af.year_type("2025") == 2025 and af.year_type("2026") == 2026 and af.year_type("2030") == 2030


def test_year_type_refuses_below_2025_naming_the_value():
    """Fails when the `year < 2025` lower bound is removed (2024 would be accepted and no error raised)."""
    import argparse
    with pytest.raises(argparse.ArgumentTypeError, match="2024"):
        af.year_type("2024")


def test_run_year_has_no_choices_restriction_and_keeps_its_default():
    """`run --year` accepts a year `choices=[2025, 2026]` never allowed before (e.g. 2030), and still defaults to
    2025 when omitted. Fails when `choices` is still attached to the argument (2030 would raise SystemExit) or the
    default changes (omitting --year would not give 2025)."""
    assert af.build_parser().parse_args(["run"]).year == 2025
    assert af.build_parser().parse_args(["run", "--year", "2030"]).year == 2030


def test_run_year_below_2025_is_refused_by_the_cli_with_a_clear_message(capsys):
    """Fails when `run --year 2024` is silently accepted instead of being refused (SystemExit(2) from argparse,
    with the reason on stderr)."""
    with pytest.raises(SystemExit) as e:
        af.build_parser().parse_args(["run", "--year", "2024"])
    assert e.value.code == 2
    assert "must be >= 2025" in capsys.readouterr().err


def test_index_year_is_consistent_with_run_year_same_type_same_floor():
    """`index --year` and `run --year` share one type function: both accept any year >= 2025 (not just 2025/2026),
    and both refuse the same way below it. Fails when `index --year` keeps its own unchecked str type instead of
    year_type (2024 would be accepted, or 2030 would stay a plain string instead of becoming the int 2030)."""
    assert af.build_parser().parse_args(["index", "--year", "2030"]).year == 2030
    with pytest.raises(SystemExit) as e:
        af.build_parser().parse_args(["index", "--year", "2024"])
    assert e.value.code == 2


# ---------------------------------------------------------------- unparseable members are counted (closing audit 2026-10-07)

def test_an_unparseable_member_is_counted_not_dropped_silently():
    """Reviewer M3: a member whose gzip or JSON is corrupt returned [] and nothing counted it, so a truncated day lost
    points silently on the documented rebuild path. Fails when the counter is not offered or not incremented."""
    bad = []
    assert af.parse_member_v2(b"\x1f\x8b" + b"not gzip at all", bad) == []
    assert af.parse_member_v2(b"{not json", bad) == []
    assert af.parse_member_v2(_trace([_pt(0, EHAM, "ground")]), bad) != []
    assert len(bad) == 2


def test_extract_day_counts_the_unparseable_members_of_an_archive(tmp_path):
    """One good trace member and one corrupt one in a tar: the good one's rows come back and the corrupt one is counted.
    Fails when extract_day drops the count on the floor between parse_member_v2 and the day record."""
    tar_path = tmp_path / "day.tar"
    with tarfile.open(tar_path, "w") as tf:
        for name, payload in (("x/trace_full_a.json", _trace([_pt(0, EHAM, "ground"), _pt(10, EHAM, "ground")])),
                              ("x/trace_full_b.json", b"{corrupt")):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    bad = []
    df = af.extract_day([tar_path], "2026-01-01", unparseable=bad)
    assert len(df) == 2 and len(bad) == 1


def test_the_day_record_carries_the_count_of_unparseable_members(tmp_path):
    """Fails when run_days does not hand extract a counter, or drops its length from the day record."""
    man, _ = _run(tmp_path, FakeIO(tmp_path, corrupt=3), ["2025-01-01"])
    assert man["days"][0]["status"] == "ok" and man["days"][0]["unparseable_members"] == 3
