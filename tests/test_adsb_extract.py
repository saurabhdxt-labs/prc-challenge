"""The ADS-B trace parser must not fail silently.

A parser that returns [] for every member produces an empty parquet and a clean-looking
"no signal" result — indistinguishable from a real negative. Two bugs of exactly this shape
already cost runs in this project: a NUL composite key that collapsed 2,277 stand keys to 10,
and `.astype("int64")` on timestamp[us] yielding microseconds, which made every ADS-B join
miss and printed 0% coverage. These tests pin the parser's contract on synthetic input.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("adsb_extract", ROOT / "scripts" / "adsb_extract.py")
ax = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ax)


def _trace(points, icao="abc123", ts=1_736_380_800.0):
    return {"icao": icao, "r": "G-TEST", "t": "A320", "timestamp": ts, "trace": points}


def test_which_airport_places_each_reference_point_in_its_own_box():
    """Every airport's own ARP must resolve to itself, and nowhere else.

    Fails when a box is mis-centred or PAD grows enough to overlap two airports.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    longitude test dropped from which_airport -> RED.
    """
    for ap, (lat, lon) in ax.APTS.items():
        assert ax.which_airport(lat, lon) == ap, f"{ap} ARP did not resolve to {ap}"
    assert ax.which_airport(0.0, 0.0) is None            # Gulf of Guinea
    assert ax.which_airport(40.6413, -73.7781) is None    # KJFK, not a scored airport


def test_boxes_do_not_overlap():
    """No two airport boxes may intersect, or samples are assigned to the wrong airport.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): PAD 0.15
    -> 2.0 -> RED.
    """
    items = list(ax.APTS.items())
    for i, (a, (la, lo)) in enumerate(items):
        for b, (lb, ob) in items[i + 1:]:
            assert abs(la - lb) > 2 * ax.PAD or abs(lo - ob) > 2 * ax.PAD, \
                f"{a} and {b} boxes overlap at PAD={ax.PAD}"


def test_parse_member_extracts_in_box_points_and_drops_the_rest():
    """Only points inside a box survive; time is base + offset; ground alt sets on_ground.

    Fails when the offset is treated as absolute (a classic epoch bug), when the box filter
    is inverted, or when "ground" is not recognised.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    offset treated as absolute time -> RED; "ground" not recognised -> RED.
    """
    lat, lon = ax.APTS["EHAM"]
    raw = json.dumps(_trace([
        [0.0, lat, lon, "ground", 0.0, 0, 0, 0, {"flight": "KLM836 "}],
        [30.0, lat, lon, 350.0, 12.5, 0, 0, 0, None],
        [60.0, 0.0, 0.0, 500.0, 200.0, 0, 0, 0, None],      # out of every box
    ])).encode()
    rows = ax.parse_member(raw)
    assert len(rows) == 2, f"expected 2 in-box points, got {len(rows)}"
    icao, reg, typ, cs, ap, t, la, lo, og, alt, gs = rows[0]
    assert ap == "EHAM" and og == 1 and cs == "KLM836"
    assert t == pytest.approx(1_736_380_800.0), "time must be timestamp + offset"
    assert rows[1][5] == pytest.approx(1_736_380_830.0)
    assert rows[1][8] == 0 and rows[1][9] == pytest.approx(350.0)
    assert rows[1][3] == "KLM836", "callsign must carry forward to later points"


def test_parse_member_handles_gzip_and_refuses_to_crash_on_junk():
    """Members may be gzip-compressed; malformed ones return [] rather than killing the run.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the gzip
    branch removed from parse_member -> RED; malformed JSON re-raised -> RED.
    """
    lat, lon = ax.APTS["LIRF"]
    blob = json.dumps(_trace([[0.0, lat, lon, "ground", 0.0, 0, 0, 0, None]])).encode()
    assert len(ax.parse_member(gzip.compress(blob))) == 1
    assert ax.parse_member(b"not json at all") == []
    assert ax.parse_member(json.dumps({"icao": "x"}).encode()) == []       # no trace key
    assert ax.parse_member(json.dumps(_trace([])).encode()) == []          # empty trace
    assert ax.parse_member(json.dumps(_trace([[0.0, None, None, 0]])).encode()) == []


def _archive(n_heat: int = 5, heat_size: int = 200_000, n_traces: int = 40, with_traces: bool = True) -> tuple:
    """(tar bytes, header offset of the first trace member) laid out like the 2026 archives: ./ and ./heatmap/ first
    with large members, then ./traces/xx/trace_full_<icao>.json (gzip), each with two EHAM points."""
    import io
    import tarfile
    buf = io.BytesIO()
    lat, lon = ax.APTS["EHAM"]
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tf:
        for name in ("./", "./heatmap"):
            ti = tarfile.TarInfo(name)
            ti.type = tarfile.DIRTYPE
            tf.addfile(ti)
        for k in range(n_heat):
            ti = tarfile.TarInfo(f"./heatmap/{k:02d}.bin.ttf")
            ti.size = heat_size + k                                   # odd sizes: the walk must round up to 512
            tf.addfile(ti, io.BytesIO(bytes([k % 251]) * ti.size))
        if with_traces:
            for k in range(n_traces):
                blob = gzip.compress(json.dumps(_trace([[0.0, lat, lon, "ground", 0.0, 0, 0, 0, {"flight": f"KLM{k}"}],
                                                        [10.0, lat, lon, "ground", 2.0, 0, 0, 0, None]],
                                                       icao=f"{k:06x}")).encode())
                ti = tarfile.TarInfo(f"./traces/{k % 256:02x}/trace_full_{k:06x}.json")
                ti.size = len(blob)
                tf.addfile(ti, io.BytesIO(blob))
    data = buf.getvalue()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tf:
        first = next((m.offset for m in tf.getmembers() if m.isfile() and "trace" in m.name), None)
    return data, first


def test_first_trace_offset_walks_past_the_heatmap_head_without_downloading_it():
    """The 2026 archives open with ~0.5 GB of ./heatmap/ members, so the old probe (bytes 0-60 MB) held no trace.
    The walk must land on the first trace member's header and read one window per large member, not the members.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the walk
    stops at the first file (a heatmap) -> RED; member data not rounded up to 512 -> RED; the window grows to
    swallow each member (the member bytes are downloaded) -> RED.
    """
    data, first = _archive()
    calls = []

    def rr(s, n):
        calls.append((s, n))
        return data[s:s + n]

    off, n_hdr, nbytes = ax.first_trace_offset(rr, window=65_536)
    assert off == first and n_hdr == 2 + 5 + 1                 # ./, ./heatmap, five heatmaps, then the trace
    heat = 5 * 200_000
    assert nbytes <= 7 * 65_536 < heat and len(calls) <= 7       # one window per 200 kB member, never the member itself
    assert all(n == 65_536 for _, n in calls)


def test_first_trace_offset_fails_loud_on_no_trace_on_junk_and_on_the_cap():
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the zero-
    block end-of-archive check removed -> RED.
    """
    data, _ = _archive(with_traces=False)
    with pytest.raises(SystemExit, match="end of archive"):
        ax.first_trace_offset(lambda s, n: data[s:s + n])
    with pytest.raises(SystemExit, match="no valid tar header"):
        ax.first_trace_offset(lambda s, n: (b"<html>not found</html>" + b"x" * 1000)[s:s + n])
    data, _ = _archive()
    with pytest.raises(SystemExit, match="within the first 3 tar headers"):
        ax.first_trace_offset(lambda s, n: data[s:s + n], max_headers=3)


@pytest.mark.parametrize("cut", ["data", "header"])
def test_a_probe_stream_starting_at_the_found_offset_parses_and_tolerates_truncation(tmp_path, cut):
    """The probe file begins at a member header in the middle of the archive and ends at an arbitrary byte -- inside a
    member's data (tarfile raises ReadError, which only probe mode may swallow) or inside a header (tarfile stops
    quietly). Either way the 20 complete trace members before the cut are parsed. The full-day mode re-raises.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again):
    truncation re-raised in probe mode -> RED. First version SURVIVED (its cut fell inside a header, where tarfile
    stops quietly); the data cut was added.
    """
    import io
    import tarfile
    data, first = _archive(n_traces=40)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tf:
        traces = [m for m in tf.getmembers() if m.isfile() and "trace" in m.name]
    end = traces[20].offset_data + 10 if cut == "data" else traces[20].offset + 300
    p = tmp_path / "x.tar.aa.probe"
    p.write_bytes(data[first:end])
    df = ax.extract([p], probe=True)
    assert df.icao.nunique() == 20 and (df.airport == "EHAM").all() and len(df) == 40
    assert (df.t - 1_736_380_800.0).isin([0.0, 10.0]).all()
    if cut == "data":
        with pytest.raises(tarfile.ReadError):
            ax.extract([p], probe=False)


def test_fetch_probe_requests_probe_bytes_from_the_first_trace_and_the_full_day_is_unchanged(tmp_path, monkeypatch):
    """Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the old
    probe range 0-60 MB reinstated -> RED.
    """
    monkeypatch.setattr(ax, "RAWDIR", tmp_path)
    monkeypatch.setattr(ax, "first_trace_offset", lambda rr: (123_904, 50, 3_276_800))
    cmds = []

    class R:
        returncode = 0

    monkeypatch.setattr(ax.subprocess, "run", lambda cmd, *a, **k: cmds.append(cmd) or R())
    url = ax.BASE.format(yr="2026", tag="v2026.01.05-planes-readsb-prod-0", part="aa")
    paths = ax.fetch("2026-01-05", probe=True)
    assert paths == [tmp_path / "v2026.01.05-planes-readsb-prod-0.tar.aa.probe"]
    assert cmds == [["curl", "-sfL", "--retry", "3", "-o", str(paths[0]), "-r", f"123904-{123904 + ax.PROBE_BYTES - 1}", url]]
    cmds.clear()
    paths = ax.fetch("2026-01-05", probe=False)
    assert [c[-1][-2:] for c in cmds] == ["aa", "ab"] and all("-r" not in c for c in cmds) and len(paths) == 2
    assert cmds[0] == ["curl", "-sfL", "--retry", "3", "-o", str(tmp_path / "v2026.01.05-planes-readsb-prod-0.tar.aa"), url]


def test_parse_member_tolerates_short_and_missing_fields():
    """Shape drift in the trace array must not raise; short points are skipped.
    Mutation-rehearsal 2026-09-10 (backup, break, run, restore byte-identical via diff -q, GREEN again): the
    short-point skip removed -> RED.
    """
    lat, lon = ax.APTS["EDDF"]
    rows = ax.parse_member(json.dumps(_trace([
        [0.0, lat, lon],                                   # too short -> skipped
        [10.0, lat, lon, None, None, 0, 0, 0, None],       # null alt -> treated as ground
    ])).encode())
    assert len(rows) == 1 and rows[0][8] == 1
