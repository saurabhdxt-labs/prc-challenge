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
    """
    for ap, (lat, lon) in ax.APTS.items():
        assert ax.which_airport(lat, lon) == ap, f"{ap} ARP did not resolve to {ap}"
    assert ax.which_airport(0.0, 0.0) is None            # Gulf of Guinea
    assert ax.which_airport(40.6413, -73.7781) is None    # KJFK, not a scored airport


def test_boxes_do_not_overlap():
    """No two airport boxes may intersect, or samples are assigned to the wrong airport."""
    items = list(ax.APTS.items())
    for i, (a, (la, lo)) in enumerate(items):
        for b, (lb, ob) in items[i + 1:]:
            assert abs(la - lb) > 2 * ax.PAD or abs(lo - ob) > 2 * ax.PAD, \
                f"{a} and {b} boxes overlap at PAD={ax.PAD}"


def test_parse_member_extracts_in_box_points_and_drops_the_rest():
    """Only points inside a box survive; time is base + offset; ground alt sets on_ground.

    Fails when the offset is treated as absolute (a classic epoch bug), when the box filter
    is inverted, or when "ground" is not recognised.
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
    """Members may be gzip-compressed; malformed ones return [] rather than killing the run."""
    lat, lon = ax.APTS["LIRF"]
    blob = json.dumps(_trace([[0.0, lat, lon, "ground", 0.0, 0, 0, 0, None]])).encode()
    assert len(ax.parse_member(gzip.compress(blob))) == 1
    assert ax.parse_member(b"not json at all") == []
    assert ax.parse_member(json.dumps({"icao": "x"}).encode()) == []       # no trace key
    assert ax.parse_member(json.dumps(_trace([])).encode()) == []          # empty trace
    assert ax.parse_member(json.dumps(_trace([[0.0, None, None, 0]])).encode()) == []


def test_parse_member_tolerates_short_and_missing_fields():
    """Shape drift in the trace array must not raise; short points are skipped."""
    lat, lon = ax.APTS["EDDF"]
    rows = ax.parse_member(json.dumps(_trace([
        [0.0, lat, lon],                                   # too short -> skipped
        [10.0, lat, lon, None, None, 0, 0, 0, None],       # null alt -> treated as ground
    ])).encode())
    assert len(rows) == 1 and rows[0][8] == 1
