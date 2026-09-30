"""Tests for scripts/analysis/orl_build.py (arm ORL, plans/PREREG_orl_osm_route_length_2026_09_23.md).

Every test names the failure it catches. The post-compute lines (attach, the fit-vs-serve NaN gate, the audit
summary, the domain AUC) are tested on synthetic frames BEFORE the first real run (BC-24 shape: a post-compute
crash after the work is done). The geometry is tested on a hand-built airport whose route lengths are known
by construction, so a wrong threshold end, a wrong snap, or a runway crossing counted twice each fails one test.

Mutation rehearsal (write time, recorded here; re-run in logs/orl_scratch_tools/ on a REAL-copy mirror):
  flip the threshold end in `runway_thresholds`      -> test_runway_thresholds_assigns_designator_to_the_end_it_departs_from
  drop the snap distance from the route              -> test_route_table_route_is_snap_plus_graph_path
  include apron ways in the graph                    -> test_build_graph_uses_only_taxiway_runway_and_parking_position_ways
  make orl_matched NaN for unmatched rows            -> test_attach_flag_is_never_nan
  skip the "NaN mode absent in fit" check            -> test_nan_share_gate_refuses_a_serve_nan_mode_absent_in_fit
"""
from __future__ import annotations

import hashlib, importlib.util, json, math, pathlib, sys
import numpy as np, pandas as pd, pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(n, r):
    if n in sys.modules:
        return sys.modules[n]
    s = importlib.util.spec_from_file_location(n, ROOT / r); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


O = _load("orl_build", "scripts/analysis/orl_build.py")


def test_imported_module_is_the_one_under_test():
    """Catches the AGS symlink-mirror landmine: the module under test must be the file next to this ROOT."""
    assert pathlib.Path(O.__file__).resolve() == (ROOT / "scripts/analysis/orl_build.py").resolve()


# ---- constants and the query ---------------------------------------------------------------------------------
def test_constants_are_the_registered_ones():
    """Catches a column renamed, an airport dropped, or the rate limit relaxed after the hash."""
    assert list(O.APTS) == ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
    assert O.ORL_COLS == ["orl_route_m", "orl_route_med_m", "orl_straight_m", "orl_n_edges", "orl_rank"]
    assert O.ORL_FLAG == "orl_matched"
    assert O.SLEEP_S >= 10.0 and "prc-challenge" in O.USER_AGENT
    assert O.OVERPASS_URL.startswith("https://") and O.OVERPASS_URL.endswith("/api/interpreter")
    assert O.NAN_SHARE_TOL == 0.05 and O.MAX_SNAP_M == 300.0
    assert O.RADIUS_M == 8_000.0 and O.MAX_RUNWAY_SPAN_M == 6_000.0
    assert O.GRAPH_AEROWAYS == ("taxiway", "runway", "parking_position")


def test_overpass_query_asks_for_every_registered_feature_class_in_json():
    """Catches a feature class silently dropped from the pull (holding positions, gates, aprons are in the brief)."""
    q = O.overpass_query("LIRF", 41.8003, 12.2389)
    assert "[out:json]" in q
    for kind in ("taxiway", "runway", "parking_position", "holding_position", "apron", "gate"):
        assert kind in q, kind
    # bbox around the reference point, south,west,north,east
    assert f"{41.8003 - O.PAD_DEG:.4f},{12.2389 - O.PAD_DEG:.4f},{41.8003 + O.PAD_DEG:.4f},{12.2389 + O.PAD_DEG:.4f}" in q
    assert "out body" in q and ">;" in q


# ---- fetch and cache -----------------------------------------------------------------------------------------
def _fake_json(icao):
    return json.dumps({"version": 0.6, "generator": "fake", "osm3s": {"copyright": "ODbL"},
                       "elements": [{"type": "node", "id": 1, "lat": 0.0, "lon": 0.0,
                                     "tags": {"aeroway": "parking_position", "ref": icao}}]}).encode()


def test_fetch_airport_caches_raw_bytes_and_records_sha256_in_the_manifest(tmp_path):
    """Catches a manifest that does not describe the bytes on disk (the reproducibility contract)."""
    calls, sleeps = [], []
    fetcher = lambda q: (calls.append(q), _fake_json("LIRF"))[1]
    p = O.fetch_airport("LIRF", cache_dir=tmp_path, fetcher=fetcher, sleeper=sleeps.append)
    assert p == tmp_path / "osm_LIRF.json" and p.read_bytes() == _fake_json("LIRF")
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert man["LIRF"]["sha256"] == hashlib.sha256(_fake_json("LIRF")).hexdigest()
    assert man["LIRF"]["bytes"] == len(_fake_json("LIRF")) and man["LIRF"]["fetched_utc"].endswith("Z")
    assert man["LIRF"]["query_sha256"] == hashlib.sha256(calls[0].encode()).hexdigest()
    assert man["LIRF"]["licence"] == "ODbL 1.0" and man["LIRF"]["endpoint"] == O.OVERPASS_URL
    assert sleeps == [O.SLEEP_S], "one sleep after every network fetch (rate limit)"


def test_fetch_airport_records_the_endpoint_that_actually_answered(tmp_path):
    """Catches the provenance defect found on the first real run: the manifest named overpass-api.de even when the
    retry had fallen through to the mirror."""
    O.fetch_airport("LIRF", cache_dir=tmp_path, fetcher=lambda q: (_fake_json("LIRF"), "https://mirror/api/interpreter"), sleeper=lambda s: None)
    assert json.loads((tmp_path / "manifest.json").read_text())["LIRF"]["endpoint"] == "https://mirror/api/interpreter"


def test_fetch_airport_serves_the_cache_without_touching_the_network(tmp_path):
    """Catches a re-download on every run (rate limits, and the frozen-bytes contract)."""
    n = []
    O.fetch_airport("LIRF", cache_dir=tmp_path, fetcher=lambda q: (n.append(1), _fake_json("LIRF"))[1], sleeper=lambda s: None)
    sleeps = []
    O.fetch_airport("LIRF", cache_dir=tmp_path, fetcher=lambda q: (n.append(1), b"")[1], sleeper=sleeps.append)
    assert n == [1] and sleeps == []


def test_fetch_airport_refuses_an_overpass_error_page(tmp_path):
    """Catches D1's rate-limited HTML error pages being cached as data."""
    with pytest.raises(RuntimeError, match="LIRF.*not an Overpass JSON"):
        O.fetch_airport("LIRF", cache_dir=tmp_path, fetcher=lambda q: b"<html>rate limited</html>", sleeper=lambda s: None)
    assert not (tmp_path / "osm_LIRF.json").exists()


def test_fetch_airport_refuses_an_empty_element_list(tmp_path):
    """Catches a bbox typo that returns a valid but empty answer."""
    empty = json.dumps({"elements": []}).encode()
    with pytest.raises(RuntimeError, match="LIRF.*no elements"):
        O.fetch_airport("LIRF", cache_dir=tmp_path, fetcher=lambda q: empty, sleeper=lambda s: None)


def test_fetch_with_retry_backs_off_then_falls_to_the_mirror():
    """Catches the 504 that killed the first real fetch: a busy answer is retried with growing sleeps, then the next
    endpoint is tried; success returns the bytes; total exhaustion names every error."""
    calls, sleeps = [], []
    def opener(url, q):
        calls.append(url)
        if url == "A":
            raise O.OverpassBusy(f"{url}: HTTP 504")
        return b"ok"
    out = O.fetch_with_retry("q", endpoints=("A", "B"), attempts=3, backoff_s=10.0, opener=opener, sleeper=sleeps.append, log=lambda *a, **k: None)
    assert out == (b"ok", "B") and calls == ["A", "A", "A", "B"] and sleeps == [10.0, 20.0]
    with pytest.raises(RuntimeError, match="unavailable after retries.*A: HTTP 504.*B: HTTP 504"):
        O.fetch_with_retry("q", endpoints=("A", "B"), attempts=1, opener=lambda u, q: (_ for _ in ()).throw(O.OverpassBusy(f"{u}: HTTP 504")),
                           sleeper=sleeps.append, log=lambda *a, **k: None)


def test_open_maps_busy_http_codes_to_overpass_busy_and_raises_others(monkeypatch):
    """Catches a 406 (missing User-Agent) being retried forever, or a 504 escaping as a crash."""
    import urllib.error, io
    def urlopen(req, timeout):
        assert req.get_header("User-agent") == O.USER_AGENT
        raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b""))
    monkeypatch.setattr(O.urllib.request, "urlopen", urlopen)
    for code in O.RETRY_HTTP:
        with pytest.raises(O.OverpassBusy, match=str(code)):
            O._open("https://x/api/interpreter", "q")
    code = 406
    with pytest.raises(urllib.error.HTTPError):
        O._open("https://x/api/interpreter", "q")
    assert O.ENDPOINTS[0] == O.OVERPASS_URL and len(O.ENDPOINTS) == 2 and O.RETRY_ATTEMPTS == 3 and O.RETRY_BACKOFF_S == 30.0


def test_fetch_airport_refuses_an_unknown_airport(tmp_path):
    with pytest.raises(KeyError, match="XXXX"):
        O.fetch_airport("XXXX", cache_dir=tmp_path, fetcher=lambda q: b"", sleeper=lambda s: None)


def test_load_airport_verifies_the_manifest_sha(tmp_path):
    """Catches a cached file edited after the manifest was written."""
    O.fetch_airport("LIRF", cache_dir=tmp_path, fetcher=lambda q: _fake_json("LIRF"), sleeper=lambda s: None)
    assert O.load_airport("LIRF", cache_dir=tmp_path)["elements"][0]["id"] == 1
    (tmp_path / "osm_LIRF.json").write_bytes(_fake_json("LIRF") + b" ")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        O.load_airport("LIRF", cache_dir=tmp_path)


# ---- geometry primitives -------------------------------------------------------------------------------------
def test_haversine_one_degree_of_longitude_at_the_equator():
    """Catches radians/degrees or radius errors (1 deg at the equator = 111,195 m, 0.1 %)."""
    assert abs(O.haversine_m(0.0, 0.0, 0.0, 1.0) - 111_195.0) < 111.0
    assert O.haversine_m(10.0, 10.0, 10.0, 10.0) == 0.0
    a = O.haversine_m(np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0]))
    assert a.shape == (2,) and abs(a[0] - a[1]) < 200.0


def test_bearing_cardinal_directions():
    """Catches a swapped lat/lon in the bearing (north must be 0, east 90)."""
    assert abs(O.bearing_deg(0.0, 0.0, 1.0, 0.0) - 0.0) < 1e-6
    assert abs(O.bearing_deg(0.0, 0.0, 0.0, 1.0) - 90.0) < 1e-6
    assert abs(O.bearing_deg(0.0, 0.0, -1.0, 0.0) - 180.0) < 1e-6
    assert abs(O.bearing_deg(0.0, 0.0, 0.0, -1.0) - 270.0) < 1e-6


def test_angle_diff_wraps():
    assert O.angle_diff(350.0, 10.0) == 20.0 and O.angle_diff(10.0, 350.0) == 20.0 and O.angle_diff(90.0, 270.0) == 180.0


# ---- the synthetic airport -----------------------------------------------------------------------------------
def _node(i, lat, lon, **tags):
    e = {"type": "node", "id": i, "lat": lat, "lon": lon}
    if tags:
        e["tags"] = tags
    return e


def _way(i, nodes, **tags):
    return {"type": "way", "id": i, "nodes": list(nodes), "tags": tags}


def synthetic_elements():
    """One E-W runway 09/27 (nodes 101-102), a parallel taxiway A (201-204-202), two end connectors, a stub taxiway B
    (203-204), stands S1 (node, twice), S2 (lead-in way), S3;S4 (composite ref), FAR (unreachable), a gate G1 and a
    gate mis-tagged S1, an apron, an unnamed runway, a holding position."""
    return [
        _node(101, 0.0, 0.0), _node(102, 0.0, 0.03),                       # runway ends, west and east
        _node(201, 0.001, 0.0), _node(204, 0.001, 0.015), _node(202, 0.001, 0.03),
        _node(203, 0.002, 0.015),
        _node(301, 0.0021, 0.015, aeroway="parking_position", ref="s1 "),   # lower case + space: normalised
        _node(305, 0.0021, 0.0152, aeroway="parking_position", ref="S1"),   # duplicate ref: averaged
        _node(302, 0.0025, 0.0), _node(304, 0.0021, 0.0151, aeroway="parking_position", ref="S3;S4"),
        _node(303, 0.05, 0.05, aeroway="parking_position", ref="FAR"),          # 7.8 km away: a neighbour's stand
        _node(309, 0.006, 0.015, aeroway="parking_position", ref="OFFPAV"),     # 445 m off any pavement: snap cap
        _node(701, 0.0025, 0.0155), _node(702, 0.0022, 0.0155),                  # a lead-in way drawn DISCONNECTED (EDDF shape)
        _node(601, 0.06, 0.0), _node(602, 0.06, 0.03),                           # neighbour airfield 09/27, 6.6 km N
        _node(603, 0.0605, 0.01, aeroway="parking_position", ref="S1"),         # neighbour stand sharing our ref S1
        _node(306, 0.0021, 0.0149, aeroway="gate", ref="G1"),
        _node(307, 0.0022, 0.0149, aeroway="gate", ref="S1"),
        _node(308, 0.001, 0.001, aeroway="holding_position"),
        _node(401, 0.003, 0.0), _node(402, 0.003, 0.001), _node(403, 0.0031, 0.0),
        _node(501, 0.01, 0.0), _node(502, 0.01, 0.01),
        _way(1, [101, 102], aeroway="runway", ref="09/27"),
        _way(2, [201, 204, 202], aeroway="taxiway", ref="A"),
        _way(3, [201, 101], aeroway="taxiway"), _way(4, [202, 102], aeroway="taxiway"),
        _way(5, [203, 204], aeroway="taxiway", ref="B"),
        _way(6, [302, 201], aeroway="parking_position", ref="S2"),
        _way(7, [401, 402, 403, 401], aeroway="apron"),
        _way(8, [501, 502], aeroway="runway"),
        _way(9, [601, 602], aeroway="runway", ref="09/27"),
        _way(10, [701, 702], aeroway="parking_position", ref="ISO"),
    ]


REF = (0.001, 0.015)      # the synthetic airport's reference point (mid taxiway A)
RAD = 5_000.0


def test_parse_elements_splits_nodes_and_ways_and_refuses_a_dangling_node_ref():
    """Catches a truncated Overpass answer (a way whose node was never returned)."""
    nodes, ways = O.parse_elements(synthetic_elements())
    assert nodes[101] == (0.0, 0.0) and len(ways) == 10 and ways[0]["tags"]["ref"] == "09/27"
    with pytest.raises(ValueError, match="way 11.*node 999"):
        O.parse_elements(synthetic_elements() + [_way(11, [101, 999], aeroway="taxiway")])


def test_build_graph_uses_only_taxiway_runway_and_parking_position_ways():
    """Catches apron outlines or unnamed runways leaking into the routable graph (they are not taxi routes)."""
    nodes, ways = O.parse_elements(synthetic_elements())
    adj = O.build_graph(nodes, ways)
    assert 401 not in adj and 402 not in adj, "apron ring must not be routable"
    assert 501 in adj, "an unnamed runway is still pavement you can taxi on"
    assert 302 in adj and 201 in adj[302], "parking_position lead-in ways connect stands"
    assert abs(adj[201][204] - O.haversine_m(0.001, 0.0, 0.001, 0.015)) < 1e-9
    assert adj[204][201] == adj[201][204], "undirected"
    assert set(adj[204]) == {201, 202, 203}


def test_build_graph_two_ways_over_the_same_node_pair_give_one_edge():
    nodes = {1: (0.0, 0.0), 2: (0.0, 0.001)}
    ways = [{"id": 1, "nodes": [1, 2], "tags": {"aeroway": "taxiway"}}, {"id": 2, "nodes": [2, 1], "tags": {"aeroway": "taxiway"}}]
    adj = O.build_graph(nodes, ways)
    assert len(adj[1]) == 1 and abs(adj[1][2] - O.haversine_m(0.0, 0.0, 0.0, 0.001)) < 1e-9


def test_norm_ref_strips_and_uppercases():
    assert O.norm_ref(" s1 ") == "S1" and O.norm_ref("A58b") == "A58B" and O.norm_ref(None) == "" and O.norm_ref(float("nan")) == ""


def test_stand_points_normalises_splits_averages_and_prefers_parking_position_over_gate():
    """Catches: unnormalised refs (no join), composite refs lost, duplicates picked arbitrarily, gates overriding stands."""
    nodes, ways = O.parse_elements(synthetic_elements())
    st = O.stand_points(nodes, ways, synthetic_elements(), REF, RAD)
    assert set(st) == {"S1", "S2", "S3", "S4", "OFFPAV", "G1", "ISO"}, "FAR (7.8 km) and the neighbour's S1 (6.6 km) are outside the radius"
    assert st["ISO"] == (0.00235, 0.0155, "parking_position_way"), "both ends free -> centroid"
    lat, lon, src = st["S1"]
    assert abs(lat - 0.0021) < 1e-12 and abs(lon - 0.0151) < 1e-12 and src == "parking_position"   # mean of 301 and 305
    assert st["S3"][:2] == st["S4"][:2] == (0.0021, 0.0151)
    assert st["G1"][2] == "gate"
    assert st["S2"][2] == "parking_position_way" and st["S2"][:2] == (0.0025, 0.0)   # the lead-in way's free end


def test_stand_points_lead_in_way_falls_back_to_centroid_when_no_single_free_end():
    """Catches a lead-in line that is a stub with both ends free (or both joined) being placed at an arbitrary end."""
    nodes = {1: (0.0, 0.0), 2: (0.0, 0.001)}
    ways = [{"id": 1, "nodes": [1, 2], "tags": {"aeroway": "parking_position", "ref": "P"}}]
    st = O.stand_points(nodes, ways, [])
    assert st["P"] == (0.0, 0.0005, "parking_position_way")


def test_stand_points_requires_the_tagged_elements():
    """Catches the silent-loss path: without the tagged nodes every node stand would vanish."""
    with pytest.raises(TypeError):
        O.stand_points({}, [])


def test_stand_points_without_a_reference_point_keeps_everything_and_mixes_the_neighbours_ref():
    """Documents WHY the radius exists: without it, the neighbour's S1 is averaged into ours (lat pulled to 0.0216)."""
    nodes, ways = O.parse_elements(synthetic_elements())
    st = O.stand_points(nodes, ways, synthetic_elements())
    assert "FAR" in st and abs(st["S1"][0] - (0.0021 + 0.0021 + 0.0605) / 3) < 1e-12


def test_stand_points_ignores_unreffed_positions():
    nodes = {1: (0.0, 0.0)}
    st = O.stand_points(nodes, [], [{"type": "node", "id": 1, "lat": 0.0, "lon": 0.0, "tags": {"aeroway": "parking_position"}}])
    assert st == {}


def test_runway_thresholds_assigns_designator_to_the_end_it_departs_from():
    """Catches the threshold-end flip: a 09 departure STARTS at the west end and rolls east (bearing 90)."""
    nodes, ways = O.parse_elements(synthetic_elements())
    rep = {}
    th = O.runway_thresholds(nodes, ways, REF, RAD, report=rep)
    assert set(th) == {"09", "27"}
    assert th["09"] == (0.0, 0.0) and th["27"] == (0.0, 0.03)
    assert list(rep["runways_dropped"]) == ["09/27#9"] and rep["runway_refs_kept"] == ["09/27"]


def test_runway_thresholds_refuses_a_designator_from_two_airfields_when_no_radius_is_given():
    """Catches the silent collision: our 09/27 and the neighbour's 09/27 merge into one 6.6 km 'runway'."""
    nodes, ways = O.parse_elements(synthetic_elements())
    with pytest.raises(ValueError, match="09/27.*spans.*two airfields"):
        O.runway_thresholds(nodes, ways)


def test_runway_thresholds_refuses_a_designator_produced_twice():
    nodes = {1: (0.0, 0.0), 2: (0.0, 0.03), 3: (0.01, 0.0), 4: (0.01, 0.03)}
    ways = [{"id": 1, "nodes": [1, 2], "tags": {"aeroway": "runway", "ref": "09/27"}},
            {"id": 2, "nodes": [3, 4], "tags": {"aeroway": "runway", "ref": "09"}}]
    with pytest.raises(ValueError, match="designator 09 produced by two runway refs"):
        O.runway_thresholds(nodes, ways)


def test_runway_thresholds_accepts_a_single_designator_ref():
    """EDDF '18' and LTFM '09' are take-off-only runways with one designator (seen in the real pulls)."""
    nodes = {1: (0.0, 0.0), 2: (-0.03, 0.0)}   # runs N -> S
    ways = [{"id": 1, "nodes": [1, 2], "tags": {"aeroway": "runway", "ref": "18"}}]
    assert O.runway_thresholds(nodes, ways) == {"18": (0.0, 0.0)}


def test_runway_thresholds_merges_a_runway_split_into_several_ways():
    """Catches picking one segment's ends instead of the full runway's (OSM splits runways at intersections)."""
    nodes = {1: (0.0, 0.0), 2: (0.0, 0.01), 3: (0.0, 0.03), 4: (0.001, 0.0)}
    ways = [{"id": 1, "nodes": [1, 2], "tags": {"aeroway": "runway", "ref": "09/27"}},
            {"id": 2, "nodes": [2, 3], "tags": {"aeroway": "runway", "ref": "09/27"}},
            {"id": 3, "nodes": [1, 4], "tags": {"aeroway": "taxiway"}}]
    th = O.runway_thresholds(nodes, ways)
    assert th["09"] == (0.0, 0.0) and th["27"] == (0.0, 0.03)


def test_runway_thresholds_handles_parallel_designators_with_letters():
    nodes = {1: (0.0, 0.0), 2: (0.01, 0.0)}   # S-N runway
    ways = [{"id": 1, "nodes": [1, 2], "tags": {"aeroway": "runway", "ref": "36L/18R"}}]
    th = O.runway_thresholds(nodes, ways)
    assert th["36L"] == (0.0, 0.0) and th["18R"] == (0.01, 0.0)


def test_runway_thresholds_refuses_a_designator_that_matches_neither_end():
    """Catches a mis-tagged ref (or a lat/lon swap) silently producing a threshold."""
    nodes = {1: (0.0, 0.0), 2: (0.01, 0.0)}
    ways = [{"id": 1, "nodes": [1, 2], "tags": {"aeroway": "runway", "ref": "09/27"}}]
    with pytest.raises(ValueError, match="09/27.*bearing"):
        O.runway_thresholds(nodes, ways)


def test_runway_thresholds_skips_and_reports_a_ref_that_is_not_designators():
    """LSZH's heliport 'H01/H19' lies inside the radius: not a runway, not a collision — skipped and reported."""
    nodes = {1: (0.0, 0.0), 2: (0.01, 0.0), 3: (0.0, 0.03)}
    ways = [{"id": 1, "nodes": [1, 2], "tags": {"aeroway": "runway", "ref": "H01/H19"}},
            {"id": 2, "nodes": [1, 3], "tags": {"aeroway": "runway", "ref": "09/27"}}]
    rep = {}
    th = O.runway_thresholds(nodes, ways, report=rep)
    assert set(th) == {"09", "27"} and "H01/H19" in rep["runways_dropped"] and rep["runway_refs_kept"] == ["09/27"]


def test_nearest_node_returns_the_closest_graph_node_and_its_distance():
    nodes, ways = O.parse_elements(synthetic_elements())
    adj = O.build_graph(nodes, ways)
    nid, d = O.nearest_node(adj, nodes, 0.0021, 0.015)
    assert nid == 203 and abs(d - O.haversine_m(0.0021, 0.015, 0.002, 0.015)) < 1e-9


def test_dijkstra_distances_and_hop_counts():
    """Catches a hop count that counts nodes instead of edges, or a distance that ignores edge weights."""
    adj = {1: {2: 1.0, 3: 5.0}, 2: {1: 1.0, 3: 1.0}, 3: {1: 5.0, 2: 1.0}, 9: {}}
    dist, hops = O.dijkstra(adj, 1)
    assert dist == {1: 0.0, 2: 1.0, 3: 2.0} and hops == {1: 0, 2: 1, 3: 2}
    assert 9 not in dist, "unreachable nodes are absent, not infinite"


def test_route_table_route_is_snap_plus_graph_path():
    """Catches the snap distance dropped (a stand 100 m from the taxiway is not on it) and a wrong threshold end."""
    data = {"elements": synthetic_elements()}
    t = O.route_table("TEST", data, ref_point=REF, radius_m=RAD)
    r = t.set_index(["stand", "rwy"])
    H = O.haversine_m
    snap = H(0.0021, 0.0151, 0.002, 0.015)
    path09 = H(0.002, 0.015, 0.001, 0.015) + H(0.001, 0.015, 0.001, 0.0) + H(0.001, 0.0, 0.0, 0.0)
    assert abs(r.loc[("S1", "09"), "route_m"] - (snap + path09)) < 1e-6
    assert r.loc[("S1", "09"), "n_edges"] == 3
    assert abs(r.loc[("S1", "09"), "straight_m"] - H(0.0021, 0.0151, 0.0, 0.0)) < 1e-6
    assert abs(r.loc[("S1", "09"), "stand_snap_m"] - snap) < 1e-9 and r.loc[("S1", "09"), "rwy_snap_m"] == 0.0
    path27 = H(0.002, 0.015, 0.001, 0.015) + H(0.001, 0.015, 0.001, 0.03) + H(0.001, 0.03, 0.0, 0.03)
    assert abs(r.loc[("S1", "27"), "route_m"] - (snap + path27)) < 1e-6
    # S2 sits at its lead-in way's free end (node 302, in the graph): snap 0, path 302->201->101
    assert r.loc[("S2", "09"), "stand_snap_m"] == 0.0
    assert abs(r.loc[("S2", "09"), "route_m"] - (H(0.0025, 0.0, 0.001, 0.0) + H(0.001, 0.0, 0.0, 0.0))) < 1e-6 and r.loc[("S2", "09"), "n_edges"] == 2
    assert (t.icao == "TEST").all() and t.straight_m.le(t.route_m + 1e-9).all()


def test_route_table_snaps_a_disconnected_lead_in_stand_to_reachable_pavement():
    """The EDDF finding (187 of 337 stands 'no path'): a stand on an isolated stub must snap to the nearest node the
    runway can reach, with the real snap distance, not to its own unreachable stub at distance 0."""
    t = O.route_table("TEST", {"elements": synthetic_elements()}, ref_point=REF, radius_m=RAD).set_index(["stand", "rwy"])
    H = O.haversine_m
    snap = H(0.00235, 0.0155, 0.002, 0.015)             # to node 203, the nearest REACHABLE node
    path09 = H(0.002, 0.015, 0.001, 0.015) + H(0.001, 0.015, 0.001, 0.0) + H(0.001, 0.0, 0.0, 0.0)
    assert abs(t.loc[("ISO", "09"), "stand_snap_m"] - snap) < 1e-9 and snap > 60.0
    assert abs(t.loc[("ISO", "09"), "route_m"] - (snap + path09)) < 1e-6 and t.loc[("ISO", "09"), "n_edges"] == 3


def test_route_table_unreachable_or_far_stands_are_absent_with_a_reason():
    """Catches a far-away stand snapped to some node kilometres away and given a meaningless route."""
    t = O.route_table("TEST", {"elements": synthetic_elements()}, ref_point=REF, radius_m=RAD)
    assert "OFFPAV" not in set(t.stand) and "FAR" not in set(t.stand)
    assert t.attrs["dropped"] == {"OFFPAV": f"stand snap 445 m > {O.MAX_SNAP_M:.0f} m"}


def test_route_table_refuses_a_bbox_with_a_colliding_neighbour_when_the_radius_is_too_wide():
    with pytest.raises(ValueError, match="two airfields"):
        O.route_table("TEST", {"elements": synthetic_elements()}, ref_point=REF, radius_m=50_000.0)


def test_route_table_defaults_the_reference_point_to_the_airport_table():
    with pytest.raises(KeyError):
        O.route_table("TEST", {"elements": synthetic_elements()})


def test_route_table_reports_counts():
    rep = {}
    O.route_table("TEST", {"elements": synthetic_elements()}, report=rep, ref_point=REF, radius_m=RAD)
    assert rep["n_stands"] == 7 and rep["n_runway_thresholds"] == 2 and rep["n_graph_nodes"] == 13
    assert rep["n_graph_nodes_reachable"] == 7, "the isolated stub (2), the unnamed runway (2) and the neighbour runway (2) are unreachable"
    assert rep["n_stands_routed"] == 6 and rep["stand_sources"] == {"parking_position": 4, "parking_position_way": 2, "gate": 1}
    assert rep["n_holding_positions"] == 1 and rep["n_stands_dropped"] == 1
    assert rep["ref_point"] == list(REF) and rep["radius_m"] == RAD and rep["runway_refs_kept"] == ["09/27"]


# ---- attach (the ONE code path for fit and serve) -------------------------------------------------------------
def _routes():
    return pd.DataFrame({
        "icao": ["TEST"] * 4 + ["OTHR"] * 3,
        "stand": ["S1", "S1", "S2", "S2", "Z", "Z", "Z"],
        "rwy": ["09", "27", "09", "27", "09", "27", "18"],
        "route_m": [1000.0, 3000.0, 200.0, 3400.0, 500.0, 700.0, 6000.0],
        "n_edges": [3, 3, 2, 4, 1, 1, 9],
        "straight_m": [900.0, 2900.0, 150.0, 3300.0, 400.0, 600.0, 5000.0],
        "stand_snap_m": [10.0] * 7, "rwy_snap_m": [0.0] * 7,
    })


def _frame(stands, rwys, icao="TEST"):
    return pd.DataFrame({"ADEP_mvt": [icao] * len(stands), "STAND_mvt": stands, "RUNWAY_mvt": rwys})


def test_attach_matched_row_values():
    """Catches a wrong lookup (stand x runway swapped) or a rank computed over the wrong population."""
    a = O.attach(_frame(["S1", "S2"], ["09", "27"]), _routes())
    assert list(a.columns) == O.ORL_COLS + [O.ORL_FLAG]
    assert a.orl_route_m.tolist() == [1000.0, 3400.0] and a.orl_n_edges.tolist() == [3, 4]
    assert a.orl_straight_m.tolist() == [900.0, 3300.0]
    assert a.orl_route_med_m.tolist() == [2000.0, 1800.0]            # median over the stand's runways
    # rank = fraction of the airport's route-table values strictly below, among TEST's 4 values (200, 1000, 3000, 3400)
    assert a.orl_rank.tolist() == [0.25, 0.75]
    assert a.orl_matched.tolist() == [1, 1]


def test_attach_unmatched_stand_is_nan_everywhere_and_flag_zero():
    a = O.attach(_frame(["NOPE"], ["09"]), _routes())
    assert a[O.ORL_COLS].isna().all().all() and a.orl_matched.tolist() == [0]


def test_attach_unmapped_runway_keeps_the_stand_median_only():
    """Catches the fallback column going NaN when only the runway is unknown (NA / HEL / H values exist in RUNWAY_mvt)."""
    a = O.attach(_frame(["S1", "S1", "S1"], ["NA", "HEL", "H"]), _routes())
    assert a.orl_route_m.isna().all() and a.orl_straight_m.isna().all() and a.orl_n_edges.isna().all()
    assert a.orl_rank.isna().all()
    assert a.orl_route_med_m.tolist() == [2000.0] * 3 and a.orl_matched.tolist() == [0, 0, 0]


def test_attach_normalises_keys_and_is_airport_scoped():
    """Catches a stand ref of another airport matching (refs repeat across airports) and unnormalised keys."""
    a = O.attach(_frame([" s1", "Z"], ["09", "09"]), _routes())
    assert a.orl_route_m.tolist()[0] == 1000.0 and np.isnan(a.orl_route_m.tolist()[1])
    b = O.attach(_frame(["Z"], ["09"], icao="OTHR"), _routes())
    assert b.orl_route_m.tolist() == [500.0] and b.orl_rank.tolist() == [0.0]
    assert b.orl_route_med_m.tolist() == [700.0], "median over 500/700/6000, not the mean (2400)"


def test_attach_flag_is_never_nan_and_is_int():
    a = O.attach(_frame(["S1", "NOPE", None], ["09", "09", "09"]), _routes())
    assert a.orl_matched.notna().all() and a.orl_matched.dtype.kind in "iu" and a.orl_matched.tolist() == [1, 0, 0]


def test_attach_is_a_pure_function_of_the_three_keys():
    """The fit/serve contract: extra columns (labels!) and row order never change the output for the same keys."""
    f = _frame(["S1", "S2", "NOPE"], ["09", "NA", "27"])
    g = f.copy(); g["TAXITIME_SEC_mvt"] = [1.0, 2.0, 3.0]; g["y"] = 5.0
    a, b = O.attach(f, _routes()), O.attach(g, _routes())
    pd.testing.assert_frame_equal(a, b)
    c = O.attach(f.iloc[::-1].reset_index(drop=True), _routes())
    pd.testing.assert_frame_equal(a.iloc[::-1].reset_index(drop=True), c)
    assert (a.index == f.index).all()


def test_attach_refuses_a_frame_without_the_keys():
    with pytest.raises(KeyError, match="RUNWAY_mvt"):
        O.attach(pd.DataFrame({"ADEP_mvt": ["TEST"], "STAND_mvt": ["S1"]}), _routes())


def test_designator_from_mvt():
    """Catches junk RUNWAY_mvt values (NA, HEL, H) being looked up as runways."""
    assert O.designator_from_mvt("25C") == "25C" and O.designator_from_mvt(" 07r ") == "07R" and O.designator_from_mvt("36") == "36"
    for junk in ("NA", "HEL", "H", None, float("nan"), "", "1", "123"):
        assert O.designator_from_mvt(junk) is None, junk


# ---- the fit-vs-serve NaN gate -------------------------------------------------------------------------------
def _shares(**per_ap):
    """per_ap: icao -> (n_rows, nan_share) for orl_route_m; other columns copy it."""
    rows = []
    for ap, (n, s) in per_ap.items():
        k = int(round(n * s))
        rows.append(pd.DataFrame({"ADEP_mvt": [ap] * n,
                                  "orl_route_m": [np.nan] * k + [1.0] * (n - k),
                                  "orl_route_med_m": [np.nan] * k + [1.0] * (n - k),
                                  "orl_straight_m": [np.nan] * k + [1.0] * (n - k),
                                  "orl_n_edges": [np.nan] * k + [1.0] * (n - k),
                                  "orl_rank": [np.nan] * k + [1.0] * (n - k),
                                  "orl_matched": [0] * k + [1] * (n - k)}))
    return pd.concat(rows, ignore_index=True)


def test_nan_share_gate_passes_when_fit_and_serve_agree_and_reports_both():
    fit, serve = _shares(A=(100, 0.10), B=(100, 0.0)), _shares(A=(50, 0.12), B=(50, 0.0))
    rep = O.nan_share_gate(fit, serve)
    assert rep["A"]["orl_route_m"] == {"fit": 0.10, "serve": 0.12}
    assert rep["B"]["orl_route_m"] == {"fit": 0.0, "serve": 0.0}


def test_nan_share_gate_refuses_a_serve_nan_mode_absent_in_fit():
    """The DFC F2 lesson: a NaN mode the learner never saw in fit must not exist at serve."""
    fit, serve = _shares(A=(100, 0.0)), _shares(A=(100, 0.01))
    with pytest.raises(AssertionError, match="A.*orl_route_m.*serve NaN share 0.0100.*fit 0.0000"):
        O.nan_share_gate(fit, serve)


def test_nan_share_gate_refuses_a_share_drift_beyond_tolerance():
    fit, serve = _shares(A=(100, 0.10)), _shares(A=(100, 0.20))
    with pytest.raises(AssertionError, match="A.*orl_route_m.*drift 0.1000 > 0.05"):
        O.nan_share_gate(fit, serve)
    O.nan_share_gate(fit, serve, tol=0.15)


def test_nan_share_gate_refuses_an_airport_missing_on_either_side():
    with pytest.raises(AssertionError, match="airports differ"):
        O.nan_share_gate(_shares(A=(10, 0.0)), _shares(A=(10, 0.0), B=(10, 0.0)))


# ---- audit ---------------------------------------------------------------------------------------------------
def test_domain_auc_is_half_for_identical_distributions_and_one_for_separated():
    """Catches an AUC that reads the labels (always 1) or the wrong orientation."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(3000, 2)), rng.normal(size=(3000, 2))
    auc = O.domain_auc(a, b, seed=0)
    assert 0.45 <= auc <= 0.55, auc
    assert O.domain_auc(a, b + 5.0, seed=0) > 0.97


def test_domain_auc_handles_nan_columns():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2000, 2)), rng.normal(size=(2000, 2))
    a[:100, 0] = np.nan; b[:100, 0] = np.nan
    assert 0.44 <= O.domain_auc(a, b, seed=0) <= 0.56


def test_column_summary_quantiles_by_airport_and_null_share():
    f = _shares(A=(100, 0.10), B=(100, 0.0))
    s = O.column_summary(f)
    assert s["orl_route_m"]["A"]["null_share"] == 0.10 and s["orl_route_m"]["B"]["null_share"] == 0.0
    assert s["orl_route_m"]["A"]["q50"] == 1.0 and set(s["orl_route_m"]["A"]) >= {"n", "null_share", "q01", "q50", "q99", "mean"}
    assert s["orl_matched"]["A"]["mean"] == 0.9


def test_match_rate_table_per_airport():
    """Catches the match rate being computed on rows with a mapped runway only (it is the STAND join that D1 measured)."""
    a = O.attach(_frame(["S1", "S2", "NOPE", "S1"], ["09", "NA", "09", "HEL"]), _routes())
    f = _frame(["S1", "S2", "NOPE", "S1"], ["09", "NA", "09", "HEL"])
    m = O.match_rates(f, a)
    assert m["TEST"] == {"n": 4, "stand_match": 0.75, "rwy_mapped": 0.5, "both": 0.25}


def test_runway_mapping_table_lists_every_mvt_value_with_its_coverage():
    f = _frame(["S1", "S1", "S1", "S2"], ["09", "09", "NA", "27"])
    t = O.runway_mapping(f, _routes())
    assert t == {"TEST": {"09": {"n": 2, "mapped": True}, "NA": {"n": 1, "mapped": False}, "27": {"n": 1, "mapped": True}}}


def test_unseen_key_share_counts_serve_rows_whose_pair_or_stand_never_appears_in_fit():
    """INV section 6.3: the share of serve rows on populations with zero fit rows, per airport, for the pair and the stand."""
    fit = _frame(["S1", "S1", "S2"], ["09", "27", "09"])
    serve = _frame(["S1", "S2", "S2", "S9"], ["09", "27", "09", "09"])
    r = O.unseen_key_share(fit, serve)
    assert r["TEST"] == {"n": 4, "pair_unseen": 0.5, "stand_unseen": 0.25}


# ---- the CLI -------------------------------------------------------------------------------------------------
def test_build_writes_the_route_table_and_report_from_a_cache_dir(tmp_path):
    """The build stage end to end on the synthetic airport: parquet + JSON report, no network."""
    (tmp_path / "osm_TEST.json").write_text(json.dumps({"elements": synthetic_elements()}))
    man = {"TEST": {"sha256": hashlib.sha256((tmp_path / "osm_TEST.json").read_bytes()).hexdigest()}}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    monkey = O.APTS.copy(); O.APTS["TEST"] = REF
    try:
        out = O.run_build(cache_dir=tmp_path, apts=["TEST"], out_path=tmp_path / "routes.parquet", report_path=tmp_path / "r.json", radius_m=RAD)
    finally:
        O.APTS.clear(); O.APTS.update(monkey)
    t = pd.read_parquet(tmp_path / "routes.parquet")
    assert len(t) == 12 and set(t.icao) == {"TEST"} and out["TEST"]["n_stands_routed"] == 6
    assert json.loads((tmp_path / "r.json").read_text())["TEST"]["n_runway_thresholds"] == 2


def test_run_build_refuses_a_sha_mismatch(tmp_path):
    (tmp_path / "osm_TEST.json").write_text(json.dumps({"elements": synthetic_elements()}))
    (tmp_path / "manifest.json").write_text(json.dumps({"TEST": {"sha256": "0" * 64}}))
    O.APTS["TEST"] = REF
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        O.run_build(cache_dir=tmp_path, apts=["TEST"], out_path=tmp_path / "routes.parquet", report_path=tmp_path / "r.json")


def test_main_refuses_unknown_command(capsys):
    with pytest.raises(SystemExit):
        O.main(["nonsense"])
