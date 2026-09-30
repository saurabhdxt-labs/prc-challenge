"""ORL — physical stand -> runway-threshold taxi route length from OpenStreetMap (ODbL), as matched-lane features.
Registration: plans/PREREG_orl_osm_route_length_2026_09_23.md. PREPARE-ONLY: nothing here fits a model, uploads, or
touches git. The geometry is static, so one lookup table serves 2025 (fit) and 2026 (serve) through ONE function
(`attach`) — the DFC F2 lesson: fit and serve share a code path, and `nan_share_gate` refuses a NaN mode at serve
that fit never saw.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/orl_build.py fetch          # Overpass, one query per airport,
                                                                                # SLEEP_S between, raw JSON cached under
                                                                                # data/osm/ with a sha256 manifest
    /opt/homebrew/bin/python3.11 -B scripts/analysis/orl_build.py build          # taxiway graph -> data/osm/orl_routes.parquet
    /opt/homebrew/bin/python3.11 -B scripts/analysis/orl_build.py audit          # join to 2025 fit + 2026 serve rows
                                                                                # (label-free), NaN gate, domain AUC

Data: OpenStreetMap contributors, Open Database License 1.0 (ODbL) — https://www.openstreetmap.org/copyright.
Overpass: https://overpass-api.de/api/interpreter (public instance; needs a User-Agent; 406 without one — D1 §5).

Route definition (per stand s, runway designator d at one airport):
    route_m(s, d) = snap(s) + shortest path over the graph of aeroway=taxiway|runway|parking_position ways from the
                    graph node nearest s to the graph node nearest the threshold of d (Dijkstra, haversine edge weights)
    threshold(d)  = the end of the runway from which a departure on d rolls: the end e such that the bearing e -> other
                    end is within 45 deg of 10 x the designator number (runway ways with the same ref are merged first)
    snap(s)       = haversine distance from the stand point to its nearest graph node AMONG THOSE REACHABLE from a
                    runway threshold (a lead-in stub drawn disconnected from the taxiways is not taxiable pavement);
                    stands beyond MAX_SNAP_M are dropped (reported), never routed from a node kilometres away
Runway edges are routable (taxi via a runway, back-taxi) at their true length; no penalty. Stated, not tuned.

Columns (all NaN when the (airport, stand) is not in the table; the flag is 0/1, never NaN):
    orl_route_m      route_m(stand, RUNWAY_mvt)                       NaN if stand unmatched or runway unmapped
    orl_route_med_m  median over the airport's designators of route_m(stand, .)   NaN only if stand unmatched
    orl_straight_m   haversine(stand point, threshold(RUNWAY_mvt))
    orl_n_edges      graph edges on the shortest path
    orl_rank         fraction of the airport's route-table values strictly below orl_route_m (static, in [0, 1])
    orl_matched      1 if orl_route_m is finite else 0
"""
from __future__ import annotations

import argparse, hashlib, heapq, json, math, pathlib, sys, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

#: airport reference points (scripts/adsb_extract.py:34-39, the same table the ADS-B extractor uses)
APTS = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785), "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}
PAD_DEG = 0.15                       # bbox half-width; the extractor's box, holds apron + runways at all ten
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
#: public instances, in order; the kumi mirror answered EGLL when overpass-api.de returned 504/429 (ADSBX report §1.5)
ENDPOINTS = (OVERPASS_URL, "https://overpass.kumi.systems/api/interpreter")
RETRY_HTTP = (429, 502, 503, 504)
RETRY_ATTEMPTS, RETRY_BACKOFF_S = 3, 30.0
USER_AGENT = "prc-challenge-orl/1.0 (PRC data challenge research; one query per airport, cached)"
SLEEP_S = 12.0                       # D1 §5: rate limits hit on 6/10 first attempts, fine with >= 12 s pauses
TIMEOUT_S = 180
CACHE = ROOT / "data" / "osm"
ROUTES = CACHE / "orl_routes.parquet"
REPORT_BUILD = ROOT / "reports" / "orl_build_report.json"
REPORT_AUDIT = ROOT / "reports" / "orl_audit_report.json"
RAW = ROOT / "data" / "raw"
KEY_COLS = ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
ORL_COLS = ["orl_route_m", "orl_route_med_m", "orl_straight_m", "orl_n_edges", "orl_rank"]
ORL_FLAG = "orl_matched"
GRAPH_AEROWAYS = ("taxiway", "runway", "parking_position")
MAX_SNAP_M = 300.0                   # a stand further than this from any taxi pavement is unmapped, not mis-routed
BEARING_TOL_DEG = 45.0
RADIUS_M = 8_000.0                   # stands/runways further than this from the reference point belong to a neighbour airfield
                                     # (Le Bourget 9.0 km from LFPG's point, Egelsbach 9.5 km from EDDF's; Polderbaan's centroid 5.3 km)
MAX_RUNWAY_SPAN_M = 6_000.0          # a ref group spanning more than this is two airfields sharing a ref, refused
NAN_SHARE_TOL = 0.05                 # per airport, per column, |fit share - serve share| beyond this refuses
EARTH_R_M = 6_371_008.8
AUDIT_N_PER_SIDE = 40_000
AUDIT_SEED = 0


# ---- fetch and cache ------------------------------------------------------------------------------------------
def overpass_query(icao: str, lat: float, lon: float, pad: float = PAD_DEG) -> str:
    bbox = f"{lat - pad:.4f},{lon - pad:.4f},{lat + pad:.4f},{lon + pad:.4f}"
    return (f"[out:json][timeout:{TIMEOUT_S}];"
            f"(node({bbox})[\"aeroway\"~\"^(parking_position|holding_position|gate)$\"];"
            f"way({bbox})[\"aeroway\"~\"^(parking_position|taxiway|runway|apron|holding_position)$\"];);"
            f"out body;>;out skel qt;")  # icao {icao}


def sha256_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OverpassBusy(RuntimeError):
    """A 429/504 (or connection failure) from an Overpass endpoint: retried, then the next endpoint."""


def _open(url: str, query: str) -> bytes:
    req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": query}).encode(),
                                 headers={"User-Agent": USER_AGENT, "Accept": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S + 60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code in RETRY_HTTP:
            raise OverpassBusy(f"{url}: HTTP {e.code}") from e
        raise
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise OverpassBusy(f"{url}: {e}") from e


def fetch_with_retry(query: str, endpoints=None, attempts: int = RETRY_ATTEMPTS, backoff_s: float = RETRY_BACKOFF_S,
                     opener=_open, sleeper=time.sleep, log=print):
    """Try each endpoint up to `attempts` times on a busy answer, sleeping `backoff_s` x attempt between tries.
    Returns (bytes, url_that_answered) so the manifest records the true provenance."""
    errors = []
    for url in (endpoints or ENDPOINTS):
        for k in range(1, attempts + 1):
            try:
                return opener(url, query), url
            except OverpassBusy as e:
                errors.append(str(e)); log(f"  busy ({e}); attempt {k}/{attempts}", flush=True)
                if k < attempts:
                    sleeper(backoff_s * k)
    raise RuntimeError("Overpass unavailable after retries: " + " | ".join(errors))


def _default_fetcher(query: str):
    return fetch_with_retry(query)


def _read_manifest(cache_dir: pathlib.Path) -> dict:
    p = cache_dir / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def fetch_airport(icao: str, cache_dir: pathlib.Path = CACHE, fetcher=None, sleeper=time.sleep,
                  sleep_s: float = SLEEP_S) -> pathlib.Path:
    """One Overpass pull per airport, cached as raw bytes; a cache hit never touches the network and never sleeps."""
    if icao not in APTS:
        raise KeyError(f"{icao}: not one of the ten challenge airports")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"osm_{icao}.json"
    if out.exists() and icao in _read_manifest(cache_dir):
        return out
    fetcher = fetcher or _default_fetcher
    query = overpass_query(icao, *APTS[icao])
    got = fetcher(query)
    raw, endpoint = got if isinstance(got, tuple) else (got, OVERPASS_URL)
    try:
        doc = json.loads(raw)
    except ValueError:
        doc = None
    if not isinstance(doc, dict) or "elements" not in doc:
        raise RuntimeError(f"{icao}: response is not an Overpass JSON document (rate-limit/error page?): {raw[:120]!r}")
    if not doc["elements"]:
        raise RuntimeError(f"{icao}: Overpass returned no elements for the bbox")
    out.write_bytes(raw)
    man = _read_manifest(cache_dir)
    man[icao] = {"file": out.name, "sha256": sha256_file(out), "bytes": len(raw),
                 "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "query_sha256": hashlib.sha256(query.encode()).hexdigest(), "endpoint": endpoint,
                 "osm_base": (doc.get("osm3s") or {}).get("timestamp_osm_base"),
                 "licence": "ODbL 1.0", "attribution": "(c) OpenStreetMap contributors, https://www.openstreetmap.org/copyright"}
    (cache_dir / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    sleeper(sleep_s)
    return out


def load_airport(icao: str, cache_dir: pathlib.Path = CACHE) -> dict:
    """Read a cached pull, refusing bytes that no longer match the manifest."""
    p = cache_dir / f"osm_{icao}.json"
    man = _read_manifest(cache_dir)
    if icao not in man:
        raise RuntimeError(f"{icao}: not in {cache_dir / 'manifest.json'}")
    if sha256_file(p) != man[icao]["sha256"]:
        raise RuntimeError(f"{icao}: sha256 mismatch between {p.name} and the manifest")
    return json.loads(p.read_bytes())


# ---- geometry primitives ---------------------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    d = 2 * EARTH_R_M * np.arcsin(np.sqrt(a))
    return float(d) if np.ndim(d) == 0 else d


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    p1, p2, dl = math.radians(lat1), math.radians(lat2), math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def norm_ref(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    return str(s).strip().upper()


# ---- OSM parsing -----------------------------------------------------------------------------------------------
def parse_elements(elements: list):
    nodes = {e["id"]: (float(e["lat"]), float(e["lon"])) for e in elements if e["type"] == "node"}
    ways = []
    for e in elements:
        if e["type"] != "way":
            continue
        for n in e["nodes"]:
            if n not in nodes:
                raise ValueError(f"way {e['id']} references node {n} that the answer does not contain (truncated pull?)")
        ways.append({"id": e["id"], "nodes": list(e["nodes"]), "tags": dict(e.get("tags") or {})})
    return nodes, ways


def build_graph(nodes: dict, ways: list, kinds=GRAPH_AEROWAYS) -> dict:
    """Undirected adjacency {node: {nbr: metres}} over the routable aeroway ways (one edge per node pair)."""
    adj: dict = {}
    for w in ways:
        if w["tags"].get("aeroway") not in kinds:
            continue
        for a, b in zip(w["nodes"][:-1], w["nodes"][1:]):
            if a == b:
                continue
            d = haversine_m(*nodes[a], *nodes[b])
            adj.setdefault(a, {})[b] = d      # a repeated (a, b) pair rewrites the same haversine length
            adj.setdefault(b, {})[a] = d
    return adj


def _within(pt, ref_point, radius_m) -> bool:
    return ref_point is None or haversine_m(pt[0], pt[1], ref_point[0], ref_point[1]) <= radius_m


def stand_points(nodes: dict, ways: list, elements: list, ref_point=None, radius_m: float = RADIUS_M) -> dict:
    """ref -> (lat, lon, source). Refs normalised; 'A;B' split; duplicates averaged; parking_position (node, then
    lead-in way) beats a gate with the same ref. A lead-in way's stand point is its free end (graph degree 1); if
    both or neither end is free, its centroid. With `ref_point`, candidates beyond `radius_m` are ignored (a
    neighbouring airfield's stands share refs with ours; the bbox holds both)."""
    cand: dict = {}      # (priority, ref) -> list of (lat, lon)
    src_of = {0: "parking_position", 1: "parking_position_way", 2: "gate"}
    for e in elements:
        if e["type"] != "node":
            continue
        t = e.get("tags") or {}
        pr = 0 if t.get("aeroway") == "parking_position" else 2 if t.get("aeroway") == "gate" else None
        if pr is None:
            continue
        pt = (float(e["lat"]), float(e["lon"]))
        if not _within(pt, ref_point, radius_m):
            continue
        for r in norm_ref(t.get("ref")).split(";"):
            if r.strip():
                cand.setdefault((pr, r.strip()), []).append(pt)
    deg: dict = {}
    for w in ways:
        if w["tags"].get("aeroway") in GRAPH_AEROWAYS:
            for a, b in zip(w["nodes"][:-1], w["nodes"][1:]):
                if a != b:
                    deg[a] = deg.get(a, 0) + 1; deg[b] = deg.get(b, 0) + 1
    for w in ways:
        if w["tags"].get("aeroway") != "parking_position":
            continue
        pts = [nodes[n] for n in w["nodes"]]
        free = [n for n in {w["nodes"][0], w["nodes"][-1]} if deg.get(n, 0) == 1]
        c = nodes[free[0]] if len(free) == 1 else (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        if not _within(c, ref_point, radius_m):
            continue
        for r in norm_ref(w["tags"].get("ref")).split(";"):
            if r.strip():
                cand.setdefault((1, r.strip()), []).append(c)
    out: dict = {}
    for (pr, r) in sorted(cand, key=lambda k: (k[1], k[0])):
        if r in out:
            continue                 # a lower-priority source never overrides
        pts = cand[(pr, r)]
        out[r] = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts), src_of[pr])
    return out


def runway_thresholds(nodes: dict, ways: list, ref_point=None, radius_m: float = RADIUS_M, report: dict | None = None) -> dict:
    """designator -> (lat, lon) of the end a departure on that designator rolls FROM. Ways sharing a ref are merged;
    the runway's two ends are its most distant node pair; each designator is assigned to the end whose outbound
    bearing matches 10 x its number within BEARING_TOL_DEG, or the ref is refused. A ref may name one designator
    (EDDF '18', LTFM '09': take-off-only runways) or two. With `ref_point`, runway ways whose centroid lies beyond
    `radius_m` are dropped (reported) — neighbouring airfields inside the bbox. A ref group spanning more than
    MAX_RUNWAY_SPAN_M, or a designator produced twice, is refused: that is two airfields sharing a ref."""
    groups: dict = {}
    dropped: dict = {}
    for w in ways:
        if w["tags"].get("aeroway") != "runway" or not norm_ref(w["tags"].get("ref")):
            continue
        ref = norm_ref(w["tags"]["ref"])
        pts = [nodes[n] for n in w["nodes"]]
        c = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        if not _within(c, ref_point, radius_m):
            dropped[f"{ref}#{w['id']}"] = f"centroid {haversine_m(c[0], c[1], ref_point[0], ref_point[1]):.0f} m from the reference point"
            continue
        groups.setdefault(ref, set()).update(w["nodes"])
    out: dict = {}
    for ref, nid in groups.items():
        desigs = [d.strip() for d in ref.split("/") if d.strip()]
        if not 1 <= len(desigs) <= 2 or any(designator_from_mvt(d) is None for d in desigs):
            dropped[ref] = "ref does not name one or two runway designators (heliport / unnamed pavement)"
            continue
        pts = [nodes[n] for n in sorted(nid)]
        best, e1, e2 = -1.0, None, None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = haversine_m(*pts[i], *pts[j])
                if d > best:
                    best, e1, e2 = d, pts[i], pts[j]
        if best > MAX_RUNWAY_SPAN_M:
            raise ValueError(f"runway ref {ref!r} spans {best:.0f} m > {MAX_RUNWAY_SPAN_M:.0f}: two airfields share this ref")
        b12, b21 = bearing_deg(*e1, *e2), bearing_deg(*e2, *e1)
        for d in desigs:
            num = int(d[:2]) * 10.0
            if angle_diff(b12, num) <= BEARING_TOL_DEG:
                end = e1
            elif angle_diff(b21, num) <= BEARING_TOL_DEG:
                end = e2
            else:
                raise ValueError(f"runway {ref}: designator {d} matches neither end bearing ({b12:.0f}/{b21:.0f})")
            if d in out:
                raise ValueError(f"designator {d} produced by two runway refs (another airfield inside the bbox?)")
            out[d] = end
    if report is not None:
        report["runways_dropped"] = dropped
        report["runway_refs_kept"] = sorted(out and {r for r in groups if r not in dropped})
    return out


def nearest_node(adj: dict, nodes: dict, lat: float, lon: float):
    ids = list(adj)
    la = np.array([nodes[i][0] for i in ids]); lo = np.array([nodes[i][1] for i in ids])
    d = haversine_m(lat, lon, la, lo)
    k = int(np.argmin(d))
    return ids[k], float(d[k])


def dijkstra(adj: dict, source):
    """(distance, hops) from source to every reachable node; unreachable nodes are absent."""
    dist, hops = {source: 0.0}, {source: 0}
    heap = [(0.0, 0, source)]
    done = set()
    while heap:
        d, h, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v, w in adj.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v], hops[v] = nd, h + 1
                heapq.heappush(heap, (nd, h + 1, v))
    return dist, hops


def route_table(icao: str, data: dict, report: dict | None = None, ref_point=None, radius_m: float = RADIUS_M) -> pd.DataFrame:
    """One row per (stand, designator): route_m, n_edges, straight_m, stand_snap_m, rwy_snap_m. `ref_point` defaults
    to APTS[icao]; stands and runways beyond `radius_m` of it are a neighbour's (dropped, reported)."""
    ref_point = ref_point or APTS[icao]
    els = data["elements"]
    nodes, ways = parse_elements(els)
    adj = build_graph(nodes, ways)
    if not adj:
        raise ValueError(f"{icao}: no routable ways")
    stands = stand_points(nodes, ways, els, ref_point, radius_m)
    rw_rep: dict = {}
    thr = runway_thresholds(nodes, ways, ref_point, radius_m, report=rw_rep)
    if not thr:
        raise ValueError(f"{icao}: no runway with a ref")
    thr_node = {d: nearest_node(adj, nodes, *p) for d, p in thr.items()}
    fields = {d: dijkstra(adj, n) for d, (n, _) in thr_node.items()}
    # stands snap only to pavement a runway can reach: a lead-in stub drawn disconnected from the taxiway
    # centrelines (EDDF: 187 of 337 stands) is not a place a departure can taxi from
    reachable = set().union(*(set(f[0]) for f in fields.values()))
    adj_reach = {n: adj[n] for n in reachable}
    rows, dropped, srcs = [], {}, {}
    for s, (la, lo, src) in stands.items():
        srcs[src] = srcs.get(src, 0) + 1
        n, snap = nearest_node(adj_reach, nodes, la, lo)
        if snap > MAX_SNAP_M:
            dropped[s] = f"stand snap {snap:.0f} m > {MAX_SNAP_M:.0f} m"
            continue
        routed = False
        for d, (tn, tsnap) in thr_node.items():
            dist, hops = fields[d]
            if n not in dist:
                continue
            routed = True
            rows.append({"icao": icao, "stand": s, "rwy": d, "route_m": snap + dist[n], "n_edges": hops[n],
                         "straight_m": haversine_m(la, lo, *thr[d]), "stand_snap_m": snap, "rwy_snap_m": tsnap})
        if not routed:
            dropped[s] = "no path to any runway threshold"
    t = pd.DataFrame(rows, columns=["icao", "stand", "rwy", "route_m", "n_edges", "straight_m", "stand_snap_m", "rwy_snap_m"])
    t.attrs["dropped"] = dropped
    if report is not None:
        report.update({
            "n_elements": len(els), "n_graph_nodes": len(adj), "n_graph_nodes_reachable": len(reachable),
            "n_graph_edges": sum(len(v) for v in adj.values()) // 2,
            "n_stands": len(stands), "n_stands_routed": int(t.stand.nunique()), "n_stands_dropped": len(dropped),
            "dropped": dropped, "stand_sources": srcs,
            "n_runway_thresholds": len(thr), "runway_thresholds": {d: {"lat": p[0], "lon": p[1], "snap_m": thr_node[d][1]} for d, p in thr.items()},
            **rw_rep, "ref_point": list(ref_point), "radius_m": radius_m,
            "n_holding_positions": sum(1 for e in els if (e.get("tags") or {}).get("aeroway") == "holding_position"),
            "snap_m_quantiles": {} if t.empty else {q: float(t.drop_duplicates("stand").stand_snap_m.quantile(float(q))) for q in ("0.5", "0.9", "0.99", "1.0")},
            "route_m_quantiles": {} if t.empty else {q: float(t.route_m.quantile(float(q))) for q in ("0.01", "0.5", "0.99")},
        })
    return t


# ---- the ONE attach path for fit and serve ------------------------------------------------------------------------
def designator_from_mvt(v):
    """'25C' -> '25C'; junk (NA, HEL, H, blanks, out-of-range numbers) -> None."""
    r = norm_ref(v)
    if len(r) < 2 or len(r) > 3 or not r[:2].isdigit() or not (1 <= int(r[:2]) <= 36):
        return None
    if len(r) == 3 and r[2] not in "LRC":
        return None
    return r


def attach(frame: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """ORL columns for every row of `frame`, a pure function of (ADEP_mvt, STAND_mvt, RUNWAY_mvt) and the static table.
    Called identically on 2025 fit rows and 2026 serve rows; labels are never read."""
    for c in KEY_COLS:
        if c not in frame.columns:
            raise KeyError(c)
    ap = frame["ADEP_mvt"].map(norm_ref).to_numpy()
    st = frame["STAND_mvt"].map(norm_ref).to_numpy()
    rw = frame["RUNWAY_mvt"].map(designator_from_mvt).to_numpy(dtype=object)
    key_full = pd.Index([f"{a}|{s}|{r}" for a, s, r in zip(ap, st, rw)])
    key_stand = pd.Index([f"{a}|{s}" for a, s in zip(ap, st)])
    R = routes
    idx_full = pd.Index(R.icao + "|" + R.stand + "|" + R.rwy)
    route = pd.Series(R.route_m.to_numpy(), index=idx_full)
    edges = pd.Series(R.n_edges.to_numpy(float), index=idx_full)
    straight = pd.Series(R.straight_m.to_numpy(), index=idx_full)
    med = R.groupby([R.icao + "|" + R.stand]).route_m.median()
    # static rank: share of the airport's route-table values strictly below the row's route
    ranks = {}
    for icao, g in R.groupby("icao"):
        vals = np.sort(g.route_m.to_numpy())
        ranks[icao] = vals
    rk = np.full(len(frame), np.nan)
    rm = route.reindex(key_full).to_numpy()
    for i, (a, v) in enumerate(zip(ap, rm)):
        if np.isfinite(v) and a in ranks:
            rk[i] = np.searchsorted(ranks[a], v, side="left") / len(ranks[a])
    out = pd.DataFrame({
        "orl_route_m": rm,
        "orl_route_med_m": med.reindex(key_stand).to_numpy(),
        "orl_straight_m": straight.reindex(key_full).to_numpy(),
        "orl_n_edges": edges.reindex(key_full).to_numpy(),
        "orl_rank": rk,
    }, index=frame.index)
    out[ORL_FLAG] = np.isfinite(rm).astype(np.int8)
    return out


def nan_share_gate(fit: pd.DataFrame, serve: pd.DataFrame, tol: float = NAN_SHARE_TOL) -> dict:
    """Per airport, per column: the NaN share in fit and in serve. Refuses (AssertionError) a serve NaN share at an
    airport whose fit share is exactly 0 (a NaN mode the learner never saw), or a drift beyond `tol`."""
    fa, sa = set(fit.ADEP_mvt.unique()), set(serve.ADEP_mvt.unique())
    if fa != sa:
        raise AssertionError(f"airports differ between fit {sorted(fa)} and serve {sorted(sa)}")
    rep: dict = {}
    problems = []
    for ap in sorted(fa):
        rep[ap] = {}
        f, s = fit[fit.ADEP_mvt == ap], serve[serve.ADEP_mvt == ap]
        for c in ORL_COLS:
            pf, ps = float(f[c].isna().mean()), float(s[c].isna().mean())
            rep[ap][c] = {"fit": pf, "serve": ps}
            if ps > 0.0 and pf == 0.0:
                problems.append(f"{ap} {c}: serve NaN share {ps:.4f} with fit {pf:.4f} (NaN mode absent in fit)")
            elif abs(ps - pf) > tol:
                problems.append(f"{ap} {c}: drift {abs(ps - pf):.4f} > {tol} (fit {pf:.4f}, serve {ps:.4f})")
    if problems:
        raise AssertionError("fit/serve NaN gate refused: " + "; ".join(problems))
    return rep


# ---- audit ------------------------------------------------------------------------------------------------------
def domain_auc(a: np.ndarray, b: np.ndarray, seed: int = AUDIT_SEED) -> float:
    """AUC of a small gradient-boosted classifier telling rows of `a` from rows of `b` (held-out 30 %); NaN-tolerant."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X = np.vstack([np.asarray(a, float), np.asarray(b, float)])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y)); X, y = X[perm], y[perm]
    n_tr = int(0.7 * len(y))
    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_leaf_nodes=15, random_state=seed)
    clf.fit(X[:n_tr], y[:n_tr])
    return float(roc_auc_score(y[n_tr:], clf.predict_proba(X[n_tr:])[:, 1]))


def column_summary(frame: pd.DataFrame) -> dict:
    out: dict = {}
    for c in ORL_COLS + [ORL_FLAG]:
        out[c] = {}
        for ap, g in frame.groupby("ADEP_mvt"):
            v = g[c].astype(float)
            out[c][ap] = {"n": int(len(v)), "null_share": float(v.isna().mean()), "mean": float(v.mean()),
                          **{f"q{int(q * 100):02d}": float(v.quantile(q)) for q in (0.01, 0.10, 0.50, 0.90, 0.99)}}
    return out


def match_rates(frame: pd.DataFrame, attached: pd.DataFrame) -> dict:
    out = {}
    for ap in sorted(frame.ADEP_mvt.unique()):
        m = (frame.ADEP_mvt == ap).to_numpy()
        a = attached[m]
        rw = frame.loc[m, "RUNWAY_mvt"].map(designator_from_mvt)
        out[ap] = {"n": int(m.sum()), "stand_match": float(a.orl_route_med_m.notna().mean()),
                   "rwy_mapped": float(rw.notna().mean()), "both": float(a.orl_matched.mean())}
    return out


def unseen_key_share(fit: pd.DataFrame, serve: pd.DataFrame) -> dict:
    """Per airport: the share of serve rows whose normalised (stand, runway) pair, and whose stand, has ZERO fit rows
    (INV section 6.3's label-free tripwire: a tree has no evidence on those populations)."""
    def keys(f):
        ap = f.ADEP_mvt.map(norm_ref); st = f.STAND_mvt.map(norm_ref); rw = f.RUNWAY_mvt.map(lambda v: designator_from_mvt(v) or "")
        return ap, ap + "|" + st + "|" + rw, ap + "|" + st
    fa, fpair, fstand = keys(fit); sa, spair, sstand = keys(serve)
    seen_pair, seen_stand = set(fpair), set(fstand)
    out = {}
    for ap in sorted(sa.unique()):
        m = (sa == ap).to_numpy()
        out[ap] = {"n": int(m.sum()), "pair_unseen": float((~spair[m].isin(seen_pair)).mean()),
                   "stand_unseen": float((~sstand[m].isin(seen_stand)).mean())}
    return out


def runway_mapping(frame: pd.DataFrame, routes: pd.DataFrame) -> dict:
    known = {ic: set(g.rwy) for ic, g in routes.groupby("icao")}
    out: dict = {}
    for (ap, rw), n in frame.groupby(["ADEP_mvt", "RUNWAY_mvt"], dropna=False).size().items():
        d = designator_from_mvt(rw)
        out.setdefault(ap, {})[str(rw)] = {"n": int(n), "mapped": bool(d is not None and d in known.get(ap, set()))}
    return out


# ---- stages -----------------------------------------------------------------------------------------------------
def run_fetch(apts=APTS, cache_dir: pathlib.Path = CACHE) -> None:
    for icao in apts:
        p = fetch_airport(icao, cache_dir=cache_dir)
        print(f"{icao}: {p} ({p.stat().st_size:,} B)", flush=True)


def run_build(cache_dir: pathlib.Path = CACHE, apts=None, out_path: pathlib.Path = ROUTES,
              report_path: pathlib.Path = REPORT_BUILD, radius_m: float = RADIUS_M) -> dict:
    tables, report = [], {}
    for icao in (apts or list(APTS)):
        rep: dict = {}
        t = route_table(icao, load_airport(icao, cache_dir), report=rep, radius_m=radius_m)
        report[icao] = rep
        tables.append(t)
        print(f"{icao}: stands {rep['n_stands']} routed {rep['n_stands_routed']} thresholds {rep['n_runway_thresholds']} "
              f"graph {rep['n_graph_nodes']}/{rep['n_graph_edges']}", flush=True)
    all_t = pd.concat(tables, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_t.to_parquet(out_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    return report


def _read_keys(path: pathlib.Path, serve: bool) -> pd.DataFrame:
    """The row filters of stand_ab.build_month (scripts/stand_ab.py:232-237): fit needs the label present, serve does not.
    Only the three join keys and the clock leave this function; TAXITIME/BLOCK are read for the FILTER only."""
    import pyarrow.parquet as pq
    cols = ["PHASE_mvt", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt"]
    if not serve:
        cols += ["TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"]
    t = pq.read_table(path, columns=cols).to_pandas()
    m = (t.PHASE_mvt == "DEP") & t.AOBT_3_flt.notna() & t.MVT_TIME_UTC_mvt.notna() & t.SCHED_TIME_UTC_mvt.notna()
    if not serve:
        m &= t.TAXITIME_SEC_mvt.notna() & t.BLOCK_TIME_UTC_mvt.notna() & (t.TAXITIME_SEC_mvt > 0)
    d = t[m & t.ADEP_mvt.isin(APTS)]
    mvt = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True)
    return pd.DataFrame({"ADEP_mvt": d.ADEP_mvt.astype(str).to_numpy(), "STAND_mvt": d.STAND_mvt.to_numpy(),
                         "RUNWAY_mvt": d.RUNWAY_mvt.to_numpy(), "doy": mvt.dt.dayofyear.to_numpy(),
                         "dow": mvt.dt.dayofweek.to_numpy(), "month": mvt.dt.month.to_numpy()})


def run_audit(routes_path: pathlib.Path = ROUTES, raw: pathlib.Path = RAW, report_path: pathlib.Path = REPORT_AUDIT,
              n_side: int = AUDIT_N_PER_SIDE, seed: int = AUDIT_SEED) -> dict:
    routes = pd.read_parquet(routes_path)
    fit = pd.concat([_read_keys(p, serve=False) for p in sorted(raw.glob("training_2025-*.parquet"))], ignore_index=True)
    serve = _read_keys(raw / "ranking.parquet", serve=True)
    fa, sa = attach(fit, routes), attach(serve, routes)
    fit_all, serve_all = pd.concat([fit, fa], axis=1), pd.concat([serve, sa], axis=1)
    rep = {"n_fit": int(len(fit)), "n_serve": int(len(serve)),
           "match_fit": match_rates(fit, fa), "match_serve": match_rates(serve, sa),
           "runway_mapping_serve": runway_mapping(serve, routes), "runway_mapping_fit": runway_mapping(fit, routes),
           "unseen_key_share_serve": unseen_key_share(fit, serve)}
    try:
        rep["nan_gate"] = nan_share_gate(fit_all, serve_all)
        rep["nan_gate_verdict"] = "PASS"
    except AssertionError as e:
        rep["nan_gate_verdict"] = f"REFUSED: {e}"
    rep["summary_fit"] = column_summary(fit_all)
    rep["summary_serve"] = column_summary(serve_all)
    # domain AUC: season-matched 2025 Jan+Jul vs 2026, ORL columns vs the doy/dow control, plus a within-2025 null
    rng = np.random.default_rng(seed)
    f25 = fit_all[fit_all.month.isin([1, 7])]
    ia = rng.choice(len(f25), min(n_side, len(f25)), replace=False)
    ib = rng.choice(len(serve_all), min(n_side, len(serve_all)), replace=False)
    A, B = f25.iloc[ia], serve_all.iloc[ib]
    odd = A[A.doy % 2 == 1]; even = A[A.doy % 2 == 0]
    rep["domain_auc"] = {
        "orl_cols_2025JanJul_vs_2026": domain_auc(A[ORL_COLS].to_numpy(), B[ORL_COLS].to_numpy(), seed),
        "doy_dow_control_2025JanJul_vs_2026": domain_auc(A[["doy", "dow"]].to_numpy(), B[["doy", "dow"]].to_numpy(), seed),
        "orl_cols_null_2025_odd_vs_even_day": domain_auc(odd[ORL_COLS].to_numpy(), even[ORL_COLS].to_numpy(), seed),
        "n_per_side": int(min(n_side, len(f25), len(serve_all))),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str))
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fetch", "build", "audit"])
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        run_fetch()
    elif a.cmd == "build":
        run_build()
    else:
        r = run_audit()
        print(json.dumps({k: r[k] for k in ("n_fit", "n_serve", "match_serve", "nan_gate_verdict", "domain_auc", "unseen_key_share_serve")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
