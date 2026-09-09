"""Fetch one adsb.lol daily archive and extract ground tracks for the ten scored airports.

    python3.11 scripts/adsb_extract.py --day 2025-01-09            # fetch, extract, delete raw
    python3.11 scripts/adsb_extract.py --day 2025-01-09 --probe    # first 60 MB only, no delete

Data source: ADSB.lol globe_history, ODbL 1.0 (see plans/PREREG §Amendment 4.9 and 6.7).
Attribution and the extraction procedure are part of the repository; the archives themselves
are never redistributed.

Disk discipline (Amendment 6.6): the two release parts are streamed to data/adsb_raw/, parsed
without ever untarring to disk, and deleted before the next day is fetched. Peak disk ~3.5 GiB
on a volume that is 93% full.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import pathlib
import subprocess
import sys
import tarfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAWDIR = ROOT / "data" / "adsb_raw"
OUTDIR = ROOT / "data" / "adsb"
BASE = "https://github.com/adsblol/globe_history_{yr}/releases/download/{tag}/{tag}.tar.{part}"

#: airport reference points; box is +-0.15 deg (~16 km N-S), enough for apron and runways
APTS = {
    "EDDF": (50.0333, 8.5706), "EDDM": (48.3538, 11.7861), "EGLL": (51.4706, -0.4619),
    "EHAM": (52.3086, 4.7639), "LEBL": (41.2971, 2.0785),  "LEMD": (40.4719, -3.5626),
    "LFPG": (49.0097, 2.5479), "LIRF": (41.8003, 12.2389), "LSZH": (47.4647, 8.5492),
    "LTFM": (41.2753, 28.7519),
}
PAD = 0.15


def which_airport(lat, lon):
    for ap, (a, o) in APTS.items():
        if abs(lat - a) <= PAD and abs(lon - o) <= PAD:
            return ap
    return None


def parse_member(raw: bytes):
    """One trace_full_*.json -> rows inside any airport box. Tolerant of shape drift."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    try:
        d = json.loads(raw)
    except Exception:
        return []
    base = d.get("timestamp")
    tr = d.get("trace")
    if base is None or not tr:
        return []
    icao, reg, typ = d.get("icao"), d.get("r"), d.get("t")
    out, callsign = [], None
    for p in tr:
        if len(p) < 5:
            continue
        lat, lon = p[1], p[2]
        if lat is None or lon is None:
            continue
        ap = which_airport(lat, lon)
        if ap is None:
            continue
        # the per-point aircraft dict carries the callsign when it changes
        if len(p) > 8 and isinstance(p[8], dict):
            f = p[8].get("flight")
            if f:
                callsign = f.strip()
        alt = p[3]
        out.append((icao, reg, typ, callsign, ap, base + p[0], lat, lon,
                    1 if (alt == "ground" or alt is None) else 0,
                    np.nan if not isinstance(alt, (int, float)) else float(alt),
                    float(p[4]) if isinstance(p[4], (int, float)) else np.nan))
    return out


def fetch(day: str, probe: bool) -> list[pathlib.Path]:
    yr = day[:4]
    tag = f"v{day.replace('-', '.')}-planes-readsb-prod-0"
    RAWDIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for part in ("aa", "ab"):
        url = BASE.format(yr=yr, tag=tag, part=part)
        dst = RAWDIR / f"{tag}.tar.{part}"
        cmd = ["curl", "-sfL", "--retry", "3", "-o", str(dst)]
        if probe:
            cmd += ["-r", "0-62914560"]
        cmd.append(url)
        if subprocess.run(cmd).returncode != 0:
            if part == "ab":
                break                      # some days are a single part
            raise SystemExit(f"download failed: {url}")
        paths.append(dst)
        if probe:
            break
    return paths


def extract(paths, probe: bool) -> pd.DataFrame:
    cat = subprocess.Popen(["cat", *map(str, paths)], stdout=subprocess.PIPE)
    rows, n_members, diagnosed = [], 0, False
    try:
        with tarfile.open(fileobj=cat.stdout, mode="r|") as tf:
            for m in tf:
                if not m.isfile() or "trace" not in m.name:
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                blob = f.read()
                if not diagnosed:
                    # surface a parse failure in the first seconds, not after 3 GiB
                    diagnosed = True
                    b = gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob
                    try:
                        j = json.loads(b)
                        print(f"    first trace member {m.name}: keys={list(j)[:8]} "
                              f"points={len(j.get('trace', []))}", flush=True)
                    except Exception as e:
                        raise SystemExit(f"trace member is not parseable JSON: {e!r}")
                rows.extend(parse_member(blob))
                n_members += 1
                if n_members % 20000 == 0:
                    print(f"    {n_members:,} traces, {len(rows):,} in-box samples",
                          flush=True)
    except (tarfile.ReadError, EOFError) as e:
        if not probe:
            raise
        print(f"    probe truncated as expected: {e}")
    finally:
        cat.stdout.close()
        cat.wait()
    print(f"    parsed {n_members:,} trace files -> {len(rows):,} in-box samples")
    if not probe and n_members == 0:
        raise SystemExit("no trace members found - archive layout changed?")
    if not probe and not rows:
        raise SystemExit("trace members parsed but zero in-box samples - bbox or parser wrong")
    return pd.DataFrame(rows, columns=["icao", "reg", "type", "callsign", "airport",
                                       "t", "lat", "lon", "on_ground", "alt", "gs"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--day", required=True, help="YYYY-MM-DD")
    ap.add_argument("--probe", action="store_true", help="first 60 MB only; keeps the part file")
    a = ap.parse_args()

    print(f"[{a.day}] fetching{' (probe)' if a.probe else ''} ...", flush=True)
    paths = fetch(a.day, a.probe)
    print(f"[{a.day}] {sum(p.stat().st_size for p in paths)/2**30:.2f} GiB on disk; extracting",
          flush=True)
    df = extract(paths, a.probe)
    if not a.probe:
        for p in paths:
            p.unlink()                      # Amendment 6.6: delete before the next day
        OUTDIR.mkdir(parents=True, exist_ok=True)
        out = OUTDIR / f"adsb_{a.day}.parquet"
        df.to_parquet(out, index=False)
        print(f"[{a.day}] wrote {out.name}: {len(df):,} rows")
        print(df.airport.value_counts().to_string())
    else:
        print(df.airport.value_counts().to_string() if len(df) else "  (no in-box samples yet)")
        for p in paths:
            p.unlink()


if __name__ == "__main__":
    main()
