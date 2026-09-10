#!/usr/bin/env python3.11
"""Archive the surface weather the Amendment 24 block reads, once, before any run uses it.

Source: the Iowa Environmental Mesonet ASOS/METAR archive
(https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py), **public domain** — the service
states no licence restriction and none is claimed here. Routine (MTR) and special (SPECI) reports
for the ten scored airports, one CSV per (airport, month), written to data/weather/ and NEVER
re-fetched by a run: the 2024 winner was bitten when upstream data was back-corrected mid-
competition, so the archive is frozen on disk and every later build reads the frozen copy.

    python3.11 scripts/fetch_weather.py            # every month the caches use + both scored months
    python3.11 scripts/fetch_weather.py --smoke    # one airport, one month, to a scratch dir

Each file is written atomically (a .part rename) so an interrupted fetch cannot leave a truncated
CSV that a later build would silently read as complete. A month already on disk is skipped, so the
script is safe to re-run.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEATHER = ROOT / "data" / "weather"
BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
#: the ten scored airports; the archive keys on the ICAO identifier, as the movement table does
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
#: the twelve training months plus the two scored months of 2026
MONTHS = [(2025, m) for m in range(1, 13)] + [(2026, 1), (2026, 7)]
#: what the block reads: the raw fields only - every derived flag is computed in stand_ab, so a
#: change to the derivation never requires a re-fetch
FIELDS = "tmpf,dwpf,sknt,gust,vsby,p01i,wxcodes,metar"
TIMEOUT_S = 120
RETRIES = 4
BACKOFF_S = 5.0


def month_url(station: str, year: int, month: int) -> str:
    """The archive query for one (station, month): the whole month in UTC, both report types."""
    y2, m2 = (year + 1, 1) if month == 12 else (year, month + 1)
    q = [("station", station), ("data", FIELDS), ("year1", year), ("month1", month), ("day1", 1),
         ("year2", y2), ("month2", m2), ("day2", 1), ("tz", "UTC"), ("format", "onlycomma"),
         ("latlon", "no"), ("elev", "no"), ("missing", "M"), ("trace", "T"), ("direct", "no"),
         ("report_type", 3), ("report_type", 4)]
    return BASE + "?" + urllib.parse.urlencode(q)


def fetch(url: str, retries: int = RETRIES) -> str:
    """GET with a bounded retry; a transient 5xx or timeout must not lose a month."""
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, OSError) as e:   # noqa: PERF203 - the retry IS the point
            last = e
            if attempt < retries - 1:
                time.sleep(BACKOFF_S * (attempt + 1))
    raise RuntimeError(f"fetch failed after {retries} attempts: {last}") from last


def check(text: str, station: str, year: int, month: int) -> int:
    """The rows a month must have to count as complete: a header, the station on every data row,
    and enough observations that a truncated response cannot pass (24 h x 28 days = 672 at one an
    hour; the floor is deliberately half that, since a small airport can miss reports)."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines or not lines[0].startswith("station,valid,"):
        raise ValueError(f"{station} {year}-{month:02d}: no header, got {lines[0][:80] if lines else '<empty>'!r}")
    rows = lines[1:]
    bad = [ln for ln in rows[:50] if not ln.startswith(f"{station},")]
    if bad:
        raise ValueError(f"{station} {year}-{month:02d}: a row is not this station: {bad[0][:80]!r}")
    if len(rows) < 336:
        raise ValueError(f"{station} {year}-{month:02d}: only {len(rows)} observations, expected >= 336 "
                         "(a truncated or empty response)")
    return len(rows)


def name(station: str, year: int, month: int) -> str:
    return f"{station}_{year}-{month:02d}.csv"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="one airport, one month, into data/weather_smoke/")
    ap.add_argument("--out-dir", default=None, help="override the destination (tests)")
    ap.add_argument("--refetch", action="store_true", help="fetch months already on disk again (default: skip)")
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out_dir) if a.out_dir else (ROOT / "data" / "weather_smoke" if a.smoke else WEATHER)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [("EHAM", 2025, 1)] if a.smoke else [(s, y, m) for s in APTS for (y, m) in MONTHS]
    got = skipped = 0
    t0 = time.time()
    for station, year, month in jobs:
        dest = out / name(station, year, month)
        if dest.exists() and not a.refetch:
            skipped += 1
            continue
        text = fetch(month_url(station, year, month))
        n = check(text, station, year, month)
        part = dest.with_suffix(".part")
        part.write_text(text)
        part.rename(dest)                      # atomic: a truncated file never takes the real name
        got += 1
        print(f"{dest.name}  {n:6,} observations  {len(text)/1e6:5.2f} MB  [{time.time() - t0:5.0f}s]", flush=True)
    print(f"done: {got} fetched, {skipped} already on disk, {len(jobs)} asked; -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
