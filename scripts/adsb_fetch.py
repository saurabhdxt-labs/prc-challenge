"""Fetch adsb.lol daily archives and extract ground-level samples at the ten scored airports (v2 parser).

    python3.11 scripts/adsb_fetch.py run                  # the 62 days of January + July 2025, resumable
    python3.11 scripts/adsb_fetch.py run --limit 1        # one day (smoke)

Data source: ADSB.lol globe_history (https://www.adsb.lol/docs/open-data/historical/), ODbL 1.0.
The archives are never redistributed; only per-airport samples derived from them are kept under data/.

Why a v2 parser (scripts/adsb_extract.py is left untouched):
  * the callsign is tracked across EVERY point of a trace, not only in-box points, so a callsign that was set
    before the aircraft entered an airport box is not lost;
  * a missing altitude is NOT counted as on-ground (`alt_missing` flags it instead);
  * the ground track and the position source (adsb_icao / mlat / ...) are kept.

Disk discipline: each day's release parts are streamed to data/adsb_raw_v2/, parsed without untarring, and
deleted before the next day; the day file is written atomically; a run stops below MIN_FREE_GIB free.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import importlib.util
import json
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time
import zlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAWDIR = ROOT / "data" / "adsb_raw_v2"
OUTDIR = ROOT / "data" / "adsb" / "v2"

_spec = importlib.util.spec_from_file_location("adsb_extract", ROOT / "scripts" / "adsb_extract.py")
ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae)

COLUMNS = ["icao", "reg", "type", "callsign", "airport", "t", "lat", "lon",
           "on_ground", "alt_missing", "alt", "gs", "track", "src"]
MIN_FREE_GIB = 20.0
MAX_CONSECUTIVE_FAILURES = 3
SLOW_DAY_S = 15 * 60
#: replica preference for one day's archive: prod first (Jan/Jul were all built from prod-0), then the same day's replicas
REPLICAS = ("prod-0", "prod-0tmp", "staging-0", "staging-0tmp")
SPENT_DAYS = ("2025-01-09",)          # already read by earlier ADS-B work: fetched last, never used for a claim
SOURCE = "adsb.lol globe_history (https://www.adsb.lol/docs/open-data/historical/)"
LICENCE = "ODbL 1.0 (Open Database License); attribution: ADSB.lol contributors"


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else np.nan


def parse_member_v2(raw: bytes, unparseable=None) -> list:
    """One trace_full_*.json -> rows (COLUMNS order) for points inside any airport box. A member whose gzip or JSON is
    corrupt yields no rows and, when `unparseable` (a list) is given, is counted there so the day record can report it
    (adsb_extract.parse_member does the same)."""
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError, zlib.error):
            if unparseable is not None:
                unparseable.append(1)
            return []
    try:
        d = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        if unparseable is not None:
            unparseable.append(1)
        return []
    if not isinstance(d, dict):
        return []
    base, tr = d.get("timestamp"), d.get("trace")
    if base is None or not tr:
        return []
    icao, reg, typ = d.get("icao"), d.get("r"), d.get("t")
    out, callsign = [], None
    for p in tr:
        if not isinstance(p, list) or len(p) < 5:
            continue
        if len(p) > 8 and isinstance(p[8], dict):          # callsign updates are read on EVERY point
            f = p[8].get("flight")
            if f and f.strip():
                callsign = f.strip()
        lat, lon = p[1], p[2]
        if lat is None or lon is None:
            continue
        ap = ae.which_airport(lat, lon)
        if ap is None:
            continue
        alt = p[3]
        out.append((icao, reg, typ, callsign, ap, base + p[0], lat, lon,
                    1 if alt == "ground" else 0,
                    1 if alt is None else 0,
                    _num(alt),
                    _num(p[4]),
                    _num(p[5]) if len(p) > 5 else np.nan,
                    p[9] if len(p) > 9 and isinstance(p[9], str) else None))
    return out


def day_list(year: int = 2025) -> list[str]:
    """January and July of `year`, interleaved (a stop mid-run leaves both months represented); spent days last."""
    spent = [d for d in SPENT_DAYS if d.startswith(str(year))]
    jan = [f"{year}-01-{d:02d}" for d in range(1, 32) if f"{year}-01-{d:02d}" not in spent]
    jul = [f"{year}-07-{d:02d}" for d in range(1, 32) if f"{year}-07-{d:02d}" not in spent]
    out = []
    for i in range(max(len(jan), len(jul))):
        out += jan[i:i + 1] + jul[i:i + 1]
    return out + spent


def day_list_months(year: int, months) -> list[str]:
    """Every day of the named months of `year`, in date order (the extra-months fetch; no spent-day handling)."""
    out = []
    for m in sorted(set(months)):
        if not 1 <= int(m) <= 12:
            raise ValueError(f"month must be 1..12, got {m!r}")
        d = dt.date(year, int(m), 1)
        while d.month == int(m):
            out.append(d.isoformat())
            d += dt.timedelta(days=1)
    return out


def out_dir_for(year: int, months=None) -> pathlib.Path:
    """2025 (the evaluation year) keeps the original directory; any other year gets its own, with its own MANIFEST.
    An extra-months run (`months` given) writes to OUTDIR/y<year>_other: never into the Jan/Jul directory, whose
    builders glob adsb_*.parquet and whose MANIFEST.json records how Jan/Jul were fetched."""
    if months is not None:
        return OUTDIR / f"y{year}_other"
    return OUTDIR if year == 2025 else OUTDIR / f"y{year}"


GITHUB_PER_PAGE = 100
GITHUB_MAX_PAGES = 10              # GitHub lists at most the first 1,000 results (HTTP 422 beyond)
GITHUB_API = "https://api.github.com/repos/adsblol/globe_history_{yr}/releases?per_page=100&page={page}"


def index_path(year: str) -> pathlib.Path:
    return OUTDIR / f"RELEASES_{year}.json"


def build_index(year: str, get=None) -> dict:
    """{tag: [{name, size, url}, ...]} for every release of globe_history_<year>, from the GitHub API, paged
    until an empty page. Written to index_path(year). The asset list is authoritative: days are published either
    as '.tar.aa' + '.tar.ab' or as one '.tar', and the published byte sizes verify each download."""
    if get is None:
        def get(url):
            r = subprocess.run(["curl", "-sfL", url], capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"GitHub API request failed (curl {r.returncode}): {url}")
            return json.loads(r.stdout)
    idx, page = {}, 1
    while True:
        if page > GITHUB_MAX_PAGES:
            raise RuntimeError(f"globe_history_{year} has more than 1,000 releases: GitHub's listing cap would "
                               f"silently drop the oldest ones")
        rel = get(GITHUB_API.format(yr=year, page=page))
        if not isinstance(rel, list):
            raise RuntimeError(f"GitHub API returned {type(rel).__name__}, not a release list: {str(rel)[:200]}")
        for r in rel:
            idx[r["tag_name"]] = [{"name": a["name"], "size": int(a["size"]), "url": a["browser_download_url"]}
                                  for a in r.get("assets", [])]
        if len(rel) < GITHUB_PER_PAGE:                      # a short page is the last one; the next would be a 422
            break
        page += 1
    OUTDIR.mkdir(parents=True, exist_ok=True)
    index_path(year).write_text(json.dumps(idx, indent=0, sort_keys=True))
    return idx


def _load_index(year: str) -> dict:
    p = index_path(year)
    if not p.exists():
        raise RuntimeError(f"no release index {p}; run `adsb_fetch.py index --year {year}` first")
    return json.loads(p.read_text())


def download_day(day: str, index: dict | None = None, raw_dir: pathlib.Path = RAWDIR,
                 runner=subprocess.run) -> list[pathlib.Path]:
    """Every asset of the day's release to raw_dir, in name order, each verified against its published size.
    The release is the first of REPLICAS present in the index: prod-0, else a replica of the same day (some 2025
    days are published only under -prod-0tmp / -staging-0 / -staging-0tmp; PREFERRED_RELEASES.txt in the repo).
    mlatonly-0 is a partial feed, never a replica. On any failure, removes everything it wrote and raises."""
    tags = [f"v{day.replace('-', '.')}-planes-readsb-{r}" for r in REPLICAS]
    if index is None:
        idx = _load_index(day[:4])
        if not any(idx.get(t) for t in tags) and index_path(str(int(day[:4]) + 1)).exists():
            idx = _load_index(str(int(day[:4]) + 1))       # year-end days are filed under the next year's repo
    else:
        idx = index
    tag = next((t for t in tags if idx.get(t)), None)
    if tag is None:
        raise RuntimeError(f"no release of {day} in the index (tried {', '.join(REPLICAS)})")
    assets = sorted(idx[tag], key=lambda a: a["name"])
    raw_dir = pathlib.Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for a in assets:
            dst = raw_dir / a["name"]
            paths.append(dst)
            r = runner(["curl", "-sfL", "--retry", "3", "-o", str(dst), a["url"]])
            if r.returncode != 0:
                raise RuntimeError(f"download failed (curl {r.returncode}): {a['url']}")
            got = dst.stat().st_size
            if got != a["size"]:
                raise RuntimeError(f"size mismatch on {a['name']}: got {got:,} bytes, published {a['size']:,}")
    except BaseException:
        for p in paths:
            p.unlink(missing_ok=True)
        raise
    return paths


def extract_day(paths, day: str, unparseable=None) -> pd.DataFrame:
    """Rows of every trace member in the day's archives; corrupt members are counted in `unparseable` (a list)."""
    cat = subprocess.Popen(["cat", *map(str, paths)], stdout=subprocess.PIPE)
    rows, n = [], 0
    try:
        with tarfile.open(fileobj=cat.stdout, mode="r|") as tf:
            for m in tf:
                if not m.isfile() or "trace" not in m.name:
                    continue
                f = tf.extractfile(m)
                if f is not None:
                    rows.extend(parse_member_v2(f.read(), unparseable))
                    n += 1
    finally:
        cat.stdout.close()
        cat.wait()
    if n == 0:
        raise RuntimeError(f"{day}: no trace members in the archive (layout changed?)")
    return pd.DataFrame(rows, columns=COLUMNS)


def _free_gib() -> float:
    return shutil.disk_usage(ROOT).free / 2**30


def _now() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def run_days(days, out_dir: pathlib.Path = OUTDIR, fetch=download_day, extract=extract_day, log=print,
             free_gib=_free_gib, min_free_gib: float = MIN_FREE_GIB,
             max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES) -> dict:
    """Fetch + extract each day into out_dir/adsb_<day>.parquet. Resumable; per-day failures do not stop the run."""
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "MANIFEST.json"
    prev = {}
    if mpath.exists():
        for d in json.loads(mpath.read_text()).get("days", []):
            # the ORIGINAL ok record travels through any number of resumes (a day skipped last run carries its
            # record in `previous`; reading only status == "ok" dropped it on the second resume: post-mortem 2026-09-26)
            rec = d if d.get("status") == "ok" else (d.get("previous") if d.get("status") == "skipped" else None)
            if rec and rec.get("status") == "ok":
                prev[d["day"]] = rec
    man = {"source": SOURCE, "licence": LICENCE, "parser": "adsb_fetch.parse_member_v2", "started": _now(),
           "days": [], "stopped": None}

    def save():
        mpath.write_text(json.dumps(man, indent=1, default=str))

    fails = 0
    for day in days:
        dest = out_dir / f"adsb_{day}.parquet"
        if dest.exists():
            man["days"].append({"day": day, "status": "skipped", "previous": prev.get(day)})
            save()
            continue
        free = free_gib()
        if free < min_free_gib:
            man["stopped"] = f"free disk {free:.1f} GiB < {min_free_gib:.0f} GiB"
            log(f"[{_now()}] STOP: {man['stopped']}")
            break
        t0, rec, paths = time.time(), {"day": day}, []
        try:
            paths = fetch(day)
            rec["bytes"] = int(sum(p.stat().st_size for p in paths))
            rec["tag"] = pathlib.Path(paths[0]).name.split(".tar")[0] if paths else None
            unparseable = []
            df = extract(paths, day, unparseable=unparseable)
            if len(df) == 0:
                raise RuntimeError("zero in-box rows")
            tmp = out_dir / f".adsb_{day}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            tmp.rename(dest)
            rec.update(status="ok", rows=int(len(df)), unparseable_members=len(unparseable),
                       per_airport={k: int(v) for k, v in df.airport.value_counts().items()},
                       on_ground_share=float(df.on_ground.mean()), with_callsign=float(df.callsign.notna().mean()))
            fails = 0
        except (Exception, SystemExit) as e:                # one bad day never kills the batch
            rec.update(status="failed", error=f"{type(e).__name__}: {e}")
            fails += 1
        finally:
            for p in paths:
                pathlib.Path(p).unlink(missing_ok=True)
        rec["wall_s"] = round(time.time() - t0, 1)
        man["days"].append(rec)
        save()
        slow = "  SLOW (> 15 min)" if rec["wall_s"] > SLOW_DAY_S else ""
        log(f"[{_now()}] {day}: {rec['status']} rows={rec.get('rows', 0):,} wall={rec['wall_s']}s "
            f"free={free:.0f}GiB {rec.get('error', '')}{slow}")
        if fails >= max_consecutive_failures:
            man["stopped"] = f"{max_consecutive_failures} consecutive failures"
            log(f"[{_now()}] STOP: {man['stopped']}")
            break
    man["finished"] = _now()
    save()
    return man


def exit_code(man: dict) -> int:
    """0 only when every day is ok or skipped; 3 when the run stopped early; 4 when any day failed."""
    if man.get("stopped"):
        return 3
    return 4 if any(d["status"] == "failed" for d in man["days"]) else 0


def year_type(value: str) -> int:
    """--year for both `index` and `run`: any integer >= 2025 (no ADS-B source predates the evaluation year).
    Raises argparse.ArgumentTypeError naming the refused value, so both subcommands refuse the same way."""
    year = int(value)
    if year < 2025:
        raise argparse.ArgumentTypeError(f"--year must be >= 2025 (no ADS-B source predates it), got {year}")
    return year


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("index", help="build data/adsb/v2/RELEASES_<year>.json from the GitHub API")
    i.add_argument("--year", required=True, type=year_type)
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, default=None, help="only the first N days of the list")
    r.add_argument("--year", type=year_type, default=2025)
    r.add_argument("--months", type=lambda t: tuple(int(x) for x in t.split(",")), default=None,
                   help="extra months, e.g. 2,3,4,5,6,8,9,10,11,12 -> data/adsb/v2/y<year>_other")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "index":
        idx = build_index(a.year)
        print(f"[{_now()}] {len(idx)} releases -> {index_path(a.year)}", flush=True)
        return
    full = day_list_months(a.year, a.months) if a.months else day_list(a.year)
    days = full[: a.limit] if a.limit else full
    out = out_dir_for(a.year, a.months)
    print(f"[{_now()}] {len(days)} days -> {out}", flush=True)
    man = run_days(days, out_dir=out, log=lambda *x: print(*x, flush=True))
    ok = sum(d["status"] == "ok" for d in man["days"])
    print(f"[{_now()}] finished: {ok} ok, {sum(d['status'] == 'failed' for d in man['days'])} failed, "
          f"stopped={man['stopped']}", flush=True)
    sys.exit(exit_code(man))


if __name__ == "__main__":
    main()
