"""AFI (plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md) — the owner's adsb.fi July-2026 capture into FM1's serve table.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/afi_build.py days      # union day files + MANIFEST.json
    /opt/homebrew/bin/python3.11 -B scripts/analysis/afi_build.py store     # a copy of the R12 store without July 2026
    (then r12_build features --sources <union dir> --days <July 2026> and r12_build tables, through r12_launch)
    /opt/homebrew/bin/python3.11 -B scripts/analysis/afi_build.py check --old <tables> --new <tables> --out <json>

days   July 2026: every adsb.lol sample + each adsb.fi sample >= 1 s from the same aircraft's adsb.lol samples
       (scripts/analysis/adsb_union.py, rule AUN.1), written with a MANIFEST.json declaring both sources.
store  hard links of every record of the base store except the July-2026 days, so only those are rebuilt (same feature
       code and stamp) from the union days.
check  the registered label-free acceptance: (1) train.parquet and the January-2026 serve rows equal the old tables';
       (2) July-2026 joined share never falls, per airport (flights joined before and not after are reported); (3) on flights
       newly joined, the share with |dt_takeoff| > 120 s is <= 0.01 per airport (a misjoin shows there); and the new
       tables declare adsb.fi. Reads no label.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
LOL_DIR = ROOT / "data" / "adsb" / "v2" / "y2026"
FI_DIR = ROOT / "data" / "adsb" / "fi" / "y2026"
UNION_DIR = ROOT / "data" / "adsb" / "union" / "y2026_r12"
BASE_STORE = ROOT / "data" / "r12" / "features_51348d3e1cb3"
AFI_STORE = ROOT / "data" / "r12" / "features_51348d3e1cb3_afi"
JULY = [f"2026-07-{d:02d}" for d in range(1, 32)]
#: AFI.1: adsb.fi samples left out of the union (LTFM: never trained with ADS-B; the join-quality gate failed there)
FI_EXCLUDE_AIRPORTS = ("LTFM",)
MISJOIN_S = 120.0
MISJOIN_MAX_SHARE = 0.01
MIN_ROWS = 50                     # airports with fewer newly joined flights are reported, not judged
SOURCE = ("adsb.lol globe_history (ODbL 1.0) + adsb.fi opendata, the owner's own 60 s poll archive of July 2026 "
          "(non-commercial terms; never redistributed; cannot be re-fetched by others)")
LICENCE = "adsb.lol: ODbL 1.0, attribution ADSB.lol contributors; adsb.fi: opendata.adsb.fi non-commercial terms"


def _union():
    spec = importlib.util.spec_from_file_location("adsb_union", ROOT / "scripts" / "analysis" / "adsb_union.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def link_store(base, new, month: str) -> int:
    """Hard-link every file of `base` into the new directory `new` except the day records of `month` (YYYY-MM)."""
    base, new = pathlib.Path(base), pathlib.Path(new)
    if new.exists():
        raise FileExistsError(f"refusing to write into the existing {new}")
    new.mkdir(parents=True)
    n = 0
    for p in sorted(base.iterdir()):
        if p.is_file() and not p.name.startswith(f"day_{month}-"):
            os.link(p, new / p.name)
            n += 1
    return n


def check_tables(old_dir, new_dir, min_rows: int = MIN_ROWS) -> dict:
    old_dir, new_dir = pathlib.Path(old_dir), pathlib.Path(new_dir)
    rep, failed = {}, []
    ot, nt = pd.read_parquet(old_dir / "train.parquet"), pd.read_parquet(new_dir / "train.parquet")
    rep["train_identical"] = bool(ot.equals(nt))
    os_, ns = pd.read_parquet(old_dir / "serve.parquet"), pd.read_parquet(new_dir / "serve.parquet")
    oj, nj = os_[os_.month == 1].reset_index(drop=True), ns[ns.month == 1].reset_index(drop=True)
    rep["january_identical"] = bool(oj.equals(nj))
    o7 = os_[os_.month == 7].set_index("MVT_ID_mvt")
    n7 = ns[ns.month == 7].set_index("MVT_ID_mvt").reindex(o7.index)
    was, now = o7.joined.to_numpy(float) == 1, n7.joined.to_numpy(float) == 1
    ap = o7.ap.astype(str).to_numpy()
    rep["july_joined"] = {a: {"old": float(was[ap == a].mean()), "new": float(now[ap == a].mean())} for a in sorted(set(ap))}
    rep["lost_joins"] = int((was & ~now).sum())
    falls = [a for a, v in rep["july_joined"].items() if v["new"] < v["old"] - 1e-12]
    rep["july_never_falls"] = not falls            # as registered: the share; lost joins (ambiguity refusals) reported
    newly = now & ~was
    dt = np.abs(n7.dt_takeoff.to_numpy(float))
    q = {}
    for a in sorted(set(ap[newly])):
        sel = newly & (ap == a)
        q[a] = {"n_new": int(sel.sum()), "share_over_120s": float((dt[sel] > MISJOIN_S).mean()),
                "judged": bool(sel.sum() >= min_rows)}
    rep["newly_joined"] = q
    rep["join_quality"] = all(v["share_over_120s"] <= MISJOIN_MAX_SHARE for v in q.values() if v["judged"])
    sources = json.loads((new_dir / "report.json").read_text()).get("sources", ["adsb.lol"])
    rep["new_sources"] = sources
    rep["declares_adsb_fi"] = "adsb.fi" in sources
    for k in ("train_identical", "january_identical", "july_never_falls", "join_quality", "declares_adsb_fi"):
        if not rep[k]:
            failed.append(k)
    rep["failed"], rep["ok"] = failed, not failed
    return rep


def main(argv=None) -> int:  # pragma: no cover - real files (the parts above and adsb_union are tested)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("days")
    sub.add_parser("store")
    c = sub.add_parser("check")
    c.add_argument("--old", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "days":
        U = _union()
        info = U.build_days(JULY, UNION_DIR, fi_dir=FI_DIR, lol_dir=LOL_DIR, fi_exclude_airports=FI_EXCLUDE_AIRPORTS,
                            meta={"prereg": "plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md (AFI.1)"})
        U.write_manifest(UNION_DIR, info, SOURCE, LICENCE)
        print(json.dumps({"days": len(info), "fi_kept": sum(v["n_fi_kept"] for v in info.values())}))
        return 0 if len(info) == len(JULY) else 1
    if a.cmd == "store":
        n = link_store(BASE_STORE, AFI_STORE, month="2026-07")
        print(f"linked {n} files into {AFI_STORE}")
        return 0
    r = check_tables(a.old, a.new)
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r[k] for k in ("ok", "failed", "lost_joins", "july_joined")}, default=str))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
