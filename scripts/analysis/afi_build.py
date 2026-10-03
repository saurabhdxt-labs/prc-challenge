"""AFI (plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md) — the owner's adsb.fi July-2026 capture into FM1's serve table.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/afi_build.py days      # union day files + MANIFEST.json
    /opt/homebrew/bin/python3.11 -B scripts/analysis/afi_build.py store     # a copy of the R12 store without July 2026
    (then r12_build features --sources <union dir> --days <July 2026> and r12_build tables, through r12_launch)
    /opt/homebrew/bin/python3.11 -B scripts/analysis/afi_build.py fallback --old <tables> --union <tables> --out <tables>
    /opt/homebrew/bin/python3.11 -B scripts/analysis/afi_build.py check --old <tables> --new <tables> --out <json> [--fallback]

days   July 2026: every adsb.lol sample + each adsb.fi sample >= 1 s from the same aircraft's adsb.lol samples
       (scripts/analysis/adsb_union.py, rule AUN.1), written with a MANIFEST.json declaring both sources.
store  hard links of every record of the base store except the July-2026 days, so only those are rebuilt (same feature
       code and stamp) from the union days.
fallback  AFB (plans/PREREG_afb_adsbfi_fallback_2026_10_03.md): tables whose serve rows are adsb.lol's where it joined and
       the union's elsewhere; train is the old file itself; report.json describes these tables.
check  the registered label-free acceptance: (1) train.parquet and the January-2026 serve rows equal the old tables';
       (2) July-2026 joined share never falls, per airport (flights joined before and not after are reported); (3) on flights
       newly joined, the share with |dt_takeoff| > 120 s is <= 0.01 per airport (a misjoin shows there); and the new
       tables declare adsb.fi and serve exactly the old flights. Reads no label.
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


def _script(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "analysis" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _union():
    return _script("adsb_union")


R12 = _script("r12_build")                 # table_report: the per-table statistics every report.json carries


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


def fallback_serve(old: pd.DataFrame, union: pd.DataFrame) -> pd.DataFrame:
    """AFB (plans/PREREG_afb_adsbfi_fallback_2026_10_03.md): the adsb.lol serve rows for every January flight and every
    July flight adsb.lol joined; the union row for July flights it did not join. Same rows, order and columns as `old`."""
    if set(old.MVT_ID_mvt) != set(union.MVT_ID_mvt) or len(old) != len(union):
        raise ValueError(f"fallback_serve: the flight sets differ ({len(old):,} vs {len(union):,} rows)")
    u = union.set_index("MVT_ID_mvt").reindex(old.MVT_ID_mvt.to_numpy())
    take = ((old.month == 7) & (old.joined.to_numpy(float) != 1)).to_numpy()
    out = old.copy()
    for c in old.columns:
        if c != "MVT_ID_mvt":
            out.loc[take, c] = u[c].to_numpy()[take]
    return out


def write_fallback_tables(old_dir, union_dir, out_dir) -> dict:
    """AFB's tables in `out_dir` (a new directory): train.parquet = the old file (a hard link), serve.parquet =
    fallback_serve(old, union), report.json = the union's provenance with sources, and the train / serve statistics of
    THESE tables. Returns the report."""
    old_dir, union_dir, out = pathlib.Path(old_dir), pathlib.Path(union_dir), pathlib.Path(out_dir)
    rep = json.loads((union_dir / "report.json").read_text())
    if "adsb.fi" not in rep.get("sources", []):
        raise ValueError(f"{union_dir}: the union tables' report does not declare adsb.fi (sources {rep.get('sources')})")
    if out.exists():
        raise FileExistsError(f"refusing to write into the existing {out}")
    out.mkdir(parents=True)
    os.link(old_dir / "train.parquet", out / "train.parquet")
    serve = fallback_serve(pd.read_parquet(old_dir / "serve.parquet"), pd.read_parquet(union_dir / "serve.parquet"))
    serve.to_parquet(out / "serve.parquet", index=False)
    rep.update(train=R12.table_report(pd.read_parquet(out / "train.parquet")), serve=R12.table_report(serve),
               merge="fallback: adsb.lol rows where it joined, the union elsewhere (AFB)", union_tables=str(union_dir),
               adsb_lol_tables=str(old_dir))
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


CHECKS = ("train_identical", "january_identical", "same_flights", "july_never_falls", "join_quality", "declares_adsb_fi")


def _row_checks(os_: pd.DataFrame, ns: pd.DataFrame, rep: dict, min_rows: int, fallback: bool) -> None:
    """Acceptance 2-3 (and AFB's joined rows) on serve tables that hold exactly the same flights."""
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
    if fallback:                                     # AFB: every flight adsb.lol joined keeps its adsb.lol row
        jo = os_[(os_.month == 7) & (os_.joined == 1)].set_index("MVT_ID_mvt").sort_index()
        jn = ns.set_index("MVT_ID_mvt").reindex(jo.index)
        rep["joined_rows_identical"] = bool(jo.equals(jn))


def check_tables(old_dir, new_dir, min_rows: int = MIN_ROWS, fallback: bool = False) -> dict:
    """The registered acceptance (module docstring). The row checks run only on serve tables holding exactly the old
    flights; when the flight set differs they are reported as not run and counted as failed."""
    old_dir, new_dir = pathlib.Path(old_dir), pathlib.Path(new_dir)
    rep = {}
    ot, nt = pd.read_parquet(old_dir / "train.parquet"), pd.read_parquet(new_dir / "train.parquet")
    rep["train_identical"] = bool(ot.equals(nt))
    os_, ns = pd.read_parquet(old_dir / "serve.parquet"), pd.read_parquet(new_dir / "serve.parquet")
    oj, nj = os_[os_.month == 1].reset_index(drop=True), ns[ns.month == 1].reset_index(drop=True)
    rep["january_identical"] = bool(oj.equals(nj))
    rep["same_flights"] = bool(len(os_) == len(ns) and set(os_.MVT_ID_mvt) == set(ns.MVT_ID_mvt))
    sources = json.loads((new_dir / "report.json").read_text()).get("sources", ["adsb.lol"])
    rep["new_sources"] = sources
    rep["declares_adsb_fi"] = "adsb.fi" in sources
    wanted = CHECKS + (("joined_rows_identical",) if fallback else ())
    if rep["same_flights"]:
        _row_checks(os_, ns, rep, min_rows, fallback)
    rep["not_run"] = [k for k in wanted if k not in rep]
    rep["failed"] = [k for k in wanted if not rep.get(k, False)]
    rep["ok"] = not rep["failed"]
    return rep


def main(argv=None) -> int:  # pragma: no cover - real files (the parts above and adsb_union are tested)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("days")
    sub.add_parser("store")
    fb = sub.add_parser("fallback")
    fb.add_argument("--old", required=True)
    fb.add_argument("--union", required=True)
    fb.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--old", required=True)
    c.add_argument("--new", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--fallback", action="store_true")
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
    if a.cmd == "fallback":
        rep = write_fallback_tables(a.old, a.union, a.out)
        print(json.dumps({"out": a.out, "serve": {k: rep["serve"][k] for k in ("n_rows", "coverage_counts")}}))
        return 0
    r = check_tables(a.old, a.new, fallback=a.fallback)
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps({k: r.get(k) for k in ("ok", "failed", "not_run", "lost_joins", "july_joined")}, default=str))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
