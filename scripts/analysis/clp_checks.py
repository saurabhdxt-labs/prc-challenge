"""CLP (plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md) — the EOBT clip build's label-free acceptance.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/clp_checks.py detector --manifest data/fm1/clp/merry-quicksand_v54.manifest.json
    /opt/homebrew/bin/python3.11 -B scripts/analysis/clp_checks.py scope --report logs/fm1_clp/v54_change_vs_v51.json --allow "LFPG|7"

detector  (acceptance 1-2) the build manifest's eobt_clip report: the population is the scored file, the fired cells are
          exactly LFPG 2026-07, and the clipped rows are 7,400 - 7,600 (35.1 % of LFPG July's 21,334 matched departures).
scope     (acceptance 3) a dwl_checks change report vs the base: at least one row changed and every changed row sits in an
          allowed cell ("AP|month"); a change anywhere else is a defect.
Reads no label. Exit 0 = the check passes; 1 = it fails.
"""
from __future__ import annotations

import argparse
import json
import pathlib

FIRED = [["LFPG", "2026-07"]]          # registered: the only cell whose EOBT field carries revisions
N_CLIPPED = (7_400, 7_600)             # registered: 35.1 % of 21,334 = 7,488 expected


def scope(report: dict, allowed) -> dict:
    changed = {k: int(c["n_changed"]) for k, c in report["cells"].items() if c["n_changed"] > 0}
    outside = {k: n for k, n in changed.items() if k not in set(allowed)}
    return {"ok": bool(changed) and not outside, "changed_cells": changed, "outside": outside,
            "n_changed": int(report["n_changed"]), "allowed": list(allowed)}


def detector(manifest: dict, *, fired=FIRED, n_clipped=N_CLIPPED) -> dict:
    rep = manifest.get("eobt_clip")
    if rep is None:
        return {"ok": False, "failed": ["eobt_clip"], "why": "the build manifest has no eobt_clip report"}
    got = [[c["airport"], c["month"]] for c in rep["cells"] if c["fired"]]
    failed = []
    if got != [list(f) for f in fired]:
        failed.append("fired")
    if not n_clipped[0] <= rep["n_clipped"] <= n_clipped[1]:
        failed.append("n_clipped")
    if rep["population"] != "scored_file":
        failed.append("population")
    return {"ok": not failed, "failed": failed, "fired": got, "n_clipped": int(rep["n_clipped"]),
            "population": rep["population"], "cells": rep["cells"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scope")
    s.add_argument("--report", required=True)
    s.add_argument("--allow", action="append", required=True)
    d = sub.add_parser("detector")
    d.add_argument("--manifest", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "scope":
        r = scope(json.loads(pathlib.Path(a.report).read_text()), a.allow)
    else:
        r = detector(json.loads(pathlib.Path(a.manifest).read_text()))
    print(json.dumps(r, indent=1, default=str))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
