"""FMF (plans/PREREG_fm1_final_fit_2026_09_30.md, amendment FMF.2) — the FM1 final fit's label-free gates.

    /opt/homebrew/bin/python3.11 -B research/analysis/fmf_checks.py rss --log logs/fm1_final/fmf_dry_fit.log \
        --guard-log logs/fm1_final/fmf_dry_fit_memguard.log
    /opt/homebrew/bin/python3.11 -B research/analysis/fmf_checks.py same --a logs/fm1_final/det_a/pred.parquet \
        --b logs/fm1_final/det_b/pred.parquet
    /opt/homebrew/bin/python3.11 -B research/analysis/fmf_checks.py accept --sub data/fm1/final/merry-quicksand_v50.parquet \
        --prov data/fm1/final/merry-quicksand_v50.provenance.parquet --base submissions/merry-quicksand_v46.parquet \
        --out logs/fm1_final/v50_accept.json

rss     the dry run's peak memory (the larger of /usr/bin/time -l's maximum resident set size and peak memory footprint,
        over every time block of the log's CURRENT run — the queue appends) <= 16 GB; a log without a resident-set
        line is refused. With --guard-log: a mem_guard pause voids the footprint (time -l snapshots it at the first
        stop) and max RSS is gated at 16 / 1.6 = 10 GB instead (FMF.6).
rows    BSV: a variant's change report vs its base shows >= 10,000 changed rows (a whole-model variant, not a row edit).
same    determinism (A2): two predict outputs have the same columns, dtypes and rows, and every cell is identical (floats
        compared bit for bit; NaN only where the other side has the same NaN).
accept  A3 the submission's ids are the template's in its order, every value >= 1; A4 every provenance row has a
        decided_by from FM1's closed vocabulary; A6 on the matched-lane rows of v46's 15 R12 cells, the RMS of
        (new - v46) after removing each cell's own mean difference <= 81.6 s (the DWL R2 anchor); a per-cell constant
        (MTX, which the forecaster file omits) is reported, never counted. A5 is dwl_checks.py report.
Reads no label. Exit 0 = the gate passes; 1 = it fails or the inputs are refused.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RSS_LIMIT = 16_000_000_000          # registered: abort the dry run's queue above 16 GB (24 GB machine, Phantom spikes)
A6_LIMIT_S = 81.6                   # FMF.2: the DWL R2 anchor, 2 x the measured LightGBM refit-vs-holdout RMS 40.8 s
#: v46's fifteen R12 airport-months (A6's population): seven airports x {January, July} + EGLL January. EGLL July was
#: reverted in v43; LFPG and LTFM never carried the R12 stack. (The same set as the v46 build's scope.)
R12_SCOPE = tuple((a, m) for a in ("EDDF", "EDDM", "EHAM", "LEBL", "LSZH", "LEMD", "LIRF") for m in (1, 7)) + (("EGLL", 1),)
RSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size\s*$", re.M)
FOOT_RE = re.compile(r"^\s*(\d+)\s+peak memory footprint\s*$", re.M)
RUN_RE = re.compile(r"^\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] (?:lock|light|r12) ", re.M)   # prc_queue._execute's header
PAUSE_RATIO = 1.6                   # FMF.6: footprint / max RSS, worst measured 1.53 (run 1: 17.80 / 11.65 GB)
MIN_CHANGED = 10_000                # BSV: a variant is a whole-model change of at least this many scored rows (2026-10-01)
IMPLAUSIBLE = 0.25                  # credible footprint / max RSS pairs measured 2026-10-01: 0.93 – 1.53


def last_run(log_text: str) -> str:
    """The current run of an appended queue step log: the text from the last "[date time] lock|light|r12 ..." header on
    (tools/prc_queue.py writes one per execution and appends); a log without a header is one run."""
    heads = list(RUN_RE.finditer(log_text))
    return log_text[heads[-1].start():] if heads else log_text


def guard_paused(guard_text: str) -> bool:
    """True when mem_guard paused the tree during its last run (the lines after the last MEMGUARD START; a RESTART line,
    a guard replaced while the same run continues, does not begin a new run)."""
    i = guard_text.rfind("MEMGUARD START")
    return "MEMGUARD PAUSED" in (guard_text[i:] if i >= 0 else guard_text)


def rss(log_text: str, *, limit: int = RSS_LIMIT, guard_text=None) -> dict:
    """Peak memory of the current run (every /usr/bin/time -l block after the last queue header), gated at `limit`
    bytes (<= passes). `time -l` snapshots its peak footprint at the child's first stop, so a mem_guard pause voids
    it: with a PAUSED line in `guard_text`'s last run the footprint is not used and max RSS (wait4, always right) is
    gated at limit / PAUSE_RATIO. A footprint below IMPLAUSIBLE x max RSS with no pause on record is refused."""
    seg = last_run(log_text)
    res = [int(x) for x in RSS_RE.findall(seg)]
    if not res:
        raise ValueError("no 'maximum resident set size' line in the log (was the command run under /usr/bin/time -l?)")
    foot = [int(x) for x in FOOT_RE.findall(seg)]
    out = {"max_rss_bytes": max(res), "footprint_bytes": max(foot) if foot else None}
    if guard_text is not None and guard_paused(guard_text):
        lim = int(limit / PAUSE_RATIO)
        return {**out, "peak_bytes": max(res), "footprint_used": False, "paused": True, "limit_bytes": lim,
                "ok": bool(max(res) <= lim)}
    if foot and max(foot) < IMPLAUSIBLE * max(res):
        raise ValueError(f"peak memory footprint {max(foot):,} is not credible beside a resident set of {max(res):,} "
                         "(time -l truncates it when the tree is paused: pass the step's guard log)")
    peak = max(res + foot)
    return {**out, "peak_bytes": peak, "footprint_used": bool(foot), "paused": False, "limit_bytes": int(limit),
            "ok": bool(peak <= limit)}


def min_changed(report: dict, *, minimum: int = MIN_CHANGED) -> dict:
    """BSV's variant size rule on a dwl_checks change report: a variant must change >= `minimum` scored rows vs its base."""
    if "n_changed" not in report:
        raise ValueError("no n_changed in the change report")
    n = int(report["n_changed"])
    return {"n_changed": n, "minimum": int(minimum), "ok": bool(n >= minimum)}


def _cell_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.dtype.kind == "f":
        u = f"u{a.dtype.itemsize}"
        return np.ascontiguousarray(a).view(u) != np.ascontiguousarray(b).view(u)
    if a.dtype.kind in "iub":
        return a != b
    sa, sb = pd.Series(a, dtype=object), pd.Series(b, dtype=object)
    return ~(sa.eq(sb) | (sa.isna() & sb.isna())).to_numpy()


def same(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Whole-frame identity of two predict outputs (columns, dtypes, row order, every value)."""
    if list(a.columns) != list(b.columns):
        raise ValueError(f"columns differ: {list(a.columns)} vs {list(b.columns)}")
    if len(a) != len(b):
        raise ValueError(f"rows differ: {len(a)} vs {len(b)}")
    n_diff, cols = 0, []
    for c in a.columns:
        if a[c].dtype != b[c].dtype:
            raise ValueError(f"dtype of {c} differs: {a[c].dtype} vs {b[c].dtype}")
        k = int(_cell_diff(a[c].to_numpy(), b[c].to_numpy()).sum())
        if k:
            n_diff += k
            cols.append(str(c))
    return {"rows": int(len(a)), "columns": int(a.shape[1]), "n_diff_cells": n_diff, "diff_columns": cols,
            "ok": n_diff == 0}


def accept(sub: pd.DataFrame, prov: pd.DataFrame, template: pd.DataFrame, base: pd.DataFrame, cells: pd.DataFrame, *,
           scope, vocab, limit: float = A6_LIMIT_S) -> dict:
    """A3 / A4 / A6 of the final file (amendment FMF.2). Raises ValueError on a refused input; ok = A6 within limit."""
    ids = sub.MVT_ID_mvt.to_numpy(float)
    if not np.array_equal(ids, template.MVT_ID_mvt.to_numpy(float)):
        raise ValueError("A3: ids / order differ from the template")
    low = int((sub.TAXITIME_SEC_mvt.to_numpy() < 1).sum())
    if low:
        raise ValueError(f"A3: {low} value(s) < 1")
    p = prov.set_index(prov.MVT_ID_mvt.to_numpy(float))
    if not p.index.is_unique or set(p.index) != set(ids):
        raise ValueError("A4: provenance ids differ from the submission's")
    known = p.decided_by.isin(vocab) & p.decided_by.notna()
    if not known.all():
        raise ValueError(f"A4: {int((~known).sum())} row(s) without a known decided_by")
    a4 = {"rows": int(len(p)), "decided_by": {str(k): int(v) for k, v in sorted(p.decided_by.value_counts().items())}}
    if not np.array_equal(ids, base.MVT_ID_mvt.to_numpy(float)):
        raise ValueError("A6: base ids / order differ from the submission")
    c = cells.set_index(cells.MVT_ID_mvt.to_numpy(float))
    missing = int((~pd.Index(ids).isin(c.index)).sum())
    if missing:
        raise ValueError(f"A6: {missing} submission id(s) without an airport / month cell")
    ap = c.ADEP_mvt.reindex(ids).astype(str).to_numpy()
    mo = c.month.reindex(ids).astype(int).to_numpy()
    in_scope = np.array([(a, int(m)) in set(scope) for a, m in zip(ap, mo)])
    sel = in_scope & (p.lane.reindex(ids).to_numpy() == "matched")
    if not sel.any():
        raise ValueError("A6: no matched rows in the scope cells")
    d = sub.TAXITIME_SEC_mvt.to_numpy(float)[sel] - base.TAXITIME_SEC_mvt.to_numpy(float)[sel]
    key = pd.Series([f"{a}|{m}" for a, m in zip(ap[sel], mo[sel])])
    mean = pd.Series(d).groupby(key).transform("mean").to_numpy()
    dem = float(np.sqrt(np.mean((d - mean) ** 2)))
    per = {str(k): {"n": int(len(g)), "mean_diff_s": float(g.mean())} for k, g in pd.Series(d).groupby(key)}
    a6 = {"n_rows": int(sel.sum()), "rms_demeaned_s": dem, "rms_raw_s": float(np.sqrt(np.mean(d ** 2))),
          "limit_s": float(limit), "cells": per, "ok": bool(dem <= limit)}
    return {"a3": {"rows": int(len(sub))}, "a4": a4, "a6": a6, "ok": a6["ok"]}


def _cells():  # pragma: no cover - real file
    rk = pd.read_parquet(ROOT / "data" / "raw" / "ranking.parquet", columns=["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    return pd.DataFrame({"MVT_ID_mvt": rk.MVT_ID_mvt.to_numpy(float), "ADEP_mvt": rk.ADEP_mvt.astype(str),
                         "month": pd.to_datetime(rk.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()})


def main(argv=None) -> int:  # pragma: no cover - thin I/O shell around the tested functions
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("rss")
    s.add_argument("--log", required=True)
    s.add_argument("--limit", type=int, default=RSS_LIMIT)
    s.add_argument("--guard-log", default=None, help="the step's mem_guard log (a pause voids time -l's footprint)")
    s = sub.add_parser("rows")
    s.add_argument("--report", required=True, help="a dwl_checks.py change report (variant vs its base)")
    s.add_argument("--min", type=int, default=MIN_CHANGED)
    s = sub.add_parser("same")
    s.add_argument("--a", required=True)
    s.add_argument("--b", required=True)
    s = sub.add_parser("accept")
    s.add_argument("--sub", required=True)
    s.add_argument("--prov", required=True)
    s.add_argument("--base", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=float, default=A6_LIMIT_S)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "rss":
            g = pathlib.Path(a.guard_log).read_text(errors="replace") if a.guard_log else None
            r = rss(pathlib.Path(a.log).read_text(errors="replace"), limit=a.limit, guard_text=g)
        elif a.cmd == "rows":
            r = min_changed(json.loads(pathlib.Path(a.report).read_text()), minimum=a.min)
        elif a.cmd == "same":
            r = same(pd.read_parquet(a.a), pd.read_parquet(a.b))
        else:
            out = pathlib.Path(a.out)
            if out.exists():
                raise ValueError(f"refusing to overwrite {out}")
            sys.path.insert(0, str(ROOT))
            from prc.pipeline.fm1.stages import DECIDED_BY
            r = accept(pd.read_parquet(a.sub), pd.read_parquet(a.prov),
                       pd.read_parquet(ROOT / "data" / "raw" / "submitting.parquet"), pd.read_parquet(a.base), _cells(),
                       scope=R12_SCOPE, vocab=DECIDED_BY, limit=a.limit)
            r.update(sub=a.sub, prov=a.prov, base=a.base)
            out.write_text(json.dumps(r, indent=1))
    except ValueError as e:
        print(f"REFUSED: {e}")
        return 1
    print(json.dumps(r))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
