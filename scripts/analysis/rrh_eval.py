"""RRH holdout scorer (`plans/PREREG_rrh_rome_row_hunt_2026_09_23.md`, registered 926fd85a).

    python3.11 -B scripts/analysis/rrh_eval.py      -> reports/rrh_holdout_report.json

Scores every frozen `rule_*` function in `data/rome_hunt_2026_09_23/agent_*_rules.py` on the 397 Jan+Jul 2025 LIRF
unmatched rows that no agent saw. A rule returns p_fill in [0, 1] or NaN ("keep R"); p_new = where(isnan, p_R, rule);
prediction = the SHIPPED mixture max(p*sp + (1-p)*nf, 1) (`build_submission.mixture`, injected, never re-implemented);
g = [SSE(p_R) - SSE(p_new)] / 344,841. CI = day-block bootstrap (calendar days are unique across months, so this is
the registered (month, day) block), 2,000 draws, seed 0; the Bonferroni interval for <= 15 rules uses quantiles
0.00167 / 0.99833 of the same draws.
WORKING iff CI95 lower > 0 AND g minus its top-3 |row gain| rows > 0 AND g >= 0 in both Jan and Jul.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACK = ROOT / "data/rome_hunt_2026_09_23"
PREDS = ROOT / "reports/csf2/csf2_preds.parquet"
OUT = ROOT / "reports/rrh_holdout_report.json"
BOARD_N = 344_841.0
N_BOOT, BOOT_SEED = 2000, 0
HOLDOUT_MONTHS = (1, 7)
BONF_Q = (0.05 / 15 / 2, 1 - 0.05 / 15 / 2)          # 15 rules max, two-sided
#: the shipped chain already commits fill at sp >= 24,000 s (v30 LIRF unmatched p median 1.0 in 24-50k and >= 50k),
#: so gain there, measured against R alone, is production-owned: reported separately, never as new.
OWNED_TAIL_SP = 24_000.0


def check_holdout_months(rows: pd.DataFrame) -> None:
    """The scorer runs on the locked holdout months only; any other month means the wrong rows were loaded."""
    bad = sorted(set(rows["month"].astype(int)) - set(HOLDOUT_MONTHS))
    if bad:
        raise ValueError(f"holdout frame holds non-holdout months {bad}")


def _boot(per_day: np.ndarray, qs) -> list:
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(per_day), size=(N_BOOT, len(per_day)))
    sums = per_day[idx].sum(1) / BOARD_N
    return [float(np.quantile(sums, q)) for q in qs]


def is_working(ci_lo: float, minus_top3: float, by_month: dict) -> bool:
    """The registered clause: CI95 lower > 0 AND minus-top-3 > 0 AND g >= 0 in every holdout month."""
    return bool(ci_lo > 0 and minus_top3 > 0 and all(v >= 0 for v in by_month.values()))


def score_rule(rule_p, p_R, sp, nf, y, month, day, mixture) -> dict:
    """One rule's registered comparison against R on the holdout rows (positive g = the rule improves)."""
    p_R, sp, nf, y = (np.asarray(a, float) for a in (p_R, sp, nf, y))
    rule_p = np.asarray(rule_p, float)
    if rule_p.shape != p_R.shape:
        raise ValueError(f"rule length {rule_p.shape} != rows {p_R.shape}")
    fin = np.isfinite(rule_p)
    if ((rule_p[fin] < 0) | (rule_p[fin] > 1)).any():
        raise ValueError("rule returned p_fill outside [0, 1]")
    p_new = np.where(fin, rule_p, p_R)
    d = (mixture(p_R, sp, nf) - y) ** 2 - (mixture(p_new, sp, nf) - y) ** 2
    month, day = np.asarray(month).astype(int), np.asarray(day).astype(str)
    per_day = pd.Series(d).groupby(day).sum().to_numpy()
    lo, hi, blo, bhi = _boot(per_day, (0.025, 0.975, *BONF_Q))
    order = np.argsort(-np.abs(d))
    keep = np.ones(len(d), bool)
    keep[order[:3]] = False
    g = float(d.sum() / BOARD_N)
    by_month = {str(m): float(d[month == m].sum() / BOARD_N) for m in HOLDOUT_MONTHS}
    minus3 = float(d[keep].sum() / BOARD_N)
    working = is_working(lo, minus3, by_month)
    tail = sp >= OWNED_TAIL_SP
    return {"g": g, "g_sp_lt24000": float(d[~tail].sum() / BOARD_N), "g_sp_ge24000": float(d[tail].sum() / BOARD_N),
            "ci95": [lo, hi], "ci_bonf": [blo, bhi], "minus_top3": minus3, "by_month": by_month,
            "n_fired": int(fin.sum()), "n_rows": int(len(d)), "n_days": int(len(per_day)),
            "fill_rate_fired": None, "verdict": "WORKING" if working else "NOT WORKING",
            "bonferroni_lead": bool(working and blo > 0)}


def verdict_rows(sp, arc_rule) -> np.ndarray:
    """RRH.1: the verdict rows are NOT ARC-reached (shipped `sub_rule` 'ord'/'fill') AND sp < 24,000 s."""
    arc_rule = np.asarray(arc_rule).astype(str)
    bad = set(arc_rule) - {"", "none", "ord", "fill"}
    if bad:
        raise ValueError(f"unknown sub_rule codes {sorted(bad)}")
    return ~np.isin(arc_rule, ["ord", "fill"]) & (np.asarray(sp, float) < OWNED_TAIL_SP)


def arc_rule_for(rows: pd.DataFrame) -> np.ndarray:
    """Shipped ARC's sub_rule on the holdout rows (ARR pool: 2025 months 1, 2, 7, 8, so an arrival up to 8 h after a
    month-end take-off is seen)."""
    sys.path.insert(0, str(ROOT / "scripts/analysis"))
    import arr_row_clocks as A
    files = [p for m in ("01", "02", "07", "08") for p in sorted((ROOT / "data/raw").glob(f"training_2025-{m}-*.parquet"))]
    arr = A.arr_frame(A.load_raw(A.ARR_COLS, files))
    dep = pd.DataFrame({"MVT_ID_mvt": rows.MVT_ID_mvt.to_numpy(), "ADEP_mvt": rows.ADEP_mvt.to_numpy(),
                        "ADES_mvt": rows.ADES_mvt.to_numpy(), "FLIGHT_mvt": rows.FLIGHT_mvt.to_numpy(),
                        "mvt": A.secs(rows.MVT_TIME_UTC_mvt)})
    j = A.join_arr(dep, arr)
    return A.sub_rule(rows.sp.to_numpy(float), A.secs(rows.SCHED_TIME_UTC_mvt), j.arr_lobt, j.arr_aobt3)


def load_holdout() -> pd.DataFrame:
    """The 397 holdout rows: R's stored LOMO p and nf from CSF.2, the raw columns from the Jan/Jul raw files."""
    c = pd.read_parquet(PREDS)
    c = c[c.month.isin(HOLDOUT_MONTHS)][["MVT_ID_mvt", "month", "sp", "y", "day", "cls", "nf", "p_R"]]
    raw = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "data/raw").glob("training_2025-0[17]-*.parquet"))],
                    ignore_index=True)
    rows = c.merge(raw, on="MVT_ID_mvt", how="left", validate="1:1")
    check_holdout_months(rows)
    if not np.allclose(rows.y, rows.TAXITIME_SEC_mvt):
        raise AssertionError("holdout labels do not match the raw files")
    return rows.reset_index(drop=True)


def rule_functions() -> list:
    out = []
    for path in sorted(PACK.glob("agent_*_rules.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out += [(f"{path.stem}.{n}", getattr(mod, n)) for n in sorted(dir(mod))
                if n.startswith("rule_") and callable(getattr(mod, n))]
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_submission as bs                      # the shipped mixture
    rows = load_holdout()
    serve = rows.drop(columns=["y", "cls", "nf", "day", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"])   # p_R: R's serve-time output
    fill = (rows.cls == "fill").to_numpy()
    report = {"prereg": "plans/PREREG_rrh_rome_row_hunt_2026_09_23.md", "n_rows": int(len(rows)), "rules": {}}
    oracle = score_rule(fill.astype(float), rows.p_R, rows.sp, rows.nf, rows.y, rows.month, rows.day, bs.mixture)
    report["oracle_two_class"] = oracle
    arc = arc_rule_for(rows)
    V = verdict_rows(rows.sp, arc)
    report["n_verdict_rows"] = int(V.sum())
    report["arc_counts"] = {k: int((arc == k).sum()) for k in ("", "none", "ord", "fill")}
    report["oracle_two_class_verdict_rows"] = score_rule(fill[V].astype(float), *(np.asarray(a)[V] for a in (
        rows.p_R, rows.sp, rows.nf, rows.y, rows.month, rows.day)), bs.mixture)
    for name, fn in rule_functions():
        try:
            p = np.asarray(fn(serve.copy()), float)
            r = score_rule(p, rows.p_R, rows.sp, rows.nf, rows.y, rows.month, rows.day, bs.mixture)
            fired = np.isfinite(p)
            r["fill_rate_fired"] = float(fill[fired].mean()) if fired.any() else None
            r["doc"] = (fn.__doc__ or "").strip()
            rv = score_rule(p[V], *(np.asarray(a)[V] for a in (rows.p_R, rows.sp, rows.nf, rows.y, rows.month,
                                                                   rows.day)), bs.mixture)
            r["verdict_rows"] = rv
            r["verdict_all_rows_NOT_THE_VERDICT"] = r.pop("verdict")
            r["verdict"] = rv["verdict"]
        except Exception:                                # a broken rule is recorded, never silently skipped
            r = {"verdict": "ERROR", "traceback": traceback.format_exc()}
        report["rules"][name] = r
        print(name, r.get("verdict"), {k: r.get(k) for k in ("g", "ci95", "minus_top3", "by_month", "n_fired")}, flush=True)
    with OUT.open("x") as f:
        json.dump(report, f, indent=2, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main())
