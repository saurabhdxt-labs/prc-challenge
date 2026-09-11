"""UMD — where v10's unmatched error is on the 2026 composition, and how much of it is learnable.

Registered in reports/UNMATCHED_DIAGNOSTIC_2026_09_11.md (Part 1, written before any residual was read). Light: two stored
2025 LOMO records + the scored file's airports and sp; no fitting.

    python3.11 -B scripts/unm_diag.py        -> reports/unm_diag.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_stand"
OUT = ROOT / "reports" / "unm_diag.json"
N_SCORED = 344_841
RULES_AP = "LIRF"
E1_LO, E1_HI = 24_000.0, 86_400.0
MONSTER_S, DATESLIP_Y, FILL_S = 10_800.0, 80_000.0, 60.0
BAND_EDGES = (-np.inf, 0.0, 1_800.0, 3_600.0, 10_800.0, 24_000.0, 86_400.0, np.inf)
BAND_NAMES = ("<0", "0-1800", "1800-3600", "3600-10800", "10800-24000", "24000-86400", ">=86400")
CLASSES = ("fill", "dateslip", "familyB", "monster_other", "ordinary")
GUARD_TOL = 1e-6


def row_class(y, sp) -> np.ndarray:
    """Label-defined classes, mutually exclusive, in the registered order."""
    y, sp = np.asarray(y, dtype="float64"), np.asarray(sp, dtype="float64")
    if np.isnan(y).any() or np.isnan(sp).any():
        raise ValueError("row_class needs y and sp on every row")
    fill = np.abs(sp - y) <= FILL_S
    ds = ~fill & (y > DATESLIP_Y)
    fb = ~fill & ~ds & (y > MONSTER_S) & (sp < MONSTER_S)
    mo = ~fill & ~ds & ~fb & (y > MONSTER_S)
    return np.select([fill, ds, fb, mo], list(CLASSES[:4]), default="ordinary").astype(object)


def sp_band(sp) -> np.ndarray:
    sp = np.asarray(sp, dtype="float64")
    if np.isnan(sp).any():
        raise ValueError("sp_band needs sp on every row")
    idx = np.searchsorted(np.asarray(BAND_EDGES[1:-1]), sp, side="right")   # left-closed bands
    return np.asarray(BAND_NAMES, dtype=object)[idx]


def e1_floor(pred, sp, ap) -> np.ndarray:
    """rome_bandfloor's rule on already-rounded values: max(v, rint(sp)) on LIRF rows with LO <= sp < HI."""
    pred, sp = np.asarray(pred, dtype="float64"), np.asarray(sp, dtype="float64")
    rule = (np.asarray(ap) == RULES_AP) & (sp >= E1_LO) & (sp < E1_HI)
    return np.where(rule, np.maximum(pred, np.rint(sp)), pred)


def learnable(cls, sp) -> np.ndarray:
    """ordinary rows, and fill rows with sp < 10,800 (registered split)."""
    cls, sp = np.asarray(cls), np.asarray(sp, dtype="float64")
    return (cls == "ordinary") | ((cls == "fill") & (sp < MONSTER_S))


def kish(se) -> float:
    se = np.asarray(se, dtype="float64")
    s2 = float((se ** 2).sum())
    return float(se.sum() ** 2 / s2) if s2 > 0 else float("nan")


def top_share(se, k: int) -> float:
    se = np.sort(np.asarray(se, dtype="float64"))[::-1]
    tot = float(se.sum())
    return float(se[:k].sum() / tot) if tot > 0 else float("nan")


def expectation_2026(train: pd.DataFrame, serve: pd.DataFrame, n_scored: int = N_SCORED) -> dict:
    """Expected 2026 board MSE: per (airport, band) stratum, 2025 mean SE by class × 2026 row count ÷ n_scored.
    `train`: ap, band, cls, se (2025 LOMO rows). `serve`: ap, band (2026 scored unmatched rows). A 2026 stratum with no
    2025 row borrows its airport's mean SE and class split; an airport with no 2025 row raises."""
    for c in ("ap", "band", "cls", "se"):
        if c not in train.columns:
            raise ValueError(f"train lacks {c}")
    missing_ap = sorted(set(serve.ap) - set(train.ap))
    if missing_ap:
        raise ValueError(f"2026 airports without any 2025 row: {missing_ap}")
    n_tr = train.groupby(["ap", "band"]).size()
    cls_se = train.pivot_table(index=["ap", "band"], columns="cls", values="se", aggfunc="sum", fill_value=0.0)
    ap_n = train.groupby("ap").size()
    ap_cls = train.pivot_table(index="ap", columns="cls", values="se", aggfunc="sum", fill_value=0.0)
    for c in CLASSES:
        cls_se[c] = cls_se.get(c, 0.0)
        ap_cls[c] = ap_cls.get(c, 0.0)
    cnt = serve.groupby(["ap", "band"]).size()
    by_cls = {c: 0.0 for c in CLASSES}
    by_ap = {}
    borrowed = []
    for (a, b), n26 in cnt.items():
        if (a, b) in n_tr.index:
            per_row = cls_se.loc[(a, b), list(CLASSES)] / n_tr.loc[(a, b)]
        else:
            per_row = ap_cls.loc[a, list(CLASSES)] / ap_n.loc[a]
            borrowed.append({"ap": a, "band": b, "n_2026": int(n26)})
        contrib = per_row * n26 / n_scored
        for c in CLASSES:
            by_cls[c] += float(contrib[c])
        by_ap[a] = by_ap.get(a, 0.0) + float(contrib.sum())
    return {"total": float(sum(by_cls.values())), "by_class": by_cls, "by_airport": by_ap, "borrowed_strata": borrowed,
            "n_2026_rows": int(len(serve))}


# ------------------------------------------------------------------ real data

def load_2025() -> pd.DataFrame:
    u = pd.read_parquet(CACHE / "unm_congestion_lomo.parquet", columns=["MVT_ID_mvt", "fold", "month", "ADEP_mvt", "y", "sp", "S1C"])
    u = u[u.fold == "lomo"].reset_index(drop=True)
    r = pd.read_parquet(CACHE / "rome_dateslip_preds.parquet", columns=["MVT_ID_mvt", "fold", "y", "sp", "R2"])
    r = r[r.fold == "lomo"].reset_index(drop=True)
    e3c = json.loads((ROOT / "reports" / "unm_congestion.json").read_text())["descriptive"]
    rd = json.loads((ROOT / "reports" / "rome_dateslip.json").read_text())["lomo"]["rmse"]["R2"]
    lirf = u.ADEP_mvt == RULES_AP
    guards = {}
    for name, m, want in (("S1C fold-A months", ~lirf & u.month.isin([1, 7]), e3c["fold_a_months"]["rmse"]["S1C"]),
                          ("S1C other months", ~lirf & ~u.month.isin([1, 7]), e3c["other_months"]["rmse"]["S1C"])):
        got = float(np.sqrt(((u.y[m] - u.S1C[m]) ** 2).mean()))
        if abs(got - want) > GUARD_TOL:
            raise AssertionError(f"guard {name}: {got} vs {want}")
        guards[name] = got
    got = float(np.sqrt(((r.y - r.R2) ** 2).mean()))
    if abs(got - rd) > GUARD_TOL:
        raise AssertionError(f"guard R2 LOMO: {got} vs {rd}")
    guards["R2 LOMO"] = got
    ul = u[lirf].set_index("MVT_ID_mvt")
    if set(ul.index) != set(r.MVT_ID_mvt) or len(r) != int(lirf.sum()):
        raise AssertionError("LIRF rows differ between the E3C and Rome records")
    rr = r.set_index("MVT_ID_mvt").reindex(ul.index)
    if not (np.array_equal(rr.y.to_numpy(), ul.y.to_numpy()) and np.array_equal(rr.sp.to_numpy(), ul.sp.to_numpy())):
        raise AssertionError("LIRF labels / sp differ between the records")
    pred = np.rint(u.S1C.to_numpy(dtype="float64"))
    pos = np.flatnonzero(lirf.to_numpy())
    pred[pos] = np.rint(rr.R2.to_numpy(dtype="float64"))
    pred = e1_floor(pred, u.sp.to_numpy(), u.ADEP_mvt.to_numpy())
    out = pd.DataFrame({"MVT_ID_mvt": u.MVT_ID_mvt, "month": u.month, "ap": u.ADEP_mvt.astype(str), "y": u.y, "sp": u.sp,
                        "pred": pred})
    out.attrs["guards"] = guards
    return out


def load_2026() -> pd.DataFrame:
    sys.path.insert(0, str(ROOT))
    from prc.pipeline import config as C, ingest
    sd = ingest.ingest_scored(C.load_config(ROOT / "configs" / "pipeline.yaml"))
    s = sd.scored[sd.scored.unmatched.to_numpy(dtype=bool)]
    return pd.DataFrame({"ap": s.ADEP_mvt.astype(str).to_numpy(), "sp": s.sp.to_numpy(dtype="float64")})


def main() -> int:
    tr = load_2025()
    tr["se"] = (tr.y - tr.pred) ** 2
    tr["cls"] = row_class(tr.y, tr.sp)
    tr["band"] = sp_band(tr.sp)
    tr["learnable"] = learnable(tr.cls, tr.sp)
    sv = load_2026()
    sv["band"] = sp_band(sv.sp)
    exp = expectation_2026(tr[["ap", "band", "cls", "se"]], sv)
    # learnable share of the 2026 expectation: the same arithmetic on the learnable rows' SE only
    tr_l = tr.assign(se=np.where(tr.learnable, tr.se, 0.0))
    exp_l = expectation_2026(tr_l[["ap", "band", "cls", "se"]], sv)
    cls_stats = {}
    for c in CLASSES:
        m = tr.cls == c
        cls_stats[c] = {"n_2025": int(m.sum()), "sse_share_2025": float(tr.se[m].sum() / tr.se.sum()),
                        "top10_share": top_share(tr.se[m], 10), "kish": kish(tr.se[m]),
                        "by_airport_n": tr[m].groupby("ap").size().to_dict()}
    fill_small = (tr.cls == "fill") & (tr.sp < MONSTER_S)
    res = {"tag": "UMD", "prereg": "reports/UNMATCHED_DIAGNOSTIC_2026_09_11.md#part-1",
           "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "guards": tr.attrs["guards"], "n_2025": int(len(tr)), "rmse_2025_lomo": float(np.sqrt(tr.se.mean())),
           "expectation_2026": exp, "learnable_expectation_2026": exp_l["total"],
           "learnable_by_airport_2026": exp_l["by_airport"],
           "classes_2025": cls_stats,
           "learnable_2025": {"n": int(tr.learnable.sum()), "sse_share": float(tr.se[tr.learnable].sum() / tr.se.sum()),
                              "top10_share": top_share(tr.se[tr.learnable], 10),
                              "fill_small_sp_n": int(fill_small.sum()), "fill_small_sp_top10": top_share(tr.se[fill_small], 10),
                              "ordinary_top10": top_share(tr.se[tr.cls == "ordinary"], 10)},
           "lane_top20_share_2025": top_share(tr.se, 20), "lane_kish_2025": kish(tr.se),
           "inferred_board_unmatched_mse": 79_252.0 - 0.9846596 * 222.56316251601282 ** 2}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("guards", "rmse_2025_lomo", "learnable_expectation_2026", "lane_top20_share_2025",
                                          "inferred_board_unmatched_mse")}, default=float))
    print("2026 expectation by class:", {k: round(v) for k, v in exp["by_class"].items()}, "total", round(exp["total"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
