"""`scripts/rome_dateslip.py` -- H-DS of plans/PREREG_rome_dateslip_2026_09_10.md.

    python scripts/rome_dateslip.py

No model is fitted. Reads arm R's per-row record (`data/cache_stand/rome_fill_preds.parquet`: S1
and R on every Rome unmatched row of LOMO-12 and fold A) and the unmatched frame
(`data/cache_rome/unm.parquet`) for the training-month class shares, and replaces the prediction
inside segment G -- LIRF unmatched, take-off on the day after the schedule, sp >= 40,000 s -- by
the two-class expectation

    q * (86,400 + m) + (1 - q) * sp

q = the Laplace date-slip share of the fold's TRAINING rows in the row's sub-band, m = the median
taxi of the training date-slips. Arms: S1, D2 (S1 outside G), R2 (R outside G). Clauses as
registered; the verdict word is computed from them.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_fill as rf  # noqa: E402

DS_LO, DS_HI = 86_400.0, 90_000.0
G_LO, G_SPLIT = 40_000.0, 56_000.0
M_FALLBACK = 1_000.0
FOLD_A = (1, 7)
MIN_RED = {"D2": 0.03, "R2": 0.15}
PREDS = ROOT / "data" / "cache_stand" / "rome_fill_preds.parquet"
UNM = ROOT / "data" / "cache_rome" / "unm.parquet"


def is_dateslip(y) -> np.ndarray:
    y = np.asarray(y, dtype="float64")
    return (y >= DS_LO) & (y < DS_HI)


def segment_mask(f: pd.DataFrame) -> np.ndarray:
    return ((f.ADEP_mvt.to_numpy() == "LIRF") & (f.dayoff.to_numpy() == 1)
            & (f.sp.to_numpy(dtype="float64") >= G_LO))


def sub_band(sp) -> np.ndarray:
    return np.where(np.asarray(sp, dtype="float64") < G_SPLIT, "G1", "G2")


def training_rows(unm: pd.DataFrame, held) -> pd.DataFrame:
    """LIRF unmatched rows of every month NOT in `held` (the fold's test months)."""
    return unm[(unm.ADEP_mvt == "LIRF") & ~unm.month.isin(list(held))]


def band_shares(train: pd.DataFrame) -> tuple:
    """({'G1': q, 'G2': q}, m) from training rows: Laplace date-slip share per sub-band inside G,
    and the median y - 86,400 over ALL training date-slips (fallback M_FALLBACK)."""
    g = train[segment_mask(train)]
    band, dsl = sub_band(g.sp), is_dateslip(g.y)
    q = {b: float((dsl[band == b].sum() + 1) / ((band == b).sum() + 2)) for b in ("G1", "G2")}
    all_ds = is_dateslip(train.y)
    m = float(np.median(train.y.to_numpy(dtype="float64")[all_ds] - DS_LO)) if all_ds.any() else M_FALLBACK
    return q, m


def apply(f: pd.DataFrame, base, q: dict, m: float) -> np.ndarray:
    """Inside G the two-class expectation; outside G the base prediction, copied."""
    out = np.asarray(base, dtype="float64").copy()
    g = segment_mask(f)
    if g.any():
        sp = f.sp.to_numpy(dtype="float64")[g]
        qq = np.array([q[b] for b in sub_band(sp)])
        out[g] = qq * (DS_LO + m) + (1 - qq) * sp
    return out


def run(preds: pd.DataFrame, unm: pd.DataFrame, n_boot: int = rf.N_BOOT) -> dict:
    key = unm[["MVT_ID_mvt", "ADEP_mvt", "dayoff"]].drop_duplicates("MVT_ID_mvt")
    p = preds.merge(key, on="MVT_ID_mvt", how="left")
    if p.dayoff.isna().any():
        raise ValueError(f"{int(p.dayoff.isna().sum())} stored rows have no unmatched-frame row")
    p["dayoff"] = p.dayoff.astype(int)
    for fold in ("lomo", "A"):
        if not (p.fold == fold).any():
            raise ValueError(f"no {fold!r} rows in the stored record: the registered clauses need both folds")
    p["D2"], p["R2"] = p.S1.to_numpy(dtype="float64"), p.R.to_numpy(dtype="float64")
    p["in_g"] = segment_mask(p)
    shares = {}
    for fold in ("lomo", "A"):
        groups = [(str(m), [int(m)]) for m in sorted(p[p.fold == "lomo"].month.unique())] if fold == "lomo" \
            else [("A", list(FOLD_A))]
        for name, held in groups:
            rows = (p.fold == fold).to_numpy() & p.month.isin(held).to_numpy()
            if not rows.any():
                continue
            q, m = band_shares(training_rows(unm, held))
            shares[f"{fold}:{name}"] = {"q": q, "m": m}
            sub = p[rows]
            p.loc[rows, "D2"] = apply(sub, sub.S1, q, m)
            p.loc[rows, "R2"] = apply(sub, sub.R, q, m)
    arms = ("S1", "R", "D2", "R2")
    y = p.y.to_numpy(dtype="float64")
    se = {a: (p[a].to_numpy(dtype="float64") - y) ** 2 for a in arms}

    def block(fold):
        mk = (p.fold == fold).to_numpy()
        months = p.month.to_numpy()[mk]
        bm = {a: pd.Series(se[a][mk]).groupby(months).sum() for a in arms}
        g = mk & p.in_g.to_numpy()
        return ({"n_rows": int(mk.sum()), "n_in_g": int(g.sum()), "sse": {a: float(se[a][mk].sum()) for a in arms},
                 "sse_in_g": {a: float(se[a][g].sum()) for a in arms},
                 "rmse": {a: float(np.sqrt(se[a][mk].mean())) for a in arms}}, bm)

    lomo, bm = block("lomo")
    fa, _ = block("A")
    out = {"lomo": lomo, "fold_A": fa, "shares": shares, "clauses": {}, "verdict": {}}
    for arm in ("D2", "R2"):
        boot = rf.month_block_bootstrap(bm["S1"].to_numpy(), bm[arm].to_numpy(), n_boot, rf.BOOT_SEED)
        red_l = 1 - lomo["sse"][arm] / lomo["sse"]["S1"]
        red_a = 1 - fa["sse"][arm] / fa["sse"]["S1"]
        not_worse = int((bm[arm] <= bm["S1"]).sum())
        c = {"c1_month_block_interval": bool(boot["ci95"][0] > 0),
             "c2_lomo_reduction": bool(red_l >= MIN_RED[arm]),
             "c3_foldA_reduction": bool(red_a >= MIN_RED[arm]),
             "c5_months_not_worse_ge_8": bool(not_worse >= 8)}
        out[f"{arm}_boot"], out[f"{arm}_red_lomo"], out[f"{arm}_red_A"] = boot, red_l, red_a
        out[f"{arm}_months_not_worse"] = not_worse
        out["clauses"][arm] = c
        out["verdict"][arm] = ("ESTABLISHED*" if all(c.values()) else
                               "NOT WORKING" if not c["c1_month_block_interval"] else "INCONCLUSIVE")
    out["clauses"]["R2"]["c4_see_RESULT_R_seed_sd"] = None
    out["preds"] = p
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "reports" / "rome_dateslip.json"))
    a = ap.parse_args(argv)
    res = run(pd.read_parquet(PREDS), pd.read_parquet(UNM))
    p = res.pop("preds")
    for name, blk in (("LOMO-12", res["lomo"]), ("fold A", res["fold_A"])):
        print(f"{name}: rows {blk['n_rows']:,}, in G {blk['n_in_g']}")
        for arm in ("S1", "R", "D2", "R2"):
            print(f"  {arm:3s} RMSE {blk['rmse'][arm]:8.1f}  SSE {blk['sse'][arm] / 1e6:9.1f} M   inside G {blk['sse_in_g'][arm] / 1e6:8.1f} M")
    for arm in ("D2", "R2"):
        b = res[f"{arm}_boot"]
        print(f"{arm}: LOMO red {100 * res[f'{arm}_red_lomo']:.1f}%  fold-A red {100 * res[f'{arm}_red_A']:.1f}%  "
              f"months not worse {res[f'{arm}_months_not_worse']}/12  boot {b['gain'] / 1e6:+.1f} M "
              f"[{b['ci95'][0] / 1e6:+.1f}, {b['ci95'][1] / 1e6:+.1f}]  clauses {res['clauses'][arm]}  -> {res['verdict'][arm]}")
    print("shares (fold A):", res["shares"].get("A:A"))
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    p.to_parquet(ROOT / "data" / "cache_stand" / "rome_dateslip_preds.parquet")
    print(f"json -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
