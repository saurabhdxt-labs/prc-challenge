"""DRF (plans/PREREG_drf_feature_drift_2026_10_02.md) — which inputs of the final model moved between 2025 and 2026?
Label-free: the 2026 scored file's design matrix against the same month of 2025, feature by feature, beside how much the
same feature moves between adjacent 2025 months at the same airport.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/drf_audit.py run --artifact artifacts/fm1/2d29ad344b45-1c61fcc2 \
        --out logs/drf --set adsb.table_dir=data/r12/tables_51348d3e1cb3 [--set ...]

Per airport and month m in {1, 7}: A = 2025 month m, B = 2026 month m, neighbours = 2025 months m-1 and m+1.
  drift   |AUC - 0.5| of a feature between A and B (ties 0.5, NaN ranked lowest)
  null    the larger of the same quantity between A and each neighbour month
  flag    drift >= 0.10 AND drift >= null + 0.05 AND the feature's gain share in the base model >= 1 %
Reads no label: only design columns are used (2025 frames carry y / delta; they are never selected).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = pathlib.Path(__file__).resolve().parents[2]
FLOOR, MARGIN, MIN_GAIN = 0.10, 0.05, 0.01          # the registered screening rule (PREREG "Flag")
MIN_ROWS = 3_000
NEIGHBOURS = {1: (12, 2), 7: (6, 8)}
EPS = 1e-12


def uni_auc(a, b) -> float:
    """P(a value of b ranks above a value of a), ties 0.5; NaN ranks below every value (a changed missing-rate is drift)."""
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    if len(a) == 0 or len(b) == 0:
        raise ValueError("uni_auc: an empty side")
    v = np.concatenate([a, b])
    r = rankdata(np.where(np.isnan(v), -np.inf, v), method="average")
    return float((r[len(a):].sum() - len(b) * (len(b) + 1) / 2.0) / (len(a) * len(b)))


def drift(a: pd.DataFrame, b: pd.DataFrame, cols) -> pd.Series:
    """|AUC - 0.5| per column between two frames."""
    return pd.Series({c: abs(uni_auc(a[c].to_numpy(), b[c].to_numpy()) - 0.5) for c in cols}, dtype="float64")


def flag(drift_ab, null, gain, *, floor: float = FLOOR, margin: float = MARGIN, min_gain: float = MIN_GAIN) -> np.ndarray:
    d, n, g = (np.asarray(x, dtype="float64") for x in (drift_ab, null, gain))
    return (d >= floor - EPS) & (d - n >= margin - EPS) & (g >= min_gain - EPS)


def gain_share(names, gains) -> pd.Series:
    g = np.asarray(gains, dtype="float64")
    if not g.sum() > 0:
        raise ValueError("no gain in the model's importances")
    return pd.Series(g / g.sum(), index=list(names))


def cell_table(a: pd.DataFrame, b: pd.DataFrame, neighbours: list, cols, gain: pd.Series) -> pd.DataFrame:
    """One airport-month: drift (A vs B), null (the largest A-vs-neighbour drift), excess, gain share, flag."""
    d = drift(a, b, cols)
    n = pd.concat([drift(a, x, cols) for x in neighbours], axis=1).max(axis=1) if neighbours else pd.Series(np.nan, index=d.index)
    g = gain.reindex(d.index).fillna(0.0)
    out = pd.DataFrame({"drift": d, "null": n, "excess": d - n, "gain_share": g})
    out["flag"] = flag(out["drift"], out["null"].fillna(np.inf), out["gain_share"])
    return out


def cells(a: pd.DataFrame, b: pd.DataFrame, *, min_rows: int = MIN_ROWS) -> list:
    """Airports with at least `min_rows` rows on both sides."""
    na, nb = a.ap.value_counts(), b.ap.value_counts()
    return sorted(str(x) for x in na.index if na[x] >= min_rows and nb.get(x, 0) >= min_rows)


def domain_auc(xa: np.ndarray, xb: np.ndarray, *, seed: int = 0) -> float:
    """3-fold cross-validated AUC of a gradient-boosted classifier separating the rows of xa from those of xb."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    x = np.vstack([xa, xb])
    y = np.r_[np.zeros(len(xa)), np.ones(len(xb))]
    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, random_state=seed)
    p = cross_val_predict(clf, x, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def main(argv=None) -> int:  # pragma: no cover - I/O shell around the tested functions (real frames, the model file)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--artifact", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--set", action="append", default=[], dest="overrides")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    import lightgbm as lgb
    from prc.pipeline.fm1 import engine as G, features as F
    out = pathlib.Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to write into the non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    art = ROOT / a.artifact
    cfg = G.as_config(ROOT / "configs" / "fm1.yaml", a.overrides)
    cols = json.loads((art / "design.json").read_text())["matched"]
    enc = F.EncodingTables.from_json((art / "encodings_matched.json").read_text())
    adsb = set(F.adsb_columns(cfg.path("adsb.table_dir")))
    weather = F.stand_ab().load_weather(cfg.path("data.weather_dir"))
    orl = F.read_orl_routes(cfg.path("features.orl_routes_file"))

    def design(raw, serve):
        fr = F.build_frames(raw[G.in_registry(raw, cfg)], serve=serve, cfg=cfg, weather=weather, orl_routes=orl).matched
        x = pd.DataFrame(F.matched_design(fr, enc, cols), columns=cols)
        x["ap"] = fr.ap.to_numpy()
        x["month"] = fr.day.astype(str).str[5:7].astype(int).to_numpy()
        return x

    x25 = {m: design(pd.read_parquet(G.month_file(cfg, m)), False) for m in (12, 1, 2, 6, 7, 8)}
    x26 = design(pd.read_parquet(cfg.path("data.scored_file")), True)
    bst = lgb.Booster(model_file=str(art / "learners" / "base" / "lgb_seed0.txt"))
    gain = gain_share(bst.feature_name(), bst.feature_importance(importance_type="gain"))
    if list(gain.index) != list(cols):
        raise SystemExit("the booster's feature names are not the artefact's design columns")
    del bst
    rows, multi = [], []
    non = [c for c in cols if c not in adsb]
    for m in (1, 7):
        A_all, B_all = x25[m], x26[x26.month == m]
        for apt in cells(A_all, B_all):
            A, B = A_all[A_all.ap == apt], B_all[B_all.ap == apt]
            nb = [x25[k][x25[k].ap == apt] for k in NEIGHBOURS[m]]
            nb = [x for x in nb if len(x) >= MIN_ROWS]
            t = cell_table(A, B, nb, cols, gain).reset_index(names="feature")
            t.insert(0, "cell", f"{apt}|{m}")
            t["adsb"] = t.feature.isin(adsb)
            rows.append(t)
            rec = {"cell": f"{apt}|{m}", "n_2025": int(len(A)), "n_2026": int(len(B)),
                   "auc_year_all": domain_auc(A[cols].to_numpy(), B[cols].to_numpy()),
                   "auc_year_non_adsb": domain_auc(A[non].to_numpy(), B[non].to_numpy())}
            for k, x in zip(NEIGHBOURS[m], [x25[k][x25[k].ap == apt] for k in NEIGHBOURS[m]]):
                if len(x) >= MIN_ROWS:
                    rec[f"auc_month{k}_non_adsb"] = domain_auc(A[non].to_numpy(), x[non].to_numpy())
            multi.append(rec)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    table = pd.concat(rows, ignore_index=True)
    table.to_parquet(out / "drift_table.parquet", index=False)
    pd.DataFrame(multi).to_parquet(out / "domain_auc.parquet", index=False)
    fl = table[table.flag]
    summ = {"n_cells": int(table.cell.nunique()), "n_features": len(cols), "n_non_adsb": len(non),
            "flagged_non_adsb": {f: sorted(g.cell) for f, g in fl[~fl.adsb].groupby("feature")},
            "flagged_adsb": {f: sorted(g.cell) for f, g in fl[fl.adsb].groupby("feature")},
            "rule": {"floor": FLOOR, "margin": MARGIN, "min_gain": MIN_GAIN, "min_rows": MIN_ROWS}}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({"flagged_non_adsb": len(summ["flagged_non_adsb"]), "flagged_adsb": len(summ["flagged_adsb"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
