"""GID — where can a regime gate commit? (reports/GATE_INFORMATION_2026_09_11.md, Part 1 written before any number.)

    python3.11 -B scripts/gate_info.py      -> reports/gate_info.json

Descriptive, no fitting: reads arm REG's stamped holdout record and the holdout rows' stand-cache features.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import nm_param_diag as N  # noqa: E402  (holdout_days, to_board)
from prc.pipeline import scoring as SC  # noqa: E402

REC = ROOT / "data" / "cache_stand" / "regime_experts_preds.parquet"
OUT = ROOT / "reports" / "gate_info.json"
EARLY = 2
COMMIT_P = 0.5
TOP_FRAC = 0.05
TOP_DAYS = 5


def auc(score, pos) -> float:
    """Mann–Whitney AUC with ties averaged; NaN scores are dropped; NaN when a class is empty."""
    s, p = np.asarray(score, dtype="float64"), np.asarray(pos, dtype=bool)
    ok = np.isfinite(s)
    s, p = s[ok], p[ok]
    n1, n0 = int(p.sum()), int((~p).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[p].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def top_precision(score, pos, frac: float) -> float:
    s, p = np.asarray(score, dtype="float64"), np.asarray(pos, dtype=bool)
    k = max(1, int(round(frac * len(s))))
    idx = np.argsort(-s, kind="mergesort")[:k]
    return float(p[idx].mean())


def committed(p_early, pos, thr: float = COMMIT_P) -> dict:
    m = np.asarray(p_early, dtype="float64") >= thr
    pos = np.asarray(pos, dtype=bool)
    return {"n_rows": int(m.sum()), "precision": float(pos[m].mean()) if m.any() else float("nan"),
            "early_rows_covered": int((m & pos).sum()), "share_of_early": float((m & pos).sum() / pos.sum()) if pos.any() else float("nan")}


def day_concentration(day, pos, k: int = TOP_DAYS) -> float:
    """Share of the positive rows that fall on the k days holding the most of them."""
    pos = np.asarray(pos, dtype=bool)
    if not pos.any():
        return float("nan")
    c = pd.Series(np.asarray(day)[pos]).value_counts()
    return float(c.iloc[:k].sum() / pos.sum())


def load_features(rows: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data" / "cache_stand" / "training_2025-*.parquet")))
    counts = [pq.ParquetFile(f).metadata.num_rows for f in files]
    start = np.concatenate([[0], np.cumsum(counts)])
    parts = []
    for m in (1, 7):
        f = pd.read_parquet(files[m - 1])
        sel = rows[(rows >= start[m - 1]) & (rows < start[m])] - start[m - 1]
        parts.append(f.iloc[sel])
    feat = pd.concat(parts, ignore_index=True)
    if not np.array_equal(feat.y.to_numpy(), y):
        raise AssertionError("features do not align with the REG record")
    return feat


def main() -> int:
    rec, meta = SC.read_record(REC, "taxi_time", tag="REG")
    rec = rec.sort_values("row", kind="mergesort").reset_index(drop=True)
    y, pos = rec.y.to_numpy(), rec.regime.to_numpy() == EARLY
    day = N.holdout_days(rec.row.to_numpy(), rec.month.to_numpy(), y)
    n = len(rec)
    g_reg = (y - rec.BASE) ** 2 - (y - rec.REG) ** 2
    g_or = (y - rec.BASE) ** 2 - (y - rec.ORACLE_GATE_NOT_A_RESULT) ** 2
    d1 = {}
    for a, s in rec.groupby("ap"):
        m = (rec.ap == a).to_numpy()
        pe, ps = s.p_early.to_numpy(), pos[m]
        d1[a] = {"n": int(m.sum()), "early_share": float(ps.mean()), "auc": auc(pe, ps),
                 "top5pct_precision": top_precision(pe, ps, TOP_FRAC), "commit": committed(pe, ps),
                 "early_gain_reg": N.to_board(float(g_reg[m & pos].sum()), n),
                 "early_gain_oracle": N.to_board(float(g_or[m & pos].sum()), n),
                 "day_top5_share": day_concentration(day[m], ps)}
    d2 = committed(rec.p_early.to_numpy(), pos)
    feat = load_features(rec.row.to_numpy(), y)
    num = [c for c in feat.columns if pd.api.types.is_numeric_dtype(feat[c]) and c not in ("y", "delta", "month")]
    d4 = {"pooled": {c: auc(feat[c].to_numpy(), pos) for c in num}}
    for a in sorted(d1, key=lambda k: -d1[k]["early_gain_oracle"])[:2]:
        m = (rec.ap == a).to_numpy()
        d4[a] = {c: auc(feat[c].to_numpy()[m], pos[m]) for c in num}
    def strongest(tab, k=12):
        return sorted(((c, v) for c, v in tab.items() if np.isfinite(v)), key=lambda kv: -abs(kv[1] - 0.5))[:k]
    res = {"tag": "GID", "source": str(REC.relative_to(ROOT)), "n": n, "d1_per_airport": d1, "d2_commit_pooled": d2,
           "d4_strongest": {k: strongest(v) for k, v in d4.items()}, "d4_all": d4}
    OUT.write_text(json.dumps(res, indent=1, default=float))
    print("D2 pooled commit", d2)
    for a, r in sorted(d1.items(), key=lambda kv: -kv[1]["early_gain_oracle"]):
        c = r["commit"]
        print(f"{a}: early {r['early_share']:.3f} auc {r['auc']:.3f} top5% prec {r['top5pct_precision']:.2f} "
              f"commit n {c['n_rows']} prec {c['precision']:.2f} covers {c['share_of_early']:.2f} | "
              f"oracle {r['early_gain_oracle']:.0f} reg {r['early_gain_reg']:.0f} | top5 days {r['day_top5_share']:.2f}")
    for k, v in res["d4_strongest"].items():
        print(k, [(c, round(x, 3)) for c, x in v])
    return 0


if __name__ == "__main__":
    sys.exit(main())
