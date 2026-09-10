"""Reproduce the September 10 fresh review from saved predictions; no model fitting.

    OMP_NUM_THREADS=1 /opt/homebrew/bin/python3.11 -B tools/audit_path_246.py

Weighted fold MSE is a planning unit, not an observed leaderboard decomposition.
Label-selected oracles are diagnostics, never deployable candidate gains.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache_stand"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-score", type=float, default=285.8013)
    parser.add_argument("--target", type=float, default=246.0)
    args = parser.parse_args()
    paths = []

    def read(path, columns=None):
        paths.append(path)
        return pd.read_parquet(path, columns=columns)

    f = read(CACHE / "fold_preds_queue_order.parquet")
    y = f.y.to_numpy(dtype=float)
    pred = np.maximum(f.proxy.to_numpy() - f.treatment.to_numpy(), 1.0)
    se = (y - pred) ** 2
    known_path = ROOT / "reports/lgbm_fold_queue_order.json"
    paths.append(known_path)
    known = json.loads(known_path.read_text())["arms"]["treatment"]["rmse"]
    assert len(f) == 339015 and f.row.is_unique
    assert np.isclose(np.sqrt(se.mean()), known, rtol=0, atol=1e-10)

    raw = read(ROOT / "data/raw/ranking.parquet",
               ["MVT_ID_mvt", "PHASE_mvt", "AOBT_3_flt"])
    deps = raw[raw.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt")
    matched = deps.AOBT_3_flt.notna()
    v6 = read(ROOT / "submissions/merry-quicksand_v6.parquet").set_index("MVT_ID_mvt")
    v7 = read(ROOT / "submissions/merry-quicksand_v7.parquet").set_index("MVT_ID_mvt")
    for sub in (v6, v7):
        assert sub.index.is_unique and len(sub) == len(deps)
        assert set(sub.index) == set(deps.index)
    changed = (v6.loc[deps.index, "TAXITIME_SEC_mvt"] !=
               v7.loc[deps.index, "TAXITIME_SEC_mvt"])
    assert not changed[matched].any(), "v7 changed the matched submission component"
    weight = float(matched.mean())
    scale = weight / len(f)
    fill = (np.abs(f.y - f.sp) <= 60).to_numpy()
    tail = (np.abs(f.delta) > 600).to_numpy()
    masks = {"schedule_fill": fill, "nonfill_abs_delta_gt600": ~fill & tail,
             "other_nonfill": ~fill & ~tail}
    assert np.all(np.stack(list(masks.values())).sum(axis=0) == 1)

    def describe(mask):
        return {"n": int(mask.sum()), "rmse": float(np.sqrt(se[mask].mean())),
                "weighted_fold_mse": float(se[mask].sum() * scale)}

    result = {
        "scope": "Saved predictions only; no candidate fitted or submitted.",
        "board_score_source": "reports/MSE_LEDGER.md, v7 recorded 2026-09-10 13:14; not live refreshed",
        "board_score": args.board_score, "target": args.target,
        "mse_gap": args.board_score ** 2 - args.target ** 2,
        "n_scored": len(deps), "n_scored_matched": int(matched.sum()),
        "matched_weight": weight, "v7_changed_rows": int(changed.sum()),
        "v7_matched_unchanged": True, "matched_reference_rmse": float(np.sqrt(se.mean())),
        "matched_weighted_fold_mse": float(weight * se.mean()),
        "partitions": {k: describe(m) for k, m in masks.items()},
        "airports": {},
        "months": {str(m): describe((f.month == m).to_numpy()) for m in sorted(f.month.unique())},
    }
    for airport in sorted(f.ap.unique()):
        a = (f.ap == airport).to_numpy()
        result["airports"][airport] = {
            **describe(a), "partitions": {k: describe(a & m) for k, m in masks.items()}}

    experts = {"F": pred, "schedule": np.maximum(f.sp.to_numpy(), 1),
               "NM": np.maximum(f.proxy.to_numpy(), 1)}
    for filename, column, name in [
        ("fold_preds_queue_ytarget.parquet", "Y", "Y"),
        ("fold_preds_queue_weather.parquet", "treatment", "W"),
    ]:
        other = read(CACHE / filename)
        for key in ("row", "month", "ap", "y", "proxy", "sp"):
            assert np.array_equal(f[key].to_numpy(), other[key].to_numpy()), key
        # Both records store DELTA even though one learner was trained on y.
        experts[name] = np.maximum(other.proxy.to_numpy() - other[column].to_numpy(), 1)
    errors = {k: (v - y) ** 2 for k, v in experts.items()}
    result["expert_diagnostics"] = {
        k: {"rmse": float(np.sqrt(e.mean())),
            "net_weighted_fold_mse_gain": float((se - e).sum() * scale),
            "label_oracle_switch_gain": float(np.maximum(se - e, 0).sum() * scale)}
        for k, e in errors.items()}
    result["label_oracle_best_of_five_gain"] = float(
        (se - np.stack(list(errors.values())).min(axis=0)).sum() * scale)
    result["oracle_limit"] = (
        "Uses true labels to choose a prediction per row. Not a trained gate, forecast, "
        "or bound on arbitrary models or convex ensembles. Individual gains overlap.")

    u = read(CACHE / "stratum_fold_v7_preds.parquet")
    r = read(CACHE / "rome_dateslip_preds.parquet")
    u = u[u.fold == "A"].merge(
        r.loc[r.fold == "A", ["MVT_ID_mvt", "y", "sp", "R2"]],
        on="MVT_ID_mvt", how="left", suffixes=("", "_r"), validate="one_to_one")
    replaced = u.R2.notna()
    for key in ("y", "sp"):
        assert np.array_equal(u.loc[replaced, key], u.loc[replaced, key + "_r"])
    assert len(u) == 5321 and replaced.sum() == 397
    use = (u.y - u.R2.fillna(u.S1)) ** 2
    result["unmatched_fold_A_after_R2"] = {
        "n": len(u), "rmse": float(np.sqrt(use.mean())),
        "two_largest_errors_share_sse": float(use.nlargest(2).sum() / use.sum()),
        "all_rows_fold_rmse": float(np.sqrt((se.sum() + use.sum()) / (len(f) + len(u)))),
        "warning": "Fold totals and extreme-row gains are not leaderboard forecasts.",
    }

    residual = args.board_score ** 2 - weight * se.mean()
    result["conditional_scenarios"] = {
        "assumption": "Assume current board matched RMSE equals 222.563 fold RMSE. Unverified.",
        "implied_unmatched_mse_contribution": float(residual),
        "matched_only_required_rmse": float(np.sqrt((args.target ** 2 - residual) / weight)),
        "targets": [{"matched_rmse": m,
                     "unmatched_required_rmse": float(np.sqrt((args.target ** 2 - weight*m*m) / (1-weight))),
                     "unmatched_mse_gain_needed": float(residual - (args.target ** 2 - weight*m*m))}
                    for m in (205, 195, 190, 185)],
    }
    result["score_arithmetic"] = {
        str(gain): float(np.sqrt(args.board_score ** 2 - gain))
        for gain in (3000, 5000, 10000, 15000, 20000)}
    result["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    output = ROOT / "reports/fresh_path_246_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in (
        "mse_gap", "matched_reference_rmse", "partitions",
        "unmatched_fold_A_after_R2", "conditional_scenarios")}, indent=2))
    print(f"Audit written to {output}")


if __name__ == "__main__":
    main()
