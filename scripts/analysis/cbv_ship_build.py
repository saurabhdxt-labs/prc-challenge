"""SHIP-CBV build — splice the CatBoost 0.5 blend into v13 on the no-ADS-B matched rows (plans/PREREG_ship_cbv_2026_09_13.md).
NO upload.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/cbv_ship_build.py

Reads the worker's 2026 predictions (data/cache_stand/cbv_ship_C_2026.npy, aligned to the holdout rows of cbv_ship_input.parquet),
applies rint(0.5·max(proxy − C, 1) + 0.5·v13) on rows whose v13 stage is exactly `lightgbm:delta:mean3`, keeps every other row
byte-identical, runs guards S1/S3–S6 and transfer checks T1–T3, and writes data/pipeline_runs/scbv_<UTC stamp>/.
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import ctx_g1_build as CB  # noqa: E402  (splice: the tested outside-scope identity check)

PREREG = ROOT / "plans" / "PREREG_ship_cbv_2026_09_13.md"
V13_DIR = ROOT / "data" / "pipeline_runs" / "v13_20260912T113635Z"
V13_SUB, V13_PROV = V13_DIR / "merry-quicksand_v13.parquet", V13_DIR / "merry-quicksand_v13.provenance.parquet"
V13_SHA = CB.V13_SHA
SHIP_INPUT = ROOT / "data" / "cache_stand" / "cbv_ship_input.parquet"
PRED = ROOT / "data" / "cache_stand" / "cbv_ship_C_2026.npy"
FOLD_OOF = ROOT / "data" / "adsb" / "v2" / "v13_matched_oof.parquet"
FOLD_CB_INPUT = ROOT / "data" / "cache_stand" / "catboost_native_input.parquet"
FOLD_CB_PRED = ROOT / "data" / "cache_stand" / "catboost_native_delta.npy"
RUNS = ROOT / "data" / "pipeline_runs"
SCOPE_STAGE = "lightgbm:delta:mean3"
MATCHED_STAGES = ("lightgbm:delta:mean3", "lightgbm:delta:mean3+adsb")
STAGE_COUNTS = {"lightgbm:delta:mean3+adsb": 209_460, "lightgbm:delta:mean3": 130_091}
BLEND = 0.5
MIN_ROWS, T1_BAND, T2_MAX, T3_PP = 500, (0.7, 1.3), 30.0, 10.0


def taxi(proxy, delta_hat) -> np.ndarray:
    return np.maximum(np.asarray(proxy, dtype="float64") - np.asarray(delta_hat, dtype="float64"), 1.0)


def blend_values(taxi_c, v13_values) -> np.ndarray:
    """rint(0.5·taxi(C) + 0.5·v13) — v13 is F's rounded taxi (the registered approximation)."""
    v = np.rint(BLEND * np.asarray(taxi_c, dtype="float64") + (1 - BLEND) * np.asarray(v13_values, dtype="float64"))
    if not (np.isfinite(v).all() and (v >= 1).all()):
        raise AssertionError("S5: a blended value is non-finite or below 1")
    return v


def holdout_predictions(ship: pd.DataFrame, pred) -> pd.DataFrame:
    """Align the worker's predictions (holdout rows in input order) to MVT ids. S4: count and finiteness."""
    h = ship[ship.holdout.to_numpy(bool)].reset_index(drop=True)
    pred = np.asarray(pred, dtype="float64")
    if len(pred) != len(h):
        raise AssertionError(f"S4: {len(pred)} predictions for {len(h)} holdout rows")
    if not np.isfinite(pred).all():
        raise AssertionError("S4: non-finite prediction")
    if not pd.Index(h.MVT_ID_mvt).is_unique:
        raise AssertionError("S4: duplicate holdout id")
    return pd.DataFrame({"MVT_ID_mvt": h.MVT_ID_mvt.to_numpy(dtype="float64"), "proxy": h.score_proxy.to_numpy(dtype="float64"),
                         "C": pred, "taxi_C": taxi(h.score_proxy, pred)})


def scope_frame(prov: pd.DataFrame, v13: pd.DataFrame, hold: pd.DataFrame) -> pd.DataFrame:
    """S1/S3: stage counts as registered; holdout ids == v13's matched-lane ids; returns scope rows with v13 value and taxi(C)."""
    counts = prov.stage.value_counts().to_dict()
    for k, n in STAGE_COUNTS.items():
        if counts.get(k) != n:
            raise AssertionError(f"S1: stage {k} has {counts.get(k)} rows, registered {n}")
    matched = set(prov.MVT_ID_mvt[prov.stage.isin(MATCHED_STAGES)].astype(float))
    if set(hold.MVT_ID_mvt) != matched:
        raise AssertionError("S3: 2026 holdout ids are not exactly v13's matched-lane rows")
    ids = prov.MVT_ID_mvt[prov.stage == SCOPE_STAGE].astype(float).to_numpy()
    s = pd.DataFrame({"MVT_ID_mvt": ids}).merge(hold, on="MVT_ID_mvt", how="left", validate="one_to_one")
    s = s.merge(v13.rename(columns={"TAXITIME_SEC_mvt": "v13"}), on="MVT_ID_mvt", how="left", validate="one_to_one")
    if s.taxi_C.isna().any() or s.v13.isna().any():
        raise AssertionError("S3: a scope row lacks a prediction or a v13 value")
    return s


def transfer_checks(fold: pd.DataFrame, s26: pd.DataFrame) -> dict:
    """T1–T3 per airport. Frames need ap, d (= taxi(C) − taxi(F)); change = |rint(0.5·d)|."""
    out = {}
    for ap in sorted(set(fold.ap) | set(s26.ap)):
        a, b = fold[fold.ap == ap], s26[s26.ap == ap]
        row = {"n_fold": int(len(a)), "n_2026": int(len(b))}
        if len(a) >= MIN_ROWS and len(b) >= MIN_ROWS:
            sd_a, sd_b = float(a.d.std()), float(b.d.std())
            ratio = sd_b / sd_a if sd_a > 0 else float("nan")
            sh_a = float((np.abs(np.rint(0.5 * a.d)) > 60).mean() * 100)
            sh_b = float((np.abs(np.rint(0.5 * b.d)) > 60).mean() * 100)
            row.update(sd_fold=sd_a, sd_2026=sd_b, T1_ratio=ratio, T1=bool(T1_BAND[0] <= ratio <= T1_BAND[1]),
                       mean_fold=float(a.d.mean()), mean_2026=float(b.d.mean()),
                       T2=bool(abs(b.d.mean() - a.d.mean()) <= T2_MAX),
                       share_gt60_fold_pct=sh_a, share_gt60_2026_pct=sh_b, T3=bool(abs(sh_b - sh_a) <= T3_PP))
        else:
            row["reported_only"] = True
        out[ap] = row
    return out


def sha256(p: Path) -> str:
    return CB.sha256(p)


def main() -> int:
    if CB.sha256(V13_SUB) != V13_SHA:
        raise AssertionError("S1: v13 submission sha differs from V13_FREEZE")
    v13 = pd.read_parquet(V13_SUB)
    prov = pd.read_parquet(V13_PROV, columns=["MVT_ID_mvt", "airport", "stage"])
    ship = pd.read_parquet(SHIP_INPUT, columns=["MVT_ID_mvt", "score_proxy", "holdout", "delta"])
    if np.isfinite(ship.delta.to_numpy()[ship.holdout.to_numpy(bool)]).any():
        raise AssertionError("S3: a 2026 holdout row carries a label")
    hold = holdout_predictions(ship, np.load(PRED))
    s = scope_frame(prov, v13, hold)
    s = s.merge(prov[["MVT_ID_mvt", "airport"]].astype({"MVT_ID_mvt": float}), on="MVT_ID_mvt", how="left")
    s["candidate"] = blend_values(s.taxi_C, s.v13)
    cand = CB.splice(v13, s.MVT_ID_mvt.to_numpy(), s.candidate.to_numpy())
    # fold-A reference for T1–T3: P rows (finite fold, v13 == F), C from the screen's fold-A record
    o = pd.read_parquet(FOLD_OOF, columns=["row", "ap", "fold", "proxy", "F", "v13_delta"])
    h = pd.read_parquet(FOLD_CB_INPUT, columns=["row", "holdout"])
    h = h[h.holdout].reset_index(drop=True).assign(Cf=np.load(FOLD_CB_PRED))
    o = o.merge(h[["row", "Cf"]], on="row", validate="one_to_one")
    p = o[np.isfinite(o.fold.to_numpy(float)) & (o.v13_delta == o.F)]
    fold = pd.DataFrame({"ap": p.ap.to_numpy(), "d": taxi(p.proxy, p.Cf) - taxi(p.proxy, p.F)})
    s26 = pd.DataFrame({"ap": s.airport.to_numpy(), "d": s.taxi_C.to_numpy() - s.v13.to_numpy()})
    checks = transfer_checks(fold, s26)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS / f"scbv_{stamp}"
    out_dir.mkdir(parents=False, exist_ok=False)
    sub_path, rows_path = out_dir / "merry-quicksand_scbv.parquet", out_dir / "scbv_scope_rows.parquet"
    cand.to_parquet(sub_path, index=False)
    s.to_parquet(rows_path, index=False)
    back = pd.read_parquet(sub_path).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(s.MVT_ID_mvt).to_numpy(dtype="float64")
    if not np.array_equal(back, s.candidate.to_numpy()):
        raise AssertionError("S6: the written candidate does not reproduce the scope values")
    changed = s.candidate.to_numpy() != s.v13.to_numpy()
    info = json.loads(PRED.with_suffix(".json").read_text())
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    manifest = {"tag": "SCBV", "prereg": str(PREREG.relative_to(ROOT)), "prereg_sha256": sha256(PREREG), "head": head,
                "created_utc": stamp, "worker_info": info,
                "guards": {"S1": True, "S3": True, "S4": True, "S5_outside_scope_identical": True, "S6": True},
                "scope": {"n": int(len(s)), "n_changed": int(changed.sum()),
                          "mean_abs_change_s": float(np.abs(s.candidate - s.v13).mean()),
                          "p95_abs_change_s": float(np.percentile(np.abs(s.candidate - s.v13), 95))},
                "transfer_checks": checks,
                "outputs": {str(sub_path.relative_to(ROOT)): sha256(sub_path), str(rows_path.relative_to(ROOT)): sha256(rows_path)},
                "inputs": {str(p_.relative_to(ROOT)): sha256(p_) for p_ in (SHIP_INPUT, PRED, V13_SUB, V13_PROV, FOLD_OOF,
                                                                            FOLD_CB_INPUT, FOLD_CB_PRED)},
                "code": {p_: sha256(ROOT / p_) for p_ in ("scripts/analysis/cbv_ship_build.py", "scripts/analysis/cbv_ship_input.py",
                                                          "scripts/catboost_native_worker.py", "scripts/catboost_native.py",
                                                          "scripts/analysis/ctx_g1_build.py")},
                "evidence_statement": ("CBV SHORTLIST +223 2026-eq (fold A, months agree); no confirmation fold — the board is the "
                                       "first independent test; single-seed C_delta; blend on rounded v13 F (<= 0.25 s); "
                                       "matched-lane transfer history 0.96-1.17x."),
                "upload": "NOT AUTHORISED — separate owner decision"}
    (out_dir / "scbv_manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    print(json.dumps({"out_dir": str(out_dir.relative_to(ROOT)), "scope": manifest["scope"], "transfer_checks": checks,
                      "worker": {k: info.get(k) for k in ("best_iteration", "n_ref", "tree_count", "wall_s", "peak_rss_gb")}},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
