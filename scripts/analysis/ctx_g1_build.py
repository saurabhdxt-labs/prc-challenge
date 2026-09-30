"""SHIP-CTXG1 build — the frozen G1-only delay-context candidate (plans/PREREG_ship_ctx_g1_2026_09_13.md). NO upload.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/ctx_g1_build.py

Rebuilds v13's `lirf_rules` lane through the pipeline's public functions, proves it equals v13, replaces the fill probability
on the frozen scope only, and writes a candidate submission + per-row scope file + manifest under data/pipeline_runs/.
Every other scored row keeps v13's value byte for byte. Edits nothing in prc/ or scripts/*.py.
"""
from __future__ import annotations

import datetime as dt
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import rome_delay_context as RDC  # noqa: E402
import rome_local_day as RLD  # noqa: E402

PREREG = ROOT / "plans" / "PREREG_ship_ctx_g1_2026_09_13.md"
CONFIG = ROOT / "configs" / "pipeline_adj2.yaml"
V13_SUB = ROOT / "data" / "pipeline_runs" / "v13_20260912T113635Z" / "merry-quicksand_v13.parquet"
V13_SHA = "2ac39110624144257be9a1f20c380803f7f3cf7f3ba4b640f1caf31be65309e7"
RUNS = ROOT / "data" / "pipeline_runs"
SP_LO, SP_HI = 3_600.0, 24_000.0
N_SCOPE_EXPECTED = 288
SUBBANDS = ((3_600.0, 10_800.0), (10_800.0, 24_000.0))
K1_MIN, K3_MIN = 10, 0.95
COMPOSED = ["scripts/analysis/ctx_g1_build.py", "scripts/analysis/rome_delay_context.py", "scripts/rome_fill.py",
            "scripts/rome_dateslip.py", "scripts/rome_bandfloor.py", "scripts/rome_local_day.py", "scripts/build_submission.py",
            "scripts/stratum_fold.py", "prc/pipeline/lanes.py", "prc/pipeline/ingest.py", "prc/pipeline/config.py",
            "prc/pipeline/legacy.py"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scope_mask(frame: pd.DataFrame) -> np.ndarray:
    """The frozen scope on a LIRF-unmatched lane frame: 3,600 <= sp < 24,000 and Rome local dayoff 0."""
    sp = frame.sp.to_numpy(dtype="float64")
    sp_clock = (pd.to_datetime(frame.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(frame.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    if not np.allclose(sp, sp_clock.to_numpy(), rtol=0, atol=1e-6):
        raise AssertionError("sp is not MVT − SCHED in seconds")
    ld = RLD.local_dayoff(frame.SCHED_TIME_UTC_mvt, frame.MVT_TIME_UTC_mvt)
    return (sp >= SP_LO) & (sp < SP_HI) & (np.asarray(ld) == 0)


def splice(v13: pd.DataFrame, ids, values) -> pd.DataFrame:
    """v13 with `values` written on `ids` only. Refuses unknown ids, duplicates, non-finite or sub-1 values, and verifies every
    other row is unchanged (B4)."""
    ids = np.asarray(ids, dtype="float64")
    values = np.asarray(values, dtype="float64")
    if len(ids) != len(values) or len(np.unique(ids)) != len(ids):
        raise AssertionError("scope ids are not unique / not aligned with values")
    if not (np.isfinite(values).all() and (values >= 1).all() and np.array_equal(values, np.rint(values))):
        raise AssertionError("scope values must be finite integers >= 1")
    out = v13.copy()
    pos = pd.Index(out.MVT_ID_mvt.to_numpy(dtype="float64")).get_indexer(ids)
    if (pos < 0).any():
        raise AssertionError(f"{int((pos < 0).sum())} scope ids are not in the v13 submission")
    col = out.TAXITIME_SEC_mvt.to_numpy().copy()
    col[pos] = values.astype(col.dtype)
    out["TAXITIME_SEC_mvt"] = col
    outside = np.ones(len(out), dtype=bool)
    outside[pos] = False
    if not np.array_equal(out.TAXITIME_SEC_mvt.to_numpy()[outside], v13.TAXITIME_SEC_mvt.to_numpy()[outside]):
        raise AssertionError("B4: a row outside scope changed")
    if list(out.columns) != list(v13.columns) or out.dtypes.to_dict() != v13.dtypes.to_dict() \
            or not np.array_equal(out.MVT_ID_mvt.to_numpy(), v13.MVT_ID_mvt.to_numpy()):
        raise AssertionError("B4: row set, order or schema differs from v13")
    return out


def support_checks(g1_2025: pd.DataFrame, scope_2026: pd.DataFrame) -> dict:
    """K1–K3 per month. Frames need month, sp and the three context columns."""
    res = {}
    for mo in (1, 7):
        a, b = g1_2025[g1_2025.month == mo], scope_2026[scope_2026.month == mo]
        k1 = {f"{lo:.0f}-{hi:.0f}": {"n2026": int(((b.sp >= lo) & (b.sp < hi)).sum()), "n2025": int(((a.sp >= lo) & (a.sp < hi)).sum())}
              for lo, hi in SUBBANDS}
        for v in k1.values():
            v["pass"] = v["n2025"] >= K1_MIN
        k2 = {}
        for c in RDC.CTX:
            p10, p90 = np.nanpercentile(a[c], [10, 90]) if len(a) else (np.nan, np.nan)
            med = float(np.nanmedian(b[c])) if len(b) else float("nan")
            k2[c] = {"median2026": med, "p10_2025": float(p10), "p90_2025": float(p90), "pass": bool(p10 <= med <= p90)}
        res[int(mo)] = {"n2026": int(len(b)), "n2025": int(len(a)), "K1": k1, "K2": k2}
    k3 = {}
    for c in RDC.CTX:
        p1, p99 = np.nanpercentile(g1_2025[c], [1, 99])
        share = float(((scope_2026[c] >= p1) & (scope_2026[c] <= p99)).mean()) if len(scope_2026) else float("nan")
        k3[c] = {"share_inside": share, "p1_2025": float(p1), "p99_2025": float(p99), "pass": bool(share >= K3_MIN)}
    res["K3"] = k3
    return res


def main() -> int:
    from prc.pipeline import config as C, ingest, lanes, legacy
    st = legacy.stratum()
    log = lambda m: print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
    guards = {}
    if sha256(V13_SUB) != V13_SHA:
        raise AssertionError("B1: the v13 submission file does not match V13_FREEZE")
    guards["B1_v13_sha"] = True
    cfg = C.load_config(CONFIG)
    lanes.check_pins(cfg)
    scored = ingest.ingest_scored(cfg)
    routing = lanes.route(scored.scored, cfg)
    unm_rows = scored.scored.unmatched.to_numpy(dtype=bool)
    scored_unm = scored.scored[unm_rows]
    inputs = lanes.stratum_inputs(cfg)
    parts = lanes.s1_parts(inputs, scored_unm, cfg.seeds)
    lane = cfg.lanes["lirf_rules"]
    rows = routing.lane[unm_rows] == lane.id
    res = lanes.run_rules_lane(lane, cfg.airports["LIRF"], rows, scored_unm, parts, inputs, cfg.seeds)
    log(f"rebuilt lirf_rules: {len(res.ids)} rows")
    v13 = pd.read_parquet(V13_SUB)
    v13v = v13.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(res.ids).to_numpy(dtype="float64")
    if not np.array_equal(v13v, res.values):
        raise AssertionError(f"B2: rebuilt lirf_rules differs from v13 on {int((v13v != res.values).sum())} rows")
    guards["B2_lane_equals_v13_rows"] = int(len(res.ids))
    te = scored_unm[rows].reset_index(drop=True)
    scope = scope_mask(te)
    if int(scope.sum()) != N_SCOPE_EXPECTED:
        raise AssertionError(f"scope has {int(scope.sum())} rows, registered {N_SCOPE_EXPECTED}")
    sp = te.sp.to_numpy(dtype="float64")
    nf = np.asarray(st.bs.nf_hybrid(parts["nf_cells"], parts["nf_fit"]), dtype="float64")[rows]
    p_c = np.asarray(res.info["p"], dtype="float64")
    v_c = np.rint(st.rf.mixture(p_c, sp, nf))
    stage = res.provenance.set_index("MVT_ID_mvt").reindex(res.ids)
    stage_col = [c for c in stage.columns if "stage" in c][0]
    if not np.array_equal(v_c[scope], res.values[scope]) or not (stage[stage_col].to_numpy()[scope] == "R2").all():
        raise AssertionError("B3: nf or stage on scope rows is not what v13 used")
    guards["B3_nf_and_stage"] = True
    in_g = st.ds.segment_mask(te)
    if in_g[scope].any() or (sp[scope] >= st.rb.LO).any() or (sp[scope] >= float(RLD.SP_LO)).any():
        raise AssertionError("B6: a later chain rule could act on a scope row")
    guards["B6_chain_rules_cannot_act"] = True
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"]
    d25 = RDC.departures_frame([pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))])
    ctx25 = RDC.context_features(d25.MVT_ID_mvt, d25.t_s, d25.ms)
    d26 = RDC.departures_frame([pd.read_parquet(cfg.paths.scored_file, columns=cols)])
    ctx26 = RDC.context_features(d26.MVT_ID_mvt, d26.t_s, d26.ms)
    guards["B5_ctx_median_2025"], guards["B5_ctx_median_2026"] = RDC.check_context_scale(ctx25), RDC.check_context_scale(ctx26)
    tr = st.rf.rome_rows(inputs.train_unm, inputs.train_lirf,
                         sorted(set(inputs.train_unm.month) | set(inputs.train_lirf.month)))
    tr = tr.merge(ctx25, on="MVT_ID_mvt", how="left")
    te_c = te.merge(ctx26, on="MVT_ID_mvt", how="left")
    if tr[RDC.CTX[0]].isna().any() or te_c[RDC.CTX[0]].isna().any():
        raise AssertionError("a training or scope row has no context")
    p_chk = np.mean([st.rf.fit_predict_fill(tr, te_c, seed=s) for s in cfg.seeds], axis=0)
    if not np.allclose(p_chk, p_c, rtol=0, atol=1e-9):
        raise AssertionError("control refit on the context-merged frames does not reproduce the lane's p")
    p_d = np.mean([st.rf.fit_predict_fill(tr, te_c, seed=s, extra_num=RDC.CTX) for s in cfg.seeds], axis=0)
    v_d = np.rint(st.rf.mixture(p_d, sp, nf))
    ids = te.MVT_ID_mvt.to_numpy(dtype="float64")
    cand = splice(v13, ids[scope], v_d[scope])
    guards["B4_outside_scope_identical"] = True
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS / f"ctxg1_{stamp}"
    out_dir.mkdir(parents=False, exist_ok=False)
    sub_path, rows_path = out_dir / "merry-quicksand_ctxg1.parquet", out_dir / "ctxg1_scope_rows.parquet"
    cand.to_parquet(sub_path, index=False)
    month = pd.to_datetime(te.MVT_TIME_UTC_mvt, utc=True).dt.month.to_numpy()
    scope_rows = pd.DataFrame({"MVT_ID_mvt": ids, "month": month, "sp": sp, "p_C": p_c, "p_D": p_d, "nf": nf,
                               "v13": res.values, "candidate": np.where(scope, v_d, res.values), "in_scope": scope})
    scope_rows = scope_rows.merge(ctx26, on="MVT_ID_mvt", how="left")
    scope_rows.to_parquet(rows_path, index=False)
    u = inputs.train_unm
    u = u[u.ADEP_mvt == "LIRF"]
    g25 = u[scope_mask(u)].merge(ctx25, on="MVT_ID_mvt", how="left")
    s26 = scope_rows[scope_rows.in_scope]
    checks = support_checks(g25, s26)
    dv = np.abs(v_d[scope] - res.values[scope])
    checks["K4"] = {"n_changed": int((dv > 0).sum()), "n_scope": int(scope.sum()), "mean_abs_change_s": float(dv.mean()),
                    "p95_abs_change_s": float(np.percentile(dv, 95)), "mean_dp": float((p_d - p_c)[scope].mean()),
                    "sum_signed_change_s": float((v_d[scope] - res.values[scope]).sum())}
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    manifest = {"tag": "CTXG1", "prereg": str(PREREG.relative_to(ROOT)), "prereg_sha256": sha256(PREREG), "head": head,
                "created_utc": stamp, "seeds": list(cfg.seeds), "guards": guards, "checks": checks,
                "outputs": {str(sub_path.relative_to(ROOT)): sha256(sub_path), str(rows_path.relative_to(ROOT)): sha256(rows_path)},
                "inputs": {str(p): sha256(ROOT / p) for p in ["configs/pipeline_adj2.yaml", "data/raw/ranking.parquet",
                           "data/cache_rome/unm.parquet", "data/cache_rome/lirf.parquet",
                           str(V13_SUB.relative_to(ROOT))]},
                "code": {p: sha256(ROOT / p) for p in COMPOSED},
                "evidence_statement": ("~+353 projected board MSE on G1 (RDC run 2; unchanged through the shipped chain, RDX); "
                                       "G1 preregistered; routing to G1 only chosen POST-HOC after RDC C6; RDX routing "
                                       "precondition NOT met as written; no independent confirmation of the final candidate; "
                                       "an upload is deployment under acknowledged validation uncertainty."),
                "upload": "NOT AUTHORISED — separate owner decision"}
    (out_dir / "ctxg1_manifest.json").write_text(json.dumps(manifest, indent=1, default=float))
    print(json.dumps({"out_dir": str(out_dir.relative_to(ROOT)), "guards": guards, "checks": checks}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
