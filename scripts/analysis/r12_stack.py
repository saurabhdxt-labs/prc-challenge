"""R12 stack integration — v34's downstream matched stages recomposed on an R12 out-of-fold F, and served for 2026.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/r12_stack.py oof --val data/r12/<val dir> --tables data/r12/tables_<stamp>
        --arms r12,r12_noeobt,janjul_adsb --out data/r12/<stack dir> --rss-limit-gib 6 [--smoke]
    /opt/homebrew/bin/python3.11 -B scripts/analysis/r12_stack.py serve --stack data/r12/<stack dir> --arm r12
        --f26 data/r12/<refit dir>/predictions_2026.parquet --tables data/r12/tables_<stamp> --out data/r12/<dir> ...
    (heavy runs go through scripts/analysis/r12_launch.py --step stack)

Why: a single FW-type model scored 47,755 vs the composed incumbent's 42,046 bmse on fold A (CLN step 0), so a single
R12 model alone would lose to v34. The R12 holdout fit (fit on ten months, predict Jan + Jul) is exactly the kind of
out-of-fold base F that v34's ADS-B residual stack is trained on (v34 step 2 trained it on step 1's F-NSC fold record).

oof, per arm (NO shared module is edited; every stage is called through its public entry point):
  record   the val run's Jan + Jul test rows as a v34 fold record (row, month, ap, y, proxy, delta, treatment = the arm's
           raw delta), rows = global cache rows (the fold records' layout), delta = proxy - y exactly (as in the records)
  table    the R12 (TRC-fixed) training table for those rows in the v4 layout (row, MVT_ID_mvt, FEATURES, coverage)
  frame    v34_adsb_stack.load_frame(record, table, the Jan / Jul raw months): day, hour, day folds
  stack    adsb_stack.train_artifacts (models_v4's 33 feats, seeds 0/1/2, 5 day folds, ship learner B; the MLP half in
           scripts/adsb_nn_worker.py subprocesses, BC-5), v34_adsb_stack.transfer (ADN.7), adsb_ship.gate_table on the
           R12 tables' joined shares with models_v4's LIRF override, written into the models' MANIFEST as step 2 does
  OOF      adsb_stack.predict_residual(frame, models, oof=True)
  compose  on PIPE_M's 334,132 rows (rcr2_build.load_incumbent + cln_build's PIPE_M, joined by row, label-guarded):
             single              max(proxy - F, 1)
             stacked_v34lanes    PIPE_M's value on the incumbent's CBV ("pop") rows, max(proxy - (F + resid), 1) on the
                                 ADS-B-stage rows: v34's lanes with the new F and the re-trained stack
             stacked_r12partner  as above, but 0.5 (taxi_C + max(proxy - F, 1)) on the pop rows (R12 replaces FW / ORL)
           each vs PIPE_M (unweighted and coverage-reweighted with the val run's weights), and arm vs arm.
serve: v34_compose.matched_values (v34's lanes: N rows through prc.pipeline.adsb_stage with the new F, stack and R12
  serve table; R rows with v34 steps 4-6's C / ORL / FW, unchanged) -> 2026 matched values by MVT id (+ move stats vs a
  previous submission file when --compare-to names one). F must come from a 12-month refit (r12_validate --split refit); a fold-A F is code-path only.
LightGBM only in this process (the stack's MLP runs in its worker; BC-5).
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("r12_validate", "scripts/analysis/r12_validate.py")
B = R.B
RECORD_COLS = ["row", "month", "ap", "y", "proxy", "delta", "treatment"]
SPENT_DAYS = ("2025-01-09",)                             # adsb_stack.SPENT_DAYS (asserted equal in main)
VARIANTS = ("single", "stacked_v34lanes", "stacked_r12partner")
DERIVED_DELTA_ARMS = ("lgbcat", "blendy")                # R13 blends built by r13_cat.assemble (delta-scale Fs)
V34_INPUT_STEPS = (3, 4, 5, 6)                  # every step report v34_compose.new_inputs reads


# ---------------------------------------------------------------- record and table

def month_pos_to_rows(month, pos, counts: dict) -> np.ndarray:
    """(cache month, position in the month file) -> the global row over the months concatenated in ascending order."""
    months = sorted(counts)
    start = dict(zip(months, np.cumsum([0] + [counts[m] for m in months[:-1]])))
    month, pos = np.asarray(month).astype(int), np.asarray(pos).astype("int64")
    bad = sorted(set(month.tolist()) - set(months))
    if bad:
        raise ValueError(f"month {bad[0]} is not a cache month")
    size = np.array([counts[m] for m in month])
    if len(pos) and ((pos < 0) | (pos >= size)).any():
        raise ValueError("position out of range for its month")
    return np.array([start[m] for m in month], dtype="int64") + pos


def stack_record(pred: pd.DataFrame, arm: str, counts: dict) -> pd.DataFrame:
    """The val run's test rows as a v34 fold record, sorted by global row; treatment = the arm's raw delta."""
    if R.TARGETS.get(arm) != "delta" and arm not in DERIVED_DELTA_ARMS:
        raise ValueError(f"{arm} is not a delta arm: the stack's F is a delta")
    col = f"pred_{arm}"
    if col not in pred.columns:
        raise KeyError(f"{col} is not in the val predictions")
    t = pred[col].to_numpy(float)
    if not np.isfinite(t).all():
        raise ValueError(f"{col}: non-finite predictions")
    rec = pd.DataFrame({"row": month_pos_to_rows(pred.month, pred.pos, counts), "month": pred.month.astype("int32"),
                        "ap": pred.ap.astype(str), "y": pred.y.to_numpy(float), "proxy": pred.proxy.to_numpy(float),
                        "delta": pred.proxy.to_numpy(float) - pred.y.to_numpy(float), "treatment": t})
    return rec.sort_values("row", kind="stable").reset_index(drop=True)[RECORD_COLS]


def stack_table(train_table: pd.DataFrame, rec: pd.DataFrame, counts: dict, allow_missing: bool) -> pd.DataFrame:
    """The R12 table for the record's rows, in record order, in the v4 layout (row, MVT_ID_mvt, FEATURES, coverage).
    allow_missing (smoke only): a record row without a table row gets coverage 0 and NaN features."""
    m, p = R.rows_to_month_pos(rec.row.to_numpy(), counts)
    t = train_table.set_index(train_table.month.to_numpy().astype("int64") * R.KEY_MULT
                              + train_table.pos.to_numpy().astype("int64"))
    j = t.reindex(m.astype("int64") * R.KEY_MULT + p.astype("int64"))
    absent = j.coverage.isna().to_numpy()
    if absent.any() and not allow_missing:
        raise ValueError(f"{int(absent.sum())} record rows have no R12 table row")
    out = pd.DataFrame({"row": rec.row.to_numpy(), "MVT_ID_mvt": j.MVT_ID_mvt.to_numpy(float)})
    for c in B.FEATURES:
        out[c] = j[c].to_numpy(float)
    out["coverage"] = np.where(absent, 0, j.coverage.fillna(0).to_numpy()).astype("int64")
    return out


def joined_shares(train_table: pd.DataFrame, serve_table: pd.DataFrame) -> tuple:
    """(c25, c26) = adsb_ship.joined_share semantics (share of coverage >= 2 per airport) over the Jan + Jul 2025 table
    rows without the spent day, and over the 2026 serve table."""
    t = train_table[train_table.month.isin((1, 7)) & ~train_table.mvt_day.isin(SPENT_DAYS)]

    def share(f):
        return {str(a): float((g.coverage >= 2).mean()) for a, g in f.groupby("ap")}
    return share(t), share(serve_table)


def gate_bools(gate: dict) -> dict:
    return {a: bool(v["allowed"]) if isinstance(v, dict) else bool(v) for a, v in gate.items()}


# ---------------------------------------------------------------- composition

def align_incumbent(inc: pd.DataFrame, fr: pd.DataFrame) -> tuple:
    """(inc, positions in fr of each inc row), joined by `row`, with the label guard (y equal to 1e-6)."""
    pos = pd.Series(np.arange(len(fr)), index=fr.row.to_numpy()).reindex(inc.row.to_numpy())
    if pos.isna().any():
        raise ValueError(f"{int(pos.isna().sum())} incumbent rows missing from the stack frame")
    idx = pos.to_numpy().astype("int64")
    if np.abs(fr.y.to_numpy(float)[idx] - inc.y.to_numpy(float)).max() > 1e-6:
        raise AssertionError("incumbent y differs from the stack frame's y: misaligned")
    if not np.array_equal(fr.ap.astype(str).to_numpy()[idx], inc.ap.astype(str).to_numpy()):
        raise AssertionError("incumbent ap differs from the stack frame's ap: misaligned")
    return inc.reset_index(drop=True), idx


def compose_oof(inc: pd.DataFrame, F, resid) -> dict:
    """The composed OOF predictions on the incumbent's rows (see the module docstring)."""
    F, resid = np.asarray(F, float), np.asarray(resid, float)
    if len(F) != len(inc) or len(resid) != len(inc):
        raise ValueError(f"length mismatch: {len(F)} F, {len(resid)} resid, {len(inc)} incumbent rows")
    if not (np.isfinite(F).all() and np.isfinite(resid).all()):
        raise ValueError("non-finite F or residual")
    proxy, pop = inc.proxy.to_numpy(float), inc["pop"].to_numpy(bool)
    single = np.maximum(proxy - F, 1.0)
    stage = np.maximum(proxy - (F + resid), 1.0)
    return {"single": single, "stacked_v34lanes": np.where(pop, inc["pipe"].to_numpy(float), stage),
            "stacked_r12partner": np.where(pop, 0.5 * (inc["taxi_C"].to_numpy(float) + single), stage),
            "PIPE_M": inc["pipe"].to_numpy(float)}


def compare_arms(oofs: dict, n_boot: int = R.N_BOOT) -> dict:
    """Every ordered pair (a, b) in the given order: a vs b on each variant both carry (gain > 0 = a better)."""
    names = list(oofs)
    ref = oofs[names[0]]
    for n in names[1:]:
        if len(oofs[n]) != len(ref) or not np.array_equal(oofs[n].y.to_numpy(), ref.y.to_numpy()):
            raise ValueError(f"arms {names[0]} and {n} differ in rows")
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            A, Bf = oofs[a], oofs[b]
            rec = {}
            for v in ("stacked_v34lanes", "single"):
                if v in A.columns and v in Bf.columns:
                    y, blk, w = A.y.to_numpy(float), A.block.to_numpy(), A.w.to_numpy(float)
                    rec[v] = {"unweighted": R.paired_gain(y, A[v], Bf[v], blk, None, n_boot),
                              "weighted": R.paired_gain(y, A[v], Bf[v], blk, w, n_boot)}
            out[f"{a}_vs_{b}"] = rec
    return out


def stack_arm(arm: str, pred: pd.DataFrame, train_table: pd.DataFrame, serve_table: pd.DataFrame, out_dir, deps: dict,
              counts: dict, smoke: bool, root=ROOT, n_boot: int = R.N_BOOT, log=print) -> dict:
    """One arm: record + table -> stack trained, gated -> OOF residual -> composition -> report (see the docstring).
    `deps`: load_frame(rec_path, table_path), train(fr, feats, models_dir, seeds), transfer(fr, feats),
    predict(fr, models_dir, oof), incumbent(), gate(c25, c26, transfer_pass), v4_gate, feats."""
    t0 = time.time()
    out = B.check_out_dir(out_dir, root)
    paths = {k: out / f"{k}_{arm}.{ext}" for k, ext in (("record", "parquet"), ("table", "parquet"), ("oof", "parquet"),
                                                         ("report", "json"))}
    models = out / f"models_{arm}"
    for p in list(paths.values()) + [models]:
        if p.exists():
            raise FileExistsError(f"{p} exists: an R12 stack output is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    rec = stack_record(pred, arm, counts)
    tab = stack_table(train_table, rec, counts, allow_missing=smoke)
    rec.to_parquet(paths["record"], index=False)
    tab.to_parquet(paths["table"], index=False)
    fr = deps["load_frame"](paths["record"], paths["table"])
    seeds = (0,) if smoke else (0, 1, 2)
    deps["train"](fr, deps["feats"], models, seeds)
    tr = deps["transfer"](fr, deps["feats"])
    c25, c26 = joined_shares(train_table, serve_table)
    gate = deps["gate"](c25, c26, bool(tr["rule"]["PASS"]))
    same = gate == deps["v4_gate"]
    mp = models / "MANIFEST.json"
    man = json.loads(mp.read_text())
    if man.get("gate"):
        raise AssertionError("train_artifacts wrote a gate; refusing to overwrite it")
    man["gate"] = gate
    man["gate_evidence"] = {"source": "r12_stack: adsb_ship.gate_table on the R12 (TRC) tables' joined shares",
                            "transfer_PASS": bool(tr["rule"]["PASS"]), "joined_share_2025": c25, "joined_share_2026": c26,
                            "equals_models_v4_gate": bool(same),
                            # relative to `root` (check_out_dir keeps `out` under it): a shipped manifest never carries a
                            # machine's absolute path
                            "fold_record": str(paths["record"].resolve().relative_to(pathlib.Path(root).resolve()))}
    mp.write_text(json.dumps(R.json_clean(man), indent=1))
    resid = np.asarray(deps["predict"](fr, models, True), float)
    inc = deps["incumbent"]()
    if smoke:
        inc = inc[inc.row.isin(fr.row)].reset_index(drop=True)
    inc, idx = align_incumbent(inc, fr)
    F = fr.F.to_numpy(float)[idx]
    v = compose_oof(inc, F, resid[idx])
    wmap = pd.Series(pred.w.to_numpy(float), index=month_pos_to_rows(pred.month, pred.pos, counts))
    w = wmap.reindex(inc.row.to_numpy()).to_numpy(float)
    if not np.isfinite(w).all():
        raise AssertionError("an incumbent row has no validation weight")
    blocks = inc.month.to_numpy().astype("int64") * 1000 + inc.doy.to_numpy().astype("int64")
    y = inc.y.to_numpy(float)
    oof = pd.DataFrame({"row": inc.row.to_numpy(), "MVT_ID_mvt": inc.MVT_ID_mvt.to_numpy(float),
                        "ap": fr.ap.astype(str).to_numpy()[idx], "month": inc.month.to_numpy(), "doy": inc.doy.to_numpy(),
                        "y": y, "proxy": inc.proxy.to_numpy(float), "pop": inc["pop"].to_numpy(bool),
                        "coverage": fr.coverage.to_numpy()[idx], "F": F, "resid": resid[idx], "w": w, "block": blocks,
                        **{k: v[k] for k in VARIANTS + ("PIPE_M",)}})
    oof.to_parquet(paths["oof"], index=False)
    stage = ~oof["pop"].to_numpy()
    rep = {"arm": arm, "smoke": bool(smoke), "n_rows": int(len(oof)), "n_stage_rows": int(stage.sum()),
           "n_stage_rows_moved": int((stage & (oof.resid.to_numpy() != 0)).sum()), "seeds": list(seeds),
           "feats": list(deps["feats"]), "gate": gate, "gate_equals_models_v4": bool(same), "transfer": tr,
           "joined_share_2025": c25, "joined_share_2026": c26,
           "rmse": {k: R.weighted_rmse(y, v[k]) for k in VARIANTS + ("PIPE_M",)},
           "rmse_weighted": {k: R.weighted_rmse(y, v[k], w) for k in VARIANTS + ("PIPE_M",)},
           "vs_PIPE_M": {k: {"unweighted": R.paired_gain(y, v[k], v["PIPE_M"], blocks, None, n_boot),
                             "weighted": R.paired_gain(y, v[k], v["PIPE_M"], blocks, w, n_boot)} for k in VARIANTS},
           "units": "board MSE = SSE/344,841; gain > 0 = the variant beats PIPE_M",
           "wall_s": round(time.time() - t0, 1), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    R.write_json_new(rep, paths["report"])
    for k in VARIANTS:
        g = rep["vs_PIPE_M"][k]
        log(f"{arm} {k} vs PIPE_M: {g['unweighted']['gain_bmse']:+.2f} CI {[round(x, 1) for x in g['unweighted']['ci95']]}"
            f"  reweighted {g['weighted']['gain_bmse']:+.2f} CI {[round(x, 1) for x in g['weighted']['ci95']]}")
    return rep


# ---------------------------------------------------------------- serving

def serve_stack_table(serve_table: pd.DataFrame, ids, allow_missing: bool) -> pd.DataFrame:
    """The 2026 stage table for prc.pipeline.adsb_stage.stage_frame: indexed by MVT id, MVT_ID_mvt + FEATURES +
    coverage only (the stage adds ap / hour / proxy / F itself). allow_missing: smoke only."""
    ids = np.asarray(ids, dtype="float64")
    j = serve_table.set_index(serve_table.MVT_ID_mvt.to_numpy(float)).reindex(ids)
    absent = j.coverage.isna().to_numpy()
    if absent.any() and not allow_missing:
        raise ValueError(f"{int(absent.sum())} serve ids have no R12 table row")
    out = pd.DataFrame({"MVT_ID_mvt": ids}, index=ids)
    for c in B.FEATURES:
        out[c] = j[c].to_numpy(float)
    out["coverage"] = np.where(absent, 0, j.coverage.fillna(0).to_numpy()).astype("int64")
    return out


def serve(models_dir, arm: str, f26: pd.DataFrame, serve_table: pd.DataFrame, smoke: bool, compare_to=None,
          log=print):  # pragma: no cover - real files
    """v34's 2026 matched composition with the new F / stack / table (v34_compose.matched_values), all real files.
    `compare_to`: an optional earlier submission file; when given, the stats carry the move against it."""
    V = _load("v34_compose", "scripts/analysis/v34_compose.py")
    AS = _load("adsb_stack", "scripts/adsb_stack.py")
    L = V.load_lanes()
    old = V.old_inputs(L)
    new = V.new_inputs({s: V.C.read_step_report(s) for s in V34_INPUT_STEPS}, old)
    rows = L["rows"].assign(table="v4")
    F = pd.Series(f26[f"pred_{arm}"].to_numpy(float), index=f26.MVT_ID_mvt.to_numpy(float))
    tables = {"v4": serve_stack_table(serve_table, rows.MVT_ID_mvt.to_numpy(), allow_missing=smoke)}
    man = json.loads((pathlib.Path(models_dir) / "MANIFEST.json").read_text())
    gate = gate_bools(man["gate"])
    vals, detail = V.matched_values(rows, F, tables, lambda fr, _n: AS.predict_residual(fr, models_dir, oof=False), gate,
                                    V.r_rows(L, new["proxy_c"]), new["C"], new["orl_int"], new["fw_int"])
    stats = {"n_matched": int(len(vals)), "gate": gate,
             "f_source": "refit" if not smoke else "SMOKE: fold-A F (code path only, never a candidate)"}
    if compare_to is not None:
        old = pd.read_parquet(compare_to)
        prev = pd.Series(old.iloc[:, 1].to_numpy(float), index=old.iloc[:, 0].to_numpy(float)).reindex(vals.index.to_numpy())
        move = vals.to_numpy(float) - prev.to_numpy(float)
        stats.update(compare_to=str(compare_to), n_moved=int((move != 0).sum()),
                     S_vs_previous=float((move ** 2).sum() / R.BOARD_N), rms_move_s=float(np.sqrt((move ** 2).mean())))
    log(f"serve {arm}: {stats}")
    return vals, detail, stats


# ---------------------------------------------------------------- CLI

def real_deps(smoke: bool) -> dict:  # pragma: no cover - the real stages; exercised by the smoke run
    V34S = _load("v34_adsb_stack", "scripts/analysis/v34_adsb_stack.py")
    AS, SH = V34S.AS, V34S.SH
    if tuple(AS.SPENT_DAYS) != SPENT_DAYS:
        raise AssertionError("adsb_stack.SPENT_DAYS changed")
    con = V34S.v4_contract()
    feats = list(con["feats"])

    def incumbent():
        RC = _load("rcr2_build", "scripts/analysis/rcr2_build.py")
        CB = _load("cln_build", "scripts/analysis/cln_build.py")
        inc = RC.load_incumbent(subset_ok=smoke)
        m = CB.load_comparators()["M"].set_index("MVT_ID_mvt")["pipe"]
        inc = inc.assign(pipe=m.reindex(inc.MVT_ID_mvt.to_numpy(float)).to_numpy(float))
        if inc["pipe"].isna().any():
            raise AssertionError("an incumbent row has no PIPE_M value")
        return inc
    return dict(load_frame=lambda rp, tp: V34S.load_frame(rp, tp, V34S.RAW_MONTHS),
                train=lambda fr, f, out, seeds: AS.train_artifacts(fr, f, out, seeds=seeds, gate={},
                                                                   ship_learner=con["ship_learner"]),
                transfer=lambda fr, f: V34S.transfer(fr, f, AS),
                predict=lambda fr, m, oof: AS.predict_residual(fr, m, oof=oof),
                incumbent=incumbent, feats=feats, v4_gate=con["gate"],
                gate=lambda c25, c26, ok: SH.gate_table(c25, c26, transfer_pass=ok, holds={},
                                                        coverage_overrides=V34S.V4_OVERRIDES))


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    o = sub.add_parser("oof")
    o.add_argument("--val", required=True, help="an r12_validate holdout output directory")
    o.add_argument("--arms", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--stack", required=True, help="an r12_stack oof output directory (its models_<arm>)")
    s.add_argument("--arm", required=True)
    s.add_argument("--f26", required=True, help="predictions_2026.parquet of a --split refit run")
    s.add_argument("--compare-to", default=None, help="optional: an earlier submission file to report the move against")
    for p in (o, s):
        p.add_argument("--tables", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--rss-limit-gib", type=float, required=True)
        p.add_argument("--smoke", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:  # pragma: no cover - real-data entry point; exercised by the smoke runs
    import run_guards as G
    a = parse_args(argv)
    log = lambda m: print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)
    tables = pathlib.Path(a.tables)
    trep = json.loads((tables / "report.json").read_text())
    if trep.get("features_version") != B.require_trc(B.AF.FEATURES_VERSION):
        raise RuntimeError("the tables were built with another features_version")
    fh = None if a.smoke else G.acquire_lock()
    G.start_rss_guard(int(a.rss_limit_gib * 1024 ** 3))
    try:
        train = pd.read_parquet(tables / "train.parquet")
        serve_t = pd.read_parquet(tables / "serve.parquet")
        if a.cmd == "oof":
            vrep = json.loads((pathlib.Path(a.val) / "report.json").read_text())
            if vrep["split"] != "holdout" or bool(vrep["smoke"]) != bool(a.smoke):
                raise RuntimeError(f"--val must be a holdout run of the same mode (got {vrep['split']}, smoke "
                                   f"{vrep['smoke']})")
            pred = pd.read_parquet(pathlib.Path(a.val) / "predictions.parquet")
            if a.smoke:                                   # the smoke stack needs >= 5 days with features
                pred = pred[np.isfinite(pred.coverage.to_numpy(float))].reset_index(drop=True)
            deps, counts, reps = real_deps(a.smoke), R.cache_month_counts(), {}
            oofs = {}
            for arm in [x for x in a.arms.split(",") if x]:
                reps[arm] = stack_arm(arm, pred, train, serve_t, a.out, deps, counts, a.smoke, log=log)
                oofs[arm] = pd.read_parquet(pathlib.Path(a.out) / f"oof_{arm}.parquet")
            if len(oofs) > 1:
                R.write_json_new(compare_arms(oofs), pathlib.Path(a.out) / "compare_arms.json")
        else:
            out = B.check_out_dir(a.out)
            if out.exists():
                raise SystemExit(f"refusing to overwrite {out}")
            f26 = pd.read_parquet(a.f26)
            frep = json.loads((pathlib.Path(a.f26).parent / "report.json").read_text())
            if frep["split"] != "refit" and not a.smoke:
                raise RuntimeError("serve needs the F of a --split refit run (a fold-A F is smoke-only)")
            vals, detail, stats = serve(pathlib.Path(a.stack) / f"models_{a.arm}", a.arm, f26, serve_t, a.smoke,
                                        compare_to=a.compare_to, log=log)
            out.mkdir(parents=True)
            pd.DataFrame({"MVT_ID_mvt": vals.index.to_numpy(float), "taxi": vals.to_numpy(float)}).to_parquet(
                out / f"matched_2026_{a.arm}.parquet", index=False)
            detail.to_parquet(out / f"matched_2026_{a.arm}_detail.parquet", index=False)
            R.write_json_new(stats, out / f"serve_{a.arm}.json")
        log(f"done; peak RSS {G.peak_rss() / 1024 ** 3:.2f} GiB")
    finally:
        if fh is not None:
            G.release_lock(fh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
