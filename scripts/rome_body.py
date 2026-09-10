"""`scripts/rome_body.py` -- arm B of plans/PREREG_rome_body_2026_09_10.md.

    python scripts/rome_body.py fit [--smoke]     # heavy: arm F's design refitted on NON-FILL rows
    python scripts/rome_body.py evaluate          # light: the Rome matched mixture with that body

`fit` is arm F exactly -- FEATS + QUEUE + ORDER (83 columns), L.P, A2 (no per-airport models),
seeds 0,1,2, best_iter re-found through lgbm_fold.fit_arm -- except that the training, fit and
early-stopping masks exclude fill rows (|y - sp| <= 60 s). The holdout is untouched. It writes
`data/cache_stand/rome_body_preds.parquet` (the fold's `row` key + the body delta per seed).

`evaluate` joins that record with arm F's (`fold_preds_queue_order.parquet`) on `row`, maps rows
to MVT_ID through the order cache, takes arm M's classifier p (rome_matched.classifier_p), and
scores B and B_hyb with rome_matched.score -- the same five clauses as arm M.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_fill as rf  # noqa: E402
import rome_matched as rm  # noqa: E402

GATE = rm.GATE
BODY_PREDS = ROOT / "data" / "cache_stand" / "rome_body_preds.parquet"
BODY_JSON = ROOT / "reports" / "rome_body_fit.json"


def nonfill_fold(fold: dict) -> tuple:
    """(a copy of `fold` whose tr / fit / es masks exclude fill rows, the fill mask). te untouched."""
    fill = np.abs(np.asarray(fold["y"], dtype="float64") - np.asarray(fold["sp"], dtype="float64")) <= rf.FILL_TOL_S
    out = dict(fold)
    for k in ("tr", "fit", "es"):
        out[k] = np.asarray(fold[k], dtype=bool) & ~fill
    return out, fill


def assemble(p, sp, F, body) -> dict:
    """The registered arms from the classifier p, the schedule offset, arm F and the conditioned body."""
    p, sp, F, body = (np.asarray(v, dtype="float64") for v in (p, sp, F, body))
    b = rf.mixture(p, sp, body)
    return {"B": b, "B_hyb": np.where(p >= GATE, b, F)}


def assemble_all(p, sp, F, body) -> dict:
    """Amendment B.1's arm B_all: the conditioned mixture where p >= 0.5, F elsewhere (the B_hyb form);
    Amendment B.2's arm B_all_cont: the continuous mixture on every row, no gate (the B form)."""
    a = assemble(p, sp, F, body)
    return {"B_all": a["B_hyb"], "B_all_cont": a["B"]}


MATCHED_ALL = ROOT / "data" / "cache_rome" / "matched_all.parquet"
MATCHED_COLS = ["MVT_ID_mvt", "ADEP_mvt", "month", "y", "unmatched", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
                "sp", "dayoff", "hr", "tmin", "dow"] + rf.FEATS_CAT + rm.NM_FEATS


def load_matched_all(log=print) -> pd.DataFrame:
    """Every admissible MATCHED departure of 2025 through the shipped derive(), reduced to the
    classifier's columns, cached in data/cache_rome/. Heavy on first build (~2.4 GB): run alone."""
    if MATCHED_ALL.exists():
        return pd.read_parquet(MATCHED_ALL)
    import glob
    bs = rf.sf.bs
    files = sorted(glob.glob(str(ROOT / "data" / "raw" / "training_2025-*.parquet")))
    f = bs.derive(bs.admissible(bs.load_movements(files)))
    f["y"] = f.TAXITIME_SEC_mvt.astype("float64")
    m = f.loc[~f.unmatched, MATCHED_COLS].reset_index(drop=True)
    MATCHED_ALL.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(MATCHED_ALL)
    log(f"matched frame cached: {len(m):,} rows -> {MATCHED_ALL}")
    return m


def all_airport_p(frame: pd.DataFrame, rec: pd.DataFrame, seeds=rf.SEEDS) -> dict:
    """{seed: p aligned with `rec`}: per airport, the registered classifier trained on that
    airport's MATCHED rows of the non-holdout months only, scored on its rows of `rec`."""
    ps = {s: np.full(len(rec), np.nan) for s in seeds}
    idx = frame.set_index("MVT_ID_mvt", drop=False)
    for ap in sorted(set(rec.ap)):
        rows = np.flatnonzero(rec.ap.to_numpy() == ap)
        tr = frame[(frame.ADEP_mvt == ap) & ~frame.unmatched.astype(bool) & ~frame.month.isin(rm.FOLD_A)]
        te = idx.loc[rec.MVT_ID_mvt.to_numpy()[rows]]
        for s in seeds:
            ps[s][rows] = rf.fit_predict_fill(rm._with_nm(tr), rm._with_nm(te), seed=s, extra_num=rm.NM_FEATS)
    return ps


def evaluate_all(out=None) -> dict:
    """Amendment B.1: B_all on ALL matched fold-A rows against F, with the rescaled bars."""
    f = pd.read_parquet(rm.F_RECORD, columns=["row", "month", "ap", "y", "proxy", "sp", "treatment"])
    f["F"] = np.maximum(f.proxy - f.treatment, 1.0)
    f["MVT_ID_mvt"] = rm.row_to_mvt_id(ROOT / "data" / "cache_stand", ROOT / "data" / "cache_order")[f.row.to_numpy()]
    body = pd.read_parquet(BODY_PREDS)
    seed_cols = sorted(c for c in body.columns if c.startswith("delta_body_seed"))
    rec = f.merge(body[["row", "y"] + seed_cols], on="row", suffixes=("", "_b"))
    if len(rec) != len(f) or not np.array_equal(rec.y.to_numpy(), rec.y_b.to_numpy()):
        raise AssertionError("arm B's record does not align with arm F's on `row`")
    frame = load_matched_all()
    chk = frame.set_index("MVT_ID_mvt").y.reindex(rec.MVT_ID_mvt.to_numpy()).to_numpy()
    if not np.array_equal(chk, rec.y.to_numpy()):
        raise AssertionError("matched frame labels differ from the fold record: wrong MVT_ID mapping")
    ps = all_airport_p(frame, rec)
    p = np.mean(list(ps.values()), axis=0)
    sp, F, proxy = (rec[c].to_numpy(dtype="float64") for c in ("sp", "F", "proxy"))
    body_mean = np.maximum(proxy - np.mean([rec[c].to_numpy(dtype="float64") for c in seed_cols], axis=0), 1.0)
    arms = {"F": F, **assemble_all(p, sp, F, body_mean)}
    for c in seed_cols:
        s = int(c.removeprefix("delta_body_seed"))
        for k, v in assemble_all(p, sp, F, np.maximum(proxy - rec[c].to_numpy(dtype="float64"), 1.0)).items():
            arms[f"{k}_seed{s}"] = v
    res = rm.score(rec, arms, ("B_all", "B_all_cont"), tuple(int(c.removeprefix("delta_body_seed")) for c in seed_cols),
                   bar=0.5, body_tol=0.2)
    y = rec.y.to_numpy(dtype="float64")
    res["per_airport_gain_s"] = {arm: {ap: float(np.sqrt(((F - y)[m] ** 2).mean()) - np.sqrt(((arms[arm] - y)[m] ** 2).mean()))
                                       for ap in sorted(set(rec.ap)) for m in [rec.ap.to_numpy() == ap]}
                                 for arm in ("B_all", "B_all_cont")}
    res["shortlist_net_ge_1000"] = {arm: bool(res["detail"][arm]["projected_board_mse"] >= 1_000) for arm in ("B_all", "B_all_cont")}
    res.pop("boot", None)
    if out:
        pathlib.Path(out).write_text(json.dumps(res, indent=1, default=float))
    return res


def fit(smoke: bool, out_dir=None) -> int:
    import lgbm_fold as lf
    import lgbm_submit as L
    params, nest, patience, months, pa_floor = dict(L.P), L.NEST, L.PATIENCE, None, L.PA_TREE_FLOOR
    seeds = rf.SEEDS
    if smoke:
        params["learning_rate"] = lf.SMOKE["learning_rate"]
        nest, patience, months = lf.SMOKE["nest"], lf.SMOKE["patience"], lf.SMOKE["months"]
    out = pathlib.Path(out_dir) if out_dir else (ROOT / "data" / "smoke_body" if smoke else None)
    preds_path = (out / "rome_body_preds.parquet") if out else BODY_PREDS
    json_path = (out / "rome_body_fit.json") if out else BODY_JSON
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    log = lf._Log(json_path.with_suffix(".log"))
    lf.log = L.log = log
    t0 = time.time()
    if smoke:
        log(L.SMOKE_BANNER)
    feats = lf.FEATS_QUEUE_ORDER
    fold = lf.load_fold(lf.CACHE, months, feats, qcache=lf.QCACHE, ocache=lf.OCACHE, check_counts=not smoke,
                        label="holdout (months (1, 7))")
    nf, fill = nonfill_fold(fold)
    log(f"arm B body: training rows {int(fold['tr'].sum()):,} -> {int(nf['tr'].sum()):,} non-fill; fit "
        f"{int(nf['fit'].sum()):,}; early-stop {int(nf['es'].sum()):,}; holdout {int(fold['te'].sum()):,} (untouched)")
    base = fold["base"].copy()
    cols = {}

    def on_seed(s, pred):
        cols[f"delta_body_seed{s}"] = pred
        pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1).to_parquet(preds_path)
    arm = lf.fit_arm(nf, fold["X"], params, nest, patience, pa_floor, seeds, per_airport=False, pa_trees="es",
                     name="body", on_seed=on_seed)
    cols["body"] = arm["delta"]
    pd.concat([base, pd.DataFrame(cols, index=base.index)], axis=1).to_parquet(preds_path)
    rec = {"prereg": "plans/PREREG_rome_body_2026_09_10.md", "smoke": smoke, "features": feats, "params": params,
           "best_iter": arm["best_iter"], "n_ref": arm["n_ref"], "single_seed_rmse_all_holdout": {str(k): v for k, v in arm["single_rmse"].items()},
           "n_train_nonfill": int(nf["tr"].sum()), "n_fill_dropped": int((fold["tr"] & fill).sum()),
           "wall_s": round(time.time() - t0, 1), "peak_rss_gb": round(L.peak_rss_gb(), 2), "preds": str(preds_path),
           "written": dt.datetime.now().isoformat(timespec="seconds")}
    json_path.write_text(json.dumps(rec, indent=1, default=float))
    log(f"json -> {json_path}   best_iter {arm['best_iter']:,} n_ref {arm['n_ref']:,}   wall {rec['wall_s']:.0f}s")
    if smoke:
        log(L.SMOKE_BANNER)
    log.close()
    return 0


def evaluate(out=None) -> dict:
    fr = rm.load_record()                                      # arm F, LIRF rows, with MVT_ID and `F`
    frow = pd.read_parquet(rm.F_RECORD, columns=["row", "ap"])
    fr = fr.assign(row=frow.loc[frow.ap == "LIRF", "row"].to_numpy())
    body = pd.read_parquet(BODY_PREDS)
    rec = fr.merge(body[["row", "y", "proxy"] + [c for c in body.columns if c.startswith("delta_body_seed")]],
                   on="row", suffixes=("", "_b"))
    if len(rec) != len(fr) or not np.array_equal(rec.y.to_numpy(), rec.y_b.to_numpy()):
        raise AssertionError("arm B's record does not align with arm F's on `row` (count or labels differ)")
    rec = rec.reset_index(drop=True)
    unm, lirf = rf.load_frames(lambda m: None)
    ps, _ = rm.classifier_p(unm, lirf, rec, rf.SEEDS)
    p = np.mean(list(ps.values()), axis=0)
    sp, F, proxy = (rec[c].to_numpy(dtype="float64") for c in ("sp", "F", "proxy"))
    seed_cols = sorted(c for c in rec.columns if c.startswith("delta_body_seed"))
    bodies = {int(c.removeprefix("delta_body_seed")): np.maximum(proxy - rec[c].to_numpy(dtype="float64"), 1.0) for c in seed_cols}
    body_mean = np.maximum(proxy - np.mean([rec[c].to_numpy(dtype="float64") for c in seed_cols], axis=0), 1.0)
    arms = {"F": F, **assemble(p, sp, F, body_mean)}
    for s, bs_ in bodies.items():
        for k, v in assemble(p, sp, F, bs_).items():
            arms[f"{k}_seed{s}"] = v
    res = rm.score(rec, arms, ("B", "B_hyb"), tuple(sorted(bodies)))
    fill = np.abs(rec.y.to_numpy() - sp) <= rf.FILL_TOL_S
    y = rec.y.to_numpy(dtype="float64")
    res["body_quality_lirf_nonfill_rmse"] = {"F": float(np.sqrt(((F - y)[~fill] ** 2).mean())),
                                             "body": float(np.sqrt(((body_mean - y)[~fill] ** 2).mean()))}
    res.pop("boot", None)
    if out:
        pathlib.Path(out).write_text(json.dumps(res, indent=1, default=float))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("fit", "evaluate", "evaluate-all"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    if a.step == "fit":
        return fit(a.smoke, a.out_dir)
    if a.step == "evaluate-all":
        res = evaluate_all(ROOT / "reports" / "rome_body_all.json")
        print(json.dumps({k: res[k] for k in ("rmse", "clauses", "verdict", "per_airport_gain_s")}, indent=1, default=float))
        for arm in ("B_all", "B_all_cont"):
            print(arm, res["detail"][arm], "shortlist:", res["shortlist_net_ge_1000"][arm])
        return 0
    res = evaluate(ROOT / "reports" / "rome_body.json")
    print(json.dumps({k: res[k] for k in ("rmse", "clauses", "verdict", "body_quality_lirf_nonfill_rmse")}, indent=1, default=float))
    for arm in ("B", "B_hyb"):
        print(arm, {k: v for k, v in res["detail"][arm].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
