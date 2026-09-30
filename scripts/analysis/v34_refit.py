"""v34 steps 3, 4 and 5 — the 12-month production refits without the three label-seconds columns
(plans/PREREG_nsc_no_seconds_2026_09_24.md §7 steps 3-5, AMENDMENTS V34.3 / V34.4 / V34.5).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/v34_launch.py --step 3|4|5 --detach [--smoke]
    (the launcher runs: v34_refit.py --design F|FW|ORL --rss-limit-gib 8|8|6 [--smoke])

design  step  columns                                                shipped counterpart (never overwritten)
F       3     nsc_fold.FEATS_NSC_F  = FEATS_QUEUE_ORDER - 3   (80)    data/cache_stand/lgbm_v6_queue_order_seed{0,1,2}.txt
FW      4     nsc_fold.FEATS_NSC_FW = FEATS_QUEUE_ORDER_WEATHER - 3 (90) data/cache_stand/lgbm_v6_queue_order_weather_seed*.txt
ORL     5     orl_stage2.stage2_feats arm (99) - 3            (96)    data/orl_build/lgbm_orl99_seed*.txt

The recipe is the shipped one, by import: lgbm_submit.load_frames (12 training months + the 339,551-row ranking cache, the
queue / order [/ weather] blocks; for ORL the route block by orl_stage1.attach_to_cache_frame with the registered route
table sha), then orl_stage3.production_refit = lgbm_submit.split_masks (ES months 3, 9) -> stand_ab.infold_encodings on every
training row -> design_matrix(feats) -> early_stop (ES seed 0, lgbm_submit.P) -> fit_seeds at n_ref for seeds 0, 1, 2 ->
mean_delta -> recover_taxi_time (int32). That is `lgbm_submit.py --version 6 --queue --orderfeats [--weatherfeats]
--seeds 0,1,2` with the design list swapped (lgbm_submit itself is not used: its main hard-codes FEATS and writes its boosters
over the shipped ones in data/cache_stand). No per-airport blend (none of the three shipped refits used one).

Outputs, new paths only (data/nsc_v34/step{3,4,5}_*/): lgbm_nsc_<tag>_seed{s}.txt + .fit.json, predictions_2026.parquet
(MVT_ID_mvt, month, ap, proxy, delta_mean, d_seed0..2, taxi_int) and report.json.
Acceptance: every booster's .fit.json n_features == the design width; the design holds no *_sec column; the label-free
tripwire RMS(taxi_int - shipped counterpart's int taxi) on common ids <= MOVE_BAR_FACTOR x the fold RMS move (F: step 1's
F-NSC vs F record; FW and ORL: FW-NSC vs the FW record, 14.47 s) -- above it the build is broken, status FAIL.
LightGBM only in this process (BC-5).
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
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


C = _load("v34_common", "scripts/analysis/v34_common.py")
NSC = _load("nsc_fold", "scripts/analysis/nsc_fold.py")
O3 = _load("orl_stage3", "scripts/analysis/orl_stage3.py")
LS = O3.LS                                              # lgbm_submit, the instance orl_stage3 / lgbm_fold loaded
S2 = O3.S2

SEEDS = (0, 1, 2)
PRED_NAME = "predictions_2026.parquet"
PRED_COLS = ["MVT_ID_mvt", "month", "ap", "proxy", "delta_mean", "d_seed0", "d_seed1", "d_seed2", "taxi_int"]
N_TRAIN, N_RANK = O3.N_TRAIN, O3.N_RANK                 # 2,062,440 / 339,551
#: RMS taxi move of FW-NSC vs the FW record on fold A (data/nsc_run/nsc_fold_preds.parquet; tested)
FW_FOLD_RMS_MOVE_S = 14.47487297993766
MOVE_BAR_FACTOR = 3.0
V13_DETAIL = ROOT / "data" / "adsb" / "v2" / "v13_detail.parquet"
FW_SHIPPED = ROOT / "data" / "fwp_build" / "fw" / "merry-quicksand_v6.parquet"
ORL_SHIPPED = ROOT / "data" / "orl_build" / "orl_2026_predictions.parquet"


@dataclasses.dataclass(frozen=True)
class Design:
    name: str
    step: int
    tag: str
    feats: list
    queue: bool
    order: bool
    weather: bool
    orl: bool


def _orl_feats() -> list:
    _, arm = S2.stage2_feats(LS.S)
    return NSC.drop_seconds(arm)


_DESIGNS = {
    "F": lambda: Design("F", 3, "f80", C.assert_no_seconds(NSC.FEATS_NSC_F, "F-NSC"), True, True, False, False),
    "FW": lambda: Design("FW", 4, "fw90", C.assert_no_seconds(NSC.FEATS_NSC_FW, "FW-NSC"), True, True, True, False),
    "ORL": lambda: Design("ORL", 5, "orl96", C.assert_no_seconds(_orl_feats(), "ORL-NSC"), True, True, True, True),
}


def design(name: str) -> Design:
    if name not in _DESIGNS:
        raise ValueError(f"unknown design {name!r}; one of {sorted(_DESIGNS)}")
    return _DESIGNS[name]()


def booster_files(des: Design, out_dir, allow_roots=(C.OUT_ROOT, C.SMOKE_ROOT)) -> list:
    out_dir = pathlib.Path(out_dir)
    return [(C.check_output_path(out_dir / f"lgbm_nsc_{des.tag}_seed{s}.txt", allow_roots=allow_roots, allow_files=()),
             C.check_output_path(out_dir / f"lgbm_nsc_{des.tag}_seed{s}.fit.json", allow_roots=allow_roots, allow_files=()))
            for s in SEEDS]


def refit_config(smoke: bool) -> tuple:
    params, nest, patience = dict(LS.P), LS.NEST, LS.PATIENCE
    if smoke:
        params["learning_rate"] = LS.SMOKE["learning_rate"]
        nest, patience = LS.SMOKE["nest"], LS.SMOKE["patience"]
    return params, nest, patience


def move_tripwire(new: pd.Series, old: pd.Series, bar_s: float) -> dict:
    """RMS of (new - old) integer taxi times on the ids both carry; pass iff <= bar_s."""
    common = new.index.intersection(old.index)
    if not len(common):
        raise AssertionError("the counterpart shares no id with the new predictions")
    m = new.reindex(common).to_numpy(float) - old.reindex(common).to_numpy(float)
    rms = float(np.sqrt(np.mean(m ** 2)))
    return {"n_common": int(len(common)), "rms_move_s": rms, "mean_move_s": float(m.mean()),
            "p99_abs_move_s": float(np.percentile(np.abs(m), 99)), "bar_s": float(bar_s), "pass": bool(rms <= bar_s)}


def run(des: Design, smoke: bool, out_dir, d: pd.DataFrame, is_rank, rank_ids, refit=None, counterpart=None, bar_s=None,
        log=print, allow_roots=(C.OUT_ROOT, C.SMOKE_ROOT), extra=None) -> dict:
    """The refit on frame `d` (training rows + ranking rows, flag `is_rank`), predictions + report written.
    `refit` = orl_stage3.production_refit; `counterpart` = the shipped model's int taxi by MVT_ID (None: no tripwire)."""
    t0 = time.time()
    refit = O3.production_refit if refit is None else refit
    out_dir = pathlib.Path(out_dir)
    pred_path = C.check_output_path(out_dir / PRED_NAME, allow_roots=allow_roots, allow_files=())
    rep_path = C.check_output_path(out_dir / C.REPORT_NAME, allow_roots=allow_roots, allow_files=())
    feats = C.assert_no_seconds(des.feats, f"{des.name}-NSC design")
    enc = {f"{p}_{k}" for k in LS.S.ENC_KEYS for p in ("te", "de")}      # made inside the refit (infold_encodings)
    missing = [c for c in feats if c not in d.columns and c not in enc] + [k for k in LS.S.ENC_KEYS if k not in d.columns]
    if missing:
        raise KeyError(f"design columns absent from the frame: {missing}")
    is_rank = np.asarray(is_rank, bool)
    files = booster_files(des, out_dir, allow_roots=allow_roots)
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = d.loc[is_rank, ["month", "ap", "proxy"]].reset_index(drop=True)
    params, nest, patience = refit_config(smoke)
    res = refit(d, is_rank, feats, params, nest, patience, SEEDS, files=files, smoke=smoke)
    widths = [int(json.loads(j.read_text())["n_features"]) for _, j in files]
    if widths != [len(feats)] * len(SEEDS):
        raise AssertionError(f"booster widths {widths}: a booster with {max(widths)} features is not the "
                             f"{len(feats)}-column {des.name}-NSC design")
    pred = pd.DataFrame({"MVT_ID_mvt": np.asarray(rank_ids, dtype="float64"), "month": rank.month.to_numpy().astype(int),
                         "ap": rank.ap.astype(str).to_numpy(), "proxy": rank.proxy.to_numpy(float),
                         "delta_mean": np.asarray(res["delta_mean"], float),
                         **{f"d_seed{s}": np.asarray(res["delta_by_seed"][s], float) for s in SEEDS},
                         "taxi_int": np.asarray(res["taxi_int"]).astype("int32")})[PRED_COLS]
    if len(pred) != int(is_rank.sum()) or not pred.MVT_ID_mvt.is_unique or not np.isfinite(pred.delta_mean).all():
        raise AssertionError("predictions: wrong length, duplicate ids or non-finite deltas")
    sha = C.write_parquet_new(pred, pred_path, allow_roots=allow_roots, allow_files=())
    trip = None
    if counterpart is not None:
        trip = move_tripwire(pred.set_index("MVT_ID_mvt").taxi_int, counterpart, bar_s)
    ok = trip is not None and trip["pass"]
    status = "SMOKE" if smoke else ("PASS" if ok else "FAIL")
    outputs = {C.rel(pred_path): sha}
    for p, j in files:
        outputs[C.rel(p)] = C.sha256(p)
        outputs[C.rel(j)] = C.sha256(j)
    rep = C.jsonable({"step": des.step, "tag": f"V34.{des.step} {des.name}-NSC production refit", "design": des.name,
                      "smoke": bool(smoke), "status": status, "prereg": C.rel(C.PREREG), "features": feats,
                      "n_features": len(feats), "dropped": list(C.SECONDS_COLS), "seeds": list(SEEDS),
                      "booster_n_features": widths, "es_info": res["info"], "floor_bind_rate": res["floor_bind_rate"],
                      "seed_sd_mean_s": res["seed_sd_mean_s"], "n_train": res["n_train"], "n_rank": res["n_rank"],
                      "tripwire": trip, "extra": extra or {}, "outputs": outputs, "wall_s": round(time.time() - t0, 1),
                      "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    C.write_json_new(rep, rep_path, allow_roots=allow_roots, allow_files=())
    log(f"{des.name}-NSC: {len(pred):,} rows predicted; tripwire {trip}; status {status}")
    return rep


# ---------------------------------------------------------------- real inputs

def counterpart_of(des: Design, rank_ids) -> pd.Series:  # pragma: no cover - reads the shipped files
    """The shipped counterpart's int taxi times by MVT_ID: F -> v10 (the Sep-10 F boosters' value, v13_detail, 209,463
    ADS-B-covered matched rows); FW -> FW_int (fwp_build/fw, matched rows); ORL -> orl_taxi_int (orl_build)."""
    ids = pd.Index(np.asarray(rank_ids, dtype="float64"))
    if des.name == "F":
        t = pd.read_parquet(V13_DETAIL, columns=["MVT_ID_mvt", "v10"])
        s = t.set_index("MVT_ID_mvt").v10.astype(float)
    elif des.name == "FW":
        t = pd.read_parquet(FW_SHIPPED)
        s = t.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.astype(float)
    else:
        t = pd.read_parquet(ORL_SHIPPED, columns=["MVT_ID_mvt", "orl_taxi_int"])
        s = t.set_index("MVT_ID_mvt").orl_taxi_int.astype(float)
    return s[s.index.isin(ids)]


def load_frame(des: Design, smoke: bool, log=print) -> tuple:  # pragma: no cover - real caches; exercised by --smoke
    months, n_rank = (LS.SMOKE["months"], LS.SMOKE["n_rank"]) if smoke else (None, None)
    d, is_rank, rank_ids = LS.load_frames(LS.CACHE, months=months, n_rank=n_rank, queue=des.queue, order=des.order,
                                          weather=des.weather)
    if not smoke and (int((~is_rank).sum()) != N_TRAIN or int(is_rank.sum()) != N_RANK):
        raise AssertionError(f"frame rows train {int((~is_rank).sum())} / rank {int(is_rank.sum())}")
    extra = {}
    if des.orl:
        import run_guards as G
        if G.sha256_file(O3.O.ROUTES) != O3.ROUTES_SHA:
            raise AssertionError("route table sha256 is not the registered one")
        routes = pd.read_parquet(O3.O.ROUTES)
        d = O3.S1.attach_to_cache_frame(d, routes)
        extra["n_serve_orl_matched"] = O3.serve_crosscheck(d.loc[is_rank], O3.raw_serve_keys(rank_ids), routes)
        extra["nan_gate"] = O3.nan_gate(d[O3.O.ORL_COLS], d.ap.to_numpy(), is_rank)
        del routes
    gc.collect()
    log(f"frame {len(d):,} rows ({int(is_rank.sum()):,} ranking); peak RSS {LS.peak_rss_gb():.2f} GB")
    return d, is_rank, rank_ids, extra


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--design", required=True, choices=sorted(_DESIGNS))
    ap.add_argument("--rss-limit-gib", type=float, required=True, help="in-process peak-RSS abort (the launcher passes it)")
    ap.add_argument("--smoke", action="store_true", help="lgbm_submit's SMOKE slice (months 1, 3; 5,000 ranking rows)")
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:  # pragma: no cover - real-data entry point; exercised by the --smoke run
    import run_guards as G
    args = parse_args(argv)
    des = design(args.design)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        C.SMOKE_ROOT / f"smoke_step{des.step}_{stamp}" if args.smoke else C.step_dir(des.step))
    log = lambda m: print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)
    LS.log = log
    fh = None
    if not args.smoke:
        sha = C.require_registration(des.step)
        log(f"registration sha256 {sha} logged; AMENDMENT V34.{des.step} present")
        bar = MOVE_BAR_FACTOR * (float(C.read_step_report(1)["rms_move_s"]) if des.name == "F" else FW_FOLD_RMS_MOVE_S)
        fh = G.acquire_lock()
        log("heavy-compute lock acquired (nonblocking)")
    else:
        bar = MOVE_BAR_FACTOR * FW_FOLD_RMS_MOVE_S
    G.start_rss_guard(int(args.rss_limit_gib * 1024 ** 3))
    log(f"design {des.name} ({len(des.feats)} columns); in-process RSS abort at {args.rss_limit_gib:g} GiB; "
        f"tripwire bar {bar:.2f} s")
    try:
        d, is_rank, rank_ids, extra = load_frame(des, args.smoke, log)
        cp = counterpart_of(des, rank_ids)
        rep = run(des, args.smoke, out_dir, d, is_rank, rank_ids, counterpart=cp, bar_s=bar, log=log, extra=extra)
        log(f"report -> {out_dir / C.REPORT_NAME}   wall {rep['wall_s']:.0f}s   peak RSS "
            f"{G.peak_rss() / 1024 ** 3:.2f} GiB")
    finally:
        if fh is not None:
            G.release_lock(fh)
    return 0 if rep["status"] in ("PASS", "SMOKE") else 3


if __name__ == "__main__":
    raise SystemExit(main())
