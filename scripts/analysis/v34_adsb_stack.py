"""v34 step 2 — the ADS-B stack retrained on the F-NSC fold record (plans/PREREG_nsc_no_seconds_2026_09_24.md §7 step 2,
AMENDMENT V34.2).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/v34_launch.py --step 2 --detach [--smoke]
    (the launcher runs: v34_adsb_stack.py --rss-limit-gib 6 [--smoke])

The shipped stack (data/adsb/v2/models_v4, MANIFEST sha b251f316...) was trained by `adsb_stack.py train-artifacts` on fold
A's matched rows with F = the F record's `treatment` and target r = delta - F, joined to features_2025janjul_v4.parquet;
its gate was written from the 2025/2026 v4 tables' joined shares, the ADN.7 transfer verdict and the owner's LIRF override.
The stack carries no seconds column itself (33 ADS-B features + F, proxy, hour, ap), but its input F and its target are
relative to F, so v34 retrains it on step 1's record. Identical recipe, one change (the record):

  frame      load_frame(step-1 record, features_2025janjul_v4.parquet, the Jan/Jul 2025 raw months): adsb_stack.load_frame's
             construction with the record as a parameter (its F guard value is step 1's reported treatment RMSE)
  train      adsb_stack.train_artifacts(frame, models_v4's 33 feats, seeds 0/1/2, 5 day folds, ship_learner B), gate empty
  transfer   ADN.7 re-run on this frame (adsb_stack.fit_stacker arm G seed 0, leave-one-airport-out on EHAM/EDDM/LSZH/LEBL,
             adsb_stack.score_arm / transfer_rule) -- never written to reports/adsb_transfer.json
  gate       adsb_ship.gate_table(joined shares of the SAME v4 tables, transfer verdict, no KS holds, the LIRF RWC.3 override)
             re-check: it must EQUAL models_v4's gate entry for entry, else status FAIL
  OOF        adsb_stack.predict_residual(frame, the new models, oof=True) -> B = F + resid vs F (adsb_stack.score_arm):
             clause C1 (day-block CI lower > 0) must hold, as it did for models_v4 (+11.4 s)

Outputs (new paths only): data/nsc_v34/step2_adsb_stack/models_v4_nsc/ (G_s*_k*.txt, N_s*_k*.pt + _stats.npz, MANIFEST.json
with the gate), report.json. The NN trains and predicts in scripts/adsb_nn_worker.py subprocesses (BC-5): this process
imports LightGBM, never torch.
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
for _p in (ROOT / "scripts", ROOT / "scripts" / "analysis"):
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
AS = _load("adsb_stack", "scripts/adsb_stack.py")
SH = _load("adsb_ship", "scripts/adsb_ship.py")

STEP = 2
ADSB = ROOT / "data" / "adsb" / "v2"
MODELS_V4 = ADSB / "models_v4"
MODELS_V4_MANIFEST_SHA = "b251f3162d00d477113717031feffb7774929e3045a7705fd17a1e938f502628"
T25 = ADSB / "features_2025janjul_v4.parquet"
T26 = ADSB / "features_2026janjul_v4.parquet"
RAW_MONTHS = [ROOT / "data" / "raw" / "training_2025-01-01_2025-02-01.parquet",
              ROOT / "data" / "raw" / "training_2025-07-01_2025-08-01.parquet"]
RANKING = ROOT / "data" / "raw" / "ranking.parquet"
MODELS_NAME = "models_v4_nsc"
V4_OVERRIDES = {"LIRF": "RWC.3 SHIP"}           # models_v4 gate_evidence["overrides"] (owner, SHIP-ADN.5 / RWC.3)
GUARD_TOL = 1e-6


def v4_contract(models_dir=MODELS_V4) -> dict:
    """The shipped stack's recipe: feats, seeds, folds, ship learner, gate (read, never written)."""
    man = json.loads((pathlib.Path(models_dir) / "MANIFEST.json").read_text())
    return {"feats": list(man["feats"]), "seeds": list(man["seeds"]), "folds": int(man["folds"]),
            "ship_learner": man.get("ship_learner", "B"), "gate": man["gate"]}


def load_frame(record, features_path, raw_months) -> pd.DataFrame:
    """adsb_stack.load_frame's frame with the fold record as a parameter (its F guard is the caller's: f_guard)."""
    rec = pd.read_parquet(record, columns=["row", "month", "ap", "y", "proxy", "delta", "treatment"])
    feat = pd.read_parquet(features_path)
    if not (len(feat) == len(rec) and (feat.row.to_numpy() == rec.row.to_numpy()).all()):
        raise ValueError("feature table rows are not the record's rows in order")
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]) for p in raw_months])
    mvt = raw.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").MVT_TIME_UTC_mvt
    t = pd.to_datetime(feat.MVT_ID_mvt.map(mvt), utc=True)
    if t.isna().any():
        raise ValueError("feature rows without an MVT time")
    fr = pd.concat([rec.reset_index(drop=True),
                    feat.drop(columns=["row"]).reset_index(drop=True)], axis=1).rename(columns={"treatment": "F"})
    fr["day"] = t.dt.strftime("%Y-%m-%d").to_numpy()
    fr["hour"] = t.dt.hour.to_numpy()
    fr["fold"] = AS.day_folds(fr.day)
    return fr


def f_guard(fr: pd.DataFrame) -> float:
    """adsb_stack's F guard statistic: RMSE of max(proxy - F, 1) against y on every frame row."""
    return float(np.sqrt(np.mean((C.taxi(fr.proxy, fr.F) - fr.y.to_numpy(float)) ** 2)))


def transfer(fr: pd.DataFrame, feats, stack=AS) -> dict:
    """ADN.7 (adsb_stack.transfer_test's procedure, results returned instead of written to reports/adsb_transfer.json)."""
    res = {}
    for a in stack.TRANSFER_AIRPORTS:
        tr, pr = (fr.ap != a).to_numpy(), (fr.ap == a).to_numpy()
        g = stack.fit_stacker(fr, list(feats), "G", 0, train_rows=tr, predict_rows=pr)
        b = stack.fit_stacker(fr, [], "G", 0, train_rows=tr, predict_rows=pr)
        sub = fr[pr].reset_index(drop=True)
        res[a] = stack.score_arm(sub, g[pr], base_delta=b[pr])
    return {"airports": res, "rule": stack.transfer_rule(res)}


def recheck_gate(c25: dict, c26: dict, transfer_pass: bool, v4_gate: dict) -> tuple:
    """(gate, equal_to_v4): adsb_ship.gate_table on the tables' joined shares, the transfer verdict, no KS holds and the
    LIRF override; compared entry for entry (allowed, rule text, c25, c26) with the shipped gate."""
    gate = SH.gate_table(c25, c26, transfer_pass=bool(transfer_pass), holds={}, coverage_overrides=V4_OVERRIDES)
    return gate, gate == v4_gate


def run(smoke: bool, out_dir, fr: pd.DataFrame, c25: dict, c26: dict, stack=AS, step1_rmse=None, log=print,
        allow_roots=(C.OUT_ROOT, C.SMOKE_ROOT)) -> dict:
    """Train, transfer, gate, OOF score, report. `fr` from load_frame (the smoke subset under --smoke); c25/c26 the joined
    shares of the v4 tables; `step1_rmse` step 1's reported treatment RMSE (the F guard; None under --smoke)."""
    t0 = time.time()
    out_dir = pathlib.Path(out_dir)
    models = C.check_output_path(out_dir / MODELS_NAME, allow_roots=allow_roots, allow_files=())
    rep_path = C.check_output_path(out_dir / C.REPORT_NAME, allow_roots=allow_roots, allow_files=())
    v4 = v4_contract()
    feats = C.assert_no_seconds(v4["feats"], "models_v4 feats")
    guard = f_guard(fr)
    if not smoke and abs(guard - float(step1_rmse)) > GUARD_TOL * max(1.0, abs(guard)):
        raise AssertionError(f"F guard failed: the frame's F RMSE {guard:.6f} != step 1's reported {step1_rmse}")
    seeds = (0,) if smoke else tuple(v4["seeds"])
    man = stack.train_artifacts(fr, feats, models, seeds=seeds, gate={}, ship_learner=v4["ship_learner"])
    log(f"trained {len(seeds)} seed(s) x {stack.N_FOLDS} folds -> {models}")
    tr = transfer(fr, feats, stack)
    gate, same = recheck_gate(c25, c26, tr["rule"]["PASS"], v4["gate"])
    mp = models / "MANIFEST.json"
    man = json.loads(mp.read_text())
    if man.get("gate"):
        raise AssertionError("train_artifacts wrote a gate; refusing to overwrite it")
    man["gate"] = gate
    man["gate_evidence"] = {"source": "v34 step 2 (AMENDMENT V34.2): adsb_ship.gate_table re-derived on the v4 tables",
                            "transfer_PASS": bool(tr["rule"]["PASS"]), "transfer_n_pass": tr["rule"]["n_pass"],
                            "joined_share_2025": c25, "joined_share_2026": c26, "overrides": V4_OVERRIDES, "holds": {},
                            "equals_models_v4_gate": bool(same), "fold_record": "step 1 F-NSC record"}
    mp.write_text(json.dumps(C.jsonable(man), indent=1))
    resid = np.asarray(stack.predict_residual(fr, models, oof=True), float)
    oof = stack.score_arm(fr, fr.F.to_numpy(float) + resid)
    ok = bool(same and oof["clauses"]["C1"] and tr["rule"]["PASS"])
    status = "SMOKE" if smoke else ("PASS" if ok else "FAIL")
    outputs = {C.rel(p): C.sha256(p) for p in sorted(models.iterdir()) if p.is_file()}
    rep = {"step": STEP, "tag": "V34.2 ADS-B stack on F-NSC", "smoke": bool(smoke), "status": status,
           "prereg": C.rel(C.PREREG), "features": feats, "n_features": len(feats), "seeds": list(seeds),
           "folds": stack.N_FOLDS, "ship_learner": v4["ship_learner"], "n_rows": int(len(fr)),
           "n_train_joined_rows": man.get("n_train_joined_rows"),
           "f_guard_rmse": guard, "step1_treatment_rmse": step1_rmse,
           "transfer": tr, "gate": gate, "gate_equals_v4": bool(same), "oof": oof,
           "models_dir": C.rel(models), "outputs": outputs, "wall_s": round(time.time() - t0, 1),
           "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    rep = C.jsonable(rep)                      # what is written is what is returned
    C.write_json_new(rep, rep_path, allow_roots=allow_roots, allow_files=())
    log(f"gate equals models_v4: {same}; transfer PASS {tr['rule']['PASS']}; OOF B vs F-NSC gain {oof['gain_s']:+.3f} s "
        f"CI {oof['ci95_day_block']} C1 {oof['clauses']['C1']}; status {status}")
    return rep


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rss-limit-gib", type=float, required=True, help="in-process peak-RSS abort (the launcher passes it)")
    ap.add_argument("--smoke", action="store_true",
                    help="the SHIPPED F record's first 5 days/month, seed 0, scratch dir, no lock (code path only)")
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:  # pragma: no cover - real-data entry point; exercised by the --smoke run
    import run_guards as G
    args = parse_args(argv)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        C.SMOKE_ROOT / f"smoke_step2_{stamp}" if args.smoke else C.step_dir(STEP))
    fh = None
    log = lambda m: print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)
    if G.sha256_file(MODELS_V4 / "MANIFEST.json") != MODELS_V4_MANIFEST_SHA:
        raise RuntimeError("models_v4 MANIFEST.json is not the shipped one")
    if args.smoke:
        # smoke: the shipped F record stands in for step 1's (not yet run); code path only, never a result
        record, step1_rmse = ROOT / "data" / "cache_stand" / "fold_preds_queue_order.parquet", None
    else:
        sha = C.require_registration(STEP)
        log(f"registration sha256 {sha} logged; AMENDMENT V34.2 present")
        s1 = C.read_step_report(1)
        record = ROOT / next(iter(s1["outputs"]))
        step1_rmse = float(s1["treatment_rmse"])
        fh = G.acquire_lock()
        log("heavy-compute lock acquired (nonblocking)")
    G.start_rss_guard(int(args.rss_limit_gib * 1024 ** 3))
    log(f"in-process RSS abort at {args.rss_limit_gib:g} GiB")
    try:
        fr = load_frame(record, T25, RAW_MONTHS)
        if args.smoke:
            fr = AS.smoke_subset(fr)
        t25, t26 = SH.tables_with_airport(T25, record, RAW_MONTHS, T26, RANKING)
        rep = run(args.smoke, out_dir, fr, SH.joined_share(t25), SH.joined_share(t26), step1_rmse=step1_rmse, log=log)
        log(f"report -> {out_dir / C.REPORT_NAME}   wall {rep['wall_s']:.0f}s   peak RSS "
            f"{G.peak_rss() / 1024 ** 3:.2f} GiB")
    finally:
        if fh is not None:
            G.release_lock(fh)
    return 0 if rep["status"] in ("PASS", "SMOKE") else 3


if __name__ == "__main__":
    raise SystemExit(main())
