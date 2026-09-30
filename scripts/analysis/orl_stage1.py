"""ORL Stage 1 — both month-forward legs at the CHR/RCR deviation: production 93 features (control) vs 93 + the six
ORL columns (arm), target `delta`, identical test rows; gate G1 (plans/PREREG_orl_osm_route_length_2026_09_23.md).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/orl_stage1.py [--smoke]

Thin by design: leg masks, the early-stopped fit, the paired day-blocked statistics, move energy and the
prediction floor are rcr2_build's (tested there); what is new here is tested in tests/test_orl_stage1.py.
Refuses to run while the heavy lock is held (run_guards), aborts at 6 GiB RSS, refuses a silent rerun.
NOT launched by the preparing session (compute held by another lane); no upload, no git.
"""
from __future__ import annotations

import argparse, gc, hashlib, importlib.util, json, pathlib, sys, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
PREREG = ROOT / "plans" / "PREREG_orl_osm_route_length_2026_09_23.md"
HASHLOG = ROOT / "reports" / "orl_prereg_hashes.log"
REPORT1 = ROOT / "reports" / "orl_stage1_report.json"
PREDS1 = ROOT / "reports" / "orl_stage1_predictions.parquet"
LR, MAX_ROUNDS, PATIENCE = 0.05, 8000, 200
LEGS = {"L-Jul": {"fit_months": (1, 2, 3, 4, 5, 6), "test_month": 7},
        "L-Jan": {"fit_months": (8, 9, 10, 11, 12), "test_month": 1}}
BANNED = ("y", "delta", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "regime")
RSS_LIMIT = 6 * 1024 ** 3


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


O = _load("orl_build", "scripts/analysis/orl_build.py")
ARM_COLS = O.ORL_COLS + [O.ORL_FLAG]


def arm_feats(S) -> tuple:
    """(control, arm): the shipped 93 and the 93 + six ORL columns; leak gate by name on both."""
    ctrl = list(S.BASELINE_FEATS) + list(S.QUEUE_FEATS) + list(S.ORDER_FEATS) + list(S.WEATHER_FEATS)
    arm = ctrl + ARM_COLS
    leaks = [c for c in arm if c in BANNED]
    if leaks:
        raise AssertionError(f"leak gate: feature list contains label column(s) {leaks}")
    if len(ctrl) != 93:
        raise AssertionError(f"production recipe is 93 features, got {len(ctrl)}")
    return ctrl, arm


def attach_to_cache_frame(d: pd.DataFrame, routes: pd.DataFrame) -> pd.DataFrame:
    """The cache frame names the airport `ap`; attach() wants the organiser names. Same attach as serve (one path)."""
    for c in ("ap", "STAND_mvt", "RUNWAY_mvt"):
        if c not in d.columns:
            raise KeyError(c)
    keys = pd.DataFrame({"ADEP_mvt": d["ap"].to_numpy(), "STAND_mvt": d["STAND_mvt"].to_numpy(),
                         "RUNWAY_mvt": d["RUNWAY_mvt"].to_numpy()}, index=d.index)
    return pd.concat([d, O.attach(keys, routes)], axis=1)


def matched_share(flag) -> float:
    s = float(np.mean(np.asarray(flag, float)))
    if s == 0.0:
        raise AssertionError("no matched rows: the arm would be the control plus six NaN columns")
    return s


def gate_g1(stage1: dict) -> bool:
    """G1: Stage 2 runs only if BOTH legs have raw gain >= 0 AND minus-top-3 gain >= 0 (arm vs control)."""
    legs = [stage1[leg]["arm_vs_ctrl"] for leg in LEGS]
    return all(s["gain_s"] >= 0 and s["gain_minus_top3_s"] >= 0 for s in legs)


def _prereg_sha() -> str:
    h = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if f"(full {h})" not in HASHLOG.read_text():
        raise AssertionError("registration text changed after hashing (not in reports/orl_prereg_hashes.log)")
    return h


def run_stage1(smoke: bool = False) -> dict:
    started = time.monotonic()
    if not smoke and REPORT1.exists():
        raise FileExistsError("Stage 1 result exists; no silent rerun")
    R = _load("rcr2_build", "scripts/analysis/rcr2_build.py")
    G = _load("run_guards", "scripts/run_guards.py")
    fh = G.acquire_lock(); G.start_rss_guard(RSS_LIMIT)
    sha = _prereg_sha()
    S = _load("stand_ab", "scripts/stand_ab.py"); LS = _load("lgbm_submit", "scripts/lgbm_submit.py")
    ctrl, arm = arm_feats(S)
    routes = pd.read_parquet(O.ROUTES)
    months = sorted({m for leg in LEGS.values() for m in leg["fit_months"]} | {leg["test_month"] for leg in LEGS.values()})
    d = R._load_months_frame(months, ctrl)          # its `keep` carries STAND_mvt / RUNWAY_mvt through ENC_KEYS
    if smoke:
        d = d.iloc[::40].reset_index(drop=True)
    d = attach_to_cache_frame(d, routes)
    labels = R.check_labels(d.y.to_numpy(float))
    month = d.month.to_numpy().astype(int); day = d.doy.to_numpy().astype(int)
    y_all = d.y.to_numpy(float); delta_all = d.delta.to_numpy(float); proxy = d.proxy.to_numpy(float)
    ap_all = d.ap.to_numpy().astype(str)
    keys = d[S.ENC_KEYS]
    report, preds_out = {"labels": labels, "matched_share_all": matched_share(d[O.ORL_FLAG].to_numpy())}, []
    for leg, spec in LEGS.items():
        fit, es, test = R.leg_masks(month, day, spec["fit_months"], spec["test_month"])
        enc = S.infold_encodings(keys, y_all, delta_all, fit)
        for c, v in (enc.items() if isinstance(enc, dict) else zip(S.ENC, np.asarray(enc).T)):
            d[c] = v
        p = dict(LS.P); p.update(learning_rate=LR, num_threads=4, verbosity=-1)
        taxis, iters = {}, {}
        for name, feats in (("ctrl", ctrl), ("arm", arm)):
            X = LS.design_matrix(d, feats); gc.collect()
            raw, iters[name] = R.fit_leg(X, delta_all, fit, es, X[test], p, 60 if smoke else MAX_ROUNDS, PATIENCE)
            taxis[name] = R.taxi(raw, proxy[test], "delta")
            print(json.dumps({"leg": leg, "model": name, "best_iter": iters[name], "rmse": R.rmse(y_all[test], taxis[name]),
                              "n_fit": int(fit.sum()), "n_es": int(es.sum()), "n_test": int(test.sum()),
                              "secs": round(time.monotonic() - started, 1), "peak_GiB": round(G.peak_rss() / 1024 ** 3, 2)}), flush=True)
            del X; gc.collect()
        st = R.paired_stats(y_all[test], day[test], ap_all[test], taxis["ctrl"], taxis["arm"])
        flag = d[O.ORL_FLAG].to_numpy()[test]
        m1 = flag == 1
        st["gain_matched_rows_s"] = float(R.rmse(y_all[test][m1], taxis["ctrl"][m1]) - R.rmse(y_all[test][m1], taxis["arm"][m1]))
        st["gain_unmatched_rows_s"] = float(R.rmse(y_all[test][~m1], taxis["ctrl"][~m1]) - R.rmse(y_all[test][~m1], taxis["arm"][~m1])) if (~m1).any() else float("nan")
        report[leg] = {"fit_months": list(spec["fit_months"]), "test_month": spec["test_month"], "n_fit": int(fit.sum()),
                       "n_es": int(es.sum()), "n_test": int(test.sum()), "best_iter": iters, "arm_vs_ctrl": st,
                       "matched_share_test": matched_share(flag),
                       "move_energy": R.move_energy(taxis["ctrl"], taxis["arm"], ap_all[test], month[test])}
        preds_out.append(pd.DataFrame({"leg": leg, "doy": day[test], "ap": ap_all[test], "y": y_all[test], "proxy": proxy[test],
                                       "taxi_ctrl": taxis["ctrl"], "taxi_arm": taxis["arm"], "orl_matched": flag}))
        print(json.dumps({"leg": leg, "arm_vs_ctrl": {k: v for k, v in st.items() if k != "per_airport_gain_s"}}), flush=True)
    report["gate_G1_pass"] = gate_g1(report)
    report.update(tag="ORL", stage=1, seconds=time.monotonic() - started, peak_rss_GiB=G.peak_rss() / 1024 ** 3,
                  prereg_sha256=sha, smoke=smoke,
                  scope="Stage 1: single seed, lr 0.05 (registered deviation), 2025 only; a recipe-shape gate, not a board number")
    if smoke:
        print(json.dumps({"smoke": "passed; scores not inspected", "gate_G1_pass": report["gate_G1_pass"]})); G.release_lock(fh); return report
    pd.concat(preds_out, ignore_index=True).to_parquet(PREDS1, index=False)
    with REPORT1.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({"gate_G1_pass": report["gate_G1_pass"]}), flush=True)
    G.release_lock(fh)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    run_stage1(smoke=a.smoke)
    return 0


if __name__ == "__main__":
    sys.exit(main())
