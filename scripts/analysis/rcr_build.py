"""RCR — recipe rebuild under CHRONOLOGICAL validation
(plans/PREREG_rcr_recipe_chronological_rebuild_2026_09_19.md, sha256 9a0fa766).

Five frozen arms on one chronological split: every arm trains ONLY on days strictly before the
earliest TEST day, on identical TEST rows, with encodings fitted on its own fit rows. Months 1 and 7
are never touched. Nothing here is uploadable; a WORKING verdict authorises a ship registration only.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/rcr_build.py run [--smoke]
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, pathlib, resource, sys, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREREG = ROOT / "plans" / "PREREG_rcr_recipe_chronological_rebuild_2026_09_19.md"
REPORT = ROOT / "reports" / "rcr_report.json"
PREDS = ROOT / "reports" / "rcr_test_predictions.parquet"

HOLDOUT_MONTHS = (1, 7)
TEST_MONTHS = (11, 12)
TEST_DAY_FRAC = 0.40
SEED = 1709
LR, MAX_ROUNDS, PATIENCE = 0.05, 8000, 200      # registered deviation, applied identically to every arm
ES_DAY_FRAC = 0.15
GAIN_BAR_S = 2.0
RSS_LIMIT_GIB, TIME_LIMIT_S = 12.0, 16200.0     # 4.5 h ceiling for all five arms

#: The five arms, frozen by the registration. `blocks` names feature blocks in stand_ab.
ARMS = {
    "A0": {"blocks": ("BASELINE", "QUEUE", "ORDER", "WEATHER"), "target": "delta", "params": {}},
    "A1": {"blocks": ("BASELINE", "QUEUE", "ORDER", "WEATHER", "DAY"), "target": "delta", "params": {}},
    "A2": {"blocks": ("BASELINE", "QUEUE", "ORDER", "WEATHER"), "target": "y", "params": {}},
    "A3": {"blocks": ("BASELINE", "QUEUE", "ORDER", "WEATHER"), "target": "delta",
           "params": {"num_leaves": 511, "min_data_in_leaf": 20}},
    "A4": {"blocks": ("BASELINE", "QUEUE", "ORDER", "WEATHER"), "target": "delta",
           "params": {"num_leaves": 127, "min_data_in_leaf": 20, "feature_fraction": 0.6}},
}
BANNED_IN_FEATURES = ("y", "delta", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "regime")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def _blocks(S):
    return {"BASELINE": list(S.BASELINE_FEATS), "QUEUE": list(S.QUEUE_FEATS),
            "ORDER": list(S.ORDER_FEATS), "WEATHER": list(S.WEATHER_FEATS), "DAY": list(S.DAY_FEATS)}


def arm_feats(arm: str, S=None) -> list:
    """The exact, ordered feature list for `arm`. Order is fixed so design matrices are comparable."""
    S = S or _load("stand_ab", "scripts/stand_ab.py")
    b = _blocks(S)
    out = []
    for name in ARMS[arm]["blocks"]:
        out.extend(b[name])
    leaks = [c for c in out if c in BANNED_IN_FEATURES]
    if leaks:
        raise AssertionError(f"leak gate 1: {arm} feature list contains label column(s) {leaks}")
    return out


def chrono_train_mask(day, is_test) -> np.ndarray:
    """Leak gate 4: training is every row on a day strictly before the earliest TEST day."""
    day = np.asarray(day); is_test = np.asarray(is_test, bool)
    if not is_test.any():
        raise ValueError("no TEST rows")
    first_test_day = int(np.min(day[is_test]))
    tr = np.asarray(day) < first_test_day
    if (tr & is_test).any():
        raise AssertionError("leak gate 4 violated: a TEST row is in training")
    return tr


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def evaluate_arm(gain: float, ci) -> dict:
    """Registered shapes: WORKING iff gain >= 2.0 s AND the day-blocked CI excludes zero.
    0 < gain < 2.0 is NOT WORKING (ambiguous), per the registration."""
    excludes_zero = ci[0] > 0 or ci[1] < 0
    if gain >= GAIN_BAR_S and excludes_zero and ci[0] > 0:
        v = "WORKING"
    elif 0 < gain < GAIN_BAR_S and excludes_zero:
        v = "NOT WORKING (ambiguous: real but below the registered bar)"
    else:
        v = "NOT WORKING"
    return {"verdict": v, "gain_s": gain, "ci95": list(ci), "ci_excludes_zero": bool(excludes_zero)}


def run(smoke: bool = False) -> None:
    started = time.monotonic()
    if not smoke and REPORT.exists():
        raise FileExistsError("RCR result already exists — a rerun would be a second look at the same bar")
    import lightgbm as lgb
    S = _load("stand_ab", "scripts/stand_ab.py")
    LS = _load("lgbm_submit", "scripts/lgbm_submit.py")
    LF = _load("lgbm_fold", "scripts/lgbm_fold.py")
    CHR = _load("chr_build", "scripts/analysis/chr_build.py")   # reuse the validated day_block_ci

    months = [m for m in range(1, 13) if m not in HOLDOUT_MONTHS]
    all_feats = arm_feats("A1", S)                              # A1 is the superset (includes DAY)
    frames, _ = LF._load_months(S.CACHE, months, all_feats, S.QCACHE, S.DCACHE, S.OCACHE, S.WCACHE)
    d = pd.concat(frames, ignore_index=True); del frames
    assert {"month", "doy"} <= set(d.columns), "cache lacks month/doy"
    d["real_day"] = d.doy.astype(int)
    assert not d.month.isin(HOLDOUT_MONTHS).any(), "holdout month leaked in"
    d = d[np.isfinite(d.delta.to_numpy(float))].reset_index(drop=True)
    if smoke:
        d = d.iloc[::40].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    pool = np.array(sorted(d.loc[d.month.isin(TEST_MONTHS), "real_day"].unique()), dtype=int)
    test_days = set(rng.choice(pool, size=max(1, int(round(TEST_DAY_FRAC * len(pool)))), replace=False))
    day = d.real_day.to_numpy()
    is_test = np.isin(day, list(test_days))
    tr_all = chrono_train_mask(day, is_test)
    first_test_day = int(day[is_test].min())
    assert day[tr_all].max() < first_test_day, "leak gate 4"
    print(json.dumps({"stage": "split", "n_train": int(tr_all.sum()), "n_test": int(is_test.sum()),
                      "first_test_day": first_test_day, "test_days": len(test_days)}), flush=True)

    y_all = d.y.to_numpy(float); delta_all = d.delta.to_numpy(float); proxy = d.proxy.to_numpy(float)
    keys = d[S.ENC_KEYS]
    r = np.random.default_rng(SEED + 7)
    trd = np.array(sorted(set(day[tr_all])))
    es_days = set(r.choice(trd, size=max(1, int(round(ES_DAY_FRAC * len(trd)))), replace=False))
    es = tr_all & np.isin(day, list(es_days))
    fit = tr_all & ~es
    assert not (es & is_test).any() and not (fit & is_test).any(), "leak gate: TEST row in fit/es"

    enc = S.infold_encodings(keys, y_all, delta_all, fit)        # fitted on FIT rows only, shared split
    frame = d.copy()
    for c, v in (enc.items() if isinstance(enc, dict) else zip(S.ENC, np.asarray(enc).T)):
        frame[c] = v

    results, preds = {}, {}
    for arm in ARMS:
        feats = arm_feats(arm, S)
        tgt = ARMS[arm]["target"]
        label = delta_all if tgt == "delta" else y_all
        X = LS.design_matrix(frame, feats)
        p = dict(LS.P); p.update(learning_rate=LR, num_threads=4, verbosity=-1)
        p.update(ARMS[arm]["params"])
        ds = lgb.Dataset(X[fit], label=label[fit], free_raw_data=True)
        dv = lgb.Dataset(X[es], label=label[es], reference=ds, free_raw_data=True)
        m = lgb.train(p, ds, 60 if smoke else MAX_ROUNDS, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        raw = m.predict(X[is_test], num_iteration=m.best_iteration)
        if not np.all(np.isfinite(raw)):                         # in-flight abort
            raise RuntimeError(f"{arm}: non-finite predictions")
        yhat = np.maximum(proxy[is_test] - raw, 1.0) if tgt == "delta" else np.maximum(raw, 1.0)
        preds[arm] = yhat
        results[arm] = {"rmse": rmse(y_all[is_test], yhat), "best_iter": int(m.best_iteration),
                        "n_feats": len(feats), "target": tgt, "params": ARMS[arm]["params"]}
        print(json.dumps({"arm": arm, **results[arm], "secs": round(time.monotonic() - started, 1)}), flush=True)
        del m, ds, dv, X
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3
        if rss >= RSS_LIMIT_GIB or time.monotonic() - started > TIME_LIMIT_S:
            raise RuntimeError(f"RCR bounded scope exceeded: {rss:.2f} GiB / {time.monotonic()-started:.0f}s")

    base = results["A0"]["rmse"]
    table = {}
    for arm in ARMS:
        if arm == "A0":
            continue
        ci = CHR.day_block_ci(y_all[is_test], day[is_test], preds["A0"], preds[arm])
        table[arm] = evaluate_arm(base - results[arm]["rmse"], ci)
    best = max(table, key=lambda a: table[a]["gain_s"])
    verdict = table[best]["verdict"] if table[best]["verdict"] == "WORKING" else "NOT WORKING"

    if smoke:
        print(json.dumps({"smoke": "passed; scores not inspected"}), flush=True); return
    pd.DataFrame({"real_day": day[is_test], "y": y_all[is_test], "proxy": proxy[is_test],
                  **{f"yhat_{a}": preds[a] for a in ARMS}}).to_parquet(PREDS, index=False)
    out = {"tag": "RCR", "verdict": verdict, "best_arm": best, "a0_rmse": base,
           "arms": results, "vs_A0": table,
           "scope": "2025-only, chronological, lr 0.05 registered deviation. A 2025 gain is NOT a board gain "
                    "(v22 was -2.60x wrong-signed). Nothing uploadable.",
           "split": {"n_train": int(tr_all.sum()), "n_test": int(is_test.sum()),
                     "first_test_day": first_test_day, "test_days": len(test_days)},
           "seconds": time.monotonic() - started,
           "peak_rss_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3,
           "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest()}
    with REPORT.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps({"verdict": verdict, "best_arm": best, "vs_A0": table}, indent=2), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("mode", choices=["run"]); a.add_argument("--smoke", action="store_true")
    run(a.parse_args().smoke)
