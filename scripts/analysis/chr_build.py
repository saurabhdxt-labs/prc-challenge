"""CHR — chronological vs interleaved training on the ACTUAL incumbent recipe
(plans/PREREG_chr_chronological_vs_interleaved_2026_09_19.md).

Repairs the three verified defects in the retracted FWD arm: real dates, equal training volume,
identical test rows, and the real recipe (BASELINE_FEATS + in-fold encodings, target delta).
Months 1 and 7 are never touched. No lever, nothing uploadable.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/chr_build.py run [--smoke]
"""
from __future__ import annotations

import argparse, glob, hashlib, importlib.util, json, pathlib, resource, sys, time
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREREG = ROOT / "plans" / "PREREG_chr_chronological_vs_interleaved_2026_09_19.md"
REPORT = ROOT / "reports" / "chr_report.json"
PREDS = ROOT / "reports" / "chr_test_predictions.parquet"

HOLDOUT_MONTHS = (1, 7)
TEST_MONTHS = (11, 12)
TEST_DAY_FRAC = 0.40
SEED = 1709
LR, MAX_ROUNDS, PATIENCE = 0.05, 8000, 200      # registered deviation from lr 0.01 / 50k
ES_DAY_FRAC = 0.15
GAP_BAR_S = 1.0
RSS_LIMIT_GIB, TIME_LIMIT_S = 10.0, 5400.0


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def day_block_ci(y, days, pred_a, pred_b, n=2000, seed=0):
    """Bootstrap the RMSE difference (a − b) over whole days. Positive = b is better."""
    y = np.asarray(y, float)
    frame = pd.DataFrame({"d": np.asarray(days), "sa": (y - pred_a) ** 2, "sb": (y - pred_b) ** 2})
    g = frame.groupby("d").agg(sa=("sa", "sum"), sb=("sb", "sum"), n=("sa", "size"))
    sa, sb, cnt = g.sa.to_numpy(), g.sb.to_numpy(), g.n.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n, len(g)))
    out = np.sqrt(sa[idx].sum(1) / cnt[idx].sum(1)) - np.sqrt(sb[idx].sum(1) / cnt[idx].sum(1))
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def evaluate(chrono_rmse, inter_rmse, ci, bar=GAP_BAR_S):
    gap = chrono_rmse - inter_rmse          # positive = interleaved looks better = harness overstates
    ci_excludes_zero = ci[0] > 0 or ci[1] < 0
    true_shape = gap >= bar and ci_excludes_zero and ci[0] > 0
    false_shape = (not ci_excludes_zero) or gap <= 0
    return {"verdict": "WORKING" if true_shape else
                       ("NOT WORKING" if false_shape else "NOT WORKING (ambiguous, per registration)"),
            "chrono_rmse": chrono_rmse, "interleaved_rmse": inter_rmse,
            "gap_s_chrono_minus_interleaved": gap, "ci95": ci,
            "ci_excludes_zero": bool(ci_excludes_zero)}


def run(smoke=False):
    started = time.monotonic()
    if not smoke and REPORT.exists():
        raise FileExistsError("CHR result already exists")
    import lightgbm as lgb
    S = _load("stand_ab", "scripts/stand_ab.py")
    LS = _load("lgbm_submit", "scripts/lgbm_submit.py")

    paths = sorted(glob.glob(str(ROOT / "data/cache_stand/training_2025-*.parquet")))
    d = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    # `month` and `doy` are both real columns in the stand_ab caches. `doy` IS the real date-of-year and
    # is kept as an INTEGER so chronological ordering is numeric — a string doy would compare "99" > "302".
    assert {"month", "doy"} <= set(d.columns), "cache lacks month/doy"
    d["real_day"] = d.doy.astype(int)
    d = d[~d.month.isin(HOLDOUT_MONTHS)].reset_index(drop=True)
    assert not d.month.isin(HOLDOUT_MONTHS).any(), "holdout month leaked in"
    d = d[np.isfinite(d.delta.to_numpy(float))].reset_index(drop=True)
    if smoke:
        d = d.iloc[::40].reset_index(drop=True)

    rng = np.random.default_rng(SEED)
    test_pool = np.array(sorted(d.loc[d.month.isin(TEST_MONTHS), "real_day"].unique()), dtype=int)
    test_days = set(rng.choice(test_pool, size=max(1, int(round(TEST_DAY_FRAC * len(test_pool)))), replace=False))
    day = d.real_day.to_numpy()
    is_test = np.isin(day, list(test_days))
    first_test_day = int(min(test_days))

    chrono_days = np.array(sorted(int(x) for x in set(day) if int(x) < first_test_day), dtype=int)
    inter_days = np.array(sorted(int(x) for x in set(day) if x not in test_days), dtype=int)
    assert not (set(chrono_days) & test_days) and not (set(inter_days) & test_days), "TEST day in training"

    n_chrono_avail = int(np.isin(day, chrono_days).sum())
    n_inter_avail = int(np.isin(day, inter_days).sum())
    N = min(n_chrono_avail, n_inter_avail)

    def pick(days_pool, n_target, seed):
        r = np.random.default_rng(seed)
        order = r.permutation(days_pool)
        chosen, total = [], 0
        for dd in order:
            chosen.append(dd); total += int((day == dd).sum())
            if total >= n_target:
                break
        # Whole days are added until the pool overshoots, then trimmed to EXACTLY n_target rows so the
        # two arms are volume-identical by construction (the registration's equal-volume repair).
        idx = np.flatnonzero(np.isin(day, chosen))[:n_target]
        out = np.zeros(len(d), bool); out[idx] = True
        return out, chosen

    tr_chrono, days_c = pick(chrono_days, N, SEED)
    tr_inter, days_i = pick(inter_days, N, SEED + 1)
    assert tr_chrono.sum() == tr_inter.sum() == N, f"unequal volume {tr_chrono.sum()} vs {tr_inter.sum()}"
    assert not (tr_chrono & is_test).any() and not (tr_inter & is_test).any(), "TEST row in training"
    print(json.dumps({"stage": "split", "N_train_each": int(N), "n_test": int(is_test.sum()),
                      "test_days": len(test_days), "first_test_day": first_test_day,
                      "chrono_days": len(days_c), "interleaved_days": len(days_i),
                      "interleaved_days_at_or_after_first_test_day": int(sum(
                          1 for x in days_i if int(x) >= first_test_day))}), flush=True)

    feats = list(S.BASELINE_FEATS)
    y_all = d.y.to_numpy(float); delta_all = d.delta.to_numpy(float); proxy = d.proxy.to_numpy(float)
    keys = d[S.ENC_KEYS]
    results, preds = {}, {}
    for name, tr in (("CHRONO", tr_chrono), ("INTERLEAVED", tr_inter)):
        r = np.random.default_rng(SEED + 7)
        trd = np.array(sorted(set(day[tr])))
        es_days = set(r.choice(trd, size=max(1, int(round(ES_DAY_FRAC * len(trd)))), replace=False))
        es = tr & np.isin(day, list(es_days))
        fit = tr & ~es
        assert not (es & is_test).any() and not (fit & is_test).any()
        enc = S.infold_encodings(keys, y_all, delta_all, fit)     # fitted on FIT rows only
        frame = d.copy()
        for c, v in (enc.items() if isinstance(enc, dict) else zip(S.ENC, np.asarray(enc).T)):
            frame[c] = v
        X = LS.design_matrix(frame, feats)
        p = dict(LS.P); p.update(learning_rate=LR, num_threads=4, verbosity=-1)
        ds = lgb.Dataset(X[fit], label=delta_all[fit], free_raw_data=True)
        dv = lgb.Dataset(X[es], label=delta_all[es], reference=ds, free_raw_data=True)
        m = lgb.train(p, ds, 60 if smoke else MAX_ROUNDS, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        dh = m.predict(X[is_test], num_iteration=m.best_iteration)
        yhat = np.maximum(proxy[is_test] - dh, 1.0)
        results[name] = {"rmse_yhat": rmse(y_all[is_test], yhat), "rmse_delta": rmse(delta_all[is_test], dh),
                         "best_iter": int(m.best_iteration), "n_fit": int(fit.sum()), "n_es": int(es.sum())}
        preds[name] = yhat
        print(json.dumps({"arm": name, **results[name], "secs": round(time.monotonic() - started, 1)}), flush=True)
        del m, ds, dv, X, frame
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3
        if rss >= RSS_LIMIT_GIB or time.monotonic() - started > TIME_LIMIT_S:
            raise RuntimeError(f"CHR bounded scope exceeded: {rss:.2f} GiB")

    ci = day_block_ci(y_all[is_test], day[is_test], preds["CHRONO"], preds["INTERLEAVED"])
    ev = evaluate(results["CHRONO"]["rmse_yhat"], results["INTERLEAVED"]["rmse_yhat"], ci)
    if smoke:
        print(json.dumps({"smoke": "passed; scores not inspected", **ev}), flush=True); return
    pd.DataFrame({"real_day": day[is_test], "y": y_all[is_test], "proxy": proxy[is_test],
                  "yhat_chrono": preds["CHRONO"], "yhat_interleaved": preds["INTERLEAVED"]}).to_parquet(PREDS, index=False)
    out = {"tag": "CHR", "verdict": ev["verdict"], "verdict_detail": ev, "arms": results,
           "scope": "2025-only, learner+encodings on the real recipe at lr 0.05 (registered deviation). "
                    "Measures how much INTERLEAVED validation overstates vs chronological deployment. "
                    "NOT recoverable board error, nothing uploadable.",
           "split": {"N_train_each": int(N), "n_test": int(is_test.sum()), "test_days": len(test_days),
                     "first_test_day": int(first_test_day)},
           "row_predictions": str(PREDS.relative_to(ROOT)),
           "seconds": time.monotonic() - started,
           "peak_rss_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3,
           "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest()}
    with REPORT.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps(ev, indent=2), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("mode", choices=["run"]); a.add_argument("--smoke", action="store_true")
    run(a.parse_args().smoke)
