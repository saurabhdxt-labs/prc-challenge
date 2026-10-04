"""R13 (plans/PREREG_r13_tuning_2026_09_27.md) — a CatBoost base F on the R12 design, derived blend arms, and the
screen comparison against the v38 screen baseline.

    CatBoost arm: r12_validate --arms cat (the fit is routed here by `dispatch`; the fit itself runs in
        scripts/analysis/r13_cat_worker.py under ~/.venvs/prc-catboost, BC-5)
    /opt/homebrew/bin/python3.11 -B scripts/analysis/r13_cat.py assemble --base-val <val dir> --cat-val <val dir> \
        --yhead-val <val dir> --out data/r12/<arms dir>
    /opt/homebrew/bin/python3.11 -B scripts/analysis/r13_cat.py compare --base-oof <oof_r12.parquet> \
        --stack <r13 stack dir> --arms-dir <arms dir> --arms cat,lgbcat,blendy --out <json>

CatFitter has lgbm_fold.fit_arm's signature and return contract (delta = pooled mean over seeds on the prediction
rows, single = per seed, best_iter, n_ref), so r12_validate.run scores it exactly as it scores a LightGBM arm:
early stopping on the months-3+9 rows at seed 0 -> n_ref = lgbm_submit.n_refit(best_iter, n_train, n_fit) -> a refit
per seed on every training row. The stopping-set guard is lgbm_submit.early_stop's (RMSE <= 0.85 x proxy-only).
Derived arms (the stack's F is a delta): lgbcat = 0.5 F_r12 + 0.5 F_cat; blendy = proxy - blend_taxi_times(taxi_yhead,
taxi_r12, 0.5) (candidate B).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
CAT_PY = pathlib.Path.home() / ".venvs" / "prc-catboost" / "bin" / "python3"
WORKER = ROOT / "scripts" / "analysis" / "r13_cat_worker.py"

#: registered in PREREG R13 (arm 1); the learning rate is set per tier by cat_config. CB_LR["full"] = 0.02 is the
#: pre-amendment value: the registration's AMEND-A1 replaced the full-tier rate by 0.05, and the CatBoost holdout behind
#: the stored lgbcat stack was run with --screen (0.05, seeds 0-2). The constant is kept as it was when that stack was
#: built; a rebuild of the stack follows the amendment by running --screen.
CB_BASE = dict(loss_function="RMSE", depth=8, l2_leaf_reg=3.0, border_count=254, boosting_type="Plain",
               bootstrap_type="Bernoulli", subsample=0.8, thread_count=8)
CB_LR = {"screen": 0.05, "full": 0.02, "smoke": 0.05}
CB_CAP = {"screen": 30_000, "full": 30_000, "smoke": 60}
CB_PATIENCE = {"screen": 200, "full": 200, "smoke": 50}
ES_SEED = 0
ES_MAX_RATIO = 0.85                                        # lgbm_submit.ES_MAX_RATIO_TO_PROXY_ONLY
BLEND_W = 0.5
KEY_COLS = ("month", "pos", "MVT_ID_mvt", "y", "proxy")
KEY_COLS_2026 = ("MVT_ID_mvt", "proxy")
AIRPORTS_5 = ("EDDF", "EDDM", "EHAM", "LEBL", "LSZH")
MIN_DG5 = 100.0
N_ARMS = 3                                                 # Bonferroni family (cat, lgbcat, blendy)


def cat_config(tier: str) -> dict:
    """The registered CatBoost configuration for a tier (screen / full / smoke)."""
    if tier not in CB_LR:
        raise ValueError(f"unknown tier {tier!r}; tiers are {sorted(CB_LR)}")
    return {"params": dict(CB_BASE, learning_rate=CB_LR[tier]), "cap": CB_CAP[tier], "patience": CB_PATIENCE[tier]}


def n_refit(best_iter: int, n_all: int, n_fit: int) -> int:
    """lgbm_submit.n_refit, restated here so the driver never needs lightgbm to size a CatBoost refit (tested equal)."""
    return max(50, int(best_iter * n_all / n_fit))


# ---------------------------------------------------------------- work files

def write_work(workdir, X, label, masks: dict, seeds, cfg: dict, n_ref=None) -> None:
    """X.npy (float32, written once) + in.npz for the worker. masks: fit, es, tr, pred (bool, len(X))."""
    d = pathlib.Path(workdir)
    d.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X)
    label = np.asarray(label, dtype="float64")
    if label.shape != (len(X),):
        raise ValueError(f"label shape {label.shape} for {len(X)} rows")
    for k in ("fit", "es", "tr", "pred"):
        m = np.asarray(masks[k], bool)
        if m.shape != (len(X),):
            raise ValueError(f"mask {k} has shape {m.shape} for {len(X)} rows")
        if not m.any():
            raise ValueError(f"mask {k} selects no row")
    for k in ("fit", "es", "tr"):
        if not np.isfinite(label[np.asarray(masks[k], bool)]).all():
            raise ValueError(f"non-finite label on {k} rows")
    xp = d / "X.npy"
    if not xp.exists():
        np.save(xp, X.astype(np.float32, copy=False))
    z = {k: np.asarray(masks[k], bool) for k in ("fit", "es", "tr", "pred")}
    z.update(label=label, seeds=np.asarray(list(seeds), dtype="int64"), es_seed=np.array(ES_SEED),
             params=np.array(json.dumps(cfg["params"], sort_keys=True)), cap=np.array(int(cfg["cap"])),
             patience=np.array(int(cfg["patience"])))
    if n_ref is not None:
        z["n_ref"] = np.array(int(n_ref))
    np.savez(d / "in.npz", **z)


def read_out(workdir, mode: str, rc: int) -> dict:
    """The worker's result for `mode`; raises RuntimeError with the worker's message on a failure."""
    p = pathlib.Path(workdir) / f"{mode}_out.npz"
    if not p.exists():
        raise RuntimeError(f"CatBoost worker ({mode}) exited {rc} and wrote no result")
    z = dict(np.load(p, allow_pickle=False))
    if "error" in z:
        raise RuntimeError(f"CatBoost worker ({mode}) failed: {z['error']}")
    if rc != 0:
        raise RuntimeError(f"CatBoost worker ({mode}) exited {rc}")
    return z


def _run_worker(mode: str, workdir) -> int:  # pragma: no cover - the real subprocess (process-level test covers it)
    return subprocess.run([str(CAT_PY), "-B", str(WORKER), mode, str(workdir)], cwd=ROOT).returncode


def _rmse(a, b) -> float:
    return float(np.sqrt(((np.asarray(a, float) - np.asarray(b, float)) ** 2).mean()))


class CatFitter:
    """A drop-in for lgbm_fold.fit_arm that fits CatBoost in the worker process. target must be delta. The LightGBM
    arguments params / nest / patience / pa_floor / pa_trees are accepted for the call contract and not used (CatBoost
    takes its parameters, cap and patience from `cfg`, cat_config); fit_arm's optional on_seed / weight / es_eval have
    no CatBoost meaning and are refused (TypeError)."""

    def __init__(self, workdir, cfg: dict, runner=_run_worker, log=print):
        self.workdir, self.cfg, self.runner, self.log = pathlib.Path(workdir), cfg, runner, log

    def __call__(self, fold, X, params, nest, patience, pa_floor, seeds, per_airport, pa_trees, name,
                 target="delta") -> dict:
        if target != "delta":
            raise ValueError(f"CatFitter fits the delta target only, got {target!r}")
        if per_airport:
            raise ValueError("CatFitter has no per-airport blend")
        seeds = list(seeds)
        te, tr, fit, es = (np.asarray(fold[k], bool) for k in ("te", "tr", "fit", "es"))
        if (fit & es).any() or ((fit | es) & ~tr).any() or (te & tr).any():
            raise ValueError("masks must satisfy fit, es disjoint subsets of tr, and te outside tr")
        y, dlt, proxy = (np.asarray(fold[k], float) for k in ("y", "dlt", "proxy"))
        masks = {"fit": fit, "es": es, "tr": tr, "pred": te}
        self.log(f"[{name}] CatBoost early stopping on {X.shape[1]} features, {int(fit.sum()):,} fit / "
                 f"{int(es.sum()):,} stopping rows, lr {self.cfg['params']['learning_rate']}, cap {self.cfg['cap']:,}")
        write_work(self.workdir, X, dlt, masks, seeds, self.cfg)
        es_out = read_out(self.workdir, "es", self.runner("es", self.workdir))
        best = int(es_out["best_iter"][0])
        pred_es = np.asarray(es_out["pred_es"], float)
        es_rmse = _rmse(y[es], np.maximum(proxy[es] - pred_es, 1.0))
        proxy_only = _rmse(y[es], np.maximum(proxy[es] - dlt[fit].mean(), 1.0))
        if es_rmse > ES_MAX_RATIO * proxy_only:
            raise RuntimeError(f"stopping-set RMSE {es_rmse:.2f} is not {ES_MAX_RATIO:.0%} of proxy-only "
                               f"{proxy_only:.2f}: the feature path is broken, refusing to refit")
        n_ref = n_refit(best, int(tr.sum()), int(fit.sum()))
        self.log(f"[{name}] best_iter {best:,} (cap {self.cfg['cap']:,}{', CAP-BOUND' if best >= self.cfg['cap'] else ''})"
                 f"; stopping-set RMSE {es_rmse:.2f} vs proxy-only {proxy_only:.2f}; n_ref {n_ref:,}")
        write_work(self.workdir, X, dlt, masks, seeds, self.cfg, n_ref=n_ref)
        ref = read_out(self.workdir, "refit", self.runner("refit", self.workdir))
        preds = np.asarray(ref["preds"], float)
        if preds.shape != (len(seeds), int(te.sum())):
            raise RuntimeError(f"worker returned predictions of shape {preds.shape}, expected "
                               f"{(len(seeds), int(te.sum()))}")
        single = {s: preds[i] for i, s in enumerate(seeds)}
        return {"delta": preds.mean(axis=0), "pooled": preds.mean(axis=0), "single": single, "best_iter": best,
                "n_ref": n_ref, "cap_bound": bool(best >= self.cfg["cap"]), "es_rmse_NOT_A_RESULT": es_rmse,
                "es_proxy_only_rmse": proxy_only, "seeds": seeds, "n_features": int(X.shape[1]), "target": target}


def dispatch(lgb_fit, cat_fit, cat_name: str = "r12:cat"):
    """A fit callable for r12_validate.run: the arm named `cat_name` goes to cat_fit, every other arm to lgb_fit."""
    def fit(*args, **kw):
        name = args[9] if len(args) > 9 else kw.get("name")
        return (cat_fit if name == cat_name else lgb_fit)(*args, **kw)
    return fit


# ---------------------------------------------------------------- derived arms

def _check_keys(a: pd.DataFrame, b: pd.DataFrame, cols, what: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{what}: {len(a)} vs {len(b)} rows")
    for c in cols:
        x, z = a[c].to_numpy(), b[c].to_numpy()
        same = (np.allclose(x.astype(float), z.astype(float), rtol=0, atol=1e-9, equal_nan=True)
                if x.dtype.kind in "fiu" and z.dtype.kind in "fiu" else np.array_equal(x, z))
        if not same:
            raise ValueError(f"{what}: column {c} differs")


def derive(base: pd.DataFrame, cat: pd.DataFrame, yhead, blend, keys=KEY_COLS) -> pd.DataFrame:
    """base (with pred_r12, taxi_r12) + pred_cat / taxi_cat from `cat` + taxi_yhead from `yhead` -> pred_cat, pred_lgbcat,
    pred_blendy (+ taxi_ columns). Rows must be identical on `keys`. `blend` = lgbm_fold.blend_taxi_times. yhead None
    (AMEND-A1 full tier: the winner lgbcat only) -> no blendy."""
    _check_keys(base, cat, keys, "cat vs base")
    if yhead is not None:
        _check_keys(base, yhead, keys, "yhead vs base")
    out = base.copy()
    proxy = out.proxy.to_numpy(float)
    f_r12, f_cat = out.pred_r12.to_numpy(float), cat.pred_cat.to_numpy(float)
    if not (np.isfinite(f_r12).all() and np.isfinite(f_cat).all()):
        raise ValueError("non-finite F in a blend operand")
    out["pred_cat"], out["taxi_cat"] = f_cat, cat.taxi_cat.to_numpy(float)
    out["pred_lgbcat"] = BLEND_W * f_r12 + (1.0 - BLEND_W) * f_cat
    out["taxi_lgbcat"] = np.maximum(proxy - out.pred_lgbcat.to_numpy(float), 1.0)
    if yhead is None:
        return out
    tb = np.asarray(blend(yhead.taxi_yhead.to_numpy(float), out.taxi_r12.to_numpy(float), BLEND_W), float)
    out["taxi_blendy"], out["pred_blendy"] = tb, proxy - tb
    return out


def assemble(base_dir, cat_dir, yhead_dir, out_dir, blend, now=None, split: str = "holdout") -> dict:
    """A val-format directory (predictions.parquet [holdout only], predictions_2026.parquet, report.json with the
    split) holding the base, cat and derived arms: holdout for r12_stack oof, refit for r12_stack serve's --f26.
    yhead_dir None: no blendy (AMEND-A1). Never overwrites."""
    if split not in ("holdout", "refit"):
        raise ValueError(f"split must be holdout or refit, got {split!r}")
    out = pathlib.Path(out_dir)
    if out.exists():
        raise FileExistsError(f"{out} exists: never overwritten")
    dirs = [("base", base_dir), ("cat", cat_dir)] + ([("yhead", yhead_dir)] if yhead_dir is not None else [])
    reps = {}
    for tag, d in dirs:
        reps[tag] = json.loads((pathlib.Path(d) / "report.json").read_text())
        if reps[tag]["split"] != split or reps[tag]["smoke"] != reps["base"]["smoke"]:
            raise ValueError(f"{tag} run is not a {split} run of the base's mode")
    f26 = {t: pd.read_parquet(pathlib.Path(d) / "predictions_2026.parquet") for t, d in dirs}
    p26 = derive(f26["base"], f26["cat"], f26.get("yhead"), blend, keys=KEY_COLS_2026)
    pred = None
    if split == "holdout":
        fr = {t: pd.read_parquet(pathlib.Path(d) / "predictions.parquet") for t, d in dirs}
        pred = derive(fr["base"], fr["cat"], fr.get("yhead"), blend)
    out.mkdir(parents=True)
    if pred is not None:
        pred.to_parquet(out / "predictions.parquet", index=False)
    p26.to_parquet(out / "predictions_2026.parquet", index=False)
    fname = "predictions.parquet" if split == "holdout" else "predictions_2026.parquet"
    rep = {"split": split, "smoke": bool(reps["base"]["smoke"]), "assembled": True,
           "prereg": "plans/PREREG_r13_tuning_2026_09_27.md", "blend_weight": BLEND_W,
           "inputs": {t: str(d) for t, d in dirs},
           "input_sha256": {t: hashlib.sha256((pathlib.Path(d) / fname).read_bytes()).hexdigest() for t, d in dirs},
           "fits": {t: reps[t].get("fits", {}) for t in reps},
           "arms": ["r12", "cat", "lgbcat"] + (["blendy"] if yhead_dir is not None else []),
           "n_test": int(len(pred)) if pred is not None else 0, "n_rank": int(len(p26)),
           "finished_utc": (now or dt.datetime.now(dt.timezone.utc)).isoformat()}
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


# ---------------------------------------------------------------- the screen comparison

def paired_ci(y, cand, comp, blocks, level: float, n_boot: int, seed: int, board_n: float) -> list:
    """r12_validate.paired_gain's day-block bootstrap at another confidence level (same draws for the same seed)."""
    y, cand, comp = (np.asarray(a, float) for a in (y, cand, comp))
    m = np.isfinite(comp)
    g = (comp[m] - y[m]) ** 2 - (cand[m] - y[m]) ** 2
    per = pd.Series(g).groupby(np.asarray(blocks)[m]).sum().to_numpy()
    rng = np.random.default_rng(seed)
    draws = per[rng.integers(0, len(per), size=(n_boot, len(per)))].sum(1) / board_n
    a = (1.0 - level) / 2.0
    return [float(np.percentile(draws, 100 * a)), float(np.percentile(draws, 100 * (1 - a)))]


def verdict(dg5: float, ci95) -> str:
    """PREREG R13 H-arm: WORKING iff dG5 >= +100 bmse and the 95 % CI lower bound > 0."""
    return "WORKING" if (np.isfinite(dg5) and dg5 >= MIN_DG5 and ci95[0] > 0) else "NOT WORKING"


def exposure(p25: pd.DataFrame, p26: pd.DataFrame, arm: str, ref: str = "r12") -> dict:
    """Label-free tripwire per airport: RMS(F_arm - F_ref) on the 2026 ranking rows / the same on the 2025 test rows."""
    out = {}
    for ap in sorted(set(p25.ap.astype(str)) | set(p26.ap.astype(str))):
        a = p25[p25.ap.astype(str) == ap]
        b = p26[p26.ap.astype(str) == ap]
        r25 = float(np.sqrt(((a[f"pred_{arm}"] - a[f"pred_{ref}"]) ** 2).mean())) if len(a) else None
        r26 = float(np.sqrt(((b[f"pred_{arm}"] - b[f"pred_{ref}"]) ** 2).mean())) if len(b) else None
        ratio = r26 / r25 if (r25 and r26 is not None) else None
        out[ap] = {"rms_2025": r25, "rms_2026": r26, "ratio": ratio, "flag": bool(ratio is not None and ratio > 2.0)}
    return out


def compare_arm(base_oof: pd.DataFrame, arm_oof: pd.DataFrame, V, R, n_boot: int) -> dict:
    """V = v38_compare (align / per-airport tables / paired), R = r12_validate; adds the verdict and Bonferroni CI."""
    r = V.compare(base_oof, arm_oof, n_boot)
    m = V.airport_mask(base_oof, AIRPORTS_5)
    lvl = 1.0 - 0.05 / N_ARMS
    p5 = r["stacked_v34lanes"]["paired_new_vs_old"]["five"]
    r["h_arm"] = {"dG5": p5["gain_bmse"], "ci95": p5["ci95"],
                  "ci_bonferroni": paired_ci(base_oof.y.to_numpy()[m], arm_oof.stacked_v34lanes.to_numpy()[m],
                                             base_oof.stacked_v34lanes.to_numpy()[m], base_oof.block.to_numpy()[m],
                                             lvl, n_boot, R.BOOT_SEED, R.BOARD_N),
                  "bonferroni_level": lvl, "threshold": MIN_DG5, "verdict": verdict(p5["gain_bmse"], p5["ci95"]),
                  "dG_all": r["stacked_v34lanes"]["paired_new_vs_old"]["all"]["gain_bmse"],
                  "dG_all_ci95": r["stacked_v34lanes"]["paired_new_vs_old"]["all"]["ci95"]}
    del r["h1"]
    return r


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv=None) -> int:  # pragma: no cover - file I/O around assemble / compare_arm (both unit-tested)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("assemble")
    for f in ("--base-val", "--cat-val", "--out"):
        a1.add_argument(f, required=True)
    a1.add_argument("--yhead-val", default=None, help="omit for lgbcat only (AMEND-A1)")
    a1.add_argument("--split", default="holdout", choices=("holdout", "refit"))
    a2 = sub.add_parser("compare")
    for f in ("--base-oof", "--stack", "--arms-dir", "--arms", "--out"):
        a2.add_argument(f, required=True)
    a = ap.parse_args(argv)
    R = _load("r12_validate", "scripts/analysis/r12_validate.py")
    if a.cmd == "assemble":
        print(json.dumps(assemble(a.base_val, a.cat_val, a.yhead_val, a.out, R.LF.blend_taxi_times,
                                           split=a.split), indent=1)[:2000])
        return 0
    V = _load("v38_compare", "scripts/analysis/v38_compare.py")
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    base = pd.read_parquet(a.base_oof)
    p25 = pd.read_parquet(pathlib.Path(a.arms_dir) / "predictions.parquet")
    p26 = pd.read_parquet(pathlib.Path(a.arms_dir) / "predictions_2026.parquet")
    res = {"base_oof": a.base_oof, "stack": a.stack, "prereg": "plans/PREREG_r13_tuning_2026_09_27.md", "arms": {}}
    for arm in [x for x in a.arms.split(",") if x]:
        oof = pd.read_parquet(pathlib.Path(a.stack) / f"oof_{arm}.parquet")
        rec = compare_arm(base, oof, V, R, R.N_BOOT)
        if arm != "r12":
            rec["exposure_2026"] = exposure(p25, p26, arm)
        res["arms"][arm] = rec
        print(arm, json.dumps(rec["h_arm"]))
    out.write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
