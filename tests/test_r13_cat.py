"""R13 (plans/PREREG_r13_tuning_2026_09_27.md): the CatBoost fitter contract, the worker at PROCESS level in the
CatBoost venv (BC-5: in-process fakes never exercise the real import graph), the derived arms and the screen verdict.

Mutation rehearsal 2026-09-27: every mutant (14 + 4 AMEND-A1 assemble + 2 r12_move in r12_validate) of r13_cat /
r13_cat_worker / the r12_stack + r12_validate arm edits killed (N_ARMS escaped first; the Bonferroni-level assert was added for it)."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("r13_cat", "scripts/analysis/r13_cat.py")
R = _load("r12_validate", "scripts/analysis/r12_validate.py")
V = _load("v38_compare", "scripts/analysis/v38_compare.py")


# ---------------------------------------------------------------- configuration

def test_cat_config_is_the_registered_one():
    """PREREG R13 arm 1: depth 8, l2 3, 254 borders, Plain, Bernoulli 0.8, 8 threads; lr 0.05 screen / 0.02 full;
    cap 30,000, patience 200. Fails when any registered value drifts."""
    s = C.cat_config("screen")
    assert s["params"] == dict(loss_function="RMSE", depth=8, l2_leaf_reg=3.0, border_count=254, boosting_type="Plain",
                               bootstrap_type="Bernoulli", subsample=0.8, thread_count=8, learning_rate=0.05)
    assert (s["cap"], s["patience"]) == (30_000, 200)
    assert C.cat_config("full")["params"]["learning_rate"] == 0.02
    assert C.cat_config("smoke")["cap"] == 60
    with pytest.raises(ValueError, match="unknown tier"):
        C.cat_config("fast")


def test_n_refit_equals_lgbm_submit():
    """The restated refit rule must be lgbm_submit's, value for value (incl. the floor of 50)."""
    for b, a, f in ((6534, 2_000_000, 1_500_000), (10, 100, 90), (1, 5, 5), (12_345, 1_234_567, 1_000_001)):
        assert C.n_refit(b, a, f) == R.LF.L.n_refit(b, a, f)


# ---------------------------------------------------------------- the fitter, with a fake worker

def _fold(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    dlt = 100 * X[:, 0] + rng.normal(scale=5, size=n)
    proxy = 600 + rng.uniform(0, 50, n)
    y = proxy - dlt
    month = np.repeat([1, 2, 3, 4], n // 4)
    te = month == 1
    es = month == 3
    tr = ~te
    return dict(X=X, y=y, dlt=dlt, proxy=proxy, te=te, tr=tr, fit=tr & ~es, es=es)


class FakeWorker:
    """Plays the worker: es -> best_iter 40 and an exact (or broken) stopping prediction; refit -> seed-tagged preds."""

    def __init__(self, fold, broken=False, fail=None, bad_shape=False):
        self.fold, self.broken, self.fail, self.bad_shape, self.calls = fold, broken, fail, bad_shape, []

    def __call__(self, mode, workdir):
        self.calls.append(mode)
        z = dict(np.load(pathlib.Path(workdir) / "in.npz"))
        if self.fail == mode:
            np.savez(pathlib.Path(workdir) / f"{mode}_out.npz", error=np.array("boom"))
            return 3
        if mode == "es":
            pe = np.zeros(z["es"].sum()) if self.broken else z["label"][z["es"]]
            np.savez(pathlib.Path(workdir) / "es_out.npz", best_iter=np.array([40]), pred_es=pe)
        else:
            n = int(z["pred"].sum()) + (1 if self.bad_shape else 0)
            preds = np.vstack([np.full(n, 10.0 * (s + 1)) for s in z["seeds"]])
            self.n_ref = int(z["n_ref"])
            np.savez(pathlib.Path(workdir) / "refit_out.npz", preds=preds, seeds=z["seeds"])
        return 0


def test_cat_fitter_returns_the_fit_arm_contract(tmp_path):
    """delta = mean over seeds on the prediction rows, single per seed, best_iter from the stopping run, n_ref by the
    refit rule on tr / fit counts. Fails when the pooled mean, the seed map or n_ref are computed wrongly."""
    f = _fold()
    w = FakeWorker(f)
    fit = C.CatFitter(tmp_path / "work", C.cat_config("smoke"), runner=w, log=lambda m: None)
    res = fit(f, f["X"], None, None, None, None, (0, 1, 2), False, "es", "r12:cat", target="delta")
    assert w.calls == ["es", "refit"]
    assert res["best_iter"] == 40 and res["n_ref"] == C.n_refit(40, int(f["tr"].sum()), int(f["fit"].sum()))
    assert w.n_ref == res["n_ref"]
    np.testing.assert_allclose(res["delta"], 20.0)
    assert sorted(res["single"]) == [0, 1, 2] and np.allclose(res["single"][2], 30.0)
    assert len(res["delta"]) == f["te"].sum()
    assert res["cap_bound"] is False


def test_cat_fitter_writes_the_worker_inputs(tmp_path):
    """X.npy float32 of every row; in.npz carries the delta label, the four masks, seeds, ES seed 0 and the params."""
    f = _fold()
    fit = C.CatFitter(tmp_path / "w", C.cat_config("screen"), runner=FakeWorker(f), log=lambda m: None)
    fit(f, f["X"], None, None, None, None, (0,), False, "es", "r12:cat")
    X = np.load(tmp_path / "w" / "X.npy")
    assert X.dtype == np.float32 and X.shape == f["X"].shape
    z = np.load(tmp_path / "w" / "in.npz")
    np.testing.assert_array_equal(z["label"], f["dlt"])
    np.testing.assert_array_equal(z["pred"], f["te"])
    np.testing.assert_array_equal(z["fit"], f["fit"])
    assert int(z["es_seed"]) == 0 and json.loads(str(z["params"]))["learning_rate"] == 0.05
    assert int(z["cap"]) == 30_000 and int(z["patience"]) == 200


def test_cat_fitter_stopping_set_guard_refuses_a_broken_feature_path(tmp_path):
    """lgbm_submit.early_stop's guard: a stopping RMSE above 0.85 x proxy-only refuses the refit."""
    f = _fold()
    w = FakeWorker(f, broken=True)
    fit = C.CatFitter(tmp_path / "w", C.cat_config("smoke"), runner=w, log=lambda m: None)
    with pytest.raises(RuntimeError, match="feature path is broken"):
        fit(f, f["X"], None, None, None, None, (0,), False, "es", "r12:cat")
    assert w.calls == ["es"]


@pytest.mark.parametrize("mode", ["es", "refit"])
def test_cat_fitter_surfaces_worker_errors(tmp_path, mode):
    f = _fold()
    fit = C.CatFitter(tmp_path / "w", C.cat_config("smoke"), runner=FakeWorker(f, fail=mode), log=lambda m: None)
    with pytest.raises(RuntimeError, match=f"worker \\({mode}\\) failed: boom"):
        fit(f, f["X"], None, None, None, None, (0,), False, "es", "r12:cat")


def test_cat_fitter_refuses_wrong_shapes_targets_and_masks(tmp_path):
    f = _fold()
    fit = C.CatFitter(tmp_path / "w", C.cat_config("smoke"), runner=FakeWorker(f, bad_shape=True), log=lambda m: None)
    with pytest.raises(RuntimeError, match="shape"):
        fit(f, f["X"], None, None, None, None, (0,), False, "es", "r12:cat")
    with pytest.raises(ValueError, match="delta target only"):
        fit(f, f["X"], None, None, None, None, (0,), False, "es", "r12:cat", target="y")
    with pytest.raises(ValueError, match="per-airport"):
        fit(f, f["X"], None, None, None, None, (0,), True, "es", "r12:cat")
    g = dict(f, te=f["te"] | f["es"])
    with pytest.raises(ValueError, match="masks must satisfy"):
        fit(g, g["X"], None, None, None, None, (0,), False, "es", "r12:cat")


def test_cat_fitter_refuses_an_argument_it_does_not_know(tmp_path):
    """fit_arm's extra keywords (on_seed, weight, es_eval) change a LightGBM fit; CatFitter cannot honour them, so it must
    refuse them by name instead of fitting as if they were not given. Mutation: accept **kwargs again -> red."""
    f = _fold()
    fit = C.CatFitter(tmp_path / "w", C.cat_config("smoke"), runner=FakeWorker(f), log=lambda m: None)
    with pytest.raises(TypeError, match="weight"):
        fit(f, f["X"], None, None, None, None, (0,), False, "es", "r12:cat", weight=np.ones(len(f["y"])))


def test_read_out_without_a_file_or_with_a_bad_status(tmp_path):
    with pytest.raises(RuntimeError, match="wrote no result"):
        C.read_out(tmp_path, "es", 1)
    np.savez(tmp_path / "es_out.npz", best_iter=np.array([3]))
    with pytest.raises(RuntimeError, match="exited 9"):
        C.read_out(tmp_path, "es", 9)
    assert int(C.read_out(tmp_path, "es", 0)["best_iter"][0]) == 3


def test_write_work_refuses_bad_masks_and_labels(tmp_path):
    f = _fold()
    m = {"fit": f["fit"], "es": f["es"], "tr": f["tr"], "pred": f["te"]}
    with pytest.raises(ValueError, match="selects no row"):
        C.write_work(tmp_path, f["X"], f["dlt"], dict(m, es=np.zeros(len(f["X"]), bool)), (0,), C.cat_config("smoke"))
    with pytest.raises(ValueError, match="shape"):
        C.write_work(tmp_path, f["X"], f["dlt"], dict(m, fit=f["fit"][:-1]), (0,), C.cat_config("smoke"))
    lab = f["dlt"].copy()
    lab[np.flatnonzero(f["fit"])[0]] = np.nan
    with pytest.raises(ValueError, match="non-finite label on fit"):
        C.write_work(tmp_path, f["X"], lab, m, (0,), C.cat_config("smoke"))


def test_dispatch_routes_only_the_cat_arm():
    calls = []
    fit = C.dispatch(lambda *a, **k: calls.append("lgb"), lambda *a, **k: calls.append("cat"))
    args = [None] * 9
    fit(*args, "r12:cat", target="delta")
    fit(*args, "r12:yhead", target="y")
    fit(*args, name="r12:cat")
    assert calls == ["cat", "lgb", "cat"]


# ---------------------------------------------------------------- the real worker, at process level

@pytest.mark.skipif(not C.CAT_PY.exists(), reason="the CatBoost venv is not installed")
def test_worker_process_level_es_then_refit(tmp_path):
    """The real worker in the CatBoost venv: early stopping finds a best_iter <= cap, the stopping prediction beats the
    mean, and the refit returns one finite, seed-dependent prediction row per seed. ~3 s."""
    f = _fold(n=800)
    cfg = C.cat_config("smoke")
    cfg["params"]["thread_count"] = 2
    fitter = C.CatFitter(tmp_path / "w", cfg, runner=C._run_worker, log=lambda m: None)
    res = fitter(f, f["X"], None, None, None, None, (0, 1), False, "es", "r12:cat")
    assert 1 <= res["best_iter"] <= 60
    assert np.isfinite(res["delta"]).all() and len(res["delta"]) == f["te"].sum()
    assert not np.array_equal(res["single"][0], res["single"][1])            # bagging makes the seed live
    r = np.corrcoef(res["delta"], f["dlt"][f["te"]])[0, 1]
    assert r > 0.9, r
    z = np.load(tmp_path / "w" / "es_out.npz")
    assert int(z["best_iter"][0]) == int(z["tree_count"][0])      # use_best_model shrinks to the 1-based best count


@pytest.mark.skipif(not C.CAT_PY.exists(), reason="the CatBoost venv is not installed")
def test_worker_process_level_reports_an_error_and_exit_3(tmp_path):
    (tmp_path / "X.npy").write_bytes(b"not an array")
    np.savez(tmp_path / "in.npz", label=np.zeros(3))
    rc = subprocess.run([str(C.CAT_PY), "-B", str(C.WORKER), "es", str(tmp_path)], capture_output=True).returncode
    assert rc == 3
    with pytest.raises(RuntimeError, match="worker \\(es\\) failed"):
        C.read_out(tmp_path, "es", rc)
    assert subprocess.run([str(C.CAT_PY), "-B", str(C.WORKER), "bogus", str(tmp_path)],
                          capture_output=True).returncode == 2


# ---------------------------------------------------------------- derived arms and assembly

def _val(tag, n=6):
    d = pd.DataFrame({"month": [1, 1, 1, 7, 7, 7][:n], "pos": np.arange(n), "MVT_ID_mvt": np.arange(n) + 100.0,
                      "ap": ["EDDF", "EDDM", "LFPG", "EDDF", "LIRF", "LEBL"][:n],
                      "y": np.linspace(300, 800, n), "proxy": np.linspace(500, 900, n)})
    if tag == "base":
        d["pred_r12"] = 100.0
        d["taxi_r12"] = np.maximum(d.proxy - 100.0, 1)
    elif tag == "cat":
        d["pred_cat"] = 200.0
        d["taxi_cat"] = np.maximum(d.proxy - 200.0, 1)
    else:
        d["pred_yhead"] = d.y + 10
        d["taxi_yhead"] = d.y + 10
    return d


def test_derive_lgbcat_is_the_half_delta_blend_and_blendy_is_candidate_b():
    """lgbcat F = 0.5 F_r12 + 0.5 F_cat; blendy F = proxy - blend_taxi_times(taxi_yhead, taxi_r12, 0.5)."""
    out = C.derive(_val("base"), _val("cat"), _val("yhead"), R.LF.blend_taxi_times)
    np.testing.assert_allclose(out.pred_lgbcat, 150.0)
    np.testing.assert_allclose(out.taxi_lgbcat, out.proxy - 150.0)
    tb = 0.5 * (out.y + 10) + 0.5 * (out.proxy - 100.0)
    np.testing.assert_allclose(out.taxi_blendy, tb)
    np.testing.assert_allclose(out.pred_blendy, out.proxy - tb)
    np.testing.assert_allclose(out.pred_cat, 200.0)


def test_derive_refuses_misaligned_rows_and_nan():
    c = _val("cat")
    c.loc[2, "y"] = 1.0
    with pytest.raises(ValueError, match="column y differs"):
        C.derive(_val("base"), c, _val("yhead"), R.LF.blend_taxi_times)
    with pytest.raises(ValueError, match="rows"):
        C.derive(_val("base"), _val("cat", 5), _val("yhead"), R.LF.blend_taxi_times)
    c = _val("cat")
    c.loc[0, "pred_cat"] = np.nan
    with pytest.raises(ValueError, match="non-finite F"):
        C.derive(_val("base"), c, _val("yhead"), R.LF.blend_taxi_times)


def _write_val(d, tag, split="holdout", smoke=False):
    d.mkdir(parents=True)
    _val(tag).to_parquet(d / "predictions.parquet", index=False)
    _val(tag).drop(columns=["y", "month", "pos"]).to_parquet(d / "predictions_2026.parquet", index=False)
    (d / "report.json").write_text(json.dumps({"split": split, "smoke": smoke, "fits": {tag: {"best_iter": 1}}}))


def test_assemble_writes_a_holdout_val_dir_the_stack_reads(tmp_path):
    for t in ("base", "cat", "yhead"):
        _write_val(tmp_path / t, t)
    rep = C.assemble(tmp_path / "base", tmp_path / "cat", tmp_path / "yhead", tmp_path / "out", R.LF.blend_taxi_times)
    r = json.loads((tmp_path / "out" / "report.json").read_text())
    assert r["split"] == "holdout" and r["smoke"] is False and rep["arms"] == ["r12", "cat", "lgbcat", "blendy"]
    p = pd.read_parquet(tmp_path / "out" / "predictions.parquet")
    assert {"pred_r12", "pred_cat", "pred_lgbcat", "pred_blendy"} <= set(p.columns)
    p26 = pd.read_parquet(tmp_path / "out" / "predictions_2026.parquet")
    np.testing.assert_allclose(p26.pred_lgbcat, 150.0)
    with pytest.raises(FileExistsError):
        C.assemble(tmp_path / "base", tmp_path / "cat", tmp_path / "yhead", tmp_path / "out", R.LF.blend_taxi_times)


def test_assemble_refuses_a_non_holdout_or_mixed_mode_input(tmp_path):
    _write_val(tmp_path / "base", "base")
    _write_val(tmp_path / "cat", "cat", split="refit")
    _write_val(tmp_path / "yhead", "yhead", smoke=True)
    with pytest.raises(ValueError, match="cat run is not a holdout run"):
        C.assemble(tmp_path / "base", tmp_path / "cat", tmp_path / "yhead", tmp_path / "o", R.LF.blend_taxi_times)


# ---------------------------------------------------------------- verdict, CI, exposure

def test_verdict_threshold_and_ci():
    assert C.verdict(100.0, [0.1, 200]) == "WORKING"
    assert C.verdict(99.9, [5, 200]) == "NOT WORKING"
    assert C.verdict(150.0, [0.0, 300]) == "NOT WORKING"
    assert C.verdict(float("nan"), [1, 2]) == "NOT WORKING"


def test_paired_ci_at_95_equals_paired_gain():
    rng = np.random.default_rng(3)
    y = rng.normal(500, 100, 600)
    comp, cand = y + rng.normal(0, 50, 600), y + rng.normal(0, 40, 600)
    blocks = np.repeat(np.arange(60), 10)
    g = R.paired_gain(y, cand, comp, blocks)
    assert C.paired_ci(y, cand, comp, blocks, 0.95, R.N_BOOT, R.BOOT_SEED, R.BOARD_N) == pytest.approx(g["ci95"])
    wide = C.paired_ci(y, cand, comp, blocks, 1 - 0.05 / 3, R.N_BOOT, R.BOOT_SEED, R.BOARD_N)
    assert wide[0] < g["ci95"][0] and wide[1] > g["ci95"][1]


def _oof(n=200, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(600, 150, n)
    d = pd.DataFrame({"row": np.arange(n), "y": y, "block": np.repeat(np.arange(20), n // 20),
                      "ap": np.tile(["EDDF", "EDDM", "EHAM", "LEBL", "LSZH", "LIRF", "LFPG", "EGLL", "LEMD", "LTFM"],
                                    n // 10)})
    d["PIPE_M"] = y + rng.normal(0, 200, n)
    d["stacked_v34lanes"] = y + rng.normal(0, 150, n) * (1 - shift)
    d["single"] = d.stacked_v34lanes
    return d


def test_compare_arm_reads_the_five_airport_paired_gain():
    """dG5 is the paired new-vs-base gain on the 5 airports only; identical frames give 0 and NOT WORKING."""
    base = _oof()
    same = C.compare_arm(base, base.copy(), V, R, 200)
    assert same["h_arm"]["dG5"] == 0.0 and same["h_arm"]["verdict"] == "NOT WORKING"
    better = base.copy()
    better["stacked_v34lanes"] = better.y + 1.0
    r = C.compare_arm(base, better, V, R, 200)
    m = base.ap.isin(C.AIRPORTS_5)
    exp = (((base.stacked_v34lanes - base.y) ** 2)[m].sum() - m.sum() * 1.0) / R.BOARD_N
    assert r["h_arm"]["dG5"] == pytest.approx(exp)
    assert r["h_arm"]["bonferroni_level"] == pytest.approx(1 - 0.05 / 3)       # three registered arms
    assert r["h_arm"]["ci_bonferroni"][0] < r["h_arm"]["ci95"][0]
    assert "h1" not in r


def test_exposure_ratio_and_flag():
    p25 = pd.DataFrame({"ap": ["A", "A", "B"], "pred_r12": [0.0, 0.0, 0.0], "pred_cat": [1.0, -1.0, 2.0]})
    p26 = pd.DataFrame({"ap": ["A", "B"], "pred_r12": [0.0, 0.0], "pred_cat": [3.0, 2.0]})
    e = C.exposure(p25, p26, "cat")
    assert e["A"]["ratio"] == pytest.approx(3.0) and e["A"]["flag"] is True
    assert e["B"]["ratio"] == pytest.approx(1.0) and e["B"]["flag"] is False


# ---------------------------------------------------------------- AMEND-A1: the full tier (no yhead) and the refit split

def test_derive_without_yhead_builds_lgbcat_only():
    out = C.derive(_val("base"), _val("cat"), None, R.LF.blend_taxi_times)
    np.testing.assert_allclose(out.pred_lgbcat, 150.0)
    assert "pred_blendy" not in out.columns


def _write_refit(d, tag):
    d.mkdir(parents=True)
    _val(tag).drop(columns=["y", "pos"]).to_parquet(d / "predictions_2026.parquet", index=False)
    (d / "report.json").write_text(json.dumps({"split": "refit", "smoke": False, "fits": {tag: {"best_iter": 1}}}))


def test_assemble_refit_split_writes_2026_only_with_split_refit(tmp_path):
    """A1: the 12-month F26 = 0.5 F_r12(refit) + 0.5 F_cat(refit) in a dir r12_stack serve accepts (split refit)."""
    for t in ("base", "cat"):
        _write_refit(tmp_path / t, t)
    rep = C.assemble(tmp_path / "base", tmp_path / "cat", None, tmp_path / "out", R.LF.blend_taxi_times,
                     split="refit")
    r = json.loads((tmp_path / "out" / "report.json").read_text())
    assert r["split"] == "refit" and rep["arms"] == ["r12", "cat", "lgbcat"] and rep["n_test"] == 0
    assert not (tmp_path / "out" / "predictions.parquet").exists()
    p26 = pd.read_parquet(tmp_path / "out" / "predictions_2026.parquet")
    np.testing.assert_allclose(p26.pred_lgbcat, 150.0)


def test_assemble_refuses_a_split_mismatch(tmp_path):
    _write_refit(tmp_path / "base", "base")
    _write_val(tmp_path / "cat", "cat")
    with pytest.raises(ValueError, match="cat run is not a refit run"):
        C.assemble(tmp_path / "base", tmp_path / "cat", None, tmp_path / "o", R.LF.blend_taxi_times, split="refit")
    with pytest.raises(ValueError, match="split must be"):
        C.assemble(tmp_path / "base", tmp_path / "cat", None, tmp_path / "o", R.LF.blend_taxi_times, split="forward")
