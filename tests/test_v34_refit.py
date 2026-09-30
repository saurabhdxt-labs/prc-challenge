"""scripts/analysis/v34_refit.py — v34 steps 3, 4 and 5 (AMENDMENTS V34.3-V34.5): the 12-month production refits of F (80
columns), FW (90) and ORL (96) without the three label-seconds columns, through orl_stage3.production_refit (lgbm_submit's
split_masks -> in-fold encodings -> design_matrix -> early_stop -> fit_seeds -> mean_delta -> recover_taxi_time), boosters and
2026 predictions written to NEW paths only, and a label-free move tripwire against the shipped counterpart.

Fakes and synthetic frames only (no real data, no lock). Mutation rehearsal 2026-09-24: see docstrings."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("v34_refit", "scripts/analysis/v34_refit.py")
LF = sys.modules["nsc_fold"].LF
SEC = ("aobt_sec", "mvt_sec", "sched_sec")


# ---------------------------------------------------------------- the three designs

def test_f_design_is_the_shipped_83_minus_the_three():
    d = R.design("F")
    assert d.step == 3 and d.feats == [c for c in LF.FEATS_QUEUE_ORDER if c not in SEC] and len(d.feats) == 80
    assert (d.queue, d.order, d.weather, d.orl) == (True, True, False, False)


def test_fw_design_is_the_shipped_93_minus_the_three():
    d = R.design("FW")
    assert d.step == 4 and d.feats == [c for c in LF.FEATS_QUEUE_ORDER_WEATHER if c not in SEC] and len(d.feats) == 90
    assert (d.queue, d.order, d.weather, d.orl) == (True, True, True, False)


def test_orl_design_is_orl_stage2s_99_minus_the_three():
    """Rehearsed RED: building ORL as FW-NSC + the six ORL columns in a different order."""
    S2 = sys.modules["orl_stage2"]
    _, arm = S2.stage2_feats(sys.modules["lgbm_submit"].S)
    d = R.design("ORL")
    assert d.step == 5 and d.feats == [c for c in arm if c not in SEC] and len(d.feats) == 96
    assert d.feats[:90] == R.design("FW").feats
    assert (d.queue, d.order, d.weather, d.orl) == (True, True, True, True)


def test_no_design_carries_a_seconds_column_and_unknown_designs_are_refused():
    for name in ("F", "FW", "ORL"):
        assert not any(c.endswith("_sec") for c in R.design(name).feats)
    with pytest.raises(ValueError, match="unknown design"):
        R.design("FWX")


def test_booster_files_are_new_paths_under_the_step_dir_never_the_shipped_names(tmp_path):
    files = R.booster_files(R.design("F"), tmp_path, allow_roots=(tmp_path,))
    assert [p.name for p, _ in files] == [f"lgbm_nsc_f80_seed{s}.txt" for s in (0, 1, 2)]
    assert [j.name for _, j in files] == [f"lgbm_nsc_f80_seed{s}.fit.json" for s in (0, 1, 2)]
    with pytest.raises(PermissionError):
        R.booster_files(R.design("F"), ROOT / "data" / "cache_stand")


def test_fold_move_bar_is_three_times_the_fw_nsc_fold_rms_move():
    """The FW / ORL tripwire bar: 3 x the RMS taxi move FW-NSC made vs the FW record on fold A (data/nsc_run)."""
    p = pd.read_parquet(ROOT / "data/nsc_run/nsc_fold_preds.parquet", columns=["proxy", "incumbent", "nsc"])
    d = np.maximum(p.proxy - p.nsc, 1) - np.maximum(p.proxy - p.incumbent, 1)
    rms = float(np.sqrt(np.mean(d ** 2)))
    assert R.FW_FOLD_RMS_MOVE_S == pytest.approx(rms, abs=1e-6) and 14.0 < rms < 15.0
    assert R.MOVE_BAR_FACTOR == 3.0


def test_move_tripwire_uses_common_ids_and_the_bar():
    """Rehearsed RED: comparing positionally instead of by id; '<' vs '<=' is not distinguishable; dropping the bar."""
    ids = np.array([1.0, 2.0, 3.0, 4.0])
    new = pd.Series([100, 200, 300, 400], index=ids)
    old = pd.Series([410, 110, 190, 999], index=[4.0, 1.0, 2.0, 7.0])
    t = R.move_tripwire(new, old, bar_s=12.0)
    assert t["n_common"] == 3 and t["rms_move_s"] == pytest.approx(np.sqrt((100 + 100 + 100) / 3))
    assert t["pass"] is True
    assert R.move_tripwire(new, old, bar_s=9.0)["pass"] is False


# ---------------------------------------------------------------- orchestration with a fake refit (BC-24)

def _frame(n_train=300, n_rank=40, feats=None):
    rng = np.random.default_rng(0)
    n = n_train + n_rank
    is_rank = np.r_[np.zeros(n_train, bool), np.ones(n_rank, bool)]
    # real-shaped: lgbm_submit.load_frames carries the ENC_KEYS, not the te_/de_ encodings (production_refit adds those)
    d = pd.DataFrame({c: rng.normal(size=n) for c in feats if not c.startswith(("te_", "de_"))})
    for k in sys.modules["lgbm_submit"].S.ENC_KEYS:
        d[k] = rng.choice(["a", "b"], n)
    for c in SEC:
        d[c] = rng.integers(0, 60, n).astype(float)          # the frame carries them; the design must not
    d["month"] = rng.integers(1, 13, n)
    d.loc[is_rank, "month"] = np.r_[np.ones(n_rank // 2, int), np.full(n_rank - n_rank // 2, 7)]
    d["ap"] = np.where(np.arange(n) % 2 == 0, "EGLL", "LFPG")
    d["proxy"] = rng.uniform(400, 900, n)
    d["y"] = np.where(is_rank, np.nan, 600.0)
    d["delta"] = np.where(is_rank, np.nan, d.proxy - 600.0)
    return d, is_rank, np.arange(1, n_rank + 1, dtype=float) * 10.0


CALLS = []


def _fake_refit(d, is_rank, feats, params, nest, patience, seeds=(0, 1, 2), files=None, smoke=False, enc_keys=None):
    CALLS.append(dict(feats=list(feats), seeds=tuple(seeds), nest=nest, lr=params["learning_rate"]))
    for p, j in files:
        p.write_text("tree")
        j.write_text(json.dumps({"n_features": len(feats), "best_iter": 5, "n_ref": 6}))
    n = int(np.asarray(is_rank).sum())
    by = {s: np.full(n, 10.0 + s) for s in seeds}
    mean = np.mean([by[s] for s in seeds], axis=0)
    proxy = d.proxy.to_numpy(float)[np.asarray(is_rank, bool)]
    return {"delta_by_seed": by, "delta_mean": mean, "taxi_int": np.rint(np.maximum(proxy - mean, 1)).astype("int32"),
            "info": {"best_iter": 5, "n_ref": 6}, "floor_bind_rate": 0.0, "seed_sd_mean_s": 1.0,
            "n_train": int((~np.asarray(is_rank, bool)).sum()), "n_rank": n}


@pytest.mark.parametrize("name", ["F", "FW", "ORL"])
def test_run_fits_the_nsc_design_and_writes_predictions_boosters_and_report(tmp_path, name):
    """Every line after the compute (BC-24). Rehearsed RED: passing d.columns (with the seconds) as the design;
    taxi_int from seed 0 only; writing the booster into data/cache_stand."""
    CALLS.clear()
    des = R.design(name)
    d, is_rank, ids = _frame(feats=des.feats)
    old = pd.Series(np.rint(np.maximum(d.proxy.to_numpy()[is_rank] - 11.0, 1)) + 2, index=ids)
    rep = R.run(des, False, tmp_path, d, is_rank, ids, refit=_fake_refit, counterpart=old, bar_s=30.0,
                allow_roots=(tmp_path,))
    assert CALLS == [dict(feats=des.feats, seeds=(0, 1, 2), nest=LF.L.NEST, lr=LF.L.P["learning_rate"])]
    p = pd.read_parquet(tmp_path / R.PRED_NAME)
    assert list(p.columns) == R.PRED_COLS and len(p) == len(ids) and p.taxi_int.dtype == np.int32
    assert np.allclose(p.delta_mean, 11.0) and np.allclose(p.d_seed2, 12.0)
    assert np.array_equal(p.MVT_ID_mvt.to_numpy(), ids)
    assert rep["status"] == "PASS" and rep["n_features"] == len(des.feats) and rep["features"] == des.feats
    assert rep["tripwire"]["rms_move_s"] == pytest.approx(2.0) and rep["tripwire"]["pass"]
    assert rep["booster_n_features"] == [len(des.feats)] * 3
    j = json.loads((tmp_path / "report.json").read_text())
    assert j == rep and len(j["outputs"]) == 7            # preds + 3 boosters + 3 fit.json


def test_run_fails_on_a_move_above_the_bar_and_on_a_booster_width_mismatch(tmp_path):
    des = R.design("FW")
    d, is_rank, ids = _frame(feats=des.feats)
    old = pd.Series(np.full(len(ids), 5_000.0), index=ids)
    rep = R.run(des, False, tmp_path / "a", d, is_rank, ids, refit=_fake_refit, counterpart=old, bar_s=43.0,
                allow_roots=(tmp_path,))
    assert rep["status"] == "FAIL" and not rep["tripwire"]["pass"]

    def wide(*a, **k):
        out = _fake_refit(*a, **k)
        for p, j in k["files"]:
            j.write_text(json.dumps({"n_features": 93}))
        return out
    with pytest.raises(AssertionError, match="93 features"):
        R.run(des, False, tmp_path / "b", d, is_rank, ids, refit=wide, counterpart=None, bar_s=43.0,
              allow_roots=(tmp_path,))


def test_regression_smoke_20260924_encodings_are_made_by_the_refit_not_required_in_the_frame(tmp_path):
    """Found by the first step-3 smoke (2026-09-24 20:55): run() demanded the 24 te_/de_ columns before the refit, which is
    what creates them (stand_ab.infold_encodings inside production_refit). The frame fixture is now real-shaped."""
    des = R.design("F")
    d, is_rank, ids = _frame(feats=des.feats)
    assert not any(c.startswith(("te_", "de_")) for c in d.columns)
    rep = R.run(des, True, tmp_path, d, is_rank, ids, refit=_fake_refit, counterpart=None, bar_s=None,
                allow_roots=(tmp_path,))
    assert rep["status"] == "SMOKE"


def test_run_refuses_a_frame_missing_a_design_column_and_existing_outputs(tmp_path):
    des = R.design("F")
    d, is_rank, ids = _frame(feats=des.feats)
    with pytest.raises(KeyError, match="design columns absent"):
        R.run(des, False, tmp_path, d.drop(columns=[des.feats[5]]), is_rank, ids,
              refit=_fake_refit, counterpart=None, bar_s=1.0, allow_roots=(tmp_path,))
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / R.PRED_NAME).write_text("x")
    with pytest.raises(FileExistsError):
        R.run(des, False, tmp_path / "b", d, is_rank, ids, refit=_fake_refit, counterpart=None, bar_s=1.0,
              allow_roots=(tmp_path,))


def test_smoke_uses_the_smoke_trees_and_reports_smoke(tmp_path):
    CALLS.clear()
    des = R.design("F")
    d, is_rank, ids = _frame(feats=des.feats)
    rep = R.run(des, True, tmp_path, d, is_rank, ids, refit=_fake_refit, counterpart=None, bar_s=None,
                allow_roots=(tmp_path,))
    assert CALLS[0]["nest"] == LF.L.SMOKE["nest"] and CALLS[0]["lr"] == LF.L.SMOKE["learning_rate"]
    assert rep["status"] == "SMOKE" and rep["tripwire"] is None


def test_parse_args_requires_design_and_rss_limit():
    with pytest.raises(SystemExit):
        R.parse_args(["--design", "F"])
    with pytest.raises(SystemExit):
        R.parse_args(["--rss-limit-gib", "8"])
    a = R.parse_args(["--design", "ORL", "--rss-limit-gib", "6", "--smoke"])
    assert (a.design, a.rss_limit_gib, a.smoke) == ("ORL", 6.0, True)


def test_refit_config_is_lgbm_submits_production_or_smoke_values():
    p, nest, pat = R.refit_config(False)
    assert p == dict(LF.L.P) and nest == LF.L.NEST and pat == LF.L.PATIENCE
    p, nest, pat = R.refit_config(True)
    assert p["learning_rate"] == LF.L.SMOKE["learning_rate"] and nest == LF.L.SMOKE["nest"] and pat == LF.L.SMOKE["patience"]
    assert R.refit_config(False)[0]["learning_rate"] == 0.01          # the smoke never leaks into production params
