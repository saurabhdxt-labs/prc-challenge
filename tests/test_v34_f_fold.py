"""scripts/analysis/v34_f_fold.py — v34 step 1 (AMENDMENT V34.1): the arm-F fold-A recipe (lgbm_fold queue_order, baseline A2,
pa-trees es, seeds 0/1/2, pooled mean, no per-airport blend) on FEATS_NSC_F (80 columns), writing a NEW F fold record in the
schema the ADS-B stack reads, paired against the shipped F record under the NSC decision rule.

Fakes only (no real data, no lock). Mutation rehearsal 2026-09-24: see docstrings."""
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


F1 = _load("v34_f_fold", "scripts/analysis/v34_f_fold.py")
NSC = sys.modules["nsc_fold"]
LF = NSC.LF
SEC = ("aobt_sec", "mvt_sec", "sched_sec")


def test_design_is_the_shipped_f_83_minus_exactly_the_three_in_order():
    """Rehearsed RED: using FEATS_NSC_FW (90) or FEATS_QUEUE_ORDER (83) as the design."""
    assert F1.FEATS == [c for c in LF.FEATS_QUEUE_ORDER if c not in SEC] and len(F1.FEATS) == 80
    assert not any(c.endswith("_sec") for c in F1.FEATS)


def test_record_constants_are_the_shipped_f_record():
    assert F1.RECORD == ROOT / "data/cache_stand/fold_preds_queue_order.parquet"
    assert F1.RECORD_SHA == "8536bb872ebae4770ee705bcbcb7af25aa8fe083d19dd507bf9a8bcda3fb6c06"
    assert F1.RECORD_COLS[:7] == ["row", "month", "ap", "y", "proxy", "delta", "sp"]
    assert F1.RECORD_COLS[-1] == "treatment"


def test_fit_config_is_the_record_runs_a2_es_three_seeds():
    """The F record was written by `lgbm_fold.py --orderfeats --queue --baseline A2 --pa-trees es` (per_airport False)."""
    p, nest, pat, floor = F1.fit_config(False)
    assert p == dict(LF.L.P) and nest == LF.L.NEST and pat == LF.L.PATIENCE and floor == LF.L.PA_TREE_FLOOR
    ps, nests, pats, _ = F1.fit_config(True)
    assert ps["learning_rate"] == LF.SMOKE["learning_rate"] and nests == LF.SMOKE["nest"] and pats == LF.SMOKE["patience"]
    assert F1.SEEDS == (0, 1, 2)


# ---------------------------------------------------------------- orchestration with fakes (BC-24)

def _fake_fold(feats, n_te=8, n_tr=6):
    n = n_te + n_tr
    rng = np.random.default_rng(1)
    feats = list(feats)
    proxy = rng.uniform(300, 900, n)
    dlt = rng.normal(0, 40, n)
    X = rng.normal(size=(n, len(feats))).astype("float32")
    X[:, feats.index("doy")] = np.r_[np.repeat([5, 6, 190, 191], 2), np.arange(n_tr)]
    te = np.r_[np.ones(n_te, bool), np.zeros(n_tr, bool)]
    y = proxy - dlt
    base = pd.DataFrame({"row": np.flatnonzero(te).astype("int64"), "month": np.r_[1, 1, 1, 1, 7, 7, 7, 7].astype("int32"),
                         "ap": ["EGLL", "LIRF"] * (n_te // 2), "y": y[te], "proxy": proxy[te], "delta": dlt[te],
                         "sp": np.arange(n_te, dtype=float)})
    return dict(X=X, y=y, dlt=dlt, proxy=proxy, te=te, feats=feats, base=base,
                airports=["EGLL", "LIRF"], ap_code=np.r_[np.tile([0, 1], n_te // 2), np.zeros(n_tr, int)])


FIT_CALLS = []


def _fake_fit(fold, X, params, nest, patience, pa_floor, seeds, per_airport, pa_trees, name, **k):
    FIT_CALLS.append(dict(per_airport=per_airport, pa_trees=pa_trees, seeds=tuple(seeds), width=X.shape[1]))
    te = fold["te"]
    single = {s: fold["dlt"][te] + 5.0 + s for s in seeds}
    pooled = np.mean([single[s] for s in seeds], axis=0)
    return dict(delta=pooled, single=single, single_rmse={s: 1.0 + s for s in seeds}, best_iter=10, n_ref=12,
                info={"es_rmse_taxi_time_NOT_A_RESULT": 1.0}, n_features=X.shape[1])


def _record(fold):
    rec = fold["base"][["row", "month", "ap", "y", "proxy", "delta"]].copy()
    rec["treatment"] = rec.delta + 3.0
    return rec


def test_run_writes_the_new_record_in_the_adsb_stack_contract_and_the_report(tmp_path):
    """Every line after the compute on real-shaped output, ending with the runner's own writers (BC-24).
    Rehearsed RED: writing `treatment` as seed 0 only (not the pooled mean); omitting `sp`; per_airport=True."""
    FIT_CALLS.clear()
    fold = _fake_fold(F1.FEATS)
    rep = F1.run(False, tmp_path, lambda f, s: fold, _fake_fit, _record(fold), n_boot=30, allow_roots=(tmp_path,))
    assert FIT_CALLS == [dict(per_airport=False, pa_trees="es", seeds=(0, 1, 2), width=80)]
    rec = pd.read_parquet(tmp_path / F1.RECORD_NAME)
    assert list(rec.columns) == F1.RECORD_COLS
    assert np.allclose(rec.treatment, fold["base"].delta + 6.0) and np.allclose(rec.delta_hat_seed2, fold["base"].delta + 7.0)
    assert rec.sp.tolist() == fold["base"].sp.tolist() and rec.row.dtype == np.int64
    j = json.loads((tmp_path / "report.json").read_text())
    assert j == rep and j["status"] == "PASS" and j["n_features"] == 80 and j["features"] == F1.FEATS
    assert j["decision"]["decision"] in ("PROCEED", "ESCALATE")
    se_inc = (fold["base"].y - np.maximum(fold["base"].proxy - (fold["base"].delta + 3.0), 1)) ** 2
    se_new = (fold["base"].y - np.maximum(fold["base"].proxy - (fold["base"].delta + 6.0), 1)) ** 2
    assert j["pairing"]["cost_bmse"] == pytest.approx(float((se_new - se_inc).sum()) / 344_841)
    assert j["treatment_rmse"] == pytest.approx(float(np.sqrt(se_new.mean())))
    assert j["rms_move_s"] == pytest.approx(3.0)
    assert j["outputs"] == {F1.C.rel(tmp_path / F1.RECORD_NAME): F1.C.sha256(tmp_path / F1.RECORD_NAME)}


def test_run_status_fails_when_the_registered_rule_escalates(tmp_path):
    """A cost above the 337 bound is a defect signal: the report is written with status FAIL (step 2 then refuses)."""
    fold = _fake_fold(F1.FEATS)
    rec = _record(fold)
    rec["treatment"] = rec.delta                       # the record is exact

    def bad_fit(*a, **k):
        out = _fake_fit(*a, **k)
        out["delta"] = out["delta"] - 1.0e6                # taxi ~1e6 s: a cost far above the 337 bound
        return out
    rep = F1.run(False, tmp_path, lambda f, s: fold, bad_fit, rec, n_boot=20, allow_roots=(tmp_path,))
    assert rep["decision"]["decision"] == "ESCALATE" and rep["status"] == "FAIL"


def test_run_refuses_a_design_with_a_seconds_column_and_a_misaligned_record(tmp_path):
    fold = _fake_fold(LF.FEATS_QUEUE_ORDER)
    with pytest.raises(AssertionError, match="label-seconds"):
        F1.run(False, tmp_path, lambda f, s: fold, _fake_fit, None, n_boot=5, allow_roots=(tmp_path,))
    fold = _fake_fold(F1.FEATS)
    rec = _record(fold)
    rec.loc[0, "y"] += 1.0
    with pytest.raises(AssertionError, match="differ in column y"):
        F1.run(False, tmp_path / "b", lambda f, s: fold, _fake_fit, rec, n_boot=5, allow_roots=(tmp_path,))


def test_run_refuses_to_overwrite_and_to_write_outside_the_roots(tmp_path):
    fold = _fake_fold(F1.FEATS)
    (tmp_path / F1.RECORD_NAME).write_text("x")
    with pytest.raises(FileExistsError):
        F1.run(False, tmp_path, lambda f, s: fold, _fake_fit, _record(fold), n_boot=5, allow_roots=(tmp_path,))
    with pytest.raises(PermissionError):
        F1.run(False, ROOT / "data" / "cache_stand", lambda f, s: fold, _fake_fit, _record(fold), n_boot=5)


def test_smoke_pairs_against_an_in_process_83_column_refit(tmp_path):
    calls = []

    def load(feats, smoke):
        calls.append(len(feats))
        return _fake_fold(feats)
    rep = F1.run(True, tmp_path, load, _fake_fit, None, n_boot=10, allow_roots=(tmp_path,))
    assert calls == [80, 83] and rep["smoke"] is True and rep["status"] == "SMOKE"
    assert rep["incumbent_source"] == "in-process 83-column refit (smoke)"


def test_parse_args_requires_the_rss_limit():
    """The launcher passes the limit explicitly; a runner without one refuses to start. Rehearsed RED: a 6 GiB default."""
    with pytest.raises(SystemExit):
        F1.parse_args([])
    a = F1.parse_args(["--rss-limit-gib", "6", "--smoke"])
    assert a.rss_limit_gib == 6.0 and a.smoke is True
