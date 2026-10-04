"""scripts/analysis/v34_catboost.py — v34 step 6 (AMENDMENT V34.6): CatBoost C-NSC with the CBV ship recipe. The NSC input is
the shipped cbv_ship_input.parquet with the three seconds columns physically removed (every other column byte-for-byte the
same rows and values), the same venv worker (scripts/catboost_native_worker.py, BC-5: never imported here) with the recorded
CBV params, the worker's RSS polled and killed at the step's limit.

Synthetic parquet + a fake worker only (no real data, no CatBoost, no lock). Mutation rehearsal 2026-09-24: docstrings."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
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


CB = _load("v34_catboost", "scripts/analysis/v34_catboost.py")
SEC = ["aobt_sec", "mvt_sec", "sched_sec"]


@pytest.mark.devtree
def test_params_are_the_recorded_cbv_ship_params_and_the_worker_defaults():
    """CBV's worker record (data/cache_stand/cbv_ship_C_2026.json) and the worker's own defaults agree with PARAMS."""
    rec = json.loads((ROOT / "data/cache_stand/cbv_ship_C_2026.json").read_text())["params"]
    assert CB.PARAMS == rec == dict(depth=8, learning_rate=0.15, iterations=12_000, od_wait=200, seed=0, threads=6, ctr=2)
    src = (ROOT / "scripts/catboost_native_worker.py").read_text()
    assert "DEFAULTS = dict(depth=8, learning_rate=0.15, iterations=12_000, od_wait=200, seed=0, threads=6, ctr=2)" in src


def test_src_is_the_shipped_cbv_input_pinned():
    assert CB.SRC == ROOT / "data/cache_stand/cbv_ship_input.parquet"
    assert CB.SRC_SHA == "a210da02a6bbe4c0f91ae81d1b3512148f6d9f6c4b426dc18914e7a22b482916"


@pytest.mark.devtree
def test_nsc_cols_drop_exactly_the_three_numeric_seconds_and_equal_nsc_fold():
    cols = json.loads((ROOT / "data/cache_stand/cbv_ship_input.parquet.cols.json").read_text())
    got = CB.nsc_cols(cols)
    assert got["cat"] == cols["cat"] and got["num"] == [c for c in cols["num"] if c not in SEC] and len(got["num"]) == 56
    assert got["num"] == _load("nsc_fold", "scripts/analysis/nsc_fold.py").nsc_catboost_numeric()


def _src(tmp_path, n=50):
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"proxy": rng.normal(size=n).astype("float32"), "aobt_sec": rng.integers(0, 60, n).astype("float32"),
                      "mvt_sec": rng.integers(0, 60, n).astype("float32"), "sched_sec": np.zeros(n, "float32"),
                      "hr": rng.integers(0, 24, n).astype("float32"), "ap": np.where(np.arange(n) % 2, "EGLL", "LFPG"),
                      "delta": np.where(np.arange(n) < 40, rng.normal(size=n), np.nan),
                      "score_proxy": rng.uniform(400, 900, n), "month": np.r_[np.arange(40) % 12 + 1, np.ones(10, int)],
                      "MVT_ID_mvt": np.arange(n, dtype=float) + 7.0})
    d["train"] = np.arange(n) < 40
    d["es"] = d.train & d.month.isin([3, 9])
    d["fit"] = d.train & ~d.es
    d["holdout"] = ~d.train
    p = tmp_path / "src.parquet"
    d.to_parquet(p)
    pathlib.Path(str(p) + ".cols.json").write_text(json.dumps({"num": ["proxy", "aobt_sec", "mvt_sec", "sched_sec", "hr"],
                                                               "cat": ["ap"]}))
    return p, d


def test_build_input_removes_the_seconds_physically_and_keeps_every_other_column(tmp_path):
    """Rehearsed RED: keeping the columns but dropping them only from cols.json (the file still carries them)."""
    src, d = _src(tmp_path)
    dst = tmp_path / "out" / "nsc.parquet"
    info = CB.build_input(src, dst, batch_rows=7, allow_roots=(tmp_path,))
    got = pd.read_parquet(dst)
    assert not any(c in got.columns for c in SEC) and list(got.columns) == [c for c in d.columns if c not in SEC]
    pd.testing.assert_frame_equal(got, d.drop(columns=SEC).reset_index(drop=True))
    cols = json.loads(pathlib.Path(str(dst) + ".cols.json").read_text())
    assert cols == {"num": ["proxy", "hr"], "cat": ["ap"]}
    assert info["n_rows"] == 50 and info["verified_columns"] == len(got.columns)
    with pytest.raises(FileExistsError):
        CB.build_input(src, dst, allow_roots=(tmp_path,))


def test_build_input_smoke_sample_keeps_the_mask_contract(tmp_path):
    src, d = _src(tmp_path)
    dst = tmp_path / "s.parquet"
    CB.build_input(src, dst, every=3, n_holdout=4, allow_roots=(tmp_path,))
    got = pd.read_parquet(dst)
    assert int(got.holdout.sum()) == 4 and int(got.train.sum()) == len(range(0, 40, 3))
    assert (got.fit | got.es).equals(got.train) and not (got.train & got.holdout).any()


def test_worker_cmd_is_the_venv_worker_with_every_param():
    cmd = CB.worker_cmd(pathlib.Path("/x/in.parquet"), pathlib.Path("/x/out.npy"), CB.PARAMS)
    assert cmd[:3] == [str(CB.CN.VENV_PY), "-u", str(ROOT / "scripts/catboost_native_worker.py")]
    assert cmd[3:9] == ["--input", "/x/in.parquet", "--target", "delta", "--output", "/x/out.npy"]
    for k, v in CB.PARAMS.items():
        assert cmd[cmd.index("--" + k.replace("_", "-")) + 1] == str(v)


class _P:
    def __init__(self, n, rc=0):
        self.pid, self.n, self.rc, self.killed = 55, n, rc, False

    def poll(self):
        if self.killed:
            return -9
        self.n -= 1
        return None if self.n >= 0 else self.rc

    def kill(self):
        self.killed = True

    def wait(self):
        return -9 if self.killed else self.rc


def test_guard_worker_kills_it_above_the_limit_as_exit_70():
    """The worker is a venv subprocess the runner's own RSS guard cannot see. Rehearsed RED: '>' vs '>=' not
    distinguishable; dropping the kill."""
    rss = iter([1.0, 5.9, 6.2])
    rc, peak = CB.guard_worker(_P(10), lambda pid: next(rss), limit_gib=6.0, sleep=lambda s: None, log=lambda m: None)
    assert (rc, peak) == (70, 6.2)
    rc, peak = CB.guard_worker(_P(2, rc=0), lambda pid: 1.5, limit_gib=6.0, sleep=lambda s: None, log=lambda m: None)
    assert (rc, peak) == (0, 1.5)


def test_run_checks_the_worker_output_and_writes_the_report(tmp_path):
    """Every line after the worker (BC-24): prediction count / finiteness, holdout ids in input order, move vs shipped C,
    report. Rehearsed RED: aligning the npy to all rows instead of the holdout rows."""
    src, d = _src(tmp_path)
    ship_c = np.full(10, 3.0)

    def fake_worker(cmd, log):
        out = pathlib.Path(cmd[cmd.index("--output") + 1])
        np.save(out, np.full(10, 5.0))
        out.with_suffix(".json").write_text(json.dumps({"best_iteration": 3, "n_ref": 4, "peak_rss_gb": 0.1}))
        return 0, 0.1
    rep = CB.run(False, tmp_path / "o", src, shipped_c=ship_c, worker=fake_worker, allow_roots=(tmp_path,),
                 n_expected=10)
    assert rep["status"] == "PASS" and rep["n_holdout"] == 10 and rep["numeric"] == ["proxy", "hr"]
    assert rep["move_vs_shipped_c"]["rms_taxi_move_s"] == pytest.approx(2.0, abs=1e-9)
    assert rep["holdout_ids_sha256"] == CB.ids_sha(d.MVT_ID_mvt[d.holdout].to_numpy())
    assert json.loads((tmp_path / "o" / "report.json").read_text()) == rep
    assert len(rep["outputs"]) == 4                       # input + cols.json + npy + worker json


def test_run_fails_on_a_worker_error_or_a_bad_prediction(tmp_path):
    src, _ = _src(tmp_path)
    with pytest.raises(RuntimeError, match="worker exited 70"):
        CB.run(False, tmp_path / "a", src, shipped_c=None, worker=lambda c, log: (70, 6.2), allow_roots=(tmp_path,))

    def nan_worker(cmd, log):
        out = pathlib.Path(cmd[cmd.index("--output") + 1])
        np.save(out, np.r_[np.full(9, 1.0), np.nan])
        out.with_suffix(".json").write_text("{}")
        return 0, 0.1
    with pytest.raises(AssertionError, match="non-finite"):
        CB.run(False, tmp_path / "b", src, shipped_c=None, worker=nan_worker, allow_roots=(tmp_path,))


def test_this_process_never_imports_catboost():
    assert "catboost" not in sys.modules


def test_parse_args_requires_the_rss_limit():
    with pytest.raises(SystemExit):
        CB.parse_args([])
    assert CB.parse_args(["--rss-limit-gib", "6", "--smoke"]).smoke is True
