"""FM1 learners behind one protocol (prc/pipeline/fm1/learners.py) — AT3 (determinism), AT4 (bit-exact save/load), AT11.

Every data draw is seeded. Tolerances: bit-exact (0.0) where the claim is bit-exactness; the only float tolerance used is
stated where it appears.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np
import pytest

from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import learners as L

FEATS = ["a", "b", "c", "d"]


def _data(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATS)))
    X[rng.random(n) < 0.05, 2] = np.nan                      # NaN mode, as the ADS-B block has
    y = 50 * X[:, 0] - 20 * X[:, 1] ** 2 + np.nan_to_num(X[:, 2]) * 5 + rng.normal(scale=5, size=n)
    es = np.zeros(n, bool)
    es[rng.choice(n, n // 5, replace=False)] = True
    return X, y, ~es, es


LGB_SMALL = dict(learning_rate=0.1, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1)


def _lgb(**kw):
    return L.LightGBMLearner(params={**LGB_SMALL, **kw}, max_rounds=200, patience=10, threads=2)


# ---------------------------------------------------------------------------------------------- LightGBM

def test_lightgbm_fits_and_beats_the_mean():
    X, y, fit, es = _data()
    m = _lgb()
    m.fit(X, y, fit, es, seeds=(0, 1), feature_names=FEATS)
    p = m.predict(X)
    assert p.shape == (len(y),) and np.isfinite(p).all()
    assert np.sqrt(np.mean((p - y) ** 2)) < 0.5 * np.std(y)
    d = m.describe()
    assert d["name"] == "lightgbm" and d["seeds"] == [0, 1] and d["n_ref"] >= 50 and d["best_iter"] >= 1


def test_lightgbm_refit_count_rule():
    """n_ref = max(50, int(best_iter * n_all / n_fit)) (lgbm_submit.n_refit). Mutation: drop the max(50, .) floor -> red."""
    assert L.n_refit(10, 1000, 800) == 50
    assert L.n_refit(400, 1000, 800) == 500
    with pytest.raises(ValueError, match="n_refit"):
        L.n_refit(0, 10, 5)


def test_lightgbm_is_bit_deterministic():
    """AT3: two fits, bit-identical predictions. The flags themselves are asserted too: whether a 2-thread fit on 600 rows
    would diverge without them was not measured, so the flags are pinned directly (M15 in MUTATION_FM1.md is killed by
    that assertion)."""
    X, y, fit, es = _data()
    a, b = _lgb(), _lgb()
    a.fit(X, y, fit, es, seeds=(0, 1), feature_names=FEATS)
    b.fit(X, y, fit, es, seeds=(0, 1), feature_names=FEATS)
    assert np.array_equal(a.predict(X), b.predict(X))
    p = L.lgb_params(LGB_SMALL, seed=3, threads=2)
    assert p["deterministic"] is True and p["force_col_wise"] is True and p["num_threads"] == 2
    assert p["seed"] == p["bagging_seed"] == p["feature_fraction_seed"] == p["data_random_seed"] == 3


def test_lightgbm_seeds_matter():
    X, y, fit, es = _data()
    a, b = _lgb(), _lgb()
    a.fit(X, y, fit, es, seeds=(0,), feature_names=FEATS)
    b.fit(X, y, fit, es, seeds=(1,), feature_names=FEATS)
    assert not np.array_equal(a.predict(X), b.predict(X))


def test_lightgbm_save_load_is_bit_exact(tmp_path):
    """AT4. Mutation: save with num_iteration=best_iter instead of the full refit -> predictions differ -> red."""
    X, y, fit, es = _data()
    m = _lgb()
    m.fit(X, y, fit, es, seeds=(0, 1), feature_names=FEATS)
    files = m.save(tmp_path)
    assert sorted(files) == ["learner.json", "lgb_seed0.txt", "lgb_seed1.txt"]
    back = L.load_learner(tmp_path)
    assert isinstance(back, L.LightGBMLearner)
    assert np.array_equal(back.predict(X), m.predict(X))
    assert back.describe() == m.describe()


def test_lightgbm_refuses_bad_inputs():
    X, y, fit, es = _data()
    m = _lgb()
    with pytest.raises(E.SchemaError, match="no early-stopping rows"):
        m.fit(X, y, fit, np.zeros(len(y), bool), seeds=(0,), feature_names=FEATS)
    with pytest.raises(E.SchemaError, match="overlap"):
        m.fit(X, y, fit, fit, seeds=(0,), feature_names=FEATS)
    with pytest.raises(E.SchemaError, match="non-finite target"):
        yy = y.copy()
        yy[3] = np.nan
        m.fit(X, yy, fit, es, seeds=(0,), feature_names=FEATS)
    with pytest.raises(E.SchemaError, match="4 feature names for 3 columns"):
        m.fit(X[:, :3], y, fit, es, seeds=(0,), feature_names=FEATS)
    with pytest.raises(E.SchemaError, match="length"):
        m.fit(X, y[:-1], fit, es, seeds=(0,), feature_names=FEATS)
    with pytest.raises(E.PipelineError, match="not fitted"):
        m.predict(X)
    m.fit(X, y, fit, es, seeds=(0,), feature_names=FEATS)
    with pytest.raises(E.SchemaError, match="expected 4 columns"):
        m.predict(X[:, :3])


def test_lightgbm_refuses_a_forbidden_feature_name():
    """AT2 at the learner boundary: a design with a label-seconds column never fits."""
    X, y, fit, es = _data()
    with pytest.raises(E.ForbiddenInputError, match="mvt_sec"):
        _lgb().fit(X, y, fit, es, seeds=(0,), feature_names=["a", "b", "c", "mvt_sec"])


# ---------------------------------------------------------------------------------------------- CatBoost worker

@pytest.mark.slow          # > 1 s (rule 26): real fits / a worker process / loading lgbm_fold
def test_catboost_runs_in_the_worker_and_round_trips(tmp_path):
    """AT11: CatBoost only in ~/.venvs/prc-catboost; this process never imports catboost. AT3 + AT4 for the worker learner.
    Mutation: have CatBoostWorkerLearner import catboost in-process -> the sys.modules assertion goes red (run in a
    child interpreter so an earlier test's import cannot mask it)."""
    code = f"""
import sys, numpy as np, pathlib
sys.path.insert(0, {str(pathlib.Path(__file__).resolve().parents[2])!r})
from prc.pipeline.fm1 import learners as L
rng = np.random.default_rng(0)
X = rng.normal(size=(400, 3)); y = 30 * X[:, 0] + rng.normal(size=400)
es = np.zeros(400, bool); es[:80] = True
kw = dict(params=dict(learning_rate=0.2, depth=4, l2_leaf_reg=3.0, subsample=0.8), max_rounds=60, patience=10, threads=2)
a = L.CatBoostWorkerLearner(**kw); a.fit(X, y, ~es, es, seeds=(0, 1), feature_names=["a", "b", "c"])
b = L.CatBoostWorkerLearner(**kw); b.fit(X, y, ~es, es, seeds=(0, 1), feature_names=["a", "b", "c"])
pa, pb = a.predict(X), b.predict(X)
assert np.array_equal(pa, pb), "catboost not deterministic"
assert np.sqrt(np.mean((pa - y) ** 2)) < 0.5 * np.std(y)
d = pathlib.Path({str(tmp_path)!r})
files = a.save(d)
assert sorted(files) == ["cb_seed0.cbm", "cb_seed1.cbm", "learner.json"], files
c = L.load_learner(d)
assert np.array_equal(c.predict(X), pa), "reload not bit-exact"
assert "catboost" not in sys.modules, "catboost was imported in the main process"
print("OK")
"""
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-3000:] + r.stderr[-3000:]


def test_catboost_worker_failure_is_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "CATBOOST_PYTHON", tmp_path / "no-such-python")
    X, y, fit, es = _data(100)
    m = L.CatBoostWorkerLearner(params=dict(learning_rate=0.2, depth=4, l2_leaf_reg=3.0, subsample=0.8), max_rounds=10,
                                patience=5, threads=1)
    with pytest.raises(E.WorkerError, match="CatBoost worker interpreter .* does not exist"):
        m.fit(X, y, fit, es, seeds=(0,), feature_names=FEATS)


def test_catboost_worker_error_output_is_surfaced(tmp_path, monkeypatch):
    """A worker that exits non-zero raises WorkerError with its stderr tail (fake worker = a script that fails)."""
    fake = tmp_path / "fake_worker.py"
    fake.write_text("import sys\nsys.stderr.write('boom: bad matrix\\n')\nsys.exit(3)\n")
    monkeypatch.setattr(L, "CATBOOST_WORKER", fake)
    X, y, fit, es = _data(100)
    m = L.CatBoostWorkerLearner(params=dict(learning_rate=0.2, depth=4, l2_leaf_reg=3.0, subsample=0.8), max_rounds=10,
                                patience=5, threads=1)
    with pytest.raises(E.WorkerError, match=r"exit 3.*boom: bad matrix"):
        m.fit(X, y, fit, es, seeds=(0,), feature_names=FEATS)


# ---------------------------------------------------------------------------------------------- HGB (unmatched body)

@pytest.mark.slow          # > 1 s (rule 26): real fits / a worker process / loading lgbm_fold
def test_hgb_fits_deterministically_and_round_trips(tmp_path):
    X, y, fit, es = _data()
    kw = dict(params=dict(max_iter=100, n_iter_no_change=10, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.1))
    a, b = L.HGBLearner(**kw), L.HGBLearner(**kw)
    a.fit(X, y, fit, es, seeds=(0, 1), feature_names=FEATS)
    b.fit(X, y, fit, es, seeds=(0, 1), feature_names=FEATS)
    assert np.array_equal(a.predict(X), b.predict(X))
    assert np.sqrt(np.mean((a.predict(X) - y) ** 2)) < 0.6 * np.std(y)
    files = a.save(tmp_path)
    assert sorted(files) == ["hgb_seed0.pkl", "hgb_seed1.pkl", "learner.json"]
    assert np.array_equal(L.load_learner(tmp_path).predict(X), a.predict(X))


def test_hgb_without_es_rows_uses_its_internal_split():
    X, y, fit, es = _data()
    m = L.HGBLearner(params=dict(max_iter=50, n_iter_no_change=5, max_leaf_nodes=15, min_samples_leaf=20,
                                 learning_rate=0.1))
    m.fit(X, y, np.ones(len(y), bool), np.zeros(len(y), bool), seeds=(0,), feature_names=FEATS)
    assert m.describe()["validation"] == "internal 10% split"


# ---------------------------------------------------------------------------------------------- registry

def test_make_learner_from_config():
    from prc.pipeline.fm1 import knobs as K
    cfg = K.parse_config({})
    m = L.make_learner("lightgbm", cfg)
    assert isinstance(m, L.LightGBMLearner) and m.params["learning_rate"] == 0.01 and m.threads == 4
    assert m.max_rounds == 50_000 and m.patience == 200
    c = L.make_learner("catboost", cfg)
    assert isinstance(c, L.CatBoostWorkerLearner) and c.params["depth"] == 8 and c.max_rounds == 30_000
    h = L.make_learner("hgb", cfg)
    assert isinstance(h, L.HGBLearner) and h.params["max_iter"] == 1000 and h.params["n_iter_no_change"] == 50
    s = L.make_learner("lightgbm", cfg, learning_rate=0.05)
    assert s.params["learning_rate"] == 0.05


@pytest.mark.parametrize("name,match", [("xgboost", "CUT"), ("mlp", "not a days 1-3 learner"),
                                        ("randomforest", "unknown learner")])
def test_make_learner_refuses(name, match):
    from prc.pipeline.fm1 import knobs as K
    with pytest.raises(E.LearnerCutError, match=match):
        L.make_learner(name, K.parse_config({}))


def test_load_learner_refuses_a_bad_directory(tmp_path):
    with pytest.raises(E.ArtifactError, match="learner.json"):
        L.load_learner(tmp_path)
    (tmp_path / "learner.json").write_text('{"name": "xgboost"}')
    with pytest.raises(E.ArtifactError, match="unknown learner 'xgboost'"):
        L.load_learner(tmp_path)


# ---------------------------------------------------------------------------------------------- memory (2026-10-01)

def test_lightgbm_fit_is_unchanged_by_the_memory_refactor_pinned():
    """Pins the 2026-09-28 learner's predictions (sha256 of predict(X) bytes, 600 rows, seeds 0 + 1, 2 threads,
    computed with the pre-refactor code on 2026-10-01 before any edit). The memory refactor (free_raw_data default,
    no X[tr] copy when fit | es covers every row, datasets released before the refit) must not move a single bit:
    LightGBM's bins are fixed at Dataset construction, so a changed construction order or reference would show here.
    Mutation: build the refit Dataset from X[fit_mask] instead of the fit + es rows -> red."""
    import hashlib
    X, y, fit, es = _data()
    m = _lgb()
    m.fit(X, y, fit, es, seeds=(0, 1), feature_names=FEATS)
    assert hashlib.sha256(m.predict(X).tobytes()).hexdigest() == \
        "91d45222d88f38d64a9d24efabc3afd537b44d02fd1ad4b73aa7805b03e15353"


def test_lightgbm_fit_does_not_hold_three_row_copies_of_the_design():
    """The 2026-09-28 fit kept X[fit], X[es] (free_raw_data=False) and X[tr] — a full copy when fit | es is every
    row — alive through the refit: 3.2 x X.nbytes traced (measured 2026-10-01). Now the ES datasets drop their raw
    rows after construction and the refit uses X itself: peak traced allocation <= 1.8 x X.nbytes. numpy copies are
    traced; LightGBM's own bins are not (and are the same either way). Seen red against the old code (3.2 x)."""
    import tracemalloc
    X, y, fit, es = _data(n=50_000, seed=1)
    tracemalloc.start()
    m = _lgb()
    m.fit(X, y, fit, es, seeds=(0,), feature_names=FEATS)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert m.best_iter >= 1
    assert peak <= 1.8 * X.nbytes, f"peak {peak / X.nbytes:.2f} x X.nbytes"


def test_lightgbm_rows_in_neither_mask_are_excluded_from_the_refit():
    """The invariant behind the no-copy refit: rows that are neither fit nor early-stopping rows take no part. A fit on
    the full X with 10 % of rows masked out must equal a fit on the masked subset alone. Mutation: refit on X when
    tr.any() instead of tr.all() -> red."""
    X, y, fit, es = _data(n=2_000, seed=2)
    rng = np.random.default_rng(3)
    drop = rng.random(len(y)) < 0.1
    fit, es = fit & ~drop, es & ~drop
    a, b = _lgb(), _lgb()
    a.fit(X, y, fit, es, seeds=(0,), feature_names=FEATS)
    tr = fit | es
    b.fit(X[tr], y[tr], fit[tr], es[tr], seeds=(0,), feature_names=FEATS)
    assert a.n_ref == b.n_ref and a.best_iter == b.best_iter
    assert np.array_equal(a.predict(X), b.predict(X))
