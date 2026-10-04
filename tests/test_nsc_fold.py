"""scripts/analysis/nsc_fold.py — NSC: the shipped FW fold recipe minus the three label-seconds columns
(plans/PREREG_nsc_no_seconds_2026_09_24.md), and the project-wide negative invariant that no *_sec column enters a
production feature list outside the frozen v33 legacy set.

Nothing here loads real data or takes the heavy lock: the fold loader and the fitter are injected as fakes.
Mutation rehearsal (2026-09-24): see each test's docstring.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


N = _load("nsc_fold", "scripts/analysis/nsc_fold.py")
LF = N.LF
SEC = ("aobt_sec", "mvt_sec", "sched_sec")


# ---------------------------------------------------------------- the feature lists

def test_fw_premise_the_shipped_93_hold_the_three_seconds_at_14_15_16():
    """Pins the audit's C1 premise, so a later edit of BASE cannot silently make the ablation a no-op."""
    fw = LF.FEATS_QUEUE_ORDER_WEATHER
    assert len(fw) == 93
    assert [fw.index(c) for c in SEC] == [14, 15, 16]


def test_nsc_fw_is_the_shipped_93_minus_exactly_the_three_in_order():
    """Rehearsed RED: dropping only mvt_sec (len 92) and re-sorting the list both fail."""
    assert N.FEATS_NSC_FW == [c for c in LF.FEATS_QUEUE_ORDER_WEATHER if c not in SEC]
    assert len(N.FEATS_NSC_FW) == 90
    assert not any(c.endswith("_sec") for c in N.FEATS_NSC_FW)


def test_nsc_f_and_catboost_lists_are_their_production_lists_minus_the_three():
    import catboost_native as cn
    assert N.FEATS_NSC_F == [c for c in LF.FEATS_QUEUE_ORDER if c not in SEC] and len(N.FEATS_NSC_F) == 80
    num = cn.numeric_columns()
    assert N.nsc_catboost_numeric() == [c for c in num if c not in SEC] and len(N.nsc_catboost_numeric()) == 56


def test_drop_seconds_refuses_a_list_that_does_not_hold_all_three():
    """A silent no-op on the wrong list is the failure this guards. Rehearsed RED: returning the input unchanged."""
    with pytest.raises(ValueError, match=r"does not hold all three seconds columns.*missing \['mvt_sec'\]"):
        N.drop_seconds(["proxy", "aobt_sec", "sched_sec"])
    assert N.drop_seconds(["a", "aobt_sec", "mvt_sec", "b", "sched_sec"]) == ["a", "b"]


def test_drop_seconds_refuses_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        N.drop_seconds(["a", "a", "aobt_sec", "mvt_sec", "sched_sec"])


# ---------------------------------------------------------------- the negative invariant

def test_seconds_scanner_flags_a_new_list_and_ignores_non_lists():
    """The scanner itself must go RED on a planted list (a check never seen red proves nothing)."""
    fake = types.SimpleNamespace(GOOD=["proxy", "sp"], BAD=("hr", "mvt_sec"), TEXT="mvt_sec", N=3)
    assert N.lists_with_seconds({"fake": fake}) == {"fake.BAD"}


def test_no_seconds_column_enters_a_production_list_outside_the_frozen_v33_legacy_set():
    """Negative invariant: every module-level list/tuple in the production modules that holds a *_sec name must be in
    LEGACY_SECONDS_LISTS (the v33 lists, frozen so v33 stays reproducible); every legacy entry must still hold one (no rot);
    every NSC list is clean. Rehearsed RED: adding a list `X = BASE + ["q"]` to stand_ab, and adding FEATS_NSC_FW to
    the legacy set, each fail."""
    found = N.lists_with_seconds(N.production_modules())
    assert found == set(N.LEGACY_SECONDS_LISTS), sorted(found ^ set(N.LEGACY_SECONDS_LISTS))
    assert not any(name.split(".")[-1].startswith(("FEATS_NSC", "NSC_")) for name in found)


def test_sklearn_side_production_sources_never_mention_a_seconds_column():
    """The unmatched lane and Rome chain are imported by AST, not executed: no string literal *_sec anywhere."""
    for rel in N.NO_SECONDS_SOURCES:
        tree = ast.parse((ROOT / rel).read_text())
        bad = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
               and n.value in SEC]
        assert bad == [], (rel, bad)


@pytest.mark.devtree
def test_adsb_stack_manifest_features_hold_no_seconds_column():
    feats = json.loads((ROOT / "data/adsb/v2/models_v4/MANIFEST.json").read_text())["feats"]
    assert len(feats) == 33 and not any(c in SEC for c in feats)


# ---------------------------------------------------------------- scoring

def _record(n=12, seed=0):
    rng = np.random.default_rng(seed)
    proxy = rng.uniform(400, 1200, n)
    delta = rng.normal(0, 50, n)
    y = proxy - delta
    return pd.DataFrame({"row": np.arange(n) * 3, "month": np.r_[np.ones(n // 2, int), np.full(n - n // 2, 7)],
                         "ap": ["EGLL", "LIRF"] * (n // 2), "y": y, "proxy": proxy, "delta": delta,
                         "treatment": delta + rng.normal(0, 20, n)})


def test_pair_cost_is_zero_with_a_zero_width_interval_for_identical_predictions():
    r = _record()
    day = np.repeat(np.arange(6), 2)
    out = N.pair_cost(r.y, r.proxy, r.treatment, r.treatment, day, n_boot=50, seed=0)
    assert out["cost_bmse"] == 0.0 and out["ci95_day"] == [0.0, 0.0] and out["ci95_row"] == [0.0, 0.0]


def test_pair_cost_is_sse_difference_over_the_board_row_count_with_the_one_second_floor():
    """cost = (SSE_nsc - SSE_inc) / 344,841; the prediction is max(proxy - delta, 1).
    Rehearsed RED: dividing by the fold row count, and dropping the floor, each fail."""
    y = np.array([100.0, 50.0])
    proxy = np.array([100.0, 10.0])
    inc = np.array([0.0, 0.0])            # predictions 100, 10  -> se 0, 1600
    nsc = np.array([10.0, 100.0])         # predictions 90, max(-90,1)=1 -> se 100, 2401
    out = N.pair_cost(y, proxy, inc, nsc, np.array([0, 1]), n_boot=20, seed=0)
    assert out["cost_bmse"] == pytest.approx((100 + 2401 - 1600) / 344_841, rel=1e-12)
    assert out["sse_inc"] == pytest.approx(1600.0) and out["sse_nsc"] == pytest.approx(2501.0)


def test_pair_cost_refuses_misaligned_lengths_and_non_finite_input():
    with pytest.raises(ValueError, match="length"):
        N.pair_cost(np.ones(3), np.ones(3), np.ones(3), np.ones(2), np.zeros(3), n_boot=5, seed=0)
    with pytest.raises(ValueError, match="finite"):
        N.pair_cost(np.ones(2), np.ones(2), np.array([0.0, np.nan]), np.ones(2), np.zeros(2), n_boot=5, seed=0)


def test_day_block_bootstrap_resamples_whole_days():
    """A cost that lives on one day of two: row bootstrap and day bootstrap must differ; the day CI must reach 0."""
    y = np.full(40, 500.0)
    proxy = np.full(40, 500.0)
    inc = np.zeros(40)
    nsc = np.where(np.arange(40) < 20, 30.0, 0.0)
    day = (np.arange(40) >= 20).astype(int)
    out = N.pair_cost(y, proxy, inc, nsc, day, n_boot=400, seed=0)
    assert out["ci95_day"][0] == pytest.approx(0.0)
    assert out["ci95_row"][0] > 0.0


def test_check_alignment_requires_row_month_ap_y_proxy_delta_to_match_exactly():
    r = _record()
    base = r[["row", "month", "ap", "y", "proxy", "delta"]].copy()
    N.check_alignment(base, r)
    bad = r.copy()
    bad.loc[3, "y"] += 1e-9
    with pytest.raises(AssertionError, match="column y"):
        N.check_alignment(base, bad)
    with pytest.raises(AssertionError, match="rows"):
        N.check_alignment(base.iloc[:-1], r)


# ---------------------------------------------------------------- the registered decision rule

@pytest.mark.parametrize("cost, lo, want", [
    (40.0, -30.0, "PROCEED"),
    (-25.0, -90.0, "PROCEED"),
    (336.9, 100.0, "PROCEED"),
    (337.0, 100.0, "ESCALATE"),
    (200.0, 150.0, "ESCALATE"),
    (200.0, 149.9, "PROCEED"),
])
def test_decide_follows_the_registered_rule(cost, lo, want):
    """Rehearsed RED: '>' instead of '>=' at the 337 bound and at the 150 lower-bound bar."""
    assert N.decide(cost, lo)["decision"] == want


def test_decide_refuses_a_non_finite_cost():
    with pytest.raises(ValueError, match="finite"):
        N.decide(float("nan"), 0.0)


def test_mde_is_2_8_standard_errors_from_the_day_interval():
    assert N.mde_from_ci([-39.2, 39.2]) == pytest.approx(2.8 * 20.0)


# ---------------------------------------------------------------- orchestration (fakes; BC-24 post-compute path)

def _fake_fold(n_te=8, n_tr=6, feats=None):
    n = n_te + n_tr
    rng = np.random.default_rng(1)
    feats = list(feats)
    proxy = rng.uniform(300, 900, n)          # labels first: the same fold whatever the design width
    dlt = rng.normal(0, 40, n)
    X = rng.normal(size=(n, len(feats))).astype("float32")
    X[:, feats.index("doy")] = np.r_[np.repeat([5, 6, 190, 191], 2), np.arange(n_tr)]
    te = np.r_[np.ones(n_te, bool), np.zeros(n_tr, bool)]
    y = proxy - dlt
    base = pd.DataFrame({"row": np.flatnonzero(te).astype("int64"), "month": np.r_[1, 1, 1, 1, 7, 7, 7, 7].astype("int32"),
                         "ap": ["EGLL", "LIRF"] * (n_te // 2), "y": y[te], "proxy": proxy[te], "delta": dlt[te],
                         "sp": np.zeros(n_te)})
    return dict(X=X, y=y, dlt=dlt, proxy=proxy, te=te, feats=feats, base=base,
                airports=["EGLL", "LIRF"], ap_code=np.r_[np.tile([0, 1], n_te // 2), np.zeros(n_tr, int)])


def _fake_fit(fold, X, *a, seeds=(0, 1, 2), **k):
    te = fold["te"]
    single = {s: fold["dlt"][te] + 5.0 + s for s in seeds}
    pooled = np.mean([single[s] for s in seeds], axis=0)
    return dict(delta=pooled, single=single, single_rmse={s: 1.0 + s for s in seeds}, best_iter=10, n_ref=12,
                info={"es_rmse_taxi_time_NOT_A_RESULT": 1.0}, n_features=X.shape[1])


def test_run_scores_the_fit_against_the_record_and_writes_every_output(tmp_path):
    """Executes every line after the compute on real-shaped synthetic output, ending with the runner's own writers."""
    fold = _fake_fold(feats=N.FEATS_NSC_FW)
    rec = fold["base"][["row", "month", "ap", "y", "proxy", "delta"]].copy()
    rec["treatment"] = rec.delta + 3.0
    loaded = {}

    def load(feats, smoke):
        loaded["feats"] = list(feats)
        return fold

    out = N.run(smoke=False, out_dir=tmp_path, load=load, fit=_fake_fit, record=rec, n_boot=30)
    assert loaded["feats"] == N.FEATS_NSC_FW
    rep = json.loads((tmp_path / "nsc_fold_report.json").read_text())
    assert rep["decision"]["decision"] in ("PROCEED", "ESCALATE")
    assert rep["n_rows"] == 8 and rep["features"] == N.FEATS_NSC_FW and rep["n_features"] == 90
    se_inc = (rec.y - np.maximum(rec.proxy - rec.treatment, 1)) ** 2
    se_nsc = (rec.y - np.maximum(rec.proxy - (rec.delta + 6.0), 1)) ** 2
    assert rep["pooled"]["cost_bmse"] == pytest.approx(float((se_nsc - se_inc).sum()) / 344_841)
    assert set(rep["by_airport"]) == {"EGLL", "LIRF"} and set(rep["by_month"]) == {"1", "7"}
    p = pd.read_parquet(tmp_path / "nsc_fold_preds.parquet")
    assert list(p.columns) == ["row", "month", "ap", "y", "proxy", "delta", "doy", "incumbent",
                               "delta_hat_seed0", "delta_hat_seed1", "delta_hat_seed2", "nsc"]
    assert out["pooled"]["cost_bmse"] == rep["pooled"]["cost_bmse"]


def test_run_refuses_to_overwrite_outputs(tmp_path):
    (tmp_path / "nsc_fold_report.json").write_text("{}")
    with pytest.raises(FileExistsError):
        N.run(smoke=False, out_dir=tmp_path, load=None, fit=None, record=None)


def test_run_refuses_a_fold_whose_design_carries_a_seconds_column(tmp_path):
    fold = _fake_fold(feats=LF.FEATS_QUEUE_ORDER_WEATHER)
    with pytest.raises(AssertionError, match="seconds"):
        N.run(smoke=False, out_dir=tmp_path, load=lambda f, s: fold, fit=_fake_fit, record=None, n_boot=5)


def test_smoke_pairs_against_an_in_process_incumbent_refit(tmp_path):
    """In smoke there is no record for months 1-3: the incumbent is the same fitter on the 93 columns."""
    calls = []

    def load(feats, smoke):
        calls.append(len(feats))
        return _fake_fold(feats=feats)

    rep = N.run(smoke=True, out_dir=tmp_path, load=load, fit=_fake_fit, record=None, n_boot=10)
    assert calls == [90, 93] and rep["smoke"] is True and rep["incumbent_source"] == "in-process refit (smoke)"


def test_parse_args_defaults_and_smoke():
    a = N.parse_args([])
    assert a.smoke is False and a.out_dir is None
    assert N.parse_args(["--smoke"]).smoke is True


def test_regression_nsc_smoke_bootstrap_memory_is_bounded():
    """Found by the 2026-09-24 real-data smoke: the (n_boot x n_rows) index matrix of the row bootstrap peaked the
    process at 7.1 GB on 152,250 rows (2,000 draws), and would need ~10.8 GB on the real 339,015. The bootstrap must be
    chunked: tracemalloc peak on the real row count at 200 draws stays under 100 MB (it was 542 MB+ unchunked)."""
    import tracemalloc
    n = 339_015
    g = np.random.default_rng(0).normal(size=n)
    tracemalloc.start()
    ci = N._boot(g, np.arange(n), 200, 0)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert peak < 100 * 1024 ** 2, peak / 1024 ** 2
    assert ci[0] < g.sum() / N.N_BOARD < ci[1]
