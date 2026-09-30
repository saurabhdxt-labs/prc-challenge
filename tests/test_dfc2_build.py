"""Tests for scripts/analysis/dfc2_build.py (arm DFC2, plans/PREREG_dfc2_networkflow_forward_2026_09_23.md).

Every test names the failure it catches. The post-compute lines (gate G1, prediction scoring, the T7 tripwire
clauses, the Stage-2 verdict merge) are tested on synthetic inputs BEFORE the first real run (BC-24: a post-compute
crash after a long fit). The DFC block wrapper is tested against the fixed builder's F1 guard (BC-25).
"""
from __future__ import annotations

import hashlib, importlib.util, pathlib, sys
import numpy as np, pandas as pd, pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(n, r):
    if n in sys.modules:
        return sys.modules[n]
    s = importlib.util.spec_from_file_location(n, ROOT / r); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


B = _load("dfc2_build", "scripts/analysis/dfc2_build.py")
D = B.D
R = B.R


def test_imported_module_is_the_one_under_test():
    """Catches the AGS symlink-mirror landmine: the module under test must be the file next to this ROOT."""
    assert pathlib.Path(B.__file__).resolve() == (ROOT / "scripts/analysis/dfc2_build.py").resolve()
    assert pathlib.Path(D.__file__).resolve() == (ROOT / "scripts/analysis/dfc_build.py").resolve()
    assert pathlib.Path(R.__file__).resolve() == (ROOT / "scripts/analysis/rcr2_build.py").resolve()


# ---- constants and the registration -------------------------------------------------------------------------
def test_registration_hash_matches_the_log():
    """The runner's PREREG_SHA is the REGISTERED line's full hash; the file's pre-RESULT text still hashes to it."""
    log = (ROOT / "reports/dfc2_prereg_hashes.log").read_text().splitlines()
    reg = [l for l in log if "  REGISTERED  " in l]
    assert len(reg) == 1 and f"(full {B.PREREG_SHA})" in reg[0]
    assert B._prereg_ok() is not None


def test_prereg_ok_refuses_a_tampered_registration(tmp_path, monkeypatch):
    p = tmp_path / "prereg.md"; p.write_text("# tampered\n")
    monkeypatch.setattr(B, "PREREG", p)
    with pytest.raises(AssertionError, match="registration text changed"):
        B._prereg_ok()


def test_prereg_ok_accepts_a_result_appended_after_the_hash(tmp_path, monkeypatch):
    body = "# reg\nclauses\n"
    p = tmp_path / "prereg.md"; p.write_text(body + "\n## RESULT\nnumbers\n")
    monkeypatch.setattr(B, "PREREG", p)
    monkeypatch.setattr(B, "PREREG_SHA", hashlib.sha256(body.encode()).hexdigest())
    assert B._prereg_ok() == hashlib.sha256(p.read_bytes()).hexdigest()


def test_constants_match_the_registration():
    assert B.YEAR == 2025 and B.N_ARRIVAL_FILES == 12
    assert B.DEP_ONLY == ["dfc_ades_dep_1h", "dfc_ades_dep_3h", "dfc_ades_dep_1h_dev"]
    assert B.ARMS == {"A6": list(D.DFC_FEATS), "A7": B.DEP_ONLY}
    assert B.RSS_LIMIT == 6 * 1024 ** 3
    assert B.SMOKE_STRIDE == 8          # 1-in-40 empties hour 23 of in-set fit rows on L-Jan (docstring at the constant)
    assert B.REPORT1.name == "dfc2_stage1_report.json" and B.PREDS1.name == "dfc2_stage1_predictions.parquet"
    assert B.REPORT2.name == "dfc2_stage2_report.json" and B.PREDS2.name == "dfc2_fold_predictions.parquet"


def test_rcr2_base_rmse_constants_match_the_stored_report():
    """Item 4 of the baseline read: the CHR forward-leg incumbents are read from the artifact, never asserted."""
    import json
    r = json.loads((ROOT / "reports/rcr2_stage1_report.json").read_text())
    for leg in ("L-Jul", "L-Jan"):
        assert B.RCR2_BASE_RMSE[leg] == r[leg]["y_vs_delta"]["rmse_ref"]


# ---- feature lists ------------------------------------------------------------------------------------------
def test_arm_feats_widths_and_order():
    base = B.arm_feats("base"); a6 = B.arm_feats("A6"); a7 = B.arm_feats("A7")
    assert len(base) == 93 and base == R.production_feats()
    assert a6 == base + list(D.DFC_FEATS) and len(a6) == 99
    assert a7 == base + B.DEP_ONLY and len(a7) == 96


def test_arm_feats_refuses_a_label_column(monkeypatch):
    monkeypatch.setitem(B.ARMS, "A6", list(D.DFC_FEATS) + ["TAXITIME_SEC_mvt"])
    with pytest.raises(AssertionError, match="leak gate 1"):
        B.arm_feats("A6")


def test_arm_feats_refuses_an_unknown_arm():
    with pytest.raises(KeyError):
        B.arm_feats("A8")


def test_arrival_paths_are_the_twelve_2025_files():
    p = B.arrival_paths()
    assert len(p) == 12 and all("training_2025-" in str(x) for x in p) and p == sorted(p)


def test_arrival_paths_refuses_a_partial_year(tmp_path):
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/training_2025-01-01_2025-02-01.parquet").write_bytes(b"")
    with pytest.raises(AssertionError, match="12 monthly"):
        B.arrival_paths(tmp_path)


# ---- the DFC block wrapper -----------------------------------------------------------------------------------
def _frame(n=60, seed=0):
    rng = np.random.default_rng(seed)
    ades = rng.choice(["LFPG", "EGLL", "ZZZZ", "KJFK"], size=n)
    hr = rng.integers(8, 11, n)                      # three hours, so every hour has fit rows under any half mask
    return pd.DataFrame({"ADES_mvt": ades, "doy": rng.integers(100, 103, n), "hr": hr,
                         "tmin": hr * 60 + rng.integers(0, 60, n), "mvt_sec": rng.integers(0, 60, n),
                         "y": rng.normal(600, 100, n), "other": np.arange(n)})


def _stream(year, aps=("LFPG", "EGLL")):
    base = (year - D.EPOCH_YEAR) * 366.0 * 86400.0
    by = {ap: np.sort((base + np.array([100.0, 101.0, 102.0]) * 86400.0)[:, None] + np.arange(0, 86400, 900)[None, :]).ravel()
          for ap in aps}
    allv = np.concatenate(list(by.values()))
    return D.ArrivalStream(by_airport=by, years=frozenset({year}), span=(float(allv.min()), float(allv.max())))


def test_dfc_block_builds_the_six_columns_with_the_registered_nan_pattern():
    f = _frame(); fit = np.zeros(len(f), bool); fit[::2] = True
    out = B.dfc_block(f, fit, _stream(2025))
    assert list(out.columns) == list(D.DFC_FEATS) and len(out) == len(f)
    in_set = out.dfc_ades_in_set.to_numpy()
    assert set(in_set.tolist()) <= {0.0, 1.0}
    assert np.isnan(out.dfc_ades_arr_1h.to_numpy()[in_set == 0]).all()
    assert np.isfinite(out.dfc_ades_arr_1h.to_numpy()[in_set == 1]).all()
    for c in ("dfc_ades_dep_1h", "dfc_ades_dep_3h", "dfc_ades_dep_1h_dev"):
        assert np.isfinite(out[c].to_numpy()).all()


def test_dfc_block_refuses_an_arrival_stream_from_another_year():
    """BC-25 / F1: the fixed builder's guard must reach the caller, not be swallowed."""
    f = _frame(); fit = np.ones(len(f), bool)
    with pytest.raises(AssertionError, match="F1 fail-loud"):
        B.dfc_block(f, fit, _stream(2026))


def test_dfc_block_dev_medians_depend_on_the_fit_mask():
    """Leak gate 2: a wrapper that ignored `fit_mask` (all rows) would give identical _dev columns for two masks."""
    f = _frame(n=200, seed=1)
    a = np.zeros(len(f), bool); a[:100] = True
    b = np.zeros(len(f), bool); b[100:] = True
    oa = B.dfc_block(f, a, _stream(2025)); ob = B.dfc_block(f, b, _stream(2025))
    assert not np.allclose(oa.dfc_ades_dep_1h_dev, ob.dfc_ades_dep_1h_dev)
    assert np.array_equal(oa.dfc_ades_dep_1h, ob.dfc_ades_dep_1h)


def test_dfc_block_reads_only_input_cols(monkeypatch):
    seen = {}
    real = D.build_dfc
    def spy(frame, fit_mask, year, arrivals=None, dev_tables=None):
        seen["cols"] = list(frame.columns); return real(frame, fit_mask, year, arrivals, dev_tables=dev_tables)
    monkeypatch.setattr(D, "build_dfc", spy)
    f = _frame(); B.dfc_block(f, np.ones(len(f), bool), _stream(2025))
    assert seen["cols"] == list(D.INPUT_COLS)


def test_dfc_block_refuses_nan_in_a_departure_column(monkeypatch):
    f = _frame()
    real = D.build_dfc
    def bad(frame, fit_mask, year, arrivals=None, dev_tables=None):
        o = real(frame, fit_mask, year, arrivals); o.loc[0, "dfc_ades_dep_1h_dev"] = np.nan; return o
    monkeypatch.setattr(D, "build_dfc", bad)
    with pytest.raises(AssertionError, match="dfc_ades_dep_1h_dev"):
        B.dfc_block(f, np.ones(len(f), bool), _stream(2025))


def test_dfc_block_refuses_a_finite_arrival_count_out_of_set(monkeypatch):
    f = _frame()
    real = D.build_dfc
    def bad(frame, fit_mask, year, arrivals=None, dev_tables=None):
        o = real(frame, fit_mask, year, arrivals); o["dfc_ades_arr_1h"] = 3.0; return o
    monkeypatch.setattr(D, "build_dfc", bad)
    with pytest.raises(AssertionError, match="arr_1h"):
        B.dfc_block(f, np.ones(len(f), bool), _stream(2025))


def test_regression_dfc3_smoke_dfc_block_without_an_arrival_stream():
    """2026-09-23: DFC.3 (dep columns only) calls dfc_block(..., arrivals=None). The builder then leaves every arrival
    count NaN (nothing observed), but the wrapper asserted the WITH-stream contract (finite where in_set == 1) and the
    DFC.3 smoke died on real data. No stream -> the arrival columns are NaN everywhere; departure contract unchanged."""
    f = _frame(); fit = np.zeros(len(f), bool); fit[::2] = True
    out = B.dfc_block(f, fit, None)
    assert (out.dfc_ades_in_set == 1).any()                                  # the failing shape: in-set rows exist
    for c in ("dfc_ades_arr_1h", "dfc_ades_arr_1h_dev"):
        assert np.isnan(out[c].to_numpy()).all()
    for c in B.DEP_ONLY:
        assert np.isfinite(out[c].to_numpy()).all()
    with_stream = B.dfc_block(f, fit, _stream(2025))
    for c in B.DEP_ONLY:                                                     # departure columns do not depend on the stream
        assert np.array_equal(out[c].to_numpy(), with_stream[c].to_numpy())


def test_dfc_block_without_a_stream_refuses_a_finite_arrival_count(monkeypatch):
    f = _frame()
    real = D.build_dfc
    def bad(frame, fit_mask, year, arrivals=None, dev_tables=None):
        o = real(frame, fit_mask, year, arrivals); o.loc[o.dfc_ades_in_set == 1, "dfc_ades_arr_1h"] = 3.0; return o
    monkeypatch.setattr(D, "build_dfc", bad)
    with pytest.raises(AssertionError, match="no arrival stream"):
        B.dfc_block(f, np.ones(len(f), bool), None)


def test_dfc_block_without_a_stream_still_refuses_nan_in_a_departure_column(monkeypatch):
    f = _frame()
    real = D.build_dfc
    def bad(frame, fit_mask, year, arrivals=None, dev_tables=None):
        o = real(frame, fit_mask, year, arrivals); o.loc[0, "dfc_ades_dep_1h"] = np.nan; return o
    monkeypatch.setattr(D, "build_dfc", bad)
    with pytest.raises(AssertionError, match="dfc_ades_dep_1h"):
        B.dfc_block(f, np.ones(len(f), bool), None)


def test_attach_writes_exactly_the_six_columns_in_place():
    f = _frame(); block = B.dfc_block(f, np.ones(len(f), bool), _stream(2025))
    before = set(f.columns)
    B.attach(f, block)
    assert set(f.columns) - before == set(D.DFC_FEATS)
    for c in D.DFC_FEATS:
        assert np.array_equal(f[c].to_numpy(), block[c].to_numpy(), equal_nan=True)
    with pytest.raises(ValueError, match=r"block length \d+ != frame length \d+"):   # OUR message, not pandas'
        B.attach(f, block.iloc[:-1])


# ---- fitting -------------------------------------------------------------------------------------------------
def test_fit_arm_learns_a_linear_signal_and_reports_importance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3)).astype(np.float32); label = 3 * X[:, 0] + rng.normal(scale=.1, size=600)
    fit = np.zeros(600, bool); fit[:400] = True; es = ~fit
    p = dict(objective="regression", learning_rate=0.3, num_leaves=7, min_data_in_leaf=5, verbosity=-1, num_threads=1, seed=0)
    raw, best, imp = B.fit_arm(X, label, fit, es, X[es], p, ["a", "b", "c"], max_rounds=100, patience=10)
    assert raw.shape == (200,) and best >= 1 and R.rmse(label[es], raw) < 1.0
    assert set(imp) == {"a", "b", "c"} and imp["a"] > imp["b"] and imp["a"] > imp["c"]


def test_fit_arm_refuses_non_finite_predictions(monkeypatch):
    import lightgbm as lgb
    class M:
        best_iteration = 3
        def predict(self, X, num_iteration=None): return np.full(len(X), np.nan)
        def feature_importance(self, kind): return np.zeros(2)
    monkeypatch.setattr(lgb, "train", lambda *a, **k: M())
    X = np.zeros((10, 2), np.float32); m = np.ones(10, bool)
    with pytest.raises(RuntimeError, match="non-finite"):
        B.fit_arm(X, np.zeros(10), m, m, X, {}, ["a", "b"])


def test_fit_arm_refuses_a_feature_name_width_mismatch():
    X = np.zeros((10, 2), np.float32); m = np.ones(10, bool)
    with pytest.raises(ValueError, match="feature names"):
        B.fit_arm(X, np.zeros(10), m, m, X, {}, ["a"])


def test_importance_share_is_the_named_columns_over_the_total():
    imp = {"a": 1.0, "b": 3.0, "dfc_x": 1.0}
    assert B.importance_share(imp, ["dfc_x"]) == pytest.approx(0.2)
    assert B.importance_share({"a": 0.0}, ["a"]) == 0.0


# ---- gate, predictions, T7, verdict ---------------------------------------------------------------------------
def _stage1(jul_raw, jul_m3, jan_raw, jan_m3, arr=(0.1, [-0.2, 0.4]), base_diff=0.1, share=0.01):
    def mk(r, m):
        return {"A6_vs_base": {"gain_s": r, "gain_minus_top3_s": m},
                "A6_vs_A7": {"gain_s": arr[0], "ci95": list(arr[1])},
                "base_rmse_minus_rcr2": base_diff, "dfc_importance_share_A6": share}
    return {"L-Jul": mk(jul_raw, jul_m3), "L-Jan": mk(jan_raw, jan_m3)}


@pytest.mark.parametrize("args, want", [((1, 1, 1, 1), True), ((0, 0, 0, 0), True), ((1, -0.1, 1, 1), False),
                                        ((-0.1, 1, 1, 1), False), ((1, 1, -0.1, 1), False), ((1, 1, 1, -0.1), False)])
def test_gate_g1_requires_raw_and_minus_top3_on_both_legs(args, want):
    assert B.gate_g1(_stage1(*args)) is want


def test_gate_g1_ignores_a7():
    s = _stage1(1, 1, 1, 1)
    s["L-Jul"]["A7_vs_base"] = {"gain_s": -9, "gain_minus_top3_s": -9}
    assert B.gate_g1(s) is True


def test_predictions_scored_stage1_hits_and_misses():
    p = B.predictions_scored(_stage1(0.5, 0.4, 0.3, 0.2), None)
    assert p == {"P1_LJul_band": "HIT", "P2_LJan_band": "HIT", "P3_diffuse": "HIT", "P4_g1_passes": "HIT",
                 "P5_arrival_contribution_null": "HIT", "P6_base_reproduces": "HIT", "P7_importance_share": "HIT"}
    p2 = B.predictions_scored(_stage1(2.0, 1.0, -1.0, -0.5, arr=(0.9, [0.1, 1.7]), base_diff=0.6, share=0.05), None)
    assert p2 == {"P1_LJul_band": "MISS", "P2_LJan_band": "MISS", "P3_diffuse": "MISS", "P4_g1_passes": "MISS",
                  "P5_arrival_contribution_null": "MISS", "P6_base_reproduces": "MISS", "P7_importance_share": "MISS"}
    assert B.predictions_scored(None, None) == {}
    # P5 has two clauses: a gain inside the band whose CI EXCLUDES zero is still a MISS
    assert B.predictions_scored(_stage1(0.5, 0.4, 0.3, 0.2, arr=(0.3, [0.1, 0.5])), None)["P5_arrival_contribution_null"] == "MISS"


def test_predictions_scored_p3_is_the_absolute_gap_on_both_legs():
    assert B.predictions_scored(_stage1(0.5, 0.85, 0.3, 0.2), None)["P3_diffuse"] == "MISS"
    assert B.predictions_scored(_stage1(0.5, 0.2, 0.3, 0.65), None)["P3_diffuse"] == "MISS"
    # a gap of -0.4 (minus-top-3 BELOW raw) is just as tail-driven: a signed comparison would call it a HIT
    assert B.predictions_scored(_stage1(0.5, 0.1, 0.3, 0.2), None)["P3_diffuse"] == "MISS"


def _comp(**kw):
    d = {"delta": 50.0, "ci95": [10.0, 90.0], "top1_share": 0.05}
    d.update(kw); return d


def test_predictions_scored_stage2_bands():
    s2 = {"composed": _comp(delta=100.0, ci95=[0.0, 200.0]), "f_alone": _comp(delta=200.0), "best_iter": 21101,
          "tripwires": {"pass": True}}
    assert B.predictions_scored(None, s2) == {"P9_bands": "HIT", "P10_best_iter": "HIT", "P11_tripwires": "HIT"}
    s3 = {"composed": _comp(delta=250.0, ci95=[0.0, 20.0]), "f_alone": _comp(delta=500.0), "best_iter": 30000,
          "tripwires": {"pass": False}}
    assert B.predictions_scored(None, s3) == {"P9_bands": "MISS", "P10_best_iter": "MISS", "P11_tripwires": "MISS"}


def _tw(**kw):
    d = dict(guard_raised_on_2025_stream=True, guard_raised_on_wrong_year=True,
             nan_share={"dfc_ades_dep_1h": 0.0, "dfc_ades_dep_3h": 0.0, "dfc_ades_dep_1h_dev": 0.0},
             out_of_range_share={c: 0.0001 for c in D.DFC_FEATS},
             mean_shift_in_sd={c: 0.05 for c in D.DFC_FEATS},
             in_set_share_2025=0.1500, in_set_share_2026=0.1497, unseen_ades_share=0.0005)
    d.update(kw); return d


def test_tripwire_eval_passes_the_audit_shape():
    t = B.tripwire_eval(_tw())
    assert t["pass"] is True and all(t["clauses"].values()) and set(t["clauses"]) == {"a", "b", "c", "d", "e"}


@pytest.mark.parametrize("kw, failing", [
    (dict(guard_raised_on_2025_stream=False), "a"), (dict(guard_raised_on_wrong_year=False), "a"),
    (dict(nan_share={"dfc_ades_dep_1h": 0.0, "dfc_ades_dep_3h": 0.0, "dfc_ades_dep_1h_dev": 0.001}), "b"),
    (dict(out_of_range_share={**{c: 0.0 for c in D.DFC_FEATS}, "dfc_ades_dep_3h": 0.0011}), "c"),
    (dict(mean_shift_in_sd={**{c: 0.0 for c in D.DFC_FEATS}, "dfc_ades_arr_1h": 0.26}), "c"),
    (dict(in_set_share_2026=0.1620), "d"), (dict(unseen_ades_share=0.0011), "e")])
def test_tripwire_eval_each_clause_fails_alone(kw, failing):
    t = B.tripwire_eval(_tw(**kw))
    assert t["pass"] is False
    assert [k for k, v in t["clauses"].items() if not v] == [failing]


def _s2_inputs():
    comp = {"delta": 80.0, "ci95": [20.0, 140.0], "per_month": {1: 30.0, 7: 50.0}, "delta_minus_top3": 70.0,
            "delta_ex_LFPG": 40.0, "ci_excludes_zero": True}
    fa = {"delta": 120.0, "ci95": [30.0, 210.0]}
    return comp, fa, 0.001, [220.0, 220.5, 219.8]


def test_stage2_verdict_working_only_when_t2_to_t7_hold():
    comp, fa, fl, seeds = _s2_inputs()
    v = B.stage2_verdict(comp, fa, fl, seeds, B.tripwire_eval(_tw()))
    assert v["verdict"] == "WORKING" and v["clauses"]["T7_tripwires"] is True
    v2 = B.stage2_verdict(comp, fa, fl, seeds, B.tripwire_eval(_tw(unseen_ades_share=0.5)))
    assert v2["verdict"] == "NOT WORKING" and v2["clauses"]["T7_tripwires"] is False
    assert all(v2["clauses"][k] for k in v2["clauses"] if k != "T7_tripwires")


def test_stage2_verdict_keeps_the_not_established_label():
    comp, fa, fl, seeds = _s2_inputs()
    comp = dict(comp, delta=5.0, ci95=[-10.0, 20.0], ci_excludes_zero=False)
    v = B.stage2_verdict(comp, fa, fl, seeds, B.tripwire_eval(_tw()))
    assert v["verdict"].startswith("NOT WORKING (not established")


# ---- guards and loading --------------------------------------------------------------------------------------
def test_guards_take_and_release_a_private_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("PHANTOM_HEAVY_LOCK_PATH", str(tmp_path / "lock"))
    fh, G = B._guards()
    with pytest.raises(RuntimeError, match="held by another job"):
        G.acquire_lock()
    G.release_lock(fh)
    fh2 = G.acquire_lock(); G.release_lock(fh2)


def test_arrival_stream_from_the_real_files_is_2025_and_ten_airports():
    """The real 2025 arrival stream (12 files): declared year asserted, the ten scored airports present."""
    a = B.load_arrivals_2025()
    assert a.years == frozenset({2025}) and set(D.SCORED_AIRPORTS) <= set(a.by_airport)
    with pytest.raises(AssertionError, match="F1 fail-loud"):
        D.load_arrivals(B.arrival_paths()[:1], year=2026)


@pytest.mark.slow
def test_load_frame_keeps_the_dfc_inputs_and_the_leg_columns():
    """The frame loader must keep INPUT_COLS (hr, tmin, mvt_sec are NOT among the 93) or the block cannot be built."""
    d = B.load_frame((1,), B.arm_feats("base"))
    for c in list(D.INPUT_COLS) + ["y", "delta", "proxy", "month", "doy", "ap"]:
        assert c in d.columns, c
    assert np.isfinite(d.delta.to_numpy(float)).all() and set(d.month.unique()) == {1}


# ---- Stage 2: fold A, the 2026 serve construction and the T7 measurement ---------------------------------------
def test_compose_delta_is_the_cbv_blend_with_max_proxy_minus_fd():
    fr = pd.DataFrame({"pop": [True, True, False], "taxi_C": [100.0, 200.0, 300.0], "taxi_v13_delta": [1.0, 2.0, 3.0],
                       "proxy": [400.0, 400.0, 400.0]})
    got = B.compose_delta(fr, np.array([10.0, 500.0, 0.0]))
    assert np.array_equal(got, [0.5 * (100 + 390), 0.5 * (200 + 1), 3.0])     # floor at 1 on the second row
    with pytest.raises(ValueError, match="Fd shape"):          # OUR check, before rcr2_build.compose's own
        B.compose_delta(fr, np.zeros(2))


def _blocks():
    rng = np.random.default_rng(3)
    def mk(n, shift=0.0, in_set_p=0.15):
        ins = (rng.random(n) < in_set_p).astype(float)
        arr = np.where(ins == 1, rng.normal(60 + shift, 20, n), np.nan)
        return pd.DataFrame({"dfc_ades_dep_1h": rng.integers(0, 10, n).astype(float), "dfc_ades_dep_3h": rng.integers(0, 30, n).astype(float),
                             "dfc_ades_dep_1h_dev": rng.normal(0, 2, n), "dfc_ades_in_set": ins, "dfc_ades_arr_1h": arr,
                             "dfc_ades_arr_1h_dev": arr - 60})[list(D.DFC_FEATS)]
    return mk(4000), mk(3000)


def test_tripwire_measure_computes_the_registered_quantities():
    b25, b26 = _blocks()
    m = B.tripwire_measure(b25, b26, ades_fit=np.array(["LFPG", "EGLL"]), ades26=np.array(["LFPG"] * 2999 + ["ZZZZ"]),
                           guards=(True, True))
    assert m["guard_raised_on_2025_stream"] is True and m["guard_raised_on_wrong_year"] is True
    assert set(m["nan_share"]) == set(B.DEP_ONLY) and all(v == 0.0 for v in m["nan_share"].values())
    assert set(m["out_of_range_share"]) == set(D.DFC_FEATS) and all(0 <= v <= 1 for v in m["out_of_range_share"].values())
    assert m["in_set_share_2025"] == pytest.approx(b25.dfc_ades_in_set.mean()) and m["in_set_share_2026"] == pytest.approx(b26.dfc_ades_in_set.mean())
    assert m["unseen_ades_share"] == pytest.approx(1 / 3000)
    assert abs(m["mean_shift_in_sd"]["dfc_ades_arr_1h"]) < 0.25      # same distribution -> small shift, NaN-aware
    assert m["n_2025"] == 4000 and m["n_2026"] == 3000


def test_tripwire_measure_sees_a_planted_range_excursion_and_nan():
    b25, b26 = _blocks()
    b26 = b26.copy(); b26.loc[:29, "dfc_ades_dep_3h"] = 999.0; b26.loc[0, "dfc_ades_dep_1h_dev"] = np.nan
    m = B.tripwire_measure(b25, b26, ades_fit=np.array(["LFPG"]), ades26=np.array(["LFPG"] * 3000), guards=(True, False))
    assert m["out_of_range_share"]["dfc_ades_dep_3h"] == pytest.approx(30 / 3000)
    assert m["nan_share"]["dfc_ades_dep_1h_dev"] == pytest.approx(1 / 3000)
    assert m["guard_raised_on_wrong_year"] is False
    assert B.tripwire_eval(m)["pass"] is False


def test_tripwire_measure_cols_restricts_the_range_and_shift_columns():
    b25, b26 = _blocks()
    m = B.tripwire_measure(b25, b26, ades_fit=np.array(["LFPG"]), ades26=np.array(["LFPG"] * 3000), guards=(True, True), cols=B.DEP_ONLY)
    assert set(m["out_of_range_share"]) == set(B.DEP_ONLY) and set(m["mean_shift_in_sd"]) == set(B.DEP_ONLY)
    full = B.tripwire_measure(b25, b26, ades_fit=np.array(["LFPG"]), ades26=np.array(["LFPG"] * 3000), guards=(True, True))
    assert set(full["out_of_range_share"]) == set(D.DFC_FEATS)


def test_tripwire_measure_mean_shift_is_in_2025_sd_units():
    b25, b26 = _blocks()
    b26 = b26.copy(); b26["dfc_ades_dep_1h"] = b26["dfc_ades_dep_1h"] + b25["dfc_ades_dep_1h"].std()
    m = B.tripwire_measure(b25, b26, ades_fit=np.array(["LFPG"]), ades26=np.array(["LFPG"] * 3000), guards=(True, True))
    assert m["mean_shift_in_sd"]["dfc_ades_dep_1h"] == pytest.approx(1.0, abs=0.1)


def test_dfc_block_passes_explicit_dev_tables_through_to_the_builder():
    """Tables fitted on HALF the rows: a wrapper that silently refitted on all rows would not reproduce them."""
    f = _frame(n=200, seed=1); half = np.zeros(len(f), bool); half[:100] = True
    tables = D.fit_dev_tables(f[list(D.INPUT_COLS)], half, 2025, _stream(2025))
    a = B.dfc_block(f, half, _stream(2025))
    b = B.dfc_block(f, None, _stream(2025), dev_tables=tables)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(b.dfc_ades_dep_1h_dev, B.dfc_block(f, np.ones(len(f), bool), _stream(2025)).dfc_ades_dep_1h_dev)


def test_serve_guards_fire_reports_both_f1_guards():
    """The tripwire (a) demonstration: the 2025 stream on a 2026 frame, and a 2025 `year` for a 2026 stream, both raise."""
    f26 = pd.DataFrame({"ADES_mvt": ["EGLL"], "doy": [15], "hr": [10], "tmin": [600], "mvt_sec": [0]})
    s25 = D.ArrivalStream(by_airport={"EGLL": np.array([1.0])}, years=frozenset({2025}), span=(1.0, 1.0))
    base26 = (2026 - D.EPOCH_YEAR) * 366.0 * 86400.0 + 15 * 86400.0
    s26 = D.ArrivalStream(by_airport={"EGLL": np.array([base26 + 36000.0])}, years=frozenset({2026}), span=(base26 + 36000.0, base26 + 36000.0))
    tables = D.fit_dev_tables(f26.assign(doy=15), fit_mask=np.ones(1, bool), year=2025)
    assert B.serve_guards_fire(f26, s25, s26, tables) == (True, True)
    assert B.serve_guards_fire(f26, s26, s26, tables) == (False, True)      # a same-year stream does NOT raise


def test_serve_guards_fire_only_counts_an_f1_assertion(monkeypatch):
    """Any OTHER AssertionError out of the builder is not a guard firing; reporting it as one would be a false pass."""
    monkeypatch.setattr(D, "build_dfc", lambda *a, **k: (_ for _ in ()).throw(AssertionError("block shape")))
    f26 = pd.DataFrame({"ADES_mvt": ["EGLL"], "doy": [15], "hr": [10], "tmin": [600], "mvt_sec": [0]})
    s = D.ArrivalStream(by_airport={}, years=frozenset({2025}), span=(np.nan, np.nan))
    assert B.serve_guards_fire(f26, s, s, {}) == (False, False)


@pytest.mark.slow
def test_serve_frame_2026_is_the_339551_serve_rows_with_a_consistent_clock():
    f = B.serve_frame_2026()
    assert len(f) == 339_551 and list(f.columns) == list(D.INPUT_COLS)
    assert np.array_equal(f.tmin.to_numpy() // 60, f.hr.to_numpy()) and f.hr.between(0, 23).all()
    assert set(f.doy.unique()) <= set(range(1, 32)) | set(range(182, 213)) and (f.ADES_mvt == "NA").sum() == 7


@pytest.mark.slow
def test_fold_a_frame_reproduces_lgbm_ab_counts_and_keeps_the_dfc_inputs():
    d, masks = B.fold_a_frame(B.arm_feats("base"))
    te, tr, fit, es = masks
    LF = B._load("lgbm_fold", "scripts/lgbm_fold.py")
    assert (int(te.sum()), int(fit.sum()), int(es.sum()), int(tr.sum())) == tuple(LF.LGBM_AB[k] for k in ("n_test", "n_fit", "n_es", "n_all"))
    for c in list(D.INPUT_COLS) + ["y", "delta", "proxy", "month", "ap"]:
        assert c in d.columns, c
    assert len(d) == 2_062_440


@pytest.mark.slow
def test_fold_a_frame_refuses_a_fold_whose_counts_differ_from_lgbm_ab(monkeypatch):
    LF = B._load("lgbm_fold", "scripts/lgbm_fold.py")
    monkeypatch.setitem(LF.LGBM_AB, "n_test", LF.LGBM_AB["n_test"] + 1)
    with pytest.raises(RuntimeError, match="not the same fold"):
        B.fold_a_frame(B.arm_feats("base"))
