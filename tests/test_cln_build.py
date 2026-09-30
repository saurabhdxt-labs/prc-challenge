"""scripts/analysis/cln_build.py -- arm CLN of plans/PREREG_cln_clean_room_model_2026_09_23.md: a
clean-room LightGBM model (raw columns only, no pipeline stages/caches) scored against the full
pipeline's comparators on fold A.

Covers the registration's 8 harness checks (comparator reproduction, row counts, label-leak
invariance, serve availability, count-feature brute force, scorer sanity, vocabulary-from-training,
the early_stop breakage guard staying active) plus unit coverage of the time-diff arithmetic and the
verdict function's three branches, and that LightGBM recognises a category-dtype column natively.

Manual mutation rehearsal (scratch copy, never the real file), 6 mutants on count_features /
verdict_from_ci / score / fold_masks:
  M1 count_features: drop `- 1` self-exclusion in _grouped_counts        -> KILLED (test_count_features_vs_brute_force)
  M2 count_features: window `- window`/`+window` swapped (exclusive-ish) -> KILLED (test_count_features_window_edges_inclusive)
  M3 verdict_from_ci: `lo > 0` -> `lo >= 0`                               -> KILLED (test_verdict_from_ci_branches)
  M4 verdict_from_ci: `hi < -margin` -> `hi < margin`                    -> KILLED (test_verdict_from_ci_branches)
  M5 fold_masks: `> 0` -> `>= 0` on the label filter                     -> KILLED (test_fold_masks_label_and_phase_filter)
  M6 score: `(y - pipe) ** 2 - (y - clean) ** 2` sign flipped            -> KILLED (test_scorer_self_and_planted_noise)
All 6 mutants were killed by an existing test in this module; none escaped.
"""
from __future__ import annotations

import functools
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import cln_build as C  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
HAS_RAW = (RAW_DIR / "training_2025-01-01_2025-02-01.parquet").exists()
HAS_COMPARATORS = (ROOT / "reports" / "orl_fold_predictions.parquet").exists() and \
    (ROOT / "data" / "cache_stand" / "stratum_fold_v7_preds.parquet").exists()

requires_raw = pytest.mark.skipif(not HAS_RAW, reason="real raw/2025 parquet files are not present")
requires_comparators = pytest.mark.skipif(not HAS_COMPARATORS, reason="comparator artifacts are not present")
CHILD_ENV = "CLN_TEST_CHILD"


def isolated(fn):
    """Run a real-data test in a CHILD pytest process. Loading the 4.2 M-row raw frame in the shared pytest process raised
    its lifetime peak RSS (ru_maxrss) to ~5 GiB, and every later test that calls a CLI run() with an in-process
    `run_guards.start_rss_guard` (limit 2 GiB in ctxg1_v26_build) then read that inherited peak and os._exit(70)-ed the
    whole suite (2026-09-23, tree pass FAILED (70) right after this module). The child's memory never reaches the parent."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get(CHILD_ENV) == "1":
            return fn(*args, **kwargs)
        node = f"{pathlib.Path(__file__).resolve()}::{fn.__name__}"
        r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
                           env={**os.environ, CHILD_ENV: "1"}, capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    return wrapper


# ---- synthetic fixtures -----------------------------------------------------------------------
def _ts(base, offsets_s):
    return [base + pd.Timedelta(seconds=s) if s is not None else pd.NaT for s in offsets_s]


def synthetic_raw_frame(n_per_ap=6, seed=0):
    """A small, hand-built raw-shaped frame: 2 airports x DEP/ARR, enough rows to exercise the
    count features' window edges and self-exclusion, plus the 12 raw columns time-diffs read."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-03-01T00:00:00", tz="UTC")
    rows = []
    mvt_id = 0
    for ap in ("AAA", "BBB"):
        for phase in ("DEP", "ARR"):
            for i in range(n_per_ap):
                t = base + pd.Timedelta(minutes=10 * i)
                rows.append({
                    "MVT_ID_mvt": float(mvt_id), "FLIGHT_ID_mvt": float(mvt_id),
                    "FLIGHT_mvt": f"F{mvt_id}", "FLIGHT_RULE_mvt": "I", "ADEP_mvt": ap if phase == "DEP" else "ZZZ",
                    "ADES_mvt": "ZZZ" if phase == "DEP" else ap, "PHASE_mvt": phase, "MVT_TIME_UTC_mvt": t,
                    "BLOCK_TIME_UTC_mvt": t - pd.Timedelta(seconds=120),
                    "SCHED_TIME_UTC_mvt": t - pd.Timedelta(seconds=300),
                    "AIRCRAFT_TYPE_mvt": "A320", "RUNWAY_mvt": "09L" if ap == "AAA" else "27R",
                    "STAND_mvt": f"S{mvt_id % 3}", "TAXITIME_SEC_mvt": np.int32(600 + 10 * i),
                    "LOBT_flt": t - pd.Timedelta(seconds=900), "CALLSIGN_flt": f"CS{mvt_id}",
                    "ADEP_flt": ap if phase == "DEP" else "ZZZ", "ADES_flt": "ZZZ" if phase == "DEP" else ap,
                    "ADES_FILED_flt": "ZZZ", "MARKET_SEGMENT_flt": "PAX", "IOBT_flt": t - pd.Timedelta(seconds=800),
                    "FLIGHT_RULE_flt": "I", "FLIGHT_TYPE_flt": "S", "AIRCRAFT_TYPE_flt": "A320",
                    "WK_TBL_CAT_flt": "W1", "AIRCRAFT_OPERATOR_flt": "OP1", "EOBT_1_flt": t - pd.Timedelta(seconds=700),
                    "ARVT_1_flt": t + pd.Timedelta(seconds=1000), "AOBT_3_flt": t - pd.Timedelta(seconds=600),
                    "ARVT_3_flt": t + pd.Timedelta(seconds=1100), "month": 3,
                })
                mvt_id += 1
    return pd.DataFrame(rows)


def _brute_force_counts(frame):
    """O(n^2) reference for count_features, used only in the test."""
    t = pd.to_datetime(frame["MVT_TIME_UTC_mvt"], utc=True).to_numpy().astype("datetime64[s]").astype("int64")
    phase = frame["PHASE_mvt"].to_numpy()
    adep = frame["ADEP_mvt"].to_numpy()
    ades = frame["ADES_mvt"].to_numpy()
    rwy = frame["RUNWAY_mvt"].to_numpy()
    n = len(frame)
    out = {k: np.zeros(n, dtype=np.int64) for k in C.COUNT_FEATURES}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dt = abs(int(t[j]) - int(t[i]))
            if phase[j] == "DEP" and adep[j] == adep[i] and dt <= 1800:
                out["n_dep_ap_30"][i] += 1
            if phase[j] == "DEP" and adep[j] == adep[i] and dt <= 3600:
                out["n_dep_ap_60"][i] += 1
            if phase[j] == "DEP" and adep[j] == adep[i] and rwy[j] == rwy[i] and dt <= 1800:
                out["n_dep_rwy_30"][i] += 1
        for j in range(n):
            if phase[j] == "ARR" and ades[j] == adep[i] and abs(int(t[j]) - int(t[i])) <= 1800:
                out["n_arr_ap_30"][i] += 1
    return out


# ---- test 5: count features vs brute force, window edges + self-exclusion ----------------------
def test_count_features_vs_brute_force():
    frame = synthetic_raw_frame(n_per_ap=5)
    out = C.count_features(frame)
    ref = _brute_force_counts(frame)
    for k in C.COUNT_FEATURES:
        assert np.array_equal(out[k].to_numpy(), ref[k]), k


def test_count_features_window_edges_inclusive():
    """Two DEP rows exactly 1800s apart at the same airport: both counted (<=, not <)."""
    base = pd.Timestamp("2025-03-01T00:00:00", tz="UTC")
    frame = pd.DataFrame({
        "MVT_TIME_UTC_mvt": [base, base + pd.Timedelta(seconds=1800), base + pd.Timedelta(seconds=1801)],
        "ADEP_mvt": ["AAA", "AAA", "AAA"], "ADES_mvt": ["ZZZ", "ZZZ", "ZZZ"],
        "RUNWAY_mvt": ["09L", "09L", "09L"], "PHASE_mvt": ["DEP", "DEP", "DEP"],
    })
    out = C.count_features(frame)
    assert out["n_dep_ap_30"].tolist() == [1, 2, 1]     # row0-row1 within 1800 (edge); row2 (1801s) excluded from row0


def test_count_features_self_exclusion():
    frame = synthetic_raw_frame(n_per_ap=1)
    out = C.count_features(frame)
    # a lone DEP row at an airport (no siblings) must not count itself
    dep_rows = frame["PHASE_mvt"] == "DEP"
    assert (out.loc[dep_rows, "n_dep_ap_30"] == 0).all()


def test_count_features_refuses_missing_columns():
    with pytest.raises(ValueError):
        C.count_features(pd.DataFrame({"MVT_TIME_UTC_mvt": [pd.Timestamp.now(tz="UTC")]}))


# ---- test 3: label-leak invariance --------------------------------------------------------------
def test_build_features_refuses_banned_feature_list():
    frame = synthetic_raw_frame(n_per_ap=3)
    frame = C.count_features(frame)
    with pytest.raises(ValueError):
        C.build_features(frame, feats=["TAXITIME_SEC_mvt"])
    with pytest.raises(ValueError):
        C.build_features(frame, feats=["BLOCK_TIME_UTC_mvt"])


def test_build_features_bit_identical_after_permuting_banned_columns():
    frame = synthetic_raw_frame(n_per_ap=4)
    frame = C.count_features(frame)
    X0, vocab = C.build_features(frame)

    frame_perm = frame.copy()
    rng = np.random.default_rng(1)
    frame_perm["TAXITIME_SEC_mvt"] = rng.permutation(frame_perm["TAXITIME_SEC_mvt"].to_numpy())
    frame_perm["BLOCK_TIME_UTC_mvt"] = rng.permutation(frame_perm["BLOCK_TIME_UTC_mvt"].to_numpy())
    X1, _ = C.build_features(frame_perm, vocab=vocab)
    pd.testing.assert_frame_equal(X0, X1)

    frame_null = frame.copy()
    frame_null["TAXITIME_SEC_mvt"] = np.nan
    frame_null["BLOCK_TIME_UTC_mvt"] = pd.NaT
    X2, _ = C.build_features(frame_null, vocab=vocab)
    pd.testing.assert_frame_equal(X0, X2)


def test_build_features_never_reads_banned_columns_even_if_absent():
    frame = synthetic_raw_frame(n_per_ap=3)
    frame = C.count_features(frame)
    frame = frame.drop(columns=["TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt"])
    X, _ = C.build_features(frame)   # must not raise: those columns are never read
    assert list(X.columns) == C.FEATURE_ORDER


def test_build_features_returns_exactly_37_columns_in_order():
    frame = synthetic_raw_frame(n_per_ap=3)
    frame = C.count_features(frame)
    X, _ = C.build_features(frame)
    assert list(X.columns) == C.FEATURE_ORDER
    assert X.shape[1] == 37


# ---- test 7: vocabulary from training rows only --------------------------------------------------
def test_vocab_from_training_rows_only_holdout_level_is_nan():
    frame = synthetic_raw_frame(n_per_ap=4)
    frame = C.count_features(frame)
    train = frame.iloc[:8].copy()
    holdout = frame.iloc[8:].copy()
    holdout = holdout.copy()
    holdout.loc[holdout.index[0], "AIRCRAFT_TYPE_mvt"] = "UNSEEN_TYPE"
    vocab = C.fit_vocab(train)
    assert "UNSEEN_TYPE" not in vocab["AIRCRAFT_TYPE_mvt"]
    X, _ = C.build_features(holdout, vocab=vocab)
    assert pd.isna(X.loc[holdout.index[0], "AIRCRAFT_TYPE_mvt"])
    assert X["AIRCRAFT_TYPE_mvt"].dtype.name == "category"


def test_lightgbm_recognises_category_dtype_natively():
    lgb = pytest.importorskip("lightgbm")
    frame = synthetic_raw_frame(n_per_ap=20)
    frame = C.count_features(frame)
    vocab = C.fit_vocab(frame)
    X, _ = C.build_features(frame, vocab=vocab)
    y = frame["TAXITIME_SEC_mvt"].astype("float64").to_numpy()
    ds = lgb.Dataset(X, y)
    b = lgb.train({"objective": "regression", "verbosity": -1, "num_leaves": 7}, ds, num_boost_round=5)
    info = b.dump_model()
    cat_names = set(info.get("pandas_categorical_names", info.get("feature_names", [])))
    # LightGBM records which features it treated as categorical either via feature_infos
    # (non-numeric "categories" entries) or via pandas_categorical
    feature_infos = info.get("feature_infos", {})
    treated_cat = any(v == "" or isinstance(v, list) for v in feature_infos.values()) or bool(info.get("pandas_categorical"))
    assert treated_cat or b.pandas_categorical not in (None, [])


# ---- time-diff arithmetic, hand-built row --------------------------------------------------------
def test_time_diff_seconds_hand_built_row():
    base = pd.Timestamp("2025-03-01T12:00:00", tz="UTC")
    frame = pd.DataFrame({
        "MVT_TIME_UTC_mvt": [base], "SCHED_TIME_UTC_mvt": [base - pd.Timedelta(seconds=300)],
        "AOBT_3_flt": [base - pd.Timedelta(seconds=600)], "LOBT_flt": [base - pd.Timedelta(seconds=900)],
        "IOBT_flt": [base - pd.Timedelta(seconds=800)], "EOBT_1_flt": [base - pd.Timedelta(seconds=700)],
        "ARVT_3_flt": [base + pd.Timedelta(seconds=1100)], "ARVT_1_flt": [base + pd.Timedelta(seconds=1000)],
        "ADEP_mvt": ["AAA"], "ADES_mvt": ["ZZZ"], "FLIGHT_RULE_mvt": ["I"], "AIRCRAFT_TYPE_mvt": ["A320"],
        "RUNWAY_mvt": ["09L"], "STAND_mvt": ["S1"], "FLIGHT_mvt": ["F1"], "CALLSIGN_flt": ["C1"],
        "ADEP_flt": ["AAA"], "ADES_flt": ["ZZZ"], "ADES_FILED_flt": ["ZZZ"], "MARKET_SEGMENT_flt": ["PAX"],
        "FLIGHT_RULE_flt": ["I"], "FLIGHT_TYPE_flt": ["S"], "AIRCRAFT_TYPE_flt": ["A320"],
        "WK_TBL_CAT_flt": ["W1"], "AIRCRAFT_OPERATOR_flt": ["OP1"],
        "n_dep_ap_30": [0], "n_dep_ap_60": [0], "n_dep_rwy_30": [0], "n_arr_ap_30": [0],
    })
    X, _ = C.build_features(frame)
    assert X["d_mvt_sched"].iloc[0] == pytest.approx(300.0)
    assert X["d_mvt_aobt3"].iloc[0] == pytest.approx(600.0)
    assert X["d_mvt_lobt"].iloc[0] == pytest.approx(900.0)
    assert X["d_mvt_iobt"].iloc[0] == pytest.approx(800.0)
    assert X["d_mvt_eobt1"].iloc[0] == pytest.approx(700.0)
    assert X["d_aobt3_sched"].iloc[0] == pytest.approx(-300.0)
    assert X["d_aobt3_eobt1"].iloc[0] == pytest.approx(100.0)
    assert X["d_eobt1_lobt"].iloc[0] == pytest.approx(200.0)
    assert X["d_iobt_lobt"].iloc[0] == pytest.approx(100.0)
    assert X["d_arvt3_mvt"].iloc[0] == pytest.approx(1100.0)
    assert X["d_arvt1_mvt"].iloc[0] == pytest.approx(1000.0)
    assert X["d_arvt3_arvt1"].iloc[0] == pytest.approx(100.0)
    assert X["mvt_hr"].iloc[0] == 12
    assert X["mvt_min"].iloc[0] == 12 * 60
    assert X["sched_hr"].iloc[0] == 11


def test_time_diff_is_nan_when_operand_missing():
    base = pd.Timestamp("2025-03-01T12:00:00", tz="UTC")
    frame = pd.DataFrame({
        "MVT_TIME_UTC_mvt": [base], "SCHED_TIME_UTC_mvt": [pd.NaT],
        "AOBT_3_flt": [pd.NaT], "LOBT_flt": [pd.NaT], "IOBT_flt": [pd.NaT], "EOBT_1_flt": [pd.NaT],
        "ARVT_3_flt": [pd.NaT], "ARVT_1_flt": [pd.NaT],
        "ADEP_mvt": ["AAA"], "ADES_mvt": ["ZZZ"], "FLIGHT_RULE_mvt": ["I"], "AIRCRAFT_TYPE_mvt": ["A320"],
        "RUNWAY_mvt": ["09L"], "STAND_mvt": ["S1"], "FLIGHT_mvt": ["F1"], "CALLSIGN_flt": ["C1"],
        "ADEP_flt": ["AAA"], "ADES_flt": ["ZZZ"], "ADES_FILED_flt": ["ZZZ"], "MARKET_SEGMENT_flt": ["PAX"],
        "FLIGHT_RULE_flt": ["I"], "FLIGHT_TYPE_flt": ["S"], "AIRCRAFT_TYPE_flt": ["A320"],
        "WK_TBL_CAT_flt": ["W1"], "AIRCRAFT_OPERATOR_flt": ["OP1"],
        "n_dep_ap_30": [0], "n_dep_ap_60": [0], "n_dep_rwy_30": [0], "n_arr_ap_30": [0],
    })
    X, _ = C.build_features(frame)
    for name in C.TIME_DIFFS:
        assert pd.isna(X[name].iloc[0]), name


# ---- fold masks (row filters) ---------------------------------------------------------------------
def test_fold_masks_label_and_phase_filter():
    frame = pd.DataFrame({
        "PHASE_mvt": ["DEP", "DEP", "ARR", "DEP"],
        "TAXITIME_SEC_mvt": [600, 0, 500, np.nan],
        "AOBT_3_flt": [pd.Timestamp.now(tz="UTC"), pd.NaT, pd.Timestamp.now(tz="UTC"), pd.Timestamp.now(tz="UTC")],
        "month": [2, 2, 2, 2],
    })
    m = C.fold_masks(frame)
    assert m["base"].tolist() == [True, False, False, False]     # ARR excluded, label<=0 excluded, NaN label excluded
    assert m["matched"].tolist() == [True, False, True, True]


def test_fold_masks_month_membership():
    frame = pd.DataFrame({
        "PHASE_mvt": ["DEP"] * 3, "TAXITIME_SEC_mvt": [600, 600, 600],
        "AOBT_3_flt": [pd.Timestamp.now(tz="UTC")] * 3, "month": [1, 2, 3],
    })
    m = C.fold_masks(frame)
    assert m["holdout"].tolist() == [True, False, False]
    assert m["fit"].tolist() == [False, True, False]
    assert m["stop"].tolist() == [False, False, True]


@requires_raw
@isolated
def test_check_counts_on_real_data():
    got = C.check_counts()
    assert got == {"fit_matched": C.N_FIT_MATCHED, "stop_matched": C.N_STOP_MATCHED,
                   "holdout_matched": C.N_HOLD_MATCHED, "fit_all": C.N_FIT_ALL, "stop_all": C.N_STOP_ALL,
                   "holdout_unmatched": C.N_HOLD_UNMATCHED}


# ---- test 4: serve availability -------------------------------------------------------------------
@requires_raw
@isolated
def test_serve_availability_ranking_sample():
    """The builder runs unchanged on ranking's DEP rows (BLOCK/TAXI null there) and returns the same
    37 columns. Non-null-rate parity is checked on the RAW column values (the registration's own
    audit: "flight-record columns are 1.5% null in ranking vs 2.1% null in training", `PHASE_mvt`
    absent), not on the post-vocabulary categorical codes -- a high-cardinality per-flight identifier
    such as FLIGHT_mvt legitimately has much lower TRAIN-vocabulary overlap on a single serve day
    than a raw null-rate comparison would suggest, and that is expected, not a builder defect."""
    ranking = pd.read_parquet(ROOT / "data" / "raw" / "ranking.parquet")
    dep = ranking[ranking["PHASE_mvt"] == "DEP"].copy()
    sample = pd.concat([dep.iloc[:2000], ranking[ranking["PHASE_mvt"] == "ARR"].iloc[:2000]], ignore_index=True)
    sample = C.count_features(sample)
    dep_sample = sample[sample["PHASE_mvt"] == "DEP"].reset_index(drop=True)

    train = C.load_raw(list(C.FIT_MONTHS) + list(C.STOP_MONTHS))
    m = C.fold_masks(train)
    train_dep = train[m["base"]].reset_index(drop=True)
    train_dep = C.count_features(train_dep)

    raw_cols = sorted({c for pair in C.TIME_DIFFS.values() for c in pair} | set(C.CATEGORICALS))
    for col in raw_cols:
        r_train = train_dep[col].notna().mean()
        r_serve = dep_sample[col].notna().mean()
        assert abs(r_train - r_serve) <= 0.02, (col, r_train, r_serve)

    vocab = C.fit_vocab(train_dep)
    X_train, _ = C.build_features(train_dep, vocab=vocab)
    X_serve, _ = C.build_features(dep_sample, vocab=vocab)
    assert list(X_serve.columns) == C.FEATURE_ORDER == list(X_train.columns)

    # non-categorical features are not subject to vocabulary-unseen NaN and keep the 2pp bound
    for col in C.CLOCK_FEATURES + list(C.TIME_DIFFS) + C.COUNT_FEATURES:
        r_train = X_train[col].notna().mean()
        r_serve = X_serve[col].notna().mean()
        assert abs(r_train - r_serve) <= 0.02, (col, r_train, r_serve)


# ---- test 1: comparator reproduction -----------------------------------------------------------
@requires_comparators
@isolated
def test_load_comparators_reproduces_registered_numbers():
    comp = C.load_comparators()
    lane_m, lane_u = comp["M"], comp["U"]
    assert len(lane_m) == C.N_LANE_M
    assert len(lane_u) == C.N_LANE_U
    rmse_m = np.sqrt(((lane_m["y"] - lane_m["pipe"]) ** 2).mean())
    assert rmse_m == pytest.approx(C.PIPE_M_RMSE, abs=1e-4)
    sse_m = ((lane_m["y"] - lane_m["pipe"]) ** 2).sum()
    assert sse_m / C.BOARD_N == pytest.approx(C.PIPE_M_SSE_344841, abs=0.05)
    sse_u = ((lane_u["y"] - lane_u["pipe"]) ** 2).sum()
    assert sse_u / C.BOARD_N == pytest.approx(C.PIPE_U_SSE_344841, abs=0.05)


# ---- test 6: scorer sanity --------------------------------------------------------------------
def test_scorer_self_and_planted_noise():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.normal(600, 200, n)
    pipe = y + rng.normal(0, 50, n)
    month = rng.integers(1, 3, n)
    doy = rng.integers(1, 60, n)

    st_self = C.score(y, pipe, pipe, month, doy)
    assert st_self["G"] == 0.0

    noisy = pipe + rng.normal(0, 200, n)
    st_noise = C.score(y, pipe, noisy, month, doy)
    assert st_noise["G"] < 0
    assert st_noise["ci95"][1] < 0


def test_verdict_from_ci_branches():
    assert C.verdict_from_ci((1.0, 5.0), margin=10.0) == "TRUE"
    assert C.verdict_from_ci((-1.0, -20.0 - 1.0), margin=10.0) == "FALSE"     # hi < -margin
    assert C.verdict_from_ci((-1.0, 1.0), margin=10.0) == "NEAR"
    assert C.verdict_from_ci((0.0, 1.0), margin=10.0) == "NEAR"              # lower == 0 is not > 0


# ---- test 8: the early_stop breakage guard stays active ----------------------------------------
@pytest.fixture
def fake_run_env(monkeypatch, tmp_path):
    """run() on a synthetic 12-month frame with early_stop / fit_seeds faked; `state["peaks"]` feeds peak_rss in order."""
    state = {"calls": {}, "peaks": None}
    real_load = C._load

    def fake_load(name, rel):
        if name == "run_guards":
            G = real_load("run_guards", "scripts/run_guards.py")
            # never touch the machine-wide heavy lock from a unit test (Phantom holds it at times; 2026-09-24 it did).
            # monkeypatch, never plain assignment: the module object is shared, and a plain assignment leaked these
            # stubs into tests/test_run_guards.py and three *_build lock tests (5 suite failures, 2026-09-24).
            monkeypatch.setattr(G, "acquire_lock", lambda *a, **k: None)
            monkeypatch.setattr(G, "release_lock", lambda fh: None)
            if state["peaks"] is not None:
                seq = iter(state["peaks"])
                last = [0]

                def peak_rss():
                    last[0] = next(seq, last[0])
                    return last[0]
                monkeypatch.setattr(G, "peak_rss", peak_rss)
            return G
        if name == "lgbm_submit":
            mod = real_load("lgbm_submit", "scripts/lgbm_submit.py")

            def fake_early_stop(*a, **k):
                state["calls"]["early_stop"] = True
                return {"best_iter": 5, "n_ref": 5, "n_fit": 1, "n_es": 1, "n_es_months": 1, "n_all": 1,
                        "es_months": [3, 9], "target": k.get("target", "delta"), "es_rmse_taxi_time_NOT_A_RESULT": 0.0,
                        "es_proxy_only_rmse": 1.0, "es_guard_rows": 1}

            def fake_fit_seeds(X, label, train, X_pred, params, n_ref, seeds, **k):
                state["calls"]["fit_seeds"] = True
                return {s: np.ones(int(np.asarray(X_pred).shape[0]) if hasattr(X_pred, "shape") else len(X_pred))
                        for s in seeds}
            monkeypatch.setattr(mod, "early_stop", fake_early_stop)
            monkeypatch.setattr(mod, "fit_seeds", fake_fit_seeds)
            return mod
        return real_load(name, rel)

    monkeypatch.setattr(C, "_load", fake_load)
    monkeypatch.setattr(C, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(C, "SMOKE_DIR", tmp_path / "smoke")
    frame = synthetic_raw_frame(n_per_ap=30)
    months = list(C.FIT_MONTHS) + list(C.STOP_MONTHS) + list(C.HOLDOUT_MONTHS)
    frame["month"] = [months[i % len(months)] for i in range(len(frame))]
    dep = frame["PHASE_mvt"] == "DEP"
    frame.loc[dep, "AOBT_3_flt"] = frame.loc[dep, "MVT_TIME_UTC_mvt"] - pd.Timedelta(seconds=600)

    def fake_build_lgbm_frame(months_arg, root=C.ROOT):
        raw = C.count_features(frame.copy())
        return raw, C.fold_masks(raw)

    monkeypatch.setattr(C, "_build_lgbm_frame", fake_build_lgbm_frame)
    state["tmp"] = tmp_path
    return state


def test_run_path_calls_early_stop(fake_run_env):
    """run() must call lgbm_submit.early_stop (never bypass its proxy-only breakage guard)."""
    C.run("CLN-D", seeds=(0,), smoke=True)
    assert fake_run_env["calls"].get("early_stop") is True
    assert fake_run_env["calls"].get("fit_seeds") is True


def test_regression_post_fit_rss_over_limit_keeps_the_predictions(fake_run_env):
    """2026-09-23 real CLN-D: all 3 seeds finished, then a post-fit peak RSS of 8.38 GiB > the 8 GiB limit raised and threw
    the predictions away. A post-fit check cannot prevent anything; the overrun is recorded, the work is kept (the
    in-flight guard is the external watchdog)."""
    fake_run_env["peaks"] = [1 * 1024 ** 3, 9 * 1024 ** 3]
    rep = C.run("CLN-D", seeds=(0,), smoke=True)
    assert (fake_run_env["tmp"] / "smoke" / "cln_CLN-D_preds.parquet").exists()
    assert rep["rss_over_limit_after_fit"] is True


def test_pre_fit_rss_over_limit_still_aborts(fake_run_env):
    fake_run_env["peaks"] = [9 * 1024 ** 3]
    with pytest.raises(RuntimeError, match="before any fit"):
        C.run("CLN-D", seeds=(0,), smoke=True)
    assert fake_run_env["calls"].get("early_stop") is None


@requires_raw
def test_regression_real_data_tests_do_not_raise_the_pytest_peak_rss():
    """The suite crash (exit 70, 2026-09-23): the real-data tests must not raise this process's peak RSS. Calling one of them
    here must leave ru_maxrss within 300 MiB of where it was (in-process it adds > 1 GiB)."""
    if os.environ.get(CHILD_ENV) == "1":
        pytest.skip("runs only in the parent process (the child is where the load happens, by design)")
    import run_guards as G
    before = G.peak_rss()
    test_check_counts_on_real_data()
    assert G.peak_rss() - before < 300 * 1024 ** 2

