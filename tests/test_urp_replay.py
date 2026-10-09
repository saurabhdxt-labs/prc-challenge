"""scripts/analysis/urp_replay.py -- URP, the fold replay of the shipped v33 unmatched chain
(plans/PREREG_urp_u_chain_replay_2026_09_24.md).

Unit tests use the SHIPPED pure functions (load_deps) with the fitters replaced by recording fakes, so the orchestration,
the lane routing, the rule precedence and the leak guard are tested on small synthetic frames; the real-data checks run in a
CHILD pytest process (the `isolated` pattern of tests/test_cln_build.py).
The negative invariant (registration, "Leakage rules"): in fold_a and lomo, no fitter ever receives a training row from a
held-out month -- asserted twice, once through train_view's own record and once from the fitters' inputs directly.

Manual mutation rehearsal 2026-09-24 (each mutant written over the new module, run with -x, then the saved copy restored
and its sha re-checked; no production file touched), 8 mutants, 8 KILLED, 0 escaped:
  M1 train_view: `bad = ... & held` -> `set()` (guard disabled)             -> KILLED (test_train_view_raises_on_held_month,
                                                                               test_guard_fires_when_a_fold_is_misconfigured)
  M2 folds: fold_a train months keep month 7                                -> KILLED (test_folds_each_scheme, negative invariant)
  M3 assemble_chain: v9 routed to the LIRF lane                              -> KILLED (test_assemble_chain_lane_routing)
  M4 lirf_tail: ARN applied over ARC-ruled rows                              -> KILLED (test_lirf_tail_precedence)
  M5 e2_apply: v13 comparison dropped (row_set(fired, v_now, v_now, sp))     -> KILLED (test_e2_apply_rows_and_day_check)
  M6 e2w_apply: S = all fired rows instead of the rows v31 changed           -> KILLED (test_e2w_apply_only_v31_rows)
  M7 nested_lirf_inputs: inner months keep t                                 -> KILLED (test_negative_invariant_...[fold_a], first to fail)
  M8 other_bodies: MJS patience 50 dropped                                   -> KILLED (test_other_bodies_arms_and_guard)
"""
from __future__ import annotations

import functools
import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import urp_replay as U  # noqa: E402
from isolation import run_guarded

CHILD_ENV = "URP_TEST_CHILD"
HAS_REAL = (ROOT / "data/cache_rome/unm.parquet").exists() and (ROOT / "data/raw/ranking.parquet").exists()
requires_real = pytest.mark.skipif(not HAS_REAL, reason="real 2025 caches / ranking file not present")


def isolated(fn):
    """Run a real-data test in a CHILD pytest process so its peak RSS never reaches this process."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if os.environ.get(CHILD_ENV) == "1":
            return fn(*args, **kwargs)
        node = f"{pathlib.Path(__file__).resolve()}::{fn.__name__}"
        r = run_guarded([sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", node],
                           env={**os.environ, CHILD_ENV: "1"}, capture_output=True, text=True, timeout=280)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
        assert "1 passed" in r.stdout, "the child did not run the test (skipped?): " + r.stdout[-2000:]
    return wrapper


@pytest.fixture(scope="module")
def real():
    return U.load_deps()


# ============================================================================================ tables and folds
def test_steps_table_versions_order_and_labels():
    assert tuple(s.version for s in U.STEPS) == U.CHAIN
    assert len(U.STEPS) == 13
    labels = {s.version: s.label for s in U.STEPS}
    assert [v for v, lab in labels.items() if lab == "OUT-OF-SAMPLE"] == ["v21"]
    assert all(s.fit in ("OOF", "none") for s in U.STEPS)
    assert U.LIRF_STEPS | U.OTHER_STEPS == set(U.CHAIN[1:]) and not (U.LIRF_STEPS & U.OTHER_STEPS)
    assert {s.version for s in U.STEPS if s.lane == U.LIRF} == U.LIRF_STEPS
    assert {s.version for s in U.STEPS if s.lane == "other"} == U.OTHER_STEPS


def test_step_label_branches_and_refusals():
    assert U.Step("x", "n", "LIRF", "none", "OOS").label == "OUT-OF-SAMPLE"
    assert U.Step("x", "n", "LIRF", "OOF", "IN").label == "IN-SAMPLE"
    with pytest.raises(ValueError, match="unknown fit tag 'LEAK'"):
        U.Step("x", "n", "LIRF", "LEAK", "OOS").label
    with pytest.raises(ValueError, match="unknown selection tag 'maybe'"):
        U.Step("x", "n", "LIRF", "OOF", "maybe").label


def test_folds_each_scheme():
    lomo = U.folds("lomo")
    assert lomo == [((1,), (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)), ((7,), (1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12))]
    assert U.folds("fold_a") == [((1, 7), (2, 3, 4, 5, 6, 8, 9, 10, 11, 12))]
    assert U.folds("serve") == [((), tuple(range(1, 13)))]
    for test, train in lomo + U.folds("fold_a"):
        assert not set(test) & set(train)
    with pytest.raises(ValueError, match=r"unknown scheme 'fold_b'; expected one of \('lomo', 'fold_a', 'serve'\)"):
        U.folds("fold_b")


def test_train_view_passes_clean_frame_unchanged():
    f = pd.DataFrame({"month": [2, 3, 12], "y": [1.0, 2.0, 3.0]})
    assert U.train_view(f, (1, 7), "x") is f
    assert U.train_view(f, (), "serve") is f


def test_train_view_raises_on_held_month():
    f = pd.DataFrame({"month": [2, 7, 7, 1], "y": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(U.LeakError, match=r"body rows: 3 training rows from held-out months \[1, 7\]"):
        U.train_view(f, (1, 7), "body rows")
    with pytest.raises(KeyError, match="no 'month' column"):
        U.train_view(f.drop(columns="month"), (1,), "x")


# ============================================================================================ overwrite / compare / output
def test_overwrite_values_mask_and_refusals():
    out, m = U.overwrite([10, 20, 30], [1, 2, 3], [3, 1], [33, 10])
    assert out.tolist() == [10.0, 20.0, 33.0] and m.tolist() == [False, False, True]   # an equal value is not "changed"
    with pytest.raises(ValueError, match="duplicate rule ids"):
        U.overwrite([1, 2], [1, 2], [1, 1], [5, 6])
    with pytest.raises(ValueError, match="1 rule ids are not among the rows"):
        U.overwrite([1, 2], [1, 2], [9], [5])
    with pytest.raises(ValueError, match="base has 2 values for 3 ids"):
        U.overwrite([1, 2], [1, 2, 3], [], [])
    with pytest.raises(ValueError, match="1 rule ids vs 2 rule values"):
        U.overwrite([1, 2], [1, 2], [1], [5, 6])


def test_compare_pass_fail_nan_and_shape():
    ok = U.compare([1.0, 2.0], [1.0, 2.0 + 1e-8], "a", 1e-6)
    assert ok["verdict"] == "PASS" and ok["n_diff"] == 0
    bad = U.compare([1.0, 5.0], [1.0, 2.0], "b", 0.0, ids=[11, 12])
    assert bad["verdict"] == "FAIL" and bad["n_diff"] == 1 and bad["first"] == {"row": 1, "id": 12.0, "got": 5.0, "want": 2.0}
    assert bad["max_abs"] == 3.0
    nan = U.compare([1.0, np.nan], [1.0, 2.0], "c", 1.0)
    assert nan["verdict"] == "FAIL" and nan["n_diff"] == 1
    assert U.compare([1.0], [1.0, 2.0], "d")["verdict"] == "FAIL"


def test_regression_bug_24_a10_compares_only_the_rows_where_e2_fires():
    """BC-24 instance (post-compute acceptance code first executed by the real run, 2026-09-24 17:43 EDT): acceptance_full
    NaN-masked the out-of-band rows on BOTH sides and compare() counts NaN as a mismatch, so the real run failed A10 with
    41 "diffs" whose finite max |d| was 3.3e-16. A10 as registered is "on the Jan/Jul LIRF rows where E2 fires": compare
    that subset. Out-of-band rows must not count; an in-band mismatch or an in-band NaN still must."""
    ids = [1.0, 2.0, 3.0, 4.0]
    p_e2 = np.array([0.4, np.nan, 0.7, np.nan])            # rows 2 and 4 are outside E2's band
    got = np.array([0.10, np.nan, 0.30, 0.99])
    ref = np.array([0.10, np.nan, 0.30 + 1e-12, 0.10])     # out-of-band values may differ or be NaN
    ok = U.a10_band_compare(got, ref, p_e2, ids)
    assert ok["verdict"] == "PASS" and ok["n"] == 2 and ok["n_diff"] == 0
    bad = U.a10_band_compare(np.array([0.10, np.nan, 0.31, 0.0]), ref, p_e2, ids)
    assert bad["verdict"] == "FAIL" and bad["first"]["id"] == 3.0
    nan_in_band = U.a10_band_compare(np.array([np.nan, 0.0, 0.30, 0.0]), ref, p_e2, ids)
    assert nan_in_band["verdict"] == "FAIL" and nan_in_band["n_diff"] == 1
    with pytest.raises(ValueError, match="A10: no rows where E2 fires"):
        U.a10_band_compare(got, ref, np.full(4, np.nan), ids)


def _output_frame(n=3):
    d = {"MVT_ID_mvt": np.arange(n, dtype=float), "y": np.full(n, 500.0), "month": np.ones(n, int), "doy": np.ones(n, int),
         "ap": ["EHAM"] * n, "cln_spent_day": [False] * n}
    for c in U.AFTER:
        d[c] = np.full(n, 600.0)
    d["pred"] = d["after_v33"].copy()
    return pd.DataFrame(d)


def test_check_output_schema_and_refusals():
    U.check_output(_output_frame())
    with pytest.raises(ValueError, match="columns"):
        U.check_output(_output_frame().drop(columns="doy"))
    f = _output_frame()
    f.loc[1, "MVT_ID_mvt"] = 0.0
    with pytest.raises(ValueError, match="duplicate MVT_ID_mvt"):
        U.check_output(f)
    for bad in (0.0, np.nan, 600.5):
        f = _output_frame()
        f.loc[0, "after_v10"] = bad
        with pytest.raises(ValueError, match="non-finite, below 1 s or not a whole second"):
            U.check_output(f)
    f = _output_frame()
    f.loc[0, "pred"] = 601.0
    with pytest.raises(ValueError, match="pred must equal after_v33"):
        U.check_output(f)


def test_write_output_refuses_overwrite_and_round_trips(tmp_path):
    out, side = tmp_path / "u.parquet", tmp_path / "u.json"
    f = _output_frame()
    U.write_output(f, {"status": "OK", "arr": np.array([1.0])}, out, side)
    pd.testing.assert_frame_equal(pd.read_parquet(out), f)
    assert '"output_sha256"' in side.read_text() and U.sha256(out) in side.read_text()
    with pytest.raises(FileExistsError):
        U.write_output(f, {}, out, tmp_path / "other.json")


def test_check_frozen_and_registered(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("hello")
    want = U.sha256(a)
    assert U.check_frozen(tmp_path, {"a.txt": want}) == {"a.txt": want}
    with pytest.raises(AssertionError, match=r"a.txt: sha 2cf24dba5fb0a30e != registered 0000000000000000"):
        U.check_frozen(tmp_path, {"a.txt": "0" * 64})
    log = tmp_path / "h.log"
    with pytest.raises(AssertionError, match="is not in h.log; hash it before any run"):
        U.check_registered(a, log)
    log.write_text(f"REGISTERED (full {want})\n")
    assert U.check_registered(a, log) == want


def test_guard_rss(monkeypatch):
    monkeypatch.setattr(U, "peak_rss_gib", lambda: 3.0)
    assert U.guard_rss(4.0, "s", log=lambda m: None) == 3.0
    with pytest.raises(MemoryError, match=r"s: peak RSS 3.00 GiB > 2.0 GiB"):
        U.guard_rss(2.0, "s", log=lambda m: None)


# ============================================================================================ chain pieces
def test_assemble_chain_lane_routing():
    L = np.array([True, False, True])
    lirf = {v: np.array([float(i), float(i) + 0.0]) * 0 + 100 * (k + 1) for k, (i, v) in
            enumerate((1, v) for v in sorted(U.LIRF_STEPS))}
    other = {v: np.array([1000.0 * (k + 1)]) for k, v in enumerate(sorted(U.OTHER_STEPS))}
    out = U.assemble_chain(L, [1.4, 2.6, 3.5], lirf, other)
    assert list(out) == list(U.AFTER)
    assert out["after_v4"].tolist() == [1.0, 3.0, 4.0]
    prev = out["after_v4"]
    for v in U.CHAIN[1:]:
        cur = out[f"after_{v}"]
        lane = L if v in U.LIRF_STEPS else ~L
        assert (cur[~lane] == prev[~lane]).all(), v             # a version never touches the other lane
        src = lirf if v in U.LIRF_STEPS else other
        assert (cur[lane] == src[v]).all(), v
        prev = cur
    with pytest.raises(ValueError, match="v7: 1 values for 2 lane rows"):
        U.assemble_chain(L, [1, 2, 3], {**lirf, "v7": np.array([5.0])}, other)
    with pytest.raises(ValueError, match="S1 has 2 rows, the lane mask 3"):
        U.assemble_chain(L, [1, 2], lirf, other)


def test_e2_apply_rows_and_day_check(real):
    # rows: 0 fires & unchanged; 1 fires but ARC changed it (v_now != v13); 2 does not fire; 3 fires but sp >= 24,000;
    # 4 fires on a second day whose move energy is huge -> reverted by the per-day check
    v13 = np.array([1000.0, 2000.0, 3000.0, 25_000.0, 1000.0])
    v_now = np.array([1000.0, 2500.0, 3000.0, 25_000.0, 1000.0])
    p = np.array([0.9, 0.9, np.nan, 0.9, 1.0])
    sp = np.array([5000.0, 5000.0, 5000.0, 25_000.0, 20_000.0])
    nf = np.full(5, 800.0)
    day = np.array(["d1", "d1", "d1", "d1", "d2"])
    # d1 moves row 0 by 3,580 s (37.2 per row), d2 moves row 4 by 19,000 s (1,046.8): p99 = 100 keeps d1, reverts d2
    new, S = U.e2_apply(real, v13, v_now, p, sp, nf, day, p99=100.0)
    assert S.tolist() == [True, False, False, False, False]
    assert new[0] == np.rint(0.9 * 5000 + 0.1 * 800) and new[1:].tolist() == v_now[1:].tolist()


def test_e2w_apply_only_v31_rows(real):
    v21 = np.array([1000.0, 1000.0, 1000.0])
    v31 = np.array([4000.0, 1000.0, 1000.0])          # only row 0 was changed by E2
    pw = np.array([0.5, 0.9, np.nan])                 # row 1 would fire under W, but v31 did not change it
    new, S = U.e2w_apply(real, v21, v31, pw, np.full(3, 5000.0), np.full(3, 1000.0), np.array(["d"] * 3), p99=1e9)
    assert S.tolist() == [True, False, False] and new.tolist() == [3000.0, 1000.0, 1000.0]
    with pytest.raises(ValueError, match="1 rows changed by v31 have no p_W"):
        U.e2w_apply(real, v21, v31, np.array([np.nan, 0.5, 0.5]), np.full(3, 5000.0), np.full(3, 1000.0),
                    np.array(["d"] * 3))
    _, S2 = U.e2w_apply(real, v21, v31, pw, np.full(3, 5000.0), np.full(3, 1000.0), np.array(["d"] * 3), p99=0.0)
    assert not S2.any()                                  # the per-day check reverts the day


def test_lirf_tail_precedence(real):
    ids = np.array([1.0, 2.0, 3.0])
    v12 = np.array([100.0, 200.0, 300.0])
    arc = (np.array([1.0]), np.array([111.0]))
    arn = (np.array([1.0, 2.0]), np.array([999.0, 222.0]))       # ARN on ARC's row 1 must be skipped
    nan = np.full(3, np.nan)
    t = U.lirf_tail(real, ids, v12, arc, arn, nan, nan, np.full(3, 5000.0), np.full(3, 800.0), np.array(["d"] * 3))
    assert t["v17"].tolist() == [111.0, 200.0, 300.0]
    assert t["v21"].tolist() == [111.0, 222.0, 300.0]
    assert t["v31"].tolist() == t["v21"].tolist() == t["v33"].tolist()
    assert (t["n_arc"], t["n_arn"], t["n_arn_skipped_arc"], t["n_e2"], t["n_e2w"]) == (1, 1, 1, 0, 0)


def test_compose_other_uses_shipped_mixture_hybrid_and_anchor(real):
    p, sp, cells = np.array([0.0, 0.5]), np.array([1000.0, 2000.0]), np.array([900.0, 5000.0])   # row 1 routes to the cells
    bodies = {k: np.array([700.0 + i, 700.0 + i]) for i, k in enumerate(("S1C", "MAU", "MXI", "MJS"))}
    A = np.array([500.0, np.nan])
    out = U.compose_other(real, p, sp, cells, bodies, A)
    assert out["v9"].tolist() == [700.0, 3500.0]
    assert out["v16"].tolist() == [900.0, 3500.0]                 # + max(A - 300, 0); NaN A -> no shift
    assert out["v24"].tolist() == [901.0, 3500.0] and out["v26"].tolist() == [903.0, 3500.0]


def test_arc_rows_with_shipped_rule_functions(real):
    t0 = pd.Timestamp("2025-01-10 10:00", tz="UTC")
    dep = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "PHASE_mvt": "DEP", "ADEP_mvt": "LIRF", "ADES_mvt": ["LFPG", "EGLL"],
                        "FLIGHT_mvt": ["AZA1", "AZA2"], "MVT_TIME_UTC_mvt": [t0 + pd.Timedelta(hours=2)] * 2,
                        "SCHED_TIME_UTC_mvt": [t0] * 2, "ADEP_flt": None, "CALLSIGN_flt": None, "LOBT_flt": pd.NaT,
                        "AOBT_3_flt": pd.NaT})
    arr = pd.DataFrame({"MVT_ID_mvt": [9.0], "PHASE_mvt": "ARR", "ADEP_mvt": None, "ADES_mvt": ["LFPG"], "FLIGHT_mvt": None,
                        "MVT_TIME_UTC_mvt": [t0 + pd.Timedelta(hours=4)], "SCHED_TIME_UTC_mvt": pd.NaT, "ADEP_flt": ["LIRF"],
                        "CALLSIGN_flt": ["AZA1"], "LOBT_flt": [t0 + pd.Timedelta(minutes=10)],
                        "AOBT_3_flt": [t0 + pd.Timedelta(minutes=100)]})
    raw = pd.concat([dep, arr], ignore_index=True)
    rule, ids, vals = U.arc_rows(real, raw, raw, [1.0, 2.0])
    assert rule.tolist() == ["ord", ""]
    assert ids.tolist() == [1.0] and vals.tolist() == [20.0 * 60]     # MVT - AOBT_3 of the arrival's record
    rule, ids, vals = U.arc_rows(real, raw, dep, [1.0, 2.0])
    assert rule.tolist() == ["", ""] and len(ids) == 0 and len(vals) == 0


# ============================================================================================ synthetic fold
def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    rows, mid = [], 0
    for m in range(1, 13):
        for ap in ("LIRF", "EHAM"):
            for k in range(6):
                mid += 1
                sched = pd.Timestamp(f"2025-{m:02d}-10 08:00", tz="UTC") + pd.Timedelta(hours=k)
                sp = float([900, 5000, 4000, 30_000, 2000, 4500][k])
                mvt = sched + pd.Timedelta(seconds=sp)
                y = float(sp if k in (1, 3) else 600 + 50 * k + rng.integers(0, 50))
                rows.append({"MVT_ID_mvt": float(mid), "month": m, "ADEP_mvt": ap, "ADES_mvt": "LFPG", "FLIGHT_mvt": f"F{mid}",
                             "STAND_mvt": f"S{k}", "sp": sp, "dayoff": int(k == 3), "y": y, "unmatched": True,
                             "SCHED_TIME_UTC_mvt": sched, "MVT_TIME_UTC_mvt": mvt,
                             "BLOCK_TIME_UTC_mvt": mvt - pd.Timedelta(seconds=y), "hprox_med": 900.0, "hprox_n": 9})
    unm = pd.DataFrame(rows)
    lirf = pd.DataFrame({"MVT_ID_mvt": np.arange(5000, 5024, dtype=float), "month": np.repeat(np.arange(1, 13), 2),
                         "ADEP_mvt": "LIRF", "y": 700.0})
    extra = pd.DataFrame({"MVT_ID_mvt": np.arange(9000, 9036, dtype=float), "month": np.repeat(np.arange(1, 13), 3),
                          "y": 650.0, "sp": 800.0})
    csf = unm[unm.ADEP_mvt == "LIRF"][["MVT_ID_mvt", "month"]].assign(
        cls=np.where(unm[unm.ADEP_mvt == "LIRF"].y == unm[unm.ADEP_mvt == "LIRF"].sp, "fill", "ordinary"), p_R=0.4, nf=800.0)

    def raw(m):
        d = unm[unm.month == m]
        dep = pd.DataFrame({"MVT_ID_mvt": d.MVT_ID_mvt, "PHASE_mvt": "DEP", "ADEP_mvt": d.ADEP_mvt, "ADES_mvt": d.ADES_mvt,
                            "FLIGHT_mvt": d.FLIGHT_mvt, "MVT_TIME_UTC_mvt": d.MVT_TIME_UTC_mvt,
                            "SCHED_TIME_UTC_mvt": d.SCHED_TIME_UTC_mvt, "ADEP_flt": None, "ADES_flt": None, "CALLSIGN_flt": None,
                            "LOBT_flt": pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]"),
                            "AOBT_3_flt": pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns, UTC]")})
        mat = dep.head(4).assign(MVT_ID_mvt=dep.head(4).MVT_ID_mvt + 100_000,
                                 AOBT_3_flt=dep.head(4).MVT_TIME_UTC_mvt - pd.Timedelta(minutes=20))
        return pd.concat([dep, mat], ignore_index=True)
    return SimpleNamespace(unm=unm, lirf=lirf, extra=extra, csf=csf, raw=raw)


class _Rec:
    """Records the months of every labelled frame a fitter receives."""

    def __init__(self):
        self.calls = []

    def note(self, who, frame):
        self.calls.append((who, frozenset(int(m) for m in frame.month.unique())))


def _spy_deps(real, rec: _Rec):
    d = SimpleNamespace(**vars(real))

    def fit_unmatched(train, test, train_matched=None, hybrid=False, seeds=(0,), parts=None):
        rec.note("fit_unmatched", train), rec.note("fit_unmatched.matched", train_matched)
        n = len(test)
        parts.update(p=np.full(n, 0.3), sp=test.sp.to_numpy(float), nf_cells=np.full(n, 1000.0), nf=np.full(n, 900.0),
                     nf_fit=np.full(n, 900.0), seeds=tuple(seeds))
        return np.maximum(0.3 * test.sp.to_numpy(float) + 0.7 * 900.0, 1.0)

    def fit_predict_fill(train, test, seed):
        rec.note("fit_predict_fill", train)
        return np.full(len(test), 0.4)

    def band_shares(train):
        rec.note("band_shares", train)
        return {"G1": 0.1, "G2": 0.6}, 1000.0

    class Ctrl:
        def __init__(self, nonfill, seed):
            rec.note("fit_congestion_regressor", nonfill)

        def predict(self, f):
            return np.full(len(f), 700.0)

        def design(self, f):
            return f[["sp", "month"]].reset_index(drop=True)

    def fit_body(x_fit, y_fit, dex, y_extra, seed, x_val, y_val, params):
        rec.note("fit_body.x_fit", x_fit), rec.note("fit_body.extra", dex), rec.note("fit_body.x_val", x_val)
        rec.calls.append(("params", frozenset(params.items())))
        return SimpleNamespace(predict=lambda X: np.full(len(X), 650.0 + params["max_iter"] / 100))

    def fit_brier_w(X, fill, w):
        rec.calls.append(("fit_brier_w", len(X)))
        return np.array(real.E2_BETA, float)

    def rule_E2(rows):
        sp = rows.sp.to_numpy(float)
        return np.where((sp >= 3600) & (sp <= 24_000), 0.7, np.nan)

    def stand_inbound(rows):
        return pd.DataFrame({"k_in": np.ones(len(rows)), "ta": 3000.0, "turn": 1.0}, index=rows.index)

    def arn_rule_rows(raw, src):
        return pd.DataFrame({"MVT_ID_mvt": pd.Series(dtype=float), "rule_value": pd.Series(dtype=float)})

    d.fit_unmatched, d.fit_predict_fill, d.band_shares = fit_unmatched, fit_predict_fill, band_shares
    d.fit_congestion_regressor, d.fit_body, d.fit_brier_w = Ctrl, fit_body, fit_brier_w
    d.rule_E2, d.stand_inbound, d.arn_rule_rows = rule_E2, stand_inbound, arn_rule_rows
    d.stopping_split = lambda n, seed: (np.arange(n)[: max(1, n - 2)], np.arange(n)[max(1, n - 2):])
    return d


def _run_fold(real, scheme, monkeypatch):
    rec, views = _Rec(), []
    orig = U.train_view

    def spy_view(frame, held, what):
        views.append((what, frozenset(int(m) for m in frame.month.unique()), frozenset(held)))
        return orig(frame, held, what)
    monkeypatch.setattr(U, "train_view", spy_view)
    deps, data = _spy_deps(real, rec), _synthetic()
    outs = [U.replay_fold(deps, data, te, tr, (0, 1), scheme, log=lambda m: None) for te, tr in U.folds(scheme)]
    return rec, views, outs, data


@pytest.mark.parametrize("scheme", ["fold_a", "lomo"])
def test_negative_invariant_no_fitter_sees_a_held_out_label(real, scheme, monkeypatch):
    """Registration, leakage rules: no step reads a held-out 2025 label. fold_a holds out {1, 7}; lomo holds out its month."""
    rec, views, outs, _ = _run_fold(real, scheme, monkeypatch)
    for (te, _tr), (out, _) in zip(U.folds(scheme), outs):
        assert set(out.month) == set(te)
    held_all = set(U.HOLDOUT) if scheme == "fold_a" else None
    for what, months, held in views:
        assert not (months & held), (what, months, held)
        if held_all is not None:
            assert held >= held_all, what                   # every view in fold_a holds out BOTH months (nested adds t)
    labelled = [c for c in rec.calls if c[0] not in ("params", "fit_brier_w", "fit_body.x_val")]
    assert labelled, "the spies recorded no fit"
    if scheme == "fold_a":
        for who, months in labelled:
            assert not (months & {1, 7}), (who, months)
    guarded = {w for w, _, _ in views}
    assert guarded >= {"S1 unmatched rows", "S1 LIRF matched rows", "R2 fill training rows", "date-slip shares",
                       "congestion body rows", "MAU extra matched rows", "E2W fit rows"}


def test_nested_inputs_exclude_inner_month(real, monkeypatch):
    rec, views, _, _ = _run_fold(real, "fold_a", monkeypatch)
    nested = [(m, h) for w, m, h in views if w == "R2 fill training rows" and len(h) == 3]
    assert len(nested) == 10                                   # one per training month
    for months, held in nested:
        (t,) = held - {1, 7}
        assert t not in months and not (months & {1, 7})


def test_replay_fold_output_and_chain(real, monkeypatch):
    _, _, outs, data = _run_fold(real, "fold_a", monkeypatch)
    out, extra = outs[0]
    assert len(out) == 24 and set(out.ap) == {"LIRF", "EHAM"} and out.MVT_ID_mvt.is_unique
    U.check_output(out[U.OUTPUT_COLUMNS])
    L = (out.ap == "LIRF").to_numpy()
    sp = data.unm.set_index("MVT_ID_mvt").sp.reindex(out.MVT_ID_mvt).to_numpy()
    # EHAM: cells 1,000 < T_TAIL -> the body; fake MAU body 650 + 200/100, MXI / MJS 660; EHAM has no matched row -> A NaN, no shift
    assert (out.after_v24[~L] == np.rint(0.3 * sp[~L] + 0.7 * 652.0)).all()
    assert (out.after_v26[~L] == np.rint(0.3 * sp[~L] + 0.7 * 660.0)).all()
    assert (out.after_v9[~L] == np.rint(0.3 * sp[~L] + 0.7 * 700.0)).all()
    assert (out.after_v24[L] == out.after_v21[L]).all() and (out.after_v9[L] == out.after_v7[L]).all()
    e2_rows = L & (out.after_v31 != out.after_v21).to_numpy()
    assert e2_rows.sum() == extra["info"]["n_e2"] > 0
    assert extra["info"]["n_arc"] == 0 and extra["info"]["test_months"] == [1, 7]


def test_other_bodies_arms_and_guard(real):
    rec = _Rec()
    deps, data = _spy_deps(real, rec), _synthetic()
    tr = data.unm[data.unm.month.isin([2, 3])]
    te = data.unm[data.unm.month == 1]
    b = U.other_bodies(deps, tr, te, data.extra[data.extra.month.isin([2, 3])], (0, 1), (1,))
    assert set(b) == {"S1C", "MAU", "MXI", "MJS"}
    params = [dict(p) for w, p in rec.calls if w == "params"]
    assert params[:3] == [{"max_leaf_nodes": 15, "max_iter": 200, "min_samples_leaf": 100},
                          {"max_leaf_nodes": 15, "max_iter": 1000, "min_samples_leaf": 100},
                          {"max_leaf_nodes": 15, "max_iter": 1000, "min_samples_leaf": 100, "n_iter_no_change": 50}]
    assert b["MAU"].tolist() == [652.0] * len(te) and b["S1C"].tolist() == [700.0] * len(te)
    with pytest.raises(U.LeakError, match="MAU extra matched rows"):
        U.other_bodies(deps, tr, te, data.extra, (0,), (1,))


def test_guard_fires_when_a_fold_is_misconfigured(real):
    deps, data = _spy_deps(real, _Rec()), _synthetic()
    with pytest.raises(U.LeakError, match=r"S1 unmatched rows: \d+ training rows from held-out months \[7\]"):
        U.replay_fold(deps, data, (1, 7), (2, 3, 7), (0,), "fold_a", log=lambda m: None)


def test_e2w_inputs_refuses_missing_label_class(real):
    deps, data = _spy_deps(real, _Rec()), _synthetic()
    drop = data.csf[data.csf.month == 2].MVT_ID_mvt.iloc[0]          # a TRAINING-month row (month 1 is held out)
    data.csf = data.csf[data.csf.MVT_ID_mvt != drop]
    with pytest.raises(ValueError, match="1 LIRF training rows have no csf2 label class"):
        U.e2w_inputs(deps, data, "lomo", (1,), tuple(range(2, 13)), (0,))


def test_fit_e2w_beta_refuses_empty_fit(real):
    t = pd.DataFrame({"x0": [1.0], "x1": [0.0], "x2": [0.0], "x3": [1.0], "month": [2], "fill": [True], "w": [1.0],
                      "fit": [False]})
    with pytest.raises(ValueError, match="E2W: no fit rows in the training view"):
        U.fit_e2w_beta(real, t, (1,))
    with pytest.raises(U.LeakError, match="E2W fit rows"):
        U.fit_e2w_beta(real, t.assign(month=1), (1,))


def test_main_requires_a_mode():
    with pytest.raises(SystemExit):
        U.main([])


# ============================================================================================ real data (child process)
@requires_real
@isolated
def test_real_frozen_inputs_and_row_counts():
    """The registered inputs, and the counts every downstream clause relies on: 22,219 2025 U rows, 5,321 in Jan+Jul, 397
    LIRF among them, 1,488 csf2 rows, and 5,290 / 383 scored 2026 U rows."""
    U.check_frozen()
    unm = pd.read_parquet(ROOT / "data/cache_rome/unm.parquet", columns=["MVT_ID_mvt", "month", "ADEP_mvt"])
    assert len(unm) == 22_219
    ja = unm[unm.month.isin(U.HOLDOUT)]
    assert len(ja) == 5_321 and int((ja.ADEP_mvt == "LIRF").sum()) == 397
    assert len(pd.read_parquet(ROOT / "reports/csf2/csf2_preds.parquet", columns=["MVT_ID_mvt"])) == 1_488
    r = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=["MVT_ID_mvt", "AOBT_3_flt", "ADEP_mvt"])
    t = pd.read_parquet(ROOT / "data/raw/submitting.parquet", columns=["MVT_ID_mvt"])
    s = r[r.MVT_ID_mvt.isin(set(t.MVT_ID_mvt))].drop_duplicates("MVT_ID_mvt")
    u = s[s.AOBT_3_flt.isna()]
    assert len(u) == 5_290 and int((u.ADEP_mvt == "LIRF").sum()) == 383


@requires_real
@isolated
def test_real_load_2025_frames_are_label_free_where_they_must_be():
    deps = U.load_deps()
    data = U.load_2025(deps, log=lambda m: None)
    assert len(data.extra) == 93_165 and data.extra.MVT_ID_mvt.is_unique
    raw = data.raw(1)
    assert "TAXITIME_SEC_mvt" not in raw.columns and "BLOCK_TIME_UTC_mvt" not in raw.columns
    assert set(deps.uc.WITNESS) <= set(data.unm.columns)
    # fit_unmatched keys its cells on derive()'s sp bins, which the cache drops (rome_fill.load_frames:349): the smoke's
    # first failure (KeyError 'sp_band'). Both frames must carry them, as prc/pipeline/ingest.load_rome_frames re-attaches them.
    for f in (data.unm, data.lirf):
        assert {"sp_band", "sp_fine"} <= set(f.columns)
    assert U.peak_rss_gib() < 4.0
