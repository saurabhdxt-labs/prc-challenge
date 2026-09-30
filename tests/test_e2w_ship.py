"""Tests for scripts/analysis/e2w_ship.py (plans/PREREG_e2w_ship_v33_2026_09_24.md): v33 = v31 with the W refit on v31's rows.

Red run (2026-09-24, before e2w_ship.py existed): `python3.11 -B -m pytest tests/test_e2w_ship.py -p no:randomly -q` ->
collection ERROR, FileNotFoundError on scripts/analysis/e2w_ship.py (0 passed). Green after the build: see the BUILT section of the
registration for counts.

Manual mutation rehearsal (2026-09-24, copy mirror of e2w_ship.py, one mutant at a time, each seen caught):
  M1 e2_form: drop the `k_in.notna()` term of the fire mask         -> test_e2_form_equals_the_frozen_rule_with_e2_coefficients RED
  M2 e2_form: `sp <= SP_HI` -> `sp < SP_HI`                          -> test_e2_form_band_edges_are_inclusive RED
  M3 k_ratio: numerator D*D instead of D*e                           -> test_k_ratio_is_the_projection_of_e_on_d RED
  M4 k_day_ci: resample rows instead of day blocks                   -> test_k_day_ci_resamples_whole_days RED (first
     ESCAPED on a two-day fixture where both bootstraps give (0, 1); fixture replaced by 12 random days, then CAUGHT)
  M5 assemble: write values for every id (drop the id mask)          -> test_assemble_changes_only_the_named_ids RED
  M6 check_rows: skip the ranking-DEP id-set comparison              -> test_check_rows_refuses_ids_that_are_not_ranking_dep RED
  M7 evidence_from_rows: fit W on all rows (ignore fit_mask)         -> test_evidence_from_rows_fits_w_on_fit_rows_only RED
  M8 serve_rows_2026: keep the DEP label columns                     -> test_serve_rows_2026_drops_dep_labels RED
Recorded here per the repo convention; none equivalent.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RULE_PATH = ROOT / "data/rome_hunt_2026_09_23/agent_E_rules.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = _load("e2w_ship", ROOT / "scripts/analysis/e2w_ship.py")
W = S.W
T = pd.Timestamp
BOARD_N = 344_841.0


def _mix(p, sp, nf):
    return np.maximum(p * sp + (1 - p) * nf, 1.0)


# ---------------------------------------------------------------- pinned identities

def test_pins_and_key():
    assert S.RULE_SHA == "bd01f4ed927d473585b6a182192073c59ed75fa220101db34035343c7629b606"
    assert S.V31_SHA == "c1b65a22907d0aa7cc066734beebdd39c8ea47f7ccc71f0fccf59dfd7295fac5"
    assert S.V31_MAN_SHA == "d5a0e01268911a83d84d25b64634f6d0e0dae83b21cc0ca78d11f5639d609839"
    assert S.REPORT_SHA == "fca3ebeb0cf776222337a28c3e78f5e1df4ab5b1061cf325e3103f0d8d5e2993"
    assert S.KEY == "merry-quicksand_v33.parquet"
    assert S.N_S == 272 and S.K_PT == pytest.approx(0.8453611792929924, abs=1e-15)
    assert S.OUT_FINAL.name == S.KEY and S.OUT_CAND.name == "merry-quicksand_v33_e2w.parquet"


@pytest.mark.skipif(not (RULE_PATH.exists() and S.V31.exists() and S.V31_MAN.exists() and S.REPORT.exists()),
                    reason="local artifacts absent")
def test_pinned_files_on_disk_match():
    assert S.B.sha256(RULE_PATH) == S.RULE_SHA
    assert S.B.sha256(S.V31) == S.V31_SHA
    assert S.B.sha256(S.V31_MAN) == S.V31_MAN_SHA
    assert S.B.sha256(S.REPORT) == S.REPORT_SHA


# ---------------------------------------------------------------- e2_form (the E2 form with any coefficients)

def _raw_world(tmp_path, arrivals, fname="training_2025-03-01_2025-04-01.parquet"):
    rec = [dict(ADES_mvt="LIRF", PHASE_mvt="ARR", STAND_mvt=s, BLOCK_TIME_UTC_mvt=T(t, tz="UTC")) for s, t in arrivals]
    pd.DataFrame(rec).to_parquet(tmp_path / fname, index=False)
    return str(tmp_path)


def _rows():
    """Six departures: turn+k_in, neither flag, below band, unknown stand, stand with no arrivals, NaN p_R."""
    sched = ["2025-03-10 09:00", "2025-03-10 06:00", "2025-03-10 11:00", "2025-03-10 09:00", "2025-03-10 08:00",
             "2025-03-10 09:00"]
    mvt = ["2025-03-10 11:30", "2025-03-10 11:30", "2025-03-10 11:30", "2025-03-10 11:30", "2025-03-10 11:30",
           "2025-03-10 11:30"]
    return pd.DataFrame({"STAND_mvt": ["A1", "B2", "A1", None, "Z9", "A1"],
                         "SCHED_TIME_UTC_mvt": [T(s, tz="UTC") for s in sched],
                         "MVT_TIME_UTC_mvt": [T(m, tz="UTC") for m in mvt],
                         "p_R": [0.3, 0.6, 0.3, 0.3, 0.2, np.nan]})


@pytest.mark.skipif(not RULE_PATH.exists(), reason="frozen rule absent")
def test_e2_form_equals_the_frozen_rule_with_e2_coefficients(tmp_path):
    E = _load("agent_E_rules", RULE_PATH)
    raw = _raw_world(tmp_path, [("A1", "2025-03-10 10:30"), ("B2", "2025-03-10 02:00")])
    rows = _rows()
    f = E.stand_inbound_features(rows, raw_dir=raw)
    sp = (rows.MVT_TIME_UTC_mvt - rows.SCHED_TIME_UTC_mvt).dt.total_seconds().to_numpy()
    got = S.e2_form(rows.p_R, f.k_in, f.turn, sp, W.E2_BETA)
    want = E.rule_E2(rows, raw_dir=raw)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    fin = np.isfinite(want)
    assert fin.sum() == 3                                   # rows 0, 1, 4 fire; 2 below band, 3 no stand, 5 NaN p_R
    assert np.allclose(got[fin], want[fin], rtol=1e-12, atol=0)


def test_e2_form_band_edges_are_inclusive():
    sp = np.array([3599.0, 3600.0, 24000.0, 24001.0])
    p = S.e2_form(np.full(4, 0.5), np.zeros(4), np.zeros(4), sp, W.E2_BETA)
    assert np.isnan(p[0]) and np.isfinite(p[1]) and np.isfinite(p[2]) and np.isnan(p[3])


def test_e2_form_uses_the_given_coefficients():
    beta = np.array([0.7175, -0.3101, 2.61, -0.5053])
    p = S.e2_form(np.array([0.4]), np.array([2.0]), np.array([1.0]), np.array([5000.0]), beta)
    z = 0.7175 * np.log(0.4 / 0.6) - 0.3101 + 2.61 - 0.5053
    assert p[0] == pytest.approx(1 / (1 + np.exp(-z)), rel=1e-12)


def test_e2_form_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length"):
        S.e2_form(np.ones(2) * 0.5, np.zeros(3), np.zeros(2), np.full(2, 5e3), W.E2_BETA)


# ---------------------------------------------------------------- k and the G range

def test_k_ratio_is_the_projection_of_e_on_d():
    D = np.array([1.0, 2.0, -1.0]); e = np.array([2.0, 1.0, 0.0])
    assert S.k_ratio(D, e) == pytest.approx((2 + 2 + 0) / (1 + 4 + 1))


def test_k_ratio_refuses_no_move():
    with pytest.raises(ValueError, match="no move"):
        S.k_ratio(np.zeros(3), np.ones(3))


def test_k_day_ci_is_degenerate_when_e_is_a_fixed_multiple_of_d():
    rng = np.random.default_rng(1)
    D = rng.normal(size=60); day = np.repeat(np.arange(20).astype(str), 3)
    lo, hi = S.k_day_ci(D, 2.0 * D, day)
    assert lo == pytest.approx(2.0, abs=1e-12) and hi == pytest.approx(2.0, abs=1e-12)


def test_k_day_ci_resamples_whole_days():
    """12 days x 3 rows of random D, e. The CI must equal the quantiles of the day-block ratio sum(D e)/sum(D^2) over 2,000 draws
    of 12 whole days with default_rng(0). (A first two-day version let a row-bootstrap mutant escape: both gave (0, 1).)"""
    rng = np.random.default_rng(3)
    D = rng.normal(size=36); e = 0.8 * D + rng.normal(size=36); day = np.repeat([f"d{i:02d}" for i in range(12)], 3)
    num = pd.Series(D * e).groupby(day).sum().to_numpy(); den = pd.Series(D * D).groupby(day).sum().to_numpy()
    idx = np.random.default_rng(S.SEED).integers(0, 12, size=(S.N_BOOT, 12))
    ks = num[idx].sum(1) / den[idx].sum(1)
    want = (float(np.quantile(ks, 0.025)), float(np.quantile(ks, 0.975)))
    assert S.k_day_ci(D, e, day) == pytest.approx(want, rel=1e-12)
    ridx = np.random.default_rng(S.SEED).integers(0, 36, size=(S.N_BOOT, 36))
    rows_ci = np.quantile((D * e)[ridx].sum(1) / (D * D)[ridx].sum(1), [0.025, 0.975])
    assert not np.allclose(want, rows_ci, rtol=1e-6)          # the fixture separates day blocks from a row bootstrap


def test_k_day_ci_rejects_unequal_lengths():
    with pytest.raises(ValueError, match="length"):
        S.k_day_ci(np.ones(3), np.ones(3), np.array(["a", "b"]))


def test_g_range_identity():
    assert S.g_range(100.0, (0.2, 0.845, 1.5)) == pytest.approx((100 * -0.6, 100 * 0.69, 100 * 2.0))


# ---------------------------------------------------------------- row set, assembly, row checks

def test_row_set_marks_the_changed_ids():
    assert S.row_set(np.array([5, 6, 7]), [7, 5]).tolist() == [True, False, True]


def test_row_set_refuses_ids_outside_the_scope():
    with pytest.raises(ValueError, match="not in the scope rows"):
        S.row_set(np.array([5, 6]), [5, 9])


def _base(n=5):
    return pd.DataFrame({"MVT_ID_mvt": np.arange(1, n + 1, dtype="float64"),
                         "TAXITIME_SEC_mvt": np.arange(100, 100 + n, dtype="int32")})


def test_assemble_changes_only_the_named_ids():
    base = _base()
    out = S.assemble(base, np.array([4.0, 2.0]), np.array([900.0, 700.0]))
    assert out.TAXITIME_SEC_mvt.tolist() == [100, 700, 102, 900, 104]
    assert out.dtypes.equals(base.dtypes) and base.TAXITIME_SEC_mvt.tolist() == [100, 101, 102, 103, 104]


def test_assemble_refuses_unknown_ids():
    with pytest.raises(KeyError, match="not in the base"):
        S.assemble(_base(), np.array([99.0]), np.array([900.0]))


@pytest.mark.parametrize("bad", [900.5, np.nan])
def test_assemble_refuses_non_integral_values(bad):
    with pytest.raises(ValueError, match="integral"):
        S.assemble(_base(), np.array([1.0]), np.array([bad]))


def test_check_rows_passes_a_well_formed_candidate():
    base = _base()
    S.check_rows(base.copy(), base, set(base.MVT_ID_mvt), n_rows=5)


def test_check_rows_refuses_wrong_row_count():
    base = _base()
    with pytest.raises(AssertionError, match="rows"):
        S.check_rows(base.iloc[:4].copy(), base, set(base.MVT_ID_mvt), n_rows=5)


def test_check_rows_refuses_reordered_ids():
    base = _base()
    with pytest.raises(AssertionError, match="order"):
        S.check_rows(base.iloc[::-1].reset_index(drop=True), base, set(base.MVT_ID_mvt), n_rows=5)


def test_check_rows_refuses_changed_dtypes():
    base = _base(); c = base.copy(); c["TAXITIME_SEC_mvt"] = c.TAXITIME_SEC_mvt.astype("int64")
    with pytest.raises(AssertionError, match="dtypes"):
        S.check_rows(c, base, set(base.MVT_ID_mvt), n_rows=5)


def test_check_rows_refuses_ids_that_are_not_ranking_dep():
    base = _base()
    with pytest.raises(AssertionError, match="ranking DEP"):
        S.check_rows(base.copy(), base, set(base.MVT_ID_mvt) - {3.0} | {77.0}, n_rows=5)


def test_check_beta():
    S.check_beta([1.0, 2.0], [1.0 + 5e-7, 2.0])
    with pytest.raises(AssertionError, match="B2"):
        S.check_beta([1.0, 2.0], [1.0 + 2e-6, 2.0])


# ---------------------------------------------------------------- 2026 serve rows

def test_serve_rows_2026_drops_dep_labels():
    scope = pd.DataFrame({"MVT_ID_mvt": [1, 2], "p_C": [0.3, 0.4], "nf": [900.0, 800.0]})
    rk = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "STAND_mvt": ["A", "B", "C"],
                       "BLOCK_TIME_UTC_mvt": [T("2026-01-01", tz="UTC")] * 3, "TAXITIME_SEC_mvt": [1, 2, 3]})
    out = S.serve_rows_2026(scope, rk)
    assert "BLOCK_TIME_UTC_mvt" not in out.columns and "TAXITIME_SEC_mvt" not in out.columns
    assert out.MVT_ID_mvt.tolist() == [1, 2] and out.p_R.tolist() == [0.3, 0.4] and out.STAND_mvt.tolist() == ["A", "B"]


def test_serve_rows_2026_refuses_scope_ids_missing_from_ranking():
    scope = pd.DataFrame({"MVT_ID_mvt": [1, 9], "p_C": [0.3, 0.4], "nf": [900.0, 800.0]})
    rk = pd.DataFrame({"MVT_ID_mvt": [1.0], "STAND_mvt": ["A"]})
    with pytest.raises(ValueError, match="missing from ranking"):
        S.serve_rows_2026(scope, rk)


# ---------------------------------------------------------------- 2025 evidence (pure part)

def _world(n=2400, seed=0):
    rng = np.random.default_rng(seed)
    p_R = rng.uniform(0.05, 0.95, n); k_in = (rng.random(n) < 0.4).astype(float); turn = (rng.random(n) < 0.3).astype(float)
    X = W.design(p_R, k_in, turn)
    fill = rng.random(n) < W.predict(X, np.array([0.7, 0.3, 1.5, -0.5]))
    sp = rng.uniform(3600, 20000, n); nf = rng.uniform(600, 1500, n)
    y = np.where(fill, sp, nf) + rng.normal(0, 60, n)
    month = rng.integers(1, 13, n); day = np.array([f"2025-{m:02d}-{d:02d}" for m, d in zip(month, rng.integers(1, 20, n))])
    fit_mask = rng.random(n) < 0.9
    V = rng.random(n) < 0.85
    return dict(X=X, fill=fill, w=(sp - nf) ** 2, month=month, fit_mask=fit_mask, V=V, sp=sp, nf=nf, y=y, day=day)


def test_evidence_from_rows_fits_w_on_fit_rows_only():
    d = _world()
    ev = S.evidence_from_rows(**d, mixture=_mix, n_boot=200)
    m = d["fit_mask"]
    assert np.allclose(ev["beta_W"], W.fit_brier_w(d["X"][m], d["fill"][m], d["w"][m]), atol=1e-10)
    assert not np.allclose(ev["beta_W"], W.fit_brier_w(d["X"], d["fill"], d["w"]), atol=1e-4)


def test_evidence_from_rows_g_and_k_match_the_e2w_definitions():
    d = _world()
    ev = S.evidence_from_rows(**d, mixture=_mix, n_boot=200)
    pL = W.lomo(d["X"], d["fill"], d["w"], d["month"], lambda X_, y_, w_: W.fit_logloss(X_, y_), d["fit_mask"])
    pW = W.lomo(d["X"], d["fill"], d["w"], d["month"], W.fit_brier_w, d["fit_mask"])
    V, sp, nf, y = d["V"], d["sp"], d["nf"], d["y"]
    g = float((((_mix(pL, sp, nf) - y) ** 2 - (_mix(pW, sp, nf) - y) ** 2)[V]).sum() / BOARD_N)
    D = (_mix(pW, sp, nf) - _mix(pL, sp, nf))[V]; e = (y - _mix(pL, sp, nf))[V]
    assert ev["g_WL"] == pytest.approx(g, rel=1e-12)
    assert ev["k"] == pytest.approx(float((D * e).sum() / (D ** 2).sum()), rel=1e-12)
    lo, hi = ev["k_ci"]
    assert lo <= ev["k"] <= hi


def test_evidence_from_rows_reference_is_w_all12_vs_e2_on_verdict_rows():
    d = _world()
    ev = S.evidence_from_rows(**d, mixture=_mix, n_boot=50)
    V = d["V"]
    base = np.rint(_mix(W.predict(d["X"], W.E2_BETA), d["sp"], d["nf"]))[V]
    new = np.rint(_mix(W.predict(d["X"], ev["beta_W"]), d["sp"], d["nf"]))[V]
    assert ev["ref_day_energy"] == pytest.approx(S.B.day_energy(base, new, d["day"][V]))


def test_evidence_from_rows_reports_the_in_sample_sse_shift_of_w_all12_vs_e2():
    d = _world()
    ev = S.evidence_from_rows(**d, mixture=_mix, n_boot=50)
    V, sp, nf, y = d["V"], d["sp"], d["nf"], d["y"]
    pE = W.predict(d["X"], W.E2_BETA); pW = W.predict(d["X"], ev["beta_W"])
    want = float((((_mix(pE, sp, nf) - y) ** 2 - (_mix(pW, sp, nf) - y) ** 2)[V]).sum() / BOARD_N)
    assert ev["sse_shift_insample"] == pytest.approx(want, rel=1e-12)


def test_evidence_from_rows_uses_ref_day_for_the_reference_when_given():
    d = _world()
    ref_day = np.where(np.arange(len(d["day"])) % 2 == 0, "2025-01-01", "2025-01-02")
    ev = S.evidence_from_rows(**d, mixture=_mix, n_boot=50, ref_day=ref_day)
    assert set(ev["ref_day_energy"]) <= {"2025-01-01", "2025-01-02"}


def test_largest_moves_orders_by_absolute_move():
    ids = np.array([10.0, 11.0, 12.0, 13.0]); before = np.array([100., 100., 100., 100.])
    after = np.array([150., 20., 100., 130.])
    got = S.largest_moves(ids, before, after, n=2)
    assert [r["MVT_ID_mvt"] for r in got] == [11.0, 10.0] and got[0]["delta"] == -80.0
