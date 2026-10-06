"""Tests for scripts/analysis/msw_build.py (MSW, plans/PREREG_msw_mau_sweep_2026_09_21.md).

Written FIRST and seen RED (ImportError: msw_build did not exist) before the module was written.

The registration's GATE: the parameterised extra-row construction at m = 20 must reproduce
`mau_build.extra_rows` ROW-FOR-ROW and COLUMN-FOR-COLUMN (index, dtypes and values), on a fixture AND on
all twelve real 2025 months (93,165 rows). If it does not, the sweep does not run.

Mutation rehearsal, EXECUTED 2026-09-21 and watched, __pycache__ purged between mutants (BC-16), source sha
verified restored after every round. Runners in the session scratchpad (msw_mutants*.py).
  Round 1 (with -x, first failure only): M1-M13; M9 (m=20 runtime gate deleted) ESCAPED because the nested
  m=5/10 gates masked it and the test only matched "GATE" -> test strengthened to pin WHICH gate fires.
  Round 2: M9 re-run KILLED; M14 (m>20 nested gate deleted), M15 (m<20 containment gate deleted) KILLED.
  Round 3: M16 (registration marker back to one newline -> hashes original + '\n') KILLED.
  End-of-session round, WITHOUT -x (every failing test recorded): M1-M34 all KILLED, incl. M17 unpadded cell key,
  M18 schedule-fill filter dropped, M19 modulus validation removed, M20 airline slice drift, M21 find() for
  index(), M22a/b weight validation removed, M23 design check removed, M24 equal_nan dropped, M25 wrong
  bootstrap base, M26 swapped args, M27 c2 on the upper bound, M28 no winner -> WORKING, M29 leg check
  removed, M31 only m=20 built, M32 n_fit/n_extra swapped, M33 confirm never passes, M34 min for max.
  Grid round: M35 a registered modulus dropped, M36 duplicate cell -> KILLED (the only test function not yet
  red, test_grid_is_exactly_the_registered_twelve_cells, went red on both).
  36 distinct mutants executed, 36 killed; ALL 37 test functions seen red against a targeted mutant.
  Review round (after an adversarial reviewer found two tests weaker than their names): the outlier test
  moved only ONE of the two rows (a single-id MONSTER_IDS passed) and the fixture's inadmissible rows had ids
  the modulus dropped anyway (the filters were never exercised). Both strengthened, then: M37 single outlier
  id, M38 AOBT filter dropped, M40 DEP filter dropped -> KILLED; M39 build_submission.admissible disabled
  IN MEMORY (another lane's file, never edited) -> AssertionError at the inadmissible-row assertion, KILLED.
  40 mutants executed, 40 killed.

`main()` (the orchestration) was a NAMED GAP until 2026-09-21; it is now covered end to end on the real data by
test_main_end_to_end_reproduces_the_persisted_run (marked `slow`, ~2 min), on top of its runtime gates.
Fast tier: `pytest -m "not slow" tests/test_msw_build.py`.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "scripts/analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mau_build as MB  # noqa: E402  (another lane's validated file; imported, never edited)
from research.analysis import msw_build as W  # noqa: E402  (moved here, FP-W4)

RAW_FILES = sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))
MAU_TRAIN = ROOT / "reports/mau_training_rows.parquet"


# ----------------------------------------------------------------------------------------------------
# fixture: a raw month shaped like data/raw/training_2025-*.parquet (same dtypes, tz-aware [us] clocks)
# ----------------------------------------------------------------------------------------------------
def _raw_fixture(seed: int = 0) -> pd.DataFrame:
    """Two airports x two hours of departures, >= MIN_N + 1 per group so the witness is non-NaN, plus
    every row class `extra_rows` must drop: ARR rows, null AOBT, zero and null taxi, null block, a
    schedule-fill row, a duplicate MVT_ID, and ids that are / are not multiples of 5, 10, 20 and 40."""
    rng = np.random.default_rng(seed)
    rows = []
    mvt_id = 1000
    for ap in ("AAAA", "BBBB"):
        for hour in (9, 10):
            for k in range(10):
                t = pd.Timestamp(f"2025-01-15 {hour:02d}:{5 * k:02d}:17", tz="UTC")
                rows.append(dict(PHASE_mvt="DEP", MVT_ID_mvt=float(mvt_id), ADEP_mvt=ap,
                                 RUNWAY_mvt=str(rng.choice(["09L", "27R"])),
                                 STAND_mvt=f"{'AB'[k % 2]}{k}", FLIGHT_mvt=f"XYZ{100 + k}",
                                 MVT_TIME_UTC_mvt=t,
                                 SCHED_TIME_UTC_mvt=t - pd.Timedelta(seconds=int(rng.integers(300, 3000))),
                                 BLOCK_TIME_UTC_mvt=t - pd.Timedelta(seconds=int(rng.integers(400, 900))),
                                 TAXITIME_SEC_mvt=int(rng.integers(300, 1500)),
                                 AOBT_3_flt=t - pd.Timedelta(seconds=int(rng.integers(200, 1200)))))
                mvt_id += 5
    f = pd.DataFrame(rows)
    extra = f.iloc[:8].copy()
    # The four inadmissible rows get ids that are multiples of 40, so EVERY modulus would sample them if its
    # filter failed (reviewer, 2026-09-21: ids 1001-1011 were dropped by the modulus, never by the filters).
    extra["MVT_ID_mvt"] = [3000.0, 3040.0, 3080.0, 3120.0, 2000.0, 2020.0, 2040.0, 2060.0]
    extra.loc[extra.index[0], "PHASE_mvt"] = "ARR"                       # arrival: dropped
    extra.loc[extra.index[1], "AOBT_3_flt"] = pd.NaT                     # no AOBT: dropped
    extra.loc[extra.index[2], "TAXITIME_SEC_mvt"] = 0                    # zero taxi: inadmissible
    extra.loc[extra.index[3], "BLOCK_TIME_UTC_mvt"] = pd.NaT             # null block: inadmissible
    extra.loc[extra.index[4], "BLOCK_TIME_UTC_mvt"] = extra.loc[extra.index[4], "SCHED_TIME_UTC_mvt"]  # fill
    extra.loc[extra.index[5], "SCHED_TIME_UTC_mvt"] -= pd.Timedelta(days=1)   # dayoff 1
    extra.loc[extra.index[6], "SCHED_TIME_UTC_mvt"] -= pd.Timedelta(days=2)   # dayoff clipped to 1
    dup = f.iloc[[4]].copy()                                             # duplicate id 1020 (% 20 == 0)
    dup["MVT_TIME_UTC_mvt"] += pd.Timedelta(seconds=30)
    f = pd.concat([f, extra, dup], ignore_index=True).sample(frac=1.0, random_state=seed)
    f = f.reset_index(drop=True)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "AOBT_3_flt"):
        f[c] = f[c].astype("datetime64[us, UTC]")
    f["TAXITIME_SEC_mvt"] = f["TAXITIME_SEC_mvt"].astype("int32")
    return f[MB.RAW_COLS]


# ----------------------------------------------------------------------------------------------------
# the grid
# ----------------------------------------------------------------------------------------------------
def test_grid_is_exactly_the_registered_twelve_cells():
    assert W.MODULI == (5, 10, 20, 40)
    assert W.RATIOS == (0.15, 0.25, 0.40)
    assert W.INCUMBENT == (20, 0.25)
    g = W.grid()
    assert len(g) == 12 and len(set(g)) == 12
    assert W.INCUMBENT in g
    assert g == [(m, r) for m in (5, 10, 20, 40) for r in (0.15, 0.25, 0.40)]


def test_cell_keys_are_unique_and_stable():
    keys = [W.cell_key(m, r) for m, r in W.grid()]
    assert len(set(keys)) == 12
    assert W.cell_key(20, 0.25) == "m20_r0.25"
    assert W.cell_key(5, 0.40) == "m5_r0.40"


# ----------------------------------------------------------------------------------------------------
# construction fidelity -- THE GATE
# ----------------------------------------------------------------------------------------------------
def test_gate_fixture_extra_rows_at_20_is_mau_row_for_row_column_for_column():
    raw = _raw_fixture()
    want = MB.extra_rows(raw)
    got = W.extra_rows_at(raw, 20)
    assert len(want) > 0, "fixture must exercise the sampled path"
    pd.testing.assert_frame_equal(got, want, check_exact=True)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gate_fixture_holds_across_shuffles(seed):
    raw = _raw_fixture(seed)
    pd.testing.assert_frame_equal(W.extra_rows_at(raw, 20), MB.extra_rows(raw), check_exact=True)


def test_fixture_drops_every_inadmissible_class():
    """The fixture must actually carry the rows it claims to, or the gate above tests less than it says."""
    got = W.extra_rows_at(_raw_fixture(), 5)
    ids = set(got.MVT_ID_mvt)
    for m in W.MODULI:                                   # ARR, null AOBT, zero taxi, null block: all % 40 == 0
        assert not set(W.extra_rows_at(_raw_fixture(), m).MVT_ID_mvt) & {3000.0, 3040.0, 3080.0, 3120.0}
    assert 2000.0 not in ids                             # schedule-fill row removed
    assert {2020.0, 2040.0} <= ids                       # the dayoff rows survive
    assert got.set_index("MVT_ID_mvt").loc[[2020.0, 2040.0], "dayoff"].tolist() == [1, 1]
    assert got.MVT_ID_mvt.is_unique                      # the duplicate id was de-duplicated
    assert got.hprox_med.notna().any()                   # the witness is exercised, not all-NaN


@pytest.mark.parametrize("m", [40])
def test_nested_subsample_equals_mau_filtered(m):
    """m = 40 must be exactly MAU's m = 20 rows filtered to MVT_ID % 40 == 0 (same index, same values)."""
    raw = _raw_fixture()
    want = MB.extra_rows(raw)
    want = want.loc[want.MVT_ID_mvt.astype("int64") % m == 0]
    pd.testing.assert_frame_equal(W.extra_rows_at(raw, m), want, check_exact=True)


@pytest.mark.parametrize("m", [5, 10])
def test_nested_supersample_contains_mau_exactly(m):
    """m in {5, 10} restricted to MVT_ID % 20 == 0 must be exactly MAU's rows; the rest are new rows."""
    raw = _raw_fixture()
    got = W.extra_rows_at(raw, m)
    assert (got.MVT_ID_mvt.astype("int64") % m == 0).all()
    pd.testing.assert_frame_equal(got.loc[got.MVT_ID_mvt.astype("int64") % 20 == 0],
                                  MB.extra_rows(raw), check_exact=True)
    assert len(got) > len(MB.extra_rows(raw))


@pytest.mark.parametrize("bad", [0, -20, 2.5, 20.0, True, "20", None])
def test_extra_rows_at_refuses_a_bad_modulus(bad):
    with pytest.raises(ValueError, match="modulus"):
        W.extra_rows_at(_raw_fixture(), bad)


def test_source_is_mau_verbatim_except_the_modulus():
    """Drift tripwire: if mau_build.extra_rows is ever edited, the copy must be re-verified, not trusted."""
    mau = inspect.getsource(MB.extra_rows).splitlines()[1:]
    ours = inspect.getsource(W.extra_rows_at).splitlines()
    start = ours.index("    # ---- verbatim mau_build.extra_rows body, modulus parameterised ----") + 1
    body = [ln.replace("% modulus == 0", "% 20 == 0") for ln in ours[start:]]
    assert body == mau


def test_extras_for_file_returns_every_modulus_and_enforces_the_gate():
    raw = _raw_fixture()
    out = W.extras_for_file(raw)
    assert set(out) == set(W.MODULI)
    pd.testing.assert_frame_equal(out[20], MB.extra_rows(raw), check_exact=True)


def test_extras_for_file_gate_fires_when_the_copy_disagrees(monkeypatch):
    """The runtime gate must actually stop the run: make MAU's reference drop one row. The m = 20 gate
    must be the one that fires (mutation M9 showed the nested gates alone would mask its removal)."""
    real = MB.extra_rows
    monkeypatch.setattr(W.MB, "extra_rows", lambda raw: real(raw).iloc[1:])
    with pytest.raises(AssertionError, match=r"GATE FAILED \(m=20 vs mau_build.extra_rows\)"):
        W.extras_for_file(_raw_fixture())


@pytest.mark.parametrize("m,what", [(40, r"m=40 nested in MAU"), (5, r"m=5 contains MAU"),
                                    (10, r"m=10 contains MAU")])
def test_extras_for_file_nested_gates_fire_per_modulus(monkeypatch, m, what):
    """Corrupt ONE modulus's rows (drop its first MAU-shared row); that modulus's gate must fire."""
    real = W.extra_rows_at

    def corrupted(raw, modulus):
        out = real(raw, modulus)
        if modulus == m:
            shared = out.index[out.MVT_ID_mvt.astype("int64") % 40 == 0]
            out = out.drop(shared[0])
        return out

    monkeypatch.setattr(W, "extra_rows_at", corrupted)
    with pytest.raises(AssertionError, match=rf"GATE FAILED \({what}\)"):
        W.extras_for_file(_raw_fixture())


@pytest.mark.slow
@pytest.mark.skipif(len(RAW_FILES) != 12 or not MAU_TRAIN.exists(),
                    reason="real 2025 training months absent (data/ is not in git)")
def test_gate_real_data_all_twelve_months_93165_rows():
    """THE GATE on real data (2.71 s measured 2026-09-21, hence `slow`): every month row-for-row and
    column-for-column, 93,165 rows in total, and the concatenation equals the training rows MAU itself persisted."""
    parts = []
    for path in RAW_FILES:
        raw = pd.read_parquet(path, columns=MB.RAW_COLS)
        want = MB.extra_rows(raw)
        got = W.extra_rows_at(raw, 20)
        pd.testing.assert_frame_equal(got, want, check_exact=True)
        parts.append(got)
    extra = pd.concat(parts, ignore_index=True)
    assert len(extra) == W.EXPECTED_EXTRA_AT_20 == 93_165
    pd.testing.assert_frame_equal(extra, pd.read_parquet(MAU_TRAIN), check_exact=True)


def test_registration_sha_is_the_bytes_above_the_first_appendix(tmp_path, monkeypatch):
    original = b"# Registration\n\nbody line.\n"
    appended = original + b"\n---\nsha256 of everything above this line = abc\n\n## A1\nmore\n"
    p = tmp_path / "prereg.md"
    p.write_bytes(appended)
    monkeypatch.setattr(W, "PREREG", p)
    import hashlib
    registered, current = W.registration_sha()
    assert registered == hashlib.sha256(original).hexdigest()
    assert current == hashlib.sha256(appended).hexdigest()


def test_registration_sha_refuses_an_unappended_file(tmp_path, monkeypatch):
    p = tmp_path / "prereg.md"
    p.write_bytes(b"# Registration\nno appendix\n")
    monkeypatch.setattr(W, "PREREG", p)
    with pytest.raises(ValueError):
        W.registration_sha()


# ----------------------------------------------------------------------------------------------------
# the mass ratio
# ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("r", [0.15, 0.25, 0.40])
def test_extra_weight_gives_the_registered_mass(r):
    w = W.extra_weight(17_532, 86_322, r)
    sw = np.r_[np.ones(17_532), np.full(86_322, w)]
    assert sw[17_532:].sum() / sw[:17_532].sum() == pytest.approx(r, rel=1e-12)


def test_extra_weight_at_the_incumbent_is_maus_arithmetic_bit_for_bit():
    """MAU: `weight = 0.25*len(fi)/len(ex)`; the incumbent's weight must be MAU's recorded float exactly.
    Guards the COUNTS and formula (mutant M32, n_fit/n_extra swapped -> KILLED). It does NOT guard evaluation
    order: 0.25 is a power of two, so both orders are exact here and mutant M2 (reordered) passes this test;
    M2 is caught by test_fit_candidate_is_maus_inline_fit_exactly at r = 0.15."""
    for n_fit, n_extra in [(17_532, 86_322), (18_000, 93_165), (1, 3)]:
        assert W.extra_weight(n_fit, n_extra, 0.25) == 0.25 * n_fit / n_extra
    assert W.extra_weight(17_532, 86_322, 0.25) == 0.05077500521303955   # mau_report.json fits[0]


@pytest.mark.parametrize("n_fit,n_extra,r", [(0, 10, 0.25), (10, 0, 0.25), (10, 10, 0.0),
                                             (10, 10, -0.25), (10, 10, float("nan")),
                                             (10, 10, float("inf"))])
def test_extra_weight_refuses_bad_inputs(n_fit, n_extra, r):
    with pytest.raises(ValueError):
        W.extra_weight(n_fit, n_extra, r)


# ----------------------------------------------------------------------------------------------------
# the candidate fit -- must be MAU's inline arithmetic (mau_build.py:192-199) at every ratio
# ----------------------------------------------------------------------------------------------------
def _fit_data(seed=0, n_fit=600, n_extra=900, n_val=150):
    rng = np.random.default_rng(seed)
    cols = ["a", "b", "c"]
    x = pd.DataFrame(rng.normal(size=(n_fit, 3)), columns=cols)
    ex = pd.DataFrame(rng.normal(size=(n_extra, 3)), columns=cols)
    xv = pd.DataFrame(rng.normal(size=(n_val, 3)), columns=cols)
    y = 400 * x.a.to_numpy() + rng.normal(0, 50, n_fit)
    ye = 400 * ex.a.to_numpy() + 300 * ex.b.to_numpy() + rng.normal(0, 50, n_extra)
    ye[:5] = 9_000.0                                     # beyond the 3,000 s cap: must be clipped
    yv = 400 * xv.a.to_numpy()
    return x, y, ex, ye, xv, yv


@pytest.mark.parametrize("r", [0.15, 0.25, 0.40])
def test_fit_candidate_is_maus_inline_fit_exactly(r):
    x, y, ex, ye, xv, yv = _fit_data()
    model, info = W.fit_candidate(x, y, ex, ye, r, seed=1, x_val=xv, y_val=yv)
    weight = r * len(x) / len(ex)                                    # mau_build.py:192, r in place of 0.25
    tx = pd.concat([x, ex], ignore_index=True)
    ty = np.r_[y, np.clip(ye, -3000, 3000)]
    sw = np.r_[np.ones(len(x)), np.full(len(ex), weight)]
    ref = MB.HGR(random_state=1, early_stopping=True, **MB.uc.bs.NF_PARAMS)
    ref.fit(tx, ty, sample_weight=sw, X_val=xv, y_val=yv, sample_weight_val=np.ones(len(xv)))
    np.testing.assert_array_equal(model.predict(xv), ref.predict(xv))
    assert info["mass_ratio"] == pytest.approx(r, rel=1e-12)
    assert info["weight"] == weight and info["iter"] == int(ref.n_iter_)


def test_fit_candidate_ratio_changes_the_model():
    """A mutation guard: if the ratio were ignored, 0.15 and 0.40 would give identical predictions."""
    x, y, ex, ye, xv, yv = _fit_data()
    a, _ = W.fit_candidate(x, y, ex, ye, 0.15, seed=0, x_val=xv, y_val=yv)
    b, _ = W.fit_candidate(x, y, ex, ye, 0.40, seed=0, x_val=xv, y_val=yv)
    assert np.abs(a.predict(xv) - b.predict(xv)).max() > 1.0


def test_fit_candidate_clips_the_extra_target():
    x, y, ex, ye, xv, yv = _fit_data()
    clipped = ye.copy()
    clipped[:5] = 3000.0
    a, _ = W.fit_candidate(x, y, ex, ye, 0.25, seed=0, x_val=xv, y_val=yv)
    b, _ = W.fit_candidate(x, y, ex, clipped, 0.25, seed=0, x_val=xv, y_val=yv)
    np.testing.assert_array_equal(a.predict(xv), b.predict(xv))


def test_fit_candidate_refuses_a_design_mismatch():
    x, y, ex, ye, xv, yv = _fit_data()
    with pytest.raises(ValueError, match="columns"):
        W.fit_candidate(x, y, ex.rename(columns={"a": "z"}), ye, 0.25, seed=0, x_val=xv, y_val=yv)


# ----------------------------------------------------------------------------------------------------
# the determinism null and the selection statistic
# ----------------------------------------------------------------------------------------------------
def test_rerun_identical_passes_on_identical_arrays_including_nan():
    """A check that fires on equal runs would VOID a valid sweep: NaN in the same place in both runs is
    identical, not a difference (NaN != NaN would otherwise count it)."""
    a = np.array([1.0, np.nan, 3.0])
    W.assert_rerun_identical(a, a.copy(), "x")


def test_rerun_identical_fails_on_one_ulp():
    a = np.array([1.0, 2.0, 3.0])
    b = a.copy()
    b[1] = np.nextafter(b[1], 10.0)
    with pytest.raises(AssertionError, match="VOID"):
        W.assert_rerun_identical(a, b, "x")


def _days(n_per_month=200, seed=0):
    rng = np.random.default_rng(seed)
    month = np.r_[np.full(n_per_month, 1), np.full(n_per_month, 7)]
    day = np.r_[rng.integers(1, 32, n_per_month), rng.integers(182, 213, n_per_month)]
    y = rng.normal(900, 300, 2 * n_per_month)
    return y, day, month


def test_delta_null_is_exactly_zero_with_a_degenerate_ci():
    y, day, month = _days()
    inc = y + 100.0
    st = W.delta_stats(y, inc, inc.copy(), day, month)
    assert st["gain"] == 0.0 and st["ci95"] == [0.0, 0.0]


def test_delta_refuses_degenerate_day_blocks_bc23():
    """BC-23: a `day` that fell back to `month` must be refused, not bootstrapped into a zero-width CI."""
    y, day, month = _days()
    with pytest.raises(ValueError, match="day blocks"):
        W.delta_stats(y, y + 50.0, y + 40.0, month.copy(), month)


def test_delta_sign_and_value_is_the_board_difference():
    y, day, month = _days()
    inc, cell = y + 100.0, y + 50.0
    st = W.delta_stats(y, inc, cell, day, month)
    want = (((y - inc) ** 2).sum() - ((y - cell) ** 2).sum()) / 344_841
    assert st["gain"] == pytest.approx(want, rel=1e-12) and st["gain"] > 0
    assert st["ci95"][0] > 0


def test_delta_equals_gain_over_control_minus_incumbent_gain():
    """Clause 1 ('a cell's gain exceeds the incumbent's') is the same statement as delta > 0."""
    y, day, month = _days()
    ctrl, inc, cell = y + 200.0, y + 100.0, y - 60.0
    d = W.delta_stats(y, inc, cell, day, month)["gain"]
    g_cell = W.delta_stats(y, ctrl, cell, day, month)["gain"]
    g_inc = W.delta_stats(y, ctrl, inc, day, month)["gain"]
    assert d == pytest.approx(g_cell - g_inc, rel=1e-9)


def test_monster_rows_contribution_is_isolated():
    ids = np.array([1.0, 182378789.0, 2.0, 182377662.0])
    y = np.array([100.0, 84_240.0, 100.0, 58_206.0])
    inc = np.array([100.0, 877.0, 100.0, 832.0])
    cell = np.array([100.0, 977.0, 150.0, 732.0])       # BOTH outlier rows move, so dropping either id bites
    got = W.monster_delta(ids, y, inc, cell)
    want = (((84_240 - 877) ** 2 - (84_240 - 977) ** 2) + ((58_206 - 832) ** 2 - (58_206 - 732) ** 2)) / 344_841
    assert got == pytest.approx(want, rel=1e-12)


# ----------------------------------------------------------------------------------------------------
# the clauses -- the oracle must pass all four; an identical cell and a noise winner must fail
# ----------------------------------------------------------------------------------------------------
def _st(gain, lo, hi):
    return {"gain": gain, "ci95": [lo, hi], "monthly": {}, "n": 100}


def test_screen_clauses():
    assert W.screen_clauses(_st(5.0, 1.0, 9.0)) == {"c1_beats_incumbent": True, "c2_ci_excludes_0": True}
    assert W.screen_clauses(_st(5.0, -1.0, 9.0)) == {"c1_beats_incumbent": True, "c2_ci_excludes_0": False}
    assert W.screen_clauses(_st(0.0, 0.0, 0.0)) == {"c1_beats_incumbent": False, "c2_ci_excludes_0": False}
    assert W.screen_clauses(_st(-5.0, -9.0, -1.0))["c1_beats_incumbent"] is False


def test_confirm_clause_reads_ambiguous_as_not_working():
    assert W.confirm_clause(_st(3.0, 0.5, 6.0)) is True
    assert W.confirm_clause(_st(3.0, -0.5, 6.0)) is False      # ahead on the point, CI spans 0: ambiguous
    assert W.confirm_clause(_st(0.0, 0.0, 0.0)) is False
    assert W.confirm_clause(_st(-3.0, -6.0, -0.5)) is False


def test_pick_winner_none_when_nothing_passes():
    screens = {"m5_r0.15": _st(3.0, -1.0, 7.0), "m20_r0.25": _st(0.0, 0.0, 0.0)}
    assert W.pick_winner(screens) is None


def test_pick_winner_takes_the_largest_passing_delta_and_never_the_incumbent():
    screens = {"m5_r0.15": _st(3.0, 0.5, 7.0), "m10_r0.40": _st(8.0, 1.0, 15.0),
               "m40_r0.15": _st(20.0, -2.0, 40.0), "m20_r0.25": _st(99.0, 50.0, 150.0)}
    assert W.pick_winner(screens) == "m10_r0.40"


def test_oracle_passes_every_clause_through_the_real_code_path():
    """Run the oracle through the acceptance clauses."""
    y, day, month = _days()
    inc = y + np.random.default_rng(5).normal(0, 300, len(y))
    oracle = W.delta_stats(y, inc, y, day, month)
    screens = {"m20_r0.25": W.delta_stats(y, inc, inc, day, month), "m5_r0.40": oracle}
    winner = W.pick_winner(screens)
    assert winner == "m5_r0.40"
    assert W.verdict(winner, forward=oracle, lfpg_excluded=oracle) == "WORKING"


def test_identical_cell_fails():
    y, day, month = _days()
    inc = y + 50.0
    same = W.delta_stats(y, inc, inc.copy(), day, month)
    assert W.pick_winner({"m20_r0.25": same, "m5_r0.15": same}) is None
    assert W.verdict(None, forward=None, lfpg_excluded=None) == "NOT WORKING"


def test_noise_winner_dies_on_an_ambiguous_forward_leg():
    y, day, month = _days()
    rng = np.random.default_rng(11)
    inc = y + rng.normal(0, 300, len(y))
    winner_screen = W.delta_stats(y, inc, y, day, month)          # looks great on the screen
    noise = W.delta_stats(y, inc, inc + rng.normal(0, 1, len(y)), day, month)
    assert noise["ci95"][0] < 0 < noise["ci95"][1]
    assert W.verdict("m5_r0.40", forward=noise, lfpg_excluded=winner_screen) == "NOT WORKING"
    assert W.verdict("m5_r0.40", forward=winner_screen, lfpg_excluded=noise) == "NOT WORKING"


def test_verdict_refuses_a_winner_without_both_legs():
    with pytest.raises(ValueError, match="both"):
        W.verdict("m5_r0.40", forward=None, lfpg_excluded=_st(1.0, 0.5, 2.0))


# ----------------------------------------------------------------------------------------------------
# main(): the orchestration end to end, on the real data (closes UNTESTED.md's msw_build.main() entry)
# ----------------------------------------------------------------------------------------------------
PERSISTED_REPORT, PERSISTED_PREDS = W.REPORT, W.PREDS
MAIN_INPUTS = (PERSISTED_REPORT, PERSISTED_PREDS, W.HASHLOG, MB.REPORT, MB.PREDS, MB.TRAIN,
               ROOT / "data/cache_rome/unm.parquet", ROOT / "data/cache_stand/unm_physics_lomo.parquet")
# Not results: wall clock, process RSS, the parquet's byte-level hash (the frame itself is compared exactly), and
# two live-file hashes checked against the files on disk instead of against the persisted run.
VOLATILE_REPORT_KEYS = {"seconds", "peak_rss_gib", "pred_sha256", "prereg_file_sha256", "runner_sha256"}


def _run_main_into(tmp_path):
    """main() in a FRESH interpreter with its two outputs redirected. A subprocess, not an in-process call:
    main() asserts the process's lifetime peak RSS (`ru_maxrss`) < 2 GiB, which inside a long pytest session
    depends on whichever tests ran first -- an order-dependent failure."""
    report, preds = tmp_path / "msw_report.json", tmp_path / "msw_cell_predictions.parquet"
    code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from research.analysis import msw_build as W; "
            "W.REPORT, W.PREDS = Path(sys.argv[2]), Path(sys.argv[3]); W.main()")
    proc = subprocess.run([sys.executable, "-B", "-c", code, str(ROOT), str(report), str(preds)],
                          capture_output=True, text=True, timeout=W.ALARM_S + 120)
    return proc, report, preds


@pytest.mark.slow
@pytest.mark.skipif(len(RAW_FILES) != 12 or not all(p.exists() for p in MAIN_INPUTS),
                    reason="real 2025 data or the persisted MSW/MAU run absent (data/ and parquets are not in git)")
def test_main_end_to_end_reproduces_the_persisted_run(tmp_path):
    """main() re-run on the real data (~2 min, hence `slow`) must reproduce the registered run.

    Snapshot INVARIANT: every cell's per-row predictions, every seed column, and
    every reported number (grid deltas and CIs, per-seed deltas, outlier deltas, fit records, null, oracle,
    verdict) are bit-identical to reports/msw_cell_predictions.parquet and reports/msw_report.json. Exact
    equality is the claim, not a tolerance choice: the harness is deterministic (fixed HGB random_state, sklearn
    pinned to 1.8.0 by mau_build.stopping_split; the in-run rerun null was exactly 0.0), so ANY difference is a
    changed input, library or orchestration. The snapshot may be updated only by a hashed amendment to
    plans/PREREG_msw_mau_sweep_2026_09_21.md, because it IS the registered result.

    WIRING (UNTESTED.md's named uncovered edge: a wrong `params[key]` lookup for a non-incumbent cell passes
    every incumbent gate): each of the 78 fit records carries the (m, r) its cell key names, the rerun carries
    the incumbent's, the fitted extra mass equals r, one extra-row count per (m, month), and m = 20 January
    equals MAU's own record (86,322).

    Mutation rehearsal, EXECUTED 2026-09-21 and watched, WITHOUT -x, python -B with __pycache__ purged (BC-16),
    in a scratch mirror (symlinked repo + a mutated COPY of msw_build.py; the real runner was never written and
    its sha f20f0ca1... was verified before and after each run):
      M0 unmutated copy (the mirror itself)             -> PASSED, 144 s.
      M1 `m, r = params[key]` -> `params[inc_key]`      -> KILLED at the wiring assertion:
         cell m5_r0.15 fitted at (20, 0.25). This is UNTESTED.md's edge; every incumbent gate passed it.
      M2 grid 'absolute_vs_control' scored against `inc` instead of `result.control`
                                                        -> KILLED at the snapshot: report['grid'] differs.
    """
    before = {p: W.U.sha(p) for p in (PERSISTED_REPORT, PERSISTED_PREDS)}
    proc, report_path, preds_path = _run_main_into(tmp_path)
    assert proc.returncode == 0, proc.stderr[-4000:]
    # observable state: the persisted run is untouched and main() wrote both of its outputs
    assert {p: W.U.sha(p) for p in (PERSISTED_REPORT, PERSISTED_PREDS)} == before
    assert report_path.exists() and preds_path.exists()

    new = json.loads(report_path.read_text())
    # the wiring FIRST, so a wrong params[key] lookup fails on its own assertion, not on the snapshot below
    fits, grid = new["fits"], new["grid"]
    assert len(fits) == (len(W.grid()) + 1) * len(MB.U.MONTHS) * len(MB.U.SEEDS) == 78
    assert {f["cell"] for f in fits} == set(grid) | {W.RERUN}
    for key, cell in grid.items():
        assert key == W.cell_key(cell["m"], cell["r"])
    for f in fits:
        want = W.INCUMBENT if f["cell"] == W.RERUN else (grid[f["cell"]]["m"], grid[f["cell"]]["r"])
        assert (f["m"], f["r"]) == want, f
        assert np.isclose(f["mass_ratio"], f["r"], rtol=0, atol=1e-12), f   # float-sum rounding floor; r >= 0.15
    extra_n = {}
    for f in fits:
        extra_n.setdefault((f["m"], f["month"]), set()).add(f["extra_n"])
    assert len(extra_n) == len(W.MODULI) * len(MB.U.MONTHS)
    assert all(len(v) == 1 for v in extra_n.values()), extra_n
    mau_jan = {f["extra_n"] for f in json.loads(MB.REPORT.read_text())["fits"] if f["month"] == 1}
    assert extra_n[(20, 1)] == mau_jan == {86_322}

    # then the snapshot
    pd.testing.assert_frame_equal(pd.read_parquet(preds_path), pd.read_parquet(PERSISTED_PREDS), check_exact=True)
    old = json.loads(PERSISTED_REPORT.read_text())
    assert set(new) == set(old)
    for key in sorted(set(old) - VOLATILE_REPORT_KEYS):
        assert new[key] == old[key], f"report[{key!r}] differs from the persisted run"
    assert new["runner_sha256"] == W.U.sha(Path(W.__file__))
    assert new["prereg_file_sha256"] == W.U.sha(W.PREREG)
    assert new["prereg_file_sha256"] in W.HASHLOG.read_text().split()
    assert new["seconds"] < W.ALARM_S and new["peak_rss_gib"] < W.RSS_LIMIT_GIB
