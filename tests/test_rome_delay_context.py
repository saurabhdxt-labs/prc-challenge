"""Tests for RDC's harness (scripts/analysis/rome_delay_context.py; plans/PREREG_rome_delay_context_2026_09_13.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import rome_delay_context as R  # noqa: E402


def test_context_excludes_self_and_uses_an_inclusive_window():
    """Fails if the row itself is counted, if the window is exclusive at ±7,200 s, or if the late share uses '>'."""
    ids = [1, 2, 3, 4]
    t = [0.0, 7200.0, 7201.0, 100.0]
    ms = [3600.0, 0.0, 9999.0, 3600.0]
    c = R.context_features(ids, t, ms).set_index("MVT_ID_mvt")
    assert c.loc[1, "ctx_n"] == 2                      # ids 4 (t=100) and 2 (t=7200, inclusive); not itself; not 3
    assert c.loc[1, "ctx_med_ms"] == pytest.approx(1800.0, abs=1e-12)   # median of {0, 3600}
    assert c.loc[1, "ctx_late_share"] == pytest.approx(0.5, abs=1e-12)  # 3600 counts as late (>=)
    assert c.loc[3, "ctx_n"] == 2 and c.loc[2, "ctx_n"] == 3   # id 3 at 7,201: id 1 (t=0) is outside its window


def test_context_for_an_isolated_departure_is_empty_not_zero():
    c = R.context_features([1, 2], [0.0, 1e6], [0.0, 0.0])
    assert c.ctx_n.tolist() == [0.0, 0.0] and c.ctx_med_ms.isna().all() and c.ctx_late_share.isna().all()


def test_permute_within_month_keeps_each_months_values():
    f = pd.DataFrame({"month": [1, 1, 1, 7, 7], "x": [1.0, 2.0, 3.0, 70.0, 80.0]})
    p = R.permute_within_month(f, ["x"], seed=3)
    assert sorted(p.x[p.month == 1]) == [1.0, 2.0, 3.0] and sorted(p.x[p.month == 7]) == [70.0, 80.0]


def test_priced_scales_per_row_to_2026_count():
    assert R.priced(344_841.0 * 10, 100, 50) == pytest.approx(5.0, abs=1e-12)
    with pytest.raises(ValueError, match="empty population"):
        R.priced(1.0, 0, 5)


def test_minus_top_positive_removes_only_positive_rows():
    """Fails if negative rows are removed or k is ignored."""
    assert R.minus_top_positive([10.0, -5.0, 3.0], 1) == pytest.approx(-2.0, abs=1e-12)
    assert R.minus_top_positive([10.0, -5.0, 3.0], 5) == pytest.approx(-5.0, abs=1e-12)


@pytest.mark.parametrize("c, want", [
    (dict(C1=True, C2=True, C3=True, C4=True, C5=True, C6=True), "SHORTLIST"),
    (dict(C1=False, C2=True, C3=False, C4=True, C5=True, C6=True), "NOT WORKING"),
    (dict(C1=True, C2=False, C3=True, C4=True, C5=True, C6=True), "PARTIAL"),
    (dict(C1=True, C2=True, C3=True, C4=True, C5=False, C6=True), "HARNESS INVALID"),
])
def test_verdict(c, want):
    assert R.verdict(c) == want


def _synthetic():
    rng = np.random.default_rng(0)
    rows = []
    i = 0
    base = pd.Timestamp("2025-01-01 08:00", tz="UTC")
    for mo in range(1, 13):
        for k in range(40):
            fill = k % 2 == 0
            sp = float(rng.uniform(4000, 20000)) if k < 36 else 600.0     # 4 non-G1 rows per month (sp < 3,600)
            sched = base.replace(month=mo) + pd.Timedelta(hours=int(k % 6))
            mvt = sched + pd.Timedelta(seconds=sp)
            rows.append({"MVT_ID_mvt": float(i), "ADEP_mvt": "LIRF", "month": mo, "sp": sp,
                         "y": sp if fill else 900.0, "nf": 900.0, "fill": fill,
                         "SCHED_TIME_UTC_mvt": sched, "MVT_TIME_UTC_mvt": mvt, "unmatched": True, "fold": "lomo"})
            i += 1
    st = pd.DataFrame(rows)
    st["p_R"] = 0.5
    ctx = pd.DataFrame({"MVT_ID_mvt": st.MVT_ID_mvt, "ctx_n": 10.0,
                        "ctx_med_ms": np.where(st.fill, 0.0, 5000.0), "ctx_late_share": np.where(st.fill, 0.0, 1.0)})
    unm = st.drop(columns=["y", "nf", "fill", "fold", "p_R"])
    lirf = unm.iloc[:0].copy()
    return unm, lirf, st, ctx


def _stub_fit(tr, te, seed, extra_num=()):
    """C (no extra columns) knows nothing: p = 0.5. D reads the context: p = 1 - late_share."""
    if not extra_num:
        return np.full(len(te), 0.5)
    return 1.0 - te["ctx_late_share"].to_numpy(dtype="float64")


def test_run_wires_context_only_into_the_arm_and_scores_against_the_control():
    """Plant: context perfectly separates fills; the control carries no context. Fails if the control also receives the
    context, if G1 selection is wrong, or if the permuted control is not permuted."""
    unm, lirf, st, ctx = _synthetic()
    res = R.run(unm, lirf, st, ctx, fit=_stub_fit, seeds=(0,), log=lambda m: None)
    assert res["n_g1"] == 36 * 12
    assert res["auc_g1"]["D"] == 1.0 and res["auc_g1"]["C"] == 0.5
    assert res["priced_net_g1"] > 300 and res["months_positive"] == 12
    assert res["permuted_priced_net_g1"] < res["priced_net_g1"]
    assert res["guard_control_vs_stored_pR_maxabs"] == 0.0


def test_run_refuses_a_departure_without_context():
    unm, lirf, st, ctx = _synthetic()
    with pytest.raises(AssertionError, match="no context row"):
        R.run(unm, lirf, st, ctx.iloc[1:], fit=_stub_fit, seeds=(0,), log=lambda m: None)


def test_main_refuses_to_overwrite(tmp_path):
    out = tmp_path / "x.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        R.main(["--out", str(out)])


def _raw(unit: str):
    sched = pd.to_datetime(["2025-03-01 10:00:00", "2025-03-01 10:30:00", "2025-03-01 13:00:00", "2025-03-02 10:00:00"],
                           utc=True).astype(f"datetime64[{unit}, UTC]")
    mvt = (sched + pd.to_timedelta([600, 4000, 700, 800], unit="s")).astype(f"datetime64[{unit}, UTC]")
    return pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], "PHASE_mvt": "DEP", "ADEP_mvt": "LIRF",
                         "MVT_TIME_UTC_mvt": mvt, "SCHED_TIME_UTC_mvt": sched})


def test_departures_frame_seconds_do_not_depend_on_timestamp_resolution():
    """RDC run 1's defect: the raw files are datetime64[us]; astype(int64)//1e9 produced kiloseconds and a ±83-day window.
    Fails if take-off seconds depend on the stored resolution."""
    a, b = R.departures_frame([_raw("us")]), R.departures_frame([_raw("ns")])
    assert a.t_s.tolist() == b.t_s.tolist()
    assert a.t_s.iloc[1] - a.t_s.iloc[0] == pytest.approx(1800.0 + 3400.0, abs=1e-6)   # 10:30+4000 s minus 10:00+600 s
    ca = R.context_features(a.MVT_ID_mvt, a.t_s, a.ms).set_index("MVT_ID_mvt")
    assert ca.loc[1.0, "ctx_n"] == 1 and ca.loc[2.0, "ctx_n"] == 2 and ca.loc[4.0, "ctx_n"] == 0   # take-offs 10:10, 11:36:40, 13:11:40, next day


def test_context_scale_guard_refuses_a_year_wide_window():
    R.check_context_scale(pd.DataFrame({"ctx_n": [20.0, 40.0, 60.0]}))
    with pytest.raises(AssertionError, match="time unit is wrong"):
        R.check_context_scale(pd.DataFrame({"ctx_n": [15_000.0, 16_000.0]}))


def test_month_block_lower_is_a_percentile_interval_over_resampled_months():
    """Decisional for RDC C2 and DPN. Two months of 100 rows, every row +1 priced-unit in month 1 and +3 in month 2: resampling
    MONTHS gives totals {200, 400, 600} so the interval is [200, 600]; resampling ROWS would give ≈ [372, 428]. Fails if the
    resampling unit is rows or the pricing is dropped (end-work rehearsal: the row-resampling mutation survived the old test)."""
    k = 344_841.0
    g = np.concatenate([np.full(100, 1.0), np.full(100, 3.0)]) * k
    m = np.repeat([1, 2], 100)
    lo, hi = R.month_block_lower(g, m, n_rows=200, n_2026=200, n_boot=2000, seed=0)
    assert lo == pytest.approx(200.0, abs=1e-6) and hi == pytest.approx(600.0, abs=1e-6)   # exact month-sum combinations


def test_permute_within_month_actually_moves_values():
    """End-work review: the original test passed for an identity function. Fails if no value moves."""
    f = pd.DataFrame({"month": [1] * 50, "x": np.arange(50, dtype=float)})
    p = R.permute_within_month(f, ["x"], seed=0)
    assert (p.x.to_numpy() != f.x.to_numpy()).sum() > 40
