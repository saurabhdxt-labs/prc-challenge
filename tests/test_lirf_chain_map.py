"""Tests for LCM's harness (scripts/analysis/lirf_chain_map.py; plans/PREREG_lirf_chain_map_2026_09_12.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import lirf_chain_map as L  # noqa: E402


def test_cell_combines_band_and_next_day_flag():
    """Fails if the dayoff flag is inverted or the band edges are not the registered left-closed ones."""
    got = L.cell([100.0, 1_800.0, 30_000.0, 90_000.0], [0, 1, 2, 0]).tolist()
    assert got == ["0-1800|same", "1800-3600|next", "24000-86400|next", ">=86400|same"]


def test_reconcile_raises_when_parts_do_not_sum():
    L.reconcile({"a": 1.0, "b": 2.0}, 3.0)
    with pytest.raises(AssertionError, match="does not reconcile"):
        L.reconcile({"a": 1.0, "b": 2.0}, 3.1)


def test_slip_target_offers_sp_and_sp_plus_one_day():
    lo, hi = L.slip_target([1000.4, 50_000.6])
    assert lo.tolist() == [1000.0, 50_001.0] and hi.tolist() == [87_400.0, 136_401.0]


def test_oracle_saving_is_signed_and_masked():
    """Fails if the saving sign is flipped or the mask is ignored (the unmasked row would add 10,000)."""
    got = L.oracle_saving([100.0, 100.0, 100.0], [0.0, 0.0, 0.0], [100.0, 200.0, 100.0], [True, True, False])
    assert got == 10_000.0 + 0.0


def test_project_2026_prices_per_cell_mean_and_lists_unsupported_cells():
    got = L.project_2026([4.0, 8.0, 100.0], ["a", "a", "b"], {"a": 10, "b": 2, "z": 5}, n_scored=100)
    assert got["by_cell"] == {"a": pytest.approx(0.6, abs=1e-12), "b": pytest.approx(2.0, abs=1e-12)}  # 6*10/100, 100*2/100
    assert got["cells_without_2025_rows"] == ["z"]
    assert got["total"] == pytest.approx(2.6, abs=1e-12)


def _frames():
    sched = pd.Timestamp("2025-03-10 08:00", tz="UTC")
    specs = [  # (sp seconds, y, R2)
        (600.0, 700.0, 650.0),          # ordinary
        (30_000.0, 30_000.0, 2_000.0),  # same-day large sp fill: E1 floors up, RLD sets rint(sp)
        (1_000.0, 1_010.0, 800.0),      # fill (|y-sp| <= 60)
        (60_000.0, 146_400.0, 60_000.0),  # dateslip (y > 80,000), next local day
    ]
    rows = []
    for i, (sp, y, r2) in enumerate(specs):
        mvt = sched + pd.Timedelta(seconds=sp)
        rows.append({"MVT_ID_mvt": float(i), "month": 3, "sp": sp, "y": y, "R2": r2,
                     "SCHED_TIME_UTC_mvt": sched, "MVT_TIME_UTC_mvt": mvt})
    m = pd.DataFrame(rows)
    r26 = pd.DataFrame({"SCHED_TIME_UTC_mvt": [sched, sched], "MVT_TIME_UTC_mvt": [sched + pd.Timedelta(seconds=500),
                                                                                    sched + pd.Timedelta(seconds=9e4)]})
    return m, r26


def _want(m):
    e1 = L.LD.e1_floor(m.R2.to_numpy(float), m.sp.to_numpy(float))
    return float(np.sqrt(np.mean((m.y.to_numpy(float) - e1) ** 2)))


def test_run_composes_e1_then_rld_and_decomposes_exactly():
    """Fails if RLD is skipped (row 1 would keep E1's 30,000 — identical here, so check the changed count), if the class
    decomposition drops a class, or if the guard is bypassed."""
    m, r26 = _frames()
    res = L.run(m, r26, want_rmse=_want(m), want_n=4, want_n26=2)
    assert res["n_lomo"] == 4
    assert set(res["by_class"]) == {"ordinary", "fill", "dateslip"}
    assert res["by_class"]["dateslip"]["sse"] == pytest.approx((146_400.0 - 60_000.0) ** 2, rel=1e-12)
    assert sum(v["share"] for v in res["by_class"].values()) == pytest.approx(1.0, abs=1e-12)
    assert res["top_k_share"][1] == pytest.approx(res["by_class"]["dateslip"]["share"], rel=1e-12)
    # the slip's per-row target sp + 86,400 is exact: saves all of its SSE
    assert res["oracle_slip_sp_plus_day_sse_saved"] == pytest.approx((146_400.0 - 60_000.0) ** 2, rel=1e-12)
    assert res["oracle_slip_rint_sp_sse_saved"] == 0.0
    assert res["cells_2026"] == {"0-1800|same": 1, ">=86400|next": 1}
    assert res["projection_2026_by_cell"]["cells_without_2025_rows"] == [">=86400|next"]


def test_run_rld_changes_a_same_day_large_sp_row():
    """Fails if apply_rld is not composed: a same-local-day sp in [24,000, 86,400) with R2 below sp must become rint(sp)
    even when E1 would not apply (here E1 is bypassed by a y-consistent guard on a row where R2 > sp)."""
    m, r26 = _frames()
    m.loc[1, "R2"] = 40_000.0           # E1 is up-only, keeps 40,000; RLD lowers to 30,000
    res = L.run(m, r26, want_rmse=_want(m), want_n=4, want_n26=2)
    assert res["rld_rows_changed_2025"] == 1
    assert res["by_class"]["fill"]["sse"] == pytest.approx((1_010.0 - 800.0) ** 2, rel=1e-12)   # row 1 now exact


@pytest.mark.parametrize("kw, msg", [({"want_rmse": 1.0}, "guard R2E1"), ({"want_n": 5}, "expected 5 LOMO"),
                                     ({"want_n26": 3}, "expected 3 scored")])
def test_run_refuses_on_guard_failures(kw, msg):
    m, r26 = _frames()
    args = {"want_rmse": _want(m), "want_n": 4, "want_n26": 2}
    args.update(kw)
    with pytest.raises(AssertionError, match=msg):
        L.run(m, r26, **args)


def test_run_refuses_an_sp_convention_mismatch():
    m, r26 = _frames()
    m.loc[0, "sp"] = 601.0
    with pytest.raises(AssertionError, match="convention mismatch"):
        L.run(m, r26, want_rmse=_want(m), want_n=4, want_n26=2)


def test_main_refuses_to_overwrite(tmp_path):
    out = tmp_path / "x.json"
    out.write_text("{}")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        L.main(["--out", str(out)])


def _join_inputs():
    p = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "fold": ["lomo", "lomo", "A"], "y": [10.0, 20.0, 30.0],
                      "ADEP_mvt": "LIRF"})
    t = pd.Timestamp("2025-01-01", tz="UTC")
    l = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 2.0, 9.0], "ADEP_mvt": ["LIRF", "LIRF", "LSZH", "LIRF"],
                      "unmatched": [True, True, True, True], "SCHED_TIME_UTC_mvt": t, "MVT_TIME_UTC_mvt": t,
                      "y": [10.0, 20.0, 99.0, 1.0]})
    return p, l


def test_join_frames_takes_lomo_rows_and_only_lirf_unmatched_frame_rows():
    """AMENDMENT 1's defect: the times must come from LIRF UNMATCHED rows. Fails if the LIRF filter is dropped (id 2
    would duplicate with the LSZH row) or if fold A rows leak into the LOMO map."""
    p, l = _join_inputs()
    m = L.join_frames(p, l)
    assert m.MVT_ID_mvt.tolist() == [1.0, 2.0]


def test_join_frames_refuses_rows_without_a_frame_row():
    p, l = _join_inputs()
    l = l.assign(unmatched=[False, True, True, True])     # id 1 now only in a matched frame: the lirf.parquet failure
    with pytest.raises(AssertionError, match="1 R2 rows have no LIRF frame row"):
        L.join_frames(p, l)


def test_join_frames_refuses_label_disagreement():
    p, l = _join_inputs()
    l.loc[0, "y"] = 11.0
    with pytest.raises(AssertionError, match="labels differ"):
        L.join_frames(p, l)
