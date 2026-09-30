"""Tests for the glue in scripts/analysis/mtx_run.py (arm MTX). Mutation rehearsal recorded at the bottom."""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("mtx_run", ROOT / "scripts" / "analysis" / "mtx_run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mtx_run"] = mod
    spec.loader.exec_module(mod)
    return mod


mr = _load()


def _pub(months=range(1, 13), ap="LTFM"):
    return pd.DataFrame({"ap": ap, "year": 2025, "month": list(months), "TF": 3.0, "VALID": 3.0, "NR": 0.0,
                         "P": 3.0, "INV": 0.0, "pub_mean_s": 100.0, "ref_mean_s": 50.0})


def test_label_cells_groups_by_local_month_and_drops_other_years():
    """LTFM is UTC+3: 2024-12-31 22:00 UTC is Jan-2025 local (kept, Jan); 2025-12-31 22:00 UTC is 2026 (dropped)."""
    dep = pd.DataFrame({"ADEP_mvt": ["LTFM"] * 3,
                        "MVT_TIME_UTC_mvt": pd.to_datetime(["2024-12-31 22:00", "2025-01-15 10:00", "2025-12-31 22:00"], utc=True),
                        "TAXITIME_SEC_mvt": [100.0, 300.0, 999.0]})
    c = mr.label_cells(dep, _pub())
    assert c[["ap", "month", "n", "mean_label"]].values.tolist() == [["LTFM", 1, 2, 200.0]]


def test_label_cells_refuses_a_cell_without_a_published_row():
    dep = pd.DataFrame({"ADEP_mvt": ["LTFM"], "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-03-10 10:00"], utc=True),
                        "TAXITIME_SEC_mvt": [100.0]})
    with pytest.raises(ValueError, match="unjoined"):
        mr.label_cells(dep, _pub(months=[1, 2]))


def _cells():
    rng = np.random.default_rng(3)
    n = rng.integers(1000, 2000, 12).astype(float)
    nr = rng.integers(10, 90, 12).astype(float)
    inv = rng.integers(0, 300, 12).astype(float)
    pub = rng.uniform(700, 900, 12)
    return pd.DataFrame({"ap": "EDDM", "month": np.arange(1, 13), "n": n, "TF": n, "NR": nr, "INV": inv,
                         "pub_mean_s": pub, "mean_label": pub + (nr * 30 + inv * 800) / n + rng.normal(0, 3, 12)})


def test_lmo_params_never_see_the_scored_month_labels():
    """Kills leakage into T (c, e) and into sigma (the nested LMO errors)."""
    d = _cells()
    c, e, lmo = mr.lmo_params(d, 1)
    d2 = d.copy()
    d2.loc[d2.month == 1, "mean_label"] += 1e4
    c2, e2, lmo2 = mr.lmo_params(d2, 1)
    assert (c, e) == (c2, e2)
    pd.testing.assert_frame_equal(lmo, lmo2)
    assert 1 not in lmo.month.tolist()


def test_calibrate_moves_each_cell_by_the_shrunk_gap_and_runway_cells_override():
    """Two cells; lmo errors give sigma 5; gap 10 -> lambda 0.75. Runway variant: rows on a published runway get
    their own target (pub_r + excess), rows on an unpublished runway keep the airport shift."""
    fr = pd.DataFrame({"ap": ["EDDM"] * 4 + ["EGLL"], "month": [1, 1, 1, 1, 7], "pred": [100.0, 100.0, 100.0, 100.0, 50.0],
                       "rwy": ["08L", "08L", "26R", "26R", "27L"]})
    pub = pd.DataFrame({"ap": ["EDDM", "EGLL"], "month": [1, 7], "TF": [10.0, 10.0], "NR": [0.0, 0.0],
                        "INV": [0.0, 0.0], "pub_mean_s": [110.0, 50.0]})
    lmo = pd.DataFrame({"inv_share": [0.0, 0.0], "err": [5.0, -5.0]})
    params = {("EDDM", 1): (0.0, 0.0, lmo), ("EGLL", 7): (0.0, 0.0, lmo)}
    out, rep = mr.calibrate(fr, pub, params, runway=None, mode="add")
    assert out.tolist() == pytest.approx([107.5, 107.5, 107.5, 107.5, 50.0])
    assert rep["EDDM|1"]["lam"] == pytest.approx(0.75) and rep["EGLL|7"]["delta"] == 0.0
    rw = pd.DataFrame({"ap": ["EDDM"], "month": [1], "runway": ["08L"], "pub_mean_s": [120.0]})
    out_r, _ = mr.calibrate(fr, pub, params, runway=rw, mode="add")
    lam_r = 1 - 25 / 400
    assert out_r.tolist() == pytest.approx([100 + 20 * lam_r, 100 + 20 * lam_r, 107.5, 107.5, 50.0])
    out_f, _ = mr.calibrate(fr, pub, params, runway=None, mode="add", full_lambda=True)
    assert out_f.tolist() == pytest.approx([110.0] * 4 + [50.0])

# Mutation rehearsal (2026-09-28, manual): (1) label_cells grouping by UTC month -> test 1 RED; (2) lmo_params using
# the full cells for (c, e) -> test 3 RED; (3) calibrate full_lambda ignored -> test 4 RED; (4) runway rows not
# overriding (np.where branches swapped) -> test 4 RED. Results in the MTX session log.
