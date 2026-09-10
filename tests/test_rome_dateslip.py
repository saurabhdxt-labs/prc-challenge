"""scripts/rome_dateslip.py - H-DS of plans/PREREG_rome_dateslip_2026_09_10.md.

Pins: the date-slip definition; the segment G mask; the Laplace share per sub-band and the taxi
median with its fallback; that a fold's shares never read its own test months; that rows outside
G keep the base arm's prediction bit-exact; and that a planted date-slip mix is recovered.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_dateslip as ds  # noqa: E402


def _unm(rows):
    return pd.DataFrame(rows, columns=["MVT_ID_mvt", "ADEP_mvt", "month", "dayoff", "sp", "y"])


def test_dateslip_definition_is_the_registered_window():
    y = np.array([86_399.0, 86_400.0, 87_000.0, 89_999.0, 90_000.0])
    np.testing.assert_array_equal(ds.is_dateslip(y), [False, True, True, True, False])


def test_segment_mask_needs_lirf_next_day_and_sp_40k():
    """Rehearsed 2026-09-10 RED: the dayoff condition dropped."""
    f = pd.DataFrame({"ADEP_mvt": ["LIRF", "LIRF", "LIRF", "EGLL"], "dayoff": [1, 0, 1, 1],
                      "sp": [45_000.0, 45_000.0, 39_999.0, 60_000.0]})
    np.testing.assert_array_equal(ds.segment_mask(f), [True, False, False, False])


def test_band_shares_are_laplace_and_the_taxi_median_falls_back():
    """G1 = [40k, 56k): 0 of 3 date-slips -> q = 1/5; G2 = [56k, inf): 2 of 2 -> q = 3/4; m = the
    median of y - 86,400 over the date-slips (500, 1,500 -> 1,000). With no date-slip at all, m =
    1,000 s. Rehearsed 2026-09-10 RED: the +1/+2 smoothing removed (q = 0 and 1)."""
    tr = _unm([(1, "LIRF", 2, 1, 41_000.0, 41_000.0), (2, "LIRF", 2, 1, 50_000.0, 50_000.0),
               (3, "LIRF", 3, 1, 55_000.0, 55_000.0), (4, "LIRF", 3, 1, 60_000.0, 86_900.0),
               (5, "LIRF", 4, 1, 70_000.0, 87_900.0)])
    q, m = ds.band_shares(tr)
    assert q == {"G1": pytest.approx(1 / 5), "G2": pytest.approx(3 / 4)}
    assert m == pytest.approx(1_000.0)
    q0, m0 = ds.band_shares(tr[tr.y < 86_400])
    assert m0 == 1_000.0 and q0["G2"] == pytest.approx(1 / 2)          # no G2 rows left: (0+1)/(0+2)


def test_expectation_inside_g_and_bit_exact_base_outside():
    """Rehearsed 2026-09-10 RED: the outside-G rows recomputed instead of copied."""
    f = pd.DataFrame({"ADEP_mvt": ["LIRF"] * 3, "dayoff": [1, 1, 0], "sp": [60_000.0, 45_000.0, 60_000.0]})
    base = np.array([61_000.0, 45_500.0, 1_234.5678])
    out = ds.apply(f, base, q={"G1": 0.2, "G2": 0.75}, m=1_000.0)
    assert out[0] == pytest.approx(0.75 * 87_400.0 + 0.25 * 60_000.0)
    assert out[1] == pytest.approx(0.2 * 87_400.0 + 0.8 * 45_000.0)
    assert out[2] == base[2]


def test_a_fold_never_reads_its_own_test_months():
    """A date-slip planted ONLY in month 7 must not move fold A's shares (fold A trains without 1
    and 7) nor LOMO month 7's; it must move LOMO month 2's. Rehearsed 2026-09-10 RED: the training
    filter replaced by all months."""
    base_rows = [(i, "LIRF", m, 1, 60_000.0, 60_000.0) for i, m in enumerate(range(1, 13))]
    unm = _unm(base_rows + [(99, "LIRF", 7, 1, 60_000.0, 87_400.0)])
    qa, _ = ds.band_shares(ds.training_rows(unm, held=[1, 7]))
    q7, _ = ds.band_shares(ds.training_rows(unm, held=[7]))
    q2, _ = ds.band_shares(ds.training_rows(unm, held=[2]))
    assert qa["G2"] == pytest.approx(1 / 12) and q7["G2"] == pytest.approx(1 / 13)
    assert q2["G2"] == pytest.approx(2 / 14)                               # 11 base rows + the planted one


def test_planted_mix_d2_beats_s1_end_to_end():
    """Synthetic stored predictions: G2 rows are date-slips 2 times in 3, S1 predicts ~sp there
    (the incumbent's lean). D2 must lower SSE and leave every non-G row bit-exact."""
    rng = np.random.default_rng(0)
    rows, preds = [], []
    k = 0
    for m in range(1, 13):
        for j in range(6):
            sp = float(rng.uniform(56_000, 80_000))
            y = 87_000.0 + rng.uniform(0, 800) if j % 3 else sp
            rows.append((k, "LIRF", m, 1, sp, y))
            preds.append((k, "lomo", m, y, sp, sp + 500.0, sp + 400.0))
            if m in (1, 7):
                preds.append((k, "A", m, y, sp, sp + 500.0, sp + 400.0))
            k += 1
        rows.append((k, "LIRF", m, 0, 5_000.0, 900.0))
        preds.append((k, "lomo", m, 900.0, 5_000.0, 950.0, 940.0))
        k += 1
    unm = _unm(rows)
    p = pd.DataFrame(preds, columns=["MVT_ID_mvt", "fold", "month", "y", "sp", "S1", "R"])
    res = ds.run(p, unm, n_boot=200)
    assert res["lomo"]["sse"]["D2"] < res["lomo"]["sse"]["S1"]
    out = res["preds"]
    nong = ~out.in_g.to_numpy()
    np.testing.assert_array_equal(out.D2.to_numpy()[nong], out.S1.to_numpy()[nong])
    np.testing.assert_array_equal(out.R2.to_numpy()[nong], out.R.to_numpy()[nong])


def test_a_record_missing_a_fold_is_refused():
    p = pd.DataFrame({"MVT_ID_mvt": [1], "fold": ["lomo"], "month": [2], "y": [900.0], "sp": [5_000.0],
                      "S1": [950.0], "R": [940.0]})
    unm = _unm([(1, "LIRF", 2, 0, 5_000.0, 900.0)])
    with pytest.raises(ValueError, match="need both folds"):
        ds.run(p, unm, n_boot=10)
