"""scripts/rome_body.py - arm B of plans/PREREG_rome_body_2026_09_10.md: the conditioned body model.

Pins: the non-fill fold restricts training / fit / early-stopping rows and never the holdout; the
arm assembly (B = mixture with the conditioned body, B_hyb = B only where p >= 0.5) and that every
non-LIRF row is out of scope; and that a planted body improvement is recovered by the scoring.
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rome_body as rb  # noqa: E402


def _fold():
    y = np.array([1000.0, 5000.0, 900.0, 7000.0, 800.0, 1200.0])
    sp = np.array([200.0, 5010.0, 100.0, 6950.0, 50.0, 1210.0])       # rows 1, 3 (training) and 5 (holdout) are fills
    return {"y": y, "sp": sp, "te": np.array([0, 0, 0, 0, 1, 1], bool), "tr": np.array([1, 1, 1, 1, 0, 0], bool),
            "fit": np.array([1, 1, 0, 0, 0, 0], bool), "es": np.array([0, 0, 1, 1, 0, 0], bool), "X": None}


def test_nonfill_fold_drops_fill_rows_from_training_and_stopping_only():
    """The holdout keeps its fill row (row 5). Rehearsed 2026-09-10: restricting te too SURVIVED the
    first version of this fixture, which had no fill row in the holdout; row 5 is that row."""
    f = _fold()
    nf, fill = rb.nonfill_fold(f)
    np.testing.assert_array_equal(fill, [False, True, False, True, False, True])
    np.testing.assert_array_equal(nf["tr"], [1, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(nf["fit"], [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(nf["es"], [0, 0, 1, 0, 0, 0])
    np.testing.assert_array_equal(nf["te"], f["te"])
    assert nf["y"] is f["y"] and f["tr"].sum() == 4                    # the input fold is not mutated


def test_arms_are_the_registered_formulas():
    """B = max(p sp + (1-p) body, 1); B_hyb = B where p >= 0.5 else F. Rehearsed 2026-09-10 RED:
    the gate written p > 0.5 (the boundary row keeps F)."""
    sp, F, body = np.array([5000.0, 800.0, 900.0]), np.array([3000.0, 850.0, 700.0]), np.array([1000.0, 820.0, 760.0])
    p = np.array([0.9, 0.1, 0.5])
    arms = rb.assemble(p, sp, F, body)
    np.testing.assert_allclose(arms["B"], np.maximum(p * sp + (1 - p) * body, 1.0), rtol=0, atol=0)
    np.testing.assert_allclose(arms["B_hyb"], [arms["B"][0], F[1], arms["B"][2]], rtol=0, atol=0)


def test_scoring_recovers_a_planted_body_improvement():
    """F hedges on fills AND is noisy on the body; the conditioned body is exact on body rows and p
    is sharp: B must win, and its body clause must hold."""
    rng = np.random.default_rng(0)
    n = 4000
    fill = rng.random(n) < 0.2
    sp = rng.uniform(600, 6000, n)
    y = np.where(fill, sp, rng.uniform(500, 1500, n))
    F = np.where(fill, 0.5 * sp + 0.5 * 1000, y + rng.normal(0, 150, n))
    body = np.where(fill, 1000.0, y + rng.normal(0, 60, n))
    p = np.clip(np.where(fill, 0.95, 0.03) + rng.normal(0, 0.01, n), 0, 1)
    rec = pd.DataFrame({"month": np.where(np.arange(n) % 2, 1, 7), "y": y, "sp": sp, "F": F})
    arms = {"F": F, **rb.assemble(p, sp, F, body)}
    for s in (0, 1):
        for k, v in rb.assemble(np.clip(p + 0.01 * s, 0, 1), sp, F, body).items():
            arms[f"{k}_seed{s}"] = v
    res = rb.rm.score(rec, arms, ("B", "B_hyb"), (0, 1), n_boot=200)
    assert res["rmse"]["B"] < res["rmse"]["F"] and res["clauses"]["B"]["c4_body_bounded"] is True
    assert res["verdict"]["B"] == "ESTABLISHED"


def test_assemble_all_is_the_gated_mixture():
    """B_all = max(p sp + (1-p) body, 1) where p >= 0.5, F elsewhere."""
    sp, F, body = np.array([5000.0, 800.0]), np.array([3000.0, 850.0]), np.array([1000.0, 820.0])
    p = np.array([0.8, 0.49])
    out = rb.assemble_all(p, sp, F, body)["B_all"]
    np.testing.assert_allclose(out, [max(0.8 * 5000 + 0.2 * 1000, 1.0), 850.0], rtol=0, atol=0)


def test_all_airport_p_trains_each_airport_on_its_own_training_rows(monkeypatch):
    """Each airport's classifier sees ONLY that airport's matched rows of the non-holdout months,
    and each returned p lands on that airport's holdout rows. Rehearsed 2026-09-10 RED: the
    airport filter dropped from the training frame."""
    frame = pd.DataFrame({"MVT_ID_mvt": np.arange(12.0), "ADEP_mvt": ["AAAA"] * 6 + ["BBBB"] * 6,
                          "month": [1, 2, 3, 7, 2, 3] * 2, "unmatched": [False] * 11 + [True]})
    rec = pd.DataFrame({"MVT_ID_mvt": [0.0, 3.0, 6.0, 9.0], "ap": ["AAAA", "AAAA", "BBBB", "BBBB"]})
    seen = []

    def fake(tr, te, seed, permute=False, extra_num=()):
        seen.append((set(tr.ADEP_mvt), set(tr.month), bool(tr.unmatched.any()), list(te.MVT_ID_mvt)))
        return np.full(len(te), 0.1 if set(tr.ADEP_mvt) == {"AAAA"} else 0.9)
    monkeypatch.setattr(rb.rf, "fit_predict_fill", fake)
    ps = rb.all_airport_p(frame, rec, seeds=(0,))
    np.testing.assert_allclose(ps[0], [0.1, 0.1, 0.9, 0.9])
    for aps, months, has_unm, _ in seen:
        assert len(aps) == 1 and months <= {2, 3} and not has_unm


def test_score_default_bars_are_arm_m_and_overrides_apply():
    rng = np.random.default_rng(1)
    n = 2000
    y = rng.uniform(500, 1500, n)
    sp = y + 5000.0                                   # no fill rows
    rec = pd.DataFrame({"month": np.where(np.arange(n) % 2, 1, 7), "y": y, "sp": sp})
    F = y + rng.normal(0, 100, n)
    A = y + rng.normal(0, 99, n)                      # a tiny, real-ish improvement
    arms = {"F": F, "A": A, "A_seed0": A, "A_seed1": A + 0.01}
    r5 = rb.rm.score(rec, arms, ("A",), (0, 1), n_boot=100)
    r0 = rb.rm.score(rec, arms, ("A",), (0, 1), n_boot=100, bar=-1e9, body_tol=1e9)
    assert r5["clauses"]["A"]["c2_gain_ge_5s"] is False and r0["clauses"]["A"]["c2_gain_ge_5s"] is True
    assert r0["clauses"]["A"]["c4_body_bounded"] is True


def test_assemble_all_includes_the_continuous_b2_arm():
    """Amendment B.2: B_all_cont = max(p sp + (1-p) body, 1) on every row, no gate.
    Rehearsed 2026-09-10 RED: B_all_cont written with the 0.5 gate (identical to B_all)."""
    sp, F, body = np.array([5000.0, 800.0]), np.array([3000.0, 850.0]), np.array([1000.0, 820.0])
    p = np.array([0.8, 0.2])
    arms = rb.assemble_all(p, sp, F, body)
    np.testing.assert_allclose(arms["B_all_cont"], np.maximum(p * sp + (1 - p) * body, 1.0), rtol=0, atol=0)
    assert arms["B_all"][1] == 850.0 and arms["B_all_cont"][1] != 850.0
