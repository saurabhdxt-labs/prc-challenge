"""Training-time ADS-B dropout (plans/PREREG_acr_adsb_coverage_regime_2026_10_02.md): the two "not heard" states, the seeded
row choice, the knobs, and the fit wiring (fit only; predict never hides ADS-B).

Mutation rehearsal per test in its docstring and in tests/fm1/MUTATION_FM1.md (M-ACR rows).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import features as F
from prc.pipeline.fm1 import knobs as K

FLIGHT = ["join_callsign", "dt_takeoff", "first_move_rel_mvt", "n_ground"]
COLS = ["joined", *FLIGHT, "ground_seen_mvt_m10", "moving_mvt_m10", "coverage", "adsb_day_present"]


def _frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float), "ap": ["LIRF"] * 3 + ["EDDF"] * (n - 3),
                         "day": ["2025-07-01", "2025-07-01", "2025-07-02"] + ["2025-07-01"] * (n - 3),
                         "joined": [1.0, 1.0, 0.0] + [1.0] * (n - 3), "join_callsign": 1.0, "dt_takeoff": 4.0,
                         "first_move_rel_mvt": -600.0, "n_ground": 30.0, "ground_seen_mvt_m10": 12.0,
                         "moving_mvt_m10": 3.0, "coverage": 3.0, "adsb_day_present": 1.0, "q_dep": 7.0})


def _cfg(**over):
    return K.parse_config({}, overrides=[f"{k}={json.dumps(v)}" for k, v in over.items()])


# ---------------------------------------------------------------- the two states

def test_an_unheard_flight_looks_like_a_2025_unjoined_row_and_keeps_the_airport_counts():
    """FL = joined 0, coverage 1, day present 1, every flight-level column NaN; the airport-level counts and every
    non-ADS-B column unchanged; unselected rows untouched. Mutation: also blank the airport counts -> red; coverage 0
    instead of 1 -> red."""
    f = _frame()
    rows = np.array([True, False, False, True, False, False])
    F.set_unheard(f, COLS, rows, network=False)
    hit = f[rows]
    assert (hit.joined == 0).all() and (hit.coverage == 1).all() and (hit.adsb_day_present == 1).all()
    assert hit[FLIGHT].isna().all().all()
    assert (hit.ground_seen_mvt_m10 == 12).all() and (hit.moving_mvt_m10 == 3).all() and (hit.q_dep == 7).all()
    pd.testing.assert_frame_equal(f[~rows], _frame()[~rows])


def test_a_network_blackout_also_empties_the_airport_counts():
    """NB = FL + ground_seen_mvt_m10 0 and moving_mvt_m10 NaN (LIRF July 2026; LIRF January-April 2025). Mutation: NB
    leaves moving_mvt_m10 -> red."""
    f = _frame()
    rows = np.array([True, True, True, False, False, False])
    F.set_unheard(f, COLS, rows, network=True)
    hit = f[rows]
    assert (hit.ground_seen_mvt_m10 == 0).all() and hit.moving_mvt_m10.isna().all() and hit[FLIGHT].isna().all().all()
    assert (hit.joined == 0).all() and (hit.coverage == 1).all()
    assert (f.loc[~rows, "ground_seen_mvt_m10"] == 12).all()


def test_set_unheard_refuses_a_wrong_mask_or_a_frame_without_the_state_columns():
    """Mutation: drop the length check -> a misaligned mask silently hides the wrong rows -> red."""
    with pytest.raises(ValueError, match="mask"):
        F.set_unheard(_frame(), COLS, np.ones(5, bool), network=False)
    with pytest.raises(E.MissingColumnError, match="joined"):
        F.set_unheard(_frame().drop(columns="joined"), COLS, np.ones(6, bool), network=False)


# ---------------------------------------------------------------- which rows

def _big(n_days: int = 100, per_day: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = n_days * per_day
    return pd.DataFrame({"ap": np.repeat(np.array(["LIRF", "EDDF"]), n // 2),
                         "day": np.tile(np.repeat([f"d{i:03d}" for i in range(n_days // 2)], per_day), 2),
                         "joined": (rng.random(n) < 0.7).astype(float)})


def test_dropout_masks_are_seeded_disjoint_and_hit_their_shares():
    """FL picks only joined rows (~30 % of them); NB takes whole (airport, day) groups (~15 % of groups); NB rows are
    never also FL; the same seed gives the same masks, another seed different ones. Mutation: FL over all rows (not only
    joined) -> red; NB per row instead of per group -> red; no seed -> red."""
    f = _big()
    fl, nb = F.dropout_masks(f, 0.30, 0.15, seed=0)
    joined = f.joined.to_numpy() == 1
    assert not (fl & ~joined).any() and not (fl & nb).any()
    groups = f.assign(nb=nb).groupby(["ap", "day"]).nb
    assert groups.nunique().max() == 1                                   # a group is all in or all out
    assert abs(groups.first().mean() - 0.15) < 0.06
    assert abs(fl[joined & ~nb].mean() - 0.30) < 0.02
    fl2, nb2 = F.dropout_masks(f, 0.30, 0.15, seed=0)
    assert (fl == fl2).all() and (nb == nb2).all()
    fl3, _ = F.dropout_masks(f, 0.30, 0.15, seed=1)
    assert (fl != fl3).any()


def test_zero_shares_hide_nothing():
    """The default (0, 0) changes no training row. Mutation (2026-10-03): `>=` for `<` in either draw -> every group /
    joined row hidden -> red."""
    fl, nb = F.dropout_masks(_big(), 0.0, 0.0, seed=0)
    assert not fl.any() and not nb.any()


# ---------------------------------------------------------------- knobs

def test_the_dropout_knobs_are_late_off_by_default_and_refit_knobs():
    """Off by default (stored artefacts keep their hashes); a learner input, so a derived fit may not change them.
    Mutation: put them in the reuse allow-list -> red."""
    for p in ("adsb.train_dropout.flight", "adsb.train_dropout.day"):
        assert K.KNOBS[p].late and K.normalise_default(K.KNOBS[p]) == 0.0
        assert p not in K.REUSE_FREE_KNOBS
    on = _cfg(**{"adsb.train_dropout.flight": 0.3})
    assert K.reuse_blockers(on, _cfg()) == ["adsb.train_dropout.flight"]


@pytest.mark.parametrize("over, msg", [
    ({"adsb.train_dropout.flight": 1.0}, "adsb.train_dropout.flight"),
    ({"adsb.train_dropout.day": -0.1}, "adsb.train_dropout.day"),
    ({"adsb.train_dropout.flight": 0.3, "features.adsb": False, "adsb.stack.enabled": False},
     "adsb.train_dropout needs features.adsb"),
])
def test_the_dropout_knobs_refuse_bad_values(over, msg):
    """Mutation: drop the features.adsb cross-check -> red."""
    with pytest.raises(E.ConfigValueError, match=msg):
        _cfg(**over)


# ---------------------------------------------------------------- fit wiring (slow)

@pytest.fixture(scope="module")
def dropout_fits(e2e_world, fm1_world_mod):
    """(world, fit without dropout, fit with flight 0.3 / day 0.15, what the dropout fit drew and hid) — one pair of world
    fits for the slow tests below. `seen`: the (flight, network) masks dropout_masks returned and every set_unheard call
    as (rows hidden, network)."""
    from prc import api
    world = e2e_world[0]
    quiet = lambda *a, **k: None  # noqa: E731
    base = K.parse_config(fm1_world_mod.config_raw(world))
    on = base.replace(**{"adsb.train_dropout.flight": 0.3, "adsb.train_dropout.day": 0.15})
    seen = {"masks": None, "unheard": []}
    real_masks, real_unheard = F.dropout_masks, F.set_unheard

    def masks(*a, **k):
        seen["masks"] = real_masks(*a, **k)
        return seen["masks"]

    def unheard(frame, cols, rows, network):
        seen["unheard"].append((int(np.asarray(rows).sum()), network))
        return real_unheard(frame, cols, rows, network=network)
    a0 = api.fit(base, save=False, log=quiet)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(F, "dropout_masks", masks)
        mp.setattr(F, "set_unheard", unheard)
        a1 = api.fit(on, save=False, log=quiet)
    return world, a0, a1, seen


@pytest.mark.slow
def test_fit_hides_adsb_on_training_rows_only_and_reports_it(dropout_fits, monkeypatch):
    """A fit with dropout reports how many rows it hid and trains a different model; predict never hides ADS-B (the
    rows are served as they are). Mutation: skip the dropout in fit -> no report, identical model -> red; call it in
    predict -> red."""
    from prc import api
    world, a0, a1, seen = dropout_fits
    quiet = lambda *a, **k: None  # noqa: E731
    rep = a1.fit_report["adsb_dropout"]
    fl, nb = seen["masks"]
    assert rep == {"flight_share": 0.3, "day_share": 0.15, "n_flight": int(fl.sum()), "n_network": int(nb.sum())}
    assert fl.sum() > 0 and nb.sum() > 0 and not (fl & nb).any()
    assert seen["unheard"] == [(int(fl.sum()), False), (int(nb.sum()), True)]     # exactly the drawn rows were hidden
    assert "adsb_dropout" not in a0.fit_report

    def refuse(*a, **k):
        raise AssertionError("ADS-B hidden at predict time")
    monkeypatch.setattr(F, "set_unheard", refuse)
    scored = pd.read_parquet(world / "raw" / "scored.parquet")
    p0, p1 = api.predict(a0, scored, log=quiet), api.predict(a1, scored, log=quiet)
    assert (p0.TAXITIME_SEC_mvt.to_numpy() != p1.TAXITIME_SEC_mvt.to_numpy()).any()


@pytest.mark.slow
def test_regression_bug_42_a_fit_without_the_adsb_block_reads_no_coverage(e2e_world, fm1_world_mod):
    """BC-42 (2026-10-04, line-by-line audit; origin prc/pipeline/fm1/engine.py:232, introduced by commit 75943ef):
    the fix above hoisted `coverage_recorded = m.coverage...` out of the ADS-B stack branch, so every fit with a base
    model read the `coverage` column — which exists only when the ADS-B block is attached. A legal configuration
    without ADS-B (features.adsb=false, adsb.stack.enabled=false; the knob cross-checks allow it) then died with an
    unnamed AttributeError after every frame was built. Seen red 2026-10-04 before the fix. A fit without ADS-B must
    succeed, record no training coverage share and serve every scored departure."""
    from prc import api
    world = e2e_world[0]
    quiet = lambda *a, **k: None  # noqa: E731
    cfg = K.parse_config(fm1_world_mod.config_raw(world),
                         overrides=["features.adsb=false", "adsb.stack.enabled=false"])
    art = api.fit(cfg, save=False, log=quiet)
    assert art.train_covered_share == {}
    scored = pd.read_parquet(world / "raw" / "scored.parquet")
    pred = api.predict(art, scored, log=quiet)
    assert len(pred) == int((scored.PHASE_mvt == "DEP").sum())
    assert not pred.decided_by.astype(str).str.contains("adsb_stack").any()


@pytest.mark.slow
def test_the_support_gate_share_describes_the_training_data_not_the_dropout(dropout_fits):
    """End-of-session review 2026-10-03 (finding 6, owner: "fix the dropout one now"): the support gate's training
    coverage share (coverage_ratio rule) is the share of the data as recorded — the dropout hides tracks to teach the
    base model, it does not change which coverage regime the stack was trained on. Before the fix the share was read
    after set_unheard rewrote `coverage`, so a dropout fit stored a lower share and shifted the gate (seen red
    2026-10-03 09:47 EDT on the world: EDDF 0.658 / LSZH 0.503 stored vs 0.939 / 0.969 as recorded). Mutation: read the share from m.coverage after the dropout -> red."""
    _, a0, a1, _ = dropout_fits
    assert a0.train_covered_share and a1.train_covered_share == a0.train_covered_share
