"""FM1 knob `modules.tail_cells.enabled` (prc/pipeline/fm1/unmatched.py `not_fill_value`; registration
plans/PREREG_tc_tail_cell_routing_2026_10_05.md). Governs the unmatched body's not-copy value IN THE BODY MIXTURE
only: true (the shipped rule, default) keeps `build_submission.nf_hybrid`'s cell mean on a row whose cell estimate
is at or above T_TAIL_S; false uses the learned body there instead. A LIRF row is always decided by the `rome`
stage from S1's own hybrid (parts["nf"], never nf_cells), regardless of this knob, so this knob never changes a
LIRF value -- in the 2026 scored file, 43 unmatched rows clear T_TAIL_S and 33 of them are LIRF.

Tiers: `not_fill_value` against the REAL shipped `nf_hybrid` (fast, no world); the knob's declaration, cross-check
and config hashes (fast, no world); the knob reaching the lane through a real (but small) fit + `serve()` on
fm1_world.E2E, the smallest FM1 test world (slow).

Mutation rehearsal per test is recorded in its docstring and in tests/fm1/MUTATION_FM1.md (M-TC rows).
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import engine as G
from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import shipped
from prc.pipeline.fm1 import stages as S
from prc.pipeline.fm1 import unmatched as U

ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIGS = ("configs/fm1.yaml", "configs/fm1_v50.yaml", "configs/fm1_v55.yaml", "configs/fm1_v57.yaml")


def _cfg(**over):
    return K.parse_config({}, overrides=[f"{k}={json.dumps(v)}" for k, v in over.items()])


# ============================================================================================ not_fill_value()

def test_not_fill_value_true_is_byte_identical_to_the_real_nf_hybrid_below_at_and_above_t_tail_s():
    """tail_cells=True must delegate to the SHIPPED nf_hybrid, never a reimplementation: cells one below T_TAIL_S
    takes the body, cells AT or above keep the cell mean (nf_hybrid's own strict-below rule). Mutation: swap the
    two branches (true returns body, false delegates) -> every value below flips -> red."""
    D = shipped.unmatched_deps()
    T = D.bs.T_TAIL_S
    cells = np.array([T - 1.0, T, T + 1.0])
    body = np.array([10.0, 20.0, 30.0])
    got = U.not_fill_value(D, cells, body, tail_cells=True)
    assert np.array_equal(got, D.bs.nf_hybrid(cells, body))
    assert got.tolist() == [10.0, T, T + 1.0]


def test_not_fill_value_false_returns_the_body_on_every_row_regardless_of_cells():
    """tail_cells=False ignores the cell mean entirely, even far above T_TAIL_S. Mutation: the false branch still
    consults nf_hybrid (or cells) for some rows -> the T + 5000 row would keep its cell mean instead of 40.0 ->
    red."""
    D = shipped.unmatched_deps()
    T = D.bs.T_TAIL_S
    cells = np.array([T - 1.0, T, T + 1.0, T + 5000.0])
    body = np.array([10.0, 20.0, 30.0, 40.0])
    got = U.not_fill_value(D, cells, body, tail_cells=False)
    assert got.tolist() == [10.0, 20.0, 30.0, 40.0]
    assert got.dtype == np.dtype("float64")


def test_not_fill_value_false_refuses_a_shape_mismatch():
    """Mutation: drop the false branch's shape check -> np.asarray silently broadcasts or raises an unnamed numpy
    error -> red (a different exception type / no "shape mismatch" text)."""
    D = shipped.unmatched_deps()
    with pytest.raises(ValueError) as exc:
        U.not_fill_value(D, np.array([1.0, 2.0]), np.array([1.0]), tail_cells=False)
    assert "shape mismatch" in str(exc.value)


def test_not_fill_value_false_refuses_a_non_finite_input():
    """Mutation: drop the false branch's isfinite check -> a NaN cell silently passes through (never read, since
    tail_cells=False does not use cells for its VALUE, but the check must still fire) -> red."""
    D = shipped.unmatched_deps()
    with pytest.raises(ValueError) as exc:
        U.not_fill_value(D, np.array([1.0, np.nan]), np.array([1.0, 2.0]), tail_cells=False)
    assert "finite" in str(exc.value)


def test_not_fill_value_true_still_raises_the_real_nf_hybrids_own_refusal():
    """Proves the true branch really delegates (a stub or no-op would not raise): nf_hybrid's own finite check
    fires, with ITS wording, not a duplicate. Mutation: the true branch catches/swallows nf_hybrid's exception ->
    red."""
    D = shipped.unmatched_deps()
    with pytest.raises(ValueError, match="nf_cells and nf_fit must be finite"):
        U.not_fill_value(D, np.array([1.0, np.nan]), np.array([1.0, 2.0]), tail_cells=True)


# ==================================================================================== knob declaration + hashes

def test_knob_is_bool_default_true_late_generation_4_and_hashed():
    """Pins type, default, generation and hashed-ness together (BC-41: an unhashed late knob is refused by
    check_schema, and the generation decides which stored configs' hashes include it). Mutation: default False,
    or late 3 (collides with uad), or hashed False -> any one assertion goes red."""
    k = K.KNOBS["modules.tail_cells.enabled"]
    assert k.kind == "bool"
    assert K.normalise_default(k) is True
    assert k.late == 4
    assert k.hashed is True
    assert k.status == "implemented"
    assert k.evidence.split(" ")[0].split(":")[0] == "plans/PREREG_tc_tail_cell_routing_2026_10_05.md"


def test_cross_check_false_needs_lanes_unmatched():
    """Mutation: drop the cross-check, or guard it with the wrong knob -> red."""
    with pytest.raises(E.ConfigValueError, match="modules.tail_cells.enabled"):
        _cfg(**{"modules.tail_cells.enabled": False, "lanes.unmatched": False, "modules.rome.enabled": False,
               "modules.arc.enabled": False})


def test_cross_check_false_needs_unmatched_learner_v26():
    """Mutation: drop the cross-check -> red."""
    with pytest.raises(E.ConfigValueError, match="modules.tail_cells.enabled"):
        _cfg(**{"modules.tail_cells.enabled": False, "unmatched.learner": "hgb", "modules.rome.enabled": False,
               "modules.arc.enabled": False})


def test_false_with_a_satisfying_combination_loads():
    """Fails when either cross-check misfires on the satisfying combination (lanes.unmatched true, learner v26 —
    both the schema defaults)."""
    cfg = _cfg(**{"modules.tail_cells.enabled": False})
    assert cfg["modules.tail_cells.enabled"] is False


@pytest.mark.parametrize("relpath", CONFIGS)
def test_each_shipped_config_states_the_knob_at_its_default(relpath):
    """Every one of the four shipped configs resolves the knob to True (nothing overrides it). Mutation: one file
    accidentally sets it false (or omits a schema key so a default drifts silently) -> red."""
    cfg = K.load_config(ROOT / relpath)
    assert cfg["modules.tail_cells.enabled"] is True


@pytest.mark.parametrize("relpath", CONFIGS)
def test_turning_the_knob_off_changes_that_configs_hash(relpath):
    """The knob is late generation 4 at its default (True) in every shipped file, so its PRESENCE must not move
    any file's hash (cross-checked independently by tests/fm1/test_fm1_final_configs.py's pinned absolute
    hashes, which this change leaves untouched); turning it OFF must move the hash (proving it is actually part
    of the hashed config, not a dead parameter). Mutation: hashed=False on the knob -> this goes red (off.hash ==
    cfg.hash)."""
    cfg = K.load_config(ROOT / relpath)
    off = cfg.replace(**{"modules.tail_cells.enabled": False})
    assert off.hash != cfg.hash


# ============================================================================================ reaching the lane

@pytest.mark.slow
def test_tail_cells_knob_reaches_the_lane_through_serve(e2e_world, fm1_world_mod, monkeypatch):
    """A real (but small) fit of ONLY the unmatched lane (U.fit, not api.fit: no matched-lane cost) on
    fm1_world.E2E, served twice with the SAME fitted model under the two knob values (the knob is predict-time
    only, so one fit suffices and the two serves must reuse it bit-reproducibly — AT3/AT4, already pinned by
    test_fm1_unmatched.py). Spies on unmatched.not_fill_value: the recorded `tail_cells` argument must match each
    config. The lane's info must carry {"enabled", "n_tail_rows", "n_tail_rows_decided"} for each, with both
    counts checked against independent counts (n_tail_rows from comp["unm_nf_cells"] >= T_TAIL_S, never against
    themselves; n_tail_rows_decided from the ledger's own decided_of on exactly those ids, the same accessor
    prc/pipeline/fm1/uad.py's `serve` uses for its SCOPE check) -- never a circular self-comparison. Every row
    whose cell mean is below T_TAIL_S must be BIT-IDENTICAL between the two knob values (nf_hybrid already sends
    those rows to the body regardless of the knob).

    fm1_world.E2E is tiny and its fitted cell means are not guaranteed to clear T_TAIL_S (3,000 s) on their own, so
    this test forces one: it wraps the process-cached urp_replay module's own `s1_parts` (monkeypatched on the
    object `shipped.urp_replay()` returns -- the SAME one `serve()` calls through, since analysis_script caches it
    once per process) to set one non-LIRF row's nf_cells to 3,500.0 right after the real fit's s1_parts call,
    leaving every other row exactly as the real fit produced it. This also lets the test assert the knob's
    concrete effect on that one forced row: with the knob false its body-stage value is rint(mixture(p, sp,
    body)); with it true, rint(mixture(p, sp, 3500.0)); the two must differ, and the row must still be decided
    unmatched_body or storm under both settings (never claimed by LIRF/ARC), proving it is one of the rows
    n_tail_rows_decided counts.

    Mutation: serve() passes a literal True/False instead of cfg["modules.tail_cells.enabled"] -> the False run's
    recorded argument is True -> red; drop the info recording -> the info assertions go red; a stray extra
    not_fill_value branch that also moves below-T_TAIL_S rows -> the bit-identical assertion goes red;
    body_values calls not_fill_value and then throws the result away, using D.nf_hybrid(parts["nf_cells"], mean)
    directly -> the spy still records [True, False] (so that assertion alone would NOT catch this one) but the
    forced row's knob-false value stops differing from its knob-true value -> red."""
    raw = fm1_world_mod.config_raw(e2e_world[0])
    cfg_on = K.parse_config(raw)
    assert cfg_on["modules.tail_cells.enabled"] is True            # the shipped default, confirmed before flipping it
    cfg_off = cfg_on.replace(**{"modules.tail_cells.enabled": False})

    files = dict(sorted(G.training_files(cfg_on).items()))
    model = U.fit(cfg_on, files, cfg_on["run.seeds"], log=lambda *a, **k: None)
    scored = pd.read_parquet(e2e_world[0] / "raw" / "scored.parquet")

    calls = []
    real_not_fill_value = U.not_fill_value

    def spy(D, nf_cells, body, tail_cells):
        calls.append(tail_cells)
        return real_not_fill_value(D, nf_cells, body, tail_cells)

    monkeypatch.setattr(U, "not_fill_value", spy)

    # force one non-LIRF row's nf_cells to 3,500.0 (>= T_TAIL_S), the boundary case the knob exists for, so this
    # tiny world is guaranteed to have at least one such row regardless of what the real fit happened to produce.
    urp_mod = shipped.urp_replay()
    real_s1_parts = urp_mod.s1_parts
    forced = {}

    def forced_s1_parts(D, train_unm, train_lirf, test, seeds, held):
        s1, parts = real_s1_parts(D, train_unm, train_lirf, test, seeds, held)
        non_lirf = (test.ADEP_mvt.astype(str) != U.ROME).to_numpy()
        assert non_lirf.any(), "the world has no non-LIRF row to force a tail cell onto"
        idx = int(np.flatnonzero(non_lirf)[0])
        parts = dict(parts)
        nf_cells = np.array(parts["nf_cells"], dtype="float64", copy=True)
        nf_cells[idx] = 3500.0
        parts["nf_cells"] = nf_cells
        forced["mvt_id"] = float(test.MVT_ID_mvt.to_numpy()[idx])
        return s1, parts

    monkeypatch.setattr(urp_mod, "s1_parts", forced_s1_parts)

    plan_on = [s.name for s in S.plan(cfg_on)]
    plan_off = [s.name for s in S.plan(cfg_off)]
    assert plan_on == plan_off          # the knob changes a VALUE inside unmatched_body, never the stage plan
    stages_map = {s.name: s for s in S.plan(cfg_on)}
    ids = U.unmatched_ids(scored)

    led_on, led_off = S.Ledger(ids), S.Ledger(ids)
    comp_on, info_on, _ = U.serve(model, scored, cfg_on, led_on, stages_map, log=lambda *a, **k: None)
    comp_off, info_off, _ = U.serve(model, scored, cfg_off, led_off, stages_map, log=lambda *a, **k: None)

    assert calls == [True, False]
    assert set(info_on["tail_cells"]) == {"enabled", "n_tail_rows", "n_tail_rows_decided"}
    assert set(info_off["tail_cells"]) == {"enabled", "n_tail_rows", "n_tail_rows_decided"}
    assert info_on["tail_cells"]["enabled"] is True and info_off["tail_cells"]["enabled"] is False

    cells = comp_on["unm_nf_cells"].to_numpy(dtype="float64")
    T = shipped.unmatched_deps().bs.T_TAIL_S
    assert np.array_equal(cells, comp_off["unm_nf_cells"].to_numpy(dtype="float64"))      # same input, either way
    tail = cells >= T
    assert tail.sum() >= 1, "the forced row must clear T_TAIL_S"
    expect_n_tail_rows = int(tail.sum())                                      # independent of info["tail_cells"]
    assert info_on["tail_cells"]["n_tail_rows"] == expect_n_tail_rows
    assert info_off["tail_cells"]["n_tail_rows"] == expect_n_tail_rows
    assert isinstance(info_on["tail_cells"]["n_tail_rows"], int)

    tail_ids = ids[tail]
    expect_decided_on = int(np.isin(led_on.decided_of(tail_ids), ["unmatched_body", "storm"]).sum())
    expect_decided_off = int(np.isin(led_off.decided_of(tail_ids), ["unmatched_body", "storm"]).sum())
    assert info_on["tail_cells"]["n_tail_rows_decided"] == expect_decided_on
    assert info_off["tail_cells"]["n_tail_rows_decided"] == expect_decided_off
    assert isinstance(info_on["tail_cells"]["n_tail_rows_decided"], int)

    below = cells < T
    assert below.any(), "the world has no cell below T_TAIL_S to prove the knob leaves those rows alone"
    np.testing.assert_array_equal(comp_on["unmatched_pred"].to_numpy()[below],
                                  comp_off["unmatched_pred"].to_numpy()[below])
    np.testing.assert_array_equal(comp_on["unm_body"].to_numpy(), comp_off["unm_body"].to_numpy())

    # the forced row itself: knob false keeps the learned body in the mixture, knob true keeps the forced cell
    # mean (3,500.0); the two must differ, and the row must still be decided unmatched_body/storm under both
    # settings (never LIRF/ARC) -- the concrete case the whole knob exists for, and exactly what
    # n_tail_rows_decided is counting above.
    forced_id = forced["mvt_id"]
    assert forced_id in tail_ids.tolist()
    assert led_on.decided_of(np.array([forced_id]))[0] in ("unmatched_body", "storm")
    assert led_off.decided_of(np.array([forced_id]))[0] in ("unmatched_body", "storm")
    srow = scored.set_index("MVT_ID_mvt").loc[forced_id]
    sp_f = (srow["MVT_TIME_UTC_mvt"] - srow["SCHED_TIME_UTC_mvt"]).total_seconds()
    p_f = float(comp_off.loc[forced_id, "unm_p_fill"])
    body_f = float(comp_off.loc[forced_id, "unm_body"])
    assert comp_on.loc[forced_id, "unm_body"] == pytest.approx(body_f)        # the body itself ignores the knob
    D_test = shipped.unmatched_deps()
    want_false = float(np.rint(D_test.mixture(np.array([p_f]), np.array([sp_f]), np.array([body_f])))[0])
    want_true = float(np.rint(D_test.mixture(np.array([p_f]), np.array([sp_f]), np.array([3500.0])))[0])
    assert want_false != want_true, "the forced cell mean and the body must round to different mixture values"
    assert float(np.rint(comp_off.loc[forced_id, "unmatched_pred"])) == want_false
    assert float(np.rint(comp_on.loc[forced_id, "unmatched_pred"])) == want_true
