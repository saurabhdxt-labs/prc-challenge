"""FM1 UAD (prc/pipeline/fm1/uad.py): the aircraft's own ADS-B ground track as the off-block estimate for an unmatched
departure without a flight-plan record (plans/PREREG_uad_unmatched_adsb_offblock_2026_10_04.md, REG 17a570a6). Runs
after the unmatched lane's storm anchor and before LDI.

Tiers: the knobs, the cross-checks and the plan; `blend` on hand numbers; a differential test against the registered
reference (scripts/analysis/uad_price.py); `serve` on constructed ledgers/tables/comp frames; `load_constants` /
`load_table` refusals; `apply` from a config; the e2e world through api.predict (slow); engine.build_inputs.

Mutation rehearsal per test is recorded in tests/fm1/MUTATION_FM1.md (M-UAD rows).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import types

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import artifact as A
from prc.pipeline.fm1 import engine as G
from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import stages as S
from prc.pipeline.fm1 import uad as U

ROOT = pathlib.Path(__file__).resolve().parents[2]
STAGE = {s.name: s for s in S.STAGE_ORDER}
NAN = float("nan")
#: same registered constants a flat load_constants() would hand serve(); registration is a free label here
CONST = {"by_airport": {"EDDF": 50.0}, "pooled": 70.0, "w0": 0.5, "w1": 0.9, "registration": "test-reg"}


def _cfg(**over):
    return K.parse_config({}, overrides=[f"{k}={json.dumps(v)}" for k, v in over.items()])


def _mk_rows(specs) -> pd.DataFrame:
    """specs: [(id, ap, mvt_iso, sched_iso), ...] -> the raw columns serve() reads (ADEP_mvt, the two clocks)."""
    return pd.DataFrame({"MVT_ID_mvt": [s[0] for s in specs], "ADEP_mvt": [s[1] for s in specs],
                        "MVT_TIME_UTC_mvt": pd.to_datetime([s[2] for s in specs], utc=True),
                        "SCHED_TIME_UTC_mvt": pd.to_datetime([s[3] for s in specs], utc=True)})


def _mk_table(specs) -> pd.DataFrame:
    """specs: [(id, unmatched, cls, exit_rel_s), ...]."""
    if not specs:
        return pd.DataFrame({"MVT_ID_mvt": pd.Series(dtype="float64"), "unmatched": pd.Series(dtype=bool),
                             "cls": pd.Series(dtype=object), "exit_rel_s": pd.Series(dtype="float64")})
    return pd.DataFrame({"MVT_ID_mvt": [s[0] for s in specs], "unmatched": [s[1] for s in specs],
                        "cls": [s[2] for s in specs], "exit_rel_s": [s[3] for s in specs]})


def _mk_comp(specs) -> pd.DataFrame:
    """specs: [(id, unm_p_fill, storm_A), ...]."""
    comp = pd.DataFrame({"MVT_ID_mvt": [s[0] for s in specs]}).set_index("MVT_ID_mvt")
    comp["unm_p_fill"] = [s[1] for s in specs]
    comp["storm_A"] = [s[2] for s in specs]
    return comp


def _write_constants(path: pathlib.Path, overrides: dict | None = None) -> pathlib.Path:
    base = {"offsets": {"by_airport": {"EDDF": 40.0}, "pooled": 60.0}, "weights": {"w0": 0.5, "w1": 0.8},
           "registration": "reg-test"}
    for dotted, v in (overrides or {}).items():
        node = base
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = v
    path.write_text(json.dumps(base))
    return path


# ======================================================================================================= knobs + plan

def test_knobs_default_to_off_and_the_registered_paths_and_are_late_generation_3():
    """Fails when a default drifts from the registered table path / constants file, or enabled defaults True."""
    assert K.normalise_default(K.KNOBS["modules.uad.enabled"]) is False
    assert K.normalise_default(K.KNOBS["modules.uad.table"]) == "data/uad/2026_table_a3/table.parquet"
    assert K.normalise_default(K.KNOBS["modules.uad.constants"]) == "configs/uad_constants_2025.json"
    for p in ("modules.uad.enabled", "modules.uad.table", "modules.uad.constants"):
        assert K.KNOBS[p].late == 3 and K.KNOBS[p].hashed is True and K.KNOBS[p].status == "implemented"


def test_uad_knobs_leave_the_default_config_hash_unchanged():
    """The three knobs are late, so an all-defaults config hashes exactly as it did before this module existed
    (test_fm1_eobt_clip.DEFAULTS_HASH_BEFORE_LATE_KNOBS, duplicated here as an independent cross-check). Fails when a
    knob here is not marked late: its default would then always enter the hash, moving every stored config's hash the
    moment this module is added."""
    assert K.parse_config({}).hash == "ad0af3a77a716bce8361486ad0a26d7acd6abee3f027f106ef106fa7535afe32"
    on = K.parse_config({"modules": {"uad": {"enabled": True}, "storm": {"mode": "anchor"}}})
    assert on.hash != K.parse_config({}).hash


def test_cross_check_enabled_needs_lanes_unmatched():
    """Fails when the cross-check is dropped or guarded by the wrong knob."""
    with pytest.raises(E.ConfigValueError, match="modules.uad.enabled needs lanes.unmatched"):
        _cfg(**{"modules.uad.enabled": True, "lanes.unmatched": False, "modules.rome.enabled": False,
               "modules.arc.enabled": False})


def test_cross_check_enabled_needs_unmatched_learner_v26():
    """Fails when the cross-check is dropped."""
    with pytest.raises(E.ConfigValueError, match="modules.uad.enabled needs unmatched.learner v26"):
        _cfg(**{"modules.uad.enabled": True, "unmatched.learner": "hgb", "modules.rome.enabled": False,
               "modules.arc.enabled": False})


def test_cross_check_enabled_needs_storm_anchor():
    """Fails when the cross-check is dropped, or checks modules.storm.feature instead of modules.storm.mode."""
    with pytest.raises(E.ConfigValueError, match="modules.uad.enabled needs modules.storm.mode anchor"):
        _cfg(**{"modules.uad.enabled": True})


def test_a_valid_enabled_config_loads():
    """Fails when any cross-check misfires on the satisfying combination."""
    cfg = _cfg(**{"modules.uad.enabled": True, "modules.storm.mode": "anchor"})
    assert cfg["modules.uad.enabled"] is True and cfg["modules.uad.table"] and cfg["modules.uad.constants"]


def test_plan_places_uad_after_storm_and_before_ldi():
    """Fails when uad maps to the wrong knob in plan()'s `on` dict, or STAGE_ORDER's gap is in the wrong place."""
    on = [s.name for s in S.plan(_cfg(**{"modules.uad.enabled": True, "modules.storm.mode": "anchor",
                                         "modules.ldi.enabled": True}))]
    i = {n: idx for idx, n in enumerate(on)}
    assert i["storm"] < i["uad"] < i["ldi"]
    off = [s.name for s in S.plan(_cfg())]
    assert "uad" not in off


# ============================================================================================================= blend

def test_blend_calm_vs_storm_weight_storm_a_exactly_300_is_calm():
    """storm_A == 300.0 exactly must take w0 (the reference's `> 300` comparison, never `>=`). Fails when the
    threshold comparison in blend is changed to `>=`."""
    v, est, c, cls = np.array([1000.0]), np.array([500.0]), np.array([0.0]), np.array(["A1"])
    p = sp = np.array([NAN])
    calm = U.blend(v, est, p, sp, np.array([300.0]), cls, c, w0=0.2, w1=0.8)
    storm = U.blend(v, est, p, sp, np.array([300.0001]), cls, c, w0=0.2, w1=0.8)
    assert calm.tolist() == [900.0]
    assert storm.tolist() == [600.0]


def test_blend_nan_anomaly_is_calm():
    """Fails when a NaN anomaly is treated as storm (e.g. `np.nan_to_num` defaulting it to a large number)."""
    v, est, c, cls = np.array([1000.0]), np.array([500.0]), np.array([0.0]), np.array(["A1"])
    p = sp = np.array([NAN])
    nan_a = U.blend(v, est, p, sp, np.array([NAN]), cls, c, w0=0.2, w1=0.8)
    calm = U.blend(v, est, p, sp, np.array([300.0]), cls, c, w0=0.2, w1=0.8)
    assert nan_a.tolist() == calm.tolist() == [900.0]


def test_blend_a2_takes_half_the_weight():
    """A2's shift is exactly half of the A1 shift under identical inputs. Fails when the A2 factor is dropped or
    applied to the wrong term (e.g. halving v instead of w)."""
    v, est, c = np.array([1000.0]), np.array([500.0]), np.array([0.0])
    p = sp = np.array([NAN])
    a1 = U.blend(v, est, p, sp, np.array([301.0]), np.array(["A1"]), c, w0=0.2, w1=0.8)
    a2 = U.blend(v, est, p, sp, np.array([301.0]), np.array(["A2"]), c, w0=0.2, w1=0.8)
    assert a1.tolist() == [600.0] and a2.tolist() == [800.0]
    assert (a2[0] - v[0]) == pytest.approx((a1[0] - v[0]) / 2)


def test_blend_nan_p_or_nan_sp_falls_back_to_u():
    """Fails when the fallback is dropped (a NaN propagates through p*sp+(1-p)*u instead of being replaced by u)."""
    v, est, c, cls = np.array([1000.0] * 3), np.array([500.0] * 3), np.array([0.0] * 3), np.array(["A1"] * 3)
    p = np.array([0.5, NAN, 0.5])
    sp = np.array([1200.0, 1200.0, NAN])
    storm_a = np.array([0.0] * 3)
    got = U.blend(v, est, p, sp, storm_a, cls, c, w0=1.0, w1=1.0)       # w = 1 isolates u' itself
    assert got[0] == pytest.approx(850.0)        # both defined: the schedule blend 0.5*1200 + 0.5*500
    assert got[1] == pytest.approx(500.0)        # NaN p -> fallback to u = est + c
    assert got[2] == pytest.approx(500.0)        # NaN sp -> fallback to u


def test_blend_floors_at_one_second():
    """Fails when the floor at 1 s is dropped (a negative or zero result reaches rint unclamped)."""
    got = U.blend(np.array([5.0]), np.array([-100.0]), np.array([NAN]), np.array([NAN]), np.array([0.0]),
                 np.array(["A1"]), np.array([0.0]), w0=1.0, w1=1.0)
    assert got.tolist() == [1.0]


def test_blend_rounds_half_to_even_like_rint():
    """2.5 and 3.5 both round to the nearest EVEN integer (np.rint's banker's rounding), not round-half-up. Fails
    when blend rounds with e.g. `np.floor(x + 0.5)` instead of `np.rint`."""
    v, est, c, cls = np.array([2.0, 3.0]), np.array([3.0, 4.0]), np.array([0.0, 0.0]), np.array(["A1", "A1"])
    p = sp = np.array([NAN, NAN])
    got = U.blend(v, est, p, sp, np.array([0.0, 0.0]), cls, c, w0=0.5, w1=0.9)
    assert got.tolist() == [2.0, 4.0]            # 2.5 -> 2 (even), 3.5 -> 4 (even)


def test_blend_refuses_a_cls_outside_a1_a2():
    """Fails when blend silently treats an unrecognised class (e.g. '') as A1 instead of refusing."""
    with pytest.raises(E.SchemaError, match="cls must be one of"):
        U.blend(np.array([1.0]), np.array([1.0]), np.array([NAN]), np.array([NAN]), np.array([0.0]),
               np.array([""]), np.array([0.0]), w0=0.5, w1=0.5)


def test_blend_matches_the_reference_uad_price_apply_exactly():
    """scripts/analysis/uad_price.py's apply() is the registered value rule; blend must reproduce it bit for bit on a
    seeded random frame (several airports incl. one missing from by_airport, both classes, calm and storm rows, NaN p
    and NaN sp), with the airport offset resolved the way serve resolves it. Fails when any operation in blend is
    reordered, re-cast or rounded differently from the reference."""
    spec = importlib.util.spec_from_file_location("uad_price", ROOT / "scripts" / "analysis" / "uad_price.py")
    uad_price = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uad_price)

    rng = np.random.default_rng(20261005)
    n = 400
    aps = np.array(["EDDF", "EGLL", "LIRF", "ZZZZ"])            # ZZZZ: deliberately missing from by_airport
    ap = rng.choice(aps, n)
    cls = rng.choice(["A1", "A2"], n)
    v = rng.uniform(100.0, 2000.0, n)
    est = rng.uniform(-500.0, 3000.0, n)
    sp = rng.uniform(-200.0, 4000.0, n)
    p = rng.uniform(0.0, 1.0, n)
    storm_a = rng.uniform(0.0, 1000.0, n)
    nan_p = rng.choice(n, n // 10, replace=False)
    p[nan_p] = np.nan
    remaining = np.setdiff1d(np.arange(n), nan_p)
    nan_sp = rng.choice(remaining, n // 10, replace=False)
    sp[nan_sp] = np.nan
    storm = storm_a > uad_price.STORM_A_S
    frame = pd.DataFrame({"v": v, "p": p, "sp": sp, "est": est, "storm": storm, "cls": cls, "scope": True, "ap": ap})
    by_airport = {"EDDF": 40.0, "EGLL": 184.0, "LIRF": 254.0}
    pooled = 64.0
    const_nested = {"offsets": {"by_airport": by_airport, "pooled": pooled}, "weights": {"w0": 0.3, "w1": 0.85}}
    want = uad_price.apply(frame, const_nested)
    c = pd.Series(ap).map(by_airport).fillna(pooled).to_numpy(dtype="float64")
    got = U.blend(v, est, p, sp, storm_a, cls, c, w0=0.3, w1=0.85)
    np.testing.assert_array_equal(got, want)


# ============================================================================================================= serve

def test_serve_only_touches_scope_rows_even_when_every_row_would_classify():
    """A table row exists and classifies for every id, but only the ids decided unmatched_body / storm are written;
    rome / arc / rome_e2 / rome_e2w / a matched row (base_lightgbm) keep their value and label untouched. Fails when
    serve selects rows by table class alone instead of ledger.decided_of in SCOPE."""
    kept = [("unmatched_body", "unmatched_body"), ("storm", "storm"), ("rome", "rome"), ("arc", "arc"),
           ("rome_e2", "rome_e2"), ("rome_e2w", "rome_e2w"), ("base_model", "base_lightgbm")]
    ids = [float(i) for i in range(1, 8)]
    led = S.Ledger(np.array(ids))
    for i, (stage_name, label) in enumerate(kept, start=1):
        led.write(STAGE[stage_name], np.array([float(i)]), np.array([1000.0]), label=label)
    rows = _mk_rows([(float(i), "EDDF", "2026-01-10 10:16:40", "2026-01-10 10:00:00") for i in range(1, 8)])
    table = _mk_table([(float(i), True, "A1", 400.0) for i in range(1, 8)])        # every row classifies
    comp = _mk_comp([(float(i), 0.5, 100.0) for i in range(1, 8)])
    comp_out, rep = U.serve(rows, comp, led, STAGE["uad"], table, CONST)
    assert sorted(comp_out.index.tolist()) == [1.0, 2.0]
    assert rep["n_scope"] == 2
    untouched = dict(zip(ids[2:], led.decided_of(np.array(ids[2:]))))
    assert untouched == {3.0: "rome", 4.0: "arc", 5.0: "rome_e2", 6.0: "rome_e2w", 7.0: "base_lightgbm"}
    assert (led.values_of(np.array(ids[2:])) == 1000.0).all()


def test_serve_writes_uad_even_when_the_value_does_not_change():
    """UAD's own rule (unlike LDI's zero-delta skip): decided_by == 'uad' means 'this row has an ADS-B estimate', even
    when w0 == 0 leaves the value exactly where it was. Fails when serve only writes rows whose value actually moved
    (LDI's rule, ported here by mistake)."""
    ids = [1.0]
    led = S.Ledger(np.array(ids))
    led.write(STAGE["unmatched_body"], np.array(ids), np.array([500.0]))
    rows = _mk_rows([(1.0, "EDDF", "2026-01-10 10:16:40", "2026-01-10 10:00:00")])
    table = _mk_table([(1.0, True, "A1", 400.0)])
    comp = _mk_comp([(1.0, 0.5, 100.0)])
    const = {**CONST, "w0": 0.0}
    comp_out, rep = U.serve(rows, comp, led, STAGE["uad"], table, const)
    assert led.values_of(np.array(ids))[0] == 500.0
    assert led.decided_of(np.array(ids))[0] == "uad"
    assert comp_out.loc[1.0, "uad_shift"] == 0.0


def test_serve_comp_out_and_report_on_hand_numbers():
    """id1 EDDF A1 calm with a defined p/sp blend; id2 ZZZZ (pooled c) A2 storm with a NaN p fallback; id3 unclassed
    (cls ""), untouched; id4 decided by 'rome' (out of scope: no table row exists for it at all, so serve must never
    look it up). Numbers verified by hand (session scratch, python3.11). Fails when the report counts scope rows
    instead of written rows for n_a1/n_a2/n_storm/by_airport/shift_mean_s/shift_rms_s, or looks up a non-scope row."""
    ids = [1.0, 2.0, 3.0, 4.0]
    led = S.Ledger(np.array(ids))
    led.write(STAGE["unmatched_body"], np.array([1.0, 3.0]), np.array([800.0, 500.0]))
    led.write(STAGE["storm"], np.array([2.0]), np.array([900.0]))
    led.write(STAGE["rome"], np.array([4.0]), np.array([600.0]))
    rows = _mk_rows([(1.0, "EDDF", "2026-01-10 10:16:40", "2026-01-10 10:00:00"),
                     (2.0, "ZZZZ", "2026-01-10 11:08:20", "2026-01-10 11:00:00")])
    table = _mk_table([(1.0, True, "A1", 400.0), (2.0, True, "A2", 1000.0), (3.0, True, "", NAN)])
    comp = _mk_comp([(1.0, 0.4, 100.0), (2.0, NAN, 350.0), (3.0, 0.9, 50.0), (4.0, NAN, NAN)])
    comp_out, rep = U.serve(rows, comp, led, STAGE["uad"], table, CONST)
    assert sorted(comp_out.index.tolist()) == [1.0, 2.0]
    assert comp_out.loc[1.0].to_dict() == {"uad_cls": "A1", "uad_est": 400.0, "uad_shift": -65.0}
    assert comp_out.loc[2.0, "uad_cls"] == "A2" and comp_out.loc[2.0, "uad_est"] == 1000.0
    assert comp_out.loc[2.0, "uad_shift"] == 76.0
    assert dict(zip(ids, led.values_of(np.array(ids)))) == {1.0: 735.0, 2.0: 976.0, 3.0: 500.0, 4.0: 600.0}
    assert dict(zip(ids, led.decided_of(np.array(ids)))) == {1.0: "uad", 2.0: "uad", 3.0: "unmatched_body",
                                                              4.0: "rome"}
    assert rep["n_scope"] == 3 and rep["n_a1"] == 1 and rep["n_a2"] == 1 and rep["n_storm"] == 1
    assert rep["by_airport"] == {"EDDF": 1, "ZZZZ": 1}
    assert rep["w0"] == 0.5 and rep["w1"] == 0.9 and rep["registration"] == "test-reg"
    assert rep["shift_mean_s"] == pytest.approx(5.5)
    assert rep["shift_rms_s"] == pytest.approx(70.71421356417676)


def test_serve_falls_back_to_the_pooled_offset_for_an_airport_missing_from_by_airport():
    """Fails when the pooled fallback is skipped (a KeyError, or a silent 0.0 c) for an airport the offsets never
    named."""
    ids = [10.0]
    led = S.Ledger(np.array(ids))
    led.write(STAGE["unmatched_body"], np.array(ids), np.array([1000.0]))
    rows = _mk_rows([(10.0, "ZZZZ", "2026-01-10 10:16:40", "2026-01-10 10:00:00")])
    table = _mk_table([(10.0, True, "A1", 400.0)])
    comp = _mk_comp([(10.0, NAN, NAN)])                 # NaN p -> u' = u = est + pooled c
    const = {**CONST, "by_airport": {}}                 # no airport listed at all
    comp_out, rep = U.serve(rows, comp, led, STAGE["uad"], table, const)
    want = float(np.rint(max(1000.0 + 1.0 * const["w0"] * ((400.0 + const["pooled"]) - 1000.0), 1.0)))
    assert led.values_of(np.array(ids))[0] == want
    assert comp_out.loc[10.0, "uad_est"] == 400.0


def test_serve_refuses_a_scope_row_missing_from_the_table():
    """Fails when serve treats a missing table row as unclassed instead of refusing by name."""
    ids = [1.0]
    led = S.Ledger(np.array(ids))
    led.write(STAGE["unmatched_body"], np.array(ids), np.array([500.0]))
    rows = _mk_rows([(1.0, "EDDF", "2026-01-10 10:16:40", "2026-01-10 10:00:00")])
    comp = _mk_comp([(1.0, 0.5, 100.0)])
    with pytest.raises(E.RoutingError, match="no table row"):
        U.serve(rows, comp, led, STAGE["uad"], _mk_table([]), CONST)


def test_serve_refuses_a_table_row_marked_unmatched_false():
    """Fails when serve ignores the table's own `unmatched` flag (the table was built for other rows)."""
    ids = [1.0]
    led = S.Ledger(np.array(ids))
    led.write(STAGE["unmatched_body"], np.array(ids), np.array([500.0]))
    rows = _mk_rows([(1.0, "EDDF", "2026-01-10 10:16:40", "2026-01-10 10:00:00")])
    table = _mk_table([(1.0, False, "A1", 400.0)])
    comp = _mk_comp([(1.0, 0.5, 100.0)])
    with pytest.raises(E.SchemaError, match="unmatched False"):
        U.serve(rows, comp, led, STAGE["uad"], table, CONST)


@pytest.mark.parametrize("est", [0.0, -5.0, float("nan"), float("inf")])
def test_serve_refuses_a_classed_row_with_a_bad_exit_rel_s(est):
    """Fails when the finite-and-positive check on exit_rel_s is dropped."""
    ids = [1.0]
    led = S.Ledger(np.array(ids))
    led.write(STAGE["unmatched_body"], np.array(ids), np.array([500.0]))
    rows = _mk_rows([(1.0, "EDDF", "2026-01-10 10:16:40", "2026-01-10 10:00:00")])
    table = _mk_table([(1.0, True, "A1", est)])
    comp = _mk_comp([(1.0, 0.5, 100.0)])
    with pytest.raises(E.RangeError, match="exit_rel_s"):
        U.serve(rows, comp, led, STAGE["uad"], table, CONST)


def test_serve_refuses_a_classed_row_with_no_ledger_value():
    """decided_of and values_of are always written together through the real write() path, so a scope row with no
    value cannot be built through the public API; this reaches the ledger's own (non-underscore) arrays directly to
    simulate the gap, mirroring ldi.py's identical defensive check. Fails when serve trusts decided_of without
    checking values_of is finite."""
    ids = np.array([1.0])
    led = S.Ledger(ids)
    led.decided_by[0] = "unmatched_body"
    rows = _mk_rows([(1.0, "EDDF", "2026-01-10 10:16:40", "2026-01-10 10:00:00")])
    table = _mk_table([(1.0, True, "A1", 400.0)])
    comp = _mk_comp([(1.0, 0.5, 100.0)])
    with pytest.raises(E.RoutingError, match="no value"):
        U.serve(rows, comp, led, STAGE["uad"], table, CONST)


def test_serve_with_no_scope_row_is_a_reported_noop():
    """Fails when serve raises (instead of returning empty) when nothing is in scope."""
    ids = [1.0]
    led = S.Ledger(np.array(ids))
    led.write(STAGE["rome"], np.array(ids), np.array([500.0]))
    rows = _mk_rows([(1.0, "EDDF", "2026-01-10 10:16:40", "2026-01-10 10:00:00")])
    comp = _mk_comp([(1.0, NAN, NAN)])
    comp_out, rep = U.serve(rows, comp, led, STAGE["uad"], _mk_table([]), CONST)
    assert len(comp_out) == 0 and rep["n_scope"] == 0 and rep["n_a1"] == 0 and rep["n_a2"] == 0
    assert led.decided_of(np.array(ids))[0] == "rome"


def test_serve_with_no_classed_row_is_a_reported_noop():
    """Fails when an all-unclassed scope is treated as an error, or n_scope is reported as 0 instead of the true
    count."""
    ids = [1.0]
    led = S.Ledger(np.array(ids))
    led.write(STAGE["unmatched_body"], np.array(ids), np.array([500.0]))
    rows = _mk_rows([(1.0, "EDDF", "2026-01-10 10:16:40", "2026-01-10 10:00:00")])
    table = _mk_table([(1.0, True, "", NAN)])
    comp = _mk_comp([(1.0, 0.5, 100.0)])
    comp_out, rep = U.serve(rows, comp, led, STAGE["uad"], table, CONST)
    assert len(comp_out) == 0 and rep["n_scope"] == 1 and rep["n_a1"] == 0 and rep["n_a2"] == 0
    assert led.values_of(np.array(ids))[0] == 500.0 and led.decided_of(np.array(ids))[0] == "unmatched_body"


# ================================================================================================= load_* refusals

def test_load_constants_returns_the_flat_dict(tmp_path):
    """Fails when load_constants returns the raw nested shape instead of the flat one serve() reads."""
    p = _write_constants(tmp_path / "c.json")
    got = U.load_constants(p)
    assert got == {"by_airport": {"EDDF": 40.0}, "pooled": 60.0, "w0": 0.5, "w1": 0.8, "registration": "reg-test"}


@pytest.mark.parametrize("drop", ["offsets", "offsets.by_airport", "offsets.pooled", "weights", "weights.w0",
                                  "weights.w1", "registration"])
def test_load_constants_refuses_a_missing_key(tmp_path, drop):
    """Fails when a missing key in any of these positions is read as a default / None instead of refused."""
    base = {"offsets": {"by_airport": {"EDDF": 40.0}, "pooled": 60.0}, "weights": {"w0": 0.5, "w1": 0.8},
           "registration": "reg-test"}
    node = base
    parts = drop.split(".")
    for p in parts[:-1]:
        node = node[p]
    del node[parts[-1]]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(base))
    with pytest.raises(E.SchemaError, match="missing key"):
        U.load_constants(p)


@pytest.mark.parametrize("key, bad", [("offsets.pooled", float("nan")), ("offsets.pooled", float("inf")),
                                      ("weights.w0", float("nan")), ("weights.w1", float("inf"))])
def test_load_constants_refuses_a_non_finite_number(tmp_path, key, bad):
    """Fails when a NaN or infinite number in the constants file is accepted."""
    p = _write_constants(tmp_path / "c.json", {key: bad})
    with pytest.raises(E.SchemaError, match="finite"):
        U.load_constants(p)


@pytest.mark.parametrize("key, bad", [("weights.w0", -0.1), ("weights.w0", 1.1), ("weights.w1", -0.5),
                                      ("weights.w1", 1.5)])
def test_load_constants_refuses_a_weight_outside_0_1(tmp_path, key, bad):
    """Fails when the [0, 1] range check on a weight is dropped."""
    p = _write_constants(tmp_path / "c.json", {key: bad})
    with pytest.raises(E.SchemaError, match=r"outside \[0, 1\]"):
        U.load_constants(p)


def test_load_table_reads_only_table_cols(tmp_path):
    """Fails when load_table reads every column in the file instead of just TABLE_COLS."""
    p = tmp_path / "t.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "unmatched": [True, True], "cls": ["A1", ""],
                 "exit_rel_s": [400.0, NAN], "extra_col": ["x", "y"]}).to_parquet(p, index=False)
    got = U.load_table(p)
    assert list(got.columns) == U.TABLE_COLS


def test_load_table_refuses_a_null_mvt_id(tmp_path):
    """Fails when a null MVT_ID_mvt is silently kept (it cannot be joined to any ledger id)."""
    p = tmp_path / "t.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, NAN], "unmatched": [True, True], "cls": ["A1", ""],
                 "exit_rel_s": [400.0, NAN]}).to_parquet(p, index=False)
    with pytest.raises(E.SchemaError, match="null MVT_ID_mvt"):
        U.load_table(p)


def test_load_table_refuses_a_duplicated_mvt_id(tmp_path):
    """Fails when a duplicated MVT_ID_mvt is silently kept (serve's table lookup would be ambiguous)."""
    p = tmp_path / "t.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 1.0], "unmatched": [True, True], "cls": ["A1", "A1"],
                 "exit_rel_s": [400.0, 400.0]}).to_parquet(p, index=False)
    with pytest.raises(E.DuplicateIdError, match="duplicate MVT_ID_mvt"):
        U.load_table(p)


# ================================================================================================================ apply

def test_apply_reads_the_table_and_constants_named_by_the_config(tmp_path):
    """Fails when apply reads a hardcoded path instead of cfg.path('modules.uad.table' / '.constants'), or reads them
    at import time instead of inside the call."""
    const_path = _write_constants(tmp_path / "const.json")
    table_path = tmp_path / "table.parquet"
    _mk_table([(1.0, True, "A1", 400.0)]).to_parquet(table_path, index=False)
    cfg = _cfg(**{"modules.uad.enabled": True, "modules.uad.table": str(table_path),
                 "modules.uad.constants": str(const_path), "modules.storm.mode": "anchor"})
    ids = [1.0]
    led = S.Ledger(np.array(ids))
    led.write(STAGE["unmatched_body"], np.array(ids), np.array([500.0]))
    rows = _mk_rows([(1.0, "EDDF", "2026-01-10 10:16:40", "2026-01-10 10:00:00")])
    comp = _mk_comp([(1.0, 0.5, 100.0)])
    comp_out, rep = U.apply(cfg, rows, comp, led, STAGE["uad"])
    assert len(comp_out) == 1 and rep["n_a1"] == 1
    other_cfg = cfg.replace(**{"modules.uad.table": str(tmp_path / "missing.parquet")})
    with pytest.raises(FileNotFoundError):
        U.apply(other_cfg, rows, comp, S.Ledger(np.array(ids)), STAGE["uad"])


# ========================================================================================================= the e2e world

@pytest.mark.slow
def test_world_predict_serves_uad_through_the_engine(e2e_world, fm1_world_mod, tmp_path):
    """api.predict with the stage on: the engine calls uad after the storm anchor's columns are merged into `comp`
    and before LDI; a constructed table classifies two of the world's unmatched-lane rows (A1, A2), the rest stay
    cls ""; their value follows the registered rule from the value the same world gives with the stage off, the
    output frame carries uad_cls / uad_est / uad_shift, and the report carries info["uad"]. Mutation: drop the
    engine's uad call -> no 'uad' rows, no report -> red."""
    from prc import api
    world = e2e_world[0]
    over = {"modules.storm.mode": "anchor", "run.artifacts_dir": str(world / "art_uad")}
    cfg_off = K.parse_config(fm1_world_mod.config_raw(world, **over))
    art = api.fit(cfg_off, save=False, log=lambda *a, **k: None)
    scored = pd.read_parquet(world / "raw" / "scored.parquet")
    quiet = lambda *a, **k: None  # noqa: E731
    base = api.predict(art, scored, log=quiet).set_index("MVT_ID_mvt")
    scope_ids = base.index[base.decided_by.isin(list(U.SCOPE))].to_numpy(dtype="float64")
    assert len(scope_ids) >= 2, "the world has no unmatched-lane row to classify; see report item 7"
    a1_id, a2_id = float(scope_ids[0]), float(scope_ids[1])
    table = pd.DataFrame({"MVT_ID_mvt": scope_ids, "unmatched": True, "cls": "", "exit_rel_s": np.nan})
    table.loc[table.MVT_ID_mvt == a1_id, ["cls", "exit_rel_s"]] = ["A1", 400.0]
    table.loc[table.MVT_ID_mvt == a2_id, ["cls", "exit_rel_s"]] = ["A2", 1000.0]
    table_path = tmp_path / "table.parquet"
    table.to_parquet(table_path, index=False)
    const = {"offsets": {"by_airport": {"EDDF": 50.0, "LSZH": 70.0}, "pooled": 60.0},
            "weights": {"w0": 0.3, "w1": 0.8}, "registration": "test-e2e"}
    const_path = tmp_path / "const.json"
    const_path.write_text(json.dumps(const))
    cfg_on = cfg_off.replace(**{"modules.uad.enabled": True, "modules.uad.table": str(table_path),
                                "modules.uad.constants": str(const_path)})
    art_on = dataclasses.replace(art, config=cfg_on)
    pred = api.predict(art_on, scored, log=quiet).set_index("MVT_ID_mvt")
    rep = pred.attrs["fm1_report"]["uad"]
    assert rep["n_a1"] == 1 and rep["n_a2"] == 1
    hit = [a1_id, a2_id]
    assert (pred.decided_by[hit] == "uad").all()
    assert pred.uad_cls[a1_id] == "A1" and pred.uad_cls[a2_id] == "A2"
    assert pred.uad_est[a1_id] == 400.0 and pred.uad_est[a2_id] == 1000.0
    # hand-check row a1's value against the registered rule, reading every input from the actual run (robust to
    # whatever this real small world's p / sp / storm_A happen to be)
    v1 = float(base.TAXITIME_SEC_mvt[a1_id])
    sraw = scored.set_index("MVT_ID_mvt")
    ap1 = str(sraw.ADEP_mvt[a1_id])
    c1 = const["offsets"]["by_airport"].get(ap1, const["offsets"]["pooled"])
    p1 = float(pred.unm_p_fill[a1_id])
    sp1 = (sraw.MVT_TIME_UTC_mvt[a1_id] - sraw.SCHED_TIME_UTC_mvt[a1_id]).total_seconds()
    storm_a1 = float(pred.storm_A[a1_id])
    want1 = U.blend(np.array([v1]), np.array([400.0]), np.array([p1]), np.array([sp1]), np.array([storm_a1]),
                    np.array(["A1"]), np.array([c1]), 0.3, 0.8)[0]
    assert pred.TAXITIME_SEC_mvt[a1_id] == int(want1)
    same = base.index.difference(hit)
    assert (pred.TAXITIME_SEC_mvt[same] == base.TAXITIME_SEC_mvt[same]).all()
    assert "uad" in pred.attrs["fm1_report"]["stages"]
    # mtx still runs after it when both are configured on (cheap structural check: no second fit needed)
    on = [s.name for s in S.plan(K.parse_config({"run": {"purpose": "scored_period"},
                                                 "modules": {"storm": {"mode": "anchor"}, "uad": {"enabled": True}}}))]
    assert on.index("storm") < on.index("uad") < on.index("mtx")


# ========================================================================================================= build_inputs

def _raw12(tmp_path) -> pathlib.Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    for m in range(1, 13):
        nxt = "2026-01-01" if m == 12 else f"2025-{m + 1:02d}-01"
        (raw / f"training_2025-{m:02d}-01_{nxt}.parquet").write_bytes(f"month {m}".encode())
    return raw


def test_build_inputs_records_the_two_files_when_the_stage_is_on_and_a_missing_one_stops_the_build(tmp_path):
    """Mirrors the mtx/published precedent exactly (test_fm1_build_inputs.py). Fails when the two sha256 keys are
    missing, wrong, or when a missing file is tolerated instead of stopping the build before predicting."""
    raw = _raw12(tmp_path)
    table = tmp_path / "table.parquet"
    const = tmp_path / "const.json"
    table.write_bytes(b"table bytes")
    const.write_bytes(b"const bytes")
    cfg = K.parse_config(K.nest({"data.raw_dir": str(raw), "lanes.matched": False, "modules.rome.enabled": False,
                                 "modules.storm.mode": "anchor", "modules.uad.enabled": True,
                                 "modules.uad.table": str(table), "modules.uad.constants": str(const)}))
    art = types.SimpleNamespace(inputs=G.fit_inputs(cfg, G.training_files(cfg)))
    bi = G.build_inputs(cfg, art)
    assert bi["inputs"]["uad/table"] == A.sha256_file(table)
    assert bi["inputs"]["uad/constants"] == A.sha256_file(const)
    assert bi["changed_since_fit"] == []
    table.unlink()
    with pytest.raises(FileNotFoundError):
        G.build_inputs(cfg, art)


def test_build_inputs_omits_the_uad_keys_when_the_stage_is_off(tmp_path):
    """The table/constants paths point at files that do not exist, so an unguarded hash would raise; the stage being
    off must mean they are never touched. Fails when the keys appear anyway, or when build_inputs raises."""
    raw = _raw12(tmp_path)
    cfg = K.parse_config(K.nest({"data.raw_dir": str(raw), "lanes.matched": False, "modules.rome.enabled": False,
                                 "modules.uad.table": str(tmp_path / "missing_table.parquet"),
                                 "modules.uad.constants": str(tmp_path / "missing_const.json")}))
    art = types.SimpleNamespace(inputs=G.fit_inputs(cfg, G.training_files(cfg)))
    bi = G.build_inputs(cfg, art)
    assert "uad/table" not in bi["inputs"] and "uad/constants" not in bi["inputs"]
