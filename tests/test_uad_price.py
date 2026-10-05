"""scripts/analysis/uad_price.py — UAD: the aircraft's own ADS-B ground track as the off-block estimate for
departures without a flight-plan record.

Registration: plans/PREREG_uad_unmatched_adsb_offblock_2026_10_04.md (REG 17a570a6). Constructed rows only: the
frame loader's refusals and derived columns, the per-airport offset fit and its pooling/exclude rules, the blend
formula and its NaN fallback, the calm/storm weight fit and its floor rule, the constants wrapper's month
isolation, the apply rule's class/scope gates and its floor-and-round, the population/records split, the H1 and
H2 verdict rules (including one call into the real protocols.compare/promotion_gate), the gain breakdown, and the
CLI's overwrite guards and scored-month isolation.

Written first and seen red 2026-10-04 (no module).

Mutation rehearsal before the measurement (in place, one at a time, module restored byte-identical and
hash-checked after every mutant): 16 / 16 killed, 3 on a second try after a fixture or a test gap was found and
fixed — UP1 the A1-only filter in offsets, UP2 the 900 s bound, UP3 the 200-row pooling rule, UP4 the month filter
in offsets (survived first: the decoy January row was also excluded by the 900 s bound, so the fixture never
isolated the month filter; fixed by moving the decoy inside the bound), UP5 the month filter in weights, UP6 the
scope filter in weights, UP7 the clip in weights, UP8 the 30-row storm floor rule, UP9 the A2 half-weight factor in
apply, UP10 the floor-at-1 in apply, UP11 the storm threshold (> vs >=) in load_frame, UP12 the exclude passed
from h2 to constants, UP13 the gate condition in h1's verdict, UP14 the rmse-fall bar (>= vs >) in h1's verdict
(survived first: 1.0 - ratio can never land bit-exact on 0.10 from a ratio near 0.9, a float64 grid mismatch, not a
missing test — fixed by extracting the comparison into `_rmse_ok()` and testing it directly at the literal
threshold value, which also surfaced and fixed a real numpy.bool_-vs-bool inconsistency), UP15 the scored-month
drop in the `constants` CLI command (survived first: constants() already filters by month internally, so a missing
drop does not change any fitted value, only `n_rows`, which the test checked only for A-vs-B equality, not the
absolute count — fixed by asserting n_rows == 14), UP16 (extra, beyond the registered list) the
both-scored-months-populated condition in h1's verdict.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


U = _load("uad_price", "scripts/analysis/uad_price.py")


# ==================================================================================== shared fixture helpers

def _frame(rows, default_ap="EDDF", default_day="2025-02-10", default_month=2):
    """Builds a post-load_frame-shaped frame directly (the columns offsets/weights/apply/population/records/h1/h2/
    breakdown read: MVT_ID_mvt, ap, day, month, y, v, p, sp, est, cls, scope, storm, storm_A). Each row dict may
    override any of: id, ap, day, month, y, v, p, sp, est, cls, scope, storm, storm_A."""
    out = []
    for i, r in enumerate(rows):
        storm = bool(r.get("storm", False))
        out.append({
            "MVT_ID_mvt": float(r.get("id", i)),
            "ap": r.get("ap", default_ap),
            "day": r.get("day", default_day),
            "month": int(r.get("month", default_month)),
            "y": float(r.get("y", 1000.0)),
            "v": float(r.get("v", 1000.0)),
            "p": float(r.get("p", np.nan)),
            "sp": float(r.get("sp", np.nan)),
            "est": float(r.get("est", np.nan)),
            "cls": r.get("cls", "A1"),
            "scope": bool(r.get("scope", True)),
            "storm": storm,
            "storm_A": float(r.get("storm_A", 350.0 if storm else np.nan)),
        })
    return pd.DataFrame(out)


def _planted_rows(off_c, ratio, n, start_id, storm, ap="EDDF", day="2025-02-10", month=2):
    """n rows at (ap, day, month) whose y exactly satisfies y = v + ratio * (u' - v), using the real u_prime()
    with p = sp = NaN (so u' = est + off_c). Used to plant a weight the fit must recover exactly."""
    rows = []
    for i in range(n):
        est = 500.0 + i
        up = U.u_prime(np.array([np.nan]), np.array([np.nan]), np.array([est]), np.array([off_c]))[0]
        v = 1000.0
        rows.append({"id": start_id + i, "cls": "A1", "scope": True, "v": v, "y": v + ratio * (up - v), "est": est,
                     "storm": storm, "ap": ap, "day": day, "month": month})
    return rows


def _sp_rows(ratio, n, start_id, ap="EDDF", day="2025-02-10", month=2, storm=True):
    """n rows whose u' = sp exactly (p = 1.0 makes u' independent of est/c), and y = ratio * sp exactly (v = 0),
    so the fitted weight equals `ratio` regardless of whatever offset the frame's other rows produce."""
    rows = []
    for i in range(n):
        sp = 1000.0 + i
        rows.append({"id": start_id + i, "cls": "A1", "scope": True, "v": 0.0, "est": 0.0, "p": 1.0, "sp": sp,
                     "y": ratio * sp, "storm": storm, "ap": ap, "day": day, "month": month})
    return rows


def _calm_filler(start_id=90000, ap="EDDF", day="2025-06-01"):
    """A handful of non-storm population rows so the calm regime is never empty (weights() raises on an empty
    regime via a zero denominator; only an empty STORM regime is exempted from that, by design). The planted ratio
    is irrelevant to tests that only assert on w1."""
    return _sp_rows(0.5, 5, start_id, ap=ap, day=day, storm=False)


# ==================================================================================== stop rules (AMEND-A1 item 1;
# AMEND-A2: rule (a) reads callsign_matched_type_equal_share, not the builder's alias-aware share of the same joins;
# AMEND-A3: TABLE_PARAMS_HASH = "d196b652a9a7", rule (b) is read on the stand joins, rule (d) and the bracket are new)

STOP_STATS = ["callsign_matched_type_equal_share", "stand_unmatched_type_agree_share", "matched_exit_mad_s",
              "stand_check_right_share", "stand_check_wrong_share"]


def _report(callsign_equal=0.99, callsign_agree_alias=0.999, stand_agree=0.99, time_agree=0.999, mad=50.0,
           check_right=0.99, check_wrong=0.0, classed=((600, 20), (30, 0)), params_hash=None, airport=None,
           no_stop_rules=False, no_airport=False, extra_eham=None):
    """A table-builder report.json dict, passing every stop rule by default. `params_hash=None` uses
    U.TABLE_PARAMS_HASH; pass a different string to simulate a mismatch. `callsign_agree_alias` and `time_agree`
    fill keys the report also carries and that rules (a) and (b) must never read. `classed` is the stop airport's
    (unmatched_A1, unmatched_A2) per month; another airport's month is always present and must not be counted.
    `extra_eham` overrides / extends the stop airport's statistics."""
    ph = U.TABLE_PARAMS_HASH if params_hash is None else params_hash
    ap = U.STOP_AIRPORT if airport is None else airport
    stats = {"callsign_matched_type_equal_share": callsign_equal,
            "callsign_matched_type_agree_share": callsign_agree_alias,
            "stand_unmatched_type_agree_share": stand_agree, "time_unmatched_type_agree_share": time_agree,
            "matched_exit_mad_s": mad, "stand_check_right_share": check_right, "stand_check_wrong_share": check_wrong}
    if extra_eham:
        stats.update(extra_eham)
    rep = {"params_hash": ph, "by_airport_month": {"LSZH|1": {"unmatched_A1": 5000, "unmatched_A2": 5000}}}
    for i, (a1, a2) in enumerate(classed or ()):
        rep["by_airport_month"][f"{U.STOP_AIRPORT}|{i + 1}"] = {"unmatched_A1": a1, "unmatched_A2": a2}
    if not no_stop_rules:
        rep["stop_rules"] = {} if no_airport else {ap: stats}
    return rep


def test_table_params_hash_is_the_registered_amendment_a3_value():
    """Pins the literal registered value (AMEND-A3) so an accidental constant edit is caught even though every
    other test compares against the constant itself. Fails when TABLE_PARAMS_HASH drifts from "d196b652a9a7"."""
    assert U.TABLE_PARAMS_HASH == "d196b652a9a7"


@pytest.mark.parametrize("value, want_ok", [(0.97, True), (0.9699, False)])
def test_check_stop_rules_callsign_equal_share_boundary(value, want_ok):
    """Fails when the >= comparison on rule (a) is loosened to > or the bar constant drifts from 0.97."""
    out = U.check_stop_rules(_report(callsign_equal=value))
    assert out["ok"] is want_ok
    if not want_ok:
        assert out["failed"] == ["callsign_matched_type_equal_share below 0.97"]


def test_check_stop_rules_reads_the_equal_share_not_the_agree_alias():
    """The builder's report also carries callsign_matched_type_agree_share (a looser, non-exact-type-equality
    statistic). Fails when rule (a) reads that alias instead of callsign_matched_type_equal_share: here the alias
    is high (0.999, would pass the 0.97 bar) but the real statistic is 0.96 (below it)."""
    out = U.check_stop_rules(_report(callsign_equal=0.96, callsign_agree_alias=0.999))
    assert out["ok"] is False
    assert out["failed"] == ["callsign_matched_type_equal_share below 0.97"]
    assert out["values"]["callsign_matched_type_equal_share"] == pytest.approx(0.96)


@pytest.mark.parametrize("value, want_ok", [(0.95, True), (0.9499, False)])
def test_check_stop_rules_stand_agree_share_boundary(value, want_ok):
    """Fails when the >= comparison on rule (b) is loosened to > or the bar constant drifts from 0.95."""
    out = U.check_stop_rules(_report(stand_agree=value))
    assert out["ok"] is want_ok
    if not want_ok:
        assert out["failed"] == ["stand_unmatched_type_agree_share below 0.95"]


def test_check_stop_rules_reads_rule_b_on_the_stand_joins_not_the_time_joins():
    """AMEND-A3 reads rule (b) on the stand joins of unmatched rows; the report still carries the old statistic on
    their time joins. Fails when rule (b) reads the time-join share (0.999 here, the stand-join share is 0.94)."""
    out = U.check_stop_rules(_report(stand_agree=0.94, time_agree=0.999))
    assert out["ok"] is False and out["failed"] == ["stand_unmatched_type_agree_share below 0.95"]


@pytest.mark.parametrize("right, wrong, failed", [
    (0.95, 0.005, []),
    (0.9499, 0.005, ["stand_check_right_share below 0.95"]),
    (0.95, 0.0051, ["stand_check_wrong_share above 0.005"]),
])
def test_check_stop_rules_rule_d_boundaries(right, wrong, failed):
    """AMEND-A3 stop rule (d): the stand rule, scored against the callsign join, is right on >= 95 % and wrong on
    <= 0.5 % of the counted rows. Fails when either comparison is strict on the wrong side or a bar drifts."""
    out = U.check_stop_rules(_report(check_right=right, check_wrong=wrong))
    assert out["failed"] == failed and out["ok"] is (not failed)


@pytest.mark.parametrize("classed, want_ok", [
    (((500, 8),), True), (((500, 7),), False), (((990, 5), (5, 0)), True), (((990, 5), (5, 1)), False),
    ((), False),
])
def test_check_stop_rules_class_count_bracket(classed, want_ok):
    """AMEND-A3's sanity bracket: the stop airport's unmatched rows with a class (A1 + A2 over its months) number
    between 508 and 1,000, both included. Fails when an end is strict, the months are not summed, another airport's
    months are counted (LSZH holds 10,000 in every report here), or a report without the counts passes."""
    out = U.check_stop_rules(_report(classed=classed))
    assert out["ok"] is want_ok
    if classed:
        assert out["values"]["unmatched_classed"] == sum(a + b for a, b in classed)
    else:                                          # no month of the stop airport: the count is missing, not zero
        assert out["values"]["unmatched_classed"] is None and "count missing" in out["failed"][0]
    if not want_ok:
        assert len(out["failed"]) == 1 and "unmatched rows with a class" in out["failed"][0]


@pytest.mark.parametrize("value, want_ok", [(200.0, True), (200.01, False)])
def test_check_stop_rules_exit_mad_boundary(value, want_ok):
    """Fails when the <= comparison on rule (c) is loosened to < or the bar constant drifts from 200.0."""
    out = U.check_stop_rules(_report(mad=value))
    assert out["ok"] is want_ok
    if not want_ok:
        assert out["failed"] == ["matched_exit_mad_s above 200.0"]


@pytest.mark.parametrize("stat", STOP_STATS)
def test_check_stop_rules_nan_fails_naming_the_statistic(stat):
    """Fails when a NaN value is treated as passing instead of as a missing-statistic failure."""
    out = U.check_stop_rules(_report(extra_eham={stat: float("nan")}))
    assert out["ok"] is False
    assert len(out["failed"]) == 1
    assert stat in out["failed"][0]


@pytest.mark.parametrize("stat", STOP_STATS)
def test_check_stop_rules_none_fails_naming_the_statistic(stat):
    """Fails when a None value is treated as passing instead of as a missing-statistic failure."""
    out = U.check_stop_rules(_report(extra_eham={stat: None}))
    assert out["ok"] is False
    assert len(out["failed"]) == 1
    assert stat in out["failed"][0]


@pytest.mark.parametrize("stat", STOP_STATS)
def test_check_stop_rules_missing_key_fails_naming_the_statistic(stat):
    """Fails when a key absent from the report is treated as passing instead of as a missing-statistic failure."""
    rep = _report()
    del rep["stop_rules"][U.STOP_AIRPORT][stat]
    out = U.check_stop_rules(rep)
    assert out["ok"] is False
    assert len(out["failed"]) == 1
    assert stat in out["failed"][0]


def test_check_stop_rules_wrong_params_hash_fails():
    """Fails when the params_hash check is removed or compares against the wrong constant."""
    out = U.check_stop_rules(_report(params_hash="deadbeef0000"))
    assert out["ok"] is False
    assert len(out["failed"]) == 1
    assert "params_hash" in out["failed"][0]
    assert out["values"]["params_hash"] == "deadbeef0000"


def test_check_stop_rules_no_stop_rules_key_does_not_raise():
    """Fails when a report missing the whole stop_rules block crashes instead of failing every statistic."""
    out = U.check_stop_rules(_report(no_stop_rules=True))
    assert out["ok"] is False
    assert len(out["failed"]) == 5


def test_check_stop_rules_no_eham_entry_does_not_raise():
    """Fails when a report whose stop_rules holds no EHAM entry crashes instead of failing every statistic."""
    out = U.check_stop_rules(_report(no_airport=True))
    assert out["ok"] is False
    assert len(out["failed"]) == 5


def test_check_stop_rules_malformed_report_does_not_raise():
    """Fails when a non-dict / wrongly-typed report crashes instead of being treated as wholly failing."""
    assert U.check_stop_rules(None)["ok"] is False
    assert U.check_stop_rules({"stop_rules": "not a dict"})["ok"] is False
    assert U.check_stop_rules({"stop_rules": {U.STOP_AIRPORT: "not a dict either"}})["ok"] is False
    bad_counts = _report()
    bad_counts["by_airport_month"] = {f"{U.STOP_AIRPORT}|1": {"unmatched_A1": "many"}, 7: "not a dict"}
    out = U.check_stop_rules(bad_counts)
    assert out["ok"] is False and len(out["failed"]) == 1 and out["values"]["unmatched_classed"] is None
    assert U.check_stop_rules(dict(_report(), by_airport_month="not a dict"))["ok"] is False


def test_check_stop_rules_all_good_returns_ok_true_and_empty_failed():
    """Fails when a fully passing report is still flagged as failing somewhere."""
    out = U.check_stop_rules(_report())
    assert out["ok"] is True
    assert out["failed"] == []
    assert out["values"]["params_hash"] == U.TABLE_PARAMS_HASH


def test_require_stop_rules_missing_report_raises_naming_the_path(tmp_path):
    """Fails when require_stop_rules stops naming the exact report.json path it looked for."""
    table_path = tmp_path / "table.parquet"
    table_path.write_bytes(b"")
    report_path = tmp_path / "report.json"
    with pytest.raises(ValueError, match=re.escape(str(report_path))):
        U.require_stop_rules(table_path)


def test_require_stop_rules_failing_report_raises_listing_every_reason(tmp_path):
    """Fails when require_stop_rules swallows a failure instead of raising, or drops a reason from the message."""
    table_path = tmp_path / "table.parquet"
    table_path.write_bytes(b"")
    (tmp_path / "report.json").write_text(json.dumps(_report(callsign_equal=0.5, stand_agree=0.5)))
    with pytest.raises(ValueError) as exc:
        U.require_stop_rules(table_path)
    assert "callsign_matched_type_equal_share" in str(exc.value)
    assert "stand_unmatched_type_agree_share" in str(exc.value)


def test_require_stop_rules_passing_report_returns_the_check(tmp_path):
    """Fails when a passing report is refused, or the returned dict loses its ok/failed/values shape."""
    table_path = tmp_path / "table.parquet"
    table_path.write_bytes(b"")
    (tmp_path / "report.json").write_text(json.dumps(_report()))
    out = U.require_stop_rules(table_path)
    assert out["ok"] is True
    assert out["failed"] == []


# ==================================================================================== load_frame

LOMO_RAW_PLAN_COLS = ["LOBT_flt", "IOBT_flt", "EOBT_1_flt", "AOBT_3_flt"]


def _write_lomo_and_raw(lomo_dir, raw_dir, month, rows):
    """One oof_M<month>.parquet and one matching raw file. rows: dicts with id, lane, decided_by, y, and optional
    pred, ap, day, p (unm_p_fill), storm_A, sp (seconds from SCHED to MVT)."""
    lomo = pd.DataFrame([{
        "MVT_ID_mvt": float(r["id"]), "lane": r["lane"], "decided_by": r["decided_by"], "month": month,
        "day": r.get("day", f"2025-{month:02d}-10"), "ap": r.get("ap", "EDDF"), "y": float(r["y"]),
        "pred": float(r.get("pred", r["y"])), "unm_p_fill": r.get("p", np.nan), "storm_A": r.get("storm_A", np.nan),
    } for r in rows])
    lomo.to_parquet(lomo_dir / f"oof_M{month:02d}.parquet", index=False)
    base_t = pd.Timestamp(f"2025-{month:02d}-01 09:00:00", tz="UTC")
    raw = pd.DataFrame([{
        "MVT_ID_mvt": float(r["id"]), "PHASE_mvt": "DEP", "TAXITIME_SEC_mvt": float(r["y"]),
        "FLIGHT_ID_mvt": np.nan if r["lane"] == "unmatched" else 1.0,
        "MVT_TIME_UTC_mvt": base_t, "SCHED_TIME_UTC_mvt": base_t - pd.Timedelta(seconds=float(r.get("sp", 600.0))),
        "LOBT_flt": pd.NaT, "IOBT_flt": pd.NaT, "EOBT_1_flt": pd.NaT, "AOBT_3_flt": pd.NaT,
    } for r in rows])
    raw.to_parquet(raw_dir / f"training_2025-{month:02d}-fixture.parquet", index=False)


def _table_row(r):
    """One table row for a LOMO row dict, matched or unmatched alike (AMEND-A1 item 6: every baseline row needs a
    table row, not only unmatched ones)."""
    return {"MVT_ID_mvt": float(r["id"]), "unmatched": r["lane"] == "unmatched", "usable": True,
            "exit_rel_s": r.get("est", np.nan), "cls": r.get("cls", "")}


def _write_table(table_path, table_rows):
    if table_rows:
        t = pd.DataFrame(table_rows)
    else:
        t = pd.DataFrame({"MVT_ID_mvt": pd.Series([], dtype="float64"), "unmatched": pd.Series([], dtype="bool"),
                          "usable": pd.Series([], dtype="bool"), "exit_rel_s": pd.Series([], dtype="float64"),
                          "cls": pd.Series([], dtype="object")})
    t.to_parquet(table_path, index=False)


def _write_report(table_path, report=None):
    """Writes report.json next to table_path. report=None (default) writes one that passes every stop rule;
    pass a dict to write exactly that instead."""
    if report is None:
        report = _report()
    (pathlib.Path(table_path).parent / "report.json").write_text(json.dumps(report))


def _minimal_12month_world(tmp_path, overrides=None, report=True):
    """Writes 12 LOMO files + matching raw files + one table.parquet + (by default) a passing report.json. Every
    month not named in `overrides` gets a single trivial matched placeholder row (auto-numbered, small ids).
    `overrides[m]` replaces month m's row list (rows without an explicit "id" are auto-numbered too; give
    overridden rows large explicit ids to avoid colliding with the small auto-numbered placeholders). `report=True`
    (default) writes a passing report.json next to the table; `report=False` writes none (for stop-rule-refusal
    tests); a dict writes exactly that report."""
    overrides = overrides or {}
    lomo_dir, raw_dir = tmp_path / "lomo", tmp_path / "raw"
    lomo_dir.mkdir(exist_ok=True)
    raw_dir.mkdir(exist_ok=True)
    table_rows = []
    next_id = 0
    for m in range(1, 13):
        rows = overrides.get(m)
        if rows is None:
            rows = [{"id": next_id, "lane": "matched", "decided_by": "base_lightgbm", "y": 1000.0}]
            next_id += 1
        else:
            rows = [dict(r) for r in rows]
            for r in rows:
                if "id" not in r:
                    r["id"] = next_id
                    next_id += 1
        _write_lomo_and_raw(lomo_dir, raw_dir, m, rows)
        for r in rows:
            table_rows.append(_table_row(r))
    table_path = tmp_path / "table.parquet"
    _write_table(table_path, table_rows)
    if report is True:
        _write_report(table_path)
    elif report is not False:
        _write_report(table_path, report)
    return lomo_dir, table_path, raw_dir


def test_load_frame_refuses_a_missing_month_file(tmp_path):
    """Fails when the check that all 12 month files exist is removed (a short LOMO set would silently under-count
    instead of refusing)."""
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path)
    (lomo_dir / "oof_M07.parquet").unlink()
    with pytest.raises(ValueError, match="missing"):
        U.load_frame(lomo_dir, table_path, sorted(raw_dir.glob("*.parquet")))


def test_load_frame_refuses_a_duplicated_lomo_id(tmp_path):
    """Fails when the duplicate-id check on the combined 12-month frame is removed."""
    dup = {"id": 9999, "lane": "matched", "decided_by": "base_lightgbm", "y": 1000.0}
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {1: [dup], 2: [dup]})
    with pytest.raises(ValueError, match="duplicate"):
        U.load_frame(lomo_dir, table_path, sorted(raw_dir.glob("*.parquet")))


def test_load_frame_refuses_a_duplicated_table_id(tmp_path):
    """Fails when the duplicate check on the table's own ids is removed."""
    rows = [{"id": 500, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "est": 900.0, "cls": "A1"}]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: rows})
    t = pd.read_parquet(table_path)
    t = pd.concat([t, t.iloc[[0]]], ignore_index=True)
    table_path.unlink()
    t.to_parquet(table_path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        U.load_frame(lomo_dir, table_path, sorted(raw_dir.glob("*.parquet")))


def test_load_frame_refuses_an_unmatched_row_missing_from_the_table(tmp_path):
    """Fails when the check for an unmatched LOMO row absent from the table is removed."""
    rows = [{"id": 501, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "est": 900.0, "cls": "A1"}]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: rows})
    t = pd.read_parquet(table_path)
    t = t[t.MVT_ID_mvt != 501.0]
    table_path.unlink()
    t.to_parquet(table_path, index=False)
    with pytest.raises(ValueError, match="no table row"):
        U.load_frame(lomo_dir, table_path, sorted(raw_dir.glob("*.parquet")))


def test_load_frame_refuses_a_lane_unmatched_disagreement(tmp_path):
    """Fails when the lane-vs-table.unmatched agreement check is removed."""
    rows = [{"id": 502, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "est": 900.0, "cls": "A1"}]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: rows})
    t = pd.read_parquet(table_path)
    t.loc[t.MVT_ID_mvt == 502.0, "unmatched"] = False
    table_path.unlink()
    t.to_parquet(table_path, index=False)
    with pytest.raises(ValueError, match="disagree"):
        U.load_frame(lomo_dir, table_path, sorted(raw_dir.glob("*.parquet")))


def test_load_frame_computes_sp_to_the_second(tmp_path):
    """Fails when sp is computed from the wrong pair of timestamps or with the wrong sign."""
    rows = [{"id": 503, "lane": "matched", "decided_by": "base_lightgbm", "y": 1000.0, "sp": 777.0}]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: rows})
    frame = U.load_frame(lomo_dir, table_path, sorted(raw_dir.glob("*.parquet")))
    row = frame.loc[frame.MVT_ID_mvt == 503.0].iloc[0]
    assert row.sp == pytest.approx(777.0)


@pytest.mark.parametrize("decided_by, want", [("unmatched_body", True), ("storm", True),
                                              ("rome_e2w", False), ("arc", False), ("rome", False)])
def test_load_frame_scope_matches_the_registered_decided_by_values(tmp_path, decided_by, want):
    """Fails when SCOPE drifts to include or exclude a decided_by value it should not."""
    row = {"id": 600, "lane": "unmatched", "decided_by": decided_by, "y": 1000.0, "est": 900.0, "cls": "A1"}
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: [row]})
    frame = U.load_frame(lomo_dir, table_path, sorted(raw_dir.glob("*.parquet")))
    assert bool(frame.loc[frame.MVT_ID_mvt == 600.0, "scope"].iloc[0]) is want


@pytest.mark.parametrize("storm_a, want", [(np.nan, False), (300.0, False), (300.5, True)])
def test_load_frame_storm_boundary(tmp_path, storm_a, want):
    """Fails when the storm threshold flips from > to >= (or NaN stops mapping to False)."""
    row = {"id": 700, "lane": "unmatched", "decided_by": "storm", "y": 1000.0, "est": 900.0, "cls": "A1",
           "storm_A": storm_a}
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: [row]})
    frame = U.load_frame(lomo_dir, table_path, sorted(raw_dir.glob("*.parquet")))
    assert bool(frame.loc[frame.MVT_ID_mvt == 700.0, "storm"].iloc[0]) is want


def test_load_frame_refuses_a_matched_row_with_no_table_row(tmp_path):
    """Fails when the table-coverage check is narrowed back to unmatched rows only (AMEND-A1 item 6: every
    baseline row, matched or unmatched, needs a table row; today only unmatched rows are checked)."""
    rows = [{"id": 900, "lane": "matched", "decided_by": "base_lightgbm", "y": 1000.0}]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: rows})
    t = pd.read_parquet(table_path)
    t = t[t.MVT_ID_mvt != 900.0]
    table_path.unlink()
    t.to_parquet(table_path, index=False)
    with pytest.raises(ValueError, match="no table row"):
        U.load_frame(lomo_dir, table_path, sorted(raw_dir.glob("*.parquet")))


@pytest.mark.parametrize("p_value", [0.25, 0.8])
def test_load_frame_p_equals_unm_p_fill_on_unmatched_rows(tmp_path, p_value):
    """Fails when p is zeroed or defaulted instead of being copied from unm_p_fill."""
    row = {"id": 800, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "est": 900.0, "cls": "A1",
          "p": p_value}
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: [row]})
    frame = U.load_frame(lomo_dir, table_path, sorted(raw_dir.glob("*.parquet")))
    assert frame.loc[frame.MVT_ID_mvt == 800.0, "p"].iloc[0] == pytest.approx(p_value)


def test_load_frame_months_opens_only_the_requested_months(tmp_path):
    """Builds a normal 12-month world, then corrupts January's and July's LOMO and raw files with garbage bytes.
    Fails when load_frame(months=DESIGN_MONTHS) still opens a non-requested month's file (it would then raise on
    the garbage bytes instead of succeeding and excluding those months), or when the default (all months) stops
    raising on the same garbage (proving it otherwise would have opened them)."""
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path)
    for m in (1, 7):
        (lomo_dir / f"oof_M{m:02d}.parquet").write_bytes(b"not a parquet file")
        (raw_dir / f"training_2025-{m:02d}-fixture.parquet").write_bytes(b"not a parquet file either")
    raw_files = sorted(raw_dir.glob("*.parquet"))

    frame = U.load_frame(lomo_dir, table_path, raw_files, months=U.DESIGN_MONTHS)
    months_seen = set(frame.month.astype(int).unique().tolist())
    assert months_seen == set(U.DESIGN_MONTHS)
    assert 1 not in months_seen and 7 not in months_seen

    with pytest.raises(Exception):   # the garbage month-1/7 files: a real parquet-read failure, not our ValueError
        U.load_frame(lomo_dir, table_path, raw_files)


def test_load_frame_months_missing_raw_file_for_requested_month_raises(tmp_path):
    """Fails when a requested month with no matching raw file is silently skipped instead of refused."""
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path)
    (raw_dir / "training_2025-03-fixture.parquet").unlink()
    raw_files = sorted(raw_dir.glob("*.parquet"))
    with pytest.raises(ValueError, match="no raw file"):
        U.load_frame(lomo_dir, table_path, raw_files, months=U.DESIGN_MONTHS)


# ==================================================================================== u_prime

def test_u_prime_formula_on_hand_numbers():
    """Fails when the (1 - p) weight on the schedule-copy term is dropped or swapped with p."""
    out = U.u_prime(np.array([0.3]), np.array([1000.0]), np.array([500.0]), np.array([50.0]))
    assert out[0] == pytest.approx(0.3 * 1000.0 + 0.7 * 550.0)


def test_u_prime_falls_back_to_est_plus_c_when_p_or_sp_is_nan():
    """Fails when the NaN fallback branch is removed (a NaN would then propagate through the arithmetic)."""
    p = np.array([np.nan, 0.3, np.nan])
    sp = np.array([1000.0, np.nan, np.nan])
    est = np.array([500.0, 500.0, 500.0])
    c = np.array([50.0, 50.0, 50.0])
    np.testing.assert_allclose(U.u_prime(p, sp, est, c), [550.0, 550.0, 550.0])


# ==================================================================================== offsets

def test_offsets_uses_any_lane_a1_rows_and_ignores_a2_and_blank():
    """Fails when the A1-only filter in offsets is dropped or widened to another class."""
    rows = [
        {"id": 0, "cls": "A1", "y": 1100.0, "est": 1000.0, "scope": True},    # diff 100
        {"id": 1, "cls": "A1", "y": 1300.0, "est": 1200.0, "scope": False},   # diff 100 (scope ignored: ANY lane)
        {"id": 2, "cls": "A2", "y": 9000.0, "est": 1000.0, "scope": True},    # must be ignored
        {"id": 3, "cls": "",   "y": 9000.0, "est": 1000.0, "scope": True},    # must be ignored
    ]
    off = U.offsets(_frame(rows), U.DESIGN_MONTHS)
    assert off["by_airport"]["EDDF"] == pytest.approx(100.0)
    assert off["n_by_airport"]["EDDF"] == 2


def test_offsets_ignores_rows_outside_the_900s_bound():
    """Fails when the 900 s bound in offsets is loosened (e.g. changed from <= to < or widened)."""
    rows = [{"id": 0, "cls": "A1", "y": 1000.0, "est": 100.0},     # diff 900: included (boundary)
            {"id": 1, "cls": "A1", "y": 1000.0, "est": 99.0}]      # diff 901: excluded
    off = U.offsets(_frame(rows), U.DESIGN_MONTHS)
    assert off["by_airport"]["EDDF"] == pytest.approx(900.0)
    assert off["n_by_airport"]["EDDF"] == 1


def test_offsets_uses_only_the_requested_months():
    """Fails when the month filter in offsets is dropped (a January row, still inside the 900 s bound so it is not
    caught by that filter instead, would then move the design result)."""
    design_rows = [{"id": 0, "cls": "A1", "y": 1100.0, "est": 1000.0, "month": 2}]
    wild_january = {"id": 1, "cls": "A1", "y": 1890.0, "est": 1000.0, "month": 1}   # diff 890: inside the bound
    off_without = U.offsets(_frame(design_rows), U.DESIGN_MONTHS)
    off_with = U.offsets(_frame(design_rows + [wild_january]), U.DESIGN_MONTHS)
    assert off_with == off_without


def test_offsets_pools_an_airport_under_200_rows():
    """Fails when the 200-row pooling rule uses <= instead of < (off by one at the boundary)."""
    a_rows = [{"id": i, "cls": "A1", "ap": "AAAA", "y": 1050.0, "est": 1000.0} for i in range(199)]
    b_rows = [{"id": 200 + i, "cls": "A1", "ap": "BBBB", "y": 1080.0, "est": 1000.0} for i in range(200)]
    off = U.offsets(_frame(a_rows + b_rows), U.DESIGN_MONTHS)
    pooled = (199 * 50.0 + 200 * 80.0) / 399.0
    assert off["pooled"] == pytest.approx(pooled)
    assert off["by_airport"]["AAAA"] == pytest.approx(pooled)
    assert off["by_airport"]["BBBB"] == pytest.approx(80.0)
    assert off["pooled_used"] == ["AAAA"]


def test_offsets_exclude_removes_exactly_that_airport_day():
    """Fails when exclude is ignored, or matches on airport or day alone instead of the pair."""
    rows = [{"id": 0, "cls": "A1", "ap": "EDDF", "day": "2025-02-10", "y": 1100.0, "est": 1000.0},
            {"id": 1, "cls": "A1", "ap": "EDDF", "day": "2025-02-11", "y": 1300.0, "est": 1000.0}]
    off_all = U.offsets(_frame(rows), U.DESIGN_MONTHS)
    off_ex = U.offsets(_frame(rows), U.DESIGN_MONTHS, exclude=("EDDF", "2025-02-10"))
    assert off_all["n_by_airport"]["EDDF"] == 2
    assert off_ex["n_by_airport"]["EDDF"] == 1
    assert off_ex["by_airport"]["EDDF"] == pytest.approx(300.0)


def test_offsets_raises_when_nothing_qualifies():
    """Fails when the empty-result guard is removed (would otherwise crash or silently return NaN)."""
    rows = [{"id": 0, "cls": "A2", "y": 1100.0, "est": 1000.0}]
    with pytest.raises(ValueError, match="no qualifying row"):
        U.offsets(_frame(rows), U.DESIGN_MONTHS)


# ==================================================================================== weights

def test_weights_recovers_a_planted_calm_weight_and_floors_an_empty_storm_regime():
    """Fails when the calm-regime fit formula is broken (wrong numerator/denominator), or an empty storm regime
    raises instead of taking the floor rule."""
    off = {"by_airport": {"EDDF": 50.0}, "pooled": 50.0}
    rows = _planted_rows(50.0, 0.7, 10, 0, storm=False)
    w = U.weights(_frame(rows), off, U.DESIGN_MONTHS)
    assert w["w0"] == pytest.approx(0.7)
    assert w["n0"] == 10
    assert w["n1"] == 0
    assert w["w1_rule"] == "floor"
    assert w["w1"] == pytest.approx(max(0.7, U.W1_FLOOR))


@pytest.mark.parametrize("ratio, want", [(1.3, 1.0), (-0.2, 0.0)])
def test_weights_clips_w0_to_0_1(ratio, want):
    """Fails when the clip in weights is removed or its bounds are changed."""
    off = {"by_airport": {"EDDF": 0.0}, "pooled": 0.0}
    rows = _planted_rows(0.0, ratio, 5, 0, storm=False)
    w = U.weights(_frame(rows), off, U.DESIGN_MONTHS)
    assert w["w0"] == pytest.approx(want)
    assert w["w0_raw"] == pytest.approx(ratio)


def test_weights_fits_the_storm_regime_separately():
    """Fails when storm and calm rows are pooled into one fit instead of two."""
    off = {"by_airport": {"EDDF": 0.0}, "pooled": 0.0}
    calm = _planted_rows(0.0, 0.7, 40, 0, storm=False)
    storm = _planted_rows(0.0, 0.9, 40, 1000, storm=True)
    w = U.weights(_frame(calm + storm), off, U.DESIGN_MONTHS)
    assert w["w0"] == pytest.approx(0.7)
    assert w["w1"] == pytest.approx(0.9)
    assert w["w1_rule"] == "fit"
    assert w["n1"] == 40


def test_weights_floors_w1_under_30_storm_rows_and_fits_at_30():
    """Fails when the 30-row floor rule's boundary is off by one (<= instead of <)."""
    off = {"by_airport": {"EDDF": 0.0}, "pooled": 0.0}
    calm = _planted_rows(0.0, 0.7, 50, 0, storm=False)
    storm29 = _planted_rows(0.0, 0.95, 29, 2000, storm=True)
    storm30 = _planted_rows(0.0, 0.95, 30, 3000, storm=True)
    w29 = U.weights(_frame(calm + storm29), off, U.DESIGN_MONTHS)
    w30 = U.weights(_frame(calm + storm30), off, U.DESIGN_MONTHS)
    assert w29["w1_rule"] == "floor"
    assert w29["w1"] == pytest.approx(max(0.7, U.W1_FLOOR))
    assert w30["w1_rule"] == "fit"
    assert w30["w1"] == pytest.approx(0.95)


def test_weights_raises_on_zero_denominator_naming_the_regime():
    """Fails when the zero-denominator guard is removed (would otherwise return NaN/inf silently)."""
    off = {"by_airport": {"EDDF": 0.0}, "pooled": 0.0}
    rows = [{"id": i, "cls": "A1", "scope": True, "v": 1000.0, "y": 1000.0, "est": 1000.0} for i in range(5)]
    with pytest.raises(ValueError, match="calm"):
        U.weights(_frame(rows), off, U.DESIGN_MONTHS)


def test_weights_ignores_out_of_scope_and_a2_rows():
    """Fails when the scope filter in weights is dropped (an out-of-scope or A2 decoy would then move w0)."""
    off = {"by_airport": {"EDDF": 0.0}, "pooled": 0.0}
    good = _planted_rows(0.0, 0.7, 20, 0, storm=False)
    decoy_scope = [dict(r, scope=False, y=999999.0) for r in _planted_rows(0.0, 0.2, 20, 100, storm=False)]
    decoy_a2 = [dict(r, cls="A2", y=999999.0) for r in _planted_rows(0.0, 0.2, 20, 200, storm=False)]
    w = U.weights(_frame(good + decoy_scope + decoy_a2), off, U.DESIGN_MONTHS)
    assert w["w0"] == pytest.approx(0.7)
    assert w["n0"] == 20


def test_weights_ignores_scored_months():
    """Fails when weights(), asked for DESIGN_MONTHS, still lets a January/July decoy row move the fit."""
    off = {"by_airport": {"EDDF": 0.0}, "pooled": 0.0}
    good = _planted_rows(0.0, 0.7, 20, 0, storm=False)
    decoy_january = [dict(r, month=1, y=999999.0) for r in _planted_rows(0.0, 0.2, 20, 100, storm=False)]
    w = U.weights(_frame(good + decoy_january), off, U.DESIGN_MONTHS)
    assert w["w0"] == pytest.approx(0.7)


def test_weights_exclude_removes_exactly_that_airport_day():
    """Fails when exclude in weights() is ignored, or matches on airport or day alone instead of the pair."""
    off = {"by_airport": {"EDDF": 0.0}, "pooled": 0.0}
    keep = _planted_rows(0.0, 0.5, 40, 0, storm=True, day="2025-02-20")
    event = _planted_rows(0.0, 0.95, 30, 1000, storm=True, day="2025-02-10")
    frame = _frame(keep + event + _calm_filler())
    w_all = U.weights(frame, off, U.DESIGN_MONTHS)
    w_ex = U.weights(frame, off, U.DESIGN_MONTHS, exclude=("EDDF", "2025-02-10"))
    assert w_ex["w1"] == pytest.approx(0.5)
    assert w_ex["n1"] == 40
    assert w_all["w1"] != pytest.approx(0.5, abs=1e-6)


def test_weights_top10_share_on_hand_numbers_and_leaves_w0_unchanged():
    """With v=0, y=1, each row's (u'-v)(y-v) term equals its own est value directly (u'=est+c, c=0; p/sp default
    NaN so u'=est): 10 rows of est=100 (identical large terms) and 2 rows of est=1 (small terms), all calm, no
    storm rows at all. Fails when the top-10 share is computed over the smallest instead of the largest |term|
    values, when it is not divided by the sum of every term, or when adding the diagnostic changes w0 itself; also
    fails when an empty storm regime's share is not None."""
    off = {"by_airport": {"EDDF": 0.0}, "pooled": 0.0}
    big = [{"id": i, "cls": "A1", "scope": True, "v": 0.0, "y": 1.0, "est": 100.0, "storm": False} for i in range(10)]
    small = [{"id": 10 + i, "cls": "A1", "scope": True, "v": 0.0, "y": 1.0, "est": 1.0, "storm": False}
            for i in range(2)]
    w = U.weights(_frame(big + small), off, U.DESIGN_MONTHS)

    expected_w0 = (10 * 100.0 + 2 * 1.0) / (10 * 100.0 ** 2 + 2 * 1.0 ** 2)
    assert w["w0"] == pytest.approx(expected_w0)
    assert w["w0_top10_share"] == pytest.approx((10 * 100.0) / (10 * 100.0 + 2 * 1.0))
    assert w["n1"] == 0
    assert w["w1_top10_share"] is None


# ==================================================================================== constants

def test_constants_bundles_offsets_and_weights():
    """Fails when constants() drops a key from its result or stops restricting to the requested months."""
    rows = _planted_rows(0.0, 0.7, 250, 0, storm=False)
    const = U.constants(_frame(rows), U.DESIGN_MONTHS)
    assert set(const) == {"offsets", "weights", "months", "registration"}
    assert const["registration"] == "17a570a6"
    assert const["months"] == sorted(U.DESIGN_MONTHS)


def test_constants_never_sees_january_or_july():
    """Fails when constants() (or offsets/weights beneath it) stops filtering by month, letting a January/July
    row with garbage values change the design-fit constants."""
    good = _planted_rows(0.0, 0.7, 250, 0, storm=False)
    frame_a = _frame(good)
    garbage_jan = [dict(r, month=1, y=-999999.0, est=123456.0) for r in _planted_rows(5.0, 0.1, 50, 10000, storm=False)]
    garbage_jul = [dict(r, month=7, y=777777.0, est=-54321.0) for r in _planted_rows(9.0, 0.3, 50, 20000, storm=True)]
    frame_b = _frame(good + garbage_jan + garbage_jul)
    assert U.constants(frame_a, U.DESIGN_MONTHS) == U.constants(frame_b, U.DESIGN_MONTHS)


# ==================================================================================== check_constants (AMEND-A1 item 6)

def _build_const(tmp_path, design_rows=None):
    """Runs the real `constants` CLI command on a minimal fixture (month 3 carries design rows) and returns
    (const_dict, table_path, lomo_dir)."""
    design_rows = design_rows or [{"id": 9400 + i, "lane": "unmatched", "decided_by": "unmatched_body",
                                   "y": 1000.0, "pred": 900.0, "est": 700.0, "cls": "A1"} for i in range(5)]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {3: design_rows})
    out = tmp_path / "const.json"
    assert U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
                  "--raw-dir", str(raw_dir), "--out", str(out)]) == 0
    return json.loads(out.read_text()), table_path, lomo_dir


def test_check_constants_passes_on_a_matching_file(tmp_path):
    """Fails when check_constants raises on a constants file that genuinely matches its current inputs."""
    const, table_path, lomo_dir = _build_const(tmp_path)
    U.check_constants(const, table_path, lomo_dir)   # must not raise


def test_check_constants_wrong_registration_raises(tmp_path):
    """Fails when the registration check is removed."""
    const, table_path, lomo_dir = _build_const(tmp_path)
    const["registration"] = "deadbeef"
    with pytest.raises(ValueError, match="registration"):
        U.check_constants(const, table_path, lomo_dir)


def test_check_constants_wrong_months_raises(tmp_path):
    """Fails when the months check is removed."""
    const, table_path, lomo_dir = _build_const(tmp_path)
    const["months"] = [1, 2, 3]
    with pytest.raises(ValueError, match="months"):
        U.check_constants(const, table_path, lomo_dir)


def test_check_constants_changed_table_raises(tmp_path):
    """Fails when the table_sha256 check is removed."""
    const, table_path, lomo_dir = _build_const(tmp_path)
    t = pd.read_parquet(table_path)
    t.loc[0, "usable"] = not bool(t.loc[0, "usable"])
    table_path.unlink()
    t.to_parquet(table_path, index=False)
    with pytest.raises(ValueError, match="table_sha256"):
        U.check_constants(const, table_path, lomo_dir)


def test_check_constants_changed_baseline_file_raises(tmp_path):
    """Fails when the per-lomo-file sha256 check is removed, letting a changed design-month baseline file through."""
    const, table_path, lomo_dir = _build_const(tmp_path)
    f = lomo_dir / f"oof_M{sorted(U.DESIGN_MONTHS)[0]:02d}.parquet"
    df = pd.read_parquet(f)
    df.loc[0, "y"] = df.loc[0, "y"] + 1.0
    f.unlink()
    df.to_parquet(f, index=False)
    with pytest.raises(ValueError, match="sha256"):
        U.check_constants(const, table_path, lomo_dir)


def test_check_constants_eleventh_file_raises(tmp_path):
    """Fails when the exactly-ten-files check is removed, letting an eleventh entry (e.g. a scored month) through."""
    const, table_path, lomo_dir = _build_const(tmp_path)
    extra_path = lomo_dir / "oof_M01.parquet"
    const["lomo_files"] = const["lomo_files"] + [{"file": extra_path.name, "sha256": U._sha256(extra_path)}]
    with pytest.raises(ValueError, match="lomo_files"):
        U.check_constants(const, table_path, lomo_dir)


# ==================================================================================== apply

def test_apply_a1_calm_storm_and_a2_half_weight_on_hand_numbers():
    """Fails when the storm/calm weight selection is swapped, or the A2 half-weight factor is dropped."""
    off = {"by_airport": {"EDDF": 100.0}, "pooled": 100.0}
    const = {"offsets": off, "weights": {"w0": 0.6, "w1": 0.9}}
    rows = [
        {"id": 0, "cls": "A1", "scope": True, "storm": False, "v": 1000.0, "p": 0.4, "sp": 500.0, "est": 600.0},
        {"id": 1, "cls": "A1", "scope": True, "storm": True, "v": 1000.0, "p": 0.4, "sp": 500.0, "est": 600.0},
        {"id": 2, "cls": "A2", "scope": True, "storm": False, "v": 1000.0, "p": 0.4, "sp": 500.0, "est": 600.0},
    ]
    # u = est + c = 700; u' = 0.4*500 + 0.6*700 = 620; u' - v = -380.
    out = U.apply(_frame(rows), const)
    assert out[0] == pytest.approx(1000.0 + 0.6 * (620.0 - 1000.0))        # calm: 772
    assert out[1] == pytest.approx(1000.0 + 0.9 * (620.0 - 1000.0))        # storm: 658
    assert out[2] == pytest.approx(1000.0 + 0.5 * 0.6 * (620.0 - 1000.0))  # A2 half-weight calm: 886


def test_apply_floors_the_result_at_one_second_and_rounds():
    """Fails when the floor-at-1 clamp is removed or changed to floor at 0."""
    off = {"by_airport": {"EDDF": 0.0}, "pooled": 0.0}
    const = {"offsets": off, "weights": {"w0": 0.9, "w1": 0.9}}
    rows = [{"id": 0, "cls": "A1", "scope": True, "storm": False, "v": 5.0, "est": -10000.0}]
    out = U.apply(_frame(rows), const)
    assert out[0] == pytest.approx(1.0)


def test_apply_passes_through_v_unchanged_outside_scope_or_class():
    """Fails when a row outside scope or outside {A1, A2} is accidentally rounded or blended."""
    off = {"by_airport": {}, "pooled": 0.0}
    const = {"offsets": off, "weights": {"w0": 1.0, "w1": 1.0}}
    rows = [
        {"id": 0, "cls": "A1", "scope": False, "v": 111.7, "est": 0.0},   # out of scope (e.g. matched)
        {"id": 1, "cls": "", "scope": True, "v": 222.3, "est": 0.0},     # blank class
    ]
    out = U.apply(_frame(rows), const)
    np.testing.assert_allclose(out, [111.7, 222.3])


def test_apply_uses_the_pooled_offset_for_a_missing_airport():
    """Fails when a missing airport silently falls back to c = 0 instead of the pooled value."""
    off = {"by_airport": {}, "pooled": 42.0}
    const = {"offsets": off, "weights": {"w0": 1.0, "w1": 1.0}}
    rows = [{"id": 0, "ap": "ZZZZ", "cls": "A1", "scope": True, "storm": False, "v": 1000.0, "est": 600.0}]
    # u' = est + pooled = 642 (p, sp NaN); new = v + 1.0*(642 - 1000) = 642.
    out = U.apply(_frame(rows), const)
    assert out[0] == pytest.approx(642.0)


def test_apply_returns_v_unchanged_when_nothing_is_in_scope():
    """Fails when the early-exit for an all-out-of-population frame crashes or mis-shapes the output."""
    off = {"by_airport": {}, "pooled": 0.0}
    const = {"offsets": off, "weights": {"w0": 0.5, "w1": 0.5}}
    rows = [{"id": 0, "cls": "", "scope": False, "v": 321.0}]
    out = U.apply(_frame(rows), const)
    np.testing.assert_allclose(out, [321.0])


# ==================================================================================== population / records

def test_population_selects_scope_and_class_and_month():
    """Fails when population() drops the month filter or widens the class set beyond {A1, A2}."""
    rows = [
        {"id": 0, "cls": "A1", "scope": True, "month": 2},
        {"id": 1, "cls": "A2", "scope": True, "month": 2},
        {"id": 2, "cls": "", "scope": True, "month": 2},
        {"id": 3, "cls": "A1", "scope": False, "month": 2},
        {"id": 4, "cls": "A1", "scope": True, "month": 1},
    ]
    pop = U.population(_frame(rows), U.DESIGN_MONTHS)
    np.testing.assert_array_equal(pop, [True, True, False, False, False])


def test_records_share_ids_and_labels_and_differ_only_in_pred():
    """Fails when records() leaks a non-population row into the compare records."""
    rows = [{"id": 0, "cls": "A1", "scope": True, "month": 2, "v": 900.0, "y": 1000.0},
            {"id": 1, "cls": "", "scope": True, "month": 2, "v": 800.0, "y": 700.0}]
    frame = _frame(rows)
    new = frame.v.to_numpy() + 50.0
    base, cand = U.records(frame, new, U.DESIGN_MONTHS)
    assert len(base) == 1 and len(cand) == 1
    assert base.MVT_ID_mvt.tolist() == cand.MVT_ID_mvt.tolist() == [0.0]
    assert base.y.tolist() == cand.y.tolist() == [1000.0]
    assert base.pred.tolist() == [900.0]
    assert cand.pred.tolist() == [950.0]


# ==================================================================================== h1

def _h1_frame(n_jan=10, n_jul=10, v=900.0, y=1000.0):
    rows = []
    rid = 0
    for _ in range(n_jan):
        rows.append({"id": rid, "cls": "A1", "scope": True, "month": 1, "v": v, "y": y, "day": "2025-01-05"})
        rid += 1
    for _ in range(n_jul):
        rows.append({"id": rid, "cls": "A1", "scope": True, "month": 7, "v": v, "y": y, "day": "2025-07-05"})
        rid += 1
    return _frame(rows)


def test_h1_true_when_gate_promotes_and_rmse_falls_10_percent(monkeypatch):
    """Fails when the rmse-fall bar (H1_MIN_RMSE_FALL) is dropped from the verdict's AND condition."""
    frame = _h1_frame()   # y = 1000, v = 900 everywhere -> rmse_base = 100 exactly
    monkeypatch.setattr(U, "apply", lambda f, c: np.full(len(f), 910.05))   # y - new = 89.95 -> fall = 0.1005
    monkeypatch.setattr(U.P, "compare", lambda a, b, n_boot, seed: {"stub": True})
    monkeypatch.setattr(U.P, "promotion_gate", lambda rep: {"promote": True})
    rep = U.h1(frame, const={}, n_boot=10, seed=0)
    assert rep["verdict"] == "TRUE"
    assert rep["rmse_fall"] >= U.H1_MIN_RMSE_FALL


def test_h1_false_at_9_9_percent_rmse_fall(monkeypatch):
    """Fails when the 0.10 threshold constant drifts (e.g. to 0.05, which 9.9% would then clear)."""
    frame = _h1_frame()
    monkeypatch.setattr(U, "apply", lambda f, c: np.full(len(f), 909.9))    # y - new = 90.1 -> fall = 0.099
    monkeypatch.setattr(U.P, "compare", lambda a, b, n_boot, seed: {"stub": True})
    monkeypatch.setattr(U.P, "promotion_gate", lambda rep: {"promote": True})
    rep = U.h1(frame, const={}, n_boot=10, seed=0)
    assert rep["verdict"] == "FALSE"
    assert rep["rmse_fall"] < U.H1_MIN_RMSE_FALL


def test_rmse_ok_boundary_is_greater_or_equal_not_strictly_greater():
    """Direct test of the H1 RMSE bar at the exact float64 value of H1_MIN_RMSE_FALL (unreachable by constructing
    rmse_base/rmse_new, since 1.0 - ratio never lands bit-exact on 0.10 from a ratio near 0.9 — verified: no hit
    within 200 ULPs across many bases). Fails when the comparison uses > instead of >=, and separately exercises
    the NaN short-circuit (an empty population must never pass)."""
    assert U._rmse_ok(U.H1_MIN_RMSE_FALL) is True
    assert U._rmse_ok(np.nextafter(U.H1_MIN_RMSE_FALL, 0.0)) is False
    assert U._rmse_ok(float("nan")) is False


def test_h1_false_when_gate_does_not_promote_even_with_a_50_percent_fall(monkeypatch):
    """Fails when the gate's promote flag is ignored by the verdict."""
    frame = _h1_frame()
    monkeypatch.setattr(U, "apply", lambda f, c: np.full(len(f), 950.0))   # fall = 1 - 50/100 = 0.50
    monkeypatch.setattr(U.P, "compare", lambda a, b, n_boot, seed: {"stub": True})
    monkeypatch.setattr(U.P, "promotion_gate", lambda rep: {"promote": False})
    rep = U.h1(frame, const={}, n_boot=10, seed=0)
    assert rep["verdict"] == "FALSE"


def test_h1_false_when_july_has_no_population_row(monkeypatch):
    """Fails when the both-scored-months-populated check is dropped from the verdict."""
    frame = _h1_frame(n_jan=10, n_jul=0)
    monkeypatch.setattr(U, "apply", lambda f, c: np.full(len(f), 910.05))   # a 10%+ fall: would be TRUE otherwise
    monkeypatch.setattr(U.P, "compare", lambda a, b, n_boot, seed: {"stub": True})
    monkeypatch.setattr(U.P, "promotion_gate", lambda rep: {"promote": True})
    rep = U.h1(frame, const={}, n_boot=10, seed=0)
    assert rep["verdict"] == "FALSE"
    assert rep["n_by_month"][7] == 0


def test_h1_with_the_real_protocols_compare_and_promotion_gate(monkeypatch):
    """Integration contract check: fails if protocols.compare / promotion_gate's output shape ever drifts from
    what h1() and promotion_gate() depend on (body, body_by_month, body_excluding_top10_rows, ci95, gain_bmse,
    promote, checks)."""
    rng = np.random.default_rng(0)
    rows = []
    rid = 0
    for month in (1, 7):
        for d in range(10):
            for _ in range(10):
                y = float(1000 + rng.normal(0, 50))
                rows.append({"id": rid, "cls": "A1", "scope": True, "month": month, "v": y - 100.0, "y": y,
                            "day": f"2025-{month:02d}-{1 + d:02d}", "ap": "EDDF" if rid % 2 == 0 else "LIRF"})
                rid += 1
    frame = _frame(rows)
    assert len(frame) == 200
    monkeypatch.setattr(U, "apply", lambda f, c: f.v.to_numpy() + 80.0)   # a real, substantial improvement
    rep = U.h1(frame, const={}, n_boot=50, seed=0)
    assert rep["verdict"] in ("TRUE", "FALSE")
    assert {"body", "body_by_month", "body_excluding_top10_rows", "ci95", "gain_bmse", "n"} <= set(rep["compare"])
    assert {"1", "7"} <= set(rep["compare"]["body_by_month"])
    assert set(rep["gate"]) == {"promote", "checks"}


def _order_frame(mode, n_days=10, rows_per_day=15, seed=0):
    """Rows spread over n_days days of January and July each, built through apply()'s real machinery (p=sp=0.0,
    not NaN, and a constants dict with every offset 0 and w0=w1=1) so the candidate blend lands exactly on a chosen
    value: u_prime(p=0, sp=0, est, c=0) = 0*0 + 1*(est+0) = est; apply() then computes new = v + 1*1*(est-v) = est
    (rounded, floored at 1 s). mode="candidate_better": the chain (v) is 300 s off the label and the blend (est) is
    exact (est=y). mode="chain_better" (the mirror): the chain is exact (v=y) and the blend is 300 s off
    (est=y-300). y is rounded to a whole second so the rint() in apply() is exact."""
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    for month in (1, 7):
        for d in range(n_days):
            for _ in range(rows_per_day):
                y = float(round(900.0 + rng.normal(0.0, 50.0)))
                if mode == "candidate_better":
                    v, est = y - 300.0, y
                else:
                    v, est = y, y - 300.0
                rows.append({"id": rid, "cls": "A1", "scope": True, "month": month, "y": y, "v": v, "p": 0.0,
                            "sp": 0.0, "est": est, "ap": "EDDF", "day": f"2025-{month:02d}-{1 + d:02d}"})
                rid += 1
    return _frame(rows)


_ORDER_CONST = {"offsets": {"by_airport": {}, "pooled": 0.0}, "weights": {"w0": 1.0, "w1": 1.0}}


def test_h1_pins_compare_argument_order_with_real_protocols():
    """The chain (base) is 300 s off every row and the blend (candidate) is exact, with the REAL protocols.compare
    and promotion_gate (no monkeypatch). Fails when h1() calls P.compare(cand, base) instead of P.compare(base,
    cand): swapping the arguments flips gain_bmse's sign and would turn this TRUE verdict FALSE."""
    frame = _order_frame("candidate_better")
    rep = U.h1(frame, _ORDER_CONST, n_boot=200, seed=0)
    assert rep["verdict"] == "TRUE"
    assert rep["compare"]["gain_bmse"] > 0
    assert rep["compare"]["body"]["gain_bmse"] > 0


def test_h1_mirrored_argument_order_frame_is_false():
    """Mirror of the above: the chain (base) is exact and the blend (candidate) is 300 s off, with the REAL
    protocols.compare and promotion_gate. Fails when h1() calls P.compare(cand, base) instead of P.compare(base,
    cand): the swap would flip this negative gain positive and the verdict TRUE."""
    frame = _order_frame("chain_better")
    rep = U.h1(frame, _ORDER_CONST, n_boot=200, seed=0)
    assert rep["verdict"] == "FALSE"
    assert rep["compare"]["gain_bmse"] < 0


def test_h1_exclude_airports_default_is_unchanged():
    """Fails when the default exclude_airports=() still drops rows (e.g. an off-by-default filter applied anyway)."""
    frame = _order_frame("candidate_better", n_days=3, rows_per_day=2)
    rep_default = U.h1(frame, _ORDER_CONST, n_boot=50, seed=0)
    rep_explicit_empty = U.h1(frame, _ORDER_CONST, n_boot=50, seed=0, exclude_airports=())
    assert rep_default["n_by_month"] == rep_explicit_empty["n_by_month"]
    assert rep_default["verdict"] == rep_explicit_empty["verdict"]


def test_h1_exclude_airports_eham_flips_verdict_and_filters_n_by_month(monkeypatch):
    """Plants EHAM rows whose blend is terrible (left badly wrong, 10,000 s off) and non-EHAM rows whose blend is
    perfect, 20 days per scored month so the real gate's "drop the 10 largest body rows" check still has rows left
    after exclusion (only 10 remaining rows would drop to zero and give a spurious zero gain). Fails when
    exclude_airports does not remove the EHAM rows from the compared population (base, candidate, RMSEs,
    n_by_month): the default verdict would then stay FALSE even after asking to exclude EHAM, or n_by_month would
    still count the excluded rows."""
    def fake_apply(f, c):
        out = f.v.to_numpy().copy()
        eham = (f.ap.to_numpy() == "EHAM")
        out[~eham] = f.y.to_numpy()[~eham]            # perfect for non-EHAM
        out[eham] = f.v.to_numpy()[eham] - 10000.0    # terrible for EHAM
        return out
    monkeypatch.setattr(U, "apply", fake_apply)

    rows = []
    rid = 0
    for month in (1, 7):
        for d in range(20):
            rows.append({"id": rid, "cls": "A1", "scope": True, "month": month, "ap": "EHAM", "v": 700.0,
                        "y": 1000.0, "day": f"2025-{month:02d}-{1 + d:02d}"})
            rid += 1
            rows.append({"id": rid, "cls": "A1", "scope": True, "month": month, "ap": "LIRF", "v": 700.0,
                        "y": 1000.0, "day": f"2025-{month:02d}-{1 + d:02d}"})
            rid += 1
    frame = _frame(rows)

    rep_default = U.h1(frame, const={}, n_boot=50, seed=0)
    rep_excl = U.h1(frame, const={}, n_boot=50, seed=0, exclude_airports=("EHAM",))

    assert rep_default["verdict"] == "FALSE"
    assert rep_excl["verdict"] == "TRUE"
    assert rep_default["n_by_month"] == {1: 40, 7: 40}
    assert rep_excl["n_by_month"] == {1: 20, 7: 20}


def test_h1_exclude_airports_one_scored_month_emptied_is_false_without_raising(monkeypatch):
    """Every July population row is at EHAM; excluding EHAM leaves July empty while January still has rows at
    another airport (the already-handled "one month empty" path: protocols.compare runs fine on January's rows
    alone, since the combined record is not empty). protocols.compare/promotion_gate and the RMSE fall are forced
    to otherwise-passing values (the same isolation test_h1_false_when_july_has_no_population_row uses) so only
    the months_ok condition is exercised here. Fails when months_ok stops catching the now-empty July population
    after exclusion, or when n_by_month is computed before exclusion instead of after."""
    monkeypatch.setattr(U, "apply", lambda f, c: np.full(len(f), 910.05))   # a comfortable 10%+ RMSE fall
    monkeypatch.setattr(U.P, "compare", lambda a, b, n_boot, seed: {"stub": True})
    monkeypatch.setattr(U.P, "promotion_gate", lambda rep: {"promote": True})
    rows = []
    for d in range(10):
        rows.append({"id": d, "cls": "A1", "scope": True, "month": 1, "ap": "LIRF", "v": 900.0, "y": 1000.0,
                    "day": f"2025-01-{1 + d:02d}"})
    for d in range(10):
        rows.append({"id": 100 + d, "cls": "A1", "scope": True, "month": 7, "ap": "EHAM", "v": 900.0, "y": 1000.0,
                    "day": f"2025-07-{1 + d:02d}"})
    frame = _frame(rows)
    rep = U.h1(frame, const={}, n_boot=10, seed=0, exclude_airports=("EHAM",))
    assert rep["verdict"] == "FALSE"
    assert rep["n_by_month"] == {1: 10, 7: 0}
    assert rep["reason"] == "a scored month has no population row"


def test_h1_exclude_airports_emptying_the_whole_population_does_not_raise(monkeypatch):
    """Every population row, in both scored months, is at EHAM; excluding EHAM leaves nothing at all. Fails when
    h1() still calls protocols.compare on a record that shares no row (it would raise
    prc.pipeline.errors.ValidationError instead of returning a graceful FALSE) -- today's code does exactly that
    for any frame whose population is fully empty, exclude_airports or not."""
    monkeypatch.setattr(U, "apply", lambda f, c: f.v.to_numpy() + 10.0)
    rows = []
    for month in (1, 7):
        for d in range(3):
            rows.append({"id": month * 1000 + d, "cls": "A1", "scope": True, "month": month, "ap": "EHAM",
                        "v": 900.0, "y": 1000.0, "day": f"2025-{month:02d}-0{d + 1}"})
    frame = _frame(rows)
    rep = U.h1(frame, const={}, n_boot=10, seed=0, exclude_airports=("EHAM",))
    assert rep["verdict"] == "FALSE"
    assert rep["n_by_month"] == {1: 0, 7: 0}
    assert rep["reason"] == "a scored month has no population row"
    assert rep["compare"] is None


# ==================================================================================== h2

def test_h2_not_testable_without_an_eligible_storm_event():
    """Fails when the NOT TESTABLE short-circuit is removed (would otherwise crash computing an empty ratio)."""
    rows = [{"id": 0, "cls": "A1", "scope": True, "month": 2, "storm": False, "v": 1000.0, "y": 1000.0}]
    rep = U.h2(_frame(rows), months=U.DESIGN_MONTHS)
    assert rep["verdict"] == "NOT TESTABLE"
    assert rep["events"] == []


def test_h2_a_7_row_storm_day_is_not_an_event():
    """Fails when the >= 8 row threshold for an h2 event is off by one."""
    rows = _sp_rows(0.5, 7, 0, day="2025-02-10")
    rep = U.h2(_frame(rows), months=U.DESIGN_MONTHS)
    assert rep["verdict"] == "NOT TESTABLE"
    assert rep["events"] == []


def test_h2_leave_one_out_uses_the_others_fit_not_the_events_own_rows():
    """Fails when h2() stops passing exclude=(ap, day) to constants() for each event, letting the event's own
    rows leak into its own leave-one-out weight fit (the event's own rows alone would fit to 0.95, not 0.5)."""
    other_days = [f"2025-04-{1 + i:02d}" for i in range(40)]
    other = _calm_filler()
    for i, d in enumerate(other_days):
        other += _sp_rows(0.5, 1, i, day=d, storm=True)
    event = _sp_rows(0.95, 10, 1000, day="2025-02-10", storm=True)
    frame = _frame(other + event)

    rep = U.h2(frame, months=U.DESIGN_MONTHS)
    assert len(rep["events"]) == 1
    ev = rep["events"][0]
    assert (ev["ap"], ev["day"]) == ("EDDF", "2025-02-10")
    assert ev["n"] == 10

    correct = U.constants(_frame(other), U.DESIGN_MONTHS)
    assert correct["weights"]["w1"] == pytest.approx(0.5, abs=1e-9)
    buggy = U.constants(frame, U.DESIGN_MONTHS)
    assert buggy["weights"]["w1"] != pytest.approx(0.5, abs=1e-3)

    new_correct = U.apply(frame, correct)
    ev_mask = (frame.ap.to_numpy() == "EDDF") & (frame.day.to_numpy() == "2025-02-10")
    y = frame.loc[ev_mask, "y"].to_numpy(dtype="float64")
    blend_rmse_correct = float(np.sqrt(np.mean((y - new_correct[ev_mask]) ** 2)))
    assert ev["blend_rmse"] == pytest.approx(blend_rmse_correct, abs=1e-6)


def test_h2_boundary_ratio_0_60_is_true_0_61_is_false():
    """Fails when the H2 ratio comparison uses < instead of <=, misclassifying the 0.60 boundary."""
    for target_ratio, want in ((0.60, "TRUE"), (0.61, "FALSE")):
        pool = _calm_filler()
        for i in range(40):
            pool += _sp_rows(0.5, 1, i, day=f"2025-04-{1 + i:02d}", storm=True)
        w1_real = U.constants(_frame(pool), U.DESIGN_MONTHS)["weights"]["w1"]
        assert w1_real == pytest.approx(0.5, abs=1e-9)

        v, y = 1000.0, 1100.0
        sp_event = v + (100.0 * (1.0 - target_ratio) / w1_real)
        event = [{"id": 1000 + i, "cls": "A1", "scope": True, "month": 2, "storm": True, "ap": "EDDF",
                 "day": "2025-02-10", "v": v, "est": 0.0, "p": 1.0, "sp": sp_event, "y": y} for i in range(10)]

        rep = U.h2(_frame(pool + event), months=U.DESIGN_MONTHS)
        assert rep["verdict"] == want, target_ratio
        ev = rep["events"][0]
        assert ev["n"] == 10
        assert ev["chain_rmse"] == pytest.approx(100.0, abs=1e-6)
        assert ev["blend_rmse"] == pytest.approx(100.0 * target_ratio, abs=1e-6)


# ==================================================================================== breakdown

def test_breakdown_airport_gains_sum_to_the_total_and_uses_slc_masks(monkeypatch):
    """Fails when by_airport stops partitioning every population row exactly once, or breakdown() stops routing
    the non-lottery split through slc_slices.masks()."""
    rows = [
        {"id": 0, "cls": "A1", "scope": True, "month": 2, "ap": "EDDF", "v": 900.0, "y": 1000.0},
        {"id": 1, "cls": "A1", "scope": True, "month": 2, "ap": "EDDF", "v": 950.0, "y": 1000.0},
        {"id": 2, "cls": "A1", "scope": True, "month": 2, "ap": "LIRF", "v": 900.0, "y": 1000.0},
        {"id": 3, "cls": "A1", "scope": True, "month": 2, "ap": "LIRF", "v": 950.0, "y": 1000.0},
    ]
    frame = _frame(rows)
    new = np.array([950.0, 1000.0, 980.0, 1000.0])

    seen = {}

    def spy_masks(f):
        seen["frame"] = f
        return {"lottery": np.array([False, True, False, False])}
    monkeypatch.setattr(U.SLC, "masks", spy_masks)

    rep = U.breakdown(frame, new, U.DESIGN_MONTHS)
    assert seen["frame"] is frame

    y, v = frame.y.to_numpy(), frame.v.to_numpy()
    expected = ((y - v) ** 2 - (y - new) ** 2) / U.P.BOARD_N
    assert rep["by_airport"]["EDDF"] == pytest.approx(expected[0] + expected[1])
    assert rep["by_airport"]["LIRF"] == pytest.approx(expected[2] + expected[3])
    assert rep["total_gain"] == pytest.approx(expected.sum())
    assert rep["by_airport"]["EDDF"] + rep["by_airport"]["LIRF"] == pytest.approx(rep["total_gain"])
    assert rep["non_lottery_gain"] == pytest.approx(expected[0] + expected[2] + expected[3])   # row 1 is "lottery"
    assert rep["by_class"] == {"A1": pytest.approx(rep["total_gain"])}
    assert rep["by_regime"]["storm"] == pytest.approx(0.0)
    assert rep["by_regime"]["calm"] == pytest.approx(rep["total_gain"])
    assert rep["body_top10_share"] == pytest.approx(1.0)   # fewer than 10 body rows: top10 = everything


# ==================================================================================== CLI

def test_cli_constants_refuses_an_existing_out(tmp_path):
    """Fails when the overwrite guard on --out is removed."""
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path)
    out = tmp_path / "const.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
               "--raw-dir", str(raw_dir), "--out", str(out)])


def test_cli_constants_refuses_on_failing_report_and_writes_no_output(tmp_path):
    """Fails when the constants command stops calling require_stop_rules before writing its output (or runs it
    after work has already started)."""
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, report=_report(callsign_equal=0.1))
    out = tmp_path / "const.json"
    with pytest.raises(ValueError, match="callsign_matched_type_equal_share"):
        U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
               "--raw-dir", str(raw_dir), "--out", str(out)])
    assert not out.exists()


def test_cli_constants_json_has_passing_stop_rules(tmp_path):
    """Fails when a passing run's output JSON loses the stop_rules check dict."""
    design_rows = [{"id": 9200 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0,
                   "pred": 900.0, "est": 700.0, "cls": "A1"} for i in range(5)]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: design_rows})
    out = tmp_path / "const.json"
    assert U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
                  "--raw-dir", str(raw_dir), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["stop_rules"]["ok"] is True


def test_cli_constants_on_garbage_jan_jul_fixture_lists_exactly_ten_lomo_files(tmp_path):
    """End-to-end: the constants command succeeds on a fixture where January's and July's LOMO and raw files are
    garbage bytes, and its output JSON's lomo_files lists exactly the ten design-month files. Fails when the CLI
    stops calling load_frame(months=DESIGN_MONTHS) (it would then crash reading the garbage files), or when
    lomo_files still lists all 12 months instead of exactly the ten design months."""
    design_rows = [{"id": 9300 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0,
                   "pred": 900.0, "est": 700.0, "cls": "A1"} for i in range(5)]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {3: design_rows})
    for m in (1, 7):
        (lomo_dir / f"oof_M{m:02d}.parquet").write_bytes(b"garbage")
        (raw_dir / f"training_2025-{m:02d}-fixture.parquet").write_bytes(b"garbage too")

    out = tmp_path / "const.json"
    assert U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
                  "--raw-dir", str(raw_dir), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert len(rep["lomo_files"]) == 10
    assert sorted(e["file"] for e in rep["lomo_files"]) == [f"oof_M{m:02d}.parquet" for m in sorted(U.DESIGN_MONTHS)]
    assert "oof_M01.parquet" not in {e["file"] for e in rep["lomo_files"]}
    assert "oof_M07.parquet" not in {e["file"] for e in rep["lomo_files"]}


def test_cli_constants_json_has_no_scored_month_information(tmp_path):
    """Fails when the CLI stops dropping scored-month rows before computing constants (the scored-month drop). The
    January/July rows are proper class-A1 unmatched rows (not matched placeholders, which cls-filter out on their
    own regardless of month) so that leaking them would actually move the fit."""
    design_rows = [{"id": 9100 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0,
                   "pred": 900.0, "est": 700.0, "cls": "A1"} for i in range(5)]
    jan_rows = [{"id": 9001, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "est": 400.0,
                "cls": "A1"}]                                                       # diff 600, within the bound
    jul_rows = [{"id": 9002, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "est": 500.0,
                "cls": "A1"}]                                                       # diff 500, within the bound
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {1: jan_rows, 3: design_rows, 7: jul_rows})

    out_a = tmp_path / "const_a.json"
    assert U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
                  "--raw-dir", str(raw_dir), "--out", str(out_a)]) == 0

    # Same ids (so the table, which already holds their est/cls, still applies); only the label changes, which
    # changes the diff (y - est) these rows would contribute if they leaked into the design fit.
    jan_rows_b = [{"id": 9001, "lane": "unmatched", "decided_by": "unmatched_body", "y": 100.0, "est": 400.0,
                  "cls": "A1"}]                                                     # diff -300
    jul_rows_b = [{"id": 9002, "lane": "unmatched", "decided_by": "unmatched_body", "y": 50.0, "est": 500.0,
                  "cls": "A1"}]                                                     # diff -450
    _write_lomo_and_raw(lomo_dir, raw_dir, 1, jan_rows_b)
    _write_lomo_and_raw(lomo_dir, raw_dir, 7, jul_rows_b)

    out_b = tmp_path / "const_b.json"
    assert U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
                  "--raw-dir", str(raw_dir), "--out", str(out_b)]) == 0

    a, b = json.loads(out_a.read_text()), json.loads(out_b.read_text())
    # n_rows must reflect ONLY the design months (5 from month 3 + 1 default placeholder from each of the other 9
    # design months); a missing drop would leave the 2 scored-month rows in `frame` too (n_rows == 16, not 14),
    # identically in both A and B, so this checks the absolute count, not just that A and B agree.
    assert a["n_rows"] == len(design_rows) + 9 == 14
    a.pop("generated_utc"), b.pop("generated_utc")
    a.pop("lomo_files"), b.pop("lomo_files")
    assert a == b


def test_cli_verdict_refuses_an_existing_out_and_writes_both_verdicts(tmp_path):
    """Fails when the verdict command's overwrite guard is removed, or h1/h2 are not both written to --out."""
    design_rows = [{"id": 200 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "pred": 900.0,
                   "est": 700.0, "cls": "A1"} for i in range(5)]
    scored_rows = [{"id": 100 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "pred": 900.0,
                   "est": 700.0, "cls": "A1"} for i in range(5)]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: design_rows, 1: scored_rows})

    const_path = tmp_path / "const.json"
    assert U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
                  "--raw-dir", str(raw_dir), "--out", str(const_path)]) == 0

    out = tmp_path / "verdict.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        U.main(["verdict", "--lomo-dir", str(lomo_dir), "--table", str(table_path), "--raw-dir", str(raw_dir),
               "--constants", str(const_path), "--out", str(out)])

    out.unlink()
    assert U.main(["verdict", "--lomo-dir", str(lomo_dir), "--table", str(table_path), "--raw-dir", str(raw_dir),
                  "--constants", str(const_path), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert "verdict" in rep["h1"]
    assert "verdict" in rep["h2"]


def test_cli_verdict_refuses_on_failing_report_and_writes_no_output(tmp_path):
    """Fails when the verdict command stops calling require_stop_rules before writing its output."""
    design_rows = [{"id": 300 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "pred": 900.0,
                   "est": 700.0, "cls": "A1"} for i in range(5)]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: design_rows})
    const_path = tmp_path / "const.json"
    assert U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
                  "--raw-dir", str(raw_dir), "--out", str(const_path)]) == 0

    _write_report(table_path, _report(mad=9999.0))   # the table's report.json now fails rule (c)

    out = tmp_path / "verdict.json"
    with pytest.raises(ValueError, match="matched_exit_mad_s"):
        U.main(["verdict", "--lomo-dir", str(lomo_dir), "--table", str(table_path), "--raw-dir", str(raw_dir),
               "--constants", str(const_path), "--out", str(out)])
    assert not out.exists()


def test_cli_verdict_json_has_passing_stop_rules(tmp_path):
    """Fails when a passing run's output JSON loses the stop_rules check dict."""
    design_rows = [{"id": 500 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "pred": 900.0,
                   "est": 700.0, "cls": "A1"} for i in range(5)]
    scored_rows = [{"id": 600 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "pred": 900.0,
                   "est": 700.0, "cls": "A1"} for i in range(5)]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: design_rows, 1: scored_rows})
    const_path = tmp_path / "const.json"
    assert U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
                  "--raw-dir", str(raw_dir), "--out", str(const_path)]) == 0
    out = tmp_path / "verdict.json"
    assert U.main(["verdict", "--lomo-dir", str(lomo_dir), "--table", str(table_path), "--raw-dir", str(raw_dir),
                  "--constants", str(const_path), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["stop_rules"]["ok"] is True


def test_cli_verdict_refuses_constants_fitted_on_a_different_table(tmp_path):
    """Fails when the verdict command stops calling check_constants before loading the frame, so a constants file
    fitted on a stale table would silently price against a different one."""
    design_rows = [{"id": 9500 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0,
                   "pred": 900.0, "est": 700.0, "cls": "A1"} for i in range(5)]
    jan_row = [{"id": 9600, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "pred": 900.0,
               "est": 700.0, "cls": "A1"}]   # a scored-month population row so h1() has something to compare,
                                             # independent of whichever cycle's fix lands first
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {3: design_rows, 1: jan_row})
    const_path = tmp_path / "const.json"
    assert U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
                  "--raw-dir", str(raw_dir), "--out", str(const_path)]) == 0

    t = pd.read_parquet(table_path)                       # the table changes after the constants file was written
    t.loc[0, "usable"] = not bool(t.loc[0, "usable"])
    table_path.unlink()
    t.to_parquet(table_path, index=False)

    out = tmp_path / "verdict.json"
    with pytest.raises(ValueError, match="table_sha256"):
        U.main(["verdict", "--lomo-dir", str(lomo_dir), "--table", str(table_path), "--raw-dir", str(raw_dir),
               "--constants", str(const_path), "--out", str(out)])
    assert not out.exists()


def test_cli_verdict_output_has_h1_verdict_and_h1_excluding_eham_readout(tmp_path):
    """Fails when the verdict command stops writing h1_excluding_EHAM, or leaves a "verdict" key in it instead of
    renaming it to "readout" (h1_excluding_EHAM decides nothing)."""
    design_rows = [{"id": 700 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "pred": 900.0,
                   "est": 700.0, "cls": "A1"} for i in range(5)]
    scored_rows = [{"id": 800 + i, "lane": "unmatched", "decided_by": "unmatched_body", "y": 1000.0, "pred": 900.0,
                   "est": 700.0, "cls": "A1"} for i in range(5)]
    lomo_dir, table_path, raw_dir = _minimal_12month_world(tmp_path, {2: design_rows, 1: scored_rows})
    const_path = tmp_path / "const.json"
    assert U.main(["constants", "--lomo-dir", str(lomo_dir), "--table", str(table_path),
                  "--raw-dir", str(raw_dir), "--out", str(const_path)]) == 0
    out = tmp_path / "verdict.json"
    assert U.main(["verdict", "--lomo-dir", str(lomo_dir), "--table", str(table_path), "--raw-dir", str(raw_dir),
                  "--constants", str(const_path), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert "verdict" in rep["h1"]
    assert "h1_excluding_EHAM" in rep
    assert "readout" in rep["h1_excluding_EHAM"]
    assert "verdict" not in rep["h1_excluding_EHAM"]
