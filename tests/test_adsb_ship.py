"""Tests for scripts/adsb_ship.py — the ADN ship step on v10 (prereg ADN, "Only if WORKING").

Written before the module (red first). Synthetic worlds; each test names the mutation it must kill.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("adsb_ship", ROOT / "scripts" / "adsb_ship.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["adsb_ship"] = mod
    spec.loader.exec_module(mod)
    return mod


sh = _load()


# ---------------------------------------------------------------- the coverage gate

def test_gate_keeps_f_where_2026_coverage_falls_below_seventy_percent_of_2025():
    """Registered ship gate: an airport ships only if its 2026 joined share >= 0.7 x its 2025 share.
    Kills: gating on 2026 alone; a >= flipped to >; an airport absent in 2026 passing."""
    c25 = {"EHAM": 0.70, "EDDM": 0.50, "LSZH": 0.60, "LTFM": 0.0, "LEBL": 0.5}
    c26 = {"EHAM": 0.60, "EDDM": 0.30, "LSZH": 0.60, "LEBL": 0.35}
    g = sh.coverage_gate(c25, c26)
    assert g == {"EHAM": True, "EDDM": False, "LSZH": True, "LTFM": False, "LEBL": True}
    assert 0.7 * 0.5 == 0.35                     # LEBL sits EXACTLY on the bar (BC-4: a boundary test must be on it)


# ---------------------------------------------------------------- assembling v11 from v10

def _base(tmp_path):
    ids = np.array([101, 102, 103, 104, 105, 106], dtype=float)
    v10 = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.array([900, 800, 700, 600, 5000, 1], dtype=np.int32)})
    p = tmp_path / "merry-quicksand_v10.parquet"
    v10.to_parquet(p, index=False)
    return v10, p


def test_assemble_changes_only_gated_joined_matched_rows_and_keeps_every_other_row_byte_identical(tmp_path):
    """Kills: touching unmatched rows (105: another lane); touching unjoined rows (103); touching an airport the gate
    closed (104, EDDM); a new value that is not max(proxy - (F + r), 1) rounded; changing the row order."""
    v10, _ = _base(tmp_path)
    rows = pd.DataFrame({"MVT_ID_mvt": [101.0, 102, 103, 104, 106], "ap": ["EHAM", "EHAM", "EHAM", "EDDM", "EHAM"],
                         "proxy": [1500.0, 1400, 1300, 1200, 50], "F": [600.0, 600, 600, 600, 60],
                         "resid": [-100.0, 49.4, 40, -30, 0], "coverage": [3, 2, 1, 3, 3]})
    out, detail = sh.assemble(v10, rows, gate={"EHAM": True, "EDDM": False})
    assert out.MVT_ID_mvt.tolist() == v10.MVT_ID_mvt.tolist()
    got = dict(zip(out.MVT_ID_mvt, out.TAXITIME_SEC_mvt))
    assert got[101] == 1000 and got[102] == 751                               # rint(1400 - 649.4) = rint(750.6)
    assert got[103] == 700 and got[104] == 600 and got[105] == 5000
    assert got[106] == 1                                                      # max(50 - 60, 1) = 1
    assert out.TAXITIME_SEC_mvt.dtype == np.int32
    assert sorted(detail.MVT_ID_mvt.tolist()) == [101.0, 102.0]              # only rows whose value changed
    assert set(detail.columns) >= {"MVT_ID_mvt", "ap", "before", "after", "F", "resid", "coverage"}


def test_f_is_recovered_from_v10_as_proxy_minus_taxi():
    """The 2026 F input is the shipped matched lane's delta: proxy - v10 taxi. Kills: feeding the taxi itself as F."""
    np.testing.assert_allclose(sh.delta_from_taxi(np.array([1500.0, 800.0]), np.array([900, 300])), [600.0, 500.0])


# ---------------------------------------------------------------- guards

def test_build_refuses_a_base_other_than_v10_and_refuses_to_overwrite(tmp_path):
    """Kills: building on a stale base (the lgbm_submit --base landmine class); clobbering a written version."""
    v10, p = _base(tmp_path)
    with pytest.raises(SystemExit, match="base must be merry-quicksand_v10"):
        sh.check_paths(tmp_path / "merry-quicksand_v7.parquet", tmp_path / "merry-quicksand_v11.parquet")
    (tmp_path / "merry-quicksand_v11.parquet").write_bytes(b"x")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        sh.check_paths(p, tmp_path / "merry-quicksand_v11.parquet")


def test_contract_refuses_a_submission_with_wrong_rows_or_values():
    """Kills: shipping a file with a lost row, a duplicate id, a NaN, or a value below 1."""
    v10 = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "TAXITIME_SEC_mvt": np.array([5, 6], dtype=np.int32)})
    ok = v10.copy()
    sh.contract(ok, v10)
    with pytest.raises(ValueError, match="rows"):
        sh.contract(ok.iloc[:1], v10)
    bad = ok.copy(); bad.loc[1, "TAXITIME_SEC_mvt"] = 0
    with pytest.raises(ValueError, match=">= 1"):
        sh.contract(bad, v10)


def test_gate_table_applies_the_07_rule_and_adn7_for_newly_covered_airports():
    """ADN.7: newly covered = 2025 share < 0.20 OR 2026 share >= 2 x 2025. Those ship only if the transfer test passed;
    the rest follow the 0.7x rule. Each entry names the rule that decided it. Kills: newly covered airports shipping
    when transfer failed; the 2x criterion ignored; LTFM (2025 share 0) shipping without a transfer pass."""
    c25 = {"EHAM": 0.72, "EDDM": 0.53, "LIRF": 0.45, "LTFM": 0.0, "LEMD": 0.13}
    c26 = {"EHAM": 0.60, "EDDM": 0.30, "LIRF": 0.95, "LTFM": 0.40, "LEMD": 0.70}
    t = sh.gate_table(c25, c26, transfer_pass=True)
    assert {a: v["allowed"] for a, v in t.items()} == {"EHAM": True, "EDDM": False, "LIRF": True, "LTFM": True,
                                                      "LEMD": True}
    assert t["EHAM"]["rule"].startswith("0.7x") and t["LIRF"]["rule"].startswith("ADN.7")
    assert t["LTFM"]["rule"].startswith("ADN.7") and t["EDDM"]["rule"].startswith("0.7x")
    f = sh.gate_table(c25, c26, transfer_pass=False)
    assert {a: v["allowed"] for a, v in f.items()} == {"EHAM": True, "EDDM": False, "LIRF": False, "LTFM": False,
                                                      "LEMD": False}


def test_joined_share_per_airport_counts_coverage_two_or_more():
    """Kills: counting coverage 1 as joined; a share over all airports instead of per airport."""
    t = pd.DataFrame({"ap": ["EHAM"] * 4 + ["LTFM"] * 2, "coverage": [3, 2, 1, 0, 1, 1]})
    assert sh.joined_share(t) == {"EHAM": 0.5, "LTFM": 0.0}


def test_write_gate_puts_the_table_and_its_evidence_into_the_manifest(tmp_path):
    """Kills: a gate written without the transfer verdict it depends on; overwriting an already-set gate silently."""
    import json
    md = tmp_path / "models"; md.mkdir()
    (md / "MANIFEST.json").write_text(json.dumps({"feats": ["x"], "gate": {}}))
    t25 = pd.DataFrame({"ap": ["EHAM"] * 10, "coverage": [3] * 7 + [1] * 3})
    t26 = pd.DataFrame({"ap": ["EHAM"] * 10 + ["LIRF"] * 10, "coverage": [3] * 6 + [1] * 4 + [2] * 9 + [1]})
    g = sh.write_gate(md, t25, t26, transfer={"rule": {"PASS": True, "n_pass": 3}})
    man = json.loads((md / "MANIFEST.json").read_text())
    assert man["gate"]["EHAM"]["allowed"] is True and man["gate"]["LIRF"]["allowed"] is True
    assert man["gate_evidence"]["transfer_PASS"] is True and man["gate_evidence"]["gate_ratio"] == 0.7
    assert g == man["gate"]
    with pytest.raises(SystemExit, match="gate already set"):
        sh.write_gate(md, t25, t26, transfer={"rule": {"PASS": True, "n_pass": 3}})


def test_ks_contract_flags_a_shifted_feature():
    """Pre-flight Q5 data contract: KS per airport per feature between 2025 and 2026 joined rows; > 0.10 is flagged.
    Kills: a KS computed on unjoined rows (all-NaN), or never flagging."""
    rng = np.random.default_rng(0)
    t25 = pd.DataFrame({"ap": "EHAM", "coverage": 3, "f1": rng.normal(0, 1, 2000), "f2": rng.normal(0, 1, 2000)})
    t26 = pd.DataFrame({"ap": "EHAM", "coverage": 3, "f1": rng.normal(0, 1, 2000), "f2": rng.normal(1.0, 1, 2000)})
    rep = sh.ks_contract(t25, t26, ["f1", "f2"])
    assert rep["EHAM"]["f1"]["flag"] is False and rep["EHAM"]["f2"]["flag"] is True


def test_tables_with_airport_attach_adep_and_drop_the_spent_day(tmp_path):
    """Kills: coverage shares computed on the spent day 2025-01-09; a 2026 row attached to the wrong airport."""
    t25 = pd.DataFrame({"row": [0, 1, 2], "MVT_ID_mvt": [1.0, 2.0, 3.0], "coverage": [3, 3, 1]})
    rec = pd.DataFrame({"row": [0, 1, 2], "ap": ["EHAM", "EHAM", "LSZH"]})
    raw = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0],
                        "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-01-09 10:00", "2025-01-10 10:00", "2025-07-01 10:00"],
                                                           utc=True)})
    t26 = pd.DataFrame({"MVT_ID_mvt": [10.0, 11.0], "coverage": [2, 3]})
    rk = pd.DataFrame({"MVT_ID_mvt": [11.0, 10.0], "ADEP_mvt": ["LIRF", "EDDM"]})
    for name, df in [("t25", t25), ("rec", rec), ("raw", raw), ("t26", t26), ("rk", rk)]:
        df.to_parquet(tmp_path / f"{name}.parquet")
    a, b = sh.tables_with_airport(tmp_path / "t25.parquet", tmp_path / "rec.parquet", [tmp_path / "raw.parquet"],
                                  tmp_path / "t26.parquet", tmp_path / "rk.parquet")
    assert a.MVT_ID_mvt.tolist() == [2.0, 3.0] and a.ap.tolist() == ["EHAM", "LSZH"]
    assert dict(zip(b.MVT_ID_mvt, b.ap)) == {10.0: "EDDM", 11.0: "LIRF"}


# ---------------------------------------------------------------- SHIP-ADN (plans/PREREG_adsb_ship_v11_2026_09_11.md)

def _mix(rng, n2, n3, mu2=0.0, mu3=3.0, ap="EHAM"):
    """Joined rows of two coverage codes whose key feature sits at different levels (code 3 seen from the stand)."""
    return pd.DataFrame({"ap": ap, "coverage": [2] * n2 + [3] * n3,
                         "first_ground_gs": np.r_[rng.normal(mu2, 1, n2), rng.normal(mu3, 1, n3)],
                         "other": rng.normal(0, 1, n2 + n3)})


def test_ks_by_code_separates_a_mixture_shift_from_a_within_code_shift():
    """SHIP-ADN: the code-3 share grows in 2026 (receiver network), which moves the POOLED distribution although each
    code's distribution is unchanged. Within-code KS must stay small there and flag a genuine within-code shift.
    Kills: KS computed on the pooled rows (the mixture flags); strata not filtered by code; code 1 rows included."""
    rng = np.random.default_rng(1)
    t25 = _mix(rng, 1000, 1000)
    t26 = _mix(rng, 300, 1700)
    t26 = pd.concat([t26, pd.DataFrame({"ap": "EHAM", "coverage": [1] * 500, "first_ground_gs": 50.0, "other": 0.0})])
    pooled = sh.ks_contract(t25, t26, ["first_ground_gs"])
    assert pooled["EHAM"]["first_ground_gs"]["flag"] is True            # the benign mixture shift shows pooled
    by = sh.ks_by_code(t25, t26, ["first_ground_gs"])
    assert set(by["EHAM"]) == {2, 3}
    assert by["EHAM"][2]["first_ground_gs"]["flag"] is False and by["EHAM"][3]["first_ground_gs"]["flag"] is False
    moved = _mix(rng, 300, 1700, mu3=3.5)
    by2 = sh.ks_by_code(t25, moved, ["first_ground_gs"])
    assert by2["EHAM"][3]["first_ground_gs"]["flag"] is True and by2["EHAM"][2]["first_ground_gs"]["flag"] is False


def test_ks_by_code_needs_thirty_rows_on_each_side_of_a_stratum():
    """A stratum with < 30 rows on either side gives ks None (no hold). Kills: a 29-row stratum flagging; the bound
    applied to one side only."""
    rng = np.random.default_rng(2)
    t25 = _mix(rng, 29, 500)
    t26 = _mix(rng, 400, 30, mu2=5.0, mu3=8.0)
    by = sh.ks_by_code(t25, t26, ["first_ground_gs"])
    assert by["EHAM"][2]["first_ground_gs"] == {"ks": None, "flag": False}
    assert by["EHAM"][3]["first_ground_gs"]["flag"] is True             # exactly 30 on the 2026 side counts


def test_ks_holds_list_only_flagged_key_features():
    """Kills: a flag on a non-key feature holding an airport; an airport with no flags listed."""
    by = {"EHAM": {2: {"first_ground_gs": {"ks": 0.2, "flag": True}, "other": {"ks": 0.5, "flag": True}},
                   3: {"first_ground_gs": {"ks": 0.01, "flag": False}}},
          "EDDM": {2: {"other": {"ks": 0.9, "flag": True}}, 3: {}}}
    assert sh.ks_holds(by, sh.KEY_FEATS) == {"EHAM": ["code2:first_ground_gs"]}
    assert sh.KEY_FEATS == ("first_ground_rel_aobt3", "first_move_rel_mvt", "first_ground_gs")


def test_gate_table_holds_a_supported_airport_on_a_key_feature_flag_but_not_a_newly_covered_one():
    """SHIP-ADN HOLD: an airport within 2025 support keeps F on a within-code key-feature flag; ADN.7 governs newly
    covered airports (their KS is reported only). Kills: holds ignored; holds applied to newly covered airports; the
    hold not named in the rule."""
    c25 = {"EHAM": 0.72, "EDDM": 0.53, "LEMD": 0.13}
    c26 = {"EHAM": 0.75, "EDDM": 0.55, "LEMD": 0.60}
    holds = {"EHAM": ["code3:first_ground_rel_aobt3"], "LEMD": ["code2:first_ground_gs"]}
    t = sh.gate_table(c25, c26, transfer_pass=True, holds=holds)
    assert t["EHAM"]["allowed"] is False and "KS hold" in t["EHAM"]["rule"] and "code3:first_ground_rel_aobt3" in t["EHAM"]["rule"]
    assert t["EDDM"]["allowed"] is True and t["LEMD"]["allowed"] is True and t["LEMD"]["rule"].startswith("ADN.7")


def test_projection_follows_the_registered_formula():
    """P = sum over allowed airports of k * g * n26 * m26 / 344,841; newly covered use the pooled g and k = 0.595.
    Kills: blocked airports counted; newly covered airports given their own g or k = 1; a different denominator;
    a supported airport without a 2025 g passing silently."""
    gate = {"EHAM": {"allowed": True, "rule": "0.7x gate"}, "EDDM": {"allowed": False, "rule": "0.7x gate"},
            "LEMD": {"allowed": True, "rule": "ADN.7 newly covered (transfer PASS)"}}
    n26 = {"EHAM": 1000, "EDDM": 1000, "LEMD": 2000}
    m26 = {"EHAM": 0.5, "EDDM": 0.5, "LEMD": 0.25}
    p = sh.projection(gate, n26, m26)
    want_eham = sh.G_PER_JOINED["EHAM"] * 1000 * 0.5 / 344_841
    want_lemd = 0.595 * sh.G_POOLED * 2000 * 0.25 / 344_841
    assert p["by_airport"]["EHAM"] == pytest.approx(want_eham, rel=1e-12)   # pure float arithmetic, same order
    assert p["by_airport"]["LEMD"] == pytest.approx(want_lemd, rel=1e-12)
    assert "EDDM" not in p["by_airport"] and p["P"] == pytest.approx(want_eham + want_lemd, rel=1e-12)
    assert sh.G_PER_JOINED["EHAM"] == 13_621 and sh.G_POOLED == 13_906
    with pytest.raises(ValueError, match="no 2025 per-joined-row gain"):
        sh.projection({"LTFM": {"allowed": True, "rule": "0.7x gate"}}, {"LTFM": 10}, {"LTFM": 0.5})


def test_write_gate_applies_and_records_the_ks_holds(tmp_path):
    """SHIP-ADN: the holds reach the manifest gate and its evidence. Kills: write_gate dropping `holds` on the way to
    gate_table; the evidence not recording which key feature held an airport."""
    import json
    md = tmp_path / "models"; md.mkdir()
    (md / "MANIFEST.json").write_text(json.dumps({"feats": ["x"], "gate": {}}))
    t25 = pd.DataFrame({"ap": ["EHAM"] * 10, "coverage": [3] * 7 + [1] * 3})
    t26 = pd.DataFrame({"ap": ["EHAM"] * 10, "coverage": [3] * 7 + [1] * 3})
    g = sh.write_gate(md, t25, t26, transfer={"rule": {"PASS": True, "n_pass": 3}},
                      holds={"EHAM": ["code3:first_move_rel_mvt"]})
    man = json.loads((md / "MANIFEST.json").read_text())
    assert g["EHAM"]["allowed"] is False and man["gate"]["EHAM"]["allowed"] is False
    assert man["gate_evidence"]["ks_holds"] == {"EHAM": ["code3:first_move_rel_mvt"]}
    assert man["gate_evidence"]["key_feats"] == list(sh.KEY_FEATS)


# ---------------------------------------------------------------- SHIP-ADN.3: hold iff KS26 > KS_season

def _cell(ks):
    return {"ks": ks, "flag": bool(ks is not None and ks > 0.10)}


def test_season_holds_fire_only_where_the_year_shift_exceeds_the_seasonal_shift():
    """SHIP-ADN.3: HOLD iff KS26 > KS_season, per airport x key feature x stratum. Kills: >= instead of > (equal must not
    hold); the old 0.10 flag still deciding (EHAM: KS26 0.30 < season 0.40 must not hold); a non-key feature holding;
    an undefined cell (None on either side) holding."""
    ks26 = {"EHAM": {2: {"first_move_rel_mvt": _cell(0.30), "other": _cell(0.9)}, 3: {"first_ground_gs": _cell(0.05)}},
            "EDDF": {2: {"first_ground_rel_aobt3": _cell(0.20)}, 3: {"first_ground_gs": _cell(None)}},
            "LSZH": {2: {"first_ground_gs": _cell(0.07)}, 3: {"first_ground_gs": _cell(0.12)}}}
    season = {"EHAM": {2: {"first_move_rel_mvt": _cell(0.40), "other": _cell(0.1)}, 3: {"first_ground_gs": _cell(0.05)}},
              "EDDF": {2: {"first_ground_rel_aobt3": _cell(0.10)}, 3: {"first_ground_gs": _cell(0.02)}},
              "LSZH": {2: {"first_ground_gs": _cell(None)}, 3: {"first_ground_gs": _cell(0.11)}}}
    h = sh.season_holds(ks26, season)
    assert h == {"EDDF": ["code2:first_ground_rel_aobt3 (0.200 > 0.100)"],
                 "LSZH": ["code3:first_ground_gs (0.120 > 0.110)"]}


def test_season_holds_reject_a_missing_baseline_airport():
    """An airport in the 2026 comparison without a seasonal baseline entry is a wiring error, not 'no hold'.
    Kills: a silent .get() default that lets every cell pass untested."""
    with pytest.raises(KeyError, match="no seasonal baseline for EHAM"):
        sh.season_holds({"EHAM": {2: {"first_ground_gs": _cell(0.3)}}}, {})


def test_tables_with_airport_attach_the_2025_month(tmp_path):
    """SHIP-ADN.3 needs the 2025 month for the Jan-vs-Jul baseline. Kills: month taken from the 2026 file; month missing."""
    t25 = pd.DataFrame({"row": [0, 1, 2], "MVT_ID_mvt": [1.0, 2.0, 3.0], "coverage": [3, 3, 1]})
    rec = pd.DataFrame({"row": [0, 1, 2], "ap": ["EHAM", "EHAM", "LSZH"]})
    raw = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0],
                        "MVT_TIME_UTC_mvt": pd.to_datetime(["2025-01-09 10:00", "2025-01-10 10:00", "2025-07-01 10:00"],
                                                           utc=True)})
    t26 = pd.DataFrame({"MVT_ID_mvt": [10.0], "coverage": [2]})
    rk = pd.DataFrame({"MVT_ID_mvt": [10.0], "ADEP_mvt": ["EDDM"]})
    for name, df in [("t25", t25), ("rec", rec), ("raw", raw), ("t26", t26), ("rk", rk)]:
        df.to_parquet(tmp_path / f"{name}.parquet")
    a, _ = sh.tables_with_airport(tmp_path / "t25.parquet", tmp_path / "rec.parquet", [tmp_path / "raw.parquet"],
                                  tmp_path / "t26.parquet", tmp_path / "rk.parquet")
    assert a.month.tolist() == [1, 7]


# ---------------------------------------------------------------- SHIP-ADN.4: the gate is "D, per RWC"

def test_rwc_holds_hold_exactly_the_disputed_airports_rwc_did_not_ship():
    """Kills: an airport outside D held from the RWC file (its numbers are reported, not acted on); a disputed airport
    missing from the RWC results passing silently; a harness FAIL not holding all of D."""
    rwc = {"harness": "PASS", "results": {"EDDM": {"decision": "SHIP"}, "EDDF": {"decision": "KEEP F"},
                                           "LEBL": {"decision": "SHIP"}, "LSZH": {"decision": "INCONCLUSIVE"},
                                           "EHAM": {"decision": "KEEP F"}}}
    assert sh.rwc_holds(rwc) == {"EDDF": ["RWC KEEP F"], "LSZH": ["RWC INCONCLUSIVE"]}
    with pytest.raises(KeyError, match="LEBL"):
        sh.rwc_holds({"harness": "PASS", "results": {"EDDM": {"decision": "SHIP"}, "EDDF": {"decision": "SHIP"},
                                                      "LSZH": {"decision": "SHIP"}}})
    failed = {**rwc, "harness": "FAIL"}
    assert sh.rwc_holds(failed) == {a: ["RWC harness FAIL"] for a in sh.RWC_DISPUTED}
    assert sh.RWC_DISPUTED == ("EDDM", "EDDF", "LEBL", "LSZH")


# ---------------------------------------------------------------- end-work review follow-ups (2026-09-11 17:0x)

def test_ks_uses_finite_values_only_and_needs_thirty_on_the_2026_side_too():
    """Review finding: the isfinite filter and the 2026-side bound were untested. Kills: KS over NaN-padded arrays (NaN
    sorts last and inflates the statistic); the n >= 30 bound checked on the 2025 side only."""
    rng = np.random.default_rng(3)
    a = rng.normal(0, 1, 500)
    b = np.r_[rng.normal(0, 1, 500), np.full(400, np.nan)]
    assert sh._ks(a, b) == pytest.approx(sh._ks(a, b[np.isfinite(b)]), abs=1e-12)   # identical arrays after the filter
    assert sh._ks(a, b) < 0.10
    assert sh._ks(a, np.r_[rng.normal(5, 1, 29), np.full(10, np.nan)]) is None       # 29 finite on the 2026 side
    assert sh._ks(np.r_[rng.normal(0, 1, 29)], rng.normal(5, 1, 500)) is None         # 29 on the 2025 side


def test_write_gate_records_the_rule_that_decided_the_holds(tmp_path):
    """Kills: the evidence not naming which hold rule (KS, season, RWC) produced the gate."""
    import json
    md = tmp_path / "models"; md.mkdir()
    (md / "MANIFEST.json").write_text(json.dumps({"feats": ["x"], "gate": {}}))
    t = pd.DataFrame({"ap": ["EHAM"] * 10, "coverage": [3] * 10})
    sh.write_gate(md, t, t, transfer={"rule": {"PASS": True, "n_pass": 3}}, holds={}, ks_rule="SHIP-ADN.4: D, per RWC")
    assert json.loads((md / "MANIFEST.json").read_text())["gate_evidence"]["ks_rule"] == "SHIP-ADN.4: D, per RWC"


def test_season_holds_reject_a_missing_stratum_in_the_baseline():
    """Fail loud: a baseline without a stratum the 2026 comparison has is a wiring error. Kills: a silent .get default."""
    with pytest.raises(KeyError, match="stratum 3"):
        sh.season_holds({"EHAM": {3: {"first_ground_gs": _cell(0.3)}}}, {"EHAM": {2: {}}})


def test_contract_refuses_non_integer_values():
    """Kills: a float submission (e.g. 812.4) passing the contract because it is finite and >= 1."""
    v10 = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "TAXITIME_SEC_mvt": np.array([5, 6], dtype=np.int32)})
    bad = v10.assign(TAXITIME_SEC_mvt=[5.0, 6.4])
    with pytest.raises(ValueError, match="integer"):
        sh.contract(bad, v10)


# ---------------------------------------------------------------- SHIP-ADN.5: LIRF "with v12" via RWC.3 (coverage override)

def test_a_coverage_override_lifts_only_the_named_airport_and_is_named_in_the_rule():
    """SHIP-ADN.5: LIRF fails the 0.7x coverage rule; RWC.3 SHIP lifts exactly that airport. Kills: the override lifting an
    airport not named (EDDM also fails 0.7x here); the override ignored; the rule string not recording it; an override
    un-holding a KS/RWC-held airport."""
    c25 = {"LIRF": 0.448, "EDDM": 0.527, "LSZH": 0.70}
    c26 = {"LIRF": 0.199, "EDDM": 0.30, "LSZH": 0.40}                   # LSZH fails 0.7x too, is overridden AND held
    t = sh.gate_table(c25, c26, transfer_pass=True, holds={"LSZH": ["RWC KEEP F"]},
                      coverage_overrides={"LIRF": "RWC.3 SHIP", "LSZH": "RWC.3 SHIP"})
    assert t["LIRF"]["allowed"] is True and "RWC.3 SHIP" in t["LIRF"]["rule"] and "0.7x" in t["LIRF"]["rule"]
    assert t["EDDM"]["allowed"] is False
    assert t["LSZH"]["allowed"] is False                                    # an override never lifts a hold
    assert sh.gate_table(c25, c26, transfer_pass=True)["LIRF"]["allowed"] is False


def test_rwc3_overrides_need_a_passed_harness_and_a_ship():
    """Kills: an override granted on a failed harness, on a non-SHIP decision, or for an airport other than LIRF."""
    ok = {"harness": "PASS", "results": {"LIRF": {"decision": "SHIP"}, "EHAM": {"decision": "SHIP"}}}
    assert sh.rwc3_overrides(ok) == {"LIRF": "RWC.3 SHIP"}
    assert sh.rwc3_overrides({**ok, "harness": "FAIL"}) == {}
    assert sh.rwc3_overrides({"harness": "PASS", "results": {"LIRF": {"decision": "KEEP F"}}}) == {}
    with pytest.raises(KeyError, match="LIRF"):
        sh.rwc3_overrides({"harness": "PASS", "results": {}})


def test_write_gate_records_the_coverage_overrides(tmp_path):
    """Kills: write_gate dropping the overrides on the way to gate_table, or not recording them in the evidence."""
    import json
    md = tmp_path / "models"; md.mkdir()
    (md / "MANIFEST.json").write_text(json.dumps({"feats": ["x"], "gate": {}}))
    t25 = pd.DataFrame({"ap": ["LIRF"] * 10, "coverage": [3] * 5 + [1] * 5})
    t26 = pd.DataFrame({"ap": ["LIRF"] * 10, "coverage": [3] * 2 + [1] * 8})
    g = sh.write_gate(md, t25, t26, transfer={"rule": {"PASS": True, "n_pass": 3}}, holds={},
                      coverage_overrides={"LIRF": "RWC.3 SHIP"}, ks_rule="SHIP-ADN.5")
    ev = json.loads((md / "MANIFEST.json").read_text())["gate_evidence"]
    assert g["LIRF"]["allowed"] is True and ev["coverage_overrides"] == {"LIRF": "RWC.3 SHIP"}
