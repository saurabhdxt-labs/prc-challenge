"""The serve-time EOBT clip (prc/pipeline/fm1/eobt_clip.py; plans/PREREG_clp_lfpg_eobt_clip_2026_10_02.md).

Tiers: the knobs, the late-knob hash rule against the REAL stored configs of v50 / v51 and the plan; the rule on a
constructed world (each definition — eligibility, the strict EOBT > LOBT test, the share and row floors, the cell key,
the population — flips at least one row when it is changed); equivalence with EPX's priced clip; the e2e world through
api.predict (slow); the detector on the REAL scored file and every 2025 month file (slow).

Mutation rehearsal per test is recorded in its docstring and in tests/fm1/MUTATION_FM1.md (M-CLP rows).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import eobt_clip as C
from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import stages as S

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEGACY = ROOT / "tests" / "fixtures" / "fm1_legacy_configs"
#: K.parse_config({}).hash before the first late knob existed (read 2026-10-02 08:5x EDT, 80 knobs)
DEFAULTS_HASH_BEFORE_LATE_KNOBS = "ad0af3a77a716bce8361486ad0a26d7acd6abee3f027f106ef106fa7535afe32"
LATE = {"modules.eobt_clip.enabled": False, "modules.eobt_clip.airports": ("LFPG",),
        "modules.eobt_clip.min_share": 0.15, "modules.eobt_clip.min_rows": 1000,
        "adsb.train_dropout.flight": 0.0, "adsb.train_dropout.day": 0.0}
NAT = None


def _ts(s):
    return pd.Timestamp(s, tz="UTC") if s is not None else pd.NaT


#: (id, phase, airport, MVT UTC, AOBT_3, EOBT_1, LOBT). Cell LFPG 2026-07: ids 1-10 eligible, 1-4 late (EOBT > LOBT),
#: 5 equal (not late), share 4/10; ids 11-14 sit in the cell but are not eligible. Cell LFPG 2026-01: ids 21-30, one late
#: (share 0.1). EGLL 2026-07 (ids 41-45, all late) is not a configured airport. Id 50 has no MVT_TIME (no cell).
def _world():
    w = []
    for i in range(1, 11):
        lobt = f"2026-07-0{1 + i % 5} 10:00"
        eobt = {1: "10:10", 2: "10:20", 3: "11:00", 4: "10:01", 5: "10:00"}.get(i, "09:50")
        w.append((float(i), "DEP", "LFPG", f"2026-07-0{1 + i % 5} 10:30", f"2026-07-0{1 + i % 5} 10:15",
                  f"{lobt[:10]} {eobt}", lobt))
    w += [(11.0, "DEP", "LFPG", "2026-07-03 11:30", NAT, "2026-07-03 11:20", "2026-07-03 11:00"),   # no AOBT_3
          (12.0, "DEP", "LFPG", "2026-07-03 11:30", "2026-07-03 11:15", NAT, "2026-07-03 11:00"),  # no EOBT_1
          (13.0, "DEP", "LFPG", "2026-07-03 11:30", "2026-07-03 11:15", "2026-07-03 11:20", NAT),  # no LOBT
          (14.0, "ARR", "LFPG", "2026-07-03 11:30", "2026-07-03 11:15", "2026-07-03 11:20", "2026-07-03 11:00")]
    for i in range(21, 31):
        eobt = "2026-01-10 08:30" if i == 21 else "2026-01-10 07:55"
        w.append((float(i), "DEP", "LFPG", "2026-01-10 08:40", "2026-01-10 08:35", eobt, "2026-01-10 08:00"))
    for i in range(41, 46):
        w.append((float(i), "DEP", "EGLL", "2026-07-02 09:30", "2026-07-02 09:20", "2026-07-02 09:10", "2026-07-02 09:00"))
    w.append((50.0, "DEP", "LFPG", NAT, "2026-07-03 11:15", "2026-07-03 11:20", "2026-07-03 11:00"))
    return w


def _rows(world=None) -> pd.DataFrame:
    world = _world() if world is None else world
    col = lambda j: pd.Series([_ts(r[j]) for r in world], dtype="datetime64[us, UTC]")  # noqa: E731
    return pd.DataFrame({"MVT_ID_mvt": [r[0] for r in world],
                         "PHASE_mvt": pd.Series([r[1] for r in world], dtype="str"),
                         "ADEP_mvt": pd.Series([r[2] for r in world], dtype="str"),
                         "MVT_TIME_UTC_mvt": col(3), "AOBT_3_flt": col(4), "EOBT_1_flt": col(5), "LOBT_flt": col(6),
                         "OTHER": np.arange(len(world), dtype="float64")})


def _serve(rows, scored=None, airports=("LFPG",), min_share=0.15, min_rows=10):
    return C.serve(rows, airports, min_share, min_rows, rows if scored is None else scored)


def _cfg(**over):
    return K.parse_config({}, overrides=[f"{k}={json.dumps(v)}" for k, v in over.items()])


# ---------------------------------------------------------------------------------------------- knobs, hash, plan

#: the generation each late knob was added in (BC-41): 1 = the EOBT clip (2026-10-02 morning, v54 written with it on),
#: 2 = the ADS-B dropout (2026-10-02 afternoon, ACR's M1 written with it on). A new late knob takes a new generation.
LATE_GENERATION = {"modules.eobt_clip.enabled": 1, "modules.eobt_clip.airports": 1, "modules.eobt_clip.min_share": 1,
                   "modules.eobt_clip.min_rows": 1, "adsb.train_dropout.flight": 2, "adsb.train_dropout.day": 2}


def test_late_knobs_and_their_defaults_are_pinned():
    """The late knobs' defaults decide what an artefact written before they existed means when it is re-loaded; changing
    one silently re-defines v50 ... v53. Their generations decide which stored configs hash them (BC-41). Mutation: default
    min_share 0.2 -> red; mark run.threads late -> red; the dropout knobs in generation 1 -> red."""
    late = {k.path: K.normalise_default(k) for k in K.SCHEMA if k.late}
    assert late == LATE
    assert {p: K.KNOBS[p].late for p in late} == LATE_GENERATION
    assert all(K.KNOBS[p].hashed for p in late)


def test_every_config_with_the_late_knobs_at_default_hashes_as_before_they_existed():
    """The REAL stored configs of v50 (the full fit) and v51 (its MTX variant) re-validate to their recorded hashes, and
    the all-defaults hash is the one read before the late knobs were added. Mutation: drop the late-knob omission in
    parse_config -> every hash moves -> red."""
    assert K.parse_config({}).hash == DEFAULTS_HASH_BEFORE_LATE_KNOBS
    for name in ("v50_2d29ad344b45.json", "v51_fa14ea853d20.json"):
        stored = json.loads((LEGACY / name).read_text())
        cfg = K.parse_config(stored["values"])
        assert cfg.hash == stored["hash"], name
        assert {p: cfg[p] for p in LATE} == LATE


def test_a_late_knob_off_its_default_hashes_its_generation_and_every_earlier_one():
    """BC-41 rule: with h = the highest generation holding a knob off its default, every late knob of generations 1 ... h
    is hashed (all of a generation, defaults included) and later generations stay out — exactly what the hash was when
    a config using generation h was written. (Before BC-41: all or none over every generation, which moved v54.)
    Mutation: omit the late knobs always -> on == off -> red; omit each late knob at its OWN default -> the 'on' hash
    lacks airports / min_share / min_rows -> red; hash every generation once any is on (the old rule) -> red."""
    off = _cfg()
    on = _cfg(**{"modules.eobt_clip.enabled": True})
    assert on.hash != off.hash
    assert _cfg(**{"modules.eobt_clip.enabled": True, "modules.eobt_clip.min_share": 0.2}).hash != on.hash
    assert _cfg(**{"modules.eobt_clip.min_rows": 5}).hash != off.hash          # off but not default: in the hash

    def blob(cfg, upto):
        return {k: v for k, v in cfg.values.items() if K.KNOBS[k].hashed and K.KNOBS[k].late <= upto}
    assert on.hash == K.canonical_hash(blob(on, 1))                             # generation 1 in full, 2 out
    drop = _cfg(**{"adsb.train_dropout.flight": 0.3})
    assert drop.hash == K.canonical_hash(blob(drop, 2))                         # generation 2 on: 1 and 2 in full


#: every stored config whose late knobs are not all at default, with the late-knob generations that existed when it was
#: written (BC-41): v54 (the clip on, written 2026-10-02 10:04 EDT before the dropout knobs existed) and ACR's M1 (dropout
#: on, written 17:03 EDT when both generations existed).
LEGACY_WITH_LATE = ("v54_8e37b9a6bf09.json", "acr_m1_4cd05b46c4d1.json")


@pytest.mark.parametrize("name", ("v50_2d29ad344b45.json", "v51_fa14ea853d20.json") + LEGACY_WITH_LATE)
def test_regression_bug_41_every_stored_config_re_validates_to_its_recorded_hash(name):
    """BC-41: the REAL stored config of every artefact on disk that holds a late knob off its default re-validates to the
    hash it was written with (artifact.load refuses it otherwise). Seen red 2026-10-03 08:25-08:42 EDT on v54: the all-or-none
    rule hashed the dropout knobs, added after v54 was written, into v54's config."""
    stored = json.loads((LEGACY / name).read_text())
    assert K.parse_config(stored["values"]).hash == stored["hash"]


def test_every_stored_config_using_a_late_knob_has_a_legacy_fixture():
    """BC-41 tripwire (development tree): every config.json written by an artefact on disk whose late knobs are not all at
    default is one of the legacy fixtures above, so the re-validation test covers it from a clean clone too. A clean clone
    holds no artefacts and passes trivially. Mutation: drop v54 from LEGACY_WITH_LATE -> red."""
    have = {json.loads((LEGACY / n).read_text())["hash"] for n in LEGACY_WITH_LATE}
    missing = []
    for root in ("artifacts", "data/fm1"):
        for f in (ROOT / root).rglob("config.json") if (ROOT / root).is_dir() else ():
            cj = json.loads(f.read_text())
            if not isinstance(cj, dict) or not {"values", "hash"} <= set(cj):
                continue
            cfg = K.parse_config(cj["values"]) if cj["hash"] not in have else None
            if cfg is not None and any(cfg[k.path] != K.normalise_default(k) for k in K.LATE_KNOBS):
                missing.append(str(f.relative_to(ROOT)))
    assert not missing, f"add these stored configs to tests/fixtures/fm1_legacy_configs: {missing}"


def test_regression_bug_41_a_later_late_knob_generation_never_moves_an_existing_hash(monkeypatch):
    """BC-41, the class: adding a late knob of a NEW generation (at its default, as it is for every config written before
    it existed) leaves the hash of every stored config unchanged — the all-defaults one and those with earlier late knobs
    on. Seen red 2026-10-03 08:25-08:42 EDT: the all-or-none rule moved v54 and ACR M1 once a third generation existed."""
    names = ("v50_2d29ad344b45.json", "v51_fa14ea853d20.json") + LEGACY_WITH_LATE
    stored = {n: json.loads((LEGACY / n).read_text()) for n in names}
    before = {n: K.parse_config(s["values"]).hash for n, s in stored.items()}
    gen = int(max(k.late for k in K.LATE_KNOBS)) + 1
    future = dataclasses.replace(K.KNOBS["modules.eobt_clip.min_rows"], path="modules.future_module.min_rows", late=gen)
    monkeypatch.setattr(K, "SCHEMA", K.SCHEMA + (future,))
    monkeypatch.setattr(K, "KNOBS", {**K.KNOBS, future.path: future})
    monkeypatch.setattr(K, "LATE_KNOBS", K.LATE_KNOBS + (future,))
    after = {n: K.parse_config(s["values"]).hash for n, s in stored.items()}
    assert after == before
    assert K.parse_config({}).hash == DEFAULTS_HASH_BEFORE_LATE_KNOBS
    on = K.parse_config({"modules": {"future_module": {"min_rows": 5}}})
    assert on.hash != K.parse_config({}).hash                       # the new knob off its default is hashed


@pytest.mark.parametrize("late", [True, -1, 1.0])
def test_check_schema_refuses_a_late_generation_that_is_not_a_whole_number(late):
    """BC-41: a generation orders the late knobs; True / a float / a negative number would order them by accident.
    Seen red 2026-10-03 08:25-08:42 EDT (no check)."""
    bad = dataclasses.replace(K.KNOBS["modules.eobt_clip.min_rows"], late=late)
    with pytest.raises(E.ConfigError, match="late generation"):
        K.check_schema((bad,))


def test_check_schema_refuses_an_unhashed_late_knob():
    """A late knob that never enters the hash could change predictions invisibly. Mutation: drop the check -> red."""
    bad = dataclasses.replace(K.KNOBS["modules.eobt_clip.min_rows"], hashed=False)
    with pytest.raises(E.ConfigError, match="late knob .* must be hashed"):
        K.check_schema((bad,))


def test_the_clip_knobs_are_reuse_free_so_a_derived_fit_may_switch_them():
    """The clip acts at predict time only: a derived fit (BSV mechanism) from v50 to v51 + clip has no blocker.
    Mutation: drop 'modules.' from _REUSE_FREE_PREFIXES -> red."""
    assert {k for k in LATE if k.startswith("modules.eobt_clip.")} <= K.REUSE_FREE_KNOBS
    v50 = K.parse_config(json.loads((LEGACY / "v50_2d29ad344b45.json").read_text())["values"])
    v54 = v50.replace(**{"run.purpose": "scored_period", "modules.eobt_clip.enabled": True})
    assert K.reuse_blockers(v54, v50) == []


@pytest.mark.parametrize("over, err, msg", [
    ({"modules.eobt_clip.airports": ["Lfpg"]}, E.UnknownAirportError, "modules.eobt_clip.airports"),
    ({"modules.eobt_clip.airports": []}, E.ConfigValueError, "must not be empty"),
    ({"modules.eobt_clip.airports": ["LFPG", "LFPG"]}, E.ConfigValueError, "repeated value"),
    ({"modules.eobt_clip.min_share": 1.5}, E.ConfigValueError, "modules.eobt_clip.min_share"),
    ({"modules.eobt_clip.min_rows": 0}, E.ConfigValueError, "modules.eobt_clip.min_rows"),
    ({"modules.eobt_clip.enabled": True, "modules.eobt_clip.airports": ["KJFK"]}, E.ConfigValueError,
     r"modules.eobt_clip.airports: \['KJFK'\] not in airports.codes"),
])
def test_clip_knobs_refuse_bad_values(over, err, msg):
    """Mutation: drop the ICAO check for this list -> 'Lfpg' passes -> red; drop the airports.codes cross-check -> red."""
    with pytest.raises(err, match=msg):
        _cfg(**over)


def test_an_unknown_airport_is_allowed_while_the_clip_is_off():
    """The off-by-default clip never blocks a config that does not use it (stored configs keep loading). Mutation
    (2026-10-03): cross-check the airports whatever `enabled` says -> ConfigValueError -> red."""
    _cfg(**{"modules.eobt_clip.airports": ["KJFK"]})


def test_plan_places_the_clip_before_the_features_and_only_when_on():
    """Mutation: plan maps eobt_clip to modules.ldi.enabled -> red."""
    on = [s.name for s in S.plan(_cfg(**{"modules.eobt_clip.enabled": True}))]
    assert on[:3] == ["input_contract", "eobt_clip", "features"]
    assert "eobt_clip" not in [s.name for s in S.plan(_cfg())]


# ---------------------------------------------------------------------------------------------- the rule

def test_eligible_rows_are_epx_selection_plus_a_movement_time():
    """EPX's `sel`: DEP at the airport with AOBT_3, EOBT_1 and LOBT present; plus MVT_TIME (the cell key). Mutation: drop
    the AOBT_3 term -> id 11 eligible -> red."""
    rows = _rows()
    got = set(rows.MVT_ID_mvt[C.eligible(rows, ("LFPG",))])
    assert got == set(range(1, 11)) | set(range(21, 31))
    assert set(rows.MVT_ID_mvt[C.eligible(rows, ("LFPG", "EGLL"))]) == got | set(range(41, 46))


def test_cell_table_counts_strictly_later_eobt_per_airport_month():
    """Mutation: EOBT >= LOBT -> id 5 counts as late -> 5/10 -> red; key on the day instead of the month -> red."""
    t = C.cell_table(_rows(), ("LFPG",))
    assert t.to_dict("records") == [
        {"airport": "LFPG", "month": "2026-01", "n": 10, "n_late": 1, "share": 0.1},
        {"airport": "LFPG", "month": "2026-07", "n": 10, "n_late": 4, "share": 0.4}]


@pytest.mark.parametrize("share, fired", [(0.15, True), (0.1499, False), (0.4, True)])
def test_decide_fires_at_or_above_the_share(share, fired):
    """Mutation: share > min_share -> 0.15 not fired -> red."""
    t = pd.DataFrame([{"airport": "LFPG", "month": "2026-07", "n": 2000, "n_late": int(share * 2000), "share": share}])
    assert C.decide(t, 0.15, 1000).fired.tolist() == [fired]


def test_decide_refuses_a_cell_below_the_row_floor_by_name():
    """A cell with 1 ... min_rows - 1 eligible rows cannot be decided (registered: refused, never a silent pass).
    Mutation: drop the floor -> red; floor as n <= min_rows -> the n == min_rows cell is refused -> red."""
    t = pd.DataFrame([{"airport": "LFPG", "month": "2026-07", "n": 999, "n_late": 500, "share": 500 / 999}])
    with pytest.raises(E.DetectorError, match="LFPG 2026-07 has 999 eligible row"):
        C.decide(t, 0.15, 1000)
    t.loc[0, "n"] = 1000
    assert C.decide(t, 0.15, 1000).fired.tolist() == [True]


def test_serve_clips_only_the_late_eligible_rows_of_fired_cells():
    """Ids 1-4 (late, eligible, fired cell) take EOBT := LOBT; id 5 (equal), 6-10 (earlier), 11-14 (not eligible),
    21 (late, unfired cell), 41-45 (other airport), 50 (no cell) keep theirs; no other column moves; the input frame is
    not mutated; dtype kept. Mutation: clip every eligible row of every configured cell -> id 21 moves -> red; write
    the clip into the input frame -> red."""
    rows = _rows()
    before = rows.copy()
    new, ids, rep = _serve(rows)
    pd.testing.assert_frame_equal(rows, before)
    assert sorted(ids.tolist()) == [1.0, 2.0, 3.0, 4.0]
    assert new.EOBT_1_flt.dtype == rows.EOBT_1_flt.dtype
    moved = new.EOBT_1_flt.ne(rows.EOBT_1_flt) & rows.EOBT_1_flt.notna()
    assert sorted(new.MVT_ID_mvt[moved].tolist()) == [1.0, 2.0, 3.0, 4.0]
    hit = rows.MVT_ID_mvt.isin(ids)
    assert (new.EOBT_1_flt[hit] == rows.LOBT_flt[hit]).all()
    pd.testing.assert_frame_equal(new.drop(columns="EOBT_1_flt"), rows.drop(columns="EOBT_1_flt"))
    assert rep["n_clipped"] == 4 and rep["population"] == "scored_file"
    assert [(c["airport"], c["month"], c["fired"]) for c in rep["cells"]] == [("LFPG", "2026-01", False),
                                                                              ("LFPG", "2026-07", True)]
    assert rep["airports"] == ["LFPG"] and rep["min_share"] == 0.15 and rep["min_rows"] == 10


def test_the_clip_equals_epx_priced_clip_on_every_eligible_row_of_a_fired_cell():
    """The rule's clip is scripts/analysis/epx_price.py::clip_eobt (the definition EPX priced), row for row.
    Mutation: clip to LOBT + 60 s -> red."""
    spec = importlib.util.spec_from_file_location("epx_price", ROOT / "scripts" / "analysis" / "epx_price.py")
    epx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(epx)
    rows = _rows()
    new, _, _ = _serve(rows)
    cell = C.eligible(rows, ("LFPG",)) & (rows.MVT_TIME_UTC_mvt.dt.month == 7).to_numpy()
    sec = lambda s: ((s - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()).to_numpy()  # noqa: E731
    want = epx.clip_eobt(sec(rows.EOBT_1_flt[cell]), sec(rows.LOBT_flt[cell]))
    np.testing.assert_array_equal(sec(new.EOBT_1_flt[cell]), want)


def test_a_batch_of_the_scored_file_is_decided_on_the_scored_files_whole_cell():
    """Batch = ids 1, 5-10 of the scored world: alone its share is 1/7 < 0.15 (and 7 rows < the floor); the scored cell's
    is 0.4, so id 1 is clipped. Mutation: decide on the batch -> DetectorError (or no clip) -> red."""
    scored = _rows()
    batch = scored[scored.MVT_ID_mvt.isin([1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])].reset_index(drop=True)
    new, ids, rep = _serve(batch, scored)
    assert ids.tolist() == [1.0] and rep["population"] == "scored_file"
    assert new.EOBT_1_flt[new.MVT_ID_mvt == 1.0].iloc[0] == batch.LOBT_flt[batch.MVT_ID_mvt == 1.0].iloc[0]


def test_a_batch_from_another_file_is_its_own_population_and_a_mixed_batch_is_refused():
    """Mutation: treat 'none in the scored file' as mixed -> red; let a mixed batch through -> red."""
    scored = _rows()
    other = scored.assign(MVT_ID_mvt=scored.MVT_ID_mvt + 1000.0)
    _, ids, rep = _serve(other, scored)
    assert rep["population"] == "batch" and sorted(ids.tolist()) == [1001.0, 1002.0, 1003.0, 1004.0]
    mixed = pd.concat([scored.iloc[:3], other.iloc[3:10]], ignore_index=True)
    with pytest.raises(E.PopulationError, match="mixes 3 scored-file"):
        _serve(mixed, scored)


def test_serve_without_eligible_rows_is_a_reported_noop():
    """A batch with no eligible row (another airport) is returned unchanged with an empty report, and no population is
    decided for it. Mutation (2026-10-03): drop the early return -> a population is decided -> red."""
    rows = _rows()
    egll = rows[rows.ADEP_mvt == "EGLL"].reset_index(drop=True)
    new, ids, rep = _serve(egll, rows)
    pd.testing.assert_frame_equal(new, egll)
    assert len(ids) == 0 and rep["n_clipped"] == 0 and rep["cells"] == [] and rep["population"] is None


def test_the_rule_refuses_rows_without_a_column_it_reads():
    """Fail loud at the boundary: a frame without LOBT_flt is refused by name, never treated as 'not eligible'.
    Mutation: drop _check -> a KeyError (or a silent pass) instead -> red."""
    with pytest.raises(E.MissingColumnError, match="LOBT_flt"):
        C.eligible(_rows().drop(columns="LOBT_flt"), ("LFPG",))


def test_cell_table_of_rows_without_an_eligible_departure_is_empty_with_its_columns():
    """Mutation: return None on no eligible rows -> red."""
    t = C.cell_table(_rows(), ("EDDF",))
    assert t.empty and list(t.columns) == C.TABLE_COLS
    assert C.decide(t, 0.15, 1000).empty


def test_apply_reads_the_knobs_and_the_scored_file(tmp_path):
    """Mutation: apply ignores modules.eobt_clip.min_share -> the 0.5 config still clips -> red."""
    rows = _rows()
    p = tmp_path / "scored.parquet"
    rows.to_parquet(p, index=False)
    base = {"data.scored_file": str(p), "modules.eobt_clip.enabled": True, "modules.eobt_clip.min_rows": 10}
    _, ids, rep = C.apply(_cfg(**base), rows)
    assert sorted(ids.tolist()) == [1.0, 2.0, 3.0, 4.0] and rep["population"] == "scored_file"
    _, ids, _ = C.apply(_cfg(**base, **{"modules.eobt_clip.min_share": 0.5}), rows)
    assert len(ids) == 0


# ---------------------------------------------------------------------------------------------- the e2e world

@pytest.mark.slow
def test_world_predict_with_the_clip_equals_predict_on_clipped_rows_and_fit_never_clips(e2e_world, fm1_world_mod,
                                                                                        monkeypatch):
    """The world (EDDF + LSZH) has no LFPG, so the rule is pointed at EDDF with min_share 0 (fires on every cell) and a
    floor of 1. (1) fit with the clip on never calls it (the clip is predict time only); (2) predict with the clip on
    equals predict with it off on rows clipped beforehand — every stage sees the clipped rows, as in EPX; (3) the
    provenance marks exactly the clipped ids, the report carries the stage's report and the plan, and rows_digest is
    the caller's (unclipped) rows'. Mutation: the engine clips but builds the features from the unclipped rows -> (2)
    red; rows_digest of the clipped rows -> (3) red."""
    from prc import api
    from prc.pipeline.fm1 import engine as G
    world = e2e_world[0]
    over = {"modules.eobt_clip.enabled": True, "modules.eobt_clip.airports": ["EDDF"],
            "modules.eobt_clip.min_share": 0.0, "modules.eobt_clip.min_rows": 1,
            "run.artifacts_dir": str(world / "art_clp")}
    cfg_on = K.parse_config(fm1_world_mod.config_raw(world, **over))

    def refuse(*a, **k):
        raise AssertionError("the clip ran at fit time")
    monkeypatch.setattr(C, "serve", refuse)
    art_on = api.fit(cfg_on, save=False, log=lambda *a, **k: None)
    monkeypatch.undo()
    art_off = dataclasses.replace(art_on, config=cfg_on.replace(**{"modules.eobt_clip.enabled": False}))
    scored = pd.read_parquet(world / "raw" / "scored.parquet")
    quiet = lambda *a, **k: None  # noqa: E731
    pred = api.predict(art_on, scored, log=quiet).set_index("MVT_ID_mvt")
    clipped_rows, ids, rep = C.apply(cfg_on, scored)
    ref = api.predict(art_off, clipped_rows, log=quiet).set_index("MVT_ID_mvt")
    base = api.predict(art_off, scored, log=quiet).set_index("MVT_ID_mvt")
    assert len(ids) > 0
    pd.testing.assert_series_equal(pred.TAXITIME_SEC_mvt, ref.TAXITIME_SEC_mvt)
    assert (pred.TAXITIME_SEC_mvt != base.TAXITIME_SEC_mvt).any()
    assert set(pred.index[pred.eobt_clipped.astype(bool)]) == set(ids.tolist())
    assert "eobt_clipped" not in base.columns
    r = pred.attrs["fm1_report"]
    assert r["eobt_clip"] == rep and "eobt_clip" in r["stages"]
    assert r["rows_digest"] == G.rows_digest(scored) == base.attrs["fm1_report"]["rows_digest"]


# ---------------------------------------------------------------------------------------------- the REAL files

@pytest.mark.slow
def test_on_the_real_files_the_detector_fires_at_lfpg_july_2026_only():
    """Label-free acceptance 1-2 (CLP): with the registered knobs, LFPG 2026-07 fires (share 0.351), LFPG 2026-01 does not
    (0.061), and no 2025 month file fires (each its own population; LFPG 3.9-7.1 %). Clipped = the July late rows.
    Mutation: min_share 0.05 -> January and most 2025 months fire -> red."""
    from prc.pipeline.fm1 import engine as G
    cfg = K.load_config(ROOT / "configs" / "fm1.yaml", overrides=["modules.eobt_clip.enabled=true"])
    scored = pd.read_parquet(cfg.path("data.scored_file"), columns=list(C.COLS))
    _, ids, rep = C.apply(cfg, scored)
    fired = [(c["airport"], c["month"]) for c in rep["cells"] if c["fired"]]
    assert fired == [("LFPG", "2026-07")] and rep["population"] == "scored_file"
    cells = {c["month"]: c for c in rep["cells"]}
    assert abs(cells["2026-01"]["share"] - 0.061) <= 0.01
    assert len(ids) == rep["n_clipped"] == cells["2026-07"]["n_late"] and 7_400 <= len(ids) <= 7_600
    for m in range(1, 13):
        raw = pd.read_parquet(G.month_file(cfg, m), columns=list(C.COLS))
        _, ids, rep = C.serve(raw, cfg["modules.eobt_clip.airports"], cfg["modules.eobt_clip.min_share"],
                              cfg["modules.eobt_clip.min_rows"], scored)
        assert rep["population"] == "batch" and len(ids) == 0 and not any(c["fired"] for c in rep["cells"]), m
