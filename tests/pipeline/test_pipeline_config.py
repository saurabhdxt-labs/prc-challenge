"""prc.pipeline.config -- the config is code: good configs load, every bad one fails loud with a NAMED error
that carries the dotted path of the offending key.

Mutation rehearsal (2026-09-11, scratchpad p2a/mutate.py): every test below was run against a targeted break
of prc/pipeline/config.py and went RED, then GREEN on restore; the break is named in each docstring (the
parametrized cases carry theirs in the CASES table).
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pipeline_world", pathlib.Path(__file__).with_name("pipeline_world.py"))
pw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pw)

from prc.pipeline import config as C  # noqa: E402
from prc.pipeline import errors as E  # noqa: E402

TZ = {"EDDF": "Europe/Berlin", "EDDM": "Europe/Berlin", "EGLL": "Europe/London", "EHAM": "Europe/Amsterdam",
      "LEBL": "Europe/Madrid", "LEMD": "Europe/Madrid", "LFPG": "Europe/Paris", "LIRF": "Europe/Rome",
      "LSZH": "Europe/Zurich", "LTFM": "Europe/Istanbul"}


def parse(raw):
    return C.parse_config(raw, root=ROOT)


def test_the_shipped_config_loads_with_the_ten_airports_and_routes_lirf_to_its_rules():
    """Rehearsed: KIND_FOR_CLASS['unmatched'] narrowed to ('unmatched',) -> RED (LIRF's rules lane refused)."""
    cfg = C.load_config(ROOT / "configs" / "pipeline.yaml")
    assert set(cfg.airport_codes) == set(TZ) and len(cfg.airport_codes) == 10
    assert {c: a.tz for c, a in cfg.airports.items()} == TZ
    for code in TZ:
        assert cfg.lane_for(code, "matched") == "matched"
        assert cfg.lane_for(code, "unmatched") == ("lirf_rules" if code == "LIRF" else "unmatched")
    lirf = cfg.airports["LIRF"].rules
    assert lirf.order == C.RULE_ORDER[:3] and lirf.local_day is None and (lirf.schedule_floor.lo_s, lirf.schedule_floor.hi_s) == (24_000.0, 86_400.0)
    assert cfg.seeds == (0, 1, 2) and cfg.fallback_airports() == ()
    m = cfg.lanes["matched"]
    assert (m.model, m.target, m.source) == ("lightgbm", "delta", "saved_boosters")
    assert [b for b, _ in m.feature_blocks] == ["queue", "order"]
    assert m.boosters == tuple(f"lgbm_v6_queue_order_seed{s}.txt" for s in (0, 1, 2))
    assert cfg.source_path == (ROOT / "configs" / "pipeline.yaml").resolve()


def test_config_hash_ignores_key_order_and_changes_with_content():
    """Rehearsed: canonical_hash without sort_keys -> RED (the reordered mapping hashes differently)."""
    raw = pw.real_raw()
    h = parse(raw).config_hash
    reordered = {k: raw[k] for k in reversed(list(raw))}
    assert parse(reordered).config_hash == h
    changed = pw.fresh(raw)
    changed["airports"]["EDDF"]["tz"] = "Europe/Paris"
    assert parse(changed).config_hash != h and len(h) == 64


def test_a_new_airport_is_one_entry_and_rides_the_pooled_lanes_flagged():
    """Rehearsed: Airport.fallback returning False -> RED."""
    raw = pw.fresh(pw.real_raw())
    raw["airports"]["EKCH"] = {"tz": "Europe/Copenhagen", "history": False,
                               "lanes": {"matched": "matched", "unmatched": "unmatched"}}
    cfg = parse(raw)
    assert cfg.fallback_airports() == ("EKCH",)
    assert cfg.lane_for("EKCH", "matched") == "matched" and cfg.lane_for("EKCH", "unmatched") == "unmatched"


def test_lane_lookup_for_an_airport_outside_the_registry_is_refused_by_name():
    """Rehearsed: lane_for's registry check deleted -> RED (KeyError instead of UnknownAirportError)."""
    cfg = parse(pw.real_raw())
    with pytest.raises(E.UnknownAirportError, match="EKCH"):
        cfg.lane_for("EKCH", "matched")
    with pytest.raises(E.RoutingError, match="row class"):
        cfg.lane_for("EDDF", "arrival")


def _set(path, value):
    def f(raw):
        node = raw
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    return f


def _del(path):
    def f(raw):
        node = raw
        for k in path[:-1]:
            node = node[k]
        del node[path[-1]]
    return f


def _lirf_rules_on_plain_lane(raw):
    raw["airports"]["LIRF"]["lanes"]["unmatched"] = "unmatched"          # rules kept, lane without rules


def _route_eddf_to_rules(raw):
    raw["airports"]["EDDF"]["lanes"]["unmatched"] = "lirf_rules"
    raw["airports"]["EDDF"]["rules"] = {"fill_classifier": {}}


def _route_eddf_to_rules_without_rules(raw):
    raw["airports"]["EDDF"]["lanes"]["unmatched"] = "lirf_rules"


def _unused_lane(raw):
    raw["lanes"]["spare"] = {"kind": "unmatched", "model": "nonfill_body"}


def _refit_with_boosters(raw):
    raw["lanes"]["matched"]["source"] = "refit"


#: (id, mutator, error class, message fragment, the config.py break that turns the case RED)
CASES = [
    ("unknown_top_key", _set(["extra"], 1), E.UnknownKeyError, "<config>: unknown key", "_map's unknown-key check deleted"),
    ("unknown_lane_key", _set(["lanes", "matched", "learning_rate"], 0.1), E.UnknownKeyError, "lanes.matched",
     "idem (_map's unknown-key check deleted)"),
    ("typo_airport_key", _set(["airports", "EDDF", "tzz"], "x"), E.UnknownKeyError, "airports.EDDF", "idem"),
    ("seeds_string", _set(["seeds"], "0,1,2"), E.ConfigTypeError, "seeds: expected a list", "_list's isinstance check deleted"),
    ("seed_bool", _set(["seeds"], [True, 1, 2]), E.ConfigTypeError, "seeds[0]", "_int's bool exclusion deleted"),
    ("history_string", _set(["airports", "EDDF", "history"], "yes"), E.ConfigTypeError, "airports.EDDF.history",
     "_bool's check deleted"),
    ("lanes_as_list", _set(["airports", "EDDF", "lanes"], ["matched"]), E.ConfigTypeError, "airports.EDDF.lanes",
     "_map's mapping check deleted"),
    ("repeated_seed", _set(["seeds"], [0, 0, 1]), E.ConfigValueError, "repeated seed", "_parse_seeds' repeat check deleted"),
    ("month_13", _set(["folds", "lomo_months"], [1, 13]), E.ConfigValueError, "folds.lomo_months[1]", "_int's range check deleted"),
    ("es_overlaps_fold_a", _set(["folds", "early_stopping_months"], [1, 9]), E.ConfigValueError, "overlap",
     "the ES/fold-A overlap check deleted"),
    ("bad_tz", _set(["airports", "EDDF", "tz"], "Europe/Nowhere"), E.ConfigValueError, "airports.EDDF.tz", "the ZoneInfo check deleted"),
    ("bad_code", lambda r: r["airports"].update({"EDD": r["airports"]["EDDF"]}), E.UnknownAirportError, "four-letter",
     "AIRPORT_RE check deleted"),
    ("unknown_lane_ref", _set(["airports", "EDDF", "lanes", "unmatched"], "nope"), E.UnknownLaneError, "nope",
     "the lane-reference check deleted"),
    ("matched_to_unmatched_lane", _set(["airports", "EDDF", "lanes", "matched"], "unmatched"), E.ConfigValueError,
     "airports.EDDF.lanes.matched", "the KIND_FOR_CLASS check deleted"),
    ("unknown_model", _set(["lanes", "unmatched", "model"], "xgboost"), E.UnknownModelError, "xgboost", "MODEL_NAMES check deleted"),
    ("model_wrong_kind", _set(["lanes", "unmatched", "model"], "lightgbm"), E.ConfigValueError, "cannot serve",
     "MODELS_FOR_KIND check deleted"),
    ("boosters_count", _set(["lanes", "matched", "boosters"], ["a.txt", "b.txt"]), E.ConfigValueError, "one per seed",
     "the booster-count check deleted"),
    ("blocks_out_of_order", _set(["lanes", "matched", "feature_blocks"], {"order": "x", "queue": "y"}),
     E.ConfigValueError, "block order", "the block-order check deleted"),
    ("unknown_block", _set(["lanes", "matched", "feature_blocks"], {"stand": "x"}), E.UnknownKeyError,
     "feature_blocks", "idem (unknown-key)"),
    ("floor_changed", _set(["airports", "LIRF", "rules", "schedule_floor", "lo_s"], 20_000), E.PinnedValueError,
     "rome_bandfloor", "the pinned-band check deleted"),
    ("rules_reordered", _set(["airports", "LIRF", "rules"], {"dateslip": {}, "fill_classifier": {}}), E.PinnedValueError,
     "shipped order", "the rule-order check deleted"),
    ("rules_on_plain_lane", _lirf_rules_on_plain_lane, E.ConfigValueError, "never run", "the rules-lane consistency check deleted"),
    ("rules_on_other_airport", _route_eddf_to_rules, E.PinnedValueError, "written for LIRF", "the rules-airport pin deleted"),
    ("rules_lane_without_rules", _route_eddf_to_rules_without_rules, E.ConfigValueError, "with no rules", "idem"),
    ("rules_without_history", _set(["airports", "LIRF", "history"], False), E.ConfigValueError, "own history",
     "the rules/history check deleted"),
    ("unused_lane", _unused_lane, E.ConfigValueError, "routed from no airport", "the unused-lane check deleted"),
    ("bad_team", _set(["team"], "Merry_Quicksand"), E.ConfigValueError, "team", "TEAM_RE check deleted"),
    ("schema_version", _set(["schema_version"], 2), E.ConfigValueError, "schema_version", "the version check deleted"),
    ("missing_section", _del(["paths"]), E.ConfigError, "missing required", "the required-key check deleted"),
    ("base_key", _set(["base"], "submissions/merry-quicksand_v2.parquet"), E.SpliceRefusedError, "never splices",
     "the SPLICE_KEYS check deleted"),
    ("path_into_submissions", _set(["paths", "template_file"], "submissions/merry-quicksand_v10.parquet"),
     E.SpliceRefusedError, "submissions/", "_path's submissions check deleted"),
    ("refit_with_boosters", _refit_with_boosters, E.UnknownKeyError, "source saved_boosters only",
     "the refit/booster exclusivity check deleted"),
    ("target_on_unmatched_lane", _set(["lanes", "unmatched", "target"], "delta"), E.UnknownKeyError, "matched lane only",
     "the non-matched extra-key check deleted"),
]


@pytest.mark.parametrize("case_id,mutate,err,fragment,rehearsal", CASES, ids=[c[0] for c in CASES])
def test_a_bad_config_fails_loud_with_a_named_error_and_the_key_path(case_id, mutate, err, fragment, rehearsal):
    """Each case mutates the good config in one place; the loader must raise exactly that error class with a
    message naming the key. Rehearsed per case: the break in the CASES table's last column -> RED."""
    raw = pw.fresh(pw.real_raw())
    mutate(raw)
    with pytest.raises(err, match=fragment.replace("[", r"\[").replace("]", r"\]").replace(".", r"\.")):
        parse(raw)


def test_load_config_refuses_a_missing_file_and_invalid_yaml(tmp_path):
    """Rehearsed: the exists() check deleted -> RED (FileNotFoundError is not a ConfigError)."""
    with pytest.raises(E.ConfigError, match="does not exist"):
        C.load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("airports: [unclosed\n")
    with pytest.raises(E.ConfigError, match="not valid YAML"):
        C.load_config(bad)


def test_the_matched_lane_accepts_an_optional_adsb_stage_and_refuses_it_elsewhere():
    """Absent = today's lane (A1); present = both paths required; on a non-matched lane = refused by name.
    Rehearsed 2026-09-11 (c4): parsing without the `stage` hand-off to Lane -> RED (adsb_stage None)."""
    raw = pw.fresh(pw.real_raw())
    assert C.parse_config(raw, root=ROOT).lanes["matched"].adsb_stage is None
    h0 = C.parse_config(raw, root=ROOT).config_hash
    raw["lanes"]["matched"]["adsb_stage"] = {"table": "data/adsb/v2/features_2026janjul.parquet", "models_dir": "data/adsb/v2/models"}
    cfg = C.parse_config(raw, root=ROOT)
    st = cfg.lanes["matched"].adsb_stage
    assert st.table == ROOT / "data/adsb/v2/features_2026janjul.parquet" and st.models_dir == ROOT / "data/adsb/v2/models"
    assert cfg.config_hash != h0
    bad = pw.fresh(raw)
    del bad["lanes"]["matched"]["adsb_stage"]["models_dir"]
    with pytest.raises(E.ConfigError, match="models_dir"):
        C.parse_config(bad, root=ROOT)
    bad = pw.fresh(pw.real_raw())
    bad["lanes"]["unmatched"]["adsb_stage"] = {"table": "x", "models_dir": "y"}
    with pytest.raises(E.UnknownKeyError, match="adsb_stage"):
        C.parse_config(bad, root=ROOT)


def _raw_with_rules(rules):
    import copy
    raw = copy.deepcopy(pw.real_raw())
    raw["airports"]["LIRF"]["rules"] = rules
    return raw


def test_the_local_day_schedule_rule_parses_last_with_its_pinned_band():
    """RLD (plans/PREREG_rome_local_day_rule_2026_09_11.md) is the fourth Rome rule, applied after the floor, with the
    prereg's band [24,000, 86,400). Rehearsed 2026-09-11 (c70): _parse_rules ignoring local_day_schedule -> RED (no
    band); the band pin removed -> RED (the 20,000 band parses)."""
    base = {"fill_classifier": {}, "dateslip": {}, "schedule_floor": {"lo_s": 24000, "hi_s": 86400}}
    cfg = C.parse_config(_raw_with_rules({**base, "local_day_schedule": {"lo_s": 24000, "hi_s": 86400}}), root=ROOT)
    r = cfg.airports["LIRF"].rules
    assert r.order[-1] == "local_day_schedule" and (r.local_day.lo_s, r.local_day.hi_s) == (24_000.0, 86_400.0)
    assert C.parse_config(_raw_with_rules(base), root=ROOT).airports["LIRF"].rules.local_day is None
    with pytest.raises(E.PinnedValueError, match="local_day_schedule"):
        C.parse_config(_raw_with_rules({**base, "local_day_schedule": {"lo_s": 20000, "hi_s": 86400}}), root=ROOT)
    with pytest.raises(E.PinnedValueError, match="shipped order"):
        C.parse_config(_raw_with_rules({"local_day_schedule": {"lo_s": 24000, "hi_s": 86400}, **base}), root=ROOT)


@pytest.mark.skipif(not (ROOT / "configs" / "pipeline_adn.yaml").exists(), reason="the ADN config is another lane's file")
def test_the_configs_that_built_v11_and_v12_load_and_carry_what_their_names_claim():
    """The shipped submissions were built from configs/pipeline_adn.yaml (v11) and configs/pipeline_v12.yaml (v12); until
    now every config test parsed only configs/pipeline.yaml, so the two that actually shipped were unparsed by any test.
    Rehearsed 2026-09-11 (c70): the stage block dropped from pipeline_v12.yaml -> RED; local_day_schedule dropped -> RED."""
    adn = C.load_config(ROOT / "configs" / "pipeline_adn.yaml")
    v12 = C.load_config(ROOT / "configs" / "pipeline_v12.yaml")
    base = C.load_config(ROOT / "configs" / "pipeline.yaml")
    assert base.lanes["matched"].adsb_stage is None and base.airports["LIRF"].rules.local_day is None
    for cfg in (adn, v12):
        st = cfg.lanes["matched"].adsb_stage
        assert st is not None and st.table.name.startswith("features_2026") and st.models_dir.name == "models"
    assert adn.airports["LIRF"].rules.local_day is None                      # v11 shipped WITHOUT the Rome rule
    ld = v12.airports["LIRF"].rules.local_day
    assert ld is not None and (ld.lo_s, ld.hi_s) == (24_000.0, 86_400.0) and v12.airports["LIRF"].rules.order[-1] == "local_day_schedule"
    assert adn.config_hash != base.config_hash and v12.config_hash != adn.config_hash
