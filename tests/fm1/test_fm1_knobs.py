"""FM1 config schema (prc/pipeline/fm1/knobs.py) — acceptance test AT1 of plans/PREREG_fm1_2026_09_28.md.

Config is code (testing.md Part 1 rule 11): every knob documented, every value validated, every mistake loud with the
dotted path. Mutation rehearsal (manual, 2026-09-28) is recorded per test and in tests/fm1/MUTATION_FM1.md.
"""
from __future__ import annotations

import copy
import pathlib

import pytest
import yaml

from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K

ROOT = pathlib.Path(__file__).resolve().parents[2]
FM1_YAML = ROOT / "configs" / "fm1.yaml"


def _raw() -> dict:
    return yaml.safe_load(FM1_YAML.read_text())


def _set(raw: dict, path: str, value) -> dict:
    raw = copy.deepcopy(raw)
    node = raw
    parts = path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return raw


# ---------------------------------------------------------------------------------------------- the schema itself

def test_every_knob_is_documented():
    """Mutation: blank one knob's `effect` in knobs.SCHEMA -> check_schema raises -> this test goes red."""
    K.check_schema(K.SCHEMA)          # raises on any gap
    for k in K.SCHEMA:
        assert len(k.effect) >= 20, k.path
        assert k.evidence and K.evidence_file(k.evidence).exists(), (k.path, k.evidence)   # dev tree (BC-38)
        assert k.status in K.STATUSES, k.path
        if k.kind in ("int", "float"):
            assert k.lo is not None and k.hi is not None, f"{k.path}: numeric knob without a range"
        if k.kind == "choice":
            assert k.choices, k.path


def test_check_schema_refuses_an_undocumented_knob():
    bad = K.Knob("x.y", "bool", True, effect="", evidence="plans/PREREG_fm1_2026_09_28.md", status="implemented")
    with pytest.raises(E.ConfigError, match=r"x\.y: effect"):
        K.check_schema((*K.SCHEMA, bad))


def test_check_evidence_files_refuses_a_missing_evidence_file():
    """The development tree's documentation check (BC-38: it used to run at import, inside check_schema, and broke every
    clone). Mutation: drop the exists() test -> red."""
    bad = K.Knob("x.y", "bool", True, effect="a long enough effect sentence here", evidence="plans/NO_SUCH_FILE.md",
                 status="implemented")
    with pytest.raises(E.ConfigError, match=r"x\.y: evidence .*does not exist"):
        K.check_evidence_files((*K.SCHEMA, bad))
    K.check_evidence_files(K.SCHEMA)                 # every real pointer resolves in the development tree


def test_regression_bug_38_the_import_time_schema_check_reads_no_file():
    """BC-38: check_schema runs at import, so it must not depend on files a clone does not carry; an evidence pointer to
    a missing file is a documentation gap (check_evidence_files), not a broken schema. Mutation: put the exists() test
    back into check_schema -> red here and in tests/fm1/test_fm1_clean_checkout.py."""
    bad = K.Knob("x.y", "bool", True, effect="a long enough effect sentence here", evidence="plans/NO_SUCH_FILE.md",
                 status="implemented")
    K.check_schema((*K.SCHEMA, bad))
    with pytest.raises(E.ConfigError, match=r"x\.y: evidence is missing"):
        K.check_schema((*K.SCHEMA, K.Knob("x.y", "bool", True, effect="a long enough effect sentence here", evidence="",
                                         status="implemented")))


def test_check_schema_refuses_duplicate_and_prefix_paths():
    dup = K.SCHEMA[0]
    with pytest.raises(E.ConfigError, match="duplicate knob"):
        K.check_schema((*K.SCHEMA, dup))
    child = K.Knob(K.SCHEMA[0].path + ".child", "bool", True, effect="a long enough effect sentence here",
                   evidence="plans/PREREG_fm1_2026_09_28.md", status="implemented")
    with pytest.raises(E.ConfigError, match="is both a knob and a section"):
        K.check_schema((*K.SCHEMA, child))


def test_every_default_validates_against_its_own_knob():
    """Mutation M01: an exclusive lower bound in _check_num refuses run.seeds' default 0 -> red."""
    cfg = K.parse_config({})
    for k in K.SCHEMA:
        assert cfg[k.path] == K.normalise_default(k), k.path


# ---------------------------------------------------------------------------------------------- configs/fm1.yaml

def test_fm1_yaml_states_every_knob_at_its_default():
    """fm1.yaml IS the documented forecaster: every knob stated, each at the schema default.
    Mutation: change `run.threads: 4` to 5 in a copy -> the comparison goes red (asserted below on the copy)."""
    raw = _raw()
    assert raw.pop("schema_version") == K.SCHEMA_VERSION
    flat = K.flatten(raw)
    assert set(flat) == set(K.KNOBS), (sorted(set(K.KNOBS) - set(flat)), sorted(set(flat) - set(K.KNOBS)))
    cfg = K.load_config(FM1_YAML)
    for k in K.SCHEMA:
        assert cfg[k.path] == K.normalise_default(k), k.path
    assert K.differs_from_defaults(cfg) == []
    assert K.differs_from_defaults(K.parse_config(_set(raw, "run.threads", 5))) == ["run.threads"]


def test_fm1_yaml_hash_equals_the_all_defaults_hash():
    assert K.load_config(FM1_YAML).hash == K.parse_config({}).hash


# ---------------------------------------------------------------------------------------------- loud failures

def test_unknown_key_names_the_dotted_path():
    """Mutation: make flatten() drop unknown leaves -> no error -> red."""
    with pytest.raises(E.UnknownKeyError, match=r"matched\.lightgbm\.num_leafs"):
        K.parse_config(_set(_raw(), "matched.lightgbm.num_leafs", 31))


def test_unknown_section_names_the_dotted_path():
    with pytest.raises(E.UnknownKeyError, match=r"modules\.blk"):
        K.parse_config(_set(_raw(), "modules.blk.enabled", True))


def test_a_knob_given_a_mapping_is_a_type_error():
    with pytest.raises(E.ConfigTypeError, match=r"run\.threads: expected an integer"):
        K.parse_config(_set(_raw(), "run.threads", {"a": 1}))


@pytest.mark.parametrize("path,value,match", [
    ("run.threads", True, r"run\.threads: expected an integer, got bool"),
    ("run.threads", 4.0, r"run\.threads: expected an integer, got float"),
    ("matched.lightgbm.learning_rate", "0.01", r"matched\.lightgbm\.learning_rate: expected a number, got str"),
    ("lanes.matched", "yes", r"lanes\.matched: expected true/false, got str"),
    ("run.seeds", 0, r"run\.seeds: expected a list"),
    ("run.seeds", [0, "1"], r"run\.seeds\[1\]: expected an integer"),
    ("airports.tz", ["EDDF"], r"airports\.tz: expected a mapping"),
])
def test_wrong_types_fail_with_the_path(path, value, match):
    """Mutation: accept bool as int in _check_int (drop the isinstance(v, bool) guard) -> the first case goes red."""
    with pytest.raises(E.ConfigTypeError, match=match):
        K.parse_config(_set(_raw(), path, value))


@pytest.mark.parametrize("path,value,match", [
    ("run.threads", 0, r"run\.threads: 0 is outside \[1, 64\]"),
    ("matched.lightgbm.learning_rate", 0.0, r"matched\.lightgbm\.learning_rate: 0\.0 is outside"),
    ("matched.blend.catboost_weight", 1.5, r"matched\.blend\.catboost_weight: 1\.5 is outside \[0\.0, 1\.0\]"),
    ("adsb.min_coverage", 4, r"adsb\.min_coverage: 4 is outside \[1, 3\]"),
    ("data.train_months", [1, 13], r"data\.train_months\[1\]: 13 is outside \[1, 12\]"),
])
def test_out_of_range_fails_with_the_range(path, value, match):
    """Every bound names the knob and its range. Mutation M01 (MUTATION_FM1.md): `lo < v` instead of `lo <= v` in
    _check_num -> an inclusive lower bound (seeds 0) is refused -> red."""
    with pytest.raises(E.ConfigValueError, match=match):
        K.parse_config(_set(_raw(), path, value))


def test_learning_rate_upper_bound_is_inclusive():
    cfg = K.parse_config(_set(_raw(), "matched.lightgbm.learning_rate", 1.0))
    assert cfg["matched.lightgbm.learning_rate"] == 1.0


def test_bad_choice_lists_the_choices():
    with pytest.raises(E.ConfigValueError, match=r"run\.purpose: 'board' is not one of \['forecaster', 'scored_period'\]"):
        K.parse_config(_set(_raw(), "run.purpose", "board"))


def test_unquoted_yaml_off_is_explained():
    """YAML 1.1 turns a bare `off` into False; the error says so instead of a bare type error."""
    raw = yaml.safe_load("modules:\n  storm:\n    mode: off\n")
    with pytest.raises(E.ConfigTypeError, match=r"modules\.storm\.mode: got bool False; YAML reads an unquoted"):
        K.parse_config(raw)


def test_xgboost_is_refused_as_cut():
    """Owner decision 5. Mutation: drop the CUT branch in _check_choice -> a plain choice error (different type) -> red."""
    with pytest.raises(E.LearnerCutError, match=r"matched\.learner: 'xgboost' is CUT"):
        K.parse_config(_set(_raw(), "matched.learner", "xgboost"))


def test_mlp_learner_is_refused_as_out_of_scope():
    with pytest.raises(E.LearnerCutError, match=r"matched\.learner: 'mlp' is not a days 1-3 learner"):
        K.parse_config(_set(_raw(), "matched.learner", "mlp"))


def test_a_path_into_submissions_is_a_splice():
    with pytest.raises(E.SpliceRefusedError, match=r"data\.raw_dir: .*submissions/"):
        K.parse_config(_set(_raw(), "data.raw_dir", "submissions"))


@pytest.mark.parametrize("key", ["base", "base_submission", "splice"])
def test_splice_keys_are_refused_anywhere(key):
    with pytest.raises(E.SpliceRefusedError, match="never splices"):
        K.parse_config(_set(_raw(), f"matched.{key}", "submissions/x.parquet"))


# ---------------------------------------------------------------------------------------------- cross-field rules

def test_duplicate_seeds_are_refused():
    with pytest.raises(E.ConfigValueError, match=r"run\.seeds: repeated"):
        K.parse_config(_set(_raw(), "run.seeds", [0, 0]))


def test_es_months_must_be_training_months():
    with pytest.raises(E.ConfigValueError, match=r"matched\.es_months: \[9\] not in data\.train_months"):
        K.parse_config(_set(_set(_raw(), "data.train_months", [1, 2, 3]), "matched.es_months", [3, 9]))


def test_es_months_cannot_be_all_training_months():
    with pytest.raises(E.ConfigValueError, match="no fit month left"):
        K.parse_config(_set(_set(_raw(), "data.train_months", [3, 9]), "matched.es_months", [3, 9]))


def test_tz_map_must_cover_exactly_the_airports():
    raw = _set(_raw(), "airports.codes", ["EDDF", "EKCH"])
    with pytest.raises(E.ConfigValueError, match=r"airports\.tz: .*EKCH"):
        K.parse_config(raw)


def test_tz_must_be_iana():
    tz = dict(_raw()["airports"]["tz"])
    tz["EDDF"] = "Mars/Olympus"
    with pytest.raises(E.ConfigValueError, match=r"airports\.tz\.EDDF: 'Mars/Olympus' is not an IANA time zone"):
        K.parse_config(_set(_raw(), "airports.tz", tz))


def test_airport_codes_are_icao():
    with pytest.raises(E.UnknownAirportError, match=r"airports\.codes\[0\]: 'FRA'"):
        K.parse_config(_set(_raw(), "airports.codes", ["FRA"]))


def test_catboost_learner_cannot_blend_with_itself():
    with pytest.raises(E.ConfigValueError, match="catboost_weight must be 0"):
        K.parse_config(_set(_raw(), "matched.learner", "catboost"))
    cfg = K.parse_config(_set(_set(_raw(), "matched.learner", "catboost"), "matched.blend.catboost_weight", 0.0))
    assert cfg["matched.learner"] == "catboost"


def test_stack_needs_the_adsb_block():
    with pytest.raises(E.ConfigValueError, match=r"adsb\.stack\.enabled needs features\.adsb"):
        K.parse_config(_set(_raw(), "features.adsb", False))


def test_adsb_block_needs_adsb_lol():
    raw = _set(_raw(), "adsb.sources.adsb_lol", False)
    with pytest.raises(E.ConfigValueError, match=r"features\.adsb needs adsb\.sources\.adsb_lol"):
        K.parse_config(raw)


def test_a_config_with_no_lane_is_refused():
    raw = _set(_set(_raw(), "lanes.matched", False), "lanes.unmatched", False)
    with pytest.raises(E.ConfigValueError, match="at least one lane"):
        K.parse_config(raw)


# ---------------------------------------------------------------------------------------------- resolution + overrides

@pytest.mark.parametrize("purpose,mtx,expected", [
    ("forecaster", "auto", False), ("scored_period", "auto", True),
    ("forecaster", "on", True), ("scored_period", "off", False),
])
def test_mtx_auto_follows_the_purpose(purpose, mtx, expected):
    """Owner decision 3: optional; off in the documented forecaster, on for the scored-period build.
    Mutation: resolve auto as `purpose == 'forecaster'` -> red."""
    cfg = K.parse_config(_set(_set(_raw(), "run.purpose", purpose), "calibration.mtx.enabled", mtx))
    assert cfg.mtx_enabled is expected


def test_overrides_are_typed_validated_and_hashed():
    base = K.load_config(FM1_YAML)
    over = K.load_config(FM1_YAML, overrides=["run.purpose=scored_period", "run.seeds=[0]"])
    assert over["run.purpose"] == "scored_period" and over["run.seeds"] == (0,)
    assert over.hash != base.hash
    assert over.overrides == (("run.purpose", "scored_period"), ("run.seeds", [0]))
    with pytest.raises(E.UnknownKeyError, match=r"run\.purpse"):
        K.load_config(FM1_YAML, overrides=["run.purpse=scored_period"])
    with pytest.raises(E.ConfigError, match="expected key=value"):
        K.load_config(FM1_YAML, overrides=["run.purpose"])
    with pytest.raises(E.ConfigTypeError, match=r"run\.threads"):
        K.load_config(FM1_YAML, overrides=["run.threads=four"])


def test_hash_ignores_layout_and_stated_defaults():
    raw = _raw()
    a = K.parse_config(raw).hash
    b = K.parse_config(yaml.safe_load(yaml.safe_dump(raw, sort_keys=True))).hash
    c = K.parse_config({"run": {"threads": 4}}).hash
    d = K.parse_config({"run": {"threads": 5}}).hash
    assert a == b == c != d


def test_not_ported_lists_every_enabled_unported_component():
    """plans/PREREG_fm1_2026_09_28.md §2 + Amendment A3: Rome, ARC and the storm anchor are ported, so the default config
    has nothing unported; the storm-as-feature form and adsb.fi are listed only when switched on; LDI is ported
    (owner decision 2026-09-29), so switching it on adds nothing.
    Mutation: make Knob.activates return False always -> the switched-on list is empty -> red."""
    cfg = K.load_config(FM1_YAML)
    assert cfg.not_ported() == []
    raw = _set(_set(_set(_raw(), "modules.storm.feature", True), "modules.ldi.enabled", True),
               "adsb.sources.adsb_fi", True)
    assert K.parse_config(raw).not_ported() == ["modules.storm.feature"]      # adsb.fi ported 2026-10-03 (AFI)
    for path in ("modules.rome.enabled", "modules.arc.enabled", "modules.rome.e2", "modules.rome.e2w",
                 "unmatched.learner", "modules.storm.mode", "modules.ldi.enabled"):
        assert K.KNOBS[path].status == "implemented", path


def test_unmatched_lane_defaults_are_the_ported_v36_chain():
    """Amendment A3: the documented forecaster runs the shipped v26 body and the Rome module (owner decision 1), ARC at
    LIRF (the shipped scope) and no storm anchor (owner decision 2); the E2 day-check thresholds are the shipped ones.
    Checked on the SCHEMA defaults (an empty config) and on fm1.yaml. Mutation: schema default learner `hgb` -> red."""
    assert K.parse_config({}).hash == K.load_config(FM1_YAML).hash
    cfg = K.parse_config({})
    assert cfg["unmatched.learner"] == "v26" and cfg["modules.rome.enabled"] is True
    assert cfg["modules.rome.e2"] is True and cfg["modules.rome.e2w"] is True
    assert cfg["modules.arc.enabled"] is True and cfg["modules.arc.airports"] == ("LIRF",)
    assert cfg["modules.storm.mode"] == "off" and cfg["modules.storm.feature"] is False
    assert cfg["modules.rome.e2_day_p99"] == 15.941366958491995 and cfg["modules.rome.e2w_day_p99"] == 9.042848041672684
    assert "feature" not in K.KNOBS["modules.storm.mode"].choices


#: the Rome sub-switches (e2, e2w) are read only while the module is on: turning the module off is one switch
_NO_ROME = {"modules.rome.enabled": False}


@pytest.mark.parametrize("flat, msg", [
    ({"unmatched.learner": "hgb"}, "modules.rome.enabled needs unmatched.learner v26"),
    ({"lanes.unmatched": False}, "modules.rome.enabled needs lanes.unmatched"),
    ({"airports.codes": ["EDDF"], "airports.tz": {"EDDF": "Europe/Berlin"}, "modules.arc.enabled": False},
     "modules.rome.enabled needs LIRF in airports.codes"),
    ({"modules.rome.e2": False}, "modules.rome.e2w needs modules.rome.e2"),
    ({**_NO_ROME, "modules.arc.airports": ["LIRF", "EGKK"]}, r"modules.arc.airports: \['EGKK'\] not in airports.codes"),
    ({**_NO_ROME, "lanes.unmatched": False}, "modules.arc.enabled needs lanes.unmatched"),
    ({**_NO_ROME, "modules.arc.enabled": False, "lanes.unmatched": False, "modules.storm.mode": "anchor"},
     "modules.storm.mode anchor needs lanes.unmatched"),
    ({**_NO_ROME, "unmatched.learner": "hgb"}, "modules.arc.enabled needs unmatched.learner v26"),
    ({**_NO_ROME, "modules.arc.enabled": False, "unmatched.learner": "hgb", "modules.storm.mode": "anchor"},
     "modules.storm.mode anchor needs unmatched.learner v26"),
    ({"data.dayoff_tz": "local"}, "unmatched.learner v26 reads build_submission.derive's UTC dayoff"),
    ({"unmatched.encoding_smooth": 20.0}, "unmatched.learner v26 encodes with build_submission.SMOOTH"),
])
def test_unmatched_lane_cross_checks_are_loud(flat, msg):
    """Every combination the ported lane cannot honour is refused with the dotted path, never silently ignored.
    Mutation: drop the e2w -> e2 dependency check -> the e2-off config parses -> red."""
    raw = _raw()
    for k, v in flat.items():
        raw = _set(raw, k, v)
    with pytest.raises(E.ConfigValueError, match=msg):
        K.parse_config(raw)


def test_unmatched_lane_valid_variants_parse():
    """The switch combinations the lane does honour parse (the negative of the cross-checks above)."""
    for flat in ({}, _NO_ROME, {**_NO_ROME, "modules.arc.enabled": False, "unmatched.learner": "hgb",
                                "data.dayoff_tz": "local", "unmatched.encoding_smooth": 20.0},
                 {"modules.rome.e2w": False}, {"modules.storm.mode": "anchor"}, {**_NO_ROME, "modules.rome.e2": False},
                 {"modules.arc.airports": ["LIRF", "EDDF"]}):
        raw = _raw()
        for k, v in flat.items():
            raw = _set(raw, k, v)
        K.parse_config(raw)


@pytest.mark.parametrize("sv", [2, True, "1"])
def test_schema_version_must_be_one(sv):
    with pytest.raises(E.ConfigValueError, match=r"schema_version: .* \(this loader reads 1\)"):
        K.parse_config(_set(_raw(), "schema_version", sv))


def test_config_is_immutable():
    cfg = K.load_config(FM1_YAML)
    with pytest.raises(TypeError):
        cfg.values["run.threads"] = 8          # MappingProxyType
    with pytest.raises(AttributeError):
        cfg.hash = "x"


def test_nested_round_trips():
    cfg = K.load_config(FM1_YAML)
    again = K.parse_config(cfg.nested())
    assert again.hash == cfg.hash and dict(again.values) == dict(cfg.values)


def test_paths_resolve_against_root(tmp_path):
    cfg = K.parse_config({}, root=tmp_path)
    assert cfg.path("data.raw_dir") == tmp_path / "data" / "raw"
    abs_cfg = K.parse_config({"data": {"raw_dir": str(tmp_path / "elsewhere")}}, root=tmp_path)
    assert abs_cfg.path("data.raw_dir") == tmp_path / "elsewhere"
    with pytest.raises(E.ConfigTypeError, match=r"run\.threads is not a path knob"):
        cfg.path("run.threads")


def test_missing_config_file_and_bad_yaml(tmp_path):
    with pytest.raises(E.ConfigError, match="does not exist"):
        K.load_config(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: [unclosed")
    with pytest.raises(E.ConfigError, match="not valid YAML"):
        K.load_config(bad)
    top = tmp_path / "top.yaml"
    top.write_text("- a\n- b\n")
    with pytest.raises(E.ConfigTypeError, match="<config>: expected a mapping"):
        K.load_config(top)


def test_regression_output_location_is_not_part_of_the_model_hash():
    """Found 2026-09-28 (FM1 notes N6): two fits identical except run.artifacts_dir got different artefact keys, so their
    otherwise byte-identical provenance files differed in the embedded run report. An output location does not change the
    model; every knob that can change a prediction stays in the hash. Mutation: hash every knob -> red."""
    a = K.parse_config({"run": {"artifacts_dir": "artifacts/one"}})
    b = K.parse_config({"run": {"artifacts_dir": "artifacts/two"}})
    assert a.hash == b.hash and a["run.artifacts_dir"] != b["run.artifacts_dir"]
    assert K.parse_config({"run": {"threads": 8}}).hash != a.hash
    assert [k.path for k in K.SCHEMA if not k.hashed] == ["run.artifacts_dir"]
