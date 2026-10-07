"""Refit-free derivation of a variant artefact (plans/PREREG_bsv_board_selected_variants_2026_10_01.md, "Mechanism"):
`api.fit(config, reuse_from=<donor artefact>)` reuses the donor's fitted matched learners (the 8-10 h part of a full fit)
when every knob outside the registered allow-list equals the donor's and the input data are the same; frames, encodings,
the unmatched lane and MTX are recomputed, and the recomputed encodings / design columns must equal the donor's.

The contract under test: a derived artefact predicts bit-identically to a from-scratch fit of the same variant config;
anything that could make that false is refused by name."""
from __future__ import annotations

import dataclasses
import pathlib
import re
import shutil

import pandas as pd
import pytest

from prc import api
from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import features as F
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import learners as L

QUIET = lambda *_a, **_k: None          # noqa: E731
#: the registered allow-list, as knob-path prefixes (a `.`-terminated prefix covers a whole group)
FREE_PREFIXES = ("run.purpose", "run.artifacts_dir", "calibration.mtx.", "modules.", "matched.blend.catboost_weight",
                 "adsb.stack.", "adsb.support_gate.", "adsb.min_coverage",
                 "adsb.sources.adsb_fi", "adsb.table_dir",       # AFI 2026-10-03: data-guarded (adsb/train.parquet hash)
                 "data.scored_file", "data.template_file")      # FP-W1 item 8, 2026-10-06: neither is read by engine.fit
                                                                 # (only by engine.predict / build_submission); see
                                                                 # test_fit_never_reads_the_scored_or_template_file below


def _is_free(path: str) -> bool:
    return any(path == p or (p.endswith(".") and path.startswith(p)) for p in FREE_PREFIXES)


def _variant(cfg, **changes):
    return K.parse_config(K.nest({**cfg.values, **{k.replace("__", "."): v for k, v in changes.items()}}))


# ---------------------------------------------------------------------------------------------- the allow-list (fast)

def test_the_reuse_allow_list_is_exactly_the_registered_groups():
    """Widening the list is a model decision: a knob that feeds the matched learners' X, y, masks or parameters must
    never be free. Mutation: add `features.weather` to the list -> red."""
    assert set(K.REUSE_FREE_KNOBS) == {p for p in K.KNOBS if _is_free(p)}
    for must_block in ("features.weather", "features.adsb", "matched.lightgbm.learning_rate", "matched.catboost.depth",
                       "matched.learner", "matched.target", "matched.es_months", "data.train_months", "data.raw_dir",
                       "run.seeds", "run.threads", "airports.codes", "lanes.matched",
                       "unmatched.learner"):
        assert must_block in K.KNOBS and must_block not in K.REUSE_FREE_KNOBS, must_block


def test_reuse_compares_weather_inputs_only_for_the_configured_training_months():
    """Finding 2 (FP-W3 item 2, HIGH): fit_inputs hashes EVERY file under data.weather_dir (engine.py ~102-103), so
    a new period's weather file sitting in the directory (added after the donor was fitted) must not force a
    refit -- stand_ab.load_weather's as-of join never reaches a month outside data.train_months for an earlier
    training row's anchor. _learner_input_diff compares weather/<station>_<YYYY>-<MM>.csv keys only for the
    configured training months; raw/, orl_routes and adsb/train.parquet keys are always compared, regardless of
    month. Mutation: drop the month filter (compare every weather/ key) -> the first assertion goes red; make the
    function never compare weather/ at all -> the second assertion goes red."""
    from prc.pipeline.fm1 import engine as G
    donor_inputs = {"raw/a.parquet": "h1", "weather/EDDF_2025-01.csv": "wA", "weather/EDDF_2025-02.csv": "wB",
                    "orl_routes": "o1", "adsb/train.parquet": "t1"}
    # a weather file of a month the fit does not train on (2025-03) exists only on the live side: ignored
    live_new_month = {**donor_inputs, "weather/EDDF_2025-03.csv": "wC"}
    assert G._learner_input_diff(live_new_month, donor_inputs, train_year_months={"2025-01", "2025-02"}) == []
    # a changed TRAINING month's weather file is still refused
    live_changed_training = {**donor_inputs, "weather/EDDF_2025-01.csv": "wA_changed"}
    assert G._learner_input_diff(live_changed_training, donor_inputs, train_year_months={"2025-01", "2025-02"}) == \
        ["weather/EDDF_2025-01.csv"]
    # the other learner inputs are always compared, regardless of data.train_months
    live_raw_changed = {**donor_inputs, "raw/a.parquet": "h1_changed"}
    assert G._learner_input_diff(live_raw_changed, donor_inputs, train_year_months={"2025-01", "2025-02"}) == ["raw/a.parquet"]
    # identical inputs on both sides: nothing differs
    assert G._learner_input_diff(donor_inputs, donor_inputs, train_year_months={"2025-01", "2025-02"}) == []


def test_regression_a_new_period_weather_file_does_not_block_reuse_after_a_full_training_year():
    """Regression (coordinator review 2026-10-06 of FP-W3 item 2): the first fix compared a weather file when its
    MONTH NUMBER was in data.train_months; the final configs train on months 1-12 of 2025, so an October 2026 file
    (month 10) was still compared and `fit --reuse-from` still refused a new period. The key is the training files'
    YEAR-month. Mutation: compare by month number only -> red."""
    from prc.pipeline.fm1 import engine as G
    full_year = {f"2025-{m:02d}" for m in range(1, 13)}
    donor_inputs = {"raw/a.parquet": "h1", **{f"weather/EDDF_2025-{m:02d}.csv": f"w{m}" for m in range(1, 13)}}
    live = {**donor_inputs, "weather/EDDF_2026-10.csv": "w_new"}
    assert G._learner_input_diff(live, donor_inputs, train_year_months=full_year) == []
    changed = {**donor_inputs, "weather/EDDF_2025-10.csv": "w10_changed"}
    assert G._learner_input_diff(changed, donor_inputs, train_year_months=full_year) == ["weather/EDDF_2025-10.csv"]


def test_train_year_months_come_from_the_training_file_names():
    """The year-months a fit trains on are read from its training files' names (engine.TRAIN_FILE_RE, the same
    pattern files_by_month enforces); a name outside that pattern is refused by name. Mutation: return month
    numbers only, or drop the guard -> red."""
    from prc.pipeline.fm1 import engine as G
    files = {1: pathlib.Path("raw/training_2025-01-01_2025-02-01.parquet"),
             12: pathlib.Path("raw/training_2025-12-01_2026-01-01.parquet")}
    assert G.train_year_months(files) == {"2025-01", "2025-12"}
    with pytest.raises(E.SchemaError, match=re.escape("odd.parquet")):
        G.train_year_months({1: pathlib.Path("raw/odd.parquet")})


def test_reuse_input_diff_refuses_an_unparseable_weather_key():
    """A weather/ input key that does not match weather/<station>_<YYYY>-<MM>.csv is refused by name rather than
    silently treated as always-compared or never-compared -- fit_inputs (engine.py _dir_hashes) only ever writes
    keys in that shape, so a differently-shaped key means the naming convention changed underneath this function.
    Mutation: drop the regex-match guard -> no SchemaError raised -> red."""
    from prc.pipeline.fm1 import engine as G
    with pytest.raises(E.SchemaError, match=re.escape("weather/bad_name.csv")):
        G._learner_input_diff({"weather/bad_name.csv": "x"}, {}, train_year_months={"2025-01"})


def test_eobt_clip_and_ldi_apply_read_the_scored_file_directly(monkeypatch):
    """FP-W1 item 8 addendum: positive control for test_fit_never_reads_the_scored_or_template_file below. Both
    eobt_clip.apply and ldi.apply call pd.read_parquet(cfg.path('data.scored_file'), ...) unconditionally, before
    touching any other argument -- so a trap on that exact path is not inert. (unmatched.check_e2_context, the
    third known reader, only compares the resolved path / existence; it never calls read_parquet, so it is not
    part of this control -- and it is reachable only from unmatched.serve, itself only from engine.predict.)
    Mutation: remove either read_parquet call -> the matching pytest.raises block finds nothing to catch -> red."""
    from prc.pipeline.fm1 import eobt_clip as EC
    from prc.pipeline.fm1 import ldi as LDI
    cfg = K.parse_config({})
    scored_path = cfg.path("data.scored_file").resolve()

    def trap(path, *a, **k):
        assert pathlib.Path(path).resolve() == scored_path
        raise AssertionError(f"read {path}")
    monkeypatch.setattr(pd, "read_parquet", trap)
    with pytest.raises(AssertionError, match=re.escape(str(scored_path))):
        EC.apply(cfg, rows=None)
    with pytest.raises(AssertionError, match=re.escape(str(scored_path))):
        LDI.apply(cfg, matched=None, rows=None, ledger=None, stage=None)


def test_reuse_blockers_names_every_differing_knob_outside_the_allow_list():
    base = K.parse_config({})
    assert K.reuse_blockers(base, base) == []
    free = K.parse_config({}, overrides=["modules.storm.mode=anchor", "matched.blend.catboost_weight=0",
                                         "adsb.support_gate.rule=coverage_ratio", "run.artifacts_dir=elsewhere"])
    assert K.reuse_blockers(free, base) == []
    other = K.parse_config({}, overrides=["features.weather=false", "run.threads=12", "modules.storm.mode=anchor"])
    assert K.reuse_blockers(other, base) == ["features.weather", "run.threads"]
    assert K.reuse_blockers(base, other) == ["features.weather", "run.threads"]


# ---------------------------------------------------------------------------------------------- end to end (slow)

@pytest.fixture(scope="module")
def world(e2e_world):
    return e2e_world[0]


@pytest.fixture(scope="module")
def cfg(world, fm1_world_mod, tmp_path_factory):
    raw = fm1_world_mod.config_raw(world)
    raw["run"]["artifacts_dir"] = str(tmp_path_factory.mktemp("fm1_reuse_artifacts"))
    return K.parse_config(raw)


@pytest.fixture(scope="module")
def donor(cfg):
    return api.fit(cfg, log=QUIET)


@pytest.fixture(scope="module")
def scored(world):
    return pd.read_parquet(world / "raw" / "scored.parquet")


VARIANTS = {
    "blend_weight": {"matched__blend__catboost_weight": 0.25},
    "blend_off": {"matched__blend__catboost_weight": 0.0},
    "coverage_gate": {"adsb__support_gate__rule": "coverage_ratio"},
    "mtx": {"run__purpose": "scored_period", "calibration__mtx__lambda_rule": "full",
            "calibration__mtx__published_csv": str(K.ROOT / "data/mtx/raw/taxi_out_additional_time_2025.csv")},
    "stack_off": {"adsb__stack__enabled": False},
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_a_derived_artefact_predicts_bit_identically_to_a_scratch_fit(name, cfg, donor, scored):
    """THE contract. Each variant differs from the donor in allow-listed knobs only; its derived artefact (donor's
    learners, everything else recomputed) must predict exactly what a full fit of the variant config predicts — every
    column. Mutation: reuse the donor's unmatched lane / MTX instead of recomputing -> the mtx variant goes red;
    keep the donor's CatBoost when the variant has no blend -> blend_off stays equal only if predict ignores it (it
    does: the stage plan decides) — covered by the stage-plan assertion below."""
    v = _variant(cfg, **VARIANTS[name])
    scratch = api.fit(v, save=False, log=QUIET)
    derived = api.fit(v, save=False, log=QUIET, reuse_from=donor)
    pd.testing.assert_frame_equal(api.predict(derived, scored, log=QUIET), api.predict(scratch, scored, log=QUIET),
                                  check_exact=True)
    assert derived.config.hash == v.hash != donor.config.hash
    assert derived.fit_report["reused_learners_from"] == donor.key
    assert "reused_learners_from" not in scratch.fit_report
    assert derived.fit_report["stages"] == scratch.fit_report["stages"]
    assert (derived.catboost is None) == (scratch.catboost is None)


@pytest.mark.slow
def test_a_derived_fit_never_trains_a_matched_learner(cfg, donor, scored, monkeypatch):
    """The point of the mechanism: no LightGBM / CatBoost training. Mutation: fall through to base.fit -> red."""
    def boom(self, *a, **k):
        raise AssertionError(f"{type(self).__name__}.fit called during a reuse fit")
    monkeypatch.setattr(L.LightGBMLearner, "fit", boom)
    monkeypatch.setattr(L.CatBoostWorkerLearner, "fit", boom)
    v = _variant(cfg, **VARIANTS["blend_weight"])
    derived = api.fit(v, save=False, log=QUIET, reuse_from=donor)
    assert derived.base is donor.base and derived.catboost is donor.catboost
    assert len(api.predict(derived, scored, log=QUIET)) == int((scored.PHASE_mvt == "DEP").sum())   # one row per departure


@pytest.mark.slow
def test_a_derived_artefact_saves_loads_and_accepts_a_directory_as_donor(cfg, donor, scored, tmp_path):
    v = _variant(cfg, **{**VARIANTS["blend_weight"], "run__artifacts_dir": str(tmp_path / "variants")})
    derived = api.fit(v, log=QUIET, reuse_from=donor.directory)          # donor given as its directory
    assert derived.directory.parent == tmp_path / "variants" and derived.directory.name.startswith(v.short_hash)
    back = api.load_artifact(derived.directory)
    assert back.fit_report["reused_learners_from"] == donor.key
    pd.testing.assert_frame_equal(api.predict(back, scored, log=QUIET), api.predict(derived, scored, log=QUIET),
                                  check_exact=True)


@pytest.mark.slow
def test_a_saved_artefact_holds_no_file_dated_before_its_own_save(donor):
    """Every file of a saved artefact is written at save time. A copy that keeps its source's access and modification
    times looks days old the moment it exists, and a cleaner that removes temp files by access time deletes it: on
    2026-10-05 macOS's daily temp cleaner (03:35) removed this module's donor stack/ copies from the pytest temp tree
    while the suite ran (BC-44). The stack models this world copies were written on 2026-09-26. Fails when save()
    stores a file whose access or modification time is more than an hour older than the artefact's manifest, which is
    written last."""
    floor = (donor.directory / "manifest.json").stat().st_mtime - 3600.0
    stale = sorted(str(f.relative_to(donor.directory)) for f in donor.directory.rglob("*")
                   if f.is_file() and min(f.stat().st_atime, f.stat().st_mtime) < floor)
    assert stale == []


@pytest.mark.slow
def test_reuse_refuses_a_knob_outside_the_allow_list(cfg, donor):
    with pytest.raises(E.ReuseError, match=r"knob\(s\) outside the reuse allow-list differ.*matched\.lightgbm\.num_leaves"):
        api.fit(_variant(cfg, matched__lightgbm__num_leaves=7), save=False, log=QUIET, reuse_from=donor)
    with pytest.raises(E.ReuseError, match=r"run\.seeds"):
        api.fit(_variant(cfg, run__seeds=[5]), save=False, log=QUIET, reuse_from=donor)


@pytest.mark.slow
def test_reuse_refuses_different_input_data_encodings_or_design(cfg, donor):
    v = _variant(cfg, **VARIANTS["blend_weight"])
    # the inputs the matched learners were fitted on: raw training files, weather, routes, the ADS-B tables. A donor
    # whose record of ANY of them differs is refused; inputs of the free stages (stack models, MTX csv, E2 rules) are not
    # compared (they change with the allow-listed knobs).
    for key in ("raw/", "weather/", "orl_routes", "adsb/train.parquet"):
        name = next(k for k in sorted(donor.inputs) if k.startswith(key))
        other_data = dataclasses.replace(donor, inputs={**donor.inputs, name: "0" * 64})
        with pytest.raises(E.ReuseError, match=f"input data differ.*{name}"):
            api.fit(v, save=False, log=QUIET, reuse_from=other_data)
    stack_key = next(k for k in sorted(donor.inputs) if k.startswith("stack/"))
    free_only = dataclasses.replace(donor, inputs={**donor.inputs, stack_key: "0" * 64})
    assert api.fit(v, save=False, log=QUIET, reuse_from=free_only).base is donor.base
    # the serve table is not a learner input (fit reads only train.parquet; AFI): a donor whose serve table differs is free
    serve_only = dataclasses.replace(donor, inputs={**donor.inputs, "adsb/serve.parquet": "0" * 64})
    assert api.fit(v, save=False, log=QUIET, reuse_from=serve_only).base is donor.base
    e = donor.enc_matched
    te = next(iter(e.priors))
    other_enc = dataclasses.replace(donor, enc_matched=F.EncodingTables(
        smooth=e.smooth, priors={**e.priors, te: e.priors[te] + 1.0}, tables=e.tables, targets=e.targets))
    with pytest.raises(E.ReuseError, match="encodings differ"):
        api.fit(v, save=False, log=QUIET, reuse_from=other_enc)
    other_cols = dataclasses.replace(donor, design_columns=donor.design_columns[:-1])
    with pytest.raises(E.ReuseError, match="design columns differ"):
        api.fit(v, save=False, log=QUIET, reuse_from=other_cols)
    fr = donor.fit_report
    other_rows = dataclasses.replace(donor, fit_report={**fr, "matched": {**fr["matched"], "n_rows": fr["matched"]["n_rows"] + 1}})
    with pytest.raises(E.ReuseError, match="matched rows differ"):
        api.fit(v, save=False, log=QUIET, reuse_from=other_rows)
    other_es = dataclasses.replace(donor, fit_report={**fr, "matched": {**fr["matched"], "n_es": fr["matched"]["n_es"] + 1}})
    with pytest.raises(E.ReuseError, match="matched rows differ"):
        api.fit(v, save=False, log=QUIET, reuse_from=other_es)


@pytest.mark.slow
def test_fit_never_reads_the_scored_or_template_file(cfg, donor, tmp_path):
    """FP-W1 item 8 addendum; hardened for FP-W3 item 3 (review finding 4): the original trap caught only
    pd.read_parquet, but scripts/build_submission.py:75 reads with pq.read_table and other readers could use
    pq.ParquetFile or plain open() -- none of those would have tripped a pd.read_parquet-only trap. Point
    data.scored_file / data.template_file at paths that do not exist (both are reuse-free: FP-W1 item 8 /
    knobs._REUSE_FREE_PREFIXES, exactly as test_a_different_scored_or_template_file_is_reuse_free below relies on),
    so ANY reader of either path -- pandas, pyarrow, or a bare open() -- raises if engine.fit ever touches it; a
    from-scratch fit and a derived (reuse_from=donor) fit must both complete. The positive control above
    (test_eobt_clip_and_ldi_apply_read_the_scored_file_directly) proves the known scored-file readers are not
    inert, so a regression there would still be caught even though this test no longer traps read_parquet
    directly. Mutation: make engine.fit call EC.apply / LDI.apply / read data.template_file -> FileNotFoundError
    here -> red."""
    v = _variant(cfg, data__scored_file=str(tmp_path / "does_not_exist_scored.parquet"),
                data__template_file=str(tmp_path / "does_not_exist_template.parquet"))
    api.fit(v, save=False, log=QUIET)
    api.fit(v, save=False, log=QUIET, reuse_from=donor)


@pytest.mark.slow
def test_a_different_scored_or_template_file_is_reuse_free(cfg, donor, tmp_path):
    """FP-W1 item 8 addendum: data.scored_file / data.template_file were not in the reuse allow-list, so pointing a
    variant at a new period's scored or template file forced a full refit although engine.fit never reads either
    (see test_fit_never_reads_the_scored_or_template_file above). Fails when either knob is removed from
    knobs._REUSE_FREE_PREFIXES (reuse_blockers would name it and the derived fit below would raise ReuseError
    instead of completing with the donor's learners)."""
    alt_scored, alt_template = tmp_path / "scored_alt.parquet", tmp_path / "template_alt.parquet"
    shutil.copy(cfg.path("data.scored_file"), alt_scored)       # same content, a different path (BC-44: plain copy)
    shutil.copy(cfg.path("data.template_file"), alt_template)
    v = _variant(cfg, data__scored_file=str(alt_scored), data__template_file=str(alt_template))
    assert K.reuse_blockers(v, cfg) == []
    derived = api.fit(v, save=False, log=QUIET, reuse_from=donor)
    assert derived.base is donor.base and derived.catboost is donor.catboost
    assert derived.fit_report["reused_learners_from"] == donor.key


@pytest.mark.slow
def test_reuse_refuses_a_variant_that_needs_a_learner_the_donor_lacks(cfg):
    """A donor fitted without CatBoost (no blend, LightGBM stack base) cannot serve a variant that blends."""
    lean = api.fit(_variant(cfg, matched__blend__catboost_weight=0.0, adsb__stack__f_model="lightgbm"), save=False, log=QUIET)
    assert lean.catboost is None
    with pytest.raises(E.ReuseError, match="needs the CatBoost learner; the donor has none"):
        api.fit(_variant(cfg, matched__blend__catboost_weight=0.5, adsb__stack__f_model="lightgbm"), save=False,
                log=QUIET, reuse_from=lean)


@pytest.mark.slow
def test_reuse_refuses_validation_splits_and_tiers(cfg, donor):
    """Reuse is for the full configured fit only: a protocol fold (split) or a screen tier trains different models."""
    from prc.pipeline.fm1 import engine as G
    v = _variant(cfg, **VARIANTS["blend_weight"])
    with pytest.raises(E.ReuseError, match="split / tier"):
        G.fit(v, save=False, log=QUIET, reuse_from=donor, split={"train_months": [1], "es_months": [1]})
    with pytest.raises(E.ReuseError, match="split / tier"):
        G.fit(v, save=False, log=QUIET, reuse_from=donor, tier_overrides={"learning_rate": 0.05})


def test_a_non_csv_file_in_the_weather_directory_never_blocks_reuse():
    """Review 2 finding 7: fit_inputs hashes every file under data.weather_dir, but the weather loader reads only *.csv;
    a `.DS_Store` or an interrupted fetch's `.part` raised SchemaError in the donor comparison although no learner can
    read it. Keys the loader never reads are skipped; a .csv whose name breaks the naming convention is still refused
    (the loader WOULD read it). Mutations: refuse every unparseable weather/ key again, or stop skipping sub-folder
    files -> red."""
    from prc.pipeline.fm1 import engine as G
    ym = {"2025-01"}
    donor = {"weather/EDDF_2025-01.csv": "w1"}
    live = {**donor, "weather/.DS_Store": "x", "weather/EDDF_2025-02.csv.part": "y",
            "weather/old/EDDF_2025-01.csv": "w_old"}            # a sub-folder: the top-level glob never reads it
    assert G._learner_input_diff(live, donor, train_year_months=ym) == []
    with pytest.raises(E.SchemaError, match=re.escape("weather/odd.csv")):
        G._learner_input_diff({**donor, "weather/odd.csv": "z"}, donor, train_year_months=ym)
