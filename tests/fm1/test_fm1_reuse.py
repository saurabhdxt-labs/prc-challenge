"""Refit-free derivation of a variant artefact (plans/PREREG_bsv_board_selected_variants_2026_10_01.md, "Mechanism"):
`api.fit(config, reuse_from=<donor artefact>)` reuses the donor's fitted matched learners (the 8-10 h part of a full fit)
when every knob outside the registered allow-list equals the donor's and the input data are the same; frames, encodings,
the unmatched lane and MTX are recomputed, and the recomputed encodings / design columns must equal the donor's.

The contract under test: a derived artefact predicts bit-identically to a from-scratch fit of the same variant config;
anything that could make that false is refused by name."""
from __future__ import annotations

import dataclasses

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
                 "adsb.stack.", "adsb.support_gate.", "adsb.min_coverage")


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
                       "run.seeds", "run.threads", "adsb.table_dir", "airports.codes", "lanes.matched",
                       "unmatched.learner"):
        assert must_block in K.KNOBS and must_block not in K.REUSE_FREE_KNOBS, must_block


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
    assert len(api.predict(derived, scored, log=QUIET)) > 0


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
    for key in ("raw/", "weather/", "orl_routes", "adsb/"):
        name = next(k for k in sorted(donor.inputs) if k.startswith(key))
        other_data = dataclasses.replace(donor, inputs={**donor.inputs, name: "0" * 64})
        with pytest.raises(E.ReuseError, match=f"input data differ.*{name}"):
            api.fit(v, save=False, log=QUIET, reuse_from=other_data)
    stack_key = next(k for k in sorted(donor.inputs) if k.startswith("stack/"))
    free_only = dataclasses.replace(donor, inputs={**donor.inputs, stack_key: "0" * 64})
    assert api.fit(v, save=False, log=QUIET, reuse_from=free_only).base is donor.base
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
