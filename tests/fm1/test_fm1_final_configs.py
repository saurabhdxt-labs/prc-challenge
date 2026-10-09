"""The two final-model configs resolve to exactly the configurations of the stored artefacts behind the board files.

configs/fm1.yaml holds every knob at its documented default (tests/fm1/test_fm1_knobs.py). The final runs were made from
it with command-line overrides, which a reader of the repository could not see. configs/fm1_v50.yaml (the forecaster,
board 251.4767) and configs/fm1_v55.yaml (the scored-period build, board 249.9168) write those overrides into files;
configs/fm1_v57.yaml (board 247.3187) adds the stand-exit stage, and configs/fm1_v57_tc.yaml is a candidate with no
stored artefact (v57 with the tail-cell rule off): for it the pinned hash is the configuration's own. The
config hash is computed over the resolved values, so each file must reproduce its artefact's recorded hash exactly and
must differ from fm1.yaml in exactly the recorded overrides. Parsing needs no data, so a fresh clone runs these tests.

Written first and seen red 2026-10-04 (the two files did not exist). Mutation rehearsal 2026-10-04 on the config files
(scratch copy, one at a time): 4 / 4 killed — FC1 v55 threads 12 -> 4, FC2 v55 adsb_fi true -> false, FC3 v55 seeds
[0, 1, 2] -> [0, 1] (a knob outside the overrides), FC4 v50 serving the adsb.fi table; each fails the hash test and the
exact-overrides test.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prc import api

ROOT = Path(__file__).resolve().parents[2]

#: (config file, config hash recorded in the stored artefact's config.json, the overrides recorded with it)
FINAL = {
    "v50": ("configs/fm1_v50.yaml", "2d29ad344b458eddb8286de25fc41308b13008943cdcb9872c3cee5fcc56f4d0", {
        "adsb.stack.f_model": "lgbcat",
        "adsb.stack.models_dir": "data/r12/r13_stack_full/models_lgbcat",
        "adsb.table_dir": "data/r12/tables_51348d3e1cb3",
        "modules.storm.mode": "anchor",
        "run.purpose": "forecaster",
        "run.threads": 12,
        "modules.ldi.enabled": True,
    }),
    "v55": ("configs/fm1_v55.yaml", "d811c69dc93e4ba214c54438bdfcc48474673c0eb1cc60e6201fcf8b9b5d3327", {
        "adsb.stack.f_model": "lgbcat",
        "adsb.stack.models_dir": "data/r12/r13_stack_full/models_lgbcat",
        "adsb.table_dir": "data/r12/tables_51348d3e1cb3_afi",
        "modules.storm.mode": "anchor",
        "run.threads": 12,
        "modules.ldi.enabled": True,
        "run.purpose": "scored_period",
        "adsb.sources.adsb_fi": True,
    }),
    "v57": ("configs/fm1_v57.yaml", "e3c1d6b473f422fcbbc0661d21bf8f7de13b7370552c6cc68fe426b45aff7979", {
        "adsb.stack.f_model": "lgbcat",
        "adsb.stack.models_dir": "data/r12/r13_stack_full/models_lgbcat",
        "adsb.table_dir": "data/r12/tables_51348d3e1cb3_afi",
        "modules.storm.mode": "anchor",
        "run.threads": 12,
        "modules.ldi.enabled": True,
        "run.purpose": "scored_period",
        "adsb.sources.adsb_fi": True,
        "modules.uad.enabled": True,
    }),
    #: THE final model (owner, 2026-10-06): v57 on open data only — adsb.fi off, so the open v51 table. Its artefact
    #: (derived from the v50 donor on 2026-10-06; board file v59) records exactly this config hash.
    "v57_open": ("configs/fm1_v57_open.yaml", "1dc07433357e5757c790d0a6e20ed7b5b22c056336e782f94cdc5cb7da165927", {
        "adsb.stack.f_model": "lgbcat",
        "adsb.stack.models_dir": "data/r12/r13_stack_full/models_lgbcat",
        "adsb.table_dir": "data/r12/tables_51348d3e1cb3",
        "modules.storm.mode": "anchor",
        "run.threads": 12,
        "modules.ldi.enabled": True,
        "run.purpose": "scored_period",
        "modules.uad.enabled": True,
    }),
    #: the final phase (2026-10-08): v57_open with the final-period paths (three REUSE_FREE knobs); artefact 2950c7552ed7-12679217
    "final": ("configs/fm1_final.yaml", "2950c7552ed7fcdc82f18e6c4df9a41ecbc45764e77768287b093216a79b8dbc", {
        "adsb.stack.f_model": "lgbcat",
        "adsb.stack.models_dir": "data/r12/r13_stack_full/models_lgbcat",
        "adsb.table_dir": "data/r12/tables_final",
        "modules.storm.mode": "anchor",
        "run.threads": 12,
        "modules.ldi.enabled": True,
        "run.purpose": "scored_period",
        "modules.uad.enabled": True,
        "modules.uad.table": "data/uad/final_table_a3/table.parquet",
        "data.template_file": "data/raw_final/final_submitting.parquet",
    }),
    #: a candidate, not behind a board file and without a stored artefact: the hash below is the config's own
    "v57_tc": ("configs/fm1_v57_tc.yaml", "9ba126281d66ccc6677ffe7dcd432d99a21470765e5385034356d20319064415", {
        "adsb.stack.f_model": "lgbcat",
        "adsb.stack.models_dir": "data/r12/r13_stack_full/models_lgbcat",
        "adsb.table_dir": "data/r12/tables_51348d3e1cb3_afi",
        "modules.storm.mode": "anchor",
        "run.threads": 12,
        "modules.ldi.enabled": True,
        "run.purpose": "scored_period",
        "adsb.sources.adsb_fi": True,
        "modules.uad.enabled": True,
        "modules.tail_cells.enabled": False,
    }),
}


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


@pytest.mark.parametrize("name", sorted(FINAL))
def test_the_final_config_resolves_to_the_stored_artefacts_hash(name):
    path, recorded_hash, _ = FINAL[name]
    assert api.load_config(ROOT / path).hash == recorded_hash


@pytest.mark.parametrize("name", sorted(FINAL))
def test_the_final_config_differs_from_the_defaults_in_exactly_the_recorded_overrides(name):
    path, _, overrides = FINAL[name]
    base = _flat(yaml.safe_load((ROOT / "configs" / "fm1.yaml").read_text()))
    final = _flat(yaml.safe_load((ROOT / path).read_text()))
    assert final.keys() == base.keys()
    changed = {k: final[k] for k in final if final[k] != base[k]}
    expected = {k: v for k, v in overrides.items() if base[k] != v}
    assert changed == expected


@pytest.mark.parametrize("name", sorted(FINAL))
def test_the_overrides_on_the_defaults_give_the_same_hash(name):
    """The recorded command line (fm1.yaml + --set ...) and the file are the same configuration."""
    _, recorded_hash, overrides = FINAL[name]
    sets = [f"{k}={'true' if v is True else 'false' if v is False else v}" for k, v in overrides.items()]
    assert api.load_config(ROOT / "configs" / "fm1.yaml", overrides=sets).hash == recorded_hash
