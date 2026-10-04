"""engine.build_inputs on constructed inputs (no data world): what a submission's manifest records about the build's inputs.

The world-level checks (the manifest of a real build, the published MTX file) are in tests/fm1/test_fm1_api.py.
Seen red against the previous build_inputs (SchemaError: month 7 has 0 training files). Mutation BI4 (training_files
back) -> RED. Second review: BI5 the one-file-per-month filter dropped, BI6 the training-month filter dropped, BI7 the
tolerant hashing dropped -> each RED.
"""
from __future__ import annotations

import types

import pytest

from prc.pipeline.fm1 import engine as G
from prc.pipeline.fm1 import knobs as K


def test_a_build_without_every_training_file_names_the_missing_ones(tmp_path):
    """Protocol B review 2026-10-04 (MEDIUM-2): build_inputs called training_files, which refuses a missing month, so a
    build on a machine without all twelve training files failed AFTER the whole predict. A build does not read them: a
    missing month is now named among the inputs changed since the fit, and the build goes on."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for m in range(1, 13):
        nxt = "2026-01-01" if m == 12 else f"2025-{m + 1:02d}-01"
        (raw / f"training_2025-{m:02d}-01_{nxt}.parquet").write_bytes(f"month {m}".encode())
    cfg = K.parse_config(K.nest({"data.raw_dir": str(raw), "lanes.matched": False, "modules.rome.enabled": False}))
    art = types.SimpleNamespace(inputs=G.fit_inputs(cfg, G.training_files(cfg)))
    assert G.build_inputs(cfg, art)["changed_since_fit"] == []
    july = next(raw.glob("training_2025-07-01_*.parquet"))
    july.unlink()
    bi = G.build_inputs(cfg, art)
    assert bi["changed_since_fit"] == [f"raw/{july.name}"] and f"raw/{july.name}" not in bi["inputs"]
    assert len([k for k in bi["inputs"] if k.startswith("raw/")]) == 11


def _raw(tmp_path, months=range(1, 13)):
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    for m in months:
        nxt = "2026-01-01" if m == 12 else f"2025-{m + 1:02d}-01"
        (raw / f"training_2025-{m:02d}-01_{nxt}.parquet").write_bytes(f"month {m}".encode())
    return raw


def test_an_ambiguous_month_and_a_month_outside_the_training_set_are_not_hashed(tmp_path):
    """Two files for one month cannot be told apart, so neither is recorded and the month is named as changed; a file for
    a month the config does not train on is never an input (second review: both filters were untested)."""
    raw = _raw(tmp_path)
    months = [m for m in range(1, 13) if m != 5]                       # es_months 3 and 9 stay in
    cfg = K.parse_config(K.nest({"data.raw_dir": str(raw), "data.train_months": months, "lanes.matched": False,
                                 "modules.rome.enabled": False}))
    art = types.SimpleNamespace(inputs=G.fit_inputs(cfg, G.training_files(cfg)))
    may = next(raw.glob("training_2025-05-01_*.parquet")).name
    assert f"raw/{may}" not in G.build_inputs(cfg, art)["inputs"]       # month 5 is not trained on
    july = next(raw.glob("training_2025-07-01_*.parquet")).name
    (raw / "training_2025-07-01_2025-07-31.parquet").write_bytes(b"a second July file")
    bi = G.build_inputs(cfg, art)
    assert bi["changed_since_fit"] == [f"raw/{july}"]
    assert not any(k.startswith("raw/training_2025-07") for k in bi["inputs"])


def test_a_missing_mtx_reference_is_named_not_fatal_and_the_published_file_is_required(tmp_path):
    """The MTX reference is read at fit time only: a build without it names it as changed. The published file is read by
    the build itself, so a missing one is refused at once (before any prediction)."""
    raw = _raw(tmp_path)
    ref, pub = tmp_path / "ref.csv", tmp_path / "pub.csv"
    ref.write_text("reference\n")
    pub.write_text("published\n")
    cfg = K.parse_config(K.nest({"data.raw_dir": str(raw), "lanes.matched": False, "modules.rome.enabled": False,
                                 "run.purpose": "scored_period", "calibration.mtx.reference_csv": str(ref),
                                 "calibration.mtx.published_csv": str(pub)}))
    assert cfg.mtx_enabled
    art = types.SimpleNamespace(inputs=G.fit_inputs(cfg, G.training_files(cfg)))
    ref.unlink()
    bi = G.build_inputs(cfg, art)
    assert bi["changed_since_fit"] == ["mtx/reference"] and "mtx/published" in bi["inputs"]
    pub.unlink()
    with pytest.raises(FileNotFoundError):
        G.build_inputs(cfg, art)
