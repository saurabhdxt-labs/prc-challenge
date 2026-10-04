"""engine.build_inputs on constructed inputs (no data world): what a submission's manifest records about the build's inputs.

The world-level checks (the manifest of a real build, the published MTX file) are in tests/fm1/test_fm1_api.py.
Seen red against the previous build_inputs (SchemaError: month 7 has 0 training files). Mutation BI4 (training_files
back) -> RED.
"""
from __future__ import annotations

import types

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
