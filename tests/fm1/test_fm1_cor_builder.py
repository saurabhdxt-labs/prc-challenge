"""scripts/analysis/cor_corrector.py's real feature path (FM1's build_frames) on the real-data-shaped world.

The rest of COR is tested on constructed rows in tests/test_cor_corrector.py.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("cor_corrector", ROOT / "research" / "analysis" / "cor_corrector.py")
C = importlib.util.module_from_spec(_spec)
sys.modules["cor_corrector"] = C
_spec.loader.exec_module(C)


@pytest.mark.slow
def test_the_fm1_builder_returns_label_free_raw_features_for_every_matched_row(e2e_world, fm1_world_mod):
    """On the real-data-shaped world (months 1 and 2): one row per matched departure, unique ids, finite labels, the
    raw design columns present, no encoding column, the encoding keys carried for COR-E."""
    from prc.pipeline.fm1 import knobs as K
    cfg = K.parse_config(fm1_world_mod.config_raw(e2e_world[0]))
    builder, cols, keys = C.fm1_builder(cfg, log=lambda *a: None)
    f = builder((1, 2))
    assert len(f) > 0 and not f.MVT_ID_mvt.duplicated().any() and np.isfinite(f.y).all()
    assert set(cols) <= set(f.columns) and not any(c.startswith(("te_", "de_")) for c in cols)
    assert set(keys) <= set(f.columns) and "delta" in f.columns
