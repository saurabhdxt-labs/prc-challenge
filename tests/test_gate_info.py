"""Tests for scripts/gate_info.py (GID). Exact small cases."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gate_info as G  # noqa: E402


def test_auc_matches_sklearn_with_ties_and_drops_nan():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 500).astype(float)
    p = rng.random(500) < 0.3
    assert G.auc(s, p) == pytest.approx(roc_auc_score(p, s))
    s2 = s.copy()
    s2[:10] = np.nan
    assert G.auc(s2, p) == pytest.approx(roc_auc_score(p[10:], s[10:]))
    assert np.isnan(G.auc(s, np.zeros(500, bool)))


def test_top_precision_and_commit():
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.6])
    p = np.array([True, False, True, False, False, True])
    assert G.top_precision(s, p, 0.5) == pytest.approx(2 / 3)          # top 3: 0.9 T, 0.8 F, 0.7 T
    c = G.committed(s, p, 0.65)
    assert c == {"n_rows": 3, "precision": pytest.approx(2 / 3), "early_rows_covered": 2, "share_of_early": pytest.approx(2 / 3)}
    assert G.committed(s, p, 0.99)["n_rows"] == 0


def test_day_concentration():
    day = np.array(["a"] * 6 + ["b"] * 3 + ["c"] * 1)
    pos = np.array([True] * 6 + [True, True, False] + [True])
    assert G.day_concentration(day, pos, k=1) == pytest.approx(6 / 9)
    assert G.day_concentration(day, pos, k=2) == pytest.approx(8 / 9)
    assert np.isnan(G.day_concentration(day, np.zeros(10, bool)))
