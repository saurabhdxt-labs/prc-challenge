"""The fit stage's two in-fold statistics must never see held-out targets.

`build_month` is covered by test_stand_features.py, but every quantity that reads the
TARGET is produced in the fit stage instead: the 24 smoothed target encodings and the
per-stand slack statistics. A dropped fold mask at either site lets held-out labels into
the features, and the permutation test in test_stand_features.py cannot catch it — it
excludes STAND_INFOLD precisely because those columns are meant to read the target.

The invariant tested here is the one that matters: perturb ONLY the held-out targets and
require every in-fold statistic to be unchanged.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("stand_ab", ROOT / "scripts" / "stand_ab.py")
stand_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stand_ab)


@pytest.fixture
def fold():
    rng = np.random.default_rng(0)
    n = 4_000
    tr = np.zeros(n, dtype=bool)
    tr[: n // 2] = True                      # first half trains, second half is held out
    keys = pd.DataFrame({k: rng.integers(0, 40, n).astype(str) for k in stand_ab.ENC_KEYS})
    keys["ars"] = rng.integers(0, 25, n).astype(str)
    y = rng.normal(900, 300, n)
    gapa = np.abs(rng.normal(600, 300, n))
    # slack = -(gapa + delta) is positive only where the airport clock is far EARLIER than
    # NM's, i.e. the witness rows. A fixture with delta ~ N(50, .) has no admissible slack
    # rows at all and exercises nothing - match the real sign structure.
    delta = rng.normal(-1200, 350, n)
    assert ((-(gapa + delta) > 0) & (-(gapa + delta) < 21600)).mean() > 0.5
    return tr, keys, y, delta, gapa


def test_encodings_ignore_held_out_targets(fold):
    """Perturbing held-out y/delta must not move any of the 24 encodings.

    Fails when `keys[tr_mask]` becomes `keys` at the call site. Rehearsed 2026-09-08:
    dropping the mask made all 24 columns move and this test went red.
    """
    tr, keys, y, delta, _ = fold
    a = stand_ab.infold_encodings(keys, y, delta, tr)
    y2, d2 = y.copy(), delta.copy()
    y2[~tr] += 10_000.0
    d2[~tr] -= 7_500.0
    b = stand_ab.infold_encodings(keys, y2, d2, tr)

    assert len(a) == 2 * len(stand_ab.ENC_KEYS) == 24, f"expected 24 encodings, got {len(a)}"
    moved = [k for k in a if not np.allclose(a[k], b[k], equal_nan=True)]
    assert not moved, f"these encodings read held-out targets: {moved}"

    # control: perturbing TRAINING targets must move them, or the test proves nothing
    y3 = y.copy()
    y3[tr] += 10_000.0
    c = stand_ab.infold_encodings(keys, y3, delta, tr)
    assert any(not np.allclose(a[k], c[k], equal_nan=True) for k in a), \
        "training-target perturbation changed nothing - the test is vacuous"


def test_slack_stats_ignore_held_out_targets(fold):
    """Per-stand slack statistics must be fitted on training rows only.

    slack = -(gapa + delta) reads the target. Fails when `tr_mask` is dropped from the
    admissibility mask. Rehearsed 2026-09-08: removing `tr_mask &` moved both columns red.
    """
    tr, keys, _, delta, gapa = fold
    ars = keys["ars"].to_numpy()
    m0, i0 = stand_ab.infold_slack_stats(gapa, delta, ars, tr)
    d2 = delta.copy()
    d2[~tr] += 5_000.0
    m1, i1 = stand_ab.infold_slack_stats(gapa, d2, ars, tr)
    assert np.allclose(m0, m1, equal_nan=True), "stand_slack_med reads held-out targets"
    assert np.allclose(i0, i1, equal_nan=True), "stand_slack_iqr reads held-out targets"

    # the control perturbation must change the statistics WITHOUT pushing slack out of the
    # admissible band - +5000 would make every training row inadmissible and merely trip the
    # empty-fold guard, which is a different assertion entirely
    d3 = delta.copy()
    d3[tr] -= 400.0
    m2, _ = stand_ab.infold_slack_stats(gapa, d3, ars, tr)
    assert not np.allclose(m0, m2, equal_nan=True), \
        "training-target perturbation changed nothing - the test is vacuous"


def test_slack_stats_refuse_an_empty_training_fold(fold):
    """An all-False mask must raise, not silently return the global fallback everywhere."""
    _, keys, _, delta, gapa = fold
    with pytest.raises(ValueError, match="no admissible training rows"):
        stand_ab.infold_slack_stats(gapa, delta, keys["ars"].to_numpy(),
                                    np.zeros(len(delta), dtype=bool))
