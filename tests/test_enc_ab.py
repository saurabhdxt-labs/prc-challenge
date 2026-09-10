"""`scripts/enc_ab.py` — the three-variant encoding A/B harness.

The harness exists to answer two questions on one fold load: does removing the
early-stopping leak change model selection (Priority 0), and are airport-scoped keys
worth their columns (Priority 1). Its correctness risk is almost entirely WIRING —
a stale column left in the design matrix, or an encoder and a feature list that
disagree, would look exactly like a model difference. These tests target that.
"""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


EA = _load("enc_ab", ROOT / "scripts" / "enc_ab.py")
SC = _load("synthetic_caches", ROOT / "tests" / "synthetic_caches.py")


def _frame(n=480, seed=0):
    rng = np.random.default_rng(seed)
    month = np.repeat(np.arange(1, 13), n // 12)
    d = pd.DataFrame({k: rng.integers(0, 4, len(month)).astype(str)
                      for k in EA.S.ENC_KEYS})
    d["ADEP_mvt"] = np.array(["EDDF", "LIRF"])[rng.integers(0, 2, len(month))]
    y = 900.0 + rng.normal(0, 50, len(month))
    dlt = 50.0 + rng.normal(0, 10, len(month))
    te = np.isin(month, (1, 7))
    tr = ~te
    es = tr & np.isin(month, (3, 9))
    fit = tr & ~es
    return d, y, dlt, month, fit, es, tr


# ---------------------------------------------------------------- variant construction

def test_incumbent_uses_one_column_set_for_both_stages():
    """The control must be the SHIPPED behaviour: one encoder, used for stopping and refit.

    Fails if `incumbent` is ever quietly given separated stages — which would destroy the
    comparison by making every variant clean.
    """
    d, y, dlt, month, fit, es, tr = _frame()
    s1, s2, names = EA.encoding_columns("incumbent", d, y, dlt, month, fit, es, tr)
    assert s1 is s2, "the incumbent control must reuse one column set across both stages"
    for c in names:
        np.testing.assert_array_equal(s1[c], s2[c])
    want = EA.S.infold_encodings(d[EA.S.ENC_KEYS], y, dlt, tr)
    assert sorted(want) == names
    for c in names:
        np.testing.assert_allclose(s1[c], want[c], rtol=0, atol=0)


def test_separated_gives_the_stopping_rows_different_columns():
    """Fails if stage 1 and stage 2 are the same array — i.e. the repair is not wired in."""
    d, y, dlt, month, fit, es, tr = _frame()
    s1, s2, names = EA.encoding_columns("separated", d, y, dlt, month, fit, es, tr)
    assert s1 is not s2
    moved = [c for c in names if not np.allclose(s1[c][es], s2[c][es])]
    assert moved, "no encoding column differs on the early-stopping rows"


def test_scoped_adds_two_columns_per_scoped_key_and_keeps_the_originals():
    """Fails if scoping REPLACES the unscoped keys instead of adding to them.

    Replacing would confound the comparison: the arm must be the incumbent key set PLUS
    the composites, so a difference is attributable to the new information alone.
    """
    d, y, dlt, month, fit, es, tr = _frame()
    _, _, base = EA.encoding_columns("separated", d, y, dlt, month, fit, es, tr)
    _, _, scoped = EA.encoding_columns("scoped", d, y, dlt, month, fit, es, tr)
    assert set(base) < set(scoped), "scoped must be a strict superset of separated"
    assert len(scoped) - len(base) == 2 * len(EA.E.SCOPED_DEFAULT)
    for k in EA.E.SCOPED_DEFAULT:
        assert f"te_ap_{k}" in scoped and f"de_ap_{k}" in scoped


def test_unknown_variant_is_refused():
    """Fails if a typo silently falls through to a default arm."""
    d, y, dlt, month, fit, es, tr = _frame()
    with pytest.raises(ValueError, match="unknown variant"):
        EA.encoding_columns("scopped", d, y, dlt, month, fit, es, tr)


# ---------------------------------------------------------------- the column swap

def test_write_enc_overwrites_exactly_the_named_columns():
    """Fails if the swap writes the wrong column index, or leaves a stale one behind."""
    feats = ["a", "te_x", "b", "de_x"]
    X = np.zeros((5, 4), dtype=np.float32)
    X[:, 0] = 7.0
    X[:, 2] = 9.0
    n = EA.write_enc(X, feats, {"te_x": np.arange(5.0), "de_x": np.full(5, 3.0)})
    assert n == 2
    np.testing.assert_array_equal(X[:, 0], np.full(5, 7.0))    # untouched
    np.testing.assert_array_equal(X[:, 2], np.full(5, 9.0))    # untouched
    np.testing.assert_array_equal(X[:, 1], np.arange(5.0))
    np.testing.assert_array_equal(X[:, 3], np.full(5, 3.0))


def test_write_enc_refuses_a_column_the_matrix_does_not_have():
    """A silent skip would leave stage-2 values in X during early stopping. Fails if silent."""
    X = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(KeyError, match="absent from the design matrix"):
        EA.write_enc(X, ["a", "te_x"], {"te_x": np.zeros(3), "te_missing": np.zeros(3)})


def test_write_enc_round_trips_between_the_two_stages():
    """Swapping to stage 1 and back must restore stage 2 exactly.

    This is the property the harness relies on to serve both stages from one matrix; if it
    fails, the refit trains on stopping-stage columns.
    """
    feats = ["a", "te_x"]
    X = np.zeros((6, 2), dtype=np.float32)
    s1 = {"te_x": np.arange(6.0)}
    s2 = {"te_x": np.arange(6.0) * 10}
    EA.write_enc(X, feats, s2)
    keep = X[:, 1].copy()
    EA.write_enc(X, feats, s1)
    assert not np.allclose(X[:, 1], keep)
    EA.write_enc(X, feats, s2)
    np.testing.assert_array_equal(X[:, 1], keep)


# ---------------------------------------------------------------- end to end

@pytest.mark.parametrize("variants", ["incumbent,separated", "scoped"])
def test_smoke_runs_end_to_end_on_synthetic_caches(tmp_path, monkeypatch, variants):
    """The whole path on three synthetic months. Proves the code runs; NO magnitude is read.

    Fails on any wiring error between the encoder, the feature list and the design matrix —
    which is the failure mode this harness is most exposed to.
    """
    stand, queue = SC.synthetic_caches(tmp_path / "data", (1, 2, 3), n=600)
    monkeypatch.setattr(EA.LF, "CACHE", stand)
    monkeypatch.setattr(EA.LF, "QCACHE", queue)
    out = tmp_path / "enc_ab.json"
    assert EA.main(["--full", "--smoke", "--variants", variants, "--out", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert rec["ranking_only"] is True, "a smoke record must mark itself as ranking-only"
    for v in variants.split(","):
        r = rec["results"][v]
        assert r["best_iter"] >= 1 and r["n_ref"] >= 1
        assert np.isfinite(r["holdout_rmse"]) and r["holdout_rmse"] > 0
        assert r["n_enc"] == len(EA.S.ENC_KEYS) * 2 + (
            2 * len(EA.E.SCOPED_DEFAULT) if v == "scoped" else 0)


def test_the_stage_swap_is_observable_and_enforced(tmp_path, monkeypatch):
    """The stopping fit and the refit must see DIFFERENT encoding columns under `separated`.

    Fails when `write_enc(X, feats, s1)` is dropped before early stopping, or when the stage-2
    write is dropped before the refit — the two wiring errors that would silently make the
    repair a no-op while every other test still passes.
    """
    stand, queue = SC.synthetic_caches(tmp_path / "data", (1, 2, 3), n=600)
    monkeypatch.setattr(EA.LF, "CACHE", stand)
    monkeypatch.setattr(EA.LF, "QCACHE", queue)
    out = tmp_path / "swap.json"
    assert EA.main(["--full", "--smoke", "--variants", "incumbent,separated",
                    "--out", str(out)]) == 0
    rec = json.loads(out.read_text())["results"]
    inc, sep = rec["incumbent"], rec["separated"]
    assert inc["es_enc_checksum"] == inc["refit_enc_checksum"], \
        "the control must use ONE column set across both stages"
    assert sep["es_enc_checksum"] != sep["refit_enc_checksum"], \
        "the repaired arm must early-stop and refit on different encodings"


def test_screen_masks_hold_out_a_month_no_arm_ever_sees():
    """The screening tier must produce an INDEPENDENT score, not a stopping metric.

    `lgbm_fold.sweep_masks` returns (stop, train, train, stop) — its holdout IS its stopping
    set. Ranking the leaking encoder against the repaired one on a stopping metric is invalid
    by construction: the leaking arm's stopping rows carry their own labels in their features,
    so it wins whatever the truth is. Fails if the score month ever overlaps fit or stop.
    """
    month = np.repeat(np.arange(1, 13), 5)
    te, tr, fit, es = EA.screen_masks(month)
    assert not (te & tr).any(), "the score month leaked into training"
    assert not (te & fit).any() and not (te & es).any()
    assert ((fit | es) == tr).all(), "fit | es must be exactly the training mask"
    assert set(month[fit]) == set(EA.SCREEN["fit"])
    assert set(month[es]) == {EA.SCREEN["stop"]}
    assert set(month[te]) == {EA.SCREEN["score"]}


def test_screen_masks_refuse_an_overlapping_or_absent_score_month():
    """Fails if a mis-set SCREEN dict silently scores on a month the model trained on."""
    month = np.repeat(np.arange(1, 13), 5)
    with pytest.raises(ValueError, match="overlaps the fit or stopping"):
        EA.screen_masks(month, cfg=dict(fit=(2, 3, 4, 5), stop=6, score=5))
    with pytest.raises(ValueError, match="absent from the loaded months"):
        EA.screen_masks(month[month < 7], cfg=dict(fit=(2, 3, 4, 5), stop=6, score=8))


def test_scoped_default_excludes_ars_because_it_is_already_airport_scoped():
    """`ars` is `<airport>|<stand>|<runway>` in every cached row (stand_ab.py:340).

    Fails if `ars` is put back into SCOPED_DEFAULT, which would emit "LFPG|LFPG|I18|27L" —
    a redundant encoding of a key that is already scoped. An earlier version of this module
    included it, and reports/PRIORITY1_IDENTITY_SCREEN.md asserted the opposite of the truth.
    """
    assert "ars" not in EA.E.SCOPED_DEFAULT
    assert set(EA.E.SCOPED_DEFAULT) == {"STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
                                        "ADES_mvt"}
