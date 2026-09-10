"""Priority 0 — the corrected encoder/stopping separation.

The defect this module exists to repair, verified in the shipped code on 2026-09-10:
`scripts/lgbm_fold.py::load_fold` calls `S.infold_encodings(d, y, dlt, tr)`, and
`fold_masks` puts the early-stopping months INSIDE `tr` ("ES months come out of the
TRAINING fold", lgbm_fold.py:657). So every early-stopping row's 24 target/delta
encodings were fitted on data that includes that row's own label. The outer Jan+Jul
holdout is unaffected — it is excluded from `tr` — so reported RMSEs and paired
intervals remain valid measurements; what is distorted is `best_iter` selection, and
therefore every cross-arm tree-count comparison.

These tests pin the repair. Each one is written to go RED against the incumbent
behaviour, and `test_incumbent_leak_is_real` asserts the defect itself so the repair
cannot be declared unnecessary later.
"""
import numpy as np
import pandas as pd
import pytest

from prc import encoding as E
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import stand_ab as S


KEYS = ["ADEP_mvt", "STAND_mvt"]


def _frame(n=600, seed=0):
    """Twelve months, two encoding keys, a target correlated with the key."""
    rng = np.random.default_rng(seed)
    month = np.repeat(np.arange(1, 13), n // 12)
    ap = np.array(["EDDF", "LIRF"])[rng.integers(0, 2, len(month))]
    stand = np.array([f"S{i:02d}" for i in rng.integers(0, 8, len(month))])
    keys = pd.DataFrame({"ADEP_mvt": ap, "STAND_mvt": stand})
    y = 900.0 + 300.0 * (ap == "LIRF") + rng.normal(0, 50, len(month))
    delta = 50.0 + 20.0 * (ap == "LIRF") + rng.normal(0, 10, len(month))
    return keys, y, delta, month


def _masks(month, holdout=(1, 7), es=(3, 9)):
    te = np.isin(month, holdout)
    tr = ~te
    es_m = tr & np.isin(month, es)
    fit = tr & ~es_m
    return te, tr, fit, es_m


def _sensitivity(fn, keys, y, delta, month, target_rows, watch_rows, col):
    """Does perturbing the labels of `target_rows` change `col` on `watch_rows`?"""
    a = fn(keys, y, delta, month)[col]
    y2, d2 = y.copy(), delta.copy()
    y2[target_rows] += 5000.0
    d2[target_rows] += 5000.0
    b = fn(keys, y2, d2, month)[col]
    return not np.allclose(a[watch_rows], b[watch_rows], equal_nan=True)


# ---------------------------------------------------------------- the defect itself

def test_incumbent_leak_is_real():
    """The shipped encoder lets an early-stopping row's own label into its own feature.

    Fails if `infold_encodings` is ever changed to exclude the ES months from its fit
    without this test being updated — i.e. it pins the BEFORE state of the repair.
    """
    keys, y, delta, month = _frame()
    te, tr, fit, es = _masks(month)
    inc = lambda k, yy, dd, mo: S.infold_encodings(k, yy, dd, tr, enc_keys=KEYS)
    assert _sensitivity(inc, keys, y, delta, month, es, es, "te_ADEP_mvt"), \
        "expected the incumbent to leak ES labels into ES features"
    assert not _sensitivity(inc, keys, y, delta, month, te, te, "te_ADEP_mvt"), \
        "the outer holdout must NOT be sensitive to its own labels"


# ---------------------------------------------------------------- stage 1 (early stop)

def test_stage1_es_rows_do_not_see_their_own_labels():
    """Fails when `separated_encodings` fits the ES-row encoder on anything but the fit months."""
    keys, y, delta, month = _frame()
    te, tr, fit, es = _masks(month)
    fn = lambda k, yy, dd, mo: E.separated_encodings(
        k, yy, dd, mo, fit_mask=fit, es_mask=es, tr_mask=tr, enc_keys=KEYS)["stage1"]
    assert not _sensitivity(fn, keys, y, delta, month, es, es, "te_ADEP_mvt")
    assert not _sensitivity(fn, keys, y, delta, month, es, es, "de_STAND_mvt")


def test_stage1_fit_rows_are_cross_fitted_by_month():
    """A fit row's encoding must not move when its OWN month's labels move.

    Fails when the cross-fit degenerates to a single encoder over all fit months.
    """
    keys, y, delta, month = _frame()
    te, tr, fit, es = _masks(month)
    fn = lambda k, yy, dd, mo: E.separated_encodings(
        k, yy, dd, mo, fit_mask=fit, es_mask=es, tr_mask=tr, enc_keys=KEYS)["stage1"]
    m5 = fit & (month == 5)
    assert not _sensitivity(fn, keys, y, delta, month, m5, m5, "te_ADEP_mvt")
    # but a DIFFERENT fit month's labels must still reach it, or nothing is being fitted
    m6 = fit & (month == 6)
    assert _sensitivity(fn, keys, y, delta, month, m6, m5, "te_ADEP_mvt")


# ---------------------------------------------------------------- stage 2 (refit)

def test_stage2_training_rows_are_cross_fitted_and_holdout_is_clean():
    """Fails when the refit encodings are fitted in-sample, or when holdout labels leak."""
    keys, y, delta, month = _frame()
    te, tr, fit, es = _masks(month)
    fn = lambda k, yy, dd, mo: E.separated_encodings(
        k, yy, dd, mo, fit_mask=fit, es_mask=es, tr_mask=tr, enc_keys=KEYS)["stage2"]
    m5 = tr & (month == 5)
    assert not _sensitivity(fn, keys, y, delta, month, m5, m5, "te_ADEP_mvt")
    assert not _sensitivity(fn, keys, y, delta, month, te, te, "te_ADEP_mvt")
    # the holdout must still be encoded from the training months
    m2 = tr & (month == 2)
    assert _sensitivity(fn, keys, y, delta, month, m2, te, "te_ADEP_mvt")


# ---------------------------------------------------------------- formula equivalence

def test_scope_change_only_the_formula_is_the_incumbent_smoothing():
    """Given identical fitting rows, the new encoder equals `_smooth_enc` exactly.

    Fails if the smoothing constant, the prior, or the unseen-level fallback drifts —
    the repair must be a change of SCOPE, not of formula, or the control is not a control.
    """
    keys, y, delta, month = _frame()
    te, tr, fit, es = _masks(month)
    got = E.fit_apply(keys, y, delta, fit_rows=tr, out_rows=te, enc_keys=KEYS)
    want = S.infold_encodings(keys, y, delta, tr, enc_keys=KEYS)
    for c in ("te_ADEP_mvt", "de_ADEP_mvt", "te_STAND_mvt", "de_STAND_mvt"):
        np.testing.assert_allclose(got[c][te], want[c][te], rtol=0, atol=0,
                                   err_msg=f"{c} drifted from the incumbent smoothing")


def test_unseen_level_falls_back_to_the_prior():
    """Fails when an unseen category yields NaN instead of the fitted prior."""
    keys, y, delta, month = _frame()
    te, tr, fit, es = _masks(month)
    keys = keys.copy()
    keys.loc[keys.index[te][0], "STAND_mvt"] = "NEVER_SEEN"
    out = E.fit_apply(keys, y, delta, fit_rows=tr, out_rows=te, enc_keys=KEYS)
    i = int(np.flatnonzero(te)[0])
    assert np.isfinite(out["te_STAND_mvt"][i])
    np.testing.assert_allclose(out["te_STAND_mvt"][i], float(np.asarray(y)[tr].mean()),
                               rtol=1e-6)


def test_counts_are_emitted_and_respect_the_fitting_scope():
    """Cross-fitting changes the support for rare levels; the count must say so.

    Fails when `n_` columns are absent or are computed over all rows rather than the
    rows the encoder was actually fitted on.
    """
    keys, y, delta, month = _frame()
    te, tr, fit, es = _masks(month)
    out = E.fit_apply(keys, y, delta, fit_rows=tr, out_rows=te, enc_keys=KEYS,
                      with_counts=True)
    assert "n_ADEP_mvt" in out
    n_lirf = out["n_ADEP_mvt"][te][np.asarray(keys.ADEP_mvt)[te] == "LIRF"]
    assert np.all(n_lirf == ((np.asarray(keys.ADEP_mvt) == "LIRF") & tr).sum())


def test_single_fit_month_cross_fit_falls_back_to_prior_not_nan():
    """With one fit month, LOMO leaves nothing to fit on. Fails if it returns NaN or raises."""
    keys, y, delta, month = _frame()
    fit = month == 5
    es = month == 3
    tr = fit | es
    out = E.separated_encodings(keys, y, delta, month, fit_mask=fit, es_mask=es,
                                tr_mask=tr, enc_keys=KEYS)["stage1"]
    assert np.all(np.isfinite(out["te_ADEP_mvt"][fit]))
    np.testing.assert_allclose(out["te_ADEP_mvt"][fit], float(np.asarray(y)[fit].mean()),
                               rtol=1e-6)


def test_nan_delta_on_the_encoding_set_is_refused():
    """Same guard as the incumbent: a NaN delta would poison the prior. Fails if silent."""
    keys, y, delta, month = _frame()
    te, tr, fit, es = _masks(month)
    delta = delta.copy()
    delta[np.flatnonzero(tr)[0]] = np.nan
    with pytest.raises(ValueError, match="NaN delta"):
        E.separated_encodings(keys, y, delta, month, fit_mask=fit, es_mask=es,
                              tr_mask=tr, enc_keys=KEYS)


def test_delta_mask_restricts_only_the_delta_encodings():
    """The all-rows design fits `de_` on matched rows and `te_` on every training row.

    Fails if `delta_mask` is ignored, or if it wrongly narrows the y encodings too.
    """
    keys, y, delta, month = _frame()
    te, tr, fit, es = _masks(month)
    dm = np.ones(len(y), bool)
    dm[np.flatnonzero(tr)[:50]] = False
    delta = delta.copy()
    delta[~dm] = np.nan
    out = E.separated_encodings(keys, y, delta, month, fit_mask=fit, es_mask=es,
                                tr_mask=tr, enc_keys=KEYS, delta_mask=dm)["stage2"]
    assert np.all(np.isfinite(out["de_ADEP_mvt"]))
    assert np.all(np.isfinite(out["te_ADEP_mvt"]))
    # the y encodings must still use EVERY training row, including the excluded-delta ones:
    # narrowing them too would silently shrink the support of te_ without saying so
    wide = E.fit_apply(keys, y, delta, fit_rows=tr, enc_keys=KEYS, delta_mask=dm,
                       with_counts=True)
    narrow = E.fit_apply(keys, y, delta, fit_rows=(tr & dm), enc_keys=KEYS, delta_mask=dm,
                         with_counts=True)
    assert wide["n_ADEP_mvt"].max() > narrow["n_ADEP_mvt"].max(), \
        "delta_mask wrongly narrowed the y encodings' support"
    # and te_ must equal what a clean-delta fit over the SAME wide row set produces
    clean = np.where(np.isnan(delta), 0.0, delta)
    np.testing.assert_allclose(
        E.fit_apply(keys, y, clean, fit_rows=tr, enc_keys=KEYS)["te_ADEP_mvt"],
        wide["te_ADEP_mvt"], rtol=0, atol=0)


# ---------------------------------------------------------------- airport-scoped keys

def test_airport_scoping_separates_names_reused_at_two_airports():
    """Fails if the composite drops the airport, or if the two airports collapse to one level.

    The defect: 55.3% of real rows carry a STAND_mvt name that also exists at another
    airport, and the shipped `ars` key is STAND|RUNWAY with no airport in it.
    """
    k = pd.DataFrame({"ADEP_mvt": ["EDDF", "LFPG", "EDDF"],
                      "STAND_mvt": ["210", "210", "211"],
                      "RUNWAY_mvt": ["08L", "08L", "08L"],
                      "AIRCRAFT_TYPE_mvt": ["B738"] * 3,
                      "ADES_mvt": ["EGLL"] * 3, "ars": ["210|08L", "210|08L", "211|08L"]})
    out, made = E.add_airport_scoped(k)
    assert made == ["ap_STAND_mvt", "ap_RUNWAY_mvt", "ap_AIRCRAFT_TYPE_mvt", "ap_ADES_mvt"]
    assert "ap_ars" not in made, ("`ars` is already <airport>|<stand>|<runway> in every cached "
                                 "row; scoping it again is redundant")
    assert out.ap_STAND_mvt.iloc[0] != out.ap_STAND_mvt.iloc[1], "airports collapsed"
    assert "ap_STAND_mvt" not in k.columns, "the caller's frame was mutated in place"


def test_scoped_keys_refuse_a_nul_separator_and_missing_columns():
    """Fails if a NUL separator is accepted (Arrow truncates on it) or a typo passes silently."""
    k = pd.DataFrame({"ADEP_mvt": ["EDDF"], "STAND_mvt": ["1"]})
    with pytest.raises(ValueError, match="separator"):
        E.add_airport_scoped(k, scope=("STAND_mvt",), sep="\x00")
    # assert the MESSAGE, not just the type: pandas raises a bare KeyError for a missing
    # column too, so a type-only assertion would pass with the explicit guard deleted.
    with pytest.raises(KeyError, match="asked for scoping but absent"):
        E.add_airport_scoped(k, scope=("NOPE",))
    with pytest.raises(KeyError, match="absent from the key frame"):
        E.add_airport_scoped(k.drop(columns=["ADEP_mvt"]), scope=("STAND_mvt",))


def test_null_key_cannot_swallow_the_airport_prefix():
    """A null stand must still yield two distinct levels at two airports. Fails on NaN joins."""
    k = pd.DataFrame({"ADEP_mvt": ["EDDF", "LFPG"], "STAND_mvt": [None, None]})
    out, _ = E.add_airport_scoped(k, scope=("STAND_mvt",))
    assert out.ap_STAND_mvt.nunique() == 2
    assert out.ap_STAND_mvt.iloc[0] == "EDDF|NA"


def test_screen_tier_mask_shape_is_accepted():
    """`sweep_masks` returns (stop, train, train, stop): fit == tr and es is OUTSIDE tr.

    Fails against the first version of `separated_encodings`, which required
    `fit | es == tr` and therefore raised on every screening-tier call — silently making the
    advertised ranking tier unrunnable for the repaired variants. Found by an adversarial
    review of this session's diff, 2026-09-10.
    """
    keys, y, delta, month = _frame()
    stop = month == 6
    tr = np.isin(month, (2, 3, 4, 5))
    fit = tr.copy()
    out = E.separated_encodings(keys, y, delta, month, fit_mask=fit, es_mask=stop,
                                tr_mask=tr, enc_keys=KEYS)
    for stage in ("stage1", "stage2"):
        assert np.all(np.isfinite(out[stage]["te_ADEP_mvt"]))
    # the stop month is encoded from the training months and must not see its own labels
    fn = lambda k, yy, dd, mo: E.separated_encodings(
        k, yy, dd, mo, fit_mask=fit, es_mask=stop, tr_mask=tr, enc_keys=KEYS)["stage1"]
    assert not _sensitivity(fn, keys, y, delta, month, stop, stop, "te_ADEP_mvt")


def test_a_fit_mask_outside_the_training_mask_is_still_refused():
    """Relaxing the es-inside-tr rule must not relax the fit-inside-tr one. Fails if both go."""
    keys, y, delta, month = _frame()
    tr = np.isin(month, (2, 3, 4, 5))
    fit = np.isin(month, (2, 3, 4, 5, 11))          # month 11 is not a training month
    with pytest.raises(ValueError, match="fit mask must lie inside"):
        E.separated_encodings(keys, y, delta, month, fit_mask=fit, es_mask=(month == 6),
                              tr_mask=tr, enc_keys=KEYS)
