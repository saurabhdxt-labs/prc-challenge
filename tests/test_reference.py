"""`prc/reference.py` — the reusable baseline reference.

The danger this module carries is that a WRONG reuse is silent: a stored prediction vector
paired against a different row set, a different scale, or a different feature order looks
exactly like a model difference and would be believed. This repository has already paid for
that shape of error twice in one night (a floored arm compared against an unfloored one; a
delta-scale record read as taxi times). So every test here targets a mismatch that must raise.
"""
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prc import reference as R  # noqa: E402


def _meta(**over):
    m = dict(feats=["a", "b", "te_x"], params={"num_leaves": 255, "learning_rate": 0.01},
             seeds=[0, 1, 2], n_ref=27155, months=list(range(1, 13)), split="fold_masks",
             variant="separated", scale="taxi_time", rows="abc123", n_rows=5)
    m.update(over)
    return m


def test_round_trip_returns_the_same_predictions(tmp_path):
    """Fails if the payload is written or read on the wrong key or dtype."""
    p = tmp_path / "x.npz"
    preds = np.array([1.0, 2.5, 3.0, 4.0, 5.0])
    R.save(p, preds, _meta())
    got, stored = R.load(p, _meta())
    np.testing.assert_allclose(got, preds)
    assert stored["variant"] == "separated" and stored["n_rows"] == 5


@pytest.mark.parametrize("field,bad", [
    ("scale", "delta"),                       # the -109 s error's exact shape
    ("rows", "different_checksum"),           # a different or reordered holdout
    ("variant", "incumbent"),                 # the leaking encoder's predictions
    ("feats", ["b", "a", "te_x"]),            # same features, different ORDER
    ("params", {"num_leaves": 127, "learning_rate": 0.01}),
    ("seeds", [0]),
    ("n_ref", 21948),
    ("months", [1, 2, 3]),
    ("split", "sweep_masks"),
])
def test_every_mismatch_raises_and_names_the_field(field, bad, tmp_path):
    """One parametrized case per field that must block reuse.

    Fails if any field is dropped from the comparison list — which is how a silent wrong
    reuse would get in. `feats` order matters because the design matrix is positional.
    """
    p = tmp_path / "ref.npz"
    R.save(p, np.arange(5.0), _meta())
    with pytest.raises(ValueError, match=f"{field!r} differs"):
        R.load(p, _meta(**{field: bad}))


def test_a_reference_of_the_wrong_LENGTH_is_refused(tmp_path):
    """A 5-row reference must not be handed to a caller expecting 99,999 rows.

    Fails when the n_rows check is disabled — which it silently was: the original guard read
    `len(meta.get("rows_index", preds))`, and "rows_index" is never present in any meta, so it
    compared len(preds) against itself. Found by an adversarial review, 2026-09-10.
    """
    p = tmp_path / "len.npz"
    R.save(p, np.arange(5.0), _meta())
    with pytest.raises(ValueError, match="but the caller expects"):
        R.load(p, _meta(n_rows=99999))


def test_row_checksum_is_order_sensitive_and_rejects_non_finite():
    """A reordered holdout must not reuse a reference. Fails if the hash sorts its input."""
    a = R.row_checksum([1, 2, 3, 4])
    assert a != R.row_checksum([1, 2, 4, 3]), "the checksum is order-insensitive"
    assert a == R.row_checksum([1, 2, 3, 4])
    with pytest.raises(ValueError, match="non-finite MVT_ID"):
        R.row_checksum([1, np.nan, 3])


def test_an_unknown_scale_is_refused():
    """Only the two scales this repo actually uses. Fails if a typo passes as a new scale."""
    with pytest.raises(ValueError, match="scale must be one of"):
        R.signature(_meta(scale="taxitime"))


def test_missing_metadata_is_refused_rather_than_defaulted():
    """A defaulted field would silently widen what counts as a match."""
    m = _meta()
    del m["n_ref"]
    # assert the MESSAGE: a bare dict comprehension raises KeyError('n_ref') too, so a
    # type-and-key-only assertion passes with the explicit guard deleted.
    with pytest.raises(KeyError, match="reference metadata is missing"):
        R.signature(m)


def test_non_finite_predictions_are_refused_at_save_time(tmp_path):
    """A NaN reference would poison every later paired comparison. Fails if saved silently.

    Writes under tmp_path, never the repo root: when this test's guard was mutated away during
    the rehearsal it actually created the file, leaving a stray artifact in the working tree.
    """
    target = tmp_path / "never.npz"
    with pytest.raises(ValueError, match="non-finite predictions"):
        R.save(target, np.array([1.0, np.nan]), _meta(n_rows=2))
    assert not target.exists(), "a refused save must not leave a file behind"


def test_absent_reference_returns_none_only_when_not_strict(tmp_path):
    """Absence is recoverable; a MISMATCH never is. Fails if strict=False also relaxes fields."""
    p = tmp_path / "missing.npz"
    assert R.load(p, _meta(), strict=False) is None
    with pytest.raises(FileNotFoundError):
        R.load(p, _meta())
    R.save(p, np.arange(5.0), _meta())
    with pytest.raises(ValueError, match="'scale' differs"):
        R.load(p, _meta(scale="delta"), strict=False)


def test_signature_ignores_fields_outside_the_required_set():
    """Extra metadata must not change the signature, or reuse never fires.

    Callers pass fields the match does not depend on — `n_rows`, `holdout_rmse`,
    `reused_from`. If `signature` were ever changed to hash the whole `meta` dict instead of
    projecting onto `required`, those would perturb the hash and every lookup would miss.

    (This replaces an earlier test that compared a dict against its own reversed copy. That
    test was VACUOUS: signature() projects onto a fixed tuple, so insertion order can never
    matter and no mutation could make it fail. Found by mutation rehearsal, 2026-09-10.)
    """
    base = R.signature(_meta())
    assert R.signature(_meta(holdout_rmse=224.2576, reused_from="/tmp/x.npz")) == base
    assert R.signature(dict(reversed(list(_meta().items())))) == base
    # and a field INSIDE the required set must still change it
    assert R.signature(_meta(n_ref=1)) != base
