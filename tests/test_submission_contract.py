"""The submission file must be well-formed BEFORE it is uploaded.

A malformed file wastes one of five daily submission slots. A misnamed one is worse: the
bucket rejects it with a 403, which reads as a permissions failure and sent this project
chasing a non-existent access problem for a day. Every scored row on the public leaderboard
matches `<team>_v<N>.parquet`; nothing else does.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bs", ROOT / "scripts" / "build_submission.py")
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)


def _template(n=50):
    return pd.DataFrame({"MVT_ID_mvt": np.arange(n),
                         "TAXITIME_SEC_mvt": np.full(n, np.nan)})


def _good(n=50):
    return pd.DataFrame({"MVT_ID_mvt": np.arange(n),
                         "TAXITIME_SEC_mvt": np.full(n, 900, dtype="int32")})


@pytest.mark.parametrize("v,expected", [(1, "merry-quicksand_v1.parquet"),
                                        (7, "merry-quicksand_v7.parquet"),
                                        (123, "merry-quicksand_v123.parquet")])
def test_submission_name_matches_the_leaderboard_convention(v, expected):
    assert bs.submission_name(v) == expected
    assert bs.SUBMISSION_RE.match(expected)


@pytest.mark.parametrize("bad", [0, -1, "3", None, 1.5])
def test_submission_name_refuses_anything_that_is_not_a_positive_integer(bad):
    with pytest.raises((ValueError, TypeError)):
        bs.submission_name(bad)


@pytest.mark.parametrize("name", [
    "merry-quicksand.parquet",        # the old default - this exact name caused the 403
    "merry-quicksand_v0.parquet",     # versions start at 1
    "merry_quicksand_v1.parquet",     # underscore instead of hyphen in the team name
    "merry-quicksand_v1.csv",         # wrong extension
    "MERRY-QUICKSAND_v1.parquet",     # case
    "merry-quicksand_v1 .parquet",    # trailing space
])
def test_the_regex_rejects_names_the_scorer_would_403(name):
    assert not bs.SUBMISSION_RE.match(name), f"{name!r} would have been accepted"


def test_check_submission_accepts_a_well_formed_file():
    bs.check_submission(_good(), _template())


@pytest.mark.parametrize("mutate,msg", [
    (lambda f: f.iloc[:-1], "row count"),
    (lambda f: f.assign(MVT_ID_mvt=f.MVT_ID_mvt + 1000), "identifier set"),
    (lambda f: f.assign(TAXITIME_SEC_mvt=f.TAXITIME_SEC_mvt.astype(float).mask(
        f.index < 3)), "null predictions"),
    (lambda f: f.assign(TAXITIME_SEC_mvt=f.TAXITIME_SEC_mvt.astype(float).mask(
        f.index < 2, np.inf)), "non-finite"),
    (lambda f: f.assign(TAXITIME_SEC_mvt=f.TAXITIME_SEC_mvt.astype(float).mask(
        f.index < 4, 0.0)), "non-positive"),
    (lambda f: f.rename(columns={"TAXITIME_SEC_mvt": "taxi"}), "columns"),
    (lambda f: f[["TAXITIME_SEC_mvt", "MVT_ID_mvt"]], "columns"),
])
def test_check_submission_rejects_each_way_a_file_can_be_wrong(mutate, msg):
    """Each mutation is a real way a submission has been or could be malformed.

    Column ORDER matters as much as membership: a file whose two columns are swapped has
    the right names, the right length and the right identifiers, and is still wrong.
    """
    with pytest.raises(ValueError, match=msg):
        bs.check_submission(mutate(_good()), _template())
