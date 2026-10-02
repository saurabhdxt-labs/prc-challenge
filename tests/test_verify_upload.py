"""tools/verify_upload.py — the generic pre-upload verifier (dry run by default; --upload only after the owner's per-file go):
the key rule and never-overwrite, the 5-per-UTC-day count read from the bucket, the registration's current sha logged,
the file against the template / base / its build meta, and an INDEPENDENT re-derivation of the HPX-S rule (no import of
scripts/analysis/hpx_lirf.py). Synthetic frames only; no bucket is touched.
Seen red first 2026-09-30 (collection error). Mutation rehearsal 2026-09-30: 17 of 17 RED (overwrite allowed, result
presence ignored, non-result counted and wrong rint killed only after the v49-status and 1800.6 cases were added, any day
counted, rejected counted, limit >, prereg unchecked, order / dtype / template unchecked, meta sha unchecked, tau >,
upward-only off, slip <, serve id check off, compare always 0)."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("verify_upload", ROOT / "tools/verify_upload.py")
U = importlib.util.module_from_spec(_spec)
sys.modules["verify_upload"] = U
_spec.loader.exec_module(U)

NOW = dt.datetime(2026, 9, 30, 11, 20, tzinfo=dt.timezone.utc)


# ------------------------------------------------------------------ key

def test_key_is_the_submission_convention_and_must_be_new():
    assert U.key_for(48) == "merry-quicksand_v48.parquet"
    U.key_free("merry-quicksand_v48.parquet", ["merry-quicksand_v46.parquet", "merry-quicksand_v46.parquet_result.json"])
    with pytest.raises(ValueError, match="merry-quicksand_v48.parquet is already in the bucket; never overwritten"):
        U.key_free("merry-quicksand_v48.parquet", ["merry-quicksand_v48.parquet"])
    with pytest.raises(ValueError, match="merry-quicksand_v48.parquet already has a result"):
        U.key_free("merry-quicksand_v48.parquet", ["merry-quicksand_v48.parquet_result.json"])


# ------------------------------------------------------------------ daily limit

def _obj(key, when):
    return {"key": key, "last_modified": when}


def test_uploads_today_counts_succeeded_results_inside_the_current_utc_day_only():
    """5 per UTC day (resets 00:00 UTC = 20:00 EDT); a DAILY_LIMIT rejection did not use a slot; only *_result.json
    objects count (the challenge allows 5 uploads per UTC day)."""
    objs = [_obj("merry-quicksand_v46.parquet_result.json", "2026-09-29T00:02:10.000Z"),     # yesterday (UTC)
            _obj("merry-quicksand_v47.parquet_result.json", "2026-09-30T00:00:01.000Z"),     # today, first second
            _obj("merry-quicksand_v48.parquet_result.json", "2026-09-30T10:00:00.000Z"),     # today, rejected
            _obj("merry-quicksand_v49.parquet", "2026-09-30T10:05:00.000Z"),                 # an upload, not a result
            _obj("merry-quicksand_v50.parquet_result.json", "2026-09-30T23:59:59.000Z")]     # today, last second
    status = {"merry-quicksand_v47.parquet_result.json": "Succeeded",
              "merry-quicksand_v48.parquet_result.json": "Rejected",
              "merry-quicksand_v50.parquet_result.json": "Succeeded",
              "merry-quicksand_v46.parquet_result.json": "Succeeded",
              "merry-quicksand_v49.parquet": "Succeeded"}                                   # not a result: never counted
    assert U.uploads_today(objs, status, NOW) == 2


def test_daily_gate_refuses_at_the_limit():
    U.daily_gate(4)
    with pytest.raises(ValueError, match="5 of 5 uploads already scored today \\(UTC\\); the next slot opens at 00:00 UTC"):
        U.daily_gate(5)


# ------------------------------------------------------------------ registration

def test_prereg_must_be_logged_at_its_current_sha(tmp_path):
    p = tmp_path / "PREREG_x.md"
    p.write_text("rule T\n")
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    assert U.prereg_logged(p, f"{sha}  2026-09-30T04:39:39Z  BUILD-REPORT  PREREG_x.md\n") == sha
    p.write_text("rule T\nedited after logging\n")
    with pytest.raises(ValueError, match="PREREG_x.md's current sha .* is not in the hash log"):
        U.prereg_logged(p, f"{sha}  2026-09-30T04:39:39Z  BUILD-REPORT  PREREG_x.md\n")


# ------------------------------------------------------------------ the file

def _sub(vals, ids=None):
    ids = np.arange(1.0, len(vals) + 1) if ids is None else np.asarray(ids, float)
    return pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.asarray(vals, "int32")})


def test_check_file_passes_a_clean_file_and_counts_changes_vs_base():
    base, new = _sub([100, 200, 300]), _sub([100, 250, 300])
    assert U.check_file(new, base, base.copy(), n_rows=3) == {"rows": 3, "n_changed_vs_base": 1}


@pytest.mark.parametrize("new,msg", [
    (_sub([100, 200]), "row count 2 != template 3"),
    (_sub([100, 200, 300], ids=[1, 3, 2]), "ids / order differ from the base"),
    (_sub([100, 0, 300]), "1 non-positive predictions"),                  # build_submission.check_submission
    (_sub([100, 200, 300]).astype({"TAXITIME_SEC_mvt": "float64"}), "dtype float64 != base int32"),
])
def test_check_file_refuses_each_defect(new, msg):
    base = _sub([100, 200, 300])
    with pytest.raises(ValueError, match=msg):
        U.check_file(new, base, base.copy(), n_rows=3)


def test_meta_sha_must_match_the_file(tmp_path):
    f = tmp_path / "x.parquet"
    f.write_bytes(b"abc")
    sha = hashlib.sha256(b"abc").hexdigest()
    assert U.meta_sha_ok(f, {"sha256": sha}) == sha
    assert U.meta_sha_ok(f, {"out_sha": sha}) == sha
    with pytest.raises(ValueError, match="x.parquet sha .* != its build meta"):
        U.meta_sha_ok(f, {"sha256": "0" * 64})
    with pytest.raises(ValueError, match="build meta has no sha"):
        U.meta_sha_ok(f, {})


# ------------------------------------------------------------------ HPX-S independent re-derivation

def test_rederive_hpx_moves_only_confident_upward_moves_within_six_hours():
    """value -> rint(m) iff p >= tau AND m > value AND m <= 21,600 (plans/PREREG_hpx_serve_2026_09_29.md, 'The rule');
    rows absent from the serve file keep the base value."""
    base = _sub([600, 600, 600, 600, 600, 900, 700])
    serve = pd.DataFrame({"MVT_ID_mvt": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                          "m": [1800.6, 1800.0, 30_000.0, 1500.0, 800.0, 21_600.0],
                          "p": [0.63, 0.62, 0.99, 0.99, 0.99, 0.70]})
    got = U.rederive_hpx(base, serve, tau=0.63)
    assert got.TAXITIME_SEC_mvt.tolist() == [600, 1801, 600, 600, 1500, 900, 21_600]          # rint(1800.6) = 1801
    assert got.TAXITIME_SEC_mvt.dtype == np.int32 and got.MVT_ID_mvt.tolist() == base.MVT_ID_mvt.tolist()


def test_rederive_hpx_refuses_duplicate_or_unknown_serve_ids():
    base = _sub([600, 600])
    with pytest.raises(ValueError, match="serve ids must be unique and present in the base"):
        U.rederive_hpx(base, pd.DataFrame({"MVT_ID_mvt": [1.0, 1.0], "m": [900.0, 900.0], "p": [1.0, 1.0]}), tau=0.5)
    with pytest.raises(ValueError, match="serve ids must be unique and present in the base"):
        U.rederive_hpx(base, pd.DataFrame({"MVT_ID_mvt": [9.0], "m": [900.0], "p": [1.0]}), tau=0.5)


def test_compare_counts_every_row_that_differs():
    assert U.compare(_sub([1, 2, 3]), _sub([1, 2, 3])) == 0
    assert U.compare(_sub([1, 2, 3]), _sub([1, 5, 4])) == 2
