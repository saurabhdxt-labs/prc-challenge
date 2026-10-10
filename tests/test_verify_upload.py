"""tools/verify_upload.py — the generic pre-upload verifier (dry run by default; --upload only after the owner's per-file go):
the key rule and never-overwrite, the 5-per-UTC-day count read from the bucket, the registration's current sha logged,
the file against the template / base / its build meta, and an INDEPENDENT re-derivation of the HPX-S rule (no import of
scripts/analysis/hpx_lirf.py). Also the FINAL phase mode (--final): one '<team>_final.parquet', checked against the
template only (no prereg/base), and the argparse post-check that --version/--final are mutually exclusive and
--version requires --prereg/--base. Synthetic frames only; no bucket is touched.
Seen red first 2026-09-30 (collection error). Mutation rehearsal 2026-09-30: 17 of 17 RED (overwrite allowed, result
presence ignored, non-result counted and wrong rint killed only after the v49-status and 1800.6 cases were added, any day
counted, rejected counted, limit >, prereg unchecked, order / dtype / template unchecked, meta sha unchecked, tau >,
upward-only off, slip <, serve id check off, compare always 0).
FINAL-mode tests seen red first 2026-10-08 (functions did not exist: AttributeError on U.final_key / U.check_final_file
/ U.parse_args; 14 of 14 new tests failing; existing 13 passed unaffected). The prediction-dtype gate was corrected
2026-10-08 after measuring data/raw_final/final_submitting.parquet (TAXITIME_SEC_mvt float64, a placeholder) against
submissions/merry-quicksand_v59.parquet (int32, an accepted upload): check_final_file does NOT compare the prediction
dtype to the template, it requires an integer dtype; the MVT_ID_mvt dtype IS compared to the template (both float64).
Mutation rehearsal 2026-10-08: 6 mutants, all red (order check disabled -> red on the ids=[1,3,2] case; final_key's
f-string changed to insert "_v" -> red; parse_args' --final+--version combo check dropped -> red; parse_args'
--prereg/--base requirement dropped -> red (DID NOT RAISE); the integer-dtype check disabled -> red on the float64-
prediction case; the MVT_ID_mvt dtype check disabled -> red on the int64-ids case); each reverted and the file
reconfirmed green. build_meta_path (FM1 writes <stem>.manifest.json) added 2026-10-08, red first, 3 mutants red.
Reviewer 2026-10-09: the upload branch had no tests -> confirm_uploaded (the listing after the PUT), result_rc (a
Failed result exits 1) and the cross-mode flag refusals, red first (11 cases), 4 mutants red. The file now holds 23
test functions (40 collected cases); the --version path's 13 original tests are unchanged."""
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


def test_build_meta_path_prefers_meta_json_then_fm1_manifest(tmp_path):
    """The FINAL build comes from FM1's build-submission, which writes `<stem>.manifest.json` (with "sha256") beside the
    file; the older copy step wrote `<name>.meta.json`. build_meta_path takes the .meta.json when present, else the
    manifest, and refuses when neither exists (naming both). Seen red first 2026-10-08 (AttributeError: no
    build_meta_path). Mutation rehearsal 2026-10-08: fallback dropped -> red (manifest-only case); preference order
    swapped -> red (both-present case); existence check dropped -> red (neither case)."""
    f = tmp_path / "team_v1.parquet"
    f.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"team_v1\.parquet\.meta\.json.*team_v1\.manifest\.json"):
        U.build_meta_path(f)
    man = tmp_path / "team_v1.manifest.json"
    man.write_text("{}")
    assert U.build_meta_path(f) == man
    meta = tmp_path / "team_v1.parquet.meta.json"
    meta.write_text("{}")
    assert U.build_meta_path(f) == meta


def _listing(*entries):
    """Objects in the shape prc.s3.S3.list_objects really returns, parsed from the wire XML (BC-51: the first version of
    the test below faked the listing with a `size` key the producer never emits, and the real upload's confirmation
    crashed on `bytes`)."""
    xml = ('<?xml version="1.0" encoding="UTF-8"?><ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
           + "".join(f"<Contents><Key>{k}</Key><Size>{n}</Size><LastModified>2026-10-10T04:05:50.534Z</LastModified>"
                     f'<ETag>"e"</ETag></Contents>' for k, n in entries) + "</ListBucketResult>").encode()
    from prc import s3

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return xml
    return s3.S3(s3.Credentials("AK", "SK"), opener=lambda req: R()).list_objects("b")


def test_regression_bug_51_confirm_uploaded_reads_the_size_the_listing_really_carries():
    """BC-51 (2026-10-10 04:05Z, the FINAL upload): the PUT returned 200, then confirm_uploaded raised "bucket size
    4,080,675 expected, listed -1" because it read `size` while S3.list_objects returns `bytes`. The listing objects here
    come from the real parser, not a hand-written dict. Seen red first 2026-10-10 (ValueError, listed -1). Mutation
    rehearsal 2026-10-10: reading `size` again -> red; presence check dropped -> red (missing-key case); size comparison
    dropped -> red (wrong-size case)."""
    objs = _listing(("merry-quicksand_final.parquet", 4_080_675), ("merry-quicksand_v59.parquet", 2))
    assert set(objs[0]) == {"key", "bytes", "last_modified", "etag"}
    U.confirm_uploaded(objs, "merry-quicksand_final.parquet", 4_080_675)
    with pytest.raises(ValueError, match="merry-quicksand_final.parquet.*not listed"):
        U.confirm_uploaded(objs[1:], "merry-quicksand_final.parquet", 4_080_675)
    with pytest.raises(ValueError, match="merry-quicksand_final.parquet.*size 4,080,675 .* 4,080,674"):
        U.confirm_uploaded(_listing(("merry-quicksand_final.parquet", 4_080_674)), "merry-quicksand_final.parquet", 4_080_675)
    with pytest.raises(KeyError):                       # an object without the producer's size key is not silently -1
        U.confirm_uploaded([{"key": "merry-quicksand_final.parquet"}], "merry-quicksand_final.parquet", 4_080_675)


@pytest.mark.parametrize("result,rc", [(None, 0), ({"status": "Succeeded"}, 0), ({"status": "Failed"}, 1),
                                       ({"status": "Rejected", "message": "x"}, 1), ({}, 1)])
def test_result_rc_is_zero_only_for_no_result_or_succeeded(result, rc):
    """The old poll loop printed any result and returned 0; a FINAL result of status Failed must exit 1 so the one upload
    is never reported as accepted when the scorer refused it. No result (blind scoring) is 0. Seen red first 2026-10-09.
    Mutation rehearsal 2026-10-09: `return 0` for every result -> red (Failed / Rejected / {} cases)."""
    assert U.result_rc(result) == rc


@pytest.mark.parametrize("argv,msg", [
    (["--final", "--prereg", "x"], "--prereg"), (["--final", "--base", "x"], "--base"), (["--final", "--rederive", "hpx"], "--rederive"),
    (["--version", "3", "--prereg", "p", "--base", "b", "--file", "f"], "--file"),
    (["--version", "3", "--prereg", "p", "--base", "b", "--template", "t"], "--template"),
])
def test_parse_args_refuses_flags_that_belong_to_the_other_mode(argv, msg):
    """A flag the chosen mode ignores is a mis-typed command, not a no-op: --final with --prereg/--base/--rederive, or
    --version with --file/--template, is refused by name. Seen red first 2026-10-09 (accepted silently). Mutation
    rehearsal 2026-10-09: the cross-mode check dropped -> red on every case."""
    with pytest.raises(ValueError, match=msg):
        U.parse_args(argv)


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


# ------------------------------------------------------------------ FINAL phase

def test_final_key_is_team_name_plus_final():
    """Organiser's page: '<team-name>_final.parquet'. Mutation rehearsal 2026-10-08: f-string ->
    f"{team}_v_final.parquet" -> red (expected literal mismatch)."""
    assert U.final_key() == "merry-quicksand_final.parquet"
    assert U.final_key("other-team") == "other-team_final.parquet"


def test_check_final_file_passes_a_clean_file():
    """The template's TAXITIME_SEC_mvt is float64 (a placeholder, measured 2026-10-08 on
    data/raw_final/final_submitting.parquet) but an accepted upload is int32 (measured on
    submissions/merry-quicksand_v59.parquet); check_final_file does NOT compare the prediction
    dtype to the template, only requires it be an integer dtype. int64 passes too."""
    template = _sub([100, 200, 300]).astype({"TAXITIME_SEC_mvt": "float64"})
    new = _sub([100, 250, 300])
    assert U.check_final_file(new, template) == {"rows": 3, "min": 100.0, "max": 300.0, "mean": 650 / 3}
    new64 = _sub([100, 250, 300]).astype({"TAXITIME_SEC_mvt": "int64"})
    assert U.check_final_file(new64, template) == {"rows": 3, "min": 100.0, "max": 300.0, "mean": 650 / 3}


@pytest.mark.parametrize("new,msg", [
    (_sub([100, 200]), "row count 2 != template 3"),
    (_sub([100, 200, 300], ids=[1, 3, 2]), "ids / order differ from the template"),
    (_sub([100, 200, 300], ids=[1, 2, 4]), "identifier set does not match the template"),
    (_sub([100, 0, 300]), "1 non-positive predictions"),
    (_sub([100, 200, 300]).astype({"TAXITIME_SEC_mvt": "float64"}),
     "dtype float64 is not an integer dtype \\(accepted uploads are int32\\)"),
    (pd.DataFrame({"MVT_ID_mvt": np.array([1, 2, 3], dtype="int64"), "TAXITIME_SEC_mvt": np.array([100, 200, 300], "int32")}),
     r"MVT_ID_mvt dtype int64 != template float64"),
    (pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "WRONG": [100, 200, 300]}),
     r"columns \['MVT_ID_mvt', 'WRONG'\] != \['MVT_ID_mvt', 'TAXITIME_SEC_mvt'\]"),
])
def test_check_final_file_refuses_each_defect(new, msg):
    """Each defect refused by check_final_file, either via bs.check_submission (columns, row count,
    id set, non-positive) or the final-only checks (order, id dtype, prediction-is-integer).
    Mutation rehearsal 2026-10-08:
      - drop the order (array_equal) comparison entirely -> the ids=[1,3,2] case goes GREEN
        incorrectly (false pass); restoring the check -> red on the mutant, confirming the test
        catches it.
      - replace `np.issubdtype(..., np.integer)` with `True` (so the integer check never fires)
        -> the float64-prediction case goes GREEN incorrectly, AND the int64-passes half of
        test_check_final_file_passes_a_clean_file stops proving anything; restoring the check ->
        red on the mutant.
      - drop the MVT_ID_mvt dtype comparison -> the int64-ids case goes GREEN incorrectly;
        restoring the check -> red on the mutant.
    """
    template = _sub([100, 200, 300])
    with pytest.raises(ValueError, match=msg):
        U.check_final_file(new, template)


def test_check_final_file_refuses_a_null_prediction():
    template = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "TAXITIME_SEC_mvt": [100.0, 200.0, 300.0]})
    new = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "TAXITIME_SEC_mvt": [100.0, np.nan, 300.0]})
    with pytest.raises(ValueError, match="1 null predictions"):
        U.check_final_file(new, template)


# ------------------------------------------------------------------ CLI argument validation

def test_parse_args_version_mode_requires_prereg_and_base():
    """Mutation rehearsal 2026-10-08: drop this post-check -> red (SystemExit/AttributeError
    instead of the intended ValueError, or a downstream crash) on both sub-cases below."""
    with pytest.raises(ValueError, match="--version mode requires --prereg and --base"):
        U.parse_args(["--version", "48"])
    with pytest.raises(ValueError, match="--version mode requires --prereg and --base"):
        U.parse_args(["--version", "48", "--prereg", "x.md"])


def test_parse_args_version_mode_ok_with_prereg_and_base():
    a = U.parse_args(["--version", "48", "--prereg", "x.md", "--base", "y.parquet"])
    assert a.version == 48 and a.final is False and a.prereg == "x.md" and a.base == "y.parquet"


def test_parse_args_final_mode_needs_no_prereg_or_base():
    a = U.parse_args(["--final"])
    assert a.final is True and a.version is None


def test_parse_args_refuses_final_with_version():
    """Mutation rehearsal 2026-10-08: drop this check -> red (no exception raised; the test's
    pytest.raises block fails with 'DID NOT RAISE')."""
    with pytest.raises(ValueError, match="--final cannot be combined with --version"):
        U.parse_args(["--version", "48", "--final"])


def test_parse_args_refuses_neither_mode():
    with pytest.raises(ValueError, match="exactly one of --version or --final is required"):
        U.parse_args([])
