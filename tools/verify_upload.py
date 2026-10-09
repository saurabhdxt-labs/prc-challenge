"""Generic pre-upload verifier for a built submission. Dry run by default; `--upload` only after the owner's per-file go,
and it refuses unless every gate passes (owner rule: every upload needs its own per-file go).

    /opt/homebrew/bin/python3.11 -B tools/verify_upload.py --version 48 --prereg plans/PREREG_hpx_serve_2026_09_29.md \
        --base submissions/merry-quicksand_v46.parquet --rederive hpx --serve logs/hpx_scratch/serve_2026 [--upload]

FINAL phase (one blind submission, no prereg/base, scored blind -- no result file ever appears):

    /opt/homebrew/bin/python3.11 -B tools/verify_upload.py --final [--file submissions/final/<key>] \
        [--template data/raw_final/final_submitting.parquet] [--upload]

Gates (each tested in tests/test_verify_upload.py):
  G1 key = build_submission.submission_name(N); neither the key nor its _result.json is in the bucket (never overwrite)
  G2 fewer than 5 Succeeded results in the current UTC day, counted from the bucket's own result files
  G3 the registration's CURRENT sha is in reports/r12_prereg_hashes.log (registered, not edited since it was logged)
  G4 check_submission vs the template (incl. no value <= 0); 344,841 rows; ids and order = the base file; dtype = base
  G5 the file's sha = its build meta (<file>.meta.json: sha256 or out_sha)
  G6 (--rederive hpx) the file equals an INDEPENDENT re-derivation of the registered rule on the base (this module does
     not import scripts/analysis/hpx_lirf.py); any differing row fails
After --upload: polls the bucket up to 180 s for <key>_result.json, saves it beside the file, prints status and score.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_submission as bs  # noqa: E402

BUCKET = "prc-2026-merry-quicksand"
N_ROWS = 344_841
DAILY_LIMIT = 5
HASH_LOG = ROOT / "reports" / "r12_prereg_hashes.log"
HPX_SLIP_S = 21_600.0            # the registered rule's upper bound on m (PREREG_hpx_serve "The rule")
RESULT_WAIT_S = 180


def key_for(version: int) -> str:
    return bs.submission_name(version)


def final_key(team: str = bs.TEAM) -> str:
    """The FINAL phase's one filename: '<team-name>_final.parquet' (organiser's page)."""
    return f"{team}_final.parquet"


def key_free(key: str, bucket_keys) -> None:
    keys = set(bucket_keys)
    if key in keys:
        raise ValueError(f"{key} is already in the bucket; never overwritten")
    if key + "_result.json" in keys:
        raise ValueError(f"{key} already has a result in the bucket")


def uploads_today(objects, status: dict, now_utc: dt.datetime) -> int:
    """Succeeded *_result.json objects whose last_modified falls in now_utc's UTC day."""
    day = now_utc.astimezone(dt.timezone.utc).date()
    n = 0
    for o in objects:
        k = o["key"]
        if not k.endswith("_result.json"):
            continue
        when = dt.datetime.fromisoformat(o["last_modified"].replace("Z", "+00:00")).astimezone(dt.timezone.utc)
        if when.date() == day and status.get(k) == "Succeeded":
            n += 1
    return n


def daily_gate(n_today: int) -> None:
    if n_today >= DAILY_LIMIT:
        raise ValueError(f"{n_today} of {DAILY_LIMIT} uploads already scored today (UTC); the next slot opens at 00:00 UTC")


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def prereg_logged(prereg: pathlib.Path, hash_log_text: str) -> str:
    sha = _sha(prereg)
    if sha not in hash_log_text:
        raise ValueError(f"{pathlib.Path(prereg).name}'s current sha {sha[:8]} is not in the hash log (edited after logging, "
                         "or never registered)")
    return sha


def check_file(new: pd.DataFrame, base: pd.DataFrame, template: pd.DataFrame, *, n_rows: int = N_ROWS) -> dict:
    bs.check_submission(new, template)
    if len(new) != n_rows:
        raise ValueError(f"{len(new)} rows != {n_rows}")
    if not np.array_equal(new.MVT_ID_mvt.to_numpy(float), base.MVT_ID_mvt.to_numpy(float)):
        raise ValueError("ids / order differ from the base")
    if new.TAXITIME_SEC_mvt.dtype != base.TAXITIME_SEC_mvt.dtype:
        raise ValueError(f"dtype {new.TAXITIME_SEC_mvt.dtype} != base {base.TAXITIME_SEC_mvt.dtype}")
    return {"rows": int(len(new)), "n_changed_vs_base": int((new.TAXITIME_SEC_mvt.to_numpy() != base.TAXITIME_SEC_mvt.to_numpy()).sum())}


def check_final_file(new: pd.DataFrame, template: pd.DataFrame) -> dict:
    """FINAL phase: no base file to compare against, only the template (same two columns,
    670,790 rows). bs.check_submission already refuses wrong columns, wrong row count, a
    mismatched id SET, nulls and non-positive values; this adds the checks a set comparison
    cannot make: row ORDER against the template, the id dtype against the template (measured
    2026-10-08: data/raw_final/final_submitting.parquet's MVT_ID_mvt is float64), and the
    prediction dtype -- NOT compared to the template (its TAXITIME_SEC_mvt is also float64, a
    placeholder column), but required to be an integer dtype: every accepted upload including
    submissions/merry-quicksand_v59.parquet is int32, and build_submission.py writes int32 by
    design."""
    bs.check_submission(new, template)
    if not np.array_equal(new.MVT_ID_mvt.to_numpy(float), template.MVT_ID_mvt.to_numpy(float)):
        raise ValueError("ids / order differ from the template")
    if new.MVT_ID_mvt.dtype != template.MVT_ID_mvt.dtype:
        raise ValueError(f"MVT_ID_mvt dtype {new.MVT_ID_mvt.dtype} != template {template.MVT_ID_mvt.dtype}")
    if not np.issubdtype(new.TAXITIME_SEC_mvt.dtype, np.integer):
        raise ValueError(f"dtype {new.TAXITIME_SEC_mvt.dtype} is not an integer dtype (accepted uploads are int32)")
    v = new.TAXITIME_SEC_mvt
    return {"rows": int(len(new)), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}


def build_meta_path(path: pathlib.Path) -> pathlib.Path:
    """The build record that carries the file's sha: `<name>.meta.json` (the copy step's record) when present, else
    FM1 build-submission's `<stem>.manifest.json` (key "sha256"); refused when neither exists."""
    path = pathlib.Path(path)
    meta = path.with_name(path.name + ".meta.json")
    manifest = path.with_name(path.stem + ".manifest.json")
    for cand in (meta, manifest):
        if cand.exists():
            return cand
    raise ValueError(f"no build record beside {path.name}: neither {meta.name} nor {manifest.name} exists")


def meta_sha_ok(path: pathlib.Path, meta: dict) -> str:
    want = meta.get("sha256") or meta.get("out_sha")
    if not want:
        raise ValueError("build meta has no sha")
    got = _sha(path)
    if got != want:
        raise ValueError(f"{pathlib.Path(path).name} sha {got[:8]} != its build meta ({want[:8]})")
    return got


def rederive_hpx(base: pd.DataFrame, serve: pd.DataFrame, *, tau: float, slip_s: float = HPX_SLIP_S) -> pd.DataFrame:
    """The registered HPX-S rule, written from the prereg text: value -> rint(m) iff p >= tau AND m > value AND
    m <= slip_s, on the rows in `serve`; every other row keeps the base value (dtype kept)."""
    if serve.MVT_ID_mvt.duplicated().any() or not serve.MVT_ID_mvt.isin(base.MVT_ID_mvt).all():
        raise ValueError("serve ids must be unique and present in the base")
    s = serve.set_index("MVT_ID_mvt")
    out = base.copy()
    v = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    m = s.m.reindex(v.index)
    p = s.p.reindex(v.index)
    move = (p >= tau) & (m > v.astype(float)) & (m <= slip_s)
    new = np.where(move.to_numpy(), np.rint(m.to_numpy()), v.to_numpy())
    out["TAXITIME_SEC_mvt"] = new.astype(base.TAXITIME_SEC_mvt.dtype)
    return out


def compare(expected: pd.DataFrame, got: pd.DataFrame) -> int:
    return int((expected.TAXITIME_SEC_mvt.to_numpy() != got.TAXITIME_SEC_mvt.to_numpy()).sum())


def confirm_uploaded(objects, key: str, size: int) -> None:
    """After the PUT: the bucket listing must show `key` with exactly the local byte size — for the FINAL upload the
    listing is the only confirmation there is (no result file is promised)."""
    found = [o for o in objects if o.get("key") == key]
    if not found:
        raise ValueError(f"{key} is not listed in the bucket after the upload")
    got = int(found[0].get("size", -1))
    if got != int(size):
        raise ValueError(f"{key}: bucket size {size:,} expected, listed {got:,}")


def result_rc(result) -> int:
    """Exit code for a scorer result: 0 for no result (blind scoring) or status Succeeded; 1 for anything else, so a
    Failed / Rejected / malformed result is never reported as an accepted upload."""
    if result is None:
        return 0
    return 0 if isinstance(result, dict) and result.get("status") == "Succeeded" else 1


def parse_args(argv=None):
    """Parse + validate CLI args. Exactly one of --version (ordinary submission; needs --prereg
    and --base) or --final (FINAL phase; no prereg/base, template-checked instead)."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--prereg", default=None)
    ap.add_argument("--base", default=None)
    ap.add_argument("--rederive", choices=["hpx"], default=None)
    ap.add_argument("--serve", default=None, help="--rederive hpx: the hpx_lirf serve directory")
    ap.add_argument("--file", default=None, help="--final: explicit built file path (default submissions/final/<key>)")
    ap.add_argument("--template", default=None, help="--final: template path (default data/raw_final/final_submitting.parquet)")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args(argv)
    if a.final and a.version is not None:
        raise ValueError("--final cannot be combined with --version")
    if not a.final and a.version is None:
        raise ValueError("exactly one of --version or --final is required")
    if not a.final and (not a.prereg or not a.base):
        raise ValueError("--version mode requires --prereg and --base")
    if a.final:
        for flag, val in (("--prereg", a.prereg), ("--base", a.base), ("--rederive", a.rederive), ("--serve", a.serve)):
            if val:
                raise ValueError(f"{flag} belongs to --version mode and is ignored by --final: refused")
    else:
        for flag, val in (("--file", a.file), ("--template", a.template)):
            if val:
                raise ValueError(f"{flag} belongs to --final mode and is ignored by --version: refused")
    return a


def _today_result_status(s3, objs) -> tuple[dict, dt.datetime]:  # pragma: no cover - thin I/O helper, exercised via main
    status = {}
    today = dt.datetime.now(dt.timezone.utc)
    for o in objs:
        if o["key"].endswith("_result.json") and dt.datetime.fromisoformat(
                o["last_modified"].replace("Z", "+00:00")).date() == today.date():
            dest = ROOT / "logs" / "campaign" / "bucket_results" / o["key"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download(BUCKET, o["key"], dest)
            status[o["key"]] = json.loads(dest.read_text()).get("status")
    return status, today


def main(argv=None) -> int:  # pragma: no cover - I/O shell around the tested gates (bucket, files)
    a = parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from prc.s3 import S3, load_credentials
    s3 = S3(load_credentials())

    if a.final:
        key = final_key()
        path = pathlib.Path(a.file) if a.file else ROOT / "submissions" / "final" / key
        template_path = pathlib.Path(a.template) if a.template else ROOT / "data" / "raw_final" / "final_submitting.parquet"
        report = {"key": key, "file": str(path)}
        objs = s3.list_objects(BUCKET)
        key_free(key, [o["key"] for o in objs])
        status, today = _today_result_status(s3, objs)
        n_today = uploads_today(objs, status, today)
        daily_gate(n_today)
        report["uploads_today_utc"] = n_today
        new = pd.read_parquet(path)
        template = pd.read_parquet(template_path)
        report.update(check_final_file(new, template))
        report["sha"] = meta_sha_ok(path, json.loads(build_meta_path(path).read_text()))
        print(json.dumps(report, indent=1))
        if not a.upload:
            print("DRY RUN: every gate passed; nothing uploaded")
            return 0
        rc = s3.upload(BUCKET, key, path)
        print(f"uploaded {key}: HTTP {rc}")
        confirm_uploaded(s3.list_objects(BUCKET), key, path.stat().st_size)      # the only confirmation a blind upload gets
        print(f"confirmed: {key} listed with {path.stat().st_size:,} bytes")
        uploaded_sha = _sha(path)
        path.with_name(path.name + ".uploaded.json").write_text(json.dumps(
            {"key": key, "sha256": uploaded_sha, "uploaded_utc": dt.datetime.now(dt.timezone.utc).isoformat()},
            indent=1))
        print("FINAL is scored blind: no result file may ever appear for this key")
        res_key = key + "_result.json"
        dest = ROOT / "submissions" / "final" / res_key
        t0 = time.time()
        while time.time() - t0 < RESULT_WAIT_S:
            if res_key in [o["key"] for o in s3.list_objects(BUCKET)]:
                s3.download(BUCKET, res_key, dest)
                print(dest.read_text())
                return result_rc(json.loads(dest.read_text()))
            time.sleep(10)
        print(f"no result expected for FINAL ({key}); upload confirmed")
        return result_rc(None)

    key = key_for(a.version)
    path = ROOT / "submissions" / key
    report = {"key": key, "file": str(path.relative_to(ROOT))}
    objs = s3.list_objects(BUCKET)
    key_free(key, [o["key"] for o in objs])
    status, today = _today_result_status(s3, objs)
    n_today = uploads_today(objs, status, today)
    daily_gate(n_today)
    report["uploads_today_utc"] = n_today
    report["prereg_sha"] = prereg_logged(ROOT / a.prereg, HASH_LOG.read_text())
    new, base = pd.read_parquet(path), pd.read_parquet(ROOT / a.base)
    report.update(check_file(new, base, pd.read_parquet(ROOT / "data" / "raw" / "submitting.parquet")))
    report["sha"] = meta_sha_ok(path, json.loads(path.with_name(path.name + ".meta.json").read_text()))
    if a.rederive == "hpx":
        sd = ROOT / a.serve
        tau = json.loads((sd / "serve.json").read_text())["tau"]
        n_diff = compare(rederive_hpx(base, pd.read_parquet(sd / "serve_2026.parquet"), tau=tau), new)
        if n_diff:
            raise ValueError(f"G6: {n_diff} rows differ from the independent HPX re-derivation")
        report["g6_rederive_hpx"] = "0 rows differ"
    print(json.dumps(report, indent=1))
    if not a.upload:
        print("DRY RUN: every gate passed; nothing uploaded")
        return 0
    rc = s3.upload(BUCKET, key, path)
    print(f"uploaded {key}: HTTP {rc}")
    res_key = key + "_result.json"
    dest = ROOT / "submissions" / res_key
    t0 = time.time()
    while time.time() - t0 < RESULT_WAIT_S:
        if res_key in [o["key"] for o in s3.list_objects(BUCKET)]:
            s3.download(BUCKET, res_key, dest)
            print(dest.read_text())
            return 0
        time.sleep(10)
    print(f"no result after {RESULT_WAIT_S} s — check the key ({key})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
