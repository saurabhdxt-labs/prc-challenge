"""Generic pre-upload verifier for a built submission. Dry run by default; `--upload` only after the owner's per-file go,
and it refuses unless every gate passes (owner rule: every upload needs its own per-file go).

    /opt/homebrew/bin/python3.11 -B tools/verify_upload.py --version 48 --prereg plans/PREREG_hpx_serve_2026_09_29.md \
        --base submissions/merry-quicksand_v46.parquet --rederive hpx --serve logs/hpx_scratch/serve_2026 [--upload]

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


def main(argv=None) -> int:  # pragma: no cover - I/O shell around the tested gates (bucket, files)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--prereg", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--rederive", choices=["hpx"], default=None)
    ap.add_argument("--serve", default=None, help="--rederive hpx: the hpx_lirf serve directory")
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    from prc.s3 import S3, load_credentials
    s3 = S3(load_credentials())
    key = key_for(a.version)
    path = ROOT / "submissions" / key
    report = {"key": key, "file": str(path.relative_to(ROOT))}
    objs = s3.list_objects(BUCKET)
    key_free(key, [o["key"] for o in objs])
    status = {}
    today = dt.datetime.now(dt.timezone.utc)
    for o in objs:
        if o["key"].endswith("_result.json") and dt.datetime.fromisoformat(
                o["last_modified"].replace("Z", "+00:00")).date() == today.date():
            dest = ROOT / "logs" / "campaign" / "bucket_results" / o["key"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download(BUCKET, o["key"], dest)
            status[o["key"]] = json.loads(dest.read_text()).get("status")
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
