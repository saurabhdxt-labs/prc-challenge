"""fetch_data — pull the competition files from the bucket and prove the bytes
are the bytes the bucket holds.

Browser downloads from the console proved unreliable on 2026-09-08: it produced
one truncated archive with a valid header and no central directory, and stalled
repeatedly on individual files. Fetching over the API instead lets us verify
each object against the ETag the server reports, which a browser download never
gives us.

ETag semantics, because they are not uniform: for a single-part upload MinIO's
ETag is the MD5 of the content, so it can be checked directly. For a multipart
upload it is a digest of part digests with a `-N` suffix and is NOT an MD5 of
the whole object; those are size-checked only, and reported as such rather than
being quietly called verified.

Usage:
  python3.11 scripts/fetch_data.py [--bucket prc-2026-datasets] [--dst data/raw] [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prc.s3 import S3, load_credentials  # noqa: E402

DEFAULT_BUCKET = "prc-2026-datasets"


def md5_of(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, obj: dict) -> dict:
    """Compare a downloaded file against the object the bucket reported."""
    size = path.stat().st_size
    result = {"key": obj["key"], "bytes": size, "expected_bytes": obj["bytes"],
              "etag": obj["etag"], "size_ok": size == obj["bytes"]}
    if "-" in obj["etag"]:
        result["etag_check"] = "skipped_multipart"
        result["verified"] = result["size_ok"]
    else:
        result["md5"] = md5_of(path)
        result["etag_check"] = "md5"
        result["md5_ok"] = result["md5"] == obj["etag"]
        result["verified"] = result["size_ok"] and result["md5_ok"]
    return result


def fetch(client, bucket: str, dst: Path, force: bool = False, log=print) -> dict:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    objects = client.list_objects(bucket)
    log(f"{bucket}: {len(objects)} objects, "
        f"{sum(o['bytes'] for o in objects) / 1048576:.1f} MiB")
    results, failed = {}, []
    for obj in sorted(objects, key=lambda o: o["key"]):
        target = dst / obj["key"]
        if target.exists() and not force:
            check = verify(target, obj)
            check["action"] = "kept"
            results[obj["key"]] = check
            log(f"  kept      {obj['key']:44} {'verified' if check['verified'] else 'MISMATCH'}")
            if not check["verified"]:
                failed.append(obj["key"])
            continue
        client.download(bucket, obj["key"], target)
        check = verify(target, obj)
        check["action"] = "downloaded"
        results[obj["key"]] = check
        log(f"  fetched   {obj['key']:44} {check['bytes'] / 1048576:6.1f} MiB "
            f"{'verified' if check['verified'] else 'MISMATCH'}")
        if not check["verified"]:
            # A file that does not match what the bucket reported is worse than
            # a missing one, because it looks usable. Remove it.
            target.unlink(missing_ok=True)
            failed.append(obj["key"])
    manifest = {"bucket": bucket, "objects": results, "failed": failed,
                "complete": not failed and len(results) == len(objects),
                "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (dst / "FETCH_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--dst", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    a = ap.parse_args(argv)
    m = fetch(S3(load_credentials()), a.bucket, a.dst, force=a.force)
    if m["failed"]:
        print(f"\nFAILED verification ({len(m['failed'])}): " + ", ".join(m["failed"]))
        return 1
    print(f"\nall {len(m['objects'])} objects verified against the bucket")
    return 0


if __name__ == "__main__":
    sys.exit(main())
