"""Fetch verifies downloaded bytes against what the bucket reported, and a file
that fails verification is deleted rather than left looking usable.

That deletion is the point. The console handed us a truncated archive with a
valid header on 2026-09-08; a partial parquet with a plausible size is the same
failure and would be silently trained on.

Mutation rehearsals at write time, each restored immediately:
  * keep a mismatched file instead of unlinking  -> test_a_corrupt_download_is_removed RED
  * treat a multipart ETag as an MD5             -> test_multipart_etag_is_size_checked RED
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import fetch_data as fd  # noqa: E402


class FakeS3:
    """Serves fixed payloads; records what was asked for."""

    def __init__(self, payloads: dict[str, bytes], corrupt: set[str] | None = None):
        self.payloads, self.corrupt, self.downloaded = payloads, corrupt or set(), []

    def list_objects(self, bucket):
        out = []
        for key, body in sorted(self.payloads.items()):
            etag = ("deadbeef-2" if key.endswith("_multipart")
                    else hashlib.md5(body).hexdigest())
            out.append({"key": key, "bytes": len(body), "etag": etag,
                        "last_modified": "2026-09-04T08:30:00.000Z"})
        return out

    def download(self, bucket, key, dest):
        self.downloaded.append(key)
        body = self.payloads[key]
        if key in self.corrupt:
            body = body[: len(body) // 2]        # a truncated transfer
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(body)
        return Path(dest)


def test_verified_download_is_kept_and_recorded(tmp_path):
    client = FakeS3({"a.parquet": b"hello world" * 100})
    m = fd.fetch(client, "bucket", tmp_path, log=lambda *_: None)
    assert m["complete"] is True and m["failed"] == []
    entry = m["objects"]["a.parquet"]
    assert entry["verified"] and entry["etag_check"] == "md5" and entry["md5_ok"]
    assert entry["action"] == "downloaded"
    assert (tmp_path / "a.parquet").exists()
    written = json.loads((tmp_path / "FETCH_MANIFEST.json").read_text())
    assert written["objects"]["a.parquet"]["bytes"] == 1100


def test_a_corrupt_download_is_removed_not_left_behind(tmp_path):
    client = FakeS3({"good.parquet": b"g" * 500, "bad.parquet": b"b" * 500},
                    corrupt={"bad.parquet"})
    m = fd.fetch(client, "bucket", tmp_path, log=lambda *_: None)
    assert m["failed"] == ["bad.parquet"] and m["complete"] is False
    assert (tmp_path / "good.parquet").exists()
    assert not (tmp_path / "bad.parquet").exists(), (
        "a file failing verification must not survive — it looks usable")


def test_multipart_etag_is_size_checked_and_labelled_not_silently_verified(tmp_path):
    client = FakeS3({"big_multipart": b"m" * 4096})
    m = fd.fetch(client, "bucket", tmp_path, log=lambda *_: None)
    e = m["objects"]["big_multipart"]
    assert e["etag_check"] == "skipped_multipart", "a -N ETag is not an MD5"
    assert "md5" not in e, "no MD5 may be claimed for a multipart object"
    assert e["verified"] is True and e["size_ok"] is True


def test_existing_files_are_kept_and_re_verified_not_re_downloaded(tmp_path):
    body = b"z" * 300
    (tmp_path / "a.parquet").write_bytes(body)
    client = FakeS3({"a.parquet": body})
    m = fd.fetch(client, "bucket", tmp_path, log=lambda *_: None)
    assert client.downloaded == [], "an already-correct file is not re-fetched"
    assert m["objects"]["a.parquet"]["action"] == "kept" and m["complete"]


def test_an_existing_file_that_does_not_match_is_reported(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"stale contents")
    client = FakeS3({"a.parquet": b"fresh contents entirely"})
    m = fd.fetch(client, "bucket", tmp_path, log=lambda *_: None)
    assert m["failed"] == ["a.parquet"] and not m["complete"]
    assert m["objects"]["a.parquet"]["md5_ok"] is False


def test_force_redownloads(tmp_path):
    body = b"z" * 300
    (tmp_path / "a.parquet").write_bytes(body)
    client = FakeS3({"a.parquet": body})
    fd.fetch(client, "bucket", tmp_path, force=True, log=lambda *_: None)
    assert client.downloaded == ["a.parquet"]
