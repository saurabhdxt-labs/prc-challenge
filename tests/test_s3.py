"""The SigV4 signer, tested against the specification's own published example.

A hand-rolled signer fails silently: the server returns 403 and gives no hint
which of a dozen canonicalisation rules was broken. So the signing key is
checked against AWS's documented worked example, where the expected hex digest
is published, rather than against my own output.

Mutation rehearsals at write time, each restored immediately:
  * drop the region step from `signing_key`      -> test_signing_key_matches RED
  * skip header lower-casing in canonical_request -> test_headers_are_lowercased RED
  * rename the .part file before the body is written -> test_failed_download RED
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prc import s3  # noqa: E402


def test_signing_key_matches_the_published_aws_worked_example():
    """AWS's SigV4 documentation publishes this exact derivation. Any change to
    the four-step chain, its order, or the literals breaks this."""
    key = s3.signing_key("wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
                         "20150830", "us-east-1", "iam")
    assert key.hex() == "c4afb1cc5771d871763a393e44b703571b55cc28424d1a5e86da6ed3c154a4b9"


def test_signing_key_is_sensitive_to_every_input():
    base = s3.signing_key("secret", "20260908", "us-east-1", "s3")
    assert base != s3.signing_key("secret2", "20260908", "us-east-1", "s3")
    assert base != s3.signing_key("secret", "20260909", "us-east-1", "s3")
    assert base != s3.signing_key("secret", "20260908", "eu-west-1", "s3")
    assert base != s3.signing_key("secret", "20260908", "us-east-1", "iam")


def test_headers_are_lowercased_trimmed_and_sorted():
    canonical, signed = s3.canonical_request(
        "GET", "/bucket/key", "",
        {"X-Amz-Date": "20260908T000000Z", "Host": "example.org",
         "x-amz-content-sha256": "  abc   def  "}, "PAYLOADHASH")
    assert signed == "host;x-amz-content-sha256;x-amz-date"
    assert "host:example.org\n" in canonical
    assert "x-amz-content-sha256:abc def\n" in canonical, "internal whitespace must collapse"
    assert canonical.endswith("\nPAYLOADHASH")
    assert canonical.startswith("GET\n/bucket/key\n\n")


def test_credentials_never_appear_in_repr_or_str():
    c = s3.Credentials("AKIAEXAMPLE", "super-secret-value")
    assert "super-secret-value" not in repr(c) and "super-secret-value" not in str(c)
    assert "AKIA" in repr(c), "the key prefix stays visible for debugging"


def test_credentials_load_from_env_then_file(tmp_path):
    env = {"PRC_ACCESS_KEY": "from-env", "PRC_SECRET_KEY": "env-secret"}
    assert s3.load_credentials(env=env).access_key == "from-env"
    f = tmp_path / "creds.env"
    f.write_text("# comment\nPRC_ACCESS_KEY = from-file \nPRC_SECRET_KEY=file-secret\n\n")
    c = s3.load_credentials(path=f, env={})
    assert (c.access_key, c.secret_key) == ("from-file", "file-secret")
    with pytest.raises(FileNotFoundError, match="no credentials"):
        s3.load_credentials(path=tmp_path / "absent.env", env={})
    bad = tmp_path / "bad.env"
    bad.write_text("PRC_ACCESS_KEY=only-one\n")
    with pytest.raises(ValueError, match="missing"):
        s3.load_credentials(path=bad, env={})


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200, fail_after: int | None = None):
        self._body, self.status, self._fail_after, self._sent = body, status, fail_after, 0
    def read(self, n: int = -1) -> bytes:
        if self._fail_after is not None and self._sent >= self._fail_after:
            raise ConnectionError("transfer died mid-stream")
        chunk = self._body[self._sent:self._sent + (n if n and n > 0 else len(self._body))]
        self._sent += len(chunk)
        return chunk
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_request_carries_the_required_signed_headers():
    client = s3.S3(s3.Credentials("AK", "SK"), endpoint="https://example.org")
    req = client._signed_request("GET", "bucket/some key.parquet")
    assert req.full_url == "https://example.org/bucket/some%20key.parquet", "segments escaped, / kept"
    auth = req.headers["Authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AK/")
    assert "SignedHeaders=host;x-amz-content-sha256;x-amz-date" in auth
    assert "/s3/aws4_request" in auth


def test_download_streams_and_only_renames_on_success(tmp_path):
    payload = b"x" * (3 << 20)
    client = s3.S3(s3.Credentials("AK", "SK"), opener=lambda req: _FakeResponse(payload))
    out = client.download("bucket", "f.parquet", tmp_path / "f.parquet")
    assert out.read_bytes() == payload
    assert not (tmp_path / "f.parquet.part").exists(), "the .part must be gone after success"


def test_failed_download_leaves_no_plausible_file(tmp_path):
    """A truncated transfer must never leave something that looks like data —
    the console handed us exactly that failure on 2026-09-08."""
    client = s3.S3(s3.Credentials("AK", "SK"),
                   opener=lambda req: _FakeResponse(b"y" * (3 << 20), fail_after=1 << 20))
    with pytest.raises(ConnectionError):
        client.download("bucket", "f.parquet", tmp_path / "f.parquet")
    assert not (tmp_path / "f.parquet").exists(), "no file at the real name"


LISTING = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents><Key>ranking.parquet</Key><Size>43541234</Size>
    <LastModified>2026-09-04T08:30:00.000Z</LastModified><ETag>"abc123"</ETag></Contents>
  <Contents><Key>submitting.parquet</Key><Size>1677721</Size>
    <LastModified>2026-09-04T08:27:00.000Z</LastModified><ETag>"def456"</ETag></Contents>
</ListBucketResult>"""


def test_list_objects_parses_namespaced_xml():
    client = s3.S3(s3.Credentials("AK", "SK"), opener=lambda req: _FakeResponse(LISTING))
    got = client.list_objects("prc-2026-datasets")
    assert [o["key"] for o in got] == ["ranking.parquet", "submitting.parquet"]
    assert got[0]["bytes"] == 43541234 and got[0]["etag"] == "abc123"
    assert got[1]["last_modified"].startswith("2026-09-04")


class _CapturingOpener:
    """Records the Request it was handed so the test can assert on the wire form."""
    def __init__(self, status: int = 200):
        self.request = None
        self._status = status
    def __call__(self, req):
        self.request = req
        return _FakeResponse(b"", status=self._status)


def test_upload_puts_the_exact_file_bytes_under_the_bucket_and_key(tmp_path):
    """The submission upload path had NO test until 2026-09-09, and it is the call that
    carries the competition entry. Asserts method, key and body on the wire.

    Fails when `upload` sends `method="POST"`, drops the body, or builds the key as
    `key` instead of `f"{bucket}/{key}"` (prc/s3.py:154-157)."""
    src = tmp_path / "merry-quicksand_v2.parquet"
    payload = bytes(range(256)) * 400              # 102,400 bytes, not valid UTF-8
    src.write_bytes(payload)
    opener = _CapturingOpener(status=200)
    client = s3.S3(s3.Credentials("AK", "SK"), endpoint="https://example.org", opener=opener)

    status = client.upload("prc-2026-merry-quicksand", "merry-quicksand_v2.parquet", src)

    assert status == 200
    req = opener.request
    assert req.get_method() == "PUT"
    assert req.full_url == (
        "https://example.org/prc-2026-merry-quicksand/merry-quicksand_v2.parquet")
    assert req.data == payload, "the object body must be the file's bytes, unmodified"


def test_upload_signs_the_payload_rather_than_declaring_it_unsigned(tmp_path):
    """MinIO verifies x-amz-content-sha256 against the body. A stale or UNSIGNED hash is
    accepted by the signer and rejected by the server, which surfaces as a 403 — the
    failure mode that already cost this project a day (see build_submission.py:291).

    Fails when `_signed_request` passes `_UNSIGNED` or `_sha256(b"")` as the payload hash
    for a PUT with a body (prc/s3.py:100)."""
    src = tmp_path / "merry-quicksand_v2.parquet"
    src.write_bytes(b"the bytes that get signed")
    opener = _CapturingOpener()
    client = s3.S3(s3.Credentials("AK", "SK"), endpoint="https://example.org", opener=opener)

    client.upload("bucket", "merry-quicksand_v2.parquet", src)

    sent = opener.request.headers["X-amz-content-sha256"]
    assert sent == hashlib.sha256(b"the bytes that get signed").hexdigest()
    assert sent != s3._UNSIGNED
    assert sent != hashlib.sha256(b"").hexdigest(), "the empty-body hash means the body is unsigned"


def test_upload_signature_changes_when_the_body_changes(tmp_path):
    """Two different submissions must not produce the same Authorization signature; if they
    do, the payload hash is not actually inside the string-to-sign and a corrupted upload
    would still authenticate.

    Fails when the payload hash is hard-coded (e.g. always `_sha256(b"")`) in
    `_signed_request` (prc/s3.py:100-107)."""
    import datetime as dt
    frozen = dt.datetime(2026, 9, 9, 12, 0, 0, tzinfo=dt.timezone.utc)
    client = s3.S3(s3.Credentials("AK", "SK"), endpoint="https://example.org")
    a = client._signed_request("PUT", "bucket/k.parquet", body=b"AAA", now=frozen)
    b = client._signed_request("PUT", "bucket/k.parquet", body=b"BBB", now=frozen)
    sig_a = a.headers["Authorization"].split("Signature=")[1]
    sig_b = b.headers["Authorization"].split("Signature=")[1]
    assert sig_a != sig_b, "same signature for different bodies means the payload is not signed"
    assert len(sig_a) == 64 and set(sig_a) <= set("0123456789abcdef")
