"""Minimal S3 client: AWS Signature Version 4 over the standard library.

Written instead of taking a dependency. The whole surface needed here is list,
get and put against one MinIO endpoint, and SigV4 is a published, stable
specification, so a third-party SDK would add supply-chain surface to a public
repository in exchange for convenience we do not need.

Credentials are read from the environment or from a file outside the repository
tree. Nothing here ever writes a credential to disk, to a log line, or to an
exception message.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_UNSIGNED = "UNSIGNED-PAYLOAD"
_ALGO = "AWS4-HMAC-SHA256"
DEFAULT_ENDPOINT = "https://s3.opensky-network.org"
DEFAULT_REGION = "us-east-1"          # MinIO ignores it; SigV4 requires a value


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def signing_key(secret: str, date_stamp: str, region: str, service: str = "s3") -> bytes:
    """The SigV4 four-step derived key. Split out so it is directly testable."""
    k = _hmac(f"AWS4{secret}".encode("utf-8"), date_stamp)
    k = _hmac(k, region)
    k = _hmac(k, service)
    return _hmac(k, "aws4_request")


def canonical_request(method: str, path: str, query: str, headers: dict[str, str],
                      payload_hash: str) -> tuple[str, str]:
    """(canonical request, signed header list). Headers are lower-cased and
    sorted, which is the part that silently breaks a hand-rolled signer."""
    items = sorted((k.lower(), " ".join(str(v).split())) for k, v in headers.items())
    canonical_headers = "".join(f"{k}:{v}\n" for k, v in items)
    signed_headers = ";".join(k for k, _ in items)
    canonical = "\n".join([method, path, query, canonical_headers, signed_headers, payload_hash])
    return canonical, signed_headers


@dataclass(frozen=True)
class Credentials:
    access_key: str
    secret_key: str

    def __repr__(self) -> str:                      # never leak the secret into a traceback
        return f"Credentials(access_key={self.access_key[:4]}…, secret_key=<hidden>)"


def load_credentials(path: "Path | None" = None, env: "dict | None" = None) -> Credentials:
    """Environment first, then a file of KEY=value lines outside the repo."""
    env = os.environ if env is None else env
    access, secret = env.get("PRC_ACCESS_KEY"), env.get("PRC_SECRET_KEY")
    if not (access and secret):
        path = Path(path or Path.home() / ".prc-challenge.env")
        if not path.exists():
            raise FileNotFoundError(
                f"no credentials: set PRC_ACCESS_KEY and PRC_SECRET_KEY, or create {path}")
        values: dict[str, str] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                values[k.strip()] = v.strip()
        access, secret = values.get("PRC_ACCESS_KEY"), values.get("PRC_SECRET_KEY")
        if not (access and secret):
            raise ValueError(f"{path} is missing PRC_ACCESS_KEY or PRC_SECRET_KEY")
    return Credentials(access, secret)


class S3:
    def __init__(self, credentials: Credentials, endpoint: str = DEFAULT_ENDPOINT,
                 region: str = DEFAULT_REGION, opener=None):
        self.credentials, self.endpoint, self.region = credentials, endpoint.rstrip("/"), region
        self.host = urllib.parse.urlsplit(self.endpoint).netloc
        self._opener = opener or urllib.request.urlopen

    def _signed_request(self, method: str, key: str, *, query: str = "", body: bytes = b"",
                        now: "_dt.datetime | None" = None) -> urllib.request.Request:
        now = now or _dt.datetime.now(_dt.timezone.utc)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        date_stamp = now.strftime("%Y%m%d")
        # Each path segment is escaped, but the separators are not.
        path = "/" + "/".join(urllib.parse.quote(p, safe="") for p in key.split("/") if p != "")
        payload_hash = _sha256(body) if body else _sha256(b"")
        headers = {"host": self.host, "x-amz-date": amz_date,
                   "x-amz-content-sha256": payload_hash}
        canonical, signed_headers = canonical_request(method, path, query, headers, payload_hash)
        scope = f"{date_stamp}/{self.region}/s3/aws4_request"
        to_sign = "\n".join([_ALGO, amz_date, scope, _sha256(canonical.encode("utf-8"))])
        signature = hmac.new(signing_key(self.credentials.secret_key, date_stamp, self.region),
                             to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        headers["Authorization"] = (
            f"{_ALGO} Credential={self.credentials.access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}")
        url = f"{self.endpoint}{path}" + (f"?{query}" if query else "")
        return urllib.request.Request(url, data=body or None, headers=headers, method=method)

    def list_objects(self, bucket: str) -> list[dict]:
        """Every object in a bucket, following continuation tokens."""
        out: list[dict] = []
        token = None
        while True:
            params = {"list-type": "2"}
            if token:
                params["continuation-token"] = token
            query = "&".join(f"{k}={urllib.parse.quote(v, safe='')}"
                             for k, v in sorted(params.items()))
            with self._opener(self._signed_request("GET", bucket, query=query)) as r:
                root = ET.fromstring(r.read())
            ns = {"s3": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
            find = (lambda e, t: e.find(f"s3:{t}", ns)) if ns else (lambda e, t: e.find(t))
            for c in (root.findall("s3:Contents", ns) if ns else root.findall("Contents")):
                out.append({"key": find(c, "Key").text,
                            "bytes": int(find(c, "Size").text),
                            "last_modified": find(c, "LastModified").text,
                            "etag": (find(c, "ETag").text or "").strip('"')})
            truncated = find(root, "IsTruncated")
            if truncated is None or truncated.text != "true":
                return out
            nxt = find(root, "NextContinuationToken")
            token = nxt.text if nxt is not None else None
            if token is None:
                return out

    def download(self, bucket: str, key: str, dest: Path) -> Path:
        """Stream one object to disk. Writes to a .part file and renames on
        success, so a failed transfer never leaves a plausible-looking file."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(dest.suffix + ".part")
        with self._opener(self._signed_request("GET", f"{bucket}/{key}")) as r, open(part, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        part.rename(dest)
        return dest

    def upload(self, bucket: str, key: str, source: Path) -> int:
        body = Path(source).read_bytes()
        with self._opener(self._signed_request("PUT", f"{bucket}/{key}", body=body)) as r:
            return r.status
