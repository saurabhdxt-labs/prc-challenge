"""ingest_downloads — move competition parquet files into the project and record
exactly which bytes arrived.

The manifest is the point. The evaluation-side files (`submitting.parquet`,
`ranking.parquet`) were regenerated on 2026-09-04, three days after the
challenge opened, so the organisers can reissue them. A submission built
against a stale copy would be keyed to rows that no longer exist. Every
ingested file therefore carries its content hash, its size, its source
modification time, and its row and column shape.

Ingest is first-write-final: a file already in place is never overwritten. To
take a reissued file, delete the local copy deliberately and re-run, which
leaves the change visible in the manifest history rather than silent.

Usage:
  python3.11 scripts/ingest_downloads.py [--src ~/Downloads] [--dst data/raw]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

#: The bucket listing as read from the MinIO console on 2026-09-08. Pinned so a
#: renamed, added or withdrawn file is caught rather than silently absorbed.
EXPECTED: tuple[str, ...] = (
    "ranking.parquet",
    "submitting.parquet",
    "training_2025-01-01_2025-02-01.parquet",
    "training_2025-02-01_2025-03-01.parquet",
    "training_2025-03-01_2025-04-01.parquet",
    "training_2025-04-01_2025-05-01.parquet",
    "training_2025-05-01_2025-06-01.parquet",
    "training_2025-06-01_2025-07-01.parquet",
    "training_2025-07-01_2025-08-01.parquet",
    "training_2025-08-01_2025-09-01.parquet",
    "training_2025-09-01_2025-10-01.parquet",
    "training_2025-10-01_2025-11-01.parquet",
    "training_2025-11-01_2025-12-01.parquet",
    "training_2025-12-01_2026-01-01.parquet",
)


def sha256_of(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def describe(path: Path) -> dict:
    """Shape and provenance of one parquet file. Raises if it will not parse —
    a truncated download is the failure mode this exists to catch, and the
    console has already produced one truncated archive today."""
    try:
        meta = pq.ParquetFile(path)
        rows = meta.metadata.num_rows
        columns = list(meta.schema_arrow.names)
    except Exception as exc:                        # noqa: BLE001 - any parse failure is the same finding
        raise ValueError(f"{path.name} is not readable as parquet: {exc}") from None
    stat = path.stat()
    return {"sha256": sha256_of(path), "bytes": stat.st_size,
            "source_mtime_utc": _utc(stat.st_mtime), "rows": rows,
            "columns": columns, "n_columns": len(columns)}


def ingest(src: Path, dst: Path, expected=EXPECTED, log=print) -> dict:
    src, dst = Path(src), Path(dst)
    expected = list(expected)
    dst.mkdir(parents=True, exist_ok=True)

    present = {p.name for p in src.glob("*.parquet")}
    missing = [name for name in expected if name not in present]
    unexpected = sorted(present - set(expected))

    files: dict[str, dict] = {}
    for name in expected:
        source = src / name
        if not source.exists():
            continue
        target = dst / name
        if target.exists():
            raise FileExistsError(
                f"{name} was already ingested at {target}. Ingest is "
                f"first-write-final; delete it deliberately to take a reissued copy.")
        # Describe BEFORE copying: a file that will not parse never lands.
        info = describe(source)
        shutil.copy2(source, target)
        info["ingested_utc"] = _utc(time.time())
        files[name] = info
        log(f"  {name:44} {info['bytes'] / 1048576:7.1f} MiB  "
            f"{info['rows']:>10,} rows x {info['n_columns']:>3} cols")

    manifest = {"expected": expected, "files": files, "missing": missing,
                "unexpected": unexpected, "complete": not missing,
                "source": str(src), "written_utc": _utc(time.time())}
    (dst / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path.home() / "Downloads")
    ap.add_argument("--dst", type=Path, default=root / "data" / "raw")
    a = ap.parse_args(argv)
    m = ingest(a.src, a.dst)
    print(f"\ningested {len(m['files'])} of {len(m['expected'])}")
    if m["missing"]:
        print(f"MISSING ({len(m['missing'])}): " + ", ".join(m["missing"]))
    if m["unexpected"]:
        print(f"unexpected parquet files in {a.src} (ignored): " + ", ".join(m["unexpected"]))
    return 0 if m["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
