"""Ingest moves competition files into the project and records what arrived.

This is load-bearing despite being plumbing: the manifest it writes is the only
record of WHICH bytes we trained on. The evaluation-side files (`submitting`,
`ranking`) were regenerated on 2026-09-04, three days after the challenge
opened, so they can change again mid-competition. A submission built against a
stale copy would be keyed to rows that no longer exist.

Mutation rehearsals performed at write time, each restored immediately:
  * `sha256` -> constant string           -> test_manifest_records_content_hash RED
  * drop the `exists()` refusal           -> test_never_overwrites RED
  * accept any file without parsing it    -> test_rejects_a_file_that_is_not_parquet RED
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ingest_downloads as ing  # noqa: E402


def _parquet(path: Path, rows: int = 3) -> Path:
    pd.DataFrame({"a": range(rows)}).to_parquet(path, index=False)
    return path


def test_manifest_records_content_hash_size_and_source_mtime(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _parquet(src / "training_2025-01-01_2025-02-01.parquet")
    manifest = ing.ingest(src, dst, expected=["training_2025-01-01_2025-02-01.parquet"])
    entry = manifest["files"]["training_2025-01-01_2025-02-01.parquet"]
    assert len(entry["sha256"]) == 64
    assert entry["bytes"] == (dst / "training_2025-01-01_2025-02-01.parquet").stat().st_size
    assert entry["source_mtime_utc"].endswith("Z")
    assert entry["rows"] == 3 and entry["columns"] == ["a"]
    # the hash must be OF THE CONTENT: a different file must hash differently
    other = _parquet(src / "other.parquet", rows=99)
    assert ing.sha256_of(other) != entry["sha256"]


def test_never_overwrites_an_already_ingested_file(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _parquet(src / "a.parquet")
    ing.ingest(src, dst, expected=["a.parquet"])
    _parquet(src / "a.parquet", rows=999)          # a DIFFERENT file, same name
    with pytest.raises(FileExistsError, match="already ingested"):
        ing.ingest(src, dst, expected=["a.parquet"])
    # and the original survived untouched
    assert len(pd.read_parquet(dst / "a.parquet")) == 3


def test_rejects_a_file_that_is_not_parquet(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "broken.parquet").write_bytes(b"PK\x03\x04 not a parquet at all")
    with pytest.raises(ValueError, match="not readable as parquet"):
        ing.ingest(src, dst, expected=["broken.parquet"])
    assert not (dst / "broken.parquet").exists(), "a rejected file must not be left behind"


def test_reports_missing_and_unexpected_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _parquet(src / "present.parquet")
    _parquet(src / "surprise.parquet")
    m = ing.ingest(src, dst, expected=["present.parquet", "absent.parquet"])
    assert m["missing"] == ["absent.parquet"]
    assert m["unexpected"] == ["surprise.parquet"]
    assert "surprise.parquet" not in m["files"], "only expected files are ingested"
    assert m["complete"] is False


def test_complete_only_when_every_expected_file_arrived(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    _parquet(src / "one.parquet")
    _parquet(src / "two.parquet")
    m = ing.ingest(src, dst, expected=["one.parquet", "two.parquet"])
    assert m["complete"] is True and m["missing"] == []
    written = json.loads((dst / "MANIFEST.json").read_text())
    assert written["files"].keys() == {"one.parquet", "two.parquet"}


def test_expected_file_list_matches_the_bucket_listing():
    """The 14 names are pinned so a silently-renamed or added file is caught."""
    assert len(ing.EXPECTED) == 14
    assert "ranking.parquet" in ing.EXPECTED and "submitting.parquet" in ing.EXPECTED
    months = [f for f in ing.EXPECTED if f.startswith("training_")]
    assert len(months) == 12, "2025 has twelve months of training data"
    assert months == sorted(months), "kept in chronological order"
