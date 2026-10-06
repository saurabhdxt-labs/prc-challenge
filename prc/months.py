"""The one month-range parser for CLI `--months` arguments (prc/months.py).

`parse_months` turns a comma-separated spec of single months ("YYYY-MM") and inclusive ranges ("A..B") into a
sorted, de-duplicated list of "YYYY-MM" strings. Stdlib only (no pandas): a CLI's argument parsing must not pull in
a heavy import just to validate a string.
"""
from __future__ import annotations

import re

#: today's day universe: all of 2025, plus January and July 2026 (reference_prc_opdi_and_aobt3_rules.md).
DEFAULT_MONTHS = "2025-01..2025-12,2026-01,2026-07"

_MONTH_RE = re.compile(r"^(\d{4})-(\d{2})$")


def _parse_one(piece: str) -> tuple:
    """A single "YYYY-MM" piece -> (year, month) ints, or raise naming `piece`."""
    m = _MONTH_RE.match(piece)
    if not m:
        raise ValueError(f"malformed month {piece!r}")
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        raise ValueError(f"malformed month {piece!r}: month must be 01-12")
    return year, month


def _month_range(lo: tuple, hi: tuple) -> list:
    """Every "YYYY-MM" from `lo` to `hi` inclusive (both (year, month) tuples, lo <= hi already checked)."""
    out = []
    year, month = lo
    while (year, month) <= hi:
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return out


def parse_months(spec: str) -> list:
    """"2025-01..2025-12,2026-01,2026-07" -> sorted unique ["2025-01", ..., "2026-07"].

    Pieces are comma-separated; each is a single month "YYYY-MM" or an inclusive range "A..B". Whitespace around a
    piece or around a range's ".." is stripped; empty pieces (from "a,,b" or trailing/leading commas) are ignored.
    Raises ValueError naming the bad piece for: a malformed month, a month outside 01-12, a range whose end is
    before its start, or a spec that yields no months at all.
    """
    months = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ".." in piece:
            lo_s, hi_s = piece.split("..", 1)
            lo, hi = _parse_one(lo_s.strip()), _parse_one(hi_s.strip())
            if hi < lo:
                raise ValueError(f"range end before its start: {piece!r}")
            months.update(_month_range(lo, hi))
        else:
            _parse_one(piece)
            months.add(piece)
    if not months:
        raise ValueError(f"empty month spec (no months parsed): {spec!r}")
    return sorted(months)
