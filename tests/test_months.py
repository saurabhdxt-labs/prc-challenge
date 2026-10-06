"""Tests for prc/months.py — the one month-range parser the CLIs use for --months.

Each test's docstring ends with the mutation that turns it red (its mutation rehearsal).
"""
from __future__ import annotations

import pathlib

import pytest

import prc.months as M

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_single_month():
    """Fails when the single-month branch returns [] or raises instead of ["2025-01"]."""
    assert M.parse_months("2025-01") == ["2025-01"]


def test_inclusive_range():
    """Fails when the range is exclusive of its end (would drop "2025-03")."""
    assert M.parse_months("2025-01..2025-03") == ["2025-01", "2025-02", "2025-03"]


def test_range_crosses_year_boundary():
    """Fails when the month increment does not roll 12 -> 01 with a year bump."""
    assert M.parse_months("2025-11..2026-02") == ["2025-11", "2025-12", "2026-01", "2026-02"]


def test_mixed_pieces_sorted_and_deduplicated():
    """Fails when dedup is removed (a duplicate "2025-01" would appear twice) or sorting is removed (order would be
    input order, not calendar order)."""
    assert M.parse_months("2025-03,2025-01..2025-02,2025-01") == ["2025-01", "2025-02", "2025-03"]


def test_whitespace_around_pieces_and_range_dots_allowed():
    """Fails when strip() is removed from piece/bound parsing (the regex would reject " 2025-01 ")."""
    assert M.parse_months(" 2025-01 .. 2025-02 , 2025-03 ") == ["2025-01", "2025-02", "2025-03"]


def test_empty_pieces_between_commas_are_ignored():
    """Fails when the empty-piece skip (`if not piece: continue`) is removed (empty piece would raise 'malformed
    month')."""
    assert M.parse_months("2025-01,,2025-02,") == ["2025-01", "2025-02"]


def test_default_months_is_2025_all_plus_2026_janjul():
    """Fails when DEFAULT_MONTHS is edited to drop a month (the 14-month count or a specific month would differ)."""
    months = M.parse_months(M.DEFAULT_MONTHS)
    assert len(months) == 14
    assert months[0] == "2025-01" and months[11] == "2025-12"
    assert "2026-01" in months and "2026-07" in months and "2026-02" not in months


def test_malformed_month_raises_value_error_naming_the_piece():
    """Fails when the regex match guard is removed (a non-matching piece would not raise, or the message would not
    name the bad piece "2025-1")."""
    with pytest.raises(ValueError, match="2025-1\\b"):
        M.parse_months("2025-1")


def test_month_00_raises():
    """Fails when the 1 <= month <= 12 lower-bound check is removed (month 00 would be accepted)."""
    with pytest.raises(ValueError, match="2025-00"):
        M.parse_months("2025-00")


def test_month_13_raises():
    """Fails when the 1 <= month <= 12 upper-bound check is removed (month 13 would be accepted)."""
    with pytest.raises(ValueError, match="2025-13"):
        M.parse_months("2025-13")


def test_range_end_before_start_raises_naming_the_piece():
    """Fails when the `hi < lo` order check on a range is removed: _month_range then returns [] for the inverted
    range (while (year, month) <= hi is false from the first iteration) and the error surfaces from the unrelated
    empty-result guard instead, whose message does not say "before its start" -- this match pins the specific
    error, not just any ValueError naming the spec."""
    with pytest.raises(ValueError, match="before its start.*2025-12\\.\\.2025-01"):
        M.parse_months("2025-12..2025-01")


def test_empty_spec_raises_empty_result():
    """Fails when the final `if not months: raise` guard is removed (an empty spec would return [] instead of
    raising)."""
    with pytest.raises(ValueError, match="empty"):
        M.parse_months("")


def test_all_empty_pieces_raises_empty_result():
    """Fails when the final empty-result guard is removed (a spec of only commas would return [] instead of
    raising)."""
    with pytest.raises(ValueError, match="empty"):
        M.parse_months(",, ,")


def test_no_pandas_import():
    """Fails when a pandas import is added to prc/months.py (the module must be stdlib-only so CLI argument parsing
    does not pull in a heavy dependency just to validate a string)."""
    src = (ROOT / "prc" / "months.py").read_text()
    assert "import pandas" not in src and "from pandas" not in src
