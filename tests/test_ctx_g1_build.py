"""Tests for the SHIP-CTXG1 build (scripts/analysis/ctx_g1_build.py; plans/PREREG_ship_ctx_g1_2026_09_13.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import ctx_g1_build as B  # noqa: E402


def _lane(rows):
    """rows: (sched UTC string, sp seconds)."""
    sched = pd.to_datetime([r[0] for r in rows], utc=True)
    sp = np.array([r[1] for r in rows], dtype="float64")
    return pd.DataFrame({"sp": sp, "SCHED_TIME_UTC_mvt": sched, "MVT_TIME_UTC_mvt": sched + pd.to_timedelta(sp, unit="s")})


def test_scope_mask_is_the_frozen_band_on_the_same_rome_day():
    """Fails if a bound becomes inclusive at 24,000 or exclusive at 3,600, or if the local-day condition is dropped
    (row 3: 22:30 UTC = 23:30 Rome, + 5,000 s = 00:53 the next Rome day)."""
    f = _lane([("2025-03-10 06:00", 3_600.0), ("2025-03-10 06:00", 23_999.0), ("2025-03-10 06:00", 24_000.0),
               ("2025-03-10 22:30", 5_000.0), ("2025-03-10 06:00", 3_599.0)])
    assert B.scope_mask(f).tolist() == [True, True, False, False, False]


def test_scope_mask_refuses_an_sp_that_is_not_the_clock_difference():
    f = _lane([("2025-03-10 06:00", 5_000.0)])
    f.loc[0, "sp"] = 5_001.0
    with pytest.raises(AssertionError, match="MVT − SCHED"):
        B.scope_mask(f)


def _v13():
    return pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], "TAXITIME_SEC_mvt": np.array([100, 200, 300, 400], dtype="int32")})


def test_splice_writes_only_scope_rows_and_keeps_schema():
    """Fails if the splice writes by position instead of id, or changes order, dtype or any outside row."""
    out = B.splice(_v13(), [3.0, 1.0], [333.0, 111.0])
    assert out.TAXITIME_SEC_mvt.tolist() == [111, 200, 333, 400]
    assert out.dtypes.to_dict() == _v13().dtypes.to_dict() and out.MVT_ID_mvt.tolist() == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize("ids, vals, msg", [
    ([9.0], [5.0], "not in the v13 submission"),
    ([1.0, 1.0], [5.0, 6.0], "not unique"),
    ([1.0], [0.0], "finite integers >= 1"),
    ([1.0], [5.5], "finite integers >= 1"),
])
def test_splice_refuses_bad_inputs(ids, vals, msg):
    with pytest.raises(AssertionError, match=msg):
        B.splice(_v13(), ids, vals)


def _ctx(month, sp, n, med, share):
    return pd.DataFrame({"month": month, "sp": sp, "ctx_n": n, "ctx_med_ms": med, "ctx_late_share": share})


def test_support_checks_bands_per_month_and_coverage():
    """K1 needs >= 10 same-month 2025 rows per sub-band; K2 the 2026 median inside 2025 same-month [p10, p90]; K3 >= 95%
    inside the 2025 [p1, p99]. Fails if a month's 2025 rows are taken from the other month or a bar is loosened."""
    rng = np.random.default_rng(0)
    a = pd.concat([_ctx(1, rng.uniform(3600, 24000, 200), rng.normal(100, 10, 200), rng.normal(2000, 200, 200),
                        rng.uniform(0.1, 0.3, 200)),
                   _ctx(7, np.full(5, 5000.0), np.full(5, 300.0), np.full(5, 9000.0), np.full(5, 0.9))], ignore_index=True)
    b = pd.concat([_ctx(1, np.full(50, 5000.0), np.full(50, 100.0), np.full(50, 2000.0), np.full(50, 0.2)),
                   _ctx(7, np.full(50, 15000.0), np.full(50, 100.0), np.full(50, 2000.0), np.full(50, 0.2))], ignore_index=True)
    r = B.support_checks(a, b)
    assert r[1]["K1"]["3600-10800"]["pass"] is True
    assert r[7]["K1"]["10800-24000"]["pass"] is False and r[7]["K1"]["10800-24000"]["n2025"] == 0
    assert r[1]["K2"]["ctx_n"]["pass"] is True and r[7]["K2"]["ctx_n"]["pass"] is False   # July 2025 median is 300
    assert r["K3"]["ctx_late_share"]["pass"] is True


def test_sha256_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc" * 1000)
    assert B.sha256(p) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_support_checks_k3_fails_when_scope_rows_leave_the_2025_range():
    """End-of-session review: K3 was never seen failing in tests, yet it failed in the real build. 2026 ctx_n all above the
    2025 p99 must FAIL K3 for ctx_n and still PASS for the in-range features. Fails if K3 is hard-coded to pass or its bar is
    loosened below the share observed (0.0 here)."""
    a = _ctx(1, np.full(50, 5000.0), np.linspace(50, 100, 50), np.full(50, 2000.0), np.full(50, 0.2))
    b = _ctx(1, np.full(20, 5000.0), np.full(20, 400.0), np.full(20, 2000.0), np.full(20, 0.2))
    r = B.support_checks(a, b)
    assert r["K3"]["ctx_n"]["share_inside"] == 0.0 and r["K3"]["ctx_n"]["pass"] is False
    assert r["K3"]["ctx_med_ms"]["pass"] is True and r[1]["K2"]["ctx_n"]["pass"] is False
