"""The stand/witness feature block must never read the hidden off-block clock.

`BLOCK_TIME_UTC_mvt` is the target: it is present in training data and blanked on every
scored row. A feature that reads it would look excellent in cross-validation and be dead
on the real evaluation file. The serve-time contract is therefore not a style rule, it is
the difference between a real result and a fabricated one.

The permutation test below is the strongest cheap form of that contract: shuffle BLOCK
within the month, rebuild every feature, and require byte-identical output. The row set and
ordering are unaffected because the admissibility filter reads BLOCK's *nullity*, never its
value, and the sort key is take-off.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "raw" / "training_2025-01-01_2025-02-01.parquet"

_spec = importlib.util.spec_from_file_location("stand_ab", ROOT / "scripts" / "stand_ab.py")
stand_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stand_ab)

pytestmark = pytest.mark.skipif(
    not SRC.exists(),
    reason="challenge data is not redistributable and is absent from a clean checkout",
)


@pytest.fixture(scope="module")
def three_days(tmp_path_factory):
    """The first three days of January 2025 — enough rows for real stand chains."""
    t = pq.read_table(SRC, columns=stand_ab.COLS).to_pandas()
    t = t[t.MVT_TIME_UTC_mvt < pd.Timestamp("2025-01-04", tz="UTC")]
    p = tmp_path_factory.mktemp("stand") / "slice.parquet"
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False), p)
    return p


def test_target_identity_holds_exactly(three_days):
    """y == proxy - delta, to floating-point exactness.

    Fails when the delta reparameterisation is wired up wrong — e.g. if `delta` were
    built as AOBT_3 - BLOCK_TIME (sign flipped) the residual becomes 2*delta, not zero.
    """
    o = stand_ab.build_month(three_days)
    resid = o.y.to_numpy() - (o.proxy.to_numpy() - o.delta.to_numpy())
    assert len(o) > 5_000, f"slice too small to be meaningful: {len(o)} rows"
    # exact: all three quantities are differences of the same second-resolution stamps
    assert np.nanmax(np.abs(resid)) < 1e-6, f"identity broken, max |resid| = {np.nanmax(np.abs(resid))}"


def test_no_feature_reads_the_hidden_off_block_clock(three_days):
    """Permuting BLOCK_TIME must leave every feature byte-identical; only `delta` moves.

    Fails when any feature is derived from BLOCK. Rehearsed 2026-09-08 by adding
    `o["leak"] = blk - sch` to build_month and appending "leak" to STAND_BLOCK: the test
    went red naming `leak` on 100% of rows. Restored, green.
    """
    base = stand_ab.build_month(three_days)

    t = pq.read_table(three_days, columns=stand_ab.COLS).to_pandas()
    m = (t.PHASE_mvt == "DEP") & t.BLOCK_TIME_UTC_mvt.notna()
    rng = np.random.default_rng(0)
    t.loc[m, "BLOCK_TIME_UTC_mvt"] = rng.permutation(t.loc[m, "BLOCK_TIME_UTC_mvt"].to_numpy())
    p2 = three_days.parent / "permuted.parquet"
    pq.write_table(pa.Table.from_pandas(t, preserve_index=False), p2)
    perm = stand_ab.build_month(p2)

    assert len(base) == len(perm), "permuting BLOCK changed the admissible row set"

    feats = [c for c in stand_ab.BASELINE_FEATS + stand_ab.STAND_BLOCK
             if c in base.columns and c not in stand_ab.STAND_INFOLD]
    assert len(feats) > 40, f"only {len(feats)} features checked - list wired up wrong?"

    moved = []
    for c in feats:
        a, z = base[c].to_numpy(), perm[c].to_numpy()
        if a.dtype.kind in "fi":
            if not np.allclose(a, z, equal_nan=True, atol=1e-9):
                moved.append(c)
        elif not (a == z).all():
            moved.append(c)
    assert not moved, f"these features read the hidden BLOCK clock: {moved}"

    # and the control: delta MUST move, otherwise the permutation did nothing and the
    # test above would pass vacuously
    assert not np.allclose(base.delta.to_numpy(), perm.delta.to_numpy(), equal_nan=True), \
        "delta did not move under permutation - the test proved nothing"


def test_stand_block_is_populated_and_carries_the_documented_signal(three_days):
    """The witness features must be non-null and reproduce the enrichment in the report.

    Fails when the (airport, stand) key collapses, so arrivals are matched to the wrong
    stand and every witness count is meaningless. Rehearsed 2026-09-08 by mutating the key
    to `a_key = a_apt` (airport only): build_month's own `len(aidx) > 100` guard fired and
    this test went red.

    Note on the historical NUL bug (reports/STAND_OCCUPANCY.md): it arose in *pandas Arrow*
    string concatenation, which truncates at a NUL. The keys here are built from numpy
    object arrays via `sarr`, where concatenation is pure Python and NUL-safe, so switching
    the separator back to "\\x00" does NOT reproduce it — that mutation was rehearsed and
    stayed green. The structural guard against the bug class is the assertion in
    build_month, which this test exercises.
    """
    o = stand_ab.build_month(three_days)
    assert o.gapa.notna().mean() > 0.90, f"gapa mostly null: {o.gapa.notna().mean():.3f}"
    assert o.prev_arr_gap.notna().mean() > 0.90
    assert o.n_2h_mvt.notna().mean() > 0.90
    assert o.n_2h_mvt.max() > 0, "no arrivals ever found at any stand - key collapsed?"

    tail = o.delta < -600
    near = o.gapa <= 600
    assert tail.mean() > 0, "no tail rows in the slice"
    # documented on the full year: P(tail | gapa<=600) ~ 41% against a ~3% base rate
    assert tail[near].mean() > 4 * tail.mean(), (
        f"enrichment gone: P(tail|gapa<=600)={tail[near].mean():.3f} "
        f"vs base {tail.mean():.3f}")
