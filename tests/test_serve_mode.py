"""Serve-mode features must be the training features, computed on rows that carry no label.

`build_month(path, serve=True)` is what produces the design matrix for the 339,551 scored
departures that have a Network Manager off-block. Those rows have `TAXITIME_SEC_mvt` and
`BLOCK_TIME_UTC_mvt` blanked, so the training-mode row filter (which requires both) admits
none of them. The serve filter drops that requirement and NOTHING else may change: a feature
that is computed differently at serve time is a train/serve skew that cross-validation can
never see and the leaderboard always pays for.

Three invariants, each measured on real rows rather than asserted from the code:

  1. On one training month, serve mode reproduces training mode. Per-row features are
     byte-identical; the only admissible differences are in DEP-stream window features, and
     only because the serve filter admits a handful of extra reference rows (TAXITIME<=0).
     Pruning those rows from the input makes the two modes identical on every column, which
     is the causal proof that nothing else differs.
  2. On the evaluation file, serve mode yields exactly the matched scored rows, built one
     calendar month at a time (the file holds January AND July 2026, every training cache is
     one month, and `sched_day` counts departures per file), with no column dead that is
     alive in training.
  3. No serve-mode feature reads the row's own hidden clocks: permuting BLOCK_TIME and
     TAXITIME across departures leaves every serve-mode column byte-identical.
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
RAW = ROOT / "data" / "raw"
MARCH = RAW / "training_2025-03-01_2025-04-01.parquet"
JAN = RAW / "training_2025-01-01_2025-02-01.parquet"
RANKING = RAW / "ranking.parquet"
TEMPLATE = RAW / "submitting.parquet"

_spec = importlib.util.spec_from_file_location("stand_ab", ROOT / "scripts" / "stand_ab.py")
stand_ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stand_ab)

pytestmark = pytest.mark.skipif(
    not (MARCH.exists() and JAN.exists() and RANKING.exists() and TEMPLATE.exists()),
    reason="challenge data is not redistributable and is absent from a clean checkout",
)

#: the mechanism behind every window feature, so the parity test knows what may legitimately
#: differ when the serve filter admits extra DEPARTURE reference rows. ARR-stream features
#: read arrivals only and must not move at all.
DEP_STREAM = ["n_push", "rwy_b30", "apt_b30", "sched_prev60", "sched_next60", "sched_day",
              "rwy_rate", "dep_dur", "rwy_dur", "prev_dep_gap"]
ARR_STREAM = ["arr_b30", "arr_dur", "arr_ob60_taxiin", "arr_ob60_cdiff", "arr_ob180_cdiff",
              "prev_arr_gap", "prev_arr_taxiin", "prev_arr_cdiff", "gapa", "gaps",
              "n_sch_aobt", "n_aobt_mvt", "n_2h_mvt", "witness", "ty_match", "gapa_pos"]
PER_ROW = stand_ab.BASE + stand_ab.ENC_KEYS + ["ap", "month"]
LABELS = ["delta", "y"]
MIN_WINDOW_MATCH = 0.995


def _same(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Row-wise equality that treats NaN == NaN, for float and string columns alike."""
    x, z = a.to_numpy(), b.to_numpy()
    if a.dtype.kind in "fi" and b.dtype.kind in "fi":
        x, z = x.astype(float), z.astype(float)
        return (x == z) | (np.isnan(x) & np.isnan(z))
    return x == z


def _aligned(train: pd.DataFrame, serve: pd.DataFrame) -> pd.DataFrame:
    """Serve rows re-ordered to the training rows' MVT_IDs (every training row must exist)."""
    assert train.MVT_ID_mvt.is_unique and serve.MVT_ID_mvt.is_unique
    missing = set(train.MVT_ID_mvt) - set(serve.MVT_ID_mvt)
    assert not missing, f"{len(missing)} training rows are absent from the serve build"
    return serve.set_index("MVT_ID_mvt").loc[train.MVT_ID_mvt.to_numpy()].reset_index()


@pytest.fixture(scope="module")
def march_pair():
    """The same month built both ways. ~5 s; shared by the parity tests."""
    return stand_ab.build_month(MARCH, serve=False), stand_ab.build_month(MARCH, serve=True)


def _slice(src: pathlib.Path, lo: str, hi: str) -> pd.DataFrame:
    t = pq.read_table(src, columns=stand_ab.COLS).to_pandas()
    return t[(t.MVT_TIME_UTC_mvt >= pd.Timestamp(lo, tz="UTC"))
             & (t.MVT_TIME_UTC_mvt < pd.Timestamp(hi, tz="UTC"))]


def _write(frame: pd.DataFrame, path: pathlib.Path) -> pathlib.Path:
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
    return path


def test_every_output_column_is_classified_for_the_parity_test(march_pair):
    """A new feature must be assigned a mechanism before it can ship.

    Fails when build_month grows a column that is in none of PER_ROW / DEP_STREAM /
    ARR_STREAM / LABELS / MVT_ID_mvt — an unclassified column would silently escape the
    parity assertions below. Rehearsed 2026-09-09: `o["leak"] = 0.0` in build_features
    went RED naming `leak`.
    """
    train, _ = march_pair
    known = set(PER_ROW) | set(DEP_STREAM) | set(ARR_STREAM) | set(LABELS) | {"MVT_ID_mvt"}
    unclassified = [c for c in train.columns if c not in known]
    assert not unclassified, f"classify these columns for the serve parity test: {unclassified}"
    absent = [c for c in known if c not in train.columns]
    assert not absent, f"the parity test names columns build_month no longer emits: {absent}"


def test_serve_mode_reproduces_training_mode_on_the_same_month(march_pair, capsys):
    """Per-row and arrival-stream features are byte-identical; departure-stream windows
    match on >= 99.5% of rows; `sched_day` differs by exactly the number of extra
    reference rows the serve filter admitted at that airport; labels are NaN.

    Fails when the serve path computes any per-row feature differently, when it sorts
    or filters the arrival stream differently, or when it stops emitting MVT_ID_mvt.
    Rehearsed 2026-09-09, each RED: a serve-only `arr[arr.TAXITIME_SEC_mvt > 60]` arrival
    filter; dropping the `o["MVT_ID_mvt"]` line; `o["sp"] = mvt - sch + (1.0 if serve
    else 0.0)`.
    """
    train, serve = march_pair
    assert len(serve) > len(train), "the serve filter admitted no extra rows on March - the " \
                                    "window tolerance below would be untested"
    aligned = _aligned(train, serve)

    exact = [c for c in PER_ROW + ARR_STREAM if c in train.columns]
    broken = {c: float(1 - _same(train[c], aligned[c]).mean()) for c in exact
              if not _same(train[c], aligned[c]).all()}
    assert not broken, "serve mode changed per-row/arrival features: " + ", ".join(
        f"{c} differs on {100 * v:.3f}% of rows" for c, v in broken.items())

    extra = serve[~serve.MVT_ID_mvt.isin(set(train.MVT_ID_mvt))]
    n_extra = extra.groupby("ap").size()
    expected_day = train.sched_day.to_numpy() + train.ap.map(n_extra).fillna(0).to_numpy()
    assert np.array_equal(aligned.sched_day.to_numpy(), expected_day), \
        "sched_day must be the training count plus the extra reference rows at that airport"

    rates = {c: float(_same(train[c], aligned[c]).mean()) for c in DEP_STREAM
             if c != "sched_day"}
    print("\nDEP-stream window features, serve vs training on March 2025 "
          f"({len(train):,} rows; {len(extra)} extra reference rows admitted):")
    for c, r in rates.items():
        print(f"  {c:14s} equal on {100 * r:7.3f}% of rows")
    low = {c: r for c, r in rates.items() if r < MIN_WINDOW_MATCH}
    assert not low, f"window features below {MIN_WINDOW_MATCH:.1%} agreement: {low}"

    assert serve.delta.isna().all() and serve.y.isna().all(), "labels must be NaN in serve mode"
    assert train.delta.notna().all() and train.y.notna().all(), "labels missing in training mode"
    assert "MVT_ID_mvt" in train.columns and "MVT_ID_mvt" in serve.columns


def test_serve_mode_differences_are_caused_only_by_the_extra_reference_rows(march_pair,
                                                                             tmp_path):
    """Causal proof: remove the rows only the serve filter admits, and the two modes agree on
    EVERY column, not just the per-row ones.

    Fails when serve mode differs from training mode for any reason other than the extra
    reference rows - e.g. a different sort key, a different arrival filter, or a feature
    that reads a column the serve path fills differently. Rehearsed 2026-09-09: sorting on
    `"AOBT_3_flt" if serve else "MVT_TIME_UTC_mvt"` went RED on the row order.
    """
    train, serve = march_pair
    extra_ids = set(serve.MVT_ID_mvt) - set(train.MVT_ID_mvt)
    assert extra_ids, "no extra rows on March - the pruning proves nothing"
    t = pq.read_table(MARCH, columns=stand_ab.COLS).to_pandas()
    pruned = stand_ab.build_month(
        _write(t[~t.MVT_ID_mvt.isin(extra_ids)], tmp_path / "pruned.parquet"), serve=True)

    assert len(pruned) == len(train)
    assert np.array_equal(pruned.MVT_ID_mvt.to_numpy(), train.MVT_ID_mvt.to_numpy()), \
        "row order must be the take-off order in both modes"
    moved = [c for c in train.columns if c not in LABELS and not _same(train[c], pruned[c]).all()]
    assert not moved, f"serve mode differs beyond the extra reference rows on: {moved}"


def test_serve_mode_on_the_ranking_file_yields_every_matched_row_with_no_dead_column(
        march_pair, capsys):
    """Exactly the 339,551 scored departures with an NM off-block, built per calendar month,
    and every feature that is alive in training is alive here.

    A column that is NaN on the scored rows but populated in training is a serve-time bug:
    the model would consult a feature the evaluation file cannot supply. Rates are reported
    per column; zero-NaN training columns must be zero-NaN here, and a column with natural
    NaN in training may not more than triple its rate.

    Fails when the serve filter is wrong (row count), when the file is built as one unit
    instead of per month (`sched_day` is then the two-month count), or when any feature's
    inputs are blank on the evaluation rows. Rehearsed 2026-09-09, each RED:
    `parts = [build_features(t, serve=True)]` in build_ranking; dropping
    `t.AOBT_3_flt.notna()` from the serve filter (344,841 rows); `o["eobt_p"] = np.nan if
    serve else mvt - eobt` (flagged DEAD AT SERVE).
    """
    train, _ = march_pair
    rank = stand_ab.build_ranking(RANKING)

    raw = pq.read_table(RANKING, columns=["MVT_ID_mvt", "PHASE_mvt", "AOBT_3_flt"]).to_pandas()
    ids = set(pq.read_table(TEMPLATE, columns=["MVT_ID_mvt"]).to_pandas().MVT_ID_mvt)
    matched = raw[(raw.PHASE_mvt == "DEP") & raw.AOBT_3_flt.notna() & raw.MVT_ID_mvt.isin(ids)]
    assert len(rank) == 339_551, f"expected 339,551 matched rows, got {len(rank):,}"
    assert rank.MVT_ID_mvt.is_unique
    assert set(rank.MVT_ID_mvt) == set(matched.MVT_ID_mvt), "row set != raw matched ids"
    assert set(rank.month.unique()) == {1, 7}, f"months {sorted(rank.month.unique())}"

    per_month = rank.groupby(["ap", "month"]).size()
    got = rank.set_index(["ap", "month"]).sched_day
    assert (got.to_numpy() == per_month.reindex(got.index).to_numpy()).all(), \
        "sched_day must count departures per (airport, calendar month), as in training"

    assert rank.delta.isna().all() and rank.y.isna().all()

    cols = stand_ab.BASE + stand_ab.SURF + stand_ab.STANDHIST
    print(f"\nNaN rate per feature, training (March 2025, n={len(train):,}) vs "
          f"ranking serve build (n={len(rank):,}):")
    dead = {}
    for c in cols + [c for c in stand_ab.STAND_BLOCK if c in rank.columns]:
        r_tr, r_rk = float(train[c].isna().mean()), float(rank[c].isna().mean())
        flag = ""
        if c in cols and ((r_tr == 0 and r_rk > 0) or (r_tr > 0 and r_rk > 3 * r_tr)):
            dead[c] = (r_tr, r_rk)
            flag = "   <-- DEAD AT SERVE"
        print(f"  {c:18s} train {100 * r_tr:7.3f}%   ranking {100 * r_rk:7.3f}%{flag}")
    assert not dead, "features dead on the evaluation rows: " + ", ".join(
        f"{c} (train {100 * a:.3f}% NaN, ranking {100 * b:.3f}% NaN)" for c, (a, b) in dead.items())


def test_serve_mode_features_never_read_the_row_own_block_or_taxitime(tmp_path):
    """Permuting BLOCK_TIME and TAXITIME across departures, or blanking both on every
    departure as the evaluation file does, leaves serve output byte-identical.

    Serve mode must not read either column on a departure: both are blank on every scored
    row, so any dependence is a feature that is alive in the test harness and dead in
    production. The blanked variant is the load-bearing one: a serve filter that reads
    `BLOCK.notna()` empties the row set there, whereas the permutation cannot see it (no
    January departure has a null BLOCK, so a permutation never moves one). The control
    proves the permutation is real: training mode, which reads both, must change under it.

    Fails when the serve path derives any column from a departure's BLOCK_TIME or
    TAXITIME, or when the serve filter reads either of them. Rehearsed 2026-09-09: adding
    `& t.BLOCK_TIME_UTC_mvt.notna()` to the serve filter stayed GREEN under the
    permutation alone and went RED once the blanked variant was added; adding
    `TAXITIME * 1e-3` to `hr` went RED under the permutation.
    """
    base_path = _write(_slice(JAN, "2025-01-01", "2025-01-04"), tmp_path / "slice.parquet")
    base = stand_ab.build_month(base_path, serve=True)

    t = pq.read_table(base_path, columns=stand_ab.COLS).to_pandas()
    dep = (t.PHASE_mvt == "DEP").to_numpy()
    rng = np.random.default_rng(0)
    perm = rng.permutation(dep.sum())
    for col in ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"):
        vals = t.loc[dep, col].to_numpy()
        t.loc[dep, col] = vals[perm]
    perm_path = _write(t, tmp_path / "permuted.parquet")

    blank = pq.read_table(base_path, columns=stand_ab.COLS).to_pandas()
    blank.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    blank.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    assert blank.loc[dep, "BLOCK_TIME_UTC_mvt"].isna().all()
    blank_path = _write(blank, tmp_path / "blanked.parquet")

    for name, path in (("permuted", perm_path), ("blanked", blank_path)):
        served = stand_ab.build_month(path, serve=True)
        assert len(served) == len(base) > 5_000, f"{name}: the serve row set changed"
        moved = [c for c in base.columns if not _same(base[c], served[c]).all()]
        assert not moved, f"{name}: serve-mode columns read the departure's own hidden " \
                          f"clocks: {moved}"

    trained = stand_ab.build_month(perm_path, serve=False)
    ref = stand_ab.build_month(base_path, serve=False)
    assert len(trained) != len(ref) or not np.allclose(
        trained.delta.to_numpy(), ref.delta.to_numpy(), equal_nan=True), \
        "training mode did not change under the permutation - the test proved nothing"


def test_cmd_cache_ranking_writes_only_the_ranking_cache_one_month_at_a_time(tmp_path,
                                                                             monkeypatch):
    """`cmd_cache(ranking=True)` writes data/cache_stand/ranking.parquet from
    data/raw/ranking.parquet in serve mode, per calendar month, and touches nothing else.

    The fixture holds two calendar months (Jan and Mar 2025, two days each) with BLOCK and
    TAXITIME blanked on departures, as on the real file. The control shows that a single-unit
    serve build of the same file gives `sched_day` = two-month totals, so the per-month split
    is what the command does rather than what the fixture happens to permit.

    Fails when the command builds the file as one unit, forgets serve mode (labels present),
    drops MVT_ID_mvt, writes anything beside ranking.parquet, or lets --smoke through.
    Rehearsed 2026-09-09, each RED: `o = build_month(RAW / "ranking.parquet", serve=True)`
    in cmd_cache; removing the `if smoke: raise ValueError` guard.
    """
    raw, cache = tmp_path / "raw", tmp_path / "cache"
    raw.mkdir()
    two = pd.concat([_slice(JAN, "2025-01-01", "2025-01-03"),
                     _slice(MARCH, "2025-03-01", "2025-03-03")], ignore_index=True)
    dep = two.PHASE_mvt == "DEP"
    two.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    two.loc[dep, "TAXITIME_SEC_mvt"] = np.nan
    _write(two, raw / "ranking.parquet")
    monkeypatch.setattr(stand_ab, "RAW", raw)
    monkeypatch.setattr(stand_ab, "CACHE", cache)

    stand_ab.cmd_cache(smoke=False, ranking=True)

    assert sorted(p.name for p in cache.iterdir()) == ["ranking.parquet"]
    o = pd.read_parquet(cache / "ranking.parquet")
    assert "MVT_ID_mvt" in o.columns and o.MVT_ID_mvt.is_unique
    assert o.delta.isna().all() and o.y.isna().all()
    assert set(o.month.unique()) == {1, 3}
    per_month = o.groupby(["ap", "month"]).size()
    got = o.set_index(["ap", "month"]).sched_day
    assert (got.to_numpy() == per_month.reindex(got.index).to_numpy()).all()

    single = stand_ab.build_month(raw / "ranking.parquet", serve=True)
    totals = single.groupby("ap").size()
    assert (single.sched_day.to_numpy() == single.ap.map(totals).to_numpy()).all()
    assert not np.array_equal(np.sort(single.sched_day.to_numpy()), np.sort(o.sched_day.to_numpy())), \
        "single-unit and per-month builds agree on sched_day - the fixture cannot tell them apart"

    with pytest.raises(ValueError, match="smoke"):
        stand_ab.cmd_cache(smoke=True, ranking=True)
    assert sorted(p.name for p in cache.iterdir()) == ["ranking.parquet"]
