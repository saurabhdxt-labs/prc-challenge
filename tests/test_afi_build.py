"""scripts/analysis/afi_build.py (plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md): the store copy that leaves only the
July-2026 days to rebuild, and the label-free acceptance of the new tables (1–3).

Mutation rehearsal per test in its docstring.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("afi_build", ROOT / "scripts" / "analysis" / "afi_build.py")
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)


def test_link_store_copies_every_record_except_the_july_2026_days(tmp_path):
    """Mutation: exclude nothing -> July records present -> red; copy instead of refusing an existing target -> red."""
    base = tmp_path / "base"
    base.mkdir()
    for name in ("STAMP.json", "day_2025-07-01.json", "day_2025-07-01.parquet", "day_2026-01-03.json",
                 "day_2026-01-03.parquet", "day_2026-07-02.json", "day_2026-07-02.parquet"):
        (base / name).write_text(name)
    new = tmp_path / "new"
    n = A.link_store(base, new, month="2026-07")
    assert sorted(p.name for p in new.iterdir()) == ["STAMP.json", "day_2025-07-01.json", "day_2025-07-01.parquet",
                                                     "day_2026-01-03.json", "day_2026-01-03.parquet"]
    assert n == 5 and (new / "STAMP.json").read_text() == "STAMP.json"
    with pytest.raises(FileExistsError):
        A.link_store(base, new, month="2026-07")


def _t(ids, month, ap, joined, dt):
    return pd.DataFrame({"month": month, "pos": np.arange(len(ids)), "MVT_ID_mvt": np.asarray(ids, float), "ap": ap,
                         "mvt_day": "2026-07-01", "adsb_day_present": 1, "coverage": np.where(np.asarray(joined) == 1, 3, 1),
                         "joined": np.asarray(joined, float), "dt_takeoff": np.asarray(dt, float)})


def _dirs(tmp_path, new_serve, new_train=None, sources=("adsb.fi", "adsb.lol")):
    old, new = tmp_path / "old", tmp_path / "new"
    train = _t([1, 2], 7, "LIRF", [1, 0], [3.0, np.nan])
    serve = pd.concat([_t([10, 11], 1, "LIRF", [1, 0], [2.0, np.nan]),
                       _t([20, 21, 22, 23], 7, "LIRF", [0, 0, 0, 1], [np.nan, np.nan, np.nan, 4.0])], ignore_index=True)
    for d, tr, sv, rep in ((old, train, serve, {"sources": ["adsb.lol"]}),
                           (new, train if new_train is None else new_train, serve if new_serve is None else new_serve,
                            {"sources": list(sources)})):
        d.mkdir()
        tr.to_parquet(d / "train.parquet", index=False)
        sv.to_parquet(d / "serve.parquet", index=False)
        (d / "report.json").write_text(json.dumps(rep))
    return old, new, serve


def test_check_passes_when_only_july_serve_rows_gain_good_joins(tmp_path):
    """The positive control: valid tables pass every check. Mutation (2026-10-03): the share comparison reversed -> red;
    the join-quality threshold reversed -> red."""
    old, new, serve = _dirs(tmp_path, None)
    good = serve.copy()
    good.loc[[2, 3], "joined"] = 1.0
    good.loc[[2, 3], "dt_takeoff"] = [5.0, -8.0]
    good.to_parquet(new / "serve.parquet", index=False)
    r = A.check_tables(old, new, min_rows=1)
    assert r["ok"], r
    assert r["july_joined"]["LIRF"] == {"old": 0.25, "new": 0.75}


@pytest.mark.parametrize("break_it, failed", [
    ("train", "train_identical"), ("january", "january_identical"), ("fall", "july_never_falls"),
    ("misjoin", "join_quality"), ("sources", "declares_adsb_fi")])
def test_check_fails_on_each_registered_defect(tmp_path, break_it, failed):
    """Mutation: drop any one of the five checks -> its case passes -> red."""
    old, new, serve = _dirs(tmp_path, None, sources=("adsb.lol",) if break_it == "sources" else ("adsb.fi", "adsb.lol"))
    s = serve.copy()
    s.loc[[2, 3], "joined"] = 1.0
    s.loc[[2, 3], "dt_takeoff"] = [5.0, 7.0]
    if break_it == "train":
        tr = pd.read_parquet(new / "train.parquet")
        tr.loc[0, "dt_takeoff"] = 99.0
        tr.to_parquet(new / "train.parquet", index=False)
    if break_it == "january":
        s.loc[0, "dt_takeoff"] = 50.0
    if break_it == "fall":                      # the airport's July share falls (no gains, one loss)
        s.loc[[2, 3], "joined"] = 0.0
        s.loc[[2, 3], "dt_takeoff"] = np.nan
        s.loc[5, "joined"] = 0.0
    if break_it == "misjoin":
        s.loc[2, "dt_takeoff"] = 400.0
    s.to_parquet(new / "serve.parquet", index=False)
    r = A.check_tables(old, new, min_rows=1)
    assert not r["ok"] and failed in r["failed"], r


def test_link_store_hard_links_are_independent_of_the_working_directory(tmp_path, monkeypatch):
    """BC-40 sibling, category (b): a hard link names the file itself, not a path, so a store copied from a relative base
    reads the same bytes from anywhere (the protecting invariant). Changing the working directory afterwards must not
    break it. Mutation (2026-10-03): os.symlink for os.link -> a relative link dangles -> FileNotFoundError -> red."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "day_2025-07-01.parquet").write_text("payload")
    A.link_store(pathlib.Path("base"), pathlib.Path("new"), month="2026-07")
    monkeypatch.chdir(tmp_path / "base")
    assert (tmp_path / "new" / "day_2025-07-01.parquet").read_text() == "payload"


def test_a_lost_join_is_reported_but_the_registered_rule_is_the_share(tmp_path):
    """AFI.1: acceptance 2 as registered — the July share must not fall; flights joined before that the union leaves
    unjoined (ambiguity refusals) are reported. Mutation: gate on lost joins again -> red."""
    old, new, serve = _dirs(tmp_path, None)
    s = serve.copy()
    s.loc[[2, 3], "joined"] = 1.0
    s.loc[[2, 3], "dt_takeoff"] = [5.0, 7.0]
    s.loc[5, "joined"] = 0.0
    s.to_parquet(new / "serve.parquet", index=False)
    r = A.check_tables(old, new, min_rows=1)
    assert r["ok"] and r["lost_joins"] == 1


def test_fallback_serve_keeps_adsb_lol_rows_where_it_joined_and_takes_the_union_elsewhere():
    """AFB: July flights adsb.lol joined keep their adsb.lol row exactly; July flights it did not join take the union row;
    January keeps adsb.lol. Mutation: take the union for every July row -> the joined row differs -> red; take the union
    for January too -> red."""
    old = pd.concat([_t([10], 1, "LIRF", [0], [np.nan]), _t([20, 21], 7, "LIRF", [1, 0], [3.0, np.nan])], ignore_index=True)
    uni = pd.concat([_t([10], 1, "LIRF", [1], [9.0]), _t([20, 21], 7, "LIRF", [1, 1], [4.0, 6.0])], ignore_index=True)
    raw = A.fallback_serve(old, uni)
    assert list(raw.columns) == list(old.columns) and len(raw) == 3
    out = raw.set_index("MVT_ID_mvt")
    assert out.loc[10.0, "joined"] == 0 and np.isnan(out.loc[10.0, "dt_takeoff"])        # January: adsb.lol
    assert out.loc[20.0, "dt_takeoff"] == 3.0                                             # joined by adsb.lol: kept
    assert out.loc[21.0, "dt_takeoff"] == 6.0 and out.loc[21.0, "joined"] == 1            # not joined: the union


def test_fallback_serve_refuses_different_flight_sets():
    """Mutation: drop the id check -> a union missing a flight silently loses it -> red."""
    old = _t([20, 21], 7, "LIRF", [1, 0], [3.0, np.nan])
    with pytest.raises(ValueError, match="flight sets differ"):
        A.fallback_serve(old, old.iloc[:1])


def test_check_with_fallback_requires_every_adsb_lol_joined_row_unchanged(tmp_path):
    """Mutation: skip the fallback check -> a changed joined row passes -> red."""
    old, new, serve = _dirs(tmp_path, None)
    s = serve.copy()
    s.loc[[2, 3], "joined"] = 1.0
    s.loc[[2, 3], "dt_takeoff"] = [5.0, 7.0]
    s.to_parquet(new / "serve.parquet", index=False)
    assert A.check_tables(old, new, min_rows=1, fallback=True)["ok"]
    s.loc[5, "dt_takeoff"] = 99.0                          # id 23 was joined by adsb.lol
    s.to_parquet(new / "serve.parquet", index=False)
    r = A.check_tables(old, new, min_rows=1, fallback=True)
    assert not r["ok"] and "joined_rows_identical" in r["failed"]


@pytest.mark.parametrize("change", ["drop_a_july_row", "add_a_july_row", "duplicate_a_july_row"])
def test_check_refuses_a_changed_july_flight_set(tmp_path, change):
    """End-of-session review 2026-10-03 (finding 5): the new serve table must hold exactly the old flights. A missing July
    row was reindexed in as 'not joined', an extra row was ignored and a duplicate crashed the reindex; the row checks
    that cannot run are reported as not run and fail. Seen red 2026-10-03 08:25-08:42 EDT."""
    old, new, serve = _dirs(tmp_path, None)
    s = serve.copy()
    s.loc[[2, 3], "joined"] = 1.0
    s.loc[[2, 3], "dt_takeoff"] = [5.0, 7.0]
    if change == "drop_a_july_row":
        s = s.drop(index=4).reset_index(drop=True)
    elif change == "add_a_july_row":
        s = pd.concat([s, _t([99], 7, "LIRF", [0], [np.nan])], ignore_index=True)
    else:
        s = pd.concat([s, s.iloc[[4]]], ignore_index=True)
    s.to_parquet(new / "serve.parquet", index=False)
    r = A.check_tables(old, new, min_rows=1, fallback=True)
    assert not r["ok"] and "same_flights" in r["failed"], r
    assert r["not_run"] == ["july_never_falls", "join_quality", "joined_rows_identical"]   # never read as passed
    assert set(r["not_run"]) <= set(r["failed"])


def _union_dir(tmp_path, old):
    """A union table directory: train hard-linked from `old`, serve = old with July flight 21 newly joined, report
    declaring adsb.fi with deliberately stale serve statistics (the fallback must not copy them)."""
    uni = tmp_path / "union"
    uni.mkdir()
    (uni / "train.parquet").write_bytes((old / "train.parquet").read_bytes())
    s = pd.read_parquet(old / "serve.parquet")
    s.loc[s.MVT_ID_mvt == 21.0, ["joined", "coverage", "dt_takeoff"]] = [1.0, 3, 6.0]
    s.loc[s.MVT_ID_mvt == 23.0, "dt_takeoff"] = 9.0          # a joined adsb.lol row the union changed
    s.to_parquet(uni / "serve.parquet", index=False)
    (uni / "report.json").write_text(json.dumps({"stamp": "x", "sources": ["adsb.fi", "adsb.lol"],
                                                 "serve": {"n_rows": -1, "coverage_counts": {"3": -1}}}))
    return uni


def test_write_fallback_tables_writes_its_own_report(tmp_path):
    """End-of-session review 2026-10-03 (findings 2 + 4): the fallback tables are written by a tested function, not the
    CLI shell; train is the old file itself (a hard link); serve is fallback_serve; report.json declares the union's
    sources and describes THIS serve table (the AFB report copied the union's serve statistics). Seen red 2026-10-03
    08:25-08:42 EDT (no function)."""
    old, _, _ = _dirs(tmp_path, None)
    uni = _union_dir(tmp_path, old)
    out = tmp_path / "fb"
    rep = A.write_fallback_tables(old, uni, out)
    assert os.path.samefile(out / "train.parquet", old / "train.parquet")
    srv = pd.read_parquet(out / "serve.parquet")
    pd.testing.assert_frame_equal(srv, A.fallback_serve(pd.read_parquet(old / "serve.parquet"),
                                                         pd.read_parquet(uni / "serve.parquet")))
    on_disk = json.loads((out / "report.json").read_text())
    assert on_disk == rep and rep["sources"] == ["adsb.fi", "adsb.lol"] and rep["union_tables"] == str(uni)
    assert rep["serve"] == A.R12.table_report(srv)
    assert rep["serve"]["coverage_counts"] == {str(k): int(v) for k, v in srv.coverage.value_counts().sort_index().items()}
    assert rep["train"] == A.R12.table_report(pd.read_parquet(old / "train.parquet"))
    with pytest.raises(FileExistsError):
        A.write_fallback_tables(old, uni, out)


def test_write_fallback_tables_refuses_a_union_that_does_not_declare_adsb_fi(tmp_path):
    """The fallback exists to bring adsb.fi rows in; a union directory that does not declare adsb.fi is the wrong input.
    Seen red 2026-10-03 08:25-08:42 EDT (no function)."""
    old, _, _ = _dirs(tmp_path, None)
    uni = _union_dir(tmp_path, old)
    (uni / "report.json").write_text(json.dumps({"sources": ["adsb.lol"]}))
    with pytest.raises(ValueError, match="does not declare adsb.fi"):
        A.write_fallback_tables(old, uni, tmp_path / "fb")
    assert not (tmp_path / "fb").exists()
