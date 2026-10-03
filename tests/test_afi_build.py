"""scripts/analysis/afi_build.py (plans/PREREG_afi_adsbfi_into_fm1_2026_10_03.md): the store copy that leaves only the
July-2026 days to rebuild, and the label-free acceptance of the new tables (1–3).

Mutation rehearsal per test in its docstring.
"""
from __future__ import annotations

import importlib.util
import json
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
    break it."""
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
