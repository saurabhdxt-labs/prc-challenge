"""scripts/analysis/adsb_union.py — the adsb.lol + adsb.fi point union (rule AUN.1) and its manifest. The union tests moved
here with the functions from the local AUN script (unchanged assertions); the manifest test is new (AFI).
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load("adsb_union", ROOT / "scripts" / "analysis" / "adsb_union.py")
COLS = U.COLS


def _pts(icao, ts, src="adsb_icao"):
    n = len(ts)
    return pd.DataFrame({"icao": pd.array([icao] * n, dtype="string"), "reg": pd.array(["R"] * n, dtype="string"),
                         "type": pd.array(["A320"] * n, dtype="string"), "callsign": pd.array(["CS1"] * n, dtype="string"),
                         "airport": pd.array(["LEMD"] * n, dtype="string"), "t": np.asarray(ts, "float64"),
                         "lat": np.full(n, 40.0), "lon": np.full(n, -3.0), "on_ground": np.ones(n, "int64"),
                         "alt_missing": np.zeros(n, "int64"), "alt": np.full(n, np.nan), "gs": np.zeros(n),
                         "track": np.zeros(n), "src": pd.array([src] * n, dtype="string")})[COLS]


# ---------------------------------------------------------------- union_day (AUN.1 rule 1)

def test_lol_samples_are_never_dropped_even_when_sub_second():
    """Catches: collapsing adsb.lol (the defect AUN.1 removed) — lol pairs 0.3 s apart must all survive."""
    lol = _pts("abc123", [100.0, 100.3, 100.6, 101.0])
    out, info = U.union_day(lol, None)
    assert len(out) == 4
    assert np.array_equal(out.t.to_numpy(), [100.0, 100.3, 100.6, 101.0])
    assert info == {"n_lol": 4, "n_fi": 0, "n_fi_kept": 0, "n_fi_dropped_near_lol": 0}


def test_fi_sample_within_one_second_of_a_lol_sample_is_dropped_both_sides():
    """Catches: one-sided check (only the previous lol sample) and a strict/loose boundary: 99.2 (0.8 before 100) and
    101.9 (0.9 after 101) are dropped; 98.9 (1.1 before) and 103.0 are kept; 102.0 (exactly 1.0 after 101) is kept."""
    lol = _pts("abc123", [100.0, 101.0])
    fi = _pts("abc123", [98.9, 99.2, 101.9, 102.0, 103.0], src="fi")
    out, info = U.union_day(lol, fi)
    assert out.t.tolist() == [98.9, 100.0, 101.0, 102.0, 103.0]
    assert info == {"n_lol": 2, "n_fi": 5, "n_fi_kept": 3, "n_fi_dropped_near_lol": 2}


def test_proximity_is_per_aircraft():
    """Catches: comparing an fi sample with another aircraft's lol samples."""
    lol = _pts("aaaaaa", [100.0])
    fi = _pts("bbbbbb", [100.2], src="fi")
    out, info = U.union_day(lol, fi)
    assert len(out) == 2 and info["n_fi_kept"] == 1


def test_icao_case_is_normalised_before_matching():
    """Catches: 'ABC123' (fi) and 'abc123' (lol) treated as different aircraft — the near-duplicate would survive."""
    lol = _pts("abc123", [100.0])
    fi = _pts("ABC123", [100.5], src="fi")
    out, info = U.union_day(lol, fi)
    assert info["n_fi_dropped_near_lol"] == 1 and len(out) == 1


def test_output_is_sorted_by_icao_then_time_with_lol_first_on_ties():
    """Catches: unsorted output, and an fi row winning a tie (a tie is < 1 s, so fi is dropped anyway — here two aircraft
    interleave in time and the frame must still come out grouped and time-ordered)."""
    lol = pd.concat([_pts("bbbbbb", [50.0, 10.0]), _pts("aaaaaa", [30.0])], ignore_index=True)
    fi = pd.concat([_pts("aaaaaa", [5.0], src="fi"), _pts("bbbbbb", [70.0], src="fi")], ignore_index=True)
    out, _ = U.union_day(lol, fi)
    assert out.icao.tolist() == ["aaaaaa", "aaaaaa", "bbbbbb", "bbbbbb", "bbbbbb"]
    assert out.t.tolist() == [5.0, 30.0, 10.0, 50.0, 70.0]
    assert list(out.columns) == COLS


def test_lol_subset_of_union_equals_lol_file_row_for_row():
    """Catches: any mutation of lol values (dtype, rounding, dropped columns) — G6."""
    rng = np.random.default_rng(0)
    lol = pd.concat([_pts(f"{i:06x}", np.sort(rng.uniform(0, 500, 40))) for i in range(5)], ignore_index=True)
    lol["lat"] = rng.normal(40, 1, len(lol))
    fi = pd.concat([_pts(f"{i:06x}", np.arange(0, 500, 60.0) + 0.37, src="fi") for i in range(5)], ignore_index=True)
    out, info = U.union_day(lol, fi)
    got = out[out.src != "fi"].reset_index(drop=True)
    want = lol.sort_values(["icao", "t"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want)
    assert len(out) == len(lol) + info["n_fi_kept"]


def test_near_duplicate_check_is_exact_at_day_scale_times():
    """Catches: a float composite key that loses sub-second precision at epoch-scale t (1.78e9 s) with many aircraft."""
    base = 1_784_091_572.57
    lol = pd.concat([_pts(f"{i:06x}", [base + i]) for i in range(60_000, 60_050)], ignore_index=True)
    fi = pd.concat([_pts(f"{i:06x}", [base + i + 0.999], src="fi") for i in range(60_000, 60_025)]
                   + [_pts(f"{i:06x}", [base + i + 1.0005], src="fi") for i in range(60_025, 60_050)], ignore_index=True)
    out, info = U.union_day(lol, fi)
    assert info["n_fi_dropped_near_lol"] == 25 and info["n_fi_kept"] == 25


def test_empty_or_missing_fi_equals_sorted_lol():
    """Catches: a crash or a changed frame when the fi side is empty (G6)."""
    lol = _pts("abc123", [5.0, 1.0])
    a, _ = U.union_day(lol, None)
    b, _ = U.union_day(lol, lol.iloc[0:0])
    pd.testing.assert_frame_equal(a, b)
    assert a.t.tolist() == [1.0, 5.0]


def test_missing_lol_side_returns_fi_only():
    """Catches: a crash on a day without an adsb.lol file (fi rows are kept as they are)."""
    out, info = U.union_day(None, _pts("abc123", [1.0, 70.0], src="fi"))
    assert len(out) == 2 and info["n_fi_kept"] == 2


def test_column_mismatch_is_refused():
    """Catches: silently unioning frames with different schemas."""
    with pytest.raises(ValueError, match="columns"):
        U.union_day(_pts("a", [1.0]), _pts("a", [5.0]).drop(columns=["gs"]))


# ---------------------------------------------------------------- upgrade set / transitions / projection

def test_build_days_writes_union_and_passes_g6(tmp_path):
    """Catches: a day file that is not lol + kept fi, a missing neighbour day, or a leftover .partial file."""
    lol_dir, fi_dir, out = tmp_path / "lol", tmp_path / "fi", tmp_path / "out"
    lol_dir.mkdir(); fi_dir.mkdir()
    _pts("abc123", [100.0, 100.4, 200.0]).to_parquet(lol_dir / "adsb_2026-07-15.parquet", index=False)
    _pts("abc123", [100.5, 160.0], src="fi").to_parquet(fi_dir / "adsb_2026-07-15.parquet", index=False)
    _pts("abc123", [5.0]).to_parquet(lol_dir / "adsb_2026-07-14.parquet", index=False)
    info = U.build_days(["2026-07-14", "2026-07-15", "2026-07-16"], out, fi_dir=fi_dir, lol_dir=lol_dir)
    assert sorted(info) == ["2026-07-14", "2026-07-15"]
    assert info["2026-07-15"] == {"n_lol": 3, "n_fi": 2, "n_fi_kept": 1, "n_fi_dropped_near_lol": 1, "n_fi_excluded_airport": 0}
    got = pd.read_parquet(out / "adsb_2026-07-15.parquet")
    assert got.t.tolist() == [100.0, 100.4, 160.0, 200.0]
    assert not list(out.glob(".*partial"))


def test_build_days_refuses_to_overwrite(tmp_path):
    lol_dir = tmp_path / "lol"
    lol_dir.mkdir()
    _pts("abc123", [1.0]).to_parquet(lol_dir / "adsb_2026-07-15.parquet", index=False)
    U.build_days(["2026-07-15"], tmp_path / "out", fi_dir=tmp_path / "none", lol_dir=lol_dir)
    with pytest.raises(FileExistsError):
        U.build_days(["2026-07-15"], tmp_path / "out", fi_dir=tmp_path / "none", lol_dir=lol_dir)


def test_fi_rows_marks_exactly_the_non_lol_rows_including_duplicate_lol_times():
    """Catches: exact-duplicate adsb.lol timestamps (115-1,074 per July day) being mistaken for fi rows in G6."""
    lol = _pts("abc123", [1.0, 1.0, 5.0])
    fi = _pts("abc123", [70.0], src="fi")
    u, _ = U.union_day(lol, fi)
    assert U.fi_rows(u, lol).tolist() == [False, False, False, True]


def test_the_manifest_is_read_by_r12_build_and_declares_both_sources(tmp_path):
    """r12_build.manifest_days must list every written day; `contains` declares adsb.lol and adsb.fi (FM1 checks the
    tables against adsb.sources.adsb_fi through it). Mutation: status "done" instead of "ok" -> r12_build lists no day ->
    red; drop `contains` -> red."""
    RB = _load("r12_build", ROOT / "scripts" / "analysis" / "r12_build.py")
    info = {"2026-07-01": {"n_lol": 3, "n_fi": 2, "n_fi_kept": 1, "n_fi_dropped_near_lol": 1},
            "2026-07-02": {"n_lol": 1, "n_fi": 0, "n_fi_kept": 0, "n_fi_dropped_near_lol": 0}}
    p = U.write_manifest(tmp_path, info, "test source", "test licence")
    assert RB.manifest_days(tmp_path) == {"2026-07-01", "2026-07-02"}
    man = json.loads(p.read_text())
    assert man["contains"] == ["adsb.lol", "adsb.fi"] and man["rule"] == U.RULE
    with pytest.raises(FileExistsError):
        U.write_manifest(tmp_path, info, "test source", "test licence")


def test_build_days_can_leave_named_airports_adsb_fi_samples_out(tmp_path):
    """AFI.1: LTFM's adsb.fi samples are excluded (never trained with ADS-B; join-quality gate failed there); its adsb.lol
    samples stay, other airports' adsb.fi samples stay. Mutation: exclude by aircraft instead of airport -> red; drop the
    airport's adsb.lol rows too -> red."""
    lol_dir, fi_dir = tmp_path / "lol", tmp_path / "fi"
    lol_dir.mkdir(); fi_dir.mkdir()
    lol = pd.concat([_pts("aaa111", [100.0]), _pts("bbb222", [100.0]).assign(airport=pd.array(["LTFM"], dtype="string"))],
                    ignore_index=True)[COLS]
    fi = pd.concat([_pts("aaa111", [300.0], src="fi"),
                    _pts("bbb222", [300.0], src="fi").assign(airport=pd.array(["LTFM"], dtype="string"))], ignore_index=True)[COLS]
    lol.to_parquet(lol_dir / "adsb_2026-07-15.parquet", index=False)
    fi.to_parquet(fi_dir / "adsb_2026-07-15.parquet", index=False)
    info = U.build_days(["2026-07-15"], tmp_path / "out", fi_dir=fi_dir, lol_dir=lol_dir, fi_exclude_airports=("LTFM",))
    got = pd.read_parquet(tmp_path / "out" / "adsb_2026-07-15.parquet")
    assert sorted(zip(got.icao, got.t, got.airport)) == [("aaa111", 100.0, "LEMD"), ("aaa111", 300.0, "LEMD"),
                                                         ("bbb222", 100.0, "LTFM")]
    assert info["2026-07-15"]["n_fi_excluded_airport"] == 1
