"""tools/fm1_replay_gate.py — AT14, the hard end-to-end replay gate (plans/PREREG_fm1_2026_09_28.md Amendment A1).

The gate's own logic (stratified sampling, the noise-floor verdict, reference consistency) is tested on constructed inputs;
its real path is exercised by the smoke run in a CHILD process (it builds real features and runs two 0.7 GB v34 boosters,
each in its own child)."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import knobs as K
from isolation import run_guarded

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fm1_replay_gate", ROOT / "tools" / "fm1_replay_gate.py")
GATE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GATE)


def _ref(n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for ap in ("EDDF", "LTFM"):
        for month in (1, 7):
            for source, lane in (("r12", "N_v4"), ("r12", "R_orl"), ("v34", "N_afj"), ("v33", "outside_v33")):
                n = 3 if lane == "N_afj" else n_per
                for _ in range(n):
                    i += 1
                    rows.append({"MVT_ID_mvt": float(i), "ap": ap, "month": month, "source": source, "lane": lane,
                                 "v36": float(rng.integers(300, 1500))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------- sampling

def test_sample_takes_every_stratum_and_caps_each():
    """Mutation: sample only the first stratum per airport (drop the groupby over lane) -> strata missing -> red."""
    ref = _ref()
    s = GATE.sample_strata(ref, per_stratum=10, seed=1, min_rows=0)
    got = s.groupby(["ap", "month", "source", "lane"]).size()
    want = ref.groupby(["ap", "month", "source", "lane"]).size().clip(upper=10)
    pd.testing.assert_series_equal(got, want)
    again = GATE.sample_strata(ref, per_stratum=10, seed=1, min_rows=0)
    assert s.MVT_ID_mvt.tolist() == again.MVT_ID_mvt.tolist()               # seeded
    other = GATE.sample_strata(ref, per_stratum=10, seed=2, min_rows=0)
    assert s.MVT_ID_mvt.tolist() != other.MVT_ID_mvt.tolist()


def test_sample_refuses_too_few_rows():
    with pytest.raises(ValueError, match="< 5,000"):
        GATE.sample_strata(_ref(), per_stratum=10, seed=0, min_rows=5000)


# ---------------------------------------------------------------------------------------------- verdict

def _sample(v36):
    return pd.DataFrame({"MVT_ID_mvt": np.arange(len(v36), dtype=float), "ap": "EDDF", "month": 1, "source": "r12",
                         "lane": "N_v4", "v36": np.asarray(v36, float)})


def test_verdict_tolerance_is_one_second_plus_the_floor():
    """Mutation: compare against tol only (drop + floor) -> the 1.6 s row fails in the floored case -> red; use `>=`
    instead of `>` -> the exact 1.0 s row fails -> red."""
    s = _sample([100, 100, 100])
    run = np.array([101.0, 100.0, 98.5])
    rows, rep = GATE.verdict(s, [run, run], np.array(["", "", ""]))
    assert rep["noise_floor_s"] == 0.0 and rows.fail.tolist() == [False, False, True] and rep["verdict"] == "FAIL"
    # the package's own value moves by 0.7 s on row 0 between the two runs: floor 0.7, tolerance 1.7 s
    rows, rep = GATE.verdict(s, [np.array([101.6, 100.0, 100.0]), np.array([100.9, 100.0, 100.0])], np.array([""] * 3))
    assert rep["noise_floor_s"] == pytest.approx(0.7)          # exact spread of two float64 values set above
    assert rows.fail.tolist() == [False, False, False] and rep["verdict"] == "PASS"
    rows, rep = GATE.verdict(s, [np.array([101.8, 100.0, 100.0]), np.array([101.1, 100.0, 100.0])], np.array([""] * 3))
    assert rows.fail.tolist() == [True, False, False]


def test_verdict_fails_rows_the_package_cannot_produce():
    s = _sample([100, 200])
    rows, rep = GATE.verdict(s, [np.array([100.0, np.nan]), np.array([100.0, np.nan])], np.array(["", "NOT PRODUCIBLE"]))
    assert rows.fail.tolist() == [False, True] and rep["verdict"] == "FAIL" and rep["n_fail"] == 1
    assert rep["noise_floor_s"] == 0.0                         # NaN rows never set the floor


def test_verdict_reports_every_stratum():
    s = pd.concat([_sample([100, 100]), _sample([50]).assign(lane="R_orl", MVT_ID_mvt=9.0)], ignore_index=True)
    rows, rep = GATE.verdict(s, [np.array([100.0, 100.0, 50.0])] * 2, np.array(["a", "a", "b"]))
    assert set(rep["strata"]) == {"EDDF|1|r12|N_v4", "EDDF|1|r12|R_orl"}
    assert rep["strata"]["EDDF|1|r12|N_v4"] == {"n": 2, "n_fail": 0, "max_abs_diff": 0.0, "exact": 2, "note": "a"}


# ---------------------------------------------------------------------------------------------- the real reference + path

def _run_child(code: str, timeout=900):
    r = run_guarded([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r.stdout


@pytest.mark.devtree
@pytest.mark.slow
def test_reference_is_consistent_and_complete():
    """v36 = R12 serve at its five airports, v34 elsewhere, v33 on the 5,290 unmatched rows (asserted by load_reference).
    Runs in a child: the reference reads 344,841 x a few columns plus the ranking file's departures."""
    out = _run_child(f"""
import importlib.util, json
spec = importlib.util.spec_from_file_location("g", {str(ROOT / 'tools' / 'fm1_replay_gate.py')!r}); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
ref = g.load_reference()
print(json.dumps({{"n": len(ref), "by_source": ref.source.value_counts().to_dict(),
                   "r12_aps": sorted(ref[ref.source == "r12"].ap.unique()), "months": sorted(ref.month.unique().tolist())}}))
""")
    rep = json.loads(out.strip().splitlines()[-1])
    assert rep["n"] == 344_841 and rep["by_source"]["v33"] == 5_290
    assert rep["r12_aps"] == ["EDDF", "EDDM", "EHAM", "LEBL", "LSZH"] and rep["months"] == [1, 7]
    meta = json.loads((ROOT / "submissions/merry-quicksand_v36.parquet.meta.json").read_text())
    assert rep["by_source"]["r12"] == meta["n_given"]


@pytest.mark.devtree
@pytest.mark.slow
def test_reference_inconsistency_is_refused(tmp_path):
    """A v36 copy with one value changed is refused before any comparison. Mutation: drop the consistency assertion -> red."""
    v36 = pd.read_parquet(ROOT / "submissions/merry-quicksand_v36.parquet")
    v36.loc[0, "TAXITIME_SEC_mvt"] += 5
    v36.to_parquet(tmp_path / "v36.parquet", index=False)
    out = _run_child(f"""
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("g", {str(ROOT / 'tools' / 'fm1_replay_gate.py')!r}); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
paths = dict(g.PATHS, v36=pathlib.Path({str(tmp_path / 'v36.parquet')!r}))
try:
    g.load_reference(paths)
    print("NOT REFUSED")
except AssertionError as e:
    print("REFUSED", e)
""")
    assert "REFUSED reference inconsistent" in out and ("1 R12 rows" in out or "1 other rows" in out)


@pytest.mark.devtree
@pytest.mark.slow
def test_gate_smoke_passes_on_real_rows(tmp_path):
    """The real path end to end on LSZH-January R12 rows (package features -> encodings -> design -> the saved v34
    ORL/FW boosters' seed 0 in children -> the stack stage -> ledger -> finalise vs v36), two runs."""
    r = run_guarded([sys.executable, "-B", str(ROOT / "tools" / "fm1_replay_gate.py"), "--out", str(tmp_path / "g"),
                        "--smoke", "--per-stratum", "10"], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rep = json.loads((tmp_path / "g" / "report.json").read_text())
    assert rep["verdict"] == "PASS" and rep["noise_floor_s"] == 0.0 and rep["n_runs"] == 2
    for ck in rep["component_checks"]:
        for role in ("ORL96", "FW90"):
            assert ck[role]["seed0_bit_equal"] == ck[role]["n"]
        # the stack's float32 MLP rounds differently by batch size: 3.9e-7 s on a 10-row batch, 0.0 on the 1,000-row gate
        # run, against the stored full-batch residual (FM1 notes N4). Values stay exact after rint (asserted by PASS).
        assert ck["R12_stack"]["resid_max_abs_diff_vs_stored"] < 1e-3


def test_verdict_names_dropped_component_rows_without_failing_them():
    """Amendment A3: a row the shipped ARN step set (DROPPED in FM1) is listed as a named exception, never a FAIL; any
    other row beyond 1 s + floor still FAILs. Mutation: skip the exception mask -> the ARN row fails -> red; apply the
    mask to every row -> the non-ARN 3 s row passes -> red."""
    s = _sample([100, 200, 300]).assign(source="v33", lane="outside_v33", ap="LIRF")
    run = np.array([100.0, 250.0, 303.0])
    exc = pd.Series({1.0: "ARN"})
    rows, rep = GATE.verdict(s, [run, run], np.array(["", "", ""]), exceptions=exc)
    assert rows.fail.tolist() == [False, False, True] and rows.exception.tolist() == ["", "ARN", ""]
    assert rep["verdict"] == "FAIL" and rep["n_fail"] == 1
    assert rep["exceptions"] == {"ARN": {"n": 1, "n_differ": 1, "ids": [1.0], "max_abs_diff": 50.0}}
    rows, rep = GATE.verdict(s, [np.array([100.0, 250.0, 300.0])] * 2, np.array([""] * 3), exceptions=exc)
    assert rep["verdict"] == "PASS" and rep["n_fail"] == 0 and rep["exceptions"]["ARN"]["n_differ"] == 1


def test_population_check_fails_non_exception_rows_and_lists_the_dropped_ones():
    """Every scored unmatched row is compared (not only the sample): a non-exception row beyond 1 s + floor FAILS, an ARN
    row is listed with its diff; the floor is the spread of the lane across runs. Mutation: ignore the exception mask ->
    the ARN row fails -> red; compare only run 0 without the floor -> the 0.8 s-spread row fails at 1.5 s -> red."""
    ref = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0], "source": ["v33"] * 4 + ["r12"],
                        "ap": ["LIRF", "LIRF", "EDDF", "EDDF", "EDDF"], "v36": [100.0, 200.0, 300.0, 400.0, 9.0]})
    lane0 = pd.Series({1.0: 100.0, 2.0: 290.0, 3.0: 301.5, 4.0: 400.0})
    lane1 = pd.Series({1.0: 100.0, 2.0: 290.0, 3.0: 300.7, 4.0: 400.0})
    exc = pd.Series({2.0: "ARN"})
    rep = GATE.population_check(ref, [lane0, lane1], exc)
    assert rep["n"] == 4 and rep["n_fail"] == 0 and rep["floor_s"] == pytest.approx(0.8)
    assert rep["exceptions"]["ARN"] == {"n": 1, "ids": [2.0], "diffs": {"2.0": 90.0}}
    rep = GATE.population_check(ref, [lane0.replace({400.0: 404.0}), lane1.replace({400.0: 404.0})], exc)
    assert rep["n_fail"] == 1 and rep["fail_ids"] == [4.0] and rep["by_airport"]["EDDF"]["n_fail"] == 1


@pytest.mark.devtree
@pytest.mark.slow          # > 1 s (rule 26): package_values loads the METAR archive and the ORL route table first
def test_unmatched_rows_are_served_by_the_package_lane():
    """The unmatched strata run the package's unmatched lane (fit from raw + serve): package_values takes each v33 row's
    value from it, notes what was recomputed, and records the lane's full-population values for the population check.
    Mutation: read the v33 rows from the reference instead of the lane -> the values equal v36 -> red."""
    s = _sample([500, 600]).assign(source="v33", lane="outside_v33", ap="LIRF")
    lane = pd.Series({0.0: 510.0, 1.0: 612.0, 7.0: 99.0})
    vals, notes, checks = GATE.package_values(s, log=lambda *_: None,
                                              unmatched_fn=lambda cfg, log: (lane, {"decided_by": {"rome": 3}}))
    assert vals.tolist() == [510.0, 612.0]
    assert all(n.startswith("recomputed (fit from raw 2025 + serve):") for n in notes)
    assert checks["unmatched_lane"]["decided_by"] == {"rome": 3}
    pd.testing.assert_series_equal(checks["unmatched_population"], lane)


@pytest.mark.slow          # a real fit of the lane in the child
def test_unmatched_child_serves_the_lane_end_to_end(e2e_world, fm1_world_mod, tmp_path):
    """The gate's unmatched child (the path AT14 runs on real data) on the e2e world: engine.fit of the lane alone, serve
    on the scored file (its matched departures are the lane's context), finalise; values for exactly the unmatched
    departures, integer >= 1, with the lane's provenance and the child's peak RSS. Mutation: serve through a ledger over
    ALL departures -> finalise refuses the unserved matched rows -> red."""
    world = e2e_world[0]
    raw = fm1_world_mod.config_raw(world)
    raw["run"]["artifacts_dir"] = str(tmp_path / "a")
    cfg = K.parse_config(raw)
    paths = dict(GATE.PATHS, ranking=world / "raw" / "scored.parquet")
    lane, info = GATE.unmatched_values(cfg, paths, log=lambda *_: None)
    scored = pd.read_parquet(world / "raw" / "scored.parquet")
    unm = scored[(scored.PHASE_mvt == "DEP") & scored.AOBT_3_flt.isna()].MVT_ID_mvt.to_numpy(float)
    assert sorted(lane.index.tolist()) == sorted(unm.tolist()) and (lane >= 1).all() and (lane == np.rint(lane)).all()
    assert info["decided_by"] == {"unmatched_body": len(unm)} and info["child_peak_rss_gib"] > 0
    assert info["fit"]["learner"] == "v26" and info["serve"]["n_rows"] == len(unm)


def test_gate_config_replays_v36():
    """The replay configuration: the v26 body with its storm anchor (v16 inside v26), the Rome module, ARC at LIRF."""
    cfg = GATE.gate_config()
    assert cfg["unmatched.learner"] == "v26" and cfg["modules.storm.mode"] == "anchor"
    assert cfg["modules.rome.enabled"] and cfg["modules.rome.e2"] and cfg["modules.rome.e2w"]
    assert cfg["modules.arc.enabled"] and cfg["modules.arc.airports"] == ("LIRF",) and cfg.not_ported() == []


def test_regression_bug_32_exceptions_must_be_unmatched_rows():
    """BC-32: a named exception can never fail, so the exception set is guarded structurally: every id must be one of
    the reference's unmatched (v33) rows — ARN only ever set unmatched LIRF rows. A diff against the wrong predecessor
    (v17 -> v21 spans the matched-lane steps v18-v20: 127,418 rows) is refused before any comparison.
    Mutation: skip the check -> the matched id passes -> red."""
    ref = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "source": ["v33", "v33", "r12"]})
    GATE.check_exceptions(ref, pd.Series({2.0: "ARN"}))
    with pytest.raises(AssertionError, match=r"1 named exception row\(s\) are not unmatched rows"):
        GATE.check_exceptions(ref, pd.Series({2.0: "ARN", 3.0: "ARN"}))


@pytest.mark.devtree
@pytest.mark.slow
def test_dropped_rows_are_the_shipped_arn_rows():
    """The exception list is read from the shipped files: the rows whose value changed from v20 to v21 — ARN's own step
    (arn_ship_build.V20 is its base; v17 -> v21 would include the matched-lane steps v18-v20, BC-32) — and it equals the ARN
    rule's own 2026 rows (rome_arn.rows_2026). Mutation: diff v16 -> v21 -> red."""
    d = GATE.dropped_rows()
    assert sorted(d.index.tolist()) == sorted([202683690.0, 202682941.0, 213145862.0, 213146902.0, 213142735.0,
                                               213149019.0, 213148742.0, 213148858.0])
    assert set(d) == {"ARN"}
    from prc.pipeline.fm1 import shipped
    arn = shipped.unmatched_deps().R.rows_2026()
    assert set(arn.MVT_ID_mvt.astype(float)) == set(d.index)


def test_cli_refuses_one_run_and_a_non_empty_out(tmp_path):
    r = run_guarded([sys.executable, "-B", str(ROOT / "tools" / "fm1_replay_gate.py"), "--out", str(tmp_path / "a"),
                        "--runs", "1"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "--runs must be >= 2" in r.stderr
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "x").write_text("x")
    r = run_guarded([sys.executable, "-B", str(ROOT / "tools" / "fm1_replay_gate.py"), "--out", str(tmp_path / "b")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "not empty" in r.stderr
