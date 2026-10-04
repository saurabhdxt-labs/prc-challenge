"""The ported unmatched lane (prc/pipeline/fm1/unmatched.py; plans/PREREG_fm1_2026_09_28.md Amendment A3).

The lane is the shipped v36 unmatched chain, reached through the shipped modules' public functions: the v26 body (S1 cells +
MJS congestion body), the LIRF/Rome module (R2 fill + 24 h date-slip -> E1 -> RLD; then E2 -> E2W after ARC), ARC, and the
storm anchor. ARN (v21) is DROPPED (FMA §4.2).

Tiers: pure helpers on constructed arrays (fast); the real-data-shaped e2e world (slow: real fits); the REAL replay of every
stage against the stored step outputs v7 / v10 / v12 / v17 / v26 / v31 / v33 on all 5,290 scored unmatched rows (slow, in a
child process: the shipped loader unm_congestion.load_real peaks at 3.3 GB).

Mutation rehearsal per test is recorded in its docstring and in tests/fm1/MUTATION_FM1.md (M49+).
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from prc import api
from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import shipped
from prc.pipeline.fm1 import stages as S
from prc.pipeline.fm1 import unmatched as U

ROOT = pathlib.Path(__file__).resolve().parents[2]
QUIET = lambda *_a, **_k: None          # noqa: E731
#: the 8 scored rows the shipped ARN step (v21) set: rome_arn.rows_2026() == the v17 -> v21 diff (logs/fm1_port_scratch)
ARN_IDS_2026 = [202683690.0, 202682941.0, 213145862.0, 213146902.0, 213142735.0, 213149019.0, 213148742.0, 213148858.0]


# ---------------------------------------------------------------------------------------------- the shipped modules

def test_shipped_deps_bind_the_chain_and_refuse_a_changed_e2_rule_file(monkeypatch, tmp_path):
    """The deps namespace is urp_replay.load_deps(); a rule file other than the one v31/v33 shipped with is refused.
    Mutation: drop the sha comparison in shipped.unmatched_deps -> the changed file loads -> red."""
    D = shipped.unmatched_deps()
    for name in ("fit_unmatched", "mixture", "nf_hybrid", "fit_predict_fill", "e1_floor", "apply_rld", "anchored_values",
                 "day_anomaly", "arc_values", "rule_E2", "e2_form", "fit_brier_w", "fit_body", "stopping_split",
                 "fit_congestion_regressor", "row_set", "candidate", "day_energy", "apply_day_check"):
        assert callable(getattr(D, name)), name
    fake = tmp_path / "agent_E_rules.py"
    fake.write_text(shipped.E2_RULES.read_text() + "\n# edited\n")
    monkeypatch.setattr(shipped, "E2_RULES", fake)
    shipped.unmatched_deps.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="is not the shipped rule file bd01f4ed927d4735"):
            shipped.unmatched_deps()
    finally:
        monkeypatch.undo()
        shipped.unmatched_deps.cache_clear()


# ---------------------------------------------------------------------------------------------- stage plan + ledger

def test_stage_order_is_the_shipped_chain_order():
    """Rome's R2 / E1 / RLD (v7-v12) precede ARC (v17), whose values E2 / E2W (v31, v33) read; the storm anchor acts on the
    body's rows last; calibration stays last. Mutation: swap rome and arc in STAGE_ORDER -> red."""
    names = [s.name for s in S.STAGE_ORDER]
    i = {n: names.index(n) for n in names}
    assert i["unmatched_body"] < i["rome"] < i["arc"] < i["rome_e2"] < i["rome_e2w"] < i["storm"] < i["ldi"] < i["mtx"]
    assert {"rome", "arc", "rome_e2", "rome_e2w", "storm", "unmatched_body"} <= set(S.VALUE_LABELS)
    assert "rome_e2w+mtx" in S.DECIDED_BY and "storm+mtx" in S.DECIDED_BY


def test_plan_turns_the_ported_modules_on_and_off():
    """Mutation: plan maps rome_e2w to modules.rome.e2 -> the e2w-off plan still lists it -> red."""
    full = [s.name for s in S.plan(K.parse_config({"modules": {"storm": {"mode": "anchor"}}}))]
    assert full[full.index("unmatched_body"):] == ["unmatched_body", "rome", "arc", "rome_e2", "rome_e2w", "storm",
                                                   "finalise"]
    off = [s.name for s in S.plan(K.parse_config({"modules": {"rome": {"e2w": False}, "arc": {"enabled": False}}}))]
    assert off[off.index("unmatched_body"):] == ["unmatched_body", "rome", "rome_e2", "finalise"]


def test_ledger_reads_values_and_writers_by_id():
    """The lane reads the current value and writer of its rows through the ledger's public accessors.
    Mutation: values_of returns self.value unindexed -> wrong row -> red."""
    L = S.Ledger(np.array([10.0, 20.0, 30.0]))
    st = {s.name: s for s in S.STAGE_ORDER}
    L.write(st["unmatched_body"], np.array([30.0, 10.0]), np.array([300.0, 100.0]))
    L.write(st["rome"], np.array([30.0]), np.array([333.0]))
    assert L.values_of(np.array([30.0, 10.0])).tolist() == [333.0, 100.0]
    assert L.decided_of(np.array([10.0, 30.0, 20.0])).tolist() == ["unmatched_body", "rome", None]
    with pytest.raises(E.RoutingError, match="1 id\\(s\\) not in the ledger"):
        L.values_of(np.array([40.0]))


# ---------------------------------------------------------------------------------------------- pure helpers

def test_body_params_default_to_the_shipped_mjs_arm():
    """The unmatched.hgb.* defaults ARE the v26 body's parameters (urp_replay.body_arms "MJS" = mxi_build.body_params(1000)
    + patience 50, serve-parity exact) plus sklearn's default learning rate. Mutation: map max_iter to
    unmatched.hgb.n_iter_no_change in body_params -> red."""
    from sklearn.ensemble import HistGradientBoostingRegressor as HGR
    got = U.body_params(K.parse_config({}))
    want = shipped.urp_replay().body_arms(shipped.unmatched_deps())["MJS"]
    assert {k: got[k] for k in want} == want and set(got) - set(want) == {"learning_rate"}
    assert got["learning_rate"] == HGR().learning_rate == 0.1
    assert U.body_params(K.parse_config({"unmatched": {"hgb": {"max_iter": 7}}}))["max_iter"] == 7


def test_storm_moves_only_body_rows_by_the_shipped_anchor():
    """congestion_anchor_ship.anchored_values = rint(v + max(A - 300, 0)); only rows the body decided move (the shipped
    v16 scope inside v26); NaN A -> no shift. Mutation: apply to every row (drop the body mask) -> the rome row moves -> red;
    knot 0 -> the A = 250 row moves -> red."""
    v = np.array([500.0, 500.0, 500.0, 500.0, 500.0])
    A = np.array([900.4, 250.0, np.nan, 1200.0, 300.0])
    body = np.array([True, True, True, False, True])
    new, moved = U.storm_values(v, A, body)
    assert new.tolist() == [1100.0, 500.0, 500.0, 500.0, 500.0]
    assert moved.tolist() == [True, False, False, False, False]
    with pytest.raises(ValueError, match="3 values, 2 anomalies"):
        U.storm_values(v[:3], A[:2], body[:3])


def test_rome_tags_name_every_rule_that_moved_a_row():
    """Mutation: compare v10 with rint(mix) instead of v7 -> the DS+E1 row loses +E1 -> red."""
    mix = np.array([100.4, 200.0, 300.0, 400.0, 200.0])
    v7 = np.array([100.0, 250.0, 300.0, 400.0, 250.0])
    v10 = np.array([100.0, 260.0, 300.0, 400.0, 250.0])
    v12 = np.array([100.0, 260.0, 310.0, 400.0, 250.0])
    assert U.rome_tags(mix, v7, v10, v12).tolist() == ["R2", "R2+DS+E1", "R2+RLD", "R2", "R2+DS"]


def test_unmatched_ids_are_the_departures_without_aobt3():
    t = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    rows = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], "PHASE_mvt": ["DEP", "DEP", "ARR", "DEP"],
                         "AOBT_3_flt": [pd.NaT, t, pd.NaT, pd.NaT]})
    rows["AOBT_3_flt"] = pd.to_datetime(rows.AOBT_3_flt, utc=True)
    assert U.unmatched_ids(rows).tolist() == [1.0, 4.0]


def test_e2_context_must_be_the_files_e2_reads(tmp_path):
    """E2 / E2W read LIRF arrival in-blocks from data.raw_dir themselves (agent_E_rules._files_for: the training file of
    the row's month, ranking.parquet for 2026). A configuration where that is not the scored rows' context is refused.
    Mutation: skip the scored-file comparison -> a foreign ranking file is accepted -> red."""
    raw = tmp_path / "raw"
    raw.mkdir()
    cfg = K.parse_config({"data": {"raw_dir": str(raw), "scored_file": str(raw / "ranking.parquet")}})
    s26 = pd.Series(pd.to_datetime(["2026-01-03 10:00", "2026-07-09 10:00"], utc=True))
    with pytest.raises(E.SchemaError, match=r"E2 context: .*ranking\.parquet does not exist"):
        U.check_e2_context(cfg, s26)
    (raw / "ranking.parquet").write_bytes(b"x")
    U.check_e2_context(cfg, s26)
    other = K.parse_config({"data": {"raw_dir": str(raw), "scored_file": str(tmp_path / "scored.parquet")}})
    with pytest.raises(E.SchemaError, match="E2 context: .*is not the configured data.scored_file"):
        U.check_e2_context(other, s26)
    s25 = pd.Series(pd.to_datetime(["2025-03-03 10:00"], utc=True))
    with pytest.raises(E.SchemaError, match=r"E2 context: no training file for 2025-03"):
        U.check_e2_context(cfg, s25)
    (raw / "training_2025-03-01_2025-04-01.parquet").write_bytes(b"x")
    U.check_e2_context(cfg, s25)


# ---------------------------------------------------------------------------------------------- the e2e world (real fits)

@pytest.fixture(scope="module")
def wcfg(e2e_world, fm1_world_mod, tmp_path_factory):
    """The world config with the matched lane cut to its base LightGBM (fast): the scored file's matched departures are
    the lane's CONTEXT (the serve witness, the storm anomaly), so they are scored too; the lane's rows are asserted."""
    raw = fm1_world_mod.config_raw(e2e_world[0])
    raw["run"]["artifacts_dir"] = str(tmp_path_factory.mktemp("fm1_unm_artifacts"))
    raw["adsb"]["stack"]["enabled"] = False
    raw["matched"]["blend"] = {"catboost_weight": 0.0}
    return K.parse_config(raw)


@pytest.fixture(scope="module")
def wscored(e2e_world):
    return pd.read_parquet(e2e_world[0] / "raw" / "scored.parquet")


@pytest.mark.slow          # real fits of the shipped estimators on the world
def test_world_lane_serves_every_unmatched_departure_and_round_trips(wcfg, wscored):
    """AT3 + AT4 + AT5 for the lane: every unmatched departure is served by `unmatched_body`, the component columns are
    present, a second fit is bit-identical and the saved artefact (the lane's training frames) predicts bit-exactly.
    Mutation: drop the sp-bin re-attachment on load -> fit_unmatched's cells KeyError -> red."""
    art = api.fit(wcfg, log=QUIET)
    assert art.unmatched_lane is not None and art.unmatched is None and art.enc_unmatched is None
    assert {"unmatched_lane/unm.parquet", "unmatched_lane/lirf.parquet", "unmatched_lane/extra.parquet",
            "unmatched_lane/lane.json"} <= set(json.loads((art.directory / "manifest.json").read_text())["files"])
    full = api.predict(art, wscored, log=QUIET)
    p = full[full.lane == "unmatched"]
    n_unm = int(((wscored.PHASE_mvt == "DEP") & wscored.AOBT_3_flt.isna()).sum())
    assert n_unm > 0 and len(p) == n_unm and (p.decided_by == "unmatched_body").all()
    assert np.array_equal(p.TAXITIME_SEC_mvt, np.rint(p.unmatched_pred)) and (p.unmatched_pred >= 1).all()
    for c in ("unm_p_fill", "unm_nf_cells", "unm_body"):
        assert p[c].notna().all(), c
    assert full[full.lane == "matched"].unm_body.isna().all()
    assert full.attrs["fm1_report"]["unmatched"]["n_rows"] == n_unm
    again = api.predict(api.fit(wcfg, save=False, log=QUIET), wscored, log=QUIET)
    pd.testing.assert_frame_equal(again, full, check_exact=True)
    back = api.load_artifact(art.directory)
    pd.testing.assert_frame_equal(api.predict(back, wscored, log=QUIET), full, check_exact=True)


@pytest.mark.slow
def test_regression_zero_unmatched_rows_skip_the_lane(wcfg, wscored):
    """BC-30 tripwire: a genuinely empty real-data slice for the lane (a scored file whose departures are all matched)
    must not reach the shipped string-key builders with zero rows. Mutation: remove the empty-lane early return -> the
    shipped `_key` / witness code sees 0 rows -> red."""
    art = api.fit(wcfg, save=False, log=QUIET)
    unm = wscored[(wscored.PHASE_mvt == "DEP") & wscored.AOBT_3_flt.isna()]
    rows = wscored[~wscored.MVT_ID_mvt.isin(unm.MVT_ID_mvt)].reset_index(drop=True)
    p = api.predict(art, rows, log=QUIET)
    assert len(p) == int((rows.PHASE_mvt == "DEP").sum()) and (p.decided_by == "base_lightgbm").all()


@pytest.mark.slow
def test_world_storm_anchor_moves_only_body_rows_and_labels_them(wcfg, e2e_world):
    """modules.storm.mode=anchor: rows whose airport-day anomaly A exceeds the 300 s knot move to rint(v + A - 300) and
    read `storm`; the others keep `unmatched_body`. The world's Jan 14 + 15 rows (labels hidden) with EDDF's Jan-15 matched
    AOBT_3 moved 1,000 s earlier: that day's MVT - AOBT_3 median rises 1,000 s, so A(EDDF, Jan 15) is about +500 (the
    median of two day medians is their mean) and its unmatched rows shift by about 200 s. Mutation: write every body row
    to `storm` (unmoved ones too) -> red; knot 0 -> the shift is 500 -> the formula check fails -> red."""
    from prc.pipeline.fm1 import rows as R
    raw = pd.read_parquet(e2e_world[0] / "raw" / "training_2025-01-01_2025-02-01.parquet")
    served, _ = R.hide_labels(raw)
    day15 = served.MVT_TIME_UTC_mvt.dt.day.to_numpy() == 15
    bump = day15 & (served.PHASE_mvt == "DEP").to_numpy() & (served.ADEP_mvt == "EDDF").to_numpy() & \
        served.AOBT_3_flt.notna().to_numpy()
    served.loc[bump, "AOBT_3_flt"] = served.loc[bump, "AOBT_3_flt"] - pd.Timedelta(seconds=1000)
    unm15 = day15 & (served.PHASE_mvt == "DEP").to_numpy() & (served.ADEP_mvt == "EDDF").to_numpy() & \
        served.AOBT_3_flt.isna().to_numpy()
    assert unm15.any(), "the world has no EDDF Jan-15 unmatched departure to shift"
    art = api.fit(wcfg.replace(**{"modules.storm.mode": "anchor"}), save=False, log=QUIET)
    full = api.predict(art, served, log=QUIET)
    p = full[full.lane == "unmatched"]
    shift = np.where(np.isfinite(p.storm_A), np.maximum(p.storm_A - 300.0, 0.0), 0.0)
    want = np.rint(np.rint(p.unmatched_pred) + shift)
    assert np.array_equal(p.TAXITIME_SEC_mvt.to_numpy(float), want)
    moved = want != np.rint(p.unmatched_pred)
    assert moved.any() and set(p.MVT_ID_mvt[moved]) <= set(served.MVT_ID_mvt[unm15])
    assert (p.decided_by[moved] == "storm").all() and (p.decided_by[~moved] == "unmatched_body").all()
    assert np.array_equal(p.storm_shift.to_numpy(float), want - np.rint(p.unmatched_pred))
    assert full.attrs["fm1_report"]["unmatched"]["storm"]["n_moved"] == int(moved.sum())


@pytest.mark.slow
def test_regression_bug_31_v26_lane_fits_on_a_single_training_month(wcfg):
    """BC-31: the strict swap folds (Amendment A2) fit on ONE month. The shipped loader unm_congestion.load_real embeds a
    self-check fold (fit on the months after the first, test on the first) that is empty with one month and raises, so
    the v26 lane refused every single-month fit. Seen red 2026-09-28 23:16Z: AssertionError "fit_unmatched(all matched)
    != fit_unmatched(LIRF matched) on the first fold" (logs/fm1_port_scratch/red_log.txt)."""
    from prc.pipeline.fm1 import engine as G
    files = dict(sorted(G.training_files(wcfg).items()))
    model = U.fit(wcfg, {1: files[1]}, (0, 1), log=QUIET)
    assert model.train_months == (1,) and len(model.unm) > 0 and set(model.unm.month.astype(int)) == {1}
    assert model.beta_w is None and len(model.extra) > 0


@pytest.mark.slow
def test_lane_frames_equal_the_shipped_loader(wcfg):
    """The lane's training frames are unm_congestion.load_real's recipe (derive(admissible(load_movements(files))) + y /
    delta + the matched admissible departures' airport-hour witness) WITHOUT its first-fold self-check: on two months
    (where load_real runs) every column, dtype, value and row order equals load_real's, and so do the witness counts.
    Mutation: attach the witness from ALL departures (drop the ~unmatched filter) -> hprox differs -> red; skip the
    admissible filter -> row counts differ -> red."""
    from prc.pipeline.fm1 import engine as G
    D = shipped.unmatched_deps()
    files = dict(sorted(G.training_files(wcfg).items()))
    unm, lirf, summ = U.training_frames(D, files)
    ref_u, ref_l, ref_s = D.uc.load_real([str(files[m]) for m in sorted(files)], lambda *_: None)
    for got, want in ((unm, ref_u), (lirf, ref_l)):
        assert list(got.columns) == list(want.columns) and len(got) == len(want)
        for c in want.columns:
            assert got[c].reset_index(drop=True).equals(want[c].reset_index(drop=True)), c
    assert {k: summ[k] for k in ("n_airport_hours", "n_matched_rows", "unmatched_nan_share")} == \
        {k: ref_s[k] for k in ("n_airport_hours", "n_matched_rows", "unmatched_nan_share")}


# ---- BC-31 siblings: every other shipped call with a hidden month / row-count precondition

@pytest.mark.devtree
def test_regression_bug_31_mtx_refuses_a_single_training_month_by_name():
    """BC-31 sibling (a): MTX's leave-month-out (mtx_calib.lmo_targets) fits each month on the OTHER months, so one
    training month gives an empty fit and the shipped `ValueError: fit_excess: empty cells`. FM1 refuses it by name first.
    Seen red 2026-09-28 (the bare shipped ValueError)."""
    from prc.pipeline.fm1 import components as C
    t = pd.to_datetime(["2025-01-10 10:00", "2025-01-11 12:00", "2025-01-12 09:00"], utc=True)
    dep = pd.DataFrame({"ADEP_mvt": ["EDDF", "EDDF", "LSZH"], "MVT_TIME_UTC_mvt": t, "TAXITIME_SEC_mvt": [600, 700, 650]})
    ref = K.ROOT / "data/mtx/raw/taxi_out_additional_time_2025.csv"
    with pytest.raises(E.SchemaError, match=r"MTX: leave-month-out needs >= 2 training months per airport; "
                                           r"\['EDDF', 'LSZH'\] have 1"):
        C.MtxCalibration().fit(dep, ref)


def test_e2w_refuses_fewer_than_two_training_months_with_lirf_rows():
    """BC-31 sibling (b): beta_W's nested leave-one-month-out needs LIRF rows in >= 2 training months; the refusal is
    FM1's, named, before any shipped fit runs (protecting invariant: fit_beta_w checks the months first)."""
    unm = pd.DataFrame({"ADEP_mvt": ["LIRF", "LIRF", "EDDF"], "month": [1, 1, 2]})
    with pytest.raises(E.SchemaError, match=r"needs LIRF unmatched rows in >= 2 training months, got \[1\]"):
        U.fit_beta_w(None, unm, unm.iloc[:0], (1, 2), (0,))


@pytest.mark.slow
def test_lane_refuses_a_fit_without_mau_extra_rows(wcfg, monkeypatch):
    """BC-31 sibling (b): the MJS body weights MAU's extra rows 0.25 * n_fit / n_extra, so zero extra rows would divide
    by zero inside the shipped fit; U.fit refuses by name (protecting invariant: the len(extra) check precedes the fit)."""
    from prc.pipeline.fm1 import engine as G
    real = U.deps

    def no_extra(raw_dir):
        D = real(raw_dir)
        MB = type("MB", (), {"RAW_COLS": D.MB.RAW_COLS, "extra_rows": staticmethod(lambda raw: D.MB.extra_rows(raw).iloc[:0])})
        return __import__("types").SimpleNamespace(**{**vars(D), "MB": MB})
    monkeypatch.setattr(U, "deps", no_extra)
    with pytest.raises(E.SchemaError, match=r"no MAU extra matched rows"):
        U.fit(wcfg, dict(sorted(G.training_files(wcfg).items())), (0,), log=QUIET)


@pytest.mark.slow
def test_serve_refuses_scored_rows_without_a_matched_departure(wcfg, wscored):
    """BC-31 sibling (b): the MJS body's witness is the scored file's own airport-hour matched departures; a scored input
    with none is refused loudly by the shipped unm_congestion_ship.attach_serve_witness (protecting invariant: its
    `len(wit) == 0` refusal), never served with a silently empty witness."""
    from prc.pipeline.fm1 import engine as G
    art = api.fit(wcfg, save=False, log=QUIET)
    only = wscored[~((wscored.PHASE_mvt == "DEP") & wscored.AOBT_3_flt.notna())].reset_index(drop=True)
    cfg = art.config.replace(**{"lanes.matched": False})
    led = S.Ledger(U.unmatched_ids(only))
    with pytest.raises(ValueError, match="the serve frame has no matched departure"):
        U.serve(art.unmatched_lane, only, cfg, led, {s.name: s for s in S.plan(cfg)}, log=QUIET)


# ---------------------------------------------------------------------------------------------- the REAL replay

@pytest.mark.devtree
@pytest.mark.slow
def test_unmatched_lane_replays_every_shipped_step_on_all_5290_scored_rows():
    """Per-component replay (Amendment A3): FM1 fits the lane from the twelve raw 2025 files and serves the 2026 scored
    file with the v36 replay configuration (storm anchor, Rome on, ARC at LIRF). TRUE shape (predicted exact, URP serve
    parity + logs/fm1_port_scratch/explore_chain_20260928T184444.json):
      fit      unm / lirf == data/cache_rome (the shipped cache), witness == unm_physics_lomo; beta_W == e2w_report (1e-6)
      body     non-LIRF after storm == v26 on all 4,907 rows
      rome     LIRF comp v7 / v10 / v12 == v7 / v10 / v12; after arc == v17 (42 ARC rows)
      e2, e2w  == v31 / v33 on every LIRF row except the 8 ARN rows (ARN dropped: they keep Rome/E2 values)
    FALSE: any other row off by more than 0 s. Runs in a child (3.3 GB peak)."""
    code = f"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, {str(ROOT)!r})
from prc.pipeline.fm1 import engine as G, knobs as K, stages as S, unmatched as U
cfg = K.parse_config({{"lanes": {{"matched": False}}, "modules": {{"storm": {{"mode": "anchor"}}}}}})
art = G.fit(cfg, save=False, allow_heavy=True, log=lambda *a: None)
m = art.unmatched_lane
ROOT = {str(ROOT)!r}
cu = pd.read_parquet(ROOT + '/data/cache_rome/unm.parquet'); cl = pd.read_parquet(ROOT + '/data/cache_rome/lirf.parquet')
def eq(a, b):   # every cache column, same dtype, same values, NaN where the cache has NaN
    return all(a[c].reset_index(drop=True).equals(b[c].reset_index(drop=True)) for c in b.columns)
w = pd.read_parquet(ROOT + '/data/cache_stand/unm_physics_lomo.parquet').set_index('MVT_ID_mvt').reindex(m.unm.MVT_ID_mvt)
beta_want = json.load(open(ROOT + '/reports/e2w_report.json'))['beta_W_all12']
rows = pd.read_parquet(cfg.path('data.scored_file'))
ids = U.unmatched_ids(rows)
led = S.Ledger(ids)
st = {{s.name: s for s in S.plan(cfg)}}
comp, info, snap = U.serve(m, rows, cfg, led, st, log=lambda *a: None)
out = led.finalise().set_index('MVT_ID_mvt')
def sub(v):
    f = {{'v7': 'v7', 'v10': 'v10', 'v12': 'v12', 'v17': 'v17', 'v26': 'v26', 'v31': 'v31_rse', 'v33': 'v33'}}[v]
    return pd.read_parquet(ROOT + f'/submissions/merry-quicksand_{{f}}.parquet').set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
ap = rows.set_index(rows.MVT_ID_mvt.to_numpy(float)).ADEP_mvt.reindex(ids).astype(str).to_numpy()
lirf, oth = ids[ap == 'LIRF'], ids[ap != 'LIRF']
def diff(got, ver, sel):
    want = sub(ver).reindex(sel).to_numpy(float)
    g = pd.Series(got).reindex(sel).to_numpy(float) if isinstance(got, pd.Series) else np.asarray(got, float)
    return sorted(float(i) for i in sel[g != want])
rep = {{
  'n': int(len(ids)), 'n_lirf': int(len(lirf)), 'n_other': int(len(oth)),
  'unm_equal_cache': bool(len(m.unm) == len(cu) and eq(m.unm, cu)), 'lirf_equal_cache': bool(len(m.lirf) == len(cl) and eq(m.lirf, cl)),
  'witness_equal': all(bool(np.array_equal(m.unm[c].to_numpy(float), w[c].to_numpy(float), equal_nan=True)) for c in ('hprox_med', 'hprox_n')),
  'beta_max_abs': float(np.max(np.abs(np.asarray(m.beta_w) - np.asarray(beta_want)))),
  'v7': diff(comp.rome_v7, 'v7', lirf), 'v10': diff(comp.rome_v10, 'v10', lirf), 'v12': diff(snap['rome'], 'v12', lirf),
  'v17': diff(snap['arc'], 'v17', lirf), 'v31': diff(snap['rome_e2'], 'v31', lirf), 'v33': diff(snap['rome_e2w'], 'v33', lirf),
  'v26': diff(snap['storm'], 'v26', oth), 'final_vs_v33': diff(out.TAXITIME_SEC_mvt.astype(float), 'v33', ids),
  'decided_by': led.counts(), 'n_arc': int((comp.arc_rule.isin(['ord', 'fill'])).sum()),
}}
import resource
rep['peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3
print(json.dumps(rep))
"""
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=900, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert (rep["n"], rep["n_lirf"], rep["n_other"]) == (5_290, 383, 4_907)
    assert rep["peak_rss_gib"] <= 4.5            # Amendment A3's bound (the shipped loader alone: 3.33 GB measured)
    assert rep["unm_equal_cache"] and rep["lirf_equal_cache"] and rep["witness_equal"]
    assert rep["beta_max_abs"] <= 1e-6
    for v in ("v7", "v10", "v12", "v17", "v26"):
        assert rep[v] == [], (v, rep[v][:5])
    for v in ("v31", "v33", "final_vs_v33"):
        assert rep[v] == sorted(ARN_IDS_2026), (v, rep[v][:10])
    # provenance, predicted from the registration's step table (URP: SCA changed 581 rows, ARC 42): the LIRF labels
    # partition the 383 LIRF rows, the body + storm labels the 4,907 others
    db = rep["decided_by"]
    assert set(db) <= S.DECIDED_BY and sum(db.values()) == 5_290
    assert rep["n_arc"] == 42 and db["arc"] == 42 and db["storm"] == 581
    assert db["unmatched_body"] + db["storm"] == 4_907
    assert db["rome"] + db["arc"] + db.get("rome_e2", 0) + db["rome_e2w"] == 383
