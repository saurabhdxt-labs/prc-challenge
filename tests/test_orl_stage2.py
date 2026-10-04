"""Tests for scripts/analysis/orl_stage2.py (arm ORL Stage 2; plans/PREREG_orl_osm_route_length_2026_09_23.md).

Stage 2 is the RCR2 / DFC.3 fold-A code path with `F_orl` (the production 93 + the six ORL columns) against the stored
FW fold prediction. The fold loader, fit, composition, delta statistics and incumbent come from rcr2_build / dfc2_build
(tested there). What is new — the T1 read of Stage 1, the 99-column list, the matched/unmatched split, the seed summary,
the floor rate, the verdict merge, prediction scoring, the 2026 serve keys, the label-free tripwires and the
post-compute scoring of the fold predictions — is tested here on synthetic frames BEFORE any Stage-2 number exists
(BC-24: every line after the compute is exercised by `test_score_fold_*` before the first real run). No test fits.

Mutation rehearsal 2026-09-23 (logs/orl_scratch_tools/mutate_stage2.py on the REAL-copy mirror mirror_stage2/, imported
path asserted by the first test, __pycache__ purged per mutant): 36/36 killed; the one first-pass escape (the 99-width
check, unreachable through arm_feats) is now killed by test_stage2_feats_refuses_an_arm_that_is_not_93_plus_6. Every
test here was seen red against a NotImplementedError stub (logs/orl_scratch_tools/stage2_red_before_impl.log).
"""
from __future__ import annotations

import hashlib, importlib.util, json, pathlib, sys
import numpy as np, pandas as pd, pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(n, r):
    if n in sys.modules:
        return sys.modules[n]
    s = importlib.util.spec_from_file_location(n, ROOT / r); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m


T = _load("orl_stage2", "scripts/analysis/orl_stage2.py")
S1 = T.S1
R = T.R


def test_imported_module_is_the_one_under_test():
    assert pathlib.Path(T.__file__).resolve() == (ROOT / "scripts/analysis/orl_stage2.py").resolve()
    assert pathlib.Path(S1.__file__).resolve() == (ROOT / "scripts/analysis/orl_stage1.py").resolve()
    assert pathlib.Path(R.__file__).resolve() == (ROOT / "scripts/analysis/rcr2_build.py").resolve()


def test_registration_constants():
    """Catches a seed, bar, artifact path or registered sha moved after the hash."""
    assert T.SEEDS == (0, 1, 2) and T.FW_BEST_ITER == 21_101 and T.N_SERVE == 339_551 and T.SMOKE_STRIDE == 40
    assert T.PREREG == S1.PREREG and T.HASHLOG == S1.HASHLOG and T.REPORT1 == S1.REPORT1
    assert T.REPORT2.name == "orl_stage2_report.json" and T.PREDS2.name == "orl_fold_predictions.parquet"
    assert T.REPORT2_SMOKE.name == "orl_stage2_report_smoke.json" and T.PREDS2_SMOKE.name == "orl_fold_predictions_smoke.parquet"
    assert T.ROUTES_SHA == "4436b00edf558e8f451dba400cfb949c06935fe7fbed7dc20b232d0537f93590"
    assert T.RSS_LIMIT == 6 * 1024 ** 3 and T.WALL_MAX_S == 3.0 * 3600


def test_prereg_sha_accepts_the_logged_file_and_refuses_an_unlogged_one(tmp_path):
    p = tmp_path / "PREREG_x.md"; p.write_text("# reg\n")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    log = tmp_path / "log"; log.write_text(f"2026  REGISTERED  PREREG_x.md  sha256={h[:8]}  (full {h})  -- x\n")
    assert T.prereg_sha(p, log) == h
    p.write_text("# reg\nappended without a hash line\n")
    with pytest.raises(AssertionError, match="not in"):
        T.prereg_sha(p, log)
    log.write_text(f"2026  REGISTERED  OTHER.md  (full {hashlib.sha256(p.read_bytes()).hexdigest()})\n")
    with pytest.raises(AssertionError, match="not in"):
        T.prereg_sha(p, log)


@pytest.mark.devtree
def test_prereg_sha_on_the_real_registration():
    """The on-disk registration's current sha is logged (every append was hashed)."""
    assert len(T.prereg_sha()) == 64


def _s1(g_jul=1.0, t3_jul=0.9, m_jul=1.3, g_jan=1.4, t3_jan=1.2, m_jan=1.6):
    return {"L-Jul": {"arm_vs_ctrl": {"gain_s": g_jul, "gain_minus_top3_s": t3_jul, "gain_matched_rows_s": m_jul}},
            "L-Jan": {"arm_vs_ctrl": {"gain_s": g_jan, "gain_minus_top3_s": t3_jan, "gain_matched_rows_s": m_jan}}}


def test_stage1_t1_is_g1_plus_matched_rows_nonnegative_both_legs():
    """Catches T1 read as G1 alone (the registration adds matched-row gain >= 0 in BOTH legs)."""
    assert T.stage1_t1(_s1()) == {"G1": True, "matched_rows_nonneg": True, "pass": True}
    assert T.stage1_t1(_s1(m_jan=-0.01)) == {"G1": True, "matched_rows_nonneg": False, "pass": False}
    assert T.stage1_t1(_s1(t3_jul=-0.01))["G1"] is False
    assert T.stage1_t1(_s1(g_jan=0.0, t3_jan=0.0, m_jul=0.0))["pass"] is True


@pytest.mark.devtree
def test_stage1_t1_on_the_real_stage1_report():
    s1 = json.loads(T.REPORT1.read_text())
    assert T.stage1_t1(s1)["pass"] is True and s1["gate_G1_pass"] is True


class _FakeS:
    BASELINE_FEATS = [f"b{i}" for i in range(68)]; QUEUE_FEATS = [f"q{i}" for i in range(12)]
    ORDER_FEATS = ["o0", "o1", "o2"]; WEATHER_FEATS = [f"w{i}" for i in range(10)]


def test_stage2_feats_returns_the_93_control_and_the_99_arm():
    """Catches the arm fitted without the ORL columns (it would silently reproduce FW) or a column lost."""
    ctrl, arm = T.stage2_feats(_FakeS)
    assert len(ctrl) == 93 and len(arm) == 99 and arm[:93] == ctrl and arm[93:] == S1.ARM_COLS


def test_stage2_feats_refuses_a_non_93_production_list():
    class Short(_FakeS):
        WEATHER_FEATS = [f"w{i}" for i in range(9)]
    with pytest.raises(AssertionError, match="93"):
        T.stage2_feats(Short)


def test_stage2_feats_refuses_an_arm_that_is_not_93_plus_6(monkeypatch):
    """Catches orl_stage1's ARM_COLS changing under Stage 2 (the registered arm is exactly six columns)."""
    monkeypatch.setattr(S1, "ARM_COLS", S1.ARM_COLS[:5])
    with pytest.raises(AssertionError, match="93 \\+ 6"):
        T.stage2_feats(_FakeS)


def _rows(n=400, seed=3):
    rng = np.random.default_rng(seed)
    y = rng.uniform(300, 1500, n); base = y + rng.normal(0, 120, n); cand = y + rng.normal(0, 100, n)
    doy = np.where(np.arange(n) % 2 == 0, rng.integers(1, 31, n), rng.integers(182, 212, n))
    month = np.where(doy <= 31, 1, 7)
    ap = np.array(["LFPG", "EDDF", "LIRF", "EDDM"])[np.arange(n) % 4]
    matched = (np.arange(n) % 5 != 0)
    return y, doy, ap, month, base, cand, matched


def test_split_by_matched_equals_delta_stats_on_each_subset_and_sums_to_the_total():
    """Catches the split computed on the wrong rows, or the two halves not partitioning the total."""
    y, doy, ap, month, base, cand, m = _rows()
    out = T.split_by_matched(y, doy, ap, month, base, cand, m)
    want_m = R.delta_stats(y[m], doy[m], ap[m], month[m], base[m], cand[m])
    assert out["matched"]["delta"] == pytest.approx(want_m["delta"], rel=1e-12)
    assert out["matched"]["n"] == int(m.sum()) and out["unmatched"]["n"] == int((~m).sum())
    total = R.delta_stats(y, doy, ap, month, base, cand)["delta"]
    assert out["matched"]["delta"] + out["unmatched"]["delta"] == pytest.approx(total, rel=1e-9)
    assert out["matched_share"] == pytest.approx(m.mean())


def test_split_by_matched_all_matched_gives_no_unmatched_block_and_zero_matched_refuses():
    y, doy, ap, month, base, cand, m = _rows()
    assert T.split_by_matched(y, doy, ap, month, base, cand, np.ones(len(y), bool))["unmatched"] is None
    with pytest.raises(AssertionError, match="no matched rows"):
        T.split_by_matched(y, doy, ap, month, base, cand, np.zeros(len(y), bool))


def test_seed_summary_rmses_mean_sd_and_spread():
    """Catches the seed RMSE scored on the raw delta instead of the floored taxi time, or the spread misread."""
    y = np.array([100.0, 200.0]); proxy = np.array([150.0, 260.0])
    preds = {0: np.array([50.0, 60.0]), 1: np.array([40.0, 60.0]), 2: np.array([200.0, 60.0])}
    s = T.seed_summary(y, proxy, preds, (0, 1, 2))
    # seed 0 exact; seed 1 off by 10 on row 0; seed 2: max(150-200, 1) = 1 -> off by 99 on row 0
    assert s["rmse"] == pytest.approx([0.0, np.sqrt(50.0), np.sqrt(99.0 ** 2 / 2)])
    assert s["mean"] == pytest.approx(np.mean(s["rmse"])) and s["sd"] == pytest.approx(np.std(s["rmse"], ddof=1))
    assert s["spread"] == pytest.approx(max(abs(v - s["mean"]) for v in s["rmse"]))


def test_floor_bind_rate_counts_rows_where_the_floor_binds():
    assert T.floor_bind_rate(np.array([100.0, 100.0, 100.0, 100.0]), np.array([0.0, 99.0, 99.5, 150.0])) == 0.5


def _comp(delta=50.0, lo=10.0, per_month=None, top3=40.0, exl=30.0):
    return {"delta": delta, "ci95": [lo, delta + (delta - lo)], "ci_excludes_zero": bool(lo > 0),
            "per_month": per_month or {1: 20.0, 7: 30.0}, "delta_minus_top3": top3, "delta_ex_LFPG": exl}


def test_verdict_working_only_when_every_registered_clause_holds():
    fd = np.array([1.0, 2.0])
    v = T.verdict({"pass": True}, _comp(), _comp(), 0.001, [200.0, 200.5, 201.0], fd)
    assert v["verdict"] == "WORKING" and all(v["clauses"].values())
    assert set(v["clauses"]) == {"T1_stage1", "T2_comp_positive_ci", "T3_both_months_nonneg", "T4_minus_top3_and_ex_LFPG",
                                 "T5_F_alone_positive_ci", "T6_floor_seeds_finite"}


@pytest.mark.parametrize("kw, clause", [
    (dict(t1={"pass": False}), "T1_stage1"),
    (dict(comp=_comp(per_month={1: -0.1, 7: 30.0})), "T3_both_months_nonneg"),
    (dict(comp=_comp(top3=-1.0)), "T4_minus_top3_and_ex_LFPG"),
    (dict(comp=_comp(exl=0.0)), "T4_minus_top3_and_ex_LFPG"),
    (dict(floor=0.005), "T6_floor_seeds_finite"),
    (dict(seeds=[200.0, 200.0, 203.1]), "T6_floor_seeds_finite"),
    (dict(fd=np.array([1.0, np.nan])), "T6_floor_seeds_finite"),
])
def test_verdict_each_clause_failing_alone_is_not_working(kw, clause):
    a = dict(t1={"pass": True}, comp=_comp(), fa=_comp(), floor=0.001, seeds=[200.0, 200.5, 201.0], fd=np.array([1.0]))
    a.update(kw)
    v = T.verdict(a["t1"], a["comp"], a["fa"], a["floor"], a["seeds"], a["fd"])
    assert v["verdict"].startswith("NOT WORKING") and v["clauses"][clause] is False
    assert sum(not x for x in v["clauses"].values()) == 1


def test_verdict_ci_including_zero_is_worded_not_established():
    v = T.verdict({"pass": True}, _comp(delta=20.0, lo=-5.0), _comp(), 0.001, [200.0] * 3, np.array([1.0]))
    assert v["verdict"] == "NOT WORKING (not established: CI includes 0; see mde)" and v["clauses"]["T2_comp_positive_ci"] is False


def test_verdict_flags_the_blend_only_shape_when_t5_fails_and_t2_passes():
    """FALSE shape: 'T5 fails while T2 passes -> NOT WORKING and flagged (blend-only gain, v22's shape)'."""
    v = T.verdict({"pass": True}, _comp(), _comp(delta=5.0, lo=-3.0), 0.001, [200.0] * 3, np.array([1.0]))
    assert v["verdict"] == "NOT WORKING" and v["blend_only_flag"] is True
    assert T.verdict({"pass": True}, _comp(), _comp(), 0.001, [200.0] * 3, np.array([1.0]))["blend_only_flag"] is False


def _rep(dF=50.0, dC=40.0, hw=60.0, best=21_101, top1=0.1, secs=7000.0, rss=5.0):
    return {"f_alone": {"delta": dF}, "composed": {"delta": dC, "ci95": [dC - hw, dC + hw], "top1_share": top1},
            "best_iter": best, "seconds": secs, "peak_rss_GiB": rss}


def test_predictions_scored_bands_and_boundaries():
    """P5 bands, P6 best_iter and top-row share, P7 (Stage 2 part) wall and RSS, at and just past each edge."""
    p = T.predictions_scored(_rep())
    assert p == {"P5_dF_band": "HIT", "P5_dcomp_band": "HIT", "P5_halfwidth": "HIT", "P5": "HIT",
                 "P6_best_iter": "HIT", "P6_top1_share": "HIT", "P6": "HIT", "P7_stage2_wall_rss": "HIT"}
    assert T.predictions_scored(_rep(dF=-80.0, dC=120.0, hw=15.0))["P5"] == "HIT"
    assert T.predictions_scored(_rep(dF=200.1))["P5_dF_band"] == "MISS"
    assert T.predictions_scored(_rep(dC=-40.1))["P5_dcomp_band"] == "MISS"
    assert T.predictions_scored(_rep(hw=120.1))["P5"] == "MISS"
    assert T.predictions_scored(_rep(hw=14.9))["P5_halfwidth"] == "MISS"
    assert T.predictions_scored(_rep(best=int(0.8 * 21_101) - 1))["P6"] == "MISS"
    assert T.predictions_scored(_rep(best=int(1.2 * 21_101) + 1))["P6_best_iter"] == "MISS"
    assert T.predictions_scored(_rep(top1=0.25))["P6_top1_share"] == "MISS"
    assert T.predictions_scored(_rep(secs=10_800.1))["P7_stage2_wall_rss"] == "MISS"
    assert T.predictions_scored(_rep(rss=6.0))["P7_stage2_wall_rss"] == "MISS"


def _ranking(tmp_path, year=2026):
    t = f"{year}-01-05T10:00:00Z"
    d = pd.DataFrame({
        "PHASE_mvt": ["DEP", "DEP", "ARR", "DEP", "DEP", "DEP"],
        "ADEP_mvt": ["EDDF", "LIRF", "EDDF", "KJFK", "EDDF", "LFPG"],
        "STAND_mvt": ["A1", "B2", "A1", "C3", "A9", None],
        "RUNWAY_mvt": ["25C", "25", "25C", "04L", "18", "27L"],
        "MVT_TIME_UTC_mvt": [t, t, t, t, None, t],
        "SCHED_TIME_UTC_mvt": [t] * 6,
        "AOBT_3_flt": [t, t, t, t, t, t],
    })
    p = tmp_path / "ranking.parquet"; d.to_parquet(p, index=False); return p


def test_serve_keys_2026_applies_the_serve_filter_and_keeps_the_three_keys(tmp_path):
    """Catches ARR rows, non-competition airports or a missing movement time leaking into the serve key set."""
    k = T.serve_keys_2026(_ranking(tmp_path))
    assert list(k.columns) == ["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"]
    assert k.ADEP_mvt.tolist() == ["EDDF", "LIRF", "LFPG"] and k.STAND_mvt.tolist()[:2] == ["A1", "B2"]


def test_serve_keys_2026_refuses_a_non_2026_frame(tmp_path):
    with pytest.raises(AssertionError, match="2026"):
        T.serve_keys_2026(_ranking(tmp_path, year=2025))


def test_tripwires_move_energy_on_pop_rows_only_and_unseen_pairs():
    """Catches move energy averaged over non-CBV rows (diluting it: the registered reference is the pop rows), and the
    unseen-pair share read against the wrong fit set."""
    j = pd.DataFrame({"month": [1, 1, 1, 7], "ap": ["EDDF", "EDDF", "EDDF", "LIRF"], "pop": [True, True, False, True],
                      "inc": [100.0, 100.0, 100.0, 100.0]})
    cand = np.array([110.0, 120.0, 100.0, 103.0])
    fit_tr = pd.DataFrame({"ADEP_mvt": ["EDDF"], "STAND_mvt": ["A1"], "RUNWAY_mvt": ["25C"]})
    fit_all = pd.DataFrame({"ADEP_mvt": ["EDDF", "EDDF"], "STAND_mvt": ["A1", "A2"], "RUNWAY_mvt": ["25C", "25C"]})
    serve = pd.DataFrame({"ADEP_mvt": ["EDDF", "EDDF"], "STAND_mvt": ["A1", "A2"], "RUNWAY_mvt": ["25C", "25C"]})
    tw = T.tripwires(j, cand, T.unseen_pairs(fit_tr, fit_all, serve))
    assert tw["move_energy_2025_pop_by_month_airport"] == {"1:EDDF": 250.0, "7:LIRF": 9.0}
    assert tw["unseen_2026_vs_foldA_training"]["EDDF"]["pair_unseen"] == 0.5
    assert tw["unseen_2026_vs_all_2025"]["EDDF"]["pair_unseen"] == 0.0
    assert "Stage 3" in tw["move_energy_2026"]


# ---- the post-compute path, exercised before the first real run (BC-24) -------------------------------------------
def _synthetic_fold(n=240, seed=5):
    """A fold of n test rows (row ids with gaps, as te_idx has), an incumbent covering a subset of them, FW on all."""
    rng = np.random.default_rng(seed)
    row = np.sort(rng.choice(10_000, n, replace=False)).astype("int64")
    y = rng.uniform(300, 1500, n); proxy = y + rng.normal(0, 200, n)
    doy = np.where(np.arange(n) % 2 == 0, rng.integers(1, 31, n), rng.integers(182, 212, n))
    month = np.where(doy <= 31, 1, 7)
    ap = np.array(["LFPG", "EDDF", "LIRF"])[np.arange(n) % 3]
    base = pd.DataFrame({"row": row, "month": month, "ap": ap, "y": y, "proxy": proxy, "doy": doy,
                         "orl_matched": (np.arange(n) % 4 != 0).astype(np.int8)})
    for s in (0, 1, 2):
        base[f"d_seed{s}"] = (proxy - y) + rng.normal(0, 80, n)
    base["F_orl"] = base[[f"d_seed{s}" for s in (0, 1, 2)]].mean(axis=1)
    fw = (proxy - y) + rng.normal(0, 90, n)
    day = pd.to_datetime("2025-01-01") + pd.to_timedelta(doy - 1, unit="D")
    s = pd.DataFrame({"row": row, "ap": ap, "month": month, "day": day, "fold": 0.0, "y": y, "proxy": proxy,
                      "F": np.where(np.arange(n) % 3 == 0, 7.0, 5.0), "v13_delta": 5.0,
                      "taxi_v13_delta": y + rng.normal(0, 150, n), "C": (proxy - y) + rng.normal(0, 100, n), "FW": fw})
    inc = R.finish_incumbent(s.iloc[: n - 20], subset_ok=True)           # incumbent covers a subset (smoke shape)
    fw_all = pd.DataFrame({"row": row, "FW": fw, "doy": doy})
    return base, inc, fw_all


def test_score_fold_matches_hand_computed_composed_and_f_alone():
    """Catches a wrong join key, F_orl substituted on non-pop rows, or F-alone compared on the wrong scale."""
    base, inc, fw_all = _synthetic_fold()
    out = T.score_fold(base, inc, fw_all, smoke=True)
    j = inc.merge(base[["row", "F_orl", "orl_matched"]], on="row")
    cand = np.where(j["pop"], 0.5 * (j.taxi_C + np.maximum(j.proxy - j.F_orl, 1)), j.taxi_v13_delta)
    want = ((j.y - j.inc) ** 2 - (j.y - cand) ** 2).sum() / R.BOARD_N
    assert out["composed"]["delta"] == pytest.approx(want, rel=1e-10)
    assert out["composed"]["n"] == len(inc) and out["n_pop"] == int(inc["pop"].sum())
    fa_want = ((base.y - np.maximum(base.proxy - fw_all.FW, 1)) ** 2 - (base.y - np.maximum(base.proxy - base.F_orl, 1)) ** 2).sum() / R.BOARD_N
    assert out["f_alone"]["delta"] == pytest.approx(fa_want, rel=1e-10) and out["f_alone"]["n"] == len(base)
    m = j.orl_matched.to_numpy() == 1
    assert out["composed_by_match"]["matched"]["n"] == int(m.sum())
    assert out["composed_by_match"]["matched"]["delta"] + out["composed_by_match"]["unmatched"]["delta"] == pytest.approx(want, rel=1e-9)
    assert out["f_alone_by_match"]["matched"]["n"] == int((base.orl_matched == 1).sum())
    assert out["seeds"]["rmse"] == pytest.approx([R.rmse(base.y, np.maximum(base.proxy - base[f"d_seed{s}"], 1)) for s in (0, 1, 2)])
    assert out["floor_bind_rate"] == pytest.approx(float((base.proxy - base.F_orl < 1).mean()))
    assert np.array_equal(out["_cand"], cand) and len(out["_j"]) == len(j)


def test_score_fold_non_smoke_refuses_a_partial_cover_and_a_mismatched_row():
    """Catches the full run silently scoring on a subset of the incumbent, or a `row` pointing at a different flight."""
    base, inc, fw_all = _synthetic_fold()
    with pytest.raises(AssertionError, match="cover"):
        T.score_fold(base.iloc[5:], inc, fw_all, smoke=False)
    bad = base.copy(); bad.loc[bad.index[0], "y"] += 1.0
    with pytest.raises(AssertionError, match="same row"):
        T.score_fold(bad, inc, fw_all, smoke=True)
    with pytest.raises(AssertionError, match="FW"):
        T.score_fold(base, inc, fw_all.iloc[3:], smoke=False)


def test_score_fold_refuses_non_finite_f_orl():
    base, inc, fw_all = _synthetic_fold()
    base.loc[base.index[2], "F_orl"] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        T.score_fold(base, inc, fw_all, smoke=True)


# ---- orchestration refusals (before any lock or load) -------------------------------------------------------------
def test_run_stage2_refuses_a_silent_rerun(tmp_path, monkeypatch):
    p = tmp_path / "r.json"; p.write_text("{}")
    monkeypatch.setattr(T, "REPORT2", p)
    with pytest.raises(FileExistsError, match="no silent rerun"):
        T.run_stage2(smoke=False)


def test_run_stage2_refuses_when_stage1_t1_failed(tmp_path, monkeypatch):
    s1 = tmp_path / "s1.json"; s1.write_text(json.dumps(_s1(m_jan=-0.5)))
    monkeypatch.setattr(T, "REPORT1", s1); monkeypatch.setattr(T, "REPORT2", tmp_path / "absent.json")
    with pytest.raises(RuntimeError, match="T1"):
        T.run_stage2(smoke=True)


def test_main_refuses_unknown_args():
    with pytest.raises(SystemExit):
        T.main(["--bogus"])
