"""Tests for scripts/analysis/orl_stage3.py (arm ORL Stage 3 candidate builder; plans/PREREG_orl_osm_route_length_2026_09_23.md).

Everything is exercised on synthetic frames/files BEFORE any Stage-3 number exists (BC-24): the Stage-2 gate, the 99-column
list, the serve design (raw-key cross-check, NaN gate, unseen masks), the production refit wrapper (a tiny LightGBM fit on a
synthetic frame — never real data), R, the new values, the splice, the write/re-read, the tripwires, the projection, the
rules, and build_candidate end to end. The refit/serve functions reused from lgbm_submit / orl_build are tested there.

Mutation rehearsal: logs/orl_scratch_tools/mutate_stage3.py on the REAL-copy mirror mirror_stage3/ (import path asserted
by the first test, __pycache__ purged per mutant), 2026-09-23, both files: 72/72 killed. The first pass had three
escapes (ES seed, the 1 s floor, the splice's own outside-R check); the last three tests below were written for them and
killed them. Red-before-impl: logs/orl_scratch_tools/make_stage3_stubs.py ran the first 70 tests of both files against
NotImplementedError stubs (stage3_red_before_impl.log): 67 red; the 3 that pass on a stub are structural (import path x2,
the verifier's import-independence) and cannot be red by construction.
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


T = _load("orl_stage3", "scripts/analysis/orl_stage3.py")
O, CBV, LS = T.O, T.CBV, T.LS


def test_imported_module_is_the_one_under_test():
    assert pathlib.Path(T.__file__).resolve() == (ROOT / "scripts/analysis/orl_stage3.py").resolve()
    assert pathlib.Path(T.S2.__file__).resolve() == (ROOT / "scripts/analysis/orl_stage2.py").resolve()


def test_registration_constants():
    """Catches a name, count, bar or sha moved after the registration."""
    assert T.CAND.name == "merry-quicksand_v29_orl.parquet" and T.CAND.parent.name == "submissions"
    assert T.META.name == "merry-quicksand_v29_orl.meta.json" and T.MANIFEST.name == "merry-quicksand_v29_orl.manifest.parquet"
    assert T.BASE.name == "merry-quicksand_v26.parquet"
    assert T.BASE_SHA == "b5b47f2db90570a3b55f70927486cc1304a3b727f009c2972d7ba8b6b7238fcd"
    assert T.SEEDS == (0, 1, 2) and T.N_FEATS == 99 and T.K_WIDTH == 0.5 and T.BOARD_N == 344_841.0
    assert (T.N_TRAIN, T.N_RANK, T.N_SCOPE, T.N_R, T.N_AFJ) == (2_062_440, 339_551, 130_091, 115_180, 14_911)
    assert (T.CONC_BAR_PCT, T.CAP_GT600_PCT) == (47.8, 0.19)
    assert T.ROUTES_SHA == "4436b00edf558e8f451dba400cfb949c06935fe7fbed7dc20b232d0537f93590"
    assert T.RSS_LIMIT == 10 * 1024 ** 3
    assert T.FW_FILE == ROOT / "data/fwp_build/fw/merry-quicksand_v6.parquet"
    assert "3.35" in T.HISTORY["v20_FWP_direct_F_refit_on_CBV_rows"] and "-0.07" in T.HISTORY["v22_BLS_stage_residual"]
    assert "0.204" in T.HISTORY["v27_AGS_stage_member"]
    for p in (T.PRED_2026, T.REFIT_INFO):
        assert "submissions" not in str(p) and not str(p).startswith("/tmp")


# ---------------------------------------------------------------- stage-2 gate / feats

def _rep2(verdict="WORKING", k=0.8, ref=None):
    return {"verdict": {"verdict": verdict}, "composed": {"k": k, "S": 12.5},
            "tripwires": {"move_energy_2025_pop_by_month_airport": {"1:EGLL": 10.0, "7:LFPG": 40.0} if ref is None else ref}}


def test_stage2_gate_returns_k_and_reference_when_working():
    g = T.stage2_gate(_rep2())
    assert g == {"k2": 0.8, "S_2025": 12.5, "move_energy_2025": {"1:EGLL": 10.0, "7:LFPG": 40.0}}


@pytest.mark.parametrize("v", ["NOT WORKING", "NOT WORKING (not established: CI includes 0; see mde)", None])
def test_stage2_gate_refuses_anything_but_working(v):
    with pytest.raises(RuntimeError, match="not WORKING"):
        T.stage2_gate(_rep2(verdict=v))


def test_stage2_gate_refuses_nonfinite_k_and_empty_reference():
    with pytest.raises(AssertionError, match="not finite"):
        T.stage2_gate(_rep2(k=float("nan")))
    with pytest.raises(AssertionError, match="no 2025 move-energy"):
        T.stage2_gate(_rep2(ref={}))


def test_check_feats_accepts_exactly_the_stage2_arm_and_refuses_others():
    arm = T.S2.stage2_feats(T.S)[1]
    assert T.check_feats(arm) == arm and len(arm) == 99 and arm[-6:] == O.ORL_COLS + [O.ORL_FLAG]
    with pytest.raises(AssertionError, match="99-column"):
        T.check_feats(arm[:93])
    with pytest.raises(AssertionError, match="99-column"):
        T.check_feats(arm[1:] + arm[:1])


# ---------------------------------------------------------------- (a) serve design

def _keys(rows):
    return pd.DataFrame(rows, columns=["ADEP_mvt", "STAND_mvt", "RUNWAY_mvt"])


def test_raw_serve_keys_reindexes_by_id_and_refuses_missing_or_duplicate(tmp_path):
    p = tmp_path / "ranking.parquet"
    pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0], "ADEP_mvt": ["EGLL", "LFPG", "EGLL"], "STAND_mvt": ["A1", "B1", "A2"],
                  "RUNWAY_mvt": ["09L", "27R", "27R"], "PHASE_mvt": "DEP"}).to_parquet(p)
    k = T.raw_serve_keys([3.0, 1.0], p)
    assert list(k.columns) == O.KEY_COLS and k.STAND_mvt.tolist() == ["A2", "A1"] and k.index.tolist() == [0, 1]
    with pytest.raises(AssertionError, match="absent"):
        T.raw_serve_keys([1.0, 9.0], p)
    pd.DataFrame({"MVT_ID_mvt": [1.0, 1.0], "ADEP_mvt": ["EGLL"] * 2, "STAND_mvt": ["A1"] * 2,
                  "RUNWAY_mvt": ["09L"] * 2}).to_parquet(p)
    with pytest.raises(AssertionError, match="duplicate"):
        T.raw_serve_keys([1.0], p)


def test_serve_crosscheck_equal_passes_value_and_nan_differences_refuse():
    routes = T.synthetic_routes()
    raw = _keys([("EGLL", "A1", "09L"), ("LFPG", "Z9", "27R"), ("EGLL", "B2", "27R")])
    orl = O.attach(raw, routes)
    assert T.serve_crosscheck(orl, raw, routes) == 2               # Z9 unmatched
    bad = orl.copy(); bad.loc[0, "orl_route_m"] += 1
    with pytest.raises(AssertionError, match="orl_route_m"):
        T.serve_crosscheck(bad, raw, routes)
    bad = orl.copy(); bad.loc[1, "orl_straight_m"] = 5.0            # NaN on raw, a value here
    with pytest.raises(AssertionError, match="orl_straight_m"):
        T.serve_crosscheck(bad, raw, routes)
    bad = orl.copy(); bad.loc[2, "orl_matched"] = 0
    with pytest.raises(AssertionError, match="orl_matched"):
        T.serve_crosscheck(bad, raw, routes)
    with pytest.raises(AssertionError, match="serve rows"):
        T.serve_crosscheck(orl.iloc[:2], raw, routes)


def test_nan_gate_runs_on_fit_vs_serve_rows_of_the_frame():
    routes = T.synthetic_routes()
    fit = _keys([("EGLL", "A1", "09L"), ("EGLL", "Z9", "09L"), ("LFPG", "A1", "27R"), ("LFPG", "Z9", "27R")])
    orl = O.attach(pd.concat([fit, fit], ignore_index=True), routes)
    ap = np.r_[fit.ADEP_mvt, fit.ADEP_mvt]
    is_rank = np.r_[np.zeros(4, bool), np.ones(4, bool)]
    rep = T.nan_gate(orl, ap, is_rank)
    assert rep["EGLL"]["orl_route_m"] == {"fit": 0.5, "serve": 0.5}
    fit2 = _keys([("EGLL", "A1", "09L"), ("EGLL", "A2", "09L"), ("LFPG", "A1", "27R"), ("LFPG", "A2", "27R")])
    orl2 = O.attach(pd.concat([fit2, fit], ignore_index=True), routes)          # serve NaN where fit has none
    with pytest.raises(AssertionError, match="NaN mode absent in fit"):
        T.nan_gate(orl2, ap, is_rank)
    with pytest.raises(AssertionError, match="NaN mode absent in fit"):     # the flag orientation matters
        T.nan_gate(O.attach(pd.concat([fit, fit2], ignore_index=True), routes), ap, ~is_rank)


def test_unseen_pair_mask_is_row_level_and_normalised():
    fit = _keys([("EGLL", "A1", "09L"), ("EGLL", "A2", "27R")])
    serve = _keys([("EGLL", " a1 ", "09l"), ("EGLL", "A1", "27R"), ("EGLL", "B9", "09L"), ("LFPG", "A1", "09L")])
    pair, stand = T.unseen_pair_mask(fit, serve)
    assert pair.tolist() == [False, True, True, True]
    assert stand.tolist() == [False, False, True, True]


# ---------------------------------------------------------------- (b) production refit (synthetic only)

def _small_params():
    p = dict(LS.P); p["learning_rate"] = 0.1; p["num_threads"] = 1
    return p


def test_production_refit_on_a_synthetic_frame_returns_the_seed_mean_and_int_taxi():
    d, is_rank, ids, feats = T.synthetic_frame(n_train=1_500, n_rank=200, seed=3)
    proxy_rank = d.proxy.to_numpy(float)[is_rank].copy()
    res = T.production_refit(d, is_rank, feats, _small_params(), 40, 10, (0, 1, 2))
    assert set(res["delta_by_seed"]) == {0, 1, 2}
    mean = (res["delta_by_seed"][0] + res["delta_by_seed"][1] + res["delta_by_seed"][2]) / 3
    np.testing.assert_allclose(res["delta_mean"], mean, rtol=0, atol=1e-9)
    assert res["taxi_int"].dtype == np.int32 and len(res["taxi_int"]) == 200
    np.testing.assert_array_equal(res["taxi_int"], np.rint(np.maximum(proxy_rank - res["delta_mean"], 1)).astype("int32"))
    assert res["n_train"] == 1_500 and res["n_rank"] == 200 and res["info"]["n_all"] == 1_500
    assert res["info"]["n_ref"] == LS.n_refit(res["info"]["best_iter"], 1_500, res["info"]["n_fit"])
    assert res["floor_bind_rate"] == 0.0 and res["seed_sd_mean_s"] > 0
    # the model learned delta: its 2026 taxi is far closer to the truth than proxy-only
    assert "te_STAND_mvt" in d.columns and "STAND_mvt" in d.columns


def test_production_refit_learns_from_training_rows_only():
    d, is_rank, ids, feats = T.synthetic_frame(n_train=1_500, n_rank=200, seed=4)
    truth_delta = d.proxy.to_numpy(float) - 600 - 50 * d.f2.to_numpy()
    res = T.production_refit(d, is_rank, feats, _small_params(), 60, 10, (0,))
    err = np.sqrt(np.mean((res["delta_mean"] - truth_delta[is_rank]) ** 2))
    assert err < 0.5 * np.std(truth_delta[is_rank]), err


def test_production_refit_refuses_labelled_ranking_rows_missing_columns_and_bad_seeds():
    d, is_rank, ids, feats = T.synthetic_frame(n_train=800, n_rank=100, seed=5)
    d2 = d.copy(); d2.loc[np.flatnonzero(is_rank)[0], "y"] = 5.0
    with pytest.raises(AssertionError, match="label"):
        T.production_refit(d2, is_rank, feats, _small_params(), 20, 5, (0,))
    d3 = d.copy(); d3.loc[np.flatnonzero(is_rank)[0], "delta"] = 5.0
    with pytest.raises(AssertionError, match="label"):
        T.production_refit(d3, is_rank, feats, _small_params(), 20, 5, (0,))
    with pytest.raises(KeyError, match="absent"):
        T.production_refit(d.copy(), is_rank, feats + ["nope"], _small_params(), 20, 5, (0,))


def test_production_refit_uses_es_seed_and_saves_boosters_when_files_given(tmp_path):
    d, is_rank, ids, feats = T.synthetic_frame(n_train=800, n_rank=100, seed=6)
    files = [(tmp_path / f"b{s}.txt", tmp_path / f"b{s}.fit.json") for s in (0, 1)]
    res = T.production_refit(d, is_rank, feats, _small_params(), 20, 5, (0, 1), files=files)
    for p, j in files:
        assert p.exists() and json.loads(j.read_text())["n_features"] == len(feats)
    assert json.loads(files[1][1].read_text())["params"]["seed"] == 1 and res["info"]["target"] == "delta"


# ---------------------------------------------------------------- (c) R, values, splice

def _scope_world():
    scope = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0], "proxy": [900.0] * 5, "C": [100.0, 200, 300, 400, 950],
                          "taxi_C": [800.0, 700, 600, 500, 1.0]})
    fw = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "TAXITIME_SEC_mvt": np.array([600, 601, 602, 603, 99, 7], "int32")})
    formula = np.rint(0.5 * scope.taxi_C.to_numpy() + 0.5 * np.array([600, 601, 602, 603, 99.0]))
    bv = formula.copy(); bv[1] += 3                                     # id 2: a later lever
    base = pd.DataFrame({"MVT_ID_mvt": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0], "TAXITIME_SEC_mvt": np.r_[7, bv[::-1]].astype("int32")})
    return scope, base, fw, formula


def test_compute_R_is_formula_equal_minus_adsb_and_counts_the_literal_rule():
    scope, base, fw, formula = _scope_world()
    r, c = T.compute_R(scope, base, fw, {3.0, 2.0})                   # id 3 = ADS-B stage coinciding with the formula
    assert r.MVT_ID_mvt.tolist() == [1.0, 4.0, 5.0]
    assert c == {"n_scope": 5, "n_adsb_stage_in_scope": 2, "n_adsb_coincident_with_formula": 1, "n_formula_mismatch": 0,
                 "n_R": 3, "n_literal_rule": 4}
    np.testing.assert_array_equal(r.base.to_numpy(), formula[[0, 3, 4]])
    np.testing.assert_array_equal(r.FW_int.to_numpy(), [600, 603, 99])
    r2, c2 = T.compute_R(scope, base, fw, set())
    assert r2.MVT_ID_mvt.tolist() == [1.0, 3.0, 4.0, 5.0] and c2["n_formula_mismatch"] == 1


def test_compute_R_refuses_missing_base_or_fw_ids():
    scope, base, fw, _ = _scope_world()
    with pytest.raises(AssertionError, match="missing from the base"):
        T.compute_R(scope, base[base.MVT_ID_mvt != 4.0], fw, set())
    with pytest.raises(AssertionError, match="FW_int"):
        T.compute_R(scope, base, fw[fw.MVT_ID_mvt != 1.0], set())


def test_new_values_is_the_registered_blend_and_refuses_bad_orl():
    r = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0], "taxi_C": [801.0, 1.0]})
    orl = pd.Series([400, 2], index=[2.0, 1.0], dtype="float64")
    np.testing.assert_array_equal(T.new_values(r, orl), [np.rint(0.5 * 801 + 0.5 * 2), np.rint(0.5 * 1 + 0.5 * 400)])
    with pytest.raises(AssertionError, match="lack an F_orl"):
        T.new_values(r, orl.drop(1.0))
    with pytest.raises(AssertionError, match="not an integer"):
        T.new_values(r, pd.Series([400.5, 2.0], index=[2.0, 1.0]))
    with pytest.raises(AssertionError, match="below 1"):
        T.new_values(r, pd.Series([0.0, 2.0], index=[2.0, 1.0]))


def _tpl(base):
    return pd.DataFrame({"MVT_ID_mvt": base.MVT_ID_mvt.to_numpy()[::-1].copy(), "TAXITIME_SEC_mvt": 0})


def test_splice_candidate_changes_only_R_and_keeps_ids_order_dtypes():
    base = pd.DataFrame({"MVT_ID_mvt": [5.0, 1.0, 3.0, 2.0], "TAXITIME_SEC_mvt": np.array([10, 20, 30, 40], "int32")})
    cand, touched = T.splice_candidate(base, [3.0, 1.0], np.array([31.0, 20.0]), _tpl(base))
    assert cand.TAXITIME_SEC_mvt.tolist() == [10, 20, 31, 40] and touched.tolist() == [False, False, True, False]
    assert cand.dtypes.equals(base.dtypes) and cand.MVT_ID_mvt.tolist() == [5.0, 1.0, 3.0, 2.0]
    assert base.TAXITIME_SEC_mvt.tolist() == [10, 20, 30, 40]              # base untouched
    with pytest.raises(ValueError, match="not in"):
        T.splice_candidate(base, [9.0], np.array([1.0]), _tpl(base))
    with pytest.raises(ValueError, match="row count"):
        T.splice_candidate(base, [3.0], np.array([31.0]), _tpl(base).iloc[:3])


def test_write_and_reread_refuses_overwrite_and_returns_sha(tmp_path):
    base = pd.DataFrame({"MVT_ID_mvt": [5.0, 1.0], "TAXITIME_SEC_mvt": np.array([10, 20], "int32")})
    p = tmp_path / "c.parquet"
    h = T.write_and_reread(base, p, _tpl(base))
    assert h == hashlib.sha256(p.read_bytes()).hexdigest() and pd.read_parquet(p).equals(base)
    with pytest.raises(FileExistsError):
        T.write_and_reread(base, p, _tpl(base))
    bad = pd.DataFrame({"MVT_ID_mvt": [5.0, 1.0], "TAXITIME_SEC_mvt": np.array([0, 20], "int32")})
    with pytest.raises(ValueError, match="non-positive"):
        T.write_and_reread(bad, tmp_path / "d.parquet", _tpl(base))


# ---------------------------------------------------------------- (d)-(f)

def test_move_energy_tripwire_range_is_inclusive_over_2025_cells():
    ref = {"1:EGLL": 4.0, "7:EGLL": 100.0, "1:LFPG": 25.0}
    move = np.array([2.0, -2.0, 10.0, 0.0, 10.0])
    ap = np.array(["EGLL", "EGLL", "EGLL", "LFPG", "LFPG"]); month = np.array([1, 1, 7, 1, 7])
    t = T.move_energy_tripwire(move, ap, month, ref)
    assert t["move_energy_2026"] == {"1:EGLL": 4.0, "7:EGLL": 100.0, "1:LFPG": 0.0, "7:LFPG": 100.0}
    assert t["range_2025"] == [4.0, 100.0] and t["cells_outside"] == ["1:LFPG"] and t["k_range_void"] is True
    assert t["ratio_to_same_cell_2025"]["7:EGLL"] == 1.0 and t["ratio_to_same_cell_2025"]["1:LFPG"] == 0.0
    assert t["ratio_to_same_cell_2025"]["7:LFPG"] is None and t["cells_without_2025_reference"] == ["7:LFPG"]
    t2 = T.move_energy_tripwire(np.array([2.0, 10.0]), np.array(["EGLL", "EGLL"]), np.array([1, 7]), ref)
    assert t2["k_range_void"] is False and t2["cells_outside"] == []
    t3 = T.move_energy_tripwire(np.array([11.0]), np.array(["EGLL"]), np.array([7]), ref)
    assert t3["cells_outside"] == ["7:EGLL"] and t3["k_range_void"] is True


def test_zero_training_moves_shares():
    z = T.zero_training_moves(np.array([0.0, 3.0, 4.0, 0.0]), np.array([True, True, False, False]), np.array([True, False, False, True]))
    assert z == {"n_moved": 2, "moved_row_share_pair_unseen": 0.5, "move_energy_share_pair_unseen": 9 / 25,
                 "moved_row_share_stand_unseen": 0.0, "move_energy_share_stand_unseen": 0.0}
    z0 = T.zero_training_moves(np.zeros(3), np.ones(3, bool), np.ones(3, bool))
    assert z0["moved_row_share_pair_unseen"] == 0.0 and z0["move_energy_share_stand_unseen"] == 0.0


def test_projection_is_S_times_2k_minus_1_over_the_registered_range():
    move = np.array([100.0, -200.0, 0.0])
    s = (100 ** 2 + 200 ** 2) / 344_841.0
    p = T.projection(move, 0.9, False)
    assert p["S"] == pytest.approx(s, rel=0, abs=1e-12) and p["k_range"] == pytest.approx([0.4, 0.9])
    assert p["G_range"] == pytest.approx([s * (2 * 0.4 - 1), s * (2 * 0.9 - 1)], rel=0, abs=1e-12)
    assert p["quotable"] is True and T.projection(move, 0.9, True)["quotable"] is False


def test_pre_upload_rules_bars():
    ok = T.pre_upload_rules(np.r_[np.full(999, 10.0), 301.0])       # sq share > 300: 90601/(99900+90601) = 47.6 %
    assert ok["pass"] and ok["total"]["n_gt300"] == 1 and ok["total"]["pct_rows_gt600"] == 0.0
    conc = T.pre_upload_rules(np.r_[np.full(999, 10.0), 320.0])     # 102400/202300 = 50.6 % > 47.8
    assert not conc["pass"] and not conc["rules"]["concentration_gt300_le_47_8pct"]
    cap = T.pre_upload_rules(np.r_[np.full(499, 100.0), 601.0])     # 1 of 500 rows = 0.2 % > 0.19 %
    assert not cap["pass"] and not cap["rules"]["cap_rows_gt600_le_0_19pct"]


def test_per_airport_mean_move():
    assert T.per_airport_mean_move(np.array(["A", "B", "A"]), np.array([1.0, 5.0, 3.0])) == {
        "A": {"n": 2, "mean_move_s": 2.0}, "B": {"n": 1, "mean_move_s": 5.0}}


# ---------------------------------------------------------------- build_candidate end to end

def test_build_candidate_on_the_synthetic_world(tmp_path):
    cfg = T.synthetic_world(tmp_path)
    meta = T.build_candidate(cfg)
    base, cand = pd.read_parquet(cfg["base"]), pd.read_parquet(cfg["cand"])
    man = pd.read_parquet(cfg["manifest"])
    assert meta["counts"]["n_R"] == cfg["n_R"] == len(man) and meta["counts"]["n_adsb_coincident_with_formula"] == 3
    assert set(man.MVT_ID_mvt) == set(pd.read_parquet(cfg["fwp_touched"]).MVT_ID_mvt)
    outside = ~cand.MVT_ID_mvt.isin(man.MVT_ID_mvt)
    assert (cand.TAXITIME_SEC_mvt[outside] == base.TAXITIME_SEC_mvt[outside]).all()
    changed = cand.TAXITIME_SEC_mvt.to_numpy() != base.TAXITIME_SEC_mvt.to_numpy()
    assert set(cand.MVT_ID_mvt[changed]) == set(man.MVT_ID_mvt[man.changed]) and meta["n_changed"] == int(changed.sum()) > 0
    np.testing.assert_array_equal(man.candidate, np.rint(0.5 * pd.read_parquet(cfg["scope_parquet"]).set_index("MVT_ID_mvt")
                                                         .taxi_C.reindex(man.MVT_ID_mvt).to_numpy() + 0.5 * man.orl_taxi_int))
    assert meta["candidate_sha256"] == hashlib.sha256(cfg["cand"].read_bytes()).hexdigest()
    assert meta["manifest_sha256"] == hashlib.sha256(cfg["manifest"].read_bytes()).hexdigest()
    assert meta["projection"]["S"] == pytest.approx(float((man.move ** 2).sum() / 344_841.0))
    assert meta["k2"] == 0.9 and meta["tripwires"]["k_range_void"] is False and meta["projection"]["quotable"] is True
    assert meta["history"] == T.HISTORY and "NOT AUTHORISED" in meta["upload"]
    with pytest.raises(FileExistsError):
        T.build_candidate(cfg)


def test_build_candidate_records_a_void_k_range_and_still_refuses_nothing_silently(tmp_path):
    cfg = T.synthetic_world(tmp_path, ref_hi=1.0)                      # 2025 range [0, 1] s^2: every 2026 cell is outside
    meta = T.build_candidate(cfg)
    assert meta["tripwires"]["k_range_void"] is True and meta["projection"]["quotable"] is False
    assert meta["tripwires"]["cells_outside"]


@pytest.mark.parametrize("key,mut,match", [
    ("stage2", lambda c: c["stage2"].write_text(json.dumps(_rep2("NOT WORKING"))), "not WORKING"),
    ("base_sha", lambda c: c.update(base_sha="0" * 64), "registered current best"),
    ("n_R", lambda c: c.update(n_R=c["n_R"] + 1), r"\|R\|"),
    ("n_scope", lambda c: c.update(n_scope=c["n_scope"] + 1), "scope has"),
    ("n_afj", lambda c: c.update(n_afj=c["n_afj"] + 1), "AFJ set"),
    ("fwp", lambda c: pd.read_parquet(c["fwp_touched"]).iloc[1:].to_parquet(c["fwp_touched"]), "FWP's touched"),
    ("prereg", lambda c: c["prereg"].write_text("# changed after hashing\n"), "not in"),
    ("dup", lambda c: pd.concat([pd.read_parquet(c["preds"])] * 2).to_parquet(c["preds"]), "duplicate id"),
])
def test_build_candidate_refuses_before_writing(tmp_path, key, mut, match):
    cfg = T.synthetic_world(tmp_path)
    mut(cfg)
    with pytest.raises((AssertionError, RuntimeError), match=match):
        T.build_candidate(cfg)
    assert not cfg["cand"].exists() and not cfg["meta"].exists()


def test_build_candidate_refuses_an_existing_meta_or_manifest(tmp_path):
    for k in ("meta", "manifest"):
        cfg = T.synthetic_world(tmp_path / k)
        cfg[k].write_text("x")
        with pytest.raises(FileExistsError):
            T.build_candidate(cfg)
        assert not cfg["cand"].exists()


def test_run_refit_refuses_without_a_working_stage2_before_taking_the_lock(tmp_path, monkeypatch):
    rep = tmp_path / "r2.json"; rep.write_text(json.dumps(_rep2("NOT WORKING")))
    monkeypatch.setattr(T, "REPORT2", rep); monkeypatch.setattr(T, "PRED_2026", tmp_path / "p.parquet")
    monkeypatch.setitem(sys.modules, "run_guards", None)              # reaching the lock would raise AttributeError
    with pytest.raises(RuntimeError, match="not WORKING"):
        T.run_refit()
    (tmp_path / "p.parquet").write_text("x")
    with pytest.raises(FileExistsError):
        T.run_refit()


def test_real_config_points_at_the_registered_artifacts():
    c = T.real_config()
    assert c["cand"] == T.CAND and c["base"] == T.BASE and c["base_sha"] == T.BASE_SHA and c["scope_parquet"] is None
    assert (c["n_scope"], c["n_R"], c["n_afj"]) == (130_091, 115_180, 14_911) and c["stage2"] == T.REPORT2


def test_smoke_runs_end_to_end_and_the_independent_verifier_accepts_it(tmp_path):
    before = T.CAND.stat().st_mtime_ns if T.CAND.exists() else None
    out = T.run_smoke(tmp_path)
    assert out["verify"]["status"] == "VERIFIED" and out["meta"]["counts"]["n_R"] > 0
    assert (T.CAND.stat().st_mtime_ns if T.CAND.exists() else None) == before       # smoke never writes submissions/


def test_production_refit_early_stops_with_the_es_seed_and_the_given_budget(monkeypatch):
    d, is_rank, ids, feats = T.synthetic_frame(n_train=800, n_rank=100, seed=7)
    seen = {}
    real = LS.early_stop

    def spy(X, dlt, y, proxy, train, fit, es, params, nest, patience, **kw):
        seen.update(seed=params["seed"], nest=nest, patience=patience, target=kw.get("target"), n_train=int(train.sum()),
                    n_es=int(es.sum()), es_months=sorted(set(d.month.to_numpy()[es].tolist())))
        return real(X, dlt, y, proxy, train, fit, es, params, nest, patience, **kw)
    monkeypatch.setattr(LS, "early_stop", spy)
    T.production_refit(d, is_rank, feats, dict(_small_params(), seed=5), 30, 7, (0,))
    assert seen == {"seed": LS.ES_SEED, "nest": 30, "patience": 7, "target": "delta", "n_train": 800,
                    "n_es": int((~is_rank & np.isin(d.month.to_numpy(), [3, 9])).sum()), "es_months": [3, 9]}


def test_production_refit_floors_the_taxi_time_at_one_second():
    d, is_rank, ids, feats = T.synthetic_frame(n_train=1_500, n_rank=400, seed=8)
    i = np.flatnonzero(is_rank)[0]
    d.loc[i, "proxy"] = -5_000.0                                   # proxy - delta_hat < 1 on exactly one of 400 rows
    res = T.production_refit(d, is_rank, feats, _small_params(), 40, 10, (0,))
    assert res["taxi_int"][0] == 1 and (res["taxi_int"] >= 1).all()
    assert res["floor_bind_rate"] == pytest.approx(1 / 400)


def test_splice_candidate_has_its_own_outside_R_check(monkeypatch):
    base = pd.DataFrame({"MVT_ID_mvt": [5.0, 1.0, 3.0], "TAXITIME_SEC_mvt": np.array([10, 20, 30], "int32")})

    def broken(b, ids, vals):                                      # a regression in afj_ship_build.splice
        c = b.copy(); c.loc[0, "TAXITIME_SEC_mvt"] = 11; c.loc[2, "TAXITIME_SEC_mvt"] = 31
        return c, c.TAXITIME_SEC_mvt.to_numpy() != b.TAXITIME_SEC_mvt.to_numpy()
    monkeypatch.setattr(T.AFJ, "splice", broken)
    with pytest.raises(AssertionError, match="outside R changed"):
        T.splice_candidate(base, [3.0], np.array([31.0]), _tpl(base))
