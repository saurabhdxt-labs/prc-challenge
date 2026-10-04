"""scripts/analysis/v34_compose.py — v34 step 7 (AMENDMENT V34.7): compose v34 from v33 in the v33 chain's lanes.

  N (224,371)  ADS-B-stage rows: v13 stage (v4 table) / AFJ (afj table) / AUN (union table) -> rint(max(proxy-(F+resid),1))
  R\\U (109,816) CBV rows on ORL:  rint(0.5*max(proxy-C,1) + 0.5*F_orl_int)
  U (5,364)     ORU revert rows:  rint(0.5*max(proxy-C,1) + 0.5*FW_int)
  everything else (5,290 unmatched incl. the Rome chain): BYTE-IDENTICAL to v33 (asserted)

The replay (old inputs) must reproduce v33: exactly on R / AFJ / AUN (anchored), within the v4-lane floors. Synthetic only.
Mutation rehearsal 2026-09-24: see docstrings."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V7 = _load("v34_compose", "scripts/analysis/v34_compose.py")


# ---------------------------------------------------------------- registered constants

@pytest.mark.devtree
def test_k_range_is_the_nsc_fold_result_recomputed():
    """k = sum(D*e)/sum(D^2) on data/nsc_run/nsc_fold_preds.parquet, day-block CI 2,000 draws seed 0 (registered)."""
    got = V7.fold_k(pd.read_parquet(ROOT / "data/nsc_run/nsc_fold_preds.parquet"))
    assert got["k"] == pytest.approx(V7.K_POINT, abs=1e-9)
    assert got["ci95_day"] == pytest.approx([V7.K_LO, V7.K_HI], abs=1e-9)
    assert (V7.K_LO, V7.K_POINT, V7.K_HI) == pytest.approx((0.61938549, 0.74106494, 0.86771431), abs=1e-7)
    assert V7.MDE_BMSE == 73.2


def test_lane_counts_and_shas_are_pinned():
    assert (V7.N_N, V7.N_R, V7.N_U, V7.N_M) == (224_371, 115_180, 5_364, 339_551)
    assert V7.V33_SHA.startswith("0786a41a") and V7.V19_SHA.startswith("5b5378b0")
    assert V7.KEY == "merry-quicksand_v34.parquet"


def test_projection_is_s_times_2k_minus_1_at_the_three_ks():
    move = np.array([3.0, -4.0, 0.0])
    p = V7.projection(move, (0.5, 0.75, 1.0))
    assert p["S"] == pytest.approx(25 / 344_841) and p["G"] == pytest.approx([0.0, 12.5 / 344_841, 25 / 344_841])
    assert p["clauses"]["WORKING"].startswith("G >= -73.2")


# ---------------------------------------------------------------- lanes

def test_table_source_routes_aun_then_afj_then_v4():
    """AUN beats AFJ (28 AUN rows are AFJ rows). Rehearsed RED: checking AFJ first."""
    src = V7.table_source(np.array([1.0, 2.0, 3.0, 4.0]), afj_ids={2.0, 3.0}, aun_ids={3.0, 4.0})
    assert src.tolist() == ["v4", "afj", "aun", "aun"]


def test_f_prime_uses_v10_where_the_stage_row_has_it_else_v13():
    ids = np.array([1.0, 2.0])
    f = V7.f_prime(ids, np.array([900.0, 800.0]), v10=pd.Series([700.0], index=[1.0]),
                   v13=pd.Series([650.0, 600.0], index=[1.0, 2.0]))
    assert f.tolist() == [200.0, 200.0]


# ---------------------------------------------------------------- a synthetic world

def _world(seed=0):
    rng = np.random.default_rng(seed)
    n_n, n_r, n_u, n_x = 30, 20, 5, 8
    ids_n = np.arange(n_n) + 1000.0
    ids_r = np.arange(n_r) + 2000.0                 # the last n_u of them are U
    ids_x = np.arange(n_x) + 3000.0                 # unmatched / Rome chain
    all_ids = np.r_[ids_n, ids_r, ids_x]
    proxy = pd.Series(rng.uniform(500, 900, len(all_ids)), index=all_ids)
    v33 = pd.DataFrame({"MVT_ID_mvt": all_ids[::-1].copy(),
                        "TAXITIME_SEC_mvt": rng.integers(300, 900, len(all_ids)).astype("int32")})
    tables = {k: pd.DataFrame({"coverage": np.where(np.arange(n_n) % 3 == 0, 1, 3), "f1": rng.normal(size=n_n) + i},
                              index=pd.Index(ids_n, name="MVT_ID")) for i, k in enumerate(("v4", "afj", "aun"))}
    rows = pd.DataFrame({"MVT_ID_mvt": ids_n, "ap": "EGLL", "hr": 7.0, "proxy": proxy.reindex(ids_n).to_numpy(),
                         "table": np.array(["v4", "afj", "aun"])[np.arange(n_n) % 3]})
    rrows = pd.DataFrame({"MVT_ID_mvt": ids_r, "proxy": proxy.reindex(ids_r).to_numpy(),
                          "in_U": np.r_[np.zeros(n_r - n_u, bool), np.ones(n_u, bool)]})
    return dict(v33=v33, tables=tables, rows=rows, rrows=rrows, ids_n=ids_n, ids_r=ids_r, ids_x=ids_x, rng=rng)


def _predict(frame, table):
    return np.where(frame.coverage.to_numpy() >= 2, 10.0 * frame.f1.to_numpy(), 0.0)


def _inputs(w, shift=0.0):
    n = len(w["ids_n"])
    F = pd.Series(np.linspace(-50, 50, n) + shift, index=w["ids_n"])
    C = pd.Series(np.linspace(-20, 20, len(w["ids_r"])) + shift, index=w["ids_r"])
    orl = pd.Series(np.arange(len(w["ids_r"])) + 600 + int(shift), index=w["ids_r"])
    fw = pd.Series(np.arange(len(w["ids_r"])) + 610 + int(shift), index=w["ids_r"])
    return dict(F=F, C=C, orl_int=orl, fw_int=fw)


def _gate():
    return {"EGLL": True}


def test_matched_values_follow_each_lanes_formula():
    """Rehearsed RED: U rows blended with F_orl instead of FW; N rows using resid 0 on the afj table."""
    w = _world()
    inp = _inputs(w)
    vals, detail = V7.matched_values(w["rows"], inp["F"], w["tables"], _predict, _gate(), w["rrows"], inp["C"],
                                     inp["orl_int"], inp["fw_int"])
    r = w["rows"].set_index("MVT_ID_mvt")
    t = pd.concat([w["tables"][s].loc[[i]] for i, s in zip(r.index, r.table)])
    resid = np.where(t.coverage.to_numpy() >= 2, 10.0 * t.f1.to_numpy(), 0.0)
    want_n = np.rint(np.maximum(r.proxy.to_numpy() - (inp["F"].reindex(r.index).to_numpy() + resid), 1.0))
    assert np.array_equal(vals.reindex(r.index).to_numpy(), want_n)
    rr = w["rrows"].set_index("MVT_ID_mvt")
    taxi_c = np.maximum(rr.proxy.to_numpy() - inp["C"].reindex(rr.index).to_numpy(), 1.0)
    partner = np.where(rr.in_U, inp["fw_int"].reindex(rr.index), inp["orl_int"].reindex(rr.index))
    assert np.array_equal(vals.reindex(rr.index).to_numpy(), np.rint(0.5 * taxi_c + 0.5 * partner))
    assert set(detail.lane) == {"N_v4", "N_afj", "N_aun", "R_orl", "U_fw"}
    assert len(vals) == len(w["ids_n"]) + len(w["ids_r"])


def test_splice_keeps_every_row_outside_the_matched_lanes_byte_identical():
    """The unmatched lane and the Rome chain carry no seconds: they must be v33's bytes. Rehearsed RED: a splice that
    reorders rows; a splice that rounds with astype(int) instead of the int32 dtype check."""
    w = _world()
    inp = _inputs(w)
    vals, _ = V7.matched_values(w["rows"], inp["F"], w["tables"], _predict, _gate(), w["rrows"], inp["C"],
                                inp["orl_int"], inp["fw_int"])
    cand = V7.splice(w["v33"], vals, outside_ids=w["ids_x"])
    assert np.array_equal(cand.MVT_ID_mvt.to_numpy(), w["v33"].MVT_ID_mvt.to_numpy()) and cand.dtypes.equals(w["v33"].dtypes)
    out = w["v33"].MVT_ID_mvt.isin(w["ids_x"]).to_numpy()
    assert np.array_equal(cand.TAXITIME_SEC_mvt.to_numpy()[out], w["v33"].TAXITIME_SEC_mvt.to_numpy()[out])
    bad = vals.copy()
    bad.loc[w["ids_n"][0]] = 0.5
    with pytest.raises(ValueError, match="integer"):
        V7.splice(w["v33"], bad, outside_ids=w["ids_x"])
    with pytest.raises(AssertionError, match="outside"):
        V7.splice(w["v33"], pd.concat([vals, pd.Series([5.0], index=[w["ids_x"][0]])]), outside_ids=w["ids_x"])


def test_replay_verdict_exact_lanes_and_the_v4_floors():
    """R / N_afj / N_aun exact; N_v4 >= 75 % exact and >= 99.5 % within 1 s after the floor rows. Rehearsed RED: the v4
    tolerance applied to R (a 1 s R mismatch passes); within-1 computed as < 1; floor rows not excluded."""
    ids_v4, ids_r, ids_afj = np.arange(1000) + 1.0, np.arange(10) + 5000.0, np.arange(5) + 7000.0
    idx = np.r_[ids_v4, ids_r, ids_afj]
    v33 = pd.Series(np.full(len(idx), 500.0), index=idx)
    lanes = {"R": ids_r, "N_v4": ids_v4, "N_afj": ids_afj}
    rep = v33.copy()
    rep.iloc[:4] += 1                                            # 4 of 1,000 v4 rows off by 1 s
    rep.iloc[4] += 300                                           # a floor row, excluded
    v = V7.replay_verdict(rep, v33, lanes, floor_ids={5.0})
    assert v["pass"] and v["N_v4"]["n"] == 999 and v["N_v4"]["n_floor_excluded"] == 1 and v["N_v4"]["within_1_share"] == 1.0
    bad = rep.copy()
    bad.loc[5000.0] += 1
    assert not V7.replay_verdict(bad, v33, lanes, floor_ids={5.0})["pass"]
    bad = rep.copy()
    bad.loc[7001.0] += 1
    assert not V7.replay_verdict(bad, v33, lanes, floor_ids={5.0})["pass"]
    bad = rep.copy()
    bad.iloc[10:16] += 2                                         # 0.6 % of v4 off by 2 s
    assert not V7.replay_verdict(bad, v33, lanes, floor_ids={5.0})["pass"]
    bad = rep.copy()
    bad.iloc[10:260] += 1                                        # exact share 74.6 % < 75 %
    assert not V7.replay_verdict(bad, v33, lanes, floor_ids={5.0})["pass"]


def test_anchored_value_is_aun4():
    assert V7.anchored_value([1000.0, 5.0], [10.0, 0.0], [30.0, 50.0]).tolist() == [980.0, 1.0]


def test_floor_ids_are_rows_whose_f_prime_base_is_at_the_floor():
    v10 = pd.Series([1.0, 400.0], index=[1.0, 2.0])
    v13 = pd.Series([1.0, 1.0, 300.0], index=[1.0, 3.0, 4.0])
    assert V7.floor_ids(np.array([1.0, 2.0, 3.0, 4.0]), v10, v13) == {1.0, 3.0}


def test_replay_values_anchor_the_aun_rows_on_v18():
    """Rehearsed RED: AUN rows valued directly on the union table (the v33 values came from AUN.4's anchor)."""
    w = _world()
    old = _inputs(w)
    ids = w["rows"].MVT_ID_mvt.to_numpy()
    old["v18"] = pd.Series(np.full(len(ids), 700.0), index=ids)
    vals, det = V7.replay_values(w["rows"], old, w["tables"], _predict, _gate(), w["rrows"], old["v18"])
    aun = w["rows"][w["rows"].table == "aun"]
    r = aun.MVT_ID_mvt.to_numpy()
    ru = det.set_index("MVT_ID_mvt").resid.reindex(r).to_numpy()
    t = w["tables"]["afj"].loc[r]
    rs = np.where(t.coverage.to_numpy() >= 2, 10.0 * t.f1.to_numpy(), 0.0)
    assert np.array_equal(vals.reindex(r).to_numpy(), np.rint(np.maximum(700.0 - (ru - rs), 1.0)))


# ---------------------------------------------------------------- end to end on the synthetic world (BC-24)

def test_build_writes_candidate_manifest_meta_and_final_and_they_agree(tmp_path):
    """Every line after the compute: replay gate, v34 values, splice, S/G, rules, writers, final byte copy."""
    w = _world()
    old, new = _inputs(w), _inputs(w, shift=3.0)
    old["v18"] = pd.Series(np.full(len(w["ids_n"]), 700.0), index=w["ids_n"])
    old_vals, _ = V7.replay_values(w["rows"], old, w["tables"], _predict, _gate(), w["rrows"], old["v18"])
    v33 = V7.splice(w["v33"], old_vals, outside_ids=w["ids_x"])       # a v33 the old inputs reproduce exactly
    paths = {k: tmp_path / f"{k}.x" for k in ("cand", "manifest", "meta", "final")}
    paths["final"] = tmp_path / "merry-quicksand_v34.parquet"
    template = pd.DataFrame({"MVT_ID_mvt": v33.MVT_ID_mvt.to_numpy()[::-1].copy(),
                             "TAXITIME_SEC_mvt": np.zeros(len(v33), "int32")})
    meta = V7.build(v33, template, w["rows"], w["rrows"], w["tables"], _gate(), old, new,
                    predict_old=_predict, predict_new=_predict, outside_ids=w["ids_x"], paths=paths,
                    extra_meta={"x": 1}, allow_roots=(tmp_path,), check_counts=False)
    cand = pd.read_parquet(paths["cand"])
    assert meta["status"] == "BUILT" and meta["replay"]["pass"]
    assert pd.read_parquet(paths["final"]).equals(cand) and meta["final_sha256"] == meta["candidate_sha256"]
    man = pd.read_parquet(paths["manifest"])
    assert list(man.columns) == V7.MANIFEST_COLS and len(man) == len(v33)
    move = cand.TAXITIME_SEC_mvt.to_numpy(float) - v33.TAXITIME_SEC_mvt.to_numpy(float)
    assert meta["projection"]["S"] == pytest.approx(float((move ** 2).sum()) / 344_841)
    assert meta["n_changed"] == int((move != 0).sum()) and meta["outside_identical"] is True
    assert json.loads(paths["meta"].read_text()) == meta and meta["upload_key"] == "merry-quicksand_v34.parquet"


def test_build_refuses_when_the_replay_does_not_reproduce_v33(tmp_path):
    w = _world()
    old = _inputs(w)
    old["v18"] = pd.Series(np.full(len(w["ids_n"]), 700.0), index=w["ids_n"])
    old_vals, _ = V7.replay_values(w["rows"], old, w["tables"], _predict, _gate(), w["rrows"], old["v18"])
    v33 = V7.splice(w["v33"], old_vals, outside_ids=w["ids_x"])
    wrong = dict(old, orl_int=old["orl_int"] + 2)                      # a stale ORL input: R does not reproduce
    paths = {k: tmp_path / f"{k}.x" for k in ("cand", "manifest", "meta", "final")}
    with pytest.raises(AssertionError, match="replay"):
        V7.build(v33, None, w["rows"], w["rrows"], w["tables"], _gate(), wrong, _inputs(w, 3.0), predict_old=_predict,
                 predict_new=_predict, outside_ids=w["ids_x"], paths=paths, allow_roots=(tmp_path,), check_counts=False)
    assert not any(p.exists() for p in paths.values())


def test_parse_args_modes():
    with pytest.raises(SystemExit):
        V7.parse_args([])
    assert V7.parse_args(["--rss-limit-gib", "6", "--smoke"]).smoke is True


def test_lane_ids_split_n_by_table_and_keep_all_r():
    w = _world()
    lanes = V7.lane_ids(w["rows"], w["rrows"])
    assert set(lanes) == {"R", "N_v4", "N_afj", "N_aun"}
    assert sum(len(v) for k, v in lanes.items() if k != "R") == len(w["ids_n"]) and len(lanes["R"]) == len(w["ids_r"])
