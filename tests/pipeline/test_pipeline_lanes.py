"""prc.pipeline.lanes -- routing puts every scored row in exactly one lane; the pins hold; and every lane is
BIT-EQUAL to the shipped path it wraps:

  unmatched      == unm_congestion_ship.build's values          (E3C, v9's non-LIRF rows)
  airport_rules  == rome_bandfloor.build(rome_ship.build(...))  (R2 -> DS -> E1, v7 -> v10's LIRF rows)
  matched        streaming design == lgbm_submit's full design; saved-booster prediction == in-memory

Lane-equivalence tests fit small models (sklearn / LightGBM on a few thousand synthetic rows) and take
~0.5-2 s each: they are integration tests of composition, kept in this file beside the unit tests.

Mutation rehearsal (2026-09-11, scratchpad p2a/mutate.py): the break named in each docstring turned the test
RED against prc/pipeline/lanes.py, GREEN on restore.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pipeline_world", pathlib.Path(__file__).with_name("pipeline_world.py"))
pw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pw)

from prc.pipeline import config as C  # noqa: E402
from prc.pipeline import errors as E  # noqa: E402
from prc.pipeline import lanes, legacy  # noqa: E402
from prc.pipeline import models as M  # noqa: E402

SEEDS = (0, 1)


def _cfg_with(extra_airport=True):
    raw = pw.fresh(pw.real_raw())
    if extra_airport:
        raw["airports"]["EKCH"] = {"tz": "Europe/Copenhagen", "history": False,
                                   "lanes": {"matched": "matched", "unmatched": "unmatched"}}
    return C.parse_config(raw, root=ROOT)


def _scored_frame():
    ap = ["EDDF", "EDDF", "LIRF", "LIRF", "EKCH", "EKCH", "LTFM"]
    unm = [False, True, False, True, False, True, True]
    aobt = [pd.Timestamp("2026-01-01", tz="UTC") if not u else pd.NaT for u in unm]
    return pd.DataFrame({"MVT_ID_mvt": np.arange(len(ap), dtype="float64") + 10, "ADEP_mvt": ap, "unmatched": unm,
                         "AOBT_3_flt": pd.to_datetime(aobt, utc=True)})


# ---- routing -----------------------------------------------------------------------------------------

def test_every_scored_row_lands_in_exactly_one_lane_by_airport_and_class():
    """Rehearsed: route() taking the row class from `~unmatched` (classes swapped) -> RED."""
    r = lanes.route(_scored_frame(), _cfg_with())
    assert list(r.lane) == ["matched", "unmatched", "matched", "lirf_rules", "matched", "unmatched", "unmatched"]
    assert r.counts() == {"lirf_rules": 1, "matched": 3, "unmatched": 3}
    assert sum(r.counts().values()) == len(r.ids)


def test_routing_refuses_a_row_whose_lane_is_not_a_configured_lane():
    """The partition guard itself (a row in zero lanes). parse_config refuses an unknown lane id, so the guard is reached
    only by a config built around it -- the case it exists for. (Replaces a tautological mask-sum loop, review 2026-09-11.)
    Rehearsed 2026-09-11 (c4): route()'s per-row partition check deleted -> RED."""
    cfg = _cfg_with()
    ap = dataclasses.replace(cfg.airports["EDDF"], lanes={**cfg.airports["EDDF"].lanes, "matched": "ghost"})
    bad = dataclasses.replace(cfg, airports={**cfg.airports, "EDDF": ap})
    with pytest.raises(E.RoutingError, match=r"1 scored rows land in \[0, 1\] lanes"):
        lanes.route(_scored_frame(), bad)


def test_rows_of_an_airport_without_history_ride_the_pooled_lanes_flagged():
    """Rehearsed: route() writing fallback=False for every row -> RED."""
    r = lanes.route(_scored_frame(), _cfg_with())
    assert list(r.fallback) == [False, False, False, False, True, True, False]
    assert r.by_airport()["EKCH"] == {"matched": 1, "unmatched": 1}


def test_routing_refuses_an_airport_outside_the_registry_and_an_inconsistent_class():
    """Rehearsed: route()'s check_airports call deleted -> RED (KeyError instead of UnknownAirportError)."""
    with pytest.raises(E.UnknownAirportError, match="EKCH"):
        lanes.route(_scored_frame(), _cfg_with(extra_airport=False))
    bad = _scored_frame()
    bad.loc[0, "unmatched"] = True                     # AOBT_3 present but flagged unmatched
    with pytest.raises(E.RoutingError, match="disagrees"):
        lanes.route(bad, _cfg_with())


def test_the_pins_hold_against_the_scripts_and_a_drifted_constant_is_refused(monkeypatch):
    """Rehearsed: check_pins' comparison deleted -> RED."""
    cfg = _cfg_with(extra_airport=False)
    got = lanes.check_pins(cfg)
    assert (got["schedule_floor_lo_s"], got["schedule_floor_hi_s"], got["rules_airport"]) == (24_000.0, 86_400.0, "LIRF")
    monkeypatch.setattr(legacy.stratum().rb, "LO", 20_000.0)
    with pytest.raises(E.PinnedValueError, match="schedule_floor_lo_s"):
        lanes.check_pins(cfg)


# ---- the stratum lanes against the ship scripts --------------------------------------------------------

@pytest.fixture(scope="module")
def sw():
    """> 10,000 non-fill training rows so the body's seeds give different fits (sklearn early_stopping='auto'),
    as on the real stratum -- otherwise a seed-handling bug in a lane is invisible (models-test lesson, 2026-09-11)."""
    w = pw.stratum_world(seed=5, n_per_month=1100, n_matched_per_month=500)
    w.parts = lanes.s1_parts(w.inputs, w.scored, SEEDS)
    fb = w.parts["nf_fit_by_seed"]
    assert not np.array_equal(fb[0], fb[1]), "the seeds must matter in this world"
    w.base = pw.s1_base(w.parts, w.scored)
    w.cfg = _cfg_with(extra_airport=False)
    w.lirf = (w.scored.ADEP_mvt == "LIRF").to_numpy()
    return w


def test_the_unmatched_lane_equals_unm_congestion_ship_build_bit_for_bit(sw):
    """E3C through the pipeline == the ship script's own build() on every non-LIRF row.
    Rehearsed: run_unmatched_lane using nf_cells for every row (no body) -> RED."""
    st = sw.st
    out, info = st.ucs.build(sw.train_unm, sw.scored, sw.parts, sw.base, sw.serve_dep)
    r = lanes.run_unmatched_lane(sw.cfg.lanes["unmatched"], ~sw.lirf, sw.scored, sw.parts, sw.inputs, sw.serve_dep, SEEDS)
    want = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(r.ids).to_numpy(dtype="float64")
    assert np.array_equal(r.values, want) and len(r.ids) == int((~sw.lirf).sum())
    assert (r.values != sw.base.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(r.ids).to_numpy()).any()   # not S1
    assert int((r.provenance.stage == "S1C:body").sum()) == info["n_routed_to_body"]


def test_the_unmatched_lane_with_the_plain_body_is_s1(sw):
    """Rehearsed: _body_nf ignoring model_name (always the congestion body) -> RED."""
    bs = sw.st.bs
    lane = dataclasses.replace(sw.cfg.lanes["unmatched"], model="nonfill_body")
    r = lanes.run_unmatched_lane(lane, ~sw.lirf, sw.scored, sw.parts, sw.inputs, sw.serve_dep, SEEDS)
    want = np.rint(bs.mixture(sw.parts["p"], sw.parts["sp"], sw.parts["nf"]))[~sw.lirf]
    assert np.array_equal(r.values, want)


def test_the_rules_lane_equals_rome_ship_then_rome_bandfloor_bit_for_bit(sw):
    """R2 -> DS -> E1 through the pipeline == rome_bandfloor.build(rome_ship.build(...)) on every LIRF row, in a
    world where the date-slip segment and the floor band are both populated (so each stage does work).
    Rehearsed: run_rules_lane skipping ds.apply -> RED; skipping the floor -> RED."""
    import rome_ship as rs
    st = sw.st
    out7, _ = rs.build(sw.train_unm, sw.train_lirf, sw.scored, sw.parts, sw.base, seeds=SEEDS)
    out10, finfo = st.rb.build(out7, sw.scored[["MVT_ID_mvt", "ADEP_mvt", "unmatched", "sp"]])
    r = lanes.run_rules_lane(sw.cfg.lanes["lirf_rules"], sw.cfg.airports["LIRF"], sw.lirf, sw.scored, sw.parts,
                             sw.inputs, SEEDS)
    want = out10.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(r.ids).to_numpy(dtype="float64")
    assert np.array_equal(r.values, want) and len(r.ids) == int(sw.lirf.sum())
    assert r.info["n_in_segment_g"] > 0 and r.info["n_floor_changed"] == finfo["n_values_differ"] > 0
    stages = set(r.provenance.stage)
    assert any("+DS" in s for s in stages) and any(s.endswith("+E1") for s in stages) and "R2" in stages


def test_the_rules_lane_honours_an_ordered_subset_of_the_rules(sw):
    """Without the floor it is rome_ship's R2; without the classifier it keeps S1's p.
    Rehearsed: run_rules_lane applying the floor whatever the config says -> RED."""
    st = sw.st
    import rome_ship as rs
    ap = sw.cfg.airports["LIRF"]
    no_floor = dataclasses.replace(ap, rules=C.AirportRules(order=("fill_classifier", "dateslip")))
    out7, _ = rs.build(sw.train_unm, sw.train_lirf, sw.scored, sw.parts, sw.base, seeds=SEEDS)
    r = lanes.run_rules_lane(sw.cfg.lanes["lirf_rules"], no_floor, sw.lirf, sw.scored, sw.parts, sw.inputs, SEEDS)
    assert np.array_equal(r.values, out7.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(r.ids).to_numpy(dtype="float64"))
    only_ds = dataclasses.replace(ap, rules=C.AirportRules(order=("dateslip",)))
    r2 = lanes.run_rules_lane(sw.cfg.lanes["lirf_rules"], only_ds, sw.lirf, sw.scored, sw.parts, sw.inputs, SEEDS)
    te = sw.scored[sw.lirf]
    q, m = st.ds.band_shares(sw.train_unm[sw.train_unm.ADEP_mvt == "LIRF"])
    want = np.rint(st.ds.apply(te, st.rf.mixture(sw.parts["p"][sw.lirf], te.sp.to_numpy(), sw.parts["nf"][sw.lirf]), q, m))
    assert np.array_equal(r2.values, want)


def _rld_frame():
    """Five LIRF unmatched rows: same Rome local day in band (moves), next local day (does not), below band, a matched row,
    and a same-day in-band row at another airport."""
    sched = pd.to_datetime(["2026-01-10 06:00", "2026-01-10 20:00", "2026-01-10 06:00", "2026-01-10 06:00",
                            "2026-01-10 06:00"], utc=True)
    mvt = sched + pd.to_timedelta([50_000, 50_000, 20_000, 50_000, 50_000], unit="s")
    return pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0], "ADEP_mvt": ["LIRF", "LIRF", "LIRF", "LIRF", "EDDF"],
                         "unmatched": [True, True, True, False, True], "SCHED_TIME_UTC_mvt": sched,
                         "MVT_TIME_UTC_mvt": mvt, "sp": (mvt - sched).total_seconds().to_numpy()})


def test_the_local_day_rule_sets_rint_sp_only_on_same_local_day_in_band_rome_unmatched_rows():
    """Row 1 (06:00 + 13.9 h, same Rome day) takes rint(sp) = 50,000; row 2 (20:00 + 13.9 h, next Rome day), row 3 (below
    band), row 4 (matched) and row 5 (EDDF) keep their value. Rehearsed 2026-09-11 (c70): _local_day passing the lane's
    value where RLD returned rint(sp) -> RED."""
    te = _rld_frame()
    v = np.array([1000.0, 1200.0, 900.0, 800.0, 700.0])
    new, moved = lanes._local_day(te, v)
    assert new.tolist() == [50_000.0, 1200.0, 900.0, 800.0, 700.0] and moved.tolist() == [True, False, False, False, False]


def test_the_local_day_rule_refuses_an_sp_that_is_not_mvt_minus_sched():
    """The rule recomputes sp from the two clocks; if the lane's sp column disagrees, something upstream changed its
    meaning. Rehearsed (c70): the equality check deleted -> RED."""
    te = _rld_frame()
    te.loc[0, "sp"] = 49_000.0
    with pytest.raises(E.LaneError, match="sp"):
        lanes._local_day(te, np.ones(len(te)))


def test_the_pins_cover_the_local_day_rule(monkeypatch):
    """check_pins re-reads rome_local_day's band and airport. Rehearsed (c70): its RLD entries deleted -> RED."""
    cfg = _cfg_with(extra_airport=False)
    assert lanes.check_pins(cfg)["local_day_lo_s"] == 24_000.0
    monkeypatch.setattr(legacy.rome_local_day(), "SP_LO", 20_000.0)
    with pytest.raises(E.PinnedValueError, match="local_day_lo_s"):
        lanes.check_pins(cfg)


def test_the_rules_lane_applies_the_local_day_rule_last_and_marks_it(sw):
    """With local_day_schedule enabled, the lane equals R2 -> DS -> E1 and then RLD on exactly the rows RLD selects;
    provenance marks them +RLD. Rehearsed (c70): run_rules_lane skipping _local_day -> RED."""
    ap = sw.cfg.airports["LIRF"]
    with_rld = dataclasses.replace(ap, rules=C.AirportRules(order=(*ap.rules.order, "local_day_schedule"),
                                                             schedule_floor=ap.rules.schedule_floor,
                                                             local_day=C.LocalDay(lo_s=24_000.0, hi_s=86_400.0)))
    r0 = lanes.run_rules_lane(sw.cfg.lanes["lirf_rules"], ap, sw.lirf, sw.scored, sw.parts, sw.inputs, SEEDS)
    r1 = lanes.run_rules_lane(sw.cfg.lanes["lirf_rules"], with_rld, sw.lirf, sw.scored, sw.parts, sw.inputs, SEEDS)
    want, moved = lanes._local_day(sw.scored[sw.lirf].reset_index(drop=True), r0.values)
    assert moved.any() and np.array_equal(r1.values, want) and r1.info["n_local_day_changed"] == int(moved.sum())
    assert np.array_equal(r1.provenance.stage.str.endswith("+RLD").to_numpy(), moved)


def test_the_body_plugin_must_reproduce_fit_unmatcheds_fused_body_or_the_lane_refuses(sw):
    """Rehearsed: _body_nf's equality check deleted -> RED."""
    parts = dict(sw.parts, nf_fit=np.asarray(sw.parts["nf_fit"]) + 1e-9)
    with pytest.raises(E.LaneError, match="nf_fit"):
        lanes.run_rules_lane(sw.cfg.lanes["lirf_rules"], sw.cfg.airports["LIRF"], sw.lirf, sw.scored, parts,
                             sw.inputs, SEEDS)


# ---- the matched lane ------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mw(tmp_path_factory):
    w = pw.matched_world(tmp_path_factory.mktemp("lanes_matched"))
    w.model = M.LightGBMModel(params=w.params, nest=w.nest, patience=w.patience)
    w.state = w.model.fit(w.X, w.labels, w.masks, SEEDS)
    w.model.save(w.state, w.lane.booster_dir)
    ls = legacy.matched().ls
    w.want = ls.finalise_taxi_time(ls.raw_taxi_time(w.model.predict(w.state, w.X[w.is_rank]),
                                                    w.labels.proxy[w.is_rank], "delta"))
    return w


def test_the_streaming_design_equals_lgbm_submits_full_design_bit_for_bit(mw):
    """The light path (encodings one key at a time) must build exactly the ranking slice of load_frames ->
    infold_encodings -> design_matrix. Rehearsed: scored_design fitting the encodings on every row (tr_mask
    all True, the leak) -> RED; dropping the ranking rows' block join -> RED."""
    d = lanes.scored_design(mw.lane, mw.ns.stand)
    assert np.array_equal(d.ids, mw.rank_ids)
    assert np.array_equal(d.X, mw.X[mw.is_rank], equal_nan=True) and d.X.dtype == np.float32
    assert np.array_equal(d.proxy, mw.labels.proxy[mw.is_rank]) and d.n_train == int(mw.masks.train.sum())
    assert d.feats == lanes.matched_features(mw.lane) and len(d.feats) == mw.X.shape[1]


def test_the_stand_training_months_are_read_in_lgbm_submits_order(mw):
    """Rehearsed: stand_training_paths sorted in reverse -> RED (a glob's own order is not guaranteed)."""
    ls = legacy.matched().ls
    _, paths = ls.training_frames(mw.ns.stand)
    assert lanes.stand_training_paths(mw.ns.stand) == list(paths)


def test_the_matched_lane_from_saved_boosters_equals_the_in_memory_model(mw):
    """Rehearsed: run_matched_lane recovering the taxi time as proxy + delta_hat -> RED."""
    r = lanes.run_matched_lane(mw.lane, mw.cfg, mw.rank_ids, model=M.LightGBMModel(params=mw.params), log=lambda m: None)
    assert np.array_equal(r.ids, mw.rank_ids) and np.array_equal(r.values, mw.want.astype("float64"))
    assert r.info["boosters"] == ["lgbm_seed0.txt", "lgbm_seed1.txt"] and r.info["n_features"] == mw.X.shape[1]
    assert set(r.provenance.stage) == {"lightgbm:delta:mean2"}


def test_the_matched_lane_refit_path_equals_the_same_fit(mw):
    """source: refit runs lgbm_submit's full design and the plug-in's fit (1 thread: deterministic).
    Rehearsed: full_design passing es_months=(1, 2) -> RED (a different early stop)."""
    lane = dataclasses.replace(mw.lane, source="refit", booster_dir=None, boosters=())
    model = M.LightGBMModel(params=mw.params, nest=mw.nest, patience=mw.patience)
    r = lanes.run_matched_lane(lane, mw.cfg, mw.rank_ids, model=model, log=lambda m: None)
    assert np.array_equal(r.values, mw.want.astype("float64"))


class _BroadcastingLGBM(M.LightGBMModel):
    """A plug-in whose predict returns ONE value: numpy would broadcast it over every row downstream, silently."""
    def predict(self, state, X):
        return np.array([123.0])


class _NoneLGBM(M.LightGBMModel):
    def predict(self, state, X):
        return None


@pytest.mark.parametrize("cls,msg", [(_BroadcastingLGBM, r"shape \(1,\)"), (_NoneLGBM, "NoneType")])
def test_the_matched_lane_refuses_a_plugin_prediction_that_breaks_the_contract(mw, cls, msg):
    """Rehearsed 2026-09-11 (c4): run_matched_lane without check_prediction -> RED (the broadcast case returns a
    constant for every row instead of raising)."""
    with pytest.raises(E.LaneError, match=msg):
        lanes.run_matched_lane(mw.lane, mw.cfg, mw.rank_ids, model=cls(params=mw.params), log=lambda m: None)


def test_the_unmatched_lane_refuses_a_body_plugin_prediction_that_breaks_the_contract(sw, monkeypatch):
    """Rehearsed 2026-09-11 (c4): _body_nf without check_prediction -> RED."""
    class _ShortBody(M.NonFillBodyModel):
        def predict(self, state, X):
            return super().predict(state, X)[:-1]
    monkeypatch.setitem(M.REGISTRY, "nonfill_body", lambda **kw: _ShortBody(congestion=False))
    lane = dataclasses.replace(sw.cfg.lanes["unmatched"], model="nonfill_body")
    with pytest.raises(E.LaneError, match="'nonfill_body'.*shape"):
        lanes.run_unmatched_lane(lane, ~sw.lirf, sw.scored, sw.parts, sw.inputs, sw.serve_dep, SEEDS)


def test_the_matched_lane_refuses_a_design_that_does_not_cover_its_routed_rows(mw):
    """Rehearsed: run_matched_lane's id-set check deleted -> RED."""
    with pytest.raises(E.LaneError, match="routed rows missing"):
        lanes.run_matched_lane(mw.lane, mw.cfg, np.append(mw.rank_ids, 1e12), model=M.LightGBMModel(params=mw.params),
                               log=lambda m: None)


# ---- dry-run input checks ---------------------------------------------------------------------------------

@pytest.fixture
def dw(tmp_path):
    w = pw.dry_run_world(tmp_path, seed=11)
    cfg = C.load_config(w.config)
    from prc.pipeline import ingest
    sd = ingest.ingest_scored(cfg)
    return w, cfg, lanes.route(sd.scored, cfg)


def test_lane_inputs_pass_on_a_consistent_world(dw):
    """Rehearsed: _check_matched_inputs skipping the booster provenance -> RED (the report lacks n_ref)."""
    _, cfg, routing = dw
    rep = lanes.check_lane_inputs(cfg, routing)
    assert rep["matched"]["n_ref"] == 48 and rep["matched"]["n_features"] == 83 and rep["training_files"]


def test_lane_inputs_refuse_a_stale_ranking_cache(dw):
    """A ranking cache built for another scored file must stop the run before any prediction.
    Rehearsed: the ranking-cache id-set check deleted -> RED."""
    w, cfg, routing = dw
    p = cfg.paths.stand_cache / "ranking.parquet"
    f = pd.read_parquet(p)
    f.iloc[1:].to_parquet(p, index=False)
    with pytest.raises(E.RoutingError, match="stale cache"):
        lanes.check_lane_inputs(cfg, routing)


def test_lane_inputs_refuse_a_block_twin_with_another_row_count(dw):
    """Rehearsed: the twin row-count check deleted -> RED."""
    w, cfg, routing = dw
    twin = pathlib.Path(dict(cfg.lanes["matched"].feature_blocks)["queue"]) / "training_2025-01-01_2025-02-01.parquet"
    f = pd.read_parquet(twin)
    f.iloc[:-1].to_parquet(twin, index=False)
    with pytest.raises(E.SchemaError, match="row count"):
        lanes.check_lane_inputs(cfg, routing)


def test_lane_inputs_refuse_a_booster_whose_provenance_does_not_match(dw):
    """Rehearsed: LightGBMModel.load's n_features comparison deleted -> RED."""
    import json
    w, cfg, routing = dw
    j = cfg.lanes["matched"].booster_dir / "lgbm_fixture_seed2.fit.json"
    rec = json.loads(j.read_text())
    rec["n_features"] = 80
    j.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="n_features 80"):
        lanes.check_lane_inputs(cfg, routing)


@pytest.mark.parametrize("lane_id,model", [("matched", "catboost"), ("unmatched", "nn")])
def test_a_lane_configured_with_a_stub_model_is_refused_before_anything_runs(dw, lane_id, model):
    """Rehearsed: check_lane_inputs' implemented check deleted -> RED for the unmatched lane (nothing else asks an
    unmatched lane's model anything in a dry run). The matched case is NOT individually rehearsed: two independent
    refusals guard it (this check and the stub's own load()), so deleting either alone leaves it GREEN by design."""
    w, cfg, routing = dw
    lanes_ = dict(cfg.lanes)
    lanes_[lane_id] = dataclasses.replace(cfg.lanes[lane_id], model=model)
    cfg2 = dataclasses.replace(cfg, lanes=lanes_)
    with pytest.raises(NotImplementedError, match=model):
        lanes.check_lane_inputs(cfg2, routing)


def test_the_matched_lane_with_the_adsb_stage_moves_only_covered_rows_at_gated_airports(mw, tmp_path):
    """The stage on the fixture world with a stub predictor (−30 s residual where eligible): eligible rows' taxi rises by 30 s,
    every other row equals the lane without the stage, provenance marks the moved rows. Rehearsed 2026-09-11 (c4):
    run_matched_lane skipping _adsb_stage -> RED; the gate ignored in apply_stage -> RED (the LTFM rows move)."""
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = np.asarray(mw.rank_ids, dtype="float64")
    rank = pd.read_parquet(mw.cfg.paths.stand_cache / "ranking.parquet", columns=["MVT_ID_mvt", "ap"]).set_index("MVT_ID_mvt").reindex(ids)
    cov = np.where(np.arange(len(ids)) % 2 == 0, 3, 1)
    t = pa.Table.from_pandas(pd.DataFrame({"MVT_ID_mvt": ids, "coverage": cov, "first_ground_rel_aobt3": 0.0}), preserve_index=False)
    table = tmp_path / "features.parquet"
    pq.write_table(t.replace_schema_metadata({**(t.schema.metadata or {}), b"git_sha": b"x", b"extractor": b"e",
                                              b"features_version": b"v"}), table)
    gated = sorted(set(rank.ap))[:3]
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"gate": {a: (a in gated) for a in set(rank.ap)}}))
    lane = dataclasses.replace(mw.lane, adsb_stage=C.AdsbStage(table=table, models_dir=tmp_path))
    stub = lambda f: np.where((f.coverage >= 2) & f.ap.isin(gated), -30.0, 0.0).astype("float64")
    r = lanes.run_matched_lane(lane, mw.cfg, mw.rank_ids, model=M.LightGBMModel(params=mw.params), log=lambda m: None,
                               stage_predictor=stub)
    eligible = (cov >= 2) & rank.ap.isin(gated).to_numpy()
    want = mw.want.astype("float64")
    assert eligible.any() and (~eligible).any()
    assert np.array_equal(r.values[~eligible], want[~eligible])
    assert np.array_equal(r.values[eligible], np.rint(np.maximum(want[eligible] + 30.0, 1.0)))
    moved = r.provenance.stage.str.endswith("+adsb").to_numpy()
    assert np.array_equal(moved, eligible) and r.info["adsb_stage"]["n_changed"] == int(eligible.sum())


def test_the_adsb_stage_refuses_a_base_that_differs_from_what_the_lane_ships(mw, tmp_path):
    """The stage adds a residual to the lane's own values; if its base (rint(max(proxy − F, 1))) ever differed from what the
    lane would ship, it would silently add to something else. Rehearsed 2026-09-11 (c4): the base-equality check deleted -> RED."""
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = np.asarray(mw.rank_ids, dtype="float64")
    t = pa.Table.from_pandas(pd.DataFrame({"MVT_ID_mvt": ids, "coverage": 3}), preserve_index=False)
    table = tmp_path / "features.parquet"
    pq.write_table(t.replace_schema_metadata({**(t.schema.metadata or {}), b"git_sha": b"x", b"extractor": b"e",
                                              b"features_version": b"v"}), table)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"gate": {"EHAM": True}}))
    lane = dataclasses.replace(mw.lane, adsb_stage=C.AdsbStage(table=table, models_dir=tmp_path))
    proxy = mw.labels.proxy[mw.is_rank]
    pred = np.zeros(len(ids))
    wrong = np.rint(np.maximum(proxy - pred, 1.0)) + 1.0
    with pytest.raises(E.LaneError, match="base value differs"):
        lanes._adsb_stage(lane, mw.cfg, ids, proxy, pred, wrong, lambda f: np.zeros(len(f)), lambda m: None)


def test_the_adsb_stage_calls_the_real_predictor_with_the_models_dir_and_the_full_bag(mw, tmp_path, monkeypatch):
    """With no injected predictor the stage calls 6e's predict_residual(frame, model_dir, oof=False): the lane's models_dir,
    and the full 15-model bag (oof=True would pick the fold-k model per row by a `fold` column 2026 rows do not have).
    Rehearsed 2026-09-11 (c4): oof=True -> RED; models_dir replaced by the table path -> RED."""
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    ids = np.asarray(mw.rank_ids, dtype="float64")
    t = pa.Table.from_pandas(pd.DataFrame({"MVT_ID_mvt": ids, "coverage": 3}), preserve_index=False)
    table = tmp_path / "features.parquet"
    pq.write_table(t.replace_schema_metadata({**(t.schema.metadata or {}), b"git_sha": b"x", b"extractor": b"e",
                                              b"features_version": b"v"}), table)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"gate": {"EHAM": {"allowed": False, "rule": "r", "c25": 0.0, "c26": 0.0}}}))
    calls = []

    def real(frame, model_dir, oof=None):
        calls.append((len(frame), model_dir, oof))
        return np.zeros(len(frame))
    monkeypatch.setattr(lanes.legacy, "adsb_predictor", lambda: real)
    lane = dataclasses.replace(mw.lane, adsb_stage=C.AdsbStage(table=table, models_dir=tmp_path))
    proxy = mw.labels.proxy[mw.is_rank]
    pred = np.zeros(len(ids))
    vals = np.rint(np.maximum(proxy - pred, 1.0))
    new, info, moved = lanes._adsb_stage(lane, mw.cfg, ids, proxy, pred, vals, None, lambda m: None)
    assert calls == [(len(ids), tmp_path, False)]
    assert np.array_equal(new, vals) and not moved.any() and info["gate"] == {"EHAM": False}
