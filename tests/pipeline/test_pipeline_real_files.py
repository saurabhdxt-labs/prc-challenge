"""prc.pipeline on the REAL files (integration tier; skipped when the challenge data is absent).

  * the dry run on configs/pipeline.yaml: every schema, the routing of the 344,841 scored rows, every lane input
    and the saved v6 boosters' provenance -- ~3 s, ~0.7 GB (measured 2026-09-11);
  * A1 (partial) -- the unmatched and LIRF lanes reproduce submissions/merry-quicksand_v10.parquet on every one
    of their rows. ~9 s, ~1.6 GB peak (the raw training months are read, projected, for the 2025 witness), so it
    runs only with PRC_A1=1. v10 is read by the TEST as a reference, never by the pipeline.

  * A1 (full) -- the ONE-COMMAND pipeline run (cli.run: every lane, the matched lane from the saved v6 boosters,
    assemble, validate, write, manifest) equals v10 on all 344,841 rows. Heavy: one loaded v6 booster is ~2.47 GB
    resident (measured 2026-09-11), ~3-3.5 GB process peak, tens of minutes -- so it runs only with PRC_A1_FULL=1, as
    the ONLY >= 2 GB job on the machine, on AC power, after the quality gate. (The earlier skip reason here, "over this
    machine's 2 GB single-job ceiling", misread the rule: it forbids a SECOND >= 2 GB job, not one big job.)
    PRC_A1_OUT names a persistent output directory (it must not exist); the compare report is written beside the
    manifest there.
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
V10 = ROOT / "submissions" / "merry-quicksand_v10.parquet"
pytestmark = pytest.mark.skipif(not (RAW / "ranking.parquet").exists() or not (ROOT / "data" / "cache_rome").exists(),
                                reason="challenge data not present")

from prc.pipeline import cli, ingest, lanes  # noqa: E402
from prc.pipeline import config as C  # noqa: E402

N_MATCHED, N_UNMATCHED_NON_LIRF, N_LIRF_UNMATCHED = 339_551, 4_907, 383


@pytest.fixture(scope="module")
def cfg():
    return C.load_config(ROOT / "configs" / "pipeline.yaml")


def test_the_real_dry_run_routes_every_scored_row_and_verifies_every_lane_input(cfg, tmp_path):
    """Rehearsed: the routing table mapping LIRF unmatched rows to `unmatched` -> RED (lirf_rules count 0)."""
    rep = cli.dry_run(cfg, tmp_path, log=lambda m: None)
    assert rep["routing"]["counts"] == {"matched": N_MATCHED, "unmatched": N_UNMATCHED_NON_LIRF, "lirf_rules": N_LIRF_UNMATCHED}
    assert rep["scored"] == {"file": str(cfg.paths.scored_file), "n_departures": 344_841, "n_scored": 344_841,
                             "n_template": 344_841}
    assert rep["fallback_airports"] == [] and rep["fallback_rows"] == {}
    m = rep["lane_inputs"]["matched"]
    assert (m["best_iter"], m["n_ref"], m["n_features"], m["n_training_rows"]) == (25_316, 30_398, 83, 2_062_440)
    assert all(len(v["sha256"]) == 64 for v in rep["inputs"].values())
    assert not any("submissions" in v["path"] for v in rep["inputs"].values())


@pytest.mark.skipif(os.environ.get("PRC_A1") != "1" or not V10.exists(), reason="A1 stratum run: set PRC_A1=1 (~9 s, ~1.6 GB)")
def test_a1_the_unmatched_and_lirf_lanes_reproduce_v10_on_every_row(cfg):
    """Rehearsed: run_rules_lane without the schedule floor -> RED (13 LIRF rows differ from v10)."""
    sd = ingest.ingest_scored(cfg)
    routing = lanes.route(sd.scored, cfg)
    unm = sd.scored.unmatched.to_numpy(dtype=bool)
    scored_unm = sd.scored[unm]
    inputs = lanes.stratum_inputs(cfg)
    parts = lanes.s1_parts(inputs, scored_unm, cfg.seeds)
    lane_unm = routing.lane[unm]
    r1 = lanes.run_unmatched_lane(cfg.lanes["unmatched"], lane_unm == "unmatched", scored_unm, parts, inputs, sd.dep, cfg.seeds)
    r2 = lanes.run_rules_lane(cfg.lanes["lirf_rules"], cfg.airports["LIRF"], lane_unm == "lirf_rules", scored_unm, parts,
                              inputs, cfg.seeds)
    v10 = pd.read_parquet(V10).set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    for r, n in ((r1, N_UNMATCHED_NON_LIRF), (r2, N_LIRF_UNMATCHED)):
        assert len(r.ids) == n
        assert np.array_equal(r.values.astype("int64"), v10.reindex(r.ids).to_numpy(dtype="int64")), r.lane


@pytest.mark.skipif(os.environ.get("PRC_A1_FULL") != "1" or not V10.exists(),
                    reason="A1 full run: set PRC_A1_FULL=1 (heavy: ~3-3.5 GB, the only big job; AC power; quality gate)")
def test_a1_the_one_command_pipeline_reproduces_v10_on_every_row(cfg, tmp_path):
    """A1 + A2's command: cli.run builds all 344,841 rows from data (no submission file is an input -- asserted by the run
    itself) and every row equals v10; the manifest hashes the code that ran, the three v6 boosters and the config.
    v10 is read HERE, by the test, as the reference only (validate.compare_to_reference)."""
    import json

    from prc.pipeline import validate, write
    out = pathlib.Path(os.environ["PRC_A1_OUT"]) if os.environ.get("PRC_A1_OUT") else tmp_path / "a1_full"
    if out.exists():
        raise FileExistsError(f"{out} exists; A1 never overwrites an earlier run's evidence")
    res = cli.run(cfg, out, log=lambda m: print(m, flush=True))
    cmp = validate.compare_to_reference(res["assembled"], V10)
    m = res["manifest"]
    (out / "a1_compare.json").write_text(json.dumps({"reference": str(V10.relative_to(ROOT)),
                                                     "reference_sha256": write.sha256_file(V10),
                                                     "manifest_digest": m["digest"], "compare": cmp}, indent=1))
    assert cmp["ALL"] == {"n": 344_841, "n_equal": 344_841, "n_differ": 0}, cmp
    assert {k: cmp[k]["n"] for k in ("matched", "unmatched", "lirf_rules")} == \
        {"matched": N_MATCHED, "unmatched": N_UNMATCHED_NON_LIRF, "lirf_rules": N_LIRF_UNMATCHED}
    assert {"prc/pipeline/lanes.py", "scripts/lgbm_submit.py", "scripts/stand_ab.py", "scripts/build_submission.py",
            "scripts/unm_congestion_ship.py", "scripts/rome_bandfloor.py"} <= set(m["code"]["files"])
    assert {f"boosters/lgbm_v6_queue_order_seed{s}.txt" for s in (0, 1, 2)} <= set(m["inputs"])
    assert not any("submissions" in v["path"] for v in m["inputs"].values())
    assert m["lanes"]["matched"]["n_ref"] == 30_398 and m["lanes"]["matched"]["source"] == "saved_boosters"


ADSB = ROOT / "data" / "adsb" / "v2"


def _guard_frame(cfg):
    """The 2025 guard's stage frame, built by the PIPELINE's stage code from the pipeline's own caches -- not 6e's load_frame:
    ap / hr / proxy / doy read positionally from the stand caches (fold A's `row` numbers the concat of all twelve 2025
    training months, the lgbm_fold contract; the 2025 caches carry no MVT_ID), features through adsb_stage.read_table keyed by the
    table's own MVT_ID, F = arm F's `treatment` from the fold record, fold = adsb_stack.day_folds (the models' own folds)."""
    import datetime as dt

    from prc.pipeline import adsb_stage as AS
    from prc.pipeline import legacy
    rec = pd.read_parquet(cfg.paths.stand_cache / "fold_preds_queue_order.parquet",
                          columns=["row", "month", "ap", "y", "proxy", "treatment"])
    months = sorted(cfg.paths.stand_cache.glob("training_2025-*.parquet"))       # lgbm_submit.training_frames' order
    cache = pd.concat([pd.read_parquet(p, columns=["ap", "hr", "proxy", "doy"]) for p in months], ignore_index=True)
    assert (len(months), len(cache)) == (12, 2_062_440), (len(months), len(cache))   # arm F's fold: all twelve months
    at = cache.iloc[rec.row.to_numpy()].reset_index(drop=True)
    assert (at.ap.astype(str).to_numpy() == rec.ap.astype(str).to_numpy()).all(), "stand cache ap differs from the record's"
    assert np.array_equal(at.proxy.to_numpy(float), rec.proxy.to_numpy(float)), "stand cache proxy differs from the record's"
    key = pd.read_parquet(ADSB / "features_2025janjul.parquet", columns=["row", "MVT_ID_mvt"])
    assert np.array_equal(key.row.to_numpy(), rec.row.to_numpy()), "feature table rows are not the record's rows in order"
    ids = key.MVT_ID_mvt.to_numpy(dtype="float64")
    table, meta = AS.read_table(ADSB / "features_2025janjul.parquet", ids)
    frame = AS.stage_frame(ids, at.ap.to_numpy(), at.hr.to_numpy(), at.proxy.to_numpy(), rec.treatment.to_numpy(), table)
    days = [(dt.date(2025, 1, 1) + dt.timedelta(days=int(d) - 1)).isoformat() for d in at.doy.to_numpy()]
    frame["fold"] = legacy.adsb_stack().day_folds(days)
    return frame, rec, meta


@pytest.mark.skipif(os.environ.get("PRC_ADSB_GUARD") != "1" or not (ADSB / "models" / "MANIFEST.json").exists()
                    or not (ADSB / "adn_fold_preds.parquet").exists(),
                    reason="ADS-B 2025 reproduction guard: set PRC_ADSB_GUARD=1 (measured 2.42 GB peak, 30 s: a >= 2 GB job; AC power)")
def test_the_adsb_stage_reproduces_6es_evaluated_arm_b_on_the_2025_fold(cfg, tmp_path):
    """The pipeline stage (stage_frame -> the real predict_residual, oof=True, through a copy of the models dir with every
    airport allowed -> apply_stage) reproduces 6e's evaluated arm B on fold A's 339,015 matched holdout rows: the residual
    equals adn_fold_preds' B - F to 1e-6 s (6e measured 1.6e-12 s reloading its own artefacts; a contract defect -- local
    hour, a shifted airport code, a misnamed feature, a row on the wrong fold -- moves residuals by seconds), the shipped
    seconds equal rint(max(proxy - B, 1)), and the scored-row RMSE equals RESULT ADN's arm B (reports/adsb_nn.json).
    Passed 2026-09-11 11:51 (max 1.592e-12 s, RMSE 210.41984768). Rehearsed 2026-09-11 (c4): adsb_stage.stage_frame feeding
    (hour + 1) % 24 -> RED (max |Δresid| 61.8 s)."""
    import json

    from prc.pipeline import adsb_stage as AS
    from prc.pipeline import legacy
    frame, rec, meta = _guard_frame(cfg)
    src = ADSB / "models"
    models = tmp_path / "models"
    models.mkdir()
    for p in src.iterdir():
        if p.name != "MANIFEST.json":
            (models / p.name).symlink_to(p)
    man = json.loads((src / "MANIFEST.json").read_text())
    man["gate"] = {a: {"allowed": True, "rule": "2025 guard: every airport open", "c25": None, "c26": None} for a in man["airports"]}
    (models / "MANIFEST.json").write_text(json.dumps(man))
    gate, _ = AS.read_gate(models)
    real = legacy.adsb_predictor()
    seen = {}

    def fn(f):
        seen["resid"] = real(f, models, oof=True)
        return seen["resid"]
    value, info = AS.apply_stage(frame, fn, gate)
    preds = pd.read_parquet(ADSB / "adn_fold_preds.parquet", columns=["row", "F", "B"])
    assert np.array_equal(preds.row.to_numpy(), rec.row.to_numpy())
    assert np.array_equal(preds.F.to_numpy(), rec.treatment.to_numpy())
    want = preds.B.to_numpy() - preds.F.to_numpy()
    diff = np.abs(seen["resid"] - want)
    print(f"guard: max |resid - (B - F)| = {diff.max():.3e} s over {len(diff):,} rows; eligible {info['n_eligible']:,}; "
          f"changed {info['n_changed']:,}", flush=True)
    assert diff.max() <= 1e-6
    exact = np.rint(np.maximum(frame.proxy.to_numpy() - preds.B.to_numpy(), 1.0))
    raw = frame.proxy.to_numpy() - preds.B.to_numpy()
    near_half = np.abs(np.abs(raw - np.floor(raw)) - 0.5) < 1e-6
    assert np.all(np.abs(value - exact) <= near_half)          # equal everywhere; <= 1 s only where float noise meets a .5
    covered, folded = frame.coverage.to_numpy() >= 2, np.isfinite(frame.fold.to_numpy())
    assert info["n_eligible"] == int(covered.sum())               # every airport open: eligibility is coverage alone
    assert (seen["resid"][covered & ~folded] == 0.0).all()        # the spent day's rows are in no fold: never moved
    assert int((covered & folded).sum()) == man["n_train_joined_rows"] == 127_736
    scored = np.isfinite(frame.fold.to_numpy())
    y = rec.y.to_numpy(float)[scored]
    arm = np.maximum(frame.proxy.to_numpy()[scored] - (frame.F.to_numpy()[scored] + seen["resid"][scored]), 1.0)
    rmse = float(np.sqrt(np.mean((arm - y) ** 2)))
    res = json.loads((ROOT / "reports" / "adsb_nn.json").read_text())
    assert res["smoke"] is False
    print(f"guard: scored rows {int(scored.sum()):,}, RMSE {rmse:.8f} vs RESULT ADN arm B {res['arms']['B']['rmse_arm']:.8f}", flush=True)
    assert int(scored.sum()) == res["arms"]["B"]["n_rows_scored"]
    assert abs(rmse - res["arms"]["B"]["rmse_arm"]) < 1e-6
    assert meta["features_version"].startswith("adsb_features v2")
