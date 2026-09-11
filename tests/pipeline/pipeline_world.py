"""Synthetic worlds for the prc.pipeline tests. Not a test module: the tests load it by path (importlib),
as the other suites load tests/synthetic_caches.py. Every draw is seeded.

  real_raw()             the parsed configs/pipeline.yaml (the "good" config every bad case mutates)
  stratum_world(seed)    2025-like unmatched stratum (with the witness) + LIRF matched rows, and a "scored"
                         month-12 slice with its serve-file departures -- stratum_fold / unm_congestion's own
                         synthetic generators, so every dtype is derive()'s
  dry_run_world(root)    every file a dry run reads, for ELEVEN airports: the ten plus EKCH (no history),
                         and the config that routes them; nothing in it can be fitted (the boosters are
                         provenance records only)
"""
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from prc.pipeline import ingest, legacy  # noqa: E402

TEN = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
NEW = "EKCH"
T0 = pd.Timestamp("2026-01-10 06:00:00", tz="UTC")


def real_raw() -> dict:
    return yaml.safe_load((ROOT / "configs" / "pipeline.yaml").read_text())


def fresh(raw: dict) -> dict:
    return copy.deepcopy(raw)


# =================================================================================================
# the stratum world (lane equivalence tests)
# =================================================================================================

def stratum_world(seed: int = 5, n_per_month: int = 600, n_matched_per_month: int = 500) -> SimpleNamespace:
    st = legacy.stratum()
    unm, lirf, _ = st.uc.synthetic_world(seed=seed, n_per_month=n_per_month, n_matched_per_month=n_matched_per_month)
    train_unm = unm[unm.month <= 11].reset_index(drop=True)
    train_lirf = lirf[lirf.month <= 11].reset_index(drop=True)
    scored = unm[unm.month == 12].drop(columns=list(st.uc.WITNESS)).reset_index(drop=True)
    companion = st.uc.smoke_matched_companion(scored, seed=seed + 200)
    serve_dep = pd.concat([companion[["ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]],
                           scored[["ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]]], ignore_index=True)
    from prc.pipeline.lanes import StratumInputs
    inputs = StratumInputs(train_unm=train_unm, train_lirf=train_lirf, witness_2025={})
    return SimpleNamespace(train_unm=train_unm, train_lirf=train_lirf, scored=scored, serve_dep=serve_dep,
                           inputs=inputs, st=st)


def s1_base(parts: dict, scored: pd.DataFrame, n_matched: int = 40) -> pd.DataFrame:
    """A base submission whose unmatched rows are S1's own values (so the ship scripts' rebuild guards
    pass) plus some matched placeholder rows -- the frame the ship scripts' build() splices into."""
    bs = legacy.stratum().bs
    s1 = np.rint(bs.mixture(parts["p"], parts["sp"], parts["nf"]))
    ids = np.concatenate([scored.MVT_ID_mvt.to_numpy(), 9e8 + np.arange(n_matched, dtype="float64")])
    return pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": np.concatenate([s1, np.full(n_matched, 900.0)]).astype("int32")})


# =================================================================================================
# the matched world (LightGBM adapter + matched lane tests): lgbm_submit's own synthetic caches
# =================================================================================================

def _synthetic_caches_module():
    spec = importlib.util.spec_from_file_location("synthetic_caches", ROOT / "tests" / "synthetic_caches.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def matched_world(root: pathlib.Path, seed: int = 0, n: int = 400, n_rank: int = 300) -> SimpleNamespace:
    """tests/synthetic_caches.synthetic_submission's stand / queue / order caches (months 1-3; ranking rows in
    months 1 and 7), a matched Lane over them, lgbm_submit's full design (load_frames -> encodings ->
    design_matrix) and the smoke-sized LightGBM parameters at ONE thread (a 1-thread fit is bit-deterministic)."""
    from prc.pipeline import config as C
    from prc.pipeline import lanes
    ls = legacy.matched().ls
    sc = _synthetic_caches_module()
    ns = sc.synthetic_submission(root, months=(1, 2, 3), n=n, seed=seed, n_rank=n_rank, n_unmatched=10, n_arr=10)
    lane = C.Lane(id="matched", kind="matched", model="lightgbm", target="delta",
                  feature_blocks=(("queue", ns.queue), ("order", ns.order)), source="saved_boosters",
                  booster_dir=root / "boosters", boosters=("lgbm_seed0.txt", "lgbm_seed1.txt"))
    X, labels, masks, is_rank, rank_ids = lanes.full_design(lane, ns.stand, (3, 9))
    params = dict(ls.P, learning_rate=0.05, num_threads=1)
    cfg = SimpleNamespace(paths=SimpleNamespace(stand_cache=ns.stand), seeds=(0, 1),
                          folds=SimpleNamespace(early_stopping_months=(3, 9)))
    return SimpleNamespace(ns=ns, lane=lane, X=X, labels=labels, masks=masks, is_rank=is_rank, rank_ids=rank_ids,
                           params=params, cfg=cfg, nest=60, patience=50, root=root)


# =================================================================================================
# the full world: every lane really fits and predicts, eleven airports, end to end through cli.run
# =================================================================================================

def full_world(root: pathlib.Path, seed: int = 0, n_rank: int = 240) -> SimpleNamespace:
    """Every input of a real `run`, small: the 2025 stratum (cache_rome) and raw training clock rows for the
    witness; a scored file whose unmatched departures are stratum_fold's synthetic month-12 rows (+ three at
    EKCH) and whose matched departures carry the synthetic stand-cache ranking ids (cloned from the unmatched
    rows' airport and hour, so the serve witness covers them); boosters fitted on the stand caches through the
    plug-in (1 thread) and saved in lgbm_submit's fit.json shape. The matched lane's model must be injected
    with `params` (the boosters are not lgbm_submit.P's)."""
    from prc.pipeline import config as C
    from prc.pipeline import lanes
    from prc.pipeline.models import LightGBMModel
    st = legacy.stratum()
    ls = legacy.matched().ls
    rng = np.random.default_rng(seed)
    raw_cols = [c.name for c in ingest.RAW_MOVEMENT]
    # ---- the 2025 stratum and its raw clock rows (the witness is recomputed from them) ----
    unm, lirf, _ = st.uc.synthetic_world(seed=seed, n_per_month=150, n_matched_per_month=120)
    train_unm, train_lirf = unm[unm.month <= 11].reset_index(drop=True), lirf[lirf.month <= 11].reset_index(drop=True)
    rome = root / "cache_rome"
    rome.mkdir(parents=True)
    train_unm.drop(columns=[*st.uc.WITNESS, "sp_band", "sp_fine"]).to_parquet(rome / "unm.parquet", index=False)
    train_lirf.drop(columns=["sp_band", "sp_fine"]).to_parquet(rome / "lirf.parquet", index=False)
    comp = st.uc.smoke_matched_companion(train_unm, seed=seed + 100)
    clock = pd.concat([comp.assign(PHASE_mvt="DEP"),
                       train_unm[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]].assign(PHASE_mvt="DEP"),
                       train_lirf[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]].assign(PHASE_mvt="DEP")],
                      ignore_index=True)
    clock["BLOCK_TIME_UTC_mvt"] = clock.MVT_TIME_UTC_mvt - pd.Timedelta(seconds=600)
    clock["TAXITIME_SEC_mvt"] = np.full(len(clock), 600, dtype="int32")
    clock["ADEP_mvt"] = pd.array(clock.ADEP_mvt.astype(str).tolist(), dtype="str")
    clock["PHASE_mvt"] = pd.array(clock.PHASE_mvt.tolist(), dtype="str")
    train_dir = root / "train_raw"
    train_dir.mkdir()
    clock[list(ingest.TRAINING_CLOCK_COLUMNS)].to_parquet(train_dir / "training_2025-01-01_2025-02-01.parquet", index=False)
    # ---- the matched lane's caches (lgbm_submit's synthetic ones) and boosters fitted on them ----
    sc = _synthetic_caches_module()
    ns = sc.synthetic_submission(root / "matched", months=(1, 2, 3), n=300, seed=seed, n_rank=n_rank, n_unmatched=10, n_arr=10)
    lane = C.Lane(id="matched", kind="matched", model="lightgbm", target="delta",
                  feature_blocks=(("queue", ns.queue), ("order", ns.order)), source="saved_boosters",
                  booster_dir=root / "boosters", boosters=tuple(f"lgbm_seed{s}.txt" for s in (0, 1, 2)))
    X, labels, masks, _, _ = lanes.full_design(lane, ns.stand, (3, 9))
    params = dict(ls.P, learning_rate=0.05, num_threads=1)
    model = LightGBMModel(params=params, nest=60, patience=50)
    model.save(model.fit(X, labels, masks, (0, 1, 2)), lane.booster_dir)
    # ---- the scored file: unmatched = month-12 stratum rows (+ EKCH clones); matched = the ranking ids ----
    s_unm = unm[unm.month == 12][raw_cols].reset_index(drop=True)
    ek = s_unm.sample(3, random_state=seed).assign(ADEP_mvt=NEW, MVT_ID_mvt=7e8 + np.arange(3.0))
    s_unm = pd.concat([s_unm, ek], ignore_index=True)
    src = s_unm.iloc[np.arange(len(ns.rank_ids)) % len(s_unm)].reset_index(drop=True)
    jitter = pd.to_timedelta(rng.integers(-1200, 1200, len(src)), unit="s")
    mat = src.assign(MVT_ID_mvt=ns.rank_ids, MVT_TIME_UTC_mvt=src.MVT_TIME_UTC_mvt + jitter)
    mat["AOBT_3_flt"] = mat.MVT_TIME_UTC_mvt - pd.Timedelta(seconds=900)
    dep = pd.concat([mat, s_unm], ignore_index=True)
    dep["BLOCK_TIME_UTC_mvt"] = pd.Series(pd.NaT, index=dep.index, dtype="datetime64[us, UTC]")
    dep["TAXITIME_SEC_mvt"] = pd.array([None] * len(dep), dtype="Int32")
    dep = _typed(dep[raw_cols])
    scored_dir = root / "scored"
    scored_dir.mkdir()
    dep.to_parquet(scored_dir / "ranking.parquet", index=False)
    pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": np.full(len(dep), np.nan)}).to_parquet(
        scored_dir / "submitting.parquet", index=False)
    raw = real_raw()
    raw["paths"] = {"scored_file": str(scored_dir / "ranking.parquet"), "template_file": str(scored_dir / "submitting.parquet"),
                    "training_dir": str(train_dir), "training_glob": "training_2025-*.parquet",
                    "stand_cache": str(ns.stand), "rome_cache": str(rome)}
    raw["lanes"]["matched"]["feature_blocks"] = {"queue": str(ns.queue), "order": str(ns.order)}
    raw["lanes"]["matched"]["booster_dir"] = str(lane.booster_dir)
    raw["lanes"]["matched"]["boosters"] = list(lane.boosters)
    raw["airports"][NEW] = {"tz": "Europe/Copenhagen", "history": False, "lanes": {"matched": "matched", "unmatched": "unmatched"}}
    cfg_path = root / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return SimpleNamespace(root=root, config=cfg_path, params=params, dep=dep, rank_ids=ns.rank_ids,
                           n_unmatched=len(s_unm), n_lirf_unmatched=int((s_unm.ADEP_mvt == "LIRF").sum()))


# =================================================================================================
# the dry-run world: eleven airports on disk
# =================================================================================================

def _scored_departures(rng, airports, n_matched: int, n_unmatched: int) -> pd.DataFrame:
    rows = []
    next_id = 3e8
    for a in airports:
        for matched, n in ((True, n_matched), (False, n_unmatched)):
            sched = T0 + pd.to_timedelta(rng.integers(0, 20 * 86400, n), unit="s")
            late = pd.to_timedelta(rng.integers(0, 1800, n), unit="s")
            taxi = pd.to_timedelta(rng.integers(300, 1500, n), unit="s")
            mvt = sched + late + taxi
            aobt = (sched + late) if matched else pd.Series(pd.NaT, index=range(n), dtype="datetime64[us, UTC]")
            f = pd.DataFrame({
                "PHASE_mvt": "DEP", "MVT_ID_mvt": next_id + np.arange(n, dtype="float64"), "ADEP_mvt": a,
                "ADES_mvt": rng.choice(["KJFK", "OMDB", "EGLL"], n), "STAND_mvt": rng.choice(["A1", "B2", "C3"], n),
                "RUNWAY_mvt": rng.choice(["09L", "27R"], n), "AIRCRAFT_TYPE_mvt": "A320",
                "FLIGHT_mvt": [f"XYZ{k}" for k in rng.integers(100, 999, n)], "FLIGHT_RULE_mvt": "I",
                "MVT_TIME_UTC_mvt": mvt.to_numpy(), "SCHED_TIME_UTC_mvt": sched.to_numpy(),
                "AOBT_3_flt": np.asarray(aobt), "EOBT_1_flt": sched.to_numpy(), "LOBT_flt": sched.to_numpy(),
                "IOBT_flt": sched.to_numpy(), "AIRCRAFT_OPERATOR_flt": "XYZ", "MARKET_SEGMENT_flt": "S",
                "WK_TBL_CAT_flt": "M", "FLIGHT_TYPE_flt": "S"})
            rows.append(f)
            next_id += n
    d = pd.concat(rows, ignore_index=True)
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"):
        d[c] = pd.to_datetime(d[c], utc=True)
    d["BLOCK_TIME_UTC_mvt"] = pd.Series(pd.NaT, index=d.index, dtype="datetime64[us, UTC]")
    d["TAXITIME_SEC_mvt"] = pd.array([None] * len(d), dtype="Int32")
    return d


def _arrivals(rng, n: int) -> pd.DataFrame:
    t = T0 + pd.to_timedelta(rng.integers(0, 20 * 86400, n), unit="s")
    d = pd.DataFrame({"PHASE_mvt": "ARR", "MVT_ID_mvt": 4e8 + np.arange(n, dtype="float64"),
                      "ADEP_mvt": rng.choice(["KJFK", "ZZZZ"], n), "ADES_mvt": "EDDF"})
    for c in ingest.SERVE_COLUMNS:
        if c not in d.columns:
            d[c] = None
    for c in ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "AOBT_3_flt", "EOBT_1_flt", "LOBT_flt", "IOBT_flt"):
        d[c] = pd.to_datetime(t, utc=True)
    d["BLOCK_TIME_UTC_mvt"] = pd.to_datetime(t + pd.Timedelta(seconds=400), utc=True)
    d["TAXITIME_SEC_mvt"] = pd.array(rng.integers(200, 900, n), dtype="Int32")
    return d


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    """String columns as pandas strings with real nulls (never the text 'None'), as the raw files carry them."""
    frame = frame.copy()
    for c in ingest.RAW_MOVEMENT:
        if c.kind == "string":
            vals = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in frame[c.name]]
            frame[c.name] = pd.array(vals, dtype="str")
    return frame


def dry_run_world(root: pathlib.Path, seed: int = 0, airports=None, config_airports=None, n_matched: int = 12,
                  n_unmatched: int = 3, n_train_rows: int = 50) -> SimpleNamespace:
    """Write every dry-run input for `airports` (default: the ten + EKCH) under `root` and the config for
    `config_airports` (default: the same; EKCH with history false). Returns paths, ids and the raw config."""
    mm = legacy.matched()
    ls, S = mm.ls, mm.S
    rng = np.random.default_rng(seed)
    airports = list(TEN + [NEW] if airports is None else airports)
    config_airports = list(airports if config_airports is None else config_airports)
    raw_dir, stand, queue, order, rome = (root / d for d in ("raw", "cache_stand", "cache_queue", "cache_order", "cache_rome"))
    for d in (raw_dir, stand, queue, order, rome):
        d.mkdir(parents=True, exist_ok=True)
    # ---- the scored file (departures label-free, arrivals with their block times) and the template ----
    dep = _scored_departures(rng, airports, n_matched, n_unmatched)
    scored_file = pd.concat([dep, _arrivals(rng, 30)], ignore_index=True)[[c.name for c in ingest.RAW_MOVEMENT]]
    _typed(scored_file).to_parquet(raw_dir / "ranking.parquet", index=False)
    pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt.to_numpy(), "TAXITIME_SEC_mvt": np.full(len(dep), np.nan)}).to_parquet(
        raw_dir / "submitting.parquet", index=False)
    # ---- a raw training month (the 2025 witness reads its clock columns) ----
    tr = _scored_departures(rng, airports[:3], 20, 5)
    tr["BLOCK_TIME_UTC_mvt"] = pd.to_datetime(tr.SCHED_TIME_UTC_mvt + pd.Timedelta(seconds=300), utc=True)
    tr["TAXITIME_SEC_mvt"] = pd.array(rng.integers(300, 900, len(tr)), dtype="Int32")
    cols = [c.name for c in ingest.RAW_MOVEMENT]
    tr = pd.concat([tr[cols], _arrivals(rng, 6)[cols]], ignore_index=True)       # arrivals too, as real months have
    _typed(tr).to_parquet(raw_dir / "training_2025-01-01_2025-02-01.parquet", index=False)
    # ---- the unmatched-stratum frames (schema only in a dry run) ----
    unm, lirf = legacy.stratum().sf.synthetic_frame(n_per_month=8, n_matched_per_month=8, seed=seed)
    for f, name in ((unm, "unm.parquet"), (lirf, "lirf.parquet")):
        f.drop(columns=["sp_band", "sp_fine"]).to_parquet(rome / name, index=False)
    # ---- the matched lane's caches: ranking rows = the matched scored departures ----
    matched_ids = dep.MVT_ID_mvt.to_numpy()[dep.AOBT_3_flt.notna().to_numpy()]
    base_feats = [f for f in ls.FEATS if f not in S.ENC]

    def stand_frame(n, ids=None, serve=False):
        f = {c: rng.normal(size=n) for c in base_feats}
        f.update({k: rng.choice(["a", "b", "c"], n).astype(object) for k in S.ENC_KEYS})
        f["y"] = np.full(n, np.nan) if serve else rng.uniform(300, 1500, n)
        f["delta"] = np.full(n, np.nan) if serve else rng.normal(0, 60, n)
        f["month"] = np.full(n, 1, dtype="int64")
        f["proxy"] = rng.uniform(300, 1500, n)
        f["ap"] = rng.choice(airports, n).astype(object)            # the real stand caches carry the airport code
        out = pd.DataFrame(f)
        if ids is not None:
            out.insert(0, "MVT_ID_mvt", ids)
        return out

    month_name = "training_2025-01-01_2025-02-01.parquet"
    stand_frame(len(matched_ids), matched_ids, serve=True).to_parquet(stand / "ranking.parquet", index=False)
    stand_frame(n_train_rows).to_parquet(stand / month_name, index=False)
    for bdir, feats in ((queue, ls.QUEUE_FEATS), (order, ls.ORDER_FEATS)):
        for name, ids in (("ranking.parquet", matched_ids), (month_name, np.arange(n_train_rows, dtype="float64"))):
            f = pd.DataFrame({"MVT_ID_mvt": ids, **{c: rng.normal(size=len(ids)).astype("float32") for c in feats}})
            f.to_parquet(bdir / name, index=False)
    n_features = len(ls.FEATS) + len(ls.QUEUE_FEATS) + len(ls.ORDER_FEATS)
    boosters = []
    for s in (0, 1, 2):
        b = stand / f"lgbm_fixture_seed{s}.txt"
        b.write_text("fixture booster: a provenance record for the dry run, never loaded\n")
        (stand / f"lgbm_fixture_seed{s}.fit.json").write_text(json.dumps({
            "best_iter": 40, "n_ref": 48, "n_all": n_train_rows, "smoke": False, "params": dict(ls.P, seed=s),
            "es_seed": 0, "n_features": n_features, "target": "delta", "all_rows": False, "unmatched_weight": 1.0}))
        boosters.append(b.name)
    raw = real_raw()
    raw["paths"] = {"scored_file": str(raw_dir / "ranking.parquet"), "template_file": str(raw_dir / "submitting.parquet"),
                    "training_dir": str(raw_dir), "training_glob": "training_2025-*.parquet",
                    "stand_cache": str(stand), "rome_cache": str(rome)}
    raw["lanes"]["matched"]["feature_blocks"] = {"queue": str(queue), "order": str(order)}
    raw["lanes"]["matched"]["booster_dir"] = str(stand)
    raw["lanes"]["matched"]["boosters"] = boosters
    raw["airports"] = {a: v for a, v in raw["airports"].items() if a in config_airports}
    if NEW in config_airports:
        raw["airports"][NEW] = {"tz": "Europe/Copenhagen", "history": False,
                                "lanes": {"matched": "matched", "unmatched": "unmatched"}}
    cfg_path = root / "pipeline.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return SimpleNamespace(root=root, config=cfg_path, raw=raw, dep=dep, matched_ids=matched_ids,
                           n_by_airport=dep.ADEP_mvt.value_counts().to_dict())


def assert_same_run(m1: dict, m2: dict, since: float, root) -> None:
    """Two pipeline runs on the same inputs agree (the end-to-end determinism check).

    The manifest's code block hashes every repo module loaded in the process, so in a shared pytest process on a working
    tree other sessions edit, a module saved between the two runs changes the second digest -- correctly (BC-8, 2026-09-11).
    So: the same modules must be loaded; a code hash may differ only for a file modified on disk at or after `since` (the
    first run's start); everything except the code block must hash identically; with no code change the full digests must
    be equal."""
    from prc.pipeline import write
    c1, c2 = m1["code"]["files"], m2["code"]["files"]
    assert set(c1) == set(c2), f"the two runs loaded different code files: {sorted(set(c1) ^ set(c2))}"
    changed = sorted(f for f in c1 if c1[f] != c2[f])
    stale = [f for f in changed if (pathlib.Path(root) / f).stat().st_mtime < since]
    assert not stale, f"code hash changed between the runs without an on-disk edit: {stale}"
    strip = lambda m: {k: v for k, v in m.items() if k != "code"}
    assert write.manifest_digest(strip(m1)) == write.manifest_digest(strip(m2)), "the runs differ apart from the code block"
    if not changed:
        assert m1["digest"] == m2["digest"], "digest differs with identical code and an identical manifest body"
