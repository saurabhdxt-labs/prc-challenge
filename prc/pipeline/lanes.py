"""Lanes: each owns a set of scored rows and returns (MVT_ID, prediction, provenance) for exactly those.

    matched        rows with an NM off-block      LightGBM delta model (lgbm_submit), taxi = max(proxy - d, 1)
    unmatched      rows without one (not LIRF)    S1's mixture with E3C's congestion body (unm_congestion_ship)
    airport_rules  LIRF rows without one          S1's mixture with arm R's p (rome_fill), the date-slip
                                                  expectation (rome_dateslip), the schedule floor (rome_bandfloor)

Every lane is a thin composition of the SHIPPED functions -- the same calls, in the same order, on the same
frames as the ship scripts made them -- with each script's splice-into-a-base step left out: the pipeline
builds every row from data and never reads an earlier submission. The shared upstream stage is S1's
`fit_unmatched(..., hybrid=True)` over ALL scored unmatched rows, exactly as rome_ship / unm_congestion_ship
ran it; both unmatched lanes read its parts.

Routing: a scored row's class is `derive()`'s `unmatched` (AOBT_3_flt null); its lane is the config's
airports[ADEP].lanes[class]. Every scored row lands in exactly one lane (asserted); an airport outside the
registry is refused; an airport without history rides the pooled lanes and is flagged `fallback`.
"""
from __future__ import annotations

import gc
import pathlib
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config as C
from . import errors as E
from . import ingest, legacy
from .models import Labels, Masks, check_prediction, make_model

PROVENANCE_COLUMNS = ["MVT_ID_mvt", "lane", "model", "stage"]


@dataclass
class LaneResult:
    lane: str
    ids: np.ndarray                             # MVT_ID_mvt, float64
    values: np.ndarray                          # final taxi seconds, integral, float64 or int
    provenance: pd.DataFrame                    # PROVENANCE_COLUMNS, row-aligned with ids
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.ids = np.asarray(self.ids, dtype="float64")
        v = np.asarray(self.values)
        if len(v) != len(self.ids) or len(self.provenance) != len(self.ids):
            raise E.LaneError(f"lane {self.lane}: {len(self.ids)} ids, {len(v)} values, {len(self.provenance)} provenance rows")
        if not pd.Index(self.ids).is_unique:
            raise E.LaneError(f"lane {self.lane}: duplicate ids")
        vf = v.astype("float64")
        if not np.isfinite(vf).all():
            raise E.LaneError(f"lane {self.lane}: {int((~np.isfinite(vf)).sum())} non-finite predictions")
        if not np.array_equal(vf, np.rint(vf)):
            raise E.LaneError(f"lane {self.lane}: predictions must be whole seconds (rounded by the lane)")
        if list(self.provenance.columns) != PROVENANCE_COLUMNS:
            raise E.LaneError(f"lane {self.lane}: provenance columns {list(self.provenance.columns)} != {PROVENANCE_COLUMNS}")
        if not np.array_equal(self.provenance.MVT_ID_mvt.to_numpy(dtype="float64"), self.ids):
            raise E.LaneError(f"lane {self.lane}: provenance rows are not aligned with the ids")
        self.values = vf


def _prov(ids, lane: str, model: str, stage) -> pd.DataFrame:
    n = len(ids)
    stage = np.asarray(stage, dtype=object) if not isinstance(stage, str) else np.full(n, stage, dtype=object)
    return pd.DataFrame({"MVT_ID_mvt": np.asarray(ids, dtype="float64"), "lane": np.full(n, lane, dtype=object),
                         "model": np.full(n, model, dtype=object), "stage": stage})


# =================================================================================================
# pins: the config's pinned values against the scripts' own constants, at run time
# =================================================================================================

def check_pins(cfg) -> dict:
    """Refuse to run if a wrapped script's constant differs from what config.PINNED states (a script edit
    must not drift from the config silently), or if the config's early-stopping months differ from
    lgbm_submit's while a matched lane refits. Returns the constants checked."""
    st = legacy.stratum()
    got = {"schedule_floor_lo_s": float(st.rb.LO), "schedule_floor_hi_s": float(st.rb.HI),
           "rules_airport": st.rb.AIRPORT}
    rl = legacy.rome_local_day()
    got.update(local_day_lo_s=float(rl.SP_LO), local_day_hi_s=float(rl.SP_HI))
    also = {"unm_congestion.ROME": st.uc.ROME, "unm_congestion_ship.ROME": st.ucs.ROME, "rome_local_day.AIRPORT": rl.AIRPORT}
    bad = {k: (v, C.PINNED[k]) for k, v in got.items() if v != C.PINNED[k]}
    bad.update({k: (v, C.PINNED["rules_airport"]) for k, v in also.items() if v != C.PINNED["rules_airport"]})
    if tuple(st.bs.SEEDS) != tuple(cfg.seeds) or tuple(st.sf.SEEDS) != tuple(cfg.seeds):
        bad["seeds"] = (tuple(cfg.seeds), tuple(st.bs.SEEDS))
    if any(l.kind == "matched" and l.source == "refit" for l in cfg.lanes.values()):
        ls = legacy.matched().ls
        if tuple(cfg.folds.early_stopping_months) != tuple(ls.ES_MONTHS):
            bad["early_stopping_months"] = (tuple(cfg.folds.early_stopping_months), tuple(ls.ES_MONTHS))
    if bad:
        raise E.PinnedValueError("P2a wraps the shipped code unchanged; these differ between the scripts/config and "
                                 "the pins: " + "; ".join(f"{k}: {a!r} vs {b!r}" for k, (a, b) in bad.items()))
    return {**got, **also, "seeds": tuple(cfg.seeds)}


# =================================================================================================
# routing
# =================================================================================================

@dataclass(frozen=True)
class Routing:
    ids: np.ndarray
    airport: np.ndarray
    row_class: np.ndarray
    lane: np.ndarray
    fallback: np.ndarray

    def mask(self, lane: str) -> np.ndarray:
        return self.lane == lane

    def counts(self) -> dict:
        return {str(k): int(v) for k, v in pd.Series(self.lane).value_counts().sort_index().items()}

    def by_airport(self) -> dict:
        t = pd.crosstab(pd.Series(self.airport, name="airport"), pd.Series(self.lane, name="lane"))
        return {str(a): {str(l): int(n) for l, n in row.items() if n} for a, row in t.iterrows()}


def route(scored: pd.DataFrame, cfg) -> Routing:
    """Every scored row -> exactly one lane. Refuses an unknown airport; asserts the partition."""
    for c in ("MVT_ID_mvt", "ADEP_mvt", "unmatched", "AOBT_3_flt"):
        if c not in scored.columns:
            raise E.RoutingError(f"routing needs column {c!r}")
    ap = scored.ADEP_mvt.astype(str).to_numpy()
    ingest.check_airports(ap, cfg.airport_codes, "routing")
    unm = scored.unmatched.to_numpy(dtype=bool)
    if not np.array_equal(unm, scored.AOBT_3_flt.isna().to_numpy()):
        raise E.RoutingError("derive()'s `unmatched` disagrees with AOBT_3_flt being null")
    rc = np.where(unm, "unmatched", "matched").astype(object)
    table = {(a, r): cfg.airports[a].lanes[r] for a in cfg.airports for r in C.ROW_CLASSES}
    lane = np.array([table[(a, r)] for a, r in zip(ap, rc)], dtype=object)
    fallback = np.array([cfg.airports[a].fallback for a in ap], dtype=bool)
    masks = np.stack([lane == lid for lid in cfg.lanes]) if len(lane) else np.zeros((len(cfg.lanes), 0), bool)
    per_row = masks.sum(axis=0)
    if len(lane) and not (per_row == 1).all():
        raise E.RoutingError(f"{int((per_row != 1).sum()):,} scored rows land in {sorted(set(per_row.tolist()))} lanes, not one")
    return Routing(ids=scored.MVT_ID_mvt.to_numpy(dtype="float64"), airport=ap.astype(object), row_class=rc,
                   lane=lane, fallback=fallback)


# =================================================================================================
# the unmatched stratum: training inputs, the 2025 witness, S1's parts
# =================================================================================================

@dataclass
class StratumInputs:
    train_unm: pd.DataFrame                     # the 2025 unmatched stratum, the 2025 witness attached
    train_lirf: pd.DataFrame                    # LIRF's matched 2025 rows (the only matched rows fit_unmatched reads)
    witness_2025: dict


def training_witness(cfg) -> pd.DataFrame:
    """The 2025 airport-hour witness: unm_congestion.airport_hour_witness over the ADMISSIBLE training
    departures (build_submission.admissible), read through the clock columns only. unm_congestion.load_real
    computes it on derive(admissible(load_movements(12 months)))[~unmatched]; derive adds columns and drops no
    row, and the witness drops unmatched rows itself, so the two are the same table."""
    st = legacy.stratum()
    files = cfg.paths.training_files()
    if not files:
        raise E.SchemaError(f"no training files {cfg.paths.training_glob!r} under {cfg.paths.training_dir}")
    rows = st.bs.admissible(ingest.read_training_clock_rows(files))
    return st.uc.airport_hour_witness(rows)


def stratum_inputs(cfg, witness: pd.DataFrame | None = None) -> StratumInputs:
    st = legacy.stratum()
    unm, lirf = ingest.load_rome_frames(cfg.paths.rome_cache)
    wit = training_witness(cfg) if witness is None else witness
    unm = st.uc.attach_witness(unm, wit)
    summary = {"n_airport_hours": int(len(wit)), "unmatched_nan_share": float(unm.hprox_med.isna().mean()),
               "n_train_unmatched": int(len(unm)), "n_train_lirf_matched": int(len(lirf))}
    return StratumInputs(train_unm=unm, train_lirf=lirf, witness_2025=summary)


def s1_parts(inputs: StratumInputs, scored_unm: pd.DataFrame, seeds) -> dict:
    """build_submission.fit_unmatched(hybrid=True) over every scored unmatched row: p, sp, nf_cells, nf_fit,
    nf (the S1 mixture's parts), exactly as rome_ship.main / unm_congestion_ship.main computed them."""
    bs = legacy.stratum().bs
    parts: dict = {}
    bs.fit_unmatched(inputs.train_unm, scored_unm, train_matched=inputs.train_lirf, hybrid=True,
                     seeds=tuple(seeds), parts=parts)
    return parts


def _body_nf(model_name: str, inputs: StratumInputs, scored_rows: pd.DataFrame, parts: dict, seeds) -> tuple:
    """(nf, nf_body): the non-fill term through the model PLUG-IN, fitted on fit_unmatched's own non-fill
    rows (`train[~schedule_fill(train)]`), nf = nf_hybrid(nf_cells, body). For the plain body the plug-in must
    reproduce fit_unmatched's fused nf_fit bit for bit, or the lane refuses."""
    bs = legacy.stratum().bs
    model = make_model(model_name)
    train = inputs.train_unm
    state = model.fit(train, Labels(taxi=train.y.to_numpy(dtype="float64")),
                      Masks(train=~bs.schedule_fill(train)), seeds)
    body = check_prediction(model, model.predict(state, scored_rows), len(scored_rows))
    if model_name == "nonfill_body" and not np.array_equal(body, np.asarray(parts["nf_fit"], dtype="float64")):
        raise E.LaneError("the nonfill_body plug-in does not reproduce fit_unmatched's nf_fit exactly")
    return bs.nf_hybrid(parts["nf_cells"], body), body


def run_unmatched_lane(lane: C.Lane, rows: np.ndarray, scored_unm: pd.DataFrame, parts: dict,
                       inputs: StratumInputs, serve_dep: pd.DataFrame, seeds) -> LaneResult:
    """E3C (unm_congestion_ship.build without its splice): the serve file's witness attached by
    `attach_serve_witness`, S1's p / sp / nf_cells untouched, the body re-fitted through the plug-in,
    pred = rint(mixture(p, sp, nf_hybrid(nf_cells, body)))."""
    st = legacy.stratum()
    bs = st.bs
    rows = np.asarray(rows, dtype=bool)
    if lane.model == "nonfill_body_congestion":
        scored_w, wsum = st.ucs.attach_serve_witness(scored_unm, serve_dep[st.ucs.WITNESS_COLUMNS])
    else:
        scored_w, wsum = scored_unm, None
    nf, body = _body_nf(lane.model, inputs, scored_w, parts, seeds)
    pred = bs.mixture(parts["p"], parts["sp"], nf)
    vals = np.rint(pred[rows])
    if not (np.isfinite(vals).all() and (vals >= 1.0).all()):
        raise E.LaneError(f"lane {lane.id}: a non-finite or sub-floor value")
    routed = np.asarray(parts["nf_cells"], dtype="float64")[rows] < bs.T_TAIL_S
    tag = "S1C" if lane.model == "nonfill_body_congestion" else "S1"
    stage = np.where(routed, f"{tag}:body", f"{tag}:cells")
    ids = scored_unm.MVT_ID_mvt.to_numpy(dtype="float64")[rows]
    info = {"n_rows": int(rows.sum()), "n_routed_to_body": int(routed.sum()), "witness_serve": wsum,
            "n_train_nonfill": int((~bs.schedule_fill(inputs.train_unm)).sum()), "seeds": list(seeds)}
    return LaneResult(lane=lane.id, ids=ids, values=vals, provenance=_prov(ids, lane.id, lane.model, stage), info=info)


def run_rules_lane(lane: C.Lane, airport: C.Airport, rows: np.ndarray, scored_unm: pd.DataFrame, parts: dict,
                   inputs: StratumInputs, seeds) -> LaneResult:
    """The LIRF rules, in the shipped order, on the airport's unmatched rows:
      fill_classifier  p = mean over seeds of rome_fill.fit_predict_fill on every LIRF 2025 departure
                       (rome_ship.build's lines, without its rebuild-the-base guard or splice);
                       else S1's p;
      the mixture      rome_fill.mixture(p, sp, nf), nf = S1's hybrid nf through the plug-in;
      dateslip         rome_dateslip.apply with band_shares from the airport's 2025 unmatched rows;
      rounding         np.rint (the value a submission stores);
      schedule_floor   rome_bandfloor.build on the lane's own frame: max(v, rint(sp)) on lo <= sp < hi.
    """
    st = legacy.stratum()
    rf, ds, rb = st.rf, st.ds, st.rb
    rows = np.asarray(rows, dtype=bool)
    te = scored_unm[rows]
    nf, _ = _body_nf(lane.model, inputs, scored_unm, parts, seeds)
    rules = airport.rules
    train_unm, train_lirf = inputs.train_unm, inputs.train_lirf
    info: dict = {"rules": list(rules.order), "n_rows": int(rows.sum()), "seeds": list(seeds)}
    if rules.fill_classifier:
        tr = rf.rome_rows(train_unm, train_lirf, sorted(set(train_unm.month) | set(train_lirf.month)))
        p = np.mean([rf.fit_predict_fill(tr, te, seed=s) for s in seeds], axis=0)
        info["n_fill_train"] = int(len(tr))
    else:
        p = np.asarray(parts["p"], dtype="float64")[rows]
    sp = te.sp.to_numpy(dtype="float64")
    v = rf.mixture(p, sp, np.asarray(nf)[rows])
    in_g = np.zeros(len(te), dtype=bool)
    if rules.dateslip:
        q, mt = ds.band_shares(train_unm[train_unm.ADEP_mvt == airport.code])
        v = ds.apply(te, v, q, mt)
        in_g = ds.segment_mask(te)
        info.update(q=q, m=mt, n_in_segment_g=int(in_g.sum()))
    v = np.rint(v)
    ids = te.MVT_ID_mvt.to_numpy(dtype="float64")
    floored = np.zeros(len(te), dtype=bool)
    if rules.schedule_floor is not None:
        frame = pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": v.astype("int32")})
        if not np.array_equal(frame.TAXITIME_SEC_mvt.to_numpy(dtype="float64"), v):
            raise E.LaneError(f"lane {lane.id}: a value does not fit int32 before the floor")
        out, finfo = rb.build(frame, te[["MVT_ID_mvt", "ADEP_mvt", "unmatched", "sp"]])
        after = out.TAXITIME_SEC_mvt.to_numpy(dtype="float64")
        floored = after != v
        v = after
        info.update(n_floor_rule_rows=finfo["n_rule_rows"], n_floor_changed=finfo["n_values_differ"])
    rld = np.zeros(len(te), dtype=bool)
    if rules.local_day is not None:
        v, rld = _local_day(te.reset_index(drop=True), v)
        info.update(n_local_day_changed=int(rld.sum()), local_day_changed_ids=ids[rld].tolist())
    head = "R2" if rules.fill_classifier else "S1"
    stage = np.array([head + ("+DS" if g else "") + ("+E1" if f else "") + ("+RLD" if r else "")
                      for g, f, r in zip(in_g, floored, rld)], dtype=object)
    info["p"] = p
    model = lane.model + ("+rome_fill_R" if rules.fill_classifier else "")
    return LaneResult(lane=lane.id, ids=ids, values=v, provenance=_prov(ids, lane.id, model, stage), info=info)


def _local_day(te: pd.DataFrame, v) -> tuple:
    """RLD after the floor: rome_local_day.apply_rld on the lane's own values. (new values float64, rows changed).
    Refuses a lane whose sp is not MVT − SCHED in seconds: the rule recomputes it from the two clocks."""
    rl = legacy.rome_local_day()
    sp = (pd.to_datetime(te.MVT_TIME_UTC_mvt, utc=True) - pd.to_datetime(te.SCHED_TIME_UTC_mvt, utc=True)).dt.total_seconds()
    if not np.allclose(sp.to_numpy(), te.sp.to_numpy(dtype="float64"), rtol=0, atol=1e-6):
        raise E.LaneError("the lane's sp differs from MVT - SCHED; RLD would key on another quantity")
    frame = te[["MVT_ID_mvt", "ADEP_mvt", "unmatched", "SCHED_TIME_UTC_mvt", "MVT_TIME_UTC_mvt"]].assign(
        base=np.asarray(v, dtype="float64"))
    new, moved = rl.apply_rld(frame)
    return np.asarray(new, dtype="float64"), np.asarray(moved, dtype=bool)


# =================================================================================================
# the matched lane
# =================================================================================================

@dataclass
class MatchedDesign:
    ids: np.ndarray                             # MVT_ID_mvt of the ranking cache rows, cache order
    X: np.ndarray                               # float32, len(ids) x len(feats)
    proxy: np.ndarray                           # float64
    feats: list
    n_train: int                                # training rows the encodings were fitted on


_BLOCK_API = {   # block -> (lgbm_submit feature list attr, reader, cache-path fn, by-id attach)
    "queue": ("QUEUE_FEATS", "read_queue_cache", "queue_cache_path", "attach_queue_by_id"),
    "day": ("DAY_FEATS", "read_day_cache", "day_cache_path", "attach_day_by_id"),
    "order": ("ORDER_FEATS", "read_order_cache", "order_cache_path", "attach_order_by_id"),
    "weather": ("WEATHER_FEATS", "read_weather_cache", "weather_cache_path", "attach_weather_by_id"),
}


def matched_features(lane: C.Lane) -> list:
    """lgbm_submit's column order: FEATS + the configured blocks in FEATURE_BLOCKS order."""
    ls = legacy.matched().ls
    return list(ls.FEATS) + [f for b, _ in lane.feature_blocks for f in getattr(ls, _BLOCK_API[b][0])]


def stand_training_paths(stand_cache) -> list:
    """The stand-cache training months in lgbm_submit.training_frames' order (sorted glob)."""
    return sorted(pathlib.Path(stand_cache).glob("training_2025-*.parquet"))


def scored_design(lane: C.Lane, stand_cache) -> MatchedDesign:
    """The ranking rows' design matrix, identical to the ranking slice of lgbm_submit.main's (load_frames ->
    infold_encodings on every training row -> design_matrix), built without materialising the 2.06M-row
    training frame: the encodings are fitted ONE KEY AT A TIME through stand_ab.infold_encodings(enc_keys=[k])
    on the training months' key column plus the ranking rows' (the same rows, order, prior and smoothing), and
    the blocks are joined by id with lgbm_submit's attach functions."""
    mm = legacy.matched()
    ls, S = mm.ls, mm.S
    stand_cache = pathlib.Path(stand_cache)
    paths = stand_training_paths(stand_cache)
    if not paths:
        raise FileNotFoundError(f"no training caches under {stand_cache}")
    rank_path = stand_cache / "ranking.parquet"
    rank = pd.read_parquet(rank_path)
    if not (rank.y.isna().all() and rank.delta.isna().all()):
        raise E.LabelLeakError("the ranking cache carries labels - not a serve-mode build")
    if not set(rank.month.unique()) <= {1, 7}:
        raise E.SchemaError(f"ranking cache months {sorted(rank.month.unique())}, expected 1 and 7")
    rank_ids = rank.MVT_ID_mvt.to_numpy(dtype="float64")
    if not pd.Index(rank_ids).is_unique:
        raise E.SchemaError("duplicate MVT_ID in the ranking cache")
    for block, bdir in lane.feature_blocks:
        feats_attr, reader, path_fn, attach = _BLOCK_API[block]
        q = getattr(ls, reader)(getattr(ls, path_fn)(bdir, rank_path))
        got = getattr(ls, attach)(rank_ids, q, subset_ok=False)
        for c in getattr(ls, feats_attr):
            rank[c] = got[c].to_numpy()
    # training labels, in training_frames' order, refused if NaN (lgbm_submit.training_frames' guard)
    ys, ds_ = [], []
    for p in paths:
        t = pq.read_table(p, columns=["y", "delta"]).to_pandas()
        if t.y.isna().any() or t.delta.isna().any():
            raise E.LabelLeakError(f"{p.name}: training cache has NaN labels - built in serve mode?")
        ys.append(t.y.to_numpy())
        ds_.append(t.delta.to_numpy())
    y_tr, d_tr = np.concatenate(ys), np.concatenate(ds_)
    n_train = len(y_tr)
    y_all = np.concatenate([y_tr, rank.y.to_numpy()])
    d_all = np.concatenate([d_tr, rank.delta.to_numpy()])
    tr_mask = np.zeros(len(y_all), dtype=bool)
    tr_mask[:n_train] = True
    del ys, ds_
    for k in S.ENC_KEYS:
        col = pd.concat([pq.read_table(p, columns=[k]).to_pandas()[k] for p in paths] + [rank[k]], ignore_index=True)
        enc = S.infold_encodings(pd.DataFrame({k: col}), y_all, d_all, tr_mask, enc_keys=[k])
        for name, vals in enc.items():
            rank[name] = vals[n_train:]
        del col, enc
        gc.collect()
    nan_enc = [c for c in S.ENC if np.isnan(rank[c].to_numpy()).any()]
    if nan_enc:
        raise E.LaneError(f"NaN encodings on ranking rows: {nan_enc}")
    feats = matched_features(lane)
    X = ls.design_matrix(rank, feats)
    return MatchedDesign(ids=rank_ids, X=X, proxy=rank.proxy.to_numpy(dtype="float64"), feats=feats, n_train=n_train)


def booster_pairs(lane: C.Lane) -> list:
    return [(lane.booster_dir / b, lane.booster_dir / (pathlib.Path(b).stem + ".fit.json")) for b in lane.boosters]


def full_design(lane: C.Lane, stand_cache, es_months):
    """The refit path's inputs, exactly lgbm_submit.main's steps 1-3: load_frames, split_masks, encodings on
    every training row, design_matrix. Heavy on the real caches (~5 GB peak with the fit)."""
    mm = legacy.matched()
    ls, S = mm.ls, mm.S
    kw = {}
    for block, bdir in lane.feature_blocks:
        kw[block] = True
        kw[{"queue": "qcache", "day": "dcache", "order": "ocache", "weather": "wcache"}[block]] = bdir
    d, is_rank, rank_ids = ls.load_frames(pathlib.Path(stand_cache), **kw)
    train, fit, es = ls.split_masks(d.month.to_numpy(), is_rank, es_months=tuple(es_months))
    y, dlt, proxy = d.y.to_numpy(), d.delta.to_numpy(), d.proxy.to_numpy()
    for col, vals in S.infold_encodings(d, y, dlt, train).items():
        d[col] = vals
    X = ls.design_matrix(d, matched_features(lane))
    return X, Labels(taxi=y, delta=dlt, proxy=proxy), Masks(train=train, fit=fit, es=es), is_rank, rank_ids


def run_matched_lane(lane: C.Lane, cfg, expected_ids, model=None, log=print, stage_predictor=None) -> LaneResult:
    """Saved boosters: the streaming design + the boosters loaded one at a time (provenance-checked).
    Refit: lgbm_submit's full design and the plug-in's fit. Either way taxi = finalise(raw_taxi_time)."""
    ls = legacy.matched().ls
    model = model if model is not None else make_model(lane.model, target=lane.target)
    t0 = time.time()
    if lane.source == "saved_boosters":
        design = scored_design(lane, cfg.paths.stand_cache)
        log(f"matched design {design.X.shape} float32 in {time.time() - t0:.0f}s (encodings on {design.n_train:,} training rows)")
        state = model.load(booster_pairs(lane), n_all=design.n_train, n_features=len(design.feats), seeds=cfg.seeds)
        ids, X, proxy = design.ids, design.X, design.proxy
    else:
        X_all, labels, masks, is_rank, ids = full_design(lane, cfg.paths.stand_cache, cfg.folds.early_stopping_months)
        state = model.fit(X_all, labels, masks, cfg.seeds)
        X, proxy = X_all[is_rank], labels.proxy[is_rank]
        del X_all
    expected = set(np.asarray(expected_ids, dtype="float64").tolist())
    if set(ids.tolist()) != expected:
        raise E.LaneError(f"lane {lane.id}: the design covers {len(ids):,} rows, the routing gives it {len(expected):,} "
                          f"({len(expected - set(ids.tolist())):,} routed rows missing from the design)")
    t1 = time.time()
    pred = check_prediction(model, model.predict(state, X), len(X))
    vals = ls.finalise_taxi_time(ls.raw_taxi_time(pred, proxy, lane.target))
    log(f"matched lane: predicted {len(ids):,} rows in {time.time() - t1:.0f}s")
    stage_info, stage_rows = None, np.zeros(len(ids), dtype=bool)
    if lane.adsb_stage is not None:
        vals, stage_info, stage_rows = _adsb_stage(lane, cfg, ids, proxy, pred, vals, stage_predictor, log)
    # no wall-clock value in `info`: it lands in the manifest's deterministic part (timings live in timings_s)
    info = {"n_rows": int(len(ids)), "source": lane.source, "n_features": int(X.shape[1]), "seeds": list(state.seeds),
            "best_iter": state.info.get("best_iter"), "n_ref": state.info.get("n_ref"), "n_all": state.n_all,
            "boosters": [p.name for p, _ in state.files] if state.files else None}
    if stage_info is not None:
        info["adsb_stage"] = stage_info
    base_stage = f"{lane.model}:{lane.target}:mean{len(state.seeds)}"
    stage = np.where(stage_rows, base_stage + "+adsb", base_stage).astype(object)
    return LaneResult(lane=lane.id, ids=ids, values=vals, provenance=_prov(ids, lane.id, lane.model, stage), info=info)


def _adsb_stage(lane: C.Lane, cfg, ids, proxy, pred, vals, stage_predictor, log) -> tuple:
    """The ADS-B stack stage (prc/pipeline/adsb_stage.py) on the matched lane's delta prediction. Needs lane.target delta."""
    from . import adsb_stage as AS
    if lane.target != "delta":
        raise E.LaneError(f"lane {lane.id}: the ADS-B stage adds to a DELTA prediction; target is {lane.target!r}")
    table, tmeta = AS.read_table(lane.adsb_stage.table, ids)
    gate, man = AS.read_gate(lane.adsb_stage.models_dir)
    rank = pd.read_parquet(cfg.paths.stand_cache / "ranking.parquet", columns=["MVT_ID_mvt", "ap", "hr"])
    rank = rank.set_index(rank.MVT_ID_mvt.to_numpy(dtype="float64")).reindex(ids)
    if rank.ap.isna().any():
        raise E.LaneError("ranking cache rows missing for the stage's ids")
    frame = AS.stage_frame(ids, rank.ap.to_numpy(), rank.hr.to_numpy(), proxy, pred, table)
    if stage_predictor is not None:
        fn = stage_predictor
    else:
        real = legacy.adsb_predictor()
        fn = lambda f: real(f, lane.adsb_stage.models_dir, oof=False)   # the full bag: 2026 rows only
    new, info = AS.apply_stage(frame, fn, gate)
    base = np.asarray(vals, dtype="float64")
    if not np.array_equal(np.rint(np.maximum(frame.proxy.to_numpy() - frame.F.to_numpy(), 1.0)), base):
        raise E.LaneError("the stage's own base value differs from the lane's; the stage would not add to what ships")
    info.update(table_meta=tmeta, gate=gate, models_manifest={k: man[k] for k in man if k != "gate"})
    log(f"ADS-B stage: {info['n_changed']:,} of {info['n_rows']:,} matched rows changed ({info['changed_by_airport']})")
    return new, info, new != base


# =================================================================================================
# dry run: every lane input present, schema-valid and consistent with the routing -- nothing is fitted
# =================================================================================================

def check_lane_inputs(cfg, routing: Routing) -> dict:
    """Metadata-level checks of every input the routed lanes will read. Returns a report; raises a named
    error on the first failure."""
    report: dict = {}
    kinds = {cfg.lanes[l].kind for l in set(routing.lane.tolist())}
    models = {cfg.lanes[l].model for l in set(routing.lane.tolist())}
    for lid in sorted(set(routing.lane.tolist())):
        m = make_model(cfg.lanes[lid].model, **({"target": cfg.lanes[lid].target} if cfg.lanes[lid].kind == "matched" else {}))
        d = m.describe()
        if not d["implemented"]:
            raise NotImplementedError(f"lane {lid}: model {d['name']!r} is not implemented in P2a -- {d.get('why')}")
    if kinds & {"unmatched", "airport_rules"}:
        for name in ("unm.parquet", "lirf.parquet"):
            ingest.check_file_schema(cfg.paths.rome_cache / name, ingest.ROME_FRAME, where=f"rome cache {name}")
        report["rome_cache"] = str(cfg.paths.rome_cache)
        if "nonfill_body_congestion" in models:
            files = cfg.paths.training_files()
            if not files:
                raise E.SchemaError(f"no training files {cfg.paths.training_glob!r} under {cfg.paths.training_dir}")
            for f in files:
                ingest.check_file_schema(f, [ingest.RAW_BY_NAME[c] for c in ingest.TRAINING_CLOCK_COLUMNS],
                                         where=f"training file {f.name}")
            report["training_files"] = [f.name for f in files]
    for lid, lane in cfg.lanes.items():
        if lane.kind != "matched" or not routing.mask(lid).any():
            continue
        report[lid] = _check_matched_inputs(lane, cfg, routing.ids[routing.mask(lid)])
    return report


def _check_matched_inputs(lane: C.Lane, cfg, routed_ids) -> dict:
    mm = legacy.matched()
    ls, S = mm.ls, mm.S
    sc = cfg.paths.stand_cache
    paths = stand_training_paths(sc)
    if not paths:
        raise E.SchemaError(f"no stand-cache training months under {sc}")
    rank_path = sc / "ranking.parquet"
    base_feats = [f for f in ls.FEATS if f not in S.ENC]
    need_rank = ["MVT_ID_mvt", "y", "delta", "month", "proxy", *base_feats, *S.ENC_KEYS]
    if lane.adsb_stage is not None:
        need_rank += [c for c in ("ap", "hr") if c not in need_rank]            # _adsb_stage reads both from the cache
    for p, need in [(rank_path, need_rank)] + [(p, ["y", "delta", *base_feats, *S.ENC_KEYS]) for p in paths]:
        if not p.exists():
            raise E.SchemaError(f"matched lane input {p} does not exist")
        missing = [c for c in need if c not in pq.read_schema(p).names]
        if missing:
            raise E.MissingColumnError(f"{p.name}: missing column(s) {missing}")
    rank_ids = pq.read_table(rank_path, columns=["MVT_ID_mvt"]).column(0).to_numpy()
    if set(rank_ids.tolist()) != set(np.asarray(routed_ids, dtype="float64").tolist()):
        raise E.RoutingError(f"{rank_path}: the ranking cache holds {len(rank_ids):,} rows, the matched lane is routed "
                             f"{len(routed_ids):,}; a stale cache for another scored file?")
    n_rows = {p.name: pq.ParquetFile(p).metadata.num_rows for p in paths}
    for block, bdir in lane.feature_blocks:
        feats = getattr(ls, _BLOCK_API[block][0])
        want = ["MVT_ID_mvt", *feats]
        for p in [rank_path, *paths]:
            twin = pathlib.Path(bdir) / p.name
            if not twin.exists():
                raise E.SchemaError(f"{block} block twin {twin} does not exist")
            names = pq.read_schema(twin).names
            if names != want:
                raise E.MissingColumnError(f"{twin}: {block} cache columns {names} != the contract {want}")
            if p.name in n_rows and pq.ParquetFile(twin).metadata.num_rows != n_rows[p.name]:
                raise E.SchemaError(f"{twin}: row count differs from the stand cache's {p.name} (positional join)")
    out = {"stand_cache": str(sc), "n_training_months": len(paths), "n_training_rows": int(sum(n_rows.values())),
           "blocks": [b for b, _ in lane.feature_blocks], "n_features": len(matched_features(lane))}
    if lane.source == "saved_boosters":
        model = make_model(lane.model, target=lane.target)
        st = model.load(booster_pairs(lane), n_all=out["n_training_rows"], n_features=out["n_features"], seeds=cfg.seeds)
        out.update(boosters=[p.name for p, _ in st.files], best_iter=st.info.get("best_iter"), n_ref=st.info.get("n_ref"))
    if lane.adsb_stage is not None:
        from . import adsb_stage as AS
        out["adsb_stage"] = AS.check_stage_inputs(lane.adsb_stage.table, lane.adsb_stage.models_dir, rank_ids)
    return out
