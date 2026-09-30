"""DFC — destination / downstream-flow congestion features
(plans/PREREG_dfc_destination_flow_congestion_2026_09_19.md, sha256 c217c0cd).

Arm A6 = shipped 93 + 6 DFC columns, on RCR's chronological split, vs RCR's SAVED A0 predictions.

The arrival columns are a VALUE LOOKUP keyed on (destination, time) — both present on the cached frame —
never a row join against `data/raw`, whose DEP row count differs from the stand cache's (164,450 vs
163,361 in March). A positional join there is this repo's documented misjoin bug class.

    /opt/homebrew/bin/python3.11 -B scripts/analysis/dfc_build.py run [--smoke]
"""
from __future__ import annotations

import argparse, glob, hashlib, importlib.util, json, pathlib, resource, sys, time
from dataclasses import dataclass
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PREREG = ROOT / "plans" / "PREREG_dfc_destination_flow_congestion_2026_09_19.md"
REPORT = ROOT / "reports" / "dfc_report.json"
RCR_PREDS = ROOT / "reports" / "rcr_test_predictions.parquet"

DFC_FEATS = ["dfc_ades_dep_1h", "dfc_ades_dep_3h", "dfc_ades_dep_1h_dev",
             "dfc_ades_in_set", "dfc_ades_arr_1h", "dfc_ades_arr_1h_dev"]
#: the ONLY columns the builder reads — leak gates 1 and 3 are enforced by this list
INPUT_COLS = ["ADES_mvt", "doy", "hr", "tmin", "mvt_sec"]
SCORED_AIRPORTS = ("LFPG", "LTFM", "LIRF", "EGLL", "EHAM", "EDDF", "EDDM", "LEMD", "LEBL", "LSZH")
W1, W3 = 3600.0, 10800.0
GAIN_BAR_S = 2.0
EXPECT_N_TRAIN, EXPECT_N_TEST, EXPECT_FIRST_TEST_DAY = 1_403_707, 127_296, 306
#: arbitrary fixed reference year for `timestamp()`'s absolute clock (F1 fix) — only relative
#: offsets are ever used, so the choice of epoch does not matter as long as it is shared.
EPOCH_YEAR = 2020


@dataclass(frozen=True)
class ArrivalStream:
    """Landing times for a SINGLE declared calendar year, on `timestamp()`'s absolute scale.

    `by_airport` maps airport -> sorted seconds-since-EPOCH_YEAR array. `years` is the set of
    calendar years actually found in the source files (used to fail loud if a caller's declared
    `year` does not match what the files contain). `span` is (min, max) landing time across every
    airport, used to fail loud if a departure frame's window reaches outside what the arrival
    files actually observed.
    """
    by_airport: dict
    years: frozenset
    span: tuple


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod


def timestamp(f, year) -> np.ndarray:
    """Absolute seconds since EPOCH_YEAR-01-01T00:00:00Z (F1 fix).

    `year` is the single calendar year every row in `f` belongs to. It is a REQUIRED, explicit
    argument — never a default and never inferred — so a caller cannot silently omit it and
    reproduce the defect this replaces: a bare `doy * 86400` origin let a 2026 departure and a
    2025 landing at the same day-of-year occupy the same coordinate (F1,
    reports/audit_239_2026_09_22/DFC_serve_construction_audit.md). A frame spanning more than one
    calendar year is out of scope for a single call — split it by year and call `build_dfc`/
    `timestamp` once per year.

    stand_ab stores tmin as minutes SINCE MIDNIGHT (hour * 60 + minute), and mvt_sec as seconds
    within the minute. Adding hr again shifts the clock by 0-23 hours (the separate, already-fixed
    double-count defect; see `stand_ab.py:202` and the docstring this replaced).
    """
    y = float(year)
    return ((y - EPOCH_YEAR) * 366.0 * 86400.0 + f.doy.to_numpy(float) * 86400.0
            + f.tmin.to_numpy(float) * 60.0 + f.mvt_sec.to_numpy(float))


def _window_counts(t: np.ndarray, w: float, exclude_self: bool) -> np.ndarray:
    """For each t_i, #{j : |t_j - t_i| <= w}, computed per already-grouped array."""
    order = np.argsort(t, kind="mergesort")
    ts = t[order]
    lo = np.searchsorted(ts, ts - w, side="left")
    hi = np.searchsorted(ts, ts + w, side="right")
    c = (hi - lo - (1 if exclude_self else 0)).astype(float)
    out = np.empty_like(c); out[order] = c
    return out


@dataclass(frozen=True)
class DevTable:
    """The `_dev` reference for ONE count column: `ades_hr` = (ADES, hr) -> fit-rows-only median (MultiIndex
    Series), `hr` = hr -> fit-rows-only median (the F2 fallback for a key absent from `ades_hr`)."""
    ades_hr: pd.Series
    hr: pd.Series


def fit_dev_table(key: pd.DataFrame, v: np.ndarray, fit_mask) -> DevTable:
    """The reference medians of `v` over the `fit_mask` rows only (leak gate 2). `key` has columns a (ADES), h (hr)."""
    fit = key.assign(v=v)[np.asarray(fit_mask, bool)]
    return DevTable(ades_hr=fit.groupby(["a", "h"]).v.median(), hr=fit.groupby("h").v.median())


def apply_dev_table(key: pd.DataFrame, v: np.ndarray, table: DevTable) -> np.ndarray:
    """`v` minus its reference median from `table` (F2 fix).

    Contract: an (ADES, hr) key absent from the table (an ADES unseen in training, or an hour
    never observed there for a seen ADES) falls back to the hr-only median — never to NaN. The
    SAME two-level lookup runs unconditionally, whether the table was fitted on this frame
    (training, `build_dfc(fit_mask=...)`) or on another year's training rows (serve,
    `build_dfc(dev_tables=...)`), so training and serve share one code path and one distribution:
    there is no "serve mode" branch to fall out of sync. This is why training measures 0.00% NaN
    (every fit row's own key is in the table it is compared against) while F2 showed 0.42% NaN at
    serve under the old one-level, fallback-free lookup. NaN can still occur only if `h` itself has
    zero fit rows — impossible for any frame covering the full 0-23 hour range that `fit_mask`
    selects from at least one row.
    """
    mapped = pd.MultiIndex.from_arrays([key.a, key.h]).map(table.ades_hr).to_numpy(dtype=float)
    fallback = key.h.map(table.hr).to_numpy(dtype=float)
    return v - np.where(np.isnan(mapped), fallback, mapped)


def _dev_column(key: pd.DataFrame, v: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Training path: the table is fitted on this frame's `fit_mask` rows, then applied to every row."""
    return apply_dev_table(key, v, fit_dev_table(key, v, fit_mask))


#: the count column each `_dev` column is derived from
DEV_SOURCES = (("dfc_ades_dep_1h", "dfc_ades_dep_1h_dev"), ("dfc_ades_arr_1h", "dfc_ades_arr_1h_dev"))


def _counts(frame, year, arrivals: "ArrivalStream | None") -> tuple:
    """The four count/flag columns and the (a, h) key — the stage shared by training and serve."""
    f = frame
    t = timestamp(f, year)
    ades = f.ADES_mvt.astype(str).to_numpy()
    n = len(f)
    dep1 = np.zeros(n); dep3 = np.zeros(n); arr1 = np.full(n, np.nan)
    in_set = np.isin(ades, SCORED_AIRPORTS).astype(float)

    if arrivals is not None:
        if arrivals.years != frozenset({int(year)}):
            raise AssertionError(
                f"F1 fail-loud: arrival stream covers year(s) {sorted(arrivals.years)} but the "
                f"departure frame is year {int(year)} — arrivals must be sourced from the same "
                "year as the departures they are matched against")
        obs = in_set == 1
        if obs.any() and not np.isnan(arrivals.span[0]):
            lo, hi = arrivals.span
            dep_lo, dep_hi = float(t[obs].min()) - W3, float(t[obs].max()) + W3
            # one day of slack absorbs legitimate source-file-boundary truncation (named, not
            # material: audit_239_2026_09_22/DFC_serve_construction_audit.md §5.3/F4)
            if dep_lo < lo - 86400.0 or dep_hi > hi + 86400.0:
                raise AssertionError(
                    "F1 fail-loud: an in-set departure's arrival window falls outside the "
                    "arrival stream's observed date range")

    for ap in pd.unique(ades):
        m = ades == ap
        dep1[m] = _window_counts(t[m], W1, exclude_self=True)
        dep3[m] = _window_counts(t[m], W3, exclude_self=True)
        if arrivals is not None and ap in arrivals.by_airport:
            a = arrivals.by_airport[ap]
            arr1[m] = (np.searchsorted(a, t[m] + W1, side="right")
                       - np.searchsorted(a, t[m] - W1, side="left")).astype(float)
    arr1[in_set == 0] = np.nan          # unobservable is NaN, never 0

    out = pd.DataFrame({"dfc_ades_dep_1h": dep1, "dfc_ades_dep_3h": dep3,
                        "dfc_ades_in_set": in_set, "dfc_ades_arr_1h": arr1})
    key = pd.DataFrame({"a": ades, "h": f.hr.to_numpy()})
    return out, key


def fit_dev_tables(frame, fit_mask, year, arrivals: "ArrivalStream | None" = None) -> dict:
    """The `_dev` reference tables fitted on `frame`'s `fit_mask` rows (leak gate 2): {count column: DevTable}.
    The serve path hands these to `build_dfc(..., dev_tables=...)` on a frame of ANOTHER year (T7 provenance)."""
    out, key = _counts(frame, year, arrivals)
    return {src: fit_dev_table(key, out[src].to_numpy(float), fit_mask) for src, _ in DEV_SOURCES}


def build_dfc(frame, fit_mask, year, arrivals: "ArrivalStream | None" = None, dev_tables: dict | None = None) -> pd.DataFrame:
    """The 6 registered columns. EXACTLY ONE of `fit_mask` (training: the rows the `_dev` medians may be
    computed on, leak gate 2) or `dev_tables` (serve: `fit_dev_tables` of the training frame) is given.
    `year` is the single calendar year of every row in `frame` (see `timestamp`). `arrivals` is an
    `ArrivalStream` sourced from the SAME year — passing one built from a different year raises rather
    than silently mismatching landings to takeoffs (F1 fix)."""
    if (fit_mask is None) == (dev_tables is None):
        raise ValueError("build_dfc takes exactly one of fit_mask or dev_tables")
    out, key = _counts(frame, year, arrivals)
    for src, dst in DEV_SOURCES:
        v = out[src].to_numpy(float)
        out[dst] = (_dev_column(key, v, fit_mask) if dev_tables is None else apply_dev_table(key, v, dev_tables[src]))
    return out[DFC_FEATS]


def load_arrivals(paths, year) -> ArrivalStream:
    """Landing times per arrival airport, from ARR rows in `paths`, on `timestamp()`'s absolute
    seconds-since-EPOCH_YEAR scale (F1 fix).

    `paths` is an explicit, caller-supplied list of parquet files — there is no default and no
    hidden glob, so the caller must state which files (2025 training months, or a 2026
    `ranking.parquet`) the departures it will match against actually come from. `year` is the
    calendar year those departures belong to; every row read from `paths` is asserted to be that
    year (fail loud if a source file's dates do not match what the caller declared)."""
    out = {}
    years_seen = set()
    for p in paths:
        d = pd.read_parquet(p, columns=["PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt"])
        d = d[d.PHASE_mvt.astype(str).str.upper() == "ARR"]
        ts = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True)
        years_seen |= set(int(y) for y in ts.dt.year.unique())
        sec = ((ts.dt.year.to_numpy(float) - EPOCH_YEAR) * 366.0 * 86400.0
               + ts.dt.dayofyear.to_numpy(float) * 86400.0 + ts.dt.hour.to_numpy(float) * 3600.0
               + ts.dt.minute.to_numpy(float) * 60.0 + ts.dt.second.to_numpy(float))
        for ap, g in pd.DataFrame({"ap": d.ADES_mvt.astype(str).to_numpy(), "s": sec}).groupby("ap"):
            out.setdefault(ap, []).append(g.s.to_numpy())
    if years_seen and years_seen != {int(year)}:
        raise AssertionError(
            f"F1 fail-loud: arrival source files contain year(s) {sorted(years_seen)}, not the "
            f"declared year {int(year)} — {list(paths)!r}")
    by_airport = {ap: np.sort(np.concatenate(v)) for ap, v in out.items()}
    all_sec = np.concatenate(list(by_airport.values())) if by_airport else np.array([], dtype=float)
    span = (float(all_sec.min()), float(all_sec.max())) if len(all_sec) else (np.nan, np.nan)
    return ArrivalStream(by_airport=by_airport, years=frozenset(years_seen), span=span)


def evaluate(gain: float, ci) -> dict:
    ex = ci[0] > 0 or ci[1] < 0
    ok = gain >= GAIN_BAR_S and ex and ci[0] > 0
    return {"verdict": "WORKING" if ok else
            ("NOT WORKING (ambiguous: real but below the registered bar)" if 0 < gain < GAIN_BAR_S and ex
             else "NOT WORKING"),
            "gain_s": gain, "ci95": list(ci), "ci_excludes_zero": bool(ex)}


def run(smoke: bool = False) -> None:
    started = time.monotonic()
    if not smoke and REPORT.exists():
        raise FileExistsError("DFC result already exists")
    import lightgbm as lgb
    R = _load("rcr_build", "scripts/analysis/rcr_build.py")
    S = _load("stand_ab", "scripts/stand_ab.py")
    LS = _load("lgbm_submit", "scripts/lgbm_submit.py")
    LF = _load("lgbm_fold", "scripts/lgbm_fold.py")
    CHR = _load("chr_build", "scripts/analysis/chr_build.py")

    base_feats = R.arm_feats("A0", S)
    banned = set(R.BANNED_IN_FEATURES)
    if banned & (set(base_feats) | set(DFC_FEATS) | set(INPUT_COLS)):
        raise AssertionError("leak gate 1/3: label column in features or inputs")

    months = [m for m in range(1, 13) if m not in R.HOLDOUT_MONTHS]
    frames, _ = LF._load_months(S.CACHE, months, base_feats, S.QCACHE, S.DCACHE, S.OCACHE, S.WCACHE)
    d = pd.concat(frames, ignore_index=True); del frames
    d["real_day"] = d.doy.astype(int)
    d = d[np.isfinite(d.delta.to_numpy(float))].reset_index(drop=True)
    if smoke:
        d = d.iloc[::40].reset_index(drop=True)

    rng = np.random.default_rng(R.SEED)
    pool = np.array(sorted(d.loc[d.month.isin(R.TEST_MONTHS), "real_day"].unique()), dtype=int)
    test_days = set(rng.choice(pool, size=max(1, int(round(R.TEST_DAY_FRAC * len(pool)))), replace=False))
    day = d.real_day.to_numpy()
    is_test = np.isin(day, list(test_days))
    tr_all = R.chrono_train_mask(day, is_test)
    first_test_day = int(day[is_test].min())

    saved = pd.read_parquet(RCR_PREDS)
    if not smoke:
        assert int(tr_all.sum()) == EXPECT_N_TRAIN and int(is_test.sum()) == EXPECT_N_TEST
        assert first_test_day == EXPECT_FIRST_TEST_DAY and day[tr_all].max() < first_test_day
        assert np.array_equal(day[is_test], saved.real_day.to_numpy())
        assert np.array_equal(d.y.to_numpy(float)[is_test], saved.y.to_numpy(float))

    y_all = d.y.to_numpy(float); delta_all = d.delta.to_numpy(float); proxy = d.proxy.to_numpy(float)
    r = np.random.default_rng(R.SEED + 7)
    trd = np.array(sorted(set(day[tr_all])))
    es_days = set(r.choice(trd, size=max(1, int(round(R.ES_DAY_FRAC * len(trd)))), replace=False))
    es = tr_all & np.isin(day, list(es_days)); fit = tr_all & ~es
    assert not (es & is_test).any() and not (fit & is_test).any()

    arrival_paths = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))
    arrivals = load_arrivals(arrival_paths, year=2025)
    dfc = build_dfc(d[INPUT_COLS], fit_mask=fit, year=2025, arrivals=arrivals)
    assert np.isnan(dfc.dfc_ades_arr_1h.to_numpy()[dfc.dfc_ades_in_set.to_numpy() == 0]).all()
    print(json.dumps({"stage": "dfc", "cols": DFC_FEATS,
                      "in_set_share": float(dfc.dfc_ades_in_set.mean()),
                      "dep_1h_mean": float(dfc.dfc_ades_dep_1h.mean()),
                      "arr_1h_mean_where_obs": float(np.nanmean(dfc.dfc_ades_arr_1h))}), flush=True)

    enc = S.infold_encodings(d[S.ENC_KEYS], y_all, delta_all, fit)
    frame = d.copy()
    for c, v in (enc.items() if isinstance(enc, dict) else zip(S.ENC, np.asarray(enc).T)):
        frame[c] = v
    for c in DFC_FEATS:
        frame[c] = dfc[c].to_numpy()
    feats = base_feats + DFC_FEATS
    X = LS.design_matrix(frame, feats)
    p = dict(LS.P); p.update(learning_rate=R.LR, num_threads=4, verbosity=-1)
    ds = lgb.Dataset(X[fit], label=delta_all[fit], free_raw_data=True)
    dv = lgb.Dataset(X[es], label=delta_all[es], reference=ds, free_raw_data=True)
    m = lgb.train(p, ds, 60 if smoke else R.MAX_ROUNDS, valid_sets=[dv],
                  callbacks=[lgb.early_stopping(R.PATIENCE, verbose=False)])
    raw = m.predict(X[is_test], num_iteration=m.best_iteration)
    if not np.all(np.isfinite(raw)):
        raise RuntimeError("A6: non-finite predictions")
    yhat = np.maximum(proxy[is_test] - raw, 1.0)
    if smoke:
        print(json.dumps({"smoke": "passed; scores not inspected"}), flush=True); return

    a0 = saved.yhat_A0.to_numpy(float); yt = y_all[is_test]
    ev = evaluate(R.rmse(yt, a0) - R.rmse(yt, yhat),
                  CHR.day_block_ci(yt, day[is_test], a0, yhat))
    imp = dict(zip(feats, m.feature_importance("gain")))
    ins = dfc.dfc_ades_in_set.to_numpy()[is_test] == 1
    out = {"tag": "DFC", **ev, "a6_rmse": R.rmse(yt, yhat), "a0_rmse_saved": R.rmse(yt, a0),
           "best_iter": int(m.best_iteration), "n_feats": len(feats),
           "dfc_gain_importance": {c: float(imp.get(c, 0.0)) for c in DFC_FEATS},
           "dfc_share_of_total_gain_importance": float(sum(imp.get(c, 0.0) for c in DFC_FEATS) / max(sum(imp.values()), 1)),
           "gain_on_in_set_rows_only": float(R.rmse(yt[ins], a0[ins]) - R.rmse(yt[ins], yhat[ins])),
           "n_in_set_test_rows": int(ins.sum()),
           "scope": "2025-only chronological, lr 0.05. NOT a board gain. Proxy for downstream load, not CTOT.",
           "seconds": time.monotonic() - started,
           "peak_rss_GiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3,
           "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest()}
    with REPORT.open("x") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    print(json.dumps({k: out[k] for k in ("verdict", "gain_s", "ci95", "dfc_share_of_total_gain_importance",
                                          "gain_on_in_set_rows_only")}, indent=2), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("mode", choices=["run"]); a.add_argument("--smoke", action="store_true")
    run(a.parse_args().smoke)
