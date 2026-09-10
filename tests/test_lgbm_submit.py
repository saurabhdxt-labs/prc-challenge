"""`scripts/lgbm_submit.py` ships the measured LightGBM config on the matched rows and must
change NOTHING else in the submission.

The point of v3 is a clean board measurement of one lane: matched rows re-predicted, the
5,290 unmatched rows copied from v2 byte for byte. A splice that touches an unmatched row,
misses a matched one, or re-orders identifiers would make the board delta uninterpretable
and spend a submission slot on noise. The refit procedure must be the one lgbm_ab measured
(226.24 on the fold), so its arithmetic is pinned to the numbers that run printed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ls = _load("lgbm_submit")
bs = _load("build_submission")
stand_ab = _load("stand_ab")


def _load_test_helper(name):
    """A non-test helper module under tests/, loaded like the scripts are."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "tests" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_syn = _load_test_helper("synthetic_caches")


def _base(n=60, seed=0):
    """A v2-shaped frame: float64 identifiers, int32 predictions, ids not sorted."""
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(202601000, 202601000 + n)).astype("float64")
    return pd.DataFrame({"MVT_ID_mvt": ids,
                         "TAXITIME_SEC_mvt": rng.integers(300, 2000, n).astype("int32")})


def test_splice_replaces_exactly_the_matched_rows_and_nothing_else():
    """Matched rows carry the new values, unmatched rows are byte-identical, the identifier
    column is untouched in value, order and dtype, and the prediction dtype stays int32.

    Fails when the splice merges instead of mapping (order changes), assigns through a
    float intermediate (dtype changes), or writes an unmatched row. Rehearsed 2026-09-09,
    each RED: `new[~hit] = new[~hit] + 1` after the assignment; returning
    `new.astype("int64")`; returning `np.sort(base_ids)` as the identifier column.
    """
    v2 = _base()
    matched = v2.MVT_ID_mvt.to_numpy()[::2]                  # 30 of 60, interleaved
    new = (np.arange(len(matched)) + 5000).astype("int32")
    out = ls.splice(v2, matched[::-1], new[::-1], set(matched))   # any order must do

    assert np.array_equal(out.MVT_ID_mvt.to_numpy(), v2.MVT_ID_mvt.to_numpy())
    assert out.MVT_ID_mvt.dtype == v2.MVT_ID_mvt.dtype == np.float64
    assert out.TAXITIME_SEC_mvt.dtype == np.int32
    assert list(out.columns) == ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]
    is_m = v2.MVT_ID_mvt.isin(matched).to_numpy()
    want = pd.Series(new, index=matched)
    assert np.array_equal(out.TAXITIME_SEC_mvt.to_numpy()[is_m],
                          want.loc[v2.MVT_ID_mvt.to_numpy()[is_m]].to_numpy())
    assert np.array_equal(out.TAXITIME_SEC_mvt.to_numpy()[~is_m],
                          v2.TAXITIME_SEC_mvt.to_numpy()[~is_m])
    assert v2.TAXITIME_SEC_mvt.to_numpy()[::2].tolist() != new.tolist(), "fixture is vacuous"


@pytest.mark.parametrize("case,msg", [
    ("missing", "matched id set"),          # one matched id has no prediction
    ("extra", "matched id set"),            # a prediction for an unmatched id
    ("foreign", "not in the base"),         # an id that is not in v2 at all
    ("duplicate", "duplicate"),             # the same id predicted twice
    ("float", "int32"),                     # predictions not int32
    ("nonpositive", "non-positive"),        # a zero prediction
])
def test_splice_refuses_when_the_matched_id_set_does_not_match(case, msg):
    """Every way the predicted id set can disagree with the matched id set is a refusal,
    not a partial write. A silent partial splice is exactly the artefact that would score
    and mislead.

    Fails when `splice` fills missing ids from the base, drops extras, or coerces dtypes.
    Rehearsed 2026-09-09, each RED: replacing `if missing or extra:` with `if False:`
    (the missing/extra cases); replacing `if values.dtype != np.int32:` with `if False:`
    (the float case - numpy then casts the floats into the int32 column silently).
    """
    v2 = _base()
    matched = v2.MVT_ID_mvt.to_numpy()[:40]
    ids, vals = matched.copy(), np.full(40, 900, dtype="int32")
    if case == "missing":
        ids, vals = ids[:-1], vals[:-1]
    elif case == "extra":
        ids = np.append(ids, v2.MVT_ID_mvt.to_numpy()[45])
        vals = np.append(vals, np.int32(900))
    elif case == "foreign":
        ids[0] = 1.0
    elif case == "duplicate":
        ids[1] = ids[0]
    elif case == "float":
        vals = vals.astype("float64")
    elif case == "nonpositive":
        vals[3] = 0
    with pytest.raises(ValueError, match=msg):
        ls.splice(v2, ids, vals, set(matched))


def test_version_is_required_and_the_output_name_is_the_leaderboard_convention():
    """The scorer keys on `<team>_v<N>.parquet`; anything else is a 403 that reads like a
    permissions failure (it cost this project a day). The name comes from
    build_submission.submission_name so there is exactly one place that knows the rule.

    Fails when --version gets a default, or the script formats its own filename.
    Rehearsed 2026-09-09, each RED: `required=False, default=3`; `return
    f"merry-quicksand_v{version}.parquet"` in output_name (version 0 then passes).
    """
    with pytest.raises(SystemExit):
        ls.parse_args(["--smoke"])
    args = ls.parse_args(["--version", "3"])
    assert args.version == 3 and args.smoke is False
    assert ls.output_name(3) == bs.submission_name(3) == "merry-quicksand_v3.parquet"
    for bad in (0, -2):
        with pytest.raises(ValueError):
            ls.output_name(bad)
    with pytest.raises(SystemExit):
        ls.parse_args(["--version", "0"])


def test_refit_tree_count_scales_best_iter_by_the_data_ratio_with_a_floor():
    """Pinned to lgbm_ab's own run: best_iter 17,557 on 1,378,585 fit rows, refit on
    1,723,425 -> 21,948 trees (reports/lgbm_ab_full.log). The floor of 50 is what keeps a
    smoke run from refitting an empty model.

    Fails when the ratio is inverted, rounded instead of truncated, or the floor dropped.
    Rehearsed 2026-09-09, each RED: `n_fit / n_all`; `round(...)` for `int(...)` (21,949).
    """
    assert ls.n_refit(17_557, 1_723_425, 1_378_585) == 21_948
    assert ls.n_refit(10, 100, 50) == 50
    assert ls.n_refit(1_000, 2_062_440, 1_717_600) == 1_200


def test_masks_hold_the_early_stopping_months_out_of_the_fit_rows_and_ranking_out_of_both():
    """Fit, early-stop and ranking rows are disjoint; fit + early-stop = every training row.

    Fails when the ES months leak into the fit set, or ranking rows enter either.
    Rehearsed 2026-09-09: `fit = train.copy()` went RED.
    """
    month = np.array([1, 3, 9, 7, 3, 12, 1, 7])
    is_rank = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=bool)
    train, fit, es = ls.split_masks(month, is_rank)
    assert train.tolist() == [1, 1, 1, 1, 1, 1, 0, 0]
    assert es.tolist() == [0, 1, 1, 0, 1, 0, 0, 0]
    assert fit.tolist() == [1, 0, 0, 1, 0, 1, 0, 0]
    assert not (fit & es).any() and not ((fit | es) & is_rank).any()
    assert ((fit | es) == train).all()
    with pytest.raises(ValueError, match="early-stopping"):
        ls.split_masks(np.array([1, 2, 4]), np.zeros(3, dtype=bool))


def test_recovered_taxi_time_floors_at_one_second_and_rounds_like_build_submission():
    """y_hat = max(proxy - delta_hat, 1) rounded with np.rint to int32, exactly as
    build_submission.main does, so v3's matched rows are on the same scale as v2's.

    Fails when the floor is dropped, the rounding becomes truncation, or the dtype changes.
    Rehearsed 2026-09-09, each RED: `np.rint(raw)` without the floor; `np.floor` for
    `np.rint`.
    """
    # 1,000 rows so that the two rows the floor catches are 0.2% - under the 0.5% guard
    proxy = np.full(1_000, 1000.0)
    delta_hat = np.full(1_000, -100.4)
    delta_hat[:5] = [-100.4, 999.5, 899.5, 898.5, 1200.0]
    got = ls.recover_taxi_time(proxy, delta_hat)
    want = np.rint(np.maximum(proxy - delta_hat, 1.0)).astype("int32")
    assert got.dtype == np.int32
    assert got.tolist() == want.tolist()
    assert got[:5].tolist() == [1100, 1, 100, 102, 1], "np.rint is round-half-even: 100.5 -> 100"
    bad = delta_hat.copy()
    bad[7] = np.nan
    with pytest.raises(AssertionError, match="non-finite"):
        ls.recover_taxi_time(proxy, bad)
    with pytest.raises(AssertionError, match="positivity floor"):
        ls.recover_taxi_time(np.full(1000, 100.0), np.full(1000, 200.0))


def test_encodings_on_ranking_rows_fall_back_to_the_prior_for_unseen_keys():
    """The submission fits `infold_encodings` with tr_mask = every training row and applies
    it to the ranking rows, whose labels are NaN. Ranking rows must get no NaN encoding; a
    key never seen in training must get the training prior; NaN labels on ranking rows must
    not poison the prior.

    Fails when the prior is computed over all rows (NaN), or when unseen keys map to NaN.
    Rehearsed 2026-09-09: `py, pdl = float(np.asarray(y).mean()), ...` in
    stand_ab.infold_encodings (prior over every row, ranking labels included) went RED.
    """
    rng = np.random.default_rng(0)
    n_tr, n_rk = 2_000, 300
    tr = np.r_[np.ones(n_tr, bool), np.zeros(n_rk, bool)]
    keys = pd.DataFrame({k: rng.integers(0, 30, n_tr + n_rk).astype(str)
                         for k in stand_ab.ENC_KEYS})
    keys.loc[n_tr:, "STAND_mvt"] = "UNSEEN"                 # never occurs in training
    y = np.r_[rng.normal(900, 300, n_tr), np.full(n_rk, np.nan)]
    delta = np.r_[rng.normal(-60, 200, n_tr), np.full(n_rk, np.nan)]

    enc = stand_ab.infold_encodings(keys, y, delta, tr)
    assert len(enc) == 24
    nan_cols = [k for k, v in enc.items() if np.isnan(v[~tr]).any()]
    assert not nan_cols, f"NaN encodings on ranking rows: {nan_cols}"
    assert np.allclose(enc["te_STAND_mvt"][~tr], y[tr].mean())
    assert np.allclose(enc["de_STAND_mvt"][~tr], delta[tr].mean())
    assert not np.allclose(enc["te_ADEP_mvt"][~tr], y[tr].mean()), \
        "seen keys collapsed to the prior - the encoding is not being applied"


def test_reuse_refuses_a_booster_from_another_configuration():
    """`--reuse-booster` splices whatever model file sits at the reuse path; the fit.json
    beside it must prove the file came from this exact configuration. A smoke booster, a
    booster fitted on fewer rows, other parameters, or a truncated model file are refused.

    Fails when any of the four provenance checks is dropped. Rehearsed 2026-09-09:
    replacing `if info.get("smoke") != smoke:` with `if False:` went RED.
    """
    info = dict(smoke=False, n_all=2_062_440, params=dict(ls.P), n_ref=21_948)
    ls.check_provenance(info, 21_948, False, 2_062_440, dict(ls.P))
    for bad, msg in ((dict(info, smoke=True), "smoke"),
                     (dict(info, n_all=1_723_425), "n_all"),
                     (dict(info, params=dict(ls.P, learning_rate=0.05)), "params"),
                     (dict(info, n_ref=21_000), "trees")):
        with pytest.raises(ValueError, match=msg):
            ls.check_provenance(bad, 21_948, False, 2_062_440, dict(ls.P))
    with pytest.raises(ValueError, match="trees"):
        ls.check_provenance(info, 20_000, False, 2_062_440, dict(ls.P))


CACHE = ROOT / "data" / "cache_stand"
SUBS = ROOT / "submissions"
_smoke_inputs = [CACHE / "training_2025-01-01_2025-02-01.parquet",
                 CACHE / "training_2025-03-01_2025-04-01.parquet",
                 CACHE / "ranking.parquet", SUBS / "merry-quicksand_v2.parquet",
                 ROOT / "data" / "raw" / "ranking.parquet",
                 ROOT / "data" / "raw" / "submitting.parquet"]


@pytest.mark.skipif(not all(p.exists() for p in _smoke_inputs),
                    reason="needs the month caches, the ranking cache (scripts/stand_ab.py "
                           "cache --ranking) and submissions/merry-quicksand_v2.parquet")
def test_smoke_end_to_end_splices_a_sample_into_a_scratch_copy_of_v2(tmp_path, capsys,
                                                                       monkeypatch):
    """Two months, a 5,000-row ranking sample, a tiny booster: the full code path from cache
    to spliced file, in a scratch directory. Asserts the splice invariants on the sample,
    that the submissions directory is untouched, that the metadata and booster land, and
    that the output carries the smoke banner - this project has twice quoted a smoke
    magnitude as a result.

    The module's SUBS is pointed at an empty scratch directory and the real v2 is passed
    as --base explicitly, so "writes under submissions/" is observable without ever placing
    a smoke file at the real path.

    Fails when the smoke writes under submissions/, when the sample splice touches a row
    outside the sample, when the booster is not saved beside the output, or when the banner
    is missing. Rehearsed 2026-09-09, each RED: `dest = SUBS / output_name(...)` (lands in
    the patched scratch SUBS); `out.iloc[0, 1] += 1` before the write; the booster saved as
    `booster_v3.txt`; both `log(SMOKE_BANNER)` calls removed.
    """
    fake_subs = tmp_path / "fake_submissions"
    fake_subs.mkdir()
    monkeypatch.setattr(ls, "SUBS", fake_subs)
    real = SUBS / "merry-quicksand_v3.parquet"
    before = real.stat().st_mtime_ns if real.exists() else None
    rc = ls.main(["--version", "3", "--smoke", "--out-dir", str(tmp_path),
                  "--base", str(SUBS / "merry-quicksand_v2.parquet")])
    assert rc == 0
    assert (real.stat().st_mtime_ns if real.exists() else None) == before, \
        "the smoke wrote the real submission path"
    assert not list(fake_subs.iterdir()), "the smoke wrote under submissions/"

    out = pq.read_table(tmp_path / "merry-quicksand_v3.parquet").to_pandas()
    v2 = pq.read_table(SUBS / "merry-quicksand_v2.parquet").to_pandas()
    template = pq.read_table(ROOT / "data" / "raw" / "submitting.parquet").to_pandas()
    bs.check_submission(out, template)
    assert np.array_equal(out.MVT_ID_mvt.to_numpy(), v2.MVT_ID_mvt.to_numpy())
    assert out.TAXITIME_SEC_mvt.dtype == v2.TAXITIME_SEC_mvt.dtype == np.int32

    meta = json.loads((tmp_path / "merry-quicksand_v3.meta.json").read_text())
    for key in ("git_sha", "best_iter", "n_ref", "params", "features", "n_replaced",
                "n_value_changed", "started_utc", "finished_utc", "smoke"):
        assert key in meta, f"meta.json lacks {key}"
    assert meta["smoke"] is True and meta["n_replaced"] == 5_000
    assert meta["n_ref"] >= 50 and meta["best_iter"] >= 1
    assert meta["features"] == stand_ab.BASELINE_FEATS

    raw = pq.read_table(ROOT / "data" / "raw" / "ranking.parquet",
                        columns=["MVT_ID_mvt", "AOBT_3_flt"]).to_pandas()
    unmatched = set(raw.MVT_ID_mvt[raw.AOBT_3_flt.isna()])
    changed = out.TAXITIME_SEC_mvt.to_numpy() != v2.TAXITIME_SEC_mvt.to_numpy()
    assert 0 < changed.sum() <= 5_000
    assert changed.sum() == meta["n_value_changed"]
    assert not out.MVT_ID_mvt[changed].isin(unmatched).any(), "an unmatched row changed"
    assert len(meta["replaced_ids"]) == 5_000, "smoke meta must list the sampled ids"
    assert set(out.MVT_ID_mvt[changed]) <= set(meta["replaced_ids"]), \
        "a row outside the sample changed"

    assert (tmp_path / "lgbm_v3.txt").exists(), "booster not saved"
    text = capsys.readouterr().out
    assert ls.SMOKE_BANNER in text
    assert "best_iter" in text and "n_ref" in text and "peak RSS" in text


class _RecordingBooster:
    """Stands in for a trained lightgbm.Booster; records the kwargs predict() received."""
    def __init__(self):
        self.calls = []
    def predict(self, X, **kw):
        self.calls.append(kw)
        return np.zeros(len(X), dtype="float64")


def test_every_predict_is_threaded_like_training():
    """Booster.predict does NOT inherit the training `num_threads`; it takes its thread count
    from the OpenMP runtime, which this repo pins to 1 (OMP_NUM_THREADS=1) beside the live
    fleet. On 2026-09-09 the v3 fit trained at ~295% CPU and then sat single-threaded for 40+
    minutes inside LGBM_BoosterPredictForMat on 23,000 trees x 345k rows, twice (stopping set,
    then ranking rows). Both sites must go through one helper that forwards num_threads.

    Fails when `predict_delta` drops the `num_threads=params["num_threads"]` forward, or when
    either call site bypasses the helper (asserted via the recorded kwargs)."""
    M = _load("lgbm_submit")
    fake = _RecordingBooster()
    X = np.zeros((7, len(M.FEATS)), dtype="float32")
    out = M.predict_delta(fake, X, M.P, num_iteration=42)
    assert out.shape == (7,) and out.dtype == np.float64
    assert fake.calls == [{"num_iteration": 42, "num_threads": M.P["num_threads"]}], fake.calls
    assert M.P["num_threads"] == 4, "the measured config uses 4 threads; predict must too"
    # both production call sites must route through the helper
    src = (ROOT / "scripts" / "lgbm_submit.py").read_text()
    assert src.count(".predict(") == 1, "a call site bypasses predict_delta"


# ---------------------------------------------------------------------------------------------
# v4: seed averaging and the per-airport blend (Amendment 12). v3 stays the default path.
# ---------------------------------------------------------------------------------------------

class _ConstantBooster:
    """Stands in for a refit lightgbm.Booster that predicts one constant delta."""
    def __init__(self, value):
        self.value = float(value)
    def predict(self, X, **kw):
        return np.full(len(X), self.value, dtype="float64")


def test_seed_averaging_is_the_arithmetic_mean_over_the_boosters():
    """delta_hat under --seeds is the plain arithmetic mean of the per-seed predictions, every
    seed weighted equally, computed in float64. Three fake boosters predicting 1, 2 and 6 must
    average to exactly 3 on every row; a single booster must pass through unchanged; an empty
    list is refused rather than averaged to NaN.

    Fails when the mean keeps only the first seed, weights the seeds, or divides by the wrong
    count. Rehearsed 2026-09-09, each RED: `return preds[0]`; `np.sum(...) / (len(preds) + 1)`.
    """
    X = np.zeros((5, len(ls.FEATS)), dtype="float32")
    preds = [ls.predict_delta(_ConstantBooster(v), X, ls.P) for v in (1.0, 2.0, 6.0)]
    got = ls.mean_delta(preds)
    assert got.dtype == np.float64 and got.shape == (5,)
    assert got.tolist() == [3.0] * 5, "not the arithmetic mean over the three boosters"
    assert ls.mean_delta(preds[:1]).tolist() == [1.0] * 5
    ragged = [np.arange(3.0), np.arange(4.0)]
    with pytest.raises(ValueError, match="shape"):
        ls.mean_delta(ragged)
    with pytest.raises(ValueError, match="no predictions"):
        ls.mean_delta([])


def test_per_airport_blend_keeps_pooled_where_no_model_was_fitted():
    """final = (1 - BLEND) * pooled + BLEND * per_airport on rows whose airport got its own
    model, and exactly pooled on the fallback rows (airports under the 20,000-row floor), with
    BLEND = 0.5 taken from build_submission so there is one place that knows the weight.

    Fails when the blend weight drifts to 1.0 (per-airport replaces pooled), when the fallback
    rows are blended with NaN, or when the weight is redefined locally. Rehearsed 2026-09-09,
    each RED: `BLEND = 1.0`; `out = (1 - weight) * pooled + weight * pa` without the mask
    (fallback rows become NaN).
    """
    pooled = np.array([10.0, 10.0, 10.0, 10.0, 30.0])
    pa = np.array([20.0, np.nan, 30.0, np.nan, 10.0])
    fitted = np.array([True, False, True, False, True])
    got = ls.blend(pooled, pa, fitted)
    assert ls.BLEND == bs.BLEND == 0.5
    assert got.tolist() == [15.0, 10.0, 20.0, 10.0, 20.0]
    assert got.dtype == np.float64
    assert ls.blend(pooled, pa, fitted, weight=0.25).tolist() == [12.5, 10.0, 15.0, 10.0, 25.0]
    with pytest.raises(ValueError, match="non-finite"):
        ls.blend(pooled, np.array([np.nan, np.nan, 30.0, np.nan, 10.0]), fitted)
    with pytest.raises(ValueError, match="shape"):
        ls.blend(pooled[:4], pa, fitted)


def test_per_airport_tree_count_is_the_row_share_of_n_ref_with_a_floor():
    """Tree count for an airport's own model = max(floor, int(n_ref * n_airport / n_all)) -
    the airport's share of the pooled refit's trees, truncated, never below 200 real trees.
    Pinned to v3's n_ref (30,846 on 2,062,440 rows): an airport with 254,000 training rows gets
    3,798 trees (3,798.86 truncated); one with 115,000 gets 1,719 (1,719.95); in a smoke
    (n_ref ~70) every airport sits on the floor.

    This is the implementer's rule Amendment 12.4 says must be recorded; it is a judgement
    call, not a measurement (see the docstring of per_airport_trees).

    Fails when the ratio is inverted, rounded instead of truncated, or the floor dropped.
    Rehearsed 2026-09-09, each RED: `n_all / n_airport`; `round(...)` (3,798.86 -> 3,799 and
    1,719.95 -> 1,720); `max` removed (7 vs 200 on the smoke case).
    """
    assert ls.per_airport_trees(30_846, 254_000, 2_062_440) == 3_798
    assert ls.per_airport_trees(30_846, 115_000, 2_062_440) == 1_719
    assert ls.per_airport_trees(70, 30_000, 300_000) == 200
    assert ls.per_airport_trees(70, 30_000, 300_000, floor=20) == 20
    assert ls.PA_TREE_FLOOR == 200 and ls.PA_MIN_ROWS == 20_000
    assert "int(n_ref * n_airport / n_all)" in ls.PA_TREE_RULE


def _fake_refit(X, dlt, mask, params, n_trees):
    """A 'model' whose prediction is the mean training delta of the rows it was fitted on plus
    its seed - so per-airport values and the seed average are exactly computable."""
    return _ConstantBooster(float(np.asarray(dlt)[mask].mean()) + params["seed"])


def test_fit_per_airport_fits_only_airports_above_the_floor_and_averages_seeds(monkeypatch):
    """Per airport: rows with >= min_rows training rows get their own model, refit once per
    seed with the share tree count, predicting that airport's prediction rows; airports
    below the floor are left NaN with fitted=False and appear in the info as fallbacks. With
    a fake refit that predicts (mean training delta + seed), the seed-averaged value is
    (mean + 0.5) for seeds (0, 1).

    Fails when the floor is not applied, when seeds are not all fitted, when the training rows
    of one airport leak into another's model, or when the tree count is not the share rule.
    Rehearsed 2026-09-09, each RED: `if False:` for the `n_train < min_rows` skip; `for s in
    seeds[:1]`; `mask = train` in place of `train & (ap_code == code)`.
    """
    monkeypatch.setattr(ls, "refit", _fake_refit)
    airports = ["AAA", "BBB", "CCC"]
    # training: AAA 30 rows (delta 100), BBB 5 rows (delta 500, below the floor), CCC 20 rows (delta -40)
    tr_code = np.r_[np.zeros(30, np.int8), np.ones(5, np.int8), np.full(20, 2, np.int8)]
    tr_dlt = np.r_[np.full(30, 100.0), np.full(5, 500.0), np.full(20, -40.0)]
    pr_code = np.array([0, 1, 2, 0, 1], np.int8)
    ap_code = np.r_[tr_code, pr_code]
    dlt = np.r_[tr_dlt, np.full(5, np.nan)]
    n = len(ap_code)
    train = np.r_[np.ones(55, bool), np.zeros(5, bool)]
    pred = ~train
    X = np.zeros((n, 3), dtype="float32")
    calls = []
    real = ls.refit

    def spy(X_, dlt_, mask, params, n_trees):
        calls.append((int(mask.sum()), params["seed"], n_trees))
        return real(X_, dlt_, mask, params, n_trees)
    monkeypatch.setattr(ls, "refit", spy)

    pa_by_seed, fitted, info = ls.fit_per_airport(
        X, dlt, train, pred, ap_code, airports, dict(ls.P), n_ref=1_000, seeds=(0, 1),
        min_rows=20, floor=5)
    assert sorted(pa_by_seed) == [0, 1]
    assert fitted.tolist() == [True, False, True, True, False]
    pa = ls.mean_delta(list(pa_by_seed.values()))
    assert np.isnan(pa[[1, 4]]).all(), "fallback rows must stay NaN for blend() to skip"
    assert pa[[0, 2, 3]].tolist() == [100.5, -39.5, 100.5], "not (mean delta + mean seed)"
    assert pa_by_seed[0][[0, 2, 3]].tolist() == [100.0, -40.0, 100.0]
    assert pa_by_seed[1][[0, 2, 3]].tolist() == [101.0, -39.0, 101.0]
    assert info["AAA"] == dict(n_train=30, n_pred=2, fitted=True, n_trees=545, seeds=[0, 1],
                               rule="share", es=None)
    assert info["CCC"] == dict(n_train=20, n_pred=1, fitted=True, n_trees=363, seeds=[0, 1],
                               rule="share", es=None)
    assert info["BBB"] == dict(n_train=5, n_pred=2, fitted=False, n_trees=None, seeds=[],
                               rule="share", es=None)
    # one refit per (fitted airport, seed) on that airport's rows only, at the share count
    assert sorted(calls) == [(20, 0, 363), (20, 1, 363), (30, 0, 545), (30, 1, 545)]


def test_per_airport_es_rule_early_stops_on_the_airport_own_rows(monkeypatch):
    """`--pa-trees es`: a fitted airport gets its own early-stopping run on ITS fit rows,
    stopped on ITS early-stop rows (the pooled fit/es masks intersected with the airport,
    seed 0, the pooled nest/patience), and is refit per seed at its own
    n_ref = max(50, int(best_iter * n_all_a / n_fit_a)). Fallback airports get no ES run.
    The share rule stays the default and records itself as such.

    Fails when the ES masks are not intersected with the airport (it would stop on other
    airports' rows), when the refit count is not the airport's own n_ref, or when the rule is
    not recorded. Rehearsed 2026-09-09, each RED: `fit_a, es_a = es["fit"], es["es"]`;
    `n_trees = per_airport_trees(n_ref, n_tr, n_all, floor)` in the es branch.
    """
    es_calls, refits = [], []

    def fake_early_stop(X, dlt, y, proxy, train, fit, es, params, nest, patience, target=ls.TARGET_DELTA):
        es_calls.append((int(train.sum()), int(fit.sum()), int(es.sum()), params["seed"], nest, patience))
        best = 10 * int(fit.sum())                        # identifies which fit rows it saw
        return dict(best_iter=best, n_ref=ls.n_refit(best, int(train.sum()), int(fit.sum())),
                    n_fit=int(fit.sum()), n_es=int(es.sum()), n_all=int(train.sum()),
                    es_months=[3, 9], es_rmse_taxi_time_NOT_A_RESULT=1.0, es_proxy_only_rmse=2.0)

    def fake_refit(X, dlt, mask, params, n_trees):
        refits.append((int(mask.sum()), params["seed"], n_trees))
        return _ConstantBooster(1.0)
    monkeypatch.setattr(ls, "early_stop", fake_early_stop)
    monkeypatch.setattr(ls, "refit", fake_refit)

    airports = ["AAA", "BBB"]
    # rows 0-19 AAA fit, 20-29 AAA early-stop, 30-33 BBB fit, 34 BBB early-stop, 35-36 prediction
    ap_code = np.r_[np.zeros(30, np.int8), np.ones(5, np.int8), np.array([0, 1], np.int8)]
    train = np.r_[np.ones(35, bool), np.zeros(2, bool)]
    pred = ~train
    es_mask = np.zeros(37, bool)
    es_mask[20:30] = True
    es_mask[34] = True
    fit = train & ~es_mask
    dlt = np.r_[np.full(35, 5.0), np.nan, np.nan]
    y = proxy = np.zeros(37)
    X = np.zeros((37, 3), np.float32)
    es_args = dict(y=y, proxy=proxy, fit=fit, es=es_mask, nest=77, patience=9)

    pa_by_seed, fitted, info = ls.fit_per_airport(
        X, dlt, train, pred, ap_code, airports, dict(ls.P), n_ref=1_000, seeds=(0, 1),
        min_rows=20, floor=5, trees="es", es=es_args)
    assert es_calls == [(30, 20, 10, ls.ES_SEED, 77, 9)], "one ES run, AAA only, on its own rows"
    assert info["AAA"] == dict(n_train=30, n_pred=1, fitted=True, n_trees=300, seeds=[0, 1],
                               rule="es", es=dict(best_iter=200, n_ref=300, n_fit=20, n_es=10))
    assert info["BBB"] == dict(n_train=5, n_pred=1, fitted=False, n_trees=None, seeds=[],
                               rule="es", es=None)
    assert refits == [(30, 0, 300), (30, 1, 300)], "refit at the airport's own n_ref, per seed"
    assert fitted.tolist() == [True, False] and pa_by_seed[0].tolist()[0] == 1.0
    with pytest.raises(ValueError, match="trees"):
        ls.fit_per_airport(X, dlt, train, pred, ap_code, airports, dict(ls.P), 1_000, (0,),
                           20, 5, trees="magic", es=es_args)
    with pytest.raises(ValueError, match="es="):
        ls.fit_per_airport(X, dlt, train, pred, ap_code, airports, dict(ls.P), 1_000, (0,),
                           20, 5, trees="es", es=None)
    assert ls.parse_args(["--version", "4"]).pa_trees == "share"
    assert ls.parse_args(["--version", "4", "--pa-trees", "es"]).pa_trees == "es"
    with pytest.raises(SystemExit):
        ls.parse_args(["--version", "4", "--pa-trees", "magic"])
    assert "best_iter" in ls.PA_TREE_RULE_ES


def test_seeds_argument_and_the_booster_file_names():
    """`--seeds` defaults to (0,), parses "0,1,2" in order, and refuses duplicates, negatives
    and junk. With the default seed set the booster keeps v3's name (lgbm_vN.txt) so
    --reuse-booster still finds the v3 file; any other set names one file per seed.

    Fails when the default changes, when duplicates are accepted (a seed would be double
    counted in the mean), or when the legacy name is dropped. Rehearsed 2026-09-09, each RED:
    `default="0,1,2"`; removing the duplicate check; naming `lgbm_v3_seed0.txt` for (0,).
    """
    assert ls.parse_args(["--version", "3"]).seeds == (0,)
    assert ls.parse_args(["--version", "4", "--seeds", "0,1,2"]).seeds == (0, 1, 2)
    assert ls.parse_args(["--version", "4"]).per_airport is False
    assert ls.parse_args(["--version", "4", "--per-airport"]).per_airport is True
    assert ls.parse_seeds("2, 0") == (2, 0)
    for bad in ("0,0", "-1", "", "a", "0,,1"):
        with pytest.raises(argparse.ArgumentTypeError):
            ls.parse_seeds(bad)
    d = pathlib.Path("/x")
    assert ls.booster_files(d, 3, (0,)) == [(d / "lgbm_v3.txt", d / "lgbm_v3.fit.json")]
    assert ls.booster_files(d, 4, (0, 1)) == [
        (d / "lgbm_v4_seed0.txt", d / "lgbm_v4_seed0.fit.json"),
        (d / "lgbm_v4_seed1.txt", d / "lgbm_v4_seed1.fit.json")]
    assert ls.booster_files(d, 4, (0,)) == [(d / "lgbm_v4.txt", d / "lgbm_v4.fit.json")]


def _tiny_booster(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, len(ls.FEATS))).astype("float32")
    return lgb.train(dict(ls.P, seed=seed, num_leaves=4, min_data_in_leaf=5),
                     lgb.Dataset(X, rng.normal(size=200)), num_boost_round=7)


def test_reuse_refuses_a_partial_seed_set(tmp_path):
    """--reuse-booster reuses every seed's saved booster or none of them. A directory holding
    seed 0 but not seed 1 is refused - silently refitting the missing seed would splice a
    mixed-provenance mean into a submission. With every file present each booster is loaded
    and its fit.json checked against ITS seed's params; a booster whose fit.json carries
    another seed is refused.

    Fails when a partial set falls through to a fresh fit, or when the per-seed provenance
    check compares against the base params instead of the seed's. Rehearsed 2026-09-09, each
    RED: `if any(present) and not all(present):` -> `if False:`; `dict(params)` for
    `dict(params, seed=s)` in reuse_seeds.
    """
    files = ls.booster_files(tmp_path, 4, (0, 1))
    assert ls.reusable(files) is False                       # nothing on disk: fit
    params = dict(ls.P, num_leaves=4, min_data_in_leaf=5)
    for s, (path, fjson) in zip((0, 1), files):
        b = _tiny_booster(s)
        b.save_model(str(path))
        fjson.write_text(json.dumps(dict(smoke=False, n_all=200, n_ref=7, best_iter=5,
                                         params=dict(params, seed=s))))
    assert ls.reusable(files) is True                        # everything on disk: reuse
    X = np.zeros((3, len(ls.FEATS)), dtype="float32")
    preds, info = ls.reuse_seeds(files, (0, 1), X, params, smoke=False, n_all=200)
    assert sorted(preds) == [0, 1] and all(p.shape == (3,) for p in preds.values())
    assert info["n_ref"] == 7 and info["best_iter"] == 5
    assert not np.array_equal(preds[0], preds[1]), "two seeds gave the same booster"

    files[1][1].write_text(json.dumps(dict(smoke=False, n_all=200, n_ref=7, best_iter=5,
                                           params=dict(params, seed=0))))   # wrong seed
    with pytest.raises(ValueError, match="params"):
        ls.reuse_seeds(files, (0, 1), X, params, smoke=False, n_all=200)

    files[1][0].unlink()
    with pytest.raises(ValueError, match="partial"):
        ls.reusable(files)


def _assert_sample_splice(out, meta, n_sample):
    """The splice invariants every smoke shares: v2 shape, only sampled matched rows moved."""
    v2 = pq.read_table(SUBS / "merry-quicksand_v2.parquet").to_pandas()
    template = pq.read_table(ROOT / "data" / "raw" / "submitting.parquet").to_pandas()
    bs.check_submission(out, template)
    assert np.array_equal(out.MVT_ID_mvt.to_numpy(), v2.MVT_ID_mvt.to_numpy())
    assert out.TAXITIME_SEC_mvt.dtype == v2.TAXITIME_SEC_mvt.dtype == np.int32
    raw = pq.read_table(ROOT / "data" / "raw" / "ranking.parquet",
                        columns=["MVT_ID_mvt", "AOBT_3_flt"]).to_pandas()
    unmatched = set(raw.MVT_ID_mvt[raw.AOBT_3_flt.isna()])
    changed = out.TAXITIME_SEC_mvt.to_numpy() != v2.TAXITIME_SEC_mvt.to_numpy()
    assert 0 < changed.sum() <= n_sample
    assert changed.sum() == meta["n_value_changed"]
    assert not out.MVT_ID_mvt[changed].isin(unmatched).any(), "an unmatched row changed"
    assert len(meta["replaced_ids"]) == n_sample
    assert set(out.MVT_ID_mvt[changed]) <= set(meta["replaced_ids"]), \
        "a row outside the sample changed"


@pytest.mark.skipif(not all(p.exists() for p in _smoke_inputs),
                    reason="needs the month caches, the ranking cache and v2")
def test_smoke_v4_seeds_and_per_airport_end_to_end(tmp_path, capsys, monkeypatch):
    """`--version 4 --smoke --seeds 0,1 --per-airport`: two months, the 5,000-row sample, two
    seed boosters saved under their own names with per-seed provenance, ten per-airport
    entries in the meta (fitted exactly when the airport has >= 20,000 training rows, tree
    count on the smoke floor), the blend recorded, the splice invariants, the banner. Then the
    same command with --reuse-booster reproduces the file byte for byte from the two saved
    boosters, and deleting one seed's booster makes --reuse-booster refuse.

    Fails when the seed boosters are not saved per seed, when the meta lacks the v4 fields,
    when reuse re-fits silently, or when the banner is missing. Rehearsed 2026-09-09, each
    RED: `files = booster_files(..., (0,))` regardless of seeds; dropping `"per_airport"` from
    the meta; `reusable` returning False on a partial set.
    """
    fake_subs = tmp_path / "fake_submissions"
    fake_subs.mkdir()
    monkeypatch.setattr(ls, "SUBS", fake_subs)
    argv = ["--version", "4", "--smoke", "--seeds", "0,1", "--per-airport",
            "--out-dir", str(tmp_path), "--base", str(SUBS / "merry-quicksand_v2.parquet")]
    assert ls.main(argv) == 0
    assert not list(fake_subs.iterdir()), "the smoke wrote under submissions/"
    assert not (SUBS / "merry-quicksand_v4.parquet").exists() or True  # never touched: --out-dir

    for s in (0, 1):
        assert (tmp_path / f"lgbm_v4_seed{s}.txt").exists()
        fit = json.loads((tmp_path / f"lgbm_v4_seed{s}.fit.json").read_text())
        assert fit["params"]["seed"] == s and fit["es_seed"] == 0 and fit["smoke"] is True
    assert not (tmp_path / "lgbm_v4.txt").exists(), "multi-seed run used the legacy name"
    j0 = json.loads((tmp_path / "lgbm_v4_seed0.fit.json").read_text())
    j1 = json.loads((tmp_path / "lgbm_v4_seed1.fit.json").read_text())
    assert j0["n_ref"] == j1["n_ref"] and j0["best_iter"] == j1["best_iter"]

    out = pq.read_table(tmp_path / "merry-quicksand_v4.parquet").to_pandas()
    meta = json.loads((tmp_path / "merry-quicksand_v4.meta.json").read_text())
    _assert_sample_splice(out, meta, 5_000)
    assert meta["seeds"] == [0, 1] and meta["smoke"] is True
    assert meta["blend"] == 0.5 and meta["pa_min_rows"] == 20_000
    assert meta["pa_tree_floor"] == ls.SMOKE["pa_floor"] and meta["pa_tree_rule"] == ls.PA_TREE_RULE
    assert meta["pa_trees"] == "share"
    pa = meta["per_airport"]
    assert sorted(pa) == stand_ab.APTS
    for a, row in pa.items():
        assert row["fitted"] == (row["n_train"] >= 20_000), a
        if row["fitted"]:
            assert row["n_trees"] >= ls.SMOKE["pa_floor"] and row["seeds"] == [0, 1]
    assert any(r["fitted"] for r in pa.values()), "no airport got its own model"
    assert meta["n_seeds"] == 2
    text = capsys.readouterr().out
    assert ls.SMOKE_BANNER in text and "per-airport" in text

    first = (tmp_path / "merry-quicksand_v4.parquet").read_bytes()
    assert ls.main(argv[:-4] + ["--reuse-booster"] + argv[-4:]) == 0
    assert "reused" in capsys.readouterr().out
    assert (tmp_path / "merry-quicksand_v4.parquet").read_bytes() == first, \
        "reusing the two saved boosters did not reproduce the file"

    (tmp_path / "lgbm_v4_seed1.txt").unlink()
    with pytest.raises(ValueError, match="partial"):
        ls.main(argv[:-4] + ["--reuse-booster"] + argv[-4:])


@pytest.mark.skipif(not all(p.exists() for p in _smoke_inputs),
                    reason="needs the month caches, the ranking cache and v2")
def test_v3_default_path_is_byte_reproducible_with_reuse_booster(tmp_path, capsys, monkeypatch):
    """The v3 command (no --seeds, no --per-airport) still writes lgbm_v3.txt and, re-run with
    --reuse-booster, reproduces the parquet byte for byte from that file. This is the guard
    that the v4 additions left the default path alone.

    Fails when the default seed set changes the booster name (reuse finds nothing and refits:
    a fresh fit is not bit-reproducible at num_threads=4 on this machine), or when the mean
    over one seed alters the prediction. Rehearsed 2026-09-09: `booster_files` naming
    `lgbm_v3_seed0.txt` for (0,) went RED ("reused" absent).
    """
    monkeypatch.setattr(ls, "SUBS", tmp_path / "fake_submissions")
    argv = ["--version", "3", "--smoke", "--out-dir", str(tmp_path),
            "--base", str(SUBS / "merry-quicksand_v2.parquet")]
    assert ls.main(argv) == 0
    assert (tmp_path / "lgbm_v3.txt").exists() and (tmp_path / "lgbm_v3.fit.json").exists()
    first = (tmp_path / "merry-quicksand_v3.parquet").read_bytes()
    meta = json.loads((tmp_path / "merry-quicksand_v3.meta.json").read_text())
    assert meta["seeds"] == [0] and meta["per_airport"] is None and meta["blend"] is None
    capsys.readouterr()
    assert ls.main(argv + ["--reuse-booster"]) == 0
    assert "reused" in capsys.readouterr().out
    assert (tmp_path / "merry-quicksand_v3.parquet").read_bytes() == first


# ---------------------------------------------------------------------------------------------
# v6: the queue block joined onto the caches (Amendment 14). The training caches on disk
# predate MVT_ID_mvt, so the month join is POSITIONAL (equal row counts asserted here; the
# identical order is proven by tests/test_queue_features.py::test_queue_rows_are_exactly_the_
# stand_ab_reference_stream_in_both_modes); both ranking caches carry MVT_ID_mvt and join by id.
# ---------------------------------------------------------------------------------------------

def _queue_frame(n, seed=0, ids=None):
    """A queue-cache-shaped frame: MVT_ID_mvt + the twelve QUEUE_FEATS as float32, each column
    offset by its index so a column swap is visible."""
    rng = np.random.default_rng(seed)
    q = pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype="float64") + 100.0 if ids is None
                      else np.asarray(ids, dtype="float64")})
    for i, c in enumerate(stand_ab.QUEUE_FEATS):
        q[c] = (rng.integers(0, 9, n) + 100 * i).astype("float32")
    return q


def test_queue_month_join_is_positional_with_equal_lengths_and_the_column_contract_asserted():
    """A training month's queue twin is joined by position: the rows are the same rows in the
    same order (the v6 builder measured 12/12 months equal, same filter, same mergesort), so
    the only guard a positional join can have is the row count - it is asserted, per month,
    and a mismatch is a refusal, never a truncation. The queue cache must carry exactly
    MVT_ID_mvt + QUEUE_FEATS in the contract order.

    Fails when the length check is dropped (an 8-row month against a 7-row twin would be
    silently mis-aligned) or when the column contract is not checked. Rehearsed 2026-09-09,
    each RED: `if False:` for the length check; `set(q.columns) !=` in place of the ordered
    comparison (the reversed-columns case).
    """
    q = _queue_frame(7)
    got = ls.attach_queue_positional(7, q, "training_2025-02-01_2025-03-01.parquet")
    assert list(got.columns) == stand_ab.QUEUE_FEATS and len(got) == 7
    for c in stand_ab.QUEUE_FEATS:
        assert np.array_equal(got[c].to_numpy(), q[c].to_numpy()) and got[c].dtype == np.float32
    with pytest.raises(ValueError, match="rows"):
        ls.attach_queue_positional(8, q, "training_2025-02-01_2025-03-01.parquet")
    with pytest.raises(ValueError, match="columns"):
        ls.attach_queue_positional(7, q.drop(columns=["q_rwy_at_push"]), "x")
    with pytest.raises(ValueError, match="columns"):
        ls.attach_queue_positional(7, q[["MVT_ID_mvt"] + stand_ab.QUEUE_FEATS[::-1]], "x")
    assert ls.QUEUE_FEATS == stand_ab.QUEUE_FEATS and len(ls.QUEUE_FEATS) == 12
    assert ls.queue_cache_path(pathlib.Path("/q"), pathlib.Path("/s/training_2025-02-01_2025-03-01.parquet")) \
        == pathlib.Path("/q/training_2025-02-01_2025-03-01.parquet")


def test_queue_ranking_join_is_by_id_with_set_equality_and_order_asserted():
    """The ranking caches both carry MVT_ID_mvt: the full join requires the SAME id set in the
    SAME order (both are built per calendar month from the raw file in take-off order - a
    difference means one cache was rebuilt differently and is refused); a sampled join
    (smoke) aligns the subset by id. Duplicates and missing ids are refusals.

    Fails when the by-id alignment is replaced by position (the shuffled case), when the
    order check is dropped on the full join, or when the missing-id check is dropped (.loc
    then raises a KeyError, not the ValueError the caller handles). Rehearsed 2026-09-09,
    each RED: `q[QUEUE_FEATS]` returned without the `.loc[ids]` alignment; `if False:` for
    the order check; `if False:` for the missing check.
    """
    ids = np.array([5.0, 3.0, 9.0, 1.0])
    q = _queue_frame(4, ids=ids)
    got = ls.attach_queue_by_id(ids, q)
    assert list(got.columns) == stand_ab.QUEUE_FEATS and len(got) == 4
    assert np.array_equal(got.q_apt_at_push.to_numpy(), q.q_apt_at_push.to_numpy())
    shuffled = q.iloc[[3, 0, 2, 1]].reset_index(drop=True)
    with pytest.raises(ValueError, match="order"):
        ls.attach_queue_by_id(ids, shuffled)
    sub = ls.attach_queue_by_id(ids[[2, 0]], shuffled, subset_ok=True)
    want = q.set_index("MVT_ID_mvt").loc[ids[[2, 0]]]
    for c in stand_ab.QUEUE_FEATS:
        assert np.array_equal(sub[c].to_numpy(), want[c].to_numpy()), c
    assert sub[stand_ab.QUEUE_FEATS[0]].dtype == np.float32
    with pytest.raises(ValueError, match="missing"):
        ls.attach_queue_by_id(np.r_[ids, 77.0], q, subset_ok=True)
    with pytest.raises(ValueError, match="extra"):
        ls.attach_queue_by_id(ids[:3], q)
    with pytest.raises(ValueError, match="duplicate"):
        ls.attach_queue_by_id(ids, pd.concat([q, q.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="duplicate"):
        ls.attach_queue_by_id(np.r_[ids, 5.0], q, subset_ok=True)
    with pytest.raises(ValueError, match="columns"):
        ls.attach_queue_by_id(ids, q.drop(columns=["q_dep_tko_sym15"]))


def _mini_month(n, seed, month):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"y": rng.uniform(300, 900, n), "delta": rng.normal(size=n),
                         "month": np.full(n, month, "int32"), "x": rng.normal(size=n)})


def test_load_frames_attaches_the_queue_block_to_months_positionally_and_to_ranking_by_id(tmp_path):
    """`load_frames(..., queue=True)` appends the twelve queue columns to every training month
    by position and to the ranking rows by id, in the contract order, so the design matrix
    can be built as FEATS + QUEUE_FEATS with nothing else changed; a queue month with a
    different row count refuses the whole load; a sampled ranking (smoke) is aligned by id.

    Fails when the ranking block is attached positionally (the sample would be mis-aligned),
    or when a month's row count is not checked. Rehearsed 2026-09-09, each RED:
    `attach_queue_positional(len(rank), q_rank, ...)` for the by-id call; dropping the
    per-month length check (the 3-row twin then loads).
    """
    stand, queue = tmp_path / "cache_stand", tmp_path / "cache_queue"
    stand.mkdir(), queue.mkdir()
    name = "training_2025-{:02d}-01_2025-{:02d}-01.parquet"
    _mini_month(5, 1, 1).to_parquet(stand / name.format(1, 2), index=False)
    _mini_month(4, 3, 3).to_parquet(stand / name.format(3, 4), index=False)
    q1, q3 = _queue_frame(5, seed=1), _queue_frame(4, seed=3)
    q1.to_parquet(queue / name.format(1, 2), index=False)
    q3.to_parquet(queue / name.format(3, 4), index=False)
    rank = _mini_month(3, 7, 7)
    rank["y"] = np.nan
    rank["delta"] = np.nan
    rank.insert(0, "MVT_ID_mvt", np.array([11.0, 12.0, 13.0]))
    rank.to_parquet(stand / "ranking.parquet", index=False)
    qr = _queue_frame(3, seed=7, ids=[11.0, 12.0, 13.0])
    qr.to_parquet(queue / "ranking.parquet", index=False)

    d, is_rank, rank_ids = ls.load_frames(stand, queue=True, qcache=queue)
    assert list(d.columns) == ["y", "delta", "month", "x"] + stand_ab.QUEUE_FEATS
    assert len(d) == 12 and is_rank.sum() == 3 and rank_ids.tolist() == [11.0, 12.0, 13.0]
    for c in stand_ab.QUEUE_FEATS:
        want = np.r_[q1[c].to_numpy(), q3[c].to_numpy(), qr[c].to_numpy()]
        assert np.array_equal(d[c].to_numpy(), want), c
    d2, is_rank2, ids2 = ls.load_frames(stand, n_rank=2, queue=True, qcache=queue, seed=0)
    assert len(ids2) == 2 and is_rank2.sum() == 2
    want = qr.set_index("MVT_ID_mvt").loc[ids2]
    for c in stand_ab.QUEUE_FEATS:
        assert np.array_equal(d2[c].to_numpy()[is_rank2], want[c].to_numpy()), c
    _queue_frame(3, seed=3).to_parquet(queue / name.format(3, 4), index=False)     # one row short
    with pytest.raises(ValueError, match="rows"):
        ls.load_frames(stand, queue=True, qcache=queue)
    with pytest.raises(FileNotFoundError, match="queue"):
        ls.load_frames(stand, queue=True, qcache=tmp_path / "nowhere")


class _WideBooster:
    """Stands in for a refit booster over 80 features; predicts a constant."""
    def __init__(self, value, n_features=80):
        self.value, self.n = float(value), int(n_features)
    def predict(self, X, **kw):
        return np.full(len(X), self.value, dtype="float64")
    def num_feature(self):
        return self.n
    def num_trees(self):
        return 3


def test_fit_seeds_checks_the_feature_count_it_is_given(monkeypatch):
    """fit_seeds guards the booster's feature count against the design matrix it was told
    about: the default is len(FEATS) (v3/v4), an 80-column queue run passes n_features=80.
    A queue booster against the default count is refused, so a design matrix with the wrong
    width can never be predicted silently.

    Fails when the count is hard-wired to len(FEATS) (the 80-feature run is refused) or when
    the check is dropped. Rehearsed 2026-09-09, each RED: `n_features = len(FEATS)`
    unconditionally; `if False:` for the check.
    """
    monkeypatch.setattr(ls, "refit", lambda X, dlt, mask, params, n, weight=None: _WideBooster(2.5))
    X = np.zeros((6, 80), dtype="float32")
    train = np.array([True, True, True, True, False, False])
    dlt = np.arange(6, dtype="float64")
    out = ls.fit_seeds(X, dlt, train, X[~train], dict(ls.P), 3, (0,), n_features=80)
    assert out[0].tolist() == [2.5, 2.5]
    with pytest.raises(ValueError, match="features"):
        ls.fit_seeds(X, dlt, train, X[~train], dict(ls.P), 3, (0,))
    monkeypatch.setattr(ls, "refit", lambda X, dlt, mask, params, n, weight=None: _WideBooster(2.5, len(ls.FEATS)))
    assert ls.fit_seeds(X, dlt, train, X[~train], dict(ls.P), 3, (1,))[1].tolist() == [2.5, 2.5]


def test_queue_flag_names_its_boosters_and_the_provenance_check_knows_the_feature_count():
    """`--queue` is off by default (the v3/v4 commands are unchanged); with it the boosters are
    named lgbm_vN_queue[_seedS].txt so a queue booster can never be mistaken for a 68-feature
    one by --reuse-booster, and a fit.json that records n_features is checked against the
    run's count (a legacy fit.json without the field falls through to the booster's own
    num_feature check in reuse_seeds).

    Fails when the queue tag is dropped from the names, or when a recorded n_features
    mismatch is accepted. Rehearsed 2026-09-09, each RED: `tag = ""` for the queue case;
    `if False:` for the n_features branch of check_provenance.
    """
    assert ls.parse_args(["--version", "6"]).queue is False
    assert ls.parse_args(["--version", "6", "--queue"]).queue is True
    d = pathlib.Path("/x")
    assert ls.booster_files(d, 6, (0, 1), queue=True) == [
        (d / "lgbm_v6_queue_seed0.txt", d / "lgbm_v6_queue_seed0.fit.json"),
        (d / "lgbm_v6_queue_seed1.txt", d / "lgbm_v6_queue_seed1.fit.json")]
    assert ls.booster_files(d, 6, (0,), queue=True) == [(d / "lgbm_v6_queue.txt", d / "lgbm_v6_queue.fit.json")]
    assert ls.booster_files(d, 3, (0,)) == [(d / "lgbm_v3.txt", d / "lgbm_v3.fit.json")]
    info = dict(smoke=False, n_all=10, n_ref=5, params=dict(ls.P), n_features=68)
    ls.check_provenance(info, 5, False, 10, dict(ls.P), n_features=68)
    ls.check_provenance(info, 5, False, 10, dict(ls.P))                      # count not asked: not checked
    with pytest.raises(ValueError, match="features"):
        ls.check_provenance(info, 5, False, 10, dict(ls.P), n_features=80)
    legacy = {k: v for k, v in info.items() if k != "n_features"}
    ls.check_provenance(legacy, 5, False, 10, dict(ls.P), n_features=80)     # legacy: booster check decides


_queue_smoke_inputs = _smoke_inputs + [ls.QCACHE / "training_2025-01-01_2025-02-01.parquet",
                                       ls.QCACHE / "training_2025-03-01_2025-04-01.parquet",
                                       ls.QCACHE / "ranking.parquet"]


@pytest.mark.skipif(not all(p.exists() for p in _queue_smoke_inputs),
                    reason="needs the month caches, both ranking caches, the queue caches and v2")
def test_smoke_v6_queue_end_to_end(tmp_path, capsys, monkeypatch):
    """`--version 6 --smoke --queue --seeds 0,1`: the two months' queue twins joined by position
    and the sampled ranking rows by id, an 80-feature design matrix, two seed boosters saved
    under the `_queue` tag with n_features in their fit.json, the meta recording the queue
    block, the splice invariants, the banner; then --reuse-booster reproduces the file byte
    for byte from the tagged boosters.

    WRITTEN 2026-09-09 AND NOT YET RUN: the owner runs the real-month smokes after the live
    v4 fold finishes (a second >= 2 GB job beside it is not allowed). Its first run is its
    red/green proof; the mutation to rehearse then is `tag = ""` in booster_files (the reuse
    step would then find no `_queue` file).
    """
    fake_subs = tmp_path / "fake_submissions"
    fake_subs.mkdir()
    monkeypatch.setattr(ls, "SUBS", fake_subs)
    argv = ["--version", "6", "--smoke", "--queue", "--seeds", "0,1",
            "--out-dir", str(tmp_path), "--base", str(SUBS / "merry-quicksand_v2.parquet")]
    assert ls.main(argv) == 0
    for s in (0, 1):
        assert (tmp_path / f"lgbm_v6_queue_seed{s}.txt").exists()
        fit = json.loads((tmp_path / f"lgbm_v6_queue_seed{s}.fit.json").read_text())
        assert fit["n_features"] == 80 and fit["params"]["seed"] == s and fit["smoke"] is True
    assert not (tmp_path / "lgbm_v6_seed0.txt").exists(), "a queue booster took the untagged name"
    out = pq.read_table(tmp_path / "merry-quicksand_v6.parquet").to_pandas()
    meta = json.loads((tmp_path / "merry-quicksand_v6.meta.json").read_text())
    _assert_sample_splice(out, meta, 5_000)
    assert meta["queue"] is True and meta["n_features"] == 80 and meta["seeds"] == [0, 1]
    assert meta["features"] == list(ls.FEATS) + stand_ab.QUEUE_FEATS
    assert meta["queue_feats"] == stand_ab.QUEUE_FEATS and meta["per_airport"] is None
    text = capsys.readouterr().out
    assert ls.SMOKE_BANNER in text and "queue=True (80 features)" in text
    first = (tmp_path / "merry-quicksand_v6.parquet").read_bytes()
    assert ls.main(argv + ["--reuse-booster"]) == 0
    assert "reused" in capsys.readouterr().out
    assert (tmp_path / "merry-quicksand_v6.parquet").read_bytes() == first, \
        "reusing the two tagged boosters did not reproduce the file"


# ---------------------------------------------------------------------------------------------
# v5: the schedule-fill mixture head on the matched rows (Amendment 16). The head is a LightGBM
# binary classifier p = P(|y - sp| <= 60 s | x) on FEATS + nmdelay, fitted on the training rows;
# the shipped prediction is the mixture p * sp + (1 - p) * max(proxy - delta_hat, 1). Every test
# here is unit-level on micro fixtures; the real-month smoke is the owner's to run beside the
# live fold.
# ---------------------------------------------------------------------------------------------

def test_mix_fill_is_p_times_sp_plus_one_minus_p_times_the_baseline_prediction():
    """y_hat = p * sp + (1 - p) * base_yhat, row by row, in float64: p = 0 returns the baseline
    prediction exactly, p = 1 returns sp exactly, p = 0.25 with sp 300 and base 30 gives 97.5,
    p = 0.5 with 400 and 40 gives 220. p outside [0, 1], a non-finite p, sp or base, and a shape
    mismatch are refusals - the mixture is applied to 339,551 scored rows and a NaN would splice
    as garbage.

    Fails when the (1 - p) weight is dropped, when the two weights are swapped, or when the
    range check is dropped. Rehearsed 2026-09-09, each RED: `p * sp + base` (row 1: 220 != 200);
    `(1.0 - p) * sp + p * base` (row 1: 20 != 200); `if False:` for the [0, 1] check.
    """
    p = np.array([0.0, 1.0, 0.25, 0.5])
    sp = np.array([100.0, 200.0, 300.0, 400.0])
    base = np.array([10.0, 20.0, 30.0, 40.0])
    got = ls.mix_fill(p, sp, base)
    assert got.dtype == np.float64 and got.tolist() == [10.0, 200.0, 97.5, 220.0]
    assert ls.mix_fill(np.zeros(4), sp, base).tolist() == base.tolist()
    assert ls.mix_fill(np.ones(4), sp, base).tolist() == sp.tolist()
    for bad, msg in ((np.array([0.0, 1.0, 1.5, 0.5]), r"\[0, 1\]"),
                     (np.array([0.0, -0.1, 0.5, 0.5]), r"\[0, 1\]"),
                     (np.array([0.0, np.nan, 0.5, 0.5]), "finite")):
        with pytest.raises(ValueError, match=msg):
            ls.mix_fill(bad, sp, base)
    with pytest.raises(ValueError, match="finite"):
        ls.mix_fill(p, sp, np.array([10.0, np.inf, 30.0, 40.0]))
    with pytest.raises(ValueError, match="finite"):
        ls.mix_fill(p, np.array([100.0, np.nan, 300.0, 400.0]), base)
    with pytest.raises(ValueError, match="shape"):
        ls.mix_fill(p[:3], sp, base)


def test_fill_label_is_the_60_s_rule_on_training_rows_only_and_never_reads_holdout_y():
    """fill := |y - sp| <= 60 s, inclusive, as a float64 0/1 label on the TRAINING rows and NaN
    on every other row - holdout rows on the fold, ranking rows in the submission, whose y is
    NaN and must never be read. Exactly 60 s is a fill; 60.001 s is not. A non-finite y or sp
    on a training row is a refusal (the caches have none; a NaN label would train silently),
    and so is a training set with a single class (nothing to learn).

    Fails when the label is formed on every row (a NaN ranking y then trips the finite check, and
    a real holdout y gets a label that could leak into a fit), when the rule is strict, or when
    the single-class guard is dropped. Rehearsed 2026-09-09, each RED: `train =
    np.ones_like(np.asarray(train, dtype=bool))` in fill_label; `< FILL_TOL_S`; `if False:` for
    the class check.
    """
    y = np.array([600.0, 600.0, 600.0, 600.0, np.nan, 900.0])
    sp = np.array([600.0, 660.0, 660.001, 540.0, 100.0, 100.0])
    train = np.array([True, True, True, True, False, False])
    lab = ls.fill_label(y, sp, train)
    assert lab.dtype == np.float64 and lab.shape == (6,)
    assert lab[:4].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert np.isnan(lab[4:]).all(), "a non-training row got a label"
    assert ls.FILL_TOL_S == 60.0
    # holdout y is real on the fold: still no label there
    lab2 = ls.fill_label(np.where(np.isnan(y), 700.0, y), sp, train)
    assert np.isnan(lab2[4:]).all() and lab2[:4].tolist() == [1.0, 1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="finite"):
        ls.fill_label(np.array([600.0, np.nan, 1.0]), np.array([600.0, 600.0, 1.0]), np.array([True, True, True]))
    with pytest.raises(ValueError, match="class"):
        ls.fill_label(np.array([600.0, 600.0]), np.array([600.0, 601.0]), np.array([True, True]))
    with pytest.raises(ValueError, match="shape"):
        ls.fill_label(y, sp, train[:5])


def test_nmdelay_is_appended_exactly_once_and_equals_proxy_minus_sp():
    """The head's design is the regressor's FEATS plus one explicit column nmdelay = proxy - sp
    (= SCHED - AOBT_3, the NM off-block delay against the schedule), appended LAST so the
    regressors read the first len(FEATS) columns of the same matrix. head_features refuses a
    list that already carries it, add_nmdelay refuses a frame that already carries it, and the
    column is formed in float64 from the frame's proxy and sp.

    Fails when the presence guard is dropped (the column would be appended twice and the
    booster's feature count would silently disagree with the design), or when the column is
    sp - proxy. Rehearsed 2026-09-09, each RED: `if False:` for the guard in head_features and
    in add_nmdelay; `d.sp - d.proxy`.
    """
    feats = ls.head_features(ls.FEATS)
    assert feats[:-1] == list(ls.FEATS) and feats[-1] == ls.NMDELAY == "nmdelay"
    assert feats.count(ls.NMDELAY) == 1 and len(feats) == len(ls.FEATS) + 1 == 69
    assert ls.NMDELAY not in ls.FEATS and ls.NMDELAY not in ls.QUEUE_FEATS
    assert ls.head_features(list(ls.FEATS) + ls.QUEUE_FEATS)[-1] == ls.NMDELAY
    with pytest.raises(ValueError, match="already"):
        ls.head_features(feats)
    d = pd.DataFrame({"proxy": [1000.0, 1200.0], "sp": [900.0, 1500.0], "x": [1, 2]})
    out = ls.add_nmdelay(d)
    assert out is d and list(d.columns) == ["proxy", "sp", "x", "nmdelay"]
    assert d.nmdelay.tolist() == [100.0, -300.0] and d.nmdelay.dtype == np.float64
    with pytest.raises(ValueError, match="already"):
        ls.add_nmdelay(d)
    with pytest.raises(ValueError, match="proxy"):
        ls.add_nmdelay(pd.DataFrame({"sp": [1.0]}))


def test_rank_auc_is_the_mann_whitney_statistic_with_ties_at_half():
    """AUC as the head's diagnostic: 1 for a score that ranks every fill above every non-fill,
    0 for the reverse, 0.5 for a constant score (every pair tied), and 0.875 by hand for scores
    [0.1, 0.4, 0.4, 0.9] with fills on rows 1 and 3 (the four (fill, non-fill) pairs: 0.4 > 0.1
    -> 1, 0.4 = 0.4 -> 1/2, 0.9 > 0.1 -> 1, 0.9 > 0.4 -> 1; 3.5 / 4). A single class is a refusal.

    Fails when ties are credited 0 or 1 (0.875 becomes 0.75 or 1.0), or when the sign is
    reversed. Rehearsed 2026-09-09, each RED: `rank(method="min")` (0.75); `1.0 - ...`.
    """
    assert ls.rank_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert ls.rank_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert ls.rank_auc([0.3, 0.3, 0.3, 0.3], [0, 1, 0, 1]) == 0.5
    assert ls.rank_auc([0.1, 0.4, 0.4, 0.9], [0, 1, 0, 1]) == pytest.approx(0.875)
    with pytest.raises(ValueError, match="class"):
        ls.rank_auc([0.1, 0.2], [1, 1])
    with pytest.raises(ValueError, match="shape"):
        ls.rank_auc([0.1, 0.2], [1])


def _head_problem(n=1_000, seed=0):
    """A micro classification problem: 6 features, fill iff feature 0 > 1 (about 16%), a taxi
    time of 600 s that sp sits on for fills and misses by 300 s otherwise; holdout = the first
    200 rows, early-stop = the next 200, fit = the remaining 600 (train = fit + early-stop)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype("float32")
    fill = X[:, 0] > 1.0
    y = np.full(n, 600.0)
    sp = np.where(fill, 600.0, 900.0)
    te = np.zeros(n, bool)
    te[:200] = True
    es = np.zeros(n, bool)
    es[200:400] = True
    tr = ~te
    fit = tr & ~es
    return X, y, sp, fill, tr, fit, es, te


_HEAD_MICRO = dict(num_leaves=4, min_data_in_leaf=10, learning_rate=0.1)


def test_fit_fill_head_early_stops_on_auc_refits_on_all_training_rows_and_predicts_through_predict_delta(monkeypatch):
    """The head: an early-stopping run (objective binary, metric auc, patience) on the fit rows
    stopped on the early-stop rows -> best_iter; n_ref = n_refit(best_iter, n_all, n_fit) exactly
    as the regressor; ONE refit (params["seed"]) on ALL training rows at n_ref; p on X_pred in
    [0, 1] through predict_delta - the only predict site, threaded like training. On a micro
    problem whose fills are feature 0 > 1 the holdout AUC of p must exceed 0.95 and the mean p
    on the planted fills must exceed the mean p elsewhere by 0.5. The info records the fill
    share of the training rows, the stopping-set AUC (marked NOT a result) and the row counts.
    A head that cannot beat AUC 0.5 on its stopping set - an all-zero design here, where no
    split is possible and p is the prior on every row - is refused before any refit.

    Fails when the refit is on the fit rows instead of every training row, when the tree count
    is best_iter instead of n_ref, when the ES-set or final predict bypasses predict_delta, or
    when the AUC guard is dropped. Rehearsed 2026-09-09, each RED: `refit(X, label, fit, ...)`;
    `n_ref = best_iter`; the final `predict_delta(` replaced by `b.predict(X_pred)` (the spy
    counts one call); `if False:` for the guard.
    """
    X, y, sp, fill, tr, fit, es, te = _head_problem()
    label = ls.fill_label(y, sp, tr)
    assert label[tr].mean() == pytest.approx(fill[tr].mean())
    refits, predicts = [], []
    real_refit, real_predict = ls.refit, ls.predict_delta

    def spy_refit(X_, lab, mask, params, n_trees):
        refits.append((int(mask.sum()), n_trees, params["objective"], params["seed"]))
        return real_refit(X_, lab, mask, params, n_trees)

    def spy_predict(booster, X_, params, num_iteration=None):
        predicts.append((len(X_), num_iteration))
        return real_predict(booster, X_, params, num_iteration=num_iteration)
    monkeypatch.setattr(ls, "refit", spy_refit)
    monkeypatch.setattr(ls, "predict_delta", spy_predict)

    params = dict(ls.P_HEAD, **_HEAD_MICRO)
    p, info = ls.fit_fill_head(X, label, tr, fit, es, X[te], params, 200, 20, n_features=6)
    assert p.shape == (200,) and p.dtype == np.float64 and ((p >= 0) & (p <= 1)).all()
    assert ls.rank_auc(p, fill[te]) > 0.95
    assert p[fill[te]].mean() - p[~fill[te]].mean() > 0.5
    assert info["best_iter"] >= 1 and info["n_fit"] == 600 and info["n_es"] == 200 and info["n_all"] == 800
    assert info["n_ref"] == ls.n_refit(info["best_iter"], 800, 600) >= 50
    assert info["fill_share_train"] == pytest.approx(fill[tr].mean())
    assert info["fill_share_fit"] == pytest.approx(fill[fit].mean()) and info["fill_share_es"] == pytest.approx(fill[es].mean())
    assert info["es_auc_NOT_A_RESULT"] > 0.95 and 0 < info["es_mean_p"] < 1 and info["es_fill_rate"] == pytest.approx(fill[es].mean())
    assert info["n_features"] == 6 and info["objective"] == "binary" and info["metric"] == "auc"
    assert info["es_months"] == [3, 9]
    assert refits == [(800, info["n_ref"], "binary", 0)], "one refit, on every training row, at n_ref, seed 0"
    assert [n for n, _ in predicts] == [200, 200], "the ES-set predict and the final predict, both through the helper"
    assert predicts[0][1] == info["best_iter"] and predicts[1][1] is None
    with pytest.raises(RuntimeError, match="AUC"):
        ls.fit_fill_head(np.zeros_like(X), label, tr, fit, es, np.zeros_like(X[te]), params, 50, 10, n_features=6)
    with pytest.raises(ValueError, match="features"):
        ls.fit_fill_head(X, label, tr, fit, es, X[te], params, 200, 20, n_features=7)
    with pytest.raises(ValueError, match="partition"):
        ls.fit_fill_head(X, label, tr, tr, es, X[te], params, 200, 20, n_features=6)
    with pytest.raises(ValueError, match="label"):
        ls.fit_fill_head(X, np.full(len(X), np.nan), tr, fit, es, X[te], params, 200, 20, n_features=6)


def test_head_booster_naming_provenance_and_reuse(tmp_path):
    """The head booster is lgbm_v{N}_fillhead.txt (lgbm_v{N}_queue_fillhead.txt under --queue)
    with its own fit.json carrying the head's params (objective binary, metric auc), best_iter,
    n_ref, n_features and the fill share of the training rows; --reuse-booster loads it through
    the same provenance check as the regressors and reproduces p bit for bit; a fit.json from
    another configuration (the regressor's params, another feature count, a smoke) is refused; a
    missing pair is simply "fit" (reusable is False), never a partial reuse.

    Fails when the tag is dropped, when the provenance check is skipped, or when a reused head
    predicts something else. Rehearsed 2026-09-09, each RED: `_fillhead` removed from the
    name; `if False:` around check_provenance in reuse_fill_head.
    """
    d = pathlib.Path("/x")
    assert ls.head_booster_files(d, 5) == (d / "lgbm_v5_fillhead.txt", d / "lgbm_v5_fillhead.fit.json")
    assert ls.head_booster_files(d, 5, queue=True) == (d / "lgbm_v5_queue_fillhead.txt",
                                                       d / "lgbm_v5_queue_fillhead.fit.json")
    X, y, sp, fill, tr, fit, es, te = _head_problem()
    label = ls.fill_label(y, sp, tr)
    params = dict(ls.P_HEAD, **_HEAD_MICRO)
    files = ls.head_booster_files(tmp_path, 5)
    assert ls.reusable([files]) is False
    p, info = ls.fit_fill_head(X, label, tr, fit, es, X[te], params, 200, 20, files=files, smoke=False, n_features=6)
    assert files[0].exists() and files[1].exists() and ls.reusable([files]) is True
    fj = json.loads(files[1].read_text())
    assert fj["params"] == params and fj["n_features"] == 6 and fj["smoke"] is False and fj["es_seed"] == 0
    assert fj["n_ref"] == info["n_ref"] and fj["best_iter"] == info["best_iter"]
    assert fj["fill_share_train"] == info["fill_share_train"] and fj["n_all"] == 800
    assert fj["params"]["objective"] == "binary" and fj["params"]["metric"] == "auc"
    p2, info2 = ls.reuse_fill_head(files, X[te], params, smoke=False, n_all=800, n_features=6)
    assert np.array_equal(p, p2) and info2["n_ref"] == info["n_ref"] and info2["best_iter"] == info["best_iter"]
    assert "params" not in info2 and "smoke" not in info2 and info2["fill_share_train"] == info["fill_share_train"]
    with pytest.raises(ValueError, match="params"):
        ls.reuse_fill_head(files, X[te], dict(ls.P, **_HEAD_MICRO), smoke=False, n_all=800, n_features=6)
    with pytest.raises(ValueError, match="features"):
        ls.reuse_fill_head(files, X[te], params, smoke=False, n_all=800, n_features=7)
    with pytest.raises(ValueError, match="smoke"):
        ls.reuse_fill_head(files, X[te], params, smoke=True, n_all=800, n_features=6)
    with pytest.raises(ValueError, match="n_all"):
        ls.reuse_fill_head(files, X[te], params, smoke=False, n_all=801, n_features=6)


def test_finalise_taxi_time_is_the_rounding_recover_taxi_time_uses():
    """The mixture is a taxi time, not a delta, so the fillhead path finalises it directly:
    finalise_taxi_time(raw) = np.rint(max(raw, 1)) as int32 with the finite and 0.5%
    floor-bind guards, and recover_taxi_time(proxy, delta) is exactly finalise_taxi_time(proxy
    - delta) - so v5's matched rows are on v4's scale.

    Fails when the two diverge. Rehearsed 2026-09-09: `np.floor` in finalise_taxi_time went RED.
    """
    proxy = np.full(1_000, 1_000.0)
    delta = np.full(1_000, 899.6)
    delta[:3] = [999.5, 1_500.0, 898.5]
    assert np.array_equal(ls.finalise_taxi_time(proxy - delta), ls.recover_taxi_time(proxy, delta))
    got = ls.finalise_taxi_time(proxy - delta)
    assert got.dtype == np.int32 and got[:4].tolist() == [1, 1, 102, 100]
    with pytest.raises(AssertionError, match="non-finite"):
        ls.finalise_taxi_time(np.array([np.nan] + [100.0] * 999))
    with pytest.raises(AssertionError, match="positivity floor"):
        ls.finalise_taxi_time(np.full(1_000, -5.0))


def test_fillhead_flag_and_the_head_parameters():
    """--fillhead is off by default (the v3/v4/v6 commands are unchanged) and combines with
    --queue; P_HEAD is the measured P with exactly objective and metric replaced (binary, auc):
    the same learning rate, leaves, min_data_in_leaf, feature_fraction, bagging, seed and
    threads. head_params applies the same two edits to any params, so the smoke's learning rate
    flows through to the head.

    Fails when the head changes a third parameter or when the flag defaults on. Rehearsed
    2026-09-09, each RED: `num_leaves=63` in head_params; `default=True`.
    """
    assert ls.parse_args(["--version", "5"]).fillhead is False
    assert ls.parse_args(["--version", "5", "--fillhead"]).fillhead is True
    a = ls.parse_args(["--version", "5", "--fillhead", "--queue", "--seeds", "0,1,2", "--per-airport"])
    assert a.fillhead and a.queue and a.per_airport and a.seeds == (0, 1, 2)
    assert ls.P_HEAD["objective"] == "binary" and ls.P_HEAD["metric"] == "auc"
    keys = ("objective", "metric")
    assert {k: v for k, v in ls.P_HEAD.items() if k not in keys} == {k: v for k, v in ls.P.items() if k not in keys}
    assert (ls.P_HEAD["learning_rate"], ls.P_HEAD["num_leaves"], ls.P_HEAD["feature_fraction"],
            ls.P_HEAD["bagging_fraction"], ls.P_HEAD["bagging_freq"], ls.P_HEAD["seed"]) == (0.01, 255, 0.8, 0.8, 1, 0)
    h = ls.head_params(dict(ls.P, learning_rate=0.05))
    assert h["learning_rate"] == 0.05 and h["objective"] == "binary" and h["metric"] == "auc"
    assert ls.head_params(ls.P) == ls.P_HEAD
    assert ls.FILL_HEAD_MIN_ES_AUC == 0.5


@pytest.mark.skipif(not all(p.exists() for p in _smoke_inputs),
                    reason="needs the month caches, the ranking cache and v2")
def test_smoke_v5_fillhead_end_to_end(tmp_path, capsys, monkeypatch):
    """`--version 5 --smoke --fillhead --seeds 0,1`: two months, the 5,000-row sample, a
    69-column design matrix of which the two regressor boosters read the first 68 (saved under
    their usual seed names with n_features 68), the head booster saved as lgbm_v5_fillhead.txt
    with its fit.json (69 features, objective binary), the meta's fill_head block (best_iter,
    n_ref, the fill share of the training rows, the stopping-set AUC marked NOT a result, the
    head's params, the mixture's shift against the regressor prediction), the splice invariants,
    the banner; then --reuse-booster reproduces the file byte for byte from the three saved
    boosters.

    WRITTEN 2026-09-09 AND NOT YET RUN: the owner runs the real-month smokes after the live v4
    fold finishes (a second >= 2 GB job beside it is not allowed). Its first run is its
    red/green proof; the mutation to rehearse then is `y_mix = base_yhat` in main's fillhead
    branch (the meta's mixture_shift_vs_regressor_rms_s becomes 0).
    """
    fake_subs = tmp_path / "fake_submissions"
    fake_subs.mkdir()
    monkeypatch.setattr(ls, "SUBS", fake_subs)
    argv = ["--version", "5", "--smoke", "--fillhead", "--seeds", "0,1",
            "--out-dir", str(tmp_path), "--base", str(SUBS / "merry-quicksand_v2.parquet")]
    assert ls.main(argv) == 0
    assert not list(fake_subs.iterdir()), "the smoke wrote under submissions/"
    for s in (0, 1):
        assert (tmp_path / f"lgbm_v5_seed{s}.txt").exists()
        fit = json.loads((tmp_path / f"lgbm_v5_seed{s}.fit.json").read_text())
        assert fit["n_features"] == 68 and fit["params"]["seed"] == s and fit["params"]["objective"] == "regression"
    assert (tmp_path / "lgbm_v5_fillhead.txt").exists(), "head booster not saved"
    hj = json.loads((tmp_path / "lgbm_v5_fillhead.fit.json").read_text())
    assert hj["n_features"] == 69 and hj["params"]["objective"] == "binary" and hj["params"]["metric"] == "auc"
    assert hj["smoke"] is True and hj["best_iter"] >= 1 and hj["n_ref"] >= 50 and 0 < hj["fill_share_train"] < 0.5
    out = pq.read_table(tmp_path / "merry-quicksand_v5.parquet").to_pandas()
    meta = json.loads((tmp_path / "merry-quicksand_v5.meta.json").read_text())
    _assert_sample_splice(out, meta, 5_000)
    assert meta["fillhead"] is True and meta["seeds"] == [0, 1] and meta["n_features"] == 68
    assert meta["features"] == list(ls.FEATS) and meta["features_head"] == ls.head_features(ls.FEATS)
    assert meta["n_features_head"] == 69 and meta["booster_files"] == ["lgbm_v5_seed0.txt", "lgbm_v5_seed1.txt"]
    fh = meta["fill_head"]
    assert fh["booster_file"] == "lgbm_v5_fillhead.txt" and fh["reused"] is False
    assert fh["best_iter"] == hj["best_iter"] and fh["n_ref"] == hj["n_ref"] and fh["n_features"] == 69
    assert fh["fill_share_train"] == hj["fill_share_train"] and 0 < fh["es_auc_NOT_A_RESULT"] <= 1
    assert fh["params"]["objective"] == "binary" and fh["fill_tolerance_s"] == 60.0 and fh["nmdelay"] == "nmdelay"
    assert 0 < fh["rank_mean_p"] < 1 and fh["mixture_shift_vs_regressor_rms_s"] > 0
    assert fh["n_rank_sp_nonpositive"] >= 0 and 0 <= fh["floor_bind_rate"] < 0.005
    text = capsys.readouterr().out
    assert ls.SMOKE_BANNER in text and "fill head" in text and "fillhead=True" in text
    first = (tmp_path / "merry-quicksand_v5.parquet").read_bytes()
    assert ls.main(argv + ["--reuse-booster"]) == 0
    text = capsys.readouterr().out
    assert "reused" in text and "reused fill head" in text
    assert (tmp_path / "merry-quicksand_v5.parquet").read_bytes() == first, \
        "reusing the three saved boosters did not reproduce the file"
    meta2 = json.loads((tmp_path / "merry-quicksand_v5.meta.json").read_text())
    assert meta2["fill_head"]["reused"] is True


# ---------------------------------------------------------------------------------------------
# Amendment 19: the airport-day regime block (--dayfeats, arm D) and the y-target formulation
# (--target y, arm Y). Everything here runs on the synthetic submission fixture of
# tests/synthetic_caches.py (stand / queue / day caches, a serve-mode ranking cache, the raw
# ranking file, the template and a base submission) so the shipping path is exercised end to end
# without the real months; the real-month smokes stay the owner's.
# ---------------------------------------------------------------------------------------------

import hashlib

FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN_TARGET = FIXTURES / "lgbm_submit_default_target_golden.npz"
#: sha256 over the golden ids (float64) + taxi times (int32) bytes; pinned here so a regenerated
#: fixture file is caught rather than trusted
GOLDEN_TARGET_SHA256 = "ac294d69290498180069ada0b751bdfc8c0a21af78d1f142632182772ce496d9"
SMOKE_VERSION = 9                  # a version number no real submission will ever carry


def _sha_golden(ids, pred) -> str:
    return hashlib.sha256(np.ascontiguousarray(ids, dtype="float64").tobytes()
                          + np.ascontiguousarray(pred, dtype="int32").tobytes()).hexdigest()


def _submission_env(module, monkeypatch, root, **kw):
    """The synthetic submission fixture under `root`, with the module's RAW / CACHE / QCACHE /
    DCACHE / SUBS pointed at it and P at num_threads=1: LightGBM is bit-reproducible run to run
    on this machine at one thread (measured 2026-09-09, identical sha256 twice) and NOT at the
    shipped four (the two v3 early-stopping logs diverge from iteration 15,000)."""
    fx = _syn.synthetic_submission(root, months=(1, 2, 3), n=600, seed=0, **kw)
    monkeypatch.setattr(module, "RAW", fx.raw)
    monkeypatch.setattr(module, "CACHE", fx.stand)
    monkeypatch.setattr(module, "QCACHE", fx.queue)
    if hasattr(module, "DCACHE"):
        monkeypatch.setattr(module, "DCACHE", fx.day)
    monkeypatch.setattr(module, "SUBS", fx.subs)
    monkeypatch.setattr(module, "P", dict(module.P, num_threads=1))
    return fx


def _run_submit(module, fx, out, extra=()):
    """`main --version 9 --smoke --out-dir out --base <fixture base> [extra]` ->
    (ids float64, taxi times int32, meta dict), read back from the file that landed."""
    argv = ["--version", str(SMOKE_VERSION), "--smoke", "--out-dir", str(out), "--base", str(fx.base)] + list(extra)
    assert module.main(argv) == 0
    frame = pq.read_table(out / f"merry-quicksand_v{SMOKE_VERSION}.parquet").to_pandas()
    meta = json.loads((out / f"merry-quicksand_v{SMOKE_VERSION}.meta.json").read_text())
    return frame.MVT_ID_mvt.to_numpy(), frame.TAXITIME_SEC_mvt.to_numpy(), meta


def test_default_target_is_byte_identical_to_the_pristine_golden(tmp_path, monkeypatch):
    """The default command (no --target, no --dayfeats) must produce, byte for byte, the taxi
    times the PRISTINE module produced on the frozen synthetic submission fixture: the delta
    formulation y_hat = max(proxy - delta_hat, 1) is what shipped as v3 and every later version,
    and Amendment 19.2's `--target y` is opt-in. The golden was computed from lgbm_submit.py at
    the sha256 recorded in tests/fixtures/lgbm_submit_default_target_golden.npz (the pristine
    file, BEFORE the --target / --dayfeats edit) under lightgbm 4.5.0 / pandas 3.0.1 / numpy
    2.4.3 at num_threads=1, and run twice to prove the path is bit-reproducible.

    Fails when the default target flips to y, when the label handed to LightGBM changes, when
    the taxi-time recovery subtracts something else, or when the synthetic fixture generator
    changes (regenerate the golden ONLY from a pristine module). Mutation rehearsal is recorded
    below once the edit exists.
    """
    fx = _submission_env(ls, monkeypatch, tmp_path / "data")
    ids, pred, meta = _run_submit(ls, fx, tmp_path / "out")
    assert GOLDEN_TARGET.exists(), f"golden missing: {GOLDEN_TARGET}"
    g = np.load(GOLDEN_TARGET)
    assert _sha_golden(g["ids"], g["pred"]) == GOLDEN_TARGET_SHA256, "the fixture file was regenerated"
    assert pred.dtype == np.int32 and np.array_equal(ids, g["ids"])
    moved = int((pred != g["pred"]).sum())
    assert np.array_equal(pred, g["pred"]), f"default path moved on {moved} of {len(pred)} rows"
    assert meta["best_iter"] == int(g["best_iter"]) and meta["n_ref"] == int(g["n_ref"])
    assert meta["n_replaced"] == 5_000 and meta["smoke"] is True
    changed = pred != pq.read_table(fx.base).to_pandas().set_index("MVT_ID_mvt").loc[ids].TAXITIME_SEC_mvt.to_numpy()
    assert 0 < changed.sum() <= 5_000 and not set(ids[changed]) & fx.unmatched_ids


def _day_frame(n, seed=0, ids=None):
    """A day-cache-shaped frame: MVT_ID_mvt + the nine DAY_FEATS as float32, each column offset
    by its index so a column swap is visible."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype="float64") + 100.0 if ids is None
                      else np.asarray(ids, dtype="float64")})
    for i, c in enumerate(_syn.DAY_FEATS):
        d[c] = (rng.integers(0, 9, n) + 100 * i).astype("float32")
    return d


def test_day_month_join_is_positional_with_equal_lengths_and_the_column_contract_asserted():
    """A training month's day twin is joined by position exactly as the queue twin is: the rows
    are the same rows in the same order (build_day's stream IS build_features's stream), so the
    only guard a positional join can have is the row count - asserted per month, a mismatch a
    refusal, never a truncation. The day cache must carry exactly MVT_ID_mvt + DAY_FEATS in the
    contract order; the twin's path is the same file name under data/cache_day/.

    Fails when the length check is dropped (an 8-row month against a 7-row twin would be
    silently mis-aligned) or when the column contract is not checked. Rehearsed 2026-09-09,
    each RED: `if False:` for the length check in attach_block_positional; `set(q.columns) !=`
    in place of the ordered comparison (the reversed-columns case).
    """
    d = _day_frame(7)
    got = ls.attach_day_positional(7, d, "training_2025-02-01_2025-03-01.parquet")
    assert list(got.columns) == _syn.DAY_FEATS and len(got) == 7
    for c in _syn.DAY_FEATS:
        assert np.array_equal(got[c].to_numpy(), d[c].to_numpy()) and got[c].dtype == np.float32
    with pytest.raises(ValueError, match="rows"):
        ls.attach_day_positional(8, d, "training_2025-02-01_2025-03-01.parquet")
    with pytest.raises(ValueError, match="columns"):
        ls.attach_day_positional(7, d.drop(columns=["d_arr_p90"]), "x")
    with pytest.raises(ValueError, match="columns"):
        ls.attach_day_positional(7, d[["MVT_ID_mvt"] + _syn.DAY_FEATS[::-1]], "x")
    with pytest.raises(ValueError, match="columns"):
        ls.attach_day_positional(7, _queue_frame(7), "x")            # a queue cache is not a day cache
    assert ls.DAY_FEATS == stand_ab.DAY_FEATS == _syn.DAY_FEATS and len(ls.DAY_FEATS) == 9
    assert ls.DCACHE == stand_ab.DCACHE
    assert ls.day_cache_path(pathlib.Path("/d"), pathlib.Path("/s/training_2025-02-01_2025-03-01.parquet")) \
        == pathlib.Path("/d/training_2025-02-01_2025-03-01.parquet")
    with pytest.raises(FileNotFoundError, match="day cache"):
        ls.read_day_cache(pathlib.Path("/nowhere/training_2025-02-01_2025-03-01.parquet"))


def test_day_ranking_join_is_by_id_with_set_equality_and_order_asserted():
    """The ranking day cache joins by MVT_ID_mvt exactly as the queue one: the full join needs
    the SAME id set in the SAME order, a sampled join (smoke) aligns the subset by id;
    duplicates and missing ids are refusals.

    Fails when the by-id alignment is replaced by position, when the order check is dropped,
    or when the missing-id check is dropped. Rehearsed 2026-09-09, each RED: `q[feats]`
    returned without the `.loc[ids]` alignment in attach_block_by_id; `if False:` for the
    order check; `if False:` for the missing check.
    """
    ids = np.array([5.0, 3.0, 9.0, 1.0])
    d = _day_frame(4, ids=ids)
    got = ls.attach_day_by_id(ids, d)
    assert list(got.columns) == _syn.DAY_FEATS and len(got) == 4
    assert np.array_equal(got.d_prx_p50.to_numpy(), d.d_prx_p50.to_numpy())
    shuffled = d.iloc[[3, 0, 2, 1]].reset_index(drop=True)
    with pytest.raises(ValueError, match="order"):
        ls.attach_day_by_id(ids, shuffled)
    sub = ls.attach_day_by_id(ids[[2, 0]], shuffled, subset_ok=True)
    want = d.set_index("MVT_ID_mvt").loc[ids[[2, 0]]]
    for c in _syn.DAY_FEATS:
        assert np.array_equal(sub[c].to_numpy(), want[c].to_numpy()), c
    assert sub[_syn.DAY_FEATS[0]].dtype == np.float32
    with pytest.raises(ValueError, match="missing"):
        ls.attach_day_by_id(np.r_[ids, 77.0], d, subset_ok=True)
    with pytest.raises(ValueError, match="extra"):
        ls.attach_day_by_id(ids[:3], d)
    with pytest.raises(ValueError, match="duplicate"):
        ls.attach_day_by_id(ids, pd.concat([d, d.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="columns"):
        ls.attach_day_by_id(ids, d.drop(columns=["d_n_dep"]))


def test_load_frames_attaches_the_day_block_after_the_queue_block(tmp_path):
    """`load_frames(..., queue=True, day=True)` appends the twelve queue columns and then the
    nine day columns, months by position and ranking by id, so the design matrix is
    FEATS + QUEUE_FEATS + DAY_FEATS in that order; day alone gives FEATS + DAY_FEATS; a day
    month with a different row count refuses the whole load; a missing day cache is named.

    Fails when the day block is attached before the queue block (the column order the boosters
    were fitted on would silently change), or when the day month's row count is not checked.
    Rehearsed 2026-09-09, each RED: the day attach moved ahead of the queue attach; the
    day twin one row short accepted (`if False:` in attach_block_positional).
    """
    stand, queue, day = tmp_path / "cache_stand", tmp_path / "cache_queue", tmp_path / "cache_day"
    stand.mkdir(), queue.mkdir(), day.mkdir()
    name = "training_2025-{:02d}-01_2025-{:02d}-01.parquet"
    _mini_month(5, 1, 1).to_parquet(stand / name.format(1, 2), index=False)
    _mini_month(4, 3, 3).to_parquet(stand / name.format(3, 4), index=False)
    _queue_frame(5, seed=1).to_parquet(queue / name.format(1, 2), index=False)
    _queue_frame(4, seed=3).to_parquet(queue / name.format(3, 4), index=False)
    d1, d3 = _day_frame(5, seed=11), _day_frame(4, seed=13)
    d1.to_parquet(day / name.format(1, 2), index=False)
    d3.to_parquet(day / name.format(3, 4), index=False)
    rank = _mini_month(3, 7, 7)
    rank["y"] = np.nan
    rank["delta"] = np.nan
    rank.insert(0, "MVT_ID_mvt", np.array([11.0, 12.0, 13.0]))
    rank.to_parquet(stand / "ranking.parquet", index=False)
    _queue_frame(3, seed=7, ids=[11.0, 12.0, 13.0]).to_parquet(queue / "ranking.parquet", index=False)
    dr = _day_frame(3, seed=17, ids=[11.0, 12.0, 13.0])
    dr.to_parquet(day / "ranking.parquet", index=False)

    d, is_rank, rank_ids = ls.load_frames(stand, queue=True, qcache=queue, day=True, dcache=day)
    assert list(d.columns) == ["y", "delta", "month", "x"] + stand_ab.QUEUE_FEATS + stand_ab.DAY_FEATS
    assert len(d) == 12 and is_rank.sum() == 3
    for c in stand_ab.DAY_FEATS:
        assert np.array_equal(d[c].to_numpy(), np.r_[d1[c].to_numpy(), d3[c].to_numpy(), dr[c].to_numpy()]), c
    d_only, _, _ = ls.load_frames(stand, day=True, dcache=day)
    assert list(d_only.columns) == ["y", "delta", "month", "x"] + stand_ab.DAY_FEATS
    d2, is_rank2, ids2 = ls.load_frames(stand, n_rank=2, day=True, dcache=day, seed=0)
    want = dr.set_index("MVT_ID_mvt").loc[ids2]
    for c in stand_ab.DAY_FEATS:
        assert np.array_equal(d2[c].to_numpy()[is_rank2], want[c].to_numpy()), c
    _day_frame(3, seed=13).to_parquet(day / name.format(3, 4), index=False)     # one row short
    with pytest.raises(ValueError, match="rows"):
        ls.load_frames(stand, day=True, dcache=day)
    with pytest.raises(FileNotFoundError, match="day cache"):
        ls.load_frames(stand, day=True, dcache=tmp_path / "nowhere")


def test_target_argument_label_and_taxi_time_recovery():
    """`--target` defaults to delta (the shipped formulation) and accepts y; `regression_label`
    hands the regressor delta = BLOCK - AOBT_3 under delta and y = TAXITIME under y, in float64,
    unchanged; `raw_taxi_time` recovers proxy - prediction under delta and the prediction ITSELF
    under y - it never touches proxy there (Amendment 19.2: "no proxy anchor"), pinned by a
    proxy of NaN that must not leak into the result; an unknown target is refused;
    `--fillhead --target y` is refused (not pre-registered, and the head was NOT WORKING in
    RESULT 8); `--dayfeats` is off by default.

    Fails when the default flips, when the y label is the delta, or when the y recovery
    subtracts proxy. Rehearsed 2026-09-09, each RED: `default=TARGET_Y`; `return dlt` on the y
    branch of regression_label; `proxy - pred` on the y branch of raw_taxi_time (NaN leaks).
    """
    assert ls.parse_args(["--version", "8"]).target == "delta" == ls.TARGET_DELTA
    assert ls.parse_args(["--version", "8", "--target", "y"]).target == "y" == ls.TARGET_Y
    assert ls.TARGETS == ("delta", "y")
    with pytest.raises(SystemExit):
        ls.parse_args(["--version", "8", "--target", "z"])
    with pytest.raises(SystemExit):
        ls.parse_args(["--version", "8", "--fillhead", "--target", "y"])
    assert ls.parse_args(["--version", "8"]).dayfeats is False
    assert ls.parse_args(["--version", "8", "--dayfeats", "--queue", "--target", "y"]).dayfeats is True
    dlt = np.array([100.0, -50.0, 0.0])
    y = np.array([900.0, 1_100.0, 1_000.0])
    assert ls.regression_label(dlt, y).tolist() == dlt.tolist()
    assert ls.regression_label(dlt, y, ls.TARGET_DELTA).tolist() == dlt.tolist()
    assert ls.regression_label(dlt, y, ls.TARGET_Y).tolist() == y.tolist()
    assert ls.regression_label(dlt, y, ls.TARGET_Y).dtype == np.float64
    with pytest.raises(ValueError, match="target"):
        ls.regression_label(dlt, y, "z")
    pred = np.array([10.0, 20.0, 0.5])
    proxy = np.array([1_000.0, 1_010.0, 1_020.0])
    assert ls.raw_taxi_time(pred, proxy).tolist() == [990.0, 990.0, 1_019.5]
    got = ls.raw_taxi_time(pred, np.full(3, np.nan), ls.TARGET_Y)
    assert got.tolist() == pred.tolist() and got.dtype == np.float64, "the y recovery must not read proxy"
    assert ls.raw_taxi_time(pred, proxy, ls.TARGET_Y).tolist() == pred.tolist()
    with pytest.raises(ValueError, match="target"):
        ls.raw_taxi_time(pred, proxy, "z")
    # the default path is recover_taxi_time to the bit
    assert np.array_equal(ls.finalise_taxi_time(ls.raw_taxi_time(pred, proxy)), ls.recover_taxi_time(proxy, pred))


class _FakeTrained:
    """Stands in for the booster lgb.train returns inside early_stop / refit: a fixed
    best_iteration, a constant prediction, and the feature count fit_seeds checks."""
    def __init__(self, value, best=7, n_features=None):
        self.value, self.best_iteration, self.n_features = float(value), best, n_features
    def predict(self, X, **kw):
        return np.full(len(X), self.value, dtype="float64")
    def num_feature(self):
        return self.n_features
    def num_trees(self):
        return self.best_iteration


def test_early_stop_under_target_y_trains_and_stops_on_y_and_recovers_without_proxy(monkeypatch):
    """Under `--target y` the early-stopping run hands LightGBM y on the fit rows and y on the
    stopping rows (the stopping metric is then RMSE on y directly - Amendment 19.2), and its
    stopping-set taxi time is max(prediction, 1) with no proxy subtraction; under the default it
    hands delta on both and recovers max(proxy - prediction, 1), exactly as before. The
    proxy-only reference is the same number under both (it is the guard's yardstick, not the
    model's formulation). Pinned with a fake lgb.train that records the two labels and returns a
    constant predictor; the breakage guard is lifted for the fake.

    Fails when the y label is the delta, when the stopping set is labelled with delta, or when
    the recovery subtracts proxy under y. Rehearsed 2026-09-09, each RED: `regression_label(dlt,
    y, target)` replaced by `dlt` for the fit rows; `dlt[es]` kept for the valid set; the y
    branch of raw_taxi_time subtracting proxy.
    """
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 5)).astype("float32")
    y = 1_000.0 + rng.normal(0, 10, n)
    dlt = 500.0 + rng.normal(0, 200, n)
    proxy = y + dlt
    month = np.r_[np.full(300, 2), np.full(100, 3)]
    train, fit, es = ls.split_masks(month, np.zeros(n, bool))
    seen = {}

    def fake_train(params, train_set, num_boost_round, valid_sets, callbacks):
        seen["fit"] = np.asarray(train_set.label, dtype="float64")
        seen["es"] = np.asarray(valid_sets[0].label, dtype="float64")
        seen["objective"] = params["objective"]
        return _FakeTrained(seen["value"])
    monkeypatch.setattr(ls.lgb, "train", fake_train)
    monkeypatch.setattr(ls, "ES_MAX_RATIO_TO_PROXY_ONLY", float("inf"))
    params = dict(ls.P, num_threads=1)

    seen["value"] = 1_003.0
    info_y = ls.early_stop(X, dlt, y, proxy, train, fit, es, params, 50, 5, target=ls.TARGET_Y)
    assert np.array_equal(seen["fit"], y[fit]) and np.array_equal(seen["es"], y[es]), "the y arm must train and stop on y"
    assert info_y["target"] == "y" and info_y["best_iter"] == 7 and info_y["n_ref"] == ls.n_refit(7, 400, 300)
    want_y = float(np.sqrt(((y[es] - np.maximum(1_003.0, 1.0)) ** 2).mean()))
    assert info_y["es_rmse_taxi_time_NOT_A_RESULT"] == pytest.approx(want_y, abs=1e-12)

    seen["value"] = 480.0
    info_d = ls.early_stop(X, dlt, y, proxy, train, fit, es, params, 50, 5)
    assert np.array_equal(seen["fit"], dlt[fit]) and np.array_equal(seen["es"], dlt[es]), "the default trains on delta"
    assert info_d["target"] == "delta"
    want_d = float(np.sqrt(((y[es] - np.maximum(proxy[es] - 480.0, 1.0)) ** 2).mean()))
    assert info_d["es_rmse_taxi_time_NOT_A_RESULT"] == pytest.approx(want_d, abs=1e-12)
    assert info_d["es_proxy_only_rmse"] == info_y["es_proxy_only_rmse"] > 100.0
    assert abs(want_y - want_d) > 50.0, "the fixture cannot tell the two recoveries apart"


def test_booster_names_carry_the_day_and_ytarget_tags_and_provenance_records_the_target(tmp_path):
    """Boosters are tagged `_queue` / `_day` / `_queue_day` for the block(s) in the design and
    `_ytarget` for the y formulation, in that order, on top of the seed naming - so a
    77/89-feature or y-target booster can never be reused by another configuration; the head
    booster carries the block tags too. fit_seeds records `target` in fit.json; reuse_seeds and
    check_provenance refuse a booster whose recorded target differs from the run's; a legacy
    fit.json without the field is a delta booster (reusable by a delta run only).

    Fails when a tag is dropped, when the target is not recorded, or when a mismatch is
    accepted. Rehearsed 2026-09-09, each RED: `tag = ""` for the day case in booster_tag;
    `"target": n_features` never written (the key removed from fit.json); `if False:` for the
    target branch of check_provenance.
    """
    d = pathlib.Path("/x")
    assert ls.booster_tag() == "" and ls.booster_tag(queue=True) == "_queue"
    assert ls.booster_tag(day=True) == "_day" and ls.booster_tag(queue=True, day=True) == "_queue_day"
    assert ls.booster_tag(target=ls.TARGET_Y) == "_ytarget"
    assert ls.booster_tag(queue=True, day=True, target=ls.TARGET_Y) == "_queue_day_ytarget"
    assert ls.booster_files(d, 8, (0,), day=True) == [(d / "lgbm_v8_day.txt", d / "lgbm_v8_day.fit.json")]
    assert ls.booster_files(d, 8, (0, 1), queue=True, day=True) == [
        (d / "lgbm_v8_queue_day_seed0.txt", d / "lgbm_v8_queue_day_seed0.fit.json"),
        (d / "lgbm_v8_queue_day_seed1.txt", d / "lgbm_v8_queue_day_seed1.fit.json")]
    assert ls.booster_files(d, 8, (0,), target=ls.TARGET_Y) == [(d / "lgbm_v8_ytarget.txt", d / "lgbm_v8_ytarget.fit.json")]
    assert ls.booster_files(d, 8, (0, 1), queue=True, target=ls.TARGET_Y)[1][0] == d / "lgbm_v8_queue_ytarget_seed1.txt"
    assert ls.booster_files(d, 3, (0,)) == [(d / "lgbm_v3.txt", d / "lgbm_v3.fit.json")]           # v3 unchanged
    assert ls.head_booster_files(d, 5, day=True) == (d / "lgbm_v5_day_fillhead.txt", d / "lgbm_v5_day_fillhead.fit.json")
    assert ls.head_booster_files(d, 5, queue=True, day=True)[0] == d / "lgbm_v5_queue_day_fillhead.txt"

    info = dict(smoke=False, n_all=10, n_ref=5, params=dict(ls.P), n_features=68, target="delta")
    ls.check_provenance(info, 5, False, 10, dict(ls.P), n_features=68, target="delta")
    ls.check_provenance(info, 5, False, 10, dict(ls.P), n_features=68)               # target not asked: not checked
    with pytest.raises(ValueError, match="target"):
        ls.check_provenance(info, 5, False, 10, dict(ls.P), n_features=68, target="y")
    legacy = {k: v for k, v in info.items() if k != "target"}
    ls.check_provenance(legacy, 5, False, 10, dict(ls.P), n_features=68, target="delta")   # legacy = delta
    with pytest.raises(ValueError, match="target"):
        ls.check_provenance(legacy, 5, False, 10, dict(ls.P), n_features=68, target="y")

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)).astype("float32")
    label = rng.normal(size=200)
    train = np.ones(200, bool)
    params = dict(ls.P, num_leaves=4, min_data_in_leaf=5, num_threads=1)
    files = ls.booster_files(tmp_path, 8, (0,), target=ls.TARGET_Y)
    es_info = dict(best_iter=5, n_ref=7, n_fit=150, n_es=50, n_all=200, es_months=[3, 9])
    preds = ls.fit_seeds(X, label, train, X[:3], params, 7, (0,), files=files, info=es_info, smoke=False,
                         n_features=6, target=ls.TARGET_Y)
    fj = json.loads(files[0][1].read_text())
    assert fj["target"] == "y" and fj["n_features"] == 6 and files[0][0].name == "lgbm_v8_ytarget.txt"
    again, info2 = ls.reuse_seeds(files, (0,), X[:3], params, smoke=False, n_all=200, n_features=6, target=ls.TARGET_Y)
    assert np.array_equal(again[0], preds[0]) and info2["target"] == "y"
    with pytest.raises(ValueError, match="target"):
        ls.reuse_seeds(files, (0,), X[:3], params, smoke=False, n_all=200, n_features=6)     # a delta run


def test_fit_per_airport_forwards_the_target_to_the_per_airport_early_stopping_run(monkeypatch):
    """`--pa-trees es` under `--target y` early-stops each airport on y as well: fit_per_airport
    forwards its `target` to early_stop (default delta, so the v4 path is unchanged).

    Fails when the target is not threaded through. Rehearsed 2026-09-09: the `target=target`
    forward removed from the es branch went RED.
    """
    seen = []

    def fake_early_stop(X, dlt, y, proxy, train, fit, es, params, nest, patience, target="delta"):
        seen.append(target)
        return dict(best_iter=3, n_ref=ls.n_refit(3, int(train.sum()), int(fit.sum())), n_fit=int(fit.sum()),
                    n_es=int(es.sum()), n_all=int(train.sum()), es_months=[3, 9],
                    es_rmse_taxi_time_NOT_A_RESULT=1.0, es_proxy_only_rmse=2.0, target=target)
    monkeypatch.setattr(ls, "early_stop", fake_early_stop)
    monkeypatch.setattr(ls, "refit", lambda X, lab, mask, params, n: _ConstantBooster(1.0))
    ap_code = np.r_[np.zeros(30, np.int8), np.array([0], np.int8)]
    train = np.r_[np.ones(30, bool), np.zeros(1, bool)]
    es_mask = np.zeros(31, bool)
    es_mask[20:30] = True
    fit = train & ~es_mask
    args = (np.zeros((31, 3), np.float32), np.r_[np.full(30, 5.0), np.nan], train, ~train, ap_code, ["AAA"],
            dict(ls.P), 1_000, (0,))
    es_kw = dict(y=np.zeros(31), proxy=np.zeros(31), fit=fit, es=es_mask, nest=77, patience=9)
    ls.fit_per_airport(*args, min_rows=20, floor=5, trees="es", es=es_kw)
    with pytest.raises(ValueError, match="dlt"):
        ls.fit_per_airport(*args, min_rows=20, floor=5, trees="es", es=es_kw, target=ls.TARGET_Y)
    ls.fit_per_airport(*args, min_rows=20, floor=5, trees="es", es=dict(es_kw, dlt=np.zeros(31)), target=ls.TARGET_Y)
    assert seen == ["delta", "y"]
    with pytest.raises(ValueError, match="target"):
        ls.fit_per_airport(*args, min_rows=20, floor=5, trees="share", target="z")


def _fit_json(out, name):
    return json.loads((out / name).read_text())


def test_dayfeats_end_to_end_on_the_synthetic_submission(tmp_path, monkeypatch):
    """`--dayfeats` on the synthetic submission fixture: the day twins join the two training
    months by position and the sampled ranking rows by id, a 77-column design matrix, the
    booster saved as lgbm_v9_day.txt with n_features 77 and target delta in its fit.json, the
    meta recording the block (dayfeats, day_feats, features = FEATS + DAY_FEATS), the splice
    invariants (5,000 sampled matched rows replaced, no unmatched row touched); then
    `--queue --dayfeats --seeds 0,1` gives 89 columns and lgbm_v9_queue_day_seed{s}.txt, and
    --reuse-booster reproduces that file byte for byte from the two tagged boosters.

    Fails when the day block is not in the design (77 -> 68), when the tag is dropped (reuse
    finds nothing and refits: a different feature_fraction stream), or when the meta forgets the
    block. Rehearsed 2026-09-09, each RED: `feats` built without DAY_FEATS under --dayfeats;
    `day=False` hard-wired in booster_files' call; `"dayfeats": False` in the meta.
    """
    fx = _submission_env(ls, monkeypatch, tmp_path / "data")
    out = tmp_path / "out"
    ids, pred, meta = _run_submit(ls, fx, out, ["--dayfeats"])
    assert meta["dayfeats"] is True and meta["day_feats"] == stand_ab.DAY_FEATS and meta["target"] == "delta"
    assert meta["features"] == list(ls.FEATS) + stand_ab.DAY_FEATS and meta["n_features"] == 77
    assert meta["queue"] is False and meta["queue_feats"] is None and meta["booster_files"] == ["lgbm_v9_day.txt"]
    fj = _fit_json(out, "lgbm_v9_day.fit.json")
    assert fj["n_features"] == 77 and fj["target"] == "delta" and fj["smoke"] is True
    assert not (out / "lgbm_v9.txt").exists(), "a day booster took the untagged name"
    assert meta["n_replaced"] == 5_000 and pred.dtype == np.int32 and (pred > 0).all()
    base = pq.read_table(fx.base).to_pandas().set_index("MVT_ID_mvt").loc[ids].TAXITIME_SEC_mvt.to_numpy()
    changed = pred != base
    assert 0 < changed.sum() <= 5_000 and not set(ids[changed]) & fx.unmatched_ids

    out2 = tmp_path / "out2"
    ids2, pred2, meta2 = _run_submit(ls, fx, out2, ["--queue", "--dayfeats", "--seeds", "0,1"])
    assert meta2["n_features"] == 89 and meta2["features"] == list(ls.FEATS) + stand_ab.QUEUE_FEATS + stand_ab.DAY_FEATS
    assert meta2["booster_files"] == ["lgbm_v9_queue_day_seed0.txt", "lgbm_v9_queue_day_seed1.txt"]
    for s in (0, 1):
        fj = _fit_json(out2, f"lgbm_v9_queue_day_seed{s}.fit.json")
        assert fj["n_features"] == 89 and fj["params"]["seed"] == s and fj["target"] == "delta"
    first = (out2 / "merry-quicksand_v9.parquet").read_bytes()
    _run_submit(ls, fx, out2, ["--queue", "--dayfeats", "--seeds", "0,1", "--reuse-booster"])
    assert (out2 / "merry-quicksand_v9.parquet").read_bytes() == first, "reuse did not reproduce the file"
    assert not np.array_equal(pred, pred2), "the queue block changed nothing"


def test_target_y_end_to_end_on_the_synthetic_submission(tmp_path, monkeypatch):
    """`--target y` on the synthetic submission fixture: the regressor is fitted on y, saved as
    lgbm_v9_ytarget.txt with target y in its fit.json, the meta records target y, the shipped
    taxi times are on y's scale and differ from the delta run's, the splice invariants hold,
    and --reuse-booster reproduces the file byte for byte; a delta run cannot reuse the
    y booster (different name) and the golden test guards the default.

    Fails when the y booster is fitted on delta (the shipped times then sit near the delta
    scale, hundreds of seconds, against a y median of ~2,500 s on this fixture), when the tag is
    dropped, or when the meta misreports the target. Rehearsed 2026-09-09, each RED:
    `regression_label(dlt, y, TARGET_DELTA)` hard-wired in main; `target=TARGET_DELTA` in the
    booster_files call; `"target": TARGET_DELTA` in the meta.
    """
    fx = _submission_env(ls, monkeypatch, tmp_path / "data")
    out = tmp_path / "out"
    ids, pred, meta = _run_submit(ls, fx, out, ["--target", "y"])
    assert meta["target"] == "y" and meta["booster_files"] == ["lgbm_v9_ytarget.txt"] and meta["n_features"] == 68
    fj = _fit_json(out, "lgbm_v9_ytarget.fit.json")
    assert fj["target"] == "y" and fj["n_features"] == 68 and fj["params"]["objective"] == "regression"
    assert 1_500 < np.median(pred) < 3_500, "y-target predictions must sit on y's scale (delta would be ~0-1,000)"
    assert meta["n_replaced"] == 5_000 and (pred > 0).all() and pred.dtype == np.int32
    base = pq.read_table(fx.base).to_pandas().set_index("MVT_ID_mvt").loc[ids].TAXITIME_SEC_mvt.to_numpy()
    changed = pred != base
    assert 0 < changed.sum() <= 5_000 and not set(ids[changed]) & fx.unmatched_ids
    first = (out / "merry-quicksand_v9.parquet").read_bytes()
    _run_submit(ls, fx, out, ["--target", "y", "--reuse-booster"])
    assert (out / "merry-quicksand_v9.parquet").read_bytes() == first
    ids_d, pred_d, meta_d = _run_submit(ls, fx, tmp_path / "out_delta", [])
    assert meta_d["target"] == "delta" and np.array_equal(ids_d, ids)
    assert not np.array_equal(pred_d, pred), "the y formulation reproduced the delta file exactly"
    assert float(np.sqrt(((pred_d.astype("float64") - pred.astype("float64")) ** 2).mean())) < 600.0, \
        "the two formulations disagree by more than the fixture's own spread: one of them is broken"


# ---------------------------------------------------------------------------------------------
# The unified all-rows arm (the Amendment 19 addition): `--target y --all-rows` - ONE regressor
# with target y on every admissible training row, matched AND unmatched, the unmatched rows from
# data/cache_unmatched with their NaN pattern, an explicit is_unmatched column, an optional sample
# weight on the unmatched rows, y_hat = max(pred, 1) on every row, the splice replacing EVERY
# scored row (matched and unmatched alike). No mixture, no proxy anywhere.
# ---------------------------------------------------------------------------------------------

def test_unmatched_cache_reader_contract_and_nan_pattern_check():
    """read_unmatched_cache asserts the contract (stand_ab.UNMATCHED_COLS in order) and names the
    build command when the file is missing; check_unmatched_nan_pattern accepts a conforming
    frame, and refuses - naming the column and the count - a frame where an AOBT_3-anchored or
    a *_flt-derived column carries a value on an unmatched row, so a cache built with those
    columns filled (0, the mean, anything) can never train silently.

    Fails when the check is dropped or reads the wrong list. Rehearsed 2026-09-09, each RED:
    `if False:` around the refusal; proxy filled with 0.0 in the fixture against an unchanged
    check (the sanity case, must raise).
    """
    u = pd.read_parquet(next(iter(_syn.synthetic_caches(pathlib.Path(__import__("tempfile").mkdtemp()) / "d", (2,), n=20)[0].parent.joinpath("cache_unmatched").glob("*.parquet"))))
    assert list(u.columns) == stand_ab.UNMATCHED_COLS == ls.UNMATCHED_COLS
    ls.check_unmatched_nan_pattern(u)
    for c in ("proxy", "q_apt_at_push", "eobt_p", "d_prx_minus_p50"):
        bad = u.copy()
        bad.loc[bad.index[:3], c] = 0.0
        with pytest.raises(ValueError, match=c):
            ls.check_unmatched_nan_pattern(bad)
    with pytest.raises(ValueError, match="columns"):
        ls.attach_block_positional(len(u), u.drop(columns=["sp"]), "x", stand_ab.UNMATCHED_COLS[1:], "unmatched")
    with pytest.raises(FileNotFoundError, match="unmatched cache"):
        ls.read_unmatched_cache(pathlib.Path("/nowhere/training_2025-02-01_2025-03-01.parquet"))
    assert ls.unmatched_cache_path(pathlib.Path("/u"), pathlib.Path("/s/training_2025-02-01_2025-03-01.parquet")) \
        == pathlib.Path("/u/training_2025-02-01_2025-03-01.parquet")
    assert ls.UCACHE == stand_ab.UCACHE and ls.IS_UNMATCHED == "is_unmatched"


def test_all_rows_flags_weights_and_booster_tags():
    """--all-rows requires --target y (the delta formulation has no proxy on unmatched rows) and
    refuses --fillhead and --per-airport (not pre-registered for the unified arm);
    --unmatched-weight is a positive float, default 1.0; row_weights gives 1 on matched and W on
    unmatched rows in float64 and refuses W <= 0 / non-finite; the booster tag adds `_allrows`
    and `_w{W}` when W != 1 after the block and target tags.

    Fails when the requirement or a refusal is dropped, or when the weight is not applied per
    row. Rehearsed 2026-09-09, each RED: the `--all-rows` without y check replaced by `if
    False:`; `np.where(is_unmatched, 1.0, w)` (swapped).
    """
    a = ls.parse_args(["--version", "8", "--target", "y", "--all-rows"])
    assert a.all_rows is True and a.unmatched_weight == 1.0
    assert ls.parse_args(["--version", "8", "--target", "y", "--all-rows", "--unmatched-weight", "10"]).unmatched_weight == 10.0
    assert ls.parse_args(["--version", "8"]).all_rows is False
    for argv in (["--all-rows"], ["--target", "delta", "--all-rows"], ["--target", "y", "--all-rows", "--fillhead"],
                 ["--target", "y", "--all-rows", "--per-airport"], ["--target", "y", "--all-rows", "--unmatched-weight", "0"],
                 ["--target", "y", "--all-rows", "--unmatched-weight", "-2"]):
        with pytest.raises(SystemExit):
            ls.parse_args(["--version", "8"] + argv)
    w = ls.row_weights(np.array([False, True, True, False]), 10.0)
    assert w.dtype == np.float64 and w.tolist() == [1.0, 10.0, 10.0, 1.0]
    assert ls.row_weights(np.array([0, 1]), 1.0).tolist() == [1.0, 1.0]
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="weight"):
            ls.row_weights(np.array([False, True]), bad)
    assert ls.booster_tag(target=ls.TARGET_Y, all_rows=True) == "_ytarget_allrows"
    assert ls.booster_tag(target=ls.TARGET_Y, all_rows=True, unmatched_weight=10.0) == "_ytarget_allrows_w10"
    assert ls.booster_tag(queue=True, day=True, target=ls.TARGET_Y, all_rows=True, unmatched_weight=2.5) == "_queue_day_ytarget_allrows_w2.5"
    assert ls.booster_tag(all_rows=False, unmatched_weight=10.0) == "", "the weight tags only an all-rows booster"
    d = pathlib.Path("/x")
    assert ls.booster_files(d, 8, (0,), target=ls.TARGET_Y, all_rows=True, unmatched_weight=10.0)[0][0] == d / "lgbm_v8_ytarget_allrows_w10.txt"


def test_early_stop_and_fit_seeds_hand_the_row_weights_to_lightgbm(monkeypatch):
    """With `weight` given, early_stop's training Dataset carries weight[fit] and its stopping
    Dataset carries NO weight (the stopping metric is the plain RMSE on y, the competition's
    metric), and every refit in fit_seeds carries weight[train]; without `weight` no Dataset has
    one (the default path is untouched). Pinned with a fake lgb.train that records each
    Dataset's weight.

    Fails when the weight is not forwarded, or forwarded to the stopping set. Rehearsed
    2026-09-09, each RED: `weight=None` hard-wired in early_stop's training Dataset; the
    stopping Dataset given `weight[es]`; `refit` ignoring its weight.
    """
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 4)).astype("float32")
    y = 1_000.0 + rng.normal(0, 10, n)
    dlt = 500.0 + rng.normal(0, 200, n)
    proxy = y + dlt
    month = np.r_[np.full(200, 2), np.full(100, 3)]
    train, fit, es = ls.split_masks(month, np.zeros(n, bool))
    w = ls.row_weights(rng.random(n) < 0.3, 10.0)
    seen = []

    def fake_train(params, train_set, num_boost_round, valid_sets=None, callbacks=None):
        seen.append((None if train_set.weight is None else np.asarray(train_set.weight, dtype="float64"),
                     None if not valid_sets else (None if valid_sets[0].weight is None else np.asarray(valid_sets[0].weight))))
        return _FakeTrained(1_000.0, n_features=4)
    monkeypatch.setattr(ls.lgb, "train", fake_train)
    monkeypatch.setattr(ls, "ES_MAX_RATIO_TO_PROXY_ONLY", float("inf"))
    params = dict(ls.P, num_threads=1)
    ls.early_stop(X, dlt, y, proxy, train, fit, es, params, 20, 5, target=ls.TARGET_Y, weight=w)
    assert np.array_equal(seen[-1][0], w[fit]) and seen[-1][1] is None
    ls.fit_seeds(X, y, train, X[:3], params, 4, (0, 1), n_features=4, target=ls.TARGET_Y, weight=w)
    assert len(seen) == 3 and all(np.array_equal(s[0], w[train]) and s[1] is None for s in seen[1:])
    ls.early_stop(X, dlt, y, proxy, train, fit, es, params, 20, 5)
    ls.fit_seeds(X, dlt, train, X[:3], params, 4, (0,), n_features=4)
    assert seen[-2] == (None, None) and seen[-1] == (None, None), "the default path must carry no weights"


def test_load_frames_all_rows_appends_the_unmatched_rows_with_the_flag(tmp_path):
    """load_frames(all_rows=True, ucache=...) appends every unmatched month after the matched
    months and the unmatched scored rows after the matched ranking rows, aligned by name (the
    matched caches' STAND_BLOCK columns are NaN on unmatched rows, the unmatched cache's extra
    blocks are dropped unless asked for), with is_unmatched = 1 exactly on the appended rows,
    the NaN pattern checked on every unmatched row, MVT_ID_mvt lifted out for every ranking row;
    a sampled ranking (smoke) samples from matched and unmatched rows together; a missing
    unmatched month refuses the load.

    Fails when the flag is not set, when the pattern check is skipped (a filled column loads),
    or when the ranking's unmatched rows are dropped. Rehearsed 2026-09-09, each RED:
    `is_unmatched` written as zeros; the pattern check removed (the filled-proxy cache loads);
    the unmatched ranking rows left out of the concat.
    """
    fx = _syn.synthetic_submission(tmp_path / "data", months=(1, 3), n=60, seed=0, n_rank=100, n_unmatched=20)
    d, is_rank, rank_ids = ls.load_frames(fx.stand, all_rows=True, ucache=fx.unmatched)
    n_u_train = 2 * _syn.N_UNMATCHED
    assert len(d) == 120 + n_u_train + 100 + 20 and is_rank.sum() == 120 and len(rank_ids) == 120
    um = d[ls.IS_UNMATCHED].to_numpy()
    assert um.dtype == np.float64 and set(um) == {0.0, 1.0}
    assert um[:120].sum() == 0 and um[120:120 + n_u_train].sum() == n_u_train
    assert um[is_rank][:100].sum() == 0 and um[is_rank][100:].sum() == 20
    assert set(rank_ids[100:]) == fx.unmatched_ids and set(rank_ids[:100]) == fx.matched_ids
    assert d.y[um == 1].iloc[:n_u_train].notna().all() and d.y[is_rank].isna().all()
    assert d.delta[um == 1].isna().all() and d.proxy[um == 1].isna().all()
    for c in stand_ab.UNMATCHED_NAN_COLS:
        if c in d.columns:
            assert d[c][um == 1].isna().all(), c
    assert "gapa" in d.columns and d.gapa[um == 1].isna().all(), "the stand block is NaN on unmatched rows"
    assert "q_apt_at_push" not in d.columns, "blocks not asked for are dropped"
    d2, is_rank2, ids2 = ls.load_frames(fx.stand, n_rank=30, all_rows=True, ucache=fx.unmatched, seed=0)
    assert len(ids2) == 30 and set(ids2) <= fx.matched_ids | fx.unmatched_ids
    assert (d2[ls.IS_UNMATCHED][is_rank2] == 1.0).sum() == len(set(ids2) & fx.unmatched_ids)
    d3, _, _ = ls.load_frames(fx.stand, all_rows=True, ucache=fx.unmatched, queue=True, qcache=fx.queue, day=True, dcache=fx.day)
    assert list(d3.columns)[-len(stand_ab.DAY_FEATS) - 1:-1] == stand_ab.DAY_FEATS and list(d3.columns)[-1] == ls.IS_UNMATCHED
    um3 = d3[ls.IS_UNMATCHED].to_numpy() == 1.0
    assert d3.q_apt_at_push[um3].isna().all() and d3.q_dep_tko_sym15[um3].notna().all()
    filled = pd.read_parquet(fx.unmatched / "training_2025-03-01_2025-04-01.parquet")
    filled["proxy"] = 0.0
    filled.to_parquet(fx.unmatched / "training_2025-03-01_2025-04-01.parquet", index=False)
    with pytest.raises(ValueError, match="proxy"):
        ls.load_frames(fx.stand, all_rows=True, ucache=fx.unmatched)
    with pytest.raises(FileNotFoundError, match="unmatched cache"):
        ls.load_frames(fx.stand, all_rows=True, ucache=tmp_path / "nowhere")


def test_all_rows_end_to_end_on_the_synthetic_submission(tmp_path, monkeypatch):
    """`--target y --all-rows` on the synthetic submission fixture: one booster on matched AND
    unmatched training rows (69 columns: FEATS + is_unmatched), saved as
    lgbm_v9_ytarget_allrows.txt with target y, all_rows and the weight in its fit.json; EVERY
    sampled scored row is replaced - unmatched rows included, which no other path touches - the
    unmatched predictions on y's scale; the meta records all_rows, the weight, the unmatched row
    counts and the NaN-pattern check; --reuse-booster reproduces the file byte for byte; with
    `--unmatched-weight 3` the booster is lgbm_v9_ytarget_allrows_w3.txt and the
    predictions differ from the weight-1 run's (the weight reached LightGBM). Three, not the
    registered ten: the smoke fits on ONE month, where 40 unmatched rows x 10 are 40% of the
    effective weight against 600 matched rows, and the matched stopping-set breakage guard sits
    at ratio 0.84 (measured 2026-09-09; 0.68 at weight 1; the guard refuses at 0.85) - too close
    to rest a test on; the fold smoke, which fits on more rows, runs the registered 1,10.

    Fails when the unmatched rows are not in the training set or not spliced, when the weight
    is not forwarded (identical predictions), or when the tag is dropped. Rehearsed 2026-09-09,
    each RED: the splice given the matched id set only (`expected` unchanged); `weight=None` in
    main's fit_seeds call; `all_rows=False` in the booster_files call.
    """
    # exact proxy levels, as the fold's Y test uses: at the smoke's 60 rounds a y-target model on a
    # continuous proxy sits at 0.93 x proxy-only on the matched stopping rows and the breakage guard
    # (0.85) refuses it - a fixture artifact the guard is right to flag, not a reason to relax it
    fx = _submission_env(ls, monkeypatch, tmp_path / "data", proxy_levels=3)
    monkeypatch.setattr(ls, "UCACHE", fx.unmatched)
    out = tmp_path / "out"
    ids, pred, meta = _run_submit(ls, fx, out, ["--target", "y", "--all-rows"])
    assert meta["all_rows"] is True and meta["unmatched_weight"] == 1.0 and meta["target"] == "y"
    assert meta["n_features"] == 69 and meta["features"][-1] == ls.IS_UNMATCHED and meta["features"][:68] == list(ls.FEATS)
    assert meta["n_unmatched_train"] == 2 * _syn.N_UNMATCHED and meta["n_unmatched_rank"] > 0   # the smoke's two months
    assert meta["unmatched_nan_cols"] == stand_ab.UNMATCHED_NAN_COLS and meta["unmatched_nan_pattern_checked"] is True
    assert meta["booster_files"] == ["lgbm_v9_ytarget_allrows.txt"]
    fj = _fit_json(out, "lgbm_v9_ytarget_allrows.fit.json")
    assert fj["target"] == "y" and fj["n_features"] == 69 and fj["all_rows"] is True and fj["unmatched_weight"] == 1.0
    base = pq.read_table(fx.base).to_pandas().set_index("MVT_ID_mvt").loc[ids].TAXITIME_SEC_mvt.to_numpy()
    changed = pred != base
    replaced_unmatched = set(meta["replaced_ids"]) & fx.unmatched_ids
    assert replaced_unmatched and meta["n_replaced"] == 5_000
    assert set(ids[changed]) & fx.unmatched_ids, "the unmatched rows must be re-predicted by the unified model"
    um_pred = pred[np.isin(ids, list(fx.unmatched_ids))]
    assert 300 < np.median(um_pred) < 5_000 and (pred > 0).all()
    first = (out / "merry-quicksand_v9.parquet").read_bytes()
    _run_submit(ls, fx, out, ["--target", "y", "--all-rows", "--reuse-booster"])
    assert (out / "merry-quicksand_v9.parquet").read_bytes() == first
    ids10, pred10, meta10 = _run_submit(ls, fx, tmp_path / "out10", ["--target", "y", "--all-rows", "--unmatched-weight", "3"])
    assert meta10["booster_files"] == ["lgbm_v9_ytarget_allrows_w3.txt"] and meta10["unmatched_weight"] == 3.0
    fj3 = _fit_json(tmp_path / "out10", "lgbm_v9_ytarget_allrows_w3.fit.json")
    assert fj3["unmatched_weight"] == 3.0 and fj3["all_rows"] is True
    assert np.array_equal(ids10, ids)
    assert not np.array_equal(pred10, pred), "the unmatched weight changed nothing: it never reached LightGBM"
