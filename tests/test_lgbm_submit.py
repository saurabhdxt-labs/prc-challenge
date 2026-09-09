"""`scripts/lgbm_submit.py` ships the measured LightGBM config on the matched rows and must
change NOTHING else in the submission.

The point of v3 is a clean board measurement of one lane: matched rows re-predicted, the
5,290 unmatched rows copied from v2 byte for byte. A splice that touches an unmatched row,
misses a matched one, or re-orders identifiers would make the board delta uninterpretable
and spend a submission slot on noise. The refit procedure must be the one lgbm_ab measured
(226.24 on the fold), so its arithmetic is pinned to the numbers that run printed.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

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
