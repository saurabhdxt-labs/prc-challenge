"""FM1 feature assembly (prc/pipeline/fm1/features.py): the design equals the shipped designs it claims to be, encodings
equal stand_ab.infold_encodings bit-for-bit, routing covers every scored departure, and no label column reaches a design.
Real-data-shaped: the e2e world is a slice of the real 2025 raw files + real METAR + real R12 ADS-B rows."""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import features as F
from prc.pipeline.fm1 import knobs as K

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name, rel):
    import sys
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cfg(e2e_world, fm1_world_mod):
    """The provisional `hgb` body's frames are what these tests pin (build_frames builds them only for that learner; the
    v26 lane derives its own rows, tests/fm1/test_fm1_unmatched.py)."""
    return K.parse_config(fm1_world_mod.config_raw(e2e_world[0], **{"unmatched.learner": "hgb"}))


@pytest.fixture(scope="module")
def train_raw(e2e_world):
    return pd.read_parquet(e2e_world[0] / "raw" / "training_2025-01-01_2025-02-01.parquet")


@pytest.fixture(scope="module")
def weather(cfg):
    return F.stand_ab().load_weather(cfg.path("data.weather_dir"))


# ---------------------------------------------------------------------------------------------- design = shipped designs

@pytest.mark.slow          # > 1 s (rule 26): real fits / a worker process / loading lgbm_fold
def test_base_columns_equal_nsc_fold_fw():
    """FEATS_NSC_FW (v34's FW design without the seconds columns). Mutation: keep the `*_sec` columns -> red."""
    nsc = _load("nsc_fold", "scripts/analysis/nsc_fold.py")
    assert F.base_columns(True, True, True) == list(nsc.FEATS_NSC_FW)
    assert F.base_columns(True, True, False) == list(nsc.FEATS_NSC_F)


@pytest.mark.slow          # > 1 s (rule 26): loads nsc_fold -> lgbm_fold -> lgbm_submit
def test_matched_design_without_orl_equals_the_v36_r12_refit_design():
    """The R12 refit that v36 served (data/r12/refit_801f9bcac9a5_full/report.json): FEATS_NSC_FW (90) + its recorded
    adsb_cols (35) = 125 features. Mutation: put ADSB_EXTRA before the table features -> order differs -> red."""
    rep = json.loads((ROOT / "data/r12/refit_801f9bcac9a5_full/report.json").read_text())
    nsc = _load("nsc_fold", "scripts/analysis/nsc_fold.py")
    cfg = K.parse_config({"features": {"orl_route_length": False}})
    cols = F.matched_design_columns(cfg)
    assert cols == list(nsc.FEATS_NSC_FW) + rep["adsb_cols"]
    assert len(cols) == rep["fits"]["r12"]["n_features"] == 125


def test_adsb_block_is_defined_by_the_table():
    """801f9bcac9a5 = the 33 features the v36 stack was trained on; 51348d3e1cb3 (DWG) adds stand_visible and
    stand_ref_parked_share, as r12_build.FEATURES now does."""
    man = json.loads((ROOT / "data/r12/stack_801f9bcac9a5_full/models_r12/MANIFEST.json").read_text())
    assert F.table_features(ROOT / "data/r12/tables_801f9bcac9a5") == man["feats"]
    r12b = _load("r12_build", "scripts/analysis/r12_build.py")
    assert F.table_features(ROOT / "data/r12/tables_51348d3e1cb3") == list(r12b.FEATURES)


def test_table_features_refuses_mismatched_tables(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    keys = {c: [1.0] for c in F.ADSB_TABLE_KEYS}
    pq.write_table(pa.table({**keys, "f1": [0.0]}), tmp_path / "train.parquet")
    pq.write_table(pa.table({**keys, "f2": [0.0]}), tmp_path / "serve.parquet")
    with pytest.raises(E.SchemaError, match="train and serve columns differ"):
        F.table_features(tmp_path)
    pq.write_table(pa.table({"MVT_ID_mvt": [1.0], "f1": [0.0]}), tmp_path / "train.parquet")
    pq.write_table(pa.table({"MVT_ID_mvt": [1.0], "f1": [0.0]}), tmp_path / "serve.parquet")
    with pytest.raises(E.MissingColumnError, match="missing key column"):
        F.table_features(tmp_path)


def test_orl_list_is_the_builders():
    orl = F.orl_build()
    assert F.ORL_COLS == [*orl.ORL_COLS, orl.ORL_FLAG]


def test_no_design_ever_holds_a_label_or_seconds_column():
    """Negative invariant (AT2) over every combination of the feature switches."""
    import itertools
    for q, o, w, r, a in itertools.product([True, False], repeat=5):
        cfg = K.parse_config({"features": {"queue": q, "order": o, "weather": w, "orl_route_length": r, "adsb": a},
                              "adsb": {"stack": {"enabled": a}}})
        cols = F.matched_design_columns(cfg)
        assert not [c for c in cols if c.endswith("_sec") or c in ("y", "delta", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt")]
    assert "y" not in F.unmatched_design_columns()


def test_matched_design_refuses_a_forbidden_column(train_raw, cfg, weather):
    fr = F.matched_frame(train_raw, False, cfg, weather, F.read_orl_routes(cfg.path("features.orl_routes_file")))
    enc = F.EncodingTables.fit(fr, {"te": fr.y, "de": fr.delta}, F.stand_ab().ENC_KEYS, 50.0)
    with pytest.raises(E.ForbiddenInputError, match="delta"):
        F.matched_design(fr, enc, ["proxy", "delta"])


# ---------------------------------------------------------------------------------------------- encodings

def test_encodings_equal_stand_ab_infold_encodings_bit_for_bit(train_raw):
    """The port is verified against the shipped function on real rows. Mutation: use `count + 2 * smooth` -> red."""
    S = F.stand_ab()
    fr = S.build_features(train_raw, serve=False)
    tr = np.ones(len(fr), bool)
    ref = S.infold_encodings(fr, fr.y.to_numpy(), fr.delta.to_numpy(), tr)
    enc = F.EncodingTables.fit(fr, {"te": fr.y.to_numpy(), "de": fr.delta.to_numpy()}, S.ENC_KEYS, S.SMOOTH)
    got = enc.transform(fr)
    assert set(got) == set(ref)
    for k in ref:
        assert np.array_equal(got[k], ref[k]), k


def test_encodings_apply_to_unseen_keys_and_round_trip():
    tr = pd.DataFrame({"k": ["a", "a", "b", None]})
    enc = F.EncodingTables.fit(tr, {"te": np.array([10.0, 20.0, 40.0, 70.0])}, ["k"], smooth=2.0)
    prior = 35.0
    assert enc.priors == {"te": prior}
    new = pd.DataFrame({"k": ["a", "zzz", None]})
    got = enc.transform(new)["te_k"]
    expect = np.array([(30.0 + 2 * prior) / 4, prior, (70.0 + 2 * prior) / 3], dtype=np.float32)   # None -> the "NA" key
    assert np.array_equal(got, expect)
    back = F.EncodingTables.from_json(enc.to_json())
    assert np.array_equal(back.transform(new)["te_k"], got)


def test_encodings_refuse_non_finite_targets():
    with pytest.raises(E.SchemaError, match="non-finite"):
        F.EncodingTables.fit(pd.DataFrame({"k": ["a", "b"]}), {"te": np.array([1.0, np.nan])}, ["k"], 1.0)
    with pytest.raises(E.MissingColumnError, match="encoding key 'q'"):
        F.EncodingTables.fit(pd.DataFrame({"k": ["a"]}), {"te": np.array([1.0])}, ["q"], 1.0)


# ---------------------------------------------------------------------------------------------- frames on real rows

def test_training_frames_on_real_rows(train_raw, cfg, weather):
    orl = F.read_orl_routes(cfg.path("features.orl_routes_file"))
    fr = F.build_frames(train_raw, False, cfg, weather, orl)
    S = F.stand_ab()
    assert len(fr.matched) == len(S.build_features(train_raw, serve=False))
    assert not [c for c in fr.matched.columns if c.endswith("_sec")]
    assert (fr.matched.y > 0).all() and np.allclose(fr.matched.delta, fr.matched.proxy - fr.matched.y)
    for c in F.adsb_columns(cfg.path("adsb.table_dir")) + F.ORL_COLS + list(S.QUEUE_FEATS) + list(S.WEATHER_FEATS):
        assert c in fr.matched.columns, c
    assert fr.matched.coverage.isin([1, 2, 3]).all()
    assert (fr.unmatched.y > 0).all() and len(fr.unmatched) > 0
    assert set(fr.matched.ap) <= {"EDDF", "LSZH"}


def test_regression_bug_30_unmatched_frame_string_concat_on_real_dtypes(train_raw, cfg):
    """BC-30: a real-data chunk whose unmatched-eligible slice is empty (a month-boundary chunk that is all matched
    departures, e.g. the July 2025 raw file's single-row June-spillover chunk) crashed `unmatched_frame` while
    building `ars`: `pd.Series(ap)` (empty, plain-object dtype) concatenated with `out.RUNWAY_mvt` /
    `fill(u.STAND_mvt)` (empty, pandas' pyarrow-backed 'str' dtype from the real parquet read) hits
    pyarrow.lib.ArrowNotImplementedError ('binary_join_element_wise' has no kernel for (null, large_string,
    large_string)), then TypeError('radd' not supported for dtype 'str' with dtype 'object') on fallback.
    Real dtypes matter here (an all-numpy-object synthetic frame does not reproduce it), so this fixture is built
    from the real e2e-world raw parquet, filtered to matched-only rows -> zero rows survive the unmatched filter.
    Mutation: revert the fix (put back the bare `pd.Series(ap) + "|" + out.RUNWAY_mvt + "|" + fill(u.STAND_mvt)`)
    -> this goes red with the exact TypeError above."""
    assert train_raw.RUNWAY_mvt.dtype == "str" and train_raw.STAND_mvt.dtype == "str"          # the real dtype
    chunk = train_raw[train_raw.AOBT_3_flt.notna()].reset_index(drop=True)          # matched-only -> 0 unmatched rows
    assert len(chunk) > 0
    out = F.unmatched_frame(chunk, False, cfg)
    assert len(out) == 0
    full = F.unmatched_frame(train_raw, False, cfg)
    assert list(out.columns) == list(full.columns)
    assert out.ars.dtype == object
    assert len(full) > 0
    fill = lambda s: s.astype(object).where(s.notna(), "NA").astype(str)          # noqa: E731 (mirrors features.py)
    dep = (train_raw.PHASE_mvt.astype(str) == "DEP") & train_raw.AOBT_3_flt.isna() & train_raw.MVT_TIME_UTC_mvt.notna()
    dep &= train_raw.TAXITIME_SEC_mvt.notna() & (train_raw.TAXITIME_SEC_mvt.fillna(0) > 0)
    u = train_raw[dep.to_numpy()].reset_index(drop=True)
    expect = (u.ADEP_mvt.astype(str).astype(object) + "|" + fill(u.RUNWAY_mvt).astype(object) + "|"
              + fill(u.STAND_mvt).astype(object)).to_numpy()
    assert (full.ars.to_numpy() == expect).all()


def test_ars_fills_missing_runway_and_stand_with_NA_not_a_null(train_raw, cfg):
    """The real e2e-world fixture's unmatched rows never happen to carry a null RUNWAY_mvt/STAND_mvt, so
    test_regression_bug_30's value check above cannot see whether an unfilled null leaks into `ars` unchanged
    (it would, in the raw real 'str'/pyarrow dtype, render as the *string* 'nan' via naive str() coercion, not
    'NA'). Null out one real unmatched-eligible row's RUNWAY_mvt/STAND_mvt (real dtype preserved: assigning None
    to a pandas 'str' column keeps the column 'str'-dtyped, per the real repro) and require `ars` to carry the
    contract's 'NA' token there, matching `fill()`'s convention used for every other UNM_KEYS column.
    Mutation: read `u.RUNWAY_mvt` / `u.STAND_mvt` directly instead of through `fill(...)` -> this goes red
    ('NA' not found in the resulting ars string)."""
    dep = (train_raw.PHASE_mvt.astype(str) == "DEP") & train_raw.AOBT_3_flt.isna() & train_raw.MVT_TIME_UTC_mvt.notna()
    dep &= train_raw.TAXITIME_SEC_mvt.notna() & (train_raw.TAXITIME_SEC_mvt.fillna(0) > 0)
    idx = train_raw[dep.to_numpy()].index[0]
    mod = train_raw.copy()
    mod.loc[idx, "RUNWAY_mvt"] = None
    mod.loc[idx, "STAND_mvt"] = None
    assert mod.RUNWAY_mvt.isna().loc[idx] and mod.STAND_mvt.isna().loc[idx]
    out = F.unmatched_frame(mod, False, cfg)
    row = out[out.MVT_ID_mvt == mod.loc[idx, "MVT_ID_mvt"]].iloc[0]
    assert row.RUNWAY_mvt == "NA" and row.stand_pref == "NA"
    ap = row.ADEP_mvt
    assert row.ars == f"{ap}|NA|NA"


def test_v26_builds_no_provisional_unmatched_frame(e2e_world, fm1_world_mod, weather):
    """With the shipped v26 lane the provisional frame is never built (it would be wasted work and a second, divergent
    definition of the unmatched rows). Mutation: drop the learner condition in build_frames -> a frame appears -> red."""
    v26 = K.parse_config(fm1_world_mod.config_raw(e2e_world[0]))
    scored = pd.read_parquet(e2e_world[0] / "raw" / "scored.parquet")
    fr = F.build_frames(scored, True, v26, weather, F.read_orl_routes(v26.path("features.orl_routes_file")))
    assert fr.unmatched is None and fr.matched is not None


def test_scored_frames_cover_every_departure(e2e_world, cfg, weather):
    scored = pd.read_parquet(e2e_world[0] / "raw" / "scored.parquet")
    fr = F.build_frames(scored, True, cfg, weather, F.read_orl_routes(cfg.path("features.orl_routes_file")))
    ids = set(fr.matched.MVT_ID_mvt) | set(fr.unmatched.MVT_ID_mvt)
    assert ids == set(fr.dep_ids)
    assert not set(fr.matched.MVT_ID_mvt) & set(fr.unmatched.MVT_ID_mvt)
    assert fr.matched.y.isna().all() and fr.unmatched.y.isna().all()


def test_adsb_table_must_cover_every_admitted_row(tmp_path, e2e_world):
    src = e2e_world[0] / "adsb"
    (tmp_path / "train.parquet").write_bytes((src / "train.parquet").read_bytes())
    (tmp_path / "serve.parquet").write_bytes((src / "serve.parquet").read_bytes())
    ids = pd.read_parquet(src / "train.parquet", columns=["MVT_ID_mvt"]).MVT_ID_mvt.to_numpy()
    got, missing = F.read_adsb_tables(tmp_path, ids[:5])
    assert sorted(got.index) == sorted(ids[:5].tolist()) and len(missing) == 0
    with pytest.raises(E.SchemaError, match="1 admitted row"):
        F.read_adsb_tables(tmp_path, np.r_[ids[:5], -1.0])
    got, missing = F.read_adsb_tables(tmp_path, np.r_[ids[:5], -1.0], allow_missing=True)
    assert missing.tolist() == [-1.0]
    (tmp_path / "serve.parquet").write_bytes((src / "train.parquet").read_bytes())
    with pytest.raises(E.DuplicateIdError, match="in both files"):
        F.read_adsb_tables(tmp_path, ids[:5])
    (tmp_path / "serve.parquet").unlink()
    with pytest.raises(E.SchemaError, match="does not exist"):
        F.read_adsb_tables(tmp_path, ids[:5])


def test_serve_rows_without_a_table_row_are_no_data_up_to_the_share(e2e_world):
    """FM1 notes N1. Mutation: drop the coverage-0 fill for missing rows -> coverage NaN -> red; drop the share check -> the
    refusal case goes green -> red."""
    src = e2e_world[0] / "adsb"
    ids = pd.read_parquet(src / "train.parquet", columns=["MVT_ID_mvt"]).MVT_ID_mvt.to_numpy()[:99]
    fr = pd.DataFrame({"MVT_ID_mvt": np.r_[ids, -1.0]})
    out, n = F.attach_adsb(fr, src, serve=True, max_missing_share=0.02)
    assert n == 1
    row = out[out.MVT_ID_mvt == -1.0].iloc[0]
    assert row.coverage == 0.0 and row.adsb_day_present == 0.0 and np.isnan(row.dwell_s)
    with pytest.raises(E.SchemaError, match=r"1 of 100 scored matched rows \(1\.0000%\).*serve_missing_max_share = 0\.005"):
        F.attach_adsb(fr, src, serve=True, max_missing_share=0.005)
    with pytest.raises(E.SchemaError, match="the tables must cover"):
        F.attach_adsb(fr, src, serve=False)


def test_dayoff_utc_vs_local():
    """data.dayoff_tz (owner decision 4): a 23:30 UTC schedule and a 00:10 UTC take-off are one UTC day apart but the SAME
    Berlin day. Mutation: ignore tz_mode -> the local case returns 1 -> red."""
    mvt = pd.Series(pd.to_datetime(["2025-01-15 00:10"], utc=True))
    sch = pd.Series(pd.to_datetime(["2025-01-14 23:30"], utc=True))
    ap = np.array(["EDDF"])
    assert F._dayoff(mvt, sch, ap, "UTC", {"EDDF": "Europe/Berlin"}).tolist() == [1.0]
    assert F._dayoff(mvt, sch, ap, "local", {"EDDF": "Europe/Berlin"}).tolist() == [0.0]


def test_month_chunks():
    raw = pd.DataFrame({"MVT_TIME_UTC_mvt": pd.to_datetime(["2025-02-01 00:00", "2025-01-31 23:59", "2025-02-03 00:00"],
                                                           utc=True), "x": [1, 2, 3]})
    parts = F.month_chunks(raw)
    assert [p.x.tolist() for p in parts] == [[2], [1, 3]]
    raw.loc[0, "MVT_TIME_UTC_mvt"] = pd.NaT
    with pytest.raises(E.RangeError, match="without MVT_TIME"):
        F.month_chunks(raw)


# ---------------------------------------------------------------------------------------------- design memory (2026-10-01)

def _design_frame(n=200_000, seed=0):
    """A matched-like frame: float64 / float32 / int32 columns, NaN, a string key — the dtypes of the real frame."""
    rng = np.random.default_rng(seed)
    f = pd.DataFrame({"ap": pd.array(rng.choice(["EDDF", "EGLL", "LIRF"], n), dtype="string"),
                      "y": rng.normal(600, 100, n), "delta": rng.normal(0, 50, n),
                      "a": rng.normal(size=n), "b": rng.normal(size=n).astype(np.float32),
                      "c": rng.integers(0, 9, n).astype(np.int32), "d": rng.normal(size=n), "e": rng.normal(size=n),
                      "g": rng.normal(size=n), "h": rng.normal(size=n), "i": rng.normal(size=n)})
    f.loc[rng.random(n) < 0.05, "a"] = np.nan
    return f


def _reference_design(frame, enc, columns):
    """matched_design as written on 2026-09-28 (frame.copy + column selection + float32 + float64)."""
    f = frame.copy()
    for k, v in enc.transform(frame).items():
        f[k] = v
    return f[list(columns)].to_numpy(dtype=np.float32).astype(np.float64)


def test_matched_design_equals_the_reference_formula_bit_for_bit_and_leaves_the_frame_alone():
    """Refactor guard (2026-10-01 memory fix): the design is built column by column, never via a copy of the whole
    frame; its values must equal the original formula's exactly (float32 rounding of int32 / float64 / float32 / NaN,
    encoding columns float32 -> float64). Mutation: cast to float64 directly (skip float32) -> red."""
    f = _design_frame(5_000)
    before = f.copy(deep=True)
    enc = F.EncodingTables.fit(f, {"te": f.y.to_numpy(float)}, ["ap"], 50.0)
    cols = ["a", "b", "c", "d", "te_ap", "e"]
    X = F.matched_design(f, enc, cols)
    ref = _reference_design(f, enc, cols)
    assert X.dtype == np.float64 and X.shape == (5_000, 6) and X.flags.c_contiguous
    assert np.array_equal(X.view(np.uint64), ref.view(np.uint64))      # bit-for-bit, NaN included
    pd.testing.assert_frame_equal(f, before)


def test_matched_design_allocates_at_most_twice_the_design_not_a_frame_copy():
    """The 2026-10-01 dry run peaked at 17.8 GB: matched_design held a full copy of the 2.1 GB frame, a second copy of
    the selected columns, the float32 intermediate and the float64 result at once (logs/fm1_final/mem_probe.log, +3.2 GB
    for a 2.0 GB X). Traced allocation during the call must stay within 2 x X.nbytes (float32 half + float64 result +
    per-column temporaries); the old formula needs > 3 x. Seen red against the old formula (3.5 x)."""
    import tracemalloc
    f = _design_frame()
    enc = F.EncodingTables.fit(f, {"te": f.y.to_numpy(float)}, ["ap"], 50.0)
    cols = ["a", "b", "c", "d", "e", "g", "h", "i", "te_ap"]
    tracemalloc.start()
    X = F.matched_design(f, enc, cols)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak <= 2.0 * X.nbytes, f"peak {peak / X.nbytes:.2f} x X.nbytes"


def test_matched_design_encoding_columns_take_precedence_over_frame_columns_of_the_same_name():
    """The 2026-09-28 formula assigned the encodings onto a copy of the frame, so an encoding always replaced a frame
    column of the same name; the column-wise build keeps that precedence. Mutation (survivor of the 2026-10-01
    rehearsal before this test): prefer the frame column -> red."""
    f = _design_frame(1_000)
    enc = F.EncodingTables.fit(f, {"te": f.y.to_numpy(float)}, ["ap"], 50.0)
    f["te_ap"] = -1.0                                   # a stale column with the encoding's name
    X = F.matched_design(f, enc, ["a", "te_ap"])
    assert np.array_equal(X[:, 1], enc.transform(f)["te_ap"].astype(np.float64))
    assert not (X[:, 1] == -1.0).any()
