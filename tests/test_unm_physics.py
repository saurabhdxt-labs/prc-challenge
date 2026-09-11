"""scripts/unm_physics.py - arm E5 (plans/PREREG_unm_physics_2026_09_11.md): a physics-informed unmatched body, scored by
seven locked clauses with an event-excluded C2, and its ship step.

What the tests pin, and why each exists:
  * the weather as-of join: the latest observation at or before take-off, never a later one, never another station's, an
    observation exactly 3 h old used and an older one not, a NaN field kept NaN (never filled from an earlier report);
  * the de-icing rule: the default recomputation equals the table's (a guard), the two registered sensitivities, the float
    trap at +3 C that would make `< 3` equal `<= 3`, the de-icing-airport interaction as a Kleene AND on the four airports;
  * the physics block: prev_dep_gap's no-earlier-departure negatives become NaN and nothing else moves; every row is found;
  * the dropped-input audit and the arms' feature lists;
  * the body: with S1C's inputs it IS S1C's regressor bit for bit; the replicated early-stopping split IS sklearn's; the
    two new encodings never read a stopping-row or a holdout label (BC-1) -- while S1C's own five do read the stopping rows;
  * S1C reproduction (and p_hat / nf_cells / nf_fit_c untouched), the LOMO parquet's stamps;
  * each clause's arithmetic and the verdict; the cuts, the attribution and the sensitivities;
  * the ship guards: the S1C path rebuilds the base inside build, only non-LIRF unmatched rows change and to rint(E5)
    recomputed independently, the 2026 inputs come from the serve file, the WORKING / v10 preconditions, the writer;
  * a planted-signal positive control (and its negative twin) on the synthetic world.
Every RNG is seeded. Mutation rehearsals are recorded per test ("Rehearsed 2026-09-11 RED: ..."): each mutation was
applied to the production file, the named test went RED, the file was restored from a backup (byte-identical, `diff -q`)
and the test went GREEN again.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor as HGR

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import unm_physics as up  # noqa: E402
import unm_congestion as uc  # noqa: E402
import unm_congestion_ship as us  # noqa: E402
import stand_ab as S  # noqa: E402
from prc import weather as WX  # noqa: E402

bs, sf = up.bs, up.sf
SEEDS = (0, 1, 2)
T0 = pd.Timestamp("2025-01-15 10:00:00", tz="UTC")


def _quiet(_msg):
    return None


# =============================================================================================
# fixtures
# =============================================================================================

@pytest.fixture(scope="module")
def world():
    """E3C's small synthetic world with the synthetic weather table and physics attached (signal planted), and the
    month-12 LOMO split."""
    unm, lirf, wsum, info = up.synthetic_world(seed=3, n_per_month=350, n_matched_per_month=300)
    m, tr, te = sf.lomo_folds(unm.month.to_numpy())[11]
    return unm, lirf, tr, te, m


@pytest.fixture(scope="module")
def specs(world):
    return up.arm_specs(up.audit_inputs(world[0]))


@pytest.fixture(scope="module")
def arms_and_parts(world, specs):
    unm, lirf, tr, te, m = world
    tr_mat = lirf[lirf.month != m]
    arms, parts = up.physics_arms(unm[tr], unm[te], tr_mat, specs, seeds=(0,))
    ref_arms, ref_parts = uc.congestion_arms(unm[tr], unm[te], tr_mat, seeds=(0,))
    return arms, parts, ref_arms, ref_parts


@pytest.fixture(scope="module")
def big():
    """A world whose non-fill rows exceed sklearn's 10,000-row early-stopping switch (the stopping rows exist)."""
    unm, lirf, _, _ = up.synthetic_world(seed=4, n_per_month=1000, n_matched_per_month=300)
    nonfill = unm[~bs.schedule_fill(unm)]
    assert len(nonfill) > 10_000
    return unm, nonfill


@pytest.fixture(scope="module")
def lomo_preds(world, specs):
    unm, lirf, _, _, _ = world
    small = unm[unm.month.isin([1, 2, 3])].reset_index(drop=True)      # three folds keep this under a few seconds
    lirf3 = lirf[lirf.month.isin([1, 2, 3])].reset_index(drop=True)
    preds = up.score_lomo(small, lirf3, specs, up.SEEDS, _quiet)
    ref = uc.score_lomo(small, lirf3, uc.SEEDS, _quiet)
    return small, preds, ref


def _obs(rows: list) -> pd.DataFrame:
    """A hand-built observation table: each row (station, valid, temp_c, wx_precip) with every other registered field
    filled neutral, the default de-icing columns computed by prc.weather's rule, then the variants attached."""
    recs = []
    for st, valid, temp, precip in rows:
        recs.append(dict(station=st, valid=pd.Timestamp(valid, tz="UTC") if not isinstance(valid, pd.Timestamp) else valid,
                         temp_c=temp, dewpoint_c=temp - 5.0, dewspread_c=5.0, vis_km=10.0, wind_kt=5.0, gust_kt=0.0,
                         wx_precip=precip, wx_frozen=0.0, wx_sn=0.0, wx_fzra=0.0, wx_fzdz=0.0, wx_fzfg=0.0, wx_fg=0.0, wx_br=0.0,
                         wx_intensity=np.nan if precip != 1.0 else 0.0))
    o = pd.DataFrame(recs)
    o["valid"] = o.valid.astype("datetime64[ns, UTC]")
    return _with_rule(o)


def _with_rule(o: pd.DataFrame) -> pd.DataFrame:
    moist = WX.or3(o.wx_precip.to_numpy(), o.wx_fg.to_numpy(), o.wx_br.to_numpy(), WX.le3(o.vis_km.to_numpy(), 1.5))
    o["dc_moist_cold"] = WX.and3(WX.le3(o.temp_c.to_numpy(), 3.0), moist)
    o["dc_frost"] = WX.and3(WX.le3(o.temp_c.to_numpy(), 0.0), WX.le3(o.dewspread_c.to_numpy(), 3.0))
    o["deicing_condition"] = WX.or3(o.dc_moist_cold.to_numpy(), o.dc_frost.to_numpy())
    return up.deicing_variants(o)


def _rows(spec: list) -> pd.DataFrame:
    return pd.DataFrame({"ADEP_mvt": [a for a, _ in spec],
                         "MVT_TIME_UTC_mvt": pd.Series([pd.Timestamp(t, tz="UTC") if t is not None else pd.NaT for _, t in spec],
                                                       dtype="datetime64[us, UTC]")})


# =============================================================================================
# the weather as-of join
# =============================================================================================

def test_asof_takes_the_latest_observation_at_or_before_take_off_never_a_later_one_nor_another_stations():
    """Rehearsed 2026-09-11 RED: direction='backward' -> 'nearest' (RED through weather_asof's own post-join guard, which
    raises on an observation after take-off -- not through this test's value assertions); allow_exact_matches True ->
    False (the take-off AT 10:30 reads the 10:00 report); `by='_code'` deleted (NaN pattern changes: rows read another
    station's report)."""
    obs = _obs([("EHAM", "2025-01-15 10:00", 1.0, 0.0), ("EHAM", "2025-01-15 10:30", 2.0, 0.0),
                ("EHAM", "2025-01-15 11:00", 3.5, 1.0), ("LFPG", "2025-01-15 10:10", 9.0, 0.0)])
    rows = _rows([("EHAM", "2025-01-15 10:59:00"), ("EHAM", "2025-01-15 10:29:59"), ("EHAM", "2025-01-15 10:30:00"),
                  ("LFPG", "2025-01-15 10:05:00"), ("LFPG", "2025-01-15 10:20:00"), ("EHAM", "2025-01-15 11:00:01"),
                  ("EHAM", "2025-01-15 09:59:59")])
    w = up.weather_asof(rows, obs)
    assert list(w.index) == list(range(7))                                     # the rows' order, not the join's
    np.testing.assert_array_equal(w.temp_c.to_numpy(), [2.0, 1.0, 2.0, np.nan, 9.0, 3.5, np.nan])
    np.testing.assert_array_equal(w.obs_age_s.to_numpy(), [1740.0, 1799.0, 0.0, np.nan, 600.0, 1.0, np.nan])
    np.testing.assert_array_equal(w.wx_precip.to_numpy(), [0.0, 0.0, 0.0, np.nan, 0.0, 1.0, np.nan])
    ok = w.obs_valid.notna().to_numpy()
    assert (w.obs_valid[ok].to_numpy() <= rows.MVT_TIME_UTC_mvt.astype("datetime64[ns, UTC]")[ok].to_numpy()).all()
    assert list(w.columns) == [*up.JOIN_COLUMNS, "obs_valid", "obs_age_s"]


def test_asof_caps_at_three_hours_inclusive():
    """Rehearsed 2026-09-11 RED: `tolerance=max_age` deleted (RED through weather_asof's own post-join age guard, which
    raises on the 3 h 1 s match -- not through this test's value assertions); MAX_OBS_AGE 3 h -> 2 h (the row at exactly
    3 h becomes NaN)."""
    obs = _obs([("EDDF", "2025-01-15 07:00", -2.0, 1.0)])
    rows = _rows([("EDDF", "2025-01-15 10:00:00"), ("EDDF", "2025-01-15 10:00:01"), ("EDDF", "2025-01-15 07:00:00")])
    w = up.weather_asof(rows, obs)
    np.testing.assert_array_equal(w.temp_c.to_numpy(), [-2.0, np.nan, -2.0])
    np.testing.assert_array_equal(w.obs_age_s.to_numpy(), [10_800.0, np.nan, 0.0])
    assert w.obs_valid.isna().tolist() == [False, True, False]
    assert np.isnan(w.loc[1, up.JOIN_COLUMNS].to_numpy(dtype="float64")).all()          # every column, not only temp
    assert w.loc[0, "deicing_condition"] == 1.0 and up.MAX_OBS_AGE == pd.Timedelta(hours=3)


def test_asof_keeps_nan_fields_nan_and_handles_missing_times_and_stations():
    """Rehearsed 2026-09-11 RED: the right-hand weather columns filled with 0 before the join (`.fillna(0.0)` on obs[c]:
    the unknown report reads 0); observations with a NaN temperature dropped before the join (the 10:40 row reads the
    10:00 report's 1.0 instead of the latest report's NaN)."""
    obs = _obs([("LSZH", "2025-01-15 10:00", 1.0, 0.0), ("LSZH", "2025-01-15 10:30", np.nan, np.nan)])
    rows = _rows([("LSZH", "2025-01-15 10:40:00"), ("LSZH", None), ("KJFK", "2025-01-15 10:40:00"), ("LSZH", "2025-01-15 10:10:00")])
    w = up.weather_asof(rows, obs)
    assert np.isnan(w.loc[0, "temp_c"]) and np.isnan(w.loc[0, "wx_precip"]) and w.loc[0, "obs_age_s"] == 600.0
    assert np.isnan(w.loc[0, "deicing_condition"])                             # unknown moisture at +1 C: unknown, not 0
    assert w.loc[1, ["temp_c", "obs_age_s"]].isna().all() and pd.isna(w.loc[1, "obs_valid"])     # no take-off time
    assert w.loc[2, ["temp_c", "obs_age_s"]].isna().all()                      # a station the table does not carry
    assert w.loc[3, "temp_c"] == 1.0 and w.loc[3, "wx_precip"] == 0.0 and w.loc[3, "deicing_condition"] == 0.0


def test_asof_refusals():
    """Rehearsed 2026-09-11 RED: the duplicate-key refusal deleted from check_obs (a table repeating (station, valid) is
    joined)."""
    obs = _obs([("EHAM", "2025-01-15 10:00", 1.0, 0.0)])
    with pytest.raises(ValueError, match="repeat a \\(station, valid\\) key"):
        up.weather_asof(_rows([("EHAM", "2025-01-15 10:10")]), pd.concat([obs, obs], ignore_index=True))
    naive = pd.DataFrame({"ADEP_mvt": ["EHAM"], "MVT_TIME_UTC_mvt": [pd.Timestamp("2025-01-15 10:10")]})
    with pytest.raises(ValueError, match="MVT_TIME_UTC_mvt must be timezone-aware UTC"):
        up.weather_asof(naive, obs)
    with pytest.raises(ValueError, match="`valid` must be timezone-aware UTC"):
        up.weather_asof(_rows([("EHAM", "2025-01-15 10:10")]), obs.assign(valid=obs.valid.dt.tz_localize(None)))
    with pytest.raises(ValueError, match="lacks \\['wx_br'\\]"):
        up.weather_asof(_rows([("EHAM", "2025-01-15 10:10")]), obs.drop(columns=["wx_br"]))
    with pytest.raises(ValueError, match="needs rows.ADEP_mvt"):
        up.weather_asof(_rows([("EHAM", "2025-01-15 10:10")]).drop(columns=["ADEP_mvt"]), obs)


def test_the_frozen_table_is_the_registered_one_and_a_changed_file_is_refused(tmp_path):
    """Rehearsed 2026-09-11 RED: the sha256 comparison deleted from load_weather_obs (a one-byte-changed copy loads)."""
    assert up.file_sha256(up.WEATHER_OBS) == up.WEATHER_SHA256 == "cb05439b6b9a57c159fd78049821821c6ac828087d2380c12d1c6f7f718022a8"
    obs = up.load_weather_obs()
    assert len(obs) == 205_417 and set(obs.station) == set(sf.AIRPORTS) and not obs.duplicated(["station", "valid"]).any()
    f = up.weather_findings(obs)
    assert f["temp_plus3_stored_below_3"] == 4_969 and f["spread3_stored_above_3"] == 5_504
    assert f["deicing_condition_lost_to_float_spread"] == 283 and f["variant_differs_from_default"] == {"fog15": 295, "lt3": 1_156}
    bad = tmp_path / "weather.parquet"
    pq.write_table(pq.read_table(up.WEATHER_OBS), bad, compression="gzip")    # same content, different bytes
    assert up.WEATHER_OBS.read_bytes() != bad.read_bytes()
    with pytest.raises(ValueError, match="has sha256 .* the prereg reads cb05439b6b9a"):
        up.load_weather_obs(bad)
    assert len(up.load_weather_obs(bad, sha256=None)) == 205_417


# =============================================================================================
# the de-icing rule: guard, sensitivities, interaction
# =============================================================================================

def _deice_table() -> pd.DataFrame:
    """Six observations, each built to separate the variants:
      r0 a reported +3 C (stored 2.999999999999999 by F->C) with mist: default 1, lt3 0, fog15 1 (mist is kept);
      r1 +1 C, fog at 5 km: default 1, fog15 0, lt3 1;     r2 +1 C, fog at 1 km: 1 in every variant (the vis clause);
      r3 +5 C with rain: 0 everywhere;                     r4 -1 C, a reported 3 C spread stored 3.0000000000000036,
      no moisture: dc_frost 0 in the table (the defect flag is 1);   r5 +1 C with an unknown weather group: unknown."""
    tc = (np.round(3.0 * 9 / 5 + 32, 1) - 32.0) * 5.0 / 9.0
    o = pd.DataFrame({"station": ["EHAM"] * 6, "valid": pd.date_range("2025-01-15", periods=6, freq="30min", tz="UTC"),
                      "temp_c": [tc, 1.0, 1.0, 5.0, -1.0, 1.0], "dewspread_c": [5.0, 5.0, 5.0, 5.0, 3.0000000000000036, 5.0],
                      "vis_km": [3.0, 5.0, 1.0, 10.0, 10.0, 10.0], "wx_precip": [0.0, 0.0, 0.0, 1.0, 0.0, np.nan],
                      "wx_fg": [0.0, 1.0, 1.0, 0.0, 0.0, np.nan], "wx_br": [1.0, 0.0, 0.0, 0.0, 0.0, np.nan]})
    for c in ("dewpoint_c", "wind_kt", "gust_kt", "wx_frozen", "wx_sn", "wx_fzra", "wx_fzdz", "wx_fzfg", "wx_intensity"):
        o[c] = 0.0
    return o


def test_deicing_variants_guard_the_default_rule_and_separate_the_two_sensitivities():
    """Rehearsed 2026-09-11 RED: `np.round(..., TEMP_DECIMALS)` deleted in deicing_variants (r0's lt3 stays 1: the float
    trap); fog15's moisture built with the bare fog flag (r1's fog15 becomes 1); the lt3 variant built with le3 (r0 stays
    1); the guard deleted (the tampered table is accepted)."""
    o = _with_rule(_deice_table())
    assert o.temp_c[0] == 2.999999999999999 and o.temp_c[0] < 3.0
    np.testing.assert_array_equal(o.deicing_condition.to_numpy(), [1.0, 1.0, 1.0, 0.0, 0.0, np.nan])
    np.testing.assert_array_equal(o.deicing_condition__lt3.to_numpy(), [0.0, 1.0, 1.0, 0.0, 0.0, np.nan])
    np.testing.assert_array_equal(o.deicing_condition__fog15.to_numpy(), [1.0, 0.0, 1.0, 0.0, 0.0, np.nan])
    np.testing.assert_array_equal(o.dc_frost_float_defect.to_numpy(), [0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    # the trap, pinned: a literal `< 3` on the stored value is TRUE for a reported +3 C
    assert up.lt3v([o.temp_c[0]], 3.0)[0] == 1.0 and up.lt3v([np.round(o.temp_c[0], up.TEMP_DECIMALS)], 3.0)[0] == 0.0
    assert np.isnan(up.lt3v([np.nan], 3.0)[0])
    tampered = o.drop(columns=["deicing_condition__fog15", "deicing_condition__lt3", "dc_frost_float_defect"]).copy()
    tampered.loc[3, "dc_moist_cold"] = 1.0
    with pytest.raises(ValueError, match="differs from the table's on 1 dc_moist_cold"):
        up.deicing_variants(tampered)
    with pytest.raises(ValueError, match="lacks \\['dc_frost'\\]"):
        up.deicing_variants(tampered.drop(columns=["dc_frost"]))


def test_deice_interaction_is_the_kleene_and_on_the_four_registered_airports():
    """Rehearsed 2026-09-11 RED: the interaction computed as the product cond x flag (LEMD's unknown reads NaN, not 0);
    'LSZH' dropped from DEICE_AIRPORTS (the LSZH row reads 0)."""
    got = up.deice_interaction([1.0, 1.0, 0.0, np.nan, np.nan, 1.0, 1.0], ["EHAM", "LFPG", "EDDM", "LSZH", "LEMD", "EDDF", "LSZH"])
    np.testing.assert_array_equal(got, [1.0, 0.0, 0.0, np.nan, 0.0, 1.0, 1.0])
    assert set(up.DEICE_AIRPORTS) == {"EHAM", "EDDM", "EDDF", "LSZH"}


def test_attach_physics_builds_every_input_keeps_the_rows_and_refuses_what_it_did_not_build(world):
    """Rehearsed 2026-09-11 RED: the variant interactions built from the DEFAULT deicing_condition (they no longer equal
    deice_interaction of their own variant); airline_adep built from ADES_mvt; the missing-id refusal deleted (a row
    without a physics row gets NaN physics silently)."""
    unm = world[0]
    base = unm.drop(columns=up.ADDED_COLUMNS).sample(frac=1.0, random_state=0)          # shuffled: the order must survive
    obs = up.smoke_weather_obs(seed=203)
    phys = unm[["MVT_ID_mvt", *up.PHYSICS_COLUMNS]]
    got = up.attach_physics(base, obs, phys)
    assert list(got.index) == list(base.index) and set(up.ADDED_COLUMNS) <= set(got.columns)
    for v in ("", "__fog15", "__lt3"):
        np.testing.assert_array_equal(got[f"{up.INTERACTION}{v}"].to_numpy(),
                                      up.deice_interaction(got[f"deicing_condition{v}"].to_numpy(), got.ADEP_mvt.to_numpy()))
    for v in ("__fog15", "__lt3"):                                              # the variants really differ somewhere
        assert (~up._nan_equal(got[f"{up.INTERACTION}{v}"], got[up.INTERACTION])).sum() > 0, v
    assert (got.airline_adep == got.ADEP_mvt.astype(str) + "|" + got.airline.astype(str)).all()
    sub = unm.loc[base.index]
    for c in up.PHYSICS_COLUMNS:
        np.testing.assert_array_equal(got[c].to_numpy(), sub[c].to_numpy())
    w = up.weather_asof(base, obs)
    np.testing.assert_array_equal(got.temp_c.to_numpy(), w.temp_c.to_numpy())
    assert str(got.obs_valid.dtype) == "datetime64[ns, UTC]"
    with pytest.raises(ValueError, match="already carry"):
        up.attach_physics(got, obs, phys)
    with pytest.raises(ValueError, match="1 of .* rows have no physics row"):
        up.attach_physics(base, obs, phys[phys.MVT_ID_mvt != base.MVT_ID_mvt.iloc[4]])
    with pytest.raises(ValueError, match="needs rows.\\['airline'\\]"):
        up.attach_physics(base.drop(columns=["airline"]), obs, phys)


# =============================================================================================
# the physics block
# =============================================================================================

def test_physics_frame_sets_only_negative_prev_dep_gap_to_nan():
    """Rehearsed 2026-09-11 RED: `< 0` -> `<= 0` (the zero gap is lost); the NaN rule applied to prev_arr_taxiin as well
    (its recorded -12 s taxi-in is lost)."""
    built = pd.DataFrame({"MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0], "q_dep_tko_sym15": np.float32([3, 4, 5, 6]),
                          "q_rwy_tko_sym10": np.float32([1, np.nan, 2, 3]), "q_arr_taxiin_sym30_tko": np.float32([500, 510, 520, 530]),
                          "q_rwy_ambient_proxy": np.float32([-5, 900, 910, 920]), "prev_arr_gap": np.float32([100, 0, 50, 60]),
                          "prev_arr_taxiin": np.float32([-12, 400, 500, 600]),
                          "prev_dep_gap": np.float32([-2_500_000, 0, 16_000, np.nan])})
    f, n = up.physics_frame(built)
    assert n == 1 and list(f.columns) == ["MVT_ID_mvt", *up.PHYSICS_COLUMNS] and (f.dtypes == "float64").all()
    np.testing.assert_array_equal(f.prev_dep_gap.to_numpy(), [np.nan, 0.0, 16_000.0, np.nan])
    np.testing.assert_array_equal(f.prev_arr_taxiin.to_numpy(), [-12.0, 400.0, 500.0, 600.0])
    np.testing.assert_array_equal(f.q_rwy_ambient_proxy.to_numpy(), [-5.0, 900.0, 910.0, 920.0])
    with pytest.raises(ValueError, match="duplicate MVT_IDs"):
        up.physics_frame(built.assign(MVT_ID_mvt=[1.0, 1.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="lacks \\['prev_dep_gap'\\]"):
        up.physics_frame(built.drop(columns=["prev_dep_gap"]))


def test_build_physics_uses_the_builder_and_compares_each_file_to_its_cache(tmp_path):
    """Rehearsed 2026-09-11 RED: compare_to_cache returning 'equal' without comparing (the altered cache reads equal); the
    running negative count assigned instead of accumulated (`+=` -> `=`: the total reads the last file's 1, not 4)."""
    def builder(p):
        k = 1 if p.name == "a.parquet" else 2
        return pd.DataFrame({"MVT_ID_mvt": np.arange(3, dtype="float64") + 10 * k,
                             **{c: np.float32([1.0, 2.0, 3.0]) for c in up.PHYSICS_COLUMNS if c != "prev_dep_gap"},
                             "prev_dep_gap": np.float32([-1.0, 5.0, -3.0][:3] if k == 1 else [4.0, 5.0, -6.0])})
    cache = tmp_path / "cache"
    cache.mkdir()
    builder(tmp_path / "a.parquet").to_parquet(cache / "a.parquet", index=False)
    altered = builder(tmp_path / "b.parquet")
    altered.loc[1, "q_rwy_tko_sym10"] = 99.0
    altered.to_parquet(cache / "b.parquet", index=False)
    frame, info = up.build_physics([tmp_path / "a.parquet", tmp_path / "b.parquet", tmp_path / "c.parquet"],
                                   lambda p: builder(p) if p.name != "c.parquet" else builder(p).assign(MVT_ID_mvt=[40.0, 41.0, 42.0]),
                                   cache, _quiet)
    assert len(frame) == 9 and frame.MVT_ID_mvt.is_unique
    assert info["files"]["a.parquet"]["cache"]["status"] == "equal"
    assert info["files"]["b.parquet"]["cache"] == {"status": "DIFFERENT", "n_rows": 3,
                                                   "n_differ": {c: int(c == "q_rwy_tko_sym10") for c in up.PHYSICS_COLUMNS}}
    assert info["files"]["c.parquet"]["cache"]["status"] == "no cache"
    assert info["n_prev_dep_gap_negative_to_nan"] == 2 + 1 + 1 == sum(f["n_prev_dep_gap_negative"] for f in info["files"].values())
    with pytest.raises(ValueError, match="duplicate MVT_IDs across"):
        up.build_physics([tmp_path / "a.parquet", tmp_path / "a.parquet"], builder, None, _quiet)


def test_the_registered_queue_block_is_stand_abs_and_the_stand_history_is_its_take_off_anchored_trio():
    """Rehearsed 2026-09-11 RED: REG_QUEUE's last entry changed (the import-time guard raises on `import unm_physics`)."""
    assert up.REG_QUEUE == S.UNMATCHED_QUEUE_FEATS == ["q_dep_tko_sym15", "q_rwy_tko_sym10", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy"]
    assert up.REG_STANDHIST == ["prev_arr_gap", "prev_arr_taxiin", "prev_dep_gap"] and set(up.REG_STANDHIST) <= set(S.STANDHIST)
    assert not set(up.PHYSICS_COLUMNS) & set(S.UNMATCHED_NAN_COLS)            # none of them reads the row's own AOBT_3


# =============================================================================================
# the audit and the feature lists
# =============================================================================================

def test_audit_drops_absent_and_empty_inputs_and_the_arms_exclude_them(world):
    """Rehearsed 2026-09-11 RED: the status set to 'built' whatever n is (the all-NaN input is kept); arm_specs ignoring the
    audit for the physics block (the dropped queue input stays in E5)."""
    unm = world[0].drop(columns=["wx_fzdz"]).assign(q_rwy_tko_sym10=np.nan)
    a = up.audit_inputs(unm)
    assert up.dropped(a) == ["wx_fzdz", "q_rwy_tko_sym10"]
    assert a["wx_fzdz"]["reason"].startswith("absent") and a["q_rwy_tko_sym10"]["reason"] == "no finite value on any unmatched row"
    assert a["q_rwy_tko_sym10"]["n_finite"] == 0 and a["temp_c"]["status"] == "built" and a["temp_c"]["n_finite"] > 0
    assert a["AIRCRAFT_TYPE_mvt"]["n_levels"] == 4 and a["airline_adep"]["status"] == "built"
    sp_ = up.arm_specs(a)
    for arm in up.ARMS:
        assert "wx_fzdz" not in sp_[arm]["numeric"] and "q_rwy_tko_sym10" not in sp_[arm]["numeric"]
    assert "q_dep_tko_sym15" in sp_["E5"]["numeric"]
    empty_type = world[0].assign(AIRCRAFT_TYPE_mvt=pd.Series([None] * len(world[0]), dtype="str"))
    assert up.arm_specs(up.audit_inputs(empty_type))["E5"]["extra"] == ["airline_adep"]


def test_arm_specs_on_the_full_frame_are_the_registered_lists(world, specs):
    """Rehearsed 2026-09-11 RED: the fog15 variant swapping only deicing_condition (its interaction stays the default's);
    E5w given the extra encodings."""
    s1c = ["sp", "dayoff", "hr", "hprox_med", "hprox_n"]
    weather = [*up.REG_WEATHER, "deice_x_airport"]
    physics = ["q_dep_tko_sym15", "q_rwy_tko_sym10", "q_arr_taxiin_sym30_tko", "q_rwy_ambient_proxy", "prev_arr_gap",
               "prev_arr_taxiin", "prev_dep_gap"]
    assert up.dropped(up.audit_inputs(world[0])) == []
    assert specs["E5w"] == {"numeric": s1c + weather, "extra": []}
    assert specs["E5"] == {"numeric": s1c + weather + physics, "extra": ["AIRCRAFT_TYPE_mvt", "airline_adep"]}
    for v in ("fog15", "lt3"):
        w = [f"{c}__{v}" if c in ("deicing_condition", "deice_x_airport") else c for c in weather]
        assert specs[f"E5_{v}"] == {"numeric": s1c + w + physics, "extra": ["AIRCRAFT_TYPE_mvt", "airline_adep"]}
    assert len(up.REG_WEATHER) == 15 and up.REG_WEATHER[-1] == "deicing_condition"


# =============================================================================================
# the body
# =============================================================================================

def test_physics_regressor_with_s1cs_inputs_is_s1cs_regressor_bit_for_bit(world):
    """Rehearsed 2026-09-11 RED: the design mapping every NF_ENCODED column through the FIRST column's map (predictions
    move); `super().__init__(frame, seed)` -> `(frame, seed + 1)` (another seed's regressor)."""
    unm, _, tr, te, _ = world
    nonfill = unm[tr][~bs.schedule_fill(unm[tr])]
    a = up.PhysicsRegressor(nonfill, 0, uc.CongestionRegressor.NUMERIC)
    b = uc.CongestionRegressor(nonfill, 0)
    np.testing.assert_array_equal(a.predict(unm[te]), b.predict(unm[te]))
    assert a.columns == b.columns and a.prior == b.prior and a.n_winsorised == b.n_winsorised and a.extra == [] and a.extra_maps == {}
    assert a.seed == b.seed == 0 and a.model.get_params() == b.model.get_params()
    for c in bs.NF_ENCODED:
        pd.testing.assert_series_equal(a.maps[c], b.maps[c])


def test_stopping_split_is_the_regressors_own_split():
    """Rehearsed 2026-09-11 RED: the split's random_state taken as `seed` itself instead of RandomState(seed).randint(...)
    (the explicit-validation refit no longer equals the internal one); the switch `n > 10_000` -> `n >= 10_000` (10,000
    rows get stopping rows)."""
    rng = np.random.default_rng(7)
    n = 12_000
    X = rng.normal(size=(n, 4))
    y = 3.0 * X[:, 0] + np.sin(X[:, 1]) + rng.normal(size=n)
    for seed in SEEDS:
        fit_idx, stop_idx = up.stopping_split(n, seed)
        assert len(stop_idx) == 1_200 and len(fit_idx) == 10_800 and not set(fit_idx) & set(stop_idx)
        internal = HGR(random_state=seed, **bs.NF_PARAMS).fit(X, y)
        explicit = HGR(random_state=seed, early_stopping=True, **bs.NF_PARAMS).fit(X[fit_idx], y[fit_idx], X_val=X[stop_idx], y_val=y[stop_idx])
        assert internal.do_early_stopping_ and internal.n_iter_ == explicit.n_iter_ < 200
        np.testing.assert_array_equal(internal.predict(X), explicit.predict(X))
        other_fit, other_stop = up.stopping_split(n, seed + 11)
        wrong = HGR(random_state=seed, early_stopping=True, **bs.NF_PARAMS).fit(X[other_fit], y[other_fit], X_val=X[other_stop], y_val=y[other_stop])
        assert not np.array_equal(internal.predict(X), wrong.predict(X))
    f, s = up.stopping_split(10_000, 0)
    assert len(s) == 0 and np.array_equal(f, np.arange(10_000))
    assert len(up.stopping_split(10_001, 0)[1]) == 1_001                  # ceil(0.1 x 10,001), sklearn's test-size rule
    assert up.SKLEARN_VERIFIED == "1.8.0"


def test_the_new_encodings_never_read_a_stopping_row_label_while_s1cs_own_do(big, specs):
    """BC-1. Rehearsed 2026-09-11 RED: the extra maps computed on the whole non-fill frame instead of the non-stopping rows
    (perturbing the stopping labels moves them); the extra prior taken from the parent (`self.prior`, which reads the
    stopping rows); the stopping positions taken from seed 0 whatever the seed (the maps move under seed 1's stopping rows)."""
    unm, nonfill = big
    spec = specs["E5"]
    reg = up.PhysicsRegressor(nonfill, 1, spec["numeric"], spec["extra"])
    fit_idx, stop_idx = up.stopping_split(len(nonfill), 1)
    assert reg.n_stop == len(stop_idx) > 1_000 and np.array_equal(reg.stop_positions, np.sort(stop_idx))
    assert reg.model.do_early_stopping_
    clean = nonfill.iloc[np.sort(fit_idx)]
    yw = np.clip(clean.y.to_numpy(dtype="float64"), -3_000.0, 3_000.0)
    assert reg.extra_prior == float(yw.mean()) != reg.prior
    for c in spec["extra"]:
        g = clean.assign(_y=yw).groupby(c, observed=True)["_y"].agg(["mean", "size"])
        want = (g["mean"] * g["size"] + reg.extra_prior * bs.SMOOTH) / (g["size"] + bs.SMOOTH)
        pd.testing.assert_series_equal(reg.extra_maps[c], want, check_names=False)
    # the stopping rows' labels moved: the new encodings do not move; S1C's own five DO (a pre-existing property, reported)
    moved = nonfill.copy()
    pos = moved.columns.get_loc("y")
    moved.iloc[stop_idx, pos] = np.where(moved.y.to_numpy()[stop_idx] > 1_500.0, 10.0, 2_990.0)
    reg2 = up.PhysicsRegressor(moved, 1, spec["numeric"], spec["extra"])
    assert reg2.extra_prior == reg.extra_prior
    for c in spec["extra"]:
        pd.testing.assert_series_equal(reg2.extra_maps[c], reg.extra_maps[c])
    assert reg2.prior != reg.prior and any(not reg2.maps[c].equals(reg.maps[c]) for c in bs.NF_ENCODED)
    # positive control: a NON-stopping label moves them
    again = nonfill.copy()
    again.iloc[np.sort(fit_idx)[:200], pos] = 2_999.0
    reg3 = up.PhysicsRegressor(again, 1, spec["numeric"], spec["extra"])
    assert reg3.extra_prior != reg.extra_prior


#: the columns that carry an unmatched row's label (`delta` is NaN on every unmatched row -- no AOBT_3 -- so it carries none)
LABEL_SOURCES = ("y", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt")


def _perturb_holdout_labels(frame: pd.DataFrame, te: np.ndarray, cols, seed: int) -> pd.DataFrame:
    """Every named label-bearing column moved on the holdout rows only. BLOCK is shifted by a random 5-60 min and a third
    of the holdout rows are put ON their schedule (so the fill flag, which reads BLOCK, flips on them)."""
    rng = np.random.default_rng(seed)
    moved = frame.copy()
    idx = moved.index[te]
    for c in cols:
        if c == "BLOCK_TIME_UTC_mvt":
            shift = pd.to_timedelta(rng.integers(300, 3_600, len(idx)), unit="s")
            moved.loc[idx, c] = moved.loc[idx, c] + shift
            onsched = idx[rng.random(len(idx)) < 1 / 3]
            moved.loc[onsched, c] = moved.loc[onsched, "SCHED_TIME_UTC_mvt"]
            assert moved[c].dtype == frame[c].dtype
        elif c == "TAXITIME_SEC_mvt":
            moved.loc[te, c] = (moved.loc[te, c].to_numpy() * 3 + 500).astype(frame[c].dtype)
        else:
            moved.loc[te, c] = moved.loc[te, c].to_numpy() * 3.0 + 500.0
    return moved


@pytest.mark.parametrize("cols", [(c,) for c in LABEL_SOURCES] + [LABEL_SOURCES], ids=lambda c: "+".join(c))
def test_holdout_labels_never_reach_any_arm(world, specs, cols):
    """No holdout label, through ANY column that carries it (y, TAXITIME, BLOCK -- the fill flag reads BLOCK), moves any
    arm. Rehearsed 2026-09-11 RED: physics_arms fitting the E5 bodies on train + test non-fill rows (y case).
    Rehearsed 2026-09-11 (c4) RED for the BLOCK and all-three cases, SURVIVED under the old y-only test: physics_arms routing
    holdout fills from their own BLOCK (`np.where(bs.schedule_fill(test_unm), sp, E5)`, a fill oracle)."""
    unm, lirf, tr, te, m = world
    tr_mat = lirf[lirf.month != m]
    arms, _ = up.physics_arms(unm[tr], unm[te], tr_mat, {"E5": specs["E5"]}, seeds=(0,))
    moved = _perturb_holdout_labels(unm, te, cols, seed=17)
    for c in cols:                                          # the perturbation really happened, on holdout rows only
        a, b = unm[c].to_numpy(), moved[c].to_numpy()
        assert (a[te] != b[te]).mean() > 0.9 and np.array_equal(a[tr], b[tr])
    if "BLOCK_TIME_UTC_mvt" in cols:
        assert (bs.schedule_fill(unm[te]) != bs.schedule_fill(moved[te])).sum() > 20
    arms2, _ = up.physics_arms(moved[tr], moved[te], tr_mat, {"E5": specs["E5"]}, seeds=(0,))
    for a in ("S1", "S1C", "E5", "E5_seed0"):
        np.testing.assert_array_equal(arms[a], arms2[a])


def test_fit_physics_regressor_refusals(world, specs):
    """Rehearsed 2026-09-11 RED: the fill-row refusal deleted (a frame with fill rows is fitted); the S1C-prefix check
    deleted (a body without the witness is fitted)."""
    unm, _, tr, _, _ = world
    nonfill = unm[tr][~bs.schedule_fill(unm[tr])]
    spec = specs["E5"]
    with pytest.raises(ValueError, match="NON-FILL rows only"):
        up.fit_physics_regressor(unm[tr], 0, spec["numeric"], spec["extra"])
    with pytest.raises(ValueError, match="needs non-fill training rows"):
        up.fit_physics_regressor(nonfill.iloc[:0], 0, spec["numeric"], spec["extra"])
    with pytest.raises(ValueError, match="must start with S1C's"):
        up.fit_physics_regressor(nonfill, 0, ["sp", "dayoff", "hr", "temp_c"], [])
    with pytest.raises(ValueError, match="not registered encodings"):
        up.fit_physics_regressor(nonfill, 0, spec["numeric"], ["ADES_mvt"])
    with pytest.raises(ValueError, match="duplicate inputs"):
        up.fit_physics_regressor(nonfill, 0, spec["numeric"] + ["temp_c"], [])
    with pytest.raises(ValueError, match="needs columns \\['q_dep_tko_sym15'\\]"):
        up.fit_physics_regressor(nonfill.drop(columns=["q_dep_tko_sym15"]), 0, spec["numeric"], spec["extra"])


# =============================================================================================
# the arms: S1C untouched
# =============================================================================================

def test_e5_arms_share_p_nf_cells_sp_and_the_tail_with_s1c(arms_and_parts, world, specs):
    """Rehearsed 2026-09-11 RED: the E5 mixture built on parts['nf_fit'] (S1's body) instead of the arm's; mixture(p * 0.99,
    ...) (the rebuild from S1C's parts breaks)."""
    arms, parts, ref_arms, ref_parts = arms_and_parts
    unm, _, tr, te, _ = world
    for a in ("S0", "S1", "S1_seed0", "S1C", "S1C_seed0"):
        np.testing.assert_array_equal(arms[a], ref_arms[a])
    for k in ("p", "sp", "nf_cells", "nf_fit", "nf_fit_c"):
        np.testing.assert_array_equal(parts[k], ref_parts[k])
    body = parts["nf_cells"] < bs.T_TAIL_S
    for a in up.ARMS:
        want = bs.mixture(ref_parts["p"], ref_parts["sp"], bs.nf_hybrid(ref_parts["nf_cells"], parts[f"nf_fit_{a}"]))
        np.testing.assert_array_equal(arms[a], want)
        np.testing.assert_array_equal(arms[f"{a}_seed0"], want)                  # one seed: the mean is the seed
        np.testing.assert_array_equal(arms[a][~body], arms["S1C"][~body])      # the tail is the cells' in every arm
        assert (arms[a] >= 1.0).all() and np.isfinite(arms[a]).all()
    assert (arms["E5"][body] != arms["S1C"][body]).mean() > 0.9 and (~body).sum() > 0         # 5 tail rows in this fold
    nonfill = unm[tr][~bs.schedule_fill(unm[tr])]
    assert parts["n_train_nonfill"] == len(nonfill)
    indep = up.PhysicsRegressor(nonfill, 0, specs["E5"]["numeric"], specs["E5"]["extra"]).predict(unm[te])
    np.testing.assert_array_equal(parts["nf_fit_E5"], indep)


# =============================================================================================
# LOMO and the reproduction guard
# =============================================================================================

def test_lomo_scores_every_row_once_and_the_guard_reproduces_e3c(lomo_preds):
    """Rehearsed 2026-09-11 RED: GUARD_COLUMNS reduced to ('S1C',) (a perturbed p_hat / nf_cells passes)."""
    small, preds, ref = lomo_preds
    assert len(preds) == len(small) and preds.MVT_ID_mvt.is_unique and set(preds.MVT_ID_mvt) == set(small.MVT_ID_mvt)
    assert list(preds.columns) == up.PRED_COLUMNS and (preds.fold == "lomo").all()
    assert up.PRED_COLUMNS[:len(uc.PRED_COLUMNS)] == uc.PRED_COLUMNS
    got = up.check_reproduction(preds, ref)
    assert got["status"] == "reproduced" and got["columns"] == list(up.GUARD_COLUMNS)
    assert got["max_abs_diff_s"] == {c: 0.0 for c in up.GUARD_COLUMNS}
    for col in ("S1C", "p_hat", "nf_cells", "S1C_seed1"):
        bad = ref.copy()
        bad.loc[7, col] += 1e-5
        with pytest.raises(AssertionError, match=f"{col} does not reproduce E3C's stored {col} on 1 of"):
            up.check_reproduction(preds, bad)
    assert up.GUARD_COLUMNS == ("S1C", "S1", "S1C_seed0", "S1C_seed1", "S1C_seed2", "p_hat", "nf_cells", "nf_fit_c")


def test_lomo_row_columns_carry_the_inputs_and_the_bodies(lomo_preds, specs):
    """Rehearsed 2026-09-11 RED: the stored row inputs read from the TRAINING rows' first values (misaligned);
    `nf_fit_E5` stored from the E5w body."""
    small, preds, _ = lomo_preds
    p = preds.set_index("MVT_ID_mvt").loc[small.MVT_ID_mvt.to_numpy()]
    for c in up.ROW_INPUTS:
        np.testing.assert_array_equal(p[c].to_numpy(), small[c].to_numpy(dtype="float64"))
    for a in up.ARMS:
        np.testing.assert_array_equal(p[a].to_numpy(), bs.mixture(p.p_hat.to_numpy(), p.sp.to_numpy(),
                                                                  bs.nf_hybrid(p.nf_cells.to_numpy(), p[f"nf_fit_{a}"].to_numpy())))
        assert {f"{a}_seed{s}" for s in SEEDS} <= set(p.columns)
    assert not np.array_equal(p.nf_fit_E5.to_numpy(), p.nf_fit_E5w.to_numpy())
    with pytest.raises(ValueError, match="registered seeds"):
        up.score_lomo(small, small.iloc[:0], specs, (0,), _quiet)
    with pytest.raises(ValueError, match="registered arms"):
        up.score_lomo(small, small.iloc[:0], {"E5": specs["E5"]}, up.SEEDS, _quiet)


def test_lomo_parquet_round_trips_with_its_stamps_and_a_reader_refuses_an_unstamped_or_foreign_file(lomo_preds, specs, tmp_path):
    """Rehearsed 2026-09-11 RED: write_lomo stamping convention 'delta' (the reader refuses its own file); the tag check
    deleted from read_lomo (E3C's parquet is read as E5's)."""
    _, preds, ref = lomo_preds
    path = tmp_path / "lomo.parquet"
    up.write_lomo(preds, path, {"reproduction": {"status": "reproduced"}, "git_sha": "abc123", "arms": specs,
                                "dropped_inputs": ["wx_fzdz"]})
    back, meta = up.read_lomo(path)
    pd.testing.assert_frame_equal(back, preds[up.PRED_COLUMNS])
    assert meta["reproduction"]["status"] == "reproduced" and meta["dropped_inputs"] == ["wx_fzdz"]
    sm = pq.read_schema(path).metadata
    assert sm[b"convention"] == b"taxi_time" and sm[b"tag"] == b"E5" and sm[b"prereg"] == up.PREREG.encode()
    assert sm[b"git_sha"] == b"abc123" and sm[b"weather_version"] == b"2.0.1" and json.loads(sm[b"feature_lists"]) == specs
    bare = tmp_path / "bare.parquet"
    pq.write_table(pa.Table.from_pandas(preds, preserve_index=False), bare)
    with pytest.raises(ValueError, match="carries convention ''"):
        up.read_lomo(bare)
    e3c = tmp_path / "e3c.parquet"
    uc.write_lomo(ref, e3c, {})
    with pytest.raises(ValueError, match="is not an E5 parquet"):
        up.read_lomo(e3c)


# =============================================================================================
# clauses on hand-built tables
# =============================================================================================

#: exact encodings of g = a^2 - c^2 (S1C = y + a, E5 = y + c), integers so every sum is exact in fp64 (E3R's table).
_ENC = {1_000.0: (35.0, 15.0), 2_000.0: (45.0, 5.0), 10_000.0: (100.0, 0.0), -300_000.0: (50.0, 550.0), 100_000.0: (350.0, 150.0),
        40.0: (7.0, 3.0), 60.0: (8.0, 2.0), -5.0: (2.0, 3.0), -12.0: (2.0, 4.0), 30.0: (6.5, 3.5)}
_APS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LSZH", "LTFM"]


def _arms_frame(rows: list) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    for a in ("E5w", "E5_fog15", "E5_lt3"):
        t[a] = t.E5
    for s in SEEDS:
        for a in ("S1C", *up.ARMS):
            t[f"{a}_seed{s}"] = t[a]
    return t


def _event_rows() -> pd.DataFrame:
    """Five airport-months, one bin each (E3R's table): EHAM-1 5 rows h 1,700 g -300,000 each (the largest |sum g|,
    NEGATIVE); LTFM-2 10 rows h 2,500 g +100,000; EDDF-3 29 rows h 1,800 g +10,000; LFPG-4 30 rows h 1,400 g +2,000;
    EGLL-5 40 rows h 500 g +1,000. With EHAM-1 dropped, 1500-2000 keeps 29 rows (thin: borrows 1300-1500's 2,000)."""
    spec = [("EHAM", 1, 1_700.0, -300_000.0, 5), ("LTFM", 2, 2_500.0, 100_000.0, 10), ("EDDF", 3, 1_800.0, 10_000.0, 29),
            ("LFPG", 4, 1_400.0, 2_000.0, 30), ("EGLL", 5, 500.0, 1_000.0, 40)]
    rows = []
    for ap, m, h, g, n in spec:
        a, c = _ENC[g]
        for i in range(n):
            rows.append(dict(month=m, date=f"2025-{m:02d}-{1 + i % 28:02d}", ADEP_mvt=ap, y=1_000.0, hprox_med=h, S1C=1_000.0 + a, E5=1_000.0 + c))
    return _arms_frame(rows)


_COUNTS_26 = {"NaN": 999, "<900": 100, "900-1100": 0, "1100-1300": 0, "1300-1500": 50, "1500-2000": 50, "2000-2400": 0, ">=2400": 450}


def test_c2_is_the_event_excluded_price_at_exactly_300_and_the_full_price_is_reported_only():
    """Rehearsed 2026-09-11 RED: clauses() leaving C2 as the full price (the record's C2 has no excluded airport-month and
    the price is negative); C2_MIN_MSE 300 -> 500 (the import-time check against E3R's bar raises)."""
    t = _event_rows()
    c = up.clauses(t, _COUNTS_26, "E5", n_boot=200, seed=0, n_scored=4_000)
    c2 = c["C2_priced_2026"]
    assert c2["excluded_airport_month"] == {"ADEP_mvt": "EHAM", "month": 1, "sum_g": -1_500_000.0, "n": 5, "n_rows_kept": 109,
                                            "n_rows_dropped": 5, "sum_g_kept": 1_390_000.0}
    assert c2["priced_mse"] == 300.0 and c2["passes"] and c2["c2_min_mse"] == 300.0 == up.C2_MIN_MSE
    assert c2["bins"]["1500-2000"]["thin"] and c2["bins"]["1500-2000"]["borrowed_from"] == "1300-1500"
    full = c["C2_full_price_not_decisional"]
    assert "passes" not in full and full["priced_mse"] < 0.0 and full["rule"].startswith("full price")
    assert c["treatment"] == "E5" and c["control"] == "S1C" and c["pooled_sum_g"] == -110_000.0
    assert c["airport_month_top"][0]["ADEP_mvt"] == "EHAM" and len(c["airport_month_top"]) == 5
    q = up.clauses(t, _COUNTS_26, "E5", n_boot=200, seed=0, n_scored=4_001)
    assert q["C2_priced_2026"]["priced_mse"] < 300.0 and not q["C2_priced_2026"]["passes"]


def _panel() -> pd.DataFrame:
    """E3C's 36-row boundary table with E5's names: 3 rows per month (NaN / calm 800 / hot 2,500), airports cycling, one
    date per month; pooled +634; 8 of 12 months and 6 of 9 airports positive; calm -60 vs bound -63.4; ex-monster +604."""
    rows, k = [], 0
    for m in range(1, 13):
        for j in range(3):
            g = [40.0, -5.0, 60.0][j] if m <= 8 else [-12.0, -5.0, -12.0][j]
            if (m, j) == (1, 0):
                g = 30.0
            a, c = _ENC[g]
            y = 10_801.0 if (m, j) == (1, 0) else 1_000.0
            rows.append(dict(month=m, date=f"2025-{m:02d}-15", ADEP_mvt=_APS[k % 9], y=y, hprox_med=[np.nan, 800.0, 2500.0][j],
                             S1C=y + a, E5=y + c))
            k += 1
    return _arms_frame(rows)


def test_c1_and_c3_to_c7_are_e3cs_arithmetic_on_the_renamed_view():
    """Rehearsed 2026-09-11 RED: e3c_view mapping S1 := treatment and S1C := control (the view's gains are negated and
    clauses() raises); the per-seed slots in the view taken from the pooled columns of the WRONG arm (C7's per-seed sums
    change sign)."""
    t = _panel()
    counts = {b: 100 for b in uc.BINS}
    c = up.clauses(t, counts, "E5", n_boot=200, seed=0, n_scored=1_000)
    assert c["pooled_sum_g"] == 634.0 and c["n_rows"] == 36
    c3, c4, c5, c6, c7 = c["C3_months"], c["C4_airports"], c["C5_calm_control"], c["C6_not_the_monsters"], c["C7_seeds"]
    assert c3["months_positive"] == 8 and c3["n_months"] == 12 and c3["passes"] and c3["per_month"]["1"] == {"sum_g": 85.0, "n": 3}
    assert c4["airports_positive"] == 6 and c4["n_airports"] == 9 and c4["passes"] and c4["per_airport"]["LSZH"] == {"sum_g": -20.0, "n": 4}
    assert c5["calm_sum_g"] == -60.0 and c5["n_calm"] == 12 and c5["bound"] == pytest.approx(-63.4, abs=1e-12) and c5["passes"]  # 0.1 x 634 inexact
    assert c6["exmonster_sum_g"] == 604.0 and c6["n_exmonster"] == 35 and c6["passes"]
    assert c7["seed_sd"] == 0.0 and c7["passes"] and c7["per_seed_sum_g"] == {"0": 634.0, "1": 634.0, "2": 634.0}
    c1 = c["C1_date_block_bootstrap"]
    assert c1["n_blocks"] == 12 and c1["n_draws"] == 200 and c1["seed"] == 0 and c1["sum_g"] == 634.0
    v = up.e3c_view(t, "E5")
    np.testing.assert_array_equal(v.S1.to_numpy(), t.S1C.to_numpy())
    np.testing.assert_array_equal(v.S1C.to_numpy(), t.E5.to_numpy())
    np.testing.assert_array_equal(v.S1C_seed2.to_numpy(), t.E5_seed2.to_numpy())
    np.testing.assert_array_equal(uc.gains(v), up.gains(t))
    # a reported arm is scored by the same function on its own column
    t2 = t.assign(E5w=t.S1C)
    for s in SEEDS:
        t2[f"E5w_seed{s}"] = t2.S1C
    cw = up.clauses(t2, counts, "E5w", n_boot=200, seed=0, n_scored=1_000)
    assert cw["pooled_sum_g"] == 0.0 and cw["treatment"] == "E5w" and not cw["C1_date_block_bootstrap"]["passes"]
    with pytest.raises(ValueError, match="non-LIRF rows only"):
        up.clauses(t.assign(ADEP_mvt=np.where(t.index == 3, "LIRF", t.ADEP_mvt)), counts, n_boot=200)


def test_gains_are_control_minus_treatment_on_the_raw_label():
    """Rehearsed 2026-09-11 RED: gains() with control and treatment swapped (the sign flips)."""
    t = pd.DataFrame({"y": [1_000.0, 20_000.0], "S1C": [1_100.0, 20_500.0], "E5": [1_050.0, 21_000.0], "E5w": [1_000.0, 20_000.0]})
    np.testing.assert_array_equal(up.gains(t), [100.0 ** 2 - 50.0 ** 2, 500.0 ** 2 - 1_000.0 ** 2])
    np.testing.assert_array_equal(up.gains(t, "E5w"), [100.0 ** 2, 500.0 ** 2])


def test_verdict_mapping_with_the_decisional_event_excluded_c2():
    """Rehearsed 2026-09-11 RED: in clauses(), the verdict taken from uc.clauses (computed on the FULL price) instead of
    recomputed after C2 is replaced (a table whose full price clears but whose event-excluded price does not reads WORKING)."""
    keys = ["C1_date_block_bootstrap", "C2_priced_2026", "C3_months", "C4_airports", "C5_calm_control", "C6_not_the_monsters", "C7_seeds"]
    ok = {k: {"passes": True} for k in keys}
    assert up.verdict(ok) == "WORKING"
    for k in keys[1:]:
        assert up.verdict({kk: {"passes": kk != k} for kk in keys}) == "INCONCLUSIVE", k
    assert up.verdict({**ok, "C1_date_block_bootstrap": {"passes": False}}) == "NOT WORKING"
    rows = [dict(month=m, date=f"2025-{m:02d}-15", ADEP_mvt=ap, y=1_000.0, hprox_med=800.0, S1C=1_001.0, E5=1_000.0)
            for m in range(1, 13) for ap in _APS]
    rows += [dict(month=2, date="2025-02-15", ADEP_mvt="LTFM", y=1_000.0, hprox_med=2_500.0, S1C=1_008.0, E5=1_002.0) for _ in range(100)]
    t = _arms_frame(rows)
    counts = {">=2400": 100_000, **{b: 0 for b in uc.BINS if b != ">=2400"}}
    c = up.clauses(t, counts, "E5", n_boot=200, seed=0, n_scored=1_000)
    assert c["C2_full_price_not_decisional"]["priced_mse"] == 6_000.0 and c["C2_priced_2026"]["priced_mse"] == 100.0
    for k in ("C1_date_block_bootstrap", "C3_months", "C4_airports", "C5_calm_control", "C6_not_the_monsters", "C7_seeds"):
        assert c[k]["passes"], k
    assert not c["C2_priced_2026"]["passes"] and c["verdict"] == "INCONCLUSIVE"


# =============================================================================================
# the cuts, the attribution, the sensitivities
# =============================================================================================

def test_cuts_attribution_and_sensitivities_arithmetic():
    """Rehearsed 2026-09-11 RED: SEASONS mapping December to 'SON'; the freezing key comparing temp < 0 (the 0 C row moves
    to '>0'). First rehearsal of cut_table's physics increment with its sign flipped (g(E5w) - g(E5)) SURVIVED -- only
    attribution()'s own total was checked; the per-cut increment assertions were added and the re-rehearsal went RED."""
    t = pd.DataFrame({"y": [1_000.0] * 6, "month": [12, 1, 4, 7, 10, 2], "ADEP_mvt": ["EHAM", "LFPG", "LSZH", "EDDF", "LEMD", "EHAM"],
                      "hprox_med": [800.0, np.nan, 1_200.0, 2_500.0, 800.0, 950.0],
                      "S1C": [1_010.0, 1_020.0, 1_000.0, 1_030.0, 1_005.0, 1_000.0],
                      "E5w": [1_005.0, 1_020.0, 1_000.0, 1_030.0, 1_005.0, 1_000.0],
                      "E5": [1_001.0, 1_010.0, 1_003.0, 1_000.0, 1_005.0, 1_000.0],
                      "E5_fog15": [1_001.0, 1_010.0, 1_003.0, 1_000.0, 1_005.0, 1_004.0],
                      "E5_lt3": [1_002.0, 1_010.0, 1_003.0, 1_000.0, 1_005.0, 1_000.0],
                      "deicing_condition": [1.0, 1.0, 0.0, np.nan, 0.0, 1.0], "deicing_condition__fog15": [1.0, 1.0, 0.0, np.nan, 0.0, 0.0],
                      "deicing_condition__lt3": [0.0, 1.0, 0.0, np.nan, 0.0, 1.0], "temp_c": [0.0, -2.0, 5.0, np.nan, 12.0, 2.999999999999999]})
    at = up.attribution(t)
    g5, gw = up.gains(t), up.gains(t, "E5w")
    assert at["sum_g_E5"] == g5.sum() == 99.0 + 300.0 - 9.0 + 900.0 and at["sum_g_E5w"] == gw.sum() == 75.0
    assert at["sum_physics_increment"] == at["sum_g_E5"] - at["sum_g_E5w"] and at["weather_share"] == 75.0 / 1_290.0
    assert at["per_month"]["12"]["sum_g_E5w"] == 75.0 and at["per_airport"]["EHAM"]["n"] == 2
    assert at["per_month"]["12"]["sum_physics_increment"] == 99.0 - 75.0 and at["per_month"]["1"]["sum_physics_increment"] == 300.0
    cuts = up.score_tables(t)
    assert set(cuts["per_season"]) == {"DJF", "MAM", "JJA", "SON"} and cuts["per_season"]["DJF"]["n"] == 3
    assert cuts["per_season"]["DJF"]["sum_g_E5"] == 99.0 + 300.0 and cuts["per_season"]["JJA"]["sum_g_E5"] == 900.0
    assert set(cuts["deice_airport_x_deicing_condition"]) == {"deice_apt|1", "deice_apt|0", "deice_apt|NaN", "other_apt|1", "other_apt|0"}
    assert cuts["deice_airport_x_deicing_condition"]["deice_apt|1"]["n"] == 2
    assert cuts["deice_airport_x_freezing"]["deice_apt|<=0"]["n"] == 1 and cuts["deice_airport_x_freezing"]["deice_apt|>0"]["n"] == 2
    assert cuts["deice_airport_x_freezing"]["other_apt|<=0"]["n"] == 1 and cuts["per_witness_bin"]["NaN"]["n"] == 1
    se = up.sensitivities(t)
    assert se["E5_fog15"]["n_rows_condition_differs"] == 1 and se["E5_fog15"]["sum_se_E5_minus_variant_on_those_rows"] == -16.0
    assert se["E5_lt3"]["n_rows_condition_differs"] == 1 and se["E5_lt3"]["sum_se_E5_minus_variant"] == 1.0 - 4.0
    assert se["E5_lt3"]["sum_g_vs_S1C"] == up.gains(t, "E5_lt3").sum()
    assert up.SEASONS[12] == "DJF" and up.SEASONS[3] == "MAM" and up.SEASONS[9] == "SON" and len(up.SEASONS) == 12


# =============================================================================================
# ship
# =============================================================================================

LIRF_OFFSET = 123
SERVE_SHIFT_S = 900.0


@pytest.fixture(scope="module")
def ship_world():
    """Training (months 1-11, witness + physics), a 'scored' set (month 12 rows stripped of witness and physics), a serve
    companion (prox 900 s hotter), the serve physics frame, the synthetic weather table, S1's parts, and a base whose
    non-LIRF unmatched rows are rint(S1C) on the serve witness (what v9 / v10 ship) and LIRF rows S1C + 123."""
    unm, lirf, _, _ = up.synthetic_world(seed=5, n_per_month=1000, n_matched_per_month=600)
    train_unm, train_lirf = unm[unm.month <= 11], lirf[lirf.month <= 11]
    full12 = unm[unm.month == 12].reset_index(drop=True)
    scored = full12.drop(columns=[*uc.WITNESS, *up.ADDED_COLUMNS])
    serve = uc.smoke_matched_companion(scored, seed=77)
    serve["AOBT_3_flt"] = serve.AOBT_3_flt - pd.Timedelta(seconds=SERVE_SHIFT_S)
    obs = up.smoke_weather_obs(seed=205)
    scored_physics = full12[["MVT_ID_mvt", *up.PHYSICS_COLUMNS]].copy()
    parts = {}
    bs.fit_unmatched(train_unm, scored, train_matched=train_lirf, hybrid=True, seeds=(0,), parts=parts)
    sw = uc.attach_witness(scored, uc.airport_hour_witness(serve))
    s1c, _, _, _ = us.s1c_from_parts(train_unm, sw, parts)
    vals = np.rint(s1c)
    vals[(scored.ADEP_mvt == "LIRF").to_numpy()] += LIRF_OFFSET
    matched_ids = np.arange(9_000_000, 9_000_050, dtype="float64")
    base = pd.DataFrame({"MVT_ID_mvt": np.concatenate([scored.MVT_ID_mvt.to_numpy(), matched_ids]),
                         "TAXITIME_SEC_mvt": np.concatenate([vals, np.full(50, 900.0)]).astype("int32")})
    spec = up.arm_specs(up.audit_inputs(train_unm))["E5"]
    return dict(train_unm=train_unm, scored=scored, serve=serve, obs=obs, scored_physics=scored_physics, parts=parts,
                base=base, matched_ids=matched_ids, spec=spec, s1c=s1c)


@pytest.fixture(scope="module")
def shipped(ship_world):
    w = ship_world
    snap = w["base"].copy()
    out, info = up.build_ship(w["train_unm"], w["scored"], w["parts"], w["base"], w["serve"], w["obs"], w["scored_physics"], w["spec"])
    return out, info, snap


def _independent_e5(w) -> np.ndarray:
    """rint(E5) recomputed here from the pieces: the serve witness and the physics attached to the scored rows, the body
    fitted on train[~schedule_fill(train)] at the parts' seeds, S1's p / sp / nf_cells."""
    sw = uc.attach_witness(w["scored"], uc.airport_hour_witness(w["serve"]))
    sp_ = up.attach_physics(sw, w["obs"], w["scored_physics"])
    nonfill = w["train_unm"][~bs.schedule_fill(w["train_unm"])]
    body = bs.mean_over_seeds({s: up.PhysicsRegressor(nonfill, s, w["spec"]["numeric"], w["spec"]["extra"]).predict(sp_)
                               for s in w["parts"]["seeds"]})
    return np.rint(bs.mixture(w["parts"]["p"], w["parts"]["sp"], bs.nf_hybrid(w["parts"]["nf_cells"], body)))


def test_ship_rebuild_guard_passes_on_the_true_base_refuses_a_perturbed_non_lirf_row_and_ignores_lirf(ship_world):
    """Rehearsed 2026-09-11 RED: the non-LIRF mask dropped from verify_s1c_rebuilds_base (the true base, whose LIRF rows
    are S1C + 123, is refused); the comparison given a 5 s tolerance (the +1 s perturbation passes)."""
    w = ship_world
    sw = uc.attach_witness(w["scored"], uc.airport_hour_witness(w["serve"]))
    non = us.non_lirf_mask(w["scored"])
    got = up.verify_s1c_rebuilds_base(w["s1c"], sw, w["base"])
    assert got == {"n_checked": int(non.sum()), "n_lirf_not_compared": int((~non).sum())} and 0 < got["n_lirf_not_compared"]
    bad = w["base"].copy()
    bad.loc[int(np.flatnonzero(non)[0]), "TAXITIME_SEC_mvt"] += 1
    with pytest.raises(AssertionError, match="does not rebuild the base on 1 of"):
        up.verify_s1c_rebuilds_base(w["s1c"], sw, bad)
    lirf_bad = w["base"].copy()
    lirf_bad.loc[int(np.flatnonzero(~non)[0]), "TAXITIME_SEC_mvt"] += 500
    up.verify_s1c_rebuilds_base(w["s1c"], sw, lirf_bad)
    with pytest.raises(AssertionError, match="missing from the base"):
        up.verify_s1c_rebuilds_base(w["s1c"], sw, w["base"].iloc[1:].reset_index(drop=True))


def test_build_ship_itself_refuses_a_base_the_s1c_path_does_not_rebuild(ship_world):
    """The guard must sit INSIDE build_ship. Rehearsed 2026-09-11 RED: the verify call deleted from build_ship."""
    w = ship_world
    non = us.non_lirf_mask(w["scored"])
    bad = w["base"].copy()
    bad.loc[int(np.flatnonzero(non)[3]), "TAXITIME_SEC_mvt"] += 7
    with pytest.raises(AssertionError, match="does not rebuild the base"):
        up.build_ship(w["train_unm"], w["scored"], w["parts"], bad, w["serve"], w["obs"], w["scored_physics"], w["spec"])


def test_ship_changes_only_non_lirf_unmatched_rows_to_rint_e5_recomputed_independently(ship_world, shipped):
    """Rehearsed 2026-09-11 RED: build_ship's body predicting on the witness-only rows (scored_w, no physics: KeyError ->
    the fixture errors); the E5 mixture built on nf_fit_c (the values equal the base's S1C and nothing changes)."""
    w, (out, info, _) = ship_world, shipped
    non = us.non_lirf_mask(w["scored"])
    o = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    b = w["base"].set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    changed = set(o.index[(o != b.loc[o.index]).to_numpy()])
    non_ids = set(w["scored"].MVT_ID_mvt[non])
    assert changed <= non_ids and len(changed) > 0.5 * len(non_ids)
    want = _independent_e5(w)
    np.testing.assert_array_equal(o.loc[w["scored"].MVT_ID_mvt[non]].to_numpy(), want[non].astype("int32"))
    assert info["n_rows_replaced"] == int(non.sum()) and info["n_values_differ"] == len(changed) and info["seeds"] == [0]
    assert info["spec"] == w["spec"] and set(info["coverage_2026_nan_share"]) == {*w["spec"]["numeric"][5:], *w["spec"]["extra"]}


def test_ship_keeps_lirf_and_matched_rows_byte_identical_the_dtype_the_order_and_the_base(ship_world, shipped):
    """Rehearsed 2026-09-11 RED: build_ship's non-LIRF mask replaced by all-True (LIRF rows are rewritten); the spliced
    column cast to int64."""
    w, (out, _, snap) = ship_world, shipped
    lirf_ids = w["scored"].MVT_ID_mvt[~us.non_lirf_mask(w["scored"])].to_numpy()
    o = out.set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    b = w["base"].set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    assert len(lirf_ids) > 0
    np.testing.assert_array_equal(o.loc[lirf_ids].to_numpy(), b.loc[lirf_ids].to_numpy())
    np.testing.assert_array_equal(o.loc[w["matched_ids"]].to_numpy(), np.full(50, 900, dtype="int32"))
    np.testing.assert_array_equal(out.MVT_ID_mvt.to_numpy(), w["base"].MVT_ID_mvt.to_numpy())
    assert out.TAXITIME_SEC_mvt.dtype == np.dtype("int32") and list(out.columns) == list(w["base"].columns)
    bs.check_submission(out, w["base"][["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=0))
    pd.testing.assert_frame_equal(w["base"], snap)


def test_ship_takes_2026_inputs_from_the_serve_side_only_and_refuses_an_empty_input(ship_world, shipped):
    """Rehearsed 2026-09-11 RED: check_serve_coverage's refusal deleted (an all-NaN 2026 queue input ships); the detail's
    deicing_condition read from the TRAINING rows' first values."""
    w, (_, info, _) = ship_world, shipped
    d = info["detail"]
    assert list(d.columns) == up.DETAIL_COLUMNS
    non = us.non_lirf_mask(w["scored"])
    sw = uc.attach_witness(w["scored"], uc.airport_hour_witness(w["serve"]))
    wx = up.weather_asof(sw, w["obs"])
    np.testing.assert_array_equal(d.deicing_condition.to_numpy(), wx.deicing_condition.to_numpy()[non])
    np.testing.assert_array_equal(d.hprox_med.to_numpy(), sw.hprox_med.to_numpy()[non])
    with pytest.raises(ValueError, match="already carry .*hprox_med"):
        up.build_ship(w["train_unm"], sw, w["parts"], w["base"], w["serve"], w["obs"], w["scored_physics"], w["spec"])
    pre = w["scored"].assign(temp_c=0.0)
    with pytest.raises(ValueError, match="already carry \\['temp_c'\\]"):
        up.build_ship(w["train_unm"], pre, w["parts"], w["base"], w["serve"], w["obs"], w["scored_physics"], w["spec"])
    empty = w["scored_physics"].assign(q_dep_tko_sym15=np.nan)
    with pytest.raises(ValueError, match="\\['q_dep_tko_sym15'\\] are NaN on every scored non-LIRF unmatched row"):
        up.build_ship(w["train_unm"], w["scored"], w["parts"], w["base"], w["serve"], w["obs"], empty, w["spec"])
    with pytest.raises(ValueError, match="not the hybrid path"):
        up.build_ship(w["train_unm"], w["scored"], {**w["parts"], "hybrid": False}, w["base"], w["serve"], w["obs"],
                      w["scored_physics"], w["spec"])


def test_ship_preconditions_working_reproduced_not_smoke_and_the_registered_base():
    """Rehearsed 2026-09-11 RED: the base-name check deleted (v9 is accepted); the verdict check reading `!= "NOT WORKING"`
    (INCONCLUSIVE ships)."""
    ok = {"tag": "E5", "prereg": up.PREREG, "smoke": False, "verdict": "WORKING",
          "lomo_metadata": {"reproduction": {"status": "reproduced"}}}
    up.check_ship_allowed(ok, "merry-quicksand_v10.parquet")
    for bad, msg in (({**ok, "verdict": "INCONCLUSIVE"}, "only if WORKING"), ({**ok, "verdict": "NOT WORKING"}, "only if WORKING"),
                     ({**ok, "smoke": True}, "SMOKE"), ({**ok, "tag": "E3C"}, "not E5's"),
                     ({**ok, "lomo_metadata": {"reproduction": {"status": "skipped (synthetic)"}}}, "did not reproduce")):
        with pytest.raises(SystemExit, match=msg):
            up.check_ship_allowed(bad, "merry-quicksand_v10.parquet")
    with pytest.raises(SystemExit, match="ships E5 on merry-quicksand_v10.parquet, not merry-quicksand_v9.parquet"):
        up.check_ship_allowed(ok, "merry-quicksand_v9.parquet")
    assert up.REGISTERED_BASE == "merry-quicksand_v10.parquet"
    with pytest.raises(SystemExit):
        up.parse_args(["ship", "--smoke", "--base", "x", "--version", "3"])
    with pytest.raises(SystemExit):
        up.parse_args(["lomo", "--version", "3"])


def test_write_ship_outputs_round_trips_refuses_to_overwrite_and_carries_the_band(ship_world, shipped, tmp_path):
    """Rehearsed 2026-09-11 RED: the `dest.exists()` refusal deleted (the second call overwrites); the band's conservative
    entry read from the full price."""
    w, (out, info, _) = ship_world, shipped
    template = w["base"][["MVT_ID_mvt"]].assign(TAXITIME_SEC_mvt=0)
    record = {"clauses": {"C2_priced_2026": {"priced_mse": 812.5}, "C2_full_price_not_decisional": {"priced_mse": 2_400.0}}}
    dest, meta = up.write_ship_outputs(out, info, template, 97, "merry-quicksand_v10.parquet", tmp_path, record, 0.0)
    assert dest == tmp_path / "merry-quicksand_v97.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(dest), out)
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "merry-quicksand_v97.unm_rows.parquet"), info["detail"])
    disk = json.loads((tmp_path / "merry-quicksand_v97.meta.json").read_text())
    assert disk["board_band"]["conservative_event_excluded_price"] == 812.5 and disk["board_band"]["optimistic_full_price"] == 2_400.0
    assert disk["base"] == "merry-quicksand_v10.parquet" and disk["prereg"] == up.PREREG and disk["weather_version"] == "2.0.1"
    assert disk["n_values_differ"] == info["n_values_differ"] and disk["rebuild_guard"] == info["rebuild"]
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        up.write_ship_outputs(out, info, template, 97, "merry-quicksand_v10.parquet", tmp_path, record, 0.0)
    with pytest.raises(ValueError, match="row count"):
        up.write_ship_outputs(out.iloc[:-1], info, template, 96, "merry-quicksand_v10.parquet", tmp_path, record, 0.0)
    assert not (tmp_path / "merry-quicksand_v96.parquet").exists()


# =============================================================================================
# a planted signal is recovered (and absent when not planted)
# =============================================================================================

def _reference_asof(rows: pd.DataFrame, obs: pd.DataFrame, col: str, direction: str) -> np.ndarray:
    """A test-side as-of join written independently of up.weather_asof: per station, the observation at or before
    (`backward`) or at or after (`forward`, the anticipating defect) the row's take-off, within 3 h; NaN otherwise."""
    cap = int(pd.Timedelta(hours=3).value)
    out = np.full(len(rows), np.nan)
    t_all = rows.MVT_TIME_UTC_mvt.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").to_numpy()
    st_all = rows.ADEP_mvt.astype(str).to_numpy()
    for st in np.unique(st_all):
        o = obs[obs.station.astype(str) == st].sort_values("valid", kind="mergesort")
        pos = np.flatnonzero((st_all == st) & ~np.isnat(t_all))
        if o.empty or not len(pos):
            continue
        v = o.valid.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").to_numpy().astype("int64")
        x = o[col].to_numpy(dtype="float64")
        t = t_all[pos].astype("int64")
        if direction == "backward":
            k = np.searchsorted(v, t, side="right") - 1
            ok = k >= 0
            ok[ok] = (t[ok] - v[k[ok]]) <= cap
        else:
            k = np.searchsorted(v, t, side="left")
            ok = k < len(v)
            ok[ok] = (v[k[ok]] - t[ok]) <= cap
        out[pos[ok]] = x[k[ok]]
    return out


def test_planted_deicing_signal_on_the_true_take_off_weather_is_recovered_and_the_join_does_not_anticipate():
    """The plant is defined on the TRUE take-off weather -- an independent test-side as-of join of the observation table
    -- not on the harness's own joined column (the review's finding: a plant on the joined column is recovered even by an
    anticipating join). The world must tell a backward join from a forward one (they differ on > 5% of rows); the
    harness's joined condition must equal the backward reference on every row; then +400 s planted on non-fill de-icing
    rows is lifted by E5w by well over a third, and not without the plant.
    Rehearsed 2026-09-11 (c4) RED, and SURVIVED under the old closed-loop test: weather_asof joining at take-off + 1 h
    (anticipation that its own age assertion cannot see); attach_physics writing the joined weather shifted by one row.
    Earlier rehearsals (RED): arm_specs' E5w given S1C's inputs only; the interaction built on an all-zero condition."""
    seed = 6
    lifts = {}
    for plant in (True, False):
        unm, lirf, _, _ = up.synthetic_world(seed=seed, plant=False, n_per_month=900, n_matched_per_month=300)
        obs = up.smoke_weather_obs(seed=seed + 200)                 # the table synthetic_world joined (its seed + 200)
        back = _reference_asof(unm, obs, "deicing_condition", "backward")
        fwd = _reference_asof(unm, obs, "deicing_condition", "forward")
        both = ~np.isnan(back) & ~np.isnan(fwd)
        assert both.sum() > 1_000 and (back[both] != fwd[both]).mean() > 0.05
        np.testing.assert_array_equal(unm.deicing_condition.to_numpy(dtype="float64"), back)
        truth = up.deice_interaction(back, unm.ADEP_mvt.to_numpy()) == 1.0
        nonfill = ~bs.schedule_fill(unm)
        if plant:
            unm = unm.assign(y=np.rint(unm.y.to_numpy(dtype="float64") + up.PLANT_DEICE_S * (truth & nonfill)))
        specs_ = up.arm_specs(up.audit_inputs(unm))
        m, tr, te = sf.lomo_folds(unm.month.to_numpy())[0]
        arms, parts = up.physics_arms(unm[tr], unm[te], lirf[lirf.month != m], {"E5w": specs_["E5w"]}, seeds=(0,))
        hit = truth[te] & nonfill[te] & (parts["nf_cells"] < bs.T_TAIL_S)
        assert hit.sum() > 40
        lifts[plant] = float((arms["E5w"][hit] - arms["S1C"][hit]).mean())
    assert lifts[True] > up.PLANT_DEICE_S / 3.0, lifts
    assert abs(lifts[False]) < up.PLANT_DEICE_S / 4.0, lifts


# =============================================================================================
# paths, constants, the composed modules
# =============================================================================================

def test_smoke_paths_never_touch_the_real_outputs_nor_the_parents():
    """Rehearsed 2026-09-11 RED: output_paths(True) returning the real paths."""
    smoke, real = up.output_paths(True), up.output_paths(False)
    assert set(smoke) == set(real) == {"lomo", "json", "lomo_log", "score_log"}
    assert all(p.parent == ROOT / "data" / "smoke_unm_physics" for p in smoke.values())
    assert real["lomo"] == ROOT / "data" / "cache_stand" / "unm_physics_lomo.parquet" and real["json"] == ROOT / "reports" / "unm_physics.json"
    parents = set(uc.output_paths(False).values()) | set(uc.output_paths(True).values())
    assert not (set(smoke.values()) & set(real.values())) and not ((set(smoke.values()) | set(real.values())) & parents)
    assert up.REFERENCE == ROOT / "data" / "cache_stand" / "unm_congestion_lomo.parquet"
    for cmd in (up.COMMAND_LOMO, up.COMMAND_SCORE, up.COMMAND_SHIP):
        assert "OMP_NUM_THREADS=1 nice -n 19 /opt/homebrew/bin/python3.11 -B -u scripts/unm_physics.py" in cmd and 'echo "exit $?"' in cmd


def test_registered_constants_are_the_preregs():
    """Rehearsed 2026-09-11 RED: DEICE_AIRPORTS given LFPG as a fifth airport; MAX_OBS_AGE 3 h -> 4 h."""
    assert up.REG_WEATHER == ["temp_c", "dewspread_c", "vis_km", "wind_kt", "gust_kt", "wx_precip", "wx_frozen", "wx_sn", "wx_fzra",
                              "wx_fzdz", "wx_fzfg", "wx_fg", "wx_br", "wx_intensity", "deicing_condition"]
    assert up.DEICE_AIRPORTS == ("EDDF", "EDDM", "EHAM", "LSZH") and up.MAX_OBS_AGE == pd.Timedelta(hours=3)
    assert up.REG_ENCODED == ["AIRCRAFT_TYPE_mvt", "airline_adep"] and up.ARMS == ("E5w", "E5", "E5_fog15", "E5_lt3")
    assert up.C2_MIN_MSE == 300.0 and up.SEEDS == (0, 1, 2) and up.N_BOOT == 2_000 and up.BOOT_SEED == 0
    assert up.N_SCORED_2026 == 344_841 and up.CONTROL == "S1C" and up.DECISIONAL == "E5" and up.CONVENTION == "taxi_time"
    assert up.MOIST_C == 3.0 and up.FOG_VIS_KM == 1.5 and up.WEATHER_VERSION == "2.0.1" and up.TAG == "E5"
    assert up.PREREG == "plans/PREREG_unm_physics_2026_09_11.md" and (ROOT / up.PREREG).exists()
    assert bs.WINSOR_S == 3_000.0 and bs.T_TAIL_S == 3_000.0 and bs.SMOOTH == 50.0 and uc.CongestionRegressor.NUMERIC == [
        "sp", "dayoff", "hr", "hprox_med", "hprox_n"]


def _tracked_clean(rels) -> int:
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *rels], cwd=ROOT, capture_output=True)
    return r.returncode


def test_composed_modules_are_untouched():
    """The prereg composes the shipped code and the E3C / E3R harnesses; it never edits them (all tracked: git diff).
    prc/weather.py is another session's to edit and is NOT pinned here: the frozen parquet's sha256 is (see above).
    Rehearsed 2026-09-11 RED (a TEST-side mutation, no other file edited): reports/bug_classes.md, modified in the working
    tree by another session, appended to the list -- the check reports it as touched."""
    rels = ["scripts/build_submission.py", "scripts/stratum_fold.py", "scripts/unm_congestion.py", "scripts/unm_congestion_resid.py",
            "scripts/unm_congestion_ship.py", "scripts/stand_ab.py"]
    rc = _tracked_clean(rels)
    if rc not in (0, 1):
        pytest.skip("git unavailable")
    assert rc == 0, f"one of {rels} differs from HEAD"
