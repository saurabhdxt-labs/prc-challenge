"""The ported LDI stage (prc/pipeline/fm1/ldi.py): the LTFM de-icing correction v41 shipped
(plans/PREREG_ldi_ltfm_deicing_2026_09_27.md; owner decision 2026-09-29: the REGISTERED constants, target rows computed in
the package, never re-derived from raw).

Tiers: the knobs and the plan; the rule on a constructed LTFM world (three local days built so that each definition —
band edges, the late share, the temperature cut, the UTC + 3 h local day, missing weather, other airports — flips at
least one row when it is changed); the whole-day contract against the scored file; the e2e world through api.predict
(slow); the REAL replay against v41 on every LTFM matched departure of the 2026 scored file (slow, in a child process).

Mutation rehearsal per test is recorded in its docstring and in tests/fm1/MUTATION_FM1.md (M-LDI rows).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from prc.pipeline.fm1 import errors as E
from prc.pipeline.fm1 import knobs as K
from prc.pipeline.fm1 import ldi as L
from prc.pipeline.fm1 import shipped
from prc.pipeline.fm1 import stages as S
from isolation import run_guarded

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULT = ROOT / "logs/ldi_scratch/result_20260927T1410/result.json"
STAGE = {s.name: s for s in S.STAGE_ORDER}
NAN = float("nan")


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


#: (id, airport, MVT UTC, sp = MVT - SCHED in s, w_freezing, w_temp_c). Local day = UTC + 3 h.
#: local 01-12: rows 1-5, late (> 1800 s) = 1, 2, 3 -> share 0.6 >= 0.5, min temp 0.5 <= 1.0 -> SEVERE
#: local 01-13: rows 6-7, both late but min temp 4.0 > 1.0                                   -> not severe
#: local 01-14: rows 8-10, min temp -3.0 but 1 of 3 late                                     -> not severe
#: row 6 is on UTC 01-12 but local 01-13: with a UTC day it would join 01-12 (late share 4/6, severe) and fire.
WORLD = [
    (1.0, "LTFM", "2026-01-11 21:30", 3601, 1.0, 0.5),    # first local half hour of 01-12; band >60   -> target
    (2.0, "LTFM", "2026-01-12 08:00", 2400, 1.0, 2.0),    # band 30-60                                -> target
    (3.0, "LTFM", "2026-01-12 09:00", 1801, 0.0, 3.0),    # late, not freezing                        -> untouched
    (4.0, "LTFM", "2026-01-12 10:00", 600, 1.0, 1.0),     # band <30                                  -> target
    (5.0, "LTFM", "2026-01-12 11:00", 1800, NAN, 1.5),    # sp exactly 1800: not late, band <30; weather missing
    (6.0, "LTFM", "2026-01-12 21:30", 4000, 1.0, 4.0),    # local 01-13
    (7.0, "LTFM", "2026-01-13 10:00", 3600, 1.0, 5.0),    # sp exactly 3600: band 30-60
    (8.0, "LTFM", "2026-01-14 06:00", 300, 1.0, -3.0),
    (9.0, "LTFM", "2026-01-14 07:00", 2000, 1.0, -2.0),
    (10.0, "LTFM", "2026-01-14 08:00", 100, 1.0, -1.0),
    (11.0, "EGLL", "2026-01-12 08:00", 5000, 1.0, -5.0),  # another airport, freezing and late: never touched
]
TARGETS = {1.0: ">60", 2.0: "30-60", 4.0: "<30"}


def _rows(world=WORLD) -> pd.DataFrame:
    """Raw scored rows (the columns LDI and its whole-day check read)."""
    t = pd.to_datetime([w[2] for w in world]).tz_localize("UTC")
    sp = pd.to_timedelta([w[3] for w in world], unit="s")
    return pd.DataFrame({"MVT_ID_mvt": [w[0] for w in world], "PHASE_mvt": "DEP",
                         "ADEP_mvt": [w[1] for w in world], "ADES_mvt": "LTAI",
                         "MVT_TIME_UTC_mvt": t, "SCHED_TIME_UTC_mvt": t - sp,
                         "AOBT_3_flt": t - sp - pd.Timedelta(minutes=5)})


def _matched(world=WORLD) -> pd.DataFrame:
    """The matched-lane feature frame's columns LDI reads (weather float32, as stand_ab.build_weather writes it)."""
    return pd.DataFrame({"MVT_ID_mvt": [w[0] for w in world], "ap": [w[1] for w in world],
                         "w_freezing": np.array([w[4] for w in world], dtype="float32"),
                         "w_temp_c": np.array([w[5] for w in world], dtype="float32")})


def _ledger(ids, value=1000.0) -> S.Ledger:
    led = S.Ledger(ids)
    led.write(STAGE["cbv_blend"], ids, np.full(len(ids), value))
    return led


def _deltas() -> dict:
    return L.deltas(K.parse_config({}))


# ---------------------------------------------------------------------------------------------- knobs + plan

@pytest.mark.devtree
def test_knobs_default_to_the_registered_constants_and_ldi_is_ported():
    """The defaults are LDI's registered Delta_apply (result.json jan26.delta_apply), exactly; the switch is implemented,
    so switching it on is no longer refused. Mutation: round a default to 8.0 -> red."""
    want = json.loads(RESULT.read_text())["jan26"]["delta_apply"]
    assert _deltas() == want == {"<30": 8.003118629519271, "30-60": 35.44180751815995, ">60": 122.4855891190231}
    assert K.KNOBS["modules.ldi.enabled"].status == "implemented"
    assert K.parse_config({"modules": {"ldi": {"enabled": True}}}).not_ported() == []
    for band_knob in L.DELTA_KNOBS.values():
        assert K.KNOBS[band_knob].lo == 0.0          # the registered rule applies max(0, .): never a negative shift


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_delta_knobs_refuse_negative_and_non_finite_values(value):
    with pytest.raises(E.ConfigError, match="modules.ldi.delta_gt60_s"):
        K.parse_config({"modules": {"ldi": {"delta_gt60_s": value}}})


def test_ldi_needs_the_matched_lane_and_its_weather_block():
    """LDI reads the matched frame's w_freezing / w_temp_c. Mutation: drop the features.weather check -> red."""
    with pytest.raises(E.ConfigValueError, match="modules.ldi.enabled needs lanes.matched"):
        K.parse_config({"lanes": {"matched": False}, "modules": {"ldi": {"enabled": True}}})
    with pytest.raises(E.ConfigValueError, match="modules.ldi.enabled needs features.weather"):
        K.parse_config({"features": {"weather": False}, "modules": {"ldi": {"enabled": True}}})
    K.parse_config({"lanes": {"matched": False}})    # off: no constraint


def test_plan_places_ldi_after_the_unmatched_lane_and_before_mtx():
    """Mutation: plan maps ldi to modules.storm.mode -> the ldi-on plan lacks it -> red."""
    on = [s.name for s in S.plan(K.parse_config({"run": {"purpose": "scored_period"},
                                                 "modules": {"ldi": {"enabled": True}}}))]
    assert on[-4:] == ["rome_e2w", "ldi", "mtx", "finalise"]
    off = [s.name for s in S.plan(K.parse_config({}))]
    assert "ldi" not in off


# ---------------------------------------------------------------------------------------------- the rule

def test_flags_equal_the_shipped_2026_frame():
    """L.flags composes ldi_correction's public functions; ldi_correction.frame_2026 (what v41 was built from) gives the
    same day, band, severity and target on the constructed world. Mutation: target = severe only (drop the w_freezing
    term) -> row 3 becomes a target -> red."""
    LC = shipped.ldi_correction()
    rows, m = _rows(), _matched()
    got = L.flags(L.frame(m, rows)).set_index("MVT_ID_mvt")
    ltfm = m.ap == "LTFM"
    n = int(ltfm.sum())
    base = pd.DataFrame({"MVT_ID_mvt": m.MVT_ID_mvt[ltfm], "TAXITIME_SEC_mvt": 1000})
    pr = pd.DataFrame({"MVT_ID_mvt": m.MVT_ID_mvt[ltfm], "taxi_r12": np.zeros(n)})
    ref = LC.frame_2026(rows, m[["MVT_ID_mvt", "w_freezing", "w_temp_c"]], pr, pr, base).set_index("MVT_ID_mvt")
    assert set(got.index) == set(ref.index) == set(m.MVT_ID_mvt[ltfm])
    ref = ref.reindex(got.index)
    for c in ("band", "severe", "target"):
        assert got[c].tolist() == ref[c].tolist(), c
    assert [str(d) for d in got.day] == [str(d) for d in ref.day]
    assert dict(got.band[got.target]) == TARGETS


def test_local_day_is_utc_plus_three_and_band_edges_are_the_registered_ones():
    """Mutation: local day = the UTC date -> row 6 joins 01-12 and becomes a target -> red. Mutation: band edge
    sp < 1800 -> row 5 moves to 30-60 -> red."""
    f = L.flags(L.frame(_matched(), _rows())).set_index("MVT_ID_mvt")
    assert str(f.day[1.0]) == "2026-01-12" and str(f.day[6.0]) == "2026-01-13"
    assert f.band[5.0] == "<30" and f.band[7.0] == "30-60" and f.band[3.0] == "30-60"
    assert not f.target[6.0] and not f.target[5.0]


@pytest.mark.parametrize("share_rows,tmin,severe", [(2, 1.0, True), (1, 1.0, False), (2, 1.01, False)])
def test_severity_cuts_are_inclusive(share_rows, tmin, severe):
    """share(sp > 1800) >= 0.5 AND min w_temp_c <= 1.0 (the registered cut). A 4-row day with `share_rows` late rows and
    minimum temperature `tmin`. Mutation: `>` for `>=` on the share -> the (2, 1.0) case turns False -> red."""
    world = [(float(i), "LTFM", f"2026-01-20 {8 + i:02d}:00", 3000 if i < share_rows else 60, 1.0,
              tmin if i == 0 else 5.0) for i in range(4)]
    f = L.flags(L.frame(_matched(world), _rows(world)))
    assert f.severe.tolist() == [severe] * 4


# ---------------------------------------------------------------------------------------------- serve

def test_serve_adds_the_band_delta_on_severe_freezing_ltfm_rows_only():
    """Targets get value + Delta[band] and decided_by 'ldi'; every other row (non-freezing, missing weather, other days,
    other airports) keeps its value and writer. After finalise the registered constants round to +8 / +35 / +122 s, as
    v39/v41 shipped. Mutation: target = w_freezing == 1 (no severity) -> rows 6-10 move -> red."""
    rows, m = _rows(), _matched()
    ids = m.MVT_ID_mvt.to_numpy(float)
    led = _ledger(ids)
    comp, rep = L.serve(m, rows, led, STAGE["ldi"], _deltas(), rows)
    d = _deltas()
    want = {i: 1000.0 + (d[TARGETS[i]] if i in TARGETS else 0.0) for i in ids}
    assert dict(zip(ids, led.values_of(ids))) == want
    assert dict(zip(ids, led.decided_of(ids))) == {i: ("ldi" if i in TARGETS else "cbv_blend") for i in ids}
    out = led.finalise().set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
    assert (out[[1.0, 2.0, 4.0]] - 1000).tolist() == [122, 35, 8]
    assert rep["n_rows"] == 10 and rep["n_target"] == 3 and rep["severe_days"] == ["2026-01-12"]
    assert rep["n_target_by_band"] == {"<30": 1, "30-60": 1, ">60": 1} and rep["population"] == "scored_file"
    assert comp.loc[1.0, "ldi_delta"] == d[">60"] and comp.loc[3.0, "ldi_delta"] == 0.0
    assert comp.loc[2.0, "ldi_band"] == "30-60" and bool(comp.loc[2.0, "ldi_severe"]) is True
    assert 11.0 not in comp.index                    # EGLL: not an LDI row at all


def test_a_zero_delta_band_writes_nothing():
    """v39 wrote only target rows with a positive Delta (ldi_run.values_for_v39). Mutation: write every target -> row 4
    is relabelled 'ldi' -> red."""
    rows, m = _rows(), _matched()
    ids = m.MVT_ID_mvt.to_numpy(float)
    led = _ledger(ids)
    _, rep = L.serve(m, rows, led, STAGE["ldi"], {**_deltas(), "<30": 0.0}, rows)
    assert led.decided_of([4.0])[0] == "cbv_blend" and led.values_of([4.0])[0] == 1000.0
    assert rep["n_target"] == 2


def test_serve_without_ltfm_rows_is_a_reported_noop():
    rows, m = _rows(WORLD[-1:]), _matched(WORLD[-1:])
    led = _ledger(m.MVT_ID_mvt.to_numpy(float))
    comp, rep = L.serve(m, rows, led, STAGE["ldi"], _deltas(), rows)
    assert rep["n_rows"] == 0 and rep["n_target"] == 0 and len(comp) == 0
    assert led.decided_of([11.0])[0] == "cbv_blend"
    comp, rep = L.serve(None, rows, led, STAGE["ldi"], _deltas(), rows)
    assert rep["n_rows"] == 0


def test_serve_refuses_a_target_row_no_value_stage_served():
    """LDI shifts a value; it never invents one. Mutation: treat NaN as 0 -> a 122 s value appears -> red."""
    rows, m = _rows(), _matched()
    led = S.Ledger(m.MVT_ID_mvt.to_numpy(float))
    with pytest.raises(E.RoutingError, match="LDI target row"):
        L.serve(m, rows, led, STAGE["ldi"], _deltas(), rows)


def test_serve_refuses_a_matched_frame_without_weather_or_times():
    rows, m = _rows(), _matched()
    led = _ledger(m.MVT_ID_mvt.to_numpy(float))
    with pytest.raises(E.MissingColumnError, match="w_temp_c"):
        L.serve(m.drop(columns="w_temp_c"), rows, led, STAGE["ldi"], _deltas(), rows)
    with pytest.raises(E.MissingColumnError, match="'ap'"):          # checked before the no-LTFM early return
        L.serve(m.drop(columns="ap"), rows, led, STAGE["ldi"], _deltas(), rows)
    with pytest.raises(E.NullabilityError, match="SCHED_TIME_UTC_mvt"):
        L.serve(m, rows.assign(SCHED_TIME_UTC_mvt=pd.NaT), led, STAGE["ldi"], _deltas(), rows)
    short = rows[rows.MVT_ID_mvt != 2.0]             # whole days of its own population, but matched row 2 has no raw row
    with pytest.raises(E.RoutingError, match="not in the scored rows"):
        L.serve(m, short, led, STAGE["ldi"], _deltas(), short)


def test_serve_refuses_deltas_that_are_not_the_three_bands():
    rows, m = _rows(), _matched()
    led = _ledger(m.MVT_ID_mvt.to_numpy(float))
    with pytest.raises(E.ConfigValueError, match="bands"):
        L.serve(m, rows, led, STAGE["ldi"], {"<30": 8.0}, rows)


# ---------------------------------------------------------------------------------------------- the whole-day contract

def test_a_partial_day_of_the_scored_file_is_refused_by_name():
    """The severe-day test reads EVERY LTFM departure of a local day. A batch holding part of a scored-file day would
    change which rows fire, silently (AT14 samples rows); it is refused. Mutation: skip the check -> the sampled batch
    serves (row 1's day loses rows 2-3, its late share drops) -> red."""
    rows, m = _rows(), _matched()
    scored = rows
    keep = ~rows.MVT_ID_mvt.isin([2.0, 3.0])
    led = _ledger(m.MVT_ID_mvt[keep].to_numpy(float))
    with pytest.raises(E.PartialDayError, match=r"2026-01-12: the batch holds 3 of the scored file's 5"):
        L.serve(m[keep], rows[keep], led, STAGE["ldi"], _deltas(), scored)


def test_a_batch_from_another_file_is_its_own_population_and_a_mixed_batch_is_refused():
    rows, m = _rows(), _matched()
    other = rows.assign(MVT_ID_mvt=rows.MVT_ID_mvt + 1000)
    led = _ledger(m.MVT_ID_mvt.to_numpy(float))
    _, rep = L.serve(m, rows, led, STAGE["ldi"], _deltas(), other)
    assert rep["population"] == "batch" and rep["n_target"] == 3
    mixed = pd.concat([rows.iloc[:5], other.iloc[5:]], ignore_index=True)
    with pytest.raises(E.PartialDayError, match="mixes"):
        L.serve(m, rows, _ledger(m.MVT_ID_mvt.to_numpy(float)), STAGE["ldi"], _deltas(), mixed)


def test_apply_reads_the_deltas_and_the_scored_file_from_the_config(tmp_path):
    """The engine's entry point: deltas from the knobs, the whole-day population from data.scored_file. Mutation: apply
    passes `rows` as the scored population -> the partial batch below is served -> red."""
    rows, m = _rows(), _matched()
    p = tmp_path / "scored.parquet"
    rows.to_parquet(p, index=False)
    cfg = K.parse_config({"data": {"scored_file": str(p)}, "modules": {"ldi": {"enabled": True}}})
    led = _ledger(m.MVT_ID_mvt.to_numpy(float))
    comp, rep = L.apply(cfg, m, rows, led, STAGE["ldi"])
    assert rep["n_target"] == 3 and rep["deltas"] == _deltas() and rep["population"] == "scored_file"
    keep = ~rows.MVT_ID_mvt.isin([3.0])
    with pytest.raises(E.PartialDayError):
        L.apply(cfg, m[keep], rows[keep], _ledger(m.MVT_ID_mvt[keep].to_numpy(float)), STAGE["ldi"])


# ---------------------------------------------------------------------------------------------- the e2e world

@pytest.mark.slow
def test_world_predict_serves_ldi_through_the_engine(e2e_world, fm1_world_mod, monkeypatch):
    """api.predict with LDI on: the engine calls the stage after the lanes, the ledger labels the written rows 'ldi',
    the provenance carries ldi_delta / ldi_band, and the report carries the stage's report. The world has no LTFM, so
    the rule is pointed at EDDF and its flags are forced for the wiring check (the rule itself is tested above).
    Mutation: drop the engine's LDI call -> no 'ldi' rows, no report -> red."""
    from prc import api
    world = e2e_world[0]
    raw = fm1_world_mod.config_raw(world, **{"modules.ldi.enabled": True, "run.artifacts_dir": str(world / "art_ldi")})
    cfg = K.parse_config(raw)
    art = api.fit(cfg, save=False, log=lambda *a, **k: None)
    scored = pd.read_parquet(world / "raw" / "scored.parquet")
    base = api.predict(art, scored, log=lambda *a, **k: None)
    assert base.attrs["fm1_report"]["ldi"]["n_rows"] == 0          # no LTFM in the world: a reported no-op
    real_flags = L.flags

    def forced(frame):
        f = real_flags(frame)
        f["target"] = True
        return f

    monkeypatch.setattr(L, "AIRPORT", "EDDF")
    monkeypatch.setattr(L, "flags", forced)
    pred = api.predict(art, scored, log=lambda *a, **k: None).set_index("MVT_ID_mvt")
    rep = pred.attrs["fm1_report"]["ldi"]
    hit = pred.index[pred.decided_by == "ldi"]
    assert len(hit) == rep["n_target"] > 0
    b = base.set_index("MVT_ID_mvt")
    want = pred.ldi_band[hit].map({band: art.config[knob] for band, knob in L.DELTA_KNOBS.items()})
    assert want.notna().all() and len(set(want)) == 3                        # every row in a named band; all three hit
    assert (pred.ldi_delta[hit] == want).all()          # each row's shift is ITS band's delta (LD1: bands swapped -> RED)
    # the base values are whole seconds, so the rounded output moves by the delta rounded (8 / 35 / 122 s)
    assert ((pred.TAXITIME_SEC_mvt[hit] - b.TAXITIME_SEC_mvt[hit]) == np.rint(want)).all()
    same = pred.index.difference(hit)
    assert (pred.TAXITIME_SEC_mvt[same] == b.TAXITIME_SEC_mvt[same]).all()


# ---------------------------------------------------------------------------------------------- the REAL replay

@pytest.mark.devtree
@pytest.mark.slow
def test_ldi_replays_v41_on_every_ltfm_matched_departure_of_the_2026_scored_file():
    """FM1's own features on the raw 2026 scored rows (LTFM, January + July), the pre-LDI value = v40's (v40 == v36 on
    every LDI row, and FM1 == v36 there by AT14), the LDI stage through apply() with the config's knobs. TRUE shape,
    predicted from the v40 -> v41 diff (1,885 rows by +8 / +35 / +122 s: 625 / 738 / 522) before running:
      weather   the matched frame's w_freezing / w_temp_c == data/cache_weather/ranking.parquet (the LDI run's source)
      targets   == the 1,885 ids of logs/ldi_scratch/result_20260927T1410/v39_values.parquet, none in July
      values    final == v41 on every LTFM matched departure
    FALSE: any id or value differs. Runs in a child (the feature build)."""
    code = f"""
import sys, json, resource, numpy as np, pandas as pd
ROOT = {str(ROOT)!r}
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + '/tools')
import fm1_replay_gate as GT
from prc.pipeline.fm1 import features as F, knobs as K, stages as S, ldi as L
cfg = K.parse_config(K.nest({{"adsb.table_dir": str(GT.PATHS["r12_tables"]),
                              "adsb.stack.models_dir": str(GT.PATHS["r12_models"]), "modules.ldi.enabled": True}}))
weather = F.stand_ab().load_weather(cfg.path("data.weather_dir"))
orl = F.read_orl_routes(cfg.path("features.orl_routes_file"))
rows = pd.concat([GT.airport_month_rows(GT.PATHS, "LTFM", mo) for mo in (1, 7)], ignore_index=True)
m = F.build_frames(rows, serve=True, cfg=cfg, weather=weather, orl_routes=orl).matched
m = m[m.ap.astype(str) == "LTFM"].reset_index(drop=True)
ids = m.MVT_ID_mvt.to_numpy(float)
wx = pd.read_parquet(ROOT + "/data/cache_weather/ranking.parquet").set_index("MVT_ID_mvt").reindex(ids)
sub = lambda v: pd.read_parquet(ROOT + f"/submissions/merry-quicksand_v{{v}}.parquet").set_index("MVT_ID_mvt").TAXITIME_SEC_mvt
v40, v41 = sub(40).reindex(ids).to_numpy(float), sub(41).reindex(ids).to_numpy(float)
st = {{s.name: s for s in S.STAGE_ORDER}}
led = S.Ledger(ids)
led.write(st["cbv_blend"], ids, v40)
comp, rep = L.apply(cfg, m, rows, led, st["ldi"])
out = led.finalise().set_index("MVT_ID_mvt").TAXITIME_SEC_mvt.reindex(ids).to_numpy(float)
v39 = pd.read_parquet(ROOT + "/logs/ldi_scratch/result_20260927T1410/v39_values.parquet").MVT_ID_mvt.to_numpy(float)
hit = ids[out != v40]
month = rows.set_index(rows.MVT_ID_mvt.to_numpy(float)).MVT_TIME_UTC_mvt.reindex(ids).dt.month.to_numpy()
rk = pd.read_parquet(ROOT + "/logs/ltfm_scratch/ltfm_2026_rank.parquet", columns=["MVT_ID_mvt"])
extra = np.setdiff1d(ids[month == 1], rk.MVT_ID_mvt.to_numpy(float))
print(json.dumps({{
  "n": int(len(ids)), "n_jan": int((month == 1).sum()), "n_jul": int((month == 7).sum()),
  "extra_vs_ldi_run": extra.tolist(), "extra_hit": np.isin(extra, hit).tolist(),
  "w_equal": all(bool(np.array_equal(m[c].to_numpy("float64"), wx[c].to_numpy("float64"), equal_nan=True))
                 for c in ("w_freezing", "w_temp_c")),
  "targets_equal_v39": bool(set(hit.tolist()) == set(v39.tolist())), "n_hit": int(len(hit)),
  "n_hit_july": int(np.isin(hit, ids[month == 7]).sum()),
  "steps": sorted((int(k), int(v)) for k, v in pd.Series(out - v40)[out != v40].value_counts().items()),
  "final_vs_v41": int((out != v41).sum()), "rep": {{k: rep[k] for k in ("n_rows", "n_target", "severe_days", "population")}},
  "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3}}))
"""
    r = run_guarded([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=900, cwd=str(ROOT))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-5000:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    # LDI's RESULT counts 22,061 January rows: its scratch extract (logs/ltfm_scratch/ltfm_2026_rank.parquet) lacks the
    # two LTFM departures with no ADES_mvt (203120674, 203121551; local days 01-01 / 01-02, neither severe), which FM1's
    # matched lane admits. Measured 2026-09-29 (first run of this test, red on 22,063); the targets and values below are
    # unaffected (they are not target rows, and their days stay non-severe).
    assert rep["n_jan"] == 22_063 and rep["extra_vs_ldi_run"] == [203120674.0, 203121551.0]
    assert rep["extra_hit"] == [False, False]
    assert rep["w_equal"]
    assert rep["targets_equal_v39"] and rep["n_hit"] == 1_885 and rep["n_hit_july"] == 0
    assert rep["steps"] == [[8, 625], [35, 738], [122, 522]]     # json: (step s, rows)
    assert rep["final_vs_v41"] == 0
    assert rep["rep"]["severe_days"] == ["2026-01-12", "2026-01-13", "2026-01-18", "2026-01-19"]
    assert rep["rep"]["population"] == "scored_file" and rep["rep"]["n_target"] == 1_885
    assert rep["peak_rss_gib"] <= 4.5
