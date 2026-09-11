"""prc.pipeline.ingest -- declarative schemas that fail loud, and the serve-time label guard: a scored
departure's BLOCK_TIME_UTC_mvt / TAXITIME_SEC_mvt are never requested from the file, never present in a
scored frame, and an airport outside the registry is refused.

Mutation rehearsal (2026-09-11, scratchpad p2a/mutate.py): the break named in each docstring turned the test
RED against prc/pipeline/ingest.py, GREEN on restore.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("pipeline_world", pathlib.Path(__file__).with_name("pipeline_world.py"))
pw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pw)

from prc.pipeline import config as C  # noqa: E402
from prc.pipeline import errors as E  # noqa: E402
from prc.pipeline import ingest, legacy  # noqa: E402

REAL_RANKING = ROOT / "data" / "raw" / "ranking.parquet"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return pw.dry_run_world(tmp_path_factory.mktemp("ingest_world"), seed=3)


def test_serve_columns_are_build_submission_columns_minus_exactly_the_labels():
    """Rehearsed: TAXITIME_SEC_mvt dropped from LABEL_COLUMNS -> RED (it would be read at serve time)."""
    bs = legacy.stratum().bs
    assert [c.name for c in ingest.RAW_MOVEMENT] == list(bs.COLUMNS)
    assert set(ingest.LABEL_COLUMNS) == {"BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"}
    assert list(ingest.SERVE_COLUMNS) == [c for c in bs.COLUMNS if c not in ingest.LABEL_COLUMNS]


@pytest.mark.skipif(not REAL_RANKING.exists(), reason="challenge data not present")
def test_the_real_ranking_file_passes_the_serve_schema_and_its_labels_are_outside_the_projection():
    """On the REAL scored file's schema: every serve column is present with the right kind, the label columns
    exist in the file (so excluding them is doing work) and none is in the projection.
    Rehearsed: SERVE_COLUMNS built from every RAW_MOVEMENT column -> RED."""
    schema = ingest.check_file_schema(REAL_RANKING, [ingest.RAW_BY_NAME[c] for c in ingest.SERVE_COLUMNS])
    assert set(ingest.LABEL_COLUMNS) <= set(schema.names)
    assert not set(ingest.SERVE_COLUMNS) & set(ingest.LABEL_COLUMNS)


def test_reading_the_scored_file_never_requests_a_label_column(world, monkeypatch):
    """Every pq.read_table call ingest makes for the scored file names its columns, and none is a label.
    Rehearsed: read_serve_departures reading the table without `columns=` -> RED (None = every column)."""
    calls = []
    real = pq.read_table

    def spy(path, *a, columns=None, **kw):
        calls.append((pathlib.Path(path).name, columns))
        return real(path, *a, columns=columns, **kw)

    monkeypatch.setattr(ingest.pq, "read_table", spy)
    cfg = C.load_config(world.config)
    out = ingest.read_serve_departures(cfg.paths.scored_file, cfg.airport_codes)
    scored_calls = [c for c in calls if c[0] == cfg.paths.scored_file.name]
    assert scored_calls and all(cols is not None and not set(cols) & set(ingest.LABEL_COLUMNS) for _, cols in scored_calls)
    assert not set(out.columns) & set(ingest.LABEL_COLUMNS)
    assert len(out) == len(world.dep) and (out.PHASE_mvt == "DEP").all()


def test_a_scored_frame_carrying_a_label_column_is_refused_even_if_all_null():
    """Rehearsed: guard_no_labels returning early -> RED."""
    f = pd.DataFrame({"MVT_ID_mvt": [1.0], "TAXITIME_SEC_mvt": [np.nan]})
    with pytest.raises(E.LabelLeakError, match="TAXITIME_SEC_mvt"):
        ingest.guard_no_labels(f, "scored rows")
    ingest.guard_no_labels(f.drop(columns="TAXITIME_SEC_mvt"), "scored rows")


def _write(tmp_path, frame, name="f.parquet"):
    p = tmp_path / name
    frame.to_parquet(p, index=False)
    return p


def test_a_missing_column_fails_loud_naming_the_column_and_the_file(world, tmp_path):
    """Rehearsed: check_file_schema's missing-column check deleted -> RED."""
    f = pd.read_parquet(world.raw["paths"]["scored_file"]).drop(columns=["ADEP_mvt"])
    p = _write(tmp_path, f, "scored_no_adep.parquet")
    with pytest.raises(E.MissingColumnError, match=r"scored_no_adep\.parquet.*ADEP_mvt"):
        ingest.check_file_schema(p, [ingest.RAW_BY_NAME[c] for c in ingest.SERVE_COLUMNS])


def test_a_column_of_the_wrong_kind_fails_loud(world, tmp_path):
    """A string clock and a naive (non-UTC) clock are both refused.
    Rehearsed: _arrow_kind_ok's timestamp branch returning True for any type -> RED."""
    f = pd.read_parquet(world.raw["paths"]["scored_file"])
    as_text = f.assign(MVT_TIME_UTC_mvt=f.MVT_TIME_UTC_mvt.astype(str))
    with pytest.raises(E.DtypeError, match="MVT_TIME_UTC_mvt"):
        ingest.check_file_schema(_write(tmp_path, as_text, "t1.parquet"), [ingest.RAW_BY_NAME["MVT_TIME_UTC_mvt"]])
    naive = f.assign(MVT_TIME_UTC_mvt=f.MVT_TIME_UTC_mvt.dt.tz_localize(None))
    with pytest.raises(E.DtypeError, match="MVT_TIME_UTC_mvt"):
        ingest.check_file_schema(_write(tmp_path, naive, "t2.parquet"), [ingest.RAW_BY_NAME["MVT_TIME_UTC_mvt"]])


def test_a_null_where_the_contract_forbids_it_fails_loud(world):
    """A scored departure without a scheduled time cannot be derived; the frame check names the column.
    Rehearsed: check_frame's nullability check deleted -> RED."""
    f = pd.read_parquet(world.raw["paths"]["scored_file"], columns=list(ingest.SERVE_COLUMNS))
    f = f[f.PHASE_mvt == "DEP"].copy()
    f.loc[f.index[0], "SCHED_TIME_UTC_mvt"] = pd.NaT
    with pytest.raises(E.NullabilityError, match="SCHED_TIME_UTC_mvt"):
        ingest.check_frame(f, [ingest.RAW_BY_NAME[c] for c in ingest.SERVE_COLUMNS], "scored",
                           not_null=ingest.SCORED_NOT_NULL)


def test_a_departure_at_an_airport_outside_the_registry_is_refused(world):
    """The eleven-airport file read with the ten-airport registry: EKCH is named, never routed silently.
    Rehearsed: check_airports' call deleted from read_serve_departures -> RED."""
    cfg = C.load_config(world.config)
    with pytest.raises(E.UnknownAirportError, match="EKCH"):
        ingest.read_serve_departures(cfg.paths.scored_file, [a for a in cfg.airport_codes if a != "EKCH"])


def test_the_template_join_keeps_file_order_and_refuses_an_id_without_a_departure(world):
    """Rehearsed: scored_rows' id-set check deleted -> RED."""
    cfg = C.load_config(world.config)
    sd = ingest.ingest_scored(cfg)
    assert np.array_equal(sd.scored.MVT_ID_mvt.to_numpy(), world.dep.MVT_ID_mvt.to_numpy())
    assert list(sd.scored.index) == list(range(len(sd.scored)))
    bad = pd.concat([sd.template, pd.DataFrame({"MVT_ID_mvt": [1.5]})], ignore_index=True)
    with pytest.raises(E.SchemaError, match="1 template ids have no departure"):
        ingest.scored_rows(sd.dep, bad)


def test_duplicate_ids_in_the_template_are_refused(world, tmp_path):
    """Rehearsed: load_template's uniqueness check deleted -> RED."""
    t = pd.read_parquet(world.raw["paths"]["template_file"])
    p = _write(tmp_path, pd.concat([t, t.head(1)], ignore_index=True), "tmpl.parquet")
    with pytest.raises(E.SchemaError, match="duplicate"):
        ingest.load_template(p)


def test_training_clock_rows_are_the_departures_in_file_order(world):
    """Rehearsed: the PHASE == DEP filter deleted -> RED."""
    files = C.load_config(world.config).paths.training_files()
    out = ingest.read_training_clock_rows(files)
    raw = pd.read_parquet(files[0])
    assert list(out.columns) == list(ingest.TRAINING_CLOCK_COLUMNS)
    assert np.array_equal(out.MVT_ID_mvt.to_numpy(), raw.MVT_ID_mvt[raw.PHASE_mvt == "DEP"].to_numpy())


def test_rome_frames_are_schema_checked_and_refuse_a_mislabelled_row(world, tmp_path):
    """Rehearsed: load_rome_frames' `unmatched` flag check deleted -> RED."""
    cfg = C.load_config(world.config)
    unm, lirf = ingest.load_rome_frames(cfg.paths.rome_cache)
    assert unm.unmatched.all() and not lirf.unmatched.any() and {"sp_band", "sp_fine"} <= set(unm.columns)
    bad = tmp_path / "rome"
    bad.mkdir()
    u = pd.read_parquet(cfg.paths.rome_cache / "unm.parquet")
    u.loc[0, "unmatched"] = False
    u.to_parquet(bad / "unm.parquet", index=False)
    pd.read_parquet(cfg.paths.rome_cache / "lirf.parquet").to_parquet(bad / "lirf.parquet", index=False)
    with pytest.raises(E.SchemaError, match="unmatched"):
        ingest.load_rome_frames(bad)
