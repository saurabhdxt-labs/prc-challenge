"""prc.pipeline.adsb_stage — the ADS-B stack stage's guards, against a stub predictor (the real one is prc-challenge-6e's)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prc.pipeline import adsb_stage as S
from prc.pipeline import errors as E

IDS = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
META = {b"git_sha": b"abc", b"extractor": b"parse_member_v2", b"features_version": b"v2"}


def _table(tmp_path, ids=IDS, coverage=(3, 2, 1, 0, 3), meta=META, extra=None, name="t.parquet"):
    f = pd.DataFrame({"MVT_ID_mvt": np.asarray(ids, dtype="float64"), "coverage": np.asarray(coverage, dtype="int64"),
                      "first_ground_rel_aobt3": np.linspace(-300, 300, len(ids))})
    if extra:
        for k, v in extra.items():
            f[k] = v
    t = pa.Table.from_pandas(f, preserve_index=False)
    p = tmp_path / name
    pq.write_table(t.replace_schema_metadata({**(t.schema.metadata or {}), **meta}), p)
    return p


def _frame(tmp_path, gate_ap=("EHAM", "EHAM", "EHAM", "EHAM", "LTFM")):
    table, _ = S.read_table(_table(tmp_path), IDS)
    return S.stage_frame(IDS, list(gate_ap), [8.0] * 5, [1000.0, 1200.0, 900.0, 800.0, 1500.0],
                         [100.0, 50.0, 0.0, -20.0, 300.0], table)


def test_read_table_accepts_a_complete_stamped_table(tmp_path):
    t, meta = S.read_table(_table(tmp_path), IDS)
    assert list(t.index) == IDS.tolist() and meta["features_version"] == "v2"


@pytest.mark.parametrize("kw,err,msg", [
    (dict(meta={b"git_sha": b"abc"}), E.SchemaError, "provenance"),
    (dict(extra={"TAXITIME_SEC_mvt": [1, 2, 3, 4, 5]}), E.LabelLeakError, "label column"),
    (dict(ids=[10.0, 10.0, 12.0, 13.0, 14.0]), E.SchemaError, "duplicate"),
    (dict(ids=[10.0, 11.0, 12.0, 13.0, 99.0]), E.SchemaError, "1 scored matched flights missing, 1 unexpected"),
    (dict(coverage=(3, 2, 1, 0, 7)), E.SchemaError, "unknown coverage"),
])
def test_read_table_refusals(tmp_path, kw, err, msg):
    with pytest.raises(err, match=msg):
        S.read_table(_table(tmp_path, **kw), IDS)
    with pytest.raises(E.SchemaError, match="does not exist"):
        S.read_table(tmp_path / "nope.parquet", IDS)


def test_read_gate(tmp_path):
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"gate": {"EHAM": True, "LTFM": False}}))
    gate, man = S.read_gate(tmp_path)
    assert gate == {"EHAM": True, "LTFM": False}
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"gate": {"EHAM": {"allowed": True, "rule": "ADN.7", "c25": 0.7, "c26": 0.8},
                                                                "LTFM": {"allowed": False, "rule": "coverage"}}}))
    assert S.read_gate(tmp_path)[0] == {"EHAM": True, "LTFM": False}                   # 6e's entry form
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"gate": {"EHAM": {"rule": "x"}}}))
    with pytest.raises(E.SchemaError, match="booleans"):
        S.read_gate(tmp_path)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"gate": {"EHAM": "yes"}}))
    with pytest.raises(E.SchemaError, match="booleans"):
        S.read_gate(tmp_path)
    (tmp_path / "MANIFEST.json").write_text(json.dumps({}))
    with pytest.raises(E.SchemaError, match="no `gate`"):
        S.read_gate(tmp_path)


def test_apply_moves_only_covered_rows_at_gated_airports(tmp_path):
    """Rows 0, 1 (EHAM, coverage 3 / 2) are eligible; row 2 (coverage 1), row 3 (0), row 4 (LTFM, not gated) are not.
    Fails if the eligibility mask or the value formula changes."""
    fr = _frame(tmp_path)
    stub = lambda f: np.where((f.coverage >= 2) & (f.ap == "EHAM"), -40.0, 0.0).astype("float64")
    v, info = S.apply_stage(fr, stub, {"EHAM": True, "LTFM": False})
    base = np.rint(np.maximum(fr.proxy - fr.F, 1.0)).to_numpy()
    assert v.tolist() == [base[0] + 40, base[1] + 40, base[2], base[3], base[4]]
    assert info["n_eligible"] == 2 and info["n_changed"] == 2 and info["changed_by_airport"] == {"EHAM": 2}


def test_zero_residual_is_exactly_the_lane_without_the_stage(tmp_path):
    fr = _frame(tmp_path)
    v, info = S.apply_stage(fr, lambda f: np.zeros(len(f)), {"EHAM": True})
    assert np.array_equal(v, np.rint(np.maximum(fr.proxy - fr.F, 1.0)).to_numpy()) and info["n_changed"] == 0


@pytest.mark.parametrize("stub,msg", [
    (lambda f: np.full(len(f), -10.0), "uncovered or at an ungated airport"),
    (lambda f: np.zeros(len(f) - 1), "expected"),
    (lambda f: np.where(f.coverage >= 2, np.nan, 0.0), "non-finite"),
    (lambda f: [0.0] * len(f), "returned list"),
])
def test_apply_refuses_a_predictor_that_breaks_the_contract(tmp_path, stub, msg):
    with pytest.raises(E.LaneError, match=msg):
        S.apply_stage(_frame(tmp_path), stub, {"EHAM": True, "LTFM": True})


def test_stage_frame_refuses_misaligned_inputs(tmp_path):
    table, _ = S.read_table(_table(tmp_path), IDS)
    with pytest.raises(E.LaneError, match="'F' has 4 values"):
        S.stage_frame(IDS, ["EHAM"] * 5, [8.0] * 5, [1.0] * 5, [1.0] * 4, table)


def test_legacy_adsb_predictor_is_6es_entry_point_and_refuses_when_it_is_absent(monkeypatch):
    """legacy.adsb_predictor() hands back adsb_stack.predict_residual with the contract signature (frame, model_dir, oof);
    a module without it is refused by name, never silently replaced. Rehearsed 2026-09-11 (c4): the getattr default swapped
    for a zero-residual lambda -> RED."""
    import inspect
    from types import SimpleNamespace

    from prc.pipeline import legacy
    fn = legacy.adsb_predictor()
    assert fn is legacy.adsb_stack().predict_residual
    assert list(inspect.signature(fn).parameters) == ["frame", "model_dir", "oof"]
    assert inspect.signature(fn).parameters["oof"].default is False
    monkeypatch.setattr(legacy, "adsb_stack", lambda: SimpleNamespace())
    with pytest.raises(NotImplementedError, match="no predict_residual"):
        legacy.adsb_predictor()


def test_coverage_transfer_reports_per_airport_shares_and_the_2026_to_2025_ratio():
    """Guard (3) of the stage design: per airport, the share of rows the stack may move (coverage >= 2) and seen from the
    stand (coverage 3) in each period, and the 2026 / 2025 ratio of the movable share -- ADN's 2025 gain transfers only
    where coverage holds. Rehearsed 2026-09-11 (c4): `>= MIN_COVERAGE` flipped to `> MIN_COVERAGE` -> RED; ratio inverted -> RED."""
    c25 = pd.DataFrame({"ap": ["EHAM"] * 4 + ["LTFM"] * 2, "coverage": [3, 2, 1, 0, 2, 1]})
    c26 = pd.DataFrame({"ap": ["EHAM"] * 2 + ["LTFM"] * 4 + ["LEMD"], "coverage": [3, 3, 3, 1, 1, 1, 2]})
    rep = S.coverage_transfer(c25.ap, c25.coverage, c26.ap, c26.coverage)
    assert rep["EHAM"] == {"n25": 4, "n26": 2, "movable25": 0.5, "movable26": 1.0, "stand25": 0.25, "stand26": 1.0, "ratio": 2.0}
    assert rep["LTFM"] == {"n25": 2, "n26": 4, "movable25": 0.5, "movable26": 0.25, "stand25": 0.0, "stand26": 0.25, "ratio": 0.5}
    assert rep["LEMD"]["n25"] == 0 and rep["LEMD"]["movable25"] is None and rep["LEMD"]["ratio"] is None
    zero = S.coverage_transfer(["LIRF"] * 2, [1, 0], ["LIRF"], [3])["LIRF"]     # nothing movable in 2025: no ratio, not inf
    assert zero["movable25"] == 0.0 and zero["movable26"] == 1.0 and zero["ratio"] is None
    with pytest.raises(E.SchemaError, match="unknown coverage"):
        S.coverage_transfer(c25.ap, c25.coverage.replace(0, 9), c26.ap, c26.coverage)
    with pytest.raises(E.LaneError, match="length"):
        S.coverage_transfer(c25.ap[:-1], c25.coverage, c26.ap, c26.coverage)


def _models(tmp_path, gate=None, feats=("first_ground_rel_aobt3",), drop=None):
    d = tmp_path / "models"
    d.mkdir(parents=True, exist_ok=True)
    man = {"feats": list(feats), "seeds": [0, 1], "folds": 2, "airports": ["EHAM", "LTFM"],
           "gate": {"EHAM": {"allowed": True, "rule": "r", "c25": 0.7, "c26": 0.8}} if gate is None else gate}
    (d / "MANIFEST.json").write_text(json.dumps(man))
    for s in (0, 1):
        for k in (0, 1):
            for name in (f"G_s{s}_k{k}.txt", f"N_s{s}_k{k}.pt", f"N_s{s}_k{k}_stats.npz"):
                if name != drop:
                    (d / name).write_bytes(b"x")
    return d


def test_check_stage_inputs_reports_a_complete_stage_and_names_what_is_missing(tmp_path):
    """The dry run's stage check: table, gate, every seed x fold model file and the manifest's features, before anything runs.
    Rehearsed 2026-09-11 (c4): the model-file loop skipped -> RED; the feats check skipped -> RED; the gate-airport check
    skipped -> RED."""
    table = _table(tmp_path)
    rep = S.check_stage_inputs(table, _models(tmp_path), IDS)
    assert rep["n_rows"] == 5 and rep["gate_allowed"] == ["EHAM"] and rep["n_model_files"] == 12
    assert rep["movable_share"] == 0.6 and rep["table_meta"]["features_version"] == "v2"
    with pytest.raises(E.SchemaError, match=r"model file\(s\) \['N_s1_k0.pt'\] missing"):
        S.check_stage_inputs(table, _models(tmp_path / "a", drop="N_s1_k0.pt"), IDS)
    with pytest.raises(E.MissingColumnError, match="dwell_s"):
        S.check_stage_inputs(table, _models(tmp_path / "b", feats=("first_ground_rel_aobt3", "dwell_s")), IDS)
    with pytest.raises(E.SchemaError, match="not in the manifest's airports"):
        S.check_stage_inputs(table, _models(tmp_path / "c", gate={"EHMA": True}), IDS)
    with pytest.raises(E.SchemaError, match="no `gate`"):
        S.check_stage_inputs(table, _models(tmp_path / "d", gate={}), IDS)
    with pytest.raises(E.SchemaError, match="missing"):
        S.check_stage_inputs(table, _models(tmp_path / "e"), IDS[:-1])
