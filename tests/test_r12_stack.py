"""Tests for scripts/analysis/r12_stack.py — v34's matched stages recomposed on an R12 out-of-fold F (the ADS-B residual
stack re-trained on the new F, then PIPE_M's lane composition), and served for 2026 through v34_compose.matched_values.
Synthetic fixtures; the stack itself is injected (its real code runs in the smoke). Mutation rehearsals in docstrings."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("r12_stack", "scripts/analysis/r12_stack.py")
B = S.B
COUNTS = {1: 4, 2: 3, 7: 4}                                 # cache rows: Jan 0-3, Feb 4-6, Jul 7-10


def _pred():
    """A holdout predictions.parquet as r12_validate writes it (Jan + Jul test rows, shuffled order)."""
    p = pd.DataFrame({"month": [7, 1, 1, 7, 1, 7, 1, 7], "pos": [1, 0, 2, 0, 1, 3, 3, 2],
                      "MVT_ID_mvt": [71.0, 10.0, 12.0, 70.0, 11.0, 73.0, 13.0, 72.0],
                      "ap": ["EGLL", "LFPG", "EGLL", "LFPG", "EGLL", "LFPG", "EGLL", "EGLL"],
                      "y": [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]})
    p["proxy"] = p.y + 200.0
    p["pred_r12"] = 150.0
    p["pred_yhead"] = p.y
    p["w"], p["doy"] = 1.0, np.where(p.month == 1, 5, 190)
    return p


def _train_table():
    rows = []
    for m, n in COUNTS.items():
        for pos in range(n):
            rows.append({"month": m, "pos": pos, "MVT_ID_mvt": float(m * 10 + pos), "ap": "EGLL" if pos % 2 else "LFPG",
                         "mvt_day": f"2025-{m:02d}-0{pos + 2}", "adsb_day_present": 1, "coverage": 3 if pos < 2 else 1})
    t = pd.DataFrame(rows)
    for c in B.FEATURES:
        t[c] = t.MVT_ID_mvt + 0.5
    return t[B.TABLE_COLS]


# ---------------------------------------------------------------- rows and the stack record

def test_month_pos_to_rows_inverts_rows_to_month_pos():
    """Fails when the month offsets are cumulated in the wrong order (February's rows would land in July)."""
    m, p = np.array([1, 2, 7, 7]), np.array([3, 0, 0, 3])
    rows = S.month_pos_to_rows(m, p, COUNTS)
    assert rows.tolist() == [3, 4, 7, 10]
    back = S.R.rows_to_month_pos(rows, COUNTS)
    assert back[0].tolist() == m.tolist() and back[1].tolist() == p.tolist()
    with pytest.raises(ValueError, match="position out of range"):
        S.month_pos_to_rows(np.array([2]), np.array([3]), COUNTS)
    with pytest.raises(ValueError, match="month 5"):
        S.month_pos_to_rows(np.array([5]), np.array([0]), COUNTS)


def test_stack_record_has_the_v34_record_contract_sorted_by_row():
    """row, month, ap, y, proxy, delta = proxy - y (exact, as in the fold records), treatment = the arm's raw delta."""
    rec = S.stack_record(_pred(), "r12", COUNTS)
    assert list(rec.columns) == S.RECORD_COLS
    assert rec.row.tolist() == [0, 1, 2, 3, 7, 8, 9, 10]
    np.testing.assert_array_equal(rec.delta.to_numpy(), rec.proxy.to_numpy() - rec.y.to_numpy())
    assert (rec.treatment == 150.0).all()


def test_stack_record_refuses_a_y_target_arm_and_nan():
    """The stack's F is a delta; a y-target head is not an F. Fails when yhead is let through."""
    with pytest.raises(ValueError, match="not a delta arm"):
        S.stack_record(_pred(), "yhead", COUNTS)
    p = _pred()
    p.loc[0, "pred_r12"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        S.stack_record(p, "r12", COUNTS)
    with pytest.raises(KeyError, match="pred_r12_noeobt"):
        S.stack_record(_pred(), "r12_noeobt", COUNTS)


def test_stack_table_is_the_v4_layout_row_aligned():
    """[row, MVT_ID_mvt] + FEATURES + [coverage], one row per record row, in record order; the ids are checked
    against the record's (via the train table). Fails when the table is not re-ordered to the record."""
    rec = S.stack_record(_pred(), "r12", COUNTS)
    t = S.stack_table(_train_table(), rec, COUNTS, allow_missing=False)
    assert list(t.columns) == ["row", "MVT_ID_mvt"] + B.FEATURES + ["coverage"]
    assert t.row.tolist() == rec.row.tolist()
    assert t.MVT_ID_mvt.tolist() == [10.0, 11.0, 12.0, 13.0, 70.0, 71.0, 72.0, 73.0]
    assert t.coverage.dtype.kind == "i"


def test_stack_table_missing_rows_strict_and_smoke():
    rec = S.stack_record(_pred(), "r12", COUNTS)
    tt = _train_table()
    tt = tt[~((tt.month == 7) & (tt.pos == 3))]
    with pytest.raises(ValueError, match="1 record rows have no R12 table row"):
        S.stack_table(tt, rec, COUNTS, allow_missing=False)
    t = S.stack_table(tt, rec, COUNTS, allow_missing=True)
    assert t.coverage.iloc[-1] == 0 and np.isnan(t.dwell_s.iloc[-1])


def test_joined_shares_janjul_2025_ex_spent_day_and_2026():
    """c25 over the Jan + Jul 2025 table rows without the spent day (2025-01-09, as adsb_ship.tables_with_airport),
    c26 over the serve table; share = coverage >= 2."""
    tt = pd.DataFrame({"ap": ["A", "A", "A", "A", "B"], "month": [1, 1, 7, 2, 7],
                       "mvt_day": ["2025-01-02", "2025-01-09", "2025-07-02", "2025-02-02", "2025-07-02"],
                       "coverage": [3, 0, 1, 3, 2]})
    sv = pd.DataFrame({"ap": ["A", "B", "B"], "coverage": [2, 0, 3]})
    c25, c26 = S.joined_shares(tt, sv)
    assert c25 == {"A": 0.5, "B": 1.0} and c26 == {"A": 1.0, "B": 0.5}


# ---------------------------------------------------------------- composition

def _inc():
    return pd.DataFrame({"row": [0, 1, 2, 3], "MVT_ID_mvt": [10.0, 11.0, 12.0, 13.0], "y": [400.0, 700.0, 500.0, 900.0],
                         "proxy": [600.0, 900.0, 700.0, 1100.0], "pop": [True, False, True, False],
                         "taxi_C": [390.0, 0.0, 510.0, 0.0], "pipe": [395.0, 710.0, 505.0, 880.0],
                         "month": 1, "doy": [5, 5, 6, 6], "ap": ["LFPG", "EGLL", "EGLL", "EGLL"]})


def test_compose_oof_variants():
    """single = max(proxy - F, 1); stacked_v34lanes = PIPE_M on pop rows (v34's CBV lane untouched), the new stage
    max(proxy - (F + resid), 1) elsewhere; stacked_r12partner = 0.5 (taxi_C + max(proxy - F, 1)) on pop rows. Fails
    when the pop rows take the stage value or the stage ignores the residual."""
    inc = _inc()
    F, resid = np.full(4, 150.0), np.array([0.0, 20.0, 0.0, -30.0])
    v = S.compose_oof(inc, F, resid)
    np.testing.assert_allclose(v["single"], [450, 750, 550, 950])
    np.testing.assert_allclose(v["stacked_v34lanes"], [395, 730, 505, 980])
    np.testing.assert_allclose(v["stacked_r12partner"], [420, 730, 530, 980])
    np.testing.assert_allclose(v["PIPE_M"], inc["pipe"])


def test_compose_oof_refuses_misaligned_inputs():
    with pytest.raises(ValueError, match="length"):
        S.compose_oof(_inc(), np.full(3, 1.0), np.zeros(4))
    with pytest.raises(ValueError, match="non-finite"):
        S.compose_oof(_inc(), np.array([1.0, np.nan, 1.0, 1.0]), np.zeros(4))


def test_align_incumbent_by_row_with_label_guard():
    """The incumbent (334,132 rows) and the stack frame are joined by `row`; y must agree. Fails on a positional join."""
    inc = _inc().iloc[::-1].reset_index(drop=True)
    fr = pd.DataFrame({"row": [0, 1, 2, 3, 9], "y": [400.0, 700.0, 500.0, 900.0, 1.0], "F": 150.0,
                       "ap": ["LFPG", "EGLL", "EGLL", "EGLL", "X"]})
    a, idx = S.align_incumbent(inc, fr)
    assert a.row.tolist() == [3, 2, 1, 0] and idx.tolist() == [3, 2, 1, 0]
    with pytest.raises(AssertionError, match="ap differs"):
        S.align_incumbent(inc, fr.assign(ap="ZZZZ"))
    bad = fr.assign(y=fr.y + 1)
    with pytest.raises(AssertionError, match="y differs"):
        S.align_incumbent(inc, bad)
    with pytest.raises(ValueError, match="incumbent rows missing"):
        S.align_incumbent(inc, fr.iloc[:2])


# ---------------------------------------------------------------- the orchestration with an injected stack

def _deps(tmp_path, calls):
    def load_frame(rec_path, table_path):
        rec, t = pd.read_parquet(rec_path), pd.read_parquet(table_path)
        assert (rec.row.to_numpy() == t.row.to_numpy()).all()
        fr = pd.concat([rec, t.drop(columns="row")], axis=1).rename(columns={"treatment": "F"})
        fr["fold"] = np.arange(len(fr)) % 5
        fr["hour"] = 12
        fr["day"] = "2025-01-02"
        return fr

    def train(fr, feats, out_dir, seeds):
        calls["train"] = {"n": len(fr), "seeds": list(seeds), "feats": list(feats)}
        pathlib.Path(out_dir).mkdir(parents=True)
        (pathlib.Path(out_dir) / "MANIFEST.json").write_text(json.dumps({"gate": {}, "feats": list(feats)}))
        return {}

    def predict(fr, models_dir, oof):
        man = json.loads((pathlib.Path(models_dir) / "MANIFEST.json").read_text())
        ok = (fr.coverage.to_numpy() >= 2) & fr.ap.map(lambda a: man["gate"].get(a, {}).get("allowed", False)).to_numpy()
        calls["predict_oof"] = oof
        return np.where(ok, 25.0, 0.0)

    def inc():
        i = _inc()
        return pd.concat([i, i.assign(row=[7, 8, 9, 10], MVT_ID_mvt=[70.0, 71.0, 72.0, 73.0],
                                      y=[600.0, 300.0, 1000.0, 800.0], proxy=[800.0, 500.0, 1200.0, 1000.0],
                                      month=7, doy=190, pop=[False, True, True, False],
                                      ap=["LFPG", "EGLL", "EGLL", "LFPG"])], ignore_index=True)
    return dict(load_frame=load_frame, train=train, transfer=lambda fr, feats: {"rule": {"PASS": True, "n_pass": 4}},
                predict=predict, incumbent=inc, feats=["dwell_s", "joined"],
                gate=lambda c25, c26, ok: {a: {"allowed": a == "LFPG", "rule": "t", "c25": c25.get(a, 0.0),
                                               "c26": c26.get(a, 0.0)} for a in sorted(set(c25) | set(c26))},
                v4_gate={"LFPG": {"allowed": True, "rule": "t", "c25": 0.0, "c26": 0.0}})


def test_stack_arm_end_to_end(tmp_path):
    """record + table written, stack trained on them, the gate written into the manifest, the OOF residual composed,
    the report written (single and stacked vs PIPE_M). Observable state: every output exists once; a rerun refuses."""
    calls = {}
    serve = pd.DataFrame({"ap": ["LFPG", "EGLL"], "coverage": [3, 3]})
    out = tmp_path / "logs" / "r12_scratch" / "stack"
    rep = S.stack_arm("r12", _pred(), _train_table(), serve, out, _deps(tmp_path, calls), COUNTS, smoke=True,
                      root=tmp_path, n_boot=50, log=lambda m: None)
    for f in ("record_r12.parquet", "table_r12.parquet", "report_r12.json", "oof_r12.parquet"):
        assert (out / f).exists()
    man = json.loads((out / "models_r12" / "MANIFEST.json").read_text())
    assert man["gate"]["LFPG"]["allowed"] is True and man["gate_evidence"]["transfer_PASS"] is True
    assert calls["predict_oof"] is True and calls["train"]["seeds"] == [0]
    oof = pd.read_parquet(out / "oof_r12.parquet")
    lfpg_cov = oof[(oof.ap == "LFPG") & (oof.coverage >= 2) & ~oof["pop"]]
    assert len(lfpg_cov) == 1 and (lfpg_cov.resid == 25.0).all() and (oof[oof.ap == "EGLL"].resid == 0.0).all()
    back = json.loads((out / "report_r12.json").read_text())
    assert set(back["vs_PIPE_M"]) == {"single", "stacked_v34lanes", "stacked_r12partner"}
    assert back["gate_equals_models_v4"] is False and back["n_rows"] == 8
    with pytest.raises(FileExistsError):
        S.stack_arm("r12", _pred(), _train_table(), serve, out, _deps(tmp_path, {}), COUNTS, smoke=True, root=tmp_path,
                    n_boot=50, log=lambda m: None)


def test_stack_arm_refuses_a_gate_written_by_training(tmp_path):
    """The gate is derived AFTER training (transfer + joined shares); a manifest that already carries one is refused,
    never overwritten (mutation rehearsal #11)."""
    calls = {}
    d = _deps(tmp_path, calls)
    base_train = d["train"]

    def train(fr, feats, out_dir, seeds):
        base_train(fr, feats, out_dir, seeds)
        (pathlib.Path(out_dir) / "MANIFEST.json").write_text(json.dumps({"gate": {"LFPG": True}}))
    d["train"] = train
    with pytest.raises(AssertionError, match="wrote a gate"):
        S.stack_arm("r12", _pred(), _train_table(), pd.DataFrame({"ap": ["LFPG"], "coverage": [3]}),
                    tmp_path / "logs" / "r12_scratch" / "g", d, COUNTS, smoke=True, root=tmp_path, n_boot=20,
                    log=lambda m: None)


def test_compare_stacked_arms():
    """Arm vs arm at the stacked level (the value of the 10 extra months = r12 stacked vs janjul_adsb stacked)."""
    y = np.array([1.0, 2.0, 3.0, 4.0])
    oofs = {"r12": pd.DataFrame({"y": y, "stacked_v34lanes": y + 1, "single": y + 2, "w": 1.0, "block": [1, 1, 2, 2]}),
            "janjul_adsb": pd.DataFrame({"y": y, "stacked_v34lanes": y + 3, "single": y + 3, "w": 1.0,
                                         "block": [1, 1, 2, 2]})}
    c = S.compare_arms(oofs, n_boot=50)
    g = c["r12_vs_janjul_adsb"]["stacked_v34lanes"]["unweighted"]
    assert g["gain_bmse"] == pytest.approx(4 * (9 - 1) / 344_841)
    with pytest.raises(ValueError, match="differ in rows"):
        S.compare_arms({"a": oofs["r12"], "b": oofs["r12"].iloc[:3]}, n_boot=50)


# ---------------------------------------------------------------- serving

def test_serve_stack_table_indexed_by_id():
    """The 2026 stage table for prc.pipeline.adsb_stage.stage_frame: indexed by MVT_ID_mvt, MVT_ID_mvt + FEATURES +
    coverage only (ap / hour / proxy / F would collide with the stage's own columns)."""
    sv = _train_table().rename(columns={})
    t = S.serve_stack_table(sv, sv.MVT_ID_mvt.to_numpy()[:3], allow_missing=False)
    assert list(t.columns) == ["MVT_ID_mvt"] + B.FEATURES + ["coverage"]
    assert t.index.tolist() == [10.0, 11.0, 12.0]
    with pytest.raises(ValueError, match="serve ids have no R12 table row"):
        S.serve_stack_table(sv, np.array([999.0]), allow_missing=False)
    s = S.serve_stack_table(sv, np.array([999.0]), allow_missing=True)
    assert s.coverage.tolist() == [0]


def test_gate_bools():
    assert S.gate_bools({"A": {"allowed": True}, "B": {"allowed": False}}) == {"A": True, "B": False}


def test_regression_bug_r12_2_serve_passes_every_step_report_new_inputs_reads():
    """2026-09-25 serve smoke: KeyError 3 — v34_compose.new_inputs reads the reports of steps 3, 4, 5 and 6, the wrapper
    passed 4-6. Bug class: an untested contract with a shared function's inputs. V34_INPUT_STEPS is now pinned to the
    steps new_inputs actually indexes (read from its source). Fails when a step is dropped from the constant."""
    import inspect
    import re
    V = S._load("v34_compose", "scripts/analysis/v34_compose.py")
    src = inspect.getsource(V.new_inputs)
    read = {int(n) for n in re.findall(r"preds\((\d)\)", src)} | {int(n) for n in re.findall(r"reports\[(\d)\]", src)}
    assert read and set(S.V34_INPUT_STEPS) == read


def test_r13_stack_record_accepts_the_cat_and_derived_delta_arms():
    """R13: cat (a fitted delta arm) and the derived delta blends lgbcat / blendy are stack Fs; yhead still is not."""
    p = _pred()
    for arm in ("cat", "lgbcat", "blendy"):
        p[f"pred_{arm}"] = 175.0
        rec = S.stack_record(p, arm, COUNTS)
        assert (rec.treatment == 175.0).all()
    with pytest.raises(ValueError, match="not a delta arm"):
        S.stack_record(p, "yhead", COUNTS)
    with pytest.raises(ValueError, match="not a delta arm"):
        S.stack_record(p.assign(pred_other=1.0), "other", COUNTS)
