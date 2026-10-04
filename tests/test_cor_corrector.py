"""scripts/analysis/cor_corrector.py — COR: a corrector cross-fitted on FM1's 12-month out-of-fold base predictions.

Registration: plans/PREREG_cor_crossfit_corrector_2026_10_04.md (REG 3c75c5ea, AMEND-A1 6c62c14b). Constructed rows only:
the OOF loader's refusals, the encoding-free design, the label-checked join, the month split (January / July never
train or early-stop the corrector), the corrector itself on a planted residual, the serve rule (matched rows only,
floored at 1 s), encodings fitted on training months only (COR-E), and an end-to-end run with a stub feature builder.
The real feature path (FM1's build_frames) is exercised on the real-data-shaped world (tests/fm1/test_fm1_cor_builder.py).

Written first and seen red 2026-10-04 (no module). Mutation rehearsal before any OOF exists (scratch copy, one at a
time): 10 / 10 killed — CO1 January in the training months, CO2 no label check, CO3 unmatched rows corrected, CO4 no
1 s floor, CO5 COR-E encodings fitted on every month (killed by the spy test), CO6 early stopping on the training data
(survived until the validation set was pinned), CO7 encodings kept in the design, CO8 the +150 bar ignored, CO9 every
month scored, CO10 an incomplete OOF set accepted.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("cor_corrector", ROOT / "scripts" / "analysis" / "cor_corrector.py")
C = importlib.util.module_from_spec(_spec)
sys.modules["cor_corrector"] = C
_spec.loader.exec_module(C)


def _oof(n_per_month=60, months=range(1, 13), seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for m in months:
        y = rng.uniform(300, 2500, n_per_month)
        parts.append(pd.DataFrame({
            "MVT_ID_mvt": m * 10_000 + np.arange(n_per_month, dtype=float), "month": m,
            "day": [f"2025-{m:02d}-{1 + i % 20:02d}" for i in range(n_per_month)],
            "ap": np.where(np.arange(n_per_month) % 2, "EDDF", "LIRF"), "y": y,
            "pred": np.rint(y + rng.normal(0, 100, n_per_month)),
            "lane": np.where(np.arange(n_per_month) % 10 == 0, "unmatched", "matched")}))
    return pd.concat(parts, ignore_index=True)


def _write_oof(tmp_path, frame):
    d = tmp_path / "oof"
    d.mkdir()
    for m, g in frame.groupby("month"):
        g.to_parquet(d / f"oof_M{int(m):02d}.parquet", index=False)
    return d


# ---------------------------------------------------------------------------------------------- loading

def test_the_oof_set_must_hold_twelve_months_of_unique_rows(tmp_path):
    frame = _oof()
    assert len(C.load_oof(_write_oof(tmp_path, frame))) == len(frame)
    (tmp_path / "a").mkdir()
    with pytest.raises(ValueError, match="months"):
        C.load_oof(_write_oof(tmp_path / "a", frame[frame.month != 5]))
    dup = pd.concat([frame, frame[frame.month == 3].head(1)], ignore_index=True)
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError, match="duplicate"):
        C.load_oof(_write_oof(tmp_path / "b", dup))


def test_the_design_drops_every_encoding_and_stays_label_free():
    cols = C.raw_feature_columns(["sp", "te_ADEP_mvt", "de_ars", "q_dep_1h", "adsb_x"])
    assert cols == ["sp", "q_dep_1h", "adsb_x"]
    with pytest.raises(Exception):
        C.raw_feature_columns(["sp", "TAXITIME_SEC_mvt"])                 # FM1's design-clean refusal


# ---------------------------------------------------------------------------------------------- join and split

def test_the_join_keeps_matched_rows_and_refuses_a_label_mismatch():
    oof = _oof()
    feats = oof.loc[oof.lane == "matched", ["MVT_ID_mvt", "y"]].assign(x1=1.0)
    j = C.join(oof, feats)
    assert len(j) == int((oof.lane == "matched").sum()) and set(j.lane) == {"matched"}
    with pytest.raises(ValueError, match="labels disagree"):
        C.join(oof, feats.assign(y=feats.y + 5.0))
    with pytest.raises(ValueError, match="no feature row"):
        C.join(oof, feats.iloc[1:])


def test_january_and_july_never_train_or_early_stop_the_corrector():
    j = _oof().assign(x1=0.0)
    tr, es, ev = C.split(j)
    assert set(tr.month) == set(C.TRAIN_MONTHS) and set(es.month) == set(C.ES_MONTHS) and set(ev.month) == {1, 7}
    assert not ({1, 7} & (set(C.TRAIN_MONTHS) | set(C.ES_MONTHS)))


# ---------------------------------------------------------------------------------------------- the corrector

@pytest.mark.slow
def test_the_corrector_learns_a_planted_residual_and_early_stops_on_months_3_and_9():
    rng = np.random.default_rng(1)
    j = _oof(n_per_month=400).query("lane == 'matched'").copy()
    j["x1"] = rng.normal(0, 1, len(j))
    j["pred"] = j.y - 120.0 * j.x1                                      # the base misses by 120 * x1
    tr, es, ev = C.split(j)
    booster = C.fit_corrector(tr, es, ["x1", "pred"])
    corr = booster.predict(ev[["x1", "pred"]].to_numpy())
    before = float(((ev.y - ev.pred) ** 2).mean())
    after = float(((ev.y - (ev.pred + corr)) ** 2).mean())
    assert after < 0.2 * before and booster.best_iteration < C.PARAMS_ROUNDS


def test_serving_moves_matched_rows_only_and_floors_at_one_second():
    oof = _oof(months=(1, 7))
    corr = pd.Series(-10_000.0, index=oof.loc[oof.lane == "matched", "MVT_ID_mvt"].to_numpy())
    out = C.apply_correction(oof, corr)
    m = oof.lane == "matched"
    assert (out.loc[m, "pred"] == 1.0).all()                            # floored
    assert (out.loc[~m, "pred"].to_numpy() == oof.loc[~m, "pred"].to_numpy()).all()


def test_cor_e_encodings_are_fitted_on_training_rows_only():
    tr = pd.DataFrame({"STAND_mvt": ["A", "A", "B"], "y": [100.0, 300.0, 1000.0], "pred": [150.0, 250.0, 900.0],
                       "delta": [0.0, 0.0, 0.0]})
    ev = pd.DataFrame({"STAND_mvt": ["A", "C"]})
    enc = C.training_encodings(tr, keys=["STAND_mvt"], smooth=0.0)
    out = enc(ev)
    assert out["te_STAND_mvt"].tolist() == [200.0, pytest.approx(tr.y.mean())]      # unseen C -> the training prior
    assert out["re_STAND_mvt"][0] == pytest.approx(0.0)                              # A's mean residual: (-50 + 50) / 2


# ---------------------------------------------------------------------------------------------- end to end

@pytest.mark.slow
def test_a_run_scores_only_january_and_july_and_reports_the_gate(tmp_path):
    rng = np.random.default_rng(2)
    oof = _oof(n_per_month=300)
    x1 = pd.Series(rng.normal(0, 1, len(oof)), index=oof.MVT_ID_mvt.to_numpy())
    oof["pred"] = np.where(oof.lane == "matched", np.maximum(oof.y - 150.0 * x1.to_numpy(), 1.0), oof.pred)

    def builder(months):
        f = oof.loc[oof.month.isin(months) & (oof.lane == "matched"), ["MVT_ID_mvt", "y"]].copy()
        f["x1"] = x1.reindex(f.MVT_ID_mvt.to_numpy()).to_numpy()
        f["adsb_cov"] = 0.0
        f["STAND_mvt"] = "S1"
        return f
    rep = C.run(_write_oof(tmp_path, oof), builder, feature_cols=["x1", "adsb_cov"], adsb_cols=["adsb_cov"],
                keys=["STAND_mvt"], n_boot=200)
    assert rep["primary"]["compare"]["n"] == int(oof.month.isin([1, 7]).sum())
    assert set(rep["primary"]["compare"]["by_month"]) == {"1", "7"}
    assert rep["primary"]["gate"]["promote"] is True and rep["verdict"] in ("TRUE", "FALSE")
    assert {"cor_e", "no_adsb"} <= set(rep["secondary"])


@pytest.mark.parametrize("promote, body, want", [(True, 150.0, "TRUE"), (True, 149.99, "FALSE"), (False, 900.0, "FALSE")])
def test_the_verdict_needs_the_gate_and_a_resolvable_body_gain(promote, body, want):
    assert C.verdict_of(promote, body) == want


def test_cor_e_encodings_in_a_run_are_fitted_on_the_training_months_only(tmp_path, monkeypatch):
    """The run must hand training_encodings the training-month rows, never January / July / early-stopping rows."""
    seen = {}
    real = C.training_encodings

    def spy(train, keys, smooth):
        seen["months"] = sorted(set(train.month.astype(int)))
        return real(train, keys, smooth)
    monkeypatch.setattr(C, "training_encodings", spy)
    monkeypatch.setattr(C, "_arm", lambda j, oof_eval, cols, n_boot: {"compare": {"body": {"gain_bmse": 0.0}},
                                                                       "gate": {"promote": False}})
    oof = _oof()

    def builder(months):
        f = oof.loc[oof.lane == "matched", ["MVT_ID_mvt", "y"]].copy()
        f["x1"], f["STAND_mvt"] = 0.0, "S1"
        return f
    C.run(_write_oof(tmp_path, oof), builder, feature_cols=["x1"], adsb_cols=[], keys=["STAND_mvt"], n_boot=10)
    assert seen["months"] == sorted(C.TRAIN_MONTHS)


def test_early_stopping_watches_the_es_months_residual_and_nothing_else(monkeypatch):
    """The validation set LightGBM early-stops on is exactly the ES months' residual (CO6: training data as the
    validation set would stop on a curve that never turns up)."""
    import lightgbm as lgb
    captured = {}

    def fake_train(params, train_set, num_boost_round, valid_sets, callbacks):
        captured["valid"] = [v.get_label() for v in valid_sets]
        captured["train"] = train_set.get_label()
        raise RuntimeError("captured")
    monkeypatch.setattr(lgb, "train", fake_train)
    j = _oof().query("lane == 'matched'").assign(x1=0.0)
    tr, es, _ = C.split(j)
    with pytest.raises(RuntimeError, match="captured"):
        C.fit_corrector(tr, es, ["x1", "pred"])
    assert len(captured["valid"]) == 1
    assert np.allclose(captured["valid"][0], (es.y - es.pred).to_numpy(float), rtol=0, atol=1e-6)
