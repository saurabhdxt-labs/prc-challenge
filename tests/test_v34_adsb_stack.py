"""scripts/analysis/v34_adsb_stack.py — v34 step 2 (AMENDMENT V34.2): the ADS-B stack (models_v4 recipe: same 33 feats, same
seeds, same 5 day folds, ship learner B) retrained on step 1's F-NSC fold record, the ADN.7 transfer test re-run on it, the
gate re-derived and required to equal models_v4's, and the shipped configuration's OOF gain vs F-NSC (clause C1).

Fakes and a small synthetic world only (no real data, no lock, no torch). Mutation rehearsal 2026-09-24: see docstrings."""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name, rel):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A2 = _load("v34_adsb_stack", "scripts/analysis/v34_adsb_stack.py")
AS = A2.AS
SH = A2.SH


def test_inputs_are_the_models_v4_recipes():
    assert A2.MODELS_V4 == ROOT / "data/adsb/v2/models_v4"
    assert A2.T25 == ROOT / "data/adsb/v2/features_2025janjul_v4.parquet"
    assert A2.T26 == ROOT / "data/adsb/v2/features_2026janjul_v4.parquet"
    assert A2.MODELS_NAME == "models_v4_nsc"
    assert A2.V4_OVERRIDES == {"LIRF": "RWC.3 SHIP"}


@pytest.mark.devtree
def test_v4_contract_reads_the_shipped_manifest_and_holds_no_seconds():
    c = A2.v4_contract()
    assert len(c["feats"]) == 33 and c["seeds"] == [0, 1, 2] and c["folds"] == 5 and c["ship_learner"] == "B"
    assert all(v["allowed"] for v in c["gate"].values()) and len(c["gate"]) == 10
    A2.C.assert_no_seconds(c["feats"], "models_v4 feats")


# ---------------------------------------------------------------- load_frame parity with adsb_stack.load_frame

def _world(tmp_path, n=60):
    rng = np.random.default_rng(3)
    days = [f"2025-01-{d:02d}" for d in range(1, 11)] + [f"2025-07-{d:02d}" for d in range(1, 11)]
    ids = np.arange(n, dtype=float) + 1000
    t = pd.to_datetime([days[i % len(days)] + f" {(i % 20) + 2:02d}:00:00" for i in range(n)], utc=True)
    proxy = rng.uniform(400, 900, n)
    delta = rng.normal(0, 60, n)
    rec = pd.DataFrame({"row": np.arange(n, dtype="int64"), "month": t.month.astype("int32"),
                        "ap": np.where(np.arange(n) % 2 == 0, "EGLL", "LEMD"), "y": proxy - delta, "proxy": proxy,
                        "delta": delta, "treatment": delta + rng.normal(0, 30, n)})
    feat = pd.DataFrame({"row": np.arange(n, dtype="int64"), "MVT_ID_mvt": ids, "coverage": rng.integers(0, 4, n),
                         "f1": rng.normal(size=n)})
    raw = pd.DataFrame({"MVT_ID_mvt": ids, "MVT_TIME_UTC_mvt": t})
    for name, fr in (("rec.parquet", rec), ("feat.parquet", feat), ("raw.parquet", raw)):
        fr.to_parquet(tmp_path / name, index=False)
    return rec


def test_load_frame_equals_adsb_stack_load_frame_on_the_same_files(tmp_path, monkeypatch):
    """Drift guard: the parameterised copy must build exactly adsb_stack's frame (the only difference is the F guard, which
    here is the caller's). Rehearsed RED: day folds from `month` instead of `day`; dropping the hour column."""
    rec = _world(tmp_path)
    monkeypatch.setattr(AS, "RECORD", tmp_path / "rec.parquet")
    monkeypatch.setattr(AS, "FEATURES", tmp_path / "feat.parquet")
    monkeypatch.setattr(AS, "RAW_MONTHS", [tmp_path / "raw.parquet"])
    guard = float(np.sqrt(np.mean((np.maximum(rec.proxy - rec.treatment, 1) - rec.y) ** 2)))
    monkeypatch.setattr(AS, "F_GUARD_RMSE", guard)
    want = AS.load_frame()
    got = A2.load_frame(tmp_path / "rec.parquet", tmp_path / "feat.parquet", [tmp_path / "raw.parquet"])
    pd.testing.assert_frame_equal(got, want)
    assert A2.f_guard(got) == pytest.approx(guard, abs=1e-12)


def test_load_frame_refuses_misaligned_rows(tmp_path):
    _world(tmp_path)
    f = pd.read_parquet(tmp_path / "feat.parquet")
    f.loc[0, "row"] = 99
    f.to_parquet(tmp_path / "feat.parquet", index=False)
    with pytest.raises(ValueError, match="not the record's rows"):
        A2.load_frame(tmp_path / "rec.parquet", tmp_path / "feat.parquet", [tmp_path / "raw.parquet"])


# ---------------------------------------------------------------- the gate re-check

def _v4_gate():
    return A2.v4_contract()["gate"]


@pytest.mark.devtree
def test_recheck_gate_reproduces_the_shipped_gate_from_its_coverage_and_a_passing_transfer():
    g = _v4_gate()
    c25 = {a: v["c25"] for a, v in g.items()}
    c26 = {a: v["c26"] for a, v in g.items()}
    got, same = A2.recheck_gate(c25, c26, True, g)
    assert same and got == g


@pytest.mark.devtree
def test_recheck_gate_flags_a_failed_transfer_and_changed_coverage():
    """A failed ADN.7 transfer on F-NSC shuts the newly covered airports (EGLL, LEMD, LFPG, LTFM): the gate differs -> FAIL.
    Rehearsed RED: comparing only the `allowed` flags of the 0.7x airports."""
    g = _v4_gate()
    c25 = {a: v["c25"] for a, v in g.items()}
    c26 = {a: v["c26"] for a, v in g.items()}
    got, same = A2.recheck_gate(c25, c26, False, g)
    assert not same and not got["EGLL"]["allowed"] and got["EHAM"]["allowed"]
    c26b = dict(c26, EDDF=0.1)
    got, same = A2.recheck_gate(c25, c26b, True, g)
    assert not same and not got["EDDF"]["allowed"]


# ---------------------------------------------------------------- orchestration with fakes (BC-24)

class _FakeStack:
    """adsb_stack's surface the runner uses, faked: train_artifacts writes a manifest; fit_stacker / predict_residual give
    a residual that helps (y - F scaled) so the OOF gain is positive."""
    N_FOLDS = 5
    TRANSFER_AIRPORTS = ("EHAM", "EDDM", "LSZH", "LEBL")
    transfer_rule = staticmethod(AS.transfer_rule)
    score_arm = staticmethod(AS.score_arm)

    def __init__(self, helps=True, oof_helps=None):
        self.helps, self.trained = helps, None
        self.oof_helps = helps if oof_helps is None else oof_helps

    def train_artifacts(self, fr, feats, out_dir, seeds, gate=None, ship_learner="B"):
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.trained = dict(feats=list(feats), seeds=tuple(seeds), gate=gate, ship_learner=ship_learner, n=len(fr))
        man = {"feats": list(feats), "seeds": list(seeds), "folds": 5, "gate": dict(gate or {}), "ship_learner": ship_learner}
        (out_dir / "MANIFEST.json").write_text(json.dumps(man))
        (out_dir / "G_s0_k0.txt").write_text("booster")
        return man

    def _resid(self, fr):
        r = fr.delta.to_numpy(float) - fr.F.to_numpy(float)
        return (0.95 if self.helps else -0.5) * r * (fr.coverage.to_numpy() >= 2)

    def fit_stacker(self, fr, feats, arm, seed, train_rows=None, predict_rows=None, lgb_params=None):
        out = fr.F.to_numpy(float).copy()
        if feats:
            out = out + self._resid(fr) * (1 if predict_rows is None else np.asarray(predict_rows, bool))
        return out

    def predict_residual(self, frame, model_dir, oof=False):
        man = json.loads((pathlib.Path(model_dir) / "MANIFEST.json").read_text())
        assert man["gate"], "OOF scoring before the gate is written"
        r = frame.delta.to_numpy(float) - frame.F.to_numpy(float)
        return (0.95 if self.oof_helps else -0.5) * r * (frame.coverage.to_numpy() >= 2)


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    days = [f"2025-01-{d:02d}" for d in range(1, 21)] + [f"2025-07-{d:02d}" for d in range(1, 21)]
    day = np.array([days[i % len(days)] for i in range(n)])
    proxy = rng.uniform(400, 900, n)
    delta = rng.normal(0, 80, n)
    fr = pd.DataFrame({"row": np.arange(n), "month": np.where(np.char.startswith(day, "2025-01"), 1, 7),
                       "ap": np.array(["EHAM", "EDDM", "LSZH", "LEBL", "EGLL"])[np.arange(n) % 5],
                       "y": proxy - delta, "proxy": proxy, "delta": delta, "F": delta + rng.normal(0, 90, n),
                       "coverage": np.where(np.arange(n) % 4 == 0, 1, 3), "day": day, "hour": np.arange(n) % 24})
    fr["fold"] = AS.day_folds(fr.day)
    return fr


def _tables():
    g = _v4_gate()
    return ({a: v["c25"] for a, v in g.items()}, {a: v["c26"] for a, v in g.items()})


@pytest.mark.devtree
def test_run_trains_with_the_v4_contract_writes_the_gate_and_passes(tmp_path):
    """Every line after the compute (transfer, gate, OOF score, report) on real-shaped fakes (BC-24).
    Rehearsed RED: training with fx.FEATURES instead of the manifest's feats; writing the gate before the transfer."""
    st = _FakeStack()
    fr = _frame()
    c25, c26 = _tables()
    rep = A2.run(False, tmp_path, fr, c25, c26, stack=st, step1_rmse=A2.f_guard(fr), allow_roots=(tmp_path,))
    v4 = A2.v4_contract()
    assert st.trained["feats"] == v4["feats"] and st.trained["seeds"] == (0, 1, 2) and st.trained["gate"] == {}
    man = json.loads((tmp_path / A2.MODELS_NAME / "MANIFEST.json").read_text())
    assert man["gate"] == v4["gate"] and man["gate_evidence"]["transfer_PASS"] is True
    assert man["gate_evidence"]["source"].startswith("v34 step 2")
    assert rep["status"] == "PASS" and rep["gate_equals_v4"] and rep["oof"]["clauses"]["C1"]
    assert rep["transfer"]["rule"]["PASS"] and rep["n_features"] == 33
    j = json.loads((tmp_path / "report.json").read_text())
    assert j == rep and A2.C.rel(tmp_path / A2.MODELS_NAME / "MANIFEST.json") in j["outputs"]


@pytest.mark.devtree
def test_run_fails_when_the_oof_stack_does_not_help_or_the_guard_differs(tmp_path):
    """Rehearsed RED (2026-09-24, escaped first): dropping C1 from the PASS rule -- the first fixture failed the transfer
    too, so the FAIL was not C1's; the OOF-only failure below isolates it."""
    fr = _frame()
    c25, c26 = _tables()
    rep = A2.run(False, tmp_path / "a", fr, c25, c26, stack=_FakeStack(helps=True, oof_helps=False),
                 step1_rmse=A2.f_guard(fr), allow_roots=(tmp_path,))
    assert rep["transfer"]["rule"]["PASS"] and rep["gate_equals_v4"]          # only the OOF clause fails
    assert rep["status"] == "FAIL" and not rep["oof"]["clauses"]["C1"]
    with pytest.raises(AssertionError, match="F guard"):
        A2.run(False, tmp_path / "b", fr, c25, c26, stack=_FakeStack(), step1_rmse=A2.f_guard(fr) + 0.01,
               allow_roots=(tmp_path,))


@pytest.mark.devtree
def test_run_refuses_to_overwrite_models(tmp_path):
    (tmp_path / A2.MODELS_NAME).mkdir(parents=True)
    fr = _frame()
    c25, c26 = _tables()
    with pytest.raises(FileExistsError):
        A2.run(False, tmp_path, fr, c25, c26, stack=_FakeStack(), step1_rmse=A2.f_guard(fr), allow_roots=(tmp_path,))


@pytest.mark.devtree
def test_smoke_trains_seed_0_on_the_smoke_subset_and_reports_smoke(tmp_path):
    st = _FakeStack()
    fr = _frame()
    c25, c26 = _tables()
    rep = A2.run(True, tmp_path, fr, c25, c26, stack=st, step1_rmse=None, allow_roots=(tmp_path,))
    assert st.trained["seeds"] == (0,) and rep["status"] == "SMOKE" and rep["smoke"] is True


def test_parse_args_requires_the_rss_limit():
    with pytest.raises(SystemExit):
        A2.parse_args([])
    assert A2.parse_args(["--rss-limit-gib", "6"]).rss_limit_gib == 6.0


@pytest.mark.devtree
def test_transfer_is_adn7_leave_one_airport_out_and_never_writes_the_shipped_report():
    """Rehearsed RED: training on the held-out airport too (train_rows = all rows)."""
    seen = []

    class _S(_FakeStack):
        def fit_stacker(self, fr, feats, arm, seed, train_rows=None, predict_rows=None, lgb_params=None):
            seen.append((arm, seed, bool(feats), int(np.asarray(train_rows).sum()), int(np.asarray(predict_rows).sum())))
            return super().fit_stacker(fr, feats, arm, seed, train_rows, predict_rows)
    before = (ROOT / "reports/adsb_transfer.json").read_bytes()
    fr = _frame()
    out = A2.transfer(fr, ["f1"], _S())
    assert set(out["airports"]) == {"EHAM", "EDDM", "LSZH", "LEBL"} and out["rule"]["PASS"]
    for arm, seed, _, n_tr, n_pr in seen:
        assert arm == "G" and seed == 0 and n_tr + n_pr == len(fr)
    assert (ROOT / "reports/adsb_transfer.json").read_bytes() == before
