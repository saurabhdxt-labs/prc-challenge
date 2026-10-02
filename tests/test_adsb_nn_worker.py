"""scripts/adsb_nn_worker.py — the ADS-B stack's MLP half, run as its own process (BC-5: torch never shares a process with
LightGBM). Driven here through its command line in a child process, so this test process never imports torch.

Mutation rehearsal per test in its docstring.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "adsb_nn_worker.py"


def _inputs(path: pathlib.Path, B: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(64, 3)).astype(np.float32)
    np.savez(path, A=A, ya=rng.normal(size=64), B=B, yb=rng.normal(size=len(B)), T=A[:5], seed=0, hidden=4,
             dropout=0.0, lr=1e-2, batch=16, epochs=3, patience=2, grad_spike=1e9)


def _run(tmp_path: pathlib.Path, B: np.ndarray):
    inp, out = tmp_path / "in.npz", tmp_path / "out.npz"
    _inputs(inp, B)
    r = subprocess.run([sys.executable, "-B", str(WORKER), str(inp), str(out)], capture_output=True, text=True,
                       timeout=300)
    return r, (dict(np.load(out)) if out.exists() else None)


def test_a_non_finite_validation_loss_is_a_guard_trip_not_a_crash(tmp_path):
    """Validation rows that make every epoch's loss NaN leave no best state: the worker must report it through its
    guard contract (exit 3, `error` in the out file) instead of crashing on load_state_dict(None). Mutation: drop the
    finite check on the validation loss -> TypeError, exit 1, no out file -> red."""
    B = np.full((16, 3), np.nan, dtype=np.float32)
    r, out = _run(tmp_path, B)
    assert r.returncode == 3, r.stderr[-1500:]
    assert "non-finite validation loss" in str(out["error"])


def test_finite_inputs_train_and_predict_every_requested_row(tmp_path):
    """The normal path: exit 0, one finite prediction per T row and the epoch / step meta. Mutation: predict on B
    instead of T -> wrong length -> red."""
    rng = np.random.default_rng(1)
    r, out = _run(tmp_path, rng.normal(size=(16, 3)).astype(np.float32))
    assert r.returncode == 0, r.stderr[-1500:]
    assert out["pred"].shape == (5,) and np.isfinite(out["pred"]).all()
    assert int(out["epochs"]) >= 1 and int(out["steps"]) >= 1
