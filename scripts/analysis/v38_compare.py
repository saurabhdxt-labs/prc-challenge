"""v38 (DWG, BC-26) — the SCREEN comparison of the R12 stack before and after the gate-dwell fix
(plans/PREREG_v38_dwg_2026_09_27.md, H1).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/v38_compare.py \
        --old data/r12/stack_801f9bcac9a5_screen/oof_r12.parquet --new data/r12/stack_<stamp>_screen/oof_r12.parquet \
        --out data/r12/v38_compare_screen.json

Inputs are r12_stack's OOF frames (one row per PIPE_M row: y, block, w, PIPE_M and the composed variants). For each
variant in VARIANTS, per airport and pooled (the 5 v35 airports, and all): S = move energy vs PIPE_M and G = gain vs
PIPE_M (r12_validate.paired_gain: SSE difference / 344,841 with its day-block bootstrap), for each run; and the PAIRED
difference new vs old on the same rows (positive = the new run is better). H1 reads the 5-airport paired difference.
No model is fitted here.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


R = _load("r12_validate", "scripts/analysis/r12_validate.py")
BOARD_N = R.BOARD_N
AIRPORTS_5 = ("EDDF", "EDDM", "EHAM", "LEBL", "LSZH")      # the v35 / v36 airports
VARIANTS = ("stacked_v34lanes", "single")
H1_MIN_GAIN = 100.0                                        # PREREG v38 H1 (bmse)
ALIGN_COLS = ("row", "y", "block", "PIPE_M")


def move_energy(cand, comp) -> float:
    """S = sum((cand - comp)^2) / 344,841 over rows where comp is finite."""
    cand, comp = np.asarray(cand, float), np.asarray(comp, float)
    m = np.isfinite(comp)
    return float(((cand[m] - comp[m]) ** 2).sum() / BOARD_N)


def airport_mask(oof: pd.DataFrame, airports) -> np.ndarray:
    return np.ones(len(oof), bool) if airports is None else oof.ap.isin(list(airports)).to_numpy()


def pooled(oof: pd.DataFrame, variant: str, airports, n_boot: int = R.N_BOOT) -> dict:
    """S and G (with CI) of `variant` vs PIPE_M on the rows of `airports` (None = all)."""
    d = oof[airport_mask(oof, airports)]
    g = R.paired_gain(d.y, d[variant], d.PIPE_M, d.block.to_numpy(), None, n_boot)
    return {"S": move_energy(d[variant], d.PIPE_M), "G": g["gain_bmse"], "ci95": g["ci95"], "n": g["n"]}


def airport_table(oof: pd.DataFrame, variant: str, n_boot: int = R.N_BOOT) -> dict:
    return {ap: pooled(oof, variant, (ap,), n_boot) for ap in sorted(oof.ap.unique())}


def align(old: pd.DataFrame, new: pd.DataFrame) -> None:
    """Refuse unless both frames hold the same rows in the same order, with the same labels, blocks and comparator."""
    if len(old) != len(new):
        raise ValueError(f"row count differs: old {len(old)}, new {len(new)}")
    for c in ALIGN_COLS:
        a, b = old[c].to_numpy(), new[c].to_numpy()
        same = np.array_equal(a, b) if a.dtype.kind not in "f" else np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True)
        if not same:
            raise ValueError(f"column {c} differs between the old and new OOF frames")
    if not np.array_equal(old.ap.to_numpy(), new.ap.to_numpy()):
        raise ValueError("column ap differs between the old and new OOF frames")


def _paired(old, new, variant, airports, n_boot) -> dict:
    m = airport_mask(old, airports)
    return R.paired_gain(old.y.to_numpy()[m], new[variant].to_numpy()[m], old[variant].to_numpy()[m],
                         old.block.to_numpy()[m], None, n_boot)


def h1_verdict(dg5: float, ci95) -> str:
    """PREREG v38 H1: WORKING iff the 5-airport paired gain >= +100 bmse and its CI lower bound > 0; else NOT WORKING."""
    ok = np.isfinite(dg5) and dg5 >= H1_MIN_GAIN and ci95[0] > 0
    return "WORKING" if ok else "NOT WORKING"


def compare(old: pd.DataFrame, new: pd.DataFrame, n_boot: int = R.N_BOOT) -> dict:
    align(old, new)
    out = {}
    for v in VARIANTS:
        rec = {}
        for name, d in (("old", old), ("new", new)):
            rec[name] = {"airports": airport_table(d, v, n_boot), "five": pooled(d, v, AIRPORTS_5, n_boot),
                         "all": pooled(d, v, None, n_boot)}
        rec["paired_new_vs_old"] = {"five": _paired(old, new, v, AIRPORTS_5, n_boot),
                                    "all": _paired(old, new, v, None, n_boot),
                                    "airports": {ap: _paired(old, new, v, (ap,), n_boot)
                                                 for ap in sorted(old.ap.unique())}}
        out[v] = rec
    p5 = out["stacked_v34lanes"]["paired_new_vs_old"]["five"]
    out["h1"] = {"dG5": p5["gain_bmse"], "ci95": p5["ci95"], "threshold": H1_MIN_GAIN,
                 "verdict": h1_verdict(p5["gain_bmse"], p5["ci95"])}
    out["units"] = "bmse = SSE / 344,841; G > 0 = better than PIPE_M; paired > 0 = new better than old"
    return out


def main(argv=None) -> int:  # pragma: no cover - file I/O around the tested compare()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    r = compare(pd.read_parquet(a.old), pd.read_parquet(a.new))
    r["inputs"] = {"old": a.old, "new": a.new}
    out.write_text(json.dumps(r, indent=1, default=float))
    print(json.dumps(r["h1"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
