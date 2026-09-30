"""AT7 — the package's composition functions reproduce the verified v36 R12 lane on ALL 339,551 matched rows.

Inputs (read-only): the R12 serve components (data/r12/serve_801f9bcac9a5_full/matched_2026_r12_detail.parquet: lane, F,
resid, C, partner_int), the refit's proxy (refit_801f9bcac9a5_full/predictions_2026.parquet) and the served values
(matched_2026_r12.parquet). TRUE shape: every row within 1 s (predicted exact). FALSE: any row beyond.
AT8 (the stack stage itself, and the whole package path) is tools/fm1_replay_gate.py's (tests/fm1/test_fm1_replay_gate.py).
Runs in a child process (memory feedback_prc_heavy_tests_inflate_pytest_peak_rss)."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_composition_reproduces_every_served_r12_value():
    """Mutation: blend weight 0.6 inside blend_value's call below, or drop the rint in staged_value -> rows off -> red."""
    code = f"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, {str(ROOT)!r})
from prc.pipeline.fm1 import components as C, stages as S
base = {str(ROOT / 'data/r12')!r}
det = pd.read_parquet(base + '/serve_801f9bcac9a5_full/matched_2026_r12_detail.parquet')
served = pd.read_parquet(base + '/serve_801f9bcac9a5_full/matched_2026_r12.parquet').set_index('MVT_ID_mvt').taxi
proxy = pd.read_parquet(base + '/refit_801f9bcac9a5_full/predictions_2026.parquet', columns=['MVT_ID_mvt', 'proxy']).set_index('MVT_ID_mvt').proxy
ids = det.MVT_ID_mvt.to_numpy(float)
px = proxy.reindex(ids).to_numpy(float)
n = det.lane.to_numpy() == 'N_v4'
val = np.where(n, C.staged_value(px, det.F.to_numpy(float), det.resid.to_numpy(float)),
               C.blend_value(px, det.C.to_numpy(float), det.partner_int.to_numpy(float), 0.5))
led = S.Ledger(ids)
st = {{s.name: s for s in S.STAGE_ORDER}}
led.write(st['adsb_stack'], ids[n], val[n])
led.write(st['cbv_blend'], ids[~n], val[~n])
out = led.finalise().set_index('MVT_ID_mvt')
d = np.abs(out.TAXITIME_SEC_mvt.reindex(ids).to_numpy(float) - served.reindex(ids).to_numpy(float))
print(json.dumps({{"n": int(len(ids)), "within_1s": int((d <= 1).sum()), "exact": int((d == 0).sum()), "max": float(d.max()),
                   "lanes": det.lane.value_counts().to_dict(), "decided_by": led.counts()}}))
"""
    r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["n"] == 339_551
    assert rep["within_1s"] == rep["n"] and rep["exact"] == rep["n"] and rep["max"] == 0.0
    assert rep["lanes"] == {"N_v4": 224_371, "R_orl": 109_816, "U_fw": 5_364}
    assert rep["decided_by"] == {"adsb_stack": 224_371, "cbv_blend": 115_180}
