"""v34 step 6 — CatBoost C-NSC with the CBV ship recipe (plans/PREREG_nsc_no_seconds_2026_09_24.md §7 step 6, AMENDMENT V34.6).

    /opt/homebrew/bin/python3.11 -B scripts/analysis/v34_launch.py --step 6 --detach [--smoke]
    (the launcher runs: v34_catboost.py --rss-limit-gib 6 [--smoke])

The shipped C (data/cache_stand/cbv_ship_C_2026.npy, sha 1d192deb...) came from scripts/catboost_native_worker.py (the isolated
~/.venvs/prc-catboost interpreter, BC-5) on data/cache_stand/cbv_ship_input.parquet (sha a210da02...; all 2,062,440 2025 rows
train, early stop on months 3/9; the 339,551 2026 matched rows are the holdout) with the worker defaults = the recorded
params (depth 8, lr 0.15, <= 12,000 iterations, od_wait 200, seed 0, 6 threads, ctr 2): 59 numeric + 13 categoricals.

v34: the SAME input with aobt_sec / mvt_sec / sched_sec physically removed (streamed with pyarrow, every other column verified
equal to the source column by column), its cols.json = the numeric list minus the three (= nsc_fold.nsc_catboost_numeric(),
56) + the same 13 categoricals, and the SAME worker and params. Outputs (new paths only, data/nsc_v34/step6_catboost/):
cbv_nsc_input.parquet (+ .cols.json), cbv_nsc_C_2026.npy (+ .json, the worker's record), report.json.

The worker is a subprocess of a different interpreter: this runner's in-process RSS guard cannot see it, so guard_worker polls
the worker's RSS and kills it above the step limit (exit 70, the in-process guard's code); the launcher's watchdog sums the
whole tree as the second line. MEASURED RISK: CBV's worker peaked at 7.03 GiB (cbv_ship_C_2026.json); 3 fewer float32
columns do not change that materially, so at the default 6 GiB limit this step is EXPECTED TO ABORT - an owner decision
(the 8 / 9 GiB exception given to steps 3-4, or a memory-lean path) is needed before the real launch.
This process imports neither CatBoost nor LightGBM.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (ROOT / "scripts", ROOT / "scripts" / "analysis"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, rel: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("v34_common", "scripts/analysis/v34_common.py")
CN = _load("catboost_native", "scripts/catboost_native.py")         # VENV_PY / WORKER paths only (no model import)

STEP = 6
SRC = ROOT / "data" / "cache_stand" / "cbv_ship_input.parquet"
SRC_SHA = "a210da02a6bbe4c0f91ae81d1b3512148f6d9f6c4b426dc18914e7a22b482916"
SHIPPED_C = ROOT / "data" / "cache_stand" / "cbv_ship_C_2026.npy"
SHIPPED_C_SHA = "1d192deb581738b0ba5fdd081ae840def19e28a47f0a4570f30aed839dc92035"
PARAMS = dict(depth=8, learning_rate=0.15, iterations=12_000, od_wait=200, seed=0, threads=6, ctr=2)
SMOKE_PARAMS = dict(PARAMS, iterations=60, od_wait=20, threads=4)
SMOKE_EVERY, SMOKE_HOLDOUT = 100, 2_000
INPUT_NAME, PRED_NAME = "cbv_nsc_input.parquet", "cbv_nsc_C_2026.npy"
POLL_S = 5.0
N_HOLDOUT = 339_551


def nsc_cols(cols: dict) -> dict:
    """The worker's column lists without the three seconds columns (refuses lists that lack them: drop_seconds)."""
    num = [c for c in cols["num"] if c not in C.SECONDS_COLS]
    if len(num) != len(cols["num"]) - 3:
        raise ValueError("the numeric list does not hold all three seconds columns")
    return {"num": C.assert_no_seconds(num, "C-NSC numeric"), "cat": C.assert_no_seconds(cols["cat"], "C-NSC categoricals")}


def build_input(src, dst, batch_rows: int = 200_000, every=None, n_holdout=None,
                allow_roots=(C.OUT_ROOT, C.SMOKE_ROOT)) -> dict:
    """Stream `src` into `dst` without the seconds columns (pandas metadata dropped: a plain RangeIndex frame), write
    `dst`.cols.json, verify. `every` / `n_holdout` (smoke): keep training rows whose row number is a multiple of `every` and
    the first `n_holdout` holdout rows; otherwise every row, and every kept column is re-read and compared to the source."""
    src, dst = pathlib.Path(src), C.check_output_path(dst, allow_roots=allow_roots, allow_files=())
    cols_dst = C.check_output_path(pathlib.Path(str(dst) + ".cols.json"), allow_roots=allow_roots, allow_files=())
    cols = nsc_cols(json.loads(pathlib.Path(str(src) + ".cols.json").read_text()))
    pf = pq.ParquetFile(src)
    keep = [c for c in pf.schema_arrow.names if c not in C.SECONDS_COLS]
    schema = pa.schema([pf.schema_arrow.field(c) for c in keep])
    dst.parent.mkdir(parents=True, exist_ok=True)
    n, offset, seen_ho = 0, 0, 0
    with pq.ParquetWriter(dst, schema) as w:
        for b in pf.iter_batches(batch_size=batch_rows, columns=keep):
            t = pa.Table.from_batches([b], schema=schema)
            if every is not None:
                idx = np.arange(offset, offset + t.num_rows)
                tr = np.asarray(t.column("train").to_numpy(zero_copy_only=False), bool)
                ho = np.asarray(t.column("holdout").to_numpy(zero_copy_only=False), bool)
                take_ho = ho & (np.cumsum(ho) + seen_ho <= n_holdout)
                seen_ho += int(ho.sum())
                t = t.filter(pa.array((tr & (idx % every == 0)) | take_ho))
            offset += b.num_rows
            n += t.num_rows
            w.write_table(t)
    pathlib.Path(cols_dst).write_text(json.dumps(cols))
    got = pq.ParquetFile(dst).schema_arrow.names
    if any(c in got for c in C.SECONDS_COLS):
        raise AssertionError("a seconds column survived into the C-NSC input")
    verified = 0
    if every is None:
        for c in keep:
            a = pq.read_table(src, columns=[c]).column(0)
            b2 = pq.read_table(dst, columns=[c]).column(0)
            if not a.equals(b2):
                raise AssertionError(f"column {c} of the C-NSC input differs from the source")
            verified += 1
    return {"n_rows": int(n), "columns": keep, "verified_columns": verified, "cols": cols}


def worker_cmd(inp, out, params: dict) -> list:
    cmd = [str(CN.VENV_PY), "-u", str(CN.WORKER), "--input", str(inp), "--target", "delta", "--output", str(out)]
    for k, v in params.items():
        cmd += ["--" + k.replace("_", "-"), str(v)]
    return cmd


def guard_worker(proc, rss_of, limit_gib: float, sleep=time.sleep, poll_s: float = POLL_S, log=print) -> tuple:
    """Poll the worker until it exits; above limit_gib kill it and return 70 (the in-process guard's exit code).
    Returns (returncode, peak GiB)."""
    peak = 0.0
    while proc.poll() is None:
        r = rss_of(proc.pid)
        peak = max(peak, r)
        if r > limit_gib:
            proc.kill()
            proc.wait()
            log(f"ABORT: CatBoost worker RSS {r:.2f} GiB above the {limit_gib:g} GiB limit")
            return 70, peak
        sleep(poll_s)
    return proc.wait(), peak


def ids_sha(ids) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(ids, dtype="float64")).tobytes()).hexdigest()


def run(smoke: bool, out_dir, src, shipped_c=None, worker=None, params=None, log=print,
        allow_roots=(C.OUT_ROOT, C.SMOKE_ROOT), n_expected=N_HOLDOUT) -> dict:
    """Build the NSC input, run the worker (`worker(cmd, log) -> (rc, peak_gib)`), check its output, write the report."""
    t0 = time.time()
    out_dir = pathlib.Path(out_dir)
    inp = out_dir / INPUT_NAME
    out = C.check_output_path(out_dir / PRED_NAME, allow_roots=allow_roots, allow_files=())
    out_json = C.check_output_path(out.with_suffix(".json"), allow_roots=allow_roots, allow_files=())
    rep_path = C.check_output_path(out_dir / C.REPORT_NAME, allow_roots=allow_roots, allow_files=())
    params = dict(SMOKE_PARAMS if smoke else PARAMS) if params is None else params
    info = build_input(src, inp, every=SMOKE_EVERY if smoke else None, n_holdout=SMOKE_HOLDOUT if smoke else None,
                       allow_roots=allow_roots)
    log(f"C-NSC input {inp}: {info['n_rows']:,} rows, {len(info['cols']['num'])} numeric + {len(info['cols']['cat'])} "
        f"categorical; {info['verified_columns']} columns verified against the source")
    cmd = worker_cmd(inp, out, params)
    rc, peak = worker(cmd, log)
    if rc != 0:
        raise RuntimeError(f"CatBoost worker exited {rc} (peak RSS {peak:.2f} GiB)")
    pred = np.load(out)
    h = pd.read_parquet(inp, columns=["MVT_ID_mvt", "score_proxy", "holdout"])
    h = h[h.holdout.to_numpy(bool)].reset_index(drop=True)
    if len(pred) != len(h):
        raise AssertionError(f"{len(pred)} predictions for {len(h)} holdout rows")
    if not np.isfinite(pred).all():
        raise AssertionError("non-finite C-NSC prediction")
    move = None
    if shipped_c is not None:
        sc = np.asarray(shipped_c, float)
        if len(sc) != len(pred):
            raise AssertionError("the shipped C does not align with the C-NSC holdout rows")
        m = C.taxi(h.score_proxy, pred) - C.taxi(h.score_proxy, sc)
        move = {"rms_taxi_move_s": float(np.sqrt(np.mean(m ** 2))), "mean_taxi_move_s": float(m.mean()),
                "p99_abs_taxi_move_s": float(np.percentile(np.abs(m), 99)), "note": "report only (no fold anchor for C)"}
    wrec = json.loads(out_json.read_text()) if out_json.exists() else {}
    if not smoke and len(pred) != n_expected:
        raise AssertionError(f"C-NSC has {len(pred)} holdout predictions, the matched lane {n_expected}")
    outputs = {C.rel(p): C.sha256(p) for p in (inp, pathlib.Path(str(inp) + ".cols.json"), out, out_json) if p.exists()}
    rep = C.jsonable({"step": STEP, "tag": "V34.6 CatBoost C-NSC", "smoke": bool(smoke),
                      "status": "SMOKE" if smoke else "PASS", "prereg": C.rel(C.PREREG), "source_input": C.rel(src),
                      "numeric": info["cols"]["num"], "categoricals": info["cols"]["cat"],
                      "n_numeric": len(info["cols"]["num"]), "dropped": list(C.SECONDS_COLS), "params": params,
                      "n_input_rows": info["n_rows"], "n_holdout": int(len(pred)),
                      "holdout_ids_sha256": ids_sha(h.MVT_ID_mvt.to_numpy()), "worker_record": wrec,
                      "worker_peak_rss_gib_polled": peak, "move_vs_shipped_c": move, "outputs": outputs,
                      "wall_s": round(time.time() - t0, 1), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat()})
    C.write_json_new(rep, rep_path, allow_roots=allow_roots, allow_files=())
    log(f"C-NSC: {len(pred):,} predictions; move vs shipped C {move}; status {rep['status']}")
    return rep


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rss-limit-gib", type=float, required=True, help="RSS abort for this process AND the worker")
    ap.add_argument("--smoke", action="store_true", help="every 100th training row + 2,000 holdout rows, 60 iterations")
    ap.add_argument("--out-dir", default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:  # pragma: no cover - real-data entry point; exercised by the --smoke run
    import run_guards as G
    args = parse_args(argv)
    stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else (
        C.SMOKE_ROOT / f"smoke_step6_{stamp}" if args.smoke else C.step_dir(STEP))
    log = lambda m: print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)
    for p, want in ((SRC, SRC_SHA), (SHIPPED_C, SHIPPED_C_SHA)):
        if G.sha256_file(p) != want:
            raise RuntimeError(f"{p} is not the shipped file")
    fh = None
    if not args.smoke:
        sha = C.require_registration(STEP)
        log(f"registration sha256 {sha} logged; AMENDMENT V34.6 present")
        fh = G.acquire_lock()
        log("heavy-compute lock acquired (nonblocking)")
    G.start_rss_guard(int(args.rss_limit_gib * 1024 ** 3))

    def worker(cmd, log):
        log(f"worker: {' '.join(cmd)}")
        with open(out_dir / "catboost_worker.log", "a") as wl:
            p = subprocess.Popen(cmd, cwd=ROOT, stdout=wl, stderr=subprocess.STDOUT)
            rss = lambda pid: parse_rss_gib(subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True,
                                                         text=True).stdout)
            return guard_worker(p, rss, args.rss_limit_gib, log=log)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        shipped = None if args.smoke else np.load(SHIPPED_C)
        rep = run(args.smoke, out_dir, SRC, shipped_c=shipped, worker=worker, log=log)
        log(f"report -> {out_dir / C.REPORT_NAME}   wall {rep['wall_s']:.0f}s   worker peak "
            f"{rep['worker_peak_rss_gib_polled']:.2f} GiB (polled); runner peak {G.peak_rss() / 1024 ** 3:.2f} GiB")
    finally:
        if fh is not None:
            G.release_lock(fh)
    return 0


def parse_rss_gib(ps_out: str) -> float:
    """`ps -o rss= -p PID` prints KiB; empty output (process gone) is 0."""
    s = ps_out.strip()
    return int(s) / 1024 ** 2 if s else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
