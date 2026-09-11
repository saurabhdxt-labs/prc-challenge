"""`python -m prc.pipeline run --config configs/pipeline.yaml --out <dir> [--version N] [--dry-run]`

    run --dry-run   validate the config, the pins, every input's schema, the scored file's routing and the
                    lanes' inputs on the real files; fit and predict NOTHING; write <out>/dry_run.json
    run             the whole pipeline: ingest -> route -> lanes -> assemble -> validate -> write
                    <out>/submission.parquet (or <team>_vN.parquet with --version) + provenance + manifest

There is no `--base`: the pipeline builds every row from data (argparse rejects the flag; allow_abbrev is
off so no prefix of another flag can stand in for it). A matched lane with `source: refit` is the ~2 h,
~5 GB full fit and needs `--allow-heavy` (one heavy job at a time, on AC power -- the operator's check).
"""
from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import numpy as np

from . import assemble as A
from . import config as C
from . import errors as E
from . import ingest, lanes, validate, write


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m prc.pipeline", description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", allow_abbrev=False, help="run (or --dry-run) the pipeline")
    r.add_argument("--config", required=True, help="the pipeline YAML (configs/pipeline.yaml)")
    r.add_argument("--out", required=True, help="output directory (created; files in it are never overwritten)")
    r.add_argument("--version", type=int, default=None,
                   help="write <team>_vN.parquet instead of submission.parquet (the upload name)")
    r.add_argument("--dry-run", action="store_true", help="validate everything on the real files; fit nothing")
    r.add_argument("--allow-heavy", action="store_true", help="permit a matched-lane refit (source: refit)")
    return ap.parse_args(argv)


def input_paths(cfg, routing) -> dict:
    """{name: path} of every file the routed lanes read (hashed into the manifest)."""
    used = set(routing.lane.tolist())
    kinds = {cfg.lanes[l].kind for l in used}
    out = {"scored_file": cfg.paths.scored_file, "template_file": cfg.paths.template_file}
    if kinds & {"unmatched", "airport_rules"}:
        for n in ("unm.parquet", "lirf.parquet"):
            out[f"rome_cache/{n}"] = cfg.paths.rome_cache / n
        if any(cfg.lanes[l].model == "nonfill_body_congestion" for l in used):
            for f in cfg.paths.training_files():
                out[f"training/{f.name}"] = f
    for lid in sorted(used):
        lane = cfg.lanes[lid]
        if lane.kind != "matched":
            continue
        months = lanes.stand_training_paths(cfg.paths.stand_cache)
        for p in [cfg.paths.stand_cache / "ranking.parquet", *months]:
            out[f"stand_cache/{p.name}"] = p
            for block, bdir in lane.feature_blocks:
                out[f"{block}_cache/{p.name}"] = pathlib.Path(bdir) / p.name
        if lane.source == "saved_boosters":
            for b, j in lanes.booster_pairs(lane):
                out[f"boosters/{b.name}"] = b
                out[f"boosters/{j.name}"] = j
    return out


def dry_run(cfg, out_dir: pathlib.Path, log=_log) -> dict:
    t0 = time.time()
    pins = lanes.check_pins(cfg)
    scored = ingest.ingest_scored(cfg)
    routing = lanes.route(scored.scored, cfg)
    paths = input_paths(cfg, routing)
    checked = validate.assert_no_splice(paths.values())
    lane_inputs = lanes.check_lane_inputs(cfg, routing)
    fb = {str(a): int(n) for a, n in zip(*np.unique(routing.airport[routing.fallback], return_counts=True))}
    report = {"dry_run": True, "config": {"file": str(cfg.source_path), "hash": cfg.config_hash},
              "pins": {k: (list(v) if isinstance(v, tuple) else v) for k, v in pins.items()},
              "scored": {"file": str(cfg.paths.scored_file), "n_departures": int(len(scored.dep)),
                         "n_scored": int(len(scored.scored)), "n_template": int(len(scored.template))},
              "routing": {"counts": routing.counts(), "by_airport": routing.by_airport()},
              "fallback_airports": list(cfg.fallback_airports()), "fallback_rows": fb,
              "lane_inputs": lane_inputs,
              "inputs": write.describe_inputs(paths, cfg.paths.root),
              "no_splice": {"asserted": True, "n_inputs_checked": len(checked)},
              "wall_s": round(time.time() - t0, 1)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dry_run.json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    log(f"dry run OK: {report['scored']['n_scored']:,} scored rows -> {report['routing']['counts']}; "
        f"fallback airports {report['fallback_airports']} ({fb}); {len(paths)} inputs checked; {report['wall_s']} s")
    return report


def run(cfg, out_dir: pathlib.Path, version: int | None = None, allow_heavy: bool = False, log=_log,
        matched_model=None) -> dict:
    timings: dict = {}
    t = time.time()
    lanes.check_pins(cfg)
    heavy = [l.id for l in cfg.lanes.values() if l.kind == "matched" and l.source == "refit"]
    if heavy and not allow_heavy:
        raise E.PipelineError(f"lane(s) {heavy} refit the matched model (~2 h, ~5 GB): pass --allow-heavy after the "
                              "quality gate (AC power, no other job above 2 GB)")
    scored = ingest.ingest_scored(cfg)
    routing = lanes.route(scored.scored, cfg)
    paths = input_paths(cfg, routing)
    checked = validate.assert_no_splice(paths.values())
    timings["ingest_route"] = round(time.time() - t, 1)
    log(f"routed {len(routing.ids):,} scored rows: {routing.counts()}")

    results = []
    unm_rows = scored.scored.unmatched.to_numpy(dtype=bool)
    stratum_lanes = [l for l in cfg.lanes.values() if l.kind in ("unmatched", "airport_rules") and routing.mask(l.id).any()]
    if stratum_lanes:
        t = time.time()
        scored_unm = scored.scored[unm_rows]
        inputs = lanes.stratum_inputs(cfg)
        parts = lanes.s1_parts(inputs, scored_unm, cfg.seeds)
        lane_unm = routing.lane[unm_rows]
        for lane in stratum_lanes:
            rows = lane_unm == lane.id
            if lane.kind == "unmatched":
                res = lanes.run_unmatched_lane(lane, rows, scored_unm, parts, inputs, scored.dep, cfg.seeds)
            else:
                codes = sorted(set(scored_unm.ADEP_mvt.astype(str).to_numpy()[rows]))
                if len(codes) != 1:
                    raise E.RoutingError(f"airport_rules lane {lane.id} serves {codes}; one airport per rules lane")
                res = lanes.run_rules_lane(lane, cfg.airports[codes[0]], rows, scored_unm, parts, inputs, cfg.seeds)
            results.append(res)
            log(f"lane {lane.id}: {len(res.ids):,} rows")
        timings["unmatched_lanes"] = round(time.time() - t, 1)
        del inputs, parts, scored_unm
    dep_n = len(scored.dep)
    template = scored.template
    scored = None
    gc.collect()
    for lane in cfg.lanes.values():
        if lane.kind != "matched" or not routing.mask(lane.id).any():
            continue
        t = time.time()
        res = lanes.run_matched_lane(lane, cfg, routing.ids[routing.mask(lane.id)], model=matched_model, log=log)
        results.append(res)
        timings[f"lane_{lane.id}"] = round(time.time() - t, 1)
        log(f"lane {lane.id}: {len(res.ids):,} rows ({res.info})")
        gc.collect()

    t = time.time()
    assembled = A.assemble(results, routing, template)
    report = validate.validate(assembled, template, routing, cfg)
    outputs = write.write_submission(assembled, template, out_dir, cfg.team, version)
    lane_meta = {r.lane: {"kind": cfg.lanes[r.lane].kind, "model": cfg.lanes[r.lane].model, "n_rows": int(len(r.ids)),
                          **{k: v for k, v in r.info.items() if k != "p"}} for r in results}
    timings["assemble_validate_write"] = round(time.time() - t, 1)
    manifest = write.build_manifest(cfg, write.describe_inputs(paths, cfg.paths.root), outputs, lane_meta, routing,
                                    {**report, "n_departures_in_scored_file": dep_n}, timings, checked, out_dir=out_dir)
    mpath = write.write_manifest(manifest, out_dir, cfg.team, version)
    log(f"wrote {out_dir / outputs['submission']['file']} ({outputs['submission']['rows']:,} rows), manifest {mpath.name} "
        f"digest {manifest['digest'][:12]}")
    return {"assembled": assembled, "manifest": manifest, "results": results}


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        cfg = C.load_config(args.config)
        out = pathlib.Path(args.out)
        if args.dry_run:
            dry_run(cfg, out)
        else:
            run(cfg, out, version=args.version, allow_heavy=args.allow_heavy)
    except (E.PipelineError, NotImplementedError) as exc:
        print(f"pipeline refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0
