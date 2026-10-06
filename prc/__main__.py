"""`python -m prc <command> --config configs/fm1.yaml [--set path=value ...]` — FM1's CLI, thin wrappers over prc.api.

    fit               fit and save the versioned artefact (run.artifacts_dir/<config hash>-<data digest>/)
    predict           predict a raw scored file with the config's artefact (or --artifact DIR) -> --out FILE.parquet
    validate          --protocol S | swap | forward | lomo | forward_monthly | dry-run  [--out DIR] [--save-models]
                      (S = LOMO-Jan + LOMO-Jul, primary; swap = strict Jan<->Jul, secondary; forward = F3/F4)
                      --save-models: also save each fold's fit as an artefact under --out/models/<fold name>/ (needs --out)
    build-submission  --out DIR [--version N]: <team>_v<N>.parquet + provenance + manifest
    compare           --a OOF [OOF ...] --b OOF [OOF ...] [--out FILE.json]: paired gain of b over a (pooled),
                      day-block bootstrap CI
    blend             --a OOF.. --b OOF.. --w W --out FILE.parquet: selection arm A2 offline
    config-reference  --write | --check: docs/fm1/CONFIG_REFERENCE.md from the schema

There is no `--base`: FM1 never splices a previous submission. A fit or validation over more than the heavy-row threshold
needs --allow-heavy (the pre-run checks first; one heavy job at a time). Exit 0 = done, 1 = refused (a named FM1 error), 2 = usage.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m prc", description=__doc__, allow_abbrev=False,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, config=True):
        if config:
            p.add_argument("--config", required=True, help="the FM1 YAML (configs/fm1.yaml)")
            p.add_argument("--set", dest="overrides", action="append", default=[], metavar="PATH=VALUE",
                           help="override one knob (validated; recorded in the config hash); repeatable")
        return p
    f = common(sub.add_parser("fit", allow_abbrev=False))
    f.add_argument("--allow-heavy", action="store_true")
    f.add_argument("--reuse-from", default=None,
                   help="a donor artefact directory: reuse its fitted matched learners (only allow-listed knobs may differ)")
    p = common(sub.add_parser("predict", allow_abbrev=False))
    p.add_argument("--artifact", default=None)
    p.add_argument("--rows", required=True, help="a raw scored parquet file (the organiser's 30 columns)")
    p.add_argument("--out", required=True, help="output parquet (never overwritten)")
    v = common(sub.add_parser("validate", allow_abbrev=False))
    v.add_argument("--protocol", required=True, choices=["S", "swap", "forward", "lomo", "forward_monthly", "dry-run"])
    v.add_argument("--out", default=None)
    v.add_argument("--allow-heavy", action="store_true")
    v.add_argument("--save-models", dest="save_models", action="store_true",
                   help="also save each fold's fit as an artefact under --out/models/<fold name>/ (needs --out)")
    b = common(sub.add_parser("build-submission", allow_abbrev=False))
    b.add_argument("--artifact", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--version", type=int, default=None)
    c = sub.add_parser("compare", allow_abbrev=False)
    c.add_argument("--a", required=True, nargs="+", help="one or more OOF records (pooled)")
    c.add_argument("--b", required=True, nargs="+", help="one or more OOF records (pooled)")
    c.add_argument("--out", default=None)
    c.add_argument("--n-boot", type=int, default=2000)
    c.add_argument("--seed", type=int, default=20260928)
    bl = sub.add_parser("blend", allow_abbrev=False)
    bl.add_argument("--a", required=True, nargs="+")
    bl.add_argument("--b", required=True, nargs="+")
    bl.add_argument("--w", required=True, type=float, help="weight of b")
    bl.add_argument("--out", required=True)
    r = sub.add_parser("config-reference", allow_abbrev=False)
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    return ap.parse_args(argv)


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, default=str)


def main(argv=None) -> int:
    args = parse_args(argv)
    from prc import api
    from prc.pipeline.fm1 import refdoc
    try:
        if args.cmd == "config-reference":
            if args.write:
                print(refdoc.write_reference())
                return 0
            ok = refdoc.check_reference()
            print("CONFIG_REFERENCE.md is current" if ok else "CONFIG_REFERENCE.md is STALE: run --write")
            return 0 if ok else 1
        if args.cmd == "compare":
            rep = api.compare(args.a, args.b, n_boot=args.n_boot, seed=args.seed)
            text = _dump(rep)
            if args.out:
                out = pathlib.Path(args.out)
                if out.exists():
                    raise api.errors.ArtifactError(f"{out} exists; never overwritten")
                out.write_text(text)
            print(text)
            return 0
        if args.cmd == "blend":
            out = pathlib.Path(args.out)
            if out.exists():
                raise api.errors.ArtifactError(f"{out} exists; never overwritten")
            api.blend_oof(args.a, args.b, args.w).to_parquet(out, index=False)
            print(out)
            return 0
        if args.cmd == "fit":
            art = api.fit(args.config, allow_heavy=args.allow_heavy, overrides=args.overrides, reuse_from=args.reuse_from)
            print(art.directory)
            return 0
        if args.cmd == "validate":
            rep = api.validate(args.config, args.protocol, out_dir=args.out, allow_heavy=args.allow_heavy,
                               overrides=args.overrides, save_models=args.save_models)
            print(_dump(rep if args.protocol == "dry-run" else {k: v for k, v in rep.items() if k != "folds"}))
            if args.protocol == "dry-run" and api.dry_run_refused(rep):
                return 1
            return 0
        target = args.artifact or args.config
        ov = args.overrides                          # refused by name with --artifact (api._artifact)
        if args.cmd == "predict":
            import pandas as pd
            out = pathlib.Path(args.out)
            if out.exists():
                raise api.errors.ArtifactError(f"{out} exists; never overwritten")
            pred = api.predict(target, pd.read_parquet(args.rows), overrides=ov)
            out.parent.mkdir(parents=True, exist_ok=True)
            pred.to_parquet(out, index=False)
            print(out)
            return 0
        res = api.build_submission(target, args.out, version=args.version, overrides=ov)
        print(res["submission"])
        return 0
    except api.errors.PipelineError as exc:
        print(f"fm1 refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
