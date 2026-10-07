"""Real-data-shaped fixture worlds for the FM1 tests (not a test module).

A world is a small slice of the organiser's REAL 2025 raw files (two airports, a few days per month), with the matching
METAR files, the matching rows of the real R12 ADS-B table, a scored file whose departures have their labels hidden, the
template and the hidden labels. Built in a CHILD process (`python fm1_world.py <out> <spec json>`): extracting rows from the
2 M-row ADS-B table peaks near 0.9 GB, which must not raise the shared pytest process's lifetime RSS.

    E2E        Jan 14-15 + Feb 14-15 train, Jan 16 scored; EDDF + LSZH       (AT9: fit -> predict -> build-submission)
    PROTOCOL   days 14 + 25 of months 1, 2, 3, 6, 7, 9, 10; EDDF + LSZH        (protocol folds, no scored file)

No 2026 label exists anywhere in a world (every row is 2025 and the scored day is a 2025 day with its labels hidden).
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
WEATHER = ROOT / "data" / "weather"
ADSB_TRAIN = ROOT / "data" / "r12" / "tables_801f9bcac9a5" / "train.parquet"
STACK = ROOT / "data" / "r12" / "stack_801f9bcac9a5_full" / "models_r12"
ORL = ROOT / "data" / "osm" / "orl_routes.parquet"
APS = ("EDDF", "LSZH")
E2E = {"months": [1, 2], "days": [14, 15], "aps": list(APS), "score": [1, 16]}
#: day 25 gives the strict swap folds their early-stopping rows (days >= 24 of the training month)
PROTOCOL = {"months": [1, 2, 3, 6, 7, 9, 10], "days": [14, 25], "aps": list(APS), "score": None}


def _month_file(m: int) -> pathlib.Path:
    nxt = "2026-01-01" if m == 12 else f"2025-{m + 1:02d}-01"
    return RAW / f"training_2025-{m:02d}-01_{nxt}.parquet"


def _slice(m: int, days, aps):
    import pandas as pd
    import pyarrow.dataset as ds
    parts = []
    for d in days:
        lo = pd.Timestamp(f"2025-{m:02d}-{d:02d}", tz="UTC")
        hi = lo + pd.Timedelta(days=1)
        f = (((ds.field("PHASE_mvt") == "DEP") & ds.field("ADEP_mvt").isin(aps))
             | ((ds.field("PHASE_mvt") == "ARR") & ds.field("ADES_mvt").isin(aps))) & \
            (ds.field("MVT_TIME_UTC_mvt") >= lo) & (ds.field("MVT_TIME_UTC_mvt") < hi)
        parts.append(ds.dataset(_month_file(m)).to_table(filter=f).to_pandas())
    return pd.concat(parts, ignore_index=True)


def build(out: pathlib.Path, spec: dict) -> dict:
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    sys.path.insert(0, str(ROOT))
    from prc.pipeline.fm1 import rows

    out = pathlib.Path(out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "weather").mkdir(exist_ok=True)
    (out / "adsb").mkdir(exist_ok=True)
    all_ids = []
    score = spec.get("score")
    for m in spec["months"]:
        t = _slice(m, spec["days"], spec["aps"])
        t.to_parquet(out / "raw" / _month_file(m).name, index=False)
        all_ids.append(t.MVT_ID_mvt.to_numpy())
        for a in spec["aps"]:
            shutil.copy(WEATHER / f"{a}_2025-{m:02d}.csv", out / "weather" / f"{a}_2025-{m:02d}.csv")
    info = {"spec": spec}
    if score:
        s = _slice(score[0], [score[1]], spec["aps"])
        served, labels = rows.hide_labels(s)
        served.to_parquet(out / "raw" / "scored.parquet", index=False)
        labels.rename_axis("MVT_ID_mvt").reset_index().to_parquet(out / "raw" / "labels.parquet", index=False)
        dep = served[served.PHASE_mvt == "DEP"]
        pd.DataFrame({"MVT_ID_mvt": dep.MVT_ID_mvt.to_numpy(dtype="float64"),
                      "TAXITIME_SEC_mvt": np.zeros(len(dep), dtype="int32")}).to_parquet(out / "raw" / "template.parquet",
                                                                                           index=False)
        all_ids.append(s.MVT_ID_mvt.to_numpy())
        info.update(n_scored_dep=int(len(dep)), n_scored_rows=int(len(s)))
    ids = np.unique(np.concatenate(all_ids)).astype("float64")
    tab = pq.read_table(ADSB_TRAIN, filters=[("MVT_ID_mvt", "in", ids.tolist())])
    pq.write_table(tab, out / "adsb" / "train.parquet")
    pq.write_table(tab.schema.empty_table(), out / "adsb" / "serve.parquet")
    (out / "adsb" / "report.json").write_text(json.dumps({"sources": ["adsb.lol"], "slice_of": ADSB_TRAIN.parent.name}))
    info.update(n_adsb_rows=int(tab.num_rows))
    (out / "world.json").write_text(json.dumps(info, indent=1))
    return info


def build_in_child(out: pathlib.Path, spec: dict) -> dict:
    r = subprocess.run([sys.executable, "-B", str(pathlib.Path(__file__).resolve()), str(out), json.dumps(spec)],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"fixture world build failed: {r.stdout[-2000:]} {r.stderr[-3000:]}")
    return json.loads((pathlib.Path(out) / "world.json").read_text())


def config_raw(world: pathlib.Path, **extra) -> dict:
    """A test config over a world: small learners, 2 threads, 2 seeds, the unported modules off."""
    w = pathlib.Path(world)
    raw = {
        "run": {"seeds": [0, 1], "threads": 2, "artifacts_dir": str(w / "artifacts")},
        "data": {"raw_dir": str(w / "raw"), "scored_file": str(w / "raw" / "scored.parquet"),
                 "template_file": str(w / "raw" / "template.parquet"), "weather_dir": str(w / "weather"),
                 "train_months": [1, 2], "training_glob": "training_2025-*.parquet"},
        "airports": {"codes": list(APS), "tz": {"EDDF": "Europe/Berlin", "LSZH": "Europe/Zurich"}},
        # one serve-only row in ~800 is 0.13 %, above the production 0.1 % (FM1 notes N1): fixture scale, stated here
        "adsb": {"table_dir": str(w / "adsb"), "stack": {"models_dir": str(STACK)}, "serve_missing_max_share": 0.01},
        "features": {"orl_routes_file": str(ORL)},
        "matched": {"es_months": [2],
                    "lightgbm": {"learning_rate": 0.1, "num_leaves": 15, "min_data_in_leaf": 20, "max_rounds": 60,
                                 "patience": 10},
                    "catboost": {"learning_rate": 0.2, "depth": 4, "max_rounds": 60, "patience": 10}},
        "unmatched": {"hgb": {"max_iter": 30, "n_iter_no_change": 5, "min_samples_leaf": 5}},
        "modules": {"arc": {"enabled": False}, "rome": {"enabled": False}},
    }
    for path, v in extra.items():
        node = raw
        parts = path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v
    return raw


if __name__ == "__main__":
    build(pathlib.Path(sys.argv[1]), json.loads(sys.argv[2]))
