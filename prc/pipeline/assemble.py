"""Assemble: one frame keyed by MVT_ID with every scored row's final value and its provenance.

Each lane's ids must be EXACTLY the rows the routing gave it (no more, no fewer); no id may come from two
lanes; the union must be the template. The submission is in the template's row order, int32 like every
submission this project has written. Nothing here reads a previous submission.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import errors as E

INT32_MAX = np.iinfo(np.int32).max
PROVENANCE_OUT = ["MVT_ID_mvt", "TAXITIME_SEC_mvt", "airport", "row_class", "lane", "model", "stage", "fallback"]


@dataclass(frozen=True)
class Assembled:
    submission: pd.DataFrame                    # MVT_ID_mvt float64, TAXITIME_SEC_mvt int32, template order
    provenance: pd.DataFrame                    # PROVENANCE_OUT, template order

    def lane_counts(self) -> dict:
        return {str(k): int(v) for k, v in self.provenance.lane.value_counts().sort_index().items()}


def assemble(results, routing, template: pd.DataFrame) -> Assembled:
    names = [r.lane for r in results]
    if len(set(names)) != len(names):
        raise E.LaneError(f"a lane returned twice: {names}")
    routed = set(routing.lane.tolist())
    missing_lanes = routed - set(names)
    if missing_lanes:
        raise E.LaneError(f"routed lane(s) {sorted(missing_lanes)} returned no result")
    extra_lanes = set(names) - routed
    if extra_lanes:
        raise E.LaneError(f"lane(s) {sorted(extra_lanes)} returned rows but were routed none")
    for r in results:
        want = set(routing.ids[routing.mask(r.lane)].tolist())
        got = set(r.ids.tolist())
        if got != want:
            raise E.LaneError(f"lane {r.lane}: returned {len(got):,} rows, routed {len(want):,} "
                              f"({len(want - got):,} routed rows missing, {len(got - want):,} rows not routed to it)")
    frame = pd.concat([r.provenance.assign(TAXITIME_SEC_mvt=r.values) for r in results], ignore_index=True)
    if not frame.MVT_ID_mvt.is_unique:
        raise E.RoutingError(f"{int(frame.MVT_ID_mvt.duplicated().sum()):,} ids predicted by more than one lane")
    t_ids = template.MVT_ID_mvt.to_numpy(dtype="float64")
    if set(frame.MVT_ID_mvt.tolist()) != set(t_ids.tolist()):
        raise E.ValidationError(f"the lanes cover {len(frame):,} ids, the template {len(t_ids):,}: different sets")
    v = frame.TAXITIME_SEC_mvt.to_numpy(dtype="float64")
    if (v < 1).any() or (v > INT32_MAX).any():
        raise E.ValidationError(f"values outside [1, int32 max]: {int(((v < 1) | (v > INT32_MAX)).sum())} rows")
    frame["TAXITIME_SEC_mvt"] = v.astype("int32")
    info = pd.DataFrame({"MVT_ID_mvt": routing.ids, "airport": routing.airport, "row_class": routing.row_class,
                         "fallback": routing.fallback})
    frame = frame.merge(info, on="MVT_ID_mvt", how="left", validate="one_to_one")
    frame = frame.set_index("MVT_ID_mvt").loc[t_ids].reset_index()[PROVENANCE_OUT]
    sub = pd.DataFrame({"MVT_ID_mvt": frame.MVT_ID_mvt.to_numpy(dtype="float64"),
                        "TAXITIME_SEC_mvt": frame.TAXITIME_SEC_mvt.to_numpy().astype("int32")})
    return Assembled(submission=sub, provenance=frame)
