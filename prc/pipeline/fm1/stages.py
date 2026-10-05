"""The FM1 stage plan and the value ledger (plans/PREREG_fm1_2026_09_28.md §1; AT5, AT6).

The order of stages is FIXED here (STAGE_ORDER); the config only switches stages on or off. Two guarantees:

  structural   check_order() refuses any plan in which a value stage follows a calibration stage (run at import on
               STAGE_ORDER, and on every plan).
  runtime      the Ledger refuses a value write once any row has been calibrated, and a second calibration. The v43
               EGLL-7 defect (BLK re-valued EGLL-July after MTX had computed its shift on the earlier values, FMA §0.2 / G8)
               cannot be expressed: the write raises StageOrderError.

Every row's `decided_by` is the label of the LAST value stage that wrote it, suffixed `+mtx` when calibration moved it; the
vocabulary is closed (DECIDED_BY). The ledger keeps the pre-calibration value and the calibration shift per row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import errors as E

KINDS = ("check", "features", "value", "gate", "calibration", "format")


@dataclass(frozen=True)
class StageSpec:
    name: str
    kind: str
    label: str | None = None           # decided_by token for a value stage (the base model's comes from its learner)


STAGE_ORDER = (
    StageSpec("input_contract", "check"),
    StageSpec("eobt_clip", "features"),        # a serve-time repair of the raw EOBT_1 field, before the feature path
    StageSpec("features", "features"),
    StageSpec("base_model", "value", "base"),
    StageSpec("adsb_gate", "gate"),
    StageSpec("adsb_stack", "value", "adsb_stack"),
    StageSpec("cbv_blend", "value", "cbv_blend"),
    StageSpec("unmatched_body", "value", "unmatched_body"),
    # the shipped chain's order (Amendment A3): Rome's R2 / E1 / RLD (v7-v12) -> ARC (v17) -> E2 (v31) -> E2W (v33); E2's row
    # set reads the Rome value AND the value after ARC, so ARC sits between them; the storm anchor (v16, inside v24-v26)
    # shifts only the rows the unmatched body still decides
    StageSpec("rome", "value", "rome"),
    StageSpec("arc", "value", "arc"),
    StageSpec("rome_e2", "value", "rome_e2"),
    StageSpec("rome_e2w", "value", "rome_e2w"),
    StageSpec("storm", "value", "storm"),
    StageSpec("uad", "value", "uad"),
    StageSpec("ldi", "value", "ldi"),
    StageSpec("mtx", "calibration", "mtx"),
    StageSpec("finalise", "format"),
)
BASE_LABELS = ("base_lightgbm", "base_catboost")
VALUE_LABELS = (*BASE_LABELS, "adsb_stack", "cbv_blend", "unmatched_body", "rome", "arc", "rome_e2", "rome_e2w", "storm",
                "uad", "ldi")
DECIDED_BY = frozenset(VALUE_LABELS) | frozenset(f"{v}+mtx" for v in VALUE_LABELS)


def check_order(stages) -> None:
    """Refuse an unknown stage, stages out of the fixed order, and a value stage after a calibration stage."""
    pos = {s.name: i for i, s in enumerate(STAGE_ORDER)}
    for s in stages:
        if s.name not in pos:
            raise E.StageOrderError(f"stage {s.name!r} is not a known stage")
    rest = [pos[s.name] for s in stages if s.kind != "calibration"]
    if rest != sorted(rest):
        raise E.StageOrderError(f"stages {[s.name for s in stages]} are not in the fixed order "
                                f"{[s.name for s in STAGE_ORDER]}")
    msg = _calibration_violation(stages)
    if msg:
        raise E.StageOrderError(msg)
    idx = [pos[s.name] for s in stages]
    if idx != sorted(idx):
        raise E.StageOrderError(f"stages {[s.name for s in stages]} are not in the fixed order "
                                f"{[s.name for s in STAGE_ORDER]}")


def _calibration_violation(stages) -> str | None:
    seen = None
    for s in stages:
        if s.kind == "calibration":
            seen = s
        elif s.kind == "value" and seen is not None:
            return f"value stage {s.name!r} after calibration stage {seen.name!r} (calibration must run last)"
    return None


check_order(STAGE_ORDER)


def _stage(name: str) -> StageSpec:
    return next(s for s in STAGE_ORDER if s.name == name)


def plan(cfg) -> tuple:
    """The enabled stages, in the fixed order. Refuses every enabled component FM1 has not ported."""
    todo = cfg.not_ported()
    if todo:
        raise E.ComponentNotPortedError(f"enabled component(s) not ported to FM1: {', '.join(todo)} (switch them off, e.g. "
                                        "--set modules.rome.enabled=false, or port them; plans/PREREG_fm1_2026_09_28.md §2)")
    on = {"input_contract": True, "features": True, "finalise": True, "eobt_clip": cfg["modules.eobt_clip.enabled"],
          "base_model": cfg["lanes.matched"],
          "adsb_gate": cfg["lanes.matched"] and cfg["adsb.stack.enabled"],
          "adsb_stack": cfg["lanes.matched"] and cfg["adsb.stack.enabled"],
          "cbv_blend": cfg["lanes.matched"] and cfg["matched.blend.catboost_weight"] > 0,
          "unmatched_body": cfg["lanes.unmatched"],
          "rome": cfg["modules.rome.enabled"], "arc": cfg["modules.arc.enabled"],
          "rome_e2": cfg["modules.rome.enabled"] and cfg["modules.rome.e2"],
          "rome_e2w": cfg["modules.rome.enabled"] and cfg["modules.rome.e2"] and cfg["modules.rome.e2w"],
          "storm": cfg["modules.storm.mode"] == "anchor", "uad": cfg["modules.uad.enabled"],
          "ldi": cfg["modules.ldi.enabled"], "mtx": cfg.mtx_enabled}
    stages = tuple(s for s in STAGE_ORDER if on[s.name])
    check_order(stages)
    return stages


class Ledger:
    """The per-row value store of one predict run."""

    def __init__(self, ids):
        ids = np.asarray(ids, dtype="float64")
        if not pd.Index(ids).is_unique:
            raise E.RoutingError(f"ledger: {int(pd.Index(ids).duplicated().sum())} duplicate ids")
        self.ids = ids
        self._pos = pd.Index(ids)
        n = len(ids)
        self.value = np.full(n, np.nan)
        self.before = np.full(n, np.nan)
        self.shift = np.zeros(n)
        self.decided_by = np.full(n, None, dtype=object)
        self.calibrated_by: StageSpec | None = None
        self.history: list = []

    def _rows(self, ids, values) -> tuple:
        ids = np.asarray(ids, dtype="float64")
        values = np.asarray(values, dtype="float64")
        if values.shape != ids.shape:
            raise E.LaneError(f"{len(values)} values for {len(ids)} ids")
        if not np.isfinite(values).all():
            raise E.LaneError(f"{int((~np.isfinite(values)).sum())} non-finite value(s)")
        pos = self._pos.get_indexer(ids)
        if (pos < 0).any():
            raise E.RoutingError(f"{int((pos < 0).sum())} id(s) not in the ledger")
        return pos, values

    def write(self, stage: StageSpec, ids, values, label: str | None = None) -> None:
        if stage.kind != "value":
            raise E.StageOrderError(f"{stage.name!r} is not a value stage; it cannot write values")
        if self.calibrated_by is not None:
            raise E.StageOrderError(f"{stage.name!r} wrote a value after calibration {self.calibrated_by.name!r}; calibration "
                                    "must be the last value change (the v43 EGLL-7 defect class)")
        label = label or stage.label
        if label not in VALUE_LABELS:
            raise E.StageOrderError(f"label {label!r} is not in the decided_by vocabulary {sorted(VALUE_LABELS)}")
        pos, values = self._rows(ids, values)
        self.value[pos] = values
        self.decided_by[pos] = label
        self.history.append((stage.name, int(len(pos))))

    def calibrate(self, stage: StageSpec, ids, values) -> None:
        if stage.kind != "calibration":
            raise E.StageOrderError(f"{stage.name!r} is not a calibration stage")
        if self.calibrated_by is not None:
            raise E.StageOrderError(f"calibrated twice ({self.calibrated_by.name!r}, then {stage.name!r})")
        pos, values = self._rows(ids, values)
        if np.isnan(self.value[pos]).any():
            raise E.StageOrderError("calibration of a row no value stage has served")
        self.before = self.value.copy()
        moved = pos[values != self.value[pos]]
        self.shift[pos] = values - self.value[pos]
        self.value[pos] = values
        for i in moved:
            self.decided_by[i] = f"{self.decided_by[i]}+{stage.label}"
        self.calibrated_by = stage
        self.history.append((stage.name, int(len(pos))))

    def values_of(self, ids) -> np.ndarray:
        """The current value of each id (NaN where no stage has written it); RoutingError on an id not in the ledger."""
        return self.value[self._positions(ids)].copy()

    def decided_of(self, ids) -> np.ndarray:
        """The label of the last value stage that wrote each id (None where none has)."""
        return self.decided_by[self._positions(ids)].copy()

    def _positions(self, ids) -> np.ndarray:
        pos = self._pos.get_indexer(np.asarray(ids, dtype="float64"))
        if (pos < 0).any():
            raise E.RoutingError(f"{int((pos < 0).sum())} id(s) not in the ledger")
        return pos

    def counts(self) -> dict:
        s = pd.Series(self.decided_by[pd.notna(self.decided_by)])
        return {str(k): int(v) for k, v in s.value_counts().sort_index().items()}

    def finalise(self) -> pd.DataFrame:
        unserved = np.isnan(self.value)
        if unserved.any():
            raise E.RoutingError(f"{int(unserved.sum())} row(s) no stage served (e.g. ids {self.ids[unserved][:3].tolist()})")
        before = self.before if self.calibrated_by is not None else self.value
        return pd.DataFrame({"MVT_ID_mvt": self.ids,
                             "TAXITIME_SEC_mvt": np.maximum(np.rint(self.value), 1).astype("int32"),
                             "decided_by": self.decided_by.astype(str),
                             "value_before_calibration": before.copy(),
                             "calibration_shift": self.shift.copy()})
