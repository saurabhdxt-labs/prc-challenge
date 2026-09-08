"""The free answers: what taxi-out you can predict with no model at all.

This module exists to fix the denominator. A gradient booster that beats zero
has said nothing; a gradient booster that beats the deepest rung here has said
something. Every result in this project is reported as reduction against the
best of these, computed on the same rows with the same folds.

The ladder is also where domain knowledge enters, and the deepest rung is not
arbitrary. In the taxi-out literature the *unimpeded taxi time* — the time a
movement would take with nothing in its way — is defined operationally as a
low percentile of the observed distribution for the same stand and runway
(Lee, Malik and Jung, NASA Ames, AIAA 2016-3910, which uses the 10th
percentile of the gate-spot-runway distribution). Stand and runway are both
0.0% null in this dataset, so that definition is directly computable, and the
gap between a movement's actual taxi-out and its unimpeded time is the
congestion the model is really being asked to predict.

Fallback is deliberate. An unseen or thin cell steps back exactly one rung,
never straight to a global constant, because a stand that is new at a familiar
airport-and-runway is far better described by that airport and runway than by
Europe as a whole.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LABEL = "TAXITIME_SEC_mvt"
AIRPORT = "ADEP_mvt"
RUNWAY = "RUNWAY_mvt"
STAND = "STAND_mvt"
OFF_BLOCK = "BLOCK_TIME_UTC_mvt"

#: Below this many observations a cell is not trusted and falls back one rung.
DEFAULT_MIN_COUNT = 30

#: The percentile that defines "unimpeded" in the taxi-out literature.
UNIMPEDED_QUANTILE = 0.10


@dataclass(frozen=True)
class Rung:
    name: str
    keys: tuple[str, ...]


def _hour(frame: pd.DataFrame) -> pd.Series:
    return frame[OFF_BLOCK].dt.hour


#: Coarse to fine, and a STRICT nesting: every rung extends the one before it
#: by exactly one key. That is not tidiness. Fallback walks this list backwards,
#: so if two rungs partitioned the data along different dimensions — say
#: airport-and-runway versus airport-and-hour — a thin cell would fall sideways
#: into an unrelated partition rather than into a coarser description of itself.
#: Hour sits at the fine end because time of day refines a stand-and-runway
#: pairing; it does not generalise it.
RUNGS: tuple[Rung, ...] = (
    Rung("global", ()),
    Rung("airport", (AIRPORT,)),
    Rung("airport_runway", (AIRPORT, RUNWAY)),
    Rung("airport_runway_stand", (AIRPORT, RUNWAY, STAND)),
    Rung("airport_runway_stand_hour", (AIRPORT, RUNWAY, STAND, "_hour")),
)


def rmse(actual, predicted) -> float:
    a = np.asarray(actual, dtype="float64")
    p = np.asarray(predicted, dtype="float64")
    return float(np.sqrt(np.mean((a - p) ** 2)))


def _with_hour(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(_hour=_hour(frame)) if "_hour" not in frame else frame


def unimpeded(frame: pd.DataFrame, quantile: float = UNIMPEDED_QUANTILE) -> dict:
    """Unimpeded taxi time per (airport, runway, stand), the published definition.

    Returned as a plain dict so it can be inspected, serialised and argued with,
    rather than hidden inside a fitted object.
    """
    grouped = frame.groupby([AIRPORT, RUNWAY, STAND], dropna=False)[LABEL]
    return {k: float(v) for k, v in grouped.quantile(quantile).items()}


@dataclass
class Ladder:
    """Nested group means, consulted finest first, each backed by a count."""

    tables: dict[str, pd.Series] = field(default_factory=dict)
    counts: dict[str, pd.Series] = field(default_factory=dict)
    global_value: float = 0.0
    min_count: int = DEFAULT_MIN_COUNT
    max_depth: int = len(RUNGS) - 1

    @classmethod
    def fit(cls, train: pd.DataFrame, min_count: int = DEFAULT_MIN_COUNT,
            max_depth: int | None = None) -> "Ladder":
        """Fit on TRAINING ROWS ONLY. Passing evaluation rows here would make
        every headroom number downstream a fiction."""
        df = _with_hour(train)
        depth = len(RUNGS) - 1 if max_depth is None else max_depth
        obj = cls(min_count=min_count, max_depth=depth,
                  global_value=float(df[LABEL].mean()))
        for rung in RUNGS[1:depth + 1]:
            grouped = df.groupby(list(rung.keys), dropna=False)[LABEL]
            obj.tables[rung.name] = grouped.mean()
            obj.counts[rung.name] = grouped.size()
        return obj

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Finest trusted rung per row, stepping back exactly one rung at a
        time. Never returns null; never returns a non-positive value."""
        df = _with_hour(frame)
        out = np.full(len(df), np.nan, dtype="float64")
        for rung in reversed(RUNGS[1:self.max_depth + 1]):
            unresolved = np.isnan(out)
            if not unresolved.any():
                break
            keys = pd.MultiIndex.from_frame(df.loc[unresolved, list(rung.keys)]) \
                if len(rung.keys) > 1 else pd.Index(df.loc[unresolved, rung.keys[0]])
            values = self.tables[rung.name].reindex(keys).to_numpy("float64")
            counts = self.counts[rung.name].reindex(keys).to_numpy("float64")
            # A cell backed by too few observations is not evidence, it is noise.
            values = np.where(counts >= self.min_count, values, np.nan)
            out[unresolved] = values
        out = np.where(np.isfinite(out), out, self.global_value)
        return np.maximum(out, 1.0)


def ladder_report(train: pd.DataFrame, test: pd.DataFrame,
                  min_count: int = DEFAULT_MIN_COUNT) -> dict:
    """RMSE of every rung on the same held-out rows. This table is the
    denominator for every model claim that follows."""
    report = {"train_rows": int(len(train)), "test_rows": int(len(test)),
              "min_count": min_count, "rungs": {}}
    best = None
    for depth, rung in enumerate(RUNGS):
        ladder = Ladder.fit(train, min_count=min_count, max_depth=depth)
        score = rmse(test[LABEL], ladder.predict(test))
        report["rungs"][rung.name] = score
        if best is None or score < best[1]:
            best = (rung.name, score)
    report["best_rung"], report["best_rmse"] = best
    return report
