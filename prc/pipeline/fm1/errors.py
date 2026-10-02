"""FM1's named errors. Each subclasses the v13 pipeline's errors (prc/pipeline/errors.py), so a caller that catches
`PipelineError` / `ConfigError` / `SchemaError` also catches these; each names ONE failure a test can assert on."""
from __future__ import annotations

from .. import errors as E

PipelineError = E.PipelineError
ConfigError = E.ConfigError
UnknownKeyError = E.UnknownKeyError
ConfigTypeError = E.ConfigTypeError
ConfigValueError = E.ConfigValueError
UnknownAirportError = E.UnknownAirportError
SchemaError = E.SchemaError
MissingColumnError = E.MissingColumnError
DtypeError = E.DtypeError
LabelLeakError = E.LabelLeakError
SpliceRefusedError = E.SpliceRefusedError


class ComponentNotPortedError(E.ConfigError):
    """A component the config enables has no FM1 implementation yet (plans/PREREG_fm1_2026_09_28.md §2)."""


class LearnerCutError(E.ConfigValueError):
    """A learner the owner cut (XGBoost, decision 5) or one outside the days 1-3 scope."""


class ForbiddenInputError(E.LabelLeakError):
    """A design or a scored row carries an input FM1 never reads: a label, a label-derived column or a `*_sec` column."""


class RangeError(E.SchemaError):
    """A value outside its contract range (a negative taxi time, a timestamp outside the declared period, ...)."""


class DuplicateIdError(E.SchemaError):
    """An id column that must be unique is not."""


class StageOrderError(E.PipelineError):
    """A stage plan or a stage run violates the fixed order; in particular a value stage after a calibration stage."""


class ArtifactError(E.PipelineError):
    """A model artefact is missing, incomplete, tampered with, or would be overwritten."""


class ReuseError(E.PipelineError):
    """A derived fit (`fit(..., reuse_from=donor)`) cannot take the donor's fitted matched learners: a knob outside the
    reuse allow-list, the input data, the design or the encodings differ, or the donor lacks a learner the variant needs."""


class WorkerError(E.PipelineError):
    """A learner worker process (CatBoost in ~/.venvs/prc-catboost) failed or returned a malformed result."""


class PartialDayError(E.PipelineError):
    """A set-level rule that reads every row of a day (LDI's severe-day test) was served a batch holding only part of a
    day of the scored file, or a batch mixing scored-file rows with rows from elsewhere."""

NullabilityError = E.NullabilityError
RoutingError = E.RoutingError
LaneError = E.LaneError
ValidationError = E.ValidationError
