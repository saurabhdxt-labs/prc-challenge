"""Named errors for the pipeline. Every boundary raises one of these, never a bare Exception, so a
caller (and a test) can tell a config mistake from a schema mistake from a routing mistake."""
from __future__ import annotations


class PipelineError(Exception):
    """Base of every error the pipeline raises on purpose."""


# ---- config -------------------------------------------------------------------------------------
class ConfigError(PipelineError):
    """The config file is invalid. The message names the dotted path of the offending key."""


class UnknownKeyError(ConfigError):
    """A key the schema does not know (a typo is never silently ignored)."""


class ConfigTypeError(ConfigError):
    """A value of the wrong type (a string where a list is expected, a bool where an int is, ...)."""


class ConfigValueError(ConfigError):
    """A value of the right type outside its allowed set or range."""


class UnknownAirportError(ConfigError):
    """An airport code that the registry does not carry (in the config, or in a data file)."""


class UnknownModelError(ConfigError):
    """A model name that is not in the plug-in registry."""


class UnknownLaneError(ConfigError):
    """A lane id referenced by an airport that the lanes section does not define."""


class PinnedValueError(ConfigError):
    """P2a wraps frozen, shipped code: a config value that would change its behaviour is refused
    (changing it is a registered change for a later phase, not a config edit)."""


# ---- data ---------------------------------------------------------------------------------------
class SchemaError(PipelineError):
    """A data file or frame violates its declared schema."""


class MissingColumnError(SchemaError):
    pass


class DtypeError(SchemaError):
    pass


class NullabilityError(SchemaError):
    pass


class LabelLeakError(PipelineError):
    """A label column (BLOCK_TIME_UTC_mvt / TAXITIME_SEC_mvt) reached a scored-row frame."""


# ---- lanes, assembly, output ---------------------------------------------------------------------
class RoutingError(PipelineError):
    """A scored row lands in no lane, or in more than one."""


class LaneError(PipelineError):
    """A lane's output violates its contract (wrong ids, non-finite values, ...)."""


class ValidationError(PipelineError):
    """The assembled submission fails a contract check."""


class SpliceRefusedError(PipelineError):
    """The pipeline never splices a previous submission file (the `lgbm_submit --base` landmine)."""
