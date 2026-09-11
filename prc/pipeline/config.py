"""The pipeline config: a typed, validated, immutable view of `configs/pipeline.yaml`.

Config is code (CLAUDE.md Part 1 rule 11): every key is known, every type checked, every reference
resolved at load, and a mistake fails loud with a named error that carries the dotted path of the
offending key -- an unknown key is never ignored, a bool is never accepted as an int, an airport that
the registry does not carry is never routed silently.

P2a pins: the shipped functions hard-code some values (LIRF rule constants, the rule order, the airport
the Rome rules are written for). The config may state them and must state them correctly; a different
value raises PinnedValueError. `lanes.check_pins()` re-checks the pins against the scripts' own constants
at run time, so a script edit cannot drift away from this file silently either.

No-splice: no config key may name a previous submission (`base`, `base_submission`, ...) and no path may
point into `submissions/` -- the pipeline builds every row from data, never from an earlier file
(the `lgbm_submit --base` landmine, plans/PIPELINE_DESIGN_2026_09_10.md).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re
import types
from dataclasses import dataclass, field
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import errors as E

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 1

AIRPORT_RE = re.compile(r"^[A-Z]{4}$")
TEAM_RE = re.compile(r"^[a-z]+-[a-z]+$")                  # build_submission.SUBMISSION_RE's team part
ROW_CLASSES = ("matched", "unmatched")                     # AOBT_3_flt present / absent on the scored row
LANE_KINDS = ("matched", "unmatched", "airport_rules")
#: which lane kinds may serve which row class
KIND_FOR_CLASS = {"matched": ("matched",), "unmatched": ("unmatched", "airport_rules")}
#: plug-in names (models.REGISTRY keys, spelled out here so loading a config never imports a model)
MODEL_NAMES = ("lightgbm", "nonfill_body", "nonfill_body_congestion", "catboost", "nn", "blend")
MODELS_FOR_KIND = {
    "matched": ("lightgbm", "catboost", "nn", "blend"),
    "unmatched": ("nonfill_body", "nonfill_body_congestion", "catboost", "nn", "blend"),
    "airport_rules": ("nonfill_body",),
}
TARGETS = ("delta", "y")                                  # lgbm_submit.TARGETS
SOURCES = ("saved_boosters", "refit")
#: lgbm_submit's block order (FEATS [+ QUEUE] [+ DAY] [+ ORDER] [+ WEATHER]); a block list must follow it
FEATURE_BLOCKS = ("queue", "day", "order", "weather")
#: the Rome rules, in the order v7 -> v10 applied them; P2a allows an ordered subset, never a reorder
RULE_ORDER = ("fill_classifier", "dateslip", "schedule_floor", "local_day_schedule")   # + RLD (2026-09-11), last
#: values the wrapped scripts hard-code (rome_bandfloor.LO / HI / AIRPORT; rome_fill, rome_dateslip and
#: fit_unmatched's `_lirf_fill_model` are written for this airport). lanes.check_pins() re-reads them.
PINNED = types.MappingProxyType({"schedule_floor_lo_s": 24_000.0, "schedule_floor_hi_s": 86_400.0,
                                 "rules_airport": "LIRF",
                                 # rome_local_day.SP_LO / SP_HI (plans/PREREG_rome_local_day_rule_2026_09_11.md)
                                 "local_day_lo_s": 24_000.0, "local_day_hi_s": 86_400.0})
#: keys that would splice an earlier submission; refused with the landmine named
SPLICE_KEYS = ("base", "base_submission", "base_file", "splice", "previous_submission")
SUBMISSIONS_DIR = "submissions"


# =================================================================================================
# the typed view
# =================================================================================================

@dataclass(frozen=True)
class Paths:
    root: pathlib.Path
    scored_file: pathlib.Path
    template_file: pathlib.Path
    training_dir: pathlib.Path
    training_glob: str
    stand_cache: pathlib.Path
    rome_cache: pathlib.Path

    def training_files(self) -> list:
        """The raw training months, sorted (the order every shipped loader uses)."""
        return sorted(self.training_dir.glob(self.training_glob))


@dataclass(frozen=True)
class Folds:
    fold_a_holdout_months: tuple
    early_stopping_months: tuple
    lomo_months: tuple


@dataclass(frozen=True)
class ScheduleFloor:
    lo_s: float
    hi_s: float


@dataclass(frozen=True)
class LocalDay:
    """RLD: LIRF unmatched rows with lo <= sp < hi whose take-off is on the schedule's LOCAL day take rint(sp)."""
    lo_s: float
    hi_s: float


@dataclass(frozen=True)
class AirportRules:
    order: tuple = ()                           # the enabled rules, in application order
    schedule_floor: ScheduleFloor | None = None
    local_day: LocalDay | None = None

    @property
    def fill_classifier(self) -> bool:
        return "fill_classifier" in self.order

    @property
    def dateslip(self) -> bool:
        return "dateslip" in self.order

    def __bool__(self) -> bool:
        return bool(self.order)


@dataclass(frozen=True)
class Airport:
    code: str
    tz: str
    history: bool
    lanes: Mapping[str, str]                    # row class -> lane id
    rules: AirportRules = field(default_factory=AirportRules)

    @property
    def fallback(self) -> bool:
        """No training history: its rows ride the pooled lanes, flagged in the manifest."""
        return not self.history


@dataclass(frozen=True)
class AdsbStage:
    """The matched lane's ADS-B stack stage (prc/pipeline/adsb_stage.py): prc-challenge-6e's feature table and ADN models."""
    table: pathlib.Path
    models_dir: pathlib.Path


@dataclass(frozen=True)
class Lane:
    id: str
    kind: str
    model: str
    target: str | None = None
    feature_blocks: tuple = ()                  # ((block, cache dir), ...) in lgbm_submit's order
    source: str | None = None
    booster_dir: pathlib.Path | None = None
    boosters: tuple = ()
    adsb_stage: AdsbStage | None = None        # matched lane only; absent = today's lane exactly (A1)


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    team: str
    seeds: tuple
    paths: Paths
    folds: Folds
    lanes: Mapping[str, Lane]
    airports: Mapping[str, Airport]
    config_hash: str                            # sha256 of the canonical JSON of the validated mapping
    source_path: pathlib.Path | None = None

    @property
    def airport_codes(self) -> tuple:
        return tuple(self.airports)

    def lane_for(self, airport: str, row_class: str) -> str:
        if airport not in self.airports:
            raise E.UnknownAirportError(f"airport {airport!r} is not in the config registry "
                                        f"({', '.join(sorted(self.airports))}); add an entry under `airports`")
        if row_class not in ROW_CLASSES:
            raise E.RoutingError(f"row class {row_class!r} is not one of {ROW_CLASSES}")
        return self.airports[airport].lanes[row_class]

    def fallback_airports(self) -> tuple:
        return tuple(c for c, a in self.airports.items() if a.fallback)


# =================================================================================================
# validation helpers: each names the dotted path
# =================================================================================================

def _type_name(v) -> str:
    return type(v).__name__


def _map(v, path: str, required=(), optional=()) -> dict:
    if not isinstance(v, Mapping):
        raise E.ConfigTypeError(f"{path}: expected a mapping, got {_type_name(v)} {v!r}")
    for k in v:
        if not isinstance(k, str):
            raise E.ConfigTypeError(f"{path}: keys must be strings, got {_type_name(k)} {k!r}")
        if k in SPLICE_KEYS:
            raise E.SpliceRefusedError(
                f"{path}.{k}: the pipeline never splices a previous submission (the lgbm_submit --base landmine: "
                "its default base is v2); every row is built from data")
    unknown = [k for k in v if k not in required and k not in optional]
    if unknown:
        raise E.UnknownKeyError(f"{path}: unknown key(s) {sorted(unknown)}; allowed: {sorted([*required, *optional])}")
    missing = [k for k in required if k not in v]
    if missing:
        raise E.ConfigError(f"{path}: missing required key(s) {missing}")
    return dict(v)


def _str(v, path: str) -> str:
    if not isinstance(v, str) or not v:
        raise E.ConfigTypeError(f"{path}: expected a non-empty string, got {_type_name(v)} {v!r}")
    return v


def _bool(v, path: str) -> bool:
    if not isinstance(v, bool):
        raise E.ConfigTypeError(f"{path}: expected true/false, got {_type_name(v)} {v!r}")
    return v


def _int(v, path: str, lo=None, hi=None) -> int:
    if isinstance(v, bool) or not isinstance(v, int):             # a YAML `true` is an int in Python
        raise E.ConfigTypeError(f"{path}: expected an integer, got {_type_name(v)} {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise E.ConfigValueError(f"{path}: {v} is outside [{lo}, {hi}]")
    return v


def _num(v, path: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise E.ConfigTypeError(f"{path}: expected a number, got {_type_name(v)} {v!r}")
    return float(v)


def _list(v, path: str, nonempty=True) -> list:
    if not isinstance(v, list):
        raise E.ConfigTypeError(f"{path}: expected a list, got {_type_name(v)} {v!r}")
    if nonempty and not v:
        raise E.ConfigValueError(f"{path}: must not be empty")
    return v


def _months(v, path: str) -> tuple:
    months = tuple(_int(m, f"{path}[{i}]", 1, 12) for i, m in enumerate(_list(v, path)))
    if len(set(months)) != len(months):
        raise E.ConfigValueError(f"{path}: repeated month in {list(months)}")
    return months


def _choice(v, path: str, allowed, err=E.ConfigValueError) -> str:
    _str(v, path)
    if v not in allowed:
        raise err(f"{path}: {v!r} is not one of {list(allowed)}")
    return v


def _path(v, path: str, root: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(_str(v, path))
    p = p if p.is_absolute() else root / p
    if SUBMISSIONS_DIR in p.resolve().relative_to(p.resolve().anchor).parts:
        raise E.SpliceRefusedError(f"{path}: {v!r} points into {SUBMISSIONS_DIR}/; the pipeline never reads a previous "
                                   "submission (no splice)")
    return p


# =================================================================================================
# the sections
# =================================================================================================

def _parse_paths(v, root) -> Paths:
    d = _map(v, "paths", required=("scored_file", "template_file", "training_dir", "training_glob",
                                   "stand_cache", "rome_cache"))
    return Paths(root=root, scored_file=_path(d["scored_file"], "paths.scored_file", root),
                 template_file=_path(d["template_file"], "paths.template_file", root),
                 training_dir=_path(d["training_dir"], "paths.training_dir", root),
                 training_glob=_str(d["training_glob"], "paths.training_glob"),
                 stand_cache=_path(d["stand_cache"], "paths.stand_cache", root),
                 rome_cache=_path(d["rome_cache"], "paths.rome_cache", root))


def _parse_folds(v) -> Folds:
    d = _map(v, "folds", required=("fold_a_holdout_months", "early_stopping_months", "lomo_months"))
    f = Folds(fold_a_holdout_months=_months(d["fold_a_holdout_months"], "folds.fold_a_holdout_months"),
              early_stopping_months=_months(d["early_stopping_months"], "folds.early_stopping_months"),
              lomo_months=_months(d["lomo_months"], "folds.lomo_months"))
    if set(f.early_stopping_months) & set(f.fold_a_holdout_months):
        raise E.ConfigValueError("folds: early_stopping_months overlap fold_a_holdout_months "
                                 f"({sorted(set(f.early_stopping_months) & set(f.fold_a_holdout_months))})")
    return f


def _parse_seeds(v) -> tuple:
    seeds = tuple(_int(s, f"seeds[{i}]", 0) for i, s in enumerate(_list(v, "seeds")))
    if len(set(seeds)) != len(seeds):
        raise E.ConfigValueError(f"seeds: repeated seed in {list(seeds)} (it would be double counted in a mean)")
    return seeds


def _parse_lane(lane_id: str, v, root, seeds) -> Lane:
    p = f"lanes.{lane_id}"
    base = _map(v, p, required=("kind", "model"),
                optional=("target", "feature_blocks", "source", "booster_dir", "boosters", "adsb_stage"))
    kind = _choice(base["kind"], f"{p}.kind", LANE_KINDS)
    model = _choice(base["model"], f"{p}.model", MODEL_NAMES, err=E.UnknownModelError)
    if model not in MODELS_FOR_KIND[kind]:
        raise E.ConfigValueError(f"{p}.model: {model!r} cannot serve a {kind!r} lane; allowed {list(MODELS_FOR_KIND[kind])}")
    if kind != "matched":
        extra = [k for k in ("target", "feature_blocks", "source", "booster_dir", "boosters", "adsb_stage") if k in base]
        if extra:
            raise E.UnknownKeyError(f"{p}: {extra} apply to a matched lane only (this lane is {kind!r})")
        return Lane(id=lane_id, kind=kind, model=model)
    d = _map(v, p, required=("kind", "model", "target", "feature_blocks", "source"),
             optional=("booster_dir", "boosters", "adsb_stage"))
    stage = None
    if "adsb_stage" in d:
        a = _map(d["adsb_stage"], f"{p}.adsb_stage", required=("table", "models_dir"))
        stage = AdsbStage(table=_path(a["table"], f"{p}.adsb_stage.table", root),
                          models_dir=_path(a["models_dir"], f"{p}.adsb_stage.models_dir", root))
    target = _choice(d["target"], f"{p}.target", TARGETS)
    fb = _map(d["feature_blocks"], f"{p}.feature_blocks", optional=FEATURE_BLOCKS)
    names = list(fb)
    if names != [b for b in FEATURE_BLOCKS if b in fb]:
        raise E.ConfigValueError(f"{p}.feature_blocks: {names} is not in lgbm_submit's block order {list(FEATURE_BLOCKS)}")
    blocks = tuple((b, _path(fb[b], f"{p}.feature_blocks.{b}", root)) for b in names)
    source = _choice(d["source"], f"{p}.source", SOURCES)
    if source == "saved_boosters":
        if "boosters" not in d or "booster_dir" not in d:
            raise E.ConfigError(f"{p}: source saved_boosters needs `booster_dir` and `boosters`")
        files = tuple(_str(b, f"{p}.boosters[{i}]") for i, b in enumerate(_list(d["boosters"], f"{p}.boosters")))
        if len(files) != len(seeds):
            raise E.ConfigValueError(f"{p}.boosters: {len(files)} booster(s) for {len(seeds)} seed(s); one per seed, in seed order")
        if len(set(files)) != len(files):
            raise E.ConfigValueError(f"{p}.boosters: a booster is listed twice")
        return Lane(id=lane_id, kind=kind, model=model, target=target, feature_blocks=blocks, source=source,
                    booster_dir=_path(d["booster_dir"], f"{p}.booster_dir", root), boosters=files, adsb_stage=stage)
    extra = [k for k in ("booster_dir", "boosters") if k in d]
    if extra:
        raise E.UnknownKeyError(f"{p}: {extra} apply to source saved_boosters only (source is refit)")
    return Lane(id=lane_id, kind=kind, model=model, target=target, feature_blocks=blocks, source=source, adsb_stage=stage)


def _parse_rules(code: str, v) -> AirportRules:
    p = f"airports.{code}.rules"
    d = _map(v, p, optional=RULE_ORDER)
    order = tuple(d)
    if list(order) != [r for r in RULE_ORDER if r in d]:
        raise E.PinnedValueError(f"{p}: {list(order)} is not in the shipped order {list(RULE_ORDER)} (P2a allows a subset, "
                                 "never a reorder)")
    for r in ("fill_classifier", "dateslip"):
        if r in d:
            _map(d[r] if d[r] is not None else {}, f"{p}.{r}")      # no parameters in P2a
    floor = None
    if "schedule_floor" in d:
        f = _map(d["schedule_floor"], f"{p}.schedule_floor", required=("lo_s", "hi_s"))
        floor = ScheduleFloor(lo_s=_num(f["lo_s"], f"{p}.schedule_floor.lo_s"), hi_s=_num(f["hi_s"], f"{p}.schedule_floor.hi_s"))
        if (floor.lo_s, floor.hi_s) != (PINNED["schedule_floor_lo_s"], PINNED["schedule_floor_hi_s"]):
            raise E.PinnedValueError(f"{p}.schedule_floor: ({floor.lo_s:g}, {floor.hi_s:g}) differs from rome_bandfloor's "
                                     f"(LO, HI) = ({PINNED['schedule_floor_lo_s']:g}, {PINNED['schedule_floor_hi_s']:g}); "
                                     "changing the band is a registered change, not a config edit")
    local_day = None
    if "local_day_schedule" in d:
        f = _map(d["local_day_schedule"], f"{p}.local_day_schedule", required=("lo_s", "hi_s"))
        local_day = LocalDay(lo_s=_num(f["lo_s"], f"{p}.local_day_schedule.lo_s"),
                             hi_s=_num(f["hi_s"], f"{p}.local_day_schedule.hi_s"))
        if (local_day.lo_s, local_day.hi_s) != (PINNED["local_day_lo_s"], PINNED["local_day_hi_s"]):
            raise E.PinnedValueError(f"{p}.local_day_schedule: ({local_day.lo_s:g}, {local_day.hi_s:g}) differs from "
                                     f"rome_local_day's (SP_LO, SP_HI) = ({PINNED['local_day_lo_s']:g}, "
                                     f"{PINNED['local_day_hi_s']:g}); the band is registered in RLD's prereg")
    if order and code != PINNED["rules_airport"]:
        raise E.PinnedValueError(f"{p}: the Rome rules (rome_fill / rome_dateslip / rome_bandfloor) are written for "
                                 f"{PINNED['rules_airport']} only; generalising them to {code} is a P2b change")
    return AirportRules(order=order, schedule_floor=floor, local_day=local_day)


def _parse_airport(code: str, v, lanes: Mapping[str, Lane]) -> Airport:
    p = f"airports.{code}"
    if not AIRPORT_RE.match(code):
        raise E.UnknownAirportError(f"{p}: {code!r} is not a four-letter ICAO code")
    d = _map(v, p, required=("tz", "history", "lanes"), optional=("rules",))
    tz = _str(d["tz"], f"{p}.tz")
    try:
        ZoneInfo(tz)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise E.ConfigValueError(f"{p}.tz: {tz!r} is not an IANA time zone ({exc})") from None
    history = _bool(d["history"], f"{p}.history")
    ln = _map(d["lanes"], f"{p}.lanes", required=ROW_CLASSES)
    routed = {}
    for rc in ROW_CLASSES:
        lane_id = _str(ln[rc], f"{p}.lanes.{rc}")
        if lane_id not in lanes:
            raise E.UnknownLaneError(f"{p}.lanes.{rc}: lane {lane_id!r} is not defined under `lanes` "
                                     f"({', '.join(sorted(lanes))})")
        if lanes[lane_id].kind not in KIND_FOR_CLASS[rc]:
            raise E.ConfigValueError(f"{p}.lanes.{rc}: lane {lane_id!r} is a {lanes[lane_id].kind!r} lane; "
                                     f"{rc} rows need one of {list(KIND_FOR_CLASS[rc])}")
        routed[rc] = lane_id
    rules = _parse_rules(code, d["rules"] if d.get("rules") is not None else {})
    rules_lane = lanes[routed["unmatched"]].kind == "airport_rules"
    if rules and not rules_lane:
        raise E.ConfigValueError(f"{p}: rules are configured but the unmatched lane {routed['unmatched']!r} is not an "
                                 "airport_rules lane; they would never run")
    if rules_lane and not rules:
        raise E.ConfigValueError(f"{p}: routed to the airport_rules lane {routed['unmatched']!r} with no rules")
    if rules and not history:
        raise E.ConfigValueError(f"{p}: airport rules are fitted on the airport's own history; history is false")
    return Airport(code=code, tz=tz, history=history, lanes=types.MappingProxyType(routed), rules=rules)


def canonical_hash(mapping: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace): comment- and layout-insensitive."""
    blob = json.dumps(mapping, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def parse_config(raw: Mapping[str, Any], root: pathlib.Path | None = None,
                 source_path: pathlib.Path | None = None) -> PipelineConfig:
    """Validate a mapping (the parsed YAML) into a PipelineConfig; raise a named ConfigError otherwise."""
    root = pathlib.Path(root) if root is not None else ROOT
    d = _map(raw, "<config>", required=("schema_version", "team", "seeds", "paths", "folds", "lanes", "airports"))
    if _int(d["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise E.ConfigValueError(f"schema_version: {d['schema_version']} (this loader reads {SCHEMA_VERSION})")
    team = _str(d["team"], "team")
    if not TEAM_RE.match(team):
        raise E.ConfigValueError(f"team: {team!r} does not match the submission naming convention {TEAM_RE.pattern}")
    seeds = _parse_seeds(d["seeds"])
    paths = _parse_paths(d["paths"], root)
    folds = _parse_folds(d["folds"])
    lanes_raw = _map(d["lanes"], "lanes", optional=tuple(d["lanes"]) if isinstance(d["lanes"], Mapping) else ())
    if not lanes_raw:
        raise E.ConfigValueError("lanes: at least one lane is required")
    lanes = {lid: _parse_lane(_str(lid, "lanes.<id>"), lv, root, seeds) for lid, lv in lanes_raw.items()}
    ap_raw = _map(d["airports"], "airports", optional=tuple(d["airports"]) if isinstance(d["airports"], Mapping) else ())
    if not ap_raw:
        raise E.ConfigValueError("airports: at least one airport is required")
    airports = {code: _parse_airport(code, av, lanes) for code, av in ap_raw.items()}
    unused = sorted(set(lanes) - {a.lanes[rc] for a in airports.values() for rc in ROW_CLASSES})
    if unused:
        raise E.ConfigValueError(f"lanes: {unused} defined but routed from no airport")
    return PipelineConfig(schema_version=SCHEMA_VERSION, team=team, seeds=seeds, paths=paths, folds=folds,
                          lanes=types.MappingProxyType(lanes), airports=types.MappingProxyType(airports),
                          config_hash=canonical_hash(raw), source_path=source_path)


def load_config(path, root: pathlib.Path | None = None) -> PipelineConfig:
    """Read and validate a YAML config file."""
    import yaml

    path = pathlib.Path(path)
    if not path.exists():
        raise E.ConfigError(f"config file {path} does not exist")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise E.ConfigError(f"{path}: not valid YAML ({exc})") from None
    return parse_config(raw, root=root, source_path=path.resolve())
