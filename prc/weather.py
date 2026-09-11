"""The one METAR weather parser in this repository: the frozen archive -> one row per observation.

WHY THIS EXISTS

Version 1 of the weather block (`scripts/stand_ab.load_weather`, 2026-09-09) trusted the archive's
`p01i` field as one-hour precipitation. The Iowa Environmental Mesonet returns p01i as the literal
"0.00" -- not "M" -- for every European station, because European METARs carry no US hourly-
precipitation group: 205,417 of 205,417 observations, 0 of 140 station-months with any other value,
while 18,947 observations report precipitation in their present-weather group. So `w_precip_mm` was
0.0 on every cache row, and the registered freezing rule (Amendment 24.3: cold AND (precipitation OR
a frozen code)) silently lost its precipitation clause and missed cold rain and cold drizzle -- the
conditions under which aircraft are de-iced. Arm W (RESULT 22) was measured on that block. Bug
class BC-3 `sentinel-ingested-as-measurement`, reports/bug_classes.md.

This module replaces every ad-hoc parse of the archive (stand_ab's loader and the atlas's scratch
parser) so the fix lives in one place. It is a library: it reads the frozen archive written by
`scripts/fetch_weather.py`, never fetches, and writes nothing unless `main` is called.

THE SOURCE OF EACH FIELD, AND WHAT WAS CHECKED

  * Precipitation comes from the PRESENT-WEATHER GROUP (`wxcodes`), which the archive has already
    split out of the METAR body. Measured 2026-09-11: `wxcodes` equals the weather tokens of the
    METAR main body (before TEMPO/BECMG/NOSIG/RMK) on 205,396 of 205,417 observations; the 21 others
    are corrected (COR) reports and malformed trend text. Recent weather (RERA...) is not in it.
  * `wxcodes` "M" means the METAR carries no present-weather group, which in FM 15 means no
    significant weather: every flag is 0. An EMPTY `wxcodes` (15 observations, all AUTO reports
    with a "/////////" group: the sensor could not observe) means unknown: every flag is NaN.
  * `p01i` is consumed only in a station-month whose reported values take at least two distinct
    values (`information_content`); elsewhere `precip_mm` is NaN, never 0. On the frozen archive
    that is every station-month, so `precip_mm` is NaN on every row -- which is the truth.
  * A missing gust with a REPORTED wind is 0 kt (a METAR omits the group when there is no gust);
    with the wind missing too, the gust is unknown (NaN). v1 wrote 0 kt for both.
  * Every flag and every derived condition uses three-valued logic: an unknown input yields NaN,
    never 0. v1's `(x <= c).astype(float)` turned an unknown temperature into "not freezing".

THE PRESENT-WEATHER GRAMMAR (WMO FM 15 METAR, 4678 w'w')

Each space-separated token is  [intensity or proximity] [descriptor] [phenomena...]:
    intensity   '-' light, (none) moderate, '+' heavy;  'VC' in the vicinity (not at the station)
    descriptor  MI PR BC DR BL SH TS FZ   (at most one)
    phenomena   precipitation DZ RA SN SG IC PL GR GS UP; obscuration BR FG FU VA DU SA HZ PY;
                other PO SQ FC SS DS
Every token on the frozen archive parses under this grammar (tests pin it); a token that does not
raises, rather than being silently skipped.

THE FLAGS (1.0 / 0.0, NaN when the weather group is unknown), AT THE STATION unless named VC:
    wx_ra wx_dz wx_sg wx_ic wx_pl wx_gs wx_gr wx_up   the phenomenon in any at-station token
    wx_sn         falling snow: SN with no BL/DR descriptor (blowing/drifting snow is lifted from
                  the ground, not precipitation) -- wx_blsn carries BL/DR SN separately
    wx_fzra wx_fzdz wx_fzfg   the FZ descriptor with RA / DZ / FG
    wx_fg         fog of any kind (FG with any descriptor: MI shallow, BC patches, PR partial, FZ)
    wx_br         mist
    wx_ts wx_sh   the TS / SH descriptor
    wx_vcts wx_vcsh wx_vcfg   the same in the vicinity
    wx_precip     any at-station precipitation: wx_ra|wx_dz|wx_sn|wx_sg|wx_ic|wx_pl|wx_gs|wx_gr|wx_up
    wx_intensity  -1 light, 0 moderate, +1 heavy: the heaviest intensity over the at-station
                  tokens that carry precipitation; NaN when there is none (or it is unknown)
    wx_frozen     an at-station token with the FZ descriptor or an SN PL GS GR IC phenomenon (the
                  frozen-code list registered in Amendment 24.3; it includes blowing snow)

THE DERIVED DE-ICING CONDITION -- a hypothesis for the arm to test, not a fact

    dc_moist_cold      temp_c <= +3 AND visible moisture, where visible moisture is any at-station
                       precipitation (wx_precip), fog (wx_fg), mist (wx_br), or vis_km <= 1.5
    dc_frost           temp_c <= 0 AND dewspread_c <= 3
    deicing_condition  dc_moist_cold OR dc_frost

  Sources, quoted as retrieved 2026-09-11, and exactly what each supports:
  * SKYbrary, "Aircraft and In-Flight Icing Risks" (https://skybrary.aero/articles/aircraft-and-
    flight-icing-risks): manufacturers' 'Icing Conditions' threshold "is usually given as the
    presence of visible moisture and an Outside Air Temperature (OAT) ... of less than a figure
    between +3°C and +10°C"; "Visible moisture can be defined in flight as clouds, fog with
    visibility of 1500m or less, and precipitation." This supports the SHAPE of dc_moist_cold. It
    is an ice-protection threshold, not a de-icing trigger; +3 °C is the LOW end of the quoted
    range; counting mist (BR, visibility 1-5 km) as visible moisture goes beyond the quote.
  * FAA, Ground Deicing Program General Information, Winter 2024-2025
    (https://www.faa.gov/other_visit/aviation_industry/airline_operators/airline_safety/deicing/
    24-25_FAA_General_Information_Document.pdf): "Frost HOTs are for active frost conditions in
    which frost is forming. This phenomenon occurs when aircraft surfaces are at or below 0 °C
    (32 °F) and at or below the frost point." The condition is on the aircraft SURFACE temperature
    and the FROST point, neither of which a METAR reports; dc_frost is an air-temperature proxy,
    and its 3 °C spread is this repository's choice, not an FAA number.
  The raw flags are kept beside the condition so an arm can test the rule rather than assume it.

VERSIONING

`VERSION` names the parse. "1" is the v1 loader (p01i trusted); this is "2.0.0". Any change to a
flag's definition, a threshold, or a join rule bumps it, and `main` stamps it into every file it
writes, so a cache can always be traced to the parse that built it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

VERSION = "2.0.0"
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "weather"
OUT_DIR = ROOT / "data" / "weather_obs"

#: the archive columns this parser reads (`metar` is carried through for audit, never parsed)
RAW_FIELDS = ("station", "valid", "tmpf", "dwpf", "sknt", "gust", "vsby", "p01i", "wxcodes", "metar")
#: the numeric fields whose information content is checked per station-month
NUMERIC_FIELDS = ("tmpf", "dwpf", "sknt", "gust", "vsby", "p01i")

MI_TO_KM = 1.609344
IN_TO_MM = 25.4
#: the value a "T" (trace, less than 0.01 in) takes, in MILLIMETRES. v1 substituted 0.05 before the
#: inch->mm conversion and so wrote 1.27 mm; the registered value is 0.05 mm.
TRACE_MM = 0.05
DEICE_MOIST_C = 3.0
DEICE_VIS_KM = 1.5
DEICE_FROST_C = 0.0
DEICE_SPREAD_C = 3.0

DESCRIPTORS = ("MI", "PR", "BC", "DR", "BL", "SH", "TS", "FZ")
PRECIPITATION = ("DZ", "RA", "SN", "SG", "IC", "PL", "GR", "GS", "UP")
OTHER_PHENOMENA = ("BR", "FG", "FU", "VA", "DU", "SA", "HZ", "PY", "PO", "SQ", "FC", "SS", "DS")
FROZEN_PHENOMENA = ("SN", "PL", "GS", "GR", "IC")
_TOKEN = re.compile(r"(?P<prox>[-+]|VC)?(?P<desc>" + "|".join(DESCRIPTORS) + r")?"
                    r"(?P<ph>(?:" + "|".join(PRECIPITATION + OTHER_PHENOMENA) + r")*)")

FLAGS = ["wx_ra", "wx_sn", "wx_dz", "wx_fzra", "wx_fzdz", "wx_fzfg", "wx_pl", "wx_gs", "wx_gr",
         "wx_sg", "wx_ic", "wx_up", "wx_blsn", "wx_fg", "wx_br", "wx_ts", "wx_sh",
         "wx_vcts", "wx_vcsh", "wx_vcfg", "wx_precip", "wx_frozen"]
#: the per-observation contract, in this order
OBS_COLUMNS = (["station", "valid", "temp_c", "dewpoint_c", "dewspread_c", "vis_km", "wind_kt",
                "gust_kt", "precip_mm", "wxcodes"] + FLAGS
               + ["wx_intensity", "dc_moist_cold", "dc_frost", "deicing_condition"])


def parse_token(token: str) -> tuple[str, str, tuple[str, ...]]:
    """One present-weather token -> (proximity '-' '+' 'VC' or '', descriptor or '', phenomena).
    Raises on anything the FM 15 grammar does not produce, so a new code cannot be skipped."""
    m = _TOKEN.fullmatch(token)
    if m is None or not token or (not m.group("desc") and not m.group("ph")):
        raise ValueError(f"unparseable present-weather token {token!r}")
    ph = m.group("ph")
    return m.group("prox") or "", m.group("desc") or "", tuple(ph[i:i + 2] for i in range(0, len(ph), 2))


def _code_row(codes: str) -> dict[str, float]:
    """The flags for one observation's present-weather string (already known, not empty)."""
    f = dict.fromkeys(FLAGS, 0.0)
    intensity = None
    if codes == "M":
        return {**f, "wx_intensity": np.nan}
    for tok in codes.split():
        prox, desc, ph = parse_token(tok)
        if prox == "VC":
            f["wx_vcts"] = max(f["wx_vcts"], float(desc == "TS"))
            f["wx_vcsh"] = max(f["wx_vcsh"], float(desc == "SH"))
            f["wx_vcfg"] = max(f["wx_vcfg"], float("FG" in ph))
            continue
        lifted = desc in ("BL", "DR")
        hit = {
            "wx_ra": "RA" in ph, "wx_dz": "DZ" in ph, "wx_sg": "SG" in ph, "wx_ic": "IC" in ph,
            "wx_pl": "PL" in ph, "wx_gs": "GS" in ph, "wx_gr": "GR" in ph, "wx_up": "UP" in ph,
            "wx_sn": "SN" in ph and not lifted, "wx_blsn": "SN" in ph and lifted,
            "wx_fzra": desc == "FZ" and "RA" in ph, "wx_fzdz": desc == "FZ" and "DZ" in ph,
            "wx_fzfg": desc == "FZ" and "FG" in ph, "wx_fg": "FG" in ph, "wx_br": "BR" in ph,
            "wx_ts": desc == "TS", "wx_sh": desc == "SH",
            "wx_frozen": desc == "FZ" or any(p in FROZEN_PHENOMENA for p in ph),
        }
        for k, v in hit.items():
            if v:
                f[k] = 1.0
        precip = any(p in PRECIPITATION for p in ph) and not (lifted and set(ph) <= {"SN"})
        if precip:
            f["wx_precip"] = 1.0
            level = {"-": -1.0, "": 0.0, "+": 1.0}[prox]
            intensity = level if intensity is None else max(intensity, level)
    return {**f, "wx_intensity": np.nan if intensity is None else intensity}


def parse_codes(wxcodes: pd.Series) -> pd.DataFrame:
    """Present-weather strings -> FLAGS + wx_intensity, one row per input row, same index.
    "M" (no weather group) -> every flag 0; empty/NaN (not observable) -> every column NaN."""
    known = wxcodes.notna() & (wxcodes.astype(str).str.strip() != "")
    cache: dict[str, dict[str, float]] = {}
    rows = []
    for ok, s in zip(known.to_numpy(), wxcodes.to_numpy()):
        if not ok:
            rows.append(None)
            continue
        s = str(s).strip().upper()
        if s not in cache:
            cache[s] = _code_row(s)
        rows.append(cache[s])
    nan_row = dict.fromkeys(FLAGS + ["wx_intensity"], np.nan)
    return pd.DataFrame([r if r is not None else nan_row for r in rows], index=wxcodes.index,
                        columns=FLAGS + ["wx_intensity"], dtype="float64")


def _num(series: pd.Series, field: str) -> np.ndarray:
    """An archive numeric field: "M" is missing; "T" (p01i only) is a trace, returned as NaN here
    and substituted AFTER unit conversion by the caller; anything else must parse, or it raises."""
    s = series.astype(str).str.strip()
    missing = series.isna() | s.isin(["M", ""])
    trace = s == "T"
    if trace.any() and field != "p01i":
        raise ValueError(f"{field}: a trace 'T' is only meaningful for p01i ({int(trace.sum())} rows)")
    v = pd.to_numeric(s.where(~(missing | trace)), errors="coerce")
    bad = v.isna() & ~(missing | trace)
    if bad.any():
        raise ValueError(f"{field}: {int(bad.sum())} values are neither numeric nor 'M', e.g. "
                         f"{s[bad].iloc[0]!r}")
    return v.to_numpy(dtype="float64")


def le3(x: np.ndarray, c: float) -> np.ndarray:
    """x <= c in three-valued logic: NaN where x is unknown."""
    return np.where(np.isnan(x), np.nan, (x <= c).astype("float64"))


def and3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kleene AND: 0 if either is 0, 1 if both are 1, else unknown."""
    return np.where((a == 0) | (b == 0), 0.0, np.where((a == 1) & (b == 1), 1.0, np.nan))


def or3(*xs: np.ndarray) -> np.ndarray:
    """Kleene OR: 1 if any is 1, 0 if all are 0, else unknown."""
    x = np.vstack(xs)
    return np.where((x == 1).any(axis=0), 1.0, np.where((x == 0).all(axis=0), 0.0, np.nan))


def read_archive(raw_dir: pathlib.Path | None = None) -> pd.DataFrame:
    """Every archive CSV as strings, one frame, with `valid` parsed to UTC. Raises when the archive
    is absent, a column is missing, a timestamp does not parse, or (station, valid) repeats."""
    raw_dir = RAW_DIR if raw_dir is None else pathlib.Path(raw_dir)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no weather archive in {raw_dir} (run scripts/fetch_weather.py first)")
    w = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False, na_values=[""]) for f in files],
                  ignore_index=True)
    missing = set(RAW_FIELDS) - set(w.columns)
    if missing:
        raise ValueError(f"weather archive is missing columns {sorted(missing)} (a partial fetch?)")
    w = w[list(RAW_FIELDS)].copy()
    w["valid"] = pd.to_datetime(w.valid, utc=True, errors="coerce")
    if w.valid.isna().any():
        raise ValueError(f"{int(w.valid.isna().sum())} weather rows have an unparseable timestamp")
    dup = w.duplicated(["station", "valid"])
    if dup.any():
        raise ValueError(f"{int(dup.sum())} observations repeat a (station, valid) key: the key must be unique")
    return w.sort_values(["valid", "station"], kind="mergesort").reset_index(drop=True)


def information_content(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (station, month, field): observations, reported values, distinct reported values, and
    `informative` = at least two distinct reported values. Also, per station-month, how many
    observations report precipitation in the weather group -- the evidence a constant-zero p01i
    contradicts.

    Only p01i's verdict gates a value (`parse_observations`). The others are reported: temperature,
    dew point and wind vary in every station-month of the frozen archive, and the one single-valued
    visibility month is the 10 km cap ("9999"/CAVOK = 6.21 mi) held all month -- right-censored but
    true, which is why a constant is a CANDIDATE sentinel to be explained, not proof of one.
    """
    month = raw.valid.dt.strftime("%Y-%m")
    wet = parse_codes(raw.wxcodes).wx_precip.to_numpy()
    out = []
    for field in NUMERIC_FIELDS:
        v = _num(raw[field], field)                     # distinct NUMERIC values: "0.00" == "0"
        if field == "p01i":
            v = np.where(raw[field].astype(str).str.strip().to_numpy() == "T", -1.0, v)   # a trace is its own value
        g = pd.DataFrame({"station": raw.station, "month": month, "reported": ~np.isnan(v),
                          "value": v, "wet": wet == 1.0})
        agg = g.groupby(["station", "month"], sort=True).agg(
            n_obs=("reported", "size"), n_reported=("reported", "sum"),
            n_distinct=("value", "nunique"), n_precip_codes=("wet", "sum")).reset_index()
        agg.insert(2, "field", field)
        out.append(agg)
    info = pd.concat(out, ignore_index=True)
    info["n_reported"] = info.n_reported.astype("int64")
    info["n_precip_codes"] = info.n_precip_codes.astype("int64")
    info["informative"] = info.n_distinct >= 2
    return info


def parse_observations(raw: pd.DataFrame, info: pd.DataFrame | None = None) -> pd.DataFrame:
    """`read_archive`'s frame -> OBS_COLUMNS, one row per observation, same order."""
    info = information_content(raw) if info is None else info
    tmpf, dwpf = _num(raw.tmpf, "tmpf"), _num(raw.dwpf, "dwpf")
    o = pd.DataFrame({"station": raw.station.astype(str).to_numpy(), "valid": raw.valid.array})
    o["temp_c"] = (tmpf - 32.0) * 5.0 / 9.0
    o["dewpoint_c"] = (dwpf - 32.0) * 5.0 / 9.0
    o["dewspread_c"] = o.temp_c - o.dewpoint_c
    o["vis_km"] = _num(raw.vsby, "vsby") * MI_TO_KM
    wind = _num(raw.sknt, "sknt")
    gust = _num(raw.gust, "gust")
    o["wind_kt"] = wind
    o["gust_kt"] = np.where(np.isnan(gust) & ~np.isnan(wind), 0.0, gust)
    p = raw.p01i.astype(str).str.strip()
    mm = _num(raw.p01i, "p01i") * IN_TO_MM
    mm = np.where(p.to_numpy() == "T", TRACE_MM, mm)
    ok = info[(info.field == "p01i") & info.informative]
    keep = pd.MultiIndex.from_frame(ok[["station", "month"]])
    here = pd.MultiIndex.from_arrays([raw.station.astype(str), raw.valid.dt.strftime("%Y-%m")])
    o["precip_mm"] = np.where(here.isin(keep), mm, np.nan)
    o["wxcodes"] = raw.wxcodes.to_numpy()
    codes = parse_codes(raw.wxcodes.reset_index(drop=True))
    for c in codes.columns:
        o[c] = codes[c].to_numpy()
    moisture = or3(o.wx_precip.to_numpy(), o.wx_fg.to_numpy(), o.wx_br.to_numpy(),
                   le3(o.vis_km.to_numpy(), DEICE_VIS_KM))
    o["dc_moist_cold"] = and3(le3(o.temp_c.to_numpy(), DEICE_MOIST_C), moisture)
    o["dc_frost"] = and3(le3(o.temp_c.to_numpy(), DEICE_FROST_C), le3(o.dewspread_c.to_numpy(), DEICE_SPREAD_C))
    o["deicing_condition"] = or3(o.dc_moist_cold.to_numpy(), o.dc_frost.to_numpy())
    return o[OBS_COLUMNS]


def load_observations(raw_dir: pathlib.Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The frozen archive -> (observations, information_content)."""
    raw = read_archive(raw_dir)
    info = information_content(raw)
    return parse_observations(raw, info), info


def hourly(obs: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Join observations onto (station, hour) keys. THE RULE: for key (S, H) -- H a UTC timestamp on
    the hour -- take the latest observation at station S with valid <= H + 1 h (the hour's end,
    INCLUSIVE: a report stamped exactly H + 1 h counts for hour H). `obs_age_s` = (H + 1 h) - valid,
    so a stale report says so; a key with no observation at or before its hour's end gets NaN in
    every column. No observation after the hour's end is ever used, and none from another station.

    `keys` needs `station` and `hour` (tz-aware UTC, on the hour); returned in the keys' order with
    every OBS_COLUMNS column except station, plus `obs_valid` and `obs_age_s`.
    """
    for c in ("station", "hour"):
        if c not in keys.columns:
            raise ValueError(f"keys need a {c!r} column")
    hour = pd.to_datetime(keys.hour)
    if hour.dt.tz is None:
        raise ValueError("keys.hour must be timezone-aware UTC")
    if (hour != hour.dt.floor("h")).any():
        raise ValueError("keys.hour must fall on the hour")
    end = (hour + pd.Timedelta(hours=1)).dt.tz_convert("UTC").astype("datetime64[ns, UTC]")
    left = pd.DataFrame({"_row": np.arange(len(keys)), "station": keys.station.astype(str).to_numpy(),
                         "_end": end.to_numpy()}).sort_values("_end", kind="mergesort")
    right = obs.rename(columns={"valid": "obs_valid"}).copy()
    right["obs_valid"] = right.obs_valid.astype("datetime64[ns, UTC]")
    right = right.sort_values("obs_valid", kind="mergesort")
    j = pd.merge_asof(left, right, left_on="_end", right_on="obs_valid", by="station",
                      direction="backward", allow_exact_matches=True)
    j["obs_age_s"] = (j._end - j.obs_valid).dt.total_seconds()
    j = j.sort_values("_row", kind="mergesort").reset_index(drop=True)
    cols = [c for c in OBS_COLUMNS if c not in ("station", "valid")] + ["obs_valid", "obs_age_s"]
    clash = set(cols) & set(keys.columns)
    if clash:
        raise ValueError(f"keys already carry output columns {sorted(clash)}")
    return pd.concat([keys.reset_index(drop=True), j[cols]], axis=1)


def archive_digest(raw_dir: pathlib.Path) -> str:
    """sha256 over the archive files' names and bytes, in name order: which frozen copy was parsed."""
    h = hashlib.sha256()
    for f in sorted(pathlib.Path(raw_dir).glob("*.csv")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def main(argv=None) -> int:
    """Write `weather_obs_v<VERSION>.parquet` (OBS_COLUMNS), `info_content_v<VERSION>.parquet` and a
    manifest JSON (version, archive digest, row counts, the p01i verdict) into --out-dir."""
    ap = argparse.ArgumentParser(description="parse the frozen METAR archive (prc.weather)")
    ap.add_argument("--raw-dir", default=str(RAW_DIR))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)
    raw_dir, out = pathlib.Path(a.raw_dir), pathlib.Path(a.out_dir)
    obs, info = load_observations(raw_dir)
    out.mkdir(parents=True, exist_ok=True)
    obs_path = out / f"weather_obs_v{VERSION}.parquet"
    info_path = out / f"info_content_v{VERSION}.parquet"
    obs.to_parquet(obs_path, index=False)
    info.to_parquet(info_path, index=False)
    p = info[info.field == "p01i"]
    manifest = {
        "version": VERSION, "archive_digest": archive_digest(raw_dir),
        "n_observations": int(len(obs)), "n_stations": int(obs.station.nunique()),
        "valid_min": str(obs.valid.min()), "valid_max": str(obs.valid.max()),
        "p01i_informative_station_months": int(p.informative.sum()),
        "station_months": int(len(p)),
        "uninformative": sorted(f"{r.station} {r.month} {r.field}" for r in info[~info.informative].itertuples()),
        "files": [obs_path.name, info_path.name],
    }
    (out / f"manifest_v{VERSION}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"prc.weather {VERSION}: {len(obs):,} observations -> {obs_path}; p01i informative in "
          f"{manifest['p01i_informative_station_months']} of {manifest['station_months']} station-months",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
