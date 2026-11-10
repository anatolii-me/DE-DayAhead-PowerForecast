"""
Collect hourly weather data for offshore points from midnight forecast runs.

Main mode: ECMWF IFS HRES
──────────────────────────────────────────────────────────────────────
  API:    https://single-runs-api.open-meteo.com/v1/forecast
  Run:    D 00:00 UTC
  Target: calendar day D+1 in Europe/Berlin, or
          [D+1 00:00 Berlin, D+2 00:00 Berlin).
          This follows the German market calendar. A normal day has 24 hours
          (96 quarter-hours), the spring DST day has 23 hours (92 intervals),
          and the autumn DST day has 25 hours (100 intervals). The target
          window must follow that length instead of using a fixed 24-hour UTC
          window.
  Buffer: one hour before and after the target window in physical UTC time.
          These extra hours are used as lag features. The buffer is added
          after the UTC boundaries are calculated, so DST does not affect it.
  Wind:   wind_speed_{H}m / wind_direction_{H}m. H defaults to 100 m, the
          native ECMWF height.
  Rain:   precipitation in mm, the native sum of rain, showers, and snow.

ICON-EU PLACEHOLDER / TODO
──────────────────────────────────────────────────────────────────────
  ICON-EU is temporarily disabled. It was previously used after 2025-09-01,
  but only ECMWF is kept for stable and consistent data with deeper history.
  To restore ICON-EU as an optional mode:
    - Use the same API: https://single-runs-api.open-meteo.com/v1/forecast
    - DWD ICON-EU provides wind at 80 m and 120 m, not 100 m. Its default
      height should therefore be 80 m.
    - Precipitation uses mm, as in ECMWF, so no separate conversion is needed.
    - Check for gaps in datetime_run_utc where the model segments meet.
  _VARS_ICON_TODO and ICON_EU_SINGLE_SINCE_TODO below are reference constants.
  They are not used in active logic.

Variable notes
──────────────
  precipitation:
    Do not request rain and snowfall separately. ECMWF snowfall is an
    approximation, not a native variable.

  temperature:
    Request temperature_2m and convert it to the requested height with the
    standard atmospheric lapse rate of -0.0065 K/m.

  cell_selection=sea:
    Required for offshore data. Otherwise the API may choose a land cell.

Row uniqueness
──────────────
  datetime_run_utc stores the run that produced each row, always D 00:00 UTC.
  The one-hour buffers of neighboring runs overlap by design. As a result,
  one (point_id, datetime_utc) can appear in several runs. The natural key is
  (point_id, datetime_run_utc, datetime_utc). The preprocessing step decides
  which overlapping run to use for each lag hour.

  model_source was removed because this module currently uses only ECMWF IFS
  HRES. A separate source label is unnecessary while ICON-EU is disabled.
"""
 
from typing import cast
import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
import niquests
from retry_requests import retry
from zoneinfo import ZoneInfo
import time
 
BERLIN_TZ = ZoneInfo("Europe/Berlin")
 
# ── Geographic points ─────────────────────────────────────────────────────────
wind_offshore_pts = {
    # Measurement points in North Sea
    1001: [6.558306, 54.887932, 'North Sea'],
    1002: [6.510380, 55.455643, 'North Sea'], # North
    1003: [7.704829, 55.302157, 'North Sea'], # North-East
    1004: [7.696224, 54.774739, 'North Sea'], # East
    1005: [7.017257, 54.552354, 'North Sea'], # Center
    1006: [7.393611, 54.095704, 'North Sea'], # Near Helgoland
    1007: [5.806714, 53.808608, 'North Sea'], # South-West
    1008: [5.264095, 54.282091, 'North Sea'], # West
 
    # Measurement points in Baltic Sea
    2001: [12.325362, 54.646623, 'Baltic Sea'], # East
    2002: [12.858980, 54.819784, 'Baltic Sea'], # South
    2003: [12.912983, 55.167817, 'Baltic Sea'], # North-East
    2004: [13.647442, 55.035451, 'Baltic Sea'], # North
    2005: [14.206500, 55.156726, 'Baltic Sea'], # North-West
    2006: [14.442079, 54.634527, 'Baltic Sea'], # South-West
 
    # The remaining points outline wind farm clusters for visualization only.
    # South of the North Sea Turbines
    4: [6.109906, 54.0196156, 'Turbine'],
    5: [6.2895012, 54.0210829, 'Turbine'],
    6: [6.4241695, 53.9672452, 'Turbine'],
    7: [6.4890575, 53.9164217, 'Turbine'],
    8: [6.603899, 53.9321401, 'Turbine'],
    9: [6.7476654, 53.961489, 'Turbine'],
    10: [7.134161, 54.0110473, 'Turbine'],
    11: [7.1090126, 54.0935301, 'Turbine'],
    12: [6.821394, 54.0926241, 'Turbine'],
    13: [6.5517998, 54.0892513, 'Turbine'],
    14: [6.4155865, 54.0875648, 'Turbine'],
    15: [6.2865829, 54.0773689, 'Turbine'],
    16: [6.0788727, 54.0636698, 'Turbine'],
    17: [6.0788727, 54.0636698, 'Turbine'],
 
    # East of the North Sea Turbines
    18: [7.6413345, 54.3470519, 'Turbine'],
    19: [7.6409912, 54.5388123, 'Turbine'],
    20: [7.7783203, 54.5388123, 'Turbine'],
    21: [7.7524853, 54.4338596, 'Turbine'],
    22: [7.7663898, 54.3860556, 'Turbine'],
 
    # North of the North Sea Turbines
    23: [7.7395248, 55.0090279, 'Turbine'],
    24: [7.7412415, 55.0425323, 'Turbine'],
    25: [7.7998638, 55.0605275, 'Turbine'],
    26: [7.7999496, 54.9666769, 'Turbine'],
 
    28: [7.1704674, 55.2345539, 'Turbine'],
    29: [7.2085762, 55.2227571, 'Turbine'],
    31: [7.2464275, 55.0742397, 'Turbine'],
    32: [7.1745014, 55.0675562, 'Turbine'],
 
    33: [6.847229, 55.2853724, 'Turbine'],
    34: [6.8003654, 55.2797997, 'Turbine'],
    35: [6.9066238, 55.1268824, 'Turbine'],
    36: [6.8597603, 55.1212875, 'Turbine'],
 
    # West of the North Sea Turbines
    37: [5.7440472, 54.3041051, 'Turbine'],
    38: [6.2029839, 54.495069, 'Turbine'],
    39: [6.3198853, 54.4981595, 'Turbine'],
    40: [6.3181686, 54.5409036, 'Turbine'],
    41: [6.415844, 54.540804, 'Turbine'],
    42: [6.3682938, 54.3897541, 'Turbine'],
    43: [6.2905312, 54.3294381, 'Turbine'],
    44: [5.9098721, 54.304055, 'Turbine'],
    45: [5.9101295, 54.275048, 'Turbine'],
    46: [5.8071327, 54.2760503, 'Turbine'],
    47: [6.0182762, 54.4220262, 'Turbine'],
    48: [6.133976, 54.3294381, 'Turbine'],
 
    # Baltic Sea Turbines
    481: [13.9883423, 54.9199405, 'Turbine'],
    49: [14.0302277, 54.9214204, 'Turbine'],
    50: [14.1768265, 54.8097872, 'Turbine'],
    51: [14.1397476, 54.7531612, 'Turbine'],
    52: [14.0322876, 54.8012786, 'Turbine'],
 
    53: [13.9214802, 54.7892051, 'Turbine'],
    54: [13.8972759, 54.8606038, 'Turbine'],
    55: [13.815136, 54.8565033, 'Turbine'],
    56: [13.801403, 54.8139419, 'Turbine'],
 
    57: [13.7432098, 54.7884133, 'Turbine'],
    58: [13.6129189, 54.8535882, 'Turbine'],
    59: [13.5730934, 54.8299633, 'Turbine'],
 
    60: [13.0838585, 54.9496254, 'Turbine'],
    61: [13.1516647, 55.0052871, 'Turbine'],
    62: [13.228569, 55.008585, 'Turbine'],
    63: [13.198185, 54.9772689, 'Turbine'],
 
    64: [12.6299858, 54.6109012, 'Turbine'],
    65: [12.6918697, 54.6306797, 'Turbine'],
    66: [12.6299858, 54.5839512, 'Turbine'],
}
 
# Keep only measurement points, not turbine outline points.
SEA_POINTS = {
    pid: coords
    for pid, coords in wind_offshore_pts.items()
    if coords[2] in ('North Sea', 'Baltic Sea')
}
 
# ── Earliest available date ───────────────────────────────────────────────────
ECMWF_SINGLE_SINCE = pd.Timestamp("2024-03-14")
 
# ── ECMWF request variables, native at 100 m ──────────────────────────────────
_VARS_ECMWF_BASE = [
    "temperature_2m",
    "surface_pressure",
    "precipitation",
]
 
def _vars_ecmwf(height: int) -> list[str]:
    return [f"wind_speed_{height}m", f"wind_direction_{height}m"] + _VARS_ECMWF_BASE
 
 
# ── TODO: optional ICON-EU mode; not used in active code ───────────────────────
ICON_EU_SINGLE_SINCE_TODO = pd.Timestamp("2025-09-01")
_VARS_ICON_TODO = [
    "wind_speed_80m",
    "wind_direction_80m",
    "temperature_2m",
    "surface_pressure",
    "precipitation",
]
# ICON-EU should default to 80 m because DWD ICON-EU does not provide 100 m wind.
 
# ── Output DataFrame schema ───────────────────────────────────────────────────
OUTPUT_COLS = [
    "datetime_run_utc",      # model run time: day D at 00:00 UTC
    "datetime_utc",          # time covered by the forecast
    "datetime_berlin",       # same time in Europe/Berlin (CET/CEST)
    "point_id",
    "region",
    "latitude",
    "longitude",
    "wind_speed",            # m/s at height_m
    "wind_direction",        # degrees at height_m
    "height_m",              # wind and temperature height
    "temperature_2m",        # surface temperature in °C
    "temperature_hub",       # temperature at height_m in °C
    "surface_pressure",      # hPa
    "air_density",           # kg/m³ at height_m
    "precipitation",         # mm/h water equivalent
]
 
# ── Physical constants ────────────────────────────────────────────────────────
_R_DRY_AIR  = 287.05   # J / (kg·K)
_LAPSE_RATE = 0.0065   # K/m
 
 
# ── Client with cache and retries ─────────────────────────────────────────────
def _build_client() -> openmeteo_requests.Client:
    # A 200 response can still contain an incomplete run. Never retain it
    # across the next scheduled attempt; validated parquet is our durable cache.
    cache_session = requests_cache.CachedSession(backend="memory", expire_after=0)
    retry_session = cast(
        niquests.Session,
        retry(cache_session, retries=5, backoff_factor=0.2),
    )
    return openmeteo_requests.Client(session=retry_session)
 
 
# ── Parse an API response into a UTC DataFrame ────────────────────────────────
def _parse_response(response, var_names: list[str]) -> pd.DataFrame:
    hourly = response.Hourly()
    times  = pd.date_range(
        start=pd.to_datetime(hourly.Time(),    unit="s", utc=True),
        end  =pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq =pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
    data: dict = {"datetime_utc": times}
    for i, var in enumerate(var_names):
        data[var] = hourly.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)
 
 
# ── Derived features ──────────────────────────────────────────────────────────
def _add_derived(df: pd.DataFrame, height: int) -> pd.DataFrame:
    df = df.copy()
 
    speed_col = f"wind_speed_{height}m"
    dir_col   = f"wind_direction_{height}m"
    df = df.rename(columns={speed_col: "wind_speed", dir_col: "wind_direction"})
    df["height_m"] = height
 
    # Temperature at height: °C to K, apply the lapse rate, then back to °C.
    t2m_k   = df["temperature_2m"] + 273.15
    t_hub_k = t2m_k - _LAPSE_RATE * height
    df["temperature_hub"] = t_hub_k - 273.15
 
    # Air density: rho = P / (R_d * T), with pressure converted from hPa to Pa.
    df["air_density"] = (df["surface_pressure"] * 100.0).astype(float) / (_R_DRY_AIR * t_hub_k)
 
    return df
 
 
# ── Add metadata and normalize to OUTPUT_COLS ─────────────────────────────────
def _normalize(
    df: pd.DataFrame,
    pid: int,
    region: str,
    lat: float,
    lon: float,
    run_time_utc: pd.Timestamp,
) -> pd.DataFrame:
    df = df.copy()
    df["datetime_run_utc"] = run_time_utc
    # datetime_berlin is the same instant as datetime_utc, shown in the
    # Europe/Berlin time zone. tz_convert handles CET/CEST for the date.
    df["datetime_berlin"]  = df["datetime_utc"].dt.tz_convert(BERLIN_TZ)
    df["point_id"]         = pid
    df["region"]           = region
    df["latitude"]         = lat
    df["longitude"]        = lon
 
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan
 
    return df[OUTPUT_COLS]
 
 
# ── Fetch one day through the Single Runs API ─────────────────────────────────
def _fetch_day(
    day: pd.Timestamp,
    lats: list, lons: list, ids: list, points: dict,
    model: str,
    var_names: list[str],
    height: int,
    client,
) -> pd.DataFrame:
    """
    Fetch the run from day D at 00:00 UTC.

    The target is calendar day D+1 in Europe/Berlin:
    [D+1 00:00 Berlin, D+2 00:00 Berlin). A normal day has 24 hours, the spring
    DST day has 23, and the autumn DST day has 25. ZoneInfo handles this when it
    localizes Berlin midnight. Add a one-hour UTC buffer on both sides for lag
    features.
    """
    url      = "https://single-runs-api.open-meteo.com/v1/forecast"
    run_time_utc = pd.Timestamp(day.year, day.month, day.day, tz="UTC")
    run_str  = run_time_utc.strftime("%Y-%m-%dT%H:%M")
 
    params = {
        "latitude":        lats,
        "longitude":       lons,
        "run":             run_str,
        "hourly":          var_names,
        "wind_speed_unit": "ms",
        "cell_selection":  "sea",   # use sea cells for offshore points
        "models":          model,
    }
 
    print(f"  [Single Run | {model}] run {run_str}, {len(ids)} points...")
    responses = client.weather_api(url, params=params, timeout=60)
 
    # Calendar day D+1 in Europe/Berlin. ZoneInfo converts local midnight to
    # the correct UTC instant for the date, including DST.
    next_day      = day + pd.Timedelta(days=1)
    period_start  = pd.Timestamp(
        year=next_day.year, month=next_day.month, day=next_day.day,
        tzinfo=BERLIN_TZ,
    ).tz_convert("UTC")
    period_end_local_date = next_day + pd.Timedelta(days=1)
    period_end    = pd.Timestamp(
        year=period_end_local_date.year, month=period_end_local_date.month,
        day=period_end_local_date.day, tzinfo=BERLIN_TZ,
    ).tz_convert("UTC")
 
    fetch_start = period_start - pd.Timedelta(hours=1)
    fetch_end   = period_end + pd.Timedelta(hours=1)
 
    frames = []
    for i, response in enumerate(responses):
        pid = ids[i]
        df  = _parse_response(response, var_names)
        df  = df[(df["datetime_utc"] >= fetch_start) & (df["datetime_utc"] < fetch_end)].copy()
        df  = _add_derived(df, height)
        df  = _normalize(df, pid, points[pid][2],
                         response.Latitude(), response.Longitude(),
                         run_time_utc)
        frames.append(df)
 
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=OUTPUT_COLS)
 
 
# ── Public function ───────────────────────────────────────────────────────────
def fetch_offshore_weather(
    start_date: str,
    end_date:   str,
    points:     dict | None = None,
    height:     int = 100,
    model:      str = "ecmwf_ifs",
) -> pd.DataFrame:
    """
    Collect hourly ECMWF IFS HRES weather data for offshore points.

    Parameters
    ----------
    start_date : "YYYY-MM-DD"
        Date of the first run. The run starts at 00:00 UTC on day D.
    end_date : "YYYY-MM-DD"
        Date of the last run, inclusive.
    points : dict | None
        Points to request. Defaults to SEA_POINTS.
    height : int
        Wind and temperature height in meters. The default is ECMWF's native
        100 m level.
    model : str
        Only "ecmwf_ifs" is currently supported. See the module TODO for
        ICON-EU.

    Returns
    -------
    pd.DataFrame
        Data with the OUTPUT_COLS schema. For a run on day D, the target is
        calendar day D+1 in Europe/Berlin, with 23, 24, or 25 hours depending
        on DST, plus a one-hour buffer on both sides for lag features. The row
        key is (point_id, datetime_run_utc, datetime_utc).
    """
    if model != "ecmwf_ifs":
        raise NotImplementedError(
            "Only model='ecmwf_ifs' is currently supported. "
            "ICON-EU is disabled for now; see the TODO in the module docstring."
        )
 
    if points is None:
        points = SEA_POINTS
 
    lats = [coords[1] for coords in points.values()]
    lons = [coords[0] for coords in points.values()]
    ids  = list(points.keys())
    client = _build_client()
 
    start_ts = pd.Timestamp(start_date)
    end_ts   = pd.Timestamp(end_date)
 
    if start_ts < ECMWF_SINGLE_SINCE:
        print(f"Warning: start_date {start_date} is earlier than the first available "
              f"ECMWF Single Run date, {ECMWF_SINGLE_SINCE.date()}. Clipping it.")
        start_ts   = ECMWF_SINGLE_SINCE
        start_date = start_ts.strftime("%Y-%m-%d")
 
    var_names = _vars_ecmwf(height)
 
    print(f"Run range {start_date} -> {end_date} (ECMWF IFS HRES, height {height} m)")
    print("  Target for each run: calendar day D+1 in Europe/Berlin")
    print("  (23 hours at spring DST, 25 at autumn DST, otherwise 24), +/-1-hour buffer\n")
 
    parts: list[pd.DataFrame] = []
    for day in pd.date_range(start=start_ts, end=end_ts, freq="D"):
        try:
            parts.append(
                _fetch_day(day, lats, lons, ids, points, model, var_names, height, client)
            )
        except Exception as e:
            print(f"    Warning: error for run {day.date()} [{model}]: {e}")
        # Wait 1.5 seconds between days, including after a failed request.
        time.sleep(1.5)
 
    if not parts:
        return pd.DataFrame(columns=OUTPUT_COLS)
 
    result = (
        pd.concat(parts, ignore_index=True)
          .sort_values(["point_id", "datetime_run_utc", "datetime_utc"])
          .reset_index(drop=True)
    )
    result["datetime_run_utc"] = pd.to_datetime(result["datetime_run_utc"], utc=True)
    result["datetime_utc"]     = pd.to_datetime(result["datetime_utc"], utc=True)
    # Rebuild datetime_berlin from the typed datetime_utc column. This is more
    # reliable than assuming concatenation preserves the time-zone-aware dtype.
    result["datetime_berlin"]  = result["datetime_utc"].dt.tz_convert(BERLIN_TZ)
 
    # Check for duplicates in the natural key.
    dup_mask = result.duplicated(subset=["point_id", "datetime_run_utc", "datetime_utc"], keep=False)
    n_dups   = int(dup_mask.sum())
 
    # Diagnostics
    total  = len(result)
    nan_ws = int(result["wind_speed"].isna().sum())
    rows_per_run = result.groupby(["point_id", "datetime_run_utc"]).size()
    print(f"\nTotal: {total} rows, {result['point_id'].nunique()} points.")
    print(f"   NaN wind_speed     : {nan_ws} ({100 * nan_ws / max(total, 1):.1f}%)")
    print(f"   Duplicate natural keys (point_id, datetime_run_utc, datetime_utc): {n_dups}")
    print(f"   Height (height_m):\n{result['height_m'].value_counts().to_string()}")
    print(f"   Rows per (point_id, run), distribution:\n{rows_per_run.value_counts().sort_index().to_string()}")
 
    return result
