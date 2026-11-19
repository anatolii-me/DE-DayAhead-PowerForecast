"""Hourly ECMWF single runs shared by load, wind and solar research.

Dates identify inclusive 00:00 UTC model runs. Each run covers the next
Europe/Berlin calendar day plus one physical hour on either side. The natural
key is (point_id, datetime_run_utc, datetime_utc); buffers overlap across runs.
Coordinates in input mappings are [longitude, latitude, region]. Output
coordinates are the model grid coordinates returned by Open-Meteo.

The offshore wrapper preserves the historical schema and derived formulas.
New consumers retain explicit API variable names and choose land/sea cells.
"""
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast, Literal
from zoneinfo import ZoneInfo
import time
import warnings

import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
import niquests
from retry_requests import retry

from utils.offshore.points import SEA_POINTS, wind_offshore_pts

BERLIN_TZ = ZoneInfo("Europe/Berlin")
ECMWF_SINGLE_SINCE = pd.Timestamp("2024-03-14")
SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
METADATA_COLS = [
    "datetime_run_utc", "datetime_utc", "datetime_berlin",
    "point_id", "region", "latitude", "longitude",
]
LOAD_VARIABLES = ("temperature_2m", "wind_speed_10m", "shortwave_radiation", "cloud_cover")
LOAD_OUTPUT_COLS = METADATA_COLS + list(LOAD_VARIABLES)


def _vars_ecmwf(height: int) -> list[str]:
    return [f"wind_speed_{height}m", f"wind_direction_{height}m",
            "temperature_2m", "surface_pressure", "precipitation"]


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
 
 
def _build_client() -> openmeteo_requests.Client:
    # A 200 response can still contain an incomplete run. Never retain it
    # across the next scheduled attempt; validated parquet is our durable cache.
    cache_session = requests_cache.CachedSession(backend="memory", expire_after=0)
    retry_session = cast(
        niquests.Session,
        # urllib3 otherwise retries HTTP 429 when Retry-After is present,
        # even though 429 is absent from status_forcelist. That can sleep
        # for an hour/day before our checkpoint handler ever sees the limit.
        retry(cache_session, retries=5, backoff_factor=0.2,
              respect_retry_after_header=False),
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


def _target_hours(day: pd.Timestamp) -> pd.DatetimeIndex:
    """D+1 in Berlin, including both lag hours, across either DST transition."""
    start = (day + pd.DateOffset(days=1)).tz_localize(BERLIN_TZ).tz_convert("UTC")
    end = (day + pd.DateOffset(days=2)).tz_localize(BERLIN_TZ).tz_convert("UTC")
    return pd.date_range(start - pd.Timedelta(hours=1), end, freq="h")


def _is_rate_limit_error(exc: Exception) -> bool:
    """Recognize Open-Meteo's HTTP 429, including the SDK's wrapped error."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if getattr(getattr(exc, "response", None), "status_code", None) == 429:
            return True
        for arg in exc.args:
            if isinstance(arg, dict):
                reason = str(arg.get("reason", "")).lower()
                if "limit exceeded" in reason or "too many requests" in reason:
                    return True
        if "429" in str(exc) and "too many requests" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _save_weather_checkpoint(parts: list[pd.DataFrame], path: Path,
                             columns: list[str]) -> pd.DataFrame:
    result = (pd.concat(parts, ignore_index=True)
              .sort_values(["point_id", "datetime_run_utc", "datetime_utc"])
              .reset_index(drop=True)) if parts else pd.DataFrame(columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        result[columns].to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if result.empty:
        print(f"  Saved no runs: {path}")
    else:
        dates = pd.to_datetime(result.datetime_run_utc, utc=True).dt.date
        print(f"  Saved {dates.nunique()} runs, {dates.min()} -> {dates.max()} "
              f"({len(result)} rows): {path}")
    return result


def _fetch_day(
    day: pd.Timestamp,
    points: Mapping,
    model: str,
    var_names: list[str],
    cell_selection: str,
    client,
    require_complete: bool,
) -> pd.DataFrame:
    run_time = day.tz_localize("UTC")
    params = {
        "latitude": [coords[1] for coords in points.values()],
        "longitude": [coords[0] for coords in points.values()],
        "run": run_time.strftime("%Y-%m-%dT%H:%M"),
        "hourly": var_names,
        "wind_speed_unit": "ms",
        "cell_selection": cell_selection,
        "models": model,
    }
    print(f"  [Single Run | {model}] run {params['run']}, {len(points)} points...")
    responses = client.weather_api(SINGLE_RUNS_URL, params=params, timeout=60)
    if len(responses) != len(points):
        raise ValueError(f"Expected {len(points)} point responses, received {len(responses)}.")

    expected = _target_hours(day)
    frames = []
    for (pid, coords), response in zip(points.items(), responses):
        frame = _parse_response(response, var_names)
        frame = frame.loc[frame.datetime_utc.between(expected[0], expected[-1])].copy()
        if require_complete:
            if not pd.DatetimeIndex(frame.datetime_utc).equals(expected):
                raise ValueError(f"Incomplete hourly weather grid for point {pid}.")
            if not np.isfinite(frame[var_names].to_numpy()).all():
                raise ValueError(f"Missing or non-finite weather values for point {pid}.")
        frame["datetime_run_utc"] = run_time
        frame["datetime_berlin"] = frame.datetime_utc.dt.tz_convert(BERLIN_TZ)
        frame["point_id"] = pid
        frame["region"] = coords[2]
        frame["latitude"] = response.Latitude()
        frame["longitude"] = response.Longitude()
        frames.append(frame[METADATA_COLS + var_names])
    return pd.concat(frames, ignore_index=True)


def fetch_weather_singleruns(
    start_date: str,
    end_date: str,
    points: Mapping,
    variables: Sequence[str],
    *,
    cell_selection: Literal["land", "sea", "nearest"] = "land",
    model: str = "ecmwf_ifs",
    errors: Literal["raise", "skip"] = "raise",
    require_complete: bool = True,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch inclusive midnight run dates with a common D+1 delivery window.

    ``points`` maps stable IDs to [longitude, latitude, region] in WGS84,
    matching ``generate_germany_grid``. Pass an explicit, nonempty variable
    sequence using Open-Meteo hourly names. Output consists of METADATA_COLS
    followed by those names, in the supplied order. Wind is m/s, temperature
    is Celsius, and precipitation is mm (API defaults for the latter two).

    Only ECMWF IFS is supported to keep the archive and forecast semantics
    consistent with offshore. Starts before 2024-03-14 are clipped. Dates
    must be timezone-naive midnights; they identify runs, not delivery days.

    By default a failed or incomplete run raises instead of silently creating
    gaps in research data. ``errors="skip"`` warns and continues; inspect the
    returned coverage before saving. ``require_complete=False`` leaves NaNs
    and short grids for a caller with its own validation/recovery policy.
    With ``checkpoint_path``, validated runs are saved every 30 days and on
    exit. A subsequent call resumes missing run dates from that file. HTTP
    429 stops the loop, saves the available runs and returns them. Without a
    checkpoint path, HTTP 429 still stops and returns the in-memory runs.
    """
    if model != "ecmwf_ifs":
        raise NotImplementedError("Only model='ecmwf_ifs' is currently supported.")
    if cell_selection not in {"land", "sea", "nearest"}:
        raise ValueError("cell_selection must be 'land', 'sea', or 'nearest'.")
    if errors not in {"raise", "skip"}:
        raise ValueError("errors must be 'raise' or 'skip'.")
    if not isinstance(points, Mapping) or not points:
        raise ValueError("points must be a nonempty mapping of IDs to [longitude, latitude, region].")
    for pid, coords in points.items():
        try:
            valid = (len(coords) == 3 and np.isfinite(float(coords[0]))
                     and np.isfinite(float(coords[1]))
                     and -180 <= float(coords[0]) <= 180
                     and -90 <= float(coords[1]) <= 90)
        except (TypeError, ValueError, IndexError):
            valid = False
        if not valid:
            raise ValueError(f"Invalid coordinates for point {pid}; expected [longitude, latitude, region].")
    if isinstance(variables, str):
        raise ValueError("variables must be a sequence of names, not a comma-separated string.")
    var_names = list(variables)
    if (not var_names or any(not isinstance(v, str) or not v for v in var_names)
            or len(set(var_names)) != len(var_names) or set(var_names) & set(METADATA_COLS)):
        raise ValueError("variables must contain unique, nonempty weather names, distinct from metadata.")
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    for value in (start, end):
        if pd.isna(value) or value.tzinfo is not None or value != value.normalize():
            raise ValueError("Run dates must be timezone-naive calendar dates (YYYY-MM-DD).")
    if start > end:
        raise ValueError("start_date must be no later than end_date.")
    if start < ECMWF_SINGLE_SINCE:
        warnings.warn(f"Clipping start_date to the first ECMWF Single Run date: {ECMWF_SINGLE_SINCE.date()}.",
                      RuntimeWarning, stacklevel=2)
        start = ECMWF_SINGLE_SINCE
    columns = METADATA_COLS + var_names
    if start > end:
        return pd.DataFrame(columns=columns)

    path = Path(checkpoint_path) if checkpoint_path is not None else None
    parts = []
    completed = set()
    if path is not None and path.exists():
        saved = pd.read_parquet(path)
        if list(saved.columns) != columns:
            raise ValueError(f"Checkpoint columns do not match this request: {path}")
        if not saved.empty:
            if set(saved.point_id) != set(points):
                raise ValueError(f"Checkpoint points do not match this request: {path}")
            saved_dates = pd.to_datetime(saved.datetime_run_utc, utc=True)
            if (saved_dates.min() < start.tz_localize("UTC")
                    or saved_dates.max() > end.tz_localize("UTC")):
                raise ValueError(f"Checkpoint run dates are outside the requested range: {path}")
            completed = set(saved_dates.dt.tz_localize(None).dt.normalize())
            for day in completed:
                run = saved.loc[saved_dates.eq(day.tz_localize("UTC"))]
                expected = _target_hours(day)
                if (len(run) != len(points) * len(expected)
                        or any(not pd.DatetimeIndex(run.loc[run.point_id.eq(pid), "datetime_utc"])
                               .as_unit("ns").sort_values().equals(expected.as_unit("ns"))
                               for pid in points)
                        or (require_complete and not np.isfinite(run[var_names].to_numpy()).all())):
                    raise ValueError(f"Incomplete checkpoint run {day.date()}: {path}")
            parts.append(saved)
            print(f"  Resuming {path}: {len(completed)} runs already saved")
    client = _build_client()
    new_runs = 0
    last_saved_count = 0
    # Multi-location requests consume more of the free API's minute budget.
    # Target at most 400 point-requests/minute, leaving room below its 600
    # calls/minute limit and for other users of the same IP address.
    pace_seconds = max(1.5, len(points) * 0.15)
    print(f"Run range {start.date()} -> {end.date()} (ECMWF IFS HRES)")
    print("  Target: calendar day D+1 in Europe/Berlin, +/-1-hour buffer")
    print(f"  Request pacing: {pace_seconds:g}s after each run for {len(points)} points")
    for day in pd.date_range(start, end, freq="D"):
        if day in completed:
            continue
        try:
            parts.append(_fetch_day(day, points, model, var_names, cell_selection,
                                    client, require_complete))
            new_runs += 1
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(f"  Open-Meteo rate limit at run {day.date()}: {exc}")
                if path is not None:
                    _save_weather_checkpoint(parts, path, columns)
                    last_saved_count = new_runs
                break
            if errors == "raise":
                if path is not None and new_runs:
                    _save_weather_checkpoint(parts, path, columns)
                raise
            warnings.warn(f"Error for run {day.date()} [{model}]: {exc}", RuntimeWarning, stacklevel=2)
        if path is not None and new_runs - last_saved_count >= 30:
            _save_weather_checkpoint(parts, path, columns)
            last_saved_count = new_runs
        # Preserve the scheduled pipeline's request pacing, including failures.
        time.sleep(pace_seconds)
    if path is not None and new_runs > last_saved_count:
        _save_weather_checkpoint(parts, path, columns)
    if not parts:
        return pd.DataFrame(columns=columns)
    result = (pd.concat(parts, ignore_index=True)
              .sort_values(["point_id", "datetime_run_utc", "datetime_utc"])
              .reset_index(drop=True))
    result["datetime_run_utc"] = pd.to_datetime(result.datetime_run_utc, utc=True)
    result["datetime_utc"] = pd.to_datetime(result.datetime_utc, utc=True)
    result["datetime_berlin"] = result.datetime_utc.dt.tz_convert(BERLIN_TZ)
    return result[columns]


def fetch_offshore_weather(
    start_date: str,
    end_date: str,
    points: dict | None = None,
    height: int = 100,
    model: str = "ecmwf_ifs",
) -> pd.DataFrame:
    """Backward-compatible offshore profile with the original OUTPUT_COLS.

    Defaults to the same 14 sea points, 100 m wind, native temperature_2m,
    surface pressure and precipitation. Keeps the historical lapse-rate and
    air-density formulas, units, column order and per-run error recovery.
    The daily pipeline validates full point/hour coverage before using data.
    """
    raw = fetch_weather_singleruns(
        start_date, end_date, SEA_POINTS if points is None else points,
        _vars_ecmwf(height), cell_selection="sea", model=model,
        errors="skip", require_complete=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)
    return _add_derived(raw, height)[OUTPUT_COLS]


def fetch_load_weather(
    start_date: str,
    end_date: str,
    points: Mapping,
    model: str = "ecmwf_ifs",
    *,
    errors: Literal["raise", "skip"] = "raise",
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Land weather for load: temperature_2m (C), wind_speed_10m (m/s),
    shortwave_radiation (W/m2, mean over the preceding hour), cloud_cover (%).

    Supply the load model's own points, e.g. generate_germany_grid(n_points=20).
    Any positive point count is supported; coordinates are never generated
    implicitly. Dates are inclusive run dates, with the same D+1 Berlin
    calendar and lag buffers as offshore. Only complete, finite runs pass.
    """
    return fetch_weather_singleruns(
        start_date, end_date, points, LOAD_VARIABLES, cell_selection="land",
        model=model, errors=errors, checkpoint_path=checkpoint_path,
    )
 
 
