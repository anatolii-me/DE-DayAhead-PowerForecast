"""Weather features copied from notebook 1b, with explicit daily coverage checks."""
import numpy as np
import pandas as pd

CONTINUOUS_WEATHER_COLS = ['wind_u', 'wind_v', 'temp', 'press', 'density', 'precipitation']

def add_wind_components(frame):
    """Convert meteorological wind direction into Cartesian vector components."""
    result = frame.copy()
    direction_rad = np.deg2rad(result['w_dir'])

    # Meteorological direction indicates where the wind comes from
    result['wind_u'] = -result['w_speed'] * np.sin(direction_rad)
    result['wind_v'] = -result['w_speed'] * np.cos(direction_rad)
    return result

def restore_wind_state(frame):
    """Restore wind speed and meteorological direction after component interpolation."""
    result = frame.copy()
    reconstructed_speed = np.hypot(result['wind_u'], result['wind_v'])
    reconstructed_direction = (
        np.rad2deg(np.arctan2(-result['wind_u'], -result['wind_v'])) + 360
    ) % 360

    # Existing hourly values are retained exactly; only missing quarter-hours are filled
    result['w_speed'] = result['w_speed'].fillna(reconstructed_speed).clip(lower=0)
    result['w_dir'] = result['w_dir'].fillna(reconstructed_direction) % 360
    result['precipitation'] = result['precipitation'].clip(lower=0)
    return result

def interpolate_run_point(group, method='time', frequency='15min'):
    """Interpolate one measurement point inside one forecast run without crossing gaps."""
    group = group.sort_values('dt_utc').drop_duplicates('dt_utc', keep='last')
    group = group.set_index('dt_utc')

    full_index = pd.date_range(
        start=group.index.min(),
        end=group.index.max(),
        freq=frequency,
        tz=group.index.tz,
    )
    result = group.reindex(full_index)
    result.index.name = 'dt_utc'

    is_original = result.index.isin(group.index)
    source_timestamp = pd.Series(pd.NaT, index=full_index, dtype=full_index.dtype)
    source_timestamp.loc[is_original] = full_index[is_original]

    previous_source = source_timestamp.ffill()
    next_source = source_timestamp.bfill()

    # Only fill quarter-hours between genuinely consecutive hourly source rows
    valid_interval = is_original | (
        (next_source - previous_source) == pd.Timedelta(hours=1)
    )

    interpolated_values = result[CONTINUOUS_WEATHER_COLS].interpolate(
        method=method,
        limit_area='inside',
    )
    result[CONTINUOUS_WEATHER_COLS] = interpolated_values.where(valid_interval)

    # These columns are constant inside one point and forecast run
    result['dt_run_utc'] = group['dt_run_utc'].iloc[0]
    result['point_id'] = group['point_id'].iloc[0]
    result['region'] = group['region'].iloc[0]
    result['dt_berlin'] = result.index.tz_convert('Europe/Berlin')
    result['is_interpolated'] = (~is_original).astype('int8')

    result = restore_wind_state(result)
    return result.reset_index()

def add_time_features(frame):
    """Add forecast lead time and local cyclical calendar features."""
    result = frame.copy()

    result['lead_time'] = (
        result['dt_utc'] - result['dt_run_utc']
    ).dt.total_seconds() / 3600.0

    time_of_day_hours = (
        result['dt_berlin'].dt.hour
        + result['dt_berlin'].dt.minute / 60.0
    )
    result['hour_sin'] = np.sin(2 * np.pi * time_of_day_hours / 24)
    result['hour_cos'] = np.cos(2 * np.pi * time_of_day_hours / 24)

    day_of_year = result['dt_berlin'].dt.dayofyear
    result['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365.25)
    result['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365.25)
    return result

def aggregate_weather(interpolated_weather):
    # Setting thresholds for various risk conditions
    CUTIN_THRESHOLD = 3
    CUTOUT_THRESHOLD = 25
    SATURATION_THRESHOLD = 12
    ICING_TEMP_UP_THRESHOLD = 2
    ICING_TEMP_DOWN_THRESHOLD = -10
    
    df = interpolated_weather.copy()
    
    # Recalculating physical proxy and circular wind-direction components
    df['power_proxy'] = (df['w_speed'] ** 3) * df['density']
    df['w_dir_rad'] = np.deg2rad(df['w_dir'])
    df['w_dir_sin'] = np.sin(df['w_dir_rad'])
    df['w_dir_cos'] = np.cos(df['w_dir_rad'])
    
    # Recalculating binary conditions after interpolation
    df['cut_in_risk'] = (df['w_speed'] < CUTIN_THRESHOLD).astype('int8')
    df['cut_out_risk'] = (df['w_speed'] >= CUTOUT_THRESHOLD).astype('int8')
    
    saturation_condition = (
        (df['w_speed'] >= SATURATION_THRESHOLD)
        & (df['w_speed'] < CUTOUT_THRESHOLD)
    )
    df['saturation_risk'] = saturation_condition.astype('int8')
    
    icing_condition = (
        (df['precipitation'] > 0)
        & (df['temp'] <= ICING_TEMP_UP_THRESHOLD)
        & (df['temp'] >= ICING_TEMP_DOWN_THRESHOLD)
    )
    df['icing_risk'] = icing_condition.astype('int8')
    
    # ---------------------------------------------------------
    # Pivoting raw power_proxy values by 14 measurement points
    # ---------------------------------------------------------
    POINT_COL = 'point_id'
    time_cols = ['dt_run_utc', 'dt_utc', 'dt_berlin']
    
    proxy_wide = df.pivot_table(
        index=time_cols,
        columns=POINT_COL,
        values='power_proxy',
        aggfunc='first',
    )
    
    proxy_wide.columns = [
        f"point_{str(col).replace(' ', '_')}_power_proxy"
        for col in proxy_wide.columns
    ]
    proxy_wide = proxy_wide.reset_index()
    
    # ---------------------------------------------------------
    # Aggregation of other features by regions
    # ---------------------------------------------------------
    agg_funcs = {
        'w_dir_sin': ['mean'],
        'w_dir_cos': ['mean'],
        'temp': ['mean'],
        'press': ['std'],
        'precipitation': ['mean'],
        'cut_in_risk': ['mean'],
        'cut_out_risk': ['mean'],
        'saturation_risk': ['mean'],
        'icing_risk': ['mean'],
    }
    
    grouped = (
        df.groupby(time_cols + ['region'])
        .agg(agg_funcs)
    )
    
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    
    # Circular variance measures disagreement of point-wise wind directions
    grouped['w_dir_circ_var'] = 1 - np.sqrt(
        grouped['w_dir_sin_mean'] ** 2 + grouped['w_dir_cos_mean'] ** 2
    )
    
    region_pivot = grouped.pivot(index=time_cols, columns='region')
    region_pivot.columns = [
        f"{col[1].replace(' ', '_')}_{col[0]}"
        for col in region_pivot.columns
    ]
    region_pivot = region_pivot.reset_index()
    
    # A final row is original only if all contributing point rows are original
    provenance = (
        df.groupby(time_cols, as_index=False)['is_interpolated']
        .max()
    )
    
    # Combining point-wise proxies, regional features, and provenance
    feature_table = (
        proxy_wide
        .merge(region_pivot, on=time_cols, how='inner')
        .merge(provenance, on=time_cols, how='inner')
        .sort_values(['dt_run_utc', 'dt_utc'])
        .reset_index(drop=True)
    )
    
    # Cleaning column names for regions
    feature_table.columns = feature_table.columns.str.replace(
        r'^Baltic_Sea_', 'baltic_', regex=True
    )
    feature_table.columns = feature_table.columns.str.replace(
        r'^North_Sea_', 'north_', regex=True
    )
    
    return add_time_features(feature_table)
    

