"""Incremental source cache for notebook 6. Missing labels are never interpolated."""
import hashlib
import json
import os
import time
import uuid
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd

from utils.offshore_final_features import (
    add_wind_components, interpolate_run_point, aggregate_weather,
)

TZ = 'Europe/Berlin'
FIRST_NEW_DAY = pd.Timestamp('2026-06-12')


def grid(day, freq='15min'):
    """Build a local calendar day; DST days have 92 or 100 quarter-hours."""
    day = pd.Timestamp(day)
    return pd.date_range(day.tz_localize(TZ),
                         (day + pd.DateOffset(days=1)).tz_localize(TZ),
                         freq=freq, inclusive='left').tz_convert('UTC')


def save_parquet(frame, path):
    """Replace one cache file only after its new version is fully written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        frame.to_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def frame_hash(frame):
    values = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    return hashlib.sha256(values + str(list(frame.columns)).encode()).hexdigest()


def weather_features(raw, run_day):
    """Reject incomplete runs before regional means can hide missing points."""
    from utils.offshore_utils import SEA_POINTS
    if raw.empty:
        raise ValueError('No ECMWF run found for this day')
    run_day = pd.Timestamp(run_day)
    target_day = run_day + pd.DateOffset(days=1)
    hours = grid(target_day, 'h')
    expected = pd.date_range(hours[0] - pd.Timedelta(hours=1),
                             hours[-1] + pd.Timedelta(hours=1), freq='h')
    rename = dict(datetime_run_utc='dt_run_utc', datetime_utc='dt_utc',
                  datetime_berlin='dt_berlin', wind_speed='w_speed',
                  wind_direction='w_dir', temperature_hub='temp',
                  surface_pressure='press', air_density='density')
    f = raw.rename(columns=rename).copy()
    for col in ['dt_utc', 'dt_run_utc']:
        f[col] = pd.to_datetime(f[col], utc=True)
    f = f.drop_duplicates(['dt_run_utc', 'dt_utc', 'point_id'], keep='last')
    if set(f.point_id) != set(SEA_POINTS):
        raise ValueError('The weather run does not contain all 14 points.')
    fields = ['w_speed', 'w_dir', 'temp', 'press', 'density', 'precipitation']
    missing = f[fields].columns[~np.isfinite(f[fields]).all()].tolist()
    if missing:
        raise ValueError('Incomplete ECMWF run: missing values in ' + ', '.join(missing))
    parts = []
    for point, group in f.groupby('point_id'):
        group = group.sort_values('dt_utc')
        if not pd.DatetimeIndex(group.dt_utc).equals(expected):
            raise ValueError(f'Incomplete hourly weather grid for point {point}.')
        if not group.dt_run_utc.eq(run_day.tz_localize('UTC')).all():
            raise ValueError('Unexpected weather run timestamp.')
        if not np.isfinite(group[['w_speed', 'w_dir', 'temp', 'press', 'density', 'precipitation']]).all().all():
            raise ValueError('The weather run contains missing or infinite values.')
        parts.append(interpolate_run_point(add_wind_components(group), method='pchip'))
    result = aggregate_weather(pd.concat(parts, ignore_index=True)).set_index('dt_utc')
    return result.reindex(grid(target_day))


def cached_weather_features(raw, run_day, cache_dir):
    """Reuse exact feature values until the raw run or transformation changes."""
    from utils import offshore_final_features
    normalized = raw.sort_values(['datetime_run_utc', 'datetime_utc', 'point_id']).reset_index(drop=True)
    identity = dict(raw=frame_hash(normalized), run=str(pd.Timestamp(run_day)),
                    code=hashlib.sha256(Path(__file__).read_bytes() +
                        Path(offshore_final_features.__file__).read_bytes()).hexdigest(),
                    versions={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy']})
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / f'{pd.Timestamp(run_day).date()}_{key}.parquet'
    if path.exists():
        return pd.read_parquet(path)
    result = weather_features(normalized, run_day)
    save_parquet(result, path)
    # Both the first and later calls return the exact persisted representation.
    return pd.read_parquet(path)


def refresh_sources(root, end_day=None, update=True, revision_days=7, max_requests=None, max_consecutive_failures=3):
    """Update each source independently, preserving usable cache after an outage.

    max_requests limits weather and generation requests separately for smoke tests.
    None processes every missing day. Recent generation days are requested again.
    Old missing days are retried too; a high watermark alone would lose these gaps.
    """
    root = Path(root)
    cache = root / 'results/offshore_final/source_cache'
    cache.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.now(tz=TZ).tz_localize(None).normalize()
    end_day = today - pd.DateOffset(days=1) if end_day is None else pd.Timestamp(end_day)
    end_day = min(end_day, today - pd.DateOffset(days=1))
    events = []
    generation_path = cache / 'generation.parquet'
    capacity_path = cache / 'capacity.parquet'
    generation = pd.read_parquet(generation_path if generation_path.exists() else
        root / 'data/raw_data/gen_act_per_type_2024-03-13_to_2026-06-11.parquet')[['gen_wind_off_act']]
    generation.index = pd.to_datetime(generation.index, utc=True)
    capacity = pd.read_parquet(capacity_path if capacity_path.exists() else
        root / 'data/raw_data/wind_offshore_installed_capacity_2024-01-01_to_2026-07-01.parquet')
    capacity.dt_utc = pd.to_datetime(capacity.dt_utc, utc=True)
    if update:
        try:
            from utils.get_installed_capacity import get_installed_power
            fresh = get_installed_power('2024-01-01', str(today.date()), sources=['Wind offshore'])
            fresh = fresh.rename(columns={'date': 'dt_utc', 'Wind offshore': 'offshore_capacity'})
            fresh = fresh.loc[np.isfinite(fresh.offshore_capacity) & fresh.offshore_capacity.gt(0)]
            if fresh.empty:
                raise ValueError('No valid installed capacity was returned.')
            capacity = pd.concat([capacity, fresh]).drop_duplicates('dt_utc', keep='last').sort_values('dt_utc')
            save_parquet(capacity, capacity_path)
            events.append(dict(source='capacity', day=str(today.date()), status='updated'))
        except Exception as exc:
            # Do not include request URLs: some APIs put credentials in them.
            events.append(dict(source='capacity', day=str(today.date()), status=type(exc).__name__))
    days = pd.date_range(FIRST_NEW_DAY, end_day, freq='D')
    requests_used = 0
    failures = 0
    client = None
    for day in days:
        existing = generation.reindex(grid(day)).gen_wind_off_act
        complete = np.isfinite(existing).all() and existing.ge(0).all()
        refresh = revision_days > 0 and day >= end_day - pd.Timedelta(days=revision_days - 1)
        if not update or (complete and not refresh):
            continue
        if max_requests is not None and requests_used >= max_requests:
            break
        if failures >= max_consecutive_failures:
            events.append(dict(source='generation', day=str(day.date()), status='paused after repeated errors; retry next run'))
            break
        requests_used += 1
        try:
            from entsoe import EntsoePandasClient
            key = os.getenv('ENTSOE_API_KEY')
            if not key:
                events.append(dict(source='generation', day=str(day.date()), status='missing API key'))
                break
            if client is None:
                client = EntsoePandasClient(api_key=key, timeout=30, retry_count=2, retry_delay=2)
            start = day.tz_localize(TZ)
            fresh = client.query_generation('DE_LU', start=start,
                end=(day + pd.DateOffset(days=1)).tz_localize(TZ), psr_type='B18')
            if isinstance(fresh, pd.DataFrame):
                matches = [c for c in fresh.columns if 'Actual Aggregated' in str(c)]
                if not matches:
                    matches = [c for c in fresh.columns if 'Wind Offshore' in str(c) and 'Consumption' not in str(c)]
                if len(matches) != 1:
                    raise ValueError('Ambiguous offshore generation columns.')
                fresh = fresh[matches[0]]
            fresh.index = pd.to_datetime(fresh.index, utc=True)
            fresh = fresh.loc[~fresh.index.duplicated(keep='last')].reindex(grid(day))
            # A partial response may add valid intervals, but cannot erase old values.
            fresh = fresh.loc[np.isfinite(fresh) & fresh.ge(0)].to_frame('gen_wind_off_act')
            if fresh.empty:
                raise ValueError('No valid generation intervals were returned.')
            generation = pd.concat([generation, fresh])
            generation = generation.loc[~generation.index.duplicated(keep='last')].sort_index()
            save_parquet(generation, generation_path)
            events.append(dict(source='generation', day=str(day.date()), status=f'received {len(fresh)}/{len(grid(day))}'))
            failures = 0
            print(f'Generation {day.date()}: {len(fresh)}/{len(grid(day))} intervals')
        except Exception as exc:
            failures += 1
            events.append(dict(source='generation', day=str(day.date()), status=type(exc).__name__))
        time.sleep(0.3)
    # Existing raw files already contain some runs after the first holdout.
    raw_parts = []
    for path in sorted((root / 'data/raw_data').glob('offshore_ECMWF_singleruns_only_*.parquet')):
        part = pd.read_parquet(path)
        part = part.loc[pd.to_datetime(part.datetime_run_utc, utc=True) >=
                        (FIRST_NEW_DAY - pd.Timedelta(days=1)).tz_localize('UTC')]
        raw_parts.append(part)
    raw = pd.concat(raw_parts, ignore_index=True)
    features = []
    requests_used = 0
    failures = 0
    for day in days:
        run = day - pd.DateOffset(days=1)
        path = cache / 'weather' / f'{run.date()}.parquet'
        cached = pd.read_parquet(path) if path.exists() else raw.loc[
            pd.to_datetime(raw.datetime_run_utc, utc=True).eq(run.tz_localize('UTC'))]
        downloaded = False
        try:
            feature = cached_weather_features(cached, run, cache / 'features')
        except (ValueError, KeyError, AttributeError) as cached_error:
            if not update or failures >= max_consecutive_failures or (max_requests is not None and requests_used >= max_requests):
                events.append(dict(source='weather', day=str(day.date()), status='unavailable',
                    detail=str(cached_error) if isinstance(cached_error, ValueError) else 'Invalid cached weather schema'))
                continue
            requests_used += 1
            try:
                from utils.offshore_utils import fetch_offshore_weather
                cached = fetch_offshore_weather(str(run.date()), str(run.date()))
                # Keep a partial response for diagnosis and retry it next time.
                # This branch is reached only if the old run was unusable.
                if not cached.empty:
                    save_parquet(cached, path)
                feature = cached_weather_features(cached, run, cache / 'features')
                downloaded = True
                events.append(dict(source='weather', day=str(day.date()), status='updated'))
                failures = 0
            except Exception as exc:
                failures += 1
                events.append(dict(source='weather', day=str(day.date()), status=type(exc).__name__,
                    detail=str(exc) if isinstance(exc, ValueError) else 'Weather request failed; no complete run available'))
                if failures >= max_consecutive_failures:
                    events.append(dict(source='weather', day=str(day.date()), status='API paused after repeated errors; cached runs still used'))
                continue
        if downloaded or not path.exists():
            save_parquet(cached, path)
        features.append(feature)
    feature_frame = pd.concat(features).sort_index() if features else pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))
    return feature_frame, generation, capacity, pd.DataFrame(events), end_day


def assemble_dataset(original, features, generation, capacity, end_day, feature_cols, max_capacity_age_days=62, source_events=None):
    """Keep the old benchmark snapshot and attach coverage flags to every new interval."""
    days = pd.date_range(FIRST_NEW_DAY, end_day, freq='D')
    index = pd.DatetimeIndex(np.concatenate([grid(d).values for d in days]), tz='UTC') if len(days) else pd.DatetimeIndex([], tz='UTC')
    new = features.reindex(index).rename_axis('dt_utc')
    new['dt_berlin'] = new.index.tz_convert(TZ)
    cap = capacity.rename(columns={'dt_utc': 'capacity_effective_time', 'offshore_capacity': 'capacity'})
    cap = cap.sort_values('capacity_effective_time')
    new = pd.merge_asof(new.reset_index(), cap, left_on='dt_utc', right_on='capacity_effective_time', direction='backward').set_index('dt_utc')
    new['capacity_age_days'] = (new.index.to_series() - new.capacity_effective_time).dt.total_seconds() / 86400
    new['capacity_available'] = new.capacity.gt(0) & new.capacity_age_days.le(max_capacity_age_days)
    new['generation_available'] = generation.gen_wind_off_act.reindex(index).ge(0) & np.isfinite(generation.gen_wind_off_act.reindex(index))
    new['gen_wind_off_act'] = generation.gen_wind_off_act.reindex(index)
    for col in feature_cols:
        if col not in new:
            new[col] = np.nan
    new['weather_available'] = np.isfinite(new[feature_cols]).all(axis=1)
    new['capacity_factor'] = new.gen_wind_off_act / (new.capacity * 1000)
    new['target_valid'] = new.capacity_factor.between(0, 1)
    new['row_complete'] = new[['weather_available', 'generation_available', 'capacity_available', 'target_valid']].all(axis=1)
    local_days = new.index.tz_convert(TZ).tz_localize(None).normalize()
    coverage = new.groupby(local_days)[['weather_available', 'generation_available', 'capacity_available', 'target_valid', 'row_complete']].all()
    coverage.index.name = 'day'
    coverage['expected_rows'] = [len(grid(d)) for d in coverage.index]
    coverage['complete_rows'] = new.groupby(local_days).row_complete.sum()
    weather_details = {}
    if source_events is not None and not source_events.empty and 'detail' in source_events:
        for event in source_events.loc[source_events.source.eq('weather')].itertuples():
            if isinstance(event.detail, str) and event.detail:
                weather_details[pd.Timestamp(event.day)] = event.detail
    reasons = []
    for day, row in coverage.iterrows():
        reason = []
        if not row.weather_available:
            reason.append(weather_details.get(day, 'No complete ECMWF run available for this day'))
        if not row.generation_available:
            reason.append('Actual generation is missing or incomplete (ENTSO-E)')
        if not row.capacity_available:
            reason.append('Installed capacity is missing, invalid or too old')
        if not row.target_valid and row.generation_available and row.capacity_available:
            reason.append('Capacity factor is outside [0, 1]')
        reasons.append('; '.join(reason))
    coverage['reason'] = reasons
    new['coverage_reason'] = coverage.reason.reindex(local_days).to_numpy()
    new['day_complete'] = coverage.row_complete.reindex(local_days).to_numpy()
    old = original.copy()
    old.index = pd.to_datetime(old.index, utc=True)
    old['snapshot'] = 'notebook_5_frozen'
    new['snapshot'] = 'notebook_6_refreshed'
    for col in ['weather_available', 'generation_available', 'capacity_available', 'target_valid', 'row_complete', 'day_complete']:
        old[col] = True
    dataset = pd.concat([old, new]).sort_index()
    if dataset.index.duplicated().any():
        raise ValueError('The old and new datasets overlap.')
    return dataset, coverage
