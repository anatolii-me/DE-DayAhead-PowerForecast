"""Offline request, schema, DST and package entry-point regressions."""
import contextlib
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from openmeteo_requests import OpenMeteoRequestsError

from utils import weather
from utils.offshore.final_data import fetch_weather_with_fallback, weather_features

ROOT = Path(__file__).resolve().parents[1]


def response(times, values, latitude=54.8, longitude=6.6):
    """Exercise the actual FlatBuffer parser through the SDK's public surface."""
    hourly = Mock()
    hourly.Time.return_value = int(times[0].timestamp())
    hourly.TimeEnd.return_value = int(times[-1].timestamp()) + 3600
    hourly.Interval.return_value = 3600
    hourly.Variables.side_effect = lambda i: Mock(ValuesAsNumpy=lambda: values[i])
    return Mock(Hourly=lambda: hourly, Latitude=lambda: latitude, Longitude=lambda: longitude)


def fake_api(url, params, timeout):
    times = pd.date_range(pd.Timestamp(params['run'], tz='UTC'), periods=72, freq='h')
    # Float32 matches the API. Distinct point/variable/hour values expose swaps.
    return [response(times, [np.arange(72, dtype='float32') + 100 * v + p
                            for v in range(len(params['hourly']))], lat + .01, lon + .01)
            for p, (lat, lon) in enumerate(zip(params['latitude'], params['longitude']))]


class WeatherTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock(weather_api=Mock(side_effect=fake_api))
        self.addCleanup(patch.stopall)
        patch.object(weather, '_build_client', return_value=self.client).start()
        patch.object(weather.time, 'sleep').start()
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_offshore_contract_and_dst(self):
        for day, first, last, count in [
            ('2025-01-01', '2025-01-01T22:00Z', '2025-01-03T00:00Z', 26),
            ('2025-03-29', '2025-03-29T22:00Z', '2025-03-30T23:00Z', 25),
            ('2025-10-25', '2025-10-25T21:00Z', '2025-10-27T00:00Z', 27),
        ]:
            with self.subTest(day=day):
                actual = weather.fetch_offshore_weather(day, day)
                self.assertEqual(list(actual), weather.OUTPUT_COLS)
                self.assertEqual(len(actual), 14 * count)
                group = actual.loc[actual.point_id.eq(1001)]
                self.assertEqual(group.datetime_utc.iloc[0], pd.Timestamp(first))
                self.assertEqual(group.datetime_utc.iloc[-1], pd.Timestamp(last) - pd.Timedelta(hours=1))
                self.assertTrue(group.datetime_run_utc.eq(pd.Timestamp(day, tz='UTC')).all())
                self.assertEqual(str(actual.datetime_berlin.dt.tz), 'Europe/Berlin')
                params = self.client.weather_api.call_args.kwargs['params']
                self.assertEqual(params['cell_selection'], 'sea')
                self.assertEqual(params['models'], 'ecmwf_ifs')
                self.assertEqual(params['wind_speed_unit'], 'ms')
                self.assertEqual(params['hourly'], ['wind_speed_100m', 'wind_direction_100m',
                                                   'temperature_2m', 'surface_pressure', 'precipitation'])
                self.assertEqual(params['latitude'][0], weather.SEA_POINTS[1001][1])
                self.assertTrue(actual.height_m.eq(100).all())
                self.assertEqual(actual.wind_speed.dtype, np.dtype('float32'))
                # Offshore still passes the complete-run guard used by daily publication.
                features = weather_features(actual, day)
                self.assertEqual(len(features), (count - 2) * 4)

    def test_load_twenty_points_and_parquet_roundtrip(self):
        points = {i: [7 + i * .1, 50 + i * .1, 'load'] for i in range(1, 21)}
        actual = weather.fetch_load_weather('2025-05-01', '2025-05-01', points)
        self.assertEqual(list(actual), weather.LOAD_OUTPUT_COLS)
        self.assertEqual(len(actual), 20 * 26)
        params = self.client.weather_api.call_args.kwargs['params']
        self.assertEqual(params['cell_selection'], 'land')
        self.assertEqual(params['hourly'], list(weather.LOAD_VARIABLES))
        weather.time.sleep.assert_called_once_with(3.0)
        row = actual.iloc[0]
        self.assertEqual(row.temperature_2m, 21)  # first hour is D 21:00 UTC in summer
        self.assertEqual(row.wind_speed_10m, 121)
        self.assertEqual(row.shortwave_radiation, 221)
        self.assertEqual(row.cloud_cover, 321)
        self.assertAlmostEqual(row.longitude, 7.11)
        self.assertAlmostEqual(row.latitude, 50.11)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weather.parquet'
            actual.to_parquet(path)
            restored = pd.read_parquet(path)
            # Parquet stores second-resolution timestamps as milliseconds;
            # check timezone and instants before aligning only that resolution.
            for col in ['datetime_run_utc', 'datetime_utc', 'datetime_berlin']:
                self.assertEqual(str(actual[col].dt.tz), str(restored[col].dt.tz))
                self.assertTrue(actual[col].eq(restored[col]).all())
                restored[col] = restored[col].astype(actual[col].dtype)
            pd.testing.assert_frame_equal(actual, restored, check_exact=True)

    def test_custom_profile_and_overlapping_buffers(self):
        actual = weather.fetch_weather_singleruns('2025-05-01', '2025-05-02',
            {1: [8, 51, 'solar']}, ['shortwave_radiation', 'temperature_2m'])
        self.assertEqual(list(actual), weather.METADATA_COLS + ['shortwave_radiation', 'temperature_2m'])
        self.assertTrue(actual.duplicated(['point_id', 'datetime_utc']).any())
        self.assertFalse(actual.duplicated(['point_id', 'datetime_run_utc', 'datetime_utc']).any())

    def test_offshore_physical_values_match_legacy_reference(self):
        # Reference values captured from the pre-refactor implementation with
        # float32 API data. Exact checks protect rounding and density precision.
        times = pd.date_range('2025-05-01T21:00Z', periods=26, freq='h')
        values = [np.resize(np.array(pair, dtype='float32'), 26)
                  for pair in [(7.5, 12.25), (325, 100), (10, -5), (1000, 985), (.5, 0)]]
        self.client.weather_api.side_effect = lambda *a, **k: [response(times, values)]
        for height, temperatures, densities in [
            (10, [9.93499755859375, -5.06500244140625], [1.2306246323749268, 1.2799888244630555]),
            (100, [9.350006103515625, -5.649993896484375], [1.233172969465096, 1.2827880127692757]),
        ]:
            actual = weather.fetch_offshore_weather('2025-05-01', '2025-05-01',
                points={1: [6.6, 54.8, 'North Sea']}, height=height)
            self.assertEqual(actual.air_density.dtype, np.dtype('float64'))
            self.assertEqual(actual.temperature_hub.dtype, np.dtype('float32'))
            np.testing.assert_array_equal(actual.temperature_hub, np.resize(temperatures, 26))
            np.testing.assert_array_equal(actual.air_density, np.resize(densities, 26))
            np.testing.assert_array_equal(actual.wind_speed, values[0])
            np.testing.assert_array_equal(actual.wind_direction, values[1])
            np.testing.assert_array_equal(actual.precipitation, values[4])

    def test_failed_runs_keep_offshore_recovery_and_load_raises(self):
        self.client.weather_api.side_effect = TimeoutError('unavailable')
        with self.assertWarns(RuntimeWarning):
            actual = weather.fetch_offshore_weather('2025-05-01', '2025-05-01')
        self.assertTrue(actual.empty)
        self.assertEqual(list(actual), weather.OUTPUT_COLS)
        with self.assertRaises(TimeoutError):
            weather.fetch_load_weather('2025-05-01', '2025-05-01', {1: [8, 51, 'load']})

    def test_skip_retains_successful_dates(self):
        def sometimes(url, params, timeout):
            if params['run'].startswith('2025-05-01'):
                raise TimeoutError('unavailable')
            return fake_api(url, params, timeout)
        self.client.weather_api.side_effect = sometimes
        with self.assertWarns(RuntimeWarning):
            actual = weather.fetch_load_weather('2025-05-01', '2025-05-02',
                {1: [8, 51, 'load']}, errors='skip')
        self.assertEqual(len(actual), 26)
        self.assertTrue(actual.datetime_run_utc.eq(pd.Timestamp('2025-05-02', tz='UTC')).all())

    def test_rate_limit_saves_partial_run_and_resumes(self):
        points = {1: [8, 51, 'load']}
        def limited(url, params, timeout):
            if params['run'].startswith('2025-05-02'):
                try:
                    raise OpenMeteoRequestsError({
                        'error': True,
                        'reason': 'Hourly API request limit exceeded. Please try again in the next hour.',
                    })
                except OpenMeteoRequestsError as cause:
                    raise OpenMeteoRequestsError(f'failed to request: {cause}') from cause
            return fake_api(url, params, timeout)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weather.parquet'
            self.client.weather_api.side_effect = limited
            partial = weather.fetch_load_weather('2025-05-01', '2025-05-03', points,
                                                 checkpoint_path=path)
            self.assertEqual(len(partial), 26)
            pd.testing.assert_frame_equal(partial, pd.read_parquet(path), check_dtype=False)
            self.assertEqual(self.client.weather_api.call_count, 2)

            self.client.weather_api.reset_mock(side_effect=True)
            self.client.weather_api.side_effect = fake_api
            complete = weather.fetch_load_weather('2025-05-01', '2025-05-03', points,
                                                  checkpoint_path=path)
            self.assertEqual(len(complete), 3 * 26)
            self.assertEqual(self.client.weather_api.call_count, 2)
            self.assertEqual(self.client.weather_api.call_args_list[0].kwargs['params']['run'],
                             '2025-05-02T00:00')
            pd.testing.assert_frame_equal(complete, pd.read_parquet(path), check_dtype=False)

    def test_daily_limit_before_first_run_saves_empty_checkpoint(self):
        self.client.weather_api.side_effect = OpenMeteoRequestsError({
            'error': True, 'reason': 'Daily API request limit exceeded. Please try again tomorrow.',
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'weather.parquet'
            result = weather.fetch_load_weather('2025-05-01', '2025-05-02',
                                                {1: [8, 51, 'load']}, checkpoint_path=path)
            self.assertTrue(result.empty)
            self.assertTrue(pd.read_parquet(path).empty)
            self.assertEqual(self.client.weather_api.call_count, 1)

    def test_short_point_response_triggers_existing_split_retry(self):
        def incomplete(url, params, timeout):
            result = fake_api(url, params, timeout)
            return result[:-1] if len(result) == 14 else result
        self.client.weather_api.side_effect = incomplete
        with self.assertWarns(RuntimeWarning):
            repaired = fetch_weather_with_fallback('2025-05-01')
        self.assertEqual(len(repaired), 14 * 26)
        self.assertEqual([len(c.kwargs['params']['latitude'])
                          for c in self.client.weather_api.call_args_list], [14, 7, 7])

    def test_load_rejects_missing_points_hours_and_values(self):
        points = {1: [8, 51, 'load']}
        self.client.weather_api.side_effect = lambda *a, **k: []
        with self.assertRaisesRegex(ValueError, 'point responses'):
            weather.fetch_load_weather('2025-05-01', '2025-05-01', points)
        times = pd.date_range('2025-05-01T21:00Z', periods=25, freq='h')
        self.client.weather_api.side_effect = lambda *a, **k: [response(times, [np.ones(25)] * 4)]
        with self.assertRaisesRegex(ValueError, 'hourly weather grid'):
            weather.fetch_load_weather('2025-05-01', '2025-05-01', points)
        times = pd.date_range('2025-05-01T21:00Z', periods=26, freq='h')
        self.client.weather_api.side_effect = lambda *a, **k: [response(times, [np.full(26, np.nan)] * 4)]
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            weather.fetch_load_weather('2025-05-01', '2025-05-01', points)

    def test_validation_and_archive_boundary(self):
        base = dict(start_date='2025-05-01', end_date='2025-05-01',
                    points={1: [8, 51, 'load']}, variables=['temperature_2m'])
        for changes in [dict(points={}), dict(points={1: [np.nan, 51, 'load']}),
                        dict(points={1: [8, 100, 'load']}), dict(variables=[]),
                        dict(variables=['temperature_2m'] * 2), dict(variables='temperature_2m'),
                        dict(variables=['point_id']), dict(cell_selection='water'),
                        dict(errors='ignore'), dict(start_date='2025-05-02'),
                        dict(start_date='2025-05-01T01:00'), dict(start_date='2025-05-01T00:00Z')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                weather.fetch_weather_singleruns(**(base | changes))
        self.client.weather_api.assert_not_called()
        with self.assertWarns(RuntimeWarning):
            actual = weather.fetch_weather_singleruns(**(base | dict(start_date='2024-03-01', end_date='2024-03-13')))
        self.assertTrue(actual.empty)
        self.client.weather_api.assert_not_called()
        with self.assertWarns(RuntimeWarning):
            actual = weather.fetch_weather_singleruns(**(base | dict(start_date='2024-03-13', end_date='2024-03-14')))
        self.assertEqual(len(actual), 26)
        self.assertEqual(self.client.weather_api.call_args.kwargs['params']['run'], '2024-03-14T00:00')


class PackageTests(unittest.TestCase):
    def test_http_429_is_not_retried_when_retry_after_is_present(self):
        client = weather._build_client()
        policy = client._session.get_adapter('https://').max_retries
        self.assertFalse(policy.is_retry('GET', 429, has_retry_after=True))
        self.assertTrue(policy.is_retry('GET', 500))

    def test_package_modules_import_and_reload(self):
        names = ['cf_diagnostics', 'daily', 'dashboard', 'day_ahead', 'final_baselines',
                 'final_data', 'final_evaluation', 'final_features', 'final_plot', 'simulation']
        for name in names:
            module = importlib.import_module('utils.offshore.' + name)
            self.assertIs(importlib.reload(module), module)
        with patch('utils.weather.fetch_offshore_weather') as mocked:
            self.assertIs(weather.fetch_offshore_weather, mocked)

    def test_daily_cli_defaults_to_project_root(self):
        from utils.offshore import daily
        with patch.object(sys, 'argv', ['utils.offshore.daily', '--offline']), \
             patch.object(daily, 'run') as run, patch('dotenv.load_dotenv'):
            daily.main()
            run.assert_called_once_with(ROOT, update=False, backtest=False)

    def test_all_cli_entrypoints_accept_help(self):
        for suffix in ['daily', 'day_ahead', 'simulation']:
            module = 'utils.offshore.' + suffix
            result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=ROOT,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)
            self.assertNotIn('RuntimeWarning', result.stderr)


if __name__ == '__main__':
    unittest.main()
