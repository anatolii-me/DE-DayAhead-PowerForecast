"""Execute load preparation and its opt-in weather export without network calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from utils import weather

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/research_load_model/1_data_preparation.ipynb'


class LoadNotebookTests(unittest.TestCase):
    def test_calendar_grid_weather_and_saved_files(self):
        def api(url, params, timeout):
            self.assertEqual(params['run'], '2025-03-29T00:00')
            self.assertEqual(params['cell_selection'], 'land')
            self.assertEqual(params['hourly'], list(weather.LOAD_VARIABLES))
            times = pd.date_range('2025-03-29T00:00Z', periods=72, freq='h')
            values = [np.full(72, value, dtype='float32') for value in (12, 5, 300, 40)]
            hourly = Mock()
            hourly.Time.return_value = int(times[0].timestamp())
            hourly.TimeEnd.return_value = int(times[-1].timestamp()) + 3600
            hourly.Interval.return_value = 3600
            hourly.Variables.side_effect = lambda i: Mock(ValuesAsNumpy=lambda: values[i])
            return [Mock(Hourly=lambda: hourly, Latitude=Mock(return_value=lat),
                         Longitude=Mock(return_value=lon))
                    for lat, lon in zip(params['latitude'], params['longitude'])]

        cells = json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
        namespace = {'__file__': str(NOTEBOOK), '__name__': '__notebook_test__'}
        client = Mock(weather_api=Mock(side_effect=api))
        with tempfile.TemporaryDirectory() as directory, \
             contextlib.redirect_stdout(io.StringIO()), \
             patch('dotenv.load_dotenv', return_value=False), \
             patch.object(weather, '_build_client', return_value=client), \
             patch.object(weather.time, 'sleep'), \
             patch('socket.create_connection', side_effect=AssertionError('Unexpected network access')):
            for index, cell in enumerate(cells):
                if cell['cell_type'] != 'code':
                    continue
                source = ''.join(cell['source'])
                if cell.get('id') == 'load-weather-fetch':
                    namespace.update(START='2025-03-30', END='2025-03-30', RAW_DATA_DIR=Path(directory))
                    source = source.replace('FETCH_LOAD_WEATHER = False', 'FETCH_LOAD_WEATHER = True')
                exec(compile(source, f'{NOTEBOOK}:cell{index}', 'exec'), namespace)

            self.assertIn('weighted_holiday', namespace['df'])
            self.assertTrue(namespace['df'].weighted_holiday.between(0, 1 + 1e-12).all())
            points = namespace['load_points']
            self.assertEqual(len(points), 20)
            self.assertTrue(all(len(coords) == 3 for coords in points.values()))
            actual = namespace['load_weather']
            self.assertEqual(list(actual), weather.LOAD_OUTPUT_COLS)
            self.assertEqual(len(actual), 20 * 25)  # 23-hour DST delivery day plus buffers
            self.assertEqual(set(actual.point_id), set(points))
            self.assertTrue(actual.temperature_2m.eq(12).all())
            self.assertTrue(actual.shortwave_radiation.eq(300).all())
            client.weather_api.assert_called_once()
            saved = pd.read_parquet(Path(directory) / 'load_ECMWF_singleruns_2025-03-29_to_2025-03-29.parquet')
            for col in ['datetime_run_utc', 'datetime_utc', 'datetime_berlin']:
                self.assertEqual(str(saved[col].dt.tz), str(actual[col].dt.tz))
                saved[col] = saved[col].astype(actual[col].dtype)
            pd.testing.assert_frame_equal(actual, saved, check_exact=True)
            coordinates = pd.read_json(Path(directory) / 'load_ECMWF_singleruns_2025-03-29_to_2025-03-29_points.json', orient='table')
            self.assertEqual(coordinates.index.tolist(), list(points))
            np.testing.assert_allclose(coordinates[['longitude', 'latitude']],
                                       [coords[:2] for coords in points.values()])


if __name__ == '__main__':
    unittest.main()
