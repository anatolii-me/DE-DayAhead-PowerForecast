"""Offline regression checks for the final holdout pipeline."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils import offshore_final_data as data
from utils import offshore_final_evaluation as evaluation

ROOT = Path(__file__).resolve().parents[1]
SELECTION_DIR = ROOT / 'results/offshore_joint_loss_hpo/c550de332f450772dce8/selection_d2085702a714cb374154'


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = pd.read_parquet(ROOT / 'data/datasets/offshore_dataset_15m_interpolated_14_points_proxy_2024-03-14_to_2026-06-11.parquet')
        cls.selection = json.loads((SELECTION_DIR / 'frozen_selection.json').read_text())
        cls.features = cls.selection['fit_protocol']['features']
        cls.raw = pd.read_parquet(ROOT / 'data/raw_data/offshore_ECMWF_singleruns_only_2025-01-01_to_2026-01-01.parquet')

    def test_dst_grids(self):
        self.assertEqual(len(data.grid('2025-03-30')), 92)
        self.assertEqual(len(data.grid('2025-10-26')), 100)

    def test_feature_parity_and_missing_point(self):
        for date in ['2025-03-29', '2025-05-01']:
            run = pd.Timestamp(date, tz='UTC')
            raw = self.raw.loc[self.raw.datetime_run_utc.eq(run)]
            actual = data.weather_features(raw, date)
            expected = self.original.copy()
            expected.index = pd.to_datetime(expected.index, utc=True)
            np.testing.assert_allclose(actual[self.features], expected.reindex(actual.index)[self.features], rtol=1e-7, atol=1e-7)
            with self.assertRaises(ValueError):
                data.weather_features(raw.loc[raw.point_id.ne(1001)], date)

    def test_missing_generation_and_stale_capacity(self):
        index = data.grid('2026-06-12')
        features = pd.DataFrame(1.0, index=index, columns=self.features)
        features['dt_run_utc'] = pd.Timestamp('2026-06-11', tz='UTC')
        generation = pd.DataFrame({'gen_wind_off_act': 1000.0}, index=index[:-1])
        capacity = pd.DataFrame({'dt_utc': [pd.Timestamp('2026-06-01', tz='UTC')], 'offshore_capacity': [10.0]})
        dataset, coverage = data.assemble_dataset(self.original, features, generation, capacity,
            pd.Timestamp('2026-06-12'), self.features)
        self.assertFalse(coverage.row_complete.iloc[0])
        self.assertEqual(coverage.complete_rows.iloc[0], 95)
        self.assertTrue(pd.isna(dataset.gen_wind_off_act.iloc[-1]))
        _, stale = data.assemble_dataset(self.original, features, generation, capacity,
            pd.Timestamp('2026-06-12'), self.features, max_capacity_age_days=2)
        self.assertFalse(stale.capacity_available.iloc[0])

    def test_cached_metrics_match_original(self):
        previous = evaluation.load_previous_predictions(SELECTION_DIR, self.selection)
        table, _ = evaluation.comparison_table(previous, pd.DataFrame())
        expected = pd.read_csv(SELECTION_DIR / 'holdout/model_family_comparison.csv').set_index('model')
        actual = table.loc[table.period.eq('first_holdout')].set_index('model')
        np.testing.assert_allclose(actual.mae_mean.sort_index(), expected.mae_mean.sort_index(), rtol=1e-12)

    def test_notebook4_baselines_cover_matching_historical_periods(self):
        from utils import offshore_final_baselines as baselines
        baseline_selection = baselines.baseline_selection(self.selection)
        previous = baselines.load_previous_predictions(SELECTION_DIR, self.selection)
        self.assertEqual(set(previous.model), {'CatBoost', 'LightGBM', 'XGBoost'})
        self.assertEqual(set(previous.loss), {'l2'})
        self.assertEqual(set(previous.period), {'development', 'first_holdout'})
        counts = previous.groupby(['period', 'model']).day.nunique()
        self.assertTrue(counts.groupby(level='period').nunique().eq(1).all())
        self.assertEqual(previous.loc[previous.period.eq('development'), 'day'].nunique(), 425)
        self.assertEqual(previous.loc[previous.period.eq('first_holdout'), 'day'].nunique(), 28)
        self.assertTrue(all(config['weighting'] == 'uniform'
                            for config in baseline_selection['best_by_model'].values()))

    def test_cache_reuse_and_label_correction(self):
        dataset = self.original.copy()
        dataset.index = pd.to_datetime(dataset.index, utc=True)
        dataset['row_complete'] = True
        selection = dict(self.selection)
        selection['best_by_model'] = {'LightGBM': self.selection['best_by_model']['LightGBM']}
        with tempfile.TemporaryDirectory() as cache:
            first, audit = evaluation.evaluate_new_days(dataset, selection, cache, ['2026-06-11'])
            self.assertFalse(audit.cache_hit.iloc[0])
            second, audit = evaluation.evaluate_new_days(dataset, selection, cache, ['2026-06-11'])
            self.assertTrue(audit.cache_hit.iloc[0])
            np.testing.assert_array_equal(first.prediction, second.prediction)
            dataset.loc[data.grid('2026-06-11')[0], 'capacity_factor'] += 0.001
            third, audit = evaluation.evaluate_new_days(dataset, selection, cache, ['2026-06-11'])
            self.assertTrue(audit.cache_hit.iloc[0])
            self.assertNotEqual(second.actual.iloc[0], third.actual.iloc[0])
            dataset.loc[data.grid('2026-06-01')[0], 'capacity_factor'] += 0.001
            _, audit = evaluation.evaluate_new_days(dataset, selection, cache, ['2026-06-11'])
            self.assertFalse(audit.cache_hit.iloc[0])

    def test_training_gap_skips_all_models(self):
        dataset = self.original.copy()
        dataset.index = pd.to_datetime(dataset.index, utc=True)
        dataset['row_complete'] = True
        dataset.loc[data.grid('2026-06-01')[0], 'row_complete'] = False
        with tempfile.TemporaryDirectory() as cache:
            predictions, audit = evaluation.evaluate_new_days(dataset, self.selection, cache, ['2026-06-11'], max_missing_train_days=0)
        self.assertTrue(predictions.empty)
        self.assertEqual(audit.model.iloc[0], 'ALL')

    def test_training_gap_drops_whole_day_within_budget(self):
        dataset = self.original.copy()
        dataset.index = pd.to_datetime(dataset.index, utc=True)
        dataset['row_complete'] = True
        dataset.loc[data.grid('2026-06-01')[0], 'row_complete'] = False
        train, valid, audit = evaluation.prepare_day(dataset, '2026-06-11', 3)
        self.assertEqual(audit['status'], 'ready')
        self.assertEqual(audit['train_days'], 364)
        self.assertEqual(audit['dropped_train_dates'], '2026-06-01')
        self.assertTrue(train.index.intersection(data.grid('2026-06-01')).empty)
        self.assertLess(train.index.max(), pd.Timestamp('2026-06-10', tz=data.TZ))
        for day in ['2026-06-02', '2026-06-03', '2026-06-04']:
            dataset.loc[data.grid(day)[0], 'row_complete'] = False
        _, _, audit = evaluation.prepare_day(dataset, '2026-06-11', 3)
        self.assertEqual(audit['status'], 'skipped_training')
        self.assertIn('allowed 3', audit['reason'])

    def test_feature_cache_roundtrip_and_invalidation(self):
        raw = self.raw.loc[self.raw.datetime_run_utc.eq(pd.Timestamp('2025-05-01', tz='UTC'))].copy()
        with tempfile.TemporaryDirectory() as temporary:
            first = data.cached_weather_features(raw, '2025-05-01', temporary)
            data.save_parquet(raw, Path(temporary) / 'raw.parquet')
            with patch.object(data, 'weather_features', side_effect=AssertionError('Unexpected rebuild')):
                second = data.cached_weather_features(pd.read_parquet(Path(temporary) / 'raw.parquet'), '2025-05-01', temporary)
            pd.testing.assert_frame_equal(first, second, check_exact=True)
            raw.loc[raw.index[0], 'wind_speed'] += 0.1
            third = data.cached_weather_features(raw, '2025-05-01', temporary)
            self.assertFalse(first.equals(third))

    def test_missing_weather_reason_names_fields(self):
        raw = self.raw.loc[self.raw.datetime_run_utc.eq(pd.Timestamp('2025-05-01', tz='UTC'))].copy()
        raw['temperature_hub'] = np.nan
        with self.assertRaisesRegex(ValueError, 'missing values in temp'):
            data.weather_features(raw, '2025-05-01')

    def test_chart_gaps_include_trailing_days_and_multiple_reasons(self):
        from utils.offshore_final_plot import prediction_chart
        previous = evaluation.load_previous_predictions(SELECTION_DIR, self.selection)
        dataset = self.original.copy()
        dataset.index = pd.to_datetime(dataset.index, utc=True)
        for col in ['generation_available', 'capacity_available', 'target_valid']:
            dataset[col] = True
        coverage = pd.DataFrame({'reason': ['Incomplete ECMWF run: missing values in temp',
            'Actual generation is missing or incomplete (ENTSO-E); Installed capacity is missing, invalid or too old', '']},
            index=pd.date_range('2026-06-12', periods=3))
        audit = pd.DataFrame([dict(day='2026-06-14', status='skipped_training', reason='Training history has 4 incomplete days; allowed 3')])
        figure, gaps = prediction_chart(previous, dataset, coverage, audit, self.selection)
        self.assertEqual(len(gaps), 3)
        self.assertIn('Training history', gaps.reason.iloc[2])
        rectangles = [s for s in figure.layout.shapes if s.type == 'rect']
        self.assertEqual(len(rectangles), 3)
        self.assertEqual(pd.Timestamp(rectangles[-1].x1), pd.Timestamp('2026-06-15', tz=data.TZ).tz_convert('UTC'))
        self.assertTrue(any('ECMWF' in a.text for a in figure.layout.annotations))
        self.assertTrue(any(a.text == 'Missing generation<br>Missing/stale capacity'
                            for a in figure.layout.annotations))
        self.assertEqual(gaps.label.iloc[0], 'Incomplete ECMWF data')
        self.assertEqual(gaps.label.iloc[2], 'Insufficient training history')

    def test_short_gap_labels(self):
        from utils.offshore_final_plot import short_gap_label
        self.assertEqual(short_gap_label('No ECMWF run found for this day'), 'No ECMWF run')
        self.assertEqual(short_gap_label('New source diagnostic'), 'New source diagnostic')
        self.assertEqual(short_gap_label('Not evaluated in this run (disabled or day limit)'),
                         'Evaluation not run')

    def test_source_outage_preserves_seed_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / 'data/raw_data'
            raw_dir.mkdir(parents=True)
            generation = pd.DataFrame({'gen_wind_off_act': [1000.0]}, index=[pd.Timestamp('2026-06-11', tz='UTC')])
            data.save_parquet(generation, raw_dir / 'gen_act_per_type_2024-03-13_to_2026-06-11.parquet')
            capacity = pd.DataFrame({'dt_utc': [pd.Timestamp('2026-06-01', tz='UTC')], 'offshore_capacity': [10.0]})
            data.save_parquet(capacity, raw_dir / 'wind_offshore_installed_capacity_2024-01-01_to_2026-07-01.parquet')
            data.save_parquet(self.raw.iloc[:0], raw_dir / 'offshore_ECMWF_singleruns_only_seed.parquet')
            with patch.dict('os.environ', {'ENTSOE_API_KEY': 'test-only'}), \
                 patch('utils.get_installed_capacity.get_installed_power', side_effect=ConnectionError), \
                 patch('entsoe.EntsoePandasClient.query_generation', side_effect=ConnectionError), \
                 patch('utils.offshore_utils.fetch_offshore_weather', return_value=self.raw.iloc[:0]), \
                 patch('utils.offshore_final_data.time.sleep'):
                features, kept_generation, kept_capacity, events, _ = data.refresh_sources(
                    root, end_day='2026-06-12', max_requests=1)
            self.assertTrue(features.empty)
            pd.testing.assert_frame_equal(generation, kept_generation)
            pd.testing.assert_frame_equal(capacity, kept_capacity)
            self.assertEqual(set(events.source), {'weather', 'generation', 'capacity'})

    def test_empty_weather_still_has_coverage(self):
        generation = pd.DataFrame({'gen_wind_off_act': []}, index=pd.DatetimeIndex([], tz='UTC'))
        features = pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))
        capacity = pd.DataFrame({'dt_utc': [pd.Timestamp('2026-06-01', tz='UTC')], 'offshore_capacity': [10.0]})
        dataset, coverage = data.assemble_dataset(self.original, features, generation,
            capacity, pd.Timestamp('2026-06-12'), self.features)
        self.assertFalse(coverage.row_complete.iloc[0])
        self.assertFalse(coverage.weather_available.iloc[0])


if __name__ == '__main__':
    unittest.main()
