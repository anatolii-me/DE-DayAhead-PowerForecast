"""Scheduler guards and developer sandbox isolation without external APIs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import pandas as pd

from utils.offshore_daily import run, refresh_scored_actuals, recover_current_day, refresh_display_actuals
from utils.offshore_final_data import grid
from utils.offshore_simulation import prepare
from utils.offshore_day_ahead import capacity_for_targets
from utils.offshore_dashboard import DashboardDataError
from utils.offshore_dashboard import DashboardPaths, load_forward_predictions


class OperationsTests(unittest.TestCase):
    def test_display_actuals_stop_at_logical_time_and_do_not_change_training(self):
        day = pd.Timestamp("2026-11-10")
        now = pd.Timestamp("2026-11-10 07:00Z")
        complete_day = pd.DataFrame({"gen_wind_off_act": 1000.0}, index=grid(day))
        training = complete_day.iloc[:2].copy()
        original = training.copy(deep=True)
        with tempfile.TemporaryDirectory() as tmp, patch("utils.offshore_daily.fetch_generation_day",
                return_value=(complete_day, 96)):
            log = {}
            latest = refresh_display_actuals(Path(tmp), training, now, log)
            display = pd.read_parquet(Path(tmp) / "dashboard_data/display_actuals.parquet")
            self.assertEqual(latest, pd.Timestamp("2026-11-10 06:45Z"))
            self.assertTrue((display.index + pd.Timedelta(minutes=15) <= now).all())
            pd.testing.assert_frame_equal(training, original)
            self.assertFalse((Path(tmp) / "dashboard_data/actual_dataset.parquet").exists())

    def test_recovery_uses_previous_run_and_does_not_replace_live_model(self):
        day = pd.Timestamp("2026-11-10")
        targets = grid(day)
        selection = {"fit_protocol": {"features": ["wind"]}, "deployment_seed": 1337,
                     "best_by_model": {"XGBoost": {}}}
        train = pd.DataFrame({"wind": [1.0], "capacity_factor": [0.5]})
        weather = pd.DataFrame({"wind": 1.0}, index=targets)
        model = Mock()
        model.predict.return_value = [0.4] * len(targets)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / "dashboard_data/source_cache"
            cache.mkdir(parents=True)
            pd.DataFrame({"dt_utc": [pd.Timestamp("2026-11-01", tz="UTC")],
                          "offshore_capacity": [10.0]}).to_parquet(cache / "capacity.parquet")
            sentinel = root / "dashboard_data/day_ahead_predictions.parquet"
            sentinel.write_bytes(b"tomorrow unchanged")
            with patch("utils.offshore_daily.training_rows", return_value=(train, "start", "end", [])) as training, \
                 patch("utils.offshore_daily.current_weather", return_value=weather) as fetch, \
                 patch("utils.offshore_final_evaluation.build_model", return_value=model), \
                 patch("utils.offshore_final_evaluation.make_training_weights", return_value=None):
                recover_current_day(root, pd.DataFrame(), day, selection)
            self.assertEqual(training.call_args.args[1], day)
            self.assertEqual(fetch.call_args.args[1], day - pd.DateOffset(days=1))
            pending = pd.read_parquet(root / "dashboard_data/pending_predictions.parquet")
            self.assertEqual(len(pending), 96)
            self.assertTrue(pending.provenance.eq("reconstructed").all())
            self.assertEqual(sentinel.read_bytes(), b"tomorrow unchanged")
            self.assertFalse((root / "dashboard_data/model").exists())
            # A second recovery reuses the archived reconstruction.
            with patch("utils.offshore_daily.current_weather") as fetch:
                recover_current_day(root, pd.DataFrame(), day, selection)
                fetch.assert_not_called()

    def test_recovery_preserves_issued_forecast(self):
        day = pd.Timestamp("2026-11-10")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            issued = root / "dashboard_data/issued/2026-11-10.parquet"
            issued.parent.mkdir(parents=True)
            frame = pd.DataFrame({"target_time": grid(day).astype(str), "prediction": 0.2,
                                  "capacity_mw": 10000.0})
            frame.to_parquet(issued)
            before = issued.read_bytes()
            with patch("utils.offshore_daily.training_rows") as train:
                recover_current_day(root, pd.DataFrame(), day, {})
                train.assert_not_called()
            pending = pd.read_parquet(root / "dashboard_data/pending_predictions.parquet")
            self.assertTrue(pending.provenance.eq("issued_live").all())
            self.assertEqual(issued.read_bytes(), before)

    def test_revised_labels_do_not_change_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "dashboard_data").mkdir()
            path = root / "dashboard_data/latest_predictions.parquet"
            time = pd.Timestamp("2026-11-09", tz="UTC")
            pd.DataFrame({"target_time": [str(time)], "actual": [0.2],
                          "prediction": [0.3]}).to_parquet(path)
            refresh_scored_actuals(root, pd.DataFrame({"capacity_factor": [0.25]}, index=[time]))
            result = pd.read_parquet(path)
            self.assertEqual(result.actual.iloc[0], 0.25)
            self.assertEqual(result.prediction.iloc[0], 0.3)

    def test_public_loader_rejects_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.parquet"
            pd.DataFrame({"simulation": [True]}).to_parquet(path)
            with self.assertRaisesRegex(DashboardDataError, "Simulation"):
                load_forward_predictions(DashboardPaths(path, path, None, path))

    def test_cutoff_uses_berlin_in_winter_and_summer(self):
        for instant in ("2026-01-19 10:30Z", "2026-07-19 09:30Z", "2026-11-10 04:59Z"):
            with self.subTest(instant=instant), tempfile.TemporaryDirectory() as tmp:
                # No inputs exist: returning without reading them proves the guard.
                run(Path(tmp), clock=lambda: pd.Timestamp(instant))
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_successful_day_is_not_refitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = root / "dashboard_data"
            (public / "issued").mkdir(parents=True)
            (public / "issued/2026-11-11.parquet").write_bytes(b"fixture")
            (public / "status.json").write_text(json.dumps({"forecast_day": "2026-11-11",
                "forecast_status": "issued", "fit_mode": "fresh"}))
            with patch("utils.offshore_daily.refresh_sources") as refresh:
                run(root, clock=lambda: pd.Timestamp("2026-11-10 06:00Z"))
            refresh.assert_not_called()

    def test_capacity_limit_remains_62_days(self):
        capacity = pd.DataFrame({"dt_utc": [pd.Timestamp("2026-09-01", tz="UTC")],
                                 "offshore_capacity": [9.0]})
        values, _ = capacity_for_targets(capacity, pd.DatetimeIndex(["2026-11-01T00:00Z"]))
        self.assertEqual(values[0], 9000)
        with self.assertRaises(DashboardDataError):
            capacity_for_targets(capacity, pd.DatetimeIndex(["2026-11-03T00:00Z"]))

    def test_simulation_rejects_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sentinel = root / "keep.txt"
            sentinel.write_text("unchanged")
            with self.assertRaises(ValueError):
                prepare(root, root)
            self.assertEqual(sentinel.read_text(), "unchanged")

    def test_simulation_copies_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source", Path(tmp) / "simulation"
            source.mkdir()
            (source / "dashboard_data").mkdir()
            (source / "dashboard_data/status.json").write_text("{}")
            (source / "dashboard_data/value.txt").write_text("original")
            files = ["data/raw_data/gen_act_per_type_2024-03-13_to_2026-06-11.parquet",
                     "data/raw_data/wind_offshore_installed_capacity_2024-01-01_to_2026-07-01.parquet"]
            from utils.offshore_daily import ORIGINAL
            for name in [str(ORIGINAL), *files]:
                path = source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            prepare(source, dest)
            (dest / "dashboard_data/value.txt").write_text("changed")
            self.assertEqual((source / "dashboard_data/value.txt").read_text(), "original")
            self.assertFalse((dest / "dashboard_data/status.json").exists())


if __name__ == "__main__":
    unittest.main()
