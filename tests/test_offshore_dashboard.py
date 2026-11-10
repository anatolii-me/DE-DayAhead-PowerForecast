"""Tests for dashboard calculations independent of Streamlit's UI runtime."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils.offshore_dashboard import (
    DashboardDataError,
    DashboardPaths,
    build_forecast_figure,
    calculate_period_metrics,
    load_dashboard_predictions,
    load_forward_predictions,
    metric_windows,
    week_ago_index,
)
from utils.offshore_day_ahead import capacity_for_targets, training_rows
from utils.offshore_final_data import grid, merge_generation


class OffshoreDashboardTests(unittest.TestCase):
    def test_local_delivery_grids_follow_daylight_saving(self) -> None:
        self.assertEqual(len(grid("2026-03-29")), 92)
        self.assertEqual(len(grid("2026-10-25")), 100)

    def test_partial_generation_revision_overwrites_only_returned_intervals(self) -> None:
        index = pd.date_range("2026-10-13", periods=3, freq="15min", tz="UTC")
        cached = pd.DataFrame({"gen_wind_off_act": [100.0, 200.0, 300.0]}, index=index)
        fresh = pd.DataFrame({"gen_wind_off_act": [220.0]}, index=index[1:2])
        result = merge_generation(cached, fresh)
        self.assertEqual(result.gen_wind_off_act.tolist(), [100.0, 220.0, 300.0])

    def test_dataset_selection_uses_latest_covered_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            dataset_dir = base / "data" / "datasets"
            dataset_dir.mkdir(parents=True)
            older = dataset_dir / "offshore_cf_15m_ecmwf14p_engineered_2024-01-01_to_2026-01-10.parquet"
            newer = base / "dashboard_data" / "actual_dataset.parquet"
            newer.parent.mkdir(parents=True)
            pd.DataFrame(index=pd.DatetimeIndex(["2026-01-10T00:00:00Z"])).to_parquet(older)
            pd.DataFrame(index=pd.DatetimeIndex(["2026-01-12T00:00:00Z"])).to_parquet(newer)
            self.assertEqual(DashboardPaths.from_project_root(base).actual_dataset, newer)

    def test_metrics_and_skill_score(self) -> None:
        frame = pd.DataFrame(
            {
                "actual": [0.2, 0.4, 0.6],
                "prediction": [0.3, 0.3, 0.6],
                "weekly_persistence": [0.4, 0.2, 0.7],
                "actual_mw": [2000, 4000, 6000],
                "prediction_mw": [3000, 3000, 6000],
                "weekly_persistence_mw": [4000, 2000, 7000],
            }
        )

        metrics = calculate_period_metrics(frame)

        self.assertAlmostEqual(metrics["nMAE"], 2 / 30)
        self.assertAlmostEqual(metrics["nRMSE"], np.sqrt(2 / 300))
        self.assertAlmostEqual(metrics["Skill score"], 0.6)
        self.assertEqual(metrics["observations"], 3)
        self.assertEqual(metrics["baseline_observations"], 3)

    def test_weekly_baseline_uses_same_local_time_across_dst(self) -> None:
        index = pd.DatetimeIndex([pd.Timestamp("2026-03-29 01:00:00+00:00")])
        previous = week_ago_index(index)

        # 03:00 CEST becomes 03:00 CET a week earlier, which is 02:00 UTC.
        self.assertEqual(previous[0], pd.Timestamp("2026-03-22 02:00:00+00:00"))

    def test_weekly_baseline_does_not_invent_a_nonexistent_dst_time(self) -> None:
        # 02:00 CEST on 5 April maps to the skipped 02:00 on the 29 March
        # spring-forward Sunday.  Skill score must not shift this to 03:00.
        index = pd.DatetimeIndex([pd.Timestamp("2026-04-05 00:00:00+00:00")])
        self.assertTrue(pd.isna(week_ago_index(index)[0]))

    def test_loads_selected_model_and_builds_figures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            timestamps = pd.date_range("2026-01-01", periods=18 * 96, freq="15min", tz="UTC")
            actual = np.linspace(0.1, 0.9, len(timestamps))
            capacity_mw = 10_000.0
            history = pd.DataFrame(
                {
                    "target_time": timestamps[:10 * 96].astype(str),
                    "actual": actual[:10 * 96],
                    "prediction": actual[:10 * 96] + 0.01,
                    "raw_prediction": actual[:10 * 96] + 0.01,
                    "seed": 1337,
                }
            )
            latest = pd.DataFrame(
                {
                    "target_time": timestamps[10 * 96 :].astype(str),
                    "actual": actual[10 * 96 :],
                    "prediction": actual[10 * 96 :] + 0.01,
                    "raw_prediction": actual[10 * 96 :] + 0.01,
                    "model": "XGBoost",
                    "loss": "l1",
                    "seed": 1337,
                }
            )
            gap_time = timestamps[11 * 96]
            latest = latest.drop(index=96)
            historical_path = base / "history.parquet"
            latest_path = base / "latest.parquet"
            actual_path = base / "actual.parquet"
            history.to_parquet(historical_path)
            latest.to_parquet(latest_path)
            # Simulate a complete source snapshot that was written before the
            # final eight prediction days.  Prediction actuals should fill the
            # newer baseline values without overwriting source data.
            actual_index = pd.date_range(
                timestamps.min() - pd.Timedelta(days=7),
                periods=17 * 96,
                freq="15min",
                tz="UTC",
            )
            actual_history = pd.Series(0.2, index=actual_index)
            source_times = timestamps[:10 * 96]
            actual_history.loc[source_times] = actual[:10 * 96]
            actual_history.loc[gap_time] = actual[11 * 96]
            source = pd.DataFrame(index=actual_index.union(timestamps))
            source["capacity_factor"] = actual_history
            source["capacity"] = 10.0
            source.loc[timestamps[10 * 96 :], "capacity"] = 12.0
            source["gen_wind_off_act"] = source.capacity_factor * source.capacity * 1000
            source.to_parquet(actual_path)

            predictions = load_dashboard_predictions(
                DashboardPaths(historical_path, latest_path, actual_path)
            )
            self.assertEqual(len(predictions), len(timestamps))
            self.assertTrue(pd.isna(predictions.loc[gap_time, "prediction_mw"]))
            self.assertTrue(np.isfinite(predictions.loc[gap_time, "actual_mw"]))
            self.assertTrue(predictions.weekly_persistence.notna().all())
            self.assertAlmostEqual(predictions.actual_mw.iloc[0], actual[0] * capacity_mw)
            self.assertAlmostEqual(predictions.prediction_mw.iloc[0], (actual[0] + 0.01) * capacity_mw)
            self.assertAlmostEqual(predictions.actual_mw.iloc[-1], actual[-1] * 12_000)

            windows = metric_windows(predictions)
            self.assertEqual(len(windows), 12)
            self.assertNotIn("colour_score", windows.columns)
            figure = build_forecast_figure(predictions)
            self.assertEqual(len(figure.data), 2)
            self.assertIsNone(figure.layout.title.text)
            self.assertIsNone(figure.layout.xaxis.title.text)
            self.assertEqual(figure.layout.yaxis.title.text, "Generation (MW)")

            source.loc[timestamps[0], "gen_wind_off_act"] += 1.0
            source.to_parquet(actual_path)
            with self.assertRaises(DashboardDataError):
                load_dashboard_predictions(
                    DashboardPaths(historical_path, latest_path, actual_path)
                )

    def test_forward_forecast_is_unscored_current_and_has_provenance(self) -> None:
        day = pd.Timestamp("2026-01-20", tz="Europe/Berlin")
        targets = pd.date_range(day, day + pd.DateOffset(days=1), freq="15min", inclusive="left")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forward.parquet"
            frame = pd.DataFrame({
                "target_time": targets.tz_convert("UTC").astype(str),
                "prediction": 0.4,
                "capacity_mw": 11_000.0,
                "model": "XGBoost",
                "loss": "l1",
                "fitted_at_utc": "2026-01-19 09:30:00+00:00",
                "training_start_utc": str((day - pd.DateOffset(days=366)).tz_convert("UTC")),
                "training_end_exclusive_utc": str((day - pd.DateOffset(days=1)).tz_convert("UTC")),
                "weather_run_utc": "2026-01-19 00:00:00+00:00",
            })
            frame.to_parquet(path)
            paths = DashboardPaths(Path("unused"), Path("unused"), None, path)
            forward = load_forward_predictions(
                paths, now=pd.Timestamp("2026-01-19 11:00", tz="Europe/Berlin")
            )
            self.assertIsNotNone(forward)
            self.assertEqual(len(forward), 96)
            self.assertTrue(forward.actual_mw.isna().all())
            self.assertTrue(forward.prediction_mw.eq(4_400).all())
            evaluated_index = pd.date_range("2026-01-18", periods=96, freq="15min", tz="UTC")
            evaluated = pd.DataFrame({
                "actual_mw": 3500.0, "prediction_mw": 3600.0,
            }, index=evaluated_index)
            figure = build_forecast_figure(evaluated, forward=forward)
            self.assertEqual([trace.name for trace in figure.data],
                             ["Actual", "Forecast", "Refit day", "Delivery day"])
            self.assertEqual(sum(pd.notna(figure.data[1].y)), 192)
            self.assertEqual(len(figure.layout.shapes), 3)
            self.assertEqual(figure.data[2].marker.color, figure.layout.shapes[1].fillcolor)
            self.assertEqual(figure.data[3].marker.color, figure.layout.shapes[2].fillcolor)
            self.assertEqual(figure.data[2].marker.color, "#EEF3EF")
            self.assertEqual(figure.layout.annotations[0].font.color, "#1B2524")
            self.assertIsNone(load_forward_predictions(
                paths, now=pd.Timestamp("2026-01-22 10:00", tz="Europe/Berlin")
            ))
            reused = frame.copy()
            reused["fit_mode"] = "reused"
            reused["model_age_days"] = 1
            reused["issued_at_utc"] = "2026-01-19 09:30:00+00:00"
            reused["fitted_at_utc"] = "2026-01-18 09:30:00+00:00"
            previous_fit_day = day - pd.DateOffset(days=2)
            reused["training_start_utc"] = str((previous_fit_day - pd.DateOffset(days=365)).tz_convert("UTC"))
            reused["training_end_exclusive_utc"] = str(previous_fit_day.tz_convert("UTC"))
            reused.to_parquet(path)
            self.assertEqual(load_forward_predictions(
                paths, now=pd.Timestamp("2026-01-19 11:00", tz="Europe/Berlin")
            ).fit_mode.iloc[0], "reused")
            reused["model_age_days"] = 0
            reused.to_parquet(path)
            with self.assertRaises(DashboardDataError):
                load_forward_predictions(paths, now=pd.Timestamp("2026-01-19 11:00", tz="Europe/Berlin"))
            late = frame.copy()
            late["fitted_at_utc"] = "2026-01-19 12:30:00+00:00"
            late.to_parquet(path)
            with self.assertRaises(DashboardDataError):
                load_forward_predictions(
                    paths, now=pd.Timestamp("2026-01-19 14:00", tz="Europe/Berlin")
                )
            frame.iloc[:-1].to_parquet(path)
            with self.assertRaises(DashboardDataError):
                load_forward_predictions(
                    paths, now=pd.Timestamp("2026-01-19 11:00", tz="Europe/Berlin")
                )

    def test_capacity_uses_each_interval_month_and_training_ends_at_d_minus_2(self) -> None:
        monthly = pd.DataFrame({
            "dt_utc": ["2026-09-01T00:00:00Z", "2026-10-01T00:00:00Z"],
            "offshore_capacity": [10.0, 11.0],
        })
        targets = pd.DatetimeIndex([
            "2026-09-30T23:45:00Z", "2026-10-01T00:00:00Z",
        ])
        capacity, effective = capacity_for_targets(monthly, targets)
        np.testing.assert_array_equal(capacity, [10_000, 11_000])
        self.assertEqual(effective[1], pd.Timestamp("2026-10-01T00:00:00Z"))

        day = pd.Timestamp("2026-10-16")
        cutoff = (day - pd.DateOffset(days=1)).tz_localize("Europe/Berlin")
        start = cutoff - pd.DateOffset(days=365)
        index = pd.date_range(start, cutoff, freq="15min", inclusive="left").tz_convert("UTC")
        dataset = pd.DataFrame({
            "row_complete": True, "capacity_factor": 0.4, "feature": 1.0,
        }, index=index)
        train, actual_start, actual_end, dropped = training_rows(dataset, day, ["feature"])
        self.assertEqual(len(train), len(index))
        self.assertEqual(actual_start, index[0])
        self.assertEqual(actual_end, index[-1] + pd.Timedelta(minutes=15))
        self.assertEqual(index[-1].tz_convert("Europe/Berlin").strftime("%Y-%m-%d %H:%M"),
                         "2026-10-14 23:45")
        self.assertEqual(dropped, [])
        future = pd.DataFrame({"row_complete": True, "capacity_factor": 0.99, "feature": 999.0},
                              index=pd.date_range(cutoff, periods=192, freq="15min").tz_convert("UTC"))
        with_future = pd.concat([dataset, future])
        restricted, _, restricted_end, _ = training_rows(with_future, day, ["feature"])
        pd.testing.assert_frame_equal(restricted, train)
        self.assertEqual(restricted_end, actual_end)
        dataset.loc[index[-1], "row_complete"] = False
        with self.assertRaises(DashboardDataError):
            training_rows(dataset, day, ["feature"])


if __name__ == "__main__":
    unittest.main()
