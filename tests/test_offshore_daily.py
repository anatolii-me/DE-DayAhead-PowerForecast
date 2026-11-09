"""Retry deadlines and independent source recovery for daily publication."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from utils.offshore_daily import wait_for_critical_sources
from utils.offshore_day_ahead import issue_tomorrow_forecast
from utils.offshore_dashboard import DashboardDataError, DashboardPaths, load_forward_predictions
from utils.offshore_final_data import grid


class RetryTests(unittest.TestCase):
    def test_saved_model_issues_real_fallback_forecast(self):
        source = Path(__file__).resolve().parents[1] / "dashboard_data"
        selection = json.loads((source / "frozen_selection.json").read_text())
        snapshot = json.loads((source / "model/metadata.json").read_text())
        fitted_day = pd.Timestamp(snapshot["fitted_at_utc"]).tz_convert("Europe/Berlin").tz_localize(None).normalize()
        issue_day = fitted_day + pd.DateOffset(days=1)
        issue_time = (issue_day + pd.Timedelta(hours=10)).tz_localize("Europe/Berlin")
        targets = grid(issue_day + pd.DateOffset(days=1))
        weather = pd.DataFrame(0.0, index=targets, columns=selection["fit_protocol"]["features"])
        weather["dt_run_utc"] = issue_day.tz_localize("UTC")
        with tempfile.TemporaryDirectory() as tmp:
            public = Path(tmp) / "dashboard_data"
            (public / "model").mkdir(parents=True)
            (public / "source_cache").mkdir(parents=True)
            for name in ("model/latest.ubj.gz", "model/metadata.json", "frozen_selection.json",
                         "source_cache/capacity.parquet"):
                shutil.copyfile(source / name, public / name)
            path = issue_tomorrow_forecast(
                Path(tmp), dataset=pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")),
                weather=weather, reuse_model=True,
                clock=lambda: issue_time.tz_convert("UTC"))
            forward = load_forward_predictions(
                DashboardPaths(Path("unused"), Path("unused"), None, path),
                now=issue_time)
            self.assertEqual(len(forward), len(targets))
            self.assertTrue(forward.prediction.between(0, 1).all())
            self.assertEqual(forward.fit_mode.iloc[0], "reused")
            self.assertEqual(forward.model_age_days.iloc[0], 1)
            metadata_path = public / "model/metadata.json"
            metadata = json.loads(metadata_path.read_text())
            metadata["model_sha256"] = "0" * 64
            metadata_path.write_text(json.dumps(metadata))
            with self.assertRaises(DashboardDataError):
                issue_tomorrow_forecast(
                    Path(tmp), dataset=pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC")),
                    weather=weather, reuse_model=True,
                    clock=lambda: issue_time.tz_convert("UTC"))

    def test_both_sources_recover_after_cooldown(self):
        day = pd.Timestamp("2026-01-18")
        complete = pd.DataFrame({"gen_wind_off_act": 1000.0}, index=grid(day))
        partial = complete.iloc[:-1]
        current = [pd.Timestamp("2026-01-19 10:30", tz="Europe/Berlin")]
        weather = pd.DataFrame({"ready": [True]})
        calls = {"generation": 0, "weather": 0}

        def fetch(*args):
            calls["generation"] += 1
            if calls["generation"] == 1:
                raise TimeoutError("temporary outage")
            return complete, 1

        def fetch_weather(*args):
            calls["weather"] += 1
            if calls["weather"] == 1:
                raise TimeoutError("temporary outage")
            return weather

        def advance(seconds):
            current[0] += pd.Timedelta(seconds=seconds)

        log = {}
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict("os.environ", {"ENTSOE_API_KEY": "test-placeholder"}), \
             patch("utils.offshore_daily.fetch_generation_day", side_effect=fetch), \
             patch("utils.offshore_daily.current_weather", side_effect=fetch_weather):
            result, fetched_weather, ready = wait_for_critical_sources(
                Path(tmp), partial, day, ["feature"], log,
                clock=lambda: current[0], sleep=advance)
        self.assertTrue(ready)
        self.assertIs(fetched_weather, weather)
        self.assertEqual(len(result), len(complete))
        self.assertEqual(calls, {"generation": 2, "weather": 2})
        self.assertEqual(len(log["attempts"]), 4)
        self.assertNotIn("temporary outage", str(log))

    def test_missing_weather_stops_at_safe_cutoff(self):
        day = pd.Timestamp("2026-01-18")
        generation = pd.DataFrame({"gen_wind_off_act": 1000.0}, index=grid(day))
        current = [pd.Timestamp("2026-01-19 10:50", tz="Europe/Berlin")]
        sleeps = []

        def advance(seconds):
            sleeps.append(seconds)
            current[0] += pd.Timedelta(seconds=seconds)

        log = {}
        with tempfile.TemporaryDirectory() as tmp, \
             patch("utils.offshore_daily.current_weather", side_effect=TimeoutError("unavailable")):
            _, fetched_weather, ready = wait_for_critical_sources(
                Path(tmp), generation, day, ["feature"], log,
                clock=lambda: current[0], sleep=advance)
        self.assertTrue(ready)
        self.assertIsNone(fetched_weather)
        self.assertEqual(current[0].strftime("%H:%M"), "11:15")
        self.assertEqual(sleeps, [900, 600])
        self.assertEqual(len(log["attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
