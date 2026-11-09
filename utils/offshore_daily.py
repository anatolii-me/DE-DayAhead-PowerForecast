"""Refresh offshore sources, extend the rolling backtest and issue tomorrow's forecast."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.offshore_dashboard import DashboardDataError
from utils.offshore_day_ahead import issue_tomorrow_forecast, training_rows
from utils.offshore_final_data import (
    FIRST_NEW_DAY, assemble_dataset, grid, refresh_sources, save_parquet,
)

PUBLIC = Path("dashboard_data")
ORIGINAL = Path(
    "data/datasets/offshore_dataset_15m_interpolated_14_points_proxy_2024-03-14_to_2026-06-11.parquet"
)


def extend_backtest(root: Path, dataset: pd.DataFrame, coverage: pd.DataFrame,
                    selection: dict) -> pd.DataFrame:
    """Fit only unscored complete days, preserving forecasts issued live."""
    from utils.offshore_final_evaluation import build_model, make_training_weights

    path = root / PUBLIC / "latest_predictions.parquet"
    existing = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    existing = existing.loc[existing.model.eq("XGBoost") & existing.loss.eq("l1")].copy() if not existing.empty else existing
    existing_days = set(existing.day.astype(str)) if not existing.empty else set()
    config = selection["best_by_model"]["XGBoost"]
    features = selection["fit_protocol"]["features"]
    seed = selection["deployment_seed"]
    additions = []
    for day in coverage.index[coverage.row_complete]:
        day = pd.Timestamp(day)
        if str(day.date()) in existing_days:
            continue
        issued_path = root / PUBLIC / "issued" / f"{day.date()}.parquet"
        if issued_path.exists():
            issued = pd.read_parquet(issued_path)
            expected = grid(day)
            times = pd.DatetimeIndex(pd.to_datetime(issued.target_time, utc=True))
            if not times.equals(expected):
                raise DashboardDataError(f"Issued forecast for {day.date()} has an incomplete grid.")
            raw_prediction = issued.prediction.to_numpy(dtype=float)
            provenance = "issued_live"
        else:
            try:
                train, _, _, _ = training_rows(dataset, day, features)
            except DashboardDataError:
                continue
            target = dataset.reindex(grid(day))
            run_deadline = (day - pd.DateOffset(days=1) + pd.Timedelta(hours=12)).tz_localize("Europe/Berlin").tz_convert("UTC")
            if not pd.to_datetime(target.dt_run_utc, utc=True).lt(run_deadline).all():
                raise DashboardDataError(f"Weather run for {day.date()} crossed the day-ahead deadline.")
            x_train = train[features].astype("float32")
            y_train = train.capacity_factor.to_numpy(dtype=float)
            model = build_model(config, seed)
            model.fit(x_train, y_train, sample_weight=make_training_weights(y_train, config))
            raw_prediction = np.asarray(model.predict(target[features].astype("float32")), dtype=float)
            provenance = "retrospective_simulation"
        target = dataset.reindex(grid(day))
        if not np.isfinite(raw_prediction).all():
            raise DashboardDataError(f"Non-finite backtest predictions for {day.date()}.")
        additions.append(pd.DataFrame({
            "target_time": target.index.astype(str),
            "raw_prediction": raw_prediction,
            "prediction": np.clip(raw_prediction, 0, 1),
            "actual": target.capacity_factor.to_numpy(dtype=float),
            "day": str(day.date()), "seed": seed, "model": "XGBoost",
            "loss": "l1", "provenance": provenance,
        }))
        print(f"Backtest {day.date()}: {provenance}")
    if additions:
        existing = pd.concat([existing, *additions], ignore_index=True)
    if existing.empty:
        raise DashboardDataError("No complete backtest predictions are available.")
    # ENTSO-E may revise observed generation. Update only the label; an issued
    # prediction must retain the value that was actually shown to users.
    times = pd.DatetimeIndex(pd.to_datetime(existing.target_time, utc=True))
    labels = dataset.capacity_factor.reindex(times).to_numpy(dtype=float)
    available = np.isfinite(labels)
    existing.loc[available, "actual"] = labels[available]
    existing = existing.sort_values("target_time").drop_duplicates("target_time", keep="last")
    save_parquet(existing, path)
    return existing


def run(root: Path, *, update: bool = True) -> None:
    root = root.resolve()
    public = root / PUBLIC
    selection = json.loads((public / "frozen_selection.json").read_text(encoding="utf-8"))
    original_path = root / ORIGINAL
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != selection["dataset_hash"]:
        raise DashboardDataError("The original training dataset differs from the frozen model selection.")
    original = pd.read_parquet(original_path)
    today = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None).normalize()
    # Issue day is D-1. The last training and displayed actual day is D-2.
    end_day = today - pd.DateOffset(days=1)
    weather, generation, capacity, events, _ = refresh_sources(
        root, end_day=end_day, update=update, revision_days=7,
    )
    features = selection["fit_protocol"]["features"]
    dataset, coverage = assemble_dataset(
        original, weather, generation, capacity, end_day, features,
        source_events=events,
    )
    latest = coverage.loc[end_day]
    if not latest.row_complete:
        raise DashboardDataError(f"D-2 ({end_day.date()}) is incomplete: {latest.reason}")
    extend_backtest(root, dataset, coverage, selection)
    actual = dataset[["capacity", "capacity_factor", "gen_wind_off_act"]]
    save_parquet(actual, public / "actual_dataset.parquet")
    save_parquet(coverage, public / "daily_coverage.parquet")
    if not events.empty:
        events.drop(columns=["detail"], errors="ignore").to_csv(public / "source_events.csv", index=False)
    if update:
        issued = issue_tomorrow_forecast(root, dataset=dataset)
        forward = pd.read_parquet(issued)
        target_day = pd.Timestamp(forward.target_time.iloc[0]).tz_convert("Europe/Berlin").date()
        save_parquet(forward, public / "issued" / f"{target_day}.parquet")
        status = {
            "last_updated_utc": str(pd.Timestamp.now(tz="UTC")),
            "fitted_at_utc": str(forward.fitted_at_utc.iloc[0]),
            "latest_actual_utc": str(grid(end_day)[-1]),
            "forecast_day": str(target_day),
        }
        (public / "status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    key = os.getenv("ENTSOE_API_KEY", "")
    if key:
        secret = key.encode("utf-8")
        for path in public.rglob("*"):
            if path.is_file() and secret in path.read_bytes():
                raise DashboardDataError(f"Credential value found in public artifact: {path.name}")
    print(f"Complete source days: {int(coverage.row_complete.sum())}/{len(coverage)}; latest {end_day.date()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--offline", action="store_true", help="Use cached sources and do not issue a forward forecast")
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(args.project_root / ".env", override=False)
    run(args.project_root, update=not args.offline)


if __name__ == "__main__":
    main()
