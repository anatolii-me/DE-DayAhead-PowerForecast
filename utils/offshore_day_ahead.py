"""Issue one genuinely forward XGBoost HPO forecast for tomorrow.

This command consumes refreshed notebook-6 data and the D-1 ECMWF run. It is
separate from the Streamlit process: opening a dashboard must never train a
model or present a scored holdout day as a live forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.offshore_dashboard import BERLIN_TIMEZONE, DashboardDataError, DashboardPaths


def training_rows(
    dataset: pd.DataFrame, target_day: pd.Timestamp, features: list[str],
    max_missing_days: int = 3,
) -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp, list[str]]:
    """Reproduce notebook 6's 365-local-day training grid, ending at D-2 23:45."""
    day = pd.Timestamp(target_day).normalize()
    if day.tzinfo is not None:
        day = day.tz_convert(BERLIN_TIMEZONE).tz_localize(None)
    if not 0 <= max_missing_days < 365:
        raise ValueError("max_missing_days must be in [0, 364]")
    cutoff = (day - pd.DateOffset(days=1)).tz_localize(BERLIN_TIMEZONE)
    start = cutoff - pd.DateOffset(days=365)
    grid = pd.date_range(start, cutoff, freq="15min", inclusive="left").tz_convert("UTC")
    train = dataset.reindex(grid)
    if "row_complete" not in train or "capacity_factor" not in train:
        raise DashboardDataError("The training dataset lacks notebook-6 coverage or target columns.")
    local_dates = grid.tz_convert(BERLIN_TIMEZONE).tz_localize(None).normalize()
    complete = train.row_complete.fillna(False).astype(bool).groupby(local_dates).all()
    last_training_day = day - pd.DateOffset(days=2)
    if not bool(complete.loc[last_training_day]):
        raise DashboardDataError(
            f"Training data for D-2 ({last_training_day.date()}) are incomplete. "
            "Refresh notebook-6 sources before issuing tomorrow's forecast."
        )
    dropped = [str(date.date()) for date in complete.index[~complete]]
    if len(dropped) > max_missing_days:
        raise DashboardDataError(
            f"The 365-day training window has {len(dropped)} incomplete days; "
            f"at most {max_missing_days} are allowed."
        )
    missing_features = set(features).difference(train.columns)
    if missing_features:
        raise DashboardDataError(
            "The training dataset lacks model features: " + ", ".join(sorted(missing_features))
        )
    keep = complete.reindex(local_dates).to_numpy(dtype=bool)
    train = train.loc[keep]
    if not np.isfinite(train[features + ["capacity_factor"]].to_numpy(dtype=float)).all():
        raise DashboardDataError("Training rows contain non-finite features or labels.")
    return train, start.tz_convert("UTC"), cutoff.tz_convert("UTC"), dropped


def capacity_for_targets(
    monthly: pd.DataFrame, targets: pd.DatetimeIndex, max_age_days: int = 62,
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Backward-align each forecast interval to its own monthly capacity value."""
    required = {"dt_utc", "offshore_capacity"}
    if required.difference(monthly.columns):
        raise DashboardDataError("Monthly capacity data lack dt_utc or offshore_capacity.")
    effective = pd.DatetimeIndex(pd.to_datetime(monthly.dt_utc, utc=True))
    values = pd.to_numeric(monthly.offshore_capacity, errors="coerce").to_numpy(dtype=float)
    source = pd.DataFrame({"capacity_gw": values}, index=effective)
    source = source.loc[np.isfinite(source.capacity_gw) & source.capacity_gw.gt(0)]
    source = source.loc[~source.index.duplicated(keep="last")].sort_index()
    if source.empty:
        raise DashboardDataError("No valid monthly installed capacity is available.")
    utc_targets = targets.tz_convert("UTC")
    positions = source.index.searchsorted(utc_targets, side="right") - 1
    if (positions < 0).any():
        raise DashboardDataError("Installed capacity is missing before the forecast day.")
    matched_times = source.index.take(positions)
    age = (utc_targets - matched_times).total_seconds() / 86400
    if (age > max_age_days).any():
        raise DashboardDataError(
            "Installed capacity is too old for the forecast day; refresh monthly capacity."
        )
    return source.capacity_gw.to_numpy()[positions] * 1000.0, matched_times


def issue_tomorrow_forecast(project_root: Path) -> Path:
    """Fit the selected model once and atomically publish tomorrow's 15-minute CF/MW."""
    from utils.offshore_final_data import cached_weather_features, grid, save_parquet
    from utils.offshore_final_evaluation import build_model, make_training_weights

    now = pd.Timestamp.now(tz=BERLIN_TIMEZONE)
    target_day = now.normalize() + pd.DateOffset(days=1)
    run_day = target_day.tz_localize(None) - pd.DateOffset(days=1)
    paths = DashboardPaths.from_project_root(project_root)
    if paths.actual_dataset is None or not paths.actual_dataset.exists():
        raise DashboardDataError("The refreshed notebook-6 training dataset is missing.")
    selection_path = paths.historical_predictions.parent.parent / "frozen_selection.json"
    if not selection_path.exists():
        raise DashboardDataError("The frozen notebook-6 model selection is missing.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config = selection["best_by_model"]["XGBoost"]
    if config["loss"] != "l1":
        raise DashboardDataError("The frozen production selection is not XGBoost L1.")
    features = selection["fit_protocol"]["features"]
    dataset = pd.read_parquet(paths.actual_dataset)
    dataset.index = pd.DatetimeIndex(pd.to_datetime(dataset.index, utc=True))
    dataset = dataset.sort_index()
    train, start_utc, end_utc, dropped_days = training_rows(
        dataset, target_day, features,
    )

    cache = project_root / "results/offshore_final/source_cache"
    raw_path = cache / "weather" / f"{run_day.date()}.parquet"
    if raw_path.exists():
        raw = pd.read_parquet(raw_path)
    else:
        from utils.offshore_utils import fetch_offshore_weather
        raw = fetch_offshore_weather(str(run_day.date()), str(run_day.date()))
        if raw.empty:
            raise DashboardDataError("The D-1 ECMWF weather run is unavailable.")
        save_parquet(raw, raw_path)
    weather = cached_weather_features(raw, run_day, cache / "features")
    targets = grid(target_day.tz_localize(None))
    if not weather.index.equals(targets):
        raise DashboardDataError("The D-1 weather run does not cover every target interval.")
    missing_weather_features = set(features).difference(weather.columns)
    if missing_weather_features:
        raise DashboardDataError(
            "The D-1 weather run lacks model features: "
            + ", ".join(sorted(missing_weather_features))
        )
    if not np.isfinite(weather[features].to_numpy(dtype=float)).all():
        raise DashboardDataError("The D-1 weather run has non-finite model features.")
    deadline = (run_day + pd.Timedelta(hours=12)).tz_localize(BERLIN_TIMEZONE).tz_convert("UTC")
    if not pd.to_datetime(weather.dt_run_utc, utc=True).lt(deadline).all():
        raise DashboardDataError("The weather run is later than the day-ahead feature deadline.")

    capacity_path = cache / "capacity.parquet"
    if not capacity_path.exists():
        raise DashboardDataError("The refreshed monthly capacity series is missing.")
    capacity_mw, effective_times = capacity_for_targets(
        pd.read_parquet(capacity_path), targets,
    )
    x_train = train[features].astype("float32")
    y_train = train.capacity_factor.to_numpy(dtype=float)
    model = build_model(config, selection["deployment_seed"])
    model.fit(x_train, y_train, sample_weight=make_training_weights(y_train, config))
    raw_prediction = np.asarray(model.predict(weather[features].astype("float32")))
    if not np.isfinite(raw_prediction).all():
        raise DashboardDataError("The model returned non-finite forecast values.")
    prediction = np.clip(raw_prediction, 0, 1)
    fitted_at = pd.Timestamp.now(tz="UTC")
    if fitted_at.tz_convert(BERLIN_TIMEZONE).normalize() != now.normalize():
        raise DashboardDataError("The local issue day changed during the refit; rerun the forecast.")
    output = pd.DataFrame({
        "target_time": targets.astype(str),
        "prediction": prediction,
        "capacity_mw": capacity_mw,
        "prediction_mw": prediction * capacity_mw,
        "capacity_effective_time_utc": effective_times.astype(str),
        "model": "XGBoost",
        "loss": "l1",
        "fitted_at_utc": str(fitted_at),
        "training_start_utc": str(start_utc),
        "training_end_exclusive_utc": str(end_utc),
        "weather_run_utc": str(run_day.tz_localize("UTC")),
        "train_rows": len(train),
        "dropped_train_days": ", ".join(dropped_days),
    })
    output_path = paths.forward_predictions
    if output_path is None:
        raise DashboardDataError("OFFSHORE_FORWARD_PREDICTIONS is not configured.")
    save_parquet(output, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        path = issue_tomorrow_forecast(args.project_root.resolve())
    except DashboardDataError as error:
        parser.exit(2, f"Forecast not published: {error}\n")
    print(f"Published tomorrow's forecast to {path}")


if __name__ == "__main__":
    main()
