"""Issue one genuinely forward XGBoost HPO forecast for tomorrow.

This command consumes refreshed notebook-6 data and the D-1 ECMWF run. It is
separate from the Streamlit process: opening a dashboard must never train a
model or present a scored holdout day as a live forecast.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.offshore_dashboard import BERLIN_TIMEZONE, DashboardDataError, DashboardPaths


def current_weather(project_root: Path, run_day: pd.Timestamp, features: list[str]) -> pd.DataFrame:
    """Return a complete midnight run, fetching it when the cache is incomplete."""
    from utils.offshore_final_data import cached_weather_features, fetch_weather_with_fallback, grid, save_parquet

    cache = project_root / "dashboard_data/source_cache"
    raw_path = cache / "weather" / f"{run_day.date()}.parquet"
    raw = pd.read_parquet(raw_path) if raw_path.exists() else None
    try:
        weather = cached_weather_features(raw, run_day, cache / "features") if raw is not None else None
    except (ValueError, KeyError, AttributeError):
        weather = None
    if weather is None:
        raw = fetch_weather_with_fallback(run_day)
        if raw.empty:
            raise DashboardDataError("The current ECMWF run is unavailable.")
        save_parquet(raw, raw_path)
        weather = cached_weather_features(raw, run_day, cache / "features")
    targets = grid(run_day + pd.DateOffset(days=1))
    if not weather.index.equals(targets) or not set(features).issubset(weather.columns):
        raise DashboardDataError("The current ECMWF run is incomplete.")
    if not np.isfinite(weather[features].to_numpy(dtype=float)).all():
        raise DashboardDataError("The current ECMWF run has invalid features.")
    deadline = (run_day + pd.Timedelta(hours=12)).tz_localize(BERLIN_TIMEZONE).tz_convert("UTC")
    if not pd.to_datetime(weather.dt_run_utc, utc=True).lt(deadline).all():
        raise DashboardDataError("The current ECMWF run is later than the feature deadline.")
    return weather


def _selection_hash(selection: dict) -> str:
    identity = {key: selection[key] for key in ("best_by_model", "fit_protocol", "deployment_seed")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _save_model(model, metadata: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "latest.tmp.ubj.gz"
    try:
        raw = model.get_booster().save_raw(raw_format="ubj")
        compressed = gzip.compress(raw)
        metadata["model_sha256"] = hashlib.sha256(compressed).hexdigest()
        temporary.write_bytes(compressed)
        os.replace(temporary, directory / "latest.ubj.gz")
    finally:
        temporary.unlink(missing_ok=True)
    meta_tmp = directory / "metadata.tmp.json"
    meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(meta_tmp, directory / "metadata.json")


def fit_model_snapshot(project_root: Path, dataset: pd.DataFrame,
                       target_day: pd.Timestamp, selection: dict):
    """Fit and persist the latest complete 365-day training window."""
    from utils.offshore_final_evaluation import build_model, make_training_weights

    features = selection["fit_protocol"]["features"]
    config = selection["best_by_model"]["XGBoost"]
    train, start_utc, end_utc, dropped_days = training_rows(dataset, target_day, features)
    x_train = train[features].astype("float32")
    y_train = train.capacity_factor.to_numpy(dtype=float)
    model = build_model(config, selection["deployment_seed"])
    model.fit(x_train, y_train, sample_weight=make_training_weights(y_train, config))
    fitted_at = pd.Timestamp.now(tz="UTC")
    metadata = {
        "selection_hash": _selection_hash(selection), "features": features,
        "fitted_at_utc": str(fitted_at), "training_start_utc": str(start_utc),
        "training_end_exclusive_utc": str(end_utc), "train_rows": len(train),
        "dropped_train_days": dropped_days,
    }
    _save_model(model, metadata, project_root / "dashboard_data/model")
    return model, metadata


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


def issue_tomorrow_forecast(
    project_root: Path, *, dataset: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None, reuse_model: bool = False, clock=None,
) -> Path:
    """Issue tomorrow's forecast after a fresh fit or from the last saved fit."""
    from utils.offshore_final_data import grid, save_parquet
    from utils.offshore_final_evaluation import build_model

    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    now = clock().tz_convert(BERLIN_TIMEZONE)
    target_day = now.normalize() + pd.DateOffset(days=1)
    gate_closure = (now.normalize().tz_localize(None) + pd.Timedelta(hours=12)).tz_localize(BERLIN_TIMEZONE)
    if now >= gate_closure:
        raise DashboardDataError("The day-ahead auction gate closed at 12:00 Europe/Berlin; no market-ready forecast was published.")
    run_day = target_day.tz_localize(None) - pd.DateOffset(days=1)
    paths = DashboardPaths.from_project_root(project_root)
    full_dataset_path = project_root / "results/offshore_final/actual_dataset_with_coverage.parquet"
    if dataset is None and not full_dataset_path.exists():
        raise DashboardDataError("The refreshed full training dataset is missing. Run utils.offshore_daily.")
    selection_path = project_root / "dashboard_data/frozen_selection.json"
    if not selection_path.exists():
        raise DashboardDataError("The frozen notebook-6 model selection is missing.")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    config = selection["best_by_model"]["XGBoost"]
    if config["loss"] != "l1":
        raise DashboardDataError("The frozen production selection is not XGBoost L1.")
    features = selection["fit_protocol"]["features"]
    dataset = pd.read_parquet(full_dataset_path) if dataset is None else dataset.copy()
    dataset.index = pd.DatetimeIndex(pd.to_datetime(dataset.index, utc=True))
    dataset = dataset.sort_index()
    cache = project_root / "dashboard_data/source_cache"
    weather = current_weather(project_root, run_day, features) if weather is None else weather
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
    model = build_model(config, selection["deployment_seed"])
    model_dir = project_root / "dashboard_data/model"
    if reuse_model:
        meta_path = model_dir / "metadata.json"
        model_path = model_dir / "latest.ubj.gz"
        if not meta_path.exists() or not model_path.exists():
            raise DashboardDataError("No previously fitted model is available.")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata.get("selection_hash") != _selection_hash(selection) or metadata.get("features") != features:
            raise DashboardDataError("The saved model does not match the frozen selection.")
        compressed_model = model_path.read_bytes()
        if metadata.get("model_sha256") != hashlib.sha256(compressed_model).hexdigest():
            raise DashboardDataError("The saved model does not match its metadata checksum.")
        fitted_at = pd.Timestamp(metadata["fitted_at_utc"])
        end_utc = pd.Timestamp(metadata["training_end_exclusive_utc"])
        start_utc = pd.Timestamp(metadata["training_start_utc"])
        age_days = (now.normalize() - fitted_at.tz_convert(BERLIN_TIMEZONE).normalize()).days
        if age_days < 1 or end_utc >= (target_day - pd.DateOffset(days=1)).tz_convert("UTC"):
            raise DashboardDataError("The saved model has invalid fallback provenance.")
        model.load_model(bytearray(gzip.decompress(compressed_model)))
        train_count = int(metadata["train_rows"])
        dropped_days = metadata["dropped_train_days"]
    else:
        model, metadata = fit_model_snapshot(project_root, dataset, target_day, selection)
        start_utc = pd.Timestamp(metadata["training_start_utc"])
        end_utc = pd.Timestamp(metadata["training_end_exclusive_utc"])
        dropped_days = metadata["dropped_train_days"]
        fitted_at = pd.Timestamp(metadata["fitted_at_utc"])
        train_count = metadata["train_rows"]
        age_days = 0
    raw_prediction = np.asarray(model.predict(weather[features].astype("float32")))
    if not np.isfinite(raw_prediction).all():
        raise DashboardDataError("The model returned non-finite forecast values.")
    prediction = np.clip(raw_prediction, 0, 1)
    issued_at = clock().tz_convert("UTC")
    if issued_at.tz_convert(BERLIN_TIMEZONE).normalize() != now.normalize():
        raise DashboardDataError("The local issue day changed; rerun the forecast.")
    if issued_at.tz_convert(BERLIN_TIMEZONE) >= gate_closure:
        raise DashboardDataError("The day-ahead auction gate closed at 12:00 Europe/Berlin; no market-ready forecast was published.")
    output = pd.DataFrame({
        "target_time": targets.astype(str),
        "prediction": prediction,
        "capacity_mw": capacity_mw,
        "prediction_mw": prediction * capacity_mw,
        "capacity_effective_time_utc": effective_times.astype(str),
        "model": "XGBoost",
        "loss": "l1",
        "fitted_at_utc": str(fitted_at),
        "issued_at_utc": str(issued_at),
        "fit_mode": "reused" if reuse_model else "fresh",
        "model_age_days": age_days,
        "training_start_utc": str(start_utc),
        "training_end_exclusive_utc": str(end_utc),
        "weather_run_utc": str(run_day.tz_localize("UTC")),
        "train_rows": train_count,
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
