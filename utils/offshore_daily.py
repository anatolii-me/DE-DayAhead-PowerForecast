"""Refresh offshore sources, extend the rolling backtest and issue tomorrow's forecast."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from utils.offshore_dashboard import DashboardDataError
from utils.offshore_day_ahead import current_weather, fit_model_snapshot, issue_tomorrow_forecast, training_rows
from utils.offshore_final_data import (
    FIRST_NEW_DAY, assemble_dataset, fetch_generation_day, grid, refresh_sources, save_parquet,
)

PUBLIC = Path("dashboard_data")
ORIGINAL = Path(
    "data/datasets/offshore_dataset_15m_interpolated_14_points_proxy_2024-03-14_to_2026-06-11.parquet"
)
RETRY_MINUTES = 15
CUTOFF_HOUR = 11
CUTOFF_MINUTE = 15


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _generation_complete(generation: pd.DataFrame, day: pd.Timestamp) -> bool:
    values = generation.gen_wind_off_act.reindex(grid(day))
    return bool(np.isfinite(values).all() and values.ge(0).all())


def _check_public_credentials(public: Path) -> None:
    key = os.getenv("ENTSOE_API_KEY", "")
    if key:
        secret = key.encode("utf-8")
        for path in public.rglob("*"):
            if path.is_file() and secret in path.read_bytes():
                raise DashboardDataError(f"Credential value found in public artifact: {path.name}")


def wait_for_critical_sources(root: Path, generation: pd.DataFrame, end_day: pd.Timestamp,
                              features: list[str], log: dict, *, clock=None, sleep=None):
    """Retry D-2 actuals and today's weather independently up to 11:15 Berlin."""
    clock = clock or (lambda: pd.Timestamp.now(tz="Europe/Berlin"))
    sleep = sleep or time.sleep
    today = clock().tz_convert("Europe/Berlin").normalize()
    deadline = (today.tz_localize(None) + pd.Timedelta(
        hours=CUTOFF_HOUR, minutes=CUTOFF_MINUTE)).tz_localize("Europe/Berlin")
    weather = None
    generation_ready = _generation_complete(generation, end_day)
    retry_generation = bool(os.getenv("ENTSOE_API_KEY"))
    attempts = log.setdefault("attempts", [])
    while True:
        now = clock().tz_convert("Europe/Berlin")
        if now >= deadline:
            break
        if not generation_ready and retry_generation:
            try:
                generation, count = fetch_generation_day(
                    end_day, generation, root / PUBLIC / "source_cache/generation.parquet")
                generation_ready = _generation_complete(generation, end_day)
                outcome = f"received {count}/{len(grid(end_day))}; complete={generation_ready}"
            except Exception as exc:
                outcome = type(exc).__name__
            attempts.append({"source": "ENTSO-E generation", "at_utc": str(now.tz_convert("UTC")),
                             "outcome": outcome})
            print(f"D-2 generation retry: {outcome}")
        if weather is None:
            try:
                weather = current_weather(root, today.tz_localize(None), features)
                outcome = "complete"
            except Exception as exc:
                outcome = type(exc).__name__
            attempts.append({"source": "ECMWF midnight run", "at_utc": str(now.tz_convert("UTC")),
                             "outcome": outcome})
            print(f"Current weather retry: {outcome}")
        if (generation_ready or not retry_generation) and weather is not None:
            break
        remaining = (deadline - clock().tz_convert("Europe/Berlin")).total_seconds()
        if remaining <= 0:
            break
        sleep(min(RETRY_MINUTES * 60, remaining))
    return generation, weather, generation_ready


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
    target_day = today + pd.DateOffset(days=1)
    log = {"issue_day": str(today.date()), "forecast_day": str(target_day.date()),
           "started_at_utc": str(pd.Timestamp.now(tz="UTC")),
           "retry_minutes": RETRY_MINUTES, "cutoff_local": f"{CUTOFF_HOUR:02}:{CUTOFF_MINUTE:02} Europe/Berlin",
           "attempts": []}
    weather, generation, capacity, events, _ = refresh_sources(
        root, end_day=end_day, update=update, revision_days=7,
    )
    features = selection["fit_protocol"]["features"]
    current_run = None
    generation_ready = _generation_complete(generation, end_day)
    if update:
        generation, current_run, generation_ready = wait_for_critical_sources(
            root, generation, end_day, features, log)
    dataset, coverage = assemble_dataset(
        original, weather, generation, capacity, end_day, features,
        source_events=events,
    )
    latest = coverage.loc[end_day]
    extend_backtest(root, dataset, coverage, selection)
    actual = dataset[["capacity", "capacity_factor", "gen_wind_off_act"]]
    save_parquet(actual, public / "actual_dataset.parquet")
    save_parquet(coverage, public / "daily_coverage.parquet")
    if not events.empty:
        events.drop(columns=["detail"], errors="ignore").to_csv(public / "source_events.csv", index=False)
    if update:
        training_issue = None
        if bool(latest.row_complete):
            try:
                training_rows(dataset, target_day, features)
            except DashboardDataError as exc:
                training_issue = str(exc)
        mode = "fresh" if bool(latest.row_complete) and training_issue is None else "reused"
        forecast_state = "issued"
        failure_reason = None
        forward = None
        if current_run is None:
            attempted_weather = any(item["source"] == "ECMWF midnight run" for item in log["attempts"])
            forecast_state = "weather_unavailable" if attempted_weather else "deadline_missed"
            failure_reason = ("The current ECMWF midnight run remained unavailable by 11:15 Europe/Berlin."
                              if attempted_weather else "The scheduled job reached the 11:15 Europe/Berlin cutoff before source polling.")
        else:
            try:
                issued = issue_tomorrow_forecast(
                    root, dataset=dataset, weather=current_run, reuse_model=(mode == "reused"))
                forward = pd.read_parquet(issued)
                save_parquet(forward, public / "issued" / f"{target_day.date()}.parquet")
            except DashboardDataError as exc:
                forecast_state = "model_unavailable" if mode == "reused" else "fit_failed"
                failure_reason = str(exc)
                print(f"Forecast withheld: {type(exc).__name__}: {failure_reason}")
        actual_values = actual.gen_wind_off_act.dropna()
        actual_time = str(actual_values.index.max()) if not actual_values.empty else None
        model_meta_path = public / "model/metadata.json"
        model_meta = json.loads(model_meta_path.read_text(encoding="utf-8")) if model_meta_path.exists() else {}
        status = {
            "last_updated_utc": str(pd.Timestamp.now(tz="UTC")),
            "fitted_at_utc": model_meta.get("fitted_at_utc"),
            "latest_actual_utc": actual_time,
            "forecast_day": str(target_day.date()),
            "forecast_status": forecast_state,
            "fit_mode": mode if forward is not None else None,
            "model_age_days": int(forward.model_age_days.iloc[0]) if forward is not None else None,
            "training_data_status": (
                training_issue if training_issue is not None else
                "complete" if bool(latest.row_complete) else
                "ENTSO-E generation incomplete" if not generation_ready else f"D-2 incomplete: {latest.reason}"),
            "failure_reason": failure_reason,
        }
        _write_json(public / "status.json", status)
        log.update({"completed_at_utc": status["last_updated_utc"],
                    "result": forecast_state, "fit_mode": status["fit_mode"],
                    "training_data_status": status["training_data_status"],
                    "model_fitted_at_utc": status["fitted_at_utc"],
                    "training_start_utc": model_meta.get("training_start_utc"),
                    "training_end_exclusive_utc": model_meta.get("training_end_exclusive_utc"),
                    "train_rows": model_meta.get("train_rows"),
                    "dropped_train_days": model_meta.get("dropped_train_days"),
                    "failure_reason": failure_reason})
        _write_json(public / "run_logs" / f"{today.date()}.json", log)
    _check_public_credentials(public)
    print(f"Complete source days: {int(coverage.row_complete.sum())}/{len(coverage)}; latest {end_day.date()}")


def bootstrap_model(root: Path) -> None:
    """Create an initial saved model from existing complete cached data."""
    root = root.resolve()
    public = root / PUBLIC
    selection = json.loads((public / "frozen_selection.json").read_text(encoding="utf-8"))
    original_path = root / ORIGINAL
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != selection["dataset_hash"]:
        raise DashboardDataError("The original dataset differs from the frozen model selection.")
    today = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None).normalize()
    end_day = today - pd.DateOffset(days=1)
    weather, generation, capacity, events, _ = refresh_sources(
        root, end_day=end_day, update=False)
    dataset, coverage = assemble_dataset(
        pd.read_parquet(original_path), weather, generation, capacity, end_day,
        selection["fit_protocol"]["features"], source_events=events)
    if not bool(coverage.loc[end_day].row_complete):
        raise DashboardDataError(f"Cannot bootstrap: {end_day.date()} is incomplete.")
    _, metadata = fit_model_snapshot(root, dataset, today + pd.DateOffset(days=1), selection)
    updated = str(pd.Timestamp.now(tz="UTC"))
    actuals = dataset.gen_wind_off_act.dropna()
    status = {
        "last_updated_utc": updated,
        "fitted_at_utc": metadata["fitted_at_utc"],
        "latest_actual_utc": str(actuals.index.max()) if not actuals.empty else None,
        "forecast_day": str((today + pd.DateOffset(days=1)).date()),
        "forecast_status": "bootstrap_only", "fit_mode": None,
        "model_age_days": None, "training_data_status": "complete",
        "failure_reason": None,
    }
    _write_json(public / "status.json", status)
    _write_json(public / "run_logs" / f"{today.date()}.json", {
        "issue_day": str(today.date()), "forecast_day": status["forecast_day"],
        "started_at_utc": metadata["fitted_at_utc"], "completed_at_utc": updated,
        "result": "bootstrap_only", "fit_mode": None,
        "training_data_status": "complete", "model_fitted_at_utc": metadata["fitted_at_utc"],
        "training_start_utc": metadata["training_start_utc"],
        "training_end_exclusive_utc": metadata["training_end_exclusive_utc"],
        "train_rows": metadata["train_rows"],
        "dropped_train_days": metadata["dropped_train_days"],
        "attempts": [], "failure_reason": None,
    })
    _check_public_credentials(public)
    print(f"Bootstrapped model fitted at {metadata['fitted_at_utc']}; training ends {metadata['training_end_exclusive_utc']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--offline", action="store_true", help="Use cached sources and do not issue a forward forecast")
    parser.add_argument("--bootstrap-model", action="store_true", help="Fit the initial reusable model from cached data")
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(args.project_root / ".env", override=False)
    if args.bootstrap_model:
        bootstrap_model(args.project_root)
    else:
        run(args.project_root, update=not args.offline)


if __name__ == "__main__":
    main()
