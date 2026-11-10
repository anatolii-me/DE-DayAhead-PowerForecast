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
CUTOFF_MINUTE = 30


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
    """Legacy in-process polling helper; scheduled attempts use run() instead."""
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
    existing_days = set()
    if not existing.empty:
        for day, rows in existing.groupby("day"):
            times = pd.DatetimeIndex(pd.to_datetime(rows.target_time, utc=True)).sort_values()
            if times.equals(grid(day)) and np.isfinite(rows.prediction).all():
                existing_days.add(str(day))
    config = selection["best_by_model"]["XGBoost"]
    features = selection["fit_protocol"]["features"]
    seed = selection["deployment_seed"]
    additions = []
    for day in coverage.index[coverage.weather_available & coverage.capacity_available]:
        day = pd.Timestamp(day)
        if str(day.date()) in existing_days:
            continue
        issued_path = root / PUBLIC / "issued" / f"{day.date()}.parquet"
        reconstructed_path = root / PUBLIC / "reconstructed" / f"{day.date()}.parquet"
        was_issued = issued_path.exists()
        if not was_issued and reconstructed_path.exists():
            issued_path = reconstructed_path
        if issued_path.exists():
            issued = pd.read_parquet(issued_path)
            expected = grid(day)
            times = pd.DatetimeIndex(pd.to_datetime(issued.target_time, utc=True))
            if not times.equals(expected):
                raise DashboardDataError(f"Issued forecast for {day.date()} has an incomplete grid.")
            raw_prediction = issued.prediction.to_numpy(dtype=float)
            provenance = "issued_live" if was_issued and not issued.get("simulation", pd.Series(False)).any() else "reconstructed"
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
            provenance = "reconstructed"
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


def recover_current_day(root: Path, dataset: pd.DataFrame, day: pd.Timestamp,
                        selection: dict) -> None:
    """Display today's issued forecast, or reconstruct it without today's labels.

    This never changes the operational model or tomorrow's forecast. Recovered
    values retain their provenance when actuals later become available.
    """
    from utils.offshore_day_ahead import capacity_for_targets
    from utils.offshore_final_evaluation import build_model, make_training_weights

    public = root / PUBLIC
    targets = grid(day)
    issued = public / "issued" / f"{day.date()}.parquet"
    recovered = public / "reconstructed" / f"{day.date()}.parquet"
    try:
        if issued.exists() or recovered.exists():
            frame = pd.read_parquet(issued if issued.exists() else recovered)
            times = pd.DatetimeIndex(pd.to_datetime(frame.target_time, utc=True))
            if not times.equals(targets) or not np.isfinite(frame.prediction).all():
                raise DashboardDataError("Incomplete saved current-day forecast")
            frame["provenance"] = "issued_live" if issued.exists() and not frame.get("simulation", pd.Series(False)).any() else "reconstructed"
        else:
            features = selection["fit_protocol"]["features"]
            train, start, end, dropped = training_rows(dataset, day, features)
            weather = current_weather(root, day - pd.DateOffset(days=1), features)
            config = selection["best_by_model"]["XGBoost"]
            model = build_model(config, selection["deployment_seed"])
            y = train.capacity_factor.to_numpy(dtype=float)
            model.fit(train[features].astype("float32"), y,
                      sample_weight=make_training_weights(y, config))
            raw = np.asarray(model.predict(weather[features].astype("float32")))
            if not np.isfinite(raw).all():
                raise DashboardDataError("Non-finite reconstructed predictions")
            capacity, _ = capacity_for_targets(pd.read_parquet(public / "source_cache/capacity.parquet"), targets)
            frame = pd.DataFrame({"target_time": targets.astype(str), "prediction": np.clip(raw, 0, 1),
                                  "capacity_mw": capacity, "provenance": "reconstructed",
                                  "training_start_utc": str(start), "training_end_exclusive_utc": str(end),
                                  "weather_run_utc": str((day - pd.DateOffset(days=1)).tz_localize("UTC")),
                                  "reconstructed_at_utc": str(pd.Timestamp.now(tz="UTC"))})
            save_parquet(frame, recovered)
        save_parquet(frame, public / "pending_predictions.parquet")
        result = {"day": str(day.date()), "result": str(frame.provenance.iloc[0])}
    except Exception as exc:
        result = {"day": str(day.date()), "result": "unavailable", "error_type": type(exc).__name__}
        print(f"Current-day recovery unavailable: {type(exc).__name__}")
    _write_json(public / "run_logs" / f"{day.date()}_recovery.json", result)


def refresh_scored_actuals(root: Path, dataset: pd.DataFrame) -> None:
    """Synchronize revised labels without retraining or changing predictions."""
    path = root / PUBLIC / "latest_predictions.parquet"
    if not path.exists():
        return
    existing = pd.read_parquet(path)
    times = pd.DatetimeIndex(pd.to_datetime(existing.target_time, utc=True))
    labels = dataset.capacity_factor.reindex(times).to_numpy(dtype=float)
    available = np.isfinite(labels)
    existing.loc[available, "actual"] = labels[available]
    save_parquet(existing, path)


def refresh_display_actuals(root: Path, generation: pd.DataFrame, now: pd.Timestamp,
                            log: dict) -> pd.Timestamp | None:
    """Fetch intraday observations for display only; never return training rows."""
    day = now.tz_convert("Europe/Berlin").tz_localize(None).normalize()
    public = root / PUBLIC
    try:
        observed, _ = fetch_generation_day(day, generation, public / "source_cache/generation.parquet")
        # ENTSO-E timestamps mark interval starts. Only completed intervals
        # at the logical issue time may be shown, including historical replay.
        observed = observed.reindex(grid(day))
        observed = observed.loc[(observed.index + pd.Timedelta(minutes=15) <= now)
                                & np.isfinite(observed.gen_wind_off_act)
                                & observed.gen_wind_off_act.ge(0)].copy()
        observed["as_of_utc"] = str(now.tz_convert("UTC"))
        save_parquet(observed, public / "display_actuals.parquet")
        outcome = f"received {len(observed)} completed intraday intervals"
        latest = observed.index.max() if not observed.empty else None
    except Exception as exc:
        outcome = type(exc).__name__
        latest = None
    log.setdefault("attempts", []).append({"source": "ENTSO-E display actuals",
                                           "at_utc": str(now), "outcome": outcome})
    return latest


def run(root: Path, *, update: bool = True, clock=None, simulation: bool = False,
        backtest: bool = False) -> None:
    root = root.resolve()
    public = root / PUBLIC
    clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))
    now = clock().tz_convert("Europe/Berlin")
    today = now.tz_localize(None).normalize()
    target_day = today + pd.DateOffset(days=1)
    if update and not simulation:
        if now.tz_convert("UTC").hour < 5 or (now.hour, now.minute) >= (11, 30):
            print("Outside the morning publication window; no changes.")
            return
        status_path = public / "status.json"
        if status_path.exists():
            previous = json.loads(status_path.read_text())
            if (previous.get("forecast_day") == str(target_day.date())
                    and previous.get("forecast_status") == "issued"
                    and previous.get("fit_mode") == "fresh"
                    and (public / "issued" / f"{target_day.date()}.parquet").exists()):
                print("Fresh forecast already published; no changes.")
                return
    selection = json.loads((public / "frozen_selection.json").read_text(encoding="utf-8"))
    original_path = root / ORIGINAL
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != selection["dataset_hash"]:
        raise DashboardDataError("The original training dataset differs from the frozen model selection.")
    original = pd.read_parquet(original_path)
    # Issue day is D-1. Training ends at D-2; display-only actuals are separate.
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
        # One bounded attempt per invocation. The scheduler supplies retries,
        # including retries of capacity and historical feature gaps.
        if simulation or clock().tz_convert("Europe/Berlin").time() < pd.Timestamp("11:30").time():
            try:
                current_run = current_weather(root, today, features)
                outcome = "complete"
            except Exception as exc:
                outcome = type(exc).__name__
            log["attempts"].append({"source": "ECMWF midnight run",
                                    "at_utc": str(clock()), "outcome": outcome})
    dataset, coverage = assemble_dataset(
        original, weather, generation, capacity, end_day, features,
        source_events=events,
    )
    latest = coverage.loc[end_day]
    actual = dataset[["capacity", "capacity_factor", "gen_wind_off_act"]]
    refresh_scored_actuals(root, dataset)
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
            failure_reason = ("The current ECMWF midnight run is unavailable; retry within the morning window."
                              if attempted_weather else "The job reached the 11:30 Europe/Berlin cutoff before source polling.")
        else:
            try:
                issued = issue_tomorrow_forecast(
                    root, dataset=dataset, weather=current_run, reuse_model=(mode == "reused"), clock=clock)
                forward = pd.read_parquet(issued)
                if simulation:
                    forward["simulation"] = True
                    forward["executed_at_utc"] = str(pd.Timestamp.now(tz="UTC"))
                    save_parquet(forward, issued)
                save_parquet(forward, public / "issued" / f"{target_day.date()}.parquet")
            except DashboardDataError as exc:
                forecast_state = "model_unavailable" if mode == "reused" else "fit_failed"
                failure_reason = str(exc)
                print(f"Forecast withheld: {type(exc).__name__}: {failure_reason}")
        actual_values = actual.gen_wind_off_act.dropna()
        actual_time = str(actual_values.index.max()) if not actual_values.empty else None
        if forward is not None:
            display_time = refresh_display_actuals(root, generation, clock(), log)
            if display_time is not None:
                actual_time = str(display_time)
        model_meta_path = public / "model/metadata.json"
        model_meta = json.loads(model_meta_path.read_text(encoding="utf-8")) if model_meta_path.exists() else {}
        status = {
            "simulation": simulation,
            "issue_day": str(today.date()),
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
                    "simulation": simulation,
                    "source_events": events.drop(columns=["detail"], errors="ignore").to_dict("records"),
                    "result": forecast_state, "fit_mode": status["fit_mode"],
                    "training_data_status": status["training_data_status"],
                    "model_fitted_at_utc": status["fitted_at_utc"],
                    "training_start_utc": model_meta.get("training_start_utc"),
                    "training_end_exclusive_utc": model_meta.get("training_end_exclusive_utc"),
                    "train_rows": model_meta.get("train_rows"),
                    "dropped_train_days": model_meta.get("dropped_train_days"),
                    "failure_reason": failure_reason})
        _write_json(public / "run_logs" / f"{today.date()}.json", log)
        _write_json(public / "run_logs" / f"{today.date()}_{pd.Timestamp.now(tz='UTC').strftime('%H%M%S%f')}.json", log)
    # Historical scoring must never delay today's publication. It is opt-in
    # and can be run separately with --offline --backtest.
    if backtest:
        extend_backtest(root, dataset, coverage, selection)
        recover_current_day(root, dataset, today, selection)
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
    parser.add_argument("--backtest", action="store_true", help="Extend retrospective scores after forecasting")
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(args.project_root / ".env", override=False)
    if args.bootstrap_model:
        bootstrap_model(args.project_root)
    else:
        try:
            run(args.project_root, update=not args.offline, backtest=args.backtest)
        except Exception as exc:
            # Persist a sanitized diagnostic even when fitting or assembly fails.
            # API exception messages can contain credential-bearing request URLs.
            now = pd.Timestamp.now(tz="UTC")
            issue_day = str(now.tz_convert("Europe/Berlin").date())
            _write_json(args.project_root / PUBLIC / "run_logs" /
                        f"{issue_day}_{now.strftime('%H%M%S%f')}_error.json",
                        {"issue_day": issue_day, "at_utc": str(now),
                         "result": "pipeline_error", "error_type": type(exc).__name__})
            raise SystemExit(f"Offshore pipeline failed: {type(exc).__name__}") from None


if __name__ == "__main__":
    main()
