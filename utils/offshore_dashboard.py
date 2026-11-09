"""Data and chart helpers for the offshore forecasting dashboard.

The dashboard deliberately reads the predictions produced by notebook 6 rather
than fitting a model itself.  Keeping presentation separate from evaluation
makes it safe to run in a lightweight GitHub Actions check or a web process.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


BERLIN_TIMEZONE = "Europe/Berlin"
MODEL_LABEL = "XGBoost HPO (L1)"

METRIC_ORDER = ["nMAE", "nRMSE", "R²", "Skill score", "MAE (MW)", "P90 error (MW)"]
WINDOW_ORDER = ["Last year", "All history"]


class DashboardDataError(RuntimeError):
    """Raised when notebook-6 artifacts needed by the dashboard are absent."""


@dataclass(frozen=True)
class DashboardPaths:
    """Locations of the immutable and refreshed prediction artifacts."""

    historical_predictions: Path
    latest_predictions: Path
    actual_dataset: Path | None
    forward_predictions: Path | None = None

    @classmethod
    def from_project_root(cls, project_root: Path) -> "DashboardPaths":
        """Build paths with optional environment overrides for deployment."""
        public = project_root / "dashboard_data"
        historical_default = public / "historical_predictions.parquet"
        latest_default = public / "latest_predictions.parquet"
        forward_default = public / "day_ahead_predictions.parquet"
        dataset_candidates = [
            path for path in (project_root / "data/datasets").glob(
                "offshore_cf_15m_ecmwf14p_engineered_*_to_*.parquet"
            )
            if not path.stem.endswith("_daily_coverage")
        ]
        refreshed_dataset = public / "actual_dataset.parquet"
        if refreshed_dataset.exists():
            dataset_candidates.append(refreshed_dataset)
        def coverage_key(path: Path) -> tuple[pd.Timestamp, int]:
            # A notebook refresh can extend the result parquet beyond the
            # newest dated base snapshot without renaming that base file.
            try:
                last_target = pd.read_parquet(path, columns=[]).index.max()
            except Exception as error:
                raise DashboardDataError(f"Cannot read offshore dataset '{path}': {error}") from error
            return pd.Timestamp(last_target), path.stat().st_mtime_ns
        actual_default = (
            max(dataset_candidates, key=coverage_key)
            if dataset_candidates
            else refreshed_dataset
        )

        def configured_path(name: str, default: Path) -> Path:
            configured = os.getenv(name)
            if not configured:
                return default
            candidate = Path(configured)
            return candidate if candidate.is_absolute() else project_root / candidate

        return cls(
            historical_predictions=configured_path(
                "OFFSHORE_HISTORICAL_PREDICTIONS", historical_default
            ),
            latest_predictions=configured_path(
                "OFFSHORE_LATEST_PREDICTIONS", latest_default
            ),
            actual_dataset=configured_path("OFFSHORE_ACTUAL_DATASET", actual_default),
            forward_predictions=configured_path(
                "OFFSHORE_FORWARD_PREDICTIONS", forward_default
            ),
        )


def _read_parquet(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise DashboardDataError(
            f"{description} was not found at '{path}'. "
            "Run notebook 6 to create the dashboard artifacts, or set the matching "
            "OFFSHORE_* environment variable."
        )
    return pd.read_parquet(path)


def _check_columns(frame: pd.DataFrame, required: set[str], description: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        formatted = ", ".join(sorted(missing))
        raise DashboardDataError(f"{description} is missing required columns: {formatted}.")


def _normalise_historical_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    _check_columns(
        frame,
        {"target_time", "actual", "prediction", "raw_prediction", "seed"},
        "Historical XGBoost predictions",
    )
    # Notebook 6 uses deployment seed 1337.  Filter explicitly so a later
    # experiment cannot silently become the production chart.
    if "seed" in frame and frame.seed.eq(1337).any():
        frame = frame.loc[frame.seed.eq(1337)].copy()
    else:
        frame = frame.copy()
    frame["model"] = MODEL_LABEL
    frame["source_period"] = "Frozen development and first holdout"
    return frame


def _normalise_latest_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    _check_columns(
        frame,
        {"target_time", "actual", "prediction", "raw_prediction", "model", "loss"},
        "Latest notebook-6 predictions",
    )
    selected = frame.loc[
        frame.model.eq("XGBoost") & frame.loss.eq("l1")
    ].copy()
    if selected.empty:
        raise DashboardDataError(
            "Latest predictions do not contain the selected XGBoost L1 model."
        )
    if "seed" in selected and selected.seed.eq(1337).any():
        selected = selected.loc[selected.seed.eq(1337)].copy()
    selected["model"] = MODEL_LABEL
    selected["source_period"] = "Refreshed holdout"
    return selected


def _timestamp_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Put a canonical UTC target timestamp on the index and validate it."""
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result["target_time"], utc=True))
    result.index.name = "target_time_utc"
    if result.index.isna().any():
        raise DashboardDataError("Predictions contain an invalid target timestamp.")
    return result.sort_index()


def _load_actual_dataset(path: Path | None) -> pd.DataFrame:
    """Load capacity (GW), actual CF and actual MW from the matching snapshot."""
    if path is None or not path.exists():
        raise DashboardDataError(
            "The actual dataset with installed capacity is missing. "
            "Run notebook 6 or set OFFSHORE_ACTUAL_DATASET to its output parquet file."
        )
    actuals = pd.read_parquet(
        path, columns=["capacity", "capacity_factor", "gen_wind_off_act"]
    )
    actuals.index = pd.DatetimeIndex(pd.to_datetime(actuals.index, utc=True))
    actuals = actuals.loc[~actuals.index.duplicated(keep="last")]
    return actuals.sort_index()


def week_ago_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the same local quarter-hour seven calendar days earlier.

    A calendar offset rather than a fixed 168-hour offset preserves the local
    clock time across daylight-saving changes in Germany.
    """
    local_index = index.tz_convert(BERLIN_TIMEZONE)
    previous_local_clock = local_index.tz_localize(None) - pd.DateOffset(days=7)
    # A spring-forward Sunday has no 02:00–02:45 local observations.  There is
    # no honest same-clock-time persistence value for forecasts a week later,
    # so retain NaT and exclude only those rows from skill calculation.
    return pd.DatetimeIndex(
        previous_local_clock.tz_localize(
            BERLIN_TIMEZONE, ambiguous="NaT", nonexistent="NaT"
        ).tz_convert("UTC")
    )


def load_dashboard_predictions(paths: DashboardPaths) -> pd.DataFrame:
    """Combine notebook-6 history and refreshed production-model predictions.

    The returned frame is indexed by UTC target time and includes the seasonal
    weekly-persistence baseline.  The baseline is never used to create a model
    prediction; it is only calculated after actuals are available for scoring.
    """
    historical = _normalise_historical_predictions(
        _read_parquet(paths.historical_predictions, "Historical predictions")
    )
    latest = _normalise_latest_predictions(
        _read_parquet(paths.latest_predictions, "Latest predictions")
    )
    predictions = pd.concat([historical, latest], ignore_index=True)
    predictions = _timestamp_index(predictions)

    # Refreshed values win if a rerun ever overlaps the frozen snapshot.
    predictions = predictions.loc[~predictions.index.duplicated(keep="last")]
    _check_columns(predictions, {"actual", "prediction"}, "Dashboard predictions")
    if predictions.empty:
        raise DashboardDataError("No XGBoost HPO predictions are available for the dashboard.")

    actual_dataset = _load_actual_dataset(paths.actual_dataset)
    # Keep the ENTSO-E actual curve visible even on days where an ECMWF run
    # could not be scored. Those intervals have no model prediction or metric.
    observed = actual_dataset.loc[predictions.index.min():]
    observed = observed.loc[
        np.isfinite(observed[["capacity", "capacity_factor", "gen_wind_off_act"]]).all(axis=1)
        & observed.capacity.gt(0)
    ]
    predictions = predictions.reindex(predictions.index.union(observed.index).sort_values())
    prediction_actuals = pd.Series(
        predictions.actual.to_numpy(), index=predictions.index, name="capacity_factor"
    )
    predictions["actual"] = predictions.actual.combine_first(
        actual_dataset.capacity_factor.reindex(predictions.index)
    )
    capacity = actual_dataset.capacity.reindex(predictions.index)
    if capacity.isna().any() or (~np.isfinite(capacity)).any() or capacity.le(0).any():
        missing = int((capacity.isna() | ~np.isfinite(capacity) | capacity.le(0)).sum())
        raise DashboardDataError(
            f"Installed capacity is unavailable for {missing:,} forecast intervals. "
            "Refresh the full offshore dataset or point OFFSHORE_ACTUAL_DATASET "
            "to the dataset matching the latest predictions."
        )
    predictions["capacity_mw"] = capacity.to_numpy(dtype=float) * 1000.0
    predictions["actual_mw"] = predictions.actual.to_numpy(dtype=float) * predictions.capacity_mw
    predictions["prediction_mw"] = predictions.prediction.to_numpy(dtype=float) * predictions.capacity_mw
    actual_source_mw = actual_dataset.gen_wind_off_act.reindex(predictions.index)
    comparable = actual_source_mw.notna()
    if comparable.any() and not np.allclose(
        predictions.loc[comparable, "actual_mw"], actual_source_mw.loc[comparable],
        rtol=0, atol=0.01,
    ):
        raise DashboardDataError(
            "Historical actual CF, installed capacity and source generation do not "
            "reconstruct the same MW values. Use the matching dataset snapshot."
        )

    # The source snapshot may be older than the latest evaluated predictions.
    # Prediction rows carry their own observed CF; fill only missing dates.
    actual_history = actual_dataset.capacity_factor.combine_first(prediction_actuals).sort_index()

    previous_week = week_ago_index(predictions.index)
    predictions["weekly_persistence"] = actual_history.reindex(previous_week).to_numpy()
    # Scale the week-old CF by *target-time* installed capacity. This keeps the
    # baseline comparable with the MW forecast when the fleet size changes.
    predictions["weekly_persistence_mw"] = (
        predictions.weekly_persistence.to_numpy(dtype=float) * predictions.capacity_mw
    )
    predictions["local_time"] = predictions.index.tz_convert(BERLIN_TIMEZONE)
    return predictions


def load_forward_predictions(
    paths: DashboardPaths, *, now: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    """Read a separately issued, unscored day-ahead forecast if it is current.

    Historical holdout predictions always have observed labels. A forward
    forecast must come from the dedicated artifact with a complete local-day
    grid and explicit refit provenance; old holdout rows are never relabelled
    as a live forecast.
    """
    path = paths.forward_predictions
    if path is None or not path.exists():
        return None
    frame = _read_parquet(path, "Forward day-ahead predictions")
    required = {
        "target_time", "prediction", "capacity_mw", "model", "loss",
        "fitted_at_utc", "training_start_utc", "training_end_exclusive_utc",
        "weather_run_utc",
    }
    _check_columns(frame, required, "Forward day-ahead predictions")
    if frame.empty:
        raise DashboardDataError("Forward day-ahead predictions are empty.")
    frame = _timestamp_index(frame)
    if frame.index.has_duplicates:
        raise DashboardDataError("Forward day-ahead target timestamps repeat.")
    local_days = frame.index.tz_convert(BERLIN_TIMEZONE).normalize().unique()
    if len(local_days) != 1:
        raise DashboardDataError("Forward predictions must cover exactly one local delivery day.")
    target_day = local_days[0]
    current = pd.Timestamp.now(tz=BERLIN_TIMEZONE) if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    today = current.tz_convert(BERLIN_TIMEZONE).normalize()
    if target_day < today or target_day > today + pd.DateOffset(days=1):
        return None
    expected = pd.date_range(
        target_day, target_day + pd.DateOffset(days=1),
        freq="15min", inclusive="left",
    ).tz_convert("UTC")
    if not frame.index.equals(expected):
        raise DashboardDataError(
            "Forward predictions do not contain every 15-minute interval of "
            "their local delivery day."
        )
    if not frame.model.eq("XGBoost").all() or not frame.loss.eq("l1").all():
        raise DashboardDataError("Forward predictions are not from XGBoost HPO (L1).")
    for column in ("prediction", "capacity_mw"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if (
        not np.isfinite(frame[["prediction", "capacity_mw"]].to_numpy(dtype=float)).all()
        or not frame.prediction.between(0, 1).all()
        or not frame.capacity_mw.gt(0).all()
    ):
        raise DashboardDataError("Forward predictions contain invalid CF or capacity values.")
    metadata = {}
    for column in (
        "fitted_at_utc", "training_start_utc", "training_end_exclusive_utc",
        "weather_run_utc",
    ):
        parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if parsed.isna().any() or parsed.nunique() != 1:
            raise DashboardDataError(f"Forward forecast has invalid {column} metadata.")
        metadata[column] = parsed.iloc[0]
        frame[column] = parsed
    mode = str(frame.fit_mode.iloc[0]) if "fit_mode" in frame else "fresh"
    if mode not in {"fresh", "reused"} or ("fit_mode" in frame and not frame.fit_mode.eq(mode).all()):
        raise DashboardDataError("Forward forecast has invalid model reuse metadata.")
    issued_at = metadata["fitted_at_utc"]
    if "issued_at_utc" in frame:
        parsed_issue = pd.to_datetime(frame.issued_at_utc, utc=True, errors="coerce")
        if parsed_issue.isna().any() or parsed_issue.nunique() != 1:
            raise DashboardDataError("Forward forecast has invalid issue time.")
        issued_at = parsed_issue.iloc[0]
        frame["issued_at_utc"] = parsed_issue
    fitted_day = metadata["fitted_at_utc"].tz_convert(BERLIN_TIMEZONE).normalize()
    cutoff = fitted_day.tz_convert("UTC")
    start = (fitted_day - pd.DateOffset(days=365)).tz_convert("UTC")
    if metadata["training_start_utc"] != start or metadata["training_end_exclusive_utc"] != cutoff:
        raise DashboardDataError("Forward forecast training window is not the notebook-6 365-day window.")
    issue_day = target_day - pd.DateOffset(days=1)
    if issued_at.tz_convert(BERLIN_TIMEZONE).normalize() != issue_day:
        raise DashboardDataError("Forward forecast was not issued on D-1.")
    if (mode == "fresh" and fitted_day != issue_day) or (mode == "reused" and fitted_day >= issue_day):
        raise DashboardDataError("Forward forecast has inconsistent model age.")
    if "model_age_days" in frame and not frame.model_age_days.eq((issue_day - fitted_day).days).all():
        raise DashboardDataError("Forward forecast model age is inconsistent.")
    gate_closure = ((target_day - pd.DateOffset(days=1)).tz_localize(None)
                    + pd.Timedelta(hours=12)).tz_localize(BERLIN_TIMEZONE)
    if issued_at.tz_convert(BERLIN_TIMEZONE) >= gate_closure:
        raise DashboardDataError("The forward forecast was issued after the day-ahead auction gate closed.")
    if metadata["fitted_at_utc"] > issued_at or (mode == "fresh" and
            metadata["fitted_at_utc"].tz_convert(BERLIN_TIMEZONE) >= gate_closure):
        raise DashboardDataError("Forward forecast model fit time is inconsistent with issuance.")
    # The source run is at 00:00 UTC on D-1; local and UTC midnights differ.
    expected_run = (pd.Timestamp(target_day.date()) - pd.Timedelta(days=1)).tz_localize("UTC")
    if metadata["weather_run_utc"] != expected_run:
        raise DashboardDataError("Forward forecast does not use the D-1 weather run.")
    frame["prediction_mw"] = frame.prediction * frame.capacity_mw
    frame["actual"] = np.nan
    frame["actual_mw"] = np.nan
    frame["local_time"] = frame.index.tz_convert(BERLIN_TIMEZONE)
    return frame


def calculate_period_metrics(predictions: pd.DataFrame) -> dict[str, float | int | None]:
    """Calculate normalized CF, MW error and weekly-persistence skill metrics."""
    scored = predictions.dropna(subset=["actual", "prediction"])
    if scored.empty:
        return {
            "nMAE": None,
            "nRMSE": None,
            "R²": None,
            "Skill score": None,
            "MAE (MW)": None,
            "P90 error (MW)": None,
            "observations": 0,
            "baseline_observations": 0,
        }

    actual = scored.actual.to_numpy(dtype=float)
    forecast = scored.prediction.to_numpy(dtype=float)
    error = forecast - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    centered = actual - actual.mean()
    denominator = float(np.sum(centered**2))
    r_squared = float(1 - np.sum(error**2) / denominator) if denominator > 0 else None

    mw_error = scored.prediction_mw.to_numpy(dtype=float) - scored.actual_mw.to_numpy(dtype=float)
    paired = scored.dropna(subset=["weekly_persistence_mw"])
    skill = None
    if not paired.empty:
        baseline_mae = float(
            np.mean(np.abs(paired.weekly_persistence_mw.to_numpy(dtype=float) - paired.actual_mw.to_numpy(dtype=float)))
        )
        if baseline_mae > 0:
            model_mae = float(
                np.mean(np.abs(paired.prediction_mw.to_numpy(dtype=float) - paired.actual_mw.to_numpy(dtype=float)))
            )
            skill = float(1 - model_mae / baseline_mae)

    return {
        "nMAE": mae,
        "nRMSE": rmse,
        "R²": r_squared,
        "Skill score": skill,
        "MAE (MW)": float(np.mean(np.abs(mw_error))),
        "P90 error (MW)": float(np.quantile(np.abs(mw_error), 0.9)),
        "observations": int(len(scored)),
        "baseline_observations": int(len(paired)),
    }


def metric_windows(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return metrics for the last year and all evaluated history."""
    if predictions.empty:
        raise DashboardDataError("The dashboard cannot calculate metrics from an empty frame.")
    # Use full local delivery dates, including 92/100 intervals on DST dates.
    local_days = predictions.index.tz_convert(BERLIN_TIMEZONE).normalize()
    latest_day = local_days.max()
    windows: dict[str, pd.DataFrame] = {
        "Last year": predictions.loc[local_days >= latest_day - pd.DateOffset(days=364)],
        "All history": predictions,
    }
    records = []
    for window, frame in windows.items():
        metrics = calculate_period_metrics(frame)
        for metric in METRIC_ORDER:
            records.append(
                {
                    "window": window,
                    "metric": metric,
                    "value": metrics[metric],
                    "observations": metrics["observations"],
                    "baseline_observations": metrics["baseline_observations"],
                }
            )
    result = pd.DataFrame.from_records(records)

    result["window"] = pd.Categorical(result.window, WINDOW_ORDER, ordered=True)
    result["metric"] = pd.Categorical(result.metric, METRIC_ORDER, ordered=True)
    return result.sort_values(["metric", "window"]).reset_index(drop=True)


def format_metric(metric: str, value: float | None) -> str:
    """Format metrics compactly without concealing the normalised CF scale."""
    if value is None or pd.isna(value):
        return "—"
    if metric == "Skill score":
        return f"{float(value):+.1%}"
    if metric == "R²":
        return f"{float(value):.3f}"
    if metric.endswith("(MW)"):
        return f"{float(value):,.0f} MW"
    return f"{float(value):.3f}"


def build_forecast_figure(
    predictions: pd.DataFrame, *, dark_mode: bool = False,
    labels: dict[str, str] | None = None,
    forward: pd.DataFrame | None = None,
) -> go.Figure:
    """Build an MW chart with a distinct, unscored forward segment when issued."""
    if predictions.empty:
        raise DashboardDataError("There are no predictions to plot.")
    has_forward = forward is not None and not forward.empty
    latest_time = max(predictions.index.max(), forward.index.max()) if has_forward else predictions.index.max()
    timeline = pd.date_range(predictions.index.min(), latest_time, freq="15min", tz="UTC")
    actual_mw = predictions.actual_mw.reindex(timeline)
    prediction_mw = predictions.prediction_mw.reindex(timeline)
    local_time = timeline.tz_convert(BERLIN_TIMEZONE)
    last_day = latest_time.tz_convert(BERLIN_TIMEZONE).normalize()
    start = last_day - pd.DateOffset(days=6)
    end = last_day + pd.DateOffset(days=1)
    foreground = "#F3F7F6" if dark_mode else "#1B2524"
    grid = "#344644" if dark_mode else "#E2E9E7"
    surface = "#182422" if dark_mode else "#FFFFFF"
    labels = labels or {}

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=local_time,
            y=actual_mw,
            name=labels.get("chart_actual", "Actual generation"),
            legendrank=1,
            mode="lines",
            line={"color": "#687573" if dark_mode else "#43504E", "width": 1.6},
            connectgaps=False,
            hovertemplate=("%{x|%d %b %Y, %H:%M}<br>"
                           + labels.get("chart_actual_hover", "Actual")
                           + ": %{y:,.0f} MW<extra></extra>"),
        )
    )
    observed = predictions.actual_mw.dropna()
    if not observed.empty:
        figure.add_vline(
            x=observed.index.max().tz_convert(BERLIN_TIMEZONE),
            line_color="#C68430" if dark_mode else "#A76512",
            line_width=1.5, line_dash="dash",
            annotation_text=labels.get("chart_cutoff", "Last actual data"),
            annotation_position="top left",
        )
    if has_forward:
        figure.add_trace(
            go.Scatter(
                x=forward.index.tz_convert(BERLIN_TIMEZONE),
                y=forward.prediction_mw,
                name=labels.get("chart_forward", "Forecast · actual pending"),
                legendrank=3,
                mode="lines",
                line={"color": "#42D7B3" if dark_mode else "#0B9F82", "width": 2.8},
                connectgaps=False,
                hovertemplate=("%{x|%d %b %Y, %H:%M}<br>"
                               + labels.get("chart_forward_hover", "Forecast · actual pending")
                               + ": %{y:,.0f} MW<extra></extra>"),
            )
        )
        target_day = forward.index.min().tz_convert(BERLIN_TIMEZONE).normalize()
        issue_day = target_day - pd.DateOffset(days=1)
        band = "#B7C0BD" if dark_mode else "#778581"
        figure.add_vrect(
            x0=issue_day, x1=target_day, fillcolor=band, opacity=0.10,
            line_width=0, layer="below",
        )
        figure.add_vrect(
            x0=target_day, x1=target_day + pd.DateOffset(days=1),
            fillcolor=band, opacity=0.18, line_width=0, layer="below",
        )
    figure.add_trace(
        go.Scatter(
            x=local_time,
            y=prediction_mw,
            name=labels.get("chart_forecast", "Forecast · XGBoost HPO"),
            legendrank=2,
            mode="lines",
            line={"color": "#25B899" if dark_mode else "#078B72", "width": 2.1},
            connectgaps=False,
            hovertemplate=("%{x|%d %b %Y, %H:%M}<br>"
                           + labels.get("chart_forecast_hover", "Forecast")
                           + ": %{y:,.0f} MW<extra></extra>"),
        )
    )
    figure.update_layout(
        template="plotly_dark" if dark_mode else "plotly_white",
        height=500,
        margin={"l": 28, "r": 20, "t": 20, "b": 20, "autoexpand": True},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=surface,
        font={"color": foreground},
        dragmode="zoom",
        showlegend=True,
    )
    figure.update_xaxes(
        title=None,
        range=[start, end],
        rangeslider={"visible": True, "thickness": 0.08, "bgcolor": surface, "bordercolor": grid},
        showgrid=True,
        gridcolor=grid,
    )
    figure.update_yaxes(
        title=labels.get("chart_y_axis", "Generation (MW)"), rangemode="tozero",
        tickformat=",.0f", gridcolor=grid,
    )
    return figure
