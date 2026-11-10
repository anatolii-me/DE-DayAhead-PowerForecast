"""Streamlit dashboard for German day-ahead renewable-generation forecasts."""

from __future__ import annotations

import html
import json
import os
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from utils.offshore_dashboard import (
    DashboardDataError, DashboardPaths, METRIC_ORDER, WINDOW_ORDER,
    BERLIN_TIMEZONE, build_forecast_figure, format_metric,
    load_dashboard_predictions, load_forward_predictions, metric_windows,
)

PROJECT_ROOT = Path(__file__).resolve().parent
SIMULATION_ROOT = os.getenv("OFFSHORE_SIMULATION_ROOT")
DATA_ROOT = Path(SIMULATION_ROOT).resolve() if SIMULATION_ROOT else PROJECT_ROOT
SIMULATION = json.loads((DATA_ROOT / "simulation.json").read_text()) if SIMULATION_ROOT else None
CONTENT_DIR = PROJECT_ROOT / "dashboard_content"
ASSET_DIR = PROJECT_ROOT / "dashboard_assets"
METRIC_COPY_KEYS = {
    "nMAE": "nmae", "nRMSE": "nrmse", "R²": "r2", "Skill score": "skill",
    "MAE (MW)": "mae_mw", "P90 error (MW)": "p90_mw",
}
# Add, remove or reorder forecast sub-tabs here. The matching labels live in
# dashboard_content/en.toml and de.toml under [forecasting].
FORECAST_TAB_KEYS = ("offshore", "onshore", "solar", "load", "prices")


def read_toml(path: Path) -> dict:
    """Load author-editable copy on every rerun so text edits appear promptly."""
    with path.open("rb") as stream:
        return tomllib.load(stream)


def artifact_fingerprint(paths: DashboardPaths) -> tuple:
    """Invalidate cached data when a notebook refreshes an input artifact."""
    result = []
    for path in (paths.historical_predictions, paths.latest_predictions, paths.actual_dataset,
                 paths.latest_predictions.parent / "pending_predictions.parquet",
                 paths.latest_predictions.parent / "display_actuals.parquet"):
        stat = path.stat() if path is not None and path.exists() else None
        result.append((str(path), stat.st_size if stat else None, stat.st_mtime_ns if stat else None))
    return tuple(result)


@st.cache_data(show_spinner="Loading forecast history…")
def get_predictions(paths: DashboardPaths, fingerprint: tuple) -> pd.DataFrame:
    return load_dashboard_predictions(paths)


def theme_styles(dark: bool) -> str:
    palette = (
        {"background": "#111B19", "panel": "#1B2926", "text": "#F2F7F5", "muted": "#B3C1BC",
         "border": "#354A43", "accent": "#32C6A4", "hover": "#24453B", "input": "#21332E"}
        if dark else
        {"background": "#F2F5F3", "panel": "#FFFFFF", "text": "#1A2522", "muted": "#65746F",
         "border": "#DFE8E3", "accent": "#087E65", "hover": "#E7F4EE", "input": "#FFFFFF"}
    )
    variables = "\n".join(f"--dash-{key}: {value};" for key, value in palette.items())
    return f"""
    <style>
      :root {{ {variables} }}
      .stApp {{ background: var(--dash-background); color: var(--dash-text); }}
      .block-container {{ max-width: 1160px; padding: 1.6rem 2.1rem 4rem; }}
      .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label {{ color: var(--dash-text); }}
      header[data-testid="stHeader"] {{ display: none; }}
      .stApp .dash-note {{ color: var(--dash-muted); }}
      .stApp [data-testid="stCaptionContainer"] p {{ color: var(--dash-muted); }}
      .st-key-hero, .st-key-forecast-panel, .st-key-offshore-panel,
      .st-key-onshore-placeholder-panel, .st-key-solar-placeholder-panel,
      .st-key-load-placeholder-panel, .st-key-prices-placeholder-panel,
      .st-key-info-panel, .st-key-example-panel {{
        background: var(--dash-panel); border: 1px solid var(--dash-border);
        border-radius: 14px; padding: 1.3rem 1.65rem; box-shadow: none;
      }}
      .st-key-hero {{ padding: 1.65rem 1.9rem; }}
      .st-key-hero h1 {{ font-size: clamp(1.8rem, 3vw, 2.55rem); line-height: 1.18;
        letter-spacing: -0.025em; margin: 0 0 0.65rem; }}
      .st-key-hero p {{ line-height: 1.55; }}
      .st-key-hero a {{ color: var(--dash-accent); text-decoration: none; }}
      .st-key-hero a:hover {{ text-decoration: underline; }}
      .st-key-hero [data-testid="stSelectbox"] [role="group"] {{
        background: var(--dash-input) !important;
        border: 1px solid var(--dash-border) !important;
      }}
      .st-key-hero [data-testid="stSelectbox"] input,
      .st-key-hero [data-testid="stSelectbox"] button {{ color: var(--dash-text) !important; }}
      .st-key-hero [data-testid="stWidgetLabel"] p {{ color: var(--dash-muted); }}
      .stTabs [data-baseweb="tab-list"] {{
        background: var(--dash-panel); border: 1px solid var(--dash-border);
        border-radius: 11px; padding: 0.2rem 0.35rem; gap: 0.25rem;
      }}
      .stTabs [data-baseweb="tab"] {{ color: var(--dash-muted); border-radius: 7px; }}
      .stTabs [data-baseweb="tab"]:hover {{ background: var(--dash-hover); color: var(--dash-text); }}
      .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: var(--dash-accent); font-weight: 700; background: var(--dash-hover);
      }}
      .stTabs [data-baseweb="tab-highlight"] {{ background: var(--dash-accent) !important; }}
      .stTabs [data-baseweb="tab-border"] {{ background: transparent; }}
      .accuracy-number {{ font-size: clamp(2.25rem, 5vw, 3.8rem); font-weight: 750;
        color: var(--dash-accent); letter-spacing: -0.035em; line-height: 1.12; }}
      .accuracy-label {{ font-size: 1.08rem; font-weight: 650; color: var(--dash-text); }}
      .dash-note {{ font-size: 0.86rem; color: var(--dash-muted); line-height: 1.5; }}
      .metric-wrap {{ overflow-x: auto; width: 100%; }}
      .metric-table {{ border-collapse: collapse; width: 100%; table-layout: fixed;
        color: var(--dash-text); border: 1px solid var(--dash-border); }}
      .metric-table th, .metric-table td {{ padding: 0.85rem 1rem; text-align: center;
        border-bottom: 1px solid var(--dash-border); }}
      .metric-table thead th {{ background: var(--dash-background);
        color: var(--dash-muted); font-weight: 650; }}
      .metric-table th:first-child {{ width: 36%; text-align: left; }}
      .metric-table tbody th {{ font-weight: 650; }}
      .metric-table tbody tr:nth-child(even) {{ background: var(--dash-hover); }}
      .metric-table tbody td {{ font-weight: 650; font-variant-numeric: tabular-nums; }}
      .metric-table tbody tr:last-child th, .metric-table tbody tr:last-child td {{ border-bottom: none; }}
      .metric-help {{ display: inline-flex; align-items: center; justify-content: center;
        width: 1.1rem; height: 1.1rem; border-radius: 50%; border: 1px solid var(--dash-muted);
        color: var(--dash-muted); font-size: 0.72rem; font-weight: 700; cursor: help;
        margin-left: 0.45rem; vertical-align: middle; }}
      .metric-help:focus-visible {{ outline: 2px solid var(--dash-accent); outline-offset: 2px; }}
      .help-tip {{ display: none; margin-top: 0.6rem; padding: 0.55rem 0.7rem;
        border-left: 2px solid var(--dash-accent); background: var(--dash-panel);
        color: var(--dash-muted); text-align: left; font-weight: 400;
        font-size: 0.81rem; line-height: 1.45; overflow-wrap: anywhere; }}
      .metric-help:hover + .help-tip, .metric-help:focus + .help-tip {{ display: block; }}
      .refit-note {{ margin: 0.55rem 0 0.7rem; padding: 0.8rem 1rem;
        border-left: 3px solid var(--dash-accent); border-radius: 0 8px 8px 0;
        background: var(--dash-background); color: var(--dash-muted);
        font-size: 0.87rem; line-height: 1.5; }}
      .disclaimer {{ background: var(--dash-background); border-left: 3px solid var(--dash-accent);
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; color: var(--dash-muted);
        font-size: 0.85rem; line-height: 1.5; margin-top: 1.2rem; }}
      @media (max-width: 700px) {{
        .block-container {{ padding: 1rem 0.85rem 3rem; }}
        .st-key-hero, .st-key-forecast-panel, .st-key-offshore-panel,
        .st-key-onshore-placeholder-panel, .st-key-solar-placeholder-panel,
        .st-key-load-placeholder-panel, .st-key-prices-placeholder-panel,
        .st-key-info-panel, .st-key-example-panel {{ padding: 1.05rem 0.9rem; }}
        .metric-table {{ min-width: 540px; }}
      }}
    </style>
    """


def social_links(profiles: dict[str, str]) -> str:
    """Render only configured public profile URLs."""
    links = []
    for name in ("LinkedIn", "Xing", "GitHub"):
        url = profiles.get(name.lower(), "").strip()
        parsed = urlparse(url)
        if parsed.scheme in {"https", "http"} and parsed.netloc:
            links.append(f"[{name}]({url})")
    return " · ".join(links)


def metric_table(metrics: pd.DataFrame, copy: dict) -> str:
    """Render a simple two-column table with in-flow metric explanations."""
    indexed = metrics.set_index(["metric", "window"])
    window_labels = {
        "Last year": copy["windows"]["last_year"],
        "All history": copy["windows"]["all_history"],
    }
    columns = "".join(
        f"<th scope='col'>{html.escape(window_labels[window])}</th>"
        for window in WINDOW_ORDER
    )
    rows = []
    for metric in METRIC_ORDER:
        key = METRIC_COPY_KEYS[metric]
        label = html.escape(copy["metrics"][key])
        help_text = html.escape(copy["metrics"][f"{key}_help"])
        cells = []
        for window in WINDOW_ORDER:
            row = indexed.loc[(metric, window)]
            value = html.escape(format_metric(metric, row["value"]))
            cells.append(f"<td>{value}</td>")
        rows.append(
            f"<tr><th scope='row'>{label}<span class='metric-help' tabindex='0' "
            f"aria-label='{help_text}'>?</span><span class='help-tip'>{help_text}</span>"
            f"</th>{''.join(cells)}</tr>"
        )
    return (
        "<div class='metric-wrap'><table class='metric-table'>"
        f"<thead><tr><th scope='col'></th>{columns}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_hero(copy: dict, profiles: dict, german: bool) -> None:
    with st.container(key="hero"):
        main, controls = st.columns([4.7, 1.3], gap="large", vertical_alignment="top")
        with main:
            st.markdown(f"# {copy['hero']['title']}")
            st.markdown(copy["hero"]["bio"])
            st.markdown(f"**{copy['hero']['update_label']}:** {copy['hero']['update_text']}")
            links = social_links(profiles)
            byline = copy["hero"]["byline"]
            st.caption(f"{byline} · {links}" if links else byline)
        with controls:
            st.caption(copy["controls"]["language"])
            flag, selector = st.columns([1, 4], gap="small", vertical_alignment="center")
            with flag:
                st.image(str(ASSET_DIR / ("de.svg" if german else "gb.svg")), width=30)
            with selector:
                st.selectbox(
                    copy["controls"]["language"], ["EN", "DE"],
                    key="dashboard_language", label_visibility="collapsed",
                )
            st.toggle(copy["controls"]["theme"], key="dashboard_dark_mode")


def render_offshore(copy: dict, dark: bool, german: bool) -> None:
    with st.container(key="offshore-panel"):
        try:
            paths = DashboardPaths.from_project_root(DATA_ROOT)
            predictions = get_predictions(paths, artifact_fingerprint(paths))
        except DashboardDataError as error:
            st.error(str(error))
            return
        try:
            forward = load_forward_predictions(paths, now=pd.Timestamp(SIMULATION["logical_time"]) if SIMULATION else None,
                                               allow_simulation=bool(SIMULATION))
        except DashboardDataError as error:
            st.warning(str(error))
            forward = None
        if forward is not None:
            forward = forward.loc[forward.index > predictions.index.max()]
            if forward.empty:
                forward = None
        metrics = metric_windows(predictions)
        nmae_year = metrics.loc[
            (metrics.window == "Last year") & (metrics.metric == "nMAE"), "value"
        ].iloc[0]
        accuracy = max(0.0, 1.0 - float(nmae_year))
        accuracy_text = f"{accuracy:.1%}"
        if german:
            accuracy_text = accuracy_text.replace(".", ",")
        st.markdown(
            f"<div class='accuracy-label'>{html.escape(copy['offshore']['accuracy_prefix'])}</div>"
            f"<div class='accuracy-number'>{accuracy_text}</div>"
            f"<div class='accuracy-label'>{html.escape(copy['offshore']['accuracy_suffix'])}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<p class='dash-note'>{html.escape(copy['offshore']['accuracy_note'])}</p>",
                    unsafe_allow_html=True)
        st.subheader(copy["offshore"]["chart_title"])
        figure = build_forecast_figure(
            predictions, dark_mode=dark, labels=copy["offshore"], forward=forward,
        )
        st.plotly_chart(figure, width="stretch",
                        config={"displaylogo": False, "displayModeBar": False, "scrollZoom": True})
        st.caption(copy["offshore"]["chart_help"])
        st.caption(copy["offshore"]["schedule_note"])
        status_path = DATA_ROOT / "dashboard_data/status.json"
        if status_path.exists():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                updated = pd.Timestamp(status["last_updated_utc"]).tz_convert(BERLIN_TIMEZONE)
                fitted = (pd.Timestamp(status["fitted_at_utc"]).tz_convert(BERLIN_TIMEZONE)
                          if status.get("fitted_at_utc") else None)
                actual = (pd.Timestamp(status["latest_actual_utc"]).tz_convert(BERLIN_TIMEZONE)
                          if status.get("latest_actual_utc") else None)
                status_text = copy["offshore"]["status_note"].format(
                    updated=f"{updated:%d.%m.%Y %H:%M %Z}",
                    fitted=f"{fitted:%d.%m.%Y %H:%M %Z}" if fitted is not None else "—",
                    actual=f"{actual:%d.%m.%Y %H:%M %Z}" if actual is not None else "—",
                )
                st.markdown(f"<div class='refit-note'>{html.escape(status_text)}</div>",
                            unsafe_allow_html=True)
                if status.get("forecast_status") == "weather_unavailable":
                    st.warning(copy["offshore"]["weather_unavailable"].format(day=status["forecast_day"]))
                elif status.get("forecast_status") == "bootstrap_only":
                    st.info(copy["offshore"]["bootstrap_only"])
                elif status.get("forecast_status") == "deadline_missed":
                    st.warning(copy["offshore"]["deadline_missed"].format(day=status["forecast_day"]))
                elif status.get("forecast_status") in {"model_unavailable", "fit_failed"}:
                    st.warning(copy["offshore"]["model_unavailable"].format(day=status["forecast_day"]))
                elif status.get("fit_mode") == "reused":
                    key = ("model_reused_generation" if status.get("training_data_status") == "ENTSO-E generation incomplete"
                           else "model_reused_other")
                    st.info(copy["offshore"][key].format(
                        age=status["model_age_days"], day=status["forecast_day"]))
                log_path = DATA_ROOT / "dashboard_data/run_logs" / f"{status.get('issue_day', updated.date())}.json"
                if log_path.exists():
                    with st.expander(copy["offshore"]["run_log_label"]):
                        st.json(json.loads(log_path.read_text(encoding="utf-8")))
            except (KeyError, ValueError, TypeError):
                st.warning(copy["offshore"]["status_unavailable"])
        else:
            st.caption(copy["offshore"]["status_unavailable"])
        if forward is None:
            st.markdown(
                f"<p class='dash-note'>{html.escape(copy['offshore']['forward_unavailable'])}</p>",
                unsafe_allow_html=True,
            )
        else:
            first = forward.iloc[0]
            fitted = first.fitted_at_utc.tz_convert(BERLIN_TIMEZONE)
            train_start = first.training_start_utc.tz_convert(BERLIN_TIMEZONE)
            last_train_point = (
                first.training_end_exclusive_utc - pd.Timedelta(minutes=15)
            ).tz_convert(BERLIN_TIMEZONE)
            forecast_day = forward.index.min().tz_convert(BERLIN_TIMEZONE).date()
            st.caption(copy["offshore"]["shading_note"])
            details = (
                f"{copy['offshore']['refit_label']}: {fitted:%d.%m.%Y %H:%M %Z} · "
                f"{copy['offshore']['training_range_label']}: "
                f"{train_start:%d.%m.%Y %H:%M}–{last_train_point:%d.%m.%Y %H:%M} "
                f"({BERLIN_TIMEZONE}) · {copy['offshore']['forecast_day_label']}: "
                f"{forecast_day:%d.%m.%Y}"
            )
            dropped_days = str(first.get("dropped_train_days", "")).strip()
            if dropped_days:
                details += f" · {copy['offshore']['dropped_days_label']}: {dropped_days}"
            st.markdown(f"<div class='refit-note'>{html.escape(details)}</div>",
                        unsafe_allow_html=True)
        st.subheader(copy["offshore"]["metrics_title"])
        st.markdown(metric_table(metrics, copy), unsafe_allow_html=True)
        st.caption(copy["offshore"]["metrics_note"])
        st.markdown(f"<div class='disclaimer'>{html.escape(copy['offshore']['disclaimer'])}</div>",
                    unsafe_allow_html=True)


def render_forecasting(copy: dict, dark: bool, german: bool) -> None:
    with st.container(key="forecast-panel"):
        st.markdown(copy["forecasting"]["intro"])
        tabs = st.tabs([copy["forecasting"][key] for key in FORECAST_TAB_KEYS])
        for key, tab in zip(FORECAST_TAB_KEYS, tabs):
            with tab:
                if key == "offshore":
                    render_offshore(copy, dark, german)
                else:
                    with st.container(key=f"{key}-placeholder-panel"):
                        st.markdown(f"### {copy['forecasting'][key]}")
                        st.caption(copy["forecasting"]["coming_soon"])


def main() -> None:
    st.set_page_config(page_title="Day-Ahead Forecasting for the German Energy Market",
                       page_icon="🌬️", layout="wide", initial_sidebar_state="collapsed")
    if SIMULATION:
        st.warning(f"SIMULATION — issue day {SIMULATION['issue_day']}. "
                   f"Executed {SIMULATION['executed_at_utc']}. Not a live market forecast.")
    if st.session_state.get("dashboard_language") not in {"EN", "DE"}:
        st.session_state["dashboard_language"] = "EN"
    language = st.session_state["dashboard_language"]
    dark = bool(st.session_state.get("dashboard_dark_mode", False))
    german = language == "DE"
    copy = read_toml(CONTENT_DIR / ("de.toml" if german else "en.toml"))
    profiles = read_toml(CONTENT_DIR / "links.toml")["profiles"]
    st.markdown(theme_styles(dark), unsafe_allow_html=True)
    render_hero(copy, profiles, german)
    st.write("")
    forecast, general, example = st.tabs([
        copy["navigation"]["forecasting"], copy["navigation"]["general_info"],
        copy["navigation"]["use_case"],
    ])
    with forecast:
        render_forecasting(copy, dark, german)
    with general:
        body = copy["general_info"]["body"].strip()
        if body:
            with st.container(key="info-panel"):
                st.markdown(body)
    with example:
        with st.container(key="example-panel"):
            st.subheader(copy["use_case"]["title"])
            st.markdown(copy["use_case"]["body"])


if __name__ == "__main__":
    main()
