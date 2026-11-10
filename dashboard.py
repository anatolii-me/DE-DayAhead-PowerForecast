"""Streamlit dashboard for German day-ahead renewable-generation forecasts."""

from __future__ import annotations

import base64
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


def svg_uri(svg: str | bytes) -> str:
    payload = svg.encode("utf-8") if isinstance(svg, str) else svg
    return "data:image/svg+xml;base64," + base64.b64encode(payload).decode("ascii")


def theme_styles(dark: bool, german: bool) -> str:
    palette = (
        {"background": "#111B19", "panel": "#1B2926", "text": "#F2F7F5", "muted": "#B3C1BC",
         "border": "#354A43", "accent": "#32C6A4", "hover": "#24453B", "input": "#21332E"}
        if dark else
        {"background": "#F2F5F3", "panel": "#FFFFFF", "text": "#1A2522", "muted": "#65746F",
         "border": "#DFE8E3", "accent": "#087E65", "hover": "#E7F4EE", "input": "#FFFFFF"}
    )
    variables = "\n".join(f"--dash-{key}: {value};" for key, value in palette.items())
    flag_en = svg_uri((ASSET_DIR / "gb.svg").read_bytes())
    flag_de = svg_uri((ASSET_DIR / "de.svg").read_bytes())
    icon_color = "#B3C1BC" if dark else "#65746F"
    sun = svg_uri(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.9 4.9l1.4 1.4m11.4 11.4 1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4"/></svg>')
    moon = svg_uri(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{icon_color}" stroke-width="1.8" stroke-linejoin="round"><path d="M20.5 15.5A8.5 8.5 0 0 1 8.5 3.5 8.5 8.5 0 1 0 20.5 15.5Z"/></svg>')
    theme_label = "Anzeigemodus" if german else "Display Mode"
    language_label = "Sprache" if german else "Language"
    selected_flag = flag_de if german else flag_en
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
      .st-key-hero {{ padding: 1.65rem 1.9rem; position: relative; }}
      .st-key-hero h1 {{ font-size: clamp(1.7rem, 3vw, 2.55rem); line-height: 1.18;
        letter-spacing: -0.025em; margin: 0 0 0.65rem; padding: 0 3.2rem 0 0; }}
      .st-key-hero p {{ line-height: 1.55; }}
      .st-key-hero a {{ color: var(--dash-accent); text-decoration: none; }}
      .st-key-hero a:hover {{ text-decoration: underline; }}
      .st-key-dashboard_settings {{ position: absolute; top: 1.65rem; right: 1.8rem; z-index: 2; }}
      .st-key-dashboard_settings button {{ position: relative; width: 2.25rem; height: 2.25rem;
        padding: 0 !important; border: 0 !important; border-radius: 0.35rem !important;
        background: transparent !important; box-shadow: none !important; color: var(--dash-text) !important; }}
      .st-key-dashboard_settings button:hover {{ background: var(--dash-hover) !important; }}
      .st-key-dashboard_settings button:focus-visible {{ outline: 2px solid var(--dash-accent) !important; }}
      .st-key-dashboard_settings button > div {{ width: 100%; justify-content: center; }}
      .st-key-dashboard_settings button [aria-hidden="true"] {{ display: none; }}
      .st-key-dashboard_settings button span:has(> [data-testid="stIconMaterial"]),
      .st-key-dashboard_settings button [data-testid="stIconMaterial"] {{
        width: 22px !important; height: 22px !important; font-size: 22px !important;
      }}
      .st-key-dashboard_settings button p {{ font-size: 0; width: 0; overflow: hidden; }}
      .st-key-dashboard_settings button::after {{ content: ""; position: absolute; top: 0.35rem;
        right: 0.35rem; width: 0.72rem; height: 0.72rem; border: 1px solid var(--dash-panel);
        background: url("{selected_flag}") center / cover no-repeat; }}
      [data-testid="stPopoverBody"] {{ background: var(--dash-panel) !important;
        color: var(--dash-text) !important; border: 1px solid var(--dash-border) !important; }}
      [data-testid="stPopoverBody"] h3,
      [data-testid="stPopoverBody"] .st-key-display-controls p {{ color: var(--dash-text) !important; }}
      .st-key-display-controls {{ width: min(310px, calc(100vw - 3rem)); }}
      .st-key-display-controls [data-testid="stSelectbox"] input,
      .st-key-display-controls [data-testid="stSelectbox"] button {{ color: var(--dash-text) !important; }}
      .st-key-display-controls [data-testid="stSelectbox"] [role="group"] {{
        background: var(--dash-input) !important; border: 1px solid var(--dash-border) !important;
        min-height: 2.4rem;
      }}
      .st-key-display-controls [data-testid="stSelectbox"] input {{ color: var(--dash-text) !important; }}
      .st-key-display-controls input[aria-label="{theme_label}"],
      .st-key-display-controls input[aria-label="{language_label}"] {{
        padding-left: 38px !important; background-repeat: no-repeat;
        background-position: 10px center; background-size: 20px 18px;
      }}
      .st-key-display-controls input[aria-label="{theme_label}"] {{ background-image: url("{moon if dark else sun}"); }}
      .st-key-display-controls input[aria-label="{language_label}"] {{ background-image: url("{selected_flag}"); }}
      [role="listbox"][aria-label="{theme_label}"] [role="option"][data-key="0"] [data-item-hl] {{ background-image: url("{sun}"); }}
      [role="listbox"][aria-label="{theme_label}"] [role="option"][data-key="1"] [data-item-hl] {{ background-image: url("{moon}"); }}
      [role="listbox"][aria-label="{language_label}"] [role="option"][data-key="0"] [data-item-hl] {{ background-image: url("{flag_en}"); }}
      [role="listbox"][aria-label="{language_label}"] [role="option"][data-key="1"] [data-item-hl] {{ background-image: url("{flag_de}"); }}
      [role="listbox"][aria-label="{theme_label}"] [data-item-hl],
      [role="listbox"][aria-label="{language_label}"] [data-item-hl] {{
        padding-left: 32px; background-repeat: no-repeat; background-position: 5px center;
        background-size: 20px 16px;
      }}
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
      .disclaimer {{ background: var(--dash-background); border-left: 3px solid var(--dash-accent);
        border-radius: 0 8px 8px 0; padding: 0.8rem 1rem; color: var(--dash-muted);
        font-size: 0.85rem; line-height: 1.5; margin-top: 1.2rem; }}
      @media (max-width: 700px) {{
        .block-container {{ padding: 1rem 0.85rem 3rem; }}
        .st-key-hero, .st-key-forecast-panel, .st-key-offshore-panel,
        .st-key-onshore-placeholder-panel, .st-key-solar-placeholder-panel,
        .st-key-load-placeholder-panel, .st-key-prices-placeholder-panel,
        .st-key-info-panel, .st-key-example-panel {{ padding: 1.05rem 0.9rem; }}
        .st-key-dashboard_settings {{ top: 1.05rem; right: 0.9rem; }}
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
        with st.popover(copy["controls"]["settings"], icon=":material/settings:",
                        key="dashboard_settings"):
            st.subheader(copy["controls"]["settings"])
            with st.container(key="display-controls"):
                language_label, language_choice = st.columns([1, 1.1], vertical_alignment="center")
                with language_label:
                    st.markdown(copy["controls"]["language"])
                with language_choice:
                    st.selectbox(copy["controls"]["language"], ["EN", "DE"],
                                 format_func=lambda code: "English" if code == "EN" else "German",
                                 key="dashboard_language", label_visibility="collapsed")
                theme_label, theme_choice = st.columns([1, 1.1], vertical_alignment="center")
                with theme_label:
                    st.markdown(copy["controls"]["theme"])
                with theme_choice:
                    st.selectbox(copy["controls"]["theme"], [False, True],
                                 format_func=lambda dark: "Dark" if dark else "Light",
                                 key="dashboard_dark_mode", label_visibility="collapsed")
        st.markdown(f"# {copy['hero']['title']}")
        st.markdown(copy["hero"]["bio"])
        st.markdown(hero_refit_text(copy), unsafe_allow_html=True)
        links = social_links(profiles)
        byline = copy["hero"]["byline"]
        st.caption(f"{byline} · {links}" if links else byline)


def hero_refit_text(copy: dict) -> str:
    """Public daily-update summary from the latest published status."""
    try:
        status = json.loads((DATA_ROOT / "dashboard_data/status.json").read_text(encoding="utf-8"))
        fitted_utc = pd.Timestamp(status["fitted_at_utc"]).tz_convert("UTC")
        fitted_berlin = fitted_utc.tz_convert(BERLIN_TIMEZONE)
        content = copy["hero"]["update_text"].format(
            berlin=f"{fitted_berlin:%d.%m.%Y %H:%M}", utc=f"{fitted_utc:%H:%M}",
        )
        current_day = (pd.Timestamp(SIMULATION["logical_time"]).tz_convert(BERLIN_TIMEZONE).date()
                       if SIMULATION else pd.Timestamp.now(tz=BERLIN_TIMEZONE).date())
        forecast_day = pd.Timestamp(status["forecast_day"]).date()
        if status.get("forecast_status") == "issued" and forecast_day >= current_day:
            content += " " + copy["hero"]["forecast_day"].format(day=f"{forecast_day:%d.%m.%Y}")
    except (OSError, KeyError, ValueError, TypeError):
        content = copy["hero"]["update_unavailable"]
    return f"<p class='dash-note'>{html.escape(content)}</p>"


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
    st.markdown(theme_styles(dark, german), unsafe_allow_html=True)
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
