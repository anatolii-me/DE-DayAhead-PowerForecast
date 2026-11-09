# Day-Ahead Forecasting for the German Energy Market

An English/German Streamlit dashboard for day-ahead generation forecasting in
the Germany–Luxembourg (DE_LU) market. The first live view is offshore wind;
onshore wind, solar, load and day-ahead price views are marked as in
development. The other top-level tabs are General Info and a future BESS use
case.

The offshore view displays the notebook-6 production candidate, **XGBoost HPO
with L1 loss**. The chart opens on the latest seven local delivery days and can
be zoomed or moved through the full evaluated forecast history with its range
slider. Forecast and actual generation are both shown in **MW**, converted from
capacity factor using the installed capacity at each target interval (which
can change over time). A separately issued forecast for tomorrow is drawn
without an actual-generation line until the fact arrives. D-1 and D are
shaded only when that current forward artifact exists; the refit time and its
365-day training-data range appear below the chart. It reports nMAE, nRMSE,
R², skill against the actual
capacity factor at the same local quarter-hour one week earlier, plus MAE and
90th-percentile absolute error in MW. The table compares the last 365 local
delivery days with all available evaluated history. The large accuracy number
is simply `100% − nMAE` for the last year; it is an accessible summary, not a
probabilistic guarantee.

## Edit the words and links yourself

No Python changes are needed for normal copy edits:

- [`dashboard_content/en.toml`](dashboard_content/en.toml) — all English text:
  header, labels, notes, metric explanations, disclaimer and placeholders.
- [`dashboard_content/de.toml`](dashboard_content/de.toml) — matching German
  text. English is selected by default; keep both files' keys when editing.
- [`dashboard_content/links.toml`](dashboard_content/links.toml) — LinkedIn,
  Xing and GitHub URLs. An empty value hides that link. The supplied public
  profile links are already included.

The header is under `[hero]`. To change the degree, edit the `byline` value;
Markdown such as `**bold**` is supported in the header and section paragraphs.
To add General Info later, fill the `[general_info]` `body` value in **both**
language files. A multiline TOML string works well, for example:

```toml
[general_info]
body = """
### Forecast workflow

The model trains on the most recent 365 available days ending at D-2.

### Data sources

Describe your sources here.
"""
```

The layout uses content-sized panels, so text can grow without adjusting fixed
heights. Colors and layout CSS are in [`dashboard.py`](dashboard.py), and the
base Streamlit theme is in [`.streamlit/config.toml`](.streamlit/config.toml).

### Change tab content or navigation

- Edit `[forecasting]`, `[offshore]`, `[general_info]` or `[use_case]` in both
  language files to change the text in those views. The `coming_soon` value is
  shared by all four forecast placeholders.
- To add, remove or reorder a **forecast** tab, edit `FORECAST_TAB_KEYS` near the
  top of [`dashboard.py`](dashboard.py). Add a matching key and label under
  `[forecasting]` in both language files. Any key other than `offshore` shows
  the standard in-development placeholder until you add its rendering logic
  in `render_forecasting()`.
- To add or remove a **top-level** tab, edit the `st.tabs(...)` list and matching
  `with ...:` content in `main()` in [`dashboard.py`](dashboard.py), plus its
  label under `[navigation]` in both language files.

## Run locally

Use Python 3.13. The repository contains a compact public artifact snapshot; then:

```powershell
python -m pip install -r requirements-dashboard.txt
streamlit run dashboard.py
```

The dashboard **reads** saved predictions; opening it does not retrain a model
or fetch data. The daily pipeline trains each target day's model on the most
recent 365 local calendar days ending at D-2 23:45 Europe/Berlin. DST delivery
days have 92 or 100 quarter-hours, rather than 96.

### Publish a genuine forecast for tomorrow

To refresh source data, extend the historical simulation, and issue a forecast
for tomorrow, install the full model runtime and run on **D-1**:

```powershell
python -m pip install -r requirements-offshore-final.txt
python -m utils.offshore_daily
```

The command retries missing source days, refreshes recent ENTSO-E revisions,
recreates the feature dataset, and trains the frozen XGBoost HPO L1 model. It
only publishes if D-2 and the target weather/capacity grid are complete. Public
artifacts go to `dashboard_data/`; no API credential is stored there. Forecasts
issued live are archived and remain unchanged when later actuals arrive.
Historical simulation for earlier days uses the correct D-1 weather run and
D-2 training cutoff, but historical source revisions are not point-in-time
snapshots. Its metrics therefore describe a retrospective simulation, not a
fully auditable live trading record.

`.github/workflows/offshore-daily.yml` runs once daily at 07:15 UTC and can also
be started manually. It uses the `ENTSOE_API_KEY` GitHub Actions secret, commits
only `dashboard_data/` on success, and never writes the key to the dashboard.
GitHub scheduled jobs can start late or be skipped, so check the displayed last
successful update. A forecast fitted at or after the SDAC gate closure at
12:00 Europe/Berlin is rejected. The dashboard shows a dashed line at the last actual
interval (D-2 23:45 Europe/Berlin); D-1 actuals are withheld until the next
model update.

To publish the public URL, connect the repository to Streamlit Community Cloud
and select `dashboard.py` on `main`. The root `requirements.txt` contains the
small dashboard runtime; `requirements-offshore-final.txt` is used only by the
daily workflow, and `requirements-research.txt` preserves the original notebook
environment. Community Cloud updates the app when the workflow commits new
artifacts. A public repository creates a public app by default.

`results/` remains outside Git because it contains large generated artifacts.
The dashboard reads compact public prediction and actual-data parquets from
`dashboard_data/` for time-varying installed capacity and observed capacity factor.
The loader verifies that historical `capacity_factor × capacity × 1000`
reconstructs source actual MW, so a revised or mismatched capacity snapshot
cannot silently corrupt the chart. Forecast MW likewise use the capacity
effective at each target interval; the forward command reads the monthly
capacity series using backward-as-of alignment and rejects stale values.
For deployment, make these artifacts available to the app and optionally set:

- `OFFSHORE_HISTORICAL_PREDICTIONS`
- `OFFSHORE_LATEST_PREDICTIONS`
- `OFFSHORE_ACTUAL_DATASET`
- `OFFSHORE_FORWARD_PREDICTIONS`

The capacity dataset must cover *every* forecast target interval. If it does
not, the app shows an explicit error instead of mislabelling CF as MW.

## Continuous checks

`.github/workflows/dashboard-checks.yml` installs the small dashboard runtime,
compiles the app and runs deterministic dashboard tests on relevant pull
requests and pushes to `main`. It does not retrain models or require ENTSO-E
credentials.
