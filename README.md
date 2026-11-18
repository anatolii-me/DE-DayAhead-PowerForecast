# Day-Ahead Power and Price Forecasting for the DE-LU Bidding Zone

A day-ahead forecasting project for the Germany–Luxembourg (DE-LU) electricity bidding zone. The project starts with offshore wind generation and will extend to solar, onshore wind, electricity load, and day-ahead prices. Forecasts are shown at 15-minute resolution.

**[Explore the dashboard](https://de-dayahead-powerforecast.streamlit.app/)**

## Current results

The first deployed model forecasts offshore wind generation with XGBoost. The figures below are from the latest 365 local delivery days available in the repository, ending on 8 November 2026. They cover 34,944 evaluated 15-minute intervals.

| Forecast | nMAE ↓ | nRMSE ↓ | R² ↑ |
| --- | ---: | ---: | ---: |
| Offshore wind | **0.0643** | **0.0902** | **0.8433** |
| Solar generation | In development | In development | In development |
| Onshore wind | In development | In development | In development |
| Electricity load | In development | In development | In development |
| Day-ahead prices | In development | In development | In development |

The offshore model predicts **capacity factor**: generation divided by installed capacity.  This lets it learn the relationship between weather and output while the wind fleet grows. The dashboard converts the forecast back to MW using the installed capacity for each delivery interval. The same approach is planned for subsequent generation models.

nMAE and nRMSE are errors on the 0–1 capacity-factor scale; an nMAE of 0.0643 means an average error of 6.43 percentage points of capacity factor. R² measures how much variation in observed capacity factor the predictions explain. These trailing-year figures include dates used during model development, so they should be read as retrospective performance, not as an independent test result. The [research notebooks](notebooks/research_wind_offshore_model/) document the separate chronological evaluations and model selection.

## Data and forecasting approach

- **Weather:** Hourly ECMWF IFS forecasts are retrieved through the Open-Meteo Single Runs API for 14 locations in the North Sea and Baltic Sea. For delivery day D, the model uses only the single **00:00 UTC run from D−1**. A forecast assembled from different runs could include information that was unavailable when the day-ahead prediction had to be made.
- **Target and capacity:** Actual offshore generation is queried from ENTSO-E for the `DE_LU` area. Monthly installed offshore capacity comes from Energy-Charts for Germany (`country="de"`). Generation is converted from MW to capacity factor for training. Actual generation and installed capacity are never used as predictors for the same target interval.
- **15-minute features:** The hourly weather forecast is interpolated to the 15-minute delivery grid. Interpolation stays within the same forecast run and location, and only fills intervals between consecutive known forecast hours. Wind direction is converted to vector components before interpolation; wind-power proxies and turbine operating flags are calculated afterwards. This creates a useful time grid without claiming to recover weather changes that the hourly forecast cannot show.
- **Training and evaluation:** The model is refitted for each target day using the preceding 365 calendar days, ending with D−2. Experiments use chronological splits rather than randomly mixing nearby intervals. Hyperparameters and training loss were selected on development data, then checked on later days.

I do not use ENTSO-E's published generation forecasts as model inputs. Historical downloads may contain revised or combined forecasts whose exact pre-auction version cannot be verified. That could give the model information it would not have had at prediction time. Forecasting generation from a known weather run provides a more consistent input for the future price model.

## What the offshore research found

- **Location-level weather matters.** Keeping a wind-power proxy for each of the 14 locations worked better than using only regional averages. The selected interpolated feature set also performed as well as or better than much larger alternatives, so extra features were not added just for a marginal score change.
- **Interpolation earned its place through comparison.** The first dataset copied each hourly forecast across four 15-minute intervals. A quarter-hour position feature and differences from the neighbouring hours of the same weather run gave the model some time context, but the weather values still changed in steps. I later compared interpolation methods by hiding known hourly values, then compared the resulting feature sets with chronological model validation. The selected approach uses shape-preserving cubic interpolation (PCHIP) and achieved the best balance of accuracy and feature count. Interpolated values remain estimates: the hourly source cannot tell us the true weather between forecast hours.
- **A deployment-like schedule changed the choice.** Short training windows and infrequent refits generally performed worse. A 365-day sliding window with daily refitting gave a practical balance of accuracy, stability, and available history.
- **The best development score was not the whole decision.** LightGBM led parts of development, and CatBoost led the first, short holdout. On a later 114-day comparison, the tuned XGBoost model had the lowest mean daily MAE and improved on its own simple L2 baseline. It was selected for the first release because its advantage held more consistently across the evaluated periods. Errors at high capacity factors remain a weakness, and that range has relatively few observations.
- **More complex methods did not automatically help.** Ridge regression and a small neural network lagged behind the tree models in the initial comparison. Additional weather features and alternative loss functions were retained only when their benefits survived chronological evaluation.

The dashboard is updated from a daily forecast pipeline. Its historical scores are a retrospective simulation using currently available source records, which may include later ENTSO-E corrections.

## Pipeline & Local Simulation

The forecasting pipeline automatically updates daily via GitHub Actions, combining ECMWF weather forecasts, ENTSO-E actuals, and Energy-Charts capacity data.

### Shared weather collection

`utils/weather.py` owns the Single Runs request, parsing and delivery-window logic.
`fetch_offshore_weather(start_date, end_date, points=None, height=100, model="ecmwf_ifs")`
retains the 14 sea points, column order, units and derived features used by the
deployed model. Import weather collectors from `utils.weather`.

```python
from utils.germany_grid_map import generate_germany_grid
from utils.weather import fetch_load_weather, fetch_weather_singleruns

# PROJECT_ROOT is the repository root. Use its boundary snapshot for repeatability.
points = generate_germany_grid(
    n_points=20,
    states_url=(PROJECT_ROOT / "utils/germany_states_low.geojson").as_uri(),
)
load_weather = fetch_load_weather("2026-01-01", "2026-01-01", points=points)

# The same transport can serve an explicitly chosen onshore or solar profile.
solar_weather = fetch_weather_singleruns(
    "2026-01-01", "2026-01-01", points=points,
    variables=["shortwave_radiation", "temperature_2m"], cell_selection="land",
)
```

Dates identify **inclusive run dates**, not delivery dates: the example returns
the forecast for 2 January in Europe/Berlin, plus one hourly buffer on each side.
The natural key is `(point_id, datetime_run_utc, datetime_utc)`; retain the run when
joining and interpolating. Output coordinates are the grid coordinates returned
by the API. Save the requested coordinates separately to reproduce the selection.

The load profile returns `temperature_2m` (°C), `wind_speed_10m` (m/s),
`shortwave_radiation` (W/m², preceding-hour mean) and `cloud_cover` (%), plus the
same seven metadata columns as offshore. These variable definitions follow the
[Open-Meteo documentation](https://open-meteo.com/en/docs/ecmwf-api).
The new interfaces reject failed, incomplete or non-finite runs by default;
`errors="skip"` explicitly allows gaps with warnings. Offshore retains its
existing recovery policy: fetch available data, validate all 14 points and hours
in the pipeline, then retry in two smaller batches if necessary.

The actual offshore implementation now lives under `utils/offshore/`:
`daily`, `day_ahead`, `dashboard`, `simulation`, `final_data`, `final_features`,
`final_evaluation`, `final_baselines`, `final_plot`, `cf_diagnostics`, and `points`.
Import these modules directly, for example `from utils.offshore import final_data`.
The top-level compatibility files have been removed; notebooks, tests and
workflow commands use the package paths.

Run the daily pipeline with `python -m utils.offshore.daily`. The daily workflow
uses this command. Its filename, `workflow_dispatch` and schedule remain
unchanged so external dispatchers can continue to trigger it. Moving
source files changes the source hashes used by derived-feature/model caches;
these may be recomputed once, while raw weather and published artifacts keep
their existing paths and formats.

### Environments and validation

Use Python 3.13 and install the requirements for the task:

| Task | Requirements | Entry point |
| --- | --- | --- |
| Display dashboard | `requirements-dashboard.txt` | `streamlit run dashboard.py` |
| Offshore daily forecasting and evaluation | `requirements-offshore-final.txt` | `python -m utils.offshore.daily` |
| Load data preparation and weather grid | `requirements-load.txt` | `notebooks/research_load_model/1_data_preparation.ipynb` |
| Full research environment | `requirements.txt` | All research notebooks and tests |

The load notebook currently prepares calendar features and geographic weather
inputs; load-model training and evaluation are not implemented yet. Its download
cell is opt-in (`FETCH_LOAD_WEATHER = True`). A uniform geographic grid does not
apply consumption weights to weather features.

Run `python -m unittest discover -s tests -p 'test*.py' -q` in the full environment.
Some historical offshore tests also need the local research results under
`results/`. CI runs the self-contained dashboard, weather and operational tests,
plus `test_load_notebook.py` in the dedicated load environment. The latter
executes the notebook's calendar/grid code and download/export branch using a
mock API response; it checks DST, variable values, Parquet and saved coordinates.
The **Dashboard checks** workflow also supports manual dispatch on a branch.

### Local Developer Simulation

You can run data downloads, model refits, and forecasting locally in an isolated sandbox without affecting production data:

```bash
# Run a fresh local simulation
python -m utils.offshore.simulation --output results/simulations/my-run

# Replay a specific historical date
python -m utils.offshore.simulation --issue-day 2026-11-09 --output results/simulations/replay

# Preview simulation results in the local dashboard ([http://127.0.0.1](http://127.0.0.1))
python -m utils.offshore.simulation --preview results/simulations/my-run
```
`ENTSOE_API_KEY` is read only from the local environment or the GitHub Actions secret. The pipeline checks public artifacts for the literal credential before completion. Local `.env` files are ignored by Git.

## Roadmap

| Stage | Data design | Data collection | Feature engineering | Model tuning |
| --- | --- | --- | --- | --- |
| Offshore wind | Complete | Complete | Complete | Complete; daily forecast available |
| Solar generation | Planned | Planned | Planned | Planned |
| Onshore wind | Planned | Planned | Planned | Planned |
| Electricity load | Planned | Planned | Planned | Planned |
| Day-ahead prices | Planned | Planned | Planned | Planned |

The price model is the final stage. Its planned target is the DE-LU day-ahead wholesale price in EUR/MWh. When generation and load forecasts become its inputs during training, they will need predictions made **out of sample** for each training period.

## Research notebooks

The [offshore research notebooks](notebooks/research_wind_offshore_model/) cover data preparation, interpolation, baseline models, feature selection, training-window and refit-schedule selection, joint loss and hyperparameter optimization, and the final comparison on later data.
