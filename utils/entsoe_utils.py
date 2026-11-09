import pandas as pd
from entsoe.entsoe import EntsoePandasClient
from datetime import timedelta
import time
from typing import Dict, Optional, List, Any, Generator
import os

# --- DEFAULT CONFIG ---
DEFAULT_COUNTRY_CODE = "DE_LU"
DEFAULT_START = "2026-03-01"
DEFAULT_END = "2026-03-08"

DEFAULT_QUERIES: Dict[str, str] = {
    # WARNING: By default, queries containing _forecast use process_type = "A01" (Day Ahead),
    # which in theory provides data available to traders before the Day Ahead market closes,
    # however, according to Entso-E MoP, this data can be revised retroactively,
    # so using it in Day Ahead ML tasks is not reliable and may lead to data leakage
    # --------------------------------------------------------------------------
    # ВНИМАНИЕ: По умолчанию запросы, что содержат _forecast,
    # используют process_type = "A01" (Day Ahead), что в теории
    # дает нам данные доступные трейдерам до закрытия заявок на Day Ahead рынке,
    # однако согласно Entso-E MoP эти данные могут правиться задним числом,
    # поэтому их использование в ML задачах Day Ahead не является надежным
    # и может привести к утечке данных


    # --- MAIN DATA ------------------------------------------------------------
    # Day Ahead Market Prices
    "prices": "query_day_ahead_prices",  # in Euros (EUR)

    # Actual electrical system load at the current moment
    "load_actual": "query_load",  # in Megawatts (MW) 

    # Forecasted electrical system load, process_type = "A01" (Day Ahead) by default
    # In case of load_forecast, this means: by 12:00 on the day before date D (D-1)
    "load_forecast": "query_load_forecast",  # in Megawatts (MW)

    # Actual generation per energy type (e.g., wind, solar, nuclear, etc.)
    "gen_actual": "query_generation",  # in Megawatts (MW)
    
    # Forecasted total generation, process_type = "A01" (Day Ahead) by default
    # In case of generation_forecast, this means: by 18:00 on the day before date D (D-1)
    "gen_total_forecast": "query_generation_forecast",  # in Megawatts (MW)

    # Forecasted wind and solar generation, process_type = "A01" (Day Ahead) by default
    # In case of generation_forecast, this means: by 18:00 on the day before date D (D-1)
    "gen_forecast_res": "query_wind_and_solar_forecast", # in Megawatts (MW)

    
    # --- ADDITIONAL DATA (use as required) ----------------------------------------
    # Actual generation per energy plant
    # "gen_per_plant": "query_generation_per_plant", # Not needed for now

    # Actual cross-border flows (physical) between countries, requires specific BORDER_NEIGHBORS
    # "cross_border_all": "query_physical_crossborder_allborders",

    # Net cross-border transmission capacity
    # "ntc_dayahead": "query_net_transfer_capacity_dayahead",

    # Scheduled exchanges (commercial) between countries, requires specific BORDER_NEIGHBORS
    # "scheduled_exchanges": "query_scheduled_exchanges",

    # Emergency and scheduled generation outages
    # WARNING: Requires separate processing, as it returns an event with a specific timestamp
    # rather than data at 15-minute resolution.
    # "gen_outages": "query_unavailability_of_generation_units", 

    # Emergency and scheduled transmissions outages
    # WARNING: Requires separate processing, as it returns an event with a specific timestamp
    # rather than data at 15-minute resolution.
    # "transmission_outages": "query_unavailability_transmission",

    # Installed capacity by fuel type
    # WARNING: reported on an annual or monthly basis rather than at 15-minute resolution,
    # so it requires separate processing.
    # "installed_capacity": "query_installed_generation_capacity",  

    # Water reserves (potential energy) in reservoirs
    # It is crucial for mountainous regions, but not as important for DE_LU.
    # "hydro_storage": "query_aggregate_water_reservoirs_and_hydro_storage",  # ДЛЯ ГЕРМАНИИ НЕ ВАЖНО

    # Imbalance prices, intraday realtime data, not Day Ahead
    # "imbalance_prices": "query_imbalance_prices",  # it's not needed for now

    # Balancing energy volume, intraday data, not Day Ahead
    # "balancing_energy": "query_activated_balancing_energy",  # it's not needed for now
}

DEFAULT_NEIGHBORS = [
    "AT",   # Austria
    "NL",   # Netherlands
    "FR",   # France
    "CH",   # Switzerland
    "PL",   # Poland
    "CZ",   # Czech Republic
    "DK_1", # Western Denmark
    "DK_2", # Eastern Denmark
    "SE_4", # Southern Sweden | Baltic Cable
    "NO_2"  # Norway | NordLink
]

CROSS_BORDER_FEATURES = {
    "cross_border",   # actual physical cross-border flows
    "ntc",            # maximum net transfer capacity
    "atc",            # available transfer capacity
}

# Queries that require two country codes (source and target) instead of one
PAIR_METHODS = {
    "query_crossborder_flows",
    "query_scheduled_exchanges",
    "query_unavailability_transmission",
    "query_net_transfer_capacity_dayahead",
}

# Queries that require process_type (balancing)
BALANCING_METHODS = {
    "query_activated_balancing_energy",
    "query_procured_balancing_capacity",
}

def _init_client(api_key: str) -> EntsoePandasClient:
    if not api_key:
        raise ValueError("API key must be provided")
    return EntsoePandasClient(api_key=api_key)

def _date_range_chunks(start: pd.Timestamp, end: pd.Timestamp, max_days: int) -> Generator[tuple, None, None]:
    """Generator for date range chunks."""
    current_start = start
    while current_start < end:
        current_end = min(current_start + timedelta(days=max_days), end)
        yield current_start, current_end
        current_start = current_end

def _fetch_single(
    client: EntsoePandasClient,
    feature_name: str,
    method_name: str,
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    neighbors: Optional[List[str]],
) -> Optional[pd.DataFrame]:
    """
    Universal fetcher with handling of ENTSO-E signature specifics.
    """
    try:
        method = getattr(client, method_name)
        
        # 1st CASE: Physical cross-border flows through "allborders"
        if method_name == "query_physical_crossborder_allborders":
            exp = method(country_code, start=start, end=end, export=True)
            imp = method(country_code, start=start, end=end, export=False)

            if exp is None and imp is None:
                return None

            if exp is None:
                combined = -imp.fillna(0)
            elif imp is None:
                combined = exp.fillna(0)
            else:
                exp = exp.loc[:, ~exp.columns.duplicated()]
                imp = imp.loc[:, ~imp.columns.duplicated()]
                combined = exp.fillna(0) - imp.fillna(0)

            combined = combined[
                ~combined.index.duplicated(keep="first")
            ].sort_index()

            return combined.add_suffix(f"_{feature_name}")

        # 2nd CASE: Pairwise methods
        if method_name in PAIR_METHODS:
            if not neighbors: return None
            pair_frames = []
            for nb in neighbors:
                try:
                    data = method(country_code, nb, start=start, end=end)
                    if data is None or data.empty: continue
                    
                    lbl = f"{country_code}_{nb}"
                    data = data.to_frame(name=f"{feature_name}_{lbl}") if isinstance(data, pd.Series) else data.add_suffix(f"_{feature_name}_{lbl}")
                    data = data[~data.index.duplicated(keep="first")]
                    pair_frames.append(data)
                except Exception as e:
                    # HTTP errors can include the request URL and its securityToken.
                    print(f"[WARN] Skip pair {country_code}-{nb} for {method_name}: {type(e).__name__}")

            if not pair_frames: return None
            combined_pairs = pd.concat(pair_frames, axis=1)
            return combined_pairs[~combined_pairs.index.duplicated(keep="first")].sort_index()

        # 3rd CASE: Regular queries
        kwargs: Dict[str, Any] = {"start": start, "end": end}

        if method_name == "query_procured_balancing_capacity":
            kwargs["process_type"] = "A51"
        elif method_name == "query_activated_balancing_energy":
            kwargs["business_type"] = "A19" 
        elif method_name == "query_imbalance_prices":
            kwargs["psr_type"] = None

        data = method(country_code, **kwargs)
        
        if data is None or (isinstance(data, (pd.DataFrame, pd.Series)) and data.empty):
            return None

        if isinstance(data, list):
            data = pd.concat(data, axis=0)

        if isinstance(data, pd.Series):
            data = data.to_frame(name=feature_name)
        
        return data.add_suffix(f"_{feature_name}")

    except Exception as e:
        # Never log the exception text: entsoe-py may put the API key in its URL.
        print(f"[ERROR] Failure in {feature_name} ({method_name}): {type(e).__name__}")
        return None

def collect_entsoe_data(
    api_key: str,
    country_code: str = DEFAULT_COUNTRY_CODE,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    queries: Dict[str, str] = DEFAULT_QUERIES,
    neighbors: Optional[List[str]] = DEFAULT_NEIGHBORS,
    max_days: int = 7,  # Вынесли max_days наружу
    retries: int = 2,   # Количество повторов по умолчанию
    retry_delay: float = 3.0    
) -> pd.DataFrame:
    """
    Main entry point for data collection.
    """
    client = EntsoePandasClient(api_key=api_key)
    
    # STRICT UTC TIMEZONE FIXATION
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    cache_dir = "entsoe_cache"
    os.makedirs(cache_dir, exist_ok=True)

    all_frames = []

    for feat, meth in queries.items():
            print(f"\n[PROCESS] Feature: {feat}")
            feature_chunks = []
            
            for c_start, c_end in _date_range_chunks(start_ts, end_ts, max_days=max_days):
                # Added hours and minutes to the cache name to avoid collisions
                time_str_start = c_start.strftime("%Y-%m-%d_%H-%M")
                time_str_end = c_end.strftime("%Y-%m-%d_%H-%M")
                cache_file = os.path.join(cache_dir, f"{feat}_{country_code}_{time_str_start}_{time_str_end}.parquet")

                if os.path.exists(cache_file):
                    print(f"  [CACHE] Loading {time_str_start}")
                    df_chunk = pd.read_parquet(cache_file)
                else:
                    df_chunk = None
                    # Retry logic for fetching data
                    for attempt in range(retries + 1):
                        print(f"  [API] Requesting {time_str_start} to {time_str_end} (Attempt {attempt + 1}/{retries + 1})...")
                        df_chunk = _fetch_single(client, feat, meth, country_code, c_start, c_end, neighbors)
                        
                        if df_chunk is not None and not df_chunk.empty:
                            # Pylance may not infer the type correctly, so we explicitly convert it to DatetimeIndex
                            if not isinstance(df_chunk.index, pd.DatetimeIndex):
                                df_chunk.index = pd.to_datetime(df_chunk.index)
                            # UTC timezone handling
                            if df_chunk.index.tz is None:
                                df_chunk.index = df_chunk.index.tz_localize('UTC')
                            else:
                                df_chunk.index = df_chunk.index.tz_convert('UTC')

                            df_chunk.to_parquet(cache_file)
                            break
                        
                        if attempt < retries:
                            print(f"  [RETRY] Got empty/failed response. Retrying in {retry_delay} s...")
                            time.sleep(retry_delay)
                    
                    time.sleep(1.2)  # Pause to avoid hitting API rate limits

                if df_chunk is not None and not df_chunk.empty:
                    feature_chunks.append(df_chunk)

            if feature_chunks:
                full_feat_df = pd.concat(feature_chunks, axis=0)
                # Duplicates should be cleaned here, otherwise the final concat(axis=1) 
                # might inflate the table to millions of rows if indices don't match.
                full_feat_df = full_feat_df[~full_feat_df.index.duplicated(keep="first")].sort_index()
                all_frames.append(full_feat_df)
                print(f"  [DONE] {feat} combined. Shape: {full_feat_df.shape}")

    if not all_frames:
        raise RuntimeError("No data could be fetched for any of the requested queries.")

    # Final concatenation of all features into a single DataFrame
    result = pd.concat(all_frames, axis=1)

    # Ensure UTC timezone before interpolation
    if result.index.tz is not None:
        result.index = result.index.tz_convert('UTC')
    
    # De-duplicate and sort the final result
    # result = result[~result.index.duplicated(keep="first")].sort_index() # Пока попробую отключить
    
    # Fill missing values using time-based interpolation for numeric columns
    # no more than 1 consecutive NaN will be filled to avoid overfitting
    numeric_cols = result.select_dtypes(include=['number']).columns
    result[numeric_cols] = result[numeric_cols].interpolate(method='time', limit=1)

    # FIX: Column names cleanup and renaming for better usability
    # Быстрый способ очистить названия
    result.columns = [
        str(col).replace("('", "")
        .replace("')", "")
        .replace("', '", "_")
        .replace(" ", "_")
        .lower()
        for col in result.columns
        ]

    columns_rename_map = {
        # Day Ahead prices in Euros (EUR)
        'prices_prices': 'prices',

        # Forecasted total load of the country in Megawatts (MW),
        # process_type = "A01" (Day Ahead) by default
        'forecasted_load_load_forecast': 'load_fc',

        # Forecasted total generation (MW) of the country, process_type = "A01" (Day Ahead) by default
        'gen_total_forecast_gen_total_forecast': 'gen_fc',

        # Forecasted solar generation (MW), process_type = "A01" (Day Ahead) by default
        'solar_gen_forecast_res': 'gen_fc_solar',

        # Forecasted offshore wind generation (MW), process_type = "A01" (Day Ahead) by default
        'wind_offshore_gen_forecast_res': 'gen_fc_wind_off',

        # Forecasted onshore wind generation (MW), process_type = "A01" (Day Ahead) by default
        'wind_onshore_gen_forecast_res': 'gen_fc_wind_on',

        # Actual total load in Megawatts (MW)
        'actual_load_load_actual': 'load_act',

        # Actual biomass generation (MW)
        'biomass_gen_actual_actual_aggregated_gen_actual': 'gen_biomass_act',

        # Actual low-quality brown coal/lignite generation (MW)
        'fossil_brown_coal/lignite_gen_actual_actual_aggregated_gen_actual': 'gen_lignite_act',

        # Actual coal-derived gas generation (MW)
        'fossil_coal-derived_gas_gen_actual_actual_aggregated_gen_actual': 'gen_coal_gas_act',

        # Actual natural gas generation (MW)
        'fossil_gas_gen_actual_actual_aggregated_gen_actual': 'gen_gas_act',

        # Actual hard coal generation (MW)
        'fossil_hard_coal_gen_actual_actual_aggregated_gen_actual': 'gen_hard_coal_act',

        # Actual fossil oil generation (MW)
        'fossil_oil_gen_actual_actual_aggregated_gen_actual': 'gen_oil_act',

        # Actual geothermal generation (MW)
        'geothermal_gen_actual_actual_aggregated_gen_actual': 'gen_geothermal_act',

        # Actual pumped hydro storage generation (MW)
        'hydro_pumped_storage_gen_actual_actual_aggregated_gen_actual': 'gen_hydro_pumped_act',

        # Actual consumption by pumped hydro storage (accumulation mode) (MW)
        'hydro_pumped_storage_gen_actual_actual_consumption_gen_actual': 'gen_hydro_pumped_cons_act',

        # Actual generation by run-of-river and poundage hydro plants (without reservoirs) (MW)
        'hydro_run-of-river_and_poundage_gen_actual_actual_aggregated_gen_actual': 'gen_hydro_river_act',

        # Actual generation by water reservoirs (MW)
        'hydro_water_reservoir_gen_actual_actual_aggregated_gen_actual': 'hydro_res_act',

        # Actual non-renewable sourcees generation (MW)
        'other_gen_actual_actual_aggregated_gen_actual': 'gen_other_act',

        # Actual renewable sources generation (MW)
        'other_renewable_gen_actual_actual_aggregated_gen_actual': 'gen_other_ren_act',

        # Actual solar generation (MW)
        'solar_gen_actual_actual_aggregated_gen_actual': 'gen_solar_act',

        # Actual waste burning generation (MW)
        'waste_gen_actual_actual_aggregated_gen_actual': 'gen_waste_act',

        # Actual offshore wind generation (MW)
        'wind_offshore_gen_actual_actual_aggregated_gen_actual': 'gen_wind_off_act',

        # Actual onshore wind generation (MW)
        'wind_onshore_gen_actual_actual_aggregated_gen_actual': 'gen_wind_on_act' 
    }

    result = result.rename(columns=columns_rename_map)
    
    return result
