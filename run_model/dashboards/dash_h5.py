"""
H5 Data Analysis Dashboard

This script creates a Streamlit dashboard to visualize H5 output data from the model.
It should be run from a directory containing H5 output files.

Requirements:
- streamlit
- pandas
- h5py
- plotly
- numpy

Run with (from inet-macro-dev directory):
    streamlit run path/to/macromodel/util/dash_h5.py
"""

# ruff: noqa: E402, I001

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_MODEL_PATH = REPO_ROOT / "run_model"
for path in (REPO_ROOT, RUN_MODEL_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import Config
from macro_data import DataWrapper
from macro_data.readers.economic_data.exchange_rates import ExchangeRatesReader
from src import distribution_validation as dv
from src.visual_helpers import SECTOR_CODE_TO_NAME, plot_sectoral_prices_over_time

# Set page config
st.set_page_config(page_title="Macromodel Dashboard - H5 output", page_icon="🌍", layout="wide")

# Title
st.title("Macromodel Dashboard - H5 output")

# Find all H5 files in the working directory and its subdirectories
cwd = os.getcwd()
h5_files = list(Path(cwd).glob("**/*.h5"))
if not h5_files:
    st.error("No H5 files found in the working directory")
    st.error("Please ensure you are running this script from a directory containing H5 output files")
    st.stop()

# File selection
selected_file = st.sidebar.selectbox("Select H5 File", h5_files)


# Function to load H5 data
def load_h5_data(file_path):
    with h5py.File(file_path, "r") as f:
        # Get all datasets
        datasets = {}

        def collect_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj[:]

        f.visititems(collect_datasets)
    return datasets


# Function to parse dataset names into components
def parse_dataset_name(name):
    parts = name.split("/")
    if len(parts) >= 3:
        country = parts[0]
        agent_market = parts[1]
        variable = "/".join(parts[2:])
        return country, agent_market, variable
    return None, None, name


# Function to get available options based on selections
def get_available_options(data, selected_country=None, selected_agent_market=None):
    available_agent_markets = set()
    available_variables = set()

    for name in data.keys():
        country, agent_market, variable = parse_dataset_name(name)
        if country and (selected_country is None or country == selected_country):
            if agent_market:
                available_agent_markets.add(agent_market)
                if selected_agent_market is None or agent_market == selected_agent_market:
                    if variable:
                        available_variables.add(variable)

    return sorted(list(available_agent_markets)), sorted(list(available_variables))


def decode_h5_strings(values):
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def unique_preserve_order(values):
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def infer_sector_codes(data, country, n_sectors):
    aggregate_codes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R_S"]
    if n_sectors == len(aggregate_codes):
        return aggregate_codes

    industry_path = f"{country}/firms/industries/Industry"
    if industry_path in data:
        firm_industries = unique_preserve_order(decode_h5_strings(data[industry_path]))
        if len(firm_industries) == n_sectors:
            return firm_industries

    return [f"sector {idx}" for idx in range(n_sectors)]


def sector_legend_df(sector_codes):
    return pd.DataFrame(
        {
            "Code": sector_codes,
            "Sector": [SECTOR_CODE_TO_NAME.get(code, code) for code in sector_codes],
        }
    )


def sum_to_sector_panel(values, columns=None, n_sectors=None):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values.reshape((-1, 1))

    if values.ndim == 2:
        if columns is None:
            return values
        columns = np.asarray(columns)
        if columns.ndim != 2 or columns.shape[1] < 2:
            return values
        inferred_n_sectors = int(columns[:, 1].max()) + 1 if columns.size else values.shape[1]
        n_sectors = n_sectors or inferred_n_sectors
        out = np.zeros((values.shape[0], n_sectors), dtype=float)
        for col_idx, column in enumerate(columns):
            sector = int(column[1])
            if sector < n_sectors:
                out[:, sector] += values[:, col_idx]
        return out

    if values.ndim >= 3:
        return values.sum(axis=tuple(range(1, values.ndim - 1)))

    raise ValueError("Sectoral transaction data must have time and sector dimensions.")


def read_sector_panel(data, dataset_path, n_sectors=None):
    columns_path = f"{dataset_path}_columns"
    columns = data.get(columns_path)
    return sum_to_sector_panel(data[dataset_path], columns=columns, n_sectors=n_sectors)


def build_sectoral_price_inputs(data, country):
    price_path = f"{country}/economy/good_prices"
    if price_path not in data:
        raise KeyError(price_path)

    prices = np.asarray(data[price_path], dtype=float)
    if prices.ndim != 2:
        raise ValueError(f"{price_path} must be a 2D time x sector dataset.")

    n_sectors = prices.shape[1]
    sector_codes = infer_sector_codes(data, country, n_sectors)

    quantity_paths = [
        f"{country}/firms/real_amount_bought",
        f"{country}/households/real_amount_bought",
        f"{country}/government_entities/real_amount_bought",
    ]
    missing_paths = [path for path in quantity_paths if path not in data]
    if missing_paths:
        raise KeyError(", ".join(missing_paths))

    panels = [read_sector_panel(data, path, n_sectors=n_sectors) for path in quantity_paths]
    n_periods = min(prices.shape[0], *(panel.shape[0] for panel in panels))
    real_quantities = sum(panel[:n_periods, :n_sectors] for panel in panels)

    weights = np.full((n_periods, n_sectors), np.nan)
    totals = real_quantities.sum(axis=1)
    valid = totals != 0.0
    weights[valid] = real_quantities[valid] / totals[valid, None]

    return prices[:n_periods, :n_sectors], weights, sector_codes


def country_entry(mapping, country_iso3):
    if country_iso3 in mapping:
        return mapping[country_iso3]
    for key, value in mapping.items():
        key_value = getattr(key, "value", key)
        if key_value == country_iso3:
            return value
    raise KeyError(f"Country {country_iso3!r} not found. Available keys: {list(mapping.keys())}")


def timestep_for_period(start_year, start_quarter, year, quarter, time_unit):
    periods_per_year = int(round(12 / time_unit)) if time_unit else 4
    return (year - start_year) * periods_per_year + (quarter - start_quarter)


def timestep_indices_for_year(start_year, start_quarter, year, time_unit, n_steps):
    periods_per_year = int(round(12 / time_unit)) if time_unit else 4
    indices = [
        timestep_for_period(start_year, start_quarter, year, period + 1, time_unit)
        for period in range(periods_per_year)
    ]
    if any(index < 0 or index >= n_steps for index in indices):
        raise ValueError(f"Model output does not contain a full year for {year}.")
    return indices


def clean_values(values, *, positive=False):
    array = np.asarray(values, dtype=float).ravel()
    array = array[np.isfinite(array)]
    return array[array > 0] if positive else array


def model_annual_household_values(values, year, start_year, start_quarter, time_unit, *, scale=1.0):
    values = np.asarray(values, dtype=float)
    indices = timestep_indices_for_year(start_year, start_quarter, year, time_unit, values.shape[0])
    return clean_values(values[indices].sum(axis=0) * scale)


def complete_model_years(start_year, start_quarter, time_unit, n_steps):
    periods_per_year = int(round(12 / time_unit)) if time_unit else 4
    last_possible_year = start_year + int(np.ceil((start_quarter - 1 + n_steps) / periods_per_year))
    years = []
    for year in range(start_year, last_possible_year + 1):
        try:
            timestep_indices_for_year(start_year, start_quarter, year, time_unit, n_steps)
        except ValueError:
            continue
        years.append(year)
    return years


@st.cache_data(show_spinner=False)
def load_distribution_context(data_pkl_path, raw_data_path, country_iso3, country_iso2, hfcs_years):
    data = DataWrapper.init_from_pickle(data_pkl_path)
    synthetic_country = country_entry(data.synthetic_countries, country_iso3)
    population = synthetic_country.population

    model_scale = float(getattr(population, "scale", getattr(synthetic_country, "scale", 1.0)))
    start_year = int(getattr(data.configuration, "year", getattr(population, "year", 2014)))
    start_quarter = int(getattr(data.configuration, "quarter", 1) or 1)
    time_unit = int(getattr(data.configuration, "time_unit", getattr(data, "time_unit", 3)) or 3)

    exchange_rates = ExchangeRatesReader.from_csv(Path(raw_data_path) / "exchange_rates" / "exchange_rates.csv")
    hfcs_wave_frames = dv.load_hfcs_wave_dataframes(
        Path(raw_data_path) / "hfcs",
        country_iso2,
        hfcs_years,
        eur_to_lcu_by_year={year: exchange_rates.from_eur_to_lcu(country_iso3, year) for year in hfcs_years},
    )

    pooled_hfcs_frame = pd.concat(hfcs_wave_frames.values(), ignore_index=True)
    return {
        "model_scale": model_scale,
        "unit_scale": 1.0 / model_scale,
        "start_year": start_year,
        "start_quarter": start_quarter,
        "time_unit": time_unit,
        "hfcs_wave_frames": hfcs_wave_frames,
        "pooled_hfcs_frame": pooled_hfcs_frame,
    }


def build_end_simulation_distribution_charts(
    *,
    income,
    wealth,
    context,
    trim_percentile,
    n_bins,
):
    unit_scale = context["unit_scale"]
    start_year = context["start_year"]
    start_quarter = context["start_quarter"]
    time_unit = context["time_unit"]
    pooled_hfcs_frame = context["pooled_hfcs_frame"]
    hfcs_years = sorted(context["hfcs_wave_frames"])
    comparison_year_label = f"{min(hfcs_years)}-{max(hfcs_years)}"

    years = complete_model_years(start_year, start_quarter, time_unit, income.shape[0])
    if not years:
        raise ValueError("Model output does not contain one complete calendar year for annual income comparison.")

    end_income_year = max(years)
    end_step_idx = income.shape[0] - 1
    end_sim_series = {
        "Income": {
            f"Model final year {end_income_year}": model_annual_household_values(
                income,
                end_income_year,
                start_year,
                start_quarter,
                time_unit,
                scale=unit_scale,
            ),
            f"HFCS pooled {comparison_year_label}": clean_values(pooled_hfcs_frame["Income"]),
        },
        "Wealth": {
            f"Model final step {end_step_idx}": dv.prepare_model_values(
                wealth,
                timestep=end_step_idx,
                scale=unit_scale,
                drop_nonpositive=True,
            ),
            f"HFCS pooled {comparison_year_label}": clean_values(pooled_hfcs_frame["Wealth"], positive=True),
        },
    }

    figures = {}
    summary_rows = []
    for label, series_by_name in end_sim_series.items():
        positive_only = label == "Wealth"
        display_label = "Annual Income" if label == "Income" else label
        trimmed_series, upper_limit = dv.trim_series_to_common_percentile(
            series_by_name,
            percentile=trim_percentile,
            lower_bound=0.0 if positive_only else None,
        )
        title_suffix = f"(pooled HFCS {comparison_year_label}; common p{trim_percentile:g} cutoff: {upper_limit:,.0f})"
        figures[label] = {
            "histogram": dv.build_multi_histogram_figure(
                trimmed_series,
                title=f"End-of-simulation {display_label}: model vs pooled HFCS histogram {title_suffix}",
                xaxis_title=f"{display_label}",
                nbinsx=n_bins,
            ),
            "cdf": dv.build_multi_cdf_figure(
                trimmed_series,
                title=f"End-of-simulation {display_label}: model vs pooled HFCS CDF {title_suffix}",
                xaxis_title=f"{display_label}",
            ),
        }

        for source, values in series_by_name.items():
            cleaned = clean_values(values, positive=positive_only)
            summary_rows.append(
                {
                    "metric": display_label,
                    "source": source,
                    "observations": len(cleaned),
                    "mean": float(np.mean(cleaned)),
                    "median": float(np.median(cleaned)),
                    "p90": float(np.percentile(cleaned, 90)),
                    "p99": float(np.percentile(cleaned, 99)),
                }
            )

    return figures, pd.DataFrame(summary_rows), end_income_year, end_step_idx


# Function to create fan chart
def create_fan_chart(df, title):
    # Calculate statistics
    mean = df.mean()
    q1 = df.quantile(0.25)  # First quartile
    q3 = df.quantile(0.75)  # Third quartile
    d1 = df.quantile(0.1)  # First decile
    d9 = df.quantile(0.9)  # Ninth decile

    # Create time points
    time_points = np.arange(len(df))

    # Create the fan chart
    fig = go.Figure()

    # Add the mean line
    fig.add_trace(go.Scatter(x=time_points, y=mean, name="Mean", line=dict(color="black", width=2)))

    # Add deciles (from outer to inner)
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=d9,
            fill=None,
            mode="lines",
            line_color="rgba(0,100,80,0.2)",
            name="90th Percentile",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=d1,
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,100,80,0.2)",
            name="10th-90th Percentile",
        )
    )

    # Add quartiles (from outer to inner)
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=q3,
            fill=None,
            mode="lines",
            line_color="rgba(0,100,80,0.4)",
            name="75th Percentile",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=q1,
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,100,80,0.4)",
            name="25th-75th Percentile",
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Value",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


# Load data
try:
    data = load_h5_data(selected_file)

    # Sidebar for dataset selection
    st.sidebar.header("Variable Selection")

    # Get initial lists of all components
    countries = sorted(list(set(parse_dataset_name(name)[0] for name in data.keys() if parse_dataset_name(name)[0])))

    # Create vertically stacked dropdowns in the sidebar
    selected_country = st.sidebar.selectbox("Country", countries)

    # Get available agent/markets based on selected country
    available_agent_markets, _ = get_available_options(data, selected_country)

    selected_agent_market = st.sidebar.selectbox("Agent/Market", available_agent_markets)

    # Get available variables based on selected country and agent/market
    _, available_variables = get_available_options(data, selected_country, selected_agent_market)

    selected_variable = st.sidebar.selectbox("Variable", available_variables)

    # Construct the full dataset name
    selected_dataset = f"{selected_country}/{selected_agent_market}/{selected_variable}"

    variable_tab, sectoral_prices_tab, hfcs_tab = st.tabs(
        ["Variable Explorer", "Sectoral Prices", "End Simulation HFCS"]
    )

    with variable_tab:
        st.header(f"Variable: {selected_dataset}")

        # Convert selected dataset to DataFrame if possible
        try:
            df = pd.DataFrame(data[selected_dataset])
            df_t = df.T

            # Display data shape
            st.write(f"Shape: {data[selected_dataset].shape}")

            # Create visualizations based on data shape
            if len(data[selected_dataset].shape) == 2 and data[selected_dataset].shape[1] > 1000:
                # If second dimension is large, show fan chart
                # Transpose DataFrame for fan chart
                st.subheader("Descriptive Statistics")
                st.write(df_t.describe())
                fig = create_fan_chart(df_t, f"Fan Chart of {selected_dataset}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Otherwise show time series
                # Only show descriptive statistics if there are multiple observations
                if data[selected_dataset].shape[1] > 1:
                    st.subheader("Descriptive Statistics")
                    st.write(df_t.describe())
                fig = px.line(df, title=f"Time Series of {selected_dataset}")
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Could not convert dataset to DataFrame: {str(e)}")
            st.write("Raw data shape:", data[selected_dataset].shape)
            st.write("Raw data sample:", data[selected_dataset][:5])

    with sectoral_prices_tab:
        st.header(f"Sectoral Prices: {selected_country}")
        try:
            sectoral_prices, sector_weights, sector_codes = build_sectoral_price_inputs(data, selected_country)

            selected_sector_codes = st.multiselect(
                "Sectors",
                options=sector_codes,
                default=sector_codes,
                format_func=lambda code: f"{code}: {SECTOR_CODE_TO_NAME.get(code, code)}",
            )
            plot_kind = st.radio("Plot type", ["heatmap", "lines"], horizontal=True)
            normalize_prices = st.checkbox("Normalize prices to initial value", value=True)

            with st.expander("Sector code mapping", expanded=False):
                st.dataframe(sector_legend_df(sector_codes), hide_index=True, use_container_width=True)

            if not selected_sector_codes:
                st.warning("Select at least one sector to show the sectoral price figures.")
            else:
                price_fig, contribution_fig = plot_sectoral_prices_over_time(
                    sectoral_prices=sectoral_prices,
                    sector_names=sector_codes,
                    sector_weights=sector_weights,
                    sectors=selected_sector_codes,
                    normalize=normalize_prices,
                    kind=plot_kind,
                    show=False,
                    print_sector_names=False,
                )
                st.plotly_chart(price_fig, use_container_width=True)
                st.plotly_chart(contribution_fig, use_container_width=True)
        except KeyError as e:
            st.warning(f"Sectoral price inputs are not available in this H5 file: {e}")
        except Exception as e:
            st.error(f"Could not build sectoral price figures: {str(e)}")

    with hfcs_tab:
        st.header(f"End-of-Simulation Household Distributions: {selected_country}")
        st.caption("Compares final model household income and wealth distributions with pooled HFCS waves.")

        cfg = Config().from_env()
        default_data_pkl_path = selected_file.parent / "data.pkl"
        if not default_data_pkl_path.exists():
            default_data_pkl_path = cfg.output_path / "data.pkl"
        default_raw_data_path = cfg.raw_data_path
        default_country_iso2 = getattr(cfg, "country_iso2", selected_country[:2])

        with st.expander("HFCS input settings", expanded=False):
            data_pkl_path = Path(st.text_input("Processed data pickle", value=str(default_data_pkl_path))).expanduser()
            raw_data_path = Path(st.text_input("Raw data path", value=str(default_raw_data_path))).expanduser()
            country_iso2 = st.text_input("HFCS country ISO2", value=default_country_iso2).upper()
            hfcs_years_text = st.text_input("HFCS years", value="2014, 2017, 2021")
            trim_percentile = st.number_input(
                "Common trim percentile",
                min_value=50.0,
                max_value=100.0,
                value=99.0,
                step=0.5,
            )
            n_bins = st.number_input("Histogram bins", min_value=10, max_value=300, value=80, step=10)

        try:
            hfcs_years = [int(year.strip()) for year in hfcs_years_text.split(",") if year.strip()]
            if not hfcs_years:
                raise ValueError("Enter at least one HFCS year.")

            income_path = f"{selected_country}/households/income"
            wealth_path = f"{selected_country}/households/wealth"
            if income_path not in data or wealth_path not in data:
                raise KeyError(f"{income_path} and/or {wealth_path}")

            context = load_distribution_context(
                data_pkl_path=data_pkl_path,
                raw_data_path=raw_data_path,
                country_iso3=selected_country,
                country_iso2=country_iso2,
                hfcs_years=hfcs_years,
            )
            figures, summary, end_income_year, end_step_idx = build_end_simulation_distribution_charts(
                income=np.asarray(data[income_path], dtype=float),
                wealth=np.asarray(data[wealth_path], dtype=float),
                context=context,
                trim_percentile=float(trim_percentile),
                n_bins=int(n_bins),
            )

            st.write(
                f"Model income uses final complete year {end_income_year}; "
                f"wealth uses final simulation step {end_step_idx}."
            )
            st.dataframe(summary, hide_index=True, use_container_width=True)

            for label in ("Income", "Wealth"):
                st.subheader("Annual Income" if label == "Income" else "Wealth")
                st.plotly_chart(figures[label]["histogram"], use_container_width=True)
                st.plotly_chart(figures[label]["cdf"], use_container_width=True)
        except FileNotFoundError as e:
            st.warning(f"HFCS input file not found: {e}")
        except KeyError as e:
            st.warning(f"Required model or HFCS data is not available: {e}")
        except Exception as e:
            st.error(f"Could not build HFCS comparison charts: {str(e)}")

except Exception as e:
    st.error(f"Error loading H5 file: {str(e)}")
