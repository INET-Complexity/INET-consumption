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
if str(RUN_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(RUN_MODEL_PATH))

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

    variable_tab, sectoral_prices_tab = st.tabs(["Variable Explorer", "Sectoral Prices"])

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

except Exception as e:
    st.error(f"Error loading H5 file: {str(e)}")
