"""Simulated-vs-actual validation: availability, ACF/CCF statistics, and charts.

Used by the "Validation" section of `run_model.ipynb`. See module-level
docstrings on each function for the alignment/deflation/rate-convention
assumptions; the notebook only supplies `data`, `df_scenario`, `cfg`, and
`COUNTRY`, plus the user-facing transient and lag settings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, ccf

from macro_data.readers.economic_data.policy_rates import PolicyRatesReader
from macro_data.util.frequency import annual_to_period

# Simulated/actual column pairs for the validation comparison.
#
# Deflation: the model builds `real_gdp_ppi = gdp / ppi` (visual_helpers.py) and
# the actuals build "Real GDP (Value)" as nominal/PPI. Series carrying a
# `*_deflator` key are put on that same PPI-real basis so both sides match.
#
# Investment: the actuals have no firm-only and no real GFCF series, so this
# compares the model's own `gfcf` against total actual GFCF, both PPI-deflated.
# Pairing firm-only net capital investment against total gross GFCF would mix a
# net, firm-level measure with a gross, all-sector one.
VALIDATION_SERIES = {
    "real_gdp": {
        "simulated": "real_gdp_ppi",  # gdp / ppi, same basis as the actuals
        "actual": "Real GDP (Value)",  # already nominal / PPI
    },
    "real_consumption": {
        "simulated": "household_consumption",
        "simulated_deflator": "ppi",
        "actual": "Real Household Consumption (Value)",  # already PPI-deflated
    },
    "real_investment": {
        "simulated": "gfcf",
        "simulated_deflator": "ppi",
        "actual": "Gross Fixed Capital Formation (Value)",  # nominal; deflated below
        "actual_deflator": "PPI (Value)",
    },
    "unemployment_rate": {
        "simulated": "unemployment_rate",
        "actual": "Unemployment Rate (Value)",
    },
    "cpi": {
        "simulated": "cpi_transaction",
        "actual": "CPI (Value)",
    },
    "central_bank_policy_rate": {
        "simulated": "central_bank_policy_rate",  # per-period rate
        "actual": None,  # resolved separately: not part of data.calibration_data
    },
}

SIM_COLOR = "#1f77b4"
ACT_COLOR = "#d62728"


def load_actual_policy_rate(cfg, data) -> tuple[pd.Series | None, bool]:
    """Load the BIS policy-rate actuals, converted to the model's per-period rate.

    Rate convention (BIS vs. model): `PolicyRatesReader.get_policy_rates` already
    resamples to quarter-start and averages, so the two are index-aligned at
    `time_unit=3`, but their units differ. The BIS series is annualised;
    `central_bank_policy_rate` is a per-period (per-quarter) rate - the model's
    own init divides the BIS rate by `periods_per_year(time_unit)`
    (default_synthetic_central_bank.py:278, "SmoothTaylorRule stores rates in
    per-period units inside the simulation"). The same `annual_to_period` helper
    is reused here.

    Returns (actual_policy_rate, available); actual_policy_rate is None when the
    reader/path assumptions fail against `cfg.raw_data_path`.
    """
    try:
        policy_rates_reader = PolicyRatesReader(
            path=Path(cfg.raw_data_path) / "policy_rates" / "bis_cb_policy_rates.csv",
            country_code_path=Path(cfg.raw_data_path) / "notation" / "wikipedia-iso-country-codes.csv",
        )
        actual_policy_rate_annual = policy_rates_reader.get_policy_rates(cfg.country_iso3)
        actual_policy_rate = annual_to_period(actual_policy_rate_annual, data.time_unit, column="Policy Rate")
        return actual_policy_rate, not actual_policy_rate.empty
    except Exception as exc:  # reader/path assumptions above are unverified against this raw_data_path
        print(f"Policy-rate actuals: could not load ({exc!r}); path assumptions need checking.")
        return None, False


def build_validation_availability_table(
    data, df_scenario: pd.DataFrame, country_code: str, actual_policy_rate: pd.Series | None
) -> pd.DataFrame:
    """Availability check for the simulated-vs-actual comparison, before any statistic is computed.

    Simulated side: `df_scenario` columns. Actual side: `data.calibration_data[country_code]`
    (Eurostat/IMF/World Bank-derived national accounts and labour stats), plus the BIS
    policy-rate reader loaded separately via `load_actual_policy_rate`.
    """
    calibration_columns = (
        data.calibration_data[country_code].columns
        if country_code in data.calibration_data.columns.get_level_values(0)
        else pd.Index([])
    )

    rows = []
    for label, spec in VALIDATION_SERIES.items():
        sim_cols = [spec["simulated"]] + ([spec["simulated_deflator"]] if "simulated_deflator" in spec else [])
        sim_available = all(col in df_scenario.columns for col in sim_cols)

        if label == "central_bank_policy_rate":
            act_available = actual_policy_rate is not None
            act_col = "Policy Rate (BIS, quarterly, converted to per-period)"
        else:
            act_cols = [spec["actual"]] + ([spec["actual_deflator"]] if "actual_deflator" in spec else [])
            act_available = all(col in calibration_columns for col in act_cols)
            act_col = " / ".join(act_cols)

        rows.append(
            {
                "series": label,
                "simulated_column": " / ".join(sim_cols),
                "simulated_available": sim_available,
                "actual_column": act_col,
                "actual_available": act_available,
                "both_available": sim_available and act_available,
            }
        )

    return pd.DataFrame(rows).set_index("series")


def _simulated_series(label: str, df_scenario: pd.DataFrame) -> pd.Series | None:
    """Model-period-indexed series, PPI-deflated where the spec asks for it."""
    spec = VALIDATION_SERIES[label]
    if spec["simulated"] not in df_scenario.columns:
        return None
    series = df_scenario[spec["simulated"]].astype(float)
    deflator_col = spec.get("simulated_deflator")
    if deflator_col is not None:
        if deflator_col not in df_scenario.columns:
            return None
        series = series / df_scenario[deflator_col].astype(float)
    return series


def _actual_series(
    label: str, data, country_code: str, calibration_columns: pd.Index, actual_policy_rate: pd.Series | None
) -> pd.Series | None:
    """Date-indexed actual series, PPI-deflated where the spec asks for it."""
    if label == "central_bank_policy_rate":
        return None if actual_policy_rate is None else actual_policy_rate["Policy Rate"].astype(float)
    spec = VALIDATION_SERIES[label]
    if spec["actual"] not in calibration_columns:
        return None
    country_actuals = data.calibration_data[country_code]
    series = country_actuals[spec["actual"]].astype(float)
    deflator_col = spec.get("actual_deflator")
    if deflator_col is not None:
        if deflator_col not in calibration_columns:
            return None
        series = series / country_actuals[deflator_col].astype(float)
    return series


def _two_sided_ccf(x: np.ndarray, y: np.ndarray, max_lag: int) -> pd.Series:
    """Cross-correlation of x against y at lags -max_lag..max_lag.

    Positive lag k: corr(x[t], y[t-k]), i.e. y leads x by k periods.
    statsmodels.ccf(x, y) returns only non-negative lags, so the negative side
    comes from ccf(y, x) reversed.
    """
    forward = ccf(x, y, adjusted=True, nlags=max_lag + 1)  # lags 0..max_lag
    backward = ccf(y, x, adjusted=True, nlags=max_lag + 1)  # roles swapped
    lags = list(range(-max_lag, max_lag + 1))
    values = list(backward[1 : max_lag + 1][::-1]) + list(forward)
    return pd.Series(values, index=lags)


def _validation_ccf(series: pd.Series, gdp: pd.Series, max_lag: int, label: str, sample: str) -> pd.Series:
    """Cross-correlate a series and GDP on their common periods within one sample."""
    aligned = pd.concat([series, gdp], axis=1, join="inner").dropna()
    if len(aligned) <= 2:
        print(f"{label}: only {len(aligned)} {sample} periods overlap with real GDP; cross-correlation skipped.")
        return pd.Series(dtype=float)
    return _two_sided_ccf(aligned.iloc[:, 0].to_numpy(), aligned.iloc[:, 1].to_numpy(), min(max_lag, len(aligned) - 1))


def compute_validation_results(
    data,
    df_scenario: pd.DataFrame,
    country_code: str,
    actual_policy_rate: pd.Series | None,
    max_acf_lags: int = 12,
    max_ccf_lags: int = 8,
) -> tuple[dict, pd.Series]:
    """Compute statistics independently for simulated and actual samples.

    The caller may exclude initial model periods from `df_scenario`. Actual
    data are not trimmed to the model's dates. Cross-correlations use common
    periods with GDP within each sample, not across the two samples.

    Returns (validation_results, period_to_date): `period_to_date` maps each
    model period to its calendar quarter-start date, using the configured
    starting quarter and the original model-period index.
    """
    calibration_columns = (
        data.calibration_data[country_code].columns
        if country_code in data.calibration_data.columns.get_level_values(0)
        else pd.Index([])
    )

    period_start_date = pd.Timestamp(data.configuration.year, 3 * (data.configuration.quarter - 1) + 1, 1)
    period_index = df_scenario.index
    period_dates = pd.DatetimeIndex([period_start_date + pd.DateOffset(months=3 * int(t)) for t in period_index])
    period_to_date = pd.Series(period_dates, index=period_index)

    gdp_simulated = _simulated_series("real_gdp", df_scenario)
    gdp_actual = _actual_series("real_gdp", data, country_code, calibration_columns, actual_policy_rate)
    if gdp_simulated is None or gdp_simulated.dropna().empty or gdp_actual is None or gdp_actual.dropna().empty:
        raise ValueError(
            "real_gdp simulated or actual series is unavailable or empty; cannot compute cross-correlations."
        )

    validation_results: dict = {}
    for label in VALIDATION_SERIES:
        simulated = _simulated_series(label, df_scenario)
        actual = _actual_series(label, data, country_code, calibration_columns, actual_policy_rate)
        if simulated is None or actual is None:
            print(f"{label}: simulated or actual series unavailable; skipping.")
            continue
        simulated = simulated.dropna()
        actual = actual.dropna()
        if simulated.empty or actual.empty:
            print(f"{label}: simulated or actual series empty; skipping.")
            continue

        def sample_acf(series: pd.Series, sample: str) -> pd.Series:
            if len(series) < 2:
                print(f"{label}: only {len(series)} {sample} period; autocorrelation skipped.")
                return pd.Series(dtype=float)
            return pd.Series(acf(series, nlags=min(max_acf_lags, len(series) - 1), fft=True))

        validation_results[label] = {
            "simulated": simulated,
            "actual": actual,
            "acf_simulated": sample_acf(simulated, "model"),
            "acf_actual": sample_acf(actual, "data"),
            "ccf_simulated": _validation_ccf(simulated, gdp_simulated, max_ccf_lags, label, "model"),
            "ccf_actual": _validation_ccf(actual, gdp_actual, max_ccf_lags, label, "data"),
        }

    return validation_results, period_to_date


def build_validation_coverage_table(validation_results: dict, n_total_periods: int) -> pd.DataFrame:
    """Available model and data periods per validation series."""
    return pd.DataFrame(
        {
            label: {
                "n_model_periods": len(r["simulated"]),
                "n_data_periods": len(r["actual"]),
                "n_total_model_periods": n_total_periods,
            }
            for label, r in validation_results.items()
        }
    ).T


def plot_validation_charts(validation_results: dict, period_to_date: pd.Series, country_code: str) -> go.Figure:
    """One row per validation variable, three panels each: level (simulated vs.
    actual, shared y-axis), autocorrelation, and cross-correlation against real
    GDP. The model's period index is mapped to dates via `period_to_date`;
    actual values retain their own dates.
    """
    available_labels = [label for label in VALIDATION_SERIES if label in validation_results]

    # Declutter: name the variable once per row (left panel), and label the
    # statistic columns only on the top row.
    subplot_titles = []
    for row_idx, label in enumerate(available_labels):
        subplot_titles += [
            label,
            "Autocorrelation" if row_idx == 0 else "",
            "Cross-correlation vs. real GDP" if row_idx == 0 else "",
        ]

    fig = make_subplots(
        rows=len(available_labels),
        cols=3,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.07,
    )

    line_kw = {"mode": "lines+markers", "marker": {"size": 4}}

    for row, label in enumerate(available_labels, start=1):
        result = validation_results[label]
        simulated, actual = result["simulated"], result["actual"]
        first = row == 1

        # Panel 1: levels over calendar time, shared y-axis.
        fig.add_trace(
            go.Scatter(
                x=period_to_date.reindex(simulated.index),
                y=simulated,
                name="simulated",
                line={"color": SIM_COLOR},
                legendgroup="simulated",
                showlegend=first,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual,
                name="actual",
                line={"color": ACT_COLOR},
                legendgroup="actual",
                showlegend=first,
            ),
            row=row,
            col=1,
        )

        # Panel 2: autocorrelation.
        acf_sim, acf_act = result["acf_simulated"], result["acf_actual"]
        fig.add_trace(
            go.Scatter(
                x=acf_sim.index,
                y=acf_sim.values,
                line={"color": SIM_COLOR},
                legendgroup="simulated",
                showlegend=False,
                **line_kw,
            ),
            row=row,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=acf_act.index,
                y=acf_act.values,
                line={"color": ACT_COLOR},
                legendgroup="actual",
                showlegend=False,
                **line_kw,
            ),
            row=row,
            col=2,
        )
        fig.add_hline(y=0.0, line={"color": "rgba(0,0,0,0.35)", "width": 1}, row=row, col=2)

        # Panel 3: cross-correlation against real GDP.
        ccf_sim, ccf_act = result["ccf_simulated"], result["ccf_actual"]
        if not ccf_sim.empty:
            fig.add_trace(
                go.Scatter(
                    x=ccf_sim.index,
                    y=ccf_sim.values,
                    line={"color": SIM_COLOR},
                    legendgroup="simulated",
                    showlegend=False,
                    **line_kw,
                ),
                row=row,
                col=3,
            )
        if not ccf_act.empty:
            fig.add_trace(
                go.Scatter(
                    x=ccf_act.index,
                    y=ccf_act.values,
                    line={"color": ACT_COLOR},
                    legendgroup="actual",
                    showlegend=False,
                    **line_kw,
                ),
                row=row,
                col=3,
            )
        if not ccf_sim.empty or not ccf_act.empty:
            fig.add_hline(y=0.0, line={"color": "rgba(0,0,0,0.35)", "width": 1}, row=row, col=3)
            fig.add_vline(x=0, line={"color": "rgba(0,0,0,0.2)", "width": 1, "dash": "dot"}, row=row, col=3)

        fig.update_xaxes(tickformat="%Y", row=row, col=1)

    # Axis labels only on the bottom row, so they are stated once.
    bottom = len(available_labels)
    fig.update_xaxes(title_text="year", row=bottom, col=1)
    fig.update_xaxes(title_text="lag (quarters)", row=bottom, col=2)
    fig.update_xaxes(title_text="lag (quarters; + = GDP leads)", row=bottom, col=3)

    fig.update_annotations(font_size=13)
    fig.update_layout(
        height=260 * len(available_labels),
        width=1350,
        title_text=f"Validation: {country_code} simulated vs. actual",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"t": 90},
    )
    return fig
