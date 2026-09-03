"""Simulated-vs-actual validation: availability, ACF/CCF statistics, and charts.

Used by the "Validation" section of `run_model.ipynb`. See module-level
docstrings on each function for the alignment/deflation/rate-convention
assumptions; the notebook only supplies `data`, `df_scenario`, `cfg`, and
`COUNTRY`, plus the two user-facing lag settings.
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
# Deflation: the model builds `real_gdp = gdp / ppi` (visual_helpers.py:836) and
# the actuals build "Real GDP (Value)" as nominal/PPI. Series carrying a
# `*_deflator` key are put on that same PPI-real basis so both sides match.
#
# Investment: the actuals have no firm-only and no real GFCF series, so this
# compares the model's own `gfcf` against total actual GFCF, both PPI-deflated.
# Pairing firm-only net capital investment against total gross GFCF would mix a
# net, firm-level measure with a gross, all-sector one.
VALIDATION_SERIES = {
    "real_gdp": {
        "simulated": "real_gdp",  # already gdp / ppi
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


def _aligned_pair(
    label: str,
    df_scenario: pd.DataFrame,
    data,
    country_code: str,
    calibration_columns: pd.Index,
    actual_policy_rate: pd.Series | None,
    period_index: pd.Index,
    period_dates: pd.DatetimeIndex,
) -> pd.DataFrame | None:
    """Simulated and actual series on the model's period axis, actuals-limited."""
    sim = _simulated_series(label, df_scenario)
    act_dated = _actual_series(label, data, country_code, calibration_columns, actual_policy_rate)
    if sim is None or act_dated is None:
        return None
    aligned = pd.DataFrame(
        {
            "simulated": sim.reindex(period_index).to_numpy(),
            "actual": act_dated.reindex(period_dates).to_numpy(),
        },
        index=period_index,
    )
    return aligned.dropna()


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


def compute_validation_results(
    data,
    df_scenario: pd.DataFrame,
    country_code: str,
    actual_policy_rate: pd.Series | None,
    max_acf_lags: int = 12,
    max_ccf_lags: int = 8,
) -> tuple[dict, pd.Series]:
    """Build aligned simulated/actual series, then compute autocorrelation and
    cross-correlation (vs. real GDP) for each variable in `VALIDATION_SERIES`.

    Alignment: `df_scenario` is indexed by integer model period (t=0 at
    `data.configuration.year`/`quarter`, +1 quarter per period since
    `data.time_unit == 3`). Actual series are date-indexed. Both are put on the
    model's period axis, keeping only periods where an actual quarter exists
    (no extrapolation beyond the historical actuals).

    Returns (validation_results, period_to_date): `period_to_date` maps each
    model period to its calendar quarter-start date, for calendar x-labels.
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

    def aligned_pair(label: str) -> pd.DataFrame | None:
        return _aligned_pair(
            label, df_scenario, data, country_code, calibration_columns, actual_policy_rate, period_index, period_dates
        )

    gdp_pair = aligned_pair("real_gdp")
    if gdp_pair is None or gdp_pair.empty:
        raise ValueError("real_gdp actual/simulated pair is unavailable or empty; cannot compute cross-correlations.")

    validation_results: dict = {}
    for label in VALIDATION_SERIES:
        pair = aligned_pair(label)
        if pair is None or pair.empty:
            print(f"{label}: actual series unavailable or no overlapping periods with df_scenario; skipping.")
            continue

        n = len(pair)
        if n < len(period_index):
            print(f"{label}: actuals cover {n}/{len(period_index)} model periods; statistics use the overlap only.")

        acf_lags = min(max_acf_lags, n - 1)
        acf_simulated = acf(pair["simulated"], nlags=acf_lags, fft=True)
        acf_actual = acf(pair["actual"], nlags=acf_lags, fft=True)

        # Cross-correlation against real GDP, on this variable's own valid periods.
        ccf_valid = pair.assign(
            gdp_simulated=gdp_pair["simulated"].reindex(pair.index).to_numpy(),
            gdp_actual=gdp_pair["actual"].reindex(pair.index).to_numpy(),
        ).dropna()

        if len(ccf_valid) > 2:
            ccf_lags_eff = min(max_ccf_lags, len(ccf_valid) - 1)
            ccf_simulated = _two_sided_ccf(
                ccf_valid["simulated"].to_numpy(), ccf_valid["gdp_simulated"].to_numpy(), ccf_lags_eff
            )
            ccf_actual = _two_sided_ccf(
                ccf_valid["actual"].to_numpy(), ccf_valid["gdp_actual"].to_numpy(), ccf_lags_eff
            )
        else:
            print(f"{label}: only {len(ccf_valid)} overlapping periods with real GDP; cross-correlation skipped.")
            ccf_simulated = pd.Series(dtype=float)
            ccf_actual = pd.Series(dtype=float)

        validation_results[label] = {
            "pair": pair,
            "n_periods": n,
            "acf_simulated": pd.Series(acf_simulated, index=range(acf_lags + 1)),
            "acf_actual": pd.Series(acf_actual, index=range(acf_lags + 1)),
            "ccf_simulated": ccf_simulated,
            "ccf_actual": ccf_actual,
        }

    return validation_results, period_to_date


def build_validation_coverage_table(validation_results: dict, n_total_periods: int) -> pd.DataFrame:
    """Model periods covered by actuals, per validation series."""
    return pd.DataFrame(
        {
            label: {"n_periods": r["n_periods"], "n_total_periods": n_total_periods}
            for label, r in validation_results.items()
        }
    ).T


def plot_validation_charts(
    validation_results: dict, period_to_date: pd.Series, country_code: str
) -> go.Figure:
    """One row per validation variable, three panels each: level (simulated vs.
    actual, shared y-axis), autocorrelation, and cross-correlation against real
    GDP. Column 1's x-axis is labelled with calendar years (via
    `period_to_date`) rather than raw model periods.
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
        pair = result["pair"]
        first = row == 1
        pair_dates = period_to_date.reindex(pair.index)

        # Panel 1: levels over calendar time, shared y-axis.
        fig.add_trace(
            go.Scatter(
                x=pair_dates, y=pair["simulated"], name="simulated", line={"color": SIM_COLOR},
                legendgroup="simulated", showlegend=first,
            ),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pair_dates, y=pair["actual"], name="actual", line={"color": ACT_COLOR},
                legendgroup="actual", showlegend=first,
            ),
            row=row, col=1,
        )

        # Panel 2: autocorrelation.
        acf_sim, acf_act = result["acf_simulated"], result["acf_actual"]
        fig.add_trace(
            go.Scatter(x=acf_sim.index, y=acf_sim.values, line={"color": SIM_COLOR},
                       legendgroup="simulated", showlegend=False, **line_kw),
            row=row, col=2,
        )
        fig.add_trace(
            go.Scatter(x=acf_act.index, y=acf_act.values, line={"color": ACT_COLOR},
                       legendgroup="actual", showlegend=False, **line_kw),
            row=row, col=2,
        )
        fig.add_hline(y=0.0, line={"color": "rgba(0,0,0,0.35)", "width": 1}, row=row, col=2)

        # Panel 3: cross-correlation against real GDP.
        ccf_sim, ccf_act = result["ccf_simulated"], result["ccf_actual"]
        if not ccf_sim.empty:
            fig.add_trace(
                go.Scatter(x=ccf_sim.index, y=ccf_sim.values, line={"color": SIM_COLOR},
                           legendgroup="simulated", showlegend=False, **line_kw),
                row=row, col=3,
            )
            fig.add_trace(
                go.Scatter(x=ccf_act.index, y=ccf_act.values, line={"color": ACT_COLOR},
                           legendgroup="actual", showlegend=False, **line_kw),
                row=row, col=3,
            )
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
