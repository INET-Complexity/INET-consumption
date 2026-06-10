"""Analysis helpers for paired impulse-response experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go


@dataclass(frozen=True)
class IRFVariable:
    """HDF5 variable to extract for aggregate impulse-response analysis."""

    name: str
    h5_path: str
    transform: str = "sum"
    real_cpi_path: str | None = None


DEFAULT_IRF_VARIABLES: tuple[IRFVariable, ...] = (
    IRFVariable("household_consumption", "{country}/households/consumption", "sum"),
    IRFVariable("target_household_consumption", "{country}/households/target_consumption", "sum"),
    IRFVariable("household_income", "{country}/households/income", "sum"),
    IRFVariable("government_consumption", "{country}/government_entities/total_consumption", "first"),
    IRFVariable("desired_government_consumption", "{country}/government_entities/desired_consumption_in_lcu", "sum"),
    IRFVariable("policy_rate", "{country}/central_Bank/policy_rate", "first"),
    IRFVariable("cpi", "{country}/economy/cpi_fixed_basket", "first"),
    IRFVariable("gdp_output", "{country}/economy/gdp_output", "first"),
    IRFVariable("gdp_expenditure", "{country}/economy/gdp_expenditure", "first"),
    IRFVariable("gdp_income", "{country}/economy/gdp_income", "first"),
    IRFVariable("unemployment_rate", "{country}/economy/unemployment_rate", "first"),
)
RATE_IRF_VARIABLES = frozenset({"policy_rate", "unemployment_rate"})
MONEY_IRF_VARIABLES = frozenset(
    {
        "household_consumption",
        "target_household_consumption",
        "household_income",
        "government_consumption",
        "desired_government_consumption",
        "gdp_output",
        "gdp_expenditure",
        "gdp_income",
    }
)


def _read_h5_array_from_handle(handle: h5py.File, path_template: str, *, country_code: str) -> np.ndarray | None:
    dataset_path = path_template.format(country=country_code)
    if dataset_path not in handle:
        return None
    return np.asarray(handle[dataset_path], dtype=float)


def _as_series(values: np.ndarray, *, transform: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return values
    flattened = values.reshape(values.shape[0], -1)
    if transform == "sum":
        return np.nansum(flattened, axis=1)
    if transform == "mean":
        return np.nanmean(flattened, axis=1)
    if transform == "first":
        return flattened[:, 0]
    raise ValueError("IRF variable transform must be one of 'sum', 'mean', or 'first'.")


def build_irf_panel(
    *,
    baseline_h5: str | Path,
    shock_h5: str | Path,
    seed: int,
    shock_name: str,
    shock_kind: str,
    shock_period: int,
    shock_magnitude: float,
    horizon_periods: int,
    country_code: str,
    variables: tuple[IRFVariable, ...] = DEFAULT_IRF_VARIABLES,
    strict: bool = True,
) -> pd.DataFrame:
    """Build a long impulse-response panel from paired baseline and shocked HDF5s."""

    if shock_period < 0:
        raise ValueError("shock_period must be non-negative.")
    if horizon_periods <= 0:
        raise ValueError("horizon_periods must be positive.")

    rows: list[dict[str, float | int | str]] = []
    missing: list[str] = []
    with h5py.File(baseline_h5, "r") as baseline_handle, h5py.File(shock_h5, "r") as shock_handle:
        for variable in variables:
            dataset_path = variable.h5_path.format(country=country_code)
            baseline_raw = _read_h5_array_from_handle(baseline_handle, variable.h5_path, country_code=country_code)
            shock_raw = _read_h5_array_from_handle(shock_handle, variable.h5_path, country_code=country_code)
            if baseline_raw is None or shock_raw is None:
                missing_sides = []
                if baseline_raw is None:
                    missing_sides.append("baseline")
                if shock_raw is None:
                    missing_sides.append("shock")
                missing.append(f"{variable.name} at {dataset_path} missing from {', '.join(missing_sides)}")
                continue
            baseline = _as_series(baseline_raw, transform=variable.transform)
            shocked = _as_series(shock_raw, transform=variable.transform)
            if baseline.shape != shocked.shape:
                raise ValueError(f"{variable.name} shape mismatch: {baseline.shape} != {shocked.shape}.")
            stop = min(shock_period + horizon_periods, baseline.shape[0])
            if stop <= shock_period:
                raise ValueError(f"Requested horizon starts beyond saved rows for {variable.name}.")
            for row in range(shock_period, stop):
                baseline_value = float(baseline[row])
                shock_value = float(shocked[row])
                delta = shock_value - baseline_value
                pct_delta = np.nan if baseline_value == 0.0 else delta / baseline_value
                rows.append(
                    {
                        "seed": int(seed),
                        "shock_name": shock_name,
                        "shock_kind": shock_kind,
                        "shock_period": int(shock_period),
                        "shock_magnitude": float(shock_magnitude),
                        "horizon": int(row - shock_period),
                        "row": int(row),
                        "variable": variable.name,
                        "baseline": baseline_value,
                        "shock": shock_value,
                        "delta": delta,
                        "pct_delta": pct_delta,
                    }
                )
    if strict and missing:
        raise ValueError("Missing IRF HDF5 variables: " + "; ".join(missing))
    return pd.DataFrame(rows)


def summarize_irf_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize IRFs across seeds by shock, variable, and horizon."""

    if panel.empty:
        return panel.copy()
    grouped = panel.groupby(["shock_name", "shock_kind", "variable", "horizon"], dropna=False)
    return grouped.agg(
        n=("delta", "size"),
        delta_mean=("delta", "mean"),
        delta_median=("delta", "median"),
        delta_p10=("delta", lambda values: values.quantile(0.10)),
        delta_p90=("delta", lambda values: values.quantile(0.90)),
        pct_delta_mean=("pct_delta", "mean"),
        pct_delta_median=("pct_delta", "median"),
        pct_delta_p10=("pct_delta", lambda values: values.quantile(0.10)),
        pct_delta_p90=("pct_delta", lambda values: values.quantile(0.90)),
    ).reset_index()


def _as_filter_values(values: str | list[str] | tuple[str, ...] | set[str] | None) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    return list(values)


def _load_irf_summary(summary: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(summary, pd.DataFrame):
        return summary.copy()
    path = Path(summary)
    if path.is_dir():
        path = path / "irf_summary.csv"
    return pd.read_csv(path)


def plot_irfs(
    summary: pd.DataFrame | str | Path,
    *,
    shocks: str | list[str] | tuple[str, ...] | set[str] | None = None,
    variables: str | list[str] | tuple[str, ...] | set[str] | None = None,
    percent: bool = False,
    scale_money: bool = True,
    money_scale: float = 1e9,
    money_unit: str = "billion LCU",
    title: str | None = None,
) -> go.Figure:
    """Build one notebook-friendly IRF figure for selected shocks and variables.

    Args:
        summary: IRF summary DataFrame, path to ``irf_summary.csv``, or analysis directory.
        shocks: Shock name or names to include. Defaults to all shocks.
        variables: Variable name or names to include. Defaults to all variables.
        percent: Plot percentage deviations instead of absolute deltas.
        scale_money: Divide monetary aggregate deltas by ``money_scale`` for readability.
        money_scale: Scale applied to monetary aggregate deltas when ``scale_money=True``.
        money_unit: Axis unit label for scaled monetary aggregate deltas.
        title: Optional Plotly title.
    """

    data = _load_irf_summary(summary)
    shock_filter = _as_filter_values(shocks)
    variable_filter = _as_filter_values(variables)
    if shock_filter is not None:
        data = data[data["shock_name"].isin(shock_filter)]
    if variable_filter is not None:
        data = data[data["variable"].isin(variable_filter)]
    if data.empty:
        raise ValueError("No IRF rows match the requested shocks and variables.")

    value_prefix = "pct_delta" if percent else "delta"
    mean_col = f"{value_prefix}_mean"
    p10_col = f"{value_prefix}_p10"
    p90_col = f"{value_prefix}_p90"
    data = data.sort_values(["shock_name", "variable", "horizon"]).copy()
    data["plot_mean"] = data[mean_col].astype(float)
    data["plot_p10"] = data[p10_col].astype(float)
    data["plot_p90"] = data[p90_col].astype(float)

    yaxis_title = "Percent deviation" if percent else "Delta"
    if not percent and scale_money:
        money_mask = data["variable"].isin(MONEY_IRF_VARIABLES)
        data.loc[money_mask, ["plot_mean", "plot_p10", "plot_p90"]] /= float(money_scale)
        if money_mask.all():
            yaxis_title = f"Delta, {money_unit}"
        elif money_mask.any():
            yaxis_title = f"Delta, mixed units; monetary variables in {money_unit}"

    fig = go.Figure()
    for (shock_name, variable), group in data.groupby(["shock_name", "variable"], sort=False):
        label = f"{shock_name}: {variable}"
        fig.add_trace(
            go.Scatter(
                x=group["horizon"],
                y=group["plot_mean"],
                mode="lines+markers",
                name=label,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([group["horizon"], group["horizon"].iloc[::-1]]),
                y=pd.concat([group["plot_p90"], group["plot_p10"].iloc[::-1]]),
                fill="toself",
                line={"color": "rgba(0,0,0,0)"},
                name=f"{label} p10-p90",
                showlegend=False,
            )
        )
    fig.update_layout(
        title_text=title or ("IRF percent deviations" if percent else "IRF deltas"),
        template="plotly_white",
        xaxis_title="horizon",
        yaxis_title=yaxis_title,
    )
    return fig


def write_irf_plots(
    summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    value_column: str = "delta",
    skip_variables: frozenset[str] | set[str] | tuple[str, ...] | None = None,
) -> None:
    """Write one Plotly IRF chart per shock and variable."""

    if value_column not in {"delta", "pct_delta"}:
        raise ValueError("value_column must be 'delta' or 'pct_delta'.")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    if summary.empty:
        return
    skipped = RATE_IRF_VARIABLES if skip_variables is None and value_column == "pct_delta" else (skip_variables or ())
    mean_col = f"{value_column}_mean"
    p10_col = f"{value_column}_p10"
    p90_col = f"{value_column}_p90"
    for (shock_name, variable), group in summary.groupby(["shock_name", "variable"], dropna=False):
        if variable in skipped:
            continue
        group = group.sort_values("horizon")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=group["horizon"],
                y=group[mean_col],
                mode="lines+markers",
                name="mean",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([group["horizon"], group["horizon"].iloc[::-1]]),
                y=pd.concat([group[p90_col], group[p10_col].iloc[::-1]]),
                fill="toself",
                line={"color": "rgba(0,0,0,0)"},
                name="p10-p90",
            )
        )
        fig.update_layout(
            title_text=f"{shock_name}: {variable}",
            template="plotly_white",
            xaxis_title="horizon",
            yaxis_title=value_column,
        )
        fig.write_html(output_path / f"{shock_name}_{variable}_{value_column}.html")
