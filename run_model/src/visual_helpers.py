import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.helpers import unpack_cell


def _categorical_colors(n_colors):
    if n_colors <= 0:
        return []
    return [f"hsl({int(h)}, 60%, 45%)" for h in np.linspace(0, 330, n_colors)]


SECTOR_CODE_TO_NAME = {
    "A": "Agriculture, forestry and fishing",
    "B": "Mining and quarrying",
    "C": "Manufacturing",
    "D": "Electricity, gas, steam and air conditioning supply",
    "E": "Water supply; sewerage, waste management",
    "F": "Construction",
    "G": "Wholesale and retail trade",
    "H": "Transportation and storage",
    "I": "Accommodation and food service activities",
    "J": "Information and communication",
    "K": "Financial and insurance activities",
    "L": "Real estate activities",
    "M": "Professional, scientific and technical activities",
    "N": "Administrative and support service activities",
    "O": "Public administration and defence",
    "P": "Education",
    "Q": "Human health and social work activities",
    "R": "Arts, entertainment, recreation and other services",
    "R_S": "Arts, entertainment, recreation and other services",
}


def _print_sector_code_names(sector_codes):
    rows = []
    for sector_code in sector_codes:
        sector_code = str(sector_code)
        rows.append(f"{sector_code}: {SECTOR_CODE_TO_NAME.get(sector_code, sector_code)}")
    print("Sector codes:")
    print("\n".join(rows))


def _sectoral_prices_from_model(model, country_code):
    country = model.countries[country_code]
    prices = np.asarray(country.economy.ts.historic("good_prices"), dtype=float)
    sector_names = list(country.firms.industries)
    if prices.ndim != 2:
        raise ValueError("Sectoral prices must be a 2D time x sector array.")
    if len(sector_names) != prices.shape[1]:
        sector_names = [f"sector {idx}" for idx in range(prices.shape[1])]
    return pd.DataFrame(prices, columns=sector_names)


def _sectoral_prices_to_frame(sectoral_prices, model=None, country_code=None, sector_names=None):
    if model is not None:
        if country_code is None:
            raise ValueError("country_code is required when model is provided.")
        return _sectoral_prices_from_model(model, country_code)

    if isinstance(sectoral_prices, pd.DataFrame):
        return sectoral_prices.copy()

    if sectoral_prices is None:
        raise ValueError("Provide either model and country_code or sectoral_prices.")

    prices = np.asarray(sectoral_prices, dtype=float)
    if prices.ndim != 2:
        raise ValueError("sectoral_prices must be a 2D time x sector array.")

    if sector_names is None:
        sector_names = [f"sector {idx}" for idx in range(prices.shape[1])]
    if len(sector_names) != prices.shape[1]:
        raise ValueError("sector_names length must match the number of price sectors.")
    return pd.DataFrame(prices, columns=sector_names)


def _sum_to_sector_panel(values):
    values = np.asarray(values, dtype=float)
    if values.ndim < 2:
        raise ValueError("Sectoral transaction data must have time and sector dimensions.")
    if values.ndim == 2:
        return values
    return values.sum(axis=tuple(range(1, values.ndim - 1)))


def _sectoral_ppi_weights_from_model(model, country_code, index, columns):
    country = model.countries[country_code]
    transaction_panels = [
        _sum_to_sector_panel(country.firms.ts.historic("real_amount_bought")),
        _sum_to_sector_panel(country.households.ts.historic("real_amount_bought")),
        _sum_to_sector_panel(country.government_entities.ts.historic("real_amount_bought")),
    ]
    real_quantities = sum(transaction_panels)
    n_periods = min(len(index), real_quantities.shape[0])
    n_sectors = min(len(columns), real_quantities.shape[1])

    weights = np.full((len(index), len(columns)), np.nan)
    quantities = real_quantities[:n_periods, :n_sectors]
    total_quantities = quantities.sum(axis=1)
    valid = total_quantities != 0.0
    weights[:n_periods, :n_sectors][valid] = quantities[valid] / total_quantities[valid, None]
    return pd.DataFrame(weights, index=index, columns=columns)


def _sectoral_weights_to_frame(sector_weights, prices, model=None, country_code=None):
    if model is not None:
        return _sectoral_ppi_weights_from_model(
            model=model,
            country_code=country_code,
            index=prices.index,
            columns=prices.columns,
        )

    if sector_weights is None:
        equal_weights = np.full(prices.shape, 1.0 / prices.shape[1])
        return pd.DataFrame(equal_weights, index=prices.index, columns=prices.columns)

    if isinstance(sector_weights, pd.DataFrame):
        weights = sector_weights.reindex(index=prices.index, columns=prices.columns)
    else:
        weights = np.asarray(sector_weights, dtype=float)
        if weights.ndim == 1:
            if len(weights) != prices.shape[1]:
                raise ValueError("1D sector_weights length must match the number of price sectors.")
            weights = np.tile(weights, (len(prices.index), 1))
        if weights.shape != prices.shape:
            raise ValueError("sector_weights must match sectoral_prices shape, or be one weight per sector.")
        weights = pd.DataFrame(weights, index=prices.index, columns=prices.columns)

    weight_totals = weights.sum(axis=1).replace(0.0, np.nan)
    return weights.divide(weight_totals, axis="index")


def _build_sectoral_price_figure(
    values,
    kind,
    title,
    value_title,
    color_scale,
    line_width,
    line_opacity,
    height,
    width,
):
    if kind == "heatmap":
        fig = go.Figure(
            data=go.Heatmap(
                z=values.T.values,
                x=values.index,
                y=values.columns,
                colorscale=color_scale,
                colorbar={"title": value_title},
                hovertemplate="time=%{x}<br>sector=%{y}<br>" + value_title + "=%{z:.4g}<extra></extra>",
            )
        )
        fig.update_layout(
            height=height,
            width=width,
            title_text=title,
            template="plotly_white",
            xaxis_title="time",
            yaxis_title="sector",
        )
        return fig

    if kind == "lines":
        colors = _categorical_colors(len(values.columns))
        fig = go.Figure()
        for color, column in zip(colors, values.columns):
            fig.add_trace(
                go.Scatter(
                    x=values.index,
                    y=values[column],
                    mode="lines",
                    name=str(column),
                    line={"width": line_width, "color": color},
                    opacity=line_opacity,
                )
            )
        fig.update_layout(
            height=height,
            width=width,
            title_text=title,
            template="plotly_white",
            xaxis_title="time",
            yaxis_title=value_title,
        )
        return fig

    raise ValueError("kind must be 'heatmap' or 'lines'.")


def plot_sectoral_prices_over_time(
    model=None,
    country_code=None,
    sectoral_prices=None,
    sector_names=None,
    sector_weights=None,
    sectors=None,
    normalize=True,
    kind="heatmap",
    title=None,
    contribution_title=None,
    color_scale="Viridis",
    line_width=1.8,
    line_opacity=0.85,
    height=520,
    width=900,
    print_sector_names=True,
    show=True,
):
    """Plot the evolution of sectoral goods prices over simulation time.

    Parameters
    ----------
    model, country_code:
        Optional simulation model and country code. When supplied, prices are
        read from ``model.countries[country_code].economy.ts.good_prices`` and
        sector labels from ``model.countries[country_code].firms.industries``.
    sectoral_prices:
        Optional dataframe or 2D array with shape ``time x sector``. Used when
        plotting extracted price panels instead of a live model.
    sector_weights:
        Optional sector weights for extracted price panels. With a live model,
        PPI-consistent weights are computed from real transaction quantities:
        firms, households, and government real amounts bought by sector.
    sectors:
        Optional subset of sector labels or integer positions to plot.
    normalize:
        If True, plot each sector as an index relative to its initial price.
    kind:
        ``"heatmap"`` for a compact time-sector view, or ``"lines"`` for one
        trajectory per sector.
    print_sector_names:
        If True, print the sector code-to-name mapping for plotted sectors.

    Returns
    -------
    tuple[plotly.graph_objects.Figure, plotly.graph_objects.Figure] | None
        Returns ``(price_fig, contribution_fig)`` when ``show=False``. Displays
        both figures and returns ``None`` when ``show=True`` to avoid duplicate
        notebook rendering.
    """
    prices = _sectoral_prices_to_frame(
        sectoral_prices=sectoral_prices,
        model=model,
        country_code=country_code,
        sector_names=sector_names,
    )
    weights = _sectoral_weights_to_frame(
        sector_weights=sector_weights,
        prices=prices,
        model=model,
        country_code=country_code,
    )

    if sectors is not None:
        selected_columns = []
        for sector in sectors:
            if isinstance(sector, (int, np.integer)):
                selected_columns.append(prices.columns[int(sector)])
            else:
                selected_columns.append(sector)
        missing = [sector for sector in selected_columns if sector not in prices.columns]
        if missing:
            raise ValueError(f"Requested sectors are not present: {missing}")
        prices = prices[selected_columns]
        weights = weights[selected_columns]

    if prices.empty:
        raise ValueError("No sectoral price data to plot.")

    if print_sector_names:
        _print_sector_code_names(prices.columns)

    price_title = "price index (initial=1)" if normalize else "price"
    if normalize:
        initial_prices = prices.iloc[0].replace(0.0, np.nan)
        prices = prices.divide(initial_prices, axis="columns")

    ppi_contributions = prices * weights
    contribution_value_title = f"PPI contribution to {price_title}"

    if title is None:
        prefix = f"{country_code} " if country_code is not None else ""
        title = f"{prefix}sectoral prices over time"
    if contribution_title is None:
        prefix = f"{country_code} " if country_code is not None else ""
        contribution_title = f"{prefix}sectoral price contributions to PPI"

    price_fig = _build_sectoral_price_figure(
        values=prices,
        kind=kind,
        title=title,
        value_title=price_title,
        color_scale=color_scale,
        line_width=line_width,
        line_opacity=line_opacity,
        height=height,
        width=width,
    )
    contribution_fig = _build_sectoral_price_figure(
        values=ppi_contributions,
        kind=kind,
        title=contribution_title,
        value_title=contribution_value_title,
        color_scale=color_scale,
        line_width=line_width,
        line_opacity=line_opacity,
        height=height,
        width=width,
    )

    if show:
        price_fig.show()
        contribution_fig.show()
        return None
    return price_fig, contribution_fig


def plot_output(
    df,
    no_rows,
    no_cols,
    country_code,
    x_col=None,
    line_color="#1f77b4",
    df_compare=None,
    compare_line_color="#d62728",
    base_name="baseline",
    compare_name="comparison",
    base_height=200,
    base_width=300,
):
    columns_to_plot = df.columns.to_list()
    max_plots = no_rows * no_cols
    columns_to_plot = columns_to_plot[:max_plots]
    if df_compare is not None:
        columns_to_plot = [col for col in columns_to_plot if col in df_compare.columns]

    fig = make_subplots(rows=no_rows, cols=no_cols, subplot_titles=columns_to_plot)

    # choose x axis: explicit column, otherwise index
    if x_col is not None:
        x = df[x_col]
    else:
        x = df.index
    if df_compare is not None:
        if x_col is not None and x_col in df_compare.columns:
            x_compare = df_compare[x_col]
        else:
            x_compare = df_compare.index

    for idx, col_name in enumerate(columns_to_plot):
        row = (idx // no_cols) + 1
        col = (idx % no_cols) + 1

        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col_name],
                mode="lines",
                name=base_name,
                showlegend=(idx == 0 and df_compare is not None),
                line={"color": line_color},
            ),
            row=row,
            col=col,
        )
        if df_compare is not None:
            fig.add_trace(
                go.Scatter(
                    x=x_compare,
                    y=df_compare[col_name],
                    mode="lines",
                    name=compare_name,
                    showlegend=(idx == 0),
                    line={"color": compare_line_color},
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        height=base_height * no_rows,
        width=base_width * no_cols,
        title_text=country_code,
        template="plotly_white",
    )
    fig.show()


def plot_mc(
    mc,
    cols,
    no_rows=None,
    no_cols=None,
    country_code=None,
    line_opacity=0.5,
    line_width=1.0,
    base_height=220,
    base_width=320,
):
    """Plot Monte Carlo trajectories with one subplot per requested series.

    Parameters
    ----------
    mc:
        A ``MonteCarloResult`` or a dataframe indexed by ``seed`` and simulation time.
    cols:
        Columns to plot. Each subplot overlays one line per seed.
    no_rows, no_cols:
        Optional subplot grid dimensions. If omitted, a near-square grid is used.
    country_code:
        Optional title override.
    """
    combined = mc.combined if hasattr(mc, "combined") else mc
    if not isinstance(combined, pd.DataFrame):
        raise TypeError("mc must be a MonteCarloResult or a pandas DataFrame.")
    if combined.index.nlevels < 2:
        raise ValueError("mc dataframe must be indexed by seed and simulation time.")

    available_cols = [col for col in cols if col in combined.columns]
    if not available_cols:
        raise ValueError("None of the requested cols are present in the Monte Carlo output.")

    n_plots = len(available_cols)
    if no_cols is None and no_rows is None:
        no_cols = int(np.ceil(np.sqrt(n_plots)))
        no_rows = int(np.ceil(n_plots / no_cols))
    elif no_cols is None:
        no_cols = int(np.ceil(n_plots / no_rows))
    elif no_rows is None:
        no_rows = int(np.ceil(n_plots / no_cols))

    seeds = combined.index.get_level_values(0).unique()
    fig = make_subplots(rows=no_rows, cols=no_cols, subplot_titles=available_cols)

    for idx, col_name in enumerate(available_cols):
        row = (idx // no_cols) + 1
        col = (idx % no_cols) + 1

        for seed_idx, seed in enumerate(seeds):
            seed_df = combined.xs(seed, level=0)
            fig.add_trace(
                go.Scatter(
                    x=seed_df.index,
                    y=seed_df[col_name],
                    mode="lines",
                    name=f"seed {seed}",
                    showlegend=(idx == 0),
                    opacity=line_opacity,
                    line={"width": line_width},
                    legendgroup=f"seed-{seed_idx}",
                ),
                row=row,
                col=col,
            )

    title = country_code if country_code is not None else "Monte Carlo trajectories"
    fig.update_layout(
        height=base_height * no_rows,
        width=base_width * no_cols,
        title_text=title,
        template="plotly_white",
    )
    fig.show()


def plot_sensitivity(
    sensitivity,
    cols,
    no_rows=None,
    no_cols=None,
    country_code=None,
    aggregate_seeds=True,
    line_width=2.0,
    line_opacity=1.0,
    base_height=220,
    base_width=320,
):
    """Plot sensitivity trajectories with one subplot per requested series.

    Parameters
    ----------
    sensitivity:
        A ``SensitivityResult`` or a dataframe indexed by parameter, value,
        seed, and simulation time.
    cols:
        Columns to plot. Each subplot overlays one line per parameter value.
    aggregate_seeds:
        If True, average over seeds before plotting. If False, plot every seed
        path with the same color family grouped by parameter value.
    """
    combined = sensitivity.combined if hasattr(sensitivity, "combined") else sensitivity
    if not isinstance(combined, pd.DataFrame):
        raise TypeError("sensitivity must be a SensitivityResult or a pandas DataFrame.")
    if combined.index.nlevels < 4:
        raise ValueError("Sensitivity dataframe must be indexed by parameter, value, seed, and time.")

    available_cols = [col for col in cols if col in combined.columns]
    if not available_cols:
        raise ValueError("None of the requested cols are present in the sensitivity output.")

    n_plots = len(available_cols)
    if no_cols is None and no_rows is None:
        no_cols = int(np.ceil(np.sqrt(n_plots)))
        no_rows = int(np.ceil(n_plots / no_cols))
    elif no_cols is None:
        no_cols = int(np.ceil(n_plots / no_rows))
    elif no_rows is None:
        no_rows = int(np.ceil(n_plots / no_cols))

    parameter_values = combined.index.get_level_values("value").unique()
    colors = [f"hsl({int(h)}, 60%, 45%)" for h in np.linspace(0, 330, max(len(parameter_values), 1))]
    color_by_value = {value: color for value, color in zip(parameter_values, colors)}
    fig = make_subplots(rows=no_rows, cols=no_cols, subplot_titles=available_cols)

    for idx, col_name in enumerate(available_cols):
        row = (idx // no_cols) + 1
        col = (idx % no_cols) + 1

        for value in parameter_values:
            value_df = combined.xs(value, level="value")
            color = color_by_value[value]
            if aggregate_seeds:
                series = value_df.groupby(level="time")[col_name].mean()
                x_vals = series.index
                y_vals = series.values
                name = f"{value}"
                opacity = line_opacity
            else:
                value_seed_df = value_df[col_name]
                for seed_idx, (seed, seed_series) in enumerate(value_seed_df.groupby(level="seed")):
                    time_series = seed_series.droplevel("seed")
                    fig.add_trace(
                        go.Scatter(
                            x=time_series.index,
                            y=time_series.values,
                            mode="lines",
                            name=f"{value} | seed {seed}",
                            showlegend=(idx == 0 and seed_idx == 0),
                            opacity=0.35,
                            line={"width": 1.0, "color": color},
                            legendgroup=f"value-{value}",
                        ),
                        row=row,
                        col=col,
                    )
                continue

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode="lines",
                    name=name,
                    showlegend=(idx == 0),
                    opacity=opacity,
                    line={"width": line_width, "color": color},
                    legendgroup=f"value-{value}",
                ),
                row=row,
                col=col,
            )

    title = country_code if country_code is not None else "Sensitivity trajectories"
    fig.update_layout(
        height=base_height * no_rows,
        width=base_width * no_cols,
        title_text=title,
        template="plotly_white",
    )
    fig.show()


def plot_sensitivity_summary(
    sensitivity,
    cols,
    no_rows=None,
    no_cols=None,
    country_code=None,
    band=(0.1, 0.9),
    line_width=2.0,
    band_opacity=0.18,
    base_height=220,
    base_width=320,
):
    """Plot mean trajectories plus percentile bands across seeds by parameter value."""
    combined = sensitivity.combined if hasattr(sensitivity, "combined") else sensitivity
    if not isinstance(combined, pd.DataFrame):
        raise TypeError("sensitivity must be a SensitivityResult or a pandas DataFrame.")
    if combined.index.nlevels < 4:
        raise ValueError("Sensitivity dataframe must be indexed by parameter, value, seed, and time.")
    if len(band) != 2 or not 0 <= band[0] <= band[1] <= 1:
        raise ValueError("band must be a tuple like (0.1, 0.9).")

    available_cols = [col for col in cols if col in combined.columns]
    if not available_cols:
        raise ValueError("None of the requested cols are present in the sensitivity output.")

    n_plots = len(available_cols)
    if no_cols is None and no_rows is None:
        no_cols = int(np.ceil(np.sqrt(n_plots)))
        no_rows = int(np.ceil(n_plots / no_cols))
    elif no_cols is None:
        no_cols = int(np.ceil(n_plots / no_rows))
    elif no_rows is None:
        no_rows = int(np.ceil(n_plots / no_cols))

    parameter_values = combined.index.get_level_values("value").unique()
    colors = [f"hsl({int(h)}, 60%, 45%)" for h in np.linspace(0, 330, max(len(parameter_values), 1))]
    fig = make_subplots(rows=no_rows, cols=no_cols, subplot_titles=available_cols)

    for idx, col_name in enumerate(available_cols):
        row = (idx // no_cols) + 1
        col = (idx % no_cols) + 1

        for color, value in zip(colors, parameter_values):
            value_df = combined.xs(value, level="value")
            grouped = value_df.groupby(level="time")[col_name]
            mean_ts = grouped.mean()
            lower_ts = grouped.quantile(band[0])
            upper_ts = grouped.quantile(band[1])

            fig.add_trace(
                go.Scatter(
                    x=lower_ts.index,
                    y=lower_ts.values,
                    mode="lines",
                    line={"width": 0, "color": color},
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=f"value-{value}",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=upper_ts.index,
                    y=upper_ts.values,
                    mode="lines",
                    line={"width": 0, "color": color},
                    fill="tonexty",
                    fillcolor=color.replace("45%)", f"45%, {band_opacity})").replace("hsl", "hsla"),
                    hoverinfo="skip",
                    name=f"{value} band",
                    showlegend=False,
                    legendgroup=f"value-{value}",
                ),
                row=row,
                col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=mean_ts.index,
                    y=mean_ts.values,
                    mode="lines",
                    name=f"{value}",
                    showlegend=(idx == 0),
                    line={"width": line_width, "color": color},
                    legendgroup=f"value-{value}",
                ),
                row=row,
                col=col,
            )

    title = country_code if country_code is not None else "Sensitivity summary"
    fig.update_layout(
        height=base_height * no_rows,
        width=base_width * no_cols,
        title_text=title,
        template="plotly_white",
    )
    fig.show()


def build_macro_output_df(model, country_code):
    """Build a macro output dataframe with all derived output columns.

    Observation frequency follows the model timestep. Macro interest-rate
    columns are returned as annualized values.

    Args:
        cfg: Notebook/runtime config kept for call-site compatibility.
        model: Simulation instance.
        plot_columns: Unused, kept for call-site compatibility.
        country_code: ISO3 country code.

    Returns:
        pd.DataFrame containing all available macro output columns.
    """
    shallow = model.shallow_df_dict()[country_code].copy()
    gdp_components = model.get_country_gdp_components_df(country_code).copy()
    periods_per_year = 12 / model.timestep.increment

    out = pd.DataFrame(index=shallow.index)

    def frequency_label(month_increment):
        labels = {
            1: "monthly",
            3: "quarterly",
            6: "semiannual",
            12: "annual",
        }
        return labels.get(month_increment, f"every {month_increment} months")

    def timeseries_dict_to_frame(ts_dict):
        target_len = len(out.index)
        if not ts_dict:
            return pd.DataFrame(index=out.index)

        series_dict = {}
        for key, values in ts_dict.items():
            values = list(values)
            if target_len > 1 and len(values) == 1:
                # Skip static metadata fields that are not recorded each timestep.
                continue
            series_dict[key] = pd.Series(values)

        if not series_dict:
            return pd.DataFrame(index=out.index)

        frame = pd.DataFrame(series_dict).reindex(range(target_len))
        frame.index = out.index
        return frame.map(unpack_cell)

    def first_available(*candidates):
        for candidate in candidates:
            if candidate in out.columns:
                return out[candidate]
            if candidate in shallow.columns:
                return shallow[candidate]
            if candidate in gdp_components.columns:
                return gdp_components[candidate]
        return None

    def assign(name, *candidates):
        if name in out.columns:
            return out[name]

        series = first_available(*candidates)
        if series is not None:
            out[name] = series
        return series

    def assign_annualized(name, *candidates):
        if name in out.columns:
            return out[name]

        series = first_available(*candidates)
        if series is not None:
            out[name] = periods_per_year * series
        return series

    # Start with direct aliases from model output.
    direct_columns = {
        "gdp": ("GDP_Expenditure", "GDP_Output", "GDP_Income"),
        "household consumption": ("Household Consumption", "+Household_Consumption"),
        "government consumption": ("Government Consumption", "+Government_Consumption"),
        "exports": ("Exports", "+Exports"),
        "imports": ("Imports", "-Imports"),
        "gfcf": ("+Gross_Fixed_Capital_Formation", "Capital Bought", "GFCF"),
        "cpi": ("CPI",),
        "cpi yoy inflation": ("CPI YoY Inflation", "cpi_yoy_inflation"),
        "ppi": ("PPI",),
        "output gap": ("Output Gap", "output_gap"),
        "unemployment rate": ("Unemployment Rate",),
        "central bank policy rate": ("Central Bank Policy Rate",),
        "short-term firm borrowing rate": (
            "Average Interest Rates on Short Term Firm Loans",
            "average_interest_rates_on_short_term_firm_loans",
        ),
        "long-term firm borrowing rate": (
            "Average Interest Rates on Long Term Firm Loans",
            "average_interest_rates_on_long_term_firm_loans",
        ),
        "household consumption borrowing rate": (
            "Average Interest Rates on Household Consumption Loans",
            "average_interest_rates_on_household_consumption_loans",
        ),
        "mortgage borrowing rate": (
            "Average Interest Rates on Mortgages",
            "average_interest_rates_on_mortgages",
        ),
        "consumption expansion loan debt": ("Consumption Expansion Loan Debt",),
        "mortgage debt": ("Mortgage Debt",),
        "wages": ("Wages", "+Wages"),
        "profits": ("Profits", "+Operating_Surplus"),
        "taxes paid on production": ("Taxes Paid on Production", "-Taxes_on_Production"),
        "taxes on products": ("Taxes on Products", "+Taxes_on_Products", "+Central_Government_Product_Taxes"),
    }

    def build_column(name):
        if name in out.columns:
            return out[name]
        if name == "central bank policy rate":
            policy_rate = assign_annualized(name, *direct_columns[name])
            if policy_rate is not None:
                return out[name]
            return None
        if name in {
            "short-term firm borrowing rate",
            "long-term firm borrowing rate",
            "household consumption borrowing rate",
            "mortgage borrowing rate",
        }:
            borrowing_rate = assign_annualized(name, *direct_columns[name])
            if borrowing_rate is not None:
                return out[name]
            return None
        if name in direct_columns:
            return assign(name, *direct_columns[name])
        if name == "unemployment benefits":
            if "total_unemployment_benefits" in out.columns:
                out[name] = out["total_unemployment_benefits"]
                return out[name]
            return None
        if name == "other benefits":
            if "total_household_social_transfers" in out.columns:
                out[name] = out["total_household_social_transfers"]
                return out[name]
            return None
        if name == "total consumption":
            hh_cons = build_column("household consumption")
            gov_cons = build_column("government consumption")
            if hh_cons is not None and gov_cons is not None:
                out[name] = hh_cons + gov_cons
                return out[name]
            return None
        if name == "gdp growth":
            gdp = build_column("gdp")
            if gdp is not None:
                out[name] = gdp.pct_change()
                return out[name]
            return None
        if name == "real gdp":
            gdp = build_column("gdp")
            ppi = build_column("ppi")
            if gdp is not None and ppi is not None:
                out[name] = gdp / ppi
                return out[name]
            return None
        if name == "expected gdp growth":
            if "estimated_growth" in out.columns:
                out[name] = out["estimated_growth"]
                return out[name]
            return None
        if name == "household consumption to gdp":
            hh_cons = build_column("household consumption")
            gdp = build_column("gdp")
            if hh_cons is not None and gdp is not None:
                out[name] = hh_cons / gdp
                return out[name]
            return None
        if name == "government consumption to gdp":
            gov_cons = build_column("government consumption")
            gdp = build_column("gdp")
            if gov_cons is not None and gdp is not None:
                out[name] = gov_cons / gdp
                return out[name]
            return None
        if name == "total consumption to gdp":
            total_cons = build_column("total consumption")
            gdp = build_column("gdp")
            if total_cons is not None and gdp is not None:
                out[name] = total_cons / gdp
                return out[name]
            return None
        if name == "investment to gdp":
            gfcf = build_column("gfcf")
            gdp = build_column("gdp")
            if gfcf is not None and gdp is not None:
                out[name] = gfcf / gdp
                return out[name]
            return None
        if name == "net exports to gdp":
            exports = build_column("exports")
            imports = build_column("imports")
            gdp = build_column("gdp")
            if exports is not None and imports is not None and gdp is not None:
                out[name] = (exports - imports) / gdp
                return out[name]
            return None
        if name in {"fiscal revenue", "revenues", "revenue"}:
            if "revenue" in df_gov_ts.columns:
                out[name] = df_gov_ts["revenue"]
                return out[name]
            return None
        if name in {"fiscal expenditure", "government expenditure", "spending"}:
            if "government expenditure" in df_gov_ts.columns:
                out[name] = df_gov_ts["government expenditure"]
                return out[name]
            return None
        if name == "deficit":
            if "deficit" in df_gov_ts.columns:
                out[name] = df_gov_ts["deficit"]
                return out[name]
            return None
        if name == "debt":
            if "debt" in df_gov_ts.columns:
                out[name] = df_gov_ts["debt"]
                return out[name]
            return None
        if name == "interest payments on debt":
            if "interest_payments_on_debt" in df_gov_ts.columns:
                out[name] = df_gov_ts["interest_payments_on_debt"]
                return out[name]
            return None
        if name in {"fiscal revenue to gdp", "revenues to gdp"}:
            revenues = build_column("fiscal revenue")
            gdp = build_column("gdp")
            if revenues is not None and gdp is not None:
                out[name] = revenues / gdp
                return out[name]
            return None
        if name in {"fiscal expenditure to gdp", "government expenditure to gdp", "spending to gdp"}:
            spending = build_column("fiscal expenditure")
            gdp = build_column("gdp")
            if spending is not None and gdp is not None:
                out[name] = spending / gdp
                return out[name]
            return None
        if name == "unemp_benefits_to_expenditure":
            unemployment_benefits = build_column("unemployment benefits")
            spending = build_column("fiscal expenditure")
            if unemployment_benefits is not None and spending is not None:
                out[name] = unemployment_benefits / spending
                return out[name]
            return None
        if name == "other_benefits_to_expenditure":
            other_benefits = build_column("other benefits")
            spending = build_column("fiscal expenditure")
            if other_benefits is not None and spending is not None:
                out[name] = other_benefits / spending
                return out[name]
            return None
        if name == "gov_consumption_to_expenditure":
            government_consumption = build_column("government consumption")
            spending = build_column("fiscal expenditure")
            if government_consumption is not None and spending is not None:
                out[name] = government_consumption / spending
                return out[name]
            return None
        if name == "interest_payments_on_debt_to_expenditure":
            interest_payments = build_column("interest payments on debt")
            spending = build_column("fiscal expenditure")
            if interest_payments is not None and spending is not None:
                out[name] = interest_payments / spending
                return out[name]
            return None
        if name == "deficit to gdp":
            deficit = build_column("deficit")
            gdp = build_column("gdp")
            if deficit is not None and gdp is not None:
                out[name] = deficit / gdp
                return out[name]
            return None
        if name == "debt to gdp":
            debt = build_column("debt")
            gdp = build_column("gdp")
            if debt is not None and gdp is not None:
                out[name] = debt / (periods_per_year * gdp)
                return out[name]
            return None
        return assign(name, name)

    # Analyse government revenues and spending
    gov_ts_dict = model.countries[country_code].central_government.ts.__dict__["dicts"]
    cb_ts_dict = model.countries[country_code].central_bank.ts.__dict__["dicts"]
    df_cb_ts = timeseries_dict_to_frame(cb_ts_dict)
    bank_ts_dict = model.countries[country_code].banks.ts.__dict__["dicts"]
    df_bank_ts = timeseries_dict_to_frame(bank_ts_dict)
    df_gov_ts = timeseries_dict_to_frame(gov_ts_dict)
    df_gov_ts["government expenditure"] = df_gov_ts["revenue"] + df_gov_ts["deficit"]
    for column in df_gov_ts.columns:
        out[column.lower()] = df_gov_ts[column]
    for column in df_cb_ts.columns:
        if column.lower() not in out.columns:
            out[column.lower()] = df_cb_ts[column]
    for bank_column in [
        "average_interest_rates_on_short_term_firm_loans",
        "average_interest_rates_on_long_term_firm_loans",
        "average_interest_rates_on_household_consumption_loans",
        "average_interest_rates_on_mortgages",
    ]:
        bank_series = df_bank_ts.get(bank_column)
        if bank_series is not None and bank_column not in out.columns:
            out[bank_column] = bank_series
    economy_ts_dict = model.countries[country_code].economy.ts.__dict__["dicts"]
    for economy_column in ["estimated_growth", "cpi_yoy_inflation", "output_gap"]:
        economy_series = economy_ts_dict.get(economy_column)
        if economy_series is not None and economy_column not in out.columns:
            out[economy_column] = pd.Series([unpack_cell(x) for x in economy_series], index=out.index)

    all_columns = [
        "gdp",
        "real gdp",
        "gdp growth",
        "expected gdp growth",
        "household consumption",
        "government consumption",
        "total consumption",
        "household consumption to gdp",
        "government consumption to gdp",
        "total consumption to gdp",
        "exports",
        "imports",
        "gfcf",
        "investment to gdp",
        "net exports to gdp",
        "cpi",
        "cpi yoy inflation",
        "ppi",
        "output gap",
        "unemployment rate",
        "central bank policy rate",
        "short-term firm borrowing rate",
        "long-term firm borrowing rate",
        "household consumption borrowing rate",
        "mortgage borrowing rate",
        "consumption expansion loan debt",
        "mortgage debt",
        "unemployment benefits",
        "other benefits",
        "wages",
        "profits",
        "taxes paid on production",
        "taxes on products",
        "fiscal revenue",
        "fiscal expenditure",
        "unemp_benefits_to_expenditure",
        "other_benefits_to_expenditure",
        "gov_consumption_to_expenditure",
        "interest_payments_on_debt_to_expenditure",
        "deficit",
        "debt",
        "interest payments on debt",
        "fiscal revenue to gdp",
        "fiscal expenditure to gdp",
        "deficit to gdp",
        "debt to gdp",
    ]
    for column in all_columns:
        build_column(column)

    out = out.drop(
        columns=[
            "real unemployment benefits",
            "nominal unemployment benefits",
            "real other benefits",
            "total_other_benefits",
            "unemployment_benefits_by_individual",
        ],
        errors="ignore",
    )

    out.attrs["time_unit_months"] = model.timestep.increment
    out.attrs["observation_frequency"] = frequency_label(model.timestep.increment)
    out.attrs["periods_per_year"] = periods_per_year
    out.attrs["interest_rate_observation_frequency"] = out.attrs["observation_frequency"]
    out.attrs["interest_rate_units"] = "Macro interest-rate columns are annualized."

    return out
