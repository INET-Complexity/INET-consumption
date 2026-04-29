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


def _read_h5_1d(h5_file, path):
    values = np.asarray(h5_file[path], dtype=float)
    return values.reshape(values.shape[0], -1)[:, 0]


def _timeseries_1d(values):
    return np.asarray([unpack_cell(value) for value in values], dtype=float)


def build_ppi_comparison_df(model=None, country_code=None, h5_path=None, time_unit=None):
    """Build a dataframe comparing model, fixed Laspeyres, and chained Laspeyres PPI.

    Supply either a live ``model`` plus ``country_code`` or an ``h5_path`` plus
    ``country_code``. The model PPI YoY comparison is computed from the PPI level
    using ``12 // time_unit`` periods.
    """
    if country_code is None:
        raise ValueError("country_code is required.")
    if (model is None) == (h5_path is None):
        raise ValueError("Provide exactly one of model or h5_path.")

    if model is not None:
        economy_ts = model.countries[country_code].economy.ts.__dict__["dicts"]
        if time_unit is None:
            time_unit = model.timestep.increment
        data = {
            "model_ppi": _timeseries_1d(economy_ts["ppi"]),
            "fixed_ppi": _timeseries_1d(economy_ts["ppi_fixed"]),
            "chained_ppi": _timeseries_1d(economy_ts["ppi_chained"]),
            "model_pop": _timeseries_1d(economy_ts["ppi_inflation"]),
            "fixed_pop": _timeseries_1d(economy_ts["ppi_fixed_pop_change"]),
            "chained_pop": _timeseries_1d(economy_ts["ppi_chained_pop_change"]),
            "fixed_yoy": _timeseries_1d(economy_ts["ppi_fixed_yoy_change"]),
            "chained_yoy": _timeseries_1d(economy_ts["ppi_chained_yoy_change"]),
        }
    else:
        import h5py

        base = f"{country_code}/economy"
        with h5py.File(h5_path, "r") as h5_file:
            data = {
                "model_ppi": _read_h5_1d(h5_file, f"{base}/ppi"),
                "fixed_ppi": _read_h5_1d(h5_file, f"{base}/ppi_fixed"),
                "chained_ppi": _read_h5_1d(h5_file, f"{base}/ppi_chained"),
                "model_pop": _read_h5_1d(h5_file, f"{base}/ppi_inflation"),
                "fixed_pop": _read_h5_1d(h5_file, f"{base}/ppi_fixed_pop_change"),
                "chained_pop": _read_h5_1d(h5_file, f"{base}/ppi_chained_pop_change"),
                "fixed_yoy": _read_h5_1d(h5_file, f"{base}/ppi_fixed_yoy_change"),
                "chained_yoy": _read_h5_1d(h5_file, f"{base}/ppi_chained_yoy_change"),
            }
        if time_unit is None:
            time_unit = 3

    if time_unit <= 0 or 12 % time_unit != 0:
        raise ValueError("time_unit must be a positive divisor of 12.")

    target_len = min(len(values) for values in data.values())
    out = pd.DataFrame({key: values[:target_len] for key, values in data.items()})
    out.index.name = "t"

    periods_per_year = 12 // time_unit
    out["model_yoy"] = out["model_ppi"] / out["model_ppi"].shift(periods_per_year) - 1.0
    out["fixed_minus_model"] = out["fixed_ppi"] - out["model_ppi"]
    out["chained_minus_model"] = out["chained_ppi"] - out["model_ppi"]
    out["chained_minus_fixed"] = out["chained_ppi"] - out["fixed_ppi"]
    out["fixed_pop_minus_model_pop"] = out["fixed_pop"] - out["model_pop"]
    out["chained_pop_minus_model_pop"] = out["chained_pop"] - out["model_pop"]

    out.attrs["time_unit_months"] = time_unit
    out.attrs["periods_per_year"] = periods_per_year
    return out


def build_macro_output_df(model, country_code):
    """Build a macro output dataframe with explicit canonical output columns.

    Observation frequency follows the model timestep. Macro interest-rate
    columns are returned as annualized values.
    """
    shallow = model.shallow_df_dict()[country_code].copy()
    gdp_components = model.get_country_gdp_components_df(country_code).copy()
    periods_per_year = 12 / model.timestep.increment
    yoy_periods = int(12 // model.timestep.increment)

    out_index = shallow.index
    output_columns = {}

    def frequency_label(month_increment):
        labels = {
            1: "monthly",
            3: "quarterly",
            6: "semiannual",
            12: "annual",
        }
        return labels.get(month_increment, f"every {month_increment} months")

    def as_output_series(values):
        series = pd.Series(list(values)).reindex(range(len(out_index)))
        series.index = out_index
        return series.map(unpack_cell)

    def timeseries_dict_to_frame(ts_dict):
        if not ts_dict:
            return pd.DataFrame(index=out_index)

        series_dict = {}
        for key, values in ts_dict.items():
            values = list(values)
            if len(out_index) > 1 and len(values) == 1:
                continue
            series_dict[key] = as_output_series(values)

        if not series_dict:
            return pd.DataFrame(index=out_index)

        return pd.DataFrame(series_dict, index=out_index)

    def first_available(*candidates):
        for candidate in candidates:
            if candidate in output_columns:
                return output_columns[candidate]
            if candidate in shallow.columns:
                return shallow[candidate]
            if candidate in gdp_components.columns:
                return gdp_components[candidate]
        return None

    def add_column(name, series):
        if series is None:
            return None
        if name not in output_columns:
            output_columns[name] = pd.Series(series, index=out_index)
        return output_columns[name]

    def get_column(name):
        return output_columns.get(name)

    def add_source_column(name, *candidates):
        return add_column(name, first_available(*candidates))

    def add_annualized_source_column(name, *candidates):
        series = first_available(*candidates)
        if series is not None:
            return add_column(name, periods_per_year * series)
        return None

    def add_ratio(name, numerator, denominator):
        if numerator is not None and denominator is not None:
            return add_column(name, numerator / denominator)
        return None

    country = model.countries[country_code]
    df_gov_ts = timeseries_dict_to_frame(country.central_government.ts.__dict__["dicts"])
    df_cb_ts = timeseries_dict_to_frame(country.central_bank.ts.__dict__["dicts"])
    df_bank_ts = timeseries_dict_to_frame(country.banks.ts.__dict__["dicts"])
    economy_ts_dict = country.economy.ts.__dict__["dicts"]

    source_columns = {
        "gdp": ("GDP_Expenditure", "GDP_Output", "GDP_Income"),
        "household_consumption": ("Household Consumption", "+Household_Consumption"),
        "government_consumption": ("Government Consumption", "+Government_Consumption"),
        "exports": ("Exports", "+Exports"),
        "imports": ("Imports", "-Imports"),
        "gfcf": ("+Gross_Fixed_Capital_Formation", "Capital Bought", "GFCF"),
        "cpi_transaction": ("CPI Transaction", "CPI"),
        "ppi": ("PPI",),
        "consumption_expansion_loan_debt": ("Consumption Expansion Loan Debt",),
        "mortgage_debt": ("Mortgage Debt",),
        "wages": ("Wages", "+Wages"),
        "profits": ("Profits", "+Operating_Surplus"),
        "taxes_paid_on_production": ("Taxes Paid on Production", "-Taxes_on_Production"),
        "taxes_on_products": ("Taxes on Products", "+Taxes_on_Products", "+Central_Government_Product_Taxes"),
    }
    for name, candidates in source_columns.items():
        add_source_column(name, *candidates)

    gdp = get_column("gdp")
    household_consumption = get_column("household_consumption")
    government_consumption = get_column("government_consumption")
    total_consumption = None
    if household_consumption is not None and government_consumption is not None:
        total_consumption = add_column("total_consumption", household_consumption + government_consumption)

    ppi = get_column("ppi")
    if gdp is not None:
        add_column("gdp_growth", gdp.pct_change())
    if gdp is not None and ppi is not None:
        add_column("real_gdp", gdp / ppi)
    if ppi is not None:
        add_column("ppi_yoy_change", ppi / ppi.shift(yoy_periods) - 1.0)

    add_ratio("household_consumption_to_gdp", household_consumption, gdp)
    add_ratio("government_consumption_to_gdp", government_consumption, gdp)
    add_ratio("total_consumption_to_gdp", total_consumption, gdp)
    add_ratio("investment_to_gdp", get_column("gfcf"), gdp)
    exports = get_column("exports")
    imports = get_column("imports")
    if exports is not None and imports is not None and gdp is not None:
        add_column("net_exports_to_gdp", (exports - imports) / gdp)

    if "revenue" in df_gov_ts.columns:
        add_column("fiscal_revenue", df_gov_ts["revenue"])
    if "revenue" in df_gov_ts.columns and "deficit" in df_gov_ts.columns:
        add_column("fiscal_expenditure", df_gov_ts["revenue"] + df_gov_ts["deficit"])
    for source_name, output_name in {
        "deficit": "deficit",
        "debt": "debt",
        "interest_payments_on_debt": "interest_payments_on_debt",
        "total_unemployment_benefits": "unemployment_benefits",
        "total_household_social_transfers": "other_benefits",
    }.items():
        if source_name in df_gov_ts.columns:
            add_column(output_name, df_gov_ts[source_name])

    fiscal_expenditure = get_column("fiscal_expenditure")
    add_ratio("fiscal_revenue_to_gdp", get_column("fiscal_revenue"), gdp)
    add_ratio("fiscal_expenditure_to_gdp", fiscal_expenditure, gdp)
    add_ratio("deficit_to_gdp", get_column("deficit"), gdp)
    debt = get_column("debt")
    if debt is not None and gdp is not None:
        add_column("debt_to_gdp", debt / (periods_per_year * gdp))
    add_ratio("unemployment_benefits_to_expenditure", get_column("unemployment_benefits"), fiscal_expenditure)
    add_ratio("other_benefits_to_expenditure", get_column("other_benefits"), fiscal_expenditure)
    add_ratio("government_consumption_to_expenditure", government_consumption, fiscal_expenditure)
    add_ratio("interest_payments_on_debt_to_expenditure", get_column("interest_payments_on_debt"), fiscal_expenditure)

    if "policy_rate" in df_cb_ts.columns:
        add_column("central_bank_policy_rate", periods_per_year * df_cb_ts["policy_rate"])
    else:
        add_annualized_source_column("central_bank_policy_rate", "Central Bank Policy Rate")

    for source_name, output_name, label_name in [
        (
            "average_interest_rates_on_short_term_firm_loans",
            "short_term_firm_borrowing_rate",
            "Average Interest Rates on Short Term Firm Loans",
        ),
        (
            "average_interest_rates_on_long_term_firm_loans",
            "long_term_firm_borrowing_rate",
            "Average Interest Rates on Long Term Firm Loans",
        ),
        (
            "average_interest_rates_on_household_consumption_loans",
            "household_consumption_borrowing_rate",
            "Average Interest Rates on Household Consumption Loans",
        ),
        (
            "average_interest_rates_on_mortgages",
            "mortgage_borrowing_rate",
            "Average Interest Rates on Mortgages",
        ),
    ]:
        if source_name in df_bank_ts.columns:
            add_column(output_name, periods_per_year * df_bank_ts[source_name])
        else:
            add_annualized_source_column(output_name, label_name, source_name)

    def add_economy_column(name):
        values = economy_ts_dict.get(name)
        if values is None:
            return None
        return add_column(name, as_output_series(values))

    def sector_labels(width):
        industries = list(getattr(country.firms, "industries", []))
        if len(industries) == width:
            return [str(industry) for industry in industries]
        return [str(i) for i in range(width)]

    def add_economy_vector_columns(name):
        series = add_economy_column(name)
        if series is None:
            return

        non_null = series.dropna()
        if non_null.empty or not isinstance(non_null.iloc[0], list):
            return

        labels = sector_labels(len(non_null.iloc[0]))
        expanded = pd.DataFrame(series.tolist(), index=out_index, columns=[f"{name}_{label}" for label in labels])
        for column in expanded.columns:
            output_columns[column] = expanded[column]

    for economy_column in [
        "unemployment_rate",
        "unemployment_rate_growth",
        "participation_rate",
        "participation_rate_growth",
        "vacancy_rate",
        "vacancy_rate_growth",
        "job_reallocation_rate",
        "job_reallocation_rate_growth",
        "firm_insolvency_rate",
        "bank_insolvency_rate",
        "household_insolvency_rate",
        "total_growth",
        "estimated_growth",
        "cpi_transaction",
        "cpi_transaction_pop_change",
        "cpi_transaction_yoy_change",
        "real_gross_output",
        "potential_output",
        "output_gap",
        "hpi",
        "hpi_inflation",
        "estimated_hpi_inflation",
        "total_real_rent_paid",
        "total_imp_rent_paid",
        "total_real_rent_rec",
        "npl_firm_loans",
        "npl_hh_cons_loans",
        "cpi_fixed_basket",
        "cpi_fixed_basket_pop_change",
        "cpi_fixed_basket_yoy_change",
        "cpi_chained_basket",
        "cpi_chained_basket_pop_change",
        "cpi_chained_basket_yoy_change",
        "ppi_fixed",
        "ppi_fixed_pop_change",
        "ppi_fixed_yoy_change",
        "ppi_chained",
        "ppi_chained_pop_change",
        "ppi_chained_yoy_change",
    ]:
        add_economy_column(economy_column)
    for economy_column in ["sectoral_growth", "num_insolvent_firms_by_sector"]:
        add_economy_vector_columns(economy_column)

    out = pd.DataFrame(output_columns, index=out_index).copy()
    out.attrs["time_unit_months"] = model.timestep.increment
    out.attrs["observation_frequency"] = frequency_label(model.timestep.increment)
    out.attrs["periods_per_year"] = periods_per_year
    out.attrs["interest_rate_observation_frequency"] = out.attrs["observation_frequency"]
    out.attrs["interest_rate_units"] = "Macro interest-rate columns are annualized."

    return out


def summarize_ppi_comparison(ppi_comparison_df):
    """Return descriptive statistics for PPI level and change comparisons."""
    columns = [
        "model_ppi",
        "fixed_ppi",
        "chained_ppi",
        "model_pop",
        "fixed_pop",
        "chained_pop",
        "model_yoy",
        "fixed_yoy",
        "chained_yoy",
        "fixed_minus_model",
        "chained_minus_model",
        "chained_minus_fixed",
    ]
    return ppi_comparison_df[columns].describe().T


def build_cpi_comparison_df(model=None, country_code=None, h5_path=None, time_unit=None):
    """Build a dataframe comparing model, fixed Laspeyres, and chained Laspeyres CPI.

    Supply either a live ``model`` plus ``country_code`` or an ``h5_path`` plus
    ``country_code``. The model CPI YoY comparison uses the recorded
    ``cpi_transaction_yoy_change`` series.
    """
    if country_code is None:
        raise ValueError("country_code is required.")
    if (model is None) == (h5_path is None):
        raise ValueError("Provide exactly one of model or h5_path.")

    if model is not None:
        economy_ts = model.countries[country_code].economy.ts.__dict__["dicts"]
        if time_unit is None:
            time_unit = model.timestep.increment
        data = {
            "cpi_transaction": _timeseries_1d(economy_ts["cpi_transaction"]),
            "cpi_fixed_basket": _timeseries_1d(economy_ts["cpi_fixed_basket"]),
            "cpi_chained_basket": _timeseries_1d(economy_ts["cpi_chained_basket"]),
            "cpi_transaction_pop_change": _timeseries_1d(economy_ts["cpi_transaction_pop_change"]),
            "cpi_fixed_basket_pop_change": _timeseries_1d(economy_ts["cpi_fixed_basket_pop_change"]),
            "cpi_chained_basket_pop_change": _timeseries_1d(economy_ts["cpi_chained_basket_pop_change"]),
            "cpi_transaction_yoy_change": _timeseries_1d(economy_ts["cpi_transaction_yoy_change"]),
            "cpi_fixed_basket_yoy_change": _timeseries_1d(economy_ts["cpi_fixed_basket_yoy_change"]),
            "cpi_chained_basket_yoy_change": _timeseries_1d(economy_ts["cpi_chained_basket_yoy_change"]),
        }
    else:
        import h5py

        base = f"{country_code}/economy"
        with h5py.File(h5_path, "r") as h5_file:
            data = {
                "cpi_transaction": _read_h5_1d(h5_file, f"{base}/cpi_transaction"),
                "cpi_fixed_basket": _read_h5_1d(h5_file, f"{base}/cpi_fixed_basket"),
                "cpi_chained_basket": _read_h5_1d(h5_file, f"{base}/cpi_chained_basket"),
                "cpi_transaction_pop_change": _read_h5_1d(h5_file, f"{base}/cpi_transaction_pop_change"),
                "cpi_fixed_basket_pop_change": _read_h5_1d(h5_file, f"{base}/cpi_fixed_basket_pop_change"),
                "cpi_chained_basket_pop_change": _read_h5_1d(h5_file, f"{base}/cpi_chained_basket_pop_change"),
                "cpi_transaction_yoy_change": _read_h5_1d(h5_file, f"{base}/cpi_transaction_yoy_change"),
                "cpi_fixed_basket_yoy_change": _read_h5_1d(h5_file, f"{base}/cpi_fixed_basket_yoy_change"),
                "cpi_chained_basket_yoy_change": _read_h5_1d(h5_file, f"{base}/cpi_chained_basket_yoy_change"),
            }
        if time_unit is None:
            time_unit = 3

    if time_unit <= 0 or 12 % time_unit != 0:
        raise ValueError("time_unit must be a positive divisor of 12.")

    target_len = min(len(values) for values in data.values())
    out = pd.DataFrame({key: values[:target_len] for key, values in data.items()})
    out.index.name = "t"

    periods_per_year = 12 // time_unit
    out["cpi_fixed_basket_minus_transaction"] = out["cpi_fixed_basket"] - out["cpi_transaction"]
    out["cpi_chained_basket_minus_transaction"] = out["cpi_chained_basket"] - out["cpi_transaction"]
    out["cpi_chained_basket_minus_fixed_basket"] = out["cpi_chained_basket"] - out["cpi_fixed_basket"]
    out["cpi_fixed_basket_pop_change_minus_transaction"] = (
        out["cpi_fixed_basket_pop_change"] - out["cpi_transaction_pop_change"]
    )
    out["cpi_chained_basket_pop_change_minus_transaction"] = (
        out["cpi_chained_basket_pop_change"] - out["cpi_transaction_pop_change"]
    )

    out.attrs["time_unit_months"] = time_unit
    out.attrs["periods_per_year"] = periods_per_year
    return out


def summarize_cpi_comparison(cpi_comparison_df):
    """Return descriptive statistics for CPI level and change comparisons."""
    columns = [
        "cpi_transaction",
        "cpi_fixed_basket",
        "cpi_chained_basket",
        "cpi_transaction_pop_change",
        "cpi_fixed_basket_pop_change",
        "cpi_chained_basket_pop_change",
        "cpi_transaction_yoy_change",
        "cpi_fixed_basket_yoy_change",
        "cpi_chained_basket_yoy_change",
        "cpi_fixed_basket_minus_transaction",
        "cpi_chained_basket_minus_transaction",
        "cpi_chained_basket_minus_fixed_basket",
    ]
    return cpi_comparison_df[columns].describe().T


def plot_ppi_comparison(ppi_comparison_df, title="PPI comparison", height=850, width=1000, show=True):
    """Plot model PPI against fixed and chained Laspeyres PPI."""
    colors = {
        "model": "#1f77b4",
        "fixed": "#2ca02c",
        "chained": "#d62728",
    }
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["PPI levels", "Period-on-period changes", "Year-on-year changes"],
        vertical_spacing=0.08,
    )

    trace_groups = [
        ("model_ppi", "Level: model PPI", "model", 1),
        ("fixed_ppi", "Level: fixed-basket PPI", "fixed", 1),
        ("chained_ppi", "Level: chained-basket PPI", "chained", 1),
        ("model_pop", "PoP: model PPI", "model", 2),
        ("fixed_pop", "PoP: fixed-basket PPI", "fixed", 2),
        ("chained_pop", "PoP: chained-basket PPI", "chained", 2),
        ("model_yoy", "YoY: model PPI", "model", 3),
        ("fixed_yoy", "YoY: fixed-basket PPI", "fixed", 3),
        ("chained_yoy", "YoY: chained-basket PPI", "chained", 3),
    ]
    for column, name, color_key, row in trace_groups:
        fig.add_trace(
            go.Scatter(
                x=ppi_comparison_df.index,
                y=ppi_comparison_df[column],
                mode="lines",
                name=name,
                line={"color": colors[color_key], "width": 2},
            ),
            row=row,
            col=1,
        )

    fig.add_hline(y=0.0, line_width=1, line_color="black", row=2, col=1)
    fig.add_hline(y=0.0, line_width=1, line_color="black", row=3, col=1)
    fig.update_yaxes(title_text="index", row=1, col=1)
    fig.update_yaxes(title_text="rate", row=2, col=1)
    fig.update_yaxes(title_text="rate", row=3, col=1)
    fig.update_xaxes(title_text="t", row=3, col=1)
    fig.update_layout(height=height, width=width, title_text=title, template="plotly_white")

    if show:
        fig.show()
        return None
    return fig


def plot_cpi_comparison(cpi_comparison_df, title="CPI comparison", height=850, width=1000, show=True):
    """Plot model CPI against fixed and chained Laspeyres CPI."""
    colors = {
        "model": "#1f77b4",
        "fixed": "#2ca02c",
        "chained": "#d62728",
    }
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["CPI levels", "Period-on-period changes", "Year-on-year changes"],
        vertical_spacing=0.08,
    )

    trace_groups = [
        ("cpi_transaction", "Level: transaction CPI", "model", 1),
        ("cpi_fixed_basket", "Level: fixed-basket CPI", "fixed", 1),
        ("cpi_chained_basket", "Level: chained-basket CPI", "chained", 1),
        ("cpi_transaction_pop_change", "PoP: transaction CPI", "model", 2),
        ("cpi_fixed_basket_pop_change", "PoP: fixed-basket CPI", "fixed", 2),
        ("cpi_chained_basket_pop_change", "PoP: chained-basket CPI", "chained", 2),
        ("cpi_transaction_yoy_change", "YoY: transaction CPI", "model", 3),
        ("cpi_fixed_basket_yoy_change", "YoY: fixed-basket CPI", "fixed", 3),
        ("cpi_chained_basket_yoy_change", "YoY: chained-basket CPI", "chained", 3),
    ]
    for column, name, color_key, row in trace_groups:
        fig.add_trace(
            go.Scatter(
                x=cpi_comparison_df.index,
                y=cpi_comparison_df[column],
                mode="lines",
                name=name,
                line={"color": colors[color_key], "width": 2},
            ),
            row=row,
            col=1,
        )

    fig.add_hline(y=0.0, line_width=1, line_color="black", row=2, col=1)
    fig.add_hline(y=0.0, line_width=1, line_color="black", row=3, col=1)
    fig.update_yaxes(title_text="index", row=1, col=1)
    fig.update_yaxes(title_text="rate", row=2, col=1)
    fig.update_yaxes(title_text="rate", row=3, col=1)
    fig.update_xaxes(title_text="t", row=3, col=1)
    fig.update_layout(height=height, width=width, title_text=title, template="plotly_white")

    if show:
        fig.show()
        return None
    return fig


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
