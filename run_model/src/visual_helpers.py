import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.helpers import unpack_cell


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


def build_macro_output_df(model, country_code):
    """Build a macro output dataframe with all derived output columns.

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

    def first_available(*candidates):
        for candidate in candidates:
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

    # Start with direct aliases from model output.
    direct_columns = {
        "gdp": ("GDP_Expenditure", "GDP_Output", "GDP_Income"),
        "household consumption": ("Household Consumption", "+Household_Consumption"),
        "government consumption": ("Government Consumption", "+Government_Consumption"),
        "exports": ("Exports", "+Exports"),
        "imports": ("Imports", "-Imports"),
        "gfcf": ("+Gross_Fixed_Capital_Formation", "Capital Bought", "GFCF"),
        "cpi": ("CPI",),
        "ppi": ("PPI",),
        "unemployment rate": ("Unemployment Rate",),
        "central bank policy rate": ("Central Bank Policy Rate",),
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
        if name in direct_columns:
            return assign(name, *direct_columns[name])
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
    df_cb_ts = pd.DataFrame({k: [x for x in v] for k, v in cb_ts_dict.items()})
    df_cb_ts = df_cb_ts.map(unpack_cell)
    df_gov_ts = pd.DataFrame({k: [x for x in v] for k, v in gov_ts_dict.items()})
    df_gov_ts = df_gov_ts.map(unpack_cell)
    df_gov_ts["government expenditure"] = df_gov_ts["revenue"] - df_gov_ts["deficit"]
    df_gov_ts["interest on debt"] = df_gov_ts["debt"] * df_cb_ts["policy_rate"]
    for column in df_gov_ts.columns:
        out[column.lower()] = df_gov_ts[column]
    for column in df_cb_ts.columns:
        if column.lower() not in out.columns:
            out[column.lower()] = df_cb_ts[column]

    all_columns = [
        "gdp",
        "gdp growth",
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
        "ppi",
        "unemployment rate",
        "central bank policy rate",
        "consumption expansion loan debt",
        "mortgage debt",
        "wages",
        "profits",
        "taxes paid on production",
        "taxes on products",
        "fiscal revenue",
        "fiscal expenditure",
        "deficit",
        "debt",
        "fiscal revenue to gdp",
        "fiscal expenditure to gdp",
        "deficit to gdp",
        "debt to gdp",
    ]
    for column in all_columns:
        build_column(column)

    return out
