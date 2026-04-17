import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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


def build_macro_output_df(cfg, model, country_code):
    """Build a macro output dataframe with derived indicators used in notebooks.

    Args:
        cfg: Notebook/runtime config (kept for backwards compatibility).
        model: Simulation instance.
        country_code: ISO3 country code.

    Returns:
        pd.DataFrame with derived columns such as consumption/GDP ratios and GDP growth.
    """
    del cfg

    shallow = model.shallow_df_dict()[country_code].copy()
    gdp_components = model.get_country_gdp_components_df(country_code).copy()

    out = pd.DataFrame(index=shallow.index)

    def first_available(*candidates):
        for candidate in candidates:
            if candidate in shallow.columns:
                return shallow[candidate]
            if candidate in gdp_components.columns:
                return gdp_components[candidate]
        return None

    gdp = first_available("GDP_Expenditure", "GDP_Output", "GDP_Income")
    hh_cons = first_available("Household Consumption")
    gov_cons = first_available("Government Consumption")
    exports = first_available("Exports", "+Exports")
    imports = first_available("Imports", "-Imports")
    gfcf = first_available("+Gross_Fixed_Capital_Formation", "Capital Bought", "GFCF")

    if gdp is not None:
        out["GDP"] = gdp
        out["GDP Growth"] = gdp.pct_change()

    if hh_cons is not None:
        out["Household Consumption"] = hh_cons
    if gov_cons is not None:
        out["Government Consumption"] = gov_cons
    if exports is not None:
        out["Exports"] = exports
    if imports is not None:
        out["Imports"] = imports
    if gfcf is not None:
        out["GFCF"] = gfcf

    if hh_cons is not None and gov_cons is not None:
        out["Total Consumption"] = hh_cons + gov_cons

    if gdp is not None:
        if hh_cons is not None:
            out["Household Consumption to GDP"] = hh_cons / gdp
        if gov_cons is not None:
            out["Government Consumption to GDP"] = gov_cons / gdp
        if "Total Consumption" in out.columns:
            out["Total Consumption to GDP"] = out["Total Consumption"] / gdp
        if gfcf is not None:
            out["Investment to GDP"] = gfcf / gdp
        if exports is not None and imports is not None:
            out["Net Exports to GDP"] = (exports - imports) / gdp

    passthrough_cols = [
        "CPI",
        "PPI",
        "Unemployment Rate",
        "Central Bank Policy Rate",
        "Consumption Expansion Loan Debt",
        "Mortgage Debt",
        "Wages",
        "Profits",
    ]
    for col in passthrough_cols:
        series = first_available(col)
        if series is not None:
            out[col] = series

    return out
