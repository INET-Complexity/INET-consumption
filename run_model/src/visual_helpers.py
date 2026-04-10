import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_output(df, no_rows, no_cols, country_code, x_col=None):
    columns_to_plot = df.columns.to_list()
    max_plots = no_rows * no_cols
    columns_to_plot = columns_to_plot[:max_plots]

    fig = make_subplots(
        rows=no_rows,
        cols=no_cols,
        subplot_titles=columns_to_plot
    )

    # choose x axis: explicit column, otherwise index
    if x_col is not None:
        x = df[x_col]
    else:
        x = df.index

    for idx, col_name in enumerate(columns_to_plot):
        row = (idx // no_cols) + 1
        col = (idx % no_cols) + 1

        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col_name],
                mode="lines",
                name=col_name,
                showlegend=False
            ),
            row=row,
            col=col
        )

    fig.update_layout(
        height=200 * no_rows,
        width=300 * no_cols,
        title_text=country_code,
        template="plotly_white"
    )
    fig.show()
