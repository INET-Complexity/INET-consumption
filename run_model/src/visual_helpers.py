import re
import warnings
from ast import literal_eval
from typing import Literal, overload

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


def firm_sector_by_index(
    model,
    country_code: str,
    firm_id: int | list[int] | tuple[int, ...] | np.ndarray,
    *,
    with_name: bool = False,
):
    """Return firm sector code (and optional label) for one or more firm indices.

    Notes
    -----
    - Firms store their sector as an integer index in ``firms.states['Industry']``.
    - The corresponding sector codes live in ``firms.industries`` (e.g. "A", "C", "G", ...).
    """
    country = model.countries[country_code]
    firms = country.firms

    industry_idx_by_firm = np.asarray(firms.states.get("Industry", []), dtype=int).reshape(-1)
    sector_codes = list(getattr(firms, "industries", []))

    def _one(idx: int):
        if idx < 0 or idx >= industry_idx_by_firm.size:
            raise IndexError(f"firm_id out of bounds: {idx} (n_firms={industry_idx_by_firm.size})")
        sector_idx = int(industry_idx_by_firm[idx])
        code = sector_codes[sector_idx] if 0 <= sector_idx < len(sector_codes) else str(sector_idx)
        if with_name:
            return (code, SECTOR_CODE_TO_NAME.get(code, code))
        return code

    if isinstance(firm_id, (list, tuple, np.ndarray)):
        return [_one(int(i)) for i in list(firm_id)]
    return _one(int(firm_id))


def sector_by_index(
    model,
    country_code: str,
    sector_index: int | list[int] | tuple[int, ...] | np.ndarray,
    *,
    with_name: bool = False,
):
    """Return sector code (and optional label) for one or more sector indices.

    Examples
    --------
    - ``sector_by_index(model, "FRA", 0)`` -> "A"
    - ``sector_by_index(model, "FRA", 0, with_name=True)`` -> ("A", "Agriculture, forestry and fishing")
    """
    country = model.countries[country_code]
    sector_codes = list(getattr(country.firms, "industries", []))

    def _one(idx: int):
        if idx < 0 or idx >= len(sector_codes):
            raise IndexError(f"sector_index out of bounds: {idx} (n_sectors={len(sector_codes)})")
        code = str(sector_codes[idx])
        if with_name:
            return (code, SECTOR_CODE_TO_NAME.get(code, code))
        return code

    if isinstance(sector_index, (list, tuple, np.ndarray)):
        return [_one(int(i)) for i in list(sector_index)]
    return _one(int(sector_index))


def firms_by_sector(
    model,
    country_code: str,
    *,
    with_name: bool = False,
) -> dict:
    """Return a mapping of sector -> list of firm ids (indices).

    This answers: "which firms belong to what sector?"

    Returns
    -------
    dict
        Keys are sector codes (e.g. "A") or (code, label) tuples when
        ``with_name=True``. Values are lists of firm indices.
    """
    country = model.countries[country_code]
    firms = country.firms

    industry_idx_by_firm = np.asarray(firms.states.get("Industry", []), dtype=int).reshape(-1)
    sector_codes = list(getattr(firms, "industries", []))

    out: dict = {}
    for firm_id, sector_idx in enumerate(industry_idx_by_firm.tolist()):
        if 0 <= sector_idx < len(sector_codes):
            code = str(sector_codes[sector_idx])
        else:
            code = str(sector_idx)
        key = (code, SECTOR_CODE_TO_NAME.get(code, code)) if with_name else code
        out.setdefault(key, []).append(int(firm_id))
    return out


def firm_sector_table(model, country_code: str) -> pd.DataFrame:
    """Return a table with firm_id -> sector index/code/name."""
    country = model.countries[country_code]
    firms = country.firms

    industry_idx_by_firm = np.asarray(firms.states.get("Industry", []), dtype=int).reshape(-1)
    sector_codes = list(getattr(firms, "industries", []))

    codes = []
    names = []
    for idx in industry_idx_by_firm.tolist():
        if 0 <= idx < len(sector_codes):
            code = str(sector_codes[idx])
        else:
            code = str(idx)
        codes.append(code)
        names.append(SECTOR_CODE_TO_NAME.get(code, code))

    return pd.DataFrame(
        {
            "firm_id": np.arange(industry_idx_by_firm.size, dtype=int),
            "sector_index": industry_idx_by_firm.astype(int, copy=False),
            "sector_code": np.asarray(codes, dtype=object),
            "sector_name": np.asarray(names, dtype=object),
        }
    )


def firm_sector_groups_table(model, country_code: str) -> pd.DataFrame:
    """Return a sector-level table listing which firm indices belong to each sector.

    Output columns:
    - sector_code
    - sector_name
    - firms (compressed ranges string, e.g. "0-12, 17, 20-25")
    - n_firms
    """

    def _format_ranges(ids: list[int]) -> str:
        if not ids:
            return ""
        ids_sorted = sorted(set(int(i) for i in ids))
        ranges: list[tuple[int, int]] = []
        start = prev = ids_sorted[0]
        for x in ids_sorted[1:]:
            if x == prev + 1:
                prev = x
                continue
            ranges.append((start, prev))
            start = prev = x
        ranges.append((start, prev))

        parts = []
        for a, b in ranges:
            parts.append(str(a) if a == b else f"{a}-{b}")
        return ", ".join(parts)

    mapping = firms_by_sector(model, country_code, with_name=True)
    rows = []
    for (code, name), firm_ids in sorted(mapping.items(), key=lambda x: str(x[0][0])):
        rows.append(
            {
                "sector_code": code,
                "sector_name": name,
                "firms": _format_ranges(firm_ids),
                "n_firms": int(len(firm_ids)),
            }
        )
    return pd.DataFrame(rows)


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


def _safe_ts_values(ts, name):
    """Return time-series values for ``name`` from a TimeSeries-like object.

    Supports the project's ``TimeSeries`` (via ``historic``) and simple namespaces
    used in tests. Returns ``None`` when the field is unavailable.
    """
    if ts is None:
        return None
    try:
        if hasattr(ts, "historic"):
            return ts.historic(name)
        if hasattr(ts, "dicts") and isinstance(ts.dicts, dict):
            return ts.dicts.get(name)
        return getattr(ts, name)
    except (AttributeError, KeyError):
        return None


def _safe_ts_initial_or_first(ts, name):
    if ts is None:
        return None
    if hasattr(ts, "initial"):
        try:
            return ts.initial(name)
        except (KeyError, AttributeError):
            pass
    values = _safe_ts_values(ts, name)
    if values:
        return values[0]
    return None


def _to_scalar(value, default=np.nan) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0:
        return float(default)
    return float(array[0])


def _to_1d_array(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _compute_bank_credit_supply_caps(country):
    """Return bank-level credit supply caps by loan type (time x bank arrays)."""
    banks = getattr(country, "banks", None)
    economy = getattr(country, "economy", None)
    credit_market = getattr(country, "credit_market", None)
    if banks is None or getattr(banks, "ts", None) is None:
        return None

    equity_hist = _safe_ts_values(banks.ts, "equity")
    loans_hist = _safe_ts_values(banks.ts, "total_outstanding_loans")
    if equity_hist is None or loans_hist is None:
        return None

    car = getattr(getattr(banks, "parameters", None), "capital_adequacy_ratio", None)
    if car is None or not np.isfinite(car) or car <= 0:
        return None

    npl_firm_hist = _safe_ts_values(getattr(economy, "ts", None), "npl_firm_loans") if economy is not None else None
    npl_hh_cons_hist = (
        _safe_ts_values(getattr(economy, "ts", None), "npl_hh_cons_loans") if economy is not None else None
    )
    npl_mort_hist = _safe_ts_values(getattr(economy, "ts", None), "npl_mortgages") if economy is not None else None

    clearing = None
    if credit_market is not None and hasattr(credit_market, "functions"):
        clearing = credit_market.functions.get("clearing")
    credit_supply_temperature = float(getattr(clearing, "credit_supply_temperature", 0.0) or 0.0)

    firm_share0 = _safe_ts_initial_or_first(banks.ts, "new_loans_fraction_firms")
    hh_cons_share0 = _safe_ts_initial_or_first(banks.ts, "new_loans_fraction_hh_cons")
    mort_share0 = _safe_ts_initial_or_first(banks.ts, "new_loans_fraction_mortgages")

    firm_share0 = _to_1d_array(firm_share0) if firm_share0 is not None else None
    hh_cons_share0 = _to_1d_array(hh_cons_share0) if hh_cons_share0 is not None else None
    mort_share0 = _to_1d_array(mort_share0) if mort_share0 is not None else None

    horizon = min(len(equity_hist), len(loans_hist))
    if horizon <= 0:
        return None

    equity0 = _to_1d_array(equity_hist[0])
    loans0 = _to_1d_array(loans_hist[0])
    n_banks = int(max(equity0.size, loans0.size))
    if n_banks <= 0:
        return None

    if firm_share0 is None or firm_share0.size != n_banks:
        firm_share0 = np.full(n_banks, 1.0 / 3.0)
    if hh_cons_share0 is None or hh_cons_share0.size != n_banks:
        hh_cons_share0 = np.full(n_banks, 1.0 / 3.0)
    if mort_share0 is None or mort_share0.size != n_banks:
        mort_share0 = np.full(n_banks, 1.0 / 3.0)

    total_caps = np.full((horizon, n_banks), np.nan)
    firm_caps = np.full((horizon, n_banks), np.nan)
    hh_cons_caps = np.full((horizon, n_banks), np.nan)
    mort_caps = np.full((horizon, n_banks), np.nan)

    for t in range(horizon):
        equity = _to_1d_array(equity_hist[t])
        loans = _to_1d_array(loans_hist[t])
        if equity.size != n_banks:
            equity = np.resize(equity, n_banks)
        if loans.size != n_banks:
            loans = np.resize(loans, n_banks)

        max_car = np.maximum(0.0, equity / car - loans)
        total_caps[t, :] = max_car

        npl_firm = _to_scalar(npl_firm_hist[t] if npl_firm_hist is not None and t < len(npl_firm_hist) else 0.0, 0.0)
        npl_hh = _to_scalar(
            npl_hh_cons_hist[t] if npl_hh_cons_hist is not None and t < len(npl_hh_cons_hist) else 0.0, 0.0
        )
        npl_mort = _to_scalar(npl_mort_hist[t] if npl_mort_hist is not None and t < len(npl_mort_hist) else 0.0, 0.0)

        firm_weights = firm_share0 * np.exp(-credit_supply_temperature * npl_firm)
        hh_cons_weights = hh_cons_share0 * np.exp(-credit_supply_temperature * npl_hh)
        mort_weights = mort_share0 * np.exp(-credit_supply_temperature * npl_mort)
        weights_sum = firm_weights + hh_cons_weights + mort_weights

        firm_caps[t, :] = np.divide(
            max_car * firm_weights,
            weights_sum,
            out=np.zeros(n_banks),
            where=weights_sum != 0.0,
        )
        hh_cons_caps[t, :] = np.divide(
            max_car * hh_cons_weights,
            weights_sum,
            out=np.zeros(n_banks),
            where=weights_sum != 0.0,
        )
        mort_caps[t, :] = np.divide(
            max_car * mort_weights,
            weights_sum,
            out=np.zeros(n_banks),
            where=weights_sum != 0.0,
        )

    return {
        "total": total_caps,
        "firms": firm_caps,
        "households_consumption": hh_cons_caps,
        "mortgages": mort_caps,
    }


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
        "total_public_pension_benefits": "public_pension_benefits",
        "total_other_social_transfers": "other_social_transfers",
        "total_necessity_support": "necessity_support",
        "total_household_social_transfers": "other_benefits",
    }.items():
        if source_name in df_gov_ts.columns:
            add_column(output_name, df_gov_ts[source_name])

    # `other_benefits` is retained for existing notebooks. The explicit name
    # makes its boundary clear: public pensions + other transfers + Stage-5
    # support, excluding state-contingent unemployment benefits.
    if "total_household_social_transfers" in df_gov_ts.columns:
        add_column("household_social_transfers", df_gov_ts["total_household_social_transfers"])

    fiscal_revenue_components = {
        "taxes_vat": "fiscal_revenue_vat",
        "taxes_production": "fiscal_revenue_production_taxes",
        "taxes_cf": "fiscal_revenue_capital_formation_taxes",
        "taxes_corporate_income": "fiscal_revenue_corporate_income_taxes",
        "taxes_exports": "fiscal_revenue_export_taxes",
        "taxes_income": "fiscal_revenue_income_taxes",
        "taxes_rental_income": "fiscal_revenue_rental_income_taxes",
        "taxes_employee_si": "fiscal_revenue_employee_social_insurance",
        "taxes_employer_si": "fiscal_revenue_employer_social_insurance",
        "taxes_on_products": "fiscal_revenue_taxes_on_products",
        "total_rent_received": "fiscal_revenue_social_housing_rent",
    }
    for source_name, output_name in fiscal_revenue_components.items():
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
    for component in [
        "public_pension_benefits",
        "other_social_transfers",
        "necessity_support",
        "household_social_transfers",
    ]:
        value = get_column(component)
        add_ratio(f"{component}_to_gdp", value, gdp)
        add_ratio(f"{component}_to_expenditure", value, fiscal_expenditure)

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

    bank_stock_columns = {
        "short_term_loans_to_firms": "total_short_term_loans_to_firms",
        "long_term_loans_to_firms": "total_long_term_loans_to_firms",
        "consumption_loans_to_households": "total_consumption_loans_to_households",
        "mortgages_to_households": "total_mortgages_to_households",
    }
    for output_name, source_name in bank_stock_columns.items():
        if source_name in df_bank_ts.columns:
            add_column(output_name, df_bank_ts[source_name])

    def add_agent_ts_column(output_name, ts_obj, ts_name):
        values = _safe_ts_values(ts_obj, ts_name)
        if values is not None:
            add_column(output_name, as_output_series(values))

    def add_aggregate_agent_ts_column(output_name, ts_obj, ts_name):
        values = _safe_ts_values(ts_obj, ts_name)
        if values is None:
            return None

        aggregated = []
        for value in list(values):
            array = np.asarray(value, dtype=float).reshape(-1)
            aggregated.append(float(np.nansum(array)) if array.size else np.nan)
        return add_column(output_name, as_output_series(aggregated))

    def aggregate_agent_ts(ts_obj, ts_name):
        values = _safe_ts_values(ts_obj, ts_name)
        if values is None:
            return None

        aggregated = []
        for value in list(values):
            array = np.asarray(value, dtype=float).reshape(-1)
            aggregated.append(float(np.nansum(array)) if array.size else np.nan)
        return as_output_series(aggregated)

    def aggregate_firm_ts_by_sector(ts_name):
        if firms_ts is None:
            return None
        values = _safe_ts_values(firms_ts, ts_name)
        if values is None:
            return None

        industry_index = np.asarray(getattr(country.firms, "states", {}).get("Industry", []), dtype=int).reshape(-1)
        sector_codes = list(getattr(country.firms, "industries", []))
        n_sectors = len(sector_codes)
        n_firms = industry_index.size
        if n_sectors == 0 or n_firms == 0:
            return None

        panel_rows = []
        for value in list(values):
            row = np.asarray(value, dtype=float).reshape(-1)
            if row.size != n_firms:
                row = np.resize(row, n_firms)
            safe_row = np.nan_to_num(row, nan=0.0)
            panel_rows.append(np.bincount(industry_index, weights=safe_row, minlength=n_sectors))

        panel = np.asarray(panel_rows, dtype=float)
        horizon = min(panel.shape[0], len(out_index))
        return pd.DataFrame(panel[:horizon], index=out_index[:horizon], columns=[str(code) for code in sector_codes])

    firms_ts = getattr(getattr(country, "firms", None), "ts", None)
    households_ts = getattr(getattr(country, "households", None), "ts", None)
    add_aggregate_agent_ts_column("real_demand", firms_ts, "demand")
    add_aggregate_agent_ts_column("inventory", firms_ts, "inventory")
    add_aggregate_agent_ts_column("inventory_nominal", firms_ts, "inventory_nominal")

    add_agent_ts_column("firm_credit_demand_short_term", firms_ts, "total_target_short_term_credit")
    add_agent_ts_column("firm_credit_demand_long_term", firms_ts, "total_target_long_term_credit")
    add_agent_ts_column("firm_credit_received_short_term", firms_ts, "total_received_short_term_credit")
    add_agent_ts_column("firm_credit_received_long_term", firms_ts, "total_received_long_term_credit")

    add_agent_ts_column("household_credit_demand_consumption", households_ts, "total_target_consumption_loans")
    add_agent_ts_column("household_credit_demand_mortgage", households_ts, "total_target_mortgage")
    add_agent_ts_column("household_credit_received_consumption", households_ts, "total_received_consumption_loans")
    add_agent_ts_column("household_credit_received_mortgage", households_ts, "total_received_mortgages")

    hh_assets = add_column("household_assets", aggregate_agent_ts(households_ts, "wealth"))
    hh_liabilities = add_column("household_liabilities", aggregate_agent_ts(households_ts, "debt"))
    hh_equity = add_column("household_net_worth", aggregate_agent_ts(households_ts, "net_wealth"))
    if hh_assets is not None and hh_liabilities is not None and hh_equity is not None:
        add_column("household_balance_sheet_identity_residual", hh_assets - hh_liabilities - hh_equity)

    firm_assets = None
    firm_inventory_nominal = aggregate_agent_ts(firms_ts, "inventory_nominal")
    firm_intermediate_stock_value = aggregate_agent_ts(firms_ts, "intermediate_inputs_stock_value")
    firm_capital_stock_value = aggregate_agent_ts(firms_ts, "capital_inputs_stock_value")
    firm_deposits = aggregate_agent_ts(firms_ts, "deposits")
    if (
        firm_inventory_nominal is not None
        and firm_intermediate_stock_value is not None
        and firm_capital_stock_value is not None
        and firm_deposits is not None
    ):
        firm_assets = add_column(
            "firm_assets",
            firm_inventory_nominal + firm_intermediate_stock_value + firm_capital_stock_value + firm_deposits,
        )

    firm_revolving_facility = aggregate_agent_ts(firms_ts, "operating_revolving_closing_balance")
    firm_liabilities = None
    firm_debt = aggregate_agent_ts(firms_ts, "debt")
    if firm_debt is not None and firm_revolving_facility is not None:
        firm_liabilities = add_column(
            "firm_liabilities",
            firm_debt + np.maximum(0.0, firm_revolving_facility),
        )
    firm_equity = add_column("firm_equity", aggregate_agent_ts(firms_ts, "equity"))
    firm_settlement_residual = aggregate_agent_ts(firms_ts, "firm_settlement_balance_sheet_residual")
    if firm_settlement_residual is not None:
        add_column("firm_balance_sheet_identity_residual", firm_settlement_residual)
    elif firm_assets is not None and firm_liabilities is not None and firm_equity is not None:
        add_column("firm_balance_sheet_identity_residual", firm_assets - firm_liabilities - firm_equity)

    sector_firm_assets = None
    sector_inventory_nominal = aggregate_firm_ts_by_sector("inventory_nominal")
    sector_intermediate_stock_value = aggregate_firm_ts_by_sector("intermediate_inputs_stock_value")
    sector_capital_stock_value = aggregate_firm_ts_by_sector("capital_inputs_stock_value")
    sector_deposits = aggregate_firm_ts_by_sector("deposits")
    if (
        sector_inventory_nominal is not None
        and sector_intermediate_stock_value is not None
        and sector_capital_stock_value is not None
        and sector_deposits is not None
    ):
        sector_firm_assets = (
            sector_inventory_nominal + sector_intermediate_stock_value + sector_capital_stock_value + sector_deposits
        )

    sector_firm_debt = aggregate_firm_ts_by_sector("debt")
    sector_revolving_facility = aggregate_firm_ts_by_sector("operating_revolving_closing_balance")
    sector_firm_liabilities = None
    if sector_firm_debt is not None and sector_revolving_facility is not None:
        sector_firm_liabilities = sector_firm_debt + np.maximum(0.0, sector_revolving_facility)
    sector_firm_equity = aggregate_firm_ts_by_sector("equity")
    if sector_firm_assets is not None and sector_firm_liabilities is not None and sector_firm_equity is not None:
        sector_identity_residual = sector_firm_assets - sector_firm_liabilities - sector_firm_equity
        for sector_code in sector_identity_residual.columns:
            add_column(
                f"firm_balance_sheet_identity_residual_sector_{sector_code}",
                sector_identity_residual[sector_code].reindex(out_index),
            )

    bank_assets = None
    banks_ts = getattr(getattr(country, "banks", None), "ts", None)
    bank_deposits = add_column("bank_deposits", aggregate_agent_ts(banks_ts, "deposits"))
    add_column("bank_equity", aggregate_agent_ts(banks_ts, "equity"))
    bank_total_loans = add_column(
        "bank_total_outstanding_loans",
        aggregate_agent_ts(banks_ts, "total_outstanding_loans"),
    )
    if bank_total_loans is not None and bank_deposits is not None:
        bank_assets = add_column("bank_assets_proxy", bank_total_loans + np.maximum(0.0, bank_deposits))
    bank_liabilities = None
    if banks_ts is not None:
        bank_liabilities = add_column("bank_liabilities", aggregate_agent_ts(banks_ts, "liability"))
    if bank_assets is not None and bank_liabilities is not None:
        add_column("bank_balance_sheet_identity_residual_proxy", bank_assets - bank_liabilities)

    if (
        get_column("firm_credit_demand_short_term") is not None
        and get_column("firm_credit_received_short_term") is not None
    ):
        add_column(
            "firm_credit_rationed_short_term",
            get_column("firm_credit_demand_short_term") - get_column("firm_credit_received_short_term"),
        )
    if (
        get_column("firm_credit_demand_long_term") is not None
        and get_column("firm_credit_received_long_term") is not None
    ):
        add_column(
            "firm_credit_rationed_long_term",
            get_column("firm_credit_demand_long_term") - get_column("firm_credit_received_long_term"),
        )
    if (
        get_column("household_credit_demand_consumption") is not None
        and get_column("household_credit_received_consumption") is not None
    ):
        add_column(
            "household_credit_rationed_consumption",
            get_column("household_credit_demand_consumption") - get_column("household_credit_received_consumption"),
        )
    if (
        get_column("household_credit_demand_mortgage") is not None
        and get_column("household_credit_received_mortgage") is not None
    ):
        add_column(
            "household_credit_rationed_mortgage",
            get_column("household_credit_demand_mortgage") - get_column("household_credit_received_mortgage"),
        )

    credit_market_ts = getattr(getattr(country, "credit_market", None), "ts", None)
    add_agent_ts_column(
        "new_credit_granted_firms_short_term", credit_market_ts, "total_newly_loans_granted_firms_short_term"
    )
    add_agent_ts_column(
        "new_credit_granted_firms_long_term", credit_market_ts, "total_newly_loans_granted_firms_long_term"
    )
    add_agent_ts_column(
        "new_credit_granted_households_consumption",
        credit_market_ts,
        "total_newly_loans_granted_households_consumption",
    )
    add_agent_ts_column("new_credit_granted_mortgages", credit_market_ts, "total_newly_loans_granted_mortgages")

    bank_supply_caps = _compute_bank_credit_supply_caps(country)
    if bank_supply_caps is not None:
        add_column("bank_credit_supply_cap_total", as_output_series(np.nansum(bank_supply_caps["total"], axis=1)))
        add_column("bank_credit_supply_cap_firms", as_output_series(np.nansum(bank_supply_caps["firms"], axis=1)))
        add_column(
            "bank_credit_supply_cap_households_consumption",
            as_output_series(np.nansum(bank_supply_caps["households_consumption"], axis=1)),
        )
        add_column(
            "bank_credit_supply_cap_mortgages", as_output_series(np.nansum(bank_supply_caps["mortgages"], axis=1))
        )

    def add_economy_column(name):
        values = economy_ts_dict.get(name)
        if values is None:
            return None
        return add_column(name, as_output_series(values))

    def add_average_firm_timeseries_column(output_name, source_name):
        values = getattr(country.firms.ts, source_name, None)
        if values is None:
            return None

        averages = []
        for value in list(values):
            array = np.asarray(value, dtype=float)
            averages.append(float(np.nanmean(array)) if array.size else np.nan)

        return add_column(output_name, as_output_series(averages))

    def add_firm_vector_columns(name):
        values = getattr(country.firms.ts, name, None)
        if values is None:
            return

        series = add_column(name, as_output_series(values))
        non_null = series.dropna()
        if non_null.empty or not isinstance(non_null.iloc[0], list):
            return

        labels = sector_labels(len(non_null.iloc[0]))
        expanded = pd.DataFrame(series.tolist(), index=out_index, columns=[f"{name}_{label}" for label in labels])
        for column in expanded.columns:
            output_columns[column] = expanded[column]

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
        "labour_input_shortfall_rate",
        "labour_input_shortfall_rate_growth",
        "unfilled_jobs",
        "vacancy_rate",
        "vacancy_rate_growth",
        "job_reallocation_rate",
        "job_reallocation_rate_growth",
        "firm_insolvency_rate",
        "bank_insolvency_rate",
        "household_insolvency_rate",
        "illiquid_financial_asset_return_rate",
        "total_growth",
        "estimated_growth",
        "cpi_transaction",
        "cpi_transaction_pop_change",
        "cpi_transaction_yoy_change",
        "real_gross_output",
        "real_demand",
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
    for firm_column in [
        "sector_tfp_investment_desired_mb_mc_ratio",
    ]:
        add_firm_vector_columns(firm_column)
    add_average_firm_timeseries_column("avg_tfp_multiplier", "tfp_multiplier")

    try:
        wage_rate_by_sector = build_wage_rate_by_sector_df(model, country_code)
        if "economy" in wage_rate_by_sector.columns:
            add_column("economy_wage_rate", wage_rate_by_sector["economy"].reindex(out_index))
    except (AttributeError, KeyError, ValueError):
        pass

    out = pd.DataFrame(output_columns, index=out_index).copy()
    out.attrs["time_unit_months"] = model.timestep.increment
    out.attrs["observation_frequency"] = frequency_label(model.timestep.increment)
    out.attrs["periods_per_year"] = periods_per_year
    out.attrs["interest_rate_observation_frequency"] = out.attrs["observation_frequency"]
    out.attrs["interest_rate_units"] = "Macro interest-rate columns are annualized."

    return out


def _panel_to_frame(values, index, width=None):
    values = list(values)
    horizon = min(len(index), len(values))
    arrays = []
    max_width = 0
    for value in values[:horizon]:
        array = np.asarray(value, dtype=float).reshape(-1)
        arrays.append(array)
        max_width = max(max_width, array.size)

    if width is None:
        width = max_width
    width = int(width)
    if width <= 0:
        return pd.DataFrame(index=index[:horizon])

    data = np.full((horizon, width), np.nan)
    for t, array in enumerate(arrays):
        if array.size:
            data[t, : min(width, array.size)] = array[:width]
    return pd.DataFrame(data, index=index[:horizon], columns=range(width))


FIRM_BALANCE_SHEET_COLUMNS = [
    "inventory",
    "inventory_nominal",
    "intermediate_inputs_stock",
    "intermediate_inputs_stock_value",
    "intermediate_inputs_stock_industry",
    "capital_inputs_stock",
    "capital_inputs_stock_value",
    "capital_inputs_stock_industry",
    "short_term_loan_debt",
    "long_term_loan_debt",
    "debt",
    "total_credit_exposure",
    "deposits",
    "total_deposits",
    "equity",
]


FIRM_TRANSACTION_ACCOUNT_COLUMNS = [
    "production",
    "production_nominal",
    "price",
    "price_offered",
    "price_in_usd",
    "profits",
    "total_wage",
    "unit_costs",
    "taxes_paid_on_production",
    "corporate_taxes_paid",
    "activity_finance_available",
    "activity_finance_hard_obligations",
    "activity_finance_gap_before_revision",
    "activity_finance_opening_deposits",
    "activity_finance_feasibility_residual",
    "activity_finance_realised_feasible_target_production",
    "activity_finance_realised_labour_scale",
    "intermediate_purchase_finance_scale",
    "capital_purchase_finance_scale",
    "technical_investment_finance_scale",
    "tfp_investment_finance_scale",
    "executed_productivity_investment",
    "executed_tfp_investment",
    "direct_tfp_investment_cash_expense",
    "technical_investment_by_input",
    "total_inventory_change",
    "used_intermediate_inputs",
    "used_intermediate_inputs_costs",
    "total_intermediate_inputs_bought_costs",
    "used_capital_inputs",
    "used_capital_inputs_costs",
    "capital_depreciation_costs",
    "total_capital_inputs_bought_costs",
    "gross_fixed_capital_formation",
    "total_sales",
    "credit_budget_internal_cash",
    "credit_budget_existing_overdraft",
    "credit_budget_wage_obligations",
    "credit_budget_production_tax_obligations",
    "credit_budget_corporate_tax_obligations",
    "credit_budget_interest_obligations",
    "credit_budget_debt_installments",
    "credit_budget_hard_obligations",
    "credit_budget_cash_after_hard_obligations",
    "credit_budget_available_after_hard_and_overdraft",
    "credit_budget_intermediate_costs",
    "credit_budget_tfp_costs",
    "credit_budget_working_capital_budget",
    "credit_budget_capital_costs",
    "credit_budget_technical_investment_costs",
    "credit_budget_investment_budget",
    "credit_budget_remaining_internal_finance_after_working_capital",
    "received_short_term_credit",
    "total_received_short_term_credit",
    "received_debt_rollover_credit",
    "total_received_debt_rollover_credit",
    "received_overdraft_refinance_credit",
    "total_received_overdraft_refinance_credit",
    "received_ordinary_short_term_credit",
    "total_received_ordinary_short_term_credit",
    "received_long_term_credit",
    "total_received_long_term_credit",
    "received_credit",
    "firm_settlement_available_cash_before_debt_service",
    "firm_settlement_corporate_tax_reserve",
    "firm_settlement_cash_after_tax_reserve",
    "firm_settlement_opening_interest_arrears",
    "firm_settlement_scheduled_interest_due",
    "firm_settlement_contractual_interest_due",
    "firm_settlement_payable_interest",
    "firm_settlement_closing_interest_arrears",
    "firm_settlement_unpaid_interest",
    "firm_settlement_capitalized_interest",
    "firm_settlement_opening_principal_arrears",
    "firm_settlement_scheduled_principal_due",
    "firm_settlement_contractual_principal_due",
    "firm_settlement_payable_principal",
    "firm_settlement_closing_principal_arrears",
    "firm_settlement_unpaid_principal",
    "firm_settlement_debt_rollover_shortfall",
    "firm_settlement_overdraft_refinance_used",
    "firm_settlement_overdraft_refinance_shortfall",
    "firm_settlement_residual_overdraft_exposure",
    "firm_settlement_illiquid_flag",
    "firm_settlement_default_flag",
    "credit_market_firm_cfads",
    "credit_market_firm_st_capacity",
    "credit_market_firm_st_collateral_cap",
    "credit_market_firm_st_dscr_cap",
    "credit_market_firm_st_binding_reason",
    "credit_market_firm_st_binding_amount",
    "credit_market_firm_lt_capacity",
    "credit_market_firm_lt_collateral_cap",
    "credit_market_firm_lt_dscr_cap",
    "credit_market_firm_lt_binding_reason",
    "credit_market_firm_lt_binding_amount",
    "excess_demand_finance_cash",
    "excess_demand_borrower_st_credit_room",
    "excess_demand_borrower_lt_credit_room",
    "excess_demand_borrower_total_credit_room",
    "excess_demand_repair_cash_used",
    "excess_demand_residual_repair_credit_need",
    "excess_demand_borrower_max_credit",
    "excess_demand_activity_finance_borrower",
    "excess_demand_finance_potential_output_borrower",
    "excess_demand_potential_capacity_borrower",
    "excess_demand_above_borrower_cap_share",
    "excess_demand_supply_max_credit",
    "excess_demand_activity_finance_supply",
    "excess_demand_finance_potential_output_supply",
    "excess_demand_potential_capacity_supply",
    "excess_demand_above_supply_cap_share",
    "debt_installments",
    "scheduled_debt_service",
    "total_debt_installments",
    "interest_paid_on_deposits",
    "interest_paid_on_loans",
    "interest_paid",
    "labour_inputs",
    "labour_costs",
    "gross_operating_surplus_mixed_income",
    "nominal_amount_sold_in_lcu",
    "nominal_amount_sold_in_lcu_to_FRA",
    "nominal_amount_sold_in_lcu_to_ROW",
    "nominal_amount_spent_in_usd",
    "nominal_amount_spent_in_usd_to_FRA",
    "nominal_amount_spent_in_usd_to_ROW",
    "nominal_amount_spent_in_lcu",
    "nominal_amount_spent_in_lcu_to_FRA",
    "nominal_amount_spent_in_lcu_to_ROW",
]


def split_firm_ts_balance_sheet_and_transaction_account_df(
    model,
    country_code: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``firms.ts`` into balance sheet and transaction account views.

    The split is intentionally notebook-friendly: columns that are unavailable in
    ``firms.ts`` are ignored, and only nominal realized firm series are included.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(balance_sheet_df, transaction_account_df)``.
    """
    country = model.countries[country_code]
    firms_ts = getattr(getattr(country, "firms", None), "ts", None)
    if firms_ts is None:
        raise ValueError(f"firms.ts is missing for country {country_code!r}.")

    out_index = model.shallow_df_dict()[country_code].index

    def _build_frame(columns: list[str]) -> pd.DataFrame:
        data: dict[str, pd.Series] = {}
        for col in columns:
            values = _safe_ts_values(firms_ts, col)
            if values is None:
                continue
            values = list(values)
            horizon = min(len(out_index), len(values))
            data[col] = pd.Series(values[:horizon], index=out_index[:horizon]).reindex(out_index)
        return pd.DataFrame(data, index=out_index)

    balance_sheet_df = _build_frame(FIRM_BALANCE_SHEET_COLUMNS)
    transaction_account_df = _build_frame(FIRM_TRANSACTION_ACCOUNT_COLUMNS)
    return balance_sheet_df, transaction_account_df


def build_credit_demand_by_agent_df(model, country_code, agent_kind="firms", include_received=True):
    """Return credit demand (and optionally received credit) by agent and type.

    Output is long-form with a MultiIndex (t, agent_id) and one column per loan type.
    """
    out_index = model.shallow_df_dict()[country_code].index
    country = model.countries[country_code]

    agent_kind = str(agent_kind).lower()
    if agent_kind not in {"firms", "households"}:
        raise ValueError("agent_kind must be 'firms' or 'households'.")

    if agent_kind == "firms":
        ts = getattr(getattr(country, "firms", None), "ts", None)
        if ts is None:
            raise ValueError(f"No firms.ts found for country {country_code!r}.")
        target_st = _safe_ts_values(ts, "target_short_term_credit")
        target_lt = _safe_ts_values(ts, "target_long_term_credit")
        if target_st is None or target_lt is None:
            raise ValueError("Firm target credit series not found on firms.ts.")

        df_target_st = _panel_to_frame(target_st, out_index)
        df_target_lt = _panel_to_frame(target_lt, out_index, width=df_target_st.shape[1])
        frames = {
            "target_short_term_credit": df_target_st,
            "target_long_term_credit": df_target_lt,
        }
        if include_received:
            received_st = _safe_ts_values(ts, "received_short_term_credit")
            received_lt = _safe_ts_values(ts, "received_long_term_credit")
            if received_st is not None:
                frames["received_short_term_credit"] = _panel_to_frame(
                    received_st, out_index, width=df_target_st.shape[1]
                )
            if received_lt is not None:
                frames["received_long_term_credit"] = _panel_to_frame(
                    received_lt, out_index, width=df_target_st.shape[1]
                )
    else:
        ts = getattr(getattr(country, "households", None), "ts", None)
        if ts is None:
            raise ValueError(f"No households.ts found for country {country_code!r}.")
        target_cons = _safe_ts_values(ts, "target_consumption_loans")
        target_mort = _safe_ts_values(ts, "target_mortgage")
        if target_cons is None or target_mort is None:
            raise ValueError("Household target credit series not found on households.ts.")

        df_target_cons = _panel_to_frame(target_cons, out_index)
        df_target_mort = _panel_to_frame(target_mort, out_index, width=df_target_cons.shape[1])
        frames = {
            "target_consumption_loans": df_target_cons,
            "target_mortgage": df_target_mort,
        }
        if include_received:
            received_cons = _safe_ts_values(ts, "received_consumption_loans")
            received_mort = _safe_ts_values(ts, "received_mortgages")
            if received_cons is not None:
                frames["received_consumption_loans"] = _panel_to_frame(
                    received_cons, out_index, width=df_target_cons.shape[1]
                )
            if received_mort is not None:
                frames["received_mortgages"] = _panel_to_frame(received_mort, out_index, width=df_target_cons.shape[1])

    combined = pd.concat(frames, axis=1)
    combined.columns.names = ["variable", "agent_id"]
    out = combined.stack(level="agent_id")
    out.index.names = ["t", "agent_id"]
    return out


def build_bank_credit_supply_by_type_df(model, country_code):
    """Return bank credit supply caps by type as a long-form dataframe.

    Output index is MultiIndex (t, bank_id). Columns are:
    - supply_cap_total
    - supply_cap_firms
    - supply_cap_households_consumption
    - supply_cap_mortgages
    """
    out_index = model.shallow_df_dict()[country_code].index
    country = model.countries[country_code]

    caps = _compute_bank_credit_supply_caps(country)
    if caps is None:
        raise ValueError(
            "Bank credit supply caps cannot be computed for this model/country (missing banks parameters/TS)."
        )

    horizon = min(len(out_index), caps["total"].shape[0])
    n_banks = caps["total"].shape[1]
    bank_index = range(n_banks)

    frames = {
        "supply_cap_total": pd.DataFrame(caps["total"][:horizon], index=out_index[:horizon], columns=bank_index),
        "supply_cap_firms": pd.DataFrame(caps["firms"][:horizon], index=out_index[:horizon], columns=bank_index),
        "supply_cap_households_consumption": pd.DataFrame(
            caps["households_consumption"][:horizon], index=out_index[:horizon], columns=bank_index
        ),
        "supply_cap_mortgages": pd.DataFrame(
            caps["mortgages"][:horizon], index=out_index[:horizon], columns=bank_index
        ),
    }

    combined = pd.concat(frames, axis=1)
    combined.columns.names = ["variable", "bank_id"]
    out = combined.stack(level="bank_id")
    out.index.names = ["t", "bank_id"]
    return out


def build_sector_tfp_investment_desired_mb_mc_ratio_df(
    source=None,
    country_code=None,
    column="sector_tfp_investment_desired_mb_mc_ratio",
    sector_labels=None,
):
    """Return sector desired MB/MC ratios from a model, firms ts, MC result, or dataframe."""
    if source is None:
        raise ValueError("Provide a model, firms time series, MonteCarloResult, or dataframe.")

    if hasattr(source, "countries"):
        if country_code is None:
            raise ValueError("country_code is required when source is a model.")
        country = source.countries[country_code]
        if sector_labels is None:
            sector_labels = [str(sector) for sector in getattr(country.firms, "industries", [])]
        source = country.firms.ts

    if not isinstance(source, pd.DataFrame) and hasattr(source, column):
        values = getattr(source, column)
        rows = [np.asarray(unpack_cell(value), dtype=float).reshape(-1) for value in list(values)]
        if not rows:
            raise ValueError(f"firms.ts.{column} has no values to plot.")
        width = rows[0].size
        if any(row.size != width for row in rows):
            raise ValueError(f"firms.ts.{column} has inconsistent sector widths.")
        if sector_labels is None:
            sector_labels = [str(idx) for idx in range(width)]
        elif len(sector_labels) != width:
            raise ValueError("sector_labels length must match the number of sectors.")
        return pd.DataFrame(rows, columns=[str(label) for label in sector_labels])

    combined = source.combined if hasattr(source, "combined") else source
    if not isinstance(combined, pd.DataFrame):
        raise TypeError("source must be a model, firms time series, MonteCarloResult, or pandas DataFrame.")
    if combined.index.nlevels < 2:
        raise ValueError("Monte Carlo dataframe must be indexed by seed and simulation time.")

    expanded_prefix = f"{column}_"
    expanded_columns = [col for col in combined.columns if str(col).startswith(expanded_prefix)]
    if expanded_columns:
        out = combined[expanded_columns].apply(pd.to_numeric, errors="coerce").copy()
        out = out.rename(columns={col: str(col).removeprefix(expanded_prefix) for col in expanded_columns})
        return out

    if column not in combined.columns:
        raise ValueError(f"Column {column!r} is not present in the Monte Carlo output.")

    rows = []
    width = None
    for value in combined[column]:
        array = np.asarray(unpack_cell(value), dtype=float).reshape(-1)
        if width is None:
            width = array.size
        if array.size != width:
            raise ValueError(f"Column {column!r} has inconsistent sector widths.")
        rows.append(array)

    if width is None:
        raise ValueError(f"Column {column!r} has no values to plot.")
    if sector_labels is None:
        sector_labels = [str(idx) for idx in range(width)]
    if len(sector_labels) != width:
        raise ValueError("sector_labels length must match the number of sectors.")

    return pd.DataFrame(rows, index=combined.index, columns=[str(label) for label in sector_labels])


def plot_sector_tfp_investment_desired_mb_mc_ratio(
    source=None,
    country_code=None,
    sector_labels=None,
    title="Sector TFP investment desired MB/MC ratio",
    line_opacity=0.45,
    line_width=1.2,
    height=650,
    width=1000,
    show=True,
):
    """Display sector desired marginal-benefit/marginal-cost ratios by seed."""
    ratios = build_sector_tfp_investment_desired_mb_mc_ratio_df(
        source=source,
        country_code=country_code,
        sector_labels=sector_labels,
    )
    sectors = list(ratios.columns)
    colors = _categorical_colors(len(sectors))

    fig = go.Figure()
    if ratios.index.nlevels >= 2:
        seeds = ratios.index.get_level_values(0).unique()
        for sector_idx, sector_code in enumerate(sectors):
            sector_label = SECTOR_CODE_TO_NAME.get(str(sector_code), str(sector_code))
            for seed_idx, seed in enumerate(seeds):
                seed_df = ratios.xs(seed, level=0)
                fig.add_trace(
                    go.Scatter(
                        x=seed_df.index,
                        y=seed_df[sector_code],
                        mode="lines",
                        name=f"{sector_code}: {sector_label}",
                        legendgroup=f"sector-{sector_code}",
                        showlegend=(seed_idx == 0),
                        opacity=line_opacity,
                        line={"color": colors[sector_idx], "width": line_width},
                        hovertemplate=(
                            f"seed={seed}<br>sector={sector_code}: {sector_label}"
                            "<br>time=%{x}<br>desired MB/MC=%{y:.4g}<extra></extra>"
                        ),
                    )
                )
    else:
        for sector_idx, sector_code in enumerate(sectors):
            sector_label = SECTOR_CODE_TO_NAME.get(str(sector_code), str(sector_code))
            fig.add_trace(
                go.Scatter(
                    x=ratios.index,
                    y=ratios[sector_code],
                    mode="lines",
                    name=f"{sector_code}: {sector_label}",
                    legendgroup=f"sector-{sector_code}",
                    showlegend=True,
                    line={"color": colors[sector_idx], "width": line_width},
                    hovertemplate=(
                        f"sector={sector_code}: {sector_label}<br>time=%{{x}}"
                        "<br>desired MB/MC=%{y:.4g}<extra></extra>"
                    ),
                )
            )

    fig.add_hline(y=1.0, line_width=1, line_color="black", line_dash="dash")
    fig.update_layout(
        height=height,
        width=width,
        title_text=title,
        template="plotly_white",
        xaxis_title="t",
        yaxis_title="desired MB/MC ratio",
        legend_title_text="Sector",
    )

    if show:
        fig.show()
        return None
    return fig


def build_cumulative_insolvent_firms_by_sector_df(df, base_column="num_insolvent_firms_by_sector"):
    """Return cumulative insolvent-firm counts for every expanded sector column."""
    sector_prefix = f"{base_column}_"
    sector_columns = [column for column in df.columns if column.startswith(sector_prefix)]
    if not sector_columns:
        raise ValueError(f"No sector columns found with prefix {sector_prefix!r}.")

    cumulative = df[sector_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).cumsum()
    cumulative = cumulative.rename(columns={column: column.removeprefix(sector_prefix) for column in sector_columns})
    cumulative.index = df.index
    return cumulative


def plot_cumulative_insolvent_firms_by_sector(
    df,
    title="Cumulative insolvent firms by sector",
    height=650,
    width=1000,
    show=True,
):
    """Plot cumulative insolvent-firm counts for each economic sector."""
    cumulative = build_cumulative_insolvent_firms_by_sector_df(df)
    colors = _categorical_colors(len(cumulative.columns))

    fig = go.Figure()
    for idx, sector_code in enumerate(cumulative.columns):
        sector_label = SECTOR_CODE_TO_NAME.get(str(sector_code), str(sector_code))
        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative[sector_code],
                mode="lines",
                name=f"{sector_code}: {sector_label}",
                line={"color": colors[idx], "width": 2},
            )
        )

    fig.update_layout(
        height=height,
        width=width,
        title_text=title,
        template="plotly_white",
        xaxis_title="t",
        yaxis_title="cumulative insolvent firms",
    )

    if show:
        fig.show()
        return None
    return fig


def build_employment_by_sector_df(model, country_code):
    """Return employed-individual counts by sector from a model country."""
    country = model.countries[country_code]
    values = np.asarray(country.labour_market.ts.num_employed_individuals_by_sector, dtype=float)
    if values.ndim != 2:
        raise ValueError("num_employed_individuals_by_sector must be a 2D time x sector series.")

    sectors = [str(sector) for sector in getattr(country.firms, "industries", [])]
    if len(sectors) != values.shape[1]:
        sectors = [str(idx) for idx in range(values.shape[1])]

    return pd.DataFrame(values, columns=sectors)


def plot_employment_by_sector(
    model,
    country_code,
    title="Number of employed individuals by sector",
    height=650,
    width=1000,
    show=True,
):
    """Plot employed-individual counts for each economic sector."""
    employment = build_employment_by_sector_df(model, country_code)
    colors = _categorical_colors(len(employment.columns))

    fig = go.Figure()
    for idx, sector_code in enumerate(employment.columns):
        sector_label = SECTOR_CODE_TO_NAME.get(str(sector_code), str(sector_code))
        fig.add_trace(
            go.Scatter(
                x=employment.index,
                y=employment[sector_code],
                mode="lines+markers",
                name=f"{sector_code}: {sector_label}",
                line={"color": colors[idx], "width": 2},
                marker={"size": 6},
            )
        )

    fig.update_layout(
        height=height,
        width=width,
        title_text=title,
        template="plotly_white",
        xaxis_title="Time Period",
        yaxis_title="Number of Employed Individuals",
        legend_title_text="Sector",
    )

    if show:
        fig.show()
        return None
    return fig


def build_production_by_sector_df(model, country_code):
    """Return total firm production by sector from a model country."""
    country = model.countries[country_code]
    production = np.asarray(country.firms.ts.historic("production"), dtype=float)
    if production.ndim != 2:
        raise ValueError("production must be a 2D time x firm series.")

    firm_industries = np.asarray(country.firms.states["Industry"], dtype=int)
    n_sectors = len(getattr(country.firms, "industries", [])) or int(firm_industries.max() + 1)
    by_sector = np.zeros((production.shape[0], n_sectors), dtype=float)
    for t in range(production.shape[0]):
        by_sector[t] = np.bincount(
            firm_industries,
            weights=production[t],
            minlength=n_sectors,
        )

    sectors = [str(sector) for sector in getattr(country.firms, "industries", [])]
    if len(sectors) != by_sector.shape[1]:
        sectors = [str(idx) for idx in range(by_sector.shape[1])]

    return pd.DataFrame(by_sector, columns=sectors)


def build_wage_rate_by_sector_df(model, country_code, *, real: bool = False, cpi_source: str = "cpi_fixed_basket"):
    """Return nominal or real wage rates by sector plus an economy-wide aggregate.

    The sector wage rate is computed as:

        sector wage bill / sector employment

    The economy wage rate is computed as:

        total wage bill / total employment

    If ``real=True``, both are deflated by the requested CPI series.
    """
    country = model.countries[country_code]

    wage_bills = np.asarray(country.firms.ts.historic("total_wage"), dtype=float)
    if wage_bills.ndim != 2:
        raise ValueError("total_wage must be a 2D time x firm series.")

    sector_employment = np.asarray(country.labour_market.ts.historic("num_employed_individuals_by_sector"), dtype=float)
    if sector_employment.ndim != 2:
        raise ValueError("num_employed_individuals_by_sector must be a 2D time x sector series.")

    firm_industries = np.asarray(country.firms.states["Industry"], dtype=int)
    n_sectors = len(getattr(country.firms, "industries", [])) or int(firm_industries.max() + 1)
    if wage_bills.shape[1] != firm_industries.size:
        raise ValueError("total_wage time series is not aligned with the firm industry map.")
    if sector_employment.shape[1] != n_sectors:
        raise ValueError("sector employment time series is not aligned with the firm industry map.")

    sector_wage_bills = np.zeros((wage_bills.shape[0], n_sectors), dtype=float)
    for t in range(wage_bills.shape[0]):
        sector_wage_bills[t] = np.bincount(
            firm_industries,
            weights=wage_bills[t],
            minlength=n_sectors,
        )

    sector_wage_rate = np.divide(
        sector_wage_bills,
        sector_employment,
        out=np.full_like(sector_wage_bills, np.nan),
        where=sector_employment > 0.0,
    )

    total_wage_bills = sector_wage_bills.sum(axis=1)
    total_employment = sector_employment.sum(axis=1)
    economy_wage_rate = np.divide(
        total_wage_bills,
        total_employment,
        out=np.full(total_wage_bills.shape, np.nan, dtype=float),
        where=total_employment > 0.0,
    )

    if real:
        cpi = np.asarray(country.economy.ts.historic(cpi_source), dtype=float).reshape(-1)
        if cpi.ndim != 1:
            raise ValueError(f"{cpi_source} must be a 1D time series.")
        if cpi.size != sector_wage_rate.shape[0]:
            raise ValueError(f"{cpi_source} is not aligned with the wage-rate time series.")
        sector_wage_rate = np.divide(
            sector_wage_rate,
            cpi[:, None],
            out=np.full_like(sector_wage_rate, np.nan),
            where=cpi[:, None] > 0.0,
        )
        economy_wage_rate = np.divide(
            economy_wage_rate,
            cpi,
            out=np.full_like(economy_wage_rate, np.nan),
            where=cpi > 0.0,
        )

    sectors = [str(sector) for sector in getattr(country.firms, "industries", [])]
    if len(sectors) != sector_wage_rate.shape[1]:
        sectors = [str(idx) for idx in range(sector_wage_rate.shape[1])]

    out = pd.DataFrame(sector_wage_rate, columns=sectors)
    out["economy"] = economy_wage_rate
    return out


def plot_wage_rate_by_sector(
    model,
    country_code,
    *,
    real: bool = False,
    cpi_source: str = "cpi_fixed_basket",
    title: str | None = None,
    height=650,
    width=1000,
    show=True,
):
    """Plot wage rates by sector together with the economy-wide wage rate."""
    wage_rate = build_wage_rate_by_sector_df(model, country_code, real=real, cpi_source=cpi_source)
    colors = _categorical_colors(max(0, len(wage_rate.columns) - 1))

    fig = go.Figure()
    sector_columns = [col for col in wage_rate.columns if col != "economy"]
    for idx, sector_code in enumerate(sector_columns):
        sector_label = SECTOR_CODE_TO_NAME.get(str(sector_code), str(sector_code))
        fig.add_trace(
            go.Scatter(
                x=wage_rate.index,
                y=wage_rate[sector_code],
                mode="lines",
                name=f"{sector_code}: {sector_label}",
                line={"color": colors[idx], "width": 2},
            )
        )

    fig.add_trace(
        go.Scatter(
            x=wage_rate.index,
            y=wage_rate["economy"],
            mode="lines",
            name="economy",
            line={"color": "black", "width": 3, "dash": "dash"},
        )
    )

    if title is None:
        title = "Real wage rate by sector and economy" if real else "Nominal wage rate by sector and economy"

    fig.update_layout(
        height=height,
        width=width,
        title_text=title,
        template="plotly_white",
        xaxis_title="Time Period",
        yaxis_title="Real wage rate" if real else "Nominal wage rate",
        legend_title_text="Sector",
    )

    if show:
        fig.show()
        return None
    return fig


def plot_production_by_sector(
    model,
    country_code,
    title="Production by sector (model)",
    height=650,
    width=1000,
    show=True,
):
    """Plot total firm production for each economic sector."""
    production = build_production_by_sector_df(model, country_code)
    colors = _categorical_colors(len(production.columns))

    fig = go.Figure()
    for idx, sector_code in enumerate(production.columns):
        sector_label = SECTOR_CODE_TO_NAME.get(str(sector_code), str(sector_code))
        fig.add_trace(
            go.Scatter(
                x=production.index,
                y=production[sector_code],
                mode="lines+markers",
                name=f"{sector_code}: {sector_label}",
                line={"color": colors[idx], "width": 2},
                marker={"size": 6},
            )
        )

    fig.update_layout(
        height=height,
        width=width,
        title_text=title,
        template="plotly_white",
        xaxis_title="Time Period",
        yaxis_title="Production",
        legend_title_text="Sector",
    )

    if show:
        fig.show()
        return None
    return fig


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
    base_name="scenario",
    compare_name="benchmark",
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


@overload
def plot_agent_timeseries(
    model,
    country_code: str,
    agent_type: str,
    variables,
    *,
    agent_id: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    agg: str = "sum",
    by_sector: bool = False,
    panel_titles: list[str] | None = None,
    no_cols: int | None = None,
    base_height: int = 220,
    base_width: int = 380,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    shared_xaxes: bool = True,
    show_legend: bool | None = None,
    line_width: float = 2.0,
    firm_ids: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    condition: str | None = None,
    return_info: Literal[False] = False,
    show: Literal[True] = True,
    return_df: Literal[False] = False,
): ...


@overload
def plot_agent_timeseries(
    model,
    country_code: str,
    agent_type: str,
    variables,
    *,
    agent_id: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    agg: str = "sum",
    by_sector: bool = False,
    panel_titles: list[str] | None = None,
    no_cols: int | None = None,
    base_height: int = 220,
    base_width: int = 380,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    shared_xaxes: bool = True,
    show_legend: bool | None = None,
    line_width: float = 2.0,
    firm_ids: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    condition: str | None = None,
    return_info: Literal[False] = False,
    show: Literal[False],
    return_df: Literal[False] = False,
) -> go.Figure: ...


@overload
def plot_agent_timeseries(
    model,
    country_code: str,
    agent_type: str,
    variables,
    *,
    agent_id: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    agg: str = "sum",
    by_sector: bool = False,
    panel_titles: list[str] | None = None,
    no_cols: int | None = None,
    base_height: int = 220,
    base_width: int = 380,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    shared_xaxes: bool = True,
    show_legend: bool | None = None,
    line_width: float = 2.0,
    firm_ids: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    condition: str | None = None,
    return_info: Literal[False] = False,
    show: Literal[True] = True,
    return_df: Literal[True],
) -> tuple[None, pd.DataFrame]: ...


@overload
def plot_agent_timeseries(
    model,
    country_code: str,
    agent_type: str,
    variables,
    *,
    agent_id: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    agg: str = "sum",
    by_sector: bool = False,
    panel_titles: list[str] | None = None,
    no_cols: int | None = None,
    base_height: int = 220,
    base_width: int = 380,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    shared_xaxes: bool = True,
    show_legend: bool | None = None,
    line_width: float = 2.0,
    firm_ids: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    condition: str | None = None,
    return_info: Literal[False] = False,
    show: Literal[False],
    return_df: Literal[True],
) -> tuple[go.Figure, pd.DataFrame]: ...


@overload
def plot_agent_timeseries(
    model,
    country_code: str,
    agent_type: str,
    variables,
    *,
    agent_id: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    agg: str = "sum",
    by_sector: bool = False,
    panel_titles: list[str] | None = None,
    no_cols: int | None = None,
    base_height: int = 220,
    base_width: int = 380,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    shared_xaxes: bool = True,
    show_legend: bool | None = None,
    line_width: float = 2.0,
    firm_ids: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    condition: str | None = None,
    return_info: bool = False,
    show: bool = True,
    return_df: bool = False,
) -> (
    go.Figure | tuple[go.Figure, pd.DataFrame] | tuple[None, pd.DataFrame] | tuple[go.Figure, dict] | tuple[None, dict]
): ...


def plot_agent_timeseries(
    model,
    country_code: str,
    agent_type: str,
    variables,
    *,
    agent_id: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    agg: str = "sum",
    by_sector: bool = False,
    panel_titles: list[str] | None = None,
    no_cols: int | None = None,
    base_height: int = 220,
    base_width: int = 380,
    title: str | None = None,
    height: int | None = None,
    width: int | None = None,
    shared_xaxes: bool = True,
    show_legend: bool | None = None,
    line_width: float = 2.0,
    firm_ids: int | list[int] | tuple[int, ...] | np.ndarray | None = None,
    condition: str | None = None,
    return_info: bool = False,
    show: bool = True,
    return_df: bool = False,
):
    """Plot one or more time-series from an agent's ``.ts`` container.

    Examples
    --------
    Plot scalar series:
        plot_agent_timeseries(model, "FRA", "firms", ["total_sales"])

    Plot firm-level credit demand aggregated across firms:
        plot_agent_timeseries(model, "FRA", "firms", ["target_short_term_credit"], agg="sum")

    Plot firm-level credit demand for a specific firm:
        plot_agent_timeseries(model, "FRA", "firms", ["target_short_term_credit"], agent_id=0)

    Plot firm-level series aggregated by sector (industry):
        plot_agent_timeseries(model, "FRA", "firms", ["tfp_multiplier", "deposits"], by_sector=True, agg="mean")

    Plot multiple series per panel (supports cross-agent overlays via "agent.var" strings):
        plot_agent_timeseries(
            model,
            "FRA",
            "banks",
            [
                ["average_overdraft_rate_on_household_deposits", "central_bank.policy_rate"],
                [
                    "average_overdraft_rate_on_firm_deposits",
                    "average_interest_rates_on_short_term_firm_loans",
                    "central_bank.policy_rate",
                ],
            ],
            panel_titles=["France: household rates", "France: firm rates"],
            no_cols=2,
        )

    Set ``return_df=True`` to also return a DataFrame with the plotted series.
    """
    if isinstance(variables, str):
        variables = [variables]
    variables = list(variables)
    if not variables:
        raise ValueError("variables must be a non-empty list of ts field names.")

    country = model.countries[country_code]
    agent_type = str(agent_type).strip().lower()
    agent = getattr(country, agent_type, None)
    if agent is None:
        raise ValueError(f"Unknown agent_type={agent_type!r} for country {country_code!r}.")
    ts = getattr(agent, "ts", None)
    if ts is None:
        raise ValueError(f"{agent_type}.ts is missing for country {country_code!r}.")

    out_index = model.shallow_df_dict()[country_code].index

    info: dict | None = {} if return_info else None
    if firm_ids is not None or condition is not None:
        if agent_type != "firms":
            raise ValueError("firm_ids/condition are only supported for agent_type='firms'.")

        balance_sheet_df, transaction_account_df = split_firm_ts_balance_sheet_and_transaction_account_df(
            model,
            country_code,
        )

        def _to_matrix(df: pd.DataFrame, var_name: str) -> np.ndarray:
            if var_name not in df.columns:
                raise KeyError(f"{var_name!r} not found in dataframe columns.")
            return np.asarray(list(df[var_name].to_dict().values()))

        def _reference_n_firms() -> int:
            for df in (balance_sheet_df, transaction_account_df):
                for col in df.columns:
                    values = _to_matrix(df, col)
                    if values.ndim == 2:
                        return values.shape[1]
            raise ValueError("Could not infer the number of firms from the provided dataframes.")

        def _normalize_firm_ids(ids):
            if ids is None:
                return None
            if isinstance(ids, (int, np.integer)):
                ids = [int(ids)]
            ids = [int(i) for i in ids]
            n_firms = _reference_n_firms()
            for firm_id in ids:
                if firm_id < 0 or firm_id >= n_firms:
                    raise IndexError(f"firm_id out of bounds: {firm_id} (n_firms={n_firms})")
            return ids

        def _condition_mask(expr: str) -> np.ndarray:
            match = re.fullmatch(
                r"\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op><=|>=|==|!=|<|>)\s*(?P<value>.+?)\s*",
                expr,
            )
            if match is None:
                raise ValueError(
                    "condition must be an expression like 'equity < 0', 'deposits < 0', or 'production == 0'."
                )

            var_name = match.group("var")
            op = match.group("op")
            try:
                value = literal_eval(match.group("value"))
            except Exception as exc:  # pragma: no cover - defensive parser guard
                raise ValueError(f"Could not parse condition value in {expr!r}.") from exc

            if var_name in balance_sheet_df.columns:
                values = _to_matrix(balance_sheet_df, var_name)
            elif var_name in transaction_account_df.columns:
                values = _to_matrix(transaction_account_df, var_name)
            else:
                raise KeyError(f"{var_name!r} not found in either balance_sheet_df or transaction_account_df.")

            op_map = {
                "<": np.less,
                "<=": np.less_equal,
                ">": np.greater,
                ">=": np.greater_equal,
                "==": np.equal,
                "!=": np.not_equal,
            }
            return np.any(op_map[op](values, value), axis=0)

        n_firms = _reference_n_firms()
        selected_mask = np.ones(n_firms, dtype=bool)

        normalized_firm_ids = _normalize_firm_ids(firm_ids)
        if normalized_firm_ids is not None:
            firm_mask = np.zeros(n_firms, dtype=bool)
            firm_mask[normalized_firm_ids] = True
            selected_mask &= firm_mask

        if condition is not None:
            selected_mask &= _condition_mask(condition)

        selected_firms = np.flatnonzero(selected_mask).tolist()
        if not selected_firms:
            raise ValueError("No firms matched the requested firm_ids/condition filters.")

        agent_ids = selected_firms
        info = {
            "balance_sheet_columns": list(balance_sheet_df.columns),
            "transaction_account_columns": list(transaction_account_df.columns),
            "selected_firms": selected_firms,
            "first_negative_at": {},
        }
        if "equity" in balance_sheet_df.columns:
            equity_values = _to_matrix(balance_sheet_df, "equity")
            for firm_idx in selected_firms:
                neg_mask = equity_values[:, firm_idx] < 0
                if np.any(neg_mask):
                    info["first_negative_at"][firm_idx] = out_index[int(np.argmax(neg_mask))]

    panel_mode = any(isinstance(v, (list, tuple)) for v in variables)
    panels: list[list[str]] | None
    if panel_mode:
        if by_sector:
            raise ValueError("by_sector=True is not supported when variables are provided as panels.")
        panels = []
        for entry in variables:
            if isinstance(entry, (list, tuple)):
                panel = [str(x) for x in entry]
            else:
                panel = [str(entry)]
            if not panel:
                raise ValueError("Panel variables must be non-empty.")
            panels.append(panel)
        if panel_titles is not None and len(panel_titles) != len(panels):
            raise ValueError("panel_titles length must match the number of panels.")
    else:
        panels = None

    if by_sector:
        if agent_type != "firms":
            raise ValueError("by_sector=True is only supported for agent_type='firms'.")
        if agent_id is not None:
            raise ValueError("by_sector=True is incompatible with agent_id. Omit agent_id to plot sector aggregates.")
        firm_industry_idx = np.asarray(getattr(agent, "states", {}).get("Industry", []), dtype=int).reshape(-1)
        sector_codes = list(getattr(agent, "industries", []))
        n_sectors = int(getattr(agent, "n_industries", len(sector_codes)))
        if len(sector_codes) != n_sectors:
            sector_codes = [f"sector {idx}" for idx in range(n_sectors)]

        palette = _categorical_colors(n_sectors)
        sector_color_map = {sector_codes[idx]: palette[idx] for idx in range(n_sectors)}
    else:
        firm_industry_idx = None
        sector_codes = None
        n_sectors = None
        sector_color_map = None

    agent_ids: list[int] | None
    if agent_id is None:
        agent_ids = None
    elif isinstance(agent_id, (list, tuple, np.ndarray)):
        agent_ids = [int(x) for x in list(agent_id)]
        if not agent_ids:
            agent_ids = None
    else:
        agent_ids = [int(agent_id)]

    agent_color_map: dict[int, str] | None = None
    if agent_ids is not None and len(agent_ids) > 1:
        # Keep the provided ordering so a user can pass [5, 0, 2] and consistently
        # get the same colors across variables/subplots.
        palette = _categorical_colors(len(agent_ids))
        agent_color_map = {selected_id: palette[idx] for idx, selected_id in enumerate(agent_ids)}

    def _reduce(value):
        value = unpack_cell(value)
        if isinstance(value, (float, int, np.floating, np.integer)):
            return float(value)
        if value is None:
            return np.nan

        array = np.asarray(value, dtype=float).reshape(-1)
        if array.size == 0:
            return np.nan

        if agent_ids is not None and len(agent_ids) == 1:
            selected_id = agent_ids[0]
            if selected_id < 0 or selected_id >= array.size:
                return np.nan
            return float(array[selected_id])

        reducer = agg.lower()
        if reducer == "sum":
            return float(np.nansum(array))
        if reducer == "mean":
            return float(np.nanmean(array))
        if reducer == "median":
            return float(np.nanmedian(array))
        raise ValueError("agg must be one of {'sum', 'mean', 'median'}.")

    def _resolve_series(spec: str):
        spec = str(spec)
        if "." in spec:
            agent_key, var_name = spec.split(".", 1)
            agent_key = agent_key.strip().lower()
            var_name = var_name.strip()
        else:
            agent_key = agent_type
            var_name = spec.strip()

        agent_obj = getattr(country, agent_key, None)
        if agent_obj is None:
            raise ValueError(f"Unknown agent {agent_key!r} for country {country_code!r}.")
        ts_obj = getattr(agent_obj, "ts", None)
        if ts_obj is None:
            raise ValueError(f"{agent_key}.ts is missing for country {country_code!r}.")
        values_obj = _safe_ts_values(ts_obj, var_name)
        if values_obj is None:
            raise ValueError(f"{agent_key}.ts has no field {var_name!r}.")
        return agent_key, var_name, values_obj

    def _legend_label(spec: str) -> str:
        spec = str(spec).strip()
        return spec.split(".", 1)[1] if "." in spec else spec

    def _legend_swatch(color: str | None) -> str:
        if color is None:
            return "•"
        return f'<span style="color:{color};">■</span>'

    def _panel_title(panel: list[str]) -> str:
        return " + ".join(_legend_label(spec) for spec in panel)

    n_subplots = len(panels) if panel_mode and panels is not None else len(variables)
    if no_cols is None:
        no_cols = 1 if n_subplots == 1 else 3
    if no_cols <= 0:
        raise ValueError("no_cols must be a positive integer.")
    no_rows = int(np.ceil(n_subplots / no_cols))

    if panel_mode and panels is not None:
        base_titles = list(panel_titles) if panel_titles is not None else [_panel_title(panel) for panel in panels]
    else:
        base_titles = [_legend_label(str(v)) for v in variables]
    subplot_titles = base_titles + [""] * (no_rows * no_cols - len(base_titles))

    if show_legend is None:
        show_legend = (agent_ids is not None and len(agent_ids) > 1) or by_sector or panel_mode

    plotted_series: list[tuple[str, pd.Series]] = []

    def _store_series(name: str, values: list[float] | np.ndarray | pd.Series) -> None:
        series = pd.Series(values, index=out_index[: len(values)], dtype=float).reindex(out_index)
        if name in {existing_name for existing_name, _ in plotted_series}:
            suffix = 2
            candidate = f"{name}__{suffix}"
            existing_names = {existing_name for existing_name, _ in plotted_series}
            while candidate in existing_names:
                suffix += 1
                candidate = f"{name}__{suffix}"
            name = candidate
        plotted_series.append((name, series))

    fig = make_subplots(
        rows=no_rows,
        cols=no_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=shared_xaxes,
        horizontal_spacing=0.06,
        vertical_spacing=0.09 if no_rows <= 3 else 0.05,
    )

    if panel_mode and panels is not None:
        # Stable color per (agent,var) across panels/subplots.
        unique_keys: list[str] = []
        seen: set[str] = set()
        for panel in panels:
            for spec in panel:
                agent_key, var_name, _ = _resolve_series(spec)
                key = f"{agent_key}.{var_name}"
                if key not in seen:
                    seen.add(key)
                    unique_keys.append(key)
        palette = _categorical_colors(len(unique_keys))
        series_color_map = {key: palette[idx] for idx, key in enumerate(unique_keys)}

        for idx, panel in enumerate(panels):
            row = (idx // no_cols) + 1
            col = (idx % no_cols) + 1
            for spec in panel:
                agent_key, var_name, values = _resolve_series(spec)
                key = f"{agent_key}.{var_name}"
                horizon = min(len(out_index), len(values))

                sample = None
                for value in list(values)[:horizon]:
                    unpacked = unpack_cell(value)
                    if unpacked is not None:
                        sample = unpacked
                        break
                is_vector = not isinstance(sample, (type(None), float, int, np.floating, np.integer))
                if is_vector and agent_ids is not None and len(agent_ids) > 1:
                    raise ValueError(
                        "Panel mode does not support plotting multiple agent_id lines for vector series. "
                        f"Got agent_id={agent_ids} for {key!r}. "
                        "Either omit agent_id to aggregate vectors via agg, pass a single agent_id, "
                        "or use non-panel mode."
                    )

                series = [_reduce(v) for v in list(values)[:horizon]]
                _store_series(f"panel{idx + 1}:{key}", series)
                fig.add_trace(
                    go.Scatter(
                        x=out_index[:horizon],
                        y=series,
                        mode="lines",
                        name=_legend_label(key),
                        showlegend=False,
                        line={"width": line_width, "color": series_color_map.get(key)},
                    ),
                    row=row,
                    col=col,
                )
    else:
        for idx, var in enumerate(variables):
            agent_key, var_name, values = _resolve_series(var)
            horizon = min(len(out_index), len(values))
            row = (idx // no_cols) + 1
            col = (idx % no_cols) + 1

            sample = None
            for value in list(values)[:horizon]:
                unpacked = unpack_cell(value)
                if unpacked is not None:
                    sample = unpacked
                    break
            is_vector = not isinstance(sample, (type(None), float, int, np.floating, np.integer))

            if by_sector and is_vector:

                def _sector_reduce(vec: np.ndarray) -> np.ndarray:
                    vec = np.asarray(vec, dtype=float).reshape(-1)
                    if firm_industry_idx is None or n_sectors is None:
                        return np.full(0, np.nan, dtype=float)
                    if firm_industry_idx.size != vec.size:
                        # Fall back to NaNs if we cannot align the firm vector to industries.
                        return np.full(n_sectors, np.nan, dtype=float)
                    reducer = agg.lower()
                    if reducer == "median":
                        out = np.full(n_sectors, np.nan, dtype=float)
                        for sector in range(n_sectors):
                            vals = vec[firm_industry_idx == sector]
                            if vals.size:
                                out[sector] = float(np.nanmedian(vals))
                        return out

                    ok = np.isfinite(vec)
                    sums = np.bincount(firm_industry_idx[ok], weights=vec[ok], minlength=n_sectors).astype(float)
                    cnts = np.bincount(firm_industry_idx[ok], minlength=n_sectors).astype(float)
                    if reducer == "sum":
                        return sums
                    if reducer == "mean":
                        return np.divide(sums, cnts, out=np.full(n_sectors, np.nan, dtype=float), where=cnts > 0.0)
                    raise ValueError("agg must be one of {'sum', 'mean', 'median'}.")

                series_by_sector: dict[str, list[float]] = {str(code): [] for code in sector_codes}
                for value in list(values)[:horizon]:
                    unpacked = unpack_cell(value)
                    vec = (
                        np.asarray(unpacked, dtype=float).reshape(-1) if unpacked is not None else np.asarray([], float)
                    )
                    reduced = _sector_reduce(vec)
                    for sector_idx, code in enumerate(sector_codes):
                        y = float(reduced[sector_idx]) if sector_idx < reduced.size else np.nan
                        series_by_sector[str(code)].append(y)

                for sector_idx, code in enumerate(sector_codes):
                    code = str(code)
                    _store_series(f"{agent_key}.{var_name}[{code}]", series_by_sector[code])
                    fig.add_trace(
                        go.Scatter(
                            x=out_index[:horizon],
                            y=series_by_sector[code],
                            mode="lines",
                            name=code,
                            showlegend=(show_legend and idx == 0),
                            line={
                                "width": line_width,
                                "color": sector_color_map.get(code) if sector_color_map is not None else None,
                            },
                        ),
                        row=row,
                        col=col,
                    )
            elif is_vector and agent_ids is not None and len(agent_ids) > 1:
                series_by_id: dict[int, list[float]] = {selected_id: [] for selected_id in agent_ids}
                for value in list(values)[:horizon]:
                    unpacked = unpack_cell(value)
                    array = (
                        np.asarray(unpacked, dtype=float).reshape(-1)
                        if unpacked is not None
                        else np.asarray([], dtype=float)
                    )
                    for selected_id in agent_ids:
                        if selected_id < 0 or selected_id >= array.size:
                            series_by_id[selected_id].append(np.nan)
                        else:
                            series_by_id[selected_id].append(float(array[selected_id]))

                for j, selected_id in enumerate(agent_ids):
                    _store_series(f"{agent_key}.{var_name}[id={selected_id}]", series_by_id[selected_id])
                    fig.add_trace(
                        go.Scatter(
                            x=out_index[:horizon],
                            y=series_by_id[selected_id],
                            mode="lines",
                            name=f"id={selected_id}",
                            showlegend=(show_legend and idx == 0),
                            line={
                                "width": line_width,
                                "color": agent_color_map.get(selected_id) if agent_color_map is not None else None,
                            },
                        ),
                        row=row,
                        col=col,
                    )
            else:
                series = [_reduce(v) for v in list(values)[:horizon]]
                _store_series(f"{agent_key}.{var_name}", series)
                fig.add_trace(
                    go.Scatter(
                        x=out_index[:horizon],
                        y=series,
                        mode="lines",
                        name=_legend_label(f"{agent_key}.{var_name}"),
                        showlegend=show_legend,
                        line={"width": line_width},
                    ),
                    row=row,
                    col=col,
                )

    if title is None:
        if agent_ids is not None and len(agent_ids) > 1:
            suffix = f" (agent_id={agent_ids})"
        elif agent_ids is not None and len(agent_ids) == 1:
            suffix = f" (agent_id={agent_ids[0]})"
        elif by_sector:
            suffix = f" (by_sector, agg={agg})"
        else:
            suffix = f" (agg={agg})"
        title = f"{country_code} {agent_type} time series{suffix}"

    fig.update_layout(
        height=(base_height * no_rows if height is None else height),
        width=(base_width * no_cols if width is None else width),
        title_text=title,
        template="plotly_white",
        hovermode="x unified" if shared_xaxes else "closest",
        legend_title_text="Series" if show_legend else None,
        legend={"orientation": "v", "yanchor": "top", "y": 1.0, "xanchor": "left", "x": 1.02}
        if show_legend and not panel_mode
        else None,
        margin={"t": 90, "r": 280} if show_legend else None,
    )
    if panel_mode and show_legend:
        for idx, panel in enumerate(panels or []):
            row = (idx // no_cols) + 1
            col = (idx % no_cols) + 1
            subplot = fig.get_subplot(row, col)
            labels = []
            seen_panel_keys: set[str] = set()
            for spec in panel:
                agent_key, var_name, _ = _resolve_series(spec)
                key = f"{agent_key}.{var_name}"
                if key in seen_panel_keys:
                    continue
                seen_panel_keys.add(key)
                labels.append(f"{_legend_swatch(series_color_map.get(key))} {_legend_label(var_name)}")
            if not labels:
                continue
            legend_text = "<b>Series</b><br>" + "<br>".join(labels)
            fig.add_annotation(
                x=min(float(subplot.xaxis.domain[1]) + 0.015, 0.995),
                y=float(subplot.yaxis.domain[1]),
                xref="paper",
                yref="paper",
                xanchor="left",
                yanchor="top",
                showarrow=False,
                text=legend_text,
                font={"size": 11},
                align="left",
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
            )
    for col in range(1, no_cols + 1):
        fig.update_xaxes(title_text="t", row=no_rows, col=col)

    if show:
        fig.show()
        if return_df:
            return None, pd.DataFrame({name: series for name, series in plotted_series}, index=out_index)
        return None

    if return_df:
        result_df = pd.DataFrame({name: series for name, series in plotted_series}, index=out_index)
        if info is not None:
            if show:
                return None, result_df
            return fig, result_df
        if show:
            return None, result_df
        return fig, result_df
    if info is not None:
        if show:
            return None, info
        return fig, info
    return fig


def build_credit_rationing_aggregates_df(model, country_code: str) -> pd.DataFrame:
    """Return aggregate credit rationing measures for firms and banks.

    Firms measure compares total credit demand vs total credit received:
    - firms.ts.total_target_short_term_credit / total_received_short_term_credit
    - firms.ts.total_target_long_term_credit / total_received_long_term_credit

    Firm short-term demand is the total borrower-facing request. In current firm
    finance runs this may include emergency overdraft-refinance demand as well as
    ordinary working-capital demand. Separate firm time series
    (`total_target_overdraft_refinance_credit` and
    `total_ordinary_target_short_term_credit`) expose that decomposition when the
    model writes it.

    Banks measure compares total credit supply caps vs respective total credit demands:
    - banks.ts.total_credit_supply_cap_firms_short_term vs firms.ts.total_target_short_term_credit
    - banks.ts.total_credit_supply_cap_firms_long_term vs firms.ts.total_target_long_term_credit
    - banks.ts.total_credit_supply_cap_households_consumption vs households.ts.total_target_consumption_loans
    - banks.ts.total_credit_supply_cap_mortgages vs households.ts.total_target_mortgage

    The firm short-term supply cap is split from total firm lending capacity using
    ordinary short-term demand, not emergency overdraft-refinance demand. This
    means total ST demand can exceed the plotted ST cap when refinance is large;
    use the separate refinance/ordinary ST series to interpret that case.

    Returned columns include both levels (demand/received/cap) and derived measures:
    - rationed_amount = max(demand - received_or_cap, 0)
    - rationing_rate = rationed_amount / demand, with 0 when demand <= 0
    """
    out_index = model.shallow_df_dict()[country_code].index
    country = model.countries[country_code]

    firms_ts = getattr(getattr(country, "firms", None), "ts", None)
    households_ts = getattr(getattr(country, "households", None), "ts", None)
    banks_ts = getattr(getattr(country, "banks", None), "ts", None)

    def _scalar_series(ts, name: str) -> pd.Series | None:
        values = _safe_ts_values(ts, name)
        if values is None:
            return None
        series = pd.Series(list(values)).reindex(range(len(out_index)))
        series.index = out_index
        return pd.to_numeric(series.map(unpack_cell), errors="coerce")

    def _rationed_amount(demand: pd.Series | None, supplied: pd.Series | None) -> pd.Series | None:
        if demand is None or supplied is None:
            return None
        return (demand - supplied).clip(lower=0.0)

    def _rationing_rate(demand: pd.Series | None, supplied: pd.Series | None) -> pd.Series | None:
        if demand is None or supplied is None:
            return None
        rationed = (demand - supplied).clip(lower=0.0)
        rate = pd.Series(0.0, index=demand.index, dtype=float)
        mask = demand > 0
        rate[mask] = (rationed[mask] / demand[mask]).astype(float)
        return rate.clip(lower=0.0, upper=1.0)

    firm_demand_st = _scalar_series(firms_ts, "total_target_short_term_credit")
    firm_demand_lt = _scalar_series(firms_ts, "total_target_long_term_credit")
    firm_received_st = _scalar_series(firms_ts, "total_received_short_term_credit")
    firm_received_lt = _scalar_series(firms_ts, "total_received_long_term_credit")

    hh_demand_cons = _scalar_series(households_ts, "total_target_consumption_loans")
    hh_demand_mort = _scalar_series(households_ts, "total_target_mortgage")

    bank_cap_firm_st = _scalar_series(banks_ts, "total_credit_supply_cap_firms_short_term")
    bank_cap_firm_lt = _scalar_series(banks_ts, "total_credit_supply_cap_firms_long_term")
    bank_cap_hh_cons = _scalar_series(banks_ts, "total_credit_supply_cap_households_consumption")
    bank_cap_mort = _scalar_series(banks_ts, "total_credit_supply_cap_mortgages")

    out = pd.DataFrame(
        {
            "firm_credit_demand_short_term": firm_demand_st,
            "firm_credit_received_short_term": firm_received_st,
            "firm_credit_rationed_short_term": _rationed_amount(firm_demand_st, firm_received_st),
            "firm_credit_rationing_rate_short_term": _rationing_rate(firm_demand_st, firm_received_st),
            "firm_credit_demand_long_term": firm_demand_lt,
            "firm_credit_received_long_term": firm_received_lt,
            "firm_credit_rationed_long_term": _rationed_amount(firm_demand_lt, firm_received_lt),
            "firm_credit_rationing_rate_long_term": _rationing_rate(firm_demand_lt, firm_received_lt),
            "household_credit_demand_consumption": hh_demand_cons,
            "household_credit_demand_mortgage": hh_demand_mort,
            "bank_credit_supply_cap_firms_short_term": bank_cap_firm_st,
            "bank_credit_supply_gap_firms_short_term": _rationed_amount(firm_demand_st, bank_cap_firm_st),
            "bank_credit_rationing_rate_firms_short_term": _rationing_rate(firm_demand_st, bank_cap_firm_st),
            "bank_credit_supply_cap_firms_long_term": bank_cap_firm_lt,
            "bank_credit_supply_gap_firms_long_term": _rationed_amount(firm_demand_lt, bank_cap_firm_lt),
            "bank_credit_rationing_rate_firms_long_term": _rationing_rate(firm_demand_lt, bank_cap_firm_lt),
            "bank_credit_supply_cap_households_consumption": bank_cap_hh_cons,
            "bank_credit_supply_gap_households_consumption": _rationed_amount(hh_demand_cons, bank_cap_hh_cons),
            "bank_credit_rationing_rate_households_consumption": _rationing_rate(hh_demand_cons, bank_cap_hh_cons),
            "bank_credit_supply_cap_mortgages": bank_cap_mort,
            "bank_credit_supply_gap_mortgages": _rationed_amount(hh_demand_mort, bank_cap_mort),
            "bank_credit_rationing_rate_mortgages": _rationing_rate(hh_demand_mort, bank_cap_mort),
        },
        index=out_index,
    )
    out.index.name = getattr(out_index, "name", None) or "t"
    return out


def plot_credit_rationing(
    model,
    country_code: str,
    *,
    title: str | None = None,
    height: int = 650,
    width: int = 1050,
    show: bool = True,
):
    """Plot firm and bank credit rationing rates over time (0–1)."""
    df = build_credit_rationing_aggregates_df(model, country_code)

    if title is None:
        title = f"Credit rationing ({country_code})"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Firms: rationing rate (received vs demand)", "Banks: rationing rate (supply cap vs demand)"),
    )

    firm_traces = [
        ("Short-term", "firm_credit_rationing_rate_short_term"),
        ("Long-term", "firm_credit_rationing_rate_long_term"),
    ]
    for name, col in firm_traces:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=f"Firms {name}"), row=1, col=1)

    bank_traces = [
        ("Firms short-term", "bank_credit_rationing_rate_firms_short_term"),
        ("Firms long-term", "bank_credit_rationing_rate_firms_long_term"),
        ("HH consumption", "bank_credit_rationing_rate_households_consumption"),
        ("Mortgages", "bank_credit_rationing_rate_mortgages"),
    ]
    for name, col in bank_traces:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode="lines", name=f"Banks {name}"), row=2, col=1)

    fig.add_hline(y=0.0, line_width=1, line_color="black", row=1, col=1)
    fig.add_hline(y=0.0, line_width=1, line_color="black", row=2, col=1)
    fig.update_yaxes(title_text="rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title_text="rate", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="t", row=2, col=1)
    fig.update_layout(height=height, width=width, title_text=title, template="plotly_white")

    if show:
        fig.show()
        return None
    return fig


def plot_firm_credit_to_equity_and_capital(
    model,
    country_code: str = "FRA",
    *,
    title: str | None = None,
    height: int = 650,
    width: int = 1050,
    show: bool = True,
    return_df: bool = False,
):
    """Plot firm outstanding credit relative to firm equity and capital (aggregate).

    All firm series are summed across firms each period:
    - firms.ts.capital_inputs_stock_value (total firm capital stock value)
    - firms.ts.equity (total firm equity)

    Outstanding credit is read from the credit market aggregates:
    - credit_market.ts.total_outstanding_loans_granted_firms_short_term
    - credit_market.ts.total_outstanding_loans_granted_firms_long_term
    """
    country = model.countries[country_code]
    out_index = model.shallow_df_dict()[country_code].index

    firms_ts = getattr(getattr(country, "firms", None), "ts", None)
    credit_market_ts = getattr(getattr(country, "credit_market", None), "ts", None)
    if firms_ts is None:
        raise ValueError(f"firms.ts is missing for country {country_code!r}.")
    if credit_market_ts is None:
        raise ValueError(f"credit_market.ts is missing for country {country_code!r}.")

    def _sum_over_firms(values) -> pd.Series:
        values = list(values)
        horizon = min(len(out_index), len(values))
        rows: list[float] = []
        for value in values[:horizon]:
            unpacked = unpack_cell(value)
            if unpacked is None:
                rows.append(np.nan)
                continue
            if isinstance(unpacked, (float, int, np.floating, np.integer)):
                rows.append(float(unpacked))
                continue
            array = np.asarray(unpacked, dtype=float).reshape(-1)
            rows.append(float(np.nansum(array)) if array.size else np.nan)
        series = pd.Series(rows, index=out_index[:horizon], dtype=float)
        return series.reindex(out_index)

    def _scalar_series(values) -> pd.Series:
        values = list(values)
        horizon = min(len(out_index), len(values))
        rows = [unpack_cell(value) for value in values[:horizon]]
        series = pd.Series(rows, index=out_index[:horizon])
        return pd.to_numeric(series.reindex(out_index), errors="coerce")

    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        ratio = pd.Series(np.nan, index=numerator.index, dtype=float)
        mask = denominator > 0
        ratio[mask] = (numerator[mask] / denominator[mask]).astype(float)
        return ratio

    firm_capital_stock_value = _safe_ts_values(firms_ts, "capital_inputs_stock_value")
    firm_equity = _safe_ts_values(firms_ts, "equity")
    if firm_capital_stock_value is None:
        raise ValueError(f"firms.ts has no field {'capital_inputs_stock_value'!r}.")
    if firm_equity is None:
        raise ValueError(f"firms.ts has no field {'equity'!r}.")

    outstanding_credit_short_term = _safe_ts_values(
        credit_market_ts, "total_outstanding_loans_granted_firms_short_term"
    )
    outstanding_credit_long_term = _safe_ts_values(credit_market_ts, "total_outstanding_loans_granted_firms_long_term")
    if outstanding_credit_short_term is None:
        raise ValueError("credit_market.ts has no field 'total_outstanding_loans_granted_firms_short_term'.")
    if outstanding_credit_long_term is None:
        raise ValueError("credit_market.ts has no field 'total_outstanding_loans_granted_firms_long_term'.")

    total_firm_capital_stock_value = _sum_over_firms(firm_capital_stock_value)
    total_firm_equity = _sum_over_firms(firm_equity)
    total_firm_outstanding_credit_short_term = _scalar_series(outstanding_credit_short_term)
    total_firm_outstanding_credit_long_term = _scalar_series(outstanding_credit_long_term)

    df = pd.DataFrame(
        {
            "total_firm_capital_stock_value": total_firm_capital_stock_value,
            "total_firm_equity": total_firm_equity,
            "total_firm_outstanding_credit_short_term": total_firm_outstanding_credit_short_term,
            "total_firm_outstanding_credit_long_term": total_firm_outstanding_credit_long_term,
        },
        index=out_index,
    )
    df["firm_outstanding_credit_short_term_to_equity"] = _safe_ratio(
        df["total_firm_outstanding_credit_short_term"], df["total_firm_equity"]
    )
    df["firm_outstanding_credit_long_term_to_equity"] = _safe_ratio(
        df["total_firm_outstanding_credit_long_term"], df["total_firm_equity"]
    )
    df["firm_outstanding_credit_short_term_to_capital"] = _safe_ratio(
        df["total_firm_outstanding_credit_short_term"], df["total_firm_capital_stock_value"]
    )
    df["firm_outstanding_credit_long_term_to_capital"] = _safe_ratio(
        df["total_firm_outstanding_credit_long_term"], df["total_firm_capital_stock_value"]
    )

    if title is None:
        title = f"Firm credit ratios ({country_code})"

    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        subplot_titles=[
            "Outstanding short-term credit / equity",
            "Outstanding long-term credit / equity",
            "Outstanding short-term credit / capital stock value",
            "Outstanding long-term credit / capital stock value",
        ],
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["firm_outstanding_credit_short_term_to_equity"],
            mode="lines",
            name="ST / equity",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["firm_outstanding_credit_long_term_to_equity"],
            mode="lines",
            name="LT / equity",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["firm_outstanding_credit_short_term_to_capital"],
            mode="lines",
            name="ST / capital",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["firm_outstanding_credit_long_term_to_capital"],
            mode="lines",
            name="LT / capital",
        ),
        row=2,
        col=2,
    )

    fig.update_yaxes(title_text="ratio", row=1, col=1)
    fig.update_yaxes(title_text="ratio", row=1, col=2)
    fig.update_yaxes(title_text="ratio", row=2, col=1)
    fig.update_yaxes(title_text="ratio", row=2, col=2)
    fig.update_xaxes(title_text="t", row=2, col=1)
    fig.update_xaxes(title_text="t", row=2, col=2)
    fig.update_layout(
        height=height,
        width=width,
        title_text=title,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )

    if show:
        fig.show()
        if return_df:
            return None, df
        return None

    if return_df:
        return fig, df
    return fig


def plot_housing_market_aggregates(
    model=None,
    country_code=None,
    housing_market_ts=None,
    keys=None,
    no_cols=3,
    base_height=220,
    base_width=380,
    title=None,
    line_color="#1f77b4",
    show=True,
):
    """Plot scalar aggregate time series from a housing market time-series store."""
    if housing_market_ts is None:
        if model is None or country_code is None:
            raise ValueError("Provide either housing_market_ts or both model and country_code.")
        housing_market_ts = model.countries[country_code].housing_market.ts

    keys = list(housing_market_ts.get_keys() if keys is None else keys)
    series = {}
    for key in keys:
        values = np.asarray(housing_market_ts.get_aggregate(key)).squeeze()
        if values.ndim > 1:
            continue
        series[key] = pd.Series(values)

    if not series:
        raise ValueError("No scalar housing market aggregate series found to plot.")

    df = pd.DataFrame(series)
    no_rows = int(np.ceil(len(df.columns) / no_cols))
    figure_title = title or (
        f"{country_code} housing market aggregate time series"
        if country_code is not None
        else "Housing market aggregate time series"
    )
    fig = make_subplots(rows=no_rows, cols=no_cols, subplot_titles=df.columns.to_list())

    for idx, col_name in enumerate(df.columns):
        row = (idx // no_cols) + 1
        col = (idx % no_cols) + 1
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col_name],
                mode="lines",
                name=col_name,
                showlegend=False,
                line={"color": line_color},
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=base_height * no_rows,
        width=base_width * no_cols,
        title_text=figure_title,
        template="plotly_white",
    )
    if show:
        fig.show()
        return None
    return fig


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
    no_cols:
        Optional number of subplot columns. Rows are computed automatically
        from the number of series to plot.
    no_rows:
        Deprecated. Kept for backward compatibility and ignored.
    country_code:
        Optional title override.
    """
    combined = mc.combined if hasattr(mc, "combined") else mc
    if not isinstance(combined, pd.DataFrame):
        raise TypeError("mc must be a MonteCarloResult or a pandas DataFrame.")
    if combined.index.nlevels < 2:
        raise ValueError("mc dataframe must be indexed by seed and simulation time.")

    missing_cols = [col for col in cols if col not in combined.columns]
    if missing_cols:
        warnings.warn(
            "plot_mc skipped requested columns not found in Monte Carlo output: " + ", ".join(map(str, missing_cols)),
            stacklevel=2,
        )
    available_cols = [col for col in cols if col in combined.columns]
    if not available_cols:
        raise ValueError("None of the requested cols are present in the Monte Carlo output.")

    n_plots = len(available_cols)
    if no_cols is None:
        no_cols = int(np.ceil(np.sqrt(n_plots)))
    else:
        no_cols = int(no_cols)
        if no_cols <= 0:
            raise ValueError("no_cols must be a positive integer.")
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

        # Overlay MC mean line for this subplot
        mean_series = combined.groupby(level=1)[col_name].mean()
        fig.add_trace(
            go.Scatter(
                x=mean_series.index,
                y=mean_series.values,
                mode="lines",
                name="MC mean",
                showlegend=(idx == 0),
                line={"color": "black", "width": 2},
                legendgroup="mc-mean",
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
