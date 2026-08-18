"""Firm balance-sheet and investment-finance diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

FIRM_FINANCE_SERIES = (
    "deposits",
    "equity",
    "profits",
    "debt",
    "capital_inputs_stock_value",
    "inventory",
    "price",
    "credit_budget_hard_obligations",
    "credit_budget_working_capital_budget",
    "credit_budget_remaining_internal_finance_after_working_capital",
    "credit_budget_capital_costs",
    "credit_budget_technical_investment_costs",
    "credit_budget_tfp_costs",
    "target_short_term_credit",
    "target_long_term_credit",
    "target_debt_rollover_credit",
    "target_operating_refinance_credit",
)


def _stack_series(ts, name: str) -> np.ndarray:
    if name not in ts.dicts:
        raise KeyError(f"Firm time series {name!r} is required for the finance diagnostic.")
    rows = [np.asarray(value, dtype=float).reshape(-1) for value in ts.dicts[name]]
    if not rows:
        raise ValueError(f"Firm time series {name!r} is empty.")
    width = rows[0].size
    if any(row.size != width for row in rows):
        raise ValueError(f"Firm time series {name!r} changes width across periods.")
    return np.vstack(rows)


def build_firm_balance_sheet_ratios(model, country_code: str) -> pd.DataFrame:
    """Build firm-period liquidity, leverage, return, and funding-gap measures."""
    ts = model.countries[country_code].firms.ts
    values = {name: _stack_series(ts, name) for name in FIRM_FINANCE_SERIES}
    shape = values["deposits"].shape
    mismatched = {name: value.shape for name, value in values.items() if value.shape != shape}
    if mismatched:
        raise ValueError(f"Firm finance time series must share shape {shape}; mismatches: {mismatched}")

    n_periods, n_firms = shape
    frame = pd.DataFrame(
        {
            "period": np.repeat(np.arange(n_periods), n_firms),
            "firm": np.tile(np.arange(n_firms), n_periods),
        }
    )
    for name, value in values.items():
        frame[name] = value.reshape(-1)

    frame["positive_deposits"] = frame["deposits"].clip(lower=0)
    frame["assets_proxy"] = (
        frame["capital_inputs_stock_value"]
        + frame["inventory"] * frame["price"]
        + frame["positive_deposits"]
    )
    residual = frame["credit_budget_remaining_internal_finance_after_working_capital"].clip(lower=0)
    capital_internal = np.minimum(residual, frame["credit_budget_capital_costs"])
    residual_after_capital = residual - capital_internal
    productivity_costs = (
        frame["credit_budget_technical_investment_costs"] + frame["credit_budget_tfp_costs"]
    )
    frame["capital_funding_gap"] = (frame["credit_budget_capital_costs"] - capital_internal).clip(lower=0)
    frame["productivity_funding_gap"] = (
        productivity_costs - np.minimum(residual_after_capital, productivity_costs)
    ).clip(lower=0)
    frame["cash_to_hard_obligations"] = frame["positive_deposits"] / frame[
        "credit_budget_hard_obligations"
    ].replace(0, np.nan)
    frame["internal_capital_coverage"] = capital_internal / frame["credit_budget_capital_costs"].replace(
        0, np.nan
    )
    frame["internal_investment_coverage"] = residual / (
        frame["credit_budget_capital_costs"] + productivity_costs
    ).replace(0, np.nan)
    frame["debt_to_equity"] = frame["debt"] / frame["equity"].replace(0, np.nan)
    frame["roe"] = frame["profits"] / frame["equity"].where(frame["equity"].abs() > 1e-9)
    frame["roa"] = frame["profits"] / frame["assets_proxy"].where(frame["assets_proxy"] > 1e-9)
    frame["cash_to_assets"] = frame["positive_deposits"] / frame["assets_proxy"].where(
        frame["assets_proxy"] > 1e-9
    )
    return frame


def summarize_firm_balance_sheet_ratios(
    frame: pd.DataFrame,
    *,
    periods: Iterable[int] = (1, 3, 6, 11, 21, 50),
    money_scale: float = 1e9,
) -> pd.DataFrame:
    """Aggregate the firm-level finance panel for selected periods."""
    if money_scale <= 0:
        raise ValueError("money_scale must be positive.")
    selected = frame[frame["period"].isin([int(period) for period in periods])]
    summary = selected.groupby("period").agg(
        firms_with_capital_demand=("credit_budget_capital_costs", lambda values: (values > 1e-6).mean()),
        firms_with_capital_gap=("capital_funding_gap", lambda values: (values > 1e-6).mean()),
        firms_with_productivity_gap=("productivity_funding_gap", lambda values: (values > 1e-6).mean()),
        firms_requesting_lt=("target_long_term_credit", lambda values: (values > 1e-6).mean()),
        capital_costs=("credit_budget_capital_costs", "sum"),
        capital_gap=("capital_funding_gap", "sum"),
        productivity_gap=("productivity_funding_gap", "sum"),
        lt_demand=("target_long_term_credit", "sum"),
        median_cash_to_hard=("cash_to_hard_obligations", "median"),
        median_internal_capital_coverage=("internal_capital_coverage", "median"),
        median_roe=("roe", "median"),
        median_roa=("roa", "median"),
        median_cash_to_assets=("cash_to_assets", "median"),
    )
    for column in ("capital_costs", "capital_gap", "productivity_gap", "lt_demand"):
        summary[column] /= money_scale
    return summary.round(3)
