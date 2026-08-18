"""Household income diagnostics shared by run-model notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd


def permanent_income_by_decile(
    households,
    *,
    scale: float = 5_000,
    period: int = 1,
    periods_per_year: int = 4,
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """Summarize current and permanent income by current-income quantile.

    Income is returned in thousands per year, adjusted for the model scale.
    Ranking before ``qcut`` makes tied incomes deterministic.
    """
    if scale <= 0:
        raise ValueError("scale must be positive.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2.")

    income_history = households.ts.income
    log_ratio_history = households.ts.target_consumption_permanent_income_log_ratio
    if period < 0 or period >= len(income_history):
        raise IndexError(f"period {period} is outside the income history.")

    income = np.asarray(income_history[period], dtype=float).reshape(-1) / (scale * 1e3) * periods_per_year
    log_ratio = np.asarray(log_ratio_history[period], dtype=float).reshape(-1)
    if income.shape != log_ratio.shape:
        raise ValueError("Current-income and permanent-income log-ratio arrays must have the same shape.")
    if income.size < n_quantiles:
        raise ValueError(f"At least {n_quantiles} households are required for {n_quantiles} quantiles.")

    frame = pd.DataFrame(
        {
            "income": income,
            "log_ratio": log_ratio,
            "permanent_income": income * np.exp(log_ratio),
        }
    )
    frame["quantile"] = pd.qcut(
        frame["income"].rank(method="first"),
        q=n_quantiles,
        labels=range(1, n_quantiles + 1),
    )

    return (
        frame.groupby("quantile", observed=True)
        .agg(
            mean_permanent_to_current=("log_ratio", lambda values: np.exp(values).mean()),
            mean_log_ratio=("log_ratio", "mean"),
            mean_current_income=("income", "mean"),
            mean_permanent_income=("permanent_income", "mean"),
            median_current_income=("income", "median"),
            median_permanent_income=("permanent_income", "median"),
        )
        .reset_index()
        .round(3)
    )
