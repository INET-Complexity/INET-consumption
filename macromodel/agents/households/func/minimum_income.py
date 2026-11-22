"""Pure functions of the RSA-style minimum income (PLAN.md sections 3.1, 3.2, 3.4).

All amounts are per household per period, nominal, in model monetary units (population-expanded).
No state and no RNG: the caller assembles the inputs and stores the outputs.
"""

from collections.abc import Sequence

import numpy as np


def _array(value: object, name: str, *, non_negative: bool = False, finite: bool = True) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if finite and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    if non_negative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return array


def _scalar(value: float, name: str, *, positive: bool = True) -> float:
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"{name} must be a finite positive number.")
    return value


def _assessment_rows(history: Sequence[np.ndarray], time_unit: int) -> list[int]:
    n_rows = len(history)
    if n_rows == 0:
        raise ValueError("history must contain at least the opening row.")
    if time_unit == 3:
        return [n_rows - 1]
    if time_unit == 1:
        return list(range(max(n_rows - 4, 0), max(n_rows - 2, 0) + 1))
    raise ValueError("The assessment window is defined for time_unit 1 (monthly) or 3 (quarterly) only.")


def assessment_flow(history: Sequence[np.ndarray], *, time_unit: int) -> np.ndarray:
    """Assessed value of a realised flow for period t (PLAN.md section 3.2).

    ``history`` holds one realised row per period, oldest first; row 0 is the opening value and the last row is
    t-1. Quarterly: the t-1 row. Monthly: the mean of rows t-4..t-2, restricted to the rows that exist (in the
    first periods the window starts at the opening row).
    """
    rows = [_array(history[index], "history row") for index in _assessment_rows(history, time_unit)]
    if len({row.shape for row in rows}) != 1:
        raise ValueError("history rows must have equal shapes.")
    return sum(rows) / len(rows)


def assessment_stock(history: Sequence[np.ndarray], *, time_unit: int) -> np.ndarray:
    """Assessed closing stock for period t: the t-1 row (quarterly) or the t-2 row (monthly), at least row 0."""
    return _array(history[_assessment_rows(history, time_unit)[-1]], "history row")


def notional_rate_period(annual_rate: float, time_unit: float) -> float:
    """Notional asset return per period: ``annual_rate * time_unit / 12`` (``time_unit`` in months)."""
    return _scalar(annual_rate, "annual_rate") * _scalar(time_unit, "time_unit") / 12.0


def minimum_income_level(
    *,
    rsa_base_monthly_model_units: float,
    time_unit: float,
    consumption_units: np.ndarray,
    transfer_index: float,
    cpi_ratio: float,
) -> np.ndarray:
    """Guaranteed level ``M = rsa_base_monthly * time_unit * CU * I_t * (CPI_{t-1} / CPI_0)`` per household.

    ``cpi_ratio`` is the last realised CPI over the initial CPI (PLAN.md section 3.1).

    ``rsa_base_monthly_model_units`` is the RSA base amount already multiplied by the population scale factor.
    """
    units = _array(consumption_units, "consumption_units")
    if np.any(units <= 0.0):
        raise ValueError("consumption_units must be positive.")
    return (
        _scalar(rsa_base_monthly_model_units, "rsa_base_monthly_model_units")
        * _scalar(time_unit, "time_unit")
        * units
        * _scalar(transfer_index, "transfer_index")
        * _scalar(cpi_ratio, "cpi_ratio")
    )


def assessed_income(
    *,
    employee: np.ndarray,
    unemployment: np.ndarray,
    pension: np.ndarray,
    other_transfers: np.ndarray,
    rental: np.ndarray,
    liquid_assets: np.ndarray,
    illiquid_assets: np.ndarray,
    notional_rate_period: float,
) -> np.ndarray:
    """Assessed income ``Y = flows + notional_rate_period * (liquid + illiquid financial assets)``.

    Flows may be negative (for example net rental income); ``Y`` can then be negative. Assets are gross
    financial stocks and must be non-negative (debt is not subtracted).
    """
    flows = [
        _array(employee, "employee"),
        _array(unemployment, "unemployment"),
        _array(pension, "pension"),
        _array(other_transfers, "other_transfers"),
        _array(rental, "rental"),
    ]
    stocks = [
        _array(liquid_assets, "liquid_assets", non_negative=True),
        _array(illiquid_assets, "illiquid_assets", non_negative=True),
    ]
    if len({array.shape for array in flows + stocks}) != 1:
        raise ValueError("assessed_income inputs must have equal shapes.")
    return sum(flows) + _scalar(notional_rate_period, "notional_rate_period") * (stocks[0] + stocks[1])


def minimum_income_top_up(*, level: np.ndarray, assessed: np.ndarray) -> np.ndarray:
    """Top-up ``T = max(M - Y, 0)``, euro-for-euro withdrawal.

    ``Y`` may be negative, so ``T`` can exceed ``M``: negative assessed income is topped up in full (intended).
    """
    level_array = _array(level, "level", non_negative=True)
    assessed_array = _array(assessed, "assessed")
    if level_array.shape != assessed_array.shape:
        raise ValueError("level and assessed must have equal shapes.")
    return np.maximum(level_array - assessed_array, 0.0)


def income_guard(*, level: np.ndarray, share: float) -> np.ndarray:
    """Numerical guard ``G = share * M`` with ``share`` in (0, 1)."""
    share = _scalar(share, "share")
    if share >= 1.0:
        raise ValueError("share must lie in (0, 1).")
    return share * _array(level, "level", non_negative=True)
