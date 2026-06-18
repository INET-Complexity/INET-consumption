"""Stage 5 (feasibility resolver), Increment 0: liquidity-shortfall diagnostic.

Pure, vectorized helper implementing the paper's liquidity-shortage definition
(``sec:financing_shortages``):

``L^d_it = -(s_it + b_it)``, where ``s_it`` is household saving and ``b_it <= 0``
is net scheduled debt repayment.

In model terms, with ``s_it = income_it - target_consumption_it`` (saving net of
consumption, before debt service) and ``b_it = -scheduled_debt_service_it``
(mortgage plus consumer-loan scheduled instalments, both non-negative), this
reduces to

``L^d_it = target_consumption_it + scheduled_debt_service_it - income_it``,

a positive value whenever desired consumption plus scheduled debt service
exceeds current income. This module performs no I/O and does not touch any
``Households``/``country.py`` state — Increment 0's diagnostic-only call site
that persists this as a time series is implemented separately. See
``knowledge-vault/wiki/architecture/consumption-stage-5-feasibility-resolver.md``
(Increment 0 section) for the call-site design and exit criterion.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiquidityShortfallResult:
    """Per-household liquidity-shortfall diagnostic for one period."""

    liquidity_shortfall: np.ndarray
    household_saving: np.ndarray


def compute_liquidity_shortfall(
    income: np.ndarray,
    target_consumption: np.ndarray,
    scheduled_debt_service: np.ndarray,
) -> LiquidityShortfallResult:
    """Compute the paper's liquidity shortfall ``L^d_it`` for one period.

    Args:
        income (np.ndarray): Current household income, ``y_it``.
        target_consumption (np.ndarray): Desired consumption expenditure for
            the period (the long-run/short-run CACF target, before any
            feasibility repair).
        scheduled_debt_service (np.ndarray): Total scheduled debt repayment
            for the period (mortgage plus consumer-loan instalments),
            non-negative.

    Returns:
        LiquidityShortfallResult: ``liquidity_shortfall`` is ``L^d_it``,
            positive when desired consumption plus scheduled debt service
            exceeds income, zero or negative (a surplus) otherwise.
            ``household_saving`` is ``s_it = income - target_consumption``,
            recorded separately since it is the paper's primitive quantity
            and useful to validate the shortfall's sign independently.
    """
    income = np.asarray(income, dtype=float)
    target_consumption = np.asarray(target_consumption, dtype=float)
    scheduled_debt_service = np.asarray(scheduled_debt_service, dtype=float)

    household_saving = income - target_consumption
    liquidity_shortfall = target_consumption + scheduled_debt_service - income

    return LiquidityShortfallResult(
        liquidity_shortfall=liquidity_shortfall,
        household_saving=household_saving,
    )
