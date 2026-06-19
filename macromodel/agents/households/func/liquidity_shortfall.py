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
``Households``/``country.py`` state. The diagnostic-only call site that
persists this as a time series is ``Households.compute_and_record_liquidity_shortfall()``
(``macromodel/agents/households/households.py``), invoked from
``Country._set_household_target_demand()`` right after ``target_consumption``
is finalized for the period — not inside ``compute_target_consumption()``
itself, since that method has no ``scheduled_debt_service`` input today (only
mortgage-only ``mortgage_payment``) and threading a new combined-debt-service
parameter through its already-large signature was judged worse than a sibling
call. See ``knowledge-vault/wiki/architecture/consumption-stage-5-feasibility-resolver.md``
(Increment 0 section) for the exit criterion.
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

    Following the non-finite-input fallback contract used elsewhere in this
    package (see ``portfolio_rebalancing.py``), household-periods with any
    non-finite input, or a negative ``scheduled_debt_service`` (a contract
    violation rather than an expected economic state), have both outputs
    overwritten with ``0.0`` rather than propagating NaN or a silently wrong
    sign.

    Args:
        income (np.ndarray): Current household income, ``y_it``.
        target_consumption (np.ndarray): Desired consumption expenditure for
            the period (the long-run/short-run CACF target, before any
            feasibility repair).
        scheduled_debt_service (np.ndarray): Total scheduled debt repayment
            for the period (mortgage plus consumer-loan instalments).
            Expected non-negative; negative values are treated as invalid
            (see fallback contract above) rather than asserted against, to
            keep this diagnostics-only function side-effect-free.

    Returns:
        LiquidityShortfallResult: ``liquidity_shortfall`` is ``L^d_it``,
            positive when desired consumption plus scheduled debt service
            exceeds income, zero or negative (a surplus) otherwise; ``0.0``
            for any household-period with a non-finite input or negative
            ``scheduled_debt_service``. ``household_saving`` is
            ``s_it = income - target_consumption``, recorded separately
            since it is the paper's primitive quantity and useful to
            validate the shortfall's sign independently; subject to the same
            fallback.
    """
    income = np.asarray(income, dtype=float)
    target_consumption = np.asarray(target_consumption, dtype=float)
    scheduled_debt_service = np.asarray(scheduled_debt_service, dtype=float)

    valid = (
        np.isfinite(income)
        & np.isfinite(target_consumption)
        & np.isfinite(scheduled_debt_service)
        & (scheduled_debt_service >= 0.0)
    )

    household_saving = np.where(valid, income - target_consumption, 0.0)
    liquidity_shortfall = np.where(valid, target_consumption + scheduled_debt_service - income, 0.0)

    return LiquidityShortfallResult(
        liquidity_shortfall=liquidity_shortfall,
        household_saving=household_saving,
    )
