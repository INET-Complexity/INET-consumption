"""Stage 5 Increment 3: shadow residual-capacity fallback.

This helper caps the preferred Stage 5 branch from Increment 2 using a
provisional consumer-credit DSTI proxy and feasible illiquid-asset
liquidation capacity. It is still shadow-only: it does not mutate household
stocks, debt service, or bank-clearing state.

The DSTI proxy intentionally uses the new-loan annuity cap with mortgage
service only:

``psi * Y - a^m``

where ``psi`` is the calibrated debt-service-to-income ceiling, ``Y`` is
gross household income, and ``a^m`` is scheduled mortgage service. That
proxy is a Stage 5 approximation only; settled remodulated consumer-debt
service belongs to Stage 6.
"""

from dataclasses import dataclass

import numpy as np

from macromodel.agents.households.func.borrow_vs_sell import (
    PREFERRED_MARGIN_BORROW,
    PREFERRED_MARGIN_SELL,
)

_RESIDUAL_CAPACITY_EPSILON = 1e-12


@dataclass(frozen=True)
class ResidualCapacityFallbackResult:
    """Per-household Stage 5 residual-capacity diagnostics for one period."""

    dsti_headroom: np.ndarray
    dsti_maximum_loan_size: np.ndarray
    dsti_cap_binding: np.ndarray
    borrow_planned: np.ndarray
    liquidation_planned: np.ndarray
    shadow_credit_requested: np.ndarray
    forced_liquidation_amount: np.ndarray
    residual_shortfall_after_caps: np.ndarray


def _annuity_payment_factor(period_rate: np.ndarray, loan_maturity: np.ndarray) -> np.ndarray:
    """Return the fixed-payment annuity factor used by the credit market."""
    period_rate = np.asarray(period_rate, dtype=float)
    loan_maturity = np.asarray(loan_maturity, dtype=float)

    valid = np.isfinite(period_rate) & np.isfinite(loan_maturity) & (loan_maturity > 0.0)
    safe_rate = np.where(valid & (period_rate > 0.0), period_rate, 0.0)
    safe_maturity = np.where(valid, loan_maturity, 1.0)

    zero_rate_factor = 1.0 / safe_maturity
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        positive_rate_factor = safe_rate / (1.0 - np.power(1.0 + safe_rate, -safe_maturity))
    factor = np.where(safe_rate > 0.0, positive_rate_factor, zero_rate_factor)
    return np.where(valid & np.isfinite(factor) & (factor > 0.0), factor, 0.0)


def compute_residual_capacity_fallback(
    preferred_margin_after_lfa: np.ndarray,
    preferred_margin_amount: np.ndarray,
    income: np.ndarray,
    scheduled_mortgage_payment: np.ndarray,
    r_b: np.ndarray,
    consumer_loan_maturity: np.ndarray | int,
    dsti_limit: np.ndarray | float,
    current_ifa: np.ndarray,
) -> ResidualCapacityFallbackResult:
    """Apply the shadow DSTI and liquidation caps to the preferred Stage 5 branch.

    Invalid or non-finite inputs collapse to a conservative zero-capacity
    fallback. Positive residuals are never driven negative by this helper.
    """
    preferred_margin_after_lfa = np.asarray(preferred_margin_after_lfa, dtype=float)
    preferred_margin_amount = np.asarray(preferred_margin_amount, dtype=float)
    income = np.asarray(income, dtype=float)
    scheduled_mortgage_payment = np.asarray(scheduled_mortgage_payment, dtype=float)
    r_b = np.asarray(r_b, dtype=float)
    consumer_loan_maturity = np.asarray(consumer_loan_maturity, dtype=float)
    dsti_limit = np.asarray(dsti_limit, dtype=float)
    current_ifa = np.asarray(current_ifa, dtype=float)

    (
        preferred_margin_after_lfa,
        preferred_margin_amount,
        income,
        scheduled_mortgage_payment,
        r_b,
        consumer_loan_maturity,
        dsti_limit,
        current_ifa,
    ) = np.broadcast_arrays(
        preferred_margin_after_lfa,
        preferred_margin_amount,
        income,
        scheduled_mortgage_payment,
        r_b,
        consumer_loan_maturity,
        dsti_limit,
        current_ifa,
    )

    positive_amount = np.isfinite(preferred_margin_amount) & (preferred_margin_amount > 0.0)
    borrow_branch = positive_amount & (preferred_margin_after_lfa == PREFERRED_MARGIN_BORROW)
    sell_branch = positive_amount & (preferred_margin_after_lfa == PREFERRED_MARGIN_SELL)
    valid_margin = borrow_branch | sell_branch

    proxy_valid = (
        valid_margin
        & np.isfinite(income)
        & np.isfinite(scheduled_mortgage_payment)
        & np.isfinite(r_b)
        & np.isfinite(consumer_loan_maturity)
        & np.isfinite(dsti_limit)
        & np.isfinite(current_ifa)
        & (consumer_loan_maturity > 0.0)
        & (dsti_limit > 0.0)
    )

    dsti_headroom = np.where(
        proxy_valid,
        np.maximum(dsti_limit * income - scheduled_mortgage_payment, 0.0),
        0.0,
    )
    safe_rate = np.where(proxy_valid & (r_b > 0.0), r_b, 0.0)
    annuity_factor = _annuity_payment_factor(safe_rate, consumer_loan_maturity)
    dsti_maximum_loan_size = np.where(
        proxy_valid & (annuity_factor > 0.0),
        dsti_headroom / annuity_factor,
        0.0,
    )
    dsti_maximum_loan_size = np.where(
        np.isfinite(dsti_maximum_loan_size) & (dsti_maximum_loan_size > 0.0),
        dsti_maximum_loan_size,
        0.0,
    )

    liquidation_capacity = np.where(
        proxy_valid & np.isfinite(current_ifa) & (current_ifa > 0.0),
        current_ifa,
        0.0,
    )

    borrow_planned = np.zeros_like(preferred_margin_amount, dtype=float)
    liquidation_planned = np.zeros_like(preferred_margin_amount, dtype=float)
    forced_liquidation_amount = np.zeros_like(preferred_margin_amount, dtype=float)

    borrow_request = np.where(borrow_branch, preferred_margin_amount, 0.0)
    borrow_planned = np.where(
        borrow_branch,
        np.minimum(borrow_request, dsti_maximum_loan_size),
        borrow_planned,
    )
    borrow_shortfall = np.maximum(borrow_request - borrow_planned, 0.0)

    liquidation_planned = np.where(
        borrow_branch,
        np.minimum(borrow_shortfall, liquidation_capacity),
        liquidation_planned,
    )
    forced_liquidation_amount = np.where(borrow_branch, liquidation_planned, forced_liquidation_amount)

    sell_request = np.where(sell_branch, preferred_margin_amount, 0.0)
    sell_liquidation = np.where(sell_branch, np.minimum(sell_request, liquidation_capacity), 0.0)
    liquidation_planned = np.where(sell_branch, sell_liquidation, liquidation_planned)
    sell_shortfall = np.maximum(sell_request - sell_liquidation, 0.0)
    sell_borrow = np.where(sell_branch, np.minimum(sell_shortfall, dsti_maximum_loan_size), 0.0)
    borrow_planned = np.where(sell_branch, sell_borrow, borrow_planned)

    shadow_credit_requested = borrow_planned.copy()

    branch_residual = np.where(
        borrow_branch,
        preferred_margin_amount - borrow_planned - liquidation_planned,
        np.where(sell_branch, preferred_margin_amount - liquidation_planned - borrow_planned, preferred_margin_amount),
    )
    residual_shortfall_after_caps = np.where(
        positive_amount & np.isfinite(branch_residual),
        np.maximum(branch_residual, 0.0),
        0.0,
    )

    dsti_cap_binding = np.where(
        borrow_branch,
        borrow_planned < (borrow_request - _RESIDUAL_CAPACITY_EPSILON),
        np.where(
            sell_branch,
            sell_borrow < (sell_shortfall - _RESIDUAL_CAPACITY_EPSILON),
            False,
        ),
    )

    return ResidualCapacityFallbackResult(
        dsti_headroom=dsti_headroom,
        dsti_maximum_loan_size=dsti_maximum_loan_size,
        dsti_cap_binding=dsti_cap_binding,
        borrow_planned=borrow_planned,
        liquidation_planned=liquidation_planned,
        shadow_credit_requested=shadow_credit_requested,
        forced_liquidation_amount=forced_liquidation_amount,
        residual_shortfall_after_caps=residual_shortfall_after_caps,
    )
