"""Stage 5 Increment 2: shadow borrow-vs-sell comparison diagnostic."""

from dataclasses import dataclass

import numpy as np

PREFERRED_MARGIN_NONE = 0
PREFERRED_MARGIN_BORROW = 1
PREFERRED_MARGIN_SELL = 2
BORROW_VS_SELL_EPSILON = 1e-12


@dataclass(frozen=True)
class BorrowVsSellResult:
    """Per-household shadow borrow-vs-sell diagnostics for one period."""

    preferred_margin: np.ndarray
    preferred_amount: np.ndarray
    borrow_vs_sell_threshold: np.ndarray
    borrow_vs_sell_spread: np.ndarray
    borrow_vs_sell_l_tilde: np.ndarray
    comparison_valid_flag: np.ndarray


def compute_borrow_vs_sell_choice(
    residual_shortfall_after_lfa: np.ndarray,
    delta_tilde: np.ndarray,
    opening_tfa_scale: np.ndarray,
    post_return_ifa: np.ndarray,
    r_b: np.ndarray,
    r_kappa: np.ndarray,
    phi_1: float,
    lambda_kappa: float,
    *,
    epsilon: float = BORROW_VS_SELL_EPSILON,
) -> BorrowVsSellResult:
    """Choose the preferred non-liquid repair margin in shadow mode.

    The comparison uses the Stage 4 handoff quantities as the authoritative
    source for the target-gap term and normalization scale. Invalid positive-
    shortfall rows fall back deterministically to ``borrow``.
    """
    residual_shortfall_after_lfa = np.asarray(residual_shortfall_after_lfa, dtype=float)
    delta_tilde = np.asarray(delta_tilde, dtype=float)
    opening_tfa_scale = np.asarray(opening_tfa_scale, dtype=float)
    post_return_ifa = np.asarray(post_return_ifa, dtype=float)
    r_b = np.asarray(r_b, dtype=float)
    r_kappa = np.asarray(r_kappa, dtype=float)

    positive_shortfall = np.isfinite(residual_shortfall_after_lfa) & (residual_shortfall_after_lfa > 0.0)
    comparison_valid_flag = (
        positive_shortfall
        & np.isfinite(delta_tilde)
        & np.isfinite(post_return_ifa)
        & np.isfinite(r_b)
        & np.isfinite(r_kappa)
        & np.isfinite(phi_1)
        & np.isfinite(lambda_kappa)
        & (opening_tfa_scale > 0.0)
        & (phi_1 > 0.0)
        & (lambda_kappa > 0.0)
        & (lambda_kappa <= 1.0)
    )

    safe_scale = np.where(comparison_valid_flag, opening_tfa_scale, 1.0)
    borrow_vs_sell_l_tilde = np.where(
        comparison_valid_flag,
        residual_shortfall_after_lfa / safe_scale,
        0.0,
    )
    borrow_vs_sell_threshold = np.where(
        comparison_valid_flag,
        r_kappa + (phi_1 / (2.0 * lambda_kappa)) * (2.0 * lambda_kappa * delta_tilde + borrow_vs_sell_l_tilde),
        0.0,
    )
    borrow_vs_sell_spread = np.where(
        comparison_valid_flag,
        r_b - borrow_vs_sell_threshold,
        0.0,
    )

    preferred_margin = np.full(residual_shortfall_after_lfa.shape, PREFERRED_MARGIN_NONE, dtype=np.int8)
    preferred_amount = np.zeros_like(residual_shortfall_after_lfa, dtype=float)

    invalid_positive_shortfall = positive_shortfall & ~comparison_valid_flag
    preferred_margin = np.where(invalid_positive_shortfall, PREFERRED_MARGIN_BORROW, preferred_margin)
    preferred_amount = np.where(invalid_positive_shortfall, residual_shortfall_after_lfa, preferred_amount)

    zero_ifa = comparison_valid_flag & (post_return_ifa <= 0.0)
    preferred_margin = np.where(zero_ifa, PREFERRED_MARGIN_BORROW, preferred_margin)
    preferred_amount = np.where(zero_ifa, residual_shortfall_after_lfa, preferred_amount)

    spread_prefers_sell = comparison_valid_flag & (post_return_ifa > 0.0) & (borrow_vs_sell_spread > epsilon)
    spread_prefers_borrow = comparison_valid_flag & (post_return_ifa > 0.0) & ~spread_prefers_sell

    preferred_margin = np.where(spread_prefers_sell, PREFERRED_MARGIN_SELL, preferred_margin)
    preferred_margin = np.where(spread_prefers_borrow, PREFERRED_MARGIN_BORROW, preferred_margin)
    preferred_amount = np.where(spread_prefers_sell | spread_prefers_borrow, residual_shortfall_after_lfa, preferred_amount)

    return BorrowVsSellResult(
        preferred_margin=preferred_margin,
        preferred_amount=preferred_amount,
        borrow_vs_sell_threshold=borrow_vs_sell_threshold,
        borrow_vs_sell_spread=borrow_vs_sell_spread,
        borrow_vs_sell_l_tilde=borrow_vs_sell_l_tilde,
        comparison_valid_flag=comparison_valid_flag,
    )
