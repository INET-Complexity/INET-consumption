"""Stage 4 (portfolio choice), Increment 3: per-household diagnostics composition.

Pure, vectorized helper that composes Increment 1's target-share contract
(``compute_target_illiquid_share``) with Increment 2's rebalancing helper
(``compute_portfolio_rebalancing``) into the full set of Stage 4 shadow
diagnostics for one period. This module performs no I/O and does not touch
any ``Households``/``update_wealth()`` state — the household call site that
wires this into the runtime, and persists the results as time series, is
Increment 3's remaining (non-pure) scope, implemented separately in
``Households.compute_stage4_portfolio_diagnostics()``.

See ``knowledge-vault/wiki/architecture/consumption-stage-4-portfolio-choice.md``
(Increment 3 section) for the full call-site design. ``post_surplus_lfa`` is
the household's actual, already-updated current liquid balance sheet
(``self.ts.current("wealth_deposits")``) — the real entry point for the
portfolio decision, not a parallel shadow quantity re-derived from
``investable_surplus``. The model's deposit update already embeds the
period's positive-surplus acquisition (new savings, withdrawals, interest,
debt installments, new loans, tax), so this function does not re-add
``investable_surplus`` to anything; it only echoes it back as a diagnostic.
"""

from dataclasses import dataclass

import numpy as np

from macromodel.agents.households.func.portfolio_rebalancing import (
    PortfolioRebalancingResult,
    compute_portfolio_rebalancing,
)
from macromodel.agents.households.func.portfolio_target_share import (
    compute_target_illiquid_share,
)


@dataclass(frozen=True)
class Stage4HouseholdDiagnostics:
    """Per-household Stage 4 shadow diagnostics for one period.

    Combines the Increment 1 target-share contract with the Increment 2
    rebalancing decision, plus the upstream timing variables (opening/post-
    return stocks, investable surplus) needed to interpret them.
    """

    portfolio_opening_tfa_scale: np.ndarray
    portfolio_target_tfa_base: np.ndarray
    portfolio_post_return_lfa: np.ndarray
    portfolio_post_return_ifa: np.ndarray
    portfolio_investable_surplus: np.ndarray
    portfolio_target_illiquid_share: np.ndarray
    portfolio_target_share_clipped_flag: np.ndarray
    rebalancing: PortfolioRebalancingResult


def compute_stage4_household_diagnostics(
    opening_tfa_scale: np.ndarray,
    post_surplus_lfa: np.ndarray,
    post_return_ifa: np.ndarray,
    investable_surplus: np.ndarray,
    portfolio_participates: np.ndarray,
    target_share_source: str,
    default_target_illiquid_share: float,
    phi_1: float,
    lambda_kappa: float,
    fixed_cost_share: float,
    frm_covariates: dict[str, np.ndarray] | None = None,
    frm_magnitude_coefficients: dict[str, float] | None = None,
    population_scale_factor: float | None = None,
    net_wealth_scale_divisor: float | None = None,
) -> Stage4HouseholdDiagnostics:
    """Compute the full Stage 4 shadow diagnostics set for one household-period.

    Implements the page's Paper Rule To Implement timing:

    ``target_tfa_base = post_surplus_lfa + post_return_ifa``
    ``target_illiquid_assets = target_illiquid_share * target_tfa_base``

    then calls Increment 2's ``compute_portfolio_rebalancing`` with the
    resulting ``target_illiquid_assets`` and ``target_tfa_base`` as separate,
    precomputed inputs (per Increment 2's finalized 5-array signature).

    ``post_surplus_lfa`` is supplied by the caller already inclusive of any
    positive investable surplus — this function does not floor or add
    ``investable_surplus`` to it. ``investable_surplus`` is forwarded only as
    a diagnostic value, independent of the ``post_surplus_lfa`` arithmetic.

    Args:
        opening_tfa_scale (np.ndarray): Previous-period total financial
            assets, ``LFA_{t-1} + IFA_{t-1}``.
        post_surplus_lfa (np.ndarray): Liquid assets after the period's
            return and after any positive investable surplus has already
            been incorporated (the page's ``post_surplus_lfa`), supplied
            directly by the caller from the actual household balance sheet.
        post_return_ifa (np.ndarray): Illiquid assets after the period's
            return, before the rebalancing decision.
        investable_surplus (np.ndarray): $\\tilde{s}_{it}$, unfloored (may be
            negative); recorded as a diagnostic only, not used in the
            ``target_tfa_base``/rebalancing arithmetic below.
        portfolio_participates (np.ndarray): Boolean array, frozen at init
            from ``initial_IFA > 0``.
        target_share_source (str): Forwarded to
            ``compute_target_illiquid_share``. ``"scalar"`` remains the
            default; ``"frm_magnitude"`` (Increment 5) is an opt-in
            alternative that requires the four ``frm_*``/scale arguments
            below.
        default_target_illiquid_share (float): Forwarded to
            ``compute_target_illiquid_share``. Ignored under
            ``"frm_magnitude"``.
        phi_1 (float): Forwarded to ``compute_portfolio_rebalancing``.
        lambda_kappa (float): Forwarded to ``compute_portfolio_rebalancing``.
        fixed_cost_share (float): Forwarded to ``compute_portfolio_rebalancing``.
        frm_covariates (dict[str, np.ndarray] | None): Forwarded to
            ``compute_target_illiquid_share``; required only when
            ``target_share_source="frm_magnitude"``.
        frm_magnitude_coefficients (dict[str, float] | None): Forwarded to
            ``compute_target_illiquid_share``; required only when
            ``target_share_source="frm_magnitude"``.
        population_scale_factor (float | None): Forwarded to
            ``compute_target_illiquid_share``; required only when
            ``target_share_source="frm_magnitude"``.
        net_wealth_scale_divisor (float | None): Forwarded to
            ``compute_target_illiquid_share``; required only when
            ``target_share_source="frm_magnitude"``.

    Returns:
        Stage4HouseholdDiagnostics: The full diagnostics set for this period.
    """
    investable_surplus = np.asarray(investable_surplus, dtype=float)
    post_surplus_lfa = np.asarray(post_surplus_lfa, dtype=float)
    post_return_ifa = np.asarray(post_return_ifa, dtype=float)

    target_tfa_base = post_surplus_lfa + post_return_ifa

    target_illiquid_share, target_share_clipped_flag = compute_target_illiquid_share(
        portfolio_participates=portfolio_participates,
        target_share_source=target_share_source,
        default_target_illiquid_share=default_target_illiquid_share,
        frm_covariates=frm_covariates,
        frm_magnitude_coefficients=frm_magnitude_coefficients,
        population_scale_factor=population_scale_factor,
        net_wealth_scale_divisor=net_wealth_scale_divisor,
    )
    target_illiquid_assets = target_illiquid_share * target_tfa_base

    rebalancing = compute_portfolio_rebalancing(
        opening_tfa_scale=opening_tfa_scale,
        post_return_lfa=post_surplus_lfa,
        post_return_ifa=post_return_ifa,
        target_illiquid_assets=target_illiquid_assets,
        target_tfa_base=target_tfa_base,
        portfolio_participates=portfolio_participates,
        phi_1=phi_1,
        lambda_kappa=lambda_kappa,
        fixed_cost_share=fixed_cost_share,
    )

    return Stage4HouseholdDiagnostics(
        portfolio_opening_tfa_scale=np.asarray(opening_tfa_scale, dtype=float),
        portfolio_target_tfa_base=target_tfa_base,
        portfolio_post_return_lfa=post_surplus_lfa,
        portfolio_post_return_ifa=post_return_ifa,
        portfolio_investable_surplus=investable_surplus,
        portfolio_target_illiquid_share=target_illiquid_share,
        portfolio_target_share_clipped_flag=target_share_clipped_flag,
        rebalancing=rebalancing,
    )
