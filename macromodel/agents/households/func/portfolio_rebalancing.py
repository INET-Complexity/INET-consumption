"""Stage 4 (portfolio choice), Increment 2: pure shadow rebalancing helper.

Pure, vectorized helper that decides, for each household, how much to
rebalance between liquid (``LFA``) and illiquid (``IFA``) financial assets
given a precomputed target illiquid level, subject to a quadratic adjustment
cost plus a fixed cost and hard feasibility bounds (no borrowing to buy
illiquid assets; cannot sell more illiquid assets than held). This module
implements only the "Constrained adjustment rule" and "Implementation
requirements beyond the formula" subsections of the Increment 2 section in
``knowledge-vault/wiki/architecture/consumption-stage-4-portfolio-choice.md``.

This function does not mutate any household object, perform any I/O, or
touch ``Households``/``update_wealth()`` state — it is a shadow/diagnostic
helper only. The household call site that wires this into the runtime is
Increment 3's scope.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PortfolioRebalancingResult:
    """Per-household Stage 4 Increment 2 shadow rebalancing diagnostics."""

    portfolio_participates: np.ndarray
    actual_illiquid_share: np.ndarray
    target_illiquid_assets: np.ndarray
    delta_tilde: np.ndarray
    kappa_star_tilde: np.ndarray
    kappa_tilde: np.ndarray
    desired_illiquid_adjustment: np.ndarray
    adjustment_cost: np.ndarray
    counterfactual_lfa_flow: np.ndarray
    counterfactual_ifa_flow: np.ndarray
    inaction_flag: np.ndarray
    upper_bound_flag: np.ndarray
    lower_bound_flag: np.ndarray
    infeasible_interval_flag: np.ndarray
    no_financial_assets_flag: np.ndarray
    portfolio_valid_flag: np.ndarray


_ACTUAL_SHARE_DENOMINATOR_EPS = 1e-12


def compute_portfolio_rebalancing(
    opening_tfa_scale: np.ndarray,
    post_return_lfa: np.ndarray,
    post_return_ifa: np.ndarray,
    target_illiquid_assets: np.ndarray,
    target_tfa_base: np.ndarray,
    portfolio_participates: np.ndarray,
    phi_1: float,
    lambda_kappa: float,
    fixed_cost_share: float,
) -> PortfolioRebalancingResult:
    """Compute the constrained shadow portfolio-rebalancing decision per household.

    Implements the "Constrained adjustment rule" from
    ``knowledge-vault/wiki/architecture/consumption-stage-4-portfolio-choice.md``
    (Increment 2): the unconstrained normalized adjustment ``kappa_star_tilde
    = lambda_kappa * delta_tilde`` is projected onto the feasibility interval
    ``[kappa_lower, kappa_upper]`` (no borrowing to buy illiquid assets,
    cannot sell more illiquid assets than held), then a discrete gain test
    ``gain = L(0) - L(kappa_breve)`` decides whether the household pays the
    fixed cost to adjust at all. All quantities are normalized by
    ``opening_tfa_scale`` and de-normalized back to currency units for the
    counterfactual flow/cost outputs. ``target_illiquid_assets`` is
    precomputed by the caller (Increment 1's target-share contract composed
    with ``target_tfa_base``) and is only echoed back here, not derived.

    Non-finite inputs (NaN/inf in any input array) and households with
    ``opening_tfa_scale <= 0`` get every numeric output zeroed, every boolean
    flag set to ``False``, and ``portfolio_valid_flag = False`` (per the
    wiki's "non-finite-input fallback contract"), fully vectorized via
    ``np.where``/boolean masking.

    Households whose feasible interval is empty (``kappa_lower >
    kappa_upper``, i.e. neither buying nor selling illiquid assets can cover
    the fixed cost) are forced into inaction with zero adjustment cost,
    independent of the discrete gain test, and flagged via
    ``infeasible_interval_flag``. Without this, ``np.clip`` would silently
    return the upper bound for any inverted interval regardless of the
    unconstrained solution, fabricating a transaction these households
    cannot actually afford.

    Args:
        opening_tfa_scale (np.ndarray): Previous-period total financial
            assets, ``LFA_{t-1} + IFA_{t-1}``; the normalization scale.
        post_return_lfa (np.ndarray): Liquid assets after the period's return
            and positive investable surplus, before the rebalancing decision.
        post_return_ifa (np.ndarray): Illiquid assets after the period's
            return, before the rebalancing decision.
        target_illiquid_assets (np.ndarray): Precomputed target illiquid
            asset level (``target_illiquid_share * target_tfa_base``),
            supplied by the caller; echoed back unchanged in the result.
        target_tfa_base (np.ndarray): Current-period post-return-and-surplus
            total financial assets (``post_surplus_lfa + post_return_ifa``,
            per the wiki's Paper Rule To Implement), supplied by the caller.
            Used only as the denominator for the ``actual_illiquid_share``
            diagnostic (per the wiki's Data Inputs table), distinct from
            ``opening_tfa_scale``, which normalizes the adjustment decision
            itself. Non-positive values (reachable when combined liquid and
            illiquid stocks are themselves negative, e.g. overdraft-style
            negative deposits) yield ``actual_illiquid_share = 0`` rather
            than dividing by a near-zero epsilon, which would otherwise
            produce an enormous, meaningless ratio.
        portfolio_participates (np.ndarray): Boolean array; nonparticipants
            run the identical projection/gain-test path (their
            ``target_illiquid_assets`` is already ``0`` upstream) rather than
            a dedicated short-circuit branch, per the wiki's decision.
        phi_1 (float): Penalty on deviation from the target illiquid level.
        lambda_kappa (float): Speed of adjustment toward the target illiquid
            share; must satisfy ``0 < lambda_kappa <= 1`` (config-time
            invariant of ``phi_2 = phi_1 * (1 - lambda_kappa) / lambda_kappa``
            this helper re-validates defensively). ``phi_2`` is derived
            internally from ``phi_1`` and ``lambda_kappa``.
        fixed_cost_share (float): Fixed adjustment cost as a share of
            ``opening_tfa_scale``, paid only when a nonzero adjustment occurs.

    Returns:
        PortfolioRebalancingResult: One array per named diagnostic field,
        each shape ``(n_households,)``.

    Raises:
        ValueError: If ``lambda_kappa`` is not in ``(0, 1]``.
    """
    if not (0.0 < lambda_kappa <= 1.0):
        raise ValueError(f"lambda_kappa must satisfy 0 < lambda_kappa <= 1, got {lambda_kappa}.")

    phi_1 = float(phi_1)
    fixed_cost_share = float(fixed_cost_share)
    phi_2 = phi_1 * (1.0 - lambda_kappa) / lambda_kappa

    opening_tfa_scale = np.asarray(opening_tfa_scale, dtype=float)
    post_return_lfa = np.asarray(post_return_lfa, dtype=float)
    post_return_ifa = np.asarray(post_return_ifa, dtype=float)
    target_illiquid_assets = np.asarray(target_illiquid_assets, dtype=float)
    target_tfa_base = np.asarray(target_tfa_base, dtype=float)
    portfolio_participates = np.asarray(portfolio_participates, dtype=bool)

    no_financial_assets_flag = opening_tfa_scale <= 0.0
    finite_inputs = (
        np.isfinite(opening_tfa_scale)
        & np.isfinite(post_return_lfa)
        & np.isfinite(post_return_ifa)
        & np.isfinite(target_illiquid_assets)
        & np.isfinite(target_tfa_base)
    )
    portfolio_valid_flag = finite_inputs & ~no_financial_assets_flag

    # Use a safe denominator for the division below; results for invalid
    # household-periods are overwritten with zeros/False at the end via
    # np.where, so the actual value computed under an unsafe scale is
    # discarded, not propagated.
    safe_scale = np.where(portfolio_valid_flag, opening_tfa_scale, 1.0)

    # actual_illiquid_share uses target_tfa_base (current-period
    # post-return-and-surplus TFA), not opening_tfa_scale, per the wiki's
    # Data Inputs table: it answers "what fraction of assets are illiquid
    # right now," including this period's saving, not last period's stock.
    # A non-positive target_tfa_base (reachable when post_return_lfa +
    # post_return_ifa is itself negative, e.g. overdraft-style negative
    # deposits) is treated as "no measurable share," not floored to a tiny
    # epsilon, which would otherwise blow up the ratio to an enormous,
    # meaningless value rather than signalling the no-assets state.
    has_measurable_share = portfolio_valid_flag & (target_tfa_base > 0.0)
    safe_actual_share_denominator = np.maximum(target_tfa_base, _ACTUAL_SHARE_DENOMINATOR_EPS)
    actual_illiquid_share = np.divide(
        post_return_ifa,
        safe_actual_share_denominator,
        out=np.zeros_like(safe_actual_share_denominator),
        where=has_measurable_share,
    )

    delta_tilde = (target_illiquid_assets - post_return_ifa) / safe_scale
    kappa_star_tilde = lambda_kappa * delta_tilde

    upsilon_tilde = fixed_cost_share
    kappa_lower = -((post_return_ifa / safe_scale) - upsilon_tilde)
    kappa_upper = (post_return_lfa / safe_scale) - upsilon_tilde

    upper_bound_flag = kappa_star_tilde > kappa_upper
    lower_bound_flag = kappa_star_tilde < kappa_lower

    # An empty/inverted feasible interval (kappa_lower > kappa_upper) is
    # reachable whenever a household's combined liquid+illiquid buffer cannot
    # cover the fixed cost in either direction. np.clip resolves lower > upper
    # by always returning upper regardless of the input, which would silently
    # fabricate a nonzero adjustment and a real fixed-cost charge for a
    # household that can afford neither direction. Force inaction for these
    # household-periods independently of the clip/gain test below.
    infeasible_interval_flag = kappa_lower > kappa_upper

    kappa_breve = np.clip(kappa_star_tilde, kappa_lower, kappa_upper)

    def loss(kappa_tilde: np.ndarray) -> np.ndarray:
        quadratic = (phi_1 / 2.0) * (delta_tilde - kappa_tilde) ** 2 + (phi_2 / 2.0) * kappa_tilde**2
        fixed_cost = np.where(kappa_tilde != 0.0, upsilon_tilde, 0.0)
        return quadratic + fixed_cost

    # Non-finite household-periods may carry inf/NaN through delta_tilde and
    # kappa_breve here; the resulting inf-inf/NaN warnings are expected and
    # harmless because gain/kappa_tilde_final for those rows are overwritten
    # to 0 below via portfolio_valid_flag masking.
    with np.errstate(invalid="ignore"):
        loss_zero = loss(np.zeros_like(kappa_breve))
        loss_breve = loss(kappa_breve)
        gain = loss_zero - loss_breve

    adjust = (gain > 0.0) & ~infeasible_interval_flag
    kappa_tilde_final = np.where(adjust, kappa_breve, 0.0)
    inaction_flag = ~adjust

    desired_illiquid_adjustment = kappa_tilde_final * safe_scale
    adjustment_cost = np.where(kappa_tilde_final != 0.0, upsilon_tilde * safe_scale, 0.0)
    counterfactual_ifa_flow = desired_illiquid_adjustment
    counterfactual_lfa_flow = -desired_illiquid_adjustment - adjustment_cost

    # Zero out all numeric outputs and clear all boolean flags for invalid
    # household-periods (non-finite inputs or opening_tfa_scale <= 0), per
    # the wiki's non-finite-input fallback contract.
    actual_illiquid_share = np.where(portfolio_valid_flag, actual_illiquid_share, 0.0)
    delta_tilde = np.where(portfolio_valid_flag, delta_tilde, 0.0)
    kappa_star_tilde = np.where(portfolio_valid_flag, kappa_star_tilde, 0.0)
    kappa_tilde_final = np.where(portfolio_valid_flag, kappa_tilde_final, 0.0)
    desired_illiquid_adjustment = np.where(portfolio_valid_flag, desired_illiquid_adjustment, 0.0)
    adjustment_cost = np.where(portfolio_valid_flag, adjustment_cost, 0.0)
    counterfactual_ifa_flow = np.where(portfolio_valid_flag, counterfactual_ifa_flow, 0.0)
    counterfactual_lfa_flow = np.where(portfolio_valid_flag, counterfactual_lfa_flow, 0.0)
    inaction_flag = inaction_flag & portfolio_valid_flag
    upper_bound_flag = upper_bound_flag & portfolio_valid_flag
    lower_bound_flag = lower_bound_flag & portfolio_valid_flag
    infeasible_interval_flag = infeasible_interval_flag & portfolio_valid_flag
    target_illiquid_assets_echo = np.where(portfolio_valid_flag, target_illiquid_assets, 0.0)

    return PortfolioRebalancingResult(
        portfolio_participates=portfolio_participates,
        actual_illiquid_share=actual_illiquid_share,
        target_illiquid_assets=target_illiquid_assets_echo,
        delta_tilde=delta_tilde,
        kappa_star_tilde=kappa_star_tilde,
        kappa_tilde=kappa_tilde_final,
        desired_illiquid_adjustment=desired_illiquid_adjustment,
        adjustment_cost=adjustment_cost,
        counterfactual_lfa_flow=counterfactual_lfa_flow,
        counterfactual_ifa_flow=counterfactual_ifa_flow,
        inaction_flag=inaction_flag,
        upper_bound_flag=upper_bound_flag,
        lower_bound_flag=lower_bound_flag,
        infeasible_interval_flag=infeasible_interval_flag,
        no_financial_assets_flag=no_financial_assets_flag,
        portfolio_valid_flag=portfolio_valid_flag,
    )
