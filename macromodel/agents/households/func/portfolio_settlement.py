"""Settled Stage 4 portfolio reallocation on top of the wealth update."""

from dataclasses import dataclass

import numpy as np

from macromodel.agents.households.func.portfolio_rebalancing import PortfolioRebalancingResult

SETTLEMENT_DISABLED = 0.0
# Retired: rebalancing no longer requires new saving, so this status is never assigned. The code is kept
# so the numeric status values of older output files keep their meaning.
SETTLEMENT_NO_SURPLUS = 1.0
SETTLEMENT_INACTION = 2.0
SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY = 3.0
SETTLEMENT_FORCED_LIQUIDATION = 4.0
SETTLEMENT_INVALID_INPUT = 5.0
SETTLEMENT_CONSERVATION_FAILURE = 6.0
SETTLEMENT_SETTLED = 7.0

_CLEARING_NOISE_TOL = 1e-6


@dataclass(frozen=True)
class PortfolioSettlement:
    """Atomic per-household portfolio reallocation result."""

    base_lfa: np.ndarray
    base_ifa: np.ndarray
    committed_lfa_flow: np.ndarray
    committed_ifa_flow: np.ndarray
    committed_adjustment_cost: np.ndarray
    closing_lfa: np.ndarray
    closing_ifa: np.ndarray
    closing_tfa: np.ndarray
    settlement_enabled: np.ndarray
    settlement_valid_flag: np.ndarray
    settlement_status: np.ndarray


def settle_portfolio_reallocation(
    *,
    base_lfa: np.ndarray,
    base_ifa: np.ndarray,
    rebalancing: PortfolioRebalancingResult,
    settlement_enabled: bool,
    forced_liquidation_active: np.ndarray,
    waive_fixed_cost: np.ndarray | None = None,
) -> PortfolioSettlement:
    """Commit Stage 4 reallocation flows without allocating saving twice.

    ``base_lfa`` is the current liquid balance after the ordinary wealth update;
    it already includes this period's saving, whatever its sign. The Stage 4 flows
    therefore only move wealth between LFA and IFA and charge adjustment cost on
    LFA. Households rebalance their existing portfolio whether or not they saved
    this period: the rebalancing rule is a function of the stocks, never of the
    period saving flow, so no saving input is needed here.

    ``waive_fixed_cost`` marks households that already paid the illiquid-trade
    cost this period on a Stage 5 sale. Their trade settles without the fixed
    cost: only ``committed_cost`` is zeroed, so the LFA flow stays the pure
    transfer ``-committed_ifa`` and no money is created or destroyed.
    """
    base_lfa = np.asarray(base_lfa, dtype=float)
    base_ifa = np.asarray(base_ifa, dtype=float)
    forced_liquidation_active = np.asarray(forced_liquidation_active, dtype=bool)
    portfolio_valid_flag = np.asarray(rebalancing.portfolio_valid_flag, dtype=bool)
    counterfactual_lfa_flow = np.asarray(rebalancing.counterfactual_lfa_flow, dtype=float)
    counterfactual_ifa_flow = np.asarray(rebalancing.counterfactual_ifa_flow, dtype=float)
    adjustment_cost = np.asarray(rebalancing.adjustment_cost, dtype=float)
    inaction_flag = np.asarray(rebalancing.inaction_flag, dtype=bool)
    arrays = (
        base_lfa,
        base_ifa,
        forced_liquidation_active,
        portfolio_valid_flag,
        counterfactual_lfa_flow,
        counterfactual_ifa_flow,
        adjustment_cost,
        inaction_flag,
    )
    shape = base_lfa.shape
    if any(values.shape != shape for values in arrays):
        raise ValueError("Portfolio settlement inputs must have one value per household.")

    committed_lfa = np.zeros(shape, dtype=float)
    committed_ifa = np.zeros(shape, dtype=float)
    committed_cost = np.zeros(shape, dtype=float)
    sale_while_overdrawn_eligible = np.zeros(shape, dtype=bool)
    status = np.full(shape, SETTLEMENT_DISABLED if not settlement_enabled else SETTLEMENT_SETTLED)
    valid = np.ones(shape, dtype=bool)

    if settlement_enabled:
        finite_inputs = (
            np.isfinite(base_lfa)
            & np.isfinite(base_ifa)
            & np.isfinite(counterfactual_lfa_flow)
            & np.isfinite(counterfactual_ifa_flow)
            & np.isfinite(adjustment_cost)
        )
        conservation_valid = np.isclose(
            counterfactual_lfa_flow + counterfactual_ifa_flow + adjustment_cost,
            0.0,
            rtol=1e-10,
            atol=1e-8,
        )
        invalid = ~finite_inputs | ~portfolio_valid_flag | ~conservation_valid
        conservation_failure = finite_inputs & portfolio_valid_flag & ~conservation_valid

        # Fixed precedence: invalid, forced liquidation, negative LFA, then
        # ordinary valid settlement.
        status[invalid & ~conservation_failure] = SETTLEMENT_INVALID_INPUT
        status[conservation_failure] = SETTLEMENT_CONSERVATION_FAILURE
        valid[invalid] = False

        forced = ~invalid & forced_liquidation_active
        status[forced] = SETTLEMENT_FORCED_LIQUIDATION

        # A purchase may not overdraw the account: block it when LFA is negative or would close negative.
        # A sale of illiquid assets is allowed while overdrawn: its proceeds repay the overdraft.
        overdrawn = base_lfa < 0.0
        # A sale whose proceeds net of the trade cost are not positive would deepen the overdraft: it stays blocked.
        sale_while_overdrawn = overdrawn & (counterfactual_ifa_flow < 0.0) & (counterfactual_lfa_flow > 0.0)
        negative_lfa = (
            ~invalid & ~forced & ~sale_while_overdrawn & (overdrawn | (base_lfa + counterfactual_lfa_flow < 0.0))
        )
        status[negative_lfa] = SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY

        eligible = ~invalid & ~forced & ~negative_lfa
        # Stage 4's counterfactual LFA flow already includes the fixed cost;
        # expose the transfer separately so the settled stock update charges it
        # exactly once on the liquid side.
        committed_lfa[eligible] = counterfactual_lfa_flow[eligible] + adjustment_cost[eligible]
        committed_ifa[eligible] = counterfactual_ifa_flow[eligible]
        committed_cost[eligible] = adjustment_cost[eligible]
        if waive_fixed_cost is not None:
            waived = np.asarray(waive_fixed_cost, dtype=bool)
            if waived.shape != shape:
                raise ValueError("Portfolio fixed-cost waiver must have one value per household.")
            committed_cost[eligible & waived] = 0.0
        status[eligible & inaction_flag] = SETTLEMENT_INACTION
        sale_while_overdrawn_eligible = eligible & sale_while_overdrawn

    closing_lfa = base_lfa + committed_lfa - committed_cost
    # A sale that exactly clears an overdraft can close at float noise below zero. Sell that noise too, so
    # the household closes at zero or above rather than in a (-tol, 0) residue.
    clearing_noise = sale_while_overdrawn_eligible & (closing_lfa < 0.0) & (closing_lfa > -_CLEARING_NOISE_TOL)
    if np.any(clearing_noise):
        bump = np.where(clearing_noise, -closing_lfa + np.spacing(np.abs(committed_lfa)), 0.0)
        committed_lfa = committed_lfa + bump
        committed_ifa = committed_ifa - bump
        closing_lfa = base_lfa + committed_lfa - committed_cost
    closing_ifa = base_ifa + committed_ifa
    closing_tfa = closing_lfa + closing_ifa
    return PortfolioSettlement(
        base_lfa=base_lfa.copy(),
        base_ifa=base_ifa.copy(),
        committed_lfa_flow=committed_lfa,
        committed_ifa_flow=committed_ifa,
        committed_adjustment_cost=committed_cost,
        closing_lfa=closing_lfa,
        closing_ifa=closing_ifa,
        closing_tfa=closing_tfa,
        settlement_enabled=np.full(shape, settlement_enabled, dtype=bool),
        settlement_valid_flag=valid,
        settlement_status=status,
    )
