"""Settled Stage 4 portfolio reallocation on top of the wealth update."""

from dataclasses import dataclass

import numpy as np

from macromodel.agents.households.func.portfolio_rebalancing import PortfolioRebalancingResult

SETTLEMENT_DISABLED = 0.0
SETTLEMENT_NO_SURPLUS = 1.0
SETTLEMENT_INACTION = 2.0
SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY = 3.0
SETTLEMENT_FORCED_LIQUIDATION = 4.0
SETTLEMENT_INVALID_INPUT = 5.0
SETTLEMENT_CONSERVATION_FAILURE = 6.0
SETTLEMENT_SETTLED = 7.0


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
    investable_surplus: np.ndarray,
    rebalancing: PortfolioRebalancingResult,
    settlement_enabled: bool,
    forced_liquidation_active: np.ndarray,
) -> PortfolioSettlement:
    """Commit Stage 4 reallocation flows without allocating saving twice.

    ``base_lfa`` is the current liquid balance after the ordinary wealth update;
    it already includes positive saving. The Stage 4 flows therefore only move
    wealth between LFA and IFA and charge adjustment cost on LFA.
    """
    base_lfa = np.asarray(base_lfa, dtype=float)
    base_ifa = np.asarray(base_ifa, dtype=float)
    investable_surplus = np.asarray(investable_surplus, dtype=float)
    forced_liquidation_active = np.asarray(forced_liquidation_active, dtype=bool)
    arrays = (
        base_lfa,
        base_ifa,
        investable_surplus,
        forced_liquidation_active,
        rebalancing.portfolio_valid_flag,
        rebalancing.counterfactual_lfa_flow,
        rebalancing.counterfactual_ifa_flow,
        rebalancing.adjustment_cost,
    )
    shape = base_lfa.shape
    if any(values.shape != shape for values in arrays):
        raise ValueError("Portfolio settlement inputs must have one value per household.")

    committed_lfa = np.zeros(shape, dtype=float)
    committed_ifa = np.zeros(shape, dtype=float)
    committed_cost = np.zeros(shape, dtype=float)
    status = np.full(shape, SETTLEMENT_DISABLED if not settlement_enabled else SETTLEMENT_SETTLED)
    valid = np.ones(shape, dtype=bool)

    if settlement_enabled:
        finite_inputs = (
            np.isfinite(base_lfa)
            & np.isfinite(base_ifa)
            & np.isfinite(investable_surplus)
            & np.isfinite(rebalancing.counterfactual_lfa_flow)
            & np.isfinite(rebalancing.counterfactual_ifa_flow)
            & np.isfinite(rebalancing.adjustment_cost)
        )
        conservation_valid = np.isclose(
            rebalancing.counterfactual_lfa_flow + rebalancing.counterfactual_ifa_flow + rebalancing.adjustment_cost,
            0.0,
            rtol=1e-10,
            atol=1e-8,
        )
        invalid = ~finite_inputs | ~rebalancing.portfolio_valid_flag | ~conservation_valid
        conservation_failure = finite_inputs & rebalancing.portfolio_valid_flag & ~conservation_valid

        # Fixed precedence: invalid, forced liquidation, no surplus, negative
        # LFA, then ordinary valid settlement.
        status[invalid & ~conservation_failure] = SETTLEMENT_INVALID_INPUT
        status[conservation_failure] = SETTLEMENT_CONSERVATION_FAILURE
        valid[invalid] = False

        forced = ~invalid & forced_liquidation_active
        status[forced] = SETTLEMENT_FORCED_LIQUIDATION

        no_surplus = ~invalid & ~forced & (investable_surplus <= 0.0)
        status[no_surplus] = SETTLEMENT_NO_SURPLUS

        negative_lfa = ~invalid & ~forced & ~no_surplus & (base_lfa < 0.0)
        status[negative_lfa] = SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY

        eligible = ~invalid & ~forced & ~no_surplus & ~negative_lfa
        # Stage 4's counterfactual LFA flow already includes the fixed cost;
        # expose the transfer separately so the settled stock update charges it
        # exactly once on the liquid side.
        committed_lfa[eligible] = rebalancing.counterfactual_lfa_flow[eligible] + rebalancing.adjustment_cost[eligible]
        committed_ifa[eligible] = rebalancing.counterfactual_ifa_flow[eligible]
        committed_cost[eligible] = rebalancing.adjustment_cost[eligible]
        status[eligible & rebalancing.inaction_flag] = SETTLEMENT_INACTION

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
