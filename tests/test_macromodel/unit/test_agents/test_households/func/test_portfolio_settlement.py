from dataclasses import replace

import numpy as np
import pytest

from macromodel.agents.households.func.portfolio_rebalancing import PortfolioRebalancingResult
from macromodel.agents.households.func.portfolio_settlement import (
    SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY,
    SETTLEMENT_CONSERVATION_FAILURE,
    SETTLEMENT_FORCED_LIQUIDATION,
    SETTLEMENT_INACTION,
    SETTLEMENT_INVALID_INPUT,
    SETTLEMENT_NO_SURPLUS,
    SETTLEMENT_SETTLED,
    settle_portfolio_reallocation,
)
from macromodel.agents.households.func.post_liquidation_settlement import settle_post_liquidation


def _rebalancing(*, lfa_flow, ifa_flow, cost, valid=True, inaction=False):
    values = np.asarray(lfa_flow, dtype=float)
    zeros = np.zeros_like(values)
    return PortfolioRebalancingResult(
        portfolio_participates=np.ones(values.shape, dtype=bool),
        actual_illiquid_share=zeros,
        target_illiquid_assets=zeros,
        delta_tilde=zeros,
        kappa_star_tilde=zeros,
        kappa_tilde=zeros,
        desired_illiquid_adjustment=np.asarray(ifa_flow, dtype=float),
        adjustment_cost=np.asarray(cost, dtype=float),
        counterfactual_lfa_flow=values,
        counterfactual_ifa_flow=np.asarray(ifa_flow, dtype=float),
        inaction_flag=np.full(values.shape, inaction, dtype=bool),
        upper_bound_flag=np.zeros(values.shape, dtype=bool),
        lower_bound_flag=np.zeros(values.shape, dtype=bool),
        infeasible_interval_flag=np.zeros(values.shape, dtype=bool),
        no_financial_assets_flag=np.zeros(values.shape, dtype=bool),
        portfolio_valid_flag=np.full(values.shape, valid, dtype=bool),
    )


def _settle(
    *,
    base_lfa=120.0,
    base_ifa=50.0,
    surplus=20.0,
    lfa_flow=-8.0,
    ifa_flow=8.0,
    cost=0.0,
    enabled=True,
    forced=False,
    valid=True,
    inaction=False,
):
    return settle_portfolio_reallocation(
        base_lfa=np.array([base_lfa]),
        base_ifa=np.array([base_ifa]),
        investable_surplus=np.array([surplus]),
        rebalancing=_rebalancing(
            lfa_flow=[lfa_flow],
            ifa_flow=[ifa_flow],
            cost=[cost],
            valid=valid,
            inaction=inaction,
        ),
        settlement_enabled=enabled,
        forced_liquidation_active=np.array([forced]),
    )


def test_post_surplus_base_with_no_reallocation_is_unchanged():
    result = _settle(lfa_flow=0.0, ifa_flow=0.0, cost=0.0, inaction=True)
    np.testing.assert_allclose(result.closing_lfa, [120.0])
    np.testing.assert_allclose(result.closing_ifa, [50.0])
    np.testing.assert_allclose(result.closing_tfa, [170.0])
    assert result.settlement_status[0] == SETTLEMENT_INACTION


def test_reallocation_preserves_tfa_without_adjustment_cost():
    result = _settle()
    np.testing.assert_allclose(result.closing_lfa, [112.0])
    np.testing.assert_allclose(result.closing_ifa, [58.0])
    np.testing.assert_allclose(result.closing_tfa, [170.0])
    assert result.settlement_status[0] == SETTLEMENT_SETTLED


def test_adjustment_cost_is_charged_only_to_lfa():
    result = _settle(cost=2.0, lfa_flow=-10.0)
    np.testing.assert_allclose(result.closing_lfa, [110.0])
    np.testing.assert_allclose(result.closing_ifa, [58.0])
    np.testing.assert_allclose(result.closing_tfa, [168.0])


def test_non_positive_surplus_blocks_reallocation():
    result = _settle(surplus=0.0)
    np.testing.assert_allclose(result.closing_lfa, [120.0])
    np.testing.assert_allclose(result.closing_ifa, [50.0])
    assert result.settlement_status[0] == SETTLEMENT_NO_SURPLUS


def test_negative_lfa_blocks_reallocation():
    result = _settle(base_lfa=-5.0)
    np.testing.assert_allclose(result.closing_lfa, [-5.0])
    np.testing.assert_allclose(result.closing_ifa, [50.0])
    assert result.settlement_status[0] == SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY


def test_forced_liquidation_precedes_no_surplus():
    result = _settle(surplus=0.0, forced=True)
    assert result.settlement_status[0] == SETTLEMENT_FORCED_LIQUIDATION


def test_forced_liquidation_uses_post_liquidation_base_without_reapplying_sale():
    stage5 = settle_post_liquidation(
        base_lfa=np.array([100.0]),
        base_ifa=np.array([50.0]),
        planned_liquidation_total=np.array([8.0]),
        residual_shortfall_after_granted_credit=np.array([0.0]),
    )
    result = _settle(
        base_lfa=stage5.post_liquidation_lfa[0],
        base_ifa=stage5.post_liquidation_ifa[0],
        forced=bool(stage5.settled_liquidation_total[0] > 0.0),
        surplus=0.0,
    )

    np.testing.assert_allclose(result.closing_lfa, [108.0])
    np.testing.assert_allclose(result.closing_ifa, [42.0])
    np.testing.assert_allclose(result.closing_tfa, [150.0])
    np.testing.assert_allclose(result.committed_lfa_flow, [0.0])
    np.testing.assert_allclose(result.committed_ifa_flow, [0.0])


def test_invalid_input_precedes_forced_liquidation():
    result = _settle(forced=True, valid=False)
    assert result.settlement_status[0] == SETTLEMENT_INVALID_INPUT
    assert not result.settlement_valid_flag[0]


def test_conservation_failure_commits_no_partial_flow():
    result = _settle(lfa_flow=-7.0, ifa_flow=8.0, cost=0.0)
    np.testing.assert_allclose(result.committed_lfa_flow, [0.0])
    np.testing.assert_allclose(result.committed_ifa_flow, [0.0])
    np.testing.assert_allclose(result.closing_lfa, [120.0])
    np.testing.assert_allclose(result.closing_ifa, [50.0])
    assert result.settlement_status[0] == SETTLEMENT_CONSERVATION_FAILURE


def test_disabled_settlement_preserves_base_even_with_invalid_inputs():
    result = _settle(enabled=False, valid=False)
    np.testing.assert_allclose(result.closing_lfa, [120.0])
    np.testing.assert_allclose(result.closing_ifa, [50.0])
    assert result.settlement_valid_flag[0]


def test_shape_validation_covers_inaction_flag():
    rebalancing = replace(
        _rebalancing(lfa_flow=[-8.0], ifa_flow=[8.0], cost=[0.0]),
        inaction_flag=np.array([False, False]),
    )

    with pytest.raises(ValueError, match="one value per household"):
        settle_portfolio_reallocation(
            base_lfa=np.array([120.0]),
            base_ifa=np.array([50.0]),
            investable_surplus=np.array([20.0]),
            rebalancing=rebalancing,
            settlement_enabled=True,
            forced_liquidation_active=np.array([False]),
        )
