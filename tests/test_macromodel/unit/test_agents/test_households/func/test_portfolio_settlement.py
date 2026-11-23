from dataclasses import replace

import numpy as np
import pytest

from macromodel.agents.households.func.portfolio_rebalancing import (
    PortfolioRebalancingResult,
    compute_portfolio_rebalancing,
)
from macromodel.agents.households.func.portfolio_settlement import (
    SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY,
    SETTLEMENT_CONSERVATION_FAILURE,
    SETTLEMENT_FORCED_LIQUIDATION,
    SETTLEMENT_INACTION,
    SETTLEMENT_INVALID_INPUT,
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


def test_negative_lfa_blocks_purchase():
    result = _settle(base_lfa=-5.0)
    np.testing.assert_allclose(result.closing_lfa, [-5.0])
    np.testing.assert_allclose(result.closing_ifa, [50.0])
    assert result.settlement_status[0] == SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY


def test_forced_liquidation_takes_precedence_over_a_rebalancing_trade():
    result = _settle(forced=True)
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
            rebalancing=rebalancing,
            settlement_enabled=True,
            forced_liquidation_active=np.array([False]),
        )


def test_purchase_that_would_overdraw_the_account_is_blocked_and_a_sale_is_not():
    blocked = _settle(base_lfa=5.0, lfa_flow=-8.0, ifa_flow=8.0)
    assert blocked.settlement_status[0] == SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY
    np.testing.assert_array_equal(blocked.closing_lfa, [5.0])
    np.testing.assert_array_equal(blocked.closing_ifa, [50.0])
    exact = _settle(base_lfa=8.0, lfa_flow=-8.0, ifa_flow=8.0)
    assert exact.settlement_status[0] == SETTLEMENT_SETTLED
    np.testing.assert_array_equal(exact.closing_lfa, [0.0])
    sale = _settle(base_lfa=1.0, lfa_flow=8.0, ifa_flow=-8.0)
    assert sale.settlement_status[0] == SETTLEMENT_SETTLED
    np.testing.assert_array_equal(sale.closing_lfa, [9.0])


def test_overdrawn_household_sells_ifa_and_repays_overdraft_net_of_cost():
    result = _settle(base_lfa=-5.0, base_ifa=50.0, lfa_flow=7.0, ifa_flow=-8.0, cost=1.0)
    assert result.settlement_status[0] == SETTLEMENT_SETTLED
    np.testing.assert_allclose(result.closing_lfa, [2.0])
    np.testing.assert_allclose(result.closing_ifa, [42.0])
    np.testing.assert_allclose(result.committed_adjustment_cost, [1.0])
    # TFA falls only by the trade cost.
    np.testing.assert_allclose(result.closing_tfa, [44.0])


def test_overdrawn_sale_smaller_than_overdraft_reduces_it():
    result = _settle(base_lfa=-20.0, lfa_flow=8.0, ifa_flow=-8.0)
    np.testing.assert_allclose(result.closing_lfa, [-12.0])
    np.testing.assert_allclose(result.closing_ifa, [42.0])


def test_overdrawn_household_purchase_remains_blocked():
    result = _settle(base_lfa=-5.0, lfa_flow=-8.0, ifa_flow=8.0)
    assert result.settlement_status[0] == SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY
    np.testing.assert_array_equal(result.closing_lfa, [-5.0])
    np.testing.assert_array_equal(result.closing_ifa, [50.0])


def test_overdrawn_household_without_trade_stays_blocked():
    result = _settle(base_lfa=-5.0, lfa_flow=0.0, ifa_flow=0.0, inaction=True)
    assert result.settlement_status[0] == SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY


def test_overdrawn_sale_still_requires_valid_inputs():
    assert _settle(base_lfa=-5.0, lfa_flow=8.0, ifa_flow=-8.0, valid=False).settlement_status[0] == (
        SETTLEMENT_INVALID_INPUT
    )


def test_non_overdrawn_behaviour_is_unchanged():
    # Same statuses and flows as before for base_lfa >= 0.
    for base_lfa, lfa_flow, ifa_flow, status in [
        (120.0, -8.0, 8.0, SETTLEMENT_SETTLED),
        (5.0, -8.0, 8.0, SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY),
        (1.0, 8.0, -8.0, SETTLEMENT_SETTLED),
        (0.0, 8.0, -8.0, SETTLEMENT_SETTLED),
    ]:
        result = _settle(base_lfa=base_lfa, lfa_flow=lfa_flow, ifa_flow=ifa_flow)
        assert result.settlement_status[0] == status


def test_upper_bound_sale_that_clears_overdraft_does_not_close_in_float_noise_below_zero():
    rng = np.random.default_rng(0)
    n = 20000
    scale = 10 ** rng.uniform(3, 8, n)
    lfa = -scale * rng.uniform(0.001, 0.5, n)
    ifa = scale * rng.uniform(0.6, 3.0, n)
    rebalancing = compute_portfolio_rebalancing(
        opening_tfa_scale=scale,
        post_return_lfa=lfa,
        post_return_ifa=ifa,
        target_illiquid_assets=np.zeros(n),
        target_tfa_base=lfa + ifa,
        portfolio_participates=np.ones(n, dtype=bool),
        phi_1=1.0,
        lambda_kappa=0.05,
        fixed_cost_share=0.002,
    )
    result = settle_portfolio_reallocation(
        base_lfa=lfa,
        base_ifa=ifa,
        rebalancing=rebalancing,
        settlement_enabled=True,
        forced_liquidation_active=np.zeros(n, dtype=bool),
    )
    sellers = (result.settlement_status == SETTLEMENT_SETTLED) & rebalancing.upper_bound_flag
    assert sellers.sum() > 1000
    raw_closing = lfa + rebalancing.counterfactual_lfa_flow + rebalancing.adjustment_cost - rebalancing.adjustment_cost
    # Without the guard a large share of these households closes at float noise below zero.
    assert ((raw_closing < 0.0) & sellers).sum() > 100
    assert np.all(result.closing_lfa[sellers] >= 0.0)
    np.testing.assert_allclose(result.committed_lfa_flow + result.committed_ifa_flow, 0.0, atol=1e-8)


def test_overdrawn_sale_with_non_positive_net_proceeds_is_blocked():
    for lfa_flow, cost in [(0.0, 8.0), (-1.0, 9.0)]:
        result = _settle(base_lfa=-5.0, lfa_flow=lfa_flow, ifa_flow=-8.0, cost=cost)
        assert result.settlement_status[0] == SETTLEMENT_BLOCKED_NEGATIVE_LIQUIDITY
        np.testing.assert_array_equal(result.closing_lfa, [-5.0])
        np.testing.assert_array_equal(result.closing_ifa, [50.0])


@pytest.mark.parametrize("scale", [1.0, 1e4, 1e8, 1e10])
def test_conservation_tolerates_ulp_scale_residual_on_population_expanded_flows(scale):
    ifa_flow = 0.37 * scale
    cost = 0.002 * scale
    lfa_flow = -ifa_flow - cost
    residual = 4 * np.spacing(abs(ifa_flow))  # a few ulps of the flow size
    result = _settle(
        base_lfa=5 * scale,
        base_ifa=scale,
        lfa_flow=lfa_flow - residual,
        ifa_flow=ifa_flow,
        cost=cost,
    )
    assert result.settlement_status[0] == SETTLEMENT_SETTLED
    assert result.settlement_valid_flag[0]
    np.testing.assert_allclose(result.committed_ifa_flow, [ifa_flow])


@pytest.mark.parametrize("scale", [1.0, 1e4, 1e8, 1e10])
def test_conservation_still_flags_a_genuine_imbalance(scale):
    for imbalance in (1.0, -1.0, 1e-3 * scale):
        result = _settle(
            base_lfa=5 * scale,
            base_ifa=scale,
            lfa_flow=-0.37 * scale - 0.002 * scale + imbalance,
            ifa_flow=0.37 * scale,
            cost=0.002 * scale,
        )
        assert result.settlement_status[0] == SETTLEMENT_CONSERVATION_FAILURE
        assert not result.settlement_valid_flag[0]
        np.testing.assert_array_equal(result.committed_ifa_flow, [0.0])


def test_conservation_absolute_floor_covers_tiny_flows_only():
    assert _settle(lfa_flow=-8.0 + 5e-9, ifa_flow=8.0).settlement_status[0] == SETTLEMENT_SETTLED
    assert _settle(lfa_flow=-8.0 + 1e-6, ifa_flow=8.0).settlement_status[0] == SETTLEMENT_CONSERVATION_FAILURE


def test_conservation_check_does_not_cancel_rule_outputs_on_population_expanded_stocks():
    rng = np.random.default_rng(1)
    n = 20000
    scale = 10 ** rng.uniform(8, 11, n)
    lfa = scale * rng.uniform(0.05, 2.0, n)
    ifa = scale * rng.uniform(0.05, 3.0, n)
    rebalancing = compute_portfolio_rebalancing(
        opening_tfa_scale=scale,
        post_return_lfa=lfa,
        post_return_ifa=ifa,
        target_illiquid_assets=0.5 * (lfa + ifa),
        target_tfa_base=lfa + ifa,
        portfolio_participates=np.ones(n, dtype=bool),
        phi_1=1.0,
        lambda_kappa=0.05,
        fixed_cost_share=0.002,
    )
    residual = np.abs(
        rebalancing.counterfactual_lfa_flow + rebalancing.counterfactual_ifa_flow + rebalancing.adjustment_cost
    )
    # The previous absolute tolerance (1e-8) would have cancelled these trades.
    assert (residual > 1e-8).sum() > 100
    result = settle_portfolio_reallocation(
        base_lfa=lfa,
        base_ifa=ifa,
        rebalancing=rebalancing,
        settlement_enabled=True,
        forced_liquidation_active=np.zeros(n, dtype=bool),
    )
    assert not np.any(result.settlement_status == SETTLEMENT_CONSERVATION_FAILURE)
    assert (result.committed_ifa_flow != 0.0).sum() > 1000
