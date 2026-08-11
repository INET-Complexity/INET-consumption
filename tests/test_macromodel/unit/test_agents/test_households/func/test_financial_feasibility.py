import numpy as np

from macromodel.agents.households.func.financial_feasibility import (
    HouseholdFinancialFeasibility,
    PostGrantFeasiblePlan,
    PreGrantFeasiblePlan,
)


def test__pre_and_post_grant_carriers_remain_distinct_and_pure():
    resolver = HouseholdFinancialFeasibility()
    liquid = np.array([4.0, 10.0])
    pre = resolver.build_pre_grant_plan(
        liquidity_shortfall_before_repair=np.array([10.0, 8.0]),
        funded_from_liquid_assets=liquid,
        residual_shortfall_after_lfa=np.array([6.0, 0.0]),
    )
    pre = resolver.with_credit_requested(pre, np.array([3.0, 0.0]))
    pre = resolver.with_planned_liquidation(
        pre,
        planned_liquidation_total=np.array([3.0, 0.0]),
        available_illiquid_assets=np.array([20.0, 20.0]),
    )
    post = resolver.build_post_grant_plan(pre, credit_granted=np.array([2.0, 0.0]))

    assert isinstance(pre, PreGrantFeasiblePlan)
    assert isinstance(post, PostGrantFeasiblePlan)
    np.testing.assert_allclose(pre.credit_requested, [3.0, 0.0])
    np.testing.assert_allclose(post.credit_granted, [2.0, 0.0])
    np.testing.assert_allclose(post.funded_from_liquid_assets, liquid)
    np.testing.assert_allclose(post.residual_shortfall_after_granted_credit, [1.0, 0.0])


def test__planned_liquidation_is_capped_without_mutating_inputs():
    resolver = HouseholdFinancialFeasibility()
    planned = np.array([8.0])
    available = np.array([3.0])
    pre = resolver.build_pre_grant_plan(
        liquidity_shortfall_before_repair=np.array([10.0]),
        funded_from_liquid_assets=np.array([2.0]),
        residual_shortfall_after_lfa=np.array([8.0]),
    )
    updated = resolver.with_planned_liquidation(
        pre,
        planned_liquidation_total=planned,
        available_illiquid_assets=available,
    )

    np.testing.assert_allclose(updated.planned_liquidation_total, [3.0])
    np.testing.assert_allclose(planned, [8.0])
    np.testing.assert_allclose(available, [3.0])
