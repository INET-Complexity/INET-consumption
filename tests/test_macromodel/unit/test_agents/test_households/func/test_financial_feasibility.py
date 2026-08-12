import numpy as np
import pytest

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


def test__executable_liquidation_is_reserved_from_pre_stage4_ifa_and_reconciles_residual():
    resolver = HouseholdFinancialFeasibility()
    pre = resolver.build_pre_grant_plan(
        liquidity_shortfall_before_repair=np.array([10.0]),
        funded_from_liquid_assets=np.array([2.0]),
        residual_shortfall_after_lfa=np.array([8.0]),
    )
    pre = resolver.with_credit_requested(pre, np.array([3.0]))
    pre = resolver.with_planned_liquidation(
        pre,
        planned_liquidation_total=np.array([5.0]),
        available_illiquid_assets=np.array([10.0]),
    )
    post = resolver.build_post_grant_plan(pre, credit_granted=np.array([3.0]))

    reserved = resolver.reserve_executable_liquidation(post, available_pre_stage4_ifa=np.array([2.0]))

    np.testing.assert_allclose(reserved.reserved_liquidation_total, [2.0])
    np.testing.assert_allclose(reserved.planned_liquidation_total, [2.0])
    np.testing.assert_allclose(reserved.residual_shortfall_after_granted_credit, [3.0])
    with np.testing.assert_raises_regex(RuntimeError, "already been reserved"):
        resolver.reserve_executable_liquidation(reserved, available_pre_stage4_ifa=np.array([2.0]))


def test__carriers_are_frozen_defensive_snapshots():
    resolver = HouseholdFinancialFeasibility()
    source = np.array([4.0])
    pre = resolver.build_pre_grant_plan(
        liquidity_shortfall_before_repair=source,
        funded_from_liquid_assets=np.array([1.0]),
        residual_shortfall_after_lfa=np.array([3.0]),
    )

    source[0] = 99.0
    np.testing.assert_allclose(pre.liquidity_shortfall_before_repair, [4.0])
    with pytest.raises(ValueError):
        pre.liquidity_shortfall_before_repair[0] = 0.0
    with pytest.raises(Exception):
        pre.credit_requested = np.zeros(1)


def test__consumption_floor_policy_is_pure_and_keeps_the_carrier_immutable():
    resolver = HouseholdFinancialFeasibility()
    pre = resolver.build_pre_grant_plan(
        liquidity_shortfall_before_repair=np.array([10.0]),
        funded_from_liquid_assets=np.array([2.0]),
        residual_shortfall_after_lfa=np.array([8.0]),
    )
    pre = resolver.with_credit_requested(pre, np.array([3.0]))
    pre = resolver.with_planned_liquidation(
        pre, planned_liquidation_total=np.array([1.0]), available_illiquid_assets=np.array([1.0])
    )
    post = resolver.build_post_grant_plan(pre, credit_granted=np.array([3.0]))

    settled = resolver.settle_consumption_floor(
        post, consumption_before_floor=np.array([5.0]), subsistence_floor=np.array([4.0])
    )

    assert post.consumption_after_floor is None
    np.testing.assert_allclose(settled.consumption_after_floor, [4.0])
    np.testing.assert_allclose(settled.consumption_cut_amount, [1.0])
    np.testing.assert_allclose(settled.remaining_subsistence_shortfall, [3.0])
