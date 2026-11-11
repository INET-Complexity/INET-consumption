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
        post,
        consumption_before_floor=np.array([5.0]),
        subsistence_floor=np.array([4.0]),
        cash_rent=np.array([0.0]),
        resources=np.array([1.0]),
        scheduled_consumer_service=np.array([0.0]),
        scheduled_mortgage_service=np.array([0.0]),
        investment=np.array([0.0]),
    )

    assert post.consumption_after_floor is None
    np.testing.assert_allclose(settled.consumption_after_floor, [4.0])
    np.testing.assert_allclose(settled.consumption_cut_amount, [1.0])
    np.testing.assert_allclose(settled.remaining_subsistence_shortfall, [3.0])
    np.testing.assert_allclose(settled.floor_support, [3.0])


def _settle_floor(**inputs):
    """Settle one-household-per-entry inputs; F=80, R=20, O=10+30, G=120, V=15 unless overridden."""
    values = {
        "consumption_before_floor": 120.0,
        "subsistence_floor": 80.0,
        "cash_rent": 20.0,
        "scheduled_consumer_service": 10.0,
        "scheduled_mortgage_service": 30.0,
        "investment": 15.0,
        **inputs,
    }
    resources = np.atleast_1d(np.asarray(values.pop("resources"), dtype=float))
    plan = PostGrantFeasiblePlan(
        credit_granted=np.zeros_like(resources),
        credit_rationing_gap=np.zeros_like(resources),
        planned_liquidation_total=np.zeros_like(resources),
        residual_shortfall_after_granted_credit=np.zeros_like(resources),
    )
    return HouseholdFinancialFeasibility().settle_consumption_floor(
        plan, resources=resources, **{name: np.full_like(resources, value) for name, value in values.items()}
    )


def test__floor_settlement_pays_in_order_floor_and_rent_then_service_then_goods_then_investment():
    # F + R = 100, O = 40, goods above the floor = 40, V = 15: full plan costs 195.
    resources = np.array([60.0, 100.0, 125.0, 140.0, 160.0, 180.0, 190.0, 195.0, 500.0])
    settled = _settle_floor(resources=resources)

    np.testing.assert_allclose(settled.floor_support, [40, 0, 0, 0, 0, 0, 0, 0, 0])
    # Unpaid service is suspended on consumer credit first, then on the mortgage.
    np.testing.assert_allclose(settled.unpaid_consumer_service, [10, 10, 10, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(settled.unpaid_mortgage_service, [30, 30, 5, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(settled.consumption_after_floor, [80, 80, 80, 80, 100, 120, 120, 120, 120])
    np.testing.assert_allclose(settled.investment_after_floor, [0, 0, 0, 0, 0, 0, 10, 15, 15])
    np.testing.assert_allclose(settled.consumption_cut_amount, [40, 40, 40, 40, 20, 0, 0, 0, 0])
    np.testing.assert_array_equal(settled.floor_binding, [True, True, True, True, True, False, False, False, False])
    np.testing.assert_allclose(
        settled.remaining_subsistence_shortfall,
        settled.floor_support + settled.unpaid_consumer_service + settled.unpaid_mortgage_service,
    )
    # Every euro of resources and support is spent in the order above, or left over.
    service_paid = 40.0 - settled.unpaid_consumer_service - settled.unpaid_mortgage_service
    spent = settled.consumption_after_floor + 20.0 + service_paid + settled.investment_after_floor
    np.testing.assert_allclose(spent, np.minimum(resources + settled.floor_support, 195.0))


def test__floor_support_covers_the_goods_floor_and_rent_only():
    # Resources cover the floor and rent exactly: nothing else is supported.
    for shortfall_only_in in (
        {"investment": 500.0},
        {"scheduled_consumer_service": 500.0},
        {"scheduled_mortgage_service": 500.0},
        {"consumption_before_floor": 500.0},
    ):
        settled = _settle_floor(resources=100.0, **shortfall_only_in)
        np.testing.assert_allclose(settled.floor_support, [0.0])
        np.testing.assert_allclose(settled.consumption_after_floor, [80.0])
        np.testing.assert_allclose(settled.investment_after_floor, [0.0])
    # A household on support pays no debt service, buys nothing above the floor and does not invest.
    supported = _settle_floor(resources=70.0)
    np.testing.assert_allclose(supported.floor_support, [30.0])
    np.testing.assert_allclose(supported.unpaid_consumer_service + supported.unpaid_mortgage_service, [40.0])
    np.testing.assert_allclose(supported.consumption_after_floor, [80.0])
    np.testing.assert_allclose(supported.investment_after_floor, [0.0])


def test__floor_is_compulsory_and_funded_from_own_resources_first():
    settled = _settle_floor(resources=np.array([500.0, 90.0, -10.0]), consumption_before_floor=50.0)
    np.testing.assert_allclose(settled.consumption_after_floor, [80.0, 80.0, 80.0])
    # Overdrawn resources (third household) raise support; they are not repaid by it.
    np.testing.assert_allclose(settled.floor_support, [0.0, 10.0, 110.0])
    np.testing.assert_allclose(settled.consumption_cut_amount, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(settled.floor_binding, [True, True, True])


def test__floor_settlement_rejects_non_finite_resources_and_unequal_shapes():
    with pytest.raises(ValueError, match="resources must be finite"):
        _settle_floor(resources=np.array([np.nan]))
    plan = PostGrantFeasiblePlan(
        credit_granted=np.zeros(2),
        credit_rationing_gap=np.zeros(2),
        planned_liquidation_total=np.zeros(2),
        residual_shortfall_after_granted_credit=np.zeros(2),
    )
    with pytest.raises(ValueError, match="equal-length household vectors"):
        HouseholdFinancialFeasibility().settle_consumption_floor(
            plan,
            consumption_before_floor=np.zeros(2),
            subsistence_floor=np.zeros(2),
            cash_rent=np.zeros(3),
            resources=np.zeros(2),
            scheduled_consumer_service=np.zeros(2),
            scheduled_mortgage_service=np.zeros(2),
            investment=np.zeros(2),
        )


def test__floor_settlement_exempt_resources_keep_support_and_are_spent():
    # F + R = 100. Each household receives a transfer T = 30, included in A.
    # Household 0: A = 60 + 30. Counted: S = 10. Exempt: S = 40, as without T.
    # Household 1: A = 150 + 30, never supported: the exemption changes nothing.
    resources = np.array([90.0, 180.0])
    counted = _settle_floor(resources=resources)
    exempt = _settle_floor(resources=resources, support_exempt_resources=30.0)

    np.testing.assert_allclose(counted.floor_support, [10.0, 0.0])
    np.testing.assert_allclose(exempt.floor_support, [40.0, 0.0])
    # The exempt household spends the transfer after the floor and rent: A1 = 90 + 40 - 100 = 30 of service.
    np.testing.assert_allclose(exempt.unpaid_consumer_service + exempt.unpaid_mortgage_service, [10.0, 0.0])
    np.testing.assert_allclose(counted.unpaid_consumer_service + counted.unpaid_mortgage_service, [40.0, 0.0])
    for name in ("consumption_after_floor", "investment_after_floor", "unpaid_mortgage_service"):
        np.testing.assert_allclose(getattr(exempt, name)[1], getattr(counted, name)[1])


@pytest.mark.parametrize("bad", [-1.0, np.nan])
def test__floor_settlement_rejects_invalid_exempt_resources(bad):
    with pytest.raises(ValueError, match="Support-exempt"):
        _settle_floor(resources=np.array([90.0]), support_exempt_resources=bad)
