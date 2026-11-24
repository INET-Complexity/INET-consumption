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
    support_enabled = values.pop("support_enabled", True)
    plan = PostGrantFeasiblePlan(
        credit_granted=np.zeros_like(resources),
        credit_rationing_gap=np.zeros_like(resources),
        planned_liquidation_total=np.zeros_like(resources),
        residual_shortfall_after_granted_credit=np.zeros_like(resources),
    )
    return HouseholdFinancialFeasibility().settle_consumption_floor(
        plan,
        resources=resources,
        support_enabled=support_enabled,
        **{name: np.full_like(resources, value) for name, value in values.items()},
    )


def test__floor_settlement_pays_in_order_floor_and_rent_then_service_then_goods_then_investment():
    # Exact up to the plan round-down, 1e-12 x (|A| + F + R), hence atol=1e-9.
    # F + R = 100, O = 40, goods above the floor = 40, V = 15: full plan costs 195.
    resources = np.array([60.0, 100.0, 125.0, 140.0, 160.0, 180.0, 190.0, 195.0, 500.0])
    settled = _settle_floor(resources=resources)

    np.testing.assert_allclose(settled.floor_support, [40, 0, 0, 0, 0, 0, 0, 0, 0], atol=1e-9)
    # Unpaid service is suspended on consumer credit first, then on the mortgage.
    np.testing.assert_allclose(settled.unpaid_consumer_service, [10, 10, 10, 0, 0, 0, 0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(settled.unpaid_mortgage_service, [30, 30, 5, 0, 0, 0, 0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(settled.consumption_after_floor, [80, 80, 80, 80, 100, 120, 120, 120, 120], atol=1e-9)
    np.testing.assert_allclose(settled.investment_after_floor, [0, 0, 0, 0, 0, 0, 10, 15, 15], atol=1e-9)
    np.testing.assert_allclose(settled.consumption_cut_amount, [40, 40, 40, 40, 20, 0, 0, 0, 0], atol=1e-9)
    # At 180 the plan affords desired goods exactly: the round-down trims them by about 1e-10 but is not binding.
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
        np.testing.assert_allclose(settled.floor_support, [0.0], atol=1e-9)
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


_LEGACY_GRID_RESOURCES = np.array([-10.0, 0.0, 50.0, 100.0, 125.0, 160.0, 195.0, 500.0])
_LEGACY_GRID_EXEMPT = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# Characterisation of the legacy Stage 5 allocator (F=80, R=20, O=10+30, G=120, V=15), recorded before
# the minimum-income ``support_enabled`` switch was added. A < 0 is covered by support: S = F + R - (A - E).
_LEGACY_GRID_EXPECTED = {
    "consumption_after_floor": [80.0, 80.0, 80.0, 80.0, 80.0, 100.0, 120.0, 120.0],
    "consumption_cut_amount": [40.0, 40.0, 40.0, 40.0, 40.0, 20.0, 0.0, 0.0],
    "remaining_subsistence_shortfall": [150.0, 140.0, 90.0, 40.0, 15.0, 0.0, 0.0, 0.0],
    "floor_support": [110.0, 100.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "unpaid_consumer_service": [10.0, 10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0],
    "unpaid_mortgage_service": [30.0, 30.0, 20.0, 30.0, 5.0, 0.0, 0.0, 0.0],
    "investment_after_floor": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 15.0],
    "floor_binding": [True, True, True, True, True, True, False, False],
    "residual_shortfall_before_floor": [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 5.0],
}


def _settle_legacy_grid(**switch):
    n = _LEGACY_GRID_RESOURCES.size
    plan = PostGrantFeasiblePlan(
        credit_granted=np.zeros(n),
        credit_rationing_gap=np.zeros(n),
        planned_liquidation_total=np.zeros(n),
        residual_shortfall_after_granted_credit=np.array([0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 5.0]),
    )
    return HouseholdFinancialFeasibility().settle_consumption_floor(
        plan,
        consumption_before_floor=np.full(n, 120.0),
        subsistence_floor=np.full(n, 80.0),
        cash_rent=np.full(n, 20.0),
        resources=_LEGACY_GRID_RESOURCES,
        scheduled_consumer_service=np.full(n, 10.0),
        scheduled_mortgage_service=np.full(n, 30.0),
        investment=np.full(n, 15.0),
        support_exempt_resources=_LEGACY_GRID_EXEMPT,
        **switch,
    )


def test__legacy_floor_settlement_characterisation():
    settled = _settle_legacy_grid()
    for name, expected in _LEGACY_GRID_EXPECTED.items():
        # Exact up to the plan round-down (PLAN_ROUND_DOWN x resources) on goods above the floor and investment.
        np.testing.assert_allclose(getattr(settled, name), expected, rtol=0.0, atol=1e-9, err_msg=name)


def test__support_enabled_true_is_the_legacy_allocation_bit_for_bit():
    default = _settle_legacy_grid()
    explicit = _settle_legacy_grid(support_enabled=True)
    for name in _LEGACY_GRID_EXPECTED:
        np.testing.assert_array_equal(getattr(explicit, name), getattr(default, name), err_msg=name)
    # Support covers rent and the floor, so nothing is left unpaid in the legacy allocation.
    np.testing.assert_array_equal(default.unpaid_rent, 0.0)
    np.testing.assert_array_equal(default.unmet_floor, 0.0)
    # Negative resources are covered by support in legacy: no uncovered deficit is recorded.
    np.testing.assert_array_equal(default.resource_deficit, 0.0)


# Minimum-income allocation (PLAN.md section 5.4): F=80, R=20, O=10+30, G=120, V=15, no support.
_NO_SUPPORT_RESOURCES = np.array([-10.0, 0.0, 10.0, 20.0, 60.0, 100.0, 125.0, 160.0, 195.0, 500.0])


def test__without_support_rent_is_paid_first_then_floor_then_service_goods_and_investment():
    settled = _settle_floor(resources=_NO_SUPPORT_RESOURCES, support_enabled=False)

    np.testing.assert_array_equal(settled.floor_support, 0.0)
    np.testing.assert_allclose(settled.unpaid_rent, [20, 20, 10, 0, 0, 0, 0, 0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(settled.unmet_floor, [80, 80, 80, 80, 40, 0, 0, 0, 0, 0], atol=1e-9)
    np.testing.assert_array_equal(settled.resource_deficit, [10, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(settled.unpaid_consumer_service, [10, 10, 10, 10, 10, 10, 10, 0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(settled.unpaid_mortgage_service, [30, 30, 30, 30, 30, 30, 5, 0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(settled.consumption_after_floor, [0, 0, 0, 0, 40, 80, 80, 100, 120, 120], atol=1e-9)
    np.testing.assert_allclose(settled.investment_after_floor, [0, 0, 0, 0, 0, 0, 0, 0, 15, 15], atol=1e-9)
    np.testing.assert_allclose(settled.consumption_cut_amount, 120.0 - settled.consumption_after_floor, atol=1e-9)
    # The shortfall handed on is unpaid debt service only: there is no support in it.
    np.testing.assert_allclose(
        settled.remaining_subsistence_shortfall, settled.unpaid_consumer_service + settled.unpaid_mortgage_service
    )
    # Every euro of the non-negative resources is spent in the order above, or left over; nothing more.
    rent_paid = 20.0 - settled.unpaid_rent
    service_paid = 40.0 - settled.unpaid_consumer_service - settled.unpaid_mortgage_service
    spent = rent_paid + settled.consumption_after_floor + service_paid + settled.investment_after_floor
    np.testing.assert_allclose(spent, np.minimum(np.maximum(_NO_SUPPORT_RESOURCES, 0.0), 195.0))


def test__negative_resources_are_covered_by_support_in_legacy_but_by_nobody_without_support():
    legacy = _settle_floor(resources=np.array([-10.0]))
    np.testing.assert_allclose(legacy.floor_support, [110.0])  # S = F + R - A
    without = _settle_floor(resources=np.array([-10.0]), support_enabled=False)
    # A+ = 0: nothing is paid; the overdraft stays in the household's own cash ledger.
    np.testing.assert_allclose(without.floor_support, [0.0])
    np.testing.assert_allclose(without.unpaid_rent, [20.0])
    np.testing.assert_allclose(without.unmet_floor, [80.0])
    np.testing.assert_allclose(without.consumption_after_floor, [0.0])
    np.testing.assert_allclose(without.remaining_subsistence_shortfall, [40.0])
    # The uncovered negative part of the resources is visible: max(-A, 0).
    np.testing.assert_array_equal(without.resource_deficit, [10.0])


def test__without_support_the_funded_floor_tops_up_goods_and_the_unmet_part_is_visible():
    # Desired goods G = 50 < F = 80. Enough resources: the floor is bought in full from own resources.
    settled = _settle_floor(resources=np.array([500.0, 70.0]), consumption_before_floor=50.0, support_enabled=False)
    np.testing.assert_allclose(settled.consumption_after_floor, [80.0, 50.0])
    np.testing.assert_allclose(settled.unmet_floor, [0.0, 30.0])
    np.testing.assert_allclose(settled.unpaid_rent, [0.0, 0.0])
    # Goods above desired come only from the funded floor.
    top_up = np.maximum(settled.consumption_after_floor - 50.0, 0.0)
    assert np.all(top_up <= 80.0 - settled.unmet_floor)


def test__goods_floor_off_never_raises_goods_above_desired():
    rng = np.random.default_rng(3)
    resources = rng.uniform(-50.0, 400.0, 200)
    desired = rng.uniform(0.0, 200.0, 200)
    plan = PostGrantFeasiblePlan(
        credit_granted=np.zeros(200),
        credit_rationing_gap=np.zeros(200),
        planned_liquidation_total=np.zeros(200),
        residual_shortfall_after_granted_credit=np.zeros(200),
    )
    settled = HouseholdFinancialFeasibility().settle_consumption_floor(
        plan,
        consumption_before_floor=desired,
        subsistence_floor=np.zeros(200),
        cash_rent=np.full(200, 20.0),
        resources=resources,
        scheduled_consumer_service=np.full(200, 10.0),
        scheduled_mortgage_service=np.full(200, 30.0),
        investment=np.full(200, 15.0),
        support_enabled=False,
    )
    assert np.all(settled.consumption_after_floor <= desired)
    np.testing.assert_array_equal(settled.unmet_floor, 0.0)
    np.testing.assert_array_equal(settled.floor_support, 0.0)
    np.testing.assert_allclose(settled.unpaid_rent, 20.0 - np.minimum(20.0, np.maximum(resources, 0.0)))


def test__support_exempt_resources_play_no_role_without_support():
    plain = _settle_floor(resources=np.array([90.0]), support_enabled=False)
    exempt = _settle_floor(resources=np.array([90.0]), support_exempt_resources=30.0, support_enabled=False)
    for name in ("consumption_after_floor", "unpaid_rent", "unmet_floor", "unpaid_mortgage_service", "floor_support"):
        np.testing.assert_array_equal(getattr(exempt, name), getattr(plain, name), err_msg=name)


def test__support_switch_must_be_a_boolean():
    with pytest.raises(TypeError, match="support_enabled"):
        _settle_legacy_grid(support_enabled=np.array([True, False]))


@pytest.mark.parametrize("support_enabled", [True, False])
def test__overdraft_interest_is_debt_service_paid_before_goods_above_the_floor(support_enabled):
    # F + R = 100, consumer + mortgage service = 40, overdraft interest = 5, goods above the floor = 40, V = 15.
    resources = np.array([100.0, 130.0, 142.0, 145.0, 165.0, 185.0, 200.0, 500.0])
    settled = _settle_floor(resources=resources, scheduled_overdraft_interest=5.0, support_enabled=support_enabled)
    without = _settle_floor(resources=resources, support_enabled=support_enabled)

    np.testing.assert_array_equal(settled.scheduled_overdraft_interest, 5.0)
    # Consumer and mortgage suspension are unchanged; the overdraft interest goes unpaid first.
    np.testing.assert_array_equal(settled.unpaid_consumer_service, without.unpaid_consumer_service)
    np.testing.assert_array_equal(settled.unpaid_mortgage_service, without.unpaid_mortgage_service)
    np.testing.assert_allclose(settled.unpaid_overdraft_interest, [5, 5, 3, 0, 0, 0, 0, 0], atol=1e-9)
    np.testing.assert_allclose(settled.consumption_after_floor, [80, 80, 80, 80, 100, 120, 120, 120], rtol=1e-9)
    np.testing.assert_allclose(settled.investment_after_floor, [0, 0, 0, 0, 0, 0, 15, 15], atol=1e-9)
    # The unpaid interest is recorded, not handed to support or to the subsistence shortfall.
    np.testing.assert_array_equal(settled.floor_support, without.floor_support)
    np.testing.assert_array_equal(settled.remaining_subsistence_shortfall, without.remaining_subsistence_shortfall)
    service_paid = 45.0 - settled.unpaid_consumer_service - settled.unpaid_mortgage_service
    spent = settled.consumption_after_floor + 20.0 + service_paid - settled.unpaid_overdraft_interest
    np.testing.assert_allclose(spent + settled.investment_after_floor, np.minimum(resources, 200.0), rtol=1e-9)


def test__no_overdraft_interest_leaves_the_allocation_unchanged():
    resources = np.array([60.0, 100.0, 125.0, 140.0, 195.0, 500.0])
    default = _settle_floor(resources=resources)
    explicit = _settle_floor(resources=resources, scheduled_overdraft_interest=0.0)
    for name in ("consumption_after_floor", "investment_after_floor", "unpaid_consumer_service", "floor_support"):
        np.testing.assert_array_equal(getattr(explicit, name), getattr(default, name), err_msg=name)
    np.testing.assert_array_equal(default.unpaid_overdraft_interest, 0.0)


@pytest.mark.parametrize("support_enabled", [True, False])
def test__a_plan_that_spends_everything_is_rounded_down_and_never_overspends(support_enabled):
    # Resources that buy exactly the floor and rent (support case), floor, rent, service and part of the goods,
    # or the whole plan; and resources of zero.
    resources = np.array([0.0, 100.0, 150.0, 180.0, 195.0, 1e7 + 0.1, 3.3e9 / 7.0])
    settled = _settle_floor(
        resources=resources,
        consumption_before_floor=np.array([120.0, 120.0, 120.0, 120.0, 120.0, 1e7, 3.3e9 / 7.0]),
        investment=np.array([15.0, 15.0, 15.0, 15.0, 15.0, 0.3, 0.0]),
        support_enabled=support_enabled,
    )
    service_paid = 40.0 - settled.unpaid_consumer_service - settled.unpaid_mortgage_service
    rent_paid = 20.0 - settled.unpaid_rent
    left = (
        resources
        + settled.floor_support
        - (rent_paid + service_paid + settled.consumption_after_floor + settled.investment_after_floor)
    )
    # Everything is allocated from A - m, m = 1e-12 (|A| + F + R): m is left over, never overspent.
    margin = 1e-12 * (np.abs(resources) + 100.0)
    spends_everything = resources > 0.0 if not support_enabled else np.ones(resources.shape, dtype=bool)
    assert np.all(left[spends_everything] > 0.0)
    np.testing.assert_allclose(left[spends_everything], margin[spends_everything], rtol=1e-3)
    # The round-down moves support and unpaid amounts by at most m.
    plain = _settle_floor(
        resources=resources + margin,
        consumption_before_floor=np.array([120.0, 120.0, 120.0, 120.0, 120.0, 1e7, 3.3e9 / 7.0]),
        investment=np.array([15.0, 15.0, 15.0, 15.0, 15.0, 0.3, 0.0]),
        support_enabled=support_enabled,
    )
    for name in ("floor_support", "unpaid_rent", "unmet_floor", "unpaid_consumer_service", "unpaid_mortgage_service"):
        np.testing.assert_allclose(getattr(settled, name), getattr(plain, name), rtol=0.0, atol=2 * margin.max())


@pytest.mark.parametrize("support_enabled", [True, False])
def test__opening_overdraft_is_repaid_after_debt_service_before_goods_above_the_floor(support_enabled):
    # F + R = 100, service 40, overdraft interest 5, opening overdraft B = 30, goods above floor 40, V = 15.
    resources = np.array([100.0, 145.0, 160.0, 175.0, 200.0, 230.0, 500.0])
    settled = _settle_floor(
        resources=resources, scheduled_overdraft_interest=5.0, opening_overdraft=30.0, support_enabled=support_enabled
    )
    without = _settle_floor(resources=resources, scheduled_overdraft_interest=5.0, support_enabled=support_enabled)

    np.testing.assert_allclose(settled.overdraft_repayment, [0, 0, 15, 30, 30, 30, 30], atol=1e-9)
    np.testing.assert_allclose(settled.consumption_after_floor, [80, 80, 80, 80, 105, 120, 120], rtol=1e-9)
    np.testing.assert_allclose(settled.investment_after_floor, [0, 0, 0, 0, 0, 15, 15], atol=1e-9)
    # Rent, floor, debt service and overdraft interest come first: the overdraft does not change them.
    for name in (
        "floor_support",
        "unpaid_rent",
        "unmet_floor",
        "unpaid_consumer_service",
        "unpaid_mortgage_service",
        "unpaid_overdraft_interest",
        "remaining_subsistence_shortfall",
    ):
        np.testing.assert_array_equal(getattr(settled, name), getattr(without, name), err_msg=name)
    np.testing.assert_array_equal(without.overdraft_repayment, 0.0)


def test__support_never_covers_the_opening_overdraft():
    # No own resources: support pays the floor and rent, and nothing of the overdraft.
    settled = _settle_floor(resources=np.array([0.0, 50.0]), opening_overdraft=1000.0)
    np.testing.assert_allclose(settled.floor_support, [100.0, 50.0], rtol=1e-9)
    np.testing.assert_allclose(settled.overdraft_repayment, [0.0, 0.0], atol=1e-9)


def test__cash_need_refresh_counts_the_overdraft_as_a_use():
    resolver = HouseholdFinancialFeasibility()
    plan = PostGrantFeasiblePlan(
        credit_granted=np.zeros(4),
        credit_rationing_gap=np.zeros(4),
        planned_liquidation_total=np.zeros(4),
        residual_shortfall_after_granted_credit=np.zeros(4),
        reserved_liquidation_total=np.zeros(4),
    )
    available_lfa = np.array([50.0, -30.0, -30.0, -30.0])
    refreshed = resolver.refresh_cash_needs(
        plan,
        cash_uses=np.full(4, 100.0),
        cash_income=np.array([80.0, 80.0, 120.0, 140.0]),
        available_lfa=available_lfa,
    )
    # residual = max(uses - income - (LFA - D), 0): LFA - D without the zero floor.
    np.testing.assert_allclose(refreshed.residual_shortfall_after_lfa, [0.0, 50.0, 10.0, 0.0])
    np.testing.assert_allclose(refreshed.funded_from_liquid_assets, [20.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("own_resources", [0.0, 60.0, 100.0, 150.0])
def test__overdraft_sale_is_exempt_from_support_and_repays_the_overdraft(own_resources):
    # Overdraft B = 30 repaid by a sale of Q = 30 + service 40 + overdraft interest 5: support is the support
    # of the same household without overdraft and without sale, and the whole sale reaches the overdraft.
    sale = 75.0
    with_overdraft = _settle_floor(
        resources=own_resources + sale,
        support_exempt_resources=sale,
        opening_overdraft=30.0,
        scheduled_overdraft_interest=5.0,
        consumption_before_floor=80.0,
    )
    without = _settle_floor(resources=own_resources, consumption_before_floor=80.0)
    np.testing.assert_allclose(with_overdraft.floor_support, without.floor_support, atol=1e-9)
    np.testing.assert_allclose(with_overdraft.overdraft_repayment, [30.0], atol=1e-9)


def test__without_support_an_overdraft_sale_sized_on_the_floor_repays_the_overdraft():
    # Own resources 40 < F + R = 100; the sale covers the floor gap 60, service 40, interest 5 and B = 30.
    settled = _settle_floor(
        resources=40.0 + 135.0,
        opening_overdraft=30.0,
        scheduled_overdraft_interest=5.0,
        consumption_before_floor=50.0,
        support_enabled=False,
    )
    np.testing.assert_allclose(settled.unmet_floor, [0.0], atol=1e-9)
    np.testing.assert_allclose(settled.overdraft_repayment, [30.0], atol=1e-9)


def _overdraft_sale(**inputs):
    """overdraft_sale_need with F=80, R=20, O=45, B=30, y=100 unless overridden; L defaults to 0."""
    from macromodel.agents.households.func.financial_feasibility import overdraft_sale_need

    values = {
        "opening_overdraft": 30.0,
        "income": 100.0,
        "liquidity_shortfall": 0.0,
        "goods_floor": 80.0,
        "cash_rent": 20.0,
        "scheduled_service": 45.0,
        **inputs,
    }
    support_enabled = values.pop("support_enabled", True)
    arrays = {name: np.atleast_1d(np.asarray(value, dtype=float)) for name, value in values.items()}
    shape = np.broadcast_shapes(*(array.shape for array in arrays.values()))
    return overdraft_sale_need(
        support_enabled=support_enabled, **{name: np.broadcast_to(a, shape) for name, a in arrays.items()}
    )


@pytest.mark.parametrize("support_enabled", [True, False])
def test__overdraft_sale_is_zero_without_an_opening_overdraft(support_enabled):
    np.testing.assert_array_equal(
        _overdraft_sale(opening_overdraft=[0.0, 0.0], income=[0.0, 500.0], support_enabled=support_enabled), 0.0
    )


def test__overdraft_sale_without_desired_goods_above_the_floor_or_investment_characterisation():
    # Floor and rent 100, service 45, B = 30. Own resources y + max(L, 0):
    #   legacy: sale = max(O + B - max(own - F - R, 0), 0); minimum income: max(F + R + O + B - own, 0).
    own = np.array([0.0, 60.0, 100.0, 145.0, 160.0, 175.0, 300.0])
    np.testing.assert_allclose(_overdraft_sale(income=own), [75, 75, 75, 30, 15, 0, 0])
    np.testing.assert_allclose(_overdraft_sale(income=own, support_enabled=False), [175, 115, 75, 30, 15, 0, 0])
    # A positive shortfall is financed by borrow-or-sell and counts as own resources; a negative one does not.
    np.testing.assert_allclose(_overdraft_sale(income=[100.0, 100.0], liquidity_shortfall=[45.0, -45.0]), [30.0, 75.0])
