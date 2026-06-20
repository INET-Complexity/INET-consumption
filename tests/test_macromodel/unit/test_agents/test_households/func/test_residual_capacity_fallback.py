import numpy as np

from macromodel.agents.households.func.borrow_vs_sell import (
    PREFERRED_MARGIN_BORROW,
    PREFERRED_MARGIN_SELL,
)
from macromodel.agents.households.func.residual_capacity_fallback import (
    ResidualCapacityFallbackResult,
    compute_residual_capacity_fallback,
)


def _compute(**kwargs):
    defaults = {
        "preferred_margin_after_lfa": np.asarray([PREFERRED_MARGIN_BORROW], dtype=float),
        "preferred_margin_amount": np.asarray([10.0], dtype=float),
        "income": np.asarray([100.0], dtype=float),
        "scheduled_mortgage_payment": np.asarray([0.0], dtype=float),
        "r_b": np.asarray([0.0], dtype=float),
        "consumer_loan_maturity": 10,
        "dsti_limit": 0.1,
        "current_ifa": np.asarray([50.0], dtype=float),
    }
    defaults.update(kwargs)
    return compute_residual_capacity_fallback(**defaults)


def test__borrow_preferred_below_dsti_ceiling():
    result = _compute()

    np.testing.assert_allclose(result.dsti_headroom, [10.0])
    np.testing.assert_allclose(result.dsti_maximum_loan_size, [100.0])
    np.testing.assert_array_equal(result.dsti_cap_binding, [False])
    np.testing.assert_allclose(result.borrow_planned, [10.0])
    np.testing.assert_allclose(result.liquidation_planned, [0.0])
    np.testing.assert_allclose(result.shadow_credit_requested, [10.0])
    np.testing.assert_allclose(result.forced_liquidation_amount, [0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_caps, [0.0])


def test__borrow_preferred_exactly_at_the_ceiling():
    result = _compute(
        preferred_margin_amount=np.asarray([100.0], dtype=float),
        income=np.asarray([100.0], dtype=float),
        scheduled_mortgage_payment=np.asarray([0.0], dtype=float),
        r_b=np.asarray([0.0], dtype=float),
        consumer_loan_maturity=10,
        dsti_limit=0.1,
        current_ifa=np.asarray([0.0], dtype=float),
    )

    np.testing.assert_allclose(result.dsti_headroom, [10.0])
    np.testing.assert_allclose(result.dsti_maximum_loan_size, [100.0])
    np.testing.assert_array_equal(result.dsti_cap_binding, [False])
    np.testing.assert_allclose(result.borrow_planned, [100.0])
    np.testing.assert_allclose(result.liquidation_planned, [0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_caps, [0.0])


def test__borrow_preferred_above_the_ceiling_routes_residual_to_liquidation():
    result = _compute(
        preferred_margin_amount=np.asarray([150.0], dtype=float),
        income=np.asarray([100.0], dtype=float),
        scheduled_mortgage_payment=np.asarray([0.0], dtype=float),
        r_b=np.asarray([0.0], dtype=float),
        consumer_loan_maturity=10,
        dsti_limit=0.1,
        current_ifa=np.asarray([75.0], dtype=float),
    )

    np.testing.assert_allclose(result.dsti_maximum_loan_size, [100.0])
    np.testing.assert_array_equal(result.dsti_cap_binding, [True])
    np.testing.assert_allclose(result.borrow_planned, [100.0])
    np.testing.assert_allclose(result.liquidation_planned, [50.0])
    np.testing.assert_allclose(result.shadow_credit_requested, [100.0])
    np.testing.assert_allclose(result.forced_liquidation_amount, [50.0])
    np.testing.assert_allclose(result.residual_shortfall_after_caps, [0.0])


def test__zero_rate_uses_the_annuity_limit():
    result = _compute(
        preferred_margin_amount=np.asarray([80.0], dtype=float),
        income=np.asarray([100.0], dtype=float),
        scheduled_mortgage_payment=np.asarray([0.0], dtype=float),
        r_b=np.asarray([0.0], dtype=float),
        consumer_loan_maturity=4,
        dsti_limit=0.2,
        current_ifa=np.asarray([0.0], dtype=float),
    )

    np.testing.assert_allclose(result.dsti_headroom, [20.0])
    np.testing.assert_allclose(result.dsti_maximum_loan_size, [80.0])
    np.testing.assert_allclose(result.borrow_planned, [80.0])
    np.testing.assert_array_equal(result.dsti_cap_binding, [False])


def test__sell_preferred_below_liquidation_capacity():
    result = _compute(
        preferred_margin_after_lfa=np.asarray([PREFERRED_MARGIN_SELL], dtype=float),
        preferred_margin_amount=np.asarray([5.0], dtype=float),
        current_ifa=np.asarray([20.0], dtype=float),
        r_b=np.asarray([0.0], dtype=float),
        consumer_loan_maturity=10,
        dsti_limit=0.2,
    )

    np.testing.assert_allclose(result.borrow_planned, [0.0])
    np.testing.assert_allclose(result.liquidation_planned, [5.0])
    np.testing.assert_allclose(result.shadow_credit_requested, [0.0])
    np.testing.assert_allclose(result.forced_liquidation_amount, [0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_caps, [0.0])


def test__sell_preferred_above_liquidation_capacity_routes_residual_back_to_borrowing():
    result = _compute(
        preferred_margin_after_lfa=np.asarray([PREFERRED_MARGIN_SELL], dtype=float),
        preferred_margin_amount=np.asarray([25.0], dtype=float),
        current_ifa=np.asarray([10.0], dtype=float),
        income=np.asarray([100.0], dtype=float),
        scheduled_mortgage_payment=np.asarray([0.0], dtype=float),
        r_b=np.asarray([0.0], dtype=float),
        consumer_loan_maturity=10,
        dsti_limit=0.2,
    )

    np.testing.assert_allclose(result.liquidation_planned, [10.0])
    np.testing.assert_allclose(result.borrow_planned, [15.0])
    np.testing.assert_allclose(result.shadow_credit_requested, [15.0])
    np.testing.assert_allclose(result.forced_liquidation_amount, [0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_caps, [0.0])


def test__zero_ifa_leaves_sell_branch_to_borrow_only():
    result = _compute(
        preferred_margin_after_lfa=np.asarray([PREFERRED_MARGIN_SELL], dtype=float),
        preferred_margin_amount=np.asarray([12.0], dtype=float),
        current_ifa=np.asarray([0.0], dtype=float),
        income=np.asarray([100.0], dtype=float),
        scheduled_mortgage_payment=np.asarray([0.0], dtype=float),
        r_b=np.asarray([0.0], dtype=float),
        consumer_loan_maturity=10,
        dsti_limit=0.2,
    )

    np.testing.assert_allclose(result.liquidation_planned, [0.0])
    np.testing.assert_allclose(result.borrow_planned, [12.0])
    np.testing.assert_allclose(result.residual_shortfall_after_caps, [0.0])


def test__invalid_branch_returns_none_and_leaves_the_amount_unrepaired():
    result = _compute(
        preferred_margin_after_lfa=np.asarray([99.0], dtype=float),
        preferred_margin_amount=np.asarray([12.0], dtype=float),
        current_ifa=np.asarray([20.0], dtype=float),
    )

    np.testing.assert_allclose(result.borrow_planned, [0.0])
    np.testing.assert_allclose(result.liquidation_planned, [0.0])
    np.testing.assert_allclose(result.shadow_credit_requested, [0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_caps, [12.0])


def test__zero_proxy_headroom_yields_zero_borrow_capacity():
    result = _compute(
        preferred_margin_amount=np.asarray([10.0], dtype=float),
        income=np.asarray([100.0], dtype=float),
        scheduled_mortgage_payment=np.asarray([50.0], dtype=float),
        dsti_limit=0.5,
        current_ifa=np.asarray([0.0], dtype=float),
    )

    np.testing.assert_allclose(result.dsti_headroom, [0.0])
    np.testing.assert_allclose(result.dsti_maximum_loan_size, [0.0])
    np.testing.assert_allclose(result.borrow_planned, [0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_caps, [10.0])


def test__non_finite_inputs_collapse_to_conservative_zero_capacity_behavior():
    result = _compute(
        income=np.asarray([np.nan], dtype=float),
        r_b=np.asarray([np.nan], dtype=float),
        current_ifa=np.asarray([np.nan], dtype=float),
    )

    np.testing.assert_allclose(result.dsti_headroom, [0.0])
    np.testing.assert_allclose(result.dsti_maximum_loan_size, [0.0])
    np.testing.assert_allclose(result.borrow_planned, [0.0])
    np.testing.assert_allclose(result.liquidation_planned, [0.0])
    np.testing.assert_allclose(result.shadow_credit_requested, [0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_caps, [10.0])


def test__non_finite_rate_uses_zero_rate_annuity_limit():
    result = _compute(
        preferred_margin_amount=np.asarray([80.0], dtype=float),
        income=np.asarray([100.0], dtype=float),
        scheduled_mortgage_payment=np.asarray([0.0], dtype=float),
        r_b=np.asarray([np.nan], dtype=float),
        consumer_loan_maturity=4,
        dsti_limit=0.2,
        current_ifa=np.asarray([0.0], dtype=float),
    )

    np.testing.assert_allclose(result.dsti_headroom, [20.0])
    np.testing.assert_allclose(result.dsti_maximum_loan_size, [80.0])
    np.testing.assert_allclose(result.borrow_planned, [80.0])
    np.testing.assert_array_equal(result.dsti_cap_binding, [False])


def test__negative_proxy_inputs_collapse_to_zero_capacity():
    result = _compute(
        preferred_margin_amount=np.asarray([12.0], dtype=float),
        income=np.asarray([-1.0], dtype=float),
        scheduled_mortgage_payment=np.asarray([-2.0], dtype=float),
        current_ifa=np.asarray([-3.0], dtype=float),
        r_b=np.asarray([0.04], dtype=float),
        consumer_loan_maturity=4,
        dsti_limit=0.2,
    )

    np.testing.assert_allclose(result.dsti_headroom, [0.0])
    np.testing.assert_allclose(result.dsti_maximum_loan_size, [0.0])
    np.testing.assert_allclose(result.borrow_planned, [0.0])
    np.testing.assert_allclose(result.liquidation_planned, [0.0])
    np.testing.assert_allclose(result.shadow_credit_requested, [0.0])
    np.testing.assert_allclose(result.residual_shortfall_after_caps, [12.0])


def test__returns_typed_result_without_side_effects():
    preferred_amount = np.asarray([10.0], dtype=float)
    result = _compute(preferred_margin_amount=preferred_amount)

    assert isinstance(result, ResidualCapacityFallbackResult)
    np.testing.assert_allclose(preferred_amount, [10.0])
