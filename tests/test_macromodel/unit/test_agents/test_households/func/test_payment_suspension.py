import numpy as np

from macromodel.agents.households.func.payment_suspension import (
    compute_stage5_payment_suspension_diagnostics,
)


def test_zero_residual_yields_no_payment_suspension():
    diagnostics = compute_stage5_payment_suspension_diagnostics(
        remaining_subsistence_shortfall=np.asarray([0.0, 0.0]),
        scheduled_consumer_payments=np.asarray([5.0, 2.0]),
        scheduled_mortgage_payments=np.asarray([7.0, 3.0]),
    )

    np.testing.assert_allclose(diagnostics.consumer_payment_suspension_amount, [0.0, 0.0])
    np.testing.assert_allclose(diagnostics.mortgage_payment_suspension_amount, [0.0, 0.0])
    np.testing.assert_array_equal(diagnostics.consumer_payment_suspension_needed, [False, False])
    np.testing.assert_array_equal(diagnostics.mortgage_payment_suspension_needed, [False, False])


def test_consumer_only_payment_suspension_is_capped_by_consumer_service():
    diagnostics = compute_stage5_payment_suspension_diagnostics(
        remaining_subsistence_shortfall=np.asarray([2.0, 5.0]),
        scheduled_consumer_payments=np.asarray([3.0, 7.0]),
        scheduled_mortgage_payments=np.asarray([4.0, 9.0]),
    )

    np.testing.assert_allclose(diagnostics.consumer_payment_suspension_amount, [2.0, 5.0])
    np.testing.assert_allclose(diagnostics.mortgage_payment_suspension_amount, [0.0, 0.0])
    np.testing.assert_array_equal(diagnostics.consumer_payment_suspension_needed, [True, True])
    np.testing.assert_array_equal(diagnostics.mortgage_payment_suspension_needed, [False, False])


def test_payment_suspension_spills_from_consumer_to_mortgage_by_seniority():
    diagnostics = compute_stage5_payment_suspension_diagnostics(
        remaining_subsistence_shortfall=np.asarray([8.0, 25.0]),
        scheduled_consumer_payments=np.asarray([3.0, 10.0]),
        scheduled_mortgage_payments=np.asarray([7.0, 20.0]),
    )

    np.testing.assert_allclose(diagnostics.consumer_payment_suspension_amount, [3.0, 10.0])
    np.testing.assert_allclose(diagnostics.mortgage_payment_suspension_amount, [5.0, 15.0])
    np.testing.assert_array_equal(diagnostics.consumer_payment_suspension_needed, [True, True])
    np.testing.assert_array_equal(diagnostics.mortgage_payment_suspension_needed, [True, True])


def test_zero_scheduled_payment_caps_block_suspension_allocation():
    diagnostics = compute_stage5_payment_suspension_diagnostics(
        remaining_subsistence_shortfall=np.asarray([9.0, 4.0]),
        scheduled_consumer_payments=np.asarray([0.0, 0.0]),
        scheduled_mortgage_payments=np.asarray([0.0, 2.0]),
    )

    np.testing.assert_allclose(diagnostics.consumer_payment_suspension_amount, [0.0, 0.0])
    np.testing.assert_allclose(diagnostics.mortgage_payment_suspension_amount, [0.0, 2.0])
    np.testing.assert_array_equal(diagnostics.consumer_payment_suspension_needed, [False, False])
    np.testing.assert_array_equal(diagnostics.mortgage_payment_suspension_needed, [False, True])


def test_nonfinite_and_negative_inputs_fall_back_to_zero_without_negative_outputs():
    diagnostics = compute_stage5_payment_suspension_diagnostics(
        remaining_subsistence_shortfall=np.asarray([np.nan, -5.0, np.inf]),
        scheduled_consumer_payments=np.asarray([4.0, np.nan, -2.0]),
        scheduled_mortgage_payments=np.asarray([np.inf, -3.0, 6.0]),
    )

    np.testing.assert_allclose(diagnostics.consumer_payment_suspension_amount, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(diagnostics.mortgage_payment_suspension_amount, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(diagnostics.consumer_payment_suspension_needed, [False, False, False])
    np.testing.assert_array_equal(diagnostics.mortgage_payment_suspension_needed, [False, False, False])
