import numpy as np
import pytest

from macromodel.agents.households.func.consumer_distress import (
    CURRENT,
    DELINQUENT,
    FICP,
    compute_stage6_consumer_distress_state,
)


def test_second_missed_consumer_payment_triggers_ficp_for_the_full_duration():
    state = compute_stage6_consumer_distress_state(
        scheduled_consumer_payments=np.asarray([10.0, 10.0]),
        actual_consumer_payments=np.asarray([9.0, 10.0]),
        unpaid_consumer_payments=np.asarray([1.0, 0.0]),
        prior_missed_payment_count_consumer=np.asarray([1, 0]),
        prior_ficp_exclusion_remaining_periods=np.asarray([0, 0]),
        ficp_exclusion_periods=20,
    )

    np.testing.assert_allclose(state.actual_consumer_payment, [9.0, 10.0])
    np.testing.assert_array_equal(state.consumer_payment_missed, [True, False])
    np.testing.assert_array_equal(state.missed_payment_count_consumer, [2, 0])
    np.testing.assert_array_equal(state.consumer_distress_state, [FICP, CURRENT])
    np.testing.assert_array_equal(state.ficp_state, [True, False])
    np.testing.assert_array_equal(state.ficp_exclusion_remaining_periods, [20, 0])

    following_period = compute_stage6_consumer_distress_state(
        scheduled_consumer_payments=np.asarray([10.0, 10.0]),
        actual_consumer_payments=np.asarray([10.0, 10.0]),
        unpaid_consumer_payments=np.asarray([0.0, 0.0]),
        prior_missed_payment_count_consumer=state.missed_payment_count_consumer,
        prior_ficp_exclusion_remaining_periods=state.ficp_exclusion_remaining_periods,
        ficp_exclusion_periods=20,
    )

    np.testing.assert_array_equal(following_period.consumer_distress_state, [FICP, CURRENT])
    np.testing.assert_array_equal(following_period.ficp_exclusion_remaining_periods, [19, 0])


def test_consumer_distress_does_not_treat_zero_service_as_a_missed_payment():
    state = compute_stage6_consumer_distress_state(
        scheduled_consumer_payments=np.asarray([0.0, 10.0]),
        actual_consumer_payments=np.asarray([0.0, 5.0]),
        unpaid_consumer_payments=np.asarray([0.0, 5.0]),
        prior_missed_payment_count_consumer=np.asarray([0, 0]),
        prior_ficp_exclusion_remaining_periods=np.asarray([0, 0]),
        ficp_exclusion_periods=20,
    )

    np.testing.assert_array_equal(state.consumer_payment_missed, [False, True])
    np.testing.assert_array_equal(state.consumer_distress_state, [CURRENT, DELINQUENT])


def test_consumer_distress_rejects_non_reconciling_settlement():
    with pytest.raises(ValueError, match="must reconcile"):
        compute_stage6_consumer_distress_state(
            scheduled_consumer_payments=np.asarray([10.0]),
            actual_consumer_payments=np.asarray([8.0]),
            unpaid_consumer_payments=np.asarray([1.0]),
            prior_missed_payment_count_consumer=np.asarray([0]),
            prior_ficp_exclusion_remaining_periods=np.asarray([0]),
            ficp_exclusion_periods=20,
        )
