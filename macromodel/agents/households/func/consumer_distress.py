"""Stage 6 consumer-credit distress-state transitions."""

from dataclasses import dataclass

import numpy as np

CURRENT = 0
DELINQUENT = 1
FICP = 2


@dataclass(frozen=True)
class Stage6ConsumerDistressState:
    """Settled consumer-payment outcome and persistent distress state."""

    actual_consumer_payment: np.ndarray
    consumer_payment_missed: np.ndarray
    missed_payment_count_consumer: np.ndarray
    consumer_distress_state: np.ndarray
    ficp_state: np.ndarray
    ficp_exclusion_remaining_periods: np.ndarray


def compute_stage6_consumer_distress_state(
    *,
    scheduled_consumer_payments: np.ndarray,
    consumer_payment_suspension_amount: np.ndarray,
    prior_missed_payment_count_consumer: np.ndarray,
    prior_ficp_exclusion_remaining_periods: np.ndarray,
    ficp_exclusion_periods: int,
) -> Stage6ConsumerDistressState:
    """Classify settled consumer-credit payments without changing their settlement.

    Stage 5 supplies the read-only suspended amount.  The settled actual payment
    is therefore scheduled service less that suspension.  The second missed
    consumer payment triggers a five-year FICP exclusion, represented in model
    periods by ``ficp_exclusion_periods``.
    """
    scheduled = np.asarray(scheduled_consumer_payments, dtype=float)
    suspension = np.asarray(consumer_payment_suspension_amount, dtype=float)
    prior_count = np.asarray(prior_missed_payment_count_consumer, dtype=float)
    prior_exclusion = np.asarray(prior_ficp_exclusion_remaining_periods, dtype=float)
    expected_shape = scheduled.shape
    for name, values in (
        ("consumer_payment_suspension_amount", suspension),
        ("prior_missed_payment_count_consumer", prior_count),
        ("prior_ficp_exclusion_remaining_periods", prior_exclusion),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"{name} must match scheduled_consumer_payments shape {expected_shape}; got {values.shape}.")
    if ficp_exclusion_periods <= 0:
        raise ValueError("ficp_exclusion_periods must be positive.")

    cleaned_scheduled = np.where(np.isfinite(scheduled), np.maximum(scheduled, 0.0), 0.0)
    cleaned_suspension = np.where(np.isfinite(suspension), np.maximum(suspension, 0.0), 0.0)
    cleaned_suspension = np.minimum(cleaned_suspension, cleaned_scheduled)
    actual_payment = cleaned_scheduled - cleaned_suspension
    payment_missed = cleaned_suspension > 0.0

    cleaned_prior_count = np.where(np.isfinite(prior_count), np.maximum(prior_count, 0.0), 0.0).astype(int)
    cleaned_prior_exclusion = np.where(np.isfinite(prior_exclusion), np.maximum(prior_exclusion, 0.0), 0.0).astype(int)
    missed_payment_count = cleaned_prior_count + payment_missed.astype(int)
    new_ficp_trigger = payment_missed & (missed_payment_count >= 2) & (cleaned_prior_exclusion == 0)
    exclusion_remaining = np.where(
        new_ficp_trigger,
        ficp_exclusion_periods,
        np.maximum(cleaned_prior_exclusion - 1, 0),
    )
    ficp_state = exclusion_remaining > 0
    consumer_distress_state = np.where(ficp_state, FICP, np.where(payment_missed, DELINQUENT, CURRENT))

    return Stage6ConsumerDistressState(
        actual_consumer_payment=actual_payment,
        consumer_payment_missed=payment_missed,
        missed_payment_count_consumer=missed_payment_count,
        consumer_distress_state=consumer_distress_state,
        ficp_state=ficp_state,
        ficp_exclusion_remaining_periods=exclusion_remaining,
    )
