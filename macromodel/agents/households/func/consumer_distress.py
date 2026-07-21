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
    ficp_episode_missed_payment_count: np.ndarray
    ficp_episode_status: np.ndarray
    ficp_episode_triggered: np.ndarray
    ficp_horizon_completed: np.ndarray


@dataclass(frozen=True)
class FICPForgivenessEvent:
    """Typed 4b handoff for one completed FICP episode."""

    household_id: int
    ficp_episode_id: int
    trigger_period: int
    horizon_end_period: int
    residual_contractual_principal: float
    residual_principal_arrears: float
    residual_interest_arrears: float
    emitted: bool = True
    processed: bool = False


def compute_stage6_consumer_distress_state(
    *,
    scheduled_consumer_payments: np.ndarray,
    actual_consumer_payments: np.ndarray,
    unpaid_consumer_payments: np.ndarray,
    prior_missed_payment_count_consumer: np.ndarray,
    prior_ficp_exclusion_remaining_periods: np.ndarray,
    ficp_exclusion_periods: int,
    prior_ficp_episode_missed_payment_count: np.ndarray | None = None,
    prior_ficp_episode_status: np.ndarray | None = None,
) -> Stage6ConsumerDistressState:
    """Classify authoritative consumer-payment settlement without mutating it.

    Misses accumulate over the household's history. The second miss starts the
    configured FICP exclusion; while that exclusion is active, a fully paid
    period decrements its remaining duration but does not reset the miss count.
    """
    scheduled = np.asarray(scheduled_consumer_payments, dtype=float)
    actual_payment = np.asarray(actual_consumer_payments, dtype=float)
    unpaid_payment = np.asarray(unpaid_consumer_payments, dtype=float)
    prior_count = np.asarray(prior_missed_payment_count_consumer, dtype=float)
    prior_exclusion = np.asarray(prior_ficp_exclusion_remaining_periods, dtype=float)
    expected_shape = scheduled.shape
    for name, values in (
        ("actual_consumer_payments", actual_payment),
        ("unpaid_consumer_payments", unpaid_payment),
        ("prior_missed_payment_count_consumer", prior_count),
        ("prior_ficp_exclusion_remaining_periods", prior_exclusion),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must match scheduled_consumer_payments shape {expected_shape}; got {values.shape}."
            )
    if ficp_exclusion_periods <= 0:
        raise ValueError("ficp_exclusion_periods must be positive.")
    for name, values in (
        ("scheduled_consumer_payments", scheduled),
        ("actual_consumer_payments", actual_payment),
        ("unpaid_consumer_payments", unpaid_payment),
    ):
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} must contain finite, non-negative values.")
    actual_exceeds_scheduled = (actual_payment > scheduled) & ~np.isclose(actual_payment, scheduled)
    if np.any(actual_exceeds_scheduled) or not np.allclose(
        actual_payment + unpaid_payment,
        scheduled,
    ):
        raise ValueError("Settled actual and unpaid consumer payments must reconcile with scheduled service.")

    payment_missed = unpaid_payment > 0.0

    cleaned_prior_count = np.where(np.isfinite(prior_count), np.maximum(prior_count, 0.0), 0.0).astype(int)
    cleaned_prior_exclusion = np.where(np.isfinite(prior_exclusion), np.maximum(prior_exclusion, 0.0), 0.0).astype(int)
    missed_payment_count = cleaned_prior_count + payment_missed.astype(int)
    episode_state_enabled = prior_ficp_episode_missed_payment_count is not None
    if episode_state_enabled:
        episode_count = np.asarray(prior_ficp_episode_missed_payment_count, dtype=float)
        if episode_count.shape != expected_shape:
            raise ValueError(
                "prior_ficp_episode_missed_payment_count must match "
                f"scheduled_consumer_payments shape {expected_shape}; got {episode_count.shape}."
            )
        episode_count = np.where(np.isfinite(episode_count), np.maximum(episode_count, 0.0), 0.0).astype(int)
        if prior_ficp_episode_status is None:
            episode_status = np.zeros(expected_shape, dtype=int)
        else:
            episode_status = np.asarray(prior_ficp_episode_status, dtype=float)
            if episode_status.shape != expected_shape:
                raise ValueError(
                    "prior_ficp_episode_status must match "
                    f"scheduled_consumer_payments shape {expected_shape}; got {episode_status.shape}."
                )
            episode_status = np.where(np.isfinite(episode_status), np.maximum(episode_status, 0.0), 0.0).astype(int)
        # A completed episode starts a fresh local miss count. Lifetime misses
        # remain available through missed_payment_count_consumer.
        episode_count_before_payment = np.where(episode_status == 2, 0, episode_count)
        episode_missed_payment_count = episode_count_before_payment + payment_missed.astype(int)
        new_ficp_trigger = payment_missed & (episode_missed_payment_count >= 2) & (cleaned_prior_exclusion == 0)
    else:
        episode_missed_payment_count = missed_payment_count.copy()
        new_ficp_trigger = payment_missed & (missed_payment_count >= 2) & (cleaned_prior_exclusion == 0)
    exclusion_remaining = np.where(
        new_ficp_trigger,
        ficp_exclusion_periods,
        np.maximum(cleaned_prior_exclusion - 1, 0),
    )
    ficp_state = exclusion_remaining > 0
    consumer_distress_state = np.where(ficp_state, FICP, np.where(payment_missed, DELINQUENT, CURRENT))
    horizon_completed = (cleaned_prior_exclusion == 1) & ~new_ficp_trigger
    episode_status = np.where(
        new_ficp_trigger | ficp_state,
        1,
        np.where(horizon_completed, 2, 0),
    ).astype(int)

    return Stage6ConsumerDistressState(
        actual_consumer_payment=actual_payment,
        consumer_payment_missed=payment_missed,
        missed_payment_count_consumer=missed_payment_count,
        consumer_distress_state=consumer_distress_state,
        ficp_state=ficp_state,
        ficp_exclusion_remaining_periods=exclusion_remaining,
        ficp_episode_missed_payment_count=episode_missed_payment_count,
        ficp_episode_status=episode_status,
        ficp_episode_triggered=new_ficp_trigger,
        ficp_horizon_completed=horizon_completed,
    )
