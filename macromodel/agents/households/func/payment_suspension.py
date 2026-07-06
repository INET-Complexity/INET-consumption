from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Stage5PaymentSuspensionDiagnostics:
    """Diagnostic-only split of pre-support residual distress across debt service."""

    consumer_payment_suspension_needed: np.ndarray
    consumer_payment_suspension_amount: np.ndarray
    mortgage_payment_suspension_needed: np.ndarray
    mortgage_payment_suspension_amount: np.ndarray


def compute_stage5_payment_suspension_diagnostics(
    *,
    remaining_subsistence_shortfall: np.ndarray,
    scheduled_consumer_payments: np.ndarray,
    scheduled_mortgage_payments: np.ndarray,
) -> Stage5PaymentSuspensionDiagnostics:
    """Split pre-support residual distress by Stage 6 seniority, read-only."""
    residual = np.asarray(remaining_subsistence_shortfall, dtype=float)
    consumer_payments = np.asarray(scheduled_consumer_payments, dtype=float)
    mortgage_payments = np.asarray(scheduled_mortgage_payments, dtype=float)
    expected_shape = residual.shape
    for name, values in (
        ("scheduled_consumer_payments", consumer_payments),
        ("scheduled_mortgage_payments", mortgage_payments),
    ):
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must match remaining_subsistence_shortfall shape {expected_shape}; got {values.shape}."
            )

    cleaned_residual = np.where(np.isfinite(residual), np.maximum(residual, 0.0), 0.0)
    cleaned_consumer_payments = np.where(np.isfinite(consumer_payments), np.maximum(consumer_payments, 0.0), 0.0)
    cleaned_mortgage_payments = np.where(np.isfinite(mortgage_payments), np.maximum(mortgage_payments, 0.0), 0.0)

    consumer_suspension_amount = np.minimum(cleaned_residual, cleaned_consumer_payments)
    residual_after_consumer = np.maximum(cleaned_residual - consumer_suspension_amount, 0.0)
    mortgage_suspension_amount = np.minimum(residual_after_consumer, cleaned_mortgage_payments)

    return Stage5PaymentSuspensionDiagnostics(
        consumer_payment_suspension_needed=(consumer_suspension_amount > 0.0),
        consumer_payment_suspension_amount=consumer_suspension_amount,
        mortgage_payment_suspension_needed=(mortgage_suspension_amount > 0.0),
        mortgage_payment_suspension_amount=mortgage_suspension_amount,
    )
