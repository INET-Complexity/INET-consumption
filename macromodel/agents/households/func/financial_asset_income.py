"""Calibrated decomposition of realised household financial-asset income."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FinancialAssetIncomeComponents:
    """Household-level income components and their aggregate reconciliation."""

    distribution_income: np.ndarray
    residual_portfolio_return: np.ndarray
    total_income: np.ndarray
    calibration_target: np.ndarray
    aggregate_distribution_income: float
    aggregate_residual_portfolio_return: float
    aggregate_calibration_target: float
    target_gap: float
    calibration_error: float


def _finite_1d(name: str, values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values.")
    return result


def compose_financial_asset_income(
    *,
    lagged_distribution_income: np.ndarray,
    residual_profile: np.ndarray,
    calibration_target: np.ndarray,
    tolerance: float = 1e-9,
) -> FinancialAssetIncomeComponents:
    """Split a fixed aggregate income target into distribution and residual parts.

    ``lagged_distribution_income`` is authoritative and is never rescaled. The
    existing realised portfolio-return process supplies only the cross-household
    profile of the residual. Its positive part is used because the residual is
    an income flow, not a capital-loss carrier. If that profile has no positive
    mass, the non-negative expected-income target supplies the fallback weights.

    A distribution above the aggregate target is an economic calibration error:
    it must be resolved through payout or target parameters, never silently
    clipped here.
    """
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative.")

    distribution = _finite_1d("lagged_distribution_income", lagged_distribution_income)
    profile = _finite_1d("residual_profile", residual_profile)
    target = _finite_1d("calibration_target", calibration_target)
    if profile.shape != distribution.shape or target.shape != distribution.shape:
        raise ValueError("All financial-asset income inputs must contain one value per household.")
    if np.any(distribution < -tolerance):
        raise ValueError("lagged_distribution_income contains materially negative values.")

    distribution = np.maximum(distribution, 0.0)
    aggregate_distribution = float(distribution.sum())
    aggregate_target = float(target.sum())
    if aggregate_target < -tolerance:
        raise ValueError("Aggregate financial-income calibration target must be non-negative.")
    aggregate_target = max(aggregate_target, 0.0)
    if aggregate_distribution > aggregate_target + tolerance:
        raise ValueError(
            "Lagged dividend distribution exceeds the aggregate financial-income calibration target; "
            "recalibrate payout ratios or the portfolio-return target."
        )

    aggregate_residual = max(aggregate_target - aggregate_distribution, 0.0)
    residual_weights = np.maximum(profile, 0.0)
    weight_total = float(residual_weights.sum())
    if weight_total <= tolerance:
        residual_weights = np.maximum(target, 0.0)
        weight_total = float(residual_weights.sum())
    if aggregate_residual > tolerance and weight_total <= tolerance:
        raise ValueError("A positive residual target requires a positive household allocation profile.")

    residual = np.zeros_like(distribution)
    if aggregate_residual > 0.0:
        residual = residual_weights / weight_total * aggregate_residual
    total = distribution + residual
    target_gap = float(aggregate_target - total.sum())

    return FinancialAssetIncomeComponents(
        distribution_income=distribution,
        residual_portfolio_return=residual,
        total_income=total,
        calibration_target=target.copy(),
        aggregate_distribution_income=aggregate_distribution,
        aggregate_residual_portfolio_return=float(residual.sum()),
        aggregate_calibration_target=aggregate_target,
        target_gap=target_gap,
        calibration_error=abs(target_gap),
    )
