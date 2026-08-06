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
    aggregate_expected_residual_portfolio_return: float
    aggregate_calibration_target: float
    stochastic_multiplier: float
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
    expected_non_negative_residual_profile: np.ndarray,
    calibration_target: np.ndarray,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-9,
) -> FinancialAssetIncomeComponents:
    """Combine authoritative distributions with a calibrated stochastic residual.

    ``lagged_distribution_income`` is authoritative and is never rescaled. The
    existing realised portfolio-return process supplies a non-negative stochastic
    multiplier around the *expected* residual. Consequently the calibration
    target is a long-run expectation, not a period-by-period accounting identity.

    A distribution above the aggregate target is an economic calibration error:
    it must be resolved through payout or target parameters, never silently
    clipped here.
    """
    for name, tolerance in (
        ("absolute_tolerance", absolute_tolerance),
        ("relative_tolerance", relative_tolerance),
    ):
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")

    distribution = _finite_1d("lagged_distribution_income", lagged_distribution_income)
    profile = _finite_1d("residual_profile", residual_profile)
    expected_profile = _finite_1d(
        "expected_non_negative_residual_profile", expected_non_negative_residual_profile
    )
    target = _finite_1d("calibration_target", calibration_target)
    if any(values.shape != distribution.shape for values in (profile, expected_profile, target)):
        raise ValueError("All financial-asset income inputs must contain one value per household.")
    scale = max(float(np.abs(distribution).sum()), float(np.abs(target).sum()), 1.0)
    tolerance = absolute_tolerance + relative_tolerance * scale
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
            "recalibrate payout ratios or the portfolio-return target "
            f"(distribution={aggregate_distribution:.12g}, target={aggregate_target:.12g}, "
            f"tolerance={tolerance:.12g})."
        )

    aggregate_expected_residual = max(aggregate_target - aggregate_distribution, 0.0)
    target_weights = np.maximum(target, 0.0)
    target_weight_total = float(target_weights.sum())
    expected_profile_total = float(np.maximum(expected_profile, 0.0).sum())
    if aggregate_expected_residual > tolerance and target_weight_total <= tolerance:
        raise ValueError("A positive residual target requires positive household target weights.")
    if aggregate_expected_residual > tolerance and expected_profile_total <= tolerance:
        raise ValueError("A positive residual target requires a positive expected residual profile.")

    realised_profile_total = float(np.maximum(profile, 0.0).sum())
    stochastic_multiplier = (
        realised_profile_total / expected_profile_total if expected_profile_total > tolerance else 0.0
    )

    residual = np.zeros_like(distribution)
    if aggregate_expected_residual > 0.0:
        residual = target_weights / target_weight_total * aggregate_expected_residual * stochastic_multiplier
    total = distribution + residual
    target_gap = float(aggregate_target - total.sum())
    calibration_error = abs(aggregate_target - aggregate_distribution - aggregate_expected_residual)

    return FinancialAssetIncomeComponents(
        distribution_income=distribution,
        residual_portfolio_return=residual,
        total_income=total,
        calibration_target=target.copy(),
        aggregate_distribution_income=aggregate_distribution,
        aggregate_residual_portfolio_return=float(residual.sum()),
        aggregate_expected_residual_portfolio_return=aggregate_expected_residual,
        aggregate_calibration_target=aggregate_target,
        stochastic_multiplier=stochastic_multiplier,
        target_gap=target_gap,
        calibration_error=calibration_error,
    )


def empirical_financial_income_target(
    *,
    current_ifa: np.ndarray,
    initial_ifa: np.ndarray,
    initial_financial_income: np.ndarray,
) -> np.ndarray:
    """Scale the observed initial aggregate financial-income yield by current IFA."""
    current = np.maximum(_finite_1d("current_ifa", current_ifa), 0.0)
    initial = np.maximum(_finite_1d("initial_ifa", initial_ifa), 0.0)
    initial_income = _finite_1d("initial_financial_income", initial_financial_income)
    if initial.shape != initial_income.shape or current.shape != initial.shape:
        raise ValueError("Empirical financial-income target inputs must have household shape.")
    initial_ifa_total = float(initial.sum())
    initial_income_total = float(initial_income.sum())
    if initial_ifa_total <= 0.0:
        if abs(initial_income_total) > 1e-6:
            raise ValueError("A positive empirical financial-income target requires positive initial IFA.")
        return np.zeros_like(current)
    if initial_income_total < 0.0:
        raise ValueError("Initial empirical financial income must be non-negative in aggregate.")
    return current * (initial_income_total / initial_ifa_total)
