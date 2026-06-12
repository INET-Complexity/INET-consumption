"""Opt-in household income-belief learning helpers.

The learning block is deliberately pure: it derives consumption-rule inputs
from static priors and current model state, but it does not write household
time-series state. Consumption rules opt in by requesting these outputs.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IncomeBeliefLearningOutputs:
    """Outputs consumed by opt-in consumption rules."""

    permanent_income_log_ratio: np.ndarray
    uncertainty_delta: np.ndarray
    prior_mean: np.ndarray
    prior_variance: np.ndarray
    predicted_mean: np.ndarray
    predicted_variance: np.ndarray
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    kalman_gain: np.ndarray
    income_signal: np.ndarray
    prediction_error: np.ndarray


def _household_array(priors: dict[str, np.ndarray], key: str, n_households: int) -> np.ndarray:
    if key not in priors:
        raise ValueError(f"Income-belief priors missing required key {key!r}.")
    value = np.asarray(priors[key], dtype=float)
    if value.shape != (n_households,):
        raise ValueError(f"Income-belief prior {key!r} must have shape ({n_households},), got {value.shape}.")
    return value


def compute_income_belief_learning_outputs(
    *,
    current_income: np.ndarray,
    lagged_income: np.ndarray,
    priors: dict[str, np.ndarray],
    prior_mean: np.ndarray | None = None,
    prior_variance: np.ndarray | None = None,
    common_permanent_income_log_ratio: float | np.ndarray | None = None,
    income_floor: float = 1e-12,
    rho_denominator_floor: float = 1e-12,
) -> IncomeBeliefLearningOutputs:
    """Return one-period Bayesian income-learning outputs for households.

    ``priors`` is expected to contain household-level arrays from
    ``compute_household_income_beliefs``: initial ``income_belief_mu`` and
    ``income_belief_p``, plus ``income_belief_rho``, ``sigma2_xi`` and
    ``sigma2_v``. ``sigma2_v`` is the process variance, and ``sigma2_xi`` is
    the signal/observation variance. The optional common component is added
    after the household update so macro permanent-income forecasting remains
    separately supplied.
    """
    current_income = np.asarray(current_income, dtype=float)
    lagged_income = np.asarray(lagged_income, dtype=float)
    if current_income.shape != lagged_income.shape:
        raise ValueError(
            "current_income and lagged_income must have the same shape, "
            f"got {current_income.shape} and {lagged_income.shape}."
        )
    if current_income.ndim != 1:
        raise ValueError(f"Income-belief learning expects 1D household arrays, got {current_income.shape}.")

    n_households = current_income.shape[0]
    if prior_mean is None:
        prior_mean = _household_array(priors, "income_belief_mu", n_households)
    else:
        prior_mean = np.asarray(prior_mean, dtype=float)
        if prior_mean.shape != (n_households,):
            raise ValueError(f"prior_mean must have shape ({n_households},), got {prior_mean.shape}.")
    if prior_variance is None:
        prior_variance = _household_array(priors, "income_belief_p", n_households)
    else:
        prior_variance = np.asarray(prior_variance, dtype=float)
        if prior_variance.shape != (n_households,):
            raise ValueError(f"prior_variance must have shape ({n_households},), got {prior_variance.shape}.")
    rho = _household_array(priors, "income_belief_rho", n_households)
    process_variance = _household_array(priors, "sigma2_v", n_households)
    signal_variance = _household_array(priors, "sigma2_xi", n_households)

    common_component = (
        0.0
        if common_permanent_income_log_ratio is None
        else np.asarray(
            common_permanent_income_log_ratio,
            dtype=float,
        )
    )
    income_signal = (
        np.log(np.maximum(current_income, income_floor))
        - np.log(np.maximum(lagged_income, income_floor))
        - common_component
    )
    predicted_mean = rho * prior_mean
    predicted_variance = np.maximum((rho**2) * prior_variance + process_variance, 0.0)
    prediction_error = income_signal - predicted_mean
    innovation_variance = predicted_variance + np.maximum(signal_variance, income_floor)
    kalman_gain = np.divide(
        predicted_variance,
        innovation_variance,
        out=np.zeros_like(predicted_variance),
        where=innovation_variance > 0.0,
    )
    posterior_mean = predicted_mean + kalman_gain * prediction_error
    posterior_variance = np.maximum((1.0 - kalman_gain) * predicted_variance, 0.0)
    uncertainty_delta = posterior_variance - prior_variance

    rho_denominator = np.maximum(1.0 - rho, rho_denominator_floor)
    permanent_income_log_ratio = posterior_mean / rho_denominator + common_component

    return IncomeBeliefLearningOutputs(
        permanent_income_log_ratio=permanent_income_log_ratio,
        uncertainty_delta=uncertainty_delta,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        predicted_mean=predicted_mean,
        predicted_variance=predicted_variance,
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        kalman_gain=kalman_gain,
        income_signal=income_signal,
        prediction_error=prediction_error,
    )
