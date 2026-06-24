"""Opt-in household income-belief learning helpers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IncomeBeliefLearningOutputs:
    """One-period Kalman update outputs for income-belief diagnostics."""

    prior_mean: np.ndarray
    prior_variance: np.ndarray
    predicted_mean: np.ndarray
    predicted_variance: np.ndarray
    posterior_mean: np.ndarray
    posterior_variance: np.ndarray
    kalman_gain: np.ndarray
    realised_income_growth: np.ndarray
    common_income_growth_signal: float
    income_signal: np.ndarray
    prediction_error: np.ndarray
    floor_used: np.ndarray
    posterior_fallback_used: np.ndarray
    growth_clipped: np.ndarray


def _household_array(priors: dict[str, np.ndarray], key: str, n_households: int) -> np.ndarray:
    if key not in priors:
        raise ValueError(f"Income-belief priors missing required key {key!r}.")
    value = np.asarray(priors[key], dtype=float)
    if value.shape != (n_households,):
        raise ValueError(f"Income-belief prior {key!r} must have shape ({n_households},), got {value.shape}.")
    return value


def _scalar_rho(priors: dict[str, np.ndarray], n_households: int) -> float:
    rho_values = _household_array(priors, "income_belief_rho", n_households)
    rho = float(rho_values[0])
    if not np.allclose(rho_values, rho):
        raise ValueError("Increment 2 expects income_belief_rho to be constant across households.")
    return rho


def compute_zeta(rho: float, delta: float, S: int) -> float:
    """Scalar individual weight on ``income_belief_mu`` in the permanent-income log ratio.

    Implements eq:zeta_weight from the Stage 3 architecture note::

        zeta = sum_{s=1}^{S} delta^s * (1/4) * sum_{j=0}^{s-1} rho^(4*floor(j/4))
               ----------------------------------------------------------------
                              sum_{s=0}^{S} delta^s

    ``zeta`` depends only on the scalars ``(rho, delta, S)`` and is therefore
    computed once and cached, not per period and not per household. ``S`` is a
    small horizon (40 quarters in the FR calibration), so the nested sums are
    evaluated directly without vectorisation.
    """
    rho = float(rho)
    delta = float(delta)
    S = int(S)
    if S < 0:
        raise ValueError(f"compute_zeta expects a non-negative horizon S, got {S}.")

    denominator = float(np.sum([delta**s for s in range(0, S + 1)]))
    numerator = 0.0
    for s in range(1, S + 1):
        inner = float(np.sum([rho ** (4 * (j // 4)) for j in range(0, s)]))
        numerator += (delta**s) * (1.0 / 4.0) * inner
    return numerator / denominator


def compute_permanent_income_log_ratio(
    income_belief_mu: np.ndarray,
    zeta: float,
    common_log_ratio: float,
) -> np.ndarray:
    """Total log permanent-to-current income ratio per household.

    ``zeta * income_belief_mu + common_log_ratio`` where ``common_log_ratio`` is
    the (scalar) common-forecast term broadcast across households. With
    ``common_log_ratio == 0.0`` (forecast unavailable) this reduces to the pure
    individual term ``zeta * income_belief_mu``.
    """
    income_belief_mu = np.asarray(income_belief_mu, dtype=float)
    return float(zeta) * income_belief_mu + float(common_log_ratio)


def compute_income_uncertainty(
    income_belief_p: np.ndarray,
    zeta: float,
    common_forecast_variance: float,
) -> np.ndarray:
    """Total income-uncertainty index ``theta_it`` per household.

    ``zeta**2 * income_belief_p + common_forecast_variance`` where
    ``common_forecast_variance`` is the (scalar) common-forecast prediction-error
    variance broadcast across households. With ``common_forecast_variance == 0.0``
    (forecast unavailable) this reduces to the pure individual term
    ``zeta**2 * income_belief_p``.
    """
    income_belief_p = np.asarray(income_belief_p, dtype=float)
    return (float(zeta) ** 2) * income_belief_p + float(common_forecast_variance)


def compute_income_belief_learning_outputs(
    *,
    current_income: np.ndarray,
    lagged_income: np.ndarray,
    priors: dict[str, np.ndarray],
    prior_mean: np.ndarray | None = None,
    prior_variance: np.ndarray | None = None,
    income_floor: float = 1e-12,
    growth_clip_bound: float = 1.0,
) -> IncomeBeliefLearningOutputs:
    """Return one-period Bayesian income-learning outputs for households.

    ``current_income`` and ``lagged_income`` must already be real household
    income at ``t`` and ``t-4``. Increment 2 updates posterior beliefs only;
    final permanent-income and uncertainty terms are intentionally deferred.

    ``growth_clip_bound`` bounds ``realised_income_growth`` to
    ``[-growth_clip_bound, +growth_clip_bound]`` log-units before it enters the
    Kalman update. ``income_floor`` (1e-12) only prevents ``log(0)``; on its own
    it lets a household with near-zero or negative (e.g. a bad financial-asset
    draw) income in one period imply tens of log-units of "growth", which then
    persists in posterior_mean across periods (see GH issue #90). The default
    of 1.0 (~2.7x) is already generous for one quarter.
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
    rho = _scalar_rho(priors, n_households)
    process_variance = _household_array(priors, "sigma2_v", n_households)
    signal_variance = _household_array(priors, "sigma2_xi", n_households)

    floor_used = (current_income <= income_floor) | (lagged_income <= income_floor)
    log_current_income = np.log(np.maximum(current_income, income_floor))
    log_lagged_income = np.log(np.maximum(lagged_income, income_floor))
    raw_income_growth = log_current_income - log_lagged_income
    realised_income_growth = np.clip(raw_income_growth, -growth_clip_bound, growth_clip_bound)
    growth_clipped = np.abs(raw_income_growth) > growth_clip_bound
    common_income_growth_signal = float(log_current_income.mean() - log_lagged_income.mean())
    income_signal = realised_income_growth - common_income_growth_signal
    predicted_mean = rho * prior_mean
    predicted_variance = np.maximum((rho**2) * prior_variance + process_variance, 0.0)
    prediction_error = income_signal - predicted_mean
    innovation_variance = predicted_variance + np.maximum(signal_variance, income_floor)
    # When innovation_variance is non-finite or non-positive, kalman_gain
    # defaults to 0 (no update from the signal) rather than the ~1 a true
    # Kalman gain would take as prior variance dominates uncertainty; this
    # is a defensive default, not a physically motivated gain.
    kalman_gain = np.divide(
        predicted_variance,
        innovation_variance,
        out=np.zeros_like(predicted_variance),
        where=np.isfinite(innovation_variance) & (innovation_variance > 0.0),
    )
    with np.errstate(invalid="ignore"):
        posterior_mean = predicted_mean + kalman_gain * prediction_error
        posterior_variance = np.maximum((1.0 - kalman_gain) * predicted_variance, 0.0)

    # Spec step 9: non-finite outputs fall back to the previous posterior
    # (mu) or previous/zero-floored posterior variance (p), rather than
    # propagating NaN/inf into runtime state. posterior_fallback_used flags
    # households where this triggered, so silent corrections remain visible
    # to callers even though execution continues.
    non_finite_mean = ~np.isfinite(posterior_mean)
    if non_finite_mean.any():
        fallback_mean = np.where(np.isfinite(prior_mean), prior_mean, 0.0)
        posterior_mean = np.where(non_finite_mean, fallback_mean, posterior_mean)
    non_finite_variance = ~np.isfinite(posterior_variance)
    if non_finite_variance.any():
        fallback_variance = np.where(np.isfinite(prior_variance), prior_variance, 0.0)
        posterior_variance = np.where(non_finite_variance, fallback_variance, posterior_variance)
    posterior_fallback_used = non_finite_mean | non_finite_variance

    return IncomeBeliefLearningOutputs(
        prior_mean=prior_mean,
        prior_variance=prior_variance,
        predicted_mean=predicted_mean,
        predicted_variance=predicted_variance,
        posterior_mean=posterior_mean,
        posterior_variance=posterior_variance,
        kalman_gain=kalman_gain,
        realised_income_growth=realised_income_growth,
        common_income_growth_signal=common_income_growth_signal,
        income_signal=income_signal,
        prediction_error=prediction_error,
        floor_used=floor_used,
        posterior_fallback_used=posterior_fallback_used,
        growth_clipped=growth_clipped,
    )
