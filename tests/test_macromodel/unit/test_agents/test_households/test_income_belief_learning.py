import numpy as np
import pytest

from macromodel.agents.households.income_belief_learning import (
    compute_income_belief_learning_outputs,
    compute_income_uncertainty,
    compute_permanent_income_log_ratio,
    compute_zeta,
)


def test__compute_zeta_hand_computed_s1():
    # S=1: numerator sum is only s=1, inner sum over j=0..0 is rho^0 = 1.
    #   numerator   = delta^1 * (1/4) * 1
    #   denominator = delta^0 + delta^1 = 1 + delta
    rho = 0.9519
    delta = 0.95
    expected = (delta * (1.0 / 4.0) * 1.0) / (1.0 + delta)
    assert compute_zeta(rho, delta, S=1) == pytest.approx(expected)


def test__compute_zeta_hand_computed_s2():
    # S=2: numerator = s=1 term + s=2 term.
    #   s=1: delta^1 * (1/4) * (rho^0)              = delta * 1/4
    #   s=2: delta^2 * (1/4) * (rho^0 + rho^0)      = delta^2 * 1/4 * 2
    #        (j=0 and j=1 both give 4*floor(j/4)=0 -> rho^0=1)
    #   denominator = 1 + delta + delta^2
    rho = 0.9519
    delta = 0.95
    numerator = delta * (1.0 / 4.0) * 1.0 + (delta**2) * (1.0 / 4.0) * 2.0
    denominator = 1.0 + delta + delta**2
    assert compute_zeta(rho, delta, S=2) == pytest.approx(numerator / denominator)


def test__compute_zeta_s0_is_zero():
    # S=0: numerator sum (s=1..0) is empty -> 0; denominator is delta^0 = 1.
    assert compute_zeta(0.9519, 0.95, S=0) == pytest.approx(0.0)


def test__compute_zeta_finite_and_positive_for_resolved_params():
    # zeta is a discounted sum of horizon-growing inner sums divided by a
    # geometric discount-factor sum, so it is NOT bounded by 1 for the resolved
    # (rho=0.9519, delta=0.95, S=40) calibration (it is ~2.31). The meaningful
    # invariants are that it is finite and strictly positive.
    zeta = compute_zeta(0.9519, 0.95, S=40)
    assert np.isfinite(zeta)
    assert zeta > 0.0


def test__compute_zeta_monotonic_in_horizon():
    # Adding discounted future quarters can only add non-negative numerator mass
    # relative to the denominator growth, so zeta is non-decreasing in S for the
    # resolved (rho<1, delta<1) calibration.
    zetas = [compute_zeta(0.9519, 0.95, S=s) for s in range(0, 41)]
    assert all(b >= a - 1e-12 for a, b in zip(zetas, zetas[1:]))


def test__compute_zeta_rejects_negative_horizon():
    with pytest.raises(ValueError, match="non-negative horizon"):
        compute_zeta(0.9519, 0.95, S=-1)


def test__compute_permanent_income_log_ratio_hand_computed():
    mu = np.array([0.2, -0.1, 0.0])
    zeta = 0.3
    common_log_ratio = 0.05
    expected = zeta * mu + common_log_ratio
    np.testing.assert_allclose(
        compute_permanent_income_log_ratio(mu, zeta, common_log_ratio),
        expected,
    )


def test__compute_permanent_income_log_ratio_zero_common_reduces_to_individual():
    mu = np.array([0.2, -0.1, 0.4])
    zeta = 0.3
    np.testing.assert_allclose(
        compute_permanent_income_log_ratio(mu, zeta, 0.0),
        zeta * mu,
    )


def test__compute_income_uncertainty_hand_computed():
    p = np.array([0.5, 0.2, 0.0])
    zeta = 0.3
    common_forecast_variance = 0.04
    expected = (zeta**2) * p + common_forecast_variance
    np.testing.assert_allclose(
        compute_income_uncertainty(p, zeta, common_forecast_variance),
        expected,
    )


def test__compute_income_uncertainty_zero_common_reduces_to_individual():
    p = np.array([0.5, 0.2, 0.7])
    zeta = 0.3
    np.testing.assert_allclose(
        compute_income_uncertainty(p, zeta, 0.0),
        (zeta**2) * p,
    )


def test__income_belief_learning_outputs_use_correct_variance_mapping_and_prediction_error():
    current_income = np.array([110.0, 88.0])
    lagged_income = np.array([100.0, 80.0])
    priors = {
        "income_belief_mu": np.array([0.1, -0.2]),
        "income_belief_p": np.array([0.5, 0.2]),
        "income_belief_rho": np.array([0.8, 0.8]),
        "sigma2_xi": np.array([3.0, 1.5]),
        "sigma2_v": np.array([2.0, 1.0]),
    }

    outputs = compute_income_belief_learning_outputs(
        current_income=current_income,
        lagged_income=lagged_income,
        priors=priors,
    )

    expected_growth = np.log(current_income) - np.log(lagged_income)
    expected_common_growth = np.log(current_income).mean() - np.log(lagged_income).mean()
    expected_signal = expected_growth - expected_common_growth
    expected_predicted_mean = 0.8 * priors["income_belief_mu"]
    expected_predicted_variance = (0.8**2) * priors["income_belief_p"] + priors["sigma2_v"]
    expected_prediction_error = expected_signal - expected_predicted_mean
    expected_kalman_gain = expected_predicted_variance / (expected_predicted_variance + priors["sigma2_xi"])
    expected_posterior_mean = expected_predicted_mean + expected_kalman_gain * expected_prediction_error
    expected_posterior_variance = (1.0 - expected_kalman_gain) * expected_predicted_variance

    np.testing.assert_allclose(outputs.realised_income_growth, expected_growth)
    assert outputs.common_income_growth_signal == pytest.approx(expected_common_growth)
    np.testing.assert_allclose(outputs.income_signal, expected_signal)
    assert outputs.income_signal.mean() == pytest.approx(0.0)
    np.testing.assert_allclose(outputs.predicted_mean, expected_predicted_mean)
    np.testing.assert_allclose(outputs.predicted_variance, expected_predicted_variance)
    np.testing.assert_allclose(outputs.prediction_error, expected_prediction_error)
    np.testing.assert_allclose(outputs.kalman_gain, expected_kalman_gain)
    assert np.all(outputs.kalman_gain >= 0.0)
    assert np.all(outputs.kalman_gain <= 1.0)
    np.testing.assert_allclose(outputs.posterior_mean, expected_posterior_mean)
    np.testing.assert_allclose(outputs.posterior_variance, expected_posterior_variance)
    assert not outputs.floor_used.any()
    assert not outputs.posterior_fallback_used.any()
    assert np.all(outputs.posterior_variance <= outputs.predicted_variance)


def test__income_belief_learning_accepts_updated_prior_state():
    priors = {
        "income_belief_mu": np.array([0.0]),
        "income_belief_p": np.array([0.0]),
        "income_belief_rho": np.array([0.5]),
        "sigma2_xi": np.array([2.0]),
        "sigma2_v": np.array([1.0]),
    }

    outputs = compute_income_belief_learning_outputs(
        current_income=np.array([120.0]),
        lagged_income=np.array([100.0]),
        priors=priors,
        prior_mean=np.array([0.4]),
        prior_variance=np.array([0.25]),
    )

    expected_predicted_mean = 0.2
    expected_predicted_variance = 1.0625
    expected_signal = 0.0
    expected_prediction_error = expected_signal - expected_predicted_mean
    expected_kalman_gain = expected_predicted_variance / (expected_predicted_variance + 2.0)
    np.testing.assert_allclose(outputs.prior_mean, 0.4)
    np.testing.assert_allclose(outputs.prior_variance, 0.25)
    np.testing.assert_allclose(outputs.predicted_mean, expected_predicted_mean)
    np.testing.assert_allclose(outputs.predicted_variance, expected_predicted_variance)
    np.testing.assert_allclose(outputs.prediction_error, expected_prediction_error)
    np.testing.assert_allclose(outputs.kalman_gain, expected_kalman_gain)
    expected_posterior_variance = (1.0 - expected_kalman_gain) * expected_predicted_variance
    np.testing.assert_allclose(outputs.posterior_variance, expected_posterior_variance)


def test__income_belief_learning_rejects_non_constant_rho():
    priors = {
        "income_belief_mu": np.array([0.0, 0.0]),
        "income_belief_p": np.array([0.0, 0.0]),
        "income_belief_rho": np.array([0.5, 0.6]),
        "sigma2_xi": np.array([2.0, 2.0]),
        "sigma2_v": np.array([1.0, 1.0]),
    }

    with pytest.raises(ValueError, match="income_belief_rho"):
        compute_income_belief_learning_outputs(
            current_income=np.array([120.0, 110.0]),
            lagged_income=np.array([100.0, 100.0]),
            priors=priors,
        )


def test__income_belief_learning_handles_zero_income_floor():
    priors = {
        "income_belief_mu": np.array([0.0, 0.0]),
        "income_belief_p": np.array([0.0, 0.0]),
        "income_belief_rho": np.array([0.5, 0.5]),
        "sigma2_xi": np.array([2.0, 2.0]),
        "sigma2_v": np.array([1.0, 1.0]),
    }

    outputs = compute_income_belief_learning_outputs(
        current_income=np.array([0.0, 110.0]),
        lagged_income=np.array([100.0, 0.0]),
        priors=priors,
    )

    assert outputs.floor_used.all()
    assert np.isfinite(outputs.income_signal).all()
    assert np.isfinite(outputs.posterior_mean).all()
    assert np.isfinite(outputs.posterior_variance).all()
    # Both households' raw log-growth (~-32, ~+32) is far outside the default
    # +-1.0 growth_clip_bound -- the floor alone (income_floor=1e-12) only keeps
    # this finite, it does not keep it economically plausible. See GH issue #90.
    assert outputs.growth_clipped.all()
    assert np.all(np.abs(outputs.realised_income_growth) <= 1.0)


def test__income_belief_learning_clips_extreme_growth_but_not_normal_growth():
    priors = {
        "income_belief_mu": np.array([0.0, 0.0]),
        "income_belief_p": np.array([0.0, 0.0]),
        "income_belief_rho": np.array([0.5, 0.5]),
        "sigma2_xi": np.array([2.0, 2.0]),
        "sigma2_v": np.array([1.0, 1.0]),
    }

    # Household 0: ~20% quarterly income growth, well inside the default bound.
    # Household 1: a 50x income jump, far outside it.
    outputs = compute_income_belief_learning_outputs(
        current_income=np.array([120.0, 5000.0]),
        lagged_income=np.array([100.0, 100.0]),
        priors=priors,
    )

    raw_growth = np.log(np.array([120.0, 5000.0])) - np.log(np.array([100.0, 100.0]))
    assert raw_growth[0] < 1.0
    assert raw_growth[1] > 1.0

    np.testing.assert_allclose(outputs.realised_income_growth[0], raw_growth[0])
    np.testing.assert_allclose(outputs.realised_income_growth[1], 1.0)
    assert not outputs.growth_clipped[0]
    assert outputs.growth_clipped[1]


def test__income_belief_learning_falls_back_to_zero_mean_on_non_finite_prior_mean():
    # A non-finite incoming posterior mean for the first household (e.g. a
    # corrupted runtime carry-over) propagates to a non-finite posterior
    # mean. With no finite prior to fall back to, it floors to zero. The
    # second household updates normally.
    priors = {
        "income_belief_mu": np.array([0.3, -0.1]),
        "income_belief_p": np.array([0.5, 0.2]),
        "income_belief_rho": np.array([0.8, 0.8]),
        "sigma2_v": np.array([1.0, 1.0]),
        "sigma2_xi": np.array([2.0, 2.0]),
    }
    prior_mean = np.array([np.inf, -0.1])
    prior_variance = np.array([0.5, 0.2])

    outputs = compute_income_belief_learning_outputs(
        current_income=np.array([110.0, 105.0]),
        lagged_income=np.array([100.0, 100.0]),
        priors=priors,
        prior_mean=prior_mean,
        prior_variance=prior_variance,
    )

    assert not np.isfinite(outputs.predicted_mean[0])
    assert outputs.posterior_mean[0] == 0.0
    assert np.isfinite(outputs.posterior_mean[1])
    assert outputs.posterior_fallback_used[0]
    assert not outputs.posterior_fallback_used[1]


def test__income_belief_learning_falls_back_to_prior_variance_on_non_finite_posterior_variance():
    priors = {
        "income_belief_mu": np.array([0.0, 0.0]),
        "income_belief_p": np.array([0.4, 0.2]),
        "income_belief_rho": np.array([0.5, 0.5]),
        "sigma2_v": np.array([np.inf, 1.0]),
        "sigma2_xi": np.array([2.0, 2.0]),
    }

    outputs = compute_income_belief_learning_outputs(
        current_income=np.array([110.0, 105.0]),
        lagged_income=np.array([100.0, 100.0]),
        priors=priors,
    )

    assert not np.isfinite(outputs.predicted_variance[0])
    assert outputs.posterior_variance[0] == pytest.approx(priors["income_belief_p"][0])
    assert np.isfinite(outputs.posterior_variance[1])
    assert outputs.posterior_fallback_used[0]
    assert not outputs.posterior_fallback_used[1]


def test__income_belief_learning_falls_back_to_zero_when_prior_variance_non_finite():
    priors = {
        "income_belief_mu": np.array([0.0]),
        "income_belief_p": np.array([np.inf]),
        "income_belief_rho": np.array([0.5]),
        "sigma2_v": np.array([np.inf]),
        "sigma2_xi": np.array([2.0]),
    }

    outputs = compute_income_belief_learning_outputs(
        current_income=np.array([110.0]),
        lagged_income=np.array([100.0]),
        priors=priors,
    )

    assert not np.isfinite(outputs.predicted_variance[0])
    assert outputs.posterior_variance[0] == 0.0
    assert outputs.posterior_fallback_used[0]
