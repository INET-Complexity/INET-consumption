import numpy as np
import pytest

from macromodel.agents.households.income_belief_learning import compute_income_belief_learning_outputs


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
