import numpy as np

from macromodel.agents.households.income_belief_learning import compute_income_belief_learning_outputs


def test__income_belief_learning_outputs_use_correct_variance_mapping_and_prediction_error():
    current_income = np.array([110.0, 90.0])
    lagged_income = np.array([100.0, 100.0])
    priors = {
        "income_belief_mu": np.array([0.1, -0.2]),
        "income_belief_p": np.array([0.5, 0.2]),
        "income_belief_rho": np.array([0.8, 0.5]),
        "sigma2_xi": np.array([3.0, 1.5]),
        "sigma2_v": np.array([2.0, 1.0]),
    }

    outputs = compute_income_belief_learning_outputs(
        current_income=current_income,
        lagged_income=lagged_income,
        priors=priors,
        common_permanent_income_log_ratio=0.2,
    )

    expected_signal = np.log(current_income) - np.log(lagged_income) - 0.2
    expected_predicted_mean = priors["income_belief_rho"] * priors["income_belief_mu"]
    expected_predicted_variance = (priors["income_belief_rho"] ** 2) * priors["income_belief_p"] + priors["sigma2_v"]
    expected_prediction_error = expected_signal - expected_predicted_mean
    expected_kalman_gain = expected_predicted_variance / (expected_predicted_variance + priors["sigma2_xi"])
    expected_posterior_mean = expected_predicted_mean + expected_kalman_gain * expected_prediction_error
    expected_posterior_variance = (1.0 - expected_kalman_gain) * expected_predicted_variance

    np.testing.assert_allclose(outputs.income_signal, expected_signal)
    np.testing.assert_allclose(outputs.predicted_mean, expected_predicted_mean)
    np.testing.assert_allclose(outputs.predicted_variance, expected_predicted_variance)
    np.testing.assert_allclose(outputs.prediction_error, expected_prediction_error)
    np.testing.assert_allclose(outputs.kalman_gain, expected_kalman_gain)
    assert np.all(outputs.kalman_gain >= 0.0)
    assert np.all(outputs.kalman_gain <= 1.0)
    np.testing.assert_allclose(outputs.posterior_mean, expected_posterior_mean)
    np.testing.assert_allclose(outputs.posterior_variance, expected_posterior_variance)
    np.testing.assert_allclose(outputs.uncertainty_delta, expected_posterior_variance - priors["income_belief_p"])
    np.testing.assert_allclose(
        outputs.permanent_income_log_ratio,
        0.2 + outputs.posterior_mean / (1.0 - priors["income_belief_rho"]),
    )


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
    expected_signal = np.log(1.2)
    expected_prediction_error = expected_signal - expected_predicted_mean
    expected_kalman_gain = expected_predicted_variance / (expected_predicted_variance + 2.0)
    np.testing.assert_allclose(outputs.prior_mean, 0.4)
    np.testing.assert_allclose(outputs.prior_variance, 0.25)
    np.testing.assert_allclose(outputs.predicted_mean, expected_predicted_mean)
    np.testing.assert_allclose(outputs.predicted_variance, expected_predicted_variance)
    np.testing.assert_allclose(outputs.prediction_error, expected_prediction_error)
    np.testing.assert_allclose(outputs.kalman_gain, expected_kalman_gain)
    expected_posterior_variance = (1.0 - expected_kalman_gain) * expected_predicted_variance
    np.testing.assert_allclose(outputs.uncertainty_delta, expected_posterior_variance - 0.25)
