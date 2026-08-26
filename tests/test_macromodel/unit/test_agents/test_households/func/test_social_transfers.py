import numpy as np

from macromodel.agents.households.func.social_transfers import DefaultSocialTransfersSetter


class ZeroPredictionModel:
    def predict(self, values):
        return np.zeros(values.shape[0])


def test__default_social_transfers_returns_zero_for_a_zero_budget():
    setter = DefaultSocialTransfersSetter(independents=["income"])
    values = np.array([[2.0], [3.0]])

    result = setter.get_social_transfers(
        n_households=2,
        total_other_social_transfers=0.0,
        current_independents=values,
        initial_independents=values,
        model=ZeroPredictionModel(),
    )

    np.testing.assert_allclose(result, np.zeros(2))


def test__default_social_transfers_uses_equal_fallback_for_zero_predictions_without_mutating_inputs():
    setter = DefaultSocialTransfersSetter(independents=["income"])
    values = np.array([[2.0], [3.0], [5.0]])
    original_values = values.copy()

    result = setter.get_social_transfers(
        n_households=3,
        total_other_social_transfers=12.0,
        current_independents=values,
        initial_independents=values,
        model=ZeroPredictionModel(),
    )

    np.testing.assert_allclose(result, np.full(3, 4.0))
    np.testing.assert_array_equal(values, original_values)
