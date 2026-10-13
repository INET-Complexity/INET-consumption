import numpy as np
import pytest

from macromodel.agents.households.func.social_transfers import (
    DefaultSocialTransfersSetter,
    EqualSocialTransfersSetter,
    InitialShareSocialTransfersSetter,
)


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


@pytest.mark.parametrize("budget", [np.nan, np.inf, -1.0])
def test__default_social_transfers_rejects_invalid_budgets(budget):
    setter = DefaultSocialTransfersSetter(independents=["income"])

    with pytest.raises(ValueError, match="finite non-negative"):
        setter.get_social_transfers(
            n_households=2,
            total_other_social_transfers=budget,
            current_independents=np.ones((2, 1)),
            initial_independents=np.ones((2, 1)),
            model=ZeroPredictionModel(),
        )


def test__equal_social_transfers_handles_an_empty_household_population():
    result = EqualSocialTransfersSetter(independents=[]).get_social_transfers(
        n_households=0,
        total_other_social_transfers=0.0,
        current_independents=np.empty((0, 0)),
        initial_independents=np.empty((0, 0)),
        model=None,
    )

    np.testing.assert_array_equal(result, np.zeros(0))


def test_initial_share_setter_preserves_initial_allocation_shares():
    setter = InitialShareSocialTransfersSetter(independents=["Income_Other_Social_Transfers"])
    initial = np.array([[0.0], [10.0], [30.0], [0.0]])
    out = setter.get_social_transfers(
        n_households=4,
        total_other_social_transfers=80.0,
        current_independents=np.full((4, 1), 5.0),
        initial_independents=initial,
        model=None,
    )
    np.testing.assert_allclose(out, [0.0, 20.0, 60.0, 0.0])


def test_initial_share_setter_falls_back_to_equal_without_initial_shares():
    setter = InitialShareSocialTransfersSetter(independents=["Income_Other_Social_Transfers"])
    out = setter.get_social_transfers(4, 8.0, np.zeros((4, 1)), np.zeros((4, 1)), None)
    np.testing.assert_allclose(out, [2.0, 2.0, 2.0, 2.0])


def test_initial_share_setter_requires_a_single_independent():
    setter = InitialShareSocialTransfersSetter(independents=["Income", "Debt"])
    with pytest.raises(ValueError, match="exactly one independent"):
        setter.get_social_transfers(2, 1.0, np.ones((2, 2)), np.ones((2, 2)), None)
