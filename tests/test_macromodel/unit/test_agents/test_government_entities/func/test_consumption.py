import numpy as np

from macromodel.agents.government_entities.func.consumption import (
    ConstantGrowthGovernmentConsumptionSetter,
)


def test_initial_price_normalized_weights_preserve_total_consumption():
    setter = ConstantGrowthGovernmentConsumptionSetter(
        consistency=1.0,
        default_growth=0.10,
        sectoral_weights="initial_price_normalized",
    )

    previous_desired = np.array([20.0, 30.0, 50.0])
    current_prices = np.array([2.0, 1.0, 3.0])
    target = setter.compute_target_consumption(
        previous_desired_government_consumption=previous_desired,
        model=None,
        historic_total_consumption=None,
        initial_good_prices=np.ones(3),
        current_good_prices=current_prices,
        expected_growth=0.0,
        expected_inflation=0.05,
        current_time=1,
        exogenous_total_consumption=None,
        forecasting_window=1,
    )

    expected_total = 1.05 * 1.10 * previous_desired.sum()
    expected_weights = current_prices * previous_desired / np.sum(current_prices * previous_desired)
    np.testing.assert_allclose(target.sum(), expected_total)
    np.testing.assert_allclose(target / target.sum(), expected_weights)


def test_initial_fixed_weights_ignore_relative_prices():
    setter = ConstantGrowthGovernmentConsumptionSetter(
        consistency=1.0,
        default_growth=0.10,
        sectoral_weights="initial_fixed",
    )

    previous_desired = np.array([20.0, 30.0, 50.0])
    target = setter.compute_target_consumption(
        previous_desired_government_consumption=previous_desired,
        model=None,
        historic_total_consumption=None,
        initial_good_prices=np.ones(3),
        current_good_prices=np.array([2.0, 1.0, 3.0]),
        expected_growth=0.0,
        expected_inflation=0.05,
        current_time=1,
        exogenous_total_consumption=None,
        forecasting_window=1,
    )

    expected_total = 1.05 * 1.10 * previous_desired.sum()
    expected_weights = previous_desired / previous_desired.sum()
    np.testing.assert_allclose(target.sum(), expected_total)
    np.testing.assert_allclose(target / target.sum(), expected_weights)
