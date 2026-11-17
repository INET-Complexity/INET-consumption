"""Rent indexation is symmetric and floored at an annual rent/value yield (issue #197)."""

import numpy as np
import pytest

from macromodel.agents.households.func.property import DefaultHouseholdDemandForProperty
from macromodel.configurations.households_configuration import HouseholdsConfiguration


def make_property_function(minimum_annual_rental_yield: float = 0.02, indexation: float = 1.0):
    return DefaultHouseholdDemandForProperty(
        probability_stay_in_rented_property=0.5,
        probability_stay_in_owned_property=0.5,
        maximum_price_income_coefficient=5.0,
        maximum_price_income_exponent=1.0,
        maximum_price_noise_mean=0.0,
        maximum_price_noise_variance=0.0,
        maximum_rent_income_coefficient=0.3,
        maximum_rent_income_exponent=1.0,
        psychological_pressure_of_renting=0.1,
        cost_comparison_temperature=1.0,
        price_initial_markup=0.0,
        price_decrease_probability_monthly=0.0,
        price_mean_percentage_reduction=0.0,
        std_of_price_percentage_reduction=0.0,
        rent_initial_markup=0.0,
        rent_decrease_probability_monthly=0.0,
        rent_mean_percentage_reduction=0.0,
        std_of_rent_percentage_reduction=0.0,
        partial_rent_inflation_indexation=indexation,
        partial_rent_inflation_delay=1,
        minimum_annual_rental_yield=minimum_annual_rental_yield,
    )


VALUE = np.array([1000.0, 1000.0])
QUARTER = 3


def test_rents_fall_with_deflation_above_the_floor():
    rent = make_property_function().compute_rent(
        current_rent=np.array([10.0, 20.0]),
        historic_inflation=np.array([0.0, -0.05]),
        property_value=VALUE,
        time_unit=QUARTER,
    )
    np.testing.assert_allclose(rent, [9.5, 19.0])


def test_rents_rise_with_inflation():
    rent = make_property_function().compute_rent(
        current_rent=np.array([10.0]),
        historic_inflation=np.array([0.03]),
        property_value=VALUE[:1],
        time_unit=QUARTER,
    )
    np.testing.assert_allclose(rent, [10.3])


def test_floor_is_annual_yield_scaled_to_the_period():
    # 2% a year on 1000 is 20 a year: 5 a quarter, 20/12 a month.
    function = make_property_function()
    for time_unit, floor in [(3, 5.0), (1, 20.0 / 12.0), (12, 20.0)]:
        rent = function.compute_rent(
            current_rent=np.array([0.1]),
            historic_inflation=np.array([-0.5]),
            property_value=VALUE[:1],
            time_unit=time_unit,
        )
        np.testing.assert_allclose(rent, [floor])


def test_floor_binds_only_below_it():
    rent = make_property_function().compute_rent(
        current_rent=np.array([6.0, 4.0]),
        historic_inflation=np.array([-0.05]),
        property_value=VALUE,
        time_unit=QUARTER,
    )
    np.testing.assert_allclose(rent, [5.7, 5.0])


def test_new_let_asking_rent_is_floored_by_the_rent_update():
    # compute_offered_rent_for_new_properties leaves the floor to compute_rent,
    # which update_rent applies to every rent, new listings included.
    function = make_property_function()
    offered = function.compute_offered_rent_for_new_properties(
        property_value=VALUE,
        observed_fraction_rent_value=np.array([0.001, 0.0]),
    )
    np.testing.assert_allclose(offered, [1.0, 1.0])
    rent = function.compute_rent(
        current_rent=offered,
        historic_inflation=np.array([0.0]),
        property_value=VALUE,
        time_unit=QUARTER,
    )
    np.testing.assert_allclose(rent, [5.0, 5.0])


def test_zero_floor_is_pure_symmetric_indexation():
    rent = make_property_function(minimum_annual_rental_yield=0.0).compute_rent(
        current_rent=np.array([0.1]),
        historic_inflation=np.array([-0.5]),
        property_value=VALUE[:1],
        time_unit=QUARTER,
    )
    np.testing.assert_allclose(rent, [0.05])


@pytest.mark.parametrize("bad", [-0.01, np.nan, np.inf])
def test_invalid_floor_is_rejected(bad):
    with pytest.raises(ValueError, match="minimum_annual_rental_yield"):
        make_property_function(minimum_annual_rental_yield=bad)


def test_default_configuration_disables_the_floor():
    parameters = HouseholdsConfiguration().functions.property.parameters
    assert parameters["minimum_annual_rental_yield"] == 0.0
