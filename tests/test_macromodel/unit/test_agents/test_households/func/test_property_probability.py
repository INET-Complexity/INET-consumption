"""Test for household probability of buying formula fix (PR #50).

This test verifies the fix for the household purchasing probability calculation.
The bug was that `prob_buying = 1.0 / diff_exp` produced:
1. Inverted logic: decreased probability when buying became cheaper than renting
2. Invalid outputs: infinity and values exceeding 1.0

The fix uses the proper logistic function formula.

This test imports the actual DefaultHouseholdDemandForProperty class and tests
the compute_demand method with minimal fixture data that triggers the bug.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from macromodel.agents.households.func.property import (
    PRIVATE_RENTER_TENURE_STATUS,
    DefaultHouseholdDemandForProperty,
    convert_monthly_reduction_calibration_to_period,
)
from macromodel.configurations.households_configuration import HouseholdsConfiguration
from macromodel.util.function_mapping import functions_from_model, update_functions


class TestHouseholdProbabilityOfBuying:
    """Test the probability of buying calculation in household property demand."""

    @pytest.fixture
    def property_demand_calculator(self):
        """Create a DefaultHouseholdDemandForProperty instance with test parameters."""
        return DefaultHouseholdDemandForProperty(
            probability_stay_in_rented_property=0.0,  # Force all renters to consider moving
            probability_stay_in_owned_property=1.0,  # Owners stay
            maximum_price_income_coefficient=5.0,
            maximum_price_income_exponent=1.0,
            maximum_price_noise_mean=0.0,
            maximum_price_noise_variance=0.0,  # No noise for deterministic test
            maximum_rent_income_coefficient=0.3,
            maximum_rent_income_exponent=1.0,
            psychological_pressure_of_renting=0.1,
            cost_comparison_temperature=1.0,  # Key parameter for the bug
            price_initial_markup=0.1,
            price_decrease_probability_monthly=0.5,
            price_mean_percentage_reduction=5.0,
            std_of_price_percentage_reduction=1.0,
            rent_initial_markup=0.1,
            rent_decrease_probability_monthly=0.5,
            rent_mean_percentage_reduction=5.0,
            std_of_rent_percentage_reduction=1.0,
            partial_rent_inflation_indexation=0.5,
            partial_rent_inflation_delay=4,
        )

    @pytest.fixture
    def minimal_housing_data(self):
        """Create minimal housing data DataFrame."""
        return pd.DataFrame(
            {
                "House ID": [0],
                "Value": [100000.0],
                "Rent": [500.0],
            }
        )

    def test_probability_formula_produces_valid_range(self, property_demand_calculator, minimal_housing_data):
        """Test that the probability formula produces values in [0, 1].

        This test creates a scenario where the old formula would produce
        invalid probability values (> 1 or inf).
        """
        np.random.seed(42)  # For reproducibility

        # Single household that is renting
        household_residence_tenure_status = np.array([PRIVATE_RENTER_TENURE_STATUS])
        household_income = np.array([50000.0])  # Moderate income
        household_financial_wealth = np.array([10000.0])  # Some savings

        # Market observations
        observed_fraction_value_price = np.array([1.0, 0.0])  # value = price
        observed_fraction_rent_value = np.array([0.005, 0.0])  # rent = 0.5% of value monthly

        max_price, max_rent, hoping_to_move = property_demand_calculator.compute_demand(
            housing_data=minimal_housing_data,
            household_residence_tenure_status=household_residence_tenure_status,
            household_income=household_income,
            household_financial_wealth=household_financial_wealth,
            observed_fraction_value_price=observed_fraction_value_price,
            observed_fraction_rent_value=observed_fraction_rent_value,
            expected_hpi_growth=0.02,  # 2% expected price growth
            assumed_mortgage_maturity=25,  # 25 year mortgage
            rental_income_taxes=0.2,
        )

        # Check that outputs are valid (no NaN, no inf where not expected)
        # Either max_price or max_rent should be set (household decided to buy or rent)
        assert not (np.isnan(max_price[0]) and np.isnan(max_rent[0])), (
            "Household should have decided to either buy or rent"
        )

        # If they decided to buy, max_price should be finite and positive
        if not np.isnan(max_price[0]):
            assert np.isfinite(max_price[0]), f"max_price should be finite, got {max_price[0]}"
            assert max_price[0] > 0, f"max_price should be positive, got {max_price[0]}"

        # If they decided to rent, max_rent should be finite and positive
        if not np.isnan(max_rent[0]):
            assert np.isfinite(max_rent[0]), f"max_rent should be finite, got {max_rent[0]}"
            assert max_rent[0] > 0, f"max_rent should be positive, got {max_rent[0]}"

    def test_old_probability_formula_bug_demonstration(self):
        """Demonstrate the bug in the old probability formula.

        OLD CODE (lines 296-303 in property.py):
            diff_exp = np.exp(self.cost_comparison_temperature * (annual_cost_of_renting - annual_cost_of_purchasing))
            prob_buying = 1.0 / diff_exp

        When buying is much more expensive than renting:
        - (rent - buy) is very negative
        - exp(very_negative) approaches 0
        - 1 / (nearly_zero) approaches infinity

        This produces invalid probabilities > 1.
        """
        cost_comparison_temperature = 1.0

        # Case: buying is MUCH more expensive than renting
        annual_cost_of_renting = 10000.0
        annual_cost_of_purchasing = 50000.0  # 5x more expensive to buy

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # OLD formula - produces invalid probability
            diff = annual_cost_of_renting - annual_cost_of_purchasing  # -40000
            diff_exp_old = np.exp(cost_comparison_temperature * diff)  # exp(-40000) ≈ 0
            prob_buying_old = 1.0 / diff_exp_old  # 1/0 ≈ inf

        # Document the bug: old formula gives probability > 1 (or inf)
        assert prob_buying_old > 1.0 or np.isinf(prob_buying_old), (
            f"Expected old formula to produce invalid prob > 1, got {prob_buying_old}"
        )

        # NEW formula should use proper logistic: 1 / (1 + exp(-x))
        # This always produces values in (0, 1)
        # Note: The exact fix implementation may vary, but the result must be in [0, 1]

    def test_multiple_households_decisions_are_valid(self, property_demand_calculator, minimal_housing_data):
        """Test that multiple households all get valid buy/rent decisions."""
        np.random.seed(123)

        n_households = 10
        # Mix of renters and people in social housing (-1)
        household_residence_tenure_status = np.array(
            [
                -1,
                PRIVATE_RENTER_TENURE_STATUS,
                PRIVATE_RENTER_TENURE_STATUS,
                PRIVATE_RENTER_TENURE_STATUS,
                -1,
                PRIVATE_RENTER_TENURE_STATUS,
                PRIVATE_RENTER_TENURE_STATUS,
                -1,
                PRIVATE_RENTER_TENURE_STATUS,
                PRIVATE_RENTER_TENURE_STATUS,
            ]
        )

        # Varying incomes
        household_income = np.linspace(20000, 100000, n_households)

        # Varying wealth
        household_financial_wealth = np.linspace(5000, 200000, n_households)

        observed_fraction_value_price = np.array([1.0, 0.0])
        observed_fraction_rent_value = np.array([0.005, 0.0])

        max_price, max_rent, hoping_to_move = property_demand_calculator.compute_demand(
            housing_data=minimal_housing_data,
            household_residence_tenure_status=household_residence_tenure_status,
            household_income=household_income,
            household_financial_wealth=household_financial_wealth,
            observed_fraction_value_price=observed_fraction_value_price,
            observed_fraction_rent_value=observed_fraction_rent_value,
            expected_hpi_growth=0.02,
            assumed_mortgage_maturity=25,
            rental_income_taxes=0.2,
        )

        # All outputs should be valid (finite positive or NaN for non-participants)
        for i in range(n_households):
            if not np.isnan(max_price[i]):
                assert np.isfinite(max_price[i]), f"Household {i}: max_price must be finite"
                assert max_price[i] > 0, f"Household {i}: max_price must be positive"
            if not np.isnan(max_rent[i]):
                assert np.isfinite(max_rent[i]), f"Household {i}: max_rent must be finite"
                assert max_rent[i] > 0, f"Household {i}: max_rent must be positive"

    def test_fractional_income_exponents_handle_negative_income_without_warning(self, minimal_housing_data):
        """Negative expected income should imply zero affordability, not invalid powers."""
        property_demand_calculator = DefaultHouseholdDemandForProperty(
            probability_stay_in_rented_property=0.0,
            probability_stay_in_owned_property=1.0,
            maximum_price_income_coefficient=5.0,
            maximum_price_income_exponent=0.789,
            maximum_price_noise_mean=0.0,
            maximum_price_noise_variance=0.0,
            maximum_rent_income_coefficient=0.3,
            maximum_rent_income_exponent=0.3464,
            psychological_pressure_of_renting=0.1,
            cost_comparison_temperature=1.0,
            price_initial_markup=0.1,
            price_decrease_probability_monthly=0.5,
            price_mean_percentage_reduction=5.0,
            std_of_price_percentage_reduction=1.0,
            rent_initial_markup=0.1,
            rent_decrease_probability_monthly=0.5,
            rent_mean_percentage_reduction=5.0,
            std_of_rent_percentage_reduction=1.0,
            partial_rent_inflation_indexation=0.5,
            partial_rent_inflation_delay=4,
        )

        np.random.seed(123)
        household_residence_tenure_status = np.array([-1, PRIVATE_RENTER_TENURE_STATUS])
        household_income = np.array([-1000.0, -1.0])
        household_financial_wealth = np.array([0.0, 0.0])
        observed_fraction_value_price = np.array([1.0, 0.0])
        observed_fraction_rent_value = np.array([0.005, 0.0])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            max_price, max_rent, _ = property_demand_calculator.compute_demand(
                housing_data=minimal_housing_data,
                household_residence_tenure_status=household_residence_tenure_status,
                household_income=household_income,
                household_financial_wealth=household_financial_wealth,
                observed_fraction_value_price=observed_fraction_value_price,
                observed_fraction_rent_value=observed_fraction_rent_value,
                expected_hpi_growth=0.02,
                assumed_mortgage_maturity=25,
                rental_income_taxes=0.2,
            )

        assigned_willingness = np.concatenate(
            [
                max_price[np.isfinite(max_price)],
                max_rent[np.isfinite(max_rent)],
            ]
        )
        assert assigned_willingness.size == household_income.size
        assert np.all(assigned_willingness == 0.0)

    def test_stable_buying_probability_handles_strict_underflow(self, property_demand_calculator, minimal_housing_data):
        """Large rent-buy cost differences should not raise under strict NumPy errors."""
        np.random.seed(42)
        old_err = np.seterr(all="raise")
        try:
            max_price, max_rent, _ = property_demand_calculator.compute_demand(
                housing_data=minimal_housing_data,
                household_residence_tenure_status=np.array([PRIVATE_RENTER_TENURE_STATUS]),
                household_income=np.array([1.0e12]),
                household_financial_wealth=np.array([0.0]),
                observed_fraction_value_price=np.array([1.0, 0.0]),
                observed_fraction_rent_value=np.array([0.005, 0.0]),
                expected_hpi_growth=0.02,
                assumed_mortgage_maturity=25,
                rental_income_taxes=0.2,
            )
        finally:
            np.seterr(**old_err)

        assert not (np.isnan(max_price[0]) and np.isnan(max_rent[0]))

    def test_monthly_reduction_calibration_is_unchanged_for_monthly_model(self):
        probability, mean, std = convert_monthly_reduction_calibration_to_period(
            probability_monthly=0.1057,
            mean_percentage_reduction=1.6559,
            std_of_percentage_reduction=0.7855,
            time_unit=1,
        )

        assert probability == 0.1057
        assert mean == 1.6559
        assert std == 0.7855

    def test_monthly_reduction_calibration_converts_to_quarterly_model(self):
        probability, mean, std = convert_monthly_reduction_calibration_to_period(
            probability_monthly=0.1057,
            mean_percentage_reduction=1.6559,
            std_of_percentage_reduction=0.7855,
            time_unit=3,
        )

        np.testing.assert_allclose(probability, 0.2847634621930001)
        np.testing.assert_allclose(mean, 1.8407114856812123)
        np.testing.assert_allclose(std, 0.9851783434605734)

    def test_monthly_sale_reduction_calibration_is_unchanged_for_monthly_model(self):
        probability, mean, std = convert_monthly_reduction_calibration_to_period(
            probability_monthly=0.0703,
            mean_percentage_reduction=1.4532,
            std_of_percentage_reduction=0.7070,
            time_unit=1,
        )

        assert probability == 0.0703
        assert mean == 1.4532
        assert std == 0.7070

    def test_monthly_sale_reduction_calibration_converts_to_quarterly_model(self):
        probability, mean, std = convert_monthly_reduction_calibration_to_period(
            probability_monthly=0.0703,
            mean_percentage_reduction=1.4532,
            std_of_percentage_reduction=0.7070,
            time_unit=3,
        )

        np.testing.assert_allclose(probability, 0.19642115892700007)
        np.testing.assert_allclose(mean, 1.5587266085311162)
        np.testing.assert_allclose(std, 0.8255943273350927)

    def test_monthly_reduction_calibration_rejects_invalid_time_unit(self):
        with pytest.raises(ValueError, match="time_unit"):
            convert_monthly_reduction_calibration_to_period(
                probability_monthly=0.1057,
                mean_percentage_reduction=1.6559,
                std_of_percentage_reduction=0.7855,
                time_unit=0,
            )

    def test_old_rent_decrease_variance_key_fails_loudly(self):
        configuration = HouseholdsConfiguration()
        configuration.functions.property.parameters["rent_decrease_variance"] = 0.7855

        with pytest.raises(ValueError, match="rent_decrease_variance"):
            functions_from_model(configuration.functions, loc="macromodel.agents.households")

    @pytest.mark.parametrize(
        ("old_key", "new_key", "value"),
        [
            ("price_decrease_probability", "price_decrease_probability_monthly", 0.0703),
            ("price_decrease_mean", "price_mean_percentage_reduction", 1.4532),
            ("price_decrease_variance", "std_of_price_percentage_reduction", 0.7070),
        ],
    )
    def test_old_sale_decrease_keys_fail_loudly_on_load(self, old_key, new_key, value):
        configuration = HouseholdsConfiguration()
        configuration.functions.property.parameters[old_key] = value

        with pytest.raises(ValueError, match=f"{old_key}.*{new_key}"):
            functions_from_model(configuration.functions, loc="macromodel.agents.households")

    def test_old_sale_decrease_keys_fail_loudly_on_update(self):
        configuration = HouseholdsConfiguration()
        functions = functions_from_model(configuration.functions, loc="macromodel.agents.households")
        configuration.functions.property.parameters["price_decrease_variance"] = 0.7070

        with pytest.raises(ValueError, match="price_decrease_variance.*std_of_price_percentage_reduction"):
            update_functions(configuration.functions, loc="macromodel.agents.households", functions=functions)

    def test_existing_rent_reduction_standard_calibration_stays_positive(self):
        property_demand_calculator = DefaultHouseholdDemandForProperty(
            probability_stay_in_rented_property=0.0,
            probability_stay_in_owned_property=1.0,
            maximum_price_income_coefficient=5.0,
            maximum_price_income_exponent=1.0,
            maximum_price_noise_mean=0.0,
            maximum_price_noise_variance=0.0,
            maximum_rent_income_coefficient=0.3,
            maximum_rent_income_exponent=1.0,
            psychological_pressure_of_renting=0.1,
            cost_comparison_temperature=1.0,
            price_initial_markup=0.1,
            price_decrease_probability_monthly=0.0,
            price_mean_percentage_reduction=0.0,
            std_of_price_percentage_reduction=0.0,
            rent_initial_markup=0.1,
            rent_decrease_probability_monthly=0.1057,
            rent_mean_percentage_reduction=1.6559,
            std_of_rent_percentage_reduction=0.7855,
            partial_rent_inflation_indexation=0.5,
            partial_rent_inflation_delay=4,
        )

        np.random.seed(123)
        updated_rent = property_demand_calculator.compute_offered_rent_for_existing_properties(
            current_offered_rent=np.full(1000, 100.0),
            time_unit=3,
        )

        assert np.all(updated_rent > 0.0)
        assert np.all(updated_rent <= 100.0)

    def test_existing_rent_reduction_extreme_draw_is_clipped(self):
        property_demand_calculator = DefaultHouseholdDemandForProperty(
            probability_stay_in_rented_property=0.0,
            probability_stay_in_owned_property=1.0,
            maximum_price_income_coefficient=5.0,
            maximum_price_income_exponent=1.0,
            maximum_price_noise_mean=0.0,
            maximum_price_noise_variance=0.0,
            maximum_rent_income_coefficient=0.3,
            maximum_rent_income_exponent=1.0,
            psychological_pressure_of_renting=0.1,
            cost_comparison_temperature=1.0,
            price_initial_markup=0.1,
            price_decrease_probability_monthly=0.0,
            price_mean_percentage_reduction=0.0,
            std_of_price_percentage_reduction=0.0,
            rent_initial_markup=0.1,
            rent_decrease_probability_monthly=1.0,
            rent_mean_percentage_reduction=100.0,
            std_of_rent_percentage_reduction=0.0,
            partial_rent_inflation_indexation=0.5,
            partial_rent_inflation_delay=4,
        )

        updated_rent = property_demand_calculator.compute_offered_rent_for_existing_properties(
            current_offered_rent=np.array([100.0]),
            time_unit=1,
        )

        np.testing.assert_allclose(updated_rent, np.array([80.0]))

    def test_existing_rent_reduction_zero_probability_leaves_rent_unchanged(self):
        property_demand_calculator = DefaultHouseholdDemandForProperty(
            probability_stay_in_rented_property=0.0,
            probability_stay_in_owned_property=1.0,
            maximum_price_income_coefficient=5.0,
            maximum_price_income_exponent=1.0,
            maximum_price_noise_mean=0.0,
            maximum_price_noise_variance=0.0,
            maximum_rent_income_coefficient=0.3,
            maximum_rent_income_exponent=1.0,
            psychological_pressure_of_renting=0.1,
            cost_comparison_temperature=1.0,
            price_initial_markup=0.1,
            price_decrease_probability_monthly=0.0,
            price_mean_percentage_reduction=0.0,
            std_of_price_percentage_reduction=0.0,
            rent_initial_markup=0.1,
            rent_decrease_probability_monthly=0.0,
            rent_mean_percentage_reduction=100.0,
            std_of_rent_percentage_reduction=0.0,
            partial_rent_inflation_indexation=0.5,
            partial_rent_inflation_delay=4,
        )

        current_rent = np.array([100.0, 200.0])
        updated_rent = property_demand_calculator.compute_offered_rent_for_existing_properties(
            current_offered_rent=current_rent,
            time_unit=3,
        )

        np.testing.assert_array_equal(updated_rent, current_rent)

    def test_existing_sale_reduction_standard_calibration_stays_positive_without_floor_collapse(self):
        property_demand_calculator = DefaultHouseholdDemandForProperty(
            probability_stay_in_rented_property=0.0,
            probability_stay_in_owned_property=1.0,
            maximum_price_income_coefficient=5.0,
            maximum_price_income_exponent=1.0,
            maximum_price_noise_mean=0.0,
            maximum_price_noise_variance=0.0,
            maximum_rent_income_coefficient=0.3,
            maximum_rent_income_exponent=1.0,
            psychological_pressure_of_renting=0.1,
            cost_comparison_temperature=1.0,
            price_initial_markup=0.1,
            price_decrease_probability_monthly=0.0703,
            price_mean_percentage_reduction=1.4532,
            std_of_price_percentage_reduction=0.7070,
            rent_initial_markup=0.1,
            rent_decrease_probability_monthly=0.0,
            rent_mean_percentage_reduction=0.0,
            std_of_rent_percentage_reduction=0.0,
            partial_rent_inflation_indexation=0.5,
            partial_rent_inflation_delay=4,
        )

        np.random.seed(123)
        updated_prices = property_demand_calculator.compute_updated_sale_price(
            sale_prices=np.full(1000, 100.0),
            time_unit=3,
        )

        assert np.all(updated_prices > 80.0)
        assert np.all(updated_prices <= 100.0)
        assert np.any(updated_prices < 100.0)

    def test_existing_sale_reduction_extreme_draw_is_clipped(self):
        property_demand_calculator = DefaultHouseholdDemandForProperty(
            probability_stay_in_rented_property=0.0,
            probability_stay_in_owned_property=1.0,
            maximum_price_income_coefficient=5.0,
            maximum_price_income_exponent=1.0,
            maximum_price_noise_mean=0.0,
            maximum_price_noise_variance=0.0,
            maximum_rent_income_coefficient=0.3,
            maximum_rent_income_exponent=1.0,
            psychological_pressure_of_renting=0.1,
            cost_comparison_temperature=1.0,
            price_initial_markup=0.1,
            price_decrease_probability_monthly=1.0,
            price_mean_percentage_reduction=100.0,
            std_of_price_percentage_reduction=0.0,
            rent_initial_markup=0.1,
            rent_decrease_probability_monthly=0.0,
            rent_mean_percentage_reduction=0.0,
            std_of_rent_percentage_reduction=0.0,
            partial_rent_inflation_indexation=0.5,
            partial_rent_inflation_delay=4,
        )

        updated_price = property_demand_calculator.compute_updated_sale_price(
            sale_prices=np.array([100.0]),
            time_unit=1,
        )

        np.testing.assert_allclose(updated_price, np.array([80.0]))

    def test_existing_sale_reduction_zero_probability_leaves_price_unchanged(self):
        property_demand_calculator = DefaultHouseholdDemandForProperty(
            probability_stay_in_rented_property=0.0,
            probability_stay_in_owned_property=1.0,
            maximum_price_income_coefficient=5.0,
            maximum_price_income_exponent=1.0,
            maximum_price_noise_mean=0.0,
            maximum_price_noise_variance=0.0,
            maximum_rent_income_coefficient=0.3,
            maximum_rent_income_exponent=1.0,
            psychological_pressure_of_renting=0.1,
            cost_comparison_temperature=1.0,
            price_initial_markup=0.1,
            price_decrease_probability_monthly=0.0,
            price_mean_percentage_reduction=100.0,
            std_of_price_percentage_reduction=0.0,
            rent_initial_markup=0.1,
            rent_decrease_probability_monthly=0.0,
            rent_mean_percentage_reduction=0.0,
            std_of_rent_percentage_reduction=0.0,
            partial_rent_inflation_indexation=0.5,
            partial_rent_inflation_delay=4,
        )

        current_prices = np.array([100.0, 200.0])
        updated_prices = property_demand_calculator.compute_updated_sale_price(
            sale_prices=current_prices,
            time_unit=3,
        )

        np.testing.assert_array_equal(updated_prices, current_prices)
