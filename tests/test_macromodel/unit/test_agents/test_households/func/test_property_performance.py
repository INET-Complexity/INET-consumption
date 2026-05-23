"""Performance test for household property market operations.

This module tests performance-critical operations in the household
property market, particularly the housing listing logic.
"""

import time

import numpy as np
import pandas as pd


class _StubPropertyDemand:
    def __init__(self, households_hoping_to_move: np.ndarray):
        self.households_hoping_to_move = households_hoping_to_move

    def compute_demand(self, **kwargs):
        n_households = len(self.households_hoping_to_move)
        return (
            np.zeros(n_households),
            np.zeros(n_households),
            self.households_hoping_to_move,
        )

    def compute_initial_sale_price(self, property_values):
        return np.asarray(property_values, dtype=float)

    def compute_updated_sale_price(self, sale_prices):
        return np.asarray(sale_prices, dtype=float)

    def compute_offered_rent_for_new_properties(self, property_value, observed_fraction_rent_value):
        return (
            observed_fraction_rent_value[0] * np.asarray(property_value, dtype=float) + observed_fraction_rent_value[1]
        )

    def compute_offered_rent_for_existing_properties(self, current_offered_rent, time_unit):
        return np.asarray(current_offered_rent, dtype=float)


class TestHouseholdPropertyPerformance:
    """Test performance of household property market operations."""

    def test_household_hoping_to_move_indexing(self):
        """Test that household ID indexing for housing listings is efficient.

        This is a unit test for the specific .isin() operation that was causing
        the performance bottleneck.
        """
        # Create sample data similar to what households.py processes
        n_households = 10000
        n_houses = 5000

        # Boolean array of households hoping to move
        households_hoping_to_move = np.random.random(n_households) < 0.1  # 10% moving

        # House owner IDs
        owner_ids = np.random.randint(0, n_households, size=n_houses)

        # Test the optimized approach (should be fast)
        start_time = time.time()
        household_ids_hoping_to_move = np.flatnonzero(households_hoping_to_move)
        ind_mhr_temp_sale = np.isin(owner_ids, household_ids_hoping_to_move)
        elapsed_time = time.time() - start_time

        # Should complete in < 0.1 seconds even with 10k households and 5k houses
        assert elapsed_time < 0.1, (
            f"NumPy isin took {elapsed_time:.4f}s, expected < 0.1s. "
            "Performance optimization may not be working correctly."
        )

        # Verify correctness - result should be boolean array
        assert isinstance(ind_mhr_temp_sale, np.ndarray)
        assert ind_mhr_temp_sale.dtype == bool
        assert len(ind_mhr_temp_sale) == n_houses

    def test_household_hoping_to_move_mask_lists_matching_owner_id(self, test_households):
        n_households = test_households.ts.current("n_households")
        households_hoping_to_move = np.zeros(n_households, dtype=bool)
        households_hoping_to_move[4] = True
        test_households.functions["property"] = _StubPropertyDemand(households_hoping_to_move)

        housing_data = pd.DataFrame(
            {
                "Value": [100.0, 200.0],
                "Rent": [1.0, 2.0],
                "Corresponding Owner Household ID": [4, 5],
                "Corresponding Inhabitant Household ID": [4, 6],
                "Sale Price": [np.nan, np.nan],
                "Temporarily for Sale": [False, False],
                "Up for Rent": [False, False],
                "Newly on the Rental Market": [False, False],
            },
            index=pd.Index([10, 20], name="Properties"),
        )

        test_households.prepare_housing_market_clearing(
            housing_data=housing_data,
            observed_fraction_value_price=np.array([1.0, 0.0]),
            observed_fraction_rent_value=np.array([0.01, 0.0]),
            expected_hpi_growth=0.0,
            assumed_mortgage_maturity=120,
            rental_income_taxes=0.0,
            time_unit=3,
        )

        assert bool(housing_data.loc[10, "Temporarily for Sale"])
        assert not bool(housing_data.loc[20, "Temporarily for Sale"])

    def test_vacant_negative_one_homes_are_listed_for_rent(self, test_households):
        n_households = test_households.ts.current("n_households")
        test_households.functions["property"] = _StubPropertyDemand(np.zeros(n_households, dtype=bool))
        housing_data = pd.DataFrame(
            {
                "Value": [100.0, 200.0],
                "Rent": [1.0, 2.0],
                "Corresponding Owner Household ID": [4, 5],
                "Corresponding Inhabitant Household ID": [-1, 6],
                "Sale Price": [np.nan, np.nan],
                "Temporarily for Sale": [False, False],
                "Up for Rent": [False, False],
                "Newly on the Rental Market": [False, False],
            },
            index=pd.Index([10, 20], name="Properties"),
        )

        test_households.prepare_housing_market_clearing(
            housing_data=housing_data,
            observed_fraction_value_price=np.array([1.0, 0.0]),
            observed_fraction_rent_value=np.array([0.01, 0.0]),
            expected_hpi_growth=0.0,
            assumed_mortgage_maturity=120,
            rental_income_taxes=0.0,
            time_unit=3,
        )

        assert bool(housing_data.loc[10, "Up for Rent"])
        assert bool(housing_data.loc[10, "Newly on the Rental Market"])
        assert housing_data.loc[10, "Rent"] == 1.0
        assert not bool(housing_data.loc[20, "Up for Rent"])
