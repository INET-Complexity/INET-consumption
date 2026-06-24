import numpy as np
import pytest

from macromodel.agents.households.func.consumption import (
    CESHouseholdConsumption,
    CreditAugmentedConsumption,
    DefaultHouseholdConsumption,
)
from macromodel.agents.households.utils.create_bundle_matrix import create_bundle_matrix
from macromodel.configurations.households_configuration import create_household_bundle


class TestDefaultHouseholdConsumption:
    def test_compute_target_consumption_basic(self):
        """Test basic consumption computation with default parameters."""
        consumption_obj = DefaultHouseholdConsumption(
            consumption_smoothing_fraction=0.5,
            consumption_smoothing_window=4,
            minimum_consumption_fraction=0.1,
        )

        n_households = 5
        n_industries = 4

        # Set up test data
        historic_consumption_sum = np.ones((3, n_households))  # 3 time periods
        saving_rates = np.full(n_households, 0.2)  # 20% saving rate
        income = np.full(n_households, 100.0)  # Income of 100
        household_benefits = np.full(n_households, 80.0)  # Benefits of 80
        consumption_weights = np.full(n_industries, 1.0 / n_industries)  # Equal weights
        consumption_weights_by_income = np.zeros((n_industries, n_households))
        tau_vat = 0.1  # 10% VAT

        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.02,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=historic_consumption_sum,
            saving_rates=saving_rates,
            income=income,
            household_benefits=household_benefits,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=consumption_weights_by_income,
            exogenous_total_consumption=np.zeros(10),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=tau_vat,
            lagged_housing_wealth=np.full(n_households, 120.0),
        )

        # Check output shape
        assert result.shape == (n_households, n_industries)

        # Check all values are non-negative
        assert np.all(result >= 0)

        # Check consumption is proportional to after-tax income
        expected_consumption_per_household = (1 - 0.2) * 100.0 / (1 + tau_vat)  # 80 / 1.1 ≈ 72.73
        expected_consumption_per_industry = expected_consumption_per_household / n_industries

        assert np.allclose(result, expected_consumption_per_industry, rtol=0.1)

    def test_minimum_consumption_threshold(self):
        """Test that minimum consumption threshold is respected."""
        consumption_obj = DefaultHouseholdConsumption(
            consumption_smoothing_fraction=0.0,  # No smoothing
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.5,  # High minimum threshold
        )

        n_households = 3
        n_industries = 2

        # Set up data where benefits would provide higher consumption than income
        historic_consumption_sum = np.ones((2, n_households))
        saving_rates = np.full(n_households, 0.8)  # High saving rate
        income = np.full(n_households, 50.0)  # Low income
        household_benefits = np.full(n_households, 200.0)  # High benefits
        consumption_weights = np.full(n_industries, 0.5)
        consumption_weights_by_income = np.zeros((n_industries, n_households))
        tau_vat = 0.0

        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=historic_consumption_sum,
            saving_rates=saving_rates,
            income=income,
            household_benefits=household_benefits,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=consumption_weights_by_income,
            exogenous_total_consumption=np.zeros(5),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=tau_vat,
        )

        # Expected: minimum_consumption_fraction * (1 - saving_rates) * household_benefits
        # = 0.5 * (1 - 0.8) * 200 = 0.5 * 0.2 * 200 = 20 per household
        expected_per_industry = 20.0 / n_industries

        assert np.allclose(result, expected_per_industry, rtol=0.01)


class TestCESHouseholdConsumption:
    def test_compute_target_consumption_no_substitution_data(self):
        """Test CES consumption falls back to default when substitution data is missing."""
        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.5,
            consumption_smoothing_window=4,
            minimum_consumption_fraction=0.1,
            elasticity_of_substitution=2.0,
        )

        default_consumption = DefaultHouseholdConsumption(
            consumption_smoothing_fraction=0.5,
            consumption_smoothing_window=4,
            minimum_consumption_fraction=0.1,
        )

        n_households = 3
        n_industries = 4

        # Set up identical test data
        test_args = {
            "expected_inflation": 0.02,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((3, n_households)),
            "saving_rates": np.full(n_households, 0.2),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.full(n_households, 80.0),
            "consumption_weights": np.full(n_industries, 0.25),
            "consumption_weights_by_income": np.zeros((n_industries, n_households)),
            "exogenous_total_consumption": np.zeros(10),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.1,
            "lagged_housing_wealth": np.full(n_households, 120.0),
        }

        # CES without substitution data (should fall back to default)
        ces_result = ces_consumption.compute_target_consumption(**test_args)

        # Default consumption
        default_result = default_consumption.compute_target_consumption(**test_args)

        # Should be identical
        assert np.allclose(ces_result, default_result)

    def test_ces_substitution_within_bundles(self):
        """Test CES substitution behavior within bundles."""
        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.0,  # No smoothing for cleaner test
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            elasticity_of_substitution=2.0,
        )

        n_households = 2
        n_industries = 4

        # Create two bundles: [0, 1] and [2, 3]
        bundles = [[0, 1], [2, 3]]
        bundles_grouped = create_household_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(bundles_grouped))

        # Initial setup
        consumption_weights = np.array([0.2, 0.3, 0.1, 0.4])  # Different initial weights
        initial_prices = np.array([1.0, 1.0, 1.0, 1.0])
        current_prices = np.array([2.0, 1.0, 1.0, 2.0])  # Industry 0 and 3 doubled in price
        initial_taxes = np.array([0.0, 0.0, 0.0, 0.0])
        current_taxes = np.array([0.0, 0.0, 0.0, 0.0])

        test_args = {
            "expected_inflation": 0.0,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((2, n_households)),
            "saving_rates": np.zeros(n_households),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.zeros(n_households),
            "consumption_weights": consumption_weights,
            "consumption_weights_by_income": np.zeros((n_industries, n_households)),
            "exogenous_total_consumption": np.zeros(5),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.0,
            "lagged_housing_wealth": np.full(n_households, 120.0),
            "prices": current_prices,
            "initial_prices": initial_prices,
            "taxes": current_taxes,
            "initial_taxes": initial_taxes,
            "bundle_matrix": bundle_matrix,
        }

        result = ces_consumption.compute_target_consumption(**test_args)

        # Check shape
        assert result.shape == (n_households, n_industries)

        # With elasticity = 2.0 and price doubling:
        # - Industry 0 price doubled: substitution factor = (2.0)^(-2) = 0.25
        # - Industry 1 price unchanged: substitution factor = 1.0
        # - Industry 2 price unchanged: substitution factor = 1.0
        # - Industry 3 price doubled: substitution factor = (2.0)^(-2) = 0.25

        # Within bundle [0,1]: original weights [0.2, 0.3], after substitution should favor industry 1
        # Within bundle [2,3]: original weights [0.1, 0.4], after substitution should favor industry 2

        # Check that expensive goods (0, 3) have lower consumption than cheaper alternatives (1, 2)
        avg_consumption = np.mean(result, axis=0)
        assert avg_consumption[1] > avg_consumption[0]  # Industry 1 > Industry 0 in bundle [0,1]
        # Industries 2 and 3 converge to similar consumption due to CES substitution balancing
        # industry 2's lower initial weight with industry 3's higher price
        assert np.allclose(avg_consumption[2], avg_consumption[3], rtol=1e-10)  # Should be nearly equal

    def test_ces_falls_back_when_substitution_dimensions_do_not_match(self):
        """Test CES uses default consumption when prices and bundles use different goods dimensions."""
        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            elasticity_of_substitution=2.0,
        )
        default_consumption = DefaultHouseholdConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
        )

        n_households = 2
        n_consumption_goods = 4
        n_price_goods = 6
        bundle_matrix = create_bundle_matrix(np.array([0, 1, 2, 3]))

        test_args = {
            "expected_inflation": 0.0,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((2, n_households)),
            "saving_rates": np.zeros(n_households),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.zeros(n_households),
            "consumption_weights": np.full(n_consumption_goods, 1.0 / n_consumption_goods),
            "consumption_weights_by_income": np.zeros((n_consumption_goods, n_households)),
            "exogenous_total_consumption": np.zeros(5),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.0,
            "prices": np.ones(n_price_goods),
            "initial_prices": np.ones(n_price_goods),
            "taxes": np.zeros(n_price_goods),
            "initial_taxes": np.zeros(n_price_goods),
            "bundle_matrix": bundle_matrix,
        }

        result = ces_consumption.compute_target_consumption(**test_args)
        expected = default_consumption.compute_target_consumption(**test_args)

        assert np.allclose(result, expected)

    def test_ces_bundle_normalization(self):
        """Test that CES substitution preserves bundle totals."""
        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            elasticity_of_substitution=1.5,
        )

        n_households = 1
        n_industries = 3

        # Single bundle with all industries
        bundles = [[0, 1, 2]]
        bundles_grouped = create_household_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(bundles_grouped))

        consumption_weights = np.array([0.5, 0.3, 0.2])
        initial_prices = np.array([1.0, 1.0, 1.0])
        current_prices = np.array([1.5, 1.0, 0.8])  # Varied prices
        initial_taxes = np.zeros(n_industries)
        current_taxes = np.zeros(n_industries)

        # Test with default weights (no substitution)
        test_args_default = {
            "expected_inflation": 0.0,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((2, n_households)),
            "saving_rates": np.zeros(n_households),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.zeros(n_households),
            "consumption_weights": consumption_weights,
            "consumption_weights_by_income": np.zeros((n_industries, n_households)),
            "exogenous_total_consumption": np.zeros(5),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.0,
        }

        # Test with CES substitution
        test_args_ces = test_args_default.copy()
        test_args_ces.update(
            {
                "prices": current_prices,
                "initial_prices": initial_prices,
                "taxes": current_taxes,
                "initial_taxes": initial_taxes,
                "bundle_matrix": bundle_matrix,
            }
        )

        result_default = ces_consumption.compute_target_consumption(**test_args_default)
        result_ces = ces_consumption.compute_target_consumption(**test_args_ces)

        # Total consumption should be preserved
        assert np.allclose(np.sum(result_default), np.sum(result_ces), rtol=1e-10)

        # But individual industry allocations should differ due to substitution
        assert not np.allclose(result_default, result_ces, rtol=0.1)

    def test_empty_bundle_handling(self):
        """Test CES consumption handles empty bundles gracefully."""
        n_industries = 5

        # Create bundles with some empty ones
        bundles = create_household_bundle(n_industries, [[0, 1], [], [3, 4]])  # Bundle 1 is empty
        bundle_matrix = create_bundle_matrix(np.array(bundles))

        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            elasticity_of_substitution=1.0,
        )

        n_households = 1
        consumption_weights = np.full(n_industries, 0.2)
        prices = np.full(n_industries, 1.0)
        taxes = np.zeros(n_industries)

        test_args = {
            "expected_inflation": 0.0,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((2, n_households)),
            "saving_rates": np.zeros(n_households),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.zeros(n_households),
            "consumption_weights": consumption_weights,
            "consumption_weights_by_income": np.zeros((n_industries, n_households)),
            "exogenous_total_consumption": np.zeros(5),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.0,
            "lagged_housing_wealth": np.full(n_households, 120.0),
            "prices": prices,
            "initial_prices": prices,
            "taxes": taxes,
            "initial_taxes": taxes,
            "bundle_matrix": bundle_matrix,
        }

        # Should not raise an error
        result = ces_consumption.compute_target_consumption(**test_args)
        assert result.shape == (n_households, n_industries)
        assert np.all(result >= 0)

    def test_zero_elasticity_edge_case(self):
        """Test CES consumption with zero elasticity (no substitution)."""
        ces_consumption = CESHouseholdConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            elasticity_of_substitution=0.0,  # No substitution
        )

        n_households = 1
        n_industries = 3
        bundles = [[0, 1, 2]]
        bundles_grouped = create_household_bundle(n_industries, bundles)
        bundle_matrix = create_bundle_matrix(np.array(bundles_grouped))

        consumption_weights = np.array([0.5, 0.3, 0.2])
        initial_prices = np.array([1.0, 1.0, 1.0])
        current_prices = np.array([10.0, 1.0, 0.1])  # Extreme price changes
        taxes = np.zeros(n_industries)

        test_args = {
            "expected_inflation": 0.0,
            "current_cpi": 1.0,
            "initial_cpi": 1.0,
            "historic_consumption_sum": np.ones((2, n_households)),
            "saving_rates": np.zeros(n_households),
            "income": np.full(n_households, 100.0),
            "household_benefits": np.zeros(n_households),
            "consumption_weights": consumption_weights,
            "consumption_weights_by_income": np.zeros((n_industries, n_households)),
            "exogenous_total_consumption": np.zeros(5),
            "current_time": 0,
            "take_consumption_weights_by_income_quantile": False,
            "tau_vat": 0.0,
            "prices": current_prices,
            "initial_prices": initial_prices,
            "taxes": taxes,
            "initial_taxes": taxes,
            "bundle_matrix": bundle_matrix,
        }

        result = ces_consumption.compute_target_consumption(**test_args)

        # With zero elasticity, consumption shares should remain unchanged despite price changes
        expected_shares = consumption_weights / np.sum(consumption_weights)
        result_shares = result[0, :] / np.sum(result[0, :])

        assert np.allclose(result_shares, expected_shares, rtol=1e-10)


class TestCreditAugmentedHouseholdConsumption:
    def test_compute_target_consumption_records_log_linear_decomposition_and_mpc(self):
        # partial_adjustment_speed=0.4 (rather than 1.0) keeps this scenario's implied
        # delta_log_consumption under the +-0.5 growth-sanity clip in _evaluate_target,
        # so this test exercises the unclipped decomposition math. See
        # test_compute_target_consumption_clips_extreme_growth below for the clip itself.
        consumption_obj = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            partial_adjustment_speed=0.4,
            liquid_wealth_propensity=0.1,
            illiquid_wealth_propensity=0.2,
            housing_wealth_propensity=0.3,
            house_price_propensity=0.0,
        )

        n_households = 1
        n_industries = 1

        historic_consumption_sum = np.array(
            [
                np.full(n_households, 40.0),
                np.full(n_households, 50.0),
            ]
        )
        saving_rates = np.zeros(n_households)
        income = np.full(n_households, 100.0)
        household_benefits = np.zeros(n_households)
        consumption_weights = np.full(n_industries, 1.0 / n_industries)
        consumption_weights_by_income = np.zeros((n_industries, n_households))
        liquid_wealth = np.full(n_households, 999.0)
        illiquid_wealth = np.full(n_households, 999.0)
        housing_wealth = np.full(n_households, 999.0)
        rent = np.full(n_households, 10.0)
        mortgage_debt = np.full(n_households, 999.0)
        mortgage_payment = np.full(n_households, 5.0)

        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=historic_consumption_sum,
            saving_rates=saving_rates,
            income=income,
            household_benefits=household_benefits,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=consumption_weights_by_income,
            exogenous_total_consumption=np.zeros(5),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=0.0,
            liquid_wealth=liquid_wealth,
            illiquid_wealth=illiquid_wealth,
            housing_wealth=housing_wealth,
            rent=rent,
            mortgage_debt=mortgage_debt,
            mortgage_payment=mortgage_payment,
            owner_occupied=np.ones(n_households),
            mortgagor=np.ones(n_households),
            house_price_index=1.1,
            house_price_growth=0.1,
            lagged_consumption=historic_consumption_sum[-1],
            lagged_income=np.array([80.0]),
            lagged_liquid_wealth=np.array([60.0]),
            lagged_illiquid_wealth=np.array([30.0]),
            lagged_mortgage_debt=np.array([40.0]),
            lagged_consumption_loan_debt=np.array([10.0]),
            lagged_housing_wealth=np.array([120.0]),
            lagged_house_price_index=1.0,
        )

        assert result.shape == (n_households, n_industries)
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)
        assert consumption_obj.last_target_consumption_components is not None
        assert consumption_obj.last_formula_implied_mpc is not None
        assert np.all(np.isfinite(consumption_obj.last_formula_implied_mpc))
        assert np.all(consumption_obj.last_formula_implied_mpc >= 0.0)
        components = consumption_obj.last_target_consumption_components
        assert "target_consumption_permanent_income" in components
        assert "target_consumption_partial_adjustment_gap" in components
        assert "target_consumption_interest_rate_cashflow" in components
        np.testing.assert_allclose(components["target_consumption_real_income"], 100.0)
        np.testing.assert_allclose(components["target_consumption_lagged_real_income"], 80.0)
        np.testing.assert_allclose(components["target_consumption_real_lagged_consumption"], 50.0)
        np.testing.assert_allclose(components["target_consumption_real_net_liquid_assets"], 10.0)
        np.testing.assert_allclose(components["target_consumption_real_illiquid_financial_assets"], 30.0)
        np.testing.assert_allclose(components["target_consumption_real_housing_wealth"], 999.0)
        np.testing.assert_allclose(components["target_consumption_real_lagged_housing_wealth"], 120.0)
        np.testing.assert_allclose(components["target_consumption_liquid_wealth"], 0.01)
        np.testing.assert_allclose(components["target_consumption_illiquid_wealth"], 0.06)
        np.testing.assert_allclose(components["target_consumption_housing_wealth"], 0.36)
        long_run_log_consumption_to_income = 0.01 + 0.06 + 0.36
        partial_adjustment_gap = 0.4 * (np.log(100.0 * np.exp(long_run_log_consumption_to_income)) - np.log(50.0))
        np.testing.assert_allclose(result.sum(axis=1), 50.0 * np.exp(partial_adjustment_gap))
        np.testing.assert_allclose(components["target_consumption_growth_clipped"], 0.0)
        np.testing.assert_allclose(components["target_consumption_delta_log_consumption"], partial_adjustment_gap)
        np.testing.assert_allclose(components["target_consumption_interest_rate_cashflow"], 0.0)
        np.testing.assert_allclose(components["target_consumption_uncertainty"], 0.0)
        np.testing.assert_allclose(components["target_consumption_rent"], 0.0)
        np.testing.assert_allclose(components["target_consumption_mortgage_debt"], 0.0)
        np.testing.assert_allclose(components["target_consumption_mortgage_payment"], 0.0)
        np.testing.assert_allclose(components["target_consumption_owner_occupied"], 1.0)
        np.testing.assert_allclose(components["target_consumption_mortgagor"], 1.0)

    def test_compute_target_consumption_clips_extreme_growth(self):
        # permanent_income_propensity is large and permanent_income_log_ratio is an
        # extreme outlier value, mimicking an unbounded income-belief-learning input
        # (see GH issue #90). Without the +-0.5 growth-sanity clip in _evaluate_target,
        # this would imply >e^10x real consumption growth in a single quarter.
        consumption_obj = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            partial_adjustment_speed=1.0,
            permanent_income_propensity=10.0,
            house_price_propensity=0.0,
        )

        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=np.array([np.full(1, 40.0), np.full(1, 50.0)]),
            saving_rates=np.zeros(1),
            income=np.full(1, 100.0),
            household_benefits=np.zeros(1),
            consumption_weights=np.full(1, 1.0),
            consumption_weights_by_income=np.zeros((1, 1)),
            exogenous_total_consumption=np.zeros(1),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=0.0,
            liquid_wealth=np.zeros(1),
            illiquid_wealth=np.zeros(1),
            housing_wealth=np.zeros(1),
            rent=np.zeros(1),
            mortgage_debt=np.zeros(1),
            mortgage_payment=np.zeros(1),
            owner_occupied=np.ones(1),
            mortgagor=np.ones(1),
            house_price_index=1.0,
            house_price_growth=0.0,
            lagged_consumption=np.full(1, 50.0),
            lagged_income=np.full(1, 80.0),
            lagged_liquid_wealth=np.zeros(1),
            lagged_illiquid_wealth=np.zeros(1),
            lagged_mortgage_debt=np.zeros(1),
            lagged_consumption_loan_debt=np.zeros(1),
            lagged_house_price_index=1.0,
            permanent_income_log_ratio=np.full(1, 5.0),
        )

        components = consumption_obj.last_target_consumption_components
        assert np.all(components["target_consumption_growth_clipped"] == 1.0)
        np.testing.assert_allclose(components["target_consumption_delta_log_consumption"], 0.5)
        np.testing.assert_allclose(result.sum(axis=1), 50.0 * np.exp(0.5))

    def test_compute_target_consumption_excludes_benefits_rent_and_scheduled_mortgage_from_target(self):
        consumption_obj = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            partial_adjustment_speed=1.0,
            liquid_wealth_propensity=0.1,
            illiquid_wealth_propensity=0.2,
            housing_wealth_propensity=0.3,
            house_price_propensity=0.0,
        )

        income = np.array([100.0])
        consumption_weights = np.array([1.0])
        base_args = dict(
            expected_inflation=0.1,
            current_cpi=2.0,
            initial_cpi=1.0,
            historic_consumption_sum=np.array([[80.0]]),
            saving_rates=np.zeros(1),
            income=income,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=np.zeros((1, 1)),
            exogenous_total_consumption=np.zeros(1),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=0.0,
            liquid_wealth=np.array([60.0]),
            illiquid_wealth=np.array([30.0]),
            housing_wealth=np.array([120.0]),
            mortgage_debt=np.array([40.0]),
            owner_occupied=np.array([1.0]),
            mortgagor=np.array([1.0]),
            house_price_index=1.2,
            house_price_growth=0.05,
            lagged_consumption=np.array([80.0]),
            lagged_income=np.array([160.0]),
            lagged_cpi=2.0,
            lagged_liquid_wealth=np.array([60.0]),
            lagged_illiquid_wealth=np.array([30.0]),
            lagged_mortgage_debt=np.array([40.0]),
            lagged_consumption_loan_debt=np.array([10.0]),
            lagged_house_price_index=1.0,
        )

        low_cost_result = consumption_obj.compute_target_consumption(
            **base_args,
            household_benefits=np.array([20.0]),
            rent=np.array([10.0]),
            mortgage_payment=np.array([6.0]),
        )
        high_cost_result = consumption_obj.compute_target_consumption(
            **base_args,
            household_benefits=np.array([2_000.0]),
            rent=np.array([1_000.0]),
            mortgage_payment=np.array([600.0]),
        )
        components = consumption_obj.last_target_consumption_components
        np.testing.assert_allclose(low_cost_result, high_cost_result)
        np.testing.assert_allclose(components["target_consumption_real_income"], 50.0)
        np.testing.assert_allclose(components["target_consumption_permanent_income"], 0.0)
        np.testing.assert_allclose(components["target_consumption_rent"], 0.0)
        np.testing.assert_allclose(components["target_consumption_mortgage_payment"], 0.0)
        np.testing.assert_allclose(components["target_consumption_rent_diagnostic"], 1_000.0)
        np.testing.assert_allclose(components["target_consumption_mortgage_payment_diagnostic"], 600.0)

    def test_continuous_wealth_calibration_off_by_default(self):
        consumption_obj = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            partial_adjustment_speed=0.4,
            permanent_income_propensity=0.55,
            liquid_wealth_propensity=0.14,
            house_price_propensity=0.0,
        )
        assert consumption_obj.uses_continuous_wealth_calibration is False

        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=np.array([np.full(3, 40.0), np.full(3, 50.0)]),
            saving_rates=np.zeros(3),
            income=np.array([50.0, 100.0, 500.0]),
            household_benefits=np.zeros(3),
            consumption_weights=np.full(1, 1.0),
            consumption_weights_by_income=np.zeros((1, 3)),
            exogenous_total_consumption=np.zeros(1),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=0.0,
            liquid_wealth=np.array([10.0, 999.0, 5_000.0]),
            illiquid_wealth=np.array([0.0, 999.0, 2_000.0]),
            housing_wealth=np.array([0.0, 120.0, 1_000.0]),
            rent=np.zeros(3),
            mortgage_debt=np.zeros(3),
            mortgage_payment=np.zeros(3),
            owner_occupied=np.ones(3),
            mortgagor=np.ones(3),
            house_price_index=1.0,
            house_price_growth=0.0,
            lagged_consumption=np.full(3, 50.0),
            lagged_income=np.array([40.0, 80.0, 400.0]),
            lagged_liquid_wealth=np.array([10.0, 60.0, 4_000.0]),
            lagged_illiquid_wealth=np.array([0.0, 30.0, 1_500.0]),
            lagged_mortgage_debt=np.zeros(3),
            lagged_consumption_loan_debt=np.zeros(3),
            lagged_house_price_index=1.0,
        )
        components = consumption_obj.last_target_consumption_components
        # Global scalars apply uniformly across households when the flag is off.
        np.testing.assert_allclose(components["target_consumption_alpha_2"], 0.55)
        np.testing.assert_allclose(components["target_consumption_gamma_1"], 0.14)
        np.testing.assert_allclose(components["target_consumption_wealth_drag_clipped"], 0.0)
        assert np.all(np.isfinite(result))

    def test_continuous_wealth_calibration_alpha_2_rises_with_accessibility(self):
        # A household with high NLA/IFA/HA ratios (more accessible wealth) should
        # get a higher alpha_2 (more weight on permanent income, less on liquid
        # wealth sensitivity) than a poor hand-to-mouth household, per the fitted
        # NLA > IFA > HA accessibility ranking in cacf-household-group-calibration.md.
        consumption_obj = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            partial_adjustment_speed=0.4,
            illiquid_wealth_propensity=0.022,
            housing_wealth_propensity=0.013,
            house_price_propensity=0.0,
            uses_continuous_wealth_calibration=True,
        )

        # Three households spanning low to high wealth-to-income ratios.
        income = np.array([100.0, 100.0, 100.0])
        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=np.array([np.full(3, 40.0), np.full(3, 50.0)]),
            saving_rates=np.zeros(3),
            income=income,
            household_benefits=np.zeros(3),
            consumption_weights=np.full(1, 1.0),
            consumption_weights_by_income=np.zeros((1, 3)),
            exogenous_total_consumption=np.zeros(1),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=0.0,
            liquid_wealth=np.array([0.0, 50.0, 200.0]),
            illiquid_wealth=np.array([0.0, 30.0, 350.0]),
            housing_wealth=np.array([0.0, 100.0, 1_700.0]),
            rent=np.zeros(3),
            mortgage_debt=np.zeros(3),
            mortgage_payment=np.zeros(3),
            owner_occupied=np.ones(3),
            mortgagor=np.ones(3),
            house_price_index=1.0,
            house_price_growth=0.0,
            lagged_consumption=np.full(3, 50.0),
            lagged_income=income,
            lagged_liquid_wealth=np.array([0.0, 50.0, 200.0]),
            lagged_illiquid_wealth=np.array([0.0, 30.0, 350.0]),
            lagged_mortgage_debt=np.zeros(3),
            lagged_consumption_loan_debt=np.zeros(3),
            lagged_house_price_index=1.0,
        )
        components = consumption_obj.last_target_consumption_components
        alpha_2 = components["target_consumption_alpha_2"]
        gamma_1 = components["target_consumption_gamma_1"]
        # Monotone in accessibility (poorest -> richest household).
        assert alpha_2[0] < alpha_2[1] < alpha_2[2]
        assert gamma_1[0] > gamma_1[1] > gamma_1[2]
        # Bounded within the fitted ranges regardless of input.
        assert np.all(alpha_2 >= consumption_obj.continuous_wealth_calibration_alpha_2_range[0])
        assert np.all(alpha_2 <= consumption_obj.continuous_wealth_calibration_alpha_2_range[1])
        assert np.all(gamma_1 >= consumption_obj.continuous_wealth_calibration_gamma_1_range[0])
        assert np.all(gamma_1 <= consumption_obj.continuous_wealth_calibration_gamma_1_range[1])
        assert np.all(np.isfinite(result))

    def test_continuous_wealth_calibration_clip_backstops_near_zero_income(self):
        # A household with near-zero CURRENT income blows up NLA/IFA/HA ratios
        # (the issue #90 mechanism), since net_liquid_assets_ratio/illiquid_assets_ratio/
        # housing_wealth_ratio are denominated by annual_spendable_income (built from
        # current-period income), not lagged income -- a near-zero lagged_income alone
        # does not reach this denominator. The wealth-drag clip must keep MPC_LR-implying
        # log-consumption-to-income finite and flag that it fired.
        consumption_obj = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            partial_adjustment_speed=0.4,
            illiquid_wealth_propensity=0.022,
            housing_wealth_propensity=0.013,
            house_price_propensity=0.0,
            uses_continuous_wealth_calibration=True,
        )

        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=np.array([np.full(2, 40.0), np.full(2, 50.0)]),
            saving_rates=np.zeros(2),
            # Second household has near-zero current income -- the issue #90 blowup case.
            income=np.array([100.0, 1e-6]),
            household_benefits=np.zeros(2),
            consumption_weights=np.full(1, 1.0),
            consumption_weights_by_income=np.zeros((1, 2)),
            exogenous_total_consumption=np.zeros(1),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=0.0,
            liquid_wealth=np.array([100.0, 100.0]),
            illiquid_wealth=np.array([50.0, 50.0]),
            housing_wealth=np.array([200.0, 200.0]),
            rent=np.zeros(2),
            mortgage_debt=np.zeros(2),
            mortgage_payment=np.zeros(2),
            owner_occupied=np.ones(2),
            mortgagor=np.ones(2),
            house_price_index=1.0,
            house_price_growth=0.0,
            lagged_consumption=np.full(2, 50.0),
            lagged_income=np.array([100.0, 100.0]),
            lagged_liquid_wealth=np.array([100.0, 100.0]),
            lagged_illiquid_wealth=np.array([50.0, 50.0]),
            lagged_mortgage_debt=np.zeros(2),
            lagged_consumption_loan_debt=np.zeros(2),
            lagged_house_price_index=1.0,
        )
        components = consumption_obj.last_target_consumption_components
        assert components["target_consumption_wealth_drag_clipped"][1] == 1.0
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0.0)

    def test_clip_wealth_drag_bound_formula(self):
        # Unit test of _clip_wealth_drag's own arithmetic in isolation: MPC_LR =
        # (C/y)*[(1-alpha_2) - wealth_drag] in [0,1] requires wealth_drag in
        # [(1-alpha_2) - y/C, 1-alpha_2]. This calls the static method directly
        # with an already-correct y/C argument, so it verifies the formula but
        # NOT the call site in _evaluate_target that computes that argument --
        # see test_evaluate_target_clip_uses_income_to_consumption_ratio_at_call_site
        # below for the end-to-end regression guard on the call site itself
        # (a second review round found this test alone would not have caught
        # the y/C-vs-C/y inversion bug, since the bug was at the call site).
        alpha_2 = np.array([0.4])
        income_to_consumption_ratio = np.array([50.0 / 200.0])
        upper_expected = 1.0 - 0.4
        lower_expected = upper_expected - 0.25

        below_lower = np.array([lower_expected - 1.0])
        clipped, flag = CreditAugmentedConsumption._clip_wealth_drag(
            below_lower, alpha_2, income_to_consumption_ratio
        )
        np.testing.assert_allclose(clipped, lower_expected)
        assert flag[0] == 1.0

        above_upper = np.array([upper_expected + 1.0])
        clipped, flag = CreditAugmentedConsumption._clip_wealth_drag(
            above_upper, alpha_2, income_to_consumption_ratio
        )
        np.testing.assert_allclose(clipped, upper_expected)
        assert flag[0] == 1.0

        within_bounds = np.array([(lower_expected + upper_expected) / 2.0])
        clipped, flag = CreditAugmentedConsumption._clip_wealth_drag(
            within_bounds, alpha_2, income_to_consumption_ratio
        )
        np.testing.assert_allclose(clipped, within_bounds)
        assert flag[0] == 0.0

    def test_continuous_wealth_calibration_b_index_uses_fixed_constants_not_batch(self):
        # The B-index normalization (b_raw_min/max, b0) must be the calibration-fixed
        # constants from continuous_wealth_calibration_b_raw_bounds/_b0, not derived
        # from whichever households happen to be in this call's batch. A second review
        # round found the original version of this test (comparing two
        # compute_target_consumption calls with engineered companions) passed
        # coincidentally against the pre-fix batch-dependent code too, because the
        # test household's own raw B happened to land exactly at the batch median in
        # both fixtures. This version instead compares against a direct call to the
        # private mapping method for the household alone -- under the fixed
        # (batch-independent) implementation these two code paths are provably
        # identical for any inputs, since neither depends on which other households
        # are present; under the batch-dependent bug they generally differ, since a
        # batch of 1 always hits the b_max==b_min fallback (forcing B_tilde=0)
        # while a batch of 4 with very different companions does not.
        consumption_obj = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            partial_adjustment_speed=0.4,
            illiquid_wealth_propensity=0.022,
            housing_wealth_propensity=0.013,
            house_price_propensity=0.0,
            uses_continuous_wealth_calibration=True,
        )

        # Household A's own ratios, evaluated alone via the private method.
        nla_ratio_a = np.array([0.2])
        ifa_ratio_a = np.array([0.1])
        ha_ratio_a = np.array([0.5])
        reference_alpha_2, reference_gamma_1 = consumption_obj._compute_continuous_wealth_calibration(
            nla_ratio_a, ifa_ratio_a, ha_ratio_a
        )

        # Same household A as the first of 4, alongside companions with
        # deliberately very different (large) wealth-to-income ratios.
        income = np.full(4, 100.0)
        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=np.array([np.full(4, 40.0), np.full(4, 50.0)]),
            saving_rates=np.zeros(4),
            income=income,
            household_benefits=np.zeros(4),
            consumption_weights=np.full(1, 1.0),
            consumption_weights_by_income=np.zeros((1, 4)),
            exogenous_total_consumption=np.zeros(1),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=0.0,
            # Household A: liquid=20, illiquid=10, housing=50 -> ratios (0.2, 0.1, 0.5).
            # Companions: deliberately large and varied wealth-to-income ratios.
            liquid_wealth=np.array([20.0, 0.0, 150.0, 200.0]),
            illiquid_wealth=np.array([10.0, 0.0, 250.0, 300.0]),
            housing_wealth=np.array([50.0, 0.0, 1_000.0, 1_500.0]),
            rent=np.zeros(4),
            mortgage_debt=np.zeros(4),
            mortgage_payment=np.zeros(4),
            owner_occupied=np.ones(4),
            mortgagor=np.ones(4),
            house_price_index=1.0,
            house_price_growth=0.0,
            lagged_consumption=np.full(4, 50.0),
            lagged_income=income,
            lagged_liquid_wealth=np.array([20.0, 0.0, 150.0, 200.0]),
            lagged_illiquid_wealth=np.array([10.0, 0.0, 250.0, 300.0]),
            lagged_mortgage_debt=np.zeros(4),
            lagged_consumption_loan_debt=np.zeros(4),
            lagged_house_price_index=1.0,
        )
        components = consumption_obj.last_target_consumption_components
        np.testing.assert_allclose(components["target_consumption_alpha_2"][0], reference_alpha_2[0])
        np.testing.assert_allclose(components["target_consumption_gamma_1"][0], reference_gamma_1[0])
        assert np.all(np.isfinite(result))

    def test_continuous_wealth_calibration_single_household_batch_not_pinned_to_midpoint(self):
        # Under the batch-dependent bug, a batch of exactly one household always
        # hits the b_max==b_min fallback, forcing B_tilde=0=b0 and therefore
        # alpha_2/gamma_1 to the EXACT midpoint of their ranges -- regardless of
        # the household's actual wealth ratios. This is a clean, value-independent
        # discriminator: under the fix, only a household whose true (calibration-
        # fixed) B happens to equal the calibration-fixed b0 exactly would land on
        # the midpoint, which a poor household with low/negative ratios will not.
        consumption_obj = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            partial_adjustment_speed=0.4,
            illiquid_wealth_propensity=0.022,
            housing_wealth_propensity=0.013,
            house_price_propensity=0.0,
            uses_continuous_wealth_calibration=True,
        )
        # A poor household: negative NLA/y, zero IFA/y and HA/y -- far from the
        # population-median accessibility the fixed b0 is centred on.
        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=np.array([np.full(1, 40.0), np.full(1, 50.0)]),
            saving_rates=np.zeros(1),
            income=np.full(1, 100.0),
            household_benefits=np.zeros(1),
            consumption_weights=np.full(1, 1.0),
            consumption_weights_by_income=np.zeros((1, 1)),
            exogenous_total_consumption=np.zeros(1),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=0.0,
            liquid_wealth=np.array([0.0]),
            illiquid_wealth=np.array([0.0]),
            housing_wealth=np.array([0.0]),
            rent=np.zeros(1),
            mortgage_debt=np.zeros(1),
            mortgage_payment=np.zeros(1),
            owner_occupied=np.ones(1),
            mortgagor=np.ones(1),
            house_price_index=1.0,
            house_price_growth=0.0,
            lagged_consumption=np.full(1, 50.0),
            lagged_income=np.full(1, 100.0),
            lagged_liquid_wealth=np.array([0.0]),
            lagged_illiquid_wealth=np.array([0.0]),
            # NLA/y = (0 - 0 - 300) / 100 = -3.0, within the (-4.73, 2.15) bound.
            lagged_mortgage_debt=np.zeros(1),
            lagged_consumption_loan_debt=np.array([300.0]),
            lagged_house_price_index=1.0,
        )
        components = consumption_obj.last_target_consumption_components
        alpha_2 = components["target_consumption_alpha_2"][0]
        gamma_1 = components["target_consumption_gamma_1"][0]
        alpha_2_lo, alpha_2_hi = consumption_obj.continuous_wealth_calibration_alpha_2_range
        gamma_1_lo, gamma_1_hi = consumption_obj.continuous_wealth_calibration_gamma_1_range
        midpoint_alpha_2 = (alpha_2_lo + alpha_2_hi) / 2.0
        midpoint_gamma_1 = (gamma_1_lo + gamma_1_hi) / 2.0
        assert not np.isclose(alpha_2, midpoint_alpha_2, atol=1e-3)
        assert not np.isclose(gamma_1, midpoint_gamma_1, atol=1e-3)
        # A poor household (negative NLA/y, no other assets) should sit below the
        # midpoint alpha_2 (lower smoothing capacity) and above midpoint gamma_1
        # (higher liquid-wealth sensitivity).
        assert alpha_2 < midpoint_alpha_2
        assert gamma_1 > midpoint_gamma_1
        assert np.all(np.isfinite(result))

    def test_evaluate_target_clip_uses_income_to_consumption_ratio_at_call_site(self):
        # End-to-end regression guard for the y/C-vs-C/y inversion bug at the
        # _evaluate_target call site (test_clip_wealth_drag_bound_formula above
        # only exercises _clip_wealth_drag's own arithmetic, not this call site).
        # Household: lagged_income=10, lagged_consumption=100 -> y/C=0.1, C/y=10.
        # Large negative NLA/y (no IFA/HA) drives wealth_drag well below the
        # correct lower bound (1-alpha_2)-y/C but well ABOVE the buggy lower
        # bound (1-alpha_2)-C/y=(1-alpha_2)-10 -- so the clip fires (and binds at
        # the correct lower bound) only if the call site uses y/C, not C/y.
        consumption_obj = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
            partial_adjustment_speed=0.4,
            illiquid_wealth_propensity=0.0,
            housing_wealth_propensity=0.0,
            house_price_propensity=0.0,
            uses_continuous_wealth_calibration=True,
        )
        result = consumption_obj.compute_target_consumption(
            expected_inflation=0.0,
            current_cpi=1.0,
            initial_cpi=1.0,
            historic_consumption_sum=np.array([np.full(1, 40.0), np.full(1, 100.0)]),
            saving_rates=np.zeros(1),
            income=np.full(1, 10.0),
            household_benefits=np.zeros(1),
            consumption_weights=np.full(1, 1.0),
            consumption_weights_by_income=np.zeros((1, 1)),
            exogenous_total_consumption=np.zeros(1),
            current_time=0,
            take_consumption_weights_by_income_quantile=False,
            tau_vat=0.0,
            liquid_wealth=np.array([0.0]),
            illiquid_wealth=np.array([0.0]),
            housing_wealth=np.array([0.0]),
            rent=np.zeros(1),
            mortgage_debt=np.zeros(1),
            mortgage_payment=np.zeros(1),
            owner_occupied=np.ones(1),
            mortgagor=np.ones(1),
            house_price_index=1.0,
            house_price_growth=0.0,
            lagged_consumption=np.full(1, 100.0),
            lagged_income=np.full(1, 10.0),
            lagged_liquid_wealth=np.array([0.0]),
            lagged_illiquid_wealth=np.array([0.0]),
            lagged_mortgage_debt=np.zeros(1),
            # NLA/y = (0 - 0 - 45) / 10 = -4.5, within the (-4.73, 2.15) bound.
            lagged_consumption_loan_debt=np.array([45.0]),
            lagged_house_price_index=1.0,
        )
        components = consumption_obj.last_target_consumption_components
        alpha_2 = components["target_consumption_alpha_2"][0]
        income_to_consumption_ratio = 10.0 / 100.0
        correct_lower_bound = (1.0 - alpha_2) - income_to_consumption_ratio
        buggy_lower_bound = (1.0 - alpha_2) - (100.0 / 10.0)

        # The clip must have fired (raw wealth_drag is far below the correct
        # lower bound) ...
        assert components["target_consumption_wealth_drag_clipped"][0] == 1.0
        # ... and target_consumption_log_long_run (= long_run_intercept [0] +
        # real_borrowing_rate_term [0, unset] + permanent_income_term [0,
        # permanent_income_log_ratio unset] + clipped wealth_drag + house_price_term
        # [0]) must equal the CORRECT lower bound, not the buggy one -- the two
        # differ by ~9.4 (1-alpha_2-0.1 vs 1-alpha_2-10), so this is not a
        # tolerance-sensitive comparison.
        np.testing.assert_allclose(components["target_consumption_log_long_run"][0], correct_lower_bound, atol=1e-9)
        assert abs(correct_lower_bound - buggy_lower_bound) > 5.0  # sanity: bounds are far apart
        assert not np.isclose(
            components["target_consumption_log_long_run"][0], buggy_lower_bound, atol=1e-3
        )
        assert np.all(np.isfinite(result))

    def test_evaluate_target_clip_income_to_consumption_ratio_invariant_under_time_unit(self):
        # income_to_consumption_ratio = annual_spendable_income / annual_lagged_consumption
        # annualizes both sides of a same-frequency flow ratio, so it must come out
        # identical to real_spendable_income / real_lagged_consumption regardless of
        # time_unit -- a second review round found a prior fix annualized only the
        # numerator, which would make this ratio scale with annualization_factor
        # (wrong: it should cancel). Same setup as the call-site test above
        # (income=lagged_income=10, lagged_consumption=100, NLA/y=-4.5), run twice
        # with time_unit=12 (annualization_factor=1) and time_unit=3 (factor=4); the
        # clip's resulting lower bound must be identical across both.
        def run_with_time_unit(time_unit):
            consumption_obj = CreditAugmentedConsumption(
                consumption_smoothing_fraction=0.0,
                consumption_smoothing_window=1,
                minimum_consumption_fraction=0.0,
                partial_adjustment_speed=0.4,
                illiquid_wealth_propensity=0.0,
                housing_wealth_propensity=0.0,
                house_price_propensity=0.0,
                uses_continuous_wealth_calibration=True,
            )
            consumption_obj.compute_target_consumption(
                expected_inflation=0.0,
                current_cpi=1.0,
                initial_cpi=1.0,
                historic_consumption_sum=np.array([np.full(1, 40.0), np.full(1, 100.0)]),
                saving_rates=np.zeros(1),
                income=np.full(1, 10.0),
                household_benefits=np.zeros(1),
                consumption_weights=np.full(1, 1.0),
                consumption_weights_by_income=np.zeros((1, 1)),
                exogenous_total_consumption=np.zeros(1),
                current_time=0,
                take_consumption_weights_by_income_quantile=False,
                tau_vat=0.0,
                liquid_wealth=np.array([0.0]),
                illiquid_wealth=np.array([0.0]),
                housing_wealth=np.array([0.0]),
                rent=np.zeros(1),
                mortgage_debt=np.zeros(1),
                mortgage_payment=np.zeros(1),
                owner_occupied=np.ones(1),
                mortgagor=np.ones(1),
                house_price_index=1.0,
                house_price_growth=0.0,
                lagged_consumption=np.full(1, 100.0),
                lagged_income=np.full(1, 10.0),
                lagged_liquid_wealth=np.array([0.0]),
                lagged_illiquid_wealth=np.array([0.0]),
                lagged_mortgage_debt=np.zeros(1),
                lagged_consumption_loan_debt=np.array([45.0]),
                lagged_house_price_index=1.0,
                time_unit=time_unit,
            )
            return consumption_obj.last_target_consumption_components

        components_annual = run_with_time_unit(12)
        components_quarterly = run_with_time_unit(3)
        alpha_2_annual = components_annual["target_consumption_alpha_2"][0]
        alpha_2_quarterly = components_quarterly["target_consumption_alpha_2"][0]
        correct_lower_bound_annual = (1.0 - alpha_2_annual) - (10.0 / 100.0)
        correct_lower_bound_quarterly = (1.0 - alpha_2_quarterly) - (10.0 / 100.0)
        np.testing.assert_allclose(
            components_annual["target_consumption_log_long_run"][0], correct_lower_bound_annual, atol=1e-9
        )
        np.testing.assert_allclose(
            components_quarterly["target_consumption_log_long_run"][0], correct_lower_bound_quarterly, atol=1e-9
        )

    def test_time_unit_zero_raises(self):
        consumption_obj = CreditAugmentedConsumption(
            consumption_smoothing_fraction=0.0,
            consumption_smoothing_window=1,
            minimum_consumption_fraction=0.0,
        )
        with pytest.raises(ValueError, match="time_unit"):
            consumption_obj.compute_target_consumption(
                expected_inflation=0.0,
                current_cpi=1.0,
                initial_cpi=1.0,
                historic_consumption_sum=np.array([np.full(1, 40.0), np.full(1, 100.0)]),
                saving_rates=np.zeros(1),
                income=np.full(1, 10.0),
                household_benefits=np.zeros(1),
                consumption_weights=np.full(1, 1.0),
                consumption_weights_by_income=np.zeros((1, 1)),
                exogenous_total_consumption=np.zeros(1),
                current_time=0,
                take_consumption_weights_by_income_quantile=False,
                tau_vat=0.0,
                liquid_wealth=np.array([0.0]),
                illiquid_wealth=np.array([0.0]),
                housing_wealth=np.array([0.0]),
                rent=np.zeros(1),
                mortgage_debt=np.zeros(1),
                mortgage_payment=np.zeros(1),
                owner_occupied=np.ones(1),
                mortgagor=np.ones(1),
                house_price_index=1.0,
                house_price_growth=0.0,
                lagged_consumption=np.full(1, 100.0),
                lagged_income=np.full(1, 10.0),
                lagged_liquid_wealth=np.array([0.0]),
                lagged_illiquid_wealth=np.array([0.0]),
                lagged_mortgage_debt=np.zeros(1),
                lagged_consumption_loan_debt=np.zeros(1),
                lagged_house_price_index=1.0,
                time_unit=0,
            )

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"continuous_wealth_calibration": {"net_liquid_assets_ratio_bounds": (1.0, 1.0)}}, "Wealth-ratio"),
            ({"continuous_wealth_calibration": {"b_raw_min": 1.0, "b_raw_max": 1.0}}, "b_raw"),
            ({"continuous_wealth_calibration": {"alpha_2_low": 0.5, "alpha_2_high": 0.5}}, "alpha_2"),
            ({"continuous_wealth_calibration": {"gamma_1_low": 0.1, "gamma_1_high": 0.1}}, "gamma_1"),
        ],
    )
    def test_degenerate_continuous_calibration_bounds_raise_at_construction(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CreditAugmentedConsumption(
                consumption_smoothing_fraction=0.0,
                consumption_smoothing_window=1,
                minimum_consumption_fraction=0.0,
                **kwargs,
            )
