"""Household consumption behavior implementation.

This module implements household consumption decisions through:
- Target consumption calculation
- Income-based consumption allocation
- Consumption smoothing mechanisms
- Minimum consumption thresholds
- Tax-adjusted spending

The implementation handles:
- Consumption smoothing over time
- Income and saving rate effects
- Price level adjustments
- Industry-specific allocations
- Tax considerations
"""

from abc import ABC, abstractmethod

import numpy as np
from numba import njit


class HouseholdConsumption(ABC):
    """Abstract base class for household consumption behavior.

    Defines interface for computing target consumption levels based on:
    - Income and saving rates
    - Historical consumption patterns
    - Price level changes
    - Industry allocations
    - Tax considerations

    Attributes:
        consumption_smoothing_fraction (float): Weight on historical consumption
        consumption_smoothing_window (int): Periods for smoothing calculation
        minimum_consumption_fraction (float): Floor on consumption/income ratio
    """

    def __init__(
        self,
        consumption_smoothing_fraction: float,
        consumption_smoothing_window: int,
        minimum_consumption_fraction: float,
        elasticity_of_substitution: float = 1.0,  # Ignored by default consumption
    ):
        self.consumption_smoothing_fraction = consumption_smoothing_fraction
        self.consumption_smoothing_window = consumption_smoothing_window
        self.minimum_consumption_fraction = minimum_consumption_fraction
        # Note: elasticity_of_substitution is ignored in default consumption

    @abstractmethod
    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        prices: np.ndarray = None,
        initial_prices: np.ndarray = None,
        taxes: np.ndarray = None,
        initial_taxes: np.ndarray = None,
        bundle_matrix: np.ndarray = None,
        liquid_wealth: np.ndarray = None,
        illiquid_wealth: np.ndarray = None,
        housing_wealth: np.ndarray = None,
        rent: np.ndarray = None,
        mortgage_debt: np.ndarray = None,
        mortgage_payment: np.ndarray = None,
        house_price_index: float | np.ndarray = None,
        house_price_growth: float | np.ndarray = None,
        lagged_consumption: np.ndarray = None,
    ) -> np.ndarray:
        """Calculate target consumption levels.

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            historic_consumption_sum (np.ndarray): Past consumption totals
            saving_rates (np.ndarray): Household saving rates
            income (np.ndarray): Household income
            household_benefits (np.ndarray): Social benefits received
            consumption_weights (np.ndarray): Industry consumption shares
            consumption_weights_by_income (np.ndarray): Income-based weights
            exogenous_total_consumption (np.ndarray): External consumption target
            current_time (int): Current period
            take_consumption_weights_by_income_quantile (bool): Use income quintiles
            tau_vat (float): Value added tax rate
            prices (np.ndarray | None): Current prices by industry for optional CES substitution
            initial_prices (np.ndarray | None): Initial prices by industry for optional CES substitution
            taxes (np.ndarray | None): Current tax rates by industry for optional CES substitution
            initial_taxes (np.ndarray | None): Initial tax rates by industry for optional CES substitution
            bundle_matrix (np.ndarray | None): Bundle weight matrix for optional CES substitution
            liquid_wealth (np.ndarray | None): Liquid wealth per household
            illiquid_wealth (np.ndarray | None): Illiquid financial wealth per household
            housing_wealth (np.ndarray | None): Housing wealth per household
            rent (np.ndarray | None): Rent per household
            mortgage_debt (np.ndarray | None): Mortgage debt per household
            mortgage_payment (np.ndarray | None): Mortgage payment per household
            house_price_index (float | np.ndarray | None): House-price index level
            house_price_growth (float | np.ndarray | None): House-price growth proxy
            lagged_consumption (np.ndarray | None): Previous-period consumption

        Returns:
            np.ndarray: Target consumption by household and industry
        """
        pass


class DefaultHouseholdConsumption(HouseholdConsumption):
    """Default implementation of household consumption behavior.

    Implements consumption decisions based on:
    - Income and saving rates
    - Historical consumption smoothing
    - Minimum consumption thresholds
    - Industry-specific allocations
    - Tax adjustments
    """

    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        prices: np.ndarray = None,  # Ignored in default consumption
        initial_prices: np.ndarray = None,  # Ignored in default consumption
        taxes: np.ndarray = None,  # Ignored in default consumption
        initial_taxes: np.ndarray = None,  # Ignored in default consumption
        bundle_matrix: np.ndarray = None,  # Ignored in default consumption
        liquid_wealth: np.ndarray = None,  # Ignored in default consumption
        illiquid_wealth: np.ndarray = None,  # Ignored in default consumption
        housing_wealth: np.ndarray = None,  # Ignored in default consumption
        rent: np.ndarray = None,  # Ignored in default consumption
        mortgage_debt: np.ndarray = None,  # Ignored in default consumption
        mortgage_payment: np.ndarray = None,  # Ignored in default consumption
        house_price_index: float | np.ndarray = None,  # Ignored in default consumption
        house_price_growth: float | np.ndarray = None,  # Ignored in default consumption
        lagged_consumption: np.ndarray = None,  # Ignored in default consumption
    ) -> np.ndarray:
        """Calculate target consumption using default behavior.

        Determines consumption targets based on:
        - Income after savings
        - Historical consumption patterns
        - Minimum consumption thresholds
        - Industry allocation weights
        - Tax considerations

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            historic_consumption_sum (np.ndarray): Past consumption totals
            saving_rates (np.ndarray): Household saving rates
            income (np.ndarray): Household income
            household_benefits (np.ndarray): Social benefits received
            consumption_weights (np.ndarray): Industry consumption shares
            consumption_weights_by_income (np.ndarray): Income-based weights
            exogenous_total_consumption (np.ndarray): External consumption target
            current_time (int): Current period
            take_consumption_weights_by_income_quantile (bool): Use income quintiles
            tau_vat (float): Value added tax rate

        Returns:
            np.ndarray: Target consumption by household and industry
        """
        return self._compute_target_consumption(
            historic_consumption_sum=historic_consumption_sum,
            saving_rates=saving_rates,
            income=income,
            household_benefits=household_benefits,
            consumption_weights=consumption_weights,
            consumption_weights_by_income=consumption_weights_by_income,
            take_consumption_weights_by_income_quantile=take_consumption_weights_by_income_quantile,
            tau_vat=tau_vat,
            consumption_smoothing_window=self.consumption_smoothing_window,
            consumption_smoothing_fraction=self.consumption_smoothing_fraction,
            minimum_consumption_fraction=self.minimum_consumption_fraction,
        )

    @staticmethod
    # @njit(
    #     float64[:, :](
    #         float64[:, :],  # historic_consumption_sum
    #         float64[:],  # saving_rates
    #         float64[:],  # income
    #         float64[:],  # household_benefits
    #         float64[:],  # consumption_weights
    #         float64[:, :],  # consumption_weights_by_income
    #         boolean,  # take_consumption_weights_by_income_quantile
    #         float64,  # tau_vat
    #         int64,  # consumption_smoothing_window
    #         float64,  # consumption_smoothing_fraction
    #         float64,  # minimum_consumption_fraction
    #     ),
    #     cache=True,
    # )
    @njit(cache=True)
    def _compute_target_consumption(
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,  # noqa
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        consumption_smoothing_window: int,
        consumption_smoothing_fraction: float,
        minimum_consumption_fraction: float,
    ) -> np.ndarray:
        """Internal method for consumption calculation.

        Implements the core consumption calculation logic with:
        - Historical smoothing
        - Income-based allocation
        - Minimum thresholds
        - Tax adjustments

        Args:
            historic_consumption_sum (np.ndarray): Past consumption totals
            saving_rates (np.ndarray): Household saving rates
            income (np.ndarray): Household income
            household_benefits (np.ndarray): Social benefits received
            consumption_weights (np.ndarray): Industry consumption shares
            consumption_weights_by_income (np.ndarray): Income-based weights
            take_consumption_weights_by_income_quantile (bool): Use income quintiles
            tau_vat (float): Value added tax rate
            consumption_smoothing_window (int): Periods for smoothing
            consumption_smoothing_fraction (float): Smoothing weight
            minimum_consumption_fraction (float): Consumption floor

        Returns:
            np.ndarray: Target consumption by household and industry
        """
        smoothing_window = min(consumption_smoothing_window, len(historic_consumption_sum))
        target_consumption = (
            1.0
            / (1 + tau_vat)
            * np.outer(
                consumption_weights,
                np.maximum(
                    minimum_consumption_fraction * (1 - saving_rates) * household_benefits,
                    (1 - saving_rates) * income,
                    consumption_smoothing_fraction
                    * (1 + tau_vat)
                    * (1 / smoothing_window)
                    * historic_consumption_sum[1:][-smoothing_window:].sum(axis=0),
                ),
            ).T
        )
        return np.maximum(0.0, target_consumption)


class CESHouseholdConsumption(HouseholdConsumption):
    """CES (Constant Elasticity of Substitution) household consumption implementation.

    Implements consumption decisions with substitution within bundles based on:
    - CES utility function with elasticity of substitution
    - Dynamic consumption shares based on relative prices and taxes
    - Bundle-based substitution patterns
    - Initial consumption weights as preference parameters
    """

    def __init__(
        self,
        consumption_smoothing_fraction: float,
        consumption_smoothing_window: int,
        minimum_consumption_fraction: float,
        elasticity_of_substitution: float = 1.0,
    ):
        super().__init__(
            consumption_smoothing_fraction,
            consumption_smoothing_window,
            minimum_consumption_fraction,
        )
        self.elasticity_of_substitution = elasticity_of_substitution

    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        prices: np.ndarray = None,
        initial_prices: np.ndarray = None,
        taxes: np.ndarray = None,
        initial_taxes: np.ndarray = None,
        bundle_matrix: np.ndarray = None,
        liquid_wealth: np.ndarray = None,  # Ignored in CES consumption
        illiquid_wealth: np.ndarray = None,  # Ignored in CES consumption
        housing_wealth: np.ndarray = None,  # Ignored in CES consumption
        rent: np.ndarray = None,  # Ignored in CES consumption
        mortgage_debt: np.ndarray = None,  # Ignored in CES consumption
        mortgage_payment: np.ndarray = None,  # Ignored in CES consumption
        house_price_index: float | np.ndarray = None,  # Ignored in CES consumption
        house_price_growth: float | np.ndarray = None,  # Ignored in CES consumption
        lagged_consumption: np.ndarray = None,  # Ignored in CES consumption
    ) -> np.ndarray:
        """Calculate target consumption using CES substitution within bundles.

        Determines consumption based on:
        - CES utility function with substitution elasticity
        - Dynamic consumption shares based on relative prices and taxes
        - Bundle-based substitution patterns
        - Initial consumption preferences

        Args:
            All standard args plus:
            prices (np.ndarray): Current prices by industry
            initial_prices (np.ndarray): Initial prices by industry
            taxes (np.ndarray): Current tax rates by industry
            initial_taxes (np.ndarray): Initial tax rates by industry
            bundle_matrix (np.ndarray): Bundle weight matrix (n_industries, n_bundles)

        Returns:
            np.ndarray: Target consumption by household and industry
        """
        # If substitution data is unavailable or not aligned to household
        # consumption goods, fall back to default behavior.
        if any(x is None for x in [prices, initial_prices, taxes, initial_taxes, bundle_matrix]) or (
            not self._has_aligned_ces_inputs(
                consumption_weights,
                prices,
                initial_prices,
                taxes,
                initial_taxes,
                bundle_matrix,
            )
        ):
            return self._compute_target_consumption_default(
                historic_consumption_sum,
                saving_rates,
                income,
                household_benefits,
                consumption_weights,
                consumption_weights_by_income,
                take_consumption_weights_by_income_quantile,
                tau_vat,
            )

        # Compute CES consumption shares with substitution
        ces_weights = self._compute_ces_weights(
            consumption_weights, prices, initial_prices, taxes, initial_taxes, bundle_matrix
        )

        return self._compute_target_consumption_ces(
            historic_consumption_sum,
            saving_rates,
            income,
            household_benefits,
            ces_weights,
            consumption_weights_by_income,
            take_consumption_weights_by_income_quantile,
            tau_vat,
        )

    def _compute_target_consumption_default(
        self,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
    ) -> np.ndarray:
        """Default consumption calculation when substitution data is unavailable."""
        return DefaultHouseholdConsumption._compute_target_consumption(
            historic_consumption_sum,
            saving_rates,
            income,
            household_benefits,
            consumption_weights,
            consumption_weights_by_income,
            take_consumption_weights_by_income_quantile,
            tau_vat,
            self.consumption_smoothing_window,
            self.consumption_smoothing_fraction,
            self.minimum_consumption_fraction,
        )

    @staticmethod
    def _has_aligned_ces_inputs(
        consumption_weights: np.ndarray,
        prices: np.ndarray,
        initial_prices: np.ndarray,
        taxes: np.ndarray,
        initial_taxes: np.ndarray,
        bundle_matrix: np.ndarray,
    ) -> bool:
        """Return whether CES substitution inputs share one goods dimension."""
        n_goods = len(consumption_weights)
        return (
            np.ndim(bundle_matrix) == 2
            and bundle_matrix.shape[0] == n_goods
            and len(prices) == n_goods
            and len(initial_prices) == n_goods
            and len(taxes) == n_goods
            and len(initial_taxes) == n_goods
        )

    def _compute_ces_weights(
        self,
        initial_weights: np.ndarray,
        prices: np.ndarray,
        initial_prices: np.ndarray,
        taxes: np.ndarray,
        initial_taxes: np.ndarray,
        bundle_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute CES consumption weights with substitution within bundles.

        Implements the formula:
        c_i(t) = c_i(0) * ((1+τ_i(0))/(1+τ_i(t)))^σ * p_i(t)^(-σ) * bundle_normalization
        """
        sigma = self.elasticity_of_substitution

        # Compute price and tax ratios
        price_ratio = prices / initial_prices
        tax_ratio = (1 + initial_taxes) / (1 + taxes)

        # Compute individual substitution effects
        substitution_factor = (tax_ratio**sigma) * (price_ratio ** (-sigma))

        # Apply substitution within bundles
        ces_weights = np.zeros_like(initial_weights)
        n_bundles = bundle_matrix.shape[1]

        for bundle_idx in range(n_bundles):
            # Industries in this bundle (bundle_matrix is n_industries x n_bundles)
            bundle_mask = bundle_matrix[:, bundle_idx] > 0

            if not np.any(bundle_mask):
                continue

            # Initial bundle allocation
            bundle_initial_weights = initial_weights[bundle_mask]
            bundle_total = np.sum(bundle_initial_weights)

            if bundle_total == 0:
                continue

            # Apply CES substitution within bundle
            bundle_substitution = substitution_factor[bundle_mask] * bundle_initial_weights
            bundle_substitution_total = np.sum(bundle_substitution)

            # Normalize to maintain bundle total
            if bundle_substitution_total > 0:
                ces_weights[bundle_mask] = bundle_substitution * (bundle_total / bundle_substitution_total)
            else:
                ces_weights[bundle_mask] = bundle_initial_weights

        return ces_weights

    def _compute_target_consumption_ces(
        self,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        ces_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
    ) -> np.ndarray:
        """Compute target consumption using CES-adjusted weights."""
        smoothing_window = min(self.consumption_smoothing_window, len(historic_consumption_sum))
        target_consumption = (
            1.0
            / (1 + tau_vat)
            * np.outer(
                ces_weights,
                np.maximum(
                    self.minimum_consumption_fraction * (1 - saving_rates) * household_benefits,
                    (1 - saving_rates) * income,
                    self.consumption_smoothing_fraction
                    * (1 + tau_vat)
                    * (1 / smoothing_window)
                    * historic_consumption_sum[1:][-smoothing_window:].sum(axis=0),
                ),
            ).T
        )
        return np.maximum(0.0, target_consumption)


class CreditAugmentedConsumption(HouseholdConsumption):
    """Credit-augmented household consumption implementation.

    This rule keeps the existing household consumption hook active while exposing
    a paper-oriented target-consumption decomposition. The provisional long-run
    target uses real spendable income and real balance-sheet ratios over income.
    Stage 3 learning and interest-rate inputs that are not available yet are
    retained as explicit zero placeholders in the decomposition.
    """

    def __init__(
        self,
        consumption_smoothing_fraction: float,
        consumption_smoothing_window: int,
        minimum_consumption_fraction: float,
        elasticity_of_substitution: float = 1.0,
        permanent_income_propensity: float = 1.0,
        liquid_wealth_propensity: float = 0.04,
        illiquid_wealth_propensity: float = 0.02,
        housing_wealth_propensity: float = 0.02,
        rent_propensity: float = 1.0,
        mortgage_debt_propensity: float = 0.03,
        mortgage_payment_propensity: float = 1.0,
        house_price_propensity: float = 0.02,
        interest_rate_cashflow_propensity: float | None = None,
        uncertainty_propensity: float | None = None,
        partial_adjustment_speed: float = 0.5,
    ):
        super().__init__(
            consumption_smoothing_fraction,
            consumption_smoothing_window,
            minimum_consumption_fraction,
            elasticity_of_substitution,
        )
        self.permanent_income_propensity = permanent_income_propensity
        self.liquid_wealth_propensity = liquid_wealth_propensity
        self.illiquid_wealth_propensity = illiquid_wealth_propensity
        self.housing_wealth_propensity = housing_wealth_propensity
        self.rent_propensity = rent_propensity
        self.mortgage_debt_propensity = mortgage_debt_propensity
        self.mortgage_payment_propensity = mortgage_payment_propensity
        self.house_price_propensity = house_price_propensity
        self.interest_rate_cashflow_propensity = interest_rate_cashflow_propensity
        self.uncertainty_propensity = uncertainty_propensity
        self.partial_adjustment_speed = partial_adjustment_speed
        self.last_target_consumption_components: dict[str, np.ndarray] | None = None
        self.last_formula_implied_mpc: np.ndarray | None = None

    @staticmethod
    def _as_array(reference: np.ndarray, value: np.ndarray | float | None, default: float = 0.0) -> np.ndarray:
        if value is None:
            return np.full_like(reference, default, dtype=float)
        value_array = np.asarray(value, dtype=float)
        if value_array.shape == ():
            return np.full_like(reference, float(value_array), dtype=float)
        if value_array.shape != reference.shape:
            return np.broadcast_to(value_array, reference.shape).astype(float)
        return value_array

    def _evaluate_target(
        self,
        income: np.ndarray,
        household_benefits: np.ndarray,
        lagged_consumption: np.ndarray,
        liquid_wealth: np.ndarray,
        illiquid_wealth: np.ndarray,
        housing_wealth: np.ndarray,
        rent: np.ndarray,
        mortgage_debt: np.ndarray,
        mortgage_payment: np.ndarray,
        house_price_index: float | np.ndarray | None,
        house_price_growth: float | np.ndarray | None,
        current_cpi: float,
        initial_cpi: float,
        expected_inflation: float,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        income = np.asarray(income, dtype=float)
        household_benefits = np.asarray(household_benefits, dtype=float)
        lagged_consumption = np.asarray(lagged_consumption, dtype=float)
        liquid_wealth = np.asarray(liquid_wealth, dtype=float)
        illiquid_wealth = np.asarray(illiquid_wealth, dtype=float)
        housing_wealth = np.asarray(housing_wealth, dtype=float)
        rent = np.asarray(rent, dtype=float)
        mortgage_debt = np.asarray(mortgage_debt, dtype=float)
        mortgage_payment = np.asarray(mortgage_payment, dtype=float)

        house_price_index_arr = self._as_array(income, house_price_index, default=1.0)
        house_price_growth_arr = self._as_array(income, house_price_growth, default=0.0)

        cpi_ratio = current_cpi / initial_cpi if initial_cpi != 0.0 else 1.0
        deflator = max(float(cpi_ratio), np.finfo(float).eps)
        nominalizer = deflator * (1.0 + expected_inflation)

        real_spendable_income = np.maximum((income + household_benefits) / deflator, 0.0)
        real_income_denominator = np.maximum(real_spendable_income, 1.0)
        real_lagged_consumption = np.maximum(lagged_consumption / deflator, 0.0)
        real_liquid_wealth = liquid_wealth / deflator
        real_illiquid_wealth = illiquid_wealth / deflator
        real_housing_wealth = housing_wealth / deflator
        real_rent = rent / deflator
        real_mortgage_debt = mortgage_debt / deflator
        real_mortgage_payment = mortgage_payment / deflator

        liquid_wealth_ratio = np.maximum(real_liquid_wealth, 0.0) / real_income_denominator
        illiquid_wealth_ratio = np.maximum(real_illiquid_wealth, 0.0) / real_income_denominator
        housing_wealth_ratio = np.maximum(real_housing_wealth, 0.0) / real_income_denominator
        mortgage_debt_ratio = np.maximum(real_mortgage_debt, 0.0) / real_income_denominator

        permanent_income_term_real = self.permanent_income_propensity * real_spendable_income
        liquid_wealth_term_real = self.liquid_wealth_propensity * liquid_wealth_ratio * real_spendable_income
        illiquid_wealth_term_real = self.illiquid_wealth_propensity * illiquid_wealth_ratio * real_spendable_income
        housing_wealth_term_real = self.housing_wealth_propensity * housing_wealth_ratio * real_spendable_income
        rent_term_real = -self.rent_propensity * np.maximum(real_rent, 0.0)
        mortgage_debt_term_real = -self.mortgage_debt_propensity * mortgage_debt_ratio * real_spendable_income
        mortgage_payment_term_real = -self.mortgage_payment_propensity * np.maximum(real_mortgage_payment, 0.0)
        house_price_term_real = (
            self.house_price_propensity
            * housing_wealth_ratio
            * house_price_growth_arr
            * real_spendable_income
        )
        interest_rate_cashflow_term_real = np.zeros_like(real_spendable_income)
        if self.interest_rate_cashflow_propensity is not None:
            interest_rate_cashflow_term_real *= self.interest_rate_cashflow_propensity
        uncertainty_term_real = np.zeros_like(real_spendable_income)
        if self.uncertainty_propensity is not None:
            uncertainty_term_real *= self.uncertainty_propensity

        long_run_target_real = (
            permanent_income_term_real
            + liquid_wealth_term_real
            + illiquid_wealth_term_real
            + housing_wealth_term_real
            + rent_term_real
            + mortgage_debt_term_real
            + mortgage_payment_term_real
            + house_price_term_real
            + interest_rate_cashflow_term_real
            + uncertainty_term_real
        )
        partial_adjustment_gap_real = self.partial_adjustment_speed * (
            long_run_target_real - real_lagged_consumption
        )
        target_total_real = np.maximum(0.0, real_lagged_consumption + partial_adjustment_gap_real)
        target_total = target_total_real * nominalizer

        components = {
            "target_consumption_lagged_consumption": lagged_consumption,
            "target_consumption_long_run": long_run_target_real * nominalizer,
            "target_consumption_permanent_income": permanent_income_term_real * nominalizer,
            "target_consumption_liquid_wealth": liquid_wealth_term_real * nominalizer,
            "target_consumption_illiquid_wealth": illiquid_wealth_term_real * nominalizer,
            "target_consumption_housing_wealth": housing_wealth_term_real * nominalizer,
            "target_consumption_rent": rent_term_real * nominalizer,
            "target_consumption_mortgage_debt": mortgage_debt_term_real * nominalizer,
            "target_consumption_mortgage_payment": mortgage_payment_term_real * nominalizer,
            "target_consumption_house_price": house_price_term_real * nominalizer,
            "target_consumption_interest_rate_cashflow": interest_rate_cashflow_term_real * nominalizer,
            "target_consumption_uncertainty": uncertainty_term_real * nominalizer,
            "target_consumption_partial_adjustment_gap": partial_adjustment_gap_real * nominalizer,
            "target_consumption_house_price_index": house_price_index_arr,
        }
        return components, target_total

    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        prices: np.ndarray = None,
        initial_prices: np.ndarray = None,
        taxes: np.ndarray = None,
        initial_taxes: np.ndarray = None,
        bundle_matrix: np.ndarray = None,
        liquid_wealth: np.ndarray = None,
        illiquid_wealth: np.ndarray = None,
        housing_wealth: np.ndarray = None,
        rent: np.ndarray = None,
        mortgage_debt: np.ndarray = None,
        mortgage_payment: np.ndarray = None,
        house_price_index: float | np.ndarray = None,
        house_price_growth: float | np.ndarray = None,
        lagged_consumption: np.ndarray = None,
    ) -> np.ndarray:
        if lagged_consumption is None:
            lagged_consumption = np.asarray(historic_consumption_sum, dtype=float)[-1]
        else:
            lagged_consumption = np.asarray(lagged_consumption, dtype=float)

        income = np.asarray(income, dtype=float)
        household_benefits = self._as_array(income, household_benefits)
        liquid_wealth = self._as_array(income, liquid_wealth)
        illiquid_wealth = self._as_array(income, illiquid_wealth)
        housing_wealth = self._as_array(income, housing_wealth)
        rent = self._as_array(income, rent)
        mortgage_debt = self._as_array(income, mortgage_debt)
        mortgage_payment = self._as_array(income, mortgage_payment)

        components, target_total = self._evaluate_target(
            income=income,
            household_benefits=household_benefits,
            lagged_consumption=lagged_consumption,
            liquid_wealth=liquid_wealth,
            illiquid_wealth=illiquid_wealth,
            housing_wealth=housing_wealth,
            rent=rent,
            mortgage_debt=mortgage_debt,
            mortgage_payment=mortgage_payment,
            house_price_index=house_price_index,
            house_price_growth=house_price_growth,
            current_cpi=current_cpi,
            initial_cpi=initial_cpi,
            expected_inflation=expected_inflation,
        )

        perturbation = np.maximum(1.0, np.abs(income) * 1e-4)
        _, perturbed_target = self._evaluate_target(
            income=income + perturbation,
            household_benefits=household_benefits,
            lagged_consumption=lagged_consumption,
            liquid_wealth=liquid_wealth,
            illiquid_wealth=illiquid_wealth,
            housing_wealth=housing_wealth,
            rent=rent,
            mortgage_debt=mortgage_debt,
            mortgage_payment=mortgage_payment,
            house_price_index=house_price_index,
            house_price_growth=house_price_growth,
            current_cpi=current_cpi,
            initial_cpi=initial_cpi,
            expected_inflation=expected_inflation,
        )

        self.last_target_consumption_components = components
        self.last_formula_implied_mpc = (perturbed_target - target_total) / perturbation

        target_consumption = np.maximum(
            0.0,
            1.0
            / (1 + tau_vat)
            * np.outer(
                consumption_weights,
                target_total,
            ).T,
        )
        return target_consumption


class ExogenousHouseholdConsumption(HouseholdConsumption):
    """Exogenous household consumption implementation.

    Implements consumption decisions based on:
    - External consumption targets
    - Price level adjustments
    - Income-based allocation
    - Tax considerations
    """

    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        historic_consumption_sum: np.ndarray,
        saving_rates: np.ndarray,
        income: np.ndarray,
        household_benefits: np.ndarray,
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        exogenous_total_consumption: np.ndarray,
        current_time: int,
        take_consumption_weights_by_income_quantile: bool,
        tau_vat: float,
        prices: np.ndarray = None,  # Ignored in exogenous consumption
        initial_prices: np.ndarray = None,  # Ignored in exogenous consumption
        taxes: np.ndarray = None,  # Ignored in exogenous consumption
        initial_taxes: np.ndarray = None,  # Ignored in exogenous consumption
        bundle_matrix: np.ndarray = None,  # Ignored in exogenous consumption
        liquid_wealth: np.ndarray = None,  # Ignored in exogenous consumption
        illiquid_wealth: np.ndarray = None,  # Ignored in exogenous consumption
        housing_wealth: np.ndarray = None,  # Ignored in exogenous consumption
        rent: np.ndarray = None,  # Ignored in exogenous consumption
        mortgage_debt: np.ndarray = None,  # Ignored in exogenous consumption
        mortgage_payment: np.ndarray = None,  # Ignored in exogenous consumption
        house_price_index: float | np.ndarray = None,  # Ignored in exogenous consumption
        house_price_growth: float | np.ndarray = None,  # Ignored in exogenous consumption
        lagged_consumption: np.ndarray = None,  # Ignored in exogenous consumption
    ) -> np.ndarray:
        """Calculate target consumption using exogenous targets.

        Determines consumption based on:
        - External consumption targets
        - Price level changes
        - Income-based allocation
        - Tax adjustments

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            historic_consumption_sum (np.ndarray): Past consumption totals
            saving_rates (np.ndarray): Household saving rates
            income (np.ndarray): Household income
            household_benefits (np.ndarray): Social benefits received
            consumption_weights (np.ndarray): Industry consumption shares
            consumption_weights_by_income (np.ndarray): Income-based weights
            exogenous_total_consumption (np.ndarray): External consumption target
            current_time (int): Current period
            take_consumption_weights_by_income_quantile (bool): Use income quintiles
            tau_vat (float): Value added tax rate

        Returns:
            np.ndarray: Target consumption by household and industry
        """
        target_consumption = np.maximum(
            0.0,
            (
                1.0
                / (1 + tau_vat)
                * np.outer(
                    consumption_weights,
                    (1 - saving_rates) * income,
                ).T
            ),
        )
        return (
            (1 + expected_inflation)
            * current_cpi
            / initial_cpi
            * 1.0
            / (1 + tau_vat)
            * exogenous_total_consumption[current_time]
            * target_consumption
            / target_consumption.sum()
        )
