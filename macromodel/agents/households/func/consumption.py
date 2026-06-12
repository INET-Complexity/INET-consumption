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
        owner_occupied: np.ndarray = None,
        mortgagor: np.ndarray = None,
        house_price_index: float | np.ndarray = None,
        house_price_growth: float | np.ndarray = None,
        lagged_consumption: np.ndarray = None,
        lagged_income: np.ndarray = None,
        lagged_cpi: float | None = None,
        lagged_liquid_wealth: np.ndarray = None,
        lagged_illiquid_wealth: np.ndarray = None,
        lagged_mortgage_debt: np.ndarray = None,
        lagged_consumption_loan_debt: np.ndarray = None,
        lagged_house_price_index: float | np.ndarray = None,
        real_borrowing_rate: float | np.ndarray = None,
        permanent_income_log_ratio: float | np.ndarray = None,
        consumer_debt_rate_delta: float | np.ndarray = None,
        uncertainty_delta: float | np.ndarray = None,
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
            owner_occupied (np.ndarray | None): Owner-occupied main-residence flag
            mortgagor (np.ndarray | None): Active mortgage flag
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
        owner_occupied: np.ndarray = None,  # Ignored in default consumption
        mortgagor: np.ndarray = None,  # Ignored in default consumption
        house_price_index: float | np.ndarray = None,  # Ignored in default consumption
        house_price_growth: float | np.ndarray = None,  # Ignored in default consumption
        lagged_consumption: np.ndarray = None,  # Ignored in default consumption
        lagged_income: np.ndarray = None,  # Ignored in default consumption
        lagged_cpi: float | None = None,  # Ignored in default consumption
        lagged_liquid_wealth: np.ndarray = None,  # Ignored in default consumption
        lagged_illiquid_wealth: np.ndarray = None,  # Ignored in default consumption
        lagged_mortgage_debt: np.ndarray = None,  # Ignored in default consumption
        lagged_consumption_loan_debt: np.ndarray = None,  # Ignored in default consumption
        lagged_house_price_index: float | np.ndarray = None,  # Ignored in default consumption
        real_borrowing_rate: float | np.ndarray = None,  # Ignored in default consumption
        permanent_income_log_ratio: float | np.ndarray = None,  # Ignored in default consumption
        consumer_debt_rate_delta: float | np.ndarray = None,  # Ignored in default consumption
        uncertainty_delta: float | np.ndarray = None,  # Ignored in default consumption
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
        owner_occupied: np.ndarray = None,  # Ignored in CES consumption
        mortgagor: np.ndarray = None,  # Ignored in CES consumption
        house_price_index: float | np.ndarray = None,  # Ignored in CES consumption
        house_price_growth: float | np.ndarray = None,  # Ignored in CES consumption
        lagged_consumption: np.ndarray = None,  # Ignored in CES consumption
        lagged_income: np.ndarray = None,  # Ignored in CES consumption
        lagged_cpi: float | None = None,  # Ignored in CES consumption
        lagged_liquid_wealth: np.ndarray = None,  # Ignored in CES consumption
        lagged_illiquid_wealth: np.ndarray = None,  # Ignored in CES consumption
        lagged_mortgage_debt: np.ndarray = None,  # Ignored in CES consumption
        lagged_consumption_loan_debt: np.ndarray = None,  # Ignored in CES consumption
        lagged_house_price_index: float | np.ndarray = None,  # Ignored in CES consumption
        real_borrowing_rate: float | np.ndarray = None,  # Ignored in CES consumption
        permanent_income_log_ratio: float | np.ndarray = None,  # Ignored in CES consumption
        consumer_debt_rate_delta: float | np.ndarray = None,  # Ignored in CES consumption
        uncertainty_delta: float | np.ndarray = None,  # Ignored in CES consumption
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

    Feasible Stage 2 proxy for the paper log-linear consumption equation.

    Rent and scheduled mortgage service are intentionally diagnostics only. The
    behavioural target uses current real spendable income, lagged real income,
    lagged real consumption, paper-style lagged NLA, lagged IFA, current housing
    wealth, and lagged HPI. Stage 3 permanent-income, consumer-debt-rate, and
    uncertainty terms remain explicit zero placeholders unless supplied.
    """

    def __init__(
        self,
        consumption_smoothing_fraction: float,
        consumption_smoothing_window: int,
        minimum_consumption_fraction: float,
        elasticity_of_substitution: float = 1.0,
        long_run_intercept: float = 0.0,
        real_borrowing_rate_propensity: float = 0.0,
        permanent_income_propensity: float = 1.0,
        liquid_wealth_propensity: float = 0.04,
        illiquid_wealth_propensity: float = 0.02,
        housing_wealth_propensity: float = 0.02,
        rent_propensity: float = 1.0,
        mortgage_debt_propensity: float = 0.03,
        mortgage_payment_propensity: float = 1.0,
        house_price_propensity: float = 0.02,
        income_growth_propensity: float = 0.0,
        interest_rate_cashflow_propensity: float | None = None,
        uncertainty_propensity: float | None = None,
        partial_adjustment_speed: float = 0.5,
        price_floor: float = 1e-12,
        income_floor: float = 1e-12,
        consumption_floor: float = 1e-12,
        house_price_floor: float = 1e-12,
    ):
        super().__init__(
            consumption_smoothing_fraction,
            consumption_smoothing_window,
            minimum_consumption_fraction,
            elasticity_of_substitution,
        )
        self.long_run_intercept = long_run_intercept
        self.real_borrowing_rate_propensity = real_borrowing_rate_propensity
        self.permanent_income_propensity = permanent_income_propensity
        self.liquid_wealth_propensity = liquid_wealth_propensity
        self.illiquid_wealth_propensity = illiquid_wealth_propensity
        self.housing_wealth_propensity = housing_wealth_propensity
        # Retained for config compatibility only; these do not enter Stage 2 target.
        self.rent_propensity = rent_propensity
        self.mortgage_debt_propensity = mortgage_debt_propensity
        self.mortgage_payment_propensity = mortgage_payment_propensity
        self.house_price_propensity = house_price_propensity
        self.income_growth_propensity = income_growth_propensity
        self.interest_rate_cashflow_propensity = interest_rate_cashflow_propensity
        self.uncertainty_propensity = uncertainty_propensity
        self.partial_adjustment_speed = partial_adjustment_speed
        self.price_floor = price_floor
        self.income_floor = income_floor
        self.consumption_floor = consumption_floor
        self.house_price_floor = house_price_floor
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
        lagged_income: np.ndarray,
        lagged_consumption: np.ndarray,
        liquid_wealth: np.ndarray,
        illiquid_wealth: np.ndarray,
        housing_wealth: np.ndarray,
        rent: np.ndarray,
        mortgage_debt: np.ndarray,
        mortgage_payment: np.ndarray,
        lagged_liquid_wealth: np.ndarray,
        lagged_illiquid_wealth: np.ndarray,
        lagged_mortgage_debt: np.ndarray,
        lagged_consumption_loan_debt: np.ndarray,
        owner_occupied: np.ndarray,
        mortgagor: np.ndarray,
        house_price_index: float | np.ndarray | None,
        lagged_house_price_index: float | np.ndarray | None,
        real_borrowing_rate: float | np.ndarray | None,
        permanent_income_log_ratio: float | np.ndarray | None,
        consumer_debt_rate_delta: float | np.ndarray | None,
        uncertainty_delta: float | np.ndarray | None,
        current_cpi: float,
        lagged_cpi: float | None,
        initial_cpi: float,
        expected_inflation: float,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        income = np.asarray(income, dtype=float)
        lagged_income = np.asarray(lagged_income, dtype=float)
        lagged_consumption = np.asarray(lagged_consumption, dtype=float)
        liquid_wealth = np.asarray(liquid_wealth, dtype=float)
        illiquid_wealth = np.asarray(illiquid_wealth, dtype=float)
        housing_wealth = np.asarray(housing_wealth, dtype=float)
        rent = np.asarray(rent, dtype=float)
        mortgage_debt = np.asarray(mortgage_debt, dtype=float)
        mortgage_payment = np.asarray(mortgage_payment, dtype=float)
        lagged_liquid_wealth = np.asarray(lagged_liquid_wealth, dtype=float)
        lagged_illiquid_wealth = np.asarray(lagged_illiquid_wealth, dtype=float)
        lagged_mortgage_debt = np.asarray(lagged_mortgage_debt, dtype=float)
        lagged_consumption_loan_debt = np.asarray(lagged_consumption_loan_debt, dtype=float)
        owner_occupied = np.asarray(owner_occupied, dtype=float)
        mortgagor = np.asarray(mortgagor, dtype=float)

        house_price_index_arr = self._as_array(income, house_price_index, default=1.0)
        lagged_house_price_index_arr = self._as_array(
            income,
            lagged_house_price_index if lagged_house_price_index is not None else house_price_index,
            default=1.0,
        )
        real_borrowing_rate_arr = self._as_array(income, real_borrowing_rate, default=0.0)
        permanent_income_log_ratio_arr = self._as_array(income, permanent_income_log_ratio, default=0.0)
        consumer_debt_rate_delta_arr = self._as_array(income, consumer_debt_rate_delta, default=0.0)
        uncertainty_delta_arr = self._as_array(income, uncertainty_delta, default=0.0)

        initial_cpi = float(initial_cpi) if initial_cpi != 0.0 else 1.0
        current_deflator = max(float(current_cpi) / initial_cpi, self.price_floor)
        lagged_deflator = max(float(current_cpi if lagged_cpi is None else lagged_cpi) / initial_cpi, self.price_floor)
        nominalizer = current_deflator * (1.0 + expected_inflation)

        real_spendable_income = np.maximum(income / current_deflator, self.income_floor)
        real_lagged_income = np.maximum(lagged_income / lagged_deflator, self.income_floor)
        real_lagged_consumption = np.maximum(lagged_consumption / lagged_deflator, self.consumption_floor)
        real_net_liquid_assets = (
            lagged_liquid_wealth - lagged_mortgage_debt - lagged_consumption_loan_debt
        ) / lagged_deflator
        real_illiquid_financial_assets = lagged_illiquid_wealth / lagged_deflator
        real_housing_wealth = housing_wealth / current_deflator
        real_consumer_debt = lagged_consumption_loan_debt / lagged_deflator
        real_lagged_house_price = np.maximum(lagged_house_price_index_arr / lagged_deflator, self.house_price_floor)

        net_liquid_assets_term = self.liquid_wealth_propensity * real_net_liquid_assets / real_spendable_income
        illiquid_assets_term = (
            self.illiquid_wealth_propensity * real_illiquid_financial_assets / real_spendable_income
        )
        house_price_term = self.house_price_propensity * np.log(real_lagged_house_price / real_lagged_income)
        housing_wealth_term = self.housing_wealth_propensity * real_housing_wealth / real_lagged_income
        permanent_income_term = self.permanent_income_propensity * permanent_income_log_ratio_arr
        real_borrowing_rate_term = self.real_borrowing_rate_propensity * real_borrowing_rate_arr

        long_run_log_consumption_to_income = (
            self.long_run_intercept
            + real_borrowing_rate_term
            + permanent_income_term
            + net_liquid_assets_term
            + illiquid_assets_term
            + house_price_term
            + housing_wealth_term
        )
        log_long_run_target = np.log(real_spendable_income) + long_run_log_consumption_to_income
        long_run_target_real = np.exp(np.clip(log_long_run_target, -50.0, 50.0))

        income_growth_term = self.income_growth_propensity * (
            np.log(real_spendable_income) - np.log(real_lagged_income)
        )
        interest_rate_cashflow_index = consumer_debt_rate_delta_arr * (real_consumer_debt / real_spendable_income)
        interest_rate_cashflow_term = np.zeros_like(real_spendable_income)
        if self.interest_rate_cashflow_propensity is not None:
            interest_rate_cashflow_term = self.interest_rate_cashflow_propensity * interest_rate_cashflow_index
        uncertainty_term = np.zeros_like(real_spendable_income)
        if self.uncertainty_propensity is not None:
            uncertainty_term = self.uncertainty_propensity * uncertainty_delta_arr

        partial_adjustment_gap = self.partial_adjustment_speed * (
            np.log(long_run_target_real) - np.log(real_lagged_consumption)
        )
        delta_log_consumption = (
            partial_adjustment_gap + income_growth_term + interest_rate_cashflow_term + uncertainty_term
        )
        target_total_real = np.maximum(
            self.consumption_floor,
            real_lagged_consumption * np.exp(np.clip(delta_log_consumption, -50.0, 50.0)),
        )
        target_total = target_total_real * nominalizer

        components = {
            "target_consumption_lagged_consumption": lagged_consumption,
            "target_consumption_real_income": real_spendable_income,
            "target_consumption_lagged_real_income": real_lagged_income,
            "target_consumption_real_lagged_consumption": real_lagged_consumption,
            "target_consumption_long_run": long_run_target_real * nominalizer,
            "target_consumption_log_long_run": long_run_log_consumption_to_income,
            "target_consumption_permanent_income": permanent_income_term,
            "target_consumption_liquid_wealth": net_liquid_assets_term,
            "target_consumption_illiquid_wealth": illiquid_assets_term,
            "target_consumption_housing_wealth": housing_wealth_term,
            "target_consumption_real_net_liquid_assets": real_net_liquid_assets,
            "target_consumption_real_illiquid_financial_assets": real_illiquid_financial_assets,
            "target_consumption_real_housing_wealth": real_housing_wealth,
            "target_consumption_real_consumer_debt": real_consumer_debt,
            "target_consumption_rent": np.zeros_like(real_spendable_income),
            "target_consumption_mortgage_debt": np.zeros_like(real_spendable_income),
            "target_consumption_mortgage_payment": np.zeros_like(real_spendable_income),
            "target_consumption_rent_diagnostic": rent,
            "target_consumption_mortgage_debt_diagnostic": mortgage_debt,
            "target_consumption_mortgage_payment_diagnostic": mortgage_payment,
            "target_consumption_house_price": house_price_term,
            "target_consumption_interest_rate_cashflow": interest_rate_cashflow_term,
            "target_consumption_uncertainty": uncertainty_term,
            "target_consumption_partial_adjustment_gap": partial_adjustment_gap,
            "target_consumption_income_growth": income_growth_term,
            "target_consumption_house_price_index": house_price_index_arr,
            "target_consumption_lagged_house_price_index": lagged_house_price_index_arr,
            "target_consumption_real_lagged_house_price": real_lagged_house_price,
            "target_consumption_real_borrowing_rate": real_borrowing_rate_arr,
            "target_consumption_permanent_income_log_ratio": permanent_income_log_ratio_arr,
            "target_consumption_consumer_debt_rate_delta": consumer_debt_rate_delta_arr,
            "target_consumption_interest_rate_cashflow_index": interest_rate_cashflow_index,
            "target_consumption_uncertainty_delta": uncertainty_delta_arr,
            "target_consumption_owner_occupied": owner_occupied,
            "target_consumption_mortgagor": mortgagor,
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
        owner_occupied: np.ndarray = None,
        mortgagor: np.ndarray = None,
        house_price_index: float | np.ndarray = None,
        house_price_growth: float | np.ndarray = None,
        lagged_consumption: np.ndarray = None,
        lagged_income: np.ndarray = None,
        lagged_cpi: float | None = None,
        lagged_liquid_wealth: np.ndarray = None,
        lagged_illiquid_wealth: np.ndarray = None,
        lagged_mortgage_debt: np.ndarray = None,
        lagged_consumption_loan_debt: np.ndarray = None,
        lagged_house_price_index: float | np.ndarray = None,
        real_borrowing_rate: float | np.ndarray = None,
        permanent_income_log_ratio: float | np.ndarray = None,
        consumer_debt_rate_delta: float | np.ndarray = None,
        uncertainty_delta: float | np.ndarray = None,
    ) -> np.ndarray:
        if lagged_consumption is None:
            lagged_consumption = np.asarray(historic_consumption_sum, dtype=float)[-1]
        else:
            lagged_consumption = np.asarray(lagged_consumption, dtype=float)

        income = np.asarray(income, dtype=float)
        lagged_income = income if lagged_income is None else np.asarray(lagged_income, dtype=float)
        liquid_wealth = self._as_array(income, liquid_wealth)
        illiquid_wealth = self._as_array(income, illiquid_wealth)
        housing_wealth = self._as_array(income, housing_wealth)
        rent = self._as_array(income, rent)
        mortgage_debt = self._as_array(income, mortgage_debt)
        mortgage_payment = self._as_array(income, mortgage_payment)
        lagged_liquid_wealth = self._as_array(income, lagged_liquid_wealth if lagged_liquid_wealth is not None else liquid_wealth)
        lagged_illiquid_wealth = self._as_array(
            income,
            lagged_illiquid_wealth if lagged_illiquid_wealth is not None else illiquid_wealth,
        )
        lagged_mortgage_debt = self._as_array(
            income,
            lagged_mortgage_debt if lagged_mortgage_debt is not None else mortgage_debt,
        )
        lagged_consumption_loan_debt = self._as_array(income, lagged_consumption_loan_debt)
        owner_occupied = self._as_array(income, owner_occupied)
        mortgagor = self._as_array(income, mortgagor)

        components, target_total = self._evaluate_target(
            income=income,
            lagged_income=lagged_income,
            lagged_consumption=lagged_consumption,
            liquid_wealth=liquid_wealth,
            illiquid_wealth=illiquid_wealth,
            housing_wealth=housing_wealth,
            rent=rent,
            mortgage_debt=mortgage_debt,
            mortgage_payment=mortgage_payment,
            lagged_liquid_wealth=lagged_liquid_wealth,
            lagged_illiquid_wealth=lagged_illiquid_wealth,
            lagged_mortgage_debt=lagged_mortgage_debt,
            lagged_consumption_loan_debt=lagged_consumption_loan_debt,
            owner_occupied=owner_occupied,
            mortgagor=mortgagor,
            house_price_index=house_price_index,
            lagged_house_price_index=lagged_house_price_index,
            real_borrowing_rate=real_borrowing_rate,
            permanent_income_log_ratio=permanent_income_log_ratio,
            consumer_debt_rate_delta=consumer_debt_rate_delta,
            uncertainty_delta=uncertainty_delta,
            current_cpi=current_cpi,
            lagged_cpi=lagged_cpi,
            initial_cpi=initial_cpi,
            expected_inflation=expected_inflation,
        )

        current_deflator = max(float(current_cpi) / (float(initial_cpi) if initial_cpi != 0.0 else 1.0), self.price_floor)
        real_income_perturbation = np.maximum(1.0, np.abs(income / current_deflator) * 1e-4)
        nominal_income_perturbation = real_income_perturbation * current_deflator
        _, perturbed_target = self._evaluate_target(
            income=income + nominal_income_perturbation,
            lagged_income=lagged_income,
            lagged_consumption=lagged_consumption,
            liquid_wealth=liquid_wealth,
            illiquid_wealth=illiquid_wealth,
            housing_wealth=housing_wealth,
            rent=rent,
            mortgage_debt=mortgage_debt,
            mortgage_payment=mortgage_payment,
            lagged_liquid_wealth=lagged_liquid_wealth,
            lagged_illiquid_wealth=lagged_illiquid_wealth,
            lagged_mortgage_debt=lagged_mortgage_debt,
            lagged_consumption_loan_debt=lagged_consumption_loan_debt,
            owner_occupied=owner_occupied,
            mortgagor=mortgagor,
            house_price_index=house_price_index,
            lagged_house_price_index=lagged_house_price_index,
            real_borrowing_rate=real_borrowing_rate,
            permanent_income_log_ratio=permanent_income_log_ratio,
            consumer_debt_rate_delta=consumer_debt_rate_delta,
            uncertainty_delta=uncertainty_delta,
            current_cpi=current_cpi,
            lagged_cpi=lagged_cpi,
            initial_cpi=initial_cpi,
            expected_inflation=expected_inflation,
        )

        target_consumption = np.maximum(
            0.0,
            1.0
            / (1 + tau_vat)
            * np.outer(
                consumption_weights,
                target_total,
            ).T,
        )
        perturbed_target_consumption = np.maximum(
            0.0,
            1.0 / (1 + tau_vat) * np.outer(consumption_weights, perturbed_target).T,
        )
        self.last_target_consumption_components = components
        self.last_formula_implied_mpc = (
            perturbed_target_consumption.sum(axis=1) - target_consumption.sum(axis=1)
        ) / nominal_income_perturbation
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
        owner_occupied: np.ndarray = None,  # Ignored in exogenous consumption
        mortgagor: np.ndarray = None,  # Ignored in exogenous consumption
        house_price_index: float | np.ndarray = None,  # Ignored in exogenous consumption
        house_price_growth: float | np.ndarray = None,  # Ignored in exogenous consumption
        lagged_consumption: np.ndarray = None,  # Ignored in exogenous consumption
        lagged_income: np.ndarray = None,  # Ignored in exogenous consumption
        lagged_cpi: float | None = None,  # Ignored in exogenous consumption
        lagged_liquid_wealth: np.ndarray = None,  # Ignored in exogenous consumption
        lagged_illiquid_wealth: np.ndarray = None,  # Ignored in exogenous consumption
        lagged_mortgage_debt: np.ndarray = None,  # Ignored in exogenous consumption
        lagged_consumption_loan_debt: np.ndarray = None,  # Ignored in exogenous consumption
        lagged_house_price_index: float | np.ndarray = None,  # Ignored in exogenous consumption
        real_borrowing_rate: float | np.ndarray = None,  # Ignored in exogenous consumption
        permanent_income_log_ratio: float | np.ndarray = None,  # Ignored in exogenous consumption
        consumer_debt_rate_delta: float | np.ndarray = None,  # Ignored in exogenous consumption
        uncertainty_delta: float | np.ndarray = None,  # Ignored in exogenous consumption
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
