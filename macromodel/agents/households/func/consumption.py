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
        uses_income_belief_learning (bool): Whether this rule requests optional
            income-belief learning inputs from the household wrapper.
        consumption_smoothing_fraction (float): Weight on historical consumption
        consumption_smoothing_window (int): Periods for smoothing calculation
        minimum_consumption_fraction (float): Floor on consumption/income ratio
    """

    uses_income_belief_learning = False

    def __init__(
        self,
        consumption_smoothing_fraction: float = 0.0,
        consumption_smoothing_window: int = 12,
        minimum_consumption_fraction: float = 1.0,
        elasticity_of_substitution: float = 1.0,  # Ignored by default consumption
        **kwargs,  # Tolerate rule-specific parameters left in the shared config
        # (e.g. income-belief-learning settings) when a non-supporting rule is selected.
    ):
        self.consumption_smoothing_fraction = consumption_smoothing_fraction
        self.consumption_smoothing_window = consumption_smoothing_window
        self.minimum_consumption_fraction = minimum_consumption_fraction
        # `uses_income_belief_learning` is a CreditAugmentedConsumption-only feature; it
        # stays False (class attribute) for rules that do not implement it, even when the
        # config still carries the flag from a different rule (e.g. after a name override).
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
        lagged_housing_wealth: np.ndarray = None,
        rent: np.ndarray = None,
        rent_imputed: np.ndarray = None,
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
        cashflow_consumer_debt: np.ndarray = None,
        lagged_house_price_index: float | np.ndarray = None,
        real_borrowing_rate: float | np.ndarray = None,
        permanent_income_log_ratio: float | np.ndarray = None,
        consumer_debt_rate_delta: float | np.ndarray = None,
        uncertainty_delta: float | np.ndarray = None,
        population_scale_factor: float | None = None,
        time_unit: int = 12,
        lagged_real_consumption_budget: np.ndarray = None,
        historic_income: np.ndarray = None,
        historic_deflator: np.ndarray = None,
        subsistence_income: np.ndarray | float | None = None,
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
            rent (np.ndarray | None): Actual cash rent paid per household.
                Non-zero for renters and social-housing tenants only; zero for
                owner-occupiers by construction in ``Households.compute_rent``.
            rent_imputed (np.ndarray | None): Imputed rent per household.
                Non-zero for owner-occupiers only; zero for renters by the
                same construction. Mutually exclusive with ``rent``.
            mortgage_debt (np.ndarray | None): Mortgage debt per household
            mortgage_payment (np.ndarray | None): Mortgage payment per household
            owner_occupied (np.ndarray | None): Owner-occupied main-residence flag
            mortgagor (np.ndarray | None): Active mortgage flag
            house_price_index (float | np.ndarray | None): House-price index level
            house_price_growth (float | np.ndarray | None): House-price growth proxy
            lagged_consumption (np.ndarray | None): Previous-period consumption
            population_scale_factor (float | None): Synthetic-population scale
                factor (e.g. FRA scale=5000) used to rescale model-internal
                income back to per-household units where a rule mixes income
                with an unscaled index (see GH issue #90). Ignored by rules
                that do not need it.

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
        lagged_housing_wealth: np.ndarray = None,  # Ignored in default consumption
        rent: np.ndarray = None,  # Ignored in default consumption
        rent_imputed: np.ndarray = None,  # Ignored in default consumption
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
        cashflow_consumer_debt: np.ndarray = None,  # Ignored in default consumption
        lagged_house_price_index: float | np.ndarray = None,  # Ignored in default consumption
        real_borrowing_rate: float | np.ndarray = None,  # Ignored in default consumption
        permanent_income_log_ratio: float | np.ndarray = None,  # Ignored in default consumption
        consumer_debt_rate_delta: float | np.ndarray = None,  # Ignored in default consumption
        uncertainty_delta: float | np.ndarray = None,  # Ignored in default consumption
        population_scale_factor: float | None = None,  # Ignored in default consumption
        time_unit: int = 12,  # Ignored in default consumption
        lagged_real_consumption_budget: np.ndarray = None,  # Ignored in default consumption
        historic_income: np.ndarray = None,  # Ignored in default consumption
        historic_deflator: np.ndarray = None,  # Ignored in default consumption
        subsistence_income: np.ndarray | float | None = None,  # Ignored in default consumption
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
        consumption_smoothing_fraction: float = 0.0,
        consumption_smoothing_window: int = 12,
        minimum_consumption_fraction: float = 1.0,
        elasticity_of_substitution: float = 1.0,
        **kwargs,  # Tolerate rule-specific parameters left in the shared config.
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
        lagged_housing_wealth: np.ndarray = None,  # Ignored in CES consumption
        rent: np.ndarray = None,  # Ignored in CES consumption
        rent_imputed: np.ndarray = None,  # Ignored in CES consumption
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
        cashflow_consumer_debt: np.ndarray = None,  # Ignored in CES consumption
        lagged_house_price_index: float | np.ndarray = None,  # Ignored in CES consumption
        real_borrowing_rate: float | np.ndarray = None,  # Ignored in CES consumption
        permanent_income_log_ratio: float | np.ndarray = None,  # Ignored in CES consumption
        consumer_debt_rate_delta: float | np.ndarray = None,  # Ignored in CES consumption
        uncertainty_delta: float | np.ndarray = None,  # Ignored in CES consumption
        population_scale_factor: float | None = None,  # Ignored in CES consumption
        time_unit: int = 12,  # Ignored in CES consumption
        lagged_real_consumption_budget: np.ndarray = None,  # Ignored in CES consumption
        historic_income: np.ndarray = None,  # Ignored in CES consumption
        historic_deflator: np.ndarray = None,  # Ignored in CES consumption
        subsistence_income: np.ndarray | float | None = None,  # Ignored in CES consumption
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

    Rent and scheduled mortgage service are not behavioural regressors: the
    target uses current real spendable income, lagged real income, lagged real
    consumption, paper-style lagged NLA, lagged IFA, lagged housing wealth, and
    lagged HPI. Stage 3 permanent-income, consumer-debt-rate, and uncertainty
    terms remain explicit zero placeholders unless supplied.

    The resulting ``target_total`` is the calibrated consumption concept. Cash
    rent is removed from the market-consumption demand and remains a separate
    household cash use paid to landlords. Imputed rent is diagnostic-only and
    is not removed from the behavioural target or introduced as a cash use.
    Scheduled mortgage service remains debt service and is only diagnostic here.

    VAT convention (GH #123): ``target_total`` and the persisted ECM state
    (``target_consumption_real_budget`` / ``lagged_real_consumption_budget``)
    are both gross of VAT, matching the national-accounts, purchaser-price
    basis ``credit_augmented_v1`` was calibrated against. VAT is stripped only
    once, at the goods-market allocation step (``1 / (1 + tau_vat)``), after
    the target/lag comparison is made. Realised ``ts.consumption`` is
    therefore net of VAT; call sites that need a gross-comparable aggregate
    (e.g. ``Households.update_wealth``'s ``total_consumption``) must
    re-gross it with ``(1 + tau_vat)`` rather than compare it directly
    against ``target_total``. This basis is a settled calibration decision,
    not an open question: do not recalibrate ``partial_adjustment_speed`` or
    the other propensities against it.
    """

    def __init__(
        self,
        consumption_smoothing_fraction: float = 0.0,
        consumption_smoothing_window: int = 12,
        minimum_consumption_fraction: float = 1.0,
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
        uses_income_belief_learning: bool = False,
        income_belief_learning_horizon: dict | None = None,
        income_belief_growth_clip_bound: float = 1.0,
        long_run_mpc_lower_bound: float = 0.0,
        long_run_mpc_upper_bound: float = 1.0,
        uses_continuous_wealth_calibration: bool = False,
        continuous_wealth_calibration: dict | None = None,
        idiosyncratic_sd: float = 0.0,
        idiosyncratic_persistence: str = "fixed_effect",
        idiosyncratic_seed: int | None = None,
        income_denominator: str = "current",
        income_denominator_window: int = 20,
    ):
        super().__init__(
            consumption_smoothing_fraction,
            consumption_smoothing_window,
            minimum_consumption_fraction,
            elasticity_of_substitution,
        )
        # Income-belief learning is implemented by this rule only.
        self.uses_income_belief_learning = uses_income_belief_learning
        self.income_belief_growth_clip_bound = income_belief_growth_clip_bound
        if long_run_mpc_upper_bound < long_run_mpc_lower_bound:
            raise ValueError(
                "long_run_mpc_upper_bound must be greater than or equal to "
                f"long_run_mpc_lower_bound, got "
                f"({long_run_mpc_lower_bound}, {long_run_mpc_upper_bound})."
            )
        self.long_run_mpc_lower_bound = long_run_mpc_lower_bound
        self.long_run_mpc_upper_bound = long_run_mpc_upper_bound
        self.long_run_intercept = long_run_intercept
        self.real_borrowing_rate_propensity = real_borrowing_rate_propensity
        self.permanent_income_propensity = permanent_income_propensity
        self.liquid_wealth_propensity = liquid_wealth_propensity
        self.illiquid_wealth_propensity = illiquid_wealth_propensity
        self.housing_wealth_propensity = housing_wealth_propensity
        # Continuous CACF household-group calibration (HFCS 2014 France-fitted):
        # replaces the single global (permanent_income_propensity, liquid_wealth_propensity)
        # pair with per-household alpha_2(B_i)/gamma_1(B_i), B_i a fitted accessibility
        # index over winsorized NLA/y, IFA/y, HA/y. Off by default; when off, behaviour
        # is byte-identical to the pre-existing global-coefficient path. See
        # knowledge-vault/wiki/concepts/cacf-household-group-calibration.md
        # ("Continuous Calibration Alternative") for the fitted constants' provenance.
        self.uses_continuous_wealth_calibration = uses_continuous_wealth_calibration
        calibration = continuous_wealth_calibration or {}
        self.continuous_wealth_calibration_weights = (
            calibration.get("weight_net_liquid_assets", 1.0),
            calibration.get("weight_illiquid_financial_assets", 0.502),
            calibration.get("weight_housing_assets", 0.287),
        )
        # v1 shared ONE steepness and ONE midpoint between alpha_2 and gamma_1. v2
        # estimates them separately and the restriction is rejected decisively
        # (k_alpha = 2.01 against k_gamma = 148.4). The legacy `steepness`/`b0` keys
        # remain the fallback so a v1 config keeps its exact behaviour.
        self.continuous_wealth_calibration_steepness = calibration.get("steepness", 34.3)
        self.continuous_wealth_calibration_alpha_2_steepness = calibration.get(
            "alpha_2_steepness", self.continuous_wealth_calibration_steepness
        )
        self.continuous_wealth_calibration_gamma_1_steepness = calibration.get(
            "gamma_1_steepness", self.continuous_wealth_calibration_steepness
        )
        # How the index weights combine with the ratios:
        #   "normalised_ratio" (v1) -- min-max each ratio to [0,1] first, then weight.
        #   "raw_ratio"        (v2) -- weight the clipped ratios directly, which is how
        #                              the HFCS calibration fits them. The two are NOT
        #                              interchangeable: the ratios differ by an order of
        #                              magnitude in scale, so under "raw_ratio" housing
        #                              dominates B despite the smallest weight.
        self.continuous_wealth_calibration_index_construction = calibration.get(
            "index_construction", "normalised_ratio"
        )
        if self.continuous_wealth_calibration_index_construction not in ("normalised_ratio", "raw_ratio"):
            raise ValueError(
                "index_construction must be 'normalised_ratio' or 'raw_ratio', got "
                f"{self.continuous_wealth_calibration_index_construction!r}."
            )
        self.continuous_wealth_calibration_alpha_2_range = (
            calibration.get("alpha_2_low", 0.2497),
            calibration.get("alpha_2_high", 0.6997),
        )
        self.continuous_wealth_calibration_gamma_1_range = (
            calibration.get("gamma_1_low", 0.0503),
            calibration.get("gamma_1_high", 0.1997),
        )
        # [p5,p95] winsorization bounds for NLA/y, IFA/y, HA/y, fitted once on HFCS
        # 2014 France micro-data (see the design doc's Cell-Size Check table).
        # Calibration-fixed by design: not recomputed from the live simulated
        # population each period.
        self.continuous_wealth_calibration_ratio_bounds = (
            calibration.get("net_liquid_assets_ratio_bounds", (-4.73, 2.15)),
            calibration.get("illiquid_financial_assets_ratio_bounds", (0.0, 3.54)),
            calibration.get("housing_assets_ratio_bounds", (0.0, 17.08)),
        )
        # Calibration-fixed B-index normalization range and logistic centre, computed
        # once on the HFCS 2014 France cross-section with the weights/bounds above
        # (B_raw = sum of weighted normalized ratios, each in [0,1] after clipping, so
        # its theoretical range is [0, weight_net_liquid_assets + weight_illiquid_financial_assets
        # + weight_housing_assets]; b0 is the population-weighted median of the resulting
        # B_tilde). Must NOT be recomputed from the live simulated household batch each
        # period -- doing so would make a household's alpha_2/gamma_1 depend on which
        # other households happen to be in the same call, including the income-perturbed
        # MPC probe in compute_target_consumption, contrary to the fixed mapping the
        # weights/ranges above are fitted against.
        self.continuous_wealth_calibration_b_raw_bounds = (
            calibration.get("b_raw_min", 0.0),
            calibration.get("b_raw_max", 1.789),
        )
        self.continuous_wealth_calibration_b0 = calibration.get("b0", 0.428)
        self.continuous_wealth_calibration_alpha_2_midpoint = calibration.get(
            "alpha_2_midpoint", self.continuous_wealth_calibration_b0
        )
        self.continuous_wealth_calibration_gamma_1_midpoint = calibration.get(
            "gamma_1_midpoint", self.continuous_wealth_calibration_b0
        )
        # Fail fast on degenerate calibration bounds rather than letting them silently
        # divide-by-zero into nan/inf inside _compute_continuous_wealth_calibration
        # (ratio/b_raw bounds), or silently invert the accessibility-to-coefficient
        # mapping (alpha_2/gamma_1 ranges) -- these are config-driven and only checked
        # once per object lifetime here.
        for lo, hi in self.continuous_wealth_calibration_ratio_bounds:
            if hi <= lo:
                raise ValueError(f"Wealth-ratio calibration bounds must satisfy hi > lo, got ({lo}, {hi}).")
        b_raw_min, b_raw_max = self.continuous_wealth_calibration_b_raw_bounds
        if b_raw_max <= b_raw_min:
            raise ValueError(f"b_raw calibration bounds must satisfy max > min, got ({b_raw_min}, {b_raw_max}).")
        alpha_2_lo, alpha_2_hi = self.continuous_wealth_calibration_alpha_2_range
        if alpha_2_hi <= alpha_2_lo:
            raise ValueError(f"alpha_2 calibration range must satisfy high > low, got ({alpha_2_lo}, {alpha_2_hi}).")
        gamma_1_lo, gamma_1_hi = self.continuous_wealth_calibration_gamma_1_range
        if gamma_1_hi <= gamma_1_lo:
            raise ValueError(f"gamma_1 calibration range must satisfy high > low, got ({gamma_1_lo}, {gamma_1_hi}).")
        # Idiosyncratic term eps in log(C/Y). The HFCS calibration estimates its
        # standard deviation jointly with the mapping; it accounts for 78.5% of the
        # cross-sectional variance of log(C/Y), the structural part for 21.5%.
        #
        # PERSISTENCE IS A MODELLING DECISION, NOT AN ESTIMATE. A single cross-section
        # identifies the variance of eps but says nothing about whether it is drawn
        # afresh each period or is a permanent household characteristic. Implemented as
        # a fixed household effect: i.i.d.-per-period noise would very nearly average
        # out of aggregate consumption, making sigma_eps a nuisance term, whereas a
        # fixed effect is persistent taste heterogeneity that survives aggregation and
        # interacts with the wealth distribution.
        if idiosyncratic_sd < 0.0:
            raise ValueError(f"idiosyncratic_sd must be non-negative, got {idiosyncratic_sd}.")
        if idiosyncratic_persistence not in ("fixed_effect", "iid"):
            raise ValueError(
                f"idiosyncratic_persistence must be 'fixed_effect' or 'iid', got {idiosyncratic_persistence!r}."
            )
        self.idiosyncratic_sd = idiosyncratic_sd
        self.idiosyncratic_persistence = idiosyncratic_persistence
        # Dedicated generator, NOT numpy's global stream. Drawing from the global
        # stream would shift every downstream draw in the model and invalidate all
        # existing seeded baselines for a reason unrelated to consumption.
        self.idiosyncratic_seed = idiosyncratic_seed
        self._idiosyncratic_rng = np.random.default_rng(
            0x00C0FFEE if idiosyncratic_seed is None else int(idiosyncratic_seed)
        )
        self._household_epsilon: np.ndarray | None = None

        # Denominator of the three balance-sheet ratios and of C/y.
        #   "current"           (v1) -- current-period income, annualised.
        #   "geometric_average" (v2) -- trailing geometric mean of real income over
        #                               `income_denominator_window` periods, which is
        #                               the concept the HFCS calibration divides by
        #                               (disp_geom_avg_income, a smoothed multi-year
        #                               income). The single-period denominator is the
        #                               near-zero-income mechanism behind issue #90.
        if income_denominator not in ("current", "geometric_average"):
            raise ValueError(
                f"income_denominator must be 'current' or 'geometric_average', got {income_denominator!r}."
            )
        if income_denominator_window < 1:
            raise ValueError(f"income_denominator_window must be >= 1, got {income_denominator_window}.")
        self.income_denominator = income_denominator
        self.income_denominator_window = int(income_denominator_window)

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
        # Quarterly discount factor delta and horizon S used to compute the
        # scalar individual weight zeta. Sourced from
        # stage_3.income_belief_learning.permanent_income_log_ratio via the
        # paper_parameter reference mechanism. Left as None when not configured;
        # current_income_belief_learning_inputs() raises rather than silently
        # defaulting, since zeta has real economic meaning and has no safe default.
        horizon = income_belief_learning_horizon or {}
        self.income_belief_learning_delta = horizon.get("delta")
        self.income_belief_learning_S = horizon.get("S")
        self.last_target_consumption_components: dict[str, np.ndarray] | None = None
        self.last_formula_implied_mpc: np.ndarray | None = None
        self.last_real_consumption_budget: np.ndarray | None = None

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

    def _compute_continuous_wealth_calibration(
        self,
        net_liquid_assets_ratio: np.ndarray,
        illiquid_financial_assets_ratio: np.ndarray,
        housing_assets_ratio: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-household alpha_2(B)/gamma_1(B) via the fitted accessibility-index mapping.

        B is built from the three balance-sheet ratios, each first clipped to its
        HFCS-fitted [p5,p95] bound (the issue #90 explosion mechanism is in the
        ratios, not in B, so it must be bounded there), then normalized using the
        calibration-fixed B_raw range and centred at the calibration-fixed B0 (both
        fitted once on the HFCS cross-section, not recomputed from this call's
        household batch -- a per-batch min/max/median would make alpha_2/gamma_1
        depend on which other households happen to be evaluated in the same call).
        Finally passed through a logistic. See cacf-household-group-calibration.md.
        """
        nla_bounds, ifa_bounds, ha_bounds = self.continuous_wealth_calibration_ratio_bounds
        nla_clipped = np.clip(net_liquid_assets_ratio, nla_bounds[0], nla_bounds[1])
        ifa_clipped = np.clip(illiquid_financial_assets_ratio, ifa_bounds[0], ifa_bounds[1])
        ha_clipped = np.clip(housing_assets_ratio, ha_bounds[0], ha_bounds[1])

        nla_norm = (nla_clipped - nla_bounds[0]) / (nla_bounds[1] - nla_bounds[0])
        ifa_norm = (ifa_clipped - ifa_bounds[0]) / (ifa_bounds[1] - ifa_bounds[0])
        ha_norm = (ha_clipped - ha_bounds[0]) / (ha_bounds[1] - ha_bounds[0])

        weight_nla, weight_ifa, weight_ha = self.continuous_wealth_calibration_weights
        if self.continuous_wealth_calibration_index_construction == "raw_ratio":
            # The HFCS estimator fits the weights against the raw ratios, so the
            # per-ratio normalisation above must NOT be applied before weighting --
            # doing so silently changes which ratio drives the index.
            b_raw = weight_nla * nla_clipped + weight_ifa * ifa_clipped + weight_ha * ha_clipped
        else:
            b_raw = weight_nla * nla_norm + weight_ifa * ifa_norm + weight_ha * ha_norm
        b_min, b_max = self.continuous_wealth_calibration_b_raw_bounds
        # Clipped to [0,1]: b_raw_min/max are frozen calibration constants, so a
        # simulated household outside the HFCS range would otherwise extrapolate the
        # logistic beyond the domain the mapping was fitted on.
        b_tilde = np.clip((b_raw - b_min) / (b_max - b_min), 0.0, 1.0)

        # Two INDEPENDENT logistics: alpha_2 rises in B, gamma_1 falls in B, and they
        # have their own slopes and midpoints. Under a v1 config both slopes and both
        # midpoints collapse to the shared legacy values, reproducing v1 exactly.
        alpha_2_lo, alpha_2_hi = self.continuous_wealth_calibration_alpha_2_range
        gamma_1_lo, gamma_1_hi = self.continuous_wealth_calibration_gamma_1_range
        logistic_alpha = 1.0 / (
            1.0
            + np.exp(
                np.clip(
                    -self.continuous_wealth_calibration_alpha_2_steepness
                    * (b_tilde - self.continuous_wealth_calibration_alpha_2_midpoint),
                    -700.0,
                    700.0,
                )
            )
        )
        logistic_gamma = 1.0 / (
            1.0
            + np.exp(
                np.clip(
                    -self.continuous_wealth_calibration_gamma_1_steepness
                    * (b_tilde - self.continuous_wealth_calibration_gamma_1_midpoint),
                    -700.0,
                    700.0,
                )
            )
        )
        alpha_2 = alpha_2_lo + (alpha_2_hi - alpha_2_lo) * logistic_alpha
        # gamma_1 falls (not rises) as B rises, mirroring alpha_2's increase.
        gamma_1 = gamma_1_hi - (gamma_1_hi - gamma_1_lo) * logistic_gamma
        return alpha_2, gamma_1

    def set_run_seed(self, seed: int | None) -> None:
        """Re-seed the idiosyncratic generator for a new run, and drop cached draws.

        The rule is built from static config, which has no access to the simulation
        seed, so without this every seed in a Monte Carlo would draw the SAME eps
        vector -- household i would get an identical taste shock in every replication
        and cross-seed dispersion would be understated. The simulation calls this
        wherever it sets its own seed.

        An explicit ``idiosyncratic_seed`` in config wins, so a run can be pinned
        independently of the simulation seed when that is what is wanted. ``None``
        (no simulation seed) falls back to the fixed salt, keeping unseeded runs
        reproducible.
        """
        if self.idiosyncratic_seed is not None:
            return
        self._idiosyncratic_rng = np.random.default_rng(0x00C0FFEE if seed is None else int(seed))
        self._household_epsilon = None

    def _epsilon(self, n_households: int) -> np.ndarray:
        """Idiosyncratic term in log(C/Y), as a fixed household effect.

        Drawn once per household from a DEDICATED generator and cached, so that:

        * the two ``_evaluate_target`` calls inside ``compute_target_consumption``
          (base and income-perturbed) see the SAME eps. They must: the MPC is their
          finite difference over a perturbation of order ``1e-4 * income``, so an
          independent redraw would swamp the derivative with noise of order sigma
          and turn ``last_formula_implied_mpc`` into pure noise;
        * the global numpy stream is untouched, so enabling this does not shift any
          other seeded draw in the model;
        * a household keeps its eps for the whole run.

        Households appended mid-run extend the cache; existing entries are never
        redrawn. ``persistence='iid'`` redraws every call and is provided only for
        comparison -- it breaks the MPC probe by construction and is not a
        supported production setting.

        The cache is indexed by array position, not household identity -- there is
        no household-id concept anywhere in this codebase to do otherwise. That is
        safe only because position IS identity today: nothing in the current
        demography (``NoAging``) ever shrinks the population. If a future
        demography does, a household later re-occupying a freed position would
        silently inherit the departed household's shock. Rather than build unused
        identity-tracking machinery for a scenario that cannot occur yet, a
        shrink is refused outright below, so that if this assumption is ever
        violated it fails loudly at the point of violation instead of silently
        misattributing a taste shock.
        """
        if self.idiosyncratic_sd <= 0.0:
            return np.zeros(n_households, dtype=float)
        if self.idiosyncratic_persistence == "iid":
            return self._idiosyncratic_rng.normal(0.0, self.idiosyncratic_sd, n_households)
        if self._household_epsilon is None:
            self._household_epsilon = self._idiosyncratic_rng.normal(0.0, self.idiosyncratic_sd, n_households)
        elif self._household_epsilon.size < n_households:
            extra = n_households - self._household_epsilon.size
            self._household_epsilon = np.concatenate(
                [self._household_epsilon, self._idiosyncratic_rng.normal(0.0, self.idiosyncratic_sd, extra)]
            )
        elif self._household_epsilon.size > n_households:
            raise ValueError(
                f"Household count shrank from {self._household_epsilon.size} to {n_households}. "
                "The cached idiosyncratic effect is indexed by array position, not household "
                "identity, so a later regrowth could silently hand a new household the departed "
                "household's shock at that position. Refusing rather than handling this "
                "silently; call set_run_seed() to reset the cache if the shrink is intentional."
            )
        return self._household_epsilon[:n_households]

    def _geometric_average_income(
        self,
        historic_income: np.ndarray,
        deflator: float,
        historic_deflator: np.ndarray | None = None,
        subsistence_income: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """Trailing geometric mean of real income over the configured window.

        The HFCS calibration divides consumption and the three balance-sheet stocks
        by ``disp_geom_avg_income`` -- a smoothed multi-year income -- not by a
        single period's income. Reproducing that here is what makes the fitted
        coefficients applicable to the simulated ratios at all.

        Computed in logs on positive observations only. A household with fewer than
        ``income_denominator_window`` periods of history uses whatever positive
        observations it has, so the denominator converges smoothly instead of
        jumping when the window first fills.

        EACH observation is deflated by the price level of ITS OWN period. Deflating
        the whole window by the current price level instead -- which an earlier
        version of this method did -- counts inflation twice for every lagged
        observation: a nominal income earned when the CPI was 1.0 was being divided
        by today's CPI. That biases the average downward by the cumulative inflation
        over the window, so the bias GROWS with the price level, inflating every
        wealth ratio and driving the wealth-drag clip. ``historic_deflator`` is
        therefore required whenever the window reaches back more than one period.

        A household with NO positive income anywhere in the window floors at
        ``subsistence_income`` (CU-adjusted subsistence consumption -- half net SMIC
        per consumption unit, the same concept Stage 5 uses as its floor), not at
        ``income_floor``. Flooring at ``income_floor`` (1e-12) here is exactly the
        near-zero-denominator blow-up mechanism behind issue #90 that this method
        exists to prevent -- every wealth ratio divides by this value. No silent
        fallback: if this case can occur for the caller's inputs, it must supply
        ``subsistence_income``.
        """
        history = np.asarray(historic_income, dtype=float)
        if history.ndim != 2:
            raise ValueError(f"historic_income must be 2-D (periods, households), got shape {history.shape}.")
        window_nominal = history[-self.income_denominator_window :]
        if historic_deflator is None:
            if window_nominal.shape[0] > 1:
                raise ValueError(
                    "income_denominator='geometric_average' over more than one period requires "
                    "historic_deflator (one price level per period of historic_income); deflating a "
                    "multi-period window by the current price level double-counts inflation."
                )
            deflators = np.array([max(float(deflator), self.price_floor)])
        else:
            deflators = np.asarray(historic_deflator, dtype=float).reshape(-1)[-self.income_denominator_window :]
            if deflators.shape[0] != window_nominal.shape[0]:
                raise ValueError(
                    f"historic_deflator has {deflators.shape[0]} periods but historic_income window has "
                    f"{window_nominal.shape[0]}; they must align period-for-period."
                )
            deflators = np.maximum(deflators, self.price_floor)
        window = window_nominal / deflators[:, None]
        usable = np.isfinite(window) & (window > 0.0)
        n_usable = usable.sum(axis=0)
        log_sum = np.where(usable, np.log(np.maximum(window, self.income_floor)), 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            geometric = np.exp(log_sum / np.maximum(n_usable, 1))
        # No positive observation anywhere in the window: there is nothing to average.
        # Floor at subsistence_income (see docstring), not income_floor -- that was
        # the issue #90 blow-up this method exists to prevent.
        # (`income_denominator_min_periods` used to make this threshold configurable;
        # removed as dead code -- every shipped config set it to 1, its minimum legal
        # value, which is exactly this n_usable == 0 condition.)
        insufficient = n_usable == 0
        if insufficient.any():
            if subsistence_income is None:
                raise ValueError(
                    f"{int(insufficient.sum())} household(s) have no positive income anywhere in the "
                    "geometric-average income window, and no subsistence_income floor was supplied. "
                    "Flooring at income_floor here would reintroduce the near-zero-denominator blow-up "
                    "(issue #90) this method exists to prevent; pass subsistence_income explicitly."
                )
            floor = np.maximum(
                np.broadcast_to(np.asarray(subsistence_income, dtype=float), geometric.shape),
                self.income_floor,
            )
            geometric = np.where(insufficient, floor, geometric)
        return np.maximum(geometric, self.income_floor)

    def _clip_wealth_drag(
        self,
        wealth_drag: np.ndarray,
        alpha_2: np.ndarray,
        income_to_consumption_ratio: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clip the combined wealth-drag term to the interval guaranteeing MPC_LR in configured bounds.

        MPC_LR = (C/y) * [(1-alpha_2) - wealth_drag]. Requiring
        long_run_mpc_lower_bound <= MPC_LR <= long_run_mpc_upper_bound gives
        (1-alpha_2) - mpc_upper*(y/C) <= wealth_drag <=
        (1-alpha_2) - mpc_lower*(y/C), where the bound uses
        income-over-consumption (y/C), not its reciprocal. Backstop only -- the
        continuous mapping above should already keep most households in range.
        See cacf-household-group-calibration.md, "Relationship to the Issue
        #90/#93 Clip".
        """
        upper = (1.0 - alpha_2) - self.long_run_mpc_lower_bound * income_to_consumption_ratio
        lower = (1.0 - alpha_2) - self.long_run_mpc_upper_bound * income_to_consumption_ratio
        clipped = np.clip(wealth_drag, lower, upper)
        clipped_flag = (~np.isclose(wealth_drag, clipped)).astype(float)
        return clipped, clipped_flag

    def _evaluate_target(
        self,
        income: np.ndarray,
        lagged_income: np.ndarray,
        lagged_consumption: np.ndarray,
        liquid_wealth: np.ndarray,
        illiquid_wealth: np.ndarray,
        housing_wealth: np.ndarray,
        lagged_housing_wealth: np.ndarray,
        rent: np.ndarray,
        mortgage_debt: np.ndarray,
        mortgage_payment: np.ndarray,
        lagged_liquid_wealth: np.ndarray,
        lagged_illiquid_wealth: np.ndarray,
        lagged_mortgage_debt: np.ndarray,
        lagged_consumption_loan_debt: np.ndarray,
        cashflow_consumer_debt: np.ndarray,
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
        current_time: int,
        population_scale_factor: float | None,
        time_unit: int,
        lagged_real_consumption_budget: np.ndarray | None = None,
        ratio_denominator: np.ndarray | None = None,
        epsilon: np.ndarray | None = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        income = np.asarray(income, dtype=float)
        lagged_income = np.asarray(lagged_income, dtype=float)
        lagged_consumption = np.asarray(lagged_consumption, dtype=float)
        liquid_wealth = np.asarray(liquid_wealth, dtype=float)
        illiquid_wealth = np.asarray(illiquid_wealth, dtype=float)
        housing_wealth = np.asarray(housing_wealth, dtype=float)
        lagged_housing_wealth = np.asarray(lagged_housing_wealth, dtype=float)
        rent = np.asarray(rent, dtype=float)
        mortgage_debt = np.asarray(mortgage_debt, dtype=float)
        mortgage_payment = np.asarray(mortgage_payment, dtype=float)
        lagged_liquid_wealth = np.asarray(lagged_liquid_wealth, dtype=float)
        lagged_illiquid_wealth = np.asarray(lagged_illiquid_wealth, dtype=float)
        lagged_mortgage_debt = np.asarray(lagged_mortgage_debt, dtype=float)
        lagged_consumption_loan_debt = np.asarray(lagged_consumption_loan_debt, dtype=float)
        cashflow_consumer_debt = np.asarray(cashflow_consumer_debt, dtype=float)
        owner_occupied = np.asarray(owner_occupied, dtype=float)
        mortgagor = np.asarray(mortgagor, dtype=float)

        house_price_index_arr = self._as_array(income, house_price_index, default=1.0)
        lagged_house_price_index_arr = self._as_array(
            income,
            lagged_house_price_index if lagged_house_price_index is not None else house_price_index,
            default=1.0,
        )
        # economy.ts["hpi"] is a chained index normalised to a base of 1.0 (economy.py),
        # but house_price_propensity (gamma_4) is calibrated against a conventional
        # base-100 house-price index. Rescale once, using the current-period level so
        # the same factor applies to both the current and lagged series, only when the
        # index still looks base-1.0-normalised; an already-rebased (e.g. base-100)
        # input is left untouched. See GH issue #90.
        if np.nanmax(np.abs(house_price_index_arr)) <= 10.0:
            house_price_index_arr = house_price_index_arr * 100.0
            lagged_house_price_index_arr = lagged_house_price_index_arr * 100.0
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
        # The ECM's state variable is the previous period's *real consumption
        # budget* produced by this rule -- the same concept the current period
        # solves for. When the caller persists that budget it is supplied
        # directly and must not be deflated again (it is already real).
        #
        # Realised consumption is deliberately not accepted as a fallback: it is
        # net of VAT, reflects goods-market rationing and the zero floor, and --
        # since GH #120 carved housing out of goods demand -- excludes rent.
        # Using it would compare a rent-inclusive target against a rent-exclusive
        # lag and drag the target down every period.
        if lagged_real_consumption_budget is None:
            raise ValueError(
                "CreditAugmentedConsumption requires lagged_real_consumption_budget; "
                "realised consumption is not a valid ECM state fallback."
            )
        real_lagged_consumption = np.asarray(lagged_real_consumption_budget, dtype=float)
        if real_lagged_consumption.shape != income.shape or not np.all(np.isfinite(real_lagged_consumption)):
            raise ValueError("lagged_real_consumption_budget must be finite and have one real budget per household.")
        real_lagged_consumption = np.maximum(real_lagged_consumption, self.consumption_floor)
        real_net_liquid_assets = (
            lagged_liquid_wealth - lagged_mortgage_debt - lagged_consumption_loan_debt
        ) / lagged_deflator
        real_illiquid_financial_assets = lagged_illiquid_wealth / lagged_deflator
        real_housing_wealth = housing_wealth / current_deflator
        real_lagged_housing_wealth = lagged_housing_wealth / lagged_deflator
        real_consumer_debt = cashflow_consumer_debt / lagged_deflator
        real_lagged_house_price = np.maximum(lagged_house_price_index_arr / lagged_deflator, self.house_price_floor)
        # house_price_propensity is estimated on hp/y with y at per-household
        # (unscaled) income; real_lagged_income carries the model's synthetic
        # population_scale_factor (e.g. FRA scale=5000), so it must be divided
        # back out before forming this ratio, or the index/income units mismatch
        # by orders of magnitude (see GH issue #90).
        scale = float(population_scale_factor) if population_scale_factor else 1.0
        real_lagged_income_per_household = real_lagged_income / scale

        # The long-run propensities/intercept are calibrated against annual income, but
        # real_spendable_income/real_lagged_income are at model (period) frequency. Annualize
        # them only where they sit in a stock-to-income or price-to-income ratio, so those
        # ratios are on the same scale as the calibration; the level terms that convert the
        # ratio back into a period-frequency consumption target (log_long_run_target below,
        # and income_growth_term, where the factor cancels) stay on period income.
        if time_unit <= 0:
            raise ValueError(f"time_unit must be positive (months per model period), got {time_unit}.")
        annualization_factor = 12.0 / float(time_unit)
        annual_spendable_income = real_spendable_income * annualization_factor
        annual_lagged_income_per_household = real_lagged_income_per_household * annualization_factor

        # net_liquid_assets_ratio/illiquid_assets_ratio/housing_wealth_ratio feed both the
        # legacy long-run terms below and the continuous wealth-calibration mapping
        # (cacf-household-group-calibration.md). They are paired with current-period
        # annual_spendable_income (not lagged income) to match main's convention, and
        # annualized because the HFCS fit underlying the continuous mapping's [p5,p95]
        # clip bounds and B-index constants used annual disp_income; leaving these at
        # period frequency would rescale every household's ratio by annualization_factor
        # and push them outside the calibrated bounds (see the elevated clip-fire-rate
        # finding in the 2026-06-24 session notes).
        #
        # The wealth stocks above (real_net_liquid_assets, real_illiquid_financial_assets,
        # real_lagged_housing_wealth) are deliberately lagged_deflator-based -- that is
        # their own correct real value as of t-1, and other call sites/diagnostics rely
        # on that. Dividing them directly by annual_spendable_income (current_deflator-
        # based) would bake the realised one-period inflation rate (current_deflator /
        # lagged_deflator) into the ratio. Re-deflate the same nominal stocks with
        # current_deflator here, just for ratio formation, so the ratio's deflator
        # cancels against annual_spendable_income's, consistent with pairing these
        # ratios against current-period income.
        current_period_net_liquid_assets = (
            lagged_liquid_wealth - lagged_mortgage_debt - lagged_consumption_loan_debt
        ) / current_deflator
        current_period_illiquid_financial_assets = lagged_illiquid_wealth / current_deflator
        current_period_lagged_housing_wealth = lagged_housing_wealth / current_deflator
        # Denominator of the balance-sheet ratios. Under "geometric_average" this is
        # the trailing geometric mean supplied by the caller, matching the calibration's
        # smoothed-income concept; under "current" it is this period's annualised
        # income, as in v1. It is deliberately NOT recomputed from the perturbed income
        # inside the MPC probe: a five-year average moves by ~1/window of the
        # perturbation, and holding it fixed keeps the probe a clean derivative of the
        # numerator channels.
        ratio_denominator_arr = (
            annual_spendable_income
            if ratio_denominator is None
            else np.maximum(np.asarray(ratio_denominator, dtype=float), self.income_floor)
        )
        net_liquid_assets_ratio = current_period_net_liquid_assets / ratio_denominator_arr
        illiquid_assets_ratio = current_period_illiquid_financial_assets / ratio_denominator_arr
        # Lagged (t-1), not current-period, housing wealth: main fixed a stale
        # current/lagged mixup here (see test_consumption.py's split between
        # target_consumption_real_housing_wealth and _real_lagged_housing_wealth).
        housing_wealth_ratio = current_period_lagged_housing_wealth / ratio_denominator_arr
        # house_price_term combines both unit fixes: per-household descaling (population
        # scale factor, GH issue #90) and annualization (main), since house prices are
        # quoted per household, not per the model's scaled synthetic population.
        house_price_term = self.house_price_propensity * np.log(
            real_lagged_house_price / annual_lagged_income_per_household
        )
        permanent_income_term = self.permanent_income_propensity * permanent_income_log_ratio_arr
        real_borrowing_rate_term = self.real_borrowing_rate_propensity * real_borrowing_rate_arr

        wealth_drag_clipped_flag = np.zeros_like(real_spendable_income)
        if self.uses_continuous_wealth_calibration:
            alpha_2, gamma_1 = self._compute_continuous_wealth_calibration(
                net_liquid_assets_ratio, illiquid_assets_ratio, housing_wealth_ratio
            )
            permanent_income_term = alpha_2 * permanent_income_log_ratio_arr
            wealth_drag = (
                gamma_1 * net_liquid_assets_ratio
                + self.illiquid_wealth_propensity * illiquid_assets_ratio
                + self.housing_wealth_propensity * housing_wealth_ratio
            )
            # MPC_LR = (C/y) * [(1-alpha_2) - (1/y)*(...)] (design doc) requires the
            # SAME y in both the C/y multiplier and the wealth_drag bracket above --
            # wealth_drag's ratios are annual_spendable_income-based, so this y must
            # be too, not real_lagged_income. Lagged C remains the best available
            # proxy for the target's own C -- the target itself isn't computed yet
            # at this point in _evaluate_target. Annualizing C alongside y keeps both
            # sides of this flow-to-flow ratio on the same basis (it cancels back to
            # real_spendable_income / real_lagged_consumption numerically, but written
            # this way to make the matching-y reasoning auditable rather than relying
            # on an algebraic cancellation a future edit could silently break).
            annual_lagged_consumption = real_lagged_consumption * annualization_factor
            income_to_consumption_ratio = ratio_denominator_arr / annual_lagged_consumption
            wealth_drag, wealth_drag_clipped_flag = self._clip_wealth_drag(
                wealth_drag, alpha_2, income_to_consumption_ratio
            )
            # These three diagnostics are the PRE-clip sub-terms (for transparency
            # into each wealth channel's unclipped contribution); they will not sum
            # to the post-clip `wealth_drag` below when the clip has fired -- use
            # target_consumption_wealth_drag_clipped to detect that case.
            net_liquid_assets_term = gamma_1 * net_liquid_assets_ratio
            illiquid_assets_term = self.illiquid_wealth_propensity * illiquid_assets_ratio
            housing_wealth_term = self.housing_wealth_propensity * housing_wealth_ratio
            long_run_log_consumption_to_income = (
                self.long_run_intercept
                + real_borrowing_rate_term
                + permanent_income_term
                + wealth_drag
                + house_price_term
            )
        else:
            alpha_2 = np.full_like(real_spendable_income, self.permanent_income_propensity)
            gamma_1 = np.full_like(real_spendable_income, self.liquid_wealth_propensity)
            net_liquid_assets_term = self.liquid_wealth_propensity * net_liquid_assets_ratio
            illiquid_assets_term = self.illiquid_wealth_propensity * illiquid_assets_ratio
            housing_wealth_term = self.housing_wealth_propensity * housing_wealth_ratio
            permanent_income_term = self.permanent_income_propensity * permanent_income_log_ratio_arr
            long_run_log_consumption_to_income = (
                self.long_run_intercept
                + real_borrowing_rate_term
                + permanent_income_term
                + net_liquid_assets_term
                + illiquid_assets_term
                + house_price_term
                + housing_wealth_term
            )
        # log(C/Y) = x + eps. The estimator's model moments are those of x + eps, so
        # the runtime rule must carry eps too or it is a different specification --
        # the deterministic one, whose dispersion is roughly half its HFCS target.
        if epsilon is not None:
            long_run_log_consumption_to_income = long_run_log_consumption_to_income + epsilon
        log_long_run_target = np.log(real_spendable_income) + long_run_log_consumption_to_income
        long_run_target_real = np.exp(np.clip(log_long_run_target, -50.0, 50.0))

        # At initialisation only, zero observed consumption is not an economic
        # zero: use the household's long-run target as the lagged anchor.  This
        # avoids an artificial log gap from log(epsilon) for the small number of
        # households with zero HFCS consumption.  Later periods retain the
        # realised lagged-consumption state.
        if current_time == 1:
            zero_lagged_consumption = real_lagged_consumption <= 0.0
            real_lagged_consumption = np.where(
                zero_lagged_consumption,
                long_run_target_real,
                real_lagged_consumption,
            )

        income_growth_term = self.income_growth_propensity * (
            np.log(real_spendable_income) - np.log(real_lagged_income)
        )
        interest_rate_cashflow_index = consumer_debt_rate_delta_arr * (real_consumer_debt / annual_spendable_income)
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
        # Economic sanity bound on one-period real consumption growth, distinct from
        # the +-50 exp-overflow guard below (which only prevents inf/nan, not implausible
        # swings -- see GH issue #90). The model runs quarterly; +-0.5 log-units already
        # allows roughly -39%/+65% real consumption growth in a single quarter, well beyond
        # any realistic shock, so hitting this bound flags a misbehaving upstream input
        # (e.g. an unbounded permanent_income_log_ratio) rather than genuine household behaviour.
        growth_clip_bound = 0.5
        delta_log_consumption_clipped = np.clip(delta_log_consumption, -growth_clip_bound, growth_clip_bound)
        growth_clipped_flag = (np.abs(delta_log_consumption) > growth_clip_bound).astype(float)
        target_total_real = np.maximum(
            self.consumption_floor,
            real_lagged_consumption * np.exp(np.clip(delta_log_consumption_clipped, -50.0, 50.0)),
        )
        target_total = target_total_real * nominalizer

        components = {
            "target_consumption_real_budget": target_total_real,
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
            "target_consumption_real_lagged_housing_wealth": real_lagged_housing_wealth,
            "target_consumption_real_consumer_debt": real_consumer_debt,
            # The wealth-ratio denominator actually divided by (finding 2, PR #138
            # review): under "geometric_average" this is the smoothed multi-year
            # income from _geometric_average_income; under "current" it is
            # annual_spendable_income. Persisted so the value driving the
            # wealth-drag clip is inspectable after a run instead of only
            # reconstructible by approximation.
            "target_consumption_ratio_denominator": ratio_denominator_arr,
            "target_consumption_net_liquid_assets_ratio": net_liquid_assets_ratio,
            "target_consumption_illiquid_assets_ratio": illiquid_assets_ratio,
            "target_consumption_housing_wealth_ratio": housing_wealth_ratio,
            "target_consumption_rent": np.zeros_like(real_spendable_income),
            "target_consumption_mortgage_debt": np.zeros_like(real_spendable_income),
            "target_consumption_mortgage_payment": np.zeros_like(real_spendable_income),
            "target_consumption_rent_diagnostic": rent,
            "target_consumption_mortgage_debt_diagnostic": mortgage_debt,
            "target_consumption_mortgage_payment_diagnostic": mortgage_payment,
            "target_consumption_house_price": house_price_term,
            "target_consumption_alpha_2": alpha_2,
            "target_consumption_gamma_1": gamma_1,
            "target_consumption_wealth_drag_clipped": wealth_drag_clipped_flag,
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
            "target_consumption_delta_log_consumption": delta_log_consumption_clipped,
            "target_consumption_growth_clipped": growth_clipped_flag,
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
        lagged_housing_wealth: np.ndarray = None,
        rent: np.ndarray = None,
        rent_imputed: np.ndarray = None,
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
        cashflow_consumer_debt: np.ndarray = None,
        lagged_house_price_index: float | np.ndarray = None,
        real_borrowing_rate: float | np.ndarray = None,
        permanent_income_log_ratio: float | np.ndarray = None,
        consumer_debt_rate_delta: float | np.ndarray = None,
        uncertainty_delta: float | np.ndarray = None,
        population_scale_factor: float | None = None,
        time_unit: int = 12,
        lagged_real_consumption_budget: np.ndarray = None,
        historic_income: np.ndarray = None,
        historic_deflator: np.ndarray = None,
        subsistence_income: np.ndarray | float | None = None,
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
        lagged_housing_wealth = self._as_array(
            income, lagged_housing_wealth if lagged_housing_wealth is not None else housing_wealth
        )
        rent = self._as_array(income, rent)
        rent_imputed = self._as_array(income, rent_imputed)
        mortgage_debt = self._as_array(income, mortgage_debt)
        mortgage_payment = self._as_array(income, mortgage_payment)
        lagged_liquid_wealth = self._as_array(
            income, lagged_liquid_wealth if lagged_liquid_wealth is not None else liquid_wealth
        )
        lagged_illiquid_wealth = self._as_array(
            income,
            lagged_illiquid_wealth if lagged_illiquid_wealth is not None else illiquid_wealth,
        )
        lagged_mortgage_debt = self._as_array(
            income,
            lagged_mortgage_debt if lagged_mortgage_debt is not None else mortgage_debt,
        )
        lagged_consumption_loan_debt = self._as_array(income, lagged_consumption_loan_debt)
        cashflow_consumer_debt = self._as_array(
            income,
            cashflow_consumer_debt if cashflow_consumer_debt is not None else lagged_consumption_loan_debt,
        )
        owner_occupied = self._as_array(income, owner_occupied)
        mortgagor = self._as_array(income, mortgagor)

        # Both computed once here and passed to BOTH evaluations below. The second
        # evaluation perturbs income to difference out the MPC; anything that is
        # redrawn or recomputed between the two calls contaminates that derivative.
        epsilon = self._epsilon(income.size)
        if self.income_denominator == "geometric_average":
            if historic_income is None:
                # No silent fallback: the fitted coefficients are only interpretable
                # against the smoothed denominator they were estimated with, so a
                # missing history is a wiring error, not a default.
                raise ValueError(
                    "income_denominator='geometric_average' requires historic_income "
                    "(periods x households); none was supplied by the caller."
                )
            ratio_denominator = self._geometric_average_income(
                historic_income, current_cpi, historic_deflator, subsistence_income
            )
        else:
            ratio_denominator = None

        components, target_total = self._evaluate_target(
            income=income,
            lagged_income=lagged_income,
            lagged_consumption=lagged_consumption,
            liquid_wealth=liquid_wealth,
            illiquid_wealth=illiquid_wealth,
            housing_wealth=housing_wealth,
            lagged_housing_wealth=lagged_housing_wealth,
            rent=rent,
            mortgage_debt=mortgage_debt,
            mortgage_payment=mortgage_payment,
            lagged_liquid_wealth=lagged_liquid_wealth,
            lagged_illiquid_wealth=lagged_illiquid_wealth,
            lagged_mortgage_debt=lagged_mortgage_debt,
            lagged_consumption_loan_debt=lagged_consumption_loan_debt,
            cashflow_consumer_debt=cashflow_consumer_debt,
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
            current_time=current_time,
            population_scale_factor=population_scale_factor,
            time_unit=time_unit,
            lagged_real_consumption_budget=lagged_real_consumption_budget,
            ratio_denominator=ratio_denominator,
            epsilon=epsilon,
        )

        current_deflator = max(
            float(current_cpi) / (float(initial_cpi) if initial_cpi != 0.0 else 1.0), self.price_floor
        )
        real_income_perturbation = np.maximum(1.0, np.abs(income / current_deflator) * 1e-4)
        nominal_income_perturbation = real_income_perturbation * current_deflator
        _, perturbed_target = self._evaluate_target(
            income=income + nominal_income_perturbation,
            lagged_income=lagged_income,
            lagged_consumption=lagged_consumption,
            liquid_wealth=liquid_wealth,
            illiquid_wealth=illiquid_wealth,
            housing_wealth=housing_wealth,
            lagged_housing_wealth=lagged_housing_wealth,
            rent=rent,
            mortgage_debt=mortgage_debt,
            mortgage_payment=mortgage_payment,
            lagged_liquid_wealth=lagged_liquid_wealth,
            lagged_illiquid_wealth=lagged_illiquid_wealth,
            lagged_mortgage_debt=lagged_mortgage_debt,
            lagged_consumption_loan_debt=lagged_consumption_loan_debt,
            cashflow_consumer_debt=cashflow_consumer_debt,
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
            current_time=current_time,
            population_scale_factor=population_scale_factor,
            time_unit=time_unit,
            lagged_real_consumption_budget=lagged_real_consumption_budget,
            ratio_denominator=ratio_denominator,
            epsilon=epsilon,
        )

        # ``target_total`` is the calibrated consumption budget. Cash rent is
        # removed from market purchases below; the formula-implied MPC remains
        # computed on the calibrated behavioural target.
        full_target_consumption = np.maximum(
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
        self.last_formula_implied_mpc = (
            perturbed_target_consumption.sum(axis=1) - full_target_consumption.sum(axis=1)
        ) / nominal_income_perturbation

        # Cash rent is a separate household cash use, so remove it from the
        # market-consumption demand. Imputed rent is diagnostic-only and must
        # remain behaviourally inert.
        raw_cash_rent = np.asarray(rent, dtype=float)
        raw_imputed_rent = np.asarray(rent_imputed, dtype=float)
        for name, housing_flow in (("rent", raw_cash_rent), ("rent_imputed", raw_imputed_rent)):
            if housing_flow.shape != income.shape or not np.all(np.isfinite(housing_flow)):
                raise ValueError(f"{name} must be finite and have one value per household.")
        # Preserve PR #124's existing nonnegative accounting convention for
        # malformed negative raw housing observations, then enforce the
        # economically meaningful split on the values actually routed onward.
        cash_rent = np.maximum(0.0, raw_cash_rent)
        imputed_rent = np.maximum(0.0, raw_imputed_rent)
        diagnostic_housing_component = cash_rent + imputed_rent
        market_target_total = np.maximum(0.0, target_total - cash_rent)

        target_consumption = np.maximum(
            0.0,
            1.0
            / (1 + tau_vat)
            * np.outer(
                consumption_weights,
                market_target_total,
            ).T,
        )

        # The real consumption budget is this rule's ECM state variable. The
        # caller persists it so the next period's gap term is measured against
        # the same concept the target is expressed in.
        self.last_real_consumption_budget = components["target_consumption_real_budget"]
        components["target_consumption_cash_rent"] = cash_rent
        components["target_consumption_imputed_rent"] = imputed_rent
        components["target_consumption_non_goods_housing"] = diagnostic_housing_component
        components["target_consumption_calibrated_total"] = target_total
        # Keep the historical diagnostic name as a compatibility alias for the
        # market-consumption target after the cash-rent carve-out.
        components["target_consumption_goods_total"] = market_target_total
        components["target_consumption_market_total"] = market_target_total
        self.last_target_consumption_components = components
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
        lagged_housing_wealth: np.ndarray = None,  # Ignored in exogenous consumption
        rent: np.ndarray = None,  # Ignored in exogenous consumption
        rent_imputed: np.ndarray = None,  # Ignored in exogenous consumption
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
        cashflow_consumer_debt: np.ndarray = None,  # Ignored in exogenous consumption
        lagged_house_price_index: float | np.ndarray = None,  # Ignored in exogenous consumption
        real_borrowing_rate: float | np.ndarray = None,  # Ignored in exogenous consumption
        permanent_income_log_ratio: float | np.ndarray = None,  # Ignored in exogenous consumption
        consumer_debt_rate_delta: float | np.ndarray = None,  # Ignored in exogenous consumption
        uncertainty_delta: float | np.ndarray = None,  # Ignored in exogenous consumption
        population_scale_factor: float | None = None,  # Ignored in exogenous consumption
        time_unit: int = 12,  # Ignored in exogenous consumption
        lagged_real_consumption_budget: np.ndarray = None,  # Ignored in exogenous consumption
        historic_income: np.ndarray = None,  # Ignored in exogenous consumption
        historic_deflator: np.ndarray = None,  # Ignored in exogenous consumption
        subsistence_income: np.ndarray | float | None = None,  # Ignored in exogenous consumption
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
