"""Household wealth management implementation.

This module implements household wealth management through:
- New wealth allocation
- Wealth usage decisions
- Asset value tracking
- Wealth composition updates

The implementation handles:
- Wealth distribution
- Asset depreciation
- Financial holdings
- Deposit management
"""

from abc import ABC, abstractmethod
from math import erf, exp, sqrt
from typing import Any, Optional, Tuple

import numpy as np

from macromodel.agents.households.func.portfolio_target_share import validate_target_share_source
from macromodel.timeseries import TimeSeries


class WealthSetter(ABC):
    """Abstract base class for household wealth management.

    Defines interface for managing wealth through:
    - Wealth allocation decisions
    - Asset value tracking
    - Financial holdings
    - Deposit management

    Attributes:
        other_real_assets_depreciation_rate (float): Asset depreciation rate
        independents (list[str]): Independent variables for wealth decisions
    """

    def __init__(
        self,
        other_real_assets_depreciation_rate: float,
    ):
        self.other_real_assets_depreciation_rate = other_real_assets_depreciation_rate
        self.independents = ["Income", "Debt"]

    @abstractmethod
    def distribute_new_wealth(
        self,
        new_wealth: np.ndarray,
        model: Optional[Any],
        ts: TimeSeries,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate new wealth between deposits and other assets.

        Args:
            new_wealth (np.ndarray): New wealth to allocate
            model (Optional[Any]): Allocation model
            ts (TimeSeries): Time series data

        Returns:
            Tuple[np.ndarray, np.ndarray]: New deposits and other assets
        """
        pass

    @abstractmethod
    def use_up_wealth(
        self,
        used_up_wealth: np.ndarray,
        current_wealth_in_deposits: np.ndarray,
        current_wealth_in_other_financial_assets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Determine wealth usage from deposits and other assets.

        Args:
            used_up_wealth (np.ndarray): Wealth to be used
            current_wealth_in_deposits (np.ndarray): Current deposits
            current_wealth_in_other_financial_assets (np.ndarray): Other assets

        Returns:
            Tuple[np.ndarray, np.ndarray]: Used deposits and other assets
        """
        pass

    @abstractmethod
    def compute_wealth_in_other_real_assets(
        self,
        current_wealth_in_other_real_assets: np.ndarray,
        current_investment_in_other_real_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate other real asset values.

        Args:
            current_wealth_in_other_real_assets (np.ndarray): Current assets
            current_investment_in_other_real_assets (np.ndarray): New investment

        Returns:
            np.ndarray: Updated real asset values
        """
        pass

    @abstractmethod
    def compute_wealth_in_other_financial_assets(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        new_wealth_in_other_financial_assets: np.ndarray,
        used_up_wealth_in_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate other financial asset values.

        Args:
            current_wealth_in_other_financial_assets (np.ndarray): Current assets
            new_wealth_in_other_financial_assets (np.ndarray): New assets
            used_up_wealth_in_other_financial_assets (np.ndarray): Used assets

        Returns:
            np.ndarray: Updated financial asset values
        """
        pass

    @staticmethod
    @abstractmethod
    def compute_wealth_in_deposits(
        current_wealth_in_deposits: np.ndarray,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        current_interest_paid: np.ndarray,
        price_paid_for_property: np.ndarray,
        debt_installments: np.ndarray,
        new_loans: np.ndarray,
        new_real_wealth: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate deposit values.

        Args:
            current_wealth_in_deposits (np.ndarray): Current deposits
            new_wealth_in_deposits (np.ndarray): New deposits
            used_up_wealth_in_deposits (np.ndarray): Used deposits
            current_interest_paid (np.ndarray): Interest payments
            price_paid_for_property (np.ndarray): Property purchases
            debt_installments (np.ndarray): Debt payments
            new_loans (np.ndarray): New borrowing
            new_real_wealth (np.ndarray): New real assets
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Updated deposit values
        """
        pass


class SimpleWealthSetter(WealthSetter):
    """Simple wealth management implementation.

    Implements basic wealth management through:
    - All new wealth to deposits
    - All usage from deposits
    - Basic asset tracking
    """

    def distribute_new_wealth(
        self,
        new_wealth: np.ndarray,
        model: Optional[Any],
        ts: TimeSeries,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate all new wealth to deposits.

        Args:
            new_wealth (np.ndarray): New wealth to allocate
            model (Optional[Any]): Allocation model
            ts (TimeSeries): Time series data

        Returns:
            Tuple[np.ndarray, np.ndarray]: New deposits and zero other assets
        """
        return new_wealth, np.zeros_like(new_wealth)

    def use_up_wealth(
        self,
        used_up_wealth: np.ndarray,
        current_wealth_in_deposits: np.ndarray,
        current_wealth_in_other_financial_assets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use all wealth from deposits.

        Args:
            used_up_wealth (np.ndarray): Wealth to be used
            current_wealth_in_deposits (np.ndarray): Current deposits
            current_wealth_in_other_financial_assets (np.ndarray): Other assets

        Returns:
            Tuple[np.ndarray, np.ndarray]: Used deposits and zero other assets
        """
        return used_up_wealth, np.zeros_like(used_up_wealth)

    def compute_wealth_in_other_real_assets(
        self,
        current_wealth_in_other_real_assets: np.ndarray,
        current_investment_in_other_real_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate real assets with depreciation.

        Args:
            current_wealth_in_other_real_assets (np.ndarray): Current assets
            current_investment_in_other_real_assets (np.ndarray): New investment

        Returns:
            np.ndarray: Updated real asset values
        """
        return (
            1 - self.other_real_assets_depreciation_rate
        ) * current_wealth_in_other_real_assets + current_investment_in_other_real_assets

    def compute_wealth_in_other_financial_assets(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        new_wealth_in_other_financial_assets: np.ndarray,
        used_up_wealth_in_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate financial assets with flows.

        Args:
            current_wealth_in_other_financial_assets (np.ndarray): Current assets
            new_wealth_in_other_financial_assets (np.ndarray): New assets
            used_up_wealth_in_other_financial_assets (np.ndarray): Used assets

        Returns:
            np.ndarray: Updated financial asset values
        """
        return (
            current_wealth_in_other_financial_assets
            + new_wealth_in_other_financial_assets
            - used_up_wealth_in_other_financial_assets
        )

    @staticmethod
    def compute_wealth_in_deposits(
        current_wealth_in_deposits: np.ndarray,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        current_interest_paid: np.ndarray,
        price_paid_for_property: np.ndarray,
        debt_installments: np.ndarray,
        new_loans: np.ndarray,
        new_real_wealth: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate deposits with all flows.

        Args:
            current_wealth_in_deposits (np.ndarray): Current deposits
            new_wealth_in_deposits (np.ndarray): New deposits
            used_up_wealth_in_deposits (np.ndarray): Used deposits
            current_interest_paid (np.ndarray): Interest payments
            price_paid_for_property (np.ndarray): Property purchases
            debt_installments (np.ndarray): Debt payments
            new_loans (np.ndarray): New borrowing
            new_real_wealth (np.ndarray): New real assets
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Updated deposit values
        """
        return (
            current_wealth_in_deposits
            + new_wealth_in_deposits
            - used_up_wealth_in_deposits
            - current_interest_paid
            - price_paid_for_property
            - debt_installments
            + new_loans
            - tau_cf * np.maximum(0.0, new_real_wealth)
        )


class DefaultWealthSetter(WealthSetter):
    """Default implementation of household wealth management.

    Implements wealth management through:
    - Model-based allocation
    - Priority-based usage
    - Asset tracking
    - Flow management
    """

    def distribute_new_wealth(
        self,
        new_wealth: np.ndarray,
        model: Optional[Any],
        ts: TimeSeries,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Allocate new wealth using model predictions.

        Args:
            new_wealth (np.ndarray): New wealth to allocate
            model (Optional[Any]): Allocation model
            ts (TimeSeries): Time series data

        Returns:
            Tuple[np.ndarray, np.ndarray]: New deposits and other assets
        """
        assert len(self.independents) > 0
        x = np.stack(
            [ts.current(ind.lower()) for ind in self.independents],
            axis=1,
        )
        non_zero = x.sum(axis=0) != 0.0
        x[:, non_zero] /= x.sum(axis=0)[non_zero]
        pred_deposit_fraction = model.predict(x)
        pred_deposit_fraction[pred_deposit_fraction > 1.0] = 1.0
        pred_deposit_fraction[pred_deposit_fraction < 0.0] = 0.0

        return (
            pred_deposit_fraction * new_wealth,
            (1 - pred_deposit_fraction) * new_wealth,
        )

    def use_up_wealth(
        self,
        used_up_wealth: np.ndarray,
        current_wealth_in_deposits: np.ndarray,
        current_wealth_in_other_financial_assets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use wealth with priority on other assets.

        Args:
            used_up_wealth (np.ndarray): Wealth to be used
            current_wealth_in_deposits (np.ndarray): Current deposits
            current_wealth_in_other_financial_assets (np.ndarray): Other assets

        Returns:
            Tuple[np.ndarray, np.ndarray]: Used deposits and other assets
        """
        used_up_wealth_in_other_financial_assets = np.minimum(current_wealth_in_other_financial_assets, used_up_wealth)
        used_up_wealth_in_deposits = np.minimum(
            current_wealth_in_deposits,
            used_up_wealth - used_up_wealth_in_other_financial_assets,
        )
        return (
            used_up_wealth_in_deposits,
            used_up_wealth_in_other_financial_assets,
        )

    def compute_wealth_in_other_real_assets(
        self,
        current_wealth_in_other_real_assets: np.ndarray,
        current_investment_in_other_real_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate real assets with depreciation.

        Args:
            current_wealth_in_other_real_assets (np.ndarray): Current assets
            current_investment_in_other_real_assets (np.ndarray): New investment

        Returns:
            np.ndarray: Updated real asset values
        """
        return (
            1 - self.other_real_assets_depreciation_rate
        ) * current_wealth_in_other_real_assets + current_investment_in_other_real_assets

    def compute_wealth_in_other_financial_assets(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        new_wealth_in_other_financial_assets: np.ndarray,
        used_up_wealth_in_other_financial_assets: np.ndarray,
    ) -> np.ndarray:
        """Calculate financial assets with flows.

        Args:
            current_wealth_in_other_financial_assets (np.ndarray): Current assets
            new_wealth_in_other_financial_assets (np.ndarray): New assets
            used_up_wealth_in_other_financial_assets (np.ndarray): Used assets

        Returns:
            np.ndarray: Updated financial asset values
        """
        return (
            current_wealth_in_other_financial_assets
            + new_wealth_in_other_financial_assets
            - used_up_wealth_in_other_financial_assets
        )

    @staticmethod
    def compute_wealth_in_deposits(
        current_wealth_in_deposits: np.ndarray,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        current_interest_paid: np.ndarray,
        price_paid_for_property: np.ndarray,
        debt_installments: np.ndarray,
        new_loans: np.ndarray,
        new_real_wealth: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate deposits with all flows.

        Args:
            current_wealth_in_deposits (np.ndarray): Current deposits
            new_wealth_in_deposits (np.ndarray): New deposits
            used_up_wealth_in_deposits (np.ndarray): Used deposits
            current_interest_paid (np.ndarray): Interest payments
            price_paid_for_property (np.ndarray): Property purchases
            debt_installments (np.ndarray): Debt payments
            new_loans (np.ndarray): New borrowing
            new_real_wealth (np.ndarray): New real assets
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Updated deposit values
        """
        return (
            current_wealth_in_deposits
            + new_wealth_in_deposits
            - used_up_wealth_in_deposits
            - current_interest_paid
            - price_paid_for_property
            - debt_installments
            + new_loans
            - tau_cf * np.maximum(0.0, new_real_wealth)
        )


class PaperAssetReturnWealthSetter(DefaultWealthSetter):
    """Default wealth update plus paper-style illiquid financial asset returns."""

    exclude_financial_asset_income_from_saving = True
    uses_periodic_illiquid_returns = True

    def __init__(
        self,
        other_real_assets_depreciation_rate: float,
        mu_eq: float,
        mu_bond: float,
        sigma_eq: float,
        sigma_bond: float,
        rho: float,
        equity_weight: float,
        draw_scope: str = "country_period",
        uses_portfolio_choice: bool = False,
        target_share_source: str = "scalar",
        default_target_illiquid_share: float = 0.65,
        phi_1: float = 5.0,
        lambda_kappa: float = 0.1,
        fixed_cost_share: float = 0.0,
        frm_coefficients_path: str | None = None,
        settles_portfolio_choice: bool = False,
        participation_source: str = "initial_ifa_positive",
        liquid_return_source: str = "policy_rate_markup",
        liquid_asset_policy_rate_markup: float | None = None,
        dynamic_shifters_enabled: bool = False,
        dynamic_shifters: dict | None = None,
        data_paths: dict | None = None,
        dividend_fund_payout_ratio: float = 0.05,
    ):
        super().__init__(other_real_assets_depreciation_rate=other_real_assets_depreciation_rate)
        if draw_scope != "country_period":
            raise ValueError("PaperAssetReturnWealthSetter only supports draw_scope='country_period'.")
        if sigma_eq < 0.0 or sigma_bond < 0.0:
            raise ValueError("sigma_eq and sigma_bond must be non-negative.")
        if not 0.0 <= equity_weight <= 1.0:
            raise ValueError("equity_weight must be in [0, 1].")
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must be in [-1, 1].")
        if uses_portfolio_choice and not (0.0 < lambda_kappa <= 1.0):
            raise ValueError(f"lambda_kappa must be in (0, 1] when uses_portfolio_choice=True, got {lambda_kappa}.")
        if uses_portfolio_choice and fixed_cost_share < 0.0:
            raise ValueError(f"fixed_cost_share must be non-negative, got {fixed_cost_share}.")
        if uses_portfolio_choice:
            validate_target_share_source(target_share_source)
        if settles_portfolio_choice and not uses_portfolio_choice:
            raise ValueError("settles_portfolio_choice=True requires uses_portfolio_choice=True.")
        if settles_portfolio_choice:
            # Country validates that the feasibility resolver is active because
            # the wealth setter does not own the country-level carrier.
            pass
        if dynamic_shifters_enabled:
            raise ValueError("dynamic_shifters_enabled=True is out of scope for PaperAssetReturnWealthSetter.")
        if liquid_asset_policy_rate_markup is not None:
            raise ValueError("liquid_asset_policy_rate_markup is out of scope until its policy-rate source is wired.")
        if participation_source != "initial_ifa_positive":
            raise ValueError(f"Unsupported participation_source: {participation_source}.")
        if liquid_return_source != "policy_rate_markup":
            raise ValueError(f"Unsupported liquid_return_source: {liquid_return_source}.")
        if not 0.0 <= dividend_fund_payout_ratio <= 1.0:
            raise ValueError("dividend_fund_payout_ratio must be in [0, 1].")
        self.mu_eq = mu_eq
        self.mu_bond = mu_bond
        self.sigma_eq = sigma_eq
        self.sigma_bond = sigma_bond
        self.rho = rho
        self.equity_weight = equity_weight
        self.draw_scope = draw_scope
        self.dividend_fund_payout_ratio = dividend_fund_payout_ratio
        self._current_illiquid_return_rate: float | None = None
        self._current_illiquid_return_amount: np.ndarray | None = None
        self._current_illiquid_return_period: int | None = None
        self._last_consumed_illiquid_return_period: int | None = None
        self._current_illiquid_return_consumed = True
        # Stage 4 (portfolio choice) parameters. uses_portfolio_choice gates the
        # diagnostics and optional settlement call site in Households.update_wealth();
        # it does not affect any of the asset-return behaviour above. See
        # knowledge-vault/wiki/architecture/consumption-stage-4-portfolio-choice.md.
        self.uses_portfolio_choice = uses_portfolio_choice
        self.target_share_source = target_share_source
        self.default_target_illiquid_share = default_target_illiquid_share
        self.phi_1 = phi_1
        self.lambda_kappa = lambda_kappa
        self.fixed_cost_share = fixed_cost_share
        self.frm_coefficients_path = frm_coefficients_path
        self.settles_portfolio_choice = settles_portfolio_choice

    @property
    def expected_illiquid_log_return(self) -> float:
        return self.equity_weight * self.mu_eq + (1.0 - self.equity_weight) * self.mu_bond

    @property
    def illiquid_log_return_variance(self) -> float:
        bond_weight = 1.0 - self.equity_weight
        return float(
            self.equity_weight**2 * self.sigma_eq**2
            + bond_weight**2 * self.sigma_bond**2
            + 2.0 * self.equity_weight * bond_weight * self.rho * self.sigma_eq * self.sigma_bond
        )

    @staticmethod
    def _simple_return_from_log_return(log_return: float) -> float:
        return float(np.exp(log_return) - 1.0)

    def expected_illiquid_return_rate(self) -> float:
        return self._simple_return_from_log_return(
            self.expected_illiquid_log_return + 0.5 * self.illiquid_log_return_variance
        )

    def expected_non_negative_illiquid_return_rate(self) -> float:
        """Expected positive part of the simple return implied by the log-normal draw."""
        mean = self.expected_illiquid_log_return
        variance = self.illiquid_log_return_variance
        if variance <= 0.0:
            return max(exp(mean) - 1.0, 0.0)
        sigma = sqrt(variance)

        def normal_cdf(value: float) -> float:
            return 0.5 * (1.0 + erf(value / sqrt(2.0)))

        return float(
            exp(mean + 0.5 * variance) * normal_cdf((mean + variance) / sigma)
            - normal_cdf(mean / sigma)
        )

    def draw_illiquid_return_rate(self) -> float:
        covariance = np.array(
            [
                [self.sigma_eq**2, self.rho * self.sigma_eq * self.sigma_bond],
                [self.rho * self.sigma_eq * self.sigma_bond, self.sigma_bond**2],
            ]
        )
        log_equity_return, log_bond_return = np.random.multivariate_normal(
            mean=np.array([self.mu_eq, self.mu_bond]),
            cov=covariance,
        )
        log_return = self.equity_weight * log_equity_return + (1.0 - self.equity_weight) * log_bond_return
        return self._simple_return_from_log_return(float(log_return))

    @staticmethod
    def _return_amount(current_wealth_in_other_financial_assets: np.ndarray, return_rate: float) -> np.ndarray:
        return np.maximum(current_wealth_in_other_financial_assets, 0.0) * return_rate

    def stage_illiquid_valuation_return(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        period_index: int | None = None,
    ) -> np.ndarray:
        """Draw and stage the valuation return later consumed by the wealth update."""
        if self._current_illiquid_return_amount is not None and not self._current_illiquid_return_consumed:
            if period_index is not None:
                if self._current_illiquid_return_period is None:
                    self._current_illiquid_return_period = period_index
                elif self._current_illiquid_return_period != period_index:
                    raise ValueError("Previous illiquid financial asset return has not been applied.")
            return self._current_return_amount_for(current_wealth_in_other_financial_assets)
        if period_index is not None and self._last_consumed_illiquid_return_period == period_index:
            raise ValueError("Illiquid financial asset return has already been applied for the current period.")
        return_rate = self.draw_illiquid_return_rate()
        return_amount = self._return_amount(
            current_wealth_in_other_financial_assets=current_wealth_in_other_financial_assets,
            return_rate=return_rate,
        )
        self._current_illiquid_return_rate = return_rate
        self._current_illiquid_return_amount = return_amount
        self._current_illiquid_return_period = period_index
        self._current_illiquid_return_consumed = False
        return return_amount

    def compute_income_from_financial_assets(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        period_index: int | None = None,
    ) -> np.ndarray:
        """Backward-compatible alias for the pre-fund financial-return contract."""
        return self.stage_illiquid_valuation_return(
            current_wealth_in_other_financial_assets=current_wealth_in_other_financial_assets,
            period_index=period_index,
        )

    def compute_expected_non_negative_valuation_return(
        self, current_wealth_in_other_financial_assets: np.ndarray
    ) -> np.ndarray:
        return (
            np.maximum(current_wealth_in_other_financial_assets, 0.0)
            * self.expected_non_negative_illiquid_return_rate()
        )

    def current_illiquid_return_rate(self) -> float:
        if self._current_illiquid_return_rate is None:
            return np.nan
        return self._current_illiquid_return_rate

    def current_illiquid_return_amount(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        period_index: int | None = None,
    ) -> np.ndarray:
        """Return the current-period illiquid asset return amount without consuming it."""
        return self._current_return_amount_for(
            current_wealth_in_other_financial_assets=current_wealth_in_other_financial_assets,
            period_index=period_index,
        )

    def _current_return_amount_for(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        period_index: int | None = None,
    ) -> np.ndarray:
        if self._current_illiquid_return_amount is None or self._current_illiquid_return_consumed:
            raise ValueError("Illiquid financial asset return has not been drawn for the current period.")
        if period_index is not None:
            if self._current_illiquid_return_period is None:
                self._current_illiquid_return_period = period_index
            elif self._current_illiquid_return_period != period_index:
                raise ValueError("Previous illiquid financial asset return has not been applied.")
        if self._current_illiquid_return_amount.shape != current_wealth_in_other_financial_assets.shape:
            raise ValueError("Stored illiquid return amount does not match household financial-asset shape.")
        return self._current_illiquid_return_amount

    def use_up_wealth(
        self,
        used_up_wealth: np.ndarray,
        current_wealth_in_deposits: np.ndarray,
        current_wealth_in_other_financial_assets: np.ndarray,
        period_index: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        post_return_other_financial_assets = current_wealth_in_other_financial_assets + self._current_return_amount_for(
            current_wealth_in_other_financial_assets=current_wealth_in_other_financial_assets,
            period_index=period_index,
        )
        available_other_financial_assets = np.maximum(post_return_other_financial_assets, 0.0)
        used_up_wealth_in_other_financial_assets = np.minimum(available_other_financial_assets, used_up_wealth)
        used_up_wealth_in_deposits = np.minimum(
            current_wealth_in_deposits,
            used_up_wealth - used_up_wealth_in_other_financial_assets,
        )
        return (
            used_up_wealth_in_deposits,
            used_up_wealth_in_other_financial_assets,
        )

    def compute_wealth_in_other_financial_assets(
        self,
        current_wealth_in_other_financial_assets: np.ndarray,
        new_wealth_in_other_financial_assets: np.ndarray,
        used_up_wealth_in_other_financial_assets: np.ndarray,
        period_index: int | None = None,
    ) -> np.ndarray:
        updated_wealth = (
            current_wealth_in_other_financial_assets
            + self._current_return_amount_for(
                current_wealth_in_other_financial_assets=current_wealth_in_other_financial_assets,
                period_index=period_index,
            )
            + new_wealth_in_other_financial_assets
            - used_up_wealth_in_other_financial_assets
        )
        self._last_consumed_illiquid_return_period = period_index
        self._current_illiquid_return_consumed = True
        return updated_wealth
