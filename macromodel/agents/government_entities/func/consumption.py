"""Government consumption determination strategies.

This module implements various approaches for determining government
consumption targets, including:
- Autoregressive forecasting
- Constant growth assumptions
- Exogenous consumption paths

The consumption strategies consider:
- Historical consumption patterns
- Price level adjustments
- Growth expectations
- Inflation expectations
- Financial constraints
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from macromodel.forecaster.forecaster import (
    ImplementedAutoregForecaster,  # noqa
    ManualAutoregForecaster,
)

GOVERNMENT_SECTORAL_WEIGHTS = (
    "previous_desired",
    "initial",
    "initial_price_normalized",
    "initial_fixed",
)


def _normalise_government_consumption_weights(previous_desired_government_consumption: np.ndarray) -> np.ndarray:
    """Return stable industry weights for government consumption targets."""
    weights = np.asarray(previous_desired_government_consumption, dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    weights_sum = weights.sum()
    if weights_sum <= 0.0:
        return np.full(weights.shape, 1.0 / len(weights))
    return weights / weights_sum


class GovernmentConsumptionSetter(ABC):
    """Abstract base class for government consumption strategies.

    This class defines the interface for determining government
    consumption targets based on various factors including:
    - Historical consumption patterns
    - Economic conditions
    - Price level changes
    - Growth expectations
    - Policy objectives

    The consumption setting process considers:
    - Consistency requirements
    - Default growth assumptions
    - Buffer periods for forecasting
    - Price level adjustments
    """

    def __init__(
        self,
        consistency: float,
        default_growth: Optional[float] = None,
        sectoral_weights: str = "previous_desired",
    ):
        """Initialize consumption setter.

        Args:
            consistency (float): Must be 0.0 or 1.0, determines whether
                to use consistent forecasting (1.0) or period-by-period
                adjustments (0.0)
            default_growth (float, optional): Default growth rate to use
                when historical data is unavailable
            sectoral_weights (str): Industry allocation mode. This can be
                supplied from model configuration next to ``consistency``.
        """
        assert consistency == 1.0 or consistency == 0.0
        if sectoral_weights not in GOVERNMENT_SECTORAL_WEIGHTS:
            raise ValueError(
                f"{self.__class__.__name__} sectoral_weights must be 'previous_desired', "
                "'initial', 'initial_price_normalized', or 'initial_fixed'."
            )
        self.consistency = consistency
        self.default_growth = default_growth
        self.sectoral_weights = sectoral_weights
        self.initial_government_consumption_weights = None
        self.fixed_total_government_consumption = None
        self.buffer = 20

    def _consumption_weights(self, previous_desired_government_consumption: np.ndarray) -> np.ndarray:
        if self.sectoral_weights in {"initial", "initial_price_normalized", "initial_fixed"}:
            if self.initial_government_consumption_weights is None:
                self.initial_government_consumption_weights = _normalise_government_consumption_weights(
                    previous_desired_government_consumption
                )
            return self.initial_government_consumption_weights
        return _normalise_government_consumption_weights(previous_desired_government_consumption)

    def _allocate_consumption(
        self,
        *,
        real_total_consumption: float,
        previous_desired_government_consumption: np.ndarray,
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> np.ndarray:
        """Allocate real aggregate consumption into nominal sectoral targets."""
        consumption_weights = self._consumption_weights(previous_desired_government_consumption)
        price_ratio = current_good_prices / initial_good_prices
        if self.sectoral_weights == "initial_fixed":
            return np.maximum(0.0, (1 + expected_inflation) * real_total_consumption * consumption_weights)
        if self.sectoral_weights == "initial_price_normalized":
            price_adjusted_weights = _normalise_government_consumption_weights(price_ratio * consumption_weights)
            return np.maximum(0.0, (1 + expected_inflation) * real_total_consumption * price_adjusted_weights)
        return np.maximum(
            0.0,
            (1 + expected_inflation) * price_ratio * real_total_consumption * consumption_weights,
        )

    @abstractmethod
    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: np.ndarray,
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        """Calculate target government consumption.

        Args:
            previous_desired_government_consumption (np.ndarray):
                Previous period's consumption targets
            model (Any, optional): Model for consumption forecasting
            historic_total_consumption (np.ndarray): Historical total
                consumption values
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            current_time (int): Current time period
            exogenous_total_consumption (np.ndarray, optional):
                Pre-specified consumption path
            forecasting_window (int): Window for consumption forecasting
            assume_zero_noise (bool, optional): Whether to assume
                deterministic consumption paths

        Returns:
            np.ndarray: Target consumption by industry
        """
        pass


class AutoregressiveGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    """Autoregressive consumption target determination.

    This class implements consumption targeting based on:
    - Autoregressive forecasting of total consumption
    - Price level adjustments
    - Consistency requirements
    - Industry-specific allocation

    The approach provides:
    - Data-driven consumption targets
    - Price-adjusted spending
    - Consistent or period-by-period forecasting
    - Industry-level detail
    """

    def __init__(
        self,
        consistency: float,
        default_growth: Optional[float] = None,
        sectoral_weights: str = "previous_desired",
    ):
        super().__init__(
            consistency=consistency,
            default_growth=default_growth,
            sectoral_weights=sectoral_weights,
        )

    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: np.ndarray,
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
        log_it: bool = True,
    ) -> np.ndarray:
        """Calculate consumption targets using autoregression.

        Uses autoregressive forecasting to determine targets based on:
        - Historical consumption patterns
        - Price level changes
        - Consistency requirements
        - Industry-specific shares

        Args:
            previous_desired_government_consumption (np.ndarray):
                Previous period's consumption targets
            model (Any, optional): Model for consumption forecasting
            historic_total_consumption (np.ndarray): Historical total
                consumption values
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            current_time (int): Current time period
            exogenous_total_consumption (np.ndarray, optional):
                Pre-specified consumption path
            forecasting_window (int): Window for consumption forecasting
            assume_zero_noise (bool, optional): Whether to assume
                deterministic consumption paths
            log_it (bool, optional): Whether to use log transformation
                in forecasting

        Returns:
            np.ndarray: Target consumption by industry
        """
        if historic_total_consumption[-1] == 0.0:
            return np.zeros(previous_desired_government_consumption.shape)

        # Fitting based on target consumption
        if self.consistency == 1.0:
            if (
                self.fixed_total_government_consumption is None
                or len(self.fixed_total_government_consumption) < current_time
            ):
                if log_it:
                    self.fixed_total_government_consumption = np.exp(
                        ManualAutoregForecaster().forecast(
                            data=np.log(historic_total_consumption),
                            t=max(current_time + self.buffer, current_time),
                            assume_zero_noise=assume_zero_noise,
                        )
                    )
                else:
                    self.fixed_total_government_consumption = ManualAutoregForecaster().forecast(
                        data=historic_total_consumption,
                        t=max(current_time + self.buffer, current_time),
                        assume_zero_noise=assume_zero_noise,
                    )
            real_total_consumption = self.fixed_total_government_consumption[current_time - 1]

        # Fitting based on historic consumption
        else:
            real_total_consumption = np.exp(
                ManualAutoregForecaster().forecast(
                    data=np.log(historic_total_consumption),
                    t=1,
                    assume_zero_noise=assume_zero_noise,
                )[0]
            )

        return self._allocate_consumption(
            real_total_consumption=real_total_consumption,
            previous_desired_government_consumption=previous_desired_government_consumption,
            initial_good_prices=initial_good_prices,
            current_good_prices=current_good_prices,
            expected_inflation=expected_inflation,
        )


class ConstantGrowthGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    """Constant growth consumption target determination.

    This class implements consumption targeting based on:
    - Fixed growth rate assumptions
    - Price level adjustments
    - Historical growth estimation
    - Industry-specific allocation

    The approach provides:
    - Simple growth-based targets
    - Price-adjusted spending
    - Stable consumption paths
    - Industry-level detail
    """

    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: Optional[np.ndarray],
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        """Calculate consumption targets using constant growth.

        Determines targets based on:
        - Fixed or estimated growth rate
        - Price level changes
        - Previous consumption levels
        - Industry-specific shares

        Args:
            previous_desired_government_consumption (np.ndarray):
                Previous period's consumption targets
            model (Any, optional): Model for consumption forecasting
            historic_total_consumption (np.ndarray, optional): Historical
                total consumption values
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            current_time (int): Current time period
            exogenous_total_consumption (np.ndarray, optional):
                Pre-specified consumption path
            forecasting_window (int): Window for consumption forecasting
            assume_zero_noise (bool, optional): Whether to assume
                deterministic consumption paths

        Returns:
            np.ndarray: Target consumption by industry
        """
        if historic_total_consumption is None:
            if self.default_growth is None:
                raise ValueError(
                    "ConstantGrowthGovernmentConsumptionSetter requires either "
                    "historic_total_consumption or a configured default_growth."
                )
            growth_factor = 1 + self.default_growth
        elif self.default_growth is None:
            estimated_log_growth = np.mean(
                np.log(
                    historic_total_consumption[1 : -current_time - 1]
                    / historic_total_consumption[0 : -current_time - 2]
                )
            )
            self.default_growth = np.exp(estimated_log_growth) - 1
            growth_factor = 1 + self.default_growth
        else:
            growth_factor = 1 + self.default_growth

        if self.sectoral_weights in {"initial", "initial_price_normalized", "initial_fixed"}:
            real_total_consumption = previous_desired_government_consumption.sum() * growth_factor
            return self._allocate_consumption(
                real_total_consumption=real_total_consumption,
                previous_desired_government_consumption=previous_desired_government_consumption,
                initial_good_prices=initial_good_prices,
                current_good_prices=current_good_prices,
                expected_inflation=expected_inflation,
            )

        return np.maximum(
            0.0,
            (1 + expected_inflation)
            * current_good_prices
            / initial_good_prices
            * growth_factor
            * previous_desired_government_consumption,
        )


class ExpectedGrowthGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    """Expected-output-growth consumption target determination.

    This rule updates government consumption using the model's expected output
    growth and expected inflation instead of a government-specific exogenous
    trend. Sectoral allocation follows ``sectoral_weights``.
    """

    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: Optional[np.ndarray],
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        """Calculate consumption targets using expected output growth.

        The aggregate real government consumption target is:

            G_t = (1 + expected_growth) * sum_i G_{i,t-1}

        ``_allocate_consumption`` then applies expected inflation and the
        configured sectoral allocation mode.
        """
        real_total_consumption = previous_desired_government_consumption.sum() * (1 + expected_growth)
        return self._allocate_consumption(
            real_total_consumption=real_total_consumption,
            previous_desired_government_consumption=previous_desired_government_consumption,
            initial_good_prices=initial_good_prices,
            current_good_prices=current_good_prices,
            expected_inflation=expected_inflation,
        )


class AutoregressiveGrowthGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    """Autoregressive growth consumption target determination.

    This class implements consumption targeting based on:
    - Autoregressive forecasting of growth rates
    - Price level adjustments
    - Consistency requirements
    - Industry-specific allocation

    The approach provides:
    - Data-driven growth forecasts
    - Price-adjusted spending
    - Consistent or period-by-period forecasting
    - Industry-level detail
    """

    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: np.ndarray,
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
        log_it: bool = False,
    ) -> np.ndarray:
        """Calculate consumption targets using growth autoregression.

        Uses autoregressive forecasting of growth rates to determine
        targets based on:
        - Historical growth patterns
        - Price level changes
        - Consistency requirements
        - Industry-specific shares

        Args:
            previous_desired_government_consumption (np.ndarray):
                Previous period's consumption targets
            model (Any, optional): Model for consumption forecasting
            historic_total_consumption (np.ndarray): Historical total
                consumption values
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            current_time (int): Current time period
            exogenous_total_consumption (np.ndarray, optional):
                Pre-specified consumption path
            forecasting_window (int): Window for consumption forecasting
            assume_zero_noise (bool, optional): Whether to assume
                deterministic consumption paths
            log_it (bool, optional): Whether to use log transformation
                in forecasting

        Returns:
            np.ndarray: Target consumption by industry
        """
        if historic_total_consumption[-1] == 0.0:
            return np.zeros(previous_desired_government_consumption.shape)

        # Fitting based on target consumption
        if self.consistency == 1.0:
            if (
                self.fixed_total_government_consumption is None
                or len(self.fixed_total_government_consumption) < current_time
            ):
                historic_total_consumption_growth = (
                    historic_total_consumption[1:] / historic_total_consumption[:-1] - 1.0
                )
                self.fixed_total_government_consumption = (
                    np.exp(
                        ManualAutoregForecaster().forecast(
                            data=historic_total_consumption_growth,
                            t=max(current_time + self.buffer, current_time),
                            assume_zero_noise=assume_zero_noise,
                        )
                    )
                    - 1
                )
                self.fixed_total_government_consumption = (
                    np.cumprod(1 + self.fixed_total_government_consumption) * historic_total_consumption[-1]
                )

            real_total_consumption = self.fixed_total_government_consumption[current_time - 1]

        # Fitting based on historic consumption
        else:
            real_total_consumption = np.exp(
                ManualAutoregForecaster().forecast(
                    data=np.log(historic_total_consumption),
                    t=1,
                    assume_zero_noise=assume_zero_noise,
                )[0]
            )

        return self._allocate_consumption(
            real_total_consumption=real_total_consumption,
            previous_desired_government_consumption=previous_desired_government_consumption,
            initial_good_prices=initial_good_prices,
            current_good_prices=current_good_prices,
            expected_inflation=expected_inflation,
        )


class ExogenousGovernmentConsumptionSetter(GovernmentConsumptionSetter):
    """Exogenous consumption target determination.

    This class implements consumption targeting based on:
    - Pre-specified consumption paths
    - Price level adjustments
    - Default growth fallback
    - Industry-specific allocation

    The approach provides:
    - Externally determined targets
    - Price-adjusted spending
    - Fallback to growth-based targets
    - Industry-level detail
    """

    def compute_target_consumption(
        self,
        previous_desired_government_consumption: np.ndarray,
        model: Optional[Any],
        historic_total_consumption: Optional[np.ndarray],
        initial_good_prices: np.ndarray,
        current_good_prices: np.ndarray,
        expected_growth: float,
        expected_inflation: float,
        current_time: int,
        exogenous_total_consumption: Optional[np.ndarray],
        forecasting_window: int,
        assume_zero_noise: bool = False,
    ) -> np.ndarray:
        """Calculate consumption targets using exogenous path.

        Determines targets based on:
        - Pre-specified consumption values
        - Price level changes
        - Default growth fallback
        - Industry-specific shares

        Args:
            previous_desired_government_consumption (np.ndarray):
                Previous period's consumption targets
            model (Any, optional): Model for consumption forecasting
            historic_total_consumption (np.ndarray, optional): Historical
                total consumption values
            initial_good_prices (np.ndarray): Initial price levels
            current_good_prices (np.ndarray): Current price levels
            expected_growth (float): Expected economic growth rate
            expected_inflation (float): Expected inflation rate
            current_time (int): Current time period
            exogenous_total_consumption (np.ndarray, optional):
                Pre-specified consumption path
            forecasting_window (int): Window for consumption forecasting
            assume_zero_noise (bool, optional): Whether to assume
                deterministic consumption paths

        Returns:
            np.ndarray: Target consumption by industry

        Raises:
            ValueError: If exogenous data not available for current time
        """
        if exogenous_total_consumption is None:
            return np.maximum(
                0.0,
                (1 + expected_inflation)
                * current_good_prices
                / initial_good_prices
                * (1 + self.default_growth)
                * previous_desired_government_consumption,
            )
        if current_time >= len(exogenous_total_consumption):
            raise ValueError("No exogenous data available beyond this point.")
        if not np.isfinite(exogenous_total_consumption[current_time]):
            raise ValueError(
                "ExogenousGovernmentConsumptionSetter: non-finite exogenous government consumption with "
                f"current_time={current_time}, value={exogenous_total_consumption[current_time]}"
            )
        return self._allocate_consumption(
            real_total_consumption=exogenous_total_consumption[current_time],
            previous_desired_government_consumption=previous_desired_government_consumption,
            initial_good_prices=initial_good_prices,
            current_good_prices=current_good_prices,
            expected_inflation=expected_inflation,
        )
