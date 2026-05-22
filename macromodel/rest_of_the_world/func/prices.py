"""Rest of the World price determination module.

This module implements approaches for determining Rest of the World
prices in international markets. It provides mechanisms for:

1. Price Setting:
   - Initial price adjustment
   - Domestic price level response
   - Dynamic price convergence

2. Price Dynamics:
   - Inflation-based updating
   - Speed of adjustment
   - Price floor enforcement

The module implements inflation-based price setting that ensures
price level convergence while maintaining positive prices.
"""

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interp1d


class RoWPriceSetter(ABC):
    """Abstract base class for Rest of World price determination.

    Provides interface for computing ROW prices based on domestic
    price levels and adjustment parameters.
    """

    @abstractmethod
    def compute_price(
        self,
        initial_price: np.ndarray,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        """Compute ROW prices.

        Args:
            initial_price (np.ndarray): Base prices
            aggregate_country_price_index (float): Domestic price level
            adjustment_speed (float): Price adjustment parameter

        Returns:
            np.ndarray: Computed ROW prices
        """
        pass


class InflationRoWPriceSetter(RoWPriceSetter):
    """Inflation-based price determination implementation.

    Adjusts prices based on domestic price level changes while
    ensuring a minimum positive price level.
    """

    def compute_price(
        self,
        initial_price: np.ndarray,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        """Compute prices using inflation adjustment.

        Adjusts initial prices based on:
        - Domestic price level deviations
        - Adjustment speed parameter
        - Minimum price floor (0.001)

        Args:
            initial_price (np.ndarray): Base prices
            aggregate_country_price_index (float): Domestic price level
            adjustment_speed (float): Price adjustment parameter

        Returns:
            np.ndarray: Adjusted prices with minimum floor
        """
        return np.maximum(
            1e-3,
            (1.0 + adjustment_speed * (aggregate_country_price_index - 1.0)) * initial_price,
        )


class SectorExogenousROWPriceSetter(InflationRoWPriceSetter):
    """ROW price setter that overrides selected sectors with exogenous price paths."""

    def __init__(self):
        self.firm_exo_prices = None
        self.overriden_industries: list[str] = []
        self._call_count: int = 0

    def _normalised_price(self, industry_name: str, current_quarter: int) -> float:
        """Interpolate the sector price and normalise to the initial year."""
        initial_year = self.firm_exo_prices.initial_year
        series = self.firm_exo_prices.prices[industry_name]
        years = series.index.astype(float).values
        prices = series.values.astype(float)
        fn = interp1d(years, prices, bounds_error=False, fill_value="extrapolate")
        year = initial_year + (current_quarter - 1) / 4
        return float(fn(year)) / float(fn(initial_year))

    def compute_price(
        self,
        initial_price: np.ndarray,
        aggregate_country_price_index: float,
        adjustment_speed: float,
    ) -> np.ndarray:
        """Compute ROW prices, overriding listed sectors with exogenous paths."""
        price = super().compute_price(
            initial_price=initial_price,
            aggregate_country_price_index=aggregate_country_price_index,
            adjustment_speed=adjustment_speed,
        )

        current_quarter = self._call_count
        self._call_count += 1

        if self.firm_exo_prices is None or self.firm_exo_prices.prices is None or len(self.overriden_industries) == 0:
            return price

        for industry_name in self.firm_exo_prices.prices.columns:
            if industry_name not in self.overriden_industries:
                continue
            ratio = self._normalised_price(industry_name, current_quarter)
            for idx in [i for i, name in enumerate(self.overriden_industries) if name == industry_name]:
                price[idx] = initial_price[idx] * ratio

        return price
