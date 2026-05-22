"""Reader and container for exogenous sector price data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SectorExoPrices:
    """Container for exogenous sector price trajectories.

    Holds a DataFrame of price trajectories keyed by industry name. Each column
    is a sector, the index is years, and all firms within a listed sector follow
    the same normalised trajectory at runtime.
    """

    prices: Optional[pd.DataFrame] = None
    initial_year: int = 2014
    initial_model_prices: Optional[np.ndarray] = None

    @property
    def values_dictionary(self) -> dict:
        """Dict-based access to the price DataFrame."""
        return {"prices": self.prices}

    @classmethod
    def from_reader(
        cls,
        reader: "SectorExoPricesReader",
        initial_year: int = 2014,
    ) -> "SectorExoPrices":
        """Build a SectorExoPrices container from a reader."""
        return cls(prices=reader.prices, initial_year=initial_year)


@dataclass
class SectorExoPricesReader:
    """Reader for a single exogenous sector prices CSV.

    The CSV must have years as the row index and sector or industry codes as
    column headers. Column names must match the industry codes used by the
    configured model.
    """

    prices: Optional[pd.DataFrame] = None

    @classmethod
    def read_from_raw_data(
        cls,
        prices_path: Path | str,
    ) -> "SectorExoPricesReader":
        """Load the sector prices CSV from disk."""
        if isinstance(prices_path, str):
            prices_path = Path(prices_path)
        prices = pd.read_csv(prices_path, index_col=0) if prices_path.exists() else None
        return cls(prices=prices)
