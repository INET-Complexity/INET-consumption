from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


class PriceSetter(ABC):
    """Abstract base class for determining firms' price-setting strategies.

    This class defines strategies for calculating prices based on:
    - Market conditions (supply, demand, inventories)
    - Cost factors (unit costs, inflation)
    - Competitive positioning (sector averages)
    - Adjustment speeds and noise

    The price setting process considers:
    - General inflation expectations
    - Demand-pull inflation pressures
    - Cost-push inflation pressures
    - Random price variations

    Attributes:
        price_setting_noise_std (float): Standard deviation of random
            price adjustments
        price_setting_speed_gf (float): Speed of general inflation
            pass-through (0 to 1)
        price_setting_speed_dp (float): Speed of demand-pull inflation
            adjustments (0 to 1)
        price_setting_speed_cp (float): Speed of cost-push inflation
            adjustments (0 to 1)
    """

    def __init__(
        self,
        price_setting_noise_std: float,
        price_setting_speed_gf: float,
        price_setting_speed_dp: float,
        price_setting_speed_cp: float,
    ):
        """Initialize the price setter with adjustment parameters.

        Args:
            price_setting_noise_std (float): Standard deviation of random
                price adjustments
            price_setting_speed_gf (float): Speed of general inflation
                pass-through (clipped to [0,1])
            price_setting_speed_dp (float): Speed of demand-pull inflation
                adjustments (clipped to [0,1])
            price_setting_speed_cp (float): Speed of cost-push inflation
                adjustments (clipped to [0,1])
        """
        self.price_setting_noise_std = price_setting_noise_std
        self.price_setting_speed_gf = max(0.0, min(1.0, price_setting_speed_gf))
        self.price_setting_speed_dp = max(0.0, min(1.0, price_setting_speed_dp))
        self.price_setting_speed_cp = max(0.0, min(1.0, price_setting_speed_cp))

    @abstractmethod
    def compute_price(
        self,
        prev_prices: np.ndarray,
        current_estimated_ppi_inflation: float,
        excess_demand: np.ndarray,
        inventories: np.ndarray,
        production: np.ndarray,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
        curr_unit_costs: np.ndarray,
        prev_unit_costs: np.ndarray,
        ppi_during: np.ndarray,
        current_time: int,
        prev_uc_smooth: np.ndarray | None = None,
    ) -> np.ndarray:
        """Calculate prices for each firm based on market conditions.

        Determines appropriate prices considering:
        - Previous prices and inflation expectations
        - Supply-demand balance and inventories
        - Cost changes and sector averages
        - Market positioning and competition

        Args:
            prev_prices (np.ndarray): Previous period's prices
            current_estimated_ppi_inflation (float): Expected PPI inflation
            excess_demand (np.ndarray): Excess demand by firm
            inventories (np.ndarray): Current inventory levels
            production (np.ndarray): Current production levels
            prev_average_good_prices (np.ndarray): Previous sector averages
            prev_firm_prices (np.ndarray): Previous firm-specific prices
            prev_supply (np.ndarray): Previous period's supply
            prev_demand (np.ndarray): Previous period's demand
            current_firm_sectors (np.ndarray): Sector ID for each firm
            curr_unit_costs (np.ndarray): Current unit costs
            prev_unit_costs (np.ndarray): Previous unit costs
            ppi_during (np.ndarray): PPI time series
            current_time (int): Current period index
            prev_uc_smooth (np.ndarray | None): Previous smoothed unit costs,
                used by pricing rules that smooth unit costs.

        Returns:
            np.ndarray: Updated prices by firm
        """
        pass


_MARKUP_PRICE_COMPATIBILITY_PARAMETERS = {
    "orbis_markup_path",
    "markup_year",
    "industry_to_nace_main_section",
    "markup_central_column",
    "markup_lower_column",
    "markup_upper_column",
    "unit_cost_smoothing_horizon",
    "demand_pull_speed",
    "fallback_mode",
}


class DefaultPriceSetter(PriceSetter):
    """Default implementation of price setting with multiple inflation sources.

    This class implements a strategy that adjusts prices based on:
    1. General inflation expectations
    2. Demand-pull inflation from market conditions
    3. Cost-push inflation from unit cost changes
    4. Random variations

    The approach ensures that:
    - Prices respond to market imbalances
    - Cost changes are passed through
    - Competitive positioning is maintained
    - Prices remain positive
    """

    def __init__(
        self,
        price_setting_noise_std: float = 0.05,
        price_setting_speed_gf: float = 1.0,
        price_setting_speed_dp: float = 0.0,
        price_setting_speed_cp: float = 0.0,
        **extra_parameters: object,
    ):
        unexpected_parameters = sorted(set(extra_parameters) - _MARKUP_PRICE_COMPATIBILITY_PARAMETERS)
        if unexpected_parameters:
            raise TypeError(
                f"{self.__class__.__name__} got unexpected price parameter(s): "
                f"{unexpected_parameters}"
            )

        super().__init__(
            price_setting_noise_std=price_setting_noise_std,
            price_setting_speed_gf=price_setting_speed_gf,
            price_setting_speed_dp=price_setting_speed_dp,
            price_setting_speed_cp=price_setting_speed_cp,
        )

    def compute_price(
        self,
        prev_prices: np.ndarray,
        current_estimated_ppi_inflation: float,
        excess_demand: np.ndarray,
        inventories: np.ndarray,
        production: np.ndarray,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
        curr_unit_costs: np.ndarray,
        prev_unit_costs: np.ndarray,
        ppi_during: np.ndarray,
        current_time: int,
        prev_uc_smooth: np.ndarray | None = None,
        min_inflation: float = -0.1,
        max_inflation: float = 0.1,
    ) -> np.ndarray:
        """Calculate prices using the default multi-factor strategy.

        The method:
        1. Maps sector average prices to firms
        2. Calculates demand-pull inflation based on market position
        3. Calculates cost-push inflation from unit costs
        4. Combines all factors with random noise

        Price changes are allowed when either:
        - High price (>= sector avg) and excess supply
        - Low price (< sector avg) and excess demand

        Args:
            [same as parent class]
            min_inflation (float, optional): Lower bound on inflation rates.
                Defaults to -0.1 (-10%).
            max_inflation (float, optional): Upper bound on inflation rates.
                Defaults to 0.1 (10%).

        Returns:
            np.ndarray: Updated prices by firm, guaranteed to be positive
        """
        average_price_by_firm = prev_average_good_prices[current_firm_sectors]

        # Demand-pull inflation
        demand_pull_inflation = np.zeros_like(prev_firm_prices)
        ind_canvas = np.logical_or(
            np.logical_and(
                prev_supply <= prev_demand,
                prev_firm_prices < average_price_by_firm,
            ),
            np.logical_and(
                prev_supply > prev_demand,
                prev_firm_prices >= average_price_by_firm,
            ),
        )
        demand_pull_inflation[ind_canvas] = (
            np.divide(
                prev_demand[ind_canvas],
                prev_supply[ind_canvas],
                out=np.ones_like(prev_demand[ind_canvas]),
                where=prev_supply[ind_canvas] != 0.0,
            )
            - 1.0
        )
        demand_pull_inflation = np.maximum(min_inflation, np.minimum(max_inflation, demand_pull_inflation))

        # Cost-push inflation
        cost_push_inflation = (
            np.divide(
                curr_unit_costs,
                average_price_by_firm,
                out=np.ones_like(curr_unit_costs),
                where=average_price_by_firm != 0.0,
            )
            - 1.0
        )
        cost_push_inflation = np.maximum(min_inflation, np.minimum(max_inflation, cost_push_inflation))

        return np.maximum(
            1e-2,
            prev_prices
            * (1 + np.random.normal(0.0, self.price_setting_noise_std, prev_prices.shape))
            * (1 + self.price_setting_speed_gf * current_estimated_ppi_inflation)
            * (1 + self.price_setting_speed_dp * demand_pull_inflation)
            * (1 + self.price_setting_speed_cp * cost_push_inflation),
        )


class SectorMarkupUnitCostPriceSetter(PriceSetter):
    """Set prices as sector markups over smoothed own unit costs.

    This rule intentionally does not use stochastic price noise, expected PPI
    inflation, or a separate cost-push multiplier. Unit costs enter directly
    through the markup anchor.
    """

    PRICE_FLOOR = 1e-2
    SUPPLY_EPSILON = 1e-2
    GATE_CHEAP_TIGHT = 1
    GATE_EXPENSIVE_SLACK = 2
    GATE_CHEAP_SLACK = 3
    GATE_EXPENSIVE_TIGHT = 4
    GATE_STATE_LABELS = {
        GATE_CHEAP_TIGHT: "cheap_tight",
        GATE_EXPENSIVE_SLACK: "expensive_slack",
        GATE_CHEAP_SLACK: "cheap_slack",
        GATE_EXPENSIVE_TIGHT: "expensive_tight",
    }

    def __init__(
        self,
        orbis_markup_path: str = "run_model/data/raw_data/orbis/orbis_markups_by_nace_rev_2_main_section.csv",
        markup_year: int = 2014,
        industry_to_nace_main_section: dict | None = None,
        markup_central_column: str = "mu_all_weighted_median",
        markup_lower_column: str = "mu_all_median_interval_low",
        markup_upper_column: str = "mu_all_median_interval_high",
        unit_cost_smoothing_horizon: float = 4.0,
        demand_pull_speed: float = 1.0,
        fallback_mode: str = "last_valid_then_price_then_sector_median_then_previous_price",
    ):
        if unit_cost_smoothing_horizon <= 0:
            raise ValueError("unit_cost_smoothing_horizon must be positive.")
        self.orbis_markup_path = orbis_markup_path
        self.markup_year = int(markup_year)
        mapping = industry_to_nace_main_section or self._default_industry_mapping()
        self.industry_to_nace_main_section = {str(key): value for key, value in mapping.items()}
        self.markup_central_column = markup_central_column
        self.markup_lower_column = markup_lower_column
        self.markup_upper_column = markup_upper_column
        self.unit_cost_smoothing_horizon = float(unit_cost_smoothing_horizon)
        self.unit_cost_smoothing_alpha = 2.0 / (self.unit_cost_smoothing_horizon + 1.0)
        self.demand_pull_speed = max(0.0, min(1.0, float(demand_pull_speed)))
        self.fallback_mode = fallback_mode

        self.markup_lower_by_industry, self.markup_central_by_industry, self.markup_upper_by_industry = (
            self._load_markup_arrays()
        )
        self._set_neutral_diagnostics(np.array([], dtype=float))

    @staticmethod
    def _default_industry_mapping() -> dict[str, str | list[str]]:
        return {
            "0": "A",
            "1": "B",
            "2": "C",
            "3": "D",
            "4": "E",
            "5": "F",
            "6": "G",
            "7": "H",
            "8": "I",
            "9": "J",
            "10": "K",
            "11": "L",
            "12": "M",
            "13": "N",
            "14": "O",
            "15": "P",
            "16": "Q",
            "17": ["R", "S"],
        }

    def _resolve_markup_path(self) -> Path:
        path = Path(self.orbis_markup_path).expanduser()
        if path.is_absolute():
            return path
        for parent in Path(__file__).resolve().parents:
            candidate = parent / path
            if candidate.exists():
                return candidate
        return path

    @staticmethod
    def _normalise_sections(value) -> list[str]:
        values = value if isinstance(value, (list, tuple)) else [value]
        sections = []
        for section in values:
            if section is None:
                raise ValueError("NACE main-section mapping cannot contain null values.")
            code = str(section).strip()
            if not code:
                raise ValueError("NACE main-section mapping cannot contain empty values.")
            sections.append(code[0].upper())
        return sections

    def _load_markup_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        path = self._resolve_markup_path()
        if not path.exists():
            raise ValueError(f"Orbis markup file not found: {path}")
        data = pd.read_csv(path)
        required_columns = {
            "year",
            "nace_rev_2_main_section",
            "sum_weights",
            self.markup_lower_column,
            self.markup_central_column,
            self.markup_upper_column,
        }
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            raise ValueError(f"Orbis markup file is missing columns: {sorted(missing_columns)}")

        data = data.loc[data["year"] == self.markup_year].copy()
        if data.empty:
            raise ValueError(f"No Orbis markup data for markup_year={self.markup_year}.")
        data["section_code"] = data["nace_rev_2_main_section"].astype(str).str.strip().str[0].str.upper()
        rows_by_section = {row["section_code"]: row for _, row in data.iterrows()}

        industry_keys = sorted(int(key) for key in self.industry_to_nace_main_section)
        if industry_keys != list(range(len(industry_keys))):
            raise ValueError("industry_to_nace_main_section keys must be contiguous integer industry indices from 0.")

        lower = np.zeros(len(industry_keys), dtype=float)
        central = np.zeros(len(industry_keys), dtype=float)
        upper = np.zeros(len(industry_keys), dtype=float)
        for industry in industry_keys:
            sections = self._normalise_sections(self.industry_to_nace_main_section[str(industry)])
            missing_sections = [section for section in sections if section not in rows_by_section]
            if missing_sections:
                raise ValueError(f"Missing Orbis markup rows for NACE sections {missing_sections}.")
            rows = [rows_by_section[section] for section in sections]
            weights = np.array([row["sum_weights"] for row in rows], dtype=float)
            if not np.all(np.isfinite(weights)) or weights.sum() <= 0.0:
                weights = np.ones(len(rows), dtype=float)
            lower[industry] = np.average([row[self.markup_lower_column] for row in rows], weights=weights)
            central[industry] = np.average([row[self.markup_central_column] for row in rows], weights=weights)
            upper[industry] = np.average([row[self.markup_upper_column] for row in rows], weights=weights)

        if not (
            np.all(np.isfinite(lower))
            and np.all(np.isfinite(central))
            and np.all(np.isfinite(upper))
            and np.all(lower > 0.0)
            and np.all(central > 0.0)
            and np.all(upper > 0.0)
        ):
            raise ValueError("Markup corridor values must be finite and strictly positive.")
        if not np.all((lower <= central) & (central <= upper)):
            raise ValueError("Markup corridor must satisfy markup_lower <= markup_central <= markup_upper.")
        return lower, central, upper

    def _set_neutral_diagnostics(self, prev_prices: np.ndarray) -> None:
        self.last_pricing_uc_smooth = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_target_markup = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_realized_markup = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_markup_lower = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_markup_upper = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_gate_state = np.zeros(prev_prices.shape, dtype=float)

    def _sector_median_unit_cost(self, unit_costs: np.ndarray, current_firm_sectors: np.ndarray) -> np.ndarray:
        medians = np.full(unit_costs.shape, np.nan, dtype=float)
        valid = np.isfinite(unit_costs) & (unit_costs > 0.0)
        for sector in np.unique(current_firm_sectors):
            sector_valid = valid & (current_firm_sectors == sector)
            if np.any(sector_valid):
                medians[current_firm_sectors == sector] = np.median(unit_costs[sector_valid])
        return medians

    def _compute_uc_smooth(
        self,
        curr_unit_costs: np.ndarray,
        prev_unit_costs: np.ndarray,
        prev_uc_smooth: np.ndarray | None,
        prev_prices: np.ndarray,
        current_firm_sectors: np.ndarray,
        markup_central: np.ndarray,
        current_time: int,
    ) -> np.ndarray:
        current_valid = np.isfinite(curr_unit_costs) & (curr_unit_costs > 0.0)
        previous_raw_valid = np.isfinite(prev_unit_costs) & (prev_unit_costs > 0.0)
        if prev_uc_smooth is None or prev_uc_smooth.shape != curr_unit_costs.shape or current_time <= 1:
            previous_valid_smooth = np.full(curr_unit_costs.shape, np.nan, dtype=float)
        else:
            previous_valid_smooth = np.asarray(prev_uc_smooth, dtype=float).copy()
            previous_valid_smooth[~(np.isfinite(previous_valid_smooth) & (previous_valid_smooth > 0.0))] = np.nan

        inferred_from_price = np.divide(
            prev_prices,
            markup_central,
            out=np.full_like(prev_prices, np.nan, dtype=float),
            where=markup_central > 0.0,
        )
        inferred_from_price[~(np.isfinite(inferred_from_price) & (inferred_from_price > 0.0))] = np.nan

        # Raw model unit costs and output prices are not guaranteed to share an
        # initial level. Convert current raw unit costs into the price-implied
        # unit-cost scale, then smooth the scaled series.
        unit_cost_scale = np.divide(
            previous_valid_smooth,
            prev_unit_costs,
            out=np.full_like(curr_unit_costs, np.nan, dtype=float),
            where=previous_raw_valid,
        )
        price_implied_scale = np.divide(
            inferred_from_price,
            curr_unit_costs,
            out=np.full_like(curr_unit_costs, np.nan, dtype=float),
            where=current_valid,
        )
        unit_cost_scale = np.where(
            np.isfinite(unit_cost_scale) & (unit_cost_scale > 0.0), unit_cost_scale, price_implied_scale
        )
        scaled_current_uc = curr_unit_costs * unit_cost_scale
        scaled_current_uc[~(np.isfinite(scaled_current_uc) & (scaled_current_uc > 0.0))] = np.nan
        sector_median = self._sector_median_unit_cost(scaled_current_uc, current_firm_sectors)

        prior_for_smoothing = np.where(
            np.isfinite(previous_valid_smooth) & (previous_valid_smooth > 0.0),
            previous_valid_smooth,
            inferred_from_price,
        )
        uc_smooth = np.where(
            np.isfinite(scaled_current_uc) & (scaled_current_uc > 0.0),
            self.unit_cost_smoothing_alpha * scaled_current_uc
            + (1.0 - self.unit_cost_smoothing_alpha) * prior_for_smoothing,
            np.nan,
        )
        uc_smooth = np.where(np.isfinite(uc_smooth) & (uc_smooth > 0.0), uc_smooth, previous_valid_smooth)
        uc_smooth = np.where(np.isfinite(uc_smooth) & (uc_smooth > 0.0), uc_smooth, inferred_from_price)
        uc_smooth = np.where(np.isfinite(uc_smooth) & (uc_smooth > 0.0), uc_smooth, sector_median)
        return uc_smooth

    def compute_price(
        self,
        prev_prices: np.ndarray,
        current_estimated_ppi_inflation: float,
        excess_demand: np.ndarray,
        inventories: np.ndarray,
        production: np.ndarray,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
        curr_unit_costs: np.ndarray,
        prev_unit_costs: np.ndarray,
        ppi_during: np.ndarray,
        current_time: int,
        prev_uc_smooth: np.ndarray | None = None,
        min_inflation: float = -1.0,
        max_inflation: float = 1.0,
    ) -> np.ndarray:
        """Calculate prices from markup corridors and smoothed own unit costs."""
        self._set_neutral_diagnostics(prev_prices)
        current_firm_sectors = current_firm_sectors.astype(int)
        if np.any(current_firm_sectors < 0) or np.any(current_firm_sectors >= len(self.markup_central_by_industry)):
            raise ValueError("current_firm_sectors contains sectors not covered by markup configuration.")

        markup_lower = self.markup_lower_by_industry[current_firm_sectors]
        markup_central = self.markup_central_by_industry[current_firm_sectors]
        markup_upper = self.markup_upper_by_industry[current_firm_sectors]
        uc_smooth = self._compute_uc_smooth(
            curr_unit_costs=curr_unit_costs,
            prev_unit_costs=prev_unit_costs,
            prev_uc_smooth=prev_uc_smooth,
            prev_prices=prev_prices,
            current_firm_sectors=current_firm_sectors,
            markup_central=markup_central,
            current_time=current_time,
        )

        average_price_by_firm = prev_average_good_prices[current_firm_sectors]
        cheap = prev_firm_prices < average_price_by_firm
        tight = prev_supply <= prev_demand
        cheap_tight = cheap & tight
        expensive_slack = (~cheap) & (~tight)
        cheap_slack = cheap & (~tight)
        expensive_tight = (~cheap) & tight

        gate_state = np.zeros(prev_prices.shape, dtype=float)
        gate_state[cheap_tight] = self.GATE_CHEAP_TIGHT
        gate_state[expensive_slack] = self.GATE_EXPENSIVE_SLACK
        gate_state[cheap_slack] = self.GATE_CHEAP_SLACK
        gate_state[expensive_tight] = self.GATE_EXPENSIVE_TIGHT

        demand_pull = np.zeros(prev_prices.shape, dtype=float)
        active_gate = cheap_tight | expensive_slack
        demand_pull[active_gate] = (
            np.divide(
                prev_demand[active_gate],
                np.maximum(prev_supply[active_gate], self.SUPPLY_EPSILON),
                out=np.ones_like(prev_demand[active_gate], dtype=float),
            )
            - 1.0
        )
        demand_pull = np.clip(demand_pull, min_inflation, max_inflation)

        target_markup = markup_central.copy()
        positive = demand_pull > 0.0
        negative = demand_pull < 0.0
        target_markup[positive] = markup_central[positive] + self.demand_pull_speed * demand_pull[positive] * (
            markup_upper[positive] - markup_central[positive]
        )
        target_markup[negative] = markup_central[negative] + self.demand_pull_speed * demand_pull[negative] * (
            markup_central[negative] - markup_lower[negative]
        )
        target_markup = np.clip(target_markup, markup_lower, markup_upper)

        computed_prices = target_markup * uc_smooth
        invalid_price = ~(np.isfinite(computed_prices) & (computed_prices > 0.0))
        prices = np.where(invalid_price, prev_prices, np.maximum(self.PRICE_FLOOR, computed_prices))
        realized_markup = np.divide(
            prices,
            uc_smooth,
            out=np.full_like(prices, np.nan, dtype=float),
            where=np.isfinite(uc_smooth) & (uc_smooth > 0.0),
        )

        self.last_pricing_uc_smooth = uc_smooth
        self.last_pricing_target_markup = target_markup
        self.last_pricing_realized_markup = realized_markup
        self.last_pricing_markup_lower = markup_lower
        self.last_pricing_markup_upper = markup_upper
        self.last_pricing_gate_state = gate_state
        return prices


class SectorExogenousPriceSetter(DefaultPriceSetter):
    """Price setter that overrides selected sectors with exogenous price paths."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.firm_exo_prices = None
        self.overriden_industries: list[str] = []

    def _indices_for(self, industry_name: str) -> list[int]:
        """Return all firm array indices belonging to the given sector."""
        return [i for i, name in enumerate(self.overriden_industries) if name == industry_name]

    def _normalised_price(self, industry_name: str, current_quarter: int) -> float:
        """Interpolate the sector price and normalise to the initial year."""
        initial_year = self.firm_exo_prices.initial_year
        series = self.firm_exo_prices.prices[industry_name]
        years = series.index.astype(float).values
        prices = series.values.astype(float)
        fn = interp1d(years, prices)
        year = initial_year + (current_quarter - 1) / 4
        return float(fn(year)) / float(fn(initial_year))

    def compute_price(
        self,
        prev_prices: np.ndarray,
        current_estimated_ppi_inflation: float,
        excess_demand: np.ndarray,
        inventories: np.ndarray,
        production: np.ndarray,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
        curr_unit_costs: np.ndarray,
        prev_unit_costs: np.ndarray,
        ppi_during: np.ndarray,
        current_time: int,
        prev_uc_smooth: np.ndarray | None = None,
        min_inflation: float = -0.1,
        max_inflation: float = 0.1,
    ) -> np.ndarray:
        """Compute prices, overriding listed sectors with exogenous paths."""
        price = super().compute_price(
            prev_prices=prev_prices,
            current_estimated_ppi_inflation=current_estimated_ppi_inflation,
            excess_demand=excess_demand,
            inventories=inventories,
            production=production,
            prev_average_good_prices=prev_average_good_prices,
            prev_firm_prices=prev_firm_prices,
            prev_supply=prev_supply,
            prev_demand=prev_demand,
            current_firm_sectors=current_firm_sectors,
            curr_unit_costs=curr_unit_costs,
            prev_unit_costs=prev_unit_costs,
            ppi_during=ppi_during,
            current_time=current_time,
            prev_uc_smooth=prev_uc_smooth,
            min_inflation=min_inflation,
            max_inflation=max_inflation,
        )

        if self.firm_exo_prices is None or self.firm_exo_prices.prices is None or len(self.overriden_industries) == 0:
            return price

        if self.firm_exo_prices.initial_model_prices is None:
            base_prices = prev_average_good_prices[current_firm_sectors]
        else:
            initial_model_prices = self.firm_exo_prices.initial_model_prices
            base_prices = (
                initial_model_prices
                if initial_model_prices.shape == price.shape
                else initial_model_prices[current_firm_sectors]
            )

        for industry_name in self.firm_exo_prices.prices.columns:
            if industry_name not in self.overriden_industries:
                continue
            ratio = self._normalised_price(industry_name, current_quarter=current_time)
            for idx in self._indices_for(industry_name):
                price[idx] = base_prices[idx] * ratio

        return price


class ExogenousPriceSetter(PriceSetter):
    """Implementation of price setting using exogenous price paths.

    This class implements a simplified strategy where:
    - Prices follow a pre-determined path
    - Market conditions are ignored
    - Cost changes are ignored
    - No random variations are added

    This approach is useful for:
    - Model testing and validation
    - Policy analysis with controlled prices
    - Scenarios with external price determination
    """

    def compute_price(
        self,
        prev_prices: np.ndarray,
        current_estimated_ppi_inflation: float,
        excess_demand: np.ndarray,
        inventories: np.ndarray,
        production: np.ndarray,
        prev_average_good_prices: np.ndarray,
        prev_firm_prices: np.ndarray,
        prev_supply: np.ndarray,
        prev_demand: np.ndarray,
        current_firm_sectors: np.ndarray,
        curr_unit_costs: np.ndarray,
        prev_unit_costs: np.ndarray,
        ppi_during: np.ndarray,
        current_time: int,
        prev_uc_smooth: np.ndarray | None = None,
        min_inflation: float = -0.1,
        max_inflation: float = 0.1,
    ) -> np.ndarray:
        """Set prices according to exogenous PPI path.

        Simply returns the pre-determined PPI value for the current period,
        ignoring all market conditions and other parameters.

        Args:
            [same as parent class, all unused except:]
            ppi_during (np.ndarray): PPI time series
            current_time (int): Current period index
            min_inflation (float, optional): Unused. Defaults to -0.1.
            max_inflation (float, optional): Unused. Defaults to 0.1.

        Returns:
            np.ndarray: Price level from exogenous PPI path
        """
        return ppi_during[current_time]
