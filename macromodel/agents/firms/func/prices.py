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
        pricing_material_mc: np.ndarray | None = None,
        pricing_effective_labour_inputs: np.ndarray | None = None,
        pricing_normal_output: np.ndarray | None = None,
        pricing_depreciation_unit_cost: np.ndarray | None = None,
        wage_obligation_preview: np.ndarray | None = None,
        producer_tax_rates: np.ndarray | None = None,
        prev_mc_smooth: np.ndarray | None = None,
        prev_ac_smooth: np.ndarray | None = None,
        prev_normal_output: np.ndarray | None = None,
        initial_output_weights: np.ndarray | None = None,
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
            pricing_material_mc (np.ndarray | None): Current technical material
                marginal cost from firm technology and input prices.
            pricing_effective_labour_inputs (np.ndarray | None): Current
                effective labour capacity used to convert wage obligations into
                labour marginal cost.
            pricing_normal_output (np.ndarray | None): Current normal-output
                candidate for pricing-cost allocation.
            pricing_depreciation_unit_cost (np.ndarray | None): Current normal
                depreciation unit cost for the AC floor.
            wage_obligation_preview (np.ndarray | None): Current wage-obligation
                preview used for the labour MC proxy.
            producer_tax_rates (np.ndarray | None): Producer tax/subsidy rates
                by firm.
            prev_mc_smooth (np.ndarray | None): Previous smoothed pricing MC.
            prev_ac_smooth (np.ndarray | None): Previous smoothed pricing AC.
            prev_normal_output (np.ndarray | None): Previous smoothed normal
                output used by pricing.
            initial_output_weights (np.ndarray | None): Initial real output
                weights used by optional first-period cost-level alignment.

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
    "markup_corridor_relative_cap",
    "mc_smoothing_horizon",
    "ac_smoothing_horizon",
    "normal_output_smoothing_horizon",
    "normal_output_capital_floor_lambda",
    "demand_pull_speed",
    "ac_floor_share",
    "initial_cost_normalization_mode",
    "initial_cost_normalization_lower_quantile",
    "initial_cost_normalization_upper_quantile",
    "initial_cost_normalization_min_factor",
    "initial_cost_normalization_max_factor",
    "initial_cost_normalization_min_valid_weight_share",
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
            raise TypeError(f"{self.__class__.__name__} got unexpected price parameter(s): {unexpected_parameters}")

        super().__init__(
            price_setting_noise_std=price_setting_noise_std,
            price_setting_speed_gf=price_setting_speed_gf,
            price_setting_speed_dp=price_setting_speed_dp,
            price_setting_speed_cp=price_setting_speed_cp,
        )

    def update_parameters_from_config(self, parameters: dict) -> None:
        DefaultPriceSetter.__init__(self, **parameters)

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
        pricing_material_mc: np.ndarray | None = None,
        pricing_effective_labour_inputs: np.ndarray | None = None,
        pricing_normal_output: np.ndarray | None = None,
        pricing_depreciation_unit_cost: np.ndarray | None = None,
        wage_obligation_preview: np.ndarray | None = None,
        producer_tax_rates: np.ndarray | None = None,
        prev_mc_smooth: np.ndarray | None = None,
        prev_ac_smooth: np.ndarray | None = None,
        prev_normal_output: np.ndarray | None = None,
        initial_output_weights: np.ndarray | None = None,
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


class SectorMarkupMarginalCostPriceSetter(PriceSetter):
    """Set prices as sector markups over technical marginal costs.

    The rule uses technical MC as the markup anchor and a normal-AC floor for
    break-even discipline. Realised accounting unit costs remain diagnostics
    elsewhere in the model and are not used by this price setter.
    """

    PRICE_FLOOR = 1e-2
    SUPPLY_EPSILON = 1e-2
    FALLBACK_NONE = 0
    FALLBACK_NORMAL_OUTPUT = 1
    FALLBACK_MC = 2
    FALLBACK_AC = 3
    FALLBACK_PREVIOUS_PRICE = 4
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
    INITIAL_COST_NORMALIZATION_NONE = "none"
    INITIAL_COST_NORMALIZATION_OUTPUT_WEIGHTED_ROBUST_GAP = "output_weighted_robust_gap"
    INITIAL_COST_NORMALIZATION_MODES = {
        INITIAL_COST_NORMALIZATION_NONE,
        INITIAL_COST_NORMALIZATION_OUTPUT_WEIGHTED_ROBUST_GAP,
    }
    NORMALIZATION_STATUS_DISABLED = 0
    NORMALIZATION_STATUS_APPLIED = 1
    NORMALIZATION_STATUS_CLIPPED = 2
    NORMALIZATION_STATUS_INVALID = 3
    NORMALIZATION_STATUS_LOW_VALID_WEIGHT = 4

    def __init__(
        self,
        orbis_markup_path: str = "run_model/data/raw_data/orbis/orbis_markups_by_nace_rev_2_main_section.csv",
        markup_year: int = 2014,
        industry_to_nace_main_section: dict | None = None,
        markup_central_column: str = "mu_op_weighted_median",
        markup_lower_column: str = "mu_op_median_interval_low",
        markup_upper_column: str = "mu_op_median_interval_high",
        markup_corridor_relative_cap: float = 1.0,
        mc_smoothing_horizon: float = 4.0,
        ac_smoothing_horizon: float = 8.0,
        normal_output_smoothing_horizon: float = 8.0,
        normal_output_capital_floor_lambda: float = 0.25,
        demand_pull_speed: float = 1.0,
        ac_floor_share: float = 1.0,
        initial_cost_normalization_mode: str = "none",
        initial_cost_normalization_lower_quantile: float = 0.01,
        initial_cost_normalization_upper_quantile: float = 0.99,
        initial_cost_normalization_min_factor: float = 0.5,
        initial_cost_normalization_max_factor: float = 2.0,
        initial_cost_normalization_min_valid_weight_share: float = 0.5,
    ):
        if mc_smoothing_horizon <= 0:
            raise ValueError("mc_smoothing_horizon must be positive.")
        if ac_smoothing_horizon <= 0:
            raise ValueError("ac_smoothing_horizon must be positive.")
        if normal_output_smoothing_horizon <= 0:
            raise ValueError("normal_output_smoothing_horizon must be positive.")
        if markup_corridor_relative_cap < 0:
            raise ValueError("markup_corridor_relative_cap must be non-negative.")
        if not (0.0 < normal_output_capital_floor_lambda < 1.0):
            raise ValueError("normal_output_capital_floor_lambda must be in (0, 1).")
        if ac_floor_share <= 0:
            raise ValueError("ac_floor_share must be positive.")
        if initial_cost_normalization_mode not in self.INITIAL_COST_NORMALIZATION_MODES:
            raise ValueError(
                f"initial_cost_normalization_mode must be one of {sorted(self.INITIAL_COST_NORMALIZATION_MODES)}."
            )
        if not (0.0 <= initial_cost_normalization_lower_quantile <= 1.0):
            raise ValueError("initial_cost_normalization_lower_quantile must be in [0, 1].")
        if not (0.0 <= initial_cost_normalization_upper_quantile <= 1.0):
            raise ValueError("initial_cost_normalization_upper_quantile must be in [0, 1].")
        if initial_cost_normalization_lower_quantile > initial_cost_normalization_upper_quantile:
            raise ValueError(
                "initial_cost_normalization_lower_quantile must be no greater than "
                "initial_cost_normalization_upper_quantile."
            )
        if initial_cost_normalization_min_factor <= 0.0:
            raise ValueError("initial_cost_normalization_min_factor must be positive.")
        if initial_cost_normalization_max_factor < initial_cost_normalization_min_factor:
            raise ValueError(
                "initial_cost_normalization_max_factor must be at least initial_cost_normalization_min_factor."
            )
        if not (0.0 <= initial_cost_normalization_min_valid_weight_share <= 1.0):
            raise ValueError("initial_cost_normalization_min_valid_weight_share must be in [0, 1].")
        self.orbis_markup_path = orbis_markup_path
        self.markup_year = int(markup_year)
        mapping = industry_to_nace_main_section or self._default_industry_mapping()
        self.industry_to_nace_main_section = {str(key): value for key, value in mapping.items()}
        self.markup_central_column = markup_central_column
        self.markup_lower_column = markup_lower_column
        self.markup_upper_column = markup_upper_column
        self.markup_corridor_relative_cap = float(markup_corridor_relative_cap)
        self.mc_smoothing_horizon = float(mc_smoothing_horizon)
        self.ac_smoothing_horizon = float(ac_smoothing_horizon)
        self.normal_output_smoothing_horizon = float(normal_output_smoothing_horizon)
        self.mc_smoothing_alpha = 2.0 / (self.mc_smoothing_horizon + 1.0)
        self.ac_smoothing_alpha = 2.0 / (self.ac_smoothing_horizon + 1.0)
        self.normal_output_smoothing_alpha = 2.0 / (self.normal_output_smoothing_horizon + 1.0)
        self.normal_output_capital_floor_lambda = float(normal_output_capital_floor_lambda)
        self.demand_pull_speed = max(0.0, min(1.0, float(demand_pull_speed)))
        self.ac_floor_share = float(ac_floor_share)
        self.initial_cost_normalization_mode = initial_cost_normalization_mode
        self.initial_cost_normalization_lower_quantile = float(initial_cost_normalization_lower_quantile)
        self.initial_cost_normalization_upper_quantile = float(initial_cost_normalization_upper_quantile)
        self.initial_cost_normalization_min_factor = float(initial_cost_normalization_min_factor)
        self.initial_cost_normalization_max_factor = float(initial_cost_normalization_max_factor)
        self.initial_cost_normalization_min_valid_weight_share = float(
            initial_cost_normalization_min_valid_weight_share
        )
        self.initial_cost_normalization_factor = 1.0
        self.initial_cost_normalization_status = (
            self.NORMALIZATION_STATUS_DISABLED
            if self.initial_cost_normalization_mode == self.INITIAL_COST_NORMALIZATION_NONE
            else self.NORMALIZATION_STATUS_INVALID
        )
        self._initial_cost_normalization_done = (
            self.initial_cost_normalization_mode == self.INITIAL_COST_NORMALIZATION_NONE
        )

        self.markup_lower_by_industry, self.markup_central_by_industry, self.markup_upper_by_industry = (
            self._load_markup_arrays()
        )
        self._set_neutral_diagnostics(np.array([], dtype=float))

    def update_parameters_from_config(self, parameters: dict) -> None:
        SectorMarkupMarginalCostPriceSetter.__init__(self, **parameters)

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
        lower_delta = np.clip(lower / central - 1.0, -self.markup_corridor_relative_cap, 0.0)
        upper_delta = np.clip(upper / central - 1.0, 0.0, self.markup_corridor_relative_cap)
        lower = central * (1.0 + lower_delta)
        upper = central * (1.0 + upper_delta)
        return lower, central, upper

    def _set_neutral_diagnostics(self, prev_prices: np.ndarray) -> None:
        self.last_pricing_mc = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_mc_smooth = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_ac = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_ac_smooth = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_material_mc = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_labour_mc = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_depreciation_unit_cost = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_initial_price_gap = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_normal_output = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_markup_mu = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_markup_lower = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_markup_upper = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_ac_floor_binding = np.zeros(prev_prices.shape, dtype=float)
        self.last_pricing_ac_fallback_binding = np.zeros(prev_prices.shape, dtype=float)
        self.last_pricing_gate_state = np.zeros(prev_prices.shape, dtype=float)
        self.last_pricing_fallback_code = np.zeros(prev_prices.shape, dtype=float)
        self.last_pricing_cost_normalization_factor = np.full(
            prev_prices.shape, self.initial_cost_normalization_factor, dtype=float
        )
        self.last_pricing_cost_normalization_raw_gap = np.full(prev_prices.shape, np.nan, dtype=float)
        self.last_pricing_cost_normalization_status = np.full(
            prev_prices.shape, self.initial_cost_normalization_status, dtype=float
        )

    @staticmethod
    def _sector_stat(values: np.ndarray, current_firm_sectors: np.ndarray, stat: str) -> np.ndarray:
        out = np.full(values.shape, np.nan, dtype=float)
        valid = np.isfinite(values) & (values > 0.0)
        for sector in np.unique(current_firm_sectors):
            sector_valid = valid & (current_firm_sectors == sector)
            if np.any(sector_valid):
                if stat == "median":
                    out[current_firm_sectors == sector] = np.median(values[sector_valid])
                elif stat == "p95":
                    out[current_firm_sectors == sector] = np.percentile(values[sector_valid], 95)
                else:
                    raise ValueError(f"Unsupported sector stat {stat!r}.")
        return out

    @staticmethod
    def _valid_positive(values: np.ndarray) -> np.ndarray:
        return np.isfinite(values) & (values > 0.0)

    def _smooth_with_fallback(
        self,
        current: np.ndarray,
        previous: np.ndarray | None,
        current_firm_sectors: np.ndarray,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if previous is None or previous.shape != current.shape:
            previous_valid = np.full(current.shape, np.nan, dtype=float)
        else:
            previous_valid = np.asarray(previous, dtype=float).copy()
            previous_valid[~self._valid_positive(previous_valid)] = np.nan

        current_valid = np.asarray(current, dtype=float).copy()
        current_valid[~self._valid_positive(current_valid)] = np.nan
        has_current = self._valid_positive(current_valid)
        has_previous = self._valid_positive(previous_valid)
        sector_median = self._sector_stat(current_valid, current_firm_sectors, "median")

        smoothed = np.where(
            has_current & has_previous,
            alpha * current_valid + (1.0 - alpha) * previous_valid,
            current_valid,
        )
        smoothed = np.where(self._valid_positive(smoothed), smoothed, previous_valid)
        smoothed = np.where(self._valid_positive(smoothed), smoothed, sector_median)
        fallback = (~has_current | ~self._valid_positive(smoothed)).astype(float)
        return smoothed, fallback

    def _average_cost_signal(
        self, ac_raw: np.ndarray, current_firm_sectors: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        sector_ac_median = self._sector_stat(ac_raw, current_firm_sectors, "median")
        sector_ac_p95 = self._sector_stat(ac_raw, current_firm_sectors, "p95")
        ac_current_valid = self._valid_positive(ac_raw)
        ac_outlier = ac_current_valid & np.isfinite(sector_ac_p95) & (ac_raw > sector_ac_p95)
        ac_valid = ac_current_valid & ~ac_outlier
        ac_signal = np.where(ac_valid, ac_raw, np.nan)
        ac_signal = np.where(ac_outlier, sector_ac_median, ac_signal)
        return ac_signal, ac_valid

    @staticmethod
    def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        if not np.any(valid):
            return np.nan
        ordered = np.argsort(values[valid])
        values_ordered = values[valid][ordered]
        weights_ordered = weights[valid][ordered]
        cumulative_weight = np.cumsum(weights_ordered)
        cutoff = quantile * cumulative_weight[-1]
        index = min(np.searchsorted(cumulative_weight, cutoff, side="left"), values_ordered.size - 1)
        return float(values_ordered[index])

    def _maybe_compute_initial_cost_normalization(
        self,
        previous_pre_tax_price: np.ndarray,
        raw_pre_tax_candidate: np.ndarray,
        initial_output_weights: np.ndarray | None,
    ) -> np.ndarray:
        raw_gap = np.divide(
            previous_pre_tax_price,
            raw_pre_tax_candidate,
            out=np.full_like(previous_pre_tax_price, np.nan, dtype=float),
            where=self._valid_positive(previous_pre_tax_price) & self._valid_positive(raw_pre_tax_candidate),
        )
        if self._initial_cost_normalization_done:
            return np.full_like(raw_gap, np.nan, dtype=float)

        self._initial_cost_normalization_done = True
        if self.initial_cost_normalization_mode == self.INITIAL_COST_NORMALIZATION_NONE:
            self.initial_cost_normalization_factor = 1.0
            self.initial_cost_normalization_status = self.NORMALIZATION_STATUS_DISABLED
            return np.full_like(raw_gap, np.nan, dtype=float)

        if initial_output_weights is None:
            self.initial_cost_normalization_factor = 1.0
            self.initial_cost_normalization_status = self.NORMALIZATION_STATUS_INVALID
            return raw_gap

        weights = np.asarray(initial_output_weights, dtype=float)
        if weights.shape != previous_pre_tax_price.shape:
            self.initial_cost_normalization_factor = 1.0
            self.initial_cost_normalization_status = self.NORMALIZATION_STATUS_INVALID
            return raw_gap
        positive_weight = np.isfinite(weights) & (weights > 0.0)
        total_positive_weight = weights[positive_weight].sum()
        valid = (
            positive_weight
            & self._valid_positive(raw_gap)
            & self._valid_positive(previous_pre_tax_price)
            & self._valid_positive(raw_pre_tax_candidate)
        )
        if total_positive_weight <= 0.0 or not np.any(valid):
            self.initial_cost_normalization_factor = 1.0
            self.initial_cost_normalization_status = self.NORMALIZATION_STATUS_INVALID
            return raw_gap

        valid_weight_share = weights[valid].sum() / total_positive_weight
        if valid_weight_share < self.initial_cost_normalization_min_valid_weight_share:
            self.initial_cost_normalization_factor = 1.0
            self.initial_cost_normalization_status = self.NORMALIZATION_STATUS_LOW_VALID_WEIGHT
            return raw_gap

        lower_gap = self._weighted_quantile(
            raw_gap[valid], weights[valid], self.initial_cost_normalization_lower_quantile
        )
        upper_gap = self._weighted_quantile(
            raw_gap[valid], weights[valid], self.initial_cost_normalization_upper_quantile
        )
        trimmed = valid & (raw_gap >= lower_gap) & (raw_gap <= upper_gap)
        if not np.any(trimmed) or weights[trimmed].sum() <= 0.0:
            self.initial_cost_normalization_factor = 1.0
            self.initial_cost_normalization_status = self.NORMALIZATION_STATUS_INVALID
            return raw_gap

        factor_raw = self._weighted_quantile(raw_gap[trimmed], weights[trimmed], 0.5)
        if not np.isfinite(factor_raw) or factor_raw <= 0.0:
            self.initial_cost_normalization_factor = 1.0
            self.initial_cost_normalization_status = self.NORMALIZATION_STATUS_INVALID
            return raw_gap

        factor = float(
            np.clip(
                factor_raw,
                self.initial_cost_normalization_min_factor,
                self.initial_cost_normalization_max_factor,
            )
        )
        self.initial_cost_normalization_factor = factor
        self.initial_cost_normalization_status = (
            self.NORMALIZATION_STATUS_CLIPPED
            if not np.isclose(factor, factor_raw)
            else self.NORMALIZATION_STATUS_APPLIED
        )
        return raw_gap

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
        pricing_material_mc: np.ndarray | None = None,
        pricing_effective_labour_inputs: np.ndarray | None = None,
        pricing_normal_output: np.ndarray | None = None,
        pricing_depreciation_unit_cost: np.ndarray | None = None,
        wage_obligation_preview: np.ndarray | None = None,
        producer_tax_rates: np.ndarray | None = None,
        prev_mc_smooth: np.ndarray | None = None,
        prev_ac_smooth: np.ndarray | None = None,
        prev_normal_output: np.ndarray | None = None,
        initial_output_weights: np.ndarray | None = None,
        min_inflation: float = -1.0,
        max_inflation: float = 1.0,
    ) -> np.ndarray:
        """Calculate prices from markup corridors, technical MC, and normal AC."""
        _ = (
            current_estimated_ppi_inflation,
            excess_demand,
            inventories,
            production,
            curr_unit_costs,
            prev_unit_costs,
            ppi_during,
            current_time,
        )
        self._set_neutral_diagnostics(prev_prices)
        current_firm_sectors = current_firm_sectors.astype(int)
        if np.any(current_firm_sectors < 0) or np.any(current_firm_sectors >= len(self.markup_central_by_industry)):
            raise ValueError("current_firm_sectors contains sectors not covered by markup configuration.")
        if producer_tax_rates is None:
            producer_tax_rates = np.zeros(prev_prices.shape, dtype=float)
        producer_tax_rates = np.asarray(producer_tax_rates, dtype=float)
        if np.any(producer_tax_rates >= 1.0):
            raise ValueError("Producer tax rates must be below 1.0 for gross-up pricing.")

        markup_lower = self.markup_lower_by_industry[current_firm_sectors]
        markup_central = self.markup_central_by_industry[current_firm_sectors]
        markup_upper = self.markup_upper_by_industry[current_firm_sectors]
        if pricing_material_mc is None:
            pricing_material_mc = np.full(prev_prices.shape, np.nan, dtype=float)
        if pricing_effective_labour_inputs is None:
            pricing_effective_labour_inputs = np.full(prev_prices.shape, np.nan, dtype=float)
        if pricing_normal_output is None:
            pricing_normal_output = np.full(prev_prices.shape, np.nan, dtype=float)
        if pricing_depreciation_unit_cost is None:
            pricing_depreciation_unit_cost = np.zeros(prev_prices.shape, dtype=float)
        if wage_obligation_preview is None:
            wage_obligation_preview = np.full(prev_prices.shape, np.nan, dtype=float)

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

        demand_pull = (
            np.divide(
                prev_demand,
                np.maximum(prev_supply, self.SUPPLY_EPSILON),
                out=np.ones_like(prev_demand, dtype=float),
            )
            - 1.0
        )
        demand_pull = np.clip(demand_pull, min_inflation, max_inflation)

        markup_mu = markup_central.copy()
        positive = demand_pull > 0.0
        negative = demand_pull < 0.0
        markup_mu[positive] = markup_central[positive] + self.demand_pull_speed * demand_pull[positive] * (
            markup_upper[positive] - markup_central[positive]
        )
        markup_mu[negative] = markup_central[negative] + self.demand_pull_speed * demand_pull[negative] * (
            markup_central[negative] - markup_lower[negative]
        )
        markup_mu = np.clip(markup_mu, markup_lower, markup_upper)

        normal_output, normal_output_fallback = self._smooth_with_fallback(
            np.asarray(pricing_normal_output, dtype=float),
            prev_normal_output,
            current_firm_sectors,
            self.normal_output_smoothing_alpha,
        )
        wage_obligation_preview = np.asarray(wage_obligation_preview, dtype=float)
        pricing_effective_labour_inputs = np.asarray(pricing_effective_labour_inputs, dtype=float)
        labour_mc = np.divide(
            wage_obligation_preview,
            pricing_effective_labour_inputs,
            out=np.full_like(prev_prices, np.nan, dtype=float),
            where=np.isfinite(wage_obligation_preview)
            & (wage_obligation_preview >= 0.0)
            & np.isfinite(pricing_effective_labour_inputs)
            & (pricing_effective_labour_inputs > 0.0),
        )
        material_mc_unscaled = np.asarray(pricing_material_mc, dtype=float)
        depreciation_unit_cost_unscaled = np.asarray(pricing_depreciation_unit_cost, dtype=float).copy()
        depreciation_unit_cost_unscaled[
            ~(np.isfinite(depreciation_unit_cost_unscaled) & (depreciation_unit_cost_unscaled >= 0.0))
        ] = 0.0

        mc_raw_unscaled = material_mc_unscaled + labour_mc
        mc_raw_unscaled[~self._valid_positive(mc_raw_unscaled)] = np.nan
        mc_smooth_unscaled, _ = self._smooth_with_fallback(
            mc_raw_unscaled, prev_mc_smooth, current_firm_sectors, self.mc_smoothing_alpha
        )
        ac_raw_unscaled = mc_raw_unscaled + depreciation_unit_cost_unscaled
        ac_raw_unscaled[~self._valid_positive(ac_raw_unscaled)] = np.nan
        ac_signal_unscaled, _ = self._average_cost_signal(ac_raw_unscaled, current_firm_sectors)
        ac_smooth_unscaled, _ = self._smooth_with_fallback(
            ac_signal_unscaled, prev_ac_smooth, current_firm_sectors, self.ac_smoothing_alpha
        )

        previous_pre_tax_price = np.where(
            producer_tax_rates > 0.0,
            prev_prices * (1.0 - producer_tax_rates),
            prev_prices,
        )
        raw_markup_candidate = markup_mu * mc_smooth_unscaled
        raw_ac_candidate = self.ac_floor_share * ac_smooth_unscaled
        raw_pre_tax_candidate = np.maximum(raw_markup_candidate, raw_ac_candidate)
        normalization_pending = not self._initial_cost_normalization_done
        normalization_raw_gap = self._maybe_compute_initial_cost_normalization(
            previous_pre_tax_price=previous_pre_tax_price,
            raw_pre_tax_candidate=raw_pre_tax_candidate,
            initial_output_weights=initial_output_weights,
        )

        cost_normalization_factor = self.initial_cost_normalization_factor
        material_mc = material_mc_unscaled * cost_normalization_factor
        labour_mc = labour_mc * cost_normalization_factor
        depreciation_unit_cost = depreciation_unit_cost_unscaled * cost_normalization_factor
        prev_mc_for_smoothing = prev_mc_smooth
        prev_ac_for_smoothing = prev_ac_smooth
        if normalization_pending and not np.isclose(cost_normalization_factor, 1.0):
            if prev_mc_smooth is not None and prev_mc_smooth.shape == prev_prices.shape:
                prev_mc_for_smoothing = np.asarray(prev_mc_smooth, dtype=float) * cost_normalization_factor
            if prev_ac_smooth is not None and prev_ac_smooth.shape == prev_prices.shape:
                prev_ac_for_smoothing = np.asarray(prev_ac_smooth, dtype=float) * cost_normalization_factor

        mc_raw = material_mc + labour_mc
        mc_raw[~self._valid_positive(mc_raw)] = np.nan
        mc_smooth, mc_fallback = self._smooth_with_fallback(
            mc_raw, prev_mc_for_smoothing, current_firm_sectors, self.mc_smoothing_alpha
        )

        ac_raw = mc_raw + depreciation_unit_cost
        ac_raw[~self._valid_positive(ac_raw)] = np.nan
        ac_signal, ac_valid = self._average_cost_signal(ac_raw, current_firm_sectors)
        ac_smooth, ac_smooth_fallback = self._smooth_with_fallback(
            ac_signal, prev_ac_for_smoothing, current_firm_sectors, self.ac_smoothing_alpha
        )
        ac_fallback = (~ac_valid | (ac_smooth_fallback > 0.0)).astype(float)

        markup_candidate = markup_mu * mc_smooth
        ac_candidate = self.ac_floor_share * ac_smooth
        pre_tax_candidate = np.maximum(markup_candidate, ac_candidate)
        initial_price_gap = np.divide(
            previous_pre_tax_price,
            pre_tax_candidate,
            out=np.full_like(prev_prices, np.nan, dtype=float),
            where=self._valid_positive(previous_pre_tax_price) & self._valid_positive(pre_tax_candidate),
        )
        ac_floor_binding = (ac_candidate >= markup_candidate) & self._valid_positive(ac_candidate)
        grossup = np.where(
            producer_tax_rates > 0.0,
            1.0 / (1.0 - producer_tax_rates),
            1.0,
        )
        computed_prices = pre_tax_candidate * grossup
        invalid_price = ~self._valid_positive(computed_prices)
        prices = np.where(invalid_price, prev_prices, np.maximum(self.PRICE_FLOOR, computed_prices))
        fallback_code = np.zeros(prev_prices.shape, dtype=float)
        fallback_code = np.where(normal_output_fallback > 0.0, self.FALLBACK_NORMAL_OUTPUT, fallback_code)
        fallback_code = np.where(mc_fallback > 0.0, self.FALLBACK_MC, fallback_code)
        fallback_code = np.where(ac_fallback > 0.0, self.FALLBACK_AC, fallback_code)
        fallback_code = np.where(invalid_price, self.FALLBACK_PREVIOUS_PRICE, fallback_code)

        self.last_pricing_mc = mc_raw
        self.last_pricing_mc_smooth = mc_smooth
        self.last_pricing_ac = ac_raw
        self.last_pricing_ac_smooth = ac_smooth
        self.last_pricing_material_mc = material_mc
        self.last_pricing_labour_mc = labour_mc
        self.last_pricing_depreciation_unit_cost = depreciation_unit_cost
        self.last_pricing_initial_price_gap = initial_price_gap
        self.last_pricing_normal_output = normal_output
        self.last_pricing_markup_mu = markup_mu
        self.last_pricing_markup_lower = markup_lower
        self.last_pricing_markup_upper = markup_upper
        self.last_pricing_ac_floor_binding = ac_floor_binding.astype(float)
        self.last_pricing_ac_fallback_binding = ac_fallback.astype(float)
        self.last_pricing_gate_state = gate_state
        self.last_pricing_fallback_code = fallback_code
        self.last_pricing_cost_normalization_factor = np.full(
            prev_prices.shape, self.initial_cost_normalization_factor, dtype=float
        )
        self.last_pricing_cost_normalization_raw_gap = normalization_raw_gap
        self.last_pricing_cost_normalization_status = np.full(
            prev_prices.shape, self.initial_cost_normalization_status, dtype=float
        )
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
        pricing_material_mc: np.ndarray | None = None,
        pricing_effective_labour_inputs: np.ndarray | None = None,
        pricing_normal_output: np.ndarray | None = None,
        pricing_depreciation_unit_cost: np.ndarray | None = None,
        wage_obligation_preview: np.ndarray | None = None,
        producer_tax_rates: np.ndarray | None = None,
        prev_mc_smooth: np.ndarray | None = None,
        prev_ac_smooth: np.ndarray | None = None,
        prev_normal_output: np.ndarray | None = None,
        initial_output_weights: np.ndarray | None = None,
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
            pricing_material_mc=pricing_material_mc,
            pricing_effective_labour_inputs=pricing_effective_labour_inputs,
            pricing_normal_output=pricing_normal_output,
            pricing_depreciation_unit_cost=pricing_depreciation_unit_cost,
            wage_obligation_preview=wage_obligation_preview,
            producer_tax_rates=producer_tax_rates,
            prev_mc_smooth=prev_mc_smooth,
            prev_ac_smooth=prev_ac_smooth,
            prev_normal_output=prev_normal_output,
            initial_output_weights=initial_output_weights,
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
        pricing_material_mc: np.ndarray | None = None,
        pricing_effective_labour_inputs: np.ndarray | None = None,
        pricing_normal_output: np.ndarray | None = None,
        pricing_depreciation_unit_cost: np.ndarray | None = None,
        wage_obligation_preview: np.ndarray | None = None,
        producer_tax_rates: np.ndarray | None = None,
        prev_mc_smooth: np.ndarray | None = None,
        prev_ac_smooth: np.ndarray | None = None,
        prev_normal_output: np.ndarray | None = None,
        initial_output_weights: np.ndarray | None = None,
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
