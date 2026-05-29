from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional

import h5py
import numpy as np
import pandas as pd

from macro_data import SyntheticFirms
from macro_data.readers.emission_fraction.emission_fraction_reader import EmissionFractions
from macro_data.readers.exo_prices import SectorExoPrices
from macromodel.agents.agent import Agent
from macromodel.agents.firms.firm_ts import FirmTimeSeries
from macromodel.agents.firms.utils.create_bundle_matrix import create_bundle_matrix
from macromodel.configurations import FirmsConfiguration
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.goods_market.value_type import ValueType
from macromodel.util.function_mapping import functions_from_model, update_functions


@dataclass
class FirmInsolvencyResult:
    npl_ratio: float
    default_flag: np.ndarray
    loan_writeoff_by_bank: np.ndarray
    overdraft_writeoff_by_bank: np.ndarray
    credit_loss_by_bank: np.ndarray


class Firms(Agent):
    """A collection of producing firms in the economy.

    This class represents the productive sector of the economy, managing multiple firms
    that produce goods using labor, intermediate inputs, and capital inputs. Each firm:
    - Makes production decisions based on expected demand and capacity
    - Sets prices based on costs and market conditions
    - Hires workers and sets wages
    - Manages inventory and input stocks
    - Makes investment decisions
    - Handles financial decisions (borrowing, etc.)
    - Can become insolvent

    The firms operate in discrete time steps, with each period involving:
    1. Planning (production targets, input needs, hiring)
    2. Market participation (labor, credit, goods markets)
    3. Production and sales
    4. Financial settlement (wages, loans, taxes)

    Attributes:
        country_name (str): Country the firms operate in
        all_country_names (list[str]): All countries in simulation
        n_industries (int): Number of industry sectors
        functions (dict[str, Callable]): Production and decision functions
        ts (FirmTimeSeries): Time series data for firms
        states (dict[str, np.ndarray]): Current state variables
        base_intermediate_inputs_productivity_matrix (np.ndarray): Input-output coefficients
        base_capital_inputs_productivity_matrix (np.ndarray): Capital productivity coefficients
        base_capital_input_use_matrix (np.ndarray): Physical capital-use/replacement coefficients
        goods_criticality_matrix (np.ndarray): Critical input requirements
        intermediate_inputs_utilisation_rate (float): Input capacity utilization
        capital_inputs_utilisation_rate (float): Capital capacity utilization
        capital_inputs_delay (np.ndarray): Investment implementation lags
        depreciation_rates (np.ndarray): Asset depreciation rates
        average_initial_price (np.ndarray): Initial price levels
        configuration (FirmsConfiguration): Model parameters
        industries (list[str]): Industry sector names
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Callable],
        ts: FirmTimeSeries,
        states: dict[str, np.ndarray],
        intermediate_inputs_productivity_matrix: np.ndarray,
        capital_inputs_productivity_matrix: np.ndarray,
        capital_input_use_matrix: np.ndarray,
        capital_depreciation_rates: np.ndarray,
        goods_criticality_matrix: np.ndarray,
        intermediate_inputs_utilisation_rate: float,
        capital_inputs_utilisation_rate: float,
        depreciation_rates: np.ndarray,
        capital_inputs_delay: np.ndarray,
        average_initial_price: np.ndarray,
        configuration: FirmsConfiguration,
        industries: list[str],
        bundle_matrix: np.ndarray,
        emission_fractions: Optional[EmissionFractions] = None,
    ):
        """Initialize the firms sector.

        Args:
            country_name (str): Country identifier
            all_country_names (list[str]): All countries in simulation
            n_industries (int): Number of industry sectors
            functions (dict[str, Callable]): Production and decision functions
            ts (FirmTimeSeries): Time series container
            states (dict[str, np.ndarray]): Initial state variables
            intermediate_inputs_productivity_matrix (np.ndarray): Input-output coefficients
            capital_inputs_productivity_matrix (np.ndarray): Capital productivity coefficients
            capital_input_use_matrix (np.ndarray): Physical capital-use/replacement coefficients
            capital_depreciation_rates (np.ndarray): Period CFC/output accounting ratios by industry
            goods_criticality_matrix (np.ndarray): Critical input requirements
            intermediate_inputs_utilisation_rate (float): Input capacity utilization
            capital_inputs_utilisation_rate (float): Capital capacity utilization
            depreciation_rates (np.ndarray): Asset depreciation rates
            capital_inputs_delay (np.ndarray): Investment implementation lags
            average_initial_price (np.ndarray): Initial price levels
            configuration (FirmsConfiguration): Model parameters
            industries (list[str]): Industry sector names
            bundle_matrix (np.ndarray): Matrix to manage bundles for goods (mapping each industry to an
                                               identifier of similar goods for which it can be substituted)
            emission_fractions (Optional[EmissionFractions]): Per-industry emission fraction multipliers
        """
        n_transactors = ts.current("n_firms")
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            n_transactors,
            n_transactors,
            ts,
            states,
            transactor_settings={
                "Buyer Value Type": ValueType.REAL,
                "Seller Value Type": ValueType.REAL,
                "Buyer Priority": 1,
                "Seller Priority": 1,
            },
        )

        self.functions: dict[str, Any] = functions
        self.base_intermediate_inputs_productivity_matrix = intermediate_inputs_productivity_matrix
        self.base_capital_inputs_productivity_matrix = capital_inputs_productivity_matrix
        self.base_capital_input_use_matrix = capital_input_use_matrix
        self.capital_depreciation_rates = capital_depreciation_rates
        self.goods_criticality_matrix = goods_criticality_matrix
        self.intermediate_inputs_utilisation_rate = intermediate_inputs_utilisation_rate
        self.capital_inputs_utilisation_rate = capital_inputs_utilisation_rate
        self.capital_inputs_delay = capital_inputs_delay
        self.depreciation_rates = depreciation_rates

        self.average_initial_price = average_initial_price
        self.current_good_prices = np.asarray(average_initial_price, dtype=float)
        self.pricing_rho_k_by_sector = self._initial_pricing_capital_productivity()

        self.configuration = configuration

        self.industries = industries

        self.substitution_bundles = bundle_matrix
        self.emission_fractions = emission_fractions

    @property
    def base_capital_inputs_depreciation_matrix(self) -> np.ndarray:
        """Deprecated alias for the physical capital-use/replacement matrix."""
        return self.base_capital_input_use_matrix

    @base_capital_inputs_depreciation_matrix.setter
    def base_capital_inputs_depreciation_matrix(self, value: np.ndarray) -> None:
        self.base_capital_input_use_matrix = value

    def get_effective_intermediate_coefficients(self) -> np.ndarray:
        """Get the effective intermediate input coefficients for each firm.

        Returns base coefficients adjusted by firm-specific technical multipliers.
        Shape: [n_industries, n_firms] (transposed for firm-wise operations)
        """
        # Base matrix is [n_industries, n_industries]
        # Select columns for each firm's industry
        base_coefficients = self.base_intermediate_inputs_productivity_matrix[:, self.states["Industry"]].T

        # Apply firm-specific multipliers if they exist
        multipliers = self.states.get("intermediate_tech_multipliers")
        if multipliers is not None:
            # Multipliers are [n_firms, n_industries]
            # Base coefficients are [n_firms, n_industries] after transpose
            # Element-wise multiply each firm's coefficients by their multipliers
            return base_coefficients * multipliers

        return base_coefficients

    def _initial_pricing_capital_productivity(self) -> np.ndarray:
        """Sector median normal-output productivity of installed capital."""
        rho = np.full(self.n_industries, np.nan, dtype=float)
        industries = np.asarray(self.states["Industry"], dtype=int)
        target_production = np.asarray(self.ts.current("target_production"), dtype=float)
        capital_stock = np.asarray(self.ts.current("capital_inputs_stock"), dtype=float).sum(axis=1)
        valid = np.isfinite(target_production) & (target_production > 0.0) & np.isfinite(capital_stock) & (
            capital_stock > 0.0
        )
        firm_rho = np.divide(
            target_production,
            capital_stock,
            out=np.full_like(target_production, np.nan, dtype=float),
            where=valid,
        )
        for sector in range(self.n_industries):
            sector_valid = valid & (industries == sector)
            if np.any(sector_valid):
                rho[sector] = np.median(firm_rho[sector_valid])
        return rho

    @staticmethod
    def _valid_positive(values: np.ndarray) -> np.ndarray:
        return np.isfinite(values) & (values > 0.0)

    def compute_pricing_material_mc(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Compute technical material marginal cost from reciprocal productivity."""
        good_prices = np.asarray(current_good_prices, dtype=float).copy()
        valid_prices = self._valid_positive(good_prices)
        if np.any(valid_prices):
            good_prices[~valid_prices] = np.median(good_prices[valid_prices])
        else:
            return np.full(self.ts.current("n_firms"), np.nan, dtype=float)

        productivity = np.asarray(self.get_effective_intermediate_coefficients(), dtype=float)
        return np.divide(
            good_prices[None, :],
            productivity,
            out=np.zeros_like(productivity, dtype=float),
            where=np.isfinite(productivity) & (productivity > 0.0),
        ).sum(axis=1)

    def compute_pricing_normal_output_candidate(self, capital_floor_lambda: float) -> np.ndarray:
        """Compute target-based normal output with a conservative capital floor."""
        target_output = np.asarray(self.ts.current("target_production"), dtype=float).copy()
        target_output[~self._valid_positive(target_output)] = np.nan

        capital_stock = np.asarray(self.ts.current("capital_inputs_stock"), dtype=float).sum(axis=1)
        sector_rho = self.pricing_rho_k_by_sector[self.states["Industry"]]
        capital_output = sector_rho * capital_stock
        capital_floor = capital_floor_lambda * capital_output
        capital_valid = self._valid_positive(capital_floor)

        normal_output = np.where(
            capital_valid,
            np.fmax(target_output, capital_floor),
            target_output,
        )
        normal_output[~self._valid_positive(normal_output)] = np.nan
        return normal_output

    def compute_pricing_depreciation_unit_cost(self, producer_tax_rates: np.ndarray) -> np.ndarray:
        """Compute non-circular CFC unit cost from previous pre-tax price."""
        mode = self.configuration.parameters.capital_depreciation_accounting_mode
        if mode == "none":
            return np.zeros(self.ts.current("n_firms"), dtype=float)
        if mode != "eurostat_cfc":
            raise ValueError(f"Unknown capital_depreciation_accounting_mode {mode!r}; expected 'none' or 'eurostat_cfc'.")

        producer_tax_rates = np.asarray(producer_tax_rates, dtype=float)
        previous_price = np.asarray(self.ts.current("price"), dtype=float)
        previous_pre_tax_price = np.where(
            producer_tax_rates > 0.0,
            previous_price * (1.0 - producer_tax_rates),
            previous_price,
        )
        previous_pre_tax_price[~self._valid_positive(previous_pre_tax_price)] = np.nan
        rates = self.capital_depreciation_rates[self.states["Industry"]]
        depreciation_unit_cost = rates * previous_pre_tax_price
        depreciation_unit_cost[~(np.isfinite(depreciation_unit_cost) & (depreciation_unit_cost >= 0.0))] = 0.0
        return depreciation_unit_cost

    def get_effective_capital_coefficients(self) -> np.ndarray:
        """Get the effective capital input coefficients for each firm.

        Returns base coefficients adjusted by firm-specific technical multipliers.
        Shape: [n_industries, n_firms] (transposed for firm-wise operations)
        """
        # For capital, we use capital productivity coefficients.
        base_coefficients = self.base_capital_inputs_productivity_matrix[:, self.states["Industry"]].T

        # Apply firm-specific multipliers if they exist
        multipliers = self.states.get("capital_tech_multipliers")
        if multipliers is not None:
            return base_coefficients * multipliers

        return base_coefficients

    def current_goods_criticality_by_firm(self) -> np.ndarray:
        """Return goods criticality aligned with firm-by-good arrays.

        The goods criticality data typically arrives as an (n_industries, n_industries)
        matrix: output/demand industry × input/supply good. Production routines apply
        criticality masks to firm-by-good matrices (n_firms × n_industries), so we
        select the appropriate output-industry row for each firm.
        """
        criticality = np.asarray(self.goods_criticality_matrix)
        n_firms = self.ts.current("n_firms")
        n_industries = self.n_industries

        if criticality.ndim == 2:
            if criticality.shape == (n_industries, n_industries):
                return criticality[self.states["Industry"]]
            if criticality.shape == (n_firms, n_industries):
                return criticality

        if criticality.ndim == 1 and criticality.shape == (n_industries,):
            return np.broadcast_to(criticality[None, :], (n_firms, n_industries))

        if criticality.shape == (18, 18) and n_industries != 18:
            return np.ones((n_firms, n_industries))

        raise ValueError(
            "Unsupported goods_criticality_matrix shape "
            f"{criticality.shape}; expected ({n_industries},{n_industries}), "
            f"({n_firms},{n_industries}), or ({n_industries},)."
        )

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_firms: SyntheticFirms,
        configuration: FirmsConfiguration,
        country_name: str,
        all_country_names: list[str],
        goods_criticality_matrix: pd.DataFrame | np.ndarray,
        average_initial_price: np.ndarray,
        industries: list[str],
        add_emissions: bool = False,
        emission_fractions: Optional[EmissionFractions] = None,
        firm_exo_prices: Optional[SectorExoPrices] = None,
    ):
        """Create a Firms instance from pickled synthetic data.

        Factory method that constructs a Firms object from serialized synthetic data,
        configuration parameters, and initial economic conditions.

        Args:
            synthetic_firms (SyntheticFirms): Synthetic firm data
            configuration (FirmsConfiguration): Model configuration parameters
            country_name (str): Country identifier
            all_country_names (list[str]): All countries in simulation
            goods_criticality_matrix (pd.DataFrame | np.ndarray): Input criticality coefficients
            average_initial_price (np.ndarray): Initial price levels
            industries (list[str]): Industry sector names
            add_emissions (bool, optional): Whether to track emissions. Defaults to False.
            emission_fractions (Optional[EmissionFractions]): Per-industry emission fraction multipliers.
            firm_exo_prices (Optional[SectorExoPrices]): Sector-level exogenous price paths.

        Returns:
            Firms: Initialized Firms instance
        """
        if (
            configuration.parameters.capital_depreciation_accounting_mode == "eurostat_cfc"
            and configuration.parameters.capital_replacement_matrix_source != "eurostat_cfc_output"
        ):
            raise ValueError(
                "capital_depreciation_accounting_mode='eurostat_cfc' requires "
                "capital_replacement_matrix_source='eurostat_cfc_output'."
            )
        from macromodel.agents.firms.func.prices import SectorExogenousPriceSetter

        functions = functions_from_model(model=configuration.functions, loc="macromodel.agents.firms")

        if isinstance(functions.get("prices"), SectorExogenousPriceSetter) and firm_exo_prices is not None:
            functions["prices"].firm_exo_prices = firm_exo_prices
            functions["prices"].overriden_industries = industries

        intermediate_inputs_productivity_matrix = synthetic_firms.intermediate_inputs_productivity_matrix
        capital_inputs_productivity_matrix = synthetic_firms.capital_inputs_productivity_matrix
        if hasattr(synthetic_firms, "capital_input_use_matrix"):
            capital_input_use_matrix = synthetic_firms.capital_input_use_matrix
        else:
            capital_input_use_matrix = synthetic_firms.capital_inputs_depreciation_matrix
        capital_depreciation_rates = getattr(
            synthetic_firms,
            "capital_depreciation_rates",
            np.zeros(len(synthetic_firms.industries)),
        )
        if isinstance(goods_criticality_matrix, pd.DataFrame):
            goods_criticality_matrix = goods_criticality_matrix.values

        corr_employees = synthetic_firms.firm_data["Employees ID"]

        corr_employees = [[int(x) for x in sublist if not pd.isna(x)] for sublist in corr_employees]

        data = synthetic_firms.firm_data.drop(columns=["Employees ID"]).astype(float).rename_axis("Firm ID")

        if add_emissions:
            inputs_emissions = synthetic_firms.firm_data["Input Emissions"].values
            capital_emissions = synthetic_firms.firm_data["Capital Emissions"].values
            inputs_emissions_ch4 = np.zeros_like(inputs_emissions)
            capital_emissions_ch4 = np.zeros_like(capital_emissions)
            input_dict: dict = {
                f"{key}_inputs_emissions": synthetic_firms.firm_data[f"{emitting_industry} Input Emissions"]
                for key, emitting_industry in zip(
                    ["coal", "oil", "gas", "refined_products"], ["Coal", "Oil", "Gas", "Refined Products"]
                )
            }
            capital_dict = {
                f"{key}_capital_emissions": synthetic_firms.firm_data[f"{emitting_industry} Capital Emissions"]
                for key, emitting_industry in zip(
                    ["coal", "oil", "gas", "refined_products"], ["Coal", "Oil", "Gas", "Refined Products"]
                )
            }

        else:
            inputs_emissions = None
            capital_emissions = None
            inputs_emissions_ch4 = None
            capital_emissions_ch4 = None
            input_dict = {}
            capital_dict = {}

        ts = FirmTimeSeries.from_data(
            data=data,
            intermediate_inputs_stock=synthetic_firms.intermediate_inputs_stock,
            used_intermediate_inputs=synthetic_firms.used_intermediate_inputs,
            capital_inputs_stock=synthetic_firms.capital_inputs_stock,
            used_capital_inputs=synthetic_firms.used_capital_inputs,
            initial_good_prices=average_initial_price,
            n_industries=len(synthetic_firms.industries),
            calculate_hill_exponent=configuration.calculate_hill_exponent,
            inputs_emissions=inputs_emissions,
            capital_emissions=capital_emissions,
            inputs_emissions_ch4=inputs_emissions_ch4,
            capital_emissions_ch4=capital_emissions_ch4,
            **input_dict,
            **capital_dict,
        )

        states: dict[str, Any] = {}

        for state_name in [
            "Industry",
            "Corresponding Bank ID",
        ]:
            if state_name not in data.columns:
                raise ValueError("Missing " + state_name + " from the data for initialising firms.")
            states[state_name] = data[state_name].fillna(-1).values.astype(int)

        states["Employments"] = corr_employees
        states["is_insolvent"] = np.full(data.shape[0], False)
        states["Excess Demand"] = np.zeros(data.shape[0])

        states["Labour Productivity by Industry"] = synthetic_firms.labour_productivity_by_industry

        # Initialize TFP multiplier to 1.0 (no TFP effect initially)
        states["tfp_multiplier"] = np.ones(data.shape[0])
        states["forced_productivity_investment"] = np.zeros(data.shape[0])

        # Initialize technical coefficient multipliers and cumulative improvements
        n_firms = data.shape[0]
        n_industries = len(synthetic_firms.industries)

        # Multipliers start at 1.0 (no improvement initially)
        states["intermediate_tech_multipliers"] = np.ones((n_firms, n_industries))
        states["capital_tech_multipliers"] = np.ones((n_firms, n_industries))

        # Cumulative improvements start at 0.0
        states["cumulative_intermediate_improvements"] = np.zeros((n_firms, n_industries))
        states["cumulative_capital_improvements"] = np.zeros((n_firms, n_industries))

        bundle_matrix = create_bundle_matrix(np.array(configuration.substitution_bundles))

        return cls(
            country_name,
            all_country_names,
            len(synthetic_firms.industries),
            functions,
            ts,
            states,
            intermediate_inputs_productivity_matrix,
            capital_inputs_productivity_matrix,
            capital_input_use_matrix,
            capital_depreciation_rates,
            goods_criticality_matrix,
            configuration.parameters.intermediate_inputs_utilisation_rate,
            configuration.parameters.capital_inputs_utilisation_rate,
            np.array(configuration.parameters.depreciation_rates),
            np.array(configuration.parameters.capital_inputs_delay),
            average_initial_price,
            configuration=configuration,
            industries=industries,
            bundle_matrix=bundle_matrix,
            emission_fractions=emission_fractions,
        )

    @property
    def industries_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame mapping firms to their industries.

        Returns a DataFrame with firm indices and their corresponding industry names.

        Returns:
            pd.DataFrame: DataFrame with 'Firm_ID' as the index and 'Industry' as the column.
        """
        # Get the industry indices of the firms
        industry_indices = self.states["Industry"]

        # Map industry indices to industry names
        industry_names = [self.industries[i] for i in industry_indices]

        # Assuming firms are indexed from 0 to N-1
        firm_indices = np.arange(len(industry_indices))

        # Create the DataFrame
        df = pd.DataFrame({"Industry": industry_names}, index=firm_indices)
        df.index.name = "Firm_ID"

        return df

    def reset(self, configuration: FirmsConfiguration) -> None:
        """Reset the firms to initial state with new configuration.

        Resets all time series and state variables to initial values,
        updates function configurations, and recalculates initial stocks
        based on the new configuration parameters.

        Args:
            configuration (FirmsConfiguration): New model configuration
        """
        self.gen_reset()
        update_functions(model=configuration.functions, loc="macromodel.agents.firms", functions=self.functions)

        current_inv = (
            self.ts.current("production")
            * configuration.functions.target_production.parameters["target_inventory_to_demand_fraction"]
        )

        industries = self.states["Industry"]
        initial_good_prices = self.average_initial_price

        inter_inputs_stock = (
            1.0
            / configuration.parameters.intermediate_inputs_utilisation_rate
            * np.divide(
                self.ts.current("production"),
                self.base_intermediate_inputs_productivity_matrix[:, industries],
                out=np.zeros(self.base_intermediate_inputs_productivity_matrix[:, industries].shape),
                where=self.base_intermediate_inputs_productivity_matrix[:, industries] != 0.0,
            ).T
        )

        cap_inputs_stock = (
            1.0
            / configuration.parameters.capital_inputs_utilisation_rate
            * np.divide(
                self.ts.current("production"),
                self.base_capital_inputs_productivity_matrix[:, industries],
                out=np.zeros(self.base_capital_inputs_productivity_matrix[:, industries].shape),
                where=self.base_capital_inputs_productivity_matrix[:, industries] != 0.0,
            ).T
        )

        self.ts.reset_values(  # noqa
            inventory=current_inv,
            initial_good_prices=initial_good_prices,
            intermediate_inputs_stock=inter_inputs_stock,
            capital_inputs_stock=cap_inputs_stock,
        )

        # Reset productivity multipliers to 1
        self.states["tfp_multiplier"] = np.ones_like(self.states["tfp_multiplier"])
        self.ts["tfp_multiplier"] = self.states["tfp_multiplier"].copy()
        self.states["forced_productivity_investment"] = np.zeros_like(self.states["forced_productivity_investment"])
        self.states["intermediate_tech_multipliers"] = np.ones_like(self.states["intermediate_tech_multipliers"])
        self.states["capital_tech_multipliers"] = np.ones_like(self.states["capital_tech_multipliers"])

        self.configuration = deepcopy(configuration)

    def update_number_of_firms(self) -> None:
        """Update the time series tracking number of firms.

        Records the current number of total firms and firms by industry
        in the time series data.
        """
        self.ts.n_firms.append(self.ts.current("n_firms"))
        self.ts.n_firms_by_industry.append(self.ts.current("n_firms_by_industry"))

    def set_estimates(
        self,
        current_estimated_growth: float,
        previous_average_good_prices: np.ndarray,
    ) -> None:
        """Set growth and demand estimates for firms.

        Updates time series with estimated growth rates by firm and
        estimated demand based on current economic conditions.

        Args:
            current_estimated_growth (float): Overall estimated growth rate
            previous_average_good_prices (np.ndarray): Previous period prices
        """
        self.ts.estimated_growth_by_firm.append(
            self.compute_estimated_growth_by_firm(previous_average_good_prices=previous_average_good_prices)
        )
        self.ts.estimated_demand.append(
            self.compute_estimated_demand(current_estimated_growth=current_estimated_growth)
        )

    def compute_estimated_growth_by_firm(
        self,
        previous_average_good_prices: np.ndarray,
        min_growth: float = -0.2,
        max_growth: float = 0.2,
    ) -> np.ndarray:
        """Calculate expected growth rates for each firm.

        Estimates firm-specific growth rates based on:
        - Price changes
        - Supply and demand conditions
        - Industry dynamics
        Growth rates are bounded between min_growth and max_growth.

        Args:
            previous_average_good_prices (np.ndarray): Previous period prices
            min_growth (float, optional): Minimum growth rate. Defaults to -0.2.
            max_growth (float, optional): Maximum growth rate. Defaults to 0.2.

        Returns:
            np.ndarray: Expected growth rate for each firm
        """
        if len(self.ts.historic("inventory")) == 1:
            prev_supply = self.ts.current("production") + self.ts.current("inventory")
        else:
            prev_supply = self.ts.current("production") + self.ts.prev("inventory")
        growth = self.functions["growth_estimator"].compute_growth(
            prev_average_good_prices=previous_average_good_prices,
            prev_firm_prices=self.ts.current("price"),
            prev_supply=prev_supply,
            prev_demand=self.ts.current("demand"),
            current_firm_sectors=self.states["Industry"],
        )
        return np.maximum(min_growth, np.minimum(max_growth, growth))

    def compute_estimated_demand(
        self,
        current_estimated_growth: float,
    ) -> np.ndarray:
        """Estimate future demand for each firm's output.

        Projects demand based on:
        - Previous period demand
        - Overall economic growth
        - Firm-specific growth expectations

        Args:
            current_estimated_growth (float): Overall estimated growth rate

        Returns:
            np.ndarray: Estimated demand for each firm
        """
        return self.functions["demand_estimator"].compute_estimated_demand(
            previous_demand=self.ts.current("demand"),
            current_estimated_growth=current_estimated_growth,
            estimated_growth_by_firm=self.ts.current("estimated_growth_by_firm"),
        )

    def set_targets(
        self,
        bank_overdraft_rate_on_firm_deposits: np.ndarray,
        bank_interest_rates_on_long_term_firm_loans: np.ndarray,
        estimated_growth: float,
        estimated_inflation: float,
        current_good_prices: np.ndarray,
    ) -> None:
        """Set production and input targets for firms.

        Updates time series with:
        - Input constraints (intermediate and capital)
        - Target production levels
        - Desired labor inputs
        - Target input purchases
        - Planned productivity investment

        Args:
            bank_overdraft_rate_on_firm_deposits (np.ndarray): Overdraft interest rates
            bank_interest_rates_on_long_term_firm_loans (np.ndarray): Long-term firm loan rates by bank
            estimated_growth: Expected real growth rate
            estimated_inflation: Expected inflation rate
            current_good_prices (np.ndarray): Industry-level average prices
        """
        goods_criticality_by_firm = self.current_goods_criticality_by_firm()
        self.ts.limiting_intermediate_inputs.append(
            self.functions["production"].compute_limiting_intermediate_inputs_stock(
                intermediate_inputs_productivity_matrix=self.get_effective_intermediate_coefficients(),
                intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
                intermediate_inputs_utilisation_rate=self.intermediate_inputs_utilisation_rate,
                goods_criticality_matrix=goods_criticality_by_firm,
                substitution_bundle_matrix=self.substitution_bundles,
            )
        )
        self.ts.limiting_capital_inputs.append(
            self.functions["production"].compute_limiting_capital_inputs_stock(
                capital_inputs_productivity_matrix=self.get_effective_capital_coefficients(),
                capital_inputs_stock=self.ts.current("capital_inputs_stock"),
                capital_inputs_utilisation_rate=self.capital_inputs_utilisation_rate,
                goods_criticality_matrix=goods_criticality_by_firm,
                substitution_bundle_matrix=self.substitution_bundles,
            )
        )
        self.ts.target_production.append(
            self.compute_target_production(
                bank_overdraft_rate_on_firm_deposits=bank_overdraft_rate_on_firm_deposits,
            )
        )
        self.ts.desired_labour_inputs.append(self.compute_desired_labour_inputs())
        self.ts.target_intermediate_inputs_production.append(self.compute_target_intermediate_inputs_production())
        self.ts.target_capital_inputs_production.append(self.compute_target_capital_inputs_production())

        # Plan productivity investment using industry-level prices
        total_investment, tfp_investment, technical_investment = self.plan_productivity_investment(
            estimated_inflation=estimated_inflation,
            current_good_prices=current_good_prices,
            bank_interest_rates_on_long_term_firm_loans=bank_interest_rates_on_long_term_firm_loans,
        )
        # Store total investment for backward compatibility
        self.ts.planned_productivity_investment.append(total_investment)

        # Store separate components for new allocation logic (now done directly by the planner)
        self.ts.planned_tfp_investment.append(tfp_investment)
        self.ts.planned_technical_investment.append(technical_investment)

        self.append_tfp_investment_foc_diagnostics()

    def append_tfp_investment_foc_diagnostics(self) -> None:
        """Record planner-time direct-TFP marginal benefit/cost diagnostics."""
        planner = self.functions.get("productivity_investment_planner")
        diagnostics = getattr(planner, "last_diagnostics", None)
        n_firms = self.ts.current("n_firms")
        n_industries = self.n_industries

        firm_fields = {
            "effective_cost_rate": "tfp_investment_effective_cost_rate",
            "target_intensity": "tfp_investment_target_intensity",
            "cap_intensity": "tfp_investment_cap_intensity",
            "desired_intensity": "tfp_investment_desired_intensity",
            "planned_intensity": "tfp_investment_planned_intensity",
            "desired_marginal_benefit": "tfp_investment_desired_marginal_benefit",
            "desired_marginal_cost": "tfp_investment_desired_marginal_cost",
            "desired_mb_mc_ratio": "tfp_investment_desired_mb_mc_ratio",
            "desired_marginal_gap": "tfp_investment_desired_marginal_gap",
            "planned_marginal_benefit": "tfp_investment_planned_marginal_benefit",
            "planned_marginal_cost": "tfp_investment_planned_marginal_cost",
            "planned_mb_mc_ratio": "tfp_investment_planned_mb_mc_ratio",
            "planned_marginal_gap": "tfp_investment_planned_marginal_gap",
            "cap_binding": "tfp_investment_cap_binding",
            "cash_binding": "tfp_investment_cash_binding",
        }
        sector_fields = {
            "desired_intensity": "sector_tfp_investment_desired_intensity",
            "planned_intensity": "sector_tfp_investment_planned_intensity",
            "desired_mb_mc_ratio": "sector_tfp_investment_desired_mb_mc_ratio",
            "planned_mb_mc_ratio": "sector_tfp_investment_planned_mb_mc_ratio",
            "desired_marginal_gap": "sector_tfp_investment_desired_marginal_gap",
            "planned_marginal_gap": "sector_tfp_investment_planned_marginal_gap",
            "cap_binding": "sector_tfp_investment_cap_binding_share",
            "cash_binding": "sector_tfp_investment_cash_binding_share",
        }

        if not diagnostics:
            for field in firm_fields.values():
                getattr(self.ts, field).append(np.full(n_firms, np.nan))
            for field in sector_fields.values():
                getattr(self.ts, field).append(np.full(n_industries, np.nan))
            return

        for diagnostic_key, field in firm_fields.items():
            values = diagnostics.get(diagnostic_key, np.full(n_firms, np.nan))
            getattr(self.ts, field).append(np.asarray(values, dtype=float).copy())

        firm_industries = np.asarray(self.states["Industry"], dtype=int)
        for diagnostic_key, field in sector_fields.items():
            values = np.asarray(diagnostics.get(diagnostic_key, np.full(n_firms, np.nan)), dtype=float)
            sector_values = np.full(n_industries, np.nan)
            for sector in range(n_industries):
                sector_mask = firm_industries == sector
                if not np.any(sector_mask):
                    continue
                sector_data = values[sector_mask]
                finite = np.isfinite(sector_data)
                if np.any(finite):
                    sector_values[sector] = np.mean(sector_data[finite])
            getattr(self.ts, field).append(sector_values)

    def plan_productivity_investment(
        self,
        estimated_inflation: float,
        current_good_prices: np.ndarray,
        bank_interest_rates_on_long_term_firm_loans: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Plan productivity investment amounts for each firm.

        Uses the productivity investment planner to determine optimal investment
        amounts based on current conditions and expected returns. Only invests
        if there is available cash after accounting for capital replacement needs.

        Args:
            estimated_inflation: Expected inflation rate
            current_good_prices: Industry-level average prices for inputs
            bank_interest_rates_on_long_term_firm_loans: Long-term firm loan rates by bank

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Total investment, TFP investment, technical investment
        """
        # Calculate expected capital costs using industry-level prices
        # Use current prices adjusted for inflation as expected prices
        expected_prices = (1 + estimated_inflation) * current_good_prices
        # Target capital inputs are in real units, so multiply by expected prices to get costs
        expected_capital_costs = np.matmul(self.ts.current("target_capital_inputs"), expected_prices)

        # Calculate available cash for productivity investment
        # First ensure capital replacement is covered, then use remaining capacity

        # Current liquid resources (deposits, can be negative for overdrafts)
        current_deposits = self.ts.current("deposits")

        # Total credit capacity (short-term + long-term)
        total_target_credit = self.ts.current("target_short_term_credit") + self.ts.current("target_long_term_credit")

        # Total financial capacity = deposits + available credit
        total_financial_capacity = current_deposits + total_target_credit

        # Available cash = total capacity minus capital replacement needs
        # This ensures capital replacement is prioritized
        available_cash = total_financial_capacity - expected_capital_costs

        # Only allow positive available cash (no borrowing beyond capacity)
        available_cash = np.maximum(0.0, available_cash)

        if bank_interest_rates_on_long_term_firm_loans is None:
            effective_cost_rate = None
        else:
            corresponding_bank_ids = self.states["Corresponding Bank ID"]
            effective_cost_rate = bank_interest_rates_on_long_term_firm_loans[corresponding_bank_ids]

        # Get investment allocation from planner
        total_investment, tfp_investment, technical_investment = self.functions[
            "productivity_investment_planner"
        ].plan_productivity_investment(
            current_tfp=self.states["tfp_multiplier"],
            current_production=self.ts.current("production"),
            current_unit_costs=self.compute_unit_costs(),
            available_cash=available_cash,
            current_prices=current_good_prices,
            n_industries=self.n_industries,
            input_usage=self.ts.current("used_intermediate_inputs"),
            current_tech_multipliers=self.states["intermediate_tech_multipliers"],
            substitution_bundle_matrix=self.substitution_bundles,
            current_nominal_production=self.ts.current("price") * self.ts.current("production"),
            max_cash_fraction=self.configuration.parameters.max_productivity_cash_fraction,
            max_investment_fraction=self.configuration.parameters.max_productivity_investment_fraction,
            firm_industries=self.states["Industry"],
            effective_cost_rate=effective_cost_rate,
        )

        return total_investment, tfp_investment, technical_investment

    def compute_estimated_profits(self, estimated_growth: float, estimated_inflation: float) -> np.ndarray:
        """Estimate future profits for each firm.

        Uses the profit estimator function to project future profits based on
        current profits and expected economic conditions.

        Args:
            estimated_growth (float): Expected real growth rate
            estimated_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Estimated profits for each firm
        """
        return self.functions["profit_estimator"].compute_estimated_profits(
            current_profits=self.ts.current("profits"),
            estimated_growth=estimated_growth,
            estimated_inflation=estimated_inflation,
        )

    def compute_target_production(
        self,
        bank_overdraft_rate_on_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        """Calculate production targets for each firm.

        Sets target production levels based on:
        - Estimated demand
        - Current inventory levels
        - Input constraints (labor, intermediate, capital)
        - Financial constraints (equity, debt capacity)

        Args:
            bank_overdraft_rate_on_firm_deposits (np.ndarray): Overdraft interest rates

        Returns:
            np.ndarray: Target production quantities for each firm
        """
        return self.functions["target_production"].compute_target_production(
            current_estimated_demand=self.ts.current("estimated_demand"),
            initial_inventory=self.ts.initial("inventory"),
            previous_inventory=self.ts.current("inventory"),
            previous_production=self.ts.current("production"),
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            current_firm_equity=self.ts.current("equity"),
            current_firm_debt=self.ts.current("debt"),
            previous_loans_applied_for=self.ts.current("target_short_term_credit")
            + self.ts.current("target_long_term_credit"),
            current_firm_deposits=self.ts.current("deposits"),
            interest_on_overdraft_rates=-bank_overdraft_rate_on_firm_deposits[self.states["Corresponding Bank ID"]]
            * np.minimum(0.0, self.ts.current("deposits")),
            interest_paid_on_loans=self.ts.current("interest_paid_on_loans"),
        )

    def compute_target_intermediate_inputs_production(self) -> np.ndarray:
        """Calculate target intermediate input production levels.

        Determines constrained intermediate input targets based on:
        - Previous production
        - Target production
        - Input constraints (labor, intermediate, capital)
        - Financial constraints (equity, debt)

        Returns:
            np.ndarray: Target intermediate input production for each firm
        """

        target_intermediate_inputs = self.functions[
            "target_production"
        ].compute_constrained_intermediate_inputs_target_production(
            previous_production=self.ts.current("production"),
            current_target_production=self.ts.current("target_production"),
            current_limiting_labour_inputs=self.ts.current("labour_inputs"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            current_firm_equity=self.ts.current("equity"),
            current_firm_debt=self.ts.current("debt"),
            previous_loans_applied_for=self.ts.current("target_short_term_credit")
            + self.ts.current("target_long_term_credit"),
        )

        target_intermediate_inputs = fillna(target_intermediate_inputs)

        return target_intermediate_inputs

    def compute_target_capital_inputs_production(self) -> np.ndarray:
        """Calculate target capital input production levels.

        Determines constrained capital input targets based on:
        - Previous production
        - Target production
        - Input constraints (labor, intermediate, capital)
        - Financial constraints (equity, debt)

        Returns:
            np.ndarray: Target capital input production for each firm
        """

        target_capital_inputs = self.functions[
            "target_production"
        ].compute_constrained_capital_inputs_target_production(
            previous_production=self.ts.current("production"),
            current_target_production=self.ts.current("target_production"),
            current_limiting_labour_inputs=self.ts.current("labour_inputs"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            current_firm_equity=self.ts.current("equity"),
            current_firm_debt=self.ts.current("debt"),
            previous_loans_applied_for=self.ts.current("target_short_term_credit")
            + self.ts.current("target_long_term_credit"),
        )

        target_capital_inputs = fillna(target_capital_inputs)

        return target_capital_inputs

    def compute_desired_labour_inputs(self) -> np.ndarray:
        """Calculate desired labor inputs for production.

        Determines optimal labor requirements based on:
        - Target production levels
        - Intermediate input constraints
        - Capital input constraints

        Returns:
            np.ndarray: Desired labor inputs for each firm
        """
        return self.functions["desired_labour"].compute_desired_labour(
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            current_tfp_multiplier=self.states["tfp_multiplier"],
        )

    def compute_labour_inputs(self, corresponding_firm: np.ndarray, current_labour_inputs: np.ndarray) -> np.ndarray:
        """Calculate effective labor inputs for each firm.

        Computes actual labor inputs based on:
        - Employee assignments
        - Labor productivity factors
        - Industry-specific productivity

        Args:
            corresponding_firm (np.ndarray): Mapping of employees to firms
            current_labour_inputs (np.ndarray): Raw labor inputs per employee

        Returns:
            np.ndarray: Effective labor inputs per firm
        """
        labour_inputs_from_employees = np.bincount(
            corresponding_firm[corresponding_firm >= 0],
            weights=current_labour_inputs[corresponding_firm >= 0],
            minlength=self.ts.current("n_firms"),
        )
        industry_labour_productivity_by_firm = self.states["Labour Productivity by Industry"][self.states["Industry"]]

        # Compute labour productivity
        self.ts.labour_productivity_factor.append(
            self.functions["labour_productivity"].compute_labour_productivity_factor(
                current_target_production=self.ts.current("target_production"),
                current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
                current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
                labour_inputs_from_employees=labour_inputs_from_employees,
                industry_labour_productivity_by_firm=industry_labour_productivity_by_firm,
                current_tfp_multiplier=self.states["tfp_multiplier"],
            )
        )
        self.ts.labour_productivity.append(
            self.ts.current("labour_productivity_factor") * industry_labour_productivity_by_firm
        )

        # Compute labour inputs
        self.ts.labour_inputs.append(self.ts.current("labour_productivity") * labour_inputs_from_employees)
        self.ts.normalised_labour_inputs.append(industry_labour_productivity_by_firm * labour_inputs_from_employees)

        return labour_inputs_from_employees

    def compute_n_employees(self, corresponding_firm: np.ndarray) -> np.ndarray:
        """Count number of employees per firm.

        Args:
            corresponding_firm (np.ndarray): Mapping of employees to firms

        Returns:
            np.ndarray: Number of employees for each firm
        """
        return np.bincount(
            corresponding_firm[corresponding_firm >= 0],
            minlength=self.ts.current("n_firms"),
        )

    def compute_production(self) -> np.ndarray:
        """Calculate actual production quantities.

        Determines production based on:
        - Target production levels
        - Labor input constraints (scaled by TFP)
        - Intermediate input constraints (scaled by TFP)
        - Capital input constraints (scaled by TFP)

        Returns:
            np.ndarray: Actual production quantity for each firm
        """
        return self.functions["production"].compute_production(
            desired_production=self.ts.current("target_production"),
            current_labour_inputs=self.ts.current("labour_inputs"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            tfp_multiplier=self.states.get("tfp_multiplier"),
        )

    def compute_total_sales(self) -> np.ndarray:
        """Calculate total sales revenue net of production taxes.

        Returns:
            np.ndarray: Net sales revenue for each firm
        """
        return self.ts.current("price") * self.ts.current("production") - self.ts.current("taxes_paid_on_production")

    def compute_nominal_output_value(self) -> np.ndarray:
        """Calculate nominal gross output value for each firm."""
        return self.ts.current("price") * self.ts.current("production")

    def compute_wages_markup(self) -> np.ndarray:
        """Calculate wage markup based on labor market tightness.

        Computes markup factor based on:
        - Historical desired labor inputs
        - Historical realized labor inputs

        Returns:
            np.ndarray: Wage markup factor for each firm
        """
        return self.functions["wage_setter"].compute_wage_tightness_markup(
            historic_desired_labour_inputs=self.ts.historic("desired_labour_inputs"),
            historic_realised_labour_inputs=self.ts.historic("labour_inputs"),
        )

    def compute_offered_wage_function(
        self,
        corresponding_firm: np.ndarray,
        current_individual_labour_inputs: np.ndarray,
        previous_employee_income: np.ndarray,
        unemployment_benefits_by_individual: float,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
    ) -> Callable[[int, float | np.ndarray], float | np.ndarray]:
        """Create function that computes offered wages.

        Returns a function that calculates wage offers based on:
        - Employee productivity
        - Previous wages
        - Labor market conditions
        - Tax rates
        - Unemployment benefits

        Args:
            corresponding_firm (np.ndarray): Mapping of employees to firms
            current_individual_labour_inputs (np.ndarray): Current labor inputs
            previous_employee_income (np.ndarray): Previous wages
            unemployment_benefits_by_individual (float): Unemployment benefit rate
            income_taxes (float): Income tax rate
            employee_social_insurance_tax (float): Employee SI tax rate
            employer_social_insurance_tax (float): Employer SI tax rate

        Returns:
            np.ndarray: Wage offers for each firm
        """
        return self.functions["wage_setter"].get_offered_wage_given_labour_inputs_function(
            corresponding_firm=corresponding_firm,
            current_individual_labour_inputs=current_individual_labour_inputs,
            previous_employee_income=previous_employee_income,
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            industry_labour_productivity_by_firm=self.states["Labour Productivity by Industry"][
                self.states["Industry"]
            ],
            initial_wage_per_capita=self.ts.initial("real_wage_per_capita"),
            current_wage_per_capita=self.ts.current("real_wage_per_capita"),
            current_labour_productivity_factor=self.ts.current("labour_productivity_factor"),
            prev_labour_productivity_factor=self.ts.prev("labour_productivity_factor"),
            current_wage_tightness_markup=self.ts.current("wage_tightness_markup"),
            income_taxes=income_taxes,
            employee_social_insurance_tax=employee_social_insurance_tax,
            employer_social_insurance_tax=employer_social_insurance_tax,
            unemployment_benefits_by_individual=unemployment_benefits_by_individual,
            current_tfp_multiplier=self.states["tfp_multiplier"],
            prev_tfp_multiplier=self.ts.prev("tfp_multiplier"),
        )

    def set_employee_income(
        self,
        corresponding_firm: np.ndarray,
        current_individual_labour_inputs: np.ndarray,
        current_individual_stating_new_job: np.ndarray,
        current_employee_income: np.ndarray,
        current_individual_offered_wage: np.ndarray,
        labour_inputs_from_employees: np.ndarray,
        estimated_ppi_inflation: float,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
    ) -> np.ndarray:
        """Set employee wages based on offers and market conditions.

        Updates wages considering:
        - New vs existing employees
        - Offered wages
        - Expected inflation
        - Tax rates
        - Labor productivity

        Args:
            corresponding_firm (np.ndarray): Mapping of employees to firms
            current_individual_labour_inputs (np.ndarray): Labor inputs
            current_individual_stating_new_job (np.ndarray): New job indicators
            current_employee_income (np.ndarray): Current wages
            current_individual_offered_wage (np.ndarray): Wage offers
            labour_inputs_from_employees (np.ndarray): Labor inputs by firm
            estimated_ppi_inflation (float): Expected inflation
            income_taxes (float): Income tax rate
            employee_social_insurance_tax (float): Employee SI tax rate
            employer_social_insurance_tax (float): Employer SI tax rate

        Returns:
            np.ndarray: Updated employee wages
        """
        return self.functions["wage_setter"].set_employee_income(
            corresponding_firm=corresponding_firm,
            current_individual_labour_inputs=current_individual_labour_inputs,
            current_individual_stating_new_job=current_individual_stating_new_job,
            current_employee_income=current_employee_income,
            current_individual_offered_wage=current_individual_offered_wage,
            current_target_production=self.ts.current("target_production"),
            current_limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            current_limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            labour_inputs_from_employees=labour_inputs_from_employees,
            industry_labour_productivity_by_firm=self.states["Labour Productivity by Industry"][
                self.states["Industry"]
            ],
            initial_wage_per_capita=self.ts.initial("real_wage_per_capita"),
            current_wage_per_capita=self.ts.current("real_wage_per_capita"),
            current_labour_productivity_factor=self.ts.current("labour_productivity_factor"),
            prev_labour_productivity_factor=self.ts.prev("labour_productivity_factor"),
            current_wage_tightness_markup=self.ts.current("wage_tightness_markup"),
            estimated_ppi_inflation=estimated_ppi_inflation,
            income_taxes=income_taxes,
            employee_social_insurance_tax=employee_social_insurance_tax,
            employer_social_insurance_tax=employer_social_insurance_tax,
            current_tfp_multiplier=self.states["tfp_multiplier"],
        )

    def update_total_wages_paid(
        self,
        corresponding_firm: np.ndarray,
        individual_wages: np.ndarray,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
        cpi: float,
    ) -> None:
        """Update total wage payments including taxes and adjustments.

        Calculates total wage costs including:
        - Base wages
        - Employer social insurance contributions
        - Tax adjustments
        - Real wage per capita

        Args:
            corresponding_firm (np.ndarray): Mapping of employees to firms
            individual_wages (np.ndarray): Individual wage rates
            income_taxes (float): Income tax rate
            employee_social_insurance_tax (float): Employee SI tax rate
            employer_social_insurance_tax (float): Employer SI tax rate
            cpi (float): Consumer price index
        """
        self.ts.total_wage.append(
            self.compute_total_wage_obligation(
                corresponding_firm=corresponding_firm,
                individual_wages=individual_wages,
                income_taxes=income_taxes,
                employee_social_insurance_tax=employee_social_insurance_tax,
                employer_social_insurance_tax=employer_social_insurance_tax,
                cpi=cpi,
            )
        )
        self.ts.real_wage_per_capita.append(
            self.ts.current("total_wage") / cpi / self.ts.current("number_of_employees")
        )

    def compute_total_wage_obligation(
        self,
        corresponding_firm: np.ndarray,
        individual_wages: np.ndarray,
        income_taxes: float,
        employee_social_insurance_tax: float,
        employer_social_insurance_tax: float,
        cpi: float,
    ) -> np.ndarray:
        """Preview current wage obligations without appending to time series."""
        real_wages = np.bincount(
            corresponding_firm[corresponding_firm >= 0],
            weights=individual_wages[corresponding_firm >= 0],
            minlength=self.ts.current("n_firms"),
        )
        return cpi * (
            (1.0 + employer_social_insurance_tax)
            / (1 - employee_social_insurance_tax - income_taxes * (1 - employee_social_insurance_tax))
            * real_wages
        )

    def compute_price(
        self,
        current_estimated_ppi_inflation: np.ndarray,
        previous_average_good_prices: np.ndarray,
        ppi_during: np.ndarray,
        current_good_prices: Optional[np.ndarray] = None,
        wage_obligation_preview: Optional[np.ndarray] = None,
        producer_tax_rates: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Set prices for each firm's output.

        Determines prices based on:
        - Previous prices
        - Expected inflation
        - Market conditions (excess demand)
        - Inventory levels
        - Production costs
        - Industry dynamics

        Args:
            current_estimated_ppi_inflation (np.ndarray): Expected PPI inflation
            previous_average_good_prices (np.ndarray): Previous period prices
            ppi_during (np.ndarray): Producer price indices
            current_good_prices (np.ndarray, optional): Latest known good prices
                before this price update.
            wage_obligation_preview (np.ndarray, optional): Current wage
                obligation preview used by marginal-cost pricing.
            producer_tax_rates (np.ndarray, optional): Producer tax/subsidy
                rates by firm.

        Returns:
            np.ndarray: New prices for each firm
        """
        price_setter = self.functions["prices"]
        if current_good_prices is None:
            current_good_prices = previous_average_good_prices
        if producer_tax_rates is None:
            producer_tax_rates = np.zeros(self.ts.current("n_firms"), dtype=float)
        if wage_obligation_preview is None:
            wage_obligation_preview = np.full(self.ts.current("n_firms"), np.nan, dtype=float)

        pricing_material_mc = self.compute_pricing_material_mc(current_good_prices)
        pricing_normal_output = self.compute_pricing_normal_output_candidate(
            capital_floor_lambda=getattr(price_setter, "normal_output_capital_floor_lambda", 0.25)
        )
        pricing_depreciation_unit_cost = self.compute_pricing_depreciation_unit_cost(producer_tax_rates)
        prices = price_setter.compute_price(
            prev_prices=self.ts.current("price"),
            current_estimated_ppi_inflation=current_estimated_ppi_inflation,
            excess_demand=self.states["Excess Demand"],
            inventories=self.ts.current("inventory"),
            production=self.ts.current("production"),
            prev_average_good_prices=previous_average_good_prices,
            prev_firm_prices=self.ts.current("price"),
            prev_supply=(
                self.ts.current("production") + self.ts.current("inventory")
                if len(self.ts.historic("price")) == 1
                else self.ts.prev("production") + self.ts.current("inventory")
            ),
            prev_demand=self.ts.current("demand"),
            current_firm_sectors=self.states["Industry"],
            curr_unit_costs=self.ts.current("unit_costs"),
            prev_unit_costs=(
                self.ts.current("unit_costs") if len(self.ts.historic("price")) == 1 else self.ts.prev("unit_costs")
            ),
            ppi_during=ppi_during,
            current_time=len(self.ts.historic("price")),
            pricing_material_mc=pricing_material_mc,
            pricing_normal_output=pricing_normal_output,
            pricing_depreciation_unit_cost=pricing_depreciation_unit_cost,
            wage_obligation_preview=wage_obligation_preview,
            producer_tax_rates=producer_tax_rates,
            prev_mc_smooth=self.ts.current("pricing_mc_smooth"),
            prev_ac_smooth=self.ts.current("pricing_ac_smooth"),
            prev_normal_output=self.ts.current("pricing_normal_output"),
        )
        self._append_pricing_diagnostics(price_setter=price_setter, prices=prices)
        return prices

    def _append_pricing_diagnostics(self, price_setter, prices: np.ndarray) -> None:
        """Append pricing diagnostics exposed by price setters."""
        default = np.full(prices.shape, np.nan, dtype=float)
        diagnostics = {
            "pricing_mc": getattr(price_setter, "last_pricing_mc", default),
            "pricing_mc_smooth": getattr(price_setter, "last_pricing_mc_smooth", default),
            "pricing_ac": getattr(price_setter, "last_pricing_ac", default),
            "pricing_ac_smooth": getattr(price_setter, "last_pricing_ac_smooth", default),
            "pricing_normal_output": getattr(price_setter, "last_pricing_normal_output", default),
            "pricing_markup_mu": getattr(price_setter, "last_pricing_markup_mu", default),
            "pricing_markup_lower": getattr(price_setter, "last_pricing_markup_lower", default),
            "pricing_markup_upper": getattr(price_setter, "last_pricing_markup_upper", default),
            "pricing_ac_floor_binding": getattr(price_setter, "last_pricing_ac_floor_binding", default),
            "pricing_ac_fallback_binding": getattr(price_setter, "last_pricing_ac_fallback_binding", default),
            "pricing_gate_state": getattr(price_setter, "last_pricing_gate_state", np.zeros(prices.shape, dtype=float)),
            "pricing_fallback_code": getattr(price_setter, "last_pricing_fallback_code", default),
        }
        for field, values in diagnostics.items():
            getattr(self.ts, field).append(np.asarray(values, dtype=float).copy())

    def compute_unconstrained_demand_for_intermediate_inputs(
        self, good_prices: np.ndarray, extra_taxes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate unconstrained demand for intermediate inputs.

        Determines optimal intermediate input requirements without
        considering financial or supply constraints, based on:
        - Target production
        - Input-output coefficients
        - Current stocks
        - Production history

        Args:
            good_prices (np.ndarray): Current prices for inputs
            extra_taxes (np.ndarray, optional): Additional taxes on inputs. Defaults to None.

        Returns:
            np.ndarray: Unconstrained intermediate input demand for each firm
        """
        return self.functions["target_intermediate_inputs"].compute_unconstrained_target_intermediate_inputs(
            current_target_production=self.ts.current("target_intermediate_inputs_production"),
            intermediate_inputs_productivity_matrix=self.get_effective_intermediate_coefficients(),
            prev_intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
            initial_intermediate_inputs_stock=self.ts.initial("intermediate_inputs_stock"),
            prev_production=self.ts.current("production"),
            initial_production=self.ts.initial("production"),
            previous_good_prices=good_prices,
            substitution_bundle_matrix=self.substitution_bundles,
            extra_taxes=extra_taxes,
        )

    def compute_unconstrained_demand_for_intermediate_inputs_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate value of unconstrained intermediate input demand.

        Computes the monetary value of desired intermediate inputs
        using current market prices.

        Args:
            current_good_prices (np.ndarray): Current prices for inputs

        Returns:
            np.ndarray: Value of unconstrained intermediate input demand for each firm
        """
        return np.matmul(
            self.ts.current("unconstrained_target_intermediate_inputs"),
            current_good_prices,
        )

    def compute_unconstrained_demand_for_capital_inputs(
        self, good_prices: np.ndarray, extra_taxes: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Calculate unconstrained demand for capital inputs.

        Determines optimal capital input requirements without
        considering financial or supply constraints, based on:
        - Target production
        - Capital productivity coefficients
        - Current capital stock
        - Production history

        Args:
            good_prices: np.ndarray
            extra_taxes (np.ndarray, optional): Additional taxes on inputs. Defaults to None.

        Returns:
            np.ndarray: Unconstrained capital input demand for each firm
        """
        return self.functions["target_capital_inputs"].compute_unconstrained_target_capital_inputs(
            current_target_production=self.ts.current("target_capital_inputs_production"),
            capital_input_use_matrix=self.base_capital_input_use_matrix[:, self.states["Industry"]].T,
            prev_capital_inputs_stock=self.ts.current("capital_inputs_stock"),
            initial_capital_inputs_stock=self.ts.initial("capital_inputs_stock"),
            prev_production=self.ts.current("production"),
            initial_production=self.ts.initial("production"),
            substitution_bundle_matrix=self.substitution_bundles,
            previous_good_prices=good_prices,
            extra_taxes=extra_taxes,
        )

    def compute_unconstrained_demand_for_capital_inputs_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        return np.matmul(
            self.ts.current("unconstrained_target_capital_inputs"),
            current_good_prices,
        )

    def compute_target_credit(
        self,
        estimated_growth: float,
        estimated_inflation: float,
        wage_obligation_preview: np.ndarray | None = None,
        production_tax_obligation_preview: np.ndarray | None = None,
        interest_obligation_preview: np.ndarray | None = None,
        loan_interest_obligation_preview: np.ndarray | None = None,
        debt_installment_preview: np.ndarray | None = None,
    ) -> None:
        """Calculate target borrowing levels for each firm.

        Determines desired short-term and long-term credit based on:
        - Expected growth and inflation
        - Projected cash flows
        - Existing overdraft liabilities
        - Input purchase needs
        - Investment plans

        `target_short_term_credit` remains the total borrower-facing short-term
        credit request. It is decomposed into:
        - `target_debt_rollover_credit`: the part needed to refinance scheduled
          loan interest plus principal service that expected liquidity cannot cover.
        - `target_overdraft_refinance_credit`: the part needed to repair existing
          negative deposits after hard obligations and expected sales are budgeted.
        - `ordinary_target_short_term_credit`: the remaining short-term request for
          ordinary working-capital activity.

        The split is used by the credit market so scheduled debt-service rollover
        and emergency overdraft refinance cannot determine the ordinary
        short-term/long-term firm lending cap split.

        Args:
            estimated_growth (float): Expected real growth rate
            estimated_inflation (float): Expected inflation rate
            wage_obligation_preview (np.ndarray | None): Non-appending current wage obligation preview.
            production_tax_obligation_preview (np.ndarray | None): Non-appending current production-tax preview.
            interest_obligation_preview (np.ndarray | None): Current-period total interest cash obligation preview.
            loan_interest_obligation_preview (np.ndarray | None): Current-period loan-interest obligation eligible
                for debt rollover. Deposit/overdraft interest is excluded.
            debt_installment_preview (np.ndarray | None): Current-period scheduled firm principal due preview.
        """
        n_firms = self.ts.current("n_firms")
        if wage_obligation_preview is None:
            wage_obligation_preview = self.ts.current("total_wage")
        if production_tax_obligation_preview is None:
            production_tax_obligation_preview = self.ts.current("taxes_paid_on_production")
        if interest_obligation_preview is None:
            interest_obligation_preview = self.ts.current("interest_paid")
        if loan_interest_obligation_preview is None:
            loan_interest_obligation_preview = np.zeros(n_firms)
        if debt_installment_preview is None:
            debt_installment_preview = self.ts.current("debt_installments")

        estimated_corporate_taxes = self.estimate_corporate_tax_obligation(
            estimated_growth=estimated_growth,
            estimated_inflation=estimated_inflation,
        )
        internal_cash = np.maximum(0.0, self.ts.current("deposits"))
        existing_overdraft = np.maximum(0.0, -self.ts.current("deposits"))
        expected_sales = np.maximum(
            0.0,
            (1 + estimated_growth)
            * (1 + estimated_inflation)
            * self.ts.current("price")
            * self.ts.current("target_production"),
        )
        hard_obligations = (
            wage_obligation_preview
            + production_tax_obligation_preview
            + estimated_corporate_taxes
            + interest_obligation_preview
            + debt_installment_preview
        )
        loan_debt_service_preview = loan_interest_obligation_preview + debt_installment_preview
        non_debt_service_hard_obligations = hard_obligations - loan_debt_service_preview
        planned_technical_investment_costs = (
            self.ts.current("planned_technical_investment").sum(axis=1)
            if len(self.ts.planned_technical_investment) > 0
            else np.zeros(n_firms)
        )
        planned_tfp_investment_costs = (
            self.ts.current("planned_tfp_investment") if len(self.ts.planned_tfp_investment) > 0 else np.zeros(n_firms)
        )
        intermediate_costs = self.ts.current("unconstrained_target_intermediate_inputs_costs")
        capital_costs = self.ts.current("unconstrained_target_capital_inputs_costs")
        working_capital_budget = intermediate_costs + planned_tfp_investment_costs
        investment_budget = capital_costs + planned_technical_investment_costs
        ordinary_target_short_term_credit, target_long_term_credit = self.functions[
            "target_credit"
        ].compute_target_credit(
            internal_cash=internal_cash,
            existing_overdraft=existing_overdraft,
            expected_sales=expected_sales,
            hard_obligations=hard_obligations,
            unconstrained_target_intermediate_inputs_costs=intermediate_costs,
            unconstrained_target_capital_inputs_costs=capital_costs,
            planned_technical_investment_costs=planned_technical_investment_costs,
            planned_tfp_investment_costs=planned_tfp_investment_costs,
        )
        cash_after_hard_obligations = internal_cash + expected_sales - hard_obligations
        non_debt_service_hard_obligation_shortfall = np.maximum(
            0.0,
            non_debt_service_hard_obligations - (internal_cash + expected_sales),
        )
        cash_after_non_debt_service_hard_obligations = (
            internal_cash + expected_sales - non_debt_service_hard_obligations
        )
        available_after_hard_and_overdraft = cash_after_hard_obligations - existing_overdraft
        remaining_internal_finance_after_working_capital = available_after_hard_and_overdraft - working_capital_budget
        target_debt_rollover_credit = np.maximum(
            0.0,
            loan_debt_service_preview - np.maximum(0.0, cash_after_non_debt_service_hard_obligations),
        )
        target_overdraft_refinance_credit = np.maximum(
            0.0,
            existing_overdraft - np.maximum(0.0, cash_after_hard_obligations),
        )
        ordinary_target_short_term_credit = np.maximum(
            0.0,
            ordinary_target_short_term_credit + non_debt_service_hard_obligation_shortfall,
        )
        target_long_term_credit = np.maximum(0.0, target_long_term_credit)
        target_short_term_credit = (
            target_debt_rollover_credit + target_overdraft_refinance_credit + ordinary_target_short_term_credit
        )
        self._append_credit_budget_diagnostics(
            internal_cash=internal_cash,
            existing_overdraft=existing_overdraft,
            expected_sales=expected_sales,
            wage_obligations=wage_obligation_preview,
            production_tax_obligations=production_tax_obligation_preview,
            corporate_tax_obligations=estimated_corporate_taxes,
            interest_obligations=interest_obligation_preview,
            debt_installments=debt_installment_preview,
            hard_obligations=hard_obligations,
            cash_after_hard_obligations=cash_after_hard_obligations,
            available_after_hard_and_overdraft=available_after_hard_and_overdraft,
            intermediate_costs=intermediate_costs,
            tfp_costs=planned_tfp_investment_costs,
            working_capital_budget=working_capital_budget,
            capital_costs=capital_costs,
            technical_investment_costs=planned_technical_investment_costs,
            investment_budget=investment_budget,
            remaining_internal_finance_after_working_capital=remaining_internal_finance_after_working_capital,
        )
        self.ts.target_short_term_credit.append(target_short_term_credit)
        self.ts.total_target_short_term_credit.append([target_short_term_credit.sum()])
        self.ts.target_debt_rollover_credit.append(target_debt_rollover_credit)
        self.ts.total_target_debt_rollover_credit.append([target_debt_rollover_credit.sum()])
        self.ts.target_overdraft_refinance_credit.append(target_overdraft_refinance_credit)
        self.ts.total_target_overdraft_refinance_credit.append([target_overdraft_refinance_credit.sum()])
        self.ts.ordinary_target_short_term_credit.append(ordinary_target_short_term_credit)
        self.ts.total_ordinary_target_short_term_credit.append([ordinary_target_short_term_credit.sum()])
        self.ts.target_long_term_credit.append(target_long_term_credit)
        self.ts.total_target_long_term_credit.append([target_long_term_credit.sum()])

    def _append_credit_budget_diagnostics(
        self,
        internal_cash: np.ndarray,
        existing_overdraft: np.ndarray,
        expected_sales: np.ndarray,
        wage_obligations: np.ndarray,
        production_tax_obligations: np.ndarray,
        corporate_tax_obligations: np.ndarray,
        interest_obligations: np.ndarray,
        debt_installments: np.ndarray,
        hard_obligations: np.ndarray,
        cash_after_hard_obligations: np.ndarray,
        available_after_hard_and_overdraft: np.ndarray,
        intermediate_costs: np.ndarray,
        tfp_costs: np.ndarray,
        working_capital_budget: np.ndarray,
        capital_costs: np.ndarray,
        technical_investment_costs: np.ndarray,
        investment_budget: np.ndarray,
        remaining_internal_finance_after_working_capital: np.ndarray,
    ) -> None:
        """Record the pro forma credit-budget components used by target credit."""
        diagnostics = {
            "credit_budget_internal_cash": internal_cash,
            "credit_budget_existing_overdraft": existing_overdraft,
            "expected_sales": expected_sales,
            "credit_budget_wage_obligations": wage_obligations,
            "credit_budget_production_tax_obligations": production_tax_obligations,
            "credit_budget_corporate_tax_obligations": corporate_tax_obligations,
            "credit_budget_interest_obligations": interest_obligations,
            "credit_budget_debt_installments": debt_installments,
            "credit_budget_hard_obligations": hard_obligations,
            "credit_budget_cash_after_hard_obligations": cash_after_hard_obligations,
            "credit_budget_available_after_hard_and_overdraft": available_after_hard_and_overdraft,
            "credit_budget_intermediate_costs": intermediate_costs,
            "credit_budget_tfp_costs": tfp_costs,
            "credit_budget_working_capital_budget": working_capital_budget,
            "credit_budget_capital_costs": capital_costs,
            "credit_budget_technical_investment_costs": technical_investment_costs,
            "credit_budget_investment_budget": investment_budget,
            "credit_budget_remaining_internal_finance_after_working_capital": (
                remaining_internal_finance_after_working_capital
            ),
        }
        for field, values in diagnostics.items():
            getattr(self.ts, field).append(np.asarray(values, dtype=float).copy())

    def _current_received_ordinary_short_term_credit(self) -> np.ndarray:
        """Return spendable short-term credit excluding rollover and overdraft repair buckets."""
        ordinary_credit = np.nan_to_num(
            np.asarray(self.ts.current("received_ordinary_short_term_credit"), dtype=float),
            nan=0.0,
        )
        received_debt_rollover_credit = np.nan_to_num(
            np.asarray(self.ts.current("received_debt_rollover_credit"), dtype=float),
            nan=0.0,
        )
        received_overdraft_refinance_credit = np.nan_to_num(
            np.asarray(self.ts.current("received_overdraft_refinance_credit"), dtype=float),
            nan=0.0,
        )
        if (
            float(np.nansum(ordinary_credit)) == 0.0
            and float(np.nansum(received_debt_rollover_credit)) == 0.0
            and float(np.nansum(received_overdraft_refinance_credit)) == 0.0
        ):
            ordinary_credit = np.nan_to_num(
                np.asarray(self.ts.current("received_short_term_credit"), dtype=float),
                nan=0.0,
            )
        return np.maximum(0.0, ordinary_credit)

    def begin_firm_debt_settlement(
        self,
        opening_interest_arrears: np.ndarray,
        opening_principal_arrears: np.ndarray,
        contractual_interest_due: np.ndarray,
        contractual_principal_due: np.ndarray,
        scheduled_interest_due: np.ndarray,
        scheduled_principal_due: np.ndarray,
    ) -> None:
        """Append current-period firm debt-settlement diagnostics before goods clearing."""
        n_firms = self.ts.current("n_firms")
        self.ts.firm_settlement_available_cash_before_debt_service.append(np.zeros(n_firms))
        self.ts.firm_settlement_corporate_tax_reserve.append(np.zeros(n_firms))
        self.ts.firm_settlement_cash_after_tax_reserve.append(np.zeros(n_firms))
        self.ts.firm_settlement_opening_interest_arrears.append(
            np.asarray(opening_interest_arrears, dtype=float).copy()
        )
        self.ts.firm_settlement_opening_principal_arrears.append(
            np.asarray(opening_principal_arrears, dtype=float).copy()
        )
        self.ts.firm_settlement_contractual_interest_due.append(
            np.asarray(contractual_interest_due, dtype=float).copy()
        )
        self.ts.firm_settlement_contractual_principal_due.append(
            np.asarray(contractual_principal_due, dtype=float).copy()
        )
        self.ts.firm_settlement_scheduled_interest_due.append(np.asarray(scheduled_interest_due, dtype=float).copy())
        self.ts.firm_settlement_payable_interest.append(np.zeros(n_firms))
        self.ts.firm_settlement_closing_interest_arrears.append(np.zeros(n_firms))
        self.ts.firm_settlement_unpaid_interest.append(np.zeros(n_firms))
        self.ts.firm_settlement_capitalized_interest.append(np.zeros(n_firms))
        self.ts.firm_settlement_scheduled_principal_due.append(np.asarray(scheduled_principal_due, dtype=float).copy())
        self.ts.firm_settlement_payable_principal.append(np.zeros(n_firms))
        self.ts.firm_settlement_closing_principal_arrears.append(np.zeros(n_firms))
        self.ts.firm_settlement_unpaid_principal.append(np.zeros(n_firms))
        self.ts.firm_settlement_debt_rollover_shortfall.append(np.zeros(n_firms))
        self.ts.firm_settlement_overdraft_refinance_used.append(np.zeros(n_firms))
        self.ts.firm_settlement_overdraft_refinance_shortfall.append(np.zeros(n_firms))
        self.ts.firm_settlement_residual_overdraft_exposure.append(np.zeros(n_firms))
        self.ts.firm_settlement_illiquid_flag.append(np.full(n_firms, False))
        self.ts.firm_settlement_default_flag.append(np.full(n_firms, False))
        self.ts.firm_settlement_balance_sheet_residual.append(np.zeros(n_firms))
        self.ts.firm_settlement_transaction_flow_residual.append(np.zeros(n_firms))
        self.ts.firm_settlement_accounting_control_passed.append(np.full(n_firms, False))
        self.ts.total_credit_exposure.append(np.zeros(n_firms))

    @staticmethod
    def _allocate_debt_service_bucket(
        available_capacity: np.ndarray,
        due_amount: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        payable_amount = np.minimum(np.maximum(available_capacity, 0.0), np.maximum(due_amount, 0.0))
        remaining_capacity = np.maximum(0.0, available_capacity - payable_amount)
        return payable_amount, remaining_capacity

    def plan_firm_debt_settlement(
        self,
        taxes_paid_on_production: np.ndarray,
        interest_paid_on_deposits: np.ndarray,
        corporate_tax_obligation_preview: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compute cash-feasible firm debt settlement after goods market outcomes are known."""
        n_firms = self.ts.current("n_firms")
        opening_interest_arrears = np.maximum(0.0, self.ts.current("firm_settlement_opening_interest_arrears"))
        opening_principal_arrears = np.maximum(0.0, self.ts.current("firm_settlement_opening_principal_arrears"))
        contractual_interest_due = np.maximum(0.0, self.ts.current("firm_settlement_contractual_interest_due"))
        contractual_principal_due = np.maximum(0.0, self.ts.current("firm_settlement_contractual_principal_due"))
        scheduled_interest_due = np.maximum(0.0, self.ts.current("firm_settlement_scheduled_interest_due"))
        scheduled_principal_due = np.maximum(0.0, self.ts.current("firm_settlement_scheduled_principal_due"))
        received_debt_rollover_credit = np.maximum(
            0.0,
            np.nan_to_num(self.ts.current("received_debt_rollover_credit"), nan=0.0),
        )
        received_overdraft_refinance_credit = np.maximum(
            0.0,
            np.nan_to_num(self.ts.current("received_overdraft_refinance_credit"), nan=0.0),
        )
        ordinary_short_term_credit = self._current_received_ordinary_short_term_credit()
        available_cash_before_debt_service = (
            self.ts.current("deposits")
            + self.ts.current("nominal_amount_sold_in_lcu")
            + ordinary_short_term_credit
            + np.nan_to_num(self.ts.current("received_long_term_credit"), nan=0.0)
            - self.ts.current("total_wage")
            - self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            - self.direct_tfp_investment_cash_expense()
            - taxes_paid_on_production
            - interest_paid_on_deposits
        )
        corporate_tax_reserve = np.maximum(0.0, corporate_tax_obligation_preview)
        cash_after_tax_reserve = available_cash_before_debt_service - corporate_tax_reserve
        overdraft_refinance_used = np.minimum(
            received_overdraft_refinance_credit,
            np.maximum(0.0, -cash_after_tax_reserve),
        )
        debt_service_capacity = np.maximum(
            0.0,
            cash_after_tax_reserve + overdraft_refinance_used + received_debt_rollover_credit,
        )
        payable_opening_interest_arrears, remaining_ordinary_cash = self._allocate_debt_service_bucket(
            debt_service_capacity,
            opening_interest_arrears,
        )
        payable_contractual_interest, remaining_ordinary_cash = self._allocate_debt_service_bucket(
            remaining_ordinary_cash,
            contractual_interest_due,
        )
        payable_opening_principal_arrears, remaining_principal_capacity = self._allocate_debt_service_bucket(
            remaining_ordinary_cash,
            opening_principal_arrears,
        )
        payable_contractual_principal, _ = self._allocate_debt_service_bucket(
            remaining_principal_capacity,
            contractual_principal_due,
        )
        payable_interest = payable_opening_interest_arrears + payable_contractual_interest
        payable_principal = payable_opening_principal_arrears + payable_contractual_principal
        capitalized_interest = np.maximum(0.0, scheduled_interest_due - payable_interest)
        closing_interest_arrears = np.zeros_like(capitalized_interest)
        closing_principal_arrears = np.maximum(0.0, scheduled_principal_due - payable_principal)
        illiquid_flag = np.logical_or(capitalized_interest > 1e-9, closing_principal_arrears > 1e-9)
        debt_rollover_shortfall = np.maximum(
            0.0,
            np.nan_to_num(self.ts.current("target_debt_rollover_credit"), nan=0.0) - received_debt_rollover_credit,
        )
        overdraft_refinance_shortfall = np.maximum(
            0.0,
            np.nan_to_num(self.ts.current("target_overdraft_refinance_credit"), nan=0.0)
            - received_overdraft_refinance_credit,
        )
        return {
            "available_cash_before_debt_service": available_cash_before_debt_service,
            "corporate_tax_reserve": corporate_tax_reserve,
            "cash_after_tax_reserve": cash_after_tax_reserve,
            "payable_opening_interest_arrears": payable_opening_interest_arrears,
            "payable_contractual_interest": payable_contractual_interest,
            "payable_opening_principal_arrears": payable_opening_principal_arrears,
            "payable_contractual_principal": payable_contractual_principal,
            "payable_interest": payable_interest,
            "closing_interest_arrears": closing_interest_arrears,
            "capitalized_interest": capitalized_interest,
            "payable_principal": payable_principal,
            "closing_principal_arrears": closing_principal_arrears,
            "illiquid_flag": illiquid_flag,
            "debt_rollover_shortfall": debt_rollover_shortfall,
            "overdraft_refinance_used": overdraft_refinance_used,
            "overdraft_refinance_shortfall": overdraft_refinance_shortfall,
            "default_flag": np.full(n_firms, False),
        }

    def finalize_firm_debt_settlement(
        self,
        settlement: dict[str, np.ndarray],
        residual_overdraft_exposure: np.ndarray | None = None,
    ) -> None:
        """Override current-period firm debt-settlement diagnostics with realised values."""
        fields = {
            "firm_settlement_available_cash_before_debt_service": settlement["available_cash_before_debt_service"],
            "firm_settlement_corporate_tax_reserve": settlement["corporate_tax_reserve"],
            "firm_settlement_cash_after_tax_reserve": settlement["cash_after_tax_reserve"],
            "firm_settlement_payable_interest": settlement["payable_interest"],
            "firm_settlement_closing_interest_arrears": settlement["closing_interest_arrears"],
            "firm_settlement_unpaid_interest": settlement["capitalized_interest"],
            "firm_settlement_capitalized_interest": settlement["capitalized_interest"],
            "firm_settlement_payable_principal": settlement["payable_principal"],
            "firm_settlement_closing_principal_arrears": settlement["closing_principal_arrears"],
            "firm_settlement_unpaid_principal": settlement["closing_principal_arrears"],
            "firm_settlement_debt_rollover_shortfall": settlement["debt_rollover_shortfall"],
            "firm_settlement_overdraft_refinance_used": settlement["overdraft_refinance_used"],
            "firm_settlement_overdraft_refinance_shortfall": settlement["overdraft_refinance_shortfall"],
            "firm_settlement_residual_overdraft_exposure": (
                self.ts.current("firm_settlement_residual_overdraft_exposure")
                if residual_overdraft_exposure is None
                else residual_overdraft_exposure
            ),
        }
        for field, values in fields.items():
            self.ts.override_current(field, np.asarray(values, dtype=float).copy())
        self.ts.override_current(
            "firm_settlement_illiquid_flag",
            np.asarray(settlement["illiquid_flag"], dtype=bool).copy(),
        )
        self.ts.override_current(
            "firm_settlement_default_flag",
            np.asarray(settlement["default_flag"], dtype=bool).copy(),
        )

    def check_firm_accounting_controls(
        self,
        current_good_prices: np.ndarray,
        tolerance: float = 1e-4,
        relative_tolerance: float = 1e-12,
        enforce: bool = True,
    ) -> dict[str, np.ndarray]:
        """Check the firm balance-sheet and settlement cash-flow identities.

        The control is evaluated on the pre-default settlement state. It uses the
        opening-deposits snapshot stored during activity-finance planning, the
        closing deposits written during settlement, and the current balance-sheet
        stocks. Principal arrears remain separate from explicit debt and are not
        capitalised by this check.
        """
        current_good_prices = np.asarray(current_good_prices, dtype=float)
        opening_deposits = np.nan_to_num(np.asarray(self.ts.current("activity_finance_opening_deposits"), dtype=float))
        closing_deposits = np.nan_to_num(np.asarray(self.ts.current("deposits"), dtype=float))
        received_credit = np.nan_to_num(np.asarray(self.ts.current("received_credit"), dtype=float))
        debt_installments = np.nan_to_num(np.asarray(self.ts.current("debt_installments"), dtype=float))
        nominal_amount_sold_in_lcu = np.nan_to_num(
            np.asarray(self.ts.current("nominal_amount_sold_in_lcu"), dtype=float)
        )
        nominal_amount_spent_in_lcu = np.nan_to_num(
            np.asarray(self.ts.current("nominal_amount_spent_in_lcu"), dtype=float),
            nan=0.0,
        )
        total_wage = np.nan_to_num(np.asarray(self.ts.current("total_wage"), dtype=float))
        direct_tfp_investment_cash_expense = np.nan_to_num(
            np.asarray(self.ts.current("direct_tfp_investment_cash_expense"), dtype=float)
        )
        taxes_paid_on_production = np.nan_to_num(np.asarray(self.ts.current("taxes_paid_on_production"), dtype=float))
        corporate_taxes_paid = np.nan_to_num(np.asarray(self.ts.current("corporate_taxes_paid"), dtype=float))
        interest_paid = np.nan_to_num(np.asarray(self.ts.current("interest_paid"), dtype=float))

        expected_closing_deposits = (
            opening_deposits
            + nominal_amount_sold_in_lcu
            + received_credit
            - total_wage
            - nominal_amount_spent_in_lcu.sum(axis=1)
            - direct_tfp_investment_cash_expense
            - taxes_paid_on_production
            - corporate_taxes_paid
            - interest_paid
            - debt_installments
        )
        transaction_flow_residual = expected_closing_deposits - closing_deposits

        transaction_flow_scale = np.maximum.reduce(
            [
                np.abs(opening_deposits),
                np.abs(nominal_amount_sold_in_lcu),
                np.abs(received_credit),
                np.abs(total_wage),
                np.abs(nominal_amount_spent_in_lcu).sum(axis=1),
                np.abs(direct_tfp_investment_cash_expense),
                np.abs(taxes_paid_on_production),
                np.abs(corporate_taxes_paid),
                np.abs(interest_paid),
                np.abs(debt_installments),
                np.abs(closing_deposits),
            ]
        )

        material = np.dot(self.ts.current("intermediate_inputs_stock"), current_good_prices)
        capital = np.dot(self.ts.current("capital_inputs_stock"), current_good_prices)
        expected_equity = (
            np.nan_to_num(np.asarray(self.ts.current("inventory"), dtype=float))
            * np.asarray(self.ts.current("price"), dtype=float)
            + material
            + capital
            + closing_deposits
            - np.nan_to_num(np.asarray(self.ts.current("debt"), dtype=float))
        )
        balance_sheet_residual = expected_equity - np.nan_to_num(np.asarray(self.ts.current("equity"), dtype=float))

        balance_sheet_scale = np.maximum.reduce(
            [
                np.abs(expected_equity),
                np.abs(np.nan_to_num(np.asarray(self.ts.current("equity"), dtype=float))),
                np.abs(np.nan_to_num(np.asarray(self.ts.current("inventory"), dtype=float)) * self.ts.current("price")),
                np.abs(material),
                np.abs(capital),
                np.abs(closing_deposits),
                np.abs(np.nan_to_num(np.asarray(self.ts.current("debt"), dtype=float))),
            ]
        )

        balance_sheet_tolerance = tolerance + relative_tolerance * balance_sheet_scale
        transaction_flow_tolerance = tolerance + relative_tolerance * transaction_flow_scale
        control_passed = (np.abs(balance_sheet_residual) <= balance_sheet_tolerance) & (
            np.abs(transaction_flow_residual) <= transaction_flow_tolerance
        )

        self.ts.override_current("firm_settlement_balance_sheet_residual", balance_sheet_residual.copy())
        self.ts.override_current("firm_settlement_transaction_flow_residual", transaction_flow_residual.copy())
        self.ts.override_current("firm_settlement_accounting_control_passed", control_passed.copy())

        if enforce and not np.all(control_passed):
            failing_firms = np.where(~control_passed)[0]
            max_bs_residual = float(np.max(np.abs(balance_sheet_residual[~control_passed])))
            max_cf_residual = float(np.max(np.abs(transaction_flow_residual[~control_passed])))
            max_bs_relative_residual = float(
                np.max(
                    np.divide(
                        np.abs(balance_sheet_residual[~control_passed]),
                        np.maximum(balance_sheet_scale[~control_passed], 1.0),
                    )
                )
            )
            max_cf_relative_residual = float(
                np.max(
                    np.divide(
                        np.abs(transaction_flow_residual[~control_passed]),
                        np.maximum(transaction_flow_scale[~control_passed], 1.0),
                    )
                )
            )
            raise ValueError(
                "Firm accounting control violation: "
                f"failing_firms={failing_firms.tolist()}, "
                f"max_balance_sheet_residual={max_bs_residual:.6g}, "
                f"max_transaction_flow_residual={max_cf_residual:.6g}, "
                f"max_balance_sheet_relative_residual={max_bs_relative_residual:.6g}, "
                f"max_transaction_flow_relative_residual={max_cf_relative_residual:.6g}, "
                f"tolerance={tolerance:.6g}, "
                f"relative_tolerance={relative_tolerance:.6g}"
            )

        return {
            "balance_sheet_residual": balance_sheet_residual,
            "transaction_flow_residual": transaction_flow_residual,
            "control_passed": control_passed,
        }

    def compute_profits_given_interest_paid(
        self,
        interest_paid: np.ndarray,
        taxes_paid_on_production: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return profits using supplied interest and optional production taxes."""
        if taxes_paid_on_production is None:
            taxes_paid_on_production = self.ts.current("taxes_paid_on_production")
        capital_costs = self.capital_compensation_accounting_costs()
        capital_depreciation_costs = self.capital_depreciation_accounting_costs()
        direct_tfp_expense = self.direct_tfp_investment_cash_expense()
        return (
            self.ts.current("price") * self.ts.current("production")
            - self.ts.current("total_wage")
            - self.ts.current("used_intermediate_inputs_costs")
            - capital_costs
            - capital_depreciation_costs
            - taxes_paid_on_production
            - interest_paid
            - direct_tfp_expense
        )

    def compute_corporate_taxes_paid_from_profits(
        self,
        profits: np.ndarray,
        tau_firm: float,
    ) -> np.ndarray:
        """Return corporate taxes implied by supplied profit values."""
        return tau_firm * np.maximum(0.0, profits)

    def estimate_corporate_tax_obligation(
        self,
        estimated_growth: float,
        estimated_inflation: float,
    ) -> np.ndarray:
        """Estimate current-period corporate tax cash obligation."""
        return np.maximum(
            0.0,
            (1 + estimated_growth) * (1 + estimated_inflation) * self.ts.current("corporate_taxes_paid"),
        )

    def compute_debt(self) -> np.ndarray:
        return self.ts.current("short_term_loan_debt") + self.ts.current("long_term_loan_debt")

    def compute_total_credit_exposure(self) -> np.ndarray:
        return self.compute_debt() + np.maximum(0.0, -self.ts.current("deposits"))

    def compute_interest_paid_on_deposits(
        self,
        bank_interest_rate_on_firm_deposits: np.ndarray,
        bank_overdraft_rate_on_firm_deposits: np.ndarray,
    ) -> np.ndarray:
        return -(
            bank_interest_rate_on_firm_deposits[self.states["Corresponding Bank ID"]]
            * np.maximum(0.0, self.ts.current("deposits"))
            + bank_overdraft_rate_on_firm_deposits[self.states["Corresponding Bank ID"]]
            * np.minimum(0.0, self.ts.current("deposits"))
        )

    def compute_interest_paid(self) -> np.ndarray:
        """Calculate total interest payments.

        Sums interest paid on:
        - Loans
        - Deposits/overdrafts

        Returns:
            np.ndarray: Total interest paid by each firm
        """
        return self.ts.current("interest_paid_on_loans") + self.ts.current("interest_paid_on_deposits")

    def compute_offered_price(self) -> np.ndarray:
        """Calculate offered prices by industry.

        Computes weighted average prices based on:
        - Current prices
        - Production quantities
        - Inventory levels

        Returns:
            np.ndarray: Average offered price by industry
        """
        nom = np.bincount(
            self.states["Industry"],
            weights=self.ts.current("price_in_usd") * (self.ts.current("production") + self.ts.current("inventory")),
            minlength=self.n_industries,
        )
        real = np.bincount(
            self.states["Industry"],
            weights=self.ts.current("production") + self.ts.current("inventory"),
            minlength=self.n_industries,
        )
        avg_price = np.divide(nom, real, out=np.zeros(nom.shape), where=real != 0.0)
        avg_price[avg_price == 0.0] = self.ts.current("price_offered")[avg_price == 0.0]
        assert np.all(avg_price > 0.0)
        return avg_price

    def compute_maximum_excess_demand(self) -> np.ndarray:
        """Calculate maximum potential excess demand.

        Determines maximum additional demand that could be met based on:
        - Current production
        - Target production
        - Input constraints

        Returns:
            np.ndarray: Maximum excess demand for each firm
        """
        return self.functions["excess_demand"].set_maximum_excess_demand(
            current_production=self.ts.current("production"),
            target_production=self.ts.current("target_production"),
            limiting_intermediate_inputs=self.ts.current("limiting_intermediate_inputs"),
            limiting_capital_inputs=self.ts.current("limiting_capital_inputs"),
            limiting_labour_inputs=self.ts.current("labour_inputs"),
        )

    def _append_activity_finance_diagnostics(
        self,
        activity_finance_opening_deposits: np.ndarray,
        activity_finance_available: np.ndarray,
        activity_finance_hard_obligations: np.ndarray,
        activity_finance_gap_before_revision: np.ndarray,
        activity_finance_feasible_target_production: np.ndarray,
        activity_finance_feasible_desired_labour_inputs: np.ndarray,
        activity_finance_feasibility_residual: np.ndarray,
        intermediate_scale: np.ndarray,
        capital_scale: np.ndarray,
        technical_scale: np.ndarray,
        tfp_scale: np.ndarray,
        planned_intermediate_costs: np.ndarray,
        feasible_intermediate_costs: np.ndarray,
        planned_capital_costs: np.ndarray,
        feasible_capital_costs: np.ndarray,
        planned_technical_costs: np.ndarray,
        feasible_technical_costs: np.ndarray,
        planned_tfp_costs: np.ndarray,
        feasible_tfp_costs: np.ndarray,
    ) -> None:
        self.ts.activity_finance_opening_deposits.append(activity_finance_opening_deposits)
        self.ts.activity_finance_available.append(activity_finance_available)
        self.ts.activity_finance_hard_obligations.append(activity_finance_hard_obligations)
        self.ts.activity_finance_gap_before_revision.append(activity_finance_gap_before_revision)
        self.ts.activity_finance_feasible_target_production.append(activity_finance_feasible_target_production)
        self.ts.activity_finance_feasible_desired_labour_inputs.append(activity_finance_feasible_desired_labour_inputs)
        self.ts.activity_finance_feasibility_residual.append(activity_finance_feasibility_residual)
        self.ts.intermediate_purchase_finance_scale.append(intermediate_scale)
        self.ts.capital_purchase_finance_scale.append(capital_scale)
        self.ts.technical_investment_finance_scale.append(technical_scale)
        self.ts.tfp_investment_finance_scale.append(tfp_scale)
        self.ts.planned_intermediate_purchase_expected_costs.append(planned_intermediate_costs)
        self.ts.feasible_intermediate_purchase_expected_costs.append(feasible_intermediate_costs)
        self.ts.planned_capital_purchase_expected_costs.append(planned_capital_costs)
        self.ts.feasible_capital_purchase_expected_costs.append(feasible_capital_costs)
        self.ts.planned_technical_investment_expected_costs.append(planned_technical_costs)
        self.ts.feasible_technical_investment_expected_costs.append(feasible_technical_costs)
        self.ts.planned_tfp_investment_expected_costs.append(planned_tfp_costs)
        self.ts.feasible_tfp_investment_expected_costs.append(feasible_tfp_costs)

    @staticmethod
    def _allocate_activity_finance(
        remaining_finance: np.ndarray,
        planned_costs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        affordable_costs = np.minimum(remaining_finance, planned_costs)
        scale = np.divide(
            affordable_costs,
            planned_costs,
            out=np.ones_like(planned_costs),
            where=planned_costs > 0.0,
        )
        return np.maximum(0.0, remaining_finance - affordable_costs), scale

    def _activity_finance_tfp_for_feasible_labour(self) -> np.ndarray:
        tfp_multiplier = np.asarray(self.ts.current("tfp_multiplier"), dtype=float).copy()
        valid_tfp = np.isfinite(tfp_multiplier) & (tfp_multiplier > 1e-12)
        economy_positive_tfp = tfp_multiplier[valid_tfp]
        economy_fallback = np.median(economy_positive_tfp) if economy_positive_tfp.size > 0 else 1.0
        for industry in range(self.n_industries):
            industry_mask = self.states["Industry"] == industry
            industry_positive_tfp = tfp_multiplier[industry_mask & valid_tfp]
            if industry_positive_tfp.size > 0:
                tfp_multiplier[industry_mask & ~valid_tfp] = np.median(industry_positive_tfp)
        tfp_multiplier[~np.isfinite(tfp_multiplier) | (tfp_multiplier <= 1e-12)] = economy_fallback
        return tfp_multiplier

    def _activity_finance_feasible_labour(
        self,
        target_production: np.ndarray,
        tfp_multiplier: np.ndarray,
    ) -> np.ndarray:
        base_labour = np.divide(
            target_production,
            tfp_multiplier,
            out=target_production.copy(),
            where=tfp_multiplier > 0.0,
        )
        limiting_intermediate_inputs = np.asarray(self.ts.current("limiting_intermediate_inputs"), dtype=float).copy()
        invalid_intermediate = ~np.isfinite(limiting_intermediate_inputs)
        limiting_intermediate_inputs[invalid_intermediate] = base_labour[invalid_intermediate]
        limiting_intermediate_inputs = np.maximum(0.0, limiting_intermediate_inputs)
        limiting_capital_inputs = np.asarray(self.ts.current("limiting_capital_inputs"), dtype=float).copy()
        invalid_capital = ~np.isfinite(limiting_capital_inputs)
        limiting_capital_inputs[invalid_capital] = base_labour[invalid_capital]
        limiting_capital_inputs = np.maximum(0.0, limiting_capital_inputs)
        return self.functions["desired_labour"].compute_desired_labour(
            current_target_production=target_production,
            current_limiting_intermediate_inputs=limiting_intermediate_inputs,
            current_limiting_capital_inputs=limiting_capital_inputs,
            current_tfp_multiplier=tfp_multiplier,
        )

    def _activity_finance_wage_cost_rate(self, wage_obligation_preview: np.ndarray) -> np.ndarray:
        labour_inputs = np.asarray(self.ts.current("labour_inputs"), dtype=float)
        wage_preview = np.asarray(wage_obligation_preview, dtype=float)
        wage_rate = np.full_like(wage_preview, np.nan, dtype=float)
        valid = np.isfinite(wage_preview) & np.isfinite(labour_inputs) & (labour_inputs > 1e-12)
        wage_rate[valid] = wage_preview[valid] / labour_inputs[valid]
        economy_labour = labour_inputs[valid].sum()
        economy_fallback = wage_preview[valid].sum() / economy_labour if economy_labour > 0.0 else 0.0
        for industry in range(self.n_industries):
            industry_mask = self.states["Industry"] == industry
            industry_valid = industry_mask & valid
            industry_labour = labour_inputs[industry_valid].sum()
            if industry_labour > 0.0:
                wage_rate[industry_mask & ~valid] = wage_preview[industry_valid].sum() / industry_labour
        wage_rate[~np.isfinite(wage_rate)] = economy_fallback
        return np.maximum(0.0, wage_rate)

    def _activity_finance_candidate_inputs(
        self,
        target_production: np.ndarray,
        expected_lcu_prices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target_intermediate = self.functions[
            "target_intermediate_inputs"
        ].compute_unconstrained_target_intermediate_inputs(
            current_target_production=target_production,
            intermediate_inputs_productivity_matrix=self.get_effective_intermediate_coefficients(),
            prev_intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
            initial_intermediate_inputs_stock=self.ts.initial("intermediate_inputs_stock"),
            prev_production=target_production,
            initial_production=self.ts.initial("production"),
            previous_good_prices=expected_lcu_prices,
            substitution_bundle_matrix=self.substitution_bundles,
        )
        target_capital = self.functions["target_capital_inputs"].compute_unconstrained_target_capital_inputs(
            current_target_production=target_production,
            capital_input_use_matrix=self.base_capital_input_use_matrix[:, self.states["Industry"]].T,
            prev_capital_inputs_stock=self.ts.current("capital_inputs_stock"),
            initial_capital_inputs_stock=self.ts.initial("capital_inputs_stock"),
            prev_production=target_production,
            initial_production=self.ts.initial("production"),
            previous_good_prices=expected_lcu_prices,
            substitution_bundle_matrix=self.substitution_bundles,
        )
        intermediate_costs = (target_intermediate * expected_lcu_prices[None, :]).sum(axis=1)
        capital_costs = (target_capital * expected_lcu_prices[None, :]).sum(axis=1)
        return target_intermediate, target_capital, intermediate_costs, capital_costs

    @staticmethod
    def _activity_finance_investment_ratios(
        planned_tfp_costs: np.ndarray,
        planned_technical_costs: np.ndarray,
        planned_intermediate_costs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tfp_ratio = np.divide(
            planned_tfp_costs,
            planned_intermediate_costs,
            out=np.zeros_like(planned_tfp_costs),
            where=planned_intermediate_costs > 0.0,
        )
        technical_ratio = np.divide(
            planned_technical_costs,
            planned_intermediate_costs,
            out=np.zeros_like(planned_technical_costs),
            where=planned_intermediate_costs > 0.0,
        )
        return tfp_ratio, technical_ratio

    def revise_activity_against_available_finance(
        self,
        expected_lcu_prices: np.ndarray,
        wage_obligation_preview: np.ndarray,
        production_tax_obligation_preview: np.ndarray,
        corporate_tax_obligation_preview: np.ndarray | None = None,
        interest_obligation_preview: np.ndarray | None = None,
        loan_interest_obligation_preview: np.ndarray | None = None,
        debt_installment_preview: np.ndarray | None = None,
    ) -> None:
        """Revise current planned purchases after credit clears."""
        n_firms = self.ts.current("n_firms")
        mode = self.configuration.parameters.firm_activity_finance_revision_mode
        if corporate_tax_obligation_preview is None:
            corporate_tax_obligation_preview = np.zeros(n_firms)
        if interest_obligation_preview is None:
            interest_obligation_preview = self.ts.current("interest_paid")
        if loan_interest_obligation_preview is None:
            loan_interest_obligation_preview = np.zeros(n_firms)
        if debt_installment_preview is None:
            debt_installment_preview = self.ts.current("debt_installments")
        loan_debt_service_preview = loan_interest_obligation_preview + debt_installment_preview
        non_loan_interest_obligation_preview = np.maximum(
            0.0,
            interest_obligation_preview - loan_interest_obligation_preview,
        )
        received_debt_rollover_credit = np.maximum(
            0.0,
            np.nan_to_num(self.ts.current("received_debt_rollover_credit"), nan=0.0),
        )
        unfunded_loan_debt_service = np.maximum(0.0, loan_debt_service_preview - received_debt_rollover_credit)

        target_intermediate = self.ts.current("target_intermediate_inputs").copy()
        target_capital = self.ts.current("target_capital_inputs").copy()
        planned_technical = self.ts.current("planned_technical_investment").copy()
        planned_tfp = self.ts.current("planned_tfp_investment").copy()

        planned_intermediate_costs = (target_intermediate * expected_lcu_prices[None, :]).sum(axis=1)
        planned_capital_costs = (target_capital * expected_lcu_prices[None, :]).sum(axis=1)
        planned_technical_costs = planned_technical.sum(axis=1)
        planned_tfp_costs = planned_tfp.copy()
        total_planned_costs = (
            planned_intermediate_costs + planned_capital_costs + planned_technical_costs + planned_tfp_costs
        )

        if mode == "none":
            self._append_activity_finance_diagnostics(
                activity_finance_opening_deposits=self.ts.current("deposits"),
                activity_finance_available=np.zeros(n_firms),
                activity_finance_hard_obligations=np.zeros(n_firms),
                activity_finance_gap_before_revision=np.zeros(n_firms),
                activity_finance_feasible_target_production=self.ts.current("target_production"),
                activity_finance_feasible_desired_labour_inputs=self.ts.current("desired_labour_inputs"),
                activity_finance_feasibility_residual=np.zeros(n_firms),
                intermediate_scale=np.ones(n_firms),
                capital_scale=np.ones(n_firms),
                technical_scale=np.ones(n_firms),
                tfp_scale=np.ones(n_firms),
                planned_intermediate_costs=planned_intermediate_costs,
                feasible_intermediate_costs=planned_intermediate_costs,
                planned_capital_costs=planned_capital_costs,
                feasible_capital_costs=planned_capital_costs,
                planned_technical_costs=planned_technical_costs,
                feasible_technical_costs=planned_technical_costs,
                planned_tfp_costs=planned_tfp_costs,
                feasible_tfp_costs=planned_tfp_costs,
            )
            return

        if mode != "post_credit_cash_budget":
            raise ValueError(
                f"Unknown firm_activity_finance_revision_mode {mode!r}; expected 'none' or 'post_credit_cash_budget'."
            )

        opening_deposits = self.ts.current("deposits").copy()
        hard_obligations = non_loan_interest_obligation_preview + unfunded_loan_debt_service
        available_finance = np.maximum(
            0.0,
            opening_deposits
            + self._current_received_ordinary_short_term_credit()
            + np.nan_to_num(self.ts.current("received_long_term_credit"), nan=0.0)
            - hard_obligations,
        )
        y_high = np.maximum(0.0, np.nan_to_num(self.ts.current("target_production"), nan=0.0))
        tfp_for_labour = self._activity_finance_tfp_for_feasible_labour()
        wage_cost_rate = self._activity_finance_wage_cost_rate(wage_obligation_preview)
        full_bound_labour = self._activity_finance_feasible_labour(y_high, tfp_for_labour)
        full_bound_wage_costs = wage_cost_rate * full_bound_labour
        full_bound_costs = full_bound_wage_costs + total_planned_costs
        finance_gap = np.maximum(0.0, full_bound_costs - available_finance)

        constrained = full_bound_costs > available_finance + 1e-8
        feasible_y = y_high.copy()
        feasible_labour = full_bound_labour.copy()
        feasible_intermediate = target_intermediate.copy()
        feasible_capital = target_capital.copy()
        feasible_intermediate_costs = planned_intermediate_costs.copy()
        feasible_capital_costs = planned_capital_costs.copy()
        feasible_technical = planned_technical.copy()
        feasible_technical_costs = planned_technical_costs.copy()
        feasible_tfp = planned_tfp.copy()
        feasible_tfp_costs = planned_tfp_costs.copy()

        if np.any(constrained):
            tfp_ratio, technical_ratio = self._activity_finance_investment_ratios(
                planned_tfp_costs=planned_tfp_costs,
                planned_technical_costs=planned_technical_costs,
                planned_intermediate_costs=planned_intermediate_costs,
            )
            low = np.zeros(n_firms)
            high = y_high.copy()
            for _ in range(25):
                candidate = 0.5 * (low + high)
                _, _, candidate_intermediate_costs, candidate_capital_costs = self._activity_finance_candidate_inputs(
                    candidate,
                    expected_lcu_prices,
                )
                candidate_labour = self._activity_finance_feasible_labour(candidate, tfp_for_labour)
                candidate_costs = (
                    wage_cost_rate * candidate_labour
                    + (1.0 + tfp_ratio + technical_ratio) * candidate_intermediate_costs
                    + candidate_capital_costs
                )
                candidate_feasible = candidate_costs <= available_finance + 1e-8
                update_low = constrained & candidate_feasible
                update_high = constrained & ~candidate_feasible
                low = np.where(update_low, candidate, low)
                high = np.where(update_high, candidate, high)

            (
                candidate_intermediate,
                candidate_capital,
                candidate_intermediate_costs,
                candidate_capital_costs,
            ) = self._activity_finance_candidate_inputs(low, expected_lcu_prices)
            candidate_labour = self._activity_finance_feasible_labour(low, tfp_for_labour)
            candidate_tfp = tfp_ratio * candidate_intermediate_costs
            candidate_technical_costs = technical_ratio * candidate_intermediate_costs
            candidate_technical_scale = np.divide(
                candidate_technical_costs,
                planned_technical_costs,
                out=np.zeros_like(planned_technical_costs),
                where=planned_technical_costs > 0.0,
            )
            candidate_technical = planned_technical * candidate_technical_scale[:, None]

            feasible_y = np.where(constrained, low, feasible_y)
            feasible_labour = np.where(constrained, candidate_labour, feasible_labour)
            feasible_intermediate = np.where(constrained[:, None], candidate_intermediate, feasible_intermediate)
            feasible_capital = np.where(constrained[:, None], candidate_capital, feasible_capital)
            feasible_intermediate_costs = np.where(
                constrained,
                candidate_intermediate_costs,
                feasible_intermediate_costs,
            )
            feasible_capital_costs = np.where(constrained, candidate_capital_costs, feasible_capital_costs)
            feasible_technical = np.where(constrained[:, None], candidate_technical, feasible_technical)
            feasible_technical_costs = np.where(constrained, candidate_technical_costs, feasible_technical_costs)
            feasible_tfp = np.where(constrained, candidate_tfp, feasible_tfp)
            feasible_tfp_costs = feasible_tfp.copy()

        feasible_activity_costs = (
            wage_cost_rate * feasible_labour
            + feasible_intermediate_costs
            + feasible_capital_costs
            + feasible_technical_costs
            + feasible_tfp_costs
        )
        feasibility_residual = available_finance - feasible_activity_costs
        intermediate_scale = np.divide(
            feasible_intermediate_costs,
            planned_intermediate_costs,
            out=np.ones_like(planned_intermediate_costs),
            where=planned_intermediate_costs > 0.0,
        )
        capital_scale = np.divide(
            feasible_capital_costs,
            planned_capital_costs,
            out=np.ones_like(planned_capital_costs),
            where=planned_capital_costs > 0.0,
        )
        technical_scale = np.divide(
            feasible_technical_costs,
            planned_technical_costs,
            out=np.ones_like(planned_technical_costs),
            where=planned_technical_costs > 0.0,
        )
        tfp_scale = np.divide(
            feasible_tfp_costs,
            planned_tfp_costs,
            out=np.ones_like(planned_tfp_costs),
            where=planned_tfp_costs > 0.0,
        )

        self.ts.override_current("target_intermediate_inputs", feasible_intermediate)
        self.ts.override_current("target_capital_inputs", feasible_capital)
        self.ts.override_current("planned_technical_investment", feasible_technical)
        self.ts.override_current("planned_tfp_investment", feasible_tfp)
        self.ts.override_current("planned_productivity_investment", feasible_tfp + feasible_technical.sum(axis=1))

        self._append_activity_finance_diagnostics(
            activity_finance_opening_deposits=opening_deposits,
            activity_finance_available=available_finance,
            activity_finance_hard_obligations=hard_obligations,
            activity_finance_gap_before_revision=finance_gap,
            activity_finance_feasible_target_production=feasible_y,
            activity_finance_feasible_desired_labour_inputs=feasible_labour,
            activity_finance_feasibility_residual=feasibility_residual,
            intermediate_scale=intermediate_scale,
            capital_scale=capital_scale,
            technical_scale=technical_scale,
            tfp_scale=tfp_scale,
            planned_intermediate_costs=planned_intermediate_costs,
            feasible_intermediate_costs=feasible_intermediate_costs,
            planned_capital_costs=planned_capital_costs,
            feasible_capital_costs=feasible_capital_costs,
            planned_technical_costs=planned_technical_costs,
            feasible_technical_costs=feasible_technical_costs,
            planned_tfp_costs=planned_tfp_costs,
            feasible_tfp_costs=feasible_tfp_costs,
        )

    @staticmethod
    def _post_labour_output_from_realised_labour(
        feasible_y: np.ndarray,
        feasible_labour: np.ndarray,
        realised_labour: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return proportional output scale implied by realised effective labour."""
        labour_scale = np.divide(
            realised_labour,
            feasible_labour,
            out=np.ones_like(feasible_labour),
            where=feasible_labour > 0.0,
        )
        labour_scale = np.clip(labour_scale, 0.0, 1.0)
        return feasible_y * labour_scale, labour_scale

    def revise_activity_against_realised_labour(self, expected_lcu_prices: np.ndarray) -> None:
        """Revise current activity plans after labour clearing with effective labour inputs."""
        feasible_y = np.maximum(
            0.0,
            np.nan_to_num(self.ts.current("activity_finance_feasible_target_production"), nan=0.0),
        )
        feasible_labour = np.maximum(
            0.0,
            np.nan_to_num(self.ts.current("activity_finance_feasible_desired_labour_inputs"), nan=0.0),
        )
        realised_labour = np.maximum(0.0, np.nan_to_num(self.ts.current("labour_inputs"), nan=0.0))
        realised_feasible_y, labour_scale = self._post_labour_output_from_realised_labour(
            feasible_y=feasible_y,
            feasible_labour=feasible_labour,
            realised_labour=realised_labour,
        )
        candidate_intermediate, candidate_capital, _, _ = self._activity_finance_candidate_inputs(
            realised_feasible_y,
            expected_lcu_prices,
        )

        self.ts.override_current("target_production", realised_feasible_y)
        self.ts.override_current("target_intermediate_inputs", candidate_intermediate)
        self.ts.override_current("target_capital_inputs", candidate_capital)
        self.ts.override_current(
            "planned_productivity_investment",
            self.ts.current("planned_tfp_investment") + self.ts.current("planned_technical_investment").sum(axis=1),
        )

        self.ts.activity_finance_realised_feasible_target_production.append(realised_feasible_y)
        self.ts.activity_finance_realised_labour_scale.append(labour_scale)

    def prepare_feasible_activity_plan(
        self,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
        assume_zero_growth: bool = False,
        wage_obligation_preview: np.ndarray | None = None,
        production_tax_obligation_preview: np.ndarray | None = None,
        corporate_tax_obligation_preview: np.ndarray | None = None,
        interest_obligation_preview: np.ndarray | None = None,
        loan_interest_obligation_preview: np.ndarray | None = None,
        debt_installment_preview: np.ndarray | None = None,
    ) -> None:
        """Append target purchases and revise them against post-credit finance.

        This is the activity-planning part of goods preparation. Stage 2 runs it
        after credit clears and before labour clearing so feasible labour demand
        can become the operative labour-market demand.

        Args:
            previous_good_prices (np.ndarray): Previous period prices
            expected_inflation (float): Expected inflation rate
            assume_zero_growth (bool, optional): Whether to assume no growth. Defaults to False.
            wage_obligation_preview (np.ndarray | None): Non-appending current wage obligation preview.
            production_tax_obligation_preview (np.ndarray | None): Non-appending current production-tax preview.
            corporate_tax_obligation_preview (np.ndarray | None): Estimated current corporate-tax obligation.
            interest_obligation_preview (np.ndarray | None): Current-period interest cash obligation preview.
            loan_interest_obligation_preview (np.ndarray | None): Current-period loan interest eligible for rollover.
            debt_installment_preview (np.ndarray | None): Current-period scheduled firm principal due preview.
        """
        # Target intermediate inputs
        if assume_zero_growth:
            self.ts.target_intermediate_inputs.append(self.ts.initial("target_intermediate_inputs"))
        elif self.configuration.parameters.firm_activity_finance_revision_mode == "post_credit_cash_budget":
            self.ts.target_intermediate_inputs.append(self.ts.current("unconstrained_target_intermediate_inputs"))
        else:
            self.ts.target_intermediate_inputs.append(
                self.functions["target_intermediate_inputs"].compute_target_intermediate_inputs(
                    unconstrained_target_intermediate_inputs=self.ts.current(
                        "unconstrained_target_intermediate_inputs"
                    ),
                    target_short_term_credit=self.ts.current("target_short_term_credit"),
                    received_short_term_credit=self.ts.current("received_short_term_credit"),
                    previous_good_prices=previous_good_prices,
                    expected_inflation=expected_inflation,
                )
            )

        # Target capital inputs
        if assume_zero_growth:
            self.ts.target_capital_inputs.append(self.ts.initial("target_capital_inputs"))
        elif self.configuration.parameters.firm_activity_finance_revision_mode == "post_credit_cash_budget":
            self.ts.target_capital_inputs.append(self.ts.current("unconstrained_target_capital_inputs"))
        else:
            self.ts.target_capital_inputs.append(
                self.functions["target_capital_inputs"].compute_target_capital_inputs(
                    unconstrained_target_capital_inputs=self.ts.current("unconstrained_target_capital_inputs"),
                    target_long_term_credit=self.ts.current("target_long_term_credit"),
                    received_long_term_credit=self.ts.current("received_long_term_credit"),
                    previous_good_prices=previous_good_prices,
                    expected_inflation=expected_inflation,
                )
            )

        if wage_obligation_preview is None:
            wage_obligation_preview = np.zeros(self.ts.current("n_firms"))
        if production_tax_obligation_preview is None:
            production_tax_obligation_preview = np.zeros(self.ts.current("n_firms"))
        if corporate_tax_obligation_preview is None:
            corporate_tax_obligation_preview = np.zeros(self.ts.current("n_firms"))
        expected_lcu_prices = (1 + expected_inflation) * previous_good_prices
        self.revise_activity_against_available_finance(
            expected_lcu_prices=expected_lcu_prices,
            wage_obligation_preview=wage_obligation_preview,
            production_tax_obligation_preview=production_tax_obligation_preview,
            corporate_tax_obligation_preview=corporate_tax_obligation_preview,
            interest_obligation_preview=interest_obligation_preview,
            loan_interest_obligation_preview=loan_interest_obligation_preview,
            debt_installment_preview=debt_installment_preview,
        )

    def apply_feasible_labour_demand(self) -> None:
        """Use post-credit feasible labour demand as the operative labour target."""
        self.ts.override_current(
            "desired_labour_inputs",
            self.ts.current("activity_finance_feasible_desired_labour_inputs").copy(),
        )

    def _excess_demand_finance_candidate_costs(
        self,
        candidate_output: np.ndarray,
        expected_lcu_prices: np.ndarray,
        wage_cost_rate: np.ndarray,
        tfp_for_labour: np.ndarray,
    ) -> np.ndarray:
        """Return diagnostic activity costs for candidate output levels."""
        _, _, intermediate_costs, capital_costs = self._activity_finance_candidate_inputs(
            candidate_output,
            expected_lcu_prices,
        )
        candidate_labour = self._activity_finance_feasible_labour(candidate_output, tfp_for_labour)
        costs = wage_cost_rate * candidate_labour + intermediate_costs + capital_costs
        return np.where(np.isfinite(costs), np.maximum(0.0, costs), np.inf)

    def compute_excess_demand_finance_potential_output(
        self,
        activity_finance_borrower: np.ndarray,
        expected_lcu_prices: np.ndarray,
        wage_obligation_preview: np.ndarray,
    ) -> np.ndarray:
        """Solve borrower-side finance-potential output for the excess-demand diagnostic."""
        n_firms = self.ts.current("n_firms")
        finance = np.maximum(0.0, np.nan_to_num(np.asarray(activity_finance_borrower, dtype=float), nan=0.0))
        current_target = np.maximum(0.0, np.nan_to_num(self.ts.current("target_production"), nan=0.0))
        high = np.maximum(2.0 * current_target, np.ones(n_firms))
        tfp_for_labour = self._activity_finance_tfp_for_feasible_labour()
        wage_cost_rate = self._activity_finance_wage_cost_rate(wage_obligation_preview)

        for _ in range(20):
            high_costs = self._excess_demand_finance_candidate_costs(
                candidate_output=high,
                expected_lcu_prices=expected_lcu_prices,
                wage_cost_rate=wage_cost_rate,
                tfp_for_labour=tfp_for_labour,
            )
            high = np.where(high_costs <= finance + 1e-8, 2.0 * high, high)

        low = np.zeros(n_firms)
        for _ in range(30):
            candidate = 0.5 * (low + high)
            candidate_costs = self._excess_demand_finance_candidate_costs(
                candidate_output=candidate,
                expected_lcu_prices=expected_lcu_prices,
                wage_cost_rate=wage_cost_rate,
                tfp_for_labour=tfp_for_labour,
            )
            feasible = candidate_costs <= finance + 1e-8
            low = np.where(feasible, candidate, low)
            high = np.where(feasible, high, candidate)

        return np.maximum(0.0, low)

    def reset_excess_demand_finance_potential_transients(self) -> None:
        """Clear current-period excess-demand capacity values used by the replacement allocator."""
        self._last_excess_demand_finance_potential = None

    def compute_excess_demand_finance_potential_capacities(
        self,
        expected_lcu_prices: np.ndarray,
        wage_obligation_preview: np.ndarray,
        non_loan_interest_obligation_preview: np.ndarray,
        borrower_st_credit_room: np.ndarray,
        borrower_lt_credit_room: np.ndarray,
        borrower_total_credit_room: np.ndarray,
        q_sold: np.ndarray,
        ordinary_bank_supply: float | None = None,
    ) -> None:
        """Compute current-period finance-potential capacities for excess-demand assignment."""
        n_firms = self.ts.current("n_firms")
        opening_deposits = np.nan_to_num(
            np.asarray(self.ts.current("activity_finance_opening_deposits"), dtype=float),
            nan=0.0,
        )
        non_loan_interest = np.maximum(
            0.0,
            np.nan_to_num(np.asarray(non_loan_interest_obligation_preview, dtype=float), nan=0.0),
        )
        finance_cash = np.maximum(0.0, opening_deposits - non_loan_interest)
        repair_cash_used = np.zeros(n_firms)
        residual_repair_credit_need = np.maximum(
            0.0, np.nan_to_num(self.ts.current("target_debt_rollover_credit"), nan=0.0)
        ) + np.maximum(0.0, np.nan_to_num(self.ts.current("target_overdraft_refinance_credit"), nan=0.0))
        borrower_st_credit_room = np.maximum(
            0.0,
            np.nan_to_num(np.asarray(borrower_st_credit_room, dtype=float), nan=0.0),
        )
        borrower_lt_credit_room = np.maximum(
            0.0,
            np.nan_to_num(np.asarray(borrower_lt_credit_room, dtype=float), nan=0.0),
        )
        borrower_total_credit_room = np.maximum(
            0.0,
            np.nan_to_num(np.asarray(borrower_total_credit_room, dtype=float), nan=0.0),
        )
        borrower_max_credit = np.maximum(0.0, borrower_total_credit_room - residual_repair_credit_need)
        activity_finance_borrower = finance_cash + borrower_max_credit
        finance_potential_output_borrower = self.compute_excess_demand_finance_potential_output(
            activity_finance_borrower=activity_finance_borrower,
            expected_lcu_prices=expected_lcu_prices,
            wage_obligation_preview=wage_obligation_preview,
        )
        q_sold = np.maximum(0.0, np.nan_to_num(np.asarray(q_sold, dtype=float), nan=0.0))
        potential_capacity_borrower = np.maximum(0.0, finance_potential_output_borrower - q_sold)

        supply_max_credit = None
        activity_finance_supply = None
        finance_potential_output_supply = None
        potential_capacity_supply = None
        if ordinary_bank_supply is not None:
            ordinary_bank_supply = float(ordinary_bank_supply)
            if not np.isfinite(ordinary_bank_supply) or ordinary_bank_supply < 0.0:
                raise ValueError("ordinary_bank_supply must be finite and non-negative")
            borrower_total = float(borrower_max_credit.sum())
            if borrower_total <= 0.0:
                supply_max_credit = np.zeros(n_firms)
            else:
                supply_ratio = min(1.0, ordinary_bank_supply / borrower_total)
                supply_max_credit = borrower_max_credit * supply_ratio
            activity_finance_supply = finance_cash + supply_max_credit
            finance_potential_output_supply = self.compute_excess_demand_finance_potential_output(
                activity_finance_borrower=activity_finance_supply,
                expected_lcu_prices=expected_lcu_prices,
                wage_obligation_preview=wage_obligation_preview,
            )
            potential_capacity_supply = np.maximum(0.0, finance_potential_output_supply - q_sold)

        self._last_excess_demand_finance_potential = {
            "finance_cash": finance_cash,
            "borrower_st_credit_room": borrower_st_credit_room,
            "borrower_lt_credit_room": borrower_lt_credit_room,
            "borrower_total_credit_room": borrower_total_credit_room,
            "repair_cash_used": repair_cash_used,
            "residual_repair_credit_need": residual_repair_credit_need,
            "borrower_max_credit": borrower_max_credit,
            "activity_finance_borrower": activity_finance_borrower,
            "finance_potential_output_borrower": finance_potential_output_borrower,
            "potential_capacity_borrower": potential_capacity_borrower,
            "supply_max_credit": supply_max_credit,
            "activity_finance_supply": activity_finance_supply,
            "finance_potential_output_supply": finance_potential_output_supply,
            "potential_capacity_supply": potential_capacity_supply,
        }

    def get_supply_adjusted_excess_demand_capacity(self) -> np.ndarray:
        """Return the current-period supply-adjusted capacity for the replacement allocator."""
        cache = getattr(self, "_last_excess_demand_finance_potential", None)
        if cache is None or cache.get("potential_capacity_supply") is None:
            raise ValueError("Supply-adjusted excess-demand capacity has not been computed")
        capacity = np.asarray(cache["potential_capacity_supply"], dtype=float)
        if capacity.shape != (self.ts.current("n_firms"),):
            raise ValueError("Supply-adjusted excess-demand capacity has the wrong shape")
        if np.any(~np.isfinite(capacity)) or np.any(capacity < 0.0):
            raise ValueError("Supply-adjusted excess-demand capacity must be finite and non-negative")
        return capacity

    def append_cached_excess_demand_finance_potential_diagnostics(self, q_excess: np.ndarray) -> None:
        """Append Increment 1 diagnostics using current-period cached capacity inputs."""
        cache = getattr(self, "_last_excess_demand_finance_potential", None)
        if cache is None:
            raise ValueError("Excess-demand finance-potential diagnostics have not been computed")
        n_firms = self.ts.current("n_firms")
        q_excess = np.maximum(0.0, np.nan_to_num(np.asarray(q_excess, dtype=float), nan=0.0))
        above_cap_share = np.divide(
            np.maximum(0.0, q_excess - cache["potential_capacity_borrower"]),
            q_excess,
            out=np.full(n_firms, np.nan),
            where=q_excess > 0.0,
        )
        potential_capacity_supply = cache["potential_capacity_supply"]
        if potential_capacity_supply is None:
            supply_max_credit = np.full(n_firms, np.nan)
            activity_finance_supply = np.full(n_firms, np.nan)
            finance_potential_output_supply = np.full(n_firms, np.nan)
            potential_capacity_supply = np.full(n_firms, np.nan)
            above_supply_cap_share = np.full(n_firms, np.nan)
        else:
            supply_max_credit = cache["supply_max_credit"]
            activity_finance_supply = cache["activity_finance_supply"]
            finance_potential_output_supply = cache["finance_potential_output_supply"]
            above_supply_cap_share = np.divide(
                np.maximum(0.0, q_excess - potential_capacity_supply),
                q_excess,
                out=np.full(n_firms, np.nan),
                where=q_excess > 0.0,
            )

        self.ts.excess_demand_finance_cash.append(cache["finance_cash"])
        self.ts.excess_demand_borrower_st_credit_room.append(cache["borrower_st_credit_room"])
        self.ts.excess_demand_borrower_lt_credit_room.append(cache["borrower_lt_credit_room"])
        self.ts.excess_demand_borrower_total_credit_room.append(cache["borrower_total_credit_room"])
        self.ts.excess_demand_repair_cash_used.append(cache["repair_cash_used"])
        self.ts.excess_demand_residual_repair_credit_need.append(cache["residual_repair_credit_need"])
        self.ts.excess_demand_borrower_max_credit.append(cache["borrower_max_credit"])
        self.ts.excess_demand_activity_finance_borrower.append(cache["activity_finance_borrower"])
        self.ts.excess_demand_finance_potential_output_borrower.append(cache["finance_potential_output_borrower"])
        self.ts.excess_demand_potential_capacity_borrower.append(cache["potential_capacity_borrower"])
        self.ts.excess_demand_above_borrower_cap_share.append(above_cap_share)
        self.ts.excess_demand_supply_max_credit.append(supply_max_credit)
        self.ts.excess_demand_activity_finance_supply.append(activity_finance_supply)
        self.ts.excess_demand_finance_potential_output_supply.append(finance_potential_output_supply)
        self.ts.excess_demand_potential_capacity_supply.append(potential_capacity_supply)
        self.ts.excess_demand_above_supply_cap_share.append(above_supply_cap_share)
        self.reset_excess_demand_finance_potential_transients()

    def append_excess_demand_finance_potential_diagnostics(
        self,
        expected_lcu_prices: np.ndarray,
        wage_obligation_preview: np.ndarray,
        non_loan_interest_obligation_preview: np.ndarray,
        borrower_st_credit_room: np.ndarray,
        borrower_lt_credit_room: np.ndarray,
        borrower_total_credit_room: np.ndarray,
        q_sold: np.ndarray,
        q_excess: np.ndarray,
    ) -> None:
        """Append Increment 1 borrower-side excess-demand finance diagnostics."""
        self.reset_excess_demand_finance_potential_transients()
        self.compute_excess_demand_finance_potential_capacities(
            expected_lcu_prices=expected_lcu_prices,
            wage_obligation_preview=wage_obligation_preview,
            non_loan_interest_obligation_preview=non_loan_interest_obligation_preview,
            borrower_st_credit_room=borrower_st_credit_room,
            borrower_lt_credit_room=borrower_lt_credit_room,
            borrower_total_credit_room=borrower_total_credit_room,
            q_sold=q_sold,
        )
        self.append_cached_excess_demand_finance_potential_diagnostics(q_excess=q_excess)

    def set_goods_to_buy_from_current_targets(
        self,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> None:
        """Set goods orders from already-prepared current activity targets."""
        # Setting total real amount of goods to buy
        total_goods = self.ts.current("target_intermediate_inputs") + self.ts.current("target_capital_inputs")
        # Include planned technical investment if it exists
        expected_lcu_prices = (1 + expected_inflation) * previous_good_prices
        if hasattr(self.ts, "planned_technical_investment") and len(self.ts.planned_technical_investment) > 0:
            technical_goods = np.divide(
                self.ts.current("planned_technical_investment"),
                expected_lcu_prices[None, :],
                out=np.zeros_like(self.ts.current("planned_technical_investment")),
                where=expected_lcu_prices[None, :] > 0.0,
            )
            total_goods = total_goods + technical_goods
        self.set_goods_to_buy(total_goods)

    def prepare_buying_goods(
        self,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
        assume_zero_growth: bool = False,
        wage_obligation_preview: np.ndarray | None = None,
        production_tax_obligation_preview: np.ndarray | None = None,
        corporate_tax_obligation_preview: np.ndarray | None = None,
        interest_obligation_preview: np.ndarray | None = None,
        loan_interest_obligation_preview: np.ndarray | None = None,
        debt_installment_preview: np.ndarray | None = None,
    ) -> None:
        """Prepare firms' buying plans for goods market."""
        self.prepare_feasible_activity_plan(
            previous_good_prices=previous_good_prices,
            expected_inflation=expected_inflation,
            assume_zero_growth=assume_zero_growth,
            wage_obligation_preview=wage_obligation_preview,
            production_tax_obligation_preview=production_tax_obligation_preview,
            corporate_tax_obligation_preview=corporate_tax_obligation_preview,
            interest_obligation_preview=interest_obligation_preview,
            loan_interest_obligation_preview=loan_interest_obligation_preview,
            debt_installment_preview=debt_installment_preview,
        )
        self.set_goods_to_buy_from_current_targets(
            previous_good_prices=previous_good_prices,
            expected_inflation=expected_inflation,
        )

    def prepare_selling_goods(self) -> None:
        """Prepare firms' selling plans for goods market.

        Sets up:
        - Quantities to sell (production + inventory)
        - Prices in USD
        - Industry classifications
        - Maximum excess demand
        """
        self.set_goods_to_sell(self.ts.current("production") + self.ts.current("inventory"))
        self.ts.price_in_usd.append(1.0 / self.exchange_rate_usd_to_lcu * self.ts.current("price"))
        self.ts.price_offered.append(self.compute_offered_price())
        self.set_prices(self.ts.current("price_in_usd"))
        self.set_seller_industries(self.states["Industry"])
        self.set_maximum_excess_demand(self.compute_maximum_excess_demand())

    def prepare_goods_market_clearing(
        self,
        exchange_rate_usd_to_lcu: float,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
        wage_obligation_preview: np.ndarray | None = None,
        production_tax_obligation_preview: np.ndarray | None = None,
        corporate_tax_obligation_preview: np.ndarray | None = None,
        interest_obligation_preview: np.ndarray | None = None,
        loan_interest_obligation_preview: np.ndarray | None = None,
        debt_installment_preview: np.ndarray | None = None,
        assume_zero_growth: bool = False,
    ) -> None:
        """Prepare all aspects of goods market participation.

        Coordinates:
        - Exchange rate setting
        - Buying plans
        - Selling plans

        Args:
            exchange_rate_usd_to_lcu (float): Exchange rate USD to local currency
            previous_good_prices (np.ndarray): Previous period prices
            expected_inflation (float): Expected inflation rate
            wage_obligation_preview (np.ndarray | None): Non-appending current wage obligation preview.
            production_tax_obligation_preview (np.ndarray | None): Non-appending current production-tax preview.
            corporate_tax_obligation_preview (np.ndarray | None): Estimated current corporate-tax obligation.
            interest_obligation_preview (np.ndarray | None): Current-period interest cash obligation preview.
            loan_interest_obligation_preview (np.ndarray | None): Current-period loan interest eligible for rollover.
            debt_installment_preview (np.ndarray | None): Current-period scheduled firm principal due preview.
            assume_zero_growth (bool): Whether to use initial buying targets.
        """
        self.set_exchange_rate(exchange_rate_usd_to_lcu)
        self.prepare_buying_goods(
            previous_good_prices=previous_good_prices,
            expected_inflation=expected_inflation,
            wage_obligation_preview=wage_obligation_preview,
            production_tax_obligation_preview=production_tax_obligation_preview,
            corporate_tax_obligation_preview=corporate_tax_obligation_preview,
            interest_obligation_preview=interest_obligation_preview,
            loan_interest_obligation_preview=loan_interest_obligation_preview,
            debt_installment_preview=debt_installment_preview,
            assume_zero_growth=assume_zero_growth,
        )
        self.prepare_selling_goods()

    def prepare_goods_market_orders(
        self,
        exchange_rate_usd_to_lcu: float,
        previous_good_prices: np.ndarray,
        expected_inflation: float,
    ) -> None:
        """Prepare goods-market orders from the already revised activity plan."""
        self.set_exchange_rate(exchange_rate_usd_to_lcu)
        self.set_goods_to_buy_from_current_targets(
            previous_good_prices=previous_good_prices,
            expected_inflation=expected_inflation,
        )
        self.prepare_selling_goods()

    def distribute_bought_goods(self) -> None:
        """Distribute purchased goods between intermediate and capital inputs.

        Allocates actual purchases based on:
        - Desired intermediate inputs
        - Desired investment
        - Actual quantities bought
        """
        (
            new_intermediate_inputs,
            new_capital_inputs,
        ) = self.functions["bought_goods_distributor"].distribute_bought_goods(
            desired_intermediate_inputs=self.ts.current("target_intermediate_inputs"),
            desired_investment=self.ts.current("target_capital_inputs"),
            buy_real=self.ts.current("real_amount_bought"),
        )
        self.ts.real_amount_bought_as_intermediate_inputs.append(new_intermediate_inputs)
        self.ts.real_amount_bought_as_capital_goods.append(new_capital_inputs)
        """
        print(
            "DBG",
            list(
                (
                    (self.ts.current("target_intermediate_inputs") - new_intermediate_inputs).sum(axis=0)
                    + (self.ts.current("target_capital_inputs") - new_capital_inputs).sum(axis=0)
                ).round(2)
            ),
        )
        """

    def compute_gross_fixed_capital_formation(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate gross fixed capital formation.

        Computes value of capital goods purchases at current prices.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Value of capital formation by industry
        """
        return (self.ts.current("real_amount_bought_as_capital_goods") * current_good_prices).sum(axis=0)

    def update_total_newly_bought_costs(self, current_good_prices: np.ndarray) -> None:
        """Update costs of newly purchased inputs.

        Allocates purchase costs between:
        - Intermediate inputs
        - Capital inputs
        Based on relative quantities at current prices.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation
        """
        self.current_good_prices = np.asarray(current_good_prices, dtype=float)
        amount_ii = (self.ts.current("real_amount_bought_as_intermediate_inputs") * current_good_prices).sum(axis=1)
        amount_cap = (self.ts.current("real_amount_bought_as_capital_goods") * current_good_prices).sum(axis=1)

        # Just take fractions
        self.ts.total_intermediate_inputs_bought_costs.append(
            self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            * np.divide(
                amount_ii,
                amount_ii + amount_cap,
                out=np.zeros(amount_ii.shape),
                where=amount_ii + amount_cap != 0,
            )
        )
        self.ts.total_capital_inputs_bought_costs.append(
            self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            - self.ts.current("total_intermediate_inputs_bought_costs")
        )

    def compute_demand(self) -> np.ndarray:
        """Calculate realized demand for each firm's output.

        Combines:
        - Actual sales
        - Excess demand

        Returns:
            np.ndarray: Total demand for each firm
        """
        return self.functions["demand_for_goods"].compute_demand(
            sell_real=self.ts.current("real_amount_sold"),
            excess_demand=self.ts.current("real_excess_demand"),
        )

    def compute_nominal_production(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate nominal value of production.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Nominal production value for each firm
        """
        return current_good_prices[self.states["Industry"]] * self.ts.current("production")

    def compute_inventory(self) -> np.ndarray:
        """Calculate end-of-period inventory levels.

        Considers:
        - Previous inventory
        - Current production
        - Sales
        - Depreciation

        Returns:
            np.ndarray: Updated inventory levels for each firm
        """
        return (1 - np.array(self.depreciation_rates)[self.states["Industry"]]) * np.maximum(
            0.0,
            self.ts.current("inventory") + self.ts.current("production") - self.ts.current("real_amount_sold"),
        )

    def compute_nominal_inventory(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate nominal value of inventories.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Nominal inventory value for each firm
        """
        return current_good_prices[self.states["Industry"]] * self.ts.current("inventory")

    def compute_used_intermediate_inputs(self):
        """Calculate intermediate inputs used in production.

        Determines actual intermediate input usage based on:
        - Realized production
        - Input-output coefficients
        - Available input stocks
        - Input criticality requirements

        Returns:
            np.ndarray: Used intermediate inputs for each firm
        """
        return self.functions["production"].compute_intermediate_inputs_used(
            realised_production=self.ts.current("production"),
            intermediate_inputs_productivity_matrix=self.get_effective_intermediate_coefficients(),
            intermediate_inputs_stock=self.ts.current("intermediate_inputs_stock"),
            goods_criticality_matrix=self.current_goods_criticality_by_firm(),
            substitution_bundle_matrix=self.substitution_bundles,
        )

    def compute_used_intermediate_inputs_costs(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate cost of intermediate inputs used in production.

        Args:
            current_good_prices (np.ndarray): Current prices for inputs

        Returns:
            np.ndarray: Cost of used intermediate inputs for each firm
        """
        return (self.ts.current("used_intermediate_inputs") * current_good_prices).sum(axis=1)

    def compute_intermediate_inputs_stock(self) -> np.ndarray:
        """Calculate end-of-period intermediate input stocks.

        Updates stocks based on:
        - Previous stocks
        - Inputs used in production
        - New purchases

        Returns:
            np.ndarray: Updated intermediate input stocks
        """
        return np.maximum(
            0.0,
            self.ts.current("intermediate_inputs_stock")
            - self.ts.current("used_intermediate_inputs")
            + self.ts.current("real_amount_bought_as_intermediate_inputs"),
        )

    def compute_intermediate_inputs_stock_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate value of intermediate input stocks.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Value of intermediate input stocks for each firm
        """
        return (self.ts.current("intermediate_inputs_stock") * current_good_prices).sum(axis=1)

    def compute_used_capital_inputs(self):
        """Calculate capital inputs used in production.

        Determines actual capital input usage based on:
        - Realized production
        - Capital productivity coefficients
        - Available capital stocks
        - Input criticality requirements

        Returns:
            np.ndarray: Used capital inputs for each firm
        """
        return self.functions["production"].compute_capital_inputs_used(
            realised_production=self.ts.current("production"),
            capital_input_use_matrix=self.base_capital_input_use_matrix[:, self.states["Industry"]].T,
            capital_inputs_stock=self.ts.current("capital_inputs_stock"),
            goods_criticality_matrix=self.current_goods_criticality_by_firm(),
            substitution_bundle_matrix=self.substitution_bundles,
        )

    def compute_used_capital_inputs_costs(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate cost of capital inputs used in production.

        Args:
            current_good_prices (np.ndarray): Current prices for inputs

        Returns:
            np.ndarray: Cost of used capital inputs for each firm
        """
        return (self.ts.current("used_capital_inputs") * current_good_prices).sum(axis=1)

    def compute_expected_capital_inputs_stock_value(
        self,
        current_good_prices: np.ndarray,
        estimated_inflation: float,
    ) -> np.ndarray:
        """Calculate expected future value of capital input stocks.

        Estimates future value considering:
        - Current stock levels
        - Current prices
        - Expected inflation

        Args:
            current_good_prices (np.ndarray): Current prices for valuation
            estimated_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Expected value of capital input stocks for each firm
        """
        return (1 + estimated_inflation) * (self.ts.current("capital_inputs_stock") * current_good_prices).sum(axis=1)

    def compute_capital_inputs_stock(self) -> np.ndarray:
        """Calculate end-of-period capital input stocks.

        Updates stocks considering:
        - Previous stocks
        - Used inputs
        - New purchases
        - Implementation delays

        Returns:
            np.ndarray: Updated capital input stocks
        """
        hist_bought_capital = np.array(self.ts.historic("real_amount_bought_as_capital_goods")[1:])
        delayed_bought_capital = np.zeros((self.ts.current("n_firms"), self.n_industries))
        for g in range(self.n_industries):
            delay = self.capital_inputs_delay[g]
            if delay < hist_bought_capital.shape[0]:
                delayed_bought_capital[:, g] = hist_bought_capital[-delay - 1, :, g]

        return np.maximum(
            0.0,
            self.ts.current("capital_inputs_stock") - self.ts.current("used_capital_inputs") + delayed_bought_capital,
        )

    def compute_capital_inputs_stock_value(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate value of capital input stocks.

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Value of capital input stocks for each firm
        """
        return (self.ts.current("capital_inputs_stock") * current_good_prices).sum(axis=1)

    def compute_capital_depreciation_costs(self) -> np.ndarray:
        """Calculate non-cash CFC depreciation costs for firm accounting."""
        mode = self.configuration.parameters.capital_depreciation_accounting_mode
        if mode == "none":
            return np.zeros(self.ts.current("n_firms"))
        if mode == "eurostat_cfc":
            rates = self.capital_depreciation_rates[self.states["Industry"]]
            return rates * self.ts.current("production") * self.ts.current("price")
        raise ValueError(f"Unknown capital_depreciation_accounting_mode {mode!r}; expected 'none' or 'eurostat_cfc'.")

    def compute_total_inventory_change(self) -> np.ndarray:
        """Calculate nominal change in inventory value.

        Returns:
            np.ndarray: Change in inventory value for each firm
        """
        return self.ts.current("price") * (self.ts.current("inventory") - self.ts.prev("inventory"))

    def compute_taxes_paid_on_production(self, taxes_less_subsidies_rates: np.ndarray) -> np.ndarray:
        """Calculate taxes paid on production.

        Args:
            taxes_less_subsidies_rates (np.ndarray): Net tax rates by industry

        Returns:
            np.ndarray: Production taxes paid by each firm
        """
        return (
            taxes_less_subsidies_rates[self.states["Industry"]]
            * self.ts.current("production")
            * self.ts.current("price")
        )

    def compute_profits(self) -> np.ndarray:
        """Calculate profits for each firm.

        In ``production_cost`` accounting mode, the capital-compensation-derived
        ``used_capital_inputs_costs`` are charged as a period production cost.
        In ``surplus_pool`` mode, they remain available as a separate capital
        allocation / replacement-demand diagnostic and are not deducted from
        firm profits.

        Computes:
        Revenue
        - Wages
        - Input costs
        - Production taxes
        - Interest

        Returns:
            np.ndarray: Profits for each firm
        """
        capital_costs = self.capital_compensation_accounting_costs()
        capital_depreciation_costs = self.capital_depreciation_accounting_costs()
        direct_tfp_expense = self.direct_tfp_investment_cash_expense()
        return (
            self.ts.current("price") * self.ts.current("production")
            - self.ts.current("total_wage")
            - self.ts.current("used_intermediate_inputs_costs")
            - capital_costs
            - capital_depreciation_costs
            - self.ts.current("taxes_paid_on_production")
            - self.ts.current("interest_paid")
            - direct_tfp_expense
        )

    def compute_unit_costs(self) -> np.ndarray:
        """Calculate unit costs of production.

        Includes:
        - Wages
        - Input costs
        - Production taxes
        Per unit of output

        Returns:
            np.ndarray: Unit costs for each firm
        """
        capital_costs = self.capital_compensation_accounting_costs()
        capital_depreciation_costs = self.capital_depreciation_accounting_costs()
        return np.divide(
            self.ts.current("total_wage")
            + self.ts.current("used_intermediate_inputs_costs")
            + capital_costs
            + capital_depreciation_costs
            + self.ts.current("taxes_paid_on_production"),
            self.ts.current("production"),
            out=np.zeros_like(self.ts.current("production")),
            where=self.ts.current("production") != 0.0,
        )

    def compute_corporate_taxes_paid(self, tau_firm: float) -> np.ndarray:
        """Calculate corporate income taxes.

        Args:
            tau_firm (float): Corporate tax rate

        Returns:
            np.ndarray: Corporate taxes paid by each firm
        """
        return tau_firm * np.maximum(0.0, self.ts.current("profits"))

    def compute_deposits(self) -> np.ndarray:
        """Calculate end-of-period deposit balances.

        Updates deposits based on:
        - Previous balance
        - Sales revenue
        - Costs and expenses
        - Taxes
        - Credit flows

        Returns:
            np.ndarray: Updated deposit balances for each firm
        """
        return (
            self.ts.current("deposits")
            + self.ts.current("nominal_amount_sold_in_lcu")
            - self.ts.current("total_wage")
            - self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            - self.direct_tfp_investment_cash_expense()
            - self.ts.current("taxes_paid_on_production")
            - self.ts.current("corporate_taxes_paid")
            - self.ts.current("interest_paid")
            + self.ts.current("received_credit")
            - self.ts.current("debt_installments")
        )

    def direct_tfp_investment_cash_expense(self) -> np.ndarray:
        """Return current direct-TFP spending charged to firm cash accounts.

        Forced productivity investment is excluded because it is injected by
        policy hooks rather than paid from firm deposits.
        """
        if len(self.ts.direct_tfp_investment_cash_expense) > 0:
            return self.ts.current("direct_tfp_investment_cash_expense")
        if len(self.ts.planned_tfp_investment) == 0:
            return np.zeros(self.ts.current("n_firms"))
        return self.ts.current("planned_tfp_investment")

    def capital_compensation_accounting_costs(self) -> np.ndarray:
        """Return capital-compensation-derived costs charged in firm accounting."""
        mode = self.configuration.parameters.capital_compensation_accounting_mode
        if mode == "production_cost":
            return self.ts.current("used_capital_inputs_costs")
        if mode == "surplus_pool":
            return np.zeros_like(self.ts.current("used_capital_inputs_costs"))
        raise ValueError(
            f"Unknown capital_compensation_accounting_mode {mode!r}; expected 'production_cost' or 'surplus_pool'."
        )

    def capital_depreciation_accounting_costs(self) -> np.ndarray:
        """Return true CFC depreciation charged in non-cash accounting."""
        mode = self.configuration.parameters.capital_depreciation_accounting_mode
        if mode == "eurostat_cfc":
            return self.ts.current("capital_depreciation_costs")
        if mode == "none":
            return np.zeros_like(self.ts.current("capital_depreciation_costs"))
        raise ValueError(f"Unknown capital_depreciation_accounting_mode {mode!r}; expected 'none' or 'eurostat_cfc'.")

    def compute_gross_operating_surplus_mixed_income(self) -> np.ndarray:
        """Calculate gross operating surplus and mixed income.

        Computes operating surplus as:
        Revenue from sales
        + Change in inventories
        - Wages
        - Intermediate input costs
        - Production taxes

        Returns:
            np.ndarray: Gross operating surplus for each firm
        """
        return (
            self.ts.current("nominal_amount_sold_in_lcu")
            + self.ts.current("price") * (self.ts.current("inventory") - self.ts.prev("inventory"))
            - self.ts.current("total_wage")
            - self.ts.current("used_intermediate_inputs_costs")
            - self.ts.current("taxes_paid_on_production")
        )

    def handle_insolvency(
        self,
        credit_market: CreditMarket,
        illiquid_flag: np.ndarray | None = None,
    ) -> FirmInsolvencyResult:
        """Process insolvent firms and compute non-performing loan ratios.

        Handles firms that become insolvent by:
        - Marking them as insolvent when negative equity coincides with liquidity failure
        - Removing their outstanding loans
        - Zeroing their deposits and equity
        - Computing the non-performing loan ratio

        Args:
            credit_market (CreditMarket): Credit market for loan processing
            illiquid_flag (np.ndarray | None): Current-period settlement liquidity-failure flag.

        Returns:
            FirmInsolvencyResult: Default flags, bank credit losses, and firm-loan NPL ratio.
        """
        if illiquid_flag is None:
            illiquid_flag = np.asarray(self.ts.current("firm_settlement_illiquid_flag"), dtype=bool)
        else:
            illiquid_flag = np.asarray(illiquid_flag, dtype=bool)
        default_flag = np.logical_and(self.ts.current("equity") < 0.0, illiquid_flag)
        loan_writeoff_by_bank = credit_market.compute_defaulted_firm_loan_writeoff_by_bank(default_flag)
        overdraft_writeoff_by_bank = np.bincount(
            self.states["Corresponding Bank ID"],
            weights=np.maximum(0.0, -self.ts.current("deposits")) * default_flag,
            minlength=loan_writeoff_by_bank.shape[0],
        )
        credit_loss_by_bank = loan_writeoff_by_bank + overdraft_writeoff_by_bank
        self.states["is_insolvent"] = np.logical_or(self.states["is_insolvent"], default_flag)
        self.ts.override_current("firm_settlement_default_flag", default_flag.copy())

        # Remove loans
        insolvent_firms = np.where(default_flag)[0]
        bad_firm_loans = credit_market.remove_loans_to_firm(insolvent_firms)

        # Update deposits
        new_firm_deposits = self.ts.current("deposits")
        new_firm_deposits[default_flag] = 0.0
        self.ts.deposits.pop()
        self.ts.deposits.append(new_firm_deposits)

        # Update equity
        new_firm_equity = self.ts.current("equity")
        new_firm_equity[default_flag] = 0.0
        self.ts.equity.pop()
        self.ts.equity.append(new_firm_equity)

        # Calculate the NPL ratio for firm loans
        total_loans_granted = (
            credit_market.ts.current("total_outstanding_loans_granted_firms_short_term")[0]
            + credit_market.ts.current("total_outstanding_loans_granted_firms_long_term")[0]
        )
        if total_loans_granted == 0.0:
            npl_ratio = 0.0
        else:
            npl_ratio = bad_firm_loans / total_loans_granted
        return FirmInsolvencyResult(
            npl_ratio=npl_ratio,
            default_flag=default_flag.copy(),
            loan_writeoff_by_bank=loan_writeoff_by_bank,
            overdraft_writeoff_by_bank=overdraft_writeoff_by_bank,
            credit_loss_by_bank=credit_loss_by_bank,
        )

    def compute_equity(self, current_good_prices: np.ndarray) -> np.ndarray:
        """Calculate equity value for each firm.

        Computes firm equity as:
        Assets (inventory, materials, capital, deposits)
        - Liabilities (debt)

        Args:
            current_good_prices (np.ndarray): Current prices for valuation

        Returns:
            np.ndarray: Equity values for each firm
        """
        material = np.dot(self.ts.current("intermediate_inputs_stock"), current_good_prices)
        capital = np.dot(self.ts.current("capital_inputs_stock"), current_good_prices)
        return (
            self.ts.current("inventory") * self.ts.current("price")
            + material
            + capital
            + self.ts.current("deposits")
            - self.ts.current("debt")
        )

    def compute_insolvency_rate(self) -> tuple[float, np.ndarray]:
        """Calculate insolvency statistics.

        Returns:
            tuple[float, np.ndarray]: Overall insolvency rate and count by industry
        """
        firm_insolvency_rate = self.states["is_insolvent"].mean()
        num_insolvent_firms_by_sector = np.zeros(self.n_industries)
        for g in range(self.n_industries):
            num_insolvent_firms_by_sector[g] = np.sum(self.states["is_insolvent"][self.states["Industry"] == g])
        self.states["is_insolvent"] = np.full(self.ts.current("n_firms"), False)
        return firm_insolvency_rate, num_insolvent_firms_by_sector

    def compute_total_debt(self) -> float:
        """Calculate total debt across all firms.

        Returns:
            float: Aggregate firm debt
        """
        return self.ts.current("debt").sum()

    def update_emissions(
        self,
        readjusted_factors: np.ndarray,
        emitting_indices: list | np.ndarray,
        use_emission_multiplier: bool = False,
        readjusted_factors_ch4: Optional[np.ndarray] = None,
        emitting_indices_ch4: Optional[list | np.ndarray] = None,
    ):
        """Update emissions from production activities.

        Calculates emissions from:
        - Intermediate input use
        - Capital input use
        Tracks emissions by source (coal, gas, oil, refined products)

        When use_emission_multiplier is True and emission_fractions.co2 is available
        (shape: n_emitting x n_industries), each firm's slice is scaled by its
        industry-specific CO2 fraction multiplier before applying the emission factors.

        Args:
            readjusted_factors (np.ndarray): CO2 emission factors per unit (shape: n_emitting)
            emitting_indices (list | np.ndarray): CO2 emitting sector indices
            use_emission_multiplier (bool): Whether to apply industry-specific CO2 fraction multipliers
            readjusted_factors_ch4 (Optional[np.ndarray]): CH4 emission factors per unit
            emitting_indices_ch4 (Optional[list | np.ndarray]): CH4 emitting sector indices
        """
        used_intermediate_inputs = self.compute_used_intermediate_inputs()
        used_capital_inputs = self.compute_used_capital_inputs()

        # Apply per-industry CO2 fraction multipliers when enabled.
        # emission_fractions.co2 has shape (n_emitting, n_industries); transposed and
        # indexed by each firm's industry gives (n_firms, n_emitting) multipliers.
        if use_emission_multiplier and self.emission_fractions is not None and self.emission_fractions.co2 is not None:
            firm_industries = self.states["Industry"]
            emitting_fractions = self.emission_fractions.co2.T[firm_industries]
            inputs_slice = used_intermediate_inputs[:, emitting_indices] * emitting_fractions
            capital_slice = used_capital_inputs[:, emitting_indices] * emitting_fractions
        else:
            inputs_slice = used_intermediate_inputs[:, emitting_indices]
            capital_slice = used_capital_inputs[:, emitting_indices]

        inputs_emissions = inputs_slice @ readjusted_factors
        capital_emissions = capital_slice @ readjusted_factors

        refining_firms = self.states["Industry"] == emitting_indices[-1]
        inputs_emissions[refining_firms] = 0
        capital_emissions[refining_firms] = 0

        self.ts.inputs_emissions.append(inputs_emissions)
        self.ts.capital_emissions.append(capital_emissions)

        if emitting_indices_ch4 is not None and readjusted_factors_ch4 is not None:
            inputs_emissions_ch4 = used_intermediate_inputs[:, emitting_indices_ch4] @ readjusted_factors_ch4
            capital_emissions_ch4 = used_capital_inputs[:, emitting_indices_ch4] @ readjusted_factors_ch4
            self.ts.inputs_emissions_ch4.append(inputs_emissions_ch4)
            self.ts.capital_emissions_ch4.append(capital_emissions_ch4)

        inputs_emissions_disaggregated = inputs_slice * readjusted_factors
        capital_emissions_disaggregated = capital_slice * readjusted_factors
        inputs_emissions_disaggregated[refining_firms] = 0
        capital_emissions_disaggregated[refining_firms] = 0

        self.ts.coal_inputs_emissions.append(inputs_emissions_disaggregated[:, 0])
        self.ts.gas_inputs_emissions.append(inputs_emissions_disaggregated[:, 1])
        self.ts.oil_inputs_emissions.append(inputs_emissions_disaggregated[:, 2])
        self.ts.refined_products_inputs_emissions.append(inputs_emissions_disaggregated[:, 3])
        self.ts.coal_capital_emissions.append(capital_emissions_disaggregated[:, 0])
        self.ts.gas_capital_emissions.append(capital_emissions_disaggregated[:, 1])
        self.ts.oil_capital_emissions.append(capital_emissions_disaggregated[:, 2])
        self.ts.refined_products_capital_emissions.append(capital_emissions_disaggregated[:, 3])

    def compute_total_deposits(self) -> float:
        """Calculate total deposits across all firms.

        Returns:
            float: Aggregate firm deposits
        """
        return self.ts.current("deposits").sum()

    def save_to_h5(self, group: h5py.Group):
        """Save firm data to HDF5 format.

        Saves:
        - Time series data
        - Industry classifications
        - Firm-industry mapping

        Args:
            group (h5py.Group): HDF5 group to save data into
        """
        self.ts.write_to_h5("firms", group)

        firms_group = group["firms"]

        # Save industries DataFrame under 'firms_group'
        industries_df = self.industries_dataframe

        # Create a subgroup for industries under 'firms_group'
        industries_group = firms_group.create_group("industries")

        # Save the DataFrame to the HDF5 group
        self._save_dataframe_to_h5(industries_df, industries_group)

    @staticmethod
    def _save_dataframe_to_h5(df: pd.DataFrame, group: h5py.Group):
        """
        Saves a DataFrame to an HDF5 group.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            group (h5py.Group): The HDF5 group to save data into.
        """
        # Save index
        group.create_dataset("Firm_ID", data=df.index.values, dtype="int")

        # Save industry names as variable-length UTF-8 strings
        dt = h5py.string_dtype(encoding="utf-8")
        industry_names = df["Industry"].values
        group.create_dataset("Industry", data=industry_names, dtype=dt)

    def total_sales(self):
        """Get aggregate sales across all firms.

        Returns:
            float: Total firm sales
        """
        return self.ts.get_aggregate("total_sales")

    def total_used_input_costs(self):
        """Get aggregate intermediate input costs.

        Returns:
            float: Total intermediate input costs
        """
        return self.ts.get_aggregate("used_intermediate_inputs_costs")

    def get_total_inputs_emissions(self):
        """Get total emissions from intermediate inputs.

        Returns:
            float: Total input-related emissions
        """
        return self.ts.get_aggregate("inputs_emissions")

    def get_disaggregated_input_emissions(self, input_name: str):
        """Get emissions for specific input type.

        Args:
            input_name (str): Name of input type

        Returns:
            float: Emissions for specified input
        """
        return self.ts.get_aggregate(f"{input_name}_inputs_emissions")

    def get_disaggregated_capital_emissions(self, input_name: str):
        """Get capital emissions for specific input type.

        Args:
            input_name (str): Name of input type

        Returns:
            float: Capital emissions for specified input
        """
        return self.ts.get_aggregate(f"{input_name}_capital_emissions")

    def get_total_capital_emissions(self):
        """Get total emissions from capital use.

        Returns:
            float: Total capital-related emissions
        """
        return self.ts.get_aggregate("capital_emissions")

    def total_bought_input_costs(self):
        """Get aggregate cost of purchased inputs.

        Returns:
            float: Total input purchase costs
        """
        return self.ts.get_aggregate("total_intermediate_inputs_bought_costs")

    def total_operating_surplus(self):
        """Get aggregate operating surplus.

        Returns:
            float: Total gross operating surplus
        """
        return self.ts.get_aggregate("gross_operating_surplus_mixed_income")

    def total_wages(self):
        """Get aggregate wage payments.

        Returns:
            float: Total wages paid
        """
        return self.ts.get_aggregate("total_wage")

    def total_inventory_change(self):
        """Get aggregate change in inventories.

        Returns:
            float: Total inventory value change
        """
        return self.ts.get_aggregate("total_inventory_change")

    def total_capital_bought(self):
        """Get aggregate capital purchases.

        Returns:
            float: Total capital goods bought
        """
        return self.ts.get_aggregate("total_capital_inputs_bought_costs")

    def total_production(self):
        """Get aggregate production.

        Returns:
            float: Total production quantity
        """
        return self.ts.get_aggregate("production")

    def total_profits(self):
        """Get aggregate profits.

        Returns:
            float: Total firm profits
        """
        return self.ts.get_aggregate("profits")

    def total_taxes_paid_on_production(self):
        """Get aggregate production taxes paid.

        Returns:
            float: Total production taxes
        """
        return self.ts.get_aggregate("taxes_paid_on_production")

    def increase_industry_input_productivity(self, producing_industry: str, input_industry: str, increase_pct: float):
        producing_index = self.industries.index(producing_industry)
        input_index = self.industries.index(input_industry)

        self.base_intermediate_inputs_productivity_matrix[input_index, producing_index] *= 1 + increase_pct

    def compute_net_capital_investment_above_replacement(self) -> np.ndarray:
        """Calculate ordinary capital investment above capital-use replacement.

        Separates total capital investment into:
        1. Replacement investment: covers capital-input use to maintain capacity
        2. Net investment: excess capital purchases above replacement

        Returns:
            np.ndarray: Net capital investment above replacement for each firm
        """
        current_good_prices = self.current_good_prices
        # Calculate replacement investment needed (in monetary terms)
        production = self.ts.current("production")
        capital_use_matrix = self.base_capital_input_use_matrix[:, self.states["Industry"]].T
        if current_good_prices.shape[0] != capital_use_matrix.shape[1]:
            raise ValueError(
                "current_good_prices length must match the capital-input industry dimension "
                f"({current_good_prices.shape[0]} != {capital_use_matrix.shape[1]})."
            )

        # For each firm, calculate total replacement cost across all capital types
        replacement_needs = production[:, None] * capital_use_matrix
        total_replacement_cost = (replacement_needs * current_good_prices[None, :]).sum(axis=1)

        # Actual total investment
        total_investment = self.ts.current("total_capital_inputs_bought_costs")

        # Net capital investment cannot be negative. This is an ordinary capital
        # diagnostic, not planner-executed productivity investment.
        net_capital_investment = np.maximum(0, total_investment - total_replacement_cost)

        return net_capital_investment

    def compute_capital_bundle_deflator(self) -> np.ndarray:
        """Compute a firm-specific capital-goods price for real productivity investment."""
        current_good_prices = self.current_good_prices
        production = self.ts.current("production")
        capital_use_matrix = self.base_capital_input_use_matrix[:, self.states["Industry"]].T
        if current_good_prices.shape[0] != capital_use_matrix.shape[1]:
            raise ValueError(
                "current_good_prices length must match the capital-input industry dimension "
                f"({current_good_prices.shape[0]} != {capital_use_matrix.shape[1]})."
            )

        replacement_needs = production[:, None] * capital_use_matrix
        replacement_totals = replacement_needs.sum(axis=1, keepdims=True)
        weights = np.divide(
            replacement_needs,
            replacement_totals,
            out=np.zeros_like(replacement_needs),
            where=replacement_totals > 0,
        )

        if len(self.ts.real_amount_bought_as_capital_goods) > 0:
            bought_capital = np.nan_to_num(self.ts.current("real_amount_bought_as_capital_goods"), nan=0.0)
            purchase_totals = bought_capital.sum(axis=1, keepdims=True)
            use_purchase_bundle = (replacement_totals[:, 0] <= 0) & (purchase_totals[:, 0] > 0)
            purchase_weights = np.divide(
                bought_capital,
                purchase_totals,
                out=np.zeros_like(bought_capital),
                where=purchase_totals > 0,
            )
            weights[use_purchase_bundle] = purchase_weights[use_purchase_bundle]

        weight_totals = weights.sum(axis=1, keepdims=True)
        use_baseline_bundle = weight_totals[:, 0] <= 0
        if np.any(use_baseline_bundle):
            baseline_capital_needs = self.base_capital_inputs_productivity_matrix[:, self.states["Industry"]].T
            baseline_totals = baseline_capital_needs.sum(axis=1, keepdims=True)
            baseline_weights = np.divide(
                baseline_capital_needs,
                baseline_totals,
                out=np.zeros_like(baseline_capital_needs),
                where=baseline_totals > 0,
            )
            weights[use_baseline_bundle] = baseline_weights[use_baseline_bundle]

        return (weights * current_good_prices[None, :]).sum(axis=1)

    def compute_real_productivity_investment(self, nominal_productivity_investment: np.ndarray) -> np.ndarray:
        """Deflate nominal productivity investment into capital-bundle volume units."""
        nominal_productivity_investment = np.asarray(nominal_productivity_investment, dtype=float)
        capital_bundle_deflator = self.compute_capital_bundle_deflator()
        real_investment = np.zeros_like(nominal_productivity_investment)
        positive_investment = nominal_productivity_investment > 0
        valid_deflator = capital_bundle_deflator > 0

        missing_deflator = positive_investment & ~valid_deflator
        if np.any(missing_deflator):
            missing = np.where(missing_deflator)[0].tolist()
            raise ValueError(f"Cannot deflate positive productivity investment for firms {missing}.")

        real_investment[positive_investment] = (
            nominal_productivity_investment[positive_investment] / capital_bundle_deflator[positive_investment]
        )
        return real_investment

    def execute_productivity_investment(self) -> None:
        """Execute planned productivity investment and store the realized amounts.

        This method should be called after production and investment decisions
        are finalized to record the actual productivity investment made.
        """
        # Ordinary capital purchases above replacement are a separate diagnostic
        # from productivity investment chosen by the planner.
        forced_productivity_investment = self.states["forced_productivity_investment"]
        net_capital_investment = self.compute_net_capital_investment_above_replacement()
        executed_productivity_investment = forced_productivity_investment.copy()
        executed_tfp_investment = forced_productivity_investment.copy()
        direct_tfp_cash_expense = np.zeros_like(forced_productivity_investment)
        executed_technical_investment = np.zeros_like(self.ts.current("planned_technical_investment"))
        if len(self.ts.planned_productivity_investment) > 0:
            planned_productivity_investment = self.ts.current("planned_productivity_investment")
            planned_tfp_investment = self.ts.current("planned_tfp_investment")
            planned_technical_investment = self.ts.current("planned_technical_investment")
            if getattr(self.functions["productivity_investment_planner"], "executes_direct_tfp_independently", False):
                executed_planned_investment = planned_productivity_investment.copy()
                executed_productivity_investment += executed_planned_investment
                executed_tfp_investment = planned_tfp_investment + forced_productivity_investment
                direct_tfp_cash_expense = planned_tfp_investment.copy()
                executed_technical_investment = planned_technical_investment.copy()
            else:
                executed_planned_investment = np.minimum(net_capital_investment, planned_productivity_investment)
                execution_ratio = np.divide(
                    executed_planned_investment,
                    planned_productivity_investment,
                    out=np.zeros_like(net_capital_investment),
                    where=planned_productivity_investment > 0,
                )
                executed_productivity_investment += executed_planned_investment
                executed_tfp_investment = planned_tfp_investment * execution_ratio + forced_productivity_investment
                direct_tfp_cash_expense = planned_tfp_investment * execution_ratio
                executed_technical_investment = planned_technical_investment * execution_ratio[:, np.newaxis]
        real_executed_investment = self.compute_real_productivity_investment(executed_productivity_investment)

        # Store in time series
        self.ts.net_capital_investment_above_replacement.append(net_capital_investment)
        self.ts.executed_productivity_investment.append(executed_productivity_investment)
        self.ts.executed_tfp_investment.append(executed_tfp_investment)
        self.ts.direct_tfp_investment_cash_expense.append(direct_tfp_cash_expense)
        self.ts.executed_technical_investment.append(executed_technical_investment)
        self.ts.real_executed_productivity_investment.append(real_executed_investment)
        self.states["forced_productivity_investment"] = np.zeros_like(self.states["forced_productivity_investment"])

    def compute_tfp_growth(self) -> np.ndarray:
        """Calculate TFP growth rates for all firms.

        Uses the configured productivity growth function to compute
        TFP growth based on:
        - Current TFP levels
        - Current production
        - Executed productivity investment (if available, otherwise computed)
        - Configuration parameters

        Returns:
            np.ndarray: TFP growth rates for each firm
        """
        # Use nominal executed TFP investment over nominal output value. Both
        # terms are value flows, matching the planner's nominal investment base.
        if len(self.ts.executed_tfp_investment) > 0:
            productivity_investment = self.ts.current("executed_tfp_investment")
        elif len(self.ts.executed_productivity_investment) > 0:
            productivity_investment = self.ts.current("executed_productivity_investment")
        else:
            # Fallback for initial period or if execute_productivity_investment wasn't called
            productivity_investment = np.zeros_like(self.states["tfp_multiplier"])

        # Get configuration parameters, using defaults if not specified
        base_growth = getattr(self.configuration.parameters, "tfp_base_growth_rate", 0.0025)  # 0.25% quarterly
        elasticity = getattr(self.configuration.parameters, "tfp_investment_elasticity", 0.3)

        # Use productivity growth function if available, otherwise use simple growth
        if "productivity_growth" in self.functions:
            return self.functions["productivity_growth"].compute_tfp_growth(
                current_tfp=self.states["tfp_multiplier"],
                production=self.ts.current("production"),
                productivity_investment=productivity_investment,
                output_value=self.compute_nominal_output_value(),
                base_growth_rate=base_growth,
                investment_elasticity=elasticity,
            )
        else:
            # Simple base growth if no function configured
            return np.full_like(self.states["tfp_multiplier"], base_growth)

    def update_tfp(self) -> None:
        """Update TFP multipliers based on computed growth rates.

        Computes TFP growth and updates the tfp_multiplier state variable.
        """
        tfp_growth = self.compute_tfp_growth()
        self.states["tfp_multiplier"] *= 1 + tfp_growth
        self.ts.tfp_multiplier.append(self.states["tfp_multiplier"].copy())

    def update_technical_coefficients(self) -> None:
        """Update technical coefficient multipliers based on computed growth rates.

        Computes technical coefficient growth and updates the intermediate and capital
        tech multiplier state variables based on technical investment.
        """
        if "technical_coefficients_growth" not in self.functions:
            return  # No technical growth configured

        growth_func = self.functions["technical_coefficients_growth"]

        # Get executed technical investment, after goods-market purchases determine realised investment.
        if hasattr(self.ts, "executed_technical_investment") and len(self.ts.executed_technical_investment) > 0:
            technical_investment = self.ts.current("executed_technical_investment")
        else:
            # No technical investment yet
            return

        # Use actual base technical coefficients (a_ij matrices)
        base_intermediate_coefficients = self.base_intermediate_inputs_productivity_matrix
        base_capital_coefficients = self.base_capital_inputs_productivity_matrix
        input_prices = self.current_good_prices
        if input_prices.shape[0] != technical_investment.shape[1]:
            raise ValueError(
                "input prices length must match technical-investment input dimension "
                f"({input_prices.shape[0]} != {technical_investment.shape[1]})."
            )

        # Update intermediate coefficient multipliers
        intermediate_growth = growth_func.compute_intermediate_multiplier_growth(
            current_multipliers=self.states["intermediate_tech_multipliers"],
            cumulative_improvements=self.states.get(
                "cumulative_intermediate_improvements", np.zeros_like(self.states["intermediate_tech_multipliers"])
            ),
            base_coefficients=base_intermediate_coefficients,
            firm_industries=self.states["Industry"],
            technical_investment=technical_investment,
            production=self.ts.current("production"),
            prices=input_prices,
        )

        # Update capital coefficient multipliers
        capital_growth = growth_func.compute_capital_multiplier_growth(
            current_multipliers=self.states["capital_tech_multipliers"],
            cumulative_improvements=self.states.get(
                "cumulative_capital_improvements", np.zeros_like(self.states["capital_tech_multipliers"])
            ),
            base_coefficients=base_capital_coefficients,
            firm_industries=self.states["Industry"],
            technical_investment=technical_investment,
            production=self.ts.current("production"),
            prices=input_prices,
        )

        # Apply growth to multipliers
        self.states["intermediate_tech_multipliers"] *= 1 + intermediate_growth
        self.states["capital_tech_multipliers"] *= 1 + capital_growth

        # Update cumulative improvements for diminishing returns
        if "cumulative_intermediate_improvements" not in self.states:
            self.states["cumulative_intermediate_improvements"] = intermediate_growth.copy()
        else:
            self.states["cumulative_intermediate_improvements"] += intermediate_growth

        if "cumulative_capital_improvements" not in self.states:
            self.states["cumulative_capital_improvements"] = capital_growth.copy()
        else:
            self.states["cumulative_capital_improvements"] += capital_growth


def fillna(array: np.ndarray, value: float = 0):
    """Fill NaN values in an array with a specified value.

    Args:
        array (np.ndarray): Input array with potential NaN values.
        value (float, optional): Value to replace NaN. Defaults to 0.

    Returns:
        np.ndarray: Array with NaN values replaced.
    """
    return np.where(np.isnan(array), value, array)
