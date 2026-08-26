"""Module implementing the country-level economic model.

This module provides the core Country class that integrates all economic agents,
markets, and processes within a single national economy. It orchestrates:

1. Economic Agents:
   - Individuals (workers, consumers)
   - Households (consumption, investment, housing decisions)
   - Firms (production, pricing, employment)
   - Banks (lending, deposit-taking)
   - Government entities (public spending, services)
   - Central bank (monetary policy)

2. Markets:
   - Labor market (employment, wages)
   - Credit market (loans, interest rates)
   - Housing market (property sales, rentals)
   - Goods market (production, consumption)

3. Economic Processes:
   - Production and consumption
   - Price formation and inflation
   - Wage determination
   - Credit creation and allocation
   - Fiscal and monetary policy
   - International trade

The model runs in discrete timesteps (typically quarterly but configurable),
with each iteration updating all agents and markets in sequence according
to their behavioral rules and interactions.
"""

import logging
from copy import deepcopy
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from macro_data import SyntheticCountry
from macro_data.readers.default_readers import DataPaths
from macro_data.readers.income_belief_classification import compute_household_income_beliefs
from macro_data.readers.income_belief_priors import load_income_belief_priors_table
from macro_data.readers.insee_smic import load_insee_smic_annual_table
from macro_data.readers.permanent_income_forecast import (
    PermanentIncomeForecast,
    PermanentIncomeForecastInputs,
    forecast_common_permanent_income,
    load_permanent_income_forecast_inputs,
)
from macro_data.readers.permanent_income_mapping import (
    PermanentIncomeSimulationSources,
    build_permanent_income_forecast_regressors,
    design_matrix_to_forecast_reader_names,
    load_permanent_income_design_matrix,
    rebase_real_pc_income_index,
)
from macro_data.readers.portfolio_frm import load_frm_coefficients
from macro_data.util.frequency import periods_per_year
from macro_data.util.rates import annualized_fisher_real_rate
from macromodel.agents.agent import Agent
from macromodel.agents.banks.banks import Banks
from macromodel.agents.central_bank.central_bank import CentralBank
from macromodel.agents.central_government.central_government import CentralGovernment
from macromodel.agents.firms import Firms
from macromodel.agents.government_entities.government_entities import GovernmentEntities
from macromodel.agents.households.households import Households
from macromodel.agents.individuals.individual_properties import ActivityStatus
from macromodel.agents.individuals.individuals import Individuals
from macromodel.configurations import CountryConfiguration
from macromodel.country.diagnostic_dividend_fund import compute_dividend_income_after_withholding
from macromodel.economy.economy import Economy
from macromodel.exchange_rates import ExchangeRates
from macromodel.exogenous.exogenous import Exogenous
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.credit_market.func.clearing import compute_firm_borrower_credit_room
from macromodel.markets.housing_market.housing_market import HousingMarket
from macromodel.markets.labour_market.labour_market import LabourMarket
from macromodel.rest_of_the_world import RestOfTheWorld
from macromodel.util.get_histogram import get_histogram


def _positive_int_or_default(value: object, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _positive_float_or_default(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) and parsed > 0.0 else default


def _resolve_frm_coefficients_path(
    configured_path: str | Path | None,
    data_paths: Optional[DataPaths],
) -> Path | None:
    """Resolve configured FRM paths without overriding explicit absolute paths.

    Consumption-paper parameter files historically use either the
    data/raw_data/... or run_model/data/raw_data/... form. Both describe the
    same file beneath the runner's raw-data root, not necessarily the process
    working directory.
    """
    if configured_path is None:
        return data_paths.frm_coefficients_path if data_paths is not None else None

    candidate = Path(configured_path)
    if candidate.is_absolute() or data_paths is None:
        return candidate

    default_path = data_paths.frm_coefficients_path
    if default_path is None:
        return candidate

    try:
        raw_data_index = candidate.parts.index("raw_data")
    except ValueError:
        return candidate

    raw_data_root = Path(default_path).parent.parent
    return raw_data_root.joinpath(*candidate.parts[raw_data_index + 1 :])


def compute_stage5_subsistence_support(remaining_subsistence_shortfall: np.ndarray) -> np.ndarray:
    """Return cleaned real Stage 5 support implied by the post-floor shortfall."""
    shortfall = np.asarray(remaining_subsistence_shortfall, dtype=float)
    return np.where(np.isfinite(shortfall) & (shortfall > 0.0), shortfall, 0.0)


def validate_portfolio_settlement_configuration(wealth_function, uses_feasibility_resolver: bool) -> None:
    """Validate the cross-component prerequisites for settled portfolio choice."""
    settles = getattr(wealth_function, "settles_portfolio_choice", False)
    uses_choice = getattr(wealth_function, "uses_portfolio_choice", False)
    if settles and not uses_choice:
        raise ValueError("settles_portfolio_choice=True requires uses_portfolio_choice=True.")
    if settles and not uses_feasibility_resolver:
        raise ValueError("settles_portfolio_choice=True requires uses_feasibility_resolver=True.")


def _append_ficp_bank_effects(
    banks,
    *,
    loan_writeoff: np.ndarray,
    principal_arrears: np.ndarray,
    interest_income_loss: np.ndarray,
    npl: np.ndarray,
    npl_denominator: np.ndarray,
    cumulative_loan_writeoff: np.ndarray | None = None,
    cumulative_principal_arrears: np.ndarray | None = None,
    cumulative_interest_income_loss: np.ndarray | None = None,
    append_npl_denominator: bool = True,
) -> None:
    """Append one aggregated current-period 4c bank accounting effect."""
    cumulative_loan_writeoff = loan_writeoff if cumulative_loan_writeoff is None else cumulative_loan_writeoff
    cumulative_principal_arrears = (
        principal_arrears if cumulative_principal_arrears is None else cumulative_principal_arrears
    )
    cumulative_interest_income_loss = (
        interest_income_loss if cumulative_interest_income_loss is None else cumulative_interest_income_loss
    )
    banks.ts.consumer_default_loan_writeoff.append(loan_writeoff)
    banks.ts.total_consumer_default_loan_writeoff.append(
        [float(banks.ts.current("total_consumer_default_loan_writeoff")[0]) + cumulative_loan_writeoff.sum()]
    )
    banks.ts.consumer_default_principal_arrears.append(principal_arrears)
    banks.ts.total_consumer_default_principal_arrears.append(
        [float(banks.ts.current("total_consumer_default_principal_arrears")[0]) + cumulative_principal_arrears.sum()]
    )
    banks.ts.consumer_default_credit_loss.append(loan_writeoff)
    banks.ts.total_consumer_default_credit_loss.append(
        [float(banks.ts.current("total_consumer_default_credit_loss")[0]) + cumulative_loan_writeoff.sum()]
    )
    banks.ts.consumer_default_interest_income_loss.append(interest_income_loss)
    banks.ts.total_consumer_default_interest_income_loss.append(
        [
            float(banks.ts.current("total_consumer_default_interest_income_loss")[0])
            + cumulative_interest_income_loss.sum()
        ]
    )
    banks.ts.consumer_default_npl.append(npl)
    if append_npl_denominator:
        banks.ts.consumer_default_npl_denominator.append(npl_denominator)


class Country:
    """A complete national economy with interacting agents and markets.

    This class represents a single country's economy, integrating all economic agents
    (individuals, households, firms, banks, government) and markets (labor, credit,
    housing, goods). It manages their interactions and evolution over time.

    The economy operates in discrete timesteps, with each iteration:
    1. Updating exchange rates and monetary conditions
    2. Running agent-level planning and decision-making
    3. Clearing all markets sequentially
    4. Recording economic outcomes and updating metrics

    The timestep length is configurable (typically quarterly) and determined by
    how the input data is preprocessed (see ICIOReader's yearly_factor parameter).

    Attributes:
        country_name (str): Identifier for the country
        scale (int): Scaling factor for population-based calculations
        individuals (Individuals): Population of individual economic agents
        households (Households): Collection of household units
        firms (Firms): Collection of producing firms
        central_government (CentralGovernment): Fiscal authority
        government_entities (GovernmentEntities): Public sector bodies
        banks (Banks): Commercial banking sector
        central_bank (CentralBank): Monetary authority
        economy (Economy): Aggregate economic metrics and tracking
        labour_market (LabourMarket): Employment matching and wage setting
        credit_market (CreditMarket): Lending and borrowing
        housing_market (HousingMarket): Property sales and rentals
        exchange_rate_usd_to_lcu (float): Exchange rate to USD
        exogenous (Exogenous): External economic conditions
        forecasting_window (int): Periods ahead for forecasting
        assume_zero_growth (bool): Whether to assume no growth
        assume_zero_noise (bool): Whether to assume no random shocks
        configuration (CountryConfiguration): Model parameters
    """

    country_name: str
    scale: int
    individuals: Individuals
    households: Households
    firms: Firms
    central_government: CentralGovernment
    government_entities: GovernmentEntities
    banks: Banks
    central_bank: CentralBank
    economy: Economy
    labour_market: LabourMarket
    credit_market: CreditMarket
    housing_market: HousingMarket
    exchange_rate_usd_to_lcu: float
    exogenous: Exogenous
    forecasting_window: int
    assume_zero_growth: bool
    assume_zero_noise: bool
    configuration: CountryConfiguration

    def __init__(
        self,
        country_name: str,
        scale: int,
        individuals: Individuals,
        households: Households,
        firms: Firms,
        central_government: CentralGovernment,
        government_entities: GovernmentEntities,
        banks: Banks,
        central_bank: CentralBank,
        economy: Economy,
        labour_market: LabourMarket,
        credit_market: CreditMarket,
        housing_market: HousingMarket,
        exogenous: Exogenous,
        running_multiple_countries: bool,
        configuration: CountryConfiguration = CountryConfiguration(),
        forecasting_window: int = 60,
        assume_zero_growth: bool = False,
        assume_zero_noise: bool = False,
        add_emissions: bool = False,
        emission_factors_lcu: Optional[np.ndarray] = None,
        emitting_indices: Optional[np.ndarray] = None,
        emission_factors_lcu_ch4: Optional[np.ndarray] = None,
        emitting_indices_ch4: Optional[np.ndarray] = None,
        initial_year: Optional[int] = None,
        permanent_income_forecast_inputs: Optional[PermanentIncomeForecastInputs] = None,
        permanent_income_design_matrix: Optional[pd.DataFrame] = None,
    ):
        """Initialize a new country economy.

        Args:
            country_name (str): Country identifier
            scale (int): Population scaling factor
            individuals (Individuals): Individual agents
            households (Households): Household units
            firms (Firms): Producing firms
            central_government (CentralGovernment): Fiscal authority
            government_entities (GovernmentEntities): Public sector
            banks (Banks): Banking sector
            central_bank (CentralBank): Monetary authority
            economy (Economy): Economic tracking
            labour_market (LabourMarket): Employment market
            credit_market (CreditMarket): Lending market
            housing_market (HousingMarket): Property market
            exogenous (Exogenous): External conditions
            running_multiple_countries (bool): If True, part of multi-country sim
            configuration (CountryConfiguration): Model parameters
            forecasting_window (int): Forecast periods
            assume_zero_growth (bool): If True, assume no growth
            assume_zero_noise (bool): If True, assume no shocks
            add_emissions (bool): If True, track emissions
            emission_factors_lcu (Optional[np.ndarray]): Emission factors
            emitting_indices (Optional[np.ndarray]): Industry indices that emit
        """
        # Parameters
        self.country_name = country_name
        self.scale = scale

        # Agents
        self.individuals = individuals
        self.households = households
        self.firms = firms
        self.central_government = central_government
        self.government_entities = government_entities
        self.banks = banks
        self.central_bank = central_bank

        # The economy
        self.economy = economy

        # Markets
        self.labour_market = labour_market
        self.credit_market = credit_market
        self.housing_market = housing_market

        # Exchange rate
        self.exchange_rate_usd_to_lcu = 1

        # Exogenous data
        self.exogenous = exogenous

        # Configuration
        self.forecasting_window = forecasting_window
        self.assume_zero_growth = assume_zero_growth
        self.assume_zero_noise = assume_zero_noise

        self.running_multiple_countries = running_multiple_countries

        self.configuration = configuration
        self.economy.configure_consumer_price_sources(configuration.economy)

        validate_portfolio_settlement_configuration(
            households.functions["wealth"],
            configuration.households.parameters.uses_feasibility_resolver,
        )

        self.add_emissions = add_emissions
        self.emission_factors_lcu = emission_factors_lcu
        self.emitting_indices = emitting_indices
        self.emission_factors_lcu_ch4 = emission_factors_lcu_ch4
        self.emitting_indices_ch4 = emitting_indices_ch4
        self.use_emission_multiplier = self.configuration.use_emission_multiplier
        self._clear_household_income_shock()

        # Stage 3 common permanent-income forecast: cached at construction time so
        # the (potentially expensive, PSD-validated) inputs are loaded once.
        # start_period follows the same convention as the design-matrix trend column
        # (quarter 1 of initial_year, e.g. 2014Q1), consistent with `_quarter_period(int)`
        # in permanent_income_mapping.py.
        self.start_period: Optional[pd.Period] = (
            pd.Period(f"{initial_year}Q1", freq="Q") if initial_year is not None else None
        )
        self._permanent_income_forecast_inputs = permanent_income_forecast_inputs
        self._permanent_income_design_matrix = permanent_income_design_matrix
        self._stage5_subsistence_support_by_household: np.ndarray | None = None
        self._initialize_household_unemployment_benefits()

    def _household_unemployment_benefits_from_individuals(self) -> np.ndarray:
        """Aggregate the current individual unemployment payments to households in nominal units."""
        return self.economy.current_consumer_price_level() * self.households.aggregate_individual_amount(
            individual_amount=self.individuals.ts.current("income_from_unemployment_benefits"),
            corr_households=self.individuals.states["Corresponding Household ID"],
        )

    def _initialize_household_unemployment_benefits(self) -> None:
        """Include the initial individual unemployment payments in household income exactly once."""
        unemployment_benefits = self._household_unemployment_benefits_from_individuals()
        self.households.ts.override_current("income_unemployment_benefits", unemployment_benefits)
        self.households.ts.override_current(
            "income_social_transfers",
            self.households.ts.current("income_social_transfers") + unemployment_benefits,
        )
        self.households.ts.override_current(
            "total_income_social_transfers",
            [self.households.ts.current("income_social_transfers").sum()],
        )
        self.households.ts.override_current("expected_income_unemployment_benefits", unemployment_benefits)
        self.households.ts.override_current(
            "expected_income_social_transfers",
            self.households.ts.current("expected_income_social_transfers") + unemployment_benefits,
        )

    def clear_stage5_subsistence_support(self) -> None:
        """Clear current-period Stage 5 emergency subsistence support."""
        self._stage5_subsistence_support_by_household = None

    def set_stage5_subsistence_support_by_household(self, support_by_household: np.ndarray) -> None:
        """Store targeted current-period Stage 5 support, one value per household."""
        support = np.asarray(support_by_household, dtype=float)
        expected_shape = (self.households.ts.current("n_households"),)
        if support.shape != expected_shape:
            raise ValueError(
                "Stage 5 subsistence support must contain one value per household; "
                f"expected shape {expected_shape}, got {support.shape}."
            )
        if not np.all(np.isfinite(support)):
            raise ValueError("Stage 5 subsistence support must contain only finite real amounts.")
        if np.any(support < 0.0):
            raise ValueError("Stage 5 subsistence support must be non-negative for every household.")
        self._stage5_subsistence_support_by_household = support.copy()

    def current_stage5_subsistence_support_by_household(self) -> np.ndarray:
        """Return the current-period targeted Stage 5 support vector."""
        if self._stage5_subsistence_support_by_household is None:
            return np.zeros(self.households.ts.current("n_households"), dtype=float)
        return self._stage5_subsistence_support_by_household.copy()

    def current_stage5_subsistence_support_total(self) -> float:
        """Return aggregate Stage 5 support derived from the household vector."""
        return float(self.current_stage5_subsistence_support_by_household().sum())

    def compute_realised_stage5_subsistence_support(self) -> np.ndarray:
        """Return the settled Stage 5 support amount added to realised transfers."""
        current_cpi = self.economy.current_consumer_price_level()
        return current_cpi * self.current_stage5_subsistence_support_by_household()

    def current_settled_stage5_subsistence_support_total(self) -> float:
        """Return aggregate settled Stage 5 support from the persisted household series."""
        return float(np.asarray(self.households.ts.current("stage5_subsistence_support"), dtype=float).sum())

    def current_stage5_subsistence_support_fiscal_increment(self) -> float:
        """Return the nominal fiscal expenditure increment implied by settled Stage 5 support."""
        return self.current_settled_stage5_subsistence_support_total()

    def compute_realised_household_social_income_components(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return realised pension, unemployment, other-transfer, and Stage 5 components."""
        public_pension_by_individual = self.central_government.distribute_public_pension_benefits_to_individuals(
            retirement_eligibility=self.individuals.states["Is Retired"],
            public_pension_weights=self.individuals.states["Public Pension Weight"],
        )
        current_cpi = self.economy.current_consumer_price_level()
        public_pensions = current_cpi * self.households.aggregate_individual_amount(
            individual_amount=public_pension_by_individual,
            corr_households=self.individuals.states["Corresponding Household ID"],
        )
        unemployment_benefits = self._household_unemployment_benefits_from_individuals()
        other_transfers = self.households.compute_social_transfer_income(
            total_other_social_transfers=self.central_government.ts.current("total_other_benefits")[0],
            cpi=current_cpi,
        )
        return (
            public_pensions,
            unemployment_benefits,
            other_transfers,
            self.compute_realised_stage5_subsistence_support(),
        )

    def compute_realised_household_social_transfers(self) -> np.ndarray:
        """Return the authoritative sum of all realised household social-income components."""
        return sum(self.compute_realised_household_social_income_components())

    def settle_household_social_transfers(self) -> None:
        """Persist each realised social-income component once on the late settlement path."""
        public_pensions, unemployment_benefits, other_transfers, realised_support = (
            self.compute_realised_household_social_income_components()
        )
        realised_transfers = public_pensions + unemployment_benefits + other_transfers + realised_support
        self.individuals.ts.override_current(
            "public_pension_benefits",
            self.central_government.distribute_public_pension_benefits_to_individuals(
                retirement_eligibility=self.individuals.states["Is Retired"],
                public_pension_weights=self.individuals.states["Public Pension Weight"],
            ),
        )
        self.households.ts.income_public_pension.append(public_pensions)
        self.households.ts.income_unemployment_benefits.append(unemployment_benefits)
        self.households.ts.income_other_social_transfers.append(other_transfers)
        self.households.ts.stage5_subsistence_support.append(realised_support)
        self.households.ts.income_social_transfers.append(realised_transfers)
        self.households.ts.total_income_social_transfers.append([realised_transfers.sum()])

    @classmethod
    def from_pickled_country(
        cls,
        synthetic_country: SyntheticCountry,
        country_configuration: CountryConfiguration,
        exchange_rates: ExchangeRates,
        country_name: str,
        all_country_names: list[str],
        industries: list[str],
        initial_year: int,
        t_max: int,
        time_unit: int,
        running_multiple_countries: bool,
        emission_factors_usd: np.ndarray,
        data_paths: Optional[DataPaths] = None,
    ) -> "Country":
        """Create a Country instance from preprocessed synthetic data.

        This factory method constructs a complete country economy from preprocessed
        data, initializing all agents and markets with calibrated starting conditions.

        Args:
            synthetic_country (SyntheticCountry): Preprocessed country data
            country_configuration (CountryConfiguration): Model parameters
            exchange_rates (ExchangeRates): Exchange rate dynamics
            country_name (str): Country identifier
            all_country_names (list[str]): All countries in simulation
            industries (list[str]): Industry sectors
            initial_year (int): Starting year
            t_max (int): Maximum simulation periods
            time_unit (int): Simulation period length in months
            running_multiple_countries (bool): If True, part of multi-country sim
            emission_factors_usd (np.ndarray): Emission factors in USD

        Returns:
            Country: Initialized country economy
        """
        scale = synthetic_country.scale

        emission_industries = ["B05a", "B05b", "B05c", "C19"]
        add_emissions = all([industry in industries for industry in emission_industries])

        if add_emissions:
            emitting_indices = np.array([list(industries).index(industry) for industry in emission_industries])
            emission_factors_lcu = synthetic_country.emission_factors.emissions_array
        else:
            emitting_indices = None
            emission_factors_lcu = None

        if synthetic_country.emission_factors_ch4 is not None:
            emission_factors_lcu_ch4 = synthetic_country.emission_factors_ch4.emission_factors
            emitting_indices_ch4 = synthetic_country.emission_factors_ch4.emitting_indices
        else:
            emission_factors_lcu_ch4 = None
            emitting_indices_ch4 = None

        n_industries = len(industries)

        synthetic_population = synthetic_country.population
        individuals = Individuals.from_pickled_agent(
            synthetic_population=synthetic_population,
            configuration=country_configuration.individuals,
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
            scale=scale,
        )

        initial_consumption_by_industry = synthetic_country.industry_data["industry_vectors"][
            "Household Consumption in LCU"
        ]
        emission_fractions = synthetic_country.emission_fractions

        households = Households.from_pickled_agent(
            synthetic_population=synthetic_population,
            synthetic_country=synthetic_country,
            configuration=country_configuration.households,
            country_name=country_name,
            all_country_names=all_country_names,
            industries=industries,
            initial_consumption_by_industry=initial_consumption_by_industry,
            value_added_tax=synthetic_country.tax_data.value_added_tax,
            scale=scale,
            add_emissions=add_emissions,
            emission_fractions=emission_fractions,
        )

        average_initial_price = synthetic_country.industry_data["industry_vectors"]["Average Initial Price"].values
        firm_exo_prices = synthetic_country.firm_exo_prices
        if firm_exo_prices is not None:
            firm_exo_prices.initial_year = initial_year
            firm_exo_prices.initial_model_prices = average_initial_price

        firms = Firms.from_pickled_agent(
            synthetic_firms=synthetic_country.firms,
            configuration=country_configuration.firms,
            country_name=country_name,
            all_country_names=all_country_names,
            goods_criticality_matrix=synthetic_country.goods_criticality_matrix,
            average_initial_price=average_initial_price,
            industries=industries,
            add_emissions=add_emissions,
            emission_fractions=emission_fractions,
            firm_exo_prices=firm_exo_prices,
        )

        taxes_less_subsidies = synthetic_country.industry_data["industry_vectors"]["Taxes Less Subsidies Rates"].values

        n_unemployed = (individuals.states["Activity Status"] == ActivityStatus.UNEMPLOYED).sum()

        central_government = CentralGovernment.from_pickled_agent(
            synthetic_central_government=synthetic_country.central_government,
            configuration=country_configuration.central_government,
            country_name=country_name,
            all_country_names=all_country_names,
            taxes_net_subsidies=taxes_less_subsidies,
            number_of_unemployed_individuals=n_unemployed,
            initial_unemployment_benefit=synthetic_population.individual_data["Unemployment Benefit Entitlement"].iloc[
                0
            ],
            tax_data=synthetic_country.tax_data,
            n_industries=n_industries,
        )
        tax_overrides = country_configuration.central_government.tax_overrides
        if tax_overrides.value_added_tax_rate is not None or tax_overrides.household_investment_vat_rate is not None:
            central_government.reconcile_initial_vat(
                current_household_consumption_before_vat=households.ts.current("total_consumption_before_vat")[0],
                current_household_investment=households.ts.current("investment"),
            )
            households.ts.override_current(
                "total_consumption",
                [
                    (1 + central_government.states["Value-added Tax"])
                    * households.ts.current("total_consumption_before_vat")[0]
                ],
            )
            households.ts.override_current(
                "total_investment",
                [
                    (
                        1
                        + central_government.states["Household Capital Formation Tax"]
                        + (central_government.states["Household Investment VAT Rate"] or 0.0)
                    )
                    * households.ts.current("total_investment_before_vat")[0]
                ],
            )
        if (
            tax_overrides.household_capital_formation_rate is not None
            or tax_overrides.firm_capital_formation_rate is not None
        ):
            central_government.reconcile_initial_capital_formation_tax(
                current_household_investment=households.ts.current("investment"),
                current_firm_capital_formation=firms.ts.current("total_capital_inputs_bought_costs").sum(),
            )

        government_entities = GovernmentEntities.from_pickled_agent(
            synthetic_government_entities=synthetic_country.government_entities,
            configuration=country_configuration.government_entities,
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
            add_emissions=add_emissions,
            emission_factors_lcu=emission_factors_lcu,
            emitting_indices=emitting_indices,
        )

        banks = Banks.from_pickled_agent(
            synthetic_banks=synthetic_country.banks,
            configuration=country_configuration.banks,
            policy_rate_markup=synthetic_country.policy_rate_markup,
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
            scale=scale,
        )

        central_bank = CentralBank.from_pickled_agent(
            synthetic_central_bank=synthetic_country.central_bank,
            configuration=country_configuration.central_bank,
            country_name=country_name,
            all_country_names=all_country_names,
            n_industries=n_industries,
        )

        exogenous = Exogenous.from_pickled_agent(
            synthetic_country=synthetic_country,
            exchange_rates=exchange_rates,
            country_name=country_name,
            initial_year=initial_year,
            t_max=t_max,
            time_unit=time_unit,
        )

        economy = Economy.from_agents(
            country_name=country_name,
            all_country_names=all_country_names,
            economy_configuration=country_configuration.economy,
            individuals=individuals,
            firms=firms,
            central_government=central_government,
            government_entities=government_entities,
            households=households,
            industry_vectors=synthetic_country.industry_data["industry_vectors"],
            exogenous=exogenous,
            time_unit=time_unit,
            initial_sectoral_household_consumption=initial_consumption_by_industry.values.flatten(),
        )
        net_smic_base = cls._select_net_smic_base(
            initial_year=initial_year,
            fallback_net_smic=central_government.ts.current("unemployment_benefits_by_individual")[0],
            country_name=country_name,
            smic_path=data_paths.insee_smic_path if data_paths is not None else None,
        )
        economy.ts.override_current("net_smic_base", [net_smic_base])
        economy.ts.override_current(
            "subsistence_consumption",
            cls._compute_subsistence_consumption_from_units(
                net_smic_base=net_smic_base,
                current_cpi=economy.current_consumer_price_level(),
                initial_cpi=economy.initial_consumer_price_level(),
                consumption_units=households._current_consumption_units(),
                time_unit=time_unit,
            ),
        )

        if getattr(households.functions["consumption"], "uses_income_belief_learning", False):
            income_belief_priors_path = (
                data_paths.income_belief_priors_path if (data_paths is not None and country_name == "FRA") else None
            )
            try:
                income_belief_priors_table = load_income_belief_priors_table(income_belief_priors_path)
            except (OSError, ValueError):
                income_belief_priors_table = load_income_belief_priors_table(None)
            households.states["income_belief_priors"] = compute_household_income_beliefs(
                individuals_age=individuals.states["Age"],
                individuals_activity_status=individuals.states["Activity Status"],
                individuals_household_id=individuals.states["Corresponding Household ID"],
                individuals_income=individuals.ts.current("employee_income"),
                n_households=households.ts.current("n_households"),
                priors_table=income_belief_priors_table,
            )

        # Stage 4 (portfolio choice): load the GTP-FRM coefficient table once at init
        # rather than per-period. Unlike load_income_belief_priors_table, this loader
        # has no fallback-to-module-default path, so a missing/invalid file with the
        # flag enabled is a real config error and must raise, not silently fall back.
        # The wealth-function-configured path is authoritative (it is what the YAML
        # config actually sets via wealth.parameters); data_paths.frm_coefficients_path
        # is only a fallback for callers that don't set a per-config path, since
        # DataPaths.default_paths() always populates it unconditionally.
        if getattr(households.functions["wealth"], "uses_portfolio_choice", False):
            frm_coefficients_path = households.functions["wealth"].frm_coefficients_path
            resolved_frm_path = _resolve_frm_coefficients_path(frm_coefficients_path, data_paths)
            if resolved_frm_path is None:
                raise ValueError(
                    "uses_portfolio_choice=True requires a non-null frm_coefficients_path, "
                    "either on the wealth function configuration or via DataPaths."
                )
            # NOTE: stored for a future increment's use (the precomputed/grouped
            # target_share_source paths, e.g. compute_frm_magnitude_target_share);
            # Increment 3's active "scalar" target_share_source does not read this
            # state at all, so loading the wrong file would currently go unnoticed
            # by tests. Validate this path resolution again when a later increment
            # starts reading households.states["frm_coefficients"].
            households.states["frm_coefficients"] = load_frm_coefficients(resolved_frm_path)

        permanent_income_forecast_inputs: Optional[PermanentIncomeForecastInputs] = None
        permanent_income_design_matrix: Optional[pd.DataFrame] = None
        if country_name == "FRA" and data_paths is not None:
            if (
                data_paths.permanent_income_forecast_table_path is not None
                and data_paths.permanent_income_forecast_covariance_path is not None
                and data_paths.permanent_income_forecast_residual_variance_path is not None
            ):
                try:
                    permanent_income_forecast_inputs = load_permanent_income_forecast_inputs(
                        data_paths.permanent_income_forecast_table_path,
                        data_paths.permanent_income_forecast_covariance_path,
                        data_paths.permanent_income_forecast_residual_variance_path,
                    )
                except (OSError, ValueError):
                    permanent_income_forecast_inputs = None

            if data_paths.permanent_income_design_matrix_path is not None:
                try:
                    permanent_income_design_matrix = design_matrix_to_forecast_reader_names(
                        load_permanent_income_design_matrix(data_paths.permanent_income_design_matrix_path)
                    )
                except (OSError, ValueError):
                    permanent_income_design_matrix = None

        labour_market = LabourMarket.from_agents(
            individuals=individuals,
            labour_market_configuration=country_configuration.labour_market,
            country_name=country_name,
            n_industries=n_industries,
        )

        credit_market = CreditMarket.from_pickled_market(
            synthetic_credit_market=synthetic_country.credit_market,
            credit_market_configuration=country_configuration.credit_market,
            country_name=country_name,
        )

        housing_market = HousingMarket.from_pickled_market(
            synthetic_housing_market=synthetic_country.housing_market,
            housing_market_configuration=country_configuration.housing_market,
            country_name=country_name,
            scale=scale,
        )

        return cls(
            country_name=country_name,
            scale=scale,
            individuals=individuals,
            households=households,
            firms=firms,
            central_government=central_government,
            government_entities=government_entities,
            banks=banks,
            central_bank=central_bank,
            economy=economy,
            labour_market=labour_market,
            credit_market=credit_market,
            housing_market=housing_market,
            exogenous=exogenous,
            forecasting_window=country_configuration.forecasting_window,
            assume_zero_growth=country_configuration.assume_zero_growth,
            assume_zero_noise=country_configuration.assume_zero_noise,
            running_multiple_countries=running_multiple_countries,
            configuration=country_configuration,
            add_emissions=add_emissions,
            emission_factors_lcu=emission_factors_lcu,
            emitting_indices=emitting_indices,
            emission_factors_lcu_ch4=emission_factors_lcu_ch4,
            emitting_indices_ch4=emitting_indices_ch4,
            initial_year=initial_year,
            permanent_income_forecast_inputs=permanent_income_forecast_inputs,
            permanent_income_design_matrix=permanent_income_design_matrix,
        )

    @staticmethod
    def _select_net_smic_base(
        initial_year: int,
        fallback_net_smic: float,
        country_name: str,
        smic_path: Optional[str | Path] = None,
    ) -> float:
        if country_name != "FRA":
            return float(fallback_net_smic)
        try:
            annual_smic = load_insee_smic_annual_table(smic_path)
        except (OSError, ValueError):
            return float(fallback_net_smic)
        if initial_year in annual_smic.index:
            value = annual_smic.loc[initial_year]
            if pd.isna(value):
                return float(fallback_net_smic)
            return float(value)
        return float(fallback_net_smic)

    @staticmethod
    def _compute_subsistence_consumption_from_units(
        *,
        net_smic_base: float,
        current_cpi: float,
        initial_cpi: float,
        consumption_units: np.ndarray,
        time_unit: int,
    ) -> np.ndarray:
        initial_cpi = float(initial_cpi) if initial_cpi != 0.0 else 1.0
        net_smic_monthly_t = float(net_smic_base) * max(float(current_cpi) / initial_cpi, 0.0)
        # net_smic_base (and therefore net_smic_monthly_t) is a MONTHLY level
        # (macro_data/readers/insee_smic.py's net_monthly_smic), and time_unit is
        # months per model period -- so a period's worth is time_unit MONTHS summed,
        # i.e. net_smic_monthly_t * time_unit. The previous `* (time_unit / 12.0)`
        # applied the annual-RATE-to-period conversion used elsewhere in this file
        # to a monthly LEVEL, undershooting by a factor of 12 for a quarterly model
        # (0.25x instead of 3x). Verified against the design doc's worked example
        # (consumption_micro_macro.pdf footnote 3, p.8): monthly SMIC EUR1,128.70,
        # CU=1.8 -> quarterly subsistence EUR3,047.49 = 0.5 * 1128.70 * 1.8 * 3.
        net_smic_period_t = net_smic_monthly_t * float(time_unit)
        return 0.5 * net_smic_period_t * np.asarray(consumption_units, dtype=float)

    def _update_subsistence_consumption(self, *, replace_current: bool) -> None:
        net_smic_base = self.economy.ts.current("net_smic_base")[0]
        subsistence_consumption = self._compute_subsistence_consumption_from_units(
            net_smic_base=net_smic_base,
            current_cpi=self.economy.current_consumer_price_level(),
            initial_cpi=self.economy.initial_consumer_price_level(),
            consumption_units=self.households._current_consumption_units(),
            time_unit=self.economy.time_unit,
        )
        if replace_current:
            self.economy.ts.override_current("net_smic_base", [net_smic_base])
            self.economy.ts.override_current("subsistence_consumption", subsistence_consumption)
        else:
            self.economy.ts.net_smic_base.append([net_smic_base])
            self.economy.ts.subsistence_consumption.append(subsistence_consumption)

    def reset(self, configuration: CountryConfiguration) -> None:
        """Reset the country economy to its initial state.

        Resets all agents, markets, and state variables to their initial conditions,
        optionally with new configuration parameters.

        Args:
            configuration (CountryConfiguration): New model parameters
        """
        self.forecasting_window = configuration.forecasting_window
        self.assume_zero_growth = configuration.assume_zero_growth
        self.assume_zero_noise = configuration.assume_zero_noise

        self.individuals.reset(configuration=configuration.individuals)
        self.households.reset(configuration=configuration.households)
        self.firms.reset(configuration=configuration.firms)
        self.central_government.reset(configuration=configuration.central_government)
        self.government_entities.reset(configuration=configuration.government_entities)
        self.banks.reset(configuration=configuration.banks)
        self.central_bank.reset(configuration=configuration.central_bank)

        self.economy.reset(configuration=configuration.economy)

        self.labour_market.reset(configuration=configuration.labour_market)
        self.credit_market.reset(configuration=configuration.credit_market)
        self.housing_market.reset(configuration=configuration.housing_market)

        self.exogenous.reset()

        self.configuration = deepcopy(configuration)
        self._clear_household_income_shock()

    def _clear_household_income_shock(self) -> None:
        """Clear any staged household income shock used by MPC experiments."""
        self._staged_household_income_shock: Optional[np.ndarray] = None

    def stage_household_income_shock(self, shock: np.ndarray) -> None:
        """Stage an income shock for the current MPC experiment iteration.

        The staged shock is added to expected household income during planning
        and to realised household income during accounting. It is not assigned to
        a specific fiscal transfer account, so it represents an exogenous
        household income receipt for MPC measurement.
        """
        shock = np.asarray(shock, dtype=float)
        expected_shape = self.households.ts.current("expected_income").shape
        if shock.shape != expected_shape:
            raise ValueError(
                "Household income shock shape must match expected_income shape: "
                f"got {shock.shape}, expected {expected_shape}."
            )
        if self._staged_household_income_shock is not None:
            raise ValueError("A household income shock is already staged.")

        self._staged_household_income_shock = shock.copy()

    stage_household_expected_income_shock = stage_household_income_shock

    def _apply_household_income_shock(self, income: np.ndarray) -> np.ndarray:
        """Add the staged MPC income shock when one is active."""
        if self._staged_household_income_shock is None:
            return income
        return income + self._staged_household_income_shock

    def initialisation_phase(self, exchange_rate_usd_to_lcu: float) -> None:
        """Initialize the monthly economic cycle.

        Sets up the initial conditions for the current timestep, including
        exchange rates and firm counts.

        Args:
            exchange_rate_usd_to_lcu (float): Current exchange rate to USD
        """
        self.exchange_rate_usd_to_lcu = exchange_rate_usd_to_lcu
        self.firms.update_number_of_firms()

    def estimation_phase(self) -> None:
        """Run economic forecasting and estimation.

        Updates agent expectations about growth, inflation, and other key
        economic variables used in decision-making.
        """
        self.economy.set_estimates(
            exogenous_growth=self.exogenous.national_accounts_before["Real Gross Output (Growth)"].values,
            exogenous_inflation=self.exogenous.inflation_before,
            exogenous_hpi_growth=self.exogenous.house_price_index_before,
            exogenous_ppi_inflation_during=self.exogenous.national_accounts_during["PPI (Growth)"].values.flatten(),
            exogenous_cpi_inflation_during=self.exogenous.national_accounts_during["CPI (Growth)"].values.flatten(),
            exogenous_growth_during=self.exogenous.national_accounts_during[
                "Real Gross Output (Growth)"
            ].values.flatten(),
            forecasting_window=self.forecasting_window,
            assume_zero_growth=self.assume_zero_growth,
            assume_zero_noise=self.assume_zero_noise,
        )
        self.firms.set_estimates(
            previous_average_good_prices=self.economy.ts.current("good_prices"),
            current_estimated_growth=self.economy.ts.current("estimated_growth")[0],
        )

    def target_setting_phase(self) -> None:
        """Set agent-level targets and plans.

        Updates production targets, wage offers, and reservation wages based
        on current conditions and expectations.
        """
        # Firms set production targets
        self.firms.set_targets(
            bank_overdraft_rate_on_firm_deposits=self.banks.ts.current("overdraft_rate_on_firm_deposits"),
            bank_interest_rates_on_long_term_firm_loans=self.banks.ts.current("interest_rates_on_long_term_firm_loans"),
            estimated_growth=self.economy.ts.current("estimated_growth")[0],
            estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            current_good_prices=self.economy.ts.current("good_prices"),
        )

        self.update_firm_wage_offers(replace_current=False)

        # Individuals set reservation wages
        self.individuals.ts.reservation_wages.append(
            self.individuals.compute_reservation_wages(
                unemployment_benefits_by_individual=self.central_government.ts.current(
                    "unemployment_benefits_by_individual"
                )[0],
            )
        )

    def update_firm_wage_offers(self, *, replace_current: bool) -> None:
        """Set firm wage markup/offers from the current operative labour demand."""
        wage_tightness_markup = self.firms.compute_wages_markup()
        if replace_current:
            self.firms.ts.override_current("wage_tightness_markup", wage_tightness_markup)
        else:
            self.firms.ts.wage_tightness_markup.append(wage_tightness_markup)

        self.firms.states["offered_wage_function"] = self.firms.compute_offered_wage_function(
            corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            current_individual_labour_inputs=self.individuals.ts.current("labour_inputs"),
            previous_employee_income=self.individuals.ts.current("employee_income"),
            unemployment_benefits_by_individual=self.central_government.ts.current(
                "unemployment_benefits_by_individual"
            )[0],
            income_taxes=self.central_government.states["Income Tax"],
            employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
            employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
        )

    def clear_labour_market(self) -> None:
        """Execute labor market clearing.

        Matches job seekers with vacancies and determines employment
        and wages through the labor market mechanism.
        """
        logging.info("Clearing labour market for %s", self.country_name)
        labour_costs = self.labour_market.clear(
            firms=self.firms,
            households=self.households,
            individuals=self.individuals,
        )
        self.firms.ts.labour_costs.append(labour_costs)
        self.individuals.ts.offered_wage_of_accepted_job.append(
            self.individuals.states["Offered Wage of Accepted Job"].copy()
        )
        self.individuals.ts.started_new_job.append(self.individuals.states["Started New Job"].astype(float).copy())

    def update_pre_credit_planning_metrics(self) -> None:
        """Update planning metrics needed before credit-market clearing."""
        # Firms estimate profits
        self.firms.ts.expected_profits.append(
            self.firms.compute_estimated_profits(
                estimated_growth=self.economy.ts.current("estimated_growth")[0],
                estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            )
        )

        # Banks estimate profits
        self.banks.ts.expected_profits.append(
            self.banks.compute_estimated_profits(
                estimated_growth=self.economy.ts.current("estimated_growth")[0],
                estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            )
        )

        # Firms compute the expected value of its capital stock
        self.firms.ts.expected_capital_inputs_stock_value.append(
            self.firms.compute_expected_capital_inputs_stock_value(
                current_good_prices=self.economy.ts.current("good_prices"),
                estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            )
        )

        # Central bank policy rate
        self.central_bank.ts.policy_rate.append(
            [
                self.central_bank.compute_rate(
                    inflation=self.economy.ts.current("ppi_inflation")[0],
                    growth=self.economy.ts.current("total_growth")[0],
                    cpi_yoy_inflation=self.economy.current_consumer_annual_inflation(),
                    output_gap=self.economy.ts.current("output_gap")[0],
                    time_unit=self.economy.time_unit,
                )
            ]
        )

        # Firm demand for goods, before realised credit is known.
        self.firms.ts.unconstrained_target_intermediate_inputs.append(
            self.firms.compute_unconstrained_demand_for_intermediate_inputs(
                good_prices=self.economy.ts.current("good_prices")
            )
        )
        self.firms.ts.unconstrained_target_intermediate_inputs_costs.append(
            self.firms.compute_unconstrained_demand_for_intermediate_inputs_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.unconstrained_target_capital_inputs.append(
            self.firms.compute_unconstrained_demand_for_capital_inputs(
                good_prices=self.economy.ts.current("good_prices")
            )
        )
        self.firms.ts.unconstrained_target_capital_inputs_costs.append(
            self.firms.compute_unconstrained_demand_for_capital_inputs_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        self.update_pre_credit_household_finance_metrics()

    def update_pre_credit_household_finance_metrics(self) -> None:
        """Update household finance inputs needed before housing and credit clearing."""
        self._update_benefit_planning_metrics()
        self._set_household_income_expectations(replace_current=False)
        self._set_household_target_demand(replace_current=False)

    def update_post_labour_planning_metrics(self) -> None:
        """Update planning metrics that depend on labour clearing."""
        # Individual labour inputs
        self.individuals.ts.labour_inputs.append(self.individuals.compute_labour_inputs())

        # Number of employees for each firm
        self.firms.ts.number_of_employees.append(
            self.firms.compute_n_employees(
                corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            )
        )
        self.firms.ts.number_of_employees_histogram.append(
            get_histogram(self.firms.ts.current("number_of_employees"), None)
        )
        self.firms.ts.output_by_employee_histogram.append(
            get_histogram(
                self.firms.ts.current("production") / self.firms.ts.current("number_of_employees"),
                None,
            )
        )

        # Firm labour inputs
        labour_inputs_from_employees = self.firms.compute_labour_inputs(
            corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            current_labour_inputs=self.individuals.ts.current("labour_inputs"),
        )
        self.prepare_post_labour_realised_feasible_activity_plan()

        # Firm wages
        self.individuals.ts.employee_income.append(
            self.firms.set_employee_income(
                corresponding_firm=self.individuals.states["Corresponding Firm ID"],
                current_individual_labour_inputs=self.individuals.ts.current("labour_inputs"),
                current_individual_stating_new_job=self.individuals.states["Started New Job"],
                current_employee_income=self.individuals.ts.current("employee_income"),
                current_individual_offered_wage=self.individuals.states["Offered Wage of Accepted Job"],
                labour_inputs_from_employees=labour_inputs_from_employees,
                estimated_ppi_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
                income_taxes=self.central_government.states["Income Tax"],
                employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
                employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
            )
        )
        self.individuals.ts.employee_income_histogram.append(
            get_histogram(self.individuals.ts.current("employee_income"), self.scale)
        )

        # Firm production
        if self.assume_zero_growth:
            self.firms.ts.production.append(self.firms.ts.initial("production"))
        else:
            self.firms.ts.production.append(self.firms.compute_production())
        self.firms.ts.production_histogram.append(get_histogram(self.firms.ts.current("production"), None))

        # Firm prices
        if not self.assume_zero_growth:
            firm_wage_obligation_preview = self.firms.compute_total_wage_obligation(
                corresponding_firm=self.individuals.states["Corresponding Firm ID"],
                individual_wages=self.individuals.ts.current("employee_income"),
                income_taxes=self.central_government.states["Income Tax"],
                employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
                employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
                cpi=self.economy.current_consumer_price_level(),
            )
            self.firms.ts.price.append(
                self.firms.compute_price(
                    current_estimated_ppi_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
                    previous_average_good_prices=self.economy.ts.current("good_prices"),
                    ppi_during=self.exogenous.national_accounts_during["PPI (Value)"].values.flatten(),
                    current_good_prices=self.economy.ts.current("good_prices"),
                    wage_obligation_preview=firm_wage_obligation_preview,
                    producer_tax_rates=self.central_government.states["Taxes Less Subsidies Rates"][
                        self.firms.states["Industry"]
                    ],
                )
            )

        self._set_household_income_expectations(replace_current=True)
        self._set_household_target_demand(replace_current=True)
        if self.configuration.households.parameters.uses_feasibility_resolver:
            self.settle_authoritative_household_payments()

    def _update_benefit_planning_metrics(self) -> None:
        self.clear_stage5_subsistence_support()
        # The central government updates unemployment benefits paid to individuals and social transfers to households
        self.central_government.update_benefits(
            historic_benefit_indexation_inflation=self.economy.historic_consumer_period_inflation(),
            exogenous_benefit_indexation_inflation=self.exogenous.inflation_before["CPI Inflation"].values,
            current_estimated_benefit_indexation_inflation=(self.economy.current_expected_consumer_period_inflation()),
            current_unemployment_rate=self.economy.ts.current("unemployment_rate")[0],
            current_estimated_growth=self.economy.ts.current("estimated_growth")[0],
        )

        # Individuals update their income from unemployment benefits
        self.individuals.ts.income_from_unemployment_benefits.append(
            self.central_government.distribute_unemployment_benefits_to_individuals(
                current_individual_activity_status=self.individuals.states["Activity Status"],
            )
        )
        self.individuals.ts.public_pension_benefits.append(
            self.central_government.distribute_public_pension_benefits_to_individuals(
                retirement_eligibility=self.individuals.states["Is Retired"],
                public_pension_weights=self.individuals.states["Public Pension Weight"],
            )
        )

    def _set_household_income_expectations(self, *, replace_current: bool) -> None:
        # Individual income
        expected_individual_income = self.individuals.compute_expected_income(
            expected_firm_profits=self.firms.ts.current("expected_profits"),
            expected_bank_profits=self.banks.ts.current("expected_profits"),
            cpi=self.economy.current_consumer_price_level(),
            expected_inflation=self.economy.current_expected_consumer_period_inflation(),
            income_taxes=self.central_government.states["Income Tax"],
            tau_firm=self.central_government.states["Profit Tax"],
        )
        if replace_current:
            self.individuals.ts.override_current("expected_income", expected_individual_income)
        else:
            self.individuals.ts.expected_income.append(expected_individual_income)

        # Household income
        expected_unemployment_benefits = (
            (1 + self.economy.current_expected_consumer_period_inflation())
            * self.economy.current_consumer_price_level()
            * self.households.aggregate_individual_amount(
                individual_amount=self.individuals.ts.current("income_from_unemployment_benefits"),
                corr_households=self.individuals.states["Corresponding Household ID"],
            )
        )
        expected_income_employee = (
            self.households.compute_employee_income(
                individual_income=self.individuals.ts.current("expected_income"),
                corr_households=self.individuals.states["Corresponding Household ID"],
            )
            - expected_unemployment_benefits
        )
        expected_public_pension_by_individual = (
            self.central_government.distribute_public_pension_benefits_to_individuals(
                retirement_eligibility=self.individuals.states["Is Retired"],
                public_pension_weights=self.individuals.states["Public Pension Weight"],
            )
        )
        expected_income_public_pension = (
            (1 + self.economy.current_expected_consumer_period_inflation())
            * self.economy.current_consumer_price_level()
            * self.households.aggregate_individual_amount(
                individual_amount=expected_public_pension_by_individual,
                corr_households=self.individuals.states["Corresponding Household ID"],
            )
        )
        expected_income_other_social_transfers = self.households.compute_expected_social_transfer_income(
            total_other_social_transfers=self.central_government.ts.current("total_other_benefits")[0],
            cpi=self.economy.current_consumer_price_level(),
            expected_inflation=self.economy.current_expected_consumer_period_inflation(),
        )
        expected_income_social_transfers = (
            expected_income_public_pension + expected_unemployment_benefits + expected_income_other_social_transfers
        )
        income_rental = self.households.compute_rental_income(
            housing_data=self.housing_market.states["properties"],
            income_taxes=self.central_government.states["Income Tax"],
        )
        paper_returns_are_capital_gains = getattr(
            self.households.functions["wealth"],
            "illiquid_returns_are_capital_gains",
            False,
        )
        expected_dividend_income = self.households.compute_expected_dividend_income()
        expected_income_financial_assets = (
            expected_dividend_income
            if paper_returns_are_capital_gains
            else self.households.compute_expected_income_from_financial_assets()
        )

        if replace_current:
            self.households.ts.override_current("expected_income_employee", expected_income_employee)
            self.households.ts.override_current("expected_income_unemployment_benefits", expected_unemployment_benefits)
            self.households.ts.override_current("expected_income_public_pension", expected_income_public_pension)
            self.households.ts.override_current(
                "expected_income_other_social_transfers", expected_income_other_social_transfers
            )
            self.households.ts.override_current("expected_income_social_transfers", expected_income_social_transfers)
            self.households.ts.override_current("income_rental", income_rental)
            self.households.ts.override_current("total_income_rental", [income_rental.sum()])
            self.households.ts.override_current(
                "expected_income_financial_assets",
                expected_income_financial_assets,
            )
            self.households.ts.override_current(
                "expected_income_dividend_distributions",
                expected_dividend_income,
            )
            self.households.ts.override_current(
                "total_expected_income_dividend_distributions",
                [expected_dividend_income.sum()],
            )
        else:
            self.households.ts.expected_income_employee.append(expected_income_employee)
            self.households.ts.expected_income_unemployment_benefits.append(expected_unemployment_benefits)
            self.households.ts.expected_income_public_pension.append(expected_income_public_pension)
            self.households.ts.expected_income_other_social_transfers.append(expected_income_other_social_transfers)
            self.households.ts.expected_income_social_transfers.append(expected_income_social_transfers)
            self.households.ts.income_rental.append(income_rental)
            self.households.ts.total_income_rental.append([self.households.ts.current("income_rental").sum()])
            self.households.ts.expected_income_financial_assets.append(expected_income_financial_assets)
            self.households.ts.expected_income_dividend_distributions.append(expected_dividend_income)
            self.households.ts.total_expected_income_dividend_distributions.append([expected_dividend_income.sum()])

        expected_household_income = self._apply_household_income_shock(self.households.compute_expected_income())
        if replace_current:
            self.households.ts.override_current("expected_income", expected_household_income)
        else:
            self.households.ts.expected_income.append(expected_household_income)

    def _set_household_target_demand(self, *, replace_current: bool) -> None:
        self._update_subsistence_consumption(replace_current=replace_current)

        # Household target consumption
        # Note: For CES substitution, we track additional taxes (like carbon tax) similar to firms
        # Currently no additional taxes are implemented, so both initial and current are zero
        # This will be updated when carbon tax or other additional taxes are added
        current_additional_taxes = np.zeros(len(self.firms.ts.current("price")))
        initial_additional_taxes = np.zeros(len(self.firms.ts.current("price")))

        saving_rates_histogram_len = len(self.households.ts.saving_rates_histogram)

        # Stage 3 Increment 5: wire learned individual beliefs and the common
        # macro permanent-income forecast into the consumption rule. When the
        # rule does not opt in (default / non-FRA), both stay None and
        # compute_target_consumption reproduces the prior zero behaviour exactly.
        permanent_income_log_ratio = None
        permanent_income_log_ratio_individual = None
        permanent_income_log_ratio_common = None
        uncertainty_delta = None
        if getattr(self.households.functions["consumption"], "uses_income_belief_learning", False):
            common_log_ratio, common_forecast_variance = self._common_permanent_income_terms()
            learning_inputs = self.households.current_income_belief_learning_inputs(
                common_permanent_income_log_ratio=common_log_ratio,
                common_forecast_variance=common_forecast_variance,
            )
            permanent_income_log_ratio = learning_inputs["permanent_income_log_ratio"]
            permanent_income_log_ratio_individual = learning_inputs["permanent_income_log_ratio_individual"]
            permanent_income_log_ratio_common = learning_inputs["permanent_income_log_ratio_common"]
            uncertainty_delta = learning_inputs["uncertainty_delta"]

        uses_feasibility_resolver = self.configuration.households.parameters.uses_feasibility_resolver
        if uses_feasibility_resolver:
            if replace_current:
                scheduled_consumption_loan_payment = (
                    self.credit_market.compute_opening_scheduled_consumption_payments_by_household()
                )
                scheduled_mortgage_payment = (
                    self.credit_market.compute_opening_scheduled_mortgage_payments_by_household()
                )
                expected_shape = (self.households.ts.current("n_households"),)
                if scheduled_consumption_loan_payment.shape != expected_shape:
                    scheduled_consumption_loan_payment = (
                        self.credit_market.compute_scheduled_consumption_loan_payments_by_household()
                    )
                if scheduled_mortgage_payment.shape != expected_shape:
                    scheduled_mortgage_payment = self.credit_market.compute_scheduled_mortgage_payments_by_household()
            else:
                (
                    scheduled_consumption_loan_payment,
                    scheduled_mortgage_payment,
                ) = self.credit_market.preview_opening_household_service()
                expected_shape = (self.households.ts.current("n_households"),)
                if scheduled_consumption_loan_payment.shape != expected_shape:
                    scheduled_consumption_loan_payment = (
                        self.credit_market.compute_scheduled_consumption_loan_payments_by_household()
                    )
                if scheduled_mortgage_payment.shape != expected_shape:
                    scheduled_mortgage_payment = self.credit_market.compute_scheduled_mortgage_payments_by_household()
        else:
            scheduled_mortgage_payment = self.credit_market.compute_scheduled_mortgage_payments_by_household()
            scheduled_consumption_loan_payment = (
                self.credit_market.compute_scheduled_consumption_loan_payments_by_household()
            )

        expected_consumer_inflation = self.economy.current_expected_consumer_period_inflation()
        n_periods_per_year = periods_per_year(self.economy.time_unit)
        (
            consumer_contractual_rate,
            consumer_debt,
            mortgage_contractual_rate,
            mortgage_debt,
        ) = self.credit_market.household_contractual_debt_rate_components(
            use_opening_schedule=replace_current,
        )
        expected_household_shape = (self.households.ts.current("n_households"),)
        rate_components = (
            consumer_contractual_rate,
            consumer_debt,
            mortgage_contractual_rate,
            mortgage_debt,
        )
        if any(component.shape != expected_household_shape for component in rate_components):
            # Some lightweight fixtures intentionally use a placeholder credit
            # market with no one-to-one household loan state.  They can
            # represent only the no-debt case; applying its rate state to a
            # different household population would violate the paper weights.
            if np.any(consumer_debt > 0.0) or np.any(mortgage_debt > 0.0):
                raise ValueError("Household borrowing-rate components must have one value per household.")
            consumer_contractual_rate = np.zeros(expected_household_shape)
            consumer_debt = np.zeros(expected_household_shape)
            mortgage_contractual_rate = np.zeros(expected_household_shape)
            mortgage_debt = np.zeros(expected_household_shape)
        real_consumer_rate = annualized_fisher_real_rate(
            consumer_contractual_rate,
            expected_consumer_inflation,
            n_periods_per_year,
        )
        real_mortgage_rate = annualized_fisher_real_rate(
            mortgage_contractual_rate,
            expected_consumer_inflation,
            n_periods_per_year,
        )
        annual_spendable_income = np.maximum(
            n_periods_per_year * np.asarray(self.households.ts.current("expected_income"), dtype=float),
            1e-12,
        )
        real_borrowing_rate = np.divide(
            real_consumer_rate * consumer_debt + real_mortgage_rate * mortgage_debt,
            annual_spendable_income,
            out=np.zeros(expected_household_shape),
            where=np.isfinite(annual_spendable_income) & (annual_spendable_income > 0.0),
        )
        # Existing consumer debt is fixed-rate until new borrowing remodulates
        # the aggregate balance. The CACF cashflow input therefore uses only
        # the realized contractual repricing from that event, not an
        # unconditional movement in currently offered bank rates.
        if replace_current:
            consumer_debt_rate_delta = getattr(self, "_planning_consumer_debt_rate_delta_for_cacf", None)
            if consumer_debt_rate_delta is None:
                consumer_debt_rate_delta = self.credit_market.consumer_debt_rate_delta_for_cacf()
        else:
            consumer_debt_rate_delta = self.credit_market.consumer_debt_rate_delta_for_cacf()
            self._planning_consumer_debt_rate_delta_for_cacf = consumer_debt_rate_delta.copy()
        if consumer_debt_rate_delta.shape != expected_household_shape:
            # Some lightweight fixtures intentionally use a placeholder credit
            # market with no one-to-one household loan state. It can represent
            # the no-refinancing case only; a non-zero cashflow input without a
            # household mapping would be economically meaningless.
            if np.any(consumer_debt_rate_delta != 0.0):
                raise ValueError(
                    "Consumer-debt refinancing rates must have one value per household when they are non-zero."
                )
            consumer_debt_rate_delta = np.zeros(expected_household_shape)
        consumer_debt_rate_delta = n_periods_per_year * consumer_debt_rate_delta

        target_consumption = self.households.compute_target_consumption(
            expected_inflation=expected_consumer_inflation,
            current_cpi=self.economy.current_consumer_price_level(),
            initial_cpi=self.economy.initial_consumer_price_level(),
            exogenous_total_consumption=self.exogenous.national_accounts_during[
                "Real Household Consumption (Value)"
            ].values.flatten(),
            per_capita_unemployment_benefits=self.central_government.ts.current("unemployment_benefits_by_individual")[
                0
            ],
            tau_vat=self.central_government.states["Value-added Tax"],
            assume_zero_growth=self.assume_zero_growth,
            prices=self.firms.ts.current("price"),
            initial_prices=self.firms.ts.initial("price"),
            taxes=current_additional_taxes,
            initial_taxes=initial_additional_taxes,
            house_price_index=self.economy.ts.current("hpi")[0],
            house_price_growth=self.economy.ts.current("hpi_inflation")[0],
            lagged_cpi=self.economy.ts.prev(self.economy.consumer_price_level_series_name())[0],
            # Full consumer-price history, aligned period-for-period with the income
            # history households reads, so the geometric-average income denominator can
            # deflate each observation at its own price level.
            historic_deflator=np.asarray(
                self.economy.ts.historic(self.economy.consumer_price_level_series_name())
            ).reshape(-1),
            lagged_house_price_index=self.economy.ts.prev("hpi")[0],
            real_borrowing_rate=real_borrowing_rate,
            consumer_debt_rate_delta=consumer_debt_rate_delta,
            mortgage_payment=scheduled_mortgage_payment,
            permanent_income_log_ratio=permanent_income_log_ratio,
            permanent_income_log_ratio_individual=permanent_income_log_ratio_individual,
            permanent_income_log_ratio_common=permanent_income_log_ratio_common,
            uncertainty_delta=uncertainty_delta,
            replace_current_diagnostics=replace_current,
            time_unit=self.economy.time_unit,
            # CU-adjusted subsistence consumption (half net SMIC per consumption
            # unit), already refreshed above by _update_subsistence_consumption
            # (line 1189). Used only as the geometric-average income denominator's
            # floor for a household with no positive income anywhere in its
            # history window (PR #138 review finding 1); ignored otherwise.
            subsistence_income=self.economy.ts.current("subsistence_consumption"),
        )

        if replace_current:
            self.households.ts.override_current("target_consumption", target_consumption)
            if len(self.households.ts.saving_rates_histogram) > saving_rates_histogram_len:
                saving_rates_histogram = self.households.ts.saving_rates_histogram.pop()
                self.households.ts.override_current("saving_rates_histogram", saving_rates_histogram)
        else:
            self.households.ts.target_consumption.append(target_consumption)

        # Stage 5 (feasibility resolver), Increment 0: diagnostics-only liquidity-
        # shortfall computation. Must run after target_consumption is finalized
        # above (it consumes that value) and uses the same scheduled mortgage
        # service already computed for compute_target_consumption, plus the
        # parallel consumer-loan scheduled-service accessor. Has no effect on
        # goods or credit demand at this increment. See
        # knowledge-vault/wiki/architecture/consumption-stage-5-feasibility-resolver.md
        # (Increment 0 section).
        self.households.configure_feasibility_resolver(
            uses_feasibility_resolver,
            clear_post_grant=not replace_current,
        )
        scheduled_debt_service = scheduled_mortgage_payment + scheduled_consumption_loan_payment
        liquidity_shortfall = self.households.compute_and_record_liquidity_shortfall(
            target_consumption=self.households.ts.current("target_consumption"),
            scheduled_debt_service=scheduled_debt_service,
            replace_current=replace_current,
        )
        residual_shortfall_after_lfa = self.households.compute_and_record_liquid_asset_drawdown(
            liquidity_shortfall=liquidity_shortfall,
            replace_current=replace_current,
        )
        if uses_feasibility_resolver:
            self.households.populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
                liquidity_shortfall_before_repair=self.households.ts.current("liquidity_shortfall_before_repair"),
                funded_from_liquid_assets=self.households.ts.current("funded_from_liquid_assets"),
                residual_shortfall_after_lfa=self.households.ts.current("residual_shortfall_after_lfa"),
            )
            stage5_residual_shortfall = self.households.current_live_post_drawdown_residual()
        else:
            self.households.clear_pre_grant_feasible_plan()
            stage5_residual_shortfall = residual_shortfall_after_lfa
        borrow_vs_sell_inputs = self.households.build_borrow_vs_sell_inputs(
            target_consumption_total=self.households.ts.current("target_consumption").sum(axis=1),
            scheduled_debt_service=scheduled_debt_service,
        )
        self.households.compute_and_record_borrow_vs_sell_choice(
            residual_shortfall_after_lfa=stage5_residual_shortfall,
            banks=self.banks,
            borrow_vs_sell_inputs=borrow_vs_sell_inputs,
            replace_current=replace_current,
        )
        consumer_credit_parameters = getattr(self.configuration, "consumer_credit", {}) or {}
        consumer_credit_maturity = _positive_int_or_default(
            consumer_credit_parameters.get(
                "maturity_quarters", self.banks.parameters.household_consumption_loan_maturity
            ),
            self.banks.parameters.household_consumption_loan_maturity,
        )
        consumer_credit_dsti_limit = _positive_float_or_default(
            consumer_credit_parameters.get("dsti_limit", 0.35),
            0.35,
        )
        self.households.compute_and_record_residual_capacity_fallback(
            preferred_margin_after_lfa=self.households.ts.current("preferred_margin_after_lfa"),
            preferred_margin_amount=self.households.ts.current("preferred_margin_amount"),
            banks=self.banks,
            income=self.households.ts.current("expected_income"),
            scheduled_mortgage_payment=scheduled_mortgage_payment,
            consumer_loan_maturity=consumer_credit_maturity,
            dsti_limit=consumer_credit_dsti_limit,
            current_ifa=self.households.ts.current("illiquid_financial_assets"),
            replace_current=replace_current,
        )
        # Stage 5 (feasibility resolver), Increment 5: extend the live carrier
        # with the DSTI-capped shadow_credit_requested value just computed
        # above, so compute_target_credit() (called later, from
        # prepare_credit_market_clearing()) can read it via
        # current_live_credit_requested() instead of the legacy unbounded-gap
        # formula. Must run after compute_and_record_residual_capacity_fallback
        # (which produces shadow_credit_requested) and before
        # compute_target_credit(). See
        # knowledge-vault/wiki/architecture/consumption-stage-5-feasibility-resolver.md
        # (Increment 5 section).
        #
        # Only the replace_current=False pass matters: per the simulation
        # orchestration, compute_target_credit() runs exactly once per period,
        # from prepare_credit_market_clearing(), strictly between this method's
        # two calls (update_pre_credit_planning_metrics -> ... ->
        # update_post_labour_planning_metrics). Populating again on the
        # replace_current=True pass would write a credit_requested value that
        # is never read before next period's configure_feasibility_resolver()
        # wipes the carrier -- dead work that could mislead a future reader
        # into thinking the post-labour refresh matters.
        if uses_feasibility_resolver and not replace_current:
            self.households.populate_pre_grant_feasible_plan_credit_requested(
                credit_requested=self.households.ts.current("shadow_credit_requested"),
            )
            self.households.populate_pre_grant_feasible_plan_planned_liquidation(
                planned_liquidation_total=self.households.ts.current("liquidation_planned"),
                current_ifa=self.households.ts.current("illiquid_financial_assets"),
            )

        target_investment = self.households.compute_target_investment(
            expected_inflation=self.economy.current_expected_consumer_period_inflation(),
            current_cpi=self.economy.current_consumer_price_level(),
            initial_cpi=self.economy.initial_consumer_price_level(),
            exogenous_total_investment=self.exogenous.national_accounts_during[
                "Real Household Investment (Value)"
            ].values.flatten(),
            tau_cf=self.central_government.states["Household Capital Formation Tax"],
            assume_zero_growth=self.assume_zero_growth,
        )

        if replace_current:
            self.households.ts.override_current("target_investment", target_investment)
        else:
            self.households.ts.target_investment.append(target_investment)

    def update_planning_metrics(self) -> None:
        """Compatibility wrapper for callers that still use the old phase name."""
        self.update_pre_credit_planning_metrics()
        self.update_post_labour_planning_metrics()

    def prepare_housing_market_clearing(self) -> None:
        """Prepare for housing market transactions.

        Updates property values and household housing decisions before
        market clearing.
        """
        # Update property values
        self.housing_market.update_property_value()

        # Decide on whether to remain, rent or buy
        self.households.prepare_housing_market_clearing(
            housing_data=self.housing_market.states["properties"],
            observed_fraction_value_price=self.housing_market.ts.current("observed_fraction_value_price"),
            observed_fraction_rent_value=self.housing_market.ts.current("observed_fraction_rent_value"),
            expected_hpi_growth=self.economy.ts.current("estimated_hpi_inflation")[0],
            assumed_mortgage_maturity=self.banks.parameters.mortgage_maturity,
            rental_income_taxes=self.central_government.states["Income Tax"],
            time_unit=self.economy.time_unit,
        )

        # Set rent
        self.households.update_rent(
            housing_data=self.housing_market.states["properties"],
            historic_inflation=self.economy.historic_consumer_period_inflation(),
            exogenous_inflation_before=self.exogenous.inflation_before["CPI Inflation"].values,
        )

    def clear_housing_market(self) -> None:
        """Execute housing market clearing.

        Matches buyers/renters with sellers/landlords and determines
        property transactions.
        """
        self.housing_market.clear(
            household_main_residence_tenure_status=self.households.states["Tenure Status of the Main Residence"],
            max_price_willing_to_pay=self.households.ts.current("max_price_willing_to_pay"),
            max_rent_willing_to_pay=self.households.ts.current("max_rent_willing_to_pay"),
        )

    def prepare_credit_market_clearing(self) -> None:
        """Prepare for credit market transactions.

        Updates loan demands and interest rates before market clearing.
        """
        self.banks.set_interest_rates(central_bank_policy_rate=self.central_bank.ts.current("policy_rate")[0])
        self.firms.prepare_operating_revolving_facility_interest(
            short_term_firm_loan_rates=self.banks.ts.current("interest_rates_on_short_term_firm_loans"),
        )
        firm_wage_obligation_preview = self.firms.compute_total_wage_obligation(
            corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            individual_wages=self.individuals.ts.current("employee_income"),
            income_taxes=self.central_government.states["Income Tax"],
            employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
            employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
            cpi=self.economy.current_consumer_price_level(),
        )
        firm_production_tax_obligation_preview = self.compute_pre_credit_production_tax_obligation_preview()
        scheduled_firm_installment_preview = self.credit_market.compute_scheduled_firm_installments_preview()
        firm_interest_on_deposits_preview = self.firms.compute_interest_paid_on_deposits(
            bank_interest_rate_on_firm_deposits=self.banks.ts.current("interest_rate_on_firm_deposits"),
            bank_overdraft_rate_on_firm_deposits=self.banks.ts.current("overdraft_rate_on_firm_deposits"),
        )
        self.firms.compute_target_credit(
            estimated_growth=self.economy.ts.current("estimated_growth")[0],
            estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            wage_obligation_preview=firm_wage_obligation_preview,
            production_tax_obligation_preview=firm_production_tax_obligation_preview,
            interest_obligation_preview=(
                scheduled_firm_installment_preview["scheduled_interest_due"]
                + firm_interest_on_deposits_preview
                + self.firms.ts.current("operating_revolving_interest_paid")
            ),
            loan_interest_obligation_preview=scheduled_firm_installment_preview["scheduled_interest_due"],
            debt_installment_preview=scheduled_firm_installment_preview["scheduled_principal_due"],
        )
        self.households.compute_target_credit(
            current_sales=self.housing_market.states["current_sales"],
        )

    def clear_credit_market(self) -> None:
        """Execute credit market clearing.

        Matches borrowers with lenders and determines loan allocations
        and terms.
        """
        self.credit_market.clear(
            banks=self.banks,
            firms=self.firms,
            households=self.households,
            current_npl_firm_loans=self.economy.ts.current("npl_firm_loans")[0],
            current_npl_hh_cons_loans=self.economy.ts.current("npl_hh_cons_loans")[0],
            current_npl_mortgages=self.economy.ts.current("npl_mortgages")[0],
        )

    def process_housing_market_clearing(self) -> None:
        """Process housing market outcomes.

        Updates property ownership, rents, and related financial positions
        after market clearing.
        """
        self.housing_market.process_housing_market_clearing(
            household_states=self.households.states,
            household_received_mortgages=self.households.ts.current("received_mortgages"),
            household_financial_wealth=self.households.ts.current("wealth_financial_assets"),
        )
        self.housing_market.ts.observed_fraction_value_price.append(
            self.housing_market.compute_observed_fraction_value_price()
        )
        self.housing_market.ts.observed_fraction_rent_value.append(
            self.housing_market.compute_observed_fraction_rent_value()
        )
        self.households.process_housing_market_clearing(
            housing_data=self.housing_market.states["properties"],
            social_housing_function=self.central_government.functions["social_housing"],
            current_sales=self.housing_market.states["current_sales"].loc[
                self.housing_market.states["current_sales"]["sales_types"] == "Sell"
            ],
            current_unemployment_benefits_by_individual=self.central_government.ts.current(
                "unemployment_benefits_by_individual"
            )[0],
        )

    def process_credit_market_clearing(self) -> None:
        """Process credit market outcomes.

        Updates loan positions, interest payments, and related financial
        positions after market clearing.
        """
        staged_firm_installments = self.credit_market.schedule_firm_installments()
        self.firms.begin_firm_debt_settlement(
            opening_interest_arrears=staged_firm_installments["opening_interest_arrears"],
            opening_principal_arrears=staged_firm_installments["opening_principal_arrears"],
            contractual_interest_due=staged_firm_installments["contractual_interest_due"],
            contractual_principal_due=staged_firm_installments["contractual_principal_due"],
            scheduled_interest_due=staged_firm_installments["scheduled_interest_due"],
            scheduled_principal_due=staged_firm_installments["scheduled_principal_due"],
        )

        if self.configuration.households.parameters.uses_feasibility_resolver:
            self.credit_market.pay_household_mortgage_installments()
            return

        # Legacy resolver-disabled household servicing remains unchanged.
        self.households.ts.debt_installments.append(self.credit_market.pay_household_installments())
        self.households.ts.total_debt_installments.append([self.households.ts.current("debt_installments").sum()])
        self.credit_market.remove_repaid_loans(("cons_loans", "mort_loans"))

        # Calculate household debt
        self.households.ts.consumption_loan_debt.append(
            self.credit_market.compute_outstanding_consumption_loans_by_household()
        )
        self.households.ts.mortgage_debt.append(self.credit_market.compute_outstanding_mortgages_by_household())
        self.households.ts.debt.append(self.households.compute_debt())
        self.households.ts.debt_histogram.append(get_histogram(self.households.ts.current("debt"), self.scale))

        # Calculate the interest on loans paid by households
        self.households.ts.interest_paid_on_loans.append(self.credit_market.compute_interest_paid_by_household())

        # Calculate the interest on deposits received/paid by households
        self.households.ts.interest_paid_on_deposits.append(
            self.households.compute_interest_paid_on_deposits(
                bank_interest_rate_on_household_deposits=self.banks.ts.current("interest_rate_on_household_deposits"),
                bank_overdraft_rate_on_household_deposits=self.banks.ts.current("overdraft_rate_on_household_deposits"),
            )
        )

        # Calculate paid interest of households
        self.households.ts.interest_paid.append(self.households.compute_interest_paid())

    def reconcile_post_grant_feasible_plan(self) -> None:
        """Build settled household feasibility after consumer-credit clearing."""
        if not self.configuration.households.parameters.uses_feasibility_resolver:
            self.households.clear_post_grant_feasible_plan()
            return

        n_households = self.households.ts.current("n_households")
        credit_granted = np.asarray(self.households.ts.current("received_consumption_loans"), dtype=float)
        if credit_granted.shape != (n_households,):
            raise ValueError(
                "received_consumption_loans must contain exactly one value per household; "
                f"expected shape {(n_households,)}, got {credit_granted.shape}."
            )
        if not np.all(np.isfinite(credit_granted)):
            raise RuntimeError(
                "Stage 5 post-grant reconciliation requires cleared received_consumption_loans for every household."
            )

        granted_consumer_credit_by_bank_and_household = self.credit_market.pending_granted_consumption_loans()

        self.households.populate_post_grant_feasible_plan_from_granted_credit(
            credit_granted=credit_granted,
            granted_consumer_credit_by_bank_and_household=granted_consumer_credit_by_bank_and_household,
        )
        self.households.reserve_post_grant_executable_liquidation(
            available_pre_stage4_ifa=self.households.ts.current("illiquid_financial_assets"),
        )
        settled_plan = self.households.post_grant_feasible_plan
        if settled_plan is None or settled_plan.granted_consumer_credit_by_bank_and_household is None:
            raise RuntimeError("Stage 6 consumer-credit settlement carrier was not populated.")
        self.credit_market.settle_granted_consumption_loans(
            credit_granted=settled_plan.credit_granted,
            granted_consumer_credit_by_bank_and_household=settled_plan.granted_consumer_credit_by_bank_and_household,
            consumer_loan_maturity=self.banks.parameters.household_consumption_loan_maturity,
        )
        self.credit_market.prepare_household_service_snapshot()
        self.households.persist_post_grant_planned_liquidation_total()

    def settle_authoritative_household_payments(self) -> None:
        """Apply the floor, settle consumer service, and commit household debt once."""
        if self.credit_market.consumer_payments_settled():
            raise RuntimeError("Authoritative household payments have already been settled for this period.")
        if self.economy.time_unit not in (1, 3):
            raise ValueError("Stage 6 FICP supports monthly (1) and quarterly (3) periods only.")
        target_consumption = self.households.ts.current("target_consumption")
        subsistence_consumption = self.economy.ts.current("subsistence_consumption")
        self.households.apply_consumption_floor_to_post_grant_plan(
            consumption_before_floor=target_consumption.sum(axis=1),
            subsistence_floor=subsistence_consumption,
        )
        goods_consumption = self.households._scale_consumption_matrix_to_household_totals(
            target_consumption=target_consumption,
            household_consumption_total=self.households.post_grant_feasible_plan.consumption_after_floor,
        )
        self.households.ts.override_current("target_consumption", goods_consumption)
        self.households.populate_post_grant_early_repayment_capacity(
            mortgage_service=self.credit_market.compute_opening_scheduled_mortgage_payments_by_household(),
            scheduled_consumer_service=self.credit_market.compute_opening_scheduled_consumption_payments_by_household(),
            eligible_ficp=self.households.current_ficp_active(),
        )
        settlement = self.credit_market.settle_consumer_payments(
            remaining_shortfall=self.households.current_remaining_subsistence_shortfall(),
            # CreditMarket caps this capacity against the post-service balance
            # as part of the authoritative settlement. Country only routes the
            # household feasibility carrier; it does not impose a second
            # financing-policy calculation.
            early_repayment_capacity=self.households.current_early_consumer_repayment_capacity(),
        )
        # closing_consumer_arrears = settlement.arrears.closing_interest.sum(axis=0) + settlement.arrears.closing_principal.sum(axis=0)  # DEAD CODE
        mortgage_principal = self.credit_market.compute_mortgage_principal_paid_by_household()
        debt_installments = mortgage_principal + settlement.principal_paid
        self.households.ts.debt_installments.append(debt_installments)
        self.households.ts.total_debt_installments.append([debt_installments.sum()])
        self.households.ts.scheduled_consumer_payment.append(settlement.scheduled_payment.copy())
        self.households.ts.actual_consumer_payment.append(settlement.actual_payment.copy())
        self.households.ts.unpaid_consumer_payment.append(settlement.unpaid_payment.copy())
        self.households.ts.consumer_interest_paid.append(settlement.interest_paid.copy())
        self.households.ts.consumer_principal_paid.append(settlement.principal_paid.copy())
        self.households.ts.early_consumer_repayment.append(settlement.early_repayment.copy())
        rescheduling_events = self.credit_market.prepare_first_miss_consumer_loan_rescheduling(
            prior_missed_payment_count_consumer=self.households.ts.current("missed_payment_count_consumer"),
            prior_ficp_episode_missed_payment_count=self.households.ts.current("ficp_episode_missed_payment_count"),
            prior_ficp_episode_status=self.households.ts.current("ficp_episode_status"),
            consumer_loan_maturity=self.banks.parameters.household_consumption_loan_maturity,
            period=len(self.households.ts.dicts["scheduled_consumer_payment"]) - 1,
        )
        self.credit_market.finalize_household_consumer_schedule()
        self.households.record_consumer_loan_rescheduling_events(rescheduling_events)
        (
            consumer_contractual_principal,
            consumer_principal_arrears,
            consumer_interest_arrears,
        ) = self.credit_market.current_consumer_debt_components_by_household()
        self.households.record_stage6_consumer_distress_state(
            scheduled_consumer_payments=settlement.scheduled_payment,
            actual_consumer_payments=settlement.actual_payment,
            unpaid_consumer_payments=settlement.unpaid_payment,
            time_unit=self.economy.time_unit,
            period=len(self.households.ts.dicts["scheduled_consumer_payment"]) - 1,
            consumer_contractual_principal=consumer_contractual_principal,
            consumer_principal_arrears=consumer_principal_arrears,
            consumer_interest_arrears=consumer_interest_arrears,
        )
        self.credit_market.remodulate_ficp_consumer_loan_schedule(
            active_ficp=self.households.current_ficp_active(),
            remaining_periods=self.households.ts.current("ficp_exclusion_remaining_periods"),
        )
        self.credit_market.remove_repaid_loans(("cons_loans", "mort_loans"))
        self.households.ts.consumption_loan_debt.append(
            self.credit_market.compute_outstanding_consumption_loans_by_household()
        )
        self.households.ts.mortgage_debt.append(self.credit_market.compute_outstanding_mortgages_by_household())
        self.households.ts.debt.append(self.households.compute_debt())
        self.households.ts.debt_histogram.append(get_histogram(self.households.ts.current("debt"), self.scale))
        self.households.ts.interest_paid_on_loans.append(self.credit_market.compute_interest_paid_by_household())
        self.households.ts.interest_paid_on_deposits.append(
            self.households.compute_interest_paid_on_deposits(
                bank_interest_rate_on_household_deposits=self.banks.ts.current("interest_rate_on_household_deposits"),
                bank_overdraft_rate_on_household_deposits=self.banks.ts.current("overdraft_rate_on_household_deposits"),
            )
        )
        self.households.ts.interest_paid.append(self.households.compute_interest_paid())

    def validate_stage5_closing_balance_sheets(self) -> None:
        """Verify the resolver-path Stage 5 closing-account identities.

        This is the final single-period control. It checks financial-asset and
        cash-ledger identities alongside the CreditMarket-authoritative debt and
        bank-loan identities, preventing independent household or bank booking.
        """
        if not self.configuration.households.parameters.uses_feasibility_resolver:
            return

        household_lfa = self.households.ts.current("liquid_financial_assets")
        household_ifa = self.households.ts.current("illiquid_financial_assets")
        household_financial_wealth = self.households.ts.current("wealth_financial_assets")
        cash_ledger_residual = self.households.ts.current("stage5_cash_ledger_residual")
        household_consumer_debt = self.households.ts.current("consumption_loan_debt")
        household_mortgage_debt = self.households.ts.current("mortgage_debt")
        household_debt = self.households.ts.current("debt")
        market_consumer_debt = self.credit_market.compute_outstanding_consumption_loans_by_household()
        market_mortgage_debt = self.credit_market.compute_outstanding_mortgages_by_household()
        bank_consumer_assets = self.banks.ts.current("consumption_loans_to_households")
        market_bank_consumer_assets = self.credit_market.compute_outstanding_household_consumption_loans_by_bank()

        values = (
            household_lfa,
            household_ifa,
            household_financial_wealth,
            cash_ledger_residual,
            household_consumer_debt,
            household_mortgage_debt,
            household_debt,
            market_consumer_debt,
            market_mortgage_debt,
            bank_consumer_assets,
            market_bank_consumer_assets,
        )
        if not all(np.all(np.isfinite(value)) for value in values):
            raise RuntimeError("Stage 5 closing accounts must be finite.")
        non_negative_values = (
            household_ifa,
            household_consumer_debt,
            household_mortgage_debt,
            household_debt,
            market_consumer_debt,
            market_mortgage_debt,
            bank_consumer_assets,
            market_bank_consumer_assets,
        )
        if not all(np.all(value >= 0.0) for value in non_negative_values):
            raise RuntimeError("Stage 5 closing debt and bank-loan balances must be finite and non-negative.")
        if not np.allclose(cash_ledger_residual, 0.0, rtol=1e-10, atol=1e-7):
            raise RuntimeError("Stage 5 cash-ledger residual does not reconcile at closing.")
        if not np.allclose(
            household_financial_wealth,
            household_lfa + household_ifa,
            rtol=1e-10,
            atol=1e-7,
        ):
            raise RuntimeError("Stage 5 household financial wealth does not reconcile with LFA and IFA.")
        if not np.allclose(household_consumer_debt, market_consumer_debt, rtol=1e-10, atol=1e-8):
            raise RuntimeError("Stage 5 household consumer debt does not reconcile with the CreditMarket.")
        if not np.allclose(household_mortgage_debt, market_mortgage_debt, rtol=1e-10, atol=1e-8):
            raise RuntimeError("Stage 5 household mortgage debt does not reconcile with the CreditMarket.")
        if not np.allclose(household_debt, market_consumer_debt + market_mortgage_debt, rtol=1e-10, atol=1e-8):
            raise RuntimeError("Stage 5 total household debt does not reconcile with its committed components.")
        if not np.allclose(bank_consumer_assets, market_bank_consumer_assets, rtol=1e-10, atol=1e-8):
            raise RuntimeError("Stage 5 bank consumer-loan assets do not reconcile with the CreditMarket.")

    def refresh_stage5_household_debt_views(self) -> None:
        """Refresh resolver-path household debt views after market-side write-offs."""
        consumer_debt = self.credit_market.compute_outstanding_consumption_loans_by_household()
        mortgage_debt = self.credit_market.compute_outstanding_mortgages_by_household()
        total_debt = consumer_debt + mortgage_debt
        self.households.ts.override_current("consumption_loan_debt", consumer_debt)
        self.households.ts.override_current("mortgage_debt", mortgage_debt)
        self.households.ts.override_current("debt", total_debt)
        self.households.ts.override_current("net_wealth", self.households.compute_net_wealth())
        self.households.ts.override_current("total_consumption_loan_debt", [consumer_debt.sum()])
        self.households.ts.override_current("total_mortgage_debt", [mortgage_debt.sum()])

    def compute_activity_tax_previews(
        self,
        activity_production: np.ndarray,
        reference_target_production: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return tax previews implied by an activity-consistent production plan."""
        expected_output_prices = (1 + self.economy.ts.current("estimated_ppi_inflation")[0]) * self.firms.ts.current(
            "price"
        )
        production_tax_preview = (
            self.central_government.states["Taxes Less Subsidies Rates"][self.firms.states["Industry"]]
            * activity_production
            * expected_output_prices
        )
        pre_credit_corporate_tax_preview = self.firms.estimate_corporate_tax_obligation(
            estimated_growth=self.economy.ts.current("estimated_growth")[0],
            estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
        )
        feasible_scale = np.divide(
            activity_production,
            reference_target_production,
            out=np.zeros_like(activity_production),
            where=reference_target_production > 0.0,
        )
        corporate_tax_preview = pre_credit_corporate_tax_preview * np.maximum(0.0, feasible_scale)
        return production_tax_preview, corporate_tax_preview

    def compute_post_credit_activity_tax_previews(self) -> tuple[np.ndarray, np.ndarray]:
        """Return tax previews implied by the post-credit feasible activity plan."""
        return self.compute_activity_tax_previews(
            activity_production=self.firms.ts.current("activity_finance_feasible_target_production"),
            reference_target_production=self.firms.ts.current("target_production"),
        )

    def compute_pre_credit_production_tax_obligation_preview(self) -> np.ndarray:
        """Return production-tax preview from the pre-credit target plan."""
        target_production = np.maximum(0.0, self.firms.ts.current("target_production"))
        expected_output_prices = (1 + self.economy.ts.current("estimated_ppi_inflation")[0]) * self.firms.ts.current(
            "price"
        )
        return (
            self.central_government.states["Taxes Less Subsidies Rates"][self.firms.states["Industry"]]
            * target_production
            * expected_output_prices
        )

    def current_firm_corporate_tax_obligation_preview(self) -> np.ndarray:
        """Return the corporate-tax preview to reserve during firm settlement."""
        return getattr(
            self,
            "_post_credit_corporate_tax_obligation_preview",
            self.firms.estimate_corporate_tax_obligation(
                estimated_growth=self.economy.ts.current("estimated_growth")[0],
                estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            ),
        )

    def prepare_post_credit_feasible_activity_plan(self) -> None:
        """Revise firm activity after credit clears and before labour clearing."""
        if self.configuration.households.parameters.uses_feasibility_resolver:
            self.households.current_post_grant_residual_shortfall()
        self.firms.begin_operating_revolving_facility_period()
        firm_wage_obligation_preview = self.firms.compute_total_wage_obligation(
            corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            individual_wages=self.individuals.ts.current("employee_income"),
            income_taxes=self.central_government.states["Income Tax"],
            employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
            employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
            cpi=self.economy.current_consumer_price_level(),
        )
        firm_interest_obligation_preview = (
            self.firms.ts.current("firm_settlement_scheduled_interest_due")
            + self.firms.compute_interest_paid_on_deposits(
                bank_interest_rate_on_firm_deposits=self.banks.ts.current("interest_rate_on_firm_deposits"),
                bank_overdraft_rate_on_firm_deposits=self.banks.ts.current("overdraft_rate_on_firm_deposits"),
            )
            + self.firms.ts.current("operating_revolving_interest_paid")
        )
        self.firms.prepare_feasible_activity_plan(
            previous_good_prices=self.economy.ts.current("good_prices"),
            expected_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            wage_obligation_preview=firm_wage_obligation_preview,
            production_tax_obligation_preview=self.compute_pre_credit_production_tax_obligation_preview(),
            corporate_tax_obligation_preview=self.firms.estimate_corporate_tax_obligation(
                estimated_growth=self.economy.ts.current("estimated_growth")[0],
                estimated_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            ),
            interest_obligation_preview=firm_interest_obligation_preview,
            loan_interest_obligation_preview=self.firms.ts.current("firm_settlement_scheduled_interest_due"),
            debt_installment_preview=self.firms.ts.current("firm_settlement_scheduled_principal_due"),
            assume_zero_growth=self.assume_zero_growth,
        )
        self.firms.apply_feasible_labour_demand()
        self.update_firm_wage_offers(replace_current=True)
        (
            self._post_credit_production_tax_obligation_preview,
            self._post_credit_corporate_tax_obligation_preview,
        ) = self.compute_post_credit_activity_tax_previews()

    def prepare_post_labour_realised_feasible_activity_plan(self) -> None:
        """Revise feasible activity after labour clearing, before production is computed."""
        if self.assume_zero_growth or self.firms.configuration.parameters.firm_activity_finance_revision_mode == "none":
            self.firms.ts.activity_finance_realised_feasible_target_production.append(
                self.firms.ts.current("target_production").copy()
            )
            self.firms.ts.activity_finance_realised_labour_scale.append(np.ones(self.firms.ts.current("n_firms")))
            self.firms.ts.activity_finance_realised_feasible_intermediate_inputs.append(
                self.firms.ts.current("target_intermediate_inputs").copy()
            )
            self.firms.ts.activity_finance_realised_feasible_capital_inputs.append(
                self.firms.ts.current("target_capital_inputs").copy()
            )
            self.firms.ts.activity_finance_realised_feasible_technical_investment.append(
                self.firms.ts.current("planned_technical_investment").copy()
            )
            self.firms.ts.activity_finance_realised_feasible_plan_ready.append(True)
            return

        reference_target_production = self.firms.ts.current("target_production").copy()
        expected_lcu_prices = (1 + self.economy.ts.current("estimated_ppi_inflation")[0]) * self.economy.ts.current(
            "good_prices"
        )
        self.firms.revise_activity_against_realised_labour(expected_lcu_prices=expected_lcu_prices)
        (
            self._post_credit_production_tax_obligation_preview,
            self._post_credit_corporate_tax_obligation_preview,
        ) = self.compute_activity_tax_previews(
            activity_production=self.firms.ts.current("activity_finance_realised_feasible_target_production"),
            reference_target_production=reference_target_production,
        )

    def prepare_goods_market_clearing(self) -> None:
        """Prepare for goods market transactions.

        Updates production plans, consumption demands, and prices before
        market clearing.
        """
        self.firms.prepare_goods_market_orders(
            exchange_rate_usd_to_lcu=self.exchange_rate_usd_to_lcu,
            previous_good_prices=self.economy.ts.current("good_prices"),
            expected_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
        )
        if self.configuration.households.parameters.uses_feasibility_resolver:
            # Post-credit Stage 5 consumers must read settled feasibility state.
            self.households.current_post_grant_residual_shortfall()
        subsistence_consumption_shortfall = self.households.prepare_goods_market_clearing(
            exchange_rate_usd_to_lcu=self.exchange_rate_usd_to_lcu,
            subsistence_consumption=self.economy.ts.current("subsistence_consumption"),
        )
        if self.configuration.households.parameters.uses_feasibility_resolver:
            scheduled_consumer_payments = (
                self.credit_market.compute_opening_scheduled_consumption_payments_by_household()
            )
            scheduled_mortgage_payments = self.credit_market.compute_opening_scheduled_mortgage_payments_by_household()
            expected_shape = self.households.current_remaining_subsistence_shortfall().shape
            if scheduled_consumer_payments.shape != expected_shape:
                scheduled_consumer_payments = (
                    self.credit_market.compute_scheduled_consumption_loan_payments_by_household()
                )
            if scheduled_mortgage_payments.shape != expected_shape:
                scheduled_mortgage_payments = self.credit_market.compute_scheduled_mortgage_payments_by_household()
            self.households.record_pre_support_payment_suspension_diagnostics(
                scheduled_consumer_payments=scheduled_consumer_payments,
                scheduled_mortgage_payments=scheduled_mortgage_payments,
            )
            support_by_household = compute_stage5_subsistence_support(
                self.households.current_remaining_subsistence_shortfall()
            )
            self.set_stage5_subsistence_support_by_household(support_by_household)
        self.economy.ts.override_current("subsistence_consumption_shortfall", subsistence_consumption_shortfall)
        self.government_entities.prepare_goods_market_clearing(
            n_industries=self.economy.n_industries,
            exchange_rate_usd_to_lcu=self.exchange_rate_usd_to_lcu,
            exogenous_gov_consumption_before=(
                None
                if "Real Government Consumption (Value)" not in self.exogenous.national_accounts_before.columns
                else self.exogenous.national_accounts_before["Real Government Consumption (Value)"].values.flatten()
            ),
            exogenous_gov_consumption_during=(
                None
                if "Real Government Consumption (Value)" not in self.exogenous.national_accounts_during.columns
                else self.exogenous.national_accounts_during["Real Government Consumption (Value)"].values.flatten()
            ),
            initial_good_prices=self.economy.ts.initial("good_prices"),
            current_good_prices=self.economy.ts.current("good_prices"),
            historic_ppi=np.array(self.economy.ts.historic("ppi")).flatten(),
            expected_inflation=self.economy.ts.current("estimated_ppi_inflation")[0],
            expected_growth=self.economy.ts.current("estimated_growth")[0],
            forecasting_window=self.forecasting_window,
            assume_zero_growth=self.assume_zero_growth,
            assume_zero_noise=self.assume_zero_noise,
        )

    def _excess_demand_finance_potential_inputs(self) -> dict[str, np.ndarray]:
        """Return country-level inputs for excess-demand finance-potential calculations."""
        firm_wage_obligation_preview = self.firms.compute_total_wage_obligation(
            corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            individual_wages=self.individuals.ts.current("employee_income"),
            income_taxes=self.central_government.states["Income Tax"],
            employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
            employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
            cpi=self.economy.current_consumer_price_level(),
        )
        firm_loan_interest_preview = self.firms.ts.current("firm_settlement_scheduled_interest_due")
        firm_interest_obligation_preview = firm_loan_interest_preview + self.firms.compute_interest_paid_on_deposits(
            bank_interest_rate_on_firm_deposits=self.banks.ts.current("interest_rate_on_firm_deposits"),
            bank_overdraft_rate_on_firm_deposits=self.banks.ts.current("overdraft_rate_on_firm_deposits"),
        )
        non_loan_interest_obligation_preview = np.maximum(
            0.0,
            firm_interest_obligation_preview - firm_loan_interest_preview,
        )
        credit_clearer = self.credit_market.functions.get("clearing")
        borrower_credit_room = compute_firm_borrower_credit_room(
            banks=self.banks,
            firms=self.firms,
            allow_short_term_firm_loans=getattr(credit_clearer, "allow_short_term_firm_loans", True),
        )
        expected_lcu_prices = (1 + self.economy.ts.current("estimated_ppi_inflation")[0]) * self.economy.ts.current(
            "good_prices"
        )
        return {
            "expected_lcu_prices": expected_lcu_prices,
            "wage_obligation_preview": firm_wage_obligation_preview,
            "non_loan_interest_obligation_preview": non_loan_interest_obligation_preview,
            "borrower_st_credit_room": borrower_credit_room["short_term"],
            "borrower_lt_credit_room": borrower_credit_room["long_term"],
            "borrower_total_credit_room": borrower_credit_room["total"],
        }

    def compute_excess_demand_finance_potential_capacities(self, ordinary_bank_supply: float) -> None:
        """Compute supply-adjusted finance-potential capacities before excess-demand assignment."""
        self.firms.reset_excess_demand_finance_potential_transients()
        inputs = self._excess_demand_finance_potential_inputs()
        self.firms.compute_excess_demand_finance_potential_capacities(
            **inputs,
            q_sold=self.firms.transactor_seller_states["Real Amount sold"],
            ordinary_bank_supply=ordinary_bank_supply,
        )

    def append_excess_demand_finance_potential_diagnostics(self) -> None:
        """Append borrower-side finance-potential diagnostics after goods clearing."""
        if getattr(self.firms, "_last_excess_demand_finance_potential", None) is None:
            inputs = self._excess_demand_finance_potential_inputs()
            self.firms.append_excess_demand_finance_potential_diagnostics(
                **inputs,
                q_sold=self.firms.transactor_seller_states["Real Amount sold"],
                q_excess=self.firms.transactor_seller_states["Real Excess Demand"],
            )
            return

        self.firms.append_cached_excess_demand_finance_potential_diagnostics(
            q_excess=self.firms.transactor_seller_states["Real Excess Demand"],
        )

    def supply_adjusted_excess_demand_capacity(self) -> np.ndarray:
        """Return supply-adjusted firm-level excess-demand capacity for this country."""
        return self.firms.get_supply_adjusted_excess_demand_capacity()

    def _process_ficp_forgiveness_events(
        self, period_index: int | None = None
    ) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        """Recognise 4c losses and remove only forgiven consumer-loan cells."""
        n_banks = int(self.banks.ts.current("n_banks"))
        n_households = int(self.households.ts.current("n_households"))
        balance_tolerance = 1e-6
        zero_by_bank = np.zeros(n_banks)
        zero_by_cell = np.zeros((n_banks, n_households))
        zero_exclusion = np.zeros_like(zero_by_cell, dtype=bool)
        event_mask = np.asarray(self.households.ts.current("ficp_forgiveness_event"), dtype=bool)
        event_emitted = np.asarray(self.households.ts.current("ficp_forgiveness_event_emitted"), dtype=bool)
        event_processed = np.asarray(self.households.ts.current("ficp_forgiveness_event_processed"), dtype=bool)
        event_stage = np.asarray(self.households.ts.current("ficp_forgiveness_event_stage"), dtype=float)
        episode_ids = np.asarray(self.households.ts.current("ficp_forgiveness_event_episode_id"), dtype=float)
        horizon_end_periods = np.asarray(
            self.households.ts.current("ficp_forgiveness_event_horizon_end_period"), dtype=float
        )
        contractual_events = np.asarray(
            self.households.ts.current("ficp_forgiveness_event_residual_contractual_principal"), dtype=float
        )
        principal_arrears_events = np.asarray(
            self.households.ts.current("ficp_forgiveness_event_residual_principal_arrears"), dtype=float
        )
        interest_arrears_events = np.asarray(
            self.households.ts.current("ficp_forgiveness_event_residual_interest_arrears"), dtype=float
        )
        household_vectors = (
            ("ficp event mask", event_mask),
            ("ficp emitted flags", event_emitted),
            ("ficp processed flags", event_processed),
            ("ficp event stages", event_stage),
            ("ficp episode IDs", episode_ids),
            ("ficp horizon end periods", horizon_end_periods),
            ("contractual principal", contractual_events),
            ("principal arrears", principal_arrears_events),
            ("interest arrears", interest_arrears_events),
        )
        for name, values in household_vectors:
            if values.shape != (n_households,):
                raise RuntimeError(f"{name} must contain exactly one value per household.")
        event_stage_before = event_stage.copy()
        if (
            not np.all(np.isfinite(event_stage))
            or np.any(event_stage != np.floor(event_stage))
            or np.any((event_stage < 0.0) | (event_stage > 3.0))
        ):
            raise RuntimeError("FICP forgiveness event stages must be integers from zero through three.")
        completed_mask = event_processed & ~event_mask & ~event_emitted & (event_stage == 3.0)
        if np.any(event_mask & ~event_emitted) or np.any(event_processed & ~event_emitted & ~completed_mask):
            raise RuntimeError("FICP event flags must be consistent with the emitted event record.")
        pending_mask = event_mask & ~event_processed
        if np.any((event_stage == 3.0) & pending_mask):
            raise RuntimeError("A completed FICP forgiveness event cannot remain unprocessed.")
        if np.any(episode_ids[pending_mask] <= 0.0) or np.any(
            episode_ids[pending_mask] != np.floor(episode_ids[pending_mask])
        ):
            raise RuntimeError("FICP forgiveness episode IDs must be positive integers.")
        if np.any(horizon_end_periods[pending_mask] < 0.0) or np.any(
            horizon_end_periods[pending_mask] != np.floor(horizon_end_periods[pending_mask])
        ):
            raise RuntimeError("FICP forgiveness horizon periods must be non-negative integers.")
        if (
            np.any(contractual_events[pending_mask] < 0.0)
            or np.any(principal_arrears_events[pending_mask] < 0.0)
            or np.any(interest_arrears_events[pending_mask] < 0.0)
            or not np.all(np.isfinite(contractual_events))
            or not np.all(np.isfinite(principal_arrears_events))
            or not np.all(np.isfinite(interest_arrears_events))
        ):
            raise RuntimeError("FICP forgiveness event balances must be finite and non-negative.")

        stage_zero = pending_mask & (event_stage == 0.0)
        stage_one = pending_mask & (event_stage == 1.0)
        stage_two = pending_mask & (event_stage >= 2.0)
        durable_event_mask = stage_one | stage_two

        durable_principal = np.asarray(
            self.credit_market.ts.current("consumer_default_principal_by_cell"), dtype=float
        ).copy()
        durable_principal_arrears = np.asarray(
            self.credit_market.ts.current("consumer_default_principal_arrears_by_cell"), dtype=float
        ).copy()
        durable_interest_arrears = np.asarray(
            self.credit_market.ts.current("consumer_default_interest_arrears_by_cell"), dtype=float
        ).copy()
        durable_exclusion = np.asarray(
            self.credit_market.ts.current("consumer_terminal_removal_exclusion_by_cell"), dtype=bool
        ).copy()
        durable_episode_ids = np.asarray(
            self.credit_market.ts.current("consumer_terminal_removal_episode_id_by_cell"), dtype=float
        ).copy()
        durable_fields = (
            ("consumer principal", durable_principal),
            ("consumer principal arrears", durable_principal_arrears),
            ("consumer interest arrears", durable_interest_arrears),
            ("consumer removal exclusion", durable_exclusion),
            ("consumer removal episode IDs", durable_episode_ids),
        )
        for name, values in durable_fields:
            if values.shape != zero_by_cell.shape:
                raise RuntimeError(f"Durable {name} carrier has an invalid bank-household shape.")
        for values in (durable_principal, durable_principal_arrears, durable_interest_arrears, durable_episode_ids):
            if not np.all(np.isfinite(values)):
                raise RuntimeError("Durable FICP carriers must contain finite values.")
            if np.any(values < 0.0):
                raise RuntimeError("Durable FICP carriers must contain non-negative values.")
        durable_component_mask = (
            (durable_principal > 0.0) | (durable_principal_arrears > 0.0) | (durable_interest_arrears > 0.0)
        )
        if not np.array_equal(durable_exclusion, durable_component_mask):
            raise RuntimeError("Durable FICP exclusion does not match its positive component carriers.")
        # The durable carrier is an episode-scoped handoff for stage-one/stage-
        # two events. A completed event can therefore remain in the previous
        # carrier while another household still has a pending event, so the old
        # global `if not any(pending_mask)` cleanup was insufficient. Clear
        # completed carriers before validating the current pending event set.
        # Live consumer debt is not a valid stale-carrier test: a household may
        # originate a new loan after the old FICP episode has completed.
        carrier_event_mask = durable_event_mask[None, :]
        stale_carrier_mask = durable_component_mask & ~carrier_event_mask
        if np.any(stale_carrier_mask):
            durable_principal = np.where(carrier_event_mask, durable_principal, 0.0)
            durable_principal_arrears = np.where(carrier_event_mask, durable_principal_arrears, 0.0)
            durable_interest_arrears = np.where(carrier_event_mask, durable_interest_arrears, 0.0)
            durable_exclusion = durable_exclusion & carrier_event_mask
            durable_episode_ids = np.where(durable_exclusion, durable_episode_ids, 0.0)
            durable_component_mask = (
                (durable_principal > 0.0) | (durable_principal_arrears > 0.0) | (durable_interest_arrears > 0.0)
            )
        if np.any(pending_mask):
            expected_durable_episode_ids = np.where(durable_exclusion, episode_ids[None, :], 0.0)
            episode_id_mismatch = durable_episode_ids != expected_durable_episode_ids
            if np.any(episode_id_mismatch):
                mismatch_details = []
                for bank_id, household_id in zip(*np.where(episode_id_mismatch)):
                    mismatch_details.append(
                        {
                            "bank_id": int(bank_id),
                            "household_id": int(household_id),
                            "pending_episode_id": float(episode_ids[household_id]),
                            "durable_episode_id": float(durable_episode_ids[bank_id, household_id]),
                            "durable_exclusion": bool(durable_exclusion[bank_id, household_id]),
                            "durable_principal": float(durable_principal[bank_id, household_id]),
                            "durable_principal_arrears": float(durable_principal_arrears[bank_id, household_id]),
                            "durable_interest_arrears": float(durable_interest_arrears[bank_id, household_id]),
                        }
                    )
                raise RuntimeError(
                    "Durable FICP episode IDs do not match the pending event identities: "
                    f"period_index={period_index!r}, mismatches={mismatch_details}."
                )
            if np.any(durable_component_mask & ~durable_event_mask[None, :]):
                raise RuntimeError("Durable FICP carriers contain cells outside the pending staged events.")
            for household_id in np.flatnonzero(durable_event_mask):
                if (
                    not np.allclose(
                        durable_principal[:, household_id].sum(),
                        contractual_events[household_id] + principal_arrears_events[household_id],
                        rtol=1e-10,
                        atol=1e-8,
                    )
                    or not np.allclose(
                        durable_principal_arrears[:, household_id].sum(),
                        principal_arrears_events[household_id],
                        rtol=1e-10,
                        atol=1e-8,
                    )
                    or not np.allclose(
                        durable_interest_arrears[:, household_id].sum(),
                        interest_arrears_events[household_id],
                        rtol=1e-10,
                        atol=1e-8,
                    )
                ):
                    raise RuntimeError("Durable FICP carriers do not reconcile with the event payload.")

        pending_events = tuple(
            (int(household_id), int(episode_ids[household_id])) for household_id in np.flatnonzero(pending_mask)
        )
        if not np.any(pending_mask):
            self.credit_market.ts.consumer_default_principal_by_cell.append(zero_by_cell.copy())
            self.credit_market.ts.consumer_default_principal_arrears_by_cell.append(zero_by_cell.copy())
            self.credit_market.ts.consumer_default_interest_arrears_by_cell.append(zero_by_cell.copy())
            self.credit_market.ts.consumer_terminal_removal_exclusion_by_cell.append(zero_exclusion.copy())
            self.credit_market.ts.consumer_terminal_removal_episode_id_by_cell.append(zero_by_cell.copy())
            self.credit_market._consumer_terminal_removal_exclusion = zero_exclusion.copy()
            _append_ficp_bank_effects(
                self.banks,
                loan_writeoff=zero_by_bank,
                principal_arrears=zero_by_bank,
                interest_income_loss=zero_by_bank,
                npl=zero_by_bank,
                npl_denominator=zero_by_bank,
            )
            return zero_exclusion, ()

        stage_two_same_period = stage_two & (period_index is None or horizon_end_periods == float(period_index))
        stage_two_prior_period = stage_two & ~stage_two_same_period
        if np.any(stage_two_prior_period) and period_index is None:
            stage_two_prior_period = np.zeros_like(stage_two, dtype=bool)
            stage_two_same_period = stage_two.copy()

        live_writeoff = self.credit_market.snapshot_consumer_default_writeoff(stage_zero)
        stage_one_principal = np.where(stage_one[None, :], durable_principal, 0.0)
        stage_one_principal_arrears = np.where(stage_one[None, :], durable_principal_arrears, 0.0)
        stage_one_interest_arrears = np.where(stage_one[None, :], durable_interest_arrears, 0.0)
        stage_one_exclusion = np.where(stage_one[None, :], durable_exclusion, False)
        newly_written_principal = live_writeoff.principal_by_cell + stage_one_principal
        newly_written_principal_arrears = live_writeoff.principal_arrears_by_cell + stage_one_principal_arrears
        newly_written_interest_arrears = live_writeoff.interest_arrears_by_cell + stage_one_interest_arrears
        newly_written_principal_by_bank = newly_written_principal.sum(axis=1)
        newly_written_principal_arrears_by_bank = newly_written_principal_arrears.sum(axis=1)
        newly_written_interest_arrears_by_bank = newly_written_interest_arrears.sum(axis=1)

        if np.any(stage_zero):
            contractual_principal, principal_arrears, interest_arrears = (
                self.credit_market.current_consumer_debt_components_by_household()
            )
            for household_id in np.flatnonzero(stage_zero):
                if (
                    not np.allclose(
                        contractual_principal[household_id],
                        contractual_events[household_id],
                        rtol=1e-10,
                        atol=balance_tolerance,
                    )
                    or not np.allclose(
                        principal_arrears[household_id],
                        principal_arrears_events[household_id],
                        rtol=1e-10,
                        atol=balance_tolerance,
                    )
                    or not np.allclose(
                        interest_arrears[household_id],
                        interest_arrears_events[household_id],
                        rtol=1e-10,
                        atol=balance_tolerance,
                    )
                ):
                    raise RuntimeError(
                        "FICP forgiveness event does not reconcile with the live consumer-loan state: "
                        f"period_index={period_index!r}, household_id={household_id}, "
                        f"episode_id={episode_ids[household_id]}, "
                        f"live=(contractual={contractual_principal[household_id]}, "
                        f"principal_arrears={principal_arrears[household_id]}, "
                        f"interest_arrears={interest_arrears[household_id]}), "
                        f"event=(contractual={contractual_events[household_id]}, "
                        f"principal_arrears={principal_arrears_events[household_id]}, "
                        f"interest_arrears={interest_arrears_events[household_id]})."
                    )
            if not np.allclose(
                live_writeoff.principal_by_bank.sum(),
                contractual_events[stage_zero].sum() + principal_arrears_events[stage_zero].sum(),
                rtol=1e-10,
                atol=balance_tolerance,
            ):
                raise RuntimeError("Consumer principal write-off does not reconcile with the accepted 4b event.")

        stored_denominator = np.asarray(self.banks.ts.current("consumer_default_npl_denominator"), dtype=float)
        if stored_denominator.shape != (n_banks,) or not np.all(np.isfinite(stored_denominator)):
            raise RuntimeError("Stored FICP NPL denominators must contain finite bank values.")
        stage_one_banks = np.any(stage_one_exclusion, axis=1)
        npl_denominator = live_writeoff.npl_denominator_by_bank.copy()
        if np.any(stage_one_banks):
            npl_denominator[stage_one_banks] = stored_denominator[stage_one_banks]
        if np.any(stage_two_same_period):
            npl_denominator = stored_denominator.copy()
        if np.any(stage_two_prior_period) and not np.any(stage_one_banks):
            npl_denominator = live_writeoff.npl_denominator_by_bank.copy()
        if np.any(stage_one_banks) and np.any(stage_zero):
            npl_denominator[stage_one_banks] = stored_denominator[stage_one_banks]

        current_numerator = zero_by_bank
        current_loss_base = zero_by_bank
        current_principal_arrears_base = zero_by_bank
        current_interest_income_loss_base = zero_by_bank
        if np.any(stage_two_same_period):
            current_numerator = (
                np.asarray(self.banks.ts.current("consumer_default_npl"), dtype=float) * stored_denominator
            )
            current_loss_base = np.asarray(self.banks.ts.current("consumer_default_loan_writeoff"), dtype=float)
            current_principal_arrears_base = np.asarray(
                self.banks.ts.current("consumer_default_principal_arrears"), dtype=float
            )
            current_interest_income_loss_base = np.asarray(
                self.banks.ts.current("consumer_default_interest_income_loss"), dtype=float
            )
        npl = np.divide(
            current_numerator + newly_written_principal_by_bank,
            npl_denominator,
            out=np.zeros(n_banks),
            where=npl_denominator > 0.0,
        )
        exclusion_mask = durable_exclusion | (
            live_writeoff.removal_mask
            & (
                (live_writeoff.principal_by_cell > 0.0)
                | (live_writeoff.principal_arrears_by_cell > 0.0)
                | (live_writeoff.interest_arrears_by_cell > 0.0)
            )
        )
        credit_market_snapshot = self.credit_market.states["cons_loans"].copy()
        principal_arrears_snapshot = self.credit_market._consumer_principal_arrears_by_cell.copy()
        interest_arrears_snapshot = self.credit_market._consumer_interest_arrears_by_cell.copy()
        exclusion_snapshot = self.credit_market._consumer_terminal_removal_exclusion.copy()
        carrier_fields = (
            "consumer_default_principal_by_cell",
            "consumer_default_principal_arrears_by_cell",
            "consumer_default_interest_arrears_by_cell",
            "consumer_terminal_removal_exclusion_by_cell",
            "consumer_terminal_removal_episode_id_by_cell",
        )
        bank_fields = (
            "consumer_default_loan_writeoff",
            "total_consumer_default_loan_writeoff",
            "consumer_default_principal_arrears",
            "total_consumer_default_principal_arrears",
            "consumer_default_credit_loss",
            "total_consumer_default_credit_loss",
            "consumer_default_interest_income_loss",
            "total_consumer_default_interest_income_loss",
            "consumer_default_npl",
            "consumer_default_npl_denominator",
        )
        carrier_lengths = {field: len(self.credit_market.ts.dicts[field]) for field in carrier_fields}
        bank_lengths = {field: len(self.banks.ts.dicts[field]) for field in bank_fields}
        try:
            if np.any(stage_zero):
                self.credit_market.ts.consumer_default_principal_by_cell.append(newly_written_principal.copy())
                self.credit_market.ts.consumer_default_principal_arrears_by_cell.append(
                    newly_written_principal_arrears.copy()
                )
                self.credit_market.ts.consumer_default_interest_arrears_by_cell.append(
                    newly_written_interest_arrears.copy()
                )
                self.credit_market.ts.consumer_terminal_removal_exclusion_by_cell.append(exclusion_mask.copy())
                self.credit_market.ts.consumer_terminal_removal_episode_id_by_cell.append(
                    np.where(exclusion_mask, episode_ids[None, :], 0.0)
                )

                removed = self.credit_market.remove_consumer_loans_by_cell(live_writeoff.removal_mask)
                if not np.allclose(
                    removed.principal_by_cell,
                    live_writeoff.principal_by_cell,
                    rtol=1e-10,
                    atol=1e-8,
                ):
                    raise RuntimeError("Consumer principal changed between the 4c snapshot and removal.")
            self.credit_market._consumer_terminal_removal_exclusion = exclusion_mask.copy()
            if np.any(stage_zero):
                event_stage[stage_zero] = 1.0
                self.households.ts.override_current("ficp_forgiveness_event_stage", event_stage)

            has_new_accounting = np.any(stage_zero | stage_one)
            if has_new_accounting:
                staged_denominator = np.any(stage_one) or np.any(stage_two_same_period)
                if not staged_denominator:
                    self.banks.ts.consumer_default_npl_denominator.append(npl_denominator.copy())
                _append_ficp_bank_effects(
                    self.banks,
                    loan_writeoff=current_loss_base + newly_written_principal_by_bank,
                    principal_arrears=current_principal_arrears_base + newly_written_principal_arrears_by_bank,
                    interest_income_loss=current_interest_income_loss_base + newly_written_interest_arrears_by_bank,
                    npl=npl,
                    npl_denominator=npl_denominator,
                    cumulative_loan_writeoff=newly_written_principal_by_bank,
                    cumulative_principal_arrears=newly_written_principal_arrears_by_bank,
                    cumulative_interest_income_loss=newly_written_interest_arrears_by_bank,
                    append_npl_denominator=False,
                )
                event_stage[stage_zero | stage_one] = 2.0
                self.households.ts.override_current("ficp_forgiveness_event_stage", event_stage)
            elif np.any(stage_two_prior_period):
                _append_ficp_bank_effects(
                    self.banks,
                    loan_writeoff=zero_by_bank,
                    principal_arrears=zero_by_bank,
                    interest_income_loss=zero_by_bank,
                    npl=zero_by_bank,
                    npl_denominator=zero_by_bank,
                )
        except Exception:
            self.credit_market.states["cons_loans"][:] = credit_market_snapshot
            self.credit_market._consumer_principal_arrears_by_cell[:] = principal_arrears_snapshot
            self.credit_market._consumer_interest_arrears_by_cell[:] = interest_arrears_snapshot
            self.credit_market._consumer_terminal_removal_exclusion = exclusion_snapshot
            for field, length in carrier_lengths.items():
                del self.credit_market.ts.dicts[field][length:]
            for field, length in bank_lengths.items():
                del self.banks.ts.dicts[field][length:]
            self.households.ts.override_current("ficp_forgiveness_event_stage", event_stage_before)
            raise
        return exclusion_mask, pending_events

    def record_dividend_fund_settlement_receipts(self) -> None:
        """Record actual payer-funded dividend receipts without crediting deposits.

        Declarations are settled one period later. This method only records the
        source-specific household receipt carriers; Step 3 owns their income
        treatment. The fixed quota is therefore the sole allocation authority.
        """
        quota = np.asarray(self.households.states["dividend_fund_ownership_quota"], dtype=float)
        if quota.ndim != 1 or not np.all(np.isfinite(quota)) or np.any(quota < -1e-9):
            raise ValueError("Household dividend-fund ownership quotas must be finite and non-negative.")
        quota = np.maximum(quota, 0.0)
        quota_sum = float(quota.sum())
        if quota_sum > 0.0 and not np.isclose(quota_sum, 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("Positive household dividend-fund ownership quotas must sum to one.")

        firm_debit = np.asarray(self.firms.ts.current("dividend_fund_settlement_debit"), dtype=float)
        bank_debit = np.asarray(self.banks.ts.current("dividend_fund_settlement_debit"), dtype=float)
        if not np.all(np.isfinite(firm_debit)) or not np.all(np.isfinite(bank_debit)):
            raise ValueError("Dividend-fund settlement debits must be finite.")
        firm_receipts = quota * float(np.maximum(firm_debit, 0.0).sum())
        bank_receipts = quota * float(np.maximum(bank_debit, 0.0).sum())
        ts = self.households.ts
        ts.dividend_fund_settled_firm_distribution.append(firm_receipts)
        ts.dividend_fund_settled_bank_distribution.append(bank_receipts)
        ts.dividend_fund_ownership_quota.append(quota.copy())
        ts.dividend_fund_total_settled_distribution.append([float(firm_receipts.sum() + bank_receipts.sum())])
        ts.dividend_fund_quota_sum.append([quota_sum])
        ts.dividend_fund_firm_settlement_identity_error.append([float(firm_receipts.sum() - firm_debit.sum())])
        ts.dividend_fund_bank_settlement_identity_error.append([float(bank_receipts.sum() - bank_debit.sum())])
        ts.dividend_fund_settlement_identity_error.append(
            [float(firm_receipts.sum() + bank_receipts.sum() - firm_debit.sum() - bank_debit.sum())]
        )

    def record_diagnostic_dividend_fund(self) -> None:
        """Declare next-period profit payouts from current eligible capacity."""
        from macromodel.country.diagnostic_dividend_fund import compute_diagnostic_dividend_fund

        firm_cash_profit_after_settlement = (
            self.firms.ts.current("nominal_amount_sold_in_lcu")
            - self.firms.ts.current("total_wage")
            - self.firms.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            - self.firms.direct_tfp_investment_cash_expense()
            - self.firms.ts.current("taxes_paid_on_production")
            - self.firms.ts.current("corporate_taxes_paid")
            - self.firms.ts.current("interest_paid")
            - self.firms.ts.current("debt_installments")
            - self.firms.ts.current("operating_revolving_repayment")
        )
        result = compute_diagnostic_dividend_fund(
            beginning_ifa=self.households.ts.prev("illiquid_financial_assets"),
            firm_cash_profit_after_settlement=firm_cash_profit_after_settlement,
            firm_closing_deposits=self.firms.ts.current("deposits"),
            firm_unpaid_interest=self.firms.ts.current("firm_settlement_unpaid_interest"),
            firm_closing_principal_arrears=self.firms.ts.current("firm_settlement_closing_principal_arrears"),
            firm_residual_overdraft_exposure=self.firms.ts.current("firm_settlement_residual_overdraft_exposure"),
            firm_default_flag=self.firms.ts.current("firm_settlement_default_flag"),
            bank_cash_distributable_profits=self.banks.ts.current("cash_distributable_profits"),
            ownership_quota=self.households.states["dividend_fund_ownership_quota"],
        )
        wealth_function = self.households.functions["wealth"]
        firm_payout_ratio = float(getattr(wealth_function, "dividend_fund_firm_payout_ratio", 0.0))
        bank_payout_ratio = float(getattr(wealth_function, "dividend_fund_bank_payout_ratio", 0.0))
        has_owners = float(result.ownership_quota.sum()) > 0.0
        firm_declared = (
            result.firm_distributable_profit_candidate * firm_payout_ratio
            if has_owners
            else np.zeros_like(result.firm_distributable_profit_candidate)
        )
        bank_declared = (
            result.bank_distributable_profit_candidate * bank_payout_ratio
            if has_owners
            else np.zeros_like(result.bank_distributable_profit_candidate)
        )
        firm_retained = result.firm_distributable_profit_candidate - firm_declared
        bank_retained = result.bank_distributable_profit_candidate - bank_declared
        ts = self.households.ts
        ts.dividend_fund_declared_firm_distribution.append(result.ownership_quota * float(firm_declared.sum()))
        ts.dividend_fund_declared_bank_distribution.append(result.ownership_quota * float(bank_declared.sum()))
        ts.dividend_fund_total_firm_distributable_profit_candidate.append(
            [float(result.firm_distributable_profit_candidate.sum())]
        )
        ts.dividend_fund_total_bank_distributable_profit_candidate.append(
            [float(result.bank_distributable_profit_candidate.sum())]
        )
        ts.dividend_fund_total_firm_declared_distribution.append([float(firm_declared.sum())])
        ts.dividend_fund_total_bank_declared_distribution.append([float(bank_declared.sum())])
        ts.dividend_fund_total_firm_retained_capacity.append([float(firm_retained.sum())])
        ts.dividend_fund_total_bank_retained_capacity.append([float(bank_retained.sum())])
        ts.dividend_fund_firm_retained_capacity_identity_error.append(
            [float((firm_declared + firm_retained - result.firm_distributable_profit_candidate).sum())]
        )
        ts.dividend_fund_bank_retained_capacity_identity_error.append(
            [float((bank_declared + bank_retained - result.bank_distributable_profit_candidate).sum())]
        )
        ts.dividend_fund_fund_identity_error.append([result.fund_identity_error])
        self.firms.ts.dividend_fund_cash_distributable_profit_candidate.append(
            result.firm_distributable_profit_candidate
        )
        self.firms.ts.dividend_fund_distress_gate_passed.append(result.firm_distress_gate_passed)
        self.firms.ts.dividend_fund_declared_distribution.append(firm_declared)
        self.firms.ts.dividend_fund_retained_capacity.append(firm_retained)
        self.banks.ts.dividend_fund_cash_distributable_profit_candidate.append(
            result.bank_distributable_profit_candidate
        )
        self.banks.ts.dividend_fund_declared_distribution.append(bank_declared)
        self.banks.ts.dividend_fund_retained_capacity.append(bank_retained)

    def update_realised_metrics(self, period_index: int | None = None) -> None:
        """Update realized economic outcomes after market clearing.

        This method coordinates the comprehensive updating of all economic metrics after markets
        have cleared. It processes the results of market interactions and computes derived
        economic indicators across eight major categories:

        1. Economic Indicators:
           - Price indices (PPI, CPI)
           - Economic growth rates
           - House price indices
           - Labor market metrics (employment, unemployment)
           - Rental market aggregates

        2. Global Trade and Capital Formation:
           - International trade flows
           - Import/export tracking
           - Gross fixed capital formation
           - Input costs (intermediate and capital)

        3. Firm Production and Costs:
           - Production volumes and nominal values
           - Wages and labor costs
           - Inventory management
           - Input utilization
           - Emissions (if enabled)

        4. Firm Financial Metrics:
           - Sales and revenue
           - Profits and taxes
           - Corporate financial positions
           - Insolvency metrics
           - Debt aggregates

        5. Individual and Household Income:
           - Employment income
           - Social transfers
           - Financial asset income
           - Rental income
           - Total household income

        6. Household Wealth and Consumption:
           - Consumption patterns
           - Investment decisions
           - Wealth positions
           - Debt management
           - Insolvency handling

        7. Government and Banking:
           - Tax collection
           - Government revenue and deficit
           - Bank profits and equity
           - Interest payments
           - Market share metrics

        8. GDP and National Accounts:
           - GDP components
           - Value added
           - National income
           - Sectoral balances

        The method ensures consistency between micro-level agent behaviors and macro-level
        economic outcomes, maintaining stock-flow consistency throughout the economy.

        Note:
            This method should be called after all markets (goods, labor, credit, housing)
            have cleared and agents have executed their planned transactions.

        Side Effects:
            Updates numerous time series variables across all economic agents and markets,
            reflecting the realized outcomes of the current period's economic activity.
        """
        # Firms distribute bought goods
        self.firms.distribute_bought_goods()

        current_taxes_paid_on_production = (
            self.central_government.states["Taxes Less Subsidies Rates"][self.firms.states["Industry"]]
            * self.firms.ts.current("production")
            * self.firms.ts.current("price")
        )
        current_firm_total_sales = (
            self.firms.ts.current("price") * self.firms.ts.current("production") - current_taxes_paid_on_production
        )
        sectoral_sales = np.bincount(
            self.firms.states["Industry"],
            weights=current_firm_total_sales,
            minlength=self.economy.n_industries,
        )

        # A1. ECONOMIC INDICATORS
        # Update core economic indicators and market metrics
        self.economy.compute_price_indicators(
            firm_real_amount_bought=self.firms.ts.current("real_amount_bought"),
            firm_nominal_amount_spent=self.firms.ts.current("nominal_amount_spent_in_lcu"),
            household_real_amount_bought=self.households.ts.current("real_amount_bought"),
            household_nominal_amount_spent=self.households.ts.current("nominal_amount_spent_in_lcu"),
            government_real_amount_bought=self.government_entities.ts.current("real_amount_bought"),
            government_nominal_amount_spent=self.government_entities.ts.current("nominal_amount_spent_in_lcu"),
            firms_real_amount_bought_as_capital_goods=self.firms.ts.current("real_amount_bought_as_capital_goods"),
            sectoral_producer_sales=sectoral_sales,
            sectoral_household_consumption=self.households.ts.current("industry_consumption"),
        )
        self.economy.compute_inflation()
        self.economy.compute_cpi_yoy_inflation(
            exogenous_cpi_inflation_before=self.exogenous.inflation_before["CPI Inflation"].values
        )
        self.economy.compute_growth(
            current_production=self.firms.ts.current("production"),
            prev_production=self.firms.ts.prev("production"),
            industries=self.firms.states["Industry"],
        )
        self.economy.compute_house_price_index(
            current_property_values=self.housing_market.ts.current("property_values"),
            previous_property_values=self.housing_market.ts.prev("property_values"),
        )
        self.economy.compute_labour_market_aggregates(
            current_individual_activity_status=self.individuals.states["Activity Status"],
            current_firm_labour_inputs=self.firms.ts.current("labour_inputs"),
            current_desired_firm_labour_inputs=self.firms.ts.current("desired_labour_inputs"),
            current_firm_number_of_employees=self.firms.ts.current("number_of_employees"),
            num_ind_employed_before_cleaning=self.labour_market.ts.current("num_employed_individuals_before_clearing")[
                0
            ],
            num_ind_newly_joining=self.labour_market.ts.current("num_individuals_newly_joining")[0],
            num_ind_newly_leaving=self.labour_market.ts.current("num_individuals_newly_leaving")[0],
        )
        self.economy.compute_rental_market_aggregates(
            real_rent_paid=self.households.ts.current("rent"),
            imp_rent_paid=self.households.ts.current("rent_imputed"),
            rental_income=self.households.ts.current("income_rental"),
        )

        # B1. GLOBAL TRADE AND CAPITAL FORMATION
        # Record international trade flows and capital formation
        self.economy.record_global_trade(
            firms=self.firms,
            households=self.households,
            government_entities=self.government_entities,
            tau_export=self.central_government.states["Export Tax"],
        )

        # Update gross fixed capital formation
        self.firms.ts.gross_fixed_capital_formation.append(
            self.firms.compute_gross_fixed_capital_formation(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # C1. FIRM PRODUCTION AND COSTS
        # Update input costs and production metrics
        self.firms.update_total_newly_bought_costs(
            current_good_prices=self.economy.ts.current("good_prices"),
        )

        # Execute and record productivity investment after capital purchases are known
        self.firms.execute_productivity_investment()
        if not self.assume_zero_growth:
            self.firms.update_tfp()
            self.firms.update_technical_coefficients()

        self.firms.ts.demand.append(self.firms.compute_demand())

        self.firms.ts.production_nominal.append(
            self.firms.compute_nominal_production(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )

        # C2. WAGES AND LABOR
        self.firms.update_total_wages_paid(
            corresponding_firm=self.individuals.states["Corresponding Firm ID"],
            individual_wages=self.individuals.ts.current("employee_income"),
            income_taxes=self.central_government.states["Income Tax"],
            employee_social_insurance_tax=self.central_government.states["Employee Social Insurance Tax"],
            employer_social_insurance_tax=self.central_government.states["Employer Social Insurance Tax"],
            cpi=self.economy.current_consumer_price_level(),
        )

        # C3. EMISSIONS AND INVENTORY
        if self.add_emissions:
            readjusted_factors = (
                self.emission_factors_lcu / self.economy.ts.current("good_prices")[self.emitting_indices]
            )
            readjusted_factors_ch4 = (
                self.emission_factors_lcu_ch4 / self.economy.ts.current("good_prices")[self.emitting_indices_ch4]
                if self.emission_factors_lcu_ch4 is not None
                else None
            )
            self.firms.update_emissions(
                readjusted_factors=readjusted_factors,
                emitting_indices=self.emitting_indices,
                use_emission_multiplier=self.use_emission_multiplier,
                readjusted_factors_ch4=readjusted_factors_ch4,
                emitting_indices_ch4=self.emitting_indices_ch4,
            )

        self.firms.ts.used_intermediate_inputs.append(self.firms.compute_used_intermediate_inputs())
        self.firms.ts.used_intermediate_inputs_costs.append(
            self.firms.compute_used_intermediate_inputs_costs(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.used_capital_inputs.append(self.firms.compute_used_capital_inputs())
        self.firms.ts.used_capital_inputs_costs.append(
            self.firms.compute_used_capital_inputs_costs(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.inventory.append(self.firms.compute_inventory())
        self.firms.ts.unsold_output.append(
            self.firms.ts.current("production") - self.firms.ts.current("real_amount_sold")
        )
        self.firms.ts.inventory_nominal.append(
            self.firms.compute_nominal_inventory(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.intermediate_inputs_stock.append(self.firms.compute_intermediate_inputs_stock())
        self.firms.ts.intermediate_inputs_stock_value.append(
            self.firms.compute_intermediate_inputs_stock_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.intermediate_inputs_stock_industry.append(
            self.firms.ts.current("intermediate_inputs_stock").sum(axis=0)
        )
        self.firms.ts.capital_inputs_stock.append(self.firms.compute_capital_inputs_stock())
        self.firms.ts.capital_inputs_stock_value.append(
            self.firms.compute_capital_inputs_stock_value(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.ts.capital_inputs_stock_industry.append(self.firms.ts.current("capital_inputs_stock").sum(axis=0))
        self.firms.ts.capital_depreciation_costs.append(self.firms.compute_capital_depreciation_costs())

        # D1. FIRM FINANCIAL METRICS
        # Update firm financial positions
        firm_interest_paid_on_deposits = self.firms.compute_interest_paid_on_deposits(
            bank_interest_rate_on_firm_deposits=self.banks.ts.current("interest_rate_on_firm_deposits"),
            bank_overdraft_rate_on_firm_deposits=self.banks.ts.current("overdraft_rate_on_firm_deposits"),
        )
        firm_corporate_tax_obligation_preview = self.current_firm_corporate_tax_obligation_preview()
        self.firms.draw_operating_revolving_facility(
            intermediate_purchases=self.firms.ts.current("total_intermediate_inputs_bought_costs"),
            taxes_paid_on_production=current_taxes_paid_on_production,
            non_loan_interest_paid=(
                firm_interest_paid_on_deposits + self.firms.ts.current("operating_revolving_interest_paid")
            ),
            corporate_tax_obligation_preview=firm_corporate_tax_obligation_preview,
        )
        firm_debt_settlement = self.firms.plan_firm_debt_settlement(
            taxes_paid_on_production=current_taxes_paid_on_production,
            interest_paid_on_deposits=(
                firm_interest_paid_on_deposits + self.firms.ts.current("operating_revolving_interest_paid")
            ),
            corporate_tax_obligation_preview=firm_corporate_tax_obligation_preview,
        )
        cash_available_after_debt_service = (
            firm_debt_settlement["cash_after_tax_reserve"]
            + firm_debt_settlement["overdraft_refinance_used"]
            + np.maximum(
                0.0,
                np.nan_to_num(self.firms.ts.current("received_debt_rollover_credit"), nan=0.0),
            )
            - firm_debt_settlement["payable_interest"]
            - firm_debt_settlement["payable_principal"]
        )
        self.firms.settle_operating_revolving_facility(
            cash_available_after_debt_service=cash_available_after_debt_service,
        )
        from macromodel.country.diagnostic_dividend_fund import compute_cash_feasible_firm_distribution_settlement

        firm_settlement_debit, firm_settlement_shortfall = compute_cash_feasible_firm_distribution_settlement(
            declared_distribution=self.firms.ts.current("dividend_fund_declared_distribution"),
            cash_available_after_debt_service=cash_available_after_debt_service,
            operating_revolving_repayment=self.firms.ts.current("operating_revolving_repayment"),
        )
        self.firms.ts.dividend_fund_settlement_debit.append(firm_settlement_debit)
        self.firms.ts.dividend_fund_settlement_shortfall.append(firm_settlement_shortfall)
        self.firms.ts.debt_installments.append(
            self.credit_market.settle_firm_installments(
                payable_principal_by_firm=firm_debt_settlement["payable_principal"],
                payable_interest_by_firm=firm_debt_settlement["payable_interest"],
                payable_opening_interest_arrears_by_firm=firm_debt_settlement["payable_opening_interest_arrears"],
                payable_contractual_interest_by_firm=firm_debt_settlement["payable_contractual_interest"],
                payable_opening_principal_arrears_by_firm=firm_debt_settlement["payable_opening_principal_arrears"],
                payable_contractual_principal_by_firm=firm_debt_settlement["payable_contractual_principal"],
            )
        )
        self.firms.ts.total_debt_installments.append([self.firms.ts.current("debt_installments").sum()])
        self.firms.ts.interest_paid_on_deposits.append(firm_interest_paid_on_deposits)
        self.firms.ts.interest_paid_on_loans.append(self.credit_market.compute_interest_paid_by_firm())
        self.firms.ts.interest_paid.append(self.firms.compute_interest_paid())
        operating_revolving_interest_by_bank = np.bincount(
            self.firms.states["Corresponding Bank ID"],
            weights=self.firms.ts.current("operating_revolving_interest_paid"),
            minlength=self.banks.ts.current("n_banks"),
        )
        self.banks.ts.interest_received_on_revolving_operating_facility.append(operating_revolving_interest_by_bank)
        self.banks.ts.total_interest_received_on_revolving_operating_facility.append(
            [operating_revolving_interest_by_bank.sum()]
        )
        self.banks.ts.interest_received_on_loans.append(
            self.credit_market.compute_interest_received_by_bank() + operating_revolving_interest_by_bank
        )
        self.banks.ts.consumer_opening_interest_arrears_collected.append(
            self.credit_market.compute_consumer_opening_interest_arrears_collected_by_bank()
        )
        self.banks.ts.consumer_interest_accrued.append(self.credit_market.compute_consumer_interest_accrued_by_bank())
        self.banks.ts.recognized_interest_received_on_loans.append(
            self.credit_market.compute_recognized_interest_received_by_bank()
        )
        self.credit_market.remove_repaid_loans(("st_loans", "lt_loans"))
        self.credit_market.compute_aggregates()
        self.firms.ts.short_term_loan_debt.append(self.credit_market.compute_outstanding_short_term_loans_by_firm())
        self.firms.ts.long_term_loan_debt.append(self.credit_market.compute_outstanding_long_term_loans_by_firm())
        self.firms.ts.debt.append(self.firms.compute_debt())
        self.firms.ts.scheduled_debt_service.append(self.credit_market.compute_scheduled_debt_service_by_firm())
        self.firms.ts.total_inventory_change.append(self.firms.compute_total_inventory_change())
        self.firms.ts.total_sales.append(self.firms.compute_total_sales())
        self.firms.ts.taxes_paid_on_production.append(current_taxes_paid_on_production)
        self.firms.ts.profits.append(self.firms.compute_profits())
        self.firms.ts.unit_costs.append(self.firms.compute_unit_costs())
        self.firms.ts.corporate_taxes_paid.append(
            self.firms.compute_corporate_taxes_paid(
                tau_firm=self.central_government.states["Profit Tax"],
            )
        )

        # D2. FIRM BALANCE SHEETS
        self.firms.ts.gross_operating_surplus_mixed_income.append(
            self.firms.compute_gross_operating_surplus_mixed_income()
        )
        self.firms.ts.deposits.append(self.firms.compute_deposits())
        self.firms.ts.equity.append(
            self.firms.compute_equity(
                current_good_prices=self.economy.ts.current("good_prices"),
            )
        )
        self.firms.check_firm_accounting_controls(
            current_good_prices=self.economy.ts.current("good_prices"),
            enforce=True,
        )

        # D3. FIRM INSOLVENCY
        firm_debt_settlement["default_flag"] = np.logical_and(
            self.firms.ts.current("equity") < 0.0,
            firm_debt_settlement["illiquid_flag"],
        )
        firm_insolvency_result = self.firms.handle_insolvency(
            credit_market=self.credit_market,
            illiquid_flag=firm_debt_settlement["illiquid_flag"],
        )
        self.economy.ts.npl_firm_loans.append([firm_insolvency_result.npl_ratio])
        self.banks.ts.firm_default_loan_writeoff.append(firm_insolvency_result.loan_writeoff_by_bank)
        self.banks.ts.total_firm_default_loan_writeoff.append([firm_insolvency_result.loan_writeoff_by_bank.sum()])
        self.banks.ts.firm_default_overdraft_writeoff.append(firm_insolvency_result.overdraft_writeoff_by_bank)
        self.banks.ts.total_firm_default_overdraft_writeoff.append(
            [firm_insolvency_result.overdraft_writeoff_by_bank.sum()]
        )
        self.banks.ts.firm_default_revolving_operating_facility_writeoff.append(
            firm_insolvency_result.revolving_operating_facility_writeoff_by_bank
        )
        self.banks.ts.total_firm_default_revolving_operating_facility_writeoff.append(
            [firm_insolvency_result.revolving_operating_facility_writeoff_by_bank.sum()]
        )
        self.banks.ts.firm_default_credit_loss.append(firm_insolvency_result.credit_loss_by_bank)
        self.banks.ts.total_firm_default_credit_loss.append([firm_insolvency_result.credit_loss_by_bank.sum()])
        self.firms.ts.override_current(
            "short_term_loan_debt", self.credit_market.compute_outstanding_short_term_loans_by_firm()
        )
        self.firms.ts.override_current(
            "long_term_loan_debt", self.credit_market.compute_outstanding_long_term_loans_by_firm()
        )
        self.firms.ts.override_current("debt", self.firms.compute_debt())
        self.firms.ts.override_current(
            "scheduled_debt_service", self.credit_market.compute_scheduled_debt_service_by_firm()
        )
        self.firms.finalize_firm_debt_settlement(
            settlement=firm_debt_settlement,
            residual_overdraft_exposure=np.maximum(0.0, -self.firms.ts.current("deposits")),
        )
        self.firms.ts.override_current("total_credit_exposure", self.firms.compute_total_credit_exposure())

        firm_insolvency_rate, num_insolvent_firms_by_sector = self.firms.compute_insolvency_rate()
        self.economy.ts.firm_insolvency_rate.append([firm_insolvency_rate])
        self.economy.ts.num_insolvent_firms_by_sector.append(num_insolvent_firms_by_sector)

        self.firms.ts.total_debt.append([self.firms.compute_total_debt()])
        self.firms.ts.total_deposits.append([self.firms.compute_total_deposits()])

        self.banks.ts.dividend_fund_settlement_debit.append(
            np.maximum(self.banks.ts.current("dividend_fund_declared_distribution"), 0.0).copy()
        )
        self.record_dividend_fund_settlement_receipts()

        # E1. INDIVIDUAL AND HOUSEHOLD INCOME
        # Benefits were needed during planning for reservation-wage decisions,
        # but realised payments must follow the activity status after labour
        # market clearing. Replace the planning vector before computing income.
        self.individuals.ts.override_current(
            "income_from_unemployment_benefits",
            self.central_government.distribute_unemployment_benefits_to_individuals(
                current_individual_activity_status=self.individuals.states["Activity Status"],
            ),
        )
        # Update individual income components
        self.individuals.ts.income.append(
            self.individuals.compute_income(
                firm_profits=self.firms.ts.current("profits"),
                bank_profits=self.banks.ts.current("cash_distributable_profits"),
                cpi=self.economy.current_consumer_price_level(),
                income_taxes=self.central_government.states["Income Tax"],
                tau_firm=self.central_government.states["Profit Tax"],
            )
        )
        self.individuals.ts.income_histogram.append(get_histogram(self.individuals.ts.current("income"), self.scale))

        # E2. HOUSEHOLD INCOME COMPONENTS
        # Recalculate rental income with final housing market data and overwrite the planned value
        final_income_rental = self.households.compute_rental_income(
            housing_data=self.housing_market.states["properties"],
            income_taxes=self.central_government.states["Income Tax"],
        )
        self.households.ts.dicts["income_rental"][-1] = final_income_rental
        self.households.ts.dicts["total_income_rental"][-1] = [final_income_rental.sum()]

        self.households.ts.income_employee.append(
            self.households.compute_employee_income(
                individual_income=self.individuals.ts.current("employee_income"),
                corr_households=self.individuals.states["Corresponding Household ID"],
            )
        )
        self.households.ts.total_income_employee.append([self.households.ts.current("income_employee").sum()])
        self.settle_household_social_transfers()
        paper_returns_are_capital_gains = getattr(
            self.households.functions["wealth"],
            "illiquid_returns_are_capital_gains",
            False,
        )
        gross_dividend_income = self.households.ts.current(
            "dividend_fund_settled_firm_distribution"
        ) + self.households.ts.current("dividend_fund_settled_bank_distribution")
        dividend_income, dividend_income_tax_withheld = compute_dividend_income_after_withholding(
            gross_distribution=gross_dividend_income,
            income_tax_rate=self.central_government.states["Income Tax"],
        )
        self.households.ts.dividend_fund_income_tax_withheld.append(dividend_income_tax_withheld)
        self.households.ts.total_dividend_fund_income_tax_withheld.append([dividend_income_tax_withheld.sum()])
        self.households.ts.income_dividend_distributions.append(dividend_income)
        self.households.ts.total_income_dividend_distributions.append([dividend_income.sum()])
        if paper_returns_are_capital_gains:
            self.households.stage_illiquid_valuation_return(period_index=period_index)
            financial_asset_income = dividend_income
        else:
            financial_asset_income = self.households.compute_income_from_financial_assets(period_index=period_index)
        self.households.ts.income_financial_assets.append(financial_asset_income)
        self.households.ts.total_income_financial_assets.append(
            [self.households.ts.current("income_financial_assets").sum()]
        )
        self.households.ts.income.append(self._apply_household_income_shock(self.households.compute_income()))
        if getattr(self.households.functions["consumption"], "uses_income_belief_learning", False):
            cpi_series = self.economy.consumer_price_level_series_name()
            # Use non-property income (employment + social transfers + rental), not the
            # full income total used elsewhere, as the Kalman-update signal: total income
            # includes stochastic income_financial_assets, which can be large and negative
            # after a bad market draw and is unrelated to the permanent-income concept this
            # learning module targets -- see compute_non_property_income's docstring and
            # GH issue #90. Both the current- and lagged-period signal must use the same
            # (non-property) basis, or the resulting growth ratio mixes income definitions.
            lagged_real_non_property_income = None
            if (
                len(self.households.ts.historic("income_employee")) >= 5
                and len(self.economy.ts.historic(cpi_series)) >= 5
            ):
                lagged_non_property_income = (
                    self.households.ts.dicts["income_employee"][-5]
                    + self.households.ts.dicts["income_social_transfers"][-5]
                    + self.households.ts.dicts["income_rental"][-5]
                )
                lagged_real_non_property_income = lagged_non_property_income / self.economy.ts.dicts[cpi_series][-5][0]
            self.households.update_income_belief_learning_state(
                current_income=self.households.compute_non_property_income() / self.economy.ts.current(cpi_series)[0],
                lagged_income=lagged_real_non_property_income,
            )
        self.households.ts.income_histogram.append(get_histogram(self.households.ts.current("income"), self.scale))
        self._clear_household_income_shock()

        # E3. HOUSEHOLD METRICS
        rent_div_income = np.divide(
            self.households.ts.current("rent"),
            self.households.ts.current("income"),
            out=np.zeros(self.households.ts.current("rent").shape),
            where=self.households.ts.current("income") != 0.0,
        )
        self.households.ts.rent_div_income_histogram.append(get_histogram(rent_div_income, None))

        # F1. HOUSEHOLD WEALTH AND CONSUMPTION
        # Update household financial positions
        if self.add_emissions:
            readjusted_factors = (
                self.emission_factors_lcu / self.economy.ts.current("good_prices")[self.emitting_indices]
            )
            readjusted_factors_ch4 = (
                self.emission_factors_lcu_ch4 / self.economy.ts.current("good_prices")[self.emitting_indices_ch4]
                if self.emission_factors_lcu_ch4 is not None
                else None
            )
        else:
            readjusted_factors = None
            readjusted_factors_ch4 = None

        self.households.update_consumption_and_investment(
            tau_vat=self.central_government.states["Value-added Tax"],
            tau_cf=self.central_government.states["Household Capital Formation Tax"],
            tau_vat_on_investment=self.central_government.states["Household Investment VAT Rate"] or 0.0,
            readjusted_factors=readjusted_factors,
            emitting_indices=self.emitting_indices,
            add_emissions=self.add_emissions,
            use_emission_multiplier=self.use_emission_multiplier,
            readjusted_factors_ch4=readjusted_factors_ch4,
            emitting_indices_ch4=self.emitting_indices_ch4,
        )
        illiquid_financial_asset_return_rate = self.households.update_wealth(
            housing_data=self.housing_market.states["properties"],
            tau_cf=self.central_government.states["Household Capital Formation Tax"],
            period_index=period_index,
            tau_vat_on_investment=self.central_government.states["Household Investment VAT Rate"] or 0.0,
        )
        self.economy.ts.illiquid_financial_asset_return_rate.append([illiquid_financial_asset_return_rate])
        self.households.ts.wealth_histogram.append(get_histogram(self.households.ts.current("wealth"), self.scale))
        self.households.ts.net_wealth.append(self.households.compute_net_wealth())

        # F2. HOUSEHOLD INSOLVENCY
        consumer_terminal_removal_exclusion, pending_ficp_forgiveness_events = self._process_ficp_forgiveness_events(
            period_index=period_index
        )
        if pending_ficp_forgiveness_events:
            current_consumption_debt = self.credit_market.compute_outstanding_consumption_loans_by_household()
            current_mortgage_debt = self.credit_market.compute_outstanding_mortgages_by_household()
            current_debt = current_consumption_debt + current_mortgage_debt
            self.households.ts.override_current("consumption_loan_debt", current_consumption_debt)
            self.households.ts.override_current("mortgage_debt", current_mortgage_debt)
            self.households.ts.override_current("debt", current_debt)
            self.households.ts.override_current("total_consumption_loan_debt", [current_consumption_debt.sum()])
            self.households.ts.override_current("total_mortgage_debt", [current_mortgage_debt.sum()])
            self.households.ts.override_current("net_wealth", self.households.compute_net_wealth())
        household_insolvency_rate, npl_hh_cons_loans, npl_mortgages = self.households.handle_insolvency(
            banks=self.banks,
            credit_market=self.credit_market,
            consumer_terminal_removal_exclusion=consumer_terminal_removal_exclusion,
        )
        # Reconcile household liabilities after generic insolvency completes any
        # overlapping loan removal. The CreditMarket remains the sole debt
        # authority; this overwrites the earlier payment snapshot only after a
        # realised write-off changes that market state.
        if self.configuration.households.parameters.uses_feasibility_resolver:
            self.refresh_stage5_household_debt_views()
        self.economy.ts.household_insolvency_rate.append([household_insolvency_rate])
        self.economy.ts.npl_hh_cons_loans.append([npl_hh_cons_loans])
        self.economy.ts.npl_mortgages.append([npl_mortgages])

        # G1. GOVERNMENT AND BANKING
        # Update government consumption
        self.government_entities.record_consumption(
            add_emissions=self.add_emissions,
            readjusted_factors=readjusted_factors,
            emitting_indices=self.emitting_indices,
        )

        # G2. BANKING METRICS
        self.banks.ts.interest_received_on_deposits.append(
            self.banks.compute_interest_received_on_deposits(
                central_bank_policy_rate=self.central_bank.ts.current("policy_rate"),
            )
        )
        self.banks.ts.interest_received.append(
            self.banks.ts.current("interest_received_on_loans") + self.banks.ts.current("interest_received_on_deposits")
        )
        self.banks.ts.profits.append(self.banks.compute_profits())
        self.banks.ts.cash_distributable_profits.append(self.banks.compute_cash_distributable_profits())
        self.banks.ts.profits_histogram.append(get_histogram(self.banks.ts.current("profits"), self.scale))

        # G3. BANK BALANCE SHEETS
        self.banks.update_deposits(
            current_firm_deposits=self.firms.ts.current("deposits"),
            current_household_deposits=self.households.ts.current("liquid_financial_assets"),
            firm_corresponding_bank=self.firms.states["Corresponding Bank ID"],
            households_corresponding_bank=self.households.states["Corresponding Bank ID"],
        )
        self.banks.update_loans(
            credit_market=self.credit_market,
            revolving_operating_facility_exposure=np.bincount(
                self.firms.states["Corresponding Bank ID"],
                weights=self.firms.ts.current("operating_revolving_closing_balance"),
                minlength=self.banks.ts.current("n_banks"),
            ),
        )
        self.validate_stage5_closing_balance_sheets()

        self.banks.ts.market_share.append(self.banks.compute_market_share())
        self.banks.ts.market_share_histogram.append(get_histogram(self.banks.ts.current("market_share"), None))

        self.banks.ts.equity.append(
            self.banks.compute_equity(
                profit_taxes=self.central_government.states["Profit Tax"],
                taxable_profits=self.banks.ts.current("cash_distributable_profits"),
            )
        )
        self.banks.ts.equity_histogram.append(get_histogram(self.banks.ts.current("equity"), self.scale))

        for household_id, episode_id in pending_ficp_forgiveness_events:
            self.households.mark_ficp_forgiveness_processed(household_id, episode_id)

        self.banks.ts.liability.append(self.banks.compute_liability())
        self.banks.ts.liability_histogram.append(get_histogram(self.banks.ts.current("liability"), self.scale))

        self.banks.ts.deposits.append(self.banks.compute_deposits())
        self.banks.ts.deposits_histogram.append(get_histogram(self.banks.ts.current("deposits"), self.scale))

        # G4. BANK INSOLVENCY
        self.central_government.ts.bank_equity_injection.append(
            [self.banks.handle_insolvency(credit_market=self.credit_market)]
        )
        self.economy.ts.bank_insolvency_rate.append([self.banks.compute_insolvency_rate()])
        self.record_diagnostic_dividend_fund()

        # G5. GOVERNMENT REVENUE
        # General government fields
        taxable_household_financial_income = (
            gross_dividend_income
            if getattr(self.households.functions["wealth"], "illiquid_returns_are_capital_gains", False)
            else self.households.ts.current("income_financial_assets")
        )
        self.central_government.compute_taxes(
            current_ind_employee_income=self.individuals.ts.current("employee_income"),
            current_total_rent_paid=self.households.ts.current("rent")[
                self.households.states["Tenure Status of the Main Residence"] == 3
            ].sum(),
            current_income_financial_assets=taxable_household_financial_income,
            current_ind_activity=self.individuals.states["Activity Status"],
            current_ind_realised_cons=self.households.ts.current("consumption"),
            current_bank_profits=self.banks.ts.current("cash_distributable_profits"),
            current_firm_production=self.firms.ts.current("production"),
            current_firm_price=self.firms.ts.current("price"),
            current_firm_profits=self.firms.ts.current("profits"),
            current_firm_industries=self.firms.states["Industry"],
            taxes_less_subsidies_rates=self.central_government.states["Taxes Less Subsidies Rates"],
            current_household_new_real_wealth=self.households.ts.current("investment"),
            current_total_exports=self.economy.ts.current("exports_before_taxes").sum(),
            current_firm_capital_formation=self.firms.ts.current("total_capital_inputs_bought_costs").sum(),
        )

        # General government fields
        self.central_government.ts.taxes_on_products.append([self.central_government.compute_taxes_on_products()])
        self.central_government.ts.revenue.append(
            [
                self.central_government.compute_revenue(
                    household_rent_paid_to_government=self.households.states["Rent paid to Government"]
                )
            ]
        )
        self.central_government.ts.total_unemployment_benefits.append(
            [
                self.economy.current_consumer_price_level()
                * self.individuals.ts.current("income_from_unemployment_benefits").sum()
            ]
        )
        self.central_government.ts.total_public_pension_benefits.append(
            [self.households.ts.current("income_public_pension").sum()]
        )
        self.central_government.ts.total_other_social_transfers.append(
            [self.households.ts.current("income_other_social_transfers").sum()]
        )
        self.central_government.ts.total_necessity_support.append(
            [self.households.ts.current("stage5_subsistence_support").sum()]
        )
        self.central_government.ts.total_household_social_transfers.append(
            [self.households.ts.current("income_social_transfers").sum()]
        )
        self.central_government.ts.debt_interest_rate.append(
            [
                self.central_government.functions["debt_interest"].compute_interest_rate(
                    current_policy_rate=self.central_bank.ts.current("policy_rate")[0],
                    previous_debt_interest_rate=self.central_government.ts.current("debt_interest_rate")[0],
                    time_unit=self.economy.time_unit,
                )
            ]
        )
        self.central_government.ts.interest_payments_on_debt.append(
            [
                self.central_government.ts.current("debt_interest_rate")[0]
                * self.central_government.ts.current("debt")[0]
            ]
        )
        self.central_government.ts.deficit.append(
            self.central_government.compute_deficit(
                current_ind_activity=self.individuals.states["Activity Status"],
                current_cpi=self.economy.current_consumer_price_level(),
                current_government_nominal_amount_spent=self.government_entities.ts.current(
                    "nominal_amount_spent_in_lcu"
                ),
                interest_payments_on_debt=self.central_government.ts.current("interest_payments_on_debt")[0],
            )
        )
        self.central_government.ts.debt.append(self.central_government.compute_debt())

        # Compute GDP
        self.economy.compute_gdp(
            total_output=(self.firms.ts.current("price") * self.firms.ts.current("production")).sum(),
            sectoral_sales=sectoral_sales,
            sectoral_intermediate_consumption=np.bincount(
                self.firms.states["Industry"],
                weights=self.firms.ts.current("used_intermediate_inputs_costs"),
                minlength=self.economy.n_industries,
            ),
            taxes_on_products=self.central_government.ts.current("taxes_on_products")[0],
            taxes_on_production=self.central_government.ts.current("taxes_production")[0],
            rent_paid=self.economy.ts.current("total_real_rent_paid")[0],
            rent_imputed=self.economy.ts.current("total_imp_rent_paid")[0],
            hh_consumption=self.households.ts.current("total_consumption")[0],
            gov_consumption=self.government_entities.ts.current("total_consumption")[0],
            change_in_inventories=self.firms.ts.current("total_inventory_change").sum()
            + self.firms.ts.current("total_intermediate_inputs_bought_costs").sum()
            - self.firms.ts.current("used_intermediate_inputs_costs").sum(),
            gross_fixed_capital_formation=(1 + self.central_government.states["Firm Capital Formation Tax"])
            * self.firms.ts.current("total_capital_inputs_bought_costs").sum()
            + (
                1
                + self.central_government.states["Household Capital Formation Tax"]
                + (self.central_government.states["Household Investment VAT Rate"] or 0.0)
            )
            * self.households.ts.current("investment").sum(),
            exports=self.economy.ts.current("exports").sum(),
            imports=self.economy.ts.current("imports").sum(),
            operating_surplus=self.firms.ts.current("gross_operating_surplus_mixed_income").sum(),
            wages=self.firms.ts.current("total_wage").sum(),
            rent_received=self.economy.ts.current("total_real_rent_rec")[0]
            + self.central_government.ts.current("taxes_rental_income")[0],
            central_government_rent_received=self.central_government.ts.current("total_rent_received")[0],
            running_multiple_countries=self.running_multiple_countries,
        )
        self.economy.compute_output_gap()

    def update_population_structure(self) -> None:
        """Update demographic composition.

        Applies demographic changes (aging, retirement, etc.) to the
        population of individual agents.
        """
        self.individuals.update_demography()

    def get_goods_market_participants(self) -> list[Agent | RestOfTheWorld]:
        """Get all participants in the goods market.

        Returns:
            list[Agent | RestOfTheWorld]: List of agents participating in
                goods market transactions
        """
        return [self.firms, self.households, self.government_entities]

    def save_to_h5(self, h5_file: h5py.File) -> None:
        """Save complete country state to HDF5.

        Saves the full state of all agents, markets, and economic variables
        to an HDF5 file for later analysis or continuation.

        Args:
            h5_file (h5py.File): Open HDF5 file to save to
        """
        group = h5_file.create_group(self.country_name)
        self.firms.save_to_h5(group)
        # self.firms.save_industry_firms_df(group)

        self.individuals.save_to_h5(group)
        self.households.save_to_h5(group)
        self.households.save_consumption_weights(group)
        self.households.save_ratio_diagnostics(group)
        self.government_entities.save_to_h5(group)
        self.central_government.save_to_h5(group)
        self.banks.save_to_h5(group)
        self.central_bank.save_to_h5(group)
        self.economy.save_to_h5(group)

        self.labour_market.save_to_h5(group)
        self.credit_market.save_to_h5(group)
        self.housing_market.save_to_h5(group)

        self.exogenous.save_to_h5(group)

    def shallow_output(self) -> pd.DataFrame:
        """Create summary DataFrame of key economic indicators.

        Returns:
            pd.DataFrame: DataFrame containing main economic metrics
                (GDP, inflation, unemployment, etc.)
        """
        data_dict = {
            "Sales": self.firms.total_sales(),
            "Production": self.firms.total_production(),
            "Taxes Paid on Production": self.firms.total_taxes_paid_on_production(),
            "Used Input Costs": self.firms.total_used_input_costs(),
            "Bought Input Costs": self.firms.total_bought_input_costs(),
            "Operating Surplus": self.firms.total_operating_surplus(),
            "Wages": self.firms.total_wages(),
            "Inventory Changes": self.firms.total_inventory_change(),
            "Profits": self.firms.total_profits(),
            "Capital Bought": self.firms.total_capital_bought(),
            "Household Consumption": self.households.total_consumption(),
            "Government Consumption": self.government_entities.total_consumption(),
            "Taxes on Products": self.central_government.total_taxes(),
            "Imports": self.economy.total_imports(),
            "Exports": self.economy.total_exports(),
            "PPI": self.economy.total_ppi_inflation(),
            "CPI Transaction": self.economy.total_cpi_inflation(),
            "CPI Transaction YoY Change": self.economy.ts.get_aggregate("cpi_transaction_yoy_change"),
            "CFPI": self.economy.total_cfpi_inflation(),
            "Gross Output": self.firms.total_sales() + self.firms.total_taxes_paid_on_production(),
            "Output Gap": self.economy.ts.get_aggregate("output_gap"),
            "Unemployment Rate": self.economy.unemployment_rate(),
            "Consumption Expansion Loan Debt": self.households.consumption_loan_debt(),
            "Mortgage Debt": self.households.mortgage_debt(),
            "Central Bank Policy Rate": self.central_bank.ts.get_aggregate("policy_rate"),
        }

        if self.add_emissions:
            data_dict["Firm Input Emissions"] = self.firms.get_total_inputs_emissions()
            data_dict["Firm Capital Emissions"] = self.firms.get_total_capital_emissions()
            data_dict["Household Consumption Emissions"] = self.households.consumption_emissions()
            data_dict["Household Investment Emissions"] = self.households.investment_emissions()
            data_dict["Government Emissions"] = self.government_entities.emissions()
            for input_name in ["Coal", "Gas", "Oil", "Refined Products"]:
                input_rename = input_name.replace(" ", "_").lower()
                data_dict[f"Firm Input Emissions {input_name}"] = self.firms.get_disaggregated_input_emissions(
                    input_rename
                )
                data_dict[f"Firm Capital Emissions {input_name}"] = self.firms.get_disaggregated_capital_emissions(
                    input_rename
                )
                data_dict[f"Household Consumption Emissions {input_name}"] = (
                    self.households.disaggregated_consumption_emissions(input_rename)
                )
                data_dict[f"Household Investment Emissions {input_name}"] = (
                    self.households.disaggregated_investment_emissions(input_rename)
                )
                data_dict[f"Government Emissions {input_name}"] = self.government_entities.disaggregated_emissions(
                    input_rename
                )

        return pd.DataFrame(data_dict)

    def gdp_debug_output(self) -> pd.DataFrame:
        """Create detailed DataFrame of GDP components for debugging.

        This method creates a comprehensive DataFrame containing all three GDP measures
        (output, expenditure, and income approaches) along with their constituent components.
        This is useful for debugging when the three measures don't match.

        Returns:
            pd.DataFrame: DataFrame containing GDP measures and their components
        """
        data_dict = {
            # GDP Output Approach Components
            "GDP_Output": self.economy.ts.get_aggregate("gdp_output"),
            "+Total_Output": self.economy.ts.get_aggregate("total_output"),
            "-Total_Intermediate_Consumption": self.economy.ts.get_aggregate("total_intermediate_consumption"),
            "+Taxes_on_Products": self.central_government.ts.get_aggregate("taxes_on_products"),
            "-Taxes_on_Production": self.central_government.ts.get_aggregate("taxes_production"),
            "+Total_Real_Rent_Paid": self.economy.ts.get_aggregate("total_real_rent_paid"),
            "+Total_Imputed_Rent_Paid": self.economy.ts.get_aggregate("total_imp_rent_paid"),
            # GDP Expenditure Approach Components
            "GDP_Expenditure": self.economy.ts.get_aggregate("gdp_expenditure"),
            "+Household_Consumption": self.households.ts.get_aggregate("total_consumption"),
            "+Government_Consumption": self.government_entities.ts.get_aggregate("total_consumption"),
            "+Changes_in_Inventories": self.firms.ts.get_aggregate("total_inventory_change"),
            "+Gross_Fixed_Capital_Formation": self.economy.ts.get_aggregate("total_gross_fixed_capital_formation"),
            "+Exports": self.economy.ts.get_aggregate("total_exports"),
            "-Imports": self.economy.ts.get_aggregate("total_imports"),
            "+Rent_Paid": self.economy.ts.get_aggregate("total_real_rent_paid"),
            "+Rent_Imputed": self.economy.ts.get_aggregate("total_imp_rent_paid"),
            # GDP Income Approach Components
            "GDP_Income": self.economy.ts.get_aggregate("gdp_income"),
            "+Operating_Surplus": self.firms.ts.get_aggregate("gross_operating_surplus_mixed_income"),
            "+Wages": self.firms.ts.get_aggregate("total_wage"),
            "+Rent_Received": self.economy.ts.get_aggregate("total_real_rent_rec"),
            "+Central_Government_Rent_Received": self.central_government.ts.get_aggregate("total_rent_received"),
            "+Central_Government_Rental_Taxes": self.central_government.ts.get_aggregate("taxes_rental_income"),
            "+Central_Government_Product_Taxes": self.central_government.ts.get_aggregate("taxes_on_products"),
            # Additional Value Added Components
            "Total_Gross_Value_Added": self.economy.ts.get_aggregate("total_gross_value_added"),
            "Total_Gross_Value_Added_A": self.economy.ts.get_aggregate("total_gross_value_added_a"),
            "Total_Gross_Value_Added_BCDE": self.economy.ts.get_aggregate("total_gross_value_added_bcde"),
            "Total_Gross_Value_Added_C": self.economy.ts.get_aggregate("total_gross_value_added_c"),
            "Total_Gross_Value_Added_F": self.economy.ts.get_aggregate("total_gross_value_added_f"),
            "Total_Gross_Value_Added_GHIJKLMNOPQRSTU": self.economy.ts.get_aggregate(
                "total_gross_value_added_ghijklmnopqrstu"
            ),
            "Total_Gross_Value_Added_GHI": self.economy.ts.get_aggregate("total_gross_value_added_ghi"),
            "Total_Gross_Value_Added_J": self.economy.ts.get_aggregate("total_gross_value_added_j"),
            "Total_Gross_Value_Added_K": self.economy.ts.get_aggregate("total_gross_value_added_k"),
            "Total_Gross_Value_Added_L": self.economy.ts.get_aggregate("total_gross_value_added_l"),
            "Total_Gross_Value_Added_MN": self.economy.ts.get_aggregate("total_gross_value_added_mn"),
            "Total_Gross_Value_Added_OPQ": self.economy.ts.get_aggregate("total_gross_value_added_opq"),
            "Total_Gross_Value_Added_RSTU": self.economy.ts.get_aggregate("total_gross_value_added_rstu"),
        }

        return pd.DataFrame(data_dict)

    @property
    def gdp_components_df(self) -> pd.DataFrame:
        """Create detailed DataFrame of GDP components for debugging.

        This property creates a comprehensive DataFrame containing all three GDP measures
        (output, expenditure, and income approaches) along with their constituent components.
        This is useful for debugging when the three measures don't match.

        Returns:
            pd.DataFrame: DataFrame containing GDP measures and their components
        """
        data_dict = {
            # GDP Output Approach Components
            "GDP_Output": self.economy.ts.get_aggregate("gdp_output"),
            "+Total_Output": self.economy.ts.get_aggregate("total_output"),
            "-Total_Intermediate_Consumption": self.economy.ts.get_aggregate("total_intermediate_consumption"),
            "+Taxes_on_Products": self.central_government.ts.get_aggregate("taxes_on_products"),
            "-Taxes_on_Production": self.central_government.ts.get_aggregate("taxes_production"),
            "+Total_Real_Rent_Paid": self.economy.ts.get_aggregate("total_real_rent_paid"),
            "+Total_Imputed_Rent_Paid": self.economy.ts.get_aggregate("total_imp_rent_paid"),
            # GDP Expenditure Approach Components
            "GDP_Expenditure": self.economy.ts.get_aggregate("gdp_expenditure"),
            "+Household_Consumption": self.households.ts.get_aggregate("total_consumption"),
            "+Government_Consumption": self.government_entities.ts.get_aggregate("total_consumption"),
            "+Changes_in_Inventories": self.firms.ts.get_aggregate("total_inventory_change"),
            "+Gross_Fixed_Capital_Formation": self.economy.ts.get_aggregate("total_gross_fixed_capital_formation"),
            "+Exports": self.economy.ts.get_aggregate("total_exports"),
            "-Imports": self.economy.ts.get_aggregate("total_imports"),
            "+Rent_Paid": self.economy.ts.get_aggregate("total_real_rent_paid"),
            "+Rent_Imputed": self.economy.ts.get_aggregate("total_imp_rent_paid"),
            # GDP Income Approach Components
            "GDP_Income": self.economy.ts.get_aggregate("gdp_income"),
            "+Operating_Surplus": self.firms.ts.get_aggregate("gross_operating_surplus_mixed_income"),
            "+Wages": self.firms.ts.get_aggregate("total_wage"),
            "+Rent_Received": self.economy.ts.get_aggregate("total_real_rent_rec"),
            "+Central_Government_Rent_Received": self.central_government.ts.get_aggregate("total_rent_received"),
            "+Central_Government_Rental_Taxes": self.central_government.ts.get_aggregate("taxes_rental_income"),
            "+Central_Government_Product_Taxes": self.central_government.ts.get_aggregate("taxes_on_products"),
            # Additional Value Added Components
            "Total_Gross_Value_Added": self.economy.ts.get_aggregate("total_gross_value_added"),
            "Total_Gross_Value_Added_A": self.economy.ts.get_aggregate("total_gross_value_added_a"),
            "Total_Gross_Value_Added_BCDE": self.economy.ts.get_aggregate("total_gross_value_added_bcde"),
            "Total_Gross_Value_Added_C": self.economy.ts.get_aggregate("total_gross_value_added_c"),
            "Total_Gross_Value_Added_F": self.economy.ts.get_aggregate("total_gross_value_added_f"),
            "Total_Gross_Value_Added_GHIJKLMNOPQRSTU": self.economy.ts.get_aggregate(
                "total_gross_value_added_ghijklmnopqrstu"
            ),
            "Total_Gross_Value_Added_GHI": self.economy.ts.get_aggregate("total_gross_value_added_ghi"),
            "Total_Gross_Value_Added_J": self.economy.ts.get_aggregate("total_gross_value_added_j"),
            "Total_Gross_Value_Added_K": self.economy.ts.get_aggregate("total_gross_value_added_k"),
            "Total_Gross_Value_Added_L": self.economy.ts.get_aggregate("total_gross_value_added_l"),
            "Total_Gross_Value_Added_MN": self.economy.ts.get_aggregate("total_gross_value_added_mn"),
            "Total_Gross_Value_Added_OPQ": self.economy.ts.get_aggregate("total_gross_value_added_opq"),
            "Total_Gross_Value_Added_RSTU": self.economy.ts.get_aggregate("total_gross_value_added_rstu"),
        }

        return pd.DataFrame(data_dict)

    @property
    def n_individuals(self) -> int:
        """int: Total number of individual agents in the economy."""
        return self.individuals.n_individuals

    def _permanent_income_simulation_sources(self) -> PermanentIncomeSimulationSources:
        """Build the Stage 3 common forecast simulation-source history.

        Side-effect free: only reads existing time series, does not mutate
        ``economy.ts``, ``households.ts``, or ``individuals.ts``.
        """
        current_period = self.start_period + (len(self.economy.ts.dicts["cpi_fixed_basket"]) - 1)

        cpi_fixed_basket = np.asarray(
            [np.asarray(value).reshape(-1)[0] for value in self.economy.ts.dicts["cpi_fixed_basket"]],
            dtype=float,
        )
        unemployment_rate = np.asarray(
            [np.asarray(value).reshape(-1)[0] for value in self.economy.ts.dicts["unemployment_rate"]],
            dtype=float,
        )
        policy_rate = np.asarray(
            [np.asarray(value).reshape(-1)[0] for value in self.central_bank.ts.dicts["policy_rate"]],
            dtype=float,
        )

        # Real household disposable income per capita (Stage 0 `model_spendable_income`
        # convention, i.e. households.ts["income"]), deflated by cpi_fixed_basket and
        # divided by the population history, then rebased to the notebook's
        # `real_pc_idx` convention (index = 100 at start_period).
        n_individuals_history = np.asarray(
            [np.asarray(value).reshape(-1)[0] for value in self.individuals.ts.dicts["n_individuals"]],
            dtype=float,
        )
        total_income_history = np.asarray(
            [float(np.sum(value)) for value in self.households.ts.dicts["income"]],
            dtype=float,
        )
        real_pc_income_levels = total_income_history / cpi_fixed_basket / n_individuals_history
        real_pc_income = rebase_real_pc_income_index(real_pc_income_levels, base_period_index=0)

        # policy_rate is appended during estimation_phase (start of period) while
        # income/cpi/unemployment are appended at end-of-period, so policy_rate is
        # always one entry ahead when this method is called mid-period. Trim to align.
        n_hist = len(real_pc_income)
        policy_rate = policy_rate[:n_hist]

        return PermanentIncomeSimulationSources(
            current_period=current_period,
            real_pc_income=real_pc_income,
            policy_rate=policy_rate,
            cpi_fixed_basket=cpi_fixed_basket,
            unemployment_rate=unemployment_rate,
        )

    def _common_permanent_income_terms(self) -> tuple[float, float]:
        """Return ``(common_log_ratio, common_forecast_variance)`` for this period.

        ``common_log_ratio`` is ``ln(y^p_t / y_t)`` directly -- the common
        forecast's point forecast, which is regressed against ``ln_y_p_over_y``
        and is therefore already the log permanent-to-current income ratio (no
        further transformation is needed). ``common_forecast_variance`` is the
        forecast's prediction-error variance. Both are scalar, broadcast to all
        households by the household-side helpers.

        Returns ``(0.0, 0.0)`` when the forecast is unavailable (non-FRA /
        missing inputs), preserving the pre-Increment-5 zero behaviour. This is
        the only fallback path; the household-side helpers add no second fallback.
        """
        forecast = self.forecast_common_permanent_income()
        if forecast is None:
            return 0.0, 0.0
        return float(forecast.point_forecast), float(forecast.forecast_variance)

    def forecast_common_permanent_income(self) -> Optional[PermanentIncomeForecast]:
        """Compute the Stage 3 common permanent-income forecast for the current period.

        Returns ``None`` if the forecast inputs or design matrix are unavailable
        for this country/configuration (e.g. non-FRA or missing ``data_paths``).
        Side-effect free: no mutation of model state.
        """
        if self._permanent_income_forecast_inputs is None or self._permanent_income_design_matrix is None:
            return None

        try:
            sources = self._permanent_income_simulation_sources()
            x_t = build_permanent_income_forecast_regressors(
                sources=sources,
                design_matrix=self._permanent_income_design_matrix,
                start_period=self.start_period,
                estimation_epoch=self._permanent_income_design_matrix.index.min(),
            )
        except ValueError as exc:
            # ValueError here is data-dependent (e.g. non-positive income/CPI,
            # unemployment outside [0, 1], non-finite history values) and is
            # expected to occur transiently during a simulation run, so it is
            # safe to skip this period's forecast. Errors from
            # forecast_common_permanent_income() below indicate a static
            # configuration/wiring mismatch (e.g. x_t vs. coefficient_table
            # index mismatch) and are deliberately NOT caught here.
            logging.warning(
                "Skipping common permanent-income forecast for %s due to invalid inputs: %s",
                self.country_name,
                exc,
            )
            return None

        return forecast_common_permanent_income(x_t, self._permanent_income_forecast_inputs)
