"""Household economic agent implementation.

This module implements household economic behavior through:
- Consumption and investment decisions
- Income generation and management
- Wealth accumulation and allocation
- Housing market participation
- Credit market interactions

The implementation handles:
- Household demographics
- Income sources (employment, transfers, rental, financial)
- Consumption patterns
- Investment decisions
- Property ownership
- Financial assets/liabilities
- Debt management
"""

import warnings
from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from macro_data import SyntheticCountry, SyntheticPopulation
from macro_data.readers.emission_fraction.emission_fraction_reader import EmissionFractions
from macromodel.agents.agent import Agent
from macromodel.agents.banks.banks import Banks
from macromodel.agents.households.func.borrow_vs_sell import (
    compute_borrow_vs_sell_choice,
)
from macromodel.agents.households.func.liquid_asset_drawdown import (
    compute_liquid_asset_drawdown,
)
from macromodel.agents.households.func.liquidity_shortfall import (
    compute_liquidity_shortfall,
)
from macromodel.agents.households.func.portfolio_diagnostics import (
    compute_stage4_household_diagnostics,
)
from macromodel.agents.households.func.portfolio_target_share import (
    compute_household_head_covariates,
)
from macromodel.agents.households.func.residual_capacity_fallback import (
    ResidualCapacityFallbackResult,
    compute_residual_capacity_fallback,
)
from macromodel.agents.households.household_properties import HouseholdType
from macromodel.agents.households.households_ts import create_households_timeseries
from macromodel.agents.households.income_belief_learning import (
    IncomeBeliefLearningOutputs,
    _scalar_rho,
    compute_income_belief_learning_outputs,
    compute_income_uncertainty,
    compute_permanent_income_log_ratio,
    compute_zeta,
)
from macromodel.agents.households.utils.create_bundle_matrix import create_bundle_matrix
from macromodel.configurations import HouseholdsConfiguration
from macromodel.markets.credit_market.credit_market import CreditMarket
from macromodel.markets.goods_market.value_type import ValueType
from macromodel.timeseries import TimeSeries
from macromodel.util.function_mapping import functions_from_model, update_functions
from macromodel.util.get_histogram import get_histogram
from macromodel.util.property_mapping import map_to_enum

VACANT_HOUSEHOLD_ID = -1

# Stage 4 (portfolio choice) diagnostic time series, registered at Households
# init (regardless of whether uses_portfolio_choice is active) so that
# Households.update_wealth() can always call .append(...) on them — see
# knowledge-vault/wiki/architecture/consumption-stage-4-portfolio-choice.md
# (Diagnostics section) for the full field list and meaning. Boolean flags
# default to False; numeric diagnostics default to 0.0, matching the
# Increment 2 helper's non-finite-input fallback contract for a household-
# period the call site never evaluates (uses_portfolio_choice=False).
_STAGE4_DIAGNOSTIC_INITIAL_VALUES: dict[str, float | bool] = {
    "portfolio_actual_illiquid_share": 0.0,
    "portfolio_opening_tfa_scale": 0.0,
    "portfolio_target_tfa_base": 0.0,
    "portfolio_post_return_lfa": 0.0,
    "portfolio_post_return_ifa": 0.0,
    "portfolio_liquid_return_rate": 0.0,
    "portfolio_illiquid_return_rate": 0.0,
    "portfolio_investable_surplus": 0.0,
    "portfolio_participation_probability": 0.0,
    "portfolio_participates": False,
    "portfolio_target_illiquid_share": 0.0,
    "portfolio_target_illiquid_assets": 0.0,
    "portfolio_delta_tilde": 0.0,
    "portfolio_kappa_star_tilde": 0.0,
    "portfolio_kappa_tilde": 0.0,
    "portfolio_desired_illiquid_adjustment": 0.0,
    "portfolio_adjustment_cost": 0.0,
    "portfolio_counterfactual_lfa_flow": 0.0,
    "portfolio_counterfactual_ifa_flow": 0.0,
    "portfolio_inaction_flag": False,
    "portfolio_upper_bound_flag": False,
    "portfolio_lower_bound_flag": False,
    "portfolio_infeasible_interval_flag": False,
    "portfolio_no_financial_assets_flag": False,
    "portfolio_target_share_clipped_flag": False,
    "portfolio_settlement_enabled": False,
}

# Stage 5 (feasibility resolver) diagnostic time series, registered at
# Households init unconditionally — see
# knowledge-vault/wiki/architecture/consumption-stage-5-feasibility-resolver.md
# (Increments 0-1). These are diagnostics-only: they have no effect on goods
# or credit demand at these increments.
_STAGE5_DIAGNOSTIC_INITIAL_VALUES: dict[str, float | bool] = {
    "liquidity_shortfall": 0.0,
    "household_saving": 0.0,
    "liquidity_shortfall_before_repair": 0.0,
    "funded_from_liquid_assets": 0.0,
    "residual_shortfall_after_lfa": 0.0,
    "preferred_margin_after_lfa": 0.0,
    "preferred_margin_amount": 0.0,
    "borrow_vs_sell_threshold": 0.0,
    "borrow_vs_sell_spread": 0.0,
    "borrow_vs_sell_l_tilde": 0.0,
    "borrow_vs_sell_comparison_valid_flag": False,
    "dsti_headroom": 0.0,
    "dsti_maximum_loan_size": 0.0,
    "dsti_cap_binding": False,
    "borrow_planned": 0.0,
    "liquidation_planned": 0.0,
    "shadow_credit_requested": 0.0,
    "forced_liquidation_amount": 0.0,
    "residual_shortfall_after_caps": 0.0,
    # Increment 5: the credit_requested value actually used for target_consumption_loans
    # this period (mirrors the legacy formula when the resolver is off).
    "live_credit_requested": 0.0,
}


@dataclass
class PreGrantFeasiblePlan:
    """Minimal live Stage 5 pre-grant feasibility carrier.

    ``credit_requested`` (Increment 5) defaults to ``None`` so that the
    Increment 4 populate method, which constructs this dataclass without
    knowledge of the field, keeps working unchanged.
    """

    liquidity_shortfall_before_repair: np.ndarray
    funded_from_liquid_assets: np.ndarray
    residual_shortfall_after_lfa: np.ndarray
    credit_requested: np.ndarray | None = None
    planned_liquidation_total: np.ndarray | None = None


@dataclass
class PostGrantFeasiblePlan:
    """Settled Stage 5 feasibility carrier after consumer credit clearing."""

    credit_granted: np.ndarray
    credit_rationing_gap: np.ndarray
    planned_liquidation_total: np.ndarray
    residual_shortfall_after_granted_credit: np.ndarray


class Households(Agent):
    """Economic agent representing household sector behavior.

    This class implements household economic decisions through:
    - Income generation (employment, transfers, rental, financial)
    - Consumption allocation across industries
    - Investment in real and financial assets
    - Housing market participation (buying, renting)
    - Credit market interactions (mortgages, loans)
    - Wealth management and allocation

    The implementation considers:
    - Household demographics and composition
    - Income sources and distribution
    - Consumption patterns and preferences
    - Investment strategies
    - Property ownership and rental decisions
    - Financial asset holdings
    - Debt levels and servicing

    Attributes:
        functions (dict): Mapping of function names to implementations
        independents (list): Independent variables for calculations
        consumption_weights (np.ndarray): Industry-specific consumption shares
        consumption_weights_by_income (np.ndarray): Income-based consumption patterns
        investment_weights (np.ndarray): Industry-specific investment shares
        use_consumption_weights_by_income (bool): Whether to use income-based weights
    """

    def __init__(
        self,
        country_name: str,
        all_country_names: list[str],
        n_industries: int,
        functions: dict[str, Any],
        ts: TimeSeries,
        states: dict[str, float | np.ndarray | list[np.ndarray]],
        consumption_weights: np.ndarray,
        consumption_weights_by_income: np.ndarray,
        investment_weights: np.ndarray,
        use_consumption_weights_by_income: bool,
        uses_feasibility_resolver: bool,
        independents: list[str],
        substitution_bundles: Optional[list] = None,
        emission_fractions: Optional[EmissionFractions] = None,
    ):
        """Initialize household economic agent.

        Args:
            country_name (str): Name of the country
            all_country_names (list[str]): List of all country names
            n_industries (int): Number of industries in the economy
            functions (dict[str, Any]): Function implementations for household behavior
            ts (TimeSeries): Time series data container
            states (dict): State variables and parameters
            consumption_weights (np.ndarray): Industry-specific consumption shares
            consumption_weights_by_income (np.ndarray): Income-based consumption patterns
            investment_weights (np.ndarray): Industry-specific investment shares
            use_consumption_weights_by_income (bool): Whether to use income-based weights
            uses_feasibility_resolver (bool): Whether the Stage 5 live
                feasibility handoff is enabled.
            independents (list[str]): Independent variables for calculations
            substitution_bundles (Optional[list]): Substitution bundle configuration for CES consumption
            emission_fractions (Optional[EmissionFractions]): Per-industry emission fraction multipliers
        """
        n_entities = ts.current("n_households")
        super().__init__(
            country_name,
            all_country_names,
            n_industries,
            n_entities,
            n_entities,
            ts,
            states,
            transactor_settings={
                "Buyer Value Type": ValueType.NOMINAL,
                "Seller Value Type": ValueType.NONE,
                "Buyer Priority": 0,
                "Seller Priority": 0,
            },
        )

        self.functions = functions

        self.independents = independents

        # Set initial values
        self.ts["saving_rates_histogram"] = get_histogram(self.get_saving_rates_by_household(), None)

        self.consumption_weights = consumption_weights
        self.consumption_weights_by_income = consumption_weights_by_income.astype(float)

        self.investment_weights = investment_weights

        self.use_consumption_weights_by_income = use_consumption_weights_by_income
        self.uses_feasibility_resolver = uses_feasibility_resolver
        self.pre_grant_feasible_plan: PreGrantFeasiblePlan | None = None
        self.post_grant_feasible_plan: PostGrantFeasiblePlan | None = None

        # Initialize substitution bundles and bundle matrix
        self.substitution_bundles = substitution_bundles if substitution_bundles is not None else []
        if len(self.substitution_bundles) > 0:
            self.bundle_matrix = create_bundle_matrix(np.array(self.substitution_bundles))
        else:
            self.bundle_matrix = None

        self.emission_fractions = emission_fractions
        self._consumption_units_dirty = True
        self._consumption_unit_composition_signature: tuple[tuple[int, int], ...] | None = None

    @classmethod
    def from_pickled_agent(
        cls,
        synthetic_population: SyntheticPopulation,
        synthetic_country: SyntheticCountry,
        configuration: HouseholdsConfiguration,
        country_name: str,
        all_country_names: list[str],
        industries: list[str],
        initial_consumption_by_industry: np.ndarray,
        value_added_tax: float,
        scale: int,
        add_emissions: bool = False,
        emission_fractions: Optional[EmissionFractions] = None,
    ) -> "Households":
        """Create household agent from synthetic data.

        Initializes household agent using:
        - Synthetic population demographics
        - Country-specific parameters
        - Initial consumption/investment patterns
        - Tax rates and scaling factors

        Args:
            synthetic_population (SyntheticPopulation): Synthetic household data
            synthetic_country (SyntheticCountry): Country-specific parameters
            configuration (HouseholdsConfiguration): Household behavior config
            country_name (str): Name of the country
            all_country_names (list[str]): List of all country names
            industries (list[str]): List of industry names
            initial_consumption_by_industry (np.ndarray): Initial consumption
            value_added_tax (float): VAT rate
            scale (int): Scaling factor for histograms
            add_emissions (bool): Whether to track emissions
            emission_fractions (Optional[EmissionFractions]): Per-industry emission fraction multipliers

        Returns:
            Households: Initialized household agent
        """
        individual_ages = synthetic_population.individual_data["Age"].values

        corr_individuals = synthetic_population.household_data["Corresponding Individuals ID"]
        corr_individuals = corr_individuals.rename_axis("Household ID")

        corr_renters = synthetic_population.household_data["Corresponding Renters"]
        corr_renters = corr_renters.rename_axis("Household ID")

        functions = functions_from_model(model=configuration.functions, loc="macromodel.agents.households")

        hh_data = (
            synthetic_population.household_data.drop(
                columns=[
                    "Corresponding Individuals ID",
                    "Corresponding Renters",
                    "Corresponding Additionally Owned Houses ID",
                ]
            )
            .astype(float)
            .rename_axis("Household ID")
        )

        consumption_weights = synthetic_population.consumption_weights

        consumption_weights_by_income = synthetic_population.consumption_weights_by_income.T

        investment_weights = synthetic_population.investment_weights

        # Additional states
        states: dict[str, float | np.ndarray | list[np.ndarray] | Any] = {
            "saving_rates_model": synthetic_population.saving_rates_model,
            "social_transfers_model": synthetic_population.social_transfers_model,
            "wealth_distribution_model": synthetic_population.wealth_distribution_model,
            "average_saving_rate": synthetic_population.household_data["Saving Rate"].mean(),
            "coefficient_fa_income": synthetic_population.coefficient_fa_income,
            "investment_rate": synthetic_population.household_data["Investment Rate"].values,
        }

        # Stage 4 (portfolio choice) extensive-margin participation, frozen at init.
        # participation_source: initial_ifa_positive (consumption_paper_parameters.yaml,
        # portfolio_composition block) — households with no initial illiquid financial
        # assets never participate in IFA acquisition for the rest of the simulation.
        # This is derived here (not read from an HFCS column like the state_name_aliases
        # loop below) because it is a boolean transform of an already-loaded balance-sheet
        # field, not a separate synthetic-population column.
        states["portfolio_participates"] = hh_data["Wealth in Other Financial Assets"].to_numpy(dtype=float) > 0.0

        # Stage 4 (portfolio choice) Increment 5: household-head age and employed-
        # member count, required by the opt-in target_share_source="frm_magnitude"
        # path (compute_frm_magnitude_target_share). Frozen at init like
        # portfolio_participates above, consistent with how every other FRM
        # covariate (Investment Attitudes, Tenure Status, Wealth Quintile) is
        # sourced in this method. No production code aggregates individual-level
        # state to household level via "Relation to Reference Person" (RA0100)
        # before this; per the HFCS codebook convention (also used to flag the
        # gap in the Stage 4 architecture doc's FRM Variable Mapping Audit),
        # RA0100 == 1 identifies the reference person ("household head") within
        # each household's individual_data rows.
        #
        # NOTE — coexisting "household head" definition: run_model/src/mpc_analysis.py's
        # _household_head_age_proxy() defines "head" as the oldest linked individual
        # (np.nanmax over ages), with no reference to RA0100 at all. That proxy and this
        # FRM-covariate definition can disagree whenever HFCS's reference person is not
        # the oldest household member (e.g. a younger reference person with an older
        # dependent parent) — RA0100 == 1 here is the correct/intended definition for the
        # FRM model since the GTP-FRM estimation itself is defined on the reference
        # person, not the oldest member. If a future increment needs to compare or merge
        # MPC-by-age-cohort output with FRM-driven portfolio diagnostics, reconcile these
        # two head-selection rules explicitly rather than assuming they already agree.
        individual_activity_status = synthetic_population.individual_data["Activity Status"].to_numpy(dtype=float)
        individual_relation_to_reference_person = synthetic_population.individual_data[
            "Relation to Reference Person"
        ].to_numpy(dtype=float)
        is_employed = individual_activity_status == 1.0  # ActivityStatus.EMPLOYED, raw HFCS-coded value
        is_reference_person = individual_relation_to_reference_person == 1.0

        head_age, employed_member_count = compute_household_head_covariates(
            corr_individuals=corr_individuals.values,
            individual_ages=individual_ages,
            individual_is_employed=is_employed,
            individual_is_reference_person=is_reference_person,
        )

        states["household_head_age"] = head_age
        states["household_members_in_employment"] = employed_member_count
        # Required by target_share_source="frm_magnitude" to undo the population-
        # scale inflation applied to model financial-asset state (see
        # compute_frm_magnitude_target_share's docstring and the Stage 4
        # architecture doc's FRM Variable Mapping Audit). Stored once at init,
        # same as portfolio_participates above, since scale is a fixed
        # simulation-wide constant, not a per-period value.
        states["population_scale_factor"] = float(scale)

        # Additional states. "Wealth Quintile" aliases the long HFCS column label
        # ("Country quintile, gross wealth, among households") to a usable runtime key.
        state_name_aliases = {"Country quintile, gross wealth, among households": "Wealth Quintile"}
        for column_name in [
            "Type",
            "Corresponding Bank ID",
            "Corresponding Inhabited House ID",
            "Corresponding Property Owner",
            "Tenure Status of the Main Residence",
            "Investment Attitudes",
            "Country quintile, gross wealth, among households",
            "windfall_income",
        ]:
            if column_name not in hh_data.columns:
                raise ValueError(f"Missing {column_name} from the data for initialising households.")
            state_name = state_name_aliases.get(column_name, column_name)
            if column_name == "Type":
                states[state_name] = hh_data[column_name].values.flatten()
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter(action="ignore", category=RuntimeWarning)
                    # windfall_income is a boolean field with no NaNs by construction (reader fills with False),
                    # so the fillna(-1) is inert and this converts True/False to 1/0 integers.
                    states[state_name] = hh_data[column_name].fillna(-1).values.astype(int).flatten()
                    states[state_name][states[state_name] < 0] = -1

        # TODO: this is set to 0.2 in Sam's code, and transformed somehow into 0.0945. by the time the data is exported
        #  We need to 1. make this a parameters, 2. move this to the macro-data package.
        #  In general, we should think of where to put the piece of code below.

        investment_rate = synthetic_population.household_data["Investment Rate"].values
        # investment_weights = synthetic_country.industry_data["industry_vectors"]["Household Capital Inputs in LCU"]
        # investment_weights = investment_weights.values / investment_weights.values.sum()
        tau_cf = synthetic_country.tax_data.capital_formation_tax
        income = synthetic_population.household_data["Income"].values  # Income is different from Sam's

        initial_investment = pd.DataFrame(
            data=(1.0 / (1 + tau_cf) * np.outer(investment_weights, investment_rate * income).T),
            index=pd.Index(range(len(synthetic_population.household_data))),
            columns=pd.Index(synthetic_country.industries, name="Industry"),
        )

        tau_vat = synthetic_country.tax_data.value_added_tax

        consumption_by_industry_hh = 1 / (1 + tau_vat) * synthetic_population.industry_consumption_before_vat

        if add_emissions:
            consumption_emissions = synthetic_population.household_data["Consumption Emissions"].values
            investment_emissions = synthetic_population.household_data["Investment Emissions"].values
            consumption_emissions_by_good = np.zeros_like(industries, dtype=float)
            investment_emissions_by_good = np.zeros_like(industries, dtype=float)
            consumption_emissions_ch4_by_good = np.zeros_like(industries, dtype=float)
            investment_emissions_ch4_by_good = np.zeros_like(industries, dtype=float)
            coal_consumption_emissions = synthetic_population.household_data["Coal Consumption Emissions"].values
            gas_consumption_emissions = synthetic_population.household_data["Gas Consumption Emissions"].values
            oil_consumption_emissions = synthetic_population.household_data["Oil Consumption Emissions"].values
            refined_products_consumption_emissions = synthetic_population.household_data[
                "Refined Products Consumption Emissions"
            ].values
            coal_investment_emissions = synthetic_population.household_data["Coal Investment Emissions"].values
            gas_investment_emissions = synthetic_population.household_data["Gas Investment Emissions"].values
            oil_investment_emissions = synthetic_population.household_data["Oil Investment Emissions"].values
            refined_products_investment_emissions = synthetic_population.household_data[
                "Refined Products Investment Emissions"
            ].values
        else:
            consumption_emissions = None
            investment_emissions = None
            consumption_emissions_by_good = None
            investment_emissions_by_good = None
            consumption_emissions_ch4_by_good = None
            investment_emissions_ch4_by_good = None
            coal_consumption_emissions = None
            gas_consumption_emissions = None
            oil_consumption_emissions = None
            refined_products_consumption_emissions = None
            coal_investment_emissions = None
            gas_investment_emissions = None
            oil_investment_emissions = None
            refined_products_investment_emissions = None

        ts = create_households_timeseries(
            data=hh_data,
            initial_consumption_by_industry=initial_consumption_by_industry,
            initial_hh_investment=initial_investment.values,
            initial_investment_by_industry=synthetic_population.investment,
            initial_hh_consumption=consumption_by_industry_hh,
            scale=scale,
            vat=value_added_tax,
            tau_cf=tau_cf,
            consumption_emissions=consumption_emissions,
            investment_emissions=investment_emissions,
            consumption_emissions_by_good=consumption_emissions_by_good,
            investment_emissions_by_good=investment_emissions_by_good,
            consumption_emissions_ch4_by_good=consumption_emissions_ch4_by_good,
            investment_emissions_ch4_by_good=investment_emissions_ch4_by_good,
            coal_consumption_emissions=coal_consumption_emissions,
            gas_consumption_emissions=gas_consumption_emissions,
            oil_consumption_emissions=oil_consumption_emissions,
            refined_products_consumption_emissions=refined_products_consumption_emissions,
            coal_investment_emissions=coal_investment_emissions,
            gas_investment_emissions=gas_investment_emissions,
            oil_investment_emissions=oil_investment_emissions,
            refined_products_investment_emissions=refined_products_investment_emissions,
        )

        # Stage 4 (portfolio choice): register the diagnostic time series at init,
        # regardless of whether uses_portfolio_choice is active, so that
        # Households.update_wealth() can always call .append(...) on them via
        # TimeSeries.__getattr__ (which raises KeyError on a field never assigned
        # at least once). Initial values are zero/False, matching the Increment 2
        # helper's documented non-finite-input fallback contract; this is the
        # same value the call site would produce anyway when uses_portfolio_choice
        # is False, so the t=0 row is consistent with every later disabled period.
        n_households_at_init = ts.current("n_households")
        for _field_name, _zero_value in _STAGE4_DIAGNOSTIC_INITIAL_VALUES.items():
            ts[_field_name] = np.full(n_households_at_init, _zero_value)

        # Stage 5 (feasibility resolver): register diagnostic time series at
        # init, for the same reason as the
        # Stage 4 block above (TimeSeries.__getattr__ requires at least one
        # prior assignment before .append(...) can be called).
        for _field_name, _zero_value in _STAGE5_DIAGNOSTIC_INITIAL_VALUES.items():
            ts[_field_name] = np.full(n_households_at_init, _zero_value)

        # Update the household type
        states["Type"] = map_to_enum(states["Type"], HouseholdType)

        # Corresponding individuals
        states["corr_individuals"] = list(corr_individuals.values)

        # Number of adults individuals in the household
        states["Number of Adults"] = np.array(
            [
                np.sum(individual_ages[states["corr_individuals"][hh_id]] >= 18)
                for hh_id in range(ts.current("n_households"))
            ]
        )
        # Stage 2 proxy: consumption units are computed from initial household ages.
        states["Consumption Units"] = np.array(
            [
                Households._compute_consumption_units_from_ages(individual_ages[states["corr_individuals"][hh_id]])
                for hh_id in range(ts.current("n_households"))
            ],
            dtype=float,
        )

        # Corresponding renters
        states["corr_renters"] = [[int(x) for x in sublist if not pd.isna(x)] for sublist in corr_renters]

        use_consumption_weights_by_income = configuration.take_consumption_weights_by_income_quantile
        uses_feasibility_resolver = configuration.parameters.uses_feasibility_resolver

        independents = configuration.functions.saving_rates.parameters["independents"]

        # TODO: corresponding additionally owned houses is not used

        households = cls(
            country_name,
            all_country_names,
            len(industries),
            functions,
            ts,
            states,
            consumption_weights,
            consumption_weights_by_income,
            investment_weights,
            use_consumption_weights_by_income,
            uses_feasibility_resolver,
            independents,
            configuration.substitution_bundles,
            emission_fractions=emission_fractions,
        )
        households.refresh_consumption_units_if_needed(individual_ages)
        return households

    def reset(self, configuration: HouseholdsConfiguration) -> None:
        """Reset household agent to initial state.

        Updates function implementations based on new configuration.

        Args:
            configuration (HouseholdsConfiguration): New household config
        """
        self.gen_reset()
        update_functions(functions=self.functions, model=configuration.functions, loc="macromodel.agents.households")
        self.use_consumption_weights_by_income = configuration.take_consumption_weights_by_income_quantile
        self.configure_feasibility_resolver(configuration.parameters.uses_feasibility_resolver)

    def configure_feasibility_resolver(self, uses_feasibility_resolver: bool) -> None:
        """Configure whether the live Stage 5 feasibility handoff is active."""
        self.uses_feasibility_resolver = bool(uses_feasibility_resolver)
        self.pre_grant_feasible_plan = None
        self.post_grant_feasible_plan = None

    def clear_pre_grant_feasible_plan(self) -> None:
        """Clear the runtime Stage 5 live feasibility carrier."""
        self.pre_grant_feasible_plan = None

    def clear_post_grant_feasible_plan(self) -> None:
        """Clear the runtime Stage 5 settled feasibility carrier."""
        self.post_grant_feasible_plan = None

    def populate_pre_grant_feasible_plan_from_liquid_asset_drawdown(
        self,
        *,
        liquidity_shortfall_before_repair: np.ndarray,
        funded_from_liquid_assets: np.ndarray,
        residual_shortfall_after_lfa: np.ndarray,
    ) -> None:
        """Populate the minimal Increment 4 live feasibility carrier.

        The carrier is provisional pre-grant planning state only. It mirrors
        the existing Stage 5 liquid-drawdown diagnostics and must not mutate
        wealth or debt stocks.
        """
        self.pre_grant_feasible_plan = PreGrantFeasiblePlan(
            liquidity_shortfall_before_repair=np.asarray(liquidity_shortfall_before_repair, dtype=float).copy(),
            funded_from_liquid_assets=np.asarray(funded_from_liquid_assets, dtype=float).copy(),
            residual_shortfall_after_lfa=np.asarray(residual_shortfall_after_lfa, dtype=float).copy(),
        )

    def current_live_post_drawdown_residual(self) -> np.ndarray:
        """Return the sanctioned Stage 5 live residual read path.

        When the resolver is active, later increments must consume the live
        post-drawdown residual from ``pre_grant_feasible_plan`` instead of
        recomputing from raw liquidity shortfall. When the resolver is
        disabled, this accessor must remain byte-identical to the existing
        post-liquid-drawdown shadow handoff.
        """
        if self.uses_feasibility_resolver:
            if self.pre_grant_feasible_plan is None:
                raise RuntimeError(
                    "Stage 5 live feasibility resolver is enabled but pre_grant_feasible_plan "
                    "has not been populated for the current planning pass."
                )
            residual = self.pre_grant_feasible_plan.residual_shortfall_after_lfa
        else:
            residual = self.ts.current("residual_shortfall_after_lfa")

        residual = np.asarray(residual, dtype=float)
        return np.where(np.isfinite(residual), np.maximum(residual, 0.0), 0.0)

    def populate_pre_grant_feasible_plan_credit_requested(
        self,
        *,
        credit_requested: np.ndarray,
    ) -> None:
        """Extend the live Stage 5 carrier with Increment 5's credit_requested field.

        Additive only: uses ``dataclasses.replace`` against the carrier
        Increment 4 already populated earlier in the same planning pass, so
        the Increment 4 populate method is untouched. Raises if that carrier
        does not exist yet, since credit_requested has no meaning without
        the Increment 4 fields it sits alongside.
        """
        if self.pre_grant_feasible_plan is None:
            raise RuntimeError(
                "Stage 5 live feasibility resolver is enabled but pre_grant_feasible_plan "
                "has not been populated for the current planning pass."
            )
        # dataclasses.replace only shallow-copies: re-copy the Increment 4
        # fields too, so the new carrier instance never aliases arrays with
        # the old one it replaces.
        self.pre_grant_feasible_plan = replace(
            self.pre_grant_feasible_plan,
            liquidity_shortfall_before_repair=self.pre_grant_feasible_plan.liquidity_shortfall_before_repair.copy(),
            funded_from_liquid_assets=self.pre_grant_feasible_plan.funded_from_liquid_assets.copy(),
            residual_shortfall_after_lfa=self.pre_grant_feasible_plan.residual_shortfall_after_lfa.copy(),
            credit_requested=np.asarray(credit_requested, dtype=float).copy(),
        )

    def current_live_credit_requested(self) -> np.ndarray:
        """Return the sanctioned Stage 5 live credit-demand read path.

        When the resolver is active, ``compute_target_credit`` must consume
        this live, carrier-backed value instead of the legacy unbounded-gap
        formula. When the resolver is disabled, this accessor must remain
        byte-identical to the existing shadow diagnostic
        (``shadow_credit_requested``).
        """
        if self.uses_feasibility_resolver:
            if self.pre_grant_feasible_plan is None:
                raise RuntimeError(
                    "Stage 5 live feasibility resolver is enabled but pre_grant_feasible_plan "
                    "has not been populated for the current planning pass."
                )
            if self.pre_grant_feasible_plan.credit_requested is None:
                raise RuntimeError(
                    "Stage 5 live feasibility resolver is enabled and pre_grant_feasible_plan "
                    "exists, but credit_requested has not been populated for the current "
                    "planning pass."
                )
            credit_requested = self.pre_grant_feasible_plan.credit_requested
        else:
            credit_requested = self.ts.current("shadow_credit_requested")

        credit_requested = np.asarray(credit_requested, dtype=float)
        return np.where(np.isfinite(credit_requested), np.maximum(credit_requested, 0.0), 0.0)

    def populate_pre_grant_feasible_plan_planned_liquidation(
        self,
        *,
        planned_liquidation_total: np.ndarray,
        current_ifa: np.ndarray,
    ) -> None:
        """Extend the live Stage 5 carrier with planned illiquid liquidation.

        Increment 6 promotes the already-sanctioned Increment 3 liquidation
        amount into the live carrier. It does not re-run borrow-vs-sell or
        DSTI logic; validation here is limited to keeping runtime state finite
        and within current illiquid-asset availability.
        """
        if self.pre_grant_feasible_plan is None:
            raise RuntimeError(
                "Stage 5 live feasibility resolver is enabled but pre_grant_feasible_plan "
                "has not been populated for the current planning pass."
            )

        n_households = self.ts.current("n_households")
        liquidation = np.asarray(planned_liquidation_total, dtype=float)
        ifa = np.asarray(current_ifa, dtype=float)
        if liquidation.shape != (n_households,):
            raise ValueError(
                "planned_liquidation_total must contain exactly one value per household; "
                f"expected shape {(n_households,)}, got {liquidation.shape}."
            )
        if ifa.shape != (n_households,):
            raise ValueError(
                "current_ifa must contain exactly one value per household; "
                f"expected shape {(n_households,)}, got {ifa.shape}."
            )
        feasible_ifa = np.where(np.isfinite(ifa), np.maximum(ifa, 0.0), 0.0)
        cleaned_liquidation = np.where(np.isfinite(liquidation), np.maximum(liquidation, 0.0), 0.0)
        cleaned_liquidation = np.minimum(cleaned_liquidation, feasible_ifa)

        self.pre_grant_feasible_plan = replace(
            self.pre_grant_feasible_plan,
            liquidity_shortfall_before_repair=self.pre_grant_feasible_plan.liquidity_shortfall_before_repair.copy(),
            funded_from_liquid_assets=self.pre_grant_feasible_plan.funded_from_liquid_assets.copy(),
            residual_shortfall_after_lfa=self.pre_grant_feasible_plan.residual_shortfall_after_lfa.copy(),
            credit_requested=(
                None
                if self.pre_grant_feasible_plan.credit_requested is None
                else self.pre_grant_feasible_plan.credit_requested.copy()
            ),
            planned_liquidation_total=cleaned_liquidation.copy(),
        )

    def current_live_planned_liquidation_total(self) -> np.ndarray:
        """Return the sanctioned Stage 5 live planned-liquidation read path."""
        if self.uses_feasibility_resolver:
            if self.pre_grant_feasible_plan is None:
                raise RuntimeError(
                    "Stage 5 live feasibility resolver is enabled but pre_grant_feasible_plan "
                    "has not been populated for the current planning pass."
                )
            if self.pre_grant_feasible_plan.planned_liquidation_total is None:
                raise RuntimeError(
                    "Stage 5 live feasibility resolver is enabled and pre_grant_feasible_plan "
                    "exists, but planned_liquidation_total has not been populated for the "
                    "current planning pass."
                )
            planned_liquidation = self.pre_grant_feasible_plan.planned_liquidation_total
        else:
            planned_liquidation = self.ts.current("liquidation_planned")

        planned_liquidation = np.asarray(planned_liquidation, dtype=float)
        return np.where(np.isfinite(planned_liquidation), np.maximum(planned_liquidation, 0.0), 0.0)

    def populate_post_grant_feasible_plan_from_granted_credit(
        self,
        *,
        credit_granted: np.ndarray,
    ) -> None:
        """Build the settled Stage 5 carrier from cleared consumer credit."""
        if self.pre_grant_feasible_plan is None:
            raise RuntimeError(
                "Stage 5 live feasibility resolver is enabled but pre_grant_feasible_plan "
                "has not been populated for the current planning pass."
            )
        if self.pre_grant_feasible_plan.credit_requested is None:
            raise RuntimeError(
                "Stage 5 post-grant reconciliation requires pre_grant_feasible_plan.credit_requested "
                "to be populated for the current planning pass."
            )
        if self.pre_grant_feasible_plan.planned_liquidation_total is None:
            raise RuntimeError(
                "Stage 5 post-grant reconciliation requires "
                "pre_grant_feasible_plan.planned_liquidation_total to be populated for the current "
                "planning pass."
            )

        n_households = self.ts.current("n_households")
        granted = np.asarray(credit_granted, dtype=float)
        if granted.shape != (n_households,):
            raise ValueError(
                "credit_granted must contain exactly one value per household; "
                f"expected shape {(n_households,)}, got {granted.shape}."
            )

        requested = np.asarray(self.pre_grant_feasible_plan.credit_requested, dtype=float)
        planned_liquidation = np.asarray(self.pre_grant_feasible_plan.planned_liquidation_total, dtype=float)
        residual_after_lfa = np.asarray(self.pre_grant_feasible_plan.residual_shortfall_after_lfa, dtype=float)
        for name, values in (
            ("pre_grant_feasible_plan.credit_requested", requested),
            ("pre_grant_feasible_plan.planned_liquidation_total", planned_liquidation),
            ("pre_grant_feasible_plan.residual_shortfall_after_lfa", residual_after_lfa),
        ):
            if values.shape != (n_households,):
                raise ValueError(
                    f"{name} must contain exactly one value per household; "
                    f"expected shape {(n_households,)}, got {values.shape}."
                )

        cleaned_granted = np.where(np.isfinite(granted), np.maximum(granted, 0.0), 0.0)
        cleaned_requested = np.where(np.isfinite(requested), np.maximum(requested, 0.0), 0.0)
        cleaned_liquidation = np.where(np.isfinite(planned_liquidation), np.maximum(planned_liquidation, 0.0), 0.0)
        cleaned_residual_after_lfa = np.where(np.isfinite(residual_after_lfa), np.maximum(residual_after_lfa, 0.0), 0.0)

        self.post_grant_feasible_plan = PostGrantFeasiblePlan(
            credit_granted=cleaned_granted.copy(),
            credit_rationing_gap=np.maximum(cleaned_requested - cleaned_granted, 0.0).copy(),
            planned_liquidation_total=cleaned_liquidation.copy(),
            residual_shortfall_after_granted_credit=np.maximum(
                cleaned_residual_after_lfa - cleaned_granted - cleaned_liquidation,
                0.0,
            ).copy(),
        )

    def compute_employee_income(
        self,
        individual_income: np.ndarray,
        corr_households: np.ndarray,
    ) -> np.ndarray:
        """Calculate household income from employment.

        Aggregates individual employment income to household level.

        Args:
            individual_income (np.ndarray): Income by individual
            corr_households (np.ndarray): Individual-household mapping

        Returns:
            np.ndarray: Employment income by household
        """
        return np.bincount(
            corr_households,
            weights=individual_income,
            minlength=self.ts.current("n_households"),
        )

    def compute_expected_social_transfer_income(
        self,
        total_other_social_transfers: float,
        cpi: float,
        expected_inflation: float,
    ) -> np.ndarray:
        """Calculate expected social transfer income.

        Computes expected transfers based on:
        - Total transfer budget
        - Price level changes
        - Inflation expectations

        Args:
            total_other_social_transfers (float): Total transfer budget
            cpi (float): Current price index
            expected_inflation (float): Expected inflation rate

        Returns:
            np.ndarray: Expected transfers by household
        """
        inds = self.independents
        return (
            (1 + expected_inflation)
            * cpi
            * self.functions["social_transfers"].get_social_transfers(
                n_households=self.ts.current("n_households"),
                total_other_social_transfers=total_other_social_transfers,
                current_independents=(
                    np.array([])
                    if len(inds) == 0
                    else np.stack(
                        [self.ts.current(ind.lower()) for ind in inds],
                        axis=1,
                    )
                ),
                initial_independents=(
                    np.array([])
                    if len(inds) == 0
                    else np.stack(
                        [self.ts.initial(ind.lower()) for ind in inds],
                        axis=1,
                    )
                ),
                model=self.states["social_transfers_model"],
            )
        )

    def compute_social_transfer_income(
        self,
        total_other_social_transfers: float,
        cpi: float,
    ) -> np.ndarray:
        """Calculate current social transfer income.

        Computes actual transfers based on:
        - Total transfer budget
        - Current price level

        Args:
            total_other_social_transfers (float): Total transfer budget
            cpi (float): Current price index

        Returns:
            np.ndarray: Current transfers by household
        """
        inds = self.independents
        return cpi * self.functions["social_transfers"].get_social_transfers(
            n_households=self.ts.current("n_households"),
            total_other_social_transfers=total_other_social_transfers,
            current_independents=(
                np.array([])
                if len(inds) == 0
                else np.stack(
                    [self.ts.current(ind.lower()) for ind in inds],
                    axis=1,
                )
            ),
            initial_independents=(
                np.array([])
                if len(inds) == 0
                else np.stack(
                    [self.ts.initial(ind.lower()) for ind in inds],
                    axis=1,
                )
            ),
            model=self.states["social_transfers_model"],
        )

    def compute_rental_income(
        self,
        housing_data: pd.DataFrame,
        income_taxes: float,
    ) -> np.ndarray:
        """Calculate rental income from property ownership.

        Computes after-tax rental income from:
        - Rented properties
        - Current rent levels
        - Tax rates

        Args:
            housing_data (pd.DataFrame): Property market data
            income_taxes (float): Income tax rate

        Returns:
            np.ndarray: Rental income by household
        """
        housing_data_rented_out = housing_data.loc[
            np.logical_and(
                housing_data["Is Owner-Occupied"] == 0,
                housing_data["Corresponding Inhabitant Household ID"] != -1,
            )
        ]
        housing_data_rented_out_grouped = housing_data_rented_out.groupby("Corresponding Owner Household ID")[
            "Rent"
        ].sum()
        rental_income = np.zeros(self.ts.current("n_households"))
        rental_income[housing_data_rented_out_grouped.index.values] = (
            1 - income_taxes
        ) * housing_data_rented_out_grouped.values
        return rental_income

    def compute_expected_income_from_financial_assets(self) -> np.ndarray:
        """Calculate expected income from financial assets.

        Estimates future financial income based on:
        - Asset holdings
        - Return coefficients
        - Historical patterns

        Returns:
            np.ndarray: Expected financial income by household
        """
        wealth_function = self.functions["wealth"]
        paper_expected_income = getattr(wealth_function, "compute_expected_income_from_financial_assets", None)
        if paper_expected_income is not None:
            return paper_expected_income(
                current_wealth_in_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
            )
        return self.functions["financial_assets"].compute_expected_income(
            income_coefficient=self.states["coefficient_fa_income"],
            initial_other_financial_assets=self.ts.initial("wealth_other_financial_assets"),
            current_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
        )

    def compute_income_from_financial_assets(self, period_index: int | None = None) -> np.ndarray:
        """Calculate current income from financial assets.

        Computes actual financial income based on:
        - Current asset holdings
        - Realized returns
        - Income coefficients

        Returns:
            np.ndarray: Current financial income by household
        """
        wealth_function = self.functions["wealth"]
        paper_income = getattr(wealth_function, "compute_income_from_financial_assets", None)
        if paper_income is not None:
            kwargs = {
                "current_wealth_in_other_financial_assets": self.ts.current("wealth_other_financial_assets"),
            }
            if getattr(wealth_function, "uses_periodic_illiquid_returns", False):
                kwargs["period_index"] = period_index
            return paper_income(**kwargs)
        return self.functions["financial_assets"].compute_income(
            income_coefficient=self.states["coefficient_fa_income"],
            initial_other_financial_assets=self.ts.initial("wealth_other_financial_assets"),
            current_other_financial_assets=self.ts.current("wealth_other_financial_assets"),
        )

    def current_illiquid_financial_asset_return_rate(self) -> float:
        """Return the current aggregate illiquid financial asset return rate, if available."""
        current_rate = getattr(self.functions["wealth"], "current_illiquid_return_rate", None)
        if current_rate is None:
            return np.nan
        return current_rate()

    def current_illiquid_financial_asset_return_amount(self) -> np.ndarray:
        """Return the current-period illiquid return amount vector, if available."""
        current_amount = getattr(self.functions["wealth"], "current_illiquid_return_amount", None)
        current_wealth = self.ts.current("wealth_other_financial_assets")
        if current_amount is None:
            return np.full(current_wealth.shape, np.nan)
        try:
            return current_amount(current_wealth_in_other_financial_assets=current_wealth)
        except ValueError:
            return np.full(current_wealth.shape, np.nan)

    def compute_expected_income(self) -> np.ndarray:
        """Calculate total expected income.

        Aggregates expected income from all sources:
        - Employment
        - Social transfers
        - Rental income
        - Financial assets

        Returns:
            np.ndarray: Total expected income by household
        """
        return (
            self.ts.current("expected_income_employee")
            + self.ts.current("expected_income_social_transfers")
            + self.ts.current("income_rental")
            + self.ts.current("expected_income_financial_assets")
        )

    def compute_income(self) -> np.ndarray:
        """Calculate total current income.

        Aggregates current income from all sources:
        - Employment
        - Social transfers
        - Rental income
        - Financial assets

        Returns:
            np.ndarray: Total current income by household
        """
        return (
            self.ts.current("income_employee")
            + self.ts.current("income_social_transfers")
            + self.ts.current("income_rental")
            + self.ts.current("income_financial_assets")
        )

    def compute_non_property_income(self) -> np.ndarray:
        """Calculate non-property current income (employment + social transfers + rental).

        Excludes stochastic illiquid financial asset returns, which can be large and
        negative after bad market draws and are unrelated to the permanent-income concept
        used in Bayesian income-belief learning. This series is the appropriate signal
        for the Kalman updater, consistent with the CES calibration of the learning priors.

        Returns:
            np.ndarray: Non-property income by household
        """
        return (
            self.ts.current("income_employee")
            + self.ts.current("income_social_transfers")
            + self.ts.current("income_rental")
        )

    def get_saving_rates_by_household(self) -> np.ndarray:
        """Calculate household-specific saving rates.

        Determines saving rates based on:
        - Average saving behavior
        - Household characteristics
        - Economic conditions

        Returns:
            np.ndarray: Saving rates by household
        """
        inds = self.independents
        if len(inds) > 0:
            current_independents = np.stack(
                [self.ts.current(ind.lower()) for ind in inds],
                axis=1,
            )
            initial_independents = np.stack(
                [self.ts.initial(ind.lower()) for ind in inds],
                axis=1,
            )
        else:
            current_independents = np.array([])
            initial_independents = np.array([])
        return self.functions["saving_rates"].get_saving_rates(
            n_households=self.ts.current("n_households"),
            average_saving_rate=self.states["average_saving_rate"],
            current_independents=current_independents,
            initial_independents=initial_independents,
            model=self.states["saving_rates_model"],
        )

    def compute_target_consumption(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        exogenous_total_consumption: float,
        per_capita_unemployment_benefits: float,
        tau_vat: float,
        assume_zero_growth: bool,
        prices: Optional[np.ndarray] = None,
        initial_prices: Optional[np.ndarray] = None,
        taxes: Optional[np.ndarray] = None,
        initial_taxes: Optional[np.ndarray] = None,
        income_override: Optional[np.ndarray] = None,
        lagged_income_override: Optional[np.ndarray] = None,
        lagged_cpi: Optional[float] = None,
        house_price_index: Optional[float] = None,
        house_price_growth: Optional[float] = None,
        lagged_house_price_index: Optional[float] = None,
        lagged_housing_wealth: Optional[np.ndarray] = None,
        real_borrowing_rate: Optional[float] = None,
        consumer_debt_rate_delta: Optional[float] = None,
        permanent_income_log_ratio: Optional[np.ndarray] = None,
        permanent_income_log_ratio_individual: Optional[np.ndarray] = None,
        permanent_income_log_ratio_common: Optional[np.ndarray] = None,
        uncertainty_delta: Optional[np.ndarray] = None,
        common_permanent_income_log_ratio: Optional[np.ndarray | float] = None,
        mortgage_payment: Optional[np.ndarray] = None,
        replace_current_diagnostics: bool = False,
        time_unit: int = 12,
    ) -> np.ndarray:
        """Calculate target consumption levels.

        Determines desired consumption based on:
        - Income and saving rates
        - Price level changes
        - Benefit levels
        - Growth assumptions
        - Tax rates
        - CES substitution within bundles (if enabled)

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            exogenous_total_consumption (float): External consumption target
            per_capita_unemployment_benefits (float): Per person benefits
            tau_vat (float): Value added tax rate
            assume_zero_growth (bool): Whether to assume no growth
            prices (Optional[np.ndarray]): Current prices by industry for CES substitution
            initial_prices (Optional[np.ndarray]): Initial prices by industry for CES substitution
            taxes (Optional[np.ndarray]): Current tax rates by industry for CES substitution
            initial_taxes (Optional[np.ndarray]): Initial tax rates by industry for CES substitution
            lagged_income_override (Optional[np.ndarray]): Explicit previous spendable-income base for shock tests
            lagged_cpi (Optional[float]): Previous consumer CPI level for lagged real inputs
            house_price_index (Optional[float]): House-price index level used by credit-augmented consumption
            house_price_growth (Optional[float]): House-price growth proxy used by credit-augmented consumption
            lagged_house_price_index (Optional[float]): Lagged HPI index level for the paper house-price term
            lagged_housing_wealth (Optional[np.ndarray]): Lagged gross housing wealth for the paper
                housing-wealth term; defaults to the previous period's housing stock
            real_borrowing_rate (Optional[float]): Explicit real-rate proxy; defaults to zero placeholder
            consumer_debt_rate_delta (Optional[float]): Explicit consumer-debt-rate delta placeholder
            permanent_income_log_ratio (Optional[np.ndarray]): Explicit permanent-income placeholder
            permanent_income_log_ratio_individual (Optional[np.ndarray]): Household-specific
                ``zeta * posterior_mean`` component of ``ln(y^p / p)``
            permanent_income_log_ratio_common (Optional[np.ndarray]): Broadcast
                ``common_log_ratio`` component of ``ln(y^p / p)``
            uncertainty_delta (Optional[np.ndarray]): Explicit uncertainty placeholder
            common_permanent_income_log_ratio (Optional[np.ndarray | float]): Separately supplied macro
                permanent-income component for opt-in learning rules
            mortgage_payment (Optional[np.ndarray]): Mortgage-only scheduled service by household
            replace_current_diagnostics (bool): Replace latest target diagnostics instead of appending
            time_unit (int): Model period length in months, used by credit-augmented consumption
                to annualize income in its calibrated wealth/income and price/income ratios

        Returns:
            np.ndarray: Target consumption by household
        """
        saving_rates = self.get_saving_rates_by_household()
        self.ts.saving_rates_histogram.append(get_histogram(saving_rates, None))

        # Target consumption
        if assume_zero_growth:
            target_consumption = np.outer(
                self.ts.initial("consumption"),
                self.states["consumption_weights_data"],
            ).astype(float)
            self._append_target_consumption_diagnostics(None, replace_current=replace_current_diagnostics)
            return target_consumption
        else:
            income = self.ts.current("expected_income") if income_override is None else income_override
            lagged_income = (
                self.ts.prev("expected_income") if lagged_income_override is None else lagged_income_override
            )
            mortgage_payment = (
                np.zeros(self.ts.current("n_households")) if mortgage_payment is None else mortgage_payment
            )
            tenure_status = self.states["Tenure Status of the Main Residence"]
            owner_occupied = np.isin(tenure_status, [1, 2, 4]).astype(float)
            mortgagor = (self.ts.current("mortgage_debt") > 0.0).astype(float)
            if lagged_housing_wealth is None:
                lagged_housing_wealth = self.ts.prev("wealth_main_residence") + self.ts.prev("wealth_other_properties")
            target_consumption = self.functions["consumption"].compute_target_consumption(
                expected_inflation=expected_inflation,
                current_cpi=current_cpi,
                initial_cpi=initial_cpi,
                historic_consumption_sum=np.array(self.ts.historic("consumption")),
                saving_rates=saving_rates,
                income=income,
                household_benefits=self.states["Number of Adults"] * per_capita_unemployment_benefits
                + self.ts.current("expected_income_social_transfers"),
                consumption_weights=self.consumption_weights,
                consumption_weights_by_income=self.consumption_weights_by_income,
                exogenous_total_consumption=exogenous_total_consumption,
                current_time=len(self.ts.historic("total_consumption")),
                take_consumption_weights_by_income_quantile=self.use_consumption_weights_by_income,
                tau_vat=tau_vat,
                prices=prices,
                initial_prices=initial_prices,
                taxes=taxes,
                initial_taxes=initial_taxes,
                bundle_matrix=self.bundle_matrix,
                liquid_wealth=self.ts.current("wealth_deposits"),
                illiquid_wealth=self.ts.current("wealth_other_financial_assets"),
                housing_wealth=self.ts.current("wealth_main_residence") + self.ts.current("wealth_other_properties"),
                lagged_housing_wealth=lagged_housing_wealth,
                rent=self.ts.current("rent"),
                mortgage_debt=self.ts.current("mortgage_debt"),
                mortgage_payment=mortgage_payment,
                owner_occupied=owner_occupied,
                mortgagor=mortgagor,
                house_price_index=house_price_index,
                house_price_growth=house_price_growth,
                lagged_consumption=self.ts.prev("consumption"),
                lagged_income=lagged_income,
                lagged_cpi=lagged_cpi,
                lagged_liquid_wealth=self.ts.prev("wealth_deposits"),
                lagged_illiquid_wealth=self.ts.prev("wealth_other_financial_assets"),
                lagged_mortgage_debt=self.ts.prev("mortgage_debt"),
                lagged_consumption_loan_debt=self.ts.prev("consumption_loan_debt"),
                lagged_house_price_index=lagged_house_price_index,
                real_borrowing_rate=real_borrowing_rate,
                permanent_income_log_ratio=permanent_income_log_ratio,
                consumer_debt_rate_delta=consumer_debt_rate_delta,
                uncertainty_delta=uncertainty_delta,
                population_scale_factor=self.states.get("population_scale_factor"),
                time_unit=time_unit,
            )
            components = getattr(self.functions["consumption"], "last_target_consumption_components", None)
            if components is not None:
                n_households = int(self.ts.current("n_households"))

                def _diagnostic_array(values: Optional[np.ndarray]) -> np.ndarray:
                    if values is None:
                        return np.zeros(n_households)
                    array = np.asarray(values, dtype=float)
                    if array.ndim == 0:
                        return np.full(n_households, float(array), dtype=float)
                    return array

                if permanent_income_log_ratio_individual is not None:
                    components["target_consumption_permanent_income_log_ratio_individual"] = _diagnostic_array(
                        permanent_income_log_ratio_individual
                    )
                if permanent_income_log_ratio_common is not None:
                    components["target_consumption_permanent_income_log_ratio_common"] = _diagnostic_array(
                        permanent_income_log_ratio_common
                    )
            self._append_target_consumption_diagnostics(
                self.functions["consumption"],
                replace_current=replace_current_diagnostics,
            )
            return target_consumption

    def compute_and_record_liquidity_shortfall(
        self,
        target_consumption: np.ndarray,
        scheduled_debt_service: np.ndarray,
        income_override: Optional[np.ndarray] = None,
        replace_current: bool = False,
    ) -> np.ndarray:
        """Compute and persist the Stage 5 (feasibility resolver) liquidity-shortfall diagnostic.

        Diagnostics-only (Increment 0): appends ``liquidity_shortfall`` and
        ``household_saving`` time series and has no effect on goods or credit
        demand. Must be called after ``compute_target_consumption()`` for the
        current period, since it consumes that period's ``target_consumption``.

        Uses ``expected_income`` by default (the current-period income basis
        used by ``compute_target_consumption()`` itself, see ``households.py``'s
        ``income = self.ts.current("expected_income") if income_override is
        None else income_override``), not ``income`` (the realized series,
        which is only appended later in ``Country.update_realised_metrics()``
        — after both call sites of this method run, per ``simulation.py``'s
        per-period ordering). Using ``income`` would silently compare this
        period's consumption plan against last period's realized income, an
        off-by-one-period mismatch caught in round-2 review (not by the
        original hand-computed unit tests, which use synthetic scalars and
        can't see this ordering bug). ``income_override`` mirrors
        ``compute_target_consumption()``'s own override parameter so that, if
        a future caller ever passes an override there (e.g. a shock-test
        scenario), this diagnostic stays on the same income basis rather than
        silently reverting to ``expected_income`` underneath it — the same
        class of bug round-2 found, pre-empted here rather than left latent.
        ``Country._set_household_target_demand()`` does not pass an override
        today, so this is currently always ``None`` in production.

        See ``knowledge-vault/wiki/architecture/consumption-stage-5-feasibility-resolver.md``
        (Increment 0 section) for the paper's ``L^d_it = -(s_it + b_it)``
        definition and the exit criterion.

        Args:
            target_consumption (np.ndarray): This period's per-household
                target consumption, summed across goods (i.e. the same total
                already returned by ``compute_target_consumption()``).
            scheduled_debt_service (np.ndarray): Total scheduled mortgage plus
                consumer-loan instalments for the period, per household.
            income_override (Optional[np.ndarray]): Explicit income basis,
                forwarded unchanged if supplied; defaults to
                ``self.ts.current("expected_income")`` otherwise, matching
                ``compute_target_consumption()``'s own override semantics.
            replace_current (bool): Replace the latest appended diagnostic
                row instead of appending a new one (mirrors the
                ``replace_current_diagnostics`` convention used elsewhere in
                this class).

        Returns:
            np.ndarray: Per-household liquidity shortfall, ``L^d_it``.
        """
        income = self.ts.current("expected_income") if income_override is None else income_override
        result = compute_liquidity_shortfall(
            income=income,
            target_consumption=np.asarray(target_consumption, dtype=float).sum(axis=1),
            scheduled_debt_service=scheduled_debt_service,
        )
        if replace_current:
            self.ts.override_current("liquidity_shortfall", result.liquidity_shortfall)
            self.ts.override_current("household_saving", result.household_saving)
        else:
            self.ts.liquidity_shortfall.append(result.liquidity_shortfall)
            self.ts.household_saving.append(result.household_saving)
        return result.liquidity_shortfall

    def compute_and_record_liquid_asset_drawdown(
        self,
        liquidity_shortfall: np.ndarray,
        replace_current: bool = False,
    ) -> np.ndarray:
        """Compute and persist the Stage 5 Increment 1 shadow drawdown diagnostic."""
        liquidity_shortfall = np.asarray(liquidity_shortfall, dtype=float)
        liquidity_shortfall_before_repair = np.where(
            np.isfinite(liquidity_shortfall),
            np.maximum(liquidity_shortfall, 0.0),
            0.0,
        )
        result = compute_liquid_asset_drawdown(
            liquidity_shortfall=liquidity_shortfall_before_repair,
            available_lfa=self.ts.current("wealth_deposits"),
        )
        if replace_current:
            self.ts.override_current("liquidity_shortfall_before_repair", liquidity_shortfall_before_repair)
            self.ts.override_current("funded_from_liquid_assets", result.funded_from_liquid_assets)
            self.ts.override_current("residual_shortfall_after_lfa", result.residual_shortfall_after_lfa)
        else:
            self.ts.liquidity_shortfall_before_repair.append(liquidity_shortfall_before_repair)
            self.ts.funded_from_liquid_assets.append(result.funded_from_liquid_assets)
            self.ts.residual_shortfall_after_lfa.append(result.residual_shortfall_after_lfa)
        return result.residual_shortfall_after_lfa

    def compute_and_record_borrow_vs_sell_choice(
        self,
        residual_shortfall_after_lfa: np.ndarray,
        banks: Banks,
        stage4_handoff: dict[str, np.ndarray],
        replace_current: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute and persist the Stage 5 Increment 2 shadow branch choice."""
        bank_rates = np.asarray(banks.ts.current("interest_rates_on_household_consumption_loans"), dtype=float)
        corresponding_bank_ids = np.asarray(self.states["Corresponding Bank ID"], dtype=int)
        if bank_rates.ndim == 0:
            bank_rates = np.full(self.ts.current("n_households"), float(bank_rates), dtype=float)
        elif bank_rates.ndim == 1:
            if bank_rates.shape[0] == self.ts.current("n_households"):
                pass
            elif bank_rates.shape[0] == banks.ts.current("n_banks"):
                bank_rates = bank_rates[corresponding_bank_ids]
            else:
                bank_rates = np.resize(bank_rates, banks.ts.current("n_banks"))[corresponding_bank_ids]
        else:
            raise ValueError(
                "interest_rates_on_household_consumption_loans must be a scalar, one rate per household, "
                "or one rate per bank."
            )
        result = compute_borrow_vs_sell_choice(
            residual_shortfall_after_lfa=np.asarray(residual_shortfall_after_lfa, dtype=float),
            delta_tilde=stage4_handoff["delta_tilde"],
            opening_tfa_scale=stage4_handoff["opening_tfa_scale"],
            post_return_ifa=stage4_handoff["post_return_ifa"],
            r_b=bank_rates,
            r_kappa=stage4_handoff["r_kappa"],
            phi_1=getattr(self.functions["wealth"], "phi_1", np.nan),
            lambda_kappa=getattr(self.functions["wealth"], "lambda_kappa", np.nan),
        )
        if replace_current:
            self.ts.override_current("preferred_margin_after_lfa", result.preferred_margin)
            self.ts.override_current("preferred_margin_amount", result.preferred_amount)
            self.ts.override_current("borrow_vs_sell_threshold", result.borrow_vs_sell_threshold)
            self.ts.override_current("borrow_vs_sell_spread", result.borrow_vs_sell_spread)
            self.ts.override_current("borrow_vs_sell_l_tilde", result.borrow_vs_sell_l_tilde)
            self.ts.override_current("borrow_vs_sell_comparison_valid_flag", result.comparison_valid_flag)
        else:
            self.ts.preferred_margin_after_lfa.append(result.preferred_margin)
            self.ts.preferred_margin_amount.append(result.preferred_amount)
            self.ts.borrow_vs_sell_threshold.append(result.borrow_vs_sell_threshold)
            self.ts.borrow_vs_sell_spread.append(result.borrow_vs_sell_spread)
            self.ts.borrow_vs_sell_l_tilde.append(result.borrow_vs_sell_l_tilde)
            self.ts.borrow_vs_sell_comparison_valid_flag.append(result.comparison_valid_flag)
        return result.preferred_margin, result.preferred_amount

    def compute_and_record_residual_capacity_fallback(
        self,
        preferred_margin_after_lfa: np.ndarray,
        preferred_margin_amount: np.ndarray,
        banks: Banks,
        income: np.ndarray,
        scheduled_mortgage_payment: np.ndarray,
        consumer_loan_maturity: int,
        dsti_limit: float,
        current_ifa: np.ndarray,
        replace_current: bool = False,
    ) -> ResidualCapacityFallbackResult:
        """Compute and persist the Stage 5 Increment 3 shadow residual-capacity fallback.

        This uses the provisional DSTI proxy only. It records the shadow plan
        but does not touch live balances, debt service, or market-clearing
        state.
        """
        bank_rates = np.asarray(banks.ts.current("interest_rates_on_household_consumption_loans"), dtype=float)
        corresponding_bank_ids = np.asarray(self.states["Corresponding Bank ID"], dtype=int)
        if bank_rates.ndim == 0:
            bank_rates = np.full(self.ts.current("n_households"), float(bank_rates), dtype=float)
        elif bank_rates.ndim == 1:
            if bank_rates.shape[0] == self.ts.current("n_households"):
                pass
            elif bank_rates.shape[0] == banks.ts.current("n_banks"):
                bank_rates = bank_rates[corresponding_bank_ids]
            else:
                bank_rates = np.resize(bank_rates, banks.ts.current("n_banks"))[corresponding_bank_ids]
        else:
            raise ValueError(
                "interest_rates_on_household_consumption_loans must be a scalar, one rate per household, "
                "or one rate per bank."
            )

        result = compute_residual_capacity_fallback(
            preferred_margin_after_lfa=np.asarray(preferred_margin_after_lfa, dtype=float),
            preferred_margin_amount=np.asarray(preferred_margin_amount, dtype=float),
            income=np.asarray(income, dtype=float),
            scheduled_mortgage_payment=np.asarray(scheduled_mortgage_payment, dtype=float),
            r_b=bank_rates,
            consumer_loan_maturity=consumer_loan_maturity,
            dsti_limit=dsti_limit,
            current_ifa=np.asarray(current_ifa, dtype=float),
        )
        if replace_current:
            self.ts.override_current("dsti_headroom", result.dsti_headroom)
            self.ts.override_current("dsti_maximum_loan_size", result.dsti_maximum_loan_size)
            self.ts.override_current("dsti_cap_binding", result.dsti_cap_binding)
            self.ts.override_current("borrow_planned", result.borrow_planned)
            self.ts.override_current("liquidation_planned", result.liquidation_planned)
            self.ts.override_current("shadow_credit_requested", result.shadow_credit_requested)
            self.ts.override_current("forced_liquidation_amount", result.forced_liquidation_amount)
            self.ts.override_current("residual_shortfall_after_caps", result.residual_shortfall_after_caps)
        else:
            self.ts.dsti_headroom.append(result.dsti_headroom)
            self.ts.dsti_maximum_loan_size.append(result.dsti_maximum_loan_size)
            self.ts.dsti_cap_binding.append(result.dsti_cap_binding)
            self.ts.borrow_planned.append(result.borrow_planned)
            self.ts.liquidation_planned.append(result.liquidation_planned)
            self.ts.shadow_credit_requested.append(result.shadow_credit_requested)
            self.ts.forced_liquidation_amount.append(result.forced_liquidation_amount)
            self.ts.residual_shortfall_after_caps.append(result.residual_shortfall_after_caps)
        return result

    def current_stage4_handoff_for_stage5(
        self,
        *,
        target_consumption_total: np.ndarray,
        scheduled_debt_service: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build the current-period Stage 4 handoff needed by Stage 5.

        This uses the same pure Stage 4 diagnostics helper as
        ``compute_stage4_portfolio_diagnostics()`` but does not persist any
        ``portfolio_*`` time series. When portfolio choice is disabled, the
        returned arrays intentionally force the Increment 2 fallback path.
        """
        n_households = self.ts.current("n_households")
        wealth_function = self.functions["wealth"]
        if not getattr(wealth_function, "uses_portfolio_choice", False):
            return {
                "delta_tilde": np.full(n_households, np.nan),
                "opening_tfa_scale": np.full(n_households, np.nan),
                "post_return_ifa": np.full(n_households, np.nan),
                "r_kappa": np.full(n_households, np.nan),
            }

        opening_tfa_scale = self.ts.prev("wealth_other_financial_assets") + self.ts.prev("wealth_deposits")
        current_return_amount = self.current_illiquid_financial_asset_return_amount()
        if not np.all(np.isfinite(current_return_amount)):
            current_return_amount = self.compute_income_from_financial_assets()
        post_return_ifa = self.ts.current("wealth_other_financial_assets") + current_return_amount
        investable_surplus = (
            self.ts.current("expected_income")
            - np.asarray(target_consumption_total, dtype=float)
            - np.asarray(scheduled_debt_service, dtype=float)
        )
        post_surplus_lfa = self.ts.current("wealth_deposits")

        frm_covariates = None
        frm_magnitude_coefficients = None
        population_scale_factor = None
        net_wealth_scale_divisor = None
        if wealth_function.target_share_source == "frm_magnitude":
            tenure_status = self.states["Tenure Status of the Main Residence"]
            frm_covariates = {
                "age": self.states["household_head_age"],
                "household_members_in_employment": self.states["household_members_in_employment"],
                "investment_attitudes": self.states["Investment Attitudes"],
                "mortgagor": (tenure_status == 2).astype(float),
                "owner": (tenure_status == 1).astype(float),
                "net_wealth": self.compute_net_wealth(),
            }
            frm_coefficients = self.states["frm_coefficients"]
            frm_magnitude_coefficients = frm_coefficients.magnitude_coefficients
            population_scale_factor = self.states["population_scale_factor"]
            net_wealth_scale_divisor = frm_coefficients.net_wealth_scale_divisor

        diagnostics = compute_stage4_household_diagnostics(
            opening_tfa_scale=opening_tfa_scale,
            post_surplus_lfa=post_surplus_lfa,
            post_return_ifa=post_return_ifa,
            investable_surplus=investable_surplus,
            frm_covariates=frm_covariates,
            frm_magnitude_coefficients=frm_magnitude_coefficients,
            population_scale_factor=population_scale_factor,
            net_wealth_scale_divisor=net_wealth_scale_divisor,
            portfolio_participates=self.states["portfolio_participates"],
            target_share_source=wealth_function.target_share_source,
            default_target_illiquid_share=wealth_function.default_target_illiquid_share,
            phi_1=wealth_function.phi_1,
            lambda_kappa=wealth_function.lambda_kappa,
            fixed_cost_share=wealth_function.fixed_cost_share,
        )
        return {
            "delta_tilde": diagnostics.rebalancing.delta_tilde,
            "opening_tfa_scale": diagnostics.portfolio_opening_tfa_scale,
            "post_return_ifa": diagnostics.portfolio_post_return_ifa,
            "r_kappa": np.full(n_households, self.current_illiquid_financial_asset_return_rate()),
        }

    def update_income_belief_learning_state(
        self,
        *,
        current_income: np.ndarray,
        lagged_income: np.ndarray | None,
    ) -> IncomeBeliefLearningOutputs | None:
        """Advance optional income-belief posterior state."""
        priors = self.states.get("income_belief_priors")
        if priors is None:
            raise ValueError(
                "Income-belief learning was requested by the consumption rule, "
                "but households.states['income_belief_priors'] is not available."
            )
        runtime_state = self._income_belief_runtime_state(priors)
        if lagged_income is None:
            self._append_income_belief_diagnostics(None)
            return None
        outputs = compute_income_belief_learning_outputs(
            current_income=current_income,
            lagged_income=lagged_income,
            priors=priors,
            prior_mean=runtime_state["posterior_mean"],
            prior_variance=runtime_state["posterior_variance"],
            growth_clip_bound=getattr(self.functions["consumption"], "income_belief_growth_clip_bound", 1.0),
        )
        runtime_state["posterior_mean"] = outputs.posterior_mean.copy()
        runtime_state["posterior_variance"] = outputs.posterior_variance.copy()
        self._append_income_belief_diagnostics(outputs)
        return outputs

    def _append_income_belief_diagnostics(self, outputs: IncomeBeliefLearningOutputs | None) -> None:
        """Persist floor/fallback/growth-clip flags so degenerate Kalman updates are
        visible in runtime output instead of being silently discarded (see GH issue #90).
        """
        n_households = self.ts.current("n_households")
        if outputs is None:
            zero_series = np.zeros(n_households)
            self.ts.income_belief_floor_used.append(zero_series.copy())
            self.ts.income_belief_posterior_fallback_used.append(zero_series.copy())
            self.ts.income_belief_growth_clipped.append(zero_series.copy())
            return
        self.ts.income_belief_floor_used.append(outputs.floor_used.astype(float))
        self.ts.income_belief_posterior_fallback_used.append(outputs.posterior_fallback_used.astype(float))
        self.ts.income_belief_growth_clipped.append(outputs.growth_clipped.astype(float))

    def current_income_belief_learning_inputs(
        self,
        *,
        common_permanent_income_log_ratio: float | None = None,
        common_forecast_variance: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Map posterior beliefs into Stage 3 consumption-rule inputs.

        Combines the cached scalar weight ``zeta`` with the current posterior
        beliefs (``income_belief_mu``/``income_belief_p``) and the broadcast
        common-forecast terms to produce ``permanent_income_log_ratio`` and
        ``uncertainty_delta`` per the Stage 3 architecture note (item 5).

        ``common_permanent_income_log_ratio``/``common_forecast_variance`` are
        scalar floats broadcast across households; ``None`` (forecast
        unavailable) is treated as ``0.0``, reducing each output to its pure
        individual component.
        """
        priors = self.states.get("income_belief_priors")
        if priors is None:
            raise ValueError(
                "Income-belief learning was requested by the consumption rule, "
                "but households.states['income_belief_priors'] is not available."
            )
        runtime_state = self._income_belief_runtime_state(priors)
        zeta = self._income_belief_zeta(priors)
        common_log_ratio = 0.0 if common_permanent_income_log_ratio is None else common_permanent_income_log_ratio
        common_variance = 0.0 if common_forecast_variance is None else common_forecast_variance
        individual_log_ratio = float(zeta) * runtime_state["posterior_mean"]
        common_log_ratio_arr = np.full(runtime_state["posterior_mean"].shape, float(common_log_ratio), dtype=float)
        return {
            "permanent_income_log_ratio": compute_permanent_income_log_ratio(
                runtime_state["posterior_mean"], zeta, common_log_ratio
            ),
            "permanent_income_log_ratio_individual": individual_log_ratio,
            "permanent_income_log_ratio_common": common_log_ratio_arr,
            "uncertainty_delta": compute_income_uncertainty(runtime_state["posterior_variance"], zeta, common_variance),
        }

    def _income_belief_zeta(self, priors: dict[str, np.ndarray]) -> float:
        """Return the cached scalar weight zeta, computing it once on first use.

        zeta depends only on the scalars ``(rho, delta, S)`` so it is cached on
        ``self.states['income_belief_zeta']`` rather than recomputed per period.
        ``delta``/``S`` come from the consumption rule's configured
        ``income_belief_learning_horizon``; a missing value raises rather than
        silently defaulting, since zeta has real economic meaning.
        """
        if "income_belief_zeta" not in self.states:
            consumption = self.functions["consumption"]
            delta = getattr(consumption, "income_belief_learning_delta", None)
            horizon_S = getattr(consumption, "income_belief_learning_S", None)
            if delta is None or horizon_S is None:
                raise ValueError(
                    "Income-belief learning is enabled but the consumption rule has no "
                    "income_belief_learning_horizon (delta/S). Configure "
                    "income_belief_learning.permanent_income_log_ratio "
                    "via the paper_parameter reference; there is no safe default for zeta."
                )
            n_households = int(self.ts.current("n_households"))
            rho = _scalar_rho(priors, n_households)
            self.states["income_belief_zeta"] = compute_zeta(rho, delta, horizon_S)
        return self.states["income_belief_zeta"]

    def _income_belief_runtime_state(self, priors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return mutable posterior state, seeded from static priors once."""
        n_households = int(self.ts.current("n_households"))
        runtime_state = self.states.get("income_belief_runtime_state")
        if runtime_state is None:
            runtime_state = {
                "posterior_mean": np.asarray(priors["income_belief_mu"], dtype=float).copy(),
                "posterior_variance": np.asarray(priors["income_belief_p"], dtype=float).copy(),
            }
            self.states["income_belief_runtime_state"] = runtime_state
        for key in ["posterior_mean", "posterior_variance"]:
            value = np.asarray(runtime_state[key], dtype=float)
            if value.shape != (n_households,):
                raise ValueError(
                    f"income_belief_runtime_state[{key!r}] must have shape ({n_households},), got {value.shape}."
                )
            runtime_state[key] = value
        return runtime_state

    @staticmethod
    def _target_consumption_diagnostic_keys() -> list[str]:
        return [
            "target_consumption_lagged_consumption",
            "target_consumption_real_income",
            "target_consumption_lagged_real_income",
            "target_consumption_real_lagged_consumption",
            "target_consumption_long_run",
            "target_consumption_log_long_run",
            "target_consumption_permanent_income",
            "target_consumption_liquid_wealth",
            "target_consumption_illiquid_wealth",
            "target_consumption_housing_wealth",
            "target_consumption_real_net_liquid_assets",
            "target_consumption_real_illiquid_financial_assets",
            "target_consumption_real_housing_wealth",
            "target_consumption_real_lagged_housing_wealth",
            "target_consumption_real_consumer_debt",
            "target_consumption_rent",
            "target_consumption_mortgage_debt",
            "target_consumption_mortgage_payment",
            "target_consumption_rent_diagnostic",
            "target_consumption_mortgage_debt_diagnostic",
            "target_consumption_mortgage_payment_diagnostic",
            "target_consumption_house_price",
            "target_consumption_interest_rate_cashflow",
            "target_consumption_uncertainty",
            "target_consumption_partial_adjustment_gap",
            "target_consumption_income_growth",
            "target_consumption_house_price_index",
            "target_consumption_lagged_house_price_index",
            "target_consumption_real_lagged_house_price",
            "target_consumption_real_borrowing_rate",
            "target_consumption_permanent_income_log_ratio",
            "target_consumption_permanent_income_log_ratio_individual",
            "target_consumption_permanent_income_log_ratio_common",
            "target_consumption_consumer_debt_rate_delta",
            "target_consumption_interest_rate_cashflow_index",
            "target_consumption_uncertainty_delta",
            "target_consumption_owner_occupied",
            "target_consumption_mortgagor",
            "target_consumption_delta_log_consumption",
            "target_consumption_growth_clipped",
            "target_consumption_alpha_2",
            "target_consumption_gamma_1",
            "target_consumption_wealth_drag_clipped",
        ]

    def _append_target_consumption_diagnostics(
        self,
        consumption_function: Any | None,
        *,
        replace_current: bool = False,
    ) -> None:
        n_households = self.ts.current("n_households")
        diagnostic_keys = self._target_consumption_diagnostic_keys()

        if (
            consumption_function is None
            or getattr(consumption_function, "last_target_consumption_components", None) is None
        ):
            zero_series = np.zeros(n_households)
            for key in diagnostic_keys:
                if replace_current:
                    self.ts.override_current(key, zero_series.copy())
                else:
                    getattr(self.ts, key).append(zero_series.copy())
            if replace_current:
                self.ts.override_current("formula_implied_mpc", np.zeros(n_households))
            else:
                self.ts.formula_implied_mpc.append(np.zeros(n_households))
            return

        components = consumption_function.last_target_consumption_components
        for key in diagnostic_keys:
            value = np.asarray(components.get(key, np.zeros(n_households)), dtype=float)
            if replace_current:
                self.ts.override_current(key, value)
            else:
                getattr(self.ts, key).append(value)
        formula_implied_mpc = np.asarray(consumption_function.last_formula_implied_mpc, dtype=float)
        if replace_current:
            self.ts.override_current("formula_implied_mpc", formula_implied_mpc)
        else:
            self.ts.formula_implied_mpc.append(formula_implied_mpc)

    @staticmethod
    def _compute_consumption_units_from_ages(ages: np.ndarray) -> float:
        """Return paper-style household consumption units from member ages.

        The data available here has ages, not the paper's exact adult/child
        dependency state. Stage 2 uses age >= 14 for the 0.5 CU additional-member
        category and age < 14 for the 0.3 CU child category.
        """
        ages = np.asarray(ages, dtype=float)
        if len(ages) == 0:
            return 1.0
        older_members = int(np.sum(ages >= 14.0))
        younger_children = int(np.sum(ages < 14.0))
        return 1.0 + 0.5 * max(0, older_members - 1) + 0.3 * younger_children

    def _current_consumption_units(self) -> np.ndarray:
        return np.asarray(
            self.states.get("Consumption Units", np.ones(self.ts.current("n_households"))),
            dtype=float,
        )

    def _compute_subsistence_consumption_shortfall(
        self,
        subsistence_consumption: np.ndarray | None,
        target_consumption: np.ndarray,
    ) -> np.ndarray:
        """Compute the CU-adjusted subsistence shortfall as a diagnostic.

        This is a feasibility-layer diagnostic only: it records how far actual
        target consumption falls below the subsistence floor, but has no
        behavioural effect on goods-market clearing.
        """
        n_households = int(self.ts.current("n_households"))
        if subsistence_consumption is None:
            return np.zeros(n_households, dtype=float)
        floor = np.asarray(subsistence_consumption, dtype=float)
        current_consumption_budget = np.asarray(target_consumption, dtype=float).sum(axis=1)
        return np.maximum(0.0, floor - current_consumption_budget)

    def compute_target_investment(
        self,
        expected_inflation: float,
        current_cpi: float,
        initial_cpi: float,
        exogenous_total_investment: float,
        tau_cf: float,
        assume_zero_growth: bool,
    ) -> np.ndarray:
        """Calculate target investment levels.

        Determines desired investment based on:
        - Income and investment rates
        - Price level changes
        - External targets
        - Growth assumptions
        - Tax rates

        Args:
            expected_inflation (float): Expected inflation rate
            current_cpi (float): Current price index
            initial_cpi (float): Initial price index
            exogenous_total_investment (float): External investment target
            tau_cf (float): Capital formation tax rate
            assume_zero_growth (bool): Whether to assume no growth

        Returns:
            np.ndarray: Target investment by household
        """
        if assume_zero_growth:
            return self.ts.initial("investment").astype(float)
        else:
            return self.functions["investment"].compute_target_investment(
                expected_inflation=expected_inflation,
                current_cpi=current_cpi,
                initial_cpi=initial_cpi,
                income=self.ts.current("expected_income"),
                exogenous_total_investment=exogenous_total_investment,
                current_time=len(self.ts.historic("total_investment")),
                investment_weights=self.investment_weights,
                investment_rate=self.states["investment_rate"],
                tau_cf=tau_cf,
            )

    def prepare_housing_market_clearing(
        self,
        housing_data: pd.DataFrame,
        observed_fraction_value_price: np.ndarray,
        observed_fraction_rent_value: np.ndarray,
        expected_hpi_growth: float,
        assumed_mortgage_maturity: int,
        rental_income_taxes: float,
        time_unit: int,
    ) -> None:
        """Prepare for housing market clearing.

        Sets up housing market participation through:
        - Property demand decisions
        - Price/rent willingness
        - Sale listings
        - Rental offerings

        Args:
            housing_data (pd.DataFrame): Property market data
            observed_fraction_value_price (np.ndarray): Price/value ratios
            observed_fraction_rent_value (np.ndarray): Rent/value ratios
            expected_hpi_growth (float): Expected house price growth
            assumed_mortgage_maturity (int): Mortgage term length
            rental_income_taxes (float): Tax rate on rental income
            time_unit (int): Model period length in months
        """
        if len(housing_data) == 0:
            return

        # Households make decisions on their demand for properties
        (
            max_price_willing_to_pay,
            max_rent_willing_to_pay,
            households_hoping_to_move,
        ) = self.functions["property"].compute_demand(
            housing_data=housing_data,
            household_residence_tenure_status=self.states["Tenure Status of the Main Residence"],
            household_income=self.ts.current("expected_income"),
            household_financial_wealth=self.ts.current("wealth_financial_assets"),
            observed_fraction_value_price=observed_fraction_value_price,
            observed_fraction_rent_value=observed_fraction_rent_value,
            expected_hpi_growth=expected_hpi_growth,
            assumed_mortgage_maturity=assumed_mortgage_maturity,
            rental_income_taxes=rental_income_taxes,
        )
        self.ts.max_price_willing_to_pay.append(max_price_willing_to_pay)
        self.ts.max_rent_willing_to_pay.append(max_rent_willing_to_pay)

        # Set price of owner-occupied or vacant properties whose owners are hoping to move.
        household_ids_hoping_to_move = np.flatnonzero(households_hoping_to_move)
        owner_ids = housing_data["Corresponding Owner Household ID"]
        inhabitant_ids = housing_data["Corresponding Inhabitant Household ID"]
        owner_wants_to_move = owner_ids.isin(household_ids_hoping_to_move)
        property_is_vacant = inhabitant_ids.isna() | inhabitant_ids.eq(VACANT_HOUSEHOLD_ID)
        property_is_owner_occupied = inhabitant_ids.eq(owner_ids)
        ind_mhr_temp_sale = owner_wants_to_move & (property_is_vacant | property_is_owner_occupied)
        housing_data.loc[np.logical_not(ind_mhr_temp_sale), "Sale Price"] = np.nan
        ind_still_on_sale = housing_data["Temporarily for Sale"].copy()
        housing_data["Temporarily for Sale"] = False
        housing_data.loc[ind_mhr_temp_sale, "Temporarily for Sale"] = True
        housing_data.loc[
            np.logical_and(ind_mhr_temp_sale, np.logical_not(ind_still_on_sale)),
            "Sale Price",
        ] = self.functions["property"].compute_initial_sale_price(
            property_values=housing_data.loc[
                np.logical_and(ind_mhr_temp_sale, np.logical_not(ind_still_on_sale)),
                "Value",
            ],
        )
        housing_data.loc[np.logical_and(ind_mhr_temp_sale, ind_still_on_sale), "Sale Price"] = self.functions[
            "property"
        ].compute_updated_sale_price(
            sale_prices=housing_data.loc[
                np.logical_and(ind_mhr_temp_sale, ind_still_on_sale),
                "Sale Price",
            ],
            time_unit=time_unit,
        )

        # Set what's up for rent
        prev_up_for_rent = pd.Series(housing_data["Up for Rent"], index=housing_data.index).fillna(False).astype(bool)
        inhabitant_ids = housing_data["Corresponding Inhabitant Household ID"]
        now_up_for_rent = housing_data.index[inhabitant_ids.isna() | inhabitant_ids.eq(VACANT_HOUSEHOLD_ID)]
        newly_up_for_rent = now_up_for_rent[~prev_up_for_rent.loc[now_up_for_rent].values]
        housing_data["Up for Rent"] = False
        housing_data.loc[now_up_for_rent, "Up for Rent"] = True
        housing_data["Newly on the Rental Market"] = False
        housing_data.loc[newly_up_for_rent, "Newly on the Rental Market"] = True
        not_newly_up_for_rent = np.logical_and(
            np.logical_not(housing_data["Newly on the Rental Market"]),
            housing_data["Up for Rent"],
        )

        # Calculate rent
        housing_data.loc[housing_data["Newly on the Rental Market"], "Rent"] = self.functions[
            "property"
        ].compute_offered_rent_for_new_properties(
            property_value=housing_data.loc[housing_data["Newly on the Rental Market"], "Value"].values,
            observed_fraction_rent_value=observed_fraction_rent_value,
        )
        housing_data.loc[not_newly_up_for_rent, "Rent"] = self.functions[
            "property"
        ].compute_offered_rent_for_existing_properties(
            current_offered_rent=housing_data.loc[not_newly_up_for_rent, "Rent"].values,
            time_unit=time_unit,
        )

    def update_rent(
        self,
        housing_data: pd.DataFrame,
        historic_inflation: list[np.ndarray],
        exogenous_inflation_before: np.ndarray,
    ) -> None:
        """Update rental prices.

        Adjusts rents based on:
        - Historical inflation
        - External price changes
        - Market conditions

        Args:
            housing_data (pd.DataFrame): Property market data
            historic_inflation (list[np.ndarray]): Past inflation rates
            exogenous_inflation_before (np.ndarray): External inflation
        """
        housing_data["Rent"] = self.functions["property"].compute_rent(
            current_rent=housing_data["Rent"].values,
            historic_inflation=np.concatenate(
                (
                    exogenous_inflation_before,
                    np.array(historic_inflation).flatten(),
                )
            ),
        )

    def process_housing_market_clearing(
        self,
        housing_data: pd.DataFrame,
        social_housing_function: Any,
        current_sales: pd.DataFrame,
        current_unemployment_benefits_by_individual: float,
    ) -> None:
        """Process housing market clearing results.

        Updates housing market outcomes:
        - Rent payments
        - Property purchases
        - Social housing allocation
        - Market transactions

        Args:
            housing_data (pd.DataFrame): Property market data
            social_housing_function (Any): Social housing allocator
            current_sales (pd.DataFrame): Market transactions
            current_unemployment_benefits_by_individual (float): Benefits
        """
        # Calculate rent
        rent_by_household, imputed_rent_by_household = self.compute_rent(
            housing_data=housing_data,
            social_housing_function=social_housing_function,
            current_unemployment_benefits_by_individual=current_unemployment_benefits_by_individual,
        )
        self.ts.rent.append(rent_by_household)
        self.ts.rent_imputed.append(imputed_rent_by_household)

        # Calculate the price paid for property
        price_paid_for_property = np.zeros(self.ts.current("n_households"))
        if len(current_sales) > 0:
            price_paid_for_property[current_sales["buyer_id"].values] = current_sales["price_or_rent"].values
        self.ts.price_paid_for_property.append(price_paid_for_property)

    def compute_rent(
        self,
        housing_data: pd.DataFrame,
        social_housing_function: Any,
        current_unemployment_benefits_by_individual: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate rent payments and imputed rent.

        Determines:
        - Actual rent for renters
        - Social housing rent
        - Imputed rent for owners

        Args:
            housing_data (pd.DataFrame): Property market data
            social_housing_function (Any): Social housing allocator
            current_unemployment_benefits_by_individual (float): Benefits

        Returns:
            tuple[np.ndarray, np.ndarray]: Rent paid and imputed rent
        """
        rent_by_household = np.zeros(self.ts.current("n_households"))
        imputed_rent_by_household = np.zeros(self.ts.current("n_households"))

        # Households in social housing
        ind_social_housing = np.where(self.states["Corresponding Inhabited House ID"] == -1)[0]
        social_housing_rent = social_housing_function.compute_social_housing_rent(
            current_unemployment_benefits_by_individual=current_unemployment_benefits_by_individual,
            current_household_size=self.states["Number of Adults"][ind_social_housing],
        )
        rent_by_household[ind_social_housing] = social_housing_rent
        self.states["Rent paid to Government"] = social_housing_rent.sum()

        # Households renting
        ind_renting = np.all(
            [
                self.states["Tenure Status of the Main Residence"] == 3,
                self.states["Corresponding Inhabited House ID"] != -1,
            ],
            axis=0,
        )

        rent = housing_data.loc[
            self.states["Corresponding Inhabited House ID"][ind_renting],
            "Rent",
        ].values
        rent_by_household[ind_renting] = rent

        # Households owning
        ind_owning = np.all(
            [
                np.isin(self.states["Tenure Status of the Main Residence"], [1, 2, 4]),
                self.states["Corresponding Inhabited House ID"] != -1,
            ],
            axis=0,
        )

        rent = housing_data.loc[
            self.states["Corresponding Inhabited House ID"][ind_owning],
            "Rent",
        ].values
        imputed_rent_by_household[ind_owning] = rent

        return rent_by_household, imputed_rent_by_household

    def compute_target_credit(self, current_sales: pd.DataFrame | None) -> None:
        """Calculate target credit demand.

        Determines credit needs for:
        - Consumption financing
        - Property purchases
        - Debt management

        Args:
            current_sales (pd.DataFrame): Property transactions
        """
        # Target consumption loans to cover immediate financing gaps. The legacy
        # unbounded-gap formula is always computed first; Stage 5 Increment 5
        # substitutes the DSTI-capped live carrier value only when the
        # feasibility resolver is enabled, so flag-off behaviour is unchanged
        # and live_credit_requested gives a like-for-like diagnostic either way.
        legacy_target_consumption_loans = self.functions["target_credit"].compute_target_consumption_loans(
            target_consumption=self.ts.current("target_consumption"),
            income=self.ts.current("expected_income"),
            rent=self.ts.current("rent"),
            wealth_in_financial_assets=self.ts.current("wealth_financial_assets"),
        )
        if self.uses_feasibility_resolver:
            target_consumption_loans = self.current_live_credit_requested()
        else:
            target_consumption_loans = legacy_target_consumption_loans
        self.ts.target_consumption_loans.append(target_consumption_loans)
        self.ts.live_credit_requested.append(target_consumption_loans.copy())
        self.ts.total_target_consumption_loans.append([self.ts.current("target_consumption_loans").sum()])
        # Mortgages
        target_house_price = np.zeros(self.ts.current("n_households"))
        sells = None
        if current_sales is not None and len(current_sales) > 0:
            if "sales_types" in current_sales.columns:
                sells = current_sales.loc[current_sales["sales_types"] == "Sell"]
            else:
                # Backward compatibility: allow passing already-filtered sales rows.
                sells = current_sales

            if sells is not None and len(sells) > 0:
                missing = {"buyer_id", "price_or_rent"} - set(sells.columns)
                if missing:
                    raise ValueError(
                        f"current_sales is missing columns required for mortgage targeting: {sorted(missing)}"
                    )
                buyer_ids = sells["buyer_id"].to_numpy(dtype=int, copy=False)
                target_house_price[buyer_ids] = sells["price_or_rent"].to_numpy(dtype=float, copy=False)
        self.ts.target_mortgage.append(
            self.functions["target_credit"].compute_target_mortgage(
                target_house_price=target_house_price,
                target_consumption=self.ts.current("target_consumption"),
                income=self.ts.current("expected_income"),
                rent=self.ts.current("rent"),
                wealth_in_financial_assets=self.ts.current("wealth_financial_assets"),
            )
        )
        self.ts.total_target_mortgage.append([self.ts.current("target_mortgage").sum()])

    def compute_interest_paid_on_deposits(
        self,
        bank_interest_rate_on_household_deposits: np.ndarray,
        bank_overdraft_rate_on_household_deposits: np.ndarray,
    ) -> np.ndarray:
        """Calculate interest paid on deposits.

        Computes interest flows based on:
        - Deposit balances
        - Interest rates
        - Overdraft conditions

        Args:
            bank_interest_rate_on_household_deposits (np.ndarray): Deposit rates
            bank_overdraft_rate_on_household_deposits (np.ndarray): Overdraft rates

        Returns:
            np.ndarray: Interest paid by household
        """
        return -bank_interest_rate_on_household_deposits[self.states["Corresponding Bank ID"]] * np.maximum(
            0.0, self.ts.current("wealth_deposits")
        ) - bank_overdraft_rate_on_household_deposits[self.states["Corresponding Bank ID"]] * np.minimum(
            0.0, self.ts.current("wealth_deposits")
        )

    def compute_interest_paid(self) -> np.ndarray:
        """Calculate total interest paid.

        Aggregates interest payments on:
        - Deposits
        - Loans
        - Credit facilities

        Returns:
            np.ndarray: Total interest paid by household
        """
        return self.ts.current("interest_paid_on_loans") + self.ts.current("interest_paid_on_deposits")

    @staticmethod
    def _compute_consumption_units_from_ages(ages: np.ndarray) -> float:
        """Return household consumption units from member ages."""
        ages = np.asarray(ages, dtype=float)
        if len(ages) == 0:
            return 1.0
        older_members = int(np.sum(ages >= 14.0))
        younger_children = int(np.sum(ages < 14.0))
        return 1.0 + 0.5 * max(0, older_members - 1) + 0.3 * younger_children

    @staticmethod
    def _consumption_unit_signature_from_ages(ages: np.ndarray) -> tuple[int, int]:
        ages = np.asarray(ages, dtype=float)
        return int(np.sum(ages >= 14.0)), int(np.sum(ages < 14.0))

    def _household_consumption_unit_signature(self, individual_ages: np.ndarray) -> tuple[tuple[int, int], ...]:
        ages = np.asarray(individual_ages, dtype=float)
        return tuple(
            self._consumption_unit_signature_from_ages(ages[np.asarray(corr_individuals, dtype=int)])
            for corr_individuals in self.states["corr_individuals"]
        )

    def mark_consumption_units_dirty(self) -> None:
        self._consumption_units_dirty = True

    def refresh_consumption_units_if_needed(self, individual_ages: np.ndarray) -> bool:
        """Refresh household consumption units when age-band composition changes."""
        signature = self._household_consumption_unit_signature(individual_ages)
        if not self._consumption_units_dirty and signature == self._consumption_unit_composition_signature:
            return False

        ages = np.asarray(individual_ages, dtype=float)
        self.states["Consumption Units"] = np.array(
            [
                self._compute_consumption_units_from_ages(ages[np.asarray(corr_individuals, dtype=int)])
                for corr_individuals in self.states["corr_individuals"]
            ],
            dtype=float,
        )
        self._consumption_unit_composition_signature = signature
        self._consumption_units_dirty = False
        return True

    def current_consumption_units(self) -> np.ndarray:
        return np.asarray(
            self.states.get("Consumption Units", np.ones(self.ts.current("n_households"))),
            dtype=float,
        )

    def prepare_goods_market_clearing(
        self,
        exchange_rate_usd_to_lcu: float,
        subsistence_consumption: np.ndarray | None = None,
    ) -> np.ndarray:
        """Prepare for goods market clearing.

        Sets up market participation through:
        - Exchange rate adjustment
        - Purchase preparation
        - Sale preparation

        Args:
            exchange_rate_usd_to_lcu (float): USD to local currency rate
            subsistence_consumption (np.ndarray | None): CU-adjusted subsistence
                floor used only to compute a diagnostic shortfall series.

        Returns:
            np.ndarray: Per-household subsistence consumption shortfall
                (floor minus current consumption budget, clipped at zero).
                Diagnostic only; has no effect on goods-market clearing.
        """
        # Exchange rates
        self.set_exchange_rate(exchange_rate_usd_to_lcu)

        # Compute subsistence shortfall diagnostic (no effect on clearing budget)
        shortfall = self._compute_subsistence_consumption_shortfall(
            subsistence_consumption, self.ts.current("target_consumption")
        )

        # Prepare goods market clearing
        self.prepare_buying_goods()
        self.prepare_selling_goods()

        return shortfall

    def prepare_buying_goods(self) -> None:
        """Prepare goods purchase decisions.

        Sets up buying based on:
        - Target consumption
        - Target investment
        - Exchange rates
        """
        goods_budget = self.ts.current("target_consumption") + self.ts.current("target_investment")
        self.set_goods_to_buy(1.0 / self.exchange_rate_usd_to_lcu * goods_budget)

    def prepare_selling_goods(self) -> None:
        """Prepare goods sale decisions.

        Sets up selling based on:
        - Available goods
        - Price levels
        """
        self.set_goods_to_sell(np.zeros(self.ts.current("n_households")))
        self.set_prices(np.zeros(self.ts.current("n_households")))

    def update_consumption_and_investment(
        self,
        tau_vat: float,
        tau_cf: float,
        add_emissions: bool = False,
        readjusted_factors: Optional[np.ndarray] = None,
        emitting_indices: Optional[np.ndarray] = None,
        readjusted_factors_ch4: Optional[np.ndarray] = None,
        emitting_indices_ch4: Optional[np.ndarray] = None,
        use_emission_multiplier: bool = False,
    ) -> None:
        """Update consumption and investment outcomes.

        Records actual:
        - Consumption spending
        - Investment spending
        - Tax payments
        - Emissions data

        Args:
            tau_vat (float): Value added tax rate
            tau_cf (float): Capital formation tax rate
            add_emissions (bool): Whether to track emissions
            readjusted_factors (Optional[np.ndarray]): CO2 emission factors
            emitting_indices (Optional[np.ndarray]): CO2 emitting sector indices
            readjusted_factors_ch4 (Optional[np.ndarray]): CH4 emission factors
            emitting_indices_ch4 (Optional[np.ndarray]): CH4 emitting sector indices
            use_emission_multiplier (bool): Whether to apply industry-specific fraction multipliers
        """
        # Total amount spent
        self.ts.amount_bought.append(self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1))

        # Distribute
        consumption_by_good = np.minimum(
            self.ts.current("nominal_amount_spent_in_lcu"),
            self.ts.current("target_consumption"),
        )

        if add_emissions:
            # Apply per-industry consumption fraction multipliers when enabled.
            # emission_fractions.consumption has shape (1, n_industries); index row 0
            # then select emitting columns to get (n_emitting,) broadcast multipliers.
            if (
                use_emission_multiplier
                and self.emission_fractions is not None
                and self.emission_fractions.consumption is not None
            ):
                cons_fracs = self.emission_fractions.consumption[0, emitting_indices]
                cons_slice = consumption_by_good[:, emitting_indices] * cons_fracs
            else:
                cons_slice = consumption_by_good[:, emitting_indices]

            consumption_emissions = cons_slice @ readjusted_factors
            self.ts.consumption_emissions.append(consumption_emissions)

            consumption_sum = consumption_by_good.sum(axis=0)
            consumption_emissions_by_good = np.zeros(consumption_by_good.shape[1])
            for i in emitting_indices:
                idx = np.where(emitting_indices == i)[0]
                multiplier = (
                    self.emission_fractions.consumption[0, i]
                    if use_emission_multiplier
                    and self.emission_fractions is not None
                    and self.emission_fractions.consumption is not None
                    else 1.0
                )
                consumption_emissions_by_good[i] = (consumption_sum[i] * multiplier * readjusted_factors[idx]).item()
            self.ts.consumption_emissions_by_good.append(consumption_emissions_by_good)

            if emitting_indices_ch4 is not None and readjusted_factors_ch4 is not None:
                consumption_emissions_ch4_by_good = np.zeros(consumption_by_good.shape[1])
                for i in emitting_indices_ch4:
                    idx = np.where(emitting_indices_ch4 == i)[0]
                    consumption_emissions_ch4_by_good[i] = (consumption_sum[i] * readjusted_factors_ch4[idx]).item()
                self.ts.consumption_emissions_ch4_by_good.append(consumption_emissions_ch4_by_good)

            disaggregated_emissions = cons_slice * readjusted_factors
            self.ts.coal_consumption_emissions.append(disaggregated_emissions[:, 0])
            self.ts.oil_consumption_emissions.append(disaggregated_emissions[:, 1])
            self.ts.gas_consumption_emissions.append(disaggregated_emissions[:, 2])
            self.ts.refined_products_consumption_emissions.append(disaggregated_emissions[:, 3])

        # Consumption
        self.ts.consumption.append(consumption_by_good.sum(axis=1))
        self.ts.total_consumption.append([(1 + tau_vat) * self.ts.current("consumption").sum()])
        self.ts.total_consumption_before_vat.append([self.ts.current("consumption").sum()])
        self.ts.industry_consumption.append(consumption_by_good.sum(axis=0))

        # Investment
        self.ts.investment.append(self.ts.current("nominal_amount_spent_in_lcu") - consumption_by_good)
        if add_emissions:
            inv = self.ts.current("nominal_amount_spent_in_lcu") - consumption_by_good

            # Apply per-industry investment fraction multipliers when enabled.
            if (
                use_emission_multiplier
                and self.emission_fractions is not None
                and self.emission_fractions.investment is not None
            ):
                inv_fracs = self.emission_fractions.investment[0, emitting_indices]
                inv_slice = inv[:, emitting_indices] * inv_fracs
            else:
                inv_slice = inv[:, emitting_indices]

            investment_emissions = inv_slice @ readjusted_factors
            self.ts.investment_emissions.append(investment_emissions)

            inv_sum = inv.sum(axis=0)
            investment_emissions_by_good = np.zeros(inv.shape[1])
            for i in emitting_indices:
                idx = np.where(emitting_indices == i)[0]
                multiplier = (
                    self.emission_fractions.investment[0, i]
                    if use_emission_multiplier
                    and self.emission_fractions is not None
                    and self.emission_fractions.investment is not None
                    else 1.0
                )
                investment_emissions_by_good[i] = (inv_sum[i] * multiplier * readjusted_factors[idx]).item()
            self.ts.investment_emissions_by_good.append(investment_emissions_by_good)

            if emitting_indices_ch4 is not None and readjusted_factors_ch4 is not None:
                investment_emissions_ch4_by_good = np.zeros(inv.shape[1])
                for i in emitting_indices_ch4:
                    idx = np.where(emitting_indices_ch4 == i)[0]
                    investment_emissions_ch4_by_good[i] = (inv_sum[i] * readjusted_factors_ch4[idx]).item()
                self.ts.investment_emissions_ch4_by_good.append(investment_emissions_ch4_by_good)

            disaggregated_emissions = inv_slice * readjusted_factors
            self.ts.coal_investment_emissions.append(disaggregated_emissions[:, 0])
            self.ts.oil_investment_emissions.append(disaggregated_emissions[:, 1])
            self.ts.gas_investment_emissions.append(disaggregated_emissions[:, 2])
            self.ts.refined_products_investment_emissions.append(disaggregated_emissions[:, 3])
        self.ts.total_investment.append([(1 + tau_cf) * self.ts.current("investment").sum()])
        self.ts.total_investment_before_vat.append([self.ts.current("investment").sum()])
        self.ts.industry_investment.append(self.ts.current("investment").sum(axis=0))

    def update_wealth(self, housing_data: pd.DataFrame, tau_cf: float, period_index: int | None = None) -> float:
        """Update household wealth positions.

        Updates:
        - Real asset holdings
        - Financial assets
        - Property values
        - Net wealth position

        Args:
            housing_data (pd.DataFrame): Property market data
            tau_cf (float): Capital formation tax rate
        """
        # Update real wealth
        self.ts.wealth_main_residence.append(
            self.compute_wealth_of_the_main_residence(
                housing_data=housing_data,
            )
        )
        self.ts.total_wealth_main_residence.append([self.ts.current("wealth_main_residence").sum()])
        self.ts.wealth_other_properties.append(
            self.compute_wealth_of_other_properties(
                housing_data=housing_data,
            )
        )
        self.ts.total_wealth_other_properties.append([self.ts.current("wealth_other_properties").sum()])
        self.ts.wealth_other_real_assets.append(self.compute_wealth_of_other_real_assets())
        self.ts.total_wealth_other_real_assets.append([self.ts.current("wealth_other_real_assets").sum()])
        self.ts.wealth_real_assets.append(
            self.ts.current("wealth_main_residence")
            + self.ts.current("wealth_other_properties")
            + self.ts.current("wealth_other_real_assets")
        )

        # New financial wealth
        income_for_residual_saving = self.ts.current("income")
        if getattr(self.functions["wealth"], "exclude_financial_asset_income_from_saving", False):
            income_for_residual_saving = income_for_residual_saving - self.ts.current("income_financial_assets")

        new_wealth = np.maximum(
            0.0,
            (
                income_for_residual_saving
                - self.ts.current("rent")
                - self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            ),
        )
        (
            new_wealth_in_deposits,
            new_wealth_in_other_financial_assets,
        ) = self.functions["wealth"].distribute_new_wealth(
            new_wealth=new_wealth,
            model=self.states["wealth_distribution_model"],
            ts=self.ts,
        )

        # Used-up financial wealth
        used_up_wealth = -np.minimum(
            0.0,
            (
                income_for_residual_saving
                - self.ts.current("rent")
                - self.ts.current("nominal_amount_spent_in_lcu").sum(axis=1)
            ),
        )
        use_up_wealth_kwargs = {
            "used_up_wealth": used_up_wealth,
            "current_wealth_in_deposits": self.ts.current("wealth_deposits"),
            "current_wealth_in_other_financial_assets": self.ts.current("wealth_other_financial_assets"),
        }
        if getattr(self.functions["wealth"], "uses_periodic_illiquid_returns", False):
            use_up_wealth_kwargs["period_index"] = period_index
        (
            used_up_wealth_in_deposits,
            used_up_wealth_in_other_financial_assets,
        ) = self.functions["wealth"].use_up_wealth(**use_up_wealth_kwargs)

        # Update other financial assets
        self.ts.wealth_other_financial_assets.append(
            self.compute_wealth_of_other_financial_assets(
                new_wealth_in_other_financial_assets=new_wealth_in_other_financial_assets,
                used_up_wealth_in_other_financial_assets=used_up_wealth_in_other_financial_assets,
                period_index=period_index,
            )
        )
        self.ts.total_wealth_other_financial_assets.append([self.ts.current("wealth_other_financial_assets").sum()])

        # Update deposits
        self.ts.wealth_deposits.append(
            self.compute_wealth_in_deposits(
                new_wealth_in_deposits=new_wealth_in_deposits,
                used_up_wealth_in_deposits=used_up_wealth_in_deposits,
                tau_cf=tau_cf,
            )
        )
        self.ts.total_wealth_deposits.append([self.ts.current("wealth_deposits").sum()])

        # Stage 4 (portfolio choice): diagnostics-only shadow call site. Must run
        # here — after wealth_other_financial_assets and wealth_deposits are both
        # appended above (so post_return_ifa/post_surplus_lfa are this period's
        # final values) and before wealth_financial_assets/wealth below (so this
        # never reads a stale total). Appends new diagnostic series only; does not
        # touch wealth_deposits, wealth_other_financial_assets, wealth_financial_assets,
        # wealth, debt, or any bank-side state. See
        # knowledge-vault/wiki/architecture/consumption-stage-4-portfolio-choice.md
        # (Increment 3) for the full design and data-sourcing rationale.
        if getattr(self.functions["wealth"], "uses_portfolio_choice", False):
            self.compute_stage4_portfolio_diagnostics()

        # Compute total financial assets
        self.ts.wealth_financial_assets.append(
            self.ts.current("wealth_other_financial_assets") + self.ts.current("wealth_deposits")
        )

        # Compute total wealth
        self.ts.wealth.append(self.ts.current("wealth_real_assets") + self.ts.current("wealth_financial_assets"))
        return self.current_illiquid_financial_asset_return_rate()

    def compute_stage4_portfolio_diagnostics(self) -> None:
        """Compute and persist Stage 4 (portfolio choice) shadow diagnostics for this period.

        Diagnostics-only: appends new ``portfolio_*`` time series and does not
        mutate ``wealth_deposits``, ``wealth_other_financial_assets``,
        ``wealth_financial_assets``, ``wealth``, debt, or any bank-side state.
        Must be called from ``update_wealth()`` after ``wealth_other_financial_assets``
        and ``wealth_deposits`` are both appended for the current period (see the
        call site for the exact ordering rationale).

        See ``knowledge-vault/wiki/architecture/consumption-stage-4-portfolio-choice.md``
        (Increment 3) for the full design, including the investable-surplus
        ($\\tilde{s}_{it} = Y_{it} - C_{it} - (a^b_{it}+a^m_{it})$, $T_{it}=0$
        since ``income`` is already net of tax) data-sourcing decisions.
        """
        wealth_function = self.functions["wealth"]

        opening_tfa_scale = self.ts.prev("wealth_other_financial_assets") + self.ts.prev("wealth_deposits")
        post_return_ifa = self.ts.current("wealth_other_financial_assets")
        # post_surplus_lfa ("LFA at the entry point plus s-tilde") is the
        # household's actual, already-updated current liquid balance sheet —
        # self.ts.current("wealth_deposits") — not a parallel shadow quantity
        # built from new_wealth/rent/etc. That balance already reflects this
        # period's full deposit update (new savings, withdrawals, interest,
        # debt installments, new loans, tau_cf), so any positive-surplus
        # acquisition is already embedded in it.
        post_surplus_lfa = self.ts.current("wealth_deposits")
        # investable_surplus is computed independently per the Data Inputs
        # table's canonical definition (T_it=0 since income is already net of
        # tax) and passed through purely as a diagnostic — it does not feed
        # the post_surplus_lfa/target_tfa_base arithmetic below.
        investable_surplus = (
            self.ts.current("income") - self.ts.current("consumption") - self.ts.current("debt_installments")
        )

        # Stage 4 Increment 5: only built when the opt-in target_share_source=
        # "frm_magnitude" path is selected. The default "scalar" path (unchanged
        # from Increment 3) does not need these and must not pay for building
        # this dict on every period for every country that has not opted in.
        frm_covariates = None
        frm_magnitude_coefficients = None
        population_scale_factor = None
        net_wealth_scale_divisor = None
        if wealth_function.target_share_source == "frm_magnitude":
            tenure_status = self.states["Tenure Status of the Main Residence"]
            frm_covariates = {
                "age": self.states["household_head_age"],
                "household_members_in_employment": self.states["household_members_in_employment"],
                "investment_attitudes": self.states["Investment Attitudes"],
                "mortgagor": (tenure_status == 2).astype(float),
                "owner": (tenure_status == 1).astype(float),
                "net_wealth": self.compute_net_wealth(),
            }
            frm_coefficients = self.states["frm_coefficients"]
            frm_magnitude_coefficients = frm_coefficients.magnitude_coefficients
            population_scale_factor = self.states["population_scale_factor"]
            net_wealth_scale_divisor = frm_coefficients.net_wealth_scale_divisor

        diagnostics = compute_stage4_household_diagnostics(
            opening_tfa_scale=opening_tfa_scale,
            post_surplus_lfa=post_surplus_lfa,
            post_return_ifa=post_return_ifa,
            investable_surplus=investable_surplus,
            frm_covariates=frm_covariates,
            frm_magnitude_coefficients=frm_magnitude_coefficients,
            population_scale_factor=population_scale_factor,
            net_wealth_scale_divisor=net_wealth_scale_divisor,
            portfolio_participates=self.states["portfolio_participates"],
            target_share_source=wealth_function.target_share_source,
            default_target_illiquid_share=wealth_function.default_target_illiquid_share,
            phi_1=wealth_function.phi_1,
            lambda_kappa=wealth_function.lambda_kappa,
            fixed_cost_share=wealth_function.fixed_cost_share,
        )
        rebalancing = diagnostics.rebalancing
        n_households = post_return_ifa.shape[0]

        self.ts.portfolio_actual_illiquid_share.append(rebalancing.actual_illiquid_share)
        self.ts.portfolio_opening_tfa_scale.append(diagnostics.portfolio_opening_tfa_scale)
        self.ts.portfolio_target_tfa_base.append(diagnostics.portfolio_target_tfa_base)
        self.ts.portfolio_post_return_lfa.append(diagnostics.portfolio_post_return_lfa)
        self.ts.portfolio_post_return_ifa.append(diagnostics.portfolio_post_return_ifa)
        # Not sourced in this increment: liquid_asset_policy_rate_markup is null
        # (uncalibrated) and the policy rate itself lives on a sibling agent
        # (central_bank) not reachable from Households.update_wealth(); record
        # NaN rather than thread a new cross-agent dependency for an
        # admittedly-uncalibrated value.
        self.ts.portfolio_liquid_return_rate.append(np.full(n_households, np.nan))
        self.ts.portfolio_illiquid_return_rate.append(
            np.full(n_households, self.current_illiquid_financial_asset_return_rate())
        )
        self.ts.portfolio_investable_surplus.append(diagnostics.portfolio_investable_surplus)
        # Not sourced in this increment: the scalar target_share_source path (the
        # only one active) has no participation-probability concept; that belongs
        # to the inert FRM covariate path (compute_frm_magnitude_target_share),
        # reserved for a later target_share_source switch.
        self.ts.portfolio_participation_probability.append(np.full(n_households, np.nan))
        self.ts.portfolio_participates.append(self.states["portfolio_participates"])
        self.ts.portfolio_target_illiquid_share.append(diagnostics.portfolio_target_illiquid_share)
        self.ts.portfolio_target_illiquid_assets.append(rebalancing.target_illiquid_assets)
        self.ts.portfolio_delta_tilde.append(rebalancing.delta_tilde)
        self.ts.portfolio_kappa_star_tilde.append(rebalancing.kappa_star_tilde)
        self.ts.portfolio_kappa_tilde.append(rebalancing.kappa_tilde)
        self.ts.portfolio_desired_illiquid_adjustment.append(rebalancing.desired_illiquid_adjustment)
        self.ts.portfolio_adjustment_cost.append(rebalancing.adjustment_cost)
        self.ts.portfolio_counterfactual_lfa_flow.append(rebalancing.counterfactual_lfa_flow)
        self.ts.portfolio_counterfactual_ifa_flow.append(rebalancing.counterfactual_ifa_flow)
        self.ts.portfolio_inaction_flag.append(rebalancing.inaction_flag)
        self.ts.portfolio_upper_bound_flag.append(rebalancing.upper_bound_flag)
        self.ts.portfolio_lower_bound_flag.append(rebalancing.lower_bound_flag)
        self.ts.portfolio_infeasible_interval_flag.append(rebalancing.infeasible_interval_flag)
        self.ts.portfolio_no_financial_assets_flag.append(rebalancing.no_financial_assets_flag)
        self.ts.portfolio_target_share_clipped_flag.append(diagnostics.portfolio_target_share_clipped_flag)
        # settles_portfolio_choice is not wired in this increment (diagnostics-only,
        # Stage 5 does not exist yet); always False.
        self.ts.portfolio_settlement_enabled.append(np.zeros(n_households, dtype=bool))

    def compute_wealth_of_the_main_residence(self, housing_data: pd.DataFrame) -> np.ndarray:
        """Calculate main residence wealth.

        Determines value of:
        - Owner-occupied housing
        - Primary residences

        Args:
            housing_data (pd.DataFrame): Property market data

        Returns:
            np.ndarray: Main residence value by household
        """
        wealth_of_the_main_residence = np.zeros(self.ts.current("n_households"))
        ind_owning_mhr = np.all(
            [
                np.isin(self.states["Tenure Status of the Main Residence"], [1, 2, 4]),
                self.states["Corresponding Inhabited House ID"] != -1,
            ],
            axis=0,
        )
        wealth_of_the_main_residence[ind_owning_mhr] = housing_data.loc[
            self.states["Corresponding Inhabited House ID"][ind_owning_mhr],
            "Value",
        ].values
        return wealth_of_the_main_residence

    def compute_wealth_of_other_properties(self, housing_data: pd.DataFrame) -> np.ndarray:
        """Calculate other property wealth.

        Determines value of:
        - Investment properties
        - Rental properties
        - Secondary homes

        Args:
            housing_data (pd.DataFrame): Property market data

        Returns:
            np.ndarray: Other property value by household
        """
        wealth_of_other_properties = np.zeros(self.ts.current("n_households"))
        housing_data_not_oo = housing_data.loc[housing_data["Is Owner-Occupied"] == 0]
        housing_data_not_oo_grouped = housing_data_not_oo.groupby("Corresponding Owner Household ID")["Value"].sum()
        wealth_of_other_properties[housing_data_not_oo_grouped.index.values] = housing_data_not_oo_grouped.values
        return wealth_of_other_properties

    def compute_wealth_of_other_real_assets(self) -> np.ndarray:
        """Calculate other real asset wealth.

        Determines value of:
        - Non-property real assets
        - Physical investments
        - Durable goods

        Returns:
            np.ndarray: Other real asset value by household
        """
        return self.functions["wealth"].compute_wealth_in_other_real_assets(
            current_wealth_in_other_real_assets=self.ts.current("wealth_other_real_assets"),
            current_investment_in_other_real_assets=self.ts.current("investment").sum(axis=1),
        )

    def compute_wealth_of_other_financial_assets(
        self,
        new_wealth_in_other_financial_assets: float,
        used_up_wealth_in_other_financial_assets: float,
        period_index: int | None = None,
    ) -> np.ndarray:
        """Calculate other financial asset wealth.

        Updates financial assets based on:
        - New investments
        - Asset usage
        - Market returns

        Args:
            new_wealth_in_other_financial_assets (float): New investments
            used_up_wealth_in_other_financial_assets (float): Used assets

        Returns:
            np.ndarray: Financial asset value by household
        """
        kwargs = {
            "current_wealth_in_other_financial_assets": self.ts.current("wealth_other_financial_assets"),
            "new_wealth_in_other_financial_assets": new_wealth_in_other_financial_assets,
            "used_up_wealth_in_other_financial_assets": used_up_wealth_in_other_financial_assets,
        }
        if getattr(self.functions["wealth"], "uses_periodic_illiquid_returns", False):
            kwargs["period_index"] = period_index
        return self.functions["wealth"].compute_wealth_in_other_financial_assets(**kwargs)

    def compute_wealth_in_deposits(
        self,
        new_wealth_in_deposits: np.ndarray,
        used_up_wealth_in_deposits: np.ndarray,
        tau_cf: float,
    ) -> np.ndarray:
        """Calculate deposit wealth.

        Updates deposits based on:
        - New savings
        - Withdrawals
        - Interest earned
        - Tax effects

        Args:
            new_wealth_in_deposits (np.ndarray): New deposits
            used_up_wealth_in_deposits (np.ndarray): Used deposits
            tau_cf (float): Capital formation tax rate

        Returns:
            np.ndarray: Deposit value by household
        """
        return self.functions["wealth"].compute_wealth_in_deposits(
            current_wealth_in_deposits=self.ts.current("wealth_deposits"),
            new_wealth_in_deposits=new_wealth_in_deposits,
            used_up_wealth_in_deposits=used_up_wealth_in_deposits,
            current_interest_paid=self.ts.current("interest_paid"),
            price_paid_for_property=self.ts.current("price_paid_for_property"),
            debt_installments=self.ts.current("debt_installments"),
            new_loans=self.ts.current("received_consumption_loans") + self.ts.current("received_mortgages"),
            new_real_wealth=self.ts.current("investment").sum(axis=1),
            tau_cf=tau_cf,
        )

    def compute_debt(self) -> np.ndarray:
        """Calculate total household debt.

        Aggregates debt from:
        - Consumption loans
        - Mortgages
        - Other credit

        Returns:
            np.ndarray: Total debt by household
        """
        self.ts.total_consumption_loan_debt.append([self.ts.current("consumption_loan_debt").sum()])
        self.ts.total_mortgage_debt.append([self.ts.current("mortgage_debt").sum()])
        return self.ts.current("consumption_loan_debt") + self.ts.current("mortgage_debt")

    def compute_net_wealth(self) -> np.ndarray:
        """Calculate household net wealth.

        Determines net position from:
        - Total assets
        - Total liabilities
        - Debt obligations

        Returns:
            np.ndarray: Net wealth by household
        """
        return self.ts.current("wealth") - self.ts.current("debt")

    def handle_insolvency(self, banks: Banks, credit_market: CreditMarket) -> Tuple[float, float, float]:
        """Handle household insolvency cases.

        Processes defaults through:
        - Debt restructuring
        - Asset liquidation
        - Bank interactions

        Args:
            banks (Banks): Banking sector agent
            credit_market (CreditMarket): Credit market interface

        Returns:
            Tuple[float, float, float]: Default outcomes
        """
        return self.functions["insolvency"].handle_insolvency(
            households=self,
            banks=banks,
            credit_market=credit_market,
        )

    def save_to_h5(self, group: h5py.Group):
        """Save household data to HDF5.

        Stores:
        - Time series data
        - State variables
        - Market positions

        Args:
            group (h5py.Group): HDF5 storage group
        """
        self.ts.write_to_h5("households", group)

    def save_consumption_weights(self, group: h5py.Group):
        """Save consumption weight data.

        Stores:
        - Income-based weights
        - Industry allocations
        - Consumption patterns

        Args:
            group (h5py.Group): HDF5 storage group
        """
        group.create_dataset("household_consumption_weights_by_income", data=self.consumption_weights.T)
        group["household_consumption_weights_by_income"].attrs["columns"] = list(range(self.n_industries))

    def total_consumption(self) -> np.ndarray:
        """Get total consumption time series.

        Returns:
            np.ndarray: Aggregate consumption over time
        """
        return self.ts.get_aggregate("total_consumption")

    def consumption_loan_debt(self) -> np.ndarray:
        """Get consumption loan debt time series.

        Returns:
            np.ndarray: Aggregate consumption debt over time
        """
        return self.ts.get_aggregate("consumption_loan_debt")

    def mortgage_debt(self) -> np.ndarray:
        """Get mortgage debt time series.

        Returns:
            np.ndarray: Aggregate mortgage debt over time
        """
        return self.ts.get_aggregate("mortgage_debt")

    def consumption_emissions(self) -> np.ndarray:
        """Get consumption emissions time series.

        Returns:
            np.ndarray: Aggregate consumption emissions over time
        """
        return self.ts.get_aggregate("consumption_emissions")

    def investment_emissions(self) -> np.ndarray:
        """Get investment emissions time series.

        Returns:
            np.ndarray: Aggregate investment emissions over time
        """
        return self.ts.get_aggregate("investment_emissions")

    def disaggregated_consumption_emissions(self, input_name: str) -> np.ndarray:
        """Get disaggregated consumption emissions.

        Args:
            input_name (str): Input category name

        Returns:
            np.ndarray: Category-specific consumption emissions
        """
        return self.ts.get_aggregate(f"{input_name}_consumption_emissions")

    def disaggregated_investment_emissions(self, input_name: str) -> np.ndarray:
        """Get disaggregated investment emissions.

        Args:
            input_name (str): Input category name

        Returns:
            np.ndarray: Category-specific investment emissions
        """
        return self.ts.get_aggregate(f"{input_name}_investment_emissions")
